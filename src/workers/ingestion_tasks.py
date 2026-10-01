from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.config import get_settings
from src.database import Chunk, ChunkKeyword, Document, Keyword
from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()

sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2")
sync_engine = create_engine(sync_url, echo=False, pool_size=2, max_overflow=3)


def enqueue_process_document(document_id: str, pdf_path: str) -> None:
    process_document_task.apply_async(
        args=(document_id, pdf_path), queue="ingest"
    )


def enqueue_extract_fields(document_id: str) -> None:
    extract_patient_fields_task.apply_async(args=(document_id,), queue="extract")


def _encode_batch(texts: list[str]):
    from src.services.embeddings import encode_texts

    return encode_texts(texts, batch_size=32)


class _StreamingEmbedder:
    """Same grouping as chunk_document; encode flushed groups while later pages OCR."""

    def __init__(self) -> None:
        from src.services.chunking import _count_tokens, chunk_text

        self._count = _count_tokens
        self._chunk_text = chunk_text
        self._current_text = ""
        self._current_pages: list[int] = []
        self._current_tokens = 0
        self._entries: list[tuple[int, object]] = []
        self._next_index = 0
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed")

    def add_page(self, page_num: int, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        # RAG-038 (PLAN-006 Fase 3): anonimizar datos sensibles ANTES de
        # trocear/emebedar (CP-008). El PDF original NO se modifica.
        from src.services.anonymizer import anonymize_text

        text, _n_sensitive = anonymize_text(text)
        if not text:
            return
        page_tokens = self._count(text)
        max_size = settings.CHUNK_SIZE * 2
        if self._current_tokens + page_tokens > max_size and self._current_text:
            self._flush()
        self._current_text += ("\n\n" if self._current_text else "") + text
        self._current_pages.append(page_num)
        self._current_tokens += page_tokens

    def _flush(self) -> None:
        if not self._current_text:
            return
        page_chunks = self._chunk_text(self._current_text, self._current_pages, self._next_index)
        if page_chunks:
            fut = self._pool.submit(_encode_batch, [c.content for c in page_chunks])
            for i, c in enumerate(page_chunks):
                self._entries.append((self._next_index + i, c, fut, i))
            self._next_index += len(page_chunks)
        self._current_text = ""
        self._current_pages = []
        self._current_tokens = 0

    def finish(self) -> list[tuple[Chunk, list[float]]]:
        self._flush()
        self._entries.sort(key=lambda e: e[0])
        results: list[tuple[Chunk, list[float]]] = []
        # Resolve futures in order (encoder runs serially in one worker thread)
        for _idx, chunk, fut, i in self._entries:
            emb = fut.result()[i]
            results.append((chunk, [float(x) for x in emb]))
        self._pool.shutdown(wait=True)
        for i, (chunk, _emb) in enumerate(results):
            chunk.chunk_index = i
        return results


@celery_app.task(bind=True, max_retries=2, soft_time_limit=1800)
def process_document_task(self, document_id: str, pdf_path: str):
    t0 = time.time()
    logger.info(f"Processing document {document_id}")

    from pathlib import Path
    from src.services.ocr import extract_text_pymupdf, ocr_empty_pages_iter

    with Session(sync_engine) as db:
        doc = db.get(Document, document_id)
        if not doc:
            logger.error(f"Document {document_id} not found")
            return

        doc.status = "processing"
        db.commit()

        try:
            pdf_bytes = Path(pdf_path).read_bytes()
            t0_ocr = time.time()
            native_pages = extract_text_pymupdf(pdf_bytes)
            empty_set = {i for i, t in native_pages.items() if len(t.strip()) < 20}
            empty_list = sorted(empty_set)
            n_pages = len(native_pages)
            doc.page_count = n_pages

            word_boxes: dict[int, list[dict]] = {}
            ocr_by_idx: dict[int, str] = {}

            # Interleave: OCR yields pages while we advance embedding in page order.
            # Pages before the next empty page are native and available immediately.
            embedder = _StreamingEmbedder()
            total_chars = 0
            next_page = 0
            ocr_iter = iter(ocr_empty_pages_iter(pdf_bytes, empty_list)) if empty_list else iter(())
            pending_ocr = None  # (idx, text, words) not yet consumed if out of order wait

            def advance_to(limit_exclusive: int) -> None:
                nonlocal next_page, total_chars
                while next_page < limit_exclusive and next_page < n_pages:
                    if next_page in empty_set:
                        if next_page in ocr_by_idx:
                            text = ocr_by_idx[next_page]
                        else:
                            break  # wait for OCR
                    else:
                        text = native_pages.get(next_page, "")
                    total_chars += len(text)
                    embedder.add_page(next_page, text)
                    next_page += 1

            # Seed: pages until first empty page
            first_empty = empty_list[0] if empty_list else n_pages
            advance_to(first_empty)

            got = 0
            expected = len(empty_list)
            while got < expected:
                item = next(ocr_iter, None)
                if item is None:
                    break
                idx, text, words = item
                ocr_by_idx[idx] = text
                word_boxes[idx] = words
                got += 1
                # OCR yields in ascending order of empty_list; consume up to and including idx
                advance_to(idx + 1)
                # also flush any native runs after idx already known
                advance_to(n_pages if got >= expected else _next_expected_after(idx, empty_list, got))

            # Final sweep: any remaining pages (natives after last empty, etc.)
            advance_to(n_pages)
            # Safety: if advance stalled on a gap (shouldn't), force-fill
            while next_page < n_pages:
                if next_page in empty_set and next_page not in ocr_by_idx:
                    # missed OCR page — should not happen
                    logger.error(f"Missing OCR for page {next_page}")
                    text = ""
                else:
                    text = ocr_by_idx.get(next_page) or native_pages.get(next_page, "")
                total_chars += len(text)
                embedder.add_page(next_page, text)
                next_page += 1

            ocr_elapsed = time.time() - t0_ocr
            del pdf_bytes

            if total_chars < 10:
                embedder.finish()
                doc.status = "error"
                db.commit()
                logger.error("No text extracted")
                return

            t0_emb = time.time()
            paired = embedder.finish()
            emb_elapsed = time.time() - t0_emb

            if not paired:
                doc.status = "error"
                db.commit()
                logger.error("No chunks created")
                return

            logger.info(
                f"Created {len(paired)} chunks "
                f"(ocr={ocr_elapsed:.1f}s emb_wait={emb_elapsed:.1f}s)"
            )

            # Etapa keywords (PLAN-006 Fase 4 — RAG-017/018): sección/fragmento
            # en metadata ANTES de crear los Chunk de BD (se propagan ya con ella).
            from src.services.keywords import enrich_section, extract_keyword_matches

            for chunk, _emb in paired:
                enrich_section(chunk)
            res_kw = db.execute(select(Keyword).where(Keyword.is_active == True))  # noqa: E712
            kw_rows = res_kw.scalars().all() if hasattr(res_kw, "scalars") else list(res_kw)
            catalog = {kw.term: [kw.term] for kw in kw_rows}

            db_chunks = [
                Chunk(
                    document_id=doc.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    page_numbers=chunk.page_numbers,
                    token_count=chunk.token_count,
                    chunk_metadata=chunk.chunk_metadata,
                    embedding=emb,
                )
                for chunk, emb in paired
            ]
            db.add_all(db_chunks)
            doc.total_chunks = len(db_chunks)
            # flush para obtener chunk.id antes de los FK:
            db.flush()
            for dbc in db_chunks:
                counts = extract_keyword_matches(dbc, catalog)
                for term, n in counts.items():
                    kw = next((k for k in kw_rows if k.term == term), None)
                    if kw is not None:
                        db.add(ChunkKeyword(chunk_id=dbc.id, keyword_id=kw.id, match_count=n))

            meta = doc.metadata_ or {}
            if word_boxes:
                meta = {**meta, "word_boxes": {str(k): v for k, v in word_boxes.items()}}
            meta.pop("extract_queued", None)
            # WP1: exponer los tiempos de proceso al usuario (RAG-036 "medir
            # tiempo de carga rápida").
            meta["timings"] = {
                "ocr_s": round(ocr_elapsed, 1),
                "embed_s": round(emb_elapsed, 1),
                "total_s": round(time.time() - t0, 1),
            }

            doc.metadata_ = meta  # new dict each time — JSONB dirty tracking
            doc.status = "completed"
            doc.processed_at = datetime.now(timezone.utc)
            db.commit()

            logger.info(
                f"Document {document_id} completed: {len(db_chunks)} chunks "
                f"in {time.time()-t0:.1f}s"
            )

        except Exception as e:
            doc.status = "error"
            db.commit()
            logger.error(f"Error processing {document_id}: {e}", exc_info=True)
            raise


def _next_expected_after(idx: int, empty_list: list[int], got: int) -> int:
    """Page index to advance to after consuming OCR page idx (start of next run)."""
    try:
        pos = empty_list.index(idx)
    except ValueError:
        return idx + 1
    if pos + 1 < len(empty_list):
        return empty_list[pos + 1]
    return 10**9  # all empties done → allow full advance on next call


@celery_app.task(bind=True, max_retries=1, soft_time_limit=180)
def extract_patient_fields_task(self, document_id: str):
    """Best-effort JSON extraction for the patient panel (separate from ingestion)."""
    t0 = time.time()
    with Session(sync_engine) as db:
        try:
            doc = db.get(Document, document_id)
            if not doc:
                return
            meta = dict(doc.metadata_ or {})
            if meta.get("extracted_data") is not None:
                return

            from src.services.llm import generate_json

            chunks = (
                db.query(Chunk)
                .filter(Chunk.document_id == doc.id)
                .order_by(Chunk.chunk_index)
                .all()
            )
            if not chunks:
                return

            all_text = "\n".join(c.content for c in chunks)[:3000]
            context = f"TEXTO OCR:\n{all_text}"
            question = (
                "Extrae del texto medico. Devuelve el objeto JSON con claves: "
                "nombre, edad, sexo, dni, historia_clinica, fecha_atencion, "
                "diagnostico_principal, diagnosticos_secundarios, "
                "signos_vitales (pa, fc, temp, sato2). "
                "Usa null si falta un valor."
            )
            extracted = generate_json(context, question, max_tokens=1024)
            cleaned = re.sub(r"```(?:json)?\s*", "", extracted, flags=re.IGNORECASE).strip()
            m = re.search(r"\{[\s\S]*\}", cleaned)
            if not m and cleaned.startswith("{"):
                # Grammar may truncate mid-object — close open brackets
                fixed = cleaned
                # strip trailing incomplete tokens
                fixed = re.sub(r",\s*[^,:{}\[\]]*$", "", fixed)
                stack = []
                in_str = False
                esc = False
                for ch in fixed:
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                        continue
                    if ch == '"':
                        in_str = True
                    elif ch in "{[":
                        stack.append("}" if ch == "{" else "]")
                    elif ch in "}]":
                        if stack and stack[-1] == ch:
                            stack.pop()
                if not in_str:
                    # close incomplete string
                    pass
                elif fixed.endswith("\\"):
                    fixed = fixed[:-1] + '"'
                else:
                    fixed += '"'
                while stack:
                    # drop trailing comma/key before closing
                    fixed = re.sub(r",\s*$", "", fixed)
                    fixed = re.sub(r',\s*"[^"]*"\s*:\s*$', "", fixed)
                    fixed += stack.pop()
                m = re.search(r"\{[\s\S]*\}", fixed)
                if m:
                    cleaned = m.group()
            if m:
                fresh = db.get(Document, document_id)
                if fresh is None:
                    return
                new_meta = dict(fresh.metadata_ or {})
                new_meta["extracted_data"] = json.loads(m.group())
                if "word_boxes" in meta and "word_boxes" not in new_meta:
                    new_meta["word_boxes"] = meta["word_boxes"]
                new_meta.pop("extract_queued", None)
                fresh.metadata_ = new_meta  # rebinding marks JSONB dirty
                db.commit()
                logger.info(
                    f"LLM extracted patient data for {document_id} "
                    f"in {time.time()-t0:.1f}s"
                )
            else:
                meta.pop("extract_queued", None)
                doc.metadata_ = meta  # rebind
                db.commit()
                logger.warning(
                    f"LLM no JSON for {document_id} in {time.time()-t0:.1f}s: "
                    f"{cleaned[:300]!r}"
                )
        except Exception as e:
            logger.warning(f"LLM extraction failed for {document_id}: {e}")
            try:
                fresh = db.get(Document, document_id)
                if fresh is not None:
                    meta = dict(fresh.metadata_ or {})
                    meta.pop("extract_queued", None)
                    fresh.metadata_ = meta
                    db.commit()
            except Exception:
                pass
