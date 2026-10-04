"""PLAN-007 F5 — Chat con memoria/historial persistente (RAG-028/029).

Endpoints:
- POST /api/v1/chat/conversations      → crear conversación (201)
- GET  /api/v1/chat/conversations      → listar las del usuario
- GET  /api/v1/chat/conversations/{id} → conversación + messages[]
- POST /api/v1/chat/conversations/{id}/messages → pregunta (user) + respuesta
  RAG reutilizando el pipeline de /query (hybrid_search + rerank + LLM + grounding
  + anonimización de salida). Guarda ambos mensajes con `sources` por mensaje.

Memoria (RAG-028): los últimos settings.CHAT_MEMORY_N turnos se inyectan como
"Historial de la conversacion" en el build_context del siguiente prompt.
"""
from __future__ import annotations

import datetime
import json
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.permissions import require_permission
from src.database import ChunkKeyword, Conversation, Keyword, Message, User, get_db
from src.config import get_settings
from src.services.context import build_context
from src.services.retrieval import hybrid_search
from src.services.reranker import rerank
from src.services.llm import generate_answer_timed, generate_chat_reply

settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ConversationIn(BaseModel):
    title: str = "Nueva conversacion"


class MessageIn(BaseModel):
    content: str
    doc_id: str | None = None  # consulta atada al documento abierto (opcional)
    domain: str | None = None  # filtro por dominio clinico (categoria de keyword)


class FeedbackIn(BaseModel):
    value: str  # useful | not_useful | review


# ── Detección rápida de charla general (sin costo de LLM) ─────────────
import re as _re
import unicodedata as _unicodedata

_SMALLTALK_WORDS = {
    "hola", "holaa", "buenas", "buenos", "dias", "tardes", "noches", "hey",
    "gracias", "ok", "okay", "vale", "adios", "chau", "chao", "saludos",
    "quien", "eres", "que", "haces", "puedes", "hacer", "ayuda", "ayudame",
    "tal", "como", "estas", "va", "genial", "perfecto", "buenisimo",
}
_CLINICAL_HINT = _re.compile(
    r"(diagnostic|tratamiento|paciente|sintoma|medicament|dosis|dni|historia|"
    r"documento|pagina|presion|fiebre|examen|resultado|antecedent|enfermedad|"
    r"receta|hospital|seguro|cirug|analisis|informe|fua)", _re.IGNORECASE
)


def _strip_accents(s: str) -> str:
    return "".join(c for c in _unicodedata.normalize("NFD", s) if _unicodedata.category(c) != "Mn")


def _is_small_talk(text: str) -> bool:
    """True si el mensaje es charla general (saludo/cortesia), no una consulta."""
    t = _strip_accents((text or "").strip().lower())
    if not t:
        return False
    t = _re.sub(r"[^a-z0-9\s]", " ", t)
    words = [w for w in t.split() if w]
    if not words:
        return False
    if _CLINICAL_HINT.search(t):
        return False
    # Saludo/cortesia con pocas palabras (hola, buenos dias, que tal, gracias...)
    if len(words) <= 5 and all(w in _SMALLTALK_WORDS for w in words):
        return True
    # Frases cortas de identidad/capacidad ("quien eres", "que puedes hacer")
    if len(words) <= 6 and any(w in {"hola", "buenas", "hey", "gracias", "adios", "chau"} for w in words):
        return True
    return False


def _conv_public(c: Conversation) -> dict:
    return {"id": str(c.id), "title": c.title, "created_at": str(c.created_at),
            "updated_at": str(c.updated_at)}


@router.post("/conversations", status_code=201)
async def create_conversation(
    body: ConversationIn,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("chat:write")),
):
    c = Conversation(user_id=current_user.id, title=(body.title or "Nueva conversacion")[:120])
    db.add(c)
    await db.commit()
    await db.refresh(c)
    return _conv_public(c)


@router.get("/conversations")
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("chat:write")),
):
    res = await db.execute(
        select(Conversation).where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    return [_conv_public(c) for c in res.scalars().all()]


@router.delete("/conversations/{conv_id}", status_code=204)
async def delete_conversation(
    conv_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("chat:write")),
):
    res = await db.execute(
        select(Conversation).where(Conversation.id == conv_id,
                                   Conversation.user_id == current_user.id)
    )
    c = res.scalar_one_or_none()
    if not c:
        raise HTTPException(status_code=404, detail="Conversacion no encontrada")
    await db.execute(delete(Message).where(Message.conversation_id == conv_id))
    await db.delete(c)
    await db.commit()
    return Response(status_code=204)


@router.get("/conversations/{conv_id}")
async def get_conversation(
    conv_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("chat:write")),
):
    res = await db.execute(
        select(Conversation).where(Conversation.id == conv_id,
                                   Conversation.user_id == current_user.id)
    )
    c = res.scalar_one_or_none()
    if not c:
        raise HTTPException(status_code=404, detail="Conversacion no encontrada")
    res2 = await db.execute(
        select(Message).where(Message.conversation_id == c.id).order_by(Message.created_at)
    )
    msgs = [{"id": str(m.id), "role": m.role, "content": m.content,
             "sources": m.sources, "created_at": str(m.created_at),
             "feedback": m.feedback}
            for m in res2.scalars().all()]
    return {**_conv_public(c), "messages": msgs}


@router.post("/conversations/{conv_id}/messages", status_code=201)
async def post_message(
    conv_id: UUID,
    body: MessageIn,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("chat:write")),
):
    question = (body.content or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Mensaje vacío")

    res = await db.execute(
        select(Conversation).where(Conversation.id == conv_id,
                                   Conversation.user_id == current_user.id)
    )
    conv = res.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversacion no encontrada")

    # ── Memoria (RAG-028): últimos N turnos de esta conversación ──
    res2 = await db.execute(
        select(Message).where(Message.conversation_id == conv.id)
        .order_by(Message.created_at.desc()).limit(settings.CHAT_MEMORY_N)
    )
    _min_dt = datetime.datetime(1970, 1, 1)
    memory = sorted(res2.scalars().all(), key=lambda m: (m.created_at or _min_dt))

    # La conversación se titula con la primera pregunta (si aún no tiene turnos).
    if not memory:
        conv.title = question[:120]

    # ── Rama 1: charla general (saludo/cortesia) → respuesta natural, sin citas ──
    if settings.CHAT_CONVERSATIONAL and _is_small_talk(question):
        hist = "\n".join(f"({m.role}) {m.content[:140]}" for m in memory)
        answer_text = generate_chat_reply(hist, question)
        sources_dto: list[dict] = []
    else:
        # ── Rama 2: consulta → pipeline RAG (atada al documento si se indico) ──
        results = await hybrid_search(question, db=db, doc_filter=body.doc_id)
        results = rerank(question, results, top_k=settings.RERANK_CANDIDATES)

        # Filtro por dominio (categoria de keyword): solo chunks con una keyword
        # de ese dominio (mismo criterio que POST /search).
        if body.domain:
            ids = [r.chunk_id for r in results if getattr(r, "chunk_id", None)]
            allowed: set[str] = set()
            if ids:
                try:
                    rows = (
                        await db.execute(
                            select(ChunkKeyword.chunk_id)
                            .join(Keyword, Keyword.id == ChunkKeyword.keyword_id)
                            .where(ChunkKeyword.chunk_id.in_(ids), Keyword.category == body.domain)
                        )
                    ).scalars().all()
                    allowed = {str(x) for x in rows}
                except Exception:
                    allowed = set()
            results = [r for r in results if str(getattr(r, "chunk_id", "")) in allowed]

        def _evidence_score(r):
            rs = getattr(r, "rerank_score", None)
            if rs is not None:
                return float(rs)
            return float(getattr(r, "score", 0.0) or 0.0) * 10.0

        strong = [r for r in results if _evidence_score(r) >= settings.EVIDENCE_MIN_SCORE]
        if len(strong) < max(1, settings.EVIDENCE_MIN_SOURCES):
            # C(ii) + abstención: no hay evidencia → rechazo amable, sin inventar.
            answer_text = (
                "No encuentro evidencia sobre eso en tus documentos. "
                "Solo puedo responder sobre el contenido de los documentos cargados; "
                "prueba con una pregunta concreta sobre ellos."
            )
            sources_dto = []
        else:
            results = strong
            sources_dto = [
                {"document": r.document_name, "page": r.page_numbers,
                 "snippet": (r.content or "")[:220]}
                for r in results
            ]
            context = build_context(results)
            if memory:
                hist = "\n".join(f"({m.role}) {m.content[:140]}" for m in memory)
                context = f"[Historial de la conversacion]\n{hist}\n\n[Contexto]\n{context}"
            out = generate_answer_timed(context, question)
            answer_text = out.get("text", "")
            # C(ii): si el modelo declara que no está en el contexto, no adjuntar
            # fuentes irrelevantes (refuerza el rechazo amable).
            if _re.search(
                r"no (se encuentra|encuentro|dispongo|tengo informaci|hay informaci|puedo (proporcionar|responder))",
                answer_text, _re.IGNORECASE,
            ):
                sources_dto = []

    from src.services.anonymizer import anonymize_text
    answer_text, _ = anonymize_text(answer_text)

    try:
        mu = Message(conversation_id=conv.id, role="user", content=question, sources=[])
        ma = Message(conversation_id=conv.id, role="assistant", content=answer_text,
                     sources=sources_dto)
        db.add(mu); db.add(ma)
        conv.updated_at = mu.created_at
        await db.commit()
        await db.refresh(ma)
    except Exception:
        logger.exception("chat post_message fallo")
        raise
    return {"id": str(ma.id), "role": "assistant", "content": answer_text,
            "sources": sources_dto, "created_at": str(ma.created_at),
            "feedback": None}


@router.post("/messages/{message_id}/feedback")
async def set_message_feedback(
    message_id: UUID,
    body: FeedbackIn,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("chat:write")),
):
    """Feedback del usuario (útil/no útil) sobre una respuesta (spec §49).

    No modifica el conocimiento validado; solo se registra para los KPIs de
    servicio del BSC (Respuestas útiles / no útiles)."""
    value = (body.value or "").strip().lower()
    if value not in ("useful", "not_useful", "review"):
        raise HTTPException(status_code=400, detail="Valor de feedback invalido")
    res = await db.execute(select(Message).where(Message.id == message_id))
    msg = res.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")
    conv = (
        await db.execute(
            select(Conversation).where(
                Conversation.id == msg.conversation_id,
                Conversation.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversacion no encontrada")
    msg.feedback = value
    msg.feedback_at = datetime.datetime.utcnow()
    await db.commit()
    return {"id": str(msg.id), "feedback": value}
