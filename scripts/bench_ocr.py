#!/usr/bin/env python3
"""Micro-benchmark OCR paths inside the worker container."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "/app")

# Simulate embeddings._configure_threads pollution first, then OCR fix
# Case A: OMP=4 (old bug)
# Case B: OMP=1 + PASSIVE (new)


def run_seq(pdf: bytes, pages: list[int], label: str) -> float:
    from src.services.ocr import _ocr_sequential

    t0 = time.time()
    out = _ocr_sequential(pdf, pages)
    dt = time.time() - t0
    print(f"{label}: seq {len(pages)} pages in {dt:.2f}s ({dt/len(pages):.2f}s/pg) words~{sum(len(w) for _, w in out)}", flush=True)
    return dt


def run_par(pdf: bytes, pages: list[int], label: str) -> float:
    from src.services.ocr import ocr_empty_pages

    t0 = time.time()
    out = ocr_empty_pages(pdf, pages)
    dt = time.time() - t0
    print(f"{label}: par {len(pages)} pages in {dt:.2f}s ({dt/max(len(pages),1):.2f}s/pg) n={len(out)}", flush=True)
    return dt


def main() -> None:
    from pathlib import Path
    from src.services.ocr import extract_text_pymupdf

    upload = Path("/app/uploads/1bd08ba9a653c70bd3334dc214812284dd631d32653671ad2fbf6c36cd151356.pdf")
    if not upload.exists():
        # fallback any large pdf
        cands = list(Path("/app/uploads").glob("*.pdf"))
        upload = max(cands, key=lambda p: p.stat().st_size)
    print("pdf", upload, flush=True)
    pdf = upload.read_bytes()
    native = extract_text_pymupdf(pdf)
    empty = [i for i, t in native.items() if len(t.strip()) < 20]
    print(f"pages={len(native)} empty={len(empty)}", flush=True)
    sample = empty[:6]

    # A: sequential with OMP=4 (how embeddings leaves env)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ.pop("OMP_WAIT_POLICY", None)
    run_seq(pdf, sample, "SEQ OMP=4")

    # B: sequential with OMP=1 PASSIVE
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    run_seq(pdf, sample, "SEQ OMP=1 PASSIVE")

    # C: parallel via ocr_empty_pages (sets OMP=1 internally)
    os.environ["OMP_NUM_THREADS"] = "4"  # simulate pollution; function should override
    os.environ.pop("OMP_WAIT_POLICY", None)
    run_par(pdf, empty[:12] if len(empty) >= 12 else empty, "PAR fix-path")


if __name__ == "__main__":
    main()
