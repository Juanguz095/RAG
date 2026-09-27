#!/usr/bin/env python3
"""OCR micro-benchmark: run one mode via argv[1] = seq|par."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "/app")

MODE = sys.argv[1] if len(sys.argv) > 1 else "seq"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6


def main() -> None:
    from pathlib import Path
    from src.services.ocr import extract_text_pymupdf, ocr_empty_pages_iter

    p = sorted(Path("/app/uploads").glob("*.pdf"), key=lambda x: x.stat().st_size)[-1]
    b = p.read_bytes()
    native = extract_text_pymupdf(b)
    empty = [i for i, t in native.items() if len(t.strip()) < 20]
    print(f"file={p.name} pages={len(native)} empty={len(empty)} mode={MODE} n={N}", flush=True)

    sample = empty[:N]
    if MODE == "seq":
        # Match production: OMP free + ACTIVE + render prefetch
        os.environ.pop("OMP_NUM_THREADS", None)
        os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
        os.environ.pop("MKL_NUM_THREADS", None)
        t0 = time.time()
        out = list(ocr_empty_pages_iter(b, sample))
        dt = time.time() - t0
        words = sum(len(item[2]) for item in out)
        print(f"RESULT SEQ n={len(sample)} dt={dt:.2f}s per={dt/max(len(sample),1):.2f}s words={words}", flush=True)
    else:
        # Parallel path removed — production is always sequential+prefetch
        os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
        t0 = time.time()
        out = list(ocr_empty_pages_iter(b, sample))
        dt = time.time() - t0
        print(f"RESULT PAR n={len(sample)} dt={dt:.2f}s per={dt/max(len(sample),1):.2f}s pages={len(out)}", flush=True)


if __name__ == "__main__":
    main()
