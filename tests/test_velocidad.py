"""Contratos de PLAN-003 (velocidad y eficiencia del RAG)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── P2/P3: recortes y tuning por config ────────────────────────────


def test_config_perf_defaults():
    """Los defaults de velocidad viven en config (P2/P3)."""
    from src.config import Settings

    s = Settings()
    assert s.LLM_MAX_TOKENS <= 512          # recorte de generación (P2)
    assert s.LLM_N_CTX <= 4096              # 2048 en default
    assert s.CONTEXT_MAX_CHARS <= 4000      # contexto compacto (P2)
    assert s.LLM_N_THREADS > 4              # CPU tuning (P3)
    assert s.LLM_N_BATCH >= 512             # batch up (P3)


def test_config_perfil_aplica():
    """PERFIL=rapido|calidad aplica sus defaults a las variables hijas."""
    from src.config import Settings

    s_fast = Settings(PERFIL="rapido")
    assert s_fast.PERFIL == "rapido"
    assert s_fast.CONTEXT_MAX_CHARS <= 2500
    s_q = Settings(PERFIL="calidad")
    assert s_q.CONTEXT_MAX_CHARS > s_fast.CONTEXT_MAX_CHARS


# ── P5: paralelismo y re-ranking con menos candidatos ──────────────


def test_hybrid_search_uses_gather():
    """hybrid_search ejecuta vector y keyword bajo asyncio.gather (código fuente)."""
    import inspect

    from src.services import retrieval

    src = inspect.getsource(retrieval.hybrid_search)
    assert "asyncio.gather" in src


def test_rerank_candidates_cap():
    """RERANK_CANDIDATES limita cuántos candidatos va al cross-encoder (≤15)."""
    from src.config import Settings

    assert Settings().RERANK_CANDIDATES <= 15


# ── Paso 1: instrumentación del LLM ────────────────────────────────


@pytest.mark.integration
def test_generate_answer_reports_timings():
    """generate_answer debe poder exponer prompt_eval_ms/generation_ms."""
    from src.services.llm import generate_answer_timed

    out = generate_answer_timed("CONTEXTO: prueba. ", "Responde: ok?")
    assert "text" in out and "prompt_eval_ms" in out and "generation_ms" in out
    assert out["generation_ms"] >= 0


def test_bench_llm_script_exists():
    from pathlib import Path

    assert (Path(__file__).resolve().parent.parent / "scripts" / "bench_llm.py").exists()
