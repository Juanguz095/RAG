#!/usr/bin/env python
"""Benchmark de LLM (PLAN-003 paso 1): tokens/s + prompt-eval por modelo.

Ejecutar dentro del contenedor api (tiene llama-cpp-python y /models montado):
    docker exec rag-api-1 python /app/scripts/bench_llm.py
Opcional --model para otra ruta GGUF (ej. qwen): benchmarks a ciegas con las
5 preguntas fijas sobre segundo.pdf indexado (RAG-053/RAG-024).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, "/app")

PREGUNTAS = [
    "cual es el diagnostico principal del paciente",
    "que datos vitales tiene el paciente",
    "cual es el numero de historia clinica",
    "que institucion atiende al paciente",
    "hay alguna prescripcion de medicamentos",
]


def bench_model(model_path: str, context: str) -> dict:
    """Corre las 5 preguntas fijas y devuelve métricas agregadas."""
    from src.services.llm import _find_model, generate_answer_timed

    if model_path:
        from src.config import get_settings
        import src.services.llm as llm_mod

        llm_mod.settings = get_settings()
        llm_mod.settings.LLM_MODEL_PATH = model_path
        llm_mod.unload_llm()

    rows = []
    for q in PREGUNTAS:
        out = generate_answer_timed(context, q)
        rows.append({
            "q": q[:40],
            "prompt_eval_ms": out["prompt_eval_ms"],
            "generation_ms": out["generation_ms"],
            "tokens": out["tokens_generated"],
            "tok_s": round(out["tokens_generated"] / max(0.001, out["generation_ms"] / 1000), 1),
            "answer": out["text"][:120],
        })
    agg = {
        "model": model_path or "(config default)",
        "prompt_eval_ms_avg": round(sum(r["prompt_eval_ms"] for r in rows) / len(rows), 0),
        "gen_ms_avg": round(sum(r["generation_ms"] for r in rows) / len(rows), 0),
        "tok_s_avg": round(sum(r["tok_s"] for r in rows) / len(rows), 1),
        "answers": rows,
    }
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="ruta GGUF alternativa (sino config)")
    parser.add_argument("--context-chars", type=int, default=4000)
    args = parser.parse_args()

    # Contexto de muestra: primeros chars de la BD (o placeholder si no hay BD)
    context = ""
    try:
        import asyncio

        from src.database import async_session
        from src.services.retrieval import hybrid_search

        async def grab():
            async with async_session() as db:
                results = await hybrid_search("diagnostico principal del paciente", db, top_k=5)
                return "\n\n".join(r.content[:800] for r in results)

        context = asyncio.run(grab())[: args.context_chars]
    except Exception as e:
        print(f"(BD no disponible: {e}; usando contexto de muestra)", file=sys.stderr)
        context = "CONTEXTO MEDICO de muestra: El paciente presenta diagnostico de infeccion respiratoria..."[: args.context_chars]

    if not context:
        context = "CONTEXTO MEDICO de muestra (sin BD): diagnostico: infeccion respiratoria aguda."

    result = bench_model(args.model, context)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
