from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Optional

from src.config import get_settings

logger = logging.getLogger(__name__)

_llm: Optional["Llama"] = None

settings = get_settings()

SYSTEM_PROMPT = (
    "Eres un asistente medico experto para profesionales de salud. "
    "Respondes en espanol de forma precisa, concisa y basada en evidencia. "
    "Puedes usar terminologia medica. Si no tienes informacion suficiente, lo indicas."
)


def _find_model() -> str:
    candidates = [
        settings.LLM_MODEL_PATH,
        settings.QWEN_MODEL_PATH,
        os.path.join(settings.MODEL_DIR, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    model_dir = Path(settings.MODEL_DIR)
    for f in model_dir.glob("*.gguf"):
        return str(f)
    raise FileNotFoundError(f"No GGUF model found in {model_dir}")


def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        path = _find_model()
        logger.info(f"Loading LLM: {path}")
        _llm = Llama(
            model_path=path,
            n_ctx=settings.LLM_N_CTX,
            n_threads=settings.LLM_N_THREADS,
            n_batch=settings.LLM_N_BATCH,
            verbose=False,
        )
        logger.info("LLM loaded")
    return _llm


def _build_prompt(context: str, question: str) -> str:
    # TinyLlama-chat v1.0: <|system|>...</s><|user|>...</s><|assistant|>
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\nCONTEXTO MEDICO:\n{context}\n\nPREGUNTA: {question}</s>\n"
        f"<|assistant|>\n"
    )


def generate_answer(
    context: str,
    question: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    llm = _get_llm()
    prompt = _build_prompt(context, question)
    output = llm(
        prompt=prompt,
        max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
        temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
        top_p=0.9,
        stop=["</s>", "eot"][:2],
    )
    text = output["choices"][0]["text"].strip()
    return text


def generate_answer_timed(
    context: str,
    question: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> dict:
    """Como generate_answer pero separando prompt_eval_ms / generation_ms.

    llama-cpp-python con verbose=False no expone timings dict; medimos
    manualmente: prompt_eval_ms ≈ tiempo hasta el primer token (streaming
    interno), generation_ms ≈ resto, y tokens/s sobre los tokens generados.
    """
    import time as _time

    llm = _get_llm()
    prompt = _build_prompt(context, question)
    t0 = _time.time()
    first_token_at = None
    chunks: list[str] = []
    stream = llm(
        prompt=prompt,
        max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
        temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
        top_p=0.9,
        stop=["</s>", "user"][:2],
        stream=True,
        # PLAN-003 P4: llama-cpp 0.3.35 no expone cache_prompt en __call__;
        # la reutilización del KV del prefijo depende del motor (llama.cpp
        # recicla hashes de prompt internamente cuando tokens coinciden).
        # P4 documentado como limitación de la librería en docs/bench_velocidad.md.
    )
    for chunk in stream:
        delta = chunk["choices"][0].get("text", "")
        if delta:
            if first_token_at is None:
                first_token_at = _time.time()
            chunks.append(delta)
    total = (_time.time() - t0) * 1000
    if first_token_at is None:
        return {"text": "", "prompt_eval_ms": round(total, 1), "generation_ms": 0.0, "tokens_generated": 0}
    prompt_eval_ms = (first_token_at - t0) * 1000
    generation_ms = total - prompt_eval_ms
    return {
        "text": "".join(chunks).strip(),
        "prompt_eval_ms": round(prompt_eval_ms, 1),
        "generation_ms": round(generation_ms, 1),
        "tokens_generated": len(chunks),
    }


_EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "nombre": {"type": ["string", "null"]},
        "edad": {"type": ["integer", "string", "null"]},
        "sexo": {"type": ["string", "null"]},
        "dni": {"type": ["string", "null"]},
        "historia_clinica": {"type": ["string", "null"]},
        "fecha_atencion": {"type": ["string", "null"]},
        "diagnostico_principal": {"type": ["string", "null"]},
        "diagnosticos_secundarios": {
            "type": "array",
            "items": {"type": ["string", "null"]},
        },
        "signos_vitales": {
            "type": "object",
            "properties": {
                "pa": {"type": ["string", "null"]},
                "fc": {"type": ["string", "null"]},
                "temp": {"type": ["string", "null"]},
                "sato2": {"type": ["string", "null"]},
            },
        },
    },
    "required": [
        "nombre",
        "edad",
        "sexo",
        "dni",
        "historia_clinica",
        "fecha_atencion",
        "diagnostico_principal",
        "diagnosticos_secundarios",
        "signos_vitales",
    ],
}


def generate_json(context: str, question: str, max_tokens: int = 512) -> str:
    """Force a single JSON object via grammar (TinyLlama ignores 'SOLO JSON')."""
    import json as _json

    from llama_cpp import LlamaGrammar

    llm = _get_llm()
    prompt = _build_prompt(context, question)
    grammar = LlamaGrammar.from_json_schema(_json.dumps(_EXTRACTION_JSON_SCHEMA))
    output = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=0.9,
        stop=["</s>", "<|user|>", "<|system|>"],
        grammar=grammar,
    )
    return output["choices"][0]["text"].strip()


def generate_answer_stream(context: str, question: str) -> AsyncGenerator[str, None]:
    llm = _get_llm()
    prompt = _build_prompt(context, question)
    stream = llm(
        prompt=prompt,
        max_tokens=settings.LLM_MAX_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
        top_p=0.9,
        stop=["</s>", "<|user|>", "<|system|>"],
        stream=True,
    )
    for chunk in stream:
        delta = chunk["choices"][0].get("text", "")
        if delta:
            yield delta


def unload_llm():
    global _llm
    if _llm is not None:
        del _llm
        _llm = None
        gc.collect()
        logger.info("LLM unloaded")
