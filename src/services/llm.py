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
        stop=["</s>", "<|user|>", "<|system|>"],
    )
    text = output["choices"][0]["text"].strip()
    return text


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
