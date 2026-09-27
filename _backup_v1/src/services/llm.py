import logging
import os
from collections.abc import AsyncGenerator
from pathlib import Path

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_llm_attempted = False


def _get_llm():
    global _llm, _llm_attempted
    if _llm_attempted:
        return _llm
    _llm_attempted = True

    if not settings.LLM_ENABLED:
        logger.info("LLM disabled via LLM_ENABLED=false")
        return None

    if not Path(settings.MEDALPACA_MODEL_PATH).is_file():
        logger.warning("LLM model not found: %s", settings.MEDALPACA_MODEL_PATH)
        return None

    try:
        from llama_cpp import Llama

        logger.info(f"Loading LLM from: {settings.MEDALPACA_MODEL_PATH}")
        _llm = Llama(
            model_path=settings.MEDALPACA_MODEL_PATH,
            n_ctx=settings.LLM_N_CTX,
            n_threads=settings.LLM_N_THREADS,
            n_gpu_layers=settings.LLM_N_GPU_LAYERS,
            n_batch=settings.LLM_N_BATCH,
            verbose=False,
        )
        logger.info("LLM loaded successfully")
    except Exception as e:
        logger.error("Failed to load LLM: %s", e)
        _llm = None
    return _llm


def build_medalpaca_messages(context: str, query: str) -> list[dict]:
    system_prompt = (
        "<<SYS>>\n"
        "Eres un asistente médico experto. Responde basándote EXCLUSIVAMENTE "
        "en el contexto proporcionado de documentos médicos. Si la información "
        "no está en el contexto, indica que no tienes suficiente información. "
        "Nunca inventes datos médicos.\n"
        "<</SYS>>"
    )

    user_message = (
        f"Contexto médico:\n{context}\n\n"
        f"Pregunta: {query}\n\n"
        "Responde de forma precisa y basada en el contexto:"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def generate_answer_sync(query: str, chunks: list[dict]) -> str:
    from src.services.context import build_context

    context = build_context(chunks, max_tokens=settings.CONTEXT_MAX_TOKENS)
    messages = build_medalpaca_messages(context, query)

    llm = _get_llm()
    output = llm.create_chat_completion(
        messages=messages,
        temperature=0.1,
        top_p=0.9,
        stop=["</s>", "<<SYS>>"],
        max_tokens=settings.LLM_MAX_TOKENS,
    )

    return output["choices"][0]["message"]["content"]


async def generate_answer(
    query: str, chunks: list[dict]
) -> AsyncGenerator[str, None]:
    from src.services.context import build_context

    if not chunks:
        yield "No encontré evidencia suficiente en los documentos para responder esa pregunta."
        return

    context = build_context(chunks, max_tokens=settings.CONTEXT_MAX_TOKENS)

    try:
        llm = _get_llm()
        messages = build_medalpaca_messages(context, query)
        output = llm.create_chat_completion(
            messages=messages,
            temperature=0.1,
            top_p=0.9,
            stop=["</s>", "<<SYS>>"],
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        answer = output["choices"][0]["message"]["content"]
        yield answer
        return
    except Exception as e:
        logger.warning("LLM unavailable, showing chunks: %s", e)

    yield "Basado en el documento, encontré esta información:\n\n"
    for index, chunk in enumerate(chunks[:5]):
        content = chunk.get("content", "").strip()
        if content:
            yield "--- Fragmento %d ---\n%s\n\n" % (index + 1, content)


async def generate_answer_stream(query: str, chunks: list[dict]) -> AsyncGenerator[str, None]:
    """Backwards-compatible public name used by the API."""
    async for token in generate_answer(query, chunks):
        yield token


async def extract_structured(query: str, schema: dict, context: str) -> dict:
    """Extract JSON through the local LLM while preserving an evidence boundary."""
    if not context or context.startswith("No se encontró"):
        return {"data": {}, "citations": [], "confidence": 0.0}
    # The local model is asked for JSON only; validation of arbitrary schemas is
    # intentionally left to the caller because schemas may contain custom types.
    llm = _get_llm()
    prompt = (
        f"Extrae únicamente un JSON válido compatible con este esquema: {schema}\n"
        f"Contexto: {context}\nPregunta: {query}"
    )
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=settings.LLM_MAX_TOKENS,
    )
    import json

    content = output["choices"][0]["message"]["content"]
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {"data": {}, "citations": [], "confidence": 0.0}
    return {"data": data, "citations": [], "confidence": 1.0}


def preload():
    _get_llm()
    logger.info("LLM model preloaded")
