"""Re-ranking con cross-encoder local (PLAN-001 Fase 3 — RAG-023/053).

Modelo: BAAI/bge-reranker-base (cross-encoder, CPU-friendly). El nombre es
config (RERANK_MODEL_NAME); modo degradación: si el modelo no carga, se
devuelve la lista tal cual (nunca romper la consulta).
"""
from __future__ import annotations

import gc
import logging

from src.config import get_settings

logger = logging.getLogger(__name__)

_model = None
_failed = False
settings = get_settings()


def _load_model():
    global _model, _failed
    if _model is None and not _failed:
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker: {settings.RERANK_MODEL_NAME}")
            _model = CrossEncoder(
                settings.RERANK_MODEL_NAME,
                device="cpu",
                trust_remote_code=True,
                max_length=512,
            )
            logger.info("Reranker loaded")
        except Exception as e:
            logger.warning(f"Reranker no disponible ({e}); continuando sin re-ranking")
            _failed = True
    return _model


def rerank(query: str, results: list, top_k: int | None = None):
    """Reordena resultados por relevancia query↔chunk con cross-encoder.

    `results` es una lista de objetos con atributo `content` (RetrievalResult
    u otro). Devuelve los mismos objetos reordenados y truncados a top_k.
    Si el modelo no está disponible, devuelve la lista original.
    """
    if not results:
        return results
    if len(results) <= 1:
        return results[:top_k] if top_k else results

    model = _load_model()
    if model is None:
        return results[:top_k] if top_k else results

    try:
        import math

        pairs = [(query, r.content or "") for r in results]
        scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
        # Guardar el score del cross-encoder (logit) normalizado a [0,1]
        # (sigmoid) — es la base del umbral de evidencia CP-006.
        for res_obj, s in zip(results, scores):
            try:
                setattr(res_obj, "rerank_score", round(1.0 / (1.0 + math.exp(-float(s))), 4))
            except Exception:
                setattr(res_obj, "rerank_score", 0.0)
        ranked = [r for _, r in sorted(zip(scores, results), key=lambda p: -float(p[0]))]
        return ranked[:top_k] if top_k else ranked
    except Exception as e:
        logger.warning(f"Re-ranking falló: {e}; devolviendo orden original")
        return results[:top_k] if top_k else results


def unload():
    global _model, _failed
    if _model is not None:
        del _model
        _model = None
        _failed = False
        gc.collect()
        logger.info("Reranker unloaded")
