import logging
import threading

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
EMBEDDING_DIMENSION = 1024

_model = None
_model_loading = False
_model_loaded = False
_lock = threading.Lock()


def is_loaded() -> bool:
    return _model_loaded


def _get_model():
    global _model, _model_loading, _model_loaded
    if _model is not None:
        return _model
    if _model_loading:
        return None
    with _lock:
        if _model is not None:
            return _model
        _model_loading = True
        try:
            from FlagEmbedding import BGEM3FlagModel
            logger.info(f"Loading BGE-M3 model: {settings.BGE_M3_MODEL_NAME}")
            _model = BGEM3FlagModel(settings.BGE_M3_MODEL_NAME, use_fp16=False)
            _model_loaded = True
            logger.info("BGE-M3 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            _model_loading = False
            raise
    return _model


def encode_texts(
    texts: list[str], batch_size: int | None = None
) -> list[list[float]]:
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Todos los textos deben ser cadenas no vacías")
    batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
    model = _get_model()
    if model is None:
        raise RuntimeError("Embedding model not loaded yet")

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.info(f"Encoding batch {i // batch_size + 1}, size={len(batch)}")

        output = model.encode(
            batch,
            batch_size=batch_size,
            max_length=min(settings.CHUNK_MAX_TOKENS, 512),
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_embs = output.get("dense_vecs") if "dense_vecs" in output else output.get("dense_emb")
        if dense_embs is None:
            raise RuntimeError("BGE-M3 no devolvió embeddings densos")

        for emb in dense_embs:
            emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            if len(emb_list) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"BGE-M3 devolvió {len(emb_list)} dimensiones; se esperaban "
                    f"{EMBEDDING_DIMENSION}"
                )
            all_embeddings.append(emb_list)

    return all_embeddings


def encode_single(text: str) -> list[float]:
    results = encode_texts([text])
    return results[0]


def compute_similarity(emb1: list[float], emb2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(emb1, emb2, strict=True))
    norm_a = sum(value * value for value in emb1) ** 0.5
    norm_b = sum(value * value for value in emb2) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def preload():
    _get_model()
    logger.info("Embeddings model preloaded")
