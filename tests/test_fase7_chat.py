"""PLAN-007 F5 — chat con memoria/historial persistente (RAG-028/029)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

try:
    import psycopg2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["psycopg2"] = MagicMock()


def _chat_db_override(client, monkeypatch):
    """Fake DB con stores para Conversation/Message sobre el fake de fase2."""
    import test_fase2_rbac_auditoria as _f2

    users = []
    base = _f2._make_fake_db(users)
    convs, msgs = [], []

    class _ConvRes(_f2._Res):
        def __init__(self, rows):
            super().__init__(rows[0] if len(rows) == 1 else None)
            self._rows = rows

    class ChatDb:
        async def execute(self, stmt, *a, **k):
            sql = str(stmt).lower()
            try:
                params = stmt.compile().params or {}
            except Exception:
                params = {}
            if "conversations" in sql and "count(" not in sql:
                uid = params.get("user_id_1") or params.get("user_id")
                cid = params.get("id_1") or params.get("conv_id_1") or params.get("id")
                rows = [c for c in convs
                        if (uid is None or str(c.user_id) == str(uid))
                        and (cid is None or str(c.id) == str(cid))]
                if "desc" in sql:
                    rows = sorted(rows, key=lambda c: str(c.updated_at or ""), reverse=True)
                return _ConvRes(rows)
            if "messages" in sql and "count(" not in sql:
                mid = params.get("conversation_id_1") or params.get("conversation_id")
                rows = [m for m in msgs
                        if mid is None or str(m.conversation_id) == str(mid)]
                if "desc" in sql:
                    rows = list(reversed(rows))
                return _ConvRes(rows)
            return await base.execute(stmt, *a, **k)

        def add(self, obj):
            n = type(obj).__name__
            if n == "Conversation":
                convs.append(obj); return
            if n == "Message":
                msgs.append(obj); return
            return base.add(obj)

        async def commit(self):
            pass

        async def rollback(self):
            pass

        async def refresh(self, obj):
            import uuid as _u
            if getattr(obj, "id", None) is None:
                obj.id = _u.uuid4()
            return obj

        def __getattr__(self, name):
            return getattr(base, name)

    _f2._override_db(client, ChatDb())
    return users, convs, msgs


def test_crear_conversacion_y_listar(client, monkeypatch):
    import test_fase2_rbac_auditoria as _f2
    _chat_db_override(client, monkeypatch)

    _f2._register(client, username="chatu")
    h = _f2._auth(client, "chatu")

    r = client.post("/api/v1/chat/conversations", json={"title": "Conversacion A"}, headers=h)
    assert r.status_code == 201, r.text[:200]
    cid = r.json()["id"]

    r = client.get("/api/v1/chat/conversations", headers=h)
    assert r.status_code == 200
    lst = r.json()
    assert isinstance(lst, list) and any(c["id"] == cid for c in lst)


def test_mensaje_reutiliza_pipeline_y_guarda_historia(client, monkeypatch):
    import test_fase2_rbac_auditoria as _f2
    _chat_db_override(client, monkeypatch)
    _f2._register(client, username="chatv")
    h = _f2._auth(client, "chatv")
    cid = client.post("/api/v1/chat/conversations", json={"title": "Conv"}, headers=h).json()["id"]

    class _R:
        score = 0.9; rerank_score = 0.9; chunk_id = None; document_id = None
        content = "El internamiento es por 24h."
        page_numbers = [3]; chunk_index = 0; chunk_metadata = {}
        document_name = "fua.pdf"; matched_terms = []; vector_score = 0.9
        keyword_score = 0.5; rrf_score = 0.7; chunk = None
        relevance = 90

    import src.api.v1.chat as ch
    import src.services.reranker as rr

    async def fs(*a, **k):
        return [_R()]

    monkeypatch.setattr(ch, "hybrid_search", fs, raising=False)
    monkeypatch.setattr(rr, "rerank", lambda q, res, top_k=10: res[:top_k], raising=False)
    monkeypatch.setattr(ch, "generate_answer_timed",
                        lambda ctx, q, **k: {"text": "24 horas [1]", "prompt_eval_ms": 1,
                                             "generation_ms": 1, "tokens_generated": 6})
    r = client.post(f"/api/v1/chat/conversations/{cid}/messages",
                    json={"content": "¿Cuánto dura el internamiento?"}, headers=h)
    assert r.status_code == 201, r.text[:300]
    d = r.json()
    assert d["role"] == "assistant"
    assert "24 horas" in d["content"]
    assert d["sources"]

    r = client.get(f"/api/v1/chat/conversations/{cid}", headers=h)
    msgs = r.json().get("messages", [])
    assert len(msgs) >= 2, msgs
    assert msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"


def test_memoria_ultimos_n(client, monkeypatch):
    """CHAT_MEMORY_N mensajes previos se pasan al prompt (RAG-028)."""
    import test_fase2_rbac_auditoria as _f2
    _chat_db_override(client, monkeypatch)
    _f2._register(client, username="chatm")
    h = _f2._auth(client, "chatm")
    cid = client.post("/api/v1/chat/conversations", json={"title": "M"}, headers=h).json()["id"]

    import src.api.v1.chat as ch
    prompts_seen = []

    class _R:
        score = 0.9; rerank_score = 0.9; chunk_id = None; document_id = None
        content = "contenido de prueba"
        page_numbers = [0]; chunk_index = 0; chunk_metadata = {}
        document_name = "x.pdf"; matched_terms = []; vector_score = 0.9
        keyword_score = 0.5; rrf_score = 0.7; chunk = None; relevance = 90

    async def fs(*a, **k):
        return [_R()]

    monkeypatch.setattr(ch, "hybrid_search", fs, raising=False)
    # rerank mockeado: sin cross-encoder real; conserva el score de evidencia.
    monkeypatch.setattr(ch, "rerank", lambda q, res, top_k=10: res, raising=False)
    monkeypatch.setattr(ch, "generate_answer_timed",
                        lambda ctx, q, **k: prompts_seen.append(ctx) or
                        {"text": "ok [1]", "prompt_eval_ms": 1, "generation_ms": 1,
                         "tokens_generated": 2})
    for qtxt in ("q1", "q2", "q3"):
        rr_ = client.post(f"/api/v1/chat/conversations/{cid}/messages",
                          json={"content": qtxt}, headers=h)
        if rr_.status_code != 201:
            print("FAIL msg:", rr_.status_code, rr_.text[:200])
    assert len(prompts_seen) == 3
    assert "q1" in prompts_seen[-1] and "Historial" in prompts_seen[-1]
