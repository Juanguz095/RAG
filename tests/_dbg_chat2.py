import os, sys
sys.path.insert(0, "C:/Users/juang/Downloads/RAG")
os.environ["AUTH_REQUIRED"] = "true"
try:
    import psycopg2  # noqa
except ModuleNotFoundError:
    import unittest.mock as _um; sys.modules["psycopg2"] = _um.MagicMock()
import src.services.embeddings as _emb
_emb.preload = lambda: None
import pytest
from fastapi.testclient import TestClient
import src.main as m
import test_fase2_rbac_auditoria as _f2
import test_fase7_chat as t7
import src.api.v1.chat as ch

@pytest.fixture()
def client2():
    with TestClient(m.app, raise_server_exceptions=True) as c:
        yield c

def test_dbg(client2, monkeypatch):
    t7._chat_db_override(client2, monkeypatch)
    _f2._register(client2, username="chatm")
    h = _f2._auth(client2, "chatm")
    cid = client2.post("/api/v1/chat/conversations", json={"title": "M"}, headers=h).json()["id"]

    prompts_seen = []
    async def fs(*a, **k): return []
    import src.services.reranker as _rr
    monkeypatch.setattr(_rr, "rerank", lambda q, res, top_k=10: res[:top_k], raising=False)
    monkeypatch.setattr(ch, "hybrid_search", fs, raising=False)
    monkeypatch.setattr(ch, "generate_answer_timed",
                        lambda ctx, q, **k: prompts_seen.append(ctx) or
                        {"text": "ok [1]", "prompt_eval_ms": 1, "generation_ms": 1, "tokens_generated": 2})
    for qtxt in ("q1", "q2", "q3"):
        rr_ = client2.post(f"/api/v1/chat/conversations/{cid}/messages", json={"content": qtxt}, headers=h)
        print("MSG:", qtxt, rr_.status_code, rr_.text[:260])
