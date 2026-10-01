import os, sys, traceback
sys.path.insert(0, "C:/Users/juang/Downloads/RAG"); sys.path.insert(0, "C:/Users/juang/Downloads/RAG/tests")
os.environ["AUTH_REQUIRED"]="true"
from unittest.mock import MagicMock
try: import psycopg2
except ModuleNotFoundError: sys.modules["psycopg2"]=MagicMock()
from fastapi.testclient import TestClient
import src.main as m
import src.services.embeddings as emb
emb.preload = lambda: None
import test_fase2_rbac_auditoria as _f2
import src.api.v1.chat as ch
import src.services.reranker as rr

class MP:
    def setattr(self, o, n, v, **k): setattr(o, n, v)

class _R:
    score = 0.9; rerank_score = 0.9; chunk_id=None; document_id=None
    content = "El internamiento es por 24h."
    page_numbers=[3]; chunk_index=0; chunk_metadata={}
    document_name="fua.pdf"; matched_terms=[]; vector_score=0.9
    keyword_score=0.5; rrf_score=0.7; chunk=None; relevance=90

async def fs(*a, **k): return [_R()]
ch.hybrid_search = fs
rr.rerank = lambda q, res, top_k=10: res[:top_k]
ch.generate_answer_timed = lambda ctx, q, **k: {"text":"24 horas [1]","prompt_eval_ms":1,"generation_ms":1,"tokens_generated":6}

users, convs, msgs = [], [], []
base = _f2._make_fake_db(users)

class ChatDb:
    async def execute(self, stmt, *a, **k):
        sql = str(stmt).lower()
        try: params = stmt.compile().params or {}
        except Exception: params = {}
        if "conversations" in sql and "count(" not in sql:
            uid = params.get("user_id_1") or params.get("user_id")
            cid = params.get("id_1") or params.get("conv_id_1") or params.get("id")
            rows = [c for c in convs
                    if (uid is None or str(c.user_id) == str(uid))
                    and (cid is None or str(c.id) == str(cid))]
            if "desc" in sql:
                rows = sorted(rows, key=lambda c: str(c.updated_at or ""), reverse=True)
            class R2:
                def __init__(self, r): self._rows = r
                def scalar_one_or_none(self): return rows[0] if len(rows)==1 else (rows[0] if rows else None)
                def scalars(self):
                    class S:
                        def all(self): return self._rows
                    return S()
            return R2(rows)
        if "messages" in sql and "count(" not in sql:
            mid = params.get("conversation_id_1") or params.get("conversation_id")
            rows = [mm for mm in msgs if mid is None or str(mm.conversation_id)==str(mid)]
            if "desc" in sql: rows = list(reversed(rows))
            from test_fase2_rbac_auditoria import _Res
            class R3(_Res):
                def __init__(self, r): super().__init__(r); self._rows=r
            return R3(rows)
        return await base.execute(stmt, *a, **k)
    def add(self, obj):
        n = type(obj).__name__
        if n=="Conversation": convs.append(obj); return
        if n=="Message": msgs.append(obj); return
        return base.add(obj)
    async def commit(self): pass
    async def rollback(self): pass
    async def refresh(self, obj):
        import uuid as u
        if getattr(obj, "id", None) is None: obj.id=u.uuid4()
        return obj
    def __getattr__(self, n): return getattr(base, n)

with TestClient(m.app, raise_server_exceptions=True) as c:
    _f2._override_db(c, ChatDb())
    _f2._register(c, username="dbg9")
    h = _f2._auth(c, "dbg9")
    r = c.post("/api/v1/chat/conversations", json={"title":"D"}, headers=h)
    print("CREATE:", r.status_code)
    cid = r.json()["id"]
    r2 = c.post(f"/api/v1/chat/conversations/{cid}/messages", json={"content":"q?"}, headers=h)
    print("MSG:", r2.status_code, r2.text[:400])
