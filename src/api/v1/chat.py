"""PLAN-007 F5 — Chat con memoria/historial persistente (RAG-028/029).

Endpoints:
- POST /api/v1/chat/conversations      → crear conversación (201)
- GET  /api/v1/chat/conversations      → listar las del usuario
- GET  /api/v1/chat/conversations/{id} → conversación + messages[]
- POST /api/v1/chat/conversations/{id}/messages → pregunta (user) + respuesta
  RAG reutilizando el pipeline de /query (hybrid_search + rerank + LLM + grounding
  + anonimización de salida). Guarda ambos mensajes con `sources` por mensaje.

Memoria (RAG-028): los últimos settings.CHAT_MEMORY_N turnos se inyectan como
"Historial de la conversacion" en el build_context del siguiente prompt.
"""
from __future__ import annotations

import datetime
import json
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.permissions import require_permission
from src.database import Conversation, Message, User, get_db
from src.config import get_settings
from src.services.context import build_context
from src.services.retrieval import hybrid_search
from src.services.reranker import rerank
from src.services.llm import generate_answer_timed

settings = get_settings()
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ConversationIn(BaseModel):
    title: str = "Nueva conversacion"


class MessageIn(BaseModel):
    content: str


def _conv_public(c: Conversation) -> dict:
    return {"id": str(c.id), "title": c.title, "created_at": str(c.created_at),
            "updated_at": str(c.updated_at)}


@router.post("/conversations", status_code=201)
async def create_conversation(
    body: ConversationIn,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("chat:write")),
):
    c = Conversation(user_id=current_user.id, title=(body.title or "Nueva conversacion")[:120])
    db.add(c)
    await db.commit()
    await db.refresh(c)
    return _conv_public(c)


@router.get("/conversations")
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("chat:write")),
):
    res = await db.execute(
        select(Conversation).where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    return [_conv_public(c) for c in res.scalars().all()]


@router.get("/conversations/{conv_id}")
async def get_conversation(
    conv_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("chat:write")),
):
    res = await db.execute(
        select(Conversation).where(Conversation.id == conv_id,
                                   Conversation.user_id == current_user.id)
    )
    c = res.scalar_one_or_none()
    if not c:
        raise HTTPException(status_code=404, detail="Conversacion no encontrada")
    res2 = await db.execute(
        select(Message).where(Message.conversation_id == c.id).order_by(Message.created_at)
    )
    msgs = [{"id": str(m.id), "role": m.role, "content": m.content,
             "sources": m.sources, "created_at": str(m.created_at)}
            for m in res2.scalars().all()]
    return {**_conv_public(c), "messages": msgs}


@router.post("/conversations/{conv_id}/messages", status_code=201)
async def post_message(
    conv_id: UUID,
    body: MessageIn,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("chat:write")),
):
    question = (body.content or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Mensaje vacío")

    res = await db.execute(
        select(Conversation).where(Conversation.id == conv_id,
                                   Conversation.user_id == current_user.id)
    )
    conv = res.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversacion no encontrada")

    # ── Memoria (RAG-028): últimos N turnos de esta conversación ──
    res2 = await db.execute(
        select(Message).where(Message.conversation_id == conv.id)
        .order_by(Message.created_at.desc()).limit(settings.CHAT_MEMORY_N)
    )
    _min_dt = datetime.datetime(1970, 1, 1)
    memory = sorted(res2.scalars().all(), key=lambda m: (m.created_at or _min_dt))

    # ── Pipeline RAG (mismo que /query) ──
    results = await hybrid_search(question, db=db)
    results = rerank(question, results, top_k=settings.RERANK_CANDIDATES)
    sources_dto = [
        {"document": r.document_name, "page": r.page_numbers,
         "snippet": (r.content or "")[:220]}
        for r in results
    ]
    context = build_context(results)
    if memory:
        hist = "\n".join(f"({m.role}) {m.content[:140]}" for m in memory)
        context = f"[Historial de la conversacion]\n{hist}\n\n[Contexto]\n{context}"

    out = generate_answer_timed(context, question)
    answer_text = out.get("text", "")

    from src.services.anonymizer import anonymize_text
    answer_text, _ = anonymize_text(answer_text)

    try:
        mu = Message(conversation_id=conv.id, role="user", content=question, sources=[])
        ma = Message(conversation_id=conv.id, role="assistant", content=answer_text,
                     sources=sources_dto)
        db.add(mu); db.add(ma)
        conv.updated_at = mu.created_at
        await db.commit()
        await db.refresh(ma)
    except Exception:
        logger.exception("chat post_message fallo")
        raise
    return {"id": str(ma.id), "role": "assistant", "content": answer_text,
            "sources": sources_dto, "created_at": str(ma.created_at)}
