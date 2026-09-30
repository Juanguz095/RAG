from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.permissions import require_permission
from src.database import User, get_db
from src.services.synonyms import expand_term, get_synonym_groups

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["synonyms"])


@router.get("/synonyms")
async def list_synonyms(
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("keywords:read")),
):
    groups = await get_synonym_groups(db)
    return {"groups": groups, "count": len(groups)}


@router.get("/synonyms/expand")
async def expand(
    q: str,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(require_permission("keywords:read")),
):
    groups = await get_synonym_groups(db)
    variants = expand_term(q, groups)
    return {"query": q, "variants": variants}
