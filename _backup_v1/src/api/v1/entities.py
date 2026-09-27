from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_user
from src.database import get_db

router = APIRouter()


@router.get("")
async def list_entities(
    entity_type: str | None = None,
    icd10_code: str | None = None,
    atc_code: str | None = None,
    chunk_id: UUID | None = None,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Listar entidades médicas con filtros."""
    from src.services.ner import list_medical_entities
    return await list_medical_entities(
        db, entity_type, icd10_code, atc_code, chunk_id, page, size
    )


@router.get("/stats")
async def entities_stats(
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Estadísticas de entidades por tipo."""
    from src.services.ner import get_entities_stats
    return await get_entities_stats(db)
