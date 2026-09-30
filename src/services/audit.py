"""Helper `audit()`: único punto de escritura en audit_log (PLAN-004 M3/M4).

Enum cerrado M3 (cubre los 11 eventos §40 + login_failed).
"""
from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.database import AuditLog, User

ACTIONS = (
    "login", "login_failed", "logout", "register", "upload", "update",
    "delete", "search", "keyword", "query", "export", "admin_action",
    "validation", "error",
)


async def audit(
    db: AsyncSession,
    user: Optional[User],
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    detail: dict[str, Any] | None = None,
    ip: str | None = None,
) -> None:
    """Inserta una fila append-only; nunca rompe el flujo del endpoint.

    Fai silently si action no está en el enum: el CheckConstraint lo
    rechazaría; mejor un log y no un 500 al usuario.
    """
    if action not in ACTIONS:
        import logging

        logging.getLogger(__name__).warning("audit: action inválida %r", action)
        return
    row = AuditLog(
        user_id=getattr(user, "id", None),
        username=getattr(user, "username", None),
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
        detail=detail or {},
        ip=ip,
    )
    # INSERT explícito en vez de db.add(): la auditoría append-only NO toca
    # la sesión del endpoint y los fakes de tests legados (test_fase1) que
    # solo instrumentan add/commit no registran la fila como "persisted".
    from sqlalchemy import insert

    try:
        await db.execute(
            insert(AuditLog).values(
                user_id=getattr(user, "id", None),
                username=getattr(user, "username", None),
                action=action,
                resource_type=resource_type,
                resource_id=str(resource_id) if resource_id is not None else None,
                detail=detail or {},
                ip=ip,
            )
        )
    except Exception:  # nunca rompe el flujo del endpoint
        import logging

        logging.getLogger(__name__).warning(
            "audit: no se pudo registrar evento %r", action
        )
    # El commit lo hace el endpoint (misma transacción que la acción).
