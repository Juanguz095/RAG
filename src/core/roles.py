"""Los 5 roles del spec §256-279 con su mapeo a nombres del negocio."""
from __future__ import annotations

ROLES = ("admin", "editor", "assistant", "viewer", "auditor")

# Mapeo a los perfiles del spec del profesor
ROLE_SPEC_NAMES = {
    "admin": "Administrador",
    "editor": "Especialista",
    "assistant": "Usuario autorizado",
    "viewer": "Usuario de consulta",
    "auditor": "Auditor",
}


def normalize_role(role: str | None) -> str:
    """Acepta sinónimos del spec y devuelve el rol canónico."""
    if not role:
        return "assistant"
    r = role.strip().lower()
    if r in ROLES:
        return r
    # variantes hispanas frecuentes
    return {
        "administrador": "admin",
        "especialista": "editor",
        "profesional": "editor",
        "usuario autorizado": "assistant",
        "asistente": "assistant",
        "usuario de consulta": "viewer",
        "consulta": "viewer",
        "auditoría": "auditor",
        "auditoria": "auditor",
    }.get(r, "assistant")
