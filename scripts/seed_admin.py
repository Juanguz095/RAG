#!/usr/bin/env python3
"""Seed admin user — RAG-037 (PLAN-006 Fase 1).

La contraseña inicial viene del entorno (ADMIN_PASSWORD); si no está
configurada el script falla con mensaje claro (nunca hardcodeada).
"""
import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())

from sqlalchemy import select

from src.api.deps import hash_password
from src.database import User, async_session, init_db

DEFAULT_MSG = (
    "Falta la variable de entorno ADMIN_PASSWORD "
    "(p.ej. ADMIN_PASSWORD='Tue-C1ave-Larga-2026' python scripts/seed_admin.py)"
)


async def main():
    password = os.environ.get("ADMIN_PASSWORD")
    if not password:
        raise SystemExit(DEFAULT_MSG)
    await init_db()
    async with async_session() as db:
        result = await db.execute(select(User).where(User.username == "admin"))
        if result.scalar_one_or_none():
            print("Admin user already exists")
            return
        user = User(
            username="admin",
            email="admin@rag.local",
            hashed_password=hash_password(password),
            role="admin",
        )
        db.add(user)
        await db.commit()
        print("Admin user created (admin / [ADMIN_PASSWORD del entorno])")


if __name__ == "__main__":
    asyncio.run(main())
