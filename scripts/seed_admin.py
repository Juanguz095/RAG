#!/usr/bin/env python3
"""Seed admin user."""
import asyncio
import sys
import os
sys.path.insert(0, os.getcwd())

from src.database import async_session, init_db
from src.api.deps import hash_password
from src.database import User
from sqlalchemy import select


async def main():
    await init_db()
    async with async_session() as db:
        result = await db.execute(select(User).where(User.username == "admin"))
        if result.scalar_one_or_none():
            print("Admin user already exists")
            return
        user = User(
            username="admin",
            email="admin@rag.local",
            hashed_password=hash_password("admin123"),
            role="admin",
        )
        db.add(user)
        await db.commit()
        print("Admin user created (admin / admin123)")


if __name__ == "__main__":
    asyncio.run(main())
