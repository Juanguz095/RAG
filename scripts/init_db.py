#!/usr/bin/env python3
"""Initialize database schema."""
import asyncio
import sys
import os
sys.path.insert(0, os.getcwd())

from src.database import init_db


async def main():
    await init_db()
    print("Database schema initialized")


if __name__ == "__main__":
    asyncio.run(main())
