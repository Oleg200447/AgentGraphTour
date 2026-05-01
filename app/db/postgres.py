import re

import asyncpg


def _to_asyncpg_dsn(database_url: str) -> str:
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return database_url


def validate_sql_identifier(name: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return name


async def create_postgres_pool(database_url: str) -> asyncpg.Pool:
    dsn = _to_asyncpg_dsn(database_url)
    return await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=10)
