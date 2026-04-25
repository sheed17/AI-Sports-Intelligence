import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override URL from environment so Docker/CI can inject it
db_url = os.getenv("SYNC_DATABASE_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)

# Import all models so autogenerate detects them
from backend.db.database import Base  # noqa: E402
from backend.db import models  # noqa: E402, F401

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    # Use sync driver (psycopg2) for migrations
    cfg = config.get_section(config.config_ini_section, {})
    url = cfg.get("sqlalchemy.url", "")
    # Convert async URL to sync if needed
    sync_url = url.replace("postgresql+asyncpg://", "postgresql://")
    cfg["sqlalchemy.url"] = sync_url

    connectable = async_engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        # Force sync driver
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
