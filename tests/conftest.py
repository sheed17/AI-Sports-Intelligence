"""Shared pytest fixtures."""
import os
import pytest


@pytest.fixture(autouse=True)
def set_test_env():
    """Ensure tests run in test environment."""
    os.environ.setdefault("ENVIRONMENT", "test")
    os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://filmroom:filmroom@localhost:5432/filmroom_test")
    os.environ.setdefault("SYNC_DATABASE_URL", "postgresql://filmroom:filmroom@localhost:5432/filmroom_test")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/2")
