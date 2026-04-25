"""Integration tests for video upload and status endpoints.

Requires a running backend (DATABASE_URL + Redis). Run via docker-compose.test.yml
or with a local test DB: ENVIRONMENT=test pytest tests/integration/
"""
import io
import os
import tempfile

import pytest
import pytest_asyncio

# Skip all integration tests unless integration env is configured
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests",
)


@pytest.fixture(scope="module")
def client():
    from httpx import AsyncClient
    from backend.main import app
    return AsyncClient(app=app, base_url="http://test")


@pytest.mark.asyncio
async def test_health(client):
    async with client as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json()["data"]["status"] == "ok"


@pytest.mark.asyncio
async def test_upload_video(client, tmp_path):
    # Create a tiny fake video file
    fake_video = tmp_path / "test.mp4"
    fake_video.write_bytes(b"\x00" * 1024)

    async with client as c:
        r = await c.post(
            "/videos/upload",
            files={"file": ("test.mp4", fake_video.read_bytes(), "video/mp4")},
        )
    assert r.status_code == 201
    data = r.json()["data"]
    assert "id" in data
    assert data["status"] == "uploaded"
    assert data["filename"] == "test.mp4"
    return data["id"]


@pytest.mark.asyncio
async def test_get_video_not_found(client):
    async with client as c:
        r = await c.get("/videos/nonexistent-id")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_video_status(client, tmp_path):
    fake_video = tmp_path / "status_test.mp4"
    fake_video.write_bytes(b"\x00" * 512)

    async with client as c:
        upload_r = await c.post(
            "/videos/upload",
            files={"file": ("status_test.mp4", fake_video.read_bytes(), "video/mp4")},
        )
        video_id = upload_r.json()["data"]["id"]
        status_r = await c.get(f"/videos/{video_id}/status")

    assert status_r.status_code == 200
    status_data = status_r.json()["data"]
    assert status_data["video_id"] == video_id
    assert status_data["video_status"] in ("uploaded", "processing", "done")
