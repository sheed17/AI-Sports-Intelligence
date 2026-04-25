"""API route contract checks for the documented MVP surface."""

from backend.main import app


def test_documented_routes_are_registered():
    routes = set()
    for route in app.routes:
        if not hasattr(route, "methods"):
            continue
        for method in route.methods - {"HEAD", "OPTIONS"}:
            routes.add((route.path, method))

    expected = {
        ("/videos/upload", "POST"),
        ("/videos", "GET"),
        ("/videos/{video_id}", "GET"),
        ("/videos/{video_id}/status", "GET"),
        ("/videos/{video_id}/process", "POST"),
        ("/videos/{video_id}/events", "GET"),
        ("/videos/{video_id}/events/{event_id}", "GET"),
        ("/videos/{video_id}/clips", "GET"),
        ("/clips/{clip_id}/stream", "GET"),
        ("/videos/{video_id}/ask", "POST"),
        ("/jobs/{job_id}", "GET"),
        ("/evals/summary", "GET"),
    }

    assert expected.issubset(routes)
