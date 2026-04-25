"""Run a lightweight demo check against the local FastAPI service.

Default checks avoid LLM calls. Pass --ask to include the LangGraph endpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request_json(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urlopen(req, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        payload = exc.read().decode("utf-8")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {payload}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--ask", action="store_true", help="Include the LLM-backed /ask check")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    health = request_json("GET", f"{base_url}/health")
    print(f"health: {health['data']['status']}")

    videos = request_json("GET", f"{base_url}/videos")
    video_rows = videos["data"]
    if not video_rows:
        print("no videos found; upload/process a clip first")
        return 1

    video_id = args.video_id or video_rows[0]["id"]
    print(f"video: {video_id} status={video_rows[0]['status']}")

    events = request_json("GET", f"{base_url}/videos/{video_id}/events?limit=5")
    print(f"events: {events['meta']['count']} returned")
    for event in events["data"][:3]:
        print(
            f"  {event['event_type']} @ {event['start_time_s']:.1f}s "
            f"conf={event['confidence']:.2f}"
        )

    clips = request_json("GET", f"{base_url}/videos/{video_id}/clips")
    print(f"clips: {clips['meta']['count']} generated")
    if clips["data"]:
        first_clip = clips["data"][0]
        print(f"  stream: {base_url}/clips/{first_clip['id']}/stream")

    if args.ask:
        answer = request_json(
            "POST",
            f"{base_url}/videos/{video_id}/ask",
            {"question": "When did the attack break down?"},
        )
        print(f"ask intent: {answer['data']['intent']}")
        print(f"ask evidence: {len(answer['data']['evidence'])} rows")
        print(answer["data"]["answer"][:500].replace("\n", " "))

    return 0


if __name__ == "__main__":
    sys.exit(main())
