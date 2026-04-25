"""Evaluate agent answer groundedness: verify all cited timestamps exist in the DB.

Groundedness score = % of questions where every timestamp in the answer
is a real event in the database for that video.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sqlalchemy import create_engine, text

from backend.config import get_settings


TEST_QUESTIONS = [
    "When did the first turnover happen?",
    "Show me all shots in this video.",
    "Which player had the most possession?",
    "Summarize the key attacking moments.",
    "When did the attack break down?",
    "Find the best dribble in this video.",
]


def _extract_timestamps(text_: str) -> list[float]:
    """Extract all MM:SS timestamps from text."""
    timestamps = []
    for match in re.finditer(r"\b(\d{1,2}):(\d{2})\b", text_):
        mm, ss = int(match.group(1)), int(match.group(2))
        timestamps.append(mm * 60 + ss)
    return timestamps


def _timestamp_exists_in_db(conn, video_id: str, timestamp_s: float, window_s: float = 3.0) -> bool:
    result = conn.execute(
        text("""
            SELECT 1 FROM event_predictions
            WHERE video_id = :vid
              AND start_time_s BETWEEN :ts - :w AND :ts + :w
            LIMIT 1
        """),
        {"vid": video_id, "ts": timestamp_s, "w": window_s},
    ).fetchone()
    return result is not None


async def evaluate_groundedness(video_id: str, output_path: str | None = None):
    import asyncio
    from backend.agents.film_room_graph import run_film_room_agent

    settings = get_settings()
    engine = create_engine(settings.sync_database_url)

    results = []
    grounded_count = 0

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"\nQuestion {i+1}: {question}")
        try:
            result = await run_film_room_agent(
                video_id=video_id,
                question=question,
                run_id=f"eval-{i}",
            )
            answer = result.get("answer", "")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({"question": question, "error": str(exc), "grounded": False})
            continue

        timestamps = _extract_timestamps(answer)
        print(f"  Answer snippet: {answer[:200]}...")
        print(f"  Timestamps found: {timestamps}")

        grounded = True
        with engine.connect() as conn:
            for ts in timestamps:
                exists = _timestamp_exists_in_db(conn, video_id, ts)
                if not exists:
                    print(f"  UNGROUNDED: {ts}s not found in DB")
                    grounded = False

        if grounded:
            grounded_count += 1
            print("  GROUNDED ✓")

        results.append({
            "question": question,
            "answer_snippet": answer[:300],
            "timestamps_cited": timestamps,
            "grounded": grounded,
        })

    score = grounded_count / len(TEST_QUESTIONS) if TEST_QUESTIONS else 0.0
    print(f"\n=== Groundedness Score: {score:.2%} ({grounded_count}/{len(TEST_QUESTIONS)}) ===")

    if output_path:
        Path(output_path).write_text(json.dumps({"results": results, "groundedness_score": score}, indent=2))
        print(f"Results saved to: {output_path}")

    return score


if __name__ == "__main__":
    import asyncio
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    asyncio.run(evaluate_groundedness(args.video_id, args.output))
