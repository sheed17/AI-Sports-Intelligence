"""Evaluate semantic event retrieval quality using manually curated test cases.

For each test case: a query text + expected event_type.
Score = % of top-K results that match the expected event type.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import create_engine, text

from backend.config import get_settings


TEST_CASES = [
    {"query": "player lost the ball under pressure", "expected_type": "turnover"},
    {"query": "powerful shot toward goal", "expected_type": "shot"},
    {"query": "goalkeeper saves the shot", "expected_type": "save"},
    {"query": "short pass between players building up", "expected_type": "pass"},
    {"query": "player dribbling past defenders", "expected_type": "dribble"},
    {"query": "slow buildup play in midfield", "expected_type": "buildup"},
]


def evaluate_retrieval(video_id: str, top_k: int = 5, output_path: str | None = None):
    settings = get_settings()
    engine = create_engine(settings.sync_database_url)

    try:
        if settings.openai_api_key:
            from openai import OpenAI
            client = OpenAI()
            def embed(text_: str) -> list[float]:
                return client.embeddings.create(input=[text_], model=settings.embedding_model).data[0].embedding
        else:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            def embed(text_: str) -> list[float]:
                emb = _model.encode([text_])[0]
                if len(emb) < 1536:
                    emb = np.pad(emb, (0, 1536 - len(emb)))
                return emb.tolist()
    except Exception as exc:
        print(f"Cannot initialize embedding model: {exc}")
        return

    results = []
    total_hits = 0

    with engine.connect() as conn:
        for case in TEST_CASES:
            query_emb = embed(case["query"])
            vec_literal = "[" + ",".join(str(float(x)) for x in query_emb) + "]"
            rows = conn.execute(
                text(f"""
                    SELECT event_type, confidence,
                           1 - (embedding <=> '{vec_literal}'::vector) AS similarity
                    FROM event_predictions
                    WHERE video_id = :vid AND embedding IS NOT NULL
                    ORDER BY embedding <=> '{vec_literal}'::vector
                    LIMIT :k
                """),
                {"vid": video_id, "k": top_k},
            ).fetchall()

            hits = sum(1 for r in rows if r.event_type == case["expected_type"])
            precision_at_k = hits / len(rows) if rows else 0.0
            total_hits += hits

            result = {
                "query": case["query"],
                "expected_type": case["expected_type"],
                "top_k_results": [{"event_type": r.event_type, "similarity": r.similarity} for r in rows],
                "precision_at_k": precision_at_k,
                "hits": hits,
            }
            results.append(result)
            print(f"Query: '{case['query']}'")
            print(f"  Expected: {case['expected_type']}, Hits@{top_k}: {hits}/{len(rows)}, P@{top_k}={precision_at_k:.2f}")

    overall_score = total_hits / (len(TEST_CASES) * top_k)
    print(f"\nOverall retrieval score: {overall_score:.3f}")

    if output_path:
        Path(output_path).write_text(json.dumps({"results": results, "overall_score": overall_score}, indent=2))
        print(f"Results saved to: {output_path}")

    return overall_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    evaluate_retrieval(args.video_id, args.top_k, args.output)
