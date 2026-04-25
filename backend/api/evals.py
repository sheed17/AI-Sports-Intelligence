import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from backend.api.responses import APIEnvelope, ok

router = APIRouter()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


@router.get("/summary", response_model=APIEnvelope)
async def eval_summary():
    artifacts_dir = Path("ml/artifacts")
    classifier = _read_json(artifacts_dir / "metrics.json")
    retrieval = _read_json(artifacts_dir / "retrieval_eval.json")
    groundedness = _read_json(artifacts_dir / "agent_groundedness_eval.json")

    return ok(
        {
            "classifier": {
                "available": classifier is not None,
                "data_source": classifier.get("data_source") if classifier else None,
                "test_accuracy": (classifier.get("metrics") or {}).get("test_accuracy") if classifier else None,
                "test_macro_f1": (classifier.get("metrics") or {}).get("test_macro_f1") if classifier else None,
                "samples": classifier.get("samples") if classifier else None,
                "note": classifier.get("note") if classifier else None,
            },
            "retrieval": {
                "available": retrieval is not None,
                "overall_score": retrieval.get("overall_score") if retrieval else None,
            },
            "groundedness": {
                "available": groundedness is not None,
                "score": groundedness.get("groundedness_score") if groundedness else None,
            },
            "artifacts": {
                "best_model": (artifacts_dir / "best_model.pth").exists(),
                "confusion_matrix": (artifacts_dir / "confusion_matrix.png").exists(),
                "metrics_json": (artifacts_dir / "metrics.json").exists(),
                "retrieval_eval_json": (artifacts_dir / "retrieval_eval.json").exists(),
                "agent_groundedness_eval_json": (artifacts_dir / "agent_groundedness_eval.json").exists(),
            },
        }
    )
