from typing import Any

from pydantic import BaseModel, Field


class APIEnvelope(BaseModel):
    data: Any | None = None
    error: dict[str, Any] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


def ok(data: Any = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"data": data, "error": None, "meta": meta or {}}
