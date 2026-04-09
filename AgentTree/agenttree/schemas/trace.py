from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TraceEntry(BaseModel):
    trace_entry_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str
    category: str
    message: str
    trace_id: str | None = None
    event_id: str | None = None
    parent_event_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)