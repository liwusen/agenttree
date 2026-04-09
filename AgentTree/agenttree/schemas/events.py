from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EventKind(str, Enum):
    COMMAND = "command"
    MESSAGE = "message"
    STRUCT = "struct"
    EMERGENCY = "emergency"
    EVENT = "event"


class EventPriority(int, Enum):
    COMMAND = 1
    MESSAGE = 2
    STRUCT = 3
    EMERGENCY = 4
    EVENT = 5


EVENT_PRIORITIES: dict[EventKind, EventPriority] = {
    EventKind.COMMAND: EventPriority.COMMAND,
    EventKind.MESSAGE: EventPriority.MESSAGE,
    EventKind.STRUCT: EventPriority.STRUCT,
    EventKind.EMERGENCY: EventPriority.EMERGENCY,
    EventKind.EVENT: EventPriority.EVENT,
}


class EventEnvelope(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    kind: EventKind
    source_path: str
    target_path: str
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: str = Field(default_factory=lambda: uuid4().hex)
    channel_id: str | None = None
    requires_ack: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def priority(self) -> EventPriority:
        return EVENT_PRIORITIES[self.kind]


class EventDispatchState(BaseModel):
    event: EventEnvelope
    attempts: int = 0
    dispatched_at: datetime | None = None


class PublishEventRequest(BaseModel):
    event: EventEnvelope


class SendMessageRequest(BaseModel):
    kind: EventKind
    source_path: str
    target_path: str
    text: str
    channel_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_event(self) -> EventEnvelope:
        payload = {"text": self.text}
        return EventEnvelope(
            kind=self.kind,
            source_path=self.source_path,
            target_path=self.target_path,
            payload=payload,
            channel_id=self.channel_id,
            metadata=self.metadata,
        )