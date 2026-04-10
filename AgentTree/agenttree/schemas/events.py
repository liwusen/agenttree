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


class EventMessagePurpose(str, Enum):
    INFO = "info"
    REQUEST = "request"
    ACK = "ack"
    STATUS_UPDATE = "status_update"
    ESCALATION = "escalation"
    RESPONSE = "response"


EVENT_PRIORITIES: dict[EventKind, EventPriority] = {
    EventKind.COMMAND: EventPriority.COMMAND,
    EventKind.MESSAGE: EventPriority.MESSAGE,
    EventKind.STRUCT: EventPriority.STRUCT,
    EventKind.EMERGENCY: EventPriority.EMERGENCY,
    EventKind.EVENT: EventPriority.EVENT,
}


def build_event_metadata(
    *,
    metadata: dict[str, Any] | None = None,
    require_reply: bool | None = None,
    message_purpose: EventMessagePurpose | str | None = None,
    dedupe_key: str | None = None,
) -> dict[str, Any]:
    payload = dict(metadata or {})
    if require_reply is not None:
        payload["require_reply"] = require_reply
    if message_purpose is not None:
        payload["message_purpose"] = message_purpose.value if isinstance(message_purpose, EventMessagePurpose) else str(message_purpose)
    if dedupe_key:
        payload["dedupe_key"] = dedupe_key
    return payload


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
    require_reply: bool | None = None
    message_purpose: EventMessagePurpose | None = None
    dedupe_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_event(self) -> EventEnvelope:
        payload = {"text": self.text}
        default_require_reply = self.kind == EventKind.COMMAND if self.require_reply is None else self.require_reply
        default_message_purpose = EventMessagePurpose.REQUEST if self.kind == EventKind.COMMAND else EventMessagePurpose.INFO
        return EventEnvelope(
            kind=self.kind,
            source_path=self.source_path,
            target_path=self.target_path,
            payload=payload,
            channel_id=self.channel_id,
            metadata=build_event_metadata(
                metadata=self.metadata,
                require_reply=default_require_reply,
                message_purpose=self.message_purpose or default_message_purpose,
                dedupe_key=self.dedupe_key,
            ),
        )