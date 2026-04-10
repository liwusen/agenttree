from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agenttree.schemas.events import EventEnvelope
from agenttree.schemas.nodes import NodeKind


class RuntimeMessageType(str, Enum):
    HELLO = "hello"
    HEARTBEAT = "heartbeat"
    WAKE = "wake"
    REQUEST_EVENT = "request_event"
    EVENT = "event"
    ACK_EVENT = "ack_event"
    PUBLISH_EVENT = "publish_event"
    INVOKE_EXECUTOR = "invoke_executor"
    EXECUTOR_RESULT = "executor_result"
    INVOKE_PROMPT = "invoke_prompt"
    PROMPT_RESULT = "prompt_result"
    ERROR = "error"
    LOG = "log"


class RuntimeHello(BaseModel):
    message_type: RuntimeMessageType = RuntimeMessageType.HELLO
    path: str
    kind: NodeKind
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeMessage(BaseModel):
    message_type: RuntimeMessageType
    path: str | None = None
    event: EventEnvelope | None = None
    events: list[EventEnvelope] | None = None
    event_id: str | None = None
    rpc_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None