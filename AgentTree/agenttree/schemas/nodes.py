from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def normalize_node_path(path: str) -> str:
    cleaned = "/" + "/".join(part for part in path.strip().split("/") if part)
    return cleaned or "/"


def node_parent_path(path: str) -> str | None:
    normalized = normalize_node_path(path)
    if normalized == "/":
        return None
    parts = normalized.strip("/").split("/")
    if len(parts) == 1:
        return "/"
    return "/" + "/".join(parts[:-1])


class NodeKind(str, Enum):
    ROOT = "root"
    AGENT = "agent"
    EXECUTOR = "executor"
    SUPERVISOR = "supervisor"


class NodeStatus(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    STOPPED = "stopped"


class TriggerSpec(BaseModel):
    trigger_id: str
    trigger_type: str
    config: dict[str, Any] = Field(default_factory=dict)


class NodeRecord(BaseModel):
    path: str
    kind: NodeKind
    parent_path: str | None = None
    owner_path: str | None = None
    prompt: str | None = None
    description: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.CREATED
    metadata: dict[str, Any] = Field(default_factory=dict)
    triggers: list[TriggerSpec] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return normalize_node_path(value)

    @field_validator("parent_path")
    @classmethod
    def validate_parent_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_node_path(value)

    @field_validator("owner_path")
    @classmethod
    def validate_owner_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_node_path(value)


class CreateAgentRequest(BaseModel):
    parent_path: str
    name: str
    prompt: str | None = None
    prompt_template: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    knowledge_seed_templates: list[str] = Field(default_factory=list)
    knowledge_seed_manual_entries: list[str] = Field(default_factory=list)
    knowledge_seed_root: str | None = None

    @property
    def path(self) -> str:
        return normalize_node_path(f"{self.parent_path}/{self.name}")


class CreateExecutorRequest(BaseModel):
    owner_path: str
    name: str
    executor_kind: str
    config: dict[str, Any] = Field(default_factory=dict)

    @property
    def path(self) -> str:
        return normalize_node_path(f"{self.owner_path}/{self.name}")


class RegisterExecutorRequest(BaseModel):
    path: str
    executor_kind: str
    description: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path")
    @classmethod
    def validate_executor_path(cls, value: str) -> str:
        return normalize_node_path(value)


class BindExecutorRequest(BaseModel):
    executor_path: str
    owner_path: str

    @field_validator("executor_path")
    @classmethod
    def validate_executor_path(cls, value: str) -> str:
        return normalize_node_path(value)

    @field_validator("owner_path")
    @classmethod
    def validate_owner_path(cls, value: str) -> str:
        return normalize_node_path(value)


class TransferExecutorRequest(BaseModel):
    executor_path: str
    new_owner_path: str


class UpdateNodeRequest(BaseModel):
    prompt: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpsertTriggerRequest(BaseModel):
    path: str
    trigger: TriggerSpec


class RemoveTriggerRequest(BaseModel):
    path: str
    trigger_id: str


class MoveNodeRequest(BaseModel):
    path: str
    new_parent_path: str
    new_name: str

    @property
    def new_path(self) -> str:
        return normalize_node_path(f"{self.new_parent_path}/{self.new_name}")


class ChannelRecord(BaseModel):
    channel_id: str
    members: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)