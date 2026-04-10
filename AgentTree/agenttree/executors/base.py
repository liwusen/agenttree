from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

import httpx

from agenttree.agent_runtime.client import RuntimeClient
from agenttree.config import AgentTreeSettings
from agenttree.schemas.events import EventEnvelope, EventKind, EventMessagePurpose, build_event_metadata
from agenttree.schemas.nodes import NodeKind
from agenttree.schemas.protocol import RuntimeHello, RuntimeMessage, RuntimeMessageType


def build_settings_for_core(core: str) -> AgentTreeSettings:
    raw = (core or "").strip()
    if not raw:
        raise ValueError("core is required")
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    host, sep, port_text = raw.rpartition(":")
    if not sep or not host or not port_text:
        raise ValueError(f"invalid core address: {core}")
    settings = AgentTreeSettings(host=host, port=int(port_text))
    settings.ensure_dirs()
    settings.load_persisted_runtime_config()
    return settings


def operation_field(
    name: str,
    *,
    field_type: str,
    description: str,
    required: bool,
    default: Any | None = None,
) -> dict[str, Any]:
    field = {
        "name": name,
        "type": field_type,
        "description": description,
        "required": required,
    }
    if default is not None:
        field["default"] = default
    return field


def operation_spec(
    command: str,
    *,
    summary: str,
    description: str,
    payload_schema: list[dict[str, Any]],
    returns: list[dict[str, Any]],
    aliases: list[str] | None = None,
    examples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "command": command,
        "summary": summary,
        "description": description,
        "aliases": list(aliases or []),
        "payload_schema": payload_schema,
        "returns": returns,
        "examples": list(examples or []),
    }


def default_executor_usage_guide() -> dict[str, Any]:
    return {
        "invoke_tool": "invoke_executor(executor_path, command, payload_json)",
        "guidelines": [
            "在调用执行器前，先读取节点详情或执行器指南，确认支持的 command、参数和返回字段。",
            "command 应优先使用 operations 中定义的 canonical command，不要依赖别名。",
            "payload_json 必须是 JSON 对象字符串，并且字段名称、类型和必填项要与 payload_schema 一致。",
            "如果执行器操作是异步型，返回值通常只表示任务已提交，最终结果要等待 event 或 response 事件。",
            "如果参数不完整、类型错误或路径越权，执行器会返回 handled=false 或错误摘要。",
        ],
    }


class ExternalExecutorBase(ABC):
    def __init__(
        self,
        *,
        settings: AgentTreeSettings,
        path: str,
        executor_kind: str,
        description: str = "",
        capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        operations: list[dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self.path = path
        self.executor_kind = executor_kind
        self.description = description
        self.capabilities = list(capabilities or [])
        self.operation_specs = list(operations or [])
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("executor_usage", default_executor_usage_guide())
        if self.operation_specs:
            self.metadata["operations"] = self.operation_specs
        self.client = RuntimeClient(settings)
        self.owner_path: str | None = None
        self.websocket = None
        self._send_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def register(self) -> None:
        async with httpx.AsyncClient(base_url=self.settings.base_url, timeout=30.0) as client:
            response = await client.post(
                f"{self.settings.api_prefix}/executors/register",
                json={
                    "path": self.path,
                    "executor_kind": self.executor_kind,
                    "description": self.description,
                    "capabilities": self.capabilities,
                    "metadata": self.metadata,
                },
            )
            response.raise_for_status()
            payload = response.json()
            node = payload.get("node", {})
            owner_path = node.get("owner_path")
            self.owner_path = str(owner_path) if owner_path else None

    async def run(self) -> None:
        await self.register()
        self._loop = asyncio.get_running_loop()
        hello = RuntimeHello(
            path=self.path,
            kind=NodeKind.EXECUTOR,
            capabilities=self.capabilities,
            metadata={"executor_kind": self.executor_kind, **self.metadata},
        )
        websocket = await self.client.connect(hello)
        self.websocket = websocket
        await self.log("runtime", f"{self.executor_kind} connected", self.describe_state())
        heartbeat_task = asyncio.create_task(self.client.heartbeat_loop(websocket, self.path))
        started_task = asyncio.create_task(self.on_started())
        try:
            while True:
                message = await self.client.receive(websocket)
                if message.message_type == RuntimeMessageType.LOG:
                    continue
                if message.message_type == RuntimeMessageType.WAKE:
                    await self.send_message(RuntimeMessage(message_type=RuntimeMessageType.REQUEST_EVENT, path=self.path))
                    continue
                if message.message_type == RuntimeMessageType.EVENT and message.event is not None:
                    await self._dispatch_event(message.event)
        finally:
            heartbeat_task.cancel()
            started_task.cancel()
            await self.on_shutdown()

    async def _dispatch_event(self, event: EventEnvelope) -> None:
        await self.log(
            "event_received",
            f"{self.executor_kind} received {event.kind.value} from {event.source_path}",
            {"event": event.model_dump(mode="json")},
        )
        if event.kind == EventKind.STRUCT and event.payload.get("action") == "executor_bound":
            owner_path = event.payload.get("owner_path")
            self.owner_path = str(owner_path) if owner_path else None
            await self.log("ownership", f"executor bound to {self.owner_path}", {"owner_path": self.owner_path})
            await self.send_ack(event.event_id)
            await self.on_owner_changed()
            return
        await self.handle_event(event)
        await self.send_ack(event.event_id)

    async def send_message(self, message: RuntimeMessage) -> None:
        if self.websocket is None:
            raise RuntimeError("executor websocket is not connected")
        async with self._send_lock:
            await self.client.send(self.websocket, message)

    async def send_ack(self, event_id: str) -> None:
        await self.send_message(RuntimeMessage(message_type=RuntimeMessageType.ACK_EVENT, path=self.path, event_id=event_id))

    async def publish_event(
        self,
        kind: EventKind,
        payload: dict[str, Any],
        *,
        target_path: str | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        outgoing_payload = {
            "kind": kind,
            "source_path": self.path,
            "target_path": target_path or self.owner_path or "/supervisor",
            "payload": payload,
            "metadata": metadata or {},
        }
        if trace_id is not None:
            outgoing_payload["trace_id"] = trace_id
        else:
            outgoing_payload["trace_id"] = uuid4().hex
        outgoing = EventEnvelope(**outgoing_payload)
        await self.send_message(
            RuntimeMessage(message_type=RuntimeMessageType.PUBLISH_EVENT, path=self.path, event=outgoing)
        )

    async def reply_to_event(self, event: EventEnvelope, payload: dict[str, Any], kind: EventKind = EventKind.EVENT) -> None:
        await self.publish_event(
            kind,
            payload,
            target_path=self.owner_path or "/supervisor",
            trace_id=event.trace_id,
            metadata=build_event_metadata(
                metadata={"reply_to": event.event_id},
                require_reply=False,
                message_purpose=EventMessagePurpose.RESPONSE,
                dedupe_key=f"executor-reply:{self.path}:{event.event_id}",
            ),
        )

    async def log(self, category: str, message: str, payload: dict[str, Any] | None = None) -> None:
        await self.send_message(
            RuntimeMessage(
                message_type=RuntimeMessageType.LOG,
                path=self.path,
                payload={"category": category, "message": message, **({"payload": payload} if payload else {})},
            )
        )

    def schedule_coroutine(self, coroutine: Any) -> None:
        if self._loop is None:
            raise RuntimeError("event loop is not ready")
        asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def describe_state(self) -> dict[str, Any]:
        return {
            "executor_kind": self.executor_kind,
            "path": self.path,
            "owner_path": self.owner_path,
            "capabilities": self.capabilities,
            "operations": self.operation_specs,
        }

    async def on_started(self) -> None:
        return None

    async def on_owner_changed(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    @abstractmethod
    async def handle_event(self, event: EventEnvelope) -> None:
        raise NotImplementedError