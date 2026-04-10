from __future__ import annotations

import asyncio
import json

import httpx

from agenttree.agent_runtime.client import RuntimeClient
from agenttree.config import AgentTreeSettings
from agenttree.schemas.events import EventEnvelope, EventKind, EventMessagePurpose, build_event_metadata
from agenttree.schemas.nodes import NodeKind
from agenttree.schemas.protocol import RuntimeHello, RuntimeMessage, RuntimeMessageType


class ExecutorRuntime:
    def __init__(self, settings: AgentTreeSettings, path: str, executor_kind: str = "external.generic") -> None:
        self.settings = settings
        self.path = path
        self.executor_kind = executor_kind
        self.client = RuntimeClient(settings)
        self.owner_path: str | None = None

    async def register(self) -> None:
        async with httpx.AsyncClient(base_url=self.settings.base_url, timeout=30.0) as client:
            response = await client.post(
                f"{self.settings.api_prefix}/executors/register",
                json={
                    "path": self.path,
                    "executor_kind": self.executor_kind,
                    "capabilities": ["event_push"],
                    "metadata": {},
                },
            )
            response.raise_for_status()
            payload = response.json()
            node = payload.get("node", {})
            owner_path = node.get("owner_path")
            self.owner_path = str(owner_path) if owner_path else None

    async def run(self) -> None:
        await self.register()
        hello = RuntimeHello(path=self.path, kind=NodeKind.EXECUTOR, capabilities=["event_push"], metadata={})
        websocket = await self.client.connect(hello)
        await self.log(websocket, "runtime", "executor runtime connected")
        heartbeat_task = asyncio.create_task(self.client.heartbeat_loop(websocket, self.path))
        try:
            while True:
                message = await self.client.receive(websocket)
                if message.message_type == RuntimeMessageType.LOG:
                    continue
                if message.message_type == RuntimeMessageType.WAKE:
                    await self.client.send(
                        websocket,
                        RuntimeMessage(message_type=RuntimeMessageType.REQUEST_EVENT, path=self.path),
                    )
                    continue
                if message.message_type == RuntimeMessageType.EVENT and message.event is not None:
                    await self.handle_event(websocket, message.event)
        finally:
            heartbeat_task.cancel()

    async def handle_event(self, websocket, event: EventEnvelope) -> None:
        await self.log(
            websocket,
            "event_received",
            f"executor received {event.kind.value} from {event.source_path}",
            {"event": event.model_dump(mode="json")},
        )
        if event.kind == EventKind.STRUCT and event.payload.get("action") == "executor_bound":
            owner_path = event.payload.get("owner_path")
            self.owner_path = str(owner_path) if owner_path else None
            await self.log(websocket, "ownership", f"executor bound to {self.owner_path}", {"owner_path": self.owner_path})
            await self.client.send(
                websocket,
                RuntimeMessage(message_type=RuntimeMessageType.ACK_EVENT, path=self.path, event_id=event.event_id),
            )
            return

        owner = self.owner_path or "/supervisor"
        result_event = EventEnvelope(
            kind=EventKind.EVENT,
            source_path=self.path,
            target_path=owner,
            payload={
                "executor_path": self.path,
                "handled": True,
                "input": event.payload,
                "summary": f"executor processed payload {json.dumps(event.payload, ensure_ascii=False)}",
            },
            metadata=build_event_metadata(
                metadata={"reply_to": event.event_id},
                require_reply=False,
                message_purpose=EventMessagePurpose.RESPONSE,
                dedupe_key=f"executor-reply:{self.path}:{event.event_id}",
            ),
        )
        await self.client.send(
            websocket,
            RuntimeMessage(message_type=RuntimeMessageType.PUBLISH_EVENT, path=self.path, event=result_event),
        )
        await self.log(websocket, "reply_sent", f"executor sent result to {owner}", result_event.payload)
        await self.client.send(
            websocket,
            RuntimeMessage(message_type=RuntimeMessageType.ACK_EVENT, path=self.path, event_id=event.event_id),
        )

    async def log(self, websocket, category: str, message: str, payload: dict | None = None) -> None:
        await self.client.send(
            websocket,
            RuntimeMessage(
                message_type=RuntimeMessageType.LOG,
                path=self.path,
                payload={"category": category, "message": message, **({"payload": payload} if payload else {})},
            ),
        )