from __future__ import annotations

import asyncio

import websockets

from agenttree.config import AgentTreeSettings
from agenttree.schemas.protocol import RuntimeHello, RuntimeMessage, RuntimeMessageType


class RuntimeClient:
    def __init__(self, settings: AgentTreeSettings) -> None:
        self.settings = settings

    async def connect(self, hello: RuntimeHello):
        websocket = await websockets.connect(self.settings.runtime_ws_url)
        await websocket.send(
            RuntimeMessage(
                message_type=RuntimeMessageType.HELLO,
                path=hello.path,
                payload=hello.model_dump(mode="json"),
            ).model_dump_json()
        )
        return websocket

    async def receive(self, websocket) -> RuntimeMessage:
        raw = await websocket.recv()
        return RuntimeMessage.model_validate_json(raw)

    async def send(self, websocket, message: RuntimeMessage) -> None:
        await websocket.send(message.model_dump_json())

    async def heartbeat_loop(self, websocket, path: str) -> None:
        while True:
            await asyncio.sleep(self.settings.heartbeat_interval_seconds)
            await self.send(websocket, RuntimeMessage(message_type=RuntimeMessageType.HEARTBEAT, path=path))