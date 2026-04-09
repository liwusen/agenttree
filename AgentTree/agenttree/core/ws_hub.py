from __future__ import annotations

import asyncio
import json

from fastapi import WebSocket

from agenttree.schemas.protocol import RuntimeMessage, RuntimeMessageType


class RuntimeSessionHub:
    def __init__(self) -> None:
        self._sessions: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, path: str, websocket: WebSocket) -> None:
        async with self._lock:
            self._sessions[path] = websocket

    async def disconnect(self, path: str) -> None:
        async with self._lock:
            self._sessions.pop(path, None)

    async def send_message(self, path: str, message: RuntimeMessage) -> None:
        async with self._lock:
            websocket = self._sessions.get(path)
        if websocket is None:
            raise KeyError(f"runtime session not found: {path}")
        await websocket.send_text(message.model_dump_json())

    async def wake(self, path: str) -> None:
        await self.send_message(
            path,
            RuntimeMessage(
                message_type=RuntimeMessageType.WAKE,
                path=path,
                payload={"reason": "event_available"},
            ),
        )

    async def list_paths(self) -> list[str]:
        async with self._lock:
            return sorted(self._sessions.keys())

    async def broadcast_log(self, text: str) -> None:
        paths = await self.list_paths()
        for path in paths:
            try:
                await self.send_message(
                    path,
                    RuntimeMessage(
                        message_type=RuntimeMessageType.LOG,
                        path=path,
                        payload={"text": text},
                    ),
                )
            except Exception:
                continue