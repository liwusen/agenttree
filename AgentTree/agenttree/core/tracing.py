from __future__ import annotations

import asyncio
from collections import deque

from agenttree.schemas.trace import TraceEntry


class TraceStore:
    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: deque[TraceEntry] = deque(maxlen=max_entries)
        self._subscribers: set[asyncio.Queue[TraceEntry]] = set()
        self._lock = asyncio.Lock()

    async def record(self, entry: TraceEntry) -> None:
        async with self._lock:
            self._entries.append(entry)
            stale: list[asyncio.Queue[TraceEntry]] = []
            for queue in self._subscribers:
                try:
                    queue.put_nowait(entry)
                except asyncio.QueueFull:
                    stale.append(queue)
            for queue in stale:
                self._subscribers.discard(queue)

    async def list_entries(self, limit: int = 100) -> list[TraceEntry]:
        async with self._lock:
            items = list(self._entries)
        if limit <= 0:
            return []
        return items[-limit:]

    async def subscribe(self) -> asyncio.Queue[TraceEntry]:
        queue: asyncio.Queue[TraceEntry] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[TraceEntry]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)