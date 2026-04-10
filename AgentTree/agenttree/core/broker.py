from __future__ import annotations

import asyncio
from collections import defaultdict

from agenttree.schemas.events import EventDispatchState, EventEnvelope, EventKind
from agenttree.schemas.nodes import normalize_node_path


class EventBroker:
    def __init__(self) -> None:
        self._queues: dict[str, dict[EventKind, asyncio.Queue[EventDispatchState]]] = defaultdict(self._build_agent_queues)
        self._pending_acks: dict[str, dict[str, EventDispatchState]] = defaultdict(dict)
        self._agent_events: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self._lock = asyncio.Lock()

    @staticmethod
    def _build_agent_queues() -> dict[EventKind, asyncio.Queue[EventDispatchState]]:
        return {
            EventKind.COMMAND: asyncio.Queue(),
            EventKind.MESSAGE: asyncio.Queue(),
            EventKind.STRUCT: asyncio.Queue(),
            EventKind.EMERGENCY: asyncio.Queue(),
            EventKind.EVENT: asyncio.Queue(),
        }

    async def publish(self, event: EventEnvelope) -> None:
        target = normalize_node_path(event.target_path)
        state = EventDispatchState(event=event)
        async with self._lock:
            await self._queues[target][event.kind].put(state)
            self._agent_events[target].set()

    async def drain_next(self, agent_path: str) -> EventEnvelope | None:
        target = normalize_node_path(agent_path)
        async with self._lock:
            queues = self._queues[target]
            for kind in (EventKind.COMMAND, EventKind.MESSAGE, EventKind.STRUCT, EventKind.EMERGENCY, EventKind.EVENT):
                if queues[kind].empty():
                    continue
                state = await queues[kind].get()
                state.attempts += 1
                self._pending_acks[target][state.event.event_id] = state
                if all(queue.empty() for queue in queues.values()):
                    self._agent_events[target].clear()
                return state.event
            self._agent_events[target].clear()
            return None

    async def drain_next_batch(self, agent_path: str, batch_size: int) -> list[EventEnvelope]:
        target = normalize_node_path(agent_path)
        size = max(1, batch_size)
        async with self._lock:
            queues = self._queues[target]
            selected_kind: EventKind | None = None
            for kind in (EventKind.COMMAND, EventKind.MESSAGE, EventKind.STRUCT, EventKind.EMERGENCY, EventKind.EVENT):
                if not queues[kind].empty():
                    selected_kind = kind
                    break
            if selected_kind is None:
                self._agent_events[target].clear()
                return []

            drained: list[EventEnvelope] = []
            selected_queue = queues[selected_kind]
            while len(drained) < size and not selected_queue.empty():
                state = await selected_queue.get()
                state.attempts += 1
                self._pending_acks[target][state.event.event_id] = state
                drained.append(state.event)

            if all(queue.empty() for queue in queues.values()):
                self._agent_events[target].clear()
            return drained

    async def acknowledge(self, agent_path: str, event_id: str) -> bool:
        target = normalize_node_path(agent_path)
        async with self._lock:
            state = self._pending_acks[target].pop(event_id, None)
            return state is not None

    async def wait_for_event(self, agent_path: str) -> None:
        target = normalize_node_path(agent_path)
        await self._agent_events[target].wait()

    async def queue_sizes(self, agent_path: str) -> dict[str, int]:
        target = normalize_node_path(agent_path)
        async with self._lock:
            return {kind.value: self._queues[target][kind].qsize() for kind in EventKind}

    async def queue_snapshot(self, agent_path: str) -> dict[str, list[dict]]:
        target = normalize_node_path(agent_path)
        async with self._lock:
            snapshot: dict[str, list[dict]] = {}
            for kind in EventKind:
                queue = self._queues[target][kind]
                snapshot[kind.value] = [state.model_dump(mode="json") for state in list(queue._queue)]
            return snapshot

    async def queue_snapshot_for_all(self) -> dict[str, dict[str, list[dict]]]:
        async with self._lock:
            out: dict[str, dict[str, list[dict]]] = {}
            for target, queues in self._queues.items():
                out[target] = {
                    kind.value: [state.model_dump(mode="json") for state in list(queue._queue)]
                    for kind, queue in queues.items()
                }
            return out

    async def pending_ack_snapshot(self, agent_path: str) -> list[dict]:
        target = normalize_node_path(agent_path)
        async with self._lock:
            return [state.model_dump(mode="json") for state in self._pending_acks[target].values()]

    async def pending_ack_ids(self, agent_path: str) -> list[str]:
        target = normalize_node_path(agent_path)
        async with self._lock:
            return sorted(self._pending_acks[target].keys())