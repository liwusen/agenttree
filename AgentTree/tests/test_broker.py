from __future__ import annotations

import pytest

from agenttree.core.broker import EventBroker
from agenttree.schemas.events import EventEnvelope, EventKind


@pytest.mark.asyncio
async def test_broker_respects_priority_order() -> None:
    broker = EventBroker()
    target = "/reactor/core"
    await broker.publish(EventEnvelope(kind=EventKind.EVENT, source_path="/sensor", target_path=target, payload={}))
    await broker.publish(EventEnvelope(kind=EventKind.MESSAGE, source_path="/child", target_path=target, payload={}))
    await broker.publish(EventEnvelope(kind=EventKind.COMMAND, source_path="/root", target_path=target, payload={}))

    first = await broker.drain_next(target)
    second = await broker.drain_next(target)
    third = await broker.drain_next(target)

    assert first is not None and first.kind == EventKind.COMMAND
    assert second is not None and second.kind == EventKind.MESSAGE
    assert third is not None and third.kind == EventKind.EVENT


@pytest.mark.asyncio
async def test_broker_acknowledges_pending_event() -> None:
    broker = EventBroker()
    target = "/reactor/core"
    event = EventEnvelope(kind=EventKind.COMMAND, source_path="/root", target_path=target, payload={})
    await broker.publish(event)
    drained = await broker.drain_next(target)
    assert drained is not None
    pending_before = await broker.pending_ack_ids(target)
    assert event.event_id in pending_before
    assert await broker.acknowledge(target, event.event_id) is True
    pending_after = await broker.pending_ack_ids(target)
    assert event.event_id not in pending_after