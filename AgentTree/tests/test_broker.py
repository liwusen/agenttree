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


@pytest.mark.asyncio
async def test_broker_drain_next_batch_only_uses_highest_priority_queue() -> None:
    broker = EventBroker()
    target = "/reactor/core"
    command_events = [
        EventEnvelope(kind=EventKind.COMMAND, source_path=f"/root/{index}", target_path=target, payload={"index": index})
        for index in range(3)
    ]
    message_event = EventEnvelope(kind=EventKind.MESSAGE, source_path="/child", target_path=target, payload={"index": 99})
    for event in command_events:
        await broker.publish(event)
    await broker.publish(message_event)

    drained = await broker.drain_next_batch(target, 4)

    assert len(drained) == 3
    assert all(item.kind == EventKind.COMMAND for item in drained)
    pending = await broker.pending_ack_ids(target)
    assert pending == sorted(item.event_id for item in command_events)

    next_event = await broker.drain_next(target)
    assert next_event is not None and next_event.kind == EventKind.MESSAGE


@pytest.mark.asyncio
async def test_broker_drain_next_batch_respects_requested_size() -> None:
    broker = EventBroker()
    target = "/reactor/core"
    events = [
        EventEnvelope(kind=EventKind.EVENT, source_path=f"/sensor/{index}", target_path=target, payload={"index": index})
        for index in range(5)
    ]
    for event in events:
        await broker.publish(event)

    drained = await broker.drain_next_batch(target, 2)

    assert [item.event_id for item in drained] == [events[0].event_id, events[1].event_id]
    queue_sizes = await broker.queue_sizes(target)
    assert queue_sizes[EventKind.EVENT.value] == 3