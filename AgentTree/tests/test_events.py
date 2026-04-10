from __future__ import annotations

from agenttree.schemas.events import EventKind, EventMessagePurpose, SendMessageRequest


def test_send_message_request_sets_structured_metadata_defaults() -> None:
    request = SendMessageRequest(
        kind=EventKind.MESSAGE,
        source_path="/worker",
        target_path="/manager",
        text="status updated",
    )

    event = request.to_event()

    assert event.metadata["require_reply"] is False
    assert event.metadata["message_purpose"] == EventMessagePurpose.INFO.value


def test_send_message_request_preserves_explicit_metadata_fields() -> None:
    request = SendMessageRequest(
        kind=EventKind.COMMAND,
        source_path="/manager",
        target_path="/worker",
        text="run task",
        require_reply=False,
        message_purpose=EventMessagePurpose.ACK,
        dedupe_key="manual-key",
    )

    event = request.to_event()

    assert event.metadata["require_reply"] is False
    assert event.metadata["message_purpose"] == EventMessagePurpose.ACK.value
    assert event.metadata["dedupe_key"] == "manual-key"