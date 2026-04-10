from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from agenttree.config import AgentTreeSettings
from agenttree.core.app import create_app
from agenttree.schemas.events import EventEnvelope, EventKind


def test_dashboard_snapshot_includes_knowledge_and_traces(tmp_path: Path, monkeypatch) -> None:
    settings = AgentTreeSettings(data_dir=tmp_path / ".agenttree-data", auto_start_supervisor=False)
    settings.ensure_dirs()

    monkeypatch.setattr("agenttree.core.app.get_settings", lambda: settings)

    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/knowledge/upsert",
        json={
            "owner_node_path": "/supervisor",
            "doc_path": "/supervisor/RUNBOOK.md",
            "text": "控制台知识条目",
        },
    )
    assert response.status_code == 200

    dashboard = client.get("/api/dashboard?trace_limit=20")
    assert dashboard.status_code == 200
    payload = dashboard.json()

    assert any(item["doc_path"] == "/supervisor/RUNBOOK.md" for item in payload["knowledge"]["documents"])
    assert any(item["category"] == "knowledge_op" for item in payload["traces"])


def test_startup_auto_injects_system_knowledge_templates(tmp_path: Path, monkeypatch) -> None:
    settings = AgentTreeSettings(
        data_dir=tmp_path / ".agenttree-data",
        auto_start_supervisor=False,
        auto_inject_system_knowledge_templates=True,
    )
    settings.ensure_dirs()

    monkeypatch.setattr("agenttree.core.app.get_settings", lambda: settings)

    app = create_app()

    with TestClient(app) as client:
        documents = client.get("/api/knowledge", params={"scope": "/"})
        assert documents.status_code == 200
        payload = documents.json()
        assert any(item["doc_path"] == "/rules/tree_operating_rules.md" for item in payload["documents"])
        assert any(item["doc_path"] == "/rules/queue_handling_guide.md" for item in payload["documents"])

        content = client.get("/api/knowledge/content", params={"doc_path": "/rules/tree_operating_rules.md"})
        assert content.status_code == 200
        assert "树结构操作规则" in content.json()["text"]


def test_runtime_ws_request_event_supports_batched_dispatch(tmp_path: Path, monkeypatch) -> None:
    settings = AgentTreeSettings(
        data_dir=tmp_path / ".agenttree-data",
        auto_start_supervisor=False,
        auto_inject_system_knowledge_templates=False,
    )
    settings.ensure_dirs()

    monkeypatch.setattr("agenttree.core.app.get_settings", lambda: settings)

    app = create_app()

    with TestClient(app) as client:
        state = app.state.core
        for index in range(3):
            event = EventEnvelope(
                kind=EventKind.MESSAGE,
                source_path=f"/child/{index}",
                target_path="/worker",
                payload={"index": index},
            )
            client.portal.call(state.broker.publish, event)

        with client.websocket_connect(settings.ws_path) as websocket:
            websocket.send_json(
                {
                    "message_type": "hello",
                    "path": "/worker",
                    "payload": {
                        "message_type": "hello",
                        "path": "/worker",
                        "kind": "agent",
                        "capabilities": ["agent_runtime"],
                        "metadata": {},
                    },
                }
            )
            websocket.receive_json()
            websocket.send_json(
                {
                    "message_type": "request_event",
                    "path": "/worker",
                    "payload": {"batch_size": 3},
                }
            )
            response = websocket.receive_json()

    assert response["message_type"] == "event"
    assert response["event"] is None
    assert len(response["events"]) == 3
    assert response["payload"]["batch_size"] == 3
    assert response["payload"]["queue_kind"] == "message"
