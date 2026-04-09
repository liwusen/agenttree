from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from agenttree.config import AgentTreeSettings
from agenttree.core.app import create_app


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