from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agenttree.agent_runtime.tools.knowledge_ops import build_knowledge_tools
from agenttree.config import AgentTreeSettings
from agenttree.const_prompt import get_knowledge_template, list_knowledge_templates
from agenttree.knowledge.store import KnowledgeStore
from agenttree.knowledge.sync import KnowledgeSyncService


@pytest.mark.asyncio
async def test_knowledge_delete_scope_removes_documents(tmp_path: Path) -> None:
    settings = AgentTreeSettings(data_dir=tmp_path / ".agenttree-data")
    settings.ensure_dirs()
    store = KnowledgeStore(settings)
    sync = KnowledgeSyncService(store)

    await store.upsert_document(
        owner_node_path="/root_manager/turbine_operator",
        doc_path="/root_manager/turbine_operator/INTRO.md",
        text="operator intro",
    )
    await store.upsert_document(
        owner_node_path="/root_manager/other_agent",
        doc_path="/root_manager/other_agent/INTRO.md",
        text="other intro",
    )

    removed = sync.handle_node_delete("/root_manager/turbine_operator")

    assert removed == ["/root_manager/turbine_operator/INTRO.md"]
    remaining = store.list_documents("/")
    assert len(remaining) == 1
    assert remaining[0]["doc_path"] == "/root_manager/other_agent/INTRO.md"
    assert remaining[0]["owner_node_path"] == "/root_manager/other_agent"


@pytest.mark.asyncio
async def test_read_knowledge_returns_not_found_payload_on_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AgentTreeSettings(data_dir=tmp_path / ".agenttree-data")
    settings.ensure_dirs()

    missing_doc_path = "/root_manager/turbine_operator/temperature_status"

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, params: dict[str, str]) -> httpx.Response:
            request = httpx.Request("GET", f"http://testserver{url}", params=params)
            return httpx.Response(status_code=404, request=request, json={"detail": missing_doc_path})

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    tools = build_knowledge_tools(settings, "/root_manager/turbine_operator")
    read_knowledge = tools[1]
    payload = await read_knowledge(missing_doc_path)

    assert json.loads(payload) == {
        "doc_path": missing_doc_path,
        "text": "",
        "found": False,
        "error": "knowledge_document_not_found",
    }


@pytest.mark.asyncio
async def test_knowledge_upsert_handles_legacy_file_prefix_conflict(tmp_path: Path) -> None:
    settings = AgentTreeSettings(data_dir=tmp_path / ".agenttree-data")
    settings.ensure_dirs()
    store = KnowledgeStore(settings)

    legacy_file = settings.knowledge_dir / "system" / "executors" / "temperature"
    legacy_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_file.write_text("legacy temperature executor document", encoding="utf-8")
    store.metadata["/system/executors/temperature"] = {
        "doc_id": "legacy-doc",
        "doc_path": "/system/executors/temperature",
        "owner_node_path": "/system",
        "updated_at": "2026-04-06T00:00:00+00:00",
        "content_length": len("legacy temperature executor document"),
        "chunks_count": 1,
    }

    await store.upsert_document(
        owner_node_path="/system",
        doc_path="/system/executors/temperature/intro",
        text="nested document below legacy prefix",
    )

    assert store.read_document("/system/executors/temperature") == "legacy temperature executor document"
    assert store.read_document("/system/executors/temperature/intro") == "nested document below legacy prefix"


@pytest.mark.asyncio
async def test_upsert_knowledge_returns_structured_error_on_500(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AgentTreeSettings(data_dir=tmp_path / ".agenttree-data")
    settings.ensure_dirs()

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict[str, str]) -> httpx.Response:
            request = httpx.Request("POST", f"http://testserver{url}", json=json)
            return httpx.Response(status_code=500, request=request, text="backend exploded")

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    tools = build_knowledge_tools(settings, "/root_manager/turbine_operator")
    upsert_knowledge = tools[2]
    payload = await upsert_knowledge("/docs/test", "hello")
    parsed = json.loads(payload)

    assert parsed["ok"] is False
    assert parsed["tool_name"] == "upsert_knowledge"
    assert parsed["error_kind"] == "http_status_error"
    assert parsed["details"]["status_code"] == 500


def test_manual_knowledge_templates_are_available() -> None:
    templates = list_knowledge_templates()

    assert templates
    assert any(item.name == "tree_operating_rules" for item in templates)


def test_get_manual_knowledge_template_returns_expected_content() -> None:
    template = get_knowledge_template("queue_handling_guide")

    assert template.default_doc_path == "/rules/queue_handling_guide.md"
    assert "command" in template.text