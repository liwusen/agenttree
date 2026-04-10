from __future__ import annotations

import httpx
import pytest

from agenttree.agent_runtime.tools.node_ops import build_node_tools
from agenttree.config import AgentTreeSettings


@pytest.mark.asyncio
async def test_get_tree_structure_returns_readable_text(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AgentTreeSettings(data_dir=tmp_path / ".agenttree-data")
    settings.ensure_dirs()

    tree_payload = {
        "nodes": [
            {"path": "/", "kind": "root", "status": "online", "metadata": {}},
            {"path": "/manager", "kind": "agent", "status": "online", "description": "root manager", "metadata": {}},
            {
                "path": "/manager/cmd.executor",
                "kind": "executor",
                "status": "online",
                "owner_path": "/manager",
                "capabilities": ["run_command", "command_status"],
                "metadata": {"executor_kind": "system.command"},
            },
        ],
        "children": {
            "/": ["/manager"],
            "/manager": ["/manager/cmd.executor"],
            "/manager/cmd.executor": [],
        },
        "channels": [
            {"channel_id": "ops-room", "members": ["/manager", "/manager/cmd.executor"], "metadata": {}},
        ],
    }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str) -> httpx.Response:
            request = httpx.Request("GET", f"http://testserver{url}")
            return httpx.Response(status_code=200, request=request, json=tree_payload)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    tools = build_node_tools(settings, "/manager")
    get_tree_structure = next(tool for tool in tools if tool.__name__ == "get_tree_structure")
    output = await get_tree_structure()

    assert output.startswith("Tree:\n")
    assert "/manager [agent | online] description=root manager" in output
    assert "/manager/cmd.executor -> owner=/manager | kind=system.command" in output
    assert "Channels:\n- ops-room: /manager, /manager/cmd.executor" in output
    assert '"nodes"' not in output