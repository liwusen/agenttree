from __future__ import annotations

import json

import httpx

from agenttree.agent_runtime.tools.common import ToolTraceHook, run_tool_action
from agenttree.config import AgentTreeSettings


def build_knowledge_tools(settings: AgentTreeSettings, self_path: str, trace_hook: ToolTraceHook | None = None) -> list:
    def not_found_payload(*, doc_path: str) -> str:
        """Return a structured not-found payload for knowledge reads."""
        return json.dumps(
            {
                "doc_path": doc_path,
                "text": "",
                "found": False,
                "error": "knowledge_document_not_found",
            },
            ensure_ascii=False,
        )

    async def query_knowledge(scope: str, query: str) -> str:
        """Query knowledge visible to the current node within the given scope."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.get(
                    f"{settings.api_prefix}/knowledge/query",
                    params={"node_path": self_path, "scope": scope, "query": query},
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="query_knowledge",
            args={"scope": scope, "query": query},
            action=action,
            trace_hook=trace_hook,
        )

    async def read_knowledge(doc_path: str) -> str:
        """Read the content of a knowledge document by path."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.get(
                    f"{settings.api_prefix}/knowledge/content",
                    params={"doc_path": doc_path},
                )
                if response.status_code == 404:
                    return not_found_payload(doc_path=doc_path)
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="read_knowledge",
            args={"doc_path": doc_path},
            action=action,
            trace_hook=trace_hook,
        )

    async def upsert_knowledge(doc_path: str, text: str) -> str:
        """Create or update a knowledge document owned by the current node."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.post(
                    f"{settings.api_prefix}/knowledge/upsert",
                    json={"owner_node_path": self_path, "doc_path": doc_path, "text": text},
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="upsert_knowledge",
            args={"doc_path": doc_path, "text": text},
            action=action,
            trace_hook=trace_hook,
        )

    return [query_knowledge, read_knowledge, upsert_knowledge]