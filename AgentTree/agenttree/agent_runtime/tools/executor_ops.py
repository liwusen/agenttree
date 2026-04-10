from __future__ import annotations

import json

import httpx

from agenttree.agent_runtime.tools.common import ToolTraceHook, run_tool_action
from agenttree.config import AgentTreeSettings


def build_executor_tools(settings: AgentTreeSettings, self_path: str, trace_hook: ToolTraceHook | None = None) -> list:
    def parse_json_object(text: str) -> dict:
        """Parse a JSON object string and return an empty dict on invalid input."""
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    async def bind_executor(executor_path: str, owner_path: str = "") -> str:
        """Bind a previously registered external executor to an owner node path."""
        resolved_owner = owner_path or self_path

        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.post(
                    f"{settings.api_prefix}/executors/bind",
                    json={
                        "executor_path": executor_path,
                        "owner_path": resolved_owner,
                    },
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="bind_executor",
            args={"executor_path": executor_path, "owner_path": resolved_owner},
            action=action,
            trace_hook=trace_hook,
        )

    async def invoke_executor(executor_path: str, command: str, payload_json: str = "{}") -> str:
        """Invoke an executor with a command and optional JSON payload. Prefer reading get_executor_guide first."""
        payload = parse_json_object(payload_json)

        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.post(
                    f"{settings.api_prefix}/executors/invoke",
                    json={
                        "source_path": self_path,
                        "executor_path": executor_path,
                        "command": command,
                        "payload": payload,
                    },
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="invoke_executor",
            args={"executor_path": executor_path, "command": command, "payload": payload},
            action=action,
            trace_hook=trace_hook,
        )

    async def get_executor_guide(executor_path: str) -> str:
        """Get a structured executor usage guide including supported commands, parameters, returns, and examples."""
        normalized_path = executor_path.strip("/")

        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.get(f"{settings.api_prefix}/nodes/{normalized_path}")
                response.raise_for_status()
                payload = response.json().get("node", {})
                metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
                guide = {
                    "path": payload.get("path"),
                    "kind": payload.get("kind"),
                    "description": payload.get("description"),
                    "executor_kind": metadata.get("executor_kind"),
                    "capabilities": payload.get("capabilities", []),
                    "usage": metadata.get("executor_usage", {}),
                    "operations": metadata.get("operations", []),
                }
                return json.dumps(guide, ensure_ascii=False)

        return await run_tool_action(
            tool_name="get_executor_guide",
            args={"executor_path": executor_path},
            action=action,
            trace_hook=trace_hook,
        )

    async def transfer_executor(executor_path: str, new_owner_path: str) -> str:
        """Transfer an executor to a new owner node path."""
        async def action() -> str:
            async with httpx.AsyncClient(base_url=settings.base_url, timeout=30.0) as client:
                response = await client.post(
                    f"{settings.api_prefix}/executors/transfer",
                    json={"executor_path": executor_path, "new_owner_path": new_owner_path},
                )
                response.raise_for_status()
                return response.text

        return await run_tool_action(
            tool_name="transfer_executor",
            args={"executor_path": executor_path, "new_owner_path": new_owner_path},
            action=action,
            trace_hook=trace_hook,
        )

    return [bind_executor, invoke_executor, get_executor_guide, transfer_executor]