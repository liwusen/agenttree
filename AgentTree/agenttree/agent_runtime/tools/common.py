from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx


ToolTraceHook = Callable[[str, str, dict[str, Any]], Awaitable[None]]


def error_payload(*, tool_name: str, args: dict[str, Any], error_kind: str, message: str, details: dict[str, Any] | None = None) -> str:
    payload = {
        "ok": False,
        "tool_name": tool_name,
        "error_kind": error_kind,
        "message": message,
        "args": args,
    }
    if details:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)


def preview_value(value: Any, limit: int = 400) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


async def run_tool_action(
    *,
    tool_name: str,
    args: dict[str, Any],
    action: Callable[[], Awaitable[str]],
    trace_hook: ToolTraceHook | None = None,
) -> str:
    started = time.perf_counter()
    if trace_hook is not None:
        await trace_hook("tool_call", f"tool {tool_name} started", {"tool_name": tool_name, "args": args})
    try:
        result = await action()
    except httpx.HTTPStatusError as exc:
        details = {
            "status_code": exc.response.status_code,
            "url": str(exc.request.url),
            "response_text": exc.response.text,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        if trace_hook is not None:
            await trace_hook(
                "tool_error",
                f"tool {tool_name} failed with http error",
                {
                    "tool_name": tool_name,
                    "args": args,
                    "error": str(exc),
                    **details,
                },
            )
        return error_payload(
            tool_name=tool_name,
            args=args,
            error_kind="http_status_error",
            message=str(exc),
            details=details,
        )
    except httpx.HTTPError as exc:
        details = {
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        if trace_hook is not None:
            await trace_hook(
                "tool_error",
                f"tool {tool_name} failed with http transport error",
                {
                    "tool_name": tool_name,
                    "args": args,
                    "error": str(exc),
                    **details,
                },
            )
        return error_payload(
            tool_name=tool_name,
            args=args,
            error_kind="http_error",
            message=str(exc),
            details=details,
        )
    except Exception as exc:
        if trace_hook is not None:
            await trace_hook(
                "tool_error",
                f"tool {tool_name} failed",
                {
                    "tool_name": tool_name,
                    "args": args,
                    "error": str(exc),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
        raise
    if trace_hook is not None:
        await trace_hook(
            "tool_result",
            f"tool {tool_name} completed",
            {
                "tool_name": tool_name,
                "args": args,
                "result_preview": preview_value(result),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
    return result