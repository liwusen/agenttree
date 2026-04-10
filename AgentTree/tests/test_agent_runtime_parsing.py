from __future__ import annotations

from agenttree.schemas.nodes import NodeKind
from agenttree.schemas.events import EventEnvelope, EventKind
from agenttree.agent_runtime.runtime import AgentRuntime


def test_parse_agent_response_accepts_plain_json() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)

    parsed = runtime._parse_agent_response('{"thought_summary": "done", "final_response": "ok"}')

    assert parsed == {"thought_summary": "done", "final_response": "ok"}


def test_parse_agent_response_accepts_fenced_json() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)

    parsed = runtime._parse_agent_response(
        "```json\n{\"thought_summary\": \"batched\", \"final_response\": \"ok\"}\n```"
    )

    assert parsed == {"thought_summary": "batched", "final_response": "ok"}


def test_parse_agent_response_accepts_fenced_json_with_prefix_and_suffix() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)

    parsed = runtime._parse_agent_response(
        "先分析一下\n```json\n{\"thought_summary\": \"batched\", \"final_response\": \"handled\"}\n```\n处理完成"
    )

    assert parsed == {"thought_summary": "batched", "final_response": "handled"}


def test_parse_agent_response_falls_back_to_plain_text() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)

    parsed = runtime._parse_agent_response("this is not json")

    assert parsed["thought_summary"] == "模型未按 JSON 返回，直接采用文本输出。"
    assert parsed["final_response"] == "this is not json"


def test_parse_batch_agent_response_uses_event_results() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.path = "/worker"
    events = [
        EventEnvelope(kind=EventKind.MESSAGE, source_path="/child/1", target_path="/worker", payload={}),
        EventEnvelope(kind=EventKind.MESSAGE, source_path="/child/2", target_path="/worker", payload={}),
    ]
    content = (
        '{'
        '"thought_summary": "batch done", '
        '"final_response": "all handled", '
        '"event_results": ['
        f'{{"event_id": "{events[0].event_id}", "thought_summary": "first", "final_response": "handled-1"}},'
        f'{{"event_id": "{events[1].event_id}", "thought_summary": "second", "final_response": "handled-2"}}'
        ']'
        '}'
    )

    parsed = runtime._parse_batch_agent_response(content, events)

    assert parsed["thought_summary"] == "batch done"
    assert parsed["final_response"] == "all handled"
    assert parsed["event_results"][0]["event_id"] == events[0].event_id
    assert parsed["event_results"][0]["final_response"] == "handled-1"
    assert parsed["event_results"][1]["event_id"] == events[1].event_id
    assert parsed["event_results"][1]["final_response"] == "handled-2"


def test_parse_batch_agent_response_falls_back_per_event_when_missing_event_results() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.path = "/worker"
    events = [
        EventEnvelope(kind=EventKind.MESSAGE, source_path="/child/1", target_path="/worker", payload={}),
        EventEnvelope(kind=EventKind.MESSAGE, source_path="/child/2", target_path="/worker", payload={}),
    ]

    parsed = runtime._parse_batch_agent_response('{"thought_summary": "batch", "final_response": "shared"}', events)

    assert len(parsed["event_results"]) == 2
    assert all(item["final_response"] == "shared" for item in parsed["event_results"])
    assert parsed["event_results"][0]["event_id"] == events[0].event_id
    assert parsed["event_results"][1]["event_id"] == events[1].event_id


def test_system_prompt_mentions_status_doc_and_non_important_updates() -> None:
    runtime = AgentRuntime.__new__(AgentRuntime)
    runtime.kind = NodeKind.AGENT
    runtime.path = "/manager/worker"
    runtime._node_prompt = None
    runtime._node_description = None

    prompt = AgentRuntime._system_prompt(runtime)

    assert "/agent/manager/worker/STATUS.md" in prompt
    assert "非重要情况无需上报" in prompt
    assert "管理者 Agent 应该优先从你的状态文件读取信息" in prompt