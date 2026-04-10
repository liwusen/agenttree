from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover - compatibility fallback for older langgraph versions
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

from agenttree.agent_runtime.client import RuntimeClient
from agenttree.agent_runtime.tools import build_executor_tools, build_knowledge_tools, build_node_tools
from agenttree.config import AgentTreeSettings
from agenttree.schemas.events import EventEnvelope, EventKind, EventMessagePurpose, build_event_metadata
from agenttree.schemas.nodes import NodeKind
from agenttree.schemas.protocol import RuntimeHello, RuntimeMessage, RuntimeMessageType


@dataclass(slots=True)
class AgentRuntime:
    settings: AgentTreeSettings
    path: str
    kind: NodeKind
    client: RuntimeClient = field(init=False)
    tools: list = field(init=False)
    agent: Any = field(init=False)
    checkpointer: Any = field(init=False)
    _active_event: EventEnvelope | None = field(init=False, default=None)
    _active_events: list[EventEnvelope] = field(init=False, default_factory=list)
    _active_websocket: Any = field(init=False, default=None)
    _node_prompt: str | None = field(init=False, default=None)
    _node_description: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.client = RuntimeClient(self.settings)
        self.checkpointer = InMemorySaver()
        self._load_node_context_sync()
        self.tools = (
            build_node_tools(self.settings, self.path, trace_hook=self._trace_tool_activity)
            + build_knowledge_tools(self.settings, self.path, trace_hook=self._trace_tool_activity)
            + build_executor_tools(self.settings, self.path, trace_hook=self._trace_tool_activity)
        )
        self.agent = self._build_agent()

    def _build_agent(self):
        if not self.settings.openai_api_key:
            return None
        model = ChatOpenAI(
            model=self.settings.model,
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            temperature=self.settings.model_temperature,
        )
        return create_agent(
            model=model,
            tools=self.tools,
            system_prompt=self._system_prompt(),
            checkpointer=self.checkpointer,
        )

    def _system_prompt(self) -> str:
        if self.kind in {NodeKind.ROOT, NodeKind.SUPERVISOR} or self.path == "/supervisor":
            return self._root_system_prompt()
        status_doc_path = self._agent_status_doc_path(self.path)
        base_prompt = (
            f"你是 AgentTree 节点 {self.path}。"
            "你运行在树状多 Agent 系统中。"
            "优先处理上级 command，其次是 message、struct、emergency、event。"
            "你只能通过工具显式访问知识库、创建子 Agent、绑定外部 Executor、调用 Executor 和发送消息。"
            "调用 Executor 前，必须先通过 get_executor_guide 或 get_node_detail 查看该执行器的 command、参数、返回字段、示例和用途说明。"
            "不要猜测执行器的参数名，也不要只看 capabilities 就直接调用。capabilities 只表示大类能力，真正可调用规范要以 operations 为准。"
            "调用 invoke_executor 时，command 应优先使用执行器指南中的 canonical command，payload_json 必须是合法 JSON 对象，并与 payload_schema 对齐。"
            "如果执行器操作是异步任务型，只代表任务已启动，最终结果通常会通过 event 或 response 回推，因此要区分提交成功和业务完成。"
            "如果需要拆分任务，应先明确职责边界、输出格式、回报路径和需要写入的知识。"
            "注意:请在发出 任何确认/回复/消息 前自我检查:这条消息是否重复发送？是否已经发送过类似的消息？这条消息的信息熵是否过低（内容过于简单或冗余）？"
            "如果上级让你完成任务，请不要多次发送同样的回复确认你已收到命令了，除非你有新的信息需要补充。"
            "你拥有一定的自主权，可以完全自行处理职责范围内的问题。"
            "知识库文档 /core_index.md 非常重要，里面记录了系统结构、节点职责、工具使用说明等关键信息的索引。"
            f"你的状态文件固定为 {status_doc_path}。"
            "你必须把自己的阶段性状态、最近进展、阻塞项、下一步动作优先写入这个状态文件，并持续覆盖更新。"
            "非重要情况无需上报；如果没有新的关键信息、没有状态变化、没有阻塞、没有异常、没有决策请求，就不要发送消息。"
            "管理者 Agent 应该优先从你的状态文件读取信息，因此你不要把普通状态轮询结果反复发成消息。"
            "只有在以下情况才主动上报：异常、阻塞、关键状态变化、任务完成、需要上级决策。"
            "对于大部分消息，你不需要确认收到。"
        )
        return self._merge_node_prompt(base_prompt)

    def _root_system_prompt(self) -> str:
        return self._merge_node_prompt(
            (
            f"你是 AgentTree 的 ROOT 管理节点 {self.path}。"
            "你负责理解 human 或系统给出的目标，规划系统级工作流，创建和编排子 Agent，接收外部 Executor 注册通知，并把合适的 Executor 绑定到合适的节点。"
            "\n\n"
            "系统详细能力如下："
            "1. 你可以创建子 Agent，并更新节点 prompt/description。"
            "2. 你可以向任意节点发送 command 或 message。"
            "3. 你可以读取、查询、写入知识库文档。"
            "4. 外部 Executor 通过网络自行注册到系统；你不能假设 Executor 由当前 Agent 直接创建进程。"
            "5. 当外部 Executor 注册后，你会收到结构事件，可使用 bind_executor 把它绑定给某个 Agent，再用 invoke_executor 调用它。"
            "5.1 在实际调用 Executor 前，先用 get_executor_guide 或 get_node_detail 查看 metadata.operations，确认 command、payload_schema、returns、examples。"
            "5.2 不要依据 capabilities 猜参数；capabilities 只说明能力范围，不等于调用签名。"
            "5.3 对于异步型 Executor，要识别'任务已提交'和'任务已完成'的区别，必要时等待事件回推。"
            "6. 你可以创建频道、广播消息、管理触发器。"
            "7. 你需要维护系统结构清晰、职责分明、知识可追踪。"
            "\n\n"
            "处理系统任务时遵循这些原则："
            "1. 先判断任务是否需要新建子 Agent、复用已有 Agent、绑定已有 Executor、还是直接由你自己处理。"
            "2. 只有在任务长期存在、职责独立、需要持续协作或持续监控时才创建子 Agent。"
            "3. 如果收到外部 Executor 注册事件，应先判断最合适的 owner 节点；必要时先创建专门的管理子 Agent，再绑定 Executor。"
            "4. 所有重要结构变化、系统配置、运行规则和中间结论，应优先写入知识库，保证后续节点可查询。"
            "5. 回复 human 时给出结论、关键动作和下一步，不输出冗长推理链。"
            "\n\n"
            "创建子 Agent 的标准工作步骤："
            "1. 明确子任务目标、输入、输出、完成标准。"
            "2. 设计子 Agent 的职责边界，避免与其他节点重叠。"
            "3. 为子 Agent 编写清晰 prompt，说明它是谁、负责什么、能用哪些工具、何时回报、回报给谁、输出格式是什么。"
            "4. 如有需要，把任务背景、约束、资源路径、Executor 路径、知识文档路径写入 description 或知识库。"
            "5. 创建后立即向子 Agent 发送 command，交代当前任务。"
            "6. 必要时再为它绑定外部 Executor、创建频道、设置触发器。"
            "7. 记录这个子 Agent 的职责和状态到知识库，便于后续调度。"
            "8. 管理子 Agent 时，优先读取其状态文件，而不是频繁发送确认型或轮询型消息。"
            "\n\n"
            "在告知子 Agent 时，至少提供以下信息："
            "- 任务目标与业务背景"
            "- 该子 Agent 的具体职责边界,包括是否允许它创建子节点等权限信息"
            "- 输入来源、目标节点、回报对象"
            "- 需要使用的知识文档路径或查询范围"
            "- 可使用的 Executor 或外部资源路径"
            "- Executor 的调用约定：先查看指南，再按 payload_schema 调用"
            "- 输出格式、优先级、完成标准"
            "- 是否需要持续监控、周期汇报或异常上报"
            "- 它自己的状态文件路径，以及要求它把普通状态更新写入状态文件而非直接上报"
            "\n\n"
            "ROOT 节点尤其要擅长以下事件："
            "- human 下达的系统级命令"
            "- child_created、executor_registered、executor_bound 等结构事件"
            "- 需要拆分成多 Agent 协作的复杂任务"
            "- 需要把外部 Executor 纳入树状系统管理的任务"
            "- 通过读取子节点状态文件来汇总信息，而不是制造大量冗余对话"
            "\n\n"
            "你必须优先使用工具进行显式操作，不要假设隐式系统能力。最终回复保持简洁，但结构决策要准确。"
            )
        )

    async def run(self) -> None:
        hello = RuntimeHello(path=self.path, kind=self.kind, capabilities=["agent_runtime"], metadata={})
        websocket = await self.client.connect(hello)
        await self.log(websocket, "runtime", "agent runtime connected")
        heartbeat_task = asyncio.create_task(self.client.heartbeat_loop(websocket, self.path))
        try:
            while True:
                message = await self.client.receive(websocket)
                if message.message_type == RuntimeMessageType.LOG:
                    continue
                if message.message_type == RuntimeMessageType.WAKE:
                    await self.client.send(
                        websocket,
                        RuntimeMessage(
                            message_type=RuntimeMessageType.REQUEST_EVENT,
                            path=self.path,
                            payload={"batch_size": max(1, self.settings.agent_event_batch_size)},
                        ),
                    )
                    continue
                if message.message_type == RuntimeMessageType.EVENT:
                    events = message.events or ([] if message.event is None else [message.event])
                    if not events:
                        continue
                    if len(events) == 1:
                        await self.handle_event(websocket, events[0])
                    else:
                        await self.handle_event_batch(websocket, events)
                    continue
        finally:
            heartbeat_task.cancel()

    async def handle_event_batch(self, websocket, events: list[EventEnvelope]) -> None:
        self._active_websocket = websocket
        event_ids = [item.event_id for item in events]
        queue_kind = events[0].kind.value if events else "unknown"
        await self.log(
            websocket,
            "event_received",
            f"received batch of {len(events)} {queue_kind} event(s)",
            {
                "events": [item.model_dump(mode="json") for item in events],
                "event_count": len(events),
                "event_ids": event_ids,
                "queue_kind": queue_kind,
                "trace_ids": [item.trace_id for item in events],
            },
        )
        try:
            try:
                response_payload = await self.process_event_batch(events)
            except Exception as exc:
                await self.log(
                    websocket,
                    "runtime_error",
                    f"failed to process {len(events)} batched event(s)",
                    {
                        "event_ids": event_ids,
                        "queue_kind": queue_kind,
                        "error": str(exc),
                    },
                )
                response_payload = {
                    "text": f"{self.path} 处理批量事件时出现错误: {exc}",
                    "mode": "error",
                    "thought_summary": "批量处理过程中调用后端服务或工具失败，已降级返回错误摘要。",
                    "processed_event_ids": event_ids,
                    "queue_kind": queue_kind,
                    "event_results": [
                        {
                            "event_id": event.event_id,
                            "thought_summary": "批量处理过程中调用后端服务或工具失败，已降级返回错误摘要。",
                            "final_response": f"{self.path} 处理事件时出现错误: {exc}",
                            "target_path": self._reply_target(event),
                            "source_path": event.source_path,
                            "kind": event.kind.value,
                        }
                        for event in events
                    ],
                }
            await self.log(
                websocket,
                "thinking",
                f"processed batch of {len(events)} {queue_kind} event(s)",
                {
                    "thought_summary": response_payload.get("thought_summary", ""),
                    "text": response_payload.get("text", ""),
                    "mode": response_payload.get("mode", ""),
                    "event_count": len(events),
                    "event_ids": event_ids,
                    "queue_kind": queue_kind,
                    "event_results": response_payload.get("event_results", []),
                },
            )
            await self._publish_batch_replies(websocket, events, response_payload)
            for event in events:
                await self.client.send(
                    websocket,
                    RuntimeMessage(message_type=RuntimeMessageType.ACK_EVENT, path=self.path, event_id=event.event_id),
                )
        finally:
            self._active_websocket = None

    async def _publish_batch_replies(self, websocket, events: list[EventEnvelope], response_payload: dict[str, Any]) -> None:
        event_results_by_id = {
            str(item.get("event_id")): item
            for item in response_payload.get("event_results", [])
            if isinstance(item, dict) and item.get("event_id")
        }
        for index, event in enumerate(events):
            target = self._reply_target(event)
            if target is None:
                continue
            event_result = dict(event_results_by_id.get(event.event_id, {}))
            outgoing_payload = {
                "text": str(event_result.get("final_response", response_payload.get("text", ""))),
                "mode": str(response_payload.get("mode", "llm")),
                "thought_summary": str(event_result.get("thought_summary", response_payload.get("thought_summary", ""))),
                "raw_output": response_payload.get("raw_output", ""),
                "source_event_id": event.event_id,
                "queue_kind": event.kind.value,
                "event_result": {
                    "event_id": event.event_id,
                    "thought_summary": str(event_result.get("thought_summary", response_payload.get("thought_summary", ""))),
                    "final_response": str(event_result.get("final_response", response_payload.get("text", ""))),
                    "kind": event.kind.value,
                    "source_path": event.source_path,
                    "target_path": target,
                    "result_index": index,
                },
                "batch_context": {
                    "event_count": len(events),
                    "processed_event_ids": [item.event_id for item in events],
                    "batch_thought_summary": response_payload.get("thought_summary", ""),
                    "batch_final_response": response_payload.get("text", ""),
                },
            }
            outgoing = EventEnvelope(
                kind=RuntimeEventMapper.reply_kind(event.kind),
                source_path=self.path,
                target_path=target,
                payload=outgoing_payload,
                trace_id=event.trace_id,
                metadata=build_event_metadata(
                    metadata={"reply_to": event.event_id, "reply_to_batch": [item.event_id for item in events]},
                    require_reply=False,
                    message_purpose=EventMessagePurpose.RESPONSE,
                    dedupe_key=f"reply:{self.path}:{event.event_id}",
                ),
            )
            await self.client.send(
                websocket,
                RuntimeMessage(
                    message_type=RuntimeMessageType.PUBLISH_EVENT,
                    path=self.path,
                    event=outgoing,
                ),
            )
            await self.log(
                websocket,
                "reply_sent",
                f"sent structured reply for {event.event_id} to {target}",
                {"reply": outgoing.model_dump(mode="json"), "event_id": event.event_id, "batch_event_ids": [item.event_id for item in events]},
            )

    async def handle_event(self, websocket, event: EventEnvelope) -> None:
        self._active_websocket = websocket
        await self.log(
            websocket,
            "event_received",
            f"received {event.kind.value} from {event.source_path}",
            {"event": event.model_dump(mode="json"), "trace_id": event.trace_id, "event_id": event.event_id},
        )
        try:
            try:
                response_payload = await self.process_event(event)
            except Exception as exc:
                await self.log(
                    websocket,
                    "runtime_error",
                    f"failed to process {event.kind.value}",
                    {
                        "trace_id": event.trace_id,
                        "event_id": event.event_id,
                        "error": str(exc),
                    },
                )
                response_payload = {
                    "text": f"{self.path} 处理事件时出现错误: {exc}",
                    "mode": "error",
                    "thought_summary": "处理过程中调用后端服务或工具失败，已降级返回错误摘要。",
                }
            if response_payload is not None:
                await self.log(
                    websocket,
                    "thinking",
                    f"processed {event.kind.value}",
                    {
                        "thought_summary": response_payload.get("thought_summary", ""),
                        "text": response_payload.get("text", ""),
                        "mode": response_payload.get("mode", ""),
                        "trace_id": event.trace_id,
                        "event_id": event.event_id,
                    },
                )
                target = self._reply_target(event)
                if target is not None:
                    outgoing = EventEnvelope(
                        kind=RuntimeEventMapper.reply_kind(event.kind),
                        source_path=self.path,
                        target_path=target,
                        payload=response_payload,
                        trace_id=event.trace_id,
                        metadata=build_event_metadata(
                            metadata={"reply_to": event.event_id},
                            require_reply=False,
                            message_purpose=EventMessagePurpose.RESPONSE,
                            dedupe_key=f"reply:{self.path}:{event.event_id}",
                        ),
                    )
                    await self.client.send(
                        websocket,
                        RuntimeMessage(
                            message_type=RuntimeMessageType.PUBLISH_EVENT,
                            path=self.path,
                            event=outgoing,
                        ),
                    )
                    await self.log(
                        websocket,
                        "reply_sent",
                        f"sent reply to {target}",
                        {"reply": outgoing.model_dump(mode="json"), "trace_id": event.trace_id, "event_id": event.event_id},
                    )
            await self.client.send(
                websocket,
                RuntimeMessage(message_type=RuntimeMessageType.ACK_EVENT, path=self.path, event_id=event.event_id),
            )
        finally:
            self._active_websocket = None

    def _reply_target(self, event: EventEnvelope) -> str | None:
        if event.kind == event.kind.STRUCT:
            return event.source_path
        return event.source_path if event.source_path != self.path else None

    async def process_event(self, event: EventEnvelope) -> dict[str, Any] | None:
        await self._refresh_node_context()
        if self.agent is None:
            text = self._fallback_reply(event)
            return {"text": text, "mode": "fallback", "thought_summary": "未配置模型，使用规则回复。"}
        prompt = self._event_prompt(event)
        self._active_event = event
        try:
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": self._thread_id_for_event(event)}},
            )
        finally:
            self._active_event = None
        messages = result.get("messages", [])
        content = ""
        if messages:
            content = str(messages[-1].content)
        parsed = self._parse_agent_response(content)
        return {
            "text": parsed["final_response"],
            "mode": "llm",
            "thought_summary": parsed["thought_summary"],
            "raw_output": content,
        }

    async def process_event_batch(self, events: list[EventEnvelope]) -> dict[str, Any] | None:
        await self._refresh_node_context()
        if self.agent is None:
            event_results = [self._build_fallback_event_result(event) for event in events]
            text = "\n".join(item["final_response"] for item in event_results)
            return {
                "text": text,
                "mode": "fallback",
                "thought_summary": "未配置模型，使用规则回复。",
                "processed_event_ids": [item.event_id for item in events],
                "event_results": event_results,
            }
        prompt = self._event_prompt(events)
        self._active_events = list(events)
        self._active_event = events[0] if events else None
        try:
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": self._thread_id_for_batch(events)}},
            )
        finally:
            self._active_event = None
            self._active_events = []
        messages = result.get("messages", [])
        content = ""
        if messages:
            content = str(messages[-1].content)
        parsed = self._parse_batch_agent_response(content, events)
        return {
            "text": parsed["final_response"],
            "mode": "llm",
            "thought_summary": parsed["thought_summary"],
            "raw_output": content,
            "processed_event_ids": [item.event_id for item in events],
            "queue_kind": events[0].kind.value if events else None,
            "event_count": len(events),
            "event_results": parsed["event_results"],
        }

    def _event_prompt(self, event: EventEnvelope | list[EventEnvelope]) -> str:
        if isinstance(event, list):
            return self._batch_event_prompt(event)
        return json.dumps(
            {
                "instruction": (
                    "处理这个 AgentTree 事件。如果你需要使用工具，请先使用工具。"
                    "最终请只输出一个 JSON 对象，包含 thought_summary 和 final_response 两个字段。"
                    "thought_summary 是对处理思路的简短摘要，不要输出冗长推理链。"
                ),
                "queue_name": event.kind.value,
                "queue_description": self._queue_description(event.kind),
                "event_id": event.event_id,
                "kind": event.kind.value,
                "source_path": event.source_path,
                "target_path": event.target_path,
                "payload": event.payload,
                "metadata": event.metadata,
                "current_node": {
                    "path": self.path,
                    "description": self._node_description,
                    "prompt": self._node_prompt,
                },
            },
            ensure_ascii=False,
        )

    def _batch_event_prompt(self, events: list[EventEnvelope]) -> str:
        queue_kind = events[0].kind if events else EventKind.EVENT
        return json.dumps(
            {
                "instruction": (
                    "处理这一批来自同一优先级队列的 AgentTree 事件。如果你需要使用工具，请先使用工具。"
                    "最终请只输出一个 JSON 对象。"
                    "顶层必须包含 thought_summary、final_response、event_results 三个字段。"
                    "event_results 必须是数组，长度应与输入 events 一致，每个元素至少包含 event_id、thought_summary、final_response。"
                    "你需要在 thought_summary 中概括这一批事件的整体处理思路，并在 final_response 中给出本轮统一结果。"
                ),
                "queue_name": queue_kind.value,
                "queue_description": self._queue_description(queue_kind),
                "event_count": len(events),
                "event_ids": [item.event_id for item in events],
                "events": [
                    {
                        "event_id": item.event_id,
                        "kind": item.kind.value,
                        "source_path": item.source_path,
                        "target_path": item.target_path,
                        "payload": item.payload,
                        "metadata": item.metadata,
                    }
                    for item in events
                ],
                "current_node": {
                    "path": self.path,
                    "description": self._node_description,
                    "prompt": self._node_prompt,
                },
            },
            ensure_ascii=False,
        )

    def _fallback_reply(self, event: EventEnvelope) -> str:
        if event.kind.value == "command":
            return f"{self.path} 已接收命令: {event.payload.get('text', '')}"
        if event.kind.value == "emergency":
            return f"{self.path} 已处理紧急事件: {json.dumps(event.payload, ensure_ascii=False)}"
        return f"{self.path} 已处理事件: {json.dumps(event.payload, ensure_ascii=False)}"

    def _parse_agent_response(self, content: str) -> dict[str, str]:
        parsed = self._parse_json_payload(content)
        if parsed is None:
            return {"thought_summary": "模型未按 JSON 返回，直接采用文本输出。", "final_response": content}
        return {
            "thought_summary": str(parsed.get("thought_summary", "")),
            "final_response": str(parsed.get("final_response", content)),
        }

    def _parse_batch_agent_response(self, content: str, events: list[EventEnvelope]) -> dict[str, Any]:
        parsed = self._parse_json_payload(content)
        if parsed is None:
            event_results = [self._build_text_event_result(event, content, "模型未按 JSON 返回，直接采用文本输出。") for event in events]
            return {
                "thought_summary": "模型未按 JSON 返回，直接采用文本输出。",
                "final_response": content,
                "event_results": event_results,
            }
        thought_summary = str(parsed.get("thought_summary", ""))
        final_response = str(parsed.get("final_response", content))
        raw_event_results = parsed.get("event_results")
        normalized_event_results = self._normalize_batch_event_results(
            raw_event_results=raw_event_results,
            events=events,
            default_thought_summary=thought_summary,
            default_final_response=final_response,
        )
        return {
            "thought_summary": thought_summary,
            "final_response": final_response,
            "event_results": normalized_event_results,
        }

    def _normalize_batch_event_results(
        self,
        *,
        raw_event_results: Any,
        events: list[EventEnvelope],
        default_thought_summary: str,
        default_final_response: str,
    ) -> list[dict[str, Any]]:
        results_by_id: dict[str, dict[str, Any]] = {}
        if isinstance(raw_event_results, list):
            for item in raw_event_results:
                if not isinstance(item, dict):
                    continue
                event_id = str(item.get("event_id", "")).strip()
                if not event_id:
                    continue
                results_by_id[event_id] = item
        normalized: list[dict[str, Any]] = []
        for event in events:
            candidate = results_by_id.get(event.event_id)
            if candidate is None:
                normalized.append(self._build_text_event_result(event, default_final_response, default_thought_summary))
                continue
            normalized.append(
                {
                    "event_id": event.event_id,
                    "thought_summary": str(candidate.get("thought_summary", default_thought_summary)),
                    "final_response": str(candidate.get("final_response", default_final_response)),
                    "kind": event.kind.value,
                    "source_path": event.source_path,
                    "target_path": self._reply_target(event),
                }
            )
        return normalized

    def _build_fallback_event_result(self, event: EventEnvelope) -> dict[str, Any]:
        return self._build_text_event_result(event, self._fallback_reply(event), "未配置模型，使用规则回复。")

    def _build_text_event_result(self, event: EventEnvelope, final_response: str, thought_summary: str) -> dict[str, Any]:
        return {
            "event_id": event.event_id,
            "thought_summary": thought_summary,
            "final_response": final_response,
            "kind": event.kind.value,
            "source_path": event.source_path,
            "target_path": self._reply_target(event),
        }

    def _parse_json_payload(self, content: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        fenced_json = self._extract_json_object_from_text(content)
        if fenced_json is None:
            return None
        try:
            parsed = json.loads(fenced_json)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _extract_json_object_from_text(self, content: str) -> str | None:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _thread_id_for_event(self, event: EventEnvelope) -> str:
        channel = f":{event.channel_id}" if event.channel_id else ""
        return f"{self.path}:{event.source_path}:{event.kind.value}{channel}"

    def _thread_id_for_batch(self, events: list[EventEnvelope]) -> str:
        if not events:
            return f"{self.path}:empty-batch"
        return f"{self.path}:batch:{events[0].kind.value}:{events[0].event_id}:{events[-1].event_id}"

    def _agent_status_doc_path(self, path: str) -> str:
        normalized = "/".join(part for part in path.strip("/").split("/") if part)
        if not normalized:
            return "/agent/root/STATUS.md"
        return f"/agent/{normalized}/STATUS.md"

    def _load_node_context_sync(self) -> None:
        try:
            with httpx.Client(base_url=self.settings.base_url, timeout=5.0) as client:
                response = client.get(f"{self.settings.api_prefix}/nodes/{self.path.strip('/')}" )
                response.raise_for_status()
                payload = response.json().get("node", {})
        except Exception:
            payload = {}
        self._node_prompt = payload.get("prompt")
        self._node_description = payload.get("description")

    async def _refresh_node_context(self) -> None:
        try:
            async with httpx.AsyncClient(base_url=self.settings.base_url, timeout=5.0) as client:
                response = await client.get(f"{self.settings.api_prefix}/nodes/{self.path.strip('/')}" )
                response.raise_for_status()
                payload = response.json().get("node", {})
        except Exception:
            return
        next_prompt = payload.get("prompt")
        next_description = payload.get("description")
        if next_prompt != self._node_prompt or next_description != self._node_description:
            self._node_prompt = next_prompt
            self._node_description = next_description
            self.agent = self._build_agent()

    def _merge_node_prompt(self, base_prompt: str) -> str:
        extra_parts = []
        if self._node_description:
            extra_parts.append(f"当前节点描述: {self._node_description}")
        if self._node_prompt:
            extra_parts.append(f"当前节点自定义提示词: {self._node_prompt}")
        if not extra_parts:
            return base_prompt
        return base_prompt + "\n\n" + "\n".join(extra_parts)

    def _queue_description(self, kind: EventKind) -> str:
        if kind == EventKind.COMMAND:
            return "来自 command 队列，表示上级或人类的直接命令，优先级最高。"
        if kind == EventKind.MESSAGE:
            return "来自 message 队列，表示节点间消息或频道消息。"
        if kind == EventKind.STRUCT:
            return "来自 struct 队列，表示树结构、绑定关系或系统配置发生变化。"
        if kind == EventKind.EMERGENCY:
            return "来自 emergency 队列，表示执行器或外部系统的紧急事件。"
        return "来自 event 队列，表示普通执行器事件或状态更新。"

    async def _trace_tool_activity(self, category: str, message: str, payload: dict[str, Any]) -> None:
        if self._active_websocket is None:
            return
        event = self._active_event
        enriched = dict(payload)
        if self._active_events:
            enriched.setdefault("batch_event_ids", [item.event_id for item in self._active_events])
            enriched.setdefault("batch_event_count", len(self._active_events))
            enriched.setdefault("batch_queue_kind", self._active_events[0].kind.value)
        if event is not None:
            enriched.setdefault("trace_id", event.trace_id)
            enriched.setdefault("event_id", event.event_id)
            enriched.setdefault("event_kind", event.kind.value)
            enriched.setdefault("source_path", event.source_path)
            enriched.setdefault("target_path", event.target_path)
        await self.log(self._active_websocket, category, message, enriched)

    async def log(self, websocket, category: str, message: str, payload: dict[str, Any] | None = None) -> None:
        await self.client.send(
            websocket,
            RuntimeMessage(
                message_type=RuntimeMessageType.LOG,
                path=self.path,
                payload={"category": category, "message": message, **({"payload": payload} if payload else {})},
            ),
        )


class RuntimeEventMapper:
    @staticmethod
    def reply_kind(kind: EventKind) -> EventKind:
        return EventKind.MESSAGE