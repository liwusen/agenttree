from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import websockets
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, ListItem, ListView, Pretty, Static, TabbedContent, TabPane, TextArea, Tree


JsonRecord = dict[str, Any]


@dataclass(slots=True)
class DashboardSnapshot:
    tree: JsonRecord = field(default_factory=lambda: {"nodes": [], "children": {}, "channels": []})
    status: JsonRecord = field(default_factory=lambda: {"processes": [], "sessions": [], "queues": {}})
    knowledge: JsonRecord = field(default_factory=lambda: {"documents": []})
    traces: list[JsonRecord] = field(default_factory=list)


@dataclass(slots=True)
class DisplayEntry:
    summary: str
    title: str
    detail: Any
    action: str | None = None
    value: str | None = None


def format_display_text(value: Any, indent: int = 0) -> str:
    prefix = "  " * indent
    if value is None:
        return f"{prefix}无"
    if isinstance(value, str):
        return "\n".join(f"{prefix}{line}" if line else "" for line in value.splitlines()) or prefix
    if isinstance(value, (int, float, bool)):
        return f"{prefix}{value}"
    if isinstance(value, list):
        if not value:
            return f"{prefix}无"
        lines: list[str] = []
        for index, item in enumerate(value, start=1):
            rendered = format_display_text(item, indent + 1)
            first, *rest = rendered.splitlines()
            lines.append(f"{prefix}{index}. {first.strip()}")
            lines.extend(rest)
        return "\n".join(lines)
    if isinstance(value, dict):
        if not value:
            return f"{prefix}无"
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(format_display_text(item, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {format_display_text(item, 0).strip()}")
        return "\n".join(lines)
    return f"{prefix}{value}"


class MessageScreen(ModalScreen[None]):
    def __init__(self, title: str, body: Any) -> None:
        super().__init__()
        self.title = title
        self.body = body

    def compose(self) -> ComposeResult:
        with Container(id="message-modal"):
            yield Static(self.title, classes="modal-title")
            yield TextArea(format_display_text(self.body), id="message-body", read_only=True)
            yield Button("关闭", id="close-message", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-message":
            self.dismiss(None)


class AgentTreeCLIApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
        background: #10141a;
        color: #f4f1e8;
    }

    #topbar {
        height: 3;
        layout: horizontal;
        padding: 0 1;
        background: #18212b;
        border-bottom: solid #2f4458;
    }

    #topbar Static {
        width: 1fr;
        content-align: center middle;
    }

    #summary {
        height: 4;
        layout: horizontal;
        padding: 0 1;
        background: #131a22;
        border-bottom: solid #23303d;
    }

    .metric {
        width: 1fr;
        padding: 0 1;
        content-align: center middle;
        border-right: solid #23303d;
    }

    #pages {
        height: 1fr;
    }

    .page-grid {
        layout: horizontal;
        height: 1fr;
    }

    .column {
        width: 1fr;
        padding: 1;
        border-right: solid #23303d;
    }

    .column:last-child {
        border-right: none;
    }

    .section-title {
        height: 1;
        color: #9ec4ff;
        margin-bottom: 1;
    }

    .chat-feed {
        height: 1fr;
        border: round #23303d;
        padding: 0 1;
    }

    .chat-note {
        height: auto;
        color: #c5d6ea;
        margin-top: 1;
    }

    .live-status {
        height: auto;
        color: #ffd166;
        border: round #3c4e61;
        padding: 0 1;
        margin-bottom: 1;
    }

    ListView, DataTable, Pretty, Tree, TextArea, Input {
        height: 1fr;
    }

    #knowledge-query-controls, #command-controls, #chat-controls {
        height: auto;
        layout: vertical;
        margin-top: 1;
    }

    #knowledge-query-controls Button, #command-controls Button, #chat-controls Button {
        margin-top: 1;
    }

    #message-modal {
        width: 70;
        height: auto;
        padding: 1 2;
        background: #18212b;
        border: round #5b7a99;
    }

    .modal-title {
        color: #ffd166;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("r", "refresh", "刷新"),
        ("1", "switch_tab('overview')", "概览"),
        ("2", "switch_tab('topology')", "拓扑"),
        ("3", "switch_tab('timeline')", "时间线"),
        ("4", "switch_tab('knowledge')", "知识库"),
        ("6", "switch_tab('chat')", "对话"),
        ("7", "switch_tab('queues')", "队列"),
        ("q", "quit", "退出"),
    ]

    selected_path = reactive("/")
    selected_doc_path = reactive("")
    selected_queue_kind = reactive("command")
    ws_state = reactive("connecting")
    command_state = reactive("")
    error_text = reactive("")
    chat_state = reactive("")

    def __init__(self) -> None:
        super().__init__()
        self.snapshot = DashboardSnapshot()
        self.knowledge_results: list[JsonRecord] = []
        self.doc_content: str = ""
        self.queue_snapshot: JsonRecord = {}
        self.pending_acks: list[JsonRecord] = []
        self._entry_index: dict[str, DisplayEntry] = {}
        self._row_details: dict[str, tuple[str, Any]] = {}
        self._entry_counter = 0
        self._client = httpx.AsyncClient(base_url="http://127.0.0.1:18990", timeout=30.0)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="topbar"):
            yield Static("AgentTree CLI Console")
            yield Static(id="focus-label")
            yield Static(id="status-label")
        with Horizontal(id="summary"):
            yield Static("节点: 0", classes="metric", id="metric-nodes")
            yield Static("知识: 0", classes="metric", id="metric-docs")
            yield Static("执行器: 0", classes="metric", id="metric-executors")
            yield Static("轨迹: 0", classes="metric", id="metric-traces")
        with TabbedContent(id="pages", initial="overview"):
            with TabPane("概览", id="overview"):
                with Horizontal(classes="page-grid"):
                    with Vertical(classes="column"):
                        yield Static("系统摘要", classes="section-title")
                        yield TextArea("", id="overview-summary", read_only=True)
                    with Vertical(classes="column"):
                        yield Static("最近轨迹", classes="section-title")
                        yield ListView(id="overview-traces")
                    with Vertical(classes="column"):
                        yield Static("最近知识文档", classes="section-title")
                        yield ListView(id="overview-docs")
            with TabPane("拓扑", id="topology"):
                with Horizontal(classes="page-grid"):
                    with Vertical(classes="column"):
                        yield Static("Agent 树", classes="section-title")
                        yield Tree("/", id="topology-tree")
                    with Vertical(classes="column"):
                        yield Static("节点详情", classes="section-title")
                        yield TextArea("", id="topology-node-detail", read_only=True)
                    with Vertical(classes="column"):
                        yield Static("执行器状态", classes="section-title")
                        yield DataTable(id="topology-executors")
            with TabPane("时间线", id="timeline"):
                with Horizontal(classes="page-grid"):
                    with Vertical(classes="column"):
                        yield Static("思考与工具时间线", classes="section-title")
                        yield ListView(id="timeline-traces")
                    with Vertical(classes="column"):
                        yield Static("事件流", classes="section-title")
                        yield DataTable(id="timeline-flow")
            with TabPane("知识库", id="knowledge"):
                with Horizontal(classes="page-grid"):
                    with Vertical(classes="column"):
                        yield Static("知识文档", classes="section-title")
                        yield ListView(id="knowledge-docs")
                    with Vertical(classes="column"):
                        yield Static("文档内容", classes="section-title")
                        yield TextArea("", id="knowledge-content", read_only=True)
                    with Vertical(classes="column"):
                        yield Static("知识检索", classes="section-title")
                        yield TextArea(id="knowledge-query")
                        with Vertical(id="knowledge-query-controls"):
                            yield Button("运行检索", id="run-knowledge-query", variant="primary")
                            yield ListView(id="knowledge-results")
            with TabPane("对话", id="chat"):
                with Horizontal(classes="page-grid"):
                    with Vertical(classes="column"):
                        yield Static("Human <-> Supervisor", classes="section-title")
                        yield ListView(id="chat-feed", classes="chat-feed")
                    with Vertical(classes="column"):
                        yield Static("回复与思路摘要", classes="section-title")
                        yield Static("等待对话...", id="chat-live-status", classes="live-status")
                        yield TextArea("", id="chat-detail", read_only=True)
                        yield Static("这里聚合 /human 和目标 Agent 的往返消息，默认目标为 /supervisor。", classes="chat-note")
                    with Vertical(classes="column"):
                        yield Static("发送消息", classes="section-title")
                        yield Input(value="/supervisor", placeholder="对话目标", id="chat-target")
                        yield TextArea(id="chat-text")
                        with Vertical(id="chat-controls"):
                            yield Button("发送到对话", id="send-chat", variant="primary")
                            yield Static(id="chat-status")
            with TabPane("队列", id="queues"):
                with Horizontal(classes="page-grid"):
                    with Vertical(classes="column"):
                        yield Static("五类任务队列", classes="section-title")
                        yield ListView(id="queue-kinds")
                    with Vertical(classes="column"):
                        yield Static("队列内容", classes="section-title")
                        yield ListView(id="queue-items")
                    with Vertical(classes="column"):
                        yield Static("Pending ACK / 详情", classes="section-title")
                        yield TextArea("", id="queue-detail", read_only=True)
        yield Footer()

    async def on_mount(self) -> None:
        self._init_tables()
        await self.load_dashboard()
        self.set_interval(6, self.refresh_snapshot)
        self.run_worker(self.observe_loop(), exclusive=True, group="observe")

    async def on_unmount(self) -> None:
        await self._client.aclose()

    def _init_tables(self) -> None:
        executor_table = self.query_one("#topology-executors", DataTable)
        executor_table.add_columns("路径", "类型", "Owner", "PID", "状态")
        flow_table = self.query_one("#timeline-flow", DataTable)
        flow_table.add_columns("时间", "类型", "来源", "目标", "trace")

    async def refresh_snapshot(self) -> None:
        await self.load_dashboard()

    async def load_dashboard(self) -> None:
        try:
            response = await self._client.get("/api/dashboard", params={"trace_limit": 180})
            response.raise_for_status()
            payload = response.json()
            self.snapshot.tree = payload.get("tree", {"nodes": [], "children": {}, "channels": []})
            self.snapshot.status = payload.get("status", {"processes": [], "sessions": [], "queues": {}})
            self.snapshot.knowledge = payload.get("knowledge", {"documents": []})
            self._merge_traces(payload.get("traces", []))
            if not self.selected_doc_path and self.snapshot.knowledge.get("documents"):
                first_doc = self.snapshot.knowledge["documents"][0]
                self.selected_doc_path = str(first_doc.get("doc_path", ""))
                await self.load_selected_document()
            await self.load_queue_snapshot(refresh=False)
            self.error_text = ""
            self.refresh_views()
        except Exception as exc:
            self.error_text = str(exc)
            self.refresh_views()

    async def load_queue_snapshot(self, refresh: bool = True) -> None:
        try:
            response = await self._client.get("/api/queues", params={"path": self.selected_path})
            response.raise_for_status()
            payload = response.json()
            self.queue_snapshot = dict(payload.get("queues", {}))
            self.pending_acks = list(payload.get("pending_acks", []))
        except Exception as exc:
            self.queue_snapshot = {}
            self.pending_acks = [{"error": str(exc)}]
        if refresh:
            self.refresh_views()

    def _merge_traces(self, traces: list[JsonRecord]) -> None:
        merged = {item.get("trace_entry_id"): item for item in self.snapshot.traces if item.get("trace_entry_id")}
        for item in traces:
            trace_id = item.get("trace_entry_id")
            if trace_id is None:
                continue
            merged[trace_id] = item
        self.snapshot.traces = sorted(merged.values(), key=lambda item: str(item.get("timestamp", "")))[-300:]

    async def observe_loop(self) -> None:
        while True:
            try:
                async with websockets.connect("ws://127.0.0.1:18990/ws/observe") as websocket:
                    self.ws_state = "live"
                    self.refresh_views()
                    async for raw in websocket:
                        payload = json.loads(raw)
                        if payload.get("type") == "history":
                            self._merge_traces(payload.get("entries", []))
                            self.refresh_views()
                            continue
                        if payload.get("type") == "trace":
                            entry = payload.get("entry")
                            if isinstance(entry, dict):
                                self._merge_traces([entry])
                                if str(entry.get("category")) in {"tree_change", "runtime", "knowledge_op", "ownership", "channel"}:
                                    await self.load_dashboard()
                                elif str(entry.get("category")) in {"event_publish", "event_dispatch", "event_ack", "reply_sent", "supervisor_command"}:
                                    await self.load_queue_snapshot(refresh=False)
                                    self.refresh_views()
                                else:
                                    self.refresh_views()
            except Exception as exc:
                self.ws_state = f"offline: {exc}"
                self.refresh_views()
                await asyncio.sleep(2)

    def refresh_views(self) -> None:
        self._entry_index = {}
        self._entry_counter = 0
        self.query_one("#focus-label", Static).update(f"当前聚焦: {self.selected_path}")
        status_text = f"WS: {self.ws_state}"
        if self.error_text:
            status_text += f" | ERROR: {self.error_text}"
        self.query_one("#status-label", Static).update(status_text)

        nodes = self.nodes
        docs = self.knowledge_docs
        executors = [item for item in nodes if str(item.get("kind")) == "executor"]
        self.query_one("#metric-nodes", Static).update(f"节点: {len(nodes)}")
        self.query_one("#metric-docs", Static).update(f"知识: {len(docs)}")
        self.query_one("#metric-executors", Static).update(f"执行器: {len(executors)}")
        self.query_one("#metric-traces", Static).update(f"轨迹: {len(self.snapshot.traces)}")

        self._refresh_overview()
        self._refresh_topology()
        self._refresh_timeline()
        self._refresh_knowledge()
        self._refresh_chat()
        self._refresh_queues()

    @property
    def nodes(self) -> list[JsonRecord]:
        return list(self.snapshot.tree.get("nodes", []))

    @property
    def children(self) -> JsonRecord:
        return dict(self.snapshot.tree.get("children", {}))

    @property
    def node_map(self) -> dict[str, JsonRecord]:
        return {str(item.get("path")): item for item in self.nodes}

    @property
    def process_map(self) -> dict[str, JsonRecord]:
        return {str(item.get("path")): item for item in self.snapshot.status.get("processes", [])}

    @property
    def sessions(self) -> set[str]:
        return {str(item) for item in self.snapshot.status.get("sessions", [])}

    @property
    def queue_sizes(self) -> JsonRecord:
        return dict(self.snapshot.status.get("queues", {}))

    @property
    def knowledge_docs(self) -> list[JsonRecord]:
        return list(self.snapshot.knowledge.get("documents", []))

    def _register_entry(self, entry: DisplayEntry) -> str:
        token = f"entry-{self._entry_counter}"
        self._entry_counter += 1
        self._entry_index[token] = entry
        return token

    def _open_detail(self, title: str, detail: Any) -> None:
        self.push_screen(MessageScreen(title, detail))

    def _set_text_area(self, widget_id: str, value: Any) -> None:
        self.query_one(widget_id, TextArea).text = format_display_text(value)

    def _maybe_decode(self, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if text and text[0] in "[{":
                try:
                    return json.loads(text)
                except Exception:
                    return value
            return value
        if isinstance(value, list):
            return [self._maybe_decode(item) for item in value]
        if isinstance(value, dict):
            return {key: self._maybe_decode(item) for key, item in value.items()}
        return value

    def _brief(self, value: Any, limit: int = 120) -> str:
        text = value if isinstance(value, str) else str(self._maybe_decode(value))
        text = text.replace("\n", " ").strip()
        return text if len(text) <= limit else text[: limit - 3] + "..."

    def _get_text_area_value(self, widget_id: str) -> str:
        return self.query_one(widget_id, TextArea).text.strip()

    def _trace_event(self, item: JsonRecord) -> JsonRecord:
        payload = item.get("payload", {}) or {}
        reply = payload.get("reply")
        event = payload.get("event")
        if isinstance(reply, dict):
            return reply
        if isinstance(event, dict):
            return event
        return {}

    def _trace_text(self, item: JsonRecord) -> str:
        payload = item.get("payload", {}) or {}
        event = self._trace_event(item)
        event_payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        candidates = [
            event_payload.get("text"),
            event_payload.get("summary"),
            payload.get("text"),
            payload.get("thought_summary"),
            item.get("message"),
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        if event_payload:
            return json.dumps(event_payload, ensure_ascii=False)
        if payload:
            return json.dumps(payload, ensure_ascii=False)
        return ""

    def _trace_detail(self, item: JsonRecord) -> JsonRecord:
        payload = item.get("payload", {}) or {}
        detail: JsonRecord = {
            "timestamp": item.get("timestamp"),
            "source": item.get("source"),
            "category": item.get("category"),
            "message": item.get("message"),
            "trace_id": item.get("trace_id"),
            "event_id": item.get("event_id"),
        }
        if payload.get("tool_name"):
            detail["tool"] = {
                "tool_name": payload.get("tool_name"),
                "args": self._maybe_decode(payload.get("args")),
                "result_preview": self._maybe_decode(payload.get("result_preview")),
                "error": payload.get("error"),
                "duration_ms": payload.get("duration_ms"),
            }
        if payload.get("thought_summary"):
            detail["thought_summary"] = payload.get("thought_summary")
        event = self._trace_event(item)
        if event:
            detail["event"] = self._maybe_decode(event)
        elif payload:
            detail["payload"] = self._maybe_decode(payload)
        return detail

    def _trace_summary(self, item: JsonRecord) -> str:
        category = str(item.get("category", "trace"))
        payload = item.get("payload", {}) or {}
        event = self._trace_event(item)
        event_payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if category == "supervisor_command":
            return f"Human -> {event.get('target_path', '/supervisor')} · {self._brief(event_payload.get('text') or payload.get('text'))}"
        if category == "reply_sent":
            return f"{event.get('source_path', item.get('source'))} -> {event.get('target_path')} · {self._brief(event_payload.get('text') or event_payload.get('summary') or item.get('message'))}"
        if category in {"tool_call", "tool_result", "tool_error"}:
            return f"{category} · {payload.get('tool_name')} · {self._brief(payload.get('args') or payload.get('result_preview') or payload.get('error'))}"
        return f"{category} · {item.get('source')} · {self._brief(item.get('message'))}"

    def _conversation_messages(self, target_path: str) -> list[JsonRecord]:
        messages: list[JsonRecord] = []
        for item in self.snapshot.traces:
            category = str(item.get("category", ""))
            event = self._trace_event(item)
            source_path = str(event.get("source_path", item.get("source", "")))
            target = str(event.get("target_path", ""))
            if category == "supervisor_command" and source_path == "/human" and target == target_path:
                messages.append(
                    {
                        "role": "human",
                        "source": "/human",
                        "target": target_path,
                        "text": self._trace_text(item),
                        "trace": item,
                    }
                )
            elif category in {"reply_sent", "event_publish"} and target == "/human" and (source_path == target_path or target_path == "/human/all"):
                messages.append(
                    {
                        "role": "assistant",
                        "source": source_path,
                        "target": "/human",
                        "text": self._trace_text(item),
                        "trace": item,
                    }
                )
        return messages[-80:]

    def _recent_replies_for_target(self, target_path: str) -> list[JsonRecord]:
        replies: list[JsonRecord] = []
        for item in self.snapshot.traces:
            if str(item.get("category", "")) not in {"reply_sent", "event_publish"}:
                continue
            event = self._trace_event(item)
            source_path = str(event.get("source_path", item.get("source", "")))
            target = str(event.get("target_path", ""))
            if source_path == target_path and target == "/human":
                replies.append(item)
        return replies[-20:][::-1]

    def _refresh_overview(self) -> None:
        summary = {
            "selected_path": self.selected_path,
            "ws_state": self.ws_state,
            "node_count": len(self.nodes),
            "knowledge_doc_count": len(self.knowledge_docs),
            "executor_count": len([item for item in self.nodes if str(item.get("kind")) == "executor"]),
            "latest_trace": self._trace_detail(self.snapshot.traces[-1]) if self.snapshot.traces else None,
        }
        self._set_text_area("#overview-summary", summary)
        self._populate_list(
            self.query_one("#overview-traces", ListView),
            [DisplayEntry(self._trace_summary(item), f"轨迹 · {item.get('category')}", self._trace_detail(item)) for item in self.snapshot.traces[-8:][::-1]],
        )
        self._populate_list(
            self.query_one("#overview-docs", ListView),
            [
                DisplayEntry(
                    summary=f"{item.get('doc_path')} · 更新={item.get('updated_at')}",
                    title=f"知识文档 · {item.get('doc_path')}",
                    detail=self._maybe_decode(item),
                    action="load_doc",
                    value=str(item.get("doc_path", "")),
                )
                for item in self.knowledge_docs[-6:][::-1]
            ],
        )

    def _refresh_topology(self) -> None:
        tree = self.query_one("#topology-tree", Tree)
        tree.clear()
        root = tree.root
        root.set_label("/")
        self._build_tree(root, "/")
        tree.root.expand_all()
        node_detail = self.node_map.get(self.selected_path)
        self._set_text_area("#topology-node-detail", {
            "node": node_detail or {"path": self.selected_path, "message": "node not found"},
            "queue_sizes": self.queue_sizes.get(self.selected_path, {}),
        })
        table = self.query_one("#topology-executors", DataTable)
        table.clear(columns=False)
        for item in [node for node in self.nodes if str(node.get("kind")) == "executor"]:
            process = self.process_map.get(str(item.get("path")), {})
            state = "live" if str(item.get("path")) in self.sessions else str(item.get("status", "unknown"))
            row_key = f"executor:{item.get('path')}"
            table.add_row(
                str(item.get("path", "")),
                str(item.get("metadata", {}).get("executor_kind", item.get("kind", ""))),
                str(item.get("owner_path") or "-"),
                str(process.get("pid") or "-"),
                state,
                key=row_key,
            )
            self._row_details[row_key] = (f"执行器 · {item.get('path')}", {"node": self._maybe_decode(item), "process": self._maybe_decode(process)})

    def _build_tree(self, node, path: str) -> None:
        for child_path in self.children.get(path, []):
            child = self.node_map.get(str(child_path), {})
            label = f"{child_path} [{child.get('kind', '?')}|{child.get('status', '?')}]"
            branch = node.add(label, data=str(child_path))
            self._build_tree(branch, str(child_path))

    def _filtered_traces(self) -> list[JsonRecord]:
        if self.selected_path == "/":
            return self.snapshot.traces[-80:][::-1]
        out: list[JsonRecord] = []
        for item in self.snapshot.traces:
            if str(item.get("source")) == self.selected_path:
                out.append(item)
                continue
            payload = item.get("payload", {}) or {}
            target_path = payload.get("target_path")
            owner_node = payload.get("owner_node_path")
            event = payload.get("event") or {}
            reply = payload.get("reply") or {}
            if (
                target_path == self.selected_path
                or owner_node == self.selected_path
                or event.get("source_path") == self.selected_path
                or event.get("target_path") == self.selected_path
                or reply.get("source_path") == self.selected_path
                or reply.get("target_path") == self.selected_path
            ):
                out.append(item)
        return out[-80:][::-1]

    def _refresh_timeline(self) -> None:
        self._populate_list(
            self.query_one("#timeline-traces", ListView),
            [DisplayEntry(self._trace_summary(item), f"轨迹 · {item.get('category')}", self._trace_detail(item)) for item in self._filtered_traces()],
        )
        flow_table = self.query_one("#timeline-flow", DataTable)
        flow_table.clear(columns=False)
        flow_entries = []
        for item in self.snapshot.traces:
            category = str(item.get("category", ""))
            if category not in {"event_publish", "event_received", "reply_sent"}:
                continue
            payload = item.get("payload", {}) or {}
            event = payload.get("event") or payload.get("reply") or {}
            flow_entries.append(
                (
                    str(item.get("timestamp", "")),
                    str(event.get("kind", category)),
                    str(event.get("source_path", item.get("source", ""))),
                    str(event.get("target_path", payload.get("target_path", "-"))),
                    str(event.get("trace_id", item.get("trace_id", "-")))[:8],
                )
            )
        for index, row in enumerate(flow_entries[-18:][::-1]):
            row_key = f"flow:{index}"
            flow_table.add_row(*row, key=row_key)
            self._row_details[row_key] = ("事件流详情", {"row": row})

    def _refresh_knowledge(self) -> None:
        related_docs = self.knowledge_docs if self.selected_path == "/" else [
            item
            for item in self.knowledge_docs
            if str(item.get("doc_path", "")).startswith(self.selected_path) or str(item.get("owner_node_path", "")) == self.selected_path
        ]
        self._populate_list(
            self.query_one("#knowledge-docs", ListView),
            [
                DisplayEntry(
                    summary=f"{item.get('doc_path')} · owner={item.get('owner_node_path') or '-'}",
                    title=f"知识文档 · {item.get('doc_path')}",
                    detail=self._maybe_decode(item),
                    action="load_doc",
                    value=str(item.get("doc_path", "")),
                )
                for item in related_docs
            ],
        )
        self._set_text_area("#knowledge-content", {"doc_path": self.selected_doc_path, "content": self.doc_content or "请选择文档。"})
        self._populate_list(
            self.query_one("#knowledge-results", ListView),
            [
                DisplayEntry(
                    summary=f"{item.get('doc_path')} · 分数={round(float(item.get('score', 0.0)), 4)} · {self._brief(item.get('text'))}",
                    title=f"检索结果 · {item.get('doc_path')}",
                    detail=self._maybe_decode(item),
                )
                for item in self.knowledge_results
            ],
        )

    def _refresh_queues(self) -> None:
        self._populate_list(
            self.query_one("#queue-kinds", ListView),
            [
                DisplayEntry(f"{kind.upper()} · {len(self.queue_snapshot.get(kind, []))} 条", f"队列 · {kind}", self._maybe_decode(self.queue_snapshot.get(kind, [])), action="select_queue_kind", value=kind)
                for kind in ("command", "message", "struct", "emergency", "event")
            ],
        )
        current_items = list(self.queue_snapshot.get(self.selected_queue_kind, []))
        self._populate_list(
            self.query_one("#queue-items", ListView),
            [DisplayEntry(f"#{index + 1} · {self._brief((item.get('event', {}) or {}).get('payload', {}) or item)}", f"队列项 · {self.selected_queue_kind} #{index + 1}", self._maybe_decode(item)) for index, item in enumerate(current_items)],
        )
        self._set_text_area("#queue-detail", {"path": self.selected_path, "selected_queue_kind": self.selected_queue_kind, "pending_acks": self._maybe_decode(self.pending_acks)})

    def _refresh_chat(self) -> None:
        target = self.query_one("#chat-target", Input).value.strip() or "/supervisor"
        messages = self._conversation_messages(target)
        self._populate_list(
            self.query_one("#chat-feed", ListView),
            [
                DisplayEntry(
                    summary=self._conversation_label(item),
                    title=f"对话 · {item.get('source')} -> {item.get('target')}",
                    detail={"text": item.get("text"), "trace": self._trace_detail(item.get("trace", {}))},
                )
                for item in messages
            ],
        )
        live_status, live_detail = self._chat_live_status(target)
        self.query_one("#chat-live-status", Static).update(live_status)
        detail = messages[-1] if messages else {"target": target, "message": "暂无对话记录"}
        self._set_text_area("#chat-detail", {"latest": detail, "live": live_detail})
        self.query_one("#chat-status", Static).update(self.chat_state)

    def _chat_live_status(self, target_path: str) -> tuple[str, JsonRecord]:
        latest_command = None
        thinking_events = []
        tool_events = []
        reply_events = []
        for item in self.snapshot.traces:
            category = str(item.get("category", ""))
            event = self._trace_event(item)
            source_path = str(event.get("source_path", item.get("source", "")))
            target = str(event.get("target_path", ""))
            if category == "supervisor_command" and source_path == "/human" and target == target_path:
                latest_command = item
            elif category == "thinking" and str(item.get("source", "")) == target_path:
                thinking_events.append(item)
            elif category in {"tool_call", "tool_result", "tool_error", "executor_invoke"} and str(item.get("source", "")) == target_path:
                tool_events.append(item)
            elif category in {"reply_sent", "event_publish"} and source_path == target_path and target == "/human":
                reply_events.append(item)
        if latest_command is None:
            return "等待对话...", {"status": "idle", "target": target_path}
        latest_trace_id = latest_command.get("trace_id")
        matched_replies = [item for item in reply_events if item.get("trace_id") == latest_trace_id]
        matched_thinking = [item for item in thinking_events if item.get("trace_id") == latest_trace_id]
        matched_tools = [item for item in tool_events if item.get("trace_id") == latest_trace_id]
        if matched_replies:
            return "回复已完成", {"thoughts": [self._trace_detail(item) for item in matched_thinking[-3:]], "tools": [self._trace_detail(item) for item in matched_tools[-6:]], "reply": self._trace_detail(matched_replies[-1])}
        if matched_tools:
            return "正在调用工具/执行器", {"thoughts": [self._trace_detail(item) for item in matched_thinking[-3:]], "tools": [self._trace_detail(item) for item in matched_tools[-6:]]}
        if matched_thinking:
            return "正在思考", {"thoughts": [self._trace_detail(item) for item in matched_thinking[-3:]]}
        return "等待回复", {"command": self._trace_detail(latest_command)}

    def _populate_list(self, list_view: ListView, rows: list[str] | list[DisplayEntry]) -> None:
        list_view.clear()
        if not rows:
            list_view.append(ListItem(Label("暂无数据"), name=""))
            return
        for row in rows:
            if isinstance(row, DisplayEntry):
                token = self._register_entry(row)
                list_view.append(ListItem(Label(row.summary), name=token))
                continue
            list_view.append(ListItem(Label(row), name=row))

    def _trace_label(self, item: JsonRecord) -> str:
        payload = item.get("payload", {}) or {}
        thought = payload.get("thought_summary")
        suffix = f" | {thought}" if isinstance(thought, str) and thought else ""
        return f"[{item.get('category')}] {item.get('source')} :: {item.get('message')}{suffix}"

    def _reply_label(self, item: JsonRecord) -> str:
        event = self._trace_event(item)
        source = str(event.get("source_path", item.get("source", "")))
        target = str(event.get("target_path", ""))
        text = self._trace_text(item).replace("\n", " ")
        if len(text) > 120:
            text = text[:117] + "..."
        return f"{source} -> {target} | {text}"

    def _conversation_label(self, item: JsonRecord) -> str:
        prefix = "You" if item.get("role") == "human" else str(item.get("source") or "Agent")
        text = str(item.get("text", "")).replace("\n", " ")
        if len(text) > 140:
            text = text[:137] + "..."
        return f"{prefix}: {text}"

    async def load_selected_document(self) -> None:
        if not self.selected_doc_path:
            self.doc_content = ""
            self.refresh_views()
            return
        try:
            response = await self._client.get("/api/knowledge/content", params={"doc_path": self.selected_doc_path})
            response.raise_for_status()
            payload = response.json()
            self.doc_content = str(payload.get("text", ""))
        except Exception as exc:
            self.doc_content = f"文档暂不可读: {exc}"
        self.refresh_views()

    async def action_refresh(self) -> None:
        await self.load_dashboard()

    async def action_switch_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id

    async def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        data = event.node.data
        if isinstance(data, str):
            self.selected_path = data
            await self.load_queue_snapshot(refresh=False)
            self.refresh_views()

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        token = event.item.name or ""
        if not token:
            return
        entry = self._entry_index.get(token)
        if entry is not None:
            if entry.action == "select_queue_kind" and entry.value:
                self.selected_queue_kind = entry.value
                self.refresh_views()
                return
            if entry.action == "load_doc" and entry.value:
                self.selected_doc_path = entry.value
                await self.load_selected_document()
                self.query_one(TabbedContent).active = "knowledge"
            self._open_detail(entry.title, entry.detail)
            return
        if event.list_view.id in {"knowledge-docs", "overview-docs"}:
            self.selected_doc_path = token.split(" | ", 1)[0]
            self.query_one(TabbedContent).active = "knowledge"
            await self.load_selected_document()
            self._open_detail("知识文档", {"doc_path": self.selected_doc_path, "content": self.doc_content})
            return
        self._open_detail("详细内容", token)

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_key = str(event.row_key.value)
        detail = self._row_details.get(row_key)
        if detail is not None:
            self._open_detail(detail[0], detail[1])

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "run-knowledge-query":
            await self.run_knowledge_query()
        elif button_id == "send-command":
            await self.send_command()
        elif button_id == "send-chat":
            await self.send_chat_message()

    async def run_knowledge_query(self) -> None:
        query = self._get_text_area_value("#knowledge-query")
        if not query:
            self.knowledge_results = []
            self.refresh_views()
            return
        try:
            response = await self._client.get(
                "/api/knowledge/query",
                params={"node_path": self.selected_path, "scope": self.selected_path, "query": query},
            )
            response.raise_for_status()
            payload = response.json()
            self.knowledge_results = list(payload.get("hits", []))
            self.error_text = ""
        except Exception as exc:
            self.error_text = str(exc)
        self.refresh_views()

    async def send_command(self) -> None:
        target = self.query_one("#command-target", Input).value.strip() or "/supervisor"
        text = self._get_text_area_value("#command-text")
        if not text:
            self.command_state = "请输入命令内容。"
            self.refresh_views()
            return
        self.command_state = "发送中..."
        self.refresh_views()
        try:
            response = await self._client.post(
                "/api/supervisor/command",
                json={"target_path": target, "text": text},
            )
            response.raise_for_status()
            self.command_state = "命令已发送，等待轨迹回流。"
            self.selected_path = target
            self.query_one("#command-text", TextArea).text = ""
            await self.load_dashboard()
        except Exception as exc:
            self.command_state = str(exc)
            self.refresh_views()

    async def send_chat_message(self) -> None:
        target = self.query_one("#chat-target", Input).value.strip() or "/supervisor"
        text = self._get_text_area_value("#chat-text")
        if not text:
            self.chat_state = "请输入消息内容。"
            self.refresh_views()
            return
        self.chat_state = "发送中..."
        self.refresh_views()
        try:
            response = await self._client.post(
                "/api/supervisor/command",
                json={"target_path": target, "text": text},
            )
            response.raise_for_status()
            self.chat_state = f"消息已发送到 {target}，等待回复。"
            self.query_one("#chat-text", TextArea).text = ""
            await self.load_dashboard()
        except Exception as exc:
            self.chat_state = str(exc)
            self.refresh_views()

    async def on_key(self, event: events.Key) -> None:
        if event.key == "enter" and self.focused and self.focused.id == "command-text":
            return
