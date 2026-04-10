from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from agenttree.config import get_settings
from agenttree.const_prompt import (
    get_knowledge_template,
    get_prompt_template,
    list_knowledge_templates,
    list_prompt_templates,
)
from agenttree.core.state import CoreState, build_core_state
from agenttree.schemas.config import ModelConfigRequest, ModelConfigResponse
from agenttree.schemas.events import EventEnvelope, EventKind, PublishEventRequest, SendMessageRequest
from agenttree.schemas.nodes import (
    BindExecutorRequest,
    ChannelRecord,
    CreateAgentRequest,
    CreateExecutorRequest,
    MoveNodeRequest,
    NodeKind,
    NodeStatus,
    RegisterExecutorRequest,
    RemoveTriggerRequest,
    TransferExecutorRequest,
    UpdateNodeRequest,
    UpsertTriggerRequest,
)
from agenttree.schemas.protocol import RuntimeHello, RuntimeMessage, RuntimeMessageType
from agenttree.schemas.prompt import (
    ExportKnowledgeTemplatesRequest,
    ExportPromptsRequest,
    KnowledgeTemplateRecord,
    PromptTemplateRecord,
)
from agenttree.schemas.trace import TraceEntry


SYSTEM_KNOWLEDGE_OWNER_PATH = "/supervisor"


async def _bootstrap_system_knowledge_templates(state: CoreState) -> list[dict]:
    injected: list[dict] = []
    for item in list_knowledge_templates():
        doc_path = (item.default_doc_path or f"/library/{item.name}.md").strip()
        if not doc_path.startswith("/"):
            doc_path = f"/{doc_path.lstrip('/')}"
        record = await state.knowledge_store.upsert_document(
            owner_node_path=SYSTEM_KNOWLEDGE_OWNER_PATH,
            doc_path=doc_path,
            text=item.text,
        )
        injected.append({"knowledge_name": item.name, "doc_path": doc_path, "document": record})
    if injected:
        await state.trace_store.record(
            TraceEntry(
                source=SYSTEM_KNOWLEDGE_OWNER_PATH,
                category="knowledge_bootstrap",
                message=f"bootstrapped {len(injected)} system knowledge template(s)",
                payload={"injected": injected},
            )
        )
    return injected


def create_app(settings=None) -> FastAPI:
    settings = settings or get_settings()
    state = build_core_state(settings)
    ui_dist = Path(__file__).resolve().parents[2] / "ui" / "dist"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.auto_inject_system_knowledge_templates:
            await _bootstrap_system_knowledge_templates(state)
        if settings.auto_start_supervisor:
            supervisor = state.registry.upsert_runtime_node(
                path="/supervisor",
                kind=NodeKind.SUPERVISOR,
                capabilities=["admin", "query", "spawn"],
                metadata={"system": True},
            )
            state.process_manager.ensure_process(supervisor)
        try:
            yield
        finally:
            state.process_manager.shutdown()

    app = FastAPI(title="AgentTree Core", version="0.1.0", lifespan=lifespan)
    app.state.core = state
    if ui_dist.exists():
        app.mount("/dashboard", StaticFiles(directory=str(ui_dist), html=True), name="dashboard")

    async def trace(source: str, category: str, message: str, payload: dict | None = None) -> None:
        trace_payload = payload or {}
        event = trace_payload.get("event") or trace_payload.get("reply")
        trace_id = None
        event_id = None
        parent_event_id = None
        if isinstance(event, dict):
            trace_id = event.get("trace_id")
            event_id = event.get("event_id")
            metadata = event.get("metadata", {}) or {}
            if isinstance(metadata, dict):
                parent_event_id = metadata.get("reply_to")
        await state.trace_store.record(
            TraceEntry(
                source=source,
                category=category,
                message=message,
                trace_id=trace_payload.get("trace_id") or trace_id,
                event_id=trace_payload.get("event_id") or event_id,
                parent_event_id=trace_payload.get("parent_event_id") or parent_event_id,
                payload=trace_payload,
            )
        )

    def render_tree() -> str:
        snapshot = state.registry.snapshot()
        lines: list[str] = []

        def walk(path: str, prefix: str = "") -> None:
            node = snapshot.nodes[path]
            lines.append(f"{prefix}{node.path} [{node.kind.value}|{node.status.value}]")
            for child in snapshot.children.get(path, []):
                walk(child, prefix + "  ")

        walk("/")
        return "\n".join(lines)

    @app.get("/")
    async def root() -> dict:
        return {
            "status": "ok",
            "runtime_ws": settings.runtime_ws_url,
            "nodes": len(state.registry.list_nodes()),
            "dashboard_url": "/dashboard" if ui_dist.exists() else None,
        }

    @app.get(f"{settings.api_prefix}/status")
    async def status() -> dict:
        return {
            "status": "ok",
            "processes": state.process_manager.list_processes(),
            "sessions": await state.ws_hub.list_paths(),
        }

    @app.get(f"{settings.api_prefix}/dashboard")
    async def dashboard_snapshot(trace_limit: int = 120, knowledge_scope: str = "/") -> dict:
        snapshot = state.registry.snapshot()
        entries = await state.trace_store.list_entries(limit=trace_limit)
        queue_sizes = {
            path: await state.broker.queue_sizes(path)
            for path in snapshot.nodes.keys()
        }
        return {
            "tree": {
                "nodes": [node.model_dump(mode="json") for node in snapshot.nodes.values()],
                "children": snapshot.children,
                "channels": [channel.model_dump(mode="json") for channel in snapshot.channels.values()],
            },
            "status": {
                "status": "ok",
                "processes": state.process_manager.list_processes(),
                "sessions": await state.ws_hub.list_paths(),
                "queues": queue_sizes,
            },
            "knowledge": {"documents": state.knowledge_store.list_documents(knowledge_scope)},
            "traces": [item.model_dump(mode="json") for item in entries],
        }

    @app.get(f"{settings.api_prefix}/queues")
    async def queue_snapshot(path: str | None = None) -> dict:
        if path:
            return {
                "path": path,
                "queues": await state.broker.queue_snapshot(path),
                "pending_acks": await state.broker.pending_ack_snapshot(path),
            }
        return {"queues": await state.broker.queue_snapshot_for_all()}

    @app.get(f"{settings.api_prefix}/model-config", response_model=ModelConfigResponse)
    async def get_model_config() -> ModelConfigResponse:
        return ModelConfigResponse(
            model=settings.model,
            openai_api_key_configured=bool(settings.openai_api_key),
            openai_base_url=settings.openai_base_url,
            model_temperature=settings.model_temperature,
        )

    @app.put(f"{settings.api_prefix}/model-config", response_model=ModelConfigResponse)
    async def update_model_config(request: ModelConfigRequest) -> ModelConfigResponse:
        settings.update_model_runtime_config(
            model=request.model,
            openai_api_key=request.openai_api_key,
            openai_base_url=request.openai_base_url,
            model_temperature=request.model_temperature,
        )
        await trace(
            "/supervisor",
            "model_config",
            "updated model runtime config",
            {
                "model": settings.model,
                "openai_api_key_configured": bool(settings.openai_api_key),
                "openai_base_url": settings.openai_base_url,
                "model_temperature": settings.model_temperature,
            },
        )
        return ModelConfigResponse(
            model=settings.model,
            openai_api_key_configured=bool(settings.openai_api_key),
            openai_base_url=settings.openai_base_url,
            model_temperature=settings.model_temperature,
        )

    @app.get(f"{settings.api_prefix}/tree")
    async def list_tree() -> dict:
        snapshot = state.registry.snapshot()
        return {
            "nodes": [node.model_dump(mode="json") for node in snapshot.nodes.values()],
            "children": snapshot.children,
            "channels": [channel.model_dump(mode="json") for channel in snapshot.channels.values()],
        }

    @app.get(f"{settings.api_prefix}/prompts")
    async def list_prompts() -> dict:
        items = [
            PromptTemplateRecord(
                name=item.name,
                description=item.description,
                prompt=item.prompt,
                default_knowledge_doc_path=item.default_knowledge_doc_path,
                tags=list(item.tags),
            ).model_dump(mode="json")
            for item in list_prompt_templates()
        ]
        return {"prompts": items}

    @app.get(f"{settings.api_prefix}/prompts/{{name}}")
    async def get_prompt(name: str) -> dict:
        try:
            item = get_prompt_template(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        record = PromptTemplateRecord(
            name=item.name,
            description=item.description,
            prompt=item.prompt,
            default_knowledge_doc_path=item.default_knowledge_doc_path,
            tags=list(item.tags),
        )
        return {"prompt": record.model_dump(mode="json")}

    @app.get(f"{settings.api_prefix}/knowledge-templates")
    async def list_knowledge_template_items() -> dict:
        items = [
            KnowledgeTemplateRecord(
                name=item.name,
                description=item.description,
                text=item.text,
                default_doc_path=item.default_doc_path,
                tags=list(item.tags),
            ).model_dump(mode="json")
            for item in list_knowledge_templates()
        ]
        return {"knowledge_templates": items}

    @app.get(f"{settings.api_prefix}/knowledge-templates/{{name}}")
    async def get_knowledge_template_item(name: str) -> dict:
        try:
            item = get_knowledge_template(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        record = KnowledgeTemplateRecord(
            name=item.name,
            description=item.description,
            text=item.text,
            default_doc_path=item.default_doc_path,
            tags=list(item.tags),
        )
        return {"knowledge_template": record.model_dump(mode="json")}

    @app.post(f"{settings.api_prefix}/prompts/export-to-knowledge")
    async def export_prompts_to_knowledge(request: ExportPromptsRequest) -> dict:
        exported = []
        for prompt_name in request.prompt_names:
            try:
                item = get_prompt_template(prompt_name)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            doc_path = f"{request.target_root_path.rstrip('/')}/{item.name}.md"
            record = await state.knowledge_store.upsert_document(
                owner_node_path=request.owner_node_path,
                doc_path=doc_path,
                text=item.prompt,
            )
            exported.append({"prompt_name": item.name, "doc_path": doc_path, "document": record})
        await trace(
            request.owner_node_path,
            "prompt_export",
            f"exported {len(exported)} prompt template(s) to knowledge",
            {"exported": exported},
        )
        return {"exported": exported}

    @app.post(f"{settings.api_prefix}/knowledge-templates/export-to-knowledge")
    async def export_manual_knowledge_to_knowledge(request: ExportKnowledgeTemplatesRequest) -> dict:
        exported = []
        for knowledge_name in request.knowledge_names:
            try:
                item = get_knowledge_template(knowledge_name)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            doc_path = f"{request.target_root_path.rstrip('/')}/{item.name}.md"
            record = await state.knowledge_store.upsert_document(
                owner_node_path=request.owner_node_path,
                doc_path=doc_path,
                text=item.text,
            )
            exported.append({"knowledge_name": item.name, "doc_path": doc_path, "document": record})
        await trace(
            request.owner_node_path,
            "knowledge_template_export",
            f"exported {len(exported)} manual knowledge template(s) to knowledge",
            {"exported": exported},
        )
        return {"exported": exported}

    @app.get(f"{settings.api_prefix}/nodes/{{path:path}}")
    async def get_node(path: str) -> dict:
        normalized = "/" + path.strip("/") if path.strip("/") else "/"
        try:
            node = state.registry.get_node(normalized)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"node": node.model_dump(mode="json")}

    @app.patch(f"{settings.api_prefix}/nodes/{{path:path}}")
    async def update_node(path: str, request: UpdateNodeRequest) -> dict:
        normalized = "/" + path.strip("/") if path.strip("/") else "/"
        try:
            node = state.registry.update_node(normalized, request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"node": node.model_dump(mode="json")}

    @app.post(f"{settings.api_prefix}/agents")
    async def create_agent(request: CreateAgentRequest) -> dict:
        resolved_prompt = request.prompt
        if not resolved_prompt and request.prompt_template:
            try:
                resolved_prompt = get_prompt_template(request.prompt_template).prompt
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not resolved_prompt:
            raise HTTPException(status_code=400, detail="prompt or prompt_template is required")
        request.prompt = resolved_prompt
        try:
            node = state.registry.create_agent(request)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        struct_event = EventEnvelope(
            kind=EventKind.STRUCT,
            source_path="/supervisor",
            target_path=request.parent_path,
            payload={"action": "child_created", "path": node.path, "kind": node.kind.value},
        )
        await state.broker.publish(struct_event)
        state.process_manager.ensure_process(node)
        exported = []
        if request.knowledge_seed_templates:
            seed_root = request.knowledge_seed_root or f"{node.path}/prompts"
            for prompt_name in request.knowledge_seed_templates:
                try:
                    item = get_prompt_template(prompt_name)
                except KeyError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                doc_path = f"{seed_root.rstrip('/')}/{item.name}.md"
                record = await state.knowledge_store.upsert_document(
                    owner_node_path=node.path,
                    doc_path=doc_path,
                    text=item.prompt,
                )
                exported.append({"prompt_name": item.name, "doc_path": doc_path, "document": record})
        manual_seeded = []
        if request.knowledge_seed_manual_entries:
            seed_root = request.knowledge_seed_root or f"{node.path}/library"
            for knowledge_name in request.knowledge_seed_manual_entries:
                try:
                    item = get_knowledge_template(knowledge_name)
                except KeyError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                doc_path = f"{seed_root.rstrip('/')}/{item.name}.md"
                record = await state.knowledge_store.upsert_document(
                    owner_node_path=node.path,
                    doc_path=doc_path,
                    text=item.text,
                )
                manual_seeded.append({"knowledge_name": item.name, "doc_path": doc_path, "document": record})
        await trace("/supervisor", "tree_change", f"created agent {node.path}", {"tree": render_tree()})
        return {
            "node": node.model_dump(mode="json"),
            "knowledge_seeded_prompts": exported,
            "knowledge_seeded_manual_entries": manual_seeded,
        }

    @app.post(f"{settings.api_prefix}/executors")
    async def create_executor(request: CreateExecutorRequest) -> dict:
        try:
            node = state.registry.create_executor(request)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await state.broker.publish(
            EventEnvelope(
                kind=EventKind.STRUCT,
                source_path="/supervisor",
                target_path=request.owner_path,
                payload={"action": "executor_created", "path": node.path},
            )
        )
        await trace("/supervisor", "tree_change", f"created executor {node.path}", {"tree": render_tree()})
        return {"node": node.model_dump(mode="json")}

    @app.post(f"{settings.api_prefix}/executors/register")
    async def register_executor(request: RegisterExecutorRequest) -> dict:
        try:
            node = state.registry.register_executor(request)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await _notify_root_agents_of_executor_registration(state, node)
        await trace(
            node.path,
            "executor_registration",
            f"registered external executor {node.path}",
            {"node": node.model_dump(mode="json")},
        )
        return {"node": node.model_dump(mode="json")}

    @app.post(f"{settings.api_prefix}/executors/bind")
    async def bind_executor(request: BindExecutorRequest) -> dict:
        try:
            node = state.registry.bind_executor(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        owner_event = EventEnvelope(
            kind=EventKind.STRUCT,
            source_path="/supervisor",
            target_path=request.owner_path,
            payload={
                "action": "executor_bound",
                "path": node.path,
                "owner_path": node.owner_path,
                "executor_kind": node.metadata.get("executor_kind"),
            },
        )
        await state.broker.publish(owner_event)
        await _wake_target(state, request.owner_path)

        executor_event = EventEnvelope(
            kind=EventKind.STRUCT,
            source_path="/supervisor",
            target_path=node.path,
            payload={
                "action": "executor_bound",
                "path": node.path,
                "owner_path": node.owner_path,
                "executor_kind": node.metadata.get("executor_kind"),
            },
        )
        await state.broker.publish(executor_event)
        await _wake_target(state, node.path)

        await trace("/supervisor", "ownership", f"bound executor {node.path} to {request.owner_path}", {"node": node.model_dump(mode="json")})
        return {"node": node.model_dump(mode="json")}

    @app.post(f"{settings.api_prefix}/executors/transfer")
    async def transfer_executor(request: TransferExecutorRequest) -> dict:
        try:
            node = state.registry.transfer_executor(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await state.broker.publish(
            EventEnvelope(
                kind=EventKind.STRUCT,
                source_path="/supervisor",
                target_path=node.owner_path or "/",
                payload={"action": "executor_transferred", "path": node.path},
            )
        )
        await trace("/supervisor", "ownership", f"transferred executor {node.path}", {"node": node.model_dump(mode="json")})
        return {"node": node.model_dump(mode="json")}

    @app.delete(f"{settings.api_prefix}/nodes/{{path:path}}")
    async def delete_node(path: str) -> dict:
        normalized = "/" + path.strip("/") if path.strip("/") else "/"
        try:
            removed = state.registry.delete_node(normalized)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        removed_docs = state.knowledge_sync.handle_node_delete(normalized)
        for item in removed:
            state.process_manager.stop_process(item)
        await trace("/supervisor", "tree_change", f"deleted subtree {normalized}", {"removed": removed, "tree": render_tree()})
        return {"removed": removed, "removed_documents": removed_docs}

    @app.post(f"{settings.api_prefix}/nodes/move")
    async def move_node(request: MoveNodeRequest) -> dict:
        try:
            moved = state.registry.move_node(request)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        moved_documents = state.knowledge_sync.handle_node_move(request.path, request.new_path)
        for old_path, new_record in moved:
            state.process_manager.stop_process(old_path)
            if new_record.kind != NodeKind.ROOT:
                state.process_manager.ensure_process(new_record)
        await state.broker.publish(
            EventEnvelope(
                kind=EventKind.STRUCT,
                source_path="/supervisor",
                target_path=request.new_parent_path,
                payload={
                    "action": "node_moved",
                    "old_path": request.path,
                    "new_path": request.new_path,
                    "moved_documents": moved_documents,
                },
            )
        )
        await _wake_target(state, request.new_parent_path)
        await trace(
            "/supervisor",
            "tree_change",
            f"moved node {request.path} -> {request.new_path}",
            {"moved_documents": moved_documents, "tree": render_tree()},
        )
        return {
            "moved": [{"old_path": old_path, "new_node": record.model_dump(mode="json")} for old_path, record in moved],
            "moved_documents": moved_documents,
        }

    @app.post(f"{settings.api_prefix}/messages")
    async def send_message(request: SendMessageRequest) -> dict:
        event = request.to_event()
        await state.broker.publish(event)
        await _wake_target(state, event.target_path)
        await trace(event.source_path, "event_publish", f"published {event.kind.value} to {event.target_path}", {"event": event.model_dump(mode="json")})
        return {"event": event.model_dump(mode="json")}

    @app.post(f"{settings.api_prefix}/events")
    async def publish_event(request: PublishEventRequest) -> dict:
        await state.broker.publish(request.event)
        await _wake_target(state, request.event.target_path)
        await trace(request.event.source_path, "event_publish", f"published {request.event.kind.value} to {request.event.target_path}", {"event": request.event.model_dump(mode="json")})
        return {"event": request.event.model_dump(mode="json")}

    @app.post(f"{settings.api_prefix}/channels")
    async def create_channel(channel: ChannelRecord) -> dict:
        try:
            record = state.registry.create_channel(channel.channel_id, channel.members, channel.metadata)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await trace("/supervisor", "channel", f"created channel {channel.channel_id}", {"channel": record.model_dump(mode="json")})
        return {"channel": record.model_dump(mode="json")}

    @app.get(f"{settings.api_prefix}/channels")
    async def list_channels() -> dict:
        return {"channels": [channel.model_dump(mode="json") for channel in state.registry.list_channels()]}

    @app.post(f"{settings.api_prefix}/channels/{{channel_id}}/broadcast")
    async def broadcast_channel(channel_id: str, payload: dict) -> dict:
        try:
            channel = state.registry.get_channel(channel_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        source_path = str(payload.get("source_path", "/supervisor"))
        text = str(payload.get("text", ""))
        published = []
        for member in channel.members:
            if member == source_path:
                continue
            event = EventEnvelope(
                kind=EventKind.MESSAGE,
                source_path=source_path,
                target_path=member,
                payload={"text": text},
                channel_id=channel_id,
            )
            await state.broker.publish(event)
            await _wake_target(state, member)
            published.append(event.model_dump(mode="json"))
        await trace(source_path, "channel_broadcast", f"broadcast on {channel_id}", {"events": published})
        return {"events": published}

    @app.post(f"{settings.api_prefix}/triggers/upsert")
    async def upsert_trigger(request: UpsertTriggerRequest) -> dict:
        try:
            node = state.registry.upsert_trigger(request.path, request.trigger)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await trace(request.path, "trigger", f"upserted trigger {request.trigger.trigger_id}", {"node": node.model_dump(mode="json")})
        return {"node": node.model_dump(mode="json")}

    @app.delete(f"{settings.api_prefix}/triggers")
    async def remove_trigger(request: RemoveTriggerRequest) -> dict:
        try:
            node = state.registry.remove_trigger(request.path, request.trigger_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await trace(request.path, "trigger", f"removed trigger {request.trigger_id}", {"node": node.model_dump(mode="json")})
        return {"node": node.model_dump(mode="json")}

    @app.post(f"{settings.api_prefix}/executors/invoke")
    async def invoke_executor(payload: dict) -> dict:
        executor_path = str(payload.get("executor_path", ""))
        source_path = str(payload.get("source_path", "/supervisor"))
        command = str(payload.get("command", "execute"))
        command_payload = dict(payload.get("payload", {}))
        event = EventEnvelope(
            kind=EventKind.COMMAND,
            source_path=source_path,
            target_path=executor_path,
            payload={"command": command, **command_payload},
        )
        await state.broker.publish(event)
        await _wake_target(state, executor_path)
        await trace(source_path, "executor_invoke", f"invoke executor {executor_path}", {"event": event.model_dump(mode="json")})
        return {"event": event.model_dump(mode="json")}

    @app.post(f"{settings.api_prefix}/supervisor/command")
    async def supervisor_command(payload: dict) -> dict:
        target_path = str(payload.get("target_path", "/supervisor"))
        text = str(payload.get("text", ""))
        event = EventEnvelope(
            kind=EventKind.COMMAND,
            source_path="/human",
            target_path=target_path,
            payload={"text": text},
        )
        await state.broker.publish(event)
        await _wake_target(state, target_path)
        await trace("/human", "supervisor_command", f"issued command to {target_path}", {"event": event.model_dump(mode="json")})
        return {"event": event.model_dump(mode="json")}

    @app.get(f"{settings.api_prefix}/traces")
    async def list_traces(limit: int = 100) -> dict:
        entries = await state.trace_store.list_entries(limit=limit)
        return {"entries": [item.model_dump(mode="json") for item in entries]}

    @app.websocket("/ws/observe")
    async def observe_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = await state.trace_store.subscribe()
        history = await state.trace_store.list_entries(limit=30)
        try:
            await websocket.send_json(
                {
                    "type": "history",
                    "entries": [item.model_dump(mode="json") for item in history],
                }
            )
            while True:
                entry = await queue.get()
                await websocket.send_json({"type": "trace", "entry": entry.model_dump(mode="json")})
        except WebSocketDisconnect:
            pass
        finally:
            await state.trace_store.unsubscribe(queue)

    @app.get(f"{settings.api_prefix}/knowledge")
    async def list_knowledge(scope: str = "/") -> dict:
        return {"documents": state.knowledge_store.list_documents(scope)}

    @app.get(f"{settings.api_prefix}/knowledge/content")
    async def read_knowledge(doc_path: str) -> dict:
        try:
            text = state.knowledge_store.read_document(doc_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"doc_path": doc_path, "text": text}

    @app.post(f"{settings.api_prefix}/knowledge/upsert")
    async def upsert_knowledge(payload: dict) -> dict:
        owner_node_path = str(payload.get("owner_node_path", ""))
        doc_path = str(payload.get("doc_path", ""))
        text = str(payload.get("text", ""))
        record = await state.knowledge_store.upsert_document(
            owner_node_path=owner_node_path,
            doc_path=doc_path,
            text=text,
        )
        await trace(
            owner_node_path or "/supervisor",
            "knowledge_op",
            f"upserted knowledge {record['doc_path']}",
            {
                "operation": "upsert",
                "document": record,
                "doc_path": record["doc_path"],
                "owner_node_path": record["owner_node_path"],
            },
        )
        return {"document": record}

    @app.get(f"{settings.api_prefix}/knowledge/query")
    async def query_knowledge(node_path: str, scope: str, query: str, top_k: int = 8) -> dict:
        hits = await state.knowledge_store.query(node_path=node_path, scope=scope, query=query, top_k=top_k)
        return {
            "hits": [
                {
                    "doc_path": hit.doc_path,
                    "score": hit.score,
                    "text": hit.text,
                    "owner_node_path": hit.owner_node_path,
                }
                for hit in hits
            ]
        }

    @app.websocket(settings.ws_path)
    async def runtime_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        registered_path: str | None = None
        try:
            while True:
                raw = await websocket.receive_text()
                message = RuntimeMessage.model_validate_json(raw)
                if message.message_type == RuntimeMessageType.HELLO:
                    hello_payload = dict(message.payload)
                    hello_payload.setdefault("message_type", RuntimeMessageType.HELLO)
                    if message.path is not None:
                        hello_payload.setdefault("path", message.path)
                    hello = RuntimeHello.model_validate(hello_payload)
                    registered_path = hello.path
                    node = state.registry.upsert_runtime_node(
                        path=hello.path,
                        kind=hello.kind,
                        capabilities=hello.capabilities,
                        metadata=hello.metadata,
                    )
                    await state.ws_hub.connect(hello.path, websocket)
                    await trace(hello.path, "runtime", f"runtime registered as {hello.kind.value}", {"tree": render_tree()})
                    await websocket.send_text(
                        RuntimeMessage(
                            message_type=RuntimeMessageType.LOG,
                            path=hello.path,
                            payload={"text": f"registered {node.path}"},
                        ).model_dump_json()
                    )
                    continue

                if registered_path is None:
                    await websocket.send_text(
                        RuntimeMessage(
                            message_type=RuntimeMessageType.ERROR,
                            error="hello required before other messages",
                        ).model_dump_json()
                    )
                    continue

                if message.message_type == RuntimeMessageType.REQUEST_EVENT:
                    requested_batch_size = int(message.payload.get("batch_size", 1) or 1)
                    if requested_batch_size > 1:
                        events = await state.broker.drain_next_batch(registered_path, requested_batch_size)
                    else:
                        single_event = await state.broker.drain_next(registered_path)
                        events = [single_event] if single_event is not None else []
                    if events:
                        trace_payload = {
                            "event_count": len(events),
                            "queue_kind": events[0].kind.value,
                            "event_ids": [item.event_id for item in events],
                        }
                        if len(events) == 1:
                            trace_payload["event"] = events[0].model_dump(mode="json")
                            await trace(registered_path, "event_dispatch", f"dispatching {events[0].kind.value}", trace_payload)
                        else:
                            trace_payload["events"] = [item.model_dump(mode="json") for item in events]
                            await trace(registered_path, "event_dispatch", f"dispatching batch of {len(events)} {events[0].kind.value} event(s)", trace_payload)
                    await websocket.send_text(
                        RuntimeMessage(
                            message_type=RuntimeMessageType.EVENT,
                            path=registered_path,
                            event=events[0] if len(events) == 1 else None,
                            events=events or None,
                            payload={
                                "batch_size": len(events),
                                "queue_kind": events[0].kind.value if events else None,
                            },
                        ).model_dump_json()
                    )
                    continue

                if message.message_type == RuntimeMessageType.ACK_EVENT and message.event_id:
                    await state.broker.acknowledge(registered_path, message.event_id)
                    await trace(registered_path, "event_ack", f"acknowledged {message.event_id}", {})
                    continue

                if message.message_type == RuntimeMessageType.PUBLISH_EVENT and message.event is not None:
                    await state.broker.publish(message.event)
                    await _wake_target(state, message.event.target_path)
                    await trace(registered_path, "event_publish", f"runtime published {message.event.kind.value}", {"event": message.event.model_dump(mode="json")})
                    continue

                if message.message_type == RuntimeMessageType.LOG:
                    await trace(
                        registered_path,
                        str(message.payload.get("category", "runtime_log")),
                        str(message.payload.get("message", "")),
                        dict(message.payload.get("payload", {})),
                    )
                    continue

                if message.message_type == RuntimeMessageType.HEARTBEAT:
                    state.registry.set_status(registered_path, NodeStatus.ONLINE)
                    continue
        except WebSocketDisconnect:
            pass
        finally:
            if registered_path is not None:
                await state.ws_hub.disconnect(registered_path)
                state.registry.set_status(registered_path, NodeStatus.OFFLINE)
                await trace(registered_path, "runtime", "runtime disconnected", {"tree": render_tree()})

    return app


async def _wake_target(state: CoreState, path: str) -> None:
    try:
        await state.ws_hub.wake(path)
    except Exception:
        return


async def _notify_root_agents_of_executor_registration(state: CoreState, node) -> None:
    root_children = state.registry.get_children("/")
    notified = False
    for child in root_children:
        if child.kind != NodeKind.AGENT:
            continue
        event = EventEnvelope(
            kind=EventKind.STRUCT,
            source_path="/supervisor",
            target_path=child.path,
            payload={
                "action": "executor_registered",
                "path": node.path,
                "executor_kind": node.metadata.get("executor_kind"),
                "description": node.description,
                "metadata": node.metadata,
            },
        )
        await state.broker.publish(event)
        await _wake_target(state, child.path)
        notified = True

    if notified:
        return

    fallback = EventEnvelope(
        kind=EventKind.STRUCT,
        source_path="/supervisor",
        target_path="/supervisor",
        payload={
            "action": "executor_registered",
            "path": node.path,
            "executor_kind": node.metadata.get("executor_kind"),
            "description": node.description,
            "metadata": node.metadata,
        },
    )
    await state.broker.publish(fallback)
    await _wake_target(state, "/supervisor")