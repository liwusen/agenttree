from __future__ import annotations

from dataclasses import dataclass

from agenttree.config import AgentTreeSettings
from agenttree.core.broker import EventBroker
from agenttree.core.process_manager import ProcessManager
from agenttree.core.registry import NodeRegistry
from agenttree.core.tracing import TraceStore
from agenttree.core.ws_hub import RuntimeSessionHub
from agenttree.knowledge.store import KnowledgeStore
from agenttree.knowledge.sync import KnowledgeSyncService


@dataclass(slots=True)
class CoreState:
    settings: AgentTreeSettings
    registry: NodeRegistry
    broker: EventBroker
    ws_hub: RuntimeSessionHub
    process_manager: ProcessManager
    trace_store: TraceStore
    knowledge_store: KnowledgeStore
    knowledge_sync: KnowledgeSyncService


def build_core_state(settings: AgentTreeSettings) -> CoreState:
    knowledge_store = KnowledgeStore(settings)
    return CoreState(
        settings=settings,
        registry=NodeRegistry(),
        broker=EventBroker(),
        ws_hub=RuntimeSessionHub(),
        process_manager=ProcessManager(settings),
        trace_store=TraceStore(),
        knowledge_store=knowledge_store,
        knowledge_sync=KnowledgeSyncService(knowledge_store),
    )