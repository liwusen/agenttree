from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock

from agenttree.schemas.nodes import (
    BindExecutorRequest,
    ChannelRecord,
    CreateAgentRequest,
    CreateExecutorRequest,
    MoveNodeRequest,
    NodeKind,
    NodeRecord,
    NodeStatus,
    RegisterExecutorRequest,
    TriggerSpec,
    TransferExecutorRequest,
    UpdateNodeRequest,
    node_parent_path,
    normalize_node_path,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class RegistrySnapshot:
    nodes: dict[str, NodeRecord] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)
    channels: dict[str, ChannelRecord] = field(default_factory=dict)


class NodeRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._nodes: dict[str, NodeRecord] = {}
        self._children: dict[str, set[str]] = {}
        self._channels: dict[str, ChannelRecord] = {}
        self._bootstrap_root()

    def _bootstrap_root(self) -> None:
        root = NodeRecord(path="/", kind=NodeKind.ROOT, parent_path=None, status=NodeStatus.ONLINE)
        self._nodes[root.path] = root
        self._children[root.path] = set()

    def snapshot(self) -> RegistrySnapshot:
        with self._lock:
            return RegistrySnapshot(
                nodes={path: record.model_copy(deep=True) for path, record in self._nodes.items()},
                children={path: sorted(children) for path, children in self._children.items()},
                channels={cid: record.model_copy(deep=True) for cid, record in self._channels.items()},
            )

    def list_nodes(self) -> list[NodeRecord]:
        with self._lock:
            return [record.model_copy(deep=True) for _, record in sorted(self._nodes.items())]

    def get_node(self, path: str) -> NodeRecord:
        normalized = normalize_node_path(path)
        with self._lock:
            record = self._nodes.get(normalized)
            if record is None:
                raise KeyError(f"node not found: {normalized}")
            return record.model_copy(deep=True)

    def get_children(self, path: str) -> list[NodeRecord]:
        normalized = normalize_node_path(path)
        with self._lock:
            child_paths = sorted(self._children.get(normalized, set()))
            return [self._nodes[item].model_copy(deep=True) for item in child_paths]

    def upsert_runtime_node(
        self,
        *,
        path: str,
        kind: NodeKind,
        capabilities: list[str] | None = None,
        metadata: dict | None = None,
    ) -> NodeRecord:
        normalized = normalize_node_path(path)
        with self._lock:
            existing = self._nodes.get(normalized)
            if existing is None:
                parent_path = node_parent_path(normalized)
                if parent_path is not None and parent_path not in self._nodes:
                    raise KeyError(f"parent not found: {parent_path}")
                record = NodeRecord(
                    path=normalized,
                    kind=kind,
                    parent_path=parent_path,
                    owner_path=parent_path if kind == NodeKind.EXECUTOR else None,
                    capabilities=capabilities or [],
                    metadata=metadata or {},
                    status=NodeStatus.ONLINE,
                )
                self._nodes[normalized] = record
                self._children.setdefault(normalized, set())
                if parent_path is not None:
                    self._children.setdefault(parent_path, set()).add(normalized)
                return record.model_copy(deep=True)

            existing.capabilities = list(capabilities or existing.capabilities)
            existing.metadata.update(metadata or {})
            existing.status = NodeStatus.ONLINE
            existing.updated_at = utc_now()
            return existing.model_copy(deep=True)

    def create_agent(self, request: CreateAgentRequest) -> NodeRecord:
        with self._lock:
            parent = self._nodes.get(normalize_node_path(request.parent_path))
            if parent is None:
                raise KeyError(f"parent not found: {request.parent_path}")
            path = request.path
            if path in self._nodes:
                raise ValueError(f"node already exists: {path}")
            record = NodeRecord(
                path=path,
                kind=NodeKind.AGENT,
                parent_path=parent.path,
                prompt=request.prompt,
                description=request.description,
                metadata=dict(request.metadata),
                status=NodeStatus.CREATED,
            )
            self._nodes[path] = record
            self._children.setdefault(path, set())
            self._children.setdefault(parent.path, set()).add(path)
            parent.updated_at = utc_now()
            return record.model_copy(deep=True)

    def create_executor(self, request: CreateExecutorRequest) -> NodeRecord:
        with self._lock:
            owner = self._nodes.get(normalize_node_path(request.owner_path))
            if owner is None:
                raise KeyError(f"owner not found: {request.owner_path}")
            path = request.path
            if path in self._nodes:
                raise ValueError(f"node already exists: {path}")
            record = NodeRecord(
                path=path,
                kind=NodeKind.EXECUTOR,
                parent_path=owner.path,
                owner_path=owner.path,
                metadata={"executor_kind": request.executor_kind, **request.config},
                status=NodeStatus.CREATED,
            )
            self._nodes[path] = record
            self._children.setdefault(path, set())
            self._children.setdefault(owner.path, set()).add(path)
            owner.updated_at = utc_now()
            return record.model_copy(deep=True)

    def register_executor(self, request: RegisterExecutorRequest) -> NodeRecord:
        with self._lock:
            path = normalize_node_path(request.path)
            if path in self._nodes:
                existing = self._nodes[path]
                if existing.kind != NodeKind.EXECUTOR:
                    raise ValueError(f"node already exists and is not executor: {path}")
                existing.description = request.description
                existing.capabilities = list(request.capabilities or existing.capabilities)
                existing.metadata.update({"executor_kind": request.executor_kind, "external": True, **request.metadata})
                existing.updated_at = utc_now()
                return existing.model_copy(deep=True)

            parent_path = node_parent_path(path)
            if parent_path is not None and parent_path not in self._nodes:
                raise KeyError(f"parent not found: {parent_path}")

            record = NodeRecord(
                path=path,
                kind=NodeKind.EXECUTOR,
                parent_path=parent_path,
                owner_path=None,
                description=request.description,
                capabilities=list(request.capabilities),
                metadata={"executor_kind": request.executor_kind, "external": True, **request.metadata},
                status=NodeStatus.CREATED,
            )
            self._nodes[path] = record
            self._children.setdefault(path, set())
            if parent_path is not None:
                self._children.setdefault(parent_path, set()).add(path)
            return record.model_copy(deep=True)

    def bind_executor(self, request: BindExecutorRequest) -> NodeRecord:
        with self._lock:
            executor = self._nodes.get(normalize_node_path(request.executor_path))
            if executor is None or executor.kind != NodeKind.EXECUTOR:
                raise KeyError(f"executor not found: {request.executor_path}")
            owner = self._nodes.get(normalize_node_path(request.owner_path))
            if owner is None:
                raise KeyError(f"owner not found: {request.owner_path}")
            executor.owner_path = owner.path
            executor.updated_at = utc_now()
            return executor.model_copy(deep=True)

    def transfer_executor(self, request: TransferExecutorRequest) -> NodeRecord:
        return self.bind_executor(BindExecutorRequest(executor_path=request.executor_path, owner_path=request.new_owner_path))

    def delete_node(self, path: str) -> list[str]:
        normalized = normalize_node_path(path)
        if normalized == "/":
            raise ValueError("root cannot be deleted")
        with self._lock:
            if normalized not in self._nodes:
                raise KeyError(f"node not found: {normalized}")
            removed: list[str] = []

            def _delete(current_path: str) -> None:
                for child_path in sorted(self._children.get(current_path, set())):
                    _delete(child_path)
                parent = self._nodes[current_path].parent_path
                if parent is not None:
                    self._children.setdefault(parent, set()).discard(current_path)
                self._children.pop(current_path, None)
                self._nodes.pop(current_path, None)
                removed.append(current_path)

            _delete(normalized)
            for channel_id, record in list(self._channels.items()):
                record.members = [member for member in record.members if member not in removed]
                if not record.members:
                    self._channels.pop(channel_id, None)
            return removed

    def move_node(self, request: MoveNodeRequest) -> list[tuple[str, NodeRecord]]:
        normalized = normalize_node_path(request.path)
        if normalized == "/":
            raise ValueError("root cannot be moved")
        destination_parent = normalize_node_path(request.new_parent_path)
        destination_path = request.new_path
        with self._lock:
            if normalized not in self._nodes:
                raise KeyError(f"node not found: {normalized}")
            if destination_parent not in self._nodes:
                raise KeyError(f"parent not found: {destination_parent}")
            if destination_path in self._nodes and destination_path != normalized:
                raise ValueError(f"node already exists: {destination_path}")
            if destination_parent == normalized or destination_parent.startswith(normalized + "/"):
                raise ValueError("cannot move a node into its own subtree")

            subtree = [path for path in sorted(self._nodes.keys(), key=lambda item: item.count("/")) if path == normalized or path.startswith(normalized + "/")]
            path_map = {old_path: destination_path + old_path[len(normalized):] for old_path in subtree}

            new_nodes = dict(self._nodes)
            for old_path in subtree:
                record = self._nodes[old_path].model_copy(deep=True)
                record.path = path_map[old_path]
                if old_path == normalized:
                    record.parent_path = destination_parent
                elif record.parent_path is not None:
                    record.parent_path = path_map.get(record.parent_path, record.parent_path)
                if record.owner_path is not None:
                    record.owner_path = path_map.get(record.owner_path, record.owner_path)
                record.updated_at = utc_now()
                new_nodes.pop(old_path, None)
                new_nodes[record.path] = record

            new_children = {path: set(children) for path, children in self._children.items()}
            for old_path in subtree:
                new_children.pop(old_path, None)
            for parent, children in new_children.items():
                rewritten = set()
                for child in children:
                    rewritten.add(path_map.get(child, child))
                new_children[parent] = rewritten
            for old_path in subtree:
                new_children[path_map[old_path]] = {
                    path_map.get(child, child) for child in self._children.get(old_path, set())
                }
            old_parent = self._nodes[normalized].parent_path
            if old_parent is not None:
                new_children.setdefault(old_parent, set()).discard(normalized)
            new_children.setdefault(destination_parent, set()).add(destination_path)

            for record in self._channels.values():
                record.members = [path_map.get(member, member) for member in record.members]

            self._nodes = new_nodes
            self._children = new_children
            return [(old_path, self._nodes[path_map[old_path]].model_copy(deep=True)) for old_path in subtree]

    def set_status(self, path: str, status: NodeStatus) -> NodeRecord:
        normalized = normalize_node_path(path)
        with self._lock:
            record = self._nodes.get(normalized)
            if record is None:
                raise KeyError(f"node not found: {normalized}")
            record.status = status
            record.updated_at = utc_now()
            return record.model_copy(deep=True)

    def update_node(self, path: str, request: UpdateNodeRequest) -> NodeRecord:
        normalized = normalize_node_path(path)
        with self._lock:
            record = self._nodes.get(normalized)
            if record is None:
                raise KeyError(f"node not found: {normalized}")
            if request.prompt is not None:
                record.prompt = request.prompt
            if request.description is not None:
                record.description = request.description
            if request.metadata:
                record.metadata.update(request.metadata)
            record.updated_at = utc_now()
            return record.model_copy(deep=True)

    def upsert_trigger(self, path: str, trigger: TriggerSpec) -> NodeRecord:
        normalized = normalize_node_path(path)
        with self._lock:
            record = self._nodes.get(normalized)
            if record is None:
                raise KeyError(f"node not found: {normalized}")
            replaced = False
            for index, existing in enumerate(record.triggers):
                if existing.trigger_id == trigger.trigger_id:
                    record.triggers[index] = trigger
                    replaced = True
                    break
            if not replaced:
                record.triggers.append(trigger)
            record.updated_at = utc_now()
            return record.model_copy(deep=True)

    def remove_trigger(self, path: str, trigger_id: str) -> NodeRecord:
        normalized = normalize_node_path(path)
        with self._lock:
            record = self._nodes.get(normalized)
            if record is None:
                raise KeyError(f"node not found: {normalized}")
            record.triggers = [trigger for trigger in record.triggers if trigger.trigger_id != trigger_id]
            record.updated_at = utc_now()
            return record.model_copy(deep=True)

    def create_channel(self, channel_id: str, members: list[str], metadata: dict | None = None) -> ChannelRecord:
        normalized_members = [normalize_node_path(item) for item in members]
        with self._lock:
            for item in normalized_members:
                if item not in self._nodes:
                    raise KeyError(f"member not found: {item}")
            record = ChannelRecord(channel_id=channel_id, members=sorted(set(normalized_members)), metadata=metadata or {})
            self._channels[channel_id] = record
            return record.model_copy(deep=True)

    def get_channel(self, channel_id: str) -> ChannelRecord:
        with self._lock:
            record = self._channels.get(channel_id)
            if record is None:
                raise KeyError(f"channel not found: {channel_id}")
            return record.model_copy(deep=True)

    def list_channels(self) -> list[ChannelRecord]:
        with self._lock:
            return [record.model_copy(deep=True) for _, record in sorted(self._channels.items())]