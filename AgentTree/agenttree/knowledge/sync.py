from __future__ import annotations

from agenttree.knowledge.store import KnowledgeStore


class KnowledgeSyncService:
    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def handle_node_move(self, old_path: str, new_path: str) -> list[tuple[str, str]]:
        return self.store.move_subtree(old_path, new_path)

    def handle_node_delete(self, path: str) -> list[str]:
        return self.store.delete_scope(path)