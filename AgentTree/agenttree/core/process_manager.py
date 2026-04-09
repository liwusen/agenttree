from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from agenttree.config import AgentTreeSettings
from agenttree.schemas.nodes import NodeKind, NodeRecord, normalize_node_path


@dataclass(slots=True)
class ManagedProcess:
    path: str
    kind: NodeKind
    process: subprocess.Popen[str]


class ProcessManager:
    def __init__(self, settings: AgentTreeSettings) -> None:
        self.settings = settings
        self._processes: dict[str, ManagedProcess] = {}

    def _python_cmd(self) -> list[str]:
        return [sys.executable]

    def _base_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env["AGENTTREE_HOST"] = self.settings.host
        env["AGENTTREE_PORT"] = str(self.settings.port)
        if self.settings.openai_api_key:
            env["AGENTTREE_OPENAI_API_KEY"] = self.settings.openai_api_key
        if self.settings.openai_base_url:
            env["AGENTTREE_OPENAI_BASE_URL"] = self.settings.openai_base_url
        env["AGENTTREE_MODEL"] = self.settings.model
        return env

    def ensure_process(self, node: NodeRecord) -> None:
        normalized = normalize_node_path(node.path)
        managed = self._processes.get(normalized)
        if managed is not None and managed.process.poll() is None:
            return
        command = self._python_cmd()
        if node.kind in (NodeKind.AGENT, NodeKind.SUPERVISOR, NodeKind.ROOT):
            command += ["-m", "agenttree.agent_runtime.main", "--path", normalized, "--kind", node.kind.value]
        elif node.kind == NodeKind.EXECUTOR:
            return
        else:
            return
        process = subprocess.Popen(
            command,
            cwd=str(self.settings.data_dir.parent if self.settings.data_dir.parent.exists() else os.getcwd()),
            env=self._base_env(),
            text=True,
        )
        self._processes[normalized] = ManagedProcess(path=normalized, kind=node.kind, process=process)

    def stop_process(self, path: str) -> None:
        managed = self._processes.pop(normalize_node_path(path), None)
        if managed is None:
            return
        if managed.process.poll() is None:
            managed.process.terminate()

    def list_processes(self) -> list[dict[str, str | int | None]]:
        items = []
        for path, managed in sorted(self._processes.items()):
            items.append(
                {
                    "path": path,
                    "kind": managed.kind.value,
                    "pid": managed.process.pid,
                    "returncode": managed.process.poll(),
                }
            )
        return items

    def shutdown(self) -> None:
        for path in list(self._processes.keys()):
            self.stop_process(path)