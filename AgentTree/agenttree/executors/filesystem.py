from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from agenttree.executors.base import ExternalExecutorBase, build_settings_for_core
from agenttree.schemas.events import EventEnvelope


class FileSystemExecutor(ExternalExecutorBase):
    def __init__(self, *, core: str, path: str, root_dir: str | None = None) -> None:
        self.root_dir = Path(root_dir).resolve() if root_dir else None
        super().__init__(
            settings=build_settings_for_core(core),
            path=path,
            executor_kind="filesystem.access",
            description="Read and modify system files with optional root directory restriction.",
            capabilities=["event_push", "read_file", "write_file", "append_file", "list_dir", "delete_path", "make_dir", "stat_path"],
            metadata={"root_dir": str(self.root_dir) if self.root_dir else None},
        )

    async def handle_event(self, event: EventEnvelope) -> None:
        command = str(event.payload.get("command") or "").strip().lower()
        try:
            if command == "read_file":
                result = self._read_file(event.payload)
            elif command == "write_file":
                result = self._write_file(event.payload, append=False)
            elif command == "append_file":
                result = self._write_file(event.payload, append=True)
            elif command == "list_dir":
                result = self._list_dir(event.payload)
            elif command == "make_dir":
                result = self._make_dir(event.payload)
            elif command == "delete_path":
                result = self._delete_path(event.payload)
            elif command == "stat_path":
                result = self._stat_path(event.payload)
            else:
                result = {
                    "handled": False,
                    "summary": f"unsupported filesystem command: {command or 'unknown'}",
                    "available_commands": ["read_file", "write_file", "append_file", "list_dir", "make_dir", "delete_path", "stat_path"],
                }
        except Exception as exc:
            result = {"handled": False, "summary": str(exc)}
        await self.reply_to_event(event, {"executor_path": self.path, **result})

    def _resolve_path(self, raw_path: Any) -> Path:
        path_text = str(raw_path or "").strip()
        if not path_text:
            raise ValueError("path is required")
        candidate = Path(path_text)
        if not candidate.is_absolute():
            candidate = (self.root_dir or Path.cwd()) / candidate
        resolved = candidate.resolve()
        if self.root_dir is not None:
            root = self.root_dir.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"path escapes root_dir: {resolved}") from exc
        return resolved

    def _read_file(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(payload.get("path"))
        encoding = str(payload.get("encoding") or "utf-8")
        text = path.read_text(encoding=encoding)
        return {"handled": True, "path": str(path), "text": text, "summary": f"read file {path}"}

    def _write_file(self, payload: dict[str, Any], *, append: bool) -> dict[str, Any]:
        path = self._resolve_path(payload.get("path"))
        encoding = str(payload.get("encoding") or "utf-8")
        content = str(payload.get("content") or "")
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with path.open(mode, encoding=encoding) as handle:
            handle.write(content)
        return {
            "handled": True,
            "path": str(path),
            "bytes_written": len(content.encode(encoding, errors="replace")),
            "summary": f"{'appended to' if append else 'wrote'} file {path}",
        }

    def _list_dir(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(payload.get("path") or ".")
        entries = []
        for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
            entries.append({"name": child.name, "path": str(child), "is_dir": child.is_dir()})
        return {"handled": True, "path": str(path), "entries": entries, "summary": f"listed directory {path}"}

    def _make_dir(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(payload.get("path"))
        path.mkdir(parents=bool(payload.get("parents", True)), exist_ok=bool(payload.get("exist_ok", True)))
        return {"handled": True, "path": str(path), "summary": f"created directory {path}"}

    def _delete_path(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(payload.get("path"))
        recursive = bool(payload.get("recursive", False))
        if path.is_dir():
            if recursive:
                shutil.rmtree(path)
            else:
                path.rmdir()
        else:
            path.unlink(missing_ok=False)
        return {"handled": True, "path": str(path), "summary": f"deleted path {path}"}

    def _stat_path(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._resolve_path(payload.get("path"))
        stat = path.stat()
        return {
            "handled": True,
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "is_file": path.is_file(),
            "size": stat.st_size,
            "summary": f"stat for {path}",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentTree filesystem executor")
    parser.add_argument("--core", required=True, help="Core address in host:port form")
    parser.add_argument("--path", required=True, help="Executor path in the Agent tree")
    parser.add_argument("--root-dir", help="Optional root directory restriction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = FileSystemExecutor(core=args.core, path=args.path, root_dir=args.root_dir)
    import asyncio
    asyncio.run(runtime.run())


if __name__ == "__main__":
    main()