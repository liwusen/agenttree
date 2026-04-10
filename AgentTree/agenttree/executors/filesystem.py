from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from agenttree.executors.base import ExternalExecutorBase, build_settings_for_core, operation_field, operation_spec
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
            operations=[
                operation_spec(
                    "read_file",
                    summary="读取文件内容。",
                    description="读取指定路径文件的文本内容。适用于配置、日志、脚本或知识文本读取。",
                    payload_schema=[
                        operation_field("path", field_type="string", description="要读取的文件路径。相对路径会基于 root_dir 或当前工作目录解析。", required=True),
                        operation_field("encoding", field_type="string", description="文本编码，默认 utf-8。", required=False, default="utf-8"),
                    ],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否读取成功。", required=True),
                        operation_field("path", field_type="string", description="解析后的真实路径。", required=True),
                        operation_field("text", field_type="string", description="文件内容。", required=False),
                        operation_field("summary", field_type="string", description="结果摘要。", required=True),
                    ],
                ),
                operation_spec(
                    "write_file",
                    summary="覆盖写入文件。",
                    description="将 content 写入目标文件；如果文件不存在会自动创建。",
                    payload_schema=[
                        operation_field("path", field_type="string", description="目标文件路径。", required=True),
                        operation_field("content", field_type="string", description="要写入的文本内容。", required=True),
                        operation_field("encoding", field_type="string", description="文本编码，默认 utf-8。", required=False, default="utf-8"),
                    ],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否写入成功。", required=True),
                        operation_field("path", field_type="string", description="解析后的真实路径。", required=True),
                        operation_field("bytes_written", field_type="integer", description="写入字节数。", required=True),
                        operation_field("summary", field_type="string", description="结果摘要。", required=True),
                    ],
                ),
                operation_spec(
                    "append_file",
                    summary="向文件追加文本。",
                    description="将 content 追加到目标文件尾部；如果文件不存在会自动创建。",
                    payload_schema=[
                        operation_field("path", field_type="string", description="目标文件路径。", required=True),
                        operation_field("content", field_type="string", description="要追加的文本内容。", required=True),
                        operation_field("encoding", field_type="string", description="文本编码，默认 utf-8。", required=False, default="utf-8"),
                    ],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否追加成功。", required=True),
                        operation_field("path", field_type="string", description="解析后的真实路径。", required=True),
                        operation_field("bytes_written", field_type="integer", description="追加写入字节数。", required=True),
                        operation_field("summary", field_type="string", description="结果摘要。", required=True),
                    ],
                ),
                operation_spec(
                    "list_dir",
                    summary="列出目录内容。",
                    description="返回目录下的文件和子目录列表。",
                    payload_schema=[
                        operation_field("path", field_type="string", description="目录路径，默认当前目录。", required=False, default="."),
                    ],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否查询成功。", required=True),
                        operation_field("entries", field_type="array[object]", description="目录项列表，每项包含 name/path/is_dir。", required=True),
                        operation_field("summary", field_type="string", description="结果摘要。", required=True),
                    ],
                ),
                operation_spec(
                    "make_dir",
                    summary="创建目录。",
                    description="创建目标目录，可递归创建父目录。",
                    payload_schema=[
                        operation_field("path", field_type="string", description="目录路径。", required=True),
                        operation_field("parents", field_type="boolean", description="是否递归创建父目录，默认 true。", required=False, default=True),
                        operation_field("exist_ok", field_type="boolean", description="目录已存在时是否忽略错误，默认 true。", required=False, default=True),
                    ],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否创建成功。", required=True),
                        operation_field("path", field_type="string", description="解析后的真实路径。", required=True),
                        operation_field("summary", field_type="string", description="结果摘要。", required=True),
                    ],
                ),
                operation_spec(
                    "delete_path",
                    summary="删除文件或目录。",
                    description="删除指定文件；如果是目录且要递归删除，必须显式传 recursive=true。",
                    payload_schema=[
                        operation_field("path", field_type="string", description="要删除的文件或目录路径。", required=True),
                        operation_field("recursive", field_type="boolean", description="删除目录时是否递归删除，默认 false。", required=False, default=False),
                    ],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否删除成功。", required=True),
                        operation_field("path", field_type="string", description="解析后的真实路径。", required=True),
                        operation_field("summary", field_type="string", description="结果摘要。", required=True),
                    ],
                ),
                operation_spec(
                    "stat_path",
                    summary="获取路径状态信息。",
                    description="查询文件或目录是否存在、大小、类型等基础属性。",
                    payload_schema=[
                        operation_field("path", field_type="string", description="要查询的路径。", required=True),
                    ],
                    returns=[
                        operation_field("handled", field_type="boolean", description="是否查询成功。", required=True),
                        operation_field("exists", field_type="boolean", description="路径是否存在。", required=True),
                        operation_field("is_dir", field_type="boolean", description="是否是目录。", required=True),
                        operation_field("is_file", field_type="boolean", description="是否是文件。", required=True),
                        operation_field("size", field_type="integer", description="文件大小字节数。", required=True),
                        operation_field("summary", field_type="string", description="结果摘要。", required=True),
                    ],
                ),
            ],
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