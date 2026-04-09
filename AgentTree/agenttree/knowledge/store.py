from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path
from typing import Any

import numpy as np
from nano_vectordb import NanoVectorDB
from openai import AsyncOpenAI

from agenttree.config import AgentTreeSettings
from agenttree.schemas.nodes import normalize_node_path


DOC_CONTENT_FILENAME = "__doc__.txt"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_path(path: str) -> str:
    return path.replace("\\", "/").strip()


def split_text(text: str, chunk_size: int = 600, overlap: int = 120) -> list[str]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start += step
    return chunks


@dataclass(slots=True)
class QueryHit:
    doc_path: str
    score: float | str
    text: str
    owner_node_path: str


class KnowledgeStore:
    def __init__(self, settings: AgentTreeSettings) -> None:
        self.settings = settings
        self.root = settings.knowledge_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "metadata.json"
        self.chunks_path = self.root / "chunks.json"
        self.vdb_path = self.root / "knowledge.vdb"
        self.metadata: dict[str, dict[str, Any]] = self._load_json(self.meta_path, {})
        self.chunks: dict[str, dict[str, Any]] = self._load_json(self.chunks_path, {})
        self.vdb = NanoVectorDB(1536, storage_file=str(self.vdb_path))

    def _load_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _save_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def flush(self) -> None:
        self._save_json(self.meta_path, self.metadata)
        self._save_json(self.chunks_path, self.chunks)

    def file_path_for_doc(self, doc_path: str) -> Path:
        normalized = sanitize_path(doc_path).lstrip("/")
        return self.root / normalized / DOC_CONTENT_FILENAME

    def _legacy_file_path_for_doc(self, doc_path: str) -> Path:
        normalized = sanitize_path(doc_path).lstrip("/")
        return self.root / normalized

    def _ensure_doc_directory(self, directory: Path) -> None:
        relative_parts = directory.relative_to(self.root).parts if directory != self.root else ()
        current = self.root
        for part in relative_parts:
            current = current / part
            if current.exists() and current.is_file():
                legacy_text = current.read_text(encoding="utf-8")
                current.unlink()
                current.mkdir(parents=True, exist_ok=True)
                (current / DOC_CONTENT_FILENAME).write_text(legacy_text, encoding="utf-8")
                continue
            current.mkdir(exist_ok=True)

    def _write_document_text(self, doc_path: str, text: str) -> None:
        file_path = self.file_path_for_doc(doc_path)
        self._ensure_doc_directory(file_path.parent)
        file_path.write_text(text, encoding="utf-8")

    def _delete_document_storage(self, doc_path: str) -> None:
        current_path = self.file_path_for_doc(doc_path)
        legacy_path = self._legacy_file_path_for_doc(doc_path)

        if current_path.exists():
            current_path.unlink()
            directory = current_path.parent
            while directory != self.root and directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
                directory = directory.parent
            return

        if legacy_path.exists() and legacy_path.is_file():
            legacy_path.unlink()

    def resolve_scope(self, node_path: str, scope: str) -> str:
        normalized_node = normalize_node_path(node_path)
        target = (scope or "./").strip()
        if target.startswith("/"):
            return sanitize_path(target).rstrip("/") + "/"
        relative_base = normalized_node.strip("/")
        if not relative_base:
            relative_base = ""
        combined = Path(relative_base) / target
        return "/" + sanitize_path(str(combined)).strip("/") + ("/" if target.endswith("/") or target in (".", "./") else "")

    async def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)
        if not self.settings.openai_api_key:
            return np.zeros((len(texts), 1536), dtype=np.float32)
        client = AsyncOpenAI(api_key=self.settings.openai_api_key, base_url=self.settings.openai_base_url)
        response = await client.embeddings.create(model="text-embedding-3-small", input=texts)
        return np.array([item.embedding for item in response.data], dtype=np.float32)

    async def upsert_document(self, *, owner_node_path: str, doc_path: str, text: str) -> dict[str, Any]:
        normalized_doc_path = "/" + sanitize_path(doc_path).lstrip("/")
        self._write_document_text(normalized_doc_path, text)

        doc_id = md5(normalized_doc_path.encode("utf-8")).hexdigest()
        old_chunk_ids = [chunk_id for chunk_id, meta in self.chunks.items() if meta.get("doc_id") == doc_id]
        if old_chunk_ids:
            try:
                self.vdb.delete(old_chunk_ids)
            except Exception:
                pass
            for chunk_id in old_chunk_ids:
                self.chunks.pop(chunk_id, None)

        chunks = split_text(text)
        embeddings = await self._embed_texts(chunks)
        rows = []
        for index, chunk in enumerate(chunks, start=1):
            chunk_id = f"{doc_id}::chunk::{index}"
            vector = embeddings[index - 1] if len(embeddings) >= index else np.zeros((1536,), dtype=np.float32)
            row = {
                "__id__": chunk_id,
                "__vector__": np.asarray(vector, dtype=np.float32),
                "doc_id": doc_id,
                "doc_path": normalized_doc_path,
                "owner_node_path": normalize_node_path(owner_node_path),
                "text": chunk,
                "text_preview": chunk[:80],
                "chunk_index": index,
            }
            self.chunks[chunk_id] = {k: v for k, v in row.items() if k != "__vector__"}
            rows.append(row)
        if rows:
            self.vdb.upsert(rows)

        record = {
            "doc_id": doc_id,
            "doc_path": normalized_doc_path,
            "owner_node_path": normalize_node_path(owner_node_path),
            "updated_at": utc_now_iso(),
            "content_length": len(text),
            "chunks_count": len(chunks),
        }
        self.metadata[normalized_doc_path] = record
        self.flush()
        return record

    def read_document(self, doc_path: str) -> str:
        normalized_doc_path = "/" + sanitize_path(doc_path).lstrip("/")
        file_path = self.file_path_for_doc(normalized_doc_path)
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        legacy_file_path = self._legacy_file_path_for_doc(normalized_doc_path)
        if legacy_file_path.exists() and legacy_file_path.is_file():
            return legacy_file_path.read_text(encoding="utf-8")
        raise FileNotFoundError(normalized_doc_path)

    def list_documents(self, scope: str) -> list[dict[str, Any]]:
        normalized_scope = sanitize_path(scope).rstrip("/")
        prefix = normalized_scope if normalized_scope.startswith("/") else "/" + normalized_scope
        if prefix != "/":
            prefix += "/"
        out = []
        for path, record in sorted(self.metadata.items()):
            if prefix == "/" or path.startswith(prefix):
                out.append(dict(record))
        return out

    def delete_document(self, doc_path: str) -> bool:
        normalized_doc_path = "/" + sanitize_path(doc_path).lstrip("/")
        record = self.metadata.pop(normalized_doc_path, None)
        if record is None:
            return False
        doc_id = str(record.get("doc_id", ""))
        chunk_ids = [chunk_id for chunk_id, meta in self.chunks.items() if meta.get("doc_id") == doc_id]
        if chunk_ids:
            try:
                self.vdb.delete(chunk_ids)
            except Exception:
                pass
            for chunk_id in chunk_ids:
                self.chunks.pop(chunk_id, None)
        self._delete_document_storage(normalized_doc_path)
        self.flush()
        return True

    def delete_scope(self, scope: str) -> list[str]:
        removed: list[str] = []
        for item in self.list_documents(scope):
            doc_path = str(item.get("doc_path", ""))
            if doc_path and self.delete_document(doc_path):
                removed.append(doc_path)
        return removed

    async def query(self, *, node_path: str, scope: str, query: str, top_k: int = 8) -> list[QueryHit]:
        scope_prefix = self.resolve_scope(node_path, scope).rstrip("/")
        vectors = await self._embed_texts([query])
        vector = vectors[0].tolist() if len(vectors) else np.zeros((1536,), dtype=np.float32).tolist()
        try:
            raw_hits = self.vdb.query(query=vector, top_k=top_k, better_than_threshold=None)
        except TypeError:
            raw_hits = self.vdb.query(vector, top_k=top_k)

        if isinstance(raw_hits, dict):
            raw_hits = [raw_hits]
        hits: list[QueryHit] = []
        for item in raw_hits or []:
            if isinstance(item, dict):
                meta = item
                score = item.get("__score__", "")
            elif isinstance(item, (list, tuple)) and item:
                if isinstance(item[0], dict):
                    meta = item[0]
                else:
                    meta = self.chunks.get(item[0], {})
                score = item[1] if len(item) > 1 else ""
            else:
                meta = self.chunks.get(str(item), {})
                score = ""
            doc_path = str(meta.get("doc_path", "") or "")
            if scope_prefix and scope_prefix != "/" and not doc_path.startswith(scope_prefix + "/") and doc_path != scope_prefix:
                continue
            hits.append(
                QueryHit(
                    doc_path=doc_path,
                    score=score,
                    text=str(meta.get("text", "") or meta.get("text_preview", "")),
                    owner_node_path=str(meta.get("owner_node_path", "") or ""),
                )
            )
        return hits

    def move_subtree(self, old_prefix: str, new_prefix: str) -> list[tuple[str, str]]:
        old_prefix = normalize_node_path(old_prefix)
        new_prefix = normalize_node_path(new_prefix)
        moved: list[tuple[str, str]] = []
        for doc_path, record in list(self.metadata.items()):
            owner = str(record.get("owner_node_path", ""))
            if not owner.startswith(old_prefix):
                continue
            suffix = owner[len(old_prefix):]
            new_owner = new_prefix + suffix
            new_doc_path = doc_path
            if doc_path.startswith(old_prefix):
                new_doc_path = new_prefix + doc_path[len(old_prefix):]
            content = self.read_document(doc_path)
            self._write_document_text(new_doc_path, content)
            self._delete_document_storage(doc_path)
            self.metadata.pop(doc_path, None)
            record["doc_path"] = new_doc_path
            record["owner_node_path"] = new_owner
            self.metadata[new_doc_path] = record
            for chunk_id, meta in self.chunks.items():
                if meta.get("doc_id") == record.get("doc_id"):
                    meta["doc_path"] = new_doc_path
                    meta["owner_node_path"] = new_owner
            moved.append((doc_path, new_doc_path))
        self.flush()
        return moved