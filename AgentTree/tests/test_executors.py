from __future__ import annotations

from pathlib import Path

import pytest

from agenttree.executors.base import build_settings_for_core
from agenttree.executors.filesystem import FileSystemExecutor


def test_build_settings_for_core_parses_host_port() -> None:
    settings = build_settings_for_core("localhost:12345")
    assert settings.host == "localhost"
    assert settings.port == 12345


def test_build_settings_for_core_parses_scheme() -> None:
    settings = build_settings_for_core("http://127.0.0.1:18990")
    assert settings.host == "127.0.0.1"
    assert settings.port == 18990


def test_filesystem_executor_blocks_escape(tmp_path: Path) -> None:
    executor = FileSystemExecutor(core="localhost:18990", path="/supervisor/files.executor", root_dir=str(tmp_path))
    with pytest.raises(ValueError):
        executor._resolve_path("../outside.txt")


def test_filesystem_executor_resolves_inside_root(tmp_path: Path) -> None:
    executor = FileSystemExecutor(core="localhost:18990", path="/supervisor/files.executor", root_dir=str(tmp_path))
    resolved = executor._resolve_path("nested/file.txt")
    assert resolved == (tmp_path / "nested" / "file.txt").resolve()