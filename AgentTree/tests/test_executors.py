from __future__ import annotations

from pathlib import Path

import pytest

from agenttree.executors.base import build_settings_for_core
from agenttree.executors.command import SystemCommandExecutor
from agenttree.executors.filesystem import FileSystemExecutor
from agenttree.executors.mqtt import MQTTClientExecutor


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


def test_command_executor_exposes_structured_operation_specs(tmp_path: Path) -> None:
    executor = SystemCommandExecutor(core="localhost:18990", path="/supervisor/command.executor", working_dir=str(tmp_path))

    operations = executor.metadata.get("operations", [])

    assert operations
    run_command = next(item for item in operations if item["command"] == "run_command")
    assert any(field["name"] == "shell_command" for field in run_command["payload_schema"])
    assert any(field["name"] == "jobs" for field in run_command["returns"])
    assert executor.metadata.get("executor_usage", {}).get("invoke_tool") == "invoke_executor(executor_path, command, payload_json)"


def test_filesystem_executor_exposes_parameter_and_return_guides(tmp_path: Path) -> None:
    executor = FileSystemExecutor(core="localhost:18990", path="/supervisor/files.executor", root_dir=str(tmp_path))

    operations = executor.metadata.get("operations", [])

    assert operations
    read_file = next(item for item in operations if item["command"] == "read_file")
    assert any(field["name"] == "path" and field["required"] for field in read_file["payload_schema"])
    assert any(field["name"] == "text" for field in read_file["returns"])


def test_mqtt_executor_exposes_operation_guides() -> None:
    executor = MQTTClientExecutor(core="localhost:18990", path="/supervisor/mqtt.executor", broker="localhost:1883")

    operations = executor.metadata.get("operations", [])

    assert operations
    publish_op = next(item for item in operations if item["command"] == "mqtt_publish")
    assert any(field["name"] == "topic" for field in publish_op["payload_schema"])
    assert any(field["name"] == "mid" for field in publish_op["returns"])