from __future__ import annotations

from pathlib import Path

from agenttree.config import AgentTreeSettings
from agenttree.core.main import apply_cli_runtime_overrides, build_settings_from_args, parse_args


def test_parse_args_supports_openai_runtime_overrides() -> None:
    args = parse_args(
        [
            "--openai-base",
            "https://api.example.com/v1",
            "--openai-key",
            "secret-key",
            "--openai-main-model",
            "gpt-test",
            "--openai-temperature",
            "0.3",
        ]
    )

    assert args.openai_base_url == "https://api.example.com/v1"
    assert args.openai_api_key == "secret-key"
    assert args.openai_main_model == "gpt-test"
    assert args.model_temperature == 0.3


def test_parse_args_supports_data_dir_override(tmp_path: Path) -> None:
    args = parse_args(["--data-dir", str(tmp_path)])

    assert args.data_dir == str(tmp_path)


def test_build_settings_from_args_uses_explicit_data_dir(tmp_path: Path) -> None:
    target_dir = tmp_path / "runtime-data"
    args = parse_args(["--data-dir", str(target_dir)])

    settings = build_settings_from_args(args)

    assert settings.data_dir == target_dir
    assert settings.runtime_config_path.parent == target_dir / settings.state_dirname
    assert settings.knowledge_dir == target_dir / settings.knowledge_dirname
    assert target_dir.exists()


def test_apply_cli_runtime_overrides_updates_settings(tmp_path) -> None:
    settings = AgentTreeSettings(data_dir=tmp_path)
    settings.ensure_dirs()
    args = parse_args(
        [
            "--openai-base",
            "https://api.example.com/v1",
            "--openai-key",
            "secret-key",
            "--openai-main-model",
            "gpt-test",
        ]
    )

    apply_cli_runtime_overrides(settings, args)

    assert settings.openai_base_url == "https://api.example.com/v1"
    assert settings.openai_api_key == "secret-key"
    assert settings.model == "gpt-test"
    assert settings.runtime_config_path.exists()