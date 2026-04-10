from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from agenttree.config import AgentTreeSettings, get_settings
from agenttree.core.app import create_app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentTree core service")
    parser.add_argument("--data-dir", dest="data_dir", help="Runtime data directory for state, logs, and knowledge")
    parser.add_argument("--openai-base", dest="openai_base_url", help="OpenAI-compatible base URL")
    parser.add_argument("--openai-key", dest="openai_api_key", help="OpenAI-compatible API key")
    parser.add_argument("--openai-main-model", dest="openai_main_model", help="Primary OpenAI-compatible chat model")
    parser.add_argument("--openai-temperature", dest="model_temperature", type=float, help="Model temperature override")
    return parser.parse_args(argv)


def build_settings_from_args(args: argparse.Namespace) -> AgentTreeSettings:
    if args.data_dir is None:
        return get_settings()
    settings = AgentTreeSettings(data_dir=Path(args.data_dir).expanduser())
    settings.ensure_dirs()
    settings.load_persisted_runtime_config()
    return settings


def apply_cli_runtime_overrides(settings: AgentTreeSettings, args: argparse.Namespace) -> None:
    if (
        args.openai_api_key is None
        and args.openai_base_url is None
        and args.openai_main_model is None
        and args.model_temperature is None
    ):
        return
    settings.update_model_runtime_config(
        model=args.openai_main_model,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        model_temperature=args.model_temperature,
    )


def main() -> None:
    args = parse_args()
    settings = build_settings_from_args(args)
    apply_cli_runtime_overrides(settings, args)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()