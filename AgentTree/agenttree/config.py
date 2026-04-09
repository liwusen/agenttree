from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentTreeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTTREE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 18990
    ws_path: str = "/ws/runtime"
    api_prefix: str = "/api"
    data_dir: Path = Field(default_factory=lambda: Path(".agenttree-data"))
    knowledge_dirname: str = "knowledge"
    state_dirname: str = "state"
    logs_dirname: str = "logs"

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    model: str = "DeepSeek-V3.2"
    model_temperature: float = 0.0

    heartbeat_interval_seconds: float = 5.0
    runtime_poll_interval_seconds: float = 1.0
    reconnect_delay_seconds: float = 2.0
    executor_rpc_timeout_seconds: float = 30.0

    auto_start_supervisor: bool = True
    auto_inject_system_knowledge_templates: bool = True
    demo_bootstrap: bool = True

    persisted_model_fields: tuple[str, ...] = (
        "model",
        "openai_api_key",
        "openai_base_url",
        "model_temperature",
    )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def runtime_ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}{self.ws_path}"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / self.state_dirname

    @property
    def knowledge_dir(self) -> Path:
        return self.data_dir / self.knowledge_dirname

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / self.logs_dirname

    @property
    def runtime_config_path(self) -> Path:
        return self.state_dir / "runtime_config.json"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def load_persisted_runtime_config(self) -> None:
        path = self.runtime_config_path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        for field_name in self.persisted_model_fields:
            if field_name in payload:
                setattr(self, field_name, payload[field_name])

    def save_persisted_runtime_config(self) -> None:
        payload = {field_name: getattr(self, field_name) for field_name in self.persisted_model_fields}
        self.runtime_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_model_runtime_config(
        self,
        *,
        model: str | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        model_temperature: float | None = None,
    ) -> None:
        if model is not None:
            self.model = model
        if openai_api_key is not None:
            self.openai_api_key = openai_api_key
        if openai_base_url is not None:
            self.openai_base_url = openai_base_url
        if model_temperature is not None:
            self.model_temperature = model_temperature
        self.save_persisted_runtime_config()


@lru_cache(maxsize=1)
def get_settings() -> AgentTreeSettings:
    settings = AgentTreeSettings()
    settings.ensure_dirs()
    settings.load_persisted_runtime_config()
    return settings