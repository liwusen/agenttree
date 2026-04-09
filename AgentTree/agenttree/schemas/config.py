from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfigRequest(BaseModel):
    model: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)
    model_temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ModelConfigResponse(BaseModel):
    model: str
    openai_api_key_configured: bool
    openai_base_url: str | None
    model_temperature: float