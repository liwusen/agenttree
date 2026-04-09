from __future__ import annotations

from pydantic import BaseModel, Field


class ExportPromptsRequest(BaseModel):
    owner_node_path: str
    target_root_path: str
    prompt_names: list[str] = Field(default_factory=list)


class ExportKnowledgeTemplatesRequest(BaseModel):
    owner_node_path: str
    target_root_path: str
    knowledge_names: list[str] = Field(default_factory=list)


class PromptTemplateRecord(BaseModel):
    name: str
    description: str = ""
    prompt: str
    default_knowledge_doc_path: str | None = None
    tags: list[str] = Field(default_factory=list)


class KnowledgeTemplateRecord(BaseModel):
    name: str
    description: str = ""
    text: str
    default_doc_path: str | None = None
    tags: list[str] = Field(default_factory=list)