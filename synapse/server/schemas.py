"""Pydantic schemas for the Synapse server layer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from synapse.models import NodeStatus, NodeType, SensitivityLevel
from synapse.server.write_path import IntegrateAction


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ErrorBody


class IntegrateKnowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    content: str = ""
    type: NodeType = NodeType.TRANSIENT
    links: list[str] = Field(default_factory=list)
    sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL
    # Explicit decision from the external agent/skill — Synapse only executes.
    action: IntegrateAction = IntegrateAction.CREATE
    target_node_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""


class DecideMemoryWriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    content: str = ""
    type: NodeType = NodeType.TRANSIENT
    links: list[str] = Field(default_factory=list)
    sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL
    query_hint: str | None = None
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class IntegrateMemoryWithSamplingRequest(DecideMemoryWriteRequest):
    allow_default_create_fallback: bool = False
    require_confident_decision: bool = False
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class RunDreamerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=8, ge=1, le=20)


class UpdateNodeStatusToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    status: NodeStatus
    superseded_by: str | None = None


class SearchExistingNodesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class SearchMemoryToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=25)


class GetNodeToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
