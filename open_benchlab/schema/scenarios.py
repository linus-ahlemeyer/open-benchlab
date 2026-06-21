from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PromptFamily(str, Enum):
    REPETITIVE = "repetitive"
    PROSE = "prose"
    CODE = "code"
    AGENT = "agent"
    LOGS = "logs"
    MIXED = "mixed"


class PromptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: PromptFamily = PromptFamily.CODE
    system_prompt: str = "You are a precise local benchmark assistant."
    seed: int = 42
    token_tolerance: float = Field(default=0.02, gt=0, le=0.25)
    suffix: str = "\nRespond with a detailed but finite answer."


class ScenarioType(str, Enum):
    CONTEXT_CURVE = "context_curve"
    CONCURRENCY = "concurrency"
    JOIN_SHOCK = "join_shock"
    CANCELLATION = "cancellation"
    SMOKE = "smoke"


class ScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: ScenarioType
    prompt: PromptSpec = Field(default_factory=PromptSpec)
    context_tokens: list[int] | Literal["auto"] = "auto"
    runners: list[int] | Literal["auto"] = "auto"
    output_tokens: int = Field(default=256, ge=1)
    repeats: int = Field(default=1, ge=1)
    stagger_s: float = Field(default=0.0, ge=0)
    warmup: bool = True
    cancel_after_s: float = Field(default=3.0, gt=0)
    cancel_verify_s: float = Field(default=8.0, gt=0)
    join_delay_s: float = Field(default=2.0, ge=0)
    recovery_timeout_s: float = Field(default=60.0, gt=0)


class RankingWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    throughput: float = 0.35
    context_retention: float = 0.25
    concurrency_efficiency: float = 0.15
    join_resilience: float = 0.10
    ttft: float = 0.05
    stability: float = 0.10

    @model_validator(mode="after")
    def positive_sum(self) -> RankingWeights:
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("ranking weights must have a positive sum")
        return self
