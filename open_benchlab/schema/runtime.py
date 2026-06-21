from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeFlavor(str, Enum):
    STANDARD = "standard"
    TURBOQUANT = "turboquant"
    ATOMIC = "atomic"
    CUSTOM = "custom"


class RuntimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: Literal["llama_cpp", "openai_compatible"] = "llama_cpp"
    flavor: RuntimeFlavor = RuntimeFlavor.STANDARD
    binary: Path | None = None
    workdir: Path | None = None
    base_url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)
    command_template: list[str] | None = None
    startup_timeout_s: float = Field(default=900.0, gt=0)
    shutdown_timeout_s: float = Field(default=20.0, gt=0)
    health_path: str = "/health"
    chat_path: str = "/v1/chat/completions"
    tokenize_path: str = "/tokenize"
    slots_path: str = "/slots"

    @model_validator(mode="after")
    def validate_runtime(self) -> RuntimeSpec:
        if self.adapter == "llama_cpp" and self.binary is None:
            raise ValueError("llama_cpp runtime requires 'binary'")
        if self.adapter == "openai_compatible" and not self.base_url:
            raise ValueError("openai_compatible runtime requires 'base_url'")
        if self.flavor is RuntimeFlavor.CUSTOM and not self.command_template:
            raise ValueError("custom runtime flavor requires 'command_template'")
        return self


class ServerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    api_key: str | None = None
    add_metrics_flag: bool = True
    add_slots_flag: bool = True
