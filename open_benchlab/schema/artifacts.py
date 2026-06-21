from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArtifactSource(str, Enum):
    LOCAL = "local"
    HUGGINGFACE = "huggingface"


class ArtifactSpec(BaseModel):
    """A model-related file or Hugging Face selector."""

    model_config = ConfigDict(extra="forbid")

    source: ArtifactSource = ArtifactSource.HUGGINGFACE
    repo: str | None = None
    selector: str | None = None
    filename: str | None = None
    file_pattern: str | None = None
    revision: str = "main"
    path: Path | None = None
    local_dir: Path | None = None
    use_hf_selector: bool = False
    force_download: bool = False
    optional: bool = False

    @model_validator(mode="after")
    def validate_source(self) -> ArtifactSpec:
        if self.source is ArtifactSource.LOCAL:
            if self.path is None:
                raise ValueError("local artifact requires 'path'")
            return self
        if not self.repo:
            raise ValueError("Hugging Face artifact requires 'repo'")
        if self.use_hf_selector and not self.selector:
            raise ValueError("use_hf_selector=true requires 'selector'")
        if not self.use_hf_selector and not (self.filename or self.file_pattern):
            raise ValueError(
                "downloaded Hugging Face artifact requires 'filename' or 'file_pattern'"
            )
        return self


class DraftMode(str, Enum):
    NONE = "none"
    AUTO = "auto"
    EXPLICIT = "explicit"
    ATOMIC_HEAD = "atomic_head"


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    target: ArtifactSpec
    draft_mode: DraftMode = DraftMode.NONE
    draft: ArtifactSpec | None = None
    mmproj: ArtifactSpec | None = None
    context_limit: int | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_draft(self) -> ModelSpec:
        if self.draft_mode in {DraftMode.EXPLICIT, DraftMode.ATOMIC_HEAD} and not self.draft:
            raise ValueError(f"draft_mode={self.draft_mode.value!r} requires 'draft'")
        return self
