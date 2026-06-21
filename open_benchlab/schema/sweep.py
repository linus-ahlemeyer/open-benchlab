from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AxisSpec(BaseModel):
    """One explicit or hardware-estimated sweep axis."""

    model_config = ConfigDict(extra="forbid")

    values: list[Any] | None = None
    auto: bool | Literal["conservative", "balanced", "aggressive"] = False
    minimum: float | int | None = None
    maximum: float | int | None = None
    count: int | None = Field(default=None, ge=1)

    @classmethod
    def from_raw(cls, raw: Any) -> AxisSpec:
        if isinstance(raw, AxisSpec):
            return raw
        if isinstance(raw, list):
            return cls(values=raw)
        if raw == "auto":
            return cls(auto=True)
        if isinstance(raw, dict):
            normalized = dict(raw)
            if "min" in normalized:
                normalized["minimum"] = normalized.pop("min")
            if "max" in normalized:
                normalized["maximum"] = normalized.pop("max")
            return cls.model_validate(normalized)
        return cls(values=[raw])


class SweepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_parameters: dict[str, Any] = Field(default_factory=dict)
    axes: dict[str, AxisSpec] = Field(default_factory=dict)
    max_cases: int = Field(default=128, ge=1)
    sampling: Literal["full", "random"] = "full"
    seed: int = 42
    exclude: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("axes", mode="before")
    @classmethod
    def normalize_axes(cls, value: Any) -> dict[str, AxisSpec]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("sweep.axes must be a mapping")
        return {key: AxisSpec.from_raw(raw) for key, raw in value.items()}
