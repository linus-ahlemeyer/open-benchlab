from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .schema import AxisSpec


@dataclass(slots=True)
class GPUInfo:
    index: int
    name: str
    memory_total_mib: int
    memory_free_mib: int | None = None
    driver_version: str | None = None
    compute_capability: str | None = None
    power_limit_w: float | None = None


@dataclass(slots=True)
class HardwareInfo:
    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_model: str | None = None
    logical_cpus: int | None = None
    ram_total_mib: int | None = None
    hostname: str = field(default_factory=platform.node)
    os: str = field(default_factory=platform.platform)
    python: str = field(default_factory=platform.python_version)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def primary_gpu(self) -> GPUInfo | None:
        return self.gpus[0] if self.gpus else None


def detect_hardware() -> HardwareInfo:
    return HardwareInfo(
        gpus=_detect_nvidia_gpus(),
        cpu_model=_cpu_model(),
        logical_cpus=_logical_cpu_count(),
        ram_total_mib=_ram_total_mib(),
    )


def _detect_nvidia_gpus() -> list[GPUInfo]:
    fields = ["index", "name", "memory.total", "memory.free", "driver_version", "compute_cap", "power.limit"]
    output: str | None = None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        fallback = fields[:5] + ["power.limit"]
        try:
            output = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={','.join(fallback)}", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            fields = fallback
        except (FileNotFoundError, subprocess.CalledProcessError):
            return []

    result: list[GPUInfo] = []
    for line in (output or "").splitlines():
        values = [part.strip() for part in line.split(",")]
        row = dict(zip(fields, values, strict=False))
        try:
            result.append(
                GPUInfo(
                    index=int(row.get("index", len(result))),
                    name=row.get("name", "unknown"),
                    memory_total_mib=int(float(row.get("memory.total", "0"))),
                    memory_free_mib=_optional_int(row.get("memory.free")),
                    driver_version=_optional_str(row.get("driver_version")),
                    compute_capability=_optional_str(row.get("compute_cap")),
                    power_limit_w=_optional_float(row.get("power.limit")),
                )
            )
        except ValueError:
            continue
    return result


def _cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _logical_cpu_count() -> int | None:
    try:
        return int(subprocess.check_output(["nproc"], text=True).strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def _ram_total_mib() -> int | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", path.read_text(), re.MULTILINE)
    return int(match.group(1)) // 1024 if match else None


def _optional_str(value: str | None) -> str | None:
    return value if value and value.upper() not in {"N/A", "[N/A]"} else None


def _optional_int(value: str | None) -> int | None:
    try:
        return int(float(value)) if _optional_str(value) else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: str | None) -> float | None:
    try:
        return float(value) if _optional_str(value) else None
    except (TypeError, ValueError):
        return None


class RangeEstimator:
    """Suggest bounded candidates; it never predicts throughput.

    Explicit YAML values bypass this class. Automatic values are deliberately
    conservative and are always measured by the real backend before ranking.
    """

    def __init__(
        self,
        hardware: HardwareInfo,
        *,
        model_size_bytes: int | None = None,
        context_limit: int | None = None,
    ) -> None:
        self.hardware = hardware
        self.model_size_bytes = model_size_bytes
        self.context_limit = context_limit or 262_144

    @property
    def vram_gib(self) -> float:
        gpu = self.hardware.primary_gpu
        return gpu.memory_total_mib / 1024 if gpu else 0.0

    @property
    def ram_gib(self) -> float:
        return self.hardware.ram_total_mib / 1024 if self.hardware.ram_total_mib else 0.0

    def values_for(self, name: str, axis: AxisSpec) -> list[Any]:
        profile = axis.auto if isinstance(axis.auto, str) else "balanced"
        values = self._raw_values(name, profile)
        values = self._apply_bounds(values, axis)
        if axis.count and len(values) > axis.count:
            values = _even_sample(values, axis.count)
        return values

    def scenario_contexts(self, profile: str = "balanced") -> list[int]:
        return [int(value) for value in self._raw_values("context_tokens", profile)]

    def scenario_runners(self, profile: str = "balanced") -> list[int]:
        return [int(value) for value in self._raw_values("runners", profile)]

    def _raw_values(self, name: str, profile: str) -> list[Any]:
        vram = self.vram_gib
        if name in {"context_tokens", "server_context"}:
            candidates = [2_048, 8_192, 16_384, 32_768]
            if vram >= 12:
                candidates += [49_152, 65_536]
            if vram >= 20:
                candidates += [98_304, 131_072]
            if vram >= 40:
                candidates += [196_608, 262_144]
            if profile == "conservative":
                candidates = candidates[: max(3, len(candidates) - 2)]
            elif profile == "aggressive" and self.context_limit > max(candidates):
                candidates.append(self.context_limit)
            return sorted({value for value in candidates if value <= self.context_limit})

        if name in {"server_parallel", "parallel"}:
            values = [1, 2]
            if vram >= 10:
                values.append(4)
            if profile == "aggressive" and vram >= 16:
                values += [6, 8]
            return values
        if name == "runners":
            values = [1, 2]
            if vram >= 10:
                values.append(4)
            return values
        if name == "spec_draft_n_max":
            return [2, 3] if profile != "aggressive" else [1, 2, 3, 4]
        if name == "mtp_enabled":
            return [False, True]
        if name == "cache_ram_mb":
            values = [0, 1024]
            if profile == "aggressive" and self.ram_gib >= 64:
                values.append(4096)
            return values
        if name == "kv_unified":
            return [False, True]
        if name == "kv_cache_profile":
            return ["none", "q8_q4"] if profile != "aggressive" else ["none", "q8_q4", "turbo4_turbo3"]
        if name == "ctx_checkpoints":
            return [0, 8] if profile == "aggressive" else [0]
        if name == "slot_prompt_similarity":
            return [0.0, 0.1] if profile == "aggressive" else [0.0]
        if name == "batch_size":
            return [None] if profile == "conservative" else [None, 512, 1024]
        if name == "ubatch_size":
            return [None] if profile != "aggressive" else [None, 256, 512]
        if name == "n_gpu_layers":
            return [999]
        if name == "n_cpu_moe":
            if self.model_size_bytes and vram:
                model_gib = self.model_size_bytes / (1024**3)
                if model_gib > vram * 0.9:
                    return [8, 16, 24]
            return [0, 8, 16] if profile == "aggressive" else [0]
        if name in {"cache_prompt", "cache_idle_slots"}:
            return [False, True]
        raise ValueError(
            f"no automatic estimator exists for axis {name!r}; provide explicit YAML values"
        )

    @staticmethod
    def _apply_bounds(values: list[Any], axis: AxisSpec) -> list[Any]:
        result = values
        if axis.minimum is not None:
            result = [value for value in result if isinstance(value, (int, float)) and value >= axis.minimum]
        if axis.maximum is not None:
            result = [value for value in result if isinstance(value, (int, float)) and value <= axis.maximum]
        return result


def _even_sample(values: list[Any], count: int) -> list[Any]:
    if count >= len(values):
        return values
    if count == 1:
        return [values[-1]]
    indexes = {round(index * (len(values) - 1) / (count - 1)) for index in range(count)}
    return [values[index] for index in sorted(indexes)]


def hardware_json(hardware: HardwareInfo) -> str:
    return json.dumps(hardware.to_dict(), indent=2, sort_keys=True)
