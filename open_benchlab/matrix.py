from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .config import BenchmarkConfig
from .hardware import RangeEstimator


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    config_name: str
    case_id: str
    parameters: dict[str, Any]


def expand_cases(config: BenchmarkConfig, estimator: RangeEstimator) -> list[BenchmarkCase]:
    axes: dict[str, list[Any]] = {}
    for name, axis in config.sweep.axes.items():
        if axis.values is not None:
            values = axis.values
        elif axis.auto:
            values = estimator.values_for(name, axis)
        else:
            raise ValueError(f"sweep axis {name!r} needs values or auto=true")
        if not values:
            raise ValueError(f"sweep axis {name!r} has no values")
        axes[name] = values

    names = sorted(axes)
    products: Iterable[tuple[Any, ...]] = itertools.product(*(axes[name] for name in names))
    raw: list[dict[str, Any]] = []
    for values in products:
        parameters = dict(config.sweep.base_parameters)
        parameters.update(dict(zip(names, values, strict=True)))
        parameters = _canonicalize(parameters)
        if _excluded(parameters, config.sweep.exclude):
            continue
        raw.append(parameters)

    raw = _deduplicate(raw)
    if len(raw) > config.sweep.max_cases:
        if config.sweep.sampling == "full":
            raise ValueError(
                f"{config.name}: sweep expands to {len(raw)} cases, over max_cases="
                f"{config.sweep.max_cases}; narrow ranges or set sampling: random"
            )
        rng = random.Random(config.sweep.seed)
        raw = rng.sample(raw, config.sweep.max_cases)
        raw.sort(key=_canonical_json)

    return [
        BenchmarkCase(
            config_name=config.name,
            case_id=_case_id(config.name, parameters),
            parameters=parameters,
        )
        for parameters in raw
    ]


def _canonicalize(parameters: dict[str, Any]) -> dict[str, Any]:
    result = dict(parameters)
    if result.get("mtp_enabled") is False:
        result["spec_draft_n_max"] = None
        result["spec_type"] = None
    if result.get("spec_draft_n_max") in {None, 0} and "mtp_enabled" not in result:
        result["mtp_enabled"] = False
    elif result.get("spec_draft_n_max"):
        result.setdefault("mtp_enabled", True)
    return result


def _excluded(parameters: dict[str, Any], exclusions: list[dict[str, Any]]) -> bool:
    return any(
        all(parameters.get(key) == value for key, value in rule.items())
        for rule in exclusions
    )


def _deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = _canonical_json(row)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _case_id(config_name: str, parameters: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        f"{config_name}\0{_canonical_json(parameters)}".encode()
    ).hexdigest()[:12]
    return f"{_slug(config_name)}-{digest}"


def _slug(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "-" for character in value
    ).strip("-")
