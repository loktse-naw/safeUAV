from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    parent = config.get("extends")
    if parent:
        parent_path = (path.parent / parent).resolve()
        config = _deep_merge(load_config(parent_path), config)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    slots = int(config["time"]["slots"])
    slot_s = float(config["time"]["slot_s"])
    duty = float(config["time"]["tx_duty"])
    if slots <= 0 or slot_s <= 0 or not 0 < duty < 1:
        raise ValueError("slots and slot_s must be positive and tx_duty must be in (0, 1)")
    for name in ("alice_peak_w", "alice_radiated_budget_j", "willie_peak_w", "willie_budget_j"):
        if float(config["power"][name]) < 0:
            raise ValueError(f"power.{name} must be nonnegative")
    if float(config["energy"]["alice_battery_j"]) <= 0:
        raise ValueError("energy.alice_battery_j must be positive")


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
