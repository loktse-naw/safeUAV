from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .environment import SecureLinkEnv


def fixed_action(method: str) -> np.ndarray:
    shapes = {"B00": 1, "B01": 1, "B10": 2, "B11": 3}
    return np.zeros(shapes[method], dtype=np.float32)


def run_episode(
    env: SecureLinkEnv,
    seed: int,
    predictor: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        if predictor is None:
            action = fixed_action(env.method)
        else:
            action = np.asarray(predictor(observation), dtype=np.float32)
        observation, _, terminated, truncated, info = env.step(action)
    summary = dict(info["episode_summary"])
    return summary, [dict(row) for row in env.step_records]


def evaluate_policy(
    config: dict[str, Any],
    method: str,
    attack_rule: str,
    seeds: list[int],
    predictor: Callable[[np.ndarray], np.ndarray] | None = None,
    train_seed: int | None = None,
    purpose: str = "development",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    env = SecureLinkEnv(
        config,
        method=method,
        attack_rule=attack_rule,
        stream_context={"purpose": purpose, "run_seed": 0, "worker_id": 0},
    )
    for episode_index, seed in enumerate(seeds):
        summary, rows = run_episode(env, seed, predictor)
        summary["episode_index"] = episode_index
        summary["train_seed"] = train_seed
        summaries.append(summary)
        for row in rows:
            row["episode_index"] = episode_index
            row["train_seed"] = train_seed
            steps.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(steps)


def dataframe_digest(frame: pd.DataFrame) -> str:
    payload = frame.to_json(orient="records", double_precision=15)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_records(
    episode_frame: pd.DataFrame,
    step_frame: pd.DataFrame,
    output_dir: str | Path,
    prefix: str,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / f"{prefix}_episodes.csv"
    step_path = output_dir / f"{prefix}_steps.csv"
    episode_frame.to_csv(episode_path, index=False)
    step_frame.to_csv(step_path, index=False)
    return episode_path, step_path


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
