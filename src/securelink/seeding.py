from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


SEED_SCHEMA_VERSION = 2


def _label_u32(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


@dataclass(frozen=True)
class SeedDescriptor:
    purpose: str
    run_seed: int
    worker_id: int
    episode_id: int
    scenario_id: int | None = None

    @property
    def key(self) -> str:
        scenario = "stream" if self.scenario_id is None else str(self.scenario_id)
        return (
            f"v{SEED_SCHEMA_VERSION}:{self.purpose}:run={self.run_seed}:"
            f"worker={self.worker_id}:episode={self.episode_id}:scenario={scenario}"
        )

    def seed_sequence(self) -> np.random.SeedSequence:
        scenario_flag = 0 if self.scenario_id is None else 1
        scenario_value = 0 if self.scenario_id is None else int(self.scenario_id)
        entropy = [
            SEED_SCHEMA_VERSION,
            _label_u32(self.purpose),
            int(self.run_seed),
            int(self.worker_id),
            int(self.episode_id),
            scenario_flag,
            scenario_value,
        ]
        return np.random.SeedSequence(entropy)

    def fingerprint(self) -> str:
        state = self.seed_sequence().generate_state(4, dtype=np.uint32)
        return "".join(f"{int(value):08x}" for value in state)


def substream_fingerprints(descriptor: SeedDescriptor, names: tuple[str, ...]) -> dict[str, str]:
    children = descriptor.seed_sequence().spawn(len(names))
    return {
        name: "".join(f"{int(value):08x}" for value in child.generate_state(2, dtype=np.uint32))
        for name, child in zip(names, children)
    }
