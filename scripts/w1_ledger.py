"""Durable cumulative call budget. Ambiguous interrupted calls cannot be replayed."""
import json
import os
from pathlib import Path


class CallLedger:
    def __init__(self, path, cap, episode_cap):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap, self.episode_cap = cap, episode_cap
        self.calls = self.real = self.padded = self.episodes = 0
        self.pending = None
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                self._apply(json.loads(line))

    def _apply(self, item):
        kind = item["kind"]
        if kind == "episode":
            self.episodes += 1
        elif kind == "reserve":
            if self.pending is not None or item["call"] != self.calls+1:
                raise RuntimeError("Duplicate or unresolved call in ledger")
            self.calls += 1
            self.pending = item["call"]
        elif kind == "commit":
            if self.pending != item["call"]:
                raise RuntimeError("Unmatched committed call")
            self.real += item["real"]
            self.padded += item["padded"]
            self.pending = None
        if self.calls > self.cap or self.episodes > self.episode_cap:
            raise RuntimeError("Cumulative budget exceeded")

    def append(self, item):
        with self.path.open("a",encoding="utf-8",newline="\n") as stream:
            stream.write(json.dumps(item,ensure_ascii=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._apply(item)

    def begin_episode(self, **context):
        if self.pending is not None or self.episodes >= self.episode_cap:
            raise RuntimeError("Cannot reset with unresolved call or exhausted episode budget")
        self.append({"kind":"episode",**context})

    def reserve(self, **context):
        if self.pending is not None or self.calls >= self.cap:
            raise RuntimeError("Cannot replay unresolved call or exceed sampling cap")
        self.append({"kind":"reserve","call":self.calls+1,**context})

    def commit(self, real, padded):
        if real not in (0,1) or padded < 0 or self.pending is None:
            raise ValueError("One call may contain zero/one physical step and failure log padding")
        self.append({"kind":"commit","call":self.pending,"real":int(real),"padded":int(padded)})

    def summary(self):
        return dict(sampler_calls=self.calls,real_transitions=self.real,padded_slots=self.padded,
                    episode_starts=self.episodes,unresolved_call=self.pending,
                    zero_physical_step_calls=self.calls-self.real if self.pending is None else None)
