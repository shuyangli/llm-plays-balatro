from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class DeterministicRNG:
    seed: int
    counters: dict[str, int] = field(default_factory=dict)

    def _next_random(self, stream: str) -> random.Random:
        counter = self.counters.get(stream, 0)
        self.counters[stream] = counter + 1
        material = f"{self.seed}:{stream}:{counter}".encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return random.Random(derived_seed)

    def shuffle(self, values: Iterable[T], stream: str) -> list[T]:
        items = list(values)
        self._next_random(stream).shuffle(items)
        return items

    def choice(self, values: Sequence[T], stream: str) -> T:
        if not values:
            raise ValueError("choice() requires a non-empty sequence")
        generator = self._next_random(stream)
        return values[generator.randrange(len(values))]

    def snapshot(self) -> dict[str, object]:
        return {"seed": self.seed, "counters": dict(self.counters)}

    @classmethod
    def restore(cls, snapshot: dict[str, object]) -> "DeterministicRNG":
        counters = snapshot.get("counters", {})
        return cls(seed=int(snapshot["seed"]), counters=dict(counters))
