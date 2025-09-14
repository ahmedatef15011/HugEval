from __future__ import annotations

import time
from typing import Callable, Protocol, Tuple


class Metric(Protocol):
    name: str

    def compute(self, ctx: dict) -> Tuple[float, int]:
        ...


def timed(fn: Callable[..., float]) -> Callable[..., Tuple[float, int]]:
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        score = float(fn(*args, **kwargs))
        dt_ms = int((time.perf_counter() - t0) * 1000)
        return max(0.0, min(1.0, score)), max(0, dt_ms)
    return wrapper
