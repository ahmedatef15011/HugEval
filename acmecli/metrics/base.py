from __future__ import annotations

import time
from typing import Any, Callable, Dict, Protocol, Tuple


class Metric(Protocol):
    name: str

    def compute(self, ctx: Dict[str, Any]) -> Tuple[float, int]:
        """Return (score in [0,1], latency_ms)."""
        ...


def timed(fn: Callable[..., float]) -> Callable[..., Tuple[float, int]]:
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[float, int]:
        t0 = time.perf_counter()
        score = float(fn(*args, **kwargs))
        dt_ms = int((time.perf_counter() - t0) * 1000)
        # clamp score to [0,1] and ensure non-negative latency
        return max(0.0, min(1.0, score)), max(0, dt_ms)

    return wrapper
