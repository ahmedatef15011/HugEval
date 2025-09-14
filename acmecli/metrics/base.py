from __future__ import annotations
from typing import Protocol, Tuple, Callable, Dict, Any
import time


class Metric(Protocol):
    name: str

    def compute(self, ctx: Dict[str, Any]) -> Tuple[float, int]:
        ...


def timed(fn: Callable[..., float]) -> Callable[..., Tuple[float, int]]:
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[float, int]:
        t0 = time.perf_counter()
        score = float(fn(*args, **kwargs))
        dt_ms = int((time.perf_counter() - t0) * 1000)
        # clamp score to [0,1], latency non-negative
        return max(0.0, min(1.0, score)), max(0, dt_ms)

    return wrapper
