from __future__ import annotations

import math

from .base import timed


@timed
def popularity_downloads_likes(
    downloads: int, likes: int, d_cap: int = 100_000, l_cap: int = 1_000
) -> float:
    """Log-scaled normalization for popularity."""
    d_norm = min(1.0, math.log1p(max(0, downloads)) / math.log1p(d_cap))
    likes_norm = min(1.0, math.log1p(max(0, likes)) / math.log1p(l_cap))
    return 0.6 * d_norm + 0.4 * likes_norm


@timed
def freshness_days_since_update(days: int) -> float:
    """0 days → 1.0; 365+ days → 0.0 linearly."""
    return max(0.0, min(1.0, 1 - (max(0, days) / 365)))
