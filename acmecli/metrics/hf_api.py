from __future__ import annotations
from .base import timed

# In Step 3 of implementation, you'll replace these placeholders with real HF Hub calls.
# For Milestone 2, keep them deterministic & testable.

@timed
def popularity_downloads_likes(downloads: int, likes: int, d_cap: int = 100_000, l_cap: int = 1_000) -> float:
    import math
    d = min(1.0, math.log1p(max(0, downloads)) / math.log1p(d_cap))
    l = min(1.0, math.log1p(max(0, likes)) / math.log1p(l_cap))
    return 0.6 * d + 0.4 * l

@timed
def freshness_days_since_update(days: int) -> float:
    # 0 days → 1.0, 365+ days → 0.0 linearly
    return max(0.0, min(1.0, 1 - (max(0, days) / 365)))
