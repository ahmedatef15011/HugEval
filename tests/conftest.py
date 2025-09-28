"""Pytest configuration to ensure local package import resolution.

Ensures the repository root is first on sys.path so that imports like
`import acmecli` resolve to this working tree rather than any other copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if not sys.path or sys.path[0] != root_str:
    # Prepend repo root for priority
    sys.path.insert(0, root_str)
