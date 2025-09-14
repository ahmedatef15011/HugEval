from __future__ import annotations
from .base import timed

# ----- Size (smaller = better; thresholds can be tuned) -----
@timed
def size_score(total_bytes: int, L: int = 50_000_000, U: int = 500_000_000) -> float:
    if U <= L:
        return 0.0
    return max(0.0, min(1.0, (U - total_bytes) / (U - L)))

# ----- License (LGPL v2.1 focus) -----
@timed
def license_score(license_text: str) -> float:
    if not license_text:
        return 0.5
    tex = license_text.lower()
    if "lgpl-2.1" in tex or "lgpl v2.1" in tex or "gnu lesser general public license v2.1" in tex:
        return 1.0
    return 0.0

# ----- Ramp Up Time (5 equally weighted doc checks) -----
@timed
def rampup_score(readme: float, quickstart: float, tutorials: float, api_docs: float, reproducibility: float) -> float:
    # inputs can be 0, 0.5, 1
    vals = [readme, quickstart, tutorials, api_docs, reproducibility]
    return sum(vals) / 5.0

# ----- Bus Factor (saturating with k) -----
@timed
def bus_factor_score(contributors: int, k: int = 5) -> float:
    c = max(0, contributors)
    return c / (c + k) if c + k > 0 else 0.0

# ----- Available Dataset & Code -----
@timed
def dataset_and_code_score(dataset_present: bool, code_present: bool) -> float:
    return (int(bool(dataset_present)) + int(bool(code_present))) / 2.0

# ----- Dataset Quality (4 documentation checks) -----
@timed
def dataset_quality_score(source: float, license_: float, splits: float, ethics: float) -> float:
    # each ∈ {0, 0.5, 1}
    return (source + license_ + splits + ethics) / 4.0

# ----- Code Quality (flake8/isort/mypy with weights 0.4/0.2/0.4) -----
@timed
def code_quality_score(flake8_errors: int, isort_sorted: bool, mypy_errors: int, emax: int = 50, tmax: int = 20) -> float:
    flake8_score = max(0.0, 1.0 - (flake8_errors / emax)) if emax > 0 else 0.0
    isort_score = 1.0 if isort_sorted else 0.0
    mypy_score = max(0.0, 1.0 - (mypy_errors / tmax)) if tmax > 0 else 0.0
    return 0.4 * flake8_score + 0.2 * isort_score + 0.4 * mypy_score

# ----- Performance Claims (benchmarks + citations) -----
@timed
def perf_claims_score(benchmarks_present: bool, citations_present: bool) -> float:
    return (int(bool(benchmarks_present)) + int(bool(citations_present))) / 2.0
