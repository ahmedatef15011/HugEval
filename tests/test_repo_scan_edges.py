from acmecli.metrics.repo_scan import (
    bus_factor_score,
    code_quality_score,
    dataset_and_code_score,
    size_score,
)


def test_size_score_upper_less_equal_lower_returns_zero():
    s, _ = size_score(total_bytes=10, L=100, U=100)  # U <= L edge
    assert s == 0.0


def test_bus_factor_negative_contributors():
    s, _ = bus_factor_score(-5)
    assert 0.0 <= s <= 1.0


def test_code_quality_extremes():
    best, _ = code_quality_score(0, True, 0)
    worst, _ = code_quality_score(10_000, False, 10_000)
    assert 0 <= worst < best <= 1


def test_dataset_and_code_combinations():
    s0, _ = dataset_and_code_score(False, False)
    s1, _ = dataset_and_code_score(True, False)
    s2, _ = dataset_and_code_score(False, True)
    s3, _ = dataset_and_code_score(True, True)
    assert (s0, s1, s2, s3) == (0.0, 0.5, 0.5, 1.0)
