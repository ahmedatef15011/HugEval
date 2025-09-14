from acmecli.metrics.repo_scan import (
    size_score, license_score, rampup_score, bus_factor_score,
    dataset_and_code_score, dataset_quality_score, code_quality_score, perf_claims_score
)

def test_metrics_return_in_range():
    assert 0 <= size_score(100_000_000)[0] <= 1
    assert 0 <= license_score("LGPL-2.1")[0] <= 1
    assert 0 <= rampup_score(1,1,0,1,0)[0] <= 1
    assert 0 <= bus_factor_score(10)[0] <= 1
    assert 0 <= dataset_and_code_score(True, False)[0] <= 1
    assert 0 <= dataset_quality_score(1, 1, 0.5, 0)[0] <= 1
    assert 0 <= code_quality_score(10, True, 5)[0] <= 1
    assert 0 <= perf_claims_score(True, True)[0] <= 1

def test_bus_factor_monotonic():
    a, _ = bus_factor_score(1)
    b, _ = bus_factor_score(5)
    c, _ = bus_factor_score(20)
    assert a < b < c

def test_license_unclear_when_missing():
    s, _ = license_score("")
    assert s == 0.5

def test_perf_claims_half_when_one_present():
    s1, _ = perf_claims_score(True, False)
    s2, _ = perf_claims_score(False, True)
    assert s1 == 0.5 and s2 == 0.5
