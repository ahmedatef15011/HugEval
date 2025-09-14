from acmecli.scoring import compute_all_scores


def test_compute_all_scores_returns_required_keys():
    ctx = {
        "total_bytes": 100_000_000,
        "license_text": "LGPL-2.1",
        "docs": {
            "readme": 1,
            "quickstart": 1,
            "tutorials": 1,
            "api_docs": 1,
            "reproducibility": 0.5,
        },
        "contributors": 5,
        "dataset_present": True,
        "code_present": True,
        "dataset_doc": {"source": 1, "license": 1, "splits": 1, "ethics": 0.5},
        "flake8_errors": 5,
        "isort_sorted": True,
        "mypy_errors": 3,
        "perf": {"benchmarks": True, "citations": True},
    }
    out = compute_all_scores(ctx)
    required = [
        "net_score",
        "net_score_latency",
        "ramp_up_time",
        "ramp_up_time_latency",
        "bus_factor",
        "bus_factor_latency",
        "performance_claims",
        "performance_claims_latency",
        "license",
        "license_latency",
        "size_score",
        "size_score_latency",
        "dataset_and_code_score",
        "dataset_and_code_score_latency",
        "dataset_quality",
        "dataset_quality_latency",
        "code_quality",
        "code_quality_latency",
    ]
    for k in required:
        assert k in out

    # ranges
    for k, v in out.items():
        if isinstance(v, (int, float)) and not k.endswith("latency"):
            assert 0.0 <= float(v) <= 1.0
