from __future__ import annotations

import time
from typing import Any, Dict

from .metrics.repo_scan import (
    bus_factor_score,
    code_quality_score,
    dataset_and_code_score,
    dataset_quality_score,
    license_score,
    perf_claims_score,
    rampup_score,
    size_score,
)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "size": 0.05,
    "license": 0.20,
    "ramp_up_time": 0.15,
    "bus_factor": 0.10,
    "dataset_and_code": 0.20,
    "dataset_quality": 0.05,
    "code_quality": 0.15,
    "performance_claims": 0.10,
}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_all_scores(ctx: Dict[str, Any]) -> Dict[str, Any]:
    # repo-scan metrics
    size, size_ms = size_score(ctx.get("total_bytes", 0))
    lic, lic_ms = license_score(ctx.get("license_text", ""))
    d = ctx.get("docs", {})
    ramp, ramp_ms = rampup_score(
        d.get("readme", 0),
        d.get("quickstart", 0),
        d.get("tutorials", 0),
        d.get("api_docs", 0),
        d.get("reproducibility", 0),
    )
    bus, bus_ms = bus_factor_score(ctx.get("contributors", 0))
    dac, dac_ms = dataset_and_code_score(
        ctx.get("dataset_present", False), ctx.get("code_present", False)
    )
    dd = ctx.get("dataset_doc", {})
    dq, dq_ms = dataset_quality_score(
        dd.get("source", 0), dd.get("license", 0), dd.get("splits", 0), dd.get("ethics", 0)
    )
    cq, cq_ms = code_quality_score(
        ctx.get("flake8_errors", 0), ctx.get("isort_sorted", True), ctx.get("mypy_errors", 0)
    )
    p = ctx.get("perf", {})
    pc, pc_ms = perf_claims_score(p.get("benchmarks", False), p.get("citations", False))

    scores = {
        "size": clamp01(size),
        "license": clamp01(lic),
        "ramp_up_time": clamp01(ramp),
        "bus_factor": clamp01(bus),
        "dataset_and_code": clamp01(dac),
        "dataset_quality": clamp01(dq),
        "code_quality": clamp01(cq),
        "performance_claims": clamp01(pc),
    }
    latencies = {
        "size_score_latency": size_ms,
        "license_latency": lic_ms,
        "ramp_up_time_latency": ramp_ms,
        "bus_factor_latency": bus_ms,
        "dataset_and_code_score_latency": dac_ms,
        "dataset_quality_latency": dq_ms,
        "code_quality_latency": cq_ms,
        "performance_claims_latency": pc_ms,
    }
    # size object (same score for all devices for now — replace later if you
    # want device-specific thresholds)
    size_obj = {
        "raspberry_pi": scores["size"],
        "jetson_nano": scores["size"],
        "desktop_pc": scores["size"],
        "aws_server": scores["size"],
    }
    # NetScore
    t0 = time.perf_counter()
    net = sum(scores[k] * DEFAULT_WEIGHTS[k] for k in DEFAULT_WEIGHTS)
    net_latency = int((time.perf_counter() - t0) * 1000)
    result = {
        "net_score": clamp01(net),
        "net_score_latency": net_latency,
        "ramp_up_time": scores["ramp_up_time"],
        "ramp_up_time_latency": latencies["ramp_up_time_latency"],
        "bus_factor": scores["bus_factor"],
        "bus_factor_latency": latencies["bus_factor_latency"],
        "performance_claims": scores["performance_claims"],
        "performance_claims_latency": latencies["performance_claims_latency"],
        "license": scores["license"],
        "license_latency": latencies["license_latency"],
        "size_score": size_obj,
        "size_score_latency": latencies["size_score_latency"],
        "dataset_and_code_score": scores["dataset_and_code"],
        "dataset_and_code_score_latency": latencies["dataset_and_code_score_latency"],
        "dataset_quality": scores["dataset_quality"],
        "dataset_quality_latency": latencies["dataset_quality_latency"],
        "code_quality": scores["code_quality"],
        "code_quality_latency": latencies["code_quality_latency"],
    }
    return result
