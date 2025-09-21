"""
ACME Model Scoring Engine

This module implements the core scoring algorithms for trustworthy machine learning model
evaluation. It combines multiple quality dimensions into weighted composite scores that
reflect real-world deployment readiness and reliability.

The scoring system evaluates eight key dimensions: model size efficiency, license compliance,
documentation quality (ramp-up time), team sustainability (bus factor), dataset availability,
data quality, code quality, and performance validation. Each metric is carefully weighted
based on industry best practices for ML model deployment.

All scores are normalized to [0,1] range with automatic latency tracking for performance
monitoring in production environments.
"""

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

# Weighted scoring matrix optimized for production ML model assessment
# These weights reflect industry consensus on critical factors for model deployment
DEFAULT_WEIGHTS: Dict[str, float] = {
    "size": 0.05,  # Model efficiency and deployment feasibility
    "license": 0.20,  # Legal compliance and usage rights
    "ramp_up_time": 0.15,  # Documentation quality and developer experience
    "bus_factor": 0.10,  # Team sustainability and maintenance risk
    "dataset_and_code": 0.20,  # Reproducibility and transparency
    "dataset_quality": 0.05,  # Data provenance and documentation
    "code_quality": 0.15,  # Software engineering best practices
    "performance_claims": 0.10,  # Validation and benchmarking rigor
}


def clamp01(x: float) -> float:
    """
    Normalize score to valid [0,1] range with type safety.

    Essential for maintaining score consistency across different metric scales
    and ensuring reliable composite score calculations.
    """
    return max(0.0, min(1.0, float(x)))


def compute_all_scores(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute comprehensive model evaluation across all quality dimensions.

    This is the primary scoring orchestrator that coordinates individual metric
    calculations and combines them into weighted composite scores. Each metric
    includes latency tracking for performance monitoring and system optimization.

    Args:
        ctx: Context dictionary containing model metadata, API responses, and
             repository analysis results from build_context_from_api()

    Returns:
        Dict containing individual scores, latencies, and composite net_score
        representing overall model trustworthiness and deployment readiness
    """
    # Repository analysis metrics with performance timing
    size, size_ms = size_score(ctx.get("total_bytes", 0))
    lic, lic_ms = license_score(ctx.get("license_text", ""))

    # Documentation quality assessment using LLM-enhanced analysis
    d = ctx.get("docs", {})
    ramp, ramp_ms = rampup_score(
        d.get("readme", 0),
        d.get("quickstart", 0),
        d.get("tutorials", 0),
        d.get("api_docs", 0),
        d.get("reproducibility", 0),
    )

    # Team sustainability and maintenance risk evaluation
    bus, bus_ms = bus_factor_score(ctx.get("contributors", 0))

    # Reproducibility and transparency scoring
    dac, dac_ms = dataset_and_code_score(
        ctx.get("dataset_present", False), ctx.get("code_present", False)
    )

    # Data provenance and documentation quality
    dd = ctx.get("dataset_doc", {})
    dq, dq_ms = dataset_quality_score(
        dd.get("source", 0), dd.get("license", 0), dd.get("splits", 0), dd.get("ethics", 0)
    )

    # Software engineering best practices assessment
    cq, cq_ms = code_quality_score(
        ctx.get("flake8_errors", 0), ctx.get("isort_sorted", True), ctx.get("mypy_errors", 0)
    )

    # Performance validation and benchmarking rigor
    p = ctx.get("perf", {})
    pc, pc_ms = perf_claims_score(p.get("benchmarks", False), p.get("citations", False))

    # Normalize all scores to [0,1] range for consistent weighting
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

    # Performance monitoring data for system optimization
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

    # Device-specific deployment compatibility matrix
    # Currently unified across platforms - can be enhanced for device-specific optimization
    size_obj = {
        "raspberry_pi": scores["size"],
        "jetson_nano": scores["size"],
        "desktop_pc": scores["size"],
        "aws_server": scores["size"],
    }

    # Calculate weighted composite score representing overall model trustworthiness
    t0 = time.perf_counter()
    net = sum(scores[k] * DEFAULT_WEIGHTS[k] for k in DEFAULT_WEIGHTS)
    net_latency = int((time.perf_counter() - t0) * 1000)

    # Comprehensive results package for API consumers and business reporting
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
