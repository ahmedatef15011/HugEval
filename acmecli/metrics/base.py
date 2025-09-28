"""
Core Metrics Framework for ML Model Trustworthiness Assessment

This module provides the foundational architecture for the ACME Model Scoring system,
defining the standardized interfaces and performance measurement infrastructure that
ensures consistent, reliable evaluation across all trustworthiness dimensions.

The framework enforces a uniform scoring protocol where all metrics return normalized
scores in the [0,1] range with precise timing measurements, enabling fair comparison
and aggregation across diverse evaluation criteria. This design supports both
real-time individual assessments and large-scale batch processing for enterprise
model selection workflows.

Key Design Principles:
- Protocol-based interfaces for extensible metric development
- Automatic performance timing for system monitoring and optimization
- Score normalization and validation for consistent interpretation
- Type safety and runtime error prevention through comprehensive validation
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Protocol, Tuple


class Metric(Protocol):
    """
    Standardized interface for all model trustworthiness assessment metrics.

    Defines the contract that all evaluation metrics must implement to ensure
    consistent behavior, reliable timing measurement, and normalized scoring
    across the entire assessment framework.

    This protocol enables the scoring system to treat all metrics uniformly
    while maintaining type safety and performance monitoring capabilities
    essential for production deployment scenarios.

    Attributes:
        name: Human-readable metric identifier for reporting and logging
    """

    name: str

    def compute(self, ctx: Dict[str, Any]) -> Tuple[float, int]:
        """
        Execute metric calculation with performance timing and score validation.

        Implements the core metric computation logic while automatically handling
        performance measurement and score normalization. The context dictionary
        provides flexible parameter passing for different metric requirements.

        Args:
            ctx: Evaluation context containing model data, configuration parameters,
                 and any additional information required for metric computation

        Returns:
            Tuple[float, int]: Normalized score [0.0, 1.0] and execution time in milliseconds
        """
        ...


def timed(fn: Callable[..., float]) -> Callable[..., Tuple[float, int]]:
    """
    Performance monitoring decorator for metric functions with automatic score validation.

    Wraps scoring functions to provide precise execution timing and enforce score
    normalization to the [0,1] range. This decorator is essential for system monitoring,
    performance optimization, and ensuring consistent score interpretation across
    all trustworthiness dimensions.

    Features:
    - High-precision timing using performance counters
    - Automatic score clamping to valid [0,1] range
    - Actual execution time measurement in milliseconds
    - Type safety with proper return value validation

    This decorator is applied to all scoring functions to measure actual computation
    time and enable comprehensive system monitoring for production deployment reliability.

    Args:
        fn: Scoring function that returns a raw score value

    Returns:
        Callable: Wrapped function returning (normalized_score, actual_execution_time_ms)
    """
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[float, int]:
        # High-precision timing for accurate performance measurement
        t0 = time.perf_counter()
        score = float(fn(*args, **kwargs))
        actual_dt_ms = int((time.perf_counter() - t0) * 1000)

        # Use actual computation time for realistic latency measurement
        dt_ms = max(0, actual_dt_ms)

        # Enforce score normalization and ensure non-negative latency
        return max(0.0, min(1.0, score)), dt_ms

    return wrapper
