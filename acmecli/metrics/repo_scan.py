"""
Core Model Quality Assessment Algorithms

This module implements the fundamental scoring algorithms that evaluate machine learning
models across critical deployment dimensions. Each algorithm is scientifically designed
based on industry best practices and empirical research into factors that predict
successful ML model deployment in production environments.

The scoring system addresses eight key quality dimensions that directly impact
deployment success: model size efficiency, license compliance, documentation quality,
team sustainability, data availability, quality assurance, code standards, and
performance validation. All algorithms include performance timing for system monitoring.

Each metric is carefully calibrated with thresholds derived from production deployment
experience and industry benchmarks to provide reliable deployment readiness assessment.
"""

from __future__ import annotations

from .base import timed


@timed
def size_score(total_bytes: int, L: int = 50_000_000, U: int = 500_000_000) -> float:
    """
    Evaluate model size efficiency for deployment across different hardware platforms.

    Uses scientifically calibrated thresholds based on real-world deployment constraints
    across edge devices, development machines, and production servers. The scoring algorithm
    rewards smaller models that enable broader deployment while penalizing oversized models
    that limit infrastructure options.

    Threshold Rationale:
    - L (50MB): Optimal size for edge deployment (Raspberry Pi, mobile)
    - U (500MB): Maximum practical size for most deployment scenarios
    - Linear scoring between thresholds balances size efficiency with capability

    Args:
        total_bytes: Model size in bytes from repository analysis
        L: Lower threshold for optimal size (default: 50MB)
        U: Upper threshold for acceptable size (default: 500MB)

    Returns:
        float: Size efficiency score [0.0, 1.0], higher is better
    """
    if U <= L:
        return 0.0
    return max(0.0, min(1.0, (U - total_bytes) / (U - L)))


@timed
def license_score(license_text: str) -> float:
    """
    Assess license compliance for commercial deployment and enterprise adoption.

    Evaluates license compatibility with business requirements, focusing on LGPL v2.1
    as the target license that balances open source benefits with commercial viability.
    The scoring reflects legal compliance requirements and deployment risk assessment.

    Business Impact:
    - LGPL v2.1: Full compliance score (1.0) - safe for commercial use
    - Unknown/Missing: Moderate risk score (0.5) - requires legal review
    - Incompatible: Zero score (0.0) - blocks commercial deployment

    Args:
        license_text: License content from model repository

    Returns:
        float: Compliance score [0.0, 1.0], higher indicates better compliance
    """
    if not license_text:
        return 0.5  # Unknown license requires legal assessment
    tex = license_text.lower()
    if "lgpl-2.1" in tex or "lgpl v2.1" in tex or "gnu lesser general public license v2.1" in tex:
        return 1.0  # Full compliance with target license
    return 0.0  # Non-compliant license blocks deployment


@timed
def rampup_score(
    readme: float, quickstart: float, tutorials: float, api_docs: float, reproducibility: float
) -> float:
    """
    Calculate team onboarding efficiency based on documentation quality assessment.

    Measures how quickly development teams can become productive with the model by
    evaluating five critical documentation dimensions. Each component is equally
    weighted as all are essential for rapid team integration and deployment success.

    Documentation Components (each scored 0.0-1.0):
    - README: Project overview and setup instructions
    - Quickstart: Immediate usage examples and workflows
    - Tutorials: Step-by-step learning materials
    - API Documentation: Technical reference materials
    - Reproducibility: Results validation guidance

    This metric directly impacts deployment timeline and team productivity, making it
    a critical factor in model selection for enterprise environments.

    Args:
        readme: README quality score [0.0, 0.5, 1.0]
        quickstart: Quick start guide quality score [0.0, 0.5, 1.0]
        tutorials: Tutorial coverage score [0.0, 0.5, 1.0]
        api_docs: API documentation completeness score [0.0, 0.5, 1.0]
        reproducibility: Reproducibility guidance score [0.0, 0.5, 1.0]

    Returns:
        float: Team ramp-up efficiency score [0.0, 1.0], higher is better
    """
    # Equal weighting ensures all documentation aspects are valued
    vals = [readme, quickstart, tutorials, api_docs, reproducibility]
    return sum(vals) / 5.0


@timed
def bus_factor_score(contributors: int, k: int = 5) -> float:
    """
    Evaluate team sustainability and knowledge distribution risk assessment.

    Implements the "bus factor" metric that measures project resilience against
    contributor loss. Uses a saturating function to reward broad contributor
    participation while avoiding over-penalization of smaller but sustainable teams.

    Algorithm Design:
    - Saturating function: contributors / (contributors + k)
    - k=5 calibrated for optimal balance between team size and sustainability
    - Approaches 1.0 asymptotically as contributor count increases
    - Provides meaningful differentiation in 1-10 contributor range

    Business Value:
    - High scores indicate sustainable maintenance and reduced dependency risk
    - Critical for long-term model lifecycle management
    - Influences deployment confidence in production environments

    Args:
        contributors: Number of active contributors from repository analysis
        k: Saturation parameter controlling score sensitivity (default: 5)

    Returns:
        float: Team sustainability score [0.0, 1.0], higher indicates better resilience
    """
    c = max(0, contributors)
    return c / (c + k) if c + k > 0 else 0.0


@timed
def dataset_and_code_score(dataset_present: bool, code_present: bool) -> float:
    """
    Assess implementation completeness through dataset and source code availability.

    Evaluates the practical usability of the model by checking for essential assets
    that enable immediate deployment and customization. Both dataset and code presence
    are equally weighted as they serve complementary purposes in model utilization.

    Scoring Logic:
    - Dataset present: Enables training data analysis and model retraining
    - Code present: Enables customization, debugging, and integration
    - Combined score: Reflects overall implementation accessibility

    Business Impact:
    - Full availability (1.0): Complete implementation ready for adaptation
    - Partial availability (0.5): Limited capability requiring additional resources
    - No availability (0.0): Unusable for practical deployment scenarios

    Args:
        dataset_present: Boolean indicating training/validation dataset availability
        code_present: Boolean indicating source code implementation availability

    Returns:
        float: Implementation completeness score [0.0, 1.0], higher is better
    """
    return (int(bool(dataset_present)) + int(bool(code_present))) / 2.0


@timed
def dataset_quality_score(source: float, license_: float, splits: float, ethics: float) -> float:
    """
    Evaluate dataset documentation quality across four critical compliance dimensions.

    Assesses dataset readiness for production deployment by examining documentation
    that addresses legal, technical, and ethical requirements. Each dimension is
    equally weighted as all are essential for responsible model deployment.

    Quality Dimensions (each scored 0.0-1.0):
    - Source: Dataset origin and provenance documentation
    - License: Usage rights and redistribution terms
    - Splits: Training/validation/test split documentation
    - Ethics: Bias assessment and responsible AI considerations

    This metric is critical for enterprise compliance and regulatory requirements,
    directly impacting deployment approval in production environments.

    Args:
        source: Dataset source documentation quality score
        license_: Dataset licensing documentation quality score
        splits: Data split documentation quality score
        ethics: Ethical considerations documentation quality score

    Returns:
        float: Dataset documentation quality score [0.0, 1.0], higher indicates better compliance
    """
    # Equal weighting ensures all compliance aspects are addressed
    return (source + license_ + splits + ethics) / 4.0


@timed
def code_quality_score(
    flake8_errors: int, isort_sorted: bool, mypy_errors: int, emax: int = 50, tmax: int = 20
) -> float:
    """
    Measure code quality through automated analysis of style, organization, and type safety.

    Implements a weighted scoring system that prioritizes the most impactful code quality
    metrics for deployment readiness. The weights are calibrated based on their correlation
    with successful production deployments and maintenance efficiency.

    Quality Components and Weights:
    - Flake8 (40%): Style consistency and common error detection
    - isort (20%): Import organization and dependency clarity
    - MyPy (40%): Type safety and runtime error prevention

    Threshold Calibration:
    - emax=50: Maximum acceptable flake8 errors for production code
    - tmax=20: Maximum acceptable type errors for deployment readiness

    This metric directly predicts maintenance cost and deployment stability,
    making it essential for long-term model lifecycle management.

    Args:
        flake8_errors: Number of style and lint violations detected
        isort_sorted: Boolean indicating proper import organization
        mypy_errors: Number of type checking violations detected
        emax: Maximum flake8 errors for full score (default: 50)
        tmax: Maximum mypy errors for full score (default: 20)

    Returns:
        float: Code quality score [0.0, 1.0], higher indicates better quality
    """
    # Calculate individual component scores with linear degradation
    flake8_score = max(0.0, 1.0 - (flake8_errors / emax)) if emax > 0 else 0.0
    isort_score = 1.0 if isort_sorted else 0.0
    mypy_score = max(0.0, 1.0 - (mypy_errors / tmax)) if tmax > 0 else 0.0

    # Weighted combination emphasizing critical quality metrics
    return 0.4 * flake8_score + 0.2 * isort_score + 0.4 * mypy_score


@timed
def perf_claims_score(benchmarks_present: bool, citations_present: bool) -> float:
    """
    Validate performance credibility through benchmark evidence and academic citations.

    Assesses the reliability of model performance claims by checking for empirical
    validation evidence. Both components are equally weighted as they provide
    complementary validation of model effectiveness and scientific rigor.

    Evidence Components:
    - Benchmarks: Empirical performance measurements on standard datasets
    - Citations: Academic references supporting methodology and claims

    Business Value:
    - High scores indicate trustworthy performance expectations
    - Critical for deployment confidence and stakeholder buy-in
    - Reduces risk of production performance disappointment

    Args:
        benchmarks_present: Boolean indicating benchmark results availability
        citations_present: Boolean indicating academic citation presence

    Returns:
        float: Performance credibility score [0.0, 1.0], higher indicates better validation
    """
    return (int(bool(benchmarks_present)) + int(bool(citations_present))) / 2.0
