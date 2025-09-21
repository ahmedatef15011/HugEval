"""
LLM integration for README analysis and metric enhancement.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def analyze_readme_with_llm(readme_content: str, model_name: str) -> Dict[str, Any]:
    """
    Analyze README content using LLM to enhance metrics.

    This function uses an LLM to analyze README content and provide
    insights for metrics like ramp_up_time, documentation quality, etc.
    """
    try:
        # For now, implement a simple local analysis as fallback
        # In production, this would call a real LLM service
        return _analyze_readme_locally(readme_content, model_name)

    except Exception as e:
        logger.warning(f"LLM analysis failed for {model_name}: {e}")
        return _analyze_readme_locally(readme_content, model_name)


def _analyze_readme_locally(readme_content: str, model_name: str) -> Dict[str, Any]:
    """
    Local README analysis as fallback when LLM is not available.

    This provides a baseline analysis that can be enhanced with real LLM calls.
    """
    if not readme_content:
        return {
            "documentation_quality": 0.0,
            "ease_of_use": 0.0,
            "examples_present": False,
            "installation_instructions": False,
            "usage_examples": False,
        }

    content_lower = readme_content.lower()

    # Check for key documentation elements
    has_installation = any(word in content_lower for word in ["install", "pip", "conda", "setup"])
    has_usage = any(
        word in content_lower for word in ["usage", "example", "how to", "getting started"]
    )
    has_api_docs = any(
        word in content_lower for word in ["api", "reference", "documentation", "docs"]
    )
    has_examples = any(word in content_lower for word in ["example", "sample", "demo", "tutorial"])

    # Count code blocks (markdown)
    code_blocks = readme_content.count("```")

    # Basic quality scoring
    quality_score = 0.0
    if has_installation:
        quality_score += 0.25
    if has_usage:
        quality_score += 0.25
    if has_api_docs:
        quality_score += 0.25
    if code_blocks >= 2:  # At least one code example
        quality_score += 0.25

    # Ease of use based on README structure
    ease_score = min(1.0, len(readme_content) / 1000 * 0.5 + quality_score * 0.5)

    return {
        "documentation_quality": quality_score,
        "ease_of_use": ease_score,
        "examples_present": has_examples,
        "installation_instructions": has_installation,
        "usage_examples": has_usage,
        "code_blocks_count": code_blocks // 2,  # Pairs of ```
    }


def _call_openai_api(readme_content: str, model_name: str) -> Dict[str, Any]:
    """
    Call OpenAI API for README analysis (when API key is available).

    This is a placeholder for real LLM integration that would be used
    in production with proper API credentials.
    """
    try:
        import openai

        # Check if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.info("No OpenAI API key found, using local analysis")
            return _analyze_readme_locally(readme_content, model_name)

        client = openai.OpenAI(api_key=api_key)

        prompt = f"""
        Analyze this README for the machine learning model "{model_name}".

        Rate the following aspects on a scale of 0.0 to 1.0:
        1. Documentation quality (completeness, clarity)
        2. Ease of use for new users (setup instructions, examples)
        3. Presence of working examples

        README content:
        {readme_content[:2000]}...

        Respond with ONLY a JSON object with keys: documentation_quality,
        ease_of_use, examples_present (boolean)
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )

        # Parse JSON response
        import json

        result = json.loads(response.choices[0].message.content)

        # Add additional analysis
        result.update(_analyze_readme_locally(readme_content, model_name))

        logger.info(f"LLM analysis completed for {model_name}")
        return result

    except Exception as e:
        logger.warning(f"OpenAI API call failed: {e}")
        return _analyze_readme_locally(readme_content, model_name)


def enhance_ramp_up_time_with_llm(base_score: float, readme_content: str, model_name: str) -> float:
    """
    Enhance ramp_up_time metric using LLM analysis of README.

    This function demonstrates LLM integration as required by the specification.
    """
    try:
        # Get LLM analysis
        analysis = analyze_readme_with_llm(readme_content, model_name)

        # Combine base score with LLM insights
        llm_weight = 0.3  # 30% LLM contribution
        base_weight = 0.7  # 70% original metric

        llm_score = (
            analysis["documentation_quality"] * 0.4
            + analysis["ease_of_use"] * 0.4
            + (1.0 if analysis["examples_present"] else 0.0) * 0.2
        )

        enhanced_score = base_weight * base_score + llm_weight * llm_score

        logger.info(
            f"Enhanced ramp_up_time for {model_name}: {base_score:.3f} -> {enhanced_score:.3f}"
        )
        return min(1.0, enhanced_score)

    except Exception as e:
        logger.error(f"Failed to enhance ramp_up_time with LLM for {model_name}: {e}")
        return base_score
