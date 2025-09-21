"""
URL Classification System for Machine Learning Resources

This module provides intelligent categorization of URLs pointing to different types
of machine learning resources. Currently optimized for Hugging Face ecosystem with
extensible architecture for additional platforms like GitHub, Papers with Code, etc.

The classification system enables type-specific processing workflows and ensures
appropriate evaluation strategies are applied to different resource categories.
"""

from enum import Enum


class Category(str, Enum):
    """
    Resource type classification for ML evaluation workflows.

    Defines the primary categories of machine learning resources that require
    different evaluation approaches and scoring methodologies.
    """

    MODEL = "MODEL"  # Pre-trained models ready for inference or fine-tuning
    DATASET = "DATASET"  # Training/evaluation datasets with documentation
    CODE = "CODE"  # Source code repositories and implementations


def classify(url: str) -> Category:
    """
    Intelligently categorize ML resource URLs for appropriate processing workflows.

    Uses pattern matching to identify resource types from popular ML platforms,
    enabling type-specific evaluation strategies. Designed for high-throughput
    batch processing with consistent classification results.

    Args:
        url: URL pointing to a machine learning resource

    Returns:
        Category: Classified resource type for workflow routing

    Examples:
        >>> classify("https://huggingface.co/bert-base-uncased")
        Category.MODEL
        >>> classify("https://huggingface.co/datasets/squad")
        Category.DATASET
        >>> classify("https://github.com/user/ml-project")
        Category.CODE
    """
    u = url.lower()
    if "huggingface.co/datasets" in u:
        return Category.DATASET
    if "huggingface.co" in u:
        return Category.MODEL
    return Category.CODE
