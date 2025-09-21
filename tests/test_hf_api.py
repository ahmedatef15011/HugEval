from unittest.mock import Mock, patch

import requests

from acmecli.metrics.hf_api import (
    build_context_from_api,
    calculate_model_size,
    extract_model_id,
    fetch_model_info,
    freshness_days_since_update,
    get_model_downloads,
    get_model_license,
    popularity_downloads_likes,
)


def test_popularity_extremes():
    s0, _ = popularity_downloads_likes(0, 0)
    s1, _ = popularity_downloads_likes(1_000_000, 100_000)
    assert 0 <= s0 <= 1 and 0 <= s1 <= 1
    assert s1 >= s0


def test_freshness_days_since_update():
    fresh, _ = freshness_days_since_update(0)
    stale, _ = freshness_days_since_update(365)
    assert fresh == 1.0
    assert 0 <= stale <= 1
    assert fresh >= stale


def test_extract_model_id():
    """Test extracting model ID from various URL formats."""
    assert extract_model_id("https://huggingface.co/gpt2") == "gpt2"
    assert extract_model_id("https://huggingface.co/bert-base-uncased") == "bert-base-uncased"
    assert (
        extract_model_id("https://huggingface.co/microsoft/DialoGPT-medium")
        == "microsoft/DialoGPT-medium"
    )


def test_calculate_model_size():
    """Test model size calculation from file data."""
    files_data = [
        {"size": 1000},
        {"size": 2000},
        {"other_field": "value"},  # No size field
        {"size": 500},
    ]
    total_size = calculate_model_size(files_data)
    assert total_size == 3500


def test_get_model_downloads():
    """Test download count extraction with type checking."""
    assert get_model_downloads({"downloads": 1000}) == 1000
    assert get_model_downloads({"downloads": "not_int"}) == 0
    assert get_model_downloads({}) == 0


def test_get_model_license():
    """Test license extraction from various sources."""
    # License in cardData
    model_info = {"cardData": {"license": "apache-2.0"}}
    assert get_model_license(model_info) == "apache-2.0"

    # License in direct field
    model_info = {"license": "mit"}
    assert get_model_license(model_info) == "mit"

    # License in tags
    model_info = {"tags": ["license:lgpl-2.1", "other-tag"]}
    assert get_model_license(model_info) == "license:lgpl-2.1"

    # No license found
    model_info = {}
    assert get_model_license(model_info) == ""


@patch("acmecli.metrics.hf_api.requests.get")
def test_fetch_model_info_success(mock_get):
    """Test successful model info fetching."""
    mock_response = Mock()
    mock_response.json.return_value = {"name": "gpt2", "downloads": 1000}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_model_info("gpt2")
    assert result == {"name": "gpt2", "downloads": 1000}


@patch("acmecli.metrics.hf_api.requests.get")
def test_fetch_model_info_failure(mock_get):
    """Test model info fetching with API failure."""
    mock_get.side_effect = requests.RequestException("API Error")

    result = fetch_model_info("nonexistent-model")
    assert result == {}


@patch("acmecli.metrics.hf_api.fetch_model_info")
@patch("acmecli.metrics.hf_api.fetch_model_files")
def test_build_context_from_api_success(mock_fetch_files, mock_fetch_info):
    """Test building context from API with successful responses."""
    mock_fetch_info.return_value = {
        "downloads": 1000,
        "likes": 50,
        "lastModified": "2025-09-01T00:00:00Z",
    }
    mock_fetch_files.return_value = [{"size": 1000}, {"size": 2000}]

    context = build_context_from_api("https://huggingface.co/gpt2")

    assert context["downloads"] == 1000
    assert context["likes"] == 50
    assert context["total_bytes"] == 3000
    assert "days_since_update" in context


@patch("acmecli.metrics.hf_api.fetch_model_info")
def test_build_context_from_api_failure(mock_fetch_info):
    """Test building context when API calls fail."""
    mock_fetch_info.return_value = {}

    context = build_context_from_api("https://huggingface.co/nonexistent")

    # Should return fallback context
    assert "total_bytes" in context
    assert "downloads" in context
