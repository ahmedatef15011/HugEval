"""
Additional tests for LLM analysis functions to improve coverage.
"""

import json
import os
from unittest.mock import Mock, patch

from acmecli.llm_analysis import _call_openai_api, enhance_ramp_up_time_with_llm


def test_call_openai_api_with_api_key():
    """Test _call_openai_api with valid API key."""
    readme_content = "This is a comprehensive README with examples and setup instructions."
    model_name = "test-model"

    # Mock OpenAI response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps(
        {"documentation_quality": 0.9, "ease_of_use": 0.8, "examples_present": True}
    )

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI", return_value=mock_client):
            result = _call_openai_api(readme_content, model_name)

    # Should combine LLM results with local analysis
    assert "documentation_quality" in result
    assert "ease_of_use" in result
    assert "examples_present" in result
    # Local analysis keys should also be present
    assert "installation_instructions" in result
    assert "usage_examples" in result


def test_call_openai_api_no_api_key():
    """Test _call_openai_api falls back to local analysis when no API key."""
    readme_content = "This is a test README."
    model_name = "test-model"

    with patch.dict(os.environ, {}, clear=True):  # No OPENAI_API_KEY
        result = _call_openai_api(readme_content, model_name)

    # Should return local analysis results
    assert "examples_present" in result
    assert "usage_examples" in result
    assert "installation_instructions" in result


def test_call_openai_api_empty_response():
    """Test _call_openai_api handles empty OpenAI response."""
    readme_content = "Test README content."
    model_name = "test-model"

    # Mock OpenAI response with None content
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = None

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI", return_value=mock_client):
            result = _call_openai_api(readme_content, model_name)

    # Should fall back to local analysis
    assert "examples_present" in result
    assert "usage_examples" in result


def test_call_openai_api_json_parse_error():
    """Test _call_openai_api handles JSON parsing errors."""
    readme_content = "Test README content."
    model_name = "test-model"

    # Mock OpenAI response with invalid JSON
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Invalid JSON response"

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI", return_value=mock_client):
            result = _call_openai_api(readme_content, model_name)

    # Should fall back to local analysis
    assert "examples_present" in result
    assert "usage_examples" in result


def test_call_openai_api_import_error():
    """Test _call_openai_api handles missing openai package."""
    readme_content = "Test README content."
    model_name = "test-model"

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("builtins.__import__", side_effect=ImportError("No module named openai")):
            result = _call_openai_api(readme_content, model_name)

    # Should fall back to local analysis
    assert "examples_present" in result
    assert "usage_examples" in result


def test_call_openai_api_client_creation_error():
    """Test _call_openai_api handles OpenAI client creation errors."""
    readme_content = "Test README content."
    model_name = "test-model"

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI", side_effect=Exception("Client creation failed")):
            result = _call_openai_api(readme_content, model_name)

    # Should fall back to local analysis
    assert "examples_present" in result
    assert "usage_examples" in result


def test_call_openai_api_request_error():
    """Test _call_openai_api handles API request errors."""
    readme_content = "Test README content."
    model_name = "test-model"

    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("API request failed")

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI", return_value=mock_client):
            result = _call_openai_api(readme_content, model_name)

    # Should fall back to local analysis
    assert "examples_present" in result
    assert "usage_examples" in result


def test_enhance_ramp_up_time_with_llm_success():
    """Test enhance_ramp_up_time_with_llm with successful LLM analysis."""
    base_score = 0.6
    readme_content = "Comprehensive README with detailed setup instructions and examples."
    model_name = "test-model"

    # Mock OpenAI response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = json.dumps({"ease_of_use_score": 0.9})

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI", return_value=mock_client):
            result = enhance_ramp_up_time_with_llm(base_score, readme_content, model_name)

    # Should be enhanced score
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_enhance_ramp_up_time_with_llm_fallback():
    """Test enhance_ramp_up_time_with_llm with no API key - uses local analysis."""
    base_score = 0.6
    readme_content = "Test README content."
    model_name = "test-model"

    with patch.dict(os.environ, {}, clear=True):  # No API key
        result = enhance_ramp_up_time_with_llm(base_score, readme_content, model_name)

    # Should return enhanced score using local analysis (not just base score)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_enhance_ramp_up_time_with_llm_api_error():
    """Test enhance_ramp_up_time_with_llm handles API errors gracefully."""
    base_score = 0.7
    readme_content = "Test README content."
    model_name = "test-model"

    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("API error")

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI", return_value=mock_client):
            result = enhance_ramp_up_time_with_llm(base_score, readme_content, model_name)

    # Should return enhanced score using local fallback analysis
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_enhance_ramp_up_time_with_llm_invalid_response():
    """Test enhance_ramp_up_time_with_llm handles invalid API responses."""
    base_score = 0.5
    readme_content = "Test README content."
    model_name = "test-model"

    # Mock OpenAI response with invalid JSON
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Not valid JSON"

    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("openai.OpenAI", return_value=mock_client):
            result = enhance_ramp_up_time_with_llm(base_score, readme_content, model_name)

    # Should return enhanced score using local fallback analysis
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
