"""
Additional tests for report.py functions to improve coverage.
"""

import json
from pathlib import Path

from acmecli.report import generate_summary_from_file, load_ndjson_results


def test_load_ndjson_results_success(tmp_path):
    """Test loading NDJSON results from a valid file."""
    ndjson_file = tmp_path / "results.jsonl"

    # Create test data
    test_data = [
        {"name": "model1", "net_score": 0.8, "category": "MODEL"},
        {"name": "model2", "net_score": 0.6, "category": "MODEL"},
    ]

    # Write test data to file
    with open(ndjson_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    # Test loading
    results = load_ndjson_results(str(ndjson_file))

    assert len(results) == 2
    assert results[0]["name"] == "model1"
    assert results[1]["name"] == "model2"


def test_load_ndjson_results_file_not_found(tmp_path, capsys):
    """Test loading NDJSON results from non-existent file."""
    non_existent_file = tmp_path / "does_not_exist.jsonl"

    results = load_ndjson_results(str(non_existent_file))

    assert results == []
    captured = capsys.readouterr()
    assert "Error: File" in captured.out
    assert "not found" in captured.out


def test_load_ndjson_results_invalid_json(tmp_path, capsys):
    """Test loading NDJSON results with invalid JSON."""
    ndjson_file = tmp_path / "invalid.jsonl"

    # Write invalid JSON
    with open(ndjson_file, "w") as f:
        f.write('{"name": "model1", "score": 0.8}\n')  # Valid line
        f.write('{"name": "model2", "score":}\n')  # Invalid JSON
        f.write('{"name": "model3", "score": 0.6}\n')  # Valid line

    results = load_ndjson_results(str(ndjson_file))

    # The function processes what it can before hitting the JSON error,
    # so it returns the first valid line before failing on the invalid JSON
    assert len(results) == 1
    assert results[0]["name"] == "model1"
    captured = capsys.readouterr()
    assert "Error parsing JSON:" in captured.out


def test_load_ndjson_results_empty_lines(tmp_path):
    """Test loading NDJSON results with empty lines."""
    ndjson_file = tmp_path / "with_empty_lines.jsonl"

    # Write data with empty lines
    with open(ndjson_file, "w") as f:
        f.write('{"name": "model1", "score": 0.8}\n')
        f.write("\n")  # Empty line
        f.write("  \n")  # Whitespace only line
        f.write('{"name": "model2", "score": 0.6}\n')
        f.write("\n")

    results = load_ndjson_results(str(ndjson_file))

    assert len(results) == 2
    assert results[0]["name"] == "model1"
    assert results[1]["name"] == "model2"


def test_generate_summary_from_file_success(tmp_path):
    """Test generating summary from NDJSON file."""
    ndjson_file = tmp_path / "results.jsonl"

    # Create test data with all required fields
    test_data = [
        {
            "name": "https://huggingface.co/gpt2",
            "category": "MODEL",
            "net_score": 0.8,
            "ramp_up_time": 0.9,
            "bus_factor": 0.7,
            "performance_claims": 0.8,
            "license": 1.0,
            "size_score": {
                "raspberry_pi": 0.5,
                "jetson_nano": 0.6,
                "desktop_pc": 0.9,
                "aws_server": 1.0,
            },
            "dataset_and_code_score": 0.9,
            "dataset_quality": 0.8,
            "code_quality": 0.9,
        }
    ]

    # Write test data to file
    with open(ndjson_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    # Test generating summary
    summary_file = generate_summary_from_file(str(ndjson_file))

    # Check that summary file was created
    summary_path = Path(summary_file)
    assert summary_path.exists()

    # Check summary content
    content = summary_path.read_text()
    assert "🤖 ACME MODEL EVALUATION SUMMARY REPORT" in content
    assert "gpt2" in content


def test_generate_summary_from_file_with_custom_output(tmp_path):
    """Test generating summary from NDJSON file with custom output filename."""
    ndjson_file = tmp_path / "results.jsonl"
    custom_summary = tmp_path / "custom_summary.txt"

    # Create test data
    test_data = [
        {
            "name": "https://huggingface.co/bert-base-uncased",
            "category": "MODEL",
            "net_score": 0.7,
            "ramp_up_time": 0.8,
            "bus_factor": 0.6,
            "performance_claims": 0.7,
            "license": 0.8,
            "size_score": {
                "raspberry_pi": 0.4,
                "jetson_nano": 0.5,
                "desktop_pc": 0.8,
                "aws_server": 0.9,
            },
            "dataset_and_code_score": 0.8,
            "dataset_quality": 0.7,
            "code_quality": 0.8,
        }
    ]

    # Write test data to file
    with open(ndjson_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    # Test generating summary with custom filename
    summary_file = generate_summary_from_file(str(ndjson_file), str(custom_summary))

    # Check that custom summary file was created
    assert Path(summary_file).exists()
    assert summary_file == str(custom_summary)

    # Check summary content
    content = Path(summary_file).read_text()
    assert "🤖 ACME MODEL EVALUATION SUMMARY REPORT" in content
    assert "bert-base-uncased" in content


def test_generate_summary_from_file_empty_results(tmp_path):
    """Test generating summary from empty NDJSON file."""
    ndjson_file = tmp_path / "empty_results.jsonl"

    # Create empty file
    ndjson_file.touch()

    # Test generating summary
    summary_file = generate_summary_from_file(str(ndjson_file))

    # Check that summary file was created
    summary_path = Path(summary_file)
    assert summary_path.exists()

    # Check summary content for empty case
    content = summary_path.read_text()
    assert "🤖 ACME MODEL EVALUATION SUMMARY REPORT" in content


def test_generate_summary_from_file_nonexistent_file(tmp_path):
    """Test generating summary from non-existent NDJSON file."""
    non_existent_file = tmp_path / "does_not_exist.jsonl"

    # Test generating summary from non-existent file
    summary_file = generate_summary_from_file(str(non_existent_file))

    # Should still create summary file (though with empty results)
    summary_path = Path(summary_file)
    assert summary_path.exists()

    content = summary_path.read_text()
    assert "🤖 ACME MODEL EVALUATION SUMMARY REPORT" in content
