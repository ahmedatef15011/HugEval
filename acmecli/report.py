"""
Human-readable report generation for model evaluation results.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_model_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse and analyze model results for summary generation."""
    if not results:
        return {"total_models": 0, "models": []}

    # Sort models by net score (highest first)
    sorted_models = sorted(results, key=lambda x: x.get("net_score", 0), reverse=True)

    # Calculate statistics
    net_scores = [model.get("net_score", 0) for model in results]
    avg_score = sum(net_scores) / len(net_scores)

    # Categorize models
    excellent_models = [m for m in results if m.get("net_score", 0) >= 0.8]
    good_models = [m for m in results if 0.6 <= m.get("net_score", 0) < 0.8]
    acceptable_models = [m for m in results if 0.4 <= m.get("net_score", 0) < 0.6]
    poor_models = [m for m in results if m.get("net_score", 0) < 0.4]

    # License compliance
    compliant_models = [m for m in results if m.get("license", 0) >= 1.0]
    non_compliant_models = [m for m in results if m.get("license", 0) < 1.0]

    # Device compatibility
    raspberry_pi_compatible = [
        m for m in results if m.get("size_score", {}).get("raspberry_pi", 0) > 0.5
    ]
    desktop_compatible = [m for m in results if m.get("size_score", {}).get("desktop_pc", 0) > 0.5]

    return {
        "total_models": len(results),
        "models": sorted_models,
        "statistics": {
            "average_score": avg_score,
            "highest_score": max(net_scores),
            "lowest_score": min(net_scores),
        },
        "categories": {
            "excellent": len(excellent_models),
            "good": len(good_models),
            "acceptable": len(acceptable_models),
            "poor": len(poor_models),
        },
        "compliance": {
            "lgpl_compliant": len(compliant_models),
            "non_compliant": len(non_compliant_models),
        },
        "device_compatibility": {
            "raspberry_pi": len(raspberry_pi_compatible),
            "desktop_pc": len(desktop_compatible),
        },
        "top_models": sorted_models[:5],  # Top 5 models
        "compliant_models": compliant_models,
        "raspberry_pi_models": raspberry_pi_compatible,
    }


def extract_model_name(url: str) -> str:
    """Extract a clean model name from Hugging Face URL."""
    if "huggingface.co/" in url:
        return url.split("/")[-1] if url.endswith("/") else url.split("/")[-1]
    return url


def format_score(score: float) -> str:
    """Format score as percentage with rating."""
    percentage = score * 100
    if percentage >= 80:
        return f"{percentage:.1f}% (Excellent)"
    elif percentage >= 60:
        return f"{percentage:.1f}% (Good)"
    elif percentage >= 40:
        return f"{percentage:.1f}% (Acceptable)"
    else:
        return f"{percentage:.1f}% (Poor)"


def generate_summary_report(
    results: List[Dict[str, Any]], output_file: str = "model_evaluation_summary.txt"
) -> str:
    """Generate a human-readable summary report."""

    analysis = parse_model_results(results)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = []

    # Header
    report_lines.extend(
        [
            "=" * 80,
            "🤖 ACME MODEL EVALUATION SUMMARY REPORT",
            "=" * 80,
            f"Generated: {timestamp}",
            f"Total Models Evaluated: {analysis['total_models']}",
            "",
        ]
    )

    # Executive Summary
    if analysis["total_models"] > 0:
        avg_score = analysis["statistics"]["average_score"]
        report_lines.extend(
            [
                "📊 EXECUTIVE SUMMARY",
                "-" * 40,
                f"Average Quality Score: {format_score(avg_score)}",
                f"Highest Score: {format_score(analysis['statistics']['highest_score'])}",
                f"Lowest Score: {format_score(analysis['statistics']['lowest_score'])}",
                "",
                "📈 QUALITY DISTRIBUTION:",
                f"  🟢 Excellent (≥80%): {analysis['categories']['excellent']} models",
                f"  🟡 Good (60-79%):     {analysis['categories']['good']} models",
                f"  🟠 Acceptable (40-59%): {analysis['categories']['acceptable']} models",
                f"  🔴 Poor (<40%):       {analysis['categories']['poor']} models",
                "",
            ]
        )

        # License Compliance
        report_lines.extend(
            [
                "⚖️  LICENSE COMPLIANCE",
                "-" * 40,
                f"✅ LGPL-2.1 Compliant: {analysis['compliance']['lgpl_compliant']} models",
                f"❌ Non-Compliant:      {analysis['compliance']['non_compliant']} models",
                "",
            ]
        )

        # Device Compatibility
        report_lines.extend(
            [
                "💻 DEVICE COMPATIBILITY",
                "-" * 40,
                (
                    f"🥧 Raspberry Pi Compatible: "
                    f"{analysis['device_compatibility']['raspberry_pi']} models"
                ),
                (
                    f"🖥️  Desktop PC Compatible:   "
                    f"{analysis['device_compatibility']['desktop_pc']} models"
                ),
                "",
            ]
        )

        # Top Models Ranking
        if analysis["top_models"]:
            report_lines.extend(
                [
                    "🏆 TOP MODELS RANKING",
                    "-" * 40,
                ]
            )

            for i, model in enumerate(analysis["top_models"], 1):
                model_name = extract_model_name(model["name"])
                net_score = format_score(model.get("net_score", 0))
                license_status = "✅ LGPL" if model.get("license", 0) >= 1.0 else "❌ Other"

                report_lines.extend(
                    [
                        f"{i}. {model_name}",
                        f"   Score: {net_score}",
                        f"   License: {license_status}",
                        f"   URL: {model['name']}",
                        "",
                    ]
                )

        # Recommendations
        report_lines.extend(
            [
                "💡 RECOMMENDATIONS",
                "-" * 40,
            ]
        )

        if analysis["compliance"]["lgpl_compliant"] > 0:
            best_compliant = max(analysis["compliant_models"], key=lambda x: x.get("net_score", 0))
            model_name = extract_model_name(best_compliant["name"])
            score_text = format_score(best_compliant.get("net_score", 0))
            report_lines.append(f"🎯 Best LGPL-Compliant Model: {model_name} ({score_text})")
        else:
            report_lines.append(
                "⚠️  No LGPL-2.1 compliant models found. Consider license implications."
            )

        if analysis["device_compatibility"]["raspberry_pi"] > 0:
            pi_count = analysis["device_compatibility"]["raspberry_pi"]
            report_lines.append(f"🥧 {pi_count} models are suitable for Raspberry Pi deployment")
        else:
            report_lines.append("⚠️  No models suitable for Raspberry Pi deployment found.")

        if avg_score < 0.6:
            report_lines.append(
                "⚠️  Overall model quality is below recommended threshold. "
                "Consider alternative models."
            )

        report_lines.extend(
            [
                "",
                "📋 DETAILED METRICS EXPLANATION",
                "-" * 40,
                "• Net Score: Overall quality (weighted average of all metrics)",
                "• License: LGPL-2.1 compatibility (1.0 = fully compatible)",
                "• Size Score: Model size suitability for different devices",
                "• Ramp Up Time: Documentation and ease-of-use quality",
                "• Bus Factor: Project sustainability and team size",
                "• Code Quality: Static analysis and coding standards",
                "",
            ]
        )

    else:
        report_lines.extend(
            [
                "❌ No models were successfully evaluated.",
                "Please check your input URLs and network connection.",
                "",
            ]
        )

    # Footer
    report_lines.extend(
        [
            "=" * 80,
            "🔗 For detailed JSON data, see the NDJSON output files.",
            "🛠️  Generated by ACME Model Scoring CLI",
            "=" * 80,
        ]
    )

    # Write to file
    report_content = "\n".join(report_lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    return output_file


def load_ndjson_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from NDJSON file."""
    results = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")

    return results


def generate_summary_from_file(ndjson_file: str, summary_file: Optional[str] = None) -> str:
    """Generate summary report from existing NDJSON file."""
    if summary_file is None:
        base_name = Path(ndjson_file).stem
        summary_file = f"{base_name}_summary.txt"

    results = load_ndjson_results(ndjson_file)
    return generate_summary_report(results, summary_file)


def capture_and_summarize_results(
    results: List[Dict[str, Any]], base_filename: str = "evaluation"
) -> tuple[str, str]:
    """Capture results and generate both NDJSON and summary files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save NDJSON results
    ndjson_file = f"{base_filename}_{timestamp}.jsonl"
    with open(ndjson_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Generate summary
    summary_file = f"{base_filename}_{timestamp}_summary.txt"
    generate_summary_report(results, summary_file)

    return ndjson_file, summary_file
