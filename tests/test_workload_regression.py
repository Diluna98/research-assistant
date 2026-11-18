
import json

from workload_regression import compare_with_baseline


def test_detects_performance_regression(tmp_path):

    baseline = {
        "metrics": {
            "model": "qwen3:4b-instruct",
            "runs": 5,
            "average_speed": 60.0,
            "average_latency": 3.5
        }
    }

    baseline_file = tmp_path / "baseline.json"

    baseline_file.write_text(
        json.dumps(baseline),
        encoding="utf-8"
    )

    current = {
        "model": "qwen3:4b-instruct",
        "runs": 5,
        "average_speed": 54.0,
        "average_latency": 3.9
    }

    result = compare_with_baseline(
        current,
        baseline_path=baseline_file
    )

    assert result["status"] == "possible_regression"
    assert result["speed_change_percent"] == -10.0


def test_accepts_normal_variation(tmp_path):

    baseline = {
        "metrics": {
            "model": "qwen3:4b-instruct",
            "runs": 5,
            "average_speed": 60.0,
            "average_latency": 3.5
        }
    }

    baseline_file = tmp_path / "baseline.json"

    baseline_file.write_text(
        json.dumps(baseline),
        encoding="utf-8"
    )

    current = {
        "model": "qwen3:4b-instruct",
        "runs": 5,
        "average_speed": 59.0,
        "average_latency": 3.55
    }

    result = compare_with_baseline(
        current,
        baseline_path=baseline_file
    )

    assert result["status"] == "within_tolerance"