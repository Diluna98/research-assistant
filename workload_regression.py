
import json
from pathlib import Path


def compare_with_baseline(
    current,
    baseline_path="results/workload_baseline_summary.json",
    tolerance=0.05
):
    baseline = json.loads(
        Path(baseline_path).read_text(encoding="utf-8")
    )

    reference = baseline["metrics"]

    if current["model"] != reference["model"]:
        raise ValueError("Model mismatch")

    if current["runs"] != reference["runs"]:
        raise ValueError("Benchmark run count mismatch")

    baseline_speed = reference["average_speed"]
    baseline_latency = reference["average_latency"]

    if baseline_speed <= 0 or baseline_latency <= 0:
        raise ValueError("Invalid baseline metrics")

    speed_change = (
        (current["average_speed"] - baseline_speed)
        / baseline_speed
    ) * 100

    latency_change = (
        (current["average_latency"] - baseline_latency)
        / baseline_latency
    ) * 100

    regression = (
        speed_change < -tolerance * 100
        or latency_change > tolerance * 100
    )

    return {
        "baseline_speed": baseline_speed,
        "current_speed": current["average_speed"],
        "speed_change_percent": round(speed_change, 2),
        "baseline_latency": baseline_latency,
        "current_latency": current["average_latency"],
        "latency_change_percent": round(latency_change, 2),
        "status": (
            "possible_regression"
            if regression
            else "within_tolerance"
        )
    }