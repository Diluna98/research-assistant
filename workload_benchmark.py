
import statistics
from pathlib import Path
from workload_runner import run_workload
from logging_config import configure_logging
from workload_regression import compare_with_baseline

def benchmark_workload(runs=5):

    if type(runs) is not int or not 1 <= runs <= 20:
        raise ValueError("runs must be between 1 and 20")

    # Warm up the model
    run_workload()

    results = []

    for i in range(runs):

        result = run_workload()
        results.append(result)

        print(
            f"Run {i + 1}: "
            f"{result['tokens_per_second']:.2f} tokens/s, "
            f"{result['wall_time_s']:.3f} s"
        )

    speeds = [
        result["tokens_per_second"]
        for result in results
    ]

    latencies = [
        result["wall_time_s"]
        for result in results
    ]

    return {
        "model": results[0]["model"],
        "runs": runs,
        "average_speed": statistics.mean(speeds),
        "speed_std": (
            statistics.stdev(speeds)
            if runs > 1 else 0.0
        ),
        "average_latency": statistics.mean(latencies),
        "median_latency": statistics.median(latencies),
        "min_latency": min(latencies),
        "max_latency": max(latencies)
    }


if __name__ == "__main__":

    summary = benchmark_workload()

    logger = configure_logging()

    logger.info(
        "Workload benchmark completed",
        extra={
            "event": "workload_benchmark",
            "model": summary["model"],
            "metrics": summary
        }
    )


    baseline_path = Path(
        "results/workload_baseline_summary.json"
    )

    if baseline_path.exists():

        comparison = compare_with_baseline(
            summary,
            baseline_path=baseline_path
        )

        print("\nPerformance comparison:")

        for key, value in comparison.items():
            print(f"{key}: {value}")

        logger.info(
            "Workload performance comparison completed",
            extra={
                "event": "workload_comparison",
                "model": summary["model"],
                "metrics": comparison
            }
        )

    else:
        print("\nNo baseline available for comparison.")


    print("\nBenchmark summary:")

    for key, value in summary.items():
        print(f"{key}: {value}")
