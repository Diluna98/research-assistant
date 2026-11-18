
from workload_benchmark import benchmark_workload
from workload_regression import compare_with_baseline
from logging_config import configure_logging


def evaluate_workload():

    # Execute one warmup and five measured runs
    summary = benchmark_workload(runs=5)

    # Compare against the existing baseline
    comparison = compare_with_baseline(summary)

    # Record the evaluation
    logger = configure_logging()

    logger.info(
        "Workload evaluation completed",
        extra={
            "event": "workload_evaluation",
            "model": summary["model"],
            "metrics": {
                "benchmark": summary,
                "comparison": comparison
            }
        }
    )

    return {
        "benchmark": summary,
        "comparison": comparison
    }