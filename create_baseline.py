
import json
from pathlib import Path
from datetime import datetime, timezone

from workload_benchmark import benchmark_workload
from logging_config import configure_logging


def create_baseline():

    path = Path("results/workload_baseline_summary.json")

    if path.exists():
        raise FileExistsError(
            "Baseline already exists. Refusing to overwrite it."
        )

    summary = benchmark_workload(runs=5)

    baseline = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workload": {
            "prompt": (
                "Explain the difference between "
                "CPU and GPU computing."
            ),
            "temperature": 0,
            "num_predict": 200
        },
        "metrics": summary
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as file:
        json.dump(baseline, file, indent=2)

    logger = configure_logging()

    logger.info(
        "Workload baseline established",
        extra={
            "event": "workload_baseline",
            "model": summary["model"],
            "metrics": summary
        }
    )

    print("Baseline saved:", path)


if __name__ == "__main__":
    create_baseline()