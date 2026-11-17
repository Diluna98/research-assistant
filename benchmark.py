
import statistics
import requests

import json
from pathlib import Path
from datetime import datetime, timezone


URL = "http://localhost:11434/api/chat"

payload = {
    "model": "qwen3:4b",
    "messages": [
        {
            "role": "user",
            "content": "Explain the difference between CPU and GPU computing."
        }
    ],
    "stream": False,
    "think": False,
    "keep_alive": "10m",
    "options": {
        "temperature": 0,
        "num_predict": 200
    }
}


def run_inference():
    response = requests.post(URL, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()

    total = data["total_duration"] / 1e9
    loading = data["load_duration"] / 1e9
    prompt = data["prompt_eval_duration"] / 1e9
    generation = data["eval_duration"] / 1e9
    tokens = data["eval_count"]

    speed = tokens / generation if generation > 0 else 0

    return {
        "total": total,
        "loading": loading,
        "prompt": prompt,
        "generation": generation,
        "tokens": tokens,
        "speed": speed
    }


print("Warming up the model...")
run_inference()


results_path = Path("results") / "benchmark_runs.jsonl"

results_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

results = []

for i in range(5):
    result = run_inference()
    results.append(result)


    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run": i + 1,
        "model": payload["model"],
        "temperature": payload["options"]["temperature"],
        "num_predict": payload["options"]["num_predict"],
        "prompt": payload["messages"][0]["content"],
        **result
    }

    with results_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


    print(
        f"Run {i + 1}: "
        f"{result['total']:.2f} s, "
        f"{result['tokens']} tokens, "
        f"{result['speed']:.2f} tokens/s"
    )

speeds = [result["speed"] for result in results]
durations = [result["total"] for result in results]

print("\nBenchmark summary:")
print(f"Average generation speed: {statistics.mean(speeds):.2f} tokens/s")
print(f"Speed standard deviation: {statistics.stdev(speeds):.2f} tokens/s")
print(f"Average total duration: {statistics.mean(durations):.2f} seconds")
