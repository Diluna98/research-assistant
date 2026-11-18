
import statistics
import json
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
from config import load_config
from ollama_client import post_chat

payload = {
    "model": load_config()["model"],
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
    start = perf_counter()

    data = post_chat(payload)

    wall_time = perf_counter() - start

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
        "wall_time": wall_time,
        "speed": speed
    }


print("Warming up the model...")
run_inference()


results_path = Path("results") / "workload_baseline.jsonl"

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
wall_times = [result["wall_time"] for result in results]

print("\nBenchmark summary:")
print(f"Average generation speed: {statistics.mean(speeds):.2f} tokens/s")
print(f"Speed standard deviation: {statistics.stdev(speeds):.2f} tokens/s")
print(f"Average total duration: {statistics.mean(durations):.2f} seconds")
print(
    f"Average wall time: "
    f"{statistics.mean(wall_times):.3f} seconds"
)
