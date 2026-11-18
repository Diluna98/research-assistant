
from time import perf_counter
from config import load_config
from ollama_client import post_chat


def run_workload():

    config = load_config()

    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain the difference between "
                    "CPU and GPU computing."
                )
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

    start = perf_counter()

    response = post_chat(payload)

    wall_time = perf_counter() - start

    generation_time = (
        response["eval_duration"] / 1e9
    )

    tokens = response["eval_count"]

    speed = (
        tokens / generation_time
        if generation_time > 0
        else 0
    )

    return {
        "model": config["model"],
        "wall_time_s": wall_time,
        "total_duration_s": (
            response["total_duration"] / 1e9
        ),
        "generation_time_s": generation_time,
        "tokens": tokens,
        "tokens_per_second": speed,
        "stop_reason": response.get("done_reason")
    }