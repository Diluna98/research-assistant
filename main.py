
import requests

url = "http://localhost:11434/api/chat"

question = "Explain the difference between CPU and GPU computing."


payload = {
    "model": "qwen3:4b",
    "messages": [
        {"role": "user", "content": question}
    ],
    "stream": False,
    "think": False,
    "options": {
        "temperature": 0,
        "num_predict": 200
    }
}

response = requests.post(url, json=payload, timeout=180)
response.raise_for_status()

data = response.json()

answer = data["message"]["content"]

print("\nAI response:")
print(answer)


print("\nPerformance statistics:")

total_time = data["total_duration"] / 1_000_000_000
generation_time = data["eval_duration"] / 1_000_000_000
generated_tokens = data["eval_count"]

print(f"Total duration: {total_time:.2f} seconds")
print(f"Generated tokens: {generated_tokens}")

if generation_time > 0:
    tokens_per_second = generated_tokens / generation_time
    print(f"Generation speed: {tokens_per_second:.2f} tokens/s")

print("\nDetailed performance breakdown:")

load_time = data["load_duration"] / 1_000_000_000

prompt_time = data["prompt_eval_duration"] / 1_000_000_000

generation_time = data["eval_duration"] / 1_000_000_000

prompt_tokens = data["prompt_eval_count"]

print(f"Model loading: {load_time:.4f} s")
print(f"Prompt processing: {prompt_time:.4f} s")
print(f"Token generation: {generation_time:.4f} s")
print(f"Prompt tokens: {prompt_tokens}")

other_time = total_time - load_time - prompt_time - generation_time

print(f"Other overhead: {other_time:.4f} s")
print(f"Stop reason: {data.get('done_reason')}")
