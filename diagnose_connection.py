
import requests
from time import perf_counter

tests = [
    ("Original localhost", "http://localhost:11434", True),
    ("Explicit IPv4", "http://127.0.0.1:11434", True),
    ("IPv4 without proxy discovery", "http://127.0.0.1:11434", False)
]

for name, url, trust_env in tests:

    print(f"\n{name}")

    with requests.Session() as session:

        session.trust_env = trust_env

        for i in range(3):
            start = perf_counter()

            response = session.get(
                url + "/api/tags",
                timeout=15
            )

            response.raise_for_status()

            elapsed = perf_counter() - start

            print(f"Run {i + 1}: {elapsed:.4f} s")
