
import json
import math
from pathlib import Path
import requests

MODEL = "embeddinggemma:300m"
URL = "http://localhost:11434/api/embed"

chunks = json.loads(
    Path("results/paper_chunks.json").read_text(
        encoding="utf-8"
    )
)

cache_path = Path("results/paper_embeddings_retrieval.json")


def embed(texts):
    response = requests.post(
        URL,
        json={
            "model": MODEL,
            "input": texts,
            "truncate": False,
            "keep_alive": "5m"
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# Generate and cache document embeddings once
if cache_path.exists():
    cache = json.loads(cache_path.read_text(encoding="utf-8"))

    if cache["model"] != MODEL or len(cache["embeddings"]) != len(chunks):
        raise ValueError("Embedding cache is outdated. Delete it and rerun.")

    embeddings = cache["embeddings"]

else:
    embeddings = []

    for start in range(0, len(chunks), 8):
        batch = chunks[start:start + 8]
        document_inputs = [
            f"title: none | text: {c['text']}"
            for c in batch
        ]

        vectors = embed(document_inputs)

        embeddings.extend(vectors)

        print(f"Embedded {len(embeddings)} chunks")

    cache_path.write_text(
        json.dumps({
            "model": MODEL,
            "embeddings": embeddings
        }),
        encoding="utf-8"
    )


question = input("\nAsk about the paper: ")


query_input = f"task: search result | query: {question}"

query_vector = embed([query_input])[0]


scores = [
    cosine_similarity(query_vector, vector)
    for vector in embeddings
]

ranked = sorted(
    range(len(chunks)),
    key=lambda i: scores[i],
    reverse=True
)

for i in ranked[:5]:
    print(f"\nChunk {chunks[i]['chunk_id']}")
    print(f"Page: {chunks[i]['page']}")
    print(f"Similarity: {scores[i]:.4f}")
    print(f"Text: {chunks[i]['text'][:500]}")