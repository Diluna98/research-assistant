
import json
import re
import math
import requests
from config import load_config
from pathlib import Path
from rank_bm25 import BM25Okapi

# Load existing document chunks and embeddings
chunks = json.loads(
    Path("results/paper_chunks.json").read_text(
        encoding="utf-8"
    )
)

cache = json.loads(
    Path("results/paper_embeddings_retrieval.json").read_text(
        encoding="utf-8"
    )
)

embeddings = cache["embeddings"]

if len(embeddings) != len(chunks):
    raise ValueError("Embedding index does not match chunks")


# BM25 preparation
stop_words = {
    "the", "a", "an", "is", "are", "does",
    "do", "this", "that", "of", "in",
    "on", "to", "for", "and", "or",
    "what", "how", "which", "paper",
    "not", "was", "were"
}


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in stop_words]


corpus = [tokenize(c["text"]) for c in chunks]
bm25 = BM25Okapi(corpus)


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_scores(question):
    response = requests.post(
        load_config()["ollama_url"].rstrip("/") + "/api/embed",
        json={
            "model": cache["model"],
            "input": [
                f"task: search result | query: {question}"
            ]
        },
        timeout=180
    )
    response.raise_for_status()

    query_vector = response.json()["embeddings"][0]

    return [
        cosine_similarity(query_vector, v)
        for v in embeddings
    ]


def hybrid_search(question, top_k=5):
    bm_scores = bm25.get_scores(tokenize(question))
    sem_scores = semantic_scores(question)

    bm_ranking = sorted(
        (i for i in range(len(chunks)) if bm_scores[i] > 0),
        key=lambda i: bm_scores[i],
        reverse=True
    )[:10]

    sem_ranking = sorted(
        range(len(chunks)),
        key=lambda i: sem_scores[i],
        reverse=True
    )[:10]

    # Reciprocal Rank Fusion
    fused = {}

    for ranking in [bm_ranking, sem_ranking]:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0) + 1 / (60 + rank)

    final_ranking = sorted(
        fused,
        key=lambda i: fused[i],
        reverse=True
    )

    return [
        {**chunks[i], "fusion_score": fused[i]}
        for i in final_ranking[:top_k]
    ]



if __name__ == "__main__":

    question = input("Ask about the paper: ")

    results = hybrid_search(question)

    for result in results:
        print(f"\nChunk {result['chunk_id']}")
        print(f"Page: {result['page']}")
        print(f"Fusion score: {result['fusion_score']:.5f}")
        print(f"Text: {result['text'][:500]}")