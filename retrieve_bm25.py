
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

chunks = json.loads(
    Path("results/paper_chunks.json").read_text(
        encoding="utf-8"
    )
)

stop_words = {
    "the", "a", "an", "is", "are", "does",
    "do", "this", "that", "of", "in",
    "on", "to", "for", "and", "or",
    "what", "how", "which", "paper",
    "not", "was", "were"
}


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in stop_words]


# Build the search index
corpus = [tokenize(chunk["text"]) for chunk in chunks]
bm25 = BM25Okapi(corpus)


def search_chunks(question, top_k=5):
    query_tokens = tokenize(question)
    scores = bm25.get_scores(query_tokens)

    ranked_ids = sorted(
        range(len(chunks)),
        key=lambda i: scores[i],
        reverse=True
    )

    return [
        {**chunks[i], "score": float(scores[i])}
        for i in ranked_ids
        if scores[i] > 0
    ][:top_k]


question = input("Ask about the paper: ")

results = search_chunks(question)

for result in results:
    print(f"\nChunk {result['chunk_id']}")
    print(f"Page: {result['page']}")
    print(f"BM25 score: {result['score']:.3f}")
    print(f"Text: {result['text'][:500]}")
