
import json
import re
from pathlib import Path

index_path = Path("results/paper_chunks.json")

chunks = json.loads(
    index_path.read_text(encoding="utf-8")
)

stop_words = {
    "the", "a", "an", "is", "are", "does",
    "do", "this", "that", "of", "in",
    "on", "to", "for", "and", "or",
    "what", "how", "which", "paper"
}


def tokenize(text):
    words = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return set(words) - stop_words


def search_chunks(question, top_k=3):

    query_words = tokenize(question)

    results = []

    for chunk in chunks:

        chunk_words = tokenize(chunk["text"])

        matching_words = query_words.intersection(chunk_words)
        score = len(matching_words)

        if score > 0:
            results.append({
                **chunk,
                "score": score,
                "matches": sorted(matching_words)
            })

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results[:top_k]


question = input("Ask about the paper: ")

results = search_chunks(question)

for result in results:

    print(f"\nChunk {result['chunk_id']}")
    print(f"Page: {result['page']}")
    print(f"Score: {result['score']}")
    print(f"Matched words: {result['matches']}")
    print(f"Text: {result['text'][:600]}")
