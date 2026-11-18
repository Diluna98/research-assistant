
import json
from pathlib import Path

from equation_utils import (
    find_equation_numbers,
    select_equation_chunks
)


def load_all_chunks():
    return json.loads(
        Path("results/paper_chunks.json").read_text(
            encoding="utf-8"
        )
    )


def retrieve_evidence(question: str, top_k: int = 5):

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty")

    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("Invalid top_k")

    from retrieve_hybrid import hybrid_search

    chunks = hybrid_search(
        question.strip(),
        top_k=top_k
    )

    equation_numbers = find_equation_numbers(question)

    if equation_numbers:
        equation_chunks = select_equation_chunks(
            load_all_chunks(),
            equation_numbers
        )

        if equation_chunks:
            chunks = equation_chunks

    return [
        {
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"]
        }
        for chunk in chunks
    ]