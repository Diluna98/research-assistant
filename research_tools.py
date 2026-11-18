
from research_retrieval import retrieve_evidence


def search_paper(question: str, top_k: int = 5):

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty")

    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")

    question = question.strip()

    chunks = retrieve_evidence(
        question,
        top_k=top_k
    )

    return {
        "question": question,
        "results": chunks
    }