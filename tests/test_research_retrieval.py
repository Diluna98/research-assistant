
import sys
from types import SimpleNamespace

import research_retrieval


def test_general_question(monkeypatch):

    def fake_search(question, top_k):
        assert question == "How are pedestrians detected?"
        assert top_k == 3

        return [{
            "page": 4,
            "chunk_id": 22,
            "text": "Pedestrian detection evidence."
        }]

    monkeypatch.setitem(
        sys.modules,
        "retrieve_hybrid",
        SimpleNamespace(hybrid_search=fake_search)
    )

    result = research_retrieval.retrieve_evidence(
        "How are pedestrians detected?",
        top_k=3
    )

    assert result[0]["chunk_id"] == 22


def test_explicit_equation_retrieval(monkeypatch):

    def fake_search(question, top_k):
        return [{
            "page": 1,
            "chunk_id": 1,
            "text": "Unrelated passage."
        }]

    monkeypatch.setitem(
        sys.modules,
        "retrieve_hybrid",
        SimpleNamespace(hybrid_search=fake_search)
    )

    sample_chunks = [
        {
            "page": 4,
            "chunk_id": 22,
            "text": "Equation (11) describes detection value."
        },
        {
            "page": 4,
            "chunk_id": 23,
            "text": "Additional explanation."
        }
    ]

    monkeypatch.setattr(
        research_retrieval,
        "load_all_chunks",
        lambda: sample_chunks
    )

    result = research_retrieval.retrieve_evidence(
        "Explain Equation 11"
    )

    assert [c["chunk_id"] for c in result] == [22, 23]
