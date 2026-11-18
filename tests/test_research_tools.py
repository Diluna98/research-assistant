
from research_tools import search_paper


def test_search_paper_uses_shared_retrieval(monkeypatch):

    def fake_retrieve(question, top_k):

        assert question == "Explain Equation 11"
        assert top_k == 3

        return [
            {
                "page": 4,
                "chunk_id": 22,
                "text": "Evidence about Equation 11."
            }
        ]

    monkeypatch.setattr(
        "research_tools.retrieve_evidence",
        fake_retrieve
    )

    result = search_paper(
        "Explain Equation 11",
        top_k=3
    )

    assert result["question"] == "Explain Equation 11"
    assert len(result["results"]) == 1
    assert result["results"][0]["chunk_id"] == 22
