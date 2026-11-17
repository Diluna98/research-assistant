
from equation_utils import find_equation_numbers, select_equation_chunks


def test_extracts_two_equations():
    question = "Compare Equation (8) and Equation (11)."

    result = find_equation_numbers(question)

    assert result == ["8", "11"]


def test_question_without_equations():
    question = "How does camera selection work?"

    result = find_equation_numbers(question)

    assert result == []


def test_sample_document_contains_equations(sample_chunks):

    equation_8 = [
        chunk for chunk in sample_chunks
        if "(8)" in chunk["text"]
    ]

    assert len(equation_8) == 2

    assert equation_8[0]["chunk_id"] == 21


def test_selects_equation_chunks(sample_chunks):

    result = select_equation_chunks(
        sample_chunks,
        ["8", "11"]
    )

    selected_ids = [
        chunk["chunk_id"]
        for chunk in result
    ]

    assert selected_ids == [21, 22]