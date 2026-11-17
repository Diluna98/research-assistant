
import pytest


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": 20,
            "page": 3,
            "text": "Introduction to the scheduler."
        },
        {
            "chunk_id": 21,
            "page": 4,
            "text": "Equation (8) describes tracked pedestrians."
        },
        {
            "chunk_id": 22,
            "page": 4,
            "text": "Equation (11) describes undetected pedestrians."
        },
        {
            "chunk_id": 27,
            "page": 5,
            "text": "Equations (8) and (11) are discussed again."
        }
    ]
