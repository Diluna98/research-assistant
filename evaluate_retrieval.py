
from retrieve_hybrid import hybrid_search

tests = [
    {
        "question": "How are undetected pedestrians modeled?",
        "relevant": {21, 22}
    },
    {
        "question": "How is the pedestrian arrival rate calibrated?",
        "relevant": {22, 29}
    },
    {
        "question": "How is the value of observing tracked pedestrians calculated?",
        "relevant": {17, 20, 21}
    }
]

successful = 0

for test in tests:
    results = hybrid_search(test["question"], top_k=5)

    retrieved_ids = [
        chunk["chunk_id"] for chunk in results
    ]

    matches = test["relevant"].intersection(retrieved_ids)

    hit = len(matches) > 0
    successful += int(hit)

    print(f"\nQuestion: {test['question']}")
    print(f"Retrieved chunks: {retrieved_ids}")
    print(f"Relevant chunks found: {sorted(matches)}")
    print(f"Hit at 5: {hit}")

hit_rate = successful / len(tests)

print("\nEvaluation summary:")
print(f"Hit rate at 5: {hit_rate:.2%}")
