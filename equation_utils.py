
import re


def find_equation_numbers(question: str) -> list[str]:
    return re.findall(
        r"\b(?:equation|eq\.?)\s*\(?(\d+)\)?",
        question,
        flags=re.IGNORECASE
    )


def select_equation_chunks(chunks, equation_numbers):
    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    selected = {}

    for number in equation_numbers:
        pattern = re.compile(
            rf"\(\s*{re.escape(str(number))}\s*\)"
        )

        matches = [
            chunk for chunk in chunks
            if pattern.search(chunk["text"])
        ]

        if not matches:
            continue

        primary = min(
            matches,
            key=lambda chunk: chunk["chunk_id"]
        )

        selected[primary["chunk_id"]] = primary

        neighbor = chunk_lookup.get(primary["chunk_id"] + 1)

        if neighbor is not None and neighbor["page"] == primary["page"]:
            selected[neighbor["chunk_id"]] = neighbor

    return [selected[i] for i in sorted(selected)]
