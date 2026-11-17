
import json
from pathlib import Path
import pymupdf4llm

pdf_path = Path("papers/paper.pdf")

chunks = []
chunk_id = 0


pages = pymupdf4llm.to_markdown(
    str(pdf_path),
    page_chunks=True
)

for page_number, page in enumerate(pages, start=1):

    text = page["text"]
    words = text.split()

    chunk_size = 180
    overlap = 30
    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            continue

        chunks.append({
            "chunk_id": chunk_id,
            "page": page_number,
            "text": " ".join(chunk_words)
        })

        chunk_id += 1

output_path = Path("results/paper_chunks.json")
output_path.parent.mkdir(exist_ok=True)

output_path.write_text(
    json.dumps(chunks, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Total chunks: {len(chunks)}")

for chunk in chunks[:3]:
    print(f"\nChunk {chunk['chunk_id']}")
    print(f"Page: {chunk['page']}")
    print(f"Text: {chunk['text'][:200]}...")
