
import pymupdf4llm

pdf_path = "papers/paper.pdf"

pages = pymupdf4llm.to_markdown(
    pdf_path,
    page_chunks=True
)

print(f"Total pages: {len(pages)}")

print("\nFIRST PAGE:")
print(pages[0]["text"][:2500])

print("\nSECOND PAGE:")
print(pages[1]["text"][:1500])
