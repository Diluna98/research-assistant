
import pymupdf
from pathlib import Path

pdf_path = Path("papers/paper.pdf")

with pymupdf.open(pdf_path) as document:

    print(f"Number of pages: {len(document)}")

    for page_number, page in enumerate(document, start=1):

        text = page.get_text(sort=True)

        print(f"\nPage {page_number}")
        print(f"Extracted characters: {len(text)}")

        if page_number == 1:
            print("\nExtracted text preview:")
            print(text[:1000])
