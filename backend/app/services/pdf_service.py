import fitz


def extract_text_from_pdf(file_path: str) -> dict:
    """
    Extract text from a PDF file using PyMuPDF.
    """

    document = fitz.open(file_path)

    extracted_pages = []
    full_text = ""

    for page_number, page in enumerate(document, start=1):
        page_text = page.get_text()

        extracted_pages.append(
            {
                "page_number": page_number,
                "text": page_text
            }
        )

        full_text += f"\n\n--- Page {page_number} ---\n\n"
        full_text += page_text

    document.close()

    return {
        "total_pages": len(extracted_pages),
        "full_text": full_text,
        "pages": extracted_pages
    }