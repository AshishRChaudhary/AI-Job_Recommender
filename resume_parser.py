import pdfplumber


def extract_text_from_pdf(pdf):

    pages = []
    with pdfplumber.open(pdf) as doc:
        for page in doc.pages:
            # extract_text() returns None for pages with no text layer (e.g. scanned images)
            pages.append(page.extract_text() or "")

    return "\n".join(pages).strip()
