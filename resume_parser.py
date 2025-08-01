import pdfplumber

def extract_text_from_pdf(pdf):

    txt=""
    with pdfplumber.open(pdf) as pdf:
        for page in pdf.pages:
            txt+=page.extract_text() + "\n"

    return txt.strip()