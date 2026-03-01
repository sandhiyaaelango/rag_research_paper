import io
import PyPDF2

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts and returns all text from a given PDF file represented in bytes.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")
