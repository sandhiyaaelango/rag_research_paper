import re

def clean_text(text: str) -> str:
    """
    Cleans the extracted text to be more suitable for embedding and summarization.
    It removes excessive newlines and whitespace.
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    # Strip leading/trailing whitespaces
    return text.strip()
