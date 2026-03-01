from typing import List

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks based on word count.
    
    Args:
        text: The input text string
        chunk_size: Maximum number of words in a chunk
        overlap: Number of overlapping words between consecutive chunks
        
    Returns:
        A list of string chunks.
    """
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    step = chunk_size - overlap
    if step <= 0:
        step = 1  # Failsafe if overlap >= chunk_size
        
    for i in range(0, len(words), step):
        # Slice out up to `chunk_size` words
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        
        # Break early if we reached the end of the full text
        if i + chunk_size >= len(words):
            break
            
    return chunks
