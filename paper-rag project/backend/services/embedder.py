import numpy as np
from sentence_transformers import SentenceTransformer

# Load global model to avoid reload on every request
# all-MiniLM-L6-v2 is fast and effective for semantic search
print("Loading embedding model (all-MiniLM-L6-v2)...")
_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.")

def get_embedding(text: str) -> np.ndarray:
    """
    Generates an embedding vector for a given text string.
    
    Args:
        text: The input text to embed.
        
    Returns:
        A 1D numpy array of shape (384,) representing the semantic embedding.
    """
    # model.encode returns a numpy array by default
    vector = _embedding_model.encode(text)
    return vector

def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Generates embedding vectors for a list of text strings.
    
    Args:
        texts: A list of input strings to embed.
        
    Returns:
        A 2D numpy array of shape (N, 384).
    """
    vectors = _embedding_model.encode(texts)
    return vectors
