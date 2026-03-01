import faiss
import numpy as np

# In-memory session store to map document chunks
# In a real database scenario, this would be a DB or persistent vector store
_global_chunks = []
_faiss_index = None
_embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2

def build_index(chunks: list[str], embeddings_np: np.ndarray) -> None:
    """
    Builds the FAISS index from the provided chunks and their embeddings.
    Uses IndexFlatL2 for exact nearest neighbor search, configured for float32.
    
    Args:
        chunks: List of text chunks corresponding to the embeddings.
        embeddings_np: Numpy array of shape (N, d) containing embeddings.
    """
    global _global_chunks, _faiss_index
    
    # Needs to be float32 for FAISS
    embeddings_np = np.array(embeddings_np, dtype=np.float32)
    
    # Initialize the index
    index = faiss.IndexFlatL2(_embedding_dimension)
    
    # Add vectors to index
    index.add(embeddings_np)
    
    # Store globally for the session
    _faiss_index = index
    _global_chunks = chunks

def get_index() -> faiss.Index:
    """Returns the current global FAISS index."""
    return _faiss_index

def get_chunk(index: int) -> str:
    """Returns a specific chunk of text by its index."""
    if 0 <= index < len(_global_chunks):
        return _global_chunks[index]
    return ""

def get_chunks_count() -> int:
    """Returns the total number of indexed chunks."""
    return len(_global_chunks)
