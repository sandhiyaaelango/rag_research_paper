import numpy as np
from .vector_store import get_index, get_chunk
from .embedder import get_embedding

def retrieve_top_k(query: str, k: int = 5) -> list[str]:
    """
    Embeds the user query and retrieves the top-k most relevant chunks 
    from the FAISS vector store.
    
    Args:
        query: The user's question or search query.
        k: The number of relevant chunks to retrieve.
        
    Returns:
        A list of strings containing the retrieved text chunks.
    """
    index = get_index()
    if index is None:
        raise ValueError("Vector store is empty. Please upload a document first.")
        
    # Generate the query embedding
    query_vector = get_embedding(query)
    
    # Needs to be a 2D array of shape (1, d) and float32 dtype for FAISS
    query_np = np.array([query_vector], dtype=np.float32)
    
    # Perform the search
    # D contains squared distances, I contains indices of the nearest neighbors
    D, I = index.search(query_np, k)
    
    # Map index IDs back to actual text chunks
    retrieved_chunks = []
    
    # Flatten I array which is of shape (1, k)
    for idx in I[0]:
        if idx != -1: # -1 means no neighbor found (e.g., if there are < k vector in store)
            chunk = get_chunk(int(idx))
            if chunk:
                retrieved_chunks.append(chunk)
                
    return retrieved_chunks
