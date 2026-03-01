from transformers import pipeline

# Load global summarization model
# We use facebook/bart-large-cnn which is specialized for abstractive summarization.
# Using pipeline abstracts away tokenizer and model boilerplate.
print("Loading summarization model (facebook/bart-large-cnn)...")
_summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
print("Summarization model loaded.")

def generate_summary(context_chunks: list[str]) -> str:
    """
    Generates an abstractive summary from the retrieved context chunks using BART.
    
    Args:
        context_chunks: A list of relevant text chunks retrieved from FAISS.
        
    Returns:
        The generated abstractive summary string.
    """
    # Combine chunks into a single context string
    # We join with double newlines for clearer separation
    context = "\n\n".join(context_chunks)
    
    # BART has a max input length of 1024 tokens. Using a safe heuristic limit
    # based on characters or simply letting transformers truncate.
    # To limit input length safely (assuming ~4 chars per token, keeping < 4000 chars)
    max_input_chars = 4000
    if len(context) > max_input_chars:
        context = context[:max_input_chars]
        
    if not context.strip():
        return "No relevant context found to summarize."
        
    try:
        # Generate summary 
        # max_length defines upper bound of summary length
        result = _summarizer_pipeline(context, max_length=150, min_length=40, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Failed to generate summary: {str(e)}"
