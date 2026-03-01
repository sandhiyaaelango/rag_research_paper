import re

def extract_citations(text: str) -> list[str]:
    """
    Extracts citations/references from the document text.
    It attempts to find a "References" or "Bibliography" section,
    or falls back to regex matching for [1], [2] citation patterns block.
    
    Args:
        text: The full cleaned text of the document.
        
    Returns:
        A list of citation strings.
    """
    citations = []
    
    # 1. Attempt to find the References section
    # Look for "References", "Bibliography" on a newline, maybe with numbering like "10. References"
    ref_match = re.search(r'\n(?:\d+\.?\s*)?(?:References|Bibliography)\s*\n', text, re.IGNORECASE)
    
    references_text = ""
    if ref_match:
        # Extract everything after the "References" header
        references_text = text[ref_match.end():]
    else:
        # If no explicit section, maybe the whole document has them sprinkled or at the end
        references_text = text
        
    # 2. Extract individual citations
    # Common formats:
    # [1] Author, A. ...
    # 1. Author, B. ...
    
    # Try to find bracketed citations like [1] ... [2] ...
    bracket_pattern = r'\[\d+\]\s*(.*?)(?=\[\d+\]|$)'
    bracket_matches = re.finditer(bracket_pattern, references_text, re.DOTALL)
    
    for match in bracket_matches:
        cit = match.group(0).strip()
        # Clean up internal newlines in the citation block
        cit = re.sub(r'\s+', ' ', cit)
        if len(cit) > 5:  # Avoid empty/trivial matches
            citations.append(cit)
            
    # If no bracketed matches found, try numbered lists (1. xxx, 2. yyy) at the end of the doc
    if not citations and ref_match:
        numbered_pattern = r'\n\d+\.\s+(.*?)(?=\n\d+\.\s+|$)'
        num_matches = re.finditer(numbered_pattern, references_text, re.DOTALL)
        for match in num_matches:
            cit = match.group(0).strip()
            cit = re.sub(r'\s+', ' ', cit)
            if len(cit) > 5:
                citations.append(cit)
                
    return citations
