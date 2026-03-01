import os
import fitz
import faiss
import re
import numpy as np
from groq import Groq

# ================= CONFIG =================
# Securely loading the API Key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.1-8b-instant"
EMBED = "all-MiniLM-L6-v2"

CHUNK_SIZE = 300
OVERLAP = 60
TOP_K = 3

# Initialize models - lazy loaded to save memory at startup
_embed = None
def get_embed_model():
    global _embed
    if _embed is None:
        from sentence_transformers import SentenceTransformer
        _embed = SentenceTransformer(EMBED)
    return _embed

# Initialize Groq client - Wrapped in a property
_client = None
def get_groq_client():
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment. Please check your .env file.")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

# Global storage for the index and chunks
index = None
chunks = []

def norm(x):
    n=np.linalg.norm(x,axis=1,keepdims=True)
    n[n==0]=1e-10
    return x/n

# ================= EXTRACT =================
def extract(p):
    d=fitz.open(p)
    return " ".join([pg.get_text() for pg in d]).replace("-\n"," ")

# ================= SECTION MAP =================
SECTION_MAP = {
    "deep residual learning":"methodology",
    "residual learning":"methodology",
    "implementation":"methodology",
    "experiments":"results",
    "experiment":"results",
    "results":"results",
    "conclusion":"conclusion"
}

# ================= DETECT =================
def detect(text):
    pattern=r"\n\d+\.\s([A-Za-z\s]+)"
    matches=re.findall(pattern,text)
    sec={}
    for m in matches:
        s=m.strip().lower()
        for key in SECTION_MAP:
            if key in s:
                sec[SECTION_MAP[key]]=s
    return sec

# ================= CHUNK =================
def chunk(t):
    w=t.split()
    out=[]
    i=0
    while i<len(w):
        out.append(" ".join(w[i:i+CHUNK_SIZE]))
        i+=CHUNK_SIZE-OVERLAP
    return out

# ================= BUILD =================
def process_pdf(path):
    global index, chunks
    
    # Reset for new document
    chunks = []
    
    txt = extract(path)
    secs = detect(txt)
    
    filename = os.path.basename(path)

    if not secs:
        # Fallback: Process the entire text if no sections are detected
        ch = chunk(txt)
        for c in ch:
            chunks.append({
                "doc": filename,
                "section": "unstructured/general",
                "text": c
            })
    else:
        for mapped in secs:
            raw = secs[mapped]
            start = txt.lower().find(raw)
            # Take a 3000 char block from the section start
            block = txt[start:start+3000] if start != -1 else txt[:3000]

            ch = chunk(block)
            for i, c in enumerate(ch):
                chunks.append({
                    "doc": filename,
                    "section": mapped,
                    "text": c
                })

    if not chunks:
        print("No chunks found to index.")
        return

    texts = [c["text"] for c in chunks]
    embed = get_embed_model()
    emb = embed.encode(texts)
    
    if len(emb) == 0:
        print("Embeddings are empty.")
        return
        
    emb = norm(emb).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

# ================= ASK =================
def query_rag(q):
    global index, chunks
    
    if index is None or not chunks:
        return "Please upload and index a PDF first."
    
    try:
        client = get_groq_client()
    except ValueError as e:
        return str(e)
    
    embed = get_embed_model()
    q_emb = embed.encode([q])
    q_emb = norm(q_emb).astype("float32")

    scores, inds = index.search(q_emb, TOP_K)

    ctx = ""
    for i, j in enumerate(inds[0]):
        ctx += f"\n[{i+1}] {chunks[j]['text']}\n"

    prompt = f"""
Answer ONLY from context.
Cite chunk numbers after each claim (e.g., [1]).
If missing say: Not found in paper

Context:
{ctx}

Question:
{q}
"""

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.1
        )
        
        answer = r.choices[0].message.content
        
        # Add references to the final string
        refs = []
        for i, j in enumerate(inds[0]):
            refs.append(f"[{i+1}] {chunks[j]['doc']} | {chunks[j]['section']}")
            
        full_response = f"{answer}\n\nREFERENCES:\n" + "\n".join(refs)
        return full_response

    except Exception as e:
        return f"Groq Error: {str(e)}"
