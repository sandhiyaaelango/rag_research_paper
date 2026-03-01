# PaperSense AI - Research Paper RAG Assistant

A modern AI Assistant Dashboard for querying research papers using Retrieval-Augmented Generation (RAG).

## 🚀 Features
- **Modern Dashboard UI**: Glassmorphism design with drag-and-drop support.
- **RAG Pipeline**: Semantic chunking, sentence embeddings (SentenceTransformers), and FAISS indexing.
- **LLM Context**: Driven by Groq (Llama-3.1) for citation-aware answers.
- **Section Detection**: Automatically identifies paper sections (Methodology, Results, etc.) for better context mapping.

## 🛠️ Tech Stack
- **Frontend**: Tailwind CSS, Vanilla JS
- **Backend**: Flask, Flask-CORS
- **AI/ML**: PyMuPDF, SentenceTransformers, FAISS, Groq API

## 📋 Prerequisites
- Python 3.8+
- Groq API Key

## 🔧 Setup & Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/sandhiyaaelango/rag_research_paper.git
   cd rag_research_paper/backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your API Key**:
   Create a `.env` file or export `GROQ_API_KEY`:
   ```bash
   GROQ_API_KEY=your_key_here
   ```

4. **Run the server**:
   ```bash
   python app.py
   ```

5. **Open Frontend**:
   Open `backend/index.html` in your browser.

## 📄 License
MIT License
