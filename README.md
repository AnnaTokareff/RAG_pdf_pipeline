# 📄 PDF RAG Pipeline with Streamlit

This project is a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to upload complex PDF documents and interactively ask questions about their content via a user-friendly **Streamlit** interface.

The app processes PDFs using OCR when needed, extracts structured data (including tables), chunks content semantically, builds a vector index with embeddings, and retrieves the most relevant information using hybrid search and reranking.

---

## Features

- Upload multiple PDF files at once
- Semantic chunking with OpenAI embeddings
- Hybrid search with BM25 + vector similarity
- Reciprocal rerank fusion
- Table extraction (pdfplumber + OCR fallback)
- Multilingual support (e.g., French)
- Custom LLM prompt for grounded answers
- Streamlit web interface

---

## 🛠️ Tech Stack

- `LlamaIndex`
- `ChromaDB`
- `OpenAI Embeddings`
- `Streamlit`
- `pdfplumber`, `PyMuPDF`, `Tesseract OCR`
- `BM25`, `Reciprocal Reranker (FlagEmbedding)`
- `Python 3.11+`

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/AnnaTokareff/RAG_pdf_pipeline.git
cd RAG_pdf_pipeline

# 2. Create a virtual environment
python3 -m venv rag_venv
source rag_venv/bin/activate  # or .\rag_venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run rag_pipeline_streamlit.py
