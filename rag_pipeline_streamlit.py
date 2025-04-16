import sys

if 'torch._classes' in sys.modules:
    del sys.modules['torch._classes']

import os
import logging
import streamlit as st
import tempfile
import pandas as pd
from time import time
import asyncio
import nest_asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAG_Pipeline")

try:
    nest_asyncio.apply()
    logger.info("nest_asyncio applied successfully")
except Exception as e:
    logger.warning(f"Failed to apply nest_asyncio: {e}")

from pdf_processor import PDFProcessor, prepare_llama_documents
from llama_index_builder import get_node_parser, build_index_from_nodes, build_query_engine

# Session state initialization
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
    
def handle_multiple_pdfs(uploaded_files, openai_api_key, language="french"):
    """
    Processes uploaded PDF files, extracts content, builds a vector index, and returns a query engine.
    """
    start = time()
    logger.info(f"Processing {len(uploaded_files)} file(s)")

    all_docs = []
    pdf_processor = PDFProcessor(ocr_lang=language[:3])  

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            logger.info(f"Reading {file.name}")
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getvalue())
                    pdf_path = tmp.name

                raw_data = pdf_processor.process_pdf(pdf_path)
                cleaned = {p: pdf_processor.clean_document(c) for p, c in raw_data.items()}
                docs = prepare_llama_documents(cleaned)

                for doc in docs:
                    doc.metadata["source_file"] = file.name

                all_docs.extend(docs)
                os.unlink(pdf_path)

            except Exception as e:
                logger.error(f"Failed on {file.name}: {e}")
                st.error(f"Error processing {file.name}: {e}")

        if not all_docs:
            logger.warning("No documents processed.")
            return None

        try:
            parser = get_node_parser(openai_api_key)
            nodes = parser.get_nodes_from_documents(all_docs)
            logger.info(f"{len(nodes)} nodes created")

            index = build_index_from_nodes(nodes, temp_dir, openai_api_key)
            engine = build_query_engine(index, nodes, openai_api_key, language)

            logger.info(f"Processing complete in {time() - start:.1f}s")
            return engine, all_docs

        except Exception as e:
            logger.error(f"Indexing error: {e}")
            st.error(f"Error building query engine: {e}")
            return None
        
def main():
    st.title("PDF RAG Pipeline")
    st.write("Upload your PDFs and ask questions about their content.")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        language = st.selectbox("Document Language", ["french", "english", "german", "spanish", "italian"])

        st.header("Upload PDFs")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

        if uploaded_files and uploaded_files != st.session_state.uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.session_state.processing_complete = False

        if st.button("Process") and uploaded_files and openai_api_key:
            with st.spinner("Processing PDFs..."):
                try:
                    result = handle_multiple_pdfs(uploaded_files, openai_api_key, language)
                    if result:
                        st.session_state.query_engine, st.session_state.documents = result
                        st.session_state.processing_complete = True
                        st.success("Processing complete!")
                    else:
                        st.error("No documents were processed.")
                except Exception as e:
                    logger.exception("Processing error")
                    st.error(f"Something went wrong: {e}")

    # Results and query
    if st.session_state.processing_complete:
        st.subheader("Documents Overview")
        st.write(f"Processed {len(st.session_state.documents)} documents.")

        doc_info = pd.DataFrame([
            {
                "Type": doc.metadata.get("type", "unknown"),
                "Page": doc.metadata.get("page", "N/A"),
                "Source": doc.metadata.get("source_file", "?")
            }
            for doc in st.session_state.documents
        ])
        summary = doc_info["Type"].value_counts().reset_index()
        summary.columns = ["Type", "Count"]
        st.dataframe(summary)

        st.subheader("Ask a Question")
        query = st.text_input("Your question:")

        if query and st.session_state.query_engine:
            with st.spinner("Searching for an answer..."):
                try:
                    start = time()
                    response = st.session_state.query_engine.query(query)
                    elapsed = time() - start

                    st.write(f"**Answered (in {elapsed:.1f}s):**")
                    st.write(response.response)

                    if hasattr(response, "source_nodes") and response.source_nodes:
                        st.subheader("Sources")
                        for i, node in enumerate(response.source_nodes):
                            with st.expander(f"Source {i+1} - {node.metadata.get('source_file', 'Unknown')} (Page {node.metadata.get('page', 'N/A')})"):
                                st.write(node.text[:500] + "..." if len(node.text) > 500 else node.text)
                                if hasattr(node, "score"):
                                    st.caption(f"Score: {node.score:.3f}")
                except Exception as e:
                    logger.exception("Query error")
                    st.error(f"Failed to get an answer: {e}")
    else:
        if not uploaded_files:
            st.info("Please, upload some PDFs to get started")
        elif not openai_api_key:
            st.warning("Enter your OpenAI API key: ")
        else:
            st.info("Ready to process. Click the button when you're ready.")

if __name__ == "__main__":
    main()
