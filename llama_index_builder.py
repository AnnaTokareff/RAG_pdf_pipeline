import os
import Stemmer
import chromadb
from typing import List

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine

def get_node_parser(openai_api_key: str):
    """Create a semantic node parser using OpenAI embeddings"""
    os.environ["OPENAI_API_KEY"] = openai_api_key
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    return SemanticSplitterNodeParser(
        embed_model=embed_model,
        chunk_size=1024,
        chunk_overlap=40
    )

def build_index_from_nodes(nodes, persist_dir: str, openai_api_key: str) -> VectorStoreIndex:
    """Builds vector index from document nodes"""
    os.environ["OPENAI_API_KEY"] = openai_api_key
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    db_client = chromadb.PersistentClient(path=persist_dir)
    collection = db_client.get_or_create_collection("rag_pdf_docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes=nodes, storage_context=storage_context)

def build_query_engine(index: VectorStoreIndex, nodes: List[Document], openai_api_key: str, language: str = "french") -> RetrieverQueryEngine:
    """Builds a query engine with hybrid search and reranking"""
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # vector retriever
    vector_retriever = index.as_retriever(similarity_top_k=4)
    
    # BM25 retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=3,
        stemmer=Stemmer.Stemmer(language),
        language=language
    )
    
    # Hybrid retriever
    retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=2,
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True
    )
    
    # Reranker
    # reranker = FlagEmbeddingReranker(top_n=3, model="BAAI/bge-reranker-v2-m3")
    reranker = FlagEmbeddingReranker(top_n=3, model="BAAI/bge-reranker-base")

    
    # LLM
    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    
    # prompt to llm
    custom_prompt = PromptTemplate(
        "Tu es un expert en analyse de documents. "
        "Réponds à la question en te basant uniquement sur le contexte fourni. "
        "Si le contexte ne contient pas l'information nécessaire pour répondre, indique-le clairement. "
        "N'invente aucune information qui ne se trouve pas dans le contexte. "
        "Fournis des réponses précises. "
        "Réponds toujours dans la même langue que celle de la question posée.\n\n"
        "Contexte: {context_str}\n\n"
        "Question: {query_str}\n\n"
        "Réponse: "
    )
    
    #  query engine
    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        node_postprocessors=[reranker],
        text_qa_template=custom_prompt
    )