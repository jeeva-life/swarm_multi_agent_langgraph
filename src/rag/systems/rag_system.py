"""
Comprehensive RAG (Retrieval Augmented Generation) system.
Handles document loading, chunking, embedding, and retrieval for multiple document types.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import asyncio

# LangChain imports
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    JSONLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

from core.config.config import config
from ..optimization.query_optimizer import get_query_optimizer


class RAGSystem:
    """
    Comprehensive RAG system for document-based question answering.
    Supports multiple document types and provides efficient retrieval.
    """
    
    def __init__(
        self, 
        folder_path: str = "docs",
        collection_name: str = "rag_collection",
        persist_directory: str = "./chroma_db_swarm",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        enable_query_optimization: bool = True,
        mmr_lambda: float = 0.7
    ):
        """
        Initialize the RAG system.
        
        Args:
            folder_path: Directory where knowledge documents are stored
            collection_name: Name for the Chroma collection
            persist_directory: Directory to persist the vector store
            chunk_size: Size of document chunks for processing
            chunk_overlap: Overlap between chunks for context continuity
            enable_query_optimization: Whether to enable query rewriting and MMR reranking
            mmr_lambda: MMR balance parameter (0.0 = diversity, 1.0 = relevance)
        """
        self.logger = logging.getLogger("rag_system")
        self.folder_path = folder_path
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_query_optimization = enable_query_optimization
        self.mmr_lambda = mmr_lambda
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model=config.anthropic.model,
            api_key=config.anthropic.api_key,
            temperature=config.anthropic.temperature,
            max_tokens=config.anthropic.max_tokens
        )
        
        # Initialize components
        self.vectorstore = None
        self.retriever = None
        self.embedding_function = None
        self.query_optimizer = None
        
        # Initialize query optimizer if enabled
        if self.enable_query_optimization:
            try:
                self.query_optimizer = get_query_optimizer()
                self.logger.info("Query optimizer initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize query optimizer: {str(e)}")
                self.enable_query_optimization = False
        
        # Document type mappings
        self.document_loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.doc': Docx2txtLoader,
            '.html': UnstructuredHTMLLoader,
            '.htm': UnstructuredHTMLLoader,
            '.md': UnstructuredMarkdownLoader,
            '.csv': CSVLoader,
            '.json': JSONLoader
        }
        
        # Initialize the RAG system
        self.setup_rag()
    
    def load_documents(self, folder_path: str) -> List[Document]:
        """
        Load all supported documents from the folder into LangChain Document objects.
        
        Args:
            folder_path: Path to the documents folder
            
        Returns:
            List of loaded documents
        """
        documents = []
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            self.logger.warning(f"Document folder {folder_path} does not exist")
            return documents
        
        # Get all files in the folder
        files = []
        for ext in self.document_loaders.keys():
            files.extend(folder_path.glob(f"**/*{ext}"))
        
        self.logger.info(f"Found {len(files)} supported documents in {folder_path}")
        
        for file_path in files:
            try:
                file_ext = file_path.suffix.lower()
                loader_class = self.document_loaders.get(file_ext)
                
                if loader_class:
                    self.logger.info(f"Loading document: {file_path}")
                    
                    # Special handling for different loader types
                    if file_ext == '.json':
                        loader = loader_class(
                            str(file_path),
                            jq_schema='.content',
                            text_content=False
                        )
                    else:
                        loader = loader_class(str(file_path))
                    
                    # Load the document
                    docs = loader.load()
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata.update({
                            'source': str(file_path),
                            'file_type': file_ext,
                            'file_name': file_path.name
                        })
                    
                    documents.extend(docs)
                    self.logger.info(f"Successfully loaded {len(docs)} pages from {file_path.name}")
                
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {str(e)}")
                continue
        
        self.logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def setup_rag(self):
        """
        Set up the complete RAG pipeline:
        1. Load documents from the specified folder
        2. Split documents into chunks for better retrieval
        3. Create embeddings for each chunk
        4. Store embeddings in a persistent vector database (Chroma)
        """
        try:
            self.logger.info("Setting up RAG system...")
            
            # Load documents
            documents = self.load_documents(self.folder_path)
            
            if not documents:
                self.logger.warning("No documents found. RAG system will be initialized with empty vector store.")
                # Create empty vector store
                self.embedding_function = SentenceTransformerEmbeddings(
                    model_name=config.huggingface.model_name
                )
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_function,
                    persist_directory=self.persist_directory
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                return
            
            # Split documents into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            splits = text_splitter.split_documents(documents)
            self.logger.info(f"Split documents into {len(splits)} chunks")
            
            # Generate vector embeddings for each chunk
            self.embedding_function = SentenceTransformerEmbeddings(
                model_name=config.huggingface.model_name
            )
            
            # Create a Chroma vector store and persist to disk
            self.vectorstore = Chroma.from_documents(
                collection_name=self.collection_name,
                documents=splits,
                embedding=self.embedding_function,
                persist_directory=self.persist_directory
            )
            
            self.logger.info("Vector store created and persisted")
            
            # Configure retriever to get top 3 relevant chunks for a query
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            self.logger.info("RAG system setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up RAG system: {str(e)}")
            raise
    
    def rag_response_generator(self, query: str) -> str:
        """
        Generate a response for a query using the RAG pipeline with optional query optimization.
        
        Args:
            query: User query
            
        Returns:
            Generated response based on retrieved context
        """
        try:
            if not self.retriever:
                return "RAG system not properly initialized. No knowledge base available."
            
            # Retrieve documents with optional optimization
            retrieved_docs = self._retrieve_documents_optimized(query)
            
            if not retrieved_docs:
                return "No relevant information found in the knowledge base for your query."
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Create a prompt template for RAG
            rag_template = ChatPromptTemplate.from_messages([
                ("system", """You are an assistant for question-answering tasks using RAG (Retrieval Augmented Generation).
                
                IMPORTANT RULES:
                - Answer ONLY based on the provided context
                - If the context doesn't contain enough information to answer the question, say so clearly
                - Be accurate and cite specific information from the context when possible
                - If the query is about database information (albums, customers, invoices, etc.), politely redirect to the appropriate agent
                - Maintain a professional and helpful tone
                """),
                ("human", "Query: {question}\n\nContext: {context}")
            ])
            
            # Chain prompt -> LLM -> output parser
            rag_chain = rag_template | self.llm | StrOutputParser()
            
            # Invoke the chain with query + retrieved context
            response = rag_chain.invoke({
                "question": query,
                "context": context
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating RAG response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _retrieve_documents_optimized(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve documents with optional query optimization and MMR reranking.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of optimized retrieved documents
        """
        try:
            # First, retrieve more documents than needed for optimization
            initial_k = k * 3 if self.enable_query_optimization else k
            self.retriever.search_kwargs = {"k": initial_k}
            
            # Get initial documents
            initial_docs = self.retriever.get_relevant_documents(query)
            
            if not initial_docs:
                return []
            
            # Apply optimization if enabled
            if self.enable_query_optimization and self.query_optimizer:
                try:
                    optimized_docs, optimization_metadata = self.query_optimizer.optimize_retrieval_sync(
                        query=query,
                        documents=initial_docs,
                        lambda_param=self.mmr_lambda,
                        k=k,
                        use_query_rewriting=True
                    )
                    
                    self.logger.info(f"Query optimization applied. Retrieved {len(optimized_docs)} optimized documents")
                    return optimized_docs
                    
                except Exception as e:
                    self.logger.warning(f"Query optimization failed, using standard retrieval: {str(e)}")
                    return initial_docs[:k]
            else:
                return initial_docs[:k]
                
        except Exception as e:
            self.logger.error(f"Error in optimized document retrieval: {str(e)}")
            # Fallback to standard retrieval
            try:
                self.retriever.search_kwargs = {"k": k}
                return self.retriever.get_relevant_documents(query)
            except Exception as fallback_error:
                self.logger.error(f"Fallback retrieval also failed: {str(fallback_error)}")
                return []
    
    async def rag_response_generator_async(self, query: str) -> str:
        """
        Async version of rag_response_generator with optimized retrieval.
        
        Args:
            query: User query
            
        Returns:
            Generated response based on retrieved context
        """
        try:
            if not self.retriever:
                return "RAG system not properly initialized. No knowledge base available."
            
            # Retrieve documents with optional optimization (async)
            retrieved_docs = await self._retrieve_documents_optimized_async(query)
            
            if not retrieved_docs:
                return "No relevant information found in the knowledge base for your query."
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Create a prompt template for RAG
            rag_template = ChatPromptTemplate.from_messages([
                ("system", """You are an assistant for question-answering tasks using RAG (Retrieval Augmented Generation).
                
                IMPORTANT RULES:
                - Answer ONLY based on the provided context
                - If the context doesn't contain enough information to answer the question, say so clearly
                - Be accurate and cite specific information from the context when possible
                - If the query is about database information (albums, customers, invoices, etc.), politely redirect to the appropriate agent
                - Maintain a professional and helpful tone
                """),
                ("human", "Query: {question}\n\nContext: {context}")
            ])
            
            # Chain prompt -> LLM -> output parser
            rag_chain = rag_template | self.llm | StrOutputParser()
            
            # Invoke the chain with query + retrieved context
            response = await rag_chain.ainvoke({
                "question": query,
                "context": context
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in async RAG response generation: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    async def _retrieve_documents_optimized_async(self, query: str, k: int = 3) -> List[Document]:
        """
        Async version of optimized document retrieval.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of optimized retrieved documents
        """
        try:
            # First, retrieve more documents than needed for optimization
            initial_k = k * 3 if self.enable_query_optimization else k
            self.retriever.search_kwargs = {"k": initial_k}
            
            # Get initial documents
            initial_docs = self.retriever.get_relevant_documents(query)
            
            if not initial_docs:
                return []
            
            # Apply optimization if enabled
            if self.enable_query_optimization and self.query_optimizer:
                try:
                    optimized_docs, optimization_metadata = await self.query_optimizer.optimize_retrieval(
                        query=query,
                        documents=initial_docs,
                        lambda_param=self.mmr_lambda,
                        k=k,
                        use_query_rewriting=True
                    )
                    
                    self.logger.info(f"Async query optimization applied. Retrieved {len(optimized_docs)} optimized documents")
                    return optimized_docs
                    
                except Exception as e:
                    self.logger.warning(f"Async query optimization failed, using standard retrieval: {str(e)}")
                    return initial_docs[:k]
            else:
                return initial_docs[:k]
                
        except Exception as e:
            self.logger.error(f"Error in async optimized document retrieval: {str(e)}")
            # Fallback to standard retrieval
            try:
                self.retriever.search_kwargs = {"k": k}
                return self.retriever.get_relevant_documents(query)
            except Exception as fallback_error:
                self.logger.error(f"Async fallback retrieval also failed: {str(fallback_error)}")
                return []
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add new documents to the existing vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.vectorstore:
                self.logger.error("Vector store not initialized")
                return False
            
            # Split new documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len
            )
            
            splits = text_splitter.split_documents(documents)
            
            # Add to vector store
            self.vectorstore.add_documents(splits)
            
            self.logger.info(f"Added {len(splits)} new document chunks to vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            return False
    
    def search_documents(self, query: str, k: int = 3, use_optimization: bool = True) -> List[Document]:
        """
        Search for relevant documents with optional optimization.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            use_optimization: Whether to use query optimization and MMR reranking
            
        Returns:
            List of relevant documents
        """
        try:
            if not self.retriever:
                return []
            
            if use_optimization and self.enable_query_optimization:
                return self._retrieve_documents_optimized(query, k)
            else:
                # Standard retrieval
                self.retriever.search_kwargs = {"k": k}
                docs = self.retriever.get_relevant_documents(query)
                return docs
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about query optimization configuration.
        
        Returns:
            Dictionary with optimization statistics
        """
        return {
            "query_optimization_enabled": self.enable_query_optimization,
            "mmr_lambda": self.mmr_lambda,
            "query_optimizer_available": self.query_optimizer is not None,
            "optimization_balance": {
                "lambda_param": self.mmr_lambda,
                "relevance_weight": self.mmr_lambda,
                "diversity_weight": 1.0 - self.mmr_lambda
            }
        }
    
    def get_vectorstore_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store.
        
        Returns:
            Dictionary with vector store information
        """
        try:
            if not self.vectorstore:
                return {"error": "Vector store not initialized"}
            
            # Get collection info
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
            
        except Exception as e:
            self.logger.error(f"Error getting vector store info: {str(e)}")
            return {"error": str(e)}
    
    def clear_vectorstore(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.vectorstore:
                return False
            
            # Delete the collection
            self.vectorstore.delete_collection()
            
            # Recreate empty collection
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
            
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            self.logger.info("Vector store cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing vector store: {str(e)}")
            return False


# Global RAG system instance
_rag_system_instance = None


def get_rag_system(folder_path: str = "docs") -> RAGSystem:
    """
    Get or create the global RAG system instance.
    
    Args:
        folder_path: Path to documents folder
        
    Returns:
        RAGSystem instance
    """
    global _rag_system_instance
    
    if _rag_system_instance is None:
        _rag_system_instance = RAGSystem(folder_path=folder_path)
    
    return _rag_system_instance


# LangChain tool for RAG functionality
@tool
def rag_tool(query: str) -> str:
    """
    Tool for handling general information queries using RAG system.
    Retrieves relevant information from the knowledge base and generates answers.
    
    Args:
        query: The question or query to answer
        
    Returns:
        Answer based on retrieved knowledge base content
    """
    try:
        rag_system = get_rag_system()
        return rag_system.rag_response_generator(query)
    except Exception as e:
        return f"Error retrieving information: {str(e)}"


# Export the tool for use in agents
general_qa_tool = rag_tool
