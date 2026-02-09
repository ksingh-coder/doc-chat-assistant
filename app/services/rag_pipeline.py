"""RAG pipeline for question answering"""

from typing import Dict, List

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from app.core.config import settings
from app.core.logging_config import logger
from app.services.vectorstore import VectorStoreService


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline"""
    
    def __init__(self, vectorstore_service: VectorStoreService):
        """
        Initialize RAG pipeline
        
        Args:
            vectorstore_service: Vector store service instance
        """
        self.vectorstore_service = vectorstore_service
        self.llm = None
        
        # Initialize LLM
        self._initialize_llm()
        
        # Define prompt template
        self._setup_prompt()
        
        logger.info("RAG Pipeline initialized")
    
    def _initialize_llm(self):
        """Initialize the Groq LLM"""
        logger.info(f"Initializing Groq LLM: {settings.groq_model}")
        
        try:
            self.llm = ChatGroq(
                groq_api_key=settings.groq_api_key,
                model_name=settings.groq_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )
            logger.info("Groq LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Groq LLM: {str(e)}")
            raise
    
    def _setup_prompt(self):
        """Setup the prompt template for RAG"""
        template = """You are a helpful AI assistant that answers questions based on the provided context.

Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Keep your answer concise and relevant to the question.

Context:
{context}

Question: {question}

Answer:"""
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        logger.info("Prompt template configured")
    
    def query(self, question: str, k: int = 4, temperature: float = None, max_tokens: int = None) -> Dict[str, any]:
        """
        Process a question through the RAG pipeline
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            temperature: Optional LLM temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Dictionary with answer and source documents
        """
        logger.info(f"Processing query: '{question[:100]}...' (k={k})")
        
        if not self.vectorstore_service.is_ready():
            logger.warning("Vectorstore not ready for queries")
            return {
                "answer": "No documents have been uploaded yet. Please upload documents first.",
                "sources": [],
                "question": question
            }
        
        try:
            # Get retriever
            retriever = self.vectorstore_service.get_retriever(k=k)
            
            if retriever is None:
                return {
                    "answer": "Vector store is not available.",
                    "sources": [],
                    "question": question
                }
            
            # Use provided values or defaults from settings
            temp = temperature if temperature is not None else settings.temperature
            max_tok = max_tokens if max_tokens is not None else settings.max_tokens
            
            # Create LLM with potentially overridden parameters
            llm = ChatGroq(
                groq_api_key=settings.groq_api_key,
                model_name=settings.groq_model,
                temperature=temp,
                max_tokens=max_tok
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt}
            )
            
            # Run the chain
            logger.info(f"Executing RAG chain (temp={temp}, max_tokens={max_tok})")
            result = qa_chain.invoke({"query": question})
            
            # Extract answer and sources
            answer = result["result"]
            source_docs = result["source_documents"]
            
            logger.info(f"Query processed successfully, {len(source_docs)} sources used")
            
            # Format sources
            sources = self._format_sources(source_docs)
            
            return {
                "answer": answer,
                "sources": sources,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, any]]:
        """
        Format source documents for response
        
        Args:
            documents: List of source documents
            
        Returns:
            List of formatted source information
        """
        sources = []
        
        for doc in documents:
            source_info = {
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
            }
            
            # Add page number if available
            if "page" in doc.metadata:
                source_info["page"] = doc.metadata["page"]
            
            sources.append(source_info)
        
        return sources
    
    def health_check(self) -> Dict[str, any]:
        """
        Check health status of the RAG pipeline
        
        Returns:
            Dictionary with health status information
        """
        return {
            "llm_ready": self.llm is not None,
            "vectorstore_ready": self.vectorstore_service.is_ready(),
            "document_count": self.vectorstore_service.get_document_count()
        }
