"""API route definitions"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import List

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    HealthResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    DocumentStatsResponse
)
from app.services.document_processor import DocumentProcessor
from app.services.vectorstore import VectorStoreService
from app.services.rag_pipeline import RAGPipeline
from app.core.config import settings
from app.core.logging_config import logger


# Initialize services
document_processor = DocumentProcessor()
vectorstore_service = VectorStoreService()
rag_pipeline = RAGPipeline(vectorstore_service)

# Create router
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    
    health_status = rag_pipeline.health_check()
    
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        vectorstore_ready=health_status["vectorstore_ready"]
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document
    
    Supports: PDF, TXT, and Markdown files
    """
    logger.info(f"Document upload requested: {file.filename}")
    
    # Validate file type
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_extension = "." + file.filename.split(".")[-1].lower()
    
    if file_extension not in allowed_extensions:
        logger.warning(f"Unsupported file type attempted: {file_extension}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Save the file
        file_path = document_processor.save_uploaded_file(content, file.filename)
        
        # Process the document
        chunks = document_processor.process_file(file_path)
        
        # Add to vectorstore with filename tracking
        num_chunks = vectorstore_service.add_documents(chunks, file.filename)
        
        logger.info(f"Document uploaded and processed successfully: {file.filename}")
        
        return UploadResponse(
            message="Document processed successfully",
            filename=file.filename,
            chunks_created=num_chunks
        )
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question
    
    Returns an answer based on uploaded documents
    """
    logger.info(f"Query requested: '{request.question[:100]}...'")
    
    try:
        # Process the query through RAG pipeline
        result = rag_pipeline.query(
            question=request.question,
            k=request.k,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            question=result["question"]
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/documents/count", response_model=DocumentStatsResponse)
async def get_document_count():
    """Get statistics about documents and chunks in the vectorstore"""
    stats = vectorstore_service.get_stats()
    logger.info(f"Document stats requested: {stats['total_documents']} documents, {stats['total_chunks']} chunks")
    
    return DocumentStatsResponse(
        total_documents=stats['total_documents'],
        total_chunks=stats['total_chunks'],
        ready=vectorstore_service.is_ready()
    )


@router.get("/documents/list", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents with their chunk counts"""
    documents = vectorstore_service.list_documents()
    logger.info(f"Document list requested: {len(documents)} documents")
    
    return DocumentListResponse(
        documents=documents,
        total=len(documents)
    )


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(filename: str):
    """Delete a document and its embeddings from the vectorstore"""
    logger.info(f"Delete requested for document: {filename}")
    
    success = vectorstore_service.delete_document(filename)
    
    if success:
        return DocumentDeleteResponse(
            message=f"Document '{filename}' deleted successfully",
            filename=filename
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found"
        )
