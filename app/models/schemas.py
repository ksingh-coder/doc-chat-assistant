"""Pydantic models for API request/response schemas"""

from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    """Schema for RAG query requests"""
    question: str = Field(..., description="User's question", min_length=1)
    k: Optional[int] = Field(4, description="Number of documents to retrieve", ge=1, le=10)
    temperature: Optional[float] = Field(None, description="LLM temperature (0-1)", ge=0, le=1)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens in response", ge=1, le=4096)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the main topic of the document?",
                "k": 4,
                "temperature": 0.7,
                "max_tokens": 1024
            }
        }


class DocumentSource(BaseModel):
    """Schema for source document information"""
    content: str = Field(..., description="Retrieved document content")
    source: str = Field(..., description="Source file name")
    page: Optional[int] = Field(None, description="Page number if applicable")


class QueryResponse(BaseModel):
    """Schema for RAG query responses"""
    answer: str = Field(..., description="Generated answer")
    sources: List[DocumentSource] = Field(..., description="Source documents used")
    question: str = Field(..., description="Original question")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The main topic is...",
                "sources": [
                    {
                        "content": "Sample content excerpt...",
                        "source": "document.pdf",
                        "page": 1
                    }
                ],
                "question": "What is the main topic?"
            }
        }


class UploadResponse(BaseModel):
    """Schema for document upload responses"""
    message: str = Field(..., description="Status message")
    filename: str = Field(..., description="Uploaded filename")
    chunks_created: int = Field(..., description="Number of text chunks created")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Document processed successfully",
                "filename": "document.pdf",
                "chunks_created": 42
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check responses"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    vectorstore_ready: bool = Field(..., description="Vector store availability")


class DocumentInfo(BaseModel):
    """Schema for document information"""
    filename: str = Field(..., description="Document filename")
    chunks: int = Field(..., description="Number of chunks")


class DocumentListResponse(BaseModel):
    """Schema for document list responses"""
    documents: List[DocumentInfo] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")


class DocumentDeleteResponse(BaseModel):
    """Schema for document deletion responses"""
    message: str = Field(..., description="Deletion status message")
    filename: str = Field(..., description="Deleted filename")


class DocumentStatsResponse(BaseModel):
    """Schema for document statistics"""
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    ready: bool = Field(..., description="Vector store ready status")
