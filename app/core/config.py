"""Application configuration and settings"""

import os
from pathlib import Path
from typing import Optional
import torch
from pydantic_settings import BaseSettings


def detect_device() -> str:
    """Auto-detect GPU availability"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Settings
    app_name: str = "RAG Document Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Groq API
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"
    
    # Embedding Model
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    embedding_device: Optional[str] = None  # auto-detects if not set
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    documents_dir: Path = data_dir / "documents"
    vectorstore_dir: Path = data_dir / "vectorstore"
    logs_dir: Path = base_dir / "logs"
    
    # FAISS Settings
    vectorstore_index_name: str = "faiss_index"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # RAG Settings
    retrieval_k: int = 4  # Number of documents to retrieve
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Auto-detect GPU if not specified
        if self.embedding_device is None:
            self.embedding_device = detect_device()
        
        # Create directories if they don't exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
