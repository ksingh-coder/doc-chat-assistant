"""Document processing and ingestion service"""

import os
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.schema import Document

from app.core.config import settings
from app.core.logging_config import logger


class DocumentProcessor:
    """Handle document loading and chunking"""
    
    def __init__(self):
        """Initialize document processor"""
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info(
            f"DocumentProcessor initialized with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )
    
    def load_document(self, file_path: Path) -> List[Document]:
        """
        Load a document based on its file type
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of loaded documents
            
        Raises:
            ValueError: If file type is not supported
        """
        logger.info(f"Loading document: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_extension == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            elif file_extension == ".md":
                loader = UnstructuredMarkdownLoader(str(file_path))
            else:
                raise ValueError(
                    f"Unsupported file type: {file_extension}. "
                    f"Supported types: .pdf, .txt, .md"
                )
            
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} document(s) from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        logger.info(f"Splitting {len(documents)} document(s) into chunks")
        
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from documents")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def process_file(self, file_path: Path) -> List[Document]:
        """
        Complete processing pipeline for a file
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of processed document chunks
        """
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Load the document
            documents = self.load_document(file_path)
            
            # Split into chunks
            chunks = self.split_documents(documents)
            
            logger.info(
                f"File processing complete for {file_path.name}: "
                f"{len(chunks)} chunks created"
            )
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> Path:
        """
        Save uploaded file to documents directory
        
        Args:
            file_content: File content as bytes
            filename: Name of the file
            
        Returns:
            Path to saved file
        """
        file_path = settings.documents_dir / filename
        
        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
            logger.info(f"Saved uploaded file: {filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving file {filename}: {str(e)}")
            raise
