"""Vector store management service using FAISS"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Set
from collections import defaultdict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from app.core.config import settings
from app.core.logging_config import logger


class VectorStoreService:
    """Manage FAISS vector store for document embeddings"""
    
    def __init__(self):
        """Initialize vector store service"""
        self.embeddings = None
        self.vectorstore = None
        self.index_path = settings.vectorstore_dir / settings.vectorstore_index_name
        self.metadata_path = settings.vectorstore_dir / "documents_metadata.json"
        self.documents_metadata: Dict[str, Dict] = {}  # Track uploaded documents
        
        # Initialize embeddings
        self._initialize_embeddings()
        
        # Load existing vectorstore if available
        self._load_vectorstore()
        
        # Load metadata
        self._load_metadata()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        logger.info(f"Initializing embedding model: {settings.embedding_model_name}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={'device': settings.embedding_device},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def _load_vectorstore(self):
        """Load existing FAISS vectorstore if available"""
        try:
            if self.index_path.exists():
                logger.info(f"Loading existing vectorstore from {self.index_path}")
                self.vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vectorstore loaded successfully")
            else:
                logger.info("No existing vectorstore found")
                
        except Exception as e:
            logger.error(f"Error loading vectorstore: {str(e)}")
            logger.info("Will create new vectorstore when documents are added")
    
    def _load_metadata(self):
        """Load documents metadata"""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.documents_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.documents_metadata)} documents")
            else:
                logger.info("No metadata file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            self.documents_metadata = {}
    
    def _save_metadata(self):
        """Save documents metadata"""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.documents_metadata, f, indent=2)
            logger.info("Metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
    
    def add_documents(self, documents: List[Document], filename: str) -> int:
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents to add
            filename: Original filename for tracking
            
        Returns:
            Number of documents added
        """
        logger.info(f"Adding {len(documents)} chunks from '{filename}' to vectorstore")
        
        try:
            # Tag documents with filename for tracking
            for doc in documents:
                doc.metadata['filename'] = filename
            
            if self.vectorstore is None:
                # Create new vectorstore
                logger.info("Creating new vectorstore")
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            else:
                # Add to existing vectorstore
                logger.info("Adding to existing vectorstore")
                self.vectorstore.add_documents(documents)
            
            # Update metadata
            if filename in self.documents_metadata:
                self.documents_metadata[filename]['chunks'] += len(documents)
            else:
                self.documents_metadata[filename] = {
                    'chunks': len(documents),
                    'filename': filename
                }
            
            # Save the vectorstore and metadata
            self._save_vectorstore()
            self._save_metadata()
            
            logger.info(f"Successfully added {len(documents)} chunks from '{filename}' to vectorstore")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error adding documents to vectorstore: {str(e)}")
            raise
    
    def _save_vectorstore(self):
        """Save the vectorstore to disk"""
        try:
            logger.info(f"Saving vectorstore to {self.index_path}")
            self.vectorstore.save_local(str(self.index_path))
            logger.info("Vectorstore saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving vectorstore: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            logger.warning("Vectorstore is empty, no documents to search")
            return []
        
        try:
            logger.info(f"Performing similarity search for query: '{query[:50]}...' (k={k})")
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise
    
    def get_retriever(self, k: int = 4):
        """
        Get a retriever interface for the vectorstore
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever instance
        """
        if self.vectorstore is None:
            logger.warning("Vectorstore is empty")
            return None
        
        try:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            )
            logger.info(f"Created retriever with k={k}")
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}")
            raise
    
    def is_ready(self) -> bool:
        """Check if vectorstore is ready for queries"""
        return self.vectorstore is not None
    
    def get_document_count(self) -> int:
        """Get approximate count of documents in vectorstore"""
        if self.vectorstore is None:
            return 0
        try:
            return self.vectorstore.index.ntotal
        except:
            return 0
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            'total_chunks': self.get_document_count(),
            'total_documents': len(self.documents_metadata),
            'documents': self.documents_metadata
        }
    
    def list_documents(self) -> List[Dict]:
        """List all uploaded documents with their chunk counts"""
        return [
            {
                'filename': filename,
                'chunks': info['chunks']
            }
            for filename, info in self.documents_metadata.items()
        ]
    
    def delete_document(self, filename: str) -> bool:
        """
        Delete a document and its chunks from the vector store
        
        Args:
            filename: Name of the document to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        logger.info(f"Deleting document: {filename}")
        
        if filename not in self.documents_metadata:
            logger.warning(f"Document {filename} not found in metadata")
            return False
        
        try:
            if self.vectorstore is None:
                logger.warning("No vectorstore to delete from")
                return False
            
            # Get all documents from vectorstore
            docstore = self.vectorstore.docstore._dict
            index_to_docstore_id = self.vectorstore.index_to_docstore_id
            
            # Find documents that should be kept (not matching the filename)
            docs_to_keep = []
            for idx, doc_id in index_to_docstore_id.items():
                doc = docstore.get(doc_id)
                if doc and doc.metadata.get('filename') != filename:
                    docs_to_keep.append(doc)
            
            logger.info(f"Keeping {len(docs_to_keep)} chunks, removing {self.documents_metadata[filename]['chunks']} chunks")
            
            # Remove from metadata
            del self.documents_metadata[filename]
            
            # Rebuild vectorstore with remaining documents
            if docs_to_keep:
                self.vectorstore = FAISS.from_documents(docs_to_keep, self.embeddings)
                self._save_vectorstore()
            else:
                # No documents left, delete the vectorstore
                logger.info("No documents left, clearing vectorstore")
                self.vectorstore = None
                if self.index_path.exists():
                    import shutil
                    shutil.rmtree(self.index_path)
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Successfully deleted document: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {str(e)}")
            return False
