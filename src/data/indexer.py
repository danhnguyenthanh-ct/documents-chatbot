"""
Document Indexing Module
Handles batch processing, embedding generation, and vector storage with progress tracking.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import datetime
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import time

from ..core.embeddings import GeminiEmbeddings, EmbeddingError
from ..core.vector_store import QdrantVectorStore, VectorStoreError
from .loader import LoadedDocument, DocumentLoader
from .preprocessor import TextPreprocessor, ProcessedText
from .chunker import TextChunker, TextChunk

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Custom exception for indexing operations"""
    pass


@dataclass
class IndexedDocument:
    """Container for indexed document metadata"""
    document_id: str
    file_path: str
    content_hash: str
    chunk_count: int
    total_tokens: int
    indexed_at: datetime
    processing_time_seconds: float
    metadata: Dict[str, Any]


class DocumentIndexer:
    """
    Production-ready document indexing system with batch processing
    """
    
    def __init__(
        self,
        embeddings_service: GeminiEmbeddings,
        vector_store: QdrantVectorStore,
        collection_name: str = "documents",
        batch_size: int = 10,
        max_concurrent: int = 3,
        enable_deduplication: bool = True
    ):
        """
        Initialize document indexer
        
        Args:
            embeddings_service: Gemini embeddings service
            vector_store: Qdrant vector store
            collection_name: Name of the collection to store vectors
            batch_size: Number of chunks to process in each batch
            max_concurrent: Maximum concurrent embedding requests
            enable_deduplication: Enable duplicate detection
        """
        self.embeddings_service = embeddings_service
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.enable_deduplication = enable_deduplication
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_preprocessor = TextPreprocessor()
        self.text_chunker = TextChunker()
        
        # Track indexed documents
        self.indexed_documents: Dict[str, IndexedDocument] = {}
        self.content_hashes: Set[str] = set()
        
        # Initialize statistics
        self.stats = {
            "documents_processed": 0,
            "chunks_indexed": 0,
            "total_processing_time": 0.0,
            "embedding_time": 0.0,
            "storage_time": 0.0,
            "duplicates_skipped": 0,
            "failed_documents": 0
        }
        
        # Ensure collection exists
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize the vector collection"""
        try:
            # Get embedding dimension
            embedding_dim = self.embeddings_service.get_embedding_dimension()
            
            if not self.vector_store.collection_exists(self.collection_name):
                success = self.vector_store.create_collection(
                    collection_name=self.collection_name,
                    vector_size=embedding_dim,
                    distance_metric="Cosine"
                )
                
                if not success:
                    raise IndexingError(f"Failed to create collection: {self.collection_name}")
                
                logger.info(f"Created collection '{self.collection_name}' with dimension {embedding_dim}")
            else:
                logger.info(f"Using existing collection '{self.collection_name}'")
                
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise IndexingError(f"Collection initialization failed: {e}")
    
    def _generate_document_id(self, file_path: str, content_hash: str) -> str:
        """Generate unique document ID"""
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f"doc_{path_hash}_{content_hash[:8]}"
    
    def _is_duplicate(self, content_hash: str) -> bool:
        """Check if document is a duplicate"""
        return self.enable_deduplication and content_hash in self.content_hashes
    
    def _create_chunk_metadata(self, chunk: TextChunk, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for vector storage"""
        metadata = {
            "chunk_id": chunk.chunk_id,
            "document_id": document_metadata.get("document_id"),
            "file_path": document_metadata.get("file_path"),
            "chunk_index": chunk.metadata.get("chunk_index", 0),
            "start_position": chunk.start_index,
            "end_position": chunk.end_index,
            "word_count": chunk.word_count,
            "character_count": len(chunk.content),
            "chunk_method": chunk.metadata.get("chunk_method", "unknown"),
            "indexed_at": datetime.now().isoformat()
        }
        
        # Add document metadata
        metadata.update({
            f"doc_{k}": v for k, v in document_metadata.items()
            if k not in ["content", "chunks"] and isinstance(v, (str, int, float, bool))
        })
        
        return metadata
    
    async def _process_chunk_batch(self, chunk_batch: List[Tuple[TextChunk, Dict[str, Any]]]) -> List[Tuple[str, List[float], Dict[str, Any]]]:
        """Process a batch of chunks for embedding and storage"""
        try:
            # Extract chunks and metadata
            chunks = [item[0] for item in chunk_batch]
            chunk_texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings
            start_time = time.time()
            embeddings = await self.embeddings_service.generate_embeddings_async(
                chunk_texts, 
                max_concurrent=self.max_concurrent
            )
            self.stats["embedding_time"] += time.time() - start_time
            
            # Prepare data for storage
            storage_data = []
            for i, (chunk, doc_metadata) in enumerate(chunk_batch):
                chunk_metadata = self._create_chunk_metadata(chunk, doc_metadata)
                chunk_id = f"{doc_metadata['document_id']}_{chunk.chunk_id}"
                
                storage_data.append((chunk_id, embeddings[i], chunk_metadata))
            
            return storage_data
            
        except Exception as e:
            logger.error(f"Failed to process chunk batch: {e}")
            raise IndexingError(f"Batch processing failed: {e}")
    
    async def _store_vectors_batch(self, storage_data: List[Tuple[str, List[float], Dict[str, Any]]]):
        """Store a batch of vectors in the vector store"""
        try:
            start_time = time.time()
            
            # Prepare data for Qdrant
            vector_ids = [item[0] for item in storage_data]
            vectors = [item[1] for item in storage_data]
            payloads = [item[2] for item in storage_data]
            
            # Store in vector database
            success = self.vector_store.upsert_vectors(
                collection_name=self.collection_name,
                vectors=vectors,
                payloads=payloads,
                ids=vector_ids
            )
            
            if not success:
                raise IndexingError("Failed to store vectors in database")
            
            self.stats["storage_time"] += time.time() - start_time
            self.stats["chunks_indexed"] += len(storage_data)
            
        except Exception as e:
            logger.error(f"Failed to store vector batch: {e}")
            raise IndexingError(f"Vector storage failed: {e}")
    
    async def index_document(self, file_path: Union[str, Path]) -> IndexedDocument:
        """
        Index a single document
        
        Args:
            file_path: Path to the document file
            
        Returns:
            IndexedDocument: Indexing result metadata
            
        Raises:
            IndexingError: If indexing fails
        """
        file_path = Path(file_path)
        start_time = time.time()
        
        try:
            # Load document
            logger.info(f"Loading document: {file_path}")
            loaded_doc = self.document_loader.load_document(file_path)
            
            # Check for duplicates
            if self._is_duplicate(loaded_doc.metadata.content_hash):
                self.stats["duplicates_skipped"] += 1
                logger.info(f"Skipping duplicate document: {file_path}")
                raise IndexingError(f"Document is a duplicate: {file_path}")
            
            # Preprocess text
            logger.debug("Preprocessing document text")
            processed_text = self.text_preprocessor.process_text(
                loaded_doc.content,
                preserve_metadata=loaded_doc.metadata.to_dict()
            )
            
            # Chunk text
            logger.debug("Chunking document text")
            chunks = self.text_chunker.chunk_adaptive(
                processed_text.content,
                source_metadata=processed_text.metadata
            )
            
            if not chunks:
                raise IndexingError("No chunks created from document")
            
            # Generate document ID
            document_id = self._generate_document_id(str(file_path), loaded_doc.metadata.content_hash)
            
            # Prepare document metadata
            doc_metadata = {
                "document_id": document_id,
                "file_path": str(file_path),
                "content_hash": loaded_doc.metadata.content_hash,
                "file_type": loaded_doc.metadata.file_type,
                "file_size": loaded_doc.metadata.file_size,
                "chunk_count": len(chunks),
                "processing_metadata": {
                    "preprocessing_transformations": processed_text.transformations,
                    "chunking_method": "adaptive",
                    "original_length": processed_text.original_length,
                    "processed_length": processed_text.processed_length
                }
            }
            
            # Process chunks in batches
            logger.info(f"Processing {len(chunks)} chunks in batches of {self.batch_size}")
            total_tokens = 0
            
            for i in range(0, len(chunks), self.batch_size):
                batch_chunks = chunks[i:i + self.batch_size]
                chunk_batch = [(chunk, doc_metadata) for chunk in batch_chunks]
                
                # Process batch
                storage_data = await self._process_chunk_batch(chunk_batch)
                
                # Store batch
                await self._store_vectors_batch(storage_data)
                
                # Update token count
                total_tokens += sum(chunk.word_count for chunk in batch_chunks)
                
                logger.debug(f"Processed batch {i // self.batch_size + 1}/{(len(chunks) - 1) // self.batch_size + 1}")
            
            # Record successful indexing
            processing_time = time.time() - start_time
            
            indexed_doc = IndexedDocument(
                document_id=document_id,
                file_path=str(file_path),
                content_hash=loaded_doc.metadata.content_hash,
                chunk_count=len(chunks),
                total_tokens=total_tokens,
                indexed_at=datetime.now(),
                processing_time_seconds=processing_time,
                metadata=doc_metadata
            )
            
            # Update tracking
            self.indexed_documents[document_id] = indexed_doc
            self.content_hashes.add(loaded_doc.metadata.content_hash)
            
            # Update statistics
            self.stats["documents_processed"] += 1
            self.stats["total_processing_time"] += processing_time
            
            logger.info(f"Successfully indexed document: {file_path} (ID: {document_id})")
            return indexed_doc
            
        except Exception as e:
            self.stats["failed_documents"] += 1
            logger.error(f"Failed to index document {file_path}: {e}")
            raise IndexingError(f"Document indexing failed: {e}")
    
    async def index_documents(self, file_paths: List[Union[str, Path]]) -> List[IndexedDocument]:
        """
        Index multiple documents
        
        Args:
            file_paths: List of file paths to index
            
        Returns:
            List[IndexedDocument]: List of successfully indexed documents
        """
        indexed_docs = []
        
        logger.info(f"Starting indexing of {len(file_paths)} documents")
        
        for i, file_path in enumerate(file_paths):
            try:
                indexed_doc = await self.index_document(file_path)
                indexed_docs.append(indexed_doc)
                
                logger.info(f"Progress: {i + 1}/{len(file_paths)} documents indexed")
                
            except IndexingError as e:
                logger.warning(f"Skipped document {file_path}: {e}")
                continue
        
        logger.info(f"Indexing complete: {len(indexed_docs)} documents indexed successfully")
        return indexed_docs
    
    def get_document_info(self, document_id: str) -> Optional[IndexedDocument]:
        """Get information about an indexed document"""
        return self.indexed_documents.get(document_id)
    
    def list_indexed_documents(self) -> List[IndexedDocument]:
        """Get list of all indexed documents"""
        return list(self.indexed_documents.values())
    
    def remove_document(self, document_id: str) -> bool:
        """
        Remove an indexed document from the vector store
        
        Args:
            document_id: ID of the document to remove
            
        Returns:
            bool: True if removed successfully
        """
        try:
            if document_id not in self.indexed_documents:
                logger.warning(f"Document not found: {document_id}")
                return False
            
            indexed_doc = self.indexed_documents[document_id]
            
            # Generate chunk IDs to remove
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(indexed_doc.chunk_count)]
            
            # Remove from vector store
            success = self.vector_store.delete_vectors(self.collection_name, chunk_ids)
            
            if success:
                # Remove from tracking
                del self.indexed_documents[document_id]
                if indexed_doc.content_hash in self.content_hashes:
                    self.content_hashes.remove(indexed_doc.content_hash)
                
                logger.info(f"Removed document: {document_id}")
                return True
            else:
                logger.error(f"Failed to remove document from vector store: {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing document {document_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        stats = self.stats.copy()
        stats.update({
            "total_indexed_documents": len(self.indexed_documents),
            "total_unique_content": len(self.content_hashes),
            "collection_name": self.collection_name
        })
        return stats
    
    def export_index_metadata(self, export_path: Union[str, Path]) -> bool:
        """
        Export index metadata to JSON file
        
        Args:
            export_path: Path to export file
            
        Returns:
            bool: True if exported successfully
        """
        try:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "collection_name": self.collection_name,
                "statistics": self.get_stats(),
                "documents": [asdict(doc) for doc in self.indexed_documents.values()]
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Index metadata exported to: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export index metadata: {e}")
            return False


async def create_document_indexer(
    embeddings_service: GeminiEmbeddings,
    vector_store: QdrantVectorStore,
    collection_name: str = "documents"
) -> DocumentIndexer:
    """
    Factory function to create document indexer
    
    Args:
        embeddings_service: Gemini embeddings service
        vector_store: Qdrant vector store
        collection_name: Name of the collection
        
    Returns:
        DocumentIndexer: Configured document indexer
    """
    indexer = DocumentIndexer(
        embeddings_service=embeddings_service,
        vector_store=vector_store,
        collection_name=collection_name,
        batch_size=10,
        max_concurrent=3
    )
    
    return indexer 