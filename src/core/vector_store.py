"""
Vector Store Integration with Qdrant
Handles all vector database operations including collection management,
vector insertion, similarity search, and health monitoring.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
import asyncio

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    CollectionInfo,
    PointStruct,
    Filter,
    FieldCondition,
    Range,
    MatchValue
)

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Custom exception for vector store operations"""
    pass


class QdrantVectorStore:
    """
    Production-ready Qdrant vector store implementation
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        api_key: Optional[str] = None,
        https: bool = False,
        timeout: float = 30.0
    ):
        """
        Initialize Qdrant client connection
        
        Args:
            host: Qdrant server host
            port: Qdrant HTTP API port
            grpc_port: Qdrant gRPC port
            api_key: Optional API key for authentication
            https: Use HTTPS connection
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.api_key = api_key
        self.https = https
        self.timeout = timeout
        
        # Initialize client
        self.client = self._create_client()
        
    def _create_client(self) -> QdrantClient:
        """Create and configure Qdrant client"""
        try:
            client = QdrantClient(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                api_key=self.api_key,
                https=self.https,
                timeout=self.timeout
            )
            logger.info(f"Qdrant client initialized for {self.host}:{self.port}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise VectorStoreError(f"Client initialization failed: {e}")
    
    async def health_check(self) -> bool:
        """
        Check Qdrant server health and connectivity
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Test basic connectivity
            collections = self.client.get_collections()
            logger.info("Qdrant health check passed")
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "Cosine",
        on_disk_payload: bool = True
    ) -> bool:
        """
        Create a new collection in Qdrant
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance_metric: Distance metric (Cosine, Dot, Euclid)
            on_disk_payload: Store payload on disk for memory efficiency
            
        Returns:
            bool: True if created successfully
        """
        try:
            # Map distance metric string to Qdrant enum
            distance_map = {
                "Cosine": Distance.COSINE,
                "Dot": Distance.DOT,
                "Euclid": Distance.EUCLID
            }
            
            if distance_metric not in distance_map:
                raise VectorStoreError(f"Unsupported distance metric: {distance_metric}")
            
            # Check if collection already exists
            if self.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map[distance_metric],
                    on_disk=on_disk_payload
                )
            )
            
            logger.info(f"Collection '{collection_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise VectorStoreError(f"Collection creation failed: {e}")
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            bool: True if exists, False otherwise
        """
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return collection_name in collection_names
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """
        Get collection information and statistics
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            CollectionInfo: Collection information or None if not found
        """
        try:
            if not self.collection_exists(collection_name):
                return None
            
            info = self.client.get_collection(collection_name)
            logger.info(f"Retrieved info for collection '{collection_name}'")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def upsert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Insert or update vectors in the collection
        
        Args:
            collection_name: Name of the collection
            vectors: List of vector embeddings
            payloads: List of metadata dictionaries
            ids: Optional list of vector IDs (auto-generated if None)
            
        Returns:
            bool: True if successful
        """
        try:
            if not self.collection_exists(collection_name):
                raise VectorStoreError(f"Collection '{collection_name}' does not exist")
            
            if len(vectors) != len(payloads):
                raise VectorStoreError("Vectors and payloads must have the same length")
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid4()) for _ in range(len(vectors))]
            elif len(ids) != len(vectors):
                raise VectorStoreError("IDs must match the number of vectors")
            
            # Create points
            points = [
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                for point_id, vector, payload in zip(ids, vectors, payloads)
            ]
            
            # Upsert points
            operation_info = self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Upserted {len(points)} vectors to '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise VectorStoreError(f"Vector upsert failed: {e}")
    
    def similarity_search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform similarity search in the collection
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector for similarity search
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold
            metadata_filter: Optional metadata filtering conditions
            
        Returns:
            List of tuples: (id, score, payload)
        """
        try:
            if not self.collection_exists(collection_name):
                raise VectorStoreError(f"Collection '{collection_name}' does not exist")
            
            # Build filter if provided
            query_filter = None
            if metadata_filter:
                query_filter = self._build_metadata_filter(metadata_filter)
            
            # Perform search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            results = [
                (str(hit.id), hit.score, hit.payload or {})
                for hit in search_result
            ]
            
            logger.info(f"Found {len(results)} similar vectors in '{collection_name}'")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise VectorStoreError(f"Similarity search failed: {e}")
    
    def _build_metadata_filter(self, metadata_filter: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from metadata conditions
        
        Args:
            metadata_filter: Dictionary of filter conditions
            
        Returns:
            Filter: Qdrant filter object
        """
        conditions = []
        
        for field, value in metadata_filter.items():
            if isinstance(value, dict):
                # Handle range queries
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    range_condition = Range(
                        gte=value.get("gte"),
                        lte=value.get("lte"),
                        gt=value.get("gt"),
                        lt=value.get("lt")
                    )
                    conditions.append(
                        FieldCondition(key=field, range=range_condition)
                    )
                else:
                    # Handle exact match for dict values
                    conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )
            else:
                # Handle exact match
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
        
        return Filter(must=conditions)
    
    def delete_vectors(
        self,
        collection_name: str,
        vector_ids: List[str]
    ) -> bool:
        """
        Delete vectors by IDs
        
        Args:
            collection_name: Name of the collection
            vector_ids: List of vector IDs to delete
            
        Returns:
            bool: True if successful
        """
        try:
            if not self.collection_exists(collection_name):
                raise VectorStoreError(f"Collection '{collection_name}' does not exist")
            
            operation_info = self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=vector_ids
                )
            )
            
            logger.info(f"Deleted {len(vector_ids)} vectors from '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise VectorStoreError(f"Vector deletion failed: {e}")
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete an entire collection
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if successful
        """
        try:
            if not self.collection_exists(collection_name):
                logger.warning(f"Collection '{collection_name}' does not exist")
                return True
            
            self.client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise VectorStoreError(f"Collection deletion failed: {e}")
    
    def get_collection_stats(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get collection statistics
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict with collection statistics or None if collection doesn't exist
        """
        try:
            info = self.get_collection_info(collection_name)
            if not info:
                return None
            
            stats = {
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value,
                }
            }
            
            logger.info(f"Retrieved stats for collection '{collection_name}'")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return None
    
    def close(self):
        """Close the Qdrant client connection"""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
            logger.info("Qdrant client connection closed")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")
