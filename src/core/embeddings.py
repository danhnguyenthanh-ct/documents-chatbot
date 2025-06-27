"""
Gemini Embeddings Service
Handles text embedding generation using Google's Gemini embedding models
with caching, batch processing, and robust error handling.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import time
import hashlib
import json
from functools import lru_cache
import asyncio

import google.generativeai as genai
from google.generativeai.types import EmbedContentResponse
from google.api_core import retry
from google.api_core.exceptions import (
    GoogleAPIError, 
    RetryError, 
    ResourceExhausted,
    InvalidArgument
)

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding operations"""
    pass


class GeminiEmbeddings:
    """
    Production-ready Gemini embeddings service with caching and batch processing
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "models/embedding-001",
        max_batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_size: int = 1000
    ):
        """
        Initialize Gemini embeddings client
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini embedding model name
            max_batch_size: Maximum documents per batch
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Base delay between retries in seconds
            cache_size: LRU cache size for embeddings
        """
        self.api_key = api_key
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_size = cache_size
        
        # Configure Gemini client
        genai.configure(api_key=api_key)
        
        # Initialize statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens_processed": 0,
            "failed_requests": 0,
            "batch_requests": 0
        }
        
        # Initialize embedding cache
        self._cache: Dict[str, List[float]] = {}
        
        logger.info(f"Gemini embeddings initialized with model: {model_name}")
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if exists"""
        cache_key = self._generate_cache_key(text)
        if cache_key in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[cache_key]
        
        self.stats["cache_misses"] += 1
        return None
    
    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding with LRU eviction"""
        cache_key = self._generate_cache_key(text)
        
        # Implement simple LRU by removing oldest when cache is full
        if len(self._cache) >= self.cache_size:
            # Remove first item (oldest)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = embedding
    
    @retry.Retry(
        predicate=retry.if_exception_type(
            GoogleAPIError, 
            ResourceExhausted,
            ConnectionError
        ),
        initial=1.0,
        maximum=60.0,
        multiplier=2.0,
        deadline=300.0
    )
    def _generate_embedding_with_retry(self, text: str) -> List[float]:
        """Generate embedding with automatic retry logic"""
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            
            if not response or not hasattr(response, 'embedding'):
                raise EmbeddingError("Invalid response from Gemini embedding API")
            
            return response.embedding
            
        except InvalidArgument as e:
            logger.error(f"Invalid argument for embedding generation: {e}")
            raise EmbeddingError(f"Invalid input: {e}")
        except ResourceExhausted as e:
            logger.warning(f"Rate limit exceeded, will retry: {e}")
            raise
        except GoogleAPIError as e:
            logger.error(f"Google API error during embedding generation: {e}")
            raise EmbeddingError(f"API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during embedding generation: {e}")
            raise EmbeddingError(f"Unexpected error: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingError("Input text cannot be empty")
        
        # Check cache first
        cached_embedding = self._get_cached_embedding(text)
        if cached_embedding is not None:
            logger.debug("Retrieved embedding from cache")
            return cached_embedding
        
        try:
            # Generate new embedding
            embedding = self._generate_embedding_with_retry(text)
            
            # Cache the result
            self._cache_embedding(text, embedding)
            
            # Update statistics
            self.stats["total_requests"] += 1
            self.stats["total_tokens_processed"] += len(text.split())
            
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def generate_embeddings_batch(
        self, 
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of input texts to embed
            batch_size: Override default batch size
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            EmbeddingError: If batch embedding generation fails
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.max_batch_size
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            logger.info(f"Processing batch {i//batch_size + 1} of {len(batch)} texts")
            
            for text in batch:
                try:
                    embedding = self.generate_embedding(text)
                    batch_embeddings.append(embedding)
                except EmbeddingError as e:
                    logger.error(f"Failed to generate embedding for text in batch: {e}")
                    # Return zero vector for failed embeddings
                    batch_embeddings.append([0.0] * 768)  # Default dimension
            
            embeddings.extend(batch_embeddings)
            self.stats["batch_requests"] += 1
            
            # Add small delay between batches to respect rate limits
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        logger.info(f"Generated {len(embeddings)} embeddings in batches")
        return embeddings
    
    async def generate_embeddings_async(
        self, 
        texts: List[str],
        batch_size: Optional[int] = None,
        max_concurrent: int = 5
    ) -> List[List[float]]:
        """
        Generate embeddings asynchronously with concurrency control
        
        Args:
            texts: List of input texts to embed
            batch_size: Override default batch size
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.max_batch_size
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single_async(text: str) -> List[float]:
            async with semaphore:
                # Run synchronous embedding generation in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self.generate_embedding, text
                )
        
        # Create tasks for all texts
        tasks = [generate_single_async(text) for text in texts]
        
        # Execute with concurrency control
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and convert to embeddings list
        result = []
        for i, embedding in enumerate(embeddings):
            if isinstance(embedding, Exception):
                logger.error(f"Failed to generate embedding for text {i}: {embedding}")
                result.append([0.0] * 768)  # Default dimension for failed embeddings
            else:
                result.append(embedding)
        
        return result
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for this model
        
        Returns:
            int: Embedding dimension
        """
        # Test with a simple text to get dimension
        try:
            test_embedding = self.generate_embedding("test")
            return len(test_embedding)
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            return 768  # Default Gemini embedding dimension
    
    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = (
            self.stats["cache_hits"] / total_requests 
            if total_requests > 0 else 0
        )
        
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
            "cache_hit_rate": hit_rate,
            "total_cached_items": len(self._cache)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        cache_stats = self.get_cache_stats()
        
        return {
            **self.stats,
            **cache_stats,
            "model_name": self.model_name,
            "max_batch_size": self.max_batch_size
        }
    
    def health_check(self) -> bool:
        """
        Check if the embedding service is healthy
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Test with a simple embedding
            test_embedding = self.generate_embedding("health check")
            return len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return False


# Utility functions for common operations
def create_embeddings_service(
    api_key: str, 
    model_name: str = "models/embedding-001"
) -> GeminiEmbeddings:
    """
    Factory function to create embeddings service
    
    Args:
        api_key: Google API key
        model_name: Gemini embedding model name
        
    Returns:
        GeminiEmbeddings: Configured embeddings service
    """
    return GeminiEmbeddings(api_key=api_key, model_name=model_name)


@lru_cache(maxsize=1)
def get_default_embeddings_service() -> Optional[GeminiEmbeddings]:
    """
    Get default embeddings service (singleton pattern)
    Requires GOOGLE_API_KEY environment variable
    
    Returns:
        GeminiEmbeddings: Default embeddings service or None if not configured
    """
    import os
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not found in environment variables")
        return None
    
    model_name = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
    return GeminiEmbeddings(api_key=api_key, model_name=model_name) 