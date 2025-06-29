"""
Retrieval System for RAG
Implements semantic similarity retrieval with advanced features including
relevance scoring, re-ranking, query expansion, and hybrid search capabilities.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

from .vector_store import QdrantVectorStore
from .embeddings import GeminiEmbeddings

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Search modes for retrieval"""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


@dataclass
class RetrievalResult:
    """Single retrieval result"""
    content: str
    score: float
    metadata: Dict[str, Any]
    document_id: str
    chunk_index: Optional[int] = None
    relevance_score: Optional[float] = None


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations"""
    max_results: int = 10
    score_threshold: float = 0.7
    enable_reranking: bool = True
    enable_query_expansion: bool = False
    search_mode: SearchMode = SearchMode.SEMANTIC
    diversity_penalty: float = 0.1
    metadata_boost: Dict[str, float] = None


class SemanticRetriever:
    """
    Production-ready semantic retrieval system
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embeddings_service: GeminiEmbeddings,
        collection_name: str,
        default_config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize semantic retriever
        
        Args:
            vector_store: Vector database instance
            embeddings_service: Embeddings service
            collection_name: Name of the collection to search
            default_config: Default retrieval configuration
        """
        self.vector_store = vector_store
        self.embeddings_service = embeddings_service
        self.collection_name = collection_name
        self.default_config = default_config or RetrievalConfig()
        
        # Initialize statistics
        self.stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "average_results_returned": 0.0,
            "query_expansions": 0,
            "reranking_operations": 0
        }
        
        logger.info(f"Semantic retriever initialized for collection: {collection_name}")
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms
        
        Args:
            query: Original query
            
        Returns:
            List[str]: Expanded queries including original
        """
        # Simple query expansion - can be enhanced with more sophisticated methods
        expansions = [query]
        
        # Add variations by removing common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = query.lower().split()
        filtered_words = [w for w in words if w not in stop_words]
        
        if len(filtered_words) < len(words):
            expansions.append(' '.join(filtered_words))
        
        # Add individual important terms
        if len(filtered_words) > 1:
            expansions.extend(filtered_words[-2:])  # Last 2 words often most important
        
        self.stats["query_expansions"] += len(expansions) - 1
        logger.debug(f"Expanded query to {len(expansions)} variations")
        
        return expansions
    
    def _calculate_relevance_score(
        self, 
        query: str, 
        content: str, 
        similarity_score: float,
        metadata: Dict[str, Any]
    ) -> float:
        """
        Calculate enhanced relevance score
        
        Args:
            query: Original query
            content: Retrieved content
            similarity_score: Vector similarity score
            metadata: Document metadata
            
        Returns:
            float: Enhanced relevance score
        """
        # Start with similarity score
        score = similarity_score
        
        # Boost based on exact keyword matches
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        keyword_overlap = len(query_words.intersection(content_words)) / len(query_words)
        score += keyword_overlap * 0.1
        
        # Boost based on content length (prefer substantial content)
        length_score = min(len(content) / 1000, 1.0) * 0.05
        score += length_score
        
        # Apply metadata boosts if configured
        if self.default_config.metadata_boost:
            for key, boost in self.default_config.metadata_boost.items():
                if key in metadata and metadata[key]:
                    score += boost
        
        return min(score, 1.0)
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[RetrievalResult],
        config: RetrievalConfig
    ) -> List[RetrievalResult]:
        """
        Re-rank results based on enhanced scoring
        
        Args:
            query: Original query
            results: Initial retrieval results
            config: Retrieval configuration
            
        Returns:
            List[RetrievalResult]: Re-ranked results
        """
        if not config.enable_reranking or not results:
            return results
        
        # Calculate enhanced relevance scores
        for result in results:
            result.relevance_score = self._calculate_relevance_score(
                query, result.content, result.score, result.metadata
            )
        
        # Sort by relevance score
        reranked = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        
        # Apply diversity penalty to reduce redundancy
        if config.diversity_penalty > 0:
            reranked = self._apply_diversity_penalty(reranked, config.diversity_penalty)
        
        self.stats["reranking_operations"] += 1
        logger.debug(f"Re-ranked {len(results)} results")
        
        return reranked
    
    def _apply_diversity_penalty(
        self, 
        results: List[RetrievalResult], 
        penalty: float
    ) -> List[RetrievalResult]:
        """
        Apply diversity penalty to reduce redundant results
        
        Args:
            results: Results to diversify
            penalty: Penalty factor for similar content
            
        Returns:
            List[RetrievalResult]: Diversified results
        """
        if len(results) <= 1:
            return results
        
        diversified = [results[0]]  # Always include top result
        
        for candidate in results[1:]:
            # Calculate similarity to already selected results
            max_similarity = 0.0
            
            for selected in diversified:
                # Simple content similarity based on word overlap
                candidate_words = set(candidate.content.lower().split())
                selected_words = set(selected.content.lower().split())
                
                if candidate_words and selected_words:
                    overlap = len(candidate_words.intersection(selected_words))
                    similarity = overlap / len(candidate_words.union(selected_words))
                    max_similarity = max(max_similarity, similarity)
            
            # Apply penalty based on similarity
            candidate.relevance_score *= (1 - penalty * max_similarity)
            diversified.append(candidate)
        
        # Re-sort after applying penalties
        return sorted(diversified, key=lambda x: x.relevance_score, reverse=True)
    
    def _search_semantic(
        self, 
        query: str, 
        config: RetrievalConfig
    ) -> List[RetrievalResult]:
        """
        Perform semantic search using embeddings
        
        Args:
            query: Search query
            config: Retrieval configuration
            
        Returns:
            List[RetrievalResult]: Search results
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings_service.generate_embedding(query)
            
            # Perform vector search
            raw_results = self.vector_store.similarity_search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=config.max_results * 2,  # Get more for re-ranking
                score_threshold=config.score_threshold
            )
            
            # Convert to RetrievalResult objects
            results = []
            for doc_id, score, metadata in raw_results:
                content = metadata.get('content', '')
                
                result = RetrievalResult(
                    content=content,
                    score=score,
                    metadata=metadata,
                    document_id=doc_id,
                    chunk_index=metadata.get('chunk_index')
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    def _search_keyword(
        self, 
        query: str, 
        config: RetrievalConfig
    ) -> List[RetrievalResult]:
        """
        Perform keyword-based search
        
        Args:
            query: Search query
            config: Retrieval configuration
            
        Returns:
            List[RetrievalResult]: Search results
        """
        # Simplified keyword search - in production, use dedicated search engine
        query_words = set(query.lower().split())
        
        # This is a placeholder - in real implementation, would use
        # dedicated keyword search index like Elasticsearch
        logger.warning("Keyword search not fully implemented - falling back to semantic search")
        return self._search_semantic(query, config)
    
    def retrieve(
        self, 
        query: str, 
        config: Optional[RetrievalConfig] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            config: Retrieval configuration (uses default if None)
            metadata_filter: Optional metadata filter
            
        Returns:
            List[RetrievalResult]: Retrieved and ranked results
            
        Raises:
            Exception: If retrieval fails
        """
        if not query or not query.strip():
            return []
        
        config = config or self.default_config
        
        try:
            self.stats["total_queries"] += 1
            
            # Expand query if enabled
            queries = [query]
            if config.enable_query_expansion:
                queries = self._expand_query(query)
            
            # Retrieve results for all query variations
            all_results = []
            
            for q in queries:
                if config.search_mode == SearchMode.SEMANTIC:
                    results = self._search_semantic(q, config)
                elif config.search_mode == SearchMode.KEYWORD:
                    results = self._search_keyword(q, config)
                elif config.search_mode == SearchMode.HYBRID:
                    # Combine semantic and keyword results
                    semantic_results = self._search_semantic(q, config)
                    keyword_results = self._search_keyword(q, config)
                    results = self._merge_results(semantic_results, keyword_results)
                else:
                    results = self._search_semantic(q, config)
                
                all_results.extend(results)
            
            # Remove duplicates based on document_id
            unique_results = {}
            for result in all_results:
                key = result.document_id
                if key not in unique_results or result.score > unique_results[key].score:
                    unique_results[key] = result
            
            results = list(unique_results.values())
            
            # Re-rank results
            results = self._rerank_results(query, results, config)
            
            # Apply final filtering and limits
            filtered_results = [
                r for r in results 
                if r.relevance_score >= config.score_threshold
            ][:config.max_results]
            
            # Update statistics
            self.stats["successful_retrievals"] += 1
            current_avg = self.stats["average_results_returned"]
            total_queries = self.stats["successful_retrievals"]
            self.stats["average_results_returned"] = (
                (current_avg * (total_queries - 1) + len(filtered_results)) / total_queries
            )
            
            logger.info(f"Retrieved {len(filtered_results)} results for query: {query[:50]}...")
            return filtered_results
            
        except Exception as e:
            self.stats["failed_retrievals"] += 1
            logger.error(f"Retrieval failed for query '{query}': {e}")
            raise
    
    def _merge_results(
        self, 
        semantic_results: List[RetrievalResult], 
        keyword_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Merge semantic and keyword search results
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            
        Returns:
            List[RetrievalResult]: Merged results
        """
        # Simple merging strategy - can be enhanced
        merged = {}
        
        # Add semantic results with weight
        for result in semantic_results:
            key = result.document_id
            result.score *= 0.7  # Weight semantic score
            merged[key] = result
        
        # Add keyword results with weight
        for result in keyword_results:
            key = result.document_id
            result.score *= 0.3  # Weight keyword score
            
            if key in merged:
                # Combine scores
                merged[key].score += result.score
            else:
                merged[key] = result
        
        return list(merged.values())
    
    def get_similar_documents(
        self, 
        document_id: str, 
        limit: int = 5
    ) -> List[RetrievalResult]:
        """
        Find documents similar to a given document
        
        Args:
            document_id: ID of reference document
            limit: Maximum number of similar documents
            
        Returns:
            List[RetrievalResult]: Similar documents
        """
        try:
            # Get the reference document's embedding
            # This would require storing embeddings with document IDs
            # For now, implement a placeholder
            logger.warning("get_similar_documents not fully implemented")
            return []
            
        except Exception as e:
            logger.error(f"Failed to find similar documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            **self.stats,
            "collection_name": self.collection_name,
            "default_config": {
                "max_results": self.default_config.max_results,
                "score_threshold": self.default_config.score_threshold,
                "search_mode": self.default_config.search_mode.value,
                "enable_reranking": self.default_config.enable_reranking,
                "enable_query_expansion": self.default_config.enable_query_expansion
            }
        }
    
    def reset_stats(self) -> None:
        """Reset retrieval statistics"""
        self.stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "average_results_returned": 0.0,
            "query_expansions": 0,
            "reranking_operations": 0
        }
        logger.info("Retrieval statistics reset")
    
    async def health_check(self) -> bool:
        """
        Check if the retrieval system is healthy
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Test basic retrieval
            test_results = await asyncio.get_event_loop().run_in_executor(
                None, self.retrieve, "test query", RetrievalConfig(max_results=1)
            )
            return True  # If no exception, consider healthy
        except Exception as e:
            logger.error(f"Retrieval health check failed: {e}")
            return False


# Utility functions
def create_retriever(
    vector_store: QdrantVectorStore,
    embeddings_service: GeminiEmbeddings,
    collection_name: str,
    **config_kwargs
) -> SemanticRetriever:
    """
    Factory function to create retriever
    
    Args:
        vector_store: Vector database instance
        embeddings_service: Embeddings service
        collection_name: Collection name to search
        **config_kwargs: Configuration parameters
        
    Returns:
        SemanticRetriever: Configured retriever
    """
    config = RetrievalConfig(**config_kwargs)
    return SemanticRetriever(vector_store, embeddings_service, collection_name, config)


def get_default_retriever(
    vector_store: QdrantVectorStore,
    embeddings_service: GeminiEmbeddings
) -> Optional[SemanticRetriever]:
    """
    Get default retriever from environment variables
    
    Args:
        vector_store: Vector database instance
        embeddings_service: Embeddings service
        
    Returns:
        SemanticRetriever: Default retriever or None if not configured
    """
    import os
    
    collection_name = os.getenv("COLLECTION_NAME", "confluence_documents")
    
    config = RetrievalConfig(
        max_results=int(os.getenv("RETRIEVAL_MAX_RESULTS", "10")),
        score_threshold=float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.7")),
        enable_reranking=os.getenv("ENABLE_RERANKING", "true").lower() == "true",
        enable_query_expansion=os.getenv("ENABLE_QUERY_EXPANSION", "false").lower() == "true"
    )
    
    return SemanticRetriever(vector_store, embeddings_service, collection_name, config) 