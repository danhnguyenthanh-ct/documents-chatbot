"""
Response Post-Processing Module
Handles response cleaning, formatting, citation management, validation,
safety checks, and caching for the RAG system.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Types of responses"""
    DIRECT_ANSWER = "direct_answer"
    SUMMARIZED = "summarized"
    INCOMPLETE = "incomplete"
    NO_CONTEXT = "no_context"
    ERROR = "error"


class SafetyLevel(Enum):
    """Safety levels for responses"""
    SAFE = "safe"
    NEEDS_REVIEW = "needs_review"
    UNSAFE = "unsafe"


@dataclass
class ProcessedResponse:
    """Container for processed response with metadata"""
    content: str
    response_type: ResponseType
    safety_level: SafetyLevel
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float
    cache_key: Optional[str] = None
    processed_at: datetime = None
    
    def __post_init__(self):
        if self.processed_at is None:
            self.processed_at = datetime.now()


class ResponseCache:
    """Simple in-memory cache for processed responses"""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        cached_at = datetime.fromisoformat(entry["cached_at"])
        return datetime.now() - cached_at > self.ttl
    
    def get(self, key: str) -> Optional[ProcessedResponse]:
        """Get cached response"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if self._is_expired(entry):
            del self.cache[key]
            return None
        
        # Increment hit count
        entry["hits"] += 1
        
        # Reconstruct ProcessedResponse
        response_data = entry["response"]
        response_data["processed_at"] = datetime.fromisoformat(response_data["processed_at"])
        return ProcessedResponse(**response_data)
    
    def set(self, key: str, response: ProcessedResponse):
        """Cache a response"""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            # Remove entries with lowest hit count
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1]["hits"])
            for old_key, _ in sorted_items[:10]:  # Remove 10 oldest
                del self.cache[old_key]
        
        # Store response
        response_dict = asdict(response)
        response_dict["processed_at"] = response.processed_at.isoformat()
        
        self.cache[key] = {
            "response": response_dict,
            "cached_at": datetime.now().isoformat(),
            "hits": 0
        }
    
    def clear(self):
        """Clear all cached responses"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(entry["hits"] for entry in self.cache.values())
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "ttl_hours": self.ttl.total_seconds() / 3600
        }


class ResponsePostProcessor:
    """
    Advanced response post-processing with validation, safety, and caching
    """
    
    def __init__(
        self,
        enable_caching: bool = True,
        cache_size: int = 1000,
        cache_ttl_hours: int = 24,
        max_response_length: int = 10000
    ):
        """
        Initialize response post-processor
        
        Args:
            enable_caching: Enable response caching
            cache_size: Maximum number of cached responses
            cache_ttl_hours: Cache time-to-live in hours
            max_response_length: Maximum allowed response length
        """
        self.enable_caching = enable_caching
        self.max_response_length = max_response_length
        
        # Initialize cache
        self.cache = ResponseCache(cache_size, cache_ttl_hours) if enable_caching else None
        
        # Compile regex patterns for cleaning
        self._compile_patterns()
        
        # Initialize statistics
        self.stats = {
            "responses_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "safety_violations": 0,
            "citations_added": 0,
            "responses_truncated": 0,
            "total_processing_time": 0.0
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for text cleaning"""
        # Remove excessive whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Remove markdown artifacts that might leak through
        self.markdown_artifacts = re.compile(r'```|~~|__|\*\*\*')
        
        # Detect potential system prompts or instructions in response
        self.system_prompt_pattern = re.compile(
            r'(you are|assistant|system|instruction|prompt)', 
            re.IGNORECASE
        )
        
        # Citation patterns
        self.citation_pattern = re.compile(r'\[Source:\s*([^\]]+)\]')
        
        # Harmful content patterns
        self.unsafe_patterns = [
            re.compile(r'\b(hack|exploit|bypass|illegal)\b', re.IGNORECASE),
            re.compile(r'\b(personal|private|confidential)\s+information\b', re.IGNORECASE),
            re.compile(r'\b(generate|create)\s+(virus|malware)\b', re.IGNORECASE)
        ]
    
    def _generate_cache_key(self, raw_response: str, context: str, query: str) -> str:
        """Generate cache key for response"""
        content = f"{query}:{context}:{raw_response}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _clean_response_text(self, text: str) -> str:
        """Clean and normalize response text"""
        # Remove excessive whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove markdown artifacts
        text = self.markdown_artifacts.sub('', text)
        
        # Clean up line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _detect_response_type(self, response: str, context: str) -> ResponseType:
        """Detect the type of response"""
        response_lower = response.lower()
        
        if not context.strip():
            return ResponseType.NO_CONTEXT
        
        if any(phrase in response_lower for phrase in [
            "i don't have enough information",
            "insufficient information",
            "cannot answer",
            "not enough context"
        ]):
            return ResponseType.INCOMPLETE
        
        if len(response) < 50:
            return ResponseType.DIRECT_ANSWER
        
        if any(phrase in response_lower for phrase in [
            "based on the provided",
            "according to",
            "the documents indicate",
            "summarizing"
        ]):
            return ResponseType.SUMMARIZED
        
        return ResponseType.DIRECT_ANSWER
    
    def _evaluate_safety(self, response: str) -> Tuple[SafetyLevel, List[str]]:
        """Evaluate response safety"""
        issues = []
        
        # Check for unsafe patterns
        for pattern in self.unsafe_patterns:
            if pattern.search(response):
                issues.append(f"Contains potentially unsafe content: {pattern.pattern}")
        
        # Check for system prompt leakage
        if self.system_prompt_pattern.search(response[:100]):
            issues.append("Potential system prompt leakage detected")
        
        # Check response length
        if len(response) > self.max_response_length:
            issues.append(f"Response too long: {len(response)} chars")
        
        # Determine safety level
        if not issues:
            return SafetyLevel.SAFE, []
        elif len(issues) == 1 and "too long" in issues[0]:
            return SafetyLevel.NEEDS_REVIEW, issues
        else:
            return SafetyLevel.UNSAFE, issues
    
    def _extract_citations(self, response: str) -> List[Dict[str, Any]]:
        """Extract and format citations from response"""
        citations = []
        citation_matches = self.citation_pattern.findall(response)
        
        for i, citation in enumerate(set(citation_matches)):  # Remove duplicates
            citations.append({
                "id": i + 1,
                "source": citation.strip(),
                "type": "document",
                "mentioned_in_response": True
            })
        
        return citations
    
    def _add_source_citations(
        self, 
        response: str, 
        sources: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Add proper citations to response"""
        if not sources:
            return response, []
        
        # Extract existing citations
        existing_citations = self._extract_citations(response)
        
        # Create source references
        all_citations = []
        citation_map = {}
        
        # Add sources as citations
        for i, source in enumerate(sources):
            source_name = source.get("file_path", f"Source {i+1}")
            if "/" in source_name:
                source_name = source_name.split("/")[-1]
            
            citation = {
                "id": len(all_citations) + 1,
                "source": source_name,
                "type": "document",
                "relevance_score": source.get("relevance_score", 0.0),
                "chunk_index": source.get("chunk_index", 0),
                "mentioned_in_response": any(
                    source_name.lower() in cite["source"].lower() 
                    for cite in existing_citations
                )
            }
            
            all_citations.append(citation)
            citation_map[source_name] = citation["id"]
        
        # Add citations section if sources exist
        if all_citations:
            citations_section = "\n\n**Sources:**\n"
            for citation in all_citations:
                relevance = citation.get("relevance_score", 0.0)
                citations_section += f"{citation['id']}. {citation['source']}"
                if relevance > 0:
                    citations_section += f" (relevance: {relevance:.2f})"
                citations_section += "\n"
            
            response += citations_section
        
        return response, all_citations
    
    def _validate_response_quality(self, response: str, query: str) -> List[str]:
        """Validate response quality and coherence"""
        issues = []
        
        # Check if response is too short
        if len(response.strip()) < 10:
            issues.append("Response too short")
        
        # Check if response addresses the query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        
        if overlap < min(2, len(query_words) // 2):
            issues.append("Response may not address the query")
        
        # Check for repetitive content
        sentences = response.split('.')
        if len(sentences) > 3:
            unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
            if len(unique_sentences) < len(sentences) * 0.8:
                issues.append("Response contains repetitive content")
        
        return issues
    
    def process_response(
        self,
        raw_response: str,
        query: str,
        context: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        enable_cache: bool = True
    ) -> ProcessedResponse:
        """
        Process raw LLM response with cleaning, validation, and enhancement
        
        Args:
            raw_response: Raw response from LLM
            query: Original user query
            context: Retrieved context used for generation
            sources: List of source documents
            enable_cache: Whether to use caching for this response
            
        Returns:
            ProcessedResponse: Processed response with metadata
        """
        start_time = datetime.now()
        
        # Check cache first
        cache_key = None
        if self.enable_caching and enable_cache and self.cache:
            cache_key = self._generate_cache_key(raw_response, context, query)
            cached_response = self.cache.get(cache_key)
            if cached_response:
                self.stats["cache_hits"] += 1
                return cached_response
            self.stats["cache_misses"] += 1
        
        try:
            # Clean response text
            cleaned_response = self._clean_response_text(raw_response)
            
            # Detect response type
            response_type = self._detect_response_type(cleaned_response, context)
            
            # Evaluate safety
            safety_level, safety_issues = self._evaluate_safety(cleaned_response)
            
            # Validate quality
            quality_issues = self._validate_response_quality(cleaned_response, query)
            
            # Add citations if sources provided
            citations = []
            if sources:
                cleaned_response, citations = self._add_source_citations(cleaned_response, sources)
                self.stats["citations_added"] += len(citations)
            
            # Truncate if too long
            if len(cleaned_response) > self.max_response_length:
                cleaned_response = cleaned_response[:self.max_response_length] + "... [Response truncated]"
                self.stats["responses_truncated"] += 1
            
            # Build metadata
            metadata = {
                "original_length": len(raw_response),
                "processed_length": len(cleaned_response),
                "safety_issues": safety_issues,
                "quality_issues": quality_issues,
                "source_count": len(sources) if sources else 0,
                "query_length": len(query),
                "context_length": len(context)
            }
            
            # Create processed response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            processed_response = ProcessedResponse(
                content=cleaned_response,
                response_type=response_type,
                safety_level=safety_level,
                citations=citations,
                metadata=metadata,
                processing_time=processing_time,
                cache_key=cache_key
            )
            
            # Cache the response
            if self.enable_caching and enable_cache and self.cache and cache_key:
                self.cache.set(cache_key, processed_response)
            
            # Update statistics
            self.stats["responses_processed"] += 1
            self.stats["total_processing_time"] += processing_time
            
            if safety_level == SafetyLevel.UNSAFE:
                self.stats["safety_violations"] += 1
            
            logger.info(f"Processed response: {response_type.value}, safety: {safety_level.value}")
            return processed_response
            
        except Exception as e:
            logger.error(f"Response processing failed: {e}")
            
            # Return error response
            return ProcessedResponse(
                content=f"Error processing response: {str(e)}",
                response_type=ResponseType.ERROR,
                safety_level=SafetyLevel.UNSAFE,
                citations=[],
                metadata={"error": str(e)},
                processing_time=(datetime.now() - start_time).total_seconds(),
                cache_key=None
            )
    
    def format_for_display(self, processed_response: ProcessedResponse) -> str:
        """Format processed response for display to user"""
        content = processed_response.content
        
        # Add safety warning if needed
        if processed_response.safety_level == SafetyLevel.UNSAFE:
            content = "⚠️ **Safety Warning**: This response may contain inappropriate content.\n\n" + content
        elif processed_response.safety_level == SafetyLevel.NEEDS_REVIEW:
            content = "⚠️ **Note**: This response may need review.\n\n" + content
        
        # Add response type indicator
        if processed_response.response_type == ResponseType.INCOMPLETE:
            content = "📋 **Partial Answer**: " + content
        elif processed_response.response_type == ResponseType.NO_CONTEXT:
            content = "❓ **Limited Context**: " + content
        
        return content
    
    def get_response_summary(self, processed_response: ProcessedResponse) -> Dict[str, Any]:
        """Get summary information about processed response"""
        return {
            "response_type": processed_response.response_type.value,
            "safety_level": processed_response.safety_level.value,
            "citation_count": len(processed_response.citations),
            "content_length": len(processed_response.content),
            "processing_time": processed_response.processing_time,
            "has_issues": bool(
                processed_response.metadata.get("safety_issues") or 
                processed_response.metadata.get("quality_issues")
            ),
            "cached": processed_response.cache_key is not None
        }
    
    def clear_cache(self):
        """Clear response cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Response cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get post-processor statistics"""
        stats = self.stats.copy()
        if self.cache:
            stats.update(self.cache.get_stats())
        return stats


def create_response_post_processor(
    enable_caching: bool = True,
    max_response_length: int = 10000
) -> ResponsePostProcessor:
    """
    Factory function to create response post-processor
    
    Args:
        enable_caching: Enable response caching
        max_response_length: Maximum allowed response length
        
    Returns:
        ResponsePostProcessor: Configured post-processor
    """
    return ResponsePostProcessor(
        enable_caching=enable_caching,
        max_response_length=max_response_length
    ) 