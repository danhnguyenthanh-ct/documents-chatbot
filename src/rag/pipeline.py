"""
Main RAG Pipeline Module
Orchestrates the complete RAG workflow including query processing, embedding,
document retrieval, context preparation, and response generation.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from ..core.embeddings import GeminiEmbeddings
from ..core.llm import GeminiLLM  
from ..core.vector_store import QdrantVectorStore
from ..core.retriever import SemanticRetriever
from .prompts import PromptManager
from .chains import RAGChain, StreamingRAGChain, ConversationMemoryManager
from .post_processor import ResponsePostProcessor

logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    """Pipeline execution modes"""
    SIMPLE = "simple"
    CONVERSATIONAL = "conversational"
    STREAMING = "streaming"


@dataclass
class RAGRequest:
    """Request structure for RAG pipeline"""
    query: str
    session_id: Optional[str] = None
    mode: PipelineMode = PipelineMode.SIMPLE
    max_sources: int = 5
    include_metadata: bool = True
    enable_safety_check: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass 
class RAGResponse:
    """Response structure from RAG pipeline"""
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    session_id: Optional[str]
    processing_time: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class RAGPipeline:
    """
    Main RAG Pipeline orchestrating the complete workflow
    """
    
    def __init__(
        self,
        embeddings_service: GeminiEmbeddings,
        llm: GeminiLLM,
        vector_store: QdrantVectorStore,
        collection_name: str = "documents",
        enable_caching: bool = True
    ):
        """
        Initialize RAG pipeline
        
        Args:
            embeddings_service: Gemini embeddings service
            llm: Gemini LLM service
            vector_store: Qdrant vector store
            collection_name: Vector collection name
            enable_caching: Enable response caching
        """
        self.embeddings_service = embeddings_service
        self.llm = llm
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.enable_caching = enable_caching
        
        # Initialize components
        self.retriever = SemanticRetriever(
            embeddings_service=embeddings_service,
            vector_store=vector_store,
            collection_name=collection_name
        )
        
        self.prompt_manager = PromptManager()
        self.post_processor = ResponsePostProcessor(enable_caching=enable_caching)
        
        # Session management
        self.sessions: Dict[str, ConversationMemoryManager] = {}
        
        # Pipeline components
        self.simple_chain: Optional[RAGChain] = None
        self.streaming_chain: Optional[StreamingRAGChain] = None
        
        # Initialize chains
        self._initialize_chains()
        
        # Pipeline statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "active_sessions": 0,
            "mode_usage": {
                PipelineMode.SIMPLE: 0,
                PipelineMode.CONVERSATIONAL: 0,
                PipelineMode.STREAMING: 0
            }
        }
    
    def _initialize_chains(self):
        """Initialize RAG chains"""
        try:
            # Simple chain
            self.simple_chain = RAGChain(
                llm=self.llm,
                retriever=self.retriever,
                prompt_manager=self.prompt_manager,
                post_processor=self.post_processor
            )
            
            # Streaming chain
            self.streaming_chain = StreamingRAGChain(
                llm=self.llm,
                retriever=self.retriever,
                prompt_manager=self.prompt_manager,
                post_processor=self.post_processor
            )
            
            logger.info("RAG chains initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG chains: {e}")
            raise RuntimeError(f"Chain initialization failed: {e}")
    
    def _get_or_create_session(self, session_id: str) -> ConversationMemoryManager:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemoryManager()
            self.stats["active_sessions"] = len(self.sessions)
            logger.info(f"Created new session: {session_id}")
        
        return self.sessions[session_id]
    
    def _update_stats(self, processing_time: float, mode: PipelineMode, success: bool):
        """Update pipeline statistics"""
        self.stats["total_requests"] += 1
        
        if success:
            self.stats["successful_requests"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["successful_requests"]
            )
        else:
            self.stats["failed_requests"] += 1
        
        self.stats["mode_usage"][mode] += 1
    
    async def process_query(self, request: RAGRequest) -> RAGResponse:
        """
        Process a query through the RAG pipeline
        
        Args:
            request: RAG request with query and parameters
            
        Returns:
            RAGResponse: Generated response with metadata
        """
        start_time = datetime.now()
        
        try:
            # Validate request
            if not request.query.strip():
                raise ValueError("Query cannot be empty")
            
            # Safety check if enabled
            if request.enable_safety_check:
                is_safe, safety_message = self.prompt_manager.evaluate_query_safety(request.query)
                if not is_safe:
                    response = RAGResponse(
                        answer=f"I cannot process this query: {safety_message}",
                        sources=[],
                        metadata={"safety_issue": safety_message},
                        session_id=request.session_id,
                        processing_time=0.0,
                        timestamp=datetime.now()
                    )
                    
                    self._update_stats(0.0, request.mode, False)
                    return response
            
            # Handle different modes
            if request.mode == PipelineMode.CONVERSATIONAL and request.session_id:
                # Use session-specific memory
                memory_manager = self._get_or_create_session(request.session_id)
                
                # Create conversational chain with session memory
                conv_chain = RAGChain(
                    llm=self.llm,
                    retriever=self.retriever,
                    prompt_manager=self.prompt_manager,
                    post_processor=self.post_processor,
                    memory_manager=memory_manager
                )
                
                result = await conv_chain._acall({"query": request.query})
                
            elif request.mode == PipelineMode.SIMPLE:
                # Use simple chain without memory
                result = await self.simple_chain._acall({"query": request.query})
                
            else:
                # Default to simple mode
                result = await self.simple_chain._acall({"query": request.query})
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Build response
            response = RAGResponse(
                answer=result["answer"],
                sources=result["sources"][:request.max_sources],
                metadata=result["metadata"] if request.include_metadata else {},
                session_id=request.session_id,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            # Update statistics
            self._update_stats(processing_time, request.mode, True)
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Query processing failed: {e}")
            
            # Build error response
            response = RAGResponse(
                answer=f"I apologize, but I encountered an error: {str(e)}",
                sources=[],
                metadata={"error": str(e)},
                session_id=request.session_id,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            self._update_stats(processing_time, request.mode, False)
            return response
    
    async def stream_query(self, request: RAGRequest):
        """
        Stream query processing for real-time responses
        
        Args:
            request: RAG request with query and parameters
            
        Yields:
            Dict[str, Any]: Streaming response chunks
        """
        if not self.streaming_chain:
            yield {
                "type": "error",
                "message": "Streaming not available",
                "timestamp": datetime.now().isoformat()
            }
            return
        
        try:
            # Handle session memory for streaming
            if request.mode == PipelineMode.CONVERSATIONAL and request.session_id:
                memory_manager = self._get_or_create_session(request.session_id)
                
                # Create streaming chain with session memory
                streaming_chain = StreamingRAGChain(
                    llm=self.llm,
                    retriever=self.retriever,
                    prompt_manager=self.prompt_manager,
                    post_processor=self.post_processor,
                    memory_manager=memory_manager
                )
                
                async for chunk in streaming_chain.stream_response(request.query):
                    yield chunk
            else:
                async for chunk in self.streaming_chain.stream_response(request.query):
                    yield chunk
            
            # Update statistics
            self._update_stats(0.0, PipelineMode.STREAMING, True)
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self._update_stats(0.0, PipelineMode.STREAMING, False)
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a conversation session
        
        Args:
            session_id: Session ID to clear
            
        Returns:
            bool: True if session was cleared
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.stats["active_sessions"] = len(self.sessions)
            logger.info(f"Cleared session: {session_id}")
            return True
        
        return False
    
    def clear_all_sessions(self):
        """Clear all conversation sessions"""
        self.sessions.clear()
        self.stats["active_sessions"] = 0
        logger.info("Cleared all sessions")
    
    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session ID
            
        Returns:
            List[Dict[str, str]]: Conversation history
        """
        if session_id not in self.sessions:
            return []
        
        memory_manager = self.sessions[session_id]
        messages = memory_manager.get_conversation_history()
        
        history = []
        for msg in messages:
            if hasattr(msg, 'type'):
                role = "user" if msg.type == "human" else "assistant"
            else:
                role = "user" if "User:" in str(msg) else "assistant"
            
            history.append({
                "role": role,
                "content": str(msg.content),
                "timestamp": datetime.now().isoformat()
            })
        
        return history
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all pipeline components
        
        Returns:
            Dict[str, Any]: Health status of components
        """
        health_status = {
            "pipeline": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check embeddings service
            health_status["components"]["embeddings"] = (
                "healthy" if await self.embeddings_service.health_check() else "unhealthy"
            )
            
            # Check vector store
            health_status["components"]["vector_store"] = (
                "healthy" if await self.vector_store.health_check() else "unhealthy"
            )
            
            # Check LLM
            health_status["components"]["llm"] = (
                "healthy" if await self.llm.health_check() else "unhealthy"
            )
            
            # Check retriever
            health_status["components"]["retriever"] = (
                "healthy" if self.retriever and await self.retriever.health_check() else "unhealthy"
            )
            
            # Overall health
            unhealthy_components = [
                comp for comp, status in health_status["components"].items() 
                if status == "unhealthy"
            ]
            
            if unhealthy_components:
                health_status["pipeline"] = "degraded"
                health_status["issues"] = unhealthy_components
            
        except Exception as e:
            health_status["pipeline"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        stats = self.stats.copy()
        
        # Add component stats
        stats["component_stats"] = {
            "embeddings": self.embeddings_service.get_stats(),
            "retriever": self.retriever.get_stats() if hasattr(self.retriever, 'get_stats') else {},
            "post_processor": self.post_processor.get_stats(),
            "prompt_manager": self.prompt_manager.get_stats()
        }
        
        return stats
    
    def configure_prompt_version(self, prompt_type: str, version: str):
        """Configure prompt version for A/B testing"""
        from .prompts import PromptType, PromptVersion
        
        try:
            p_type = PromptType(prompt_type)
            p_version = PromptVersion(version)
            self.prompt_manager.set_active_version(p_type, p_version)
            logger.info(f"Set {prompt_type} prompt to version {version}")
        except ValueError as e:
            logger.error(f"Invalid prompt configuration: {e}")
            raise


async def create_rag_pipeline(
    embeddings_service: GeminiEmbeddings,
    llm: GeminiLLM,
    vector_store: QdrantVectorStore,
    collection_name: str = "documents"
) -> RAGPipeline:
    """
    Factory function to create RAG pipeline
    
    Args:
        embeddings_service: Gemini embeddings service
        llm: Gemini LLM service
        vector_store: Qdrant vector store
        collection_name: Vector collection name
        
    Returns:
        RAGPipeline: Configured RAG pipeline
    """
    pipeline = RAGPipeline(
        embeddings_service=embeddings_service,
        llm=llm,
        vector_store=vector_store,
        collection_name=collection_name
    )
    
    # Verify pipeline health
    health = await pipeline.health_check()
    if health["pipeline"] != "healthy":
        logger.warning(f"Pipeline health check failed: {health}")
    
    return pipeline 