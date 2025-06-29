"""
LangChain Integration Module
Custom LangChain chains for RAG with conversation memory management,
multi-turn support, and context window optimization.
"""

import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
import asyncio

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.chains.base import Chain
from langchain.callbacks.manager import AsyncCallbackManagerForChainRun

from ..core.llm import GeminiLLM
from ..core.retriever import SemanticRetriever
from .prompts import PromptManager, PromptVersion
from .post_processor import ResponsePostProcessor

logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """Manages conversation memory with different strategies"""
    
    def __init__(
        self,
        memory_type: str = "buffer_window",
        max_token_limit: int = 8000,
        window_size: int = 5
    ):
        self.memory_type = memory_type
        self.max_token_limit = max_token_limit
        self.window_size = window_size
        
        # Initialize appropriate memory
        if memory_type == "buffer_window":
            self.memory = ConversationBufferWindowMemory(
                k=window_size,
                return_messages=True,
                memory_key="chat_history"
            )
        elif memory_type == "summary_buffer":
            # Would need LLM for summarization - simplified for now
            self.memory = ConversationBufferWindowMemory(
                k=window_size,
                return_messages=True,
                memory_key="chat_history"
            )
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
        
        self.conversation_stats = {
            "total_turns": 0,
            "memory_updates": 0,
            "context_truncations": 0
        }
    
    def add_user_message(self, message: str):
        """Add user message to memory"""
        self.memory.chat_memory.add_user_message(message)
        self.conversation_stats["total_turns"] += 1
        self.conversation_stats["memory_updates"] += 1
    
    def add_ai_message(self, message: str):
        """Add AI message to memory"""
        self.memory.chat_memory.add_ai_message(message)
        self.conversation_stats["memory_updates"] += 1
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get conversation history as messages"""
        return self.memory.chat_memory.messages
    
    def get_conversation_context(self) -> str:
        """Get conversation history as formatted string"""
        messages = self.get_conversation_history()
        
        if not messages:
            return ""
        
        context_parts = []
        for msg in messages[-self.window_size:]:  # Last N messages
            if isinstance(msg, HumanMessage):
                context_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(context_parts)
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        return self.conversation_stats.copy()


class RAGChain(Chain):
    """Custom LangChain for RAG operations"""
    
    llm: GeminiLLM
    retriever: SemanticRetriever
    prompt_manager: PromptManager
    post_processor: ResponsePostProcessor
    memory_manager: Optional[ConversationMemoryManager] = None
    max_context_length: int = 15000
    min_context_length: int = 100
    max_sources: int = 5
    stats: Dict[str, Any] = {}
    
    def __init__(
        self,
        llm: GeminiLLM,
        retriever: SemanticRetriever,
        prompt_manager: PromptManager,
        post_processor: ResponsePostProcessor,
        memory_manager: Optional[ConversationMemoryManager] = None,
        **kwargs
    ):
        super().__init__(
            llm=llm,
            retriever=retriever,
            prompt_manager=prompt_manager,
            post_processor=post_processor,
            memory_manager=memory_manager or ConversationMemoryManager(),
            max_context_length=15000,
            min_context_length=100,
            max_sources=5,
            stats={
                "queries_processed": 0,
                "successful_retrievals": 0,
                "failed_retrievals": 0,
                "context_truncations": 0,
                "total_processing_time": 0.0
            },
            **kwargs
        )
    
    @property
    def input_keys(self) -> List[str]:
        """Input keys for the chain"""
        return ["query"]
    
    @property
    def output_keys(self) -> List[str]:
        """Output keys for the chain"""
        return ["answer", "sources", "metadata"]
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents"""
        if not retrieved_docs:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs[:self.max_sources]):
            doc_content = doc.get("content", "")
            doc_source = doc.get("file_path", f"Document {i+1}")
            
            # Add source identifier
            doc_text = f"[Source: {doc_source}]\n{doc_content}\n"
            
            # Check if adding this document would exceed limit
            if current_length + len(doc_text) > self.max_context_length:
                # Try to fit partial content
                remaining_length = self.max_context_length - current_length - 100  # Buffer
                if remaining_length > self.min_context_length:
                    truncated_content = doc_content[:remaining_length] + "... [truncated]"
                    doc_text = f"[Source: {doc_source}]\n{truncated_content}\n"
                    context_parts.append(doc_text)
                    self.stats["context_truncations"] += 1
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n".join(context_parts)
    
    def _prepare_metadata(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare metadata about retrieved documents"""
        if not retrieved_docs:
            return {
                "source_count": 0,
                "relevance_scores": "None",
                "content_types": "None"
            }
        
        relevance_scores = [doc.get("relevance_score", 0.0) for doc in retrieved_docs]
        content_types = list(set(doc.get("file_type", "unknown") for doc in retrieved_docs))
        
        return {
            "source_count": len(retrieved_docs),
            "relevance_scores": f"{min(relevance_scores):.2f} - {max(relevance_scores):.2f}",
            "content_types": ", ".join(content_types)
        }
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Execute the RAG chain (sync version)"""
        # This is a simplified sync version - async version below is preferred
        import asyncio
        return asyncio.run(self._acall(inputs, run_manager))
    
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Execute the RAG chain asynchronously"""
        start_time = datetime.now()
        
        try:
            query = inputs["query"]
            self.stats["queries_processed"] += 1
            
            # Safety check
            is_safe, safety_message = self.prompt_manager.evaluate_query_safety(query)
            if not is_safe:
                return {
                    "answer": f"I cannot process this query: {safety_message}",
                    "sources": [],
                    "metadata": {"safety_issue": safety_message}
                }
            
            # Retrieve relevant documents
            try:
                retrieval_results = self.retriever.retrieve(
                    query, 
                    config=None
                )[:self.max_sources]
                
                # Convert RetrievalResult objects to dictionaries
                retrieved_docs = []
                for result in retrieval_results:
                    print(f"result: {result}")
                    doc_dict = {
                        "content": result.content,
                        "relevance_score": result.relevance_score or result.score,
                        "file_path": result.metadata.get("file_path", "Unknown"),
                        "file_type": result.metadata.get("file_type", "unknown"),
                        "document_id": result.document_id,
                        "chunk_index": result.chunk_index,
                        "metadata": result.metadata
                    }
                    retrieved_docs.append(doc_dict)
                
                self.stats["successful_retrievals"] += 1
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                self.stats["failed_retrievals"] += 1
                retrieved_docs = []
            
            # Prepare context
            context = self._prepare_context(retrieved_docs)
            context_metadata = self._prepare_metadata(retrieved_docs)
            
            # Get conversation history
            conversation_context = self.memory_manager.get_conversation_context()
            
            # Build prompt
            system_prompt = self.prompt_manager.get_system_prompt()
            
            # Add conversation context if available
            full_context = context
            if conversation_context:
                full_context = f"{conversation_context}\n\n---\n\n{context}"
            
            context_prompt = self.prompt_manager.build_context_prompt(
                query=query,
                context=full_context,
                metadata=context_metadata
            )
            
            # Generate response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context_prompt)
            ]
            
            # Convert messages to simple prompt string for Gemini
            prompt = f"{system_prompt}\n\nUser: {context_prompt}"
            
            llm_response = await self.llm.generate_async(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1
            )
            raw_response = llm_response.content
            
            # Post-process response
            processed_response = self.post_processor.process_response(
                raw_response=raw_response,
                query=query,
                context=context,
                sources=retrieved_docs
            )
            
            # Update conversation memory
            self.memory_manager.add_user_message(query)
            self.memory_manager.add_ai_message(processed_response.content)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["total_processing_time"] += processing_time
            
            # Format final response
            formatted_response = self.post_processor.format_for_display(processed_response)
            
            return {
                "answer": formatted_response,
                "sources": retrieved_docs,
                "metadata": {
                    "processing_time": processing_time,
                    "response_type": processed_response.response_type.value,
                    "safety_level": processed_response.safety_level.value,
                    "citation_count": len(processed_response.citations),
                    **context_metadata
                }
            }
            
        except Exception as e:
            logger.error(f"RAG chain execution failed: {e}")
            return {
                "answer": f"I apologize, but I encountered an error processing your request: {str(e)}",
                "sources": [],
                "metadata": {"error": str(e)}
            }
    
    def reset_conversation(self):
        """Reset conversation memory"""
        self.memory_manager.clear_memory()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chain statistics"""
        stats = self.stats.copy()
        stats.update(self.memory_manager.get_stats())
        return stats


class StreamingRAGChain(RAGChain):
    """RAG Chain with streaming response capability"""
    
    async def stream_response(
        self,
        query: str,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response chunks as they're generated"""
        start_time = datetime.now()
        
        try:
            # Initial yield with status
            yield {
                "type": "status",
                "message": "Processing query...",
                "timestamp": datetime.now().isoformat()
            }
            
            # Safety check
            is_safe, safety_message = self.prompt_manager.evaluate_query_safety(query)
            if not is_safe:
                yield {
                    "type": "error",
                    "message": f"Query safety check failed: {safety_message}",
                    "timestamp": datetime.now().isoformat()
                }
                return
            
            # Retrieval phase
            yield {
                "type": "status",
                "message": "Searching documents...",
                "timestamp": datetime.now().isoformat()
            }
            
            retrieval_results = self.retriever.retrieve(query, config=None)[:self.max_sources]
            
            # Convert RetrievalResult objects to dictionaries
            retrieved_docs = []
            for result in retrieval_results:
                doc_dict = {
                    "content": result.content,
                    "relevance_score": result.relevance_score or result.score,
                    "file_path": result.metadata.get("file_path", "Unknown"),
                    "file_type": result.metadata.get("file_type", "unknown"),
                    "document_id": result.document_id,
                    "chunk_index": result.chunk_index,
                    "metadata": result.metadata
                }
                retrieved_docs.append(doc_dict)
            
            yield {
                "type": "retrieval_complete",
                "source_count": len(retrieved_docs),
                "sources": [doc.get("file_path", "Unknown") for doc in retrieved_docs],
                "timestamp": datetime.now().isoformat()
            }
            
            # Context preparation
            context = self._prepare_context(retrieved_docs)
            context_metadata = self._prepare_metadata(retrieved_docs)
            
            # Generation phase
            yield {
                "type": "status",
                "message": "Generating response...",
                "timestamp": datetime.now().isoformat()
            }
            
            # Build prompt
            system_prompt = self.prompt_manager.get_system_prompt()
            conversation_context = self.memory_manager.get_conversation_context()
            
            full_context = context
            if conversation_context:
                full_context = f"{conversation_context}\n\n---\n\n{context}"
            
            context_prompt = self.prompt_manager.build_context_prompt(
                query=query,
                context=full_context,
                metadata=context_metadata
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context_prompt)
            ]
            
            # Stream response generation  
            prompt = f"{system_prompt}\n\nUser: {context_prompt}"
            response_chunks = []
            async for chunk in self.llm.generate_stream_async(prompt):
                response_chunks.append(chunk)
                yield {
                    "type": "response_chunk",
                    "content": chunk,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Final processing
            raw_response = "".join(response_chunks)
            processed_response = self.post_processor.process_response(
                raw_response=raw_response,
                query=query,
                context=context,
                sources=retrieved_docs
            )
            
            # Update memory
            self.memory_manager.add_user_message(query)
            self.memory_manager.add_ai_message(processed_response.content)
            
            # Final result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            yield {
                "type": "complete",
                "final_response": self.post_processor.format_for_display(processed_response),
                "metadata": {
                    "processing_time": processing_time,
                    "response_type": processed_response.response_type.value,
                    "safety_level": processed_response.safety_level.value,
                    "citation_count": len(processed_response.citations),
                    **context_metadata
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }


def create_rag_chain(
    llm: GeminiLLM,
    retriever: SemanticRetriever,
    prompt_manager: Optional[PromptManager] = None,
    post_processor: Optional[ResponsePostProcessor] = None,
    enable_streaming: bool = False,
    memory_type: str = "buffer_window"
) -> RAGChain:
    """
    Factory function to create RAG chain
    
    Args:
        llm: Gemini LLM instance
        retriever: Document retriever instance
        prompt_manager: Prompt manager (creates default if None)
        post_processor: Response post-processor (creates default if None)
        enable_streaming: Enable streaming responses
        memory_type: Type of conversation memory
        
    Returns:
        RAGChain: Configured RAG chain
    """
    if not prompt_manager:
        from .prompts import create_prompt_manager
        prompt_manager = create_prompt_manager()
    
    if not post_processor:
        from .post_processor import create_response_post_processor
        post_processor = create_response_post_processor()
    
    memory_manager = ConversationMemoryManager(memory_type=memory_type)
    
    if enable_streaming:
        return StreamingRAGChain(
            llm=llm,
            retriever=retriever,
            prompt_manager=prompt_manager,
            post_processor=post_processor,
            memory_manager=memory_manager
        )
    else:
        return RAGChain(
            llm=llm,
            retriever=retriever,
            prompt_manager=prompt_manager,
            post_processor=post_processor,
            memory_manager=memory_manager
        ) 