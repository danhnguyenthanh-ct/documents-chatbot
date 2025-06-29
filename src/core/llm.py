"""
Gemini LLM Configuration and Management
Handles LLM interactions using Google's Gemini models with streaming support,
token usage tracking, rate limiting, and comprehensive error handling.
"""

import logging
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator, Union
import time
import asyncio
from dataclasses import dataclass, field

import google.generativeai as genai
from google.generativeai.types import (
    GenerateContentResponse,
    GenerationConfig,
    HarmCategory,
    HarmBlockThreshold
)
from google.api_core import retry
from google.api_core.exceptions import (
    GoogleAPIError,
    RetryError,
    ResourceExhausted,
    InvalidArgument,
    PermissionDenied
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Custom exception for LLM operations"""
    pass


@dataclass
class TokenUsage:
    """Track token usage for requests"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add_usage(self, other: 'TokenUsage') -> None:
        """Add another usage to this one"""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


@dataclass
class LLMResponse:
    """Structured response from LLM"""
    content: str
    token_usage: TokenUsage
    model: str
    finish_reason: Optional[str] = None
    safety_ratings: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimiter:
    """Simple rate limiter for API requests"""
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 100000
    
    def __init__(self, max_requests_per_minute: int = 60, max_tokens_per_minute: int = 100000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.requests_this_minute = 0
        self.tokens_this_minute = 0
        self.minute_start = time.time()
    
    def can_make_request(self, estimated_tokens: int = 1000) -> bool:
        """Check if request can be made without exceeding rate limits"""
        current_time = time.time()
        
        # Reset counters if a minute has passed
        if current_time - self.minute_start >= 60:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.minute_start = current_time
        
        # Check limits
        if (self.requests_this_minute >= self.max_requests_per_minute or
            self.tokens_this_minute + estimated_tokens > self.max_tokens_per_minute):
            return False
        
        return True
    
    def record_request(self, token_count: int) -> None:
        """Record a completed request"""
        self.requests_this_minute += 1
        self.tokens_this_minute += token_count


class GeminiLLM:
    """
    Production-ready Gemini LLM client with comprehensive features
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.9,
        top_k: int = 40,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000
    ):
        """
        Initialize Gemini LLM client
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model name
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries
            requests_per_minute: Rate limit for requests
            tokens_per_minute: Rate limit for tokens
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Configure Gemini client
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(requests_per_minute, tokens_per_minute)
        
        # Initialize statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": TokenUsage(),
            "average_response_time": 0.0,
            "rate_limited_requests": 0
        }
        
        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        logger.info(f"Gemini LLM initialized with model: {model_name}")
    
    def _create_generation_config(self, **overrides) -> GenerationConfig:
        """Create generation configuration with optional overrides"""
        config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k
        }
        
        # Apply overrides
        config.update(overrides)
        
        return GenerationConfig(**config)
    
    def _extract_token_usage(self, response: GenerateContentResponse) -> TokenUsage:
        """Extract token usage from Gemini response"""
        usage = TokenUsage()
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            metadata = response.usage_metadata
            usage.prompt_tokens = getattr(metadata, 'prompt_token_count', 0)
            usage.completion_tokens = getattr(metadata, 'candidates_token_count', 0)
            usage.total_tokens = getattr(metadata, 'total_token_count', 0)
        
        return usage
    
    def _wait_for_rate_limit(self, estimated_tokens: int = 1000) -> None:
        """Wait if rate limit would be exceeded"""
        while not self.rate_limiter.can_make_request(estimated_tokens):
            logger.warning("Rate limit reached, waiting...")
            self.stats["rate_limited_requests"] += 1
            time.sleep(1)
    
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
    def _generate_with_retry(
        self, 
        prompt: str,
        generation_config: GenerationConfig,
        stream: bool = False
    ) -> Union[GenerateContentResponse, Iterator[GenerateContentResponse]]:
        """Generate response with automatic retry logic"""
        try:
            if stream:
                return self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings,
                    stream=True
                )
            else:
                return self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings
                )
                
        except InvalidArgument as e:
            logger.error(f"Invalid argument for generation: {e}")
            raise LLMError(f"Invalid input: {e}")
        except PermissionDenied as e:
            logger.error(f"Permission denied for Gemini API: {e}")
            raise LLMError(f"Permission denied: {e}")
        except ResourceExhausted as e:
            logger.warning(f"Rate limit exceeded, will retry: {e}")
            raise
        except GoogleAPIError as e:
            logger.error(f"Google API error during generation: {e}")
            raise LLMError(f"API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            raise LLMError(f"Unexpected error: {e}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a single response from the LLM
        
        Args:
            prompt: Input prompt for generation
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional generation config overrides
            
        Returns:
            LLMResponse: Structured response with content and metadata
            
        Raises:
            LLMError: If generation fails
        """
        if not prompt or not prompt.strip():
            raise LLMError("Prompt cannot be empty")
        
        # Estimate tokens for rate limiting
        estimated_tokens = len(prompt.split()) * 1.3  # Rough estimation
        
        print(f"prompt: {prompt}")
        
        # Check rate limits
        self._wait_for_rate_limit(int(estimated_tokens))
        
        start_time = time.time()
        
        try:
            # Create generation config
            config_overrides = {}
            if temperature is not None:
                config_overrides["temperature"] = temperature
            if max_tokens is not None:
                config_overrides["max_output_tokens"] = max_tokens
            config_overrides.update(kwargs)
            
            generation_config = self._create_generation_config(**config_overrides)
            
            # Generate response
            response = self._generate_with_retry(prompt, generation_config)
            
            # Extract content and metadata
            content = response.text if response.text else ""
            token_usage = self._extract_token_usage(response)
            
            # Extract safety ratings
            safety_ratings = []
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'safety_ratings'):
                    safety_ratings = [
                        {
                            "category": rating.category.name,
                            "probability": rating.probability.name
                        }
                        for rating in candidate.safety_ratings
                    ]
            
            # Record usage and stats
            self.rate_limiter.record_request(token_usage.total_tokens)
            self.stats["total_requests"] += 1
            self.stats["successful_requests"] += 1
            self.stats["total_tokens_used"].add_usage(token_usage)
            
            # Update average response time
            response_time = time.time() - start_time
            current_avg = self.stats["average_response_time"]
            total_requests = self.stats["successful_requests"]
            self.stats["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
            
            logger.info(f"Generated response in {response_time:.2f}s, {token_usage.total_tokens} tokens")
            
            return LLMResponse(
                content=content,
                token_usage=token_usage,
                model=self.model_name,
                finish_reason=getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None,
                safety_ratings=safety_ratings,
                metadata={
                    "response_time": response_time,
                    "prompt_length": len(prompt)
                }
            )
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate streaming response from the LLM
        
        Args:
            prompt: Input prompt for generation
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional generation config overrides
            
        Yields:
            str: Chunks of generated content
            
        Raises:
            LLMError: If generation fails
        """
        if not prompt or not prompt.strip():
            raise LLMError("Prompt cannot be empty")
        
        # Estimate tokens for rate limiting
        estimated_tokens = len(prompt.split()) * 1.3
        
        # Check rate limits
        self._wait_for_rate_limit(int(estimated_tokens))
        
        try:
            # Create generation config
            config_overrides = {}
            if temperature is not None:
                config_overrides["temperature"] = temperature
            if max_tokens is not None:
                config_overrides["max_output_tokens"] = max_tokens
            config_overrides.update(kwargs)
            
            generation_config = self._create_generation_config(**config_overrides)
            
            # Generate streaming response
            response_stream = self._generate_with_retry(
                prompt, generation_config, stream=True
            )
            
            total_tokens = 0
            
            for chunk in response_stream:
                if chunk.text:
                    total_tokens += len(chunk.text.split())
                    yield chunk.text
            
            # Record usage
            self.rate_limiter.record_request(total_tokens)
            self.stats["total_requests"] += 1
            self.stats["successful_requests"] += 1
            
            logger.info(f"Streamed response completed, ~{total_tokens} tokens")
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Failed to generate streaming response: {e}")
            raise LLMError(f"Streaming generation failed: {e}")
    
    async def generate_async(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response asynchronously
        
        Args:
            prompt: Input prompt for generation
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional generation config overrides
            
        Returns:
            LLMResponse: Structured response with content and metadata
        """
        # Run synchronous generation in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.generate, 
            prompt, 
            temperature, 
            max_tokens,
            **kwargs
        )
    
    async def generate_stream_async(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate streaming response asynchronously
        
        Args:
            prompt: Input prompt for generation
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional generation config overrides
            
        Yields:
            str: Chunks of generated content
        """
        # Create generator in thread pool
        loop = asyncio.get_event_loop()
        
        def create_generator():
            return self.generate_stream(prompt, temperature, max_tokens, **kwargs)
        
        generator = await loop.run_in_executor(None, create_generator)
        
        # Yield chunks asynchronously
        for chunk in generator:
            yield chunk
            await asyncio.sleep(0)  # Allow other coroutines to run
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        return {
            **self.stats,
            "model_name": self.model_name,
            "current_config": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "top_k": self.top_k
            },
            "rate_limits": {
                "requests_per_minute": self.rate_limiter.max_requests_per_minute,
                "tokens_per_minute": self.rate_limiter.max_tokens_per_minute,
                "current_requests": self.rate_limiter.requests_this_minute,
                "current_tokens": self.rate_limiter.tokens_this_minute
            }
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics"""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_used": TokenUsage(),
            "average_response_time": 0.0,
            "rate_limited_requests": 0
        }
        logger.info("LLM statistics reset")
    
    async def health_check(self) -> bool:
        """
        Check if the LLM service is healthy
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Test with a simple generation
            response = await asyncio.create_task(
                asyncio.to_thread(self.generate, "Hello", max_tokens=5)
            )
            return len(response.content) > 0
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False


# Utility functions
def create_llm_service(
    api_key: str,
    model_name: str = "gemini-1.5-flash",
    **kwargs
) -> GeminiLLM:
    """
    Factory function to create LLM service
    
    Args:
        api_key: Google API key
        model_name: Gemini model name
        **kwargs: Additional configuration parameters
        
    Returns:
        GeminiLLM: Configured LLM service
    """
    return GeminiLLM(api_key=api_key, model_name=model_name, **kwargs)


def get_default_llm_service() -> Optional[GeminiLLM]:
    """
    Get default LLM service from environment variables
    
    Returns:
        GeminiLLM: Default LLM service or None if not configured
    """
    import os
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not found in environment variables")
        return None
    
    model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
    
    return GeminiLLM(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    ) 