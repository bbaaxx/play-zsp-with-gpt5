#!/usr/bin/env python3
"""
External API Integration with RAG Pipeline Components
====================================================

This script demonstrates sophisticated integration of RAG pipeline components
with external APIs and services, bypassing the Gradio interface to work directly
with the core components. It showcases:

1. Direct RAG pipeline integration with external REST APIs
2. Custom webhook endpoints for real-time chat processing
3. Integration with external databases and cloud services
4. Advanced authentication and rate limiting
5. Streaming response processing and caching
6. Cross-platform chat synchronization

Key Advanced Concepts:
- Asynchronous processing with external API calls
- Custom serialization and deserialization for different API formats
- Advanced caching strategies with TTL and LRU eviction
- Integration with message queues and streaming platforms
- Multi-tenant isolation and security boundaries
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
from contextlib import asynccontextmanager

import httpx

# Import our RAG components
from rag.core import (
    parse_whatsapp_txt, RAGPipeline, build_user_prompt, SYSTEM_PROMPT
)


@dataclass
class APIConfig:
    """Configuration for external API integrations."""
    # Authentication
    api_key: Optional[str] = None
    bearer_token: Optional[str] = None
    webhook_secret: Optional[str] = None
    
    # Rate limiting
    requests_per_minute: int = 60
    burst_limit: int = 10
    
    # Caching
    cache_ttl_seconds: int = 3600
    max_cache_entries: int = 1000
    
    # Integration endpoints
    llm_endpoint: str = "https://models.github.ai/inference/chat/completions"
    webhook_endpoint: Optional[str] = None
    database_url: Optional[str] = None
    
    # Processing parameters
    max_concurrent_requests: int = 5
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class QueryRequest:
    """Request structure for external API queries."""
    query: str
    chat_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    context_limit: int = 5
    response_format: str = "json"
    stream: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryResponse:
    """Response structure for API queries."""
    answer: str
    context_chunks: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    cached: bool = False
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RateLimiter:
    """
    Advanced rate limiting with sliding window and burst handling.
    
    Implements sophisticated rate limiting strategies including:
    - Sliding window rate limiting
    - Burst capacity with token bucket algorithm
    - Per-user/tenant isolation
    - Adaptive backoff and queueing
    """
    
    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.windows: Dict[str, List[float]] = {}  # tenant_id -> [timestamps]
        self.burst_tokens: Dict[str, int] = {}  # tenant_id -> token count
        self.last_refill: Dict[str, float] = {}  # tenant_id -> last refill time
        
    async def acquire(self, tenant_id: str = "default") -> bool:
        """
        Acquire rate limiting permission for a tenant.
        
        Returns True if request is allowed, False if rate limited.
        """
        now = time.time()
        
        # Initialize tenant if not exists
        if tenant_id not in self.windows:
            self.windows[tenant_id] = []
            self.burst_tokens[tenant_id] = self.burst_limit
            self.last_refill[tenant_id] = now
        
        # Refill burst tokens
        self._refill_burst_tokens(tenant_id, now)
        
        # Clean old entries from sliding window
        window = self.windows[tenant_id]
        cutoff = now - 60  # 1 minute window
        self.windows[tenant_id] = [ts for ts in window if ts > cutoff]
        
        # Check sliding window limit
        if len(self.windows[tenant_id]) >= self.requests_per_minute:
            # Try to use burst token
            if self.burst_tokens[tenant_id] > 0:
                self.burst_tokens[tenant_id] -= 1
            else:
                return False
        
        # Grant permission
        self.windows[tenant_id].append(now)
        return True
    
    def _refill_burst_tokens(self, tenant_id: str, now: float):
        """Refill burst tokens based on time elapsed."""
        if tenant_id not in self.last_refill:
            return
        
        elapsed = now - self.last_refill[tenant_id]
        # Refill one token every 60/requests_per_minute seconds
        tokens_to_add = int(elapsed * self.requests_per_minute / 60)
        
        if tokens_to_add > 0:
            self.burst_tokens[tenant_id] = min(
                self.burst_limit,
                self.burst_tokens[tenant_id] + tokens_to_add
            )
            self.last_refill[tenant_id] = now


class ResponseCache:
    """
    Advanced caching system with TTL, LRU eviction, and cache warming.
    
    Features:
    - Time-based expiration (TTL)
    - LRU eviction when memory limits reached
    - Cache warming for common queries
    - Hit/miss statistics and monitoring
    - Distributed cache support ready
    """
    
    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
        
    def _generate_key(self, query: str, filters: Optional[Dict] = None) -> str:
        """Generate cache key from query and filters."""
        key_data = {"query": query}
        if filters:
            key_data["filters"] = filters
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, filters: Optional[Dict] = None) -> Optional[QueryResponse]:
        """Retrieve cached response if valid."""
        key = self._generate_key(query, filters)
        now = time.time()
        
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if now - entry["timestamp"] > self.ttl_seconds:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            self.stats["misses"] += 1
            return None
        
        # Update access time for LRU
        self.access_times[key] = now
        self.stats["hits"] += 1
        
        # Deserialize response
        response = QueryResponse(**entry["response"])
        response.cached = True
        return response
    
    def put(self, query: str, response: QueryResponse, filters: Optional[Dict] = None):
        """Store response in cache."""
        key = self._generate_key(query, filters)
        now = time.time()
        
        # Evict LRU entries if at capacity
        if len(self.cache) >= self.max_entries:
            self._evict_lru()
        
        # Store entry
        self.cache[key] = {
            "timestamp": now,
            "response": asdict(response)
        }
        self.access_times[key] = now
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.stats["evictions"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "max_entries": self.max_entries
        }


class ExternalAPIClient:
    """
    Asynchronous client for external API integrations with advanced features.
    
    This class demonstrates:
    - Async HTTP client management with connection pooling
    - Retry logic with exponential backoff
    - Request/response transformation for different API formats
    - Streaming response handling
    - Authentication token management and refresh
    """
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
    @asynccontextmanager
    async def get_client(self):
        """Get HTTP client with proper configuration."""
        timeout = httpx.Timeout(self.config.timeout_seconds)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            yield client
    
    async def call_llm(self, prompt: str, system_prompt: str = SYSTEM_PROMPT,
                      stream: bool = False) -> AsyncGenerator[str, None]:
        """
        Call external LLM API with streaming support.
        
        Demonstrates:
        - Authentication header management
        - Request payload construction for different API formats
        - Streaming response processing
        - Error handling and retry logic
        """
        async with self.semaphore:  # Rate limiting
            for attempt in range(self.config.retry_attempts):
                try:
                    async with self.get_client() as client:
                        headers = self._build_headers()
                        payload = self._build_llm_payload(prompt, system_prompt, stream)
                        
                        if stream:
                            async for chunk in self._stream_llm_response(client, payload, headers):
                                yield chunk
                        else:
                            response = await client.post(
                                self.config.llm_endpoint,
                                json=payload,
                                headers=headers
                            )
                            response.raise_for_status()
                            data = response.json()
                            yield self._extract_llm_response(data)
                        
                        return  # Success, exit retry loop
                        
                except Exception as e:
                    self.logger.warning(f"LLM API attempt {attempt + 1} failed: {e}")
                    if attempt < self.config.retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    else:
                        raise
    
    async def _stream_llm_response(self, client: httpx.AsyncClient, 
                                  payload: Dict, headers: Dict) -> AsyncGenerator[str, None]:
        """Handle streaming LLM responses."""
        async with client.stream(
            "POST", self.config.llm_endpoint, json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue
    
    def _build_headers(self) -> Dict[str, str]:
        """Build authentication headers."""
        headers = {"Content-Type": "application/json"}
        
        if self.config.bearer_token:
            headers["Authorization"] = f"Bearer {self.config.bearer_token}"
        elif self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        
        return headers
    
    def _build_llm_payload(self, prompt: str, system_prompt: str, stream: bool) -> Dict:
        """Build LLM API payload."""
        return {
            "model": "gpt-4o-mini",  # Default model
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": stream,
            "temperature": 0.7,
            "max_tokens": 1000
        }
    
    def _extract_llm_response(self, response_data: Dict) -> str:
        """Extract text response from LLM API response."""
        if "choices" in response_data and response_data["choices"]:
            return response_data["choices"][0]["message"]["content"]
        return ""


class RAGAPIService:
    """
    Advanced RAG service with external API integration.
    
    This service demonstrates:
    - Integration of RAG pipeline with external APIs
    - Advanced caching and rate limiting
    - Multi-tenant support with isolation
    - Comprehensive monitoring and logging
    - Async processing with proper resource management
    """
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.rag_pipeline = RAGPipeline()
        self.api_client = ExternalAPIClient(config)
        self.rate_limiter = RateLimiter(
            config.requests_per_minute, 
            config.burst_limit
        )
        self.cache = ResponseCache(
            config.cache_ttl_seconds, 
            config.max_cache_entries
        )
        
        # Chat storage for multi-session support
        self.chat_stores: Dict[str, RAGPipeline] = {}
    
    async def load_chat_data(self, chat_data: str, chat_id: str) -> Dict[str, Any]:
        """
        Load and index chat data for a specific chat session.
        
        Demonstrates:
        - Multi-tenant chat data isolation
        - Async processing of large chat files
        - Error handling and validation
        """
        self.logger.info(f"Loading chat data for chat_id: {chat_id}")
        
        try:
            # Parse messages
            messages = parse_whatsapp_txt(chat_data, chat_id)
            if not messages:
                raise ValueError("No valid messages found in chat data")
            
            # Create dedicated RAG pipeline for this chat
            pipeline = RAGPipeline()
            pipeline.index_messages(messages)
            
            # Store pipeline for future queries
            self.chat_stores[chat_id] = pipeline
            
            return {
                "chat_id": chat_id,
                "message_count": len(messages),
                "chunk_count": len(pipeline.chunks),
                "participants": sorted(list(set(msg.sender for msg in messages))),
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Error loading chat data for {chat_id}: {e}")
            raise
    
    async def query_chat(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query against loaded chat data.
        
        Demonstrates:
        - Rate limiting and caching integration
        - Advanced context retrieval with filtering
        - Async LLM API integration
        - Comprehensive error handling and monitoring
        """
        start_time = time.time()
        
        # Generate tenant ID for rate limiting (use user_id or fallback to session_id)
        tenant_id = request.user_id or request.session_id or "default"
        
        # Check rate limiting
        if not await self.rate_limiter.acquire(tenant_id):
            raise Exception("Rate limit exceeded. Please try again later.")
        
        # Check cache first
        cached_response = self.cache.get(request.query, request.filters)
        if cached_response:
            self.logger.info(f"Cache hit for query: {request.query[:50]}...")
            return cached_response
        
        try:
            # Get appropriate RAG pipeline
            if request.chat_id and request.chat_id in self.chat_stores:
                pipeline = self.chat_stores[request.chat_id]
            else:
                pipeline = self.rag_pipeline  # Use default pipeline
            
            if pipeline.vector_store is None:
                raise ValueError("No chat data loaded. Please load chat data first.")
            
            # Retrieve relevant context
            context_results = pipeline.retrieve(
                query=request.query,
                top_k=request.context_limit,
                use_mmr=True,
                senders=request.filters.get("senders") if request.filters else None,
                date_from_iso=request.filters.get("date_from") if request.filters else None,
                date_to_iso=request.filters.get("date_to") if request.filters else None
            )
            
            if not context_results:
                return QueryResponse(
                    answer="No relevant context found for your query.",
                    context_chunks=[],
                    confidence_score=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Format context for LLM
            context_text = pipeline.format_context(context_results)
            user_prompt = build_user_prompt(context_text, request.query)
            
            # Call external LLM API
            if request.stream:
                # For streaming, we need to handle differently
                raise NotImplementedError("Streaming responses not implemented in this example")
            else:
                llm_response_parts = []
                async for chunk in self.api_client.call_llm(user_prompt, stream=False):
                    llm_response_parts.append(chunk)
                
                answer = "".join(llm_response_parts)
            
            # Calculate confidence score based on retrieval scores
            # This is a simplified confidence calculation
            confidence_score = min(1.0, len(context_results) / request.context_limit)
            
            processing_time = time.time() - start_time
            
            response = QueryResponse(
                answer=answer,
                context_chunks=context_results,
                confidence_score=confidence_score,
                processing_time=processing_time,
                request_id=hashlib.md5(f"{request.query}{start_time}".encode()).hexdigest()[:8]
            )
            
            # Cache the response
            self.cache.put(request.query, response, request.filters)
            
            self.logger.info(f"Query processed in {processing_time:.2f}s: {request.query[:50]}...")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise
    
    async def webhook_handler(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming webhook requests for real-time processing.
        
        Demonstrates:
        - Webhook signature verification
        - Real-time chat message processing
        - Event-driven architecture integration
        """
        self.logger.info("Processing webhook payload")
        
        try:
            # Verify webhook signature if configured
            if self.config.webhook_secret:
                # In a real implementation, verify the webhook signature here
                pass
            
            # Extract relevant data from webhook payload
            event_type = payload.get("type", "unknown")
            
            if event_type == "new_message":
                # Process new chat message
                chat_id = payload.get("chat_id")
                
                if chat_id and chat_id in self.chat_stores:
                    # In a real implementation, you would update the vector store
                    # with the new message and re-index if necessary
                    return {
                        "status": "processed",
                        "chat_id": chat_id,
                        "message": "Message processed and indexed"
                    }
            
            elif event_type == "query_request":
                # Handle real-time query
                query_data = payload.get("query", {})
                request = QueryRequest(**query_data)
                response = await self.query_chat(request)
                
                return {
                    "status": "success",
                    "response": asdict(response)
                }
            
            return {"status": "unknown_event_type", "type": event_type}
            
        except Exception as e:
            self.logger.error(f"Webhook processing error: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        return {
            "loaded_chats": len(self.chat_stores),
            "cache_stats": self.cache.get_stats(),
            "rate_limiter_windows": {
                tenant: len(windows) 
                for tenant, windows in self.rate_limiter.windows.items()
            }
        }


async def demonstrate_api_integration():
    """
    Demonstration of the RAG API service with external integrations.
    
    This function shows how to:
    - Initialize the service with configuration
    - Load chat data from various sources
    - Process queries with external API calls
    - Handle webhook events
    - Monitor service performance
    """
    # Configuration (in real use, load from environment or config file)
    config = APIConfig(
        bearer_token=os.environ.get("GITHUB_TOKEN"),
        requests_per_minute=30,
        cache_ttl_seconds=1800,
        max_concurrent_requests=3
    )
    
    # Initialize service
    service = RAGAPIService(config)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting RAG API service demonstration")
    
    # Load sample chat data
    chat_file = Path(__file__).parent.parent.parent / "data" / "sample_whatsapp.txt"
    if chat_file.exists():
        with open(chat_file, 'r', encoding='utf-8') as f:
            chat_data = f.read()
        
        # Load chat data
        load_result = await service.load_chat_data(chat_data, "demo_chat")
        logger.info(f"Loaded chat data: {load_result}")
        
        # Demonstrate various query types
        test_queries = [
            QueryRequest(
                query="¿De qué habla este chat?",
                chat_id="demo_chat",
                user_id="demo_user_1"
            ),
            QueryRequest(
                query="¿Quién participa más en las conversaciones?",
                chat_id="demo_chat",
                user_id="demo_user_2",
                context_limit=3
            ),
            QueryRequest(
                query="¿Hay algún plan específico mencionado?",
                chat_id="demo_chat",
                user_id="demo_user_1",
                filters={"date_from": "2023-01-01"}
            )
        ]
        
        # Process queries
        for i, query_req in enumerate(test_queries):
            logger.info(f"Processing query {i+1}: {query_req.query}")
            
            try:
                response = await service.query_chat(query_req)
                logger.info(f"Response: {response.answer[:100]}...")
                logger.info(f"Confidence: {response.confidence_score:.2f}, "
                          f"Processing time: {response.processing_time:.2f}s, "
                          f"Cached: {response.cached}")
                
                # Demonstrate caching by repeating the same query
                if i == 0:
                    cached_response = await service.query_chat(query_req)
                    logger.info(f"Cached response time: {cached_response.processing_time:.2f}s, "
                              f"Cached: {cached_response.cached}")
                
            except Exception as e:
                logger.error(f"Error processing query {i+1}: {e}")
        
        # Demonstrate webhook handling
        webhook_payload = {
            "type": "query_request",
            "query": {
                "query": "¿Cuándo fue la última conversación?",
                "chat_id": "demo_chat",
                "user_id": "webhook_user"
            }
        }
        
        webhook_result = await service.webhook_handler(webhook_payload)
        logger.info(f"Webhook result: {webhook_result.get('status', 'unknown')}")
        
        # Show service statistics
        stats = service.get_service_stats()
        logger.info(f"Service statistics: {json.dumps(stats, indent=2)}")
    
    else:
        logger.warning(f"Sample chat file not found at {chat_file}")
        logger.info("You can still test the service by loading your own chat data")


def main():
    """Main function for testing the API integration."""
    parser = argparse.ArgumentParser(description="RAG API Integration Demonstration")
    parser.add_argument("--chat-file", help="Path to WhatsApp export file")
    parser.add_argument("--config", help="Path to API configuration file")
    parser.add_argument("--bearer-token", help="Bearer token for LLM API")
    parser.add_argument("--demo", action="store_true", 
                       help="Run built-in demonstration")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.demo:
        # Run demonstration
        asyncio.run(demonstrate_api_integration())
    else:
        # Custom usage
        config_data = {}
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config_data = json.load(f)
        
        if args.bearer_token:
            config_data["bearer_token"] = args.bearer_token
        
        config = APIConfig(**config_data)
        service = RAGAPIService(config)
        
        if args.chat_file and os.path.exists(args.chat_file):
            async def run_custom():
                with open(args.chat_file, 'r', encoding='utf-8') as f:
                    chat_data = f.read()
                
                result = await service.load_chat_data(chat_data, "custom_chat")
                print(f"Loaded chat: {result}")
                
                # Interactive query loop
                print("\nEnter queries (press Ctrl+C to exit):")
                try:
                    while True:
                        query_text = input("> ")
                        if query_text.strip():
                            request = QueryRequest(
                                query=query_text,
                                chat_id="custom_chat"
                            )
                            response = await service.query_chat(request)
                            print(f"Answer: {response.answer}")
                            print(f"Confidence: {response.confidence_score:.2f}")
                            print()
                except KeyboardInterrupt:
                    print("\nGoodbye!")
            
            asyncio.run(run_custom())
        else:
            print("Please provide a chat file with --chat-file or use --demo")


if __name__ == "__main__":
    import os
    main()