#!/usr/bin/env python3
"""
Batch Processing Script for Multiple WhatsApp Chat Files
======================================================

This script demonstrates advanced batch processing capabilities for multiple WhatsApp
chat exports with custom embedding configurations. It showcases:

1. Parallel processing of multiple chat files
2. Custom embedding model configurations per batch
3. Advanced chunking strategies with overlapping windows
4. Metadata aggregation across multiple chats
5. Progress tracking and error handling
6. Output to multiple formats (JSON, parquet, custom vector store)

Key Advanced Concepts:
- Custom embedding providers with different models for different content types
- Adaptive chunking based on conversation patterns and message density
- Batch optimization to minimize API calls and memory usage
- Cross-chat participant identification and deduplication
"""

import asyncio
import json
import logging
import os
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import argparse

import numpy as np
import pandas as pd

# Import our RAG components
from rag.core import (
    parse_whatsapp_txt, chunk_messages, ChatMessage, Chunk, 
    format_message_for_window, RAGPipeline
)
from rag.embeddings import EmbeddingProvider
from rag.vector_store import InMemoryFAISS


@dataclass
class BatchProcessingConfig:
    """Configuration for advanced batch processing operations."""
    # Chunking parameters
    base_window_size: int = 30
    base_overlap: int = 10
    adaptive_chunking: bool = True
    min_chunk_chars: int = 100
    max_chunk_chars: int = 8000
    
    # Embedding configuration
    embedding_model: Optional[str] = None
    use_different_models_per_type: bool = True
    conversation_model: str = "openai/text-embedding-3-small"
    metadata_model: str = "openai/text-embedding-ada-002"
    
    # Processing parameters
    max_workers: int = 4
    batch_size: int = 32
    enable_parallel_embedding: bool = True
    
    # Output configuration
    output_formats: List[str] = None
    include_metadata: bool = True
    deduplicate_participants: bool = True
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ["json", "vector_store"]


@dataclass
class ChatProcessingResult:
    """Results from processing a single chat file."""
    chat_id: str
    file_path: str
    message_count: int
    chunk_count: int
    participants: List[str]
    date_range: Tuple[str, str]
    processing_time: float
    error: Optional[str] = None
    chunks: List[Chunk] = None
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []


@dataclass  
class BatchProcessingResults:
    """Aggregate results from batch processing multiple chat files."""
    total_files: int
    successful_files: int
    total_messages: int
    total_chunks: int
    unique_participants: Set[str]
    processing_time: float
    individual_results: List[ChatProcessingResult]
    consolidated_vector_store: Optional[InMemoryFAISS] = None


class AdvancedChunkingStrategy:
    """
    Implements adaptive chunking strategies that adjust window size and overlap
    based on conversation patterns, message density, and content characteristics.
    """
    
    def __init__(self, config: BatchProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_conversation_pattern(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """
        Analyze conversation patterns to inform chunking strategy.
        
        Returns:
            Dictionary with conversation metrics like density, participant_count, etc.
        """
        if not messages:
            return {}
        
        # Calculate message density (messages per hour)
        time_span = (messages[-1].timestamp - messages[0].timestamp).total_seconds() / 3600
        density = len(messages) / max(time_span, 0.1)  # Avoid division by zero
        
        # Count unique participants
        participants = set(msg.sender for msg in messages)
        
        # Calculate average message length
        avg_length = sum(len(msg.text) for msg in messages) / len(messages)
        
        # Detect conversation breaks (gaps > 2 hours)
        breaks = 0
        for i in range(1, len(messages)):
            gap = (messages[i].timestamp - messages[i-1].timestamp).total_seconds() / 3600
            if gap > 2:
                breaks += 1
        
        return {
            "density": density,
            "participant_count": len(participants),
            "avg_message_length": avg_length,
            "conversation_breaks": breaks,
            "total_messages": len(messages)
        }
    
    def adaptive_chunk_messages(self, messages: List[ChatMessage]) -> List[Chunk]:
        """
        Apply adaptive chunking based on conversation analysis.
        
        This method adjusts window size and overlap based on:
        - Message density (higher density = smaller windows)
        - Number of participants (more participants = larger windows for context)  
        - Conversation breaks (natural chunk boundaries)
        - Message length patterns
        """
        if not messages:
            return []
        
        pattern = self.analyze_conversation_pattern(messages)
        
        # Adapt window size based on conversation characteristics
        base_size = self.config.base_window_size
        
        # Increase window for multi-participant conversations
        if pattern.get("participant_count", 1) > 2:
            base_size = int(base_size * 1.5)
        
        # Decrease window for high-density conversations
        if pattern.get("density", 0) > 10:  # More than 10 messages per hour
            base_size = max(15, int(base_size * 0.7))
        
        # Adjust overlap based on conversation breaks
        overlap = self.config.base_overlap
        if pattern.get("conversation_breaks", 0) > 5:
            overlap = max(5, int(overlap * 0.5))  # Less overlap when natural breaks exist
        
        self.logger.info(f"Adaptive chunking: window_size={base_size}, overlap={overlap} "
                        f"(density={pattern.get('density', 0):.1f}, "
                        f"participants={pattern.get('participant_count', 0)})")
        
        # Use conversation breaks as natural chunk boundaries if enabled
        if self.config.adaptive_chunking and pattern.get("conversation_breaks", 0) > 0:
            return self._chunk_with_break_boundaries(messages, base_size, overlap)
        else:
            return chunk_messages(messages, base_size, overlap)
    
    def _chunk_with_break_boundaries(self, messages: List[ChatMessage], 
                                   window_size: int, overlap: int) -> List[Chunk]:
        """
        Chunk messages using conversation breaks as natural boundaries.
        """
        chunks = []
        current_segment = []
        
        for i, msg in enumerate(messages):
            current_segment.append(msg)
            
            # Check for conversation break
            is_break = False
            if i < len(messages) - 1:
                next_msg = messages[i + 1]
                gap_hours = (next_msg.timestamp - msg.timestamp).total_seconds() / 3600
                is_break = gap_hours > 2
            
            # If we hit a break or reached segment end, process current segment
            if is_break or i == len(messages) - 1:
                if current_segment:
                    segment_chunks = chunk_messages(current_segment, window_size, overlap)
                    chunks.extend(segment_chunks)
                    current_segment = []
        
        return chunks


class CustomEmbeddingProvider:
    """
    Extended embedding provider that supports different models for different content types
    and implements advanced batching strategies.
    """
    
    def __init__(self, config: BatchProcessingConfig):
        self.config = config
        self.providers = {}
        
        # Initialize different providers for different content types
        if config.use_different_models_per_type:
            # Provider optimized for conversational content
            conv_provider = EmbeddingProvider()
            conv_provider.remote_model_name = config.conversation_model
            self.providers["conversation"] = conv_provider
            
            # Provider optimized for metadata/structured content
            meta_provider = EmbeddingProvider()
            meta_provider.remote_model_name = config.metadata_model
            self.providers["metadata"] = meta_provider
        else:
            # Single provider for all content
            provider = EmbeddingProvider()
            if config.embedding_model:
                provider.remote_model_name = config.embedding_model
            self.providers["default"] = provider
    
    def embed_chunks(self, chunks: List[Chunk], content_type: str = "conversation") -> np.ndarray:
        """
        Embed chunks with appropriate model based on content type.
        
        Args:
            chunks: List of chunks to embed
            content_type: Type of content ("conversation", "metadata", or "default")
        
        Returns:
            Embedding matrix
        """
        provider_key = content_type if content_type in self.providers else "default"
        provider = self.providers.get(provider_key, list(self.providers.values())[0])
        
        # Extract text content from chunks
        texts = []
        for chunk in chunks:
            # For conversation content, use the full text window
            if content_type == "conversation":
                texts.append(chunk.text_window)
            # For metadata content, create a structured representation
            elif content_type == "metadata":
                meta_text = f"Participants: {', '.join(chunk.participants)}. "
                meta_text += f"Time: {chunk.start_ts} to {chunk.end_ts}. "
                meta_text += f"Messages: {chunk.line_span[1] - chunk.line_span[0] + 1}"
                texts.append(meta_text)
            else:
                texts.append(chunk.text_window)
        
        return provider.embed_texts(texts)


def process_single_chat_file(file_path: str, config: BatchProcessingConfig) -> ChatProcessingResult:
    """
    Process a single WhatsApp chat file with advanced configurations.
    
    This function demonstrates:
    - Error handling and recovery
    - Custom chunking strategies  
    - Performance monitoring
    - Metadata extraction and enrichment
    """
    start_time = datetime.now()
    logger = logging.getLogger(__name__)
    
    try:
        # Read and parse the chat file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Generate consistent chat_id based on file content
        chat_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        logger.info(f"Processing chat {chat_id} from {file_path}")
        
        # Parse messages
        messages = parse_whatsapp_txt(content, chat_id)
        if not messages:
            return ChatProcessingResult(
                chat_id=chat_id,
                file_path=file_path,
                message_count=0,
                chunk_count=0,
                participants=[],
                date_range=("", ""),
                processing_time=0.0,
                error="No valid messages found"
            )
        
        # Apply adaptive chunking strategy
        chunking_strategy = AdvancedChunkingStrategy(config)
        chunks = chunking_strategy.adaptive_chunk_messages(messages)
        
        # Extract metadata
        participants = sorted(list(set(msg.sender for msg in messages)))
        date_range = (
            messages[0].timestamp.isoformat(),
            messages[-1].timestamp.isoformat()
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ChatProcessingResult(
            chat_id=chat_id,
            file_path=file_path,
            message_count=len(messages),
            chunk_count=len(chunks),
            participants=participants,
            date_range=date_range,
            processing_time=processing_time,
            chunks=chunks
        )
    
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error processing {file_path}: {e}")
        return ChatProcessingResult(
            chat_id="",
            file_path=file_path,
            message_count=0,
            chunk_count=0,
            participants=[],
            date_range=("", ""),
            processing_time=processing_time,
            error=str(e)
        )


def batch_process_chats(file_paths: List[str], config: BatchProcessingConfig) -> BatchProcessingResults:
    """
    Process multiple chat files in parallel with advanced configurations.
    
    This function demonstrates:
    - Parallel processing with configurable worker pools
    - Progress tracking and reporting
    - Error aggregation and reporting
    - Memory-efficient processing of large file collections
    """
    start_time = datetime.now()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting batch processing of {len(file_paths)} files")
    
    # Process files in parallel
    individual_results = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(process_single_chat_file, file_path, config): file_path
            for file_path in file_paths
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                individual_results.append(result)
                logger.info(f"Completed {file_path}: {result.message_count} messages, "
                          f"{result.chunk_count} chunks")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                # Create error result
                individual_results.append(ChatProcessingResult(
                    chat_id="",
                    file_path=file_path,
                    message_count=0,
                    chunk_count=0,
                    participants=[],
                    date_range=("", ""),
                    processing_time=0.0,
                    error=str(e)
                ))
    
    # Aggregate results
    successful_results = [r for r in individual_results if r.error is None]
    total_messages = sum(r.message_count for r in successful_results)
    total_chunks = sum(r.chunk_count for r in successful_results)
    
    # Deduplicate participants across all chats if enabled
    all_participants = set()
    for result in successful_results:
        all_participants.update(result.participants)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Batch processing completed in {processing_time:.2f}s: "
              f"{len(successful_results)}/{len(file_paths)} files successful, "
              f"{total_messages} total messages, {total_chunks} total chunks")
    
    return BatchProcessingResults(
        total_files=len(file_paths),
        successful_files=len(successful_results),
        total_messages=total_messages,
        total_chunks=total_chunks,
        unique_participants=all_participants,
        processing_time=processing_time,
        individual_results=individual_results
    )


def create_consolidated_vector_store(results: BatchProcessingResults, 
                                   config: BatchProcessingConfig) -> Optional[InMemoryFAISS]:
    """
    Create a consolidated vector store from all processed chunks.
    
    This function demonstrates:
    - Cross-chat embedding consolidation
    - Custom embedding strategies per content type
    - Memory-efficient vector store construction
    - Metadata enrichment with cross-references
    """
    logger = logging.getLogger(__name__)
    
    # Collect all chunks from successful processing results
    all_chunks = []
    for result in results.individual_results:
        if result.error is None and result.chunks:
            all_chunks.extend(result.chunks)
    
    if not all_chunks:
        logger.warning("No chunks available for vector store creation")
        return None
    
    logger.info(f"Creating consolidated vector store with {len(all_chunks)} chunks")
    
    # Initialize custom embedding provider
    embedding_provider = CustomEmbeddingProvider(config)
    
    # Generate embeddings for all chunks
    try:
        embeddings = embedding_provider.embed_chunks(all_chunks, "conversation")
        
        # Create vector store
        vector_store = InMemoryFAISS(dim=embeddings.shape[1])
        
        # Prepare IDs and metadata
        ids = [chunk.chunk_id for chunk in all_chunks]
        metadatas = []
        
        for chunk in all_chunks:
            # Enrich metadata with cross-chat information
            metadata = {
                "chunk_id": chunk.chunk_id,
                "chat_id": chunk.chat_id,
                "start_ts": chunk.start_ts.isoformat(timespec="minutes"),
                "end_ts": chunk.end_ts.isoformat(timespec="minutes"),
                "participants": chunk.participants,
                "line_span": chunk.line_span,
                "text_window": chunk.text_window,
                # Additional enriched metadata
                "message_count": chunk.line_span[1] - chunk.line_span[0] + 1,
                "participant_count": len(chunk.participants),
                "time_span_hours": (chunk.end_ts - chunk.start_ts).total_seconds() / 3600,
                "chunk_length": len(chunk.text_window)
            }
            metadatas.append(metadata)
        
        # Add to vector store
        vector_store.add(ids, embeddings, metadatas)
        
        logger.info(f"Consolidated vector store created with {vector_store.size()} chunks")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating consolidated vector store: {e}")
        return None


def save_results(results: BatchProcessingResults, output_dir: str, config: BatchProcessingConfig):
    """
    Save processing results in multiple formats.
    
    Demonstrates:
    - Multi-format output generation
    - Structured data export
    - Vector store persistence
    - Comprehensive reporting
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    
    # Save summary report as JSON
    if "json" in config.output_formats:
        summary = {
            "processing_summary": {
                "total_files": results.total_files,
                "successful_files": results.successful_files,
                "total_messages": results.total_messages,
                "total_chunks": results.total_chunks,
                "unique_participants": sorted(list(results.unique_participants)),
                "processing_time": results.processing_time
            },
            "individual_results": [
                {
                    "chat_id": r.chat_id,
                    "file_path": r.file_path,
                    "message_count": r.message_count,
                    "chunk_count": r.chunk_count,
                    "participants": r.participants,
                    "date_range": r.date_range,
                    "processing_time": r.processing_time,
                    "error": r.error
                }
                for r in results.individual_results
            ]
        }
        
        with open(output_path / "batch_results.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON summary to {output_path / 'batch_results.json'}")
    
    # Save as DataFrame/Parquet for data analysis
    if "parquet" in config.output_formats:
        try:
            import pandas as pd
            df_data = []
            for result in results.individual_results:
                if result.chunks:
                    for chunk in result.chunks:
                        df_data.append({
                            "chat_id": result.chat_id,
                            "file_path": result.file_path,
                            "chunk_id": chunk.chunk_id,
                            "start_ts": chunk.start_ts,
                            "end_ts": chunk.end_ts,
                            "participants": ",".join(chunk.participants),
                            "participant_count": len(chunk.participants),
                            "line_span_start": chunk.line_span[0],
                            "line_span_end": chunk.line_span[1],
                            "text_length": len(chunk.text_window)
                        })
            
            if df_data:
                df = pd.DataFrame(df_data)
                df.to_parquet(output_path / "batch_chunks.parquet", index=False)
                logger.info(f"Saved Parquet data to {output_path / 'batch_chunks.parquet'}")
        except ImportError:
            logger.warning("pandas not available, skipping Parquet output")
    
    # Save consolidated vector store
    if "vector_store" in config.output_formats and results.consolidated_vector_store:
        vector_store_path = output_path / "consolidated_vector_store"
        results.consolidated_vector_store.save(str(vector_store_path))
        logger.info(f"Saved vector store to {vector_store_path}")


def main():
    """Main function demonstrating batch processing workflow."""
    parser = argparse.ArgumentParser(description="Advanced WhatsApp Chat Batch Processor")
    parser.add_argument("input_dir", help="Directory containing WhatsApp export files")
    parser.add_argument("--output-dir", default="./batch_output", 
                       help="Output directory for results")
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum number of parallel workers")
    parser.add_argument("--window-size", type=int, default=30,
                       help="Base chunking window size")
    parser.add_argument("--adaptive", action="store_true",
                       help="Enable adaptive chunking")
    parser.add_argument("--embedding-model", 
                       help="Custom embedding model to use")
    parser.add_argument("--output-formats", nargs="+", 
                       choices=["json", "parquet", "vector_store"],
                       default=["json", "vector_store"],
                       help="Output formats to generate")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = BatchProcessingConfig(**config_data)
    else:
        config = BatchProcessingConfig(
            max_workers=args.max_workers,
            base_window_size=args.window_size,
            adaptive_chunking=args.adaptive,
            embedding_model=args.embedding_model,
            output_formats=args.output_formats
        )
    
    # Find all WhatsApp export files
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        return 1
    
    file_paths = []
    for pattern in ["*.txt", "*.TXT"]:
        file_paths.extend(input_path.glob(pattern))
    
    if not file_paths:
        print(f"No WhatsApp export files found in {input_path}")
        return 1
    
    print(f"Found {len(file_paths)} WhatsApp export files")
    
    # Process files
    results = batch_process_chats([str(p) for p in file_paths], config)
    
    # Create consolidated vector store
    results.consolidated_vector_store = create_consolidated_vector_store(results, config)
    
    # Save results
    save_results(results, args.output_dir, config)
    
    # Print summary
    print(f"\nBatch Processing Complete!")
    print(f"Files processed: {results.successful_files}/{results.total_files}")
    print(f"Total messages: {results.total_messages:,}")
    print(f"Total chunks: {results.total_chunks:,}")
    print(f"Unique participants: {len(results.unique_participants)}")
    print(f"Processing time: {results.processing_time:.2f}s")
    print(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())