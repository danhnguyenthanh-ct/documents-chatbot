"""
Text Chunking Module
Implements semantic and fixed-size chunking strategies optimized for Gemini context limits.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


class ChunkingError(Exception):
    """Custom exception for chunking operations"""
    pass


@dataclass
class TextChunk:
    """Container for text chunk with metadata"""
    content: str
    start_index: int
    end_index: int
    chunk_id: str
    metadata: Dict[str, Any]
    
    def __len__(self) -> int:
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        return len(self.content.split())


class TextChunker:
    """
    Advanced text chunking with multiple strategies
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 100,
        respect_sentence_boundaries: bool = True,
        respect_paragraph_boundaries: bool = True,
        preserve_headers: bool = True
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            max_chunk_size: Maximum allowed chunk size
            min_chunk_size: Minimum chunk size (smaller chunks will be merged)
            respect_sentence_boundaries: Avoid breaking sentences
            respect_paragraph_boundaries: Prefer paragraph boundaries
            preserve_headers: Keep headers with following content
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.respect_sentence_boundaries = respect_sentence_boundaries
        self.respect_paragraph_boundaries = respect_paragraph_boundaries
        self.preserve_headers = preserve_headers
        
        # Compile regex patterns
        self._compile_patterns()
        
        # Initialize statistics
        self.stats = {
            "documents_chunked": 0,
            "total_chunks_created": 0,
            "average_chunk_size": 0.0,
            "total_characters_processed": 0
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for text analysis"""
        # Sentence boundary detection (basic)
        self.sentence_pattern = re.compile(r'[.!?]+\s+')
        
        # Paragraph detection
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        
        # Header detection (markdown-style)
        self.header_pattern = re.compile(r'^#{1,6}\s+.*$', re.MULTILINE)
        
        # List item detection
        self.list_pattern = re.compile(r'^\s*[-*+]\s+.*$|^\s*\d+\.\s+.*$', re.MULTILINE)
        
        # Code block detection
        self.code_block_pattern = re.compile(r'```.*?```', re.DOTALL)
    
    def _find_sentence_boundaries(self, text: str) -> List[int]:
        """Find sentence boundaries in text"""
        boundaries = [0]  # Start of text
        
        for match in self.sentence_pattern.finditer(text):
            boundaries.append(match.end())
        
        if boundaries[-1] != len(text):
            boundaries.append(len(text))  # End of text
        
        return boundaries
    
    def _find_paragraph_boundaries(self, text: str) -> List[int]:
        """Find paragraph boundaries in text"""
        boundaries = [0]  # Start of text
        
        for match in self.paragraph_pattern.finditer(text):
            boundaries.append(match.start())
            boundaries.append(match.end())
        
        if boundaries[-1] != len(text):
            boundaries.append(len(text))  # End of text
        
        return sorted(set(boundaries))
    
    def _identify_structure_elements(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        """Identify structural elements in text"""
        elements = {
            "headers": [],
            "lists": [],
            "code_blocks": []
        }
        
        # Find headers
        for match in self.header_pattern.finditer(text):
            elements["headers"].append((match.start(), match.end()))
        
        # Find lists
        for match in self.list_pattern.finditer(text):
            elements["lists"].append((match.start(), match.end()))
        
        # Find code blocks
        for match in self.code_block_pattern.finditer(text):
            elements["code_blocks"].append((match.start(), match.end()))
        
        return elements
    
    def _find_best_split_point(
        self, 
        text: str, 
        target_position: int, 
        boundaries: List[int]
    ) -> int:
        """Find the best position to split text near target position"""
        if not boundaries:
            return target_position
        
        # Find the boundary closest to target position
        best_boundary = min(boundaries, key=lambda x: abs(x - target_position))
        
        # If the best boundary is too far, use target position
        if abs(best_boundary - target_position) > self.chunk_size // 4:
            return target_position
        
        return best_boundary
    
    def chunk_fixed_size(self, text: str, source_metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Create fixed-size chunks with overlap
        
        Args:
            text: Input text to chunk
            source_metadata: Optional metadata from source document
            
        Returns:
            List[TextChunk]: List of text chunks
        """
        if not text.strip():
            raise ChunkingError("Input text is empty")
        
        chunks = []
        text_length = len(text)
        
        # Get boundaries for intelligent splitting
        sentence_boundaries = self._find_sentence_boundaries(text) if self.respect_sentence_boundaries else []
        paragraph_boundaries = self._find_paragraph_boundaries(text) if self.respect_paragraph_boundaries else []
        
        # Combine and sort boundaries
        all_boundaries = sorted(set(sentence_boundaries + paragraph_boundaries))
        
        start_pos = 0
        chunk_index = 0
        
        while start_pos < text_length:
            # Calculate end position
            end_pos = min(start_pos + self.chunk_size, text_length)
            
            # If not at the end, try to find a better split point
            if end_pos < text_length:
                relevant_boundaries = [b for b in all_boundaries if start_pos < b <= end_pos + self.chunk_size // 4]
                if relevant_boundaries:
                    end_pos = self._find_best_split_point(text, end_pos, relevant_boundaries)
            
            # Ensure chunk is not too large
            if end_pos - start_pos > self.max_chunk_size:
                end_pos = start_pos + self.max_chunk_size
            
            # Extract chunk content
            chunk_content = text[start_pos:end_pos].strip()
            
            if len(chunk_content) >= self.min_chunk_size or chunk_index == 0:
                # Create chunk metadata
                chunk_metadata = {
                    "chunk_index": chunk_index,
                    "start_position": start_pos,
                    "end_position": end_pos,
                    "chunk_method": "fixed_size",
                    "word_count": len(chunk_content.split()),
                    "character_count": len(chunk_content)
                }
                
                if source_metadata:
                    chunk_metadata.update(source_metadata)
                
                chunk = TextChunk(
                    content=chunk_content,
                    start_index=start_pos,
                    end_index=end_pos,
                    chunk_id=f"chunk_{chunk_index}",
                    metadata=chunk_metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Calculate next start position with overlap
            if end_pos >= text_length:
                break
            
            # Find overlap start position
            overlap_start = max(start_pos, end_pos - self.chunk_overlap)
            
            # Adjust overlap start to respect boundaries if possible
            if self.respect_sentence_boundaries:
                relevant_boundaries = [b for b in sentence_boundaries if overlap_start <= b < end_pos]
                if relevant_boundaries:
                    overlap_start = max(relevant_boundaries)
            
            start_pos = overlap_start
        
        self._update_stats(len(text), chunks)
        logger.info(f"Created {len(chunks)} fixed-size chunks from text of length {len(text)}")
        return chunks
    
    def chunk_semantic(self, text: str, source_metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Create semantic chunks based on document structure
        
        Args:
            text: Input text to chunk
            source_metadata: Optional metadata from source document
            
        Returns:
            List[TextChunk]: List of text chunks
        """
        if not text.strip():
            raise ChunkingError("Input text is empty")
        
        chunks = []
        
        # Identify structural elements
        structure_elements = self._identify_structure_elements(text)
        
        # Split text into logical sections
        sections = self._split_into_sections(text, structure_elements)
        
        chunk_index = 0
        for section in sections:
            # If section is too large, split it further
            if len(section["content"]) > self.max_chunk_size:
                sub_chunks = self.chunk_fixed_size(section["content"])
                for sub_chunk in sub_chunks:
                    # Update metadata
                    sub_chunk.metadata.update({
                        "parent_section_type": section["type"],
                        "chunk_method": "semantic_with_split"
                    })
                    if source_metadata:
                        sub_chunk.metadata.update(source_metadata)
                    
                    sub_chunk.chunk_id = f"chunk_{chunk_index}"
                    chunks.append(sub_chunk)
                    chunk_index += 1
            else:
                # Create chunk from section
                chunk_metadata = {
                    "chunk_index": chunk_index,
                    "section_type": section["type"],
                    "chunk_method": "semantic",
                    "word_count": len(section["content"].split()),
                    "character_count": len(section["content"])
                }
                
                if source_metadata:
                    chunk_metadata.update(source_metadata)
                
                chunk = TextChunk(
                    content=section["content"].strip(),
                    start_index=section["start"],
                    end_index=section["end"],
                    chunk_id=f"chunk_{chunk_index}",
                    metadata=chunk_metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
        
        self._update_stats(len(text), chunks)
        logger.info(f"Created {len(chunks)} semantic chunks from text of length {len(text)}")
        return chunks
    
    def _split_into_sections(self, text: str, structure_elements: Dict[str, List[Tuple[int, int]]]) -> List[Dict[str, Any]]:
        """Split text into logical sections based on structure"""
        sections = []
        
        # Create sections based on headers if available
        headers = structure_elements.get("headers", [])
        
        if headers:
            # Use headers to define sections
            section_boundaries = [0] + [h[0] for h in headers] + [len(text)]
            
            for i in range(len(section_boundaries) - 1):
                start = section_boundaries[i]
                end = section_boundaries[i + 1]
                
                # Determine section type
                section_type = "header_section" if i > 0 else "introduction"
                
                sections.append({
                    "content": text[start:end],
                    "start": start,
                    "end": end,
                    "type": section_type
                })
        else:
            # Fall back to paragraph-based sections
            paragraphs = self.paragraph_pattern.split(text)
            current_section = ""
            start_pos = 0
            
            for paragraph in paragraphs:
                if len(current_section) + len(paragraph) < self.chunk_size:
                    current_section += paragraph + "\n\n"
                else:
                    if current_section.strip():
                        sections.append({
                            "content": current_section.strip(),
                            "start": start_pos,
                            "end": start_pos + len(current_section),
                            "type": "paragraph_section"
                        })
                        start_pos += len(current_section)
                    
                    current_section = paragraph + "\n\n"
            
            # Add final section
            if current_section.strip():
                sections.append({
                    "content": current_section.strip(),
                    "start": start_pos,
                    "end": start_pos + len(current_section),
                    "type": "paragraph_section"
                })
        
        return sections
    
    def chunk_adaptive(self, text: str, source_metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Adaptive chunking that chooses the best strategy based on text characteristics
        
        Args:
            text: Input text to chunk
            source_metadata: Optional metadata from source document
            
        Returns:
            List[TextChunk]: List of text chunks
        """
        if not text.strip():
            raise ChunkingError("Input text is empty")
        
        # Analyze text characteristics
        structure_elements = self._identify_structure_elements(text)
        has_clear_structure = (
            len(structure_elements["headers"]) > 0 or
            len(structure_elements["lists"]) > 2 or
            len(structure_elements["code_blocks"]) > 0
        )
        
        # Choose chunking strategy
        if has_clear_structure and len(text) > self.chunk_size * 2:
            logger.debug("Using semantic chunking for structured text")
            return self.chunk_semantic(text, source_metadata)
        else:
            logger.debug("Using fixed-size chunking for unstructured text")
            return self.chunk_fixed_size(text, source_metadata)
    
    def _update_stats(self, text_length: int, chunks: List[TextChunk]):
        """Update chunking statistics"""
        self.stats["documents_chunked"] += 1
        self.stats["total_chunks_created"] += len(chunks)
        self.stats["total_characters_processed"] += text_length
        
        if chunks:
            avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
            # Calculate rolling average
            total_chunks = self.stats["total_chunks_created"]
            current_avg = self.stats["average_chunk_size"]
            self.stats["average_chunk_size"] = (
                (current_avg * (total_chunks - len(chunks)) + avg_chunk_size * len(chunks)) / total_chunks
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics"""
        return self.stats.copy()


def create_text_chunker(
    chunk_size: int = 1000,
    strategy: str = "adaptive"
) -> TextChunker:
    """
    Factory function to create text chunker with preset configurations
    
    Args:
        chunk_size: Target chunk size in characters
        strategy: Chunking strategy ("fixed", "semantic", "adaptive")
        
    Returns:
        TextChunker: Configured text chunker
    """
    # Optimize settings based on strategy
    if strategy == "semantic":
        return TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=100,
            respect_sentence_boundaries=True,
            respect_paragraph_boundaries=True,
            preserve_headers=True
        )
    elif strategy == "fixed":
        return TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=200,
            respect_sentence_boundaries=True,
            respect_paragraph_boundaries=False,
            preserve_headers=False
        )
    else:  # adaptive
        return TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=150,
            respect_sentence_boundaries=True,
            respect_paragraph_boundaries=True,
            preserve_headers=True
        ) 