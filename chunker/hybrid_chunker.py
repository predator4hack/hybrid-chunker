"""Hybrid chunker implementation for markdown files."""

import logging
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path

from .chunk import MarkdownChunk, MarkdownMeta, MarkdownElement
from .tokenizer import BaseTokenizer
from .markdown_parser import MarkdownParser

logger = logging.getLogger(__name__)


class HybridMarkdownChunker:
    """Hybrid chunker for markdown files with hierarchical and semantic awareness."""
    
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        merge_peers: bool = True,
        delimiter: str = "\n",
        overlap_tokens: int = 50,
    ):
        self.tokenizer = tokenizer
        self.merge_peers = merge_peers
        self.delimiter = delimiter
        self.overlap_tokens = overlap_tokens
        self.parser = MarkdownParser()
        
        logger.info(f"Initialized HybridMarkdownChunker with max_tokens: {tokenizer.max_tokens}")
    
    def chunk_file(self, file_path: str) -> List[MarkdownChunk]:
        """Chunk a markdown file."""
        content = Path(file_path).read_text(encoding='utf-8')
        return self.chunk_text(content)
    
    def chunk_text(self, content: str) -> List[MarkdownChunk]:
        """
        Chunk markdown text using a 3-stage hybrid approach:
        1. Hierarchical chunking (structure-aware)
        2. Element-wise splitting (token-aware)
        3. Semantic text splitting (semchunk fallback)
        """
        # Stage 1: Parse markdown into structured elements
        elements = self.parser.parse(content)
        logger.debug(f"Parsed {len(elements)} markdown elements")
        
        # Stage 2: Create initial chunks from hierarchical structure
        hierarchical_chunks = self._create_hierarchical_chunks(elements)
        logger.debug(f"Created {len(hierarchical_chunks)} hierarchical chunks")
        
        # Stage 3: Split oversized chunks by elements
        element_split_chunks = []
        for chunk in hierarchical_chunks:
            element_split_chunks.extend(self._split_by_elements(chunk))
        
        logger.debug(f"Element splitting resulted in {len(element_split_chunks)} chunks")
        
        # Stage 4: Apply semantic text splitting to oversized chunks
        final_chunks = []
        for chunk in element_split_chunks:
            final_chunks.extend(self._split_by_semantic_text(chunk))
        
        logger.debug(f"Semantic splitting resulted in {len(final_chunks)} chunks")
        
        # Stage 5: Merge undersized chunks if enabled
        if self.merge_peers:
            final_chunks = self._merge_chunks(final_chunks)
            logger.debug(f"Merging resulted in {len(final_chunks)} final chunks")
        
        return final_chunks
    
    def _create_hierarchical_chunks(self, elements: List[MarkdownElement]) -> List[MarkdownChunk]:
        """Create chunks based on document hierarchy."""
        chunks = []
        current_headings = {}  # level -> heading text
        current_heading_levels = []
        
        i = 0
        while i < len(elements):
            element = elements[i]
            
            # Update heading context
            if element.type == "heading":
                level = element.level
                current_headings[level] = element.content
                
                # Clear deeper levels
                keys_to_remove = [k for k in current_headings.keys() if k > level]
                for k in keys_to_remove:
                    del current_headings[k]
                
                # Update heading path
                current_heading_levels = sorted(current_headings.keys())
            
            # Create chunk starting from current element
            chunk_elements = [element]
            chunk_text_parts = [element.content]
            
            # Look ahead to include more elements until we hit size limit or new section
            j = i + 1
            while j < len(elements):
                next_element = elements[j]
                
                # Stop if we hit a heading of same or higher level
                if (next_element.type == "heading" and 
                    element.type == "heading" and 
                    next_element.level <= element.level):
                    break
                
                # Calculate tokens if we add this element
                test_text = self.delimiter.join(chunk_text_parts + [next_element.content])
                test_tokens = self._count_text_tokens(test_text)
                
                # Check if adding this element would exceed token limit
                if test_tokens > self.tokenizer.max_tokens * 0.8:  # Leave room for context
                    break
                
                chunk_elements.append(next_element)
                chunk_text_parts.append(next_element.content)
                j += 1
            
            # Create chunk with current heading context
            headings = [current_headings[level] for level in current_heading_levels]
            
            chunk = self._create_chunk(
                chunk_elements, 
                headings, 
                current_heading_levels
            )
            
            chunks.append(chunk)
            i = j
        
        return chunks
    
    def _split_by_elements(self, chunk: MarkdownChunk) -> List[MarkdownChunk]:
        """Split chunks that are too large by individual elements."""
        if self._count_chunk_tokens(chunk) <= self.tokenizer.max_tokens:
            return [chunk]
        
        chunks = []
        current_elements = []
        current_text_parts = []
        
        for element in chunk.meta.elements:
            # Test adding this element
            test_elements = current_elements + [element]
            test_text = self.delimiter.join(current_text_parts + [element.content])
            test_tokens = self._count_text_tokens(test_text)
            
            # Calculate tokens with context
            test_chunk = self._create_chunk(test_elements, chunk.meta.headings, chunk.meta.heading_levels)
            test_total_tokens = self._count_chunk_tokens(test_chunk)
            
            if test_total_tokens <= self.tokenizer.max_tokens:
                current_elements.append(element)
                current_text_parts.append(element.content)
            else:
                # Create chunk with current elements if any
                if current_elements:
                    new_chunk = self._create_chunk(
                        current_elements, 
                        chunk.meta.headings, 
                        chunk.meta.heading_levels
                    )
                    chunks.append(new_chunk)
                
                # Start new chunk with current element
                current_elements = [element]
                current_text_parts = [element.content]
        
        # Add remaining elements as final chunk
        if current_elements:
            final_chunk = self._create_chunk(
                current_elements, 
                chunk.meta.headings, 
                chunk.meta.heading_levels
            )
            chunks.append(final_chunk)
        
        return chunks
    
    def _split_by_semantic_text(self, chunk: MarkdownChunk) -> List[MarkdownChunk]:
        """Split oversized chunks using semantic text splitting."""
        if self._count_chunk_tokens(chunk) <= self.tokenizer.max_tokens:
            return [chunk]
        
        # Calculate available space for text content (subtract context overhead)
        context_tokens = self._count_context_tokens(chunk.meta.headings or [])
        available_tokens = max(
            self.tokenizer.max_tokens - context_tokens - 10,  # 10 token buffer
            self.tokenizer.max_tokens // 2  # Minimum 50% for content
        )
        
        try:
            # Use semchunk for semantic splitting
            import semchunk
            chunker = semchunk.chunkerify(
                self.tokenizer.get_tokenizer(),
                chunk_size=available_tokens
            )
            
            text_segments = chunker.chunk(chunk.text)
            logger.debug(f"Semchunk split text into {len(text_segments)} segments")
            
        except ImportError:
            logger.warning("semchunk not available, falling back to simple text splitting")
            text_segments = self._simple_text_split(chunk.text, available_tokens)
        except Exception as e:
            logger.warning(f"Semchunk failed: {e}, falling back to simple splitting")
            text_segments = self._simple_text_split(chunk.text, available_tokens)
        
        # Create chunks from segments
        chunks = []
        for segment in text_segments:
            # Create simplified element for text segment
            segment_element = MarkdownElement(
                type="text_segment",
                content=segment
            )
            
            new_chunk = MarkdownChunk(
                text=segment,
                meta=MarkdownMeta(
                    elements=[segment_element],
                    headings=chunk.meta.headings,
                    heading_levels=chunk.meta.heading_levels,
                    has_code=chunk.meta.has_code,
                    has_table=chunk.meta.has_table,
                    has_images=chunk.meta.has_images,
                    has_links=chunk.meta.has_links,
                )
            )
            chunks.append(new_chunk)
        
        return chunks
    
    def _simple_text_split(self, text: str, max_tokens: int) -> List[str]:
        """Simple text splitting fallback when semchunk is not available."""
        sentences = text.split('. ')
        segments = []
        current_segment = []
        
        for sentence in sentences:
            test_segment = '. '.join(current_segment + [sentence])
            if self._count_text_tokens(test_segment) <= max_tokens:
                current_segment.append(sentence)
            else:
                if current_segment:
                    segments.append('. '.join(current_segment))
                    current_segment = [sentence]
                else:
                    # Single sentence too long, split by words
                    words = sentence.split()
                    current_words = []
                    for word in words:
                        test_text = ' '.join(current_words + [word])
                        if self._count_text_tokens(test_text) <= max_tokens:
                            current_words.append(word)
                        else:
                            if current_words:
                                segments.append(' '.join(current_words))
                            current_words = [word]
                    if current_words:
                        current_segment = [' '.join(current_words)]
        
        if current_segment:
            segments.append('. '.join(current_segment))
        
        return segments
    
    def _merge_chunks(self, chunks: List[MarkdownChunk]) -> List[MarkdownChunk]:
        """Merge undersized chunks with matching headings."""
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            # Check if chunks can be merged (same headings)
            if (current_chunk.meta.headings == next_chunk.meta.headings and
                self._can_merge_chunks(current_chunk, next_chunk)):
                
                # Merge the chunks
                merged_elements = current_chunk.meta.elements + next_chunk.meta.elements
                merged_text = current_chunk.text + self.delimiter + next_chunk.text
                
                current_chunk = MarkdownChunk(
                    text=merged_text,
                    meta=MarkdownMeta(
                        elements=merged_elements,
                        headings=current_chunk.meta.headings,
                        heading_levels=current_chunk.meta.heading_levels,
                        has_code=current_chunk.meta.has_code or next_chunk.meta.has_code,
                        has_table=current_chunk.meta.has_table or next_chunk.meta.has_table,
                        has_images=current_chunk.meta.has_images or next_chunk.meta.has_images,
                        has_links=current_chunk.meta.has_links or next_chunk.meta.has_links,
                    )
                )
            else:
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        merged_chunks.append(current_chunk)
        return merged_chunks
    
    def _can_merge_chunks(self, chunk1: MarkdownChunk, chunk2: MarkdownChunk) -> bool:
        """Check if two chunks can be merged without exceeding token limit."""
        test_text = chunk1.text + self.delimiter + chunk2.text
        test_chunk = MarkdownChunk(
            text=test_text,
            meta=MarkdownMeta(
                elements=chunk1.meta.elements + chunk2.meta.elements,
                headings=chunk1.meta.headings
            )
        )
        
        return self._count_chunk_tokens(test_chunk) <= self.tokenizer.max_tokens
    
    def _create_chunk(
        self, 
        elements: List[MarkdownElement], 
        headings: Optional[List[str]], 
        heading_levels: Optional[List[int]]
    ) -> MarkdownChunk:
        """Create a chunk from elements and metadata."""
        text = self.delimiter.join([elem.content for elem in elements])
        
        # Analyze features
        features = self.parser.get_markdown_features(elements)
        
        meta = MarkdownMeta(
            elements=elements,
            headings=headings,
            heading_levels=heading_levels,
            has_code=features["has_code"],
            has_table=features["has_tables"],
            has_images=features["has_images"],
            has_links=features["has_links"],
        )
        
        return MarkdownChunk(text=text, meta=meta)
    
    def _count_text_tokens(self, text: str) -> int:
        """Count tokens in text only."""
        return self.tokenizer.count_tokens(text)
    
    def _count_context_tokens(self, headings: List[str]) -> int:
        """Count tokens in context (headings)."""
        if not headings:
            return 0
        return self.tokenizer.count_tokens(self.delimiter.join(headings))
    
    def _count_chunk_tokens(self, chunk: MarkdownChunk) -> int:
        """Count total tokens in contextualized chunk."""
        return self.tokenizer.count_tokens(chunk.contextualize(self.delimiter))