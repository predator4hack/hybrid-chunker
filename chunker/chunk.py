"""Data models for markdown chunks and metadata."""

from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field


class MarkdownElement(BaseModel):
    """Represents a markdown element with its properties."""
    type: str  # heading, paragraph, list, table, code_block, etc.
    content: str
    level: Optional[int] = None  # For headings (1-6)
    language: Optional[str] = None  # For code blocks
    properties: Dict[str, Any] = Field(default_factory=dict)


class MarkdownMeta(BaseModel):
    """Metadata for markdown chunks."""
    elements: List[MarkdownElement]
    headings: Optional[List[str]] = None
    heading_levels: Optional[List[int]] = None
    has_code: bool = False
    has_table: bool = False
    has_images: bool = False
    has_links: bool = False
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    
    def get_context(self, delimiter: str = "\n") -> str:
        """Get contextual information for embeddings."""
        context_parts = []
        
        if self.headings:
            context_parts.extend(self.headings)
            
        return delimiter.join(context_parts) if context_parts else ""


class MarkdownChunk(BaseModel):
    """A chunk of markdown content with metadata."""
    text: str
    meta: MarkdownMeta
    
    def contextualize(self, delimiter: str = "\n") -> str:
        """Create embedding-ready text by combining metadata and content."""
        context = self.meta.get_context(delimiter)
        if context:
            return context + delimiter + self.text
        return self.text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "text": self.text,
            "meta": self.meta.model_dump(),
            "contextualized": self.contextualize()
        }