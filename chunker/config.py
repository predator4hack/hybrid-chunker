"""Configuration utilities for the hybrid chunker."""

from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from .tokenizer import create_tokenizer, BaseTokenizer


class ChunkerConfig(BaseModel):
    """Configuration for the HybridMarkdownChunker."""
    
    # Tokenizer settings
    tokenizer_type: str = Field(default="huggingface", description="Type of tokenizer: 'huggingface' or 'openai'")
    model_name: Optional[str] = Field(default=None, description="Model name for tokenizer")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens per chunk")
    
    # Chunking behavior
    merge_peers: bool = Field(default=True, description="Merge undersized chunks with same headings")
    delimiter: str = Field(default="\n", description="Delimiter for joining text")
    overlap_tokens: int = Field(default=50, description="Token overlap between chunks")
    
    # Advanced settings
    context_ratio: float = Field(default=0.2, description="Max ratio of tokens for context (headings)")
    min_chunk_size: int = Field(default=50, description="Minimum chunk size in tokens")
    
    @validator("tokenizer_type")
    def validate_tokenizer_type(cls, v):
        if v.lower() not in ["huggingface", "openai"]:
            raise ValueError("tokenizer_type must be 'huggingface' or 'openai'")
        return v.lower()
    
    @validator("context_ratio")
    def validate_context_ratio(cls, v):
        if not 0 < v < 1:
            raise ValueError("context_ratio must be between 0 and 1")
        return v
    
    def create_tokenizer(self) -> BaseTokenizer:
        """Create a tokenizer instance from the configuration."""
        kwargs = {}
        
        if self.model_name:
            if self.tokenizer_type == "huggingface":
                kwargs["model_name"] = self.model_name
            elif self.tokenizer_type == "openai":
                kwargs["model"] = self.model_name
        
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        
        return create_tokenizer(self.tokenizer_type, **kwargs)
    
    @classmethod
    def default_huggingface(cls, max_tokens: int = 512, **kwargs) -> "ChunkerConfig":
        """Create default HuggingFace configuration."""
        return cls(
            tokenizer_type="huggingface",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=max_tokens,
            **kwargs
        )
    
    @classmethod
    def default_openai(cls, max_tokens: int = 8192, **kwargs) -> "ChunkerConfig":
        """Create default OpenAI configuration."""
        return cls(
            tokenizer_type="openai",
            model_name="gpt-4o",
            max_tokens=max_tokens,
            **kwargs
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ChunkerConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()


class PresetConfigs:
    """Predefined configuration presets."""
    
    @staticmethod
    def embedding_optimized() -> ChunkerConfig:
        """Configuration optimized for embedding models."""
        return ChunkerConfig.default_huggingface(
            max_tokens=384,  # Good for most embedding models
            merge_peers=True,
            context_ratio=0.15,
            min_chunk_size=50
        )
    
    @staticmethod
    def llm_optimized() -> ChunkerConfig:
        """Configuration optimized for LLM context."""
        return ChunkerConfig.default_openai(
            max_tokens=4096,
            merge_peers=False,  # Preserve structure for LLMs
            context_ratio=0.25,
            min_chunk_size=100
        )
    
    @staticmethod
    def memory_efficient() -> ChunkerConfig:
        """Configuration for memory-constrained environments."""
        return ChunkerConfig.default_huggingface(
            max_tokens=256,
            merge_peers=True,
            context_ratio=0.1,
            min_chunk_size=25
        )
    
    @staticmethod
    def high_context() -> ChunkerConfig:
        """Configuration for high context preservation."""
        return ChunkerConfig.default_openai(
            max_tokens=16384,
            merge_peers=False,
            context_ratio=0.3,
            overlap_tokens=100
        )