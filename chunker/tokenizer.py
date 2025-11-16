"""Tokenization utilities for the hybrid chunker."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        pass
    
    @abstractmethod
    def get_tokenizer(self) -> Any:
        """Get the underlying tokenizer object."""
        pass


class HuggingFaceTokenizer(BaseTokenizer):
    """HuggingFace tokenizer implementation."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_tokens: Optional[int] = None):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("transformers package is required for HuggingFaceTokenizer")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Auto-detect max tokens if not specified
        if max_tokens is None:
            max_tokens = getattr(self.tokenizer, 'model_max_length', 512)
            if max_tokens > 100000:  # Some models have unrealistic max lengths
                max_tokens = 512
        
        super().__init__(max_tokens=max_tokens)
        logger.info(f"Initialized HuggingFace tokenizer with model {model_name}, max_tokens: {max_tokens}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using HuggingFace tokenizer."""
        if not text:
            return 0
        
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            # Fallback to rough estimation
            return len(text.split()) // 0.75  # Rough approximation
    
    def get_tokenizer(self):
        """Get the underlying HuggingFace tokenizer."""
        return self.tokenizer


class OpenAITokenizer(BaseTokenizer):
    """OpenAI tiktoken tokenizer implementation."""
    
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 128 * 1024):
        try:
            import tiktoken
        except ImportError:
            raise ImportError("tiktoken package is required for OpenAITokenizer")
        
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model)
        super().__init__(max_tokens=max_tokens)
        logger.info(f"Initialized OpenAI tokenizer for model {model}, max_tokens: {max_tokens}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if not text:
            return 0
        
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            # Fallback to rough estimation  
            return len(text.split()) // 0.75  # Rough approximation
    
    def get_tokenizer(self):
        """Get the underlying tiktoken tokenizer."""
        return self.tokenizer


def create_tokenizer(tokenizer_type: str = "huggingface", **kwargs) -> BaseTokenizer:
    """Factory function to create tokenizers."""
    if tokenizer_type.lower() == "huggingface":
        return HuggingFaceTokenizer(**kwargs)
    elif tokenizer_type.lower() == "openai":
        return OpenAITokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Supported: 'huggingface', 'openai'")