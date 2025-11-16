# Hybrid Markdown Chunker

A standalone hybrid chunker for markdown files inspired by docling-core's architecture. This chunker provides hierarchical and semantic-aware text segmentation with minimal dependencies.

## Features

- **3-Stage Chunking**: Hierarchical → Element-aware → Semantic splitting
- **Markdown Feature Preservation**: Headers, code blocks, tables, lists, links, images
- **Configurable Tokenizers**: Support for HuggingFace and OpenAI tokenizers
- **Hierarchical Context**: Maintains heading hierarchy in chunk metadata
- **Smart Merging**: Combines undersized chunks with matching context
- **Minimal Dependencies**: Lightweight implementation for easy integration

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from chunker import HybridMarkdownChunker
from chunker.config import ChunkerConfig

# Create configuration
config = ChunkerConfig.default_huggingface(max_tokens=512)
tokenizer = config.create_tokenizer()

# Initialize chunker
chunker = HybridMarkdownChunker(tokenizer=tokenizer)

# Chunk markdown content
with open('document.md', 'r') as f:
    content = f.read()

chunks = chunker.chunk_text(content)

# Process chunks
for chunk in chunks:
    print(f"Headings: {chunk.meta.headings}")
    print(f"Has code: {chunk.meta.has_code}")
    print(f"Text: {chunk.text[:100]}...")
    print("-" * 40)
```

## Original Requirements

Questions about Implementation Requirements:

1. Input Format: Should the standalone chunker:

    - Accept raw markdown files directly, or
    - Require conversion to DoclingDocument format first?
    - Support both approaches?
      Accept raw markdown files directly

2. Dependencies: For the standalone version, would you prefer:

    - Minimal dependencies (reimplementing key functionality)
    - Reusing docling_core components where possible
    - A hybrid approach (minimal deps but leverage semchunk for semantic splitting)?
      Minimal dependencies

3. Tokenization: Which tokenizer would you prefer:

    - HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
    - OpenAI (tiktoken)
    - Both options with configurable choice?
      Both, configurable choice

4. Output Format: Should the chunks be:

    - Simple text chunks with metadata dictionaries
    - Compatible with docling_core's DocChunk format
    - A simplified custom format optimized for your use case?
      A simplified custom format(similar to DocChunk format)

5. Markdown Features: Which markdown elements should be preserved:

    - Headers (# ## ###) for hierarchical chunking
    - Code blocks with language-specific handling
    - Tables, lists, links, images
    - All of the above?
      All of above

6. Configuration: What level of configurability do you need:

    - Just basic token limits and merge options
    - Full serialization customization like docling_core
    - Somewhere in between?
      Basic token limits and merge options for now

## Detailed Implementation Plan

### Architecture Overview

Based on the docling_core analysis and requirements, the standalone hybrid chunker will follow a simplified 4-stage pipeline:

1. **Markdown Parsing Stage**: Parse raw markdown into structured elements
2. **Hierarchical Chunking Stage**: Create structure-aware chunks respecting markdown hierarchy
3. **Token-Aware Splitting Stage**: Split oversized chunks using semantic segmentation
4. **Chunk Merging Stage**: Merge undersized chunks with matching hierarchical context

### Core Components to Implement

#### 1. Data Models (`models.py`)

```python
@dataclass
class ChunkMetadata:
    """Simplified metadata for chunks"""
    headings: List[str]  # Hierarchical section context
    element_types: List[str]  # Types of markdown elements in chunk
    start_line: int  # Source line number
    end_line: int  # Source line number
    token_count: int  # Actual token count

@dataclass
class MarkdownChunk:
    """Main chunk data structure"""
    text: str  # Chunk content
    metadata: ChunkMetadata  # Rich metadata

    def contextualize(self) -> str:
        """Combine metadata + text for embeddings"""
        context = "\n".join(self.metadata.headings) if self.metadata.headings else ""
        return f"{context}\n{self.text}".strip()
```

#### 2. Tokenizer Abstraction (`tokenizers.py`)

```python
class BaseTokenizer(ABC):
    """Abstract tokenizer interface"""
    def __init__(self, max_tokens: int): ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    @property
    def max_tokens(self) -> int: ...

class HuggingFaceTokenizer(BaseTokenizer):
    """HuggingFace transformers tokenizer"""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_tokens: Optional[int] = None): ...

class OpenAITokenizer(BaseTokenizer):
    """OpenAI tiktoken tokenizer"""
    def __init__(self, model_name: str = "gpt-4o", max_tokens: int): ...

def create_tokenizer(tokenizer_type: str, **kwargs) -> BaseTokenizer:
    """Factory function for tokenizer creation"""
```

#### 3. Markdown Parser (`parser.py`)

````python
@dataclass
class MarkdownElement:
    """Represents a parsed markdown element"""
    element_type: str  # heading, paragraph, code_block, table, list, etc.
    content: str  # Raw text content
    level: Optional[int]  # For headings (1-6)
    language: Optional[str]  # For code blocks
    line_start: int  # Source line number
    line_end: int  # Source line number

class MarkdownParser:
    """Parse markdown into structured elements"""

    def parse(self, markdown_text: str) -> List[MarkdownElement]:
        """Parse markdown text into elements"""
        # Use regex patterns to identify:
        # - Headers: ^#{1,6}\s+(.+)$
        # - Code blocks: ```(\w+)?\n(.*?)\n```
        # - Tables: Lines with |
        # - Lists: Lines starting with -, *, +, or numbers
        # - Paragraphs: Everything else

    def _extract_headers(self, lines: List[str]) -> List[MarkdownElement]: ...
    def _extract_code_blocks(self, lines: List[str]) -> List[MarkdownElement]: ...
    def _extract_tables(self, lines: List[str]) -> List[MarkdownElement]: ...
    def _extract_lists(self, lines: List[str]) -> List[MarkdownElement]: ...
````

#### 4. Hierarchical Chunker (`hierarchical_chunker.py`)

```python
class HierarchicalChunker:
    """Structure-aware chunking based on markdown hierarchy"""

    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer

    def chunk(self, elements: List[MarkdownElement]) -> List[MarkdownChunk]:
        """Create structure-aware chunks"""
        chunks = []
        current_chunk_elements = []
        current_headings = []

        for element in elements:
            if element.element_type == "heading":
                # Close current chunk if exists
                if current_chunk_elements:
                    chunks.append(self._create_chunk(current_chunk_elements, current_headings))
                    current_chunk_elements = []

                # Update heading hierarchy
                current_headings = self._update_headings(current_headings, element)
                current_chunk_elements.append(element)
            else:
                current_chunk_elements.append(element)

        # Handle final chunk
        if current_chunk_elements:
            chunks.append(self._create_chunk(current_chunk_elements, current_headings))

        return chunks

    def _update_headings(self, current_headings: List[str], header: MarkdownElement) -> List[str]: ...
    def _create_chunk(self, elements: List[MarkdownElement], headings: List[str]) -> MarkdownChunk: ...
```

#### 5. Semantic Text Splitter (`text_splitter.py`)

```python
class SemanticTextSplitter:
    """Reimplemented semantic text splitting logic (minimal dependency version of semchunk)"""

    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer

    def split_text(self, text: str, max_tokens: int) -> List[str]:
        """Split text semantically while respecting token limits"""
        # Implement simplified semantic splitting:
        # 1. Split by sentences first
        # 2. Try to group sentences up to token limit
        # 3. If single sentence > limit, split by clauses/phrases
        # 4. Last resort: hard character split

    def _split_by_sentences(self, text: str) -> List[str]: ...
    def _split_by_phrases(self, text: str) -> List[str]: ...
    def _hard_split(self, text: str, max_chars: int) -> List[str]: ...
```

#### 6. Main Hybrid Chunker (`hybrid_chunker.py`)

```python
class MarkdownHybridChunker:
    """Main hybrid chunker implementation"""

    def __init__(self,
                 tokenizer: BaseTokenizer,
                 merge_peers: bool = True,
                 metadata_overhead: int = 50):  # Estimated tokens for metadata
        self.tokenizer = tokenizer
        self.merge_peers = merge_peers
        self.metadata_overhead = metadata_overhead
        self.parser = MarkdownParser()
        self.hierarchical_chunker = HierarchicalChunker(tokenizer)
        self.text_splitter = SemanticTextSplitter(tokenizer)

    def chunk_markdown_file(self, file_path: str) -> List[MarkdownChunk]:
        """Main entry point for chunking markdown files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return self.chunk_markdown_text(f.read())

    def chunk_markdown_text(self, markdown_text: str) -> List[MarkdownChunk]:
        """4-stage chunking pipeline"""
        # Stage 1: Parse markdown
        elements = self.parser.parse(markdown_text)

        # Stage 2: Hierarchical chunking
        chunks = self.hierarchical_chunker.chunk(elements)

        # Stage 3: Token-aware splitting
        chunks = self._split_oversized_chunks(chunks)

        # Stage 4: Merge peers (optional)
        if self.merge_peers:
            chunks = self._merge_undersized_chunks(chunks)

        return chunks

    def _split_oversized_chunks(self, chunks: List[MarkdownChunk]) -> List[MarkdownChunk]: ...
    def _merge_undersized_chunks(self, chunks: List[MarkdownChunk]) -> List[MarkdownChunk]: ...
    def _calculate_metadata_tokens(self, metadata: ChunkMetadata) -> int: ...
```

### File Structure

```
standalone_hybrid_chunker/
├── __init__.py
├── models.py              # Data models (ChunkMetadata, MarkdownChunk)
├── tokenizers.py          # Tokenizer implementations and factory
├── parser.py              # Markdown parsing logic
├── hierarchical_chunker.py # Structure-aware chunking
├── text_splitter.py       # Semantic text splitting
├── hybrid_chunker.py      # Main hybrid chunker class
├── config.py              # Configuration classes
├── utils.py               # Utility functions
└── examples/
    ├── basic_usage.py     # Basic usage examples
    └── advanced_config.py # Advanced configuration examples
```

### Dependencies (Minimal)

```toml
[dependencies]
python = "^3.8"

# Core dependencies
typing-extensions = "^4.0.0"  # For older Python versions

# Optional dependencies for tokenizers
transformers = { version = "^4.21.0", optional = true }
torch = { version = "^2.0.0", optional = true }
tiktoken = { version = "^0.4.0", optional = true }

[extras]
huggingface = ["transformers", "torch"]
openai = ["tiktoken"]
all = ["transformers", "torch", "tiktoken"]
```

### Configuration System (`config.py`)

```python
@dataclass
class ChunkerConfig:
    """Configuration for the hybrid chunker"""
    # Tokenizer settings
    tokenizer_type: str = "huggingface"  # "huggingface" or "openai"
    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: Optional[int] = None  # Auto-detect if None

    # Chunking behavior
    merge_peers: bool = True
    metadata_overhead: int = 50

    # Parsing options
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    preserve_lists: bool = True

    @classmethod
    def for_huggingface(cls, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> "ChunkerConfig": ...

    @classmethod
    def for_openai(cls, model: str = "gpt-4o", max_tokens: int = 8192) -> "ChunkerConfig": ...
```

### Usage Examples

#### Basic Usage:

```python
from standalone_hybrid_chunker import MarkdownHybridChunker, ChunkerConfig

# Create chunker with HuggingFace tokenizer
config = ChunkerConfig.for_huggingface()
chunker = MarkdownHybridChunker.from_config(config)

# Chunk a markdown file
chunks = chunker.chunk_markdown_file("document.md")

# Process chunks
for chunk in chunks:
    print(f"Headings: {chunk.metadata.headings}")
    print(f"Text: {chunk.text}")
    print(f"Tokens: {chunk.metadata.token_count}")
    print("---")
```

#### Advanced Configuration:

```python
# Custom tokenizer configuration
config = ChunkerConfig(
    tokenizer_type="openai",
    tokenizer_model="gpt-4o",
    max_tokens=4096,
    merge_peers=False,
    metadata_overhead=100
)

chunker = MarkdownHybridChunker.from_config(config)
chunks = chunker.chunk_markdown_text(markdown_content)
```

### Implementation Phases

#### Phase 1: Core Structure (MVP)

1. Implement basic data models (`models.py`)
2. Create tokenizer abstraction with HuggingFace support (`tokenizers.py`)
3. Basic markdown parser for headers and paragraphs (`parser.py`)
4. Simple hierarchical chunker (`hierarchical_chunker.py`)
5. Main hybrid chunker with basic pipeline (`hybrid_chunker.py`)

#### Phase 2: Enhanced Parsing

1. Add support for code blocks, tables, lists in parser
2. Implement semantic text splitter (`text_splitter.py`)
3. Add token-aware splitting stage
4. Add chunk merging stage

#### Phase 3: Tokenizer Support & Polish

1. Add OpenAI tokenizer support
2. Implement configuration system (`config.py`)
3. Add comprehensive examples and documentation
4. Add error handling and validation

#### Phase 4: Testing & Optimization

1. Unit tests for all components
2. Integration tests with real markdown files
3. Performance optimizations
4. Memory usage optimization for large files

### Key Implementation Considerations

1. **Memory Efficiency**: Process large markdown files in chunks to avoid loading entire file into memory
2. **Error Handling**: Graceful handling of malformed markdown and tokenization errors
3. **Extensibility**: Plugin system for custom markdown element handlers
4. **Performance**: Efficient regex patterns and minimal object creation in parsing loops
5. **Compatibility**: Support for different markdown flavors (CommonMark, GitHub Flavored Markdown)

### Testing Strategy

1. **Unit Tests**: Individual component testing with mock data
2. **Integration Tests**: End-to-end testing with real markdown files
3. **Performance Tests**: Benchmarking with large documents
4. **Comparison Tests**: Output comparison with docling_core chunker for validation

This implementation plan provides a complete roadmap for building a standalone hybrid chunker that maintains the sophistication of docling_core's approach while being lightweight and focused on markdown processing.
