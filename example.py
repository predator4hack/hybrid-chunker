#!/usr/bin/env python3
"""Example usage of the HybridMarkdownChunker."""

import logging
from pathlib import Path
from chunker import HybridMarkdownChunker
from chunker.config import ChunkerConfig, PresetConfigs
from chunker.tokenizer import create_tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic usage example with default settings."""
    print("=== Basic Usage Example ===")
    
    # Create a simple configuration
    config = ChunkerConfig.default_huggingface(max_tokens=512)
    tokenizer = config.create_tokenizer()
    
    # Initialize chunker
    chunker = HybridMarkdownChunker(tokenizer=tokenizer)
    
    # Sample markdown content
    sample_markdown = """
# Introduction to AI

Artificial Intelligence (AI) is a rapidly evolving field that aims to create intelligent machines.

## Types of AI

### Narrow AI
Narrow AI, also known as weak AI, is designed to perform specific tasks.

#### Examples
- Voice assistants like Siri and Alexa
- Image recognition systems
- Chess-playing programs

### General AI
General AI refers to machines that can understand, learn, and apply intelligence broadly.

## Applications

AI has numerous applications across various industries:

| Industry | Application | Benefits |
|----------|-------------|----------|
| Healthcare | Diagnosis | Faster, more accurate |
| Finance | Fraud detection | Reduced losses |
| Transportation | Autonomous vehicles | Safety improvements |

### Code Example

Here's a simple Python example:

```python
def greet(name):
    return f"Hello, {name}!"

print(greet("AI"))
```

> AI will continue to shape our future in unprecedented ways.

## Conclusion

The future of AI looks promising with continued research and development.
"""
    
    # Chunk the content
    chunks = chunker.chunk_text(sample_markdown)
    
    # Display results
    print(f"Generated {len(chunks)} chunks:")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"Headings: {chunk.meta.headings}")
        print(f"Has code: {chunk.meta.has_code}")
        print(f"Has table: {chunk.meta.has_table}")
        print(f"Token count: {chunker._count_chunk_tokens(chunk)}")
        print(f"Text preview: {chunk.text[:100]}...")
        print("-" * 30)


def example_with_presets():
    """Example using predefined configuration presets."""
    print("\n=== Preset Configurations Example ===")
    
    sample_text = """
# Technical Documentation

## Overview
This is a comprehensive guide to using our API.

### Authentication
Use Bearer tokens for authentication:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" https://api.example.com
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /users | List all users |
| POST | /users | Create new user |
| PUT | /users/{id} | Update user |

## Error Handling

Common error codes:
- 400: Bad Request
- 401: Unauthorized  
- 404: Not Found
- 500: Internal Server Error
"""
    
    presets = [
        ("Embedding Optimized", PresetConfigs.embedding_optimized()),
        ("LLM Optimized", PresetConfigs.llm_optimized()),
        ("Memory Efficient", PresetConfigs.memory_efficient()),
    ]
    
    for name, config in presets:
        print(f"\n{name} Config:")
        tokenizer = config.create_tokenizer()
        chunker = HybridMarkdownChunker(
            tokenizer=tokenizer,
            merge_peers=config.merge_peers
        )
        
        chunks = chunker.chunk_text(sample_text)
        print(f"  Chunks: {len(chunks)}")
        print(f"  Max tokens: {config.max_tokens}")
        print(f"  Tokenizer: {config.tokenizer_type}")
        
        # Show first chunk details
        if chunks:
            first_chunk = chunks[0]
            print(f"  First chunk tokens: {chunker._count_chunk_tokens(first_chunk)}")


def example_custom_config():
    """Example with custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Custom config for specific needs
    config = ChunkerConfig(
        tokenizer_type="huggingface",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=300,
        merge_peers=False,  # Keep chunks separate
        delimiter=" ",  # Use space instead of newline
        context_ratio=0.1,  # Minimal context
    )
    
    tokenizer = config.create_tokenizer()
    chunker = HybridMarkdownChunker(
        tokenizer=tokenizer,
        merge_peers=config.merge_peers,
        delimiter=config.delimiter
    )
    
    sample_text = """
# Data Science Workflow

## Data Collection
Gathering relevant data from various sources.

## Data Cleaning  
Removing inconsistencies and handling missing values.

## Data Analysis
Exploring patterns and relationships in the data.

## Model Building
Creating predictive models using machine learning.

## Validation
Testing model performance on unseen data.
"""
    
    chunks = chunker.chunk_text(sample_text)
    
    print(f"Custom config generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Headings: {chunk.meta.headings}")
        print(f"  Elements: {len(chunk.meta.elements)}")
        print(f"  Tokens: {chunker._count_chunk_tokens(chunk)}")
        print(f"  Text: {chunk.text[:80]}...")


def example_file_processing():
    """Example of processing markdown files."""
    print("\n=== File Processing Example ===")
    
    # Create a sample markdown file
    sample_file = Path("sample_document.md")
    sample_content = """
# Project Documentation

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/example/project.git
cd project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## Features

Our application includes:

- **User Management**: Create, read, update, delete users
- **Data Visualization**: Interactive charts and graphs  
- **API Integration**: RESTful API endpoints
- **Real-time Updates**: WebSocket connections

### Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Response Time | 200ms | <300ms |
| Throughput | 1000 req/s | >500 req/s |
| Availability | 99.9% | >99% |

## Configuration

Create a `.env` file:

```env
DATABASE_URL=postgresql://user:pass@localhost/db
API_KEY=your_secret_key
DEBUG=false
```

## Troubleshooting

Common issues and solutions:

### Database Connection Error
Check your database URL and credentials.

### Port Already in Use  
Change the port in your configuration file.

### Permission Denied
Run with appropriate permissions or check file ownership.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.
"""
    
    # Write sample file
    sample_file.write_text(sample_content)
    
    try:
        # Process the file
        config = PresetConfigs.embedding_optimized()
        tokenizer = config.create_tokenizer()
        chunker = HybridMarkdownChunker(tokenizer=tokenizer)
        
        chunks = chunker.chunk_file(str(sample_file))
        
        print(f"Processed {sample_file.name}:")
        print(f"Generated {len(chunks)} chunks")
        
        # Analyze chunk distribution
        token_counts = [chunker._count_chunk_tokens(chunk) for chunk in chunks]
        print(f"Token distribution:")
        print(f"  Min: {min(token_counts)}")
        print(f"  Max: {max(token_counts)}")
        print(f"  Avg: {sum(token_counts) / len(token_counts):.1f}")
        
        # Show chunks with different features
        code_chunks = [c for c in chunks if c.meta.has_code]
        table_chunks = [c for c in chunks if c.meta.has_table]
        
        print(f"Chunks with code blocks: {len(code_chunks)}")
        print(f"Chunks with tables: {len(table_chunks)}")
        
        # Export chunks to JSON
        output_data = [chunk.to_dict() for chunk in chunks]
        
        print(f"\nSample chunk structure:")
        if chunks:
            sample = chunks[0].to_dict()
            print(f"Keys: {list(sample.keys())}")
            print(f"Meta keys: {list(sample['meta'].keys())}")
    
    finally:
        # Clean up
        if sample_file.exists():
            sample_file.unlink()


if __name__ == "__main__":
    try:
        example_basic_usage()
        example_with_presets()
        example_custom_config()
        example_file_processing()
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Example failed")