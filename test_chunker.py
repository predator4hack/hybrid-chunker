#!/usr/bin/env python3
"""Test script for the HybridMarkdownChunker."""

from pathlib import Path
from chunker import HybridMarkdownChunker, MarkdownChunk, MarkdownMeta
from chunker.config import ChunkerConfig
from chunker.tokenizer import create_tokenizer


def create_test_chunker(max_tokens=512):
    """Create a test chunker instance."""
    config = ChunkerConfig.default_huggingface(max_tokens=max_tokens)
    tokenizer = config.create_tokenizer()
    return HybridMarkdownChunker(tokenizer=tokenizer)


def test_simple_markdown():
    """Test chunking of simple markdown content."""
    print("Testing simple markdown...")
    
    chunker = create_test_chunker()
    
    content = """
# Title
This is a simple paragraph.

## Subtitle
Another paragraph here.
"""
    
    chunks = chunker.chunk_text(content)
    
    assert len(chunks) > 0, "Should generate at least one chunk"
    
    # Check that headings are preserved
    heading_chunks = [c for c in chunks if c.meta.headings]
    assert len(heading_chunks) > 0, "Should have chunks with headings"
    
    print(f"✓ Generated {len(chunks)} chunks from simple markdown")


def test_code_blocks():
    """Test handling of code blocks."""
    print("Testing code blocks...")
    
    chunker = create_test_chunker()
    
    content = """
# Programming Example

Here's some Python code:

```python
def hello_world():
    print("Hello, World!")
    return True
```

And some JavaScript:

```javascript
function greet(name) {
    console.log(`Hello, ${name}!`);
}
```
"""
    
    chunks = chunker.chunk_text(content)
    
    # Check for code detection
    code_chunks = [c for c in chunks if c.meta.has_code]
    assert len(code_chunks) > 0, "Should detect code blocks"
    
    # Check that code content is preserved
    code_found = any("def hello_world" in chunk.text for chunk in chunks)
    assert code_found, "Should preserve code content"
    
    print(f"✓ Generated {len(chunks)} chunks, {len(code_chunks)} with code")


def test_tables():
    """Test handling of tables."""
    print("Testing tables...")
    
    chunker = create_test_chunker()
    
    content = """
# Data Overview

Here's a summary table:

| Name | Age | City |
|------|-----|------|
| Alice | 25 | New York |
| Bob | 30 | London |
| Charlie | 35 | Tokyo |

The table shows user information.
"""
    
    chunks = chunker.chunk_text(content)
    
    # Check for table detection
    table_chunks = [c for c in chunks if c.meta.has_table]
    assert len(table_chunks) > 0, "Should detect tables"
    
    # Check table content transformation
    table_content_found = any("Alice" in chunk.text and "Age" in chunk.text for chunk in chunks)
    assert table_content_found, "Should preserve table content"
    
    print(f"✓ Generated {len(chunks)} chunks, {len(table_chunks)} with tables")


def test_hierarchical_headings():
    """Test hierarchical heading preservation."""
    print("Testing hierarchical headings...")
    
    chunker = create_test_chunker()
    
    content = """
# Chapter 1

Introduction to the topic.

## Section 1.1

First section content.

### Subsection 1.1.1

Detailed information.

#### Sub-subsection

Very detailed information.

## Section 1.2

Second section content.

# Chapter 2

New chapter begins.
"""
    
    chunks = chunker.chunk_text(content)
    
    # Check heading hierarchy
    for chunk in chunks:
        if chunk.meta.headings:
            print(f"  Chunk headings: {chunk.meta.headings}")
            print(f"  Heading levels: {chunk.meta.heading_levels}")
    
    # Verify that deep headings include parent context
    deep_chunks = [c for c in chunks if c.meta.headings and len(c.meta.headings) > 2]
    if deep_chunks:
        assert "Chapter 1" in str(deep_chunks[0].meta.headings), "Should include parent heading context"
    
    print(f"✓ Generated {len(chunks)} chunks with hierarchical headings")


def test_large_document_splitting():
    """Test splitting of large documents."""
    print("Testing large document splitting...")
    
    chunker = create_test_chunker(max_tokens=200)  # Small chunks for testing
    
    # Create a large document
    content = "# Large Document\n\n"
    for i in range(50):
        content += f"This is paragraph {i+1}. " * 10 + "\n\n"
    
    chunks = chunker.chunk_text(content)
    
    # Should create multiple chunks
    assert len(chunks) > 1, "Should split large document into multiple chunks"
    
    # Check token limits
    for chunk in chunks:
        tokens = chunker._count_chunk_tokens(chunk)
        assert tokens <= chunker.tokenizer.max_tokens, f"Chunk has {tokens} tokens, exceeds limit"
    
    print(f"✓ Split large document into {len(chunks)} chunks")


def test_mixed_content():
    """Test handling of mixed content types.""" 
    print("Testing mixed content...")
    
    chunker = create_test_chunker()
    
    content = """
# Mixed Content Document

## Overview
This document contains various types of content.

### Code Section

```python
import numpy as np

def calculate_mean(data):
    return np.mean(data)
```

### Data Table

| Metric | Q1 | Q2 | Q3 | Q4 |
|--------|----|----|----|----|
| Revenue | 100K | 120K | 110K | 140K |
| Users | 1000 | 1200 | 1100 | 1400 |

### Links and Images

Check out [our website](https://example.com) for more info.

![Architecture Diagram](diagram.png)

### Lists

Features include:
- Real-time processing
- Scalable architecture  
- User-friendly interface
- Comprehensive analytics

## Conclusion

> The system provides a robust solution for data processing.

This concludes our overview.
"""
    
    chunks = chunker.chunk_text(content)
    
    # Check feature detection
    features_found = {
        "code": any(c.meta.has_code for c in chunks),
        "tables": any(c.meta.has_table for c in chunks),
        "images": any(c.meta.has_images for c in chunks),
        "links": any(c.meta.has_links for c in chunks),
    }
    
    print(f"  Features detected: {features_found}")
    assert features_found["code"], "Should detect code blocks"
    assert features_found["tables"], "Should detect tables"
    
    print(f"✓ Generated {len(chunks)} chunks from mixed content")


def test_chunker_configurations():
    """Test different chunker configurations."""
    print("Testing different configurations...")
    
    content = """
# Configuration Test

## Small Section
Short content here.

## Medium Section
This is a medium-length section with more detailed information that should be long enough to test different chunking strategies and see how they behave.

## Large Section
This is a very large section with lots of content that will definitely need to be split across multiple chunks when using smaller token limits. It contains detailed explanations, examples, and comprehensive coverage of the topic at hand.
"""
    
    # Test different configurations
    configs = [
        ("Small chunks", 100),
        ("Medium chunks", 300), 
        ("Large chunks", 800),
    ]
    
    for name, max_tokens in configs:
        chunker = create_test_chunker(max_tokens)
        chunks = chunker.chunk_text(content)
        
        avg_tokens = sum(chunker._count_chunk_tokens(c) for c in chunks) / len(chunks)
        print(f"  {name}: {len(chunks)} chunks, avg {avg_tokens:.1f} tokens")
        
        # Verify token limits
        for chunk in chunks:
            tokens = chunker._count_chunk_tokens(chunk)
            assert tokens <= max_tokens, f"Chunk exceeds {max_tokens} token limit"


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("Testing edge cases...")
    
    chunker = create_test_chunker()
    
    # Empty content
    chunks = chunker.chunk_text("")
    assert len(chunks) == 0, "Empty content should produce no chunks"
    
    # Only whitespace
    chunks = chunker.chunk_text("   \n\n   ")
    assert len(chunks) == 0, "Whitespace-only content should produce no chunks"
    
    # Single word
    chunks = chunker.chunk_text("Hello")
    assert len(chunks) == 1, "Single word should produce one chunk"
    assert chunks[0].text.strip() == "Hello", "Should preserve single word"
    
    # Very long single line
    long_line = "word " * 1000
    chunks = chunker.chunk_text(long_line)
    assert len(chunks) > 1, "Very long line should be split"
    
    print("✓ All edge cases handled correctly")


def test_chunk_properties():
    """Test chunk object properties and methods."""
    print("Testing chunk properties...")
    
    chunker = create_test_chunker()
    
    content = """
# Test Document

## Introduction
This is a test document with some content.

```python
print("Hello, World!")
```
"""
    
    chunks = chunker.chunk_text(content)
    
    for chunk in chunks:
        # Test basic properties
        assert isinstance(chunk, MarkdownChunk), "Should be MarkdownChunk instance"
        assert isinstance(chunk.meta, MarkdownMeta), "Should have MarkdownMeta"
        assert isinstance(chunk.text, str), "Should have text content"
        
        # Test methods
        contextualized = chunk.contextualize()
        assert isinstance(contextualized, str), "contextualize() should return string"
        
        chunk_dict = chunk.to_dict()
        assert isinstance(chunk_dict, dict), "to_dict() should return dictionary"
        assert "text" in chunk_dict, "Dictionary should contain 'text'"
        assert "meta" in chunk_dict, "Dictionary should contain 'meta'"
        assert "contextualized" in chunk_dict, "Dictionary should contain 'contextualized'"
    
    print("✓ All chunk properties working correctly")


def run_all_tests():
    """Run all tests."""
    print("Running HybridMarkdownChunker tests...\n")
    
    tests = [
        test_simple_markdown,
        test_code_blocks, 
        test_tables,
        test_hierarchical_headings,
        test_large_document_splitting,
        test_mixed_content,
        test_chunker_configurations,
        test_edge_cases,
        test_chunk_properties,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
        
        print()
    
    print("="*50)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)