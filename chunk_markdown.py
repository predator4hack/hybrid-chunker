#!/usr/bin/env python3
"""
Command-line tool for chunking markdown files using the HybridMarkdownChunker.

Usage:
    python chunk_markdown.py document.md
    python chunk_markdown.py document.md --tokenizer openai --max-tokens 1024 --output json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from chunker import HybridMarkdownChunker, MarkdownChunk
from chunker.config import ChunkerConfig, PresetConfigs
from chunker.tokenizer import create_tokenizer


def validate_file(file_path: str) -> Path:
    """Validate that the file exists and is readable."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    if not path.suffix.lower() in ['.md', '.markdown']:
        print(f"Warning: File '{file_path}' doesn't have a markdown extension (.md/.markdown)")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read(100)  # Read first 100 chars to test
        if not content.strip():
            raise ValueError(f"File appears to be empty: {file_path}")
    except UnicodeDecodeError:
        raise ValueError(f"File is not valid UTF-8 text: {file_path}")
    
    return path


def create_chunker_from_args(args) -> HybridMarkdownChunker:
    """Create a chunker instance from command line arguments."""
    
    # Use preset if specified
    if args.preset:
        preset_map = {
            'embedding': PresetConfigs.embedding_optimized,
            'llm': PresetConfigs.llm_optimized, 
            'memory': PresetConfigs.memory_efficient,
            'context': PresetConfigs.high_context
        }
        
        if args.preset not in preset_map:
            raise ValueError(f"Unknown preset: {args.preset}. Available: {list(preset_map.keys())}")
        
        config = preset_map[args.preset]()
        
        # Override with command line args if provided
        if args.max_tokens:
            config.max_tokens = args.max_tokens
        if args.tokenizer:
            config.tokenizer_type = args.tokenizer
        if args.model:
            config.model_name = args.model
    
    else:
        # Create config from individual arguments
        config = ChunkerConfig(
            tokenizer_type=args.tokenizer,
            model_name=args.model,
            max_tokens=args.max_tokens,
            merge_peers=args.merge_peers,
            delimiter=args.delimiter,
            overlap_tokens=args.overlap_tokens
        )
    
    try:
        tokenizer = config.create_tokenizer()
    except Exception as e:
        raise RuntimeError(f"Failed to create tokenizer: {e}")
    
    return HybridMarkdownChunker(
        tokenizer=tokenizer,
        merge_peers=config.merge_peers,
        delimiter=config.delimiter,
        overlap_tokens=config.overlap_tokens
    )


def analyze_chunks(chunks: List[MarkdownChunk], chunker: HybridMarkdownChunker) -> Dict[str, Any]:
    """Analyze chunk statistics and features."""
    if not chunks:
        return {
            "total_chunks": 0,
            "total_tokens": 0,
            "avg_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "features": {}
        }
    
    token_counts = [chunker._count_chunk_tokens(chunk) for chunk in chunks]
    
    # Feature analysis
    features = {
        "chunks_with_code": sum(1 for c in chunks if c.meta.has_code),
        "chunks_with_tables": sum(1 for c in chunks if c.meta.has_table),
        "chunks_with_images": sum(1 for c in chunks if c.meta.has_images),
        "chunks_with_links": sum(1 for c in chunks if c.meta.has_links),
        "unique_heading_levels": len(set(
            level for c in chunks if c.meta.heading_levels 
            for level in c.meta.heading_levels
        )),
        "max_heading_depth": max(
            (len(c.meta.headings) for c in chunks if c.meta.headings), 
            default=0
        )
    }
    
    return {
        "total_chunks": len(chunks),
        "total_tokens": sum(token_counts),
        "avg_tokens": sum(token_counts) / len(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "features": features
    }


def format_text_output(chunks: List[MarkdownChunk], chunker: HybridMarkdownChunker, 
                      analysis: Dict[str, Any], args) -> str:
    """Format chunks as human-readable text."""
    output = []
    
    # Header with summary
    output.append("=" * 60)
    output.append(f"MARKDOWN CHUNKING RESULTS")
    output.append("=" * 60)
    output.append(f"File: {args.file}")
    output.append(f"Tokenizer: {args.tokenizer} (max {chunker.tokenizer.max_tokens} tokens)")
    output.append(f"Total chunks: {analysis['total_chunks']}")
    output.append(f"Total tokens: {analysis['total_tokens']}")
    output.append(f"Average tokens per chunk: {analysis['avg_tokens']:.1f}")
    output.append(f"Token range: {analysis['min_tokens']} - {analysis['max_tokens']}")
    output.append("")
    
    # Feature summary
    features = analysis['features']
    if any(features.values()):
        output.append("Features detected:")
        if features['chunks_with_code']:
            output.append(f"  • Code blocks: {features['chunks_with_code']} chunks")
        if features['chunks_with_tables']:
            output.append(f"  • Tables: {features['chunks_with_tables']} chunks")
        if features['chunks_with_images']:
            output.append(f"  • Images: {features['chunks_with_images']} chunks")
        if features['chunks_with_links']:
            output.append(f"  • Links: {features['chunks_with_links']} chunks")
        output.append(f"  • Heading levels: {features['unique_heading_levels']}")
        output.append(f"  • Max heading depth: {features['max_heading_depth']}")
        output.append("")
    
    # Individual chunks
    output.append("CHUNKS:")
    output.append("-" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        tokens = chunker._count_chunk_tokens(chunk)
        output.append(f"Chunk {i} ({tokens} tokens):")
        
        # Heading hierarchy
        if chunk.meta.headings:
            hierarchy = " > ".join(chunk.meta.headings)
            output.append(f"  Headings: {hierarchy}")
        
        # Features
        features_list = []
        if chunk.meta.has_code: features_list.append("code")
        if chunk.meta.has_table: features_list.append("table") 
        if chunk.meta.has_images: features_list.append("images")
        if chunk.meta.has_links: features_list.append("links")
        
        if features_list:
            output.append(f"  Features: {', '.join(features_list)}")
        
        # Content preview or full text
        if args.verbose:
            output.append(f"  Content:")
            # Indent each line of content
            content_lines = chunk.text.split('\n')
            for line in content_lines:
                output.append(f"    {line}")
        else:
            preview = chunk.text.replace('\n', ' ')[:100]
            if len(chunk.text) > 100:
                preview += "..."
            output.append(f"  Preview: {preview}")
        
        if args.show_metadata:
            output.append(f"  Elements: {len(chunk.meta.elements)}")
            output.append(f"  Element types: {[e.type for e in chunk.meta.elements]}")
        
        output.append("")
    
    return "\n".join(output)


def format_json_output(chunks: List[MarkdownChunk], chunker: HybridMarkdownChunker,
                      analysis: Dict[str, Any], args) -> str:
    """Format chunks as JSON."""
    
    chunk_data = []
    for i, chunk in enumerate(chunks, 1):
        chunk_dict = chunk.to_dict()
        chunk_dict.update({
            "chunk_number": i,
            "token_count": chunker._count_chunk_tokens(chunk),
            "text_tokens": chunker._count_text_tokens(chunk.text),
            "context_tokens": chunker._count_context_tokens(chunk.meta.headings or [])
        })
        
        if not args.verbose:
            # Remove full text in non-verbose mode, keep preview
            chunk_dict["text_preview"] = chunk.text[:200] + ("..." if len(chunk.text) > 200 else "")
            if not args.show_metadata:
                del chunk_dict["text"]
        
        chunk_data.append(chunk_dict)
    
    output = {
        "file": str(args.file),
        "chunker_config": {
            "tokenizer_type": args.tokenizer,
            "max_tokens": chunker.tokenizer.max_tokens,
            "merge_peers": chunker.merge_peers,
            "delimiter": chunker.delimiter
        },
        "analysis": analysis,
        "chunks": chunk_data
    }
    
    return json.dumps(output, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Chunk markdown files using HybridMarkdownChunker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.md
  %(prog)s document.md --tokenizer openai --max-tokens 1024
  %(prog)s document.md --preset embedding --output json
  %(prog)s document.md --verbose --show-metadata
        """
    )
    
    # Required arguments
    parser.add_argument("file", help="Path to markdown file to chunk")
    
    # Tokenizer configuration
    parser.add_argument("--tokenizer", choices=["huggingface", "openai"], 
                       default="huggingface", help="Tokenizer type (default: huggingface)")
    parser.add_argument("--model", help="Model name for tokenizer")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens per chunk")
    
    # Preset configurations
    parser.add_argument("--preset", choices=["embedding", "llm", "memory", "context"],
                       help="Use preset configuration (overrides individual settings)")
    
    # Chunking options
    parser.add_argument("--no-merge", dest="merge_peers", action="store_false", 
                       default=True, help="Disable chunk merging")
    parser.add_argument("--delimiter", default="\\n", help="Text delimiter (default: \\n)")
    parser.add_argument("--overlap-tokens", type=int, default=50, 
                       help="Token overlap between chunks (default: 50)")
    
    # Output options
    parser.add_argument("--output", choices=["text", "json"], default="text",
                       help="Output format (default: text)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Show full chunk content")
    parser.add_argument("--show-metadata", action="store_true",
                       help="Include detailed metadata in output")
    parser.add_argument("--quiet", action="store_true", 
                       help="Suppress progress messages")
    
    args = parser.parse_args()
    
    try:
        # Validate input file
        if not args.quiet:
            print(f"Loading file: {args.file}")
        
        file_path = validate_file(args.file)
        
        # Create chunker
        if not args.quiet:
            print(f"Initializing {args.tokenizer} tokenizer...")
        
        chunker = create_chunker_from_args(args)
        
        # Process file
        if not args.quiet:
            print("Chunking document...")
            start_time = time.time()
        
        chunks = chunker.chunk_file(str(file_path))
        
        if not args.quiet:
            elapsed = time.time() - start_time
            print(f"Completed in {elapsed:.2f} seconds")
        
        # Analyze results
        analysis = analyze_chunks(chunks, chunker)
        
        # Format and output results
        if args.output == "json":
            output = format_json_output(chunks, chunker, analysis, args)
        else:
            output = format_text_output(chunks, chunker, analysis, args)
        
        print(output)
        
    except KeyboardInterrupt:
        print("\\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()