"""Markdown parsing utilities for the hybrid chunker."""

import re
from typing import List, Dict, Any, Optional, Tuple
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

from .chunk import MarkdownElement


class MarkdownParser:
    """Parser for markdown content with hierarchical structure awareness."""
    
    def __init__(self):
        self.md = MarkdownIt("commonmark", {"breaks": True, "html": True})
        self.md.enable(["table", "strikethrough", "linkify"])
    
    def parse(self, content: str) -> List[MarkdownElement]:
        """Parse markdown content into structured elements."""
        tokens = self.md.parse(content)
        elements = []
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            element = None
            
            if token.type == "heading_open":
                element, i = self._parse_heading(tokens, i)
            elif token.type == "paragraph_open":
                element, i = self._parse_paragraph(tokens, i)
            elif token.type == "bullet_list_open" or token.type == "ordered_list_open":
                element, i = self._parse_list(tokens, i)
            elif token.type == "table_open":
                element, i = self._parse_table(tokens, i)
            elif token.type == "fence" or token.type == "code_block":
                element = self._parse_code_block(token)
            elif token.type == "blockquote_open":
                element, i = self._parse_blockquote(tokens, i)
            elif token.type == "hr":
                element = MarkdownElement(type="horizontal_rule", content="---")
            
            if element:
                elements.append(element)
            
            i += 1
        
        return elements
    
    def _parse_heading(self, tokens: List[Any], start: int) -> Tuple[MarkdownElement, int]:
        """Parse a heading element."""
        heading_open = tokens[start]
        level = int(heading_open.tag[1])  # h1 -> 1, h2 -> 2, etc.
        
        content_parts = []
        i = start + 1
        
        while i < len(tokens) and tokens[i].type != "heading_close":
            if tokens[i].type == "inline":
                content_parts.append(tokens[i].content)
            i += 1
        
        content = "".join(content_parts).strip()
        
        element = MarkdownElement(
            type="heading",
            content=content,
            level=level,
            properties={"tag": heading_open.tag}
        )
        
        return element, i
    
    def _parse_paragraph(self, tokens: List[Any], start: int) -> Tuple[MarkdownElement, int]:
        """Parse a paragraph element."""
        content_parts = []
        has_images = False
        has_links = False
        
        i = start + 1
        while i < len(tokens) and tokens[i].type != "paragraph_close":
            if tokens[i].type == "inline":
                content = tokens[i].content
                content_parts.append(content)
                
                # Check for images and links
                if "![" in content:
                    has_images = True
                if "](" in content:
                    has_links = True
            i += 1
        
        content = "\n".join(content_parts).strip()
        
        element = MarkdownElement(
            type="paragraph",
            content=content,
            properties={
                "has_images": has_images,
                "has_links": has_links
            }
        )
        
        return element, i
    
    def _parse_list(self, tokens: List[Any], start: int) -> Tuple[MarkdownElement, int]:
        """Parse a list element."""
        list_open = tokens[start]
        list_type = "ordered" if list_open.type == "ordered_list_open" else "bullet"
        
        items = []
        i = start + 1
        
        while i < len(tokens) and tokens[i].type not in ["bullet_list_close", "ordered_list_close"]:
            if tokens[i].type == "list_item_open":
                item_content, i = self._parse_list_item(tokens, i)
                items.append(item_content)
            i += 1
        
        content = self._format_list_content(items, list_type)
        
        element = MarkdownElement(
            type="list",
            content=content,
            properties={
                "list_type": list_type,
                "items": items
            }
        )
        
        return element, i
    
    def _parse_list_item(self, tokens: List[Any], start: int) -> Tuple[str, int]:
        """Parse a single list item."""
        content_parts = []
        i = start + 1
        
        while i < len(tokens) and tokens[i].type != "list_item_close":
            if tokens[i].type == "paragraph_open":
                para_content, i = self._parse_paragraph(tokens, i)
                content_parts.append(para_content.content)
            elif tokens[i].type == "inline":
                content_parts.append(tokens[i].content)
            i += 1
        
        return " ".join(content_parts).strip(), i
    
    def _format_list_content(self, items: List[str], list_type: str) -> str:
        """Format list items into text content."""
        formatted_items = []
        for i, item in enumerate(items):
            if list_type == "ordered":
                formatted_items.append(f"{i+1}. {item}")
            else:
                formatted_items.append(f"- {item}")
        return "\n".join(formatted_items)
    
    def _parse_table(self, tokens: List[Any], start: int) -> Tuple[MarkdownElement, int]:
        """Parse a table element."""
        rows = []
        i = start + 1
        
        while i < len(tokens) and tokens[i].type != "table_close":
            if tokens[i].type == "thead_open" or tokens[i].type == "tbody_open":
                i += 1
                continue
            elif tokens[i].type == "tr_open":
                row_content, i = self._parse_table_row(tokens, i)
                rows.append(row_content)
            i += 1
        
        content = self._format_table_content(rows)
        
        element = MarkdownElement(
            type="table",
            content=content,
            properties={"rows": rows}
        )
        
        return element, i
    
    def _parse_table_row(self, tokens: List[Any], start: int) -> Tuple[List[str], int]:
        """Parse a table row."""
        cells = []
        i = start + 1
        
        while i < len(tokens) and tokens[i].type != "tr_close":
            if tokens[i].type in ["th_open", "td_open"]:
                cell_content, i = self._parse_table_cell(tokens, i)
                cells.append(cell_content)
            i += 1
        
        return cells, i
    
    def _parse_table_cell(self, tokens: List[Any], start: int) -> Tuple[str, int]:
        """Parse a table cell."""
        cell_open = tokens[start]
        content_parts = []
        i = start + 1
        
        while i < len(tokens) and tokens[i].type not in ["th_close", "td_close"]:
            if tokens[i].type == "inline":
                content_parts.append(tokens[i].content)
            i += 1
        
        return " ".join(content_parts).strip(), i
    
    def _format_table_content(self, rows: List[List[str]]) -> str:
        """Format table rows into text content."""
        if not rows:
            return ""
        
        # Convert to triplet format similar to docling-core
        triplets = []
        if len(rows) > 1:  # Has header
            headers = rows[0]
            for data_row in rows[1:]:
                for i, cell in enumerate(data_row):
                    if i < len(headers) and cell.strip():
                        triplets.append(f"{headers[i]}: {cell}")
        
        return "\n".join(triplets) if triplets else "\n".join([" | ".join(row) for row in rows])
    
    def _parse_code_block(self, token: Any) -> MarkdownElement:
        """Parse a code block element."""
        content = token.content.rstrip()
        language = getattr(token, 'info', '').strip()
        
        element = MarkdownElement(
            type="code_block",
            content=content,
            language=language if language else None,
            properties={"language": language}
        )
        
        return element
    
    def _parse_blockquote(self, tokens: List[Any], start: int) -> Tuple[MarkdownElement, int]:
        """Parse a blockquote element."""
        content_parts = []
        i = start + 1
        
        while i < len(tokens) and tokens[i].type != "blockquote_close":
            if tokens[i].type == "paragraph_open":
                para_content, i = self._parse_paragraph(tokens, i)
                content_parts.append(para_content.content)
            i += 1
        
        content = "\n".join(content_parts).strip()
        
        element = MarkdownElement(
            type="blockquote",
            content=content
        )
        
        return element, i
    
    def extract_headings_hierarchy(self, elements: List[MarkdownElement]) -> List[Tuple[int, str]]:
        """Extract heading hierarchy from elements."""
        headings = []
        for element in elements:
            if element.type == "heading":
                headings.append((element.level, element.content))
        return headings
    
    def get_markdown_features(self, elements: List[MarkdownElement]) -> Dict[str, bool]:
        """Analyze which markdown features are present."""
        features = {
            "has_headings": False,
            "has_code": False,
            "has_tables": False,
            "has_lists": False,
            "has_images": False,
            "has_links": False,
            "has_blockquotes": False
        }
        
        for element in elements:
            if element.type == "heading":
                features["has_headings"] = True
            elif element.type == "code_block":
                features["has_code"] = True
            elif element.type == "table":
                features["has_tables"] = True
            elif element.type == "list":
                features["has_lists"] = True
            elif element.type == "blockquote":
                features["has_blockquotes"] = True
            elif element.type == "paragraph":
                if element.properties.get("has_images"):
                    features["has_images"] = True
                if element.properties.get("has_links"):
                    features["has_links"] = True
        
        return features