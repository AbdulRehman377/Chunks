"""
Enhanced Chunker for Azure Document Intelligence

Implements production-grade chunking strategy:
1. Dedicated table chunks with row-based formatting
2. Page-based splitting using <!-- PageBreak -->
3. Header prefixes for context
4. Deduplication via content hashing
5. Quality filters for noise removal

Based on production patterns from enterprise document processing.
"""

import json
import re
import hashlib
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict


@dataclass
class EnhancedChunk:
    """Represents a processed chunk with metadata."""
    content: str
    content_type: str  # "table", "text", "figure"
    page_number: Optional[int]
    section: Optional[str]
    metadata: Dict


class EnhancedChunker:
    """
    Production-grade chunker for Azure DI markdown output.
    
    Features:
    - Dedicated table extraction with header context
    - Page-based splitting
    - Content deduplication
    - Noise filtering
    - Contextual headers
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.min_chunk_length = self.config.get("min_chunk_length", 50)
        self.max_chunk_length = self.config.get("max_chunk_length", 4000)
        self.content_hash_length = self.config.get("content_hash_length", 100)
        self.use_one_shot = self.config.get("use_one_shot", False)
        
        # Noise patterns to filter out
        self.noise_patterns = [
            r'^Page \d+ of \d+$',
            r'^:selected:$',
            r'^\s*$',
            r'^F\d{3,4}\s+\d+\s+\d+$',  # Footer codes like "F014 11 16"
        ]
        
        # Track seen content for deduplication
        self.seen_hashes = set()
    
    def load_raw_ocr(self, file_path: str) -> Dict:
        """Load the raw Azure DI JSON output."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def extract_chunks(self, raw_ocr: Dict, filename: str = "document") -> List[EnhancedChunk]:
        """
        Main entry point: Extract enhanced chunks from Azure DI output.
        
        Args:
            raw_ocr: The raw JSON from Azure DI
            filename: Source filename for metadata
            
        Returns:
            List of EnhancedChunk objects
        """
        analyze_result = raw_ocr.get("analyzeResult", raw_ocr)
        
        content = analyze_result.get("content", "")
        content_format = analyze_result.get("contentFormat", "text")
        tables = analyze_result.get("tables", [])
        pages = analyze_result.get("pages", [])
        figures = analyze_result.get("figures", [])
        
        print(f"\n{'='*60}")
        print(f"📄 ENHANCED CHUNKER")
        print(f"{'='*60}")
        print(f"Content format: {content_format}")
        print(f"Content length: {len(content):,} chars")
        print(f"Tables: {len(tables)}")
        print(f"Pages: {len(pages)}")
        print(f"Figures: {len(figures)}")
        
        chunks = []
        self.seen_hashes.clear()
        
        # === STEP 1: Extract dedicated table chunks ===
        print(f"\n📊 Extracting table chunks...")
        table_chunks = self._extract_table_chunks(tables, filename, pages)
        chunks.extend(table_chunks)
        print(f"   Created {len(table_chunks)} table chunks")
        
        # === STEP 2: Process remaining content ===
        if self.use_one_shot:
            print(f"\n📝 ONE-SHOT MODE: Processing as single document...")
            text_chunks = self._process_one_shot(content, filename, pages)
        else:
            print(f"\n📝 DETAILED MODE: Splitting by pages and sections...")
            text_chunks = self._process_by_pages(content, filename, pages)
        
        chunks.extend(text_chunks)
        print(f"   Created {len(text_chunks)} text chunks")
        
        # === STEP 3: Extract figure chunks ===
        print(f"\n🖼️  Extracting figure chunks...")
        figure_chunks = self._extract_figure_chunks(figures, content, filename)
        chunks.extend(figure_chunks)
        print(f"   Created {len(figure_chunks)} figure chunks")
        
        # === STEP 4: Final deduplication and quality check ===
        print(f"\n🔍 Running quality filters...")
        final_chunks = self._filter_chunks(chunks)
        print(f"   Final chunks after filtering: {len(final_chunks)}")
        
        print(f"\n✅ Total chunks created: {len(final_chunks)}")
        
        return final_chunks
    
    def _extract_table_chunks(self, tables: List[Dict], filename: str, pages: List[Dict]) -> List[EnhancedChunk]:
        """
        Extract dedicated chunks for each table with row-based formatting.
        
        Creates chunks like:
        "IMPORT CUSTOMS BROKER: BOLLORE LOGISTICS | WEIGHT: 966 LB | PACKAGES: 1 PLT"
        """
        chunks = []
        
        for table_idx, table in enumerate(tables):
            row_count = table.get("rowCount", 0)
            col_count = table.get("columnCount", 0)
            cells = table.get("cells", [])
            
            if not cells:
                continue
            
            # Get page number
            bounding_regions = table.get("boundingRegions", [])
            page_num = bounding_regions[0].get("pageNumber", 1) if bounding_regions else 1
            
            # Build grid
            grid = [[None for _ in range(col_count)] for _ in range(row_count)]
            headers = [None] * col_count
            
            for cell in cells:
                row_idx = cell.get("rowIndex", 0)
                col_idx = cell.get("columnIndex", 0)
                content = cell.get("content", "").strip()
                kind = cell.get("kind", "")
                
                if row_idx < row_count and col_idx < col_count:
                    grid[row_idx][col_idx] = content
                    
                    # Track column headers
                    if kind == "columnHeader" or row_idx == 0:
                        headers[col_idx] = content
            
            # === Option A: Create one chunk per table (markdown format) ===
            table_markdown = self._table_to_markdown(grid, headers)
            if table_markdown and len(table_markdown) > 20:
                header = self._build_header(
                    filename=filename,
                    section=f"Table {table_idx + 1}",
                    page=page_num
                )
                chunks.append(EnhancedChunk(
                    content=header + table_markdown,
                    content_type="table",
                    page_number=page_num,
                    section=f"Table {table_idx + 1}",
                    metadata={
                        "table_index": table_idx,
                        "row_count": row_count,
                        "column_count": col_count,
                        "headers": [h for h in headers if h]
                    }
                ))
            
            # === Option B: Create row-centric chunks (for better retrieval) ===
            for row_idx in range(1, row_count):  # Skip header row
                row_parts = []
                for col_idx in range(col_count):
                    cell_value = grid[row_idx][col_idx]
                    col_header = headers[col_idx]
                    
                    if cell_value:
                        if col_header and col_header != cell_value:
                            row_parts.append(f"{col_header}: {cell_value}")
                        else:
                            row_parts.append(cell_value)
                
                if row_parts:
                    row_content = " | ".join(row_parts)
                    if len(row_content) > 20:
                        header = self._build_header(
                            filename=filename,
                            section=f"Table {table_idx + 1}, Row {row_idx + 1}",
                            page=page_num
                        )
                        chunks.append(EnhancedChunk(
                            content=header + row_content,
                            content_type="table_row",
                            page_number=page_num,
                            section=f"Table {table_idx + 1}",
                            metadata={
                                "table_index": table_idx,
                                "row_index": row_idx,
                                "headers": [h for h in headers if h]
                            }
                        ))
        
        return chunks
    
    def _table_to_markdown(self, grid: List[List], headers: List) -> str:
        """Convert table grid to markdown format."""
        if not grid or not grid[0]:
            return ""
        
        lines = []
        
        # Header row
        header_row = " | ".join(h or "" for h in headers)
        lines.append(f"| {header_row} |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        # Data rows
        for row in grid[1:]:
            row_content = " | ".join(cell or "" for cell in row)
            lines.append(f"| {row_content} |")
        
        return "\n".join(lines)
    
    def _process_one_shot(self, content: str, filename: str, pages: List[Dict]) -> List[EnhancedChunk]:
        """
        One-shot mode: Keep markdown as minimal chunks, split only by page breaks.
        """
        chunks = []
        
        # Remove table HTML blocks (already extracted separately)
        clean_content = self._remove_tables_from_content(content)
        
        if not clean_content or len(clean_content) < self.min_chunk_length:
            return chunks
        
        # For small documents, create single chunk
        if len(clean_content) <= self.max_chunk_length:
            header = self._build_header(
                filename=filename,
                section="Document",
                page=1
            )
            chunks.append(EnhancedChunk(
                content=header + clean_content,
                content_type="text",
                page_number=1,
                section="Document",
                metadata={"total_pages": len(pages)}
            ))
            return chunks
        
        # Split by page breaks
        parts = clean_content.split("<!-- PageBreak -->") if "<!-- PageBreak -->" in clean_content else [clean_content]
        
        for idx, part in enumerate(parts):
            trimmed = part.strip()
            if len(trimmed) >= self.min_chunk_length:
                header = self._build_header(
                    filename=filename,
                    section=f"Page {idx + 1}",
                    page=idx + 1
                )
                chunks.append(EnhancedChunk(
                    content=header + trimmed,
                    content_type="text",
                    page_number=idx + 1,
                    section=f"Page {idx + 1}",
                    metadata={}
                ))
        
        return chunks
    
    def _process_by_pages(self, content: str, filename: str, pages: List[Dict]) -> List[EnhancedChunk]:
        """
        Detailed mode: Split by pages, then by sections within pages.
        """
        chunks = []
        
        # Remove table HTML blocks
        clean_content = self._remove_tables_from_content(content)
        
        # Split by page breaks
        page_contents = clean_content.split("<!-- PageBreak -->") if "<!-- PageBreak -->" in clean_content else [clean_content]
        
        for page_idx, page_content in enumerate(page_contents):
            page_num = page_idx + 1
            
            # Split by headings within each page
            sections = self._split_by_headings(page_content)
            
            for section_title, section_content in sections:
                trimmed = section_content.strip()
                
                # Skip noise
                if self._is_noise(trimmed):
                    continue
                
                if len(trimmed) >= self.min_chunk_length:
                    header = self._build_header(
                        filename=filename,
                        section=section_title or f"Page {page_num}",
                        page=page_num
                    )
                    chunks.append(EnhancedChunk(
                        content=header + trimmed,
                        content_type="text",
                        page_number=page_num,
                        section=section_title,
                        metadata={}
                    ))
        
        return chunks
    
    def _split_by_headings(self, content: str) -> List[Tuple[str, str]]:
        """Split content by markdown headings (#, ##, etc.)."""
        # Pattern to match markdown headings
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        
        sections = []
        current_title = None
        current_content = []
        
        for line in content.split('\n'):
            match = re.match(heading_pattern, line)
            if match:
                # Save previous section
                if current_content:
                    sections.append((current_title, '\n'.join(current_content)))
                
                current_title = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_content:
            sections.append((current_title, '\n'.join(current_content)))
        
        return sections if sections else [(None, content)]
    
    def _remove_tables_from_content(self, content: str) -> str:
        """Remove HTML table blocks from markdown content."""
        # Remove <table>...</table> blocks
        clean = re.sub(r'<table>.*?</table>', '[TABLE_EXTRACTED]', content, flags=re.DOTALL)
        # Remove the placeholder
        clean = clean.replace('[TABLE_EXTRACTED]', '')
        return clean
    
    def _extract_figure_chunks(self, figures: List[Dict], content: str, filename: str) -> List[EnhancedChunk]:
        """Extract chunks for figures/images."""
        chunks = []
        
        for fig_idx, figure in enumerate(figures):
            bounding_regions = figure.get("boundingRegions", [])
            page_num = bounding_regions[0].get("pageNumber", 1) if bounding_regions else 1
            
            # Get caption if available
            caption = ""
            elements = figure.get("elements", [])
            
            # Try to extract text near the figure
            spans = figure.get("spans", [])
            if spans and content:
                for span in spans:
                    offset = span.get("offset", 0)
                    length = span.get("length", 0)
                    caption = content[offset:offset + length].strip()
            
            if caption and len(caption) > 10:
                header = self._build_header(
                    filename=filename,
                    section=f"Figure {fig_idx + 1}",
                    page=page_num
                )
                chunks.append(EnhancedChunk(
                    content=header + f"Figure: {caption}",
                    content_type="figure",
                    page_number=page_num,
                    section=f"Figure {fig_idx + 1}",
                    metadata={"figure_id": figure.get("id")}
                ))
        
        return chunks
    
    def _build_header(self, filename: str, section: str, page: int) -> str:
        """Build a contextual header prefix for chunks."""
        return f"[Source: {filename} | {section} | Page {page}]\n\n"
    
    def _is_noise(self, text: str) -> bool:
        """Check if text matches noise patterns."""
        for pattern in self.noise_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
        return False
    
    def _get_content_hash(self, text: str) -> str:
        """Generate hash for deduplication."""
        normalized = text[:self.content_hash_length].lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _filter_chunks(self, chunks: List[EnhancedChunk]) -> List[EnhancedChunk]:
        """Apply final quality filters and deduplication."""
        filtered = []
        
        for chunk in chunks:
            # Skip if too short
            if len(chunk.content) < self.min_chunk_length:
                continue
            
            # Skip noise
            if self._is_noise(chunk.content):
                continue
            
            # Deduplication
            content_hash = self._get_content_hash(chunk.content)
            if content_hash in self.seen_hashes:
                continue
            self.seen_hashes.add(content_hash)
            
            filtered.append(chunk)
        
        return filtered
    
    def to_vectordb_format(self, chunks: List[EnhancedChunk]) -> List[Dict]:
        """Convert chunks to format ready for vector DB storage."""
        return [
            {
                "text": chunk.content,
                "metadata": {
                    "content_type": chunk.content_type,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    **chunk.metadata
                }
            }
            for chunk in chunks
        ]
    
    def save_chunks(self, chunks: List[EnhancedChunk], output_path: str):
        """Save chunks to JSON file."""
        data = {
            "total_chunks": len(chunks),
            "chunks": self.to_vectordb_format(chunks)
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved {len(chunks)} chunks to {output_path}")


def main():
    """Main function to demonstrate the enhanced chunker."""
    
    # Initialize chunker
    chunker = EnhancedChunker(config={
        "min_chunk_length": 50,
        "max_chunk_length": 4000,
        "use_one_shot": False,  # Use detailed mode for better granularity
    })
    
    # Load raw OCR output
    raw_ocr = chunker.load_raw_ocr("RAW_OCR.json")
    
    # Extract chunks
    chunks = chunker.extract_chunks(raw_ocr, filename="TAX_INVOICE.PDF")
    
    # Save to file
    chunker.save_chunks(chunks, "ENHANCED_CHUNKS.json")
    
    # Print sample chunks
    print(f"\n{'='*60}")
    print("📋 SAMPLE CHUNKS")
    print(f"{'='*60}")
    
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ({chunk.content_type}) ---")
        print(f"Page: {chunk.page_number}")
        print(f"Section: {chunk.section}")
        preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        print(f"Content:\n{preview}")


if __name__ == "__main__":
    main()

