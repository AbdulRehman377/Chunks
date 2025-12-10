"""
Azure Document Intelligence → Vector DB Processor

This script processes the raw Azure OCR output and creates deduplicated,
metadata-rich chunks suitable for vector database storage.

Key features:
- No duplication: Each text unit is stored once
- Rich metadata: Page, position, role, table info preserved
- Queryable: Can retrieve text by page, table cell, or semantic role
"""

import json
from typing import Optional


def load_raw_ocr(file_path: str) -> dict:
    """Load the raw Azure OCR JSON output."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_for_vectordb(raw_ocr: dict) -> dict:
    """
    Process Azure OCR output into a deduplicated structure for vector DB storage.
    
    Returns a dict with:
    - document_metadata: Overall document info
    - chunks: List of text chunks with metadata (for vector embedding)
    - tables: Structured table data (optional: for structured queries)
    """
    analyze_result = raw_ocr.get("analyzeResult", raw_ocr)
    
    # 1. Document-level metadata
    document_metadata = {
        "api_version": analyze_result.get("apiVersion"),
        "model_id": analyze_result.get("modelId"),
        "content_format": analyze_result.get("contentFormat"),
        "total_pages": len(analyze_result.get("pages", [])),
        "created_at": raw_ocr.get("createdDateTime"),
        "full_content": analyze_result.get("content"),  # Keep original for reference
    }
    
    # 2. Create paragraph-based chunks with metadata
    chunks = []
    paragraphs = analyze_result.get("paragraphs", [])
    
    # Build a span-to-table-cell mapping for paragraphs that belong to tables
    paragraph_table_mapping = build_paragraph_table_mapping(analyze_result)
    
    for idx, para in enumerate(paragraphs):
        chunk = {
            "id": f"paragraph_{idx}",
            "type": "paragraph",
            "content": para.get("content", ""),
            "metadata": {
                "paragraph_index": idx,
                "role": para.get("role"),  # title, pageNumber, sectionHeading, etc.
                "spans": para.get("spans", []),  # offset/length for exact content location
            }
        }
        
        # Add page and position info from boundingRegions
        bounding_regions = para.get("boundingRegions", [])
        if bounding_regions:
            chunk["metadata"]["page_number"] = bounding_regions[0].get("pageNumber")
            chunk["metadata"]["bounding_box"] = bounding_regions[0].get("polygon")
        
        # Add table context if this paragraph is part of a table
        if idx in paragraph_table_mapping:
            table_info = paragraph_table_mapping[idx]
            chunk["metadata"]["table_context"] = table_info
            chunk["type"] = "table_cell"
        
        chunks.append(chunk)
    
    # 3. Process tables separately (for structured queries)
    tables = process_tables(analyze_result)
    
    # 4. Process figures
    figures = process_figures(analyze_result)
    
    return {
        "document_metadata": document_metadata,
        "chunks": chunks,
        "tables": tables,
        "figures": figures,
    }


def build_paragraph_table_mapping(analyze_result: dict) -> dict:
    """
    Build a mapping from paragraph index to table cell info.
    This helps identify which paragraphs belong to which table cells.
    """
    mapping = {}
    tables = analyze_result.get("tables", [])
    
    for table_idx, table in enumerate(tables):
        for cell in table.get("cells", []):
            elements = cell.get("elements", [])
            for element in elements:
                # Elements are like "/paragraphs/18"
                if element.startswith("/paragraphs/"):
                    para_idx = int(element.split("/")[-1])
                    mapping[para_idx] = {
                        "table_index": table_idx,
                        "row_index": cell.get("rowIndex"),
                        "column_index": cell.get("columnIndex"),
                        "cell_kind": cell.get("kind"),  # columnHeader, rowHeader, or None
                        "row_span": cell.get("rowSpan", 1),
                        "column_span": cell.get("columnSpan", 1),
                    }
    return mapping


def process_tables(analyze_result: dict) -> list:
    """
    Process tables into a structured format.
    This can be stored separately for structured table queries.
    """
    tables_data = []
    tables = analyze_result.get("tables", [])
    
    for table_idx, table in enumerate(tables):
        bounding_regions = table.get("boundingRegions", [])
        
        table_data = {
            "id": f"table_{table_idx}",
            "row_count": table.get("rowCount"),
            "column_count": table.get("columnCount"),
            "page_number": bounding_regions[0].get("pageNumber") if bounding_regions else None,
            "cells": [],
        }
        
        for cell in table.get("cells", []):
            cell_data = {
                "row": cell.get("rowIndex"),
                "column": cell.get("columnIndex"),
                "content": cell.get("content", ""),
                "kind": cell.get("kind"),  # columnHeader, rowHeader
                "row_span": cell.get("rowSpan", 1),
                "column_span": cell.get("columnSpan", 1),
            }
            table_data["cells"].append(cell_data)
        
        # Also create a markdown representation for embedding
        table_data["markdown"] = table_to_markdown(table_data)
        
        tables_data.append(table_data)
    
    return tables_data


def table_to_markdown(table_data: dict) -> str:
    """Convert table data to markdown format for embedding."""
    rows = table_data["row_count"]
    cols = table_data["column_count"]
    
    # Initialize grid
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    
    # Fill grid with cell content
    for cell in table_data["cells"]:
        row, col = cell["row"], cell["column"]
        if 0 <= row < rows and 0 <= col < cols:
            grid[row][col] = cell["content"]
    
    # Convert to markdown
    lines = []
    for i, row in enumerate(grid):
        line = "| " + " | ".join(row) + " |"
        lines.append(line)
        if i == 0:
            lines.append("|" + "|".join(["---"] * cols) + "|")
    
    return "\n".join(lines)


def process_figures(analyze_result: dict) -> list:
    """Process figures/images metadata."""
    figures_data = []
    figures = analyze_result.get("figures", [])
    
    for figure in figures:
        bounding_regions = figure.get("boundingRegions", [])
        
        figure_data = {
            "id": figure.get("id"),
            "page_number": bounding_regions[0].get("pageNumber") if bounding_regions else None,
            "bounding_box": bounding_regions[0].get("polygon") if bounding_regions else None,
            "spans": figure.get("spans", []),
            "elements": figure.get("elements", []),
        }
        figures_data.append(figure_data)
    
    return figures_data


def create_vectordb_chunks(processed_data: dict, 
                          include_tables_as_chunks: bool = True,
                          chunk_size: int = 1000) -> list:
    """
    Create final chunks ready for vector database insertion.
    
    Each chunk includes:
    - text: The content to embed
    - metadata: All associated metadata for filtering/retrieval
    """
    vectordb_chunks = []
    
    # Add paragraph chunks
    for chunk in processed_data["chunks"]:
        vectordb_chunk = {
            "text": chunk["content"],
            "metadata": {
                "chunk_id": chunk["id"],
                "chunk_type": chunk["type"],
                "page_number": chunk["metadata"].get("page_number"),
                "role": chunk["metadata"].get("role"),
                "bounding_box": chunk["metadata"].get("bounding_box"),
                "span_offset": chunk["metadata"]["spans"][0]["offset"] if chunk["metadata"].get("spans") else None,
                "span_length": chunk["metadata"]["spans"][0]["length"] if chunk["metadata"].get("spans") else None,
            }
        }
        
        # Add table context if present
        if "table_context" in chunk["metadata"]:
            tc = chunk["metadata"]["table_context"]
            vectordb_chunk["metadata"]["table_index"] = tc.get("table_index")
            vectordb_chunk["metadata"]["table_row"] = tc.get("row_index")
            vectordb_chunk["metadata"]["table_column"] = tc.get("column_index")
            vectordb_chunk["metadata"]["cell_kind"] = tc.get("cell_kind")
        
        vectordb_chunks.append(vectordb_chunk)
    
    # Optionally add tables as complete chunks (useful for table-specific queries)
    if include_tables_as_chunks:
        for table in processed_data["tables"]:
            vectordb_chunk = {
                "text": table["markdown"],
                "metadata": {
                    "chunk_id": table["id"],
                    "chunk_type": "table",
                    "page_number": table["page_number"],
                    "table_row_count": table["row_count"],
                    "table_column_count": table["column_count"],
                }
            }
            vectordb_chunks.append(vectordb_chunk)
    
    return vectordb_chunks


def save_processed_output(processed_data: dict, output_path: str):
    """Save processed data to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)


def main():
    # Load raw OCR output
    raw_ocr = load_raw_ocr("RAW_OCR.json")
    
    # Process for vector DB
    processed_data = process_for_vectordb(raw_ocr)
    
    # Create vector DB ready chunks
    vectordb_chunks = create_vectordb_chunks(processed_data)
    
    # Print statistics
    print("\n" + "="*60)
    print("📊 Processing Complete!")
    print("="*60)
    print(f"📄 Total Pages: {processed_data['document_metadata']['total_pages']}")
    print(f"📝 Paragraph Chunks: {len(processed_data['chunks'])}")
    print(f"📋 Tables: {len(processed_data['tables'])}")
    print(f"🖼️  Figures: {len(processed_data['figures'])}")
    print(f"🔢 Total Vector DB Chunks: {len(vectordb_chunks)}")
    
    # Calculate storage savings
    original_size = len(json.dumps(raw_ocr))
    processed_size = len(json.dumps({"chunks": vectordb_chunks}))
    savings = (1 - processed_size / original_size) * 100
    print(f"\n💾 Storage Comparison:")
    print(f"   Original JSON: {original_size:,} bytes")
    print(f"   Processed: {processed_size:,} bytes")
    print(f"   Savings: ~{savings:.1f}%")
    
    # Save outputs
    save_processed_output(processed_data, "PROCESSED_OCR.json")
    
    vectordb_output = {
        "document_metadata": processed_data["document_metadata"],
        "chunks": vectordb_chunks
    }
    save_processed_output(vectordb_output, "VECTORDB_READY.json")
    
    print(f"\n✅ Saved: PROCESSED_OCR.json (full processed data)")
    print(f"✅ Saved: VECTORDB_READY.json (ready for vector DB)")
    
    # Show sample chunks
    print("\n" + "="*60)
    print("📋 Sample Chunks:")
    print("="*60)
    
    for i, chunk in enumerate(vectordb_chunks[:5]):
        print(f"\n--- Chunk {i+1} ({chunk['metadata']['chunk_type']}) ---")
        print(f"Text: {chunk['text'][:100]}..." if len(chunk['text']) > 100 else f"Text: {chunk['text']}")
        print(f"Page: {chunk['metadata'].get('page_number')}")
        print(f"Role: {chunk['metadata'].get('role')}")
        if chunk['metadata'].get('table_index') is not None:
            print(f"Table: {chunk['metadata'].get('table_index')}, "
                  f"Row: {chunk['metadata'].get('table_row')}, "
                  f"Col: {chunk['metadata'].get('table_column')}")


if __name__ == "__main__":
    main()

