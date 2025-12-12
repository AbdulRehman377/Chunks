"""
Spatial Pairing Algorithm for Azure Document Intelligence

Uses bounding box coordinates to:
1. Detect column layouts based on X positions
2. Pair labels with their values based on spatial proximity
3. Clean and restructure text content

GENERIC - Works with any PDF document type.
No hardcoded keywords or document-specific logic.
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Paragraph:
    """Represents a paragraph with spatial information."""
    content: str
    page_number: int
    x_left: float
    x_right: float
    y_top: float
    y_bottom: float
    role: Optional[str] = None
    
    @property
    def x_center(self) -> float:
        return (self.x_left + self.x_right) / 2
    
    @property
    def width(self) -> float:
        return self.x_right - self.x_left
    
    @property
    def y_center(self) -> float:
        return (self.y_top + self.y_bottom) / 2


class SpatialPairing:
    """
    Pairs labels with values based on spatial proximity (column detection).
    
    GENERIC Algorithm (no hardcoded keywords):
    1. Extract paragraphs with bounding boxes
    2. Group paragraphs by page
    3. Detect column structure based on X position clustering
    4. Identify labels using generic heuristics (short, uppercase, etc.)
    5. Pair labels with values in the same column based on Y proximity
    6. Output restructured text
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Column detection threshold (X positions within this range = same column)
        self.column_threshold = self.config.get("column_threshold", 0.5)
        
        # Vertical proximity threshold (Y distance for label-value pairing)
        self.vertical_threshold = self.config.get("vertical_threshold", 0.2)
        
        # Generic label detection settings
        self.max_label_length = self.config.get("max_label_length", 25)
        self.max_label_words = self.config.get("max_label_words", 3)
        
        # Page width for column detection (standard letter = 8.5")
        self.page_width = self.config.get("page_width", 8.5)
    
    def _get_table_spans(self, raw_ocr: Dict) -> set:
        """
        Get all text spans that belong to table cells.
        
        Returns: Set of (start_offset, end_offset) tuples
        """
        analyze_result = raw_ocr.get("analyzeResult", raw_ocr)
        tables = analyze_result.get("tables", [])
        
        table_spans = set()
        for table in tables:
            for cell in table.get("cells", []):
                for span in cell.get("spans", []):
                    offset = span.get("offset", 0)
                    length = span.get("length", 0)
                    table_spans.add((offset, offset + length))
        
        return table_spans
    
    def _is_paragraph_in_table(self, para: Dict, table_spans: set) -> bool:
        """Check if a paragraph overlaps with any table span."""
        para_spans = para.get("spans", [])
        
        for span in para_spans:
            para_offset = span.get("offset", 0)
            para_length = span.get("length", 0)
            para_end = para_offset + para_length
            
            # Check overlap with any table span
            for table_start, table_end in table_spans:
                if not (para_end <= table_start or para_offset >= table_end):
                    return True
        
        return False
    
    def load_paragraphs(self, raw_ocr: Dict, exclude_tables: bool = True) -> List[Paragraph]:
        """
        Extract paragraphs with spatial information from Azure DI output.
        
        Args:
            raw_ocr: The raw Azure DI JSON output
            exclude_tables: If True, exclude paragraphs that are part of tables
                           (these are handled separately by table chunks)
        """
        analyze_result = raw_ocr.get("analyzeResult", raw_ocr)
        paragraphs_data = analyze_result.get("paragraphs", [])
        
        # Get table spans for filtering
        table_spans = self._get_table_spans(raw_ocr) if exclude_tables else set()
        
        paragraphs = []
        
        for para in paragraphs_data:
            content = para.get("content", "").strip()
            if not content:
                continue
            
            # Skip paragraphs that are part of tables
            if exclude_tables and self._is_paragraph_in_table(para, table_spans):
                continue
            
            bounding_regions = para.get("boundingRegions", [])
            if not bounding_regions:
                continue
            
            region = bounding_regions[0]
            page_num = region.get("pageNumber", 1)
            polygon = region.get("polygon", [])
            
            # Polygon format: [x1,y1, x2,y2, x3,y3, x4,y4]
            # Corners: top-left, top-right, bottom-right, bottom-left
            if len(polygon) >= 8:
                x_left = min(polygon[0], polygon[6])
                x_right = max(polygon[2], polygon[4])
                y_top = min(polygon[1], polygon[3])
                y_bottom = max(polygon[5], polygon[7])
            else:
                continue
            
            paragraphs.append(Paragraph(
                content=content,
                page_number=page_num,
                x_left=x_left,
                x_right=x_right,
                y_top=y_top,
                y_bottom=y_bottom,
                role=para.get("role")
            ))
        
        return paragraphs
    
    def group_by_page(self, paragraphs: List[Paragraph]) -> Dict[int, List[Paragraph]]:
        """Group paragraphs by page number."""
        pages = {}
        for para in paragraphs:
            if para.page_number not in pages:
                pages[para.page_number] = []
            pages[para.page_number].append(para)
        
        # Sort each page by Y position (top to bottom)
        for page_num in pages:
            pages[page_num].sort(key=lambda p: p.y_top)
        
        return pages
    
    def _detect_page_columns(self, paragraphs: List[Paragraph]) -> float:
        """
        Dynamically detect column threshold based on page content.
        
        Strategy: Find the FIRST significant gap that separates left content from right content.
        This works better for documents with label-value columns.
        
        Returns the X position that best separates left and right columns.
        """
        if not paragraphs:
            return self.page_width / 2
        
        # Get unique X positions and sort them
        x_positions = sorted(set(round(p.x_left, 1) for p in paragraphs))
        
        if len(x_positions) < 2:
            return self.page_width / 2
        
        # Find gaps and their positions
        gaps = []
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            mid_point = (x_positions[i-1] + x_positions[i]) / 2
            gaps.append((gap, mid_point, x_positions[i-1], x_positions[i]))
        
        # Sort by gap size (largest first)
        gaps.sort(reverse=True)
        
        # Use the largest gap that's > 1.0 inch and is roughly in the middle of the page
        # This avoids splits at the very edge
        for gap, mid_point, left_x, right_x in gaps:
            if gap > 1.0 and 1.5 < mid_point < (self.page_width - 1.5):
                return mid_point
        
        # If no good gap found, try a smaller threshold
        for gap, mid_point, left_x, right_x in gaps:
            if gap > 0.8 and 1.0 < mid_point < (self.page_width - 1.0):
                return mid_point
        
        # Default to page center
        return self.page_width / 2
    
    def detect_columns(self, paragraphs: List[Paragraph]) -> Dict[int, List[Paragraph]]:
        """
        Detect column structure on a page.
        
        GENERIC: Dynamically detects column split based on X position gaps.
        """
        if not paragraphs:
            return {}
        
        # Dynamically find the column split point
        split_x = self._detect_page_columns(paragraphs)
        
        columns = {0: [], 1: []}  # 0 = left, 1 = right
        
        for para in paragraphs:
            if para.x_left < split_x:
                columns[0].append(para)
            else:
                columns[1].append(para)
        
        # Sort each column by Y position
        for col_id in columns:
            columns[col_id].sort(key=lambda p: p.y_top)
        
        # Remove empty columns
        columns = {k: v for k, v in columns.items() if v}
        
        return columns
    
    def is_label(self, text: str) -> bool:
        """
        GENERIC label detection using heuristics only.
        
        A text is likely a label if:
        1. Short (< max_label_length characters)
        2. Few words (1-3 words)
        3. ALL UPPERCASE, or
        4. Ends with colon (:)
        
        NOT a label if:
        - Starts with ':' (it's a value)
        - Contains mostly numbers
        - Looks like a date, amount, or code
        """
        text = text.strip()
        
        # Empty text
        if not text:
            return False
        
        # STARTS with colon - this is a VALUE, not a label
        if text.startswith(':'):
            return False
        
        # Too long to be a label
        if len(text) > self.max_label_length:
            return False
        
        # Too many words
        words = text.split()
        if len(words) > self.max_label_words:
            return False
        
        # Skip if mostly numbers/special chars (likely a value like "762005205RT")
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) < len(text) * 0.3:  # Less than 30% letters
            return False
        
        # Ends with colon - strong indicator of a label
        if text.endswith(':'):
            return True
        
        # ALL UPPERCASE and short (1-2 words, max 20 chars)
        # But must have meaningful letters, not just codes
        if text.isupper() and len(text) <= 20 and len(words) <= 2:
            # Must have at least 2 alphabetic characters
            if len(alpha_chars) >= 2:
                return True
        
        # Title case or mixed case short text (like "Client tax id", "INV Ref")
        # These are likely labels if they're short
        if len(text) <= 15 and len(words) <= 2:
            # First word starts with capital
            if text[0].isupper():
                return True
        
        return False
    
    def pair_horizontal_labels(self, paragraphs: List[Paragraph]) -> List[Tuple[str, str]]:
        """
        Pair labels with values that are to their RIGHT (same Y, different X).
        
        This handles layouts like:
        "Client tax id" | ": 762005205RT"
        
        Returns list of (label, value) tuples and set of used indices.
        """
        horizontal_pairs = []
        used_indices = set()
        
        # Sort by Y then X
        sorted_paras = sorted(paragraphs, key=lambda p: (p.y_top, p.x_left))
        
        for i, para in enumerate(sorted_paras):
            if i in used_indices:
                continue
                
            content = para.content.strip()
            
            # Check if this looks like a label OR if next item starts with ':'
            if self.is_label(content):
                # Look for value to the RIGHT on the same line
                for j, other in enumerate(sorted_paras):
                    if j <= i or j in used_indices:
                        continue
                    
                    other_content = other.content.strip()
                    
                    # Same Y position (within tolerance) and to the right or touching
                    y_diff = abs(other.y_top - para.y_top)
                    x_diff = other.x_left - para.x_right
                    
                    # Allow touching (x_diff >= -0.1) or small gap (x_diff < 2.0)
                    if y_diff < 0.15 and x_diff >= -0.1 and x_diff < 2.0:
                        # Value is to the right of label on same line
                        # Clean the value (remove leading colon and space)
                        value = other_content
                        if value.startswith(':'):
                            value = value[1:].strip()
                        
                        horizontal_pairs.append((content, value))
                        used_indices.add(i)
                        used_indices.add(j)
                        break
            
            # Also check: if this item is NOT a label but NEXT item starts with ':'
            # This catches "Client tax id" followed by ": value"
            elif not content.startswith(':'):
                for j, other in enumerate(sorted_paras):
                    if j <= i or j in used_indices:
                        continue
                    
                    other_content = other.content.strip()
                    
                    # Check if next item starts with ':' and is on same line
                    if other_content.startswith(':'):
                        y_diff = abs(other.y_top - para.y_top)
                        x_diff = other.x_left - para.x_right
                        
                        if y_diff < 0.15 and x_diff >= -0.1 and x_diff < 2.0:
                            value = other_content[1:].strip()  # Remove leading ':'
                            horizontal_pairs.append((content, value))
                            used_indices.add(i)
                            used_indices.add(j)
                            break
        
        return horizontal_pairs, used_indices
    
    def pair_labels_with_values(self, column: List[Paragraph]) -> List[Tuple[str, str, List[str]]]:
        """
        Pair labels with their values in a column.
        
        GENERIC: Uses only spatial proximity, no hardcoded keywords.
        
        Returns list of (label, value, additional_values) tuples.
        - (label, value, []) = label paired with single value
        - (label, None, []) = standalone label
        - (None, content, []) = standalone content
        """
        pairs = []
        i = 0
        
        while i < len(column):
            para = column[i]
            content = para.content.strip()
            
            # Check if this is a label
            if self.is_label(content):
                # Look for value in the NEXT paragraph only
                if i + 1 < len(column):
                    next_para = column[i + 1]
                    next_content = next_para.content.strip()
                    
                    # Check vertical proximity (must be directly below)
                    y_distance = next_para.y_top - para.y_bottom
                    
                    # Check if this is a valid pairing:
                    # 1. Close vertically
                    # 2. Next item is NOT a label
                    # 3. Similar X position (same column)
                    if (y_distance < self.vertical_threshold and 
                        y_distance >= 0 and  # Must be below, not above
                        not self.is_label(next_content) and
                        abs(next_para.x_left - para.x_left) < self.column_threshold):
                        
                        # Single value pairing
                        pairs.append((content, next_content, []))
                        i += 2
                        continue
                
                # No valid value found, keep label as standalone
                pairs.append((content, None, []))
                i += 1
            else:
                # Not a label, keep content as-is
                pairs.append((None, content, []))
                i += 1
        
        return pairs
    
    def pair_vertical_labels_direct(self, paragraphs: List[Paragraph]) -> List[Tuple[str, str]]:
        """
        Pair labels with values that are DIRECTLY BELOW them (same X, different Y).
        
        Uses a two-pass approach:
        1. First, find the BEST match for each label (nearest value in same X column)
        2. Then, assign pairs prioritizing closer matches
        
        Returns list of (label, value) tuples and set of used indices.
        """
        vertical_pairs = []
        used_indices = set()
        
        # Sort by Y then X
        sorted_paras = sorted(paragraphs, key=lambda p: (p.y_top, p.x_left))
        
        # Build candidate pairs: (label_idx, value_idx, y_distance)
        candidates = []
        
        for i, para in enumerate(sorted_paras):
            content = para.content.strip()
            
            if not self.is_label(content):
                continue
            
            # Find the NEAREST non-label value in the same X column
            best_match = None
            best_distance = float('inf')
            
            for j, other in enumerate(sorted_paras):
                if j <= i:
                    continue
                
                other_content = other.content.strip()
                
                # Must be below and same X column
                y_diff = other.y_top - para.y_bottom
                x_diff = abs(other.x_left - para.x_left)
                
                # Valid candidate: directly below (0 <= y_diff < 0.15) and same column (x_diff < 0.5)
                # Using tighter threshold (0.15) for "directly below"
                if 0 <= y_diff < 0.15 and x_diff < 0.5:
                    if not self.is_label(other_content):
                        if y_diff < best_distance:
                            best_match = j
                            best_distance = y_diff
            
            if best_match is not None:
                candidates.append((i, best_match, best_distance))
        
        # Sort candidates by distance (closest first) to prioritize tight pairings
        candidates.sort(key=lambda x: x[2])
        
        # Assign pairs, avoiding conflicts
        for label_idx, value_idx, _ in candidates:
            if label_idx not in used_indices and value_idx not in used_indices:
                label_content = sorted_paras[label_idx].content.strip()
                value_content = sorted_paras[value_idx].content.strip()
                vertical_pairs.append((label_content, value_content))
                used_indices.add(label_idx)
                used_indices.add(value_idx)
        
        return vertical_pairs, used_indices
    
    def process_page(self, paragraphs: List[Paragraph]) -> str:
        """
        Process a single page and return restructured text.
        
        1. First, detect HORIZONTAL label-value pairs (same line)
        2. Then, detect VERTICAL label-value pairs (same X column)
        3. Output remaining content as-is
        """
        if not paragraphs:
            return ""
        
        # Step 1: Find horizontal label-value pairs (label : value on same line)
        horizontal_pairs, h_used = self.pair_horizontal_labels(paragraphs)
        
        # Step 2: Get remaining paragraphs for vertical pairing
        sorted_paras = sorted(paragraphs, key=lambda p: (p.y_top, p.x_left))
        remaining_for_vertical = [p for i, p in enumerate(sorted_paras) if i not in h_used]
        
        # Step 3: Find vertical label-value pairs (label above value, same X)
        vertical_pairs, v_used_relative = self.pair_vertical_labels_direct(remaining_for_vertical)
        
        # Map v_used_relative back to original indices
        remaining_sorted = sorted(remaining_for_vertical, key=lambda p: (p.y_top, p.x_left))
        v_used_content = {remaining_sorted[i].content for i in v_used_relative}
        
        # Step 4: Collect unpaired paragraphs
        paired_content = set()
        for label, value in horizontal_pairs:
            paired_content.add(label)
            if value:
                # Value might have been cleaned of leading ':'
                paired_content.add(value)
        for label, value in vertical_pairs:
            paired_content.add(label)
            paired_content.add(value)
        
        # Also add the original ": value" forms
        for p in paragraphs:
            if p.content.startswith(':') and p.content[1:].strip() in paired_content:
                paired_content.add(p.content)
        
        unpaired = [p for p in sorted_paras if p.content not in paired_content and p.content not in v_used_content]
        
        # Build output
        all_lines = []
        
        # Add horizontal pairs
        for label, value in horizontal_pairs:
            if label.endswith(':'):
                all_lines.append(f"{label} {value}")
            else:
                all_lines.append(f"{label}: {value}")
        
        # Add vertical pairs
        for label, value in vertical_pairs:
            all_lines.append(f"{label}: {value}")
        
        if horizontal_pairs or vertical_pairs:
            all_lines.append("")
        
        # Add unpaired content
        for para in unpaired:
            all_lines.append(para.content)
        
        return "\n".join(all_lines)
    
    def process_document(self, raw_ocr: Dict) -> Dict[int, str]:
        """
        Process entire document and return restructured text per page.
        
        Returns: Dict mapping page_number -> restructured_text
        """
        paragraphs = self.load_paragraphs(raw_ocr)
        pages = self.group_by_page(paragraphs)
        
        results = {}
        for page_num, page_paras in pages.items():
            results[page_num] = self.process_page(page_paras)
        
        return results
    
    def get_paired_content_for_page(self, raw_ocr: Dict, page_number: int) -> str:
        """Get spatially-paired content for a specific page."""
        paragraphs = self.load_paragraphs(raw_ocr)
        page_paras = [p for p in paragraphs if p.page_number == page_number]
        return self.process_page(page_paras)


class PatternDetector:
    """
    GENERIC text cleaning utilities.
    
    No hardcoded keywords - only generic patterns.
    """
    
    def __init__(self):
        pass
    
    def clean_excessive_newlines(self, text: str) -> str:
        """Replace 3+ consecutive newlines with 2 newlines."""
        return re.sub(r'\n{3,}', '\n\n', text)
    
    def clean_table_placeholders(self, text: str) -> str:
        """Remove empty lines where tables were removed."""
        text = re.sub(r'\[TABLE_EXTRACTED\]\s*', '', text)
        lines = [line for line in text.split('\n') if line.strip() or line == '']
        return '\n'.join(lines)
    
    def process_text(self, text: str) -> str:
        """
        Generic text cleaning:
        1. Clean excessive newlines
        2. Remove table placeholders
        """
        text = self.clean_excessive_newlines(text)
        text = self.clean_table_placeholders(text)
        return text


def demo():
    """Demo the spatial pairing algorithm."""
    
    # Load raw OCR
    with open("RAW_OCR.json", "r", encoding="utf-8") as f:
        raw_ocr = json.load(f)
    
    # Initialize spatial pairing with default config
    pairing = SpatialPairing(config={
        "column_threshold": 0.5,
        "vertical_threshold": 0.2,
        "max_label_length": 25,
        "max_label_words": 3,
    })
    
    # Process document
    print("="*70)
    print("SPATIAL PAIRING DEMO (GENERIC)")
    print("="*70)
    
    # Load paragraphs
    paragraphs = pairing.load_paragraphs(raw_ocr)
    print(f"\nLoaded {len(paragraphs)} paragraphs")
    
    # Show Page 1 analysis
    page1_paras = [p for p in paragraphs if p.page_number == 1]
    print(f"\nPage 1 has {len(page1_paras)} paragraphs")
    
    # Show dynamic column detection
    split_x = pairing._detect_page_columns(page1_paras)
    print(f"\nDynamic column split detected at X = {split_x:.2f}")
    
    # Show column detection
    columns = pairing.detect_columns(page1_paras)
    print(f"Detected {len(columns)} columns on Page 1:")
    
    for col_id, col_paras in columns.items():
        min_x = min(p.x_left for p in col_paras)
        max_x = max(p.x_left for p in col_paras)
        print(f"\n  Column {col_id} (X range: {min_x:.2f} - {max_x:.2f}):")
        for para in col_paras[:8]:  # Show first 8
            is_label = "📌" if pairing.is_label(para.content) else "  "
            content = para.content[:35] + "..." if len(para.content) > 35 else para.content
            print(f"    {is_label} Y={para.y_top:.2f} | '{content}'")
    
    # Show paired result
    print(f"\n{'='*70}")
    print("PAIRED RESULT FOR PAGE 1:")
    print("="*70)
    
    result = pairing.process_page(page1_paras)
    print(result[:1500])  # Show first 1500 chars


if __name__ == "__main__":
    demo()
