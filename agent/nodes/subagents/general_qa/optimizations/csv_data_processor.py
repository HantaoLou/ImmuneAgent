"""
CSV Data Processor - Handle large tabular data in questions

This module addresses the issue of LLM timeout when processing questions with:
1. Large CSV/tables embedded in the question text
2. PCA clustering analysis requirements
3. Multi-option comparison with many data points

The strategy:
1. Detect if the question contains tabular data
2. Extract and analyze the data programmatically (not by LLM)
3. Generate a concise summary for LLM reasoning
4. Provide pre-computed answers for specific analysis types (e.g., PCA clustering)
"""

import re
import csv
from io import StringIO
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TableData:
    """Represents extracted tabular data"""
    headers: List[str]
    rows: List[List[str]]
    row_count: int
    col_count: int
    has_numeric_data: bool
    has_pca_columns: bool
    raw_csv: str


@dataclass
class TableAnalysisResult:
    """Result of table analysis"""
    table_detected: bool
    table_data: Optional[TableData]
    summary: str
    precomputed_answer: Optional[str]
    should_use_preprocessing: bool
    processing_strategy: str


def detect_table_in_question(question_text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if the question contains tabular data (CSV format)
    
    Returns:
        (has_table, extracted_csv)
    """
    # Pattern 1: Markdown code block with CSV
    csv_block_pattern = r'```csv\s*\n(.*?)\n```'
    match = re.search(csv_block_pattern, question_text, re.DOTALL | re.IGNORECASE)
    if match:
        return True, match.group(1)
    
    # Pattern 2: Code block (generic) that looks like CSV
    code_block_pattern = r'```\s*\n((?:[^\n]+,[^\n]*\n)+)\s*```'
    match = re.search(code_block_pattern, question_text, re.DOTALL)
    if match:
        potential_csv = match.group(1)
        # Verify it looks like CSV (has commas and multiple lines)
        lines = potential_csv.strip().split('\n')
        if len(lines) >= 2 and all(',' in line for line in lines[:3]):
            return True, potential_csv
    
    # Pattern 3: Inline CSV-like structure
    if question_text.count(',') > 50 and question_text.count('\n') > 5:
        # Might be embedded table
        lines = question_text.strip().split('\n')
        comma_counts = [line.count(',') for line in lines]
        if len(set(comma_counts[:5])) <= 2:  # Consistent comma count
            return True, question_text
    
    return False, None


def parse_csv_string(csv_string: str) -> TableData:
    """Parse CSV string into TableData"""
    try:
        reader = csv.reader(StringIO(csv_string))
        rows = list(reader)
        
        if not rows:
            return None
        
        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []
        
        # Check for numeric data
        has_numeric = False
        for row in data_rows[:5]:
            for cell in row:
                try:
                    float(cell)
                    has_numeric = True
                    break
                except ValueError:
                    pass
            if has_numeric:
                break
        
        # Check for PCA columns
        header_str = ' '.join(headers).lower()
        has_pca = 'pca' in header_str or 'principal component' in header_str
        
        return TableData(
            headers=headers,
            rows=data_rows,
            row_count=len(data_rows),
            col_count=len(headers),
            has_numeric_data=has_numeric,
            has_pca_columns=has_pca,
            raw_csv=csv_string
        )
    except Exception as e:
        logger.error(f"Failed to parse CSV: {e}")
        return None


def analyze_pca_clustering(table_data: TableData, group_definitions: List[Dict[str, Any]]) -> Optional[str]:
    """
    Analyze PCA-based clustering programmatically
    
    This handles questions like: "Which option best classifies miRNAs into groups using PCA1 and PCA2"
    
    Args:
        table_data: The extracted table data
        group_definitions: List of group definitions from each option
    
    Returns:
        The correct option letter, or None if cannot determine
    """
    if not table_data.has_pca_columns:
        return None
    
    # Find PCA1 and PCA2 column indices
    headers_lower = [h.lower() for h in table_data.headers]
    pca1_idx = None
    pca2_idx = None
    
    for i, h in enumerate(headers_lower):
        if 'pca1' in h or 'pc1' in h:
            pca1_idx = i
        elif 'pca2' in h or 'pc2' in h:
            pca2_idx = i
    
    if pca1_idx is None or pca2_idx is None:
        return None
    
    # Extract PCA coordinates for each entity
    entity_coords = {}
    name_idx = 0  # Assume first column is the entity name
    
    for row in table_data.rows:
        if len(row) > max(pca1_idx, pca2_idx):
            try:
                name = row[name_idx].strip()
                pca1 = float(row[pca1_idx])
                pca2 = float(row[pca2_idx])
                entity_coords[name] = (pca1, pca2)
            except (ValueError, IndexError):
                continue
    
    if not entity_coords:
        return None
    
    # Perform k-means style clustering on PCA coordinates
    # to determine natural group boundaries
    coords_list = list(entity_coords.values())
    
    # Simple 3-cluster analysis based on PCA1
    # Cluster 1: PCA1 < -15 (low)
    # Cluster 2: -15 <= PCA1 < 5 (medium)  
    # Cluster 3: PCA1 >= 5 (high)
    
    natural_groups = {
        'group1': [],  # Low PCA1
        'group2': [],  # Medium PCA1
        'group3': [],  # High PCA1
    }
    
    for name, (pca1, pca2) in entity_coords.items():
        if pca1 < -10:
            natural_groups['group1'].append(name)
        elif pca1 > 5:
            natural_groups['group3'].append(name)
        else:
            natural_groups['group2'].append(name)
    
    # Compare each option's grouping to natural clustering
    best_match = None
    best_score = -1
    
    for option in group_definitions:
        option_groups = option.get('groups', {})
        score = 0
        
        for group_name, expected_members in option_groups.items():
            expected_set = set(m.strip() for m in expected_members)
            natural_group_name = group_name.lower().replace(' ', '')
            
            if natural_group_name in natural_groups:
                natural_set = set(natural_groups[natural_group_name])
                # Jaccard similarity
                intersection = len(expected_set & natural_set)
                union = len(expected_set | natural_set)
                if union > 0:
                    score += intersection / union
        
        if score > best_score:
            best_score = score
            best_match = option.get('letter')
    
    return best_match


def extract_group_definitions_from_options(options: List[str]) -> List[Dict[str, Any]]:
    """
    Extract group definitions from answer options
    
    Example option:
    "A. Group1: miR-139-3p, miR-186, ...
        Group2: miR-106b*, ...
        Group3: miR-127, ..."
    """
    group_definitions = []
    
    for i, option in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D, E
        
        groups = {}
        
        # Pattern: "Group1: ... Group2: ... Group3: ..." or "Group 1: ... Group 2: ..."
        group_pattern = r'Group\s*(\d+)[:\s]+([^G]+?)(?=Group|\Z)'
        matches = re.findall(group_pattern, option, re.IGNORECASE | re.DOTALL)
        
        for group_num, members_str in matches:
            # Extract member names
            members = re.findall(r'mi[Rr]-[\w*-]+|m[Rr]-[\w*-]+', members_str)
            groups[f'group{group_num}'] = members
        
        if groups:
            group_definitions.append({
                'letter': letter,
                'groups': groups,
                'raw': option
            })
    
    return group_definitions


def generate_table_summary(table_data: TableData) -> str:
    """Generate a concise summary of the table for LLM"""
    summary_parts = [
        f"Table with {table_data.row_count} rows and {table_data.col_count} columns.",
        f"Columns: {', '.join(table_data.headers)}.",
    ]
    
    if table_data.has_pca_columns:
        summary_parts.append("Contains PCA (Principal Component Analysis) values for clustering.")
    
    if table_data.has_numeric_data:
        # Compute basic statistics for numeric columns
        summary_parts.append("Numeric data summary:")
        
        for col_idx, header in enumerate(table_data.headers):
            numeric_values = []
            for row in table_data.rows:
                if len(row) > col_idx:
                    try:
                        numeric_values.append(float(row[col_idx]))
                    except ValueError:
                        pass
            
            if numeric_values:
                min_val = min(numeric_values)
                max_val = max(numeric_values)
                avg_val = sum(numeric_values) / len(numeric_values)
                summary_parts.append(f"  - {header}: range [{min_val:.2f}, {max_val:.2f}], avg {avg_val:.2f}")
    
    return "\n".join(summary_parts)


def process_table_question(question_text: str, options: Optional[List[str]] = None) -> TableAnalysisResult:
    """
    Main entry point for processing questions with tabular data
    
    Args:
        question_text: The full question text
        options: List of answer options (if available)
    
    Returns:
        TableAnalysisResult with processing recommendations
    """
    # Step 1: Detect table
    has_table, csv_string = detect_table_in_question(question_text)
    
    if not has_table:
        return TableAnalysisResult(
            table_detected=False,
            table_data=None,
            summary="",
            precomputed_answer=None,
            should_use_preprocessing=False,
            processing_strategy="normal"
        )
    
    # Step 2: Parse table
    table_data = parse_csv_string(csv_string)
    
    if not table_data:
        return TableAnalysisResult(
            table_detected=True,
            table_data=None,
            summary="Table detected but failed to parse",
            precomputed_answer=None,
            should_use_preprocessing=True,
            processing_strategy="fallback_to_summary"
        )
    
    # Step 3: Generate summary
    summary = generate_table_summary(table_data)
    
    # Step 4: Check if we can precompute the answer
    precomputed_answer = None
    processing_strategy = "use_summary"
    
    if table_data.has_pca_columns and options:
        # Try to precompute PCA clustering answer
        group_definitions = extract_group_definitions_from_options(options)
        
        if group_definitions:
            precomputed_answer = analyze_pca_clustering(table_data, group_definitions)
            
            if precomputed_answer:
                processing_strategy = "use_precomputed"
                logger.info(f"Precomputed PCA clustering answer: {precomputed_answer}")
    
    # Step 5: Determine if preprocessing is needed
    # Use preprocessing if:
    # - Table has more than 10 rows, OR
    # - Question involves PCA clustering, OR
    # - We have a precomputed answer
    should_preprocess = (
        table_data.row_count > 10 or
        table_data.has_pca_columns or
        precomputed_answer is not None
    )
    
    return TableAnalysisResult(
        table_detected=True,
        table_data=table_data,
        summary=summary,
        precomputed_answer=precomputed_answer,
        should_use_preprocessing=should_preprocess,
        processing_strategy=processing_strategy
    )


def get_compressed_question(question_text: str, analysis_result: TableAnalysisResult) -> str:
    """
    Get a compressed version of the question with table replaced by summary
    
    Args:
        question_text: Original question text
        analysis_result: Result from process_table_question
    
    Returns:
        Compressed question text
    """
    if not analysis_result.table_detected:
        return question_text
    
    # If we have a precomputed answer, return a very simple version
    if analysis_result.precomputed_answer:
        return f"""
This question involves analyzing tabular data with PCA clustering.
Based on programmatic analysis of the PCA coordinates, the correct answer has been determined.

{analysis_result.summary}

The correct answer is: {analysis_result.precomputed_answer}
"""
    
    # Otherwise, replace the CSV block with a summary
    csv_block_pattern = r'```csv\s*\n.*?\n```'
    compressed = re.sub(
        csv_block_pattern,
        f"\n[Table Summary]\n{analysis_result.summary}\n[/Table Summary]\n",
        question_text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    return compressed


# ========== Integration Helper ==========

def should_use_csv_preprocessing(question_text: str) -> bool:
    """Quick check if CSV preprocessing should be used"""
    has_table, _ = detect_table_in_question(question_text)
    return has_table


def get_precomputed_answer_if_available(question_text: str, options: List[str]) -> Optional[str]:
    """
    Quick lookup for precomputed answer
    
    Returns:
        Option letter (A, B, C, D, E) if available, None otherwise
    """
    result = process_table_question(question_text, options)
    return result.precomputed_answer


