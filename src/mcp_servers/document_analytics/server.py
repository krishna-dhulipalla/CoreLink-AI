import json
import logging
import uuid
from typing import Optional, Dict

import pandas as pd
import pdfplumber
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("DocumentAnalytics")

# In-memory storage for extracted tables across tool calls
_TABLES: Dict[str, pd.DataFrame] = {}

logger = logging.getLogger(__name__)


@mcp.tool()
def extract_pdf_tables(file_path: str, pages: Optional[list[int]] = None) -> list[dict]:
    """
    Extract structured tables from a PDF document. Returns reference IDs for each extracted table 
    that can be queried using get_table_rows, filter_rows, or sum_column.
    
    Args:
        file_path: Absolute path to the PDF file.
        pages: Optional list of specific page numbers (1-indexed) to extract from. Defaults to first 10 pages.
    """
    extracted_metadata = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            # Default to first 10 pages to prevent memory explosion if not specified
            pages_to_process = pages if pages else list(range(1, min(11, len(pdf.pages) + 1)))
            
            for page_num in pages_to_process:
                # pdfplumber pages are 0-indexed, user page_nums are 1-indexed
                if page_num < 1 or page_num > len(pdf.pages):
                    continue
                    
                page = pdf.pages[page_num - 1]
                tables = page.extract_tables()
                
                for idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue # Skip empty or 1-row tables
                        
                    # Standardize table into a DataFrame
                    # Assuming first row is header
                    header = [str(c).replace('\n', ' ').strip() if c else f"Col_{i}" for i, c in enumerate(table[0])]
                    
                    df = pd.DataFrame(table[1:], columns=header)
                    
                    # Clean up the dataframe
                    df = df.dropna(how='all')
                    
                    table_id = f"table_{uuid.uuid4().hex[:8]}"
                    _TABLES[table_id] = df
                    
                    extracted_metadata.append({
                        "table_id": table_id,
                        "rows": len(df),
                        "columns": header,
                        "provenance": {"file": file_path, "page": page_num, "table_index_on_page": idx}
                    })
                    
        return extracted_metadata
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")
        return [{"error": str(e)}]


@mcp.tool()
def search_document_pages(file_path: str, query: str) -> list[dict]:
    """
    Search a PDF document for a specific keyword or phrase to identify relevant page numbers.
    Useful for finding tables before extracting them.
    
    Args:
        file_path: Absolute path to the PDF file.
        query: Substring to search for.
    """
    results = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and query.lower() in text.lower():
                    # Extract a snippet around the match
                    idx = text.lower().find(query.lower())
                    start = max(0, idx - 50)
                    end = min(len(text), idx + 50)
                    snippet = text[start:end].replace('\n', ' ')
                    results.append({
                        "page": i + 1,
                        "snippet": f"...{snippet}...",
                        "provenance": {"file": file_path, "page": i + 1}
                    })
        return results if results else [{"message": "No match found."}]
    except Exception as e:
        logger.error(f"Error searching document: {e}")
        return [{"error": str(e)}]


@mcp.tool()
def get_table_rows(table_id: str, limit: int = 5) -> dict:
    """
    Retrieve rows from an extracted table.
    
    Args:
        table_id: The ID of the table returned by extract_pdf_tables.
        limit: Number of rows to return (default 5).
    """
    if table_id not in _TABLES:
        return {"error": f"Table ID {table_id} not found. Must extract first."}
        
    df = _TABLES[table_id]
    return {
        "columns": df.columns.tolist(),
        "rows": df.head(limit).to_dict(orient="records"),
        "total_rows": len(df),
        "note": f"Showing first {limit} rows."
    }


@mcp.tool()
def filter_rows(table_id: str, column_matcher: str, value_matcher: str) -> dict:
    """
    Filter rows in a table where a column matches a specific substring value.
    This does substring matching (case-insensitive) on the column string values.
    
    Args:
        table_id: The ID of the table returned by extract_pdf_tables.
        column_matcher: Substring to match the column name.
        value_matcher: Substring to match within the cells of that column.
    """
    if table_id not in _TABLES:
        return {"error": f"Table ID {table_id} not found."}
        
    df = _TABLES[table_id]
    
    # Find active column
    col = next((c for c in df.columns if column_matcher.lower() in str(c).lower()), None)
    if not col:
        return {"error": f"No column matching '{column_matcher}'. Columns: {df.columns.tolist()}"}
        
    filtered_df = df[df[col].astype(str).str.contains(value_matcher, case=False, na=False)]
    
    new_table_id = f"table_{uuid.uuid4().hex[:8]}"
    _TABLES[new_table_id] = filtered_df
    
    return {
        "new_table_id": new_table_id,
        "matched_column": col,
        "rows": filtered_df.head(5).to_dict(orient="records"),
        "total_matches": len(filtered_df),
        "note": "Use this new_table_id for chained operations."
    }


@mcp.tool()
def sum_column(table_id: str, column_matcher: str) -> dict:
    """
    Calculate the sum of a numeric column in an extracted table.
    Handles commas, dollar signs, and parentheses for negative numbers.
    
    Args:
        table_id: The ID of the table returned by extract_pdf_tables.
        column_matcher: Substring to match the column name you want to sum.
    """
    if table_id not in _TABLES:
        return {"error": f"Table ID {table_id} not found."}
        
    df = _TABLES[table_id]
    
    col = next((c for c in df.columns if column_matcher.lower() in str(c).lower()), None)
    if not col:
        return {"error": f"No column matching '{column_matcher}'. Columns: {df.columns.tolist()}"}

    def clean_numeric(val):
        if pd.isna(val):
            return 0.0
        val_str = str(val).replace('$', '').replace(',', '').strip()
        # Handle (100) as -100
        if val_str.startswith('(') and val_str.endswith(')'):
            val_str = '-' + val_str[1:-1]
        try:
            return float(val_str)
        except ValueError:
            return 0.0

    numeric_series = df[col].apply(clean_numeric)
    total_sum = numeric_series.sum()
    
    return {
        "column": col,
        "sum": float(total_sum),
        "rows_aggregated": len(numeric_series)
    }

if __name__ == "__main__":
    mcp.run()
