"""
File Handler MCP Server
========================
Provides generic document retrieval and parsing for the Purple Agent.
Supports fetching reference files (PDF, Excel, Word, CSV, JSON, TXT)
provided via URLs by benchmark green agents.

Usage pattern:
    Green agent appends "REFERENCE FILES AVAILABLE: [URL]" to the prompt.
    The Purple Agent calls fetch_reference_file(url=...) to retrieve
    and parse the file contents.

Pagination is supported for large documents to control token usage.
"""

import csv
import io
import json
import re
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("File Handler")

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT = 30.0
MAX_TEXT_CHARS = 12_000   # cap per single page/chunk before returning

# ── Helpers ────────────────────────────────────────────────────────────────

def _sniff_format(url: str, content_type: str) -> str:
    """Infer file format from URL extension or Content-Type header."""
    url_lower = url.lower().split("?")[0]
    for ext, fmt in [
        (".pdf", "pdf"), (".xlsx", "excel"), (".xls", "excel"),
        (".docx", "word"), (".doc", "word"),
        (".csv", "csv"), (".json", "json"),
        (".png", "image"), (".jpg", "image"), (".jpeg", "image"),
        (".wav", "audio"), (".mp3", "audio"),
        (".mp4", "video"), (".avi", "video"),
        (".zip", "archive"), (".tar.gz", "archive"), (".rar", "archive"),
        (".txt", "text"), (".md", "text"),
    ]:
        if url_lower.endswith(ext):
            return fmt
    ct = content_type.lower()
    if "pdf" in ct:                   return "pdf"
    if "spreadsheet" in ct:           return "excel"
    if "wordprocessing" in ct:        return "word"
    if "csv" in ct:                    return "csv"
    if "json" in ct:                   return "json"
    if "image" in ct:                  return "image"
    if "audio" in ct:                  return "audio"
    if "video" in ct:                  return "video"
    if "zip" in ct or "tar" in ct:     return "archive"
    return "text"


def _parse_pdf(raw: bytes, page_start: int, page_limit: int) -> tuple[str, int]:
    """Parse PDF and return text for the requested page range."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        total = len(reader.pages)
        end = min(page_start + page_limit, total)
        pages = []
        for i in range(page_start, end):
            text = reader.pages[i].extract_text() or ""
            pages.append(f"[Page {i + 1}]\n{text.strip()}")
        return "\n\n".join(pages), total
    except ImportError:
        return "[ERROR: pypdf not installed. Run: uv add pypdf]", 0
    except Exception as e:
        return f"[PDF parse error: {e}]", 0


def _parse_excel(raw: bytes, sheet: Optional[str], row_offset: int, row_limit: int) -> tuple[str, int]:
    """Parse Excel workbook and return rows for the requested range."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        ws = wb[sheet] if sheet and sheet in wb.sheetnames else wb.active
        all_rows = list(ws.iter_rows(values_only=True))
        total = len(all_rows)
        rows = all_rows[row_offset: row_offset + row_limit]
        lines = []
        for r in rows:
            lines.append("\t".join("" if v is None else str(v) for v in r))
        return "\n".join(lines), total
    except ImportError:
        return "[ERROR: openpyxl not installed. Run: uv add openpyxl]", 0
    except Exception as e:
        return f"[Excel parse error: {e}]", 0


def _parse_word(raw: bytes) -> str:
    """Parse Word document and return full text."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        return "[ERROR: python-docx not installed. Run: uv add python-docx]"
    except Exception as e:
        return f"[Word parse error: {e}]"


def _parse_csv(raw: bytes, row_offset: int, row_limit: int) -> tuple[str, int]:
    """Parse CSV and return rows for the requested range."""
    try:
        text = raw.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        all_rows = list(reader)
        total = len(all_rows)
        rows = all_rows[row_offset: row_offset + row_limit]
        return "\n".join(",".join(r) for r in rows), total
    except Exception as e:
        return f"[CSV parse error: {e}]", 0


# ── Tools ──────────────────────────────────────────────────────────────────

@mcp.tool()
def fetch_reference_file(
    url: str,
    page_start: int = 0,
    page_limit: int = 5,
    row_offset: int = 0,
    row_limit: int = 200,
    sheet: Optional[str] = None,
    format_hint: Optional[str] = None,
) -> str:
    """Download and parse a reference file from a URL provided by the evaluator.

    Use this tool when the task includes text like:
        "REFERENCE FILES AVAILABLE: https://example.com/data.xlsx"
    or any URL pointing to a data/document file you need to read.

    Supported formats (auto-detected): PDF, Excel (.xlsx), Word (.docx),
    CSV, JSON, TXT/Markdown.

    Pagination args (to control token size):
        - PDF: page_start (0-indexed), page_limit (default 5 pages)
        - Excel/CSV: row_offset (default 0), row_limit (default 200 rows)
        - sheet: optional Excel sheet name (default: first active sheet)
        - format_hint: optionally force format ('pdf','excel','csv','json','word','text')

    Args:
        url: Direct download URL of the reference file
        page_start: First PDF page to read (0-indexed, default 0)
        page_limit: Number of PDF pages to read (default 5)
        row_offset: First row to read for tabular data (default 0)
        row_limit: Max rows to read for tabular data (default 200)
        sheet: Excel sheet name (optional; uses first sheet by default)
        format_hint: Force a specific format ('pdf','excel','word','csv','json','text')

    Returns:
        Parsed file content as a string, with metadata header.
    """
    try:
        # ── Fetch ──────────────────────────────────────────────────────────
        with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

        raw = resp.content
        content_type = resp.headers.get("content-type", "")
        size_kb = len(raw) / 1024

        fmt = format_hint.lower() if format_hint else _sniff_format(url, content_type)

        header = (
            f"FILE: {url.split('/')[-1].split('?')[0]}\n"
            f"FORMAT: {fmt.upper()} | SIZE: {size_kb:.1f} KB\n"
            f"{'─' * 50}\n"
        )

        # ── Parse ──────────────────────────────────────────────────────────
        if fmt == "pdf":
            content, total = _parse_pdf(raw, page_start, page_limit)
            meta = f"[Pages {page_start + 1}–{min(page_start + page_limit, total)} of {total}]\n"
            return header + meta + content[:MAX_TEXT_CHARS]

        elif fmt == "excel":
            content, total = _parse_excel(raw, sheet, row_offset, row_limit)
            meta = f"[Rows {row_offset}–{row_offset + row_limit} of ~{total}]\n"
            return header + meta + content[:MAX_TEXT_CHARS]

        elif fmt == "csv":
            content, total = _parse_csv(raw, row_offset, row_limit)
            meta = f"[Rows {row_offset}–{row_offset + row_limit} of ~{total}]\n"
            return header + meta + content[:MAX_TEXT_CHARS]

        elif fmt == "word":
            content = _parse_word(raw)
            return header + content[:MAX_TEXT_CHARS]

        elif fmt == "json":
            try:
                parsed = json.loads(raw)
                pretty = json.dumps(parsed, indent=2)
                truncated = pretty[:MAX_TEXT_CHARS]
                suffix = "\n... [truncated, use pagination if needed]" if len(pretty) > MAX_TEXT_CHARS else ""
                return header + truncated + suffix
            except json.JSONDecodeError as e:
                return header + f"[JSON parse error: {e}]\nRaw:\n{raw[:1000].decode('utf-8', errors='replace')}"

        elif fmt == "image":
            return (
                header
                + "[IMAGE FILE DETECTED]\n"
                + "This tool cannot extract pixel data from images. "
                + "If the image contains a chart or table, ask the evaluator to provide data in CSV/JSON format instead."
            )

        elif fmt in ["audio", "video"]:
            return (
                header
                + f"[{fmt.upper()} FILE DETECTED]\n"
                + f"This tool cannot extract audio/video streams. "
                + "If the task requires processing media files natively, an external specialized tool must be provided."
            )

        elif fmt == "archive":
            return (
                header
                + "[ARCHIVE FILE DETECTED]\n"
                + "This tool cannot extract or traverse zip/tar archives directly. "
            )

        else:  # text / markdown / fallback
            text = raw.decode("utf-8", errors="replace")
            truncated = text[:MAX_TEXT_CHARS]
            suffix = "\n... [truncated]" if len(text) > MAX_TEXT_CHARS else ""
            return header + truncated + suffix

    except httpx.HTTPStatusError as e:
        return f"[HTTP {e.response.status_code}] Failed to fetch {url}: {e}"
    except httpx.RequestError as e:
        return f"[Network error] Could not reach {url}: {e}"
    except Exception as e:
        return f"[Unexpected error reading {url}]: {e}"


@mcp.tool()
def list_reference_files(prompt_text: str) -> str:
    """Extract all reference file URLs from the task prompt text.

    Use this as a first step when you suspect the task contains reference files.
    It finds URL patterns and detects 'REFERENCE FILES AVAILABLE' signals.

    Args:
        prompt_text: The full task prompt or description string.

    Returns:
        A list of detected file URLs and their inferred formats.
    """
    # Find explicit reference file markers
    urls: list[str] = []
    
    # Common URL pattern
    url_pattern = re.compile(
        r"https?://[^\s\)\]\"\'\,]+"
    )
    found = url_pattern.findall(prompt_text)
    
    # Filter to likely file URLs (extension or path hints)
    file_extensions = {".pdf", ".csv", ".xlsx", ".xls", ".docx", ".doc", ".json", ".txt", ".md"}
    for u in found:
        clean = u.rstrip(".,;)")
        if any(clean.lower().endswith(ext) for ext in file_extensions) or "download" in clean.lower() or "file" in clean.lower():
            urls.append(clean)
        elif u not in urls:
            # Include all URLs if "REFERENCE" keyword is near
            urls.append(clean)

    has_ref_signal = "REFERENCE FILES AVAILABLE" in prompt_text.upper() or "REFERENCE FILE" in prompt_text.upper()

    if not urls:
        return (
            "No reference file URLs detected in the prompt.\n"
            + ("⚠️  'REFERENCE FILES AVAILABLE' keyword found but no URLs extracted — check prompt formatting." 
               if has_ref_signal else "")
        )

    lines = [
        f"{'⚠️  REFERENCE FILES DETECTED' if has_ref_signal else 'URLs found in prompt'}:",
        ""
    ]
    for i, u in enumerate(urls, 1):
        # Sniff format heuristically
        ext = next(
            (ext for ext in [".pdf",".csv",".xlsx",".xls",".docx",".json",".txt"] if u.lower().endswith(ext)),
            "unknown"
        )
        lines.append(f"  {i}. [{ext.lstrip('.').upper() or 'URL'}] {u}")

    lines.append("")
    lines.append("→ Call fetch_reference_file(url=...) for each URL you need to read.")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
