"""DOCX document text extraction.

Fills the one gap in the existing document parsing infrastructure:
    - PDF:   pypdf in kb/
    - Excel: openpyxl in file_utils
    - CSV:   file_utils
    - DOCX:  **this module** (was missing)

Lazy import: python-docx loaded inside function scope only.

LangChain 1.0+ Compatibility:
    - Uses @tool decorator from langchain_core.tools
    - Can be directly bound to LLM via .bind_tools()
"""

import logging

from langchain_core.tools import tool

from ._output import truncate_output

logger = logging.getLogger(__name__)


@tool
def parse_docx(file_path: str, max_chars: int = 6000) -> str:
    """Extract text from a DOCX file.

    Args:
        file_path: Path to the .docx file
        max_chars: Maximum characters to return (default: 6000)

    Returns:
        Extracted text content, truncated if needed
    """
    try:
        import docx  # lazy import — python-docx
    except ImportError:
        return "[parse_docx] Error: python-docx not installed. Run: pip install python-docx"

    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            return f"[parse_docx] No text content found in: {file_path}"
        text = "\n\n".join(paragraphs)
        return truncate_output(f"[DOCX Content from: {file_path}]\n\n{text}", max_chars)
    except Exception as e:
        logger.error(f"parse_docx failed for {file_path}: {e}")
        return f"[parse_docx] Error reading {file_path}: {e}"


def get_docx_tools() -> list:
    """Return DOCX tools as LangChain tools.
    
    Returns:
        List of LangChain tool objects that can be directly bound to LLM.
    """
    return [parse_docx]


def get_docx_tools_dict() -> dict:
    """Return DOCX tools for backward compatibility (namespace injection)."""
    return {"parse_docx": parse_docx}
