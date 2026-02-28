"""Literature search tools for ImmuneAgent.

Provides 6 synchronous functions for searching academic literature across
PubMed, Semantic Scholar, BioRxiv/MedRxiv, and Europe PMC. All tools use
stdlib urllib for HTTP requests and return truncated formatted strings.

Design:
    - Lazy imports for heavy dependencies
    - Stdlib-only HTTP via urllib.request
    - All outputs truncated to 6000 chars max
    - Environment-based API key loading
    - Graceful error handling with descriptive messages
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

from ._output import PaperSummary, truncate_output

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment loading pattern
# ---------------------------------------------------------------------------
_env_loaded = False


def _ensure_env_loaded():
    """Load .env files from project root and deep_research subagent."""
    global _env_loaded
    if _env_loaded:
        return
    try:
        from dotenv import load_dotenv

        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # agent/tools -> agent -> project root
        for env_path in [
            project_root / ".env",
            current_dir.parent / "nodes" / "subagents" / "deep_research" / ".env",
        ]:
            if env_path.exists():
                load_dotenv(env_path, override=False)
                logger.debug(f"Loaded environment from {env_path}")
    except ImportError:
        logger.debug("python-dotenv not installed, skipping .env loading")
    _env_loaded = True


# ---------------------------------------------------------------------------
# L1: PubMed search via NCBI E-utilities
# ---------------------------------------------------------------------------
def search_pubmed(
    query: str,
    max_results: int = 10,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    mesh_terms: Optional[str] = None,
) -> str:
    """Search PubMed using NCBI E-utilities (esearch + efetch).

    Args:
        query: Search query string (supports PubMed query syntax)
        max_results: Maximum number of results to return (default 10)
        date_from: Start date in YYYY/MM/DD format (optional)
        date_to: End date in YYYY/MM/DD format (optional)
        mesh_terms: MeSH terms to filter by, comma-separated (optional)

    Returns:
        Formatted string with paper summaries, truncated to 6000 chars

    Example:
        search_pubmed("COVID-19 vaccine", max_results=5, date_from="2023/01/01")
    """
    _ensure_env_loaded()
    api_key = os.getenv("NCBI_API_KEY", "")

    try:
        # Build query with filters
        search_query = query
        if date_from and date_to:
            search_query += f" AND {date_from}:{date_to}[dp]"
        elif date_from:
            search_query += f" AND {date_from}:3000[dp]"
        elif date_to:
            search_query += f" AND 1800:{date_to}[dp]"

        if mesh_terms:
            search_query += f" AND {mesh_terms}[MeSH]"

        # Step 1: esearch to get PMIDs
        esearch_params = {
            "db": "pubmed",
            "term": search_query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        }
        if api_key:
            esearch_params["api_key"] = api_key

        esearch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"{urllib.parse.urlencode(esearch_params)}"
        )

        req = urllib.request.Request(esearch_url)
        with urllib.request.urlopen(req, timeout=30) as response:
            esearch_data = json.loads(response.read().decode())

        pmids = esearch_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return "[search_pubmed] No results found."

        # Step 2: efetch to get paper details
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        if api_key:
            efetch_params["api_key"] = api_key

        efetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"{urllib.parse.urlencode(efetch_params)}"
        )

        req = urllib.request.Request(efetch_url)
        with urllib.request.urlopen(req, timeout=30) as response:
            xml_data = response.read().decode()

        # Parse XML
        root = ET.fromstring(xml_data)
        papers = []

        for article in root.findall(".//PubmedArticle"):
            try:
                pmid = article.findtext(".//PMID")
                title = article.findtext(".//ArticleTitle") or "No title"

                # Authors
                author_list = article.findall(".//Author")
                if author_list:
                    authors = []
                    for author in author_list[:3]:  # First 3 authors
                        last = author.findtext("LastName") or ""
                        init = author.findtext("Initials") or ""
                        if last:
                            authors.append(f"{last} {init}".strip())
                    author_str = ", ".join(authors)
                    if len(author_list) > 3:
                        author_str += ", et al."
                else:
                    author_str = "No authors listed"

                # Journal and year
                journal = article.findtext(".//Journal/Title") or article.findtext(
                    ".//Journal/ISOAbbreviation"
                )
                year_elem = article.find(".//PubDate/Year")
                year = int(year_elem.text) if year_elem is not None else None

                # DOI
                doi = None
                for article_id in article.findall(".//ArticleId"):
                    if article_id.get("IdType") == "doi":
                        doi = article_id.text
                        break

                paper = PaperSummary(
                    pmid=pmid,
                    doi=doi,
                    title=title,
                    authors=author_str,
                    journal=journal,
                    year=year,
                )
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse article: {e}")
                continue

        # Format output
        if not papers:
            return "[search_pubmed] Found PMIDs but failed to parse details."

        lines = [f"[search_pubmed] Found {len(papers)} results:\n"]
        for i, paper in enumerate(papers, 1):
            lines.append(f"--- Result {i} ---")
            lines.append(paper.to_short())

        result = "\n\n".join(lines)
        return truncate_output(result)

    except urllib.error.URLError as e:
        return f"[search_pubmed] Network error: {e.reason}"
    except Exception as e:
        logger.exception("PubMed search failed")
        return f"[search_pubmed] Error: {str(e)}"


# ---------------------------------------------------------------------------
# L2: Get single PubMed abstract by PMID
# ---------------------------------------------------------------------------
def get_pubmed_abstract(pmid: str) -> str:
    """Retrieve full abstract and metadata for a single PubMed article.

    Args:
        pmid: PubMed ID (PMID) as string

    Returns:
        Formatted string with title, authors, journal, year, abstract, and MeSH terms

    Example:
        get_pubmed_abstract("36265145")
    """
    _ensure_env_loaded()
    api_key = os.getenv("NCBI_API_KEY", "")

    try:
        efetch_params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml",
        }
        if api_key:
            efetch_params["api_key"] = api_key

        efetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"{urllib.parse.urlencode(efetch_params)}"
        )

        req = urllib.request.Request(efetch_url)
        with urllib.request.urlopen(req, timeout=30) as response:
            xml_data = response.read().decode()

        root = ET.fromstring(xml_data)
        article = root.find(".//PubmedArticle")

        if article is None:
            return f"[get_pubmed_abstract] PMID {pmid} not found."

        # Extract metadata
        title = article.findtext(".//ArticleTitle") or "No title"

        # Authors
        author_list = article.findall(".//Author")
        if author_list:
            authors = []
            for author in author_list:
                last = author.findtext("LastName") or ""
                init = author.findtext("Initials") or ""
                if last:
                    authors.append(f"{last} {init}".strip())
            author_str = ", ".join(authors)
        else:
            author_str = "No authors listed"

        journal = article.findtext(".//Journal/Title") or article.findtext(
            ".//Journal/ISOAbbreviation"
        )
        year_elem = article.find(".//PubDate/Year")
        year = year_elem.text if year_elem is not None else "Unknown"

        # Abstract
        abstract_texts = article.findall(".//AbstractText")
        if abstract_texts:
            abstract_parts = []
            for abs_text in abstract_texts:
                label = abs_text.get("Label")
                text = abs_text.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = "\n".join(abstract_parts)
        else:
            abstract = "No abstract available"

        # MeSH terms
        mesh_headings = article.findall(".//MeshHeading/DescriptorName")
        mesh_terms = ", ".join([m.text for m in mesh_headings if m.text]) if mesh_headings else "None"

        # DOI
        doi = None
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break

        # Format output
        lines = [
            f"[get_pubmed_abstract] PMID: {pmid}",
            f"Title: {title}",
            f"Authors: {author_str}",
            f"Journal: {journal}",
            f"Year: {year}",
        ]
        if doi:
            lines.append(f"DOI: {doi}")
        lines.append(f"\nAbstract:\n{abstract}")
        lines.append(f"\nMeSH Terms: {mesh_terms}")

        result = "\n".join(lines)
        return truncate_output(result)

    except urllib.error.URLError as e:
        return f"[get_pubmed_abstract] Network error: {e.reason}"
    except Exception as e:
        logger.exception("Failed to fetch PubMed abstract")
        return f"[get_pubmed_abstract] Error: {str(e)}"


# ---------------------------------------------------------------------------
# L3: Semantic Scholar search
# ---------------------------------------------------------------------------
def search_semantic_scholar(
    query: str,
    fields: Optional[str] = None,
    year_range: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Search academic papers via Semantic Scholar API.

    Args:
        query: Search query string
        fields: Comma-separated field list (default: title,authors,year,citationCount,externalIds)
        year_range: Year range filter, e.g., "2020-2023" (optional)
        max_results: Maximum number of results to return (default 10)

    Returns:
        Formatted string with paper summaries including citation counts

    Example:
        search_semantic_scholar("neutralizing antibodies", year_range="2020-2023", max_results=5)
    """
    _ensure_env_loaded()
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

    try:
        # Default fields
        if fields is None:
            fields = "title,authors,year,citationCount,externalIds"

        # Build query parameters
        params = {
            "query": query,
            "fields": fields,
            "limit": str(max_results),
        }
        if year_range:
            params["year"] = year_range

        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urllib.parse.urlencode(params)}"

        # Add API key header if available
        headers = {}
        if api_key:
            headers["x-api-key"] = api_key

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

        papers_data = data.get("data", [])
        if not papers_data:
            return "[search_semantic_scholar] No results found."

        papers = []
        for paper_data in papers_data:
            # Extract authors
            authors_list = paper_data.get("authors", [])
            if authors_list:
                author_names = [a.get("name", "") for a in authors_list[:3]]
                author_str = ", ".join(author_names)
                if len(authors_list) > 3:
                    author_str += ", et al."
            else:
                author_str = "No authors listed"

            # Extract DOI and PMID from externalIds
            external_ids = paper_data.get("externalIds", {}) or {}
            doi = external_ids.get("DOI")
            pmid = external_ids.get("PubMed")

            paper = PaperSummary(
                pmid=pmid,
                doi=doi,
                title=paper_data.get("title", "No title"),
                authors=author_str,
                year=paper_data.get("year"),
                citation_count=paper_data.get("citationCount"),
            )
            papers.append(paper)

        # Format output
        lines = [f"[search_semantic_scholar] Found {len(papers)} results:\n"]
        for i, paper in enumerate(papers, 1):
            lines.append(f"--- Result {i} ---")
            lines.append(paper.to_short())

        result = "\n\n".join(lines)
        return truncate_output(result)

    except urllib.error.URLError as e:
        return f"[search_semantic_scholar] Network error: {e.reason}"
    except Exception as e:
        logger.exception("Semantic Scholar search failed")
        return f"[search_semantic_scholar] Error: {str(e)}"


# ---------------------------------------------------------------------------
# L4: Get paper citations from Semantic Scholar
# ---------------------------------------------------------------------------
def get_paper_citations(
    paper_id: str, direction: str = "citations", max_results: int = 20
) -> str:
    """Retrieve citations or references for a paper via Semantic Scholar.

    Args:
        paper_id: Semantic Scholar paper ID, DOI, or PMID (prefix with "PMID:")
        direction: "citations" (papers citing this) or "references" (papers cited by this)
        max_results: Maximum number of results to return (default 20)

    Returns:
        Formatted string with citing/cited papers

    Example:
        get_paper_citations("10.1038/s41586-020-2012-7", direction="citations", max_results=10)
        get_paper_citations("PMID:32015508", direction="references")
    """
    _ensure_env_loaded()
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

    if direction not in ["citations", "references"]:
        return "[get_paper_citations] Error: direction must be 'citations' or 'references'"

    try:
        # Build URL
        fields = "title,authors,year,citationCount,externalIds"
        params = {"fields": fields, "limit": str(max_results)}
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(paper_id, safe='')}/"
            f"{direction}?{urllib.parse.urlencode(params)}"
        )

        # Add API key header if available
        headers = {}
        if api_key:
            headers["x-api-key"] = api_key

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

        papers_data = data.get("data", [])
        if not papers_data:
            return f"[get_paper_citations] No {direction} found for paper {paper_id}."

        papers = []
        for item in papers_data:
            # The API wraps each citation in a "citingPaper" or "citedPaper" field
            paper_data = item.get("citingPaper") or item.get("citedPaper") or item

            # Extract authors
            authors_list = paper_data.get("authors", [])
            if authors_list:
                author_names = [a.get("name", "") for a in authors_list[:3]]
                author_str = ", ".join(author_names)
                if len(authors_list) > 3:
                    author_str += ", et al."
            else:
                author_str = "No authors listed"

            # Extract DOI and PMID
            external_ids = paper_data.get("externalIds", {}) or {}
            doi = external_ids.get("DOI")
            pmid = external_ids.get("PubMed")

            paper = PaperSummary(
                pmid=pmid,
                doi=doi,
                title=paper_data.get("title", "No title"),
                authors=author_str,
                year=paper_data.get("year"),
                citation_count=paper_data.get("citationCount"),
            )
            papers.append(paper)

        # Format output
        direction_label = "citing" if direction == "citations" else "referenced by"
        lines = [f"[get_paper_citations] Found {len(papers)} papers {direction_label} {paper_id}:\n"]
        for i, paper in enumerate(papers, 1):
            lines.append(f"--- Result {i} ---")
            lines.append(paper.to_short())

        result = "\n\n".join(lines)
        return truncate_output(result)

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"[get_paper_citations] Paper {paper_id} not found."
        return f"[get_paper_citations] HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"[get_paper_citations] Network error: {e.reason}"
    except Exception as e:
        logger.exception("Failed to fetch paper citations")
        return f"[get_paper_citations] Error: {str(e)}"


# ---------------------------------------------------------------------------
# L5: Search preprints via rxivist API
# ---------------------------------------------------------------------------
def search_preprints(query: str, server: str = "biorxiv", max_results: int = 10) -> str:
    """Search preprint servers (BioRxiv/MedRxiv) via rxivist API.

    Args:
        query: Search query string
        server: Preprint server ("biorxiv" or "medrxiv", default "biorxiv")
        max_results: Maximum number of results to return (default 10)

    Returns:
        Formatted string with preprint summaries

    Example:
        search_preprints("SARS-CoV-2 variants", server="biorxiv", max_results=5)

    Note:
        Uses the rxivist.org API which indexes BioRxiv and MedRxiv preprints.
    """
    try:
        # rxivist API search endpoint
        params = {
            "q": query,
            "metric": "downloads",
            "page_size": str(max_results),
        }

        url = f"https://api.rxivist.org/v1/papers?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

        results = data.get("results", [])
        if not results:
            return "[search_preprints] No results found."

        papers = []
        for paper_data in results:
            # Extract authors
            authors_list = paper_data.get("authors", [])
            if authors_list:
                author_names = [a.get("name", "") for a in authors_list[:3]]
                author_str = ", ".join(author_names)
                if len(authors_list) > 3:
                    author_str += ", et al."
            else:
                author_str = "No authors listed"

            # Extract DOI
            doi = paper_data.get("doi")

            # Extract year from posted date
            posted = paper_data.get("first_posted")
            year = None
            if posted:
                try:
                    year = int(posted.split("-")[0])
                except (ValueError, IndexError):
                    pass

            paper = PaperSummary(
                doi=doi,
                title=paper_data.get("title", "No title"),
                authors=author_str,
                year=year,
                citation_count=paper_data.get("metric_value"),  # Using downloads as citation proxy
            )
            papers.append(paper)

        # Format output
        lines = [f"[search_preprints] Found {len(papers)} preprints on {server}:\n"]
        for i, paper in enumerate(papers, 1):
            lines.append(f"--- Result {i} ---")
            lines.append(paper.to_short())

        result = "\n\n".join(lines)
        return truncate_output(result)

    except urllib.error.URLError as e:
        return f"[search_preprints] Network error: {e.reason}"
    except Exception as e:
        logger.exception("Preprint search failed")
        return f"[search_preprints] Error: {str(e)}"


# ---------------------------------------------------------------------------
# L6: Europe PMC search
# ---------------------------------------------------------------------------
def search_europe_pmc(
    query: str, source: Optional[str] = None, max_results: int = 10
) -> str:
    """Search Europe PMC database for publications and preprints.

    Args:
        query: Search query string (supports Europe PMC query syntax)
        source: Source filter (e.g., "PPR" for preprints, "MED" for PubMed, optional)
        max_results: Maximum number of results to return (default 10)

    Returns:
        Formatted string with paper summaries including abstracts

    Example:
        search_europe_pmc("immunology", source="PPR", max_results=5)
        search_europe_pmc("COVID-19 vaccine effectiveness")

    Note:
        Source codes: PPR (preprints), MED (PubMed), PMC (PubMed Central), etc.
    """
    try:
        # Build query with source filter
        search_query = query
        if source:
            search_query = f"{query} AND SRC:{source}"

        params = {
            "query": search_query,
            "format": "json",
            "pageSize": str(max_results),
            "resultType": "core",
        }

        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

        results_list = data.get("resultList", {}).get("result", [])
        if not results_list:
            return "[search_europe_pmc] No results found."

        papers = []
        for paper_data in results_list:
            # Extract authors
            author_string = paper_data.get("authorString", "")
            if author_string:
                author_parts = author_string.split(", ")
                if len(author_parts) > 3:
                    author_str = ", ".join(author_parts[:3]) + ", et al."
                else:
                    author_str = author_string
            else:
                author_str = "No authors listed"

            # Extract identifiers
            pmid = paper_data.get("pmid")
            doi = paper_data.get("doi")

            # Extract year
            pub_year = paper_data.get("pubYear")
            try:
                year = int(pub_year) if pub_year else None
            except ValueError:
                year = None

            # Journal
            journal = paper_data.get("journalTitle") or paper_data.get("bookOrReportDetails", {}).get(
                "publisher"
            )

            paper = PaperSummary(
                pmid=pmid,
                doi=doi,
                title=paper_data.get("title", "No title"),
                authors=author_str,
                journal=journal,
                year=year,
            )
            papers.append(paper)

        # Format output with abstracts
        source_label = f" (source: {source})" if source else ""
        lines = [f"[search_europe_pmc] Found {len(papers)} results{source_label}:\n"]

        for i, (paper, paper_data) in enumerate(zip(papers, results_list), 1):
            lines.append(f"--- Result {i} ---")
            lines.append(paper.to_short())

            # Add abstract if available
            abstract = paper_data.get("abstractText")
            if abstract:
                # Truncate long abstracts
                if len(abstract) > 500:
                    abstract = abstract[:500] + "..."
                lines.append(f"Abstract: {abstract}")

        result = "\n\n".join(lines)
        return truncate_output(result)

    except urllib.error.URLError as e:
        return f"[search_europe_pmc] Network error: {e.reason}"
    except Exception as e:
        logger.exception("Europe PMC search failed")
        return f"[search_europe_pmc] Error: {str(e)}"


# ---------------------------------------------------------------------------
# Export function
# ---------------------------------------------------------------------------
def get_literature_tools() -> dict:
    """Return dictionary of all literature search tool functions.

    Returns:
        Dict mapping tool names to function references
    """
    return {
        "search_pubmed": search_pubmed,
        "get_pubmed_abstract": get_pubmed_abstract,
        "search_semantic_scholar": search_semantic_scholar,
        "get_paper_citations": get_paper_citations,
        "search_preprints": search_preprints,
        "search_europe_pmc": search_europe_pmc,
    }
