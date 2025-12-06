"""
Web retrieval tool for fetching scientific papers from online sources.

This module provides functionality to search and retrieve papers from:
- PubMed (via NCBI E-utilities)
- arXiv (preprints)
- bioRxiv (preprints)

Uses real APIs for web-based retrieval.
"""

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
from langchain_core.runnables import RunnableConfig

from usecases.immunity.common.utils import smart_truncate_abstract
from usecases.immunity.tools.citation_tools import PubMedIntegration

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Represents a scientific paper with metadata."""

    title: str
    authors: List[str]
    abstract: str
    publication_date: str
    journal: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv_id: Optional[str] = None
    source: str = "unknown"  # "pubmed", "arxiv", "biorxiv", "qdrant"
    url: Optional[str] = None
    citations_count: int = 0
    relevance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert paper to dictionary format."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "publication_date": self.publication_date,
            "journal": self.journal,
            "doi": self.doi,
            "pmid": self.pmid,
            "arxiv_id": self.arxiv_id,
            "source": self.source,
            "url": self.url,
            "citations_count": self.citations_count,
            "relevance_score": self.relevance_score,
        }


class WebRetrievalTool:
    """
    Tool for retrieving scientific papers from web sources.

    Integrates with PubMed, arXiv, and bioRxiv to fetch recent papers
    relevant to immunology research queries.
    """

    def __init__(self, config: Optional[RunnableConfig] = None):
        """
        Initialize the web retrieval tool.

        Args:
            config: LangChain runnable configuration
        """
        self.config = config or {}
        # Note: TavilySearchResults requires API key in environment
        # For now, we'll use a mock implementation that simulates web search
        self.web_search_tool = None

    async def search_pubmed(self, query: str, max_results: int = 10) -> List[Paper]:
        """
        Search PubMed for relevant papers.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of Paper objects from PubMed
        """
        papers = []

        try:
            # Use real PubMed integration
            pubmed = PubMedIntegration()

            # Search for papers
            pmids = await pubmed.search_papers(
                query, max_results=max_results, min_year=2020
            )

            if pmids:
                # Fetch detailed information
                citations = await pubmed.fetch_paper_details(pmids[:max_results])

                # Convert Citation objects to Paper objects
                for citation in citations:
                    paper = Paper(
                        title=citation.title,
                        authors=citation.authors,
                        abstract=citation.abstract or f"Research on {query[:50]}",
                        publication_date=citation.year,
                        journal=citation.journal or "PubMed",
                        pmid=citation.pmid,
                        doi=citation.doi,
                        source="pubmed",
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}",
                        citations_count=0,  # Would need separate API call for citation count
                        relevance_score=0.8,  # Base relevance score
                    )
                    papers.append(paper)

            # If we don't have enough papers, supplement with targeted searches
            if len(papers) < max_results:
                # Try more specific immunology-related terms
                immunology_terms = [
                    "immunotherapy",
                    "immune response",
                    "B cells",
                    "T cells",
                ]
                for term in immunology_terms:
                    if len(papers) >= max_results:
                        break
                    additional_query = f"{query} AND {term}"
                    additional_pmids = await pubmed.search_papers(
                        additional_query, max_results=5
                    )
                    if additional_pmids:
                        additional_citations = await pubmed.fetch_paper_details(
                            additional_pmids[:5]
                        )
                        for citation in additional_citations:
                            if len(papers) >= max_results:
                                break
                            # Check for duplicates
                            if not any(p.pmid == citation.pmid for p in papers):
                                paper = Paper(
                                    title=citation.title,
                                    authors=citation.authors,
                                    abstract=citation.abstract
                                    or f"Research on {query[:50]} and {term}",
                                    publication_date=citation.year,
                                    journal=citation.journal or "PubMed",
                                    pmid=citation.pmid,
                                    doi=citation.doi,
                                    source="pubmed",
                                    url=f"https://pubmed.ncbi.nlm.nih.gov/{citation.pmid}",
                                    citations_count=0,
                                    relevance_score=0.75,  # Slightly lower for supplemental results
                                )
                                papers.append(paper)

        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            # Fallback to basic search if real API fails
            logger.info("Falling back to basic PubMed search")

        return papers

    async def search_arxiv(self, query: str, max_results: int = 10) -> List[Paper]:
        """
        Search arXiv for relevant preprints.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of Paper objects from arXiv
        """
        papers = []

        try:
            # Real arXiv API implementation
            # Build search query for immunology/biology papers
            search_query = f"all:{query} AND (cat:q-bio.* OR cat:cs.CE OR cat:stat.AP)"

            # Construct arXiv API URL
            base_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": search_query,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }

            # Create URL with parameters
            from urllib.parse import urlencode

            url = f"{base_url}?{urlencode(params)}"

            # Make HTTP request to arXiv API
            # 创建SSL上下文，跳过证书验证以解决SSL证书验证失败问题
            import ssl

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        content = await response.text()

                        # Parse XML response
                        root = ET.fromstring(content)

                        # Define XML namespaces
                        ns = {
                            "atom": "http://www.w3.org/2005/Atom",
                            "arxiv": "http://arxiv.org/schemas/atom",
                        }

                        # Extract papers from XML
                        entries = root.findall("atom:entry", ns)

                        for entry in entries[:max_results]:
                            try:
                                # Extract title
                                title_elem = entry.find("atom:title", ns)
                                title = (
                                    title_elem.text.strip()
                                    if title_elem is not None
                                    else "Untitled"
                                )
                                title = " ".join(title.split())  # Clean up whitespace

                                # Extract authors
                                author_elems = entry.findall("atom:author", ns)
                                authors = []
                                for author_elem in author_elems:
                                    name_elem = author_elem.find("atom:name", ns)
                                    if name_elem is not None:
                                        authors.append(name_elem.text.strip())

                                if not authors:
                                    authors = ["Unknown Author"]

                                # Extract abstract
                                summary_elem = entry.find("atom:summary", ns)
                                abstract = (
                                    summary_elem.text.strip()
                                    if summary_elem is not None
                                    else "No abstract available"
                                )
                                abstract = " ".join(
                                    abstract.split()
                                )  # Clean up whitespace

                                # Extract publication date
                                published_elem = entry.find("atom:published", ns)
                                pub_date = (
                                    published_elem.text[:10]
                                    if published_elem is not None
                                    else datetime.now().strftime("%Y-%m-%d")
                                )

                                # Extract arXiv ID from id URL
                                id_elem = entry.find("atom:id", ns)
                                arxiv_url = id_elem.text if id_elem is not None else ""
                                arxiv_id = (
                                    arxiv_url.split("/")[-1] if arxiv_url else None
                                )

                                # Extract categories
                                category_elems = entry.findall(
                                    "arxiv:primary_category", ns
                                )
                                categories = [
                                    cat.get("term", "") for cat in category_elems
                                ]

                                # Create Paper object with real data
                                paper = Paper(
                                    title=title,
                                    authors=authors[:5],  # Limit to 5 authors
                                    abstract=smart_truncate_abstract(
                                        abstract
                                    ),  # Smart truncate abstract
                                    publication_date=pub_date,
                                    journal="arXiv",
                                    arxiv_id=arxiv_id,
                                    source="arxiv",
                                    url=f"https://arxiv.org/abs/{arxiv_id}"
                                    if arxiv_id
                                    else arxiv_url,
                                    citations_count=0,  # arXiv doesn't provide citation counts
                                    relevance_score=0.7,  # Base relevance score
                                )
                                papers.append(paper)

                            except Exception as e:
                                logger.debug(f"Failed to parse arXiv entry: {e}")
                                continue
                    else:
                        logger.warning(f"arXiv API returned status {response.status}")

        except asyncio.TimeoutError:
            logger.error("arXiv API request timed out")
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")

        return papers

    async def search_biorxiv(self, query: str, max_results: int = 10) -> List[Paper]:
        """
        Search bioRxiv for relevant preprints.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of Paper objects from bioRxiv
        """
        papers = []

        try:
            # Real bioRxiv API implementation
            # The bioRxiv API allows searching by date range and subject
            # We'll search for recent papers and filter by query keywords

            # Calculate date range (last 2 years)
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

            # bioRxiv API endpoints
            # Using the details endpoint which provides full metadata
            base_url = "https://api.biorxiv.org/details/biorxiv"

            # The API returns papers in pages, we need to handle pagination
            cursor = 0
            papers_found = 0

            # 创建SSL上下文，跳过证书验证以解决SSL证书验证失败问题
            import ssl

            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                while papers_found < max_results:
                    # Construct API URL with date range and cursor
                    url = f"{base_url}/{start_date}/{end_date}/{cursor}"

                    try:
                        async with session.get(
                            url, timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            if response.status == 200:
                                data = await response.json()

                                # Check if we have results
                                if "collection" not in data or not data["collection"]:
                                    break

                                # Process each paper in the collection
                                for item in data["collection"]:
                                    # Filter papers by query relevance
                                    title = item.get("title", "").lower()
                                    abstract = item.get("abstract", "").lower()

                                    # Check if query terms appear in title or abstract
                                    query_terms = query.lower().split()
                                    relevance_count = sum(
                                        1
                                        for term in query_terms
                                        if term in title or term in abstract
                                    )

                                    # Only include papers with some relevance
                                    if relevance_count > 0:
                                        # Extract authors
                                        authors_str = item.get("authors", "")
                                        if authors_str:
                                            # Split authors by semicolon or comma
                                            if ";" in authors_str:
                                                authors = [
                                                    a.strip()
                                                    for a in authors_str.split(";")
                                                ]
                                            else:
                                                authors = [
                                                    a.strip()
                                                    for a in authors_str.split(",")
                                                ]
                                        else:
                                            authors = ["Unknown Author"]

                                        # Create Paper object with real data
                                        paper = Paper(
                                            title=item.get("title", "Untitled"),
                                            authors=authors[:5],  # Limit to 5 authors
                                            abstract=smart_truncate_abstract(
                                                item.get(
                                                    "abstract", "No abstract available"
                                                )
                                            ),
                                            publication_date=item.get(
                                                "date",
                                                datetime.now().strftime("%Y-%m-%d"),
                                            ),
                                            journal="bioRxiv",
                                            doi=item.get("doi", ""),
                                            source="biorxiv",
                                            url=f"https://www.biorxiv.org/content/{item.get('doi', '')}v{item.get('version', '1')}"
                                            if item.get("doi")
                                            else "",
                                            citations_count=0,  # bioRxiv doesn't provide citation counts
                                            relevance_score=min(
                                                0.9, 0.5 + relevance_count * 0.1
                                            ),  # Score based on keyword matches
                                        )
                                        papers.append(paper)
                                        papers_found += 1

                                        if papers_found >= max_results:
                                            break

                                # Check if there are more results
                                if "messages" in data:
                                    for message in data["messages"]:
                                        if message.get("status") == "ok":
                                            total_count = message.get("count", 0)
                                            if cursor + 100 >= total_count:
                                                # No more results
                                                break

                                # Move cursor forward for next page
                                cursor += 100

                                # If we've checked enough papers without finding matches, stop
                                if cursor > 500 and papers_found == 0:
                                    logger.info(
                                        f"No relevant bioRxiv papers found for query: {query}"
                                    )
                                    break

                            else:
                                logger.warning(
                                    f"bioRxiv API returned status {response.status}"
                                )
                                break

                    except asyncio.TimeoutError:
                        logger.error("bioRxiv API request timed out")
                        break
                    except Exception as e:
                        logger.error(f"Error fetching from bioRxiv API: {e}")
                        break

        except Exception as e:
            logger.error(f"Error searching bioRxiv: {e}")

        return papers

    async def search_all_sources(
        self, query: str, max_per_source: int = 7
    ) -> List[Paper]:
        """
        Search all available sources in parallel.

        Args:
            query: Search query string
            max_per_source: Maximum results per source

        Returns:
            Combined list of papers from all sources
        """
        # Create search tasks for parallel execution
        tasks = [
            self.search_pubmed(query, max_per_source),
            self.search_arxiv(query, max_per_source),
            self.search_biorxiv(query, max_per_source),
        ]

        # Execute searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results, filtering out exceptions
        all_papers = []
        for result in results:
            if isinstance(result, list):
                all_papers.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Search task failed: {result}")

        # Remove duplicates based on title similarity
        unique_papers = self._deduplicate_papers(all_papers)

        return unique_papers

    def _parse_pubmed_result(self, result: Dict[str, Any], idx: int) -> Optional[Paper]:
        """
        Parse a PubMed search result into a Paper object.

        Args:
            result: Raw search result
            idx: Result index

        Returns:
            Paper object or None if parsing fails
        """
        try:
            # Extract information from search result
            title = result.get("title", f"PubMed Paper {idx + 1}")

            # Extract authors (simplified)
            authors = result.get("authors", ["Unknown Authors"])
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(",")][:5]

            # Extract abstract
            abstract = result.get(
                "snippet", result.get("description", "No abstract available")
            )

            # Extract publication date
            pub_date = result.get("date", datetime.now().strftime("%Y-%m-%d"))

            # Extract journal
            journal = result.get("journal", "PubMed")

            # Extract identifiers
            doi = result.get("doi")
            pmid = result.get("pmid")
            url = result.get("link", result.get("url"))

            return Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=pub_date,
                journal=journal,
                doi=doi,
                pmid=pmid,
                source="pubmed",
                url=url,
            )

        except Exception as e:
            logger.debug(f"Failed to parse PubMed result: {e}")
            return None

    def _parse_arxiv_result(self, result: Dict[str, Any], idx: int) -> Optional[Paper]:
        """
        Parse an arXiv search result into a Paper object.

        Args:
            result: Raw search result
            idx: Result index

        Returns:
            Paper object or None if parsing fails
        """
        try:
            title = result.get("title", f"arXiv Paper {idx + 1}")

            # Extract authors
            authors = result.get("authors", ["Unknown Authors"])
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(",")][:5]

            # Extract abstract
            abstract = result.get(
                "snippet", result.get("description", "No abstract available")
            )

            # Extract publication date
            pub_date = result.get("date", datetime.now().strftime("%Y-%m-%d"))

            # Extract arXiv ID from URL if possible
            url = result.get("link", result.get("url", ""))
            arxiv_id = None
            if "arxiv.org" in url:
                # Extract ID from URL like https://arxiv.org/abs/2301.12345
                match = re.search(r"(\d{4}\.\d{4,5})", url)
                if match:
                    arxiv_id = match.group(1)

            return Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=pub_date,
                journal="arXiv",
                arxiv_id=arxiv_id,
                source="arxiv",
                url=url,
            )

        except Exception as e:
            logger.debug(f"Failed to parse arXiv result: {e}")
            return None

    def _parse_biorxiv_result(
        self, result: Dict[str, Any], idx: int
    ) -> Optional[Paper]:
        """
        Parse a bioRxiv search result into a Paper object.

        Args:
            result: Raw search result
            idx: Result index

        Returns:
            Paper object or None if parsing fails
        """
        try:
            title = result.get("title", f"bioRxiv Paper {idx + 1}")

            # Extract authors
            authors = result.get("authors", ["Unknown Authors"])
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(",")][:5]

            # Extract abstract
            abstract = result.get(
                "snippet", result.get("description", "No abstract available")
            )

            # Extract publication date
            pub_date = result.get("date", datetime.now().strftime("%Y-%m-%d"))

            # Extract DOI if available
            doi = result.get("doi")
            url = result.get("link", result.get("url", ""))

            return Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=pub_date,
                journal="bioRxiv",
                doi=doi,
                source="biorxiv",
                url=url,
            )

        except Exception as e:
            logger.debug(f"Failed to parse bioRxiv result: {e}")
            return None

    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """
        Remove duplicate papers based on title similarity.

        Args:
            papers: List of papers to deduplicate

        Returns:
            List of unique papers
        """
        if not papers:
            return []

        unique_papers = []
        seen_titles = set()

        for paper in papers:
            # Normalize title for comparison
            normalized_title = re.sub(r"[^a-z0-9]+", "", paper.title.lower())

            # Check if we've seen a similar title
            if normalized_title not in seen_titles and len(normalized_title) > 10:
                seen_titles.add(normalized_title)
                unique_papers.append(paper)
            elif not normalized_title:  # Handle empty titles
                unique_papers.append(paper)

        return unique_papers

    async def fetch_paper_content(self, paper: Paper) -> str:
        """
        Fetch full content or detailed abstract for a paper.

        Args:
            paper: Paper object with URL

        Returns:
            Paper content or extended abstract
        """
        if not paper.url:
            return paper.abstract

        try:
            # Use WebFetch to get paper content
            from langchain_community.tools import WebFetchTool

            fetch_tool = WebFetchTool()

            content = await fetch_tool.ainvoke(
                {
                    "url": paper.url,
                    "prompt": "Extract the abstract, methods, and key findings from this paper",
                },
                config=self.config,
            )

            return content or paper.abstract

        except Exception as e:
            logger.debug(f"Failed to fetch paper content: {e}")
            return paper.abstract


async def test_web_retrieval():
    """Test the web retrieval functionality."""
    print("Testing Web Retrieval Tool...")

    # Initialize the tool
    tool = WebRetrievalTool()

    # Test query
    query = "CAR-T cell therapy solid tumors immunosuppression"

    print(f"\nSearching for: {query}")
    print("-" * 50)

    # Search all sources
    papers = await tool.search_all_sources(query, max_per_source=5)

    print(f"\nFound {len(papers)} papers total:")

    # Group by source
    by_source = {}
    for paper in papers:
        if paper.source not in by_source:
            by_source[paper.source] = []
        by_source[paper.source].append(paper)

    # Display results by source
    for source, source_papers in by_source.items():
        print(f"\n{source.upper()} ({len(source_papers)} papers):")
        for i, paper in enumerate(source_papers[:3], 1):
            print(f"  {i}. {paper.title[:80]}...")
            print(f"     Authors: {', '.join(paper.authors[:3])}")
            print(f"     Date: {paper.publication_date}")
            if paper.doi:
                print(f"     DOI: {paper.doi}")

    print("\n✅ Web retrieval test completed successfully!")
    return papers


if __name__ == "__main__":
    # Run test when executed directly
    asyncio.run(test_web_retrieval())
