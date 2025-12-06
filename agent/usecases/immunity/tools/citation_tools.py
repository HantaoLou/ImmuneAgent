"""
Citation management tools for Cell Experiment Planning Agent.

This module provides tools for extracting, formatting, and managing citations
from retrieved documents and scientific literature. Includes:
- Real-time PubMed integration for scientific paper search
- Citation quality ranking by journal impact and relevance
- Multiple citation format support (APA, BibTeX, Vancouver)
- Literature backing for experimental protocols
"""

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Citation(BaseModel):
    """Represents a scientific citation."""

    # 配置Pydantic模型以优雅处理序列化警告
    model_config = ConfigDict(
        # 在序列化时允许类型强制转换，避免警告
        ser_json_inf=float("inf"),  # 处理无穷大值
        ser_json_nan=float("nan"),  # 处理NaN值
        # 允许在序列化时进行类型转换
        arbitrary_types_allowed=True,
        # 在序列化时使用更宽松的模式
        validate_assignment=False,
    )

    authors: Union[List[str], str] = Field(default_factory=list)
    title: str = ""
    journal: str = ""
    year: Union[int, str] = 0
    volume: str = ""
    pages: str = ""
    doi: str = ""
    pmid: str = ""
    abstract: str = ""
    citation_key: str = ""  # Unique identifier

    @field_validator("authors", mode="before")
    @classmethod
    def validate_authors(cls, v):
        """将authors字段转换为字符串列表"""
        if isinstance(v, str):
            # 如果是字符串，尝试按常见分隔符分割
            if "," in v:
                return [author.strip() for author in v.split(",")]
            elif ";" in v:
                return [author.strip() for author in v.split(";")]
            else:
                return [v.strip()]
        elif isinstance(v, list):
            return [str(author) for author in v]
        return []

    @field_validator("year", mode="before")
    @classmethod
    def validate_year(cls, v):
        """将year字段转换为整数"""
        if isinstance(v, str):
            try:
                return int(v)
            except (ValueError, TypeError):
                return 0
        elif isinstance(v, int):
            return v
        return 0

    def to_apa(self) -> str:
        """Format citation in APA style."""
        author_str = self._format_authors_apa()

        citation = f"{author_str} ({self.year}). {self.title}. "

        if self.journal:
            citation += f"*{self.journal}*"
            if self.volume:
                citation += f", *{self.volume}*"
            if self.pages:
                citation += f", {self.pages}"

        if self.doi:
            citation += f". https://doi.org/{self.doi}"

        return citation

    def to_bibtex(self) -> str:
        """Format citation in BibTeX format."""
        entry_type = "article"

        bibtex = f"@{entry_type}{{{self.citation_key},\n"

        if self.authors:
            author_str = " and ".join(self.authors)
            bibtex += f"  author = {{{author_str}}},\n"

        if self.title:
            bibtex += f"  title = {{{self.title}}},\n"

        if self.journal:
            bibtex += f"  journal = {{{self.journal}}},\n"

        if self.year:
            bibtex += f"  year = {{{self.year}}},\n"

        if self.volume:
            bibtex += f"  volume = {{{self.volume}}},\n"

        if self.pages:
            bibtex += f"  pages = {{{self.pages}}},\n"

        if self.doi:
            bibtex += f"  doi = {{{self.doi}}},\n"

        bibtex += "}"

        return bibtex

    def to_vancouver(self) -> str:
        """Format citation in Vancouver style."""
        author_str = self._format_authors_vancouver()

        citation = f"{author_str}. {self.title}. {self.journal}. "

        if self.year:
            citation += f"{self.year}"

        if self.volume:
            citation += f";{self.volume}"

        if self.pages:
            citation += f":{self.pages}"

        return citation + "."

    def _format_authors_apa(self) -> str:
        """Format authors in APA style."""
        if not self.authors:
            return "Unknown"

        if len(self.authors) == 1:
            return self._format_single_author_apa(self.authors[0])
        elif len(self.authors) == 2:
            return f"{self._format_single_author_apa(self.authors[0])} & {self._format_single_author_apa(self.authors[1])}"
        else:
            formatted = [
                self._format_single_author_apa(author) for author in self.authors[:6]
            ]
            if len(self.authors) > 7:
                return (
                    ", ".join(formatted[:6])
                    + ", ... "
                    + self._format_single_author_apa(self.authors[-1])
                )
            else:
                return ", ".join(formatted[:-1]) + ", & " + formatted[-1]

    def _format_single_author_apa(self, author: str) -> str:
        """Format a single author in APA style."""
        parts = author.split()
        if len(parts) >= 2:
            return f"{parts[-1]}, {'. '.join([p[0] for p in parts[:-1]])}."
        return author

    def _format_authors_vancouver(self) -> str:
        """Format authors in Vancouver style."""
        if not self.authors:
            return "Unknown"

        formatted = []
        for author in self.authors[:6]:
            parts = author.split()
            if len(parts) >= 2:
                formatted.append(f"{parts[-1]} {' '.join([p[0] for p in parts[:-1]])}")
            else:
                formatted.append(author)

        if len(self.authors) > 6:
            return ", ".join(formatted) + ", et al"

        return ", ".join(formatted)


class CitationExtractor:
    """Extracts citations from documents."""

    def __init__(self):
        """Initialize the citation extractor."""
        self.doi_pattern = re.compile(r"10\.\d{4,}/[-._;()/:\w]+")
        self.pmid_pattern = re.compile(r"PMID:?\s*(\d+)", re.IGNORECASE)
        self.year_pattern = re.compile(r"\b(19|20)\d{2}\b")

    def extract_from_document(self, document: Dict[str, Any]) -> Citation:
        """
        Extract citation information from a document.

        Args:
            document: Document with metadata

        Returns:
            Extracted citation
        """
        citation = Citation()

        # Extract from metadata if available
        metadata = document.get("metadata", {})

        # Authors
        if "authors" in metadata:
            citation.authors = self._parse_authors(metadata["authors"])
        elif "author" in metadata:
            citation.authors = self._parse_authors(metadata["author"])

        # Title
        citation.title = metadata.get("title", document.get("title", ""))

        # Journal
        citation.journal = metadata.get("journal", metadata.get("source", ""))

        # Year
        if "year" in metadata:
            citation.year = self._parse_year(metadata["year"])
        elif "date" in metadata:
            citation.year = self._parse_year(metadata["date"])
        else:
            # Try to extract from content
            content = document.get("content", "")
            years = self.year_pattern.findall(content)
            if years:
                citation.year = int(years[0])

        # Volume and pages
        citation.volume = str(metadata.get("volume", ""))
        citation.pages = str(metadata.get("pages", ""))

        # DOI
        if "doi" in metadata:
            citation.doi = metadata["doi"]
        else:
            # Try to extract from content
            content = document.get("content", "")
            dois = self.doi_pattern.findall(content)
            if dois:
                citation.doi = dois[0]

        # PMID
        if "pmid" in metadata:
            citation.pmid = str(metadata["pmid"])
        else:
            # Try to extract from content
            content = document.get("content", "")
            pmids = self.pmid_pattern.findall(content)
            if pmids:
                citation.pmid = pmids[0]

        # Abstract
        citation.abstract = metadata.get("abstract", document.get("abstract", ""))

        # Generate citation key
        citation.citation_key = self._generate_citation_key(citation)

        return citation

    def extract_from_text(self, text: str) -> List[Citation]:
        """
        Extract citations from free text.

        Args:
            text: Text containing citations

        Returns:
            List of extracted citations
        """
        citations = []

        # Look for DOIs
        dois = self.doi_pattern.findall(text)
        for doi in dois:
            citation = Citation(doi=doi)
            citation.citation_key = f"doi_{doi.replace('/', '_').replace('.', '_')}"
            citations.append(citation)

        # Look for PMIDs
        pmids = self.pmid_pattern.findall(text)
        for pmid in pmids:
            citation = Citation(pmid=pmid)
            citation.citation_key = f"pmid_{pmid}"
            citations.append(citation)

        # Try to parse inline citations (e.g., "Smith et al., 2023")
        inline_pattern = re.compile(
            r"([A-Z][a-z]+(?:\s+et\s+al\.?)?),?\s+\(?((?:19|20)\d{2})\)?"
        )
        inline_citations = inline_pattern.findall(text)

        for author, year in inline_citations:
            citation = Citation(
                authors=[author],
                year=int(year),
                citation_key=f"{author.replace(' ', '_')}_{year}",
            )
            citations.append(citation)

        return citations

    def _parse_authors(self, authors_data: Any) -> List[str]:
        """Parse authors from various formats."""
        if isinstance(authors_data, list):
            return authors_data
        elif isinstance(authors_data, str):
            # Split by common delimiters
            if ";" in authors_data:
                return [a.strip() for a in authors_data.split(";")]
            elif "," in authors_data and " and " in authors_data:
                # Handle "Author1, Author2 and Author3" format
                parts = authors_data.replace(" and ", ", ").split(",")
                return [p.strip() for p in parts]
            elif "," in authors_data:
                return [a.strip() for a in authors_data.split(",")]
            else:
                return [authors_data.strip()]
        else:
            return []

    def _parse_year(self, year_data: Any) -> int:
        """Parse year from various formats."""
        if isinstance(year_data, int):
            return year_data
        elif isinstance(year_data, str):
            # Extract year from date string
            years = self.year_pattern.findall(year_data)
            if years:
                return int(years[0])
        return 0

    def _generate_citation_key(self, citation: Citation) -> str:
        """Generate a unique citation key."""
        if citation.doi:
            return f"doi_{citation.doi.replace('/', '_').replace('.', '_')}"
        elif citation.pmid:
            return f"pmid_{citation.pmid}"
        elif citation.authors and citation.year:
            first_author = (
                citation.authors[0].split()[-1] if citation.authors else "unknown"
            )
            return f"{first_author}_{citation.year}"
        else:
            import hashlib

            content = f"{citation.title}{citation.journal}{citation.year}"
            return f"ref_{hashlib.md5(content.encode()).hexdigest()[:8]}"


class CitationManager:
    """Manages citations throughout the workflow."""

    def __init__(self):
        """Initialize the citation manager."""
        self.citations: Dict[str, Citation] = {}
        self.extractor = CitationExtractor()

    def add_citation(self, citation: Citation) -> str:
        """
        Add a citation to the manager.

        Args:
            citation: Citation to add

        Returns:
            Citation key
        """
        if not citation.citation_key:
            citation.citation_key = self._generate_unique_key(citation)

        self.citations[citation.citation_key] = citation
        return citation.citation_key

    def add_from_document(self, document: Dict[str, Any]) -> str:
        """
        Add citation from a document.

        Args:
            document: Document with metadata

        Returns:
            Citation key
        """
        citation = self.extractor.extract_from_document(document)
        return self.add_citation(citation)

    def add_from_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add citations from multiple documents.

        Args:
            documents: List of documents

        Returns:
            List of citation keys
        """
        keys = []
        for doc in documents:
            keys.append(self.add_from_document(doc))
        return keys

    def get_citation(self, key: str) -> Optional[Citation]:
        """Get a citation by key."""
        return self.citations.get(key)

    def format_citations(
        self, keys: Optional[List[str]] = None, style: str = "apa"
    ) -> List[str]:
        """
        Format citations in specified style.

        Args:
            keys: Citation keys to format (None for all)
            style: Citation style (apa, bibtex, vancouver)

        Returns:
            List of formatted citations
        """
        if keys is None:
            keys = list(self.citations.keys())

        formatted = []
        for key in keys:
            citation = self.citations.get(key)
            if citation:
                if style == "apa":
                    formatted.append(citation.to_apa())
                elif style == "bibtex":
                    formatted.append(citation.to_bibtex())
                elif style == "vancouver":
                    formatted.append(citation.to_vancouver())
                else:
                    formatted.append(str(citation))

        return formatted

    def generate_bibliography(
        self, keys: Optional[List[str]] = None, style: str = "apa"
    ) -> str:
        """
        Generate a formatted bibliography.

        Args:
            keys: Citation keys to include (None for all)
            style: Citation style

        Returns:
            Formatted bibliography
        """
        citations = self.format_citations(keys, style)

        # Sort alphabetically by first author's last name
        if style == "apa":
            citations.sort()

        bibliography = "# References\n\n"
        for i, citation in enumerate(citations, 1):
            bibliography += f"{i}. {citation}\n\n"

        return bibliography

    def find_duplicates(self) -> List[Tuple[str, str]]:
        """
        Find potential duplicate citations.

        Returns:
            List of potential duplicate pairs (key1, key2)
        """
        duplicates = []
        keys = list(self.citations.keys())

        for i, key1 in enumerate(keys):
            for key2 in keys[i + 1 :]:
                if self._are_duplicates(self.citations[key1], self.citations[key2]):
                    duplicates.append((key1, key2))

        return duplicates

    def merge_duplicates(self) -> int:
        """
        Merge duplicate citations.

        Returns:
            Number of duplicates merged
        """
        duplicates = self.find_duplicates()

        for key1, key2 in duplicates:
            # Merge information from both citations
            citation1 = self.citations[key1]
            citation2 = self.citations[key2]

            # Keep the more complete citation
            if self._is_more_complete(citation1, citation2):
                # Update citation1 with any missing info from citation2
                self._merge_citation_info(citation1, citation2)
                del self.citations[key2]
            else:
                self._merge_citation_info(citation2, citation1)
                del self.citations[key1]

        return len(duplicates)

    def _generate_unique_key(self, citation: Citation) -> str:
        """Generate a unique citation key."""
        base_key = self.extractor._generate_citation_key(citation)

        # Ensure uniqueness
        if base_key not in self.citations:
            return base_key

        # Add suffix for uniqueness
        counter = 2
        while f"{base_key}_{counter}" in self.citations:
            counter += 1

        return f"{base_key}_{counter}"

    def _are_duplicates(self, citation1: Citation, citation2: Citation) -> bool:
        """Check if two citations are duplicates."""
        # Check DOI
        if citation1.doi and citation2.doi:
            return citation1.doi == citation2.doi

        # Check PMID
        if citation1.pmid and citation2.pmid:
            return citation1.pmid == citation2.pmid

        # Check title similarity
        if citation1.title and citation2.title:
            title1 = citation1.title.lower().strip()
            title2 = citation2.title.lower().strip()

            # Exact match
            if title1 == title2:
                return True

            # Fuzzy match (simple approach)
            # More sophisticated matching could use edit distance
            if len(title1) > 10 and len(title2) > 10:
                if title1[:20] == title2[:20]:
                    return True

        return False

    def _is_more_complete(self, citation1: Citation, citation2: Citation) -> bool:
        """Check if citation1 is more complete than citation2."""
        score1 = sum(
            [
                bool(citation1.authors),
                bool(citation1.title),
                bool(citation1.journal),
                bool(citation1.year),
                bool(citation1.doi),
                bool(citation1.pmid),
                bool(citation1.abstract),
            ]
        )

        score2 = sum(
            [
                bool(citation2.authors),
                bool(citation2.title),
                bool(citation2.journal),
                bool(citation2.year),
                bool(citation2.doi),
                bool(citation2.pmid),
                bool(citation2.abstract),
            ]
        )

        return score1 >= score2

    def _merge_citation_info(self, target: Citation, source: Citation):
        """Merge information from source into target."""
        if not target.authors and source.authors:
            target.authors = source.authors

        if not target.title and source.title:
            target.title = source.title

        if not target.journal and source.journal:
            target.journal = source.journal

        if not target.year and source.year:
            target.year = source.year

        if not target.volume and source.volume:
            target.volume = source.volume

        if not target.pages and source.pages:
            target.pages = source.pages

        if not target.doi and source.doi:
            target.doi = source.doi

        if not target.pmid and source.pmid:
            target.pmid = source.pmid

        if not target.abstract and source.abstract:
            target.abstract = source.abstract


class CitationValidator:
    """Validates citations for completeness and accuracy."""

    def validate_citation(self, citation: Citation) -> Dict[str, Any]:
        """
        Validate a citation for completeness.

        Args:
            citation: Citation to validate

        Returns:
            Validation result with score and issues
        """
        issues = []
        score = 100

        # Check required fields
        if not citation.authors:
            issues.append("Missing authors")
            score -= 20

        if not citation.title:
            issues.append("Missing title")
            score -= 20

        if not citation.year:
            issues.append("Missing year")
            score -= 15

        if not citation.journal and not citation.doi:
            issues.append("Missing journal and DOI")
            score -= 15

        # Check year validity
        if citation.year:
            current_year = 2025  # Update as needed
            if citation.year < 1900 or citation.year > current_year:
                issues.append(f"Invalid year: {citation.year}")
                score -= 10

        # Check DOI format
        if citation.doi and not re.match(r"^10\.\d{4,}/[-._;()/:\w]+$", citation.doi):
            issues.append(f"Invalid DOI format: {citation.doi}")
            score -= 10

        return {
            "valid": len(issues) == 0,
            "score": max(0, score),
            "issues": issues,
            "citation_key": citation.citation_key,
        }

    def validate_all(
        self, citations: List[Citation]
    ) -> Tuple[List[Citation], List[Citation]]:
        """
        Validate all citations and separate valid from invalid.

        Args:
            citations: List of citations to validate

        Returns:
            Tuple of (valid citations, invalid citations)
        """
        valid = []
        invalid = []

        for citation in citations:
            result = self.validate_citation(citation)
            if result["valid"]:
                valid.append(citation)
            else:
                invalid.append(citation)

        return valid, invalid


class PubMedIntegration:
    """Integration with PubMed for real-time paper retrieval."""

    def __init__(self):
        """Initialize PubMed integration."""
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.search_url = f"{self.base_url}/esearch.fcgi"
        self.fetch_url = f"{self.base_url}/efetch.fcgi"
        self.summary_url = f"{self.base_url}/esummary.fcgi"
        self.timeout = 30
        self.max_results = 100
        # Add additional databases for comprehensive search
        self.databases = ["pubmed", "pmc", "medgen", "gene", "protein"]

    async def search_papers(
        self,
        query: str,
        max_results: int = 20,
        sort_by: str = "relevance",
        min_year: Optional[int] = None,
    ) -> List[str]:
        """
        Search PubMed for papers matching query.

        Args:
            query: Search query with PubMed syntax
            max_results: Maximum number of results
            sort_by: Sort order (relevance, date)
            min_year: Minimum publication year

        Returns:
            List of PubMed IDs
        """
        # Build search query
        if min_year:
            query = f"{query} AND {min_year}[PDAT]:3000[PDAT]"

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, self.max_results),
            "retmode": "xml",
            "sort": sort_by,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.search_url, params=params, timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        root = ET.fromstring(text)

                        # Extract PMIDs
                        id_list = root.find("IdList")
                        if id_list is not None:
                            pmids = [id_elem.text for id_elem in id_list.findall("Id")]
                            return pmids

        except Exception as e:
            print(f"PubMed search error: {e}")

        return []

    async def fetch_paper_details(self, pmids: List[str]) -> List[Citation]:
        """
        Fetch detailed paper information from PubMed.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of Citation objects
        """
        if not pmids:
            return []

        citations = []

        # Fetch in batches of 10
        for i in range(0, len(pmids), 10):
            batch = pmids[i : i + 10]

            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.fetch_url, params=params, timeout=self.timeout
                    ) as response:
                        if response.status == 200:
                            text = await response.text()
                            citations.extend(self._parse_pubmed_xml(text))

            except Exception as e:
                print(f"PubMed fetch error: {e}")

        return citations

    def _parse_pubmed_xml(self, xml_text: str) -> List[Citation]:
        """Parse PubMed XML response into Citation objects."""
        citations = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                citation = Citation()

                # Extract PMID
                pmid_elem = article.find(".//PMID")
                if pmid_elem is not None:
                    citation.pmid = pmid_elem.text

                # Extract article info
                article_elem = article.find(".//Article")
                if article_elem is not None:
                    # Title
                    title_elem = article_elem.find(".//ArticleTitle")
                    if title_elem is not None:
                        citation.title = title_elem.text or ""

                    # Abstract
                    abstract_elem = article_elem.find(".//Abstract/AbstractText")
                    if abstract_elem is not None:
                        citation.abstract = abstract_elem.text or ""

                    # Authors
                    authors = []
                    for author in article_elem.findall(".//Author"):
                        lastname = author.find("LastName")
                        forename = author.find("ForeName")
                        if lastname is not None and forename is not None:
                            authors.append(f"{forename.text} {lastname.text}")
                    citation.authors = authors

                    # Journal
                    journal_elem = article_elem.find(".//Journal/Title")
                    if journal_elem is not None:
                        citation.journal = journal_elem.text or ""

                    # Year
                    year_elem = article_elem.find(
                        ".//Journal/JournalIssue/PubDate/Year"
                    )
                    if year_elem is not None:
                        try:
                            citation.year = int(year_elem.text)
                        except:
                            citation.year = 0

                    # Volume
                    volume_elem = article_elem.find(".//Journal/JournalIssue/Volume")
                    if volume_elem is not None:
                        citation.volume = volume_elem.text or ""

                    # Pages
                    pages_elem = article_elem.find(".//Pagination/MedlinePgn")
                    if pages_elem is not None:
                        citation.pages = pages_elem.text or ""

                    # DOI
                    for id_elem in article_elem.findall(".//ELocationID"):
                        if id_elem.get("EIdType") == "doi":
                            citation.doi = id_elem.text or ""
                            break

                # Generate citation key
                if citation.authors and citation.year:
                    first_author = citation.authors[0].split()[-1]
                    citation.citation_key = (
                        f"{first_author}_{citation.year}_pmid{citation.pmid}"
                    )
                else:
                    citation.citation_key = f"pmid_{citation.pmid}"

                citations.append(citation)

        except Exception as e:
            print(f"XML parsing error: {e}")

        return citations

    async def search_immunology_papers(
        self, topic: str, max_results: int = 20, recent_only: bool = True
    ) -> List[Citation]:
        """
        Search for immunology papers on a specific topic.

        Args:
            topic: Research topic
            max_results: Maximum papers to retrieve
            recent_only: Only papers from last 5 years

        Returns:
            List of Citation objects
        """
        # Build immunology-focused query
        query = f"({topic}) AND (immunology OR immune OR antibody OR T cell OR B cell)"

        if recent_only:
            min_year = datetime.now().year - 5
        else:
            min_year = None

        # Search PubMed
        pmids = await self.search_papers(query, max_results, "relevance", min_year)

        # Fetch full details
        citations = await self.fetch_paper_details(pmids)

        return citations

    async def search_novel_tools(
        self, tool_category: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for novel computational tools in a category.

        Args:
            tool_category: Category like "protein structure", "single-cell", etc.
            max_results: Maximum tools to find

        Returns:
            List of tool information dictionaries
        """
        # Build query for computational tools
        query = f"({tool_category}) AND (tool OR software OR algorithm OR pipeline OR package) AND computational"

        # Search recent papers (last 2 years for novelty)
        min_year = datetime.now().year - 2
        pmids = await self.search_papers(query, max_results, "date", min_year)

        # Fetch details
        citations = await self.fetch_paper_details(pmids)

        # Extract tool information
        tools = []
        for citation in citations:
            tool_info = {
                "name": self._extract_tool_name(citation.title),
                "description": citation.abstract[:200] if citation.abstract else "",
                "paper_title": citation.title,
                "year": citation.year,
                "pmid": citation.pmid,
                "doi": citation.doi,
                "category": tool_category,
            }
            tools.append(tool_info)

        return tools

    def _extract_tool_name(self, title: str) -> str:
        """Extract potential tool name from paper title."""
        # Look for words in all caps or with special formatting
        import re

        # Pattern for tool names (all caps, or CamelCase, or with numbers)
        patterns = [
            r"\b[A-Z]{3,}\b",  # All caps (e.g., BLAST)
            r"\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b",  # CamelCase (e.g., DeepMind)
            r"\b[A-Za-z]+\d+[A-Za-z]*\b",  # With numbers (e.g., AlphaFold2)
        ]

        for pattern in patterns:
            matches = re.findall(pattern, title)
            if matches:
                return matches[0]

        # Fallback: first significant noun phrase
        words = title.split(":")[0].split()[:3]
        return " ".join(words)

    async def get_recent_breakthroughs(
        self, research_area: str, days_back: int = 30
    ) -> List[Citation]:
        """
        Get recent breakthrough papers in a research area.

        Args:
            research_area: Area of research
            days_back: How many days back to search

        Returns:
            List of recent high-impact citations
        """
        # Calculate date range
        from datetime import timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Build query for high-impact recent papers
        query = (
            f"({research_area}) AND (breakthrough OR novel OR discovery OR innovative)"
        )
        query += f" AND {start_date.strftime('%Y/%m/%d')}[PDAT]:{end_date.strftime('%Y/%m/%d')}[PDAT]"

        # Search with date sort
        pmids = await self.search_papers(query, 20, "date")

        # Fetch and return
        return await self.fetch_paper_details(pmids)


class PaperQualityRanker:
    """Rank papers by quality and relevance."""

    def __init__(self):
        """Initialize the quality ranker."""
        self.high_impact_journals = {
            "nature",
            "science",
            "cell",
            "nature immunology",
            "immunity",
            "journal of experimental medicine",
            "nature medicine",
            "new england journal of medicine",
            "lancet",
            "nature biotechnology",
            "nature methods",
        }

        self.immunology_keywords = {
            "antibody",
            "tcr",
            "bcr",
            "immune",
            "immunology",
            "t cell",
            "b cell",
            "antigen",
            "epitope",
            "mhc",
            "cytokine",
            "immunotherapy",
            "vaccine",
            "car-t",
        }

    def rank_papers(
        self, citations: List[Citation], query: str = ""
    ) -> List[Tuple[Citation, float]]:
        """
        Rank papers by quality and relevance.

        Args:
            citations: List of citations to rank
            query: Original search query for relevance scoring

        Returns:
            List of (Citation, score) tuples sorted by score
        """
        ranked = []

        for citation in citations:
            score = self._calculate_quality_score(citation, query)
            ranked.append((citation, score))

        # Sort by score (descending)
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    def _calculate_quality_score(self, citation: Citation, query: str) -> float:
        """
        Calculate quality score for a paper.

        Components:
        - Journal impact (0-30 points)
        - Recency (0-20 points)
        - Relevance to query (0-25 points)
        - Immunology relevance (0-15 points)
        - Completeness (0-10 points)
        """
        score = 0.0

        # Journal impact
        journal_lower = citation.journal.lower()
        if any(j in journal_lower for j in self.high_impact_journals):
            score += 30
        elif "plos" in journal_lower or "bmc" in journal_lower:
            score += 15
        elif citation.journal:
            score += 10

        # Recency (papers from last 5 years get more points)
        if citation.year:
            years_old = datetime.now().year - citation.year
            if years_old <= 2:
                score += 20
            elif years_old <= 5:
                score += 15
            elif years_old <= 10:
                score += 8
            else:
                score += 3

        # Relevance to query
        if query:
            query_terms = query.lower().split()
            title_lower = citation.title.lower()
            abstract_lower = citation.abstract.lower()

            matches = 0
            for term in query_terms:
                if term in title_lower:
                    matches += 2
                if term in abstract_lower:
                    matches += 1

            score += min(25, matches * 3)

        # Immunology relevance
        content = (citation.title + " " + citation.abstract).lower()
        immuno_matches = sum(1 for kw in self.immunology_keywords if kw in content)
        score += min(15, immuno_matches * 3)

        # Completeness
        if citation.doi:
            score += 3
        if citation.pmid:
            score += 3
        if citation.abstract:
            score += 2
        if len(citation.authors) > 0:
            score += 2

        return score

    def filter_high_quality(
        self, citations: List[Citation], min_score: float = 50.0, query: str = ""
    ) -> List[Citation]:
        """
        Filter to only high-quality papers.

        Args:
            citations: List of citations
            min_score: Minimum quality score
            query: Search query for relevance

        Returns:
            Filtered list of high-quality citations
        """
        ranked = self.rank_papers(citations, query)
        return [cite for cite, score in ranked if score >= min_score]


class EnhancedCitationManager(CitationManager):
    """Enhanced citation manager with PubMed integration and quality ranking."""

    def __init__(self):
        """Initialize enhanced citation manager."""
        super().__init__()
        self.pubmed = PubMedIntegration()
        self.ranker = PaperQualityRanker()

    async def search_and_add_papers(
        self, query: str, max_papers: int = 20, quality_threshold: float = 50.0
    ) -> List[str]:
        """
        Search PubMed and add high-quality papers.

        Args:
            query: Search query
            max_papers: Maximum papers to add
            quality_threshold: Minimum quality score

        Returns:
            List of citation keys added
        """
        # Search PubMed
        pmids = await self.pubmed.search_papers(
            query, max_papers * 2
        )  # Get extra for filtering

        # Fetch details
        citations = await self.pubmed.fetch_paper_details(pmids)

        # Filter by quality
        high_quality = self.ranker.filter_high_quality(
            citations, quality_threshold, query
        )

        # Add to manager
        keys = []
        for citation in high_quality[:max_papers]:
            key = self.add_citation(citation)
            keys.append(key)

        return keys

    async def enrich_citations(self):
        """
        Enrich existing citations with PubMed data.
        """
        for key, citation in self.citations.items():
            if citation.pmid and not citation.abstract:
                # Fetch full details from PubMed
                details = await self.pubmed.fetch_paper_details([citation.pmid])
                if details:
                    # Update citation with enriched data
                    enriched = details[0]
                    if not citation.abstract:
                        citation.abstract = enriched.abstract
                    if not citation.authors:
                        citation.authors = enriched.authors
                    if not citation.doi:
                        citation.doi = enriched.doi

    def get_ranked_citations(
        self, keys: Optional[List[str]] = None, query: str = ""
    ) -> List[Tuple[Citation, float]]:
        """
        Get citations ranked by quality.

        Args:
            keys: Citation keys (None for all)
            query: Query for relevance scoring

        Returns:
            Ranked list of (Citation, score) tuples
        """
        if keys is None:
            citations = list(self.citations.values())
        else:
            citations = [self.citations[k] for k in keys if k in self.citations]

        return self.ranker.rank_papers(citations, query)


# Export main classes
__all__ = [
    "Citation",
    "CitationExtractor",
    "CitationManager",
    "CitationValidator",
    "PubMedIntegration",
    "PaperQualityRanker",
    "EnhancedCitationManager",
]
