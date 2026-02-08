"""Vector Database Search Tool for Deep Research.

Complete implementation of vector database retrieval functionality,
strictly following retrieve_tools.py and retrieval_utils.py.

Integrated components:
- QdrantParentDocumentRetriever: Advanced chunk-to-parent document retrieval
- MMR Search: Maximum Marginal Relevance for diversity
- LLM Summarization: Context-aware document summarization  
- Academic Noise Filtering: Multi-level content cleaning
- LLM-based Ranking: Quality scoring and filtering
- Deduplication: Content-based and hash-based

This implementation uses local config and vectorstore modules.
"""

import os
import re
import asyncio
import logging
from typing import List, Optional, Annotated, Callable
from concurrent.futures import ThreadPoolExecutor

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from pydantic import BaseModel, Field

# Import only framework configuration (keep this)
from nodes.subagents.deep_research.configuration import Configuration

# Import Qdrant and LangChain components
from qdrant_client import QdrantClient
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

try:
    from diskcache import Cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    Cache = None

logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Metadata Keys
# ============================================================================

# Metadata keys for document tracking (from vectorstore/store.py)
KEY_SRC = "source"
KEY_SRC_FULL = "metadata.source"
KEY_PAGE_CONTENT = "page_content"
KEY_PAGE = "page"
KEY_ORIGINAL_CHUNKS = "original_chunks"
KEY_ORIGINAL_DOCUMENT = "original_document"

VECTOR_SEARCH_DESCRIPTION = (
    "Search internal knowledge base using vector similarity. "
    "This tool retrieves relevant documents from the organization's proprietary "
    "vector database containing historical research data, experimental results, "
    "internal publications, and domain-specific knowledge. "
    "Use this when you need institutional knowledge, past research findings, "
    "or internal experimental data that may not be available on the public internet."
)

# Prompt for document quality scoring (from retrieval_utils.py) - SINGLE document version (legacy)
DOCUMENT_SCORING_PROMPT = """You are evaluating the relevance and quality of a research document.

Query: {query}

Document Content:
{content}

Please evaluate this document on the following criteria:
1. Relevance Score (0-10): How relevant is this document to the query?
2. Quality Score (0-10): How high is the academic/scientific quality?
3. Noise Level (0-3): Does it contain noise like references, headers, footers? (0=clean, 3=very noisy)
4. Final Score (0-100): Overall score combining relevance, quality, and noise.

Return your evaluation in JSON format:
{{
    "relevance_score": <0-10>,
    "quality_score": <0-10>,
    "noise_level": <0-3>,
    "final_score": <0-100>
}}"""

# Prompt for BATCH document scoring - evaluates multiple documents in ONE LLM call
BATCH_SCORING_PROMPT = """You are evaluating the relevance and quality of multiple research documents.

Query: {query}

Documents to evaluate:
{documents}

For EACH document, evaluate on these criteria:
1. Relevance Score (0-10): How relevant to the query?
2. Quality Score (0-10): Academic/scientific quality?
3. Noise Level (0-3): Contains noise like references, headers? (0=clean, 3=noisy)
4. Final Score (0-100): Overall score.

Return a JSON array with one evaluation per document, in the SAME ORDER as the documents above:
[
    {{"doc_id": 1, "relevance_score": X, "quality_score": X, "noise_level": X, "final_score": X}},
    {{"doc_id": 2, "relevance_score": X, "quality_score": X, "noise_level": X, "final_score": X}},
    ...
]"""

# Prompt for parent retriever summarization (from vectorstore/prompts.py)
PARENT_RETRIEVER_SUMMARIZE_PROMPT = """
You are {role}. Given a QUERY and a relevant paper, your task is to summarize the paper with respect to the QUERY.

Your summary should:
- Clearly address the QUERY, focusing only on information relevant to it.
- Include the problem addressed, high-level design, methodologies, and conclusions from the paper.
- Methodologies must be emphasized, such as the data source, software and algorighms used, and how are experiments designed.
- Be strictly grounded in the provided PARENT paper; do not fabricate or infer information not present in the text.

The summary must not exceed {chunk_size} tokens.

QUERY:
{query}

PARENT:
{parent}

SUMMARY:
"""


# ============================================================================
# Pydantic Models
# ============================================================================

class DocumentEvaluation(BaseModel):
    """Document evaluation result (from retrieval_utils.py)."""
    doc_id: int = Field(default=0, description="Document ID (1-indexed)")
    relevance_score: int = Field(ge=0, le=10, description="Relevance to query (0-10)")
    quality_score: int = Field(ge=0, le=10, description="Document quality (0-10)")
    noise_level: int = Field(ge=0, le=3, description="Noise level (0-3)")
    final_score: int = Field(ge=0, le=100, description="Final score (0-100)")


class BatchDocumentEvaluation(BaseModel):
    """Batch evaluation result for multiple documents."""
    evaluations: List[DocumentEvaluation] = Field(description="List of document evaluations")


class RetrievedDocument(BaseModel):
    """Retrieved document with source and content (from retrieve_tools.py)."""
    source: str = Field(description="source of the document, can be a url or a file path")
    page_content: str = Field(description="content of the document")

    def __str__(self) -> str:
        return f"""
<document>
    <source>{self.source}</source>
    <content>{self.page_content}</content>
</document>
"""

    def __repr__(self) -> str:
        return self.__str__()


# ============================================================================
# Utility Functions (from retrieval_utils.py)
# ============================================================================

def remove_think_tags(text: str) -> str:
    """Remove <think> tags and their content from text (from retrieval_utils.py).
    
    Args:
        text: Input text that may contain think tags
        
    Returns:
        Cleaned text without think tags
    """
    if text is None:
        return ""
    
    # Remove complete <think> tags and their content
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Handle cases with only closing tag, remove content before closing tag
    cleaned_text = re.sub(r".*?</think>", "", cleaned_text, flags=re.DOTALL)
    # Remove extra blank lines
    cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)
    return cleaned_text.strip()


def is_academic_noise(content: str) -> bool:
    """Detect if text content is academic noise (from retrieval_utils.py).
    
    Args:
        content: Text content to check
        
    Returns:
        bool: True if it's noise, False otherwise
    """
    if not content or len(content) <= 80:
        return True
    
    content_lower = content.lower()
    
    # Citation format noise
    if re.search(r"^\[\d+\]", content):  # [6] at start
        return True
    elif re.search(r"^\d+\.\s*[a-z]\.\s*[a-z]", content_lower):  # "32. h. cai" format
        return True
    elif re.search(r"et al\.|crossref|pubmed|doi:", content_lower):  # Academic citation markers
        return True
    
    # Journal format and template noise - match specific format patterns
    elif re.search(
        r"(springer nature \d{4}|latex template|copyright.*reserved)", content_lower
    ):
        return True
    elif re.search(
        r"(corresponding author:.*@|published online:|received:.*accepted:)",
        content_lower,
    ):
        return True
    
    # Figure and table noise
    elif re.search(r"^(fig\.|figure|table|supplementary)\s*\d+", content_lower):
        return True
    
    # Page number and format noise
    elif re.search(r"^\d{4}\)\d{3}|volume\s*\d+|issue\s*\d+", content_lower):
        return True
    
    # Pure reference list (excessive bracket/parenthesis density)
    elif content.count("(") + content.count("[") > len(content) / 20:
        return True
    
    return False


def clean_document_content(content: str) -> str:
    """Clean document content (from retrieval_utils.py).
    
    Args:
        content: Raw document content
        
    Returns:
        str: Cleaned content
    """
    if not content:
        return ""
    
    # Remove trailing digit numbers
    content = re.sub(r"\d{3,}$", "", content)
    # Remove excessive whitespace
    content = re.sub(r"\s+", " ", content).strip()
    
    return content


def _filter_chunks(chunks: list[Document]) -> list[Document]:
    """Filter documents (from retrieve_tools.py).
    
    Args:
        chunks: List of document chunks
        
    Returns:
        Filtered chunks without academic noise
    """
    return list(filter(lambda x: not is_academic_noise(x.page_content), chunks))


# ============================================================================
# QdrantConfig (from config/config.py)
# ============================================================================

from dataclasses import dataclass
from typing import Optional
from functools import cache as fc


# Global singleton cache for Qdrant client and related objects
_qdrant_client_cache: Optional[QdrantClient] = None
_qdrant_config_cache: Optional['QdrantConfig'] = None
_embedder_cache: Optional[object] = None
_vector_store_cache: Optional[QdrantVectorStore] = None


@dataclass
class QdrantConfig:
    """Qdrant configuration (from config/config.py)."""
    host: str = "117.148.176.36"
    port: int = 6333
    grpc_port: int = 6334
    api_key: Optional[str] = None
    prefer_grpc: bool = False
    https: bool = False
    prefix: Optional[str] = None
    timeout: int = 5
    host_override: Optional[str] = None

    # Class-level singleton instance
    _instance: Optional['QdrantConfig'] = None
    _client: Optional[QdrantClient] = None

    @classmethod
    def from_env(cls):
        """Get singleton config instance from environment variables."""
        global _qdrant_config_cache
        if _qdrant_config_cache is not None:
            return _qdrant_config_cache
        
        _qdrant_config_cache = cls(
            host=os.getenv("QDRANT_HOST", "117.148.176.36"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
            grpc_port=int(os.getenv("QDRANT_GRPC_PORT", 6334)),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true",
            https=os.getenv("QDRANT_HTTPS", "false").lower() == "true",
            prefix=os.getenv("QDRANT_PREFIX"),
            timeout=int(os.getenv("QDRANT_TIMEOUT", 5)),
            host_override=os.getenv("QDRANT_HOST_OVERRIDE"),
        )
        return _qdrant_config_cache

    def get_client(self) -> QdrantClient:
        """Get singleton Qdrant client instance."""
        global _qdrant_client_cache
        if _qdrant_client_cache is not None:
            return _qdrant_client_cache
        
        logger.info(f"Creating new Qdrant client connection to {self.host}:{self.port}")
        _qdrant_client_cache = QdrantClient(
            url=f"{'https' if self.https else 'http'}://{self.host}:{self.port}",
            port=self.grpc_port if self.prefer_grpc else None,
            api_key=self.api_key,
            prefix=self.prefix,
            timeout=self.timeout,
            host=self.host_override or None,
            check_compatibility=False,  # Suppress version warning
        )
        return _qdrant_client_cache

    def __hash__(self):
        return (
            hash(self.host)
            + hash(self.port)
            + hash(self.grpc_port)
            + hash(self.api_key)
            + hash(self.prefer_grpc)
            + hash(self.https)
            + hash(self.prefix)
            + hash(self.timeout)
            + hash(self.host_override)
        )


# ============================================================================
# QdrantParentDocumentRetriever (from vectorstore/store.py)
# ============================================================================

class QdrantParentDocumentRetriever(BaseRetriever):
    """
    QdrantParentDocumentRetriever first retrieves query-relevant chunks from the document collection,
    then retrieves the parent documents based on chunk metadata, and performs context-aware summarization.
    
    From vectorstore/store.py
    """

    summarize_model: BaseChatModel
    chunk_filter: Optional[Callable[[list[Document]], list[Document]]] = None
    retriever_kwargs: Optional[dict] = None
    vector_store: QdrantVectorStore
    # Role of the summarization model
    role: str = "An academic paper reviewer"
    summarize: bool = True

    def _get_full_parent(self, parent_source: str) -> Optional[str]:
        """Retrieve full parent document by assembling all chunks."""
        chunks = self.vector_store.client.query_points(
            collection_name=self.vector_store.collection_name,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key=KEY_SRC_FULL, match=MatchValue(value=parent_source)
                    )
                ]
            ),
        ).points
        if len(chunks) == 0:
            return None

        if KEY_PAGE in chunks[0].payload:
            chunks = sorted(chunks, key=lambda x: int(x.payload[KEY_PAGE]))
        merged = ""
        for chunk in chunks:
            merged += chunk.payload[KEY_PAGE_CONTENT]
            merged += " "
        return merged

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Retrieve relevant documents using parent document strategy."""
        # Get disk cache if available
        cache = None
        if CACHE_AVAILABLE:
            from pathlib import Path
            cache_dir = Path("/tmp/antibody_gen/retriever/summarizer")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache = Cache(str(cache_dir))
        
        # first search for related chunks
        chunk_docs = self.vector_store.as_retriever(
            **(self.retriever_kwargs or {})
        ).invoke(query)
        if self.chunk_filter is not None:
            chunk_docs = self.chunk_filter(chunk_docs)
        by_parent: dict[str, list[Document]] = {}
        # group by source
        for doc in chunk_docs:
            parent_id = doc.metadata[KEY_SRC]
            if parent_id not in by_parent:
                by_parent[parent_id] = []
            by_parent[parent_id].append(doc)
        chunk_size = 512
        chain = (
            PromptTemplate.from_template(PARENT_RETRIEVER_SUMMARIZE_PROMPT)
            | self.summarize_model
            | StrOutputParser()
        )
        ret = []
        for parent_src, chunks in by_parent.items():
            cache_key = f"{parent_src}_{query}"
            parent = self._get_full_parent(parent_src)
            if parent is None:
                ret.extend(chunks)
                continue
            in_cache = cache.get(cache_key) if cache else None
            summary = parent
            if self.summarize:
                if in_cache is not None:
                    summary = in_cache
                else:
                    summary = chain.invoke(
                        {
                            "query": query,
                            "parent": parent,
                            "chunk_size": chunk_size,
                            "role": self.role,
                        }
                    )
                    if cache:
                        cache.add(cache_key, summary)
            ret.append(
                Document(
                    page_content=summary,
                    metadata={KEY_SRC: parent_src, KEY_ORIGINAL_DOCUMENT: parent, KEY_ORIGINAL_CHUNKS: chunks},
                )
            )

        return ret


# ============================================================================
# Configuration and Model Creation
# ============================================================================

def _create_embedder_from_config(config: RunnableConfig):
    """Create embedder from environment variables (from retrieve_tools.py).
    
    Uses singleton pattern to cache embedder instance.
    
    Reads configuration from environment variables:
    - EMBEDDING_PROVIDER: 'openai' or 'ollama' (default: 'openai')
    - EMBEDDING_MODEL: model name (default: 'text-embedding-3-small')
    - EMBEDDING_API_KEY or OPENAI_API_KEY: API key (required for OpenAI)
    - EMBEDDING_BASE_URL: base URL (optional, uses default if not set)
    
    Args:
        config: RunnableConfig (not used, kept for compatibility)
        
    Returns:
        Embedding model instance (cached singleton)
    """
    global _embedder_cache
    if _embedder_cache is not None:
        return _embedder_cache
    
    # Read from environment variables
    provider = os.getenv('EMBEDDING_PROVIDER', 'openai')
    model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    api_key = os.getenv('EMBEDDING_API_KEY') or os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('EMBEDDING_BASE_URL')
    
    logger.info(f"Creating new embedder: provider={provider}, model={model}")
    
    if provider.lower() == 'openai':
        # Build kwargs, only include base_url if it's set
        kwargs = {
            'model': model,
            'openai_api_key': api_key
        }
        if base_url:  # Only add base_url if it exists and is not empty
            kwargs['openai_api_base'] = base_url
        
        _embedder_cache = OpenAIEmbeddings(**kwargs)
    else:  # Ollama or other
        _embedder_cache = OllamaEmbeddings(model=model)
    
    return _embedder_cache


def _get_summarize_model(config: RunnableConfig) -> BaseChatModel:
    """Get summarization model from config with priority fallback.
    
    Priority:
        1. vector_summarization_model (if configured)
        2. summarization_model
        3. research_model (final fallback)
    
    Args:
        config: RunnableConfig
        
    Returns:
        BaseChatModel for summarization
    """
    try:
        configurable = Configuration.from_runnable_config(config)
        
        # Priority 1: Use vector-specific summarization model if configured (use getattr for compatibility)
        vector_sum_model = getattr(configurable, 'vector_summarization_model', None)
        if vector_sum_model:
            logger.info(f"Using vector_summarization_model: {vector_sum_model}")
            return init_chat_model(
                model=vector_sum_model,
                temperature=0.1
            )
        
        # Priority 2: Use general summarization model
        logger.info(f"Using summarization_model: {configurable.summarization_model}")
        return init_chat_model(
            model=configurable.summarization_model,
            temperature=0.1
        )
    except Exception as e:
        logger.warning(f"Failed to get summarization model from config: {e}")
        # Priority 3: Fallback to research model
        try:
            configurable = Configuration.from_runnable_config(config)
            logger.info(f"Falling back to research_model: {configurable.research_model}")
            return init_chat_model(
                model=configurable.research_model,
                temperature=0.1
            )
        except:
            raise ValueError("Cannot initialize summarization model")


# ============================================================================
# LLM-based Document Filtering and Ranking (from retrieval_utils.py)
# ============================================================================

def _score_document_with_llm(doc_info: tuple, config: RunnableConfig) -> tuple:
    """Score a single document using LLM with priority fallback.
    
    Priority:
        1. vector_scoring_model (if configured)
        2. research_model (fallback)
    
    Args:
        doc_info: Tuple of (doc_idx, content, questions)
        config: Runtime configuration
        
    Returns:
        Tuple of (doc_idx, DocumentEvaluation)
    """
    doc_idx, content, questions = doc_info
    
    prompt = ChatPromptTemplate.from_template(DOCUMENT_SCORING_PROMPT)
    
    try:
        configurable = Configuration.from_runnable_config(config)
        
        # Priority 1: Use vector-specific scoring model if configured (use getattr for compatibility)
        vector_score_model = getattr(configurable, 'vector_scoring_model', None)
        if vector_score_model:
            model_name = vector_score_model
            logger.debug(f"Using vector_scoring_model: {model_name}")
        else:
            # Priority 2: Fallback to research model
            model_name = configurable.research_model
            logger.debug(f"Using research_model for scoring: {model_name}")
        
        llm = init_chat_model(
            model=model_name,
            temperature=0.1
        )
        
        evaluation_model = llm.with_structured_output(DocumentEvaluation)
        evaluation_chain = prompt | evaluation_model
        
        response = evaluation_chain.invoke({"query": questions, "content": content})
        
        print(f"Document {doc_idx + 1}: relevance={response.relevance_score}, quality={response.quality_score}, noise={response.noise_level}, final={response.final_score}")
        return (doc_idx + 1, response)
        
    except Exception as e:
        print(f"Document {doc_idx + 1} scoring failed: {e}")
        # Return default low score evaluation
        default_eval = DocumentEvaluation(
            relevance_score=0,
            quality_score=0,
            noise_level=3,
            final_score=0
        )
        return (doc_idx + 1, default_eval)


def _batch_score_documents(docs_info: List[tuple], config: RunnableConfig) -> List[tuple]:
    """Batch score multiple documents in a SINGLE LLM call.
    
    Args:
        docs_info: List of (doc_idx, content, questions) tuples
        config: Runtime configuration
        
    Returns:
        List of (doc_idx, DocumentEvaluation) tuples
    """
    if not docs_info:
        return []
    
    questions = docs_info[0][2]  # All docs share the same questions
    
    # Build documents string for batch prompt (keep full content like original logic)
    docs_text = ""
    for i, (doc_idx, content, _) in enumerate(docs_info):
        docs_text += f"\n--- Document {i + 1} ---\n{content}\n"
    
    try:
        configurable = Configuration.from_runnable_config(config)
        
        # Get scoring model
        vector_score_model = getattr(configurable, 'vector_scoring_model', None)
        model_name = vector_score_model or configurable.research_model
        
        llm = init_chat_model(model=model_name, temperature=0.1)
        evaluation_model = llm.with_structured_output(BatchDocumentEvaluation)
        
        prompt = ChatPromptTemplate.from_template(BATCH_SCORING_PROMPT)
        chain = prompt | evaluation_model
        
        response = chain.invoke({"query": questions, "documents": docs_text})
        
        # Map evaluations back to original doc indices
        results = []
        for i, eval_result in enumerate(response.evaluations):
            if i < len(docs_info):
                doc_idx = docs_info[i][0]
                eval_result.doc_id = doc_idx + 1
                results.append((doc_idx + 1, eval_result))
                print(f"Document {doc_idx + 1}: relevance={eval_result.relevance_score}, quality={eval_result.quality_score}, final={eval_result.final_score}")
        
        return results
        
    except Exception as e:
        print(f"Batch scoring failed: {e}, falling back to simple scoring")
        # Fallback: return default scores
        return [
            (doc_idx + 1, DocumentEvaluation(doc_id=doc_idx + 1, relevance_score=5, quality_score=5, noise_level=1, final_score=50))
            for doc_idx, _, _ in docs_info
        ]


def model_filter_and_rank(filtered_docs: List[Document], queries: List[str], config: RunnableConfig) -> List[Document]:
    """Precise filtering of high-quality context - using BATCH scoring for efficiency.
    
    Optimized to use a single LLM call for all documents instead of one per document.
    
    Args:
        filtered_docs: Documents to filter and rank
        queries: Search queries
        config: Runtime configuration
        
    Returns:
        Top-ranked documents
    """
    # Build query string
    questions = "\n".join([f"Question {i + 1}: {query}" for i, query in enumerate(queries)])
    
    # Prepare documents for batch scoring
    tasks = []
    for i, doc in enumerate(filtered_docs):
        content = doc.page_content.strip()
        if len(content) > 30:
            tasks.append((i, content, questions))
    
    print(f"Batch scoring {len(tasks)} documents in ONE LLM call...")
    
    try:
        # BATCH scoring - single LLM call for all documents
        doc_evaluations = _batch_score_documents(tasks, config)
        
        # Sort by final_score
        doc_evaluations.sort(key=lambda x: x[1].final_score, reverse=True)
        
        # Safe document filtering strategy
        result = _safe_document_filter(filtered_docs, doc_evaluations)
        
        print(f"Filtering result: {len(result)} documents")
        return result
    except Exception as e:
        print(f"Scoring failed, using original order: {e}")
        return filtered_docs[:10]


def _safe_document_filter(documents: List[Document], doc_evaluations: List[tuple], target_count: int = 15) -> List[Document]:
    """Simplified document filtering: sort by total score and noise, return top N documents (from retrieval_utils.py).
    
    Args:
        documents: Original documents
        doc_evaluations: List of (doc_idx, evaluation) tuples
        target_count: Number of documents to return
        
    Returns:
        Top documents
    """
    # Extract documents and evaluation results
    doc_eval_pairs = []
    for doc_idx, evaluation in doc_evaluations:
        if doc_idx <= len(documents):
            doc_eval_pairs.append((documents[doc_idx - 1], evaluation))
    
    if not doc_eval_pairs:
        return []
    
    # Sort by final score, noise level as secondary criteria (lower noise is better)
    sorted_docs = sorted(
        doc_eval_pairs,
        key=lambda x: (x[1].final_score, -x[1].noise_level),  # Final score descending, noise ascending
        reverse=True
    )
    
    # Return top N documents
    return [doc for doc, _ in sorted_docs[:target_count]]


# ============================================================================
# Core Retrieval Functions (from retrieve_tools.py)
# ============================================================================

def retrieve_doc(
    query: List[str], 
    config: RunnableConfig, 
    k_per_query: int = 5
) -> list[RetrievedDocument]:
    """Retrieve related documents from the knowledge base (from retrieve_tools.py).
    
    This is the core retrieval function that:
    1. Creates QdrantParentDocumentRetriever with MMR search
    2. Retrieves documents for each query
    3. Removes think tags
    4. Deduplicates results
    5. Cleans and filters content
    6. Applies LLM-based ranking
    7. Returns structured RetrievedDocument objects

    Args:
        query: List of query strings. It is recommended that each string can be 16 to 128 tokens in length.
        config: RunnableConfig.
        k_per_query: number of retrieved documents for each query.

    Returns:
        Retrieved documents with source and page content, where source is the path or url of original paper.
    """
    try:
        global _vector_store_cache
        
        # Create Qdrant configuration from environment variables (uses singleton)
        qdrant_config = QdrantConfig.from_env()
        collection_name = os.getenv("QDRANT_COLLECTION", "Immunology")
        
        # Use cached vector_store if available
        if _vector_store_cache is None:
            embedder = _create_embedder_from_config(config)
            client = qdrant_config.get_client()
            
            logger.info(f"Creating new vector store for collection: {collection_name}")
            _vector_store_cache = QdrantVectorStore(
                client=client,
                embedding=embedder,
                collection_name=collection_name
            )
        
        vector_store = _vector_store_cache
        
        # Create QdrantParentDocumentRetriever with MMR search
        retriever = QdrantParentDocumentRetriever(
            summarize_model=_get_summarize_model(config),
            vector_store=vector_store,
            role="computational antibody design expert",
            retriever_kwargs={
                "search_type": "mmr",
                "search_kwargs": {"k": k_per_query, "lambda_mult": 0.65},
            },
            chunk_filter=_filter_chunks,
        )
        
        all_docs = []
        seen_docs = set()

        # Execute retrieval sequentially to avoid vector database concurrency conflicts
        results = []
        for q in query:
            try:
                docs = retriever.invoke(q)

                # Some retrievers may include <think/> tags
                for doc in docs:
                    doc.page_content = remove_think_tags(doc.page_content)
                    
                results.append(docs)
                print(f"Query '{q[:50]}...' retrieved {len(docs)} documents")
            except Exception as e:
                print(f"Retriever query failed: {e}")
                results.append([])

        # Merge results and deduplicate
        total_retrieved = 0
        for docs in results:
            total_retrieved += len(docs)

            for doc in docs:
                doc_hash = hash(doc.page_content)
                if doc_hash not in seen_docs:
                    seen_docs.add(doc_hash)
                    all_docs.append(doc)

        # Content cleaning and filtering
        scored_docs = []
        for doc in all_docs:
            content = clean_document_content(doc.page_content.strip())
            if not is_academic_noise(content):
                doc.page_content = content
                scored_docs.append(doc)

        print(
            f"Total retrieved {total_retrieved} documents, {len(all_docs)} after deduplication, {len(scored_docs)} after cleaning"
        )

        # Model filtering and ranking
        if len(scored_docs) > 0:
            top_docs = model_filter_and_rank(scored_docs, query, config)
            print(f"Obtained {len(top_docs)} high-quality documents after model filtering")
        else:
            top_docs = []
            
        return [
            RetrievedDocument(
                source=doc.metadata[KEY_SRC], 
                page_content=doc.page_content
            )
            for doc in top_docs
        ]
    except Exception as e:
        import traceback
        print(f"Retrieval query failed: {e}")
        print(f"Detailed error information: {traceback.format_exc()}")
        return []


async def retrieve(
    query: List[str], 
    config: RunnableConfig, 
    k_per_query: int = 10
) -> str:
    """Retrieve related documents from the knowledge base (from retrieve_tools.py).
    
    This is a wrapper around retrieve_doc that formats the output as a string.

    Args:
        query: List of query strings.
        config: RunnableConfig.
        k_per_query: number of retrieved documents for each query

    Returns:
        Retrieved documents with page_content and source. The source of a document can be a url or a file path.
        Source of the document is included in <source> tag
    """
    try:
        # Call retrieve_doc in a thread to avoid blocking
        top_docs = await asyncio.to_thread(
            retrieve_doc,
            query=query,
            config=config,
            k_per_query=k_per_query
        )
        # Build context
        context = "\n\n".join([str(doc) for doc in top_docs])
        return context
    except Exception as e:
        print(f"Retrieval query failed: {e}")
        return ""


# ============================================================================
# Tool Definitions for Deep Research
# ============================================================================

@tool(description=VECTOR_SEARCH_DESCRIPTION)
async def vector_db_search(
    queries: List[str],
    k_per_query: Annotated[int, InjectedToolArg] = 5,
    config: RunnableConfig = None
) -> str:
    """Search the internal vector database for relevant documents.
    
    This tool queries the organization's Qdrant vector database to retrieve
    relevant documents based on semantic similarity using advanced retrieval techniques:
    - Parent Document Retrieval: Retrieves chunks, then fetches full parent documents
    - MMR Search: Maximum Marginal Relevance for diversity
    - LLM Summarization: Context-aware document summarization
    - Quality Filtering: Academic noise detection and LLM-based ranking
    
    It's particularly useful for accessing:
    - Historical research data and experimental results
    - Internal publications and technical reports  
    - Proprietary domain knowledge and methodologies
    - Past project findings and lessons learned
    
    Args:
        queries: List of search queries (natural language)
        k_per_query: Number of results to return per query (default: 5)
        config: Runtime configuration for API keys and settings
        
    Returns:
        Formatted string containing retrieved documents with metadata
        
    Examples:
        Search for internal research:
        >>> results = await vector_db_search(["CAR-T cell therapy optimization"])
        
        Search multiple topics:
        >>> results = await vector_db_search([
        ...     "CRISPR gene editing protocols",
        ...     "immune checkpoint inhibitor combinations"
        ... ])
    """
    # Validate queries
    if not queries:
        return "No search queries provided. Please specify at least one query."
    
    # Limit number of queries to prevent excessive API calls
    max_queries = 5
    if len(queries) > max_queries:
        logger.warning(f"Too many queries ({len(queries)}), limiting to {max_queries}")
        queries = queries[:max_queries]
    
    # Perform retrieval
    return await retrieve(query=queries, config=config, k_per_query=k_per_query)


async def get_vector_search_tools(config: RunnableConfig) -> list:
    """Get vector database search tools if configured.
    
    Args:
        config: RunnableConfig (runtime configuration)
        
    Returns:
        List containing vector search tool, or empty list if not configured
    """
    # Check if vector database is configured by checking environment variables
    required_vars = ['QDRANT_HOST', 'QDRANT_PORT', 'QDRANT_COLLECTION']
    if not all(os.getenv(var) for var in required_vars):
        logging.warning("Vector database not fully configured (missing QDRANT_HOST/PORT/COLLECTION), skipping vector_db_search tool")
        return []
    
    # Return the vector search tool
    return [vector_db_search]
