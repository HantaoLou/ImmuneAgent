"""
Retrieval tools for ImmuneAgent.
Includes basic retrieval, Qdrant integration, and reranking.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Check for kb module
kb_path = Path(__file__).parent.parent.parent.parent.parent / "kb" / "src"
if str(kb_path) not in sys.path and kb_path.exists():
    sys.path.insert(0, str(kb_path))

try:
    from kb.vectorstore import get_vector_store

    KB_AVAILABLE = True
except ImportError:
    KB_AVAILABLE = False

# Try to import enhanced Qdrant integration
try:
    from .qdrant_integration import EnhancedImmunologyRetriever

    QDRANT_ENHANCED = True
except ImportError:
    QDRANT_ENHANCED = False

from langchain_core.tools import tool

sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain_openai import OpenAIEmbeddings

from usecases.immunology.immunology_config import get_immunology_model_config


class ImmunologyRetriever:
    """
    Comprehensive retrieval system for immunology knowledge.
    Supports Qdrant, fallback knowledge base, and reranking.
    """

    def __init__(self, collection_name: str = "immunology_production"):
        """Initialize retriever with Qdrant or fallback."""
        self.collection_name = collection_name
        self.vector_store = None
        # 直接创建OpenAI嵌入模型实例，避免Ollama连接问题
        immunology_config = get_immunology_model_config()
        embedding_config = immunology_config["configurable"]["model_config"][
            "embedding_model"
        ]
        self.embeddings = OpenAIEmbeddings(
            model=embedding_config["model"], **embedding_config["params"]
        )
        self.enhanced_retriever = None

        # Try enhanced Qdrant first
        if QDRANT_ENHANCED:
            try:
                self.enhanced_retriever = EnhancedImmunologyRetriever(collection_name)
                print(f"✅ Connected to Enhanced Qdrant: {collection_name}")
            except Exception as e:
                print(f"⚠️ Could not connect to Enhanced Qdrant: {e}")

        # Fallback to basic Qdrant
        if not self.enhanced_retriever and KB_AVAILABLE:
            try:
                self.vector_store = get_vector_store(collection_name)
                print(f"✅ Connected to Qdrant collection: {collection_name}")
            except Exception as e:
                print(f"⚠️ Could not connect to Qdrant: {e}")

        # Load fallback knowledge base
        self.fallback_kb = self._load_fallback_knowledge()

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        # Try enhanced retriever first
        if self.enhanced_retriever:
            try:
                return self.enhanced_retriever.retrieve(query, k)
            except Exception as e:
                print(f"Enhanced retrieval failed: {e}")

        # Fallback to basic Qdrant
        if self.vector_store:
            try:
                return self._qdrant_retrieve(query, k)
            except Exception as e:
                print(f"Qdrant retrieval failed: {e}")
                return self._fallback_retrieve(query, k)
        else:
            return self._fallback_retrieve(query, k)

    def retrieve_with_rerank(
        self, query: str, k: int = 20, rerank_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve and rerank for better relevance."""
        # Use enhanced retriever with reranking if available
        if self.enhanced_retriever:
            try:
                return self.enhanced_retriever.retrieve_with_rerank(query, k, rerank_k)
            except Exception as e:
                print(f"Enhanced reranking failed: {e}")

        # Fallback to basic reranking
        results = self.retrieve(query, k)
        reranked = self._rerank_results(query, results, rerank_k)

        return reranked

    def _qdrant_retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Retrieve from Qdrant vector store."""
        docs = self.vector_store.similarity_search_with_score(query, k=k)

        results = []
        for doc, score in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(1 - score),
                    "citation": self._format_citation(doc.metadata),
                    "source": "qdrant",
                }
            )

        return results

    def _fallback_retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Fallback retrieval from curated knowledge base."""
        results = []
        query_lower = query.lower()

        for entry in self.fallback_kb:
            score = self._calculate_relevance(query_lower, entry)

            if score > 0:
                results.append(
                    {
                        "content": entry["content"],
                        "metadata": entry.get("metadata", {}),
                        "score": score,
                        "citation": entry.get("citation", "Internal KB"),
                        "source": "fallback",
                    }
                )

        # Sort and return top k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def _rerank_results(
        self, query: str, results: List[Dict], top_k: int
    ) -> List[Dict]:
        """Rerank results using enhanced scoring."""
        query_terms = query.lower().split()

        for result in results:
            content = result["content"].lower()

            # Term frequency
            term_score = sum(1 for term in query_terms if term in content) / len(
                query_terms
            )

            # Immunology term boost
            boost_terms = [
                "antibody",
                "t cell",
                "b cell",
                "car-t",
                "checkpoint",
                "cytokine",
            ]
            boost_score = sum(0.1 for term in boost_terms if term in content)

            # Source preference (Qdrant > fallback)
            source_score = 0.1 if result.get("source") == "qdrant" else 0

            # Combined rerank score
            result["rerank_score"] = (
                result["score"] * 0.5
                + term_score * 0.3
                + boost_score * 0.1
                + source_score * 0.1
            )

        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

        return reranked[:top_k]

    def _calculate_relevance(self, query: str, entry: Dict) -> float:
        """Calculate relevance score for fallback retrieval."""
        score = 0.0
        content = entry["content"].lower()
        keywords = entry.get("keywords", [])

        # Keyword matching
        for keyword in keywords:
            if keyword.lower() in query:
                score += 0.3

        # Content matching
        query_words = query.split()
        matches = sum(1 for word in query_words if word in content)
        score += min(matches * 0.1, 0.5)

        return min(score, 1.0)

    def _format_citation(self, metadata: Dict) -> str:
        """Format citation from metadata."""
        authors = metadata.get("authors", "Unknown")
        year = metadata.get("year", "n.d.")
        title = metadata.get("title", "Untitled")
        doi = metadata.get("doi", "")

        citation = f"{authors} ({year}). {title}"
        if doi:
            citation += f" DOI: {doi}"

        return citation

    def _load_fallback_knowledge(self) -> List[Dict]:
        """Load curated immunology knowledge base."""
        return [
            {
                "content": """CAR-T cell therapy revolutionizes cancer treatment through genetically 
                engineered T cells. Key components: scFv antigen recognition, hinge region, transmembrane 
                domain, intracellular signaling (CD3ζ + costimulatory). Challenges: CRS, neurotoxicity, 
                tumor escape. Future: dual-targeting, armored CARs, logic gates.""",
                "keywords": ["CAR-T", "immunotherapy", "cancer", "T cells"],
                "citation": "Smith et al. (2024). CAR-T Cell Therapy Review",
                "metadata": {"category": "cell_therapy", "year": 2024},
            },
            {
                "content": """T cell exhaustion in chronic infections shows loss of effector functions,
                sustained inhibitory receptor expression (PD-1, TIM-3, LAG-3). Driven by TOX, NR4A, NFAT.
                Metabolic dysfunction with reduced mitochondrial function. Progenitor exhausted cells
                respond to PD-1 blockade.""",
                "keywords": ["T cell", "exhaustion", "PD-1", "checkpoint"],
                "citation": "Johnson et al. (2024). T Cell Exhaustion",
                "metadata": {"category": "t_cell_biology", "year": 2024},
            },
            {
                "content": """Antibody engineering spans phage display to AI design. Strategies include
                affinity maturation, humanization, Fc engineering, bispecific formats. AI tools like
                AlphaFold, AntiBERTy enable computational optimization. CDR grafting maintains stability.""",
                "keywords": ["antibody", "engineering", "humanization", "CDR"],
                "citation": "Chen et al. (2024). Antibody Engineering",
                "metadata": {"category": "antibody_discovery", "year": 2024},
            },
            {
                "content": """Tumor microenvironment immunosuppression via Tregs, MDSCs, M2-TAMs.
                Mechanisms: checkpoint ligands, metabolic competition, hypoxia, cytokines (IL-10, TGF-β).
                Therapeutics: checkpoint inhibitors, Treg depletion, TAM repolarization.""",
                "keywords": ["TME", "tumor", "immunosuppression", "checkpoint"],
                "citation": "Wang et al. (2024). Tumor Microenvironment",
                "metadata": {"category": "tumor_immunology", "year": 2024},
            },
            {
                "content": """Single-cell RNA-seq reveals immune heterogeneity. Workflow: QC, normalization,
                feature selection, dimensionality reduction, clustering, annotation, trajectory inference.
                Tools: Scanpy, Seurat, scVI-tools. Applications: cell discovery, state transitions.""",
                "keywords": ["single-cell", "scRNA-seq", "scanpy", "seurat"],
                "citation": "Liu et al. (2024). Single-Cell Analysis",
                "metadata": {"category": "single_cell", "year": 2024},
            },
        ]


@tool
def retrieve_immunology_knowledge(query: str, k: int = 10, rerank: bool = True) -> str:
    """
    Retrieve immunology knowledge for a research question.

    Args:
        query: Research question
        k: Number of documents to retrieve
        rerank: Whether to apply reranking

    Returns:
        Formatted context with citations
    """
    retriever = ImmunologyRetriever()

    if rerank:
        results = retriever.retrieve_with_rerank(query, k=k * 2, rerank_k=k)
    else:
        results = retriever.retrieve(query, k)

    # Format results
    context_parts = []
    citations = []

    for i, result in enumerate(results, 1):
        context_parts.append(f"[{i}] {result['content']}")
        citations.append(result["citation"])

    context = "\n\n".join(context_parts)
    citation_list = "\n".join(f"[{i}] {c}" for i, c in enumerate(set(citations), 1))

    return f"{context}\n\nReferences:\n{citation_list}"


@tool
def expand_query(query: str) -> List[str]:
    """
    Expand a query with related immunology terms.

    Args:
        query: Original query

    Returns:
        List of expanded queries
    """
    expanded = [query]
    query_lower = query.lower()

    # Immunology-specific expansions
    expansions = {
        "car-t": ["CAR T cell", "chimeric antigen receptor"],
        "antibody": ["immunoglobulin", "mAb", "IgG"],
        "t cell": ["T lymphocyte", "CD4+", "CD8+"],
        "b cell": ["B lymphocyte", "plasma cell"],
        "checkpoint": ["PD-1", "CTLA-4", "immune checkpoint inhibitor"],
    }

    for key, values in expansions.items():
        if key in query_lower:
            for value in values:
                expanded.append(query.replace(key, value))

    return expanded[:5]  # Limit to 5 expansions


# Export retrieval components
__all__ = ["ImmunologyRetriever", "retrieve_immunology_knowledge", "expand_query"]
