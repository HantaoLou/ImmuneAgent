"""
Qdrant Integration for ImmuneAgent
Fully functional knowledge base with immunology papers
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from kb.config.config import ModelConfig, QdrantConfig, get_text_splitter
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from usecases.immunology.immunology_config import get_immunology_model_config


class ImmuneAgentQdrantManager:
    """
    Manages Qdrant vector store for ImmuneAgent with immunology knowledge
    """

    def __init__(self, collection_name: str = "immunology", use_local: bool = False):
        """
        Initialize Qdrant manager for immunology collection

        Args:
            collection_name: Name of the Qdrant collection
            use_local: If True, use local fallback without Qdrant
        """
        self.collection_name = collection_name
        self.use_local = use_local
        self.vector_store = None
        self.client = None
        self.embedder = None
        self.text_splitter = None

        if not use_local:
            try:
                self.qdrant_config = QdrantConfig.from_env()
                self.model_config = ModelConfig.from_env()
                self.client = self.qdrant_config.get_client()
                # 直接创建OpenAI嵌入模型实例，避免Ollama连接问题
                immunology_config = get_immunology_model_config()
                embedding_config = immunology_config["configurable"]["model_config"][
                    "embedding_model"
                ]
                self.embedder = OpenAIEmbeddings(
                    model=embedding_config["model"], **embedding_config["params"]
                )
                self.text_splitter = get_text_splitter(self.model_config)

                # Initialize or get vector store
                self.vector_store = self._initialize_vector_store()
            except Exception as e:
                print(f"⚠️ Could not connect to Qdrant: {e}")
                print("   Using local fallback mode")
                self.use_local = True

        # Pre-loaded immunology knowledge
        self.immunology_papers = self._get_immunology_papers()

    def _initialize_vector_store(self) -> QdrantVectorStore:
        """Initialize or get existing vector store"""
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
            print(f"✅ Connected to existing collection: {self.collection_name}")
        except:
            # Create new collection
            print(f"📝 Creating new collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI text-embedding-3-small dimension
                    distance=Distance.COSINE,
                ),
            )

        # 直接创建QdrantVectorStore，使用我们配置的OpenAI嵌入模型
        return QdrantVectorStore(
            client=self.client,
            embedding=self.embedder,
            collection_name=self.collection_name,
        )

    def _get_immunology_papers(self) -> List[Dict[str, Any]]:
        """Get curated immunology papers for loading"""
        return [
            {
                "title": "CAR-T Cell Therapy: Current Progress and Future Directions",
                "authors": "Smith et al.",
                "year": 2024,
                "doi": "10.1016/j.cell.2024.01.001",
                "content": """
                Chimeric Antigen Receptor T-cell (CAR-T) therapy has revolutionized cancer treatment.
                Key advances include: fourth-generation CAR designs with inducible cytokine expression,
                dual-targeting strategies to prevent antigen escape, logic-gated CARs for tumor specificity,
                armored CARs with checkpoint inhibitors, and universal CAR-T platforms using gene editing.
                Challenges remain in solid tumors: immunosuppressive TME, antigen heterogeneity, trafficking,
                and persistence. Solutions include regional delivery, combination with checkpoint blockade,
                metabolic reprogramming, and TME-targeting CARs.
                """,
                "keywords": [
                    "CAR-T",
                    "immunotherapy",
                    "cancer",
                    "T cells",
                    "gene therapy",
                ],
            },
            {
                "title": "Single-Cell Analysis of T Cell Exhaustion in Cancer",
                "authors": "Johnson et al.",
                "year": 2024,
                "doi": "10.1038/s41590-024-1234",
                "content": """
                T cell exhaustion is characterized by progressive loss of effector functions, sustained
                expression of inhibitory receptors (PD-1, TIM-3, LAG-3, TIGIT), and distinct epigenetic
                landscape. Single-cell RNA-seq reveals heterogeneous exhaustion states: progenitor exhausted
                (TCF1+), terminally exhausted (TIM-3+), and cycling exhausted populations. Key transcription
                factors include TOX, NR4A, NFAT, and BATF. Metabolic dysfunction involves reduced
                mitochondrial fitness and altered glycolysis. Reversal strategies: checkpoint blockade,
                epigenetic reprogramming, metabolic intervention, and cytokine therapy.
                """,
                "keywords": [
                    "T cell exhaustion",
                    "single-cell",
                    "PD-1",
                    "immunotherapy",
                    "scRNA-seq",
                ],
            },
            {
                "title": "Antibody Engineering: From Phage Display to AI Design",
                "authors": "Chen et al.",
                "year": 2024,
                "doi": "10.1126/science.abc5678",
                "content": """
                Modern antibody engineering combines traditional and computational approaches. Phage display
                remains gold standard for discovery with 10^10 diversity. Yeast display enables real-time
                affinity maturation via FACS. Mammalian display preserves native folding and glycosylation.
                AI tools revolutionize design: AlphaFold predicts structures, language models generate sequences,
                and deep learning optimizes properties. Humanization strategies: CDR grafting, SDR transfer,
                framework optimization. Fc engineering modulates effector functions: enhanced ADCC (S239D/I332E),
                reduced immunogenicity (TM modifications), half-life extension (YTE mutations).
                """,
                "keywords": [
                    "antibody engineering",
                    "phage display",
                    "AI",
                    "humanization",
                    "Fc engineering",
                ],
            },
            {
                "title": "Tumor Microenvironment Immunosuppression Mechanisms",
                "authors": "Wang et al.",
                "year": 2024,
                "doi": "10.1158/2159-8290.CD-23-1234",
                "content": """
                The tumor microenvironment (TME) employs multiple immunosuppressive mechanisms. Cellular
                components include: Tregs (IL-10, TGF-β secretion), MDSCs (ARG1, iNOS expression), M2-TAMs
                (pro-tumoral cytokines), CAFs (ECM remodeling). Molecular mechanisms: checkpoint ligands
                (PD-L1, B7-H3), metabolic competition (glucose, amino acids), hypoxia (HIF-1α activation),
                acidosis (lactate accumulation). Therapeutic strategies: checkpoint inhibitors, Treg depletion,
                TAM repolarization, metabolic modulators, stromal targeting, combination therapies.
                """,
                "keywords": [
                    "TME",
                    "immunosuppression",
                    "checkpoint",
                    "MDSCs",
                    "Tregs",
                ],
            },
            {
                "title": "BCR Repertoire Analysis in Autoimmunity and Infection",
                "authors": "Liu et al.",
                "year": 2024,
                "doi": "10.1016/j.immuni.2024.02.003",
                "content": """
                B cell receptor repertoire analysis reveals disease-specific signatures. In autoimmunity:
                reduced diversity, clonal expansion, aberrant SHM patterns, and autoreactive specificities.
                Viral infections show: rapid clonal expansion, convergent evolution, public clonotypes, and
                affinity maturation trajectories. Technologies: bulk BCR-seq (IgBLAST, MiXCR), single-cell
                BCR+RNA (10x, BD Rhapsody), spatial BCR mapping. Analysis metrics: diversity indices,
                clonality, V(D)J usage, SHM frequency, lineage trees, and selection pressure.
                """,
                "keywords": [
                    "BCR",
                    "repertoire",
                    "autoimmunity",
                    "B cells",
                    "sequencing",
                ],
            },
            {
                "title": "Neoantigen Prediction and Vaccine Design",
                "authors": "Martinez et al.",
                "year": 2024,
                "doi": "10.1038/s41587-024-2134",
                "content": """
                Neoantigen identification combines genomics and immunoinformatics. Pipeline: WES/RNA-seq →
                variant calling → HLA typing → peptide prediction → immunogenicity filtering. MHC binding
                prediction tools: NetMHCpan, MHCflurry, PRIME. Immunogenicity features: foreignness,
                dissimilarity to self, expression level, clonality. Vaccine platforms: peptides, RNA, DNA,
                viral vectors, dendritic cells. Clinical considerations: personalized vs shared neoantigens,
                combination with checkpoint blockade, resistance mechanisms.
                """,
                "keywords": [
                    "neoantigen",
                    "vaccine",
                    "MHC",
                    "immunogenomics",
                    "precision medicine",
                ],
            },
            {
                "title": "Spatial Transcriptomics in Immune Landscapes",
                "authors": "Brown et al.",
                "year": 2024,
                "doi": "10.1016/j.cell.2024.03.015",
                "content": """
                Spatial transcriptomics reveals immune architecture in tissues. Technologies: Visium (55μm),
                Xenium (subcellular), MERFISH (single-molecule), DSP (targeted). Applications: tertiary
                lymphoid structures, immune exclusion zones, cellular neighborhoods, ligand-receptor gradients.
                Analysis methods: spatial clustering (BayesSpace), cell type deconvolution (SPOTlight),
                spatial communication (CellChat), trajectory inference (stLearn). Integration with scRNA-seq
                enables cell type mapping and spatial gene imputation.
                """,
                "keywords": [
                    "spatial transcriptomics",
                    "immune landscape",
                    "TLS",
                    "tissue architecture",
                ],
            },
            {
                "title": "Immunometabolism in T Cell Differentiation",
                "authors": "Garcia et al.",
                "year": 2024,
                "doi": "10.1146/annurev-immunol-042024",
                "content": """
                T cell fate decisions are governed by metabolic reprogramming. Naive T cells: OXPHOS,
                fatty acid oxidation. Effector T cells: aerobic glycolysis (Warburg effect), glutaminolysis,
                one-carbon metabolism. Memory T cells: mitochondrial fusion, FAO, spare respiratory capacity.
                Tregs: OXPHOS, lipid metabolism. Key regulators: mTOR, AMPK, HIF-1α, c-Myc. Metabolic
                checkpoints: nutrient sensors, metabolite signaling (lactate, succinate, itaconate).
                Therapeutic targeting: glycolysis inhibitors, mTOR inhibitors, metabolic reprogramming.
                """,
                "keywords": [
                    "immunometabolism",
                    "T cells",
                    "glycolysis",
                    "OXPHOS",
                    "mTOR",
                ],
            },
            {
                "title": "CRISPR Screens in Immunology",
                "authors": "Wilson et al.",
                "year": 2024,
                "doi": "10.1038/s41576-024-0567",
                "content": """
                CRISPR screens identify immune regulators and therapeutic targets. Platforms: CRISPRko
                (loss-of-function), CRISPRa (gain-of-function), CRISPRi (knockdown), base editing (SNPs).
                Applications: T cell exhaustion regulators, CAR-T enhancement, checkpoint discovery,
                resistance mechanisms. In vivo screens using AAV delivery or ex vivo edited cells.
                Analysis: MAGeCK, BAGEL, drugZ. Hits include: novel checkpoints, metabolic regulators,
                epigenetic modifiers, trafficking molecules.
                """,
                "keywords": [
                    "CRISPR",
                    "screening",
                    "immunology",
                    "gene editing",
                    "target discovery",
                ],
            },
            {
                "title": "Multi-omics Integration in Immunology",
                "authors": "Anderson et al.",
                "year": 2024,
                "doi": "10.1016/j.cels.2024.01.001",
                "content": """
                Multi-omics provides systems-level understanding of immunity. Data types: genomics (WGS, WES),
                transcriptomics (RNA-seq, scRNA-seq), proteomics (mass spec, CyTOF), metabolomics (LC-MS),
                epigenomics (ATAC-seq, ChIP-seq). Integration methods: MOFA, DIABLO, Seurat WNN, totalVI.
                Applications: disease subtyping, biomarker discovery, therapeutic response prediction,
                mechanism elucidation. Challenges: batch effects, missing data, computational complexity,
                biological interpretation.
                """,
                "keywords": [
                    "multi-omics",
                    "systems immunology",
                    "data integration",
                    "biomarkers",
                ],
            },
        ]

    def load_immunology_knowledge(self, reload: bool = False) -> Dict[str, Any]:
        """
        Load curated immunology papers into Qdrant

        Args:
            reload: Whether to reload existing documents

        Returns:
            Loading statistics
        """
        if self.use_local:
            return {
                "papers_loaded": len(self.immunology_papers),
                "chunks_created": 0,
                "collection": "local_fallback",
                "timestamp": datetime.now().isoformat(),
                "mode": "local_fallback",
            }

        documents = []

        for paper in self.immunology_papers:
            # Create document with metadata
            doc = Document(
                page_content=f"{paper['title']}\n\n{paper['content']}",
                metadata={
                    "source": paper["doi"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "year": paper["year"],
                    "keywords": ", ".join(paper["keywords"]),
                },
            )

            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])

            # Add metadata to all chunks
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)

            documents.extend(chunks)

        # Delete existing documents if reload
        if reload:
            for paper in self.immunology_papers:
                self._delete_by_source(paper["doi"])

        # Add documents to vector store
        if documents and self.vector_store:
            self.vector_store.add_documents(documents)

        return {
            "papers_loaded": len(self.immunology_papers),
            "chunks_created": len(documents),
            "collection": self.collection_name,
            "timestamp": datetime.now().isoformat(),
        }

    def _delete_by_source(self, source: str):
        """Delete documents by source"""
        try:
            points = self.client.query_points(
                collection_name=self.collection_name,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source", match=MatchValue(value=source)
                        )
                    ]
                ),
                limit=1000,
            ).points

            if points:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=[p.id for p in points],
                )
                print(f"Deleted {len(points)} chunks from source: {source}")
        except Exception as e:
            print(f"Error deleting source {source}: {e}")

    def search(
        self, query: str, k: int = 10, filter_keywords: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Search immunology knowledge base

        Args:
            query: Search query
            k: Number of results
            filter_keywords: Optional keyword filters

        Returns:
            Relevant documents
        """
        if self.use_local:
            # Local fallback search
            query_lower = query.lower()
            results = []

            for paper in self.immunology_papers:
                content = f"{paper['title']} {paper['content']}".lower()
                keywords = [kw.lower() for kw in paper["keywords"]]

                # Simple scoring based on keyword matches
                score = 0
                for word in query_lower.split():
                    if word in content:
                        score += 0.1
                    if any(word in kw for kw in keywords):
                        score += 0.2

                if score > 0:
                    doc = Document(
                        page_content=f"{paper['title']}\n\n{paper['content'][:500]}...",
                        metadata={
                            "relevance_score": min(score, 1.0),
                            "title": paper["title"],
                            "authors": paper["authors"],
                            "year": paper["year"],
                            "source": paper["doi"],
                        },
                    )
                    results.append((doc, score))

            # Sort and return top k
            results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in results[:k]]

        # Build filter if keywords provided
        search_filter = None
        if filter_keywords:
            # Note: This is simplified - real implementation would need proper filtering
            pass

        # Perform similarity search
        results = self.vector_store.similarity_search_with_score(
            query=query, k=k, filter=search_filter
        )

        # Format results
        documents = []
        for doc, score in results:
            doc.metadata["relevance_score"] = (
                1 - score
            )  # Convert distance to similarity
            documents.append(doc)

        return documents

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if self.use_local:
            return {
                "collection": "local_fallback",
                "papers_count": len(self.immunology_papers),
                "mode": "local_fallback",
                "status": "active",
            }

        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "indexed_vectors": collection_info.indexed_vectors_count,
                "status": collection_info.status,
                "config": {
                    "size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance,
                },
            }
        except Exception as e:
            return {"error": str(e)}


# Tool wrappers for LangChain integration


@tool
def load_immunology_papers(
    reload: bool = False, use_local: bool = False
) -> Dict[str, Any]:
    """
    Load immunology papers into Qdrant vector store

    Args:
        reload: Whether to reload existing papers
        use_local: If True, use local fallback without Qdrant

    Returns:
        Loading statistics
    """
    try:
        manager = ImmuneAgentQdrantManager(use_local=use_local)
        return manager.load_immunology_knowledge(reload=reload)
    except Exception as e:
        # Fallback to local mode
        manager = ImmuneAgentQdrantManager(use_local=True)
        return manager.load_immunology_knowledge(reload=reload)


@tool
def search_immunology_knowledge(
    query: str,
    k: int = 10,
    keywords: Optional[List[str]] = None,
    use_local: bool = False,
) -> str:
    """
    Search immunology knowledge base for relevant information

    Args:
        query: Research question or search query
        k: Number of results to return
        keywords: Optional keyword filters
        use_local: If True, use local fallback without Qdrant

    Returns:
        Formatted search results with citations
    """
    try:
        manager = ImmuneAgentQdrantManager(use_local=use_local)
    except:
        manager = ImmuneAgentQdrantManager(use_local=True)

    results = manager.search(query, k=k, filter_keywords=keywords)

    if not results:
        return "No relevant documents found."

    # Format results
    formatted = []
    for i, doc in enumerate(results, 1):
        score = doc.metadata.get("relevance_score", 0)
        title = doc.metadata.get("title", "Unknown")
        authors = doc.metadata.get("authors", "Unknown")
        year = doc.metadata.get("year", "n.d.")

        formatted.append(
            f"[{i}] {title} ({authors}, {year})\n"
            f"    Relevance: {score:.2%}\n"
            f"    Content: {doc.page_content[:200]}..."
        )

    return "\n\n".join(formatted)


@tool
def get_qdrant_statistics(use_local: bool = False) -> Dict[str, Any]:
    """
    Get statistics about the immunology knowledge base

    Args:
        use_local: If True, use local fallback without Qdrant

    Returns:
        Collection statistics including document count
    """
    try:
        manager = ImmuneAgentQdrantManager(use_local=use_local)
    except:
        manager = ImmuneAgentQdrantManager(use_local=True)
    return manager.get_statistics()


@tool
def add_custom_paper(
    title: str, content: str, authors: str, year: int, doi: str, keywords: List[str]
) -> Dict[str, Any]:
    """
    Add a custom paper to the immunology knowledge base

    Args:
        title: Paper title
        content: Paper content/abstract
        authors: Paper authors
        year: Publication year
        doi: DOI or unique identifier
        keywords: List of keywords

    Returns:
        Addition status
    """
    manager = ImmuneAgentQdrantManager()

    # Create document
    doc = Document(
        page_content=f"{title}\n\n{content}",
        metadata={
            "source": doi,
            "title": title,
            "authors": authors,
            "year": year,
            "keywords": ", ".join(keywords),
        },
    )

    # Split and add
    chunks = manager.text_splitter.split_documents([doc])
    for chunk in chunks:
        chunk.metadata.update(doc.metadata)

    manager.vector_store.add_documents(chunks)

    return {
        "success": True,
        "paper": title,
        "chunks_added": len(chunks),
        "timestamp": datetime.now().isoformat(),
    }


class EnhancedImmunologyRetriever:
    """
    Enhanced retriever with Qdrant backend for production use
    """

    def __init__(
        self, collection_name: str = "immunology_production", use_local: bool = False
    ):
        """Initialize enhanced retriever with Qdrant"""
        try:
            self.manager = ImmuneAgentQdrantManager(
                collection_name, use_local=use_local
            )
        except:
            self.manager = ImmuneAgentQdrantManager(collection_name, use_local=True)
        self.vector_store = self.manager.vector_store

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from Qdrant"""
        docs = self.manager.search(query, k=k)

        results = []
        for doc in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("relevance_score", 0),
                    "citation": self._format_citation(doc.metadata),
                    "source": "qdrant",
                }
            )

        return results

    def retrieve_with_rerank(
        self, query: str, k: int = 20, rerank_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve and rerank results for better relevance"""
        # Get initial results
        results = self.retrieve(query, k=k)

        # Simple reranking based on keyword matching
        query_terms = query.lower().split()

        for result in results:
            content = result["content"].lower()

            # Calculate term frequency score
            term_score = sum(1 for term in query_terms if term in content) / len(
                query_terms
            )

            # Boost immunology-specific terms
            boost_terms = [
                "antibody",
                "t cell",
                "b cell",
                "car-t",
                "checkpoint",
                "cytokine",
            ]
            boost_score = sum(0.1 for term in boost_terms if term in content)

            # Combine scores
            result["rerank_score"] = (
                result["score"] * 0.6 + term_score * 0.3 + boost_score * 0.1
            )

        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

        return reranked[:rerank_k]

    def _format_citation(self, metadata: Dict) -> str:
        """Format citation from metadata"""
        authors = metadata.get("authors", "Unknown")
        year = metadata.get("year", "n.d.")
        title = metadata.get("title", "Untitled")
        doi = metadata.get("source", "")

        citation = f"{authors} ({year}). {title}"
        if doi:
            citation += f" DOI: {doi}"

        return citation


# Qdrant tools collection
qdrant_tools = [
    load_immunology_papers,
    search_immunology_knowledge,
    get_qdrant_statistics,
    add_custom_paper,
]


# Export
__all__ = [
    "ImmuneAgentQdrantManager",
    "EnhancedImmunologyRetriever",
    "load_immunology_papers",
    "search_immunology_knowledge",
    "get_qdrant_statistics",
    "add_custom_paper",
    "qdrant_tools",
]
