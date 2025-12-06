#!/usr/bin/env python
"""
Comprehensive Qdrant Deployment for ImmuneAgent
Loads 800+ immunology papers from various sources
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

# Add kb module to path
kb_path = Path(__file__).parent.parent.parent.parent / "kb" / "src"
if str(kb_path) not in sys.path:
    sys.path.insert(0, str(kb_path))

# Add current module to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our constants
from constants import OPENAI_API_KEY
from kb.cli.filter import filter_by_entropy, read_and_chunk_pdfs
from kb.cli.main import add_documents
from kb.config import QdrantConfig, get_embedder
from kb.config.config import ModelConfig, get_text_splitter

# Import kb modules
from kb.vectorstore import get_vector_store
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)


class ImmuneAgentQdrantDeployment:
    """
    Complete Qdrant deployment for ImmuneAgent with 800+ immunology papers
    """

    def __init__(self, collection_name: str = "immunology_production"):
        """
        Initialize deployment manager

        Args:
            collection_name: Name of the Qdrant collection
        """
        self.collection_name = collection_name
        self.qdrant_config = QdrantConfig.from_env()
        self.model_config = ModelConfig.from_env()
        self.client = self.qdrant_config.get_client()

        # Use OpenAI embeddings for better quality
        self.embedder = OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
        )
        self.text_splitter = get_text_splitter(self.model_config)

        # Statistics
        self.stats = {
            "papers_loaded": 0,
            "chunks_created": 0,
            "sources": [],
            "errors": [],
        }

    def initialize_collection(self, recreate: bool = False):
        """Initialize or recreate the Qdrant collection"""
        try:
            if recreate:
                try:
                    self.client.delete_collection(self.collection_name)
                    print(f"🗑️ Deleted existing collection: {self.collection_name}")
                except:
                    pass

            # Check if collection exists
            try:
                collection_info = self.client.get_collection(self.collection_name)
                print(f"✅ Using existing collection: {self.collection_name}")
                print(f"   Vectors: {collection_info.vectors_count}")
                return True
            except:
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI text-embedding-3-small dimension
                        distance=Distance.COSINE,
                    ),
                )
                print(f"📝 Created new collection: {self.collection_name}")
                return True

        except Exception as e:
            print(f"❌ Error initializing collection: {e}")
            return False

    def load_pubmed_papers(self, query: str, max_papers: int = 200) -> List[Document]:
        """
        Load papers from PubMed using their API

        Args:
            query: Search query for PubMed
            max_papers: Maximum number of papers to retrieve

        Returns:
            List of Document objects
        """
        documents = []

        try:
            # PubMed E-utilities API
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

            # Search for papers
            search_url = f"{base_url}esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_papers,
                "retmode": "json",
                "sort": "relevance",
            }

            print(f"🔍 Searching PubMed for: {query}")
            response = requests.get(search_url, params=search_params)
            search_results = response.json()

            id_list = search_results.get("esearchresult", {}).get("idlist", [])
            print(f"   Found {len(id_list)} papers")

            if not id_list:
                return documents

            # Fetch paper details
            fetch_url = f"{base_url}efetch.fcgi"

            # Process in batches of 20
            for i in range(0, len(id_list), 20):
                batch_ids = id_list[i : i + 20]
                fetch_params = {
                    "db": "pubmed",
                    "id": ",".join(batch_ids),
                    "retmode": "xml",
                    "rettype": "abstract",
                }

                response = requests.get(fetch_url, params=fetch_params)

                # Parse XML response (simplified)
                # In production, use proper XML parsing
                content = response.text

                # Extract abstracts (simplified extraction)
                for pmid in batch_ids:
                    # Create document for each paper
                    doc = Document(
                        page_content=f"PubMed ID: {pmid}\n{content[:2000]}",  # Simplified
                        metadata={
                            "source": f"pubmed:{pmid}",
                            "pmid": pmid,
                            "query": query,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    documents.append(doc)

                time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"❌ Error loading PubMed papers: {e}")
            self.stats["errors"].append(f"PubMed: {str(e)}")

        return documents

    def load_biorxiv_papers(
        self, category: str = "immunology", max_papers: int = 100
    ) -> List[Document]:
        """
        Load papers from bioRxiv

        Args:
            category: bioRxiv category
            max_papers: Maximum number of papers

        Returns:
            List of Document objects
        """
        documents = []

        try:
            # bioRxiv API
            base_url = "https://api.biorxiv.org/details/biorxiv"

            # Get recent papers
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = "2023-01-01"

            url = f"{base_url}/{start_date}/{end_date}/0/{max_papers}"

            print(f"🔍 Loading bioRxiv papers from {category}")
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                papers = data.get("collection", [])

                for paper in papers:
                    # Filter by category if needed
                    if (
                        category.lower() in paper.get("category", "").lower()
                        or "immun" in paper.get("title", "").lower()
                        or "antibody" in paper.get("title", "").lower()
                    ):
                        content = f"""
Title: {paper.get("title", "N/A")}
Authors: {paper.get("authors", "N/A")}
DOI: {paper.get("doi", "N/A")}
Date: {paper.get("date", "N/A")}
Category: {paper.get("category", "N/A")}

Abstract:
{paper.get("abstract", "No abstract available")}
"""

                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": f"biorxiv:{paper.get('doi', 'unknown')}",
                                "title": paper.get("title", "N/A"),
                                "authors": paper.get("authors", "N/A"),
                                "doi": paper.get("doi", "N/A"),
                                "date": paper.get("date", "N/A"),
                                "category": paper.get("category", "N/A"),
                            },
                        )
                        documents.append(doc)

                print(f"   Loaded {len(documents)} papers from bioRxiv")

        except Exception as e:
            print(f"❌ Error loading bioRxiv papers: {e}")
            self.stats["errors"].append(f"bioRxiv: {str(e)}")

        return documents

    def load_arxiv_papers(
        self, query: str = "immunology OR antibody OR 'T cell'", max_papers: int = 100
    ) -> List[Document]:
        """
        Load papers from arXiv (computational immunology)

        Args:
            query: Search query
            max_papers: Maximum number of papers

        Returns:
            List of Document objects
        """
        documents = []

        try:
            import arxiv

            print(f"🔍 Searching arXiv for: {query}")

            # Search arXiv
            search = arxiv.Search(
                query=query,
                max_results=max_papers,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            for paper in search.results():
                content = f"""
Title: {paper.title}
Authors: {", ".join([author.name for author in paper.authors])}
Published: {paper.published}
Categories: {", ".join(paper.categories)}
arXiv ID: {paper.entry_id}

Summary:
{paper.summary}
"""

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": f"arxiv:{paper.entry_id}",
                        "title": paper.title,
                        "authors": ", ".join([author.name for author in paper.authors]),
                        "published": str(paper.published),
                        "categories": ", ".join(paper.categories),
                        "pdf_url": paper.pdf_url,
                    },
                )
                documents.append(doc)

            print(f"   Loaded {len(documents)} papers from arXiv")

        except Exception as e:
            print(f"❌ Error loading arXiv papers: {e}")
            self.stats["errors"].append(f"arXiv: {str(e)}")

        return documents

    def load_immunology_knowledge_base(self) -> List[Document]:
        """
        Load comprehensive immunology knowledge base

        Returns:
            List of Document objects
        """
        documents = []

        # Comprehensive immunology topics
        knowledge_base = [
            {
                "title": "CAR-T Cell Therapy: Mechanisms and Applications",
                "content": """
CAR-T (Chimeric Antigen Receptor T-cell) therapy represents a revolutionary approach in cancer immunotherapy.
Key components include:
1. CAR Structure: scFv (single-chain variable fragment) for antigen recognition, transmembrane domain, 
   costimulatory domains (CD28, 4-1BB), and CD3ζ signaling domain
2. Generations: 1st (CD3ζ only), 2nd (one costimulatory), 3rd (two costimulatory), 4th (TRUCK with cytokines)
3. Manufacturing: T cell collection, genetic modification (viral or non-viral), expansion, quality control
4. Clinical applications: B-ALL (CD19 CAR-T), Multiple myeloma (BCMA CAR-T), emerging solid tumor targets
5. Challenges: CRS (cytokine release syndrome), ICANS (neurotoxicity), antigen escape, tumor microenvironment
6. Future directions: Universal CAR-T, logic-gated CARs, armored CARs, combination therapies
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
                "title": "Antibody Engineering and Humanization",
                "content": """
Modern antibody engineering encompasses multiple technologies:
1. Discovery platforms: Phage display (10^10 diversity), yeast display (real-time selection), 
   mammalian display (native folding), B cell cloning
2. Humanization strategies: CDR grafting (retain murine CDRs), SDR transfer (specificity determining residues),
   framework optimization, germline humanization
3. Affinity maturation: Random mutagenesis, focused libraries, computational design, machine learning
4. Fc engineering: ADCC enhancement (S239D/I332E), CDC optimization, half-life extension (YTE, LS mutations)
5. Format engineering: Bispecifics (CrossMab, KiH, DVD-Ig), ADCs (linker chemistry, payload selection),
   fragments (Fab, scFv, nanobodies)
6. Computational tools: AlphaFold structure prediction, Rosetta design, machine learning (AntiBERTy, ABlooper)
                """,
                "keywords": [
                    "antibody",
                    "engineering",
                    "humanization",
                    "bispecific",
                    "ADC",
                ],
            },
            {
                "title": "T Cell Exhaustion in Chronic Infections and Cancer",
                "content": """
T cell exhaustion is a dysfunctional state in chronic antigen exposure:
1. Molecular signatures: PD-1, TIM-3, LAG-3, TIGIT, CTLA-4, CD39, TOX, TCF1
2. Transcriptional regulation: TOX (master regulator), NFAT, NR4A, BATF, Eomes vs T-bet balance
3. Epigenetic landscape: Chromatin remodeling, DNA methylation patterns, exhaustion-specific enhancers
4. Metabolic dysfunction: Reduced glycolysis, impaired mitochondrial function, altered lipid metabolism
5. Heterogeneity: Progenitor exhausted (TCF1+, self-renewing), terminally exhausted (TIM-3+)
6. Reversal strategies: Checkpoint blockade, metabolic reprogramming, epigenetic modulation, cytokine therapy
7. Clinical implications: Response to PD-1/PD-L1 inhibitors, combination immunotherapy strategies
                """,
                "keywords": [
                    "T cell exhaustion",
                    "PD-1",
                    "checkpoint",
                    "immunotherapy",
                    "chronic infection",
                ],
            },
            {
                "title": "Single-Cell Analysis in Immunology",
                "content": """
Single-cell technologies revolutionize immune system understanding:
1. scRNA-seq: 10x Genomics, Drop-seq, Smart-seq, spatial transcriptomics (Visium, MERFISH)
2. Protein analysis: CyTOF (mass cytometry), CITE-seq, flow cytometry (spectral, imaging)
3. Chromatin accessibility: scATAC-seq, scCUT&Tag, multi-omics integration
4. TCR/BCR sequencing: Clonotype analysis, repertoire diversity, lineage tracing
5. Computational methods: Dimensionality reduction (UMAP, t-SNE), clustering (Leiden, Louvain),
   trajectory inference (Monocle, PAGA), batch correction (Harmony, scVI)
6. Applications: Cell type discovery, state transitions, cell-cell communication, disease mechanisms
                """,
                "keywords": [
                    "single-cell",
                    "scRNA-seq",
                    "CyTOF",
                    "immunology",
                    "transcriptomics",
                ],
            },
            {
                "title": "Tumor Microenvironment and Immune Evasion",
                "content": """
The tumor microenvironment employs multiple immunosuppressive mechanisms:
1. Cellular components: Tregs (IL-10, TGF-β), MDSCs (ARG1, iNOS), M2-TAMs (pro-tumoral cytokines),
   CAFs (ECM remodeling, chemokine production)
2. Inhibitory molecules: PD-L1/PD-1, CTLA-4, B7-H3, B7-H4, VISTA, TIM-3 ligands
3. Metabolic competition: Glucose depletion, amino acid consumption (tryptophan, arginine),
   lactate accumulation, hypoxia (HIF-1α activation)
4. Physical barriers: Dense ECM, abnormal vasculature, high interstitial pressure
5. Immunosuppressive cytokines: TGF-β, IL-10, VEGF, prostaglandins
6. Therapeutic strategies: Checkpoint inhibitors, Treg depletion, TAM repolarization, 
   stromal targeting, metabolic modulation
                """,
                "keywords": [
                    "TME",
                    "tumor",
                    "immunosuppression",
                    "checkpoint",
                    "MDSCs",
                ],
            },
            {
                "title": "BCR Repertoire Analysis and B Cell Biology",
                "content": """
B cell receptor repertoire analysis reveals immune system dynamics:
1. V(D)J recombination: RAG1/2 mediated, RSS recognition, junctional diversity (N/P nucleotides)
2. Somatic hypermutation: AID-mediated, hotspot motifs, affinity maturation in germinal centers
3. Class switch recombination: IgM to IgG/IgA/IgE, cytokine regulation, switch regions
4. Repertoire metrics: Diversity indices (Shannon, Simpson), clonality, V-gene usage, 
   mutation frequency, selection pressure (CDR vs FWR)
5. Analysis tools: IgBLAST, MiXCR, Change-O, Immcantation suite, lineage tree reconstruction
6. Clinical applications: Vaccine response, autoimmunity signatures, cancer immunosurveillance,
   therapeutic antibody discovery
                """,
                "keywords": [
                    "BCR",
                    "B cells",
                    "repertoire",
                    "antibody",
                    "somatic hypermutation",
                ],
            },
            {
                "title": "Immunometabolism and T Cell Function",
                "content": """
Metabolic reprogramming controls T cell fate and function:
1. Naive T cells: OXPHOS dominant, fatty acid oxidation, low biosynthetic activity
2. Activated effector T cells: Aerobic glycolysis (Warburg effect), glutaminolysis, 
   one-carbon metabolism, polyamine synthesis
3. Memory T cells: Mitochondrial fusion, enhanced spare respiratory capacity, FAO
4. Regulatory T cells: OXPHOS, lipid metabolism, mTORC1 suppression
5. Key regulators: mTOR (central metabolic sensor), AMPK (energy sensor), HIF-1α (hypoxia response),
   c-Myc (metabolic reprogramming)
6. Metabolic checkpoints: Nutrient availability, metabolite signaling (lactate, succinate, itaconate)
7. Therapeutic targeting: Glycolysis inhibitors, mTOR inhibitors, metabolic reprogramming strategies
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
                "title": "Neoantigen Discovery and Cancer Vaccines",
                "content": """
Neoantigen-based immunotherapy targets tumor-specific mutations:
1. Identification pipeline: WES/WGS for mutations, RNA-seq for expression, HLA typing,
   MHC binding prediction (NetMHCpan, MHCflurry)
2. Immunogenicity prediction: Foreignness score, DAI (differential agretopicity index),
   similarity to known epitopes, expression level
3. Vaccine platforms: Peptides (short vs long), mRNA (self-amplifying, modified nucleotides),
   DNA vectors, viral vectors, dendritic cell vaccines
4. Combination strategies: Checkpoint blockade timing, adjuvant selection (poly-ICLC, GM-CSF),
   prime-boost regimens
5. Resistance mechanisms: Antigen loss, MHC downregulation, immunoediting
6. Clinical trials: Personalized vaccines (BioNTech, Moderna), shared neoantigens (KRAS G12D)
                """,
                "keywords": [
                    "neoantigen",
                    "cancer vaccine",
                    "immunotherapy",
                    "personalized medicine",
                ],
            },
            {
                "title": "CRISPR Screens in Immunology",
                "content": """
CRISPR technology enables systematic genetic interrogation:
1. Screen types: CRISPRko (loss-of-function), CRISPRa (activation), CRISPRi (knockdown),
   base editing (precise mutations)
2. Library design: Genome-wide, focused (pathway-specific), paired guide RNAs, tiling screens
3. Immunology applications: T cell exhaustion regulators, CAR-T enhancement targets,
   checkpoint discovery, resistance mechanisms
4. In vivo screens: AAV delivery, ex vivo editing and adoptive transfer, tumor models
5. Analysis methods: MAGeCK, BAGEL, drugZ, CRISPR-analyzer, hit validation strategies
6. Key discoveries: Novel checkpoints (CMTM6, APLNR), metabolic regulators, epigenetic modifiers
                """,
                "keywords": ["CRISPR", "genetic screen", "immunology", "gene editing"],
            },
            {
                "title": "Cytokine Networks and Signaling",
                "content": """
Cytokines orchestrate immune responses through complex networks:
1. Major families: Interleukins, interferons, TNF superfamily, chemokines, growth factors
2. JAK-STAT signaling: JAK1/2/3/TYK2, STAT1-6, SOCS feedback, specificity mechanisms
3. NF-κB pathway: Canonical vs non-canonical, IKK complex, inflammatory gene expression
4. Type I interferons: Antiviral response, ISG expression, autoimmunity connections
5. Th differentiation: Th1 (IL-12, IFN-γ), Th2 (IL-4, IL-13), Th17 (IL-6, TGF-β, IL-23),
   Treg (TGF-β, IL-2)
6. Cytokine storm: COVID-19, CAR-T CRS, therapeutic interventions (tocilizumab, anakinra)
7. Therapeutic targeting: JAK inhibitors, cytokine neutralization, receptor blockade
                """,
                "keywords": [
                    "cytokine",
                    "JAK-STAT",
                    "inflammation",
                    "signaling",
                    "interleukin",
                ],
            },
        ]

        # Convert to documents
        for item in knowledge_base:
            doc = Document(
                page_content=f"{item['title']}\n\n{item['content']}",
                metadata={
                    "source": f"knowledge_base:{item['title']}",
                    "title": item["title"],
                    "keywords": ", ".join(item["keywords"]),
                    "type": "review",
                },
            )
            documents.append(doc)

        print(f"📚 Loaded {len(documents)} knowledge base articles")
        return documents

    def load_immunology_datasets(self) -> List[Document]:
        """
        Load immunology dataset descriptions and metadata

        Returns:
            List of Document objects
        """
        documents = []

        datasets = [
            {
                "name": "10x PBMC datasets",
                "description": "Peripheral blood mononuclear cell (PBMC) reference datasets from healthy donors",
                "url": "https://www.10xgenomics.com/resources/datasets",
                "data_types": ["scRNA-seq", "scATAC-seq", "Multiome", "CITE-seq"],
            },
            {
                "name": "ImmPort",
                "description": "Immunology Database and Analysis Portal - NIAID funded resource",
                "url": "https://www.immport.org",
                "data_types": [
                    "Clinical trials",
                    "Mechanistic studies",
                    "Reference data",
                ],
            },
            {
                "name": "ImmuneSpace",
                "description": "Integrative analysis platform for immunology data",
                "url": "https://www.immunespace.org",
                "data_types": ["Vaccine studies", "Flow cytometry", "Gene expression"],
            },
            {
                "name": "Human Cell Atlas",
                "description": "Comprehensive reference maps of all human cells",
                "url": "https://www.humancellatlas.org",
                "data_types": ["scRNA-seq", "Spatial transcriptomics", "Proteomics"],
            },
            {
                "name": "IEDB",
                "description": "Immune Epitope Database - T cell and B cell epitopes",
                "url": "http://www.iedb.org",
                "data_types": ["Epitopes", "MHC binding", "T cell assays"],
            },
        ]

        for dataset in datasets:
            content = f"""
Dataset: {dataset["name"]}
Description: {dataset["description"]}
URL: {dataset["url"]}
Data Types: {", ".join(dataset["data_types"])}
"""
            doc = Document(
                page_content=content,
                metadata={
                    "source": f"dataset:{dataset['name']}",
                    "type": "dataset",
                    "url": dataset["url"],
                },
            )
            documents.append(doc)

        return documents

    def chunk_and_embed_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents and prepare for embedding

        Args:
            documents: List of documents to process

        Returns:
            List of chunked documents
        """
        chunked_docs = []

        for doc in tqdm(documents, desc="Chunking documents"):
            # Skip if document is already small
            if len(doc.page_content) < 1000:
                chunked_docs.append(doc)
                continue

            # Chunk larger documents
            chunks = self.text_splitter.split_documents([doc])

            # Preserve metadata
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
                chunked_docs.append(chunk)

        return chunked_docs

    def add_to_qdrant(self, documents: List[Document], batch_size: int = 50):
        """
        Add documents to Qdrant in batches

        Args:
            documents: Documents to add
            batch_size: Batch size for adding
        """
        if not documents:
            return

        # Create vector store
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedder,
        )

        # Add in batches
        for i in tqdm(range(0, len(documents), batch_size), desc="Adding to Qdrant"):
            batch = documents[i : i + batch_size]
            try:
                vector_store.add_documents(batch)
                self.stats["chunks_created"] += len(batch)
            except Exception as e:
                print(f"❌ Error adding batch: {e}")
                self.stats["errors"].append(f"Batch {i}: {str(e)}")

    def deploy_complete_knowledge_base(self, recreate: bool = False):
        """
        Deploy complete knowledge base with 800+ papers

        Args:
            recreate: Whether to recreate the collection
        """
        print("=" * 60)
        print("🚀 ImmuneAgent Qdrant Deployment")
        print("=" * 60)

        # Initialize collection
        if not self.initialize_collection(recreate=recreate):
            return

        all_documents = []

        # 1. Load knowledge base
        print("\n📚 Loading Immunology Knowledge Base...")
        kb_docs = self.load_immunology_knowledge_base()
        all_documents.extend(kb_docs)
        self.stats["sources"].append(f"Knowledge Base: {len(kb_docs)} articles")

        # 2. Load datasets metadata
        print("\n📊 Loading Dataset Descriptions...")
        dataset_docs = self.load_immunology_datasets()
        all_documents.extend(dataset_docs)
        self.stats["sources"].append(f"Datasets: {len(dataset_docs)} sources")

        # 3. Load PubMed papers
        pubmed_queries = [
            "CAR-T cell therapy cancer",
            "checkpoint inhibitor immunotherapy",
            "antibody engineering humanization",
            "T cell exhaustion",
            "tumor microenvironment immune",
            "single cell RNA sequencing immunology",
            "BCR repertoire analysis",
            "neoantigen vaccine",
            "CRISPR screen immunology",
            "cytokine storm COVID-19",
        ]

        print("\n📄 Loading PubMed Papers...")
        for query in pubmed_queries:
            papers = self.load_pubmed_papers(query, max_papers=100)
            all_documents.extend(papers)
            self.stats["sources"].append(f"PubMed '{query}': {len(papers)} papers")
            time.sleep(1)  # Rate limiting

        # 4. Load bioRxiv papers
        print("\n📄 Loading bioRxiv Papers...")
        biorxiv_docs = self.load_biorxiv_papers(category="immunology", max_papers=200)
        all_documents.extend(biorxiv_docs)
        self.stats["sources"].append(f"bioRxiv: {len(biorxiv_docs)} papers")

        # 5. Load arXiv papers (computational immunology)
        print("\n📄 Loading arXiv Papers...")
        try:
            arxiv_docs = self.load_arxiv_papers(max_papers=100)
            all_documents.extend(arxiv_docs)
            self.stats["sources"].append(f"arXiv: {len(arxiv_docs)} papers")
        except:
            print("   ⚠️ arXiv loading skipped (install arxiv package if needed)")

        # 6. Chunk documents
        print(f"\n✂️ Chunking {len(all_documents)} documents...")
        chunked_documents = self.chunk_and_embed_documents(all_documents)
        print(f"   Created {len(chunked_documents)} chunks")

        # 7. Add to Qdrant
        print("\n💾 Adding to Qdrant vector store...")
        self.add_to_qdrant(chunked_documents)

        # 8. Print statistics
        self.print_statistics()

        return self.stats

    def print_statistics(self):
        """Print deployment statistics"""
        print("\n" + "=" * 60)
        print("📊 Deployment Statistics")
        print("=" * 60)
        print(f"Collection: {self.collection_name}")
        print(f"Papers loaded: {self.stats['papers_loaded']}")
        print(f"Chunks created: {self.stats['chunks_created']}")
        print("\nSources:")
        for source in self.stats["sources"]:
            print(f"  - {source}")

        if self.stats["errors"]:
            print("\n⚠️ Errors encountered:")
            for error in self.stats["errors"]:
                print(f"  - {error}")

        # Get final collection stats
        try:
            collection_info = self.client.get_collection(self.collection_name)
            print(
                f"\n✅ Final collection size: {collection_info.vectors_count} vectors"
            )
        except:
            pass

    def search_test(self, query: str = "CAR-T cell therapy", k: int = 5):
        """
        Test search functionality

        Args:
            query: Test query
            k: Number of results
        """
        print(f"\n🔍 Testing search: '{query}'")

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedder,
        )

        results = vector_store.similarity_search_with_score(query, k=k)

        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. Score: {1 - score:.3f}")
            print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Content: {doc.page_content[:200]}...")


def main():
    """Main deployment function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deploy ImmuneAgent Qdrant Knowledge Base"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Recreate collection from scratch"
    )
    parser.add_argument(
        "--collection", default="immunology_production", help="Collection name"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run search test after deployment"
    )
    parser.add_argument(
        "--test-query", default="CAR-T cell therapy", help="Test search query"
    )

    args = parser.parse_args()

    # Create deployment manager
    deployment = ImmuneAgentQdrantDeployment(collection_name=args.collection)

    # Deploy knowledge base
    stats = deployment.deploy_complete_knowledge_base(recreate=args.recreate)

    # Run test if requested
    if args.test:
        deployment.search_test(query=args.test_query)

    print("\n✅ Deployment complete!")
    print(f"   Collection: {args.collection}")
    print(f"   Total chunks: {stats['chunks_created']}")


if __name__ == "__main__":
    main()
