"""Simplified wrapper functions for data_lake/tools.

These wrappers convert simple keyword arguments into Pydantic BaseModel objects
required by the underlying data_lake/tools functions. This makes them easy for
LLM agents to call directly in <execute> blocks without constructing Pydantic objects.

All functions return formatted strings suitable for LLM consumption.
"""

import sys
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Ensure data_lake is importable
_project_root = "/data/server/ImmuneAgent_2.0"
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _format_results(results: List[Dict[str, Any]], tool_name: str, max_results: int = 20) -> str:
    """Format query results into a readable string for LLM consumption."""
    if not results:
        return f"[{tool_name}] No results found."
    if results and results[0].get("error"):
        return f"[{tool_name}] Error: {results[0]['error']}"

    truncated = results[:max_results]
    lines = [f"[{tool_name}] Found {len(results)} results (showing {len(truncated)}):\n"]
    for i, row in enumerate(truncated, 1):
        parts = [f"--- Result {i} ---"]
        for k, v in row.items():
            if v is not None and str(v).strip():
                parts.append(f"  {k}: {v}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


# ============================================================
# 1. Knowledge Graph (gene-disease-drug-pathway associations)
# ============================================================
def query_kg(entity_name: str = None, entity_type: str = None,
             relation: str = None, target_type: str = None, limit: int = 50) -> str:
    """Query biomedical knowledge graph for gene-disease-drug-pathway associations.
    8.1M relationships from TxGNN Knowledge Graph.

    Args:
        entity_name: Entity name with fuzzy matching (e.g., 'BRCA1', 'aspirin', 'diabetes')
        entity_type: Filter by type: 'gene/protein', 'disease', 'drug', 'pathway', 'anatomy',
                     'biological_process', 'cellular_component', 'molecular_function'
        relation: Relation type (e.g., 'treats', 'associates', 'protein_protein')
        target_type: Target node type (same options as entity_type)
        limit: Max results (default 50)
    """
    from data_lake.tools.core_query_tools import query_knowledge_graph, KnowledgeGraphQuery, KGNodeType
    type_map = {v.value: v for v in KGNodeType}
    q = KnowledgeGraphQuery(
        entity_name=entity_name,
        entity_type=type_map.get(entity_type) if entity_type else None,
        relation=relation,
        target_type=type_map.get(target_type) if target_type else None,
        limit=limit,
    )
    return _format_results(query_knowledge_graph(q), "Knowledge Graph")


# ============================================================
# 2. Gene Expression (GTEx tissue expression)
# ============================================================
def query_expression(gene: str = None, tissue: str = None,
                     min_expression: float = 0.0, limit: int = 50) -> str:
    """Query GTEx tissue gene expression profiles (TPM across 54 human tissues).

    Args:
        gene: Gene symbol (e.g., 'TP53', 'BRCA1')
        tissue: Tissue name (e.g., 'Brain', 'Liver', 'Blood')
        min_expression: Minimum TPM value (default 0)
        limit: Max results (default 50)
    """
    from data_lake.tools.core_query_tools import query_gtex_expression, GTExQuery
    q = GTExQuery(gene=gene, tissue=tissue, min_expression=min_expression, limit=limit)
    return _format_results(query_gtex_expression(q), "GTEx Expression")


# ============================================================
# 3. Disease-Gene Associations (DisGeNET)
# ============================================================
def query_disease_gene(gene: str = None, disease: str = None,
                       min_score: float = None, limit: int = 50) -> str:
    """Query DisGeNET for disease-gene associations with evidence scores.

    Args:
        gene: Gene symbol (e.g., 'TP53', 'BRCA1')
        disease: Disease name (e.g., 'breast cancer', 'diabetes')
        min_score: Minimum association score (0-1)
        limit: Max results (default 50)
    """
    from data_lake.tools.disease_gene_tools import query_disgenet, DisGeNETQuery
    q = DisGeNETQuery(gene_symbol=gene, disease_name=disease, min_score=min_score, limit=limit)
    return _format_results(query_disgenet(q), "DisGeNET")


# ============================================================
# 4. Gene Info (Ensembl annotations)
# ============================================================
def query_gene(gene_id: str = None, chromosome: str = None, limit: int = 50) -> str:
    """Query Ensembl gene annotations (ID, name, chromosome, position, biotype).

    Args:
        gene_id: Ensembl gene ID (e.g., 'ENSG00000141510')
        chromosome: Chromosome number (e.g., '17', 'X')
        limit: Max results (default 50)
    """
    from data_lake.tools.disease_gene_tools import query_gene_info, GeneInfoQuery
    q = GeneInfoQuery(gene_id=gene_id, chromosome=chromosome, limit=limit)
    return _format_results(query_gene_info(q), "Gene Info")


# ============================================================
# 5. Protein Atlas (protein expression & localization)
# ============================================================
def query_protein_atlas(gene: str = None, tissue: str = None,
                        subcellular_location: str = None, limit: int = 50) -> str:
    """Query Human Protein Atlas for protein expression and subcellular localization.

    Args:
        gene: Gene symbol (e.g., 'TP53')
        tissue: Tissue name
        subcellular_location: Subcellular location (e.g., 'Nucleus', 'Cytoplasm')
        limit: Max results (default 50)
    """
    from data_lake.tools.disease_gene_tools import query_proteinatlas, ProteinAtlasQuery
    q = ProteinAtlasQuery(gene=gene, tissue=tissue, subcellular_location=subcellular_location, limit=limit)
    return _format_results(query_proteinatlas(q), "Protein Atlas")


# ============================================================
# 6. OMIM (Mendelian disease)
# ============================================================
def query_omim(gene: str = None, disease: str = None, limit: int = 50) -> str:
    """Query OMIM for Mendelian inheritance disease-gene associations.

    Args:
        gene: Gene symbol (e.g., 'BRCA1')
        disease: Disease name (e.g., 'breast cancer')
        limit: Max results (default 50)
    """
    from data_lake.tools.disease_gene_tools import query_omim as _query_omim, OMIMQuery
    q = OMIMQuery(gene=gene, disease=disease, limit=limit)
    return _format_results(_query_omim(q), "OMIM")


# ============================================================
# 7. Protein-Protein Interaction (BioGRID PPI)
# ============================================================
def query_ppi(gene_id: str = None, gene_id_b: str = None,
              experiment_type: str = "all", organism_id: int = None, limit: int = 50) -> str:
    """Query BioGRID physical protein-protein interactions.

    Args:
        gene_id: Gene/protein ID (Ensembl ID like 'ENSG00000141510')
        gene_id_b: Second gene ID for specific pair query
        experiment_type: 'all', 'affinity_capture_ms', 'two_hybrid', etc.
        organism_id: 9606=human, 10090=mouse, 559292=yeast
        limit: Max results (default 50)
    """
    from data_lake.tools.ppi_tools import query_ppi as _query_ppi, PPIQuery
    q = PPIQuery(gene_id=gene_id, gene_id_b=gene_id_b,
                 experiment_type=experiment_type, organism_id=organism_id, limit=limit)
    return _format_results(_query_ppi(q), "PPI")


# ============================================================
# 8. Drug-Drug Interaction (DDInter)
# ============================================================
def query_drug_interaction(drug_name: str = None, drug_name_b: str = None,
                           severity: str = "all", limit: int = 50) -> str:
    """Query DDInter for drug-drug interactions and severity levels.

    Args:
        drug_name: Drug name (e.g., 'Metformin', 'Warfarin')
        drug_name_b: Second drug name for specific pair interaction
        severity: 'all', 'Major', 'Moderate', 'Minor'
        limit: Max results (default 50)
    """
    from data_lake.tools.drug_tools import query_drug_interaction as _qdi, DrugInteractionQuery
    q = DrugInteractionQuery(drug_name=drug_name, drug_name_b=drug_name_b,
                             severity=severity, limit=limit)
    return _format_results(_qdi(q), "Drug Interaction")


# ============================================================
# 9. Drug-Target Binding Affinity (BindingDB)
# ============================================================
def query_binding(ligand_name: str = None, target_name: str = None, limit: int = 50) -> str:
    """Query BindingDB for drug-target binding affinity (Ki/Kd/IC50).

    Args:
        ligand_name: Drug/ligand name (e.g., 'Imatinib')
        target_name: Target protein name (e.g., 'ABL1', 'EGFR')
        limit: Max results (default 50)
    """
    from data_lake.tools.core_query_tools import query_bindingdb, BindingDBQuery
    q = BindingDBQuery(ligand_name=ligand_name, target_name=target_name, limit=limit)
    return _format_results(query_bindingdb(q), "BindingDB")


# ============================================================
# 10. Genetic Variant (SNP)
# ============================================================
def query_variant(rs_id: str = None, chromosome: str = None, limit: int = 50) -> str:
    """Query genetic variant data (SNP positions, alleles).

    Args:
        rs_id: RS ID (e.g., 'rs1234567')
        chromosome: Chromosome (e.g., '1', 'X')
        limit: Max results (default 50)
    """
    from data_lake.tools.core_query_tools import query_variant as _qv, VariantQuery
    q = VariantQuery(rs_id=rs_id, chromosome=chromosome, limit=limit)
    return _format_results(_qv(q), "Variant")


# ============================================================
# 11. GWAS Catalog
# ============================================================
def query_gwas(disease_trait: str = None, gene: str = None, snp: str = None,
               p_value_threshold: float = 5e-8, limit: int = 50) -> str:
    """Query GWAS Catalog for SNP-disease/trait associations.

    Args:
        disease_trait: Disease or trait name (e.g., 'diabetes', 'height')
        gene: Gene symbol (e.g., 'TP53')
        snp: SNP rsID (e.g., 'rs429358')
        p_value_threshold: P-value threshold (default 5e-8)
        limit: Max results (default 50)
    """
    from data_lake.tools.gwas_tools import query_gwas_catalog, GWASCatalogQuery
    q = GWASCatalogQuery(disease_trait=disease_trait, gene=gene, snp=snp,
                         p_value_threshold=p_value_threshold, limit=limit)
    return _format_results(query_gwas_catalog(q), "GWAS Catalog")


# ============================================================
# 12. Genebass (rare variant burden analysis)
# ============================================================
def query_genebass(gene: str = None, phenotype: str = None,
                   variant_type: str = "plof", p_value_threshold: float = 1e-6,
                   limit: int = 50) -> str:
    """Query Genebass for gene-level rare variant burden test results (UK Biobank exomes).

    Args:
        gene: Gene symbol (e.g., 'BRCA1', 'APOE')
        phenotype: Phenotype description (e.g., 'diabetes', 'BMI')
        variant_type: 'plof' (loss-of-function), 'missense_lc', 'synonymous'
        p_value_threshold: P-value threshold (default 1e-6)
        limit: Max results (default 50)
    """
    from data_lake.tools.genebass_tools import query_genebass as _qg, GenebassQuery, VariantType
    vt_map = {v.value: v for v in VariantType}
    q = GenebassQuery(gene=gene, phenotype=phenotype,
                      variant_type=vt_map.get(variant_type, VariantType.plof),
                      p_value_threshold=p_value_threshold, limit=limit)
    return _format_results(_qg(q), "Genebass")


# ============================================================
# 13. TCR-Antigen (McPAS-TCR)
# ============================================================
def query_tcr(epitope: str = None, pathology: str = None,
              cdr3_beta: str = None, mhc: str = None, limit: int = 50) -> str:
    """Query McPAS-TCR for T cell receptor sequences and antigen specificity.

    Args:
        epitope: Antigen epitope peptide sequence
        pathology: Disease/pathology (e.g., 'Cancer', 'Influenza', 'COVID')
        cdr3_beta: CDR3β sequence (supports % wildcard)
        mhc: MHC restriction (e.g., 'HLA-A*02:01')
        limit: Max results (default 50)
    """
    from data_lake.tools.core_query_tools import query_tcr_mcpas, TCRQuery
    q = TCRQuery(epitope=epitope, pathology=pathology, cdr3_beta=cdr3_beta, mhc=mhc, limit=limit)
    return _format_results(query_tcr_mcpas(q), "TCR-Antigen")


# ============================================================
# 14. miRNA Target Prediction (miRDB)
# ============================================================
def query_mirna_target(mirna: str = None, target_gene: str = None,
                       min_score: float = 80.0, limit: int = 50) -> str:
    """Query miRDB for computational miRNA target predictions.

    Args:
        mirna: miRNA name (e.g., 'hsa-miR-21-5p')
        target_gene: Target gene symbol (e.g., 'TP53')
        min_score: Minimum prediction score 0-100 (default 80)
        limit: Max results (default 50)
    """
    from data_lake.tools.core_query_tools import query_mirdb, MiRDBQuery
    q = MiRDBQuery(mirna=mirna, target_gene=target_gene, min_score=min_score, limit=limit)
    return _format_results(query_mirdb(q), "miRDB")


# ============================================================
# 15. Validated miRNA-Target (miRTarBase)
# ============================================================
def query_mirna_validated(mirna: str = None, target_gene: str = None,
                          species: str = None, limit: int = 50) -> str:
    """Query miRTarBase for experimentally validated miRNA-target interactions.

    Args:
        mirna: miRNA name (e.g., 'hsa-miR-21-5p')
        target_gene: Target gene symbol (e.g., 'TP53')
        species: Species (e.g., 'Homo sapiens')
        limit: Max results (default 50)
    """
    from data_lake.tools.core_query_tools import query_mirtarbase, MiRTarBaseQuery
    q = MiRTarBaseQuery(mirna=mirna, target_gene=target_gene, species=species, limit=limit)
    return _format_results(query_mirtarbase(q), "miRTarBase")


# ============================================================
# 16. CRISPR sgRNA Design (Human)
# ============================================================
def query_sgrna(target_gene: str, species: str = "human",
                min_efficacy: float = 0.5, limit: int = 20) -> str:
    """Query CRISPR sgRNA design sequences for gene knockout experiments.

    Args:
        target_gene: Target gene symbol (required, e.g., 'TP53')
        species: 'human' or 'mouse' (default 'human')
        min_efficacy: Minimum on-target efficacy score 0-1 (default 0.5)
        limit: Max results (default 20)
    """
    from data_lake.tools.core_query_tools import query_sgrna_human, query_sgrna_mouse, SgRNAQuery
    q = SgRNAQuery(target_gene=target_gene, min_efficacy=min_efficacy, limit=limit)
    if species.lower() == "mouse":
        return _format_results(query_sgrna_mouse(q), "sgRNA-Mouse")
    return _format_results(query_sgrna_human(q), "sgRNA-Human")


# ============================================================
# 17. Gene Ontology (GO)
# ============================================================
def query_go(term_id: str = None, name: str = None, keyword: str = None,
             namespace: str = None, limit: int = 50) -> str:
    """Query Gene Ontology terms by ID, name, or keyword.

    Args:
        term_id: GO term ID (e.g., 'GO:0006955')
        name: GO term name (e.g., 'immune response')
        keyword: Keyword to search in names and definitions (e.g., 'apoptosis')
        namespace: 'biological_process', 'molecular_function', or 'cellular_component'
        limit: Max results (default 50)
    """
    from data_lake.tools.go_tools import query_go_term, GOTermQuery
    kwargs = dict(limit=limit)
    if term_id: kwargs["term_id"] = term_id
    if name: kwargs["name"] = name
    if keyword: kwargs["keyword"] = keyword
    if namespace: kwargs["namespace"] = namespace
    q = GOTermQuery(**kwargs)
    return _format_results(query_go_term(q), "Gene Ontology")


# ============================================================
# 18. Human Phenotype Ontology (HPO)
# ============================================================
def query_hpo(term_id: str = None, name: str = None, keyword: str = None,
              limit: int = 50) -> str:
    """Query Human Phenotype Ontology terms for disease phenotype descriptions.

    Args:
        term_id: HPO term ID (e.g., 'HP:0001250')
        name: HPO term name (e.g., 'Seizure')
        keyword: Keyword to search (e.g., 'seizure', 'cardiomyopathy')
        limit: Max results (default 50)
    """
    from data_lake.tools.hpo_tools import query_hpo_term, HPOTermQuery
    q = HPOTermQuery(term_id=term_id, name=name, keyword=keyword, limit=limit)
    return _format_results(query_hpo_term(q), "HPO")


# ============================================================
# 19. Gene Set (MSigDB)
# ============================================================
def query_geneset(gene_symbol: str = None, geneset_name: str = None,
                  collection: str = "hallmark", limit: int = 50) -> str:
    """Query MSigDB gene sets (pathways, signatures, regulatory targets).

    Args:
        gene_symbol: Gene symbol to find containing gene sets (e.g., 'TP53')
        geneset_name: Gene set name keyword (e.g., 'HALLMARK_APOPTOSIS')
        collection: 'hallmark', 'positional', 'curated', 'regulatory', 'computational',
                    'go_bp', 'go_cc', 'go_mf', 'oncogenic', 'immunologic'
        limit: Max results (default 50)
    """
    from data_lake.tools.geneset_tools import query_msigdb, MSigDBQuery, MSigDBCollection
    col_map = {v.name: v for v in MSigDBCollection}
    q = MSigDBQuery(gene_symbol=gene_symbol, geneset_name=geneset_name,
                    collection=col_map.get(collection, MSigDBCollection.hallmark), limit=limit)
    return _format_results(query_msigdb(q), "MSigDB")


# ============================================================
# 20. Drug Repurposing Predictions (TxGNN)
# ============================================================
def query_drug_for_disease(disease_name: str, min_score: float = 0.5, top_k: int = 20) -> str:
    """Find AI-predicted drugs for a disease (TxGNN graph neural network).

    Args:
        disease_name: Disease name (e.g., 'diabetes', 'breast cancer')
        min_score: Minimum prediction score (default 0.5)
        top_k: Return top K drugs (default 20)
    """
    from data_lake.tools.txgnn_tools import query_drug_for_disease as _qd, DrugForDiseaseQuery
    q = DrugForDiseaseQuery(disease_name=disease_name, min_score=min_score, top_k=top_k)
    return _format_results(_qd(q), "TxGNN Drug-for-Disease")


def query_disease_for_drug(drug_name: str, min_score: float = 0.5, top_k: int = 20) -> str:
    """Find AI-predicted diseases treatable by a drug (TxGNN graph neural network).

    Args:
        drug_name: Drug name (e.g., 'Metformin', 'Aspirin')
        min_score: Minimum prediction score (default 0.5)
        top_k: Return top K diseases (default 20)
    """
    from data_lake.tools.txgnn_tools import query_disease_for_drug as _qd, DiseaseForDrugQuery
    q = DiseaseForDrugQuery(drug_name=drug_name, min_score=min_score, top_k=top_k)
    return _format_results(_qd(q), "TxGNN Disease-for-Drug")


# ============================================================
# 21. DepMap (cancer cell line dependencies)
# ============================================================
def query_depmap(cell_line: str = None, data_type: str = "crispr_dependency",
                 limit: int = 50) -> str:
    """Query DepMap cancer dependency data (CRISPR, RNAi, drug sensitivity).

    Args:
        cell_line: Cell line name (e.g., 'A549', 'MCF7')
        data_type: 'model', 'crispr_dependency', 'rnai_dependency', 'drug_sensitivity'
        limit: Max results (default 50)
    """
    from data_lake.tools.misc_tools import query_depmap as _qd, DepMapQuery, DepMapDataType
    dt_map = {v.name: v for v in DepMapDataType}
    q = DepMapQuery(cell_line=cell_line, data_type=dt_map.get(data_type, DepMapDataType.crispr_dependency), limit=limit)
    return _format_results(_qd(q), "DepMap")


# ============================================================
# 22. Cell Type Markers
# ============================================================
def query_cell_markers(cell_type: str = None, marker_gene: str = None,
                       limit: int = 50) -> str:
    """Query cell type marker genes (e.g., CD3 for T cells).

    Args:
        cell_type: Cell type (e.g., 'T cell', 'B cell', 'Macrophage')
        marker_gene: Marker gene (e.g., 'CD3', 'CD19')
        limit: Max results (default 50)
    """
    from data_lake.tools.misc_tools import query_celltype_marker, CellTypeMarkerQuery
    q = CellTypeMarkerQuery(cell_type=cell_type, marker_gene=marker_gene, limit=limit)
    return _format_results(query_celltype_marker(q), "Cell Markers")


# ============================================================
# 23. Virus-Host PPI
# ============================================================
def query_virus_host(viral_protein: str = None, host_gene: str = None,
                     limit: int = 50) -> str:
    """Query virus-host protein-protein interactions.

    Args:
        viral_protein: Viral protein name
        host_gene: Host gene symbol
        limit: Max results (default 50)
    """
    from data_lake.tools.misc_tools import query_virus_host_ppi, VirusHostPPIQuery
    q = VirusHostPPIQuery(viral_protein=viral_protein, host_gene=host_gene, limit=limit)
    return _format_results(query_virus_host_ppi(q), "Virus-Host PPI")


# ============================================================
# 24. Drug Repurposing (Broad)
# ============================================================
def query_drug_repurposing(drug_name: str = None, target: str = None,
                           moa: str = None, limit: int = 50) -> str:
    """Query Broad Institute drug repurposing hub.

    Args:
        drug_name: Drug name (e.g., 'Metformin')
        target: Drug target
        moa: Mechanism of Action
        limit: Max results (default 50)
    """
    from data_lake.tools.misc_tools import query_broad_repurposing, BroadRepurposingQuery
    q = BroadRepurposingQuery(drug_name=drug_name, target=target, moa=moa, limit=limit)
    return _format_results(query_broad_repurposing(q), "Broad Repurposing")


# ============================================================
# Tool registry: all wrapper functions for injection
# ============================================================
def get_biomedical_tools() -> dict:
    """Return a dict of all biomedical tool wrapper functions for namespace injection."""
    return {
        "query_kg": query_kg,
        "query_expression": query_expression,
        "query_disease_gene": query_disease_gene,
        "query_gene": query_gene,
        "query_protein_atlas": query_protein_atlas,
        "query_omim": query_omim,
        "query_ppi": query_ppi,
        "query_drug_interaction": query_drug_interaction,
        "query_binding": query_binding,
        "query_variant": query_variant,
        "query_gwas": query_gwas,
        "query_genebass": query_genebass,
        "query_tcr": query_tcr,
        "query_mirna_target": query_mirna_target,
        "query_mirna_validated": query_mirna_validated,
        "query_sgrna": query_sgrna,
        "query_go": query_go,
        "query_hpo": query_hpo,
        "query_geneset": query_geneset,
        "query_drug_for_disease": query_drug_for_disease,
        "query_disease_for_drug": query_disease_for_drug,
        "query_depmap": query_depmap,
        "query_cell_markers": query_cell_markers,
        "query_virus_host": query_virus_host,
        "query_drug_repurposing": query_drug_repurposing,
    }
