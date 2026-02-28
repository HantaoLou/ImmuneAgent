"""Reference database tools for biological and immunological data.

This module provides 10 synchronous functions for querying public reference
databases without external dependencies beyond Python stdlib. All functions:
- Accept primitive types (str, int, bool)
- Return formatted strings truncated to 6000 chars
- Use urllib.request for HTTP (no requests/httpx)
- Load API keys from .env when needed
"""

import os
import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Optional

from ._output import truncate_output, ProteinSummary

logger = logging.getLogger(__name__)

_env_loaded = False


def _ensure_env_loaded():
    """Load environment variables from .env files if not already loaded."""
    global _env_loaded
    if _env_loaded:
        return
    try:
        from dotenv import load_dotenv
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        for env_path in [
            project_root / ".env",
            current_dir.parent / "nodes" / "subagents" / "deep_research" / ".env",
        ]:
            if env_path.exists():
                load_dotenv(env_path, override=False)
    except ImportError:
        pass
    _env_loaded = True


def _fetch_json(url: str, headers: dict = None, timeout: int = 30) -> dict:
    """Fetch JSON from a URL using stdlib urllib.

    Args:
        url: URL to fetch
        headers: Optional HTTP headers
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON as dict

    Raises:
        urllib.error.URLError: Network errors
        json.JSONDecodeError: Invalid JSON
    """
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, data: dict, headers: dict = None, timeout: int = 30) -> dict:
    """POST JSON data and return parsed JSON response.

    Args:
        url: URL to POST to
        data: Dict to serialize as JSON body
        headers: Optional HTTP headers
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response as dict

    Raises:
        urllib.error.URLError: Network errors
        json.JSONDecodeError: Invalid JSON
    """
    json_bytes = json.dumps(data).encode("utf-8")
    headers = headers or {}
    headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=json_bytes, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# R1: UniProt
# ---------------------------------------------------------------------------
def query_uniprot(
    accession: Optional[str] = None,
    gene_name: Optional[str] = None,
    organism: Optional[str] = None,
    max_results: int = 10,
    detailed: bool = False,
) -> str:
    """Query UniProt protein database.

    Search proteins by accession ID, gene name, or organism. Returns protein
    metadata including name, function, domains, and GO annotations.

    Args:
        accession: UniProt accession (e.g., "P12345")
        gene_name: Gene symbol (e.g., "TP53")
        organism: Organism name (e.g., "human", "Homo sapiens")
        max_results: Maximum number of results to return
        detailed: If True, include domains, GO terms, and subcellular location

    Returns:
        Formatted string with protein entries

    Examples:
        >>> query_uniprot(gene_name="IGHV3-23", organism="human")
        >>> query_uniprot(accession="P01857", detailed=True)
    """
    try:
        # Build query string
        query_parts = []
        if accession:
            query_parts.append(f"accession:{accession}")
        if gene_name:
            query_parts.append(f"gene:{gene_name}")
        if organism:
            query_parts.append(f"organism_name:{organism}")

        if not query_parts:
            return "[UniProt] Error: At least one of accession, gene_name, or organism required"

        query_str = " AND ".join(query_parts)
        params = urllib.parse.urlencode({
            "query": query_str,
            "format": "json",
            "size": str(max_results),
        })

        url = f"https://rest.uniprot.org/uniprotkb/search?{params}"
        data = _fetch_json(url, timeout=30)

        results = data.get("results", [])
        if not results:
            return "[UniProt] No results found"

        lines = [f"[UniProt] Found {len(results)} results:\n"]

        for i, entry in enumerate(results, 1):
            parts = [f"--- Result {i} ---"]

            # Basic info
            accession_id = entry.get("primaryAccession", "N/A")
            parts.append(f"  Accession: {accession_id}")

            # Protein name
            protein_desc = entry.get("proteinDescription", {})
            rec_name = protein_desc.get("recommendedName", {})
            full_name = rec_name.get("fullName", {}).get("value", "N/A")
            parts.append(f"  Name: {full_name}")

            # Gene
            genes = entry.get("genes", [])
            if genes:
                gene_names = ", ".join(g.get("geneName", {}).get("value", "") for g in genes if g.get("geneName"))
                if gene_names:
                    parts.append(f"  Gene: {gene_names}")

            # Organism
            organism_obj = entry.get("organism", {})
            sci_name = organism_obj.get("scientificName", "N/A")
            parts.append(f"  Organism: {sci_name}")

            # Function summary (from comments)
            if not detailed:
                comments = entry.get("comments", [])
                for comment in comments:
                    if comment.get("commentType") == "FUNCTION":
                        texts = comment.get("texts", [])
                        if texts:
                            func_text = texts[0].get("value", "")[:200]
                            parts.append(f"  Function: {func_text}")
                            break

            # Detailed info
            if detailed:
                # Full function
                comments = entry.get("comments", [])
                for comment in comments:
                    if comment.get("commentType") == "FUNCTION":
                        texts = comment.get("texts", [])
                        if texts:
                            func_text = texts[0].get("value", "")
                            parts.append(f"  Function: {func_text}")
                    elif comment.get("commentType") == "SUBCELLULAR_LOCATION":
                        sublocations = comment.get("subcellularLocations", [])
                        if sublocations:
                            loc = sublocations[0].get("location", {}).get("value", "")
                            parts.append(f"  Subcellular Location: {loc}")

                # Domains
                features = entry.get("features", [])
                domain_list = [f.get("description", "") for f in features if f.get("type") == "Domain"]
                if domain_list:
                    parts.append(f"  Domains: {', '.join(domain_list[:5])}")

                # GO annotations
                go_terms = []
                for ref in entry.get("uniProtKBCrossReferences", []):
                    if ref.get("database") == "GO":
                        props = ref.get("properties", [])
                        for prop in props:
                            if prop.get("key") == "GoTerm":
                                go_terms.append(prop.get("value", ""))
                if go_terms:
                    parts.append(f"  GO Terms: {', '.join(go_terms[:5])}")

            lines.append("\n".join(parts))

        text = "\n\n".join(lines)
        return truncate_output(text)

    except urllib.error.URLError as e:
        logger.error(f"UniProt network error: {e}")
        return f"[UniProt] Error: Network request failed - {e}"
    except Exception as e:
        logger.error(f"UniProt error: {e}")
        return f"[UniProt] Error: {e}"


# ---------------------------------------------------------------------------
# R2: InterPro
# ---------------------------------------------------------------------------
def query_interpro(
    protein_accession: Optional[str] = None,
    type_filter: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Query InterPro protein families and domains database.

    Search for protein domains, families, and functional sites. Can query by
    UniProt accession or search by keywords.

    Args:
        protein_accession: UniProt accession to look up domains for
        type_filter: Filter by entry type: "domain", "family", "homologous_superfamily", "site", etc.
        max_results: Maximum number of results to return

    Returns:
        Formatted string with InterPro entries

    Examples:
        >>> query_interpro(protein_accession="P01857")
        >>> query_interpro(protein_accession="P12345", type_filter="domain")
    """
    try:
        if not protein_accession:
            return "[InterPro] Error: protein_accession required"

        url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{protein_accession}/?page_size={max_results}"
        data = _fetch_json(url, timeout=30)

        results = data.get("results", [])
        if not results:
            return f"[InterPro] No entries found for protein {protein_accession}"

        # Apply type filter if specified
        if type_filter:
            results = [r for r in results if r.get("metadata", {}).get("type", "").lower() == type_filter.lower()]
            if not results:
                return f"[InterPro] No entries of type '{type_filter}' found"

        lines = [f"[InterPro] Found {len(results)} entries for {protein_accession}:\n"]

        for i, entry in enumerate(results[:max_results], 1):
            parts = [f"--- Entry {i} ---"]

            metadata = entry.get("metadata", {})
            parts.append(f"  Entry ID: {metadata.get('accession', 'N/A')}")
            parts.append(f"  Name: {metadata.get('name', 'N/A')}")
            parts.append(f"  Type: {metadata.get('type', 'N/A')}")

            # Description
            desc = metadata.get("description", [])
            if desc:
                desc_text = desc[0] if isinstance(desc, list) else str(desc)
                parts.append(f"  Description: {desc_text[:200]}")

            # Member databases
            member_dbs = metadata.get("member_databases", {})
            if member_dbs:
                parts.append(f"  Member Databases: {len(member_dbs)}")

            # GO terms
            go_terms = metadata.get("go_terms", [])
            if go_terms:
                go_list = [f"{g.get('identifier', '')} ({g.get('name', '')})" for g in go_terms[:3]]
                parts.append(f"  GO Terms: {', '.join(go_list)}")

            lines.append("\n".join(parts))

        text = "\n\n".join(lines)
        return truncate_output(text)

    except urllib.error.URLError as e:
        logger.error(f"InterPro network error: {e}")
        return f"[InterPro] Error: Network request failed - {e}"
    except Exception as e:
        logger.error(f"InterPro error: {e}")
        return f"[InterPro] Error: {e}"


# ---------------------------------------------------------------------------
# R3: IMGT (via OGRDB as more accessible alternative)
# ---------------------------------------------------------------------------
def query_imgt(
    gene: Optional[str] = None,
    species: str = "human",
    locus: Optional[str] = None,
) -> str:
    """Query IMGT germline gene database via OGRDB.

    Search for immunoglobulin and T-cell receptor germline genes. Uses OGRDB
    (Open Germline Receptor Database) as a more accessible IMGT-compatible API.

    Args:
        gene: Gene name (e.g., "IGHV3-23", "TRBV12-1")
        species: Species name, default "human" (also: "mouse", "rabbit")
        locus: Receptor locus (IGH, IGK, IGL, TRA, TRB, TRD, TRG)

    Returns:
        Formatted string with germline gene entries

    Examples:
        >>> query_imgt(gene="IGHV3-23", species="human")
        >>> query_imgt(locus="IGH", species="human")
    """
    try:
        # Map species names
        species_map = {
            "human": "Homo sapiens",
            "mouse": "Mus musculus",
            "rabbit": "Oryctolagus cuniculus",
        }
        species_name = species_map.get(species.lower(), species)

        # Use OGRDB REST API
        url = "https://ogrdb.airr-community.org/api/germline/sets"
        data = _fetch_json(url, timeout=30)

        if not data:
            return "[IMGT/OGRDB] No data returned from API"

        # Filter by species
        filtered = [item for item in data if item.get("species", "").lower() == species_name.lower()]

        # Filter by locus if specified
        if locus:
            filtered = [item for item in filtered if locus.upper() in item.get("locus", "").upper()]

        # Filter by gene if specified
        if gene:
            gene_upper = gene.upper()
            filtered = [item for item in filtered if gene_upper in item.get("set_name", "").upper()]

        if not filtered:
            return f"[IMGT/OGRDB] No results found for species={species}, locus={locus}, gene={gene}"

        lines = [f"[IMGT/OGRDB] Found {len(filtered)} germline sets:\n"]

        for i, item in enumerate(filtered[:10], 1):
            parts = [f"--- Set {i} ---"]
            parts.append(f"  Set Name: {item.get('set_name', 'N/A')}")
            parts.append(f"  Species: {item.get('species', 'N/A')}")
            parts.append(f"  Locus: {item.get('locus', 'N/A')}")

            # Count alleles
            alleles = item.get("alleles", [])
            parts.append(f"  Alleles: {len(alleles)}")

            # Sample allele names
            if alleles:
                allele_names = [a.get("name", "") for a in alleles[:3]]
                parts.append(f"  Sample Alleles: {', '.join(allele_names)}")

            lines.append("\n".join(parts))

        text = "\n\n".join(lines)
        return truncate_output(text)

    except urllib.error.URLError as e:
        logger.error(f"IMGT/OGRDB network error: {e}")
        return f"[IMGT/OGRDB] Error: Network request failed - {e}"
    except Exception as e:
        logger.error(f"IMGT/OGRDB error: {e}")
        return f"[IMGT/OGRDB] Error: {e}"


# ---------------------------------------------------------------------------
# R4: Reactome
# ---------------------------------------------------------------------------
def query_reactome(
    pathway_id: Optional[str] = None,
    gene: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Query Reactome pathway database.

    Search for biological pathways by identifier or gene participation. Reactome
    provides manually curated pathway data for various species.

    Args:
        pathway_id: Reactome pathway identifier (e.g., "R-HSA-109582")
        gene: Gene symbol to find pathways for
        max_results: Maximum number of results (for gene search)

    Returns:
        Formatted string with pathway entries

    Examples:
        >>> query_reactome(gene="TP53")
        >>> query_reactome(pathway_id="R-HSA-109582")
    """
    try:
        if pathway_id:
            # Query specific pathway by ID
            url = f"https://reactome.org/ContentService/data/query/{pathway_id}"
            data = _fetch_json(url, timeout=30)

            if not data:
                return f"[Reactome] No pathway found with ID {pathway_id}"

            lines = ["[Reactome] Pathway details:\n"]
            parts = []

            parts.append(f"  Pathway ID: {data.get('stId', 'N/A')}")
            parts.append(f"  Name: {data.get('displayName', 'N/A')}")
            parts.append(f"  Type: {data.get('schemaClass', 'N/A')}")

            species = data.get("species", [])
            if species:
                species_names = [s.get("displayName", "") for s in species]
                parts.append(f"  Species: {', '.join(species_names)}")

            summation = data.get("summation", [])
            if summation:
                summary_text = summation[0].get("text", "")[:300]
                parts.append(f"  Summary: {summary_text}")

            lines.append("\n".join(parts))
            text = "\n\n".join(lines)
            return truncate_output(text)

        elif gene:
            # Search pathways by gene
            params = urllib.parse.urlencode({
                "query": gene,
                "types": "Pathway",
                "cluster": "true",
            })
            url = f"https://reactome.org/ContentService/search/query?{params}"
            data = _fetch_json(url, timeout=30)

            results = data.get("results", [])
            if not results:
                return f"[Reactome] No pathways found for gene {gene}"

            lines = [f"[Reactome] Found {len(results)} pathways for {gene}:\n"]

            for i, entry in enumerate(results[:max_results], 1):
                parts = [f"--- Pathway {i} ---"]
                parts.append(f"  Pathway ID: {entry.get('stId', 'N/A')}")
                parts.append(f"  Name: {entry.get('name', 'N/A')}")
                parts.append(f"  Type: {entry.get('type', 'N/A')}")

                species = entry.get("species", "")
                if species:
                    parts.append(f"  Species: {species}")

                exact_match = entry.get("exactType", "")
                if exact_match:
                    parts.append(f"  Match Type: {exact_match}")

                lines.append("\n".join(parts))

            text = "\n\n".join(lines)
            return truncate_output(text)
        else:
            return "[Reactome] Error: Either pathway_id or gene required"

    except urllib.error.URLError as e:
        logger.error(f"Reactome network error: {e}")
        return f"[Reactome] Error: Network request failed - {e}"
    except Exception as e:
        logger.error(f"Reactome error: {e}")
        return f"[Reactome] Error: {e}"


# ---------------------------------------------------------------------------
# R5: STRING-DB
# ---------------------------------------------------------------------------
def query_string_db(
    proteins: Optional[str] = None,
    species: int = 9606,
    score_threshold: int = 400,
    max_results: int = 10,
) -> str:
    """Query STRING protein-protein interaction database.

    Search for known and predicted protein interactions. Can query single protein
    or pipe-separated list for network context.

    Args:
        proteins: Protein name(s), pipe-separated for multiple (e.g., "TP53|MDM2")
        species: NCBI taxonomy ID (9606=human, 10090=mouse)
        score_threshold: Minimum combined interaction score (0-1000)
        max_results: Maximum number of interactions to return

    Returns:
        Formatted string with protein-protein interactions

    Examples:
        >>> query_string_db(proteins="TP53", species=9606, score_threshold=400)
        >>> query_string_db(proteins="TP53|MDM2", species=9606)
    """
    try:
        if not proteins:
            return "[STRING-DB] Error: proteins parameter required"

        params = urllib.parse.urlencode({
            "identifiers": proteins,
            "species": str(species),
            "required_score": str(score_threshold),
            "limit": str(max_results),
        })

        url = f"https://string-db.org/api/json/network?{params}"
        data = _fetch_json(url, timeout=30)

        if not data:
            return f"[STRING-DB] No interactions found for {proteins}"

        lines = [f"[STRING-DB] Found {len(data)} interactions (threshold={score_threshold}):\n"]

        for i, interaction in enumerate(data[:max_results], 1):
            parts = [f"--- Interaction {i} ---"]

            # Protein names
            protein_a = interaction.get("preferredName_A", "N/A")
            protein_b = interaction.get("preferredName_B", "N/A")
            parts.append(f"  Proteins: {protein_a} <-> {protein_b}")

            # Scores
            combined_score = interaction.get("score", 0)
            parts.append(f"  Combined Score: {combined_score}")

            # Evidence channels
            experimental = interaction.get("experimentalScore", 0)
            database = interaction.get("databaseScore", 0)
            textmining = interaction.get("textminingScore", 0)
            coexpression = interaction.get("coexpressionScore", 0)

            evidence = []
            if experimental > 0:
                evidence.append(f"Experimental({experimental})")
            if database > 0:
                evidence.append(f"Database({database})")
            if textmining > 0:
                evidence.append(f"Textmining({textmining})")
            if coexpression > 0:
                evidence.append(f"Coexpression({coexpression})")

            if evidence:
                parts.append(f"  Evidence: {', '.join(evidence)}")

            lines.append("\n".join(parts))

        text = "\n\n".join(lines)
        return truncate_output(text)

    except urllib.error.URLError as e:
        logger.error(f"STRING-DB network error: {e}")
        return f"[STRING-DB] Error: Network request failed - {e}"
    except Exception as e:
        logger.error(f"STRING-DB error: {e}")
        return f"[STRING-DB] Error: {e}"


# ---------------------------------------------------------------------------
# R6: KEGG
# ---------------------------------------------------------------------------
def query_kegg(
    pathway: Optional[str] = None,
    gene: Optional[str] = None,
    organism: str = "hsa",
) -> str:
    """Query KEGG pathway and gene database.

    Search KEGG for metabolic pathways, signaling pathways, and gene information.
    Returns tab-separated text parsed into structured format.

    ACADEMIC USE ONLY. Rate limit: 3 requests/second.

    Args:
        pathway: KEGG pathway ID (e.g., "hsa04110") or search term
        gene: Gene name or KEGG gene ID (e.g., "TP53", "hsa:7157")
        organism: Three-letter organism code (hsa=human, mmu=mouse)

    Returns:
        Formatted string with KEGG entries

    Examples:
        >>> query_kegg(gene="TP53", organism="hsa")
        >>> query_kegg(pathway="cell cycle")
    """
    try:
        if pathway:
            # Search or get pathway
            if pathway.startswith(organism):
                # Direct pathway ID lookup
                url = f"https://rest.kegg.jp/get/{pathway}"
            else:
                # Search pathways
                query_encoded = urllib.parse.quote(pathway)
                url = f"https://rest.kegg.jp/find/pathway/{query_encoded}"

            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8")

            if not text.strip():
                return f"[KEGG] No pathway found for '{pathway}'"

            # Parse tab-separated response
            lines = ["[KEGG] Pathway results:\n"]
            for i, line in enumerate(text.strip().split("\n")[:10], 1):
                if "\t" in line:
                    kegg_id, description = line.split("\t", 1)
                    lines.append(f"--- Result {i} ---\n  KEGG ID: {kegg_id}\n  Description: {description}")
                else:
                    lines.append(f"--- Result {i} ---\n  {line}")

            result = "\n\n".join(lines)
            return truncate_output(result)

        elif gene:
            # Search genes
            query_encoded = urllib.parse.quote(gene)
            url = f"https://rest.kegg.jp/find/genes/{query_encoded}"

            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8")

            if not text.strip():
                return f"[KEGG] No gene found for '{gene}'"

            # Parse tab-separated response
            lines = ["[KEGG] Gene results:\n"]
            for i, line in enumerate(text.strip().split("\n")[:10], 1):
                if "\t" in line:
                    kegg_id, description = line.split("\t", 1)
                    # Split description into gene symbol and description
                    parts_desc = description.split(";", 1)
                    gene_symbol = parts_desc[0].strip()
                    gene_desc = parts_desc[1].strip() if len(parts_desc) > 1 else ""

                    result_parts = [f"--- Result {i} ---"]
                    result_parts.append(f"  KEGG ID: {kegg_id}")
                    result_parts.append(f"  Gene: {gene_symbol}")
                    if gene_desc:
                        result_parts.append(f"  Description: {gene_desc}")

                    lines.append("\n".join(result_parts))
                else:
                    lines.append(f"--- Result {i} ---\n  {line}")

            result = "\n\n".join(lines)
            return truncate_output(result)
        else:
            return "[KEGG] Error: Either pathway or gene required"

    except urllib.error.URLError as e:
        logger.error(f"KEGG network error: {e}")
        return f"[KEGG] Error: Network request failed - {e}"
    except Exception as e:
        logger.error(f"KEGG error: {e}")
        return f"[KEGG] Error: {e}"


# ---------------------------------------------------------------------------
# R7: RCSB PDB
# ---------------------------------------------------------------------------
def query_pdb_search(
    query: Optional[str] = None,
    entity_type: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Query RCSB Protein Data Bank.

    Search for protein structures, ligands, and macromolecular assemblies.

    Args:
        query: Search term (protein name, PDB ID, author name, etc.)
        entity_type: Filter by type - "polymer_entity" (protein), "non_polymer_entity" (ligand)
        max_results: Maximum number of results to return

    Returns:
        Formatted string with PDB entries

    Examples:
        >>> query_pdb_search(query="antibody", max_results=5)
        >>> query_pdb_search(query="7FAE")
    """
    try:
        if not query:
            return "[PDB] Error: query parameter required"

        # Build JSON search query
        search_query = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": query}
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": 0,
                    "rows": max_results
                }
            }
        }

        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        data = _post_json(url, search_query, timeout=30)

        result_set = data.get("result_set", [])
        if not result_set:
            return f"[PDB] No structures found for '{query}'"

        pdb_ids = [item.get("identifier", "") for item in result_set]

        # Fetch details for each PDB entry
        lines = [f"[PDB] Found {data.get('total_count', len(pdb_ids))} structures (showing {len(pdb_ids)}):\n"]

        for i, pdb_id in enumerate(pdb_ids, 1):
            try:
                # Get entry details
                detail_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
                detail = _fetch_json(detail_url, timeout=10)

                parts = [f"--- Structure {i} ---"]
                parts.append(f"  PDB ID: {pdb_id}")

                # Title
                title = detail.get("struct", {}).get("title", "N/A")
                parts.append(f"  Title: {title}")

                # Experimental method
                exp_methods = detail.get("exptl", [])
                if exp_methods:
                    method = exp_methods[0].get("method", "N/A")
                    parts.append(f"  Method: {method}")

                # Resolution
                refine = detail.get("refine", [])
                if refine:
                    resolution = refine[0].get("ls_d_res_high", "N/A")
                    parts.append(f"  Resolution: {resolution} Å")

                # Organism
                entity_src_gen = detail.get("entity_src_gen", [])
                if entity_src_gen:
                    organism = entity_src_gen[0].get("pdbx_gene_src_scientific_name", "")
                    if organism:
                        parts.append(f"  Organism: {organism}")

                lines.append("\n".join(parts))

            except Exception as e:
                logger.warning(f"Failed to fetch details for {pdb_id}: {e}")
                lines.append(f"--- Structure {i} ---\n  PDB ID: {pdb_id}\n  (Details unavailable)")

        text = "\n\n".join(lines)
        return truncate_output(text)

    except urllib.error.URLError as e:
        logger.error(f"PDB network error: {e}")
        return f"[PDB] Error: Network request failed - {e}"
    except Exception as e:
        logger.error(f"PDB error: {e}")
        return f"[PDB] Error: {e}"


# ---------------------------------------------------------------------------
# R8: VDJdb
# ---------------------------------------------------------------------------
def query_vdjdb(
    cdr3: Optional[str] = None,
    gene: Optional[str] = None,
    epitope: Optional[str] = None,
    species: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Query VDJdb T-cell receptor database.

    Search for T-cell receptor (TCR) sequences with known antigen specificity.

    Args:
        cdr3: CDR3 amino acid sequence
        gene: V gene name (e.g., "TRBV12-1")
        epitope: Epitope peptide sequence or name
        species: Species name ("human", "mouse")
        max_results: Maximum number of results to return

    Returns:
        Formatted string with TCR entries

    Examples:
        >>> query_vdjdb(epitope="GILGFVFTL", species="human")
        >>> query_vdjdb(cdr3="CASSLAPGATNEKLFF")
    """
    try:
        # Build query parameters
        params = {"size": str(max_results)}

        if cdr3:
            params["cdr3"] = cdr3
        if gene:
            params["gene"] = gene
        if epitope:
            params["epitope"] = epitope
        if species:
            params["species"] = species

        if len(params) == 1:  # Only has 'size'
            return "[VDJdb] Error: At least one search parameter required (cdr3, gene, epitope, or species)"

        # VDJdb search endpoint
        query_string = urllib.parse.urlencode(params)
        url = f"https://vdjdb.cdr3.net/api/v2/entries?{query_string}"

        data = _fetch_json(url, timeout=30)

        if not isinstance(data, list) or len(data) == 0:
            return "[VDJdb] No TCR entries found"

        lines = [f"[VDJdb] Found {len(data)} TCR entries:\n"]

        for i, entry in enumerate(data[:max_results], 1):
            parts = [f"--- Entry {i} ---"]

            # CDR3 sequence
            cdr3_seq = entry.get("cdr3", "N/A")
            parts.append(f"  CDR3: {cdr3_seq}")

            # V and J genes
            v_gene = entry.get("v.gene", entry.get("vgene", "N/A"))
            j_gene = entry.get("j.gene", entry.get("jgene", "N/A"))
            parts.append(f"  V Gene: {v_gene}")
            parts.append(f"  J Gene: {j_gene}")

            # Epitope
            epitope_seq = entry.get("antigen.epitope", entry.get("epitope", ""))
            if epitope_seq:
                parts.append(f"  Epitope: {epitope_seq}")

            # Antigen
            antigen = entry.get("antigen.gene", entry.get("antigen", ""))
            if antigen:
                parts.append(f"  Antigen: {antigen}")

            # MHC
            mhc_a = entry.get("mhc.a", entry.get("mhc_a", ""))
            mhc_b = entry.get("mhc.b", entry.get("mhc_b", ""))
            if mhc_a:
                parts.append(f"  MHC A: {mhc_a}")
            if mhc_b:
                parts.append(f"  MHC B: {mhc_b}")

            # Species
            species_val = entry.get("species", "")
            if species_val:
                parts.append(f"  Species: {species_val}")

            # Reference
            reference = entry.get("reference.id", entry.get("reference", ""))
            if reference:
                parts.append(f"  Reference: {reference}")

            lines.append("\n".join(parts))

        text = "\n\n".join(lines)
        return truncate_output(text)

    except urllib.error.URLError as e:
        logger.error(f"VDJdb network error: {e}")
        return f"[VDJdb] Error: Network request failed - {e}"
    except Exception as e:
        logger.error(f"VDJdb error: {e}")
        return f"[VDJdb] Error: {e}"


# ---------------------------------------------------------------------------
# R9: IPD-IMGT/HLA
# ---------------------------------------------------------------------------
def query_ipd_hla(
    allele: Optional[str] = None,
    locus: Optional[str] = None,
) -> str:
    """Query IPD-IMGT/HLA human leukocyte antigen database.

    Search for HLA allele sequences and nomenclature.

    Args:
        allele: HLA allele name (e.g., "A*02:01", "DRB1*03:01")
        locus: HLA locus (A, B, C, DRB1, DQB1, etc.)

    Returns:
        Formatted string with HLA allele entries

    Examples:
        >>> query_ipd_hla(allele="A*02:01")
        >>> query_ipd_hla(locus="DRB1")
    """
    try:
        # Build query
        query_str = allele if allele else (locus if locus else "")

        if not query_str:
            return "[IPD-HLA] Error: Either allele or locus required"

        # IPD API endpoint
        params = urllib.parse.urlencode({
            "project": "HLA",
            "limit": "10",
            "query": query_str,
        })

        url = f"https://www.ebi.ac.uk/cgi-bin/ipd/api/allele?{params}"
        data = _fetch_json(url, timeout=30)

        # API returns dict with 'data' array
        alleles = data.get("data", [])
        if not alleles:
            return f"[IPD-HLA] No alleles found for '{query_str}'"

        lines = [f"[IPD-HLA] Found {len(alleles)} HLA alleles:\n"]

        for i, allele_data in enumerate(alleles[:10], 1):
            parts = [f"--- Allele {i} ---"]

            # Allele name
            allele_name = allele_data.get("name", allele_data.get("allele", "N/A"))
            parts.append(f"  Allele: {allele_name}")

            # Locus
            locus_val = allele_data.get("locus", "")
            if locus_val:
                parts.append(f"  Locus: {locus_val}")

            # Sequence length
            seq_length = allele_data.get("sequence_length", allele_data.get("length", ""))
            if seq_length:
                parts.append(f"  Sequence Length: {seq_length}")

            # Accession
            accession = allele_data.get("accession", "")
            if accession:
                parts.append(f"  Accession: {accession}")

            # Status
            status = allele_data.get("status", "")
            if status:
                parts.append(f"  Status: {status}")

            lines.append("\n".join(parts))

        text = "\n\n".join(lines)
        return truncate_output(text)

    except urllib.error.URLError as e:
        logger.error(f"IPD-HLA network error: {e}")
        return f"[IPD-HLA] Error: Network request failed - {e}"
    except Exception as e:
        logger.error(f"IPD-HLA error: {e}")
        return f"[IPD-HLA] Error: {e}"


# ---------------------------------------------------------------------------
# R10: ImmPort
# ---------------------------------------------------------------------------
def query_immport(
    study_accession: Optional[str] = None,
    dataset_type: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Query ImmPort immunology database and analysis portal.

    Search for immunology datasets, studies, and experimental data. Requires
    ImmPort account credentials in .env file.

    REQUIRES: IMMPORT_USERNAME and IMMPORT_PASSWORD in .env

    Args:
        study_accession: ImmPort study accession (e.g., "SDY123")
        dataset_type: Filter by dataset type (e.g., "flow_cytometry", "rna_seq")
        max_results: Maximum number of results to return

    Returns:
        Formatted string with ImmPort study/dataset entries

    Examples:
        >>> query_immport(study_accession="SDY123")
        >>> query_immport(dataset_type="flow_cytometry")
    """
    try:
        _ensure_env_loaded()

        username = os.getenv("IMMPORT_USERNAME")
        password = os.getenv("IMMPORT_PASSWORD")

        if not username or not password:
            return (
                "[ImmPort] Error: Credentials required. Please set IMMPORT_USERNAME and "
                "IMMPORT_PASSWORD in .env file. Register at https://www.immport.org/auth/register"
            )

        # Authenticate to get JWT token
        auth_url = "https://auth.immport.org/auth/token"
        auth_data = json.dumps({
            "username": username,
            "password": password
        }).encode("utf-8")

        auth_req = urllib.request.Request(
            auth_url,
            data=auth_data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(auth_req, timeout=30) as resp:
            auth_response = json.loads(resp.read().decode("utf-8"))
            token = auth_response.get("token")

        if not token:
            return "[ImmPort] Error: Authentication failed"

        # Query ImmPort API
        if study_accession:
            # Get specific study
            query_url = f"https://api.immport.org/data/query/result/study/{study_accession}"
        else:
            # Search studies
            query_url = "https://api.immport.org/data/query/result/study"

        headers = {
            "Authorization": f"bearer {token}",
            "Content-Type": "application/json"
        }

        query_req = urllib.request.Request(query_url, headers=headers)

        with urllib.request.urlopen(query_req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Parse response
        if isinstance(data, dict):
            studies = [data]
        elif isinstance(data, list):
            studies = data
        else:
            return "[ImmPort] Error: Unexpected API response format"

        if not studies:
            return "[ImmPort] No studies found"

        lines = [f"[ImmPort] Found {len(studies)} studies:\n"]

        for i, study in enumerate(studies[:max_results], 1):
            parts = [f"--- Study {i} ---"]

            # Study accession
            accession = study.get("study_accession", study.get("studyAccession", "N/A"))
            parts.append(f"  Study ID: {accession}")

            # Title
            title = study.get("brief_title", study.get("title", "N/A"))
            parts.append(f"  Title: {title}")

            # Condition
            condition = study.get("condition_studied", study.get("condition", ""))
            if condition:
                parts.append(f"  Condition: {condition}")

            # Organism
            organism = study.get("species", "")
            if organism:
                parts.append(f"  Organism: {organism}")

            # Type
            study_type = study.get("study_type", "")
            if study_type:
                parts.append(f"  Type: {study_type}")

            # PI
            pi_name = study.get("pi_name", "")
            if pi_name:
                parts.append(f"  PI: {pi_name}")

            lines.append("\n".join(parts))

        text = "\n\n".join(lines)
        return truncate_output(text)

    except urllib.error.URLError as e:
        logger.error(f"ImmPort network error: {e}")
        return f"[ImmPort] Error: Network request failed - {e}"
    except Exception as e:
        logger.error(f"ImmPort error: {e}")
        return f"[ImmPort] Error: {e}"


# ---------------------------------------------------------------------------
# R11: Infer CDR1/CDR2 from V gene names (for NetTCR format conversion)
# ---------------------------------------------------------------------------

# Cache for IMGT reference data
_IMGT_REFERENCE_CACHE = None


def _load_imgt_reference() -> dict:
    """Load IMGT V gene CDR reference data from JSON file.
    
    Returns:
        Dict with 'trav' and 'trbv' gene CDR1/CDR2 sequences
    """
    global _IMGT_REFERENCE_CACHE
    if _IMGT_REFERENCE_CACHE is not None:
        return _IMGT_REFERENCE_CACHE
    
    config_path = Path(__file__).parent.parent / "config" / "imgt_vgene_cdr_reference.json"
    
    if not config_path.exists():
        logger.warning(f"IMGT reference file not found: {config_path}")
        return {"trav": {}, "trbv": {}}
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _IMGT_REFERENCE_CACHE = json.load(f)
        return _IMGT_REFERENCE_CACHE
    except Exception as e:
        logger.error(f"Failed to load IMGT reference: {e}")
        return {"trav": {}, "trbv": {}}


def _normalize_vgene_name(vgene: str) -> str:
    """Normalize V gene name to standard format.
    
    Handles various formats:
    - TRBV12-1*01 -> TRBV12-1
    - TRBV12-1*02 -> TRBV12-1  
    - TRBV12-1 -> TRBV12-1
    - TRBV12_1 -> TRBV12-1
    - TRBV12-1D -> TRBV12-1
    - TRAV14DV4 -> TRAV14 (DV genes are dual-function, use TRAV CDR)
    - TRAV23DV6 -> TRAV23
    - TRAV29DV5 -> TRAV29
    - TRAV36DV7 -> TRAV36
    
    Args:
        vgene: V gene name in various formats
        
    Returns:
        Normalized V gene name (e.g., "TRBV12-1")
    """
    if not vgene:
        return ""
    
    # Convert to uppercase
    vgene = vgene.upper().strip()
    
    # Remove allele suffix (*01, *02, etc.)
    if "*" in vgene:
        vgene = vgene.split("*")[0]
    
    # Replace underscores with hyphens
    vgene = vgene.replace("_", "-")
    
    # Handle DV dual-function genes (TRAV/DV genes)
    # These genes can rearrange to either TRA or TRD locus
    # Their CDR1/CDR2 sequences are identical to the corresponding TRAV gene
    # Reference: IMGT http://www.imgt.org/IMGTrepertoire/LocusGenes/genetable/human/trav.html
    # TRAV14DV4 -> TRAV14, TRAV23DV6 -> TRAV23, TRAV29DV5 -> TRAV29, TRAV36DV7 -> TRAV36
    dv_gene_mapping = {
        "TRAV14DV4": "TRAV14",
        "TRAV23DV6": "TRAV23", 
        "TRAV29DV5": "TRAV29",
        "TRAV36DV7": "TRAV36",
        "TRAV38-2DV8": "TRAV38-2",  # Note: TRAV38-2 format
    }
    if vgene in dv_gene_mapping:
        return dv_gene_mapping[vgene]
    
    # Also handle patterns like TRAV14/DV4 -> TRAV14
    import re
    dv_pattern = re.match(r'(TRAV\d+(?:-\d+)?)/?DV\d+', vgene)
    if dv_pattern:
        return dv_pattern.group(1)
    
    # Remove D suffix (TRBV12-1D -> TRBV12-1)
    if vgene.endswith("D") and len(vgene) > 4:
        base = vgene[:-1]
        # Only remove D if it looks like a gene name (not ending with digit)
        if base[-1].isdigit():
            vgene = base
    
    return vgene


def _infer_cdr_from_vgene(vgene: str, chain_type: str = "trbv") -> tuple:
    """Infer CDR1 and CDR2 sequences from V gene name.
    
    Args:
        vgene: V gene name (e.g., "TRBV12-1", "TRAV1-1")
        chain_type: Either "trav" or "trbv"
        
    Returns:
        Tuple of (cdr1, cdr2) sequences, or ("", "") if not found
    """
    if not vgene:
        return "", ""
    
    ref_data = _load_imgt_reference()
    chain_data = ref_data.get(chain_type, {})
    
    # Normalize gene name
    normalized = _normalize_vgene_name(vgene)
    
    # Direct lookup
    if normalized in chain_data:
        cdr_data = chain_data[normalized]
        return cdr_data.get("cdr1", ""), cdr_data.get("cdr2", "")
    
    # Try fuzzy matching - remove sub-number (TRAV1-2 -> TRAV1)
    if "-" in normalized:
        base_gene = normalized.rsplit("-", 1)[0]
        # Find genes starting with base
        for gene_name, cdr_data in chain_data.items():
            if gene_name.startswith(base_gene):
                return cdr_data.get("cdr1", ""), cdr_data.get("cdr2", "")
    
    logger.warning(f"V gene not found in reference: {vgene} (normalized: {normalized})")
    return "", ""


def infer_cdr1_cdr2_from_vgene(
    tra_v_gene: Optional[str] = None,
    trb_v_gene: Optional[str] = None,
    cdr3a: Optional[str] = None,
    cdr3b: Optional[str] = None,
    peptide: Optional[str] = None,
) -> str:
    """Infer CDR1 and CDR2 sequences from V gene names using IMGT reference.
    
    CDR1 and CDR2 are germline-encoded regions that can be inferred from V gene
    names. This function uses IMGT reference data to look up these sequences.
    
    Args:
        tra_v_gene: TRAV gene name (e.g., "TRAV1-1", "TRAV1-1*01")
        trb_v_gene: TRBV gene name (e.g., "TRBV12-1", "TRBV12-1*01")
        cdr3a: CDR3 alpha chain amino acid sequence
        cdr3b: CDR3 beta chain amino acid sequence
        peptide: Target peptide sequence
        
    Returns:
        Formatted string with inferred CDR sequences and NetTCR format info
        
    Examples:
        >>> infer_cdr1_cdr2_from_vgene(
        ...     tra_v_gene="TRAV1-1",
        ...     trb_v_gene="TRBV12-1",
        ...     cdr3a="CASSLAPGATNEKLFF",
        ...     cdr3b="CAVRDSNYQLIW",
        ...     peptide="ELAGIGILTV"
        ... )
    """
    try:
        results = []
        results.append("[IMGT CDR Inference] Looking up CDR1/CDR2 from V genes\n")
        results.append("=" * 60)
        
        # Process alpha chain
        if tra_v_gene:
            cdr1a, cdr2a = _infer_cdr_from_vgene(tra_v_gene, "trav")
            results.append(f"\n[Alpha Chain]")
            results.append(f"  V Gene: {tra_v_gene}")
            results.append(f"  CDR1 (A1): {cdr1a if cdr1a else 'NOT FOUND'}")
            results.append(f"  CDR2 (A2): {cdr2a if cdr2a else 'NOT FOUND'}")
            results.append(f"  CDR3 (A3): {cdr3a if cdr3a else 'NOT PROVIDED'}")
        else:
            cdr1a, cdr2a = "", ""
            results.append(f"\n[Alpha Chain] No TRAV gene provided")
        
        # Process beta chain
        if trb_v_gene:
            cdr1b, cdr2b = _infer_cdr_from_vgene(trb_v_gene, "trbv")
            results.append(f"\n[Beta Chain]")
            results.append(f"  V Gene: {trb_v_gene}")
            results.append(f"  CDR1 (B1): {cdr1b if cdr1b else 'NOT FOUND'}")
            results.append(f"  CDR2 (B2): {cdr2b if cdr2b else 'NOT FOUND'}")
            results.append(f"  CDR3 (B3): {cdr3b if cdr3b else 'NOT PROVIDED'}")
        else:
            cdr1b, cdr2b = "", ""
            results.append(f"\n[Beta Chain] No TRBV gene provided")
        
        # NetTCR format summary
        results.append(f"\n[Peptide]: {peptide if peptide else 'NOT PROVIDED'}")
        
        results.append("\n" + "=" * 60)
        results.append("[NetTCR Format Output]")
        results.append("-" * 60)
        
        # Check completeness
        has_a1 = bool(cdr1a)
        has_a2 = bool(cdr2a)
        has_a3 = bool(cdr3a)
        has_b1 = bool(cdr1b)
        has_b2 = bool(cdr2b)
        has_b3 = bool(cdr3b)
        has_peptide = bool(peptide)
        
        completeness = sum([has_a1, has_a2, has_a3, has_b1, has_b2, has_b3, has_peptide])
        total_required = 7
        
        if completeness == total_required:
            results.append("Status: ✓ COMPLETE - Ready for NetTCR prediction")
        else:
            missing = []
            if not has_a1: missing.append("A1")
            if not has_a2: missing.append("A2")
            if not has_a3: missing.append("A3")
            if not has_b1: missing.append("B1")
            if not has_b2: missing.append("B2")
            if not has_b3: missing.append("B3")
            if not has_peptide: missing.append("peptide")
            results.append(f"Status: ⚠ INCOMPLETE - Missing: {', '.join(missing)}")
        
        results.append(f"\nNetTCR Column Format:")
        results.append(f"  A1 = {cdr1a}")
        results.append(f"  A2 = {cdr2a}")
        results.append(f"  A3 = {cdr3a if cdr3a else ''}")
        results.append(f"  B1 = {cdr1b}")
        results.append(f"  B2 = {cdr2b}")
        results.append(f"  B3 = {cdr3b if cdr3b else ''}")
        results.append(f"  peptide = {peptide if peptide else ''}")
        
        text = "\n".join(results)
        return truncate_output(text)
        
    except Exception as e:
        logger.error(f"CDR inference error: {e}")
        return f"[IMGT CDR Inference] Error: {e}"


def convert_tcr_to_nettcr_format(
    input_data: str,
    peptide: str,
    tra_vgene_column: Optional[str] = None,
    trb_vgene_column: Optional[str] = None,
    cdr3a_column: Optional[str] = None,
    cdr3b_column: Optional[str] = None,
    id_column: Optional[str] = None,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Convert TCR data with V gene names to NetTCR format CSV.
    
    This function reads a CSV file containing TCR V gene names and CDR3 sequences,
    infers CDR1 and CDR2 from the V gene names using IMGT reference data, and
    outputs a CSV file in NetTCR format (A1, A2, A3, B1, B2, B3, peptide).
    
    **INPUT COLUMN DETECTION:**
    The function auto-detects columns with these patterns (case-insensitive):
    - V gene columns: TRA_v_gene, TRAV, v_a_gene, alpha_v, TRB_v_gene, TRBV, v_b_gene, beta_v
    - CDR3 columns: CDR3a, cdr3a, CDR3_alpha, alpha_cdr3, CDR3b, cdr3b, CDR3_beta, beta_cdr3
    
    Args:
        input_data: Path to input CSV file
        peptide: Target peptide sequence (same for all rows)
        tra_vgene_column: Column name for TRAV gene (auto-detected if not specified)
        trb_vgene_column: Column name for TRBV gene (auto-detected if not specified)
        cdr3a_column: Column name for CDR3 alpha (auto-detected if not specified)
        cdr3b_column: Column name for CDR3 beta (auto-detected if not specified)
        id_column: Column name for row identifier (e.g., "main_name", "id")
        output_path: Full output file path (takes precedence over output_dir)
        output_dir: Output directory (e.g., sandbox output directory). If specified, 
                    output file will be saved to {output_dir}/{input_filename}_nettcr_format.csv
        
    Returns:
        Formatted string with conversion summary and output file path
        
    Examples:
        >>> convert_tcr_to_nettcr_format(
        ...     input_data="/path/to/tcr_data.csv",
        ...     peptide="ELAGIGILTV",
        ...     output_dir="/data/sessions/session_id/output"
        ... )
    """
    import csv
    
    try:
        # Read input CSV
        with open(input_data, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = reader.fieldnames or []
        
        if not rows:
            return f"[TCR->NetTCR] Error: No data rows found in {input_data}"
        
        results = []
        results.append(f"[TCR->NetTCR Conversion]")
        results.append(f"Input: {input_data}")
        results.append(f"Rows: {len(rows)}")
        results.append(f"Columns: {columns}")
        
        # Auto-detect column names (filter out empty column names)
        col_map = {c.lower(): c for c in columns if c and c.strip()}
        
        # TRAV column patterns
        trav_patterns = ["tra_v_gene", "trav", "v_a_gene", "alpha_v", "tra_vgene", 
                        "v_alpha", "alpha_vgene", "tra.v.gene"]
        if not tra_vgene_column:
            for pattern in trav_patterns:
                for col_lower, col_orig in col_map.items():
                    # Use more precise matching: pattern must be contained in col_lower (not the reverse)
                    # This avoids matching empty strings which are substrings of everything
                    if pattern in col_lower and len(col_lower) >= len(pattern):
                        tra_vgene_column = col_orig
                        break
                if tra_vgene_column:
                    break
        
        # TRBV column patterns
        trbv_patterns = ["trb_v_gene", "trbv", "v_b_gene", "beta_v", "trb_vgene",
                        "v_beta", "beta_vgene", "trb.v.gene"]
        if not trb_vgene_column:
            for pattern in trbv_patterns:
                for col_lower, col_orig in col_map.items():
                    # Use more precise matching
                    if pattern in col_lower and len(col_lower) >= len(pattern):
                        trb_vgene_column = col_orig
                        break
                if trb_vgene_column:
                    break
        
        # CDR3a column patterns
        cdr3a_patterns = ["cdr3a", "cdr3_a", "cdr3alpha", "alpha_cdr3", "a_cdr3",
                         "tra_cdr3", "cdr3.tra"]
        if not cdr3a_column:
            for pattern in cdr3a_patterns:
                for col_lower, col_orig in col_map.items():
                    if pattern == col_lower:
                        cdr3a_column = col_orig
                        break
                if cdr3a_column:
                    break
        
        # CDR3b column patterns  
        cdr3b_patterns = ["cdr3b", "cdr3_b", "cdr3beta", "beta_cdr3", "b_cdr3",
                         "trb_cdr3", "cdr3.trb"]
        if not cdr3b_column:
            for pattern in cdr3b_patterns:
                for col_lower, col_orig in col_map.items():
                    if pattern == col_lower:
                        cdr3b_column = col_orig
                        break
                if cdr3b_column:
                    break
        
        # ID column patterns
        id_patterns = ["main_name", "id", "name", "sample_id", "cell_id", "barcode"]
        if not id_column:
            for pattern in id_patterns:
                if pattern in col_map:
                    id_column = col_map[pattern]
                    break
        
        results.append(f"\n[Detected Columns]")
        results.append(f"  TRAV gene: {tra_vgene_column or 'NOT FOUND'}")
        results.append(f"  TRBV gene: {trb_vgene_column or 'NOT FOUND'}")
        results.append(f"  CDR3a: {cdr3a_column or 'NOT FOUND'}")
        results.append(f"  CDR3b: {cdr3b_column or 'NOT FOUND'}")
        results.append(f"  ID: {id_column or 'NOT FOUND'}")
        
        # Check required columns
        if not tra_vgene_column and not trb_vgene_column:
            return f"[TCR->NetTCR] Error: No V gene columns found. Expected patterns: {trav_patterns + trbv_patterns}"
        
        # Determine output path
        # Priority: output_path > output_dir > same directory as input
        input_path = Path(input_data)
        
        if output_path:
            # Full output path specified, use it directly
            pass
        elif output_dir:
            # Output directory specified (e.g., sandbox output directory)
            output_dir_path = Path(output_dir)
            output_path = str(output_dir_path / f"{input_path.stem}_nettcr_format.csv")
        else:
            # Default: same directory as input
            output_path = str(input_path.parent / f"{input_path.stem}_nettcr_format.csv")
        
        # Process rows and write output
        # Strategy: Keep all original columns and ADD new columns (A1, A2, A3, B1, B2, B3, peptide)
        # This avoids the need for a separate merge operation later
        nettcr_rows = []
        stats = {"success": 0, "partial": 0, "failed": 0, "missing_cdr1": 0, "missing_cdr2": 0}
        
        for i, row in enumerate(rows):
            # Get values from row
            trav = row.get(tra_vgene_column, "") if tra_vgene_column else ""
            trbv = row.get(trb_vgene_column, "") if trb_vgene_column else ""
            c3a = row.get(cdr3a_column, "") if cdr3a_column else ""
            c3b = row.get(cdr3b_column, "") if cdr3b_column else ""
            
            # Infer CDR1/CDR2
            a1, a2 = _infer_cdr_from_vgene(trav, "trav") if trav else ("", "")
            b1, b2 = _infer_cdr_from_vgene(trbv, "trbv") if trbv else ("", "")
            
            # Track statistics
            if a1 and a2 and b1 and b2:
                stats["success"] += 1
            elif a1 or a2 or b1 or b2:
                stats["partial"] += 1
            else:
                stats["failed"] += 1
            
            if not a1 and trav: stats["missing_cdr1"] += 1
            if not a2 and trav: stats["missing_cdr2"] += 1
            
            # Create output row: keep ALL original columns + add NetTCR columns at the end
            # Start with original columns
            nettcr_row = dict(row)
            # Then add NetTCR columns
            nettcr_row["A1"] = a1
            nettcr_row["A2"] = a2
            nettcr_row["A3"] = c3a
            nettcr_row["B1"] = b1
            nettcr_row["B2"] = b2
            nettcr_row["B3"] = c3b
            nettcr_row["peptide"] = peptide
            nettcr_rows.append(nettcr_row)
        
        # Build fieldnames: original columns first, then NetTCR columns at the end
        # This preserves the original column order and adds new columns at the end
        nettcr_columns = ["A1", "A2", "A3", "B1", "B2", "B3", "peptide"]
        # Original columns, excluding the NetTCR columns to avoid duplication
        original_columns = [c for c in columns if c not in nettcr_columns]
        fieldnames = original_columns + nettcr_columns
        
        # Write output CSV
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(nettcr_rows)
        
        # Summary
        results.append(f"\n[Conversion Statistics]")
        results.append(f"  Total rows: {len(rows)}")
        results.append(f"  Complete (all CDRs found): {stats['success']}")
        results.append(f"  Partial (some CDRs found): {stats['partial']}")
        results.append(f"  Failed (no CDRs found): {stats['failed']}")
        results.append(f"  Missing CDR1 lookups: {stats['missing_cdr1']}")
        results.append(f"  Missing CDR2 lookups: {stats['missing_cdr2']}")
        
        results.append(f"\n[Output]")
        results.append(f"  File: {output_path}")
        results.append(f"  Format: NetTCR (A1, A2, A3, B1, B2, B3, peptide)")
        
        text = "\n".join(results)
        return truncate_output(text)
        
    except Exception as e:
        logger.error(f"TCR->NetTCR conversion error: {e}")
        return f"[TCR->NetTCR] Error: {e}"


def get_reference_tools() -> dict:
    """Return a registry of all reference database tool functions.

    Returns:
        Dict mapping function names to callable functions
    """
    return {
        "query_uniprot": query_uniprot,
        "query_interpro": query_interpro,
        "query_imgt": query_imgt,
        "query_reactome": query_reactome,
        "query_string_db": query_string_db,
        "query_kegg": query_kegg,
        "query_pdb_search": query_pdb_search,
        "query_vdjdb": query_vdjdb,
        "query_ipd_hla": query_ipd_hla,
        "query_immport": query_immport,
        "infer_cdr1_cdr2_from_vgene": infer_cdr1_cdr2_from_vgene,
        "convert_tcr_to_nettcr_format": convert_tcr_to_nettcr_format,
    }
