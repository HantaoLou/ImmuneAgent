"""Clinical data tools for ImmuneAgent.

Provides 5 clinical data retrieval functions using stdlib urllib.request:
    1. search_clinical_trials - ClinicalTrials.gov v2.0 API
    2. query_open_targets - Open Targets Platform GraphQL API
    3. search_clinvar - ClinVar via NCBI E-utilities
    4. query_pharmgkb - PharmGKB REST API
    5. search_openfda - OpenFDA adverse events API

All functions are synchronous, return formatted strings, and use truncate_output()
to cap responses at MAX_OUTPUT_CHARS.

LangChain 1.0+ Compatibility:
    - All tools use @tool decorator from langchain_core.tools
    - Tools can be directly bound to LLM via .bind_tools()
"""

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from ._output import TrialDetail, TrialSummary, truncate_output

logger = logging.getLogger(__name__)

_env_loaded = False


def _ensure_env_loaded():
    """Load .env files for API keys (NCBI_API_KEY, OPENFDA_API_KEY)."""
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
    """Fetch JSON from URL with error handling."""
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(
    url: str, body: dict, headers: dict = None, timeout: int = 30
) -> dict:
    """POST JSON to URL and return JSON response."""
    default_headers = {"Content-Type": "application/json"}
    if headers:
        default_headers.update(headers)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=default_headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# C1: ClinicalTrials.gov
# ---------------------------------------------------------------------------


@tool
def search_clinical_trials(
    condition: Optional[str] = None,
    intervention: Optional[str] = None,
    status: Optional[str] = None,
    phase: Optional[str] = None,
    max_results: int = 10,
    detailed: bool = False,
) -> str:
    """Search clinical trials on ClinicalTrials.gov v2.0 API.

    Args:
        condition: Disease/condition (e.g., "COVID-19", "Breast Cancer")
        intervention: Drug/treatment (e.g., "Pembrolizumab")
        status: Trial status - RECRUITING, COMPLETED, ACTIVE_NOT_RECRUITING, etc.
        phase: PHASE1, PHASE2, PHASE3, PHASE4, EARLY_PHASE1
        max_results: Max number of results (default: 10)
        detailed: If True, return detailed trial info; else compact summary

    Returns:
        Formatted string with trial summaries or details
    """
    try:
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        params = {}

        if condition:
            params["query.cond"] = condition
        if intervention:
            params["query.intr"] = intervention
        if status:
            params["filter.overallStatus"] = status
        if phase:
            params["filter.phase"] = phase
        params["pageSize"] = str(max_results)

        query_string = urllib.parse.urlencode(params)
        url = f"{base_url}?{query_string}"

        response = _fetch_json(url)

        if "studies" not in response or not response["studies"]:
            return f"[search_clinical_trials] No trials found for the given criteria."

        studies = response["studies"]
        results = []

        for study in studies[:max_results]:
            try:
                protocol = study.get("protocolSection", {})
                ident_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                design_module = protocol.get("designModule", {})

                nct_id = ident_module.get("nctId", "N/A")
                title = ident_module.get("briefTitle", "No title")
                overall_status = status_module.get("overallStatus", "Unknown")
                phases = design_module.get("phases", [])
                phase_str = ", ".join(phases) if phases else None

                if detailed:
                    # Extract detailed fields
                    cond_module = protocol.get("conditionsModule", {})
                    intervention_module = protocol.get("armsInterventionsModule", {})
                    eligibility_module = protocol.get("eligibilityModule", {})
                    outcomes_module = protocol.get("outcomesModule", {})

                    conditions = ", ".join(cond_module.get("conditions", []))
                    interventions = ", ".join(
                        [
                            interv.get("name", "")
                            for interv in intervention_module.get("interventions", [])
                        ]
                    )
                    eligibility_criteria = eligibility_module.get("eligibilityCriteria", "")
                    primary_outcomes = outcomes_module.get("primaryOutcomes", [])
                    primary_outcome_str = (
                        primary_outcomes[0].get("measure", "")
                        if primary_outcomes
                        else None
                    )

                    trial = TrialDetail(
                        nct_id=nct_id,
                        title=title,
                        status=overall_status,
                        phase=phase_str,
                        conditions=conditions or None,
                        interventions=interventions or None,
                        eligibility=eligibility_criteria[:500] if eligibility_criteria else None,
                        primary_outcome=primary_outcome_str,
                    )
                else:
                    trial = TrialSummary(
                        nct_id=nct_id,
                        title=title,
                        status=overall_status,
                        phase=phase_str,
                    )

                results.append(trial.to_short())

            except Exception as e:
                logger.warning(f"Error parsing trial: {e}")
                continue

        if not results:
            return f"[search_clinical_trials] No valid trials could be parsed."

        output = f"[search_clinical_trials] Found {len(studies)} trials (showing {len(results)}):\n\n"
        output += "\n\n".join(f"--- Trial {i+1} ---\n{r}" for i, r in enumerate(results))

        return truncate_output(output)

    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error fetching clinical trials: {e}")
        return f"[search_clinical_trials] HTTP Error: {e.code} {e.reason}"
    except Exception as e:
        logger.error(f"Error in search_clinical_trials: {e}")
        return f"[search_clinical_trials] Error: {e}"


# ---------------------------------------------------------------------------
# C2: Open Targets Platform
# ---------------------------------------------------------------------------


@tool
def query_open_targets(
    target: Optional[str] = None,
    disease: Optional[str] = None,
    max_results: int = 10,
    detailed: bool = False,
) -> str:
    """Query Open Targets Platform for target-disease associations.

    Args:
        target: Gene symbol or Ensembl ID (e.g., "BRAF", "ENSG00000157764")
        disease: Disease name (e.g., "melanoma")
        max_results: Max number of results (default: 10)
        detailed: If True, include evidence type scores; else summary only

    Returns:
        Formatted string with target-disease associations
    """
    try:
        url = "https://api.platform.opentargets.org/api/v4/graphql"

        if target and not disease:
            # Search for target info
            query = f"""
            query {{
              search(queryString: "{target}", entityNames: ["target"], page: {{size: {max_results}, index: 0}}) {{
                hits {{
                  id
                  name
                  description
                  entity
                }}
              }}
            }}
            """
            response = _post_json(url, {"query": query})

            if "data" not in response or "search" not in response["data"]:
                return f"[query_open_targets] No data returned for target '{target}'."

            hits = response["data"]["search"].get("hits", [])
            if not hits:
                return f"[query_open_targets] No targets found matching '{target}'."

            results = []
            for hit in hits:
                target_id = hit.get("id", "N/A")
                name = hit.get("name", "N/A")
                desc = hit.get("description", "")
                results.append(f"ID: {target_id} | Name: {name} | Description: {desc[:200]}")

            output = f"[query_open_targets] Found {len(hits)} targets:\n\n"
            output += "\n\n".join(f"--- Target {i+1} ---\n{r}" for i, r in enumerate(results))
            return truncate_output(output)

        elif target:
            # Query for disease associations of a target
            # First, resolve target to Ensembl ID if needed
            target_id = target
            if not target.startswith("ENSG"):
                search_query = f"""
                query {{
                  search(queryString: "{target}", entityNames: ["target"], page: {{size: 1, index: 0}}) {{
                    hits {{
                      id
                    }}
                  }}
                }}
                """
                search_resp = _post_json(url, {"query": search_query})
                hits = search_resp.get("data", {}).get("search", {}).get("hits", [])
                if hits:
                    target_id = hits[0]["id"]
                else:
                    return f"[query_open_targets] Could not resolve target '{target}' to Ensembl ID."

            assoc_query = f"""
            query {{
              target(ensemblId: "{target_id}") {{
                id
                approvedSymbol
                associatedDiseases(page: {{size: {max_results}, index: 0}}) {{
                  rows {{
                    disease {{
                      id
                      name
                    }}
                    score
                    datatypeScores {{
                      id
                      score
                    }}
                  }}
                }}
              }}
            }}
            """
            response = _post_json(url, {"query": assoc_query})

            if "data" not in response or "target" not in response["data"]:
                return f"[query_open_targets] No associations found for target '{target}'."

            target_data = response["data"]["target"]
            symbol = target_data.get("approvedSymbol", target_id)
            assoc_rows = target_data.get("associatedDiseases", {}).get("rows", [])

            if not assoc_rows:
                return f"[query_open_targets] No disease associations found for {symbol}."

            results = []
            for row in assoc_rows:
                disease_info = row.get("disease", {})
                disease_name = disease_info.get("name", "N/A")
                disease_id = disease_info.get("id", "N/A")
                overall_score = row.get("score", 0)

                if detailed:
                    datatype_scores = row.get("datatypeScores", [])
                    evidence_str = ", ".join(
                        [f"{ds['id']}:{ds['score']:.2f}" for ds in datatype_scores]
                    )
                    results.append(
                        f"Disease: {disease_name} | ID: {disease_id} | Score: {overall_score:.3f} | Evidence: {evidence_str}"
                    )
                else:
                    results.append(
                        f"Disease: {disease_name} | ID: {disease_id} | Score: {overall_score:.3f}"
                    )

            output = f"[query_open_targets] Target: {symbol} | {len(assoc_rows)} disease associations:\n\n"
            output += "\n\n".join(f"--- Association {i+1} ---\n{r}" for i, r in enumerate(results))
            return truncate_output(output)

        else:
            return "[query_open_targets] Error: Must provide at least 'target' parameter."

    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error querying Open Targets: {e}")
        return f"[query_open_targets] HTTP Error: {e.code} {e.reason}"
    except Exception as e:
        logger.error(f"Error in query_open_targets: {e}")
        return f"[query_open_targets] Error: {e}"


# ---------------------------------------------------------------------------
# C3: ClinVar via NCBI E-utilities
# ---------------------------------------------------------------------------


@tool
def search_clinvar(
    gene: Optional[str] = None,
    variant: Optional[str] = None,
    condition: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Search ClinVar for genetic variants via NCBI E-utilities.

    Args:
        gene: Gene symbol (e.g., "BRCA1")
        variant: Variant notation (e.g., "c.68_69delAG")
        condition: Clinical condition (e.g., "Breast cancer")
        max_results: Max number of results (default: 10)

    Returns:
        Formatted string with variant info
    """
    try:
        _ensure_env_loaded()

        # Build search query
        query_parts = []
        if gene:
            query_parts.append(f"{gene}[gene]")
        if variant:
            query_parts.append(f"{variant}[variant]")
        if condition:
            query_parts.append(f"{condition}[condition]")

        if not query_parts:
            return "[search_clinvar] Error: Must provide at least one search parameter."

        query = " AND ".join(query_parts)

        # E-search to get IDs
        esearch_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        esearch_params = {
            "db": "clinvar",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
        }

        api_key = os.getenv("NCBI_API_KEY")
        if api_key:
            esearch_params["api_key"] = api_key

        esearch_url = f"{esearch_base}?{urllib.parse.urlencode(esearch_params)}"
        esearch_resp = _fetch_json(esearch_url)

        id_list = esearch_resp.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return f"[search_clinvar] No variants found for query: {query}"

        # E-summary to get details
        esummary_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        esummary_params = {
            "db": "clinvar",
            "id": ",".join(id_list),
            "retmode": "json",
        }
        if api_key:
            esummary_params["api_key"] = api_key

        esummary_url = f"{esummary_base}?{urllib.parse.urlencode(esummary_params)}"
        esummary_resp = _fetch_json(esummary_url)

        results = []
        result_data = esummary_resp.get("result", {})

        for uid in id_list:
            if uid not in result_data:
                continue

            variant_data = result_data[uid]
            variant_id = variant_data.get("accession", uid)
            title = variant_data.get("title", "N/A")
            clinical_sig = variant_data.get("clinical_significance", {}).get("description", "N/A")
            review_status = variant_data.get("clinical_significance", {}).get("review_status", "N/A")

            # Extract gene info
            genes = variant_data.get("genes", [])
            gene_symbols = ", ".join([g.get("symbol", "") for g in genes]) if genes else "N/A"

            results.append(
                f"ID: {variant_id} | Gene: {gene_symbols} | Title: {title} | "
                f"Significance: {clinical_sig} | Review: {review_status}"
            )

        if not results:
            return f"[search_clinvar] No valid variants could be parsed."

        output = f"[search_clinvar] Found {len(id_list)} variants (showing {len(results)}):\n\n"
        output += "\n\n".join(f"--- Variant {i+1} ---\n{r}" for i, r in enumerate(results))

        return truncate_output(output)

    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error searching ClinVar: {e}")
        return f"[search_clinvar] HTTP Error: {e.code} {e.reason}"
    except Exception as e:
        logger.error(f"Error in search_clinvar: {e}")
        return f"[search_clinvar] Error: {e}"


# ---------------------------------------------------------------------------
# C4: PharmGKB
# ---------------------------------------------------------------------------


@tool
def query_pharmgkb(
    gene: Optional[str] = None,
    drug: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Query PharmGKB for pharmacogenomics annotations.

    Args:
        gene: Gene symbol (e.g., "CYP2D6")
        drug: Drug name (e.g., "warfarin")
        max_results: Max number of results (default: 10)

    Returns:
        Formatted string with pharmacogenomics annotations
    """
    try:
        base_url = "https://api.pharmgkb.org/v1/data"

        if not gene and not drug:
            return "[query_pharmgkb] Error: Must provide at least 'gene' or 'drug' parameter."

        # Rate limiting: PharmGKB recommends max 2 req/s
        time.sleep(0.5)

        results = []

        if gene and not drug:
            # Get gene info
            gene_url = f"{base_url}/gene?symbol={urllib.parse.quote(gene)}"
            try:
                gene_resp = _fetch_json(gene_url)
                gene_data = gene_resp.get("data", [])
                if gene_data:
                    gene_info = gene_data[0]
                    gene_id = gene_info.get("id", "N/A")
                    symbol = gene_info.get("symbol", gene)
                    name = gene_info.get("name", "N/A")
                    results.append(f"Gene: {symbol} | ID: {gene_id} | Name: {name}")
            except urllib.error.HTTPError:
                pass

            # Get clinical annotations for gene
            time.sleep(0.5)
            annot_url = f"{base_url}/clinicalAnnotation?location.genes.symbol={urllib.parse.quote(gene)}"
            try:
                annot_resp = _fetch_json(annot_url)
                annotations = annot_resp.get("data", [])[:max_results]

                for annot in annotations:
                    chemicals = annot.get("relatedChemicals", [])
                    drug_names = ", ".join([c.get("name", "") for c in chemicals])
                    phenotypes = annot.get("relatedPhenotypes", [])
                    pheno_names = ", ".join([p.get("name", "") for p in phenotypes])
                    evidence_level = annot.get("level", "N/A")

                    results.append(
                        f"Gene: {gene} | Drug: {drug_names or 'N/A'} | "
                        f"Phenotype: {pheno_names or 'N/A'} | Evidence Level: {evidence_level}"
                    )
            except urllib.error.HTTPError:
                pass

        elif drug:
            # Get clinical annotations for drug
            annot_url = f"{base_url}/clinicalAnnotation?relatedChemicals.name={urllib.parse.quote(drug)}"
            try:
                annot_resp = _fetch_json(annot_url)
                annotations = annot_resp.get("data", [])[:max_results]

                for annot in annotations:
                    genes_list = annot.get("relatedGenes", [])
                    gene_symbols = ", ".join([g.get("symbol", "") for g in genes_list])
                    phenotypes = annot.get("relatedPhenotypes", [])
                    pheno_names = ", ".join([p.get("name", "") for p in phenotypes])
                    evidence_level = annot.get("level", "N/A")

                    results.append(
                        f"Gene: {gene_symbols or 'N/A'} | Drug: {drug} | "
                        f"Phenotype: {pheno_names or 'N/A'} | Evidence Level: {evidence_level}"
                    )
            except urllib.error.HTTPError:
                pass

        if not results:
            return f"[query_pharmgkb] No pharmacogenomics data found."

        output = f"[query_pharmgkb] Found {len(results)} annotations:\n\n"
        output += "\n\n".join(f"--- Annotation {i+1} ---\n{r}" for i, r in enumerate(results))

        return truncate_output(output)

    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error querying PharmGKB: {e}")
        return f"[query_pharmgkb] HTTP Error: {e.code} {e.reason}"
    except Exception as e:
        logger.error(f"Error in query_pharmgkb: {e}")
        return f"[query_pharmgkb] Error: {e}"


# ---------------------------------------------------------------------------
# C5: OpenFDA Adverse Events
# ---------------------------------------------------------------------------


@tool
def search_openfda(
    drug_name: Optional[str] = None,
    event_type: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Search OpenFDA for drug adverse event reports.

    Args:
        drug_name: Drug/medicinal product name (e.g., "aspirin")
        event_type: Adverse event type (e.g., "nausea")
        max_results: Max number of results (default: 10)

    Returns:
        Formatted string with adverse event reports
    """
    try:
        _ensure_env_loaded()

        if not drug_name:
            return "[search_openfda] Error: Must provide 'drug_name' parameter."

        base_url = "https://api.fda.gov/drug/event.json"

        # Build query
        query_parts = [f'patient.drug.medicinalproduct:"{drug_name}"']
        if event_type:
            query_parts.append(f'patient.reaction.reactionmeddrapt:"{event_type}"')

        query = "+AND+".join(query_parts)
        params = {
            "search": query,
            "limit": str(max_results),
        }

        api_key = os.getenv("OPENFDA_API_KEY")
        if api_key:
            params["api_key"] = api_key

        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        response = _fetch_json(url)

        if "results" not in response or not response["results"]:
            return f"[search_openfda] No adverse events found for drug '{drug_name}'."

        results_data = response["results"][:max_results]
        results = []

        for event in results_data:
            # Extract drug info
            patient = event.get("patient", {})
            drugs = patient.get("drug", [])
            drug_names = ", ".join(
                [d.get("medicinalproduct", "") for d in drugs if d.get("medicinalproduct")]
            )

            # Extract reactions
            reactions = patient.get("reaction", [])
            reaction_names = ", ".join(
                [r.get("reactionmeddrapt", "") for r in reactions if r.get("reactionmeddrapt")]
            )

            # Seriousness indicators
            serious = event.get("serious", "0")
            seriousness = "Serious" if serious == "1" else "Non-serious"

            # Received date
            receive_date = event.get("receivedate", "N/A")

            results.append(
                f"Drug: {drug_names or drug_name} | Event: {reaction_names or 'N/A'} | "
                f"Seriousness: {seriousness} | Received: {receive_date}"
            )

        output = f"[search_openfda] Found {len(results_data)} adverse event reports:\n\n"
        output += "\n\n".join(f"--- Report {i+1} ---\n{r}" for i, r in enumerate(results))

        return truncate_output(output)

    except urllib.error.HTTPError as e:
        logger.error(f"HTTP error searching OpenFDA: {e}")
        return f"[search_openfda] HTTP Error: {e.code} {e.reason}"
    except Exception as e:
        logger.error(f"Error in search_openfda: {e}")
        return f"[search_openfda] Error: {e}"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def get_clinical_tools() -> list:
    """Return clinical tools as LangChain tools.

    Returns:
        List of LangChain tool objects that can be directly bound to LLM.
    """
    return [
        search_clinical_trials,
        query_open_targets,
        search_clinvar,
        query_pharmgkb,
        search_openfda,
    ]


def get_clinical_tools_dict() -> dict:
    """Return clinical tools for backward compatibility (namespace injection).

    Returns:
        Dict mapping tool names to functions
    """
    return {
        "search_clinical_trials": search_clinical_trials,
        "query_open_targets": query_open_targets,
        "search_clinvar": search_clinvar,
        "query_pharmgkb": query_pharmgkb,
        "search_openfda": search_openfda,
    }
