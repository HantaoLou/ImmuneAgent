"""X-Masters Stage 4: Selector

Selects the best solution from all Rewriter outputs.  The Selector is a single
CodeActAgent instance that receives all N rewritten solutions, verifies each
using code and tools, and picks the most correct one.

Key insight from the official X-Master source code: the Selector is NOT a
simple vote or pass-through.  It is a full CodeActAgent that:
  1. Receives all N rewritten solutions
  2. Verifies each solution using code, web_search, and database tools
  3. Outputs a structured verdict: VERIFICATION → CONCLUSION → <select>
  4. The selected solution becomes the final answer

Architecture:
    Single instance (no fan-out):
        1. Clear execution namespace (isolation)
        2. Inject tools (web_search, biomedical tools)
        3. Create fresh SelectorAgent instance
        4. Run selector.go(problem + all_solutions)
        5. Parse <select>Response X</select> from output
        6. Return the selected solution as final_answer

Data flow:
    Input  — from XMastersState after Stage 3 fan-in:
        problem              (str):        the original question
        rewritten_solutions  (list[dict]): all N rewritten solutions

    Output — written to XMastersState:
        final_answer (str): the selected best solution
"""

import logging
import os
import re
import sys

from .tools import inject_lightweight_tools_to_namespace

# ---------------------------------------------------------------------------
# Path setup: make result_evaluator importable
# ---------------------------------------------------------------------------
_result_evaluator_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "result_evaluator")
)
if _result_evaluator_dir not in sys.path:
    sys.path.insert(0, _result_evaluator_dir)

from agent import CodeActAgent  # noqa: E402
import executor  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Selector system prompt
# ---------------------------------------------------------------------------
SELECTOR_SYSTEM_PROMPT = """\
You are a diligent and precise judge. You should choose the correct response from multiple responses to a problem.

To achieve this, you will be using an interactive coding environment equipped with tool functions.

At each turn, you should first provide your analysis.
After that, you have two options:

1) Interact with a programming environment and receive the corresponding output. Your code should be enclosed using "<execute>" tag, for example:
<execute>
print("Hello World!")
</execute>

2) When you have reached your final verdict, provide the answer using "<solution>" tag containing BOTH your selected response number AND the answer, for example:
<solution> Response 1: 42 </solution>

## Available Tools

You have access to the following pre-loaded tool functions that you can call directly in your code:

### Search Tools
- **web_search(query, max_results=5)** — Search the web for current information and research findings
- **read_webpage(url, max_chars=10000)** — Fetch and read the full text content of a URL (paper, article, documentation). Use this to read papers found via web_search instead of relying only on search snippets

### Biomedical Database Tools (88 tables, pre-loaded — call directly)
- **query_kg(entity_name, entity_type, relation, target_type, limit=50)** — Knowledge graph: gene-disease-drug-pathway associations (8.1M records)
- **query_expression(gene, tissue, min_expression=0, limit=50)** — GTEx tissue gene expression (TPM, 54 tissues)
- **query_disease_gene(gene, disease, min_score, limit=50)** — DisGeNET disease-gene associations
- **query_gene(gene_id, chromosome, limit=50)** — Ensembl gene annotations
- **query_protein_atlas(gene, tissue, subcellular_location, limit=50)** — Human Protein Atlas expression & localization
- **query_omim(gene, disease, limit=50)** — OMIM Mendelian disease-gene associations
- **query_ppi(gene_id, gene_id_b, experiment_type="all", organism_id, limit=50)** — BioGRID protein-protein interactions
- **query_drug_interaction(drug_name, drug_name_b, severity="all", limit=50)** — DDInter drug-drug interactions
- **query_binding(ligand_name, target_name, limit=50)** — BindingDB drug-target binding affinity (Ki/Kd/IC50)
- **query_variant(rs_id, chromosome, limit=50)** — Genetic variant/SNP data
- **query_gwas(disease_trait, gene, snp, p_value_threshold=5e-8, limit=50)** — GWAS Catalog SNP-trait associations
- **query_genebass(gene, phenotype, variant_type="plof", p_value_threshold=1e-6, limit=50)** — UK Biobank rare variant burden tests
- **query_tcr(epitope, pathology, cdr3_beta, mhc, limit=50)** — McPAS-TCR T cell receptor-antigen data
- **query_mirna_target(mirna, target_gene, min_score=80, limit=50)** — miRDB miRNA target predictions
- **query_mirna_validated(mirna, target_gene, species, limit=50)** — miRTarBase validated miRNA-target interactions
- **query_sgrna(target_gene, species="human", min_efficacy=0.5, limit=20)** — CRISPR sgRNA design sequences
- **query_go(term_id, name, keyword, namespace, limit=50)** — Gene Ontology terms
- **query_hpo(term_id, name, keyword, limit=50)** — Human Phenotype Ontology terms
- **query_geneset(gene_symbol, geneset_name, collection="hallmark", limit=50)** — MSigDB gene sets & pathways
- **query_drug_for_disease(disease_name, min_score=0.5, top_k=20)** — TxGNN AI-predicted drugs for disease
- **query_disease_for_drug(drug_name, min_score=0.5, top_k=20)** — TxGNN AI-predicted diseases for drug
- **query_depmap(cell_line, data_type="crispr_dependency", limit=50)** — DepMap cancer cell line dependencies
- **query_cell_markers(cell_type, marker_gene, limit=50)** — Cell type marker genes
- **query_virus_host(viral_protein, host_gene, limit=50)** — Virus-host protein interactions
- **query_drug_repurposing(drug_name, target, moa, limit=50)** — Broad drug repurposing hub

All tools are pre-loaded. Call them directly: `result = query_kg(entity_name="BRCA1"); print(result)`

## Code Execution

- Python (default): <execute> print("Hello") </execute>
- R code: <execute> #!R\nprint("Hello") </execute>
- Bash: <execute> #!BASH\necho "Hello" </execute>

You have access to all standard Python libraries (numpy, scipy, sympy, pandas, etc.) and can write arbitrary code for computation, verification, and analysis.

## CRITICAL RULES

1. **EVERY response MUST contain either <execute> or <solution> tag.** No exceptions.
2. **When you have the final answer, you MUST use <solution> tag immediately.**
3. **Pure text responses without tags are NOT allowed** and will be rejected.
4. **When choosing between answer choices, prefer the most specific and mechanistically precise answer.** If one answer describes a specific mechanism (e.g., a named structural motif or pathway) and another describes a more general phenomenon that could encompass it, choose the specific one. A precise mechanistic explanation is always preferred over a vague or composite description.
5. **When you find a relevant primary research paper, read it carefully using read_webpage() and base your answer on the paper's actual conclusions**, not on textbook-level generalizations.
"""


# ---------------------------------------------------------------------------
# Initial Reasoning Guidance (IRG) for Selector
# ---------------------------------------------------------------------------
SELECTOR_IRG = """\
I need to carefully evaluate each proposed response and select the most correct one. I must not be overconfident or influenced by the majority — I should verify each claim independently.

I have access to an interactive coding environment where I can write and execute Python code. I should leverage this capability to verify each response:
- Query biomedical databases (query_kg, query_expression, query_disease_gene, query_gwas, query_ppi, etc.) to fact-check claims
- Search for relevant information using web_search()
- **Read full paper content** using read_webpage(url) when web_search returns a relevant paper — search snippets are often insufficient for nuanced questions
- Write code to independently re-derive or re-compute any calculations
- Check if the proposed answers are consistent with known facts

CRITICAL: I must not trust the information, references, or assumptions in any response easily. I must write code to verify before reaching a conclusion. I should also not be influenced by the majority number of final answers — they may ALL be wrong due to shared blind spots (e.g., textbook-level generalizations when a specialized mechanism is at play). When all responses agree, I must be EXTRA skeptical and actively search for alternative explanations. I MUST use read_webpage(url) to read the actual content of relevant papers rather than guessing from titles or snippets.

Let me start by examining each response and identifying the key claims to verify.
"""


# ---------------------------------------------------------------------------
# SelectorAgent: CodeActAgent with Selector-specific system prompt + IRG
# ---------------------------------------------------------------------------
class SelectorAgent(CodeActAgent):
    """CodeActAgent specialized for X-Masters solution selection.

    Inherits all functionality from CodeActAgent but overrides:
    - System prompt: instructs the model to verify and select the best solution
    - go() method: injects Initial Reasoning Guidance (IRG) after user message

    The IRG mechanism primes the model to independently verify each solution
    using code before making a selection decision.
    """

    def _generate_system_prompt(self) -> str:
        """Generate Selector-specific system prompt."""
        return SELECTOR_SYSTEM_PROMPT

    def go(self, prompt: str):
        """Execute the selector agent with IRG injection.

        Overrides CodeActAgent.go() to inject Initial Reasoning Guidance
        as an AIMessage after the user's problem + all proposed solutions.
        """
        from langchain_core.messages import HumanMessage, AIMessage

        self.user_task = prompt

        inputs = {
            "messages": [
                HumanMessage(content=prompt),
                AIMessage(content=SELECTOR_IRG.strip()),
            ],
            "next_step": "generate",
        }
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []

        final_state = None
        step = 0
        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            step += 1
            out = self._pretty_print(message)
            self.log.append(out)
            self._verbose_log(step, message)
            final_state = s

        self._conversation_state = final_state
        return self.log, message.content


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _reset_execution_env(namespace: dict = None, captured_plots: list = None):
    """Reset execution environment.

    Args:
        namespace: Instance-level namespace to clear. If None, clears global.
        captured_plots: Instance-level plots list to clear. If None, clears global.
    """
    if namespace is not None:
        namespace.clear()
    else:
        executor._persistent_namespace.clear()
    executor.clear_captured_plots(captured_plots)


def _inject_selector_tools(namespace: dict = None):
    """Inject lightweight tools (no knowledge_search) into the execution namespace.

    Selectors receive pre-retrieved context via their prompt, so they only need
    web_search and biomedical DB tools for real-time verification.

    Args:
        namespace: Target namespace dict. If None, uses global.
    """
    if namespace is None:
        namespace = executor._persistent_namespace
    inject_lightweight_tools_to_namespace(namespace)


def _parse_selection(text: str, num_candidates: int) -> int:
    """Parse the selected response index from the Selector's output.

    Looks for patterns like:
        <solution> Response 1: ... </solution>
        Response 2
        <select>Response 3</select>  (official X-Master format)

    Args:
        text: The Selector's full output text
        num_candidates: Number of candidate solutions

    Returns:
        0-indexed integer of the selected response, defaults to 0
    """
    # Try official X-Master format: <select>Response X</select>
    m = re.search(r'<select>\s*Response\s+(\d+)\s*</select>', text, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < num_candidates:
            return idx

    # Try: "Response X" inside <solution> tags
    sol_match = re.search(r'<solution>(.*?)</solution>', text, re.DOTALL)
    if sol_match:
        sol_text = sol_match.group(1)
        m = re.search(r'Response\s+(\d+)', sol_text, re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < num_candidates:
                return idx

    # Try anywhere in text: "Response X" or "Solution X"
    m = re.search(r'(?:Response|Solution)\s+(\d+)', text, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < num_candidates:
            return idx

    logger.warning("Could not parse selector's decision. Defaulting to Response 1.")
    return 0


def _build_selector_prompt(problem: str, all_solutions: list, retrieved_context: str = "") -> str:
    """Build the user message for the Selector agent.

    Presents the problem and all rewritten solutions in a structured format
    matching the official X-Master select_user.txt template.

    Args:
        problem: The original question/problem
        all_solutions: List of solution strings from Stage 3 (Rewriter)
        retrieved_context: Pre-retrieved knowledge base context

    Returns:
        Formatted prompt string
    """
    parts = [
        "You should thoroughly analyse each response carefully by writing codes "
        "and choose the most correct one from the following responses.\n",
        f"## Problem\n\n{problem}\n",
    ]

    for i, sol in enumerate(all_solutions, 1):
        parts.append(f"### Response {i}:\n\n{sol}\n")

    if retrieved_context:
        parts.append(
            "## Reference Knowledge\n\n"
            "The following relevant information was pre-retrieved from the knowledge base. "
            "Use it as reference but verify important claims independently.\n\n"
            f"{retrieved_context}\n"
        )

    parts.append(
        "## Instructions\n\n"
        "1. Do not trust the information, references, or any assumptions in the "
        "responses easily. You must write code to verify before reaching a conclusion.\n"
        "2. Do not be influenced by the majority number of final answers. "
        "They may collude to deceive you!\n"
        "3. You should collect enough information from web_search and database tools "
        "to verify each response.\n"
        "4. Finally, analyze whether each response is correct and select the best one.\n\n"
        "Your final answer MUST include which Response you selected, e.g.:\n"
        "<solution> Response 1: [the answer] </solution>"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core selector logic
# ---------------------------------------------------------------------------
def run_selector(
    problem: str,
    all_solutions: list,
    retrieved_context: str = "",
    temperature: float = 0.7,
    llm: str = None,
    source: str = None,
    base_url: str = None,
    api_key: str = None,
    timeout_seconds: int = 120,
) -> dict:
    """Run the Selector to pick the best solution from all rewritten solutions.

    Creates a fresh SelectorAgent, feeds it the problem + all rewritten
    solutions, and parses which response it selects.

    Args:
        problem: The original problem
        all_solutions: List of all rewritten solution strings from Stage 3
        retrieved_context: Pre-retrieved knowledge base context (from Stage 0)
        temperature: LLM sampling temperature
        llm: LLM model name (None -> env default)
        source: LLM provider (None -> auto-detect)
        base_url: Custom API base URL
        api_key: API key
        timeout_seconds: Code execution timeout per block

    Returns:
        dict with keys:
            - solution (str): The selected best solution
            - selected_index (int): 0-indexed index of the selected response
            - success (bool): Whether the Selector completed without error
            - log (list): Execution log
    """
    logger.info(f"[Selector] Starting selection from {len(all_solutions)} candidates...")

    # 1. Create fresh SelectorAgent instance (owns its own _namespace)
    selector = SelectorAgent(
        llm=llm,
        source=source,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        verbose=True,
        agent_label="Selector",
    )

    # 2. Inject lightweight tools (no knowledge_search) into namespace
    _inject_selector_tools(selector._namespace)

    # 3. Build the selector prompt with pre-retrieved context and run
    logger.info(f"[Selector] received retrieved_context: {len(retrieved_context)} chars")
    selector_prompt = _build_selector_prompt(problem, all_solutions, retrieved_context)

    # Fallback: use the first solution
    fallback = all_solutions[0] if all_solutions else ""

    try:
        log, last_content = selector.go(selector_prompt)
        selected_idx = _parse_selection(last_content, len(all_solutions))
        selected_solution = all_solutions[selected_idx]
        success = True
        logger.info(f"[Selector] Selected Response {selected_idx + 1}")
    except Exception as e:
        logger.error(f"[Selector] Failed: {e}")
        selected_solution = fallback
        selected_idx = 0
        log = []
        success = False

    # 4. Clean up namespace
    _reset_execution_env(selector._namespace, selector._captured_plots)

    return {
        "solution": selected_solution,
        "selected_index": selected_idx,
        "log": log,
        "success": success,
    }
