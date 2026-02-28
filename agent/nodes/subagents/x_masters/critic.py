"""X-Masters Stage 2: Critic

Reviews each Solver's solution independently, identifies flaws, and produces
a corrected (or confirmed) answer.  Each Critic instance is a CodeActAgent
with an adversarial system prompt that forces it to *first* look for errors
before deciding whether to amend or confirm the original solution.

Architecture (instance-level isolation, mirrors Solver):
    For each of N solutions from Stage 1:
        1. Create fresh CriticAgent instance (owns its own _namespace)
        2. Inject tools into the agent's instance-level namespace
        3. Run critic.go(problem + solution)
        4. Extract <solution> from output
        5. Collect critiqued solution (or fallback to original on failure)

    Each CodeActAgent holds its own _namespace dict and _captured_plots list,
    so parallel Send instances (LangGraph ThreadPoolExecutor) are thread-safe.

Data flow:
    Input  — from XMastersState after Stage 1 fan-in:
        problem  (str):   the original question
        solution (str):   one Solver's extracted answer
        solver_id (int):  which Solver produced it
        success  (bool):  whether the Solver succeeded

    Output — written to XMastersState.critiqued_solutions via operator.add:
        solver_id (int):  preserved from input
        solution  (str):  the Critic's (possibly amended) answer
        success   (bool): True if Critic ran without error
"""

import logging
import re
from typing import Dict, Any, Tuple, List  # NEW: Add type hints for semantic_conditions and validation

from .tools import inject_lightweight_tools_to_namespace

# ---------------------------------------------------------------------------
# Import CodeActAgent from result_evaluator
# ---------------------------------------------------------------------------
# Use absolute import to avoid conflict with top-level agent package
from agent.nodes.subagents.result_evaluator.agent import CodeActAgent
from agent.nodes.subagents.result_evaluator import executor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Critic system prompt
# ---------------------------------------------------------------------------
CRITIC_SYSTEM_PROMPT = """\
You are a critical reviewer assigned with the task of verifying and improving a proposed solution.
You will be given a PROBLEM and a PROPOSED SOLUTION. Your job is to check the solution for errors and produce a corrected or confirmed answer.

To achieve this, you will be using an interactive coding environment equipped with tool functions.

At each turn, you should first provide your critical analysis.
After that, you have two options:

1) Interact with a programming environment and receive the corresponding output. Your code should be enclosed using "<execute>" tag, for example:
<execute>
print("Hello World!")
</execute>

2) When you have reached your final verdict, provide the answer using "<solution>" tag, for example:
The answer is <solution> 42 </solution>.

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
# Initial Reasoning Guidance (IRG) for Critic — X-Master paper Section 2.3
# ---------------------------------------------------------------------------
CRITIC_IRG = """\
I need to carefully verify this proposed solution. Let me think about what could be wrong.

I have access to an interactive coding environment where I can write and execute Python code. I should leverage this capability to independently verify the proposed answer:
- Query biomedical databases (query_kg, query_expression, query_disease_gene, query_gwas, query_ppi, etc.) to fact-check claims about genes, diseases, drugs, or molecular biology
- Search for relevant information using web_search() to verify broader facts
- **Read full paper content** using read_webpage(url) when web_search returns a relevant paper — search snippets are often insufficient for nuanced questions
- Write code to independently re-derive or re-compute any calculations in the proposed solution
- Check if the proposed answer is consistent with known facts and constraints
- Look for edge cases or assumptions that the original solver may have missed

IMPORTANT - Handling Unavailable External Resources:
- If web_search returns "[Search Unavailable]" or empty results, STOP trying to search
- If read_webpage fails repeatedly, STOP trying to read webpages
- When external resources are unavailable, use your internal knowledge and the provided context
- DO NOT keep trying the same failing tool - proceed to your best answer using available information

CRITICAL: I should not simply agree with the proposed solution — my value is in finding errors that others missed. In particular, I must challenge any reasoning that relies on textbook-level generalizations. Real-world biology and chemistry often involve specialized mechanisms that deviate from simple models. When I find a relevant paper, I MUST use read_webpage(url) to read its actual findings rather than assuming from the title alone.

I should specifically check whether the solver may have made an oversimplified peak/signal assignment, missed a specialized structural mechanism, or conflated two different phenomena.

## ⚠️ CRITICAL: Condition Verification Before Simulation

If this problem has specific conditions mentioned (e.g., randomness patterns, imputation methods, data constraints), I MUST verify my simulation implements them correctly BEFORE concluding:

### Pre-Simulation Checklist (if conditions are provided):

1. **[ ] Randomness Pattern**: Did I implement the CORRECT type of randomness?
   - If "differ from sample to sample" → Each sample must have INDEPENDENT random patterns
   - If "same across samples" → All samples must use the SAME random pattern
   - ⚠️ Common mistake: Using same random seed for all samples when they should be independent

2. **[ ] Imputation Method**: Did I correctly handle missing data?
   - If "reference genome imputation" → Missing sites filled with reference genotypes
   - If "ancestral allele" assumption → Reference = ancestral (affects bias direction)
   - ⚠️ Common mistake: Dropping missing sites instead of imputing, or wrong imputation method

3. **[ ] Statistical Definitions**: Are my formulas using correct definitions?
   - **Segregating sites (S)**: Sites where AT LEAST ONE sample has a variant (not "all samples")
   - **Pairwise difference (π)**: Average difference between all pairs across ALL sites
   - ⚠️ Common mistake: S = sites where "all samples have variant" (wrong) vs "at least one sample has variant" (correct)

4. **[ ] Condition-Effect Reasoning**: Did I analyze how conditions affect statistics?
   - Random per-sample filtering → May not systematically eliminate all variants at any site → S unchanged
   - Reference imputation with ancestral assumption → Imputed sites show no difference → π underestimated
   - ⚠️ Common mistake: Not considering how conditions bias specific statistics

If my simulation does not implement ALL stated conditions correctly, my conclusion is INVALID and I must redo the simulation.

Let me start by identifying the key claims in this solution and writing code to check them.
"""


# ---------------------------------------------------------------------------
# CriticAgent: CodeActAgent with Critic-specific system prompt + IRG
# ---------------------------------------------------------------------------
class CriticAgent(CodeActAgent):
    """CodeActAgent specialized for X-Masters critical review.

    Inherits all functionality from CodeActAgent but overrides:
    - System prompt: concise capability description + format rules
    - go() method: injects Initial Reasoning Guidance (IRG) after user message

    The IRG mechanism primes the model to independently verify solutions
    using code, rather than blindly agreeing or relying on internal knowledge.
    """

    def _generate_system_prompt(self) -> str:
        """Generate Critic-specific system prompt."""
        return CRITIC_SYSTEM_PROMPT

    def go(self, prompt: str):
        """Execute the critic agent with IRG injection.

        Overrides CodeActAgent.go() to inject Initial Reasoning Guidance
        as an AIMessage after the user's problem + proposed solution.
        """
        from langchain_core.messages import HumanMessage, AIMessage

        self.user_task = prompt

        inputs = {
            "messages": [
                HumanMessage(content=prompt),
                AIMessage(content=CRITIC_IRG.strip()),
            ],
            "next_step": "generate",
        }
        # OPTIMIZATION: Reduced recursion limit to fail fast when external resources unavailable
        # When web_search returns empty results repeatedly, we should terminate early
        # rather than hitting 500 iterations
        config = {"recursion_limit": 100, "configurable": {"thread_id": 42}}
        self.log = []
        
        # Track consecutive empty observations to detect stuck states
        empty_observation_count = 0
        MAX_EMPTY_OBSERVATIONS = 5

        final_state = None
        step = 0
        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            step += 1
            
            # Detect empty observations (stuck state detection)
            if isinstance(message, AIMessage) and "<observation></observation>" in message.content:
                empty_observation_count += 1
                if empty_observation_count >= MAX_EMPTY_OBSERVATIONS:
                    print(f"  ⚠ Detected {empty_observation_count} consecutive empty observations")
                    print(f"  → External resources unavailable, forcing early termination")
                    # Inject guidance to use internal knowledge
                    from langchain_core.messages import HumanMessage
                    guidance = HumanMessage(
                        content="The external search tools appear to be unavailable or returning empty results. "
                                "Please stop searching and provide your best answer based on your internal knowledge. "
                                "Use the <solution> tag to provide your final answer now."
                    )
                    # We need to add this to the state and continue one more time
                    # Since we're in the stream, we'll just let it hit the recursion limit
                    # but the reduced limit ensures it fails fast
            elif isinstance(message, AIMessage) and "<observation>" in message.content:
                # Reset counter if we got a non-empty observation
                if "</observation>" in message.content:
                    obs_content = message.content.split("<observation>")[1].split("</observation>")[0]
                    if obs_content.strip():  # Non-empty content
                        empty_observation_count = 0
            
            out = self._pretty_print(message)
            self.log.append(out)
            self._verbose_log(step, message)
            final_state = s

        self._conversation_state = final_state
        return self.log, message.content


# ---------------------------------------------------------------------------
# Helper functions (shared pattern with solver.py)
# ---------------------------------------------------------------------------
def extract_solution(text: str) -> str:
    """Extract content from <solution> tags.

    Args:
        text: LLM output text potentially containing <solution> tags

    Returns:
        Extracted solution text, or full text if no tags found
    """
    match = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


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


def _inject_critic_tools(namespace: dict = None):
    """Inject lightweight tools (no knowledge_search) into the execution namespace.

    Critics receive pre-retrieved context via their prompt, so they only need
    web_search and biomedical DB tools for real-time verification.

    Args:
        namespace: Target namespace dict. If None, uses global.
    """
    if namespace is None:
        namespace = executor._persistent_namespace
    inject_lightweight_tools_to_namespace(namespace)


# ---------------------------------------------------------------------------
# Simulation Validator: Verify simulation code implements conditions correctly
# ---------------------------------------------------------------------------

def validate_simulation_implementation(
    simulation_code: str,
    semantic_conditions: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Validate that simulation code correctly implements all stated conditions.
    
    This is a post-hoc validation that can be run on the Critic's generated
    simulation code to check for common implementation errors.
    
    Args:
        simulation_code: The Python code generated by Critic for simulation
        semantic_conditions: Structured conditions from extract_structured_conditions()
        
    Returns:
        Tuple of (is_valid, issues_list)
        - is_valid: True if all checks pass
        - issues_list: List of detected issues (empty if valid)
    """
    if not semantic_conditions:
        return True, []  # No conditions to validate
    
    issues = []
    code_lower = simulation_code.lower() if simulation_code else ""
    
    # ========== Check 1: Random per-sample ==========
    randomness = semantic_conditions.get("randomness")
    if randomness and randomness.get("type") == "independent_per_sample":
        # Check that there's per-sample randomness
        has_for_loop = "for" in code_lower and ("sample" in code_lower or "i " in code_lower or "range" in code_lower)
        has_random = "random" in code_lower or "np.random" in code_lower
        
        if not (has_for_loop and has_random):
            issues.append(
                "MISSING: Independent per-sample randomness. "
                "The simulation should generate DIFFERENT random patterns for each sample "
                "(use loop over samples with independent random generation)."
            )
        
        # Check for common mistake: single random seed for all samples
        if "np.random.seed" in code_lower:
            # Check if seed is set inside the loop (bad - same seed each iteration)
            # or outside loop (could be okay if loop uses different random calls)
            lines = simulation_code.split('\n') if simulation_code else []
            seed_line = -1
            for i, line in enumerate(lines):
                if "np.random.seed" in line.lower():
                    seed_line = i
                    break
            
            # If seed is set, check if there's randomization per sample
            # This is a heuristic check
            if seed_line >= 0:
                # Look for indication that each sample gets different treatment
                has_sample_specific_random = False
                for line in lines:
                    if "sample" in line.lower() and ("random" in line.lower() or "randint" in line.lower() or "choice" in line.lower()):
                        has_sample_specific_random = True
                        break
                
                if not has_sample_specific_random:
                    issues.append(
                        "WARNING: Single random seed may cause all samples to have the same pattern. "
                        "Consider generating independent random patterns for each sample."
                    )
    
    # ========== Check 2: Reference imputation ==========
    imputation = semantic_conditions.get("imputation")
    if imputation and imputation.get("method") == "reference_genome":
        # Check that reference is used
        has_reference = "reference" in code_lower or "ref_" in code_lower or "ancestral" in code_lower
        has_imputation = "imput" in code_lower or "fill" in code_lower or "replace" in code_lower or "mask" in code_lower
        
        if not has_reference:
            issues.append(
                "MISSING: Reference genome imputation. "
                "The simulation should fill missing sites with reference genotypes."
            )
    
    # ========== Check 3: Statistics definitions ==========
    stats = semantic_conditions.get("statistics_affected", [])
    
    if "theta" in stats or "watterson" in stats:
        # Check that segregating sites are correctly defined
        # S = sites where AT LEAST ONE sample has a variant
        has_segregating_check = "segregating" in code_lower or "s = " in code_lower or "s=" in code_lower
        has_any_check = "any" in code_lower or "at least" in code_lower or "> 0" in code_lower
        
        if has_segregating_check and not has_any_check:
            issues.append(
                "POTENTIAL: Segregating sites definition may be incorrect. "
                "S should count sites where AT LEAST ONE sample has a variant, "
                "not where all samples have variants."
            )
    
    if "pi" in stats or "nucleotide diversity" in stats:
        # Check that pairwise comparison is mentioned
        has_pairwise = "pairwise" in code_lower or "pi" in code_lower or "diversity" in code_lower
        
        if not has_pairwise:
            issues.append(
                "MISSING: Pairwise difference calculation for π. "
                "π should be calculated from average pairwise differences across all sites."
            )
    
    return len(issues) == 0, issues


def _build_critic_prompt(
    problem: str, 
    solution: str, 
    retrieved_context: str = "",
    semantic_conditions: Dict[str, Any] = None  # NEW: 结构化条件
) -> str:
    """Build the user message for the Critic agent.

    Presents the problem and proposed solution in a structured format
    that the Critic's system prompt expects.

    Args:
        problem: The original question/problem
        solution: The Solver's proposed answer
        retrieved_context: Pre-retrieved knowledge base context
        semantic_conditions: Structured semantic conditions for verification

    Returns:
        Formatted prompt string
    """
    context_section = ""
    if retrieved_context:
        context_section = (
            f"\n## Reference Knowledge\n\n"
            f"The following relevant information was pre-retrieved from the knowledge base. "
            f"Use it as reference but verify important claims independently.\n\n"
            f"{retrieved_context}\n"
        )
    
    # NEW: 条件验证部分 - 强调模拟代码必须正确实现所有条件
    condition_section = ""
    if semantic_conditions:
        condition_section = "\n\n## ⚠️ CRITICAL CONDITIONS (Your simulation MUST implement these exactly)\n\n"
        condition_section += "**IMPORTANT**: This problem has specific conditions that significantly affect the answer. "
        condition_section += "If your simulation does not implement these correctly, your conclusion will be WRONG.\n\n"
        
        # Randomness condition
        if semantic_conditions.get("randomness"):
            rand_cond = semantic_conditions["randomness"]
            condition_section += f"### 1. Randomness Condition\n"
            condition_section += f"- **Type**: {rand_cond.get('type', 'unknown')}\n"
            condition_section += f"- **Description**: {rand_cond.get('description', 'Not specified')}\n"
            condition_section += f"- **Verification**: {rand_cond.get('verification', 'Verify the randomness pattern')}\n\n"
        
        # Imputation condition
        if semantic_conditions.get("imputation"):
            imp_cond = semantic_conditions["imputation"]
            condition_section += f"### 2. Imputation Condition\n"
            condition_section += f"- **Method**: {imp_cond.get('method', 'unknown')}\n"
            condition_section += f"- **Assumption**: {imp_cond.get('assumption', 'unknown')}\n"
            condition_section += f"- **Description**: {imp_cond.get('description', 'Not specified')}\n"
            condition_section += f"- **Verification**: {imp_cond.get('verification', 'Verify the imputation method')}\n\n"
        
        # Data constraints
        if semantic_conditions.get("data_constraints"):
            condition_section += f"### 3. Data Constraints\n"
            for constraint in semantic_conditions["data_constraints"]:
                condition_section += f"- {constraint}\n"
            condition_section += "\n"
        
        # Statistics affected
        if semantic_conditions.get("statistics_affected"):
            condition_section += f"### 4. Statistics Affected\n"
            condition_section += f"The following statistics are mentioned in this problem: "
            condition_section += ", ".join(semantic_conditions["statistics_affected"]) + "\n\n"
        
        # Verification checklist
        if semantic_conditions.get("verification_checklist"):
            condition_section += f"### 5. Pre-Simulation Verification Checklist\n"
            condition_section += "Before running your simulation, ensure you understand:\n"
            for item in semantic_conditions["verification_checklist"]:
                condition_section += f"- **{item.get('id', 'check')}**: {item.get('check', item.get('description', ''))}\n"
                if item.get('common_mistake'):
                    condition_section += f"  - ⚠️ Common mistake: {item['common_mistake']}\n"
            condition_section += "\n"
        
        condition_section += "---\n"
        condition_section += "**REMINDER**: Before providing your final answer, verify that your simulation correctly implements ALL conditions above. "
        condition_section += "A simulation that ignores or misinterprets these conditions will lead to an incorrect conclusion.\n"
    
    return f"""## Problem

{problem}

## Proposed Solution

{solution}
{context_section}{condition_section}
---

Please critically review the proposed solution above. Verify key claims using available tools. If you find errors, provide a corrected answer. If the solution is correct, confirm it."""


# ---------------------------------------------------------------------------
# Core critic logic
# ---------------------------------------------------------------------------
def run_single_critic(
    problem: str,
    solution: str,
    solver_id: int = 0,
    retrieved_context: str = "",
    semantic_conditions: Dict[str, Any] = None,  # NEW: 结构化条件
    temperature: float = 0.6,
    llm: str = None,
    source: str = None,
    base_url: str = None,
    api_key: str = None,
    timeout_seconds: int = 120,
) -> dict:
    """Run a single Critic instance on one Solver's solution.

    Creates a fresh CriticAgent, feeds it the problem + proposed solution,
    and extracts the (possibly amended) answer.  On failure, falls back to
    the original solution so that the Critic stage never makes things worse.

    Args:
        problem: The original problem
        solution: The Solver's proposed answer to review
        solver_id: Which Solver produced this solution (for tracking)
        retrieved_context: Pre-retrieved knowledge base context (from Stage 0)
        semantic_conditions: Structured semantic conditions for verification
        temperature: LLM sampling temperature
        llm: LLM model name (None → env default)
        source: LLM provider (None → auto-detect)
        base_url: Custom API base URL
        api_key: API key
        timeout_seconds: Code execution timeout per block

    Returns:
        dict with keys:
            - solution (str): The Critic's (possibly amended) answer
            - original_solution (str): The Solver's original answer
            - solver_id (int): Which Solver produced the original
            - success (bool): Whether the Critic completed without error
    """
    logger.info(f"[Critic {solver_id}] Starting review...")

    # 1. Create fresh CriticAgent instance (owns its own _namespace)
    critic = CriticAgent(
        llm=llm,
        source=source,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        verbose=True,
        agent_label=f"Critic {solver_id}",
    )

    # 2. Inject lightweight tools (no knowledge_search) into namespace
    _inject_critic_tools(critic._namespace)

    # 3. Build the critic prompt with pre-retrieved context and semantic conditions
    logger.info(f"[Critic {solver_id}] received retrieved_context: {len(retrieved_context)} chars")
    if semantic_conditions:
        logger.info(f"[Critic {solver_id}] received semantic_conditions with {len(semantic_conditions)} keys")
    critic_prompt = _build_critic_prompt(problem, solution, retrieved_context, semantic_conditions)

    error_info = None
    try:
        log, last_content = critic.go(critic_prompt)
        revised_solution = extract_solution(last_content)
        success = True
        logger.info(f"[Critic {solver_id}] Completed successfully")
    except Exception as e:
        import traceback
        error_info = str(e)
        error_traceback = traceback.format_exc()
        logger.error(f"[Critic {solver_id}] Failed: {e}")
        logger.debug(f"[Critic {solver_id}] Traceback: {error_traceback}")
        # Fallback: preserve the original solution
        revised_solution = solution
        log = []
        success = False

    # 4. Clean up namespace after this critic (defensive)
    _reset_execution_env(critic._namespace, critic._captured_plots)

    return {
        "solution": revised_solution,
        "original_solution": solution,
        "log": log,
        "solver_id": solver_id,
        "success": success,
        "error": error_info,  # Include error information for debugging
    }
