---
name: task_assignment
description: Rules for how the orchestrator assigns tasks to sub-agents. Covers domain bundling, parallel vs sequential dispatch, and cross-domain handoff.
---

# Orchestrator Task Assignment Strategy

## Core Principle: One Domain = One Sub-Agent Session

A sub-agent receives a **complete workflow**, not isolated atomic steps.
All tasks within the same domain are bundled into a single sub-agent session.
The sub-agent handles internal step ordering, data flow, and error recovery.

**Example — CORRECT:**
```
Bundle "immune" (1 session):
  Step 1: Check peptide support
  Step 2: Validate TCR input format
  Step 3: Convert CDR3 data to NetTCR format
  Step 4: Run binding prediction
  Step 5: Gather and summarize results
```

**Example — WRONG:**
```
Session 1: Check peptide support
Session 2: Validate TCR input format
Session 3: Convert CDR3 data
Session 4: Run prediction
Session 5: Gather results
```

## Domain Definitions

| Domain | Sub-Agent | MCP Tools | When to Use |
|--------|-----------|-----------|-------------|
| **immune** | immune | igblast, metabcr, nettcr, mixtcrpred, bcell, tcell, immune, flu | TCR/BCR repertoire, antibody binding, V(D)J, epitope screening, influenza |
| **rna** | rna | ribonn, gemorna, codontransformer, rinalmo | mRNA design, codon optimization, RNA structure |
| **structural** | structural | spired_fitness, foldx_saturation_scan | Protein stability, fitness landscapes, mutagenesis |
| **bioinformatics** | bioinformatics | bioinformatics, data, combine_filter | scRNA-seq, trajectory, clustering, data merging |

## When to Bundle (Same Sub-Agent)

Bundle tasks into ONE sub-agent session when they:
1. Use **MCP tools from the same domain** (e.g., all nettcr/metabcr tools)
2. Share a **data pipeline** where output of step N feeds into step N+1
3. Require the **same domain knowledge** to interpret results
4. Form a **logical workflow** (preprocess → validate → analyze → summarize)

## When to Split (Different Sub-Agents)

Assign tasks to a **different sub-agent** when:
1. The task requires **MCP tools from a different domain** (e.g., switching from nettcr to foldx)
2. The **knowledge domain changes** (e.g., from TCR binding prediction to scRNA-seq analysis)
3. The task operates on **fundamentally different data types** (e.g., CSV sequences vs RDS single-cell objects)

## Parallel vs Sequential Dispatch

### Dispatch Bundles in PARALLEL when:
- Two bundles have **no data dependencies** between them
- Example: immune analysis of Dataset A + RNA analysis of Dataset B → parallel
- Example: TCR binding prediction + protein stability scan on unrelated proteins → parallel

### Dispatch Bundles SEQUENTIALLY when:
- Bundle B needs **output from Bundle A** as input
- Example: immune bundle produces binding scores → bioinformatics bundle merges scores with scRNA-seq → sequential
- Example: structural bundle identifies stable mutants → immune bundle tests binding of mutants → sequential

### Within a Bundle — always SEQUENTIAL:
- Steps within a single bundle always execute in order
- The sub-agent handles internal data flow (step 1 output → step 2 input)
- This is enforced by the prompt structure, not the orchestrator

## Inter-Bundle Dependency Detection

The orchestrator automatically detects inter-bundle dependencies:
1. Each task has explicit `dependencies: [task_id, ...]`
2. If task_X (in bundle_A) depends on task_Y (in bundle_B), then bundle_A depends on bundle_B
3. The ReAct loop only dispatches a bundle when all its dependency bundles are COMPLETED

## Error Handling

- If a bundle fails, the orchestrator retries the entire bundle (up to max_attempts)
- If a single step fails within a bundle, the sub-agent logs the error and continues
- If a bundle permanently fails, downstream bundles that depend on it are skipped
