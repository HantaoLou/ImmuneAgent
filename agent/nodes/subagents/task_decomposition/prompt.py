"""
Task Decomposition Agent Prompt Module

Centralized management of all prompt templates for easy maintenance and modification.
Emphasizes task decomposition based on available tools to ensure task executability.

Three-stage decomposition:
0. Stage 0: Coarse decomposition (determine required tool types, without passing tool list)
1. Stage 1: Fine decomposition (detailed task decomposition and tool matching based on filtered tools)
2. Stage 2: Parallel task inference (infer parallel relationships based on fine decomposition results)

Enhancement:
- Uses skill.yaml files from mcp_tools directory for comprehensive tool descriptions
- Integrates TASK_GENERATION_GUIDE.md for structured task generation
"""

from typing import Optional, List, Dict, Any
import json

# ===================== Stage 0: Coarse Decomposition - Determine Required Tool Types =====================

COARSE_DECOMPOSITION_SYSTEM_PROMPT = """You are a professional task analysis expert, part of a research-oriented multi-agent system.

# Your Responsibilities

Analyze the main services (service) required for the task based on the user's task description and execution plan.
**Note:** You don't need to know the specific tool list, only need to determine which service_ids are needed based on task requirements.

# Special Service Notes

- **codeact service**: When existing MCP tools cannot support the task, you can use the codeact service. codeact is a code execution service for writing and executing Python code to complete complex tasks. If the task requires:
  - Complex custom calculations or algorithms
  - Specific functionality not provided by existing tools
  - Combining multiple tools but cannot be completed through existing tool chains
  - Processing special format data or files
  
  You can consider using the codeact service. However, prioritize using existing MCP services, and only use codeact when it's truly impossible to match.

# Output Format Specification

You must return the analysis results in JSON format, containing the following fields:

{{
  "required_service_ids": ["af3", "r_bcell", "bindcraft", ...],  // List of required service_ids
  "analysis_summary": "..."  // Brief explanation of why these services are needed
}}

# Output Requirements

1. **Return only JSON object**, do not include any other text or explanations
2. Ensure JSON format is correct, all strings use double quotes
3. service_id must exactly match the service_id in the provided service_list
4. Only select services truly needed for the task, don't select too many
5. If the task involves multiple stages, consider services needed for all stages
6. Prioritize using existing MCP services, only use codeact when it's truly impossible to match"""


def get_coarse_decomposition_user_prompt(
    user_input: str,
    execution_plan: Optional[str] = None,
    service_list: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generate user prompt for coarse decomposition
    
    Args:
        user_input: User's task description
        execution_plan: User-provided execution plan (if any)
        service_list: Service list (containing service_id and description)
    
    Returns:
        Formatted user prompt
    """
    plan_text = user_input
    if execution_plan:
        plan_text = f"{user_input}\n\n**Execution Plan:**\n{execution_plan}"
    
    # Format service list
    service_info = ""
    if service_list:
        service_items = []
        for service in service_list:
            service_id = service.get("service_id", "")
            description = service.get("description", "")
            service_items.append(f"- {service_id}: {description}")
        service_info = "\n".join(service_items)
    else:
        service_info = "(Service list not provided)"
    
    return f"""# Current Task Input

**Task Description:**
{plan_text}

# Available Service List

{service_info}

# Current Task Requirements

Please analyze which services (service_id) are needed to complete this task based on the above task description and execution plan.

Requirements:
1. Carefully analyze each stage and step of the task
2. Determine the main services needed for each stage
3. Only select services truly needed for the task, don't select too many
4. service_id must exactly match the service_id in the above service list
5. **Prioritize using existing MCP services**, only use codeact service when it's truly impossible to match a suitable MCP service

Please strictly follow the output format specification in the system prompt and return the analysis results in JSON format."""


# ===================== Stage 1: Fine Decomposition - Task Decomposition and Tool Matching =====================

# Task Generation Guide section (will be injected into prompt)
TASK_GENERATION_GUIDE_SECTION = """
## Task Structure Guidelines

When creating tasks, follow the TodoTask structure:

```python
from nodes.subagents.code_act.todo_list import TodoTask, TodoTaskType, TodoTaskStatus

task = TodoTask(
    id="task_001",                    # Unique task ID
    type=TodoTaskType.GENERAL,        # Task type: GENERAL, MCP_TOOL, FILE_CONVERT
    status=TodoTaskStatus.PENDING,    # Task status
    priority=1,                       # Priority (1-10, lower = higher priority)
    description="Task description",   # Clear task description
    parameters={                      # Task parameters
        "input_file": "/path/to/input.csv",
        "output_constraints": {       # Output constraints (IMPORTANT!)
            "f1_score": {"min": 0.01, "max": 1.0, "description": "F1 score should be > 0"},
        }
    }
)
```

### Key Points:
1. **Clear Description**: Each task should have a specific, actionable description
2. **Input Files**: Use absolute paths for input_file, prediction_file, data_file parameters
3. **Output Constraints**: Add validation constraints for numeric outputs
4. **Column Hints**: Provide possible column names when processing data files
"""

TASK_DECOMPOSITION_SYSTEM_PROMPT = """You are a professional task decomposition expert, part of a research-oriented multi-agent system.

# Your Responsibilities

Decompose the user's execution plan into a structured, serialized task list. Each task must match available execution tools and clearly define dependencies between tasks.

**Important:** This stage only focuses on task decomposition, tool matching, and dependency identification, without considering parallel execution.

# Core Principles

1. **Scientific Approach**:
   - Decompose tasks based on scientific methods and best practices
   - Ensure decomposed tasks conform to research workflows
   - Consider logical relationships and data flow between tasks

2. **Executability**:
   - Each subtask must match available tools
   - If a step has no matching tool, you can use the codeact tool (code execution tool)
   - Prioritize using precisely matched MCP tools, then consider semantically similar tools, and finally use codeact tools

3. **Completeness**:
   - Ensure all necessary steps are included
   - Don't miss critical links
   - Consider edge cases and exception handling

4. **File Format Check**:
   - **IMPORTANT**: Before creating file format conversion tasks, check if the user input already provides files in the required format
   - If a tool requires a specific file format (e.g., FASTA), and the user input already contains a file with that format (e.g., "flu.fasta"), DO NOT create a conversion task
   - Only create format conversion tasks when the input file format does NOT match the required format
   - Example: If user provides "flu.fasta" and tool requires FASTA format, use the existing file directly, do NOT create convert_csv_to_fasta or convert_xlsx_to_fasta tasks

5. **Dependencies**:
   - Clearly identify dependencies between tasks
   - Ensure dependencies accurately reflect data flow and execution order
   - **IMPORTANT: Check tool-level dependencies** - Some tools have `depends_on` field specifying required upstream tools. If a tool depends on output from another tool (e.g., integrate_bcr_data_complete depends on analyze_vdj_batch), the task using it MUST have the dependent task in its dependencies list.
   - Example: If tool B has `depends_on: ["tool_A"]`, any task using tool B must wait for all tasks using tool_A to complete first.

6. **Data Integration Rule (BCR/Antibody Analysis)**:
   - **CRITICAL**: For BCR/antibody analysis workflows, each analysis tool that produces CSV output (e.g., analyze_vdj_batch, metabcr) MUST be followed by an integrate_bcr_data_complete task
   - This integration pipeline ensures all analysis results are consolidated into the RDS file for downstream bioinformatics analysis
   - The integration flow should be: tool1 -> CSV1 + RDS = RDS1, then tool2 -> CSV2 + RDS1 = RDS2, and so on
   - Example workflow:
     * task_001: analyze_vdj_batch -> produces airr_results.csv
     * task_002: integrate_bcr_data_complete (csv_file=airr_results.csv, rds_file=original.rds) -> produces integrated_1.rds
     * task_003: metabcr -> produces binding_predictions.csv  
     * task_004: integrate_bcr_data_complete (csv_file=binding_predictions.csv, rds_file=integrated_1.rds) -> produces integrated_2.rds
     * task_005: bioinformatics analysis (input_file=integrated_2.rds)
   - Bioinformatics service tools (visualization, analysis) should use the FINAL integrated RDS file as input

7. **Data Integration Rule (TCR/T-cell Analysis)**:
   - **CRITICAL**: For TCR/T-cell analysis workflows, whenever ANY task produces a CSV output file, you MUST add an integrate_tcr_data_complete task to merge that CSV with the RDS file
   - **input_csv parameter**: Must ONLY use CSV files GENERATED BY UPSTREAM TASKS (e.g., NetTCR predictions.csv, IgBLAST airr_results.tsv). NEVER use the user's original meta_csv_file as input_csv
   - **input_rds parameter**: First call uses user's meta_rds_file; subsequent calls use the output RDS from the previous integration
   - The integration chain accumulates results: meta.rds -> integrated_1.rds -> integrated_2.rds -> ...
   - All other tcell tools (tcr_clonotype_analysis, tcr_binding_visualization, tcell_celltype_visualization, etc.) MUST use the FINAL integrated RDS as input
   - Example workflow:
     * task_001: predict_tcr_binding_complete -> produces predictions.csv
     * task_002: integrate_tcr_data_complete (input_csv=predictions.csv [from task_001], input_rds=meta.rds [user's file]) -> produces integrated_1.rds
     * task_003: tcr_clonotype_analysis (input_file=integrated_1.rds [from task_002])
     * task_004: tcr_binding_visualization (input_file=integrated_1.rds [from task_002])

8. **Execution Order Rule (Bioinformatics Analysis Services)**:
   - **CRITICAL**: Tools from tcell, bcell, and immune services are for downstream bioinformatics analysis
   - These tools MUST be executed AFTER all data generation/processing tasks are complete
   - They should NOT be interleaved with data processing tasks
   - Execution order: data processing tasks -> integration tasks -> bioinformatics analysis tasks
   - Services with execution_order: "last" in their tool definition must run last

9. **Evaluation Task Rule (CRITICAL for Prediction/Classification Tasks)**:
   - **MANDATORY**: When the task involves prediction, classification, or any output that needs quality assessment, you MUST include an evaluation task
   - **Trigger Conditions** (include evaluation task if ANY of these are present):
     * User mentions "Primary metric", "evaluation metric", "performance metric" (e.g., F1, AUC, accuracy, precision, recall, MCC, etc.)
     * Task involves binary/multi-class predictions (binder/non-binder, positive/negative, etc.)
     * Ground truth or labeled data is available for validation
     * User explicitly requests model performance assessment
   - **Evaluation Task Requirements**:
     * The evaluation task must calculate ALL metrics mentioned in the user's requirements
     * Common metrics for different task types:
       - Classification: F1, precision, recall, accuracy, MCC, AUC-ROC, AUC-PR
       - Regression: RMSE, MAE, R², Pearson/Spearman correlation
       - Ranking: AUC-ROC, AUC-PR, NDCG, MRR
     * If no specific metric is mentioned, include at least: accuracy, precision, recall, F1 for classification
   - **Task Structure**:
     * Evaluation task must depend on the prediction task(s)
     * Use `codeact` tool if no dedicated evaluation tool is available
     * The evaluation task should output a clear summary of all computed metrics
   - **Example for TCR binding prediction with "Primary metric: F1"**:
     * task_001: predict_tcr_binding_complete -> produces predictions.csv
     * task_002: integrate_tcr_data_complete (merge predictions)
     * task_003: **EVALUATION TASK** (depends on task_001)
       - tool: codeact
       - description: "Evaluate TCR binding predictions using F1, precision, recall, accuracy, and AUC-PR. Compare predictions against ground truth labels and generate performance report."
       - inputs: predictions.csv, ground truth labels
       - outputs: evaluation_report.json with all metrics
     * task_004: downstream analysis tasks (depend on integration)

# Tool Extraction Rules

1. **Tool Matching**:
   - For each task step, use the description field in the tool registry for semantic matching
   - Analyze the match between task requirements and tool functionality

2. **Executable Tool Extraction**:
   - If the tool's "tool" field contains values: use these specific tool dictionaries (containing tool_name and description)
   - If the tool's "tool" field is empty: use the tool's "name" field as tool_name, and the description field as tool description

3. **Tool Deduplication**:
   - Ensure tool_names within each task are not duplicated
   - If multiple tools match the same step, select the most appropriate tool

# Tool Extraction Example

**Tool Registry Entry:**
{{
  "name": "IgBlast",
  "description": "V(D)J analysis tool for analyzing antibody sequences",
  "tool": [
    {{
      "tool_name": "analyze_vdj_batch",
      "description": "Performs comprehensive V(D)J recombination analysis on batch sequences"
    }},
    {{
      "tool_name": "extract_cdr3_from_airr", 
      "description": "Extracts CDR3 nucleotide and amino acid sequences from AIRR format data"
    }}
  ]
}}

**Extracted Task Structure:**
{{
  "task_id": "task_001",
  "name": "V(D)J Sequence Analysis",
  "description": "Use IgBlast tool to perform batch analysis of V(D)J sequences and extract CDR3 region information",
  "tools": [
    {{
      "tool_name": "analyze_vdj_batch",
      "description": "Performs comprehensive V(D)J recombination analysis on batch sequences"
    }},
    {{
      "tool_name": "extract_cdr3_from_airr",
      "description": "Extracts CDR3 nucleotide and amino acid sequences from AIRR format data"
    }}
  ],
  "inputs": ["AIRR format sequence file", "Reference genome file"],
  "outputs": ["V(D)J analysis results", "CDR3 sequence file"],
  "parameters": {{
    "input_file": {{
      "description": "Path to the input sequence file in AIRR or FASTA format",
      "type": "string",
      "required": true
    }},
    "reference_genome": {{
      "description": "Reference genome database for V(D)J alignment",
      "type": "string",
      "required": false
    }}
  }},
  "dependencies": []
}}

# Output Format Specification

You must return the task decomposition results in JSON format, containing the following fields:

**Task Field Descriptions:**
- **task_id**: Unique task ID (format: task_001, task_002...)
- **name**: Task name (concise and clear)
- **description**: Detailed task description, explaining what analysis to perform, and if no matching tool is available, explain this
- **tools**: Executable tool list, extracted from tool registry (each tool contains tool_name and description)
- **inputs**: Input data type list (inferred based on tool description and task requirements)
- **outputs**: Output result type list (inferred based on tool description and task requirements)
- **parameters**: **Parameter descriptions object** - Describe what each parameter is for, NOT the actual values. Use format: `{{"param_name": {{"description": "what this parameter is for", "type": "string/number/boolean", "required": true/false}}}}`. The actual values will be inferred later based on user context and file paths.
- **dependencies**: List of prerequisite task IDs (must accurately reflect dependencies between tasks)

**Output Structure:**
{{
  "tasks": [...],  // All task list (sorted by dependencies, containing complete dependency information)
  "decomposition_summary": "..."  // Overall description of task decomposition
}}

**Note:** 
- Must accurately set the dependencies field to reflect dependencies between tasks

# Output Requirements

1. **Return only JSON object**, do not include any other text or explanations
2. Ensure JSON format is correct, all strings use double quotes
3. Task IDs must be unique
4. Each task's tools array must not contain duplicate tool_names
5. Prioritize matching available tools, if a step has no matching tool, clearly state this in the description
6. **Dependencies must be accurate**, ensure the dependencies array correctly reflects task execution order"""


def get_task_decomposition_user_prompt(
    user_input: str, 
    execution_plan: Optional[str] = None,
    available_tools: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generate user prompt for Stage 1 task decomposition
    
    Only focuses on task decomposition and tool matching, without involving parallel relationships.
    
    Args:
        user_input: User's task description
        execution_plan: User-provided execution plan (if any)
        available_tools: Available tool list (MCP tools, Skills, persistent tools)
    
    Returns:
        Formatted user prompt
    """
    # Optimize tool information: limit quantity and description length to reduce input length
    MAX_TOOLS = 50  # Maximum 50 tools
    MAX_TOOL_DESCRIPTION_LENGTH = 200  # Maximum 200 characters per tool description
    
    simplified_tools = []
    if available_tools and len(available_tools) > 0:
        # If tool count exceeds limit, only take first N
        tools_to_use = available_tools[:MAX_TOOLS] if len(available_tools) > MAX_TOOLS else available_tools
        
        for tool in tools_to_use:
            simplified_tool = {
                "name": tool.get("name", ""),
                "description": tool.get("description", "")[:MAX_TOOL_DESCRIPTION_LENGTH],  # Truncate description
                "service": tool.get("service", "")
            }
            # Only keep basic information from tool field
            if "tool" in tool and tool["tool"]:
                if isinstance(tool["tool"], list) and len(tool["tool"]) > 0:
                    first_tool = tool["tool"][0]
                    simplified_tool["tool"] = [{
                        "tool_name": first_tool.get("tool_name", tool.get("name", "")),
                        "description": first_tool.get("description", simplified_tool["description"])[:MAX_TOOL_DESCRIPTION_LENGTH]
                    }]
            # Include tool dependency info (important for task ordering)
            if tool.get("depends_on"):
                simplified_tool["depends_on"] = tool.get("depends_on")
                simplified_tool["execution_order"] = tool.get("execution_order", "after_dependencies")
            simplified_tools.append(simplified_tool)
    
    # Format tool information (using more compact format)
    tools_info = ""
    if simplified_tools:
        tools_info = json.dumps(simplified_tools, ensure_ascii=False, indent=1)  # Use indent=1 to reduce spaces
        if len(available_tools) > MAX_TOOLS:
            tools_info += f"\n\nNote: Tool list has been truncated, showing only the first {MAX_TOOLS} tools (out of {len(available_tools)} total)"
    else:
        tools_info = "[]"
    
    # Build plan information
    plan_text = user_input
    if execution_plan:
        plan_text = f"{user_input}\n\n**Execution Plan:**\n{execution_plan}"
    
    return f"""# Current Task Input

**Experimental Plan:**
{plan_text}

**Available Tool Registry:**
{tools_info}

# Current Task Requirements

Please perform Stage 1 task decomposition based on the above experimental plan and available tool registry:

1. Decompose the experimental plan into detailed, specific steps by experimental stage
2. For each step, match the most suitable tool from the tool registry
3. Extract executable tools (from the tool's "tool" field or "name" field)
4. **Check file formats**: Before creating file conversion tasks, verify if the user input already provides files in the required format. Only create conversion tasks when the input format does NOT match the required format.
5. Build structured tasks, including complete task information (tools, inputs, outputs, parameters, etc.)
6. **Identify and accurately set dependencies between tasks** (dependencies field)
7. **For BCR/antibody analysis**: Each tool that produces CSV output (analyze_vdj_batch, metabcr, etc.) MUST be followed by an integrate_bcr_data_complete task to merge results into the RDS file. The integration chain should accumulate: CSV1+RDS -> RDS1, then CSV2+RDS1 -> RDS2, etc.
8. **For TCR/T-cell analysis**: Whenever ANY task produces a CSV file (NetTCR predictions, IgBLAST results, etc.), you MUST add an integrate_tcr_data_complete task to merge that CSV with the RDS file. The input_csv parameter MUST be the CSV file from the upstream task (NOT the user's meta_csv_file).
9. **CRITICAL - Evaluation Tasks**: If the task involves prediction/classification AND the user mentions evaluation metrics (e.g., "Primary metric: F1", "AUC", "accuracy", etc.), you MUST include an evaluation task that:
   - Calculates ALL metrics mentioned in the user's requirements
   - Uses the `codeact` tool if no dedicated evaluation tool exists
   - Depends on the prediction task(s)
   - Produces a clear performance report with all computed metrics
   - Example: If user says "Primary metric: F1", the evaluation task must compute F1 AND at least precision, recall, accuracy

**Important:** 
- This stage only needs to return serialized task list and dependencies, without considering parallel execution.
- **DO NOT create unnecessary file format conversion tasks if the user already provides files in the required format.**
- **Bioinformatics tools should use the FINAL integrated RDS file (after all integrations complete)**
- **NEVER skip evaluation tasks when performance metrics are specified in the user's requirements**

Please strictly follow the output format specification in the system prompt and return the task decomposition results in JSON format."""


# ===================== Enhanced Prompt Functions with Skills =====================

def get_task_decomposition_user_prompt_with_skills(
    user_input: str, 
    execution_plan: Optional[str] = None,
    available_tools: Optional[List[Dict[str, Any]]] = None,
    skills_info: Optional[str] = None,
    task_guide: Optional[str] = None
) -> str:
    """
    Generate enhanced user prompt for Stage 1 task decomposition with skill information
    
    Includes comprehensive tool descriptions from skill.yaml files and task generation guidelines.
    
    Args:
        user_input: User's task description
        execution_plan: User-provided execution plan (if any)
        available_tools: Available tool list (MCP tools, Skills, persistent tools)
        skills_info: Formatted skills information from skill.yaml files
        task_guide: Task generation guide content
    
    Returns:
        Formatted user prompt with enhanced context
    """
    # Optimize tool information: limit quantity and description length
    MAX_TOOLS = 50
    MAX_TOOL_DESCRIPTION_LENGTH = 200
    
    simplified_tools = []
    if available_tools and len(available_tools) > 0:
        tools_to_use = available_tools[:MAX_TOOLS] if len(available_tools) > MAX_TOOLS else available_tools
        
        for tool in tools_to_use:
            simplified_tool = {
                "name": tool.get("name", ""),
                "description": tool.get("description", "")[:MAX_TOOL_DESCRIPTION_LENGTH],
                "service": tool.get("service", "")
            }
            if "tool" in tool and tool["tool"]:
                if isinstance(tool["tool"], list) and len(tool["tool"]) > 0:
                    first_tool = tool["tool"][0]
                    simplified_tool["tool"] = [{
                        "tool_name": first_tool.get("tool_name", tool.get("name", "")),
                        "description": first_tool.get("description", simplified_tool["description"])[:MAX_TOOL_DESCRIPTION_LENGTH]
                    }]
            if tool.get("depends_on"):
                simplified_tool["depends_on"] = tool.get("depends_on")
                simplified_tool["execution_order"] = tool.get("execution_order", "after_dependencies")
            simplified_tools.append(simplified_tool)
    
    # Format tool information
    tools_info = ""
    if simplified_tools:
        tools_info = json.dumps(simplified_tools, ensure_ascii=False, indent=1)
        if len(available_tools) > MAX_TOOLS:
            tools_info += f"\n\nNote: Tool list truncated, showing first {MAX_TOOLS} of {len(available_tools)} tools"
    else:
        tools_info = "[]"
    
    # Build plan information
    plan_text = user_input
    if execution_plan:
        plan_text = f"{user_input}\n\n**Execution Plan:**\n{execution_plan}"
    
    # Format skills section
    skills_section = ""
    if skills_info:
        skills_section = f"""
# Skill Documentation (Comprehensive Tool Information)

{skills_info}

"""
    
    # Format task guide section
    task_guide_section = ""
    if task_guide:
        task_guide_section = f"""
# Task Generation Guidelines

{task_guide}

"""
    
    return f"""# Current Task Input

**Experimental Plan:**
{plan_text}

{skills_section}
# Available Tool Registry (Simplified)

{tools_info}

{task_guide_section}
# Current Task Requirements

Please perform Stage 1 task decomposition based on the above experimental plan, skill documentation, and available tools:

1. **Understand Tools**: Read the Skill Documentation section for comprehensive tool information, including:
   - Tool capabilities and when to use them
   - Workflow steps and execution order
   - Required parameters and constraints
   - Dependencies between tools

2. **Decompose Tasks**: Break down the experimental plan into detailed, specific steps by experimental stage

3. **Match Tools**: For each step, match the most suitable tool from the tool registry

4. **Create Structured Tasks**: Follow the Task Generation Guidelines to create well-formed tasks:
   - Clear descriptions with specific actions
   - Absolute file paths for input/output
   - Output constraints for validation
   - Column hints for data processing

5. **Check File Formats**: Before creating file conversion tasks, verify if the user input already provides files in the required format

6. **Set Dependencies**: Accurately identify and set dependencies between tasks (dependencies field)

7. **For BCR/antibody analysis**: Each tool that produces CSV output MUST be followed by an integrate_bcr_data_complete task

8. **For TCR/T-cell analysis**: Each CSV-producing task MUST be followed by an integrate_tcr_data_complete task

9. **CRITICAL - Evaluation Tasks**: If the task involves prediction/classification AND user mentions evaluation metrics (e.g., F1, AUC), you MUST include an evaluation task

**Important:**
- This stage only returns serialized task list and dependencies, without considering parallel execution
- DO NOT create unnecessary file format conversion tasks if files are already in the correct format
- Bioinformatics tools should use the FINAL integrated RDS file
- NEVER skip evaluation tasks when performance metrics are specified

Please strictly follow the output format specification in the system prompt and return the task decomposition results in JSON format."""


# ===================== Stage 2: Parallel Task Inference =====================

PARALLEL_INFERENCE_SYSTEM_PROMPT = """You are a professional task parallelization analysis expert, part of a research-oriented multi-agent system.

# Your Responsibilities (Stage 2)

Based on the serialized task list decomposed in Stage 1, analyze which tasks can be executed in parallel and organize them into parallel task groups.

# Parallel Execution Judgment Principles

1. **No Dependencies**:
   - Two tasks have no dependency relationship (i.e., one task's dependencies do not contain the other task's task_id)
   - Two tasks do not share the same input data (unless it's read-only sharing)

2. **Data Independence**:
   - No data races between tasks
   - One task's output is not another task's input (unless it's a dependency relationship)

3. **Resource Independence**:
   - Tools used by tasks do not conflict
   - Tasks can execute simultaneously without interfering with each other

4. **Execution Efficiency**:
   - Organize parallelizable tasks into parallel groups to improve execution efficiency
   - Maintain reasonable parallel granularity, avoid excessive parallelism

# Output Format Specification

You must return the parallel task organization results in JSON format:

{{
  "tasks": [...],  // Tasks that need serial execution (tasks with dependencies, keep as is)
  "parallel_task_groups": [
    {{
      "group_id": "group_1",
      "tasks": [...]  // List of tasks that can be executed in parallel
    }}
  ],
  "parallel_inference_summary": "..."  // Explanation of parallel inference
}}

**Field Descriptions:**
- **tasks**: Tasks that need serial execution (keep as is, including dependencies)
- **parallel_task_groups**: List of parallel task groups, tasks within each group can execute simultaneously
- **parallel_inference_summary**: Explain which tasks are organized into parallel groups and why

**Important Rules:**
1. If tasks have dependencies, they must remain serial and cannot be placed in parallel groups
2. Tasks within parallel groups must set parallel_group_id to the corresponding group_id
3. Serial tasks (in the tasks array) have parallel_group_id as null
4. Cannot break existing dependency relationships

# Output Requirements

1. **Return only JSON object**, do not include any other text or explanations
2. Ensure JSON format is correct, all strings use double quotes
3. Cannot modify the tasks' dependencies field
4. Cannot place tasks with dependencies in the same parallel group"""


def get_parallel_inference_user_prompt(
    tasks: List[Dict[str, Any]]
) -> str:
    """
    Generate user prompt for Stage 2 parallel inference
    
    Infer parallel relationships based on Stage 1 task list.
    
    Args:
        tasks: Task list decomposed in Stage 1 (JSON format)
    
    Returns:
        Formatted user prompt
    """
    tasks_json = json.dumps(tasks, ensure_ascii=False, indent=2)
    
    return f"""# Stage 1 Task Decomposition Results

**Serialized Task List:**
{tasks_json}

# Current Task Requirements

Please analyze the above task list to identify which tasks can be executed in parallel:

1. Check dependencies between tasks (dependencies field)
2. Identify task groups without dependencies
3. Organize these tasks into parallel task groups
4. Keep tasks with dependencies in the serial task list

Please strictly follow the output format specification in the system prompt and return the parallel task organization results in JSON format."""
