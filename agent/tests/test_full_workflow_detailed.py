"""
完整工作流详细测试用例

测试从 preprocess -> supervisor -> immunity -> task_decomposition -> executor 的全流程，并详细记录所有中间过程。

工作流步骤：
1. Preprocess: 输入预处理（提取参数、分析文件、上传到沙盒）
2. Supervisor: 任务分类
3. Immunity: 执行计划生成
4. Task Decomposition: 任务分解
5. Executor: 任务执行（参数推断、并发执行、结果汇总）

记录内容包括：
1. 用户原始输入
2. 预处理结果（会话ID、文件分析、参数提取）
3. 执行计划（如果有）
4. Supervisor 分类结果
5. Task Decomposition 任务分解结果
6. Executor 参数推断结果（包括从用户输入、执行计划、依赖任务输出中提取的参数）
7. 每个任务的输入输出及任务信息
8. 任务执行过程及顺序（包括并发执行和优先级策略）
9. 每个任务的执行汇总（execution_summary）
10. 汇总的结果

运行方式：
- 批量测试所有问题: pytest tests/test_full_workflow_detailed.py -v -s
- 测试单个问题: pytest tests/test_full_workflow_detailed.py::test_flu_benchmark -v -s -k "Q01"
- 测试特定类型: pytest tests/test_full_workflow_detailed.py -v -s -k "simple"
"""

import os
import pytest
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, TypedDict


# ==================== 测试问题定义 ====================

class BenchmarkCase(TypedDict):
    """测试用例定义"""
    id: str
    name: str
    difficulty: str  # simple, complex
    prompt: str
    services: List[str]
    parameters: Dict[str, str]
    expected_output: str
    primary_metric: str


# Flu Benchmark 测试问题
FLU_BENCHMARK_CASES: List[BenchmarkCase] = [
    # Q01: Which antibodies bind H1N1 Michigan?
    {
        "id": "Q01",
        "name": "H1N1 Michigan 结合抗体预测",
        "difficulty": "simple",
        "prompt": """Given the single-cell RNA-seq data and antibody sequences for 100 flu-specific monoclonal antibodies, predict which antibodies bind to H1N1 A/Michigan/45/2015 hemagglutinin. For each antibody (identified by `mAb`), output a binary prediction: 1 = binds, 0 = does not bind.

Expected output format:
- CSV with columns: `mAb`, `prediction` (0 or 1), optionally `probability` (0.0-1.0)
- One row per antibody (up to 100 rows)

Ground truth summary:
- 98 antibodies tested (2 are NaN)
- 82 positive (83.7%), 16 negative

Primary metric: F1""",
        "services": ["igblast", "metabcr", "r_data_integration", "bioinformatics"],
        "parameters": {
            "meta_csv_file": "/data/benchmark_data/flu_benchmark/260129_flu_metadata.csv",
            "antigen_file": "/data/benchmark_data/flu_benchmark/flu_antig_seq.csv",
            "meta_rds_file": "/data/benchmark_data/flu_benchmark/260129_flu_benchmark.rds"
        },
        "expected_output": "CSV with mAb, prediction columns",
        "primary_metric": "F1"
    },
    
    # Q05: Which antibodies neutralize H1N1 California?
    {
        "id": "Q05",
        "name": "H1N1 California 中和抗体预测",
        "difficulty": "simple",
        "prompt": """Given the single-cell RNA-seq data and antibody sequences, predict which antibodies neutralize H1N1 A/California/07/2009. For each antibody, output a binary prediction: 1 = neutralizes, 0 = does not neutralize.

Note: Neutralization is a stronger functional readout than binding.

Expected output format:
- CSV with columns: `mAb`, `prediction` (0 or 1)

Ground truth summary:
- 56 antibodies tested (44 are NaN -- not tested for neutralization)
- 17 positive (30.4%), 39 negative

Primary metric: F1""",
        "services": ["igblast", "metabcr", "r_data_integration", "bioinformatics"],
        "parameters": {
            "meta_csv_file": "/data/benchmark_data/flu_benchmark/260129_flu_metadata.csv",
            "antigen_file": "/data/benchmark_data/flu_benchmark/flu_antig_seq.csv",
            "meta_rds_file": "/data/benchmark_data/flu_benchmark/260129_flu_benchmark.rds"
        },
        "expected_output": "CSV with mAb, prediction columns",
        "primary_metric": "F1"
    },
    
    # Q09: Which B cells broadly neutralize H1N1 variants?
    {
        "id": "Q09",
        "name": "H1N1 广谱中和抗体预测",
        "difficulty": "complex",
        "prompt": """Identify which antibodies are broadly neutralizing against H1N1 influenza variants. An antibody is broadly neutralizing if it neutralizes at least 60% of the H1N1 strains tested (California, Fort Monmouth, Jiangsu, Michigan, Puerto Rico). For antibodies with missing data for some strains, compute breadth over the non-missing strains only. Output a binary prediction per antibody.

Expected output format:
- CSV with columns: `mAb`, `prediction` (0 or 1)

Ground truth summary:
- 100 antibodies tested
- 33 broadly neutralizing (33.0%), 67 not

Primary metric: F1""",
        "services": ["igblast", "metabcr", "r_data_integration", "bioinformatics"],
        "parameters": {
            "meta_csv_file": "/data/benchmark_data/flu_benchmark/260129_flu_metadata.csv",
            "antigen_file": "/data/benchmark_data/flu_benchmark/flu_antig_seq.csv",
            "meta_rds_file": "/data/benchmark_data/flu_benchmark/260129_flu_benchmark.rds"
        },
        "expected_output": "CSV with mAb, prediction columns",
        "primary_metric": "F1"
    },
    
    # Q11: Which antibodies cross-react with both H1N1 and H3N2?
    {
        "id": "Q11",
        "name": "H1N1/H3N2 交叉反应抗体预测",
        "difficulty": "complex",
        "prompt": """Identify which antibodies are cross-reactive, meaning they bind at least one H1N1 strain AND at least one H3N2 strain. The H1N1 binding columns are Michigan, Puerto Rico, Victoria, and Wisconsin. The H3N2 binding columns are Hong Kong and Singapore. Output a binary prediction per antibody.

Expected output format:
- CSV with columns: `mAb`, `prediction` (0 or 1)

Ground truth summary:
- 100 antibodies tested
- 68 cross-reactive (68.0%), 32 not

Primary metric: F1""",
        "services": ["igblast", "metabcr", "r_data_integration", "bioinformatics"],
        "parameters": {
            "meta_csv_file": "/data/benchmark_data/flu_benchmark/260129_flu_metadata.csv",
            "antigen_file": "/data/benchmark_data/flu_benchmark/flu_antig_seq.csv",
            "meta_rds_file": "/data/benchmark_data/flu_benchmark/260129_flu_benchmark.rds"
        },
        "expected_output": "CSV with mAb, prediction columns",
        "primary_metric": "F1"
    },
    
    # Q19: Which BCR isotypes are associated with flu neutralization?
    {
        "id": "Q19",
        "name": "BCR 同种型与流感中和关联分析",
        "difficulty": "complex",
        "prompt": """Analyze the relationship between BCR heavy chain isotype (IGH_isotype) and H1N1 neutralization capacity. Determine which isotypes (IGHA1, IGHA2, IGHD, IGHG1, IGHG2, IGHG3, IGHM) are statistically enriched among antibodies that neutralize at least one H1N1 strain (California, Fort Monmouth, Jiangsu, Michigan, Puerto Rico). Report the enriched isotype names.

Expected output format:
- List of isotype names enriched among neutralizers (e.g., "IGHG1", "IGHA1")

Primary metric: Entity recall (fraction of true enriched isotypes identified)""",
        "services": ["igblast", "metabcr", "r_data_integration", "bioinformatics"],
        "parameters": {
            "meta_csv_file": "/data/benchmark_data/flu_benchmark/260129_flu_metadata.csv",
            "antigen_file": "/data/benchmark_data/flu_benchmark/flu_antig_seq.csv",
            "meta_rds_file": "/data/benchmark_data/flu_benchmark/260129_flu_benchmark.rds"
        },
        "expected_output": "List of enriched isotype names",
        "primary_metric": "Entity recall"
    },
    
    # Q20: Which flu antibodies are polyreactive?
    {
        "id": "Q20",
        "name": "多反应性抗体预测",
        "difficulty": "simple",
        "prompt": """Predict which antibodies are polyreactive. An antibody is polyreactive if it binds to LPS (lipopolysaccharide) OR dsDNA (double-stranded DNA) as non-specific self-antigens. Output a binary prediction per antibody.

Note: Polyreactivity is a sequence-intrinsic property related to hydrophobicity and charge.

Expected output format:
- CSV with columns: `mAb`, `prediction` (0 or 1)

Ground truth summary:
- 35 antibodies tested (65 are NaN)
- 14 polyreactive (40.0%), 21 not

Primary metric: F1""",
        "services": ["igblast", "metabcr", "r_data_integration", "bioinformatics"],
        "parameters": {
            "meta_csv_file": "/data/benchmark_data/flu_benchmark/260129_flu_metadata.csv",
            "antigen_file": "/data/benchmark_data/flu_benchmark/flu_antig_seq.csv",
            "meta_rds_file": "/data/benchmark_data/flu_benchmark/260129_flu_benchmark.rds"
        },
        "expected_output": "CSV with mAb, prediction columns",
        "primary_metric": "F1"
    },
    
    # Q24: What B cell subtypes are enriched among flu binders?
    {
        "id": "Q24",
        "name": "流感结合抗体的B细胞亚型富集分析",
        "difficulty": "complex",
        "prompt": """Analyze which B cell subtypes (celltype annotation) are statistically enriched among antibodies that bind any flu hemagglutinin strain (H1N1 Michigan, Puerto Rico, Victoria, Wisconsin; H3N2 Hong Kong, Singapore). Use the single-cell RNA-seq data to characterize B cell phenotypes and determine enrichment. Report the enriched cell type names.

Expected output format:
- List of enriched B cell subtype names (e.g., "B.07.Bgc_DZ-like")

Primary metric: Entity recall""",
        "services": ["igblast", "metabcr", "r_data_integration", "bioinformatics"],
        "parameters": {
            "meta_csv_file": "/data/benchmark_data/flu_benchmark/260129_flu_metadata.csv",
            "antigen_file": "/data/benchmark_data/flu_benchmark/flu_antig_seq.csv",
            "meta_rds_file": "/data/benchmark_data/flu_benchmark/260129_flu_benchmark.rds"
        },
        "expected_output": "List of enriched B cell subtype names",
        "primary_metric": "Entity recall"
    },
    
    # Q25: Which B cell subtypes produce multi-strain flu neutralizers?
    {
        "id": "Q25",
        "name": "多株中和抗体相关的B细胞亚型分析",
        "difficulty": "complex",
        "prompt": """Identify which B cell subtypes are enriched among broadly neutralizing antibodies (breadth >= 60% across H1N1 neutralization strains: California, Fort Monmouth, Jiangsu, Michigan, Puerto Rico). Determine which cell types disproportionately produce multi-strain neutralizers. Report the enriched cell type names.

Expected output format:
- List of enriched B cell subtype names

Primary metric: Entity recall""",
        "services": ["igblast", "metabcr", "r_data_integration", "bioinformatics"],
        "parameters": {
            "meta_csv_file": "/data/benchmark_data/flu_benchmark/260129_flu_metadata.csv",
            "antigen_file": "/data/benchmark_data/flu_benchmark/flu_antig_seq.csv",
            "meta_rds_file": "/data/benchmark_data/flu_benchmark/260129_flu_benchmark.rds"
        },
        "expected_output": "List of enriched B cell subtype names",
        "primary_metric": "Entity recall"
    },
    
    # Q26: What structural features distinguish broadly neutralizing antibodies?
    {
        "id": "Q26",
        "name": "广谱中和抗体结构特征分析",
        "difficulty": "complex",
        "prompt": """Compare the antibody sequences of broadly neutralizing vs. non-neutralizing antibodies across both the flu and SARS-CoV-2 datasets. Identify structural features that distinguish broadly neutralizing antibodies (bnAbs). Consider CDR-H3 length, V-gene usage, somatic hypermutation (SHM) rate, light chain pairing, and other sequence-derived features. Provide a text summary of the distinguishing features.

Expected output format:
- Text description listing structural features (e.g., "CDR-H3 length", "IGHV1-69 usage", "high SHM rate")

Primary metric: Entity recall (fraction of known distinguishing features mentioned)""",
        "services": ["igblast", "metabcr", "r_data_integration", "bioinformatics"],
        "parameters": {
            "meta_csv_file": "/data/benchmark_data/flu_benchmark/260129_flu_metadata.csv",
            "antigen_file": "/data/benchmark_data/flu_benchmark/flu_antig_seq.csv",
            "meta_rds_file": "/data/benchmark_data/flu_benchmark/260129_flu_benchmark.rds"
        },
        "expected_output": "Text description of distinguishing features",
        "primary_metric": "Entity recall"
    },
]


# TCR ICON Benchmark 测试问题
TCR_ICON_BENCHMARK_CASES: List[BenchmarkCase] = [
    # Q13: Which T cells bind MART-1 cancer epitope?
    {
        "id": "Q13",
        "name": "MART-1癌症表位TCR结合预测",
        "difficulty": "simple",
        "prompt": """Given 2080 T cell receptors with paired CDR3 alpha (CDR3a) and CDR3 beta (CDR3b) sequences, predict which TCRs bind the MART-1 cancer epitope (peptide: ELAGIGILTV, presented by HLA-A*02:01). MART-1 is a melanoma-associated antigen. For each TCR (identified by `main_name`), output a binary prediction: True = binder, False = non-binder.

What to use:
- CDR3a and CDR3b amino acid sequences for each TCR
- Target peptide: ELAGIGILTV
- HLA restriction: A*02:01
- TCR V/J gene usage annotations
- TCR-epitope binding prediction tools (e.g., NetTCR-2.0)

Expected output format:
- CSV with columns: `main_name`, `prediction` (True or False), optionally `probability` (0.0-1.0)
- 2080 rows (one per TCR)

Ground truth summary:
- 2080 TCRs tested
- 60 positive (2.9%), 2020 negative
- Highly imbalanced -- only ~3% are MART-1 binders

Primary metric: F1""",
        "services": ["igblast", "nettcr", "tcell"],
        "parameters": {
            "h5ad_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.h5ad",
            "meta_csv_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv",
            "meta_rds_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.rds"
        },
        "expected_output": "CSV with main_name, prediction columns",
        "primary_metric": "F1"
    },
    
    # Q14: Which T cells bind influenza Flu-MP epitope?
    {
        "id": "Q14",
        "name": "流感Flu-MP表位TCR结合预测",
        "difficulty": "simple",
        "prompt": """Predict which TCRs bind the influenza matrix protein epitope Flu-MP (peptide: GILGFVFTL, presented by HLA-A*02:01). GILGFVFTL is the immunodominant influenza matrix protein epitope and one of the best-characterized TCR targets. Output a binary prediction per TCR.

What to use:
- CDR3a and CDR3b sequences
- Target peptide: GILGFVFTL
- HLA restriction: A*02:01
- TCR-epitope binding prediction tools (e.g., NetTCR-2.0)

Expected output format:
- CSV with columns: `main_name`, `prediction` (True or False)
- 2080 rows (one per TCR)

Ground truth summary:
- 2080 TCRs tested
- 60 positive (2.9%), 2020 negative

Primary metric: F1""",
        "services": ["igblast", "nettcr", "tcell"],
        "parameters": {
            "h5ad_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.h5ad",
            "meta_csv_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv",
            "meta_rds_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.rds"
        },
        "expected_output": "CSV with main_name, prediction columns",
        "primary_metric": "F1"
    },
    
    # Q15: Which T cells bind EBV epitopes?
    {
        "id": "Q15",
        "name": "EBV表位TCR结合预测",
        "difficulty": "medium",
        "prompt": """Predict which TCRs bind any Epstein-Barr virus (EBV) epitope. There are 5 EBV epitopes in this dataset presented by different HLA alleles:

1. LLDFVRFMGV (EBNA-3B, HLA-A*02:01)
2. RLRAEAQVK (EMNA-3A, HLA-A*03:01)
3. IVTDFSVIK (EBNA-3B, HLA-A*11:01)
4. AVFDRKSDAK (EBNA-3B, HLA-A*11:01)
5. RAKFKQLL (BZLF1, HLA-B*08:01)

A TCR is EBV-reactive if it binds ANY of these 5 epitopes (OR logic). Output a binary prediction per TCR.

What to use:
- CDR3a and CDR3b sequences
- All 5 EBV peptide sequences with their HLA restrictions
- OR logic: positive if any of the 5 predictions is True
- TCR-epitope binding prediction tools (e.g., NetTCR-2.0)

Expected output format:
- CSV with columns: `main_name`, `prediction` (True or False)
- 2080 rows (one per TCR)

Ground truth summary:
- 2080 TCRs tested
- 300 positive (14.4%), 1780 negative
- Covers 3 EBV antigens (EBNA-3A, EBNA-3B, BZLF1) across 3 HLA alleles

Primary metric: F1""",
        "services": ["igblast", "nettcr", "tcell"],
        "parameters": {
            "h5ad_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.h5ad",
            "meta_csv_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv",
            "meta_rds_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.rds"
        },
        "expected_output": "CSV with main_name, prediction columns",
        "primary_metric": "F1"
    },
]


def build_user_input(test_case: BenchmarkCase) -> str:
    """根据测试用例构建用户输入"""
    services_str = "\n".join([f"    - {s}" for s in test_case["services"]])
    params_str = "\n".join([f"    - {k}: {v}" for k, v in test_case["parameters"].items()])
    
    return f"""{test_case['prompt']}

Only use the following MCP services:
{services_str}

Parameters:
{params_str}

Note:
- All tools provided by bioinformatics service must be used.
"""

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
import sys
project_root = Path(__file__).parent.parent.parent  # bio-agent 目录
agent_dir = Path(__file__).parent.parent  # bio-agent/agent 目录
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from state import GlobalState, UserTaskType, SubTask
from main_graph import build_main_graph
from nodes.subagents.supervisor.graph import (
    build_supervisor_subgraph,
    supervisor_input_mapper,
    supervisor_output_mapper,
    preprocess_user_input_node,
    SupervisorState
)
from nodes.subagents.immunity.graph import (
    build_immunity_subgraph,
    immunity_input_mapper,
    immunity_output_mapper
)
from nodes.subagents.immunity.state import ImmunityState
from nodes.subagents.task_decomposition.graph import (
    build_task_decomposition_subgraph,
    task_decomposition_input_mapper,
    task_decomposition_output_mapper
)
from nodes.subagents.executor.graph import (
    build_executor_subgraph,
    executor_input_mapper,
    executor_output_mapper,
    execute_executor_with_interrupt_support,
    resume_executor_after_interrupt,
    ExecutorTaskStatus
)
from nodes.subagents.result_evaluator.graph import (
    build_result_evaluator_subgraph,
    result_evaluator_input_mapper,
    result_evaluator_output_mapper
)
from nodes.subagents.result_evaluator.state import ResultEvaluatorState
from utils.hitl_interaction import handle_hitl_interrupt

# 导入测试日志记录器
try:
    from test_logger import init_global_logger, get_global_logger, save_global_logger, TestCaseLogger
except ImportError:
    import sys
    test_dir = Path(__file__).parent
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))
    from test_logger import init_global_logger, get_global_logger, save_global_logger, TestCaseLogger


def _serialize_interrupt_data(data: Any) -> Dict[str, Any]:
    """序列化中断数据"""
    if data is None:
        return {}
    
    if hasattr(data, 'value') and hasattr(data, 'id'):
        return {
            "type": "Interrupt",
            "id": str(getattr(data, 'id', '')),
            "value": _serialize_interrupt_data(getattr(data, 'value', {}))
        }
    
    if isinstance(data, dict):
        return {k: _serialize_interrupt_data(v) for k, v in data.items()}
    
    if isinstance(data, list):
        return [_serialize_interrupt_data(item) for item in data]
    
    try:
        json.dumps(data)
        return data
    except (TypeError, ValueError):
        return str(data)


def extract_interrupt_value(obj: Any, max_depth: int = 5) -> Any:
    """递归提取中断值"""
    if max_depth <= 0:
        return obj
    
    if hasattr(obj, 'value'):
        return extract_interrupt_value(obj.value, max_depth - 1)
    
    if isinstance(obj, dict):
        if 'value' in obj:
            return extract_interrupt_value(obj['value'], max_depth - 1)
        if 'type' in obj:
            return obj
        return obj
    
    if isinstance(obj, tuple):
        if len(obj) >= 2:
            return extract_interrupt_value(obj[1], max_depth - 1)
        elif len(obj) == 1:
            return extract_interrupt_value(obj[0], max_depth - 1)
        return obj
    
    return obj


def _get_output_field(output: Any, field: str, default: Any = None) -> Any:
    """兼容 dict/对象输出的字段读取"""
    if hasattr(output, field):
        return getattr(output, field)
    if isinstance(output, dict):
        return output.get(field, default)
    return default


def _filter_progress_messages(output: Any) -> Any:
    """Filter progress messages from executor output for logging."""
    if output is None:
        return None
    parsed_output = output
    if isinstance(output, str):
        try:
            parsed_output = json.loads(output)
        except (TypeError, ValueError):
            return output
    if isinstance(parsed_output, dict) and isinstance(parsed_output.get("messages"), list):
        filtered_messages = []
        for msg in parsed_output["messages"]:
            if isinstance(msg, dict) and msg.get("type") == "progress":
                continue
            filtered_messages.append(msg)
        filtered_output = dict(parsed_output)
        filtered_output["messages"] = filtered_messages
        return filtered_output
    return parsed_output


def _serialize_subtask(task: SubTask) -> Dict[str, Any]:
    """Serialize SubTask for logging."""
    if hasattr(task, "model_dump"):
        return task.model_dump(mode="json")
    if hasattr(task, "__dict__"):
        return {k: v for k, v in task.__dict__.items() if not k.startswith("_")}
    return {"value": str(task)}


def _serialize_parallel_groups(groups: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize parallel task groups for logging."""
    serialized = {}
    for group_id, group in groups.items():
        if hasattr(group, "model_dump"):
            group_dict = group.model_dump(mode="json")
        elif hasattr(group, "__dict__"):
            group_dict = {k: v for k, v in group.__dict__.items() if not k.startswith("_")}
        else:
            group_dict = {"group_id": group_id, "value": str(group)}
        if "subtasks" in group_dict and isinstance(group_dict["subtasks"], list):
            group_dict["subtasks"] = [_serialize_subtask(t) for t in group_dict["subtasks"]]
        serialized[group_id] = group_dict
    return serialized


def _extract_error_type_from_output(output: Any) -> Optional[str]:
    """从任务输出中提取错误类型"""
    if not output:
        return None
    parsed_output = output
    if isinstance(output, str):
        try:
            parsed_output = json.loads(output)
        except (TypeError, ValueError):
            return None
    if not isinstance(parsed_output, dict):
        return None
    final_result = parsed_output.get("final_result")
    if isinstance(final_result, dict):
        error_type = final_result.get("error_type")
        if error_type:
            return error_type
    return parsed_output.get("error_type")


@pytest.fixture(scope="module", autouse=True)
def setup_global_logger():
    """初始化全局日志记录器"""
    test_file_name = Path(__file__).stem
    init_global_logger(test_file_name)
    yield
    save_global_logger()


def run_single_test_case(
    test_case: BenchmarkCase,
    test_dir: Path,
    logger: Any,
    use_react_supervisor: bool = False,
    use_react_executor: bool = False,
    react_max_steps: int = 3
) -> Dict[str, Any]:
    """
    执行单个测试用例的完整工作流
    
    Args:
        test_case: 测试用例定义
        test_dir: 测试目录
        logger: 日志记录器
        use_react_supervisor: 是否使用 ReAct Supervisor
        use_react_executor: 是否使用 ReAct Executor
        react_max_steps: ReAct 最大步数
        
    Returns:
        测试结果字典
    """
    # 构建用户输入
    user_input = build_user_input(test_case)
    execution_plan = None
    
    # 启用 OpenSandbox
    os.environ["OPENSANDBOX_ENABLED"] = "true"
    os.environ["CODEACT_SANDBOX_PROVIDER"] = "opensandbox"
    os.environ["OPENSANDBOX_SKIP_MCP_INSTALL"] = "true"
    
    if not os.environ.get("OPENSANDBOX_VOLUME_BINDINGS"):
        os.environ["OPENSANDBOX_VOLUME_BINDINGS"] = "/data/sessions:/tmp/sessions,/data:/data:ro"
    
    print(f"\n{'='*80}")
    print(f"【测试用例 {test_case['id']}: {test_case['name']}】")
    print(f"难度: {test_case['difficulty']}")
    print(f"{'='*80}")
    
    # 记录用户原始输入
    if logger:
        logger.log_initial_state({
            "test_case_id": test_case["id"],
            "test_case_name": test_case["name"],
            "difficulty": test_case["difficulty"],
            "user_input": user_input,
            "expected_output": test_case["expected_output"],
            "primary_metric": test_case["primary_metric"]
        }, "测试用例输入")
    
    # 创建初始全局状态
    global_state = GlobalState(
        user_input=user_input,
        execution_plan=execution_plan,
        sandbox_dir=str(test_dir),
        use_react_supervisor=use_react_supervisor,
        use_react_executor=use_react_executor,
        react_max_steps=react_max_steps
    )
    
    # ==================== 步骤 1: 输入预处理 ====================
    # 创建预处理状态
    preprocess_state = SupervisorState(
        user_input=user_input,
        uploaded_files=[],
        sandbox_dir=str(test_dir),
    )
    
    # 执行预处理
    import io
    import contextlib
    print(f"\n{'='*80}")
    print(f"【步骤 1/5】输入预处理中...")
    print(f"{'='*80}")
    print(f"  → 正在提取参数和文件信息...")
    start_time = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        preprocess_result = preprocess_user_input_node(preprocess_state)
    elapsed = time.time() - start_time
    print(f"  ✓ 预处理完成: session_id={preprocess_result.session_id} (耗时: {elapsed:.2f}秒)")
    
    # 只打印关键信息：参数表
    print(f"\n{'='*80}")
    print(f"【1. 参数表】")
    print(f"{'='*80}")
    print(json.dumps(preprocess_result.extracted_parameters, ensure_ascii=False, indent=2))
    
    # 将预处理结果同步到全局状态
    global_state.session_id = preprocess_result.session_id
    global_state.sandbox_data_dir = preprocess_result.sandbox_data_dir
    global_state.opensandbox_id = preprocess_result.opensandbox_id
    global_state.extracted_parameters = preprocess_result.extracted_parameters
    global_state.file_analyses = [
        {
            "original_path": fa.original_path,
            "sandbox_path": fa.sandbox_path,
            "file_type": fa.file_type,
            "detected_data_type": fa.detected_data_type,
            "row_count": fa.row_count,
            "column_names": fa.column_names,
            "content_summary": fa.content_summary
        }
        for fa in preprocess_result.file_analyses
    ] if preprocess_result.file_analyses else []
    
    global_state.merged_result["preprocess"] = {
        "session_id": preprocess_result.session_id,
        "sandbox_data_dir": preprocess_result.sandbox_data_dir,
        "opensandbox_id": preprocess_result.opensandbox_id,
        "extracted_parameters": preprocess_result.extracted_parameters,
        "file_analyses": global_state.file_analyses
    }
    
    # 记录预处理结果
    if logger:
        logger.log_node_execution(
            "preprocess_user_input",
            {"user_input": user_input},
            global_state.merged_result["preprocess"],
            "输入预处理"
        )
    
    # ==================== 步骤 2: Supervisor 分类 ====================
    print(f"\n{'='*80}")
    print(f"【步骤 2/5】Supervisor 任务分类中...")
    print(f"{'='*80}")
    print(f"  → 正在分析任务类型...")
    start_time = time.time()
    supervisor_subgraph = build_supervisor_subgraph()
    supervisor_input = supervisor_input_mapper(global_state)
    
    with contextlib.redirect_stdout(io.StringIO()):
        supervisor_output = supervisor_subgraph.invoke(supervisor_input)
    global_state = supervisor_output_mapper(supervisor_output, global_state)
    elapsed = time.time() - start_time
    task_type = global_state.user_task_type or "UNKNOWN"
    print(f"  ✓ 任务分类完成: {task_type} (耗时: {elapsed:.2f}秒)")
    
    # ==================== 步骤 3: Immunity 生成执行计划（已跳过以加快测试）====================
    # NOTE: Immunity 子图暂时跳过，直接进入任务分解
    # 如需恢复，取消下面的注释
    immunity_subgraph = build_immunity_subgraph()
    immunity_input = immunity_input_mapper(global_state)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            immunity_output = immunity_subgraph.invoke(immunity_input)
        if isinstance(immunity_output, dict):
            immunity_output = ImmunityState(**immunity_output)
        global_state = immunity_output_mapper(immunity_output, global_state)
        
        immunity_plan = global_state.merged_result.get("immunity_plan", {})
        generated_execution_plan = (
            immunity_plan.get("final_enhanced_plan")
            or immunity_plan.get("experimental_plan")
            or immunity_plan.get("plan_summary")
        )
        if not generated_execution_plan and immunity_plan.get("executable_plan"):
            generated_execution_plan = json.dumps(
                immunity_plan["executable_plan"],
                ensure_ascii=False,
                indent=2
            )
        if generated_execution_plan:
            global_state.execution_plan = generated_execution_plan
        
        if logger:
            logger.log_node_execution(
                "immunity_subgraph",
                {"user_input": user_input},
                {
                    "execution_plan": global_state.execution_plan,
                    "immunity_plan": immunity_plan
                },
                "执行计划"
            )
    except Exception as e:
        print(f"⚠ Immunity 执行计划生成失败: {e}")
        if logger:
            logger.log_node_execution(
                "immunity_subgraph",
                {"user_input": user_input},
                {"error": str(e)},
                "执行计划失败"
            )
    # print("  [Skip] Immunity 子图已跳过以加快测试速度")
    
    # ==================== 步骤 4: Task Decomposition ====================
    print(f"\n{'='*80}")
    print(f"【步骤 3/5】任务分解中...")
    print(f"{'='*80}")
    print(f"  → 任务分解包含三个阶段:")
    print(f"     1. 粗分解 (Coarse Decomposition) - 确定需要的服务")
    print(f"     2. 细分解 (Fine Decomposition) - 详细任务分解和工具匹配")
    print(f"     3. 并行推断 (Parallel Inference) - 推断可并行执行的任务")
    print(f"  → 这可能需要一些时间，请稍候...")
    start_time = time.time()
    
    # 构建子图
    print(f"\n  [阶段 0] 构建任务分解子图...")
    subgraph_start = time.time()
    task_decomposition_subgraph = build_task_decomposition_subgraph()
    print(f"  ✓ 子图构建完成 (耗时: {time.time() - subgraph_start:.2f}秒)")
    
    # 准备输入
    print(f"  [阶段 0] 准备输入数据...")
    input_start = time.time()
    task_decomp_input = task_decomposition_input_mapper(global_state)
    print(f"  ✓ 输入准备完成 (耗时: {time.time() - input_start:.2f}秒)")
    
    # 执行任务分解（不静默输出，让各阶段的进度信息显示）
    print(f"\n  [阶段 1] 开始粗分解...")
    print(f"  {'-'*76}")
    stage1_start = time.time()
    
    # 使用自定义输出捕获，保留重要的进度信息
    class ProgressFilter:
        def __init__(self, original_stdout):
            self.original_stdout = original_stdout
            self.buffer = []
        
        def write(self, text):
            # 保留所有输出，但立即显示
            self.original_stdout.write(text)
            self.original_stdout.flush()
            self.buffer.append(text)
        
        def flush(self):
            self.original_stdout.flush()
    
    # 不使用redirect_stdout，直接执行以显示所有进度信息
    task_decomp_output = task_decomposition_subgraph.invoke(task_decomp_input)
    stage1_elapsed = time.time() - stage1_start
    print(f"  {'-'*76}")
    print(f"  ✓ 粗分解完成 (耗时: {stage1_elapsed:.2f}秒)")
    
    global_state = task_decomposition_output_mapper(task_decomp_output, global_state)
    total_elapsed = time.time() - start_time
    
    num_tasks = len(global_state.subtasks) + sum(len(g.subtasks) for g in global_state.parallel_task_groups.values())
    print(f"\n  {'='*76}")
    print(f"  ✓ 任务分解全部完成: 共生成 {num_tasks} 个任务")
    print(f"  ✓ 总耗时: {total_elapsed:.2f}秒")
    print(f"  {'='*76}")
    
    if logger:
        logger.log_node_execution(
            "coarse_decomposition",
            {"user_input": user_input, "execution_plan": global_state.execution_plan},
            {"required_service_ids": _get_output_field(task_decomp_output, "required_service_ids", [])},
            "粗分解结果"
        )
        logger.log_node_execution(
            "parallel_inference",
            {"raw_tasks": _get_output_field(task_decomp_output, "raw_tasks", [])},
            {"subtasks": [_serialize_subtask(t) for t in global_state.subtasks]},
            "并行推断结果"
        )
    
    # 只打印关键信息：任务列表
    print(f"\n{'='*80}")
    print(f"【2. 任务列表】")
    print(f"{'='*80}")
    all_tasks_for_display = global_state.subtasks + [
        task for group in global_state.parallel_task_groups.values()
        for task in group.subtasks
    ]
    for i, task in enumerate(all_tasks_for_display, 1):
        # SubTask has: task_id, task_type, content, dependencies, parallel_group_id, result
        task_result = task.result if isinstance(task.result, dict) else {}
        tools = task_result.get("tools", [])
        tool_name = tools[0] if tools else "unknown"
        if isinstance(tool_name, dict):
            tool_name = tool_name.get("tool_name") or tool_name.get("name", "unknown")
        print(f"  {i}. [{task.task_type}/{tool_name}] {task.content[:80]}...")
    print()
    
    # 检查是否有任务需要执行
    # Note: Executor merges parallel task groups into subtasks, so we should use
    # the same logic to avoid counting tasks twice
    all_tasks = list(global_state.subtasks)
    seen_task_ids = {task.task_id for task in all_tasks}
    
    # Add tasks from parallel groups that are not already in subtasks
    for group in global_state.parallel_task_groups.values():
        if hasattr(group, 'subtasks'):
            for task in group.subtasks:
                if task.task_id not in seen_task_ids:
                    all_tasks.append(task)
                    seen_task_ids.add(task.task_id)
        elif isinstance(group, dict):
            group_subtasks = group.get('subtasks', [])
            for task in group_subtasks:
                task_id = task.task_id if hasattr(task, 'task_id') else task.get('task_id')
                if task_id and task_id not in seen_task_ids:
                    all_tasks.append(task)
                    seen_task_ids.add(task_id)
    
    if not all_tasks:
        print("⚠ 没有任务需要执行")
        return
    
    executor_subgraph = build_executor_subgraph()
    executor_input = executor_input_mapper(global_state)
    
    # 执行工作流
    thread_id = "full_workflow_detailed_test"
    
    print(f"\n{'='*80}")
    print(f"【步骤 4/5】任务执行中...")
    print(f"{'='*80}")
    print(f"  → 开始执行 {num_tasks} 个任务...")
    print(f"  → 任务将按依赖关系顺序执行...")
    print(f"  → 执行过程中会显示任务进度...")
    
    # 首次执行
    executor_start_time = time.time()
    try:
        result = execute_executor_with_interrupt_support(
            executor_subgraph,
            executor_input,
            thread_id=thread_id
        )
        elapsed = time.time() - executor_start_time
        print(f"  → Executor 首次执行完成 (耗时: {elapsed:.2f}秒)")
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"\n{'='*80}")
        print(f"【Executor 执行错误】")
        print(f"{'='*80}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        print(f"\n完整错误堆栈:")
        print(f"{error_traceback}")
        print(f"{'='*80}\n")
        
        if logger:
            logger.log_node_execution("executor_subgraph", executor_input, None,
                                     f"Executor 执行失败: {type(e).__name__}: {e}")
            # 记录完整错误信息
            logger.log_summary({
                "error": True,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_traceback": error_traceback
            })
        
        # 重新抛出异常，让测试失败
        raise
    
    # 处理中断循环
    iteration_count = 0
    max_iterations = 50
    
    while result.get("interrupted", False) and iteration_count < max_iterations:
        iteration_count += 1
        print(f"  → 处理中断 #{iteration_count}...")
        interrupt_data = result.get("interrupt_data")
        
        if not interrupt_data:
            if executor_input.parent_state and executor_input.parent_state.hitl_status:
                try:
                    interrupt_data = json.loads(executor_input.parent_state.hitl_status)
                except:
                    pass
        
        if interrupt_data:
            actual_interrupt_data = extract_interrupt_value(interrupt_data)
            
            if not isinstance(actual_interrupt_data, dict):
                actual_interrupt_data = {"value": actual_interrupt_data}
            
            print(f"\n{'='*60}")
            print(f"【HITL 中断 #{iteration_count}】")
            print(f"{'='*60}")
            print(f"中断数据: {actual_interrupt_data}")
            
            try:
                user_response = handle_hitl_interrupt(
                    actual_interrupt_data,
                    callback=None,
                    use_file=False
                )
                
                try:
                    global_state.hitl_status = json.dumps(user_response, ensure_ascii=False)
                except Exception as json_e:
                    global_state.hitl_status = json.dumps(_serialize_interrupt_data(user_response), ensure_ascii=False)
                
                print(f"\n恢复执行...\n")
                
                result = resume_executor_after_interrupt(
                    executor_subgraph,
                    thread_id=thread_id,
                    resume_value=user_response
                )
                
            except KeyboardInterrupt:
                print("\n用户退出，终止执行")
                break
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                print(f"\n⚠ HITL交互处理失败: {e}")
                print(f"错误堆栈:\n{error_traceback}")
                break
        else:
            print(f"\n⚠ 检测到中断，但无法获取中断数据")
            break
    
    # 获取最终结果
    final_state = result.get("result")
    if final_state is None:
        if executor_input.parent_state:
            try:
                executor_input_dict = executor_input.model_dump(exclude={'parent_state'}, mode='json')
                final_result = executor_subgraph.invoke(
                    executor_input_dict,
                    config={"configurable": {"thread_id": thread_id}}
                )
                final_state = final_result
            except:
                final_state = executor_input
    
    if final_state is None:
        final_state = executor_input
    
    # 映射回全局状态
    global_state = executor_output_mapper(final_state, global_state)

    # ==================== 步骤 5: Result Evaluator 总结 ====================
    print(f"\n{'='*80}")
    print(f"【步骤 5/6】结果评估与总结中...")
    print(f"{'='*80}")
    print(f"  → 正在收集所有任务执行结果...")
    print(f"  → 结合执行计划生成总结报告...")
    start_time = time.time()

    try:
        result_evaluator_subgraph = build_result_evaluator_subgraph()
        result_evaluator_input = result_evaluator_input_mapper(global_state)

        with contextlib.redirect_stdout(io.StringIO()):
            result_evaluator_output = result_evaluator_subgraph.invoke(result_evaluator_input)

        # 处理输出（可能是 dict 或 ResultEvaluatorState）
        if isinstance(result_evaluator_output, dict):
            result_evaluator_output = ResultEvaluatorState(**result_evaluator_output)

        global_state = result_evaluator_output_mapper(result_evaluator_output, global_state)

        elapsed = time.time() - start_time
        print(f"  ✓ 结果评估完成 (耗时: {elapsed:.2f}秒)")

        # 打印总结报告
        print(f"\n{'='*80}")
        print(f"【5. 总结报告】")
        print(f"{'='*80}")

        summary_report = result_evaluator_output.summary_report
        if summary_report:
            print(f"\n{summary_report}")

        key_findings = result_evaluator_output.key_findings
        if key_findings:
            print(f"\n关键发现:")
            for i, finding in enumerate(key_findings, 1):
                print(f"  {i}. {finding}")

        recommendations = result_evaluator_output.recommendations
        if recommendations:
            print(f"\n建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        output_files = result_evaluator_output.output_files
        if output_files:
            print(f"\n输出文件:")
            for f in output_files[:10]:
                print(f"  - {f}")
            if len(output_files) > 10:
                print(f"  ... 还有 {len(output_files) - 10} 个文件")

        if result_evaluator_output.report_path:
            print(f"\n完整报告路径: {result_evaluator_output.report_path}")

        # 记录到日志
        if logger:
            logger.log_node_execution(
                "result_evaluator_subgraph",
                {"user_input": user_input, "execution_plan": global_state.execution_plan},
                {
                    "summary_report": summary_report,
                    "key_findings": key_findings,
                    "recommendations": recommendations,
                    "output_files": output_files,
                    "report_path": result_evaluator_output.report_path,
                    "statistics": {
                        "total_tasks": result_evaluator_output.total_tasks,
                        "completed_tasks": result_evaluator_output.completed_tasks,
                        "failed_tasks": result_evaluator_output.failed_tasks,
                        "success_rate": result_evaluator_output.success_rate
                    }
                },
                "结果评估与总结"
            )

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"⚠ Result Evaluator 执行失败: {e}")
        print(f"错误堆栈:\n{error_traceback}")
        if logger:
            logger.log_node_execution(
                "result_evaluator_subgraph",
                {"user_input": user_input},
                {"error": str(e), "error_traceback": error_traceback},
                "结果评估失败"
            )

    print(f"\n{'='*80}")
    print(f"【步骤 6/6】结果汇总中...")
    print(f"{'='*80}")
    
    # 只打印关键信息：每个任务的结果
    print(f"\n{'='*80}")
    print(f"【3. 任务执行结果】")
    print(f"{'='*80}")
    if hasattr(final_state, 'task_results') and final_state.task_results:
        for task_id, task_result in final_state.task_results.items():
            status_icon = "✓" if task_result.status and str(task_result.status).upper() == "COMPLETED" else "✗"
            status_str = task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status)
            print(f"\n  {status_icon} Task {task_id}: {status_str}")
            if task_result.error:
                print(f"     错误: {task_result.error[:200]}..." if len(str(task_result.error)) > 200 else f"     错误: {task_result.error}")
            if task_result.output:
                output_preview = str(task_result.output)[:500]
                if len(str(task_result.output)) > 500:
                    output_preview += "..."
                print(f"     输出: {output_preview}")
    
    if logger:
        for task_id, task_result in final_state.task_results.items():
            iteration_history = []
            if task_result.result_summary and isinstance(task_result.result_summary, dict):
                iteration_history = task_result.result_summary.get("code_iterations", []) or []
            if not iteration_history:
                iteration_history = [{
                    "status": task_result.status.value if hasattr(task_result.status, "value") else str(task_result.status),
                    "error": task_result.error,
                    "error_type": task_result.error_type,
                    "execution_time": task_result.execution_time,
                    "code": task_result.code
                }]
            
            for iteration_index, iteration_entry in enumerate(iteration_history, start=1):
                output_error_type = _extract_error_type_from_output(task_result.output)
                resolved_error_type = (
                    iteration_entry.get("error_type")
                    or task_result.error_type
                    or output_error_type
                )
                result_dict = {
                    "iteration": iteration_index,
                    "parameters": task_result.parameters,
                    "missing_parameters": task_result.missing_parameters,
                    "code": iteration_entry.get("code") or task_result.code,
                    "execution_result": _filter_progress_messages(task_result.output),
                    "error": iteration_entry.get("error") or task_result.error,
                    "error_type": resolved_error_type,
                    "execution_time": iteration_entry.get("execution_time") or task_result.execution_time,
                    "confidence_score": task_result.confidence_score,
                    "retry_count": task_result.retry_count
                }
                logger.log_task_execution(
                    task_id=task_id,
                    status=iteration_entry.get("status") or (
                        task_result.status.value if (task_result.status and hasattr(task_result.status, "value"))
                        else str(task_result.status) if task_result.status else "None"
                    ),
                    execution_mode=task_result.execution_mode,
                    result=result_dict,
                    error=result_dict["error"],
                    error_type=resolved_error_type
                )
    
    # ==================== 最终汇总 ====================
    print(f"\n{'='*80}")
    print(f"【4. 执行汇总】")
    print(f"{'='*80}")
    print(f"总任务数: {final_state.total_tasks} | 完成: {final_state.completed_count} | 失败: {final_state.failed_count}")
    print(f"{'='*80}\n")
    hitl_request_history = getattr(final_state, "hitl_request_history", None)
    hitl_response_history = getattr(final_state, "hitl_response_history", None)
    hitl_request_source = hitl_request_history if isinstance(hitl_request_history, dict) else final_state.hitl_requests
    hitl_response_source = hitl_response_history if isinstance(hitl_response_history, dict) else final_state.hitl_responses
    hitl_request_count = len(hitl_request_source)
    hitl_response_count = len(hitl_response_source)
    hitl_pending_count = len([k for k in hitl_request_source.keys() if k not in hitl_response_source])
    if logger and hasattr(logger, "logs"):
        hitl_request_tasks = {
            log.get("data", {}).get("task_id")
            for log in logger.logs
            if log.get("event_type") == "hitl_request"
        }
        hitl_request_tasks.discard(None)
        hitl_response_tasks = {
            log.get("data", {}).get("task_id")
            for log in logger.logs
            if log.get("event_type") == "hitl_response"
        }
        hitl_response_tasks.discard(None)
        if hitl_request_tasks or hitl_response_tasks:
            hitl_request_count = len(hitl_request_tasks)
            hitl_response_count = len(hitl_response_tasks)
            hitl_pending_count = len(hitl_request_tasks - hitl_response_tasks)
    print(f"HITL请求数: {hitl_request_count}")
    print(f"HITL响应数: {hitl_response_count}")
    print(f"HITL未响应数: {hitl_pending_count}")
    print(f"中断迭代次数: {iteration_count}")
    
    # 打印每个任务的执行状态
    print(f"\n任务执行详情:")
    
    for task in all_tasks:
        task_result = final_state.task_results.get(task.task_id)
        if task_result:
            status_icon = "✓" if task_result.status == ExecutorTaskStatus.COMPLETED else "✗" if task_result.status == ExecutorTaskStatus.FAILED else "⏳"
            status_str = task_result.status.value if (task_result.status and hasattr(task_result.status, 'value')) else str(task_result.status) if task_result.status else "None"
            print(f"  {status_icon} {task.task_id}: {status_str}")
            if task_result.error:
                print(f"     错误: {task_result.error}")
            if task_result.retry_count > 0:
                print(f"     重试次数: {task_result.retry_count}")
            if task_result.result_summary:
                summary_full = json.dumps(task_result.result_summary, ensure_ascii=False)
                print(f"     执行汇总: {summary_full}")
        else:
            print(f"  ? {task.task_id}: 未执行")
    
    # 打印 executor 汇总信息（从 merged_result 中提取）
    if global_state.merged_result and "executor_results" in global_state.merged_result:
        executor_results = global_state.merged_result["executor_results"]
        print(f"\n  Executor 执行汇总:")
        print(f"    总任务数: {executor_results.get('total_tasks', 0)}")
        print(f"    已完成: {executor_results.get('completed', 0)}")
        print(f"    失败: {executor_results.get('failed', 0)}")
    
    # 记录汇总结果
    if logger:
        preprocess_info = global_state.merged_result.get("preprocess", {})
        logger.log_summary({
            "user_input": user_input,
            "execution_plan": global_state.execution_plan,
            "preprocess": {
                "session_id": preprocess_info.get("session_id"),
                "sandbox_data_dir": preprocess_info.get("sandbox_data_dir"),
                "file_count": len(preprocess_info.get("file_analyses", [])),
                "extracted_parameters": preprocess_info.get("extracted_parameters", {}).get("user_parameters", {})
            },
            "task_classification": global_state.user_task_type.value if hasattr(global_state.user_task_type, 'value') else str(global_state.user_task_type),
            "supervisor_decision": global_state.supervisor_decision,
            "supervisor_reasoning": global_state.supervisor_reasoning,
            "total_tasks": final_state.total_tasks,
            "completed": final_state.completed_count,
            "failed": final_state.failed_count,
            "hitl_requests_count": hitl_request_count,
            "hitl_responses_count": hitl_response_count,
            "hitl_pending_count": hitl_pending_count,
            "interrupt_iterations": iteration_count,
            "tasks_count": len(all_tasks),
            "parallel_groups_count": len(global_state.parallel_task_groups),
            "use_react_supervisor": use_react_supervisor,
            "use_react_executor": use_react_executor,
            "react_max_steps": react_max_steps
        })
    
    print(f"{'='*80}\n")
    
    # 验证基本结果
    assert final_state.total_tasks == len(all_tasks), f"任务数不匹配: {final_state.total_tasks} != {len(all_tasks)}"
    assert final_state.completed_count + final_state.failed_count <= final_state.total_tasks, "完成+失败数不应超过总任务数"
    
    # 返回测试结果
    return {
        "test_case_id": test_case["id"],
        "test_case_name": test_case["name"],
        "difficulty": test_case["difficulty"],
        "total_tasks": final_state.total_tasks,
        "completed": final_state.completed_count,
        "failed": final_state.failed_count,
        "success": final_state.failed_count == 0
    }


# ==================== pytest 参数化测试 ====================

@pytest.fixture(autouse=True)
def test_case_logger(request):
    """为每个测试用例创建日志记录器"""
    global_logger = get_global_logger()
    if global_logger:
        test_case_name = request.node.name
        logger = global_logger.get_test_case_logger(test_case_name)
        yield logger
        global_logger.finish_test_case(test_case_name)
    else:
        yield None


# 生成 pytest 参数
def get_flu_test_ids():
    """获取 Flu Benchmark 测试 ID 列表"""
    return [case["id"] for case in FLU_BENCHMARK_CASES]


def get_flu_test_cases():
    """获取 Flu Benchmark 测试用例列表"""
    return FLU_BENCHMARK_CASES


def get_tcr_icon_test_ids():
    """获取 TCR ICON Benchmark 测试 ID 列表"""
    return [case["id"] for case in TCR_ICON_BENCHMARK_CASES]


def get_tcr_icon_test_cases():
    """获取 TCR ICON Benchmark 测试用例列表"""
    return TCR_ICON_BENCHMARK_CASES


# Flu Benchmark 参数化测试
@pytest.mark.parametrize("test_case", get_flu_test_cases(), ids=get_flu_test_ids())
def test_flu_benchmark(test_case: BenchmarkCase, request, test_case_logger):
    """
    测试 Flu Benchmark 系列问题
    
    运行方式:
    - 测试所有问题: pytest tests/test_full_workflow_detailed.py::test_flu_benchmark -v -s
    - 测试单个问题: pytest tests/test_full_workflow_detailed.py::test_flu_benchmark -v -s -k "Q01"
    - 测试简单问题: pytest tests/test_full_workflow_detailed.py::test_flu_benchmark -v -s -k "simple"
    - 测试复杂问题: pytest tests/test_full_workflow_detailed.py::test_flu_benchmark -v -s -k "complex"
    """
    # 创建测试目录
    test_dir = Path(f"./sandbox/flu_benchmark_{test_case['id']}")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取日志记录器
    logger = test_case_logger
    if logger is None:
        global_logger = get_global_logger()
        if global_logger:
            logger = global_logger.get_test_case_logger(f"flu_{test_case['id']}")
    
    # 执行测试
    result = run_single_test_case(
        test_case=test_case,
        test_dir=test_dir,
        logger=logger
    )
    
    # 验证结果
    assert result is not None, f"测试用例 {test_case['id']} 返回空结果"
    assert result["total_tasks"] > 0, f"测试用例 {test_case['id']} 没有生成任务"


# TCR ICON Benchmark 参数化测试
@pytest.mark.parametrize("test_case", get_tcr_icon_test_cases(), ids=get_tcr_icon_test_ids())
def test_tcr_icon_benchmark(test_case: BenchmarkCase, request, test_case_logger):
    """
    测试 TCR ICON Benchmark 系列问题
    
    运行方式:
    - 测试所有问题: pytest tests/test_full_workflow_detailed.py::test_tcr_icon_benchmark -v -s
    - 测试单个问题: pytest tests/test_full_workflow_detailed.py::test_tcr_icon_benchmark -v -s -k "Q13"
    - 测试简单问题: pytest tests/test_full_workflow_detailed.py::test_tcr_icon_benchmark -v -s -k "simple"
    - 测试复杂问题: pytest tests/test_full_workflow_detailed.py::test_tcr_icon_benchmark -v -s -k "complex"
    """
    # 创建测试目录
    test_dir = Path(f"./sandbox/tcr_icon_benchmark_{test_case['id']}")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取日志记录器
    logger = test_case_logger
    if logger is None:
        global_logger = get_global_logger()
        if global_logger:
            logger = global_logger.get_test_case_logger(f"tcr_icon_{test_case['id']}")
    
    # 执行测试
    result = run_single_test_case(
        test_case=test_case,
        test_dir=test_dir,
        logger=logger
    )
    
    # 验证结果
    assert result is not None, f"测试用例 {test_case['id']} 返回空结果"
    assert result["total_tasks"] > 0, f"测试用例 {test_case['id']} 没有生成任务"


# ==================== 批量运行测试 ====================

def test_run_all_simple_cases(request, test_case_logger):
    """批量运行所有简单难度的测试用例"""
    simple_cases = [case for case in FLU_BENCHMARK_CASES if case["difficulty"] == "simple"]
    
    results = []
    for test_case in simple_cases:
        test_dir = Path(f"./sandbox/flu_benchmark_{test_case['id']}")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        logger = test_case_logger
        if logger is None:
            global_logger = get_global_logger()
            if global_logger:
                logger = global_logger.get_test_case_logger(f"batch_{test_case['id']}")
        
        try:
            result = run_single_test_case(
                test_case=test_case,
                test_dir=test_dir,
                logger=logger
            )
            results.append(result)
        except Exception as e:
            results.append({
                "test_case_id": test_case["id"],
                "test_case_name": test_case["name"],
                "error": str(e),
                "success": False
            })
    
    # 打印汇总结果
    print(f"\n{'='*80}")
    print(f"【批量测试汇总】")
    print(f"{'='*80}")
    for result in results:
        status = "✓" if result.get("success", False) else "✗"
        print(f"  {status} {result['test_case_id']}: {result['test_case_name']}")
        if not result.get("success", False) and "error" in result:
            print(f"     错误: {result['error'][:100]}...")
    
    success_count = sum(1 for r in results if r.get("success", False))
    print(f"\n  通过: {success_count}/{len(results)}")
    print(f"{'='*80}\n")


def test_run_all_complex_cases(request, test_case_logger):
    """批量运行所有复杂难度的测试用例"""
    complex_cases = [case for case in FLU_BENCHMARK_CASES if case["difficulty"] == "complex"]
    
    results = []
    for test_case in complex_cases:
        test_dir = Path(f"./sandbox/flu_benchmark_{test_case['id']}")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        logger = test_case_logger
        if logger is None:
            global_logger = get_global_logger()
            if global_logger:
                logger = global_logger.get_test_case_logger(f"batch_{test_case['id']}")
        
        try:
            result = run_single_test_case(
                test_case=test_case,
                test_dir=test_dir,
                logger=logger
            )
            results.append(result)
        except Exception as e:
            results.append({
                "test_case_id": test_case["id"],
                "test_case_name": test_case["name"],
                "error": str(e),
                "success": False
            })
    
    # 打印汇总结果
    print(f"\n{'='*80}")
    print(f"【批量测试汇总 - 复杂问题】")
    print(f"{'='*80}")
    for result in results:
        status = "✓" if result.get("success", False) else "✗"
        print(f"  {status} {result['test_case_id']}: {result['test_case_name']}")
        if not result.get("success", False) and "error" in result:
            print(f"     错误: {result['error'][:100]}...")
    
    success_count = sum(1 for r in results if r.get("success", False))
    print(f"\n  通过: {success_count}/{len(results)}")
    print(f"{'='*80}\n")


# ==================== 主函数，支持直接运行 ====================

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行 Benchmark 测试")
    parser.add_argument("--case", "-c", type=str, default=None,
                       help="指定测试用例 ID (如 Q01, Q13, Q15)")
    parser.add_argument("--difficulty", "-d", type=str, default=None,
                       choices=["simple", "medium", "complex"],
                       help="指定测试难度")
    parser.add_argument("--all", "-a", action="store_true",
                       help="运行所有测试用例")
    parser.add_argument("--benchmark", "-b", type=str, default="flu",
                       choices=["flu", "tcr_icon", "all"],
                       help="指定 benchmark 类型: flu (默认), tcr_icon, all")
    args = parser.parse_args()
    
    # 启用 OpenSandbox
    os.environ["OPENSANDBOX_ENABLED"] = "true"
    os.environ["CODEACT_SANDBOX_PROVIDER"] = "opensandbox"
    os.environ["OPENSANDBOX_SKIP_MCP_INSTALL"] = "true"
    
    # 初始化日志
    test_file_name = Path(__file__).stem
    init_global_logger(test_file_name)
    
    try:
        # 合并所有测试用例
        all_cases = []
        if args.benchmark in ["flu", "all"]:
            all_cases.extend(FLU_BENCHMARK_CASES)
        if args.benchmark in ["tcr_icon", "all"]:
            all_cases.extend(TCR_ICON_BENCHMARK_CASES)
        
        # 筛选测试用例
        if args.case:
            cases_to_run = [c for c in all_cases if c["id"] == args.case]
        elif args.difficulty:
            cases_to_run = [c for c in all_cases if c["difficulty"] == args.difficulty]
        elif args.all:
            cases_to_run = all_cases
        else:
            # 默认运行第一个简单用例作为演示
            cases_to_run = [c for c in all_cases if c["difficulty"] == "simple"][:1]
        
        if not cases_to_run:
            print("未找到匹配的测试用例")
            exit(1)
        
        print(f"\n{'='*80}")
        print(f"【即将运行 {len(cases_to_run)} 个测试用例】")
        print(f"{'='*80}")
        for case in cases_to_run:
            print(f"  - {case['id']}: {case['name']} ({case['difficulty']})")
        print(f"{'='*80}\n")
        
        # 运行测试
        global_logger = get_global_logger()
        results = []
        
        for test_case in cases_to_run:
            # 根据测试用例 ID 确定所属 benchmark
            if test_case["id"] in [c["id"] for c in TCR_ICON_BENCHMARK_CASES]:
                benchmark_name = "tcr_icon_benchmark"
            else:
                benchmark_name = "flu_benchmark"
            test_dir = Path(f"./sandbox/{benchmark_name}_{test_case['id']}")
            test_dir.mkdir(parents=True, exist_ok=True)
            
            logger = None
            if global_logger:
                logger = global_logger.get_test_case_logger(f"{benchmark_name}_{test_case['id']}")
            
            try:
                result = run_single_test_case(
                    test_case=test_case,
                    test_dir=test_dir,
                    logger=logger
                )
                results.append(result)
            except Exception as e:
                import traceback
                print(f"\n✗ 测试失败: {test_case['id']}")
                print(f"   错误: {e}")
                traceback.print_exc()
                results.append({
                    "test_case_id": test_case["id"],
                    "test_case_name": test_case["name"],
                    "error": str(e),
                    "success": False
                })
            
            if global_logger:
                global_logger.finish_test_case(f"flu_{test_case['id']}")
        
        # 打印最终汇总
        print(f"\n{'='*80}")
        print(f"【测试汇总】")
        print(f"{'='*80}")
        for result in results:
            status = "✓" if result.get("success", False) else "✗"
            print(f"  {status} {result['test_case_id']}: {result['test_case_name']}")
            if result.get("total_tasks"):
                print(f"     任务: {result['total_tasks']} | 完成: {result['completed']} | 失败: {result['failed']}")
            if not result.get("success", False) and "error" in result:
                print(f"     错误: {result['error'][:100]}...")
        
        success_count = sum(1 for r in results if r.get("success", False))
        print(f"\n  总计: {success_count}/{len(results)} 通过")
        print(f"{'='*80}\n")
        
    finally:
        # 保存日志
        save_global_logger()

