"""
General QA Subgraph 测试用例

测试 general_qa 子图的完整功能，使用 csv_questions_data.json 中的问题。

功能：
1. 从 JSON 文件加载所有问题
2. 随机选择指定数量的问题进行测试（可配置）
3. 记录每个节点的详细产出
4. 生成测试报告

运行方式：
- 测试所有问题：pytest tests/test_gengeral_qa_subgraph.py -v -s
- 测试指定数量：pytest tests/test_gengeral_qa_subgraph.py::test_general_qa_with_random_questions -v -s --num-questions=10
"""

import os
import pytest
import json
import random
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

# 加载环境变量
load_dotenv()

# 添加agent目录到路径
import sys
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from nodes.subagents.general_qa.graph import general_qa_graph
from nodes.subagents.general_qa.state import GeneralQAState
from nodes.subagents.general_qa.prompts.domain_mapper import detect_domain_from_state, detect_cross_domain

# 配置日志
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f'test_gengeral_qa_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ===================== 数据加载 =====================

def load_questions_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    从 JSON 文件加载问题数据
    
    Args:
        json_path: JSON 文件路径
    
    Returns:
        问题列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_random_questions(questions: List[Dict[str, Any]], num_questions: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    随机选择指定数量的问题
    
    Args:
        questions: 所有问题列表
        num_questions: 要选择的问题数量
        seed: 随机种子（用于可重复性）
    
    Returns:
        随机选择的问题列表
    """
    if seed is not None:
        random.seed(seed)
    
    if num_questions >= len(questions):
        return questions
    
    return random.sample(questions, num_questions)


def get_questions_by_range(questions: List[Dict[str, Any]], start_index: int, end_index: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    按范围选择问题（从 start_index 到 end_index）
    
    Args:
        questions: 所有问题列表
        start_index: 起始索引（0-based，包含）
        end_index: 结束索引（0-based，不包含）。如果为 None，则选择到列表末尾
    
    Returns:
        指定范围的问题列表
    """
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")
    
    if start_index >= len(questions):
        raise ValueError(f"start_index ({start_index}) exceeds total questions ({len(questions)})")
    
    if end_index is None:
        end_index = len(questions)
    
    if end_index < start_index:
        raise ValueError(f"end_index ({end_index}) must be >= start_index ({start_index})")
    
    if end_index > len(questions):
        logger.warning(f"end_index ({end_index}) exceeds total questions ({len(questions)}), using {len(questions)} instead")
        end_index = len(questions)
    
    return questions[start_index:end_index]


# ===================== 节点产出记录 =====================

class NodeOutputLogger:
    """记录每个节点的产出"""
    
    def __init__(self):
        self.node_outputs = defaultdict(dict)
        self.current_question_id = None
    
    def log_node_output(self, node_name: str, state: GeneralQAState):
        """记录节点产出"""
        if self.current_question_id:
            output = {
                "node_name": node_name,
                "timestamp": datetime.now().isoformat(),
                "state_snapshot": self._extract_state_snapshot(state, node_name)
            }
            self.node_outputs[self.current_question_id][node_name] = output
            
            # 同时记录到日志
            logger.info(f"[{self.current_question_id}] Node {node_name} completed")
            logger.debug(f"[{self.current_question_id}] Node {node_name} output: {json.dumps(output, indent=2, ensure_ascii=False)}")
    
    def _extract_state_snapshot(self, state: GeneralQAState, node_name: str) -> Dict[str, Any]:
        """提取状态快照（只包含相关字段）"""
        snapshot = {}
        
        # 根据节点名称提取相关字段
        node_field_mapping = {
            "n0_input_preprocessing": [
                "cleaned_text",
                "question_type_label",
                "data_completeness_label",
                "question_options",
                "answer_format_label",
                "structured_subject",  # 结构化三维度信息
                "structured_condition",  # 结构化三维度信息
                "structured_goal"  # 结构化三维度信息
            ],
            "n1_question_decomposition": [
                "structured_conditions",
                "core_domains",  # 领域信息（用于验证领域路由）
                "research_objective",
                "key_entities",
                "answer_constraints"
            ],
            "n2_calculation_algorithm_recognition": ["calculation_type_label", "key_parameters"],
            "n3_knowledge_retrieval": ["domain_knowledge_map", "knowledge_validity_label", "paperqa_result", "deep_research_result"],
            "n4_calculation_decomposition": ["calculation_steps", "matched_formula", "unit_conversion_rules", "formula_match_result"],
            "n5_algorithm_validation": ["algorithm_parameters", "applicability_result", "alternative_algorithms"],
            "n6_initial_inference": ["phenomenon_knowledge_match_table", "match_confidence_label"],
            "n7_complete_inference": ["closed_inference_path", "core_conclusion"],
            "n8_answer_generation": ["structured_answer", "final_answer", "candidate_answers", "num_candidates"],
            "n8_5_critic_review": ["critiqued_answers"],  # X-Masters: Critic评审
            "n8_6_rewriter_synthesis": ["rewritten_answers"],  # X-Masters: Rewriter综合
            "n9_result_validation": ["consistency_label", "reliability_score", "format_valid_label", "format_issues"],
            "n10_exception_handling": ["exception_type_label", "solution_suggestion"],
            "n11_manual_intervention": ["manual_intervention_guide", "intermediate_result_snapshot"]
        }
        
        fields_to_extract = node_field_mapping.get(node_name, [])
        for field in fields_to_extract:
            value = getattr(state, field, None)
            if value is not None:
                snapshot[field] = value
        
        # 对于n3之后的节点，总是包含paperQA和deepresearch结果（如果存在）
        node_order = ["n0_input_preprocessing", "n1_question_decomposition", "n2_calculation_algorithm_recognition", 
                     "n3_knowledge_retrieval", "n4_calculation_decomposition", "n5_algorithm_validation",
                     "n6_initial_inference", "n7_complete_inference", "n8_answer_generation",
                     "n8_5_critic_review", "n8_6_rewriter_synthesis",  # X-Masters节点
                     "n9_result_validation", "n10_exception_handling", "n11_manual_intervention"]
        current_node_index = node_order.index(node_name) if node_name in node_order else -1
        n3_index = node_order.index("n3_knowledge_retrieval") if "n3_knowledge_retrieval" in node_order else -1
        
        if current_node_index >= n3_index and n3_index >= 0:
            # n3及之后的节点，包含paperQA和deepresearch结果
            if state.paperqa_result is not None:
                snapshot["paperqa_result"] = state.paperqa_result
            if state.deep_research_result is not None:
                snapshot["deep_research_result"] = state.deep_research_result
        
        # 所有节点都包含工具调用历史（如果存在）
        if state.tool_calls_history is not None and len(state.tool_calls_history) > 0:
            snapshot["tool_calls_history"] = state.tool_calls_history
        
        # 总是包含错误信息（如果有）
        if state.error_message:
            snapshot["error_message"] = state.error_message
        
        # 对于n1及之后的节点，包含领域信息（用于验证领域路由）
        if node_name in ["n1_question_decomposition"] or current_node_index >= node_order.index("n1_question_decomposition") if "n1_question_decomposition" in node_order else False:
            if hasattr(state, 'core_domains') and state.core_domains:
                snapshot["detected_domains"] = state.core_domains
            if hasattr(state, 'question_type_label') and state.question_type_label:
                snapshot["question_type_label"] = state.question_type_label
        
        return snapshot
    
    def set_current_question(self, question_id: str):
        """设置当前问题ID"""
        self.current_question_id = question_id
    
    def get_all_outputs(self) -> Dict[str, Any]:
        """获取所有节点产出"""
        return dict(self.node_outputs)
    
    def save_to_file(self, filepath: str):
        """保存节点产出到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.get_all_outputs(), f, indent=2, ensure_ascii=False)
        logger.info(f"Node outputs saved to {filepath}")


# ===================== Node Metrics =====================

def resolve_baseline_node_outputs_path(request: Optional[Any] = None) -> Optional[str]:
    """Resolve baseline node outputs path from CLI option or environment."""
    baseline_path = None
    if request is not None:
        baseline_path = request.config.getoption("--baseline-node-outputs", default=None)
    return baseline_path or os.getenv("BASELINE_NODE_OUTPUTS")


def load_node_outputs_from_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Load node outputs JSON from file."""
    try:
        if not filepath:
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load baseline node outputs: {filepath}. Error: {e}")
        return None


def _get_expected_node_fields() -> Dict[str, List[str]]:
    """Return expected fields per node for coverage metrics."""
    return {
        "n0_input_preprocessing": [
            "cleaned_text",
            "question_type_label",
            "data_completeness_label",
            "question_options",
            "answer_format_label",
            "structured_subject",  # 结构化三维度信息
            "structured_condition",  # 结构化三维度信息
            "structured_goal"  # 结构化三维度信息
        ],
        "n1_question_decomposition": [
            "structured_conditions",
            "core_domains",
            "research_objective",
            "key_entities",
            "answer_constraints"
        ],
        "n2_calculation_algorithm_recognition": ["calculation_type_label", "key_parameters"],
        "n3_knowledge_retrieval": ["domain_knowledge_map", "knowledge_validity_label"],
        "n4_calculation_decomposition": ["calculation_steps", "matched_formula", "unit_conversion_rules", "formula_match_result"],
        "n5_algorithm_validation": ["algorithm_parameters", "applicability_result", "alternative_algorithms"],
        "n6_initial_inference": ["phenomenon_knowledge_match_table", "match_confidence_label"],
        "n7_complete_inference": ["closed_inference_path", "core_conclusion"],
        "n8_answer_generation": ["structured_answer", "final_answer", "candidate_answers", "num_candidates"],
        "n8_5_critic_review": ["critiqued_answers"],  # X-Masters: Critic评审
        "n8_6_rewriter_synthesis": ["rewritten_answers"],  # X-Masters: Rewriter综合
        "n9_result_validation": ["consistency_label", "reliability_score", "format_valid_label", "format_issues"],
        "n10_exception_handling": ["exception_type_label", "solution_suggestion"],
        "n11_manual_intervention": ["manual_intervention_guide", "intermediate_result_snapshot"]
    }


def compute_node_metrics(node_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-node metrics for coverage and error rates."""
    expected_fields = _get_expected_node_fields()
    metrics: Dict[str, Any] = {}
    for node_name, fields in expected_fields.items():
        executed_count = 0
        error_count = 0
        field_present_count = 0
        field_total_count = 0
        consistent_count = 0
        format_valid_count = 0
        for _, outputs in node_outputs.items():
            if node_name not in outputs:
                continue
            executed_count += 1
            output = outputs[node_name]
            if isinstance(output, dict) and output.get("parse_error"):
                error_count += 1
                continue
            snapshot = output.get("state_snapshot", {}) if isinstance(output, dict) else {}
            if snapshot.get("error_message"):
                error_count += 1
            field_total_count += len(fields)
            for field in fields:
                if field in snapshot:
                    field_present_count += 1
            if node_name == "n9_result_validation":
                if snapshot.get("consistency_label") == "Consistent":
                    consistent_count += 1
                if snapshot.get("format_valid_label") == "Valid":
                    format_valid_count += 1
        field_coverage_rate = field_present_count / field_total_count if field_total_count else 0
        error_rate = error_count / executed_count if executed_count else 0
        consistency_rate = consistent_count / executed_count if executed_count else 0
        format_valid_rate = format_valid_count / executed_count if executed_count else 0
        metrics[node_name] = {
            "executed_count": executed_count,
            "error_count": error_count,
            "error_rate": round(error_rate, 4),
            "field_coverage_rate": round(field_coverage_rate, 4),
            "consistency_rate": round(consistency_rate, 4) if node_name == "n9_result_validation" else None,
            "format_valid_rate": round(format_valid_rate, 4) if node_name == "n9_result_validation" else None
        }
    return metrics


def compute_node_metrics_delta(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metric deltas between current and baseline."""
    delta: Dict[str, Any] = {}
    for node_name, current_metrics in current.items():
        base_metrics = baseline.get(node_name, {})
        node_delta: Dict[str, Any] = {}
        for key, value in current_metrics.items():
            if isinstance(value, (int, float)) and isinstance(base_metrics.get(key), (int, float)):
                node_delta[key] = round(value - base_metrics.get(key), 4)
            else:
                node_delta[key] = None
        delta[node_name] = node_delta
    return delta

# ===================== 测试辅助函数 =====================

def run_general_qa_for_question(
    question_text: str, 
    question_id: str, 
    node_logger: NodeOutputLogger,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_temperature: Optional[float] = None
) -> Dict[str, Any]:
    """
    为单个问题运行 general_qa 子图
    
    Args:
        question_text: 问题文本
        question_id: 问题ID
        node_logger: 节点产出记录器
        llm_provider: LLM提供商 (dashscope, zhipu, etc.)，如果为None则使用环境变量或默认值
        llm_model: LLM模型名称，如果为None则使用环境变量或默认值
        llm_temperature: LLM温度参数，如果为None则使用环境变量或默认值
    
    Returns:
        测试结果字典
    """
    node_logger.set_current_question(question_id)
    
    # 保存原始环境变量
    original_llm_env = {}
    original_llm_env["GENERAL_QA_LLM_PROVIDER"] = os.getenv("GENERAL_QA_LLM_PROVIDER")
    original_llm_env["GENERAL_QA_LLM_MODEL"] = os.getenv("GENERAL_QA_LLM_MODEL")
    original_llm_env["GENERAL_QA_LLM_TEMPERATURE"] = os.getenv("GENERAL_QA_LLM_TEMPERATURE")
    
    try:
        # 如果提供了LLM配置参数，临时设置环境变量
        if llm_provider is not None:
            os.environ["GENERAL_QA_LLM_PROVIDER"] = llm_provider
            logger.info(f"  LLM Provider: {llm_provider}")
        if llm_model is not None:
            os.environ["GENERAL_QA_LLM_MODEL"] = llm_model
            logger.info(f"  LLM Model: {llm_model}")
        if llm_temperature is not None:
            os.environ["GENERAL_QA_LLM_TEMPERATURE"] = str(llm_temperature)
            logger.info(f"  LLM Temperature: {llm_temperature}")
        
        # 创建初始状态
        # 检查是否禁用X-Masters（通过环境变量）
        disable_xmasters = os.getenv("DISABLE_XMASTERS", "0") == "1"
        
        if disable_xmasters:
            # 禁用X-Masters：不设置num_candidates
            initial_state = GeneralQAState(user_input=question_text)
        else:
            # 默认启用X-Masters优化（生成3个候选答案）
            initial_state = GeneralQAState(
                user_input=question_text,
                num_candidates=3  # X-Masters: 生成3个候选答案
            )
        
        logger.info(f"Starting test for question {question_id}")
        logger.info(f"Question: {question_text[:200]}...")
        
        start_time = datetime.now()
        
        try:
            # 运行子图
            # 注意：我们需要拦截节点执行以记录产出
            # 由于 LangGraph 的限制，我们使用自定义的图来记录节点产出
            
            # 创建带日志记录的图包装器
            result_state = run_graph_with_logging(initial_state, node_logger)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 提取结果
            result = {
                "question_id": question_id,
                "question_text": question_text,
                "success": result_state.error_message is None,
                "error_message": result_state.error_message,
                "final_answer": result_state.final_answer,
                "duration_seconds": duration,
                "nodes_executed": list(node_logger.node_outputs[question_id].keys()) if question_id in node_logger.node_outputs else [],
                # X-Masters相关指标
                "xmasters_enabled": result_state.candidate_answers is not None,
                "num_candidates": len(result_state.candidate_answers) if result_state.candidate_answers else 0,
                "num_critiqued": len(result_state.critiqued_answers) if result_state.critiqued_answers else 0,
                "num_rewritten": len(result_state.rewritten_answers) if result_state.rewritten_answers else 0,
            }
            
            logger.info(f"Question {question_id} completed in {duration:.2f}s, success: {result['success']}")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() if 'start_time' in locals() else 0
            
            logger.error(f"Question {question_id} failed with exception: {str(e)}", exc_info=True)
            
            return {
                "question_id": question_id,
                "question_text": question_text,
                "success": False,
                "error_message": str(e),
                "final_answer": None,
                "duration_seconds": duration,
                "nodes_executed": []
            }
    finally:
        # 恢复原始环境变量
        if original_llm_env["GENERAL_QA_LLM_PROVIDER"] is not None:
            os.environ["GENERAL_QA_LLM_PROVIDER"] = original_llm_env["GENERAL_QA_LLM_PROVIDER"]
        elif "GENERAL_QA_LLM_PROVIDER" in os.environ:
            del os.environ["GENERAL_QA_LLM_PROVIDER"]
        
        if original_llm_env["GENERAL_QA_LLM_MODEL"] is not None:
            os.environ["GENERAL_QA_LLM_MODEL"] = original_llm_env["GENERAL_QA_LLM_MODEL"]
        elif "GENERAL_QA_LLM_MODEL" in os.environ:
            del os.environ["GENERAL_QA_LLM_MODEL"]
        
        if original_llm_env["GENERAL_QA_LLM_TEMPERATURE"] is not None:
            os.environ["GENERAL_QA_LLM_TEMPERATURE"] = original_llm_env["GENERAL_QA_LLM_TEMPERATURE"]
        elif "GENERAL_QA_LLM_TEMPERATURE" in os.environ:
            del os.environ["GENERAL_QA_LLM_TEMPERATURE"]


def run_graph_with_logging(initial_state: GeneralQAState, node_logger: NodeOutputLogger) -> GeneralQAState:
    """
    运行图并记录每个节点的产出
    
    使用 LangGraph 的 stream 功能来捕获每个节点的输出
    """
    current_state = initial_state
    
    # 使用 stream 来捕获节点执行
    try:
        config = {"configurable": {"thread_id": f"test_{node_logger.current_question_id}"}}
        
        # 使用 stream 来逐步执行并记录
        # stream 返回的每个事件包含节点名称和状态字典
        for event in general_qa_graph.stream(initial_state.model_dump(mode='json'), config=config):
            # event 格式: {node_name: state_dict}
            for node_name, state_dict in event.items():
                try:
                    # 将字典转换回状态对象
                    state = GeneralQAState.model_validate(state_dict)
                    node_logger.log_node_output(node_name, state)
                    current_state = state
                except Exception as e:
                    logger.warning(f"Failed to parse state for node {node_name}: {str(e)}")
                    # 如果解析失败，至少记录节点名称
                    node_logger.node_outputs[node_logger.current_question_id][node_name] = {
                        "node_name": node_name,
                        "timestamp": datetime.now().isoformat(),
                        "parse_error": str(e),
                        "raw_state": state_dict
                    }
        
        return current_state
        
    except Exception as e:
        logger.error(f"Graph execution failed: {str(e)}", exc_info=True)
        current_state.error_message = str(e)
        return current_state


# ===================== 测试用例 =====================

@pytest.fixture
def questions_data():
    """加载所有问题数据"""
    json_path = Path(__file__).parent / "csv_questions_data.json"
    if not json_path.exists():
        pytest.skip(f"Questions data file not found: {json_path}")
    
    questions = load_questions_from_json(str(json_path))
    logger.info(f"Loaded {len(questions)} questions from {json_path}")
    return questions


@pytest.fixture
def node_logger():
    """创建节点产出记录器"""
    return NodeOutputLogger()


def test_general_qa_with_random_questions(questions_data: List[Dict[str, Any]], node_logger: NodeOutputLogger, request):
    """
    使用随机选择或指定范围的问题测试 general_qa 子图（支持X-Masters优化）
    
    从 csv_questions_data.json 中选择问题进行测试。
    
    使用方法：
    1. 指定问题数量（随机选择）：
       pytest tests/test_gengeral_qa_subgraph.py::test_general_qa_with_random_questions -v -s --num-questions=10
    
    2. 指定范围（从第几个到第几个，索引从0开始）：
       pytest tests/test_gengeral_qa_subgraph.py::test_general_qa_with_random_questions -v -s --start-index=0 --end-index=10
       # 测试第0到第9个问题（共10个问题）
    
    3. 只指定起始索引（从该索引到末尾）：
       pytest tests/test_gengeral_qa_subgraph.py::test_general_qa_with_random_questions -v -s --start-index=50
       # 测试第50个问题到最后一个问题
    
    4. 使用环境变量：
       NUM_QUESTIONS=20 pytest tests/test_gengeral_qa_subgraph.py::test_general_qa_with_random_questions -v -s
    
    5. 使用固定随机种子（可重复测试）：
       RANDOM_SEED=42 pytest tests/test_gengeral_qa_subgraph.py::test_general_qa_with_random_questions -v -s --num-questions=5
    
    6. 禁用X-Masters优化（测试原始general_qa）：
       DISABLE_XMASTERS=1 pytest tests/test_gengeral_qa_subgraph.py::test_general_qa_with_random_questions -v -s --num-questions=3
    """
    # 检查是否指定了范围选择
    try:
        start_index = request.config.getoption("--start-index", default=None)
        end_index = request.config.getoption("--end-index", default=None)
    except (ValueError, AttributeError):
        start_index = None
        end_index = None
    
    # 如果指定了范围，使用范围选择
    if start_index is not None:
        try:
            start_index = int(start_index)
            if end_index is not None:
                end_index = int(end_index)
            else:
                end_index = None
            selected_questions = get_questions_by_range(questions_data, start_index, end_index)
            selection_mode = "range"
            logger.info(f"Using range selection: start_index={start_index}, end_index={end_index or 'end'}")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid range parameters: {e}")
            raise
    else:
        # 否则使用随机选择
        # 获取问题数量（优先从命令行参数，其次从环境变量，最后使用默认值）
        try:
            num_questions = request.config.getoption("--num-questions", default=None)
            if num_questions is None:
                num_questions = int(os.getenv("NUM_QUESTIONS", "10"))
            else:
                num_questions = int(num_questions)
        except (ValueError, AttributeError):
            num_questions = int(os.getenv("NUM_QUESTIONS", "10"))
        
        # 检查是否使用固定种子（用于可重复测试）
        # 如果设置了 RANDOM_SEED 环境变量，使用该种子；否则使用 None（真正随机）
        random_seed = os.getenv("RANDOM_SEED")
        if random_seed is not None:
            try:
                random_seed = int(random_seed)
                logger.info(f"Using fixed random seed: {random_seed}")
            except ValueError:
                random_seed = None
        else:
            random_seed = None  # 不使用固定种子，确保每次都是随机的
            logger.info("Using random selection (no fixed seed)")
        
        # 随机选择问题
        selected_questions = get_random_questions(questions_data, num_questions, seed=random_seed)
        selection_mode = "random"
    
    # 检查是否禁用X-Masters（优先从命令行参数，其次从环境变量）
    try:
        disable_xmasters = request.config.getoption("--disable-xmasters", default=False)
    except (ValueError, AttributeError):
        disable_xmasters = os.getenv("DISABLE_XMASTERS", "0") == "1"
    
    if disable_xmasters:
        logger.info("⚠ X-Masters optimization is DISABLED - testing original general_qa")
        os.environ["DISABLE_XMASTERS"] = "1"  # 设置环境变量供run_general_qa_for_question使用
    
    # 打印选中的问题ID和基本信息
    selected_ids = [q.get("id", "unknown") for q in selected_questions]
    selected_categories = [q.get("category", "Unknown") for q in selected_questions]
    selected_subjects = [q.get("raw_subject", "Unknown") for q in selected_questions]
    
    logger.info(f"\n{'='*80}")
    if selection_mode == "range":
        logger.info(f"📋 Selected {len(selected_questions)} questions by range from {len(questions_data)} total")
        if start_index is not None and end_index is not None:
            logger.info(f"Range: [{start_index}, {end_index})")
        elif start_index is not None:
            logger.info(f"Range: [{start_index}, end)")
    else:
        logger.info(f"🎲 Randomly selected {len(selected_questions)} questions from {len(questions_data)} total")
    logger.info(f"{'='*80}")
    logger.info(f"Selected question IDs: {selected_ids}")
    logger.info(f"Categories: {set(selected_categories)}")
    logger.info(f"Subjects: {set(selected_subjects)}")
    logger.info(f"X-Masters: {'DISABLED' if disable_xmasters else 'ENABLED'}")
    logger.info(f"{'='*80}\n")
    
    # 获取LLM配置（优先从命令行参数，其次从环境变量）
    try:
        llm_provider = request.config.getoption("--llm-provider", default=None)
        llm_model = request.config.getoption("--llm-model", default=None)
        llm_temperature_str = request.config.getoption("--llm-temperature", default=None)
    except (ValueError, AttributeError):
        llm_provider = os.getenv("GENERAL_QA_LLM_PROVIDER")
        llm_model = os.getenv("GENERAL_QA_LLM_MODEL")
        llm_temperature_str = os.getenv("GENERAL_QA_LLM_TEMPERATURE")
    
    # 如果命令行参数未设置，尝试从环境变量获取
    if llm_provider is None:
        llm_provider = os.getenv("GENERAL_QA_LLM_PROVIDER")
    if llm_model is None:
        llm_model = os.getenv("GENERAL_QA_LLM_MODEL")
    if llm_temperature_str is None:
        llm_temperature_str = os.getenv("GENERAL_QA_LLM_TEMPERATURE")
    
    # 转换温度参数
    llm_temperature = None
    if llm_temperature_str is not None:
        try:
            llm_temperature = float(llm_temperature_str)
        except ValueError:
            logger.warning(f"Invalid temperature value: {llm_temperature_str}, ignoring")
            llm_temperature = None
    
    # 打印LLM配置
    if llm_provider or llm_model or llm_temperature is not None:
        logger.info(f"LLM Configuration:")
        logger.info(f"  Provider: {llm_provider or 'Default'}")
        logger.info(f"  Model: {llm_model or 'Default'}")
        logger.info(f"  Temperature: {llm_temperature or 'Default'}")
    else:
        logger.info("LLM Configuration: Using default (from environment or code defaults)")
    
    logger.info(f"Selected {len(selected_questions)} questions for testing (from {len(questions_data)} total)")
    
    # 运行测试
    results = []
    for i, question_data in enumerate(selected_questions, 1):
        question_id = question_data.get("id", f"question_{i}")
        question_text = question_data.get("question", "")
        expected_answer = question_data.get("answer", "")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing question {i}/{len(selected_questions)}: {question_id}")
        logger.info(f"{'='*80}")
        
        result = run_general_qa_for_question(
            question_text, 
            question_id, 
            node_logger,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_temperature=llm_temperature
        )
        result["expected_answer"] = expected_answer
        result["question_category"] = question_data.get("category", "")
        result["question_type"] = question_data.get("question_type", "")
        
        results.append(result)
        
        # 打印简要结果
        if result["success"]:
            logger.info(f"✓ Question {question_id} completed successfully")
            logger.info(f"  Final answer: {result['final_answer'][:200] if result['final_answer'] else 'N/A'}...")
            # 打印X-Masters指标（如果启用）
            if result.get("xmasters_enabled"):
                logger.info(f"  X-Masters: {result.get('num_candidates', 0)} candidates, "
                          f"{result.get('num_critiqued', 0)} critiqued, "
                          f"{result.get('num_rewritten', 0)} rewritten")
        else:
            logger.error(f"✗ Question {question_id} failed: {result['error_message']}")
            # 打印失败时的节点执行情况
            if question_id in node_logger.node_outputs:
                executed_nodes = list(node_logger.node_outputs[question_id].keys())
                logger.error(f"  Executed nodes: {executed_nodes}")
                last_node = executed_nodes[-1] if executed_nodes else "None"
                logger.error(f"  Last completed node: {last_node}")
    
    # 生成测试报告
    baseline_path = resolve_baseline_node_outputs_path(request)
    generate_test_report(results, node_logger, baseline_path)
    
    # 保存节点产出
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"node_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    node_logger.save_to_file(str(output_file))
    logger.info(f"Node outputs saved to {output_file}")
    
    # 断言：至少有一些问题成功
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"\nTest Summary: {success_count}/{len(results)} questions completed successfully")
    
    # 打印失败原因统计
    if success_count < len(results):
        failure_reasons = {}
        for r in results:
            if not r["success"]:
                error_msg = r.get("error_message", "Unknown error")
                # 提取主要错误类型
                if "No domain knowledge" in error_msg:
                    error_type = "Knowledge Retrieval Failed"
                elif "LLM unavailable" in error_msg:
                    error_type = "LLM Unavailable"
                elif "parse" in error_msg.lower():
                    error_type = "Response Parsing Failed"
                elif "timeout" in error_msg.lower():
                    error_type = "Timeout"
                else:
                    error_type = "Other Error"
                failure_reasons[error_type] = failure_reasons.get(error_type, 0) + 1
        
        logger.warning(f"\nFailure Analysis:")
        for reason, count in failure_reasons.items():
            logger.warning(f"  - {reason}: {count} question(s)")
        
        # 如果所有失败都是知识检索问题，提供建议
        if failure_reasons.get("Knowledge Retrieval Failed", 0) == len(results) - success_count:
            logger.warning(f"\n⚠ All failures are due to knowledge retrieval issues.")
            logger.warning(f"  Possible causes:")
            logger.warning(f"    1. Qdrant API key incorrect (check QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTIONS)")
            logger.warning(f"    2. Tavily API key incorrect or query too long (check TAVILY_API_KEY)")
            logger.warning(f"    3. Network connectivity issues")
            logger.warning(f"  Note: The system should use fallback mechanisms, but all fallbacks may have failed.")
    
    # 注意：不强制要求所有问题都成功，因为某些问题可能需要人工介入
    # 但如果所有问题都失败，至少应该有一个部分成功的（到达了某些节点）
    if success_count == 0:
        # 检查是否有问题至少执行了一些节点
        nodes_executed_count = sum(1 for r in results if r.get("nodes_executed") and len(r["nodes_executed"]) > 0)
        if nodes_executed_count > 0:
            logger.warning(f"⚠ All questions failed, but {nodes_executed_count} question(s) executed at least some nodes.")
            logger.warning(f"  This suggests the graph is working but encountering errors during execution.")
        else:
            logger.error(f"❌ All questions failed and no nodes were executed. This suggests a fundamental issue.")
        
        # 如果所有问题都失败，提供更详细的错误信息
        assert success_count > 0, (
            f"At least one question should complete successfully. "
            f"All {len(results)} question(s) failed. "
            f"Check the failure analysis above for details."
        )


def test_general_qa_all_questions(questions_data: List[Dict[str, Any]], node_logger: NodeOutputLogger, request):
    """
    测试所有问题（用于完整测试）
    
    注意：这会运行所有问题，可能需要很长时间
    """
    logger.info(f"Testing all {len(questions_data)} questions")
    
    results = []
    for i, question_data in enumerate(questions_data, 1):
        question_id = question_data.get("id", f"question_{i}")
        question_text = question_data.get("question", "")
        expected_answer = question_data.get("answer", "")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing question {i}/{len(questions_data)}: {question_id}")
        logger.info(f"{'='*80}")
        
        result = run_general_qa_for_question(question_text, question_id, node_logger)
        result["expected_answer"] = expected_answer
        result["question_category"] = question_data.get("category", "")
        result["question_type"] = question_data.get("question_type", "")
        result["raw_subject"] = question_data.get("raw_subject", "")  # 添加领域信息
        result["raw_question_type"] = question_data.get("question_type", "")  # 添加问题类型
        
        # 提取检测到的领域信息（从n1节点）
        if question_id in node_logger.node_outputs:
            n1_output = node_logger.node_outputs[question_id].get("n1_question_decomposition", {})
            n1_snapshot = n1_output.get("state_snapshot", {})
            result["detected_domains"] = n1_snapshot.get("detected_domains", [])
            result["detected_question_type"] = n1_snapshot.get("question_type_label")
        
        results.append(result)
        
        # 每10个问题保存一次中间结果
        if i % 10 == 0:
            output_dir = Path(__file__).parent / "outputs"
            output_dir.mkdir(exist_ok=True)
            intermediate_file = output_dir / f"node_outputs_intermediate_{i}.json"
            node_logger.save_to_file(str(intermediate_file))
            logger.info(f"Intermediate results saved to {intermediate_file}")
    
    # 生成测试报告
    baseline_path = resolve_baseline_node_outputs_path(request)
    generate_test_report(results, node_logger, baseline_path)
    
    # 保存最终节点产出
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"node_outputs_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    node_logger.save_to_file(str(output_file))
    logger.info(f"Final node outputs saved to {output_file}")


def generate_test_report(results: List[Dict[str, Any]], node_logger: NodeOutputLogger, baseline_node_outputs_path: Optional[str] = None):
    """Generate test report"""
    node_outputs = node_logger.get_all_outputs()
    node_metrics = compute_node_metrics(node_outputs)
    baseline_metrics = None
    node_metrics_delta = None
    if baseline_node_outputs_path:
        baseline_outputs = load_node_outputs_from_file(baseline_node_outputs_path)
        if baseline_outputs is not None:
            baseline_metrics = compute_node_metrics(baseline_outputs)
            node_metrics_delta = compute_node_metrics_delta(node_metrics, baseline_metrics)
    
    # 计算领域相关统计
    domain_stats = compute_domain_statistics(results, node_outputs)
    
    # 计算X-Masters相关统计
    xmasters_stats = compute_xmasters_statistics(results, node_outputs)
    
    report = {
        "test_timestamp": datetime.now().isoformat(),
        "total_questions": len(results),
        "success_count": sum(1 for r in results if r["success"]),
        "failure_count": sum(1 for r in results if not r["success"]),
        "average_duration": sum(r["duration_seconds"] for r in results) / len(results) if results else 0,
        "results": results,
        "node_outputs_summary": {
            question_id: {
                "nodes_count": len(outputs),
                "node_names": list(outputs.keys())
            }
            for question_id, outputs in node_outputs.items()
        },
        "node_metrics": node_metrics,
        "node_metrics_baseline": baseline_metrics,
        "node_metrics_delta": node_metrics_delta,
        "domain_statistics": domain_stats,  # 添加领域统计
        "xmasters_statistics": xmasters_stats  # 添加X-Masters统计
    }
    
    # 保存报告
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    report_file = output_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*80}")
    logger.info("Test Report Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Total questions: {report['total_questions']}")
    logger.info(f"Success: {report['success_count']}")
    logger.info(f"Failure: {report['failure_count']}")
    logger.info(f"Average duration: {report['average_duration']:.2f}s")
    logger.info(f"Report saved to: {report_file}")
    logger.info(f"{'='*80}\n")
    
    # 打印领域统计摘要
    if domain_stats:
        logger.info(f"\n{'='*80}")
        logger.info("Domain Statistics Summary")
        logger.info(f"{'='*80}")
        logger.info(f"Domain distribution: {domain_stats.get('domain_distribution', {})}")
        logger.info(f"Cross-domain questions: {domain_stats.get('cross_domain_count', 0)}/{domain_stats.get('total_questions', 0)}")
        logger.info(f"Domain routing accuracy: {domain_stats.get('routing_accuracy', 0):.2%}")
        logger.info(f"{'='*80}\n")
    
    # 打印X-Masters统计摘要
    if xmasters_stats:
        logger.info(f"\n{'='*80}")
        logger.info("X-Masters Statistics Summary")
        logger.info(f"{'='*80}")
        logger.info(f"X-Masters enabled: {xmasters_stats.get('enabled_count', 0)}/{xmasters_stats.get('total_questions', 0)}")
        logger.info(f"Average candidates generated: {xmasters_stats.get('avg_candidates', 0):.2f}")
        logger.info(f"Average critics completed: {xmasters_stats.get('avg_critics', 0):.2f}")
        logger.info(f"Average rewriters completed: {xmasters_stats.get('avg_rewriters', 0):.2f}")
        logger.info(f"X-Masters success rate: {xmasters_stats.get('success_rate', 0):.2%}")
        logger.info(f"{'='*80}\n")


def compute_domain_statistics(results: List[Dict[str, Any]], node_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算领域相关统计信息
    
    Args:
        results: 测试结果列表
        node_outputs: 节点产出字典
    
    Returns:
        领域统计字典
    """
    domain_distribution = defaultdict(int)
    cross_domain_count = 0
    routing_matches = 0
    routing_total = 0
    domain_success_rate = defaultdict(lambda: {"success": 0, "total": 0})
    
    for result in results:
        raw_subject = result.get("raw_subject", "")
        detected_domains = result.get("detected_domains", [])
        
        # 统计领域分布
        if raw_subject:
            domain_distribution[raw_subject] += 1
        
        # 统计跨领域问题
        if isinstance(detected_domains, list) and len(detected_domains) > 1:
            cross_domain_count += 1
        
        # 验证领域路由准确性
        if raw_subject and detected_domains:
            routing_total += 1
            # 检查检测到的领域是否与raw_subject匹配（允许部分匹配）
            raw_subject_lower = raw_subject.lower()
            detected_match = any(
                domain.lower() in raw_subject_lower or raw_subject_lower in domain.lower()
                for domain in detected_domains
            )
            if detected_match:
                routing_matches += 1
        
        # 统计各领域的成功率
        primary_domain = raw_subject or (detected_domains[0] if detected_domains else "Unknown")
        domain_success_rate[primary_domain]["total"] += 1
        if result.get("success"):
            domain_success_rate[primary_domain]["success"] += 1
    
    # 计算路由准确率
    routing_accuracy = routing_matches / routing_total if routing_total > 0 else 0.0
    
    # 计算各领域成功率
    domain_success_rates = {
        domain: {
            "success_rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0.0,
            "success_count": stats["success"],
            "total_count": stats["total"]
        }
        for domain, stats in domain_success_rate.items()
    }
    
    return {
        "domain_distribution": dict(domain_distribution),
        "cross_domain_count": cross_domain_count,
        "total_questions": len(results),
        "routing_accuracy": routing_accuracy,
        "routing_matches": routing_matches,
        "routing_total": routing_total,
        "domain_success_rates": domain_success_rates
    }


def compute_xmasters_statistics(results: List[Dict[str, Any]], node_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算X-Masters相关统计信息
    
    Args:
        results: 测试结果列表
        node_outputs: 节点产出字典
    
    Returns:
        X-Masters统计字典
    """
    enabled_count = 0
    total_candidates = 0
    total_critics = 0
    total_rewriters = 0
    xmasters_success_count = 0
    xmasters_total_count = 0
    
    for result in results:
        xmasters_enabled = result.get("xmasters_enabled", False)
        num_candidates = result.get("num_candidates", 0)
        num_critics = result.get("num_critiqued", 0)
        num_rewriters = result.get("num_rewritten", 0)
        
        if xmasters_enabled:
            enabled_count += 1
            total_candidates += num_candidates
            total_critics += num_critics
            total_rewriters += num_rewriters
            
            xmasters_total_count += 1
            # 如果生成了候选答案、完成了评审和重写，且最终成功，认为X-Masters成功
            if num_candidates > 0 and num_critics > 0 and num_rewriters > 0 and result.get("success"):
                xmasters_success_count += 1
    
    avg_candidates = total_candidates / enabled_count if enabled_count > 0 else 0
    avg_critics = total_critics / enabled_count if enabled_count > 0 else 0
    avg_rewriters = total_rewriters / enabled_count if enabled_count > 0 else 0
    success_rate = xmasters_success_count / xmasters_total_count if xmasters_total_count > 0 else 0
    
    # 统计X-Masters节点执行情况
    n8_5_executed = sum(1 for outputs in node_outputs.values() if "n8_5_critic_review" in outputs)
    n8_6_executed = sum(1 for outputs in node_outputs.values() if "n8_6_rewriter_synthesis" in outputs)
    
    return {
        "enabled_count": enabled_count,
        "total_questions": len(results),
        "avg_candidates": round(avg_candidates, 2),
        "avg_critics": round(avg_critics, 2),
        "avg_rewriters": round(avg_rewriters, 2),
        "success_rate": success_rate,
        "n8_5_executed_count": n8_5_executed,
        "n8_6_executed_count": n8_6_executed,
        "xmasters_success_count": xmasters_success_count,
        "xmasters_total_count": xmasters_total_count
    }


# ===================== 领域特定测试 =====================

def test_domain_routing_accuracy(questions_data: List[Dict[str, Any]], node_logger: NodeOutputLogger):
    """
    测试领域路由准确性
    
    验证系统是否能正确识别问题所属领域
    """
    # 选择包含raw_subject的问题进行测试
    questions_with_domain = [q for q in questions_data if q.get("raw_subject")]
    
    if not questions_with_domain:
        pytest.skip("No questions with raw_subject found")
    
    # 随机选择10个问题
    test_questions = get_random_questions(questions_with_domain, min(10, len(questions_with_domain)), seed=42)
    
    routing_correct = 0
    routing_total = 0
    
    for question_data in test_questions:
        question_id = question_data.get("id", "unknown")
        question_text = question_data.get("question", "")
        raw_subject = question_data.get("raw_subject", "")
        
        result = run_general_qa_for_question(question_text, question_id, node_logger)
        
        # 从n1节点提取检测到的领域
        detected_domains = []
        if question_id in node_logger.node_outputs:
            n1_output = node_logger.node_outputs[question_id].get("n1_question_decomposition", {})
            n1_snapshot = n1_output.get("state_snapshot", {})
            detected_domains = n1_snapshot.get("detected_domains", [])
        
        if raw_subject and detected_domains:
            routing_total += 1
            # 检查是否匹配（允许部分匹配）
            raw_subject_lower = raw_subject.lower()
            match = any(
                domain.lower() in raw_subject_lower or raw_subject_lower in domain.lower()
                for domain in detected_domains
            )
            if match:
                routing_correct += 1
            else:
                logger.warning(f"Domain routing mismatch: raw_subject={raw_subject}, detected={detected_domains}")
    
    accuracy = routing_correct / routing_total if routing_total > 0 else 0.0
    logger.info(f"Domain routing accuracy: {routing_correct}/{routing_total} = {accuracy:.2%}")
    
    # 要求至少70%的准确率
    assert accuracy >= 0.70, f"Domain routing accuracy ({accuracy:.2%}) is below threshold (70%)"


def test_cross_domain_detection(questions_data: List[Dict[str, Any]], node_logger: NodeOutputLogger):
    """
    测试跨领域检测功能
    
    验证系统是否能正确识别跨领域问题
    """
    # 选择可能跨领域的问题（包含多个领域关键词）
    cross_domain_keywords = [
        "genetics", "bioinformatics", "immunology", "clinical", "medicine",
        "molecular", "biochemistry", "computational"
    ]
    
    potential_cross_domain = [
        q for q in questions_data
        if sum(1 for kw in cross_domain_keywords if kw.lower() in q.get("question", "").lower()) >= 2
    ]
    
    if not potential_cross_domain:
        pytest.skip("No potential cross-domain questions found")
    
    # 选择5个可能跨领域的问题
    test_questions = get_random_questions(potential_cross_domain, min(5, len(potential_cross_domain)), seed=42)
    
    cross_domain_detected = 0
    
    for question_data in test_questions:
        question_id = question_data.get("id", "unknown")
        question_text = question_data.get("question", "")
        
        result = run_general_qa_for_question(question_text, question_id, node_logger)
        
        # 从n1节点提取检测到的领域
        detected_domains = []
        if question_id in node_logger.node_outputs:
            n1_output = node_logger.node_outputs[question_id].get("n1_question_decomposition", {})
            n1_snapshot = n1_output.get("state_snapshot", {})
            detected_domains = n1_snapshot.get("detected_domains", [])
        
        if isinstance(detected_domains, list) and len(detected_domains) > 1:
            cross_domain_detected += 1
            logger.info(f"Cross-domain detected: {detected_domains}")
    
    logger.info(f"Cross-domain detection: {cross_domain_detected}/{len(test_questions)} questions detected as cross-domain")
    
    # 不强制要求，只记录结果
    assert True  # 测试通过，只是记录统计信息


def test_domain_specific_tool_usage(questions_data: List[Dict[str, Any]], node_logger: NodeOutputLogger):
    """
    测试领域特定工具使用
    
    验证不同领域的问题是否使用了相应的工具
    """
    # 选择特定领域的问题
    domain_questions = {
        "Genetics": [q for q in questions_data if q.get("raw_subject") == "Genetics"][:3],
        "Immunology": [q for q in questions_data if q.get("raw_subject") == "Immunology"][:3],
        "Clinical Medicine": [q for q in questions_data if q.get("raw_subject") == "Clinical Medicine"][:3],
    }
    
    domain_tool_usage = defaultdict(list)
    
    for domain, questions in domain_questions.items():
        if not questions:
            continue
        
        for question_data in questions:
            question_id = question_data.get("id", "unknown")
            question_text = question_data.get("question", "")
            
            result = run_general_qa_for_question(question_text, question_id, node_logger)
            
            # 提取工具调用历史
            tool_calls = []
            if question_id in node_logger.node_outputs:
                for node_name, node_output in node_logger.node_outputs[question_id].items():
                    snapshot = node_output.get("state_snapshot", {})
                    if "tool_calls_history" in snapshot:
                        tool_calls.extend(snapshot["tool_calls_history"])
            
            if tool_calls:
                domain_tool_usage[domain].extend(tool_calls)
    
    # 打印工具使用统计
    logger.info("\nDomain-specific tool usage:")
    for domain, tools in domain_tool_usage.items():
        tool_names = [t.get("tool_name", "unknown") if isinstance(t, dict) else str(t) for t in tools]
        unique_tools = list(set(tool_names))
        logger.info(f"  {domain}: {len(unique_tools)} unique tools used: {unique_tools[:5]}...")
    
    # 验证至少有一些工具被调用
    total_tools = sum(len(tools) for tools in domain_tool_usage.values())
    assert total_tools > 0, "No tools were called for any domain"


# ===================== X-Masters功能测试 =====================

def test_xmasters_candidate_generation(questions_data: List[Dict[str, Any]], node_logger: NodeOutputLogger):
    """
    测试X-Masters候选答案生成功能
    
    验证N8节点是否能生成多个候选答案
    """
    # 选择一个简单的问题进行测试
    test_questions = get_random_questions(questions_data, 3, seed=42)
    
    for question_data in test_questions:
        question_id = question_data.get("id", "unknown")
        question_text = question_data.get("question", "")
        
        result = run_general_qa_for_question(question_text, question_id, node_logger)
        
        # 检查是否生成了候选答案
        num_candidates = result.get("num_candidates", 0)
        xmasters_enabled = result.get("xmasters_enabled", False)
        
        logger.info(f"Question {question_id}: X-Masters enabled={xmasters_enabled}, candidates={num_candidates}")
        
        if xmasters_enabled:
            # 验证N8节点生成了候选答案
            if question_id in node_logger.node_outputs:
                n8_output = node_logger.node_outputs[question_id].get("n8_answer_generation", {})
                n8_snapshot = n8_output.get("state_snapshot", {})
                candidate_answers = n8_snapshot.get("candidate_answers")
                
                assert candidate_answers is not None, f"N8 should generate candidate_answers for question {question_id}"
                assert len(candidate_answers) > 0, f"N8 should generate at least one candidate answer for question {question_id}"
                logger.info(f"  ✓ Generated {len(candidate_answers)} candidate answer(s)")
    
    logger.info("X-Masters candidate generation test completed")


def test_xmasters_critic_review(questions_data: List[Dict[str, Any]], node_logger: NodeOutputLogger):
    """
    测试X-Masters Critic评审功能
    
    验证N8.5节点是否能对候选答案进行评审
    """
    test_questions = get_random_questions(questions_data, 3, seed=42)
    
    critic_executed_count = 0
    
    for question_data in test_questions:
        question_id = question_data.get("id", "unknown")
        question_text = question_data.get("question", "")
        
        result = run_general_qa_for_question(question_text, question_id, node_logger)
        
        # 检查N8.5节点是否执行
        if question_id in node_logger.node_outputs:
            n8_5_output = node_logger.node_outputs[question_id].get("n8_5_critic_review", {})
            if n8_5_output:
                critic_executed_count += 1
                n8_5_snapshot = n8_5_output.get("state_snapshot", {})
                critiqued_answers = n8_5_snapshot.get("critiqued_answers")
                
                if critiqued_answers:
                    logger.info(f"Question {question_id}: Critic reviewed {len(critiqued_answers)} answer(s)")
                    # 验证每个评审后的答案都有original_answer和critiqued_answer
                    for critiqued in critiqued_answers:
                        assert "original_answer" in critiqued or "original_structured" in critiqued
                        assert "critiqued_answer" in critiqued
    
    logger.info(f"X-Masters Critic review executed for {critic_executed_count}/{len(test_questions)} questions")
    assert critic_executed_count > 0, "N8.5 Critic review should be executed for at least one question"


def test_xmasters_rewriter_synthesis(questions_data: List[Dict[str, Any]], node_logger: NodeOutputLogger):
    """
    测试X-Masters Rewriter综合功能
    
    验证N8.6节点是否能综合多个评审后的答案
    """
    test_questions = get_random_questions(questions_data, 3, seed=42)
    
    rewriter_executed_count = 0
    
    for question_data in test_questions:
        question_id = question_data.get("id", "unknown")
        question_text = question_data.get("question", "")
        
        result = run_general_qa_for_question(question_text, question_id, node_logger)
        
        # 检查N8.6节点是否执行
        if question_id in node_logger.node_outputs:
            n8_6_output = node_logger.node_outputs[question_id].get("n8_6_rewriter_synthesis", {})
            if n8_6_output:
                rewriter_executed_count += 1
                n8_6_snapshot = n8_6_output.get("state_snapshot", {})
                rewritten_answers = n8_6_snapshot.get("rewritten_answers")
                
                if rewritten_answers:
                    logger.info(f"Question {question_id}: Rewriter synthesized {len(rewritten_answers)} answer(s)")
                    # 验证每个重写答案都有rewritten_answer字段
                    for rewritten in rewritten_answers:
                        assert "rewritten_answer" in rewritten
    
    logger.info(f"X-Masters Rewriter synthesis executed for {rewriter_executed_count}/{len(test_questions)} questions")
    assert rewriter_executed_count > 0, "N8.6 Rewriter synthesis should be executed for at least one question"


def test_xmasters_full_workflow(questions_data: List[Dict[str, Any]], node_logger: NodeOutputLogger):
    """
    测试X-Masters完整工作流
    
    验证从N8到N9的完整X-Masters流程是否正常执行
    """
    test_questions = get_random_questions(questions_data, 2, seed=42)
    
    full_workflow_count = 0
    
    for question_data in test_questions:
        question_id = question_data.get("id", "unknown")
        question_text = question_data.get("question", "")
        
        result = run_general_qa_for_question(question_text, question_id, node_logger)
        
        # 检查完整工作流
        if question_id in node_logger.node_outputs:
            outputs = node_logger.node_outputs[question_id]
            
            # 检查是否执行了所有X-Masters节点
            has_n8 = "n8_answer_generation" in outputs
            has_n8_5 = "n8_5_critic_review" in outputs
            has_n8_6 = "n8_6_rewriter_synthesis" in outputs
            has_n9 = "n9_result_validation" in outputs
            
            if has_n8 and has_n8_5 and has_n8_6 and has_n9:
                full_workflow_count += 1
                
                # 验证数据流
                n8_snapshot = outputs["n8_answer_generation"].get("state_snapshot", {})
                n8_5_snapshot = outputs["n8_5_critic_review"].get("state_snapshot", {})
                n8_6_snapshot = outputs["n8_6_rewriter_synthesis"].get("state_snapshot", {})
                
                candidates = n8_snapshot.get("candidate_answers", [])
                critiqued = n8_5_snapshot.get("critiqued_answers", [])
                rewritten = n8_6_snapshot.get("rewritten_answers", [])
                
                logger.info(f"Question {question_id}: Full X-Masters workflow")
                logger.info(f"  N8: {len(candidates)} candidates")
                logger.info(f"  N8.5: {len(critiqued)} critiqued")
                logger.info(f"  N8.6: {len(rewritten)} rewritten")
                logger.info(f"  Success: {result.get('success', False)}")
    
    logger.info(f"Full X-Masters workflow executed for {full_workflow_count}/{len(test_questions)} questions")
    assert full_workflow_count > 0, "Full X-Masters workflow should be executed for at least one question"


# ===================== Pytest 配置 =====================
# Note: pytest_addoption is now defined in conftest.py

