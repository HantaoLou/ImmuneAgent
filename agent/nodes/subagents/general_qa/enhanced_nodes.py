"""
Enhanced nodes for General QA subgraph

包含增强的节点实现，用于处理 HLE 复杂问题
"""

from typing import Dict, List, Any, Optional
import json
import re

# 导入状态和增强模块
from agent.nodes.subagents.general_qa.state import GeneralQAState

try:
    from agent.nodes.subagents.general_qa.enhancements import (
        SelfConsistencyEngine,
        ChainOfThoughtParser,
        CalculationVerifier,
        IterativeKnowledgeRetriever,
        MetaCognitiveMonitor,
        ExceptionDiagnostician,
        ToolIntentAnalyzer,
        create_enhanced_prompt,
        extract_numerical_result
    )
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False


# ===================== 配置常量 =====================

# Self-Consistency 配置
SC_NUM_PATHS = 3
SC_TEMPERATURES = [0.0, 0.3, 0.7]
SC_MIN_CONSENSUS_RATIO = 0.5

# 计算验证配置
CALC_VERIFICATION_TOLERANCE = 1e-6

# 迭代检索配置
MAX_RETRIEVAL_ITERATIONS = 3

# 元认知配置
METACOG_CONFIDENCE_THRESHOLD = 0.5


# ===================== 增强的 N0 节点: 工具意图识别 =====================

def enhance_n0_with_tool_intent(state: GeneralQAState) -> GeneralQAState:
    """
    增强 N0: 添加智能工具意图识别
    
    分析问题类型、领域和关键词，自动识别需要和推荐使用的工具
    """
    if not ENHANCEMENTS_AVAILABLE:
        return state
    
    analyzer = ToolIntentAnalyzer()
    
    # 分析工具需求
    requirements = analyzer.analyze_requirements(
        user_input=state.user_input,
        core_domains=state.core_domains or [],
        question_type=state.question_type_label or "",
        core_keywords=state.core_keywords or []
    )
    
    # 更新状态
    state.required_tools = requirements["required"]
    state.recommended_tools = requirements["recommended"]
    
    # 更新 tool_intent
    if not state.tool_intent:
        state.tool_intent = {}
    
    for tool in requirements["required"]:
        state.tool_intent[tool] = "REQUIRED"
    for tool in requirements["recommended"]:
        if tool not in state.tool_intent:
            state.tool_intent[tool] = "RECOMMENDED"
    
    # 打印信息
    if requirements["required"]:
        print(f"  [TOOL] Required tools identified: {requirements['required']}")
    if requirements["recommended"]:
        print(f"  💡 Recommended tools: {requirements['recommended']}")
    
    return state


# ===================== 增强的 N3 节点: 迭代式知识检索 =====================

def enhance_n3_with_iterative_retrieval(
    state: GeneralQAState,
    original_retrieval_func,
    max_iterations: int = MAX_RETRIEVAL_ITERATIONS
) -> GeneralQAState:
    """
    增强 N3: 添加迭代式知识检索
    
    如果第一次检索的知识不足以回答问题，自动进行多轮检索
    """
    if not ENHANCEMENTS_AVAILABLE:
        return state
    
    retriever = IterativeKnowledgeRetriever(max_iterations=max_iterations)
    
    # 记录迭代次数
    current_iteration = state.retrieval_iterations or 0
    
    # 执行原始检索
    state = original_retrieval_func(state)
    
    if state.error_message:
        return state
    
    # 检查知识是否足够
    if not retriever.is_knowledge_sufficient(state, state.domain_knowledge_map or {}):
        current_iteration += 1
        state.retrieval_iterations = current_iteration
        
        # 识别知识缺口
        gaps = retriever._identify_knowledge_gaps(
            state.cleaned_text or state.user_input,
            state.domain_knowledge_map or {}
        )
        state.knowledge_gaps_identified = gaps
        
        # 生成追问
        follow_ups = retriever.generate_follow_up_questions(state, state.domain_knowledge_map or {})
        state.follow_up_questions = follow_ups
        
        if gaps:
            print(f"  [RUN] Iteration {current_iteration}: Knowledge gaps identified: {gaps[:3]}")
        if follow_ups:
            print(f"  ❓ Follow-up questions generated: {follow_ups}")
        
        # 如果还有迭代次数，可以继续检索（这里只是标记，实际重试由路由决定）
        if current_iteration < max_iterations:
            # 扩展检索子问题
            if not state.retrieval_sub_questions:
                state.retrieval_sub_questions = []
            state.retrieval_sub_questions.extend(gaps[:2])
    
    return state


# ===================== 增强的 N4 节点: 计算验证 =====================

def enhance_n4_with_verification(
    state: GeneralQAState,
    llm_response: str
) -> GeneralQAState:
    """
    增强 N4: 添加计算结果验证
    
    对于数值计算问题，使用多种方法交叉验证结果
    """
    if not ENHANCEMENTS_AVAILABLE:
        return state
    
    if state.calculation_type_label != "Numerical":
        return state
    
    verifier = CalculationVerifier()
    
    # 执行验证
    result = verifier.verify(
        matched_formula=state.matched_formula or {},
        key_parameters=state.key_parameters or {},
        llm_result=llm_response
    )
    
    state.calculation_verification = result.to_dict()
    
    if result.all_match:
        print(f"  [SUCCESS] Calculation verification passed")
        state.needs_calculation_retry = False
    elif result.discrepancy:
        print(f"  [WARN]️ Calculation verification discrepancy detected:")
        print(f"    - Symbolic result: {result.symbolic_result}")
        print(f"    - Numerical result: {result.numerical_result}")
        print(f"    - Difference: {result.discrepancy.get('difference')}")
        state.needs_calculation_retry = True
    else:
        # 无法验证，不标记为需要重试
        print(f"  ℹ️ Calculation verification skipped (insufficient data)")
    
    return state


# ===================== 增强的 N7 节点: Self-Consistency + CoT =====================

def enhance_n7_with_self_consistency(
    state: GeneralQAState,
    llm,
    prompt: str,
    call_llm_func
) -> GeneralQAState:
    """
    增强 N7: 添加 Self-Consistency 多路径推理
    
    使用不同温度生成多个推理路径，然后投票选择最一致的答案
    """
    if not ENHANCEMENTS_AVAILABLE:
        return state
    
    engine = SelfConsistencyEngine(
        num_paths=SC_NUM_PATHS,
        temperatures=SC_TEMPERATURES
    )
    
    paths = []
    original_temp = getattr(llm, 'temperature', 0.3)
    
    print(f"  [RUN] Running Self-Consistency with {SC_NUM_PATHS} paths...")
    
    for i, temp in enumerate(SC_TEMPERATURES):
        print(f"    - Path {i+1}/{SC_NUM_PATHS} (temperature={temp})")
        
        # 设置温度
        if hasattr(llm, 'temperature'):
            llm.temperature = temp
        
        # 调用 LLM
        response = call_llm_func(llm, prompt, state=state, node_name=f"n7_path_{i}")
        
        if response:
            result = None
            try:
                # 尝试解析 JSON
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    result = json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
            
            if result:
                conclusion = result.get("core_conclusion", "")
                inference_path = result.get("closed_inference_path", [])
            else:
                conclusion = response[:200]  # 取前200字符作为结论
                inference_path = []
            
            from agent.nodes.subagents.general_qa.enhancements import ReasoningPath, InferenceStep
            
            path = ReasoningPath(
                path_id=i,
                temperature=temp,
                steps=[],  # 简化，不解析步骤
                final_conclusion=conclusion
            )
            paths.append(path)
    
    # 恢复原始温度
    if hasattr(llm, 'temperature'):
        llm.temperature = original_temp
    
    # 聚合结果
    sc_result = engine.aggregate_results(paths, state.question_options)
    
    # 更新状态
    state.inference_paths = [p.to_dict() for p in paths]
    state.self_consistency_result = sc_result.to_dict()
    
    # 打印结果
    print(f"  [STAT] Self-Consistency Result:")
    print(f"    - Consensus answer: {sc_result.consensus_answer[:50]}...")
    print(f"    - Consensus ratio: {sc_result.consensus_ratio:.2%}")
    print(f"    - Confidence level: {sc_result.confidence_level}")
    
    # 如果一致性低，标记需要额外验证
    if sc_result.confidence_level == "low":
        print(f"  [WARN]️ Low consensus detected, marking for additional verification")
        # 不覆盖核心结论，但记录自一致性结果供后续节点使用
    
    return state


def enhance_n7_with_cot(
    state: GeneralQAState
) -> GeneralQAState:
    """
    增强 N7: 解析和验证 Chain-of-Thought 推理链
    
    将推理路径解析为结构化的推理步骤，并验证连贯性
    """
    if not ENHANCEMENTS_AVAILABLE:
        return state
    
    if not state.closed_inference_path:
        return state
    
    parser = ChainOfThoughtParser()
    
    # 解析推理链
    steps = parser.parse_inference_path(state.closed_inference_path)
    
    # 转换为字典格式
    state.structured_inference_steps = [s.to_dict() for s in steps]
    
    # 计算推理深度
    state.reasoning_depth = len(steps)
    
    # 验证连贯性
    is_coherent, issues = parser.validate_chain_coherence(steps)
    state.inference_chain_coherent = is_coherent
    
    if not is_coherent:
        print(f"  [WARN]️ Inference chain coherence issues detected:")
        for issue in issues[:3]:
            print(f"    - {issue}")
    else:
        print(f"  [SUCCESS] Inference chain is coherent ({len(steps)} steps, depth={state.reasoning_depth})")
    
    return state


# ===================== 增强的 N9 节点前: 元认知监控 =====================

def enhance_with_metacognitive_monitoring(
    state: GeneralQAState
) -> GeneralQAState:
    """
    增强: 在关键节点之间添加元认知监控
    
    监控推理质量，检测是否需要回溯
    """
    if not ENHANCEMENTS_AVAILABLE:
        return state
    
    monitor = MetaCognitiveMonitor()
    
    # 执行评估
    assessment = monitor.assess(state)
    
    # 更新状态
    state.meta_cognitive_assessment = assessment.to_dict()
    state.needs_backtracking = assessment.needs_backtracking
    
    # 打印评估结果
    print(f"  🧠 Meta-Cognitive Assessment:")
    print(f"    - Goal alignment: {'[SUCCESS]' if assessment.goal_alignment else '[ERROR]'}")
    print(f"    - Constraint coverage: {'[SUCCESS]' if assessment.constraint_coverage else '[ERROR]'}")
    print(f"    - Reasoning coherence: {'[SUCCESS]' if assessment.reasoning_coherence else '[ERROR]'}")
    print(f"    - Confidence calibration: {assessment.confidence_calibration:.2f}")
    
    if assessment.knowledge_gaps:
        print(f"    - Knowledge gaps: {assessment.knowledge_gaps[:2]}")
    
    if assessment.needs_backtracking:
        print(f"    [WARN]️ Backtracking needed: {assessment.backtracking_reason}")
    
    return state


# ===================== 增强的 N10 节点: 智能异常诊断 =====================

def enhance_n10_with_smart_diagnosis(
    state: GeneralQAState
) -> GeneralQAState:
    """
    增强 N10: 智能异常诊断和精准重试策略
    
    分析异常根因，选择最优的重试策略
    """
    if not ENHANCEMENTS_AVAILABLE:
        return state
    
    diagnostician = ExceptionDiagnostician()
    
    # 诊断根因
    root_cause = diagnostician.diagnose_root_cause(state)
    state.root_cause_diagnosis = root_cause
    
    # 获取重试策略
    strategy = diagnostician.get_retry_strategy(root_cause)
    state.retry_strategy = strategy.to_dict()
    
    print(f"  🔍 Exception Diagnosis:")
    print(f"    - Root cause: {root_cause}")
    print(f"    - Retry target: {strategy.target_node}")
    print(f"    - Retry action: {strategy.action}")
    print(f"    - Reason: {strategy.reason}")
    
    # 根据策略更新状态
    if strategy.target_node:
        state.retry_target_node = strategy.target_node
        
        # 如果有特殊参数，应用它们
        if strategy.params:
            if strategy.action == "expand_search":
                # 标记需要扩展搜索
                if not state.tool_intent:
                    state.tool_intent = {}
                state.tool_intent["use_paperqa"] = "YES"
                state.tool_intent["use_web_search"] = "YES"
            
            elif strategy.action == "increase_sampling":
                # 标记需要增加采样
                state.num_candidates = strategy.params.get("num_paths", 5)
    
    return state


# ===================== 增强的 Prompt 生成 =====================

def get_enhanced_inference_prompt(
    base_prompt: str,
    state: GeneralQAState,
    enable_cot: bool = True,
    enable_self_consistency_hint: bool = True
) -> str:
    """
    获取增强的推理 prompt
    
    添加 CoT 和 Self-Consistency 指令
    """
    if not ENHANCEMENTS_AVAILABLE:
        return base_prompt
    
    return create_enhanced_prompt(
        base_prompt=base_prompt,
        state=state,
        enable_cot=enable_cot,
        enable_self_consistency=enable_self_consistency_hint
    )


# ===================== 综合增强函数 =====================

def apply_all_enhancements_to_n7(
    state: GeneralQAState,
    llm,
    base_prompt: str,
    call_llm_func,
    enable_self_consistency: bool = True,
    enable_cot: bool = True
) -> GeneralQAState:
    """
    对 N7 节点应用所有增强
    
    1. 增强 prompt (CoT)
    2. Self-Consistency 多路径推理
    3. 推理链解析和验证
    4. 元认知监控
    """
    # 1. 增强 prompt
    if enable_cot:
        enhanced_prompt = get_enhanced_inference_prompt(
            base_prompt,
            state,
            enable_cot=True,
            enable_self_consistency_hint=enable_self_consistency
        )
    else:
        enhanced_prompt = base_prompt
    
    # 2. Self-Consistency (可选)
    if enable_self_consistency and ENHANCEMENTS_AVAILABLE:
        state = enhance_n7_with_self_consistency(
            state,
            llm,
            enhanced_prompt,
            call_llm_func
        )
        
        # 检查自一致性结果
        if state.self_consistency_result:
            sc = state.self_consistency_result
            # 如果一致性高，使用共识答案
            if sc.get("confidence_level") in ["high", "medium"]:
                # 不覆盖 core_conclusion，让原始逻辑处理
                # 但可以记录自一致性结果供后续参考
                pass
    
    # 3. CoT 解析
    if enable_cot:
        state = enhance_n7_with_cot(state)
    
    # 4. 元认知监控
    state = enhance_with_metacognitive_monitoring(state)
    
    return state


def apply_all_enhancements_to_n10(
    state: GeneralQAState
) -> GeneralQAState:
    """
    对 N10 节点应用所有增强
    
    1. 智能异常诊断
    2. 精准重试策略
    """
    if not ENHANCEMENTS_AVAILABLE:
        return state
    
    # 1. 智能诊断
    state = enhance_n10_with_smart_diagnosis(state)
    
    return state




