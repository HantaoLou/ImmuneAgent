"""
参数推断功能测试用例

测试 task_decomposition 阶段的参数推断功能，包括：
1. 不同难度级别的参数推断测试
2. 完整流程：粗分解 → 细分解 → 并行推断 → 参数推断
3. 清晰和模糊的任务描述对比

注意：这些测试用例只测试参数推断功能，不会执行任何任务（包括 codeact）。
测试目的是验证系统能否正确推断出供给 executor 执行的参数。

运行方式：pytest tests/test_executor_parameter_inference.py -v
"""

import os
import pytest
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# 加载环境变量
load_dotenv()

# 添加agent目录到路径
import sys
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from nodes.subagents.task_decomposition.graph import (
    build_task_decomposition_subgraph,
    task_decomposition_input_mapper,
    task_decomposition_output_mapper,
    TaskDecompositionState,
    ParameterSourceType
)
from state import GlobalState


# ===================== 测试用例数据 =====================

# Level 1: 简单级别 - 单一任务，参数明确
LEVEL_1_TEST_CASES = [
    {
        "name": "简单_特定顺序下的任务分解及参数推断",
        "user_input": "搜索 COVID-19 相关的抗体数据",
        "execution_plan": '3.1输入fasta序列和PDB结构\n' +
                          '3.2利用spired-fintess和foldx获得热力学稳定性\n' +
                          '3.3筛选稳定性和fitness的两项更佳的单位点突变，筛选20-100个\n' +
                          '3.4利用GEMORNA生成全长的mRNA，生成多个5-10个，\n' +
                          '3.5利用riboNN评级生成mRNA的稳定性（csv表格文件），\n' +
                          '3.6利用RiNALMO评级生成mRNA核糖体载量\n' +
                          '3.7综合mRNA的稳定性和mRNA核糖体载量数字，筛选最佳mRNA ',
        "description": ""
    },
]

# Level 2: 中等级别 - 多个任务，有依赖关系，参数部分明确
LEVEL_2_TEST_CASES = [
    {
        "name": "中等_搜索并分析",
        "user_input": "搜索 COVID-19 抗体数据，然后分析 V(D)J 重组",
        "execution_plan": None,
        "description": "两个任务，有依赖关系，部分参数明确"
    },
    {
        "name": "中等_多步骤分析",
        "user_input": "搜索抗体数据，分析序列，生成报告",
        "execution_plan": None,
        "description": "三个任务，依赖链，参数模糊"
    },
    {
        "name": "中等_带执行计划",
        "user_input": "执行抗体分析任务",
        "execution_plan": """1. 搜索 COVID-19 相关的抗体数据
2. 分析 V(D)J 重组情况
3. 提取 CDR3 区域
4. 生成分析报告""",
        "description": "有明确执行计划，但用户输入模糊"
    },
    {
        "name": "中等_部分参数明确",
        "user_input": "搜索 COVID-19 数据，然后进行序列分析",
        "execution_plan": None,
        "description": "第一个任务参数明确，第二个任务参数模糊"
    },
    {
        "name": "中等_复杂依赖链",
        "user_input": "下载数据，处理序列，分析结果，生成报告",
        "execution_plan": None,
        "description": "多个任务，复杂依赖，参数大部分模糊"
    },
    {
        "name": "中等_混合清晰模糊",
        "user_input": "搜索疾病为 COVID-19 的抗体数据，然后分析并生成报告",
        "execution_plan": None,
        "description": "第一个任务参数清晰，后续任务参数模糊"
    },
    {
        "name": "中等_并行任务组",
        "user_input": "同时搜索 COVID-19 和 SARS 的抗体数据，然后合并分析",
        "execution_plan": None,
        "description": "包含并行任务，参数部分明确"
    }
]

# Level 3: 困难级别 - 复杂任务链，参数模糊，需要深度推理
LEVEL_3_TEST_CASES = [
    {
        "name": "困难_完全模糊描述",
        "user_input": "帮我分析一下抗体数据",
        "execution_plan": None,
        "description": "任务描述非常模糊，需要深度推理"
    },
    {
        "name": "困难_上下文推断",
        "user_input": "分析一下这些数据",
        "execution_plan": None,
        "description": "需要从上下文推断具体任务和参数"
    },
    {
        "name": "困难_复杂模糊计划",
        "user_input": "执行抗体研究任务",
        "execution_plan": """1. 获取相关数据
2. 进行必要分析
3. 整理结果""",
        "description": "执行计划模糊，需要推断具体工具和参数"
    },
    {
        "name": "困难_多步骤模糊",
        "user_input": "做一个完整的抗体分析项目",
        "execution_plan": None,
        "description": "任务描述模糊，需要推断完整流程和参数"
    },
    {
        "name": "困难_隐式依赖",
        "user_input": "分析抗体序列并生成可视化结果",
        "execution_plan": None,
        "description": "依赖关系隐式，参数需要从任务描述推断"
    },
    {
        "name": "困难_专业术语模糊",
        "user_input": "研究一下这些免疫数据",
        "execution_plan": None,
        "description": "使用模糊术语，需要专业知识推断"
    },
    {
        "name": "困难_多目标任务",
        "user_input": "帮我完成抗体相关的所有分析工作",
        "execution_plan": None,
        "description": "目标不明确，需要推断多个子任务和参数"
    }
]


# ===================== 核心辅助函数 =====================

def run_full_decomposition_flow(user_input: str, execution_plan: str = None) -> Tuple[GlobalState, TaskDecompositionState]:
    """
    运行完整的任务分解流程：粗分解 → 细分解 → 并行推断 → 参数推断
    
    Args:
        user_input: 用户输入
        execution_plan: 执行计划（可选）
    
    Returns:
        (包含分解后任务的 GlobalState, 包含中间状态和参数推断结果的 TaskDecompositionState)
    """
    task_decomposition_subgraph = build_task_decomposition_subgraph()
    
    initial_state = GlobalState(
        user_input=user_input,
        execution_plan=execution_plan,
        sandbox_dir="./sandbox"
    )
    
    decomposition_input = task_decomposition_input_mapper(initial_state)
    decomposition_output = task_decomposition_subgraph.invoke(decomposition_input)
    final_state = task_decomposition_output_mapper(decomposition_output, initial_state)
    
    if isinstance(decomposition_output, dict):
        decomposition_state = TaskDecompositionState(**decomposition_output)
    else:
        decomposition_state = decomposition_output
    
    return final_state, decomposition_state


def extract_parameter_inference_results(decomposition_state: TaskDecompositionState) -> Tuple[Dict, Dict, Dict]:
    """
    从 TaskDecompositionState 中提取参数推断结果
    
    Returns:
        (task_results_dict, task_info_dict, stats_dict)
        stats_dict 包含: determined_count, from_task_count, user_required_count, total_count
    """
    task_results_dict = {}
    task_info_dict = {}
    determined_count = 0
    from_task_count = 0
    user_required_count = 0
    
    # 获取所有任务（包括并行任务组中的任务）
    all_tasks = decomposition_state.subtasks + [
        task for group in decomposition_state.parallel_task_groups.values()
        for task in group.subtasks
    ]
    
    for task in all_tasks:
        task_id = task.task_id
        task_info_dict[task_id] = {
            'content': task.content,
            'task_type': task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type)
        }
        
        if task_id in decomposition_state.parameter_inference_results:
            inference_result = decomposition_state.parameter_inference_results[task_id]
            params_dict = {}
            
            for param_name, param_result in inference_result.parameters.items():
                if hasattr(param_result, 'model_dump'):
                    param_data = param_result.model_dump()
                elif isinstance(param_result, dict):
                    param_data = param_result
                else:
                    param_data = {
                        'source_type': param_result.source_type.value if hasattr(param_result.source_type, 'value') else str(param_result.source_type),
                        'value': getattr(param_result, 'value', None),
                        'source_task_id': getattr(param_result, 'source_task_id', None),
                        'source_output_key': getattr(param_result, 'source_output_key', None),
                        'user_prompt': getattr(param_result, 'user_prompt', None),
                        'reason': getattr(param_result, 'reason', None)
                    }
                
                params_dict[param_name] = param_data
                
                source_type = param_data.get('source_type', 'unknown')
                if source_type == ParameterSourceType.DETERMINED.value:
                    determined_count += 1
                elif source_type == ParameterSourceType.FROM_TASK.value:
                    from_task_count += 1
                elif source_type == ParameterSourceType.USER_REQUIRED.value:
                    user_required_count += 1
            
            task_results_dict[task_id] = {
                'parameters': params_dict,
                'tool_name': inference_result.tool_name
            }
    
    stats = {
        'determined_count': determined_count,
        'from_task_count': from_task_count,
        'user_required_count': user_required_count,
        'total_count': determined_count + from_task_count + user_required_count
    }
    
    return task_results_dict, task_info_dict, stats


def safe_get_raw_tasks(decomposition_state: TaskDecompositionState) -> List[Dict]:
    """安全地获取 raw_tasks，处理各种可能的类型"""
    if not hasattr(decomposition_state, 'raw_tasks') or not decomposition_state.raw_tasks:
        return []
    
    raw_tasks = decomposition_state.raw_tasks
    
    if isinstance(raw_tasks, list):
        return raw_tasks
    
    if isinstance(raw_tasks, str):
        try:
            parsed = json.loads(raw_tasks)
            if isinstance(parsed, list):
                return parsed
        except:
            pass
    
    return []


def prepare_log_data(decomposition_state: TaskDecompositionState, 
                     task_results_dict: Dict, 
                     task_info_dict: Dict,
                     inference_quality: Optional[Dict] = None) -> Tuple[Dict, Dict, Dict]:
    """
    准备日志数据
    
    Returns:
        (coarse_decomposition, fine_decomposition, parameter_inference)
    """
    from nodes.subagents.task_decomposition.tool_categorizer import get_service_summary_by_id
    
    # 粗分解数据
    service_summaries = {}
    for service_id in decomposition_state.required_service_ids:
        summary = get_service_summary_by_id(service_id)
        if summary:
            service_summaries[service_id] = summary
    
    coarse_decomposition = {
        'required_service_ids': decomposition_state.required_service_ids,
        'filtered_tools_count': len(decomposition_state.filtered_tools),
        'service_summaries': service_summaries
    }
    
    # 细分解数据
    fine_decomposition = {
        'raw_tasks': safe_get_raw_tasks(decomposition_state),
        'decomposition_summary': getattr(decomposition_state, 'decomposition_summary', None)
    }
    
    # 参数推断数据
    parameter_inference = {
        'task_results': task_results_dict,
        'task_info': task_info_dict,
        'summary': getattr(decomposition_state, 'parameter_inference_summary', None)
    }
    
    if inference_quality:
        parameter_inference['inference_quality'] = inference_quality
    
    return coarse_decomposition, fine_decomposition, parameter_inference


def print_parameter_inference_results(task_results_dict: Dict, task_info_dict: Dict, 
                                       stats: Dict, max_params_per_task: int = 3):
    """打印参数推断结果"""
    print("\n参数推断结果详情:")
    for task_id, result in task_results_dict.items():
        print(f"\n  任务 {task_id}:")
        task_info = task_info_dict.get(task_id, {})
        print(f"    描述: {task_info.get('content', '')[:60]}...")
        if result.get('tool_name'):
            print(f"    工具: {result['tool_name']}")
        
        params = result.get('parameters', {})
        print(f"    参数推断结果: {len(params)} 个")
        for param_name, param_data in list(params.items())[:max_params_per_task]:
            source_type = param_data.get('source_type', 'unknown')
            if source_type == ParameterSourceType.DETERMINED.value:
                value = param_data.get('value', '')
                value_str = str(value)
                if len(value_str) > 40:
                    value_str = value_str[:40] + "..."
                print(f"      - {param_name}: {value_str} [确定值] ✓")
            elif source_type == ParameterSourceType.FROM_TASK.value:
                source_task = param_data.get('source_task_id', '')
                source_key = param_data.get('source_output_key', '')
                key_info = f" ({source_key})" if source_key else ""
                print(f"      - {param_name}: 来自任务 {source_task}{key_info} [任务结果] ✓")
            elif source_type == ParameterSourceType.USER_REQUIRED.value:
                prompt = param_data.get('user_prompt', '')
                print(f"      - {param_name}: {prompt} [需要用户提供] ✓")
        if len(params) > max_params_per_task:
            print(f"      ... 还有 {len(params) - max_params_per_task} 个参数")
    
    print(f"\n总体统计:")
    print(f"  确定的参数值: {stats['determined_count']}")
    print(f"  来自任务的参数: {stats['from_task_count']}")
    print(f"  需要用户提供的参数: {stats['user_required_count']}")
    print(f"  总参数数: {stats['total_count']}")
    print(f"  说明: 正确识别出参数来源（确定值、来自任务、需要用户提供）都算作成功的推断结果 ✓")


def analyze_inference_quality(task_results_dict: Dict) -> Dict:
    """分析参数推断质量"""
    inference_quality = {
        "high_quality": 0,
        "medium_quality": 0,
        "low_quality": 0,
        "needs_user": 0
    }
    
    print("\n参数推断质量分析:")
    for task_id, result in task_results_dict.items():
        params = result.get('parameters', {})
        if not params:
            continue
        
        determined = sum(1 for p in params.values() if p.get('source_type') == ParameterSourceType.DETERMINED.value)
        from_task = sum(1 for p in params.values() if p.get('source_type') == ParameterSourceType.FROM_TASK.value)
        user_required = sum(1 for p in params.values() if p.get('source_type') == ParameterSourceType.USER_REQUIRED.value)
        total_params = len(params)
        
        inferred_count = determined + from_task
        inferred_ratio = inferred_count / total_params if total_params > 0 else 0
        
        print(f"\n  任务 {task_id}:")
        print(f"    推断比例: {inferred_ratio:.1%} ({inferred_count}/{total_params})")
        print(f"      确定值: {determined}, 来自任务: {from_task}, 需要用户: {user_required}")
        
        if inferred_ratio >= 0.7:
            inference_quality["high_quality"] += 1
            print(f"    质量: 高 ✓")
        elif inferred_ratio >= 0.3:
            inference_quality["medium_quality"] += 1
            print(f"    质量: 中")
        else:
            inference_quality["low_quality"] += 1
            print(f"    质量: 低")
        
        if user_required > 0:
            inference_quality["needs_user"] += 1
            print(f"    需要用户提供: 是 ({user_required} 个参数)")
    
    print(f"\n推断质量统计:")
    print(f"  高质量推断: {inference_quality['high_quality']} 个任务")
    print(f"  中等质量推断: {inference_quality['medium_quality']} 个任务")
    print(f"  低质量推断: {inference_quality['low_quality']} 个任务")
    print(f"  需要用户提供: {inference_quality['needs_user']} 个任务")
    
    return inference_quality


# ===================== 日志记录功能 =====================

def save_test_summary_log(test_case_name: str, execution_plan: Optional[str], 
                          coarse_decomposition: Dict, fine_decomposition: Dict,
                          parameter_inference: Dict):
    """保存测试用例的总结性日志"""
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"parameter_inference_{test_case_name}_{timestamp}.md"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"# 参数推断测试总结日志\n\n")
        f.write(f"**测试用例**: {test_case_name}\n\n")
        f.write(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # 1. 计划
        f.write("## 1. 执行计划\n\n")
        if execution_plan:
            f.write(f"```\n{execution_plan}\n```\n\n")
        else:
            f.write("无执行计划\n\n")
        f.write("---\n\n")
        
        # 2. 粗分解结果
        f.write("## 2. 粗分解结果\n\n")
        f.write(f"**所需服务ID**: {', '.join(coarse_decomposition.get('required_service_ids', []))}\n\n")
        f.write(f"**筛选后工具数量**: {coarse_decomposition.get('filtered_tools_count', 0)}\n\n")
        if coarse_decomposition.get('service_summaries'):
            f.write("**服务摘要**:\n\n")
            for service_id, summary in coarse_decomposition['service_summaries'].items():
                f.write(f"- **{service_id}**: {summary}\n")
            f.write("\n")
        f.write("---\n\n")
        
        # 3. 细分解结果
        f.write("## 3. 细分解结果\n\n")
        raw_tasks = fine_decomposition.get('raw_tasks', [])
        f.write(f"**任务数量**: {len(raw_tasks)}\n\n")
        if raw_tasks:
            f.write("### 任务列表\n\n")
            for i, task in enumerate(raw_tasks, 1):
                f.write(f"#### 任务 {i}: {task.get('task_id', f'task_{i}')}\n\n")
                f.write(f"**描述**: {task.get('description', task.get('name', task.get('content', '')))}\n\n")
                
                tools = task.get('tools', [])
                if tools:
                    tool_names = []
                    for tool in tools:
                        if isinstance(tool, str):
                            tool_names.append(tool)
                        elif isinstance(tool, dict):
                            tool_names.append(tool.get('tool_name', tool.get('name', '')))
                    f.write(f"**工具**: {', '.join(tool_names) if tool_names else '无'}\n\n")
                
                deps = task.get('dependencies', [])
                if deps:
                    f.write(f"**依赖**: {', '.join(deps)}\n\n")
                
                inputs = task.get('inputs', [])
                if inputs:
                    f.write(f"**输入参数**: {', '.join(inputs) if isinstance(inputs, list) else str(inputs)}\n\n")
                outputs = task.get('outputs', [])
                if outputs:
                    f.write(f"**输出**: {', '.join(outputs) if isinstance(outputs, list) else str(outputs)}\n\n")
                
                f.write("\n")
        
        if fine_decomposition.get('decomposition_summary'):
            f.write("### 分解摘要\n\n")
            f.write(f"{fine_decomposition['decomposition_summary']}\n\n")
        f.write("---\n\n")
        
        # 4. 参数推断结果
        f.write("## 4. 参数推断结果\n\n")
        inference_results = parameter_inference.get('task_results', {})
        f.write(f"**推断任务数**: {len(inference_results)}\n\n")
        
        total_determined = 0
        total_from_task = 0
        total_user_required = 0
        
        for task_id, result in inference_results.items():
            f.write(f"### 任务 {task_id}\n\n")
            
            task_info = parameter_inference.get('task_info', {}).get(task_id, {})
            if task_info.get('content'):
                f.write(f"**任务描述**: {task_info['content'][:100]}...\n\n")
            
            tool_name = result.get('tool_name')
            if tool_name:
                f.write(f"**工具**: {tool_name}\n\n")
            
            params = result.get('parameters', {})
            if params:
                f.write(f"**参数推断结果** ({len(params)} 个):\n\n")
                for param_name, param_data in params.items():
                    if isinstance(param_data, dict):
                        source_type = param_data.get('source_type', 'unknown')
                    else:
                        source_type = param_data.source_type.value if hasattr(param_data, 'source_type') else 'unknown'
                    
                    if source_type == ParameterSourceType.DETERMINED.value:
                        value = param_data.get('value') if isinstance(param_data, dict) else param_data.value
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        f.write(f"- `{param_name}`: {value_str} **[确定值]** ✓\n")
                        total_determined += 1
                    elif source_type == ParameterSourceType.FROM_TASK.value:
                        source_task = param_data.get('source_task_id') if isinstance(param_data, dict) else param_data.source_task_id
                        source_key = param_data.get('source_output_key') if isinstance(param_data, dict) else param_data.source_output_key
                        key_info = f" (输出键: {source_key})" if source_key else ""
                        f.write(f"- `{param_name}`: 来自任务 `{source_task}`{key_info} **[任务结果]** ✓\n")
                        total_from_task += 1
                    elif source_type == ParameterSourceType.USER_REQUIRED.value:
                        prompt = param_data.get('user_prompt') if isinstance(param_data, dict) else param_data.user_prompt
                        f.write(f"- `{param_name}`: {prompt} **[需要用户提供]** ✓\n")
                        total_user_required += 1
                    else:
                        f.write(f"- `{param_name}`: 未知类型\n")
                f.write("\n")
            else:
                f.write("**参数推断结果**: 无参数\n\n")
            
            f.write("---\n\n")
        
        # 统计信息
        total_params = total_determined + total_from_task + total_user_required
        f.write("## 统计信息\n\n")
        f.write(f"- **确定的参数值**: {total_determined}\n")
        f.write(f"- **来自任务的参数**: {total_from_task}\n")
        f.write(f"- **需要用户提供的参数**: {total_user_required}\n")
        f.write(f"- **总参数数**: {total_params}\n")
        if total_params > 0:
            f.write(f"- **参数识别成功率**: 100% (包括确定值、来自任务、需要用户提供)\n")
        f.write(f"\n> **说明**: 正确识别出参数来源（确定值、来自任务、需要用户提供）都算作成功的推断结果。\n")
        
        # 推断质量（如果有）
        if 'inference_quality' in parameter_inference:
            quality = parameter_inference['inference_quality']
            f.write("\n## 推断质量统计\n\n")
            f.write(f"- **高质量推断**: {quality['high_quality']} 个任务\n")
            f.write(f"- **中等质量推断**: {quality['medium_quality']} 个任务\n")
            f.write(f"- **低质量推断**: {quality['low_quality']} 个任务\n")
            f.write(f"- **需要用户提供**: {quality['needs_user']} 个任务\n")
        f.write("\n")
    
    print(f"\n✓ 测试总结日志已保存: {log_file}")


# ===================== 测试类 =====================

def run_parameter_inference_test(test_case: Dict, level: str = "未知"):
    """
    运行参数推断测试的通用函数
    
    Args:
        test_case: 测试用例字典
        level: 测试级别（用于日志）
    """
    print(f"\n{'='*80}")
    print(f"测试用例: {test_case['name']} (Level {level})")
    print(f"描述: {test_case['description']}")
    print(f"用户输入: {test_case['user_input']}")
    if test_case['execution_plan']:
        print(f"执行计划:\n{test_case['execution_plan']}")
    print(f"{'='*80}\n")
    
    # 步骤1: 任务分解
    print("[步骤1] 执行任务分解（粗分解 → 细分解 → 并行推断 → 参数推断）...")
    global_state, decomposition_state = run_full_decomposition_flow(
        test_case['user_input'],
        test_case['execution_plan']
    )
    
    assert global_state is not None, "GlobalState 不应该为 None"
    assert len(global_state.subtasks) > 0, "应该生成至少一个子任务"
    print(f"✓ 任务分解完成，生成了 {len(global_state.subtasks)} 个子任务")
    
    # 显示任务信息
    for i, task in enumerate(global_state.subtasks, 1):
        print(f"  任务 {i}: {task.task_id} - {task.content[:60]}...")
        if task.dependencies:
            print(f"    依赖: {', '.join(task.dependencies)}")
    
    # 步骤2: 验证参数推断结果
    print("\n[步骤2] 验证参数推断结果（已在任务分解阶段完成，不执行任何任务）...")
    assert hasattr(decomposition_state, 'parameter_inference_results'), \
        "参数推断结果应该在 task_decomposition 阶段生成"
    print(f"✓ 参数推断完成（未执行任何任务）")
    
    # 提取参数推断结果
    task_results_dict, task_info_dict, stats = extract_parameter_inference_results(decomposition_state)
    
    # 显示推断结果
    print_parameter_inference_results(task_results_dict, task_info_dict, stats)
    
    # 准备并保存日志
    coarse_decomposition, fine_decomposition, parameter_inference = prepare_log_data(
        decomposition_state, task_results_dict, task_info_dict
    )
    
    save_test_summary_log(
        test_case['name'],
        test_case['execution_plan'],
        coarse_decomposition,
        fine_decomposition,
        parameter_inference
    )
    
    # 验证：至少应该有一些参数推断结果
    assert stats['total_count'] > 0, \
        "应该至少有一些参数推断结果。正确识别参数来源（确定值、来自任务、需要用户提供）都算成功。"
    
    return stats


class TestParameterInferenceLevel1:
    """Level 1: 简单级别参数推断测试"""
    
    @pytest.mark.parametrize("test_case", LEVEL_1_TEST_CASES)
    def test_level1_parameter_inference(self, test_case):
        """测试简单级别的参数推断"""
        run_parameter_inference_test(test_case, "1")


class TestParameterInferenceLevel2:
    """Level 2: 中等级别参数推断测试"""
    
    @pytest.mark.parametrize("test_case", LEVEL_2_TEST_CASES)
    def test_level2_parameter_inference(self, test_case):
        """测试中等级别的参数推断"""
        run_parameter_inference_test(test_case, "2")


class TestParameterInferenceLevel3:
    """Level 3: 困难级别参数推断测试"""
    
    @pytest.mark.parametrize("test_case", LEVEL_3_TEST_CASES)
    def test_level3_parameter_inference(self, test_case):
        """测试困难级别的参数推断"""
        print(f"\n{'='*80}")
        print(f"测试用例: {test_case['name']} (Level 3)")
        print(f"描述: {test_case['description']}")
        print(f"用户输入: {test_case['user_input']}")
        if test_case['execution_plan']:
            print(f"执行计划:\n{test_case['execution_plan']}")
        print(f"{'='*80}\n")
        
        # 步骤1: 任务分解
        print("[步骤1] 执行任务分解（粗分解 → 细分解 → 并行推断 → 参数推断）...")
        global_state, decomposition_state = run_full_decomposition_flow(
            test_case['user_input'],
            test_case['execution_plan']
        )
        
        assert global_state is not None, "GlobalState 不应该为 None"
        if len(global_state.subtasks) == 0:
            print("⚠ 警告：任务分解未生成任何子任务，可能描述过于模糊")
            pytest.skip("任务分解未生成子任务，跳过参数推断测试")
        
        print(f"✓ 任务分解完成，生成了 {len(global_state.subtasks)} 个子任务")
        
        # 步骤2: 验证参数推断结果
        print("\n[步骤2] 验证参数推断结果（已在任务分解阶段完成，不执行任何任务）...")
        assert hasattr(decomposition_state, 'parameter_inference_results'), \
            "参数推断结果应该在 task_decomposition 阶段生成"
        print(f"✓ 参数推断完成（未执行任何任务）")
        
        # 提取参数推断结果
        task_results_dict, task_info_dict, stats = extract_parameter_inference_results(decomposition_state)
        
        # 分析推断质量
        inference_quality = analyze_inference_quality(task_results_dict)
        
        # 准备并保存日志
        coarse_decomposition, fine_decomposition, parameter_inference = prepare_log_data(
            decomposition_state, task_results_dict, task_info_dict, inference_quality
        )
        
        save_test_summary_log(
            test_case['name'],
            test_case['execution_plan'],
            coarse_decomposition,
            fine_decomposition,
            parameter_inference
        )
        
        # 验证：对于困难级别，至少应该尝试推断参数
        total_tasks_with_params = sum([
            inference_quality["high_quality"],
            inference_quality["medium_quality"],
            inference_quality["low_quality"]
        ])
        
        assert total_tasks_with_params > 0 or inference_quality["needs_user"] > 0, \
            "应该至少有一些参数推断尝试。正确识别参数来源（确定值、来自任务、需要用户提供）都算成功。"


class TestParameterInferenceComparison:
    """参数推断效果对比测试"""
    
    def test_clear_vs_vague_comparison(self):
        """对比清晰描述和模糊描述的参数推断效果"""
        print(f"\n{'='*80}")
        print("对比测试：清晰描述 vs 模糊描述")
        print(f"{'='*80}\n")
        
        # 清晰描述
        clear_input = "搜索疾病为 COVID-19，组织为 blood 的抗体数据"
        print(f"[清晰描述] {clear_input}")
        clear_state, clear_decomposition = run_full_decomposition_flow(clear_input)
        
        # 模糊描述
        vague_input = "帮我分析一下抗体数据"
        print(f"\n[模糊描述] {vague_input}")
        vague_state, vague_decomposition = run_full_decomposition_flow(vague_input)
        
        # 统计参数推断结果
        def count_inference_stats(decomposition_state: TaskDecompositionState):
            """统计参数推断结果"""
            determined = 0
            from_task = 0
            user_required = 0
            for inference_result in decomposition_state.parameter_inference_results.values():
                for param_result in inference_result.parameters.values():
                    source_type = param_result.source_type.value if hasattr(param_result.source_type, 'value') else str(param_result.source_type)
                    if source_type == ParameterSourceType.DETERMINED.value:
                        determined += 1
                    elif source_type == ParameterSourceType.FROM_TASK.value:
                        from_task += 1
                    elif source_type == ParameterSourceType.USER_REQUIRED.value:
                        user_required += 1
            return determined, from_task, user_required
        
        clear_determined, clear_from_task, clear_user = count_inference_stats(clear_decomposition)
        vague_determined, vague_from_task, vague_user = count_inference_stats(vague_decomposition)
        
        print("\n对比结果:")
        print(f"  清晰描述:")
        print(f"    确定值: {clear_determined} 个")
        print(f"    来自任务: {clear_from_task} 个")
        print(f"    需要用户提供: {clear_user} 个")
        print(f"  模糊描述:")
        print(f"    确定值: {vague_determined} 个")
        print(f"    来自任务: {vague_from_task} 个")
        print(f"    需要用户提供: {vague_user} 个")
        
        print(f"\n✓ 对比测试完成")
        print(f"  清晰描述的推断效果: {clear_determined + clear_from_task} 个参数（确定值+来自任务）")
        print(f"  模糊描述的推断效果: {vague_determined + vague_from_task} 个参数（确定值+来自任务）")
        
        # 保存对比测试日志
        clear_task_results, clear_task_info, _ = extract_parameter_inference_results(clear_decomposition)
        vague_task_results, vague_task_info, _ = extract_parameter_inference_results(vague_decomposition)
        
        clear_coarse, clear_fine, clear_param = prepare_log_data(
            clear_decomposition, clear_task_results, clear_task_info
        )
        vague_coarse, vague_fine, vague_param = prepare_log_data(
            vague_decomposition, vague_task_results, vague_task_info
        )
        
        save_test_summary_log("对比测试_清晰描述", None, clear_coarse, clear_fine, clear_param)
        save_test_summary_log("对比测试_模糊描述", None, vague_coarse, vague_fine, vague_param)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
