"""
Immunity Subgraph 测试用例

测试 immunity 子图的完整功能，包括：
1. 查询分解节点
2. 深度研究节点
3. 假设生成节点
4. 计划生成节点
5. 评估节点
6. 完整流程测试

运行方式：pytest tests/test_immunity_subgraph.py -v
"""

import os
import pytest
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any, Optional

# 加载环境变量
load_dotenv()

# 添加agent目录到路径
import sys
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from nodes.subagents.immunity.graph import (
    build_immunity_subgraph,
    immunity_input_mapper,
    immunity_output_mapper,
    ImmunityState
)
from state import GlobalState, UserTaskType


# ===================== 测试用例数据 =====================

IMMUNITY_TEST_CASES = [
    {
        "name": "简单_抗体设计研究",
        "user_input": "设计针对 COVID-19 的抗体",
        "description": "简单的抗体设计研究问题"
    },
    {
        "name": "中等_抗体优化研究",
        "user_input": "优化现有抗体的结合亲和力和稳定性",
        "description": "中等复杂度的抗体优化研究"
    },
    {
        "name": "复杂_多步骤免疫学研究",
        "user_input": "研究 B 细胞受体在 COVID-19 感染中的动态变化，包括 V(D)J 重组模式、CDR3 区域特征和抗体亲和力成熟过程",
        "description": "复杂的多步骤免疫学研究问题"
    },
    {
        "name": "特定_抗体-抗原相互作用",
        "user_input": "分析特定抗体与 SARS-CoV-2 刺突蛋白的相互作用机制，预测结合位点和结合强度",
        "description": "特定抗体-抗原相互作用分析"
    },
]


# ===================== 辅助函数 =====================

def _ensure_immunity_state(result):
    """确保结果是 ImmunityState 对象（处理字典返回值）"""
    if isinstance(result, dict):
        return ImmunityState(**result)
    return result


def _create_test_sandbox() -> str:
    """创建测试沙盒目录"""
    test_sandbox = Path(agent_dir) / "tests" / "sandbox" / "immunity_test"
    test_sandbox.mkdir(parents=True, exist_ok=True)
    return str(test_sandbox)


def _save_test_log(
    test_name: str,
    user_input: str,
    immunity_state: ImmunityState,
    execution_time: float
) -> str:
    """保存测试日志"""
    logs_dir = Path(agent_dir) / "tests" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"immunity_test_{test_name}_{timestamp}.md"
    
    log_content = f"""# Immunity 子图测试日志

## 测试信息
- **测试名称**: {test_name}
- **测试时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **执行时间**: {execution_time:.2f} 秒

## 用户输入
```
{user_input}
```

## Stage 1: 查询分解结果

### 优化后的查询列表
"""
    
    if immunity_state.optimized_questions:
        for i, query in enumerate(immunity_state.optimized_questions, 1):
            log_content += f"{i}. {query}\n"
    else:
        log_content += "无优化查询\n"
    
    log_content += f"""
### 优化查询数量
{len(immunity_state.optimized_questions)}

## Stage 2: 深度研究结果

### 研究主题
"""
    
    # 安全获取 deep_research_findings
    deep_research_findings = immunity_state.deep_research_findings if immunity_state.deep_research_findings else {}
    log_content += f"""{deep_research_findings.get('topic', '未指定') if isinstance(deep_research_findings, dict) else '未指定'}

### 研究置信度
{immunity_state.research_confidence:.1f}%

### 关键洞察
"""
    
    if immunity_state.research_insights:
        for i, insight in enumerate(immunity_state.research_insights, 1):
            log_content += f"{i}. {insight}\n"
    else:
        log_content += "无关键洞察\n"
    
    log_content += f"""
### 支持证据
"""
    
    if immunity_state.research_evidence:
        for i, evidence in enumerate(immunity_state.research_evidence, 1):
            log_content += f"{i}. {evidence}\n"
    else:
        log_content += "无支持证据\n"
    
    log_content += f"""
### 知识缺口
"""
    
    if immunity_state.research_gaps:
        for i, gap in enumerate(immunity_state.research_gaps, 1):
            log_content += f"{i}. {gap}\n"
    else:
        log_content += "无知识缺口\n"
    
    log_content += f"""
### 研究建议
"""
    
    if immunity_state.research_recommendations:
        for i, rec in enumerate(immunity_state.research_recommendations, 1):
            log_content += f"{i}. {rec}\n"
    else:
        log_content += "无研究建议\n"
    
    log_content += f"""
### 研究摘要
"""
    
    if immunity_state.research_summary:
        research_summary_preview = immunity_state.research_summary[:500] if len(immunity_state.research_summary) > 500 else immunity_state.research_summary
        log_content += f"{research_summary_preview}{'...' if len(immunity_state.research_summary) > 500 else ''}\n"
    else:
        log_content += "无研究摘要\n"
    
    log_content += f"""
## Stage 3: 假设生成结果

### 假设陈述
"""
    
    # 安全获取 hypothesis
    hypothesis = immunity_state.hypothesis if immunity_state.hypothesis else {}
    if isinstance(hypothesis, dict) and hypothesis.get('statement'):
        log_content += f"{hypothesis.get('statement')}\n"
    else:
        log_content += "未生成假设\n"
    
    log_content += f"""
### 假设置信度
{immunity_state.hypothesis_confidence:.1f}%

### 创新水平
{hypothesis.get('innovation_level', '未指定') if isinstance(hypothesis, dict) else '未指定'}

### 可测试的预测
"""
    
    if immunity_state.testable_predictions:
        for i, pred in enumerate(immunity_state.testable_predictions, 1):
            log_content += f"{i}. {pred}\n"
    else:
        log_content += "无可测试的预测\n"
    
    log_content += f"""
### 假设摘要
"""
    
    if immunity_state.hypothesis_summary:
        hypothesis_summary_preview = immunity_state.hypothesis_summary[:500] if len(immunity_state.hypothesis_summary) > 500 else immunity_state.hypothesis_summary
        log_content += f"{hypothesis_summary_preview}{'...' if len(immunity_state.hypothesis_summary) > 500 else ''}\n"
    else:
        log_content += "无假设摘要\n"
    
    log_content += f"""
## Stage 4: 计划生成结果 ⭐

### 实验计划长度
{len(immunity_state.final_enhanced_plan) if immunity_state.final_enhanced_plan else 0} 字符

### 实验计划预览
```
"""
    
    if immunity_state.final_enhanced_plan:
        plan_preview = immunity_state.final_enhanced_plan[:1000] if len(immunity_state.final_enhanced_plan) > 1000 else immunity_state.final_enhanced_plan
        log_content += f"{plan_preview}{'...' if len(immunity_state.final_enhanced_plan) > 1000 else ''}\n"
    else:
        log_content += "无实验计划\n"
    
    log_content += "```\n\n### 完整实验计划\n"
    
    if immunity_state.final_enhanced_plan:
        log_content += f"{immunity_state.final_enhanced_plan}\n"
    else:
        log_content += "无实验计划\n"
    
    log_content += f"""
## Stage 5: 评估结果

### 评估报告长度
{len(immunity_state.final_evaluation) if immunity_state.final_evaluation else 0} 字符

### 评估报告预览
```
"""
    
    if immunity_state.final_evaluation:
        eval_preview = immunity_state.final_evaluation[:1000] if len(immunity_state.final_evaluation) > 1000 else immunity_state.final_evaluation
        log_content += f"{eval_preview}{'...' if len(immunity_state.final_evaluation) > 1000 else ''}\n"
    else:
        log_content += "无评估报告\n"
    
    log_content += "```\n\n### 完整评估报告\n"
    
    if immunity_state.final_evaluation:
        log_content += f"{immunity_state.final_evaluation}\n"
    else:
        log_content += "无评估报告\n"
    
    # 构建总结部分（避免在 f-string 中使用 emoji）
    log_content += "\n## 总结\n\n### 流程完成情况\n"
    
    query_status = '完成' if immunity_state.optimized_questions else '未完成'
    research_status = '完成' if immunity_state.research_summary else '未完成'
    hypothesis_status = '完成' if immunity_state.hypothesis_summary else '未完成'
    plan_status = '完成' if immunity_state.final_enhanced_plan else '未完成'
    eval_status = '完成' if immunity_state.final_evaluation else '未完成'
    
    log_content += f"- ✅ 查询分解: {query_status}\n"
    log_content += f"- ✅ 深度研究: {research_status}\n"
    log_content += f"- ✅ 假设生成: {hypothesis_status}\n"
    log_content += f"- ✅ 计划生成: {plan_status}\n"
    log_content += f"- ✅ 评估: {eval_status}\n"
    
    log_content += "\n### 关键指标\n"
    log_content += f"- 优化查询数: {len(immunity_state.optimized_questions)}\n"
    log_content += f"- 研究置信度: {immunity_state.research_confidence:.1f}%\n"
    log_content += f"- 假设置信度: {immunity_state.hypothesis_confidence:.1f}%\n"
    
    plan_len = len(immunity_state.final_enhanced_plan) if immunity_state.final_enhanced_plan else 0
    eval_len = len(immunity_state.final_evaluation) if immunity_state.final_evaluation else 0
    log_content += f"- 计划文档长度: {plan_len} 字符\n"
    log_content += f"- 评估报告长度: {eval_len} 字符\n"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(log_content)
    
    print(f"\n📄 测试日志已保存到: {log_file}")
    return str(log_file)


# ===================== Fixtures =====================

@pytest.fixture(scope="module")
def immunity_subgraph():
    """构建并返回 Immunity 子图"""
    return build_immunity_subgraph()


@pytest.fixture
def sample_global_state():
    """示例全局状态"""
    sandbox_dir = _create_test_sandbox()
    return GlobalState(
        user_input="设计针对 COVID-19 的抗体",
        user_task_type=UserTaskType.IMMUNOLOGY_TASK,
        sandbox_dir=sandbox_dir
    )


# ===================== 测试类 =====================

class TestImmunitySubgraphBasic:
    """Immunity Subgraph 基础功能测试"""
    
    def test_subgraph_build(self, immunity_subgraph):
        """测试子图构建是否成功"""
        assert immunity_subgraph is not None
        print("✓ Immunity Subgraph 构建成功")
    
    def test_subgraph_invoke_basic(self, immunity_subgraph, sample_global_state):
        """测试子图基本调用"""
        # 使用 input_mapper 转换状态
        subgraph_input = immunity_input_mapper(sample_global_state)
        
        # 调用子图
        subgraph_output = immunity_subgraph.invoke(subgraph_input)
        
        # 确保输出是 ImmunityState 对象
        result = _ensure_immunity_state(subgraph_output)
        
        assert result is not None
        assert hasattr(result, 'original_question')
        assert result.original_question == sample_global_state.user_input
        print(f"✓ Immunity Subgraph 基本调用成功")


class TestImmunitySubgraphStages:
    """Immunity Subgraph 各阶段测试"""
    
    def test_query_decomposition_stage(self, immunity_subgraph, sample_global_state):
        """测试 Stage 1: 查询分解"""
        import time
        start_time = time.time()
        
        subgraph_input = immunity_input_mapper(sample_global_state)
        result = immunity_subgraph.invoke(subgraph_input)
        result = _ensure_immunity_state(result)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证查询分解结果
        assert result.optimized_questions is not None
        assert isinstance(result.optimized_questions, list)
        assert len(result.optimized_questions) > 0
        
        print(f"\n✓ Stage 1（查询分解）完成")
        print(f"  优化查询数: {len(result.optimized_questions)}")
        for i, query in enumerate(result.optimized_questions[:3], 1):
            print(f"    {i}. {query[:80]}...")
        
        # 保存测试日志
        log_file = _save_test_log(
            "Stage1_查询分解",
            sample_global_state.user_input,
            result,
            execution_time
        )
        print(f"  日志文件: {log_file}")
    
    def test_deep_research_stage(self, immunity_subgraph, sample_global_state):
        """测试 Stage 2: 深度研究"""
        import time
        start_time = time.time()
        
        subgraph_input = immunity_input_mapper(sample_global_state)
        result = immunity_subgraph.invoke(subgraph_input)
        result = _ensure_immunity_state(result)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证深度研究结果
        assert result.deep_research_findings is not None
        assert isinstance(result.deep_research_findings, dict)
        assert result.research_confidence >= 0
        assert result.research_confidence <= 100
        
        print(f"\n✓ Stage 2（深度研究）完成")
        print(f"  研究主题: {result.deep_research_findings.get('topic', '未指定') if isinstance(result.deep_research_findings, dict) else '未指定'}")
        print(f"  研究置信度: {result.research_confidence:.1f}%")
        print(f"  关键洞察数: {len(result.research_insights)}")
        print(f"  支持证据数: {len(result.research_evidence)}")
        
        # 保存测试日志
        log_file = _save_test_log(
            "Stage2_深度研究",
            sample_global_state.user_input,
            result,
            execution_time
        )
        print(f"  日志文件: {log_file}")
    
    def test_hypothesis_generation_stage(self, immunity_subgraph, sample_global_state):
        """测试 Stage 3: 假设生成"""
        import time
        start_time = time.time()
        
        subgraph_input = immunity_input_mapper(sample_global_state)
        result = immunity_subgraph.invoke(subgraph_input)
        result = _ensure_immunity_state(result)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证假设生成结果
        assert result.hypothesis is not None
        assert isinstance(result.hypothesis, dict)
        assert result.hypothesis_confidence >= 0
        assert result.hypothesis_confidence <= 100
        hypothesis = result.hypothesis if isinstance(result.hypothesis, dict) else {}
        assert hypothesis.get('statement') is not None
        
        print(f"\n✓ Stage 3（假设生成）完成")
        statement = hypothesis.get('statement', '未指定')
        print(f"  假设: {statement[:100]}..." if len(statement) > 100 else f"  假设: {statement}")
        print(f"  假设置信度: {result.hypothesis_confidence:.1f}%")
        print(f"  创新水平: {hypothesis.get('innovation_level', '未指定')}")
        print(f"  可测试预测数: {len(result.testable_predictions)}")
        
        # 保存测试日志
        log_file = _save_test_log(
            "Stage3_假设生成",
            sample_global_state.user_input,
            result,
            execution_time
        )
        print(f"  日志文件: {log_file}")
    
    def test_planning_stage(self, immunity_subgraph, sample_global_state):
        """测试 Stage 4: 计划生成 ⭐"""
        import time
        start_time = time.time()
        
        subgraph_input = immunity_input_mapper(sample_global_state)
        result = immunity_subgraph.invoke(subgraph_input)
        result = _ensure_immunity_state(result)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证计划生成结果
        assert result.final_enhanced_plan is not None
        assert len(result.final_enhanced_plan) > 0
        
        print(f"\n✓ Stage 4（计划生成）完成 ⭐")
        print(f"  计划长度: {len(result.final_enhanced_plan)} 字符")
        print(f"  计划预览: {result.final_enhanced_plan[:200]}...")
        
        # 保存测试日志
        log_file = _save_test_log(
            "Stage4_计划生成",
            sample_global_state.user_input,
            result,
            execution_time
        )
        print(f"  日志文件: {log_file}")
    
    def test_evaluation_stage(self, immunity_subgraph, sample_global_state):
        """测试 Stage 5: 评估"""
        import time
        start_time = time.time()
        
        subgraph_input = immunity_input_mapper(sample_global_state)
        result = immunity_subgraph.invoke(subgraph_input)
        result = _ensure_immunity_state(result)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证评估结果
        assert result.final_evaluation is not None
        assert len(result.final_evaluation) > 0
        
        print(f"\n✓ Stage 5（评估）完成")
        print(f"  评估报告长度: {len(result.final_evaluation)} 字符")
        print(f"  评估报告预览: {result.final_evaluation[:200]}...")
        
        # 保存测试日志
        log_file = _save_test_log(
            "Stage5_评估",
            sample_global_state.user_input,
            result,
            execution_time
        )
        print(f"  日志文件: {log_file}")


class TestImmunitySubgraphFullFlow:
    """Immunity Subgraph 完整流程测试"""
    
    @pytest.mark.parametrize("test_case", IMMUNITY_TEST_CASES)
    def test_full_flow(self, immunity_subgraph, test_case):
        """测试完整流程：查询分解 → 深度研究 → 假设生成 → 计划生成 → 评估"""
        import time
        
        # 创建测试沙盒
        sandbox_dir = _create_test_sandbox()
        
        # 创建全局状态
        global_state = GlobalState(
            user_input=test_case["user_input"],
            user_task_type=UserTaskType.IMMUNOLOGY_TASK,
            sandbox_dir=sandbox_dir
        )
        
        # 记录开始时间
        start_time = time.time()
        
        # 使用 input_mapper 转换状态
        subgraph_input = immunity_input_mapper(global_state)
        
        # 调用子图
        subgraph_output = immunity_subgraph.invoke(subgraph_input)
        
        # 确保输出是 ImmunityState 对象
        immunity_state = _ensure_immunity_state(subgraph_output)
        
        # 使用 output_mapper 转换回全局状态
        updated_global_state = immunity_output_mapper(immunity_state, global_state)
        
        # 记录结束时间
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证完整流程
        assert immunity_state.optimized_questions is not None
        assert len(immunity_state.optimized_questions) > 0, "查询分解应该生成优化查询"
        
        assert immunity_state.research_summary is not None
        assert len(immunity_state.research_summary) > 0, "深度研究应该生成研究摘要"
        
        assert immunity_state.hypothesis_summary is not None
        assert len(immunity_state.hypothesis_summary) > 0, "假设生成应该生成假设摘要"
        
        assert immunity_state.final_enhanced_plan is not None
        assert len(immunity_state.final_enhanced_plan) > 0, "计划生成应该生成实验计划"
        
        assert immunity_state.final_evaluation is not None
        assert len(immunity_state.final_evaluation) > 0, "评估应该生成评估报告"
        
        # 验证输出映射
        assert updated_global_state.merged_result is not None
        assert "immunity_plan" in updated_global_state.merged_result
        
        immunity_plan = updated_global_state.merged_result["immunity_plan"]
        assert immunity_plan["experimental_plan"] == immunity_state.final_enhanced_plan
        
        # 保存测试日志
        log_file = _save_test_log(
            test_case["name"],
            test_case["user_input"],
            immunity_state,
            execution_time
        )
        
        print(f"\n✅ 完整流程测试通过: {test_case['name']}")
        print(f"  执行时间: {execution_time:.2f} 秒")
        print(f"  优化查询数: {len(immunity_state.optimized_questions)}")
        print(f"  研究置信度: {immunity_state.research_confidence:.1f}%")
        print(f"  假设置信度: {immunity_state.hypothesis_confidence:.1f}%")
        print(f"  计划长度: {len(immunity_state.final_enhanced_plan)} 字符")
        print(f"  日志文件: {log_file}")


class TestImmunitySubgraphInputOutput:
    """Immunity Subgraph 输入/输出映射测试"""
    
    def test_input_mapper(self, sample_global_state):
        """测试输入映射"""
        immunity_state = immunity_input_mapper(sample_global_state)
        
        assert isinstance(immunity_state, ImmunityState)
        assert immunity_state.original_question == sample_global_state.user_input
        assert immunity_state.sandbox_dir == sample_global_state.sandbox_dir
        assert immunity_state.parent_state == sample_global_state
        
        print("✓ 输入映射测试通过")
    
    def test_output_mapper(self, sample_global_state):
        """测试输出映射"""
        # 创建模拟的 immunity_state
        immunity_state = ImmunityState(
            original_question=sample_global_state.user_input,
            optimized_questions=["查询1", "查询2"],
            research_summary="研究摘要",
            hypothesis_summary="假设摘要",
            final_enhanced_plan="实验计划",
            final_evaluation="评估报告",
            sandbox_dir=sample_global_state.sandbox_dir
        )
        
        # 执行输出映射
        updated_global_state = immunity_output_mapper(immunity_state, sample_global_state)
        
        # 验证输出映射结果
        assert updated_global_state.merged_result is not None
        assert "immunity_plan" in updated_global_state.merged_result
        
        immunity_plan = updated_global_state.merged_result["immunity_plan"]
        assert immunity_plan["original_question"] == immunity_state.original_question
        assert immunity_plan["experimental_plan"] == immunity_state.final_enhanced_plan
        assert immunity_plan["evaluation"] == immunity_state.final_evaluation
        
        print("✓ 输出映射测试通过")


class TestImmunitySubgraphEdgeCases:
    """Immunity Subgraph 边界情况测试"""
    
    def test_empty_user_input(self, immunity_subgraph):
        """测试空用户输入"""
        sandbox_dir = _create_test_sandbox()
        global_state = GlobalState(
            user_input="",
            user_task_type=UserTaskType.IMMUNOLOGY_TASK,
            sandbox_dir=sandbox_dir
        )
        
        subgraph_input = immunity_input_mapper(global_state)
        result = immunity_subgraph.invoke(subgraph_input)
        result = _ensure_immunity_state(result)
        
        # 空输入应该也能处理（可能生成默认查询或跳过某些阶段）
        assert result is not None
        print("✓ 空用户输入测试通过")
    
    def test_very_long_user_input(self, immunity_subgraph):
        """测试超长用户输入"""
        sandbox_dir = _create_test_sandbox()
        long_input = "设计针对 COVID-19 的抗体，需要分析 V(D)J 重组、CDR3 区域特征、抗体亲和力成熟过程、B 细胞受体动态变化、抗体-抗原相互作用机制、结合位点预测、结合强度分析、稳定性优化、表达效率提升、免疫原性评估" * 10
        
        global_state = GlobalState(
            user_input=long_input,
            user_task_type=UserTaskType.IMMUNOLOGY_TASK,
            sandbox_dir=sandbox_dir
        )
        
        subgraph_input = immunity_input_mapper(global_state)
        result = immunity_subgraph.invoke(subgraph_input)
        result = _ensure_immunity_state(result)
        
        assert result is not None
        assert result.original_question == long_input
        print("✓ 超长用户输入测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

