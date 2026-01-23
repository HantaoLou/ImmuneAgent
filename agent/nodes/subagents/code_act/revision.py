"""
CodeAct Revision 机制

参考 SE-Agent 的 Revision 机制，实现失败驱动的策略生成：
1. 深度自我反思：分析失败原因，识别根本问题
2. 正交策略生成：生成与失败路径不同的新策略
3. 架构级改进：不仅修复错误，还改进整体方法
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import json

from agent.nodes.subagents.code_act.trajectory import CodeTrajectory, TrajectoryStatus
from agent.utils.llm_factory import create_reasoning_advanced_llm, create_code_llm


class RevisionStrategy(str, Enum):
    """Revision策略类型"""
    CODE_FIX = "code_fix"  # 代码修复：修复语法错误、逻辑错误
    PARAMETER_ADJUST = "parameter_adjust"  # 参数调整：修改参数值或类型
    ARCHITECTURE_CHANGE = "architecture_change"  # 架构改进：改变整体实现方法
    DEPENDENCY_RESOLVE = "dependency_resolve"  # 依赖解决：添加缺失的依赖或导入
    ENVIRONMENT_FIX = "environment_fix"  # 环境修复：修复环境配置问题


class RevisionPlan(BaseModel):
    """Revision计划"""
    strategy: RevisionStrategy = Field(description="采用的策略")
    root_cause: str = Field(description="根本原因分析")
    action_items: List[str] = Field(default_factory=list, description="具体行动项")
    expected_outcome: str = Field(description="预期结果")
    confidence: float = Field(description="信心度（0-1）", ge=0.0, le=1.0)
    orthogonal: bool = Field(default=False, description="是否与失败路径正交（采用不同方法）")


class RevisionAnalyzer:
    """
    Revision分析器
    
    负责分析失败轨迹，生成Revision计划。
    """
    
    def __init__(self):
        self.llm = create_reasoning_advanced_llm()
    
    def analyze_failure(
        self,
        failed_trajectory: CodeTrajectory,
        previous_trajectories: Optional[List[CodeTrajectory]] = None
    ) -> RevisionPlan:
        """
        分析失败轨迹，生成Revision计划
        
        Args:
            failed_trajectory: 失败的轨迹
            previous_trajectories: 之前的尝试轨迹（用于避免重复错误）
        
        Returns:
            Revision计划
        """
        if not self.llm:
            # LLM不可用，使用简单分析
            return self._analyze_failure_simple(failed_trajectory)
        
        try:
            return self._analyze_failure_with_llm(failed_trajectory, previous_trajectories or [])
        except Exception as e:
            print(f"  ⚠ LLM分析失败: {e}，使用简单分析")
            return self._analyze_failure_simple(failed_trajectory)
    
    def _analyze_failure_with_llm(
        self,
        failed_trajectory: CodeTrajectory,
        previous_trajectories: List[CodeTrajectory]
    ) -> RevisionPlan:
        """使用LLM进行深度分析"""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        system_prompt = """你是一个代码执行失败分析专家。你的任务是深度分析代码执行失败的根本原因，并生成一个Revision计划来修复问题。

分析要求：
1. **深度自我反思**：不要只看表面错误，要识别根本问题
   - 是代码逻辑错误？参数问题？环境配置？依赖缺失？
   - 错误是否可重现？是否有模式？
   
2. **正交策略生成**：生成与失败路径不同的新策略
   - 如果之前尝试了方法A，考虑方法B
   - 避免重复相同的错误路径
   
3. **架构级改进**：不仅修复错误，还改进整体方法
   - 考虑是否有更好的实现方式
   - 是否可以通过改变架构来避免问题

输出格式：JSON格式的Revision计划，包含：
- strategy: 策略类型（code_fix/parameter_adjust/architecture_change/dependency_resolve/environment_fix）
- root_cause: 根本原因分析（详细说明）
- action_items: 具体行动项列表（每项一个字符串）
- expected_outcome: 预期结果描述
- confidence: 信心度（0-1）
- orthogonal: 是否与失败路径正交（true/false）"""
        
        # 准备失败信息
        error_info = {
            "error_type": failed_trajectory.error_type,
            "error_message": failed_trajectory.error_message,
            "error_category": failed_trajectory.error_category,
            "code_preview": failed_trajectory.generated_code[:500] if failed_trajectory.generated_code else None,
            "execution_mode": failed_trajectory.execution_mode,
            "parameters": failed_trajectory.parameters
        }
        
        # 准备之前的尝试信息
        previous_attempts = []
        for prev_traj in previous_trajectories[-3:]:  # 只考虑最近3次尝试
            prev_info = {
                "error_type": prev_traj.error_type,
                "error_message": prev_traj.error_message[:200] if prev_traj.error_message else None,
                "strategy_used": prev_traj.metadata.get("revision_strategy") if prev_traj.metadata else None
            }
            previous_attempts.append(prev_info)
        
        user_prompt = f"""请分析以下代码执行失败，并生成Revision计划：

失败信息：
{json.dumps(error_info, ensure_ascii=False, indent=2)}

之前的尝试（避免重复）：
{json.dumps(previous_attempts, ensure_ascii=False, indent=2) if previous_attempts else "无"}

请生成Revision计划（JSON格式）。"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content.strip()
        
        # 解析JSON响应
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        plan_data = json.loads(response_text)
        
        # 创建Revision计划
        return RevisionPlan(
            strategy=RevisionStrategy(plan_data.get("strategy", "code_fix")),
            root_cause=plan_data.get("root_cause", "未知原因"),
            action_items=plan_data.get("action_items", []),
            expected_outcome=plan_data.get("expected_outcome", "修复错误"),
            confidence=float(plan_data.get("confidence", 0.5)),
            orthogonal=bool(plan_data.get("orthogonal", False))
        )
    
    def _analyze_failure_simple(self, failed_trajectory: CodeTrajectory) -> RevisionPlan:
        """简单分析（LLM不可用时使用）"""
        error_type = failed_trajectory.error_type or "Unknown"
        error_message = failed_trajectory.error_message or ""
        
        # 基于错误类型推断策略
        if "SyntaxError" in error_type or "IndentationError" in error_type:
            strategy = RevisionStrategy.CODE_FIX
            root_cause = "代码语法错误"
            action_items = ["检查代码语法", "修复缩进问题", "验证代码结构"]
        elif "NameError" in error_type or "ImportError" in error_type:
            strategy = RevisionStrategy.DEPENDENCY_RESOLVE
            root_cause = "依赖或导入问题"
            action_items = ["检查导入语句", "添加缺失的依赖", "验证模块路径"]
        elif "TypeError" in error_type or "ValueError" in error_type:
            strategy = RevisionStrategy.PARAMETER_ADJUST
            root_cause = "参数类型或值错误"
            action_items = ["检查参数类型", "验证参数值", "调整参数格式"]
        elif "AttributeError" in error_type:
            strategy = RevisionStrategy.CODE_FIX
            root_cause = "对象属性访问错误"
            action_items = ["检查对象类型", "验证属性存在性", "修复属性访问"]
        else:
            strategy = RevisionStrategy.CODE_FIX
            root_cause = f"执行错误: {error_type}"
            action_items = ["检查代码逻辑", "添加错误处理", "验证执行环境"]
        
        return RevisionPlan(
            strategy=strategy,
            root_cause=root_cause,
            action_items=action_items,
            expected_outcome="修复错误并成功执行",
            confidence=0.6,
            orthogonal=False
        )


class RevisionExecutor:
    """
    Revision执行器
    
    根据Revision计划，生成修复后的代码。
    """
    
    def __init__(self):
        self.llm = create_code_llm()
        from agent.utils.llm_factory import create_code_llm
        self.llm = create_code_llm()
    
    def generate_revision_code(
        self,
        revision_plan: RevisionPlan,
        original_code: str,
        original_error: str,
        task_description: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        根据Revision计划生成修复后的代码
        
        Args:
            revision_plan: Revision计划
            original_code: 原始代码
            original_error: 原始错误信息
            task_description: 任务描述
            parameters: 任务参数
        
        Returns:
            修复后的代码
        """
        if not self.llm:
            # LLM不可用，使用简单修复
            return self._generate_revision_code_simple(revision_plan, original_code, original_error)
        
        try:
            return self._generate_revision_code_with_llm(
                revision_plan, original_code, original_error, task_description, parameters
            )
        except Exception as e:
            print(f"  ⚠ LLM生成修复代码失败: {e}，使用简单修复")
            return self._generate_revision_code_simple(revision_plan, original_code, original_error)
    
    def _generate_revision_code_with_llm(
        self,
        revision_plan: RevisionPlan,
        original_code: str,
        original_error: str,
        task_description: str,
        parameters: Dict[str, Any]
    ) -> str:
        """使用LLM生成修复代码"""
        from langchain_core.messages import SystemMessage, HumanMessage
        from agent.nodes.subagents.code_act.prompt import FIX_CODE_SYSTEM_PROMPT
        
        # 根据策略选择不同的提示词
        if revision_plan.strategy == RevisionStrategy.ARCHITECTURE_CHANGE:
            system_prompt = """你是一个代码架构改进专家。你的任务是根据失败分析，采用完全不同的架构方法重新实现代码。

要求：
1. **正交策略**：不要重复失败的方法，采用全新的实现思路
2. **架构改进**：考虑更优雅、更健壮的实现方式
3. **错误预防**：在代码中加入错误处理和验证
4. **可维护性**：代码应该清晰、易读、易维护

生成完整的、可执行的代码。"""
        else:
            system_prompt = FIX_CODE_SYSTEM_PROMPT
        
        user_prompt = f"""根据以下Revision计划，生成修复后的代码：

Revision计划：
- 策略: {revision_plan.strategy.value}
- 根本原因: {revision_plan.root_cause}
- 行动项: {', '.join(revision_plan.action_items)}
- 预期结果: {revision_plan.expected_outcome}
- 正交策略: {'是' if revision_plan.orthogonal else '否'}

原始代码：
```python
{original_code}
```

原始错误：
{original_error}

任务描述: {task_description}
参数: {json.dumps(parameters, ensure_ascii=False)}

请生成修复后的完整代码（确保可执行）。"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        code = response.content.strip()
        
        # 移除markdown代码块标记
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()
        
        return code
    
    def _generate_revision_code_simple(
        self,
        revision_plan: RevisionPlan,
        original_code: str,
        original_error: str
    ) -> str:
        """简单修复（LLM不可用时使用）"""
        # 基本的错误处理包装
        fixed_code = f"""# Revision修复: {revision_plan.root_cause}
# 策略: {revision_plan.strategy.value}
# 行动项: {', '.join(revision_plan.action_items)}

try:
{chr(10).join('    ' + line for line in original_code.split(chr(10)))}
    result = {{"status": "success", "output": "执行成功"}}
except Exception as e:
    # 错误处理
    result = {{
        "status": "failed",
        "error": str(e),
        "error_type": type(e).__name__
    }}
"""
        return fixed_code


def create_revision_plan(
    failed_trajectory: CodeTrajectory,
    previous_trajectories: Optional[List[CodeTrajectory]] = None
) -> RevisionPlan:
    """
    创建Revision计划的便捷函数
    
    Args:
        failed_trajectory: 失败的轨迹
        previous_trajectories: 之前的尝试轨迹
    
    Returns:
        Revision计划
    """
    analyzer = RevisionAnalyzer()
    return analyzer.analyze_failure(failed_trajectory, previous_trajectories)


def execute_revision_plan(
    revision_plan: RevisionPlan,
    original_code: str,
    original_error: str,
    task_description: str,
    parameters: Dict[str, Any]
) -> str:
    """
    执行Revision计划的便捷函数
    
    Args:
        revision_plan: Revision计划
        original_code: 原始代码
        original_error: 原始错误
        task_description: 任务描述
        parameters: 任务参数
    
    Returns:
        修复后的代码
    """
    executor = RevisionExecutor()
    return executor.generate_revision_code(
        revision_plan, original_code, original_error, task_description, parameters
    )

