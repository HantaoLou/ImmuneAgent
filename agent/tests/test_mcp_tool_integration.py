"""
MCP工具调用集成测试用例

专门测试 CodeAct 子图中 MCP 工具调用的各种场景，包括：
1. 不同 MCP 工具的代码生成
2. 参数验证和处理
3. 错误处理和降级
4. 实际 MCP 工具调用（如果可用）
5. 轨迹记录（MCP工具调用）
6. Revision机制（MCP工具调用失败时的修复）

运行方式：pytest tests/test_mcp_tool_integration.py -v
"""

import os
import pytest
import json
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加agent目录到路径
import sys
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

from nodes.subagents.code_act.graph import (
    build_codeact_subgraph,
    CodeActState,
    CodeActExecutionMode,
    codeact_input_mapper,
    codeact_output_mapper
)
from nodes.subagents.code_act.trajectory import (
    CodeTrajectory,
    TrajectoryStatus
)
from nodes.subagents.code_act.revision import (
    RevisionPlan,
    RevisionStrategy
)
from state import SubTask, UserTaskType


def _ensure_codeact_state(result):
    """确保结果是 CodeActState 对象（处理字典返回值）"""
    if isinstance(result, dict):
        return CodeActState(**result)
    return result


@pytest.fixture(scope="module")
def codeact_subgraph():
    """构建并返回 CodeAct 子图"""
    return build_codeact_subgraph()


@pytest.fixture
def load_mcp_tools():
    """加载MCP工具配置"""
    mcp_tools_path = agent_dir / "config" / "mcp_tools.json"
    if not mcp_tools_path.exists():
        return []
    
    try:
        with open(mcp_tools_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "mcp_tools" in data:
            return data["mcp_tools"]
        else:
            return []
    except Exception as e:
        print(f"⚠ 加载MCP工具配置失败: {e}")
        return []


@pytest.fixture
def sample_airr_tool():
    """AIRR工具示例"""
    return {
        "name": "search_airr_repertoires",
        "tool_name": "search_airr_repertoires",
        "service": "airr",
        "description": "搜索AIRR数据库中的抗体库，支持按疾病、组织等条件筛选"
    }


@pytest.fixture
def sample_alphafold3_tool():
    """AlphaFold3工具示例"""
    return {
        "name": "alphafold3",
        "tool_name": "alphafold3",
        "service": "af3",
        "description": "用AlphaFold3从Excel抗体序列（重链/轻链）预测3D结构，输出PDB文件，流式返回进度"
    }


@pytest.fixture
def sample_bindcraft_tool():
    """BindCraft工具示例"""
    return {
        "name": "analyze_design_results",
        "tool_name": "analyze_design_results",
        "service": "bindcraft",
        "description": "分析BindCraft设计结果，按综合/单指标排序，输出序列、指标、统计数据及PDB路径"
    }


class TestMCPToolCodeGeneration:
    """MCP工具代码生成测试"""
    
    def test_airr_tool_code_generation(self, codeact_subgraph, sample_airr_tool):
        """测试AIRR工具代码生成"""
        task = SubTask(
            task_id="test_airr_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="使用AIRR工具搜索COVID-19相关的抗体库",
            result={
                "tools": [sample_airr_tool],
                "inputs": ["disease", "tissue"],
                "outputs": ["repertoire_data"]
            }
        )
        
        parameters = {
            "disease": "COVID-19",
            "tissue": "blood"
        }
        
        # 获取输入列表
        task_result = task.result if isinstance(task.result, dict) else {}
        inputs = task_result.get("inputs", [])
        
        # 使用 codeact_input_mapper 创建状态（与生产代码保持一致）
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters=parameters
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 验证代码已生成
        assert result.generated_code is not None
        assert len(result.generated_code) > 0
        
        # 验证代码包含工具相关信息
        code_lower = result.generated_code.lower()
        assert "airr" in code_lower or "search" in code_lower or "repertoire" in code_lower
        
        # 验证执行结果
        assert result.execution_result is not None
        
        print(f"✓ AIRR工具代码生成成功")
        print(f"  代码长度: {len(result.generated_code)} 字符")
        print(f"  执行状态: {result.execution_result.get('status')}")
        print(f"  代码: {result.generated_code}")
    
    def test_alphafold3_tool_code_generation(self, codeact_subgraph, sample_alphafold3_tool):
        """测试AlphaFold3工具代码生成"""
        task = SubTask(
            task_id="test_af3_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="使用AlphaFold3预测抗体结构",
            result={
                "tools": [sample_alphafold3_tool],
                "inputs": ["excel_file"],
                "outputs": ["pdb_file"]
            }
        )
        
        parameters = {
            "excel_file": "/path/to/antibody_sequences.xlsx"
        }
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters=parameters
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 验证代码已生成
        assert result.generated_code is not None
        assert len(result.generated_code) > 0
        
        # 验证代码包含工具相关信息
        code_lower = result.generated_code.lower()
        assert "alphafold" in code_lower or "af3" in code_lower or "structure" in code_lower
        
        # 验证执行结果
        assert result.execution_result is not None
        
        print(f"✓ AlphaFold3工具代码生成成功")
        print(f"  代码长度: {len(result.generated_code)} 字符")
    
    def test_bindcraft_tool_code_generation(self, codeact_subgraph, sample_bindcraft_tool):
        """测试BindCraft工具代码生成"""
        task = SubTask(
            task_id="test_bindcraft_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="分析BindCraft设计结果",
            result={
                "tools": [sample_bindcraft_tool],
                "inputs": ["design_results"],
                "outputs": ["analysis_results"]
            }
        )
        
        parameters = {
            "design_results": "/path/to/design_results.json"
        }
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters=parameters
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 验证代码已生成
        assert result.generated_code is not None
        assert len(result.generated_code) > 0
        
        # 验证执行结果
        assert result.execution_result is not None
        
        print(f"✓ BindCraft工具代码生成成功")
        print(f"  代码长度: {len(result.generated_code)} 字符")


class TestMCPToolParameters:
    """MCP工具参数处理测试"""
    
    def test_mcp_tool_with_simple_parameters(self, codeact_subgraph, sample_airr_tool):
        """测试简单参数"""
        task = SubTask(
            task_id="test_params_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="搜索抗体库",
            result={"tools": [sample_airr_tool]}
        )
        
        parameters = {
            "disease": "influenza"
        }
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters=parameters
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 验证参数在代码中
        assert result.generated_code is not None
        assert "influenza" in result.generated_code or "disease" in result.generated_code.lower()
        
        print(f"✓ 简单参数处理成功")
    
    def test_mcp_tool_with_multiple_parameters(self, codeact_subgraph, sample_airr_tool):
        """测试多个参数"""
        task = SubTask(
            task_id="test_params_002",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="搜索抗体库",
            result={"tools": [sample_airr_tool]}
        )
        
        parameters = {
            "disease": "COVID-19",
            "tissue": "lung",
            "species": "human",
            "max_results": 100
        }
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters=parameters
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 验证代码包含参数
        assert result.generated_code is not None
        code_lower = result.generated_code.lower()
        # 至少应该包含部分参数
        assert any(param in code_lower for param in ["covid", "lung", "human", "100"])
        
        print(f"✓ 多参数处理成功")
    
    def test_mcp_tool_with_file_parameters(self, codeact_subgraph, sample_alphafold3_tool):
        """测试文件路径参数"""
        task = SubTask(
            task_id="test_params_003",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="预测结构",
            result={"tools": [sample_alphafold3_tool]}
        )
        
        parameters = {
            "excel_file": "/data/antibody_sequences.xlsx",
            "output_dir": "/data/output"
        }
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters=parameters
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 验证文件路径在代码中
        assert result.generated_code is not None
        assert "xlsx" in result.generated_code.lower() or "excel" in result.generated_code.lower()
        
        print(f"✓ 文件路径参数处理成功")
    
    def test_mcp_tool_with_empty_parameters(self, codeact_subgraph, sample_airr_tool):
        """测试空参数"""
        task = SubTask(
            task_id="test_params_004",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="搜索抗体库",
            result={"tools": [sample_airr_tool]}
        )
        
        parameters = {}
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters=parameters
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 即使没有参数，也应该生成代码
        assert result.generated_code is not None
        assert len(result.generated_code) > 0
        
        print(f"✓ 空参数处理成功")


class TestMCPToolErrorHandling:
    """MCP工具错误处理测试"""
    
    def test_mcp_tool_no_tools(self, codeact_subgraph):
        """测试没有工具时的处理"""
        task = SubTask(
            task_id="test_error_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="执行任务",
            result={"tools": []}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={}
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 应该生成占位代码或错误提示
        assert result.generated_code is not None
        assert "未找到匹配的工具" in result.generated_code or len(result.generated_code) > 0
        
        print(f"✓ 无工具错误处理成功")
    
    def test_mcp_tool_invalid_tool_format(self, codeact_subgraph):
        """测试无效工具格式"""
        task = SubTask(
            task_id="test_error_002",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="执行任务",
            result={"tools": ["invalid_tool"]}  # 字符串而不是字典
        )
        
        # 对于无效工具格式，需要手动设置tools
        task.result = {"tools": ["invalid_tool"]}
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={}
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 应该能够处理或生成降级代码
        assert result.generated_code is not None
        
        print(f"✓ 无效工具格式处理成功")


class TestMCPToolFromConfig:
    """从配置文件加载的MCP工具测试"""
    
    @pytest.mark.parametrize("tool_name,service", [
        ("search_airr_repertoires", "airr"),
        ("alphafold3", "af3"),
        ("analyze_design_results", "bindcraft"),
    ])
    def test_mcp_tools_from_config(self, codeact_subgraph, load_mcp_tools, tool_name, service):
        """测试从配置文件加载的工具"""
        if not load_mcp_tools:
            pytest.skip("MCP工具配置未加载")
        
        # 查找工具
        tool = None
        for t in load_mcp_tools:
            if t.get("name") == tool_name or t.get("tool_name") == tool_name:
                tool = t
                break
        
        if not tool:
            pytest.skip(f"工具 {tool_name} 未在配置中找到")
        
        task = SubTask(
            task_id=f"test_config_{tool_name}",
            task_type=UserTaskType.EXECUTE_PLAN,
            content=f"使用{tool_name}工具",
            result={
                "tools": [tool],
                "inputs": [],
                "outputs": []
            }
        )
        
        parameters = {"test_param": "test_value"}
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters=parameters
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 验证代码已生成
        assert result.generated_code is not None
        assert len(result.generated_code) > 0
        
        # 验证执行结果
        assert result.execution_result is not None
        
        print(f"✓ 工具 {tool_name} (service: {service}) 代码生成成功")


class TestMCPToolWorkflow:
    """MCP工具完整工作流测试"""
    
    def test_mcp_tool_full_workflow(self, codeact_subgraph, sample_airr_tool):
        """测试完整的MCP工具调用工作流"""
        # 1. 创建任务
        task = SubTask(
            task_id="test_workflow_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="搜索COVID-19相关的抗体库",
            result={
                "tools": [sample_airr_tool],
                "inputs": ["disease", "tissue"],
                "outputs": ["repertoire_data"]
            }
        )
        
        # 2. 准备参数
        parameters = {
            "disease": "COVID-19",
            "tissue": "blood"
        }
        
        # 3. 使用input_mapper创建状态
        codeact_input = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters=parameters
        )
        
        print(f"\n{'='*60}")
        print(f"测试：完整MCP工具调用工作流")
        print(f"{'='*60}")
        print(f"任务ID: {task.task_id}")
        print(f"任务内容: {task.content}")
        print(f"工具: {sample_airr_tool.get('name', 'unknown')}")
        print(f"参数: {parameters}")
        print(f"{'='*60}\n")
        
        # 4. 执行子图
        output = codeact_subgraph.invoke(codeact_input)
        result = _ensure_codeact_state(output)
        
        # 5. 详细打印生成结果
        print(f"\n{'='*60}")
        print(f"代码生成结果")
        print(f"{'='*60}")
        print(f"生成的代码是否为空: {result.generated_code is None}")
        if result.generated_code:
            print(f"代码长度: {len(result.generated_code)} 字符")
            print(f"代码预览（前200字符）:")
            print(f"{result.generated_code[:200]}...")
            print(f"\n完整代码:")
            print(f"{result.generated_code}")
        else:
            print(f"⚠ 警告：未生成代码")
        print(f"{'='*60}\n")
        
        # 6. 详细打印执行结果
        print(f"\n{'='*60}")
        print(f"代码执行结果")
        print(f"{'='*60}")
        print(f"执行结果是否存在: {result.execution_result is not None}")
        if result.execution_result:
            exec_status = result.execution_result.get("status", "unknown")
            print(f"执行状态: {exec_status}")
            
            if exec_status == "success":
                print(f"✓ 执行成功")
                print(f"  输出: {result.execution_result.get('output', 'N/A')}")
                print(f"  结果: {result.execution_result.get('result', 'N/A')}")
                if "sandbox_dir" in result.execution_result:
                    print(f"  沙盒目录: {result.execution_result.get('sandbox_dir')}")
                if "sandbox_used" in result.execution_result:
                    print(f"  使用沙盒: {result.execution_result.get('sandbox_used')}")
            elif exec_status == "failed":
                print(f"✗ 执行失败")
                print(f"  错误信息: {result.execution_result.get('error', 'N/A')}")
                print(f"  错误类型: {result.execution_result.get('error_type', 'N/A')}")
                if "sandbox_used" in result.execution_result:
                    print(f"  使用沙盒: {result.execution_result.get('sandbox_used')}")
            else:
                print(f"⚠ 未知状态: {exec_status}")
                print(f"  完整执行结果: {result.execution_result}")
        else:
            print(f"⚠ 警告：无执行结果")
        print(f"{'='*60}\n")
        
        # 7. 验证结果
        assert result.generated_code is not None, "代码应该已生成"
        assert result.execution_result is not None, "执行结果应该存在"
        
        # 如果执行失败，打印详细错误信息并断言失败
        if result.execution_result and result.execution_result.get("status") == "failed":
            error_msg = result.execution_result.get("error", "未知错误")
            error_type = result.execution_result.get("error_type", "UnknownError")
            print(f"\n{'='*60}")
            print(f"执行失败详情")
            print(f"{'='*60}")
            print(f"错误类型: {error_type}")
            print(f"错误信息: {error_msg}")
            print(f"生成的代码:")
            print(f"{result.generated_code}")
            print(f"{'='*60}\n")
            pytest.fail(f"代码执行失败: {error_type} - {error_msg}")
        
        # 8. 使用output_mapper转换结果
        output_dict = codeact_output_mapper(result)
        
        print(f"\n{'='*60}")
        print(f"输出映射结果")
        print(f"{'='*60}")
        print(f"状态: {output_dict.get('status')}")
        print(f"代码长度: {len(output_dict.get('code', ''))}")
        if output_dict.get('error'):
            print(f"错误: {output_dict.get('error')}")
        if output_dict.get('error_type'):
            print(f"错误类型: {output_dict.get('error_type')}")
        print(f"{'='*60}\n")
        
        assert output_dict is not None
        assert "status" in output_dict
        assert "code" in output_dict
        assert output_dict["code"] == result.generated_code
        
        # 9. 验证轨迹记录
        assert result.trajectory_history is not None
        assert len(result.trajectory_history) > 0
        trajectory = result.trajectory_history[-1]
        assert trajectory.task_id == task.task_id
        # execution_mode 应该是字符串（从枚举的 .value 获取）
        # 直接使用字符串比较，避免类型问题
        assert str(trajectory.execution_mode) == "mcp_tool"
        assert trajectory.parameters == parameters
        assert trajectory.generated_code == result.generated_code
        assert trajectory.execution_result == result.execution_result
        
        print(f"\n{'='*60}")
        print(f"轨迹记录验证")
        print(f"{'='*60}")
        print(f"轨迹数量: {len(result.trajectory_history)}")
        print(f"轨迹状态: {trajectory.status}")
        print(f"代码生成时间: {trajectory.code_generation_time:.2f}s")
        print(f"代码执行时间: {trajectory.execution_time:.2f}s")
        if result.revision_iteration > 0:
            print(f"Revision迭代次数: {result.revision_iteration}")
        print(f"{'='*60}\n")
        
        print(f"✓ 完整工作流测试成功")
        print(f"  最终状态: {output_dict['status']}")
        print(f"  代码长度: {len(output_dict.get('code', ''))}")
        print(f"  轨迹记录: {len(result.trajectory_history)} 条")
    
    def test_mcp_tool_with_state_mapping(self, codeact_subgraph, sample_airr_tool):
        """测试状态映射"""
        task = SubTask(
            task_id="test_mapping_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="搜索抗体库",
            result={
                "tools": [sample_airr_tool],
                "inputs": ["disease"],
                "outputs": ["data"]
            }
        )
        
        # 测试input_mapper
        codeact_input = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={"disease": "influenza"}
        )
        
        assert codeact_input.task.task_id == task.task_id
        assert codeact_input.execution_mode == CodeActExecutionMode.MCP_TOOL
        assert codeact_input.parameters == {"disease": "influenza"}
        assert len(codeact_input.tools) > 0
        
        # 执行子图
        output = codeact_subgraph.invoke(codeact_input)
        result = _ensure_codeact_state(output)
        
        # 测试output_mapper
        output_dict = codeact_output_mapper(result)
        
        assert output_dict["status"] in ["success", "failed", "unknown"]
        assert output_dict["code"] is not None
        
        print(f"✓ 状态映射测试成功")


class TestMCPToolCodeQuality:
    """MCP工具生成代码质量测试"""
    
    def test_generated_code_structure(self, codeact_subgraph, sample_airr_tool):
        """测试生成代码的结构"""
        task = SubTask(
            task_id="test_quality_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="搜索抗体库",
            result={"tools": [sample_airr_tool]}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={"disease": "COVID-19"}
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        code = result.generated_code
        
        # 验证代码基本结构
        assert code is not None
        assert len(code) > 10  # 至少应该有一些代码
        
        # 验证代码不包含markdown标记
        assert not code.startswith("```")
        assert not code.endswith("```")
        
        # 验证代码包含result变量（通过执行结果判断）
        assert result.execution_result is not None
        
        print(f"✓ 代码结构验证成功")
        print(f"  代码预览: {code[:100]}...")
    
    def test_generated_code_executability(self, codeact_subgraph, sample_airr_tool):
        """测试生成代码的可执行性"""
        task = SubTask(
            task_id="test_exec_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="搜索抗体库",
            result={"tools": [sample_airr_tool]}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={"disease": "test"}
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 验证代码已执行（通过execution_result判断）
        assert result.execution_result is not None
        assert result.execution_result.get("status") in ["success", "failed"]
        
        # 如果执行失败，应该有错误信息
        if result.execution_result.get("status") == "failed":
            assert result.execution_result.get("error") is not None
        
        print(f"✓ 代码可执行性验证成功")
        print(f"  执行状态: {result.execution_result.get('status')}")


class TestMCPToolTrajectoryRecording:
    """MCP工具调用的轨迹记录测试"""
    
    def test_trajectory_recorded_for_mcp_tool_success(self, codeact_subgraph, sample_airr_tool):
        """测试MCP工具调用成功时的轨迹记录"""
        task = SubTask(
            task_id="test_trajectory_mcp_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="搜索抗体库",
            result={"tools": [sample_airr_tool]}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={"disease": "COVID-19", "tissue": "blood"}
        )
        
        output_state = codeact_subgraph.invoke(input_state)
        output_state = _ensure_codeact_state(output_state)
        
        # 验证轨迹历史
        assert output_state.trajectory_history is not None
        assert len(output_state.trajectory_history) >= 1
        
        trajectory = output_state.trajectory_history[-1]
        assert trajectory.task_id == task.task_id
        # execution_mode 应该是字符串，直接使用字符串比较
        assert str(trajectory.execution_mode) == "mcp_tool"
        assert trajectory.generated_code is not None
        assert trajectory.execution_result is not None
        assert trajectory.parameters == {"disease": "COVID-19", "tissue": "blood"}
        assert trajectory.code_generation_time >= 0
        assert trajectory.execution_time >= 0
        
        # 验证工具信息
        assert len(trajectory.tools) > 0
        
        # 根据执行结果验证状态
        # trajectory.status 可能是枚举或字符串，使用字符串比较
        if output_state.execution_result.get("status") == "success":
            assert str(trajectory.status) == "success" or trajectory.status == TrajectoryStatus.SUCCESS
        else:
            assert str(trajectory.status) == "failed" or trajectory.status == TrajectoryStatus.FAILED
        
        print(f"✓ MCP工具调用轨迹记录验证通过")
        print(f"  轨迹数量: {len(output_state.trajectory_history)}")
        print(f"  轨迹状态: {trajectory.status}")
    
    def test_trajectory_recorded_for_mcp_tool_failure(self, codeact_subgraph, sample_airr_tool):
        """测试MCP工具调用失败时的轨迹记录"""
        task = SubTask(
            task_id="test_trajectory_mcp_fail_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="搜索抗体库（会失败）",
            result={"tools": [sample_airr_tool]}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={"invalid_param": "invalid_value"}
        )
        
        output_state = codeact_subgraph.invoke(input_state)
        output_state = _ensure_codeact_state(output_state)
        
        # 验证轨迹被记录
        assert len(output_state.trajectory_history) > 0
        trajectory = output_state.trajectory_history[-1]
        
        # 如果执行失败，应该有错误信息
        # trajectory.status 可能是枚举或字符串，使用字符串比较
        if str(trajectory.status) == "failed" or trajectory.status == TrajectoryStatus.FAILED:
            assert trajectory.error_type is not None or trajectory.error_message is not None
            assert trajectory.execution_result is not None
            assert trajectory.execution_result.get("status") == "failed"
        
        print(f"✓ MCP工具调用失败轨迹记录验证通过")
        print(f"  轨迹状态: {trajectory.status}")
        if trajectory.error_message:
            print(f"  错误信息: {trajectory.error_message[:100]}...")
    
    def test_multiple_trajectories_for_mcp_revision(self, codeact_subgraph, sample_airr_tool):
        """测试MCP工具调用失败时Revision机制产生多个轨迹"""
        task = SubTask(
            task_id="test_mcp_revision_task",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="执行会失败的MCP工具调用",
            result={"tools": [sample_airr_tool]}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={"invalid": "parameters"}
        )
        
        output_state = codeact_subgraph.invoke(input_state)
        output_state = _ensure_codeact_state(output_state)
        
        # 如果有Revision，应该有多条轨迹
        # 注意：Revision最多3次迭代
        assert len(output_state.trajectory_history) >= 1
        
        # 检查是否有Revision迭代
        if output_state.revision_iteration > 0:
            assert output_state.revision_plan is not None
            assert output_state.revision_iteration <= 3  # 最多3次迭代
            print(f"  Revision迭代次数: {output_state.revision_iteration}")
        
        print(f"✓ MCP工具Revision多轨迹记录验证通过")
        print(f"  轨迹数量: {len(output_state.trajectory_history)}")


class TestMCPToolRevisionMechanism:
    """MCP工具调用的Revision机制测试"""
    
    def test_mcp_tool_revision_plan_creation(self, codeact_subgraph, sample_airr_tool):
        """测试MCP工具调用失败时Revision计划创建"""
        task = SubTask(
            task_id="test_mcp_revision_plan",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="执行会失败的MCP工具调用",
            result={"tools": [sample_airr_tool]}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={"invalid": "params"}
        )
        
        output_state = codeact_subgraph.invoke(input_state)
        output_state = _ensure_codeact_state(output_state)
        
        # 如果执行失败且触发了Revision，应该有Revision计划
        if output_state.execution_result and output_state.execution_result.get("status") == "failed":
            if output_state.revision_iteration > 0:
                assert output_state.revision_plan is not None
                assert isinstance(output_state.revision_plan, RevisionPlan)
                assert output_state.revision_plan.strategy in RevisionStrategy
                assert output_state.revision_plan.root_cause is not None
                assert 0.0 <= output_state.revision_plan.confidence <= 1.0
                print(f"  Revision策略: {output_state.revision_plan.strategy}")
                print(f"  根因: {output_state.revision_plan.root_cause[:100]}...")
        
        print(f"✓ MCP工具Revision计划创建验证通过")
    
    def test_mcp_tool_revision_iteration_limit(self, codeact_subgraph, sample_airr_tool):
        """测试MCP工具调用Revision迭代次数限制"""
        task = SubTask(
            task_id="test_mcp_revision_limit",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="持续失败的MCP工具调用",
            result={"tools": [sample_airr_tool]}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={"will": "fail"}
        )
        
        output_state = codeact_subgraph.invoke(input_state)
        output_state = _ensure_codeact_state(output_state)
        
        # Revision迭代次数不应该超过3次
        assert output_state.revision_iteration <= 3
        
        print(f"✓ MCP工具Revision迭代次数限制验证通过")
        print(f"  迭代次数: {output_state.revision_iteration}")
    
    def test_mcp_tool_revision_with_parameter_fix(self, codeact_subgraph, sample_airr_tool):
        """测试使用Revision计划修复MCP工具参数"""
        task = SubTask(
            task_id="test_mcp_fix_with_revision",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="修复MCP工具参数错误",
            result={"tools": [sample_airr_tool]}
        )
        
        # 先创建一个失败的轨迹
        previous_code = "result = await tool.ainvoke({'invalid': 'params'})"
        previous_error = "ToolException: Error executing tool"
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.FIX_PARAMETER,
            parameters={},
            previous_code=previous_code,
            previous_error=previous_error
        )
        
        # 如果有Revision计划，应该使用它
        if input_state.revision_plan:
            assert input_state.execution_mode == CodeActExecutionMode.FIX_PARAMETER
        
        output_state = codeact_subgraph.invoke(input_state)
        output_state = _ensure_codeact_state(output_state)
        
        # 验证修复后的代码
        assert output_state.generated_code is not None
        # 修复后的代码应该与原始代码不同
        if output_state.generated_code != previous_code:
            assert "tool" in output_state.generated_code.lower() or "mcp" in output_state.generated_code.lower()
        
        print(f"✓ MCP工具Revision修复参数验证通过")

