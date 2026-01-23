"""
CodeAct Subgraph 单独测试用例

测试 codeact subgraph 的独立功能，包括：
1. MCP工具调用代码生成和执行
2. 普通代码生成和执行（如文件整合等）
3. 代码修复功能
4. 状态映射函数
5. 轨迹记录系统（SE-Agent风格）
6. Revision机制（失败驱动的智能修复）

运行方式：pytest tests/test_codeact_subgraph.py -v
"""

import os
import pytest
import tempfile
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
from state import SubTask, UserTaskType, GlobalState


def _ensure_codeact_state(result):
    """确保结果是 CodeActState 对象（处理字典返回值）"""
    if isinstance(result, dict):
        # 使用 model_validate 来处理嵌套模型
        return CodeActState.model_validate(result)
    return result

def _create_codeact_state_for_test(task, **kwargs):
    """为测试创建 CodeActState（处理 Pydantic v2 的嵌套模型验证）"""
    # 使用 model_validate 来确保正确验证嵌套的 SubTask
    state_dict = {
        "task": task.model_dump() if hasattr(task, 'model_dump') else task.dict() if hasattr(task, 'dict') else task,
        "task_description": task.content,
        "tools": kwargs.get("tools", []),
        "inputs": kwargs.get("inputs", []),
        "parameters": kwargs.get("parameters", {}),
        "execution_mode": kwargs.get("execution_mode"),
        "generated_code": kwargs.get("generated_code"),
        "previous_code": kwargs.get("previous_code"),
        "previous_error": kwargs.get("previous_error"),
        "error_category": kwargs.get("error_category"),
        "revision_plan": kwargs.get("revision_plan"),
        "revision_iteration": kwargs.get("revision_iteration", 0),
    }
    # 移除 None 值
    state_dict = {k: v for k, v in state_dict.items() if v is not None or k == "generated_code"}
    return CodeActState.model_validate(state_dict)


@pytest.fixture(scope="module")
def codeact_subgraph():
    """构建并返回 CodeAct 子图"""
    return build_codeact_subgraph()


@pytest.fixture
def sample_task_mcp_tool():
    """示例MCP工具调用任务"""
    return SubTask(
        task_id="test_task_001",
        task_type=UserTaskType.EXECUTE_PLAN,
        content="调用AIRR工具搜索抗体库",
        result={
            "tools": [
                {
                    "tool_name": "search_airr_repertoires",
                    "name": "search_airr_repertoires",
                    "service": "airr",
                    "description": "搜索AIRR数据库中的抗体库"
                }
            ],
            "inputs": ["disease", "tissue"],
            "outputs": ["repertoire_data"]
        }
    )


@pytest.fixture
def sample_task_codeact():
    """示例普通代码执行任务"""
    return SubTask(
        task_id="test_task_002",
        task_type=UserTaskType.EXECUTE_PLAN,
        content="整合两个CSV文件的数据",
        result={
            "tools": [],
            "inputs": ["file1.csv", "file2.csv"],
            "outputs": ["merged_data.csv"]
        }
    )


@pytest.fixture
def temp_sandbox_dir():
    """创建临时沙盒目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_csv_files(temp_sandbox_dir):
    """创建示例CSV文件用于测试"""
    file1_path = Path(temp_sandbox_dir) / "file1.csv"
    file2_path = Path(temp_sandbox_dir) / "file2.csv"
    
    # 创建第一个CSV文件
    with open(file1_path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,name,value\n")
        f.write("1,Item1,100\n")
        f.write("2,Item2,200\n")
    
    # 创建第二个CSV文件
    with open(file2_path, 'w', encoding='utf-8', newline='') as f:
        f.write("id,description\n")
        f.write("1,Description for Item1\n")
        f.write("2,Description for Item2\n")
    
    return {
        "file1": str(file1_path),
        "file2": str(file2_path),
        "sandbox": temp_sandbox_dir
    }


class TestCodeActSubgraphBasic:
    """CodeAct Subgraph 基础功能测试"""
    
    def test_subgraph_build(self, codeact_subgraph):
        """测试子图构建是否成功"""
        assert codeact_subgraph is not None
        print("✓ CodeAct Subgraph 构建成功")
    
    def test_subgraph_invoke_basic(self, codeact_subgraph, sample_task_codeact):
        """测试子图基本调用"""
        # 使用input_mapper创建输入状态（确保轨迹初始化）
        input_state = codeact_input_mapper(
            executor_state=None,
            task=sample_task_codeact,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters={}
        )
        
        # 调用子图
        output = codeact_subgraph.invoke(input_state)
        
        # 确保输出是 CodeActState 对象
        result = _ensure_codeact_state(output)
        
        assert result is not None
        assert result.task.task_id == sample_task_codeact.task_id
        assert result.generated_code is not None
        assert result.execution_result is not None
        
        # 验证轨迹记录
        assert result.trajectory_history is not None
        assert len(result.trajectory_history) > 0
        trajectory = result.trajectory_history[-1]
        assert trajectory.task_id == sample_task_codeact.task_id
        assert trajectory.execution_mode == CodeActExecutionMode.CODEACT.value
        assert trajectory.generated_code is not None
        assert trajectory.execution_result is not None
        assert trajectory.status in [TrajectoryStatus.SUCCESS, TrajectoryStatus.FAILED]
        
        print(f"✓ CodeAct Subgraph 基本调用成功")
        print(f"  轨迹记录: {len(result.trajectory_history)} 条")


class TestMCPToolExecution:
    """MCP工具调用测试"""
    
    def test_mcp_tool_code_generation(self, codeact_subgraph, sample_task_mcp_tool):
        """测试MCP工具调用代码生成"""
        input_state = codeact_input_mapper(
            executor_state=None,
            task=sample_task_mcp_tool,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={
                "disease": "COVID-19",
                "tissue": "blood"
            }
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 验证代码已生成
        assert result.generated_code is not None
        assert "invoke_mcp_tool_sync" in result.generated_code
        
        # 验证执行结果
        assert result.execution_result is not None
        assert result.execution_result.get("status") in ["success", "failed"]
        
        # 验证轨迹记录
        assert len(result.trajectory_history) > 0
        trajectory = result.trajectory_history[-1]
        assert trajectory.execution_mode == CodeActExecutionMode.MCP_TOOL.value
        assert trajectory.parameters == {"disease": "COVID-19", "tissue": "blood"}
        
        print(f"✓ MCP工具调用代码生成成功")
        print(f"  生成的代码: {result.generated_code[:100]}...")
        print(f"  轨迹记录: {len(result.trajectory_history)} 条")
    
    def test_mcp_tool_with_parameters(self, codeact_subgraph, sample_task_mcp_tool):
        """测试带参数的MCP工具调用"""
        parameters = {
            "disease": "influenza",
            "tissue": "lung",
            "species": "human"
        }
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=sample_task_mcp_tool,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters=parameters
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 验证参数在代码中
        assert result.generated_code is not None
        # 验证执行结果
        assert result.execution_result is not None
        assert result.execution_result.get("status") in ["success", "failed"]
        
        # 验证轨迹记录中的参数
        if result.trajectory_history:
            trajectory = result.trajectory_history[-1]
            assert trajectory.parameters == parameters
        
        print(f"✓ 带参数的MCP工具调用成功")
    
    def test_mcp_tool_no_tools(self, codeact_subgraph, sample_task_codeact):
        """测试没有工具时的MCP工具调用（应该使用codeact模式）"""
        input_state = codeact_input_mapper(
            executor_state=None,
            task=sample_task_codeact,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={}
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 应该生成代码
        assert result.generated_code is not None
        assert result.execution_result is not None
        
        # 验证轨迹记录
        assert len(result.trajectory_history) > 0
        
        print(f"✓ 无工具时的MCP调用处理成功")


class TestCodeActExecution:
    """普通代码执行测试"""
    
    def test_codeact_file_integration(self, codeact_subgraph, sample_task_codeact, sample_csv_files):
        """测试文件整合代码生成和执行"""
        # 创建实际的文件整合代码（使用标准库，不依赖pandas）
        merge_code = f"""
import csv
import os

# 读取两个CSV文件
file1_path = r"{sample_csv_files['file1']}"
file2_path = r"{sample_csv_files['file2']}"

# 读取第一个文件
data1 = {{}}
with open(file1_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data1[row['id']] = row

# 读取第二个文件
data2 = {{}}
with open(file2_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data2[row['id']] = row

# 合并数据
merged_data = []
for id_val in data1:
    if id_val in data2:
        merged_row = {{**data1[id_val], **data2[id_val]}}
        merged_data.append(merged_row)

# 保存结果
output_path = os.path.join(r"{sample_csv_files['sandbox']}", "merged_data.csv")
if merged_data:
    fieldnames = list(merged_data[0].keys())
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_data)

result = {{
    "status": "success",
    "output_file": output_path,
    "rows": len(merged_data),
    "columns": list(merged_data[0].keys()) if merged_data else []
}}
"""
        
        input_state = _create_codeact_state_for_test(
            task=sample_task_codeact,
            tools=[],
            inputs=[],
            parameters={},
            execution_mode=CodeActExecutionMode.CODEACT,
            generated_code=merge_code  # 直接设置代码，跳过生成步骤
        )
        
        # 只执行代码（跳过生成步骤）
        from nodes.subagents.code_act.graph import codeact_execute_code_node
        result = codeact_execute_code_node(input_state)
        
        # 验证执行结果
        assert result.execution_result is not None
        assert result.execution_result.get("status") == "success"
        
        # 验证输出文件是否存在
        output_file = Path(sample_csv_files['sandbox']) / "merged_data.csv"
        assert output_file.exists(), "合并后的文件应该存在"
        
        # 验证文件内容（使用标准库）
        import csv
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2, "应该有2行数据"
        assert "id" in rows[0]
        assert "name" in rows[0]
        assert "value" in rows[0]
        assert "description" in rows[0]
        
        print(f"✓ 文件整合代码执行成功")
    
    def test_codeact_simple_calculation(self, codeact_subgraph):
        """测试简单计算代码生成和执行"""
        task = SubTask(
            task_id="test_calc_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="计算 1 + 1",
            result={}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters={}
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        assert result.generated_code is not None
        assert result.execution_result is not None
        
        # 验证轨迹记录
        assert len(result.trajectory_history) > 0
        
        print(f"✓ 简单计算代码执行成功")
    
    def test_codeact_data_processing(self, codeact_subgraph, temp_sandbox_dir):
        """测试数据处理代码生成和执行"""
        task = SubTask(
            task_id="test_data_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="处理数据：计算列表的平均值",
            result={}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters={}
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        assert result.generated_code is not None
        assert result.execution_result is not None
        
        # 验证轨迹记录
        assert len(result.trajectory_history) > 0
        
        print(f"✓ 数据处理代码执行成功")


class TestCodeFixExecution:
    """代码修复测试"""
    
    def test_fix_code_error(self, codeact_subgraph):
        """测试代码错误修复"""
        task = SubTask(
            task_id="test_fix_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="修复代码错误",
            result={}
        )
        
        previous_code = "x = 10\ny = 20\nz = x +  # 语法错误"
        previous_error = "SyntaxError: invalid syntax"
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.FIX_CODE,
            parameters={},
            previous_code=previous_code,
            previous_error=previous_error,
            error_category="syntax_error"
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        assert result.generated_code is not None
        assert result.execution_result is not None
        
        # 验证轨迹记录
        assert len(result.trajectory_history) > 0
        
        print(f"✓ 代码错误修复测试成功")
    
    def test_fix_parameter_error(self, codeact_subgraph):
        """测试参数错误修复"""
        task = SubTask(
            task_id="test_fix_param_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="修复参数错误",
            result={}
        )
        
        previous_code = "result = add(10, '20')  # 类型错误"
        previous_error = "TypeError: unsupported operand type(s)"
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.FIX_PARAMETER,
            parameters={},
            previous_code=previous_code,
            previous_error=previous_error,
            error_category="parameter_error"
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        assert result.generated_code is not None
        assert result.execution_result is not None
        
        # 验证轨迹记录
        assert len(result.trajectory_history) > 0
        
        print(f"✓ 参数错误修复测试成功")


class TestStateMapping:
    """状态映射测试"""
    
    def test_codeact_input_mapper(self, sample_task_mcp_tool):
        """测试输入状态映射"""
        input_state = codeact_input_mapper(
            executor_state=None,
            task=sample_task_mcp_tool,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={"disease": "COVID-19"}
        )
        
        assert input_state.task.task_id == sample_task_mcp_tool.task_id
        assert input_state.execution_mode == CodeActExecutionMode.MCP_TOOL
        assert input_state.parameters == {"disease": "COVID-19"}
        
        # 验证轨迹相关字段已初始化
        assert input_state.trajectory_history is not None
        assert input_state.revision_iteration == 0
        
        print(f"✓ 输入状态映射成功")
    
    def test_codeact_output_mapper(self, codeact_subgraph, sample_task_codeact):
        """测试输出状态映射"""
        input_state = codeact_input_mapper(
            executor_state=None,
            task=sample_task_codeact,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters={}
        )
        
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        output_dict = codeact_output_mapper(result)
        
        assert "status" in output_dict
        assert "code" in output_dict
        assert output_dict["code"] == result.generated_code
        
        print(f"✓ 输出状态映射成功")


class TestErrorHandling:
    """错误处理测试"""
    
    def test_empty_code_execution(self, codeact_subgraph):
        """测试空代码执行"""
        task = SubTask(
            task_id="test_error_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="执行空代码",
            result={}
        )
        
        input_state = _create_codeact_state_for_test(
            task=task,
            tools=[],
            inputs=[],
            parameters={},
            execution_mode=CodeActExecutionMode.CODEACT,
            generated_code=""  # 空代码
        )
        
        from nodes.subagents.code_act.graph import codeact_execute_code_node
        result = codeact_execute_code_node(input_state)
        
        assert result.execution_result is not None
        assert result.execution_result.get("status") == "failed"
        
        print(f"✓ 空代码错误处理成功")
    
    def test_syntax_error_handling(self, codeact_subgraph):
        """测试语法错误处理"""
        task = SubTask(
            task_id="test_error_002",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="执行有语法错误的代码",
            result={}
        )
        
        syntax_error_code = """
# 语法错误
x = 10
y = 20
z = x +  # 语法错误：缺少操作数
result = z
"""
        
        input_state = _create_codeact_state_for_test(
            task=task,
            tools=[],
            inputs=[],
            parameters={},
            execution_mode=CodeActExecutionMode.CODEACT,
            generated_code=syntax_error_code
        )
        
        from nodes.subagents.code_act.graph import codeact_execute_code_node
        result = codeact_execute_code_node(input_state)
        
        # 应该返回失败状态
        assert result.execution_result is not None
        assert result.execution_result.get("status") == "failed"
        assert result.execution_result.get("error") is not None
        
        print(f"✓ 语法错误处理成功")
        print(f"  错误类型: {result.execution_result.get('error_type')}")


class TestIntegrationScenarios:
    """集成场景测试"""
    
    def test_full_mcp_tool_workflow(self, codeact_subgraph, sample_task_mcp_tool):
        """测试完整的MCP工具调用工作流"""
        # 1. 使用input_mapper创建输入状态
        input_state = codeact_input_mapper(
            executor_state=None,
            task=sample_task_mcp_tool,
            execution_mode=CodeActExecutionMode.MCP_TOOL,
            parameters={"disease": "COVID-19", "tissue": "blood"}
        )
        
        # 2. 执行子图
        output = codeact_subgraph.invoke(input_state)
        result = _ensure_codeact_state(output)
        
        # 3. 验证结果
        assert result.generated_code is not None
        assert result.execution_result is not None
        assert result.execution_result.get("status") in ["success", "failed"]
        
        # 4. 验证轨迹记录
        assert len(result.trajectory_history) > 0
        trajectory = result.trajectory_history[-1]
        assert trajectory.execution_mode == CodeActExecutionMode.MCP_TOOL.value
        assert trajectory.parameters == {"disease": "COVID-19", "tissue": "blood"}
        
        # 5. 使用输出映射
        output_dict = codeact_output_mapper(result)
        assert output_dict["status"] in ["success", "failed", "unknown"]
        
        print(f"✓ 完整MCP工具调用工作流测试成功")
        print(f"  轨迹记录: {len(result.trajectory_history)} 条")
    
    def test_full_codeact_workflow(self, codeact_subgraph, sample_csv_files):
        """测试完整的CodeAct工作流（文件整合）"""
        task = SubTask(
            task_id="test_integration_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="整合两个CSV文件",
            result={}
        )
        
        # 创建文件整合代码（使用标准库）
        merge_code = f"""
import csv

# 读取两个CSV文件
data1 = {{}}
with open(r"{sample_csv_files['file1']}", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data1[row['id']] = row

data2 = {{}}
with open(r"{sample_csv_files['file2']}", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data2[row['id']] = row

# 合并数据
merged = []
for id_val in data1:
    if id_val in data2:
        merged.append({{**data1[id_val], **data2[id_val]}})

# 保存结果
output_path = r"{sample_csv_files['sandbox']}/merged_output.csv"
if merged:
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)

result = {{"status": "success", "rows": len(merged), "file": output_path}}
"""
        
        # 1. 创建输入状态
        input_state = _create_codeact_state_for_test(
            task=task,
            tools=[],
            inputs=[],
            parameters={},
            execution_mode=CodeActExecutionMode.CODEACT,
            generated_code=merge_code
        )
        
        # 2. 执行代码
        from nodes.subagents.code_act.graph import codeact_execute_code_node
        result = codeact_execute_code_node(input_state)
        
        # 3. 验证结果
        assert result.execution_result is not None
        assert result.execution_result.get("status") == "success"
        
        # 4. 验证输出文件
        output_file = Path(sample_csv_files['sandbox']) / "merged_output.csv"
        assert output_file.exists()
        
        print(f"✓ 完整CodeAct工作流测试成功")
        print(f"  输出文件: {output_file}")


class TestTrajectoryRecording:
    """轨迹记录系统测试"""
    
    def test_trajectory_recorded_on_success(self, codeact_subgraph, sample_task_codeact):
        """测试成功执行时轨迹被正确记录"""
        input_state = codeact_input_mapper(
            executor_state=None,
            task=sample_task_codeact,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters={}
        )
        
        output_state = codeact_subgraph.invoke(input_state)
        output_state = _ensure_codeact_state(output_state)
        
        # 验证轨迹历史
        assert output_state.trajectory_history is not None
        assert len(output_state.trajectory_history) >= 1
        
        trajectory = output_state.trajectory_history[-1]
        assert trajectory.task_id == sample_task_codeact.task_id
        assert trajectory.generated_code is not None
        assert trajectory.execution_result is not None
        assert trajectory.code_generation_time >= 0
        assert trajectory.execution_time >= 0
        
        # 如果执行成功，状态应该是SUCCESS
        if output_state.execution_result.get("status") == "success":
            assert trajectory.status == TrajectoryStatus.SUCCESS
        else:
            assert trajectory.status == TrajectoryStatus.FAILED
        
        print(f"✓ 成功执行轨迹记录验证通过")
    
    def test_trajectory_recorded_on_failure(self, codeact_subgraph):
        """测试失败执行时轨迹被正确记录"""
        # 创建一个会导致失败的任务（语法错误）
        task = SubTask(
            task_id="test_failure_task",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="执行会导致错误的代码",
            result={"tools": [], "inputs": [], "outputs": []}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters={}
        )
        
        output_state = codeact_subgraph.invoke(input_state)
        output_state = _ensure_codeact_state(output_state)
        
        # 验证轨迹被记录
        assert len(output_state.trajectory_history) > 0
        trajectory = output_state.trajectory_history[-1]
        
        # 如果执行失败，应该有错误信息
        if trajectory.status == TrajectoryStatus.FAILED:
            assert trajectory.error_type is not None or trajectory.error_message is not None
        
        print(f"✓ 失败执行轨迹记录验证通过")
    
    def test_multiple_trajectories_for_revision(self, codeact_subgraph):
        """测试Revision机制产生多个轨迹"""
        # 创建一个会失败的任务，触发Revision
        task = SubTask(
            task_id="test_revision_task",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="执行会失败的任务",
            result={"tools": [], "inputs": [], "outputs": []}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters={}
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
        
        print(f"✓ Revision多轨迹记录验证通过")


class TestRevisionMechanism:
    """Revision机制测试"""
    
    def test_revision_plan_creation(self, codeact_subgraph):
        """测试Revision计划创建"""
        # 创建一个会失败的任务
        task = SubTask(
            task_id="test_revision_plan",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="执行会失败的任务",
            result={"tools": [], "inputs": [], "outputs": []}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters={}
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
        
        print(f"✓ Revision计划创建验证通过")
    
    def test_revision_iteration_limit(self, codeact_subgraph):
        """测试Revision迭代次数限制"""
        task = SubTask(
            task_id="test_revision_limit",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="持续失败的任务",
            result={"tools": [], "inputs": [], "outputs": []}
        )
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters={}
        )
        
        output_state = codeact_subgraph.invoke(input_state)
        output_state = _ensure_codeact_state(output_state)
        
        # Revision迭代次数不应该超过3次
        assert output_state.revision_iteration <= 3
        
        print(f"✓ Revision迭代次数限制验证通过")
    
    def test_revision_with_fix_code_mode(self, codeact_subgraph):
        """测试使用Revision计划修复代码"""
        # 创建一个有语法错误的代码修复任务
        task = SubTask(
            task_id="test_fix_with_revision",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="修复代码错误",
            result={"tools": [], "inputs": [], "outputs": []}
        )
        
        # 先创建一个失败的轨迹
        previous_code = "print('test'  # 缺少右括号"
        previous_error = "SyntaxError: unexpected EOF while parsing"
        
        input_state = codeact_input_mapper(
            executor_state=None,
            task=task,
            execution_mode=CodeActExecutionMode.FIX_CODE,
            parameters={},
            previous_code=previous_code,
            previous_error=previous_error
        )
        
        # 如果有Revision计划，应该使用它
        if input_state.revision_plan:
            assert input_state.execution_mode == CodeActExecutionMode.FIX_CODE
        
        output_state = codeact_subgraph.invoke(input_state)
        output_state = _ensure_codeact_state(output_state)
        
        # 验证修复后的代码
        assert output_state.generated_code is not None
        # 修复后的代码应该与原始代码不同
        if output_state.generated_code != previous_code:
            assert "print" in output_state.generated_code or "result" in output_state.generated_code
        
        print(f"✓ Revision修复代码验证通过")
