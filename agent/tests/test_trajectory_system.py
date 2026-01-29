"""
轨迹记录系统测试

测试 SE-Agent 风格的轨迹记录功能。
"""

import pytest
from pathlib import Path
import json
import tempfile
import shutil
from datetime import datetime

# 添加agent目录到路径
import sys
from pathlib import Path
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

# 导入轨迹相关模块
from nodes.subagents.code_act.trajectory import (
    CodeTrajectory,
    TrajectoryPool,
    TrajectoryStatus
)
from nodes.subagents.code_act.graph import (
    CodeActState,
    CodeActExecutionMode,
    _start_trajectory,
    _update_trajectory_code,
    _finalize_trajectory,
    _save_trajectory_to_pool
)
from state import SubTask, UserTaskType


class TestCodeTrajectory:
    """测试 CodeTrajectory 模型"""
    
    def test_create_trajectory(self):
        """测试创建轨迹"""
        trajectory = CodeTrajectory(
            trajectory_id="test_001",
            task_id="task_001",
            execution_mode="mcp_tool",
            generated_code="print('test')",
            status=TrajectoryStatus.SUCCESS,
            parameters={"param1": "value1"},
            tools=[{"name": "test_tool"}],
            inputs=["input1"]
        )
        
        assert trajectory.trajectory_id == "test_001"
        assert trajectory.task_id == "task_001"
        assert trajectory.execution_mode == "mcp_tool"
        assert trajectory.generated_code == "print('test')"
        assert trajectory.status == TrajectoryStatus.SUCCESS
        assert trajectory.parameters == {"param1": "value1"}
        assert len(trajectory.tools) == 1
        assert trajectory.inputs == ["input1"]
    
    def test_trajectory_serialization(self):
        """测试轨迹序列化和反序列化"""
        trajectory = CodeTrajectory(
            trajectory_id="test_002",
            task_id="task_002",
            execution_mode="codeact",
            generated_code="result = {'status': 'success'}",
            status=TrajectoryStatus.SUCCESS,
            execution_result={"status": "success", "output": "test"},
            execution_time=1.5,
            parameters={"x": 1, "y": 2}
        )
        
        # 序列化
        traj_dict = trajectory.to_dict()
        assert isinstance(traj_dict, dict)
        assert traj_dict["trajectory_id"] == "test_002"
        assert traj_dict["status"] == "success"
        
        # 反序列化
        restored = CodeTrajectory.from_dict(traj_dict)
        assert restored.trajectory_id == trajectory.trajectory_id
        assert restored.task_id == trajectory.task_id
        assert restored.generated_code == trajectory.generated_code
        assert restored.status == trajectory.status
    
    def test_trajectory_hash(self):
        """测试轨迹哈希生成"""
        trajectory = CodeTrajectory(
            trajectory_id="test_003",
            task_id="task_003",
            execution_mode="mcp_tool",
            generated_code="print('test')",
            status=TrajectoryStatus.SUCCESS,
            parameters={"a": 1}
        )
        
        hash1 = trajectory.get_hash()
        assert isinstance(hash1, str)
        assert len(hash1) > 0
        
        # 相同内容的轨迹应该有相同的哈希
        trajectory2 = CodeTrajectory(
            trajectory_id="test_004",  # ID不同
            task_id="task_003",
            execution_mode="mcp_tool",
            generated_code="print('test')",
            status=TrajectoryStatus.SUCCESS,
            parameters={"a": 1}
        )
        hash2 = trajectory2.get_hash()
        assert hash1 == hash2  # 内容相同，哈希应该相同
    
    def test_trajectory_similarity(self):
        """测试轨迹相似度判断"""
        traj1 = CodeTrajectory(
            trajectory_id="test_005",
            task_id="task_005",
            execution_mode="mcp_tool",
            generated_code="result = call_tool('tool1', {'param': 'value'})",
            status=TrajectoryStatus.SUCCESS,
            parameters={"param": "value"}
        )
        
        traj2 = CodeTrajectory(
            trajectory_id="test_006",
            task_id="task_005",
            execution_mode="mcp_tool",
            generated_code="result = call_tool('tool1', {'param': 'value'})",
            status=TrajectoryStatus.SUCCESS,
            parameters={"param": "value"}
        )
        
        # 相同内容的轨迹应该相似
        assert traj1.is_similar_to(traj2, threshold=0.8)
        
        # 不同内容的轨迹应该不相似
        traj3 = CodeTrajectory(
            trajectory_id="test_007",
            task_id="task_007",
            execution_mode="codeact",
            generated_code="print('different code')",
            status=TrajectoryStatus.SUCCESS,
            parameters={"different": "params"}
        )
        assert not traj1.is_similar_to(traj3, threshold=0.8)


class TestTrajectoryPool:
    """测试 TrajectoryPool 类"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_create_pool(self, temp_dir):
        """测试创建轨迹池"""
        pool = TrajectoryPool(pool_id="test_pool", storage_dir=temp_dir)
        
        assert pool.pool_id == "test_pool"
        assert pool.storage_dir == temp_dir
        assert len(pool.trajectories) == 0
    
    def test_add_trajectory(self, temp_dir):
        """测试添加轨迹到池中"""
        pool = TrajectoryPool(pool_id="test_pool", storage_dir=temp_dir)
        
        trajectory = CodeTrajectory(
            trajectory_id="",
            task_id="task_001",
            execution_mode="mcp_tool",
            generated_code="print('test')",
            status=TrajectoryStatus.SUCCESS
        )
        
        traj_id = pool.add_trajectory(trajectory)
        
        assert traj_id is not None
        assert len(pool.trajectories) == 1
        assert pool.trajectories[0].trajectory_id == traj_id
    
    def test_get_trajectories_by_task(self, temp_dir):
        """测试根据任务ID获取轨迹"""
        pool = TrajectoryPool(pool_id="test_pool", storage_dir=temp_dir)
        
        # 添加多个轨迹
        for i in range(3):
            traj = CodeTrajectory(
                trajectory_id=f"traj_{i}",
                task_id="task_001",
                execution_mode="mcp_tool",
                generated_code=f"code_{i}",
                status=TrajectoryStatus.SUCCESS
            )
            pool.add_trajectory(traj)
        
        # 添加另一个任务的轨迹
        traj_other = CodeTrajectory(
            trajectory_id="traj_other",
            task_id="task_002",
            execution_mode="codeact",
            generated_code="other_code",
            status=TrajectoryStatus.SUCCESS
        )
        pool.add_trajectory(traj_other)
        
        # 获取 task_001 的轨迹
        task_trajectories = pool.get_trajectories_by_task("task_001")
        assert len(task_trajectories) == 3
        assert all(t.task_id == "task_001" for t in task_trajectories)
    
    def test_get_successful_trajectories(self, temp_dir):
        """测试获取成功的轨迹"""
        pool = TrajectoryPool(pool_id="test_pool", storage_dir=temp_dir)
        
        # 添加成功和失败的轨迹
        for status in [TrajectoryStatus.SUCCESS, TrajectoryStatus.FAILED]:
            traj = CodeTrajectory(
                trajectory_id=f"traj_{status.value}",
                task_id="task_001",
                execution_mode="mcp_tool",
                generated_code="code",
                status=status
            )
            pool.add_trajectory(traj)
        
        successful = pool.get_successful_trajectories()
        assert len(successful) == 1
        assert successful[0].status == TrajectoryStatus.SUCCESS
    
    def test_save_and_load_pool(self, temp_dir):
        """测试保存和加载轨迹池"""
        pool = TrajectoryPool(pool_id="test_pool", storage_dir=temp_dir)
        
        # 添加一些轨迹
        for i in range(3):
            traj = CodeTrajectory(
                trajectory_id=f"traj_{i}",
                task_id="task_001",
                execution_mode="mcp_tool",
                generated_code=f"code_{i}",
                status=TrajectoryStatus.SUCCESS,
                execution_result={"status": "success", "output": f"output_{i}"}
            )
            pool.add_trajectory(traj)
        
        # 保存（压缩）
        saved_path = pool.save(compressed=True)
        assert saved_path.exists()
        assert saved_path.suffix == ".gz"
        
        # 加载
        loaded_pool = TrajectoryPool.load(saved_path)
        assert loaded_pool.pool_id == pool.pool_id
        assert len(loaded_pool.trajectories) == 3
        assert loaded_pool.trajectories[0].trajectory_id == "traj_0"
    
    def test_pool_statistics(self, temp_dir):
        """测试轨迹池统计信息"""
        pool = TrajectoryPool(pool_id="test_pool", storage_dir=temp_dir)
        
        # 添加成功和失败的轨迹
        for i, status in enumerate([TrajectoryStatus.SUCCESS, TrajectoryStatus.FAILED, TrajectoryStatus.SUCCESS]):
            traj = CodeTrajectory(
                trajectory_id=f"traj_{i}",
                task_id="task_001",
                execution_mode="mcp_tool",
                generated_code="code",
                status=status
            )
            pool.add_trajectory(traj)
        
        stats = pool.get_statistics()
        assert stats["total_trajectories"] == 3
        assert stats["successful"] == 2
        assert stats["failed"] == 1
        assert stats["success_rate"] == pytest.approx(2/3, rel=0.01)


class TestTrajectoryIntegration:
    """测试轨迹记录与 CodeAct 子图的集成"""
    
    @pytest.fixture
    def sample_task(self):
        """创建示例任务"""
        return SubTask(
            task_id="test_task_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="测试任务：调用MCP工具",
            result={
                "tools": [{"name": "test_tool", "tool_name": "test_tool"}],
                "inputs": ["param1"],
                "outputs": ["result"]
            }
        )
    
    @pytest.fixture
    def codeact_state(self, sample_task):
        """创建 CodeAct 状态"""
        return CodeActState(
            task=sample_task,
            task_description=sample_task.content,
            tools=[{"name": "test_tool"}],
            inputs=["param1"],
            parameters={"param1": "value1"},
            execution_mode=CodeActExecutionMode.MCP_TOOL
        )
    
    def test_start_trajectory(self, codeact_state):
        """测试开始记录轨迹"""
        trajectory = _start_trajectory(codeact_state)
        
        assert trajectory is not None
        assert trajectory.task_id == codeact_state.task.task_id
        assert trajectory.execution_mode == codeact_state.execution_mode.value
        assert trajectory.status == TrajectoryStatus.PARTIAL
        assert trajectory.parameters == codeact_state.parameters
        assert trajectory.trajectory_id is not None and len(trajectory.trajectory_id) > 0
    
    def test_update_trajectory_code(self, codeact_state):
        """测试更新轨迹代码"""
        trajectory = _start_trajectory(codeact_state)
        
        test_code = "result = {'status': 'success', 'output': 'test'}"
        generation_time = 0.5
        
        _update_trajectory_code(trajectory, test_code, generation_time)
        
        assert trajectory.generated_code == test_code
        assert trajectory.code_length == len(test_code)
        assert trajectory.code_generation_time == generation_time
    
    def test_finalize_trajectory_success(self, codeact_state):
        """测试完成成功轨迹"""
        trajectory = _start_trajectory(codeact_state)
        _update_trajectory_code(trajectory, "result = {'status': 'success'}", 0.3)
        
        execution_result = {
            "status": "success",
            "output": "执行成功",
            "result": {"data": "test"}
        }
        execution_time = 1.2
        
        _finalize_trajectory(trajectory, execution_result, execution_time)
        
        assert trajectory.status == TrajectoryStatus.SUCCESS
        assert trajectory.execution_result == execution_result
        assert trajectory.execution_time == execution_time
        assert trajectory.error_type is None
        assert trajectory.error_message is None
    
    def test_finalize_trajectory_failure(self, codeact_state):
        """测试完成失败轨迹"""
        trajectory = _start_trajectory(codeact_state)
        _update_trajectory_code(trajectory, "result = {'status': 'failed'}", 0.3)
        
        execution_result = {
            "status": "failed",
            "error": "执行失败",
            "error_type": "RuntimeError",
            "error_traceback": "Traceback...",
            "error_category": "code_error"
        }
        execution_time = 0.8
        
        _finalize_trajectory(trajectory, execution_result, execution_time)
        
        assert trajectory.status == TrajectoryStatus.FAILED
        assert trajectory.error_type == "RuntimeError"
        assert trajectory.error_message == "执行失败"
        assert trajectory.error_category == "code_error"
    
    def test_save_trajectory_to_pool(self, codeact_state):
        """测试保存轨迹到池"""
        codeact_state.current_trajectory = _start_trajectory(codeact_state)
        _update_trajectory_code(codeact_state.current_trajectory, "print('test')", 0.2)
        
        execution_result = {"status": "success", "output": "test"}
        _finalize_trajectory(codeact_state.current_trajectory, execution_result, 0.5)
        
        _save_trajectory_to_pool(codeact_state, codeact_state.current_trajectory)
        
        assert len(codeact_state.trajectory_history) == 1
        assert codeact_state.trajectory_history[0].trajectory_id == codeact_state.current_trajectory.trajectory_id
        assert codeact_state.current_trajectory is None  # 保存后应该清空
    
    def test_full_trajectory_lifecycle(self, codeact_state):
        """测试完整的轨迹生命周期"""
        # 1. 开始轨迹
        codeact_state.current_trajectory = _start_trajectory(codeact_state)
        assert codeact_state.current_trajectory.status == TrajectoryStatus.PARTIAL
        
        # 2. 更新代码
        generated_code = "result = call_tool('test_tool', {'param': 'value'})"
        _update_trajectory_code(codeact_state.current_trajectory, generated_code, 0.5)
        assert codeact_state.current_trajectory.generated_code == generated_code
        
        # 3. 完成轨迹（成功）
        execution_result = {
            "status": "success",
            "output": "工具调用成功",
            "result": {"data": "result_data"}
        }
        _finalize_trajectory(codeact_state.current_trajectory, execution_result, 1.0)
        assert codeact_state.current_trajectory.status == TrajectoryStatus.SUCCESS
        
        # 4. 保存轨迹
        _save_trajectory_to_pool(codeact_state, codeact_state.current_trajectory)
        assert len(codeact_state.trajectory_history) == 1
        assert codeact_state.trajectory_history[0].status == TrajectoryStatus.SUCCESS
        
        # 验证轨迹内容
        saved_traj = codeact_state.trajectory_history[0]
        assert saved_traj.generated_code == generated_code
        assert saved_traj.execution_result == execution_result
        assert saved_traj.code_generation_time == 0.5
        assert saved_traj.execution_time == 1.0


class TestTrajectoryInCodeActFlow:
    """测试轨迹记录在 CodeAct 子图流程中的集成"""
    
    @pytest.fixture
    def codeact_subgraph(self):
        """创建 CodeAct 子图"""
        from nodes.subagents.code_act.graph import build_codeact_subgraph
        return build_codeact_subgraph()
    
    @pytest.fixture
    def sample_task(self):
        """创建示例任务"""
        return SubTask(
            task_id="test_trajectory_task",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="测试轨迹记录",
            result={
                "tools": [],
                "inputs": [],
                "outputs": []
            }
        )
    
    def test_trajectory_recorded_in_generate_node(self, codeact_subgraph, sample_task):
        """测试代码生成节点记录轨迹"""
        from nodes.subagents.code_act.graph import codeact_input_mapper
        
        # 创建输入状态
        input_state = codeact_input_mapper(
            executor_state=None,
            task=sample_task,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters={}
        )
        
        # 执行生成节点
        output_state = codeact_subgraph.get_node("generate_code").invoke(input_state)
        
        # 验证轨迹已创建
        assert output_state.current_trajectory is not None
        assert output_state.current_trajectory.task_id == sample_task.task_id
        assert output_state.current_trajectory.execution_mode == CodeActExecutionMode.CODEACT.value
        assert output_state.current_trajectory.status == TrajectoryStatus.PARTIAL
        
        # 验证代码已记录
        if output_state.generated_code:
            assert output_state.current_trajectory.generated_code == output_state.generated_code
            assert output_state.current_trajectory.code_length > 0
    
    def test_trajectory_recorded_in_execute_node(self, codeact_subgraph, sample_task):
        """测试代码执行节点记录轨迹"""
        from nodes.subagents.code_act.graph import codeact_input_mapper
        
        # 创建输入状态（带生成的代码）
        input_state = codeact_input_mapper(
            executor_state=None,
            task=sample_task,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters={}
        )
        
        # 先执行生成节点
        generate_output = codeact_subgraph.get_node("generate_code").invoke(input_state)
        
        # 确保有生成的代码
        if not generate_output.generated_code:
            generate_output.generated_code = "result = {'status': 'success', 'output': 'test'}"
        
        # 执行执行节点
        execute_output = codeact_subgraph.get_node("execute_code").invoke(generate_output)
        
        # 验证轨迹已完成并保存
        assert execute_output.current_trajectory is None  # 应该已清空
        assert len(execute_output.trajectory_history) == 1
        
        # 验证轨迹内容
        saved_traj = execute_output.trajectory_history[0]
        assert saved_traj.generated_code is not None
        assert saved_traj.execution_result is not None
        assert saved_traj.status in [TrajectoryStatus.SUCCESS, TrajectoryStatus.FAILED]
        assert saved_traj.execution_time >= 0
    
    def test_trajectory_for_failed_execution(self, codeact_subgraph, sample_task):
        """测试失败执行的轨迹记录"""
        from nodes.subagents.code_act.graph import codeact_input_mapper
        
        # 创建输入状态
        input_state = codeact_input_mapper(
            executor_state=None,
            task=sample_task,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters={}
        )
        
        # 生成代码
        generate_output = codeact_subgraph.get_node("generate_code").invoke(input_state)
        
        # 故意设置会导致错误的代码
        generate_output.generated_code = "undefined_variable = non_existent_function()"
        
        # 执行代码（应该失败）
        execute_output = codeact_subgraph.get_node("execute_code").invoke(generate_output)
        
        # 验证轨迹记录了失败信息
        assert len(execute_output.trajectory_history) == 1
        failed_traj = execute_output.trajectory_history[0]
        assert failed_traj.status == TrajectoryStatus.FAILED
        assert failed_traj.error_type is not None
        assert failed_traj.error_message is not None
        assert failed_traj.execution_result is not None
        assert failed_traj.execution_result.get("status") == "failed"

