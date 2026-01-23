"""
CodeAct 轨迹记录系统测试

测试轨迹记录、存储、查询和压缩功能。
"""

import pytest
from pathlib import Path
import json
import tempfile
import shutil
from datetime import datetime
import sys

# 添加agent目录到路径
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
            generated_code="print('hello')",
            status=TrajectoryStatus.SUCCESS,
            parameters={"param1": "value1"},
            execution_result={"status": "success", "output": "hello"}
        )
        
        assert trajectory.trajectory_id == "test_001"
        assert trajectory.task_id == "task_001"
        assert trajectory.execution_mode == "mcp_tool"
        assert trajectory.status == TrajectoryStatus.SUCCESS
        assert trajectory.generated_code == "print('hello')"
        assert trajectory.parameters == {"param1": "value1"}
    
    def test_trajectory_serialization(self):
        """测试轨迹序列化和反序列化"""
        trajectory = CodeTrajectory(
            trajectory_id="test_002",
            task_id="task_002",
            execution_mode="codeact",
            generated_code="result = 1 + 1",
            status=TrajectoryStatus.SUCCESS,
            parameters={},
            execution_result={"status": "success", "output": "2"}
        )
        
        # 序列化
        traj_dict = trajectory.to_dict()
        assert isinstance(traj_dict, dict)
        assert traj_dict["trajectory_id"] == "test_002"
        assert isinstance(traj_dict["timestamp"], str)  # datetime 转换为字符串
        
        # 反序列化
        restored = CodeTrajectory.from_dict(traj_dict)
        assert restored.trajectory_id == trajectory.trajectory_id
        assert restored.task_id == trajectory.task_id
        assert restored.generated_code == trajectory.generated_code
    
    def test_trajectory_hash(self):
        """测试轨迹哈希生成"""
        trajectory = CodeTrajectory(
            trajectory_id="test_003",
            task_id="task_003",
            execution_mode="mcp_tool",
            generated_code="print('test')",
            status=TrajectoryStatus.SUCCESS,
            parameters={"key": "value"}
        )
        
        hash1 = trajectory.get_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 哈希长度
        
        # 相同内容应该生成相同哈希
        trajectory2 = CodeTrajectory(
            trajectory_id="test_004",  # ID不同，但内容相同
            task_id="task_003",
            execution_mode="mcp_tool",
            generated_code="print('test')",
            status=TrajectoryStatus.SUCCESS,
            parameters={"key": "value"}
        )
        hash2 = trajectory2.get_hash()
        assert hash1 == hash2
    
    def test_trajectory_similarity(self):
        """测试轨迹相似度判断"""
        traj1 = CodeTrajectory(
            trajectory_id="test_005",
            task_id="task_005",
            execution_mode="mcp_tool",
            generated_code="print('hello world')",
            status=TrajectoryStatus.SUCCESS,
            parameters={"param": "value"}
        )
        
        traj2 = CodeTrajectory(
            trajectory_id="test_006",
            task_id="task_005",
            execution_mode="mcp_tool",
            generated_code="print('hello world')",  # 相同代码
            status=TrajectoryStatus.SUCCESS,
            parameters={"param": "value"}  # 相同参数
        )
        
        # 应该相似
        assert traj1.is_similar_to(traj2, threshold=0.8)
        
        # 不同代码应该不相似
        traj3 = CodeTrajectory(
            trajectory_id="test_007",
            task_id="task_007",
            execution_mode="mcp_tool",
            generated_code="print('different code')",  # 不同代码
            status=TrajectoryStatus.SUCCESS,
            parameters={"param": "value"}
        )
        
        assert not traj1.is_similar_to(traj3, threshold=0.8)


class TestTrajectoryPool:
    """测试 TrajectoryPool 管理器"""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """创建临时存储目录"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_create_pool(self, temp_storage_dir):
        """测试创建轨迹池"""
        pool = TrajectoryPool(pool_id="test_pool", storage_dir=temp_storage_dir)
        
        assert pool.pool_id == "test_pool"
        assert pool.storage_dir == temp_storage_dir
        assert len(pool.trajectories) == 0
    
    def test_add_trajectory(self, temp_storage_dir):
        """测试添加轨迹到池中"""
        pool = TrajectoryPool(pool_id="test_pool", storage_dir=temp_storage_dir)
        
        trajectory = CodeTrajectory(
            trajectory_id="",
            task_id="task_001",
            execution_mode="mcp_tool",
            generated_code="print('test')",
            status=TrajectoryStatus.SUCCESS,
            parameters={}
        )
        
        traj_id = pool.add_trajectory(trajectory)
        assert traj_id is not None
        assert len(pool.trajectories) == 1
        assert pool.trajectories[0].trajectory_id == traj_id
    
    def test_get_trajectories_by_task(self, temp_storage_dir):
        """测试根据任务ID获取轨迹"""
        pool = TrajectoryPool(pool_id="test_pool", storage_dir=temp_storage_dir)
        
        # 添加多个轨迹
        for i in range(3):
            traj = CodeTrajectory(
                trajectory_id=f"traj_{i}",
                task_id="task_001",
                execution_mode="mcp_tool",
                generated_code=f"code_{i}",
                status=TrajectoryStatus.SUCCESS,
                parameters={}
            )
            pool.add_trajectory(traj)
        
        # 添加不同任务的轨迹
        traj_other = CodeTrajectory(
            trajectory_id="traj_other",
            task_id="task_002",
            execution_mode="codeact",
            generated_code="other_code",
            status=TrajectoryStatus.SUCCESS,
            parameters={}
        )
        pool.add_trajectory(traj_other)
        
        # 获取 task_001 的轨迹
        task_trajectories = pool.get_trajectories_by_task("task_001")
        assert len(task_trajectories) == 3
        assert all(t.task_id == "task_001" for t in task_trajectories)
    
    def test_get_successful_trajectories(self, temp_storage_dir):
        """测试获取成功的轨迹"""
        pool = TrajectoryPool(pool_id="test_pool", storage_dir=temp_storage_dir)
        
        # 添加成功和失败的轨迹
        success_traj = CodeTrajectory(
            trajectory_id="success_1",
            task_id="task_001",
            execution_mode="mcp_tool",
            generated_code="success_code",
            status=TrajectoryStatus.SUCCESS,
            parameters={}
        )
        pool.add_trajectory(success_traj)
        
        failed_traj = CodeTrajectory(
            trajectory_id="failed_1",
            task_id="task_002",
            execution_mode="mcp_tool",
            generated_code="failed_code",
            status=TrajectoryStatus.FAILED,
            parameters={},
            error_message="Test error"
        )
        pool.add_trajectory(failed_traj)
        
        successful = pool.get_successful_trajectories()
        assert len(successful) == 1
        assert successful[0].status == TrajectoryStatus.SUCCESS
    
    def test_find_similar_trajectories(self, temp_storage_dir):
        """测试查找相似轨迹"""
        pool = TrajectoryPool(pool_id="test_pool", storage_dir=temp_storage_dir)
        
        # 添加相似轨迹
        traj1 = CodeTrajectory(
            trajectory_id="traj_1",
            task_id="task_001",
            execution_mode="mcp_tool",
            generated_code="print('hello')",
            status=TrajectoryStatus.SUCCESS,
            parameters={"param": "value"}
        )
        pool.add_trajectory(traj1)
        
        traj2 = CodeTrajectory(
            trajectory_id="traj_2",
            task_id="task_002",
            execution_mode="mcp_tool",
            generated_code="print('hello')",  # 相同代码
            status=TrajectoryStatus.SUCCESS,
            parameters={"param": "value"}  # 相同参数
        )
        pool.add_trajectory(traj2)
        
        # 查找与 traj1 相似的轨迹
        similar = pool.find_similar_trajectories(traj1, threshold=0.8)
        assert len(similar) >= 1  # 至少应该找到 traj2
    
    def test_save_and_load_pool(self, temp_storage_dir):
        """测试保存和加载轨迹池"""
        pool = TrajectoryPool(pool_id="test_pool", storage_dir=temp_storage_dir)
        
        # 添加一些轨迹
        for i in range(3):
            traj = CodeTrajectory(
                trajectory_id=f"traj_{i}",
                task_id=f"task_{i}",
                execution_mode="mcp_tool",
                generated_code=f"code_{i}",
                status=TrajectoryStatus.SUCCESS,
                parameters={}
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
        
        # 保存（未压缩）
        saved_path_json = pool.save(compressed=False)
        assert saved_path_json.exists()
        assert saved_path_json.suffix == ".json"
    
    def test_pool_statistics(self, temp_storage_dir):
        """测试轨迹池统计信息"""
        pool = TrajectoryPool(pool_id="test_pool", storage_dir=temp_storage_dir)
        
        # 添加成功和失败的轨迹
        for i in range(5):
            status = TrajectoryStatus.SUCCESS if i < 3 else TrajectoryStatus.FAILED
            traj = CodeTrajectory(
                trajectory_id=f"traj_{i}",
                task_id=f"task_{i}",
                execution_mode="mcp_tool",
                generated_code=f"code_{i}",
                status=status,
                parameters={}
            )
            pool.add_trajectory(traj)
        
        stats = pool.get_statistics()
        assert stats["total_trajectories"] == 5
        assert stats["successful"] == 3
        assert stats["failed"] == 2
        assert stats["success_rate"] == 0.6


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
                "tools": [{
                    "name": "test_tool",
                    "tool_name": "test_tool",
                    "description": "测试工具"
                }],
                "inputs": [],
                "outputs": []
            }
        )
    
    def test_start_trajectory(self, sample_task):
        """测试开始记录轨迹"""
        state = CodeActState(
            task=sample_task,
            task_description=sample_task.content,
            tools=[],
            inputs=[],
            parameters={},
            execution_mode=CodeActExecutionMode.MCP_TOOL
        )
        
        trajectory = _start_trajectory(state)
        
        assert trajectory is not None
        assert trajectory.task_id == sample_task.task_id
        assert trajectory.execution_mode == CodeActExecutionMode.MCP_TOOL.value
        assert trajectory.status == TrajectoryStatus.PARTIAL
        assert trajectory.trajectory_id is not None
    
    def test_update_trajectory_code(self, sample_task):
        """测试更新轨迹代码"""
        state = CodeActState(
            task=sample_task,
            task_description=sample_task.content,
            tools=[],
            inputs=[],
            parameters={},
            execution_mode=CodeActExecutionMode.MCP_TOOL
        )
        
        trajectory = _start_trajectory(state)
        code = "print('test code')"
        generation_time = 0.5
        
        _update_trajectory_code(trajectory, code, generation_time)
        
        assert trajectory.generated_code == code
        assert trajectory.code_length == len(code)
        assert trajectory.code_generation_time == generation_time
    
    def test_finalize_trajectory_success(self, sample_task):
        """测试完成成功轨迹"""
        state = CodeActState(
            task=sample_task,
            task_description=sample_task.content,
            tools=[],
            inputs=[],
            parameters={},
            execution_mode=CodeActExecutionMode.MCP_TOOL
        )
        
        trajectory = _start_trajectory(state)
        _update_trajectory_code(trajectory, "print('success')", 0.3)
        
        execution_result = {
            "status": "success",
            "output": "执行成功"
        }
        execution_time = 0.2
        
        _finalize_trajectory(trajectory, execution_result, execution_time)
        
        assert trajectory.status == TrajectoryStatus.SUCCESS
        assert trajectory.execution_result == execution_result
        assert trajectory.execution_time == execution_time
        assert trajectory.error_type is None
    
    def test_finalize_trajectory_failed(self, sample_task):
        """测试完成失败轨迹"""
        state = CodeActState(
            task=sample_task,
            task_description=sample_task.content,
            tools=[],
            inputs=[],
            parameters={},
            execution_mode=CodeActExecutionMode.MCP_TOOL
        )
        
        trajectory = _start_trajectory(state)
        _update_trajectory_code(trajectory, "print('fail')", 0.3)
        
        execution_result = {
            "status": "failed",
            "error": "执行失败",
            "error_type": "RuntimeError",
            "error_traceback": "Traceback..."
        }
        execution_time = 0.1
        
        _finalize_trajectory(trajectory, execution_result, execution_time)
        
        assert trajectory.status == TrajectoryStatus.FAILED
        assert trajectory.error_type == "RuntimeError"
        assert trajectory.error_message == "执行失败"
        assert trajectory.error_traceback == "Traceback..."
    
    def test_save_trajectory_to_pool(self, sample_task):
        """测试保存轨迹到池中"""
        state = CodeActState(
            task=sample_task,
            task_description=sample_task.content,
            tools=[],
            inputs=[],
            parameters={},
            execution_mode=CodeActExecutionMode.MCP_TOOL
        )
        
        trajectory = _start_trajectory(state)
        _update_trajectory_code(trajectory, "print('test')", 0.2)
        
        execution_result = {"status": "success", "output": "ok"}
        _finalize_trajectory(trajectory, execution_result, 0.1)
        
        _save_trajectory_to_pool(state, trajectory)
        
        assert len(state.trajectory_history) == 1
        assert state.trajectory_history[0].trajectory_id == trajectory.trajectory_id


class TestTrajectoryInCodeActFlow:
    """测试轨迹记录在 CodeAct 流程中的完整集成"""
    
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
    
    def test_trajectory_recorded_in_codeact_flow(self, codeact_subgraph, sample_task):
        """测试在 CodeAct 流程中轨迹被正确记录"""
        from nodes.subagents.code_act.graph import codeact_input_mapper
        
        # 创建输入状态
        input_state = codeact_input_mapper(
            executor_state=None,
            task=sample_task,
            execution_mode=CodeActExecutionMode.CODEACT,
            parameters={}
        )
        
        # 执行子图
        output_state = codeact_subgraph.invoke(input_state)
        
        # 验证轨迹被记录
        assert output_state.trajectory_history is not None
        assert len(output_state.trajectory_history) > 0
        
        # 检查轨迹内容
        trajectory = output_state.trajectory_history[0]
        assert trajectory.task_id == sample_task.task_id
        assert trajectory.execution_mode == CodeActExecutionMode.CODEACT.value
        assert trajectory.generated_code is not None
        assert trajectory.execution_result is not None
        assert trajectory.status in [TrajectoryStatus.SUCCESS, TrajectoryStatus.FAILED]
        
        print(f"\n✓ 轨迹记录成功")
        print(f"  轨迹ID: {trajectory.trajectory_id}")
        print(f"  状态: {trajectory.status}")
        print(f"  代码长度: {trajectory.code_length}")
        print(f"  生成耗时: {trajectory.code_generation_time:.3f}s")
        print(f"  执行耗时: {trajectory.execution_time:.3f}s")
    
    def test_multiple_trajectories_for_same_task(self, codeact_subgraph, sample_task):
        """测试同一任务的多次执行产生多个轨迹"""
        from nodes.subagents.code_act.graph import codeact_input_mapper
        
        trajectories = []
        
        # 执行多次
        for i in range(3):
            input_state = codeact_input_mapper(
                executor_state=None,
                task=sample_task,
                execution_mode=CodeActExecutionMode.CODEACT,
                parameters={}
            )
            
            output_state = codeact_subgraph.invoke(input_state)
            
            if output_state.trajectory_history:
                trajectories.extend(output_state.trajectory_history)
        
        # 验证有多个轨迹
        assert len(trajectories) >= 3
        
        # 验证所有轨迹都属于同一任务
        assert all(t.task_id == sample_task.task_id for t in trajectories)
        
        print(f"\n✓ 多次执行产生 {len(trajectories)} 个轨迹")

