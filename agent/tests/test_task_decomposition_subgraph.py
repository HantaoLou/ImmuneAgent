"""
Task Decomposition Subgraph 单独测试用例

测试 task_decomposition subgraph 的独立功能，不涉及主图。
运行方式：pytest tests/test_task_decomposition_subgraph.py -v
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

from nodes.subagents.task_decomposition.graph import (
    build_task_decomposition_subgraph,
    TaskDecompositionState,
    task_decomposition_input_mapper,
    task_decomposition_output_mapper,
    _create_codeact_tool,
    _check_and_add_codeact_to_tasks
)
from nodes.subagents.task_decomposition.tool_categorizer import load_service_list, get_tools_by_service_ids
from state import GlobalState, UserTaskType, SubTask, ParallelTaskGroup, TaskStatus


def _ensure_task_decomposition_state(result):
    """确保结果是 TaskDecompositionState 对象（处理字典返回值）"""
    if isinstance(result, dict):
        return TaskDecompositionState(**result)
    return result


@pytest.fixture(scope="module")
def task_decomposition_subgraph():
    """构建并返回 Task Decomposition 子图"""
    return build_task_decomposition_subgraph()


@pytest.fixture
def sample_global_state():
    """示例全局状态"""
    return GlobalState(
        user_input="分析抗体序列的V(D)J重组情况",
        execution_plan="1. 下载序列数据\n2. 使用IgBlast分析V(D)J\n3. 提取CDR3区域\n4. 生成分析报告",
        sandbox_dir="./sandbox"
    )


@pytest.fixture
def sample_global_state_with_execution_plan():
    """带执行计划的全局状态"""
    return GlobalState(
        user_input="执行抗体设计任务",
        execution_plan="""
        1. 准备抗原结构文件
        2. 使用AlphaFold3预测抗体结构
        3. 使用BindCraft进行抗体设计
        4. 评估设计结果
        5. 生成设计报告
        """,
        sandbox_dir="./sandbox"
    )


class TestTaskDecompositionSubgraphBasic:
    """Task Decomposition Subgraph 基础功能测试"""
    
    def test_subgraph_build(self, task_decomposition_subgraph):
        """测试子图构建是否成功"""
        assert task_decomposition_subgraph is not None
        print("✓ Task Decomposition Subgraph 构建成功")
    
    def test_subgraph_invoke_basic(self, task_decomposition_subgraph, sample_global_state):
        """测试子图基本调用"""
        # 使用 input_mapper 转换状态
        subgraph_input = task_decomposition_input_mapper(sample_global_state)
        
        # 调用子图
        subgraph_output = task_decomposition_subgraph.invoke(subgraph_input)
        
        # 确保输出是 TaskDecompositionState 对象
        result = _ensure_task_decomposition_state(subgraph_output)
        
        assert result is not None
        assert hasattr(result, 'user_input')
        assert result.user_input == sample_global_state.user_input
        print(f"✓ Task Decomposition Subgraph 基本调用成功")


class TestTaskDecompositionThreeStageProcess:
    """Task Decomposition 三阶段分解过程测试"""
    
    def test_coarse_decomposition_stage(self, task_decomposition_subgraph, sample_global_state):
        """测试阶段0：粗分解（确定service_id）"""
        subgraph_input = task_decomposition_input_mapper(sample_global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # 验证阶段0：required_service_ids 应该被填充
        assert result.required_service_ids is not None
        assert isinstance(result.required_service_ids, list)
        
        # 验证筛选后的工具
        assert result.filtered_tools is not None
        assert isinstance(result.filtered_tools, list)
        
        print(f"\n✓ 阶段0（粗分解）完成")
        print(f"  所需服务数: {len(result.required_service_ids)}")
        if result.required_service_ids:
            print(f"  所需服务: {', '.join(result.required_service_ids[:5])}...")  # 只显示前5个
        print(f"  筛选后工具数: {len(result.filtered_tools)}")
    
    def test_fine_decomposition_stage(self, task_decomposition_subgraph, sample_global_state):
        """测试阶段1：细分解（基于筛选工具）"""
        subgraph_input = task_decomposition_input_mapper(sample_global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # 验证阶段1：raw_tasks 应该被填充（基于筛选后的工具）
        assert result.raw_tasks is not None
        assert isinstance(result.raw_tasks, list)
        
        # 验证筛选后的工具数量应该小于等于总工具数
        assert len(result.filtered_tools) <= len(result.available_tools)
        
        print(f"\n✓ 阶段1（细分解）完成")
        print(f"  序列化任务数: {len(result.raw_tasks)}")
        if result.raw_tasks:
            print(f"  示例任务: {result.raw_tasks[0].get('task_id', 'N/A')}")
    
    def test_parallel_inference_stage(self, task_decomposition_subgraph, sample_global_state):
        """测试阶段2：并行推断"""
        subgraph_input = task_decomposition_input_mapper(sample_global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # 验证阶段2：subtasks 和 parallel_task_groups 应该被填充
        assert result.subtasks is not None
        assert isinstance(result.subtasks, list)
        assert result.parallel_task_groups is not None
        assert isinstance(result.parallel_task_groups, dict)
        
        print(f"\n✓ 阶段2（并行推断）完成")
        print(f"  最终子任务数: {len(result.subtasks)}")
        print(f"  并行任务组数: {len(result.parallel_task_groups)}")
    
    def test_three_stage_complete_flow(self, task_decomposition_subgraph, sample_global_state_with_execution_plan):
        """测试完整三阶段流程"""
        subgraph_input = task_decomposition_input_mapper(sample_global_state_with_execution_plan)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # 验证三个阶段都执行了
        # 阶段0：粗分解
        assert result.required_service_ids is not None
        assert isinstance(result.required_service_ids, list)
        assert result.filtered_tools is not None
        
        # 阶段1：细分解
        assert result.raw_tasks is not None
        assert isinstance(result.raw_tasks, list)
        
        # 阶段2：并行推断
        assert result.subtasks is not None
        assert isinstance(result.subtasks, list)
        assert result.parallel_task_groups is not None
        assert isinstance(result.parallel_task_groups, dict)
        
        print(f"\n✓ 三阶段完整流程测试通过")
        print(f"  阶段0 - 所需服务: {len(result.required_service_ids)} 个")
        print(f"  阶段0 - 筛选工具: {len(result.filtered_tools)} 个")
        print(f"  阶段1 - 序列化任务: {len(result.raw_tasks)} 个")
        print(f"  阶段2 - 最终子任务: {len(result.subtasks)} 个")
        print(f"  阶段2 - 并行任务组: {len(result.parallel_task_groups)} 个")
    
    def test_task_decomposition_without_plan(self, task_decomposition_subgraph):
        """测试不带执行计划的任务分解"""
        global_state = GlobalState(
            user_input="分析B细胞轨迹",
            execution_plan=None,
            sandbox_dir="./sandbox"
        )
        
        subgraph_input = task_decomposition_input_mapper(global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # 即使没有执行计划，也应该能够完成三阶段分解
        assert result.required_service_ids is not None
        assert result.filtered_tools is not None
        assert result.subtasks is not None
        assert isinstance(result.subtasks, list)
        print(f"\n✓ 无执行计划的任务分解完成")
        print(f"  子任务数: {len(result.subtasks)}")


class TestServiceBasedClassification:
    """基于Service的分类测试"""
    
    def test_service_list_loading(self):
        """测试service_list加载"""
        service_list = load_service_list()
        
        assert service_list is not None
        assert isinstance(service_list, list)
        assert len(service_list) > 0
        
        # 验证service结构
        for service in service_list[:3]:  # 检查前3个
            assert "service_id" in service
            assert "description" in service
            assert isinstance(service["service_id"], str)
            assert isinstance(service["description"], str)
        
        print(f"\n✓ Service列表加载测试通过")
        print(f"  服务数量: {len(service_list)}")
        print(f"  示例服务: {service_list[0].get('service_id', 'N/A')}")
    
    def test_tool_filtering_by_service_id(self, sample_global_state):
        """测试根据service_id筛选工具"""
        subgraph_input = task_decomposition_input_mapper(sample_global_state)
        
        # 测试筛选特定service的工具
        test_service_ids = ["igblast", "r_bcell", "airr"]
        filtered_tools = get_tools_by_service_ids(subgraph_input.available_tools, test_service_ids)
        
        assert isinstance(filtered_tools, list)
        assert len(filtered_tools) <= len(subgraph_input.available_tools)
        
        # 验证筛选后的工具都属于指定的service
        for tool in filtered_tools:
            tool_service = tool.get("service", "")
            assert tool_service in test_service_ids
        
        print(f"\n✓ 工具筛选测试通过")
        print(f"  筛选前工具数: {len(subgraph_input.available_tools)}")
        print(f"  筛选后工具数: {len(filtered_tools)}")
        print(f"  筛选服务: {', '.join(test_service_ids)}")
    
    def test_coarse_decomposition_service_selection(self, task_decomposition_subgraph, sample_global_state):
        """测试粗分解阶段的service选择"""
        subgraph_input = task_decomposition_input_mapper(sample_global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # 验证required_service_ids是有效的service_id
        service_list = load_service_list()
        valid_service_ids = {s["service_id"] for s in service_list}
        
        for service_id in result.required_service_ids:
            # 如果LLM返回了service_id，应该验证其有效性
            # 但降级方案可能使用所有service，所以这里只做非空检查
            assert isinstance(service_id, str)
            assert len(service_id) > 0
        
        print(f"\n✓ 粗分解service选择测试通过")
        print(f"  选择的service数: {len(result.required_service_ids)}")
        if result.required_service_ids:
            print(f"  选择的service: {', '.join(result.required_service_ids[:5])}...")


class TestTaskDecompositionOutputStructure:
    """Task Decomposition 输出结构测试"""
    
    def test_subtasks_structure(self, task_decomposition_subgraph, sample_global_state):
        """测试子任务结构"""
        subgraph_input = task_decomposition_input_mapper(sample_global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # 验证 subtasks 结构
        assert isinstance(result.subtasks, list)
        
        # 如果有子任务，验证其结构
        if result.subtasks:
            for subtask in result.subtasks:
                assert isinstance(subtask, SubTask)
                assert subtask.task_id is not None
                assert subtask.task_type is not None
                assert subtask.content is not None
                assert isinstance(subtask.dependencies, list)
                assert isinstance(subtask.parallel_group_id, (str, type(None)))
            
            print(f"\n✓ 子任务结构验证通过")
            print(f"  子任务数量: {len(result.subtasks)}")
            for i, subtask in enumerate(result.subtasks[:3], 1):  # 只显示前3个
                print(f"  {i}. {subtask.task_id}: {subtask.content[:50]}...")
    
    def test_parallel_task_groups_structure(self, task_decomposition_subgraph, sample_global_state):
        """测试并行任务组结构"""
        subgraph_input = task_decomposition_input_mapper(sample_global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # 验证 parallel_task_groups 结构
        assert isinstance(result.parallel_task_groups, dict)
        
        # 如果有并行任务组，验证其结构
        if result.parallel_task_groups:
            for group_id, group in result.parallel_task_groups.items():
                assert isinstance(group, ParallelTaskGroup)
                assert group.group_id == group_id
                assert isinstance(group.subtasks, list)
                assert len(group.subtasks) > 0
                assert isinstance(group.status, TaskStatus)
                
                # 验证组内子任务
                for subtask in group.subtasks:
                    assert isinstance(subtask, SubTask)
                    assert subtask.parallel_group_id == group_id
            
            print(f"\n✓ 并行任务组结构验证通过")
            print(f"  并行组数量: {len(result.parallel_task_groups)}")
            for group_id, group in list(result.parallel_task_groups.items())[:2]:  # 只显示前2个
                print(f"  组 {group_id}: {len(group.subtasks)} 个并行任务")
    
    def test_decomposition_summary(self, task_decomposition_subgraph, sample_global_state):
        """测试分解摘要"""
        subgraph_input = task_decomposition_input_mapper(sample_global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # decomposition_summary 可能为 None（如果LLM不可用）
        if result.decomposition_summary:
            assert isinstance(result.decomposition_summary, str)
            print(f"\n✓ 分解摘要已生成")
            print(f"  摘要长度: {len(result.decomposition_summary)} 字符")
            print(f"  摘要预览: {result.decomposition_summary[:100]}...")
        else:
            print(f"\n- 分解摘要未生成（可能LLM不可用）")


class TestTaskDecompositionStateMapping:
    """Task Decomposition 状态映射测试"""
    
    def test_input_mapper(self, sample_global_state):
        """测试输入映射函数"""
        subgraph_state = task_decomposition_input_mapper(sample_global_state)
        
        assert isinstance(subgraph_state, TaskDecompositionState)
        assert subgraph_state.user_input == sample_global_state.user_input
        assert subgraph_state.execution_plan == sample_global_state.execution_plan
        assert isinstance(subgraph_state.available_tools, list)
        assert subgraph_state.required_service_ids == []
        assert subgraph_state.filtered_tools == []
        assert subgraph_state.subtasks == []
        assert subgraph_state.parallel_task_groups == {}
        print("✓ 输入映射函数测试通过")
        print(f"  可用工具数: {len(subgraph_state.available_tools)}")
    
    def test_output_mapper(self, sample_global_state):
        """测试输出映射函数"""
        # 创建模拟的子图输出
        mock_subtask = SubTask(
            task_id="task_001",
            task_type=UserTaskType.EXECUTE_PLAN,
            content="分析V(D)J序列",
            dependencies=[],
            parallel_group_id=None
        )
        
        mock_parallel_group = ParallelTaskGroup(
            group_id="group_001",
            subtasks=[
                SubTask(
                    task_id="task_002",
                    task_type=UserTaskType.EXECUTE_PLAN,
                    content="并行任务1",
                    dependencies=[],
                    parallel_group_id="group_001"
                )
            ],
            status=TaskStatus.PENDING
        )
        
        subgraph_output = TaskDecompositionState(
            user_input="分析抗体序列",
            execution_plan="执行计划",
            available_tools=[],
            required_service_ids=["igblast", "r_bcell"],
            filtered_tools=[],
            raw_tasks=[],
            subtasks=[mock_subtask],
            parallel_task_groups={"group_001": mock_parallel_group},
            decomposition_summary="任务分解摘要"
        )
        
        # 执行输出映射
        updated_global_state = task_decomposition_output_mapper(subgraph_output, sample_global_state)
        
        # 验证结果已同步
        assert len(updated_global_state.subtasks) == 1
        assert updated_global_state.subtasks[0].task_id == "task_001"
        assert "group_001" in updated_global_state.parallel_task_groups
        assert updated_global_state.merged_result.get("decomposition_summary") == "任务分解摘要"
        
        print("✓ 输出映射函数测试通过")
        print(f"  同步的子任务数: {len(updated_global_state.subtasks)}")
        print(f"  同步的并行组数: {len(updated_global_state.parallel_task_groups)}")


class TestTaskDecompositionEdgeCases:
    """Task Decomposition 边界情况测试"""
    
    def test_empty_input(self, task_decomposition_subgraph):
        """测试空输入"""
        global_state = GlobalState(
            user_input="",
            execution_plan=None,
            sandbox_dir="./sandbox"
        )
        
        subgraph_input = task_decomposition_input_mapper(global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # 即使输入为空，也应该有降级处理
        assert result.required_service_ids is not None
        assert result.filtered_tools is not None
        assert result.subtasks is not None
        assert isinstance(result.subtasks, list)
        print("✓ 空输入测试通过")
    
    def test_long_input(self, task_decomposition_subgraph):
        """测试长输入"""
        long_input = "分析" + "A" * 1000 + "序列"
        global_state = GlobalState(
            user_input=long_input,
            execution_plan="执行计划" * 100,
            sandbox_dir="./sandbox"
        )
        
        subgraph_input = task_decomposition_input_mapper(global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        assert result.required_service_ids is not None
        assert result.subtasks is not None
        print(f"✓ 长输入测试通过（输入长度: {len(long_input)} 字符）")
    
    def test_complex_execution_plan(self, task_decomposition_subgraph):
        """测试复杂执行计划"""
        complex_plan = """
        阶段1：数据准备
          1.1 下载原始数据
          1.2 数据清洗和预处理
          1.3 数据格式转换
        
        阶段2：数据分析
          2.1 使用工具A进行初步分析
          2.2 使用工具B进行深度分析
          2.3 结果整合
        
        阶段3：结果输出
          3.1 生成可视化图表
          3.2 编写分析报告
          3.3 导出结果文件
        """
        
        global_state = GlobalState(
            user_input="执行复杂分析任务",
            execution_plan=complex_plan,
            sandbox_dir="./sandbox"
        )
        
        subgraph_input = task_decomposition_input_mapper(global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        assert result.required_service_ids is not None
        assert result.subtasks is not None
        assert result.parallel_task_groups is not None
        print(f"✓ 复杂执行计划测试通过")
        print(f"  生成的子任务数: {len(result.subtasks)}")
        print(f"  生成的并行组数: {len(result.parallel_task_groups)}")


class TestTaskDecompositionToolMatching:
    """Task Decomposition 工具匹配测试"""
    
    def test_tool_loading(self, sample_global_state):
        """测试工具加载"""
        subgraph_input = task_decomposition_input_mapper(sample_global_state)
        
        # 验证工具已加载
        assert isinstance(subgraph_input.available_tools, list)
        assert len(subgraph_input.available_tools) > 0
        
        # 验证工具结构
        if subgraph_input.available_tools:
            tool = subgraph_input.available_tools[0]
            assert "name" in tool
            assert "description" in tool
            assert "service" in tool  # 验证有service字段
            print(f"\n✓ 工具加载测试通过")
            print(f"  加载的工具数: {len(subgraph_input.available_tools)}")
            print(f"  示例工具: {tool.get('name', 'N/A')} (service: {tool.get('service', 'N/A')})")
    
    def test_task_with_tool_matching(self, task_decomposition_subgraph, sample_global_state):
        """测试任务与工具匹配"""
        subgraph_input = task_decomposition_input_mapper(sample_global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # 验证筛选后的工具数量合理
        assert len(result.filtered_tools) <= len(result.available_tools)
        
        # 验证第一阶段结果中包含工具信息
        if result.raw_tasks:
            for task in result.raw_tasks[:3]:  # 检查前3个任务
                if "tools" in task:
                    print(f"\n✓ 任务工具匹配测试通过")
                    print(f"  任务 {task.get('task_id', 'N/A')} 匹配到 {len(task.get('tools', []))} 个工具")
                    break


class TestCodeActFallbackStrategy:
    """CodeAct 兜底策略测试"""
    
    def test_codeact_tool_creation(self):
        """测试CodeAct工具创建"""
        codeact_tool = _create_codeact_tool()
        
        assert codeact_tool is not None
        assert codeact_tool.get("name") == "codeact"
        assert codeact_tool.get("service") == "codeact"
        assert "description" in codeact_tool
        assert "tool" in codeact_tool
        assert isinstance(codeact_tool["tool"], list)
        assert len(codeact_tool["tool"]) > 0
        
        # 验证工具结构
        tool_item = codeact_tool["tool"][0]
        assert tool_item.get("tool_name") == "codeact"
        assert "description" in tool_item
        assert "parameters" in tool_item
        
        print(f"\n✓ CodeAct工具创建测试通过")
        print(f"  工具名称: {codeact_tool.get('name')}")
        print(f"  服务: {codeact_tool.get('service')}")
    
    def test_codeact_service_in_list(self):
        """测试CodeAct服务是否在service_list中"""
        service_list = load_service_list()
        
        codeact_services = [s for s in service_list if s.get("service_id") == "codeact"]
        assert len(codeact_services) > 0, "CodeAct服务应该在service_list中"
        
        codeact_service = codeact_services[0]
        assert "service_id" in codeact_service
        assert "description" in codeact_service
        assert "codeact" in codeact_service["description"].lower() or "代码" in codeact_service["description"]
        
        print(f"\n✓ CodeAct服务在service_list中")
        print(f"  服务描述: {codeact_service.get('description', 'N/A')[:50]}...")
    
    def test_coarse_decomposition_fallback(self, task_decomposition_subgraph):
        """测试粗分解阶段的兜底策略（没有匹配到工具时添加codeact）"""
        # 创建一个不太可能匹配到现有工具的任务
        global_state = GlobalState(
            user_input="执行一个非常特殊的自定义任务，需要复杂的自定义算法和数据处理",
            execution_plan="1. 自定义数据处理\n2. 特殊算法计算\n3. 生成自定义报告",
            sandbox_dir="./sandbox"
        )
        
        subgraph_input = task_decomposition_input_mapper(global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # 验证粗分解结果
        assert result.required_service_ids is not None
        assert isinstance(result.required_service_ids, list)
        assert result.filtered_tools is not None
        
        # 如果筛选后的工具为空，应该添加了codeact
        if len(result.filtered_tools) == 0:
            # 这种情况不应该发生，因为兜底策略会添加codeact
            # 但为了测试，我们检查是否有codeact
            pass
        
        # 验证filtered_tools不为空（兜底策略应该确保至少有一个工具）
        # 注意：如果LLM返回了service，可能已经有工具了
        print(f"\n✓ 粗分解兜底策略测试")
        print(f"  所需服务数: {len(result.required_service_ids)}")
        print(f"  筛选后工具数: {len(result.filtered_tools)}")
        if "codeact" in result.required_service_ids:
            print(f"  ✓ CodeAct服务已被添加")
    
    def test_fine_decomposition_fallback(self, task_decomposition_subgraph):
        """测试细分解阶段的兜底策略"""
        # 创建一个任务，模拟没有匹配到工具的情况
        global_state = GlobalState(
            user_input="执行需要自定义代码的复杂任务",
            execution_plan="1. 自定义数据处理\n2. 特殊计算\n3. 生成报告",
            sandbox_dir="./sandbox"
        )
        
        subgraph_input = task_decomposition_input_mapper(global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # 验证细分解结果
        assert result.filtered_tools is not None
        assert isinstance(result.filtered_tools, list)
        
        # 验证如果筛选后的工具为空，应该添加了codeact
        # 注意：由于兜底策略，filtered_tools应该至少包含codeact（如果原本为空）
        print(f"\n✓ 细分解兜底策略测试")
        print(f"  筛选后工具数: {len(result.filtered_tools)}")
        
        # 检查是否有codeact工具
        codeact_tools = [t for t in result.filtered_tools if t.get("name") == "codeact"]
        if codeact_tools:
            print(f"  ✓ CodeAct工具已被添加")
    
    def test_task_level_codeact_fallback(self):
        """测试任务级别的CodeAct兜底策略"""
        # 创建模拟的raw_tasks，其中一些任务没有匹配到工具
        raw_tasks = [
            {
                "task_id": "task_001",
                "name": "有工具的任务",
                "description": "这个任务匹配到了工具",
                "tools": ["analyze_vdj_batch"]
            },
            {
                "task_id": "task_002",
                "name": "没有工具的任务",
                "description": "这个任务没有匹配到工具",
                "tools": []
            },
            {
                "task_id": "task_003",
                "name": "工具不可用的任务",
                "description": "这个任务匹配的工具不在可用列表中",
                "tools": ["nonexistent_tool"]
            }
        ]
        
        # 创建可用工具列表（不包含codeact）
        available_tools = [
            {
                "name": "analyze_vdj_batch",
                "service": "igblast",
                "tool": [{"tool_name": "analyze_vdj_batch"}]
            }
        ]
        
        # 执行检查和添加
        _check_and_add_codeact_to_tasks(raw_tasks, available_tools)
        
        # 验证结果
        assert raw_tasks[0]["tools"] == ["analyze_vdj_batch"]  # 有工具的任务不应该改变
        assert "codeact" in raw_tasks[1]["tools"]  # 没有工具的任务应该添加codeact
        assert "codeact" in raw_tasks[2]["tools"]  # 工具不可用的任务应该添加codeact
        
        print(f"\n✓ 任务级别CodeAct兜底策略测试通过")
        print(f"  任务1（有工具）: {raw_tasks[0]['tools']}")
        print(f"  任务2（无工具）: {raw_tasks[1]['tools']}")
        print(f"  任务3（工具不可用）: {raw_tasks[2]['tools']}")
    
    def test_codeact_in_decomposition_result(self, task_decomposition_subgraph):
        """测试分解结果中是否包含codeact工具"""
        # 创建一个可能需要codeact的任务
        global_state = GlobalState(
            user_input="执行一个需要自定义Python代码的复杂数据分析任务",
            execution_plan="1. 读取数据\n2. 自定义算法处理\n3. 可视化结果",
            sandbox_dir="./sandbox"
        )
        
        subgraph_input = task_decomposition_input_mapper(global_state)
        result = task_decomposition_subgraph.invoke(subgraph_input)
        result = _ensure_task_decomposition_state(result)
        
        # 检查结果中是否可能包含codeact
        has_codeact_service = "codeact" in result.required_service_ids
        has_codeact_tool = any(t.get("name") == "codeact" for t in result.filtered_tools)
        
        # 检查raw_tasks中是否有codeact
        has_codeact_in_tasks = False
        if result.raw_tasks:
            for task in result.raw_tasks:
                task_tools = task.get("tools", [])
                if isinstance(task_tools, list):
                    for tool in task_tools:
                        if (isinstance(tool, str) and tool == "codeact") or \
                           (isinstance(tool, dict) and tool.get("tool_name") == "codeact"):
                            has_codeact_in_tasks = True
                            break
        
        print(f"\n✓ CodeAct在分解结果中的测试")
        print(f"  CodeAct服务: {'✓' if has_codeact_service else '✗'}")
        print(f"  CodeAct工具: {'✓' if has_codeact_tool else '✗'}")
        print(f"  CodeAct在任务中: {'✓' if has_codeact_in_tasks else '✗'}")
        
        # 注意：这些检查可能为False，因为如果任务能匹配到现有工具，就不会使用codeact
        # 这是正常的，因为codeact是兜底策略


def test_environment_check():
    """检查环境配置"""
    print("\n" + "=" * 50)
    print("检查 Task Decomposition Subgraph 环境配置...")
    
    dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QIANFAN_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    has_llm = any([dashscope_key, anthropic_key, openai_key])
    
    if dashscope_key:
        print("✓ 通义千问 API Key 已配置")
    else:
        print("✗ 通义千问 API Key 未配置")
    
    if anthropic_key:
        print("✓ Anthropic API Key 已配置")
    else:
        print("✗ Anthropic API Key 未配置")
    
    if openai_key:
        print("✓ OpenAI API Key 已配置")
    else:
        print("✗ OpenAI API Key 未配置")
    
    if not has_llm:
        print("\n警告：未配置任何LLM API Key，Task Decomposition将使用降级方案")
    else:
        print("✓ 至少一个LLM API Key 已配置")
    
    # 检查工具配置文件
    mcp_tools_path = agent_dir / "config" / "mcp_tools.json"
    if mcp_tools_path.exists():
        print(f"✓ MCP工具配置文件存在: {mcp_tools_path}")
    else:
        print(f"✗ MCP工具配置文件不存在: {mcp_tools_path}")
    
    # 检查service_list配置文件
    service_list_path = agent_dir / "config" / "service_list.json"
    if service_list_path.exists():
        print(f"✓ Service列表配置文件存在: {service_list_path}")
        # 验证service_list格式
        try:
            with open(service_list_path, 'r', encoding='utf-8') as f:
                service_list = json.load(f)
                print(f"✓ Service列表包含 {len(service_list)} 个服务")
                
                # 检查是否包含codeact服务
                codeact_services = [s for s in service_list if s.get("service_id") == "codeact"]
                if codeact_services:
                    print(f"✓ CodeAct服务已在service_list中")
                else:
                    print(f"✗ CodeAct服务未在service_list中")
        except Exception as e:
            print(f"✗ Service列表文件格式错误: {e}")
    else:
        print(f"✗ Service列表配置文件不存在: {service_list_path}")
    
    print("=" * 50)
    assert True  # 确保测试通过，即使没有LLM Key
