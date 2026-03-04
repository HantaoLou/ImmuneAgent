"""
Main Graph Q13 完整流程测试

测试用例 Q13: MART-1癌症表位TCR结合预测
- 任务: 预测哪些 TCR 结合 MART-1 癌症表位
- 输入: 2080 个 TCR 序列 (CDR3a + CDR3b)
- 输出: CSV 文件包含 main_name 和 prediction 列
- 主要指标: F1

测试流程:
START → supervisor → immunity → task_decomposition → executor → result_evaluator → END

架构原则 (2026-03-04 更新):
- CodeAct 子图是唯一与 OpenSandbox 沟通的入口
- 所有沙盒操作通过 utils/codeact_executor.py 统一接口
- supervisor 使用重构版本 graph_refactored.py
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# 添加 agent 目录到路径
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

# 设置环境变量
os.environ["OPENSANDBOX_ENABLED"] = "true"
os.environ["CODEACT_SANDBOX_PROVIDER"] = "opensandbox"
os.environ["OPENSANDBOX_SKIP_MCP_INSTALL"] = "true"


def verify_architecture_compliance():
    """验证架构一致性 - 确保没有直接调用 opensandbox_executor"""
    print("验证架构一致性...")
    
    import re
    from pathlib import Path
    
    # 允许直接调用的文件
    allowed_files = [
        "code_act/graph.py",
        "codeact_executor.py",
        "supervisor/graph.py",  # 旧版本（已废弃）
    ]
    
    violations = []
    agent_path = Path(agent_dir)
    
    for py_file in agent_path.rglob("*.py"):
        rel_path = str(py_file.relative_to(agent_path)).replace("\\", "/")
        
        # 跳过允许的文件
        if any(allowed in rel_path for allowed in allowed_files):
            continue
        
        # 跳过 __pycache__ 和测试文件
        if "__pycache__" in str(py_file) or "test_" in rel_path:
            continue
        
        try:
            content = py_file.read_text(encoding="utf-8")
            if "from utils.opensandbox_executor import" in content:
                violations.append(rel_path)
        except Exception:
            pass
    
    if violations:
        print("  ⚠️ 架构违规: 以下文件直接调用 opensandbox_executor:")
        for v in violations:
            print(f"    - {v}")
        return False
    else:
        print("  ✅ 架构一致: 所有非 CodeAct 子图都使用 codeact_executor")
        return True


def verify_codeact_executor_available():
    """验证 codeact_executor 模块可用"""
    print("验证 codeact_executor 模块...")
    
    try:
        from utils.codeact_executor import (
            execute_code_via_codeact,
            execute_code_via_codeact_async,
            is_codeact_available,
            CodeActResult,
            CodeActExecutionStatus,
            read_remote_file,
            list_remote_directory,
            copy_file_in_sandbox,
            convert_csv_to_fasta,
            convert_rds_to_csv,
            analyze_file_structure,
            prepare_nettcr_input,
        )
        print("  ✅ codeact_executor 模块加载成功")
        print("     包含所有新增便捷函数: convert_rds_to_csv, analyze_file_structure, prepare_nettcr_input")
        return True
    except ImportError as e:
        print(f"  ❌ codeact_executor 模块加载失败: {e}")
        return False


def verify_supervisor_refactored():
    """验证 supervisor 使用重构版本"""
    print("验证 supervisor 子图版本...")
    
    try:
        from nodes.subagents.supervisor import _USING_REFACTORED
        if _USING_REFACTORED:
            print("  ✅ 使用重构版本 (graph_refactored.py)")
            return True
        else:
            print("  ⚠️ 使用旧版本 (graph.py) - 建议迁移到重构版本")
            return False
    except ImportError as e:
        print(f"  ❌ 无法导入 supervisor: {e}")
        return False


def create_q13_test_state():
    """创建 Q13 测试用例的 GlobalState"""
    from state import GlobalState, UserTaskType
    
    # Q13 测试配置
    q13_prompt = """Given 2080 T cell receptors with paired CDR3 alpha (CDR3a) and CDR3 beta (CDR3b) sequences, predict which TCRs bind the MART-1 cancer epitope (peptide: ELAGIGILTV, presented by HLA-A*02:01). MART-1 is a melanoma-associated antigen. For each TCR (identified by `main_name`), output a binary prediction: True = binder, False = non-binder.

What to use:
- CDR3a and CDR3b amino acid sequences for each TCR
- Target peptide: ELAGIGILTV
- HLA restriction: A*02:01
- TCR V/J gene usage annotations
- TCR-epitope binding prediction tools (e.g., NetTCR-2.0)

Expected output format:
- CSV with columns: `main_name`, `prediction` (True or False), optionally `probability` (0.0-1.0)
- 2080 rows (one per TCR)

Ground truth summary:
- 2080 TCRs tested
- 60 positive (2.9%), 2020 negative
- Highly imbalanced -- only ~3% are MART-1 binders

Primary metric: F1

Input files:
- meta_csv_file: /data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv
- meta_rds_file: /data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.rds
"""
    
    # 生成 session ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{timestamp}_q13_main_graph"
    
    # 创建沙盒目录
    sandbox_dir = f"/data/sessions/{session_id}"
    sandbox_data_dir = f"/data/sessions/{session_id}"
    
    # 创建 GlobalState
    state = GlobalState(
        user_input=q13_prompt,
        user_task_type=UserTaskType.IMMUNOLOGY_TASK,  # 免疫学任务
        sandbox_dir=sandbox_dir,
        sandbox_data_dir=sandbox_data_dir,
        session_id=session_id,
        file_paths={
            "meta_csv_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv",
            "meta_rds_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.rds"
        }
    )
    
    return state, session_id


def test_main_graph_q13():
    """测试 main_graph 完整流程 - Q13"""
    print("=" * 80)
    print("Main Graph Q13 完整流程测试")
    print("=" * 80)
    print()
    
    # Step 0: 验证架构一致性
    print("Step 0: 验证架构一致性...")
    verify_codeact_executor_available()
    verify_supervisor_refactored()
    verify_architecture_compliance()
    print()
    
    # Step 1: 创建测试状态
    print("Step 1: 创建测试状态...")
    state, session_id = create_q13_test_state()
    print(f"  Session ID: {session_id}")
    print(f"  Sandbox Dir: {state.sandbox_dir}")
    print(f"  Task Type: {state.user_task_type}")
    print()
    
    # Step 2: 构建主图
    print("Step 2: 构建主图...")
    try:
        from main_graph import build_main_graph
        graph = build_main_graph()
        print(f"  节点: {list(graph.nodes.keys())}")
        print()
    except Exception as e:
        print(f"  ❌ 构建主图失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 3: 执行主图
    print("Step 3: 执行主图...")
    print("  流程: START → supervisor → immunity → task_decomposition → executor → result_evaluator → END")
    print()
    
    try:
        result = graph.invoke(state)
        print()
        print("=" * 80)
        print("主图执行完成!")
        print("=" * 80)
        
        # 处理 result 可能是 dict 的情况
        if isinstance(result, dict):
            result_state = result
        else:
            result_state = result
        
        # 打印结果摘要
        print()
        print("结果摘要:")
        print("-" * 40)
        
        # 安全获取属性，支持 dict 和对象两种形式
        def safe_get(obj, key, default="N/A"):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        
        print(f"  Session ID: {safe_get(result_state, 'session_id')}")
        print(f"  Task Type: {safe_get(result_state, 'user_task_type')}")
        
        subtasks = safe_get(result_state, 'subtasks', [])
        if subtasks:
            print(f"  Subtasks: {len(subtasks)}")
        else:
            print(f"  Subtasks: 0")
        
        parallel_groups = safe_get(result_state, 'parallel_task_groups', {})
        if parallel_groups:
            print(f"  Parallel Groups: {len(parallel_groups)}")
        
        completed_tasks = safe_get(result_state, 'completed_tasks', {})
        if completed_tasks:
            print(f"  Completed Tasks: {len(completed_tasks)}")
        
        merged_result = safe_get(result_state, 'merged_result', {})
        if merged_result:
            print()
            print("  Merged Result Keys:")
            for key in merged_result.keys():
                value = merged_result[key]
                if isinstance(value, dict):
                    print(f"    - {key}: {len(value)} items")
                elif isinstance(value, str) and len(value) > 100:
                    print(f"    - {key}: {value[:100]}...")
                else:
                    print(f"    - {key}: {value}")
        
        # 检查 result_evaluator 输出
        merged = merged_result if isinstance(merged_result, dict) else {}
        if merged and "result_evaluation" in merged:
            evaluation = merged["result_evaluation"]
            print()
            print("  Result Evaluation:")
            print(f"    - Summary: {evaluation.get('summary_report', 'N/A')[:200]}...")
            if evaluation.get("txt_report_path"):
                print(f"    - TXT Report: {evaluation['txt_report_path']}")
            if evaluation.get("key_findings"):
                print(f"    - Key Findings: {len(evaluation['key_findings'])} items")
        
        # 检查沙盒目录结构
        sandbox = safe_get(result_state, 'sandbox_dir', '')
        opensandbox_id = safe_get(result_state, 'opensandbox_id', '')
        if sandbox:
            print()
            print("  沙盒信息:")
            print(f"    - Sandbox Dir: {sandbox}")
            print(f"    - OpenSandbox ID: {opensandbox_id}")
        
        # 检查 todo-list.md
        if sandbox:
            todo_list_path = Path(sandbox) / "todo-list.md"
            if todo_list_path.exists():
                print()
                print(f"  Todo List: {todo_list_path}")
                with open(todo_list_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"    Size: {len(content)} bytes")
        
        return result
        
    except Exception as e:
        print(f"  ❌ 执行主图失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_step_by_step():
    """分步测试各个节点"""
    print("=" * 80)
    print("分步测试各个节点")
    print("=" * 80)
    print()
    
    # 验证架构一致性
    verify_codeact_executor_available()
    verify_supervisor_refactored()
    print()
    
    # 创建测试状态
    state, session_id = create_q13_test_state()
    
    # Step 1: 测试 supervisor 节点
    print("Step 1: 测试 supervisor 节点...")
    try:
        from main_graph import supervisor_node
        state = supervisor_node(state)
        print(f"  ✅ supervisor 完成")
        
        # 安全获取属性
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        
        print(f"     Session ID: {safe_get(state, 'session_id')}")
        print(f"     Task Type: {safe_get(state, 'user_task_type')}")
        extracted_params = safe_get(state, 'extracted_parameters')
        print(f"     Extracted Parameters: {len(extracted_params) if extracted_params else 0} keys")
        opensandbox_id = safe_get(state, 'opensandbox_id')
        print(f"     OpenSandbox ID: {opensandbox_id}")
    except Exception as e:
        print(f"  ❌ supervisor 失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    print()
    
    # Step 2: 测试 immunity 节点
    print("Step 2: 测试 immunity 节点...")
    try:
        from main_graph import immunity_node
        state = immunity_node(state)
        print(f"  ✅ immunity 完成")
        execution_plan = safe_get(state, 'execution_plan')
        if execution_plan:
            print(f"     Execution Plan: {execution_plan[:200]}...")
    except Exception as e:
        print(f"  ❌ immunity 失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    print()
    
    # Step 3: 测试 task_decomposition 节点
    print("Step 3: 测试 task_decomposition 节点...")
    try:
        from main_graph import task_decomposition_node
        state = task_decomposition_node(state)
        print(f"  ✅ task_decomposition 完成")
        subtasks = safe_get(state, 'subtasks')
        parallel_groups = safe_get(state, 'parallel_task_groups')
        print(f"     Subtasks: {len(subtasks) if subtasks else 0}")
        print(f"     Parallel Groups: {len(parallel_groups) if parallel_groups else 0}")
    except Exception as e:
        print(f"  ❌ task_decomposition 失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    print()
    
    # Step 4: 测试 executor 节点
    print("Step 4: 测试 executor 节点...")
    print("  (这可能需要几分钟...)")
    try:
        from main_graph import executor_node
        state = executor_node(state)
        print(f"  ✅ executor 完成")
        completed_tasks = safe_get(state, 'completed_tasks')
        print(f"     Completed Tasks: {len(completed_tasks) if completed_tasks else 0}")
        merged_result = safe_get(state, 'merged_result')
        if merged_result and "executor_results" in merged_result:
            print(f"     Executor Results: {list(merged_result['executor_results'].keys())}")
    except Exception as e:
        print(f"  ❌ executor 失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    print()
    
    # Step 5: 测试 result_evaluator 节点
    print("Step 5: 测试 result_evaluator 节点...")
    try:
        from main_graph import result_evaluator_node
        state = result_evaluator_node(state)
        print(f"  ✅ result_evaluator 完成")
        merged_result = safe_get(state, 'merged_result')
        if merged_result and "result_evaluation" in merged_result:
            evaluation = merged_result["result_evaluation"]
            print(f"     Summary Report: {evaluation.get('summary_report', 'N/A')[:100]}...")
            if evaluation.get("txt_report_path"):
                print(f"     TXT Report: {evaluation['txt_report_path']}")
    except Exception as e:
        print(f"  ❌ result_evaluator 失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    print()
    
    print("=" * 80)
    print("所有节点测试完成!")
    print("=" * 80)
    
    return state


def test_codeact_executor_functions():
    """测试 codeact_executor 新增函数"""
    print("=" * 80)
    print("测试 codeact_executor 新增函数")
    print("=" * 80)
    print()
    
    try:
        from utils.codeact_executor import (
            is_codeact_available,
            execute_code_via_codeact,
            convert_rds_to_csv,
            analyze_file_structure,
            prepare_nettcr_input,
            CodeActResult,
            CodeActExecutionStatus,
        )
        
        # 测试 1: 检查 CodeAct 可用性
        print("测试 1: 检查 CodeAct 可用性...")
        available = is_codeact_available()
        if available:
            print("  ✅ CodeAct 可用")
        else:
            print("  ⚠️ CodeAct 不可用 (可能 OpenSandbox 未启用)")
        print()
        
        # 测试 2: 测试 CodeActResult 类
        print("测试 2: 测试 CodeActResult 类...")
        result = CodeActResult(
            status=CodeActExecutionStatus.SUCCESS,
            output="test output",
            error="",
            sandbox_id="test-sandbox-id"
        )
        assert result.is_success(), "is_success() 应返回 True"
        result_dict = result.to_dict()
        assert "status" in result_dict, "to_dict() 应包含 status"
        print("  ✅ CodeActResult 类正常工作")
        print()
        
        # 测试 3: 测试函数签名
        print("测试 3: 验证函数签名...")
        import inspect
        
        # convert_rds_to_csv
        sig = inspect.signature(convert_rds_to_csv)
        params = list(sig.parameters.keys())
        assert "rds_path" in params, "convert_rds_to_csv 应有 rds_path 参数"
        assert "sandbox_id" in params, "convert_rds_to_csv 应有 sandbox_id 参数"
        print("  ✅ convert_rds_to_csv(rds_path, output_csv_path, sandbox_id)")
        
        # analyze_file_structure
        sig = inspect.signature(analyze_file_structure)
        params = list(sig.parameters.keys())
        assert "file_path" in params, "analyze_file_structure 应有 file_path 参数"
        print("  ✅ analyze_file_structure(file_path, sandbox_id)")
        
        # prepare_nettcr_input
        sig = inspect.signature(prepare_nettcr_input)
        params = list(sig.parameters.keys())
        assert "input_csv" in params, "prepare_nettcr_input 应有 input_csv 参数"
        print("  ✅ prepare_nettcr_input(input_csv, output_path, tcr_columns, sandbox_id)")
        print()
        
        print("=" * 80)
        print("🎉 codeact_executor 新增函数测试通过!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Main Graph Q13 测试")
    parser.add_argument("--step-by-step", action="store_true", help="分步测试各个节点")
    parser.add_argument("--full", action="store_true", help="执行完整流程测试")
    parser.add_argument("--verify-arch", action="store_true", help="仅验证架构一致性")
    parser.add_argument("--test-codeact", action="store_true", help="测试 codeact_executor 函数")
    args = parser.parse_args()
    
    if args.step_by_step:
        test_step_by_step()
    elif args.verify_arch:
        print("=" * 80)
        print("架构一致性验证")
        print("=" * 80)
        print()
        verify_codeact_executor_available()
        verify_supervisor_refactored()
        verify_architecture_compliance()
    elif args.test_codeact:
        test_codeact_executor_functions()
    elif args.full:
        test_main_graph_q13()
    else:
        # 默认执行架构验证 + 完整流程
        print("使用 --step-by-step 分步测试")
        print("使用 --full 执行完整流程测试")
        print("使用 --verify-arch 验证架构一致性")
        print("使用 --test-codeact 测试 codeact_executor 函数")
        print()
        test_main_graph_q13()


if __name__ == "__main__":
    main()
