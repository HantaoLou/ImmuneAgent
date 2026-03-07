# -*- coding: utf-8 -*-
"""
Iterative Executor 全流程测试 (带完整日志记录)

测试 Q13 用例在新流程 (iterative_executor) 中的数据流转：
START → supervisor → [路由]
       └── iterative_executor → result_evaluator → END

日志记录内容：
1. 用户输入
2. 参数表 (extracted_parameters)
3. Deep Research 结果
4. HYPOTHESIS
5. Planning (execution_plan)
6. 任务执行产物 (iteration_history, output_files, mcp_calls)
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# 添加 agent 目录到路径
agent_dir = Path(__file__).parent.parent
if str(agent_dir) not in sys.path:
    sys.path.insert(0, str(agent_dir))

print(f"[路径] agent_dir = {agent_dir}")

# 加载环境变量
try:
    from dotenv import load_dotenv
    env_path = agent_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[环境] 已加载 .env: {env_path}")
    else:
        print(f"[环境] .env 文件不存在: {env_path}")
except ImportError:
    print("[环境] python-dotenv 未安装，跳过 .env 加载")
except Exception as e:
    print(f"[警告] 加载环境变量失败: {e}")

# 调试：检查关键环境变量
print(f"[调试] OPENSANDBOX_ENABLED = {os.getenv('OPENSANDBOX_ENABLED', '未设置')}")
print(f"[调试] CODEACT_SANDBOX_PROVIDER = {os.getenv('CODEACT_SANDBOX_PROVIDER', '未设置')}")
print(f"[调试] SANDBOX_DOMAIN = {os.getenv('SANDBOX_DOMAIN', '未设置')}")


# ============================================================================
# 日志记录器
# ============================================================================

class WorkflowLogger:
    """全流程日志记录器"""
    
    def __init__(self, session_id: str, log_dir: str = None):
        self.session_id = session_id
        self.log_dir = Path(log_dir) if log_dir else Path(__file__).parent / "execution_logs" / session_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_data = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            
            # 1. 用户输入
            "user_input": None,
            
            # 2. 参数表
            "parameter_table": None,
            
            # 3. Supervisor 结果
            "supervisor_result": None,
            
            # 4. Deep Research 结果
            "deep_research_result": None,
            
            # 5. HYPOTHESIS
            "hypothesis": None,
            
            # 6. Planning
            "planning": None,
            
            # 7. 任务执行产物
            "execution_artifacts": {
                "iteration_history": [],
                "output_files": [],
                "mcp_calls": [],
            },
            
            # 8. 最终结果
            "final_result": None,
        }
    
    def log_user_input(self, user_input: str, test_case: Dict = None):
        """记录用户输入"""
        self.log_data["user_input"] = {
            "content": user_input,
            "length": len(user_input),
            "test_case": test_case,
        }
        self._print_section("用户输入", user_input[:500] + "..." if len(user_input) > 500 else user_input)
    
    def log_parameter_table(self, params: Dict[str, Any]):
        """记录参数表"""
        self.log_data["parameter_table"] = params
        self._print_section("参数表", json.dumps(params, indent=2, ensure_ascii=False, default=str))
    
    def log_supervisor_result(self, result: Dict[str, Any]):
        """记录 Supervisor 结果"""
        self.log_data["supervisor_result"] = {
            "session_id": result.get("session_id"),
            "user_task_type": result.get("user_task_type"),
            "supervisor_decision": result.get("supervisor_decision"),
            "file_paths": result.get("file_paths"),
            "file_analyses": result.get("file_analyses"),
        }
        self._print_section("Supervisor 结果", json.dumps(
            {k: v for k, v in result.items() if k in ["session_id", "user_task_type", "supervisor_decision"]},
            indent=2, ensure_ascii=False, default=str
        ))
    
    def log_deep_research(self, result: Any):
        """记录 Deep Research 结果"""
        if result is None:
            return
        
        deep_research_data = None
        if isinstance(result, dict):
            deep_research_data = result
        elif hasattr(result, 'deep_research_result'):
            deep_research_data = getattr(result, 'deep_research_result', None)
        elif hasattr(result, 'research_summary'):
            deep_research_data = {
                "research_summary": getattr(result, 'research_summary', None),
                "context": getattr(result, 'context', None),
            }
        
        self.log_data["deep_research_result"] = deep_research_data
        if deep_research_data:
            self._print_section("Deep Research 结果", 
                json.dumps(deep_research_data, indent=2, ensure_ascii=False, default=str)[:2000])
    
    def log_hypothesis(self, result: Any):
        """记录 HYPOTHESIS"""
        hypothesis_data = None
        
        if isinstance(result, dict):
            hypothesis_data = result
        elif hasattr(result, 'hypothesis_summary'):
            hypothesis_data = {
                "hypothesis_summary": getattr(result, 'hypothesis_summary', None),
                "hypothesis": getattr(result, 'hypothesis', None),
                "hypothesis_confidence": getattr(result, 'hypothesis_confidence', None),
                "testable_predictions": getattr(result, 'testable_predictions', None),
            }
        
        self.log_data["hypothesis"] = hypothesis_data
        if hypothesis_data:
            summary = hypothesis_data.get("hypothesis_summary", "")
            confidence = hypothesis_data.get("hypothesis_confidence", "N/A")
            self._print_section("HYPOTHESIS", 
                f"置信度: {confidence}\n\n{summary[:1500] if summary else '无'}")
    
    def log_planning(self, execution_plan: str, planning_details: Dict = None):
        """记录 Planning"""
        self.log_data["planning"] = {
            "execution_plan": execution_plan,
            "details": planning_details,
        }
        if execution_plan:
            self._print_section("Planning (execution_plan)", 
                execution_plan[:2000] if len(execution_plan) > 2000 else execution_plan)
    
    def log_execution_artifacts(self, 
                                 iteration_history: List = None, 
                                 output_files: List = None, 
                                 mcp_calls: List = None):
        """记录任务执行产物"""
        if iteration_history:
            self.log_data["execution_artifacts"]["iteration_history"] = iteration_history
        if output_files:
            self.log_data["execution_artifacts"]["output_files"] = output_files
        if mcp_calls:
            self.log_data["execution_artifacts"]["mcp_calls"] = mcp_calls
        
        artifacts = self.log_data["execution_artifacts"]
        summary = f"""
迭代次数: {len(artifacts['iteration_history'])}
输出文件: {len(artifacts['output_files'])} 个
MCP 调用: {len(artifacts['mcp_calls'])} 次

迭代历史:
{json.dumps(artifacts['iteration_history'], indent=2, ensure_ascii=False, default=str)[:1000]}

输出文件列表:
{json.dumps(artifacts['output_files'], indent=2, ensure_ascii=False, default=str)[:1000]}

MCP 调用记录:
{json.dumps(artifacts['mcp_calls'], indent=2, ensure_ascii=False, default=str)[:1000]}
"""
        self._print_section("任务执行产物", summary)
    
    def log_final_result(self, result: Dict[str, Any]):
        """记录最终结果"""
        self.log_data["final_result"] = result
        self.log_data["end_time"] = datetime.now().isoformat()
        
        summary = {
            "session_id": result.get("session_id"),
            "iteration_count": len(result.get("iteration_history", [])),
            "output_file_count": len(result.get("completed_output_files", [])),
            "mcp_call_count": len(result.get("mcp_call_records", [])),
            "merged_result_keys": list(result.get("merged_result", {}).keys()) if result.get("merged_result") else [],
        }
        self._print_section("最终结果", json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    
    def save(self):
        """保存日志到文件"""
        # 保存完整 JSON
        json_path = self.log_dir / "full_workflow_log.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[日志] 完整日志已保存: {json_path}")
        
        # 保存 Markdown 报告
        md_path = self.log_dir / "workflow_report.md"
        self._save_markdown_report(md_path)
        print(f"[日志] Markdown 报告已保存: {md_path}")
        
        return json_path, md_path
    
    def _save_markdown_report(self, path: Path):
        """保存 Markdown 格式报告"""
        lines = [
            f"# 全流程执行报告",
            "",
            f"> Session ID: `{self.session_id}`",
            f"> 开始时间: {self.log_data['start_time']}",
            f"> 结束时间: {self.log_data['end_time'] or 'N/A'}",
            "",
            "---",
            "",
            "## 1. 用户输入",
            "",
            "```",
            self.log_data["user_input"]["content"] if self.log_data.get("user_input") else "N/A",
            "```",
            "",
            "## 2. 参数表",
            "",
            "```json",
            json.dumps(self.log_data.get("parameter_table"), indent=2, ensure_ascii=False, default=str),
            "```",
            "",
            "## 3. Supervisor 结果",
            "",
            "```json",
            json.dumps(self.log_data.get("supervisor_result"), indent=2, ensure_ascii=False, default=str),
            "```",
            "",
            "## 4. Deep Research 结果",
            "",
            "```",
            str(self.log_data.get("deep_research_result") or "N/A")[:3000],
            "```",
            "",
            "## 5. HYPOTHESIS",
            "",
            "```",
            str(self.log_data.get("hypothesis") or "N/A")[:3000],
            "```",
            "",
            "## 6. Planning (execution_plan)",
            "",
            "```",
            str((self.log_data.get("planning") or {}).get("execution_plan") or "N/A")[:3000],
            "```",
            "",
            "## 7. 任务执行产物",
            "",
            "### 7.1 迭代历史",
            "",
            "```json",
            json.dumps(self.log_data.get("execution_artifacts", {}).get("iteration_history", []), 
                      indent=2, ensure_ascii=False, default=str)[:3000],
            "```",
            "",
            "### 7.2 输出文件",
            "",
            "```json",
            json.dumps(self.log_data.get("execution_artifacts", {}).get("output_files", []), 
                      indent=2, ensure_ascii=False, default=str),
            "```",
            "",
            "### 7.3 MCP 调用记录",
            "",
            "```json",
            json.dumps(self.log_data.get("execution_artifacts", {}).get("mcp_calls", []), 
                      indent=2, ensure_ascii=False, default=str)[:2000],
            "```",
            "",
            "## 8. 最终结果",
            "",
            "```json",
            json.dumps(self.log_data.get("final_result"), indent=2, ensure_ascii=False, default=str)[:3000],
            "```",
            "",
            "---",
            "",
            f"*报告生成时间: {datetime.now().isoformat()}*",
        ]
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _print_section(self, title: str, content: str):
        """打印分节"""
        print("\n" + "=" * 70)
        print(f"[{title}]")
        print("=" * 70)
        print(content)
        print("=" * 70)


# ============================================================================
# Q13 测试用例
# ============================================================================
Q13_TEST_CASE = {
    "id": "Q13",
    "name": "MART-1癌症表位TCR结合预测",
    "difficulty": "simple",
    "user_input": """Given 2080 T cell receptors with paired CDR3 alpha (CDR3a) and CDR3 beta (CDR3b) sequences, predict which TCRs bind the MART-1 cancer epitope (peptide: ELAGIGILTV, presented by HLA-A*02:01). MART-1 is a melanoma-associated antigen. For each TCR (identified by `main_name`), output a binary prediction: True = binder, False = non-binder.

What to use:
- CDR3a and CDR3b sequences for each TCR
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
""",
    "services": ["nettcr"],
    "parameters": {
        "target_peptide": "ELAGIGILTV",
        "hla": "A*02:01",
        "rds_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_benchmark.rds",
        "meta_csv_file": "/data/benchmark_data/tcr_icon_benchmark/260204_tcr_icon_metadata.csv",
    }
}


# ============================================================================
# 实际全流程执行测试
# ============================================================================

async def run_full_workflow_test(test_case: Dict = None):
    """
    执行完整的全流程测试（带详细日志记录）
    
    完整流程：supervisor → immunity → iterative_executor → result_evaluator
    
    Args:
        test_case: 测试用例，默认使用 Q13
    """
    test_case = test_case or Q13_TEST_CASE
    
    session_id = f"full_{test_case['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = WorkflowLogger(session_id)
    
    print("\n" + "=" * 70)
    print("全流程执行测试 (带详细日志记录)")
    print("=" * 70)
    print(f"测试用例: {test_case['name']} (ID: {test_case['id']})")
    print(f"Session ID: {session_id}")
    print(f"流程: supervisor → immunity → iterative_executor → result_evaluator")
    print("=" * 70)
    
    try:
        # ====================================================================
        # Step 1: 记录用户输入
        # ====================================================================
        logger.log_user_input(test_case["user_input"], test_case)
        
        # ====================================================================
        # Step 2: 准备 GlobalState
        # ====================================================================
        print("\n[Step 2] 准备 GlobalState...")
        
        from state import GlobalState, UserTaskType, ensure_global_state_rebuilt
        ensure_global_state_rebuilt()
        
        # 注意：不预设 user_task_type，让 supervisor 自动分类
        # 这样可以测试完整的 supervisor → immunity → iterative_executor 流程
        global_state = GlobalState(
            user_input=test_case["user_input"],
            sandbox_dir=f"/tmp/sessions/{session_id}",
            session_id=session_id,
            # user_task_type 不设置，由 supervisor 分类
            file_paths={},
            extracted_parameters=test_case["parameters"],
        )
        
        # 记录参数表
        logger.log_parameter_table(test_case["parameters"])
        
        print(f"  ✓ GlobalState 创建成功")
        print(f"    - session_id: {global_state.session_id}")
        print(f"    - user_task_type: {global_state.user_task_type}")
        
        # ====================================================================
        # Step 3: 构建主图
        # ====================================================================
        print("\n[Step 3] 构建主图...")
        
        from main_graph import build_main_graph
        graph = build_main_graph(use_iterative_executor=True)
        
        print(f"  ✓ 主图构建成功")
        print(f"    - 节点列表: {list(graph.nodes.keys())}")
        
        # ====================================================================
        # Step 4: 执行全流程
        # ====================================================================
        print("\n[Step 4] 执行全流程...")
        print("  这可能需要几分钟时间...")
        
        # 实际执行
        result = await graph.ainvoke(global_state)
        
        # ====================================================================
        # Step 5: 提取并记录各阶段结果
        # ====================================================================
        print("\n[Step 5] 提取并记录各阶段结果...")
        
        # 5.1 Supervisor 结果
        supervisor_result = {
            "session_id": result.get("session_id"),
            "user_task_type": result.get("user_task_type"),
            "supervisor_decision": result.get("supervisor_decision"),
            "file_paths": result.get("file_paths"),
            "file_analyses": result.get("file_analyses"),
        }
        logger.log_supervisor_result(supervisor_result)
        
        # 5.2 Deep Research 结果 (来自 immunity)
        # 如果执行了 immunity 子图，提取相关字段
        deep_research_data = result.get("merged_result", {}).get("deep_research_result")
        if not deep_research_data:
            deep_research_data = result.get("research_summary")
        logger.log_deep_research(deep_research_data)
        
        # 5.3 HYPOTHESIS (来自 immunity)
        hypothesis_data = None
        if result.get("merged_result"):
            mr = result["merged_result"]
            hypothesis_data = {
                "hypothesis_summary": mr.get("hypothesis_summary"),
                "hypothesis_confidence": mr.get("hypothesis_confidence"),
            }
        logger.log_hypothesis(hypothesis_data)
        
        # 5.4 Planning (execution_plan)
        execution_plan = result.get("execution_plan")
        planning_details = {
            "final_enhanced_plan": result.get("merged_result", {}).get("final_enhanced_plan"),
            "plan_steps": result.get("merged_result", {}).get("plan_steps"),
        }
        logger.log_planning(execution_plan, planning_details)
        
        # 5.5 任务执行产物
        logger.log_execution_artifacts(
            iteration_history=result.get("iteration_history", []),
            output_files=result.get("completed_output_files", []),
            mcp_calls=result.get("mcp_call_records", []),
        )
        
        # 5.6 最终结果
        logger.log_final_result(result)
        
        # ====================================================================
        # Step 6: 保存日志
        # ====================================================================
        print("\n[Step 6] 保存日志...")
        json_path, md_path = logger.save()
        
        print("\n" + "=" * 70)
        print("✅ 全流程测试完成!")
        print("=" * 70)
        
        return result
        
    except Exception as e:
        print(f"\n❌ 全流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 即使失败也保存已记录的日志
        logger.save()
        
        return None


# ============================================================================
# 模拟测试 (不实际执行)
# ============================================================================

def run_simulation_tests():
    """运行模拟测试（不实际执行沙盒和 MCP）"""
    print("\n" + "=" * 70)
    print("模拟测试 (不实际执行)")
    print("=" * 70)
    
    results = {}
    
    # 测试 1: 数据流转
    print("\n[1/5] 测试数据流转...")
    try:
        from state import GlobalState, UserTaskType, ensure_global_state_rebuilt
        ensure_global_state_rebuilt()
        
        global_state = GlobalState(
            user_input=Q13_TEST_CASE["user_input"],
            sandbox_dir="/tmp/test_q13_iterative",
            session_id=f"test_q13_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_task_type=UserTaskType.EXECUTE_PLAN,
            file_paths={},
            extracted_parameters=Q13_TEST_CASE["parameters"],
        )
        print("  ✓ GlobalState 创建成功")
        results["数据流转"] = True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results["数据流转"] = False
    
    # 测试 2: 输入映射
    print("\n[2/5] 测试输入映射...")
    try:
        from nodes.subagents.iterative_executor.input_mapper import iterative_executor_input_mapper
        executor_state = iterative_executor_input_mapper(global_state)
        print(f"  ✓ 映射成功, mcp_services: {executor_state.mcp_services}")
        results["输入映射"] = True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results["输入映射"] = False
    
    # 测试 3: 任务生成
    print("\n[3/5] 测试任务生成...")
    try:
        from nodes.subagents.iterative_executor.task_generator import generate_tasks_md
        tasks_md = generate_tasks_md(executor_state, use_llm=False)
        print(f"  ✓ 生成 tasks.md ({len(tasks_md)} 字符)")
        results["任务生成"] = True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results["任务生成"] = False
    
    # 测试 4: 主图构建
    print("\n[4/5] 测试主图构建...")
    try:
        from main_graph import build_main_graph
        graph = build_main_graph(use_iterative_executor=True)
        nodes = list(graph.nodes.keys())
        required = ["supervisor", "iterative_executor", "result_evaluator"]
        missing = [n for n in required if n not in nodes]
        if missing:
            print(f"  ✗ 缺少节点: {missing}")
            results["主图构建"] = False
        else:
            print(f"  ✓ 主图构建成功: {nodes}")
            results["主图构建"] = True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results["主图构建"] = False
    
    # 测试 5: 输出映射
    print("\n[5/5] 测试输出映射...")
    try:
        from nodes.subagents.iterative_executor.output_mapper import iterative_executor_output_mapper
        
        # 模拟执行结果
        executor_state.output_files = ["/tmp/output.csv"]
        executor_state.mcp_calls = [{"tool": "test", "params": {}}]
        executor_state.iteration_history = [{"iteration": 1, "status": "success"}]
        
        updated_state = iterative_executor_output_mapper(executor_state, global_state)
        print(f"  ✓ 输出映射成功")
        print(f"    - iteration_history: {len(updated_state.iteration_history)} 条")
        print(f"    - completed_output_files: {len(updated_state.completed_output_files)} 个")
        results["输出映射"] = True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results["输出映射"] = False
    
    # 汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    all_passed = True
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("🎉 所有模拟测试通过!")
        print("提示: 运行 --real 进行实际执行测试")
    else:
        print("⚠️ 部分测试失败")
    
    return all_passed


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Iterative Executor 全流程测试")
    parser.add_argument("--real", action="store_true", help="运行实际执行测试 (需要沙盒环境)")
    args = parser.parse_args()
    
    if args.real:
        # 实际执行测试（完整流程：supervisor → immunity → iterative_executor → result_evaluator）
        asyncio.run(run_full_workflow_test())
    else:
        # 模拟测试
        run_simulation_tests()
