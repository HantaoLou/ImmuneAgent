"""
Planning graph for ImmuneAgent workflow.
Orchestrates the research planning, execution, and validation pipeline.
"""

import asyncio
import uuid
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from usecases.immunology.immunology_config import get_immunology_runnable_config
from usecases.immunology.prompts.immunology_prompts import ImmunologyPrompts
from usecases.immunology.state.state import ImmuneAgentState
from usecases.immunology.tools.execution_tools import ToolExecutor
from usecases.immunology.tools.hypothesis_tools import HypothesisGenerator
from usecases.immunology.tools.planning_tools import PlanningEngine
from usecases.immunology.tools.retrieval_tools import ImmunologyRetriever


class PlanningNode:
    """Node for comprehensive research planning."""

    def __init__(self):
        """Initialize planning node."""
        self.retriever = ImmunologyRetriever()
        self.hypothesis_generator = HypothesisGenerator()
        self.planning_engine = PlanningEngine()
        self.prompts = ImmunologyPrompts()

    async def __call__(
        self, state: ImmuneAgentState, config: RunnableConfig
    ) -> ImmuneAgentState:
        """Generate comprehensive research plan."""

        print(f"\n📋 PLANNING NODE")
        print(f"  Question: {state.research_question[:80]}...")

        # Step 1: Retrieve relevant context with reranking
        print("  📚 Retrieving scientific context...")
        results = self.retriever.retrieve_with_rerank(
            state.research_question, k=20, rerank_k=10
        )

        # Format context and extract citations
        context_parts = []
        citations = []
        for result in results:
            context_parts.append(result["content"])
            citations.append(result["citation"])

        state.context = "\n\n".join(context_parts)
        state.citations = list(set(citations))  # Unique citations
        state.retrieved_documents = results

        # Step 2: Generate hypotheses
        print("  💡 Generating hypotheses...")
        hypotheses = await self.hypothesis_generator.generate_hypotheses_async(
            state.research_question, state.context, config
        )

        state.hypotheses = [h.statement for h in hypotheses]
        state.hypothesis_result = hypotheses

        # Step 3: Create research plan
        print("  🔬 Creating research plan...")
        plan = await self.planning_engine.create_research_plan_async(
            state.research_question, state.analysis_type, state.context, hypotheses
        )

        # Update state with plan details
        state.research_plan = plan
        state.selected_tools = [tool.tool_name for tool in plan.selected_tools]
        state.methodology = {
            phase: [step.description for step in steps]
            for phase, steps in plan.phases.items()
        }

        # Set confidence scores
        state.confidence_scores["planning"] = plan.confidence_score
        state.confidence_scores["feasibility"] = plan.feasibility_score
        state.confidence_scores["evidence_support"] = len(results) / 20.0

        print(f"  ✅ Plan created with {len(state.selected_tools)} tools")
        print(f"     Tools: {', '.join(state.selected_tools[:5])}...")
        print(f"     Citations: {len(state.citations)}")

        return state


class ExecutionNode:
    """Node for executing selected tools."""

    def __init__(self):
        """Initialize execution node."""
        self.tool_executor = ToolExecutor(
            use_mcp=False
        )  # Set to True when MCP is available

    async def __call__(
        self, state: ImmuneAgentState, config: RunnableConfig
    ) -> ImmuneAgentState:
        """Execute selected tools in parallel batches."""

        print(f"\n⚡ EXECUTION NODE")
        print(f"  Executing {len(state.selected_tools)} tools...")

        # Prepare tool requests
        tool_requests = [
            {
                "tool_name": tool_name,
                "parameters": state.execution_parameters.get(tool_name, {}),
            }
            for tool_name in state.selected_tools
        ]

        # Execute tools in batches
        execution_results = await self.tool_executor.execute_batch(
            tool_requests, max_parallel=5
        )

        # Update state
        state.execution_results = execution_results
        state.analysis_completed = True

        # Extract key findings
        state.key_findings = self._extract_key_findings(execution_results)

        # Count successful executions
        successful = sum(
            1 for r in execution_results.values() if r.get("status") != "error"
        )

        print(
            f"  ✅ Execution completed: {successful}/{len(execution_results)} successful"
        )

        return state

    def _extract_key_findings(self, results: Dict) -> List[str]:
        """Extract key findings from results."""
        findings = []

        for tool, result in results.items():
            if result.get("status") != "error" and "result" in result:
                # Create finding based on tool type
                if "binding_score" in str(result.get("result", {})):
                    findings.append(f"{tool}: Binding prediction completed")
                elif "structure" in str(result.get("result", {})):
                    findings.append(f"{tool}: Structure prediction completed")
                elif "clusters" in str(result.get("result", {})):
                    findings.append(f"{tool}: Clustering analysis completed")
                else:
                    findings.append(f"{tool}: Analysis completed successfully")

        return findings


class ValidationNode:
    """Node for validating and synthesizing results."""

    def __init__(self):
        """Initialize validation node."""
        self.prompts = ImmunologyPrompts()

    async def __call__(
        self, state: ImmuneAgentState, config: RunnableConfig
    ) -> ImmuneAgentState:
        """Validate and synthesize results."""

        print(f"\n✅ VALIDATION NODE")
        print(f"  Validating results and generating report...")

        # Calculate validation metrics
        successful_tools = sum(
            1 for r in state.execution_results.values() if r.get("status") != "error"
        )
        total_tools = len(state.execution_results)

        state.confidence_scores["execution"] = successful_tools / max(total_tools, 1)
        state.confidence_scores["overall"] = sum(
            state.confidence_scores.values()
        ) / len(state.confidence_scores)

        # Generate synthesis using enhanced prompt
        synthesis_prompt = ChatPromptTemplate.from_template(
            self.prompts.VALIDATION_SYNTHESIS_PROMPT
        )
        from common.factory import get_reasoning_model

        model = get_reasoning_model(config)
        chain = synthesis_prompt | model | StrOutputParser()

        # Prepare hypothesis summary
        hypothesis_summary = "\n".join(f"- {h}" for h in state.hypotheses[:3])

        # Prepare results summary
        results_summary = "\n".join(f"- {f}" for f in state.key_findings[:10])

        synthesis = await chain.ainvoke(
            {
                "question": state.research_question,
                "hypotheses": hypothesis_summary,
                "tools": ", ".join(state.selected_tools[:10]),
                "results": results_summary,
            }
        )

        state.final_report = synthesis

        # Add citations to report
        if state.citations:
            state.final_report += "\n\n## References\n"
            for i, citation in enumerate(state.citations[:10], 1):
                state.final_report += f"{i}. {citation}\n"

        # Generate recommendations
        state.recommendations = self._generate_recommendations(state)

        print(f"  ✅ Validation complete")
        print(f"     Overall confidence: {state.confidence_scores['overall']:.1%}")
        print(f"     Recommendations: {len(state.recommendations)}")
        print(f"     Citations included: {len(state.citations)}")

        return state

    def _generate_recommendations(self, state: ImmuneAgentState) -> List[str]:
        """Generate evidence-based recommendations."""
        recommendations = []

        # Tool-specific recommendations
        tool_recommendations = {
            "metabcr": "Validate antibody predictions with SPR or BLI assays",
            "alphafold3": "Confirm structures with crystallography or cryo-EM",
            "scanpy": "Validate cell populations with flow cytometry",
            "mixcr": "Confirm clonotypes with targeted sequencing",
            "netmhcpan": "Test predicted epitopes with T cell assays",
        }

        for tool in state.selected_tools:
            if tool in tool_recommendations:
                recommendations.append(tool_recommendations[tool])

        # General recommendations
        if state.confidence_scores.get("overall", 0) > 0.7:
            recommendations.append(
                "High confidence - proceed to experimental validation"
            )
        else:
            recommendations.append(
                "Moderate confidence - consider additional computational validation"
            )

        recommendations.append("Use biological replicates (n≥3) for statistical power")
        recommendations.append("Include appropriate controls in validation experiments")

        return recommendations[:6]  # Limit to 6 recommendations


def create_planning_graph():
    """Create the ImmuneAgent planning graph."""

    # Initialize graph
    graph = StateGraph(ImmuneAgentState)

    # Initialize nodes
    planning_node = PlanningNode()
    execution_node = ExecutionNode()
    validation_node = ValidationNode()

    # Add nodes to graph
    graph.add_node("planning", planning_node)
    graph.add_node("execution", execution_node)
    graph.add_node("validation", validation_node)

    # Define workflow
    graph.add_edge(START, "planning")
    graph.add_edge("planning", "execution")
    graph.add_edge("execution", "validation")
    graph.add_edge("validation", END)

    # Compile with memory
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


async def run_immune_agent(
    question: str, category: str = "general_immunology"
) -> Dict[str, Any]:
    """
    Run the complete ImmuneAgent pipeline.

    Args:
        question: Research question
        category: Analysis category

    Returns:
        Complete results dictionary
    """
    # Create initial state
    state = ImmuneAgentState(
        research_question=question,
        original_question=question,
        domain="immunology",
        analysis_type=category,
    )

    # Create and run graph
    graph = create_planning_graph()

    config = get_immunology_runnable_config(uuid.uuid4())

    # Run the graph
    final_state = await graph.ainvoke(state, config)

    # Return results
    # 容错处理：检查 final_state 的类型
    if isinstance(final_state, dict):
        # LangGraph 返回字典格式的状态
        return {
            "success": True,
            "question": question,
            "category": category,
            "hypotheses": final_state.get("hypotheses", []),
            "selected_tools": final_state.get("selected_tools", []),
            "execution_results": final_state.get("execution_results", {}),
            "key_findings": final_state.get("key_findings", []),
            "confidence_scores": final_state.get("confidence_scores", {}),
            "recommendations": final_state.get("recommendations", []),
            "citations": final_state.get("citations", []),
            "final_report": final_state.get("final_report", ""),
        }
    else:
        # Pydantic 模型对象格式的状态
        return {
            "success": True,
            "question": question,
            "category": category,
            "hypotheses": getattr(final_state, "hypotheses", []),
            "selected_tools": getattr(final_state, "selected_tools", []),
            "execution_results": getattr(final_state, "execution_results", {}),
            "key_findings": getattr(final_state, "key_findings", []),
            "confidence_scores": getattr(final_state, "confidence_scores", {}),
            "recommendations": getattr(final_state, "recommendations", []),
            "citations": getattr(final_state, "citations", []),
            "final_report": getattr(final_state, "final_report", ""),
        }


# Export graph components
__all__ = [
    "PlanningNode",
    "ExecutionNode",
    "ValidationNode",
    "create_planning_graph",
    "run_immune_agent",
]


async def test_planning_graph():
    """
    测试planning graph工作流的基本功能。
    """
    import traceback

    print("开始测试 ImmuneAgent Planning Graph...")

    # 简单测试用例
    question = "What are the key mechanisms of antibody-antigen binding specificity?"
    category = "antibody_analysis"

    try:
        print(f"测试问题: {question}")

        # 运行工作流
        result = await run_immune_agent(question, category)

        # 输出基本结果
        print(f"\n测试结果:")
        print(f"- 成功: {result.get('success', False)}")
        print(f"- 假设数量: {len(result.get('hypotheses', []))}")
        print(f"- 工具数量: {len(result.get('selected_tools', []))}")
        print(f"- 关键发现: {len(result.get('key_findings', []))}")

        if result.get("selected_tools"):
            print(f"- 选择的工具: {', '.join(result['selected_tools'][:3])}")

        print("\n测试完成!")

    except Exception as e:
        print(f"\n测试失败: {type(e).__name__}: {str(e)}")
        print("\n详细错误信息:")
        print(traceback.format_exc())


def main():
    """
    主测试函数。
    """
    print("启动测试程序...")

    try:
        asyncio.run(test_planning_graph())
    except Exception as e:
        print(f"程序错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
