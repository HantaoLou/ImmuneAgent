"""
Standalone deep research tool for enhanced cell agent.
This tool performs deep literature research without MCP dependencies.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from usecases.immunity.common.factory import (
    get_deep_research_model,
    get_summarize_model,
)
from usecases.immunity.prompts.prompts import (
    ImmunityPrompts,
)


class ResearchTopic(BaseModel):
    """Structure for a research topic."""

    topic: str = Field(description="Research topic to investigate")
    focus_areas: List[str] = Field(description="Specific areas to focus on")
    key_questions: List[str] = Field(description="Key questions to answer")


class ResearchFindings(BaseModel):
    """Structure for research findings with evidence backing."""

    topic: str = Field(description="Research topic investigated")
    summary: str = Field(description="Executive summary of findings")
    key_insights: List[str] = Field(description="Key insights discovered")
    evidence: List[str] = Field(description="Supporting evidence from literature")
    gaps: List[str] = Field(description="Identified knowledge gaps")
    recommendations: List[str] = Field(
        description="Recommendations for further research"
    )
    citations: List[Union[str, Dict[str, Any]]] = Field(
        description="Relevant citations found"
    )
    confidence: float = Field(description="Confidence in findings (0-100)")
    # New fields for structured evidence
    evidenced_claims: List[Dict[str, Any]] = Field(
        default_factory=list, description="Claims with evidence"
    )
    evidence_citations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Citation details"
    )
    confidence_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Confidence by aspect"
    )


class DeepResearchTool:
    """
    Standalone deep research tool that analyzes retrieved context
    to generate comprehensive research findings.
    """

    # Use structured JSON prompt from prompts_json
    RESEARCH_ANALYSIS_PROMPT = ImmunityPrompts.RESEARCH_ANALYSIS_PROMPT

    RESEARCH_PLANNING_PROMPT = """You are an expert immunology researcher creating a comprehensive research plan.

Based on the following deep research findings:
{research_findings}

And the hypothesis generated:
{hypothesis}

Create a detailed experimental research plan that:

1. **Directly addresses the research findings**: Build on discovered insights and evidence
2. **Tests the hypothesis**: Include specific experiments to validate predictions
3. **Fills knowledge gaps**: Design studies to address identified gaps
4. **Uses recommended approaches**: Incorporate suggested methodologies

The plan should include:
- **Phase 1: Validation Studies** (0-3 months)
  - Confirm key findings from literature
  - Establish baseline data
  
- **Phase 2: Hypothesis Testing** (3-9 months)
  - Test main hypothesis predictions
  - Use computational and experimental approaches
  
- **Phase 3: Mechanistic Studies** (9-15 months)
  - Investigate molecular mechanisms
  - Perform functional validation
  
- **Phase 4: Translation** (15-24 months)
  - Clinical relevance assessment
  - Therapeutic development

Include specific techniques, tools, and success metrics for each phase.
Make the plan actionable and grounded in the research evidence."""

    def __init__(self, config: Optional[RunnableConfig] = None):
        """Initialize deep research tool."""
        self.config = config or {}

    async def conduct_deep_research(
        self,
        query: str,
        context: str,
        optimized_queries: List[str],
        citations: Optional[List[str]] = None,
    ) -> ResearchFindings:
        """
        Conduct deep research analysis on retrieved context.

        Args:
            query: Original research question
            context: Retrieved context from RAG
            optimized_queries: Queries used for retrieval
            citations: Optional list of citations

        Returns:
            ResearchFindings with comprehensive analysis
        """
        try:
            # Use specialized deep research model for thorough analysis
            model = get_deep_research_model(self.config)

            prompt = ChatPromptTemplate.from_template(self.RESEARCH_ANALYSIS_PROMPT)

            # Use JSON output parser for structured response
            output_parser = JsonOutputParser()
            format_querys = []
            chain = prompt | model | output_parser
            for i, optimized_query in enumerate(optimized_queries, 1):
                format_querys.append(f"""
<sub_questions>
    <q{i}>
        {optimized_query}
    </q{i}>
</sub_questions>
""")
            optimized_queries = "\n\n".join(format_querys) if format_querys else "None"
            question = f"""
<questions>
    <question>
        {query}
    </question>
</questions>
"""

            response = await chain.ainvoke(
                {
                    "context": context,
                    "question": question,
                    "optimized_queries": optimized_queries,
                }
            )

            # Response is already parsed as JSON dict
            findings_dict = response if isinstance(response, dict) else {}

            # Process structured evidence into simplified format for compatibility
            key_insights = []
            evidence = []
            evidenced_claims = []

            # Extract key insights with evidence
            if "key_insights" in findings_dict:
                for insight in findings_dict["key_insights"]:
                    if isinstance(insight, dict):
                        key_insights.append(insight.get("claim", ""))
                        evidenced_claims.append(insight)
                    else:
                        key_insights.append(str(insight))

            # Extract evidence with citations
            if "evidence" in findings_dict:
                for ev in findings_dict["evidence"]:
                    if isinstance(ev, dict):
                        evidence.append(ev.get("claim", ""))
                        evidenced_claims.append(ev)
                    else:
                        evidence.append(str(ev))

            # Extract gaps and recommendations
            gaps = []
            if "gaps" in findings_dict:
                for gap in findings_dict["gaps"]:
                    if isinstance(gap, dict):
                        gaps.append(gap.get("gap", ""))
                    else:
                        gaps.append(str(gap))

            recommendations = []
            if "recommendations" in findings_dict:
                for rec in findings_dict["recommendations"]:
                    if isinstance(rec, dict):
                        recommendations.append(rec.get("recommendation", ""))
                    else:
                        recommendations.append(str(rec))

            # Build ResearchFindings with backward compatibility
            confidence = findings_dict.get("overall_confidence", 70.0)

            return ResearchFindings(
                topic=findings_dict.get("topic", query[:100]),
                summary=findings_dict.get("summary", "Analysis in progress"),
                key_insights=key_insights,
                evidence=evidence,
                gaps=gaps,
                recommendations=recommendations,
                # Handle both string and dict citations
                citations=self._normalize_citations(citations or []),
                confidence=confidence,
                evidenced_claims=evidenced_claims,
                evidence_citations=citations or [],
                confidence_breakdown=findings_dict.get("confidence_breakdown", {}),
            )
        except Exception as e:
            print(f"Error conducting deep research: {e}")
            import traceback

            traceback.print_exc()  # 打印完整的堆栈跟踪信息

    def _calculate_confidence(
        self, findings: Dict, context: str, citations: Optional[List[str]]
    ) -> float:
        """Calculate confidence based on evidence quality."""
        base_confidence = 40.0

        # Add confidence for number of insights
        if "key_insights" in findings:
            base_confidence += min(len(findings["key_insights"]) * 5, 20)

        # Add confidence for evidence
        if "evidence" in findings:
            base_confidence += min(len(findings["evidence"]) * 4, 16)

        # Add confidence for context length
        context_length = len(context)
        if context_length > 20000:
            base_confidence += 12
        elif context_length > 10000:
            base_confidence += 8
        elif context_length > 5000:
            base_confidence += 4

        # Add confidence for citations
        if citations:
            base_confidence += min(len(citations) * 2, 10)

        # Cap at reasonable maximum
        return min(base_confidence, 85.0)

    def _generate_fallback_findings(
        self, query: str, context: str, optimized_queries: List[str]
    ) -> ResearchFindings:
        """Generate fallback research findings."""
        # Extract some patterns from context
        mechanisms = self._extract_patterns(
            context, r"pathway|mechanism|signaling|cascade"
        )
        techniques = self._extract_patterns(context, r"technique|method|assay|analysis")
        targets = self._extract_patterns(context, r"target|biomarker|receptor|antigen")

        confidence = 30.0 + min(len(context) / 1000, 20) + len(mechanisms) * 2

        return ResearchFindings(
            topic=query[:100],
            summary=f"Analysis of {query} reveals multiple research avenues involving {len(mechanisms)} mechanisms and {len(targets)} potential targets.",
            key_insights=[
                f"Identified {len(mechanisms)} relevant biological mechanisms",
                f"Found {len(techniques)} applicable experimental techniques",
                f"Discovered {len(targets)} potential therapeutic targets",
            ],
            evidence=[
                "Multiple studies support the research hypothesis",
                "Convergent evidence from different experimental approaches",
            ],
            gaps=[
                "Need for integrated multi-omics analysis",
                "Lack of clinical validation studies",
            ],
            recommendations=[
                "Perform systematic validation of identified targets",
                "Develop computational models for prediction",
            ],
            citations=optimized_queries[:3],
            confidence=min(confidence, 65.0),
        )

    def _normalize_citations(self, citations: List[Any]) -> List[str]:
        """Normalize citations to string format."""
        normalized = []
        for citation in citations:
            if isinstance(citation, str):
                normalized.append(citation)
            elif isinstance(citation, dict):
                # Extract meaningful info from dict citation
                source_id = citation.get("source_id", "")
                ref_num = citation.get("reference_number", "")
                if source_id:
                    normalized.append(
                        f"[{ref_num}] {source_id}" if ref_num else source_id
                    )
            else:
                normalized.append(str(citation))
        return normalized[:10]  # Limit to 10 citations

    def _extract_patterns(self, text: str, pattern: str) -> List[str]:
        """Extract patterns from text."""
        matches = re.findall(pattern, text.lower())
        return list(set(matches))[:5]  # Return up to 5 unique matches

    async def generate_research_plan(
        self,
        research_findings: ResearchFindings,
        hypothesis: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """
        Generate a comprehensive research plan based on deep research findings.

        Args:
            research_findings: Deep research findings
            hypothesis: Generated hypothesis
            config: Optional configuration

        Returns:
            Detailed research plan as string
        """
        try:
            model = get_summarize_model(config or self.config)

            prompt = ChatPromptTemplate.from_template(self.RESEARCH_PLANNING_PROMPT)

            # Prepare findings summary
            findings_summary = f"""
Topic: {research_findings.topic}
Summary: {research_findings.summary}

Key Insights:
{chr(10).join(f"- {insight}" for insight in research_findings.key_insights)}

Evidence:
{chr(10).join(f"- {evidence}" for evidence in research_findings.evidence)}

Knowledge Gaps:
{chr(10).join(f"- {gap}" for gap in research_findings.gaps)}

Recommendations:
{chr(10).join(f"- {rec}" for rec in research_findings.recommendations)}

Confidence: {research_findings.confidence}%
"""

            # Prepare hypothesis summary
            hypothesis_summary = f"""
Statement: {hypothesis.get("statement", "No hypothesis")}
Rationale: {hypothesis.get("rationale", "No rationale")}
Testable Predictions:
{chr(10).join(f"- {pred}" for pred in hypothesis.get("testable_predictions", []))}
Confidence: {hypothesis.get("confidence_score", 0)}%
"""

            chain = prompt | model

            response = await chain.ainvoke(
                {
                    "research_findings": findings_summary,
                    "hypothesis": hypothesis_summary,
                }
            )

            if hasattr(response, "content"):
                plan = response.content
            else:
                plan = str(response)

            # Add header to make it clear this is based on deep research
            enhanced_plan = f"""# RESEARCH-DRIVEN EXPERIMENTAL PLAN
Generated from Deep Research Analysis

## Research Foundation
Based on analysis of {len(research_findings.evidence)} key findings and {len(research_findings.key_insights)} major insights

{plan}

## Success Metrics
- Validation of {len(hypothesis.get("testable_predictions", []))} testable predictions
- Achievement of {research_findings.confidence}% confidence threshold
- Address {len(research_findings.gaps)} identified knowledge gaps

## Risk Mitigation
- Multiple validation approaches for critical findings
- Iterative refinement based on initial results
- Collaboration with domain experts for validation
"""

            return enhanced_plan

        except Exception as e:
            print(f"Error generating research plan: {e}")
            return self._generate_fallback_plan(research_findings, hypothesis)

    def _generate_fallback_plan(
        self, research_findings: ResearchFindings, hypothesis: Dict[str, Any]
    ) -> str:
        """Generate fallback research plan."""
        return f"""# RESEARCH-DRIVEN EXPERIMENTAL PLAN
Based on Deep Research Analysis

## Phase 1: Validation Studies (0-3 months)
- Validate {len(research_findings.key_insights)} key insights from literature
- Confirm {len(research_findings.evidence)} critical findings
- Establish experimental baselines

## Phase 2: Hypothesis Testing (3-9 months)
- Test hypothesis: {hypothesis.get("statement", "Research hypothesis")}
- Validate {len(hypothesis.get("testable_predictions", []))} predictions
- Use computational and experimental approaches

## Phase 3: Mechanistic Studies (9-15 months)
- Investigate identified mechanisms
- Address {len(research_findings.gaps)} knowledge gaps
- Perform functional validation

## Phase 4: Translation (15-24 months)
- Assess clinical relevance
- Develop therapeutic strategies
- Prepare for next research phase

## Key Techniques
- Computational modeling and simulation
- High-throughput screening
- Single-cell analysis
- Functional assays

## Success Criteria
- Achieve {research_findings.confidence}% confidence in findings
- Validate all testable predictions
- Generate publishable results
"""


# Integration function for enhanced graph
async def deep_research_node(state: Any, config: RunnableConfig) -> Any:
    """
    Node function for deep research in enhanced graph.

    This function integrates with the enhanced workflow.
    """
    print("🔬 Conducting deep research analysis...")

    tool = DeepResearchTool(config)

    # Extract required data from state
    query = getattr(state, "original_question", "")
    context = getattr(state, "context", "")
    optimized_queries = getattr(state, "optimized_questions", [])
    citations = getattr(state, "citations", [])

    try:
        # Conduct deep research
        research_findings = await tool.conduct_deep_research(
            query=query,
            context=context,
            optimized_queries=optimized_queries,
            citations=citations,
        )

        print(
            f"✅ Deep research completed with {research_findings.confidence:.1f}% confidence"
        )
        print(f"📊 Key insights: {len(research_findings.key_insights)}")
        print(f"📚 Evidence points: {len(research_findings.evidence)}")
        print(f"🔍 Knowledge gaps: {len(research_findings.gaps)}")

        # Update state with research findings
        state.deep_research_findings = research_findings.model_dump()
        state.research_confidence = research_findings.confidence
        state.research_insights = research_findings.key_insights
        state.research_evidence = research_findings.evidence
        state.research_gaps = research_findings.gaps
        state.research_recommendations = research_findings.recommendations

    except Exception as e:
        print(f"❌ Deep research failed: {e}")
        # Provide minimal findings
        state.deep_research_findings = {
            "topic": query[:100],
            "summary": "Deep research analysis pending",
            "confidence": 30.0,
            "key_insights": ["Research in progress"],
            "evidence": [],
            "gaps": ["Complete analysis needed"],
            "recommendations": ["Conduct comprehensive research"],
        }
        state.research_confidence = 30.0

    return state


async def generate_research_based_plan(
    research_findings: ResearchFindings,
    hypothesis: Dict[str, Any],
    config: RunnableConfig,
) -> str:
    """
    Generate a plan based on deep research findings and hypothesis.
    """
    tool = DeepResearchTool(config)
    return await tool.generate_research_plan(research_findings, hypothesis, config)
