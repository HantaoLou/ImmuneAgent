"""
Multi-Agent Framework for HLE

Research shows that multi-agent approaches significantly outperform
single models on HLE (Grok-4 multi-agent: 44% vs single model: <20%).

This module provides:
- Specialized agents for different reasoning tasks
- Debate mechanism for disagreement resolution
- Voting and consensus mechanisms
- Parallel and sequential agent coordination

Key Features:
- AgentRole: Different specialized roles
- DebateMechanism: Structured debate for consensus
- MultiAgentFramework: Orchestration of multiple agents
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
from collections import Counter
import uuid


class AgentRole(Enum):
    """Roles for different agents in the framework"""
    ANALYZER = "analyzer"           # Analyzes and decomposes questions
    DOMAIN_EXPERT = "domain_expert" # Provides domain-specific knowledge
    CALCULATOR = "calculator"       # Performs calculations
    VALIDATOR = "validator"         # Validates reasoning and answers
    SYNTHESIZER = "synthesizer"     # Synthesizes final answer
    SKEPTIC = "skeptic"             # Challenges assumptions
    FACT_CHECKER = "fact_checker"   # Verifies factual claims


@dataclass
class AgentResponse:
    """Response from a single agent"""
    agent_id: str
    agent_role: AgentRole
    answer: Optional[str]
    reasoning: str
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    issues_found: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateRound:
    """A single round in the debate process"""
    round_number: int
    agent_positions: Dict[str, str]  # agent_id -> position
    challenges: Dict[str, List[str]]  # agent_id -> list of challenges
    responses: Dict[str, str]  # agent_id -> response to challenges
    consensus_reached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateResult:
    """Result of a debate process"""
    rounds: List[DebateRound]
    final_positions: Dict[str, str]
    consensus_answer: Optional[str]
    confidence: float
    agreement_level: float  # 0.0 to 1.0
    dissenting_views: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiAgentResult:
    """Final result from multi-agent reasoning"""
    answer: Optional[str]
    confidence: float
    agent_responses: List[AgentResponse]
    debate_result: Optional[DebateResult]
    voting_result: Dict[str, int]
    consensus_method: str  # "unanimous", "majority", "debate", "weighted"
    reasoning_trace: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for agents"""
    
    def __init__(
        self,
        agent_id: str = None,
        role: AgentRole = AgentRole.ANALYZER,
        llm: Any = None
    ):
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.role = role
        self.llm = llm
    
    @abstractmethod
    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process the question and return a response"""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass


class AnalyzerAgent(BaseAgent):
    """Agent specialized in question analysis and decomposition"""
    
    def __init__(self, llm: Any = None):
        super().__init__(role=AgentRole.ANALYZER, llm=llm)
    
    def get_system_prompt(self) -> str:
        return """You are a question analyzer. Your role is to:
1. Identify the core question being asked
2. Break down complex questions into sub-questions
3. Identify the domain and relevant concepts
4. Determine what type of reasoning is required
5. Identify any constraints or special requirements

Provide a structured analysis of the question."""
    
    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        context = context or {}
        
        # Analysis logic (simplified for framework)
        analysis = {
            "core_question": question,
            "sub_questions": [],
            "domain": context.get("domain", "general"),
            "reasoning_type": "deductive",
            "constraints": []
        }
        
        return AgentResponse(
            agent_id=self.agent_id,
            agent_role=self.role,
            answer=None,  # Analyzer doesn't provide final answer
            reasoning=f"Analyzed question: {question}",
            confidence=0.9,
            metadata={"analysis": analysis}
        )


class DomainExpertAgent(BaseAgent):
    """Agent specialized in domain-specific knowledge and reasoning"""
    
    DOMAIN_PROMPTS = {
        "genetics": """You are a genetics expert. Apply your knowledge of:
- Mendelian and non-Mendelian inheritance
- Molecular genetics (DNA, RNA, proteins)
- Population genetics (Hardy-Weinberg, genetic drift, selection)
- Genomics and gene regulation

Reason carefully from established genetic principles.""",
        
        "molecular_biology": """You are a molecular biology expert. Apply your knowledge of:
- Central dogma (DNA → RNA → Protein)
- Transcription and translation regulation
- Protein structure and function
- Cellular signaling pathways
- Gene expression control mechanisms

Use mechanistic reasoning based on molecular principles.""",
        
        "biochemistry": """You are a biochemistry expert. Apply your knowledge of:
- Enzyme kinetics and catalysis
- Metabolic pathways
- Thermodynamics of biochemical reactions
- Protein-ligand interactions
- Structural biochemistry

Apply quantitative and mechanistic reasoning.""",
        
        "immunology": """You are an immunology expert. Apply your knowledge of:
- Innate and adaptive immunity
- Antibody structure and function
- T cell and B cell biology
- Immunological memory
- Autoimmune and inflammatory processes

Reason from immunological principles."""
    }
    
    def __init__(self, domain: str = "general", llm: Any = None):
        super().__init__(role=AgentRole.DOMAIN_EXPERT, llm=llm)
        self.domain = domain
    
    def get_system_prompt(self) -> str:
        return self.DOMAIN_PROMPTS.get(self.domain, 
            "You are a biology expert. Apply your domain knowledge carefully.")
    
    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        context = context or {}
        
        # Domain-specific reasoning (simplified)
        return AgentResponse(
            agent_id=self.agent_id,
            agent_role=self.role,
            answer="[Domain expert reasoning result]",
            reasoning=f"Applied {self.domain} expertise to analyze: {question[:100]}...",
            confidence=0.7,
            supporting_evidence=["Domain knowledge applied"],
            metadata={"domain": self.domain}
        )


class CalculatorAgent(BaseAgent):
    """Agent specialized in numerical calculations"""
    
    def __init__(self, llm: Any = None):
        super().__init__(role=AgentRole.CALCULATOR, llm=llm)
    
    def get_system_prompt(self) -> str:
        return """You are a calculation specialist. Your role is to:
1. Identify all numerical values and their units
2. Determine the correct formula or algorithm to apply
3. Perform calculations step by step
4. Verify numerical results
5. Report results with appropriate precision and units

Always show your work and verify your calculations."""
    
    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        context = context or {}
        
        # Check if calculation is needed
        has_numbers = any(c.isdigit() for c in question)
        
        if not has_numbers:
            return AgentResponse(
                agent_id=self.agent_id,
                agent_role=self.role,
                answer=None,
                reasoning="No calculation required for this question",
                confidence=1.0,
                metadata={"calculation_needed": False}
            )
        
        return AgentResponse(
            agent_id=self.agent_id,
            agent_role=self.role,
            answer="[Calculated result]",
            reasoning="Performed calculation based on given values",
            confidence=0.85,
            metadata={"calculation_needed": True, "method": "arithmetic"}
        )


class ValidatorAgent(BaseAgent):
    """Agent specialized in validation and fact-checking"""
    
    def __init__(self, llm: Any = None):
        super().__init__(role=AgentRole.VALIDATOR, llm=llm)
    
    def get_system_prompt(self) -> str:
        return """You are a validator. Your role is to:
1. Check logical consistency of reasoning
2. Verify factual claims against known facts
3. Identify potential errors or biases
4. Assess completeness of reasoning
5. Flag any assumptions that need verification

Be thorough and critical in your analysis."""
    
    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        context = context or {}
        
        # Validation logic
        issues = []
        if context.get("proposed_answer"):
            # Check for common issues
            issues.append("Verifying logical consistency")
        
        return AgentResponse(
            agent_id=self.agent_id,
            agent_role=self.role,
            answer=context.get("proposed_answer"),
            reasoning="Validated the proposed answer for logical consistency",
            confidence=0.8,
            issues_found=issues,
            metadata={"validation_status": "passed" if not issues else "issues_found"}
        )


class SkepticAgent(BaseAgent):
    """Agent that challenges assumptions and finds weaknesses"""
    
    def __init__(self, llm: Any = None):
        super().__init__(role=AgentRole.SKEPTIC, llm=llm)
    
    def get_system_prompt(self) -> str:
        return """You are a professional skeptic. Your role is to:
1. Challenge assumptions made in reasoning
2. Consider alternative interpretations
3. Identify edge cases that might break the logic
4. Question whether the evidence is sufficient
5. Propose counterarguments

Your goal is to strengthen the final answer by finding weaknesses."""
    
    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        context = context or {}
        
        challenges = []
        if context.get("proposed_answer"):
            challenges.append("Is there an alternative explanation?")
            challenges.append("Are all assumptions justified?")
        
        return AgentResponse(
            agent_id=self.agent_id,
            agent_role=self.role,
            answer=None,  # Skeptic doesn't provide own answer
            reasoning="Challenged the proposed answer with critical questions",
            confidence=0.6,
            issues_found=challenges,
            metadata={"challenges_raised": len(challenges)}
        )


class DebateMechanism:
    """
    Structured debate mechanism for resolving disagreements between agents.
    
    Process:
    1. Each agent states their position
    2. Agents challenge each other's positions
    3. Agents respond to challenges
    4. Process repeats until consensus or max rounds
    """
    
    def __init__(
        self,
        max_rounds: int = 3,
        consensus_threshold: float = 0.8,
        llm: Any = None
    ):
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.llm = llm
    
    async def conduct_debate(
        self,
        question: str,
        initial_positions: Dict[str, AgentResponse]
    ) -> DebateResult:
        """
        Conduct a debate between agents with different positions.
        
        Args:
            question: The original question
            initial_positions: Dict of agent_id -> AgentResponse
            
        Returns:
            DebateResult with consensus or final positions
        """
        rounds = []
        current_positions = {
            aid: resp.answer 
            for aid, resp in initial_positions.items() 
            if resp.answer
        }
        
        for round_num in range(self.max_rounds):
            # Create debate round
            debate_round = DebateRound(
                round_number=round_num + 1,
                agent_positions=current_positions.copy(),
                challenges={},
                responses={},
                consensus_reached=False
            )
            
            # Generate challenges
            challenges = await self._generate_challenges(
                question, current_positions
            )
            debate_round.challenges = challenges
            
            # Generate responses
            responses = await self._generate_responses(
                question, current_positions, challenges
            )
            debate_round.responses = responses
            
            # Update positions based on responses
            current_positions = await self._update_positions(
                current_positions, responses
            )
            
            # Check for consensus
            agreement = self._calculate_agreement(current_positions)
            if agreement >= self.consensus_threshold:
                debate_round.consensus_reached = True
                rounds.append(debate_round)
                break
            
            rounds.append(debate_round)
        
        # Calculate final results
        final_agreement = self._calculate_agreement(current_positions)
        consensus_answer = self._determine_consensus(current_positions)
        
        # Identify dissenting views
        dissenting = [
            pos for pos in current_positions.values() 
            if pos != consensus_answer
        ]
        
        return DebateResult(
            rounds=rounds,
            final_positions=current_positions,
            consensus_answer=consensus_answer,
            confidence=final_agreement,
            agreement_level=final_agreement,
            dissenting_views=dissenting,
            metadata={
                "total_rounds": len(rounds),
                "consensus_reached": rounds[-1].consensus_reached if rounds else False
            }
        )
    
    async def _generate_challenges(
        self,
        question: str,
        positions: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Generate challenges for each position"""
        challenges = {}
        
        unique_positions = set(positions.values())
        
        for agent_id, position in positions.items():
            agent_challenges = []
            
            # Challenge based on other agents' positions
            for other_id, other_pos in positions.items():
                if other_id != agent_id and other_pos != position:
                    agent_challenges.append(
                        f"Agent {other_id} disagrees: they suggest '{other_pos}'"
                    )
            
            # Add generic challenges
            if position:
                agent_challenges.append(
                    "Can you provide more evidence for this position?"
                )
            
            challenges[agent_id] = agent_challenges
        
        return challenges
    
    async def _generate_responses(
        self,
        question: str,
        positions: Dict[str, str],
        challenges: Dict[str, List[str]]
    ) -> Dict[str, str]:
        """Generate responses to challenges"""
        responses = {}
        
        for agent_id, agent_challenges in challenges.items():
            if agent_challenges:
                responses[agent_id] = (
                    f"Maintaining position based on available evidence. "
                    f"Addressing {len(agent_challenges)} challenges."
                )
            else:
                responses[agent_id] = "No challenges to address."
        
        return responses
    
    async def _update_positions(
        self,
        positions: Dict[str, str],
        responses: Dict[str, str]
    ) -> Dict[str, str]:
        """Update positions based on debate (simplified)"""
        # In practice, this would involve actual position changes
        return positions.copy()
    
    def _calculate_agreement(self, positions: Dict[str, str]) -> float:
        """Calculate level of agreement among agents"""
        if not positions:
            return 0.0
        
        values = list(positions.values())
        if not values:
            return 0.0
        
        # Normalize for comparison
        normalized = [str(v).lower().strip() for v in values if v]
        if not normalized:
            return 0.0
        
        # Count most common
        counter = Counter(normalized)
        most_common_count = counter.most_common(1)[0][1]
        
        return most_common_count / len(normalized)
    
    def _determine_consensus(self, positions: Dict[str, str]) -> Optional[str]:
        """Determine the consensus answer"""
        if not positions:
            return None
        
        values = [str(v).lower().strip() for v in positions.values() if v]
        if not values:
            return None
        
        counter = Counter(values)
        return counter.most_common(1)[0][0]


class MultiAgentFramework:
    """
    Framework for coordinating multiple agents to solve HLE questions.
    
    Supports:
    - Sequential agent execution (pipeline)
    - Parallel agent execution
    - Debate-based consensus
    - Weighted voting
    """
    
    def __init__(
        self,
        llm: Any = None,
        use_debate: bool = True,
        use_voting: bool = True,
        parallel_execution: bool = True
    ):
        self.llm = llm
        self.use_debate = use_debate
        self.use_voting = use_voting
        self.parallel_execution = parallel_execution
        
        # Initialize agents
        self.agents: Dict[AgentRole, BaseAgent] = {
            AgentRole.ANALYZER: AnalyzerAgent(llm=llm),
            AgentRole.DOMAIN_EXPERT: DomainExpertAgent(llm=llm),
            AgentRole.CALCULATOR: CalculatorAgent(llm=llm),
            AgentRole.VALIDATOR: ValidatorAgent(llm=llm),
            AgentRole.SKEPTIC: SkepticAgent(llm=llm),
        }
        
        # Debate mechanism
        self.debate = DebateMechanism(llm=llm) if use_debate else None
    
    async def solve(
        self,
        question: str,
        domain: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> MultiAgentResult:
        """
        Solve a question using multi-agent reasoning.
        
        Args:
            question: The question to solve
            domain: Domain hint for expert selection
            context: Additional context
            
        Returns:
            MultiAgentResult with answer and reasoning trace
        """
        context = context or {}
        context["domain"] = domain
        
        reasoning_trace = []
        agent_responses = []
        
        # Phase 1: Analysis
        analyzer = self.agents[AgentRole.ANALYZER]
        analysis = await analyzer.process(question, context)
        agent_responses.append(analysis)
        reasoning_trace.append({
            "phase": "analysis",
            "agent": "analyzer",
            "result": analysis.reasoning
        })
        
        # Update context with analysis
        context["analysis"] = analysis.metadata.get("analysis", {})
        
        # Phase 2: Parallel expert reasoning
        expert_tasks = []
        
        # Domain expert
        domain_expert = self.agents[AgentRole.DOMAIN_EXPERT]
        if isinstance(domain_expert, DomainExpertAgent):
            domain_expert.domain = domain or "general"
        expert_tasks.append(domain_expert.process(question, context))
        
        # Calculator (if needed)
        calculator = self.agents[AgentRole.CALCULATOR]
        expert_tasks.append(calculator.process(question, context))
        
        # Execute in parallel or sequentially
        if self.parallel_execution:
            expert_results = await asyncio.gather(*expert_tasks)
        else:
            expert_results = []
            for task in expert_tasks:
                expert_results.append(await task)
        
        agent_responses.extend(expert_results)
        reasoning_trace.append({
            "phase": "expert_reasoning",
            "results": [r.reasoning for r in expert_results]
        })
        
        # Collect initial answers
        initial_answers = {
            r.agent_id: r.answer 
            for r in expert_results 
            if r.answer
        }
        
        # Phase 3: Validation and skepticism
        proposed_answer = list(initial_answers.values())[0] if initial_answers else None
        context["proposed_answer"] = proposed_answer
        
        validator = self.agents[AgentRole.VALIDATOR]
        skeptic = self.agents[AgentRole.SKEPTIC]
        
        validation = await validator.process(question, context)
        skepticism = await skeptic.process(question, context)
        
        agent_responses.extend([validation, skepticism])
        reasoning_trace.append({
            "phase": "validation",
            "validation_issues": validation.issues_found,
            "skeptic_challenges": skepticism.issues_found
        })
        
        # Phase 4: Debate (if enabled and disagreement)
        debate_result = None
        if self.use_debate and len(set(initial_answers.values())) > 1:
            debate_result = await self.debate.conduct_debate(
                question,
                {r.agent_id: r for r in expert_results if r.answer}
            )
            reasoning_trace.append({
                "phase": "debate",
                "rounds": len(debate_result.rounds),
                "consensus_reached": debate_result.metadata.get("consensus_reached", False)
            })
        
        # Phase 5: Final answer determination
        final_answer, confidence, method = self._determine_final_answer(
            agent_responses, debate_result, initial_answers
        )
        
        # Phase 6: Voting result
        voting_result = self._conduct_voting(agent_responses)
        
        return MultiAgentResult(
            answer=final_answer,
            confidence=confidence,
            agent_responses=agent_responses,
            debate_result=debate_result,
            voting_result=voting_result,
            consensus_method=method,
            reasoning_trace=reasoning_trace,
            metadata={
                "total_agents": len(agent_responses),
                "domain": domain,
                "parallel_execution": self.parallel_execution
            }
        )
    
    def _determine_final_answer(
        self,
        responses: List[AgentResponse],
        debate_result: Optional[DebateResult],
        initial_answers: Dict[str, str]
    ) -> Tuple[Optional[str], float, str]:
        """Determine the final answer from all agent responses"""
        
        # If debate reached consensus, use that
        if debate_result and debate_result.consensus_answer:
            return (
                debate_result.consensus_answer,
                debate_result.confidence,
                "debate_consensus"
            )
        
        # If unanimous agreement
        if len(set(initial_answers.values())) == 1 and initial_answers:
            answer = list(initial_answers.values())[0]
            return (answer, 0.9, "unanimous")
        
        # Weighted voting based on confidence
        weighted_answers = {}
        for response in responses:
            if response.answer:
                weight = response.confidence
                if response.agent_role == AgentRole.DOMAIN_EXPERT:
                    weight *= 1.2  # Boost expert weight
                elif response.agent_role == AgentRole.CALCULATOR:
                    weight *= 1.1  # Slight boost for calculations
                
                if response.answer not in weighted_answers:
                    weighted_answers[response.answer] = 0
                weighted_answers[response.answer] += weight
        
        if weighted_answers:
            best_answer = max(weighted_answers.items(), key=lambda x: x[1])
            total_weight = sum(weighted_answers.values())
            confidence = best_answer[1] / total_weight if total_weight > 0 else 0.5
            
            return (best_answer[0], confidence, "weighted_voting")
        
        return (None, 0.0, "no_consensus")
    
    def _conduct_voting(self, responses: List[AgentResponse]) -> Dict[str, int]:
        """Conduct simple voting among agents"""
        votes = {}
        
        for response in responses:
            if response.answer:
                answer = str(response.answer).lower().strip()
                votes[answer] = votes.get(answer, 0) + 1
        
        return votes
    
    def add_agent(self, agent: BaseAgent):
        """Add a custom agent to the framework"""
        self.agents[agent.role] = agent
    
    def remove_agent(self, role: AgentRole):
        """Remove an agent by role"""
        if role in self.agents:
            del self.agents[role]

