"""
Original Problem Handler for HLE

HLE questions are specifically designed to be original and cannot be found
through simple web search. This module provides strategies for handling
such novel problems:

1. First Principles Reasoning: Start from fundamental principles
2. Analogical Reasoning: Find similar known problems
3. Problem Decomposition: Break into known sub-problems
4. Tool-based Simulation: Use computational tools for verification

Key Features:
- Strategy selection based on problem characteristics
- Integration with domain knowledge bases
- Fallback chains when primary strategies fail
"""

import re
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class ProblemStrategy(Enum):
    """Strategies for handling original/novel problems"""
    FIRST_PRINCIPLES = "first_principles"
    ANALOGICAL = "analogical"
    DECOMPOSITION = "decomposition"
    TOOL_SIMULATION = "tool_simulation"
    ENSEMBLE = "ensemble"  # Combine multiple strategies


@dataclass
class ProblemAnalysis:
    """Analysis result for a problem"""
    problem_type: str
    key_concepts: List[str]
    constraints: List[str]
    required_knowledge: List[str]
    suggested_strategies: List[ProblemStrategy]
    difficulty_indicators: Dict[str, Any]
    similar_problems: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StrategyResult:
    """Result from applying a strategy"""
    strategy: ProblemStrategy
    success: bool
    reasoning_steps: List[Dict[str, Any]]
    intermediate_conclusions: List[str]
    final_answer: Optional[str]
    confidence: float
    issues_encountered: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OriginalProblemResult:
    """Final result for original problem handling"""
    answer: Optional[str]
    confidence: float
    primary_strategy: ProblemStrategy
    strategy_results: List[StrategyResult]
    reasoning_trace: List[Dict[str, Any]]
    verification_status: str  # "verified", "partially_verified", "unverified"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies"""
    
    @abstractmethod
    def can_apply(self, problem_analysis: ProblemAnalysis) -> bool:
        """Check if this strategy is applicable"""
        pass
    
    @abstractmethod
    def apply(
        self, 
        question: str, 
        problem_analysis: ProblemAnalysis,
        context: Optional[Dict[str, Any]] = None
    ) -> StrategyResult:
        """Apply the reasoning strategy"""
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> ProblemStrategy:
        """Return the strategy name"""
        pass


class FirstPrinciplesReasoner(ReasoningStrategy):
    """
    First Principles Reasoning Strategy
    
    Start from fundamental axioms and build up to the answer.
    Best for problems involving:
    - Fundamental laws (physics, chemistry)
    - Logical deduction
    - Mathematical proofs
    """
    
    # Fundamental principles by domain
    DOMAIN_PRINCIPLES = {
        "genetics": [
            "DNA → RNA → Protein (Central Dogma)",
            "Mendel's laws of inheritance",
            "Hardy-Weinberg equilibrium",
            "Genetic code is universal",
            "Mutations are random with respect to fitness"
        ],
        "molecular_biology": [
            "Structure determines function",
            "Enzymes catalyze specific reactions",
            "Allostery enables regulation",
            "Feedback loops maintain homeostasis",
            "Energy coupling drives unfavorable reactions"
        ],
        "biochemistry": [
            "Mass balance in reactions",
            "Thermodynamic constraints",
            "Enzyme kinetics (Michaelis-Menten)",
            "pH affects ionization states",
            "Hydrophobic effect drives protein folding"
        ],
        "cell_biology": [
            "Cells arise from pre-existing cells",
            "Membrane potential drives transport",
            "Signal transduction cascades",
            "Cell cycle checkpoints",
            "Apoptosis is programmed cell death"
        ],
        "immunology": [
            "Self vs non-self discrimination",
            "Clonal selection theory",
            "Antibody diversity through V(D)J recombination",
            "Memory cells enable faster secondary response",
            "MHC presents antigens to T cells"
        ]
    }
    
    def __init__(self):
        self.strategy = ProblemStrategy.FIRST_PRINCIPLES
    
    @property
    def strategy_name(self) -> ProblemStrategy:
        return self.strategy
    
    def can_apply(self, problem_analysis: ProblemAnalysis) -> bool:
        """First principles works best for mechanistic/quantitative problems"""
        mechanistic_indicators = [
            "mechanism", "pathway", "process", "how does",
            "calculate", "determine", "derive", "explain why"
        ]
        
        question_lower = " ".join(problem_analysis.key_concepts).lower()
        return any(ind in question_lower for ind in mechanistic_indicators)
    
    def apply(
        self,
        question: str,
        problem_analysis: ProblemAnalysis,
        context: Optional[Dict[str, Any]] = None
    ) -> StrategyResult:
        """Apply first principles reasoning"""
        context = context or {}
        reasoning_steps = []
        intermediate_conclusions = []
        issues = []
        
        # Step 1: Identify relevant domain and principles
        domain = self._identify_domain(question, problem_analysis)
        principles = self.DOMAIN_PRINCIPLES.get(domain, [])
        
        reasoning_steps.append({
            "step": 1,
            "action": "identify_principles",
            "domain": domain,
            "relevant_principles": principles[:3]  # Top 3 most relevant
        })
        
        # Step 2: Extract known facts from the question
        known_facts = self._extract_known_facts(question)
        reasoning_steps.append({
            "step": 2,
            "action": "extract_facts",
            "facts": known_facts
        })
        
        # Step 3: Build reasoning chain from principles to answer
        reasoning_chain = self._build_reasoning_chain(
            principles, known_facts, problem_analysis
        )
        reasoning_steps.append({
            "step": 3,
            "action": "build_chain",
            "chain": reasoning_chain
        })
        
        # Step 4: Derive conclusion
        conclusion = self._derive_conclusion(reasoning_chain, question)
        intermediate_conclusions.append(conclusion)
        
        # Check for issues
        if not reasoning_chain:
            issues.append("Could not build complete reasoning chain")
        if len(principles) == 0:
            issues.append(f"No principles found for domain: {domain}")
        
        return StrategyResult(
            strategy=self.strategy,
            success=len(reasoning_chain) > 0 and conclusion is not None,
            reasoning_steps=reasoning_steps,
            intermediate_conclusions=intermediate_conclusions,
            final_answer=conclusion,
            confidence=0.7 if conclusion else 0.2,
            issues_encountered=issues,
            metadata={"domain": domain, "principles_used": len(principles)}
        )
    
    def _identify_domain(self, question: str, analysis: ProblemAnalysis) -> str:
        """Identify the most relevant domain"""
        domain_keywords = {
            "genetics": ["gene", "allele", "inheritance", "mutation", "genotype", "phenotype"],
            "molecular_biology": ["protein", "rna", "transcription", "translation", "regulation"],
            "biochemistry": ["enzyme", "reaction", "metabolism", "pathway", "kinetics"],
            "cell_biology": ["cell", "membrane", "organelle", "division", "signaling"],
            "immunology": ["antibody", "antigen", "immune", "lymphocyte", "mhc"]
        }
        
        question_lower = question.lower()
        scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in question_lower)
            if score > 0:
                scores[domain] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "biochemistry"  # Default
    
    def _extract_known_facts(self, question: str) -> List[str]:
        """Extract explicit facts from the question"""
        facts = []
        
        # Extract numerical values
        numbers = re.findall(r'\d+\.?\d*\s*(?:%|mM|µM|mM|kg|g|mg|L|mL|°C|K)?', question)
        facts.extend([f"Value: {n}" for n in numbers])
        
        # Extract conditional statements
        conditionals = re.findall(r'if\s+[^,]+,', question, re.IGNORECASE)
        facts.extend([f"Condition: {c}" for c in conditionals])
        
        # Extract explicit statements (simplified)
        sentences = re.split(r'[.,]', question)
        for sentence in sentences:
            if " is " in sentence or " are " in sentence:
                facts.append(f"Statement: {sentence.strip()}")
        
        return facts
    
    def _build_reasoning_chain(
        self,
        principles: List[str],
        facts: List[str],
        analysis: ProblemAnalysis
    ) -> List[Dict[str, Any]]:
        """Build a chain of reasoning from principles to conclusion"""
        chain = []
        
        for i, principle in enumerate(principles[:3]):
            # Check if principle applies to any facts
            applicable_facts = [
                f for f in facts 
                if any(kw in f.lower() for kw in principle.lower().split()[:3])
            ]
            
            if applicable_facts or i == 0:  # First principle always included
                chain.append({
                    "principle": principle,
                    "applicable_facts": applicable_facts,
                    "inference": f"Applying: {principle}"
                })
        
        return chain
    
    def _derive_conclusion(
        self, 
        chain: List[Dict[str, Any]], 
        question: str
    ) -> Optional[str]:
        """Derive final conclusion from reasoning chain"""
        if not chain:
            return None
        
        # Simplified conclusion derivation
        # In practice, this would involve more sophisticated logic
        conclusions = []
        for step in chain:
            if step.get("applicable_facts"):
                conclusions.append(f"Based on {step['principle']}")
        
        if conclusions:
            return " → ".join(conclusions[-2:])  # Last two steps
        return None


class AnalogicalReasoner(ReasoningStrategy):
    """
    Analogical Reasoning Strategy
    
    Find similar known problems and apply their solution patterns.
    Best for problems that:
    - Have precedents in literature
    - Involve standard techniques
    - Are variations of known problem types
    """
    
    def __init__(self, knowledge_base: Optional[Any] = None):
        self.strategy = ProblemStrategy.ANALOGICAL
        self.knowledge_base = knowledge_base
    
    @property
    def strategy_name(self) -> ProblemStrategy:
        return self.strategy
    
    def can_apply(self, problem_analysis: ProblemAnalysis) -> bool:
        """Analogical reasoning works when we can find similar problems"""
        return len(problem_analysis.similar_problems) > 0
    
    def apply(
        self,
        question: str,
        problem_analysis: ProblemAnalysis,
        context: Optional[Dict[str, Any]] = None
    ) -> StrategyResult:
        """Apply analogical reasoning"""
        context = context or {}
        reasoning_steps = []
        intermediate_conclusions = []
        issues = []
        
        # Step 1: Find analogous problems
        similar = problem_analysis.similar_problems
        reasoning_steps.append({
            "step": 1,
            "action": "find_analogies",
            "similar_problems": [p.get("title", "unknown") for p in similar[:3]]
        })
        
        # Step 2: Extract solution patterns from analogies
        patterns = self._extract_patterns(similar)
        reasoning_steps.append({
            "step": 2,
            "action": "extract_patterns",
            "patterns": patterns
        })
        
        # Step 3: Apply pattern to current problem
        adaptation = self._adapt_pattern(patterns, question, problem_analysis)
        reasoning_steps.append({
            "step": 3,
            "action": "adapt_pattern",
            "adaptation": adaptation
        })
        
        # Step 4: Generate answer
        answer = adaptation.get("answer")
        if adaptation.get("warnings"):
            issues.extend(adaptation["warnings"])
        
        return StrategyResult(
            strategy=self.strategy,
            success=answer is not None,
            reasoning_steps=reasoning_steps,
            intermediate_conclusions=[adaptation.get("approach", "")],
            final_answer=answer,
            confidence=0.6 if answer else 0.2,
            issues_encountered=issues,
            metadata={"patterns_used": len(patterns)}
        )
    
    def _extract_patterns(self, similar_problems: List[Dict]) -> List[Dict]:
        """Extract solution patterns from similar problems"""
        patterns = []
        
        for problem in similar_problems[:3]:
            if "solution_approach" in problem:
                patterns.append({
                    "source": problem.get("title", "unknown"),
                    "approach": problem["solution_approach"],
                    "key_steps": problem.get("solution_steps", [])
                })
        
        return patterns
    
    def _adapt_pattern(
        self,
        patterns: List[Dict],
        question: str,
        analysis: ProblemAnalysis
    ) -> Dict[str, Any]:
        """Adapt a pattern to the current problem"""
        if not patterns:
            return {"answer": None, "warnings": ["No patterns to adapt"]}
        
        # Use the first (most similar) pattern
        best_pattern = patterns[0]
        
        return {
            "approach": f"Adapting from: {best_pattern['source']}",
            "answer": f"(Pattern-based inference)",
            "warnings": ["Answer needs verification - based on analogy"]
        }


class ProblemDecomposer(ReasoningStrategy):
    """
    Problem Decomposition Strategy
    
    Break complex problems into simpler sub-problems.
    Best for problems that:
    - Have multiple distinct parts
    - Can be solved step-by-step
    - Involve multiple domains
    """
    
    def __init__(self):
        self.strategy = ProblemStrategy.DECOMPOSITION
    
    @property
    def strategy_name(self) -> ProblemStrategy:
        return self.strategy
    
    def can_apply(self, problem_analysis: ProblemAnalysis) -> bool:
        """Decomposition works for multi-part problems"""
        # Check for indicators of multi-part structure
        multi_part_indicators = [
            "and", "then", "followed by", "after", "before",
            "first", "second", "finally", "step"
        ]
        
        combined = " ".join(problem_analysis.key_concepts).lower()
        count = sum(1 for ind in multi_part_indicators if ind in combined)
        
        return count >= 1
    
    def apply(
        self,
        question: str,
        problem_analysis: ProblemAnalysis,
        context: Optional[Dict[str, Any]] = None
    ) -> StrategyResult:
        """Apply problem decomposition"""
        context = context or {}
        reasoning_steps = []
        intermediate_conclusions = []
        issues = []
        
        # Step 1: Decompose problem
        sub_problems = self._decompose(question, problem_analysis)
        reasoning_steps.append({
            "step": 1,
            "action": "decompose",
            "sub_problems": [sp["description"] for sp in sub_problems]
        })
        
        # Step 2: Solve each sub-problem
        sub_solutions = []
        for i, sub in enumerate(sub_problems):
            solution = self._solve_subproblem(sub, context)
            sub_solutions.append(solution)
            reasoning_steps.append({
                "step": 2 + i,
                "action": f"solve_subproblem_{i+1}",
                "subproblem": sub["description"],
                "solution": solution.get("answer", "unsolved")
            })
            intermediate_conclusions.append(
                f"Sub-problem {i+1}: {solution.get('answer', 'pending')}"
            )
        
        # Step 3: Combine solutions
        final = self._combine_solutions(sub_solutions, question)
        
        # Check for unsolved sub-problems
        unsolved = [i for i, s in enumerate(sub_solutions) if not s.get("answer")]
        if unsolved:
            issues.append(f"Sub-problems {unsolved} could not be solved")
        
        return StrategyResult(
            strategy=self.strategy,
            success=len(unsolved) == 0,
            reasoning_steps=reasoning_steps,
            intermediate_conclusions=intermediate_conclusions,
            final_answer=final.get("answer"),
            confidence=0.5 * (1 - len(unsolved) / max(len(sub_problems), 1)),
            issues_encountered=issues,
            metadata={"sub_problems_count": len(sub_problems)}
        )
    
    def _decompose(self, question: str, analysis: ProblemAnalysis) -> List[Dict]:
        """Decompose problem into sub-problems"""
        sub_problems = []
        
        # Split by explicit markers
        parts = re.split(r'\s+(?:and|then|followed by|after|before)\s+', question)
        
        for i, part in enumerate(parts[:4]):  # Max 4 sub-problems
            sub_problems.append({
                "id": i,
                "description": part.strip(),
                "type": self._classify_subproblem(part)
            })
        
        return sub_problems if sub_problems else [{"id": 0, "description": question, "type": "unknown"}]
    
    def _classify_subproblem(self, text: str) -> str:
        """Classify the type of sub-problem"""
        if re.search(r'\d+', text):
            return "calculation"
        elif "what" in text.lower():
            return "identification"
        elif "why" in text.lower():
            return "explanation"
        elif "how" in text.lower():
            return "process"
        return "general"
    
    def _solve_subproblem(self, sub: Dict, context: Dict) -> Dict:
        """Solve a single sub-problem"""
        # Placeholder - in practice would call appropriate solver
        return {
            "subproblem_id": sub["id"],
            "answer": f"[Solution for: {sub['description'][:50]}...]",
            "confidence": 0.5
        }
    
    def _combine_solutions(self, solutions: List[Dict], original: str) -> Dict:
        """Combine sub-solutions into final answer"""
        answers = [s.get("answer", "") for s in solutions if s.get("answer")]
        return {
            "answer": " → ".join(answers) if answers else None,
            "method": "sequential_combination"
        }


class OriginalProblemHandler:
    """
    Main handler for original/novel problems that can't be found via search.
    
    Coordinates multiple strategies and selects the best approach.
    """
    
    def __init__(self):
        self.strategies: Dict[ProblemStrategy, ReasoningStrategy] = {
            ProblemStrategy.FIRST_PRINCIPLES: FirstPrinciplesReasoner(),
            ProblemStrategy.ANALOGICAL: AnalogicalReasoner(),
            ProblemStrategy.DECOMPOSITION: ProblemDecomposer(),
        }
        
        # Strategy priority order
        self.strategy_priority = [
            ProblemStrategy.DECOMPOSITION,  # Try to break down first
            ProblemStrategy.ANALOGICAL,     # Then look for similar problems
            ProblemStrategy.FIRST_PRINCIPLES,  # Finally reason from scratch
        ]
    
    def analyze_problem(self, question: str) -> ProblemAnalysis:
        """Analyze the problem to determine characteristics"""
        # Extract key concepts
        key_concepts = self._extract_concepts(question)
        
        # Extract constraints
        constraints = self._extract_constraints(question)
        
        # Determine required knowledge
        required_knowledge = self._identify_required_knowledge(question, key_concepts)
        
        # Find similar problems (placeholder)
        similar_problems = self._find_similar_problems(question)
        
        # Determine suggested strategies
        suggested = self._suggest_strategies(question, key_concepts)
        
        return ProblemAnalysis(
            problem_type=self._classify_problem(question),
            key_concepts=key_concepts,
            constraints=constraints,
            required_knowledge=required_knowledge,
            suggested_strategies=suggested,
            difficulty_indicators={
                "concept_count": len(key_concepts),
                "constraint_count": len(constraints),
                "has_calculation": bool(re.search(r'\d+', question))
            },
            similar_problems=similar_problems
        )
    
    def handle(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> OriginalProblemResult:
        """
        Handle an original problem using the best available strategy.
        
        Args:
            question: The original/novel question
            context: Optional context (previous results, domain hints, etc.)
            
        Returns:
            OriginalProblemResult with answer and reasoning trace
        """
        context = context or {}
        
        # Step 1: Analyze the problem
        analysis = self.analyze_problem(question)
        
        # Step 2: Try strategies in priority order
        results = []
        for strategy_name in self.strategy_priority:
            strategy = self.strategies.get(strategy_name)
            
            if strategy and strategy.can_apply(analysis):
                result = strategy.apply(question, analysis, context)
                results.append(result)
                
                # If high confidence, stop
                if result.confidence > 0.8:
                    break
        
        # Step 3: If no strategy worked well, try ensemble
        if not results or max(r.confidence for r in results) < 0.5:
            ensemble_result = self._ensemble_approach(question, analysis, results, context)
            if ensemble_result:
                results.append(ensemble_result)
        
        # Step 4: Select best result
        if results:
            best = max(results, key=lambda r: r.confidence)
            all_results = results
        else:
            best = StrategyResult(
                strategy=ProblemStrategy.FIRST_PRINCIPLES,
                success=False,
                reasoning_steps=[{"error": "No applicable strategy found"}],
                intermediate_conclusions=[],
                final_answer=None,
                confidence=0.0,
                issues_encountered=["No strategy could be applied"]
            )
            all_results = [best]
        
        # Build reasoning trace
        reasoning_trace = []
        for r in all_results:
            reasoning_trace.extend(r.reasoning_steps)
        
        return OriginalProblemResult(
            answer=best.final_answer,
            confidence=best.confidence,
            primary_strategy=best.strategy,
            strategy_results=all_results,
            reasoning_trace=reasoning_trace,
            verification_status="partially_verified" if best.confidence > 0.6 else "unverified",
            metadata={
                "strategies_tried": [r.strategy.value for r in all_results],
                "analysis": {
                    "key_concepts": analysis.key_concepts[:5],
                    "problem_type": analysis.problem_type
                }
            }
        )
    
    def _extract_concepts(self, question: str) -> List[str]:
        """Extract key concepts from question"""
        # Simple keyword extraction
        biology_terms = [
            "protein", "gene", "cell", "enzyme", "dna", "rna", "membrane",
            "receptor", "ligand", "pathway", "signaling", "metabolism",
            "transcription", "translation", "mutation", "phenotype", "genotype"
        ]
        
        concepts = []
        question_lower = question.lower()
        for term in biology_terms:
            if term in question_lower:
                concepts.append(term)
        
        return list(set(concepts))
    
    def _extract_constraints(self, question: str) -> List[str]:
        """Extract constraints from question"""
        constraints = []
        
        # Numerical constraints
        numbers = re.findall(r'(?:less than|more than|at least|at most|exactly)\s+(\d+)', question, re.I)
        constraints.extend([f"numerical: {n}" for n in numbers])
        
        # Conditional constraints
        if "if" in question.lower():
            constraints.append("conditional logic required")
        
        return constraints
    
    def _identify_required_knowledge(self, question: str, concepts: List[str]) -> List[str]:
        """Identify what knowledge is needed"""
        knowledge = []
        
        if "protein" in concepts:
            knowledge.append("protein structure and function")
        if "gene" in concepts or "dna" in concepts:
            knowledge.append("genetics and molecular biology")
        if "enzyme" in concepts:
            knowledge.append("enzyme kinetics and catalysis")
        
        return knowledge
    
    def _find_similar_problems(self, question: str) -> List[Dict]:
        """Find similar problems (placeholder for knowledge base query)"""
        # In practice, this would query a knowledge base
        return []
    
    def _suggest_strategies(self, question: str, concepts: List[str]) -> List[ProblemStrategy]:
        """Suggest strategies based on problem characteristics"""
        strategies = []
        
        question_lower = question.lower()
        
        # Decomposition for multi-step problems
        if any(word in question_lower for word in ["then", "and", "followed by"]):
            strategies.append(ProblemStrategy.DECOMPOSITION)
        
        # First principles for mechanistic questions
        if any(word in question_lower for word in ["how does", "mechanism", "why"]):
            strategies.append(ProblemStrategy.FIRST_PRINCIPLES)
        
        # Default to first principles
        if not strategies:
            strategies.append(ProblemStrategy.FIRST_PRINCIPLES)
        
        return strategies
    
    def _classify_problem(self, question: str) -> str:
        """Classify the type of problem"""
        question_lower = question.lower()
        
        if "calculate" in question_lower or re.search(r'\d+\s*[\+\-\*\/]', question):
            return "calculation"
        elif "what" in question_lower:
            return "identification"
        elif "why" in question_lower:
            return "explanation"
        elif "how" in question_lower:
            return "process"
        elif "which" in question_lower:
            return "selection"
        
        return "general"
    
    def _ensemble_approach(
        self,
        question: str,
        analysis: ProblemAnalysis,
        previous_results: List[StrategyResult],
        context: Dict
    ) -> Optional[StrategyResult]:
        """Combine multiple strategies for better results"""
        if len(previous_results) < 2:
            return None
        
        # Find agreement among strategies
        answers = [r.final_answer for r in previous_results if r.final_answer]
        
        if len(set(str(a).lower() for a in answers)) == 1:
            # All strategies agree - high confidence
            return StrategyResult(
                strategy=ProblemStrategy.ENSEMBLE,
                success=True,
                reasoning_steps=[{"agreement": "all strategies concur"}],
                intermediate_conclusions=["Multiple strategies agree"],
                final_answer=answers[0],
                confidence=0.85,
                metadata={"agreement_count": len(answers)}
            )
        
        # Partial agreement - use majority
        from collections import Counter
        answer_counts = Counter(str(a).lower() for a in answers)
        most_common = answer_counts.most_common(1)
        
        if most_common and most_common[0][1] >= 2:
            return StrategyResult(
                strategy=ProblemStrategy.ENSEMBLE,
                success=True,
                reasoning_steps=[{"partial_agreement": f"{most_common[0][1]} strategies agree"}],
                intermediate_conclusions=["Majority agreement among strategies"],
                final_answer=most_common[0][0],
                confidence=0.6,
                metadata={"agreement_count": most_common[0][1]}
            )
        
        return None

