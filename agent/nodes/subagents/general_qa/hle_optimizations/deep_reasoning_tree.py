"""
Deep Reasoning Tree for HLE

HLE questions require multi-step deep reasoning. This module provides
tree-based reasoning (instead of linear chains) that supports:
- Branching exploration of multiple hypotheses
- Backtracking when reasoning paths fail
- Depth-limited exploration
- Best-path selection based on evidence

Key Features:
- ReasoningNode: Single step in reasoning with multiple children
- ReasoningPath: Complete path from root to conclusion
- DeepReasoningTree: Tree structure with pruning and selection
- Support for forward and backward reasoning
"""

import heapq
from typing import Dict, Any, Optional, List, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class ReasoningNodeType(Enum):
    """Types of reasoning nodes"""
    ROOT = "root"                    # Starting question
    HYPOTHESIS = "hypothesis"        # A hypothesis to test
    EVIDENCE = "evidence"            # Supporting evidence
    INFERENCE = "inference"          # Logical inference
    CALCULATION = "calculation"      # Numerical calculation
    VALIDATION = "validation"        # Validation step
    CONCLUSION = "conclusion"        # Final conclusion
    DEAD_END = "dead_end"           # Failed reasoning path


class ReasoningStatus(Enum):
    """Status of a reasoning node"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass
class ReasoningNode:
    """
    A single node in the reasoning tree.
    
    Each node represents a step in the reasoning process and can have
    multiple children representing different ways to proceed.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_type: ReasoningNodeType = ReasoningNodeType.INFERENCE
    content: str = ""
    status: ReasoningStatus = ReasoningStatus.PENDING
    
    # Reasoning details
    premise: str = ""
    conclusion: str = ""
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    
    # Tree structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    
    # Scoring and selection
    score: float = 0.0
    cumulative_score: float = 0.0
    
    # Metadata
    reasoning_method: str = ""  # e.g., "deduction", "induction", "abduction"
    domain: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For heap-based priority queue"""
        return self.cumulative_score > other.cumulative_score


@dataclass
class ReasoningPath:
    """
    A complete reasoning path from root to conclusion.
    
    Represents one possible way to reach an answer.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    nodes: List[ReasoningNode] = field(default_factory=list)
    
    # Path metrics
    total_confidence: float = 0.0
    total_score: float = 0.0
    length: int = 0
    
    # Conclusion
    final_conclusion: Optional[str] = None
    final_answer: Optional[str] = None
    
    # Path quality
    coherence_score: float = 0.0
    evidence_strength: float = 0.0
    is_complete: bool = False
    
    @property
    def root(self) -> Optional[ReasoningNode]:
        return self.nodes[0] if self.nodes else None
    
    @property
    def conclusion_node(self) -> Optional[ReasoningNode]:
        for node in reversed(self.nodes):
            if node.node_type == ReasoningNodeType.CONCLUSION:
                return node
        return None
    
    def calculate_metrics(self):
        """Calculate path metrics from nodes"""
        if not self.nodes:
            return
        
        self.length = len(self.nodes)
        
        # Total confidence (product of individual confidences)
        confidence = 1.0
        for node in self.nodes:
            if node.confidence > 0:
                confidence *= node.confidence
        self.total_confidence = confidence
        
        # Total score (average of scores)
        scores = [n.score for n in self.nodes if n.score > 0]
        self.total_score = sum(scores) / len(scores) if scores else 0.0
        
        # Coherence (how well nodes connect)
        self.coherence_score = self._calculate_coherence()
        
        # Evidence strength
        all_evidence = []
        for node in self.nodes:
            all_evidence.extend(node.evidence)
        self.evidence_strength = min(1.0, len(all_evidence) * 0.2)
    
    def _calculate_coherence(self) -> float:
        """Calculate how coherent the reasoning chain is"""
        if len(self.nodes) <= 1:
            return 1.0
        
        coherence = 0.0
        for i in range(1, len(self.nodes)):
            # Check if previous conclusion connects to current premise
            prev_conclusion = self.nodes[i-1].conclusion.lower()
            curr_premise = self.nodes[i].premise.lower()
            
            # Simple keyword overlap check
            prev_words = set(prev_conclusion.split())
            curr_words = set(curr_premise.split())
            overlap = len(prev_words & curr_words)
            
            if overlap > 0:
                coherence += 0.2
        
        return min(1.0, coherence)


class DeepReasoningTree:
    """
    Tree-based reasoning for complex problems.
    
    Unlike linear reasoning chains, this tree structure allows:
    1. Exploring multiple hypotheses in parallel
    2. Backtracking when a path fails
    3. Pruning low-quality branches
    4. Selecting the best complete path
    """
    
    def __init__(
        self,
        max_depth: int = 8,
        max_branches: int = 3,
        min_confidence: float = 0.3,
        exploration_bonus: float = 0.1
    ):
        """
        Initialize the reasoning tree.
        
        Args:
            max_depth: Maximum depth of reasoning
            max_branches: Maximum branches per node
            min_confidence: Minimum confidence to continue a path
            exploration_bonus: Bonus for exploring new paths
        """
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.min_confidence = min_confidence
        self.exploration_bonus = exploration_bonus
        
        # Tree storage
        self.nodes: Dict[str, ReasoningNode] = {}
        self.root_id: Optional[str] = None
        
        # Tracking
        self.complete_paths: List[ReasoningPath] = []
        self.dead_ends: List[str] = []
    
    def initialize(self, question: str, domain: str = "") -> ReasoningNode:
        """
        Initialize the tree with the root question.
        
        Args:
            question: The question to reason about
            domain: Domain hint for reasoning
            
        Returns:
            The root node
        """
        root = ReasoningNode(
            node_type=ReasoningNodeType.ROOT,
            content=question,
            premise="",
            conclusion=question,
            confidence=1.0,
            score=1.0,
            depth=0,
            domain=domain
        )
        
        self.nodes[root.id] = root
        self.root_id = root.id
        
        return root
    
    def add_child(
        self,
        parent_id: str,
        node_type: ReasoningNodeType,
        content: str,
        premise: str = "",
        conclusion: str = "",
        confidence: float = 0.5,
        evidence: List[str] = None,
        reasoning_method: str = ""
    ) -> Optional[ReasoningNode]:
        """
        Add a child node to the tree.
        
        Args:
            parent_id: ID of parent node
            node_type: Type of the new node
            content: Content/description of the reasoning step
            premise: Premise for this step
            conclusion: Conclusion from this step
            confidence: Confidence in this step
            evidence: Supporting evidence
            reasoning_method: Method used (deduction, induction, etc.)
            
        Returns:
            The new node, or None if parent not found
        """
        if parent_id not in self.nodes:
            return None
        
        parent = self.nodes[parent_id]
        
        # Check branch limit
        if len(parent.children_ids) >= self.max_branches:
            return None
        
        # Check depth limit
        if parent.depth >= self.max_depth:
            return None
        
        child = ReasoningNode(
            node_type=node_type,
            content=content,
            premise=premise,
            conclusion=conclusion,
            confidence=confidence,
            evidence=evidence or [],
            parent_id=parent_id,
            depth=parent.depth + 1,
            reasoning_method=reasoning_method,
            domain=parent.domain,
            score=self._calculate_node_score(parent, confidence)
        )
        
        child.cumulative_score = parent.cumulative_score + child.score
        
        self.nodes[child.id] = child
        parent.children_ids.append(child.id)
        
        return child
    
    def _calculate_node_score(self, parent: ReasoningNode, confidence: float) -> float:
        """Calculate score for a new node"""
        # Base score from confidence
        score = confidence
        
        # Depth penalty (prefer shorter paths)
        depth_penalty = 0.05 * parent.depth
        
        # Exploration bonus
        exploration = self.exploration_bonus if len(parent.children_ids) == 0 else 0
        
        return score - depth_penalty + exploration
    
    def mark_dead_end(self, node_id: str, reason: str = ""):
        """Mark a node as a dead end"""
        if node_id in self.nodes:
            self.nodes[node_id].status = ReasoningStatus.FAILED
            self.nodes[node_id].node_type = ReasoningNodeType.DEAD_END
            self.nodes[node_id].metadata["dead_end_reason"] = reason
            self.dead_ends.append(node_id)
    
    def get_best_path(self) -> Optional[ReasoningPath]:
        """
        Get the best complete path through the tree.
        
        Returns:
            The highest-scoring complete path, or None if no complete paths
        """
        if not self.complete_paths:
            # Try to build paths from root
            self._build_complete_paths()
        
        if not self.complete_paths:
            return None
        
        # Sort by combined score
        self.complete_paths.sort(
            key=lambda p: (
                p.total_confidence * 0.4 + 
                p.coherence_score * 0.3 + 
                p.evidence_strength * 0.3
            ),
            reverse=True
        )
        
        return self.complete_paths[0]
    
    def _build_complete_paths(self):
        """Build all complete paths from root to conclusions"""
        if not self.root_id:
            return
        
        # DFS to find all paths to conclusions
        self._dfs_paths(self.root_id, [], [])
    
    def _dfs_paths(
        self, 
        node_id: str, 
        current_path: List[str],
        visited: List[str]
    ):
        """Depth-first search for complete paths"""
        if node_id in visited:
            return
        
        node = self.nodes.get(node_id)
        if not node:
            return
        
        visited.append(node_id)
        current_path.append(node_id)
        
        # Check if this is a conclusion
        if node.node_type == ReasoningNodeType.CONCLUSION:
            path = self._create_path(current_path)
            if path:
                self.complete_paths.append(path)
        
        # Continue to children
        for child_id in node.children_ids:
            self._dfs_paths(child_id, current_path.copy(), visited.copy())
    
    def _create_path(self, node_ids: List[str]) -> Optional[ReasoningPath]:
        """Create a ReasoningPath from a list of node IDs"""
        nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]
        
        if not nodes:
            return None
        
        path = ReasoningPath(nodes=nodes)
        path.calculate_metrics()
        path.is_complete = True
        
        # Extract final answer
        conclusion_node = path.conclusion_node
        if conclusion_node:
            path.final_conclusion = conclusion_node.conclusion
            path.final_answer = conclusion_node.conclusion
        
        return path
    
    def get_nodes_at_depth(self, depth: int) -> List[ReasoningNode]:
        """Get all nodes at a specific depth"""
        return [n for n in self.nodes.values() if n.depth == depth]
    
    def get_children(self, node_id: str) -> List[ReasoningNode]:
        """Get all children of a node"""
        node = self.nodes.get(node_id)
        if not node:
            return []
        
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]
    
    def get_path_to_node(self, node_id: str) -> List[ReasoningNode]:
        """Get the path from root to a specific node"""
        path = []
        current_id = node_id
        
        while current_id:
            node = self.nodes.get(current_id)
            if not node:
                break
            
            path.append(node)
            current_id = node.parent_id
        
        return list(reversed(path))
    
    def prune_low_confidence_branches(self, threshold: float = None):
        """Remove branches with confidence below threshold"""
        threshold = threshold or self.min_confidence
        
        for node_id, node in list(self.nodes.items()):
            if node.confidence < threshold and node.id != self.root_id:
                node.status = ReasoningStatus.PRUNED
                # Remove from parent's children
                if node.parent_id:
                    parent = self.nodes.get(node.parent_id)
                    if parent and node.id in parent.children_ids:
                        parent.children_ids.remove(node.id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the tree"""
        return {
            "total_nodes": len(self.nodes),
            "max_depth_reached": max((n.depth for n in self.nodes.values()), default=0),
            "complete_paths": len(self.complete_paths),
            "dead_ends": len(self.dead_ends),
            "node_types": {
                node_type.value: sum(
                    1 for n in self.nodes.values() 
                    if n.node_type == node_type
                )
                for node_type in ReasoningNodeType
            },
            "average_confidence": sum(
                n.confidence for n in self.nodes.values()
            ) / len(self.nodes) if self.nodes else 0
        }


class ReasoningTreeBuilder:
    """
    Builder for creating reasoning trees with various strategies.
    
    Provides high-level methods for constructing reasoning trees
    without manually managing each node.
    """
    
    def __init__(self, tree: Optional[DeepReasoningTree] = None):
        self.tree = tree or DeepReasoningTree()
    
    def from_question(
        self,
        question: str,
        domain: str = "",
        hypotheses: List[str] = None
    ) -> DeepReasoningTree:
        """
        Build a reasoning tree from a question.
        
        Args:
            question: The question to reason about
            domain: Domain hint
            hypotheses: Initial hypotheses to explore
            
        Returns:
            The constructed tree
        """
        # Initialize with root
        root = self.tree.initialize(question, domain)
        
        # Add hypothesis branches if provided
        if hypotheses:
            for hyp in hypotheses[:self.tree.max_branches]:
                self.tree.add_child(
                    parent_id=root.id,
                    node_type=ReasoningNodeType.HYPOTHESIS,
                    content=f"Hypothesis: {hyp}",
                    premise=question,
                    conclusion=hyp,
                    confidence=0.5,
                    reasoning_method="abduction"
                )
        
        return self.tree
    
    def add_evidence(
        self,
        node_id: str,
        evidence: str,
        supports: bool = True,
        confidence: float = 0.7
    ) -> ReasoningNode:
        """Add evidence to support or refute a node"""
        node_type = ReasoningNodeType.EVIDENCE
        content = f"Evidence {'supporting' if supports else 'refuting'}: {evidence}"
        
        return self.tree.add_child(
            parent_id=node_id,
            node_type=node_type,
            content=content,
            premise=evidence,
            conclusion=f"{'Supports' if supports else 'Refutes'} hypothesis",
            confidence=confidence if supports else 1 - confidence,
            evidence=[evidence],
            reasoning_method="observation"
        )
    
    def add_inference(
        self,
        node_id: str,
        inference: str,
        conclusion: str,
        confidence: float = 0.6,
        method: str = "deduction"
    ) -> ReasoningNode:
        """Add an inference step"""
        return self.tree.add_child(
            parent_id=node_id,
            node_type=ReasoningNodeType.INFERENCE,
            content=inference,
            premise=self.tree.nodes[node_id].conclusion if node_id in self.tree.nodes else "",
            conclusion=conclusion,
            confidence=confidence,
            reasoning_method=method
        )
    
    def add_calculation(
        self,
        node_id: str,
        description: str,
        result: str,
        confidence: float = 0.9
    ) -> ReasoningNode:
        """Add a calculation step"""
        return self.tree.add_child(
            parent_id=node_id,
            node_type=ReasoningNodeType.CALCULATION,
            content=description,
            premise="",
            conclusion=result,
            confidence=confidence,
            reasoning_method="calculation"
        )
    
    def add_conclusion(
        self,
        node_id: str,
        conclusion: str,
        confidence: float = 0.7
    ) -> ReasoningNode:
        """Add a conclusion node"""
        return self.tree.add_child(
            parent_id=node_id,
            node_type=ReasoningNodeType.CONCLUSION,
            content=f"Conclusion: {conclusion}",
            premise=self.tree.nodes[node_id].conclusion if node_id in self.tree.nodes else "",
            conclusion=conclusion,
            confidence=confidence,
            reasoning_method="synthesis"
        )
    
    def explore_hypothesis(
        self,
        hypothesis_id: str,
        supporting_evidence: List[str] = None,
        refuting_evidence: List[str] = None,
        inferences: List[Tuple[str, str]] = None
    ):
        """
        Explore a hypothesis by adding evidence and inferences.
        
        Args:
            hypothesis_id: ID of the hypothesis node
            supporting_evidence: List of supporting evidence
            refuting_evidence: List of refuting evidence
            inferences: List of (inference, conclusion) tuples
        """
        # Add supporting evidence
        for evidence in (supporting_evidence or []):
            self.add_evidence(hypothesis_id, evidence, supports=True)
        
        # Add refuting evidence
        for evidence in (refuting_evidence or []):
            self.add_evidence(hypothesis_id, evidence, supports=False)
        
        # Add inferences
        for inference, conclusion in (inferences or []):
            self.add_inference(hypothesis_id, inference, conclusion)
    
    def finalize(self) -> Tuple[Optional[ReasoningPath], Dict[str, Any]]:
        """
        Finalize the tree and get the best path.
        
        Returns:
            Tuple of (best_path, statistics)
        """
        # Prune low-confidence branches
        self.tree.prune_low_confidence_branches()
        
        # Get best path
        best_path = self.tree.get_best_path()
        
        # Get statistics
        stats = self.tree.get_statistics()
        
        if best_path:
            stats["best_path_confidence"] = best_path.total_confidence
            stats["best_path_length"] = best_path.length
            stats["best_path_conclusion"] = best_path.final_conclusion
        
        return best_path, stats

