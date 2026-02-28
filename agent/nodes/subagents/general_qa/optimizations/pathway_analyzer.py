"""
Metabolic Pathway Analyzer - Critical Fix for Pathway Relationship Questions

Solves the "which expression represents the relationship" problem:
1. Parse pathway notation (A -k1-> B -k2-> C)
2. Identify negative feedback loops (-| notation)
3. Determine if direct path exists
4. Calculate relationship expressions
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class EdgeType(Enum):
    POSITIVE = "positive"      # -> activation/production
    NEGATIVE = "negative"      # -| inhibition/repression
    BIDIRECTIONAL = "bidirectional"  # <->


@dataclass
class PathwayEdge:
    """Represents an edge in the pathway"""
    source: str
    target: str
    coefficient: str
    edge_type: EdgeType
    original_notation: str


@dataclass
class PathwayNode:
    """Represents a node in the pathway"""
    name: str
    symbol: str  # Short symbol like [B] for 3-Hydroxypropionate
    incoming_edges: List[PathwayEdge] = field(default_factory=list)
    outgoing_edges: List[PathwayEdge] = field(default_factory=list)


@dataclass
class PathResult:
    """Result of path analysis"""
    exists: bool
    path: List[str]
    coefficients: List[str]
    has_negative_edge: bool
    has_loop: bool
    expression: str
    reasoning: str


class PathwayAnalyzer:
    """
    Analyzer for metabolic/biochemical pathways
    
    Handles:
    - Path notation parsing (A -k1-> B)
    - Negative feedback detection (A -| B)
    - Relationship calculation
    """
    
    def __init__(self):
        self.nodes: Dict[str, PathwayNode] = {}
        self.edges: List[PathwayEdge] = []
        self.coefficients: Dict[str, float] = {}
    
    def parse_pathway(self, pathway_text: str) -> Tuple[Dict[str, PathwayNode], List[PathwayEdge]]:
        """
        Parse pathway notation from text
        
        Supports formats:
        - A -k1-> B (positive regulation)
        - A -| B (negative regulation/inhibition)
        - A -k1-| B (negative with coefficient)
        """
        self.nodes = {}
        self.edges = []
        
        # Split into lines and process each
        lines = pathway_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Pattern for positive edge with coefficient: A -k1-> B
            pos_pattern = r'(\w+(?:\([^)]+\))?)\s*-([a-z]\w*)?->\s*(\w+(?:\([^)]+\))?)'
            
            # Pattern for negative edge: A -| B or A -k1-| B
            neg_pattern = r'(\w+(?:\([^)]+\))?)\s*-([a-z]\w*)?-?\|\s*(\w+(?:\([^)]+\))?)'
            
            # Try positive pattern
            pos_match = re.search(pos_pattern, line)
            if pos_match:
                source, coef, target = pos_match.groups()
                self._add_edge(source, target, coef or '', EdgeType.POSITIVE, line)
                continue
            
            # Try negative pattern
            neg_match = re.search(neg_pattern, line)
            if neg_match:
                source, coef, target = neg_match.groups()
                self._add_edge(source, target, coef or '', EdgeType.NEGATIVE, line)
        
        return self.nodes, self.edges
    
    def _add_edge(self, source: str, target: str, coefficient: str, 
                  edge_type: EdgeType, original: str):
        """Add an edge to the pathway"""
        # Create nodes if not exist
        if source not in self.nodes:
            self.nodes[source] = PathwayNode(name=source, symbol=f"[{source[0].upper()}]")
        if target not in self.nodes:
            self.nodes[target] = PathwayNode(name=target, symbol=f"[{target[0].upper()}]")
        
        # Create edge
        edge = PathwayEdge(
            source=source,
            target=target,
            coefficient=coefficient,
            edge_type=edge_type,
            original_notation=original
        )
        
        # Add to nodes
        self.nodes[source].outgoing_edges.append(edge)
        self.nodes[target].incoming_edges.append(edge)
        
        # Add to edge list
        self.edges.append(edge)
    
    def find_all_paths(self, start: str, end: str, 
                       max_depth: int = 10) -> List[PathResult]:
        """
        Find all paths from start to end
        
        Returns list of PathResult objects
        """
        if start not in self.nodes or end not in self.nodes:
            return []
        
        paths = []
        self._dfs_paths(start, end, [], [], paths, max_depth)
        
        return paths
    
    def _dfs_paths(self, current: str, end: str, 
                   path: List[str], coefs: List[str],
                   results: List[PathResult], max_depth: int):
        """DFS to find all paths"""
        if len(path) > max_depth:
            return
        
        path = path + [current]
        
        if current == end:
            # Found a path
            has_neg = any(
                self._has_negative_edge_on_path(path)
            )
            expression = self._build_expression(coefs, has_neg)
            
            results.append(PathResult(
                exists=True,
                path=path,
                coefficients=coefs,
                has_negative_edge=has_neg,
                has_loop=self._has_loop(path),
                expression=expression,
                reasoning=self._explain_path(path, coefs, has_neg)
            ))
            return
        
        node = self.nodes[current]
        for edge in node.outgoing_edges:
            if edge.target not in path:  # Avoid cycles in simple path
                new_coefs = coefs + ([edge.coefficient] if edge.coefficient else [])
                self._dfs_paths(edge.target, end, path, new_coefs, results, max_depth)
    
    def _has_negative_edge_on_path(self, path: List[str]) -> bool:
        """Check if path has any negative edges"""
        for i in range(len(path) - 1):
            source, target = path[i], path[i+1]
            for edge in self.nodes[source].outgoing_edges:
                if edge.target == target and edge.edge_type == EdgeType.NEGATIVE:
                    return True
        return False
    
    def _has_loop(self, path: List[str]) -> bool:
        """Check if path forms a loop"""
        return len(path) != len(set(path))
    
    def _build_expression(self, coefs: List[str], has_negative: bool) -> str:
        """Build mathematical expression from coefficients"""
        if not coefs:
            return "Direct (no coefficients)"
        
        if has_negative:
            return "No direct proportional relationship (negative regulation present)"
        
        return " * ".join(coefs)
    
    def _explain_path(self, path: List[str], coefs: List[str], 
                      has_neg: bool) -> str:
        """Generate explanation for path"""
        path_str = " → ".join(path)
        
        if has_neg:
            return f"Path {path_str} contains negative regulation. No direct proportional relationship exists."
        
        if coefs:
            return f"Path {path_str} with coefficients: {' * '.join(coefs)}"
        
        return f"Direct path: {path_str}"
    
    def check_negative_feedback_loop(self, source: str, target: str) -> Tuple[bool, str]:
        """
        Check if there's negative feedback from target back to source
        
        Returns (has_negative_feedback, explanation)
        """
        if source not in self.nodes or target not in self.nodes:
            return False, "Nodes not found"
        
        # Check if target has negative edge back to source
        target_node = self.nodes[target]
        for edge in target_node.outgoing_edges:
            if edge.target == source and edge.edge_type == EdgeType.NEGATIVE:
                return True, f"Negative feedback: {target} -| {source}"
        
        # Also check intermediate paths
        # ... more complex analysis can be added here
        
        return False, "No direct negative feedback found"
    
    def analyze_relationship_question(self,
                                     pathway_text: str,
                                     entity_a: str,
                                     entity_b: str,
                                     options: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Full analysis of a relationship question
        
        Args:
            pathway_text: The pathway notation text
            entity_a: First entity (source)
            entity_b: Second entity (target)
            options: Optional dict of answer choices
            
        Returns:
            Analysis with recommended answer
        """
        # Parse pathway
        self.parse_pathway(pathway_text)
        
        # Find paths
        paths = self.find_all_paths(entity_a, entity_b)
        
        # Check for negative feedback
        has_neg_feedback, neg_feedback_explain = self.check_negative_feedback_loop(
            entity_b, entity_a
        )
        
        # Determine if direct relationship exists
        direct_path_exists = len(paths) > 0
        any_negative_in_path = any(p.has_negative_edge for p in paths)
        
        # Build analysis
        analysis = {
            'pathway_parsed': {
                'nodes': list(self.nodes.keys()),
                'edges': [(e.source, e.target, e.coefficient, e.edge_type.value) 
                         for e in self.edges]
            },
            'paths_found': len(paths),
            'path_details': [
                {
                    'path': p.path,
                    'coefficients': p.coefficients,
                    'has_negative': p.has_negative_edge,
                    'expression': p.expression
                }
                for p in paths
            ],
            'has_negative_feedback': has_neg_feedback,
            'negative_feedback_explanation': neg_feedback_explain,
            'direct_relationship_exists': direct_path_exists and not any_negative_in_path,
            'key_insight': self._determine_key_insight(
                paths, has_neg_feedback, any_negative_in_path
            )
        }
        
        # If options provided, recommend best answer
        if options:
            recommendation = self._recommend_answer(options, analysis)
            analysis['recommendation'] = recommendation
        
        return analysis
    
    def _determine_key_insight(self, paths: List[PathResult],
                               has_neg_feedback: bool,
                               any_negative: bool) -> str:
        """Determine the key insight for the answer"""
        if not paths:
            return "No direct path exists between the entities"
        
        if has_neg_feedback:
            return "Negative feedback loop breaks the proportional relationship"
        
        if any_negative:
            return "Path contains negative regulation - no simple proportional relationship"
        
        # Check if there's blocking feedback
        for p in paths:
            if p.has_loop:
                return "Path contains feedback loop - relationship is complex"
        
        if len(paths) == 1:
            return f"Single direct path exists with coefficients: {paths[0].expression}"
        
        return "Multiple paths exist - choose the most direct one"
    
    def _recommend_answer(self, options: Dict[str, str], 
                         analysis: Dict) -> Dict[str, Any]:
        """Recommend best answer from options"""
        # Key insight tells us what to look for
        insight = analysis['key_insight'].lower()
        
        # If no direct relationship or negative feedback
        if 'no direct' in insight or 'negative feedback' in insight or 'negative regulation' in insight:
            # Look for options like "[F] ¬∝ [B]" or "no relationship"
            for opt_id, opt_text in options.items():
                if '¬∝' in opt_text or 'not proportional' in opt_text.lower():
                    return {
                        'option': opt_id,
                        'text': opt_text,
                        'reason': analysis['key_insight']
                    }
        
        # If direct relationship exists
        if 'direct path' in insight or 'single direct' in insight:
            # Look for the correct coefficient expression
            if analysis['path_details']:
                correct_coefs = analysis['path_details'][0]['coefficients']
                coef_str = ' * '.join(correct_coefs)
                
                for opt_id, opt_text in options.items():
                    if coef_str in opt_text and '¬∝' not in opt_text:
                        return {
                            'option': opt_id,
                            'text': opt_text,
                            'reason': f"Correct coefficients: {coef_str}"
                        }
        
        # Fallback - return insight but no specific recommendation
        return {
            'option': None,
            'text': None,
            'reason': analysis['key_insight'] + " - Could not match to any option"
        }


# Convenience function
def analyze_pathway_relationship(pathway_text: str,
                                 entity_a: str,
                                 entity_b: str) -> Dict[str, Any]:
    """
    Quick pathway analysis function
    
    Args:
        pathway_text: The pathway notation
        entity_a: Source entity
        entity_b: Target entity
        
    Returns:
        Analysis result
    """
    analyzer = PathwayAnalyzer()
    return analyzer.analyze_relationship_question(pathway_text, entity_a, entity_b)


# Test
if __name__ == "__main__":
    test_pathway = """
    CO2 -k1-> 3-Hydroxypropionate -k2-> Malonyl-CoA -k3-> Acetyl-CoA
    Acetyl-CoA -k4-> Pyruvate -k5-> PEP -k6-> Oxaloacetate
    Oxaloacetate -k19-| Malonyl-CoA
    """
    
    analyzer = PathwayAnalyzer()
    result = analyzer.analyze_relationship_question(
        test_pathway,
        "3-Hydroxypropionate",
        "PEP"
    )
    
    print("Key Insight:", result['key_insight'])
    print("Has negative feedback:", result['has_negative_feedback'])




