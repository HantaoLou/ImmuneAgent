"""
Error Diagnostician

This module provides intelligent error diagnosis:
- Failure pattern recognition
- Root cause analysis
- Recovery suggestions
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ErrorCategory(Enum):
    """Categories of errors"""
    TOOL_FAILURE = "tool_failure"
    KNOWLEDGE_GAP = "knowledge_gap"
    REASONING_ERROR = "reasoning_error"
    VALIDATION_FAILURE = "validation_failure"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    INPUT_ERROR = "input_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity of errors"""
    CRITICAL = "critical"    # Cannot continue
    HIGH = "high"           # Major impact, needs recovery
    MEDIUM = "medium"       # Moderate impact
    LOW = "low"             # Minor issue


@dataclass
class DiagnosisResult:
    """Result of error diagnosis"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    root_cause: str
    affected_components: List[str] = field(default_factory=list)
    recovery_options: List[str] = field(default_factory=list)
    prevention_suggestions: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "root_cause": self.root_cause,
            "affected_components": self.affected_components,
            "recovery_options": self.recovery_options,
            "prevention_suggestions": self.prevention_suggestions,
            "related_patterns": self.related_patterns
        }


# Error patterns for diagnosis
ERROR_PATTERNS = {
    # Tool failures
    "no_results_from_tools": {
        "keywords": ["0 results", "no data", "empty result", "no matches"],
        "category": ErrorCategory.TOOL_FAILURE,
        "severity": ErrorSeverity.MEDIUM,
        "root_cause": "Knowledge retrieval tools returned no relevant results",
        "recovery": [
            "Try alternative search terms or synonyms",
            "Use fallback knowledge sources",
            "Relax query constraints",
            "Use LLM internal knowledge"
        ],
        "prevention": [
            "Pre-validate tool relevance to question domain",
            "Implement tool-question matching"
        ]
    },
    "tool_timeout": {
        "keywords": ["timeout", "timed out", "deadline exceeded"],
        "category": ErrorCategory.TIMEOUT,
        "severity": ErrorSeverity.HIGH,
        "root_cause": "Tool execution exceeded time limit",
        "recovery": [
            "Retry with longer timeout",
            "Use cached results if available",
            "Skip tool and use alternative"
        ],
        "prevention": [
            "Implement adaptive timeouts",
            "Cache frequently accessed data"
        ]
    },
    "tool_auth_error": {
        "keywords": ["authentication", "unauthorized", "api key", "forbidden"],
        "category": ErrorCategory.SYSTEM_ERROR,
        "severity": ErrorSeverity.HIGH,
        "root_cause": "Authentication or authorization failure",
        "recovery": [
            "Check API credentials",
            "Use alternative tool",
            "Continue without external data"
        ],
        "prevention": [
            "Validate credentials before use",
            "Implement credential rotation"
        ]
    },
    
    # Knowledge gaps
    "domain_knowledge_missing": {
        "keywords": ["no knowledge", "insufficient information", "cannot determine"],
        "category": ErrorCategory.KNOWLEDGE_GAP,
        "severity": ErrorSeverity.HIGH,
        "root_cause": "Required domain knowledge not available in knowledge base",
        "recovery": [
            "Enable deep research for this question",
            "Use web search for additional context",
            "Prompt user for clarification"
        ],
        "prevention": [
            "Expand knowledge base coverage",
            "Add domain-specific knowledge sources"
        ]
    },
    
    # Reasoning errors
    "answer_incorrect": {
        "keywords": ["incorrect", "wrong answer", "does not match"],
        "category": ErrorCategory.REASONING_ERROR,
        "severity": ErrorSeverity.HIGH,
        "root_cause": "Incorrect reasoning or inference led to wrong answer",
        "recovery": [
            "Re-run with self-consistency check",
            "Review reasoning steps",
            "Use alternative reasoning approach"
        ],
        "prevention": [
            "Implement self-consistency validation",
            "Add answer verification step"
        ]
    },
    "option_misinterpretation": {
        "keywords": ["wrong option", "misinterpreted", "incorrect choice"],
        "category": ErrorCategory.REASONING_ERROR,
        "severity": ErrorSeverity.HIGH,
        "root_cause": "Options were not correctly analyzed or compared",
        "recovery": [
            "Re-analyze options with explicit structure detection",
            "Use MCQ option analyzer",
            "Generate comparison table"
        ],
        "prevention": [
            "Implement structured MCQ analysis",
            "Add option semantics extraction"
        ]
    },
    
    # Validation failures
    "constraint_violation": {
        "keywords": ["violates constraint", "constraint not met", "invalid"],
        "category": ErrorCategory.VALIDATION_FAILURE,
        "severity": ErrorSeverity.MEDIUM,
        "root_cause": "Answer or parameters violate specified constraints",
        "recovery": [
            "Re-calculate with constraint awareness",
            "Filter results by constraints",
            "Adjust parameters"
        ],
        "prevention": [
            "Extract constraints early",
            "Validate continuously during reasoning"
        ]
    },
    
    # Input errors
    "invalid_input_format": {
        "keywords": ["invalid format", "parse error", "malformed"],
        "category": ErrorCategory.INPUT_ERROR,
        "severity": ErrorSeverity.LOW,
        "root_cause": "Input data has invalid format",
        "recovery": [
            "Normalize input format",
            "Apply input validation"
        ],
        "prevention": [
            "Implement robust input parsing",
            "Add format validation"
        ]
    }
}


class ErrorDiagnostician:
    """
    Intelligent error diagnosis system
    """
    
    def __init__(self):
        self.patterns = ERROR_PATTERNS
    
    def diagnose(
        self,
        error_context: Dict[str, Any],
        node_outputs: Optional[Dict[str, Any]] = None
    ) -> DiagnosisResult:
        """
        Diagnose an error based on context
        
        Args:
            error_context: Dictionary with error information
            node_outputs: Outputs from nodes for additional context
            
        Returns:
            DiagnosisResult with diagnosis and recommendations
        """
        error_id = str(id(error_context))[:8]
        
        # Extract error information
        error_message = error_context.get("error_message", "")
        error_type = error_context.get("error_type", "")
        node_name = error_context.get("node_name", "")
        tool_name = error_context.get("tool_name", "")
        
        # Normalize text for matching
        error_text = f"{error_message} {error_type}".lower()
        
        # Match against patterns
        best_match = None
        best_score = 0
        
        for pattern_name, pattern in self.patterns.items():
            score = 0
            for keyword in pattern["keywords"]:
                if keyword.lower() in error_text:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = pattern_name
        
        if best_match:
            pattern = self.patterns[best_match]
            return DiagnosisResult(
                error_id=error_id,
                category=pattern["category"],
                severity=pattern["severity"],
                root_cause=pattern["root_cause"],
                affected_components=[node_name, tool_name] if node_name else [],
                recovery_options=pattern["recovery"],
                prevention_suggestions=pattern["prevention"],
                related_patterns=[best_match]
            )
        
        # Default diagnosis for unknown errors
        return DiagnosisResult(
            error_id=error_id,
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            root_cause=f"Unknown error: {error_message[:100]}",
            affected_components=[node_name] if node_name else [],
            recovery_options=[
                "Retry the operation",
                "Check system logs for details",
                "Report issue for investigation"
            ],
            prevention_suggestions=[
                "Add error handling for this case"
            ]
        )
    
    def diagnose_failure_chain(
        self,
        node_outputs: List[Dict[str, Any]]
    ) -> List[DiagnosisResult]:
        """
        Diagnose a chain of failures across nodes
        
        Args:
            node_outputs: List of outputs from each node
            
        Returns:
            List of diagnosis results
        """
        diagnoses = []
        
        for i, output in enumerate(node_outputs):
            if output.get("error") or output.get("exception"):
                diagnosis = self.diagnose({
                    "error_message": str(output.get("error", output.get("exception"))),
                    "node_name": output.get("node_name", f"node_{i}"),
                    "tool_name": output.get("tool_name", "")
                }, node_outputs)
                diagnoses.append(diagnosis)
        
        return diagnoses
    
    def get_recovery_recommendations(
        self,
        diagnosis: DiagnosisResult
    ) -> Dict[str, Any]:
        """
        Get detailed recovery recommendations
        
        Returns:
            Dictionary with prioritized recovery steps
        """
        recommendations = {
            "immediate_actions": [],
            "retry_options": [],
            "fallback_options": [],
            "long_term_fixes": []
        }
        
        if diagnosis.severity == ErrorSeverity.CRITICAL:
            recommendations["immediate_actions"].append(
                "Stop processing and notify user"
            )
        
        for i, option in enumerate(diagnosis.recovery_options):
            if i < 2:
                recommendations["immediate_actions"].append(option)
            elif "retry" in option.lower():
                recommendations["retry_options"].append(option)
            elif "fallback" in option.lower() or "alternative" in option.lower():
                recommendations["fallback_options"].append(option)
            else:
                recommendations["long_term_fixes"].append(option)
        
        recommendations["long_term_fixes"].extend(diagnosis.prevention_suggestions)
        
        return recommendations


def diagnose_failure(
    error_message: str,
    node_name: str = "",
    tool_name: str = ""
) -> DiagnosisResult:
    """
    Convenience function to diagnose a failure
    
    Args:
        error_message: The error message
        node_name: Name of the node where error occurred
        tool_name: Name of the tool that failed
        
    Returns:
        DiagnosisResult
    """
    diagnostician = ErrorDiagnostician()
    return diagnostician.diagnose({
        "error_message": error_message,
        "node_name": node_name,
        "tool_name": tool_name
    })


# Test function
def test_error_diagnosis():
    """Test error diagnostician"""
    diagnostician = ErrorDiagnostician()
    
    print("=" * 80)
    print("Error Diagnostician Test")
    print("=" * 80)
    
    # Test tool failure
    diagnosis = diagnostician.diagnose({
        "error_message": "query_knowledge_graph returned 0 results for query",
        "node_name": "n3_knowledge_retrieval",
        "tool_name": "query_knowledge_graph"
    })
    
    print(f"\nDiagnosis 1 (Tool Failure):")
    print(f"  Category: {diagnosis.category.value}")
    print(f"  Severity: {diagnosis.severity.value}")
    print(f"  Root Cause: {diagnosis.root_cause}")
    print(f"  Recovery Options: {diagnosis.recovery_options[:2]}")
    
    # Test timeout
    diagnosis = diagnostician.diagnose({
        "error_message": "PaperQA timed out after 180 seconds",
        "node_name": "n3_knowledge_retrieval",
        "tool_name": "paper_qa"
    })
    
    print(f"\nDiagnosis 2 (Timeout):")
    print(f"  Category: {diagnosis.category.value}")
    print(f"  Severity: {diagnosis.severity.value}")
    print(f"  Root Cause: {diagnosis.root_cause}")
    print(f"  Recovery Options: {diagnosis.recovery_options}")
    
    # Get detailed recommendations
    recommendations = diagnostician.get_recovery_recommendations(diagnosis)
    print(f"\nDetailed Recommendations:")
    for category, items in recommendations.items():
        if items:
            print(f"  {category}: {items}")


if __name__ == "__main__":
    test_error_diagnosis()

