"""
State Completeness Checker - P2 Priority Optimization

Validates state integrity between nodes:
1. Required fields check per node
2. Type validation
3. Cross-node consistency
4. Auto-fill with sensible defaults
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = "error"      # Critical - must fix
    WARNING = "warning"  # Should fix
    INFO = "info"        # Informational


@dataclass
class ValidationIssue:
    """A single validation issue"""
    level: ValidationLevel
    field_name: str
    message: str
    suggestion: str
    current_value: Any = None


@dataclass
class ValidationResult:
    """Result of state validation"""
    is_valid: bool
    is_complete: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    auto_filled_fields: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == ValidationLevel.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == ValidationLevel.WARNING]


# Node requirements definition
NODE_REQUIREMENTS = {
    "input_preprocessing": {
        "required": ["question", "question_id"],
        "recommended": ["question_type", "options"],
        "types": {
            "question": str,
            "question_id": str,
            "question_type": str,
            "options": dict
        }
    },
    "n1_question_decomposition": {
        "required": ["question"],
        "recommended": ["sub_questions", "decomposition_type"],
        "types": {
            "sub_questions": list,
            "decomposition_type": str
        }
    },
    "n2_calculation_recognition": {
        "required": ["question"],
        "recommended": ["is_calculation", "calculation_type", "parameters_needed"],
        "types": {
            "is_calculation": bool,
            "calculation_type": str,
            "parameters_needed": list
        }
    },
    "n3_knowledge_retrieval": {
        "required": ["question"],
        "recommended": ["domain_knowledge_map", "retrieval_status", "knowledge_sources"],
        "types": {
            "domain_knowledge_map": dict,
            "retrieval_status": dict,
            "knowledge_sources": list
        }
    },
    "n4_calculation_decomposition": {
        "required": ["is_calculation"],
        "recommended": ["calculation_steps", "formula_components", "intermediate_values"],
        "types": {
            "calculation_steps": list,
            "formula_components": dict,
            "intermediate_values": dict
        },
        "conditional": {
            "condition": {"field": "is_calculation", "value": True},
            "then_required": ["calculation_steps"]
        }
    },
    "n5_parameter_extraction": {
        "required": ["question"],
        "recommended": ["key_parameters", "parameter_sources", "inferred_parameters"],
        "types": {
            "key_parameters": dict,
            "parameter_sources": dict,
            "inferred_parameters": dict
        }
    },
    "n6_initial_inference": {
        "required": ["question"],
        "recommended": ["initial_inference", "inference_confidence", "reasoning_chain"],
        "types": {
            "initial_inference": str,
            "inference_confidence": float,
            "reasoning_chain": list
        }
    },
    "n7_complete_inference": {
        "required": ["question"],
        "recommended": ["complete_inference", "final_reasoning", "constraint_validation"],
        "types": {
            "complete_inference": str,
            "final_reasoning": str,
            "constraint_validation": dict
        }
    },
    "n8_answer_generation": {
        "required": ["question"],
        "recommended": ["generated_answer", "answer_type", "answer_confidence"],
        "types": {
            "generated_answer": str,
            "answer_type": str,
            "answer_confidence": float
        }
    },
    "n9_result_validation": {
        "required": ["generated_answer"],
        "recommended": ["validation_result", "validation_errors", "corrected_answer"],
        "types": {
            "validation_result": bool,
            "validation_errors": list,
            "corrected_answer": str
        }
    }
}

# Default values for auto-fill
DEFAULT_VALUES = {
    "domain_knowledge_map": {},
    "retrieval_status": {"status": "pending"},
    "knowledge_sources": [],
    "key_parameters": {},
    "parameter_sources": {},
    "inferred_parameters": {},
    "calculation_steps": [],
    "formula_components": {},
    "intermediate_values": {},
    "initial_inference": "",
    "inference_confidence": 0.0,
    "reasoning_chain": [],
    "complete_inference": "",
    "final_reasoning": "",
    "constraint_validation": {"passed": True, "violations": []},
    "generated_answer": "",
    "answer_type": "unknown",
    "answer_confidence": 0.0,
    "validation_result": True,
    "validation_errors": [],
    "corrected_answer": "",
    "is_calculation": False,
    "calculation_type": "",
    "parameters_needed": [],
    "sub_questions": [],
    "decomposition_type": "none"
}


class StateCompletenessChecker:
    """
    Validates and ensures state completeness
    """
    
    def __init__(self, 
                 node_requirements: Optional[Dict] = None,
                 auto_fill: bool = True):
        self.node_requirements = node_requirements or NODE_REQUIREMENTS
        self.auto_fill = auto_fill
        self.default_values = DEFAULT_VALUES
    
    def validate_state(self, 
                       state: Dict[str, Any], 
                       node_name: str) -> ValidationResult:
        """
        Validate state for a specific node
        
        Args:
            state: The current state dictionary
            node_name: Name of the node to validate for
            
        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult(
            is_valid=True,
            is_complete=True
        )
        
        if node_name not in self.node_requirements:
            result.issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field_name="node_name",
                message=f"No requirements defined for node '{node_name}'",
                suggestion="Add node requirements to the configuration"
            ))
            return result
        
        requirements = self.node_requirements[node_name]
        
        # Check required fields
        for field_name in requirements.get("required", []):
            if field_name not in state or state[field_name] is None:
                result.missing_fields.append(field_name)
                result.is_complete = False
                result.is_valid = False
                result.issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    field_name=field_name,
                    message=f"Required field '{field_name}' is missing",
                    suggestion=f"Provide value for '{field_name}'"
                ))
        
        # Check conditional requirements
        if "conditional" in requirements:
            cond = requirements["conditional"]
            condition_field = cond["condition"]["field"]
            condition_value = cond["condition"]["value"]
            
            if state.get(condition_field) == condition_value:
                for field_name in cond.get("then_required", []):
                    if field_name not in state or state[field_name] is None:
                        result.missing_fields.append(field_name)
                        result.is_complete = False
                        result.issues.append(ValidationIssue(
                            level=ValidationLevel.ERROR,
                            field_name=field_name,
                            message=f"Conditional field '{field_name}' is required when {condition_field}={condition_value}",
                            suggestion=f"Provide value for '{field_name}'"
                        ))
        
        # Check recommended fields
        for field_name in requirements.get("recommended", []):
            if field_name not in state or state[field_name] is None:
                result.issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    field_name=field_name,
                    message=f"Recommended field '{field_name}' is missing",
                    suggestion=f"Consider providing value for '{field_name}'",
                    current_value=state.get(field_name)
                ))
        
        # Type validation
        type_requirements = requirements.get("types", {})
        for field_name, expected_type in type_requirements.items():
            if field_name in state and state[field_name] is not None:
                actual_value = state[field_name]
                if not isinstance(actual_value, expected_type):
                    result.issues.append(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        field_name=field_name,
                        message=f"Field '{field_name}' has wrong type: expected {expected_type.__name__}, got {type(actual_value).__name__}",
                        suggestion=f"Convert '{field_name}' to {expected_type.__name__}",
                        current_value=actual_value
                    ))
                    result.is_valid = False
        
        return result
    
    def ensure_completeness(self,
                           state: Dict[str, Any],
                           node_name: str) -> Tuple[Dict[str, Any], ValidationResult]:
        """
        Ensure state completeness by auto-filling missing fields
        
        Args:
            state: The current state dictionary
            node_name: Name of the node
            
        Returns:
            Tuple of (updated_state, validation_result)
        """
        result = self.validate_state(state, node_name)
        
        if not self.auto_fill:
            return state, result
        
        updated_state = state.copy()
        
        # Auto-fill missing fields with defaults
        for field_name in result.missing_fields:
            if field_name in self.default_values:
                default_value = self.default_values[field_name]
                updated_state[field_name] = default_value
                result.auto_filled_fields[field_name] = default_value
                
                # Check if this resolves an error
                for issue in result.errors[:]:
                    if issue.field_name == field_name:
                        issue.level = ValidationLevel.INFO
                        issue.message = f"Auto-filled field '{field_name}' with default value"
                        result.issues.remove(issue)
        
        # Re-validate
        result.missing_fields = [f for f in result.missing_fields 
                                 if f not in result.auto_filled_fields]
        
        if not result.errors:
            result.is_valid = True
        
        if not result.missing_fields:
            result.is_complete = True
        
        return updated_state, result
    
    def validate_transition(self,
                           from_node: str,
                           to_node: str,
                           state: Dict[str, Any]) -> ValidationResult:
        """
        Validate state transition between nodes
        
        Args:
            from_node: Source node name
            to_node: Target node name
            state: Current state
            
        Returns:
            ValidationResult for the transition
        """
        result = ValidationResult(is_valid=True, is_complete=True)
        
        # Check if source node produced expected outputs
        if from_node in self.node_requirements:
            from_req = self.node_requirements[from_node]
            recommended = from_req.get("recommended", [])
            
            for field_name in recommended:
                if field_name in state:
                    value = state[field_name]
                    # Check if value is meaningful
                    if value is None or value == "" or value == {} or value == []:
                        result.issues.append(ValidationIssue(
                            level=ValidationLevel.WARNING,
                            field_name=field_name,
                            message=f"Field '{field_name}' from '{from_node}' is empty",
                            suggestion=f"Check why '{field_name}' was not populated"
                        ))
        
        # Check target node requirements
        to_result = self.validate_state(state, to_node)
        result.issues.extend(to_result.issues)
        result.missing_fields.extend(to_result.missing_fields)
        result.is_valid = result.is_valid and to_result.is_valid
        result.is_complete = result.is_complete and to_result.is_complete
        
        return result
    
    def check_cross_node_consistency(self, state: Dict[str, Any]) -> List[ValidationIssue]:
        """
        Check consistency across node outputs
        
        Args:
            state: Full state dictionary
            
        Returns:
            List of consistency issues
        """
        issues = []
        
        # Check if is_calculation matches calculation_steps
        is_calc = state.get("is_calculation", False)
        calc_steps = state.get("calculation_steps", [])
        
        if is_calc and not calc_steps:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field_name="calculation_steps",
                message="is_calculation=True but no calculation_steps defined",
                suggestion="Either set is_calculation=False or provide calculation_steps"
            ))
        
        # Check if answer_confidence matches validation_result
        conf = state.get("answer_confidence", 0)
        valid = state.get("validation_result", True)
        
        if conf < 0.5 and valid:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field_name="validation_result",
                message=f"Low answer_confidence ({conf:.2f}) but validation_result=True",
                suggestion="Review validation logic or lower confidence threshold"
            ))
        
        # Check if knowledge sources match domain knowledge
        sources = state.get("knowledge_sources", [])
        knowledge = state.get("domain_knowledge_map", {})
        
        if sources and not knowledge:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field_name="domain_knowledge_map",
                message="knowledge_sources defined but domain_knowledge_map is empty",
                suggestion="Ensure knowledge retrieval populates domain_knowledge_map"
            ))
        
        return issues
    
    def get_state_report(self, state: Dict[str, Any], 
                         node_name: Optional[str] = None) -> str:
        """Generate a state completeness report"""
        lines = ["# State Completeness Report\n"]
        
        if node_name:
            result = self.validate_state(state, node_name)
            lines.append(f"## Node: {node_name}\n")
            lines.append(f"- **Valid**: {result.is_valid}")
            lines.append(f"- **Complete**: {result.is_complete}")
            lines.append(f"- **Errors**: {len(result.errors)}")
            lines.append(f"- **Warnings**: {len(result.warnings)}")
            
            if result.missing_fields:
                lines.append(f"\n### Missing Fields")
                for f in result.missing_fields:
                    lines.append(f"- {f}")
            
            if result.issues:
                lines.append(f"\n### Issues")
                for issue in result.issues:
                    icon = {"error": "[ERROR]", "warning": "[WARN]️", "info": "ℹ️"}.get(issue.level.value, "•")
                    lines.append(f"- {icon} **{issue.field_name}**: {issue.message}")
                    if issue.suggestion:
                        lines.append(f"  - Suggestion: {issue.suggestion}")
        else:
            # Report for all nodes
            lines.append("## All Nodes Summary\n")
            
            for node in self.node_requirements.keys():
                result = self.validate_state(state, node)
                status = "[SUCCESS]" if result.is_valid and result.is_complete else "[ERROR]"
                lines.append(f"- {status} **{node}**: {len(result.errors)} errors, {len(result.warnings)} warnings")
            
            # Cross-node consistency
            consistency_issues = self.check_cross_node_consistency(state)
            if consistency_issues:
                lines.append("\n## Cross-Node Consistency Issues\n")
                for issue in consistency_issues:
                    lines.append(f"- [WARN]️ **{issue.field_name}**: {issue.message}")
        
        return "\n".join(lines)
    
    def get_required_fields(self, node_name: str) -> Dict[str, List[str]]:
        """Get required and recommended fields for a node"""
        if node_name not in self.node_requirements:
            return {"required": [], "recommended": []}
        
        req = self.node_requirements[node_name]
        return {
            "required": req.get("required", []),
            "recommended": req.get("recommended", [])
        }
