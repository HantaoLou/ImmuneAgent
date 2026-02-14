"""
Decision Logger - P2 Priority Optimization

Logs all decisions for explainability:
1. Why certain tools were selected
2. Why certain knowledge was prioritized
3. Why certain answers were rejected
4. Decision confidence and reasoning
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import os
from datetime import datetime


class DecisionType(Enum):
    """Types of decisions"""
    TOOL_SELECTION = "tool_selection"
    KNOWLEDGE_PRIORITIZATION = "knowledge_prioritization"
    ANSWER_GENERATION = "answer_generation"
    ANSWER_REJECTION = "answer_rejection"
    CONSTRAINT_APPLICATION = "constraint_application"
    OPTION_COMPARISON = "option_comparison"
    FALLBACK_TRIGGER = "fallback_trigger"
    TIMEOUT_DECISION = "timeout_decision"


@dataclass
class DecisionEntry:
    """A single decision entry"""
    timestamp: float
    decision_type: DecisionType
    node_name: str
    input_context: Dict[str, Any]
    options_considered: List[Dict[str, Any]]
    selected_option: str
    reasoning: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'decision_type': self.decision_type.value,
            'node_name': self.node_name,
            'input_context': self.input_context,
            'options_considered': self.options_considered,
            'selected_option': self.selected_option,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class DecisionLog:
    """Complete decision log for a session"""
    session_id: str
    question_id: str
    question_text: str
    entries: List[DecisionEntry] = field(default_factory=list)
    final_answer: Optional[str] = None
    final_confidence: float = 0.0
    total_duration: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'session_id': self.session_id,
            'question_id': self.question_id,
            'question_text': self.question_text,
            'entries': [e.to_dict() for e in self.entries],
            'final_answer': self.final_answer,
            'final_confidence': self.final_confidence,
            'total_duration': self.total_duration,
            'created_at': self.created_at,
            'created_at_iso': datetime.fromtimestamp(self.created_at).isoformat()
        }


class DecisionLogger:
    """
    Logs and manages decision history for explainability
    """
    
    def __init__(self, 
                 log_dir: Optional[str] = None,
                 max_entries: int = 1000):
        self.log_dir = log_dir or os.path.join(os.getcwd(), 'decision_logs')
        self.max_entries = max_entries
        self.current_log: Optional[DecisionLog] = None
        self._decision_counts: Dict[DecisionType, int] = {}
    
    def start_session(self, 
                      session_id: str, 
                      question_id: str, 
                      question_text: str):
        """Start a new decision logging session"""
        self.current_log = DecisionLog(
            session_id=session_id,
            question_id=question_id,
            question_text=question_text
        )
        self._decision_counts = {}
    
    def log_decision(self,
                     decision_type: DecisionType,
                     node_name: str,
                     input_context: Dict[str, Any],
                     options_considered: List[Dict[str, Any]],
                     selected_option: str,
                     reasoning: str,
                     confidence: float,
                     metadata: Optional[Dict] = None):
        """
        Log a decision
        
        Args:
            decision_type: Type of decision
            node_name: Name of the node making the decision
            input_context: Input that influenced the decision
            options_considered: All options that were evaluated
            selected_option: The option that was selected
            reasoning: Why this option was selected
            confidence: Confidence level (0-1)
            metadata: Additional metadata
        """
        if not self.current_log:
            return
        
        entry = DecisionEntry(
            timestamp=time.time(),
            decision_type=decision_type,
            node_name=node_name,
            input_context=input_context,
            options_considered=options_considered,
            selected_option=selected_option,
            reasoning=reasoning,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.current_log.entries.append(entry)
        
        # Track counts
        self._decision_counts[decision_type] = self._decision_counts.get(decision_type, 0) + 1
        
        # Enforce max entries
        if len(self.current_log.entries) > self.max_entries:
            self.current_log.entries = self.current_log.entries[-self.max_entries:]
    
    def log_tool_selection(self,
                          node_name: str,
                          available_tools: List[str],
                          selected_tools: List[str],
                          relevance_scores: Dict[str, float],
                          domain_info: Dict[str, Any]):
        """Log tool selection decision"""
        options = [
            {'tool': t, 'score': relevance_scores.get(t, 0.0)}
            for t in available_tools
        ]
        
        self.log_decision(
            decision_type=DecisionType.TOOL_SELECTION,
            node_name=node_name,
            input_context={
                'detected_domains': domain_info.get('domains', []),
                'question_length': domain_info.get('question_length', 0)
            },
            options_considered=options,
            selected_option=', '.join(selected_tools),
            reasoning=f"Selected {len(selected_tools)} tools based on domain relevance and keyword matching",
            confidence=sum(relevance_scores.get(t, 0) for t in selected_tools) / len(selected_tools) if selected_tools else 0
        )
    
    def log_knowledge_prioritization(self,
                                     node_name: str,
                                     knowledge_sources: Dict[str, Any],
                                     prioritized_sources: List[str],
                                     reasoning: str):
        """Log knowledge prioritization decision"""
        options = [
            {'source': s, 'data_preview': str(knowledge_sources.get(s, ''))[:100]}
            for s in knowledge_sources.keys()
        ]
        
        self.log_decision(
            decision_type=DecisionType.KNOWLEDGE_PRIORITIZATION,
            node_name=node_name,
            input_context={
                'source_count': len(knowledge_sources),
                'has_disgenet': 'disgenet' in knowledge_sources,
                'has_knowledge_graph': 'knowledge_graph' in knowledge_sources
            },
            options_considered=options,
            selected_option=', '.join(prioritized_sources),
            reasoning=reasoning,
            confidence=0.8 if prioritized_sources else 0.0
        )
    
    def log_answer_rejection(self,
                            node_name: str,
                            candidate_answer: str,
                            rejection_reason: str,
                            constraint_violations: List[str]):
        """Log answer rejection"""
        self.log_decision(
            decision_type=DecisionType.ANSWER_REJECTION,
            node_name=node_name,
            input_context={
                'candidate': candidate_answer[:100],
                'violations': constraint_violations
            },
            options_considered=[{'answer': candidate_answer, 'violations': constraint_violations}],
            selected_option="REJECTED",
            reasoning=rejection_reason,
            confidence=0.9  # High confidence in rejection if constraints violated
        )
    
    def log_option_comparison(self,
                             node_name: str,
                             options: Dict[str, str],
                             scores: Dict[str, float],
                             winner: str,
                             reasoning: str):
        """Log option comparison decision"""
        options_list = [
            {'option_id': k, 'text': v[:100], 'score': scores.get(k, 0)}
            for k, v in options.items()
        ]
        
        self.log_decision(
            decision_type=DecisionType.OPTION_COMPARISON,
            node_name=node_name,
            input_context={'option_count': len(options)},
            options_considered=options_list,
            selected_option=winner,
            reasoning=reasoning,
            confidence=scores.get(winner, 0)
        )
    
    def log_fallback_trigger(self,
                            node_name: str,
                            failed_tool: str,
                            failure_reason: str,
                            fallback_tool: str):
        """Log fallback trigger"""
        self.log_decision(
            decision_type=DecisionType.FALLBACK_TRIGGER,
            node_name=node_name,
            input_context={'failed_tool': failed_tool, 'failure_reason': failure_reason},
            options_considered=[{'tool': failed_tool, 'status': 'failed'}],
            selected_option=fallback_tool,
            reasoning=f"Primary tool '{failed_tool}' failed: {failure_reason}",
            confidence=0.6  # Lower confidence for fallback
        )
    
    def finalize_session(self,
                        final_answer: str,
                        final_confidence: float):
        """Finalize the current session"""
        if not self.current_log:
            return
        
        self.current_log.final_answer = final_answer
        self.current_log.final_confidence = final_confidence
        self.current_log.total_duration = time.time() - self.current_log.created_at
    
    def save_log(self, filename: Optional[str] = None) -> str:
        """Save the current log to file"""
        if not self.current_log:
            return ""
        
        # Create log directory if needed
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"decision_log_{self.current_log.session_id}_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        # Write log
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.current_log.to_dict(), f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def get_decision_summary(self) -> str:
        """Generate a summary of all decisions"""
        if not self.current_log:
            return "No active session"
        
        lines = ["# Decision Summary\n"]
        
        lines.append(f"**Question**: {self.current_log.question_text[:100]}...\n")
        lines.append(f"## Decision Counts")
        for dt, count in self._decision_counts.items():
            lines.append(f"- {dt.value}: {count}")
        
        lines.append(f"\n## Key Decisions")
        
        # Group by type
        by_type = {}
        for entry in self.current_log.entries:
            dt = entry.decision_type
            if dt not in by_type:
                by_type[dt] = []
            by_type[dt].append(entry)
        
        # Show most important decisions
        important_types = [
            DecisionType.TOOL_SELECTION,
            DecisionType.OPTION_COMPARISON,
            DecisionType.ANSWER_REJECTION
        ]
        
        for dt in important_types:
            if dt in by_type:
                for entry in by_type[dt][:3]:  # Top 3 per type
                    lines.append(f"\n### {dt.value} @ {entry.node_name}")
                    lines.append(f"- **Selected**: {entry.selected_option}")
                    lines.append(f"- **Reasoning**: {entry.reasoning}")
                    lines.append(f"- **Confidence**: {entry.confidence:.0%}")
        
        if self.current_log.final_answer:
            lines.append(f"\n## Final Answer")
            lines.append(f"**{self.current_log.final_answer}** (confidence: {self.current_log.final_confidence:.0%})")
        
        return "\n".join(lines)
    
    def get_explainability_report(self) -> str:
        """Generate detailed explainability report"""
        if not self.current_log:
            return "No active session"
        
        lines = ["# Explainability Report\n"]
        lines.append(f"**Question ID**: {self.current_log.question_id}")
        lines.append(f"**Question**: {self.current_log.question_text}\n")
        
        # Timeline of decisions
        lines.append("## Decision Timeline\n")
        
        for i, entry in enumerate(self.current_log.entries, 1):
            time_str = datetime.fromtimestamp(entry.timestamp).strftime('%H:%M:%S')
            lines.append(f"### {i}. {entry.decision_type.value} ({time_str})")
            lines.append(f"**Node**: {entry.node_name}")
            lines.append(f"**Selected**: {entry.selected_option}")
            lines.append(f"**Confidence**: {entry.confidence:.0%}")
            lines.append(f"**Reasoning**: {entry.reasoning}")
            
            if entry.options_considered:
                lines.append("\n**Options Considered**:")
                for opt in entry.options_considered[:5]:
                    lines.append(f"- {opt}")
            
            lines.append("")
        
        if self.current_log.final_answer:
            lines.append("## Conclusion\n")
            lines.append(f"**Final Answer**: {self.current_log.final_answer}")
            lines.append(f"**Confidence**: {self.current_log.final_confidence:.0%}")
            lines.append(f"**Duration**: {self.current_log.total_duration:.2f}s")
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decision statistics"""
        if not self.current_log:
            return {}
        
        stats = {
            'total_decisions': len(self.current_log.entries),
            'by_type': {dt.value: count for dt, count in self._decision_counts.items()},
            'average_confidence': 0.0,
            'nodes_with_decisions': set()
        }
        
        if self.current_log.entries:
            total_conf = sum(e.confidence for e in self.current_log.entries)
            stats['average_confidence'] = total_conf / len(self.current_log.entries)
            stats['nodes_with_decisions'] = list(set(e.node_name for e in self.current_log.entries))
        
        return stats


# Singleton instance for easy access
_logger_instance: Optional[DecisionLogger] = None


def get_decision_logger() -> DecisionLogger:
    """Get the global decision logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = DecisionLogger()
    return _logger_instance
