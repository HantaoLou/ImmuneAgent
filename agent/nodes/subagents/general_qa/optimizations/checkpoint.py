"""
Checkpoint Manager - P3 Priority Optimization

Implements checkpoint/recovery for long-running sessions:
1. Save state at each node completion
2. Resume from last successful checkpoint
3. Partial result preservation
4. Session recovery on failure
"""

import json
import os
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class CheckpointStatus(Enum):
    """Status of a checkpoint"""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERED = "recovered"


@dataclass
class NodeCheckpoint:
    """Checkpoint for a single node"""
    node_name: str
    timestamp: float
    state: Dict[str, Any]
    status: CheckpointStatus
    duration: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_name': self.node_name,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'state': self.state,
            'status': self.status.value,
            'duration': self.duration,
            'error': self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeCheckpoint':
        return cls(
            node_name=data['node_name'],
            timestamp=data['timestamp'],
            state=data['state'],
            status=CheckpointStatus(data['status']),
            duration=data['duration'],
            error=data.get('error')
        )


@dataclass
class SessionCheckpoint:
    """Full session checkpoint"""
    session_id: str
    question_id: str
    question_text: str
    created_at: float
    updated_at: float
    node_checkpoints: Dict[str, NodeCheckpoint] = field(default_factory=dict)
    current_node: Optional[str] = None
    final_state: Optional[Dict[str, Any]] = None
    status: CheckpointStatus = CheckpointStatus.ACTIVE
    total_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'question_id': self.question_id,
            'question_text': self.question_text,
            'created_at': self.created_at,
            'created_at_iso': datetime.fromtimestamp(self.created_at).isoformat(),
            'updated_at': self.updated_at,
            'updated_at_iso': datetime.fromtimestamp(self.updated_at).isoformat(),
            'node_checkpoints': {k: v.to_dict() for k, v in self.node_checkpoints.items()},
            'current_node': self.current_node,
            'final_state': self.final_state,
            'status': self.status.value,
            'total_duration': self.total_duration
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionCheckpoint':
        checkpoints = {
            k: NodeCheckpoint.from_dict(v) 
            for k, v in data.get('node_checkpoints', {}).items()
        }
        return cls(
            session_id=data['session_id'],
            question_id=data['question_id'],
            question_text=data['question_text'],
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            node_checkpoints=checkpoints,
            current_node=data.get('current_node'),
            final_state=data.get('final_state'),
            status=CheckpointStatus(data.get('status', 'active')),
            total_duration=data.get('total_duration', 0.0)
        )


class CheckpointManager:
    """
    Manages checkpoints for session recovery
    """
    
    # Default node execution order
    NODE_ORDER = [
        "input_preprocessing",
        "n1_question_decomposition",
        "n2_calculation_recognition",
        "n3_knowledge_retrieval",
        "n4_calculation_decomposition",
        "n5_parameter_extraction",
        "n6_initial_inference",
        "n7_complete_inference",
        "n8_answer_generation",
        "n9_result_validation",
        "n10_answer_finalization"
    ]
    
    def __init__(self,
                 checkpoint_dir: Optional[str] = None,
                 auto_save: bool = True,
                    max_checkpoints: int = 100):
        self.checkpoint_dir = checkpoint_dir or os.path.join(os.getcwd(), 'checkpoints')
        self.auto_save = auto_save
        self.max_checkpoints = max_checkpoints
        self.current_session: Optional[SessionCheckpoint] = None
        self._node_start_times: Dict[str, float] = {}
    
    def start_session(self, 
                      session_id: str, 
                      question_id: str, 
                      question_text: str) -> SessionCheckpoint:
        """
        Start a new session checkpoint
        
        Args:
            session_id: Unique session identifier
            question_id: Question identifier
            question_text: The question text
            
        Returns:
            SessionCheckpoint object
        """
        now = time.time()
        
        self.current_session = SessionCheckpoint(
            session_id=session_id,
            question_id=question_id,
            question_text=question_text,
            created_at=now,
            updated_at=now,
            status=CheckpointStatus.ACTIVE
        )
        
        if self.auto_save:
            self._save_checkpoint()
        
        return self.current_session
    
    def save_node_checkpoint(self,
                            node_name: str,
                            state: Dict[str, Any],
                            status: CheckpointStatus = CheckpointStatus.COMPLETED,
                            error: Optional[str] = None):
        """
        Save checkpoint after node execution
        
        Args:
            node_name: Name of the node
            state: Current state after node execution
            status: Node execution status
            error: Error message if failed
        """
        if not self.current_session:
            return
        
        # Calculate duration
        duration = 0.0
        if node_name in self._node_start_times:
            duration = time.time() - self._node_start_times[node_name]
        
        checkpoint = NodeCheckpoint(
            node_name=node_name,
            timestamp=time.time(),
            state=state.copy(),
            status=status,
            duration=duration,
            error=error
        )
        
        self.current_session.node_checkpoints[node_name] = checkpoint
        self.current_session.current_node = node_name
        self.current_session.updated_at = time.time()
        self.current_session.total_duration += duration
        
        if status == CheckpointStatus.FAILED:
            self.current_session.status = CheckpointStatus.FAILED
        
        if self.auto_save:
            self._save_checkpoint()
    
    def start_node(self, node_name: str):
        """Mark node execution start"""
        self._node_start_times[node_name] = time.time()
    
    def finalize_session(self, final_state: Dict[str, Any]):
        """Finalize the session"""
        if not self.current_session:
            return
        
        self.current_session.final_state = final_state
        self.current_session.status = CheckpointStatus.COMPLETED
        self.current_session.updated_at = time.time()
        
        if self.auto_save:
            self._save_checkpoint()
    
    def recover_session(self, session_id: str) -> Optional[SessionCheckpoint]:
        """
        Recover a previous session
        
        Args:
            session_id: Session to recover
            
        Returns:
            SessionCheckpoint if found, None otherwise
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{session_id}.json")
        
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = SessionCheckpoint.from_dict(data)
            session.status = CheckpointStatus.RECOVERED
            self.current_session = session
            
            return session
        except Exception:
            return None
    
    def get_recovery_point(self) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Get the last successful checkpoint for recovery
        
        Returns:
            Tuple of (next_node_to_run, state_to_use)
        """
        if not self.current_session:
            return None, {}
        
        # Find the last completed node
        last_completed = None
        last_state = {}
        
        for node_name in self.NODE_ORDER:
            if node_name in self.current_session.node_checkpoints:
                checkpoint = self.current_session.node_checkpoints[node_name]
                if checkpoint.status == CheckpointStatus.COMPLETED:
                    last_completed = node_name
                    last_state = checkpoint.state.copy()
            else:
                # This node hasn't run yet
                if last_completed:
                    return node_name, last_state
                break
        
        # All nodes completed
        if self.current_session.final_state:
            return None, self.current_session.final_state
        
        return None, last_state
    
    def get_missing_nodes(self) -> List[str]:
        """Get list of nodes that haven't been executed"""
        if not self.current_session:
            return self.NODE_ORDER.copy()
        
        missing = []
        found_missing = False
        
        for node_name in self.NODE_ORDER:
            if node_name not in self.current_session.node_checkpoints:
                found_missing = True
                missing.append(node_name)
            elif found_missing:
                # If we found a missing node, all subsequent nodes should be re-run
                missing.append(node_name)
        
        return missing
    
    def _save_checkpoint(self):
        """Save current session to disk"""
        if not self.current_session:
            return
        
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            filepath = os.path.join(self.checkpoint_dir, f"{self.current_session.session_id}.json")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.current_session.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
        except Exception:
            pass
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space"""
        try:
            if not os.path.exists(self.checkpoint_dir):
                return
            
            # List all checkpoint files
            files = []
            for f in os.listdir(self.checkpoint_dir):
                if f.endswith('.json'):
                    filepath = os.path.join(self.checkpoint_dir, f)
                    files.append((filepath, os.path.getmtime(filepath)))
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old files
            for filepath, _ in files[self.max_checkpoints:]:
                os.remove(filepath)
        except Exception:
            pass
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions"""
        sessions = []
        
        if not os.path.exists(self.checkpoint_dir):
            return sessions
        
        for filename in os.listdir(self.checkpoint_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(self.checkpoint_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                sessions.append({
                    'session_id': data.get('session_id'),
                    'question_id': data.get('question_id'),
                    'status': data.get('status'),
                    'created_at': data.get('created_at_iso'),
                    'updated_at': data.get('updated_at_iso'),
                    'total_duration': data.get('total_duration'),
                    'node_count': len(data.get('node_checkpoints', {}))
                })
            except Exception:
                continue
        
        # Sort by updated_at descending
        sessions.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        
        return sessions
    
    def get_recovery_report(self) -> str:
        """Generate a recovery report"""
        if not self.current_session:
            return "No active session"
        
        lines = ["# Session Recovery Report\n"]
        
        lines.append(f"## Session Info")
        lines.append(f"- **Session ID**: {self.current_session.session_id}")
        lines.append(f"- **Question ID**: {self.current_session.question_id}")
        lines.append(f"- **Status**: {self.current_session.status.value}")
        lines.append(f"- **Total Duration**: {self.current_session.total_duration:.2f}s")
        
        lines.append(f"\n## Node Execution Status")
        
        for node_name in self.NODE_ORDER:
            if node_name in self.current_session.node_checkpoints:
                checkpoint = self.current_session.node_checkpoints[node_name]
                icon = {
                    CheckpointStatus.COMPLETED: "✅",
                    CheckpointStatus.FAILED: "❌",
                    CheckpointStatus.ACTIVE: "⏳",
                    CheckpointStatus.RECOVERED: "🔄"
                }.get(checkpoint.status, "❓")
                
                lines.append(f"- {icon} {node_name}: {checkpoint.status.value} ({checkpoint.duration:.2f}s)")
                
                if checkpoint.error:
                    lines.append(f"  - Error: {checkpoint.error}")
            else:
                lines.append(f"- ⬜ {node_name}: not executed")
        
        next_node, state = self.get_recovery_point()
        if next_node:
            lines.append(f"\n## Recovery Recommendation")
            lines.append(f"- **Resume from node**: {next_node}")
            lines.append(f"- **State keys available**: {list(state.keys())}")
        
        return "\n".join(lines)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session checkpoint"""
        filepath = os.path.join(self.checkpoint_dir, f"{session_id}.json")
        
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False


# Convenience function
def create_checkpoint_manager(checkpoint_dir: Optional[str] = None) -> CheckpointManager:
    """Create a checkpoint manager"""
    return CheckpointManager(checkpoint_dir=checkpoint_dir)




