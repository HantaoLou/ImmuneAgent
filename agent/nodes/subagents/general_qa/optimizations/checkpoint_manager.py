"""
P3-2: Checkpoint Manager

This module provides checkpointing for long-running operations:
- State serialization at key points
- Recovery from failures
- Progress tracking
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os
import uuid
import copy


class CheckpointStatus(Enum):
    """Status of a checkpoint"""
    CREATED = "created"
    RESTORED = "restored"
    EXPIRED = "expired"
    COMPLETED = "completed"


@dataclass
class Checkpoint:
    """A single checkpoint"""
    checkpoint_id: str
    session_id: str
    node_name: str
    state: Dict[str, Any]
    created_at: datetime
    status: CheckpointStatus = CheckpointStatus.CREATED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "node_name": self.node_name,
            "state": self._serialize_state(self.state),
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "metadata": self.metadata
        }
    
    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize state for JSON compatibility"""
        serialized = {}
        for key, value in state.items():
            try:
                # Try JSON serialization
                json.dumps(value)
                serialized[key] = value
            except (TypeError, ValueError):
                # Convert to string if not JSON serializable
                serialized[key] = str(value)
        return serialized


class CheckpointManager:
    """
    Manages checkpoints for session recovery
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "/tmp/bio_agent_checkpoints",
        max_checkpoints_per_session: int = 10,
        auto_cleanup: bool = True
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints_per_session
        self.auto_cleanup = auto_cleanup
        
        self.checkpoints: Dict[str, List[Checkpoint]] = {}
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self,
        session_id: str,
        node_name: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """
        Save a checkpoint
        
        Args:
            session_id: Session identifier
            node_name: Name of the node creating checkpoint
            state: Current state to save
            metadata: Additional metadata
            
        Returns:
            The created Checkpoint
        """
        checkpoint = Checkpoint(
            checkpoint_id=str(uuid.uuid4())[:8],
            session_id=session_id,
            node_name=node_name,
            state=copy.deepcopy(state),
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        # Add to memory
        if session_id not in self.checkpoints:
            self.checkpoints[session_id] = []
        
        self.checkpoints[session_id].append(checkpoint)
        
        # Trim old checkpoints
        if len(self.checkpoints[session_id]) > self.max_checkpoints:
            self.checkpoints[session_id] = self.checkpoints[session_id][-self.max_checkpoints:]
        
        # Save to disk
        self._save_to_disk(checkpoint)
        
        return checkpoint
    
    def restore_from_checkpoint(
        self,
        session_id: str,
        checkpoint_id: Optional[str] = None,
        node_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Restore state from a checkpoint
        
        Args:
            session_id: Session identifier
            checkpoint_id: Specific checkpoint ID (optional)
            node_name: Restore from last checkpoint of this node (optional)
            
        Returns:
            Restored state or None if not found
        """
        checkpoints = self.checkpoints.get(session_id, [])
        
        if checkpoint_id:
            checkpoint = next(
                (c for c in checkpoints if c.checkpoint_id == checkpoint_id),
                None
            )
        elif node_name:
            checkpoint = next(
                (c for c in reversed(checkpoints) if c.node_name == node_name),
                None
            )
        else:
            checkpoint = checkpoints[-1] if checkpoints else None
        
        if checkpoint:
            checkpoint.status = CheckpointStatus.RESTORED
            return copy.deepcopy(checkpoint.state)
        
        return None
    
    def get_last_successful_checkpoint(
        self,
        session_id: str,
        node_name: str
    ) -> Optional[Checkpoint]:
        """Get the last successful checkpoint for a node"""
        checkpoints = self.checkpoints.get(session_id, [])
        return next(
            (c for c in reversed(checkpoints) if c.node_name == node_name),
            None
        )
    
    def list_checkpoints(self, session_id: str) -> List[Checkpoint]:
        """List all checkpoints for a session"""
        return self.checkpoints.get(session_id, [])
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint"""
        for session_id, checkpoints in self.checkpoints.items():
            for i, cp in enumerate(checkpoints):
                if cp.checkpoint_id == checkpoint_id:
                    del checkpoints[i]
                    self._delete_from_disk(checkpoint_id)
                    return True
        return False
    
    def clear_session(self, session_id: str):
        """Clear all checkpoints for a session"""
        if session_id in self.checkpoints:
            del self.checkpoints[session_id]
            self._delete_session_from_disk(session_id)
    
    def _save_to_disk(self, checkpoint: Checkpoint):
        """Save checkpoint to disk"""
        filename = os.path.join(
            self.checkpoint_dir,
            f"{checkpoint.session_id}_{checkpoint.checkpoint_id}.json"
        )
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
    
    def _delete_from_disk(self, checkpoint_id: str):
        """Delete checkpoint from disk"""
        for filename in os.listdir(self.checkpoint_dir):
            if checkpoint_id in filename:
                os.remove(os.path.join(self.checkpoint_dir, filename))
                break
    
    def _delete_session_from_disk(self, session_id: str):
        """Delete all checkpoints for a session from disk"""
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(session_id):
                os.remove(os.path.join(self.checkpoint_dir, filename))
    
    def load_session_from_disk(self, session_id: str) -> List[Checkpoint]:
        """Load checkpoints from disk for a session"""
        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(session_id) and filename.endswith('.json'):
                filepath = os.path.join(self.checkpoint_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    checkpoint = Checkpoint(
                        checkpoint_id=data["checkpoint_id"],
                        session_id=data["session_id"],
                        node_name=data["node_name"],
                        state=data["state"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        status=CheckpointStatus(data["status"]),
                        metadata=data.get("metadata", {})
                    )
                    checkpoints.append(checkpoint)
                except Exception as e:
                    print(f"Failed to load checkpoint {filename}: {e}")
        
        # Sort by creation time
        checkpoints.sort(key=lambda c: c.created_at)
        return checkpoints


def save_checkpoint(
    session_id: str,
    node_name: str,
    state: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Checkpoint:
    """
    Convenience function to save a checkpoint
    """
    manager = CheckpointManager()
    return manager.save_checkpoint(session_id, node_name, state, metadata)


def restore_from_checkpoint(
    session_id: str,
    checkpoint_id: Optional[str] = None,
    node_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to restore from checkpoint
    """
    manager = CheckpointManager()
    return manager.restore_from_checkpoint(session_id, checkpoint_id, node_name)


# Test function
def test_checkpoint_manager():
    """Test the checkpoint manager"""
    manager = CheckpointManager(checkpoint_dir="/tmp/test_checkpoints")
    
    print("=" * 80)
    print("Checkpoint Manager Test")
    print("=" * 80)
    
    session_id = "test_session_001"
    
    # Create checkpoints at different nodes
    state1 = {"processed": True, "data": "input preprocessing complete"}
    cp1 = manager.save_checkpoint(session_id, "input_preprocessing", state1)
    print(f"\nCreated checkpoint 1: {cp1.checkpoint_id} at {cp1.node_name}")
    
    state2 = {"processed": True, "decomposed_questions": ["Q1", "Q2"]}
    cp2 = manager.save_checkpoint(session_id, "question_decomposition", state2)
    print(f"Created checkpoint 2: {cp2.checkpoint_id} at {cp2.node_name}")
    
    state3 = {"knowledge": "retrieved", "results": {"kg": "data", "db": "data"}}
    cp3 = manager.save_checkpoint(session_id, "knowledge_retrieval", state3)
    print(f"Created checkpoint 3: {cp3.checkpoint_id} at {cp3.node_name}")
    
    # List checkpoints
    print(f"\nCheckpoints for session {session_id}:")
    for cp in manager.list_checkpoints(session_id):
        print(f"  - {cp.checkpoint_id}: {cp.node_name} ({cp.status.value})")
    
    # Restore latest
    restored = manager.restore_from_checkpoint(session_id)
    print(f"\nRestored latest checkpoint:")
    print(f"  State: {restored}")
    
    # Restore specific node
    restored = manager.restore_from_checkpoint(session_id, node_name="question_decomposition")
    print(f"\nRestored question_decomposition checkpoint:")
    print(f"  State: {restored}")
    
    # Clear
    manager.clear_session(session_id)
    print(f"\nAfter clear: {len(manager.list_checkpoints(session_id))} checkpoints")


if __name__ == "__main__":
    test_checkpoint_manager()

