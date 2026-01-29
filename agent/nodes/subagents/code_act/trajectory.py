"""
CodeAct Trajectory Recording System

Reference SE-Agent's trajectory system, implement trajectory recording, compression, and reuse for code generation and execution.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import json
import gzip
from pathlib import Path
import hashlib
from core.react_state import ReactStep, ReactStepType


class TrajectoryStatus(str, Enum):
    """Trajectory status"""
    SUCCESS = "success"  # Execution successful
    FAILED = "failed"  # Execution failed
    PARTIAL = "partial"  # Partially successful


class CodeTrajectory(BaseModel):
    """
    Code trajectory model
    
    Records a complete code generation and execution process, including:
    - Code generation information
    - Execution results
    - Error information (if any)
    - Performance metrics
    - Timestamp
    """
    # Basic information
    trajectory_id: str = Field(description="Unique trajectory ID (based on timestamp and content hash)")
    task_id: str = Field(description="Associated task ID")
    execution_mode: str = Field(description="Execution mode (mcp_tool/codeact/fix_code/fix_parameter)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Trajectory creation time")
    
    # Code generation information
    generated_code: str = Field(description="Generated code")
    code_generation_prompt: Optional[str] = Field(default=None, description="Code generation prompt (optional, for debugging)")
    code_generation_time: float = Field(default=0.0, description="Code generation time (seconds)")
    
    # Execution information
    execution_result: Optional[Dict[str, Any]] = Field(default=None, description="Execution result")
    execution_time: float = Field(default=0.0, description="Execution time (seconds)")
    status: TrajectoryStatus = Field(description="Execution status")
    
    # Error information (if failed)
    error_type: Optional[str] = Field(default=None, description="Error type")
    error_message: Optional[str] = Field(default=None, description="Error message")
    error_traceback: Optional[str] = Field(default=None, description="Error stack trace (complete)")
    error_category: Optional[str] = Field(default=None, description="Error category (code_error/parameter_error/system_error)")
    
    # Performance metrics
    code_length: int = Field(default=0, description="Code length (character count)")
    memory_usage: Optional[float] = Field(default=None, description="Memory usage (MB, if measurable)")
    sandbox_used: bool = Field(default=False, description="Whether sandbox environment was used")
    
    # Context information
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used")
    tools: List[Dict[str, Any]] = Field(default_factory=list, description="Tools used")
    inputs: List[str] = Field(default_factory=list, description="Input parameter list")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Other metadata")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)"""
        data = self.model_dump()
        # Convert datetime to string
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeTrajectory":
        """Create from dictionary (for deserialization)"""
        # Convert string to datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
    
    def get_hash(self) -> str:
        """Get trajectory hash value (for deduplication and comparison)"""
        # Generate hash based on key information
        key_info = {
            "code": self.generated_code,
            "parameters": self.parameters,
            "execution_mode": self.execution_mode
        }
        key_str = json.dumps(key_info, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def is_similar_to(self, other: "CodeTrajectory", threshold: float = 0.8) -> bool:
        """
        Determine if two trajectories are similar
        
        Args:
            other: Another trajectory
            threshold: Similarity threshold (0-1)
        
        Returns:
            Whether similar
        """
        # Simple similarity judgment: based on code and parameter similarity
        if self.execution_mode != other.execution_mode:
            return False
        
        # Code similarity (simple string comparison)
        code_similarity = self._calculate_similarity(self.generated_code, other.generated_code)
        
        # Parameter similarity
        param_similarity = self._calculate_dict_similarity(self.parameters, other.parameters)
        
        # Overall similarity
        overall_similarity = (code_similarity * 0.7 + param_similarity * 0.3)
        
        return overall_similarity >= threshold
    
    @staticmethod
    def _calculate_similarity(str1: str, str2: str) -> float:
        """Calculate similarity between two strings (simple Jaccard similarity)"""
        if not str1 or not str2:
            return 0.0
        
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _calculate_dict_similarity(dict1: Dict, dict2: Dict) -> float:
        """Calculate similarity between two dictionaries"""
        if not dict1 and not dict2:
            return 1.0
        if not dict1 or not dict2:
            return 0.0
        
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        
        common_keys = keys1 & keys2
        if not common_keys:
            return 0.0
        
        # Calculate value similarity for common keys
        similarities = []
        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]
            if val1 == val2:
                similarities.append(1.0)
            elif isinstance(val1, str) and isinstance(val2, str):
                similarities.append(CodeTrajectory._calculate_similarity(val1, val2))
            else:
                similarities.append(0.0)
        
        return sum(similarities) / len(similarities) if similarities else 0.0


class TrajectoryPool:
    """
    Trajectory pool manager
    
    Responsible for managing multiple trajectories, supports:
    - Trajectory storage and retrieval
    - Trajectory compression (reduce 80% storage space)
    - Trajectory query and analysis
    - Cross-task knowledge reuse
    """
    
    def __init__(self, pool_id: str, storage_dir: Optional[Path] = None):
        """
        Initialize trajectory pool
        
        Args:
            pool_id: Trajectory pool ID
            storage_dir: Storage directory (if None, use default directory)
        """
        self.pool_id = pool_id
        self.trajectories: List[CodeTrajectory] = []
        
        # Set storage directory
        if storage_dir is None:
            agent_dir = Path(__file__).parent.parent.parent.parent
            storage_dir = agent_dir / "trajectories" / "codeact"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Index: for fast querying
        self._task_index: Dict[str, List[str]] = {}  # task_id -> trajectory_ids
        self._status_index: Dict[TrajectoryStatus, List[str]] = {}  # status -> trajectory_ids
        self._mode_index: Dict[str, List[str]] = {}  # execution_mode -> trajectory_ids
    
    def add_trajectory(self, trajectory: CodeTrajectory) -> str:
        """
        Add trajectory to pool
        
        Args:
            trajectory: Trajectory to add
        
        Returns:
            Trajectory ID
        """
        # Generate trajectory ID (if not already)
        if not trajectory.trajectory_id:
            trajectory.trajectory_id = self._generate_trajectory_id(trajectory)
        
        # Add to list
        self.trajectories.append(trajectory)
        
        # Update indexes
        self._update_indexes(trajectory)
        
        return trajectory.trajectory_id
    
    def get_trajectories_by_task(self, task_id: str) -> List[CodeTrajectory]:
        """Get trajectories by task ID"""
        trajectory_ids = self._task_index.get(task_id, [])
        return [t for t in self.trajectories if t.trajectory_id in trajectory_ids]
    
    def get_successful_trajectories(self, execution_mode: Optional[str] = None) -> List[CodeTrajectory]:
        """Get successful trajectories"""
        trajectory_ids = self._status_index.get(TrajectoryStatus.SUCCESS, [])
        trajectories = [t for t in self.trajectories if t.trajectory_id in trajectory_ids]
        
        if execution_mode:
            trajectories = [t for t in trajectories if t.execution_mode == execution_mode]
        
        return trajectories
    
    def get_failed_trajectories(self, execution_mode: Optional[str] = None) -> List[CodeTrajectory]:
        """Get failed trajectories"""
        trajectory_ids = self._status_index.get(TrajectoryStatus.FAILED, [])
        trajectories = [t for t in self.trajectories if t.trajectory_id in trajectory_ids]
        
        if execution_mode:
            trajectories = [t for t in trajectories if t.execution_mode == execution_mode]
        
        return trajectories
    
    def find_similar_trajectories(self, trajectory: CodeTrajectory, threshold: float = 0.8) -> List[CodeTrajectory]:
        """Find similar trajectories"""
        similar = []
        for t in self.trajectories:
            if t.is_similar_to(trajectory, threshold):
                similar.append(t)
        return similar
    
    def save(self, compressed: bool = True) -> Path:
        """
        Save trajectory pool to file
        
        Args:
            compressed: Whether to compress (default True, can reduce 80% space)
        
        Returns:
            Saved file path
        """
        # Prepare data
        data = {
            "pool_id": self.pool_id,
            "trajectories": [t.to_dict() for t in self.trajectories],
            "metadata": {
                "total_count": len(self.trajectories),
                "saved_at": datetime.now().isoformat()
            }
        }
        
        # Serialize to JSON
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        
        # Determine file path
        if compressed:
            file_path = self.storage_dir / f"{self.pool_id}.tra.gz"
            # Save compressed
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                f.write(json_str)
        else:
            file_path = self.storage_dir / f"{self.pool_id}.tra.json"
            # Save directly
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return file_path
    
    @classmethod
    def load(cls, file_path: Path) -> "TrajectoryPool":
        """
        Load trajectory pool from file
        
        Args:
            file_path: File path (.tra.json or .tra.gz)
        
        Returns:
            Trajectory pool instance
        """
        # Check if compressed
        if file_path.suffix == '.gz':
            # Decompress and load
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # Load directly
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        # Create trajectory pool
        pool = cls(pool_id=data["pool_id"], storage_dir=file_path.parent)
        
        # Restore trajectories
        for traj_data in data.get("trajectories", []):
            trajectory = CodeTrajectory.from_dict(traj_data)
            pool.add_trajectory(trajectory)
        
        return pool
    
    def compress(self, use_llm: bool = True, similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """
        Compress trajectory pool (use LLM summarization, reduce 80% space)
        
        Args:
            use_llm: Whether to use LLM for intelligent compression (default True)
            similarity_threshold: Similarity threshold, trajectories exceeding this will be merged (default 0.85)
        
        Returns:
            Compression statistics
        """
        original_size = len(self.trajectories)
        original_total_size = sum(len(t.generated_code) for t in self.trajectories)
        
        if original_size == 0:
            return {
                "original_count": 0,
                "compressed_count": 0,
                "reduction_ratio": 0.0,
                "size_reduction_ratio": 0.0
            }
        
        if use_llm:
            # Use LLM for intelligent compression
            compressed_trajectories = self._compress_with_llm(similarity_threshold)
        else:
            # Simple deduplication compression
            compressed_trajectories = self._compress_simple()
        
        self.trajectories = compressed_trajectories
        compressed_size = len(self.trajectories)
        compressed_total_size = sum(len(t.generated_code) for t in self.trajectories)
        
        return {
            "original_count": original_size,
            "compressed_count": compressed_size,
            "reduction_ratio": 1 - (compressed_size / original_size) if original_size > 0 else 0.0,
            "original_total_size": original_total_size,
            "compressed_total_size": compressed_total_size,
            "size_reduction_ratio": 1 - (compressed_total_size / original_total_size) if original_total_size > 0 else 0.0
        }
    
    def _compress_simple(self) -> List[CodeTrajectory]:
        """Simple deduplication compression"""
        unique_trajectories = []
        seen_hashes = set()
        
        for traj in self.trajectories:
            traj_hash = traj.get_hash()
            if traj_hash not in seen_hashes:
                unique_trajectories.append(traj)
                seen_hashes.add(traj_hash)
        
        return unique_trajectories
    
    def _compress_with_llm(self, similarity_threshold: float = 0.85) -> List[CodeTrajectory]:
        """
        Use LLM for intelligent trajectory compression
        
        Strategy:
        1. Group trajectories by similarity
        2. For each group of similar trajectories, use LLM to generate summary
        3. Keep one representative trajectory, replace others with summaries
        4. For failed trajectories, keep all key error information
        
        Args:
            similarity_threshold: Similarity threshold
        
        Returns:
            Compressed trajectory list
        """
        try:
            from utils.llm_factory import create_reasoning_advanced_llm
            llm = create_reasoning_advanced_llm()
        except ImportError:
            # If cannot import, use simple compression
            print("  ⚠ Cannot import LLM factory, using simple compression")
            return self._compress_simple()
        
        if not llm:
            print("  ⚠ LLM unavailable, using simple compression")
            return self._compress_simple()
        
        # 1. Group by similarity
        groups = self._group_similar_trajectories(similarity_threshold)
        
        compressed = []
        for group in groups:
            if len(group) == 1:
                # Single trajectory, keep directly
                compressed.append(group[0])
            else:
                # Multiple similar trajectories, use LLM to generate summary
                summary_traj = self._summarize_trajectory_group(llm, group)
                if summary_traj:
                    compressed.append(summary_traj)
                else:
                    # LLM summary failed, keep first one
                    compressed.append(group[0])
        
        print(f"  ✓ LLM compression completed: {len(self.trajectories)} -> {len(compressed)} trajectories")
        return compressed
    
    def _group_similar_trajectories(self, threshold: float) -> List[List[CodeTrajectory]]:
        """Group trajectories by similarity"""
        groups = []
        used = set()
        
        for i, traj1 in enumerate(self.trajectories):
            if i in used:
                continue
            
            group = [traj1]
            used.add(i)
            
            for j, traj2 in enumerate(self.trajectories[i+1:], start=i+1):
                if j in used:
                    continue
                
                if traj1.is_similar_to(traj2, threshold):
                    group.append(traj2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _summarize_trajectory_group(self, llm, group: List[CodeTrajectory]) -> Optional[CodeTrajectory]:
        """
        Use LLM to summarize a group of similar trajectories
        
        Args:
            llm: LLM instance
            group: Similar trajectory group
        
        Returns:
            Representative trajectory after summarization
        """
        if not group:
            return None
        
        # Select most successful trajectory as base
        base_traj = max(group, key=lambda t: 1 if t.status == TrajectoryStatus.SUCCESS else 0)
        
        # If only 1-2 trajectories, return base trajectory directly
        if len(group) <= 2:
            return base_traj
        
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Build prompt
            system_prompt = """You are a code execution trajectory analysis expert. Your task is to merge multiple similar code execution trajectories into one representative trajectory summary.

Requirements:
1. Preserve all key information: task ID, execution mode, core code logic, execution results
2. Merge variants of similar code, extract common patterns
3. For failed trajectories, preserve all different error types and causes
4. Generate a concise but complete trajectory summary

Output Format: Trajectory summary in JSON format, containing:
- task_id: Task ID (from first trajectory)
- execution_mode: Execution mode
- generated_code: Representative code after merging (extract common patterns)
- status: Status (success if all successful, otherwise failed)
- execution_result: Merged execution result
- error_summary: If there are failures, summarize all error types and causes
- trajectory_count: Number of merged trajectories"""
            
            # Prepare trajectory information
            traj_summaries = []
            for i, traj in enumerate(group[:5]):  # Process at most 5 trajectories
                traj_info = {
                    "index": i + 1,
                    "code": traj.generated_code[:500],  # Limit length
                    "status": traj.status.value,
                    "error": traj.error_message[:200] if traj.error_message else None,
                    "error_type": traj.error_type
                }
                traj_summaries.append(traj_info)
            
            user_prompt = f"""Please merge the following {len(group)} similar code execution trajectories into one representative summary:

Trajectory List:
{json.dumps(traj_summaries, ensure_ascii=False, indent=2)}

Base Trajectory Information:
- task_id: {base_traj.task_id}
- execution_mode: {base_traj.execution_mode}
- Base code: {base_traj.generated_code[:300]}

Please generate merged trajectory summary (JSON format)."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            summary_text = response.content.strip()
            
            # Parse JSON response
            if summary_text.startswith("```json"):
                summary_text = summary_text[7:]
            if summary_text.startswith("```"):
                summary_text = summary_text[3:]
            if summary_text.endswith("```"):
                summary_text = summary_text[:-3]
            summary_text = summary_text.strip()
            
            summary_data = json.loads(summary_text)
            
            # Create summary trajectory
            summary_traj = CodeTrajectory(
                trajectory_id=f"{base_traj.trajectory_id}_summary",
                task_id=summary_data.get("task_id", base_traj.task_id),
                execution_mode=summary_data.get("execution_mode", base_traj.execution_mode),
                generated_code=summary_data.get("generated_code", base_traj.generated_code),
                status=TrajectoryStatus(summary_data.get("status", base_traj.status.value)),
                parameters=base_traj.parameters.copy(),
                tools=base_traj.tools.copy(),
                inputs=base_traj.inputs.copy(),
                execution_result=summary_data.get("execution_result", base_traj.execution_result),
                metadata={
                    "compressed": True,
                    "original_count": len(group),
                    "error_summary": summary_data.get("error_summary"),
                    "trajectory_count": summary_data.get("trajectory_count", len(group))
                }
            )
            
            return summary_traj
            
        except Exception as e:
            print(f"  ⚠ LLM trajectory summary failed: {e}, keeping base trajectory")
            return base_traj
    
    def _generate_trajectory_id(self, trajectory: CodeTrajectory) -> str:
        """Generate trajectory ID"""
        timestamp_str = trajectory.timestamp.isoformat()
        code_hash = hashlib.md5(trajectory.generated_code.encode()).hexdigest()[:8]
        return f"{trajectory.task_id}_{timestamp_str}_{code_hash}"
    
    def _update_indexes(self, trajectory: CodeTrajectory):
        """Update index"""
        traj_id = trajectory.trajectory_id
        
        # Task index
        if trajectory.task_id not in self._task_index:
            self._task_index[trajectory.task_id] = []
        if traj_id not in self._task_index[trajectory.task_id]:
            self._task_index[trajectory.task_id].append(traj_id)
        
        # Status index
        if trajectory.status not in self._status_index:
            self._status_index[trajectory.status] = []
        if traj_id not in self._status_index[trajectory.status]:
            self._status_index[trajectory.status].append(traj_id)
        
        # Mode index
        if trajectory.execution_mode not in self._mode_index:
            self._mode_index[trajectory.execution_mode] = []
        if traj_id not in self._mode_index[trajectory.execution_mode]:
            self._mode_index[trajectory.execution_mode].append(traj_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trajectory pool statistics"""
        total = len(self.trajectories)
        successful = len(self.get_successful_trajectories())
        failed = len(self.get_failed_trajectories())
        
        return {
            "total_trajectories": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "by_mode": {
                mode: len(self._mode_index.get(mode, []))
                for mode in self._mode_index.keys()
            }
        }


def _truncate_text(value: Any, max_len: int = 200) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return text if len(text) <= max_len else text[:max_len] + "..."


def build_react_steps_from_trajectory(trajectory: CodeTrajectory) -> List[ReactStep]:
    """
    Map a CodeTrajectory into React-style steps (OBS/THINK/ACT/RESULT).
    """
    steps: List[ReactStep] = []

    obs_parts = [
        f"task_id={trajectory.task_id}",
        f"mode={trajectory.execution_mode}"
    ]
    if trajectory.inputs:
        obs_parts.append(f"inputs={trajectory.inputs}")
    elif trajectory.parameters:
        obs_parts.append(f"parameters={list(trajectory.parameters.keys())}")
    steps.append(ReactStep(step_type=ReactStepType.OBS, content="; ".join(obs_parts)))

    think_content = (
        f"Generated code length={trajectory.code_length}, "
        f"gen_time={trajectory.code_generation_time:.2f}s"
    )
    steps.append(ReactStep(step_type=ReactStepType.THINK, content=think_content))

    tool_name = None
    if trajectory.tools:
        tool = trajectory.tools[0]
        tool_name = tool.get("tool_name") or tool.get("name")
    if tool_name:
        param_keys = list(trajectory.parameters.keys()) if trajectory.parameters else []
        act_content = f"Call tool={tool_name}, params={param_keys}"
    else:
        act_content = "Execute generated code"
    steps.append(ReactStep(step_type=ReactStepType.ACT, content=act_content))

    if trajectory.execution_result:
        status = trajectory.execution_result.get("status", "unknown")
        if status == "success":
            result_payload = trajectory.execution_result.get("output")
        else:
            result_payload = trajectory.execution_result.get("error")
        result_content = f"status={status}; payload={_truncate_text(result_payload)}"
    else:
        result_content = "status=unknown; payload="
    steps.append(ReactStep(step_type=ReactStepType.RESULT, content=result_content))

    return steps


def summarize_react_steps(steps: List[ReactStep], max_steps: int = 5) -> str:
    """
    Build a compact React summary for logging (last N steps).
    """
    tail = steps[-max_steps:] if steps else []
    return " | ".join(
        f"{step.step_type.value}:{_truncate_text(step.content, 120)}" for step in tail
    )

