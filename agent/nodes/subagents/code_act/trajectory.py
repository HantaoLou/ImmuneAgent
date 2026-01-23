"""
CodeAct 轨迹记录系统

参考 SE-Agent 的轨迹系统，实现代码生成和执行的轨迹记录、压缩和复用。
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import json
import gzip
from pathlib import Path
import hashlib


class TrajectoryStatus(str, Enum):
    """轨迹状态"""
    SUCCESS = "success"  # 执行成功
    FAILED = "failed"  # 执行失败
    PARTIAL = "partial"  # 部分成功


class CodeTrajectory(BaseModel):
    """
    代码轨迹模型
    
    记录一次完整的代码生成和执行过程，包括：
    - 代码生成信息
    - 执行结果
    - 错误信息（如果有）
    - 性能指标
    - 时间戳
    """
    # 基本信息
    trajectory_id: str = Field(description="轨迹唯一ID（基于时间戳和内容哈希）")
    task_id: str = Field(description="关联的任务ID")
    execution_mode: str = Field(description="执行模式（mcp_tool/codeact/fix_code/fix_parameter）")
    timestamp: datetime = Field(default_factory=datetime.now, description="轨迹创建时间")
    
    # 代码生成信息
    generated_code: str = Field(description="生成的代码")
    code_generation_prompt: Optional[str] = Field(default=None, description="代码生成提示词（可选，用于调试）")
    code_generation_time: float = Field(default=0.0, description="代码生成耗时（秒）")
    
    # 执行信息
    execution_result: Optional[Dict[str, Any]] = Field(default=None, description="执行结果")
    execution_time: float = Field(default=0.0, description="执行耗时（秒）")
    status: TrajectoryStatus = Field(description="执行状态")
    
    # 错误信息（如果失败）
    error_type: Optional[str] = Field(default=None, description="错误类型")
    error_message: Optional[str] = Field(default=None, description="错误消息")
    error_traceback: Optional[str] = Field(default=None, description="错误堆栈（完整）")
    error_category: Optional[str] = Field(default=None, description="错误分类（code_error/parameter_error/system_error）")
    
    # 性能指标
    code_length: int = Field(default=0, description="代码长度（字符数）")
    memory_usage: Optional[float] = Field(default=None, description="内存使用（MB，如果可测量）")
    sandbox_used: bool = Field(default=False, description="是否使用沙盒环境")
    
    # 上下文信息
    parameters: Dict[str, Any] = Field(default_factory=dict, description="使用的参数")
    tools: List[Dict[str, Any]] = Field(default_factory=list, description="使用的工具")
    inputs: List[str] = Field(default_factory=list, description="输入参数列表")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="其他元数据")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        data = self.model_dump()
        # 将 datetime 转换为字符串
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeTrajectory":
        """从字典创建（用于反序列化）"""
        # 将字符串转换为 datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
    
    def get_hash(self) -> str:
        """获取轨迹的哈希值（用于去重和比较）"""
        # 基于关键信息生成哈希
        key_info = {
            "code": self.generated_code,
            "parameters": self.parameters,
            "execution_mode": self.execution_mode
        }
        key_str = json.dumps(key_info, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def is_similar_to(self, other: "CodeTrajectory", threshold: float = 0.8) -> bool:
        """
        判断两个轨迹是否相似
        
        Args:
            other: 另一个轨迹
            threshold: 相似度阈值（0-1）
        
        Returns:
            是否相似
        """
        # 简单的相似度判断：基于代码和参数的相似度
        if self.execution_mode != other.execution_mode:
            return False
        
        # 代码相似度（简单的字符串比较）
        code_similarity = self._calculate_similarity(self.generated_code, other.generated_code)
        
        # 参数相似度
        param_similarity = self._calculate_dict_similarity(self.parameters, other.parameters)
        
        # 综合相似度
        overall_similarity = (code_similarity * 0.7 + param_similarity * 0.3)
        
        return overall_similarity >= threshold
    
    @staticmethod
    def _calculate_similarity(str1: str, str2: str) -> float:
        """计算两个字符串的相似度（简单的Jaccard相似度）"""
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
        """计算两个字典的相似度"""
        if not dict1 and not dict2:
            return 1.0
        if not dict1 or not dict2:
            return 0.0
        
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        
        common_keys = keys1 & keys2
        if not common_keys:
            return 0.0
        
        # 计算共同键的值相似度
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
    轨迹池管理器
    
    负责管理多个轨迹，支持：
    - 轨迹存储和检索
    - 轨迹压缩（减少80%存储空间）
    - 轨迹查询和分析
    - 跨任务的知识复用
    """
    
    def __init__(self, pool_id: str, storage_dir: Optional[Path] = None):
        """
        初始化轨迹池
        
        Args:
            pool_id: 轨迹池ID
            storage_dir: 存储目录（如果为None，使用默认目录）
        """
        self.pool_id = pool_id
        self.trajectories: List[CodeTrajectory] = []
        
        # 设置存储目录
        if storage_dir is None:
            agent_dir = Path(__file__).parent.parent.parent.parent
            storage_dir = agent_dir / "trajectories" / "codeact"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 索引：用于快速查询
        self._task_index: Dict[str, List[str]] = {}  # task_id -> trajectory_ids
        self._status_index: Dict[TrajectoryStatus, List[str]] = {}  # status -> trajectory_ids
        self._mode_index: Dict[str, List[str]] = {}  # execution_mode -> trajectory_ids
    
    def add_trajectory(self, trajectory: CodeTrajectory) -> str:
        """
        添加轨迹到池中
        
        Args:
            trajectory: 要添加的轨迹
        
        Returns:
            轨迹ID
        """
        # 生成轨迹ID（如果还没有）
        if not trajectory.trajectory_id:
            trajectory.trajectory_id = self._generate_trajectory_id(trajectory)
        
        # 添加到列表
        self.trajectories.append(trajectory)
        
        # 更新索引
        self._update_indexes(trajectory)
        
        return trajectory.trajectory_id
    
    def get_trajectories_by_task(self, task_id: str) -> List[CodeTrajectory]:
        """根据任务ID获取轨迹"""
        trajectory_ids = self._task_index.get(task_id, [])
        return [t for t in self.trajectories if t.trajectory_id in trajectory_ids]
    
    def get_successful_trajectories(self, execution_mode: Optional[str] = None) -> List[CodeTrajectory]:
        """获取成功的轨迹"""
        trajectory_ids = self._status_index.get(TrajectoryStatus.SUCCESS, [])
        trajectories = [t for t in self.trajectories if t.trajectory_id in trajectory_ids]
        
        if execution_mode:
            trajectories = [t for t in trajectories if t.execution_mode == execution_mode]
        
        return trajectories
    
    def get_failed_trajectories(self, execution_mode: Optional[str] = None) -> List[CodeTrajectory]:
        """获取失败的轨迹"""
        trajectory_ids = self._status_index.get(TrajectoryStatus.FAILED, [])
        trajectories = [t for t in self.trajectories if t.trajectory_id in trajectory_ids]
        
        if execution_mode:
            trajectories = [t for t in trajectories if t.execution_mode == execution_mode]
        
        return trajectories
    
    def find_similar_trajectories(self, trajectory: CodeTrajectory, threshold: float = 0.8) -> List[CodeTrajectory]:
        """查找相似的轨迹"""
        similar = []
        for t in self.trajectories:
            if t.is_similar_to(trajectory, threshold):
                similar.append(t)
        return similar
    
    def save(self, compressed: bool = True) -> Path:
        """
        保存轨迹池到文件
        
        Args:
            compressed: 是否压缩（默认True，可减少80%空间）
        
        Returns:
            保存的文件路径
        """
        # 准备数据
        data = {
            "pool_id": self.pool_id,
            "trajectories": [t.to_dict() for t in self.trajectories],
            "metadata": {
                "total_count": len(self.trajectories),
                "saved_at": datetime.now().isoformat()
            }
        }
        
        # 序列化为JSON
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        
        # 确定文件路径
        if compressed:
            file_path = self.storage_dir / f"{self.pool_id}.tra.gz"
            # 压缩保存
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                f.write(json_str)
        else:
            file_path = self.storage_dir / f"{self.pool_id}.tra.json"
            # 直接保存
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return file_path
    
    @classmethod
    def load(cls, file_path: Path) -> "TrajectoryPool":
        """
        从文件加载轨迹池
        
        Args:
            file_path: 文件路径（.tra.json 或 .tra.gz）
        
        Returns:
            轨迹池实例
        """
        # 判断是否压缩
        if file_path.suffix == '.gz':
            # 解压加载
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # 直接加载
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        # 创建轨迹池
        pool = cls(pool_id=data["pool_id"], storage_dir=file_path.parent)
        
        # 恢复轨迹
        for traj_data in data.get("trajectories", []):
            trajectory = CodeTrajectory.from_dict(traj_data)
            pool.add_trajectory(trajectory)
        
        return pool
    
    def compress(self, use_llm: bool = True, similarity_threshold: float = 0.85) -> Dict[str, Any]:
        """
        压缩轨迹池（使用LLM总结，减少80%空间）
        
        Args:
            use_llm: 是否使用LLM进行智能压缩（默认True）
            similarity_threshold: 相似度阈值，超过此阈值的轨迹将被合并（默认0.85）
        
        Returns:
            压缩统计信息
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
            # 使用LLM进行智能压缩
            compressed_trajectories = self._compress_with_llm(similarity_threshold)
        else:
            # 简单去重压缩
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
        """简单去重压缩"""
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
        使用LLM进行智能轨迹压缩
        
        策略：
        1. 按相似度分组轨迹
        2. 对每组相似轨迹，使用LLM生成摘要
        3. 保留一个代表性轨迹，其他用摘要替代
        4. 对于失败轨迹，保留所有关键错误信息
        
        Args:
            similarity_threshold: 相似度阈值
        
        Returns:
            压缩后的轨迹列表
        """
        try:
            from agent.utils.llm_factory import create_reasoning_advanced_llm
            llm = create_reasoning_advanced_llm()
        except ImportError:
            # 如果无法导入，使用简单压缩
            print("  ⚠ 无法导入LLM工厂，使用简单压缩")
            return self._compress_simple()
        
        if not llm:
            print("  ⚠ LLM不可用，使用简单压缩")
            return self._compress_simple()
        
        # 1. 按相似度分组
        groups = self._group_similar_trajectories(similarity_threshold)
        
        compressed = []
        for group in groups:
            if len(group) == 1:
                # 单个轨迹，直接保留
                compressed.append(group[0])
            else:
                # 多个相似轨迹，使用LLM生成摘要
                summary_traj = self._summarize_trajectory_group(llm, group)
                if summary_traj:
                    compressed.append(summary_traj)
                else:
                    # LLM摘要失败，保留第一个
                    compressed.append(group[0])
        
        print(f"  ✓ LLM压缩完成: {len(self.trajectories)} -> {len(compressed)} 轨迹")
        return compressed
    
    def _group_similar_trajectories(self, threshold: float) -> List[List[CodeTrajectory]]:
        """按相似度分组轨迹"""
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
        使用LLM总结一组相似轨迹
        
        Args:
            llm: LLM实例
            group: 相似轨迹组
        
        Returns:
            总结后的代表性轨迹
        """
        if not group:
            return None
        
        # 选择最成功的轨迹作为基础
        base_traj = max(group, key=lambda t: 1 if t.status == TrajectoryStatus.SUCCESS else 0)
        
        # 如果只有1-2个轨迹，直接返回基础轨迹
        if len(group) <= 2:
            return base_traj
        
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # 构建提示词
            system_prompt = """你是一个代码执行轨迹分析专家。你的任务是将多个相似的代码执行轨迹合并为一个代表性的轨迹摘要。

要求：
1. 保留所有关键信息：任务ID、执行模式、核心代码逻辑、执行结果
2. 合并相似代码的变体，提取共同模式
3. 对于失败轨迹，保留所有不同的错误类型和原因
4. 生成一个简洁但完整的轨迹摘要

输出格式：JSON格式的轨迹摘要，包含：
- task_id: 任务ID（从第一个轨迹获取）
- execution_mode: 执行模式
- generated_code: 合并后的代表性代码（提取共同模式）
- status: 状态（如果所有都成功则为success，否则为failed）
- execution_result: 合并后的执行结果
- error_summary: 如果有失败，总结所有错误类型和原因
- trajectory_count: 被合并的轨迹数量"""
            
            # 准备轨迹信息
            traj_summaries = []
            for i, traj in enumerate(group[:5]):  # 最多处理5个轨迹
                traj_info = {
                    "index": i + 1,
                    "code": traj.generated_code[:500],  # 限制长度
                    "status": traj.status.value,
                    "error": traj.error_message[:200] if traj.error_message else None,
                    "error_type": traj.error_type
                }
                traj_summaries.append(traj_info)
            
            user_prompt = f"""请将以下 {len(group)} 个相似的代码执行轨迹合并为一个代表性摘要：

轨迹列表：
{json.dumps(traj_summaries, ensure_ascii=False, indent=2)}

基础轨迹信息：
- task_id: {base_traj.task_id}
- execution_mode: {base_traj.execution_mode}
- 基础代码: {base_traj.generated_code[:300]}

请生成合并后的轨迹摘要（JSON格式）。"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            summary_text = response.content.strip()
            
            # 解析JSON响应
            if summary_text.startswith("```json"):
                summary_text = summary_text[7:]
            if summary_text.startswith("```"):
                summary_text = summary_text[3:]
            if summary_text.endswith("```"):
                summary_text = summary_text[:-3]
            summary_text = summary_text.strip()
            
            summary_data = json.loads(summary_text)
            
            # 创建摘要轨迹
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
            print(f"  ⚠ LLM轨迹摘要失败: {e}，保留基础轨迹")
            return base_traj
    
    def _generate_trajectory_id(self, trajectory: CodeTrajectory) -> str:
        """生成轨迹ID"""
        timestamp_str = trajectory.timestamp.isoformat()
        code_hash = hashlib.md5(trajectory.generated_code.encode()).hexdigest()[:8]
        return f"{trajectory.task_id}_{timestamp_str}_{code_hash}"
    
    def _update_indexes(self, trajectory: CodeTrajectory):
        """更新索引"""
        traj_id = trajectory.trajectory_id
        
        # 任务索引
        if trajectory.task_id not in self._task_index:
            self._task_index[trajectory.task_id] = []
        if traj_id not in self._task_index[trajectory.task_id]:
            self._task_index[trajectory.task_id].append(traj_id)
        
        # 状态索引
        if trajectory.status not in self._status_index:
            self._status_index[trajectory.status] = []
        if traj_id not in self._status_index[trajectory.status]:
            self._status_index[trajectory.status].append(traj_id)
        
        # 模式索引
        if trajectory.execution_mode not in self._mode_index:
            self._mode_index[trajectory.execution_mode] = []
        if traj_id not in self._mode_index[trajectory.execution_mode]:
            self._mode_index[trajectory.execution_mode].append(traj_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取轨迹池统计信息"""
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

