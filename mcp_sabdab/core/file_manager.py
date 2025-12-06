"""
SAbDab文件管理器模块
提供文件保存、路径管理和基本的文件操作功能
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any


class SAbDabFileManager:
    """SAbDab文件管理器，负责处理文件的保存和管理"""
    
    def __init__(self, base_dir: str = "sabdab_data"):
        """
        初始化文件管理器
        
        Args:
            base_dir: 基础存储目录，默认为"sabdab_data"
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
    
    def _generate_timestamp(self) -> str:
        """生成时间戳字符串，格式：YYYYMMDD_HHMMSS"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _get_file_size(self, file_path: Path) -> int:
        """获取文件大小（字节）"""
        try:
            return file_path.stat().st_size
        except OSError:
            return 0
    
    def save_csv_file(self, content: str, filename_prefix: str = "sabdab_summary") -> Dict[str, Any]:
        """
        保存CSV文件
        
        Args:
            content: CSV内容字符串
            filename_prefix: 文件名前缀
            
        Returns:
            包含文件信息的字典
        """
        timestamp = self._generate_timestamp()
        filename = f"{filename_prefix}_{timestamp}.csv"
        file_path = self.base_dir / filename
        
        # 保存文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 返回文件信息
        return {
            "file_path": str(file_path.absolute()),
            "file_size_bytes": self._get_file_size(file_path),
            "created_at": datetime.now().isoformat() + "Z"
        }
    
    def save_pdb_file(self, content: str, pdb_id: str, scheme: str = "imgt") -> Dict[str, Any]:
        """
        保存PDB文件
        
        Args:
            content: PDB内容字符串
            pdb_id: PDB ID
            scheme: 编号方案
            
        Returns:
            包含文件信息的字典
        """
        timestamp = self._generate_timestamp()
        filename = f"{pdb_id}_{scheme}_{timestamp}.pdb"
        file_path = self.base_dir / filename
        
        # 保存文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 返回文件信息
        return {
            "file_path": str(file_path.absolute()),
            "file_size_bytes": self._get_file_size(file_path),
            "created_at": datetime.now().isoformat() + "Z"
        }
    
    def save_dataset_file(self, content: str, data_type: str, format_type: str) -> Dict[str, Any]:
        """
        保存数据集文件
        
        Args:
            content: 数据集内容字符串
            data_type: 数据类型（如：all, antigen_bound, nanobodies）
            format_type: 格式类型（如：csv, json, fasta）
            
        Returns:
            包含文件信息的字典
        """
        timestamp = self._generate_timestamp()
        filename = f"sabdab_dataset_{data_type}_{timestamp}.{format_type}"
        file_path = self.base_dir / filename
        
        # 保存文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 返回文件信息
        return {
            "file_path": str(file_path.absolute()),
            "file_size_bytes": self._get_file_size(file_path),
            "created_at": datetime.now().isoformat() + "Z"
        }
    
    def save_json_file(self, content: str, filename_prefix: str = "sabdab_stats") -> Dict[str, Any]:
        """
        保存JSON文件
        
        Args:
            content: JSON内容字符串
            filename_prefix: 文件名前缀
            
        Returns:
            包含文件信息的字典
        """
        timestamp = self._generate_timestamp()
        filename = f"{filename_prefix}_{timestamp}.json"
        file_path = self.base_dir / filename
        
        # 保存文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 返回文件信息
        return {
            "file_path": str(file_path.absolute()),
            "file_size_bytes": self._get_file_size(file_path),
            "created_at": datetime.now().isoformat() + "Z"
        }
    
    def cleanup_old_files(self, days: int = 7) -> int:
        """
        清理指定天数之前的旧文件
        
        Args:
            days: 保留天数，默认7天
            
        Returns:
            删除的文件数量
        """
        if not self.base_dir.exists():
            return 0
        
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted_count = 0
        
        for file_path in self.base_dir.iterdir():
            if file_path.is_file():
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
                except OSError:
                    # 忽略删除失败的文件
                    continue
        
        return deleted_count


# 创建全局文件管理器实例
file_manager = SAbDabFileManager()