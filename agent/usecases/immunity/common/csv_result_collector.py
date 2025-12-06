"""
CSV结果收集器
用于在任务执行过程中收集和合并所有工具产生的CSV/Excel文件
"""
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from common.util.mcp_utils import mcp_tool_async


class CSVResultCollector:
    """CSV结果收集器，用于收集和合并任务执行过程中的CSV/Excel文件"""
    
    def __init__(self, file_utils_service_id: str = "file_utils"):
        """
        初始化CSV结果收集器
        
        Args:
            file_utils_service_id: file_utils MCP服务的ID
        """
        self.file_utils_service_id = file_utils_service_id
        self.merged_csv_path: Optional[str] = None
        self._initialized = False
    
    async def initialize(self) -> str:
        """
        初始化收集器，创建初始CSV文件
        
        Returns:
            初始CSV文件路径
        """
        if self._initialized:
            return self.merged_csv_path
        
        try:
            # 调用file_utils的create_csv工具创建初始CSV文件
            result = await mcp_tool_async(
                service_id=self.file_utils_service_id,
                tool_name="create_csv",
                params={"args": {}}
            )
            
            # 解析结果，获取CSV文件路径
            csv_path = self._extract_file_path(result)
            if csv_path:
                self.merged_csv_path = csv_path
                self._initialized = True
                print(f"[CSVResultCollector] 初始化成功，初始CSV文件: {csv_path}")
                return csv_path
            else:
                print(f"[CSVResultCollector] 警告：无法从create_csv结果中提取文件路径")
                return None
        except Exception as e:
            print(f"[CSVResultCollector] 初始化失败: {e}")
            import traceback
            print(f"[CSVResultCollector] 详细错误: {traceback.format_exc()}")
            return None
    
    def _extract_file_path(self, result: Any) -> Optional[str]:
        """
        从工具执行结果中提取文件路径
        
        Args:
            result: 工具执行结果（可能是字符串、字典等）
            
        Returns:
            文件路径，如果提取失败则返回None
        """
        if isinstance(result, str):
            # 尝试解析JSON字符串
            try:
                result_dict = json.loads(result)
                return self._extract_file_path(result_dict)
            except json.JSONDecodeError:
                # 如果不是JSON，检查是否是文件路径
                if Path(result).suffix in ['.csv', '.xlsx', '.xls']:
                    return result
                return None
        elif isinstance(result, dict):
            # 从字典中提取文件路径
            file_path = (
                result.get("file_path") or
                result.get("path") or
                result.get("output_path") or
                result.get("output_file") or
                result.get("csv_path")
            )
            if file_path:
                return file_path
            
            # 递归查找嵌套字典中的文件路径
            for value in result.values():
                if isinstance(value, dict):
                    nested_path = self._extract_file_path(value)
                    if nested_path:
                        return nested_path
        elif hasattr(result, 'file_path'):
            return getattr(result, 'file_path')
        
        return None
    
    def _is_csv_or_excel_file(self, file_path: str) -> bool:
        """
        检查文件路径是否是CSV或Excel文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            如果是CSV或Excel文件则返回True
        """
        if not file_path:
            return False
        
        # 检查文件扩展名
        path = Path(file_path)
        ext = path.suffix.lower()
        return ext in ['.csv', '.xlsx', '.xls']
    
    def _extract_file_paths_from_result(self, result: Any) -> List[str]:
        """
        从工具执行结果中提取所有可能的CSV/Excel文件路径
        
        Args:
            result: 工具执行结果
            
        Returns:
            文件路径列表
        """
        file_paths = []
        
        if isinstance(result, str):
            # 处理分号分隔的多个文件路径
            if ";" in result:
                parts = [p.strip() for p in result.split(";") if p.strip()]
                for part in parts:
                    # 递归处理每个部分
                    file_paths.extend(self._extract_file_paths_from_result(part))
            else:
                # 尝试从字符串中提取文件路径
                # 查找类似路径的模式
                path_pattern = r'(?:file_path|path|output_path|output_file|csv_path)[\s:=]+([^\s,\n\)]+\.(?:csv|xlsx|xls))'
                matches = re.findall(path_pattern, result, re.IGNORECASE)
                file_paths.extend(matches)
                
                # 尝试解析JSON字符串
                try:
                    result_dict = json.loads(result)
                    file_paths.extend(self._extract_file_paths_from_result(result_dict))
                except json.JSONDecodeError:
                    pass
                    
                # 直接检查是否是文件路径
                if self._is_csv_or_excel_file(result):
                    file_paths.append(result)
                    
        elif isinstance(result, dict):
            # 从字典中提取所有可能的文件路径
            # 优先检查常见的输出文件字段
            for key in ["output_file", "output_path", "file_path", "csv_path", "result_file"]:
                if key in result and isinstance(result[key], str):
                    value = result[key]
                    # 处理分号分隔的多个路径
                    if ";" in value:
                        parts = [p.strip() for p in value.split(";") if p.strip()]
                        file_paths.extend([p for p in parts if self._is_csv_or_excel_file(p)])
                    elif self._is_csv_or_excel_file(value):
                        file_paths.append(value)
            
            # 递归检查其他字段
            for key, value in result.items():
                if key not in ["output_file", "output_path", "file_path", "csv_path", "result_file"]:
                    if isinstance(value, str) and self._is_csv_or_excel_file(value):
                        file_paths.append(value)
                    elif isinstance(value, (dict, list)):
                        file_paths.extend(self._extract_file_paths_from_result(value))
        elif isinstance(result, list):
            # 从列表中提取文件路径
            for item in result:
                file_paths.extend(self._extract_file_paths_from_result(item))
        
        # 去重并返回，过滤出有效的文件路径
        valid_paths = [p for p in file_paths if p and self._is_csv_or_excel_file(p)]
        return list(set(valid_paths))
    
    async def collect_tool_output(self, tool_result: Dict[str, Any]) -> Optional[str]:
        """
        收集工具执行产生的CSV/Excel文件并合并
        
        Args:
            tool_result: 工具执行结果，包含tool_name、result等字段
            
        Returns:
            合并后的CSV文件路径，如果未产生文件或合并失败则返回None
        """
        # 如果未初始化，尝试自动初始化
        if not self._initialized:
            print(f"[CSVResultCollector] 收集器未初始化，尝试自动初始化...")
            try:
                csv_path = await self.initialize()
                if not self._initialized or not csv_path:
                    print(f"[CSVResultCollector] 警告：自动初始化失败（可能是file_utils服务不可用），跳过CSV收集")
                    print(f"[CSVResultCollector] 提示：如果需要CSV结果收集功能，请确保file_utils MCP服务正常运行")
                    return None
            except Exception as init_err:
                print(f"[CSVResultCollector] 警告：自动初始化异常: {init_err}，跳过CSV收集")
                return None
        
        try:
            # 从工具结果中提取文件路径
            result_content = tool_result.get("result", "")
            print(f"[CSVResultCollector] 开始收集工具输出，工具名称: {tool_result.get('tool_name')}")
            print(f"[CSVResultCollector] 工具结果类型: {type(result_content)}, 内容预览: {str(result_content)[:200]}")
            
            # 优先从工具结果的字典中提取output_file等字段（如果result是字典）
            file_paths = []
            if isinstance(result_content, dict):
                print(f"[CSVResultCollector] 工具结果是字典类型，尝试提取文件路径字段...")
                # 直接从字典中提取常见的输出文件字段
                for key in ["output_file", "output_path", "file_path", "csv_path", "result_file"]:
                    if key in result_content:
                        value = result_content[key]
                        print(f"[CSVResultCollector] 找到字段 {key}: {value}")
                        if isinstance(value, str):
                            path = value
                            # 处理分号分隔的多个文件路径
                            if ";" in path:
                                paths = [p.strip() for p in path.split(";") if p.strip()]
                                file_paths.extend([p for p in paths if self._is_csv_or_excel_file(p)])
                                print(f"[CSVResultCollector] 分号分隔路径，拆分为: {paths}")
                            elif self._is_csv_or_excel_file(path):
                                file_paths.append(path)
                                print(f"[CSVResultCollector] 添加文件路径: {path}")
            
            # 如果result是字符串，尝试提取路径
            elif isinstance(result_content, str):
                # 处理分号分隔的多个文件路径
                if ";" in result_content:
                    paths = [p.strip() for p in result_content.split(";") if p.strip()]
                    file_paths.extend([p for p in paths if self._is_csv_or_excel_file(p)])
                else:
                    file_paths.extend(self._extract_file_paths_from_result(result_content))
            
            # 如果还没有找到文件路径，尝试递归提取
            if not file_paths:
                file_paths = self._extract_file_paths_from_result(result_content)
            
            # 也检查工具参数中是否有输出文件路径
            tool_args = tool_result.get("args", {})
            if isinstance(tool_args, dict):
                for key in ["output_file", "output_path", "base_dir"]:
                    if key in tool_args and isinstance(tool_args[key], str):
                        path = tool_args[key]
                        if self._is_csv_or_excel_file(path):
                            file_paths.append(path)
            
            # 如果没有找到，尝试从result中递归提取
            if not file_paths:
                file_paths = self._extract_file_paths_from_result(result_content)
            
            if not file_paths:
                print(f"[CSVResultCollector] 工具 {tool_result.get('tool_name')} 未产生CSV/Excel文件")
                print(f"[CSVResultCollector] 工具结果类型: {type(result_content)}")
                print(f"[CSVResultCollector] 工具结果内容: {str(result_content)[:500]}")
                return None
            
            # 去重并过滤出有效的文件路径
            file_paths = list(set([p for p in file_paths if p and self._is_csv_or_excel_file(p)]))
            print(f"[CSVResultCollector] 工具 {tool_result.get('tool_name')} 产生了 {len(file_paths)} 个文件: {file_paths}")
            
            # 处理每个文件
            last_merged_path = None
            for file_path in file_paths:
                merged_path = await self._process_and_merge_file(file_path)
                if merged_path:
                    last_merged_path = merged_path
            
            # 返回最后一次合并后的路径
            return last_merged_path if last_merged_path else self.merged_csv_path
            
        except Exception as e:
            print(f"[CSVResultCollector] 收集工具输出失败: {e}")
            import traceback
            print(f"[CSVResultCollector] 详细错误: {traceback.format_exc()}")
            return None
    
    async def _process_and_merge_file(self, file_path: str) -> Optional[str]:
        """
        处理单个文件（如果是Excel则转换为CSV）并合并到累积CSV中
        
        Args:
            file_path: 文件路径
            
        Returns:
            合并后的CSV文件路径，如果处理失败则返回None
        """
        try:
            path = Path(file_path)
            ext = path.suffix.lower()
            
            csv_file_path = file_path
            
            # 如果是Excel文件，先转换为CSV
            if ext in ['.xlsx', '.xls']:
                print(f"[CSVResultCollector] 检测到Excel文件，开始转换为CSV: {file_path}")
                csv_file_path = await self._convert_excel_to_csv(file_path)
                if not csv_file_path:
                    print(f"[CSVResultCollector] Excel转CSV失败: {file_path}")
                    return None
            
            # 合并CSV文件
            if csv_file_path:
                merged_path = await self._merge_csv_files(csv_file_path)
                return merged_path if merged_path else self.merged_csv_path
            
            return None
            
        except Exception as e:
            print(f"[CSVResultCollector] 处理文件失败 {file_path}: {e}")
            import traceback
            print(f"[CSVResultCollector] 详细错误: {traceback.format_exc()}")
            return None
    
    async def _convert_excel_to_csv(self, excel_file_path: str) -> Optional[str]:
        """
        将Excel文件转换为CSV
        
        Args:
            excel_file_path: Excel文件路径
            
        Returns:
            转换后的CSV文件路径
        """
        try:
            # 根据文件扩展名选择正确的工具名
            path = Path(excel_file_path)
            ext = path.suffix.lower()
            if ext == '.xlsx':
                tool_name = "convert_xlsx_to_csv"
            elif ext == '.xls':
                tool_name = "convert_xls_to_csv"
            else:
                # 默认尝试 xlsx
                tool_name = "convert_xlsx_to_csv"
            
            print(f"[CSVResultCollector] 调用 {tool_name} 转换Excel文件: {excel_file_path}")
            result = await mcp_tool_async(
                service_id=self.file_utils_service_id,
                tool_name=tool_name,
                params={
                    "args": {
                        "input_file": excel_file_path,
                        "output_file": None  # 让服务自动生成
                    }
                }
            )
            
            print(f"[CSVResultCollector] 转换工具返回结果类型: {type(result)}, 内容: {str(result)[:500]}")
            csv_path = self._extract_file_path(result)
            if csv_path:
                print(f"[CSVResultCollector] Excel转CSV成功: {excel_file_path} -> {csv_path}")
                return csv_path
            else:
                print(f"[CSVResultCollector] 无法从转换结果中提取CSV路径")
                print(f"[CSVResultCollector] 转换结果完整内容: {result}")
                return None
                
        except Exception as e:
            print(f"[CSVResultCollector] Excel转CSV失败: {e}")
            import traceback
            print(f"[CSVResultCollector] 详细错误: {traceback.format_exc()}")
            return None
    
    async def _merge_csv_files(self, new_csv_file: str) -> Optional[str]:
        """
        将新的CSV文件合并到累积的CSV文件中
        
        Args:
            new_csv_file: 新的CSV文件路径
            
        Returns:
            合并后的CSV文件路径，如果合并失败则返回None
        """
        if not self.merged_csv_path:
            print(f"[CSVResultCollector] 累积CSV文件路径为空，使用新文件作为初始文件: {new_csv_file}")
            self.merged_csv_path = new_csv_file
            return new_csv_file
        
        # 始终执行合并，即使初始文件是空的也要合并（用户要求）
        
        try:
            # 调用file_utils的merge_csv_by_key工具合并CSV文件
            result = await mcp_tool_async(
                service_id=self.file_utils_service_id,
                tool_name="merge_csv_by_key",
                params={
                    "args": {
                        "input_file1": self.merged_csv_path,
                        "input_file2": new_csv_file,
                        "key_column": None,  # 使用行索引合并（row-by-row）
                        "output_file": None  # 让服务自动生成
                    }
                }
            )
            
            print(f"[CSVResultCollector] 合并工具返回结果类型: {type(result)}, 内容: {str(result)[:500]}")
            merged_path = self._extract_file_path(result)
            if merged_path:
                print(f"[CSVResultCollector] CSV合并成功: {self.merged_csv_path} + {new_csv_file} -> {merged_path}")
                self.merged_csv_path = merged_path
                return merged_path
            else:
                print(f"[CSVResultCollector] 无法从合并结果中提取文件路径")
                print(f"[CSVResultCollector] 合并结果完整内容: {result}")
                return None
                
        except Exception as e:
            print(f"[CSVResultCollector] CSV合并失败: {e}")
            import traceback
            print(f"[CSVResultCollector] 详细错误: {traceback.format_exc()}")
            return None
    
    def get_merged_csv_path(self) -> Optional[str]:
        """
        获取最终合并的CSV文件路径
        
        重要说明：
        - 此路径仅用于记录和展示结果，不会作为工具参数传递给后续工具
        - 每个工具应该使用自己的原始输入文件，而不是合并后的文件
        - 合并后的CSV路径不会被添加到Agent的消息历史中，因此不会影响Agent的决策
        - 此路径仅在任务执行完成后提供给用户查看，不会影响工具执行流程
        
        Returns:
            合并后的CSV文件路径，仅用于结果记录和展示，不会作为工具参数传递
        """
        return self.merged_csv_path

