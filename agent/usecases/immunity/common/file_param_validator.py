"""
文件参数校验和转换模块

在工具调用前对文件类参数进行校验：
1. 判断文件类型是否符合标准
2. 如果类型不符合，使用file_utils服务的工具进行类型转换
3. 如果遇到http/https文件链接，使用file_utils服务的download_to_tmp_file工具下载为本地文件
4. 保证传入工具的路径相关参数，必定是符合文件类型要求的本地文件路径
"""

import os
import re
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from pathlib import Path

from langchain_core.runnables.config import RunnableConfig
from common.util.mcp_utils import mcp_tool_async
from common.factory import get_mcp_client


class FileParamValidator:
    """文件参数校验器"""
    
    def __init__(self, config: Optional[RunnableConfig] = None):
        self.config = config
        self.file_utils_service_id = "file_utils"  # file_utils服务的ID
    
    async def validate_and_convert_file_params(
        self, 
        tool_args: Dict[str, Any], 
        tool_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        校验并转换工具参数中的文件路径
        
        Args:
            tool_args: 工具参数字典
            tool_schema: 工具的args_schema，用于识别文件类型参数
        
        Returns:
            转换后的工具参数字典
        """
        if not tool_args:
            return tool_args
        
        # 识别需要校验的文件参数（包括数组类型）
        file_params, array_file_params = self._identify_file_params(tool_args, tool_schema)
        
        print(f"[FileParamValidator] 识别到 {len(file_params)} 个单个文件参数: {list(file_params.keys())}")
        print(f"[FileParamValidator] 识别到 {len(array_file_params)} 个数组文件参数: {list(array_file_params.keys())}")
        
        # 对每个文件参数进行校验和转换
        converted_args = self._deep_copy(tool_args)
        
        # 处理单个文件参数
        for param_path, param_value in file_params.items():
            if param_value is None or param_value == "":
                continue
            
            converted_value = await self._validate_and_convert_file_param(
                param_name=param_path,
                param_value=param_value,
                param_schema=self._get_param_schema(param_path, tool_schema)
            )
            
            if converted_value != param_value:
                # 更新嵌套结构中的参数值
                self._set_nested_value(converted_args, param_path, converted_value)
                print(f"[FileParamValidator] 参数 {param_path} 已转换: {param_value} -> {converted_value}")
        
        # 处理数组文件参数（多个CSV文件合并）
        for param_path, file_array in array_file_params.items():
            if not file_array or len(file_array) < 2:
                continue
            
            # 检查是否有need_merge标识
            param_schema = self._get_param_schema(param_path, tool_schema)
            need_merge = self._check_need_merge(param_schema)
            
            if not need_merge:
                print(f"[FileParamValidator] 参数 {param_path} 未设置need_merge标识，跳过合并")
                continue
            
            # 检查是否都是CSV文件
            all_csv = all(
                isinstance(f, str) and 
                (f.lower().endswith('.csv') or f.startswith(('http://', 'https://')))
                for f in file_array
            )
            
            if all_csv:
                print(f"[FileParamValidator] 检测到数组参数 {param_path} 包含 {len(file_array)} 个CSV文件，且设置了need_merge标识，开始合并...")
                merged_file = await self._merge_csv_files(file_array, param_path)
                if merged_file:
                    # 将数组参数替换为合并后的单个文件路径
                    self._set_nested_value(converted_args, param_path, merged_file)
                    print(f"[FileParamValidator] 数组参数 {param_path} 已合并为: {merged_file}")
        
        return converted_args
    
    def _deep_copy(self, obj: Any) -> Any:
        """深拷贝对象"""
        import copy
        return copy.deepcopy(obj)
    
    def _set_nested_value(self, obj: Dict[str, Any], path: str, value: Any):
        """在嵌套字典中设置值，支持args.args等嵌套结构"""
        parts = [p for p in path.split(".") if p]  # 移除空字符串
        
        if not parts:
            return
        
        current = obj
        
        # 遍历路径，创建必要的嵌套结构
        for i, part in enumerate(parts[:-1]):
            # 处理列表索引，如args[0]
            if "[" in part and "]" in part:
                key = part.split("[")[0]
                index = int(part.split("[")[1].split("]")[0])
                
                if key not in current:
                    current[key] = []
                while len(current[key]) <= index:
                    current[key].append({})
                current = current[key][index]
            else:
                if part not in current:
                    current[part] = {}
                elif not isinstance(current[part], dict):
                    # 如果当前值不是字典，创建新字典（覆盖原值）
                    current[part] = {}
                current = current[part]
        
        # 设置最终值
        final_key = parts[-1]
        if "[" in final_key and "]" in final_key:
            key = final_key.split("[")[0]
            index = int(final_key.split("[")[1].split("]")[0])
            
            if key not in current:
                current[key] = []
            while len(current[key]) <= index:
                current[key].append(None)
            current[key][index] = value
        else:
            current[final_key] = value
    
    def _identify_file_params(
        self, 
        tool_args: Dict[str, Any], 
        tool_schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """
        识别工具参数中的文件类型参数（包括单个文件和数组）
        
        识别规则：
        1. 从tool_schema中查找带有file_input、file等关键词的字段
        2. 查找参数名包含file、path等关键词的字段
        3. 检查参数值是否为字符串且可能是文件路径或URL
        4. 检查参数值是否为数组，包含多个文件路径
        
        Returns:
            (file_params, array_file_params): 
            - file_params: 单个文件参数的字典 {param_path: file_path}
            - array_file_params: 数组文件参数的字典 {param_path: [file_path1, file_path2, ...]}
        """
        file_params = {}
        array_file_params = {}
        
        # 递归查找嵌套结构中的文件参数
        def find_file_params(obj: Any, prefix: str = "") -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
            single_files = {}
            array_files = {}
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    
                    # 检查是否为数组类型的文件参数
                    if isinstance(value, list) and len(value) > 0:
                        # 检查数组中的元素是否都是文件路径
                        file_paths = []
                        for item in value:
                            if isinstance(item, str) and item.strip():
                                # 检查是否为文件路径或URL
                                if (item.startswith(('http://', 'https://')) or 
                                    os.sep in item or 
                                    any(item.lower().endswith(f".{ext}") for ext in 
                                        ["csv", "rds", "fasta", "fa", "pdb", "xlsx", "xls", "json", "txt", "tsv"])):
                                    file_paths.append(item)
                        
                        if len(file_paths) >= 2:  # 至少2个文件才需要合并
                            # 检查schema确认这是文件参数，并且有need_merge标识
                            is_file_param = self._is_file_param(key, value, tool_schema, full_key)
                            if is_file_param:
                                # 检查是否有need_merge标识
                                param_schema = self._get_param_schema(full_key, tool_schema)
                                need_merge = self._check_need_merge(param_schema)
                                if need_merge:
                                    array_files[full_key] = file_paths
                        elif len(file_paths) == 1:
                            # 单个文件，按单个文件参数处理
                            single_files[full_key] = file_paths[0]
                    elif isinstance(value, str) and value.strip():
                        # 检查schema中的字段类型
                        is_file_param = self._is_file_param(key, value, tool_schema, full_key)
                        if is_file_param:
                            single_files[full_key] = value
                    elif isinstance(value, (dict, list)):
                        # 递归查找嵌套结构
                        nested_single, nested_array = find_file_params(value, full_key)
                        single_files.update(nested_single)
                        array_files.update(nested_array)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    nested_single, nested_array = find_file_params(item, f"{prefix}[{i}]")
                    single_files.update(nested_single)
                    array_files.update(nested_array)
            
            return single_files, array_files
        
        file_params, array_file_params = find_file_params(tool_args)
        
        return file_params, array_file_params
    
    def _is_file_param(
        self, 
        param_name: str, 
        param_value: Any, 
        tool_schema: Optional[Dict[str, Any]], 
        full_key: str
    ) -> bool:
        """判断参数是否为文件参数"""
        
        # 首先检查参数值本身是否为URL或文件路径（最直接的方式）
        if isinstance(param_value, str) and param_value.strip():
            value_lower = param_value.lower().strip()
            # 检查是否为URL
            if value_lower.startswith(("http://", "https://")):
                print(f"[FileParamValidator] 参数 {full_key} 通过URL识别为文件参数: {param_value[:50]}...")
                return True
            # 检查是否为文件路径（包含路径分隔符或常见文件扩展名）
            if os.sep in param_value or any(value_lower.endswith(f".{ext}") for ext in 
                ["csv", "rds", "fasta", "fa", "pdb", "xlsx", "xls", "json", "txt", "tsv"]):
                print(f"[FileParamValidator] 参数 {full_key} 通过文件路径识别为文件参数: {param_value[:50]}...")
                return True
        
        # 然后检查schema中的字段定义（优先级次之）
        if tool_schema:
            param_schema = self._get_param_schema(full_key, tool_schema)
            if param_schema:
                # 检查json_schema_extra中的定义
                json_schema_extra = param_schema.get("json_schema_extra", {})
                if isinstance(json_schema_extra, dict):
                    ui_type = json_schema_extra.get("ui_type")
                    support_file_types = json_schema_extra.get("support_file_types")
                    
                    if ui_type in ["file_input", "file"]:
                        print(f"[FileParamValidator] 参数 {full_key} 通过schema识别为文件参数 (ui_type={ui_type})")
                        return True
                    if support_file_types:
                        print(f"[FileParamValidator] 参数 {full_key} 通过schema识别为文件参数 (support_file_types={support_file_types})")
                        return True
                
                # 检查schema顶层
                ui_type = param_schema.get("ui_type")
                support_file_types = param_schema.get("support_file_types")
                
                if ui_type in ["file_input", "file"]:
                    print(f"[FileParamValidator] 参数 {full_key} 通过schema识别为文件参数 (ui_type={ui_type})")
                    return True
                if support_file_types:
                    print(f"[FileParamValidator] 参数 {full_key} 通过schema识别为文件参数 (support_file_types={support_file_types})")
                    return True
        
        # 最后检查参数名是否包含文件相关关键词
        file_keywords = ["file", "path", "input_file", "output_file", "rds_file", 
                        "csv_file", "fasta", "pdb", "excel", "xlsx"]
        param_name_lower = param_name.lower()
        
        if any(keyword in param_name_lower for keyword in file_keywords):
            print(f"[FileParamValidator] 参数 {full_key} 通过参数名识别为文件参数")
            return True
        
        print(f"[FileParamValidator] 参数 {full_key} 未识别为文件参数")
        return False
    
    def _get_param_schema(self, param_name: str, tool_schema: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """从tool_schema中获取参数的schema定义"""
        if not tool_schema:
            return None
        
        # 处理嵌套结构（如args.args结构）
        parts = [p for p in param_name.split(".") if p]  # 移除空字符串
        schema = tool_schema
        
        for part in parts:
            if isinstance(schema, dict):
                # 首先检查properties字段
                if "properties" in schema:
                    schema = schema["properties"]
                
                # 检查当前部分是否在schema中
                if part in schema:
                    part_schema = schema[part]
                    
                    # 检查是否有$ref引用
                    if isinstance(part_schema, dict) and "$ref" in part_schema:
                        ref_path = part_schema["$ref"]
                        if ref_path.startswith("#/$defs/"):
                            def_name = ref_path.replace("#/$defs/", "")
                            # 从tool_schema的顶层获取$defs
                            if "$defs" in tool_schema and def_name in tool_schema["$defs"]:
                                schema = tool_schema["$defs"][def_name]
                                # 如果解析后的schema有properties，继续展开
                                if isinstance(schema, dict) and "properties" in schema:
                                    schema = schema["properties"]
                                continue
                            else:
                                return None
                        else:
                            return None
                    else:
                        # 没有$ref，直接使用
                        schema = part_schema
                else:
                    return None
            else:
                return None
        
        return schema if isinstance(schema, dict) else None
    
    def _check_need_merge(self, param_schema: Optional[Dict[str, Any]]) -> bool:
        """
        检查参数schema中是否有need_merge标识
        
        Args:
            param_schema: 参数的schema定义
        
        Returns:
            如果有need_merge标识且为True，返回True；否则返回False
        """
        if not param_schema:
            return False
        
        # 检查json_schema_extra中的need_merge
        json_schema_extra = param_schema.get("json_schema_extra", {})
        if isinstance(json_schema_extra, dict):
            need_merge = json_schema_extra.get("need_merge")
            if need_merge is True or (isinstance(need_merge, str) and need_merge.lower() in ["true", "yes", "1"]):
                return True
        
        # 检查schema顶层是否有need_merge
        need_merge = param_schema.get("need_merge")
        if need_merge is True or (isinstance(need_merge, str) and need_merge.lower() in ["true", "yes", "1"]):
            return True
        
        return False
    
    async def _validate_and_convert_file_param(
        self,
        param_name: str,
        param_value: str,
        param_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        校验并转换单个文件参数
        
        Returns:
            转换后的文件路径（本地路径）
        """
        param_value = param_value.strip()
        
        # 步骤1: 如果是HTTP/HTTPS URL，先下载为本地文件
        if param_value.startswith(("http://", "https://")):
            print(f"[FileParamValidator] 检测到URL: {param_value}，开始下载...")
            downloaded_path = await self._download_file_from_url(param_value)
            if downloaded_path:
                param_value = downloaded_path
            else:
                print(f"[FileParamValidator] 警告：URL下载失败，保持原值: {param_value}")
                return param_value
        
        # 步骤3: 获取期望的文件类型（从schema中）
        expected_types = self._get_expected_file_types(param_name, param_schema)
        
        # 步骤4: 检查文件类型是否符合要求
        actual_type = self._get_file_type(param_value)
        
        if expected_types:
            if actual_type not in expected_types:
                print(f"[FileParamValidator] 文件类型不匹配: 期望 {expected_types}，实际 {actual_type}，开始转换...")
                converted_path = await self._convert_file_type(param_value, expected_types[0])
                if converted_path:
                    print(f"[FileParamValidator] 文件类型转换成功: {param_value} -> {converted_path}")
                    return converted_path
                else:
                    print(f"[FileParamValidator] 警告：文件类型转换失败，保持原值: {param_value}")
            else:
                print(f"[FileParamValidator] 文件类型匹配: {actual_type}")
        else:
            print(f"[FileParamValidator] 未找到期望的文件类型定义，跳过类型检查")
        
        return param_value
    
    async def _download_file_from_url(self, url: str) -> Optional[str]:
        """使用file_utils服务的download_url工具下载文件"""
        try:
            print(f"[FileParamValidator] 调用file_utils.download_url下载: {url}")
            result = await mcp_tool_async(
                service_id=self.file_utils_service_id,
                tool_name="download_url",
                params={"args": {"url": url}}
            )
            
            # 解析结果，获取本地文件路径
            file_path = None
            result_dict = None
            
            # 处理字符串类型的结果（可能是JSON字符串）
            if isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                except json.JSONDecodeError:
                    # 如果不是JSON，可能是直接返回的路径
                    return result
            # 处理字典类型的结果
            elif isinstance(result, dict):
                result_dict = result
            
            # 从解析后的字典中提取文件路径
            if result_dict:
                file_path = (
                    result_dict.get("file_path") or 
                    result_dict.get("path") or 
                    result_dict.get("output_path") or
                    result_dict.get("output_file")
                )
                if file_path:
                    print(f"[FileParamValidator] 文件下载成功: {file_path}")
                    return file_path
            
            print(f"[FileParamValidator] 下载结果格式异常，无法提取文件路径")
            return None
            
        except Exception as e:
            print(f"[FileParamValidator] 下载文件失败: {e}")
            return None
    
    def _get_expected_file_types(
        self, 
        param_name: str, 
        param_schema: Optional[Dict[str, Any]]
    ) -> List[str]:
        """从schema中获取期望的文件类型"""
        if not param_schema:
            return []
        
        # 首先从json_schema_extra中获取support_file_types（优先级最高）
        json_schema_extra = param_schema.get("json_schema_extra", {})
        if isinstance(json_schema_extra, dict):
            support_types = json_schema_extra.get("support_file_types")
            if support_types:
                if isinstance(support_types, list):
                    return [ext.lower().lstrip(".") for ext in support_types]
                elif isinstance(support_types, str):
                    return [support_types.lower().lstrip(".")]
        
        # 然后从schema顶层获取support_file_types（优先级次之）
        support_types = param_schema.get("support_file_types")
        if support_types:
            if isinstance(support_types, list):
                return [ext.lower().lstrip(".") for ext in support_types]
            elif isinstance(support_types, str):
                return [support_types.lower().lstrip(".")]
        
        # 处理anyOf结构：如果schema有anyOf，需要检查每个选项（优先级最低）
        if "anyOf" in param_schema:
            for option in param_schema["anyOf"]:
                if isinstance(option, dict):
                    # 递归检查每个选项
                    types = self._get_expected_file_types(param_name, option)
                    if types:
                        return types
        
        # 从参数名推断（作为后备方案）
        param_lower = param_name.lower()
        if "rds" in param_lower:
            return ["rds"]
        elif "csv" in param_lower:
            return ["csv"]
        elif "fasta" in param_lower or "fa" in param_lower:
            return ["fasta", "fa"]
        elif "pdb" in param_lower:
            return ["pdb"]
        elif "xlsx" in param_lower or "excel" in param_lower:
            return ["xlsx", "xls"]
        
        return []
    
    def _get_file_type(self, file_path: str) -> str:
        """获取文件的扩展名（不含点）"""
        ext = Path(file_path).suffix.lstrip(".")
        return ext.lower() if ext else ""
    
    async def _convert_file_type(
        self, 
        file_path: str, 
        target_type: str
    ) -> Optional[str]:
        """使用file_utils服务的工具转换文件类型"""
        try:
            source_type = self._get_file_type(file_path)
            print(f"[FileParamValidator] 尝试转换文件类型: {source_type} -> {target_type}")
            
            # 根据源类型和目标类型选择合适的转换工具
            # 例如：convert_csv_to_rds, convert_xlsx_to_csv等
            
            # 构造转换工具名称
            convert_tool_name = f"convert_{source_type}_to_{target_type}"
            
            result = await mcp_tool_async(
                service_id=self.file_utils_service_id,
                tool_name=convert_tool_name,
                params={
                    "args": {
                        "input_file": file_path,
                        "output_file": None  # 让服务自动生成输出路径
                    }
                }
            )
            
            # 解析结果，获取转换后的文件路径
            # 不检查文件是否存在，由MCP工具自己完成
            if isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                    output_path = result_dict.get("output_file") or result_dict.get("file_path") or result_dict.get("path")
                    if output_path:
                        print(f"[FileParamValidator] 文件类型转换成功: {output_path}")
                        return output_path
                except json.JSONDecodeError:
                    # 如果不是JSON，可能是直接返回的路径
                    print(f"[FileParamValidator] 文件类型转换成功: {result}")
                    return result
            elif isinstance(result, dict):
                output_path = result.get("output_file") or result.get("file_path") or result.get("path")
                if output_path:
                    print(f"[FileParamValidator] 文件类型转换成功: {output_path}")
                    return output_path
            
            print(f"[FileParamValidator] 文件类型转换结果格式异常: {result}")
            return None
            
        except Exception as e:
            print(f"[FileParamValidator] 文件类型转换失败: {e}")
            return None
    
    async def _merge_csv_files(
        self, 
        file_paths: List[str], 
        param_path: str
    ) -> Optional[str]:
        """
        合并多个CSV文件
        
        如果文件数量为2，使用merge_csv_cartesian进行笛卡尔积合并
        如果文件数量大于2，先合并前两个，然后递归合并结果与后续文件
        
        Args:
            file_paths: CSV文件路径列表
            param_path: 参数路径（用于日志）
        
        Returns:
            合并后的文件路径，失败返回None
        """
        if not file_paths or len(file_paths) < 2:
            return None
        
        try:
            # 先确保所有文件都是本地文件（下载URL）
            # 不检查文件是否存在，由MCP工具自己完成
            local_files = []
            for file_path in file_paths:
                if file_path.startswith(("http://", "https://")):
                    downloaded = await self._download_file_from_url(file_path)
                    if downloaded:
                        local_files.append(downloaded)
                    else:
                        print(f"[FileParamValidator] 警告：无法下载文件 {file_path}，跳过")
                        return None
                else:
                    # 直接添加路径，不检查文件是否存在
                    local_files.append(file_path)
            
            if len(local_files) < 2:
                return None
            
            # 如果只有2个文件，直接使用merge_csv_cartesian
            if len(local_files) == 2:
                return await self._merge_two_csv_files(local_files[0], local_files[1])
            
            # 如果超过2个文件，递归合并
            # 先合并前两个
            merged = await self._merge_two_csv_files(local_files[0], local_files[1])
            if not merged:
                return None
            
            # 然后与后续文件依次合并
            for next_file in local_files[2:]:
                merged = await self._merge_two_csv_files(merged, next_file)
                if not merged:
                    return None
            
            return merged
            
        except Exception as e:
            print(f"[FileParamValidator] 合并CSV文件失败: {e}")
            import traceback
            print(f"[FileParamValidator] 详细错误: {traceback.format_exc()}")
            return None
    
    async def _merge_two_csv_files(
        self, 
        antibody_file: str, 
        antigen_file: str
    ) -> Optional[str]:
        """
        使用merge_csv_cartesian合并两个CSV文件
        
        Args:
            antibody_file: 第一个CSV文件路径（作为antibody_file）
            antigen_file: 第二个CSV文件路径（作为antigen_file）
        
        Returns:
            合并后的文件路径，失败返回None
        """
        try:
            print(f"[FileParamValidator] 调用merge_csv_cartesian合并文件: {antibody_file} + {antigen_file}")
            
            result = await mcp_tool_async(
                service_id=self.file_utils_service_id,
                tool_name="merge_csv_cartesian",
                params={
                    "args": {
                        "antibody_file": antibody_file,
                        "antigen_file": antigen_file,
                        "antibody_key": None,  # 使用默认值
                        "antigen_key": None,   # 使用默认值
                        "new_key_name": "merged_key",
                        "output_file": None    # 让服务自动生成输出路径
                    }
                }
            )
            
            # 解析结果，获取合并后的文件路径
            # 不检查文件是否存在，由MCP工具自己完成
            if isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                    output_path = result_dict.get("output_file") or result_dict.get("file_path") or result_dict.get("path")
                    if output_path:
                        print(f"[FileParamValidator] CSV文件合并成功: {output_path}")
                        return output_path
                except json.JSONDecodeError:
                    # 如果不是JSON，可能是直接返回的路径
                    print(f"[FileParamValidator] CSV文件合并成功: {result}")
                    return result
            elif isinstance(result, dict):
                output_path = result.get("output_file") or result.get("file_path") or result.get("path")
                if output_path:
                    print(f"[FileParamValidator] CSV文件合并成功: {output_path}")
                    return output_path
            
            print(f"[FileParamValidator] CSV文件合并结果格式异常: {result}")
            return None
            
        except Exception as e:
            print(f"[FileParamValidator] CSV文件合并失败: {e}")
            import traceback
            print(f"[FileParamValidator] 详细错误: {traceback.format_exc()}")
            return None


async def validate_file_params_for_tool(
    tool_args: Dict[str, Any],
    tool_schema: Optional[Dict[str, Any]] = None,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    便捷函数：校验并转换工具参数中的文件路径
    
    Args:
        tool_args: 工具参数字典
        tool_schema: 工具的args_schema
        config: RunnableConfig配置
    
    Returns:
        转换后的工具参数字典
    """
    validator = FileParamValidator(config=config)
    return await validator.validate_and_convert_file_params(tool_args, tool_schema)

