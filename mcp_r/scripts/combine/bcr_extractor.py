import pandas as pd
import json
import traceback
from typing import Dict, List, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field
from common.constants import REASONING_MODEL, QWEN_MODEL_VLLM, QWEN_BASE_URL,EMBED_MODEL,QWEN_MODEL_OLLAMA
from common.factory import get_reasoning_model

class BcrState(BaseModel):
    """BCR状态模型"""
    bar_code: List[str] = Field(description="包含条形码信息的字段名列表", default_factory=list)
    Heavy: List[str] = Field(description="包含重链序列信息的字段名列表", default_factory=list)
    Light: List[str] = Field(description="包含轻链序列信息的字段名列表", default_factory=list)

def extract_bcr_info_with_llm(bcr_file_path: str, n_rows: int = 5) -> Dict[str, Any]:
    """
    使用大模型提取BCR信息
    
    Args:
        bcr_file_path: BCR文件路径
        n_rows: 需要提取的行数
    
    Returns:
        提取的BCR信息
    """
    try:
        # 首先获取所有列名
        df_columns = pd.read_csv(bcr_file_path, nrows=0).columns.tolist()
        
        # 创建prompt模板
        prompt_template = """
        分析CSV文件字段名，从以下字段列表中找出对应的BCR相关字段：
        
        请分析以下示例数据：
        {json_data}
        
        任务：从{column_names}中识别出：
        
        1. 包含条形码/细胞标识信息的字段名（可能包含：barcode、cell_id、cell_barcode、new_barcode、Barcode等关键词）,作为bar_code字段返回
        2. 包含重链序列信息的字段名（可能包含：Heavy、H_、heavy、重链、HC等关键词）,作为Heavy字段返回
        3. 包含轻链序列信息的字段名（可能包含：Light、L_、light、轻链、LC等关键词）,作为Light字段返回
        
        请严格按照以下JSON格式返回结果：
        {{
            "bar_code": ["实际字段名1", "实际字段名2"],
            "Heavy": ["实际字段名1"],
            "Light": ["实际字段名1"]
        }}
        """
        # 读取CSV文件前N行，添加参数处理特殊字符
        df = pd.read_csv(
            bcr_file_path, 
            nrows=n_rows,
            quoting=1,  # QUOTE_ALL
            escapechar='\\',
            on_bad_lines='skip',  # 跳过有问题的行
            encoding='utf-8'
        )
        
        # 转换为字典列表格式
        data_list = []
        for _, row in df.iterrows():
            row_dict = {}
            for column in df.columns:
                row_dict[column] = str(row[column]) if pd.notna(row[column]) else ""
            data_list.append(row_dict)
        
        # 转换为JSON字符串
        json_data = json.dumps(data_list, ensure_ascii=False, indent=2)

        # 调试输出：显示传递给LLM的数据
        print(f"CSV文件所有字段名: {df_columns}")
        print(f"传递给LLM的JSON数据: {json_data}")
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        config = get_config()
        reasoning_model = get_reasoning_model(config)
        structured_model = reasoning_model.with_structured_output(BcrState)
        runnable = prompt | structured_model
        response = runnable.invoke({
            "column_names": df_columns,
            "json_data": json_data
        })
        
        # 调试输出：显示LLM的响应
        print(f"LLM响应对象: {response}")
        print(f"bar_code字段: {response.bar_code}")
        print(f"Heavy字段: {response.Heavy}")
        print(f"Light字段: {response.Light}")
    
        # 将BcrState对象转换为字典格式
        extracted_info = {
            "bar_code": response.bar_code,
            "Heavy": response.Heavy,
            "Light": response.Light
        }
        return extracted_info
    except Exception as e:
        print(f"调用大模型时出错: {e}")
        print("详细堆栈信息:")
        traceback.print_exc()
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

def get_config() -> RunnableConfig:
    return {
        "configurable": {
            "model_config": {
                "default_model": {
                    "provider": "Ollama",
                    "model": REASONING_MODEL,
                    "params": {"temperature": 0.2}
                },
                "embedding_model": {
                    "provider": "Ollama",
                    "model": EMBED_MODEL
                },
                "reasoning_model": {
                    "provider": "Ollama",
                    "model": QWEN_MODEL_OLLAMA,
                    "params": {
                        # "base_url": QWEN_BASE_URL,
                        # "api_key": SecretStr("dummy"),
                        # "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                        "temperature": 0.2
                    }
                }
            }
        }
    }

def process_csv_to_standard_format(csv_file_path: str, bar_code: str, heavy: str, light: str, 
                                   variant_seq: str, experiment: str, output_path: str = None) -> str:
    """
    处理CSV文件，提取指定字段并生成标准格式的新CSV文件
    
    Args:
        csv_file_path: 输入CSV文件路径
        bar_code: 条形码字段名
        heavy: 重链字段名
        light: 轻链字段名
        variant_seq: 变异序列值（固定值）
        experiment: 实验值（固定值）
        output_path: 输出文件路径，如果为None则自动生成
    
    Returns:
        输出文件路径
    
    Raises:
        ValueError: 当指定的字段名在CSV中不存在时
        FileNotFoundError: 当输入文件不存在时
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(
            csv_file_path,
            quoting=1,
            escapechar='\\',
            on_bad_lines='skip',
            encoding='utf-8'
        )
        
        # 验证必需字段是否存在
        required_fields = [bar_code, heavy, light]
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            raise ValueError(f"以下字段在CSV文件中不存在: {missing_fields}")
        
        # 提取指定字段的数据
        extracted_data = df[[bar_code, heavy, light]].copy()
        
        # 重命名列为标准格式
        extracted_data.columns = ['combine_barcode', 'Heavy', 'Light']
        
        # 过滤条件1: 过滤掉轻链或重链为空的数据
        extracted_data = extracted_data.dropna(subset=['Heavy', 'Light'])
        extracted_data = extracted_data[(extracted_data['Heavy'].str.strip() != '') & (extracted_data['Light'].str.strip() != '')]
        
        # 过滤条件2: 过滤掉轻链或重链长度超过235的记录
        extracted_data = extracted_data[(extracted_data['Heavy'].str.len() <= 235) & (extracted_data['Light'].str.len() <= 235)]
        
        # 添加固定值列
        extracted_data['variant_seq'] = variant_seq
        extracted_data['experiment'] = experiment
        extracted_data['Label'] = ''  # 空列
        
        # 生成输出文件路径
        import os
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
            output_dir = os.path.dirname(csv_file_path)
            output_path = os.path.join(output_dir, f"{base_name}_processed.csv")
        elif isinstance(output_path, str) and (os.path.isdir(output_path) or output_path.endswith('/') or output_path.endswith('\\')):
            # 如果output_path是目录路径，生成完整的文件路径
            base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
            output_path = os.path.join(output_path, f"{base_name}_processed.csv")
        
        # 保存新CSV文件
        extracted_data.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"成功处理CSV文件，输出文件: {output_path}")
        print(f"处理了 {len(extracted_data)} 行数据")
        print(f"输出列: {list(extracted_data.columns)}")
        
        return output_path
        
    except FileNotFoundError:
        raise FileNotFoundError(f"输入文件不存在: {csv_file_path}")
    except Exception as e:
        print(f"处理CSV文件时出错: {e}")
        traceback.print_exc()
        raise