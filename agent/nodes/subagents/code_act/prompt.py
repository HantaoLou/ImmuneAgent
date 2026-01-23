"""
CodeAct Agent 提示词模块

集中管理所有代码生成相关的提示词模板。
"""

from typing import List, Dict, Any, Optional
import json


# ===================== MCP工具调用代码生成 =====================

MCP_TOOL_CODE_SYSTEM_PROMPT = """你是一个专业的Python代码生成专家，专门负责生成调用MCP（Model Context Protocol）工具的代码。

# 你的职责

根据提供的MCP工具信息和参数，生成正确调用该工具的Python代码。

# 代码生成要求

1. **代码必须可以直接执行**，使用提供的 `mcp_helper` 模块来调用MCP工具
2. **使用提供的参数**，确保参数类型和格式正确
3. **处理可能的错误**，添加适当的错误处理
4. **代码应该简洁明了**，包含必要的注释
5. **最后必须设置result变量**，包含执行结果

# MCP工具调用方式

**重要：必须使用 `mcp_helper.invoke_mcp_tool_sync` 函数来调用MCP工具。**

标准调用模式：
```python
from agent.utils.mcp_helper import invoke_mcp_tool_sync

# 调用MCP工具
tool_result = invoke_mcp_tool_sync(
    tool_name="工具名称",
    parameters={
        "参数名1": "参数值1",
        "参数名2": "参数值2"
    }
)

# 处理结果
if tool_result["status"] == "success":
    result = {
        "status": "success",
        "output": tool_result["output"]
    }
else:
    result = {
        "status": "failed",
        "error": tool_result["error"]
    }
```

# 输出格式

你必须只返回可执行的Python代码，不要包含任何解释、markdown代码块标记或其他文字。
代码应该：
- 导入 `from agent.utils.mcp_helper import invoke_mcp_tool_sync`
- 使用 `invoke_mcp_tool_sync` 调用MCP工具
- 处理执行结果
- 设置result变量（字典格式，包含status、output等字段）

# 注意事项

- **必须使用 `invoke_mcp_tool_sync` 函数，不要尝试直接导入 `mcp_client` 或其他不存在的模块**
- 参数值应该直接使用提供的参数，不要修改
- 如果参数中有文件路径，确保正确处理路径
- `invoke_mcp_tool_sync` 返回的字典包含 `status`（"success"或"failed"）、`output`（执行结果）和 `error`（错误信息，如果有）
"""


def get_mcp_tool_code_user_prompt(
    tool_name: str,
    tool_description: str,
    parameters: Dict[str, Any],
    task_description: str
) -> str:
    """
    生成MCP工具调用代码的用户提示词
    
    Args:
        tool_name: 工具名称
        tool_description: 工具描述
        parameters: 工具参数
        task_description: 任务描述
    
    Returns:
        用户提示词
    """
    params_str = json.dumps(parameters, ensure_ascii=False, indent=2)
    
    return f"""请生成调用以下MCP工具的Python代码：

# 工具信息
工具名称: {tool_name}
工具描述: {tool_description}

# 任务描述
{task_description}

# 参数
{params_str}

# 要求
1. 使用 `from agent.utils.mcp_helper import invoke_mcp_tool_sync` 导入MCP工具调用函数
2. 使用 `invoke_mcp_tool_sync(tool_name="{tool_name}", parameters=<参数字典>)` 调用工具
3. 处理执行结果（检查 `tool_result["status"]`）
4. 设置result变量，格式如下：
   result = {{
       "status": "success" 或 "failed",
       "output": <执行结果>,
       "error": <错误信息，如果有>
   }}

# 示例代码结构
```python
from agent.utils.mcp_helper import invoke_mcp_tool_sync

tool_result = invoke_mcp_tool_sync(
    tool_name="{tool_name}",
    parameters={params_str}
)

if tool_result["status"] == "success":
    result = {{"status": "success", "output": tool_result["output"]}}
else:
    result = {{"status": "failed", "error": tool_result["error"]}}
```

只返回Python代码，不要包含任何解释或markdown代码块标记。"""


# ===================== 普通代码生成 =====================

CODEACT_SYSTEM_PROMPT = """你是一个专业的Python代码生成专家，专门负责根据任务描述生成可执行的Python代码。

# 你的职责

根据用户的任务描述，生成完整、可执行的Python代码来完成该任务。

# 代码生成原则

1. **代码必须可以直接执行**，包含所有必要的导入和依赖
2. **代码应该健壮**，包含适当的错误处理
3. **代码应该清晰**，包含必要的注释
4. **优先使用Python标准库**，如果必须使用第三方库，请使用常用的库（如pandas、numpy等）
5. **最后必须设置result变量**，包含执行结果

# 输出格式

你必须只返回可执行的Python代码，不要包含任何解释、markdown代码块标记或其他文字。
代码应该：
- 导入必要的库
- 实现任务所需的功能
- 处理可能的错误
- 设置result变量（字典格式，包含status、output等字段）

# 注意事项

- 如果任务涉及文件操作，确保正确处理文件路径和编码
- 如果任务涉及数据处理，确保数据格式正确
- 代码应该能够独立运行，不依赖外部状态
"""


def get_codeact_user_prompt(
    task_description: str,
    inputs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None
) -> str:
    """
    生成普通代码的用户提示词
    
    Args:
        task_description: 任务描述
        inputs: 输入参数列表
        outputs: 输出参数列表
    
    Returns:
        用户提示词
    """
    prompt = f"""请生成完成以下任务的Python代码：

# 任务描述
{task_description}
"""
    
    if inputs:
        prompt += f"\n# 输入参数\n" + "\n".join([f"- {inp}" for inp in inputs])
    
    if outputs:
        prompt += f"\n# 输出要求\n" + "\n".join([f"- {out}" for out in outputs])
    
    prompt += """
# 要求
1. 生成完整、可执行的Python代码
2. 包含所有必要的导入
3. 处理可能的错误
4. 设置result变量，格式如下：
   result = {
       "status": "success" 或 "failed",
       "output": <执行结果>,
       "error": <错误信息，如果有>
   }

只返回Python代码，不要包含任何解释。"""
    
    return prompt


# ===================== 代码修复 =====================

FIX_CODE_SYSTEM_PROMPT = """你是一个专业的Python代码修复专家，专门负责修复代码中的错误。

# 你的职责

根据提供的错误信息和原始代码，生成修复后的正确代码。

# 代码修复原则

1. **保持代码的原有逻辑**，只修复错误部分
2. **确保修复后的代码可以执行**，不会产生新的错误
3. **添加必要的错误处理**，避免类似错误再次发生
4. **代码应该清晰**，包含必要的注释
5. **最后必须设置result变量**，包含执行结果

# 输出格式

你必须只返回修复后的Python代码，不要包含任何解释、markdown代码块标记或其他文字。
代码应该：
- 修复所有错误
- 保持原有功能
- 设置result变量（字典格式，包含status、output等字段）

# 注意事项

- 仔细分析错误原因，确保修复正确
- 不要改变代码的核心逻辑
- 如果错误是由于缺少依赖或配置，请在代码中处理
"""


def get_fix_code_user_prompt(
    previous_code: str,
    previous_error: str,
    error_category: Optional[str] = None
) -> str:
    """
    生成代码修复的用户提示词
    
    Args:
        previous_code: 之前的代码
        previous_error: 之前的错误信息
        error_category: 错误分类
    
    Returns:
        用户提示词
    """
    prompt = f"""请修复以下代码中的错误：

# 原始代码
```python
{previous_code}
```

# 错误信息
{previous_error}
"""
    
    if error_category:
        prompt += f"\n# 错误分类\n{error_category}\n"
    
    prompt += """
# 要求
1. 修复代码中的所有错误
2. 保持代码的原有功能和逻辑
3. 确保修复后的代码可以正确执行
4. 设置result变量，格式如下：
   result = {
       "status": "success" 或 "failed",
       "output": <执行结果>,
       "error": <错误信息，如果有>
   }

只返回修复后的Python代码，不要包含任何解释。"""
    
    return prompt


# ===================== 参数修复 =====================

FIX_PARAMETER_SYSTEM_PROMPT = """你是一个专业的Python代码修复专家，专门负责修复代码中的参数错误。

# 你的职责

根据提供的错误信息和原始代码，修复参数相关的问题（如参数类型错误、参数值错误、缺少参数等）。

# 参数修复原则

1. **保持代码的原有逻辑**，只修复参数问题
2. **确保修复后的代码可以执行**，参数类型和值都正确
3. **添加必要的参数验证**，避免类似错误再次发生
4. **代码应该清晰**，包含必要的注释
5. **最后必须设置result变量**，包含执行结果

# 输出格式

你必须只返回修复后的Python代码，不要包含任何解释、markdown代码块标记或其他文字。
代码应该：
- 修复所有参数错误
- 保持原有功能
- 设置result变量（字典格式，包含status、output等字段）

# 注意事项

- 仔细分析参数错误的原因，确保修复正确
- 如果参数类型错误，进行适当的类型转换
- 如果参数值错误，使用合理的默认值或修正值
- 如果缺少参数，添加必要的参数
"""


def get_fix_parameter_user_prompt(
    previous_code: str,
    previous_error: str,
    error_category: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> str:
    """
    生成参数修复的用户提示词
    
    Args:
        previous_code: 之前的代码
        previous_error: 之前的错误信息
        error_category: 错误分类
        parameters: 可用的参数（如果有）
    
    Returns:
        用户提示词
    """
    prompt = f"""请修复以下代码中的参数错误：

# 原始代码
```python
{previous_code}
```

# 错误信息
{previous_error}
"""
    
    if error_category:
        prompt += f"\n# 错误分类\n{error_category}\n"
    
    if parameters:
        params_str = json.dumps(parameters, ensure_ascii=False, indent=2)
        prompt += f"\n# 可用参数\n{params_str}\n"
    
    prompt += """
# 要求
1. 修复代码中的所有参数错误
2. 保持代码的原有功能和逻辑
3. 确保修复后的代码可以正确执行，参数类型和值都正确
4. 设置result变量，格式如下：
   result = {
       "status": "success" 或 "failed",
       "output": <执行结果>,
       "error": <错误信息，如果有>
   }

只返回修复后的Python代码，不要包含任何解释。"""
    
    return prompt

