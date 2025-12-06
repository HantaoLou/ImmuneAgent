"""
FDG MCP Server - Core FDG Tool Wrapper

This server exposes the core FDG (Foldx, DDG, GearBind) process via MCP protocol.
"""

from mcp.server.fastmcp import FastMCP, Context
import os
import pandas as pd

# Create MCP server
mcp = FastMCP("IMMGPT Core Server")


@mcp.tool()
def analyse_fdg_result() -> list:
    """Perform analysis on the output of FDG, and return the result. First row is table head, which
    tells you name of columns. 
    
    Returns:
        table content indicating the analysis result.
    """
    import os
    from pathlib import Path
    # 设置CUDA环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
   # 导入analyzer.py中的关键组件
    from immagents.immroles import Analyzer
    from analyzer import create_analyzing_prompt, IMMGPT_DATA_PATH
    
    # 设置输入文件
    input_file = "H5N1_first-batch/0307_first-batch_exp-results.xlsx"
    
    # 构建文件路径（直接从analyzer.py中提取逻辑）
    result_name = os.path.basename(input_file).split(".")[0]
    analysis_file = result_name.replace("results", "analysis.txt")
    log_file = result_name.replace("results", "log.txt")
    selected_file = result_name.replace("results", "selected.csv")
    
    results_path = os.path.join(IMMGPT_DATA_PATH, input_file)
    analysis_path = os.path.join(os.path.dirname(results_path), "analysis")
    os.makedirs(analysis_path, exist_ok=True)
    selected_path = os.path.join(os.path.dirname(results_path), "selected")
    os.makedirs(selected_path, exist_ok=True)
    log_path = os.path.join(os.path.dirname(results_path), "log")
    os.makedirs(log_path, exist_ok=True)
    
    file_paths = {
        "results": results_path,
        "analysis": os.path.join(analysis_path, analysis_file),
        "selected": os.path.join(selected_path, selected_file),
        "log": os.path.join(log_path, log_file)
    }
    
    # 创建分析提示（直接从analyzer.py中提取逻辑）
    action_prompt = "Can you select the 5-top antibodies that you think have the highest potential to broadly against viruses?"
    query_idea = create_analyzing_prompt(action_prompt=action_prompt)
    print(f"分析文件: {results_path}")
    print(f"分析提示: {query_idea}")
    
    # 创建分析器
    analyzer = Analyzer()
    
    # 运行分析（处理异步调用）
    import asyncio
    
    try:
        # 尝试在当前事件循环中运行
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果当前事件循环正在运行，使用非阻塞方式
            import threading
            import concurrent.futures
            
            def run_async_in_thread():
                _loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_loop)
                return _loop.run_until_complete(analyzer.run(query_idea, file_paths))
            
            # 在线程中运行异步任务
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async_in_thread)
                response = future.result()
        else:
            # 如果当前事件循环不在运行，直接运行
            response = loop.run_until_complete(analyzer.run(query_idea, file_paths))
    except RuntimeError:
        # 后备方案
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(analyzer.run(query_idea, file_paths))
    
    # 保存日志
    with open(file_paths['log'], 'w') as f:
        f.write(str(response))
        print(f"日志已保存至: {file_paths['log']}")
    
    # 返回选定的抗体
    return _read_table(file_paths['selected'])


def _read_table(file_path: str) -> list:
    """Helper function to read a table file and return its content as a list of lists.
    
    Args:
        file_path: Path to the table file.
    
    Returns:
        A list of lists containing the table content.
    """
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_excel(file_path)
    return [df.columns.values.tolist()] + df.values.tolist()


# 添加生命周期管理
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

@asynccontextmanager
async def fdg_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """处理服务器启动和关闭"""
    print("IMMGPT MCP Server 正在初始化...")
    
    try:
        yield {"initialized": True}
    finally:
        print("FDG MCP Server 正在关闭...")

# 设置生命周期
mcp.lifespan = fdg_lifespan

if __name__ == "__main__":
    print("启动IMMGPT MCP服务器...")
    
    # 设置MCP标准路径
    # mcp.settings.sse_path = "/_mcp/v1/sse"
    # mcp.settings.message_path = "/_mcp/v1/messages/"
    
    # 设置网络参数
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = 8081
    
    # 使用SSE模式启动
    mcp.run(transport="sse")