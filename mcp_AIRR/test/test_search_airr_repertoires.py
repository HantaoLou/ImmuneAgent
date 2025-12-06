"""
测试 search_airr_repertoires 工具

此测试文件用于测试 search_airr_repertoires 工具的功能，
该工具用于搜索 AIRR 仓库中的 BCR 库。
"""

import sys
import os
import json
import shutil
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入要测试的函数和缓存管理器
from airr_mcp_server import search_airr_repertoires, cache_manager

# 清除缓存
def clear_cache():
    """清除缓存目录，确保测试使用最新数据"""
    # 获取当前工作目录
    cwd = os.getcwd()
    print(f"当前工作目录: {cwd}")
    
    # 清除项目根目录的缓存
    root_cache = Path(cwd).parent / "cache"
    if root_cache.exists():
        print(f"清除项目根目录缓存: {root_cache}")
        try:
            for item in root_cache.glob("queries/*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        except Exception as e:
            print(f"清除根目录缓存时出错: {e}")
    
    # 清除mcp_AIRR目录的缓存
    mcp_cache = Path(cwd) / "cache"
    if mcp_cache.exists():
        print(f"清除mcp_AIRR目录缓存: {mcp_cache}")
        try:
            for item in mcp_cache.glob("queries/*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        except Exception as e:
            print(f"清除mcp_AIRR目录缓存时出错: {e}")
            
    # 使用缓存管理器的API清除缓存
    try:
        count = cache_manager.invalidate("queries")
        print(f"通过缓存管理器API清除了 {count} 个缓存条目")
    except Exception as e:
        print(f"通过API清除缓存时出错: {e}")


def test_search_airr_repertoires():
    """测试搜索 AIRR 仓库中的 BCR 库"""
    
    # 首先清除缓存，确保使用最新数据
    print("\n清除缓存以确保使用最新数据...")
    clear_cache()
    
    # 设置测试参数 - 使用更宽泛的条件以增加找到结果的可能性
    params = {
        "disease": None,  # 不指定疾病条件，更宽泛
        "tissue": None,  # 不指定组织类型，更宽泛
        "species": "human",  # 物种
        "cell_subset": None,  # 不指定B细胞子集，更宽泛
        "repository": "ireceptor",  # 只使用ireceptor仓库
        "max_results": 10  # 增加最大结果数量
    }
    
    print(f"\n使用参数: {params}")
    
    # 调用函数
    result = search_airr_repertoires(**params)
    
    # 打印结果
    print("\n===== search_airr_repertoires 测试结果 =====")
    print(f"状态: {result.get('status')}")
    print(f"查询参数: {result.get('query_parameters')}")
    
    # 打印找到的库数量
    repertoires = result.get('repertoires', [])
    print(f"找到的库数量: {len(repertoires)}")
    
    # 打印搜索的仓库
    print(f"搜索的仓库: {result.get('repositories_searched', [])}")
    
    # 如果找到了库，打印第一个库的详细信息
    if repertoires:
        print("\n第一个库的详细信息:")
        first_rep = repertoires[0]
        for key, value in first_rep.items():
            print(f"  {key}: {value}")
        
        # 打印找到的repertoire_id，可用于其他测试
        print("\n可用的repertoire_id列表 (用于其他测试):")
        for i, rep in enumerate(repertoires[:5]):  # 只显示前5个
            print(f"  {i+1}. {rep.get('repertoire_id')}")
    else:
        print("\n未找到任何库，请检查API连接和查询参数")
    
    # 返回结果以便可能的进一步处理
    return result


if __name__ == "__main__":
    # 执行测试
    test_result = test_search_airr_repertoires()
    
    # 可选：将结果保存到文件
    # with open("search_result.json", "w") as f:
    #     json.dump(test_result, f, indent=2)