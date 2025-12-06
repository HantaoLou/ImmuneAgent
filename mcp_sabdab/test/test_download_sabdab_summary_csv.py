"""
测试 download_sabdab_summary_csv 方法

该测试文件通过直接调用 sabdab_mcp_server 中的 download_sabdab_summary_csv 方法来验证其功能。
"""

import sys
import os

# 添加父目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sabdab_mcp_server import download_sabdab_summary_csv


def test_download_sabdab_summary_csv_no_filters():
    """测试不带过滤条件的 CSV 下载"""
    print("测试 download_sabdab_summary_csv (无过滤条件)...")
    
    try:
        # 调用方法，不传入过滤条件
        result = download_sabdab_summary_csv()
        
        # 验证返回结果的结构
        assert isinstance(result, dict), "返回结果应该是字典类型"
        assert "status" in result, "返回结果应包含 status 字段"
        
        print(f"状态: {result.get('status')}")
        print(f"条目数量: {result.get('num_entries', 'N/A')}")
        print(f"文件大小: {result.get('file_size_bytes', 'N/A')} bytes")
        
        # 如果有 CSV 内容，显示前几行
        if "csv_content" in result and result["csv_content"]:
            lines = result["csv_content"].split('\n')[:3]  # 显示前3行
            print("CSV 内容预览:")
            for line in lines:
                print(f"  {line}")
        
        print("✓ 测试通过: download_sabdab_summary_csv (无过滤条件)")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


def test_download_sabdab_summary_csv_with_filters():
    """测试带过滤条件的 CSV 下载"""
    print("\n测试 download_sabdab_summary_csv (带过滤条件)...")
    
    try:
        # 设置过滤条件
        filters = {
            "resolution": "<2.5",
            "antigen": "yes"
        }
        
        # 调用方法，传入过滤条件
        result = download_sabdab_summary_csv(filters)
        
        # 验证返回结果的结构
        assert isinstance(result, dict), "返回结果应该是字典类型"
        assert "status" in result, "返回结果应包含 status 字段"
        
        print(f"状态: {result.get('status')}")
        print(f"过滤条件: {filters}")
        print(f"条目数量: {result.get('num_entries', 'N/A')}")
        print(f"文件大小: {result.get('file_size_bytes', 'N/A')} bytes")
        
        # 如果有 CSV 内容，显示前几行
        if "csv_content" in result and result["csv_content"]:
            lines = result["csv_content"].split('\n')[:3]  # 显示前3行
            print("CSV 内容预览:")
            for line in lines:
                print(f"  {line}")
        
        print("✓ 测试通过: download_sabdab_summary_csv (带过滤条件)")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试 download_sabdab_summary_csv 方法")
    print("=" * 60)
    
    # 运行测试
    test1_passed = test_download_sabdab_summary_csv_no_filters()
    test2_passed = test_download_sabdab_summary_csv_with_filters()
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"无过滤条件测试: {'通过' if test1_passed else '失败'}")
    print(f"带过滤条件测试: {'通过' if test2_passed else '失败'}")
    
    if test1_passed and test2_passed:
        print("✓ 所有测试通过!")
        exit(0)
    else:
        print("✗ 部分测试失败!")
        exit(1)