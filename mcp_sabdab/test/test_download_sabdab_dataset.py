"""
测试 download_sabdab_dataset 方法

该测试文件通过直接调用 sabdab_mcp_server 中的 download_sabdab_dataset 方法来验证其功能。
"""

import sys
import os

# 添加父目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sabdab_mcp_server import download_sabdab_dataset


def test_download_sabdab_dataset_all_csv():
    """测试下载所有数据集 (CSV 格式)"""
    print("测试 download_sabdab_dataset (all, csv)...")
    
    try:
        # 调用方法，使用默认参数 (all, csv)
        result = download_sabdab_dataset()
        
        # 验证返回结果的结构
        assert isinstance(result, dict), "返回结果应该是字典类型"
        assert "status" in result, "返回结果应包含 status 字段"
        
        print(f"状态: {result.get('status')}")
        print(f"数据集类型: all")
        print(f"输出格式: csv")
        
        # 显示其他可能的字段
        for key, value in result.items():
            if key not in ["status"] and value is not None:
                if isinstance(value, str) and len(value) > 100:
                    print(f"{key}: {len(value)} 字符 (内容过长，已截断)")
                else:
                    print(f"{key}: {value}")
        
        print("✓ 测试通过: download_sabdab_dataset (all, csv)")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


def test_download_sabdab_dataset_antigen_bound():
    """测试下载抗原结合数据集"""
    print("\n测试 download_sabdab_dataset (antigen_bound, csv)...")
    
    try:
        # 调用方法，指定抗原结合数据集
        result = download_sabdab_dataset("antigen_bound", "csv")
        
        # 验证返回结果的结构
        assert isinstance(result, dict), "返回结果应该是字典类型"
        assert "status" in result, "返回结果应包含 status 字段"
        
        print(f"状态: {result.get('status')}")
        print(f"数据集类型: antigen_bound")
        print(f"输出格式: csv")
        
        # 显示其他可能的字段
        for key, value in result.items():
            if key not in ["status"] and value is not None:
                if isinstance(value, str) and len(value) > 100:
                    print(f"{key}: {len(value)} 字符 (内容过长，已截断)")
                else:
                    print(f"{key}: {value}")
        
        print("✓ 测试通过: download_sabdab_dataset (antigen_bound, csv)")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


def test_download_sabdab_dataset_nanobodies():
    """测试下载纳米抗体数据集"""
    print("\n测试 download_sabdab_dataset (nanobodies, csv)...")
    
    try:
        # 调用方法，指定纳米抗体数据集
        result = download_sabdab_dataset("nanobodies", "csv")
        
        # 验证返回结果的结构
        assert isinstance(result, dict), "返回结果应该是字典类型"
        assert "status" in result, "返回结果应包含 status 字段"
        
        print(f"状态: {result.get('status')}")
        print(f"数据集类型: nanobodies")
        print(f"输出格式: csv")
        
        # 显示其他可能的字段
        for key, value in result.items():
            if key not in ["status"] and value is not None:
                if isinstance(value, str) and len(value) > 100:
                    print(f"{key}: {len(value)} 字符 (内容过长，已截断)")
                else:
                    print(f"{key}: {value}")
        
        print("✓ 测试通过: download_sabdab_dataset (nanobodies, csv)")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


def test_download_sabdab_dataset_json_format():
    """测试下载数据集 (JSON 格式)"""
    print("\n测试 download_sabdab_dataset (all, json)...")
    
    try:
        # 调用方法，指定 JSON 格式
        result = download_sabdab_dataset("all", "json")
        
        # 验证返回结果的结构
        assert isinstance(result, dict), "返回结果应该是字典类型"
        assert "status" in result, "返回结果应包含 status 字段"
        
        print(f"状态: {result.get('status')}")
        print(f"数据集类型: all")
        print(f"输出格式: json")
        
        # 显示其他可能的字段
        for key, value in result.items():
            if key not in ["status"] and value is not None:
                if isinstance(value, str) and len(value) > 100:
                    print(f"{key}: {len(value)} 字符 (内容过长，已截断)")
                else:
                    print(f"{key}: {value}")
        
        print("✓ 测试通过: download_sabdab_dataset (all, json)")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


def test_download_sabdab_dataset_fasta_format():
    """测试下载数据集 (FASTA 格式)"""
    print("\n测试 download_sabdab_dataset (all, fasta)...")
    
    try:
        # 调用方法，指定 FASTA 格式
        result = download_sabdab_dataset("all", "fasta")
        
        # 验证返回结果的结构
        assert isinstance(result, dict), "返回结果应该是字典类型"
        assert "status" in result, "返回结果应包含 status 字段"
        
        print(f"状态: {result.get('status')}")
        print(f"数据集类型: all")
        print(f"输出格式: fasta")
        
        # 显示其他可能的字段
        for key, value in result.items():
            if key not in ["status"] and value is not None:
                if isinstance(value, str) and len(value) > 100:
                    print(f"{key}: {len(value)} 字符 (内容过长，已截断)")
                    # 如果是 FASTA 内容，显示前几行
                    if key.lower().find('fasta') != -1 or key.lower().find('content') != -1:
                        lines = value.split('\n')[:5]
                        print("内容预览:")
                        for line in lines:
                            if line.strip():
                                print(f"  {line}")
                else:
                    print(f"{key}: {value}")
        
        print("✓ 测试通过: download_sabdab_dataset (all, fasta)")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试 download_sabdab_dataset 方法")
    print("=" * 60)
    
    # 运行测试
    test1_passed = test_download_sabdab_dataset_all_csv()
    test2_passed = test_download_sabdab_dataset_antigen_bound()
    test3_passed = test_download_sabdab_dataset_nanobodies()
    test4_passed = test_download_sabdab_dataset_json_format()
    test5_passed = test_download_sabdab_dataset_fasta_format()
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"所有数据集 (CSV) 测试: {'通过' if test1_passed else '失败'}")
    print(f"抗原结合数据集测试: {'通过' if test2_passed else '失败'}")
    print(f"纳米抗体数据集测试: {'通过' if test3_passed else '失败'}")
    print(f"JSON 格式测试: {'通过' if test4_passed else '失败'}")
    print(f"FASTA 格式测试: {'通过' if test5_passed else '失败'}")
    
    all_passed = all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed])
    
    if all_passed:
        print("✓ 所有测试通过!")
        exit(0)
    else:
        print("✗ 部分测试失败!")
        exit(1)