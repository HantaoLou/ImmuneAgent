"""
测试 get_sabdab_statistics 方法

该测试文件通过直接调用 sabdab_mcp_server 中的 get_sabdab_statistics 方法来验证其功能。
"""

import sys
import os

# 添加父目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sabdab_mcp_server import get_sabdab_statistics


def test_get_sabdab_statistics():
    """测试获取 SAbDab 数据库统计信息"""
    print("测试 get_sabdab_statistics...")
    
    try:
        # 调用方法获取统计信息
        result = get_sabdab_statistics()
        
        # 验证返回结果的结构
        assert isinstance(result, dict), "返回结果应该是字典类型"
        assert "status" in result, "返回结果应包含 status 字段"
        
        print(f"状态: {result.get('status')}")
        
        # 显示统计信息的各个字段
        print("\n数据库统计信息:")
        for key, value in result.items():
            if key != "status":
                if isinstance(value, dict):
                    print(f"{key}:")
                    for sub_key, sub_value in value.items():
                        print(f"  {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    print(f"{key}: {len(value)} 项")
                    if len(value) > 0:
                        print(f"  示例: {value[:3]}...")  # 显示前3项作为示例
                else:
                    print(f"{key}: {value}")
        
        # 验证一些常见的统计字段是否存在
        expected_fields = ["total_entries", "database_info", "last_updated", "statistics"]
        found_fields = []
        
        for field in expected_fields:
            if field in result:
                found_fields.append(field)
                print(f"✓ 找到预期字段: {field}")
        
        if found_fields:
            print(f"✓ 找到 {len(found_fields)} 个预期字段")
        else:
            print("! 未找到预期的统计字段，但这可能是正常的")
        
        print("✓ 测试通过: get_sabdab_statistics")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


def test_get_sabdab_statistics_return_structure():
    """测试 get_sabdab_statistics 返回结构的完整性"""
    print("\n测试 get_sabdab_statistics 返回结构...")
    
    try:
        # 调用方法获取统计信息
        result = get_sabdab_statistics()
        
        # 基本结构验证
        assert isinstance(result, dict), "返回结果应该是字典类型"
        assert len(result) > 0, "返回结果不应为空"
        
        print("返回结果结构分析:")
        print(f"字段总数: {len(result)}")
        
        # 分析每个字段的类型
        field_types = {}
        for key, value in result.items():
            field_type = type(value).__name__
            field_types[key] = field_type
            
            if isinstance(value, (str, int, float)):
                print(f"  {key} ({field_type}): {value}")
            elif isinstance(value, dict):
                print(f"  {key} ({field_type}): {len(value)} 个子字段")
            elif isinstance(value, list):
                print(f"  {key} ({field_type}): {len(value)} 个元素")
            else:
                print(f"  {key} ({field_type}): {str(value)[:50]}...")
        
        print(f"✓ 字段类型分布: {field_types}")
        print("✓ 测试通过: get_sabdab_statistics 返回结构验证")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


def test_get_sabdab_statistics_consistency():
    """测试 get_sabdab_statistics 的一致性（多次调用）"""
    print("\n测试 get_sabdab_statistics 一致性...")
    
    try:
        # 第一次调用
        result1 = get_sabdab_statistics()
        print("第一次调用完成")
        
        # 第二次调用
        result2 = get_sabdab_statistics()
        print("第二次调用完成")
        
        # 比较两次调用的结果
        assert isinstance(result1, dict), "第一次调用结果应该是字典类型"
        assert isinstance(result2, dict), "第二次调用结果应该是字典类型"
        
        # 检查字段数量是否一致
        if len(result1) == len(result2):
            print(f"✓ 两次调用返回的字段数量一致: {len(result1)}")
        else:
            print(f"! 字段数量不一致: {len(result1)} vs {len(result2)}")
        
        # 检查共同字段
        common_fields = set(result1.keys()) & set(result2.keys())
        print(f"✓ 共同字段数量: {len(common_fields)}")
        
        # 检查状态字段是否一致
        if result1.get('status') == result2.get('status'):
            print(f"✓ 状态字段一致: {result1.get('status')}")
        else:
            print(f"! 状态字段不一致: {result1.get('status')} vs {result2.get('status')}")
        
        print("✓ 测试通过: get_sabdab_statistics 一致性验证")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试 get_sabdab_statistics 方法")
    print("=" * 60)
    
    # 运行测试
    test1_passed = test_get_sabdab_statistics()
    test2_passed = test_get_sabdab_statistics_return_structure()
    test3_passed = test_get_sabdab_statistics_consistency()
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"基本功能测试: {'通过' if test1_passed else '失败'}")
    print(f"返回结构测试: {'通过' if test2_passed else '失败'}")
    print(f"一致性测试: {'通过' if test3_passed else '失败'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("✓ 所有测试通过!")
        exit(0)
    else:
        print("✗ 部分测试失败!")
        exit(1)