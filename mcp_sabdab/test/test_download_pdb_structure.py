"""
测试 download_pdb_structure 方法

该测试文件通过直接调用 sabdab_mcp_server 中的 download_pdb_structure 方法来验证其功能。
"""

import sys
import os

# 添加父目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sabdab_mcp_server import download_pdb_structure


def test_download_pdb_structure_default():
    """测试使用默认编号方案下载 PDB 结构"""
    print("测试 download_pdb_structure (默认编号方案)...")
    
    try:
        # 使用一个已知的 PDB ID 进行测试
        pdb_id = "6m0j"
        
        # 调用方法，使用默认编号方案 (chothia)
        result = download_pdb_structure(pdb_id)
        
        # 验证返回结果的结构
        assert isinstance(result, dict), "返回结果应该是字典类型"
        assert "status" in result, "返回结果应包含 status 字段"
        
        print(f"状态: {result.get('status')}")
        print(f"PDB ID: {result.get('pdb_id', 'N/A')}")
        print(f"编号方案: {result.get('numbering_scheme', 'N/A')}")
        print(f"文件大小: {result.get('file_size_bytes', 'N/A')} bytes")
        
        # 如果有 PDB 内容，显示前几行
        if "pdb_content" in result and result["pdb_content"]:
            lines = result["pdb_content"].split('\n')[:5]  # 显示前5行
            print("PDB 内容预览:")
            for line in lines:
                if line.strip():  # 跳过空行
                    print(f"  {line}")
        
        print("✓ 测试通过: download_pdb_structure (默认编号方案)")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


def test_download_pdb_structure_kabat():
    """测试使用 Kabat 编号方案下载 PDB 结构"""
    print("\n测试 download_pdb_structure (Kabat 编号方案)...")
    
    try:
        # 使用一个已知的 PDB ID 进行测试
        pdb_id = "6m0j"
        numbering_scheme = "kabat"
        
        # 调用方法，指定 Kabat 编号方案
        result = download_pdb_structure(pdb_id, numbering_scheme)
        
        # 验证返回结果的结构
        assert isinstance(result, dict), "返回结果应该是字典类型"
        assert "status" in result, "返回结果应包含 status 字段"
        
        print(f"状态: {result.get('status')}")
        print(f"PDB ID: {result.get('pdb_id', 'N/A')}")
        print(f"编号方案: {result.get('numbering_scheme', 'N/A')}")
        print(f"文件大小: {result.get('file_size_bytes', 'N/A')} bytes")
        
        # 验证编号方案是否正确设置
        if result.get('numbering_scheme') == numbering_scheme:
            print(f"✓ 编号方案设置正确: {numbering_scheme}")
        
        # 如果有 PDB 内容，显示前几行
        if "pdb_content" in result and result["pdb_content"]:
            lines = result["pdb_content"].split('\n')[:5]  # 显示前5行
            print("PDB 内容预览:")
            for line in lines:
                if line.strip():  # 跳过空行
                    print(f"  {line}")
        
        print("✓ 测试通过: download_pdb_structure (Kabat 编号方案)")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


def test_download_pdb_structure_imgt():
    """测试使用 IMGT 编号方案下载 PDB 结构"""
    print("\n测试 download_pdb_structure (IMGT 编号方案)...")
    
    try:
        # 使用一个已知的 PDB ID 进行测试
        pdb_id = "6m0j"
        numbering_scheme = "imgt"
        
        # 调用方法，指定 IMGT 编号方案
        result = download_pdb_structure(pdb_id, numbering_scheme)
        
        # 验证返回结果的结构
        assert isinstance(result, dict), "返回结果应该是字典类型"
        assert "status" in result, "返回结果应包含 status 字段"
        
        print(f"状态: {result.get('status')}")
        print(f"PDB ID: {result.get('pdb_id', 'N/A')}")
        print(f"编号方案: {result.get('numbering_scheme', 'N/A')}")
        print(f"文件大小: {result.get('file_size_bytes', 'N/A')} bytes")
        
        # 验证编号方案是否正确设置
        if result.get('numbering_scheme') == numbering_scheme:
            print(f"✓ 编号方案设置正确: {numbering_scheme}")
        
        # 如果有 PDB 内容，显示前几行
        if "pdb_content" in result and result["pdb_content"]:
            lines = result["pdb_content"].split('\n')[:5]  # 显示前5行
            print("PDB 内容预览:")
            for line in lines:
                if line.strip():  # 跳过空行
                    print(f"  {line}")
        
        print("✓ 测试通过: download_pdb_structure (IMGT 编号方案)")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试 download_pdb_structure 方法")
    print("=" * 60)
    
    # 运行测试
    test1_passed = test_download_pdb_structure_default()
    test2_passed = test_download_pdb_structure_kabat()
    test3_passed = test_download_pdb_structure_imgt()
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"默认编号方案测试: {'通过' if test1_passed else '失败'}")
    print(f"Kabat 编号方案测试: {'通过' if test2_passed else '失败'}")
    print(f"IMGT 编号方案测试: {'通过' if test3_passed else '失败'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("✓ 所有测试通过!")
        exit(0)
    else:
        print("✗ 部分测试失败!")
        exit(1)