#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生信分析模块化MCP服务器所有工具测试文件

运行所有Figure2-Figure5相关工具的测试
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入各个测试模块
from test_figure2_tools import run_all_figure2_tests
from test_figure3_tools import run_all_figure3_tests
from test_figure4_tools import run_all_figure4_tests
from test_figure5_tools import run_all_figure5_tests

def run_all_tests():
    """运行所有工具测试"""
    start_time = time.time()
    
    print("*" * 80)
    print("生信分析模块化MCP服务器 - 所有工具测试")
    print("*" * 80)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"输入文件: D:\\data\\test_data_20251001\\Age_Bcells.rds")
    print(f"输出目录: D:\\data\\test_data_20251001")
    print("*" * 80)
    
    try:
        # 运行Figure2工具测试
        print("\n" + "=" * 80)
        print("第1部分: Figure2工具测试")
        print("=" * 80)
        run_all_figure2_tests()
        
        # 运行Figure3工具测试
        print("\n" + "=" * 80)
        print("第2部分: Figure3工具测试")
        print("=" * 80)
        run_all_figure3_tests()
        
        # 运行Figure4工具测试
        print("\n" + "=" * 80)
        print("第3部分: Figure4工具测试")
        print("=" * 80)
        run_all_figure4_tests()
        
        # 运行Figure5工具测试
        print("\n" + "=" * 80)
        print("第4部分: Figure5工具测试")
        print("=" * 80)
        run_all_figure5_tests()
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {str(e)}")
        print("请检查输入文件是否存在以及R环境是否正确配置")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "*" * 80)
    print("所有工具测试完成")
    print("*" * 80)
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print(f"总耗时: {duration:.2f} 秒")
    print("*" * 80)

def run_specific_figure_tests(figure_num):
    """运行特定Figure的工具测试"""
    print(f"运行Figure{figure_num}工具测试...")
    
    if figure_num == 2:
        run_all_figure2_tests()
    elif figure_num == 3:
        run_all_figure3_tests()
    elif figure_num == 4:
        run_all_figure4_tests()
    elif figure_num == 5:
        run_all_figure5_tests()
    else:
        print(f"不支持的Figure编号: {figure_num}")
        print("支持的编号: 2, 3, 4, 5")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 如果提供了参数，运行特定Figure的测试
        try:
            figure_num = int(sys.argv[1])
            run_specific_figure_tests(figure_num)
        except ValueError:
            print("请提供有效的Figure编号 (2, 3, 4, 5)")
            print("用法: python test_all_tools.py [figure_number]")
            print("示例: python test_all_tools.py 2")
    else:
        # 没有参数时运行所有测试
        run_all_tests()