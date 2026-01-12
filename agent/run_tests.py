"""
快速运行测试脚本

使用方法：
    python run_tests.py          # 运行所有测试
    python run_tests.py -v      # 详细输出
    python run_tests.py -k test_general  # 运行特定测试
"""

import sys
import pytest

if __name__ == "__main__":
    # 默认参数
    args = ["tests/test_agent.py", "-v", "-s"]
    
    # 添加用户传入的参数
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    
    # 运行测试
    exit_code = pytest.main(args)
    sys.exit(exit_code)

