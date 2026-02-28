"""
Pytest configuration file for general QA subgraph tests
"""

import pytest
import os

# 全局变量存储命令行参数
_question_range = None
_question_ids = None
_question_indices = None
_start_index = None
_end_index = None


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--num-questions",
        action="store",
        default=10,
        type=int,
        help="Number of random questions to test (default: 10)"
    )
    parser.addoption(
        "--baseline-node-outputs",
        action="store",
        default=None,
        help="Path to baseline node_outputs JSON for per-node delta metrics"
    )
    parser.addoption(
        "--disable-xmasters",
        action="store_true",
        default=False,
        help="Disable X-Masters optimization (test original general_qa)"
    )
    parser.addoption(
        "--llm-provider",
        action="store",
        default=None,
        help="LLM provider (dashscope, zhipu, etc.)"
    )
    parser.addoption(
        "--llm-model",
        action="store",
        default=None,
        help="LLM model name (e.g., qwen-max, qwen-turbo, glm-4.5-air:1131206110::21rbvay4)"
    )
    parser.addoption(
        "--llm-temperature",
        action="store",
        default=None,
        type=float,
        help="LLM temperature parameter (e.g., 0.1, 0.3, 0.5)"
    )
    parser.addoption(
        "--start-index",
        action="store",
        default=None,
        type=int,
        help="Start index for question range (0-based, inclusive). If specified, questions will be selected from this index instead of random selection."
    )
    parser.addoption(
        "--end-index",
        action="store",
        default=None,
        type=int,
        help="End index for question range (0-based, exclusive). If specified with --start-index, questions will be selected from [start_index, end_index)."
    )
    parser.addoption(
        "--question-range",
        action="store",
        default=None,
        help="指定问题范围，格式: start-end (如 0-5 表示第1-6个问题)"
    )
    parser.addoption(
        "--questions",
        action="store",
        default=None,
        help="指定问题ID列表，逗号分隔 (如 66e88728,66e8add1)"
    )
    parser.addoption(
        "--question-indices",
        action="store",
        default=None,
        help="指定问题索引列表，逗号分隔 (如 0,2,5 表示第1,3,6个问题)"
    )


def pytest_configure(config):
    """读取命令行参数并存储到全局变量"""
    global _question_range, _question_ids, _question_indices, _start_index, _end_index
    
    _question_range = config.getoption("--question-range", default=None)
    _question_ids = config.getoption("--questions", default=None)
    _question_indices = config.getoption("--question-indices", default=None)
    _start_index = config.getoption("--start-index", default=None)
    _end_index = config.getoption("--end-index", default=None)
    
    # 打印参数信息
    if _question_range:
        print(f"\n📍 问题范围: {_question_range}")
    if _question_ids:
        print(f"\n📍 指定问题ID: {_question_ids}")
    if _question_indices:
        print(f"\n📍 指定问题索引: {_question_indices}")
        os.environ["TEST_QUESTION_INDICES"] = _question_indices
    if _start_index is not None or _end_index is not None:
        print(f"\n📍 起始索引: {_start_index}, 结束索引: {_end_index}")


def get_question_filters():
    """返回当前的问题过滤参数"""
    return {
        "question_range": _question_range,
        "question_ids": _question_ids,
        "question_indices": _question_indices,
        "start_index": _start_index,
        "end_index": _end_index
    }

