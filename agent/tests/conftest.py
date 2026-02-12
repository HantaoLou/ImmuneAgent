"""
Pytest configuration file for general QA subgraph tests
"""

import pytest


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

