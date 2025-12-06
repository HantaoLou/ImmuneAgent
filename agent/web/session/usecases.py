from typing import Callable

from langgraph.graph import StateGraph

from usecases.research.deep_researcher import build_graph
from usecases.immunity.graph.planning_graph import build_improved_graph
from usecases.immunity.state.state import ImprovedCellState
from config.api_keys import APIKeys


class Usecase:
    def __init__(
        self,
        name: str,
        default_configuration: dict,
        init_state_factory: Callable[[str], dict],
        graph_factory: Callable[[], StateGraph],
        result_factory: Callable[dict, str] = lambda x: str(x),
    ) -> None:
        self.name = name
        self.default_configuration = default_configuration
        self.init_state_factory = init_state_factory
        self.graph_factory = graph_factory
        self.result_factory = result_factory


class Usecases:
    RESEARCH = Usecase(
        name="research",
        default_configuration={
            "max_concurrent_research_units": 1,
            "max_researcher_iterations": 6,
            "model_config": {
                "summarize_model": {
                    "provider": "OpenAI",
                    "model": "gpt-4.1",
                },
                "reasoning_model": {
                    "provider": "OpenAI",
                    "model": "gpt-4.1",
                },
            },
            "mcp_config": {"service_ids": []},
        },
        init_state_factory=lambda user_message: {
            "supervisor_messages": [],
            "research_brief": None,
            "final_report": "",
            "messages": [user_message],
        },
        graph_factory=build_graph,
        result_factory=lambda x: x["final_report"],
    )

    IMMUNITY = Usecase(
        name="immunity",
        default_configuration={
            "mcp_config": {"service_ids": ["metabcr", "airr", "af3", "anarci", "geo", "lgblast", "oas", "bioinformatics", "annotation", "bcell", "communication", "multimodal", "scrna"]},
            "tavily_api_key": APIKeys.TAVILY_API_KEY,
             "model_config": {
                "default_model": {
                    "provider": "OpenAI",
                    "model": "qwen-plus-latest",
                    "params": {
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key": APIKeys.QWEN_API_KEY,
                        "temperature": 0.2,
                        "extra_body": {"enable_thinking": False},
                    },
                },
                "embedding_model": {
                    "provider": "OpenAI",
                    "model": "text-embedding-3-small",
                    "params": {
                        "base_url": "https://xiaoai.plus/v1",
                        "api_key": APIKeys.XIAOAI_API_KEY,
                    },
                },
                "summarize_model": {
                    "provider": "OpenAI",
                    "model": "qwen-plus-latest",
                    "params": {
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key": APIKeys.QWEN_API_KEY,
                        "temperature": 0.2,
                        "extra_body": {"enable_thinking": False},
                    },
                },
                "reasoning_model": {
                    "provider": "OpenAI",
                    "model": "qwen-plus-latest",
                    "params": {
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key": APIKeys.QWEN_API_KEY,
                        "temperature": 0.2,
                        "extra_body": {"enable_thinking": False},
                    },
                },
                "deep_research_model": {
                    "provider": "OpenAI",
                    "model": "qwen-plus-latest",
                    "params": {
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key": APIKeys.QWEN_API_KEY,
                        "temperature": 0.2,
                        "extra_body": {"enable_thinking": False},
                    },
                },
                "hypothesis_model": {
                    "provider": "OpenAI",
                    "model": "qwen-plus-latest",
                    "params": {
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key": APIKeys.QWEN_API_KEY,
                        "temperature": 0.2,
                        "extra_body": {"enable_thinking": False},
                    },
                },
                "planning_model": {
                    "provider": "OpenAI",
                    "model": "qwen-plus-latest",
                    "params": {
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key": APIKeys.QWEN_API_KEY,
                        "temperature": 0.2,
                        "extra_body": {"enable_thinking": False},
                    },
                },
            },
        },
        init_state_factory=lambda user_message: ImprovedCellState(
            original_question=user_message,
            query=user_message,
            optimized_questions=[],
            context="",
            individual_plans=[],
            generated_plan="",
            deep_research_findings={},
            hypothesis={},
            research_informed_plan="",
            final_enhanced_plan="",
        ).model_dump(),
        graph_factory=build_improved_graph,
        result_factory=lambda x: x.get("final_enhanced_plan", "") or x.get("research_informed_plan", "") or str(x),
    )

    @classmethod
    def get_usecase(cls, name: str) -> Usecase:
        for usecase in cls.__dict__.values():
            if isinstance(usecase, Usecase):
                if usecase.name == name:
                    return usecase
        raise ValueError(f"Usecase {name} not found")

    @classmethod
    def list_usecases(cls):
        ret = []
        for usecase in cls.__dict__.values():
            if isinstance(usecase, Usecase):
                ret.append(usecase)
        return ret
