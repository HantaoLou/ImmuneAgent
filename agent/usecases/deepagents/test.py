LOCAL_TOOLS = []
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt.tool_node import TOOL_CALL_ERROR_TEMPLATE

from common.factory import get_reasoning_model
from usecases.deepagents.graph import create_deep_agent
from usecases.deepagents.prompts import TOOL_EXEC_INSTRUCTION
from usecases.deepagents.tools import hil

if __name__ == "__main__":

    @tool(parse_docstring=True)
    def metabcr_data_preprocess(raw_data_path: str = "") -> str:
        """
        perform data preprocess for metabcr, and get final input file

        Args:
            raw_data_path: raw data path. Default to ""

        Returns:
            metabcr_input_file_path: metabcr input file path

        """
        print(f"input: {raw_data_path}")

        return raw_data_path + "-processed"

    @tool(parse_docstring=True)
    def metabcr_predict(input_path: str = "") -> str:
        """
        Perform metabcr predict on BCR seq against antigen

        Args:
            input_path: input path, default to ""


        Returns:
            metabcr_output_file_path: metabcr output file path
        """
        print(f"input: {input_path}")

        return "/c/d/e/f"

    LOCAL_TOOLS.extend([hil(metabcr_data_preprocess), hil(metabcr_predict)])
    s = {
        "messages": (
            [
                "pre process data for metabcr",
                "use metabcr to predict the functionality of the BCR against the antigen?",
            ]
        )
    }

    # 会先调用 metabcr_data_preprocess
    # 得到的结果作为 metabcr 的输入

    import asyncio

    from langgraph.checkpoint.memory import InMemorySaver

    from common.runner import GraphRunner

    config = {
        "configurable": {
            "thread_id": "1",
            "model_config": {
                "summarize_model": {
                    "provider": "OpenAI",
                    "model": "gpt-4.1",
                    "params": {
                        "temperature": 0,
                    },
                },
                "reasoning_model": {
                    "provider": "OpenAI",
                    "model": "gpt-4.1",
                    "params": {
                        "temperature": 0,
                    },
                },
            },
            "mcp_config": {"service_ids": []},
        }
    }
    model = get_reasoning_model(config)
    wf = create_deep_agent(
        model=model,
        tools=LOCAL_TOOLS,
        checkpointer=InMemorySaver(),
        instructions=TOOL_EXEC_INSTRUCTION,
    )
    runner = GraphRunner(wf).with_message_handler(lambda x: print(x.content, end=""))

    async def run():
        ret = await runner.run(s, config)
        while ret is not None:
            msg = input(ret)
            ret = await runner.resume(msg, config)
        return ret

    ret = asyncio.run(run())
# {"accept": true, "args": {"raw_data_path": "/user-input"}}
