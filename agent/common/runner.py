"""
common logics of running a graph
"""

import operator
from typing import Any, Callable, Optional, Union

from langchain_core.messages import AIMessageChunk
from langgraph.config import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt, StreamMode, interrupt

type ValueHandler = Callable[[dict[str, Any]], None]

type MessageHandler = Callable[[AIMessageChunk], None]


class GraphRunner:
    """
    提供 Graph stream 接口的上层封装，只需要声明中断处理器和 messages/values 回调，不需要自己编写 stream 逻辑
    """

    @staticmethod
    def _default_message_handler(_: AIMessageChunk):
        pass

    @staticmethod
    def _default_value_handler(_: dict[str, Any]):
        pass

    def __init__(self, graph: CompiledStateGraph):
        self.graph = graph
        self.message_handler = GraphRunner._default_message_handler
        self.value_handler = GraphRunner._default_value_handler
        self.terminated = False

    def with_message_handler(self, handler: MessageHandler) -> "GraphRunner":
        """
        注册一个回调，用于处理 LLM 返回的每个 Token。适合 AI Chat Bot 接口的场景
        """
        self.message_handler = handler
        return self

    def with_value_handler(self, handler: ValueHandler) -> "GraphRunner":
        """
        注册一个回调，用于处理 Graph 中每个节点生成的 State 更新值。适合用于获取结构化的结果。
        """
        self.value_handler = handler
        return self

    def stop(self):
        self.terminated = True
        self.graph = None

    def get_state(self, config: RunnableConfig) -> dict[str, Any]:
        return self.graph.get_state(config).values

    async def run(
        self, data: dict[str, Any], rc: RunnableConfig
    ) -> Optional[Interrupt]:
        """
        首次运行 Graph。

        :param rc: runnable config
        :return: 如果发生中断，返回中断的值。如果图结束运行了，反返回 None
        """
        # 订阅 messages / values / updates，确保可以捕获到中断
        stream = self.graph.astream(
            data, config=rc, stream_mode=["messages", "values", "updates"]
        )
        return await self._stream(stream)

    async def resume(
        self, data: dict[str, Any], rc: RunnableConfig
    ) -> Optional[Interrupt]:
        """
        中断后继续运行 Graph

        :param data: 中断处理器返回的值，用于传递给 Graph
        """
        if self.terminated:
            raise ValueError("GraphRunner has been terminated, cannot resume")
        # 订阅 messages / values / updates，确保可以捕获到中断
        stream = self.graph.astream(
            Command(resume=data),
            config=rc,
            stream_mode=["messages", "values", "updates"],
        )
        return await self._stream(stream)

    async def _stream(self, stream) -> Union[Interrupt, None]:
        async for item_type, item in stream:
            item_type: StreamMode
            # 无论在何种 stream mode 下，如果出现 __interrupt__，都应当立即返回
            if isinstance(item, dict) and "__interrupt__" in item:
                if len(item["__interrupt__"]) == 0:
                    return "unknown interrupt"

                try:
                    _int: Interrupt = item["__interrupt__"][0]
                except Exception:
                    # 兜底：有些实现可能直接返回 Interrupt 或列表
                    val = item["__interrupt__"]
                    _int = val[0] if isinstance(val, (list, tuple)) else val
                return _int

            if item_type == "values":
                self.value_handler(item)
            elif item_type == "messages":
                # 处理消息 token
                if (
                    isinstance(item, (list, tuple))
                    and item
                    and isinstance(item[0], AIMessageChunk)
                ):
                    self.message_handler(item[0])
        return None


# 示例代码
if __name__ == "__main__":
    # test graph runnder
    from langchain_ollama import ChatOllama
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START
    from typing_extensions import Annotated, TypedDict

    class State(TypedDict):
        """State"""

        history: Annotated[list[str], ..., operator.add]

    llm = ChatOllama(model="llama2:latest")

    def human_step():
        """
        用户输入
        """
        res = interrupt({"message": "$>"})
        return {"history": [res["content"]]}

    def machine_step(s: State):
        """
        LLM 回复
        """
        res = llm.invoke(input=s["history"][-1])
        return {"history": [res.content]}

    test_graph: StateGraph = (
        StateGraph()
        .add_node(human_step)
        .add_node(machine_step)
        .add_edge(START, human_step.__name__)
        .add_edge(human_step.__name__, machine_step.__name__)
        .add_edge(machine_step.__name__, END)
    )

    runner = (
        GraphRunner(test_graph.compile(checkpointer=InMemorySaver()))
        .with_message_handler(lambda msg: print(msg.content, end=""))
        .with_value_handler(print)
    )
    c = {"thread_id": "1"}

    async def run():
        """运行一个图，直到结束"""

        async def handle_int(i: Interrupt):
            """简单的中断处理器，从 stdin 读取输入"""
            ret = input(i.value)
            return {"content": ret}

        ret = await runner.run({}, c)
        # 在循环中 resume 直到 Graph 结束
        while ret is not None:
            ret = await runner.resume(handle_int(ret), rc=c)

    import asyncio

    asyncio.run(run())
