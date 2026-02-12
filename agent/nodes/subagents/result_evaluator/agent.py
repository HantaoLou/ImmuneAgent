"""
CodeAct Agent Module
Extracted from Biomni framework (biomni/agent/a1.py)
"""

import re
from datetime import datetime
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import sys
import os

# Ensure the package directory is in the path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from executor import (
    clear_captured_plots,
    get_captured_plots,
    run_bash_script,
    run_python_repl,
    run_r_code,
    run_with_timeout,
)
from llm import get_llm


class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: str | None


class CodeActAgent:
    """CodeAct Agent that generates and executes code based on task descriptions.
    
    This agent follows the CodeAct paradigm:
    1. Generate: LLM generates code in <execute> tags
    2. Execute: Code is executed and results captured
    3. Observe: Results are returned to LLM in <observation> tags
    4. Iterate: Until LLM provides final answer in <solution> tags
    """

    def __init__(
        self,
        llm: str | None = None,
        source: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 600,
        temperature: float = 0.7,
        verbose: bool = False,
        agent_label: str = "Agent",
    ):
        """Initialize the CodeAct Agent.

        Args:
            llm: LLM model name to use
            source: Source provider (OpenAI, Anthropic, Custom, etc.)
            base_url: Base URL for custom model serving
            api_key: API key for the LLM
            timeout_seconds: Timeout for code execution in seconds
            temperature: Temperature setting for generation
            verbose: If True, print each step to stderr in real-time
            agent_label: Label prefix for verbose output (e.g. 'Solver 0')
        """
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose
        self.agent_label = agent_label
        self.llm = get_llm(
            model=llm,
            temperature=temperature,
            stop_sequences=["</execute>", "</solution>"],
            source=source,
            base_url=base_url,
            api_key=api_key,
        )
        
        self.system_prompt = ""
        self.log = []
        self._execution_results = []
        self.user_task = ""
        self._namespace = {}
        self._captured_plots = []
        
        # Configure the agent
        self.configure()

    def configure(self):
        """Configure the agent with system prompt and workflow."""
        
        # Generate system prompt
        self.system_prompt = self._generate_system_prompt()
        
        # Define the nodes
        def generate(state: AgentState) -> AgentState:
            # Add OpenAI-specific formatting reminders if using OpenAI models
            system_prompt = self.system_prompt
            if hasattr(self.llm, "model_name") and (
                "gpt" in str(self.llm.model_name).lower() or "openai" in str(type(self.llm)).lower()
            ):
                system_prompt += "\n\nIMPORTANT FOR GPT MODELS: You MUST use XML tags <execute> or <solution> in EVERY response. Do not use markdown code blocks (```) - use <execute> tags instead."

            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = self.llm.invoke(messages)

            # Normalize Responses API content blocks (list of dicts) into a plain string
            content = response.content
            if isinstance(content, list):
                # Concatenate textual parts; ignore tool_use or other non-text blocks
                text_parts: list[str] = []
                for block in content:
                    try:
                        if isinstance(block, dict):
                            btype = block.get("type")
                            if btype in ("text", "output_text", "redacted_text"):
                                part = block.get("text") or block.get("content") or ""
                                if isinstance(part, str):
                                    text_parts.append(part)
                    except Exception:
                        # Be conservative; skip malformed blocks
                        continue
                msg = "".join(text_parts)
            else:
                # Fallback to string conversion for legacy content
                msg = str(content)

            # Enhanced parsing for better OpenAI compatibility
            # Check for incomplete tags and fix them
            if "<execute>" in msg and "</execute>" not in msg:
                msg += "</execute>"
            if "<solution>" in msg and "</solution>" not in msg:
                msg += "</solution>"
            if "<think>" in msg and "</think>" not in msg:
                msg += "</think>"

            # More flexible pattern matching for different LLM styles
            think_match = re.search(r"<think>(.*?)</think>", msg, re.DOTALL | re.IGNORECASE)
            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL | re.IGNORECASE)
            answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL | re.IGNORECASE)

            # Alternative patterns for OpenAI models that might use different formatting
            if not execute_match:
                # Try to find code blocks that might be intended as execute blocks
                code_block_match = re.search(r"```(?:python|bash|r)?\s*(.*?)```", msg, re.DOTALL)
                if code_block_match and not answer_match:
                    # If we found a code block and no solution, treat it as execute
                    execute_match = code_block_match

            # Add the message to the state before checking for errors
            state["messages"].append(AIMessage(content=msg.strip()))

            if answer_match:
                state["next_step"] = "end"
            elif execute_match:
                state["next_step"] = "execute"
            elif think_match:
                state["next_step"] = "generate"
            else:
                print("parsing error...")

                error_count = sum(
                    1 for m in state["messages"] if isinstance(m, AIMessage) and "There are no tags" in m.content
                )

                if error_count >= 2:
                    # If we've already tried to correct the model twice, just end the conversation
                    print("Detected repeated parsing errors, ending conversation")
                    state["next_step"] = "end"
                    # Add a final message explaining the termination
                    state["messages"].append(
                        AIMessage(
                            content="Execution terminated due to repeated parsing errors. Please check your input and try again."
                        )
                    )
                else:
                    # Try to correct it
                    state["messages"].append(
                        HumanMessage(
                            content="Each response must include thinking process followed by either <execute> or <solution> tag. But there are no tags in the current response. Please follow the instruction, fix and regenerate the response again."
                        )
                    )
                    state["next_step"] = "generate"
            return state

        def execute(state: AgentState) -> AgentState:
            last_message = state["messages"][-1].content
            # Only add the closing tag if it's not already there
            if "<execute>" in last_message and "</execute>" not in last_message:
                last_message += "</execute>"

            execute_match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL)
            if execute_match:
                code = execute_match.group(1)

                # Set timeout duration
                timeout = self.timeout_seconds

                # Check if the code is R code
                if (
                    code.strip().startswith("#!R")
                    or code.strip().startswith("# R code")
                    or code.strip().startswith("# R script")
                ):
                    # Remove the R marker and run as R code
                    r_code = re.sub(r"^#!R|^# R code|^# R script", "", code, count=1).strip()
                    result = run_with_timeout(run_r_code, [r_code], timeout=timeout)
                # Check if the code is a Bash script or CLI command
                elif (
                    code.strip().startswith("#!BASH")
                    or code.strip().startswith("# Bash script")
                    or code.strip().startswith("#!CLI")
                ):
                    # Handle both Bash scripts and CLI commands with the same function
                    if code.strip().startswith("#!CLI"):
                        # For CLI commands, extract the command and run it as a simple bash script
                        cli_command = re.sub(r"^#!CLI", "", code, count=1).strip()
                        # Remove any newlines to ensure it's a single command
                        cli_command = cli_command.replace("\n", " ")
                        result = run_with_timeout(run_bash_script, [cli_command], timeout=timeout)
                    else:
                        # For Bash scripts, remove the marker and run as a bash script
                        bash_script = re.sub(r"^#!BASH|^# Bash script", "", code, count=1).strip()
                        result = run_with_timeout(run_bash_script, [bash_script], timeout=timeout)
                # Otherwise, run as Python code
                else:
                    # Clear any previous plots before execution
                    self._clear_execution_plots()
                    result = run_with_timeout(
                        run_python_repl, [code],
                        kwargs={"namespace": self._namespace, "captured_plots": self._captured_plots},
                        timeout=timeout,
                    )

                if len(result) > 10000:
                    result = (
                        "The output is too long to be added to context. Here are the first 10K characters...\n"
                        + result[:10000]
                    )

                # Get any plots that were generated during this execution
                execution_plots = []
                try:
                    current_plots = get_captured_plots(self._captured_plots)
                    execution_plots = current_plots.copy()
                except Exception as e:
                    print(f"Warning: Could not capture plots from execution: {e}")
                    execution_plots = []

                # Store the execution result with metadata
                execution_entry = {
                    "triggering_message": last_message,
                    "images": execution_plots,
                    "timestamp": datetime.now().isoformat(),
                }
                self._execution_results.append(execution_entry)

                observation = f"\n<observation>{result}</observation>"
                state["messages"].append(AIMessage(content=observation.strip()))

            return state

        def routing_function(
            state: AgentState,
        ) -> Literal["execute", "generate", "end"]:
            next_step = state.get("next_step")
            if next_step == "execute":
                return "execute"
            elif next_step == "generate":
                return "generate"
            elif next_step == "end":
                return "end"
            else:
                raise ValueError(f"Unexpected next_step: {next_step}")

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate", generate)
        workflow.add_node("execute", execute)

        # Add conditional edges
        workflow.add_conditional_edges(
            "generate",
            routing_function,
            path_map={"execute": "execute", "generate": "generate", "end": END},
        )
        workflow.add_edge("execute", "generate")
        workflow.add_edge(START, "generate")

        # Compile the workflow
        self.app = workflow.compile()
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer

    def _generate_system_prompt(self) -> str:
        """Generate the system prompt for the CodeAct agent."""
        
        prompt = """
You are a helpful assistant assigned with the task of problem-solving.
To achieve this, you will be using an interactive coding environment equipped with a variety of tool functions to assist you throughout the process.

Given a task, make a plan first. The plan should be a numbered list of steps that you will take to solve the task. Be specific and detailed.
Format your plan as a checklist with empty checkboxes like this:
1. [ ] First step
2. [ ] Second step
3. [ ] Third step

Follow the plan step by step. After completing each step, update the checklist by replacing the empty checkbox with a checkmark:
1. [✓] First step (completed)
2. [ ] Second step
3. [ ] Third step

If a step fails or needs modification, mark it with an X and explain why:
1. [✓] First step (completed)
2. [✗] Second step (failed because...)
3. [ ] Modified second step
4. [ ] Third step

Always show the updated plan after each step so the user can track progress.

At each turn, you should first provide your thinking and reasoning given the conversation history.
After that, you have two options:

1) Interact with a programming environment and receive the corresponding output within <observe></observe>. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>. IMPORTANT: You must end the code block with </execute> tag.
   - For Python code (default): <execute> print("Hello World!") </execute>
   - For R code: <execute> #!R\nlibrary(ggplot2)\nprint("Hello from R") </execute>
   - For Bash scripts and commands: <execute> #!BASH\necho "Hello from Bash"\nls -la </execute>
   - For CLI softwares, use Bash scripts.

2) When you think it is ready, directly provide a solution that adheres to the required format for the given task to the user. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>. IMPORTANT: You must end the solution block with </solution> tag.

You have many chances to interact with the environment to receive the observation. So you can decompose your code into multiple steps.
Don't overcomplicate the code. Keep it simple and easy to understand.
When writing the code, please print out the steps and results in a clear and concise manner, like a research log.

For R code, use the #!R marker at the beginning of your code block to indicate it's R code.
For Bash scripts and commands, use the #!BASH marker at the beginning of your code block. This allows for both simple commands and multi-line scripts with variables, loops, conditionals, loops, and other Bash features.

In each response, you must include EITHER <execute> or <solution> tag. Not both at the same time. Do not respond with messages without any tags. No empty messages.
"""
        return prompt

    def _clear_execution_plots(self):
        """Clear execution plots before new execution."""
        try:
            clear_captured_plots(self._captured_plots)
        except Exception as e:
            print(f"Warning: Could not clear execution plots: {e}")

    def _verbose_log(self, step: int, message):
        """Print a step summary to stderr when verbose mode is on."""
        if not self.verbose:
            return
        import time
        ts = time.strftime("%H:%M:%S")
        content = message.content
        if isinstance(message, HumanMessage):
            tag = "HUMAN"
        elif isinstance(message, AIMessage):
            if "<execute>" in content:
                tag = "CODE"
            elif "<solution>" in content:
                tag = "SOLUTION"
            elif "<observation>" in content:
                tag = "OBSERVE"
            else:
                tag = "AI"
        else:
            tag = "SYS"
        # Truncate long content for readability
        preview = content.replace("\n", " ")[:200]
        sys.stderr.write(f"[{ts}] [{self.agent_label}] step={step} ({tag}) {preview}\n")
        sys.stderr.flush()

    def go(self, prompt: str):
        """Execute the agent with the given prompt.

        Args:
            prompt: The user's query/task description

        Returns:
            Tuple of (log, final_message_content)
        """
        self.user_task = prompt

        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []

        # Store the final conversation state
        final_state = None
        step = 0

        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            step += 1
            out = self._pretty_print(message)
            self.log.append(out)
            self._verbose_log(step, message)
            final_state = s

        # Store the conversation state
        self._conversation_state = final_state

        return self.log, message.content

    def _pretty_print(self, message) -> str:
        """Pretty print a message."""
        if isinstance(message, HumanMessage):
            return f"👤 Human: {message.content}"
        elif isinstance(message, AIMessage):
            return f"🤖 AI: {message.content}"
        elif isinstance(message, SystemMessage):
            return f"⚙️ System: {message.content}"
        else:
            return f"📝 {message.content}"
