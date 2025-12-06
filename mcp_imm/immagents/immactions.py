
import re
import subprocess
import asyncio
from metagpt.actions import Action
import pandas as pd


class ReadTable(Action):
    async def run(self, file_path):
        self.df = pd.read_excel(file_path)
        return self.df

class ProcTable(Action):
    async def run(self, df):
        df = df.dropna()  # Drop rows with NaN values
        df = df.replace([float('inf'), float('-inf')], 0)  # Replace inf values with 0
        return df

class AnalyzeTable(Action):
    async def run(self, df):
        stable_antibodies = df[df['expasy'] == 'pass']
        return stable_antibodies

class WriteTable(Action):
    async def run(self, df, file_path):
        df.to_csv(file_path, index=False)
        print(f"Selected antibodies saved to {file_path}")

class WriteSteps(Action):
    async def run(self, steps, file_path):
        with open(file_path, 'w') as f:
            f.write(steps)
        print(f"Analysis steps saved to {file_path}")


# @staticmethod
def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text

class SimpleWriteCode(Action):
    PROMPT_TEMPLATE: str = """
    Write a python function that can {instruction}.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """
    name: str = "SimpleWriteCode"

    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        code_text = parse_code(rsp)

        return code_text

    

class SimpleWriteTest(Action):
    PROMPT_TEMPLATE: str = """
    Context: {context}
    Write {k} unit tests using pytest for the given function, assuming you have imported it.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """

    name: str = "SimpleWriteTest"

    async def run(self, context: str, k: int = 3):
        prompt = self.PROMPT_TEMPLATE.format(context=context, k=k)

        rsp = await self._aask(prompt)

        code_text = parse_code(rsp)

        return code_text


class SimpleWriteReview(Action):
    PROMPT_TEMPLATE: str = """
    Context: {context}
    Review the test cases and provide one critical comments:
    """

    name: str = "SimpleWriteReview"

    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context)

        rsp = await self._aask(prompt)

        return rsp