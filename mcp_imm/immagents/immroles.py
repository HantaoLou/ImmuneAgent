import re
import subprocess
import asyncio
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import os
from io import StringIO

from metagpt.actions import Action
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message
from metagpt.logs import logger

# from immagents.immactions import *




# =============================================================================
# Table analyzer
# =============================================================================

class Analyzer(Role):
    def __init__(self, name="Analyzer", profile="Analyze antibody data", goal="Analyze and process antibody data", constraints=""):
        super().__init__()
        self.name = name
        self.profile = profile
        self.goal = goal
        self.constraints = constraints
        self.set_actions([ReadTable(), ProcTable(), AnalyzeTable(), SelectRows(), SaveTable(), WriteSteps()])
        self._watch([ReadTable, ProcTable, AnalyzeTable, SelectRows, SaveTable, WriteSteps])

    async def run(self, prompt, file_paths):
        steps = "Steps:\n"
        steps += f"Reading file from {file_paths['results']}\n"
        df = await self.actions[0].run(file_paths["results"])
        self.rc.memory.add(Message(content=f"Read file from {file_paths['results']}\n"))

        steps += "\nProcessing table to remove NaN/inf values and unnecessary columns\n"
        proc_df = await self.actions[1].run(df)
        # self.rc.memory.add(Message(content="Processed table to remove NaN/inf values and unnecessary columns\n"))

        steps += "\nAnalyzing table for antibodies\n"
        # context = self.get_memories()
        context = ""
        response = await self.actions[2].run(proc_df, prompt, context)
        self.rc.memory.add(Message(content=f"Analyzed table: {proc_df}\n\n My question: {prompt}\n\n Your response: {response}\n"))

        steps += "\nSelecting top rows\n"
        context = self.get_memories()
        selected_ids = await self.actions[3].run(proc_df, context)
        self.rc.memory.add(Message(content=f"Selected antibodies include: {selected_ids}"))

        steps += f"\nSaving selected antibodies to {file_paths['selected']}\n"
        await self.actions[4].run(df[df['clone_id'].isin(selected_ids)], file_paths["selected"])
        # self.rc.memory.add(Message(content=f"Selected antibodies saved in {file_paths['selected']}"))

        steps += f"\nWriting analysis steps to {file_paths['analysis']}\n"
        context = self.get_memories()
        reason = await self.actions[5].run(steps + "\n\nThoughts:" + response, context, file_paths["analysis"])
        self.rc.memory.add(Message(content=f"Written analysis steps to {file_paths['analysis']}"))

        return self.get_memories()


class ReadTable(Action):
    async def run(self, file_path):
        self.df = pd.read_excel(file_path)
        return self.df

class ProcTable(Action):
    async def run(self, df):
        exclude_columns = ['Heavy','Light']
        # df = df.dropna()  # Drop rows with NaN values
        # df = df.replace([float('inf'), float('-inf')], 'NaN')  # Replace inf values with 0
        proc_df = df.replace([None, float('nan')], 'NaN')  # Replace inf values with 0
        proc_df = proc_df.drop(columns=exclude_columns)
        return proc_df

class AnalyzeTable(Action):
    PROMPT_TEMPLATE: str = """
    Context: {context}\n
    Based on the table:
    >>start of table<<\n
    {table}\n
    >>end of table<<\n
    Query:
    {prompt}\n
    and tell me your chain of thoughts:
    """
    async def run(self, df, prompt, context):
        analysis_prompt = f"{prompt}\n\n{df.to_string(index=False)}"
        response = await self._aask(self.PROMPT_TEMPLATE.format(context=context, table=df, prompt=analysis_prompt))
        print(f"Analysis response: {response}")
        return response

class SelectRows(Action):
    PROMPT_TEMPLATE: str = """
    Based on the context:\n
    {context}\n
    Can you write the {k} selected antibodies as previously queried from the table according to the analysis results.
    Please write in this format: ```NKx_xxx,INF-xxx(xxx),IGHVx-xxx,IGKVx-xxx,IGHJx-xxx,IGKJx-xxx\nNKx_xxx,INF-xxx(xxx),IGHVx-xxx,IGKVx-xxx,IGHJx-xxx,IGKJx-xxx\n...``` with NO index or other texts,
    WITH NO INDEX or OTHER TEXT! your selected rows of antibodies are:
    """
    async def run(self, df, context, select_num=5):
        columns_name = ["clone_id", "mAb", "v_gene_Heavy", "v_gene_Light", "j_gene_Heavy", "j_gene_Light"]
        selected_ids = []
        check_prompt = ""
        while len(selected_ids) < select_num: 
            # selected_rows = await self._aask(check_prompt+self.PROMPT_TEMPLATE.format(context=df.to_string(index=False), k=select_num))
            selected_rows = await self._aask(self.PROMPT_TEMPLATE.format(context=context, k=select_num)+check_prompt)
            selected_rows = selected_rows.strip().split("\n")
            proposed_ids = [row.split(",")[0] for row in selected_rows]
            selected_df = df[df[columns_name[0]].isin(proposed_ids)]
            selected_ids = selected_df[columns_name[0]].tolist()
            if len(selected_ids) < select_num:
                proposed_ids = df[df[columns_name[0]].isin(selected_ids)]
                inexist_ids = set(proposed_ids) - set(selected_ids)
                check_prompt = f"\nYou have selected some rows:{inexist_ids} that do not exist in the table, please select again."
        return selected_ids

class SaveTable(Action):
    async def run(self, df, file_path):
        df.to_csv(file_path, index=False)
        print(f"Selected antibodies saved to {file_path}")

class WriteSteps(Action):
    PROMPT_TEMPLATE: str = """
    Context: {context}\n
    Can you write the reason for each of your selected antibody?
    """
    async def run(self, steps,context, file_path):
        steps += "Reasons for the selected antibodies:\n"
        reason = await self._aask(self.PROMPT_TEMPLATE.format(context=context))
        steps += reason
        with open(file_path, 'w') as f:
            f.write(steps)
        print(f"Analysis steps saved to {file_path}")
        return reason





# =============================================================================
# test functions
# =============================================================================

# =============================================================================
# InvoiceOCR assistant
# =============================================================================
class InvoiceOCRAssistant(Role):
    """Invoice OCR assistant, support OCR text recognition of invoice PDF, png, jpg, and zip files,
    generate a table for the payee, city, total amount, and invoicing date of the invoice,
    and ask questions for a single file based on the OCR recognition results of the invoice.

    Args:
        name: The name of the role.
        profile: The role profile description.
        goal: The goal of the role.
        constraints: Constraints or requirements for the role.
        language: The language in which the invoice table will be generated.
    """

    def __init__(
        self,
        name: str = "Stitch",
        profile: str = "Invoice OCR Assistant",
        goal: str = "OCR identifies invoice files and generates invoice main information table",
        constraints: str = "",
        language: str = "ch",
    ):
        super().__init__(name, profile, goal, constraints)
        self.set_actions([InvoiceOCR])
        self.language = language
        self.filename = ""
        self.origin_query = ""
        self.orc_data = None
        self._set_react_mode(react_mode="by_order")
    async def _act(self) -> Message:
        """Perform an action as determined by the role.

        Returns:
            A message containing the result of the action.
        """
        msg = self.rc.memory.get(k=1)[0]
        todo = self.rc.todo
        if isinstance(todo, InvoiceOCR):
            self.origin_query = msg.content
            file_path = msg.instruct_content.get("file_path")
            self.filename = file_path.name
            if not file_path:
                raise Exception("Invoice file not uploaded")

            resp = await todo.run(file_path)
            if len(resp) == 1:
                # Single file support for questioning based on OCR recognition results
                self.set_actions([GenerateTable, ReplyQuestion])
                self.orc_data = resp[0]
            else:
                self.set_actions([GenerateTable])

            self.rc.todo = None
            content = INVOICE_OCR_SUCCESS
        elif isinstance(todo, GenerateTable):
            ocr_results = msg.instruct_content
            resp = await todo.run(ocr_results, self.filename)

            # Convert list to Markdown format string
            df = pd.DataFrame(resp)
            markdown_table = df.to_markdown(index=False)
            content = f"{markdown_table}\n\n\n"
        else:
            resp = await todo.run(self.origin_query, self.orc_data)
            content = resp

        msg = Message(content=content, instruct_content=resp)
        self.rc.memory.add(msg)
        return msg



class GenerateTable(Action):
    """Action class for generating tables from OCR results.

    Args:
        name: The name of the action. Defaults to an empty string.
        language: The language used for the generated table. Defaults to "ch" (Chinese).

    """

    def __init__(self, name: str = "", language: str = "ch", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.language = language

    async def run(self, ocr_results: list, filename: str, *args, **kwargs) -> dict[str, str]:
        """Processes OCR results, extracts invoice information, generates a table, and saves it as an Excel file.

        Args:
            ocr_results: A list of OCR results obtained from invoice processing.
            filename: The name of the output Excel file.

        Returns:
            A dictionary containing the invoice information.

        """
        table_data = []
        pathname = INVOICE_OCR_TABLE_PATH
        pathname.mkdir(parents=True, exist_ok=True)

        for ocr_result in ocr_results:
            # Extract invoice OCR main information
            prompt = EXTRACT_OCR_MAIN_INFO_PROMPT.format(ocr_result=ocr_result, language=self.language)
            ocr_info = await self._aask(prompt=prompt)
            invoice_data = OutputParser.extract_struct(ocr_info, dict)
            if invoice_data:
                table_data.append(invoice_data)

        # Generate Excel file
        filename = f"{filename.split('.')[0]}.xlsx"
        full_filename = f"{pathname}/{filename}"
        df = pd.DataFrame(table_data)
        df.to_excel(full_filename, index=False)
        return table_data


class ReplyQuestion(Action):
    """Action class for generating replies to questions based on OCR results.

    Args:
        name: The name of the action. Defaults to an empty string.
        language: The language used for generating the reply. Defaults to "ch" (Chinese).

    """

    def __init__(self, name: str = "", language: str = "ch", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.language = language

    async def run(self, query: str, ocr_result: list, *args, **kwargs) -> str:
        """Reply to questions based on ocr results.

        Args:
            query: The question for which a reply is generated.
            ocr_result: A list of OCR results.

        Returns:
            A reply result of string type.
        """
        prompt = REPLY_OCR_QUESTION_PROMPT.format(query=query, ocr_result=ocr_result, language=self.language)
        resp = await self._aask(prompt=prompt)
        return resp
    
class InvoiceOCR(Action):
    """Action class for performing OCR on invoice files, including zip, PDF, png, and jpg files.

    Args:
        name: The name of the action. Defaults to an empty string.
        language: The language for OCR output. Defaults to "ch" (Chinese).

    """

    def __init__(self, name: str = "", *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    async def run(self, file_path: Path, *args, **kwargs) -> list:
        """Execute the action to identify invoice files through OCR.

        Args:
            file_path: The path to the input file.

        Returns:
            A list of OCR results.
        """
        file_ext = await self._check_file_type(file_path)

        if file_ext == ".zip":
            # OCR recognizes zip batch files
            unzip_path = await self._unzip(file_path)
            ocr_list = []
            for root, _, files in os.walk(unzip_path):
                for filename in files:
                    invoice_file_path = Path(root) / Path(filename)
                    # Identify files that match the type
                    if Path(filename).suffix in [".zip", ".pdf", ".png", ".jpg"]:
                        ocr_result = await self._ocr(str(invoice_file_path))
                        ocr_list.append(ocr_result)
            return ocr_list

        else:
            #  OCR identifies single file
            ocr_result = await self._ocr(file_path)
            return [ocr_result]

    @staticmethod
    async def _check_file_type(file_path: Path) -> str:
        """Check the file type of the given filename.

        Args:
            file_path: The path of the file.

        Returns:
            The file type based on FileExtensionType enum.

        Raises:
            Exception: If the file format is not zip, pdf, png, or jpg.
        """
        ext = file_path.suffix
        if ext not in [".zip", ".pdf", ".png", ".jpg"]:
            raise Exception("The invoice format is not zip, pdf, png, or jpg")

        return ext

    @staticmethod
    async def _unzip(file_path: Path) -> Path:
        """Unzip a file and return the path to the unzipped directory.

        Args:
            file_path: The path to the zip file.

        Returns:
            The path to the unzipped directory.
        """
        file_directory = file_path.parent / "unzip_invoices" / datetime.now().strftime("%Y%m%d%H%M%S")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                # Use CP437 to encode the file name, and then use GBK decoding to prevent Chinese garbled code
                relative_name = Path(zip_info.filename.encode("cp437").decode("gbk"))
                if relative_name.suffix:
                    full_filename = file_directory / relative_name
                    await File.write(full_filename.parent, relative_name.name, zip_ref.read(zip_info.filename))

        logger.info(f"unzip_path: {file_directory}")
        return file_directory

    @staticmethod
    async def _ocr(invoice_file_path: Path):
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=1)
        ocr_result = ocr.ocr(str(invoice_file_path), cls=True)
        return ocr_result


# =============================================================================
# =============================================================================

# class SimpleCoder(Role):
#     name: str = "Alice"
#     profile: str = "SimpleCoder"

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self._watch([UserRequirement])
#         self.set_actions([SimpleWriteCode])

# =============================================================================
# Code writing assistant
# =============================================================================



class SimpleCoder(Role):
    name: str = "Alice"
    profile: str = "SimpleCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleWriteCode])

    # async def _act(self) -> Message:
    #     logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
    #     todo = self.rc.todo  # todo will be SimpleWriteCode()

    #     msg = self.get_memories(k=1)[0]  # find the most recent messages
    #     code_text = await todo.run(msg.content)
    #     msg = Message(content=code_text, role=self.profile, cause_by=type(todo))
    #     return msg

class SimpleTester(Role):
    name: str = "Bob"
    profile: str = "SimpleTester"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleWriteTest])
        self._watch([SimpleWriteCode])
        # self._watch([SimpleWriteCode, SimpleWriteReview])  # feel free to try this too

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo

        # context = self.get_memories(k=1)[0].content # use the most recent memory as context
        context = self.get_memories()  # use all memories as context

        code_text = await todo.run(context, k=5)  # specify arguments
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))

        return msg

class SimpleReviewer(Role):
    name: str = "Charlie"
    profile: str = "SimpleReviewer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleWriteReview])
        self._watch([SimpleWriteTest])


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




# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================


IMMGPT_DATA_PATH = "/data/lht/immgpt/data"
ANALYZE_COLUMNS = {
    "sum_score": "The sum of the estimated neutralizing scores of the antibodies, higher is better.",
    "expasy": "The protein instability index test result for the antibodies, pass is better.",
    "leiden": "The Leiden test result for the antibodies,.",
    "total_energy": "The total energy of the antibodies, lower is better.",
    "H5N1_TE/H1N1_WI/H1N1_VI (bind)(experiment)": "The binding affinity of the antibodies with this antigen, higher is better."
}



def create_analyzing_prompt(
        analyze_column: dict=ANALYZE_COLUMNS,
        export_column: list[str]=None,
        action_prompt: str = "select the 5-top antibodies that you think have the highest potential to broadly against viruses?"
        ) -> str:
    analyzing_prompt = (
        f"Please provide a detailed analysis of the table with focusing on the following columns:\n"
        # f"In the file, please provide a simple analysis of the following columns:\n"
    )
    for key, value in analyze_column.items():
        analyzing_prompt += f"- {key}: {value}\n"
    analyzing_prompt += "\n"
    analyzing_prompt += (
        f"{action_prompt}?\n"
    )
    analyzing_prompt += "\n"
    if export_column:
        analyzing_prompt += f"and then export the selected rows with the columns: {export_column}\n"

    return analyzing_prompt

if __name__ == "__main__":
    # fire.Fire(main)
    # action_prompt = "can you sort the antibodies from the table file that you think have the highest potential to broadly against the viruses?"
    action_prompt = "Can you select the 5-top antibodies that you think have the highest potential to broadly against viruses?"

    result_file = "H5N1_first-batch/0307_first-batch_exp-results.xlsx"
    
    result_name = os.path.basename(result_file).split(".")[0]
    analysis_file = result_name.replace("results", "analysis.txt")
    selected_file = result_name.replace("results", "selected.csv")
    
    results_path = os.path.join(IMMGPT_DATA_PATH, result_file)
    analysis_path = os.path.join(os.path.dirname(results_path), "analysis")
    os.makedirs(analysis_path, exist_ok=True)
    selected_path = os.path.join(os.path.dirname(results_path), "selected")
    os.makedirs(selected_path, exist_ok=True)
    
    file_paths = {
        "results": results_path,
        "analysis": os.path.join(analysis_path, analysis_file),
        "selected": os.path.join(selected_path, selected_file)
    }
    query_idea=""
    query_idea += create_analyzing_prompt(action_prompt=action_prompt)
    n_round = 5
    print(query_idea)


    analyzer = Analyzer()
    response=asyncio.run(analyzer.run(query_idea,file_paths))
    print(response)