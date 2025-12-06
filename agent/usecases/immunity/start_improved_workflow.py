"""
Start script for testing the improved workflow with reordered stages.

This script demonstrates the new workflow order where research and hypothesis
generation happen BEFORE planning, ensuring scientific grounding.
"""

import asyncio
import argparse
import functools
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.runnables import RunnableConfig

from common.util.mcp_utils import mcp_tool_async
from usecases.immunity.config.immunity_config import get_runnable_config
from usecases.immunity.graph.planning_graph import (
    ImprovedCellState,
    build_improved_graph,
)


async def run_improved_workflow(
    query: str, 
    file_url: Optional[str] = None,
    config: RunnableConfig = None
) -> Dict[str, Any]:
    """
    Execute the improved workflow with a given query and optional file URL.

    Args:
        query: The immunology research question
        file_url: Optional HTTP/HTTPS URL to download initial file
        config: Optional configuration for the workflow

    Returns:
        Dictionary containing workflow results
    """
    print(f"\nQuery: {query}")
    if file_url:
        print(f"Initial File: {file_url}")

    # Build and compile the improved graph
    graph = build_improved_graph()
    compiled_graph = graph.compile()

    # Initialize state with the query
    initial_state = ImprovedCellState(
        original_question=query,
        query=query,
        optimized_questions=[],
        context="",
        individual_plans=[],
        generated_plan="",
        deep_research_findings={},
        hypothesis={},
        research_informed_plan="",
        final_enhanced_plan="",
        merged_csv_result_path="",  # 初始化为空字符串
    )
    
    # Download initial file if provided
    if file_url:
        print(f"Downloading initial file...")
        try:
            result = await mcp_tool_async(
                service_id="file_utils",
                tool_name="download_url",
                params={"args": {"url": file_url}}
            )
            
            # 解析结果，获取本地文件路径
            file_path = None
            result_dict = None
            
            # 处理字符串类型的结果（可能是JSON字符串）
            if isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                except json.JSONDecodeError:
                    # 如果不是JSON，可能是直接返回的路径
                    file_path = result
            # 处理字典类型的结果
            elif isinstance(result, dict):
                result_dict = result
            
            # 从解析后的字典中提取文件路径
            if result_dict:
                file_path = (
                    result_dict.get("file_path") or 
                    result_dict.get("path") or 
                    result_dict.get("output_path") or
                    result_dict.get("output_file")
                )
            
            if file_path:
                # 检查文件是否为 xlsx 格式，如果是则转换为 CSV
                file_path_obj = Path(file_path)
                if file_path_obj.suffix.lower() in ['.xlsx', '.xls']:
                    print(f"Converting Excel to CSV...")
                    try:
                        # 根据文件扩展名选择正确的工具名
                        tool_name = "convert_xlsx_to_csv" if file_path_obj.suffix.lower() == '.xlsx' else "convert_xls_to_csv"
                        
                        # 调用转换工具
                        convert_result = await mcp_tool_async(
                            service_id="file_utils",
                            tool_name=tool_name,
                            params={
                                "args": {
                                    "input_file": file_path,
                                    "output_file": None  # 让服务自动生成
                                }
                            }
                        )
                        
                        # 解析转换结果，获取 CSV 文件路径
                        csv_path = None
                        if isinstance(convert_result, str):
                            try:
                                convert_dict = json.loads(convert_result)
                                csv_path = (
                                    convert_dict.get("file_path") or 
                                    convert_dict.get("path") or 
                                    convert_dict.get("output_path") or
                                    convert_dict.get("output_file")
                                )
                            except json.JSONDecodeError:
                                csv_path = convert_result
                        elif isinstance(convert_result, dict):
                            csv_path = (
                                convert_result.get("file_path") or 
                                convert_result.get("path") or 
                                convert_result.get("output_path") or
                                convert_result.get("output_file")
                            )
                        
                        if csv_path:
                            file_path = csv_path
                            print(f"File converted to CSV: {csv_path}")
                        else:
                            print(f"Warning: Could not extract CSV path from conversion result, using original file")
                    except Exception as e:
                        print(f"Warning: Excel to CSV conversion failed: {e}, using original file")
                
                initial_state.merged_csv_result_path = file_path
                print(f"File downloaded: {file_path}")
            else:
                print(f"Warning: Could not extract file path from download result")
        except Exception as e:
            print(f"Error: File download failed: {e}")
            # Continue workflow even if download fails
    
    # 添加UUID确保多线程环境下的唯一性
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    config = get_runnable_config(thread_id=timestamp)

    try:
        # Execute the workflow
        result = await compiled_graph.ainvoke(initial_state.model_dump(), config=config)
        print("\nWorkflow completed successfully")

        return {
            "success": True,
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"\nError: Workflow execution failed: {e}")

        return {
            "success": False,
            "query": query,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def main():
    """
    Main function to test the improved workflow.
    
    Supports command-line arguments:
    - --query: User query (if provided, test_queries will be ignored)
    - --file_url: Initial file URL to download
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Improved ImmuneAgent Workflow")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="User query (if provided, test_queries will be ignored)"
    )
    parser.add_argument(
        "--file_url",
        type=str,
        default=None,
        help="Initial file URL (HTTP/HTTPS) to download and use as input"
    )
    args = parser.parse_args()

    # If query is provided, use it; otherwise use test_queries
    if args.query:
        # Run async workflow in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                run_improved_workflow(query=args.query, file_url=args.file_url)
            )
            if not result.get("success"):
                print(f"Error: {result.get('error', 'Unknown error')}")
        finally:
            loop.close()
        return

    # Test queries (仅在未提供query参数时使用)
    test_queries = [
        "1. 使用Igblast的analyze_vdj_batch工具；2. 使用metaBCR的metabcr工具；3. 使用flu的所有工具；4. 使用af3的两个工具； 5. 使用bioinformatics的一系列工具",
        # "please design a computational method to identifiy broadly neutralizing antibodies against H5N1.",
        # "What is the phenotype of broadly reactive B cells in response to vaccination?",
        # "What structural features differentiate broadly neutralizing antibodies from those with narrow specificity?",
        # "What molecular programs determine why some germinal center B cells become long-lived plasma cells while others become memory B cells?",
        # "How do atypical memory B cells contribute to protection vs. immune dysfunction in chronic infection and vaccination?",
        # "Can we predict which naïve B cells have the intrinsic potential to evolve into broadly neutralizing antibodies?",
        # "What is the precise role of T follicular regulatory cells (Tfr) in balancing tolerance and affinity maturation?",
        # "How do tissue-resident memory T cells (Trm) communicate with circulating memory compartments to coordinate systemic immunity?",
        # "Why do some individuals generate highly diverse TCR repertoires against the same pathogen, while others generate narrow, biased responses?",
        # "How do clonal competition and selection dynamics shape the outcome of adaptive immune responses at single-cell resolution?",
        # "What are the immune correlates of sterilizing immunity against malaria — and can they be consistently induced by vaccination?",
        # "Why do broadly neutralizing antibodies arise naturally in only a subset of HIV-infected individuals, and can this process be accelerated?",
        # "How does persistent antigen exposure during chronic infections reprogram B cell differentiation pathways?",
        # "What are the early immunological determinants of long COVID, and are they distinct from responses to acute SARS-CoV-2 infection?",
        # "How does the gut microbiome shape systemic antiviral immunity beyond mucosal IgA production?",
        # "Can we design universal influenza vaccines that generate stable, cross-reactive T cell immunity without driving exhaustion?",
        # "How does innate sensing of Mycobacterium tuberculosis influence the later establishment of protective vs. non-protective T cell memory?",
        # "What is the developmental origin of autoreactive B cells in systemic lupus erythematosus — germinal center vs. extrafollicular?",
        # "How do breakdowns in germinal center checkpoint regulation contribute to autoantibody generation?",
        # "Why do some self-reactive T cells escape thymic deletion yet remain quiescent until disease onset?",
        # "How do metabolic states of Tregs influence their capacity to suppress autoimmunity in inflamed tissues?",
        # "What is the role of tissue-resident memory T cells in maintaining chronic autoimmunity (e.g., MS, type 1 diabetes)?",
        # "Can immune repertoire dynamics (BCR/TCR evolution) predict autoimmune flares before they are clinically detectable?"
        # "What is the transcriptional and epigenetic phenotype of RSV broadly neutralizing antibody–producing B cells in human bone marrow?",
        # "How do atypical memory B cells differ in their ability to re-enter germinal centers after influenza vaccination?",
        # "What are the metabolic signatures that distinguish long-lived plasma cells from short-lived ones in human bone marrow niches?",
        # "Which transcription factors govern the bifurcation of germinal center B cells into memory B cells vs. plasma cells?",
        # "Which B cell subsets are enriched for broadly neutralizing antibody precursors in chronic HIV infection compared to acute infection?",
        # "What is the single-cell transcriptomic profile of protective T cell responses against malaria sporozoites in vaccinated individuals?",
        # "How does persistent hepatitis B virus antigen exposure reprogram the transcriptional state of exhausted B cells?",
        # "What are the spatial interactions between innate immune cells and B cells in early Mycobacterium tuberculosis lung granulomas?",
        # "Which cytokine signatures in acute SARS-CoV-2 infection predict the development of long COVID?",
        # "What is the clonal trajectory of IgA⁺ B cells in the gut during enteric infection, and how is it shaped by the microbiome?",
        # "How do influenza vaccines drive differential selection of cross-reactive vs. strain-specific T cell clonotypes?",
        # "How do specific glycosylation patterns at Asn297 in the Fc region of IgG antibodies influence their ability to mediate antibody-dependent cellular cytotoxicity against intracellular pathogens?",
        # "What structural features distinguish polyreactive broadly neutralizing antibodies targeting conserved viral epitopes from autoreactive antibodies in systemic autoimmunity?",
        # "How do somatic hypermutations in framework regions versus complementarity-determining regions differentially impact antibody thermostability and neutralization breadth?",
        # "What are the transcriptional and epigenetic signatures that distinguish early memory B cell precursors from germinal center B cells during acute viral infections?",
        # "How do tissue-resident memory B cells in mucosal surfaces differ in their BCR repertoire and activation thresholds compared to circulating memory B cells following respiratory viral infections?",
        # "What are the molecular mechanisms governing VDJ recombination bias in neonatal B cells that limit their ability to generate broadly neutralizing antibodies against encapsulated bacteria?",
        # "What are the spatiotemporal dynamics of T follicular helper cell interactions with B cells expressing different affinity BCRs during affinity maturation in response to protein subunit vaccines?",
        # "How do follicular dendritic cells modulate antigen retention and presentation differently for viral glycoproteins versus bacterial polysaccharides in germinal centers?",
        # "What metabolic checkpoints regulate the light zone to dark zone transition of B cells undergoing clonal expansion in chronic parasitic infections?",
        # "What phenotypic markers and functional characteristics define atypical memory B cells that accumulate during chronic malaria infection and impair vaccine responses?",
        # "How do regulatory B cells modulate CD8+ T cell exhaustion through IL-10 and TGF-β production during chronic viral infections like hepatitis B?",
        # "What are the BCR signaling thresholds that distinguish anergic B cells from ignorant B cells in the context of HIV envelope protein immunization?",
        # "ow can bispecific antibody formats be optimized to simultaneously neutralize viral entry and recruit effector cells for clearance of HIV-infected reservoir cells?",
        # "What computational approaches can predict antibody developability issues (aggregation, immunogenicity) for therapeutic antibodies targeting bacterial toxins?",
        # "What are the early transcriptomic signatures in circulating B cells that predict durable antibody responses to seasonal influenza vaccination in elderly populations?",
        # "How do pre-existing cross-reactive memory B cell populations influence the immunodominance hierarchy of antibody responses to variant SARS-CoV-2 strains?",
        # "What is the transcriptional phenotype of autoreactive B cells producing anti-dsDNA antibodies in systemic lupus erythematosus?",
        # "How do germinal center T follicular regulatory cells fail to suppress autoreactive B cell clones in lupus-prone mice?",
        # "What are the early transcriptomic and epigenetic changes in self-reactive T cells escaping thymic deletion in type 1 diabetes?",
        # "What are the BCR repertoire features of autoreactive B cells recognizing citrullinated antigens in rheumatoid arthritis?",
        # "Can single-cell immune profiling identify predictive transcriptional states in B or T cells that precede autoimmune flares in lupus patients?"
        # "please design a computational method to identifiy broadly neutralizing antibodies against H5N1.",
        # "please design a computational method to identifiy broadly neutralizing antibodies against SARS-CoV2.",
        # "how to find the transcriptional phenotype of b cell reservoir for broadly neutralizing antibodies against Influenza, RSV and SARS-CoV-2?",
        # "what is the structural characteristics of broadly neutralizing antibodies against SARS-CoV-2 and Influenza?"
    ]

    # Define sync wrapper function for thread pool execution
    def run_workflow_sync(query_info):
        """Sync wrapper function to run async workflow in thread pool"""
        i, query = query_info
        # Run async workflow in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_improved_workflow(query=query, file_url=None))
            return result
        finally:
            loop.close()

    # Execute tasks with thread pool
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Prepare task data: (index, query)
        query_tasks = [(i + 1, query) for i, query in enumerate(test_queries)]

        # Submit all tasks and wait for completion
        futures = [executor.submit(run_workflow_sync, task) for task in query_tasks]

        # Wait for all tasks to complete
        for future in futures:
            try:
                result = future.result()
                if not result.get("success"):
                    print(f"Error: Task failed - {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"Error: Task execution exception: {e}")


if __name__ == "__main__":
    # Run the main function
    main()
