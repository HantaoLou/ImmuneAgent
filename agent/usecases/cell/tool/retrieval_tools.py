import os
from typing import List

from kb.vectorstore.store import get_vector_store
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from common.util.retrieval_utils import (
    clean_document_content,
    is_academic_noise,
    model_filter_and_rank,
)


@tool
def retrieve(query: List[str], config: RunnableConfig) -> str:
    """优化的文档检索 - 减少线程竞争"""
    try:
        vector_store = get_vector_store("immune_test")
        all_docs = []
        seen_docs = set()

        # 顺序执行检索，避免向量数据库的并发冲突
        results = []
        for q in query:
            try:
                # 使用带评分的搜索方法
                docs_with_scores = vector_store.similarity_search_with_score(
                    q, k=30, score_threshold=0.6
                )
                print(f"查询 '{q[:50]}...' 检索到 {len(docs_with_scores)} 个文档")
                results.append(docs_with_scores)
            except Exception as e:
                print(f"检索查询失败: {e}")
                results.append([])

        # 合并结果并去重
        total_retrieved = 0
        for docs_with_scores in results:
            total_retrieved += len(docs_with_scores)

            for doc, score in docs_with_scores:  # 正确解包元组：(Document, score)
                doc_hash = hash(doc.page_content)
                if doc_hash not in seen_docs:
                    seen_docs.add(doc_hash)
                    all_docs.append(doc)

        # 内容清理和过滤
        scored_docs = []
        for doc in all_docs:
            content = clean_document_content(doc.page_content.strip())
            if not is_academic_noise(content):
                doc.page_content = content
                scored_docs.append(doc)

        print(
            f"总检索 {total_retrieved} 个文档，去重后 {len(all_docs)} 个，清理后 {len(scored_docs)} 个"
        )

        # 模型筛选和排序
        if len(scored_docs) > 0:
            top_docs = model_filter_and_rank(scored_docs, query, config)
            print(f"模型筛选后获得 {len(top_docs)} 个高质量文档")
        else:
            top_docs = []

        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in top_docs])
        return context
    except Exception as e:
        print(f"检索查询失败: {e}")
        return ""


@tool
def web_search_node(query: List[str]) -> str:
    """学术搜索工具"""
    # 设置TAVILY_API_KEY
    if not os.environ.get("TAVILY_API_KEY"):
        from config.api_keys import APIKeys
        os.environ["TAVILY_API_KEY"] = APIKeys.TAVILY_API_KEY

    try:
        from langchain_tavily import TavilySearch

        all_results = []  # 存储所有结果
        seen_urls = set()  # 用于去重的URL集合

        for q in query:
            academic_search = TavilySearch(
                max_results=15,
                include_answer=True,
                include_raw_content=True,
                search_depth="advanced",
                include_domains=[
                    "pubmed.ncbi.nlm.nih.gov",
                    "scholar.google.com",
                    "arxiv.org",
                    "biorxiv.org",
                    "nature.com",
                    "science.org",
                    "cell.com",
                    "pnas.org",
                    "ncbi.nlm.nih.gov",
                    "doi.org",
                    "researchgate.net",
                    "semanticscholar.org",
                ],
            )
            results = academic_search.invoke({"query": q})

            if "results" in results:
                # 每组按分数排序，取前5个
                sorted_results = sorted(
                    results["results"], key=lambda x: x.get("score", 0), reverse=True
                )[:5]

                # 基于URL去重
                for result in sorted_results:
                    url = result.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(result)

        # 所有结果按分数排序，取前10条
        final_results = sorted(
            all_results, key=lambda x: x.get("score", 0), reverse=True
        )[:10]

        # 提取content并清理内容
        cleaned_contents = []
        for result in final_results:
            raw_content = result.get("content", "")
            if raw_content:
                cleaned_content = _clean_search_content(raw_content)
                if cleaned_content:  # 只添加非空的清理后内容
                    cleaned_contents.append(cleaned_content)

        combined_content = "\n\n".join(cleaned_contents)

        print(f"搜索查询完成，清理后获得 {len(cleaned_contents)} 个内容片段")

        return combined_content
    except Exception as e:
        print(f"搜索查询失败: {e}")
        return ""


# 创建工具列表和节点
tools = [retrieve, web_search_node]
tool_node = ToolNode(tools)


def _clean_search_content(content: str) -> str:
    """通用的搜索内容清理器，去除导航链接和无用信息"""
    if not content:
        return ""

    import re

    # 1. 通用的网页导航模式
    general_patterns = [
        # 常见导航元素
        r"Skip to Main Content.*?\n",
        r"Skip to main.*?\n",
        r"Jump to.*?\n",
        r"Go to.*?\n",
        # 登录相关
        r"Log in.*?\n",
        r"Sign in.*?\n",
        r"Login.*?\n",
        r"Register.*?\n",
        # 搜索相关
        r"Search:.*?\n",
        r"Advanced\s+search.*?\n",
        # 页面元数据
        r"Last update.*?\n",
        r"Updated.*?\n",
        r"Published.*?\n",
        # 社交媒体
        r"Follow us.*?\n",
        r"Share.*?\n",
        # 版权信息
        r"Copyright.*?\n",
        r"© \d{4}.*?\n",
    ]

    # 2. 通用的链接导航模式（不针对特定网站）
    link_patterns = [
        # 独立的链接行模式：* [文本](链接)
        r"^\s*\*\s*\[.*?\]\(.*?\)\s*$",
        # 常见的功能性链接模式
        r"\*\s*\[Search in.*?\]\(.*?\)",
        r"\*\s*\[Add to.*?\]\(.*?\)",
        r"\*\s*\[View in.*?\]\(.*?\)",
        r"\*\s*\[Download.*?\]\(.*?\)",
        r"\*\s*\[Export.*?\]\(.*?\)",
        r"\*\s*\[Save.*?\]\(.*?\)",
        # DOI和引用链接
        r"DOI:.*?(?=\n|\s)",
        r"Cite this.*?\n",
        r"Reference.*?\n",
    ]

    # 3. 政府网站模式
    gov_patterns = [
        r"U\.S\. flag.*?\n",
        r"An official website.*?\n",
        r"Here\'s how you know.*?\n",
        r"The \.gov means.*?\n",
        r"The site is secure.*?\n",
    ]

    # 4. 重复链接模式（通用）
    repetitive_patterns = [
        # 连续3个以上相似的链接
        r"(\*\s+\[.*?\]\(.*?\)\s*){3,}",
        # 重复的按钮或链接
        r"(\[.*?\]\(.*?\)\s*){4,}",
        # 重复的菜单项
        r"(^\s*\*\s+.*?\n){5,}",
    ]

    # 应用所有过滤模式
    all_patterns = general_patterns + link_patterns + gov_patterns + repetitive_patterns

    for pattern in all_patterns:
        content = re.sub(
            pattern, "", content, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE
        )

    # 5. 清理格式
    # 移除多余的空行
    content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

    # 移除行首行尾空格
    content = re.sub(r"^\s+|\s+$", "", content, flags=re.MULTILINE)

    # 移除过多的空格
    content = re.sub(r" {2,}", " ", content)

    # 6. 内容质量检查
    cleaned_content = content.strip()

    # 如果内容太短或主要是符号，可能过度过滤了
    if len(cleaned_content) < 100 or len(re.sub(r"[^\w\s]", "", cleaned_content)) < 50:
        # 返回原始内容的摘要
        return content[:500] if content else ""

    return cleaned_content
