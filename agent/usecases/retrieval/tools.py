import os
from typing import List

from diskcache import Cache
from kb.vectorstore.store import (
    KEY_SRC,
    QdrantParentDocumentRetriever,
    get_vector_store,
)
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from common.factory import get_summarize_model
from common.util.retrieval_utils import (
    clean_document_content,
    is_academic_noise,
    model_filter_and_rank,
    remove_think_tags,
)

c = Cache("/tmp/antibody_gen/web_search/")


def _filter_chunks(chunks: list[Document]) -> list[Document]:
    """过滤文档"""
    return filter(lambda x: not is_academic_noise(x.page_content), chunks)


class RetrievedDocument(BaseModel):
    source: str = Field(
        description="source of the document, can be a url or a file path"
    )
    page_content: str = Field(description="content of the document")

    def __str__(self) -> str:
        return f"""
        <document>
            <source>{self.source}</source>
            <page_content>{self.page_content}</page_content>
        </document>
        """

    def __repr__(self) -> str:
        return self.__str__()


@tool(parse_docstring=True)
def retrieve_doc(
    query: List[str], config: RunnableConfig, k_per_query: int = 5
) -> list[RetrievedDocument]:
    """
    Retrieve related documents from the knowledge base. Source of the document is in "source" field of metadata

    Args:
        query: List of query strings. It is recommended that each string can be 16 to 128 tokens in lenth, so that the query can capture the user's intention.
        config: RunnableConfig.
        k_per_query: number of retrived documents for each query.

    Returns:
        Retrieved documents with source and page content, where source is the path or url of original paper. Source of the document is included in <source> tag
    """
    try:
        vector_store = get_vector_store("immune")
        retriever = QdrantParentDocumentRetriever(
            summarize_model=get_summarize_model(config),
            vector_store=vector_store,
            role="computational antibody design expert",
            retriever_kwargs={
                "search_type": "mmr",
                "search_kwargs": {"k": k_per_query},
            },
            chunk_filter=_filter_chunks,
        )
        all_docs = []
        seen_docs = set()

        # 顺序执行检索，避免向量数据库的并发冲突
        results = []
        for q in query:
            try:
                docs = retriever.invoke(
                    q
                )  # 修复：传入单个查询字符串 q，而不是整个列表 query

                # 有些 retriever 可能会带 <think/>
                for doc in docs:
                    doc.page_content = remove_think_tags(doc.page_content)
                results.append(docs)
                print(f"查询 '{q[:50]}...' 检索到 {len(docs)} 个文档")
            except Exception as e:
                print(f"Retriever 查询失败: {e}")
                results.append([])

        # 合并结果并去重
        total_retrieved = 0
        for docs in results:
            total_retrieved += len(docs)

            for doc in docs:
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
        return [
            RetrievedDocument(
                source=doc.metadata[KEY_SRC], page_content=doc.page_content
            )
            for doc in top_docs
        ]
    except Exception as e:
        print(f"检索查询失败: {e}")
        return []


@tool(parse_docstring=True)
def retrieve(query: List[str], config: RunnableConfig, k_per_query: int = 10) -> str:
    """
    Retrieve related documents from the knowledge base.

    Args:
        query: List of query strings.
        config: RunnableConfig.
        k_per_query: number of retrived documents for each query

    Returns:
        Retrieved documents with page_content and source. The source of a document can be a url or a file path. Source of the document is included in <source> tag
    """
    try:
        top_docs = retrieve_doc.invoke(
            {"query": query, "k_per_query": k_per_query}, config
        )
        # 构建上下文
        context = "\n\n".join([str(doc) for doc in top_docs])
        return context
    except Exception as e:
        print(f"检索查询失败: {e}")
        return ""


@c.memoize()
def search_one(q: str, k_per_query: int = 10) -> str:
    academic_search = TavilySearch(
        max_results=k_per_query,
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
    return results


@tool(parse_docstring=True)
def web_search_node(query: List[str], k_per_query: int = 10, k_total: int = 10) -> str:
    """
    Search for online resources related to the query.

    Args:
        query: List of query strings.
        k_per_query: number of retrived documents for each query.
        k_total: number of documents in final results.

    Returns:
        Search results.
    """
    print(f"在线搜索 query: {query}")
    # 设置TAVILY_API_KEY
    if not os.environ.get("TAVILY_API_KEY"):
        from config.api_keys import APIKeys
        os.environ["TAVILY_API_KEY"] = APIKeys.TAVILY_API_KEY

    try:
        all_results = []  # 存储所有结果
        seen_urls = set()  # 用于去重的URL集合

        for q in query:
            results = search_one(q, k_per_query)
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
        )[:k_total]

        # 提取content并连接
        content_list = [
            f"""
            <document>
            <source>{result.get("url", "")}</source>
            <content>{result.get("content", "")}</content>
            </document>
            """
            for result in final_results
            if result.get("content", "")
        ]
        combined_content = "\n\n".join(content_list)
        return combined_content
    except Exception as e:
        print(f"搜索查询失败: {e}")
        return ""
