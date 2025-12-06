import re

from langchain_core.runnables.config import RunnableConfig

from common.factory import get_reasoning_model
from common.prompts import RetreiverPrompts


def remove_think_tags(text: str) -> str:
    """移除文本中的<think>标签及其内容"""
    if text is None:
        return ""

    # 移除完整的<think>标签及其内容
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 处理只有闭合标签的情况，移除闭合标签及其之前的内容
    cleaned_text = re.sub(r".*?</think>", "", cleaned_text, flags=re.DOTALL)
    # 移除多余的空行
    cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)
    return cleaned_text.strip()


def is_academic_noise(content: str) -> bool:
    """检测文本内容是否为学术噪音

    Args:
        content: 待检测的文本内容

    Returns:
        bool: True表示是噪音，False表示不是噪音
    """
    if not content or len(content) <= 80:
        return True

    content_lower = content.lower()

    # 引用格式噪音
    if re.search(r"^\[\d+\]", content):  # [6] 开头
        return True
    elif re.search(r"^\d+\.\s*[a-z]\.\s*[a-z]", content_lower):  # "32. h. cai" 格式
        return True
    elif re.search(r"et al\.|crossref|pubmed|doi:", content_lower):  # 学术引用标识
        return True

    # 期刊格式和模板噪音 - 匹配特定格式模式
    elif re.search(
        r"(springer nature \d{4}|latex template|copyright.*reserved)", content_lower
    ):
        return True
    elif re.search(
        r"(corresponding author:.*@|published online:|received:.*accepted:)",
        content_lower,
    ):
        return True

    # 图表目录噪音
    elif re.search(r"^(fig\.|figure|table|supplementary)\s*\d+", content_lower):
        return True

    # 页码和格式噪音
    elif re.search(r"^\d{4}\)\d{3}|volume\s*\d+|issue\s*\d+", content_lower):
        return True

    # 纯引用列表（括号数字密度过高）
    elif content.count("(") + content.count("[") > len(content) / 20:
        return True

    return False


def clean_document_content(content: str) -> str:
    """清理文档内容

    Args:
        content: 原始文档内容

    Returns:
        str: 清理后的内容
    """
    if not content:
        return ""

    # 移除行尾数字编号
    content = re.sub(r"\d{3,}$", "", content)
    # 移除多余空白
    content = re.sub(r"\s+", " ", content).strip()

    return content


def model_filter_and_rank(filtered_docs, queries, config: RunnableConfig):
    """精准筛选高质量上下文 - 使用结构化评分"""
    from concurrent.futures import ThreadPoolExecutor

    from langchain_core.prompts import ChatPromptTemplate

    from schema.common_schemas import DocumentEvaluation

    def score_document(doc_info):
        """评分单个文档"""
        doc_idx, content, questions = doc_info

        prompt = ChatPromptTemplate.from_template(RetreiverPrompts.PAPER_SCORING_PROMPT)

        try:
            reasoning_model = get_reasoning_model(config)
            evaluation_model = reasoning_model.with_structured_output(
                DocumentEvaluation
            )
            evaluation_chain = prompt | evaluation_model

            response = evaluation_chain.invoke({"query": questions, "content": content})

            print(
                f"文档{doc_idx + 1}: 相关性={response.relevance_score}, 质量={response.quality_score}, 噪声={response.noise_level}, 总分={response.final_score}"
            )
            return (doc_idx + 1, response)

        except Exception as e:
            print(f"文档{doc_idx + 1}评分失败: {e}")
            # 返回默认低分评估
            default_eval = DocumentEvaluation(
                relevance_score=0, quality_score=0, noise_level=3, final_score=0
            )
            return (doc_idx + 1, default_eval)

    # 构建查询字符串
    questions = "\n".join([f"问题{i + 1}: {query}" for i, query in enumerate(queries)])

    # 准备评分任务
    tasks = []
    for i, doc in enumerate(filtered_docs):
        content = doc.page_content.strip()
        if len(content) > 30:
            tasks.append((i, content, questions))

    print(f"开始评分 {len(tasks)} 个文档...")

    try:
        # 并发评分
        with ThreadPoolExecutor(max_workers=4) as executor:
            doc_evaluations = list(executor.map(score_document, tasks))

        # 按final_score排序
        doc_evaluations.sort(key=lambda x: x[1].final_score, reverse=True)

        # 安全筛选策略
        result = safe_document_filter(filtered_docs, doc_evaluations)

        print(f"筛选结果: {len(result)}个文档")
        return result
    except Exception as e:
        print(f"评分失败，使用原始排序: {e}")
        return filtered_docs[:10]


def safe_document_filter(documents, doc_evaluations, target_count=15):
    """简化的文档筛选：按总分和噪声排序，返回前N个文档"""

    # 提取文档和评估结果
    doc_eval_pairs = []
    for doc_idx, evaluation in doc_evaluations:
        if doc_idx <= len(documents):
            doc_eval_pairs.append((documents[doc_idx - 1], evaluation))

    if not doc_eval_pairs:
        return []

    # 按总分排序，噪声等级作为次要排序条件（噪声越低越好）
    sorted_docs = sorted(
        doc_eval_pairs,
        key=lambda x: (x[1].final_score, -x[1].noise_level),  # 总分降序，噪声升序
        reverse=True,
    )

    # 返回前N个文档
    return [doc for doc, _ in sorted_docs[:target_count]]
