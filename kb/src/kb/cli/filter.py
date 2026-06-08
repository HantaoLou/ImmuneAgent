import math
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
from diskcache import Cache
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from nltk.tokenize import word_tokenize

from kb.config import get_text_splitter

cache = Cache("tmp")


def compute_token_probs(chunks: List[Document]) -> dict[str, float]:
    """计算文档块中所有token的概率分布"""
    # 合并所有文档块的文本
    all_text = " ".join([chunk.page_content for chunk in chunks])

    # 对文本进行分词并转换为小写
    tokens = word_tokenize(all_text.lower())

    # 过滤掉非字母数字的token
    tokens = [token for token in tokens if token.isalnum()]

    # 计算每个token的频率
    token_freq = Counter(tokens)

    # 计算总token数
    total_tokens = len(tokens)

    # 计算每个token的概率
    token_probs = {token: count / total_tokens for token, count in token_freq.items()}

    return token_probs


def compute_token_tfidf(chunks: List[Document]) -> dict[str, float]:
    """计算文档块中token的信息熵"""
    # 合并所有文档块的文本
    all_text = " ".join([chunk.page_content for chunk in chunks])

    # 对文本进行分词
    tokens = word_tokenize(all_text.lower())

    # 计算每个token的频率
    token_freq = Counter(tokens)
    total_tokens = len(tokens)

    # 计算每个token的概率
    token_probs = {token: count / total_tokens for token, count in token_freq.items()}
    # 计算每个文档的token频率
    doc_token_freqs = []
    for chunk in chunks:
        # 对文本进行分词并移除特殊字符
        tokens = [
            token.lower()
            for token in word_tokenize(chunk.page_content)
            if token.isalnum()
        ]
        # 计算当前文档的token频率
        token_freq = Counter(tokens)
        doc_token_freqs.append(token_freq)

    # 计算所有不同的tokens
    all_tokens = set()
    for freq in doc_token_freqs:
        all_tokens.update(freq.keys())

    # 计算每个token的TF-IDF
    token_tfidf = {}
    num_docs = len(chunks)

    for token in all_tokens:
        # 计算文档频率(DF)
        doc_freq = sum(1 for freq in doc_token_freqs if token in freq)
        # 计算逆文档频率(IDF)
        idf = math.log(num_docs / (1 + doc_freq))

        # 计算所有文档的平均TF
        tf_sum = sum(
            freq[token] / sum(freq.values())
            for freq in doc_token_freqs
            if token in freq
        )
        avg_tf = tf_sum / num_docs

        # 计算最终的TF-IDF值
        token_tfidf[token] = avg_tf * idf
    return token_tfidf


def filter_by_entropy(
    chunks: List[Document], threshold: float
) -> List[tuple[Document, float]]:
    """根据信息熵过滤文档块"""
    # 计算token概率分布
    token_probs = compute_token_probs(chunks)

    filtered_chunks = []
    for chunk in chunks:
        # 对文档块进行分词
        tokens = [token.lower() for token in word_tokenize(chunk.page_content)]

        if not tokens:
            continue

        # 计算文档块的平均信息熵
        chunk_entropy = 0
        for token in tokens:
            if token in token_probs:
                p = token_probs[token]
                chunk_entropy -= p * math.log2(p)

        # 归一化信息熵
        chunk_entropy /= len(tokens)

        # 如果信息熵超过阈值，保留该文档块
        if chunk_entropy >= threshold:
            filtered_chunks.append((chunk, chunk_entropy))

    return filtered_chunks


def filter_by_tf_idf(
    chunks: List[Document], threshold: float
) -> List[tuple[Document, float]]:
    """过滤掉TF-IDF值低于阈值的文档块"""
    token_tfidf = compute_token_tfidf(chunks)
    filtered_chunks = []
    for chunk in chunks:
        tokens = [
            token.lower()
            for token in word_tokenize(chunk.page_content)
            if token.isalnum()
        ]
        # 计算文档块的总TF-IDF值并根据文档长度进行归一化
        chunk_tfidf = (
            sum(token_tfidf[token] for token in tokens if token in token_tfidf)
            / len(tokens)
            if tokens
            else 0
        )
        if chunk_tfidf >= threshold:
            filtered_chunks.append((chunk, chunk_tfidf))
    return filtered_chunks


@cache.memoize()
def split_document(pdf_path) -> List[Document]:
    # 加载PDF文件
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    print(f"切分PDF文件:{pdf_path}")
    # 对每页进行分块
    text_splitter = get_text_splitter()
    chunks: list[Document] = text_splitter.split_documents(pages)
    return chunks


def read_and_chunk_pdfs(pdf_dir) -> List[Document]:
    """读取PDF文件并分块"""
    from multiprocessing import Pool, Queue

    p = Pool(8)

    # 存储所有文档块
    all_chunks = p.starmap(
        split_document, [(pdf_path,) for pdf_path in Path(pdf_dir).glob("*.pdf")]
    )
    p.close()
    p.join()

    return [chunk for chunks in all_chunks for chunk in chunks]


def read_and_chunk_pdf_files(files) -> list[Document]:
    """读取PDF文件并分块"""
    from multiprocessing import Pool

    p = Pool(8)

    # 存储所有文档块
    all_chunks = p.map(split_document, files)
    p.close()
    p.join()

    return [chunk for chunks in all_chunks for chunk in chunks]


# test code
if __name__ == "__main__":
    chunks = read_and_chunk_pdf_files(
        list(Path("/data_new/lht/Immune_papers/").glob("*.pdf"))[:8]
    )
    filtered_chunks = filter_by_entropy(chunks, 0)
    sorted_chunks = sorted(filtered_chunks, key=lambda x: x[1], reverse=True)
    print("highest:\n\n\n")
    for chunk, entropy in sorted_chunks[:200]:
        print(f"Chunk entropy: {entropy}")
        print(chunk.page_content[:400])
        print("=" * 20)
    print("lowest:\n\n\n")
    for chunk, entropy in sorted_chunks[-500:]:
        print(f"Chunk entropy: {entropy}")
        print(chunk.page_content[:400])
        print("=" * 20)
