#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOI元数据获取工具模块

提供通过DOI获取论文元数据（标题、作者、期刊等信息）的功能
支持CrossRef和OpenAlex两个主要的学术数据API
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from usecases.immunity.schema.common_schemas import Citation


class DOIMetadataRetriever:
    """DOI元数据检索器

    通过CrossRef和OpenAlex API获取DOI对应的论文元数据
    """

    def __init__(self, user_agent: str = "DOI-Retriever/1.0"):
        """初始化检索器

        Args:
            user_agent: HTTP请求的用户代理字符串，建议包含联系信息
        """
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": self.user_agent, "Accept": "application/json"}
        )

    def get_crossref_metadata(self, doi: str) -> Optional[Dict[str, Any]]:
        """通过CrossRef API获取DOI元数据

        Args:
            doi: DOI标识符

        Returns:
            元数据字典，失败时返回None
        """
        try:
            # 清理DOI格式，移除URL前缀
            clean_doi = (
                doi.strip()
                .replace("https://doi.org/", "")
                .replace("http://dx.doi.org/", "")
            )

            # 构建CrossRef API请求URL
            url = f"https://api.crossref.org/works/{clean_doi}"

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            # 检查API响应状态
            if data.get("status") == "ok" and "message" in data:
                return data["message"]
            else:
                return None

        except (requests.exceptions.RequestException, json.JSONDecodeError, Exception):
            return None

    def get_openalex_metadata(self, doi: str) -> Optional[Dict[str, Any]]:
        """通过OpenAlex API获取DOI元数据

        Args:
            doi: DOI标识符

        Returns:
            元数据字典，失败时返回None
        """
        try:
            # 清理DOI格式
            clean_doi = (
                doi.strip()
                .replace("https://doi.org/", "")
                .replace("http://dx.doi.org/", "")
            )

            # 构建OpenAlex API请求URL
            url = f"https://api.openalex.org/works/doi:{clean_doi}"

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data

        except (requests.exceptions.RequestException, json.JSONDecodeError, Exception):
            return None

    def parse_crossref_metadata(self, metadata: Dict[str, Any]) -> Citation:
        """解析CrossRef元数据为Citation对象

        Args:
            metadata: CrossRef API返回的元数据字典

        Returns:
            Citation对象
        """
        citation = Citation()

        try:
            # 解析标题
            if "title" in metadata and metadata["title"]:
                citation.title = metadata["title"][0]

            # 解析作者信息
            if "author" in metadata:
                authors = []
                for author in metadata["author"]:
                    given = author.get("given", "")
                    family = author.get("family", "")
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
                citation.authors = authors

            # 解析期刊名称
            if "container-title" in metadata and metadata["container-title"]:
                citation.journal = metadata["container-title"][0]

            # 解析发表年份
            if "published-print" in metadata:
                date_parts = metadata["published-print"].get("date-parts", [[]])
                if date_parts and date_parts[0]:
                    citation.year = str(date_parts[0][0])
            elif "published-online" in metadata:
                date_parts = metadata["published-online"].get("date-parts", [[]])
                if date_parts and date_parts[0]:
                    citation.year = str(date_parts[0][0])

            # 解析卷号、期号、页码
            citation.volume = metadata.get("volume", "")
            citation.issue = metadata.get("issue", "")
            citation.pages = metadata.get("page", "")

            # 设置DOI和URL
            citation.doi = metadata.get("DOI", "")
            if citation.doi:
                citation.url = f"https://doi.org/{citation.doi}"

            # 解析PMID - CrossRef中可能在link字段中包含PubMed链接
            if "link" in metadata:
                for link in metadata["link"]:
                    if (
                        link.get("intended-application") == "text-mining"
                        and "URL" in link
                    ):
                        url = link["URL"]
                        # 检查是否为PubMed链接
                        if "pubmed" in url.lower() or "ncbi.nlm.nih.gov/pubmed" in url:
                            # 从URL中提取PMID
                            import re

                            pmid_match = re.search(r"pubmed/(\d+)", url)
                            if pmid_match:
                                citation.pmid = pmid_match.group(1)
                                break

            # 解析摘要 - CrossRef中摘要通常在abstract字段中
            if "abstract" in metadata:
                # CrossRef的摘要可能包含HTML标签，需要清理
                abstract_text = metadata["abstract"]
                if abstract_text:
                    # 简单的HTML标签清理
                    import re

                    clean_abstract = re.sub(r"<[^>]+>", "", abstract_text)
                    citation.abstract = clean_abstract.strip()

        except Exception:
            pass  # 忽略解析错误，返回部分解析的结果

        return citation

    def parse_openalex_metadata(self, metadata: Dict[str, Any]) -> Citation:
        """解析OpenAlex元数据为Citation对象

        Args:
            metadata: OpenAlex API返回的元数据字典

        Returns:
            Citation对象
        """
        citation = Citation()

        try:
            # 解析标题
            citation.title = metadata.get("title", "")

            # 解析作者信息
            if "authorships" in metadata:
                authors = []
                for authorship in metadata["authorships"]:
                    author = authorship.get("author", {})
                    display_name = author.get("display_name", "")
                    if display_name:
                        authors.append(display_name)
                citation.authors = authors

            # 解析期刊信息
            if "primary_location" in metadata and metadata["primary_location"]:
                source = metadata["primary_location"].get("source", {})
                if source:
                    citation.journal = source.get("display_name", "")

            # 解析发表年份
            citation.year = str(metadata.get("publication_year", ""))

            # 解析DOI和URL
            citation.doi = metadata.get("doi", "").replace("https://doi.org/", "")
            if citation.doi:
                citation.url = f"https://doi.org/{citation.doi}"

            # 解析PMID - OpenAlex中PMID通常在ids字段中
            if "ids" in metadata:
                ids = metadata["ids"]
                # 检查pmid字段
                if "pmid" in ids and ids["pmid"]:
                    pmid_url = ids["pmid"]
                    # PMID URL格式通常为 https://pubmed.ncbi.nlm.nih.gov/XXXXXXX
                    if pmid_url:
                        import re

                        pmid_match = re.search(
                            r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", pmid_url
                        )
                        if pmid_match:
                            citation.pmid = pmid_match.group(1)
                        else:
                            # 如果直接是数字格式
                            pmid_match = re.search(r"(\d+)", pmid_url)
                            if pmid_match:
                                citation.pmid = pmid_match.group(1)

            # 解析摘要 - OpenAlex中摘要在abstract_inverted_index字段中
            if (
                "abstract_inverted_index" in metadata
                and metadata["abstract_inverted_index"]
            ):
                # OpenAlex使用倒排索引存储摘要，需要重构
                inverted_index = metadata["abstract_inverted_index"]
                if inverted_index:
                    # 重构摘要文本
                    word_positions = []
                    for word, positions in inverted_index.items():
                        for pos in positions:
                            word_positions.append((pos, word))

                    # 按位置排序并组合成文本
                    word_positions.sort(key=lambda x: x[0])
                    abstract_words = [word for _, word in word_positions]
                    citation.abstract = " ".join(abstract_words)

        except Exception:
            pass  # 忽略解析错误，返回部分解析的结果

        return citation

    def get_metadata_by_doi(
        self, doi: str, prefer_crossref: bool = True
    ) -> Optional[Citation]:
        """通过DOI获取论文元数据

        Args:
            doi: DOI标识符
            prefer_crossref: 是否优先使用CrossRef API，默认为True

        Returns:
            Citation对象，失败时返回None
        """
        if prefer_crossref:
            # 优先尝试CrossRef API
            crossref_data = self.get_crossref_metadata(doi)
            if crossref_data:
                return self.parse_crossref_metadata(crossref_data)

            # CrossRef失败时尝试OpenAlex
            openalex_data = self.get_openalex_metadata(doi)
            if openalex_data:
                return self.parse_openalex_metadata(openalex_data)
        else:
            # 优先尝试OpenAlex API
            openalex_data = self.get_openalex_metadata(doi)
            if openalex_data:
                return self.parse_openalex_metadata(openalex_data)

            # OpenAlex失败时尝试CrossRef
            crossref_data = self.get_crossref_metadata(doi)
            if crossref_data:
                return self.parse_crossref_metadata(crossref_data)

        return None


def get_citation_by_doi(
    doi: str, user_agent: str = "DOI-Retriever/1.0"
) -> Optional[Citation]:
    """便捷函数：通过DOI获取论文引用信息

    Args:
        doi: DOI标识符
        user_agent: HTTP请求的用户代理字符串

    Returns:
        Citation对象，失败时返回None

    Example:
        >>> citation = get_citation_by_doi("10.1038/nature12373")
        >>> if citation:
        ...     print(f"标题: {citation.title}")
        ...     print(f"作者: {', '.join(citation.authors)}")
    """
    retriever = DOIMetadataRetriever(user_agent=user_agent)
    return retriever.get_metadata_by_doi(doi)


def enhance_citation_with_doi(citation: Citation) -> Citation:
    """使用DOI增强现有的Citation对象

    Args:
        citation: 需要增强的Citation对象

    Returns:
        增强后的Citation对象
    """
    if not citation.doi:
        return citation

    # 获取完整的元数据
    enhanced = get_citation_by_doi(citation.doi)
    if not enhanced:
        return citation

    # 用获取到的数据填充空字段
    if not citation.title and enhanced.title:
        citation.title = enhanced.title
    if not citation.authors and enhanced.authors:
        citation.authors = enhanced.authors
    if not citation.journal and enhanced.journal:
        citation.journal = enhanced.journal
    if not citation.year and enhanced.year:
        citation.year = enhanced.year
    if not citation.volume and enhanced.volume:
        citation.volume = enhanced.volume
    if not citation.issue and enhanced.issue:
        citation.issue = enhanced.issue
    if not citation.pages and enhanced.pages:
        citation.pages = enhanced.pages
    if not citation.url and enhanced.url:
        citation.url = enhanced.url

    return citation


if __name__ == "__main__":
    """简单测试指定的三个DOI"""
    test_dois = [
        "10.1101/cshperspect.a028795",
        "10.1084/jem.194.3.375",
        "10.1073/pnas.1301810110",
        "10.1126/science.adr6896",
    ]

    print("DOI元数据获取测试（包含PMID和摘要）")
    print("=" * 60)

    for i, doi in enumerate(test_dois, 1):
        print(f"\n[{i}/3] 测试DOI: {doi}")
        try:
            citation = get_citation_by_doi(doi)

            if citation and citation.title:
                print(f"标题: {citation.title}")
                print(
                    f"作者: {', '.join(citation.authors) if citation.authors else '未知'}"
                )
                print(f"期刊: {citation.journal}")
                print(f"年份: {citation.year}")
                print(f"卷号: {citation.volume if citation.volume else '未知'}")
                print(f"期号: {citation.issue if citation.issue else '未知'}")
                print(f"页码: {citation.pages if citation.pages else '未知'}")
                print(f"DOI: {citation.doi}")
                print(f"URL: {citation.url}")
                print(f"PMID: {citation.pmid if citation.pmid else '未知'}")

                # 显示摘要（如果存在且不为空）
                if citation.abstract:
                    # 限制摘要显示长度，避免输出过长
                    abstract_preview = (
                        citation.abstract[:200] + "..."
                        if len(citation.abstract) > 200
                        else citation.abstract
                    )
                    print(f"摘要: {abstract_preview}")
                else:
                    print("摘要: 未知")
            else:
                print("获取失败 - 未找到有效的元数据")
        except Exception as e:
            print(f"获取失败 - 发生异常: {str(e)}")

    print("\n测试完成")
