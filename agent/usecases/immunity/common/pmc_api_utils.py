"""
PMC API 工具类

提供从PMC URL中提取PMCID并通过PMC ID Converter API获取DOI等元数据的功能。
支持批量处理和单个URL处理。

作者: AI Assistant
创建时间: 2024
"""

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests


class PMCAPIUtils:
    """PMC API工具类，用于处理PMC URL和获取DOI信息"""

    # PMC ID Converter API基础URL
    PMC_API_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

    # PMC URL正则表达式模式
    PMC_URL_PATTERNS = [
        r"pmc\.ncbi\.nlm\.nih\.gov/articles/PMC(\d+)",  # 标准PMC URL
        r"ncbi\.nlm\.nih\.gov/pmc/articles/PMC(\d+)",  # 简化PMC URL
        r"PMC(\d+)",  # 直接PMCID
    ]

    def __init__(self, timeout: int = 10, max_retries: int = 3):
        """
        初始化PMC API工具类

        Args:
            timeout: API请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    def extract_pmcid_from_url(self, url: str) -> Optional[str]:
        """
        从URL中提取PMCID

        Args:
            url: 包含PMC信息的URL

        Returns:
            提取到的PMCID（不包含PMC前缀），如果未找到则返回None
        """
        if not url:
            return None

        # 尝试所有的正则表达式模式
        for pattern in self.PMC_URL_PATTERNS:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                pmcid = match.group(1)
                self.logger.debug(f"从URL '{url}' 中提取到PMCID: {pmcid}")
                return pmcid

        self.logger.warning(f"无法从URL '{url}' 中提取PMCID")
        return None

    def get_doi_from_pmcid(self, pmcid: str) -> Optional[Dict[str, Any]]:
        """
        通过PMCID获取DOI和其他元数据

        Args:
            pmcid: PMC ID（不包含PMC前缀）

        Returns:
            包含DOI和其他元数据的字典，如果获取失败则返回None
        """
        if not pmcid:
            return None

        # 构建API请求URL
        api_url = f"{self.PMC_API_BASE_URL}?ids=PMC{pmcid}&format=json"

        for attempt in range(self.max_retries):
            try:
                self.logger.debug(
                    f"正在请求PMC API: {api_url} (尝试 {attempt + 1}/{self.max_retries})"
                )

                response = requests.get(api_url, timeout=self.timeout)
                response.raise_for_status()

                data = response.json()

                # 检查API响应格式
                if "records" not in data or not data["records"]:
                    self.logger.warning(f"PMC API返回空记录，PMCID: {pmcid}")
                    return None

                record = data["records"][0]

                # 提取元数据
                metadata = {
                    "pmcid": record.get("pmcid", f"PMC{pmcid}"),
                    "pmid": record.get("pmid"),
                    "doi": record.get("doi"),
                    "status": record.get("status"),
                    "errmsg": record.get("errmsg"),
                }

                # 检查是否有错误信息
                if metadata.get("errmsg"):
                    self.logger.error(f"PMC API返回错误: {metadata['errmsg']}")
                    return None

                self.logger.info(
                    f"成功获取PMCID {pmcid} 的元数据，DOI: {metadata.get('doi', 'N/A')}"
                )
                return metadata

            except requests.exceptions.RequestException as e:
                self.logger.warning(
                    f"PMC API请求失败 (尝试 {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    self.logger.error(f"PMC API请求最终失败，PMCID: {pmcid}")
                    return None
            except Exception as e:
                self.logger.error(f"处理PMC API响应时发生错误: {e}")
                return None

        return None

    def get_doi_from_url(self, url: str) -> Optional[str]:
        """
        从PMC URL中提取DOI

        Args:
            url: PMC URL

        Returns:
            DOI字符串，如果获取失败则返回None
        """
        pmcid = self.extract_pmcid_from_url(url)
        if not pmcid:
            return None

        metadata = self.get_doi_from_pmcid(pmcid)
        if metadata and metadata.get("doi"):
            return metadata["doi"]

        return None

    def get_metadata_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        从PMC URL中获取完整元数据

        Args:
            url: PMC URL

        Returns:
            包含完整元数据的字典，如果获取失败则返回None
        """
        pmcid = self.extract_pmcid_from_url(url)
        if not pmcid:
            return None

        return self.get_doi_from_pmcid(pmcid)

    def batch_get_dois_from_urls(self, urls: List[str]) -> Dict[str, Optional[str]]:
        """
        批量从PMC URL列表中获取DOI

        Args:
            urls: PMC URL列表

        Returns:
            URL到DOI的映射字典
        """
        results = {}

        for url in urls:
            try:
                doi = self.get_doi_from_url(url)
                results[url] = doi
                self.logger.debug(f"URL: {url} -> DOI: {doi}")
            except Exception as e:
                self.logger.error(f"处理URL '{url}' 时发生错误: {e}")
                results[url] = None

        return results

    def is_pmc_url(self, url: str) -> bool:
        """
        检查URL是否为PMC URL

        Args:
            url: 要检查的URL

        Returns:
            如果是PMC URL则返回True，否则返回False
        """
        if not url:
            return False

        # 检查是否包含PMC相关的域名或模式
        for pattern in self.PMC_URL_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return True

        return False


# 便捷函数，提供简单的接口
def extract_doi_from_pmc_url(url: str, timeout: int = 10) -> Optional[str]:
    """
    便捷函数：从PMC URL中提取DOI

    Args:
        url: PMC URL
        timeout: 请求超时时间

    Returns:
        DOI字符串，如果获取失败则返回None
    """
    utils = PMCAPIUtils(timeout=timeout)
    return utils.get_doi_from_url(url)


def batch_extract_dois_from_pmc_urls(
    urls: List[str], timeout: int = 10
) -> Dict[str, Optional[str]]:
    """
    便捷函数：批量从PMC URL列表中提取DOI

    Args:
        urls: PMC URL列表
        timeout: 请求超时时间

    Returns:
        URL到DOI的映射字典
    """
    utils = PMCAPIUtils(timeout=timeout)
    return utils.batch_get_dois_from_urls(urls)


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 创建工具实例
    pmc_utils = PMCAPIUtils()

    # 测试URL列表
    test_urls = [
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11938350/",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7840891/",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC1193645/",
    ]

    print("=== PMC API工具类测试 ===")

    # 测试单个URL
    test_url = test_urls[0]
    print(f"\n测试URL: {test_url}")

    # 提取PMCID
    pmcid = pmc_utils.extract_pmcid_from_url(test_url)
    print(f"提取的PMCID: {pmcid}")

    # 获取DOI
    doi = pmc_utils.get_doi_from_url(test_url)
    print(f"获取的DOI: {doi}")

    # 获取完整元数据
    metadata = pmc_utils.get_metadata_from_url(test_url)
    print(f"完整元数据: {metadata}")

    # 批量测试
    print(f"\n批量测试 {len(test_urls)} 个URL:")
    batch_results = pmc_utils.batch_get_dois_from_urls(test_urls)
    for url, doi in batch_results.items():
        print(f"  {url} -> {doi}")
