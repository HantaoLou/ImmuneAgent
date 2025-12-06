"""
Repository Manager for AIRR Data Commons

Manages connections to multiple AIRR repositories with failover support.
"""

import requests
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class RepositoryManager:
    """Manages connections to multiple AIRR Data Commons repositories"""

    REPOSITORIES = {
        # VDJServer V1 已停用，改为使用 iReceptor 作为主要仓库
        "ireceptor": {
            "base_url": "https://ipa1.ireceptor.org/airr/v1",
            "name": "iReceptor Public Archive",
            "timeout": 30,
            "description": "General immunology studies"
        },
        "covid19": {
            "base_url": "https://covid19-1.ireceptor.org/airr/v1",
            "name": "iReceptor COVID-19 Archive",
            "timeout": 30,
            "description": "COVID-19 specific studies"
        },
        # 保留 vdjserver 键但将其放在最后，以便在 query_with_failover 中最后尝试
        "vdjserver": {
            "base_url": "https://ipa1.ireceptor.org/airr/v1",  # 临时使用 iReceptor 的 API 地址
            "name": "VDJServer Community Data Portal (Redirected to iReceptor)",
            "timeout": 30,
            "description": "VDJServer V1 已停用，此处重定向到 iReceptor"
        }
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'AIRR-MCP-Server/1.0'
        })

    def get_repository_url(self, repo_id: str, endpoint: str) -> str:
        """
        Get full URL for repository endpoint

        Args:
            repo_id: Repository identifier (vdjserver, ireceptor, covid19)
            endpoint: API endpoint (repertoire, rearrangement, etc.)

        Returns:
            Full URL
        """
        if repo_id not in self.REPOSITORIES:
            raise ValueError(f"Unknown repository: {repo_id}")

        base_url = self.REPOSITORIES[repo_id]["base_url"]
        return f"{base_url}/{endpoint}"

    def query_single(
        self,
        repo_id: str,
        endpoint: str,
        filters: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query a single repository

        Args:
            repo_id: Repository identifier
            endpoint: API endpoint
            filters: Query filters
            timeout: Optional timeout override

        Returns:
            API response data
        """
        url = self.get_repository_url(repo_id, endpoint)
        repo_timeout = timeout or self.REPOSITORIES[repo_id]["timeout"]

        # 打印请求信息以进行调试
        logger.info(f"Querying {repo_id} at {url}")
        logger.info(f"Request filters: {filters}")
        
        # 确保filters是有效的JSON
        if "filters" in filters and not filters["filters"]:
            # 如果filters为空，删除它以避免API错误
            logger.info(f"Removing empty filters from request")
            del filters["filters"]
        
        try:
            # 添加适当的请求头
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            response = self.session.post(
                url,
                json=filters,
                headers=headers,
                timeout=repo_timeout
            )
            
            # 打印响应状态码
            logger.info(f"Response status code: {response.status_code}")
            
            # 尝试获取响应内容
            try:
                response_text = response.text[:500]  # 限制长度
                logger.info(f"Response text: {response_text}...")
            except:
                logger.warning("Could not get response text")
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析JSON响应
            try:
                json_response = response.json()
                return json_response
            except ValueError as e:
                logger.error(f"Invalid JSON response from {repo_id}: {e}")
                return {
                    "error": "invalid_json",
                    "message": f"Invalid JSON response: {str(e)}",
                    "response_text": response.text[:1000]  # 包含部分响应文本以便调试
                }

        except requests.exceptions.Timeout:
            logger.error(f"Timeout querying {repo_id}")
            return {
                "error": "timeout",
                "message": f"Request to {repo_id} timed out"
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying {repo_id}: {e}")
            return {
                "error": "request_failed",
                "message": str(e)
            }

    def query_all(
        self,
        endpoint: str,
        filters: Dict[str, Any],
        repositories: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Query all repositories in parallel

        Args:
            endpoint: API endpoint
            filters: Query filters
            repositories: Optional list of specific repositories to query

        Returns:
            Dict mapping repository ID to results
        """
        repos_to_query = repositories if repositories else list(self.REPOSITORIES.keys())
        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.query_single, repo, endpoint, filters): repo
                for repo in repos_to_query
            }

            for future in as_completed(futures):
                repo = futures[future]
                try:
                    results[repo] = future.result()
                except Exception as e:
                    logger.error(f"Exception querying {repo}: {e}")
                    results[repo] = {
                        "error": "exception",
                        "message": str(e)
                    }

        return results

    def query_with_failover(
        self,
        endpoint: str,
        filters: Dict[str, Any],
        repositories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query repositories with failover - return first successful result

        Args:
            endpoint: API endpoint
            filters: Query filters
            repositories: Optional list of repositories (in priority order)

        Returns:
            First successful result or error if all fail
        """
        repos_to_try = repositories if repositories else ["vdjserver", "ireceptor", "covid19"]

        for repo in repos_to_try:
            result = self.query_single(repo, endpoint, filters)

            if "error" not in result:
                logger.info(f"Successfully retrieved data from {repo}")
                result["_repository"] = repo
                return result

            logger.warning(f"Failed to get data from {repo}: {result.get('message')}")

        return {
            "status": "error",
            "error": "all_repositories_failed",
            "message": "All repositories failed to respond",
            "repositories_tried": repos_to_try
        }

    def get_repository_info(self, repo_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about available repositories

        Args:
            repo_id: Optional specific repository ID

        Returns:
            Repository information
        """
        if repo_id:
            if repo_id not in self.REPOSITORIES:
                return {"error": f"Unknown repository: {repo_id}"}
            return self.REPOSITORIES[repo_id]

        return {
            "repositories": self.REPOSITORIES,
            "total_count": len(self.REPOSITORIES)
        }

    def test_connection(self, repo_id: str) -> Dict[str, Any]:
        """
        Test connection to a repository

        Args:
            repo_id: Repository identifier

        Returns:
            Connection test results
        """
        try:
            url = self.get_repository_url(repo_id, "repertoire")

            # Simple query to test connection
            test_query = {
                "filters": {},
                "from": 0,
                "size": 1
            }

            response = self.session.post(url, json=test_query, timeout=10)

            return {
                "status": "success",
                "repository": repo_id,
                "response_code": response.status_code,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "available": response.status_code == 200
            }

        except Exception as e:
            return {
                "status": "error",
                "repository": repo_id,
                "available": False,
                "error": str(e)
            }

    def test_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """
        Test connections to all repositories

        Returns:
            Test results for all repositories
        """
        results = {}

        for repo_id in self.REPOSITORIES:
            results[repo_id] = self.test_connection(repo_id)

        return {
            "status": "complete",
            "results": results,
            "available_count": sum(1 for r in results.values() if r.get("available", False))
        }
