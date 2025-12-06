import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union
from urllib.parse import urlparse
from ftplib import FTP, error_perm
import posixpath
from contextlib import contextmanager

import requests


def save_planning_report(planning_result: str, filepath: str, filename: str) -> str:
    """
    保存规划结果为MD文件 / Save planning result as MD file

    Args:
        planning_result: 规划结果内容 / Planning result content

    Returns:
        str: MD文件路径 / MD file path
    """
    try:
        # 创建输出目录 / Create output directory
        output_dir = Path(
            "/data/szh/antibody_gen/agent/analysis_results/" + filepath + "/"
        )
        output_dir.mkdir(exist_ok=True)

        # 生成文件名 / Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}.md"
        filepath = output_dir / filename

        # 直接写入planning结果 / Write planning result directly
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(planning_result)

        print(f"📄 Planning saved to: {filepath}")
        return str(filepath)

    except Exception as e:
        print(f"⚠️ Failed to save planning: {str(e)}")
        return ""


def clean_json_response(response_text) -> dict:
    """
    Extract and clean JSON from LLM response that may contain markdown or extra text.
    Enhanced version with better error recovery.

    Args:
        response_text: Raw response from LLM that should contain JSON (str or response object)

    Returns:
        Parsed JSON dictionary or empty dict if extraction fails
    """
    try:
        # If it's already a dict, return it
        if isinstance(response_text, dict):
            return response_text

        # Extract text from response object if needed
        if hasattr(response_text, "content"):
            text = response_text.content
        else:
            text = str(response_text)

        # Check for empty or very short responses
        if not text or len(text.strip()) < 2:
            return {}

        # First, try to parse as-is
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Remove markdown code blocks
        if "```json" in text.lower():
            # Extract content between ```json and ```
            match = re.search(
                r"```(?:json|JSON)\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL
            )
            if match:
                text = match.group(1)
        elif "```" in text:
            # Generic code block
            parts = text.split("```")
            if len(parts) >= 3:  # Has opening and closing ```
                text = parts[1]
            elif len(parts) == 2:  # Only opening ```
                text = parts[1]

        # Try to parse after markdown removal
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Extract JSON object using balanced braces
        json_start = -1
        json_end = -1
        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == "\\" and not escape_next:
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    if json_start == -1:
                        json_start = i
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0 and json_start != -1:
                        json_end = i + 1
                        break

        if json_start != -1 and json_end != -1:
            json_str = text[json_start:json_end]

            # Clean common JSON issues
            # Remove trailing commas
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

            # Try to parse
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # More aggressive fixes
                # Fix unquoted keys
                json_str = re.sub(r"(\w+):", r'"\1":', json_str)
                # Fix single quotes
                json_str = json_str.replace("'", '"')
                # Fix Python booleans
                json_str = (
                    json_str.replace("True", "true")
                    .replace("False", "false")
                    .replace("None", "null")
                )

                try:
                    return json.loads(json_str)
                except:
                    pass

        # Last resort: try to find any JSON-like structure
        json_patterns = [
            r"\{[^{}]*\}",  # Simple object
            r"\[[^\[\]]*\]",  # Simple array
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    result = json.loads(match)
                    if isinstance(result, (dict, list)):
                        return result if isinstance(result, dict) else {"data": result}
                except:
                    continue

        # If no valid JSON found, return empty dict (safer than None)
        return {}

    except Exception as e:
        print(f"  ⚠️ JSON extraction error: {e}")
        return {}


def smart_truncate_abstract(abstract: str, max_length: int = 2500) -> str:
    """
    智能截取摘要内容，当长度超过限制时向前查找最后一个断句符号进行截取
    Smart truncate abstract content by finding the last punctuation mark when exceeding length limit

    Args:
        abstract: 原始摘要内容 / Original abstract content
        max_length: 最大长度限制 / Maximum length limit (default: 2500)

    Returns:
        str: 截取后的摘要内容 / Truncated abstract content
    """
    if not abstract or len(abstract) <= max_length:
        return abstract

    # 截取到最大长度 / Truncate to max length
    truncated = abstract[:max_length]

    # 定义断句符号（包括句号、感叹号、问号、逗号等） / Define punctuation marks for sentence breaking
    punctuation_marks = [".", "!", "?", ",", ";", ":"]

    # 从后往前查找最后一个断句符号 / Find the last punctuation mark from the end
    for i in range(len(truncated) - 1, -1, -1):
        if truncated[i] in punctuation_marks:
            # 在断句符号后截取，保持完整语句 / Truncate after the punctuation mark to keep complete sentences
            return truncated[: i + 1].strip()

    # 如果没有找到断句符号，查找最后一个空格避免截断单词 / If no punctuation found, find last space to avoid cutting words
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space].strip()

    # 最后方案：直接截取 / Last resort: direct truncation
    return truncated.strip()


def _ensure_dict_from_payload(payload: Union[str, dict]) -> Optional[dict]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    return None


DEFAULT_GEO_FILE_EXTENSIONS: tuple[str, ...] = (
    ".fastq",
    ".fastq.gz",
    ".fq",
    ".fq.gz",
    ".txt",
    ".txt.gz",
    ".tsv",
    ".tsv.gz",
    ".csv",
    ".csv.gz",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
    ".rds",
)


@contextmanager
def _ftp_cwd(ftp: FTP, path: str):
    """Context manager to temporarily change FTP working directory."""
    original = ftp.pwd()
    try:
        if path and path != ".":
            ftp.cwd(path)
        yield
    finally:
        try:
            ftp.cwd(original)
        except Exception:
            pass


def _ftp_is_directory(ftp: FTP, entry: str) -> bool:
    current = ftp.pwd()
    try:
        ftp.cwd(entry)
        ftp.cwd(current)
        return True
    except error_perm:
        try:
            ftp.cwd(current)
        except Exception:
            pass
        return False


def _download_ftp_tree(
    ftp_url: str,
    destination: Path,
    include_extensions: Optional[Iterable[str]],
    max_files: Optional[int],
    skip_existing: bool,
) -> list[str]:
    parsed = urlparse(ftp_url)
    if parsed.scheme.lower() not in {"ftp", "ftps"}:
        raise ValueError(f"Unsupported scheme for FTP download: {parsed.scheme}")

    host = parsed.hostname
    if not host:
        raise ValueError("FTP URL is missing hostname")

    port = parsed.port or 21
    ftp = FTP()
    ftp.connect(host, port, timeout=60)
    ftp.login(parsed.username or "anonymous", parsed.password or "anonymous@")

    downloaded_files: list[str] = []
    target_path = parsed.path or "/"

    def should_include(filename: str) -> bool:
        if not include_extensions:
            return True
        lower_name = filename.lower()
        return any(lower_name.endswith(ext.lower()) for ext in include_extensions)

    def download_file(remote_name: str, local_path: Path):
        if skip_existing and local_path.exists():
            downloaded_files.append(str(local_path))
            return
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as fh:
            ftp.retrbinary(f"RETR {remote_name}", fh.write)
        downloaded_files.append(str(local_path))

    def walk_directory(local_dir: Path):
        nonlocal downloaded_files
        if max_files is not None and len(downloaded_files) >= max_files:
            return
        try:
            entries = list(ftp.mlsd())
        except (error_perm, AttributeError):
            entries = [(name, {}) for name in ftp.nlst()]
        for name, facts in entries:
            if name in {".", ".."}:
                continue
            if max_files is not None and len(downloaded_files) >= max_files:
                break
            entry_type = facts.get("type") if isinstance(facts, dict) else None
            is_dir = entry_type == "dir"
            if entry_type is None:
                is_dir = _ftp_is_directory(ftp, name)
            if is_dir:
                with _ftp_cwd(ftp, name):
                    walk_directory(local_dir / name)
            else:
                if not should_include(name):
                    continue
                download_file(name, local_dir / name)

    # Determine whether we target a directory or a single file
    if target_path.endswith("/") or target_path == "/":
        with _ftp_cwd(ftp, target_path):
            walk_directory(destination)
    else:
        directory, filename = posixpath.split(target_path)
        if directory:
            with _ftp_cwd(ftp, directory):
                if should_include(filename):
                    download_file(filename, destination / filename)
        else:
            if should_include(filename):
                download_file(filename, destination / filename)

    ftp.quit()
    return downloaded_files


def _download_http_resource(url: str, destination: Path, skip_existing: bool) -> list[str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and destination.exists():
        return [str(destination)]
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(destination, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)
    return [str(destination)]


def download_geo_dataset(
    payload: Union[str, dict],
    destination_root: Union[str, Path] = "/data_new/workspace/geo",
    include_extensions: Optional[Iterable[str]] = DEFAULT_GEO_FILE_EXTENSIONS,
    max_files: Optional[int] = None,
    skip_existing: bool = True,
    cache: Optional[set] = None,
) -> Optional[dict]:
    """
    根据download_geo_sequences工具的返回结果，自动下载GEO数据集文件。

    Args:
        payload: 工具返回的结果（dict或JSON字符串）
        destination_root: 下载目标根目录
        include_extensions: 允许下载的文件后缀列表，None表示不限制
        max_files: 限制下载的最大文件数，None表示不限制
        skip_existing: 已存在文件是否跳过下载
        cache: 可选的缓存集合，用于去重（例如基于ftp_url）

    Returns:
        dict: 包含下载信息（ftp_url、destination、files等），若无法处理返回None
    """
    data = _ensure_dict_from_payload(payload)
    if not data:
        return None

    ftp_url = (
        data.get("ftp_url")
        or data.get("download_url")
        or data.get("ftp")
        or data.get("url")
    )
    if not ftp_url:
        return None

    geo_id = data.get("geo_id") or data.get("accession") or data.get("id")
    dest_root = Path(destination_root)
    target_dir = dest_root / (geo_id or "geo_dataset")
    target_dir.mkdir(parents=True, exist_ok=True)

    cache_key = ftp_url
    if cache is not None and cache_key in cache:
        existing_files = [str(p) for p in target_dir.iterdir() if p.is_file()]
        return {
            "ftp_url": ftp_url,
            "destination": str(target_dir),
            "files": existing_files,
            "cached": True,
        }

    parsed = urlparse(ftp_url)
    scheme = parsed.scheme.lower()

    downloaded_files: list[str] = []
    try:
        if scheme in {"ftp", "ftps"}:
            downloaded_files = _download_ftp_tree(
                ftp_url,
                target_dir,
                include_extensions=include_extensions,
                max_files=max_files,
                skip_existing=skip_existing,
            )
        elif scheme in {"http", "https"}:
            filename = posixpath.basename(parsed.path) or f"{geo_id or 'geo_dataset'}.dat"
            downloaded_files = _download_http_resource(
                ftp_url,
                target_dir / filename,
                skip_existing=skip_existing,
            )
        else:
            raise ValueError(f"Unsupported scheme for GEO download: {scheme}")
    except Exception as exc:  # noqa: BLE001
        print(f"[GEO Download] Failed to download from {ftp_url}: {exc}")
        return None

    if cache is not None:
        cache.add(cache_key)

    result = {
        "ftp_url": ftp_url,
        "destination": str(target_dir),
        "files": downloaded_files,
        "cached": False,
    }
    if geo_id:
        result["geo_id"] = geo_id
    return result
