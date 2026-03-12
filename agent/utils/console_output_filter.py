"""
控制台输出过滤器 - 过滤敏感信息并转换为用户友好的进度信息
"""

import re
from typing import Optional
from datetime import datetime


class ConsoleOutputFilter:
    """简化的控制台输出过滤器"""

    def __init__(self):
        self.last_summary = ""
        self.buffer = []

    def add_output(self, text: str) -> Optional[str]:
        """处理新的控制台输出"""
        if not text or not text.strip():
            return None

        self.buffer.append(text)

        # 保持缓冲区在合理大小
        if len(self.buffer) > 50:
            self.buffer = self.buffer[-20:]

        # 合并最近的文本
        combined_text = "\n".join(self.buffer[-10:])

        # 过滤并生成总结
        summary = self._quick_filter_and_summarize(combined_text)

        if summary != self.last_summary:
            self.last_summary = summary
            return summary

        return None

    def _quick_filter_and_summarize(self, text: str) -> str:
        """快速过滤和总结"""
        lines = text.split("\n")
        events = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 过滤敏感信息
            line = self._remove_sensitive(line)

            # 提取关键事件
            event = self._extract_event(line)
            if event:
                events.append(event)

        # 生成总结
        return self._generate_summary(events[-5:])  # 只使用最近5个事件

    def _remove_sensitive(self, line: str) -> str:
        """移除敏感信息"""
        # 移除API keys
        line = re.sub(
            r'api[_-]?key["\']?\s*[:=]\s*["\']?[^"\']+',
            "[API_KEY]",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r'token["\']?\s*[:=]\s*["\']?[^"\']+', "[TOKEN]", line, flags=re.IGNORECASE
        )
        line = re.sub(r"Bearer\s+[\w\-\.]+", "[TOKEN]", line, flags=re.IGNORECASE)

        # 移除密码和密钥
        line = re.sub(
            r'password["\']?\s*[:=]\s*["\']?[^"\']+',
            "[PASSWORD]",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r'secret["\']?\s*[:=]\s*["\']?[^"\']+',
            "[SECRET]",
            line,
            flags=re.IGNORECASE,
        )

        # 移除IP地址和端口
        line = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP_ADDRESS]", line)
        line = re.sub(r"port=\d+", "[PORT]", line, flags=re.IGNORECASE)
        line = re.sub(r"domain=[^\s,]+", "[DOMAIN]", line, flags=re.IGNORECASE)

        # 移除配置文件路径
        line = re.sub(r"[/\\][^/\\]*[/\\]config[/\\][^/\\]*\.json", "[CONFIG]", line)
        line = re.sub(r"\.env", "[CONFIG_FILE]", line)

        # 移除详细的错误堆栈
        line = re.sub(r"at\s+.*\.js:\d+:\d+", "[STACK_TRACE]", line)
        line = re.sub(r"Traceback.*", "[ERROR_STACK]", line)

        # 移除过长的配置字符串
        line = re.sub(r"\{[^}]{100,}\}", "[CONFIG]", line)
        line = re.sub(r"\[[^\]]{50,}\]", "[CONFIG]", line)

        return line

    def _extract_event(self, line: str) -> Optional[str]:
        """提取事件信息"""
        # 成功事件
        if re.search(r"[SUCCESS]|[OK]|完成|completed|成功|success", line, flags=re.IGNORECASE):
            return f"success {self._clean_line(line)}"

        # 错误事件
        if re.search(
            r"[ERROR]|[FAIL]|错误|失败|error|failed|exception", line, flags=re.IGNORECASE
        ):
            return f"error {self._clean_line(line)}"

        # 进度事件
        if re.search(
            r"执行|处理|加载|分析|运行|开始|start|process|execute",
            line,
            flags=re.IGNORECASE,
        ):
            return f"progress {self._clean_line(line)}"

        # 结果事件
        if re.search(
            r"生成|创建|文件|结果|输出|generated|created|files|output",
            line,
            flags=re.IGNORECASE,
        ):
            return f"result {self._clean_line(line)}"

        # 其他重要信息
        if re.search(
            r"\[.*Graph\]|[OpenCode]|sandbox|LLM|模型", line, flags=re.IGNORECASE
        ):
            return f"info {self._clean_line(line)}"

        return None

    def _clean_line(self, line: str) -> str:
        """清理行内容"""
        # 移除前缀
        line = re.sub(r"^\[[^\]]+\]\s*", "", line)

        # 移除时间戳
        line = re.sub(r"^\d{1,2}:\d{2}:\d{2}\s*", "", line)

        # 清理多余空格
        line = re.sub(r"\s+", " ", line).strip()

        # 限制长度
        if len(line) > 80:
            line = line[:77] + "..."

        return line

    def _generate_summary(self, events: list) -> str:
        """生成事件总结"""
        if not events:
            return "system processing"

        # 去重并分类
        successes = []
        errors = []
        progresses = []
        results = []
        others = []

        for event in events:
            if event.startswith("success"):
                successes.append(event[7:].strip())  # 移除前缀
            elif event.startswith("error"):
                errors.append(event[6:].strip())
            elif event.startswith("progress"):
                progresses.append(event[9:].strip())
            elif event.startswith("result"):
                results.append(event[7:].strip())
            else:
                others.append(event[5:].strip())

        # 构建总结
        summary_parts = []

        if errors:
            if len(errors) == 1:
                summary_parts.append(f"warning {errors[0]}")
            else:
                summary_parts.append(f"warning encountered {len(errors)} issues")

        if progresses:
            # 只显示最新的进度
            summary_parts.append(f"progress {progresses[-1]}")

        if results:
            if len(results) == 1:
                summary_parts.append(f"result {results[0]}")
            else:
                summary_parts.append(f"generated {len(results)} results")

        if successes:
            if len(successes) == 1:
                summary_parts.append(f"success {successes[0]}")
            else:
                summary_parts.append(f"completed {len(successes)} tasks")

        if not summary_parts and others:
            # 如果没有分类，但还有其他事件，显示最新的
            if others:
                summary_parts.append(f"info {others[-1]}")

        return " | ".join(summary_parts) if summary_parts else "system processing"

    def get_current_status(self) -> str:
        """获取当前状态"""
        return self.last_summary or "系统初始化中..."
