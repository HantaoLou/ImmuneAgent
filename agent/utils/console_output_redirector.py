"""
控制台输出重定向器 - 简化版本
只捕获print输出，避免复杂逻辑
"""

import sys
import io
import threading
from datetime import datetime
from typing import Callable, Dict, Any, Optional

try:
    from .console_output_filter import ConsoleOutputFilter
except ImportError:
    ConsoleOutputFilter = None


class ConsoleOutputRedirector:
    """控制台输出重定向器（简化版）"""

    def __init__(
        self,
        progress_callback: Optional[Callable] = None,
        capture_print: bool = True,
        min_interval_ms: int = 100,
    ):
        self.progress_callback = progress_callback
        self.capture_print = capture_print
        self.min_interval_ms = min_interval_ms

        self._original_stdout = None
        self._is_capturing = False
        self._lock = threading.Lock()
        self._last_flush_time = datetime.now()
        self._buffer = ""
        self._last_sent = ""

        # 初始化过滤器
        self._filter = ConsoleOutputFilter() if ConsoleOutputFilter else None

    def start_capture(self):
        """开始捕获"""
        if self._is_capturing:
            return

        with self._lock:
            self._is_capturing = True
            self._buffer = ""
            self._last_sent = ""
            self._original_stdout = sys.stdout
            sys.stdout = _SimpleStreamCapture(self._on_write)

    def stop_capture(self):
        """停止捕获"""
        if not self._is_capturing:
            return

        with self._lock:
            self._is_capturing = False
            if self._original_stdout:
                sys.stdout = self._original_stdout
            self._flush(final=True)

    def __enter__(self):
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_capture()
        return False

    def _on_write(self, text: str):
        """处理写入"""
        if not text:
            return

        try:
            self._buffer += text
            display_text = None

            if self._filter:
                filtered_text = self._filter.add_output(text)
                if filtered_text and filtered_text != self._last_sent:
                    display_text = filtered_text
                    self._buffer = ""
                    self._last_sent = display_text
            else:
                if len(self._buffer) >= 500:
                    display_text = self._buffer[-500:]
                    self._buffer = ""

            if display_text and self.progress_callback:
                self.progress_callback(
                    event_type="console_output",
                    message=display_text,
                    details={
                        "final": False,
                        "timestamp": datetime.now().isoformat(),
                        "filtered": self._filter is not None,
                    },
                )
        except Exception as e:
            pass

        if self._original_stdout:
            try:
                self._original_stdout.write(text)
                self._original_stdout.flush()
            except Exception:
                pass

    def _flush(self, final: bool = False):
        """刷新缓冲区，推送到前端"""
        if not self._buffer:
            return

        text = self._buffer
        self._buffer = ""
        self._last_flush_time = datetime.now()

        if self.progress_callback and text.strip():
            try:
                display_text = text[-500:] if len(text) > 500 else text

                self.progress_callback(
                    event_type="console_output",
                    message=display_text,
                    details={
                        "final": final,
                        "timestamp": datetime.now().isoformat(),
                        "filtered": self._filter is not None,
                    },
                )
            except Exception as e:
                pass

        if self._original_stdout:
            try:
                self._original_stdout.write(text)
                self._original_stdout.flush()
            except Exception:
                pass


class _SimpleStreamCapture(io.StringIO):
    """自定义流捕获器"""

    def __init__(self, callback: Callable[[str], None]):
        super().__init__()
        self._callback = callback

    def write(self, text: str) -> int:
        """写入时触发回调"""
        if text:
            self._callback(text)
        return len(text) if text else 0

    def flush(self):
        """刷新"""
        pass


_global_redirector: Optional[ConsoleOutputRedirector] = None


def get_global_redirector() -> Optional[ConsoleOutputRedirector]:
    """获取全局重定向器"""
    return _global_redirector


def set_global_redirector(redirector: Optional[ConsoleOutputRedirector]):
    """设置全局重定向器"""
    global _global_redirector
    _global_redirector = redirector
