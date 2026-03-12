"""
控制台输出重定向器 - 线程安全版本

使用 threading.local() 为每个线程维护独立的 redirector，
避免多会话并发时输出混乱。
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


# 线程本地存储，用于隔离不同会话的输出
_thread_local = threading.local()

# 全局的 stdout 捕获器（只设置一次）
_global_stream_capture = None
_original_stdout = None
_install_lock = threading.Lock()
_installed = False


class _ThreadAwareStreamCapture(io.StringIO):
    """
    线程感知的流捕获器

    所有线程共享同一个 sys.stdout，但写入时会查找当前线程的 redirector
    """

    def __init__(self):
        super().__init__()

    def write(self, text: str) -> int:
        """写入时，根据当前线程路由到对应的 redirector"""
        if not text:
            return 0

        # 获取当前线程的 redirector
        redirector = getattr(_thread_local, "redirector", None)

        if redirector and redirector._is_capturing:
            # 路由到当前线程的 redirector
            redirector._on_write(text)

        # 同时写入原始 stdout（用于调试）
        if _original_stdout:
            try:
                _original_stdout.write(text)
                _original_stdout.flush()
            except Exception:
                pass

        return len(text) if text else 0

    def flush(self):
        """刷新"""
        if _original_stdout:
            try:
                _original_stdout.flush()
            except Exception:
                pass


def _ensure_global_capture_installed():
    """
    确保全局的 stdout 捕获器已安装（只执行一次）

    这个函数会在第一次创建 ConsoleOutputRedirector 时调用，
    替换 sys.stdout 为线程感知的捕获器。
    """
    global _global_stream_capture, _original_stdout, _installed

    with _install_lock:
        if _installed:
            return

        # 保存原始 stdout
        _original_stdout = sys.stdout

        # 创建并安装全局捕获器
        _global_stream_capture = _ThreadAwareStreamCapture()
        sys.stdout = _global_stream_capture

        _installed = True
        print("[ConsoleOutputRedirector] Global capture installed (thread-safe mode)")


class ConsoleOutputRedirector:
    """
    控制台输出重定向器（线程安全版本）

    每个会话创建一个实例，通过 threading.local 隔离不同会话的输出。

    使用方式:
        redirector = ConsoleOutputRedirector(progress_callback=callback)
        redirector.start_capture()
        try:
            # 执行代码...
        finally:
            redirector.stop_capture()
    """

    def __init__(
        self,
        progress_callback: Optional[Callable] = None,
        capture_print: bool = True,
        min_interval_ms: int = 100,
    ):
        self.progress_callback = progress_callback
        self.capture_print = capture_print
        self.min_interval_ms = min_interval_ms

        self._is_capturing = False
        self._lock = threading.Lock()
        self._last_flush_time = datetime.now()
        self._buffer = ""
        self._last_sent = ""

        # 初始化过滤器
        self._filter = ConsoleOutputFilter() if ConsoleOutputFilter else None

        # 确保全局捕获器已安装
        _ensure_global_capture_installed()

    def start_capture(self):
        """开始捕获（设置当前线程的 redirector）"""
        if self._is_capturing:
            return

        with self._lock:
            self._is_capturing = True
            self._buffer = ""
            self._last_sent = ""

            # 将当前 redirector 绑定到线程本地存储
            _thread_local.redirector = self

    def stop_capture(self):
        """停止捕获（清除当前线程的 redirector）"""
        if not self._is_capturing:
            return

        with self._lock:
            self._is_capturing = False
            self._flush(final=True)

            # 清除线程本地的 redirector
            _thread_local.redirector = None

    def __enter__(self):
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_capture()
        return False

    def _on_write(self, text: str):
        """处理写入（由 _ThreadAwareStreamCapture 调用）"""
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


# ============================================================================
# 兼容旧的全局 API（已废弃，但保留以兼容旧代码）
# ============================================================================

_global_redirector: Optional[ConsoleOutputRedirector] = None


def get_global_redirector() -> Optional[ConsoleOutputRedirector]:
    """获取全局重定向器（已废弃）"""
    return _global_redirector


def set_global_redirector(redirector: Optional[ConsoleOutputRedirector]):
    """设置全局重定向器（已废弃）"""
    global _global_redirector
    _global_redirector = redirector


def get_thread_redirector() -> Optional[ConsoleOutputRedirector]:
    """获取当前线程的重定向器（推荐使用）"""
    return getattr(_thread_local, "redirector", None)


def set_thread_redirector(redirector: Optional[ConsoleOutputRedirector]):
    """
    设置当前线程的重定向器（用于ThreadPoolExecutor场景）

    当在新线程中执行代码时，需要显式设置当前线程的redirector，
    以确保控制台输出能够被正确捕获。

    Args:
        redirector: ConsoleOutputRedirector 实例或 None
    """
    _thread_local.redirector = redirector
