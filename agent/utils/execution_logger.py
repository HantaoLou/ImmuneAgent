"""
执行日志捕获器 - 捕获print输出并转换为进度事件
"""

import sys
from io import StringIO
from typing import Optional, Callable
from contextlib import contextmanager


class ExecutionLogger:
    """执行日志捕获器"""

    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self.buffer = StringIO()
        self.original_stdout = None
        self.original_stderr = None

    def start_capture(self):
        """开始捕获stdout和stderr"""
        self.buffer = StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # 创建包装器
        class StreamWrapper:
            def __init__(self, original, logger):
                self.original = original
                self.logger = logger

            def write(self, text):
                # 写入原始输出
                if self.original:
                    self.original.write(text)

                # 写入缓冲区
                self.logger.buffer.write(text)

                # 发送进度事件
                if self.logger.progress_callback and text.strip():
                    self.logger.progress_callback(
                        event_type="info",
                        message=text.strip(),
                        details={"stream": "stdout"},
                    )

            def flush(self):
                if self.original:
                    self.original.flush()
                self.logger.buffer.flush()

        # 替换stdout和stderr
        sys.stdout = StreamWrapper(self.original_stdout, self)
        sys.stderr = StreamWrapper(self.original_stderr, self)

    def stop_capture(self):
        """停止捕获并恢复原始输出"""
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr

    def get_logs(self) -> str:
        """获取捕获的日志"""
        return self.buffer.getvalue()

    def clear(self):
        """清空缓冲区"""
        self.buffer = StringIO()


@contextmanager
def capture_execution_logs(progress_callback: Optional[Callable] = None):
    """上下文管理器：捕获执行日志"""
    logger = ExecutionLogger(progress_callback)
    try:
        logger.start_capture()
        yield logger
    finally:
        logger.stop_capture()


def create_progress_logger(progress_callback: Optional[Callable]):
    """创建进度日志记录器"""

    def log(message: str, level: str = "info", **kwargs):
        """记录日志并发送进度事件"""
        if progress_callback:
            progress_callback(event_type=level, message=message, **kwargs)
        # 同时打印到控制台
        print(message)

    return log
