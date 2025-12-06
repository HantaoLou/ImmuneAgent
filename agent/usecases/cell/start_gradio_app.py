#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抗体分析工作流 Gradio Web界面启动器

这个脚本启动基于Gradio的Web界面，为LangGraph抗体分析工作流提供可视化交互。
"""

import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gradio_interface import create_gradio_app


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("gradio_app.log", encoding="utf-8"),
        ],
    )


def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("正在启动抗体分析工作流 Gradio Web界面...")

        # 创建Gradio应用
        app = create_gradio_app()

        # 启动应用
        logger.info("Gradio应用已创建，正在启动服务器...")
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True,
            quiet=False,
        )

    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭应用...")
    except Exception as e:
        logger.error(f"启动应用时发生错误: {str(e)}")
        raise
    finally:
        logger.info("应用已关闭")


if __name__ == "__main__":
    main()
