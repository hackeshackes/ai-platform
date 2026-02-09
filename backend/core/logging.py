"""Logging configuration"""
from loguru import logger
import sys

# 配置日志格式
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="DEBUG"
)

logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO"
)

__all__ = ["logger"]
