"""
结构化日志模块 - StructuredLogger

提供JSON格式的结构化日志输出
"""

import logging
import sys
import json
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import threading

# 尝试导入structlog，如果不可用则使用标准logging
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False


class JSONFormatter(logging.Formatter):
    """JSON格式日志formatter"""
    
    def __init__(self, include_fields: Optional[list] = None):
        """
        初始化JSON formatter
        
        Args:
            include_fields: 要包含的额外字段
        """
        super().__init__()
        self.include_fields = include_fields or []
        self._lock = threading.Lock()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录为JSON
        
        Args:
            record: logging.LogRecord
            
        Returns:
            JSON格式的日志字符串
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        # 添加自定义字段
        for field in self.include_fields:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)
        
        # 添加请求ID（如果存在）
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        
        # 添加用户ID（如果存在）
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        
        return json.dumps(log_data, ensure_ascii=False)


class StructuredLogger:
    """
    结构化日志记录器
    
    提供JSON格式的日志输出，支持日志级别过滤、日志格式化等功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化结构化日志器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._initialized = False
        self._loggers: Dict[str, logging.Logger] = {}
        self._request_id = None
        self._user_id = None
        
        # 配置参数
        self._log_level = getattr(logging, self.config.get('level', 'INFO').upper())
        self._log_file = self.config.get('log_file')
        self._json_format = self.config.get('json_format', True)
        self._include_fields = self.config.get('include_fields', [])
        self._add_timestamp = self.config.get('add_timestamp', True)
        self._add_logger_name = self.config.get('add_logger_name', True)
        
        # 设置根日志器
        self._setup_root_logger()
    
    def _setup_root_logger(self) -> None:
        """设置根日志器"""
        self._root_logger = logging.getLogger()
        self._root_logger.setLevel(self._log_level)
        
        # 清除现有处理器
        self._root_logger.handlers.clear()
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self._log_level)
        
        if self._json_format:
            console_handler.setFormatter(JSONFormatter(self._include_fields))
        else:
            console_handler.setFormatter(
                logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s')
            )
        
        self._root_logger.addHandler(console_handler)
        
        # 添加文件处理器（如果配置）
        if self._log_file:
            self._add_file_handler()
    
    def _add_file_handler(self) -> None:
        """添加文件处理器"""
        try:
            log_path = Path(self._log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(self._log_file, encoding='utf-8')
            file_handler.setLevel(self._log_level)
            
            if self._json_format:
                file_handler.setFormatter(JSONFormatter(self._include_fields))
            else:
                file_handler.setFormatter(
                    logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s')
                )
            
            self._root_logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"警告: 无法创建日志文件处理器: {e}", file=sys.stderr)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        获取命名日志器
        
        Args:
            name: 日志器名称
            
        Returns:
            logging.Logger实例
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self._log_level)
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def configure_structlog(self) -> None:
        """配置structlog（如果可用）"""
        if not STRUCTLOG_AVAILABLE:
            return
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True
        )
    
    def initialize(self) -> None:
        """初始化日志系统"""
        if self._initialized:
            return
        
        if STRUCTLOG_AVAILABLE:
            self.configure_structlog()
        
        self._initialized = True
        self.info("日志系统初始化完成", event="initialization")
    
    def shutdown(self) -> None:
        """关闭日志系统"""
        handlers = self._root_logger.handlers[:]
        for handler in handlers:
            handler.close()
            self._root_logger.removeHandler(handler)
        
        self._loggers.clear()
        self._initialized = False
    
    # ==================== 日志记录方法 ====================
    
    def debug(self, message: str, **extra) -> None:
        """DEBUG级别日志"""
        self._log('DEBUG', message, **extra)
    
    def info(self, message: str, **extra) -> None:
        """INFO级别日志"""
        self._log('INFO', message, **extra)
    
    def warning(self, message: str, **extra) -> None:
        """WARNING级别日志"""
        self._log('WARNING', message, **extra)
    
    def warn(self, message: str, **extra) -> None:
        """WARNING级别日志（别名）"""
        self._log('WARNING', message, **extra)
    
    def error(self, message: str, **extra) -> None:
        """ERROR级别日志"""
        self._log('ERROR', message, **extra)
    
    def exception(self, message: str, **extra) -> None:
        """ERROR级别日志（带异常信息）"""
        self._log('ERROR', message, exc_info=True, **extra)
    
    def critical(self, message: str, **extra) -> None:
        """CRITICAL级别日志"""
        self._log('CRITICAL', message, **extra)
    
    def _log(self, level: str, message: str, exc_info: bool = False, **extra) -> None:
        """
        内部日志记录方法
        
        Args:
            level: 日志级别
            message: 日志消息
            exc_info: 是否包含异常信息
            **extra: 额外字段
        """
        # 添加上下文信息
        extra_data = {
            'extra_data': extra
        }
        
        # 添加请求ID和用户ID
        if self._request_id:
            extra_data['request_id'] = self._request_id
        if self._user_id:
            extra_data['user_id'] = self._user_id
        
        logger = self.get_logger('app')
        
        # 根据级别调用相应的日志方法
        log_method = getattr(logger, level.lower())
        
        if exc_info:
            log_method(message, extra=extra_data)
        else:
            log_method(message, extra=extra_data)
    
    # ==================== 上下文管理 ====================
    
    def set_request_id(self, request_id: Optional[str]) -> None:
        """
        设置请求ID
        
        Args:
            request_id: 请求ID
        """
        self._request_id = request_id
    
    def set_user_id(self, user_id: Optional[str]) -> None:
        """
        设置用户ID
        
        Args:
            user_id: 用户ID
        """
        self._user_id = user_id
    
    def clear_context(self) -> None:
        """清除上下文信息"""
        self._request_id = None
        self._user_id = None
    
    # ==================== 便捷方法 ====================
    
    def log_request(self, method: str, path: str, status_code: int, 
                    latency: float, request_id: Optional[str] = None) -> None:
        """
        记录HTTP请求日志
        
        Args:
            method: HTTP方法
            path: 请求路径
            status_code: 状态码
            latency: 响应延迟（秒）
            request_id: 请求ID
        """
        self._log('INFO', f"{method} {path} {status_code}",
                 request_id=request_id,
                 event="http_request",
                 method=method,
                 path=path,
                 status_code=status_code,
                 latency=latency,
                 latency_ms=round(latency * 1000, 2))
    
    def log_database_query(self, query: str, duration: float, 
                           success: bool = True, **extra) -> None:
        """
        记录数据库查询日志
        
        Args:
            query: 查询语句
            duration: 查询耗时（秒）
            success: 是否成功
            **extra: 额外字段
        """
        level = 'ERROR' if not success else 'DEBUG'
        self._log(level, f"Database query: {query[:100]}...",
                 event="db_query",
                 query=query[:200] if query else "",
                 duration=duration,
                 duration_ms=round(duration * 1000, 2),
                 success=success,
                 **extra)
    
    def log_agent_event(self, agent_id: str, event: str, 
                        duration: Optional[float] = None, **extra) -> None:
        """
        记录Agent事件日志
        
        Args:
            agent_id: Agent ID
            event: 事件类型
            duration: 耗时（秒）
            **extra: 额外字段
        """
        self._log('INFO', f"Agent {agent_id}: {event}",
                 event=event,
                 agent_id=agent_id,
                 duration=duration,
                 duration_ms=round(duration * 1000, 2) if duration else None,
                 **extra)
    
    def log_alert(self, alert_name: str, severity: str, 
                  message: str, **extra) -> None:
        """
        记录告警日志
        
        Args:
            alert_name: 告警名称
            severity: 严重程度
            message: 告警消息
            **extra: 额外字段
        """
        level = 'WARNING' if severity in ['warning', 'warn'] else 'ERROR'
        self._log(level, message,
                 event="alert",
                 alert_name=alert_name,
                 severity=severity,
                 **extra)
    
    # ==================== 统计信息 ====================
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取日志系统状态
        
        Returns:
            状态信息
        """
        return {
            "initialized": self._initialized,
            "log_level": logging.getLevelName(self._log_level),
            "json_format": self._json_format,
            "log_file": self._log_file,
            "loggers_count": len(self._loggers)
        }


# 全局实例
_structured_logger: Optional[StructuredLogger] = None


def get_logger(name: str = 'app') -> logging.Logger:
    """
    获取日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        logging.Logger实例
    """
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger()
    return _structured_logger.get_logger(name)


def init_logger(config: Optional[Dict[str, Any]] = None) -> StructuredLogger:
    """
    初始化日志系统
    
    Args:
        config: 配置字典
        
    Returns:
        StructuredLogger实例
    """
    global _structured_logger
    _structured_logger = StructuredLogger(config)
    _structured_logger.initialize()
    return _structured_logger
