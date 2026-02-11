"""
链路追踪模块 - TracingManager

提供分布式链路追踪功能
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """Span类型"""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Span状态"""
    OK = "ok"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class SpanContext:
    """Span上下文"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    is_sampled: bool = True
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class Span:
    """Span（追踪单元）"""
    name: str
    kind: SpanKind
    context: SpanContext
    status: SpanStatus = SpanStatus.OK
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    error_stack: Optional[str] = None
    
    def set_attribute(self, key: str, value: Any) -> None:
        """设置属性"""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """添加事件"""
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {}
        })
    
    def add_log(self, message: str, level: str = "info", **kwargs) -> None:
        """添加日志"""
        self.logs.append({
            "message": message,
            "level": level,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        })
    
    def set_error(self, message: str, stack: Optional[str] = None) -> None:
        """设置错误"""
        self.status = SpanStatus.ERROR
        self.error_message = message
        self.error_stack = stack
    
    def finish(self) -> None:
        """结束Span"""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "context": asdict(self.context),
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "logs": self.logs,
            "error_message": self.error_message,
            "error_stack": self.error_stack
        }


class TracingManager:
    """
    链路追踪管理器
    
    提供Trace/Span创建、管理和导出功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化追踪管理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._initialized = False
        self._spans: List[Span] = []
        self._current_span: Optional[Span] = None
        self._span_stack: List[Span] = []
        
        # 配置
        self._service_name = self.config.get('service_name', 'ai-platform')
        self._sampling_rate = self.config.get('sampling_rate', 1.0)
        self._max_spans = self.config.get('max_spans', 1000)
        self._export_on_finish = self.config.get('export_on_finish', True)
        
        # 导出器
        self._exporters: List[Callable] = []
        
        # 生成器
        self._span_id_generator = self._generate_span_id
        self._trace_id_generator = self._generate_trace_id
    
    def _generate_span_id(self) -> str:
        """生成Span ID"""
        return uuid.uuid4().hex[:16]
    
    def _generate_trace_id(self) -> str:
        """生成Trace ID"""
        return uuid.uuid4().hex
    
    async def initialize(self) -> None:
        """初始化追踪管理器"""
        self._initialized = True
        logger.info("链路追踪管理器初始化完成")
    
    async def shutdown(self) -> None:
        """关闭追踪管理器"""
        self._spans.clear()
        self._span_stack.clear()
        self._current_span = None
        self._initialized = False
        logger.info("链路追踪管理器已关闭")
    
    # ==================== Trace/Span管理 ====================
    
    def start_trace(self, 
                    name: str,
                    kind: SpanKind = SpanKind.INTERNAL,
                    attributes: Optional[Dict[str, Any]] = None,
                    parent_context: Optional[SpanContext] = None) -> Span:
        """
        开始新的Trace
        
        Args:
            name: Trace名称
            kind: Span类型
            attributes: 初始属性
            parent_context: 父Span上下文
            
        Returns:
            Span实例
        """
        # 决定是否采样
        is_sampled = self._should_sample()
        
        # 生成ID
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
            baggage = parent_context.baggage.copy()
        else:
            trace_id = self._trace_id_generator()
            parent_span_id = None
            baggage = {}
        
        context = SpanContext(
            trace_id=trace_id,
            span_id=self._span_id_generator(),
            parent_span_id=parent_span_id,
            is_sampled=is_sampled,
            baggage=baggage
        )
        
        span = Span(
            name=name,
            kind=kind,
            context=context,
            attributes=attributes or {}
        )
        
        span.set_attribute("service.name", self._service_name)
        
        # 保存当前span
        self._current_span = span
        self._span_stack.append(span)
        self._spans.append(span)
        
        # 限制span数量
        if len(self._spans) > self._max_spans:
            self._spans = self._spans[-self._max_spans:]
        
        logger.debug(f"开始Trace: {trace_id[:8]}... - {name}")
        
        return span
    
    def start_span(self,
                   name: str,
                   kind: SpanKind = SpanKind.INTERNAL,
                   attributes: Optional[Dict[str, Any]] = None) -> Span:
        """
        开始新的Span（作为当前Span的子Span）
        
        Args:
            name: Span名称
            kind: Span类型
            attributes: 初始属性
            
        Returns:
            Span实例
        """
        if self._current_span:
            parent_context = self._current_span.context
        else:
            parent_context = None
        
        return self.start_trace(name, kind, attributes, parent_context)
    
    def finish_span(self, span: Optional[Span] = None) -> Span:
        """
        结束Span
        
        Args:
            span: Span实例，默认为当前Span
            
        Returns:
            结束的Span
        """
        if span is None:
            span = self._span_stack.pop() if self._span_stack else self._current_span
        else:
            if span in self._span_stack:
                self._span_stack.remove(span)
        
        if span:
            span.finish()
            
            # 更新父span
            if self._span_stack:
                self._current_span = self._span_stack[-1]
            else:
                self._current_span = None
            
            # 导出
            if self._export_on_finish:
                self._export_span(span)
            
            logger.debug(f"结束Span: {span.name} - {span.duration_ms:.2f}ms")
        
        return span
    
    @contextmanager
    def trace(self,
              name: str,
              kind: SpanKind = SpanKind.INTERNAL,
              attributes: Optional[Dict[str, Any]] = None):
        """
        Trace上下文管理器
        
        用法:
            with tracer.trace("operation"):
                # 操作
        """
        span = self.start_trace(name, kind, attributes)
        try:
            yield span
        except Exception as e:
            span.set_error(str(e))
            span.error_stack = getattr(e, '__traceback__', None)
            raise
        finally:
            self.finish_span(span)
    
    @contextmanager
    def span(self,
             name: str,
             kind: SpanKind = SpanKind.INTERNAL,
             attributes: Optional[Dict[str, Any]] = None):
        """
        Span上下文管理器
        """
        span = self.start_span(name, kind, attributes)
        try:
            yield span
        except Exception as e:
            span.set_error(str(e))
            raise
        finally:
            self.finish_span(span)
    
    # ==================== 当前上下文 ====================
    
    def get_current_span(self) -> Optional[Span]:
        """获取当前Span"""
        return self._current_span
    
    def get_current_trace_id(self) -> Optional[str]:
        """获取当前Trace ID"""
        if self._current_span:
            return self._current_span.context.trace_id
        return None
    
    def get_current_span_id(self) -> Optional[str]:
        """获取当前Span ID"""
        if self._current_span:
            return self._current_span.context.span_id
        return None
    
    def add_span_attribute(self, key: str, value: Any) -> None:
        """添加Span属性"""
        if self._current_span:
            self._current_span.set_attribute(key, value)
    
    def add_span_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """添加Span事件"""
        if self._current_span:
            self._current_span.add_event(name, attributes)
    
    # ==================== 采样控制 ====================
    
    def _should_sample(self) -> bool:
        """决定是否采样"""
        import random
        return random.random() < self._sampling_rate
    
    def set_sampling_rate(self, rate: float) -> None:
        """
        设置采样率
        
        Args:
            rate: 采样率 (0.0 - 1.0)
        """
        self._sampling_rate = max(0.0, min(1.0, rate))
    
    def force_sample(self) -> None:
        """强制采样"""
        self._sampling_rate = 1.0
    
    def force_no_sample(self) -> None:
        """强制不采样"""
        self._sampling_rate = 0.0
    
    # ==================== 导出 ====================
    
    def register_exporter(self, exporter: Callable) -> None:
        """
        注册Span导出器
        
        Args:
            exporter: 导出函数，接收Span列表
        """
        self._exporters.append(exporter)
    
    def _export_span(self, span: Span) -> None:
        """导出Span"""
        for exporter in self._exporters:
            try:
                exporter([span])
            except Exception as e:
                logger.error(f"Span导出失败: {e}")
    
    def export_all(self) -> List[Dict[str, Any]]:
        """
        导出所有Spans
        
        Returns:
            Span字典列表
        """
        # 结束所有未完成的span
        while self._span_stack:
            self.finish_span()
        
        # 转换为字典
        result = [span.to_dict() for span in self._spans]
        
        # 发送到所有导出器
        for exporter in self._exporters:
            try:
                exporter(self._spans)
            except Exception as e:
                logger.error(f"批量导出失败: {e}")
        
        return result
    
    def to_json(self) -> str:
        """
        导出为JSON字符串
        
        Returns:
            JSON字符串
        """
        import json
        return json.dumps(self.export_all(), ensure_ascii=False, indent=2)
    
    # ==================== HTTP中间件 ====================
    
    async def http_middleware(self, request, call_next):
        """
        HTTP请求追踪中间件
        
        用法:
            app.middleware("http")(tracing.http_middleware)
        """
        from fastapi import Request
        
        # 生成或提取请求ID
        request_id = request.headers.get("X-Request-ID", self._generate_trace_id())
        
        # 开始Trace
        with self.trace(
            f"{request.method} {request.url.path}",
            kind=SpanKind.SERVER,
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.target": request.url.path,
                "http.host": request.headers.get("host", ""),
                "http.user_agent": request.headers.get("user-agent", ""),
                "http.request_id": request_id
            }
        ) as span:
            # 记录请求头
            span.add_event("request_received")
            
            # 调用下一个中间件/路由
            try:
                response = await call_next(request)
                span.set_attribute("http.status_code", response.status_code)
                
                if response.status_code >= 400:
                    span.set_error(f"HTTP {response.status_code}")
                
                span.add_event("response_sent")
                
            except Exception as e:
                span.set_error(str(e))
                span.error_stack = getattr(e, '__traceback__', None)
                raise
        
        return response
    
    # ==================== 状态查询 ====================
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取追踪管理器状态
        
        Returns:
            状态信息
        """
        return {
            "initialized": self._initialized,
            "service_name": self._service_name,
            "sampling_rate": self._sampling_rate,
            "spans_count": len(self._spans),
            "active_spans_count": len(self._span_stack),
            "exporters_count": len(self._exporters)
        }
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """
        获取指定Trace的所有Spans
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Span列表
        """
        return [span for span in self._spans if span.context.trace_id == trace_id]
    
    def clear(self) -> None:
        """清除所有Spans"""
        self._spans.clear()
        self._span_stack.clear()
        self._current_span = None


# 全局实例
_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager() -> TracingManager:
    """获取追踪管理器单例"""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager()
    return _tracing_manager


def init_tracing(config: Optional[Dict[str, Any]] = None) -> TracingManager:
    """
    初始化追踪系统
    
    Args:
        config: 配置字典
        
    Returns:
        TracingManager实例
    """
    global _tracing_manager
    _tracing_manager = TracingManager(config)
    _tracing_manager.initialize()
    return _tracing_manager


# 便捷函数
def get_current_trace_id() -> Optional[str]:
    """获取当前Trace ID"""
    return get_tracing_manager().get_current_trace_id()


def get_current_span_id() -> Optional[str]:
    """获取当前Span ID"""
    return get_tracing_manager().get_current_span_id()


def add_span_attribute(key: str, value: Any) -> None:
    """添加Span属性"""
    get_tracing_manager().add_span_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """添加Span事件"""
    get_tracing_manager().add_span_event(name, attributes)
