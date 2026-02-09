"""
LLM Tracing模块 v2.2
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import json

@dataclass
class TraceSpan:
    """Trace Span"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    span_type: str  # llm, chain, tool, custom
    inputs: Dict = field(default_factory=dict)
    outputs: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    status: str = "ok"  # ok, error
    error_message: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None

@dataclass
class Trace:
    """完整Trace"""
    trace_id: str
    name: str
    spans: List[TraceSpan] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

class LLMTracer:
    """LLM Tracing服务"""
    
    def __init__(self):
        self.traces: Dict[str, Trace] = {}
        self.current_trace_id: Optional[str] = None
    
    def start_trace(self, name: str, metadata: Optional[Dict] = None) -> str:
        """开始一个Trace"""
        trace_id = str(uuid4())
        trace = Trace(
            trace_id=trace_id,
            name=name,
            metadata=metadata or {}
        )
        self.traces[trace_id] = trace
        self.current_trace_id = trace_id
        return trace_id
    
    def end_trace(self, trace_id: Optional[str] = None):
        """结束Trace"""
        trace_id = trace_id or self.current_trace_id
        if trace_id and trace_id in self.traces:
            trace = self.traces[trace_id]
            # 计算总时长
            if trace.spans:
                start = min(s.start_time for s in trace.spans)
                end = max(s.end_time or s.start_time for s in trace.spans)
                trace.metadata["total_duration_ms"] = (end - start).total_seconds() * 1000
        self.current_trace_id = None
    
    def start_span(
        self,
        name: str,
        span_type: str,
        inputs: Optional[Dict] = None,
        parent_span_id: Optional[str] = None
    ) -> str:
        """开始一个Span"""
        trace_id = self.current_trace_id or str(uuid4())
        
        span = TraceSpan(
            span_id=str(uuid4()),
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name,
            span_type=span_type,
            inputs=inputs or {}
        )
        
        if trace_id not in self.traces:
            self.traces[trace_id] = Trace(trace_id=trace_id, name=name)
        
        self.traces[trace_id].spans.append(span)
        return span.span_id
    
    def end_span(
        self,
        span_id: str,
        outputs: Optional[Dict] = None,
        status: str = "ok",
        error_message: Optional[str] = None
    ):
        """结束一个Span"""
        for trace in self.traces.values():
            for span in trace.spans:
                if span.span_id == span_id:
                    span.end_time = datetime.utcnow()
                    span.outputs = outputs or {}
                    span.status = status
                    span.error_message = error_message
                    span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
                    return
    
    def trace_llm_call(
        self,
        model: str,
        prompt: str,
        call_fn: Callable,
        **kwargs
    ) -> Any:
        """包装LLM调用，自动追踪"""
        span_id = self.start_span(
            name=f"llm:{model}",
            span_type="llm",
            inputs={"prompt": prompt, "model": model, **kwargs}
        )
        
        try:
            result = call_fn(prompt=prompt, **kwargs)
            self.end_span(span_id, outputs={"response": str(result)[:1000]})
            return result
        except Exception as e:
            self.end_span(span_id, status="error", error_message=str(e))
            raise
    
    def trace_chain(
        self,
        chain_name: str,
        call_fn: Callable,
        inputs: Dict
    ) -> Any:
        """包装Chain调用，自动追踪"""
        span_id = self.start_span(
            name=f"chain:{chain_name}",
            span_type="chain",
            inputs=inputs
        )
        
        try:
            result = call_fn(inputs)
            self.end_span(span_id, outputs=result)
            return result
        except Exception as e:
            self.end_span(span_id, status="error", error_message=str(e))
            raise
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """获取Trace"""
        return self.traces.get(trace_id)
    
    def list_traces(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """列出Traces"""
        traces = list(self.traces.values())
        traces.sort(key=lambda t: t.created_at, reverse=True)
        
        return [
            {
                "trace_id": t.trace_id,
                "name": t.name,
                "spans_count": len(t.spans),
                "duration_ms": t.metadata.get("total_duration_ms"),
                "created_at": t.created_at.isoformat()
            }
            for t in traces[offset:offset+limit]
        ]
    
    def get_span_tree(self, trace_id: str) -> Dict[str, Any]:
        """获取Span树结构"""
        trace = self.traces.get(trace_id)
        if not trace:
            return {}
        
        # 构建树
        span_dict = {s.span_id: s for s in trace.spans}
        root_spans = [s for s in trace.spans if s.parent_span_id is None]
        
        def build_tree(span: TraceSpan) -> Dict:
            children = [s for s in trace.spans if s.parent_span_id == span.span_id]
            return {
                "span_id": span.span_id,
                "name": span.name,
                "type": span.span_type,
                "status": span.status,
                "duration_ms": span.duration_ms,
                "children": [build_tree(c) for c in children]
            }
        
        return {
            "trace_id": trace_id,
            "name": trace.name,
            "tree": [build_tree(s) for s in root_spans]
        }
    
    def export_trace(self, trace_id: str, format: str = "json") -> str:
        """导出Trace"""
        trace = self.traces.get(trace_id)
        if not trace:
            return ""
        
        data = {
            "trace_id": trace.trace_id,
            "name": trace.name,
            "created_at": trace.created_at.isoformat(),
            "spans": [
                {
                    "span_id": s.span_id,
                    "parent_span_id": s.parent_span_id,
                    "name": s.name,
                    "type": s.span_type,
                    "status": s.status,
                    "inputs": s.inputs,
                    "outputs": s.outputs,
                    "duration_ms": s.duration_ms,
                    "start_time": s.start_time.isoformat(),
                    "end_time": s.end_time.isoformat() if s.end_time else None
                }
                for s in trace.spans
            ]
        }
        
        return json.dumps(data, indent=2)

# Tracer实例
llm_tracer = LLMTracer()
