"""
LLM Tracing API端点 v2.2
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

from backend.llm.tracing import llm_tracer
from backend.core.auth import get_current_user

router = APIRouter()

class StartTraceModel(BaseModel):
    name: str
    metadata: Optional[Dict[str, Any]] = None

class EndTraceModel(BaseModel):
    trace_id: Optional[str] = None

class StartSpanModel(BaseModel):
    name: str
    span_type: str  # llm, chain, tool, custom
    inputs: Optional[Dict[str, Any]] = None
    parent_span_id: Optional[str] = None

class EndSpanModel(BaseModel):
    span_id: str
    outputs: Optional[Dict[str, Any]] = None
    status: str = "ok"
    error_message: Optional[str] = None

class TraceLLMModel(BaseModel):
    model: str
    prompt: str
    call_type: str = "openai"  # openai, langchain, llama_index

@router.post("/traces")
async def start_trace(request: StartTraceModel):
    """
    开始一个Trace
    
    v2.2: LLM Tracing
    """
    trace_id = llm_tracer.start_trace(
        name=request.name,
        metadata=request.metadata
    )
    
    return {
        "trace_id": trace_id,
        "name": request.name,
        "message": "Trace started"
    }

@router.delete("/traces")
async def end_trace(request: EndTraceModel):
    """
    结束一个Trace
    
    v2.2: LLM Tracing
    """
    llm_tracer.end_trace(request.trace_id)
    
    return {"message": "Trace ended"}

@router.get("/traces")
async def list_traces(
    limit: int = 100,
    offset: int = 0
):
    """
    列出所有Traces
    
    v2.2: LLM Tracing
    """
    traces = llm_tracer.list_traces(limit=limit, offset=offset)
    
    return {
        "total": len(traces),
        "traces": traces
    }

@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    """
    获取Trace详情
    
    v2.2: LLM Tracing
    """
    trace = llm_tracer.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    return {
        "trace_id": trace.trace_id,
        "name": trace.name,
        "spans_count": len(trace.spans),
        "metadata": trace.metadata,
        "created_at": trace.created_at.isoformat(),
        "spans": [
            {
                "span_id": s.span_id,
                "parent_span_id": s.parent_span_id,
                "name": s.name,
                "type": s.span_type,
                "status": s.status,
                "duration_ms": s.duration_ms,
                "created_at": s.created_at.isoformat()
            }
            for s in trace.spans
        ]
    }

@router.post("/spans")
async def start_span(request: StartSpanModel):
    """
    开始一个Span
    
    v2.2: LLM Tracing
    """
    span_id = llm_tracer.start_span(
        name=request.name,
        span_type=request.span_type,
        inputs=request.inputs,
        parent_span_id=request.parent_span_id
    )
    
    return {
        "span_id": span_id,
        "trace_id": llm_tracer.current_trace_id
    }

@router.put("/spans/{span_id}")
async def end_span(span_id: str, request: EndSpanModel):
    """
    结束一个Span
    
    v2.2: LLM Tracing
    """
    llm_tracer.end_span(
        span_id=request.span_id,
        outputs=request.outputs,
        status=request.status,
        error_message=request.error_message
    )
    
    return {"message": "Span ended"}

@router.get("/traces/{trace_id}/tree")
async def get_span_tree(trace_id: str):
    """
    获取Span树结构
    
    v2.2: LLM Tracing
    """
    tree = llm_tracer.get_span_tree(trace_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    return tree

@router.get("/traces/{trace_id}/export")
async def export_trace(trace_id: str, format: str = "json"):
    """
    导出Trace
    
    v2.2: LLM Tracing
    """
    data = llm_tracer.export_trace(trace_id, format=format)
    if not data:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    return {"exported_data": data}

@router.post("/trace-llm")
async def trace_llm_call(request: TraceLLMModel):
    """
    追踪LLM调用 (简化版)
    
    v2.2: LLM Tracing
    """
    # 模拟LLM调用
    span_id = llm_tracer.start_span(
        name=f"llm:{request.model}",
        span_type="llm",
        inputs={"model": request.model, "prompt": request.prompt}
    )
    
    # 模拟响应
    response = {
        "id": f"chatcmpl-{uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Response from {request.model}"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": 10,
            "total_tokens": len(request.prompt.split()) + 10
        }
    }
    
    llm_tracer.end_span(span_id, outputs=response)
    
    return response

# 便捷装饰器
def traced(name: str, span_type: str = "custom"):
    """装饰器方式追踪函数"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            span_id = llm_tracer.start_span(name=name, span_type=span_type)
            try:
                result = await func(*args, **kwargs)
                llm_tracer.end_span(span_id, outputs={"result": str(result)[:1000]})
                return result
            except Exception as e:
                llm_tracer.end_span(span_id, status="error", error_message=str(e))
                raise
        return wrapper
    return decorator

# 辅助函数
def uuid4():
    import uuid
    return uuid.uuid4()
