"""
监控API端点 - Monitoring Endpoints

提供健康检查、性能指标、日志查询和告警状态的REST API
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["监控"])


def get_monitoring_system():
    """获取监控系统依赖（实际项目中可能需要从依赖注入获取）"""
    from .monitoring import get_monitoring_system
    return get_monitoring_system()


# ==================== 响应模型 ====================

class HealthCheckResponse(BaseModel):
    """健康检查响应"""
    status: str
    overall_status: str
    checks: Dict[str, Any]
    summary: Dict[str, int]
    total_duration_ms: float
    timestamp: str


class MetricsResponse(BaseModel):
    """性能指标响应"""
    timestamp: str
    system: Dict[str, Any]
    api: Dict[str, Any]
    database: Dict[str, Any]
    counters: Dict[str, float]
    gauges: Dict[str, float]
    histograms: Dict[str, Any]


class AlertResponse(BaseModel):
    """告警响应"""
    name: str
    severity: str
    status: str
    message: str
    current_value: float
    started_at: str
    evaluation_count: int = 0


class AlertRuleResponse(BaseModel):
    """告警规则响应"""
    name: str
    condition: str
    severity: str
    message: str
    enabled: bool
    duration_seconds: int


class LogQueryResponse(BaseModel):
    """日志查询响应"""
    logs: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int


class TracingResponse(BaseModel):
    """链路追踪响应"""
    trace_id: str
    spans_count: int
    total_duration_ms: float
    spans: List[Dict[str, Any]]


# ==================== API端点 ====================

@router.get("/health", response_model=HealthCheckResponse)
async def health_check(system=Depends(get_monitoring_system)):
    """
    健康检查
    
    返回系统各组件的健康状态
    """
    try:
        result = await system.run_health_check()
        return result
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")


@router.get("/health/{component}")
async def health_check_component(
    component: str,
    system=Depends(get_monitoring_system)
):
    """
    单个组件健康检查
    
    Args:
        component: 组件名称
    """
    try:
        result = await system.health.run_check(component)
        if result:
            return {
                "name": result.name,
                "status": result.status.value,
                "message": result.message,
                "details": result.details,
                "duration_ms": result.duration_ms
            }
        else:
            raise HTTPException(status_code=404, detail=f"未找到组件: {component}")
    except Exception as e:
        logger.error(f"组件健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(system=Depends(get_monitoring_system)):
    """
    获取性能指标
    
    返回系统、API、数据库等性能指标
    """
    try:
        metrics = await system.collect_metrics()
        return metrics
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/prometheus")
async def get_prometheus_metrics(system=Depends(get_monitoring_system)):
    """
    Prometheus格式指标
    
    返回Prometheus格式的性能指标
    """
    try:
        prometheus_format = system.metrics.to_prometheus_format()
        return Response(
            content=prometheus_format,
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"获取Prometheus指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=List[AlertResponse])
async def get_active_alerts(system=Depends(get_monitoring_system)):
    """
    获取活跃告警
    
    返回当前活跃的告警列表
    """
    try:
        alerts = system.alerts.get_active_alerts()
        return alerts
    except Exception as e:
        logger.error(f"获取活跃告警失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/history")
async def get_alert_history(
    since: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    system=Depends(get_monitoring_system)
):
    """
    获取告警历史
    
    Args:
        since: 开始时间（ISO格式）
        status: 告警状态
        limit: 返回数量限制
    """
    try:
        from datetime import datetime as dt
        
        since_dt = None
        if since:
            since_dt = dt.fromisoformat(since)
        
        from .alerts import AlertStatus
        alert_status = None
        if status:
            try:
                alert_status = AlertStatus(status)
            except ValueError:
                pass
        
        history = system.alerts.get_alert_history(
            since=since_dt,
            status=alert_status,
            limit=limit
        )
        
        return {
            "alerts": history,
            "total": len(history)
        }
    except Exception as e:
        logger.error(f"获取告警历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/rules", response_model=List[AlertRuleResponse])
async def get_alert_rules(system=Depends(get_monitoring_system)):
    """
    获取告警规则列表
    """
    try:
        rules = system.alerts.list_rules()
        return rules
    except Exception as e:
        logger.error(f"获取告警规则失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/rules/{rule_name}/silence")
async def silence_alert(
    rule_name: str,
    duration_seconds: int = Query(3600, ge=60, le=86400),
    system=Depends(get_monitoring_system)
):
    """
    静默告警规则
    
    Args:
        rule_name: 规则名称
        duration_seconds: 静默时长（秒）
    """
    try:
        success = system.alerts.silence_alert(rule_name, duration_seconds)
        if success:
            return {
                "message": f"告警规则 {rule_name} 已静默 {duration_seconds} 秒"
            }
        else:
            raise HTTPException(status_code=404, detail=f"未找到告警规则: {rule_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"静默告警失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/rules/{rule_name}/resolve")
async def resolve_alert(
    rule_name: str,
    system=Depends(get_monitoring_system)
):
    """
    手动解决告警
    
    Args:
        rule_name: 规则名称
    """
    try:
        success = await system.alerts.resolve_alert(rule_name)
        if success:
            return {
                "message": f"告警 {rule_name} 已解决"
            }
        else:
            raise HTTPException(status_code=404, detail=f"未找到活跃告警: {rule_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"解决告警失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def query_logs(
    level: Optional[str] = Query(None, regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    keyword: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    system=Depends(get_monitoring_system)
):
    """
    查询日志
    
    Args:
        level: 日志级别
        keyword: 关键词
        since: 开始时间
        limit: 返回数量
    """
    try:
        from datetime import datetime as dt
        
        # 构建查询条件
        filters = {}
        if level:
            filters["level"] = level.upper()
        if keyword:
            filters["keyword"] = keyword
        if since:
            filters["since"] = dt.fromisoformat(since)
        
        # 获取日志（这里简化处理，实际应该从日志收集系统查询）
        logs = system.logger.get_log_records(limit=limit, **filters) if hasattr(system.logger, 'get_log_records') else []
        
        return {
            "logs": logs,
            "total": len(logs),
            "filters": filters
        }
    except Exception as e:
        logger.error(f"查询日志失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tracing/{trace_id}", response_model=TracingResponse)
async def get_trace(
    trace_id: str,
    system=Depends(get_monitoring_system)
):
    """
    获取指定Trace的详情
    
    Args:
        trace_id: Trace ID
    """
    try:
        spans = system.tracing.get_trace(trace_id)
        
        if not spans:
            raise HTTPException(status_code=404, detail=f"未找到Trace: {trace_id}")
        
        total_duration = max(span.duration_ms for span in spans) if spans else 0
        
        return {
            "trace_id": trace_id,
            "spans_count": len(spans),
            "total_duration_ms": total_duration,
            "spans": [span.to_dict() for span in spans]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取Trace失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tracing")
async def list_traces(
    limit: int = Query(100, ge=1, le=1000),
    system=Depends(get_monitoring_system)
):
    """
    列出最近的Traces
    """
    try:
        all_spans = system.tracing.export_all()[-limit:]
        
        # 按trace_id分组
        traces = {}
        for span in all_spans:
            trace_id = span.get('context', {}).get('trace_id')
            if trace_id:
                if trace_id not in traces:
                    traces[trace_id] = []
                traces[trace_id].append(span)
        
        result = []
        for trace_id, spans in traces.items():
            total_duration = max(
                span.get('duration_ms', 0) 
                for span in spans
            )
            result.append({
                "trace_id": trace_id,
                "spans_count": len(spans),
                "total_duration_ms": total_duration,
                "start_time": spans[0].get('start_time', ''),
                "name": spans[0].get('name', '')
            })
        
        return {
            "traces": result,
            "total": len(result)
        }
    except Exception as e:
        logger.error(f"列出Traces失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_monitoring_status(system=Depends(get_monitoring_system)):
    """
    获取监控系统整体状态
    """
    try:
        status = system.get_status()
        return status
    except Exception as e:
        logger.error(f"获取监控状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uptime")
async def get_uptime(system=Depends(get_monitoring_system)):
    """
    获取系统运行时间
    """
    try:
        uptime = system.get_uptime()
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": format_duration(uptime)
        }
    except Exception as e:
        logger.error(f"获取运行时间失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 辅助函数 ====================

def format_duration(seconds: float) -> str:
    """
    格式化时长
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.0f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.0f}h"
    else:
        days = seconds / 86400
        return f"{days:.0f}d"


from fastapi.responses import Response

# 重新定义prometheus端点的Response
@router.get("/metrics/prometheus", response_class=Response)
async def get_prometheus_metrics(system=Depends(get_monitoring_system)):
    """
    Prometheus格式指标
    """
    try:
        prometheus_format = system.metrics.to_prometheus_format()
        return Response(
            content=prometheus_format,
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"获取Prometheus指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
