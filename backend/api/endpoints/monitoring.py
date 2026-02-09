"""
监控告警API端点 v2.0 Phase 2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

from monitoring.monitor import alert_manager, AlertSeverity, AlertStatus
from pipeline.data.pipeline import data_pipeline
from api.endpoints.auth import get_current_user

router = APIRouter()

class AlertRuleCreate(BaseModel):
    """创建告警规则"""
    rule_id: str
    name: str
    condition: str
    threshold: float
    duration_seconds: int = 300
    severity: str
    channels: List[str]

@router.get("/alerts/rules")
async def list_alert_rules():
    """
    获取告警规则列表
    
    v2.0 Phase 2: 监控告警
    """
    return {"rules": alert_manager.get_rules()}

@router.post("/alerts/rules")
async def create_alert_rule(rule: AlertRuleCreate, current_user = Depends(get_current_user)):
    """
    创建告警规则
    
    v2.0 Phase 2: 监控告警
    """
    from monitoring.monitor import AlertRule
    
    try:
        severity = AlertSeverity(rule.severity)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid severity: {rule.severity}")
    
    new_rule = AlertRule(
        rule_id=rule.rule_id,
        name=rule.name,
        condition=rule.condition,
        threshold=rule.threshold,
        duration_seconds=rule.duration_seconds,
        severity=severity,
        channels=rule.channels
    )
    
    alert_manager.add_custom_rule(new_rule)
    
    return {"message": "Rule created", "rule_id": rule.rule_id}

@router.get("/alerts/active")
async def get_active_alerts(severity: Optional[str] = None):
    """
    获取活跃告警
    
    v2.0 Phase 2: 监控告警
    """
    try:
        severity_enum = AlertSeverity(severity) if severity else None
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
    
    alerts = alert_manager.get_active_alerts(severity_enum)
    
    return {
        "total": len(alerts),
        "alerts": [
            {
                "alert_id": a.alert_id,
                "rule_name": a.rule_name,
                "severity": a.severity.value,
                "value": a.value,
                "threshold": a.threshold,
                "message": a.message,
                "fired_at": a.fired_at.isoformat()
            }
            for a in alerts
        ]
    }

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, current_user = Depends(get_current_user)):
    """
    解决告警
    
    v2.0 Phase 2: 监控告警
    """
    try:
        await alert_manager.resolve_alert(alert_id)
        return {"message": "Alert resolved"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/alerts/{alert_id}/silence")
async def silence_alert(alert_id: str, duration_hours: int = 24):
    """
    静默告警
    
    v2.0 Phase 2: 监控告警
    """
    alert_manager.silence_alert(alert_id, duration_hours)
    return {"message": "Alert silenced"}

@router.get("/alerts/history")
async def get_alert_history(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(default=100, le=1000)
):
    """
    获取告警历史
    
    v2.0 Phase 2: 监控告警
    """
    history = alert_manager.get_alert_history(start_time, end_time, limit)
    
    return {
        "total": len(history),
        "alerts": [
            {
                "alert_id": a.alert_id,
                "rule_name": a.rule_name,
                "severity": a.severity.value,
                "status": a.status.value,
                "fired_at": a.fired_at.isoformat(),
                "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None
            }
            for a in history
        ]
    }

@router.post("/check")
async def check_condition(
    rule_id: str,
    value: float,
    labels: Optional[dict] = None
):
    """
    检查条件
    
    v2.0 Phase 2: 监控告警
    """
    alert = await alert_manager.check_condition(rule_id, value, labels or {})
    
    if alert:
        return {
            "triggered": True,
            "alert": {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "message": alert.message
            }
        }
    
    return {"triggered": False}

# 数据流水线API
class CreateDatasetRequest(BaseModel):
    name: str
    description: Optional[str] = None

class RunPipelineRequest(BaseModel):
    dataset_id: str
    steps: List[str]
    config: Optional[dict] = None

@router.post("/data/datasets")
async def create_dataset(request: CreateDatasetRequest, current_user = Depends(get_current_user)):
    """
    创建数据集
    
    v2.0 Phase 2: 数据流水线
    """
    dataset = await data_pipeline.create_dataset(
        name=request.name,
        description=request.description or "",
        user_id=str(current_user.id)
    )
    
    return dataset

@router.post("/data/pipelines/run")
async def run_pipeline(request: RunPipelineRequest, current_user = Depends(get_current_user)):
    """
    执行数据处理流水线
    
    v2.0 Phase 2: 数据流水线
    """
    from pipeline.data.pipeline import DataStep
    
    steps = [DataStep(s) for s in request.steps]
    
    result = await data_pipeline.run_pipeline(
        dataset_id=request.dataset_id,
        steps=steps,
        config=request.config or {}
    )
    
    return result

@router.get("/data/datasets/{dataset_id}/versions")
async def get_dataset_versions(dataset_id: str):
    """
    获取数据集版本
    
    v2.0 Phase 2: 数据流水线
    """
    versions = await data_pipeline.get_versions(dataset_id)
    return {"versions": versions}
