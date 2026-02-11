"""
安全API端点 (Security API Endpoints)
提供RESTful API接口
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, '/Users/yubao/.openclaw/workspace/backend')

from security.masking import DataMaskingEngine, get_masking_engine, MaskingStrategy
from security.access import AccessControlEngine, get_access_control, ResourceType, Permission
from security.audit import AuditLogger, get_audit_logger, AuditAction, AuditSeverity
from security.compliance import ComplianceReportGenerator, get_compliance_generator, ComplianceStandard

router = APIRouter(prefix="/security", tags=["security"])

# 全局实例
_masking_engine = get_masking_engine()
_access_control = get_access_control()
_audit_logger = get_audit_logger()
_compliance_generator = get_compliance_generator()


# ============ 数据脱敏 API ============

class MaskRequest(BaseModel):
    """数据脱敏请求"""
    data: Dict[str, Any]
    rules: Optional[Dict[str, Any]] = None
    auto_detect: bool = True


class MaskResponse(BaseModel):
    """数据脱敏响应"""
    masked_data: Dict[str, Any]
    applied_rules: List[str]
    timestamp: str


@router.post("/mask", response_model=MaskResponse)
async def mask_data(request: MaskRequest):
    """
    数据脱敏API
    
    对传入的数据进行脱敏处理，支持自动检测敏感字段或使用预定义规则。
    """
    try:
        # 应用自动检测脱敏
        if request.auto_detect:
            masked = _masking_engine.auto_detect_and_mask(request.data)
            applied = ["auto_detect"]
        else:
            masked = _masking_engine.mask_document(request.data)
            applied = list(request.rules.keys()) if request.rules else []
        
        # 记录审计日志
        _audit_logger.log(
            action=AuditAction.DATA_READ,
            user_id="api_user",
            resource_type="data",
            details={"operation": "mask", "fields_masked": applied},
            severity=AuditSeverity.INFO
        )
        
        return MaskResponse(
            masked_data=masked,
            applied_rules=applied,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"脱敏失败: {str(e)}")


@router.post("/mask/batch")
async def mask_batch(request: List[MaskRequest]):
    """批量数据脱敏"""
    results = []
    for req in request:
        if req.auto_detect:
            masked = _masking_engine.auto_detect_and_mask(req.data)
        else:
            masked = _masking_engine.mask_document(req.data)
        results.append(masked)
    return {"results": results}


# ============ 访问控制 API ============

class AccessCheckRequest(BaseModel):
    """访问权限检查请求"""
    user_id: str
    resource_type: str
    resource_id: Optional[str] = None
    permission: str
    context: Optional[Dict[str, Any]] = None


class AccessCheckResponse(BaseModel):
    """访问权限检查响应"""
    allowed: bool
    reason: str
    user_id: str
    resource_type: str
    permission: str
    timestamp: str


@router.post("/access/check", response_model=AccessCheckResponse)
async def check_access(request: AccessCheckRequest):
    """
    访问权限检查API
    
    检查用户是否有权限执行指定操作。
    """
    try:
        resource_type = ResourceType(request.resource_type)
        permission = Permission(request.permission)
        
        allowed, reason = _access_control.check_permission(
            user_id=request.user_id,
            resource_type=resource_type,
            permission=permission,
            resource_id=request.resource_id,
            context=request.context
        )
        
        # 记录审计
        audit_result = _access_control.audit_access(
            request,
            allowed,
            reason
        )
        
        _audit_logger.log(
            action=AuditAction.API_ACCESS,
            user_id=request.user_id,
            resource_type=request.resource_type,
            details={"permission": request.permission, "result": allowed},
            severity=AuditSeverity.INFO if allowed else AuditSeverity.WARNING
        )
        
        return AccessCheckResponse(
            allowed=allowed,
            reason=reason,
            user_id=request.user_id,
            resource_type=request.resource_type,
            permission=request.permission,
            timestamp=datetime.utcnow().isoformat()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"无效参数: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检查失败: {str(e)}")


@router.get("/access/permissions/{user_id}")
async def get_user_permissions(user_id: str):
    """获取用户权限"""
    permissions = _access_control.get_user_permissions(user_id)
    return {"user_id": user_id, "permissions": permissions}


@router.post("/access/roles/assign")
async def assign_role(user_id: str, role_name: str):
    """分配角色"""
    success = _access_control.assign_role(user_id, role_name)
    if success:
        _audit_logger.log(
            action=AuditAction.ROLE_ASSIGN,
            user_id=user_id,
            resource_type="user",
            details={"role": role_name},
            severity=AuditSeverity.INFO
        )
        return {"success": True, "message": f"角色 {role_name} 已分配给 {user_id}"}
    raise HTTPException(status_code=400, detail="角色分配失败")


# ============ 审计日志 API ============

class AuditQueryParams(BaseModel):
    """审计日志查询参数"""
    user_id: Optional[str] = None
    action: Optional[str] = None
    resource_type: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = 100
    offset: int = 0


@router.get("/audit/logs")
async def get_audit_logs(
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    获取审计日志API
    
    支持多维度查询和分页。
    """
    from security.audit import AuditAction as AA
    
    action_enum = None
    if action:
        try:
            action_enum = AA(action)
        except ValueError:
            pass
    
    logs = _audit_logger.query(
        user_id=user_id,
        action=action_enum,
        resource_type=resource_type,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset
    )
    
    return {
        "logs": [log.to_dict() for log in logs],
        "total": len(logs),
        "limit": limit,
        "offset": offset
    }


@router.get("/audit/statistics")
async def get_audit_statistics(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """获取审计统计"""
    stats = _audit_logger.get_statistics(start_time, end_time)
    return stats


@router.get("/audit/alerts")
async def get_security_alerts(hours: int = 24):
    """获取安全警报"""
    alerts = _audit_logger.get_security_alerts(hours)
    return {
        "alerts": [alert.to_dict() for alert in alerts],
        "count": len(alerts)
    }


@router.get("/audit/export")
async def export_audit_logs(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    format: str = "json"
):
    """导出审计日志"""
    if format not in ["json", "csv"]:
        raise HTTPException(status_code=400, detail="只支持json或csv格式")
    
    exported = _audit_logger.export(format, start_time, end_time)
    return {"data": exported, "format": format}


# ============ 合规报告 API ============

class ComplianceReportRequest(BaseModel):
    """合规报告生成请求"""
    standard: str
    scope: str
    period_start: datetime
    period_end: datetime
    audit_data: Optional[Dict[str, Any]] = None
    approved_by: Optional[str] = None


@router.get("/compliance/report")
async def generate_compliance_report(request: ComplianceReportRequest):
    """
    生成合规报告API
    
    支持SOX、GDPR、ISO 27001等合规标准。
    """
    try:
        standard = ComplianceStandard(request.standard)
        
        report = _compliance_generator.generate_report(
            standard=standard,
            scope=request.scope,
            period_start=request.period_start,
            period_end=request.period_end,
            audit_data=request.audit_data or {},
            approved_by=request.approved_by or ""
        )
        
        _audit_logger.log(
            action=AuditAction.DATA_EXPORT,
            user_id="api_user",
            resource_type="report",
            details={"standard": request.standard, "report_id": report.id},
            severity=AuditSeverity.INFO
        )
        
        return {
            "report": {
                "id": report.id,
                "standard": report.standard.value,
                "generated_at": report.generated_at.isoformat(),
                "period": f"{report.period_start.date()} - {report.period_end.date()}",
                "scope": report.scope,
                "summary": report.summary,
                "findings": [
                    {
                        "requirement_id": f.status.value,
                        "status": f.status.value,
                        "remediation": f.remediation
                    }
                    for f in report.findings
                ],
                "recommendations": report.recommendations,
                "status": report.status
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"无效参数: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"报告生成失败: {str(e)}")


@router.get("/compliance/history")
async def get_compliance_history(standard: Optional[str] = None):
    """获取合规报告历史"""
    if standard:
        try:
            std_enum = ComplianceStandard(standard)
            history = _compliance_generator.get_report_history(std_enum)
        except ValueError:
            history = _compliance_generator.get_report_history()
    else:
        history = _compliance_generator.get_report_history()
    
    return {"reports": history}


@router.get("/compliance/standards")
async def get_supported_standards():
    """获取支持的合规标准"""
    return {
        "standards": [
            {"id": s.value, "name": s.name}
            for s in ComplianceStandard
        ]
    }


# ============ 健康检查 ============

@router.get("/health")
async def security_health_check():
    """安全模块健康检查"""
    return {
        "status": "healthy",
        "module": "security",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "masking": "active",
            "access_control": "active",
            "audit": "active",
            "compliance": "active"
        }
    }
