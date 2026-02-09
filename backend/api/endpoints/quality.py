"""
数据质量API端点 v2.1
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

from backend.quality.engine import quality_engine
from backend.core.auth import get_current_user

router = APIRouter()

class CreateRuleModel(BaseModel):
    name: str
    description: str
    rule_type: str  # schema, range, uniqueness, completeness, custom
    column: Optional[str] = None
    condition: str = ""
    severity: str = "error"

class ValidateDatasetModel(BaseModel):
    dataset_id: str
    data: List[Dict[str, Any]]
    rule_ids: Optional[List[str]] = None

@router.post("/rules")
async def create_quality_rule(request: CreateRuleModel):
    """
    创建质量规则
    
    v2.1: 数据质量
    """
    try:
        rule = quality_engine.create_rule(
            name=request.name,
            description=request.description,
            rule_type=request.rule_type,
            column=request.column,
            condition=request.condition,
            severity=request.severity
        )
        
        return {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "description": rule.description,
            "rule_type": rule.rule_type,
            "severity": rule.severity,
            "created_at": rule.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/rules")
async def list_quality_rules(
    enabled: Optional[bool] = None
):
    """
    列出质量规则
    
    v2.1: 数据质量
    """
    rules = quality_engine.get_rules(enabled=enabled)
    
    return {
        "total": len(rules),
        "rules": [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "description": r.description,
                "rule_type": r.rule_type,
                "column": r.column,
                "condition": r.condition,
                "severity": r.severity,
                "enabled": r.enabled,
                "created_at": r.created_at.isoformat()
            }
            for r in rules
        ]
    }

@router.post("/rules/{rule_id}/enable")
async def enable_rule(rule_id: str):
    """
    启用规则
    
    v2.1: 数据质量
    """
    rule = quality_engine.rules.get(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    rule.enabled = True
    return {"message": "Rule enabled"}

@router.post("/rules/{rule_id}/disable")
async def disable_rule(rule_id: str):
    """
    禁用规则
    
    v2.1: 数据质量
    """
    rule = quality_engine.rules.get(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    rule.enabled = False
    return {"message": "Rule disabled"}

@router.post("/validate")
async def validate_dataset(request: ValidateDatasetModel):
    """
    验证数据集
    
    v2.1: 数据质量
    """
    try:
        check = await quality_engine.validate_dataset(
            dataset_id=request.dataset_id,
            data=request.data,
            rule_ids=request.rule_ids
        )
        
        return {
            "check_id": check.check_id,
            "dataset_id": check.dataset_id,
            "status": check.status,
            "total_count": check.total_count,
            "passed_count": check.passed_count,
            "failed_count": check.failed_count,
            "pass_rate": f"{(check.passed_count / check.total_count * 100) if check.total_count > 0 else 100:.1f}%",
            "executed_at": check.executed_at.isoformat() if check.executed_at else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/checks/{check_id}")
async def get_check_result(check_id: str):
    """
    获取检查结果
    
    v2.1: 数据质量
    """
    check = quality_engine.checks.get(check_id)
    if not check:
        raise HTTPException(status_code=404, detail="Check not found")
    
    return {
        "check_id": check.check_id,
        "dataset_id": check.dataset_id,
        "status": check.status,
        "total_count": check.total_count,
        "passed_count": check.passed_count,
        "failed_count": check.failed_count,
        "results_summary": {
            "total": len(check.results),
            "passed": sum(1 for r in check.results if r["passed"]),
            "failed": sum(1 for r in check.results if not r["passed"])
        },
        "executed_at": check.executed_at.isoformat() if check.executed_at else None
    }

@router.post("/checks/{check_id}/report")
async def generate_report(check_id: str):
    """
    生成质量报告
    
    v2.1: 数据质量
    """
    try:
        report = quality_engine.generate_report(
            dataset_id="",
            check_id=check_id
        )
        return report
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/reports")
async def list_reports(dataset_id: Optional[str] = None):
    """
    列出质量报告
    
    v2.1: 数据质量
    """
    reports = quality_engine.get_reports(dataset_id)
    
    return {
        "total": len(reports),
        "reports": [
            {
                "report_id": r.report_id,
                "dataset_id": r.dataset_id,
                "check_id": r.check_id,
                "score": r.score,
                "status": r.status,
                "summary": r.summary,
                "recommendations": r.recommendations,
                "created_at": r.created_at.isoformat()
            }
            for r in reports
        ]
    }

@router.get("/reports/{report_id}")
async def get_report(report_id: str):
    """
    获取质量报告详情
    
    v2.1: 数据质量
    """
    report = quality_engine.reports.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return {
        "report_id": report.report_id,
        "dataset_id": report.dataset_id,
        "check_id": report.check_id,
        "score": report.score,
        "status": report.status,
        "summary": report.summary,
        "recommendations": report.recommendations,
        "created_at": report.created_at.isoformat()
    }

@router.get("/summary")
async def get_quality_summary():
    """
    获取质量总览
    
    v2.1: 数据质量
    """
    rules = list(quality_engine.rules.values())
    enabled_rules = [r for r in rules if r.enabled]
    checks = list(quality_engine.checks.values())
    recent_checks = sorted(checks, key=lambda c: c.created_at, reverse=True)[:10]
    
    passed_checks = sum(1 for c in checks if c.status == "passed")
    failed_checks = sum(1 for c in checks if c.status == "failed")
    
    return {
        "total_rules": len(rules),
        "enabled_rules": len(enabled_rules),
        "total_checks": len(checks),
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "pass_rate": f"{(passed_checks / len(checks) * 100) if checks else 100:.1f}%",
        "recent_checks": [
            {
                "check_id": c.check_id,
                "status": c.status,
                "pass_rate": f"{(c.passed_count / c.total_count * 100) if c.total_count > 0 else 100:.1f}%",
                "executed_at": c.executed_at.isoformat() if c.executed_at else None
            }
            for c in recent_checks
        ]
    }
