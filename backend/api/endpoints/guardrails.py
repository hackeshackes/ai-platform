"""
LLM Guardrails API端点 v2.4
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any

# 直接导入模块
import importlib.util
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'guardrails/engine.py')

spec = importlib.util.spec_from_file_location("guardrails_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    guardrails_engine = module.guardrails_engine
    GuardrailType = module.GuardrailType
    Severity = module.Severity
    Action = module.Action
except Exception as e:
    print(f"Failed to import guardrails module: {e}")
    guardrails_engine = None
    GuardrailType = None
    Severity = None
    Action = None

router = APIRouter()

@router.get("/rules")
async def list_rules(
    guardrail_type: Optional[str] = None,
    enabled: Optional[bool] = None
):
    """
    列出护栏规则
    
    v2.4: LLM Guardrails
    """
    gtype = GuardrailType(guardrail_type) if guardrail_type else None
    
    rules = guardrails_engine.list_rules(
        guardrail_type=gtype,
        enabled=enabled
    )
    
    return {
        "total": len(rules),
        "rules": [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "description": r.description,
                "type": r.guardrail_type.value,
                "severity": r.severity.value,
                "action": r.action.value,
                "enabled": r.enabled,
                "keywords_count": len(r.keywords),
                "created_at": r.created_at.isoformat()
            }
            for r in rules
        ]
    }

@router.post("/rules")
async def create_rule(
    name: str,
    description: str,
    guardrail_type: str,
    severity: str,
    keywords: Optional[List[str]] = None,
    pattern: Optional[str] = None,
    action: str = "block"
):
    """
    创建护栏规则
    
    v2.4: LLM Guardrails
    """
    try:
        gtype = GuardrailType(guardrail_type)
        sev = Severity(severity)
        act = Action(action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    rule = guardrails_engine.create_rule(
        name=name,
        description=description,
        guardrail_type=gtype,
        severity=sev,
        keywords=keywords,
        pattern=pattern,
        action=act
    )
    
    return {
        "rule_id": rule.rule_id,
        "name": rule.name,
        "message": "Rule created"
    }

@router.get("/rules/{rule_id}")
async def get_rule(rule_id: str):
    """
    获取规则详情
    
    v2.4: LLM Guardrails
    """
    rule = guardrails_engine.get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    return {
        "rule_id": rule.rule_id,
        "name": rule.name,
        "description": rule.description,
        "type": rule.guardrail_type.value,
        "severity": rule.severity.value,
        "keywords": rule.keywords,
        "pattern": rule.pattern,
        "action": rule.action.value,
        "config": rule.config,
        "enabled": rule.enabled,
        "created_at": rule.created_at.isoformat()
    }

@router.put("/rules/{rule_id}")
async def update_rule(
    rule_id: str,
    enabled: Optional[bool] = None,
    action: Optional[str] = None,
    config: Optional[Dict] = None
):
    """
    更新规则
    
    v2.4: LLM Guardrails
    """
    act = Action(action) if action else None
    
    result = guardrails_engine.update_rule(
        rule_id=rule_id,
        enabled=enabled,
        action=act,
        config=config
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    return {"message": "Rule updated"}

@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """
    删除规则
    
    v2.4: LLM Guardrails
    """
    result = guardrails_engine.delete_rule(rule_id)
    if not result:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    return {"message": "Rule deleted"}

@router.post("/check/input")
async def check_input(text: str):
    """
    检查输入
    
    v2.4: LLM Guardrails
    """
    violations = guardrails_engine.check_input(text)
    
    passed = all(v.passed for v in violations)
    
    return {
        "passed": passed,
        "violations_count": len([v for v in violations if not v.passed]),
        "results": [
            {
                "rule_id": v.rule_id,
                "rule_name": v.rule_name,
                "passed": v.passed,
                "action": v.action.value,
                "message": v.message
            }
            for v in violations
        ]
    }

@router.post("/check/output")
async def check_output(text: str):
    """
    检查输出
    
    v2.4: LLM Guardrails
    """
    violations = guardrails_engine.check_output(text)
    
    passed = all(v.passed for v in violations)
    
    return {
        "passed": passed,
        "violations_count": len([v for v in violations if not v.passed]),
        "results": [
            {
                "rule_id": v.rule_id,
                "rule_name": v.rule_name,
                "passed": v.passed,
                "action": v.action.value,
                "message": v.message
            }
            for v in violations
        ]
    }

@router.post("/sanitize")
async def sanitize(text: str, guardrail_type: str = "output"):
    """
    净化文本
    
    v2.4: LLM Guardrails
    """
    gtype = GuardrailType(guardrail_type)
    sanitized = guardrails_engine.sanitize(text, gtype)
    
    return {
        "original": text,
        "sanitized": sanitized
    }

@router.get("/violations")
async def get_violations(limit: int = 100):
    """
    获取违规记录
    
    v2.4: LLM Guardrails
    """
    violations = guardrails_engine.get_violations(limit)
    
    return {
        "total": len(violations),
        "violations": [
            {
                "result_id": v.result_id,
                "rule_name": v.rule_name,
                "type": v.guardrail_type.value,
                "passed": v.passed,
                "action": v.action.value,
                "message": v.message,
                "checked_at": v.checked_at.isoformat()
            }
            for v in violations
        ]
    }

@router.get("/stats")
async def get_stats():
    """
    获取统计数据
    
    v2.4: LLM Guardrails
    """
    stats = guardrails_engine.get_stats()
    return stats

@router.get("/health")
async def guardrails_health():
    """
    Guardrails健康检查
    
    v2.4: LLM Guardrails
    """
    return {
        "status": "healthy",
        "enabled_rules": len([r for r in guardrails_engine.rules.values() if r.enabled])
    }
