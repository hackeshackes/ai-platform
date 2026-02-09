"""
Cost Intelligence API端点 v2.4
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel

# 直接导入模块
import importlib.util
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'cost/intelligence.py')

spec = importlib.util.spec_from_file_location("cost_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    cost_intelligence = module.cost_intelligence
    CostType = module.CostType
    Provider = module.Provider
except Exception as e:
    print(f"Failed to import cost module: {e}")
    cost_intelligence = None
    CostType = None
    Provider = None

router = APIRouter()

class TrackTokenModel(BaseModel):
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float = 0.0
    project_id: Optional[str] = None

class CreateBudgetModel(BaseModel):
    name: str
    total_limit: float
    cost_type: str = "api_call"
    period: str = "monthly"
    alert_threshold: float = 0.8
    project_id: Optional[str] = None

@router.post("/track/token")
async def track_token_usage(request: TrackTokenModel):
    """
    记录Token使用
    
    v2.4: Cost Intelligence
    """
    try:
        provider = Provider(request.provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {request.provider}")
    
    usage = cost_intelligence.track_token_usage(
        provider=provider,
        model=request.model,
        prompt_tokens=request.prompt_tokens,
        completion_tokens=request.completion_tokens,
        latency_ms=request.latency_ms,
        project_id=request.project_id
    )
    
    return {
        "usage_id": usage.usage_id,
        "total_tokens": usage.total_tokens,
        "cost": usage.cost,
        "message": "Token usage tracked"
    }

@router.post("/track")
async def record_cost(
    cost_type: str,
    provider: str,
    amount: float,
    unit: str,
    metadata: Optional[Dict] = None,
    project_id: Optional[str] = None
):
    """
    记录成本
    
    v2.4: Cost Intelligence
    """
    try:
        ctype = CostType(cost_type)
        prov = Provider(provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    entry = cost_intelligence.record_cost(
        cost_type=ctype,
        provider=prov,
        amount=amount,
        unit=unit,
        metadata=metadata,
        project_id=project_id
    )
    
    return {
        "entry_id": entry.entry_id,
        "amount": entry.amount,
        "message": "Cost recorded"
    }

@router.post("/budgets")
async def create_budget(request: CreateBudgetModel):
    """
    创建预算
    
    v2.4: Cost Intelligence
    """
    try:
        ctype = CostType(request.cost_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid cost type: {request.cost_type}")
    
    budget = cost_intelligence.create_budget(
        name=request.name,
        total_limit=request.total_limit,
        cost_type=ctype,
        period=request.period,
        alert_threshold=request.alert_threshold,
        project_id=request.project_id
    )
    
    return {
        "budget_id": budget.budget_id,
        "name": budget.name,
        "limit": budget.total_limit,
        "message": "Budget created"
    }

@router.get("/budgets")
async def list_budgets(
    project_id: Optional[str] = None
):
    """
    列出预算
    
    v2.4: Cost Intelligence
    """
    budgets = cost_intelligence.list_budgets(project_id=project_id)
    
    return {
        "total": len(budgets),
        "budgets": [
            {
                "budget_id": b.budget_id,
                "name": b.name,
                "limit": b.total_limit,
                "used": b.used_amount,
                "remaining": b.total_limit - b.used_amount,
                "usage_percent": round(b.used_amount / b.total_limit * 100, 2) if b.total_limit > 0 else 0,
                "type": b.cost_type.value,
                "period": b.period,
                "enabled": b.enabled,
                "created_at": b.created_at.isoformat()
            }
            for b in budgets
        ]
    }

@router.get("/budgets/{budget_id}")
async def get_budget(budget_id: str):
    """
    获取预算详情
    
    v2.4: Cost Intelligence
    """
    budget = cost_intelligence.get_budget(budget_id)
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    
    return {
        "budget_id": budget.budget_id,
        "name": budget.name,
        "limit": budget.total_limit,
        "used": budget.used_amount,
        "remaining": budget.total_limit - budget.used_amount,
        "usage_percent": round(budget.used_amount / budget.total_limit * 100, 2) if budget.total_limit > 0 else 0,
        "type": budget.cost_type.value,
        "period": budget.period,
        "alert_threshold": budget.alert_threshold,
        "enabled": budget.enabled,
        "created_at": budget.created_at.isoformat()
    }

@router.get("/summary")
async def get_cost_summary(
    days: int = 30,
    project_id: Optional[str] = None,
    provider: Optional[str] = None
):
    """
    获取成本汇总
    
    v2.4: Cost Intelligence
    """
    from datetime import datetime, timedelta
    prov = Provider(provider) if provider else None
    
    summary = cost_intelligence.get_cost_summary(
        start_date=datetime.utcnow() - timedelta(days=days),
        end_date=datetime.utcnow(),
        project_id=project_id,
        provider=prov
    )
    
    return summary

@router.get("/tokens")
async def get_token_summary(days: int = 30):
    """
    获取Token使用汇总
    
    v2.4: Cost Intelligence
    """
    summary = cost_intelligence.get_token_summary(days)
    return summary

@router.get("/forecast")
async def get_cost_forecast(days: int = 30):
    """
    获取成本预测
    
    v2.4: Cost Intelligence
    """
    forecast = cost_intelligence.forecast_cost(days)
    
    return {
        "current_spend": forecast.current_spend,
        "predicted_daily": forecast.predicted_daily,
        "predicted_weekly": forecast.predicted_weekly,
        "predicted_monthly": forecast.predicted_monthly,
        "trend": forecast.trend,
        "confidence": forecast.confidence,
        "based_on_days": forecast.based_on_days,
        "generated_at": forecast.created_at.isoformat()
    }

@router.get("/suggestions")
async def get_optimization_suggestions():
    """
    获取优化建议
    
    v2.4: Cost Intelligence
    """
    suggestions = cost_intelligence.get_optimization_suggestions()
    
    return {
        "total": len(suggestions),
        "suggestions": suggestions
    }

@router.post("/report")
async def export_report(
    start_date: datetime,
    end_date: datetime,
    format: str = "json"
):
    """
    导出报告
    
    v2.4: Cost Intelligence
    """
    if end_date < start_date:
        raise HTTPException(status_code=400, detail="End date must be after start date")
    
    report = cost_intelligence.export_report(start_date, end_date, format)
    return report

@router.get("/pricing")
async def get_pricing():
    """
    获取定价配置
    
    v2.4: Cost Intelligence
    """
    return {
        "providers": {
            prov.value: models
            for prov, models in cost_intelligence.pricing.items()
        }
    }

@router.get("/health")
async def cost_health():
    """
    Cost Intelligence健康检查
    
    v2.4: Cost Intelligence
    """
    summary = cost_intelligence.get_token_summary(7)
    
    return {
        "status": "healthy",
        "total_cost_7d": summary["total_cost"],
        "total_tokens_7d": summary["total_tokens"]
    }
