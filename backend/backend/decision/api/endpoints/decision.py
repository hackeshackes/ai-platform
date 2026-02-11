"""
Decision API Endpoints - 决策引擎API端点
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_engine import (
    DecisionEngine, DecisionRequest, DecisionContext, DecisionResult,
    DecisionType, DecisionPriority, ConfidenceLevel
)
from risk_assessor import RiskAssessor, RiskAssessment, RiskLevel
from predictor import Predictor, PredictionResult


router = APIRouter(prefix="/decision", tags=["Decision Engine"])

# 全局引擎实例
_engine: Optional[DecisionEngine] = None
_risk_assessor: Optional[RiskAssessor] = None
_predictor: Optional[Predictor] = None


def get_engine() -> DecisionEngine:
    """获取决策引擎实例"""
    global _engine
    if _engine is None:
        _engine = DecisionEngine()
    return _engine


def get_risk_assessor() -> RiskAssessor:
    """获取风险评估器实例"""
    global _risk_assessor
    if _risk_assessor is None:
        _risk_assessor = RiskAssessor()
    return _risk_assessor


def get_predictor() -> Predictor:
    """获取预测器实例"""
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


# ============ Pydantic Models ============

from pydantic import BaseModel, Field


class BusinessDataModel(BaseModel):
    """业务数据模型"""
    revenue_growth: float = Field(0.05, description="收入增长率")
    market_opportunity: float = Field(0.5, description="市场机会评分")
    competitive_advantage: float = Field(0.5, description="竞争优势评分")
    cash_flow_status: str = Field("stable", description="现金流状态")
    debt_ratio: float = Field(0.3, description="负债率")
    profit_margin: float = Field(0.1, description="利润率")
    market_volatility: float = Field(0.3, description="市场波动性")
    competition_intensity: float = Field(0.5, description="竞争强度")
    time_series: Optional[List[dict]] = Field(None, description="时间序列数据")


class DecisionContextModel(BaseModel):
    """决策上下文模型"""
    business_data: BusinessDataModel
    constraints: Optional[dict] = Field(default_factory=dict)
    objectives: Optional[List[str]] = Field(default_factory=list)
    time_horizon: int = Field(30, description="时间范围(天)")
    risk_tolerance: float = Field(0.5, description="风险容忍度")


class DecisionRequestModel(BaseModel):
    """决策请求模型"""
    context: DecisionContextModel
    options: Optional[List[str]] = Field(default_factory=list)
    enable_auto_execute: bool = Field(False, description="是否自动执行")


class RiskAssessmentModel(BaseModel):
    """风险评估模型"""
    overall_score: float
    level: str
    category_scores: dict
    trend: str
    recommendations: List[str]
    risk_tolerance_status: str
    assessed_at: datetime


class PredictionModel(BaseModel):
    """预测模型"""
    forecast_horizon: int
    trend_direction: str
    trend_strength: float
    has_seasonality: bool
    seasonality_type: Optional[str]
    anomalies_count: int
    model_accuracy: float
    confidence_interval: tuple


class AutoDecisionRequest(BaseModel):
    """自动决策请求"""
    decision_id: str
    confirmed: bool
    execution_params: Optional[dict] = None


class DecisionFilter(BaseModel):
    """决策过滤条件"""
    decision_type: Optional[str] = None
    priority: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: Optional[str] = None


# ============ API Endpoints ============

@router.post("/analyze", response_model=dict)
async def analyze_decision(request: DecisionRequestModel):
    """
    决策分析
    
    输入业务数据和上下文，返回完整的决策分析结果
    """
    engine = get_engine()
    
    # 构建决策上下文
    context = DecisionContext(
        business_data=request.context.business_data.dict(),
        constraints=request.context.constraints,
        objectives=request.context.objectives,
        time_horizon=request.context.time_horizon,
        risk_tolerance=request.context.risk_tolerance
    )
    
    # 构建决策请求
    decision_request = DecisionRequest(
        context=context,
        options=request.options,
        enable_auto_execute=request.enable_auto_execute
    )
    
    # 执行决策分析
    result = await engine.analyze(decision_request)
    
    return {
        "success": True,
        "data": {
            "decision": {
                "type": result.decision.type.value,
                "priority": result.decision.priority.name,
                "reasoning": result.decision.reasoning,
                "details": result.decision.details,
                "conditions": result.decision.conditions
            },
            "confidence": result.confidence,
            "confidence_level": result.confidence_level.value,
            "risk_score": result.risk_score,
            "expected_reward": result.expected_reward,
            "explanation": result.explanation,
            "recommendations": result.recommendations,
            "alternatives": [
                {
                    "id": alt.id,
                    "type": alt.type.value,
                    "description": alt.description,
                    "expected_outcome": alt.expected_outcome,
                    "risk_factors": alt.risk_factors,
                    "confidence": alt.confidence
                }
                for alt in result.alternatives
            ],
            "timestamp": result.timestamp.isoformat()
        }
    }


@router.post("/recommend")
async def get_recommendation(context: DecisionContextModel):
    """
    获取决策建议
    
    基于当前业务上下文生成决策建议
    """
    engine = get_engine()
    risk_assessor = get_risk_assessor()
    
    # 构建上下文
    decision_context = DecisionContext(
        business_data=context.business_data.dict(),
        constraints=context.constraints,
        objectives=context.objectives,
        time_horizon=context.time_horizon,
        risk_tolerance=context.risk_tolerance
    )
    
    # 执行决策分析
    request = DecisionRequest(context=decision_context)
    result = await engine.analyze(request)
    
    return {
        "success": True,
        "data": {
            "primary_recommendation": {
                "action": result.decision.type.value,
                "priority": result.decision.priority.name,
                "reasoning": result.decision.reasoning,
                "confidence": result.confidence
            },
            "risk_summary": {
                "score": result.risk_score,
                "level": result.confidence_level.value
            },
            "expected_outcome": {
                "reward": result.expected_reward
            },
            "recommendations": result.recommendations
        }
    }


@router.post("/automate")
async def automate_decision(request: AutoDecisionRequest):
    """
    自动执行决策
    
    自动执行已确认的决策
    """
    if not request.confirmed:
        return {
            "success": False,
            "message": "决策未确认，无法自动执行",
            "decision_id": request.decision_id
        }
    
    # TODO: 实现自动执行逻辑
    # - 调用相关业务系统API
    # - 记录执行日志
    # - 监控执行状态
    
    return {
        "success": True,
        "data": {
            "decision_id": request.decision_id,
            "status": "executed",
            "executed_at": datetime.now().isoformat(),
            "execution_params": request.execution_params
        }
    }


@router.get("/history")
async def get_decision_history(
    decision_type: Optional[str] = Query(None, description="决策类型"),
    priority: Optional[str] = Query(None, description="优先级"),
    start_date: Optional[datetime] = Query(None, description="开始日期"),
    end_date: Optional[datetime] = Query(None, description="结束日期"),
    limit: int = Query(50, description="返回数量限制")
):
    """
    获取决策历史
    
    根据条件筛选历史决策记录
    """
    # TODO: 实现历史记录查询
    # - 从数据库加载历史记录
    # - 应用过滤条件
    # - 返回结果
    
    return {
        "success": True,
        "data": {
            "total": 0,
            "decisions": [],
            "filters": {
                "decision_type": decision_type,
                "priority": priority,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
        }
    }


@router.post("/risk/assess")
async def assess_risk(business_data: BusinessDataModel):
    """
    风险评估
    
    对业务数据进行全面风险评估
    """
    risk_assessor = get_risk_assessor()
    
    result = await risk_assessor.assess(business_data.dict())
    
    return {
        "success": True,
        "data": {
            "overall_score": result.overall_score,
            "level": result.level.value,
            "category_scores": {
                cat.value: score for cat, score in result.category_scores.items()
            },
            "trend": result.trend.value,
            "risk_tolerance_status": result.risk_tolerance_status,
            "recommendations": result.recommendations,
            "top_risks": [
                {
                    "name": factor.name,
                    "category": factor.category.value,
                    "score": factor.score,
                    "severity": factor.severity,
                    "description": factor.description
                }
                for factor in sorted(result.factors, key=lambda x: x.score, reverse=True)[:5]
            ],
            "next_review_date": result.next_review_date.isoformat() if result.next_review_date else None,
            "assessed_at": result.assessed_at.isoformat()
        }
    }


@router.post("/predict")
async def generate_prediction(
    business_data: BusinessDataModel,
    horizon: int = Query(30, description="预测天数")
):
    """
    预测分析
    
    基于业务数据生成预测分析
    """
    predictor = get_predictor()
    
    result = await predictor.predict(business_data.dict(), horizon)
    
    return {
        "success": True,
        "data": {
            "forecast_horizon": result.forecast_horizon,
            "trend": {
                "direction": result.trend.direction.value,
                "strength": result.trend.strength,
                "slope": result.trend.slope,
                "r_squared": result.trend.r_squared
            },
            "seasonality": {
                "has_seasonality": result.seasonality.has_seasonality,
                "type": result.seasonality.type.value if result.seasonality.type else None,
                "strength": result.seasonality.strength,
                "amplitude": result.seasonality.amplitude,
                "peaks": result.seasonality.peaks,
                "troughs": result.seasonality.troughs
            },
            "anomalies": [
                {
                    "timestamp": anomaly.timestamp.isoformat(),
                    "value": anomaly.value,
                    "deviation": anomaly.deviation,
                    "severity": anomaly.severity,
                    "description": anomaly.description,
                    "category": anomaly.category
                }
                for anomaly in result.anomalies
            ],
            "confidence_interval": list(result.confidence_interval),
            "model_accuracy": result.model_accuracy,
            "forecast_points": [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "value": point.value,
                    "lower_bound": point.lower_bound,
                    "upper_bound": point.upper_bound
                }
                for point in result.forecast[:10]  # 只返回前10个预测点
            ],
            "generated_at": result.generated_at.isoformat()
        }
    }


@router.get("/types")
async def get_decision_types():
    """
    获取支持的决策类型
    """
    return {
        "success": True,
        "data": {
            "decision_types": [
                {"value": dt.value, "name": dt.name}
                for dt in DecisionType
            ],
            "priority_levels": [
                {"value": p.value, "name": p.name}
                for p in DecisionPriority
            ],
            "confidence_levels": [
                {"value": cl.value, "name": cl.name}
                for cl in ConfidenceLevel
            ],
            "risk_levels": [
                {"value": rl.value, "name": rl.name}
                for rl in RiskLevel
            ]
        }
    }


@router.get("/health")
async def health_check():
    """
    健康检查
    """
    return {
        "status": "healthy",
        "service": "decision-engine",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }
