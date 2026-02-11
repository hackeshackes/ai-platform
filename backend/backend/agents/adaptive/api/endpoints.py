"""
API Endpoints - Adaptive Learning
API端点 - 自适应学习
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ..learner import AdaptiveLearner, InteractionRequest
from ..evaluator import Evaluator
from ..knowledge_base import KnowledgeBase

# 创建路由
router = APIRouter(prefix="/adaptive", tags=["Adaptive Learning"])

# 全局组件实例
_global_kb: Optional[KnowledgeBase] = None
_global_evaluator: Optional[Evaluator] = None
_learners: Dict[str, AdaptiveLearner] = {}


def get_learner(agent_id: str) -> AdaptiveLearner:
    """获取或创建Learner实例"""
    global _learners, _global_kb, _global_evaluator
    
    if agent_id not in _learners:
        _learners[agent_id] = AdaptiveLearner(
            agent_id=agent_id,
            knowledge_base=_global_kb,
            evaluator=_global_evaluator
        )
    
    return _learners[agent_id]


# 请求/响应模型
class InteractionRequestModel(BaseModel):
    """交互请求模型"""
    text: str
    context: Optional[Dict] = None
    actions: Optional[List[Dict]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class LearningResultModel(BaseModel):
    """学习结果模型"""
    success: bool
    pattern_id: str
    message: str = ""
    details: Dict = {}


class StrategyRequestModel(BaseModel):
    """策略请求模型"""
    interaction: InteractionRequestModel
    reward: float = 0.0


class PatternResponseModel(BaseModel):
    """模式响应模型"""
    id: str
    pattern_type: str
    frequency: int
    success_count: int
    avg_reward: float
    last_seen: str


class StatusResponseModel(BaseModel):
    """状态响应模型"""
    agent_id: str
    success_rate: float
    improvement_rate: float
    consistency: float
    total_interactions: int
    details: Dict = {}


# API端点

@router.post("/agents/{agent_id}/learn", response_model=LearningResultModel)
async def agent_learn(agent_id: str, request: InteractionRequestModel):
    """
    Agent从交互中学习
    
    接收用户交互，提取模式，更新知识库，优化策略
    """
    learner = get_learner(agent_id)
    
    # 转换为内部请求
    interaction_request = InteractionRequest(
        text=request.text,
        context=request.context,
        actions=request.actions,
        session_id=request.session_id,
        user_id=request.user_id
    )
    
    result = await learner.learn_from_request(interaction_request)
    
    return {
        "success": result.success,
        "pattern_id": result.pattern_id,
        "message": result.message,
        "details": result.details
    }


@router.get("/agents/{agent_id}/status", response_model=StatusResponseModel)
async def get_status(agent_id: str):
    """
    获取Agent学习状态
    
    返回Agent的学习效果评估结果
    """
    learner = get_learner(agent_id)
    status = await learner.get_learning_status()
    
    evaluation = status.get("evaluation", {})
    
    return {
        "agent_id": agent_id,
        "success_rate": evaluation.get("success_rate", 0.0),
        "improvement_rate": evaluation.get("improvement_rate", 0.0),
        "consistency": evaluation.get("consistency", 0.0),
        "total_interactions": evaluation.get("total_interactions", 0),
        "details": {
            "learning_count": status.get("learning_count", 0),
            "uptime_seconds": status.get("uptime_seconds", 0),
            "optimizer_stats": status.get("optimizer_stats", {}),
            "knowledge_base_stats": status.get("knowledge_base_stats", {})
        }
    }


@router.get("/patterns", response_model=List[PatternResponseModel])
async def list_patterns(intent_type: Optional[str] = None, limit: int = 100):
    """
    列出所有学习模式
    
    按意图类型过滤，返回模式列表
    """
    from .models import IntentType
    
    kb = _global_kb or KnowledgeBase()
    
    intent = None
    if intent_type:
        try:
            intent = IntentType(intent_type)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid intent type")
    
    patterns = await kb.list(intent_type=intent, limit=limit)
    
    return [
        {
            "id": p.id,
            "pattern_type": p.pattern_type,
            "frequency": p.frequency,
            "success_count": p.success_count,
            "avg_reward": p.avg_reward,
            "last_seen": p.last_seen.isoformat()
        }
        for p in patterns
    ]


@router.post("/agents/{agent_id}/optimize")
async def optimize_strategy(
    agent_id: str,
    request: StrategyRequestModel
):
    """
    优化Agent策略
    
    基于给定交互和奖励优化策略
    """
    learner = get_learner(agent_id)
    
    # 提取模式
    from .learner import Interaction
    interaction = Interaction(
        text=request.interaction.text,
        context=request.interaction.context or {},
        actions=request.interaction.actions or []
    )
    
    pattern = await learner.extractor.extract(interaction)
    
    # 优化策略
    strategy = await learner.optimizer.optimize(pattern, request.reward)
    
    return {
        "action": strategy.action,
        "q_value": strategy.q_value,
        "exploration": strategy.exploration,
        "confidence": strategy.confidence,
        "alternatives": strategy.alternatives
    }


@router.get("/agents/{agent_id}/recommend")
async def get_recommendation(agent_id: str, text: str, context: Optional[Dict] = None):
    """
    获取策略推荐
    
    基于给定文本返回推荐策略
    """
    learner = get_learner(agent_id)
    
    interaction = Interaction(
        text=text,
        context=context or {}
    )
    
    recommendation = await learner.get_recommended_strategy(interaction)
    
    return recommendation


@router.delete("/agents/{agent_id}")
async def reset_agent(agent_id: str):
    """
    重置Agent学习状态
    
    清空该Agent的学习历史和知识库
    """
    global _learners
    
    if agent_id in _learners:
        _learners[agent_id].reset()
        del _learners[agent_id]
    
    return {"message": f"Agent {agent_id} has been reset"}


@router.get("/stats")
async def get_global_stats():
    """
    获取全局统计
    
    返回所有Agent和学习模式的全局统计信息
    """
    kb = _global_kb or KnowledgeBase()
    kb_stats = await kb.get_stats()
    
    return {
        "total_patterns": kb_stats.get("total_patterns", 0),
        "total_interactions": kb_stats.get("total_interactions", 0),
        "success_rate": kb_stats.get("success_rate", 0.0),
        "intent_distribution": kb_stats.get("intent_distribution", {}),
        "active_agents": len(_learners)
    }


# 依赖注入配置
def set_global_knowledge_base(kb: KnowledgeBase):
    """设置全局知识库"""
    global _global_kb
    _global_kb = kb


def set_global_evaluator(evaluator: Evaluator):
    """设置全局评估器"""
    global _global_evaluator
    _global_evaluator = evaluator


def clear_learners():
    """清除所有Learner实例"""
    global _learners
    _learners.clear()
