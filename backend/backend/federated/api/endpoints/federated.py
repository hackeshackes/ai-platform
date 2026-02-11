"""
API endpoints for Federated Learning Platform
"""
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

from ...models import (
    FLSession,
    FLConfig,
    LocalModel,
    GlobalModel,
    FLClientInfo,
    SessionStatus,
    TrainingResult,
    AggregationResult
)

router = APIRouter(prefix="/federated", tags=["Federated Learning"])


class CreateSessionRequest(BaseModel):
    """创建会话请求"""
    model_name: str = "default_model"
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    min_clients: int = 2
    max_clients: int = 10
    differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_norm: float = 1.0
    rounds: int = 10


class JoinSessionRequest(BaseModel):
    """加入会话请求"""
    client_id: str
    data_size: int
    model_version: str = "1.0"


class SubmitModelRequest(BaseModel):
    """提交模型请求"""
    client_id: str
    weights: Dict[str, Any]
    gradients: Optional[Dict[str, Any]] = None
    data_size: int
    accuracy: float = 0.0
    loss: float = 0.0
    version: str = "1.0"


class SessionResponse(BaseModel):
    """会话响应"""
    id: str
    status: str
    participants: List[Dict[str, Any]]
    current_round: int
    created_at: datetime


class GlobalModelResponse(BaseModel):
    """全局模型响应"""
    session_id: str
    version: str
    round_number: int
    weights: Dict[str, Any]


class StatusResponse(BaseModel):
    """状态响应"""
    session_id: str
    status: str
    current_round: int
    total_rounds: int
    participants_count: int
    global_model_version: str
    created_at: datetime
    completed_at: Optional[datetime] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    active_sessions: int
    total_sessions: int


# 全局平台实例 (实际应用中应使用依赖注入)
platform = None


def get_platform():
    """获取平台实例"""
    global platform
    if platform is None:
        from ..fl_platform import FederatedLearningPlatform
        from ..storage import SessionStore
        from ..aggregator import Aggregator
        platform = FederatedLearningPlatform(
            storage=SessionStore(),
            aggregator=Aggregator(aggregation_method="fedavg"),
            tls_enabled=False
        )
    return platform


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    platform = Depends(get_platform)
):
    """
    创建联邦训练会话
    
    - **model_name**: 模型名称
    - **local_epochs**: 本地训练轮数
    - **learning_rate**: 学习率
    - **min_clients**: 最少客户端数
    - **max_clients**: 最多客户端数
    - **differential_privacy**: 是否启用差分隐私
    """
    try:
        config = FLConfig(
            model_name=request.model_name,
            local_epochs=request.local_epochs,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            min_clients=request.min_clients,
            max_clients=request.max_clients,
            differential_privacy=request.differential_privacy,
            dp_epsilon=request.dp_epsilon,
            dp_delta=request.dp_delta,
            dp_max_norm=request.dp_max_norm,
            rounds=request.rounds
        )
        
        session = await platform.create_session(config)
        
        return SessionResponse(
            id=session.id,
            status=session.status.value,
            participants=[p.model_dump() for p in session.participants],
            current_round=session.current_round,
            created_at=session.created_at
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )


@router.post("/sessions/{session_id}/join", response_model=dict)
async def join_session(
    session_id: str,
    request: JoinSessionRequest,
    platform = Depends(get_platform)
):
    """
    加入联邦训练会话
    
    - **session_id**: 会话ID
    - **client_id**: 客户端ID
    - **data_size**: 数据量
    """
    client = FLClientInfo(
        client_id=request.client_id,
        data_size=request.data_size,
        model_version=request.model_version
    )
    
    success = await platform.join_session(session_id, client)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to join session"
        )
    
    return {"status": "joined", "session_id": session_id, "client_id": request.client_id}


@router.get("/sessions/{session_id}/status", response_model=StatusResponse)
async def get_status(
    session_id: str,
    platform = Depends(get_platform)
):
    """
    获取训练状态
    
    - **session_id**: 会话ID
    """
    session = await platform.get_session_status(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return StatusResponse(
        session_id=session.id,
        status=session.status.value,
        current_round=session.current_round,
        total_rounds=session.config.rounds,
        participants_count=len(session.participants),
        global_model_version=session.global_model_version,
        created_at=session.created_at,
        completed_at=session.completed_at
    )


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    status: Optional[str] = None,
    limit: int = 100,
    platform = Depends(get_platform)
):
    """
    列出所有会话
    
    - **status**: 按状态过滤 (pending, training, aggregating, completed, failed)
    - **limit**: 返回数量限制
    """
    filter_status = SessionStatus(status) if status else None
    sessions = await platform.list_sessions(filter_status, limit)
    
    return [
        SessionResponse(
            id=s.id,
            status=s.status.value,
            participants=[p.model_dump() for p in s.participants],
            current_round=s.current_round,
            created_at=s.created_at
        )
        for s in sessions
    ]


@router.post("/sessions/{session_id}/start", response_model=StatusResponse)
async def start_training(
    session_id: str,
    platform = Depends(get_platform)
):
    """
    开始训练
    
    - **session_id**: 会话ID
    """
    try:
        session = await platform.start_training(session_id)
        
        return StatusResponse(
            session_id=session.id,
            status=session.status.value,
            current_round=session.current_round,
            total_rounds=session.config.rounds,
            participants_count=len(session.participants),
            global_model_version=session.global_model_version,
            created_at=session.created_at
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/sessions/{session_id}/aggregate", response_model=GlobalModelResponse)
async def aggregate_models(
    session_id: str,
    platform = Depends(get_platform)
):
    """
    聚合模型
    
    - **session_id**: 会话ID
    """
    try:
        global_model = await platform.aggregate_models(session_id)
        
        return GlobalModelResponse(
            session_id=global_model.session_id,
            version=global_model.version,
            round_number=global_model.round_number,
            weights=global_model.weights
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/sessions/{session_id}/model", response_model=GlobalModelResponse)
async def get_global_model(
    session_id: str,
    platform = Depends(get_platform)
):
    """
    获取全局模型
    
    - **session_id**: 会话ID
    """
    global_model = await platform.get_global_model(session_id)
    
    if not global_model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Global model not found"
        )
    
    return GlobalModelResponse(
        session_id=global_model.session_id,
        version=global_model.version,
        round_number=global_model.round_number,
        weights=global_model.weights
    )


@router.post("/sessions/{session_id}/submit", response_model=dict)
async def submit_local_model(
    session_id: str,
    request: SubmitModelRequest,
    platform = Depends(get_platform)
):
    """
    提交本地模型
    
    - **session_id**: 会话ID
    """
    try:
        local_model = LocalModel(
            client_id=request.client_id,
            weights=request.weights,
            gradients=request.gradients or {},
            data_size=request.data_size,
            accuracy=request.accuracy,
            loss=request.loss,
            version=request.version
        )
        
        await platform.submit_local_model(session_id, local_model)
        
        return {"status": "submitted", "session_id": session_id, "client_id": request.client_id}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/sessions/{session_id}/privacy", response_model=dict)
async def get_privacy_report(
    session_id: str,
    platform = Depends(get_platform)
):
    """
    获取隐私报告
    
    - **session_id**: 会话ID
    """
    report = await platform.get_privacy_report(session_id)
    return report


@router.get("/health", response_model=HealthResponse)
async def health_check(
    platform = Depends(get_platform)
):
    """
    健康检查
    """
    return HealthResponse(
        status="healthy",
        active_sessions=platform.active_session_count,
        total_sessions=platform.total_sessions
    )
