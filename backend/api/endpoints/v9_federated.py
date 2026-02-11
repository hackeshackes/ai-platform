"""
V9 Federated Learning API (修复版)
"""
from typing import Dict, List
from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/v9/federated", tags=["V9 Federated Learning"])

_sessions = {}

class SessionRequest(BaseModel):
    name: str
    model_type: str = "classifier"
    num_rounds: int = 10
    min_clients: int = 2

class JoinRequest(BaseModel):
    client_id: str
    data_size: int

@router.get("/sessions")
async def list_sessions():
    """列出所有会话"""
    return {"sessions": list(_sessions.values()), "total": len(_sessions)}

@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """获取会话详情"""
    if session_id not in _sessions:
        return {"error": "session not found"}
    return _sessions[session_id]

@router.post("/sessions")
async def create_session(request: SessionRequest):
    """创建联邦训练会话"""
    import uuid
    session_id = f"fl-{uuid.uuid4().hex[:8]}"
    _sessions[session_id] = {
        "session_id": session_id,
        "name": request.name,
        "model_type": request.model_type,
        "num_rounds": request.num_rounds,
        "min_clients": request.min_clients,
        "status": "pending",
        "participants": [],
        "created_at": "2026-02-10T15:00:00"
    }
    return _sessions[session_id]

@router.post("/sessions/{session_id}/join")
async def join_session(session_id: str, request: JoinRequest):
    """加入会话"""
    if session_id not in _sessions:
        return {"error": "session not found"}
    
    _sessions[session_id]["participants"].append({
        "client_id": request.client_id,
        "data_size": request.data_size
    })
    
    return {
        "session_id": session_id,
        "client_id": request.client_id,
        "status": "joined",
        "total_participants": len(_sessions[session_id]["participants"])
    }

@router.get("/privacy/config")
async def privacy_config():
    """隐私配置"""
    return {
        "epsilon": 1.0,
        "delta": 1e-5,
        "clip_norm": 1.0,
        "noise_type": "gaussian"
    }

@router.get("/aggregators")
async def list_aggregators():
    """聚合算法"""
    return {
        "algorithms": ["FedAvg", "FedMedian", "FedTrimmedAvg", "FedProx"],
        "default": "FedAvg"
    }
