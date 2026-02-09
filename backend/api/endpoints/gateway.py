"""
gateway.py - AI Platform v2.3
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

# 直接导入模块
import importlib.util
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'gateway/gateway.py')

spec = importlib.util.spec_from_file_location("gateway_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    ai_gateway = module.ai_gateway
    ProviderType = module.ProviderType
except Exception as e:
    print(f"Failed to import module: {e}")
    ai_gateway = None
    ProviderType = None

from api.endpoints.auth import get_current_user

router = APIRouter()
class RegisterProviderModel(BaseModel):
    name: str
    provider_type: str  # openai, anthropic, local, azure
    base_url: str
    api_key: str
    models: Optional[List[str]] = None
    priority: int = 0
    rate_limit: Optional[Dict[str, int]] = None

class UpdateProviderModel(BaseModel):
    name: Optional[str] = None
    api_key: Optional[str] = None
    enabled: Optional[bool] = None
    priority: Optional[int] = None
    rate_limit: Optional[Dict[str, int]] = None

class CreateRouteModel(BaseModel):
    name: str
    patterns: List[str]
    provider_id: Optional[str] = None
    weight: int = 100

class ChatRequestModel(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    provider_id: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

@router.get("/providers")
async def list_providers(enabled: Optional[bool] = None):
    """
    列出LLM提供商
    
    v2.3: AI Gateway
    """
    providers = ai_gateway.list_providers(enabled=enabled)
    
    return {
        "total": len(providers),
        "providers": [
            {
                "provider_id": p.provider_id,
                "name": p.name,
                "type": p.provider_type,
                "models": p.models,
                "enabled": p.enabled,
                "priority": p.priority,
                "rate_limit": p.rate_limit
            }
            for p in providers
        ]
    }

@router.post("/providers")
async def register_provider(request: RegisterProviderModel):
    """
    注册LLM提供商
    
    v2.3: AI Gateway
    """
    try:
        provider = ai_gateway.register_provider(
            name=request.name,
            provider_type=ProviderType(request.provider_type),
            base_url=request.base_url,
            api_key=request.api_key,
            models=request.models,
            priority=request.priority,
            rate_limit=request.rate_limit
        )
        
        return {
            "provider_id": provider.provider_id,
            "name": provider.name,
            "type": provider.provider_type,
            "message": "Provider registered"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/providers/{provider_id}")
async def update_provider(
    provider_id: str,
    request: UpdateProviderModel
):
    """
    更新提供商
    
    v2.3: AI Gateway
    """
    kwargs = request.model_dump(exclude_unset=True)
    result = ai_gateway.update_provider(provider_id, **kwargs)
    
    if not result:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    return {"message": "Provider updated"}

@router.get("/providers/{provider_id}")
async def get_provider(provider_id: str):
    """
    获取提供商详情
    
    v2.3: AI Gateway
    """
    provider = ai_gateway.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    return {
        "provider_id": provider.provider_id,
        "name": provider.name,
        "type": provider.provider_type,
        "base_url": provider.base_url,
        "models": provider.models,
        "enabled": provider.enabled,
        "priority": provider.priority,
        "rate_limit": provider.rate_limit
    }

@router.post("/routes")
async def create_route(request: CreateRouteModel):
    """
    创建路由规则
    
    v2.3: AI Gateway
    """
    route = ai_gateway.create_route(
        name=request.name,
        patterns=request.patterns,
        provider_id=request.provider_id,
        weight=request.weight
    )
    
    return {
        "route_id": route.route_id,
        "name": route.name,
        "patterns": route.patterns,
        "message": "Route created"
    }

@router.get("/routes")
async def list_routes():
    """
    列出路由规则
    
    v2.3: AI Gateway
    """
    routes = list(ai_gateway.routes.values())
    
    return {
        "total": len(routes),
        "routes": [
            {
                "route_id": r.route_id,
                "name": r.name,
                "patterns": r.patterns,
                "provider_id": r.provider_id,
                "weight": r.weight
            }
            for r in routes
        ]
    }

@router.post("/chat/completions")
async def chat_completions(request: ChatRequestModel):
    """
    Chat Completions (统一接口)
    
    v2.3: AI Gateway
    """
    try:
        response = await ai_gateway.chat_completions(
            model=request.model,
            messages=request.messages,
            provider_id=request.provider_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/usage")
async def get_usage_stats(provider_id: Optional[str] = None):
    """
    获取使用统计
    
    v2.3: AI Gateway
    """
    stats = ai_gateway.get_usage_stats(provider_id=provider_id)
    return stats

@router.get("/cost")
async def get_cost_breakdown(days: int = 30):
    """
    获取成本分析
    
    v2.3: AI Gateway
    """
    breakdown = ai_gateway.get_cost_breakdown(days=days)
    return breakdown

@router.get("/logs")
async def get_request_logs(
    provider_id: Optional[str] = None,
    limit: int = 100
):
    """
    获取请求日志
    
    v2.3: AI Gateway
    """
    logs = ai_gateway.request_logs[-limit:]
    
    if provider_id:
        logs = [l for l in logs if l.provider_id == provider_id]
    
    return {
        "total": len(logs),
        "logs": [
            {
                "log_id": l.log_id,
                "provider_id": l.provider_id,
                "model": l.model,
                "prompt_tokens": l.prompt_tokens,
                "completion_tokens": l.completion_tokens,
                "latency_ms": l.latency_ms,
                "cost": l.cost,
                "status": l.status,
                "timestamp": l.timestamp.isoformat()
            }
            for l in logs
        ]
    }

@router.get("/health")
async def gateway_health():
    """
    Gateway健康检查
    
    v2.3: AI Gateway
    """
    providers = ai_gateway.list_providers(enabled=True)
    
    return {
        "status": "healthy",
        "providers_count": len(providers),
        "routes_count": len(ai_gateway.routes),
        "total_requests": sum(
            s.total_requests 
            for s in ai_gateway.usage_stats.values()
        )
    }
