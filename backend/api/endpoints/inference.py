"""Inference API endpoints"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastapi import APIRouter, HTTPException
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

# 预配置模型
PREDEFINED_MODELS = [
    {
        "id": "llama2-7b-chat",
        "name": "Llama-2-7b-chat-hf",
        "provider": "meta",
        "size": "7B",
        "status": "ready",
        "backend": "vLLM"
    },
    {
        "id": "qwen-7b-chat",
        "name": "Qwen-7B-Chat",
        "provider": "alibaba",
        "size": "7B",
        "status": "ready",
        "backend": "vLLM"
    },
    {
        "id": "baichuan-7b-chat",
        "name": "Baichuan-7B-Chat",
        "provider": "baichuan",
        "size": "7B",
        "status": "ready",
        "backend": "vLLM"
    }
]

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class InferenceRequest(BaseModel):
    model_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class InferenceResponse(BaseModel):
    id: str
    model: str
    prompt: str
    output: str
    usage: Usage
    latency_ms: float

@router.get("/models")
async def list_models():
    """获取可用推理模型"""
    return {"models": PREDEFINED_MODELS}

@router.post("/generate")
async def generate(request: InferenceRequest):
    """推理生成"""
    import uuid
    
    response = {
        "id": str(uuid.uuid4())[:8],
        "model": request.model_id,
        "prompt": request.prompt,
        "output": f"这是[{request.model_id}]的模拟推理输出。\n\n实际部署需要配置vLLM或Ollama服务。\n\n参数: max_tokens={request.max_tokens}, temperature={request.temperature}",
        "usage": {
            "prompt_tokens": len(request.prompt) // 4,
            "completion_tokens": request.max_tokens,
            "total_tokens": len(request.prompt) // 4 + request.max_tokens
        },
        "latency_ms": 123.45
    }
    
    return response

@router.get("/history")
async def get_history():
    """获取推理历史"""
    return {
        "history": [
            {
                "id": "abc123",
                "model": "llama2-7b-chat",
                "prompt": "什么是机器学习？",
                "output": "机器学习是...",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
    }
