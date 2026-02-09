"""Ollama integration endpoints"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import httpx
import ollama
from datetime import datetime, timedelta

router = APIRouter()

OLLAMA_BASE_URL = "http://localhost:11434"

class ChatRequest(BaseModel):
    model: str
    messages: List[dict]  # [{"role": "user", "content": "..."}]
    stream: bool = False
    options: Optional[dict] = None

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    options: Optional[dict] = None

@router.get("/status")
async def ollama_status():
    """检查Ollama连接状态"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                return {"status": "connected", "base_url": OLLAMA_BASE_URL}
            else:
                return {"status": "disconnected", "code": response.status_code}
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}

@router.get("/models")
async def list_ollama_models():
    """获取已下载模型列表"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to fetch models")
            
            data = response.json()
            return {
                "models": [
                    {
                        "name": m["name"],
                        "size": m["size"],
                        "modified_at": m.get("modified_at")
                    }
                    for m in data.get("models", [])
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pull")
async def pull_model(model_name: str):
    """拉取模型"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": model_name}
            )
            return {"status": "pulling", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat(request: ChatRequest):
    """聊天对话"""
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "model": request.model,
                "messages": request.messages,
                "stream": request.stream
            }
            if request.options:
                payload["options"] = request.options
            
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Chat failed")
            
            data = response.json()
            return {
                "model": request.model,
                "response": data["message"]["content"],
                "total_duration": data.get("total_duration"),
                "eval_count": data.get("eval_count")
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def generate(request: GenerateRequest):
    """生成文本"""
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "stream": request.stream
            }
            if request.options:
                payload["options"] = request.options
            
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Generate failed")
            
            data = response.json()
            return {
                "model": request.model,
                "response": data["response"],
                "total_duration": data.get("total_duration"),
                "eval_count": data.get("eval_count")
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """删除模型"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{OLLAMA_BASE_URL}/api/delete",
                json={"name": model_name}
            )
            return {"status": "deleted", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
