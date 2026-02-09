"""
assistant.py - AI Platform v2.3
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
module_path = os.path.join(backend_dir, 'assistant/assistant.py')

spec = importlib.util.spec_from_file_location("gateway_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    ai_assistant = module.ai_assistant
except Exception as e:
    print(f"Failed to import module: {e}")
    ai_assistant = None

from api.endpoints.auth import get_current_user

router = APIRouter()
class CreateConversationModel(BaseModel):
    pass

class ChatModel(BaseModel):
    conversation_id: str
    message: str

class DiagnoseModel(BaseModel):
    problem: str

@router.post("/conversations")
async def create_conversation(current_user = Depends(get_current_user)):
    """
    创建对话
    
    v2.3: AI Assistant
    """
    conversation = await ai_assistant.create_conversation(
        user_id=str(current_user.id)
    )
    
    return {
        "conversation_id": conversation.conversation_id,
        "created_at": conversation.created_at.isoformat(),
        "message": "Conversation created"
    }

@router.get("/conversations")
async def list_conversations(
    limit: int = 100,
    current_user = Depends(get_current_user)
):
    """
    列出对话
    
    v2.3: AI Assistant
    """
    conversations = await ai_assistant.list_conversations(
        user_id=str(current_user.id),
        limit=limit
    )
    
    return {
        "total": len(conversations),
        "conversations": [
            {
                "conversation_id": c.conversation_id,
                "messages_count": len(c.messages),
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat()
            }
            for c in conversations
        ]
    }

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    获取对话详情
    
    v2.3: AI Assistant
    """
    conversation = await ai_assistant.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation.conversation_id,
        "user_id": conversation.user_id,
        "messages": conversation.messages,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat()
    }

@router.post("/chat")
async def chat(request: ChatModel):
    """
    发送消息
    
    v2.3: AI Assistant
    """
    try:
        response = await ai_assistant.chat(
            conversation_id=request.conversation_id,
            message=request.message
        )
        
        return {
            "response_id": response.response_id,
            "conversation_id": response.conversation_id,
            "message": response.message,
            "suggestions": response.suggestions,
            "actions": response.actions,
            "confidence": response.confidence,
            "sources": response.sources
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/diagnose")
async def diagnose(request: DiagnoseModel):
    """
    问题诊断
    
    v2.3: AI Assistant
    """
    diagnosis = await ai_assistant.diagnose(request.problem)
    
    return diagnosis

@router.get("/knowledge")
async def search_knowledge(query: str):
    """
    搜索知识库
    
    v2.3: AI Assistant
    """
    results = ai_assistant.search_knowledge(query)
    
    return {
        "total": len(results),
        "results": results
    }

@router.get("/health")
async def assistant_health():
    """
    Assistant健康检查
    
    v2.3: AI Assistant
    """
    conversations = len(ai_assistant.conversations)
    knowledge_count = len(ai_assistant.knowledge_base)
    rules_count = len(ai_assistant.rule_engine.rules)
    
    return {
        "status": "healthy",
        "conversations": conversations,
        "knowledge_entries": knowledge_count,
        "rules": rules_count
    }
