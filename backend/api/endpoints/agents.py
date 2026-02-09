"""
Agent API端点 v2.0 Phase 3
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict
from pydantic import BaseModel
from datetime import datetime

from backend.agents.framework import agent_framework, AgentRole
from backend.core.auth import get_current_user

router = APIRouter()

class CreateAgentModel(BaseModel):
    name: str
    role: str
    system_prompt: str
    tool_names: List[str]
    metadata: Optional[Dict] = None

class ExecuteAgentModel(BaseModel):
    task: str
    context: Optional[Dict] = None

class OrchestrateModel(BaseModel):
    task: str
    agent_ids: List[str]
    strategy: str = "sequential"

class AddMemoryModel(BaseModel):
    content: str
    importance: float = 0.5

@router.post("/agents")
async def create_agent(request: CreateAgentModel, current_user = Depends(get_current_user)):
    """
    创建Agent
    
    v2.0 Phase 3: Agent编排
    """
    try:
        role = AgentRole(request.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}")
    
    agent = agent_framework.create_agent(
        name=request.name,
        role=role,
        system_prompt=request.system_prompt,
        tool_names=request.tool_names,
        metadata=request.metadata
    )
    
    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "role": agent.role.value,
        "status": agent.status.value
    }

@router.get("/agents")
async def list_agents():
    """
    列出所有Agent
    
    v2.0 Phase 3: Agent编排
    """
    return {"agents": agent_framework.list_agents()}

@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """
    获取Agent详情
    
    v2.0 Phase 3: Agent编排
    """
    agent = agent_framework.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "role": agent.role.value,
        "status": agent.status.value,
        "system_prompt": agent.system_prompt,
        "tools": [t.name for t in agent.tools],
        "memory_count": len(agent.memories),
        "created_at": agent.created_at.isoformat()
    }

@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """
    删除Agent
    
    v2.0 Phase 3: Agent编排
    """
    if agent_framework.delete_agent(agent_id):
        return {"message": "Agent deleted"}
    raise HTTPException(status_code=404, detail="Agent not found")

@router.post("/agents/{agent_id}/execute")
async def execute_agent(agent_id: str, request: ExecuteAgentModel):
    """
    执行Agent
    
    v2.0 Phase 3: Agent编排
    """
    try:
        result = await agent_framework.execute_agent(
            agent_id=agent_id,
            task=request.task,
            context=request.context
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/agents/{agent_id}/memory")
async def add_memory(agent_id: str, request: AddMemoryModel):
    """
    添加记忆
    
    v2.0 Phase 3: Agent编排
    """
    agent_framework.add_memory(
        agent_id=agent_id,
        content=request.content,
        importance=request.importance
    )
    return {"message": "Memory added"}

@router.get("/tools")
async def list_tools():
    """
    列出可用工具
    
    v2.0 Phase 3: Agent编排
    """
    return {"tools": agent_framework.get_tools()}

@router.post("/orchestrate")
async def orchestrate(request: OrchestrateModel):
    """
    编排多Agent协作
    
    v2.0 Phase 3: Agent编排
    """
    result = await agent_framework.orchestrate(
        task=request.task,
        agent_ids=request.agent_ids,
        strategy=request.strategy
    )
    return result

@router.get("/roles")
async def list_roles():
    """
    列出Agent角色
    
    v2.0 Phase 3: Agent编排
    """
    return {
        "roles": [
            {"id": r.value, "name": r.name}
            for r in AgentRole
        ]
    }
