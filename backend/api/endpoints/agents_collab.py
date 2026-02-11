"""
Agent协作API端点 - Agent Collaboration API Endpoints
提供多Agent协作的REST API接口
"""

from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

from agents.collab.manager import (
    collab_manager, 
    TaskStatus, 
    SessionStatus,
    Task,
    CollaborationSession
)
from agents.langchain.adapter import langchain_adapter
from agents.llamaindex.adapter import llamaindex_adapter

router = APIRouter(prefix="/agents", tags=["Agent Collaboration"])


# ============== 请求/响应模型 ==============

class CreateSessionRequest(BaseModel):
    """创建协作会话请求"""
    name: str = Field(..., min_length=1, max_length=200, description="会话名称")
    description: str = Field(default="", description="会话描述")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class CreateSessionResponse(BaseModel):
    """创建协作会话响应"""
    success: bool
    session: Dict[str, Any]
    message: str


class CreateTaskRequest(BaseModel):
    """创建任务请求"""
    name: str = Field(..., min_length=1, max_length=200, description="任务名称")
    description: str = Field(default="", description="任务描述")
    agent_type: str = Field(..., description="Agent类型标识")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="输入数据")
    dependencies: List[str] = Field(default_factory=list, description="依赖任务ID列表")


class ExecuteTaskRequest(BaseModel):
    """执行任务请求"""
    task_id: str = Field(..., description="任务ID")
    sync: bool = Field(default=True, description="是否同步执行")


class ImportLangChainRequest(BaseModel):
    """导入LangChain Agent请求"""
    name: str = Field(..., description="Agent名称")
    description: str = Field(default="", description="Agent描述")
    agent_class: str = Field(..., description="Agent类名")
    llm_type: str = Field(..., description="LLM类型")
    llm_config: Dict[str, Any] = Field(..., description="LLM配置")
    tools: List[str] = Field(default_factory=list, description="工具列表")
    prompt_template: str = Field(default=None, description="提示模板")
    memory_config: Dict[str, Any] = Field(default=None, description="记忆配置")


class ImportLlamaIndexRequest(BaseModel):
    """导入LlamaIndex Agent请求"""
    name: str = Field(..., description="Agent名称")
    description: str = Field(default="", description="Agent描述")
    agent_type: str = Field(..., description="Agent类型")
    llm_type: str = Field(..., description="LLM类型")
    llm_config: Dict[str, Any] = Field(..., description="LLM配置")
    index_type: str = Field(default=None, description="索引类型")
    index_config: Dict[str, Any] = Field(default=None, description="索引配置")
    system_prompt: str = Field(default=None, description="系统提示")
    tools: List[str] = Field(default_factory=list, description="工具列表")


# ============== 协作会话API ==============

@router.post("/collab/create", response_model=CreateSessionResponse)
async def create_collaboration_session(request: CreateSessionRequest):
    """
    创建新的协作会话
    
    创建一个新的多Agent协作会话，用于管理多个Agent的协同工作。
    """
    try:
        session = collab_manager.create_session(
            name=request.name,
            description=request.description,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "session": session.to_dict(),
            "message": f"Session '{request.name}' created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collab/sessions")
async def list_collaboration_sessions(status: str = None):
    """
    列出协作会话
    
    返回所有协作会话，或根据状态过滤。
    """
    try:
        session_status = None
        if status:
            try:
                session_status = SessionStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid status: {status}. Valid values: active, paused, completed, terminated"
                )
        
        sessions = collab_manager.list_sessions(status=session_status)
        return {
            "success": True,
            "sessions": [s.to_dict() for s in sessions],
            "count": len(sessions)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collab/sessions/{session_id}")
async def get_collaboration_session(session_id: str):
    """获取协作会话详情"""
    session = collab_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "session": session.to_dict()}


@router.delete("/collab/sessions/{session_id}")
async def delete_collaboration_session(session_id: str):
    """删除协作会话"""
    success = collab_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "message": "Session deleted"}


@router.post("/collab/sessions/{session_id}/tasks")
async def add_task(session_id: str, request: CreateTaskRequest):
    """向会话添加任务"""
    session = collab_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    task = collab_manager.create_task(
        session_id=session_id,
        name=request.name,
        description=request.description,
        agent_type=request.agent_type,
        input_data=request.input_data,
        dependencies=request.dependencies
    )
    
    if not task:
        raise HTTPException(status_code=500, detail="Failed to create task")
    
    return {
        "success": True,
        "task": task.to_dict(),
        "message": f"Task '{request.name}' added to session"
    }


@router.get("/collab/sessions/{session_id}/tasks")
async def list_session_tasks(session_id: str):
    """列出会话中的所有任务"""
    session = collab_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "success": True,
        "tasks": [t.to_dict() for t in session.tasks],
        "count": len(session.tasks)
    }


@router.get("/collab/sessions/{session_id}/tasks/{task_id}")
async def get_task(session_id: str, task_id: str):
    """获取任务详情"""
    task = collab_manager.get_task(session_id, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True, "task": task.to_dict()}


@router.post("/collab/execute")
async def execute_collaboration_task(request: ExecuteTaskRequest):
    """
    执行协作任务
    
    在协作会话中执行单个任务。
    """
    result = collab_manager.execute_task(
        session_id=request.task_id.split(":")[0] if ":" in request.task_id else "",
        task_id=request.task_id
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Execution failed"))
    
    return {
        "success": True,
        "result": result.get("result"),
        "task_id": request.task_id
    }


@router.post("/collab/sessions/{session_id}/execute-all")
async def execute_all_tasks(session_id: str, background_tasks: BackgroundTasks):
    """
    执行会话中的所有任务
    
    按依赖顺序执行所有任务。
    """
    session = collab_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 使用后台任务执行
    background_tasks.add_task(_run_all_tasks, session_id)
    
    return {
        "success": True,
        "message": "Execution started",
        "session_id": session_id,
        "task_count": len(session.tasks)
    }


async def _run_all_tasks(session_id: str):
    """内部方法：运行所有任务"""
    try:
        await collab_manager.execute_session_async(session_id)
    except Exception as e:
        print(f"Error executing session {session_id}: {e}")


@router.post("/collab/sessions/{session_id}/export")
async def export_session(session_id: str):
    """导出会话"""
    session_data = collab_manager.export_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_data


@router.post("/collab/import")
async def import_session(session_data: Dict[str, Any]):
    """导入会话"""
    session = collab_manager.import_session(session_data)
    return {
        "success": True,
        "session": session.to_dict(),
        "message": "Session imported successfully"
    }


# ============== LangChain Agent API ==============

@router.post("/langchain/import")
async def import_langchain_agent(request: ImportLangChainRequest):
    """
    导入LangChain Agent
    
    将LangChain Agent导入到协作平台。
    """
    result = langchain_adapter.import_agent({
        "name": request.name,
        "description": request.description,
        "agent_class": request.agent_class,
        "llm_type": request.llm_type,
        "llm_config": request.llm_config,
        "tools": request.tools,
        "prompt_template": request.prompt_template,
        "memory_config": request.memory_config
    })
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Import failed"))
    
    return result


@router.get("/langchain/agents")
async def list_langchain_agents():
    """列出所有导入的LangChain Agents"""
    agents = langchain_adapter.list_agents()
    return {
        "success": True,
        "agents": agents,
        "count": len(agents)
    }


@router.get("/langchain/agents/{agent_id}")
async def get_langchain_agent(agent_id: str):
    """获取LangChain Agent详情"""
    config = langchain_adapter.get_agent_config(agent_id)
    if not config:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"success": True, "agent": config.__dict__}


@router.post("/langchain/agents/{agent_id}/create")
async def create_langchain_instance(agent_id: str):
    """创建LangChain Agent实例"""
    result = langchain_adapter.create_agent_instance(agent_id)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Creation failed"))
    return result


@router.post("/langchain/agents/{agent_id}/execute")
async def execute_langchain_agent(agent_id: str, input_data: Dict[str, Any]):
    """执行LangChain Agent"""
    result = langchain_adapter.execute_agent(agent_id, input_data)
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Execution failed"))
    return result


@router.delete("/langchain/agents/{agent_id}")
async def delete_langchain_agent(agent_id: str):
    """删除LangChain Agent"""
    success = langchain_adapter.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"success": True, "message": "Agent deleted"}


@router.get("/langchain/agents/{agent_id}/export")
async def export_langchain_agent(agent_id: str):
    """导出LangChain Agent配置"""
    config = langchain_adapter.export_agent(agent_id)
    if not config:
        raise HTTPException(status_code=404, detail="Agent not found")
    return config


# ============== LlamaIndex Agent API ==============

@router.post("/llamaindex/import")
async def import_llamaindex_agent(request: ImportLlamaIndexRequest):
    """
    导入LlamaIndex Agent
    
    将LlamaIndex Agent/Engine导入到协作平台。
    """
    result = llamaindex_adapter.import_agent({
        "name": request.name,
        "description": request.description,
        "agent_type": request.agent_type,
        "llm_type": request.llm_type,
        "llm_config": request.llm_config,
        "index_type": request.index_type,
        "index_config": request.index_config,
        "system_prompt": request.system_prompt,
        "tools": request.tools
    })
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Import failed"))
    
    return result


@router.get("/llamaindex/agents")
async def list_llamaindex_agents():
    """列出所有导入的LlamaIndex Agents"""
    agents = llamaindex_adapter.list_agents()
    return {
        "success": True,
        "agents": agents,
        "count": len(agents)
    }


@router.get("/llamaindex/agents/{agent_id}")
async def get_llamaindex_agent(agent_id: str):
    """获取LlamaIndex Agent详情"""
    config = llamaindex_adapter.get_agent_config(agent_id)
    if not config:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"success": True, "agent": config.__dict__}


@router.post("/llamaindex/agents/{agent_id}/create")
async def create_llamaindex_instance(agent_id: str):
    """创建LlamaIndex Agent实例"""
    result = llamaindex_adapter.create_agent_instance(agent_id)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Creation failed"))
    return result


@router.post("/llamaindex/agents/{agent_id}/execute")
async def execute_llamaindex_agent(agent_id: str, input_data: Dict[str, Any]):
    """执行LlamaIndex Agent"""
    result = llamaindex_adapter.execute_agent(agent_id, input_data)
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Execution failed"))
    return result


@router.delete("/llamaindex/agents/{agent_id}")
async def delete_llamaindex_agent(agent_id: str):
    """删除LlamaIndex Agent"""
    success = llamaindex_adapter.delete_agent(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"success": True, "message": "Agent deleted"}


@router.get("/llamaindex/agents/{agent_id}/export")
async def export_llamaindex_agent(agent_id: str):
    """导出LlamaIndex Agent配置"""
    config = llamaindex_adapter.export_agent(agent_id)
    if not config:
        raise HTTPException(status_code=404, detail="Agent not found")
    return config


@router.post("/llamaindex/agents/{agent_id}/add-documents")
async def add_documents_to_index(agent_id: str, documents: List[Dict[str, Any]]):
    """向LlamaIndex索引添加文档"""
    result = llamaindex_adapter.add_documents_to_index(agent_id, documents)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to add documents"))
    return result


# ============== 注册到协作管理器 ==============

def register_agents_to_collab():
    """将LangChain和LlamaIndex Agents注册到协作管理器"""
    
    def langchain_wrapper(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """LangChain Agent包装器"""
        agent_id = input_data.pop("_agent_id", None)
        if not agent_id:
            return {"error": "No agent_id provided"}
        result = langchain_adapter.execute_agent(agent_id, input_data)
        return result
    
    def llamaindex_wrapper(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """LlamaIndex Agent包装器"""
        agent_id = input_data.pop("_agent_id", None)
        if not agent_id:
            return {"error": "No agent_id provided"}
        result = llamaindex_adapter.execute_agent(agent_id, input_data)
        return result
    
    collab_manager.registry.register("langchain", langchain_wrapper)
    collab_manager.registry.register("llamaindex", llamaindex_wrapper)


# 初始化注册
register_agents_to_collab()
