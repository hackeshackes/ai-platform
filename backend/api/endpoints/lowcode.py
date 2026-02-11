"""
低代码Agent构建器API端点
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import uuid
import json
from datetime import datetime

from backend.lowcode.builder import FlowBuilder, FlowExecutor, AgentFlow, FlowTemplate
from backend.lowcode.compiler import FlowCompiler, Serializer
from backend.lowcode.validator import FlowValidator, ValidationResult
from backend.lowcode.nodes import node_registry


# 创建路由器
router = APIRouter(prefix="/api/v1/lowcode", tags=["低代码Agent构建器"])


# ============ Pydantic Models ============

class NodeCreate(BaseModel):
    """创建节点请求"""
    type: str = Field(..., description="节点类型")
    name: str = Field("", description="节点名称")
    position: Optional[Dict[str, float]] = Field(None, description="位置坐标")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class NodeUpdate(BaseModel):
    """更新节点请求"""
    name: Optional[str] = None
    position: Optional[Dict[str, float]] = None
    config: Optional[Dict[str, Any]] = None


class EdgeCreate(BaseModel):
    """创建边请求"""
    source: str = Field(..., description="源节点ID")
    source_port: str = Field(..., description="源端口名")
    target: str = Field(..., description="目标节点ID")
    target_port: str = Field(..., description="目标端口名")
    label: Optional[str] = Field("", description="边标签")


class FlowCreate(BaseModel):
    """创建流程请求"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = ""
    nodes: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    edges: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    tags: Optional[List[str]] = Field(default_factory=list)
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class FlowUpdate(BaseModel):
    """更新流程请求"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None
    tags: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None


class FlowExecuteRequest(BaseModel):
    """执行流程请求"""
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    sync: bool = Field(True, description="是否同步执行")


class ValidateRequest(BaseModel):
    """验证流程请求"""
    flow: Dict[str, Any]
    strict: bool = Field(False, description="严格模式")


class TemplateCreate(BaseModel):
    """从模板创建请求"""
    template_id: str = Field(..., description="模板ID")
    name: Optional[str] = Field(None, description="流程名称")


# ============ 内存存储（生产环境请替换为数据库） ============

_flow_store: Dict[str, Dict] = {}


# ============ API Endpoints ============

@router.post("/agent", response_model=Dict[str, Any])
async def create_agent(flow_data: FlowCreate) -> Dict[str, Any]:
    """
    创建新的Agent流程
    
    - **name**: Agent名称
    - **description**: 描述（可选）
    - **nodes**: 初始节点列表（可选）
    - **edges**: 初始边列表（可选）
    """
    builder = FlowBuilder()
    
    if flow_data.name:
        builder.flow.name = flow_data.name
    if flow_data.description:
        builder.flow.description = flow_data.description
    if flow_data.tags:
        builder.flow.tags = flow_data.tags
    if flow_data.config:
        builder.flow.config = flow_data.config
    
    # 添加初始节点和边
    for node in flow_data.nodes:
        builder.add_node(
            node_type=node["type"],
            name=node.get("name", ""),
            position=node.get("position"),
            config=node.get("config", {})
        )
    
    for edge in flow_data.edges:
        builder.add_edge(
            source=edge["source"],
            source_port=edge.get("source_port", "output"),
            target=edge["target"],
            target_port=edge.get("target_port", "input"),
            label=edge.get("label", "")
        )
    
    # 验证
    validator = FlowValidator()
    result = validator.validate(builder.flow.to_dict())
    if not result.valid:
        return {
            "success": False,
            "errors": result.summary()
        }
    
    # 保存
    flow_dict = builder.flow.to_dict()
    _flow_store[builder.flow.id] = flow_dict
    
    return {
        "success": True,
        "data": flow_dict
    }


@router.get("/agent/{agent_id}", response_model=Dict[str, Any])
async def get_agent(agent_id: str) -> Dict[str, Any]:
    """
    获取Agent流程详情
    """
    if agent_id not in _flow_store:
        raise HTTPException(status_code=404, detail="Agent不存在")
    
    return {
        "success": True,
        "data": _flow_store[agent_id]
    }


@router.put("/agent/{agent_id}", response_model=Dict[str, Any])
async def update_agent(agent_id: str, 
                        updates: FlowUpdate) -> Dict[str, Any]:
    """
    更新Agent流程
    """
    if agent_id not in _flow_store:
        raise HTTPException(status_code=404, detail="Agent不存在")
    
    stored = _flow_store[agent_id]
    
    # 更新字段
    if updates.name is not None:
        stored["name"] = updates.name
    if updates.description is not None:
        stored["description"] = updates.description
    if updates.tags is not None:
        stored["tags"] = updates.tags
    if updates.config is not None:
        stored["config"] = updates.config
    if updates.nodes is not None:
        stored["nodes"] = updates.nodes
    if updates.edges is not None:
        stored["edges"] = updates.edges
    
    stored["updated_at"] = datetime.now().isoformat()
    
    # 验证更新后的流程
    validator = FlowValidator()
    result = validator.validate(stored)
    if not result.valid:
        return {
            "success": False,
            "errors": result.summary()
        }
    
    return {
        "success": True,
        "data": stored
    }


@router.delete("/agent/{agent_id}", response_model=Dict[str, Any])
async def delete_agent(agent_id: str) -> Dict[str, Any]:
    """
    删除Agent流程
    """
    if agent_id not in _flow_store:
        raise HTTPException(status_code=404, detail="Agent不存在")
    
    del _flow_store[agent_id]
    
    return {
        "success": True,
        "message": "Agent已删除"
    }


@router.post("/agent/{agent_id}/deploy", response_model=Dict[str, Any])
async def deploy_agent(agent_id: str) -> Dict[str, Any]:
    """
    部署Agent流程
    """
    if agent_id not in _flow_store:
        raise HTTPException(status_code=404, detail="Agent不存在")
    
    flow_data = _flow_store[agent_id]
    
    # 编译流程
    compiler = FlowCompiler()
    compiled = compiler.compile(
        AgentFlow(**flow_data),
        target="python"
    )
    
    # 更新状态
    _flow_store[agent_id]["status"] = "published"
    _flow_store[agent_id]["deployed_at"] = datetime.now().isoformat()
    _flow_store[agent_id]["compiled_code"] = compiled["code"]
    
    return {
        "success": True,
        "message": "Agent已部署",
        "data": {
            "agent_id": agent_id,
            "status": "published",
            "deployed_at": _flow_store[agent_id]["deployed_at"],
            "code": compiled["code"]
        }
    }


@router.post("/agent/{agent_id}/execute", response_model=Dict[str, Any])
async def execute_agent(agent_id: str,
                         request: FlowExecuteRequest) -> Dict[str, Any]:
    """
    执行Agent流程
    """
    if agent_id not in _flow_store:
        raise HTTPException(status_code=404, detail="Agent不存在")
    
    flow_data = _flow_store[agent_id]
    
    try:
        executor = FlowExecutor()
        result = executor.execute(
            AgentFlow(**flow_data),
            initial_context=request.context
        )
        
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/nodes", response_model=Dict[str, Any])
async def get_node_library() -> Dict[str, Any]:
    """
    获取可用节点库
    """
    nodes = node_registry.list_all()
    
    # 按分类组织
    categorized = {}
    for node in nodes:
        category = node.get("category", "unknown")
        if category not in categorized:
            categorized[category] = []
        categorized[category].append({
            "type": node["type"],
            "name": node["name"],
            "description": node.get("description", ""),
            "config": [
                {
                    "key": c["key"],
                    "name": c["name"],
                    "type": c["type"],
                    "required": c.get("required", False),
                    "default": c.get("default")
                }
                for c in node.get("config", [])
            ]
        })
    
    return {
        "success": True,
        "data": {
            "nodes": nodes,
            "categorized": categorized,
            "categories": list(categorized.keys())
        }
    }


@router.post("/validate", response_model=Dict[str, Any])
async def validate_flow(request: ValidateRequest) -> Dict[str, Any]:
    """
    验证流程定义
    """
    validator = FlowValidator()
    result = validator.validate(request.flow)
    
    return {
        "success": result.valid,
        "result": result.summary()
    }


@router.get("/templates", response_model=Dict[str, Any])
async def list_templates() -> Dict[str, Any]:
    """
    获取流程模板列表
    """
    templates = FlowTemplate.list_templates()
    
    return {
        "success": True,
        "data": templates
    }


@router.post("/templates", response_model=Dict[str, Any])
async def create_from_template(request: TemplateCreate) -> Dict[str, Any]:
    """
    从模板创建Agent流程
    """
    try:
        flow = FlowTemplate.create_from_template(
            template_id=request.template_id,
            name=request.name
        )
        
        # 保存
        flow_dict = flow.to_dict()
        flow_id = flow.id
        _flow_store[flow_id] = flow_dict
        
        return {
            "success": True,
            "data": flow_dict
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agent/{agent_id}/compile", response_model=Dict[str, Any])
async def compile_agent(agent_id: str, 
                         target: str = "python") -> Dict[str, Any]:
    """
    编译Agent流程
    """
    if agent_id not in _flow_store:
        raise HTTPException(status_code=404, detail="Agent不存在")
    
    flow_data = _flow_store[agent_id]
    
    try:
        compiler = FlowCompiler()
        result = compiler.compile(
            AgentFlow(**flow_data),
            target=target
        )
        
        return {
            "success": True,
            "data": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agent/{agent_id}/export", response_model=Dict[str, Any])
async def export_agent(agent_id: str, 
                        format: str = "json") -> Dict[str, Any]:
    """
    导出Agent流程
    """
    if agent_id not in _flow_store:
        raise HTTPException(status_code=404, detail="Agent不存在")
    
    flow_data = _flow_store[agent_id]
    flow = AgentFlow(**flow_data)
    
    if format == "json":
        content = Serializer.to_json(flow, pretty=True)
        return {
            "success": True,
            "format": "json",
            "content": content
        }
    elif format == "yaml":
        content = Serializer.to_yaml(flow)
        return {
            "success": True,
            "format": "yaml",
            "content": content
        }
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format: {format}"
        )


@router.get("/agents", response_model=Dict[str, Any])
async def list_agents(status: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       limit: int = 100) -> Dict[str, Any]:
    """
    列出所有Agent流程
    """
    results = []
    
    for flow_id, flow_data in _flow_store.items():
        # 过滤
        if status and flow_data.get("status") != status:
            continue
        if tags:
            flow_tags = set(flow_data.get("tags", []))
            if not set(tags).intersection(flow_tags):
                continue
        
        results.append({
            "id": flow_id,
            "name": flow_data.get("name", ""),
            "description": flow_data.get("description", ""),
            "status": flow_data.get("status", "draft"),
            "tags": flow_data.get("tags", []),
            "node_count": len(flow_data.get("nodes", [])),
            "created_at": flow_data.get("created_at", ""),
            "updated_at": flow_data.get("updated_at", "")
        })
    
    return {
        "success": True,
        "data": results[:limit],
        "total": len(results)
    }
