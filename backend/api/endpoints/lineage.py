"""
Lineage API端点 v2.1
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

from lineage.graph import lineage_graph
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from api.endpoints.auth import get_current_user
except ImportError:
    get_current_user = None

router = APIRouter()

class CreateNodeModel(BaseModel):
    node_type: str  # dataset, run, model, artifact
    external_id: str
    name: str
    metadata: Optional[Dict[str, Any]] = None

class CreateEdgeModel(BaseModel):
    source: tuple  # (node_type, external_id)
    target: tuple  # (node_type, external_id)
    edge_type: str  # data_flow, execution_flow
    metadata: Optional[Dict[str, Any]] = None

class ConnectModel(BaseModel):
    source_type: str
    source_id: str
    target_type: str
    target_id: str
    edge_type: str
    metadata: Optional[Dict[str, Any]] = None

class TraceModelLineageModel(BaseModel):
    model_id: str

@router.post("/nodes")
async def create_node(request: CreateNodeModel):
    """
    创建血缘节点
    
    v2.1: Model Lineage
    """
    try:
        node = await lineage_graph.create_node(
            node_type=request.node_type,
            external_id=request.external_id,
            name=request.name,
            metadata=request.metadata
        )
        
        return {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "name": node.name,
            "created_at": node.created_at.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/nodes/{node_id}")
async def get_node(node_id: str):
    """
    获取血缘节点
    
    v2.1: Model Lineage
    """
    node = await lineage_graph.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    return {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "external_id": node.external_id,
        "name": node.name,
        "metadata": node.metadata,
        "created_at": node.created_at.isoformat()
    }

@router.get("/nodes/by-type/{node_type}/{external_id}")
async def get_node_by_external_id(node_type: str, external_id: str):
    """
    根据外部ID获取节点
    
    v2.1: Model Lineage
    """
    node = await lineage_graph.get_node_by_external_id(node_type, external_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    return {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "external_id": node.external_id,
        "name": node.name
    }

@router.post("/edges")
async def create_edge(request: CreateEdgeModel):
    """
    创建血缘边
    
    v2.1: Model Lineage
    """
    try:
        edge = await lineage_graph.create_edge(
            source_node_id=request.source[1],  # external_id
            target_node_id=request.target[1],  # external_id
            edge_type=request.edge_type,
            metadata=request.metadata
        )
        
        return {
            "edge_id": edge.edge_id,
            "source": edge.source_node_id,
            "target": edge.target_node_id,
            "edge_type": edge.edge_type
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/connect")
async def connect_nodes(request: ConnectModel):
    """
    连接两个节点 (自动创建节点如果不存在)
    
    v2.1: Model Lineage
    """
    try:
        edge = await lineage_graph.connect(
            source=(request.source_type, request.source_id),
            target=(request.target_type, request.target_id),
            edge_type=request.edge_type,
            metadata=request.metadata
        )
        
        return {
            "edge_id": edge.edge_id,
            "source": request.source_id,
            "target": request.target_id,
            "edge_type": edge.edge_type
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/lineage/{node_id}")
async def get_lineage(
    node_id: str,
    direction: str = "upstream",
    max_depth: int = 10
):
    """
    获取节点血缘路径
    
    v2.1: Model Lineage
    
    direction: upstream (上游), downstream (下游), both (双向)
    """
    try:
        lineage = await lineage_graph.get_lineage(
            node_id=node_id,
            direction=direction,
            max_depth=max_depth
        )
        return lineage
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/execution/{run_id}")
async def get_execution_lineage(run_id: str):
    """
    获取运行血缘
    
    v2.1: Model Lineage
    """
    try:
        lineage = await lineage_graph.get_execution_lineage(run_id)
        return lineage
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/model/trace")
async def trace_model_lineage(request: TraceModelLineageModel):
    """
    追踪模型完整血缘
    
    v2.1: Model Lineage
    """
    try:
        lineage = await lineage_graph.trace_model_lineage(request.model_id)
        return lineage
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/graph")
async def get_graph():
    """
    获取完整血缘图
    
    v2.1: Model Lineage
    """
    return await lineage_graph.get_graph()

@router.delete("/nodes/{node_id}")
async def delete_node(node_id: str):
    """
    删除血缘节点
    
    v2.1: Model Lineage
    """
    result = await lineage_graph.delete_node(node_id)
    if not result:
        raise HTTPException(status_code=404, detail="Node not found")
    
    return {"message": "Node deleted"}

# 便捷函数: 记录训练运行血缘
@router.post("/track/training")
async def track_training_run(
    run_id: str,
    dataset_ids: List[str],
    model_id: str,
    experiment_id: Optional[str] = None
):
    """
    记录训练运行血缘
    
    v2.1: Model Lineage
    
    自动创建: Dataset -> Run -> Model
    """
    # 创建Run节点
    run_node = await lineage_graph.create_node(
        node_type="run",
        external_id=run_id,
        name=f"Training Run {run_id}",
        metadata={"experiment_id": experiment_id}
    )
    
    # 连接Dataset -> Run
    for dataset_id in dataset_ids:
        await lineage_graph.connect(
            source=("dataset", dataset_id),
            target=("run", run_id),
            edge_type="execution_flow"
        )
    
    # 连接Run -> Model
    await lineage_graph.connect(
        source=("run", run_id),
        target=("model", model_id),
        edge_type="execution_flow"
    )
    
    return {
        "run_id": run_id,
        "dataset_count": len(dataset_ids),
        "model_id": model_id,
        "message": "Lineage tracked"
    }

# 便捷函数: 记录数据血缘
@router.post("/track/data")
async def track_data_flow(
    source_dataset_id: str,
    target_dataset_id: str,
    transformation: str
):
    """
    记录数据血缘
    
    v2.1: Model Lineage
    
    记录: Dataset -> Dataset (数据转换)
    """
    edge = await lineage_graph.connect(
        source=("dataset", source_dataset_id),
        target=("dataset", target_dataset_id),
        edge_type="data_flow",
        metadata={"transformation": transformation}
    )
    
    return {
        "source": source_dataset_id,
        "target": target_dataset_id,
        "transformation": transformation,
        "message": "Data lineage tracked"
    }
