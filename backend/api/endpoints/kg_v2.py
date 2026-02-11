"""
知识图谱v2 API端点

提供知识图谱2.0的RESTful API接口。
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ReasoningType(str, Enum):
    """推理类型"""
    RULE_BASED = "rule"
    NEURAL = "neural"
    HYBRID = "hybrid"
    PATH_FINDING = "path"


# ============ 请求/响应模型 ============

class KGEntityV2Request(BaseModel):
    """添加实体请求"""
    name: str = Field(..., description="实体名称")
    type: str = Field(..., description="实体类型 (Person, Organization, Concept...)")
    properties: Optional[Dict] = Field(default=None, description="实体属性")
    embeddings: Optional[List[float]] = Field(default=None, description="向量嵌入")


class KGEntityV2Response(BaseModel):
    """实体响应"""
    id: str
    name: str
    type: str
    properties: Dict = {}
    embeddings: Optional[List[float]] = None
    created_at: datetime
    updated_at: datetime


class KGRelationV2Request(BaseModel):
    """添加关系请求"""
    source_id: str = Field(..., description="源实体ID")
    target_id: str = Field(..., description="目标实体ID")
    relation_type: str = Field(..., description="关系类型")
    properties: Optional[Dict] = Field(default=None, description="关系属性")
    weight: float = Field(default=1.0, description="关系权重")


class KGReasoningRequest(BaseModel):
    """推理请求"""
    entity_id: Optional[str] = Field(default=None, description="实体ID")
    entity_name: Optional[str] = Field(default=None, description="实体名称")
    query: Optional[str] = Field(default=None, description="推理查询")
    reasoning_type: ReasoningType = Field(default=ReasoningType.HYBRID, description="推理类型")
    max_depth: int = Field(default=3, description="推理深度")


class KGSemanticSearchRequest(BaseModel):
    """语义搜索请求"""
    query: str = Field(..., description="搜索查询")
    query_vector: Optional[List[float]] = Field(default=None, description="查询向量")
    entity_type: Optional[str] = Field(default=None, description="实体类型过滤")
    top_k: int = Field(default=10, description="返回结果数量")


class HybridSearchConfigRequest(BaseModel):
    """混合搜索配置"""
    vector_weight: float = Field(default=0.5, ge=0, le=1)
    graph_weight: float = Field(default=0.5, ge=0, le=1)
    fusion_method: str = Field(default="rrf")


# ============ 创建路由 ============

router = APIRouter(prefix="/kg/v2", tags=["knowledge_graph_v2"])

# 全局实例
_graph_builder = None
_query_engine = None
_hybrid_retriever = None
_hybrid_reasoner = None


def get_graph_builder():
    """获取图构建器"""
    global _graph_builder
    if _graph_builder is None:
        from kg.v2.graphdb.networkx_builder import get_networkx_graph
        _graph_builder = get_networkx_graph()
    return _graph_builder


def get_query_engine():
    """获取查询引擎"""
    global _query_engine
    if _query_engine is None:
        from kg.v2.graphdb.query_engine import get_query_engine
        _query_engine = get_query_engine()
    return _query_engine


def get_hybrid_retriever_v2():
    """获取混合检索器"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from kg.v2.semantic.hybrid_retriever import get_hybrid_retriever
        _hybrid_retriever = get_hybrid_retriever()
        _hybrid_retriever.set_graph_builder(get_graph_builder())
    return _hybrid_retriever


def get_hybrid_reasoner_v2():
    """获取混合推理器"""
    global _hybrid_reasoner
    if _hybrid_reasoner is None:
        from kg.v2.reasoning.hybrid_reasoner import get_hybrid_reasoner
        _hybrid_reasoner = get_hybrid_reasoner()
    return _hybrid_reasoner


# ============ 实体端点 ============

@router.post("/entities", response_model=Dict)
async def create_entity(request: KGEntityV2Request):
    """
    创建实体
    
    - **name**: 实体名称
    - **type**: 实体类型
    - **properties**: 实体属性 (可选)
    - **embeddings**: 向量嵌入 (可选)
    """
    graph = get_graph_builder()
    
    entity = graph.add_entity(
        name=request.name,
        entity_type=request.type,
        properties=request.properties,
        embeddings=request.embeddings
    )
    
    return {
        "success": True,
        "entity": entity.to_dict()
    }


@router.get("/entities", response_model=List[Dict])
async def list_entities(
    entity_type: Optional[str] = Query(default=None, description="实体类型过滤"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """列出实体"""
    graph = get_graph_builder()
    
    entities = graph.list_entities(
        entity_type=entity_type,
        limit=limit,
        offset=offset
    )
    
    return [e.to_dict() for e in entities]


@router.get("/entities/{entity_id}", response_model=Dict)
async def get_entity(entity_id: str):
    """获取实体详情"""
    graph = get_graph_builder()
    
    entity = graph.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="实体不存在")
    
    return entity.to_dict()


@router.put("/entities/{entity_id}", response_model=Dict)
async def update_entity(
    entity_id: str,
    request: KGEntityV2Request
):
    """更新实体"""
    graph = get_graph_builder()
    
    entity = graph.update_entity(
        entity_id=entity_id,
        name=request.name,
        properties=request.properties,
        embeddings=request.embeddings
    )
    
    if not entity:
        raise HTTPException(status_code=404, detail="实体不存在")
    
    return {
        "success": True,
        "entity": entity.to_dict()
    }


@router.delete("/entities/{entity_id}", response_model=Dict)
async def delete_entity(entity_id: str):
    """删除实体"""
    graph = get_graph_builder()
    
    success = graph.delete_entity(entity_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="实体不存在")
    
    return {"success": True}


@router.get("/entities/{entity_id}/neighbors", response_model=List[Dict])
async def get_entity_neighbors(
    entity_id: str,
    relation_type: Optional[str] = None
):
    """获取实体的邻居"""
    graph = get_graph_builder()
    
    neighbors = graph.get_neighbors(entity_id, relation_type)
    
    return [
        {
            "entity": neighbor.to_dict(),
            "relation": relation.to_dict()
        }
        for neighbor, relation in neighbors
    ]


# ============ 关系端点 ============

@router.post("/relations", response_model=Dict)
async def create_relation(request: KGRelationV2Request):
    """
    创建关系
    
    - **source_id**: 源实体ID
    - **target_id**: 目标实体ID
    - **relation_type**: 关系类型
    - **properties**: 关系属性 (可选)
    - **weight**: 关系权重 (默认1.0)
    """
    graph = get_graph_builder()
    
    relation = graph.add_relation(
        source_id=request.source_id,
        target_id=request.target_id,
        relation_type=request.relation_type,
        properties=request.properties,
        weight=request.weight
    )
    
    if not relation:
        raise HTTPException(status_code=400, detail="无法创建关系，请检查实体ID是否存在")
    
    return {
        "success": True,
        "relation": relation.to_dict()
    }


@router.get("/relations", response_model=List[Dict])
async def list_relations(
    source_id: Optional[str] = Query(default=None),
    target_id: Optional[str] = Query(default=None),
    relation_type: Optional[str] = Query(default=None)
):
    """列出关系"""
    graph = get_graph_builder()
    
    relations = graph.get_relations(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type
    )
    
    return [r.to_dict() for r in relations]


@router.delete("/relations/{relation_id}", response_model=Dict)
async def delete_relation(relation_id: str):
    """删除关系"""
    graph = get_graph_builder()
    
    success = graph.delete_relation(relation_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="关系不存在")
    
    return {"success": True}


# ============ 推理端点 ============

@router.post("/reasoning", response_model=List[Dict])
async def knowledge_reasoning(request: KGReasoningRequest):
    """
    知识推理
    
    - **rule**: 规则推理
    - **neural**: 神经网络推理
    - **hybrid**: 混合推理
    - **path**: 路径查找
    """
    graph = get_graph_builder()
    reasoner = get_hybrid_reasoner_v2()
    
    results = []
    
    if request.query:
        # 基于查询推理
        results = reasoner.reason_with_query(request.query, graph, request.reasoning_type)
    elif request.entity_id:
        # 基于实体推理
        results = reasoner.reason(request.entity_id, graph, request.reasoning_type)
    elif request.entity_name:
        # 根据名称查找实体
        entity = graph.get_entity_by_name(request.entity_name)
        if entity:
            results = reasoner.reason(entity.id, graph, request.reasoning_type)
    
    return [r.to_dict() for r in results]


# ============ 语义搜索端点 ============

@router.get("/semantic-search", response_model=List[Dict])
async def semantic_search(
    q: str = Query(..., description="搜索查询"),
    entity_type: Optional[str] = Query(default=None, description="实体类型过滤"),
    top_k: int = Query(default=10, ge=1, le=100)
):
    """
    语义搜索
    
    支持关键词和向量混合搜索，性能要求 < 500ms
    """
    retriever = get_hybrid_retriever_v2()
    
    results = retriever.search(
        query=q,
        entity_type=entity_type,
        top_k=top_k
    )
    
    return [r.to_dict() for r in results]


@router.post("/semantic-search", response_model=List[Dict])
async def semantic_search_post(request: KGSemanticSearchRequest):
    """语义搜索 (POST)"""
    retriever = get_hybrid_retriever_v2()
    
    results = retriever.search(
        query=request.query,
        query_vector=request.query_vector,
        entity_type=request.entity_type,
        top_k=request.top_k
    )
    
    return [r.to_dict() for r in results]


# ============ 路径查找端点 ============

@router.get("/path/{src}/{dst}", response_model=List[Dict])
async def find_path(src: str, dst: str):
    """
    路径查找
    
    查找两个实体之间的最短路径
    """
    graph = get_graph_builder()
    reasoner = get_hybrid_reasoner_v2()
    
    path_results = reasoner.find_path(src, dst, graph)
    
    return [r.to_dict() for r in path_results]


# ============ 可视化端点 ============

@router.post("/visualize", response_model=Dict)
async def visualize_graph(
    entity_ids: List[str] = None,
    depth: int = Query(default=1, ge=1, le=3),
    limit: int = Query(default=100, ge=1, le=500)
):
    """
    图谱可视化
    
    返回可被前端可视化库使用的数据格式
    """
    graph = get_graph_builder()
    
    if entity_ids:
        subgraph = graph.get_subgraph(entity_ids, depth)
        return subgraph.export_graph()
    else:
        # 返回整个图
        entities = graph.list_entities(limit=limit)
        relations = graph.get_relations()
        
        nodes = [e.to_dict() for e in entities]
        edges = [r.to_dict() for r in relations]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": graph.stats()
        }


# ============ 统计端点 ============

@router.get("/stats", response_model=Dict)
async def get_stats():
    """获取图谱统计信息"""
    graph = get_graph_builder()
    return graph.stats()
