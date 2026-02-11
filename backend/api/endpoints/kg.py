"""
知识图谱API端点
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from datetime import datetime

from kg.manager import (
    KnowledgeGraphManager,
    get_graph_manager,
    list_graphs,
    delete_graph
)
from kg.ner import (
    extract_entities,
    batch_extract_entities,
    get_ner_pipeline,
    EntityMention
)
from kg.retriever import (
    hybrid_search,
    get_hybrid_retriever,
    HybridSearchConfig,
    SearchResult
)
from kg.reasoner import (
    get_reasoner,
    KnowledgeReasoner,
    InferenceResult
)


# 创建路由
router = APIRouter(prefix="/kg", tags=["knowledge_graph"])

# ============ 请求/响应模型 ============

class AddEntityRequest(BaseModel):
    """添加实体请求"""
    name: str = Field(..., description="实体名称")
    entity_type: str = Field(..., description="实体类型")
    properties: Optional[Dict] = Field(default=None, description="实体属性")
    embeddings: Optional[List[float]] = Field(default=None, description="向量嵌入")


class UpdateEntityRequest(BaseModel):
    """更新实体请求"""
    name: Optional[str] = Field(default=None, description="实体名称")
    properties: Optional[Dict] = Field(default=None, description="实体属性")


class AddRelationRequest(BaseModel):
    """添加关系请求"""
    source_id: str = Field(..., description="源实体ID")
    target_id: str = Field(..., description="目标实体ID")
    relation_type: str = Field(..., description="关系类型")
    properties: Optional[Dict] = Field(default=None, description="关系属性")
    weight: float = Field(default=1.0, description="关系权重")


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., description="搜索查询")
    query_vector: Optional[List[float]] = Field(default=None, description="查询向量")
    entity_type: Optional[str] = Field(default=None, description="实体类型过滤")
    vector_weight: float = Field(default=0.5, description="向量检索权重")
    graph_weight: float = Field(default=0.5, description="图谱检索权重")
    top_k: int = Field(default=10, description="返回结果数量")


class ReasonRequest(BaseModel):
    """推理请求"""
    entity_id: Optional[str] = Field(default=None, description="实体ID")
    entity_name: Optional[str] = Field(default=None, description="实体名称")
    query: Optional[str] = Field(default=None, description="推理查询")
    max_depth: int = Field(default=3, description="推理深度")


class ExtractEntitiesRequest(BaseModel):
    """实体识别请求"""
    text: str = Field(..., description="输入文本")
    entity_types: Optional[List[str]] = Field(default=None, description="实体类型过滤")


class GraphExportRequest(BaseModel):
    """图谱导出/导入请求"""
    merge: bool = Field(default=False, description="是否合并到现有图谱")


# ============ 实体端点 ============

@router.post("/entities", response_model=Dict)
async def add_entity(request: AddEntityRequest, graph_id: str = "default"):
    """
    添加实体
    
    - **name**: 实体名称
    - **entity_type**: 实体类型 (PERSON, ORGANIZATION, LOCATION, etc.)
    - **properties**: 实体属性 (可选)
    - **embeddings**: 向量嵌入 (可选)
    """
    graph_manager = get_graph_manager(graph_id)
    
    entity = graph_manager.add_entity(
        name=request.name,
        entity_type=request.entity_type,
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
    offset: int = Query(default=0, ge=0),
    graph_id: str = "default"
):
    """列出实体"""
    graph_manager = get_graph_manager(graph_id)
    
    entities = graph_manager.list_entities(
        entity_type=entity_type,
        limit=limit,
        offset=offset
    )
    
    return [e.to_dict() for e in entities]


@router.get("/entities/{entity_id}", response_model=Dict)
async def get_entity(entity_id: str, graph_id: str = "default"):
    """获取实体详情"""
    graph_manager = get_graph_manager(graph_id)
    
    entity = graph_manager.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="实体不存在")
    
    return entity.to_dict()


@router.put("/entities/{entity_id}", response_model=Dict)
async def update_entity(
    entity_id: str,
    request: UpdateEntityRequest,
    graph_id: str = "default"
):
    """更新实体"""
    graph_manager = get_graph_manager(graph_id)
    
    entity = graph_manager.update_entity(
        entity_id=entity_id,
        name=request.name,
        properties=request.properties
    )
    
    if not entity:
        raise HTTPException(status_code=404, detail="实体不存在")
    
    return {
        "success": True,
        "entity": entity.to_dict()
    }


@router.delete("/entities/{entity_id}", response_model=Dict)
async def delete_entity(entity_id: str, graph_id: str = "default"):
    """删除实体"""
    graph_manager = get_graph_manager(graph_id)
    
    success = graph_manager.delete_entity(entity_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="实体不存在")
    
    return {"success": True}


@router.get("/entities/{entity_id}/neighbors", response_model=List[Dict])
async def get_entity_neighbors(
    entity_id: str,
    relation_type: Optional[str] = None,
    graph_id: str = "default"
):
    """获取实体的邻居"""
    graph_manager = get_graph_manager(graph_id)
    
    neighbors = graph_manager.get_neighbors(entity_id, relation_type)
    
    return [
        {
            "entity": neighbor.to_dict(),
            "relation": relation.to_dict()
        }
        for neighbor, relation in neighbors
    ]


# ============ 关系端点 ============

@router.post("/relations", response_model=Dict)
async def add_relation(request: AddRelationRequest, graph_id: str = "default"):
    """
    添加关系
    
    - **source_id**: 源实体ID
    - **target_id**: 目标实体ID
    - **relation_type**: 关系类型 (如 "WORKS_AT", "LOCATED_IN", "IS_A" 等)
    - **properties**: 关系属性 (可选)
    - **weight**: 关系权重 (默认1.0)
    """
    graph_manager = get_graph_manager(graph_id)
    
    relation = graph_manager.add_relation(
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
    source_id: Optional[str] = Query(default=None, description="源实体ID"),
    target_id: Optional[str] = Query(default=None, description="目标实体ID"),
    relation_type: Optional[str] = Query(default=None, description="关系类型"),
    graph_id: str = "default"
):
    """列出关系"""
    graph_manager = get_graph_manager(graph_id)
    
    relations = graph_manager.get_relations(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type
    )
    
    return [r.to_dict() for r in relations]


@router.delete("/relations/{relation_id}", response_model=Dict)
async def delete_relation(relation_id: str, graph_id: str = "default"):
    """删除关系"""
    graph_manager = get_graph_manager(graph_id)
    
    success = graph_manager.delete_relation(relation_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="关系不存在")
    
    return {"success": True}


# ============ 搜索端点 ============

@router.get("/search", response_model=List[Dict])
async def hybrid_search_endpoint(
    q: str = Query(..., description="搜索查询"),
    entity_type: Optional[str] = Query(default=None, description="实体类型过滤"),
    vector_weight: float = Query(default=0.5, ge=0, le=1),
    graph_weight: float = Query(default=0.5, ge=0, le=1),
    top_k: int = Query(default=10, ge=1, le=100),
    graph_id: str = "default"
):
    """
    混合搜索
    
    结合向量检索和图谱检索，返回融合结果。
    
    - **RRF融合**: 默认使用Reciprocal Rank Fusion
    - **权重可调**: 可以调整向量和图谱的权重比例
    """
    graph_manager = get_graph_manager(graph_id)
    retriever = get_hybrid_retriever()
    retriever.set_graph_manager(graph_manager)
    
    # 配置
    config = HybridSearchConfig(
        vector_weight=vector_weight,
        graph_weight=graph_weight,
        vector_top_k=top_k,
        graph_top_k=top_k,
        fusion_method="rrf"
    )
    
    # 执行搜索
    results = hybrid_search(
        query=q,
        entity_type=entity_type,
        config=config
    )
    
    return [r.to_dict() for r in results]


@router.post("/search", response_model=List[Dict])
async def hybrid_search_post(request: SearchRequest, graph_id: str = "default"):
    """混合搜索 (POST)"""
    graph_manager = get_graph_manager(graph_id)
    retriever = get_hybrid_retriever()
    retriever.set_graph_manager(graph_manager)
    
    config = HybridSearchConfig(
        vector_weight=request.vector_weight,
        graph_weight=request.graph_weight,
        vector_top_k=request.top_k,
        graph_top_k=request.top_k
    )
    
    results = hybrid_search(
        query=request.query,
        query_vector=request.query_vector,
        entity_type=request.entity_type,
        config=config
    )
    
    return [r.to_dict() for r in results]


# ============ 推理端点 ============

@router.post("/reason", response_model=List[Dict])
async def reason(request: ReasonRequest, graph_id: str = "default"):
    """
    知识推理
    
    基于规则和图结构进行推理：
    
    1. **实体推理**: 基于实体ID或名称推理
    2. **查询推理**: 基于自然语言查询推理
    3. **规则推理**: 应用预定义的传递性、逆关系等规则
    """
    graph_manager = get_graph_manager(graph_id)
    reasoner = get_reasoner(graph_manager)
    
    results = []
    
    if request.query:
        # 基于查询推理
        results = reasoner.reason_with_query(request.query)
    else:
        # 基于实体推理
        results = reasoner.reason(
            entity_id=request.entity_id,
            entity_name=request.entity_name,
            max_depth=request.max_depth
        )
    
    return [r.to_dict() for r in results]


@router.get("/reason/rules", response_model=List[Dict])
async def get_reasoning_rules(graph_id: str = "default"):
    """获取推理规则列表"""
    reasoner = get_reasoner()
    return reasoner.export_rules()


# ============ 可视化端点 ============

@router.get("/visualize", response_model=Dict)
async def visualize_graph(
    entity_types: Optional[str] = Query(default=None, description="实体类型过滤(逗号分隔)"),
    depth: int = Query(default=1, ge=1, le=3, description="邻居深度"),
    limit: int = Query(default=100, ge=1, le=500, description="实体数量限制"),
    graph_id: str = "default"
):
    """
    图谱可视化
    
    返回可被前端可视化库(如D3.js, ECharts)使用的数据格式。
    
    - **节点**: 实体
    - **边**: 关系
    - 支持按类型和深度过滤
    """
    graph_manager = get_graph_manager(graph_id)
    
    # 获取所有实体
    entities = graph_manager.list_entities(limit=limit)
    
    # 按类型过滤
    if entity_types:
        type_list = [t.strip() for t in entity_types.split(",")]
        entities = [e for e in entities if e.type in type_list]
    
    # 构建节点和边
    nodes = []
    edges = []
    node_ids = set()
    
    for entity in entities:
        nodes.append({
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "properties": entity.properties
        })
        node_ids.add(entity.id)
    
    # 获取关系
    for relation in graph_manager.relations.values():
        if relation.source_id in node_ids and relation.target_id in node_ids:
            edges.append({
                "id": relation.id,
                "source": relation.source_id,
                "target": relation.target_id,
                "type": relation.relation_type,
                "weight": relation.weight
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": graph_manager.stats()
    }


@router.get("/visualize/entity/{entity_id}", response_model=Dict)
async def visualize_entity_subgraph(
    entity_id: str,
    depth: int = Query(default=2, ge=1, le=3),
    graph_id: str = "default"
):
    """可视化实体的子图"""
    graph_manager = get_graph_manager(graph_id)
    
    entity = graph_manager.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="实体不存在")
    
    subgraph = graph_manager.get_subgraph([entity_id], depth=depth)
    
    return subgraph


# ============ 图谱管理端点 ============

@router.get("/graphs", response_model=List[str])
async def list_graphs():
    """列出所有图谱"""
    return list_graphs()


@router.post("/graphs", response_model=Dict)
async def create_graph(graph_id: str = "default"):
    """创建新图谱"""
    manager = get_graph_manager(graph_id)
    return {
        "success": True,
        "graph_id": manager.graph_id
    }


@router.delete("/graphs/{graph_id}", response_model=Dict)
async def delete_graph_endpoint(graph_id: str):
    """删除图谱"""
    success = delete_graph(graph_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="图谱不存在")
    
    return {"success": True}


@router.get("/stats", response_model=Dict)
async def get_graph_stats(graph_id: str = "default"):
    """获取图谱统计信息"""
    graph_manager = get_graph_manager(graph_id)
    return graph_manager.stats()


@router.get("/export", response_model=Dict)
async def export_graph(graph_id: str = "default"):
    """导出图谱"""
    graph_manager = get_graph_manager(graph_id)
    return graph_manager.export_graph()


@router.post("/import", response_model=Dict)
async def import_graph(
    graph_data: Dict,
    merge: bool = Query(default=False),
    graph_id: str = "default"
):
    """导入图谱"""
    graph_manager = get_graph_manager(graph_id)
    
    entity_count, relation_count = graph_manager.import_graph(graph_data, merge=merge)
    
    return {
        "success": True,
        "imported_entities": entity_count,
        "imported_relations": relation_count,
        "graph_id": graph_id
    }


# ============ NER端点 ============

@router.post("/extract", response_model=List[Dict])
async def extract_entities_endpoint(request: ExtractEntitiesRequest):
    """
    实体识别
    
    从文本中提取命名实体。
    """
    mentions = extract_entities(request.text, request.entity_types)
    
    return [m.to_dict() for m in mentions]


@router.post("/extract/batch", response_model=List[List[Dict]])
async def batch_extract_entities_endpoint(texts: List[str]):
    """批量实体识别"""
    results = batch_extract_entities(texts)
    return [[m.to_dict() for m in r] for r in results]
