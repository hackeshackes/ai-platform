"""
知识图谱模块 - Knowledge Graph + RAG

核心组件:
- manager: 知识图谱管理器
- ner: 实体识别
- retriever: 混合检索
- reasoner: 知识推理
"""

from kg.manager import (
    KnowledgeGraphManager,
    Entity,
    Relation,
    get_graph_manager,
    list_graphs,
    delete_graph
)

from kg.retriever import (
    HybridRetriever,
    VectorStore,
    HybridSearchConfig,
    SearchResult,
    get_hybrid_retriever,
    hybrid_search
)

from kg.reasoner import (
    KnowledgeReasoner,
    InferenceRule,
    InferenceResult,
    get_reasoner
)

from kg.ner import (
    NERPipeline,
    EntityMention,
    EntityType,
    extract_entities,
    batch_extract_entities,
    get_ner_pipeline
)

__all__ = [
    # Manager
    "KnowledgeGraphManager",
    "Entity",
    "Relation",
    "get_graph_manager",
    "list_graphs",
    "delete_graph",
    
    # Retriever
    "HybridRetriever",
    "VectorStore",
    "HybridSearchConfig",
    "SearchResult",
    "get_hybrid_retriever",
    "hybrid_search",
    
    # Reasoner
    "KnowledgeReasoner",
    "InferenceRule",
    "InferenceResult",
    "get_reasoner",
    
    # NER
    "NERPipeline",
    "EntityMention",
    "EntityType",
    "extract_entities",
    "batch_extract_entities",
    "get_ner_pipeline"
]
