"""
知识图谱v2模块

图数据库适配器、推理引擎和语义增强
"""

from kg.v2.graphdb.neo4j_adapter import (
    Neo4jAdapter,
    Neo4jDriver,
    get_neo4j_adapter,
    close_neo4j_adapter
)

from kg.v2.graphdb.networkx_builder import (
    NetworkXGraphBuilder,
    KGEntityV2,
    KGRelationV2,
    get_networkx_graph,
    list_networkx_graphs,
    delete_networkx_graph
)

from kg.v2.graphdb.query_engine import (
    QueryEngine,
    GraphDBType,
    get_query_engine,
    reset_query_engine
)

from kg.v2.reasoning.rule_engine import (
    RuleEngine,
    InferenceRule,
    InferenceResult,
    ReasoningType,
    get_rule_engine
)

from kg.v2.reasoning.neural_reasoner import (
    NeuralReasoner,
    NeuralInferenceResult,
    get_neural_reasoner
)

from kg.v2.reasoning.hybrid_reasoner import (
    HybridReasoner,
    HybridInferenceResult,
    ReasoningConfig,
    get_hybrid_reasoner
)

from kg.v2.semantic.embedding_search import (
    EmbeddingSearch,
    VectorStore,
    SearchResult,
    get_embedding_search
)

from kg.v2.semantic.hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
    HybridSearchConfig,
    get_hybrid_retriever
)

__all__ = [
    # 图数据库
    "Neo4jAdapter",
    "Neo4jDriver",
    "get_neo4j_adapter",
    "close_neo4j_adapter",
    "NetworkXGraphBuilder",
    "KGEntityV2",
    "KGRelationV2",
    "get_networkx_graph",
    "list_networkx_graphs",
    "delete_networkx_graph",
    "QueryEngine",
    "GraphDBType",
    "get_query_engine",
    "reset_query_engine",
    # 推理引擎
    "RuleEngine",
    "InferenceRule",
    "InferenceResult",
    "ReasoningType",
    "get_rule_engine",
    "NeuralReasoner",
    "NeuralInferenceResult",
    "get_neural_reasoner",
    "HybridReasoner",
    "HybridInferenceResult",
    "ReasoningConfig",
    "get_hybrid_reasoner",
    # 语义增强
    "EmbeddingSearch",
    "VectorStore",
    "SearchResult",
    "get_embedding_search",
    "HybridRetriever",
    "HybridSearchResult",
    "HybridSearchConfig",
    "get_hybrid_retriever"
]
