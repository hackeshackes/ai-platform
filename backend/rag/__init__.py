"""
RAG Module - Enhanced Knowledge Base Retrieval

增强RAG功能模块：
- 混合检索 (Hybrid Search)
- 重排序 (Reranker)
- 多格式文档解析 (Document Parser)
"""
from .enhanced_retriever import (
    HybridRetriever,
    SearchType,
    HybridSearchConfig,
    DenseRetriever,
    SparseRetriever,
    create_hybrid_retriever,
    SearchResult
)
from .reranker import (
    RerankerPipeline,
    RerankConfig,
    RerankStrategy,
    CrossEncoderReranker,
    MMRReranker,
    DiversityReranker,
    create_reranker,
    RerankResult
)
from .document_parser import (
    DocumentParser,
    DocumentFormat,
    MarkdownParser,
    PDFParser,
    WordParser,
    TextParser,
    ParsedDocument
)

__all__ = [
    # Enhanced Retriever
    "HybridRetriever",
    "SearchType",
    "HybridSearchConfig",
    "DenseRetriever",
    "SparseRetriever",
    "create_hybrid_retriever",
    "SearchResult",
    
    # Reranker
    "RerankerPipeline",
    "RerankConfig",
    "RerankStrategy",
    "CrossEncoderReranker",
    "MMRReranker",
    "DiversityReranker",
    "create_reranker",
    "RerankResult",
    
    # Document Parser
    "DocumentParser",
    "DocumentFormat",
    "MarkdownParser",
    "PDFParser",
    "WordParser",
    "TextParser",
    "ParsedDocument"
]
