"""
RAG流水线 - Phase 3
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from uuid import uuid4
import re

from backend.rag.vectorstore import vector_store, VectorStore

class RetrieverType(Enum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"

class RerankerType(Enum):
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"
    MONO_T5 = "mono_t5"

@dataclass
class RAGConfig:
    """RAG配置"""
    retriever: RetrieverType = RetrieverType.VECTOR
    reranker: RerankerType = RerankerType.NONE
    top_k: int = 5
    rerank_top_k: int = 10
    chunk_size: int = 1000
    chunk_overlap: int = 200
    system_prompt: str = "请根据以下文档回答问题。"

@dataclass
class RAGResponse:
    """RAG响应"""
    answer: str
    sources: List[Dict]
    metadata: Dict

class RAGPipeline:
    """RAG流水线"""
    
    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or vector_store
        self.collections: Dict[str, Dict] = {}
    
    async def create_collection(
        self,
        name: str,
        description: str = ""
    ) -> Dict:
        """创建文档集合"""
        collection_id = str(uuid4())
        
        collection = {
            "collection_id": collection_id,
            "name": name,
            "description": description,
            "document_count": 0,
            "chunk_count": 0,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.collections[collection_id] = collection
        return collection
    
    async def add_documents(
        self,
        collection_id: str,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> Dict:
        """添加文档"""
        collection = self.collections.get(collection_id)
        if not collection:
            raise ValueError(f"Collection {collection_id} not found")
        
        # 分块
        chunks = self._chunk_documents(documents, chunk_size=1000, overlap=200)
        
        # 添加到向量存储
        chunk_ids = await self.vector_store.add_texts(
            texts=chunks,
            metadatas=[{"collection_id": collection_id}] * len(chunks)
        )
        
        # 更新集合
        collection["document_count"] += len(documents)
        collection["chunk_count"] += len(chunks)
        
        return {
            "documents_added": len(documents),
            "chunks_created": len(chunks),
            "collection_id": collection_id
        }
    
    def _chunk_documents(
        self,
        documents: List[str],
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """文档分块"""
        chunks = []
        for doc in documents:
            # 简单分块
            words = doc.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)
        return chunks
    
    async def query(
        self,
        collection_id: str,
        question: str,
        config: RAGConfig = None
    ) -> RAGResponse:
        """执行RAG查询"""
        config = config or RAGConfig()
        
        # 检索
        results = await self.vector_store.search(query, k=config.top_k)
        
        # 重排序
        if config.reranker != RerankerType.NONE:
            results = results[:config.rerank_top_k]
        
        # 生成答案 (简化)
        context = "\n".join([r["text"] for r in results])
        answer = f"基于以下上下文回答: {question}\n\n上下文: {context[:500]}..."
        
        return RAGResponse(
            answer=answer,
            sources=[{"chunk_id": r["chunk_id"], "score": r["score"]} for r in results],
            metadata={"collection_id": collection_id, "retrieved_chunks": len(results)}
        )
    
    async def list_collections(self) -> List[Dict]:
        """列出集合"""
        return list(self.collections.values())

# RAG流水线实例
rag_pipeline = RAGPipeline()
