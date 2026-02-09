"""
Hybrid Search Retriever - RAG Enhancement

混合检索实现：结合稠密检索(Dense)和稀疏检索(Sparse)
"""
import os
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class SearchType(Enum):
    """搜索类型"""
    DENSE = "dense"          # 稠密检索（向量相似度）
    SPARSE = "sparse"        # 稀疏检索（BM25/关键词）
    HYBRID = "hybrid"        # 混合检索


@dataclass
class SearchResult:
    """搜索结果"""
    doc_id: str
    chunk_id: str
    content: str
    score: float
    search_type: SearchType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "search_type": self.search_type.value,
            "metadata": self.metadata
        }


@dataclass
class HybridSearchConfig:
    """混合检索配置"""
    search_type: SearchType = SearchType.HYBRID
    dense_weight: float = 0.6        # 稠密检索权重
    sparse_weight: float = 0.4       # 稀疏检索权重
    top_k_dense: int = 20            # 稠密检索返回数
    top_k_sparse: int = 20           # 稀疏检索返回数
    top_k_final: int = 10            # 最终返回数
    similarity_threshold: float = 0.3
    rerank: bool = True              # 是否使用重排序


class BaseRetriever(ABC):
    """检索器基类"""
    
    @abstractmethod
    def index(self, documents: List[Dict[str, Any]], collection_id: str):
        """建立索引"""
        pass
    
    @abstractmethod
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """搜索"""
        pass
    
    @abstractmethod
    def delete(self, doc_ids: List[str]):
        """删除文档"""
        pass
    
    @abstractmethod
    def update_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        """更新元数据"""
        pass


class DenseRetriever(BaseRetriever):
    """稠密检索器（基于向量相似度）"""
    
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True
    ):
        self.embedding_model = embedding_model
        self.device = device
        self.normalize = normalize
        self._index = {}  # doc_id -> (content, embedding, metadata)
        self._doc_embeddings: Dict[str, np.ndarray] = {}
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """获取文本嵌入"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer(self.embedding_model)
            embeddings = model.encode(
                texts, 
                device=self.device,
                normalize_embeddings=self.normalize
            )
            return embeddings
            
        except ImportError:
            # 占位实现：随机嵌入
            dim = 384  # all-MiniLM-L6-v2 维度
            np.random.seed(42)
            return np.random.randn(len(texts), dim).astype(np.float32)
    
    def _cosine_similarity(
        self, 
        query_embedding: np.ndarray, 
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """计算余弦相似度"""
        # 归一化
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(doc_embeddings, axis=1)
        
        # 避免除零
        query_norm = np.maximum(query_norm, 1e-10)
        doc_norms = np.maximum(doc_norms, 1e-10)
        
        # 计算相似度
        similarities = np.dot(doc_embeddings, query_embedding) / (
            doc_norms * query_norm
        )
        
        return similarities
    
    def index(
        self, 
        documents: List[Dict[str, Any]], 
        collection_id: str
    ):
        """建立向量索引"""
        texts = []
        doc_map = []
        
        for doc in documents:
            doc_id = doc.get("doc_id", f"doc_{len(doc_map)}")
            chunks = doc.get("chunks", [])
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                content = chunk.get("content", "")
                texts.append(content)
                doc_map.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "content": content,
                    "metadata": doc.get("metadata", {})
                })
        
        # 计算嵌入
        embeddings = self._get_embeddings(texts)
        
        # 存储索引
        self._index[collection_id] = {
            "documents": doc_map,
            "embeddings": embeddings
        }
        
        print(f"Indexed {len(texts)} chunks in collection {collection_id}")
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """向量检索"""
        # 计算查询嵌入
        query_embedding = self._get_embeddings([query])[0]
        
        # 搜索所有索引的集合
        all_results = []
        
        for collection_id, index_data in self._index.items():
            documents = index_data["documents"]
            embeddings = index_data["embeddings"]
            
            # 计算相似度
            similarities = self._cosine_similarity(query_embedding, embeddings)
            
            # 排序并获取top_k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for idx in top_indices:
                if similarities[idx] >= 0.0:  # 过滤负相似度
                    doc = documents[idx]
                    all_results.append(SearchResult(
                        doc_id=doc["doc_id"],
                        chunk_id=doc["chunk_id"],
                        content=doc["content"],
                        score=float(similarities[idx]),
                        search_type=SearchType.DENSE,
                        metadata={
                            "collection_id": collection_id,
                            **doc["metadata"]
                        }
                    ))
        
        # 按分数排序
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # 应用过滤
        if filters:
            filtered_results = []
            for result in all_results:
                match = True
                for key, value in filters.items():
                    if result.metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_results.append(result)
            return filtered_results
        
        return all_results
    
    def delete(self, doc_ids: List[str]):
        """删除文档"""
        # 在完整实现中需要从索引中移除
        for doc_id in doc_ids:
            self._doc_embeddings.pop(doc_id, None)
    
    def update_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        """更新元数据"""
        pass  # 在完整实现中实现


class SparseRetriever(BaseRetriever):
    """稀疏检索器（基于BM25关键词匹配）"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._index = {}  # collection_id -> {doc_id: {term: tf, ...}}
        self._documents: Dict[str, Dict[str, str]] = {}  # chunk_id -> content
        self._doc_lengths: Dict[str, float] = {}
        self._avg_doc_length: float = 0
        self._num_docs: int = 0
        self._idf: Dict[str, float] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        import re
        # 转小写，提取词
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        # 过滤停用词
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'again', 'further', 'then', 'once'}
        return [w for w in words if w not in stopwords and len(w) > 1]
    
    def _compute_idf(self, terms: List[str]) -> Dict[str, float]:
        """计算IDF"""
        import math
        
        idf = {}
        n = self._num_docs
        for term in set(terms):
            # 包含term的文档数
            df = sum(1 for doc_terms in self._index.values() if term in doc_terms)
            idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)
        
        return idf
    
    def index(
        self, 
        documents: List[Dict[str, Any]], 
        collection_id: str
    ):
        """建立BM25索引"""
        index_data = {}
        all_terms = []
        
        for doc in documents:
            doc_id = doc.get("doc_id", f"doc_{len(index_data)}")
            chunks = doc.get("chunks", [])
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                content = chunk.get("content", "")
                terms = self._tokenize(content)
                
                # TF
                tf = {}
                for term in terms:
                    tf[term] = tf.get(term, 0) + 1
                
                index_data[chunk_id] = {
                    "doc_id": doc_id,
                    "content": content,
                    "terms": terms,
                    "tf": tf,
                    "metadata": doc.get("metadata", {})
                }
                
                all_terms.extend(terms)
        
        # 更新索引
        if collection_id not in self._index:
            self._index[collection_id] = {}
        
        self._index[collection_id].update(index_data)
        
        # 更新文档统计
        for chunk_id, data in index_data.items():
            self._documents[chunk_id] = {"content": data["content"]}
            self._doc_lengths[chunk_id] = len(data["terms"])
        
        # 更新全局统计
        self._num_docs = len(self._documents)
        self._avg_doc_length = (
            sum(self._doc_lengths.values()) / self._num_docs 
            if self._num_docs > 0 else 0
        )
        
        # 预计算IDF
        self._idf = self._compute_idf(all_terms)
        
        print(f"Indexed {len(index_data)} chunks (BM25) in collection {collection_id}")
    
    def _bm25_score(
        self, 
        query_terms: List[str], 
        chunk_id: str,
        collection_id: str
    ) -> float:
        """计算BM25分数"""
        score = 0.0
        doc_length = self._doc_lengths.get(chunk_id, 0)
        doc_tf = self._index.get(collection_id, {}).get(chunk_id, {}).get("tf", {})
        
        for term in query_terms:
            tf = doc_tf.get(term, 0)
            
            # IDF
            idf = self._idf.get(term, 0)
            
            # BM25公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self._avg_doc_length)
            )
            
            score += idf * numerator / denominator
        
        return score
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """BM25检索"""
        query_terms = self._tokenize(query)
        
        all_results = []
        
        for collection_id, index_data in self._index.items():
            for chunk_id, data in index_data.items():
                score = self._bm25_score(query_terms, chunk_id, collection_id)
                
                if score > 0:
                    all_results.append(SearchResult(
                        doc_id=data["doc_id"],
                        chunk_id=chunk_id,
                        content=data["content"],
                        score=score,
                        search_type=SearchType.SPARSE,
                        metadata={
                            "collection_id": collection_id,
                            **data["metadata"]
                        }
                    ))
        
        # 归一化分数
        if all_results:
            max_score = max(r.score for r in all_results)
            if max_score > 0:
                for r in all_results:
                    r.score /= max_score
        
        # 排序
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # 过滤并返回top_k
        if filters:
            filtered = [r for r in all_results if self._matches_filters(r, filters)]
            return filtered[:top_k]
        
        return all_results[:top_k]
    
    def _matches_filters(
        self, 
        result: SearchResult, 
        filters: Dict[str, Any]
    ) -> bool:
        """检查是否匹配过滤条件"""
        for key, value in filters.items():
            if result.metadata.get(key) != value:
                return False
        return True
    
    def delete(self, doc_ids: List[str]):
        """删除文档"""
        chunks_to_delete = [
            cid for cid, data in self._index.items()
            if data["doc_id"] in doc_ids
        ]
        
        for chunk_id in chunks_to_delete:
            self._index.pop(chunk_id, None)
            self._documents.pop(chunk_id, None)
            self._doc_lengths.pop(chunk_id, None)
        
        self._num_docs = len(self._documents)
    
    def update_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        """更新元数据"""
        pass


class HybridRetriever:
    """混合检索器"""
    
    def __init__(self, config: Optional[HybridSearchConfig] = None):
        self.config = config or HybridSearchConfig()
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self._indexed = False
    
    def index(
        self, 
        documents: List[Dict[str, Any]], 
        collection_id: str
    ):
        """建立混合索引"""
        self.dense_retriever.index(documents, collection_id)
        self.sparse_retriever.index(documents, collection_id)
        self._indexed = True
        print(f"Built hybrid index for collection {collection_id}")
    
    def search(
        self, 
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """混合检索"""
        if not self._indexed:
            return []
        
        # 并行执行两种检索
        dense_results = self.dense_retriever.search(
            query, 
            top_k=self.config.top_k_dense,
            filters=filters
        )
        
        sparse_results = self.sparse_retriever.search(
            query,
            top_k=self.config.top_k_sparse,
            filters=filters
        )
        
        # 合并结果
        merged_scores: Dict[str, Tuple[float, SearchResult]] = {}
        
        # 添加稠密检索结果
        for result in dense_results:
            key = f"{result.doc_id}:{result.chunk_id}"
            merged_scores[key] = (
                result.score * self.config.dense_weight,
                result
            )
        
        # 添加稀疏检索结果并加权
        for result in sparse_results:
            key = f"{result.doc_id}:{result.chunk_id}"
            if key in merged_scores:
                # 已存在，累加分数
                old_score, old_result = merged_scores[key]
                merged_scores[key] = (
                    old_score + result.score * self.config.sparse_weight,
                    old_result
                )
            else:
                merged_scores[key] = (
                    result.score * self.config.sparse_weight,
                    result
                )
        
        # 重新排序
        final_results = sorted(
            merged_scores.values(),
            key=lambda x: x[0],
            reverse=True
        )[:self.config.top_k_final]
        
        # 更新分数为综合分数
        results = []
        for score, result in final_results:
            result.score = score
            result.search_type = SearchType.HYBRID
            results.append(result)
        
        return results
    
    def delete(self, doc_ids: List[str]):
        """删除文档"""
        self.dense_retriever.delete(doc_ids)
        self.sparse_retriever.delete(doc_ids)
    
    def update_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        """更新元数据"""
        self.dense_retriever.update_metadata(doc_id, metadata)
        self.sparse_retriever.update_metadata(doc_id, metadata)


# 创建混合检索器实例的工厂函数
def create_hybrid_retriever(
    search_type: str = "hybrid",
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4
) -> HybridRetriever:
    """创建混合检索器"""
    config = HybridSearchConfig(
        search_type=SearchType(search_type),
        dense_weight=dense_weight,
        sparse_weight=sparse_weight
    )
    return HybridRetriever(config)
