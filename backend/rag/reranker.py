"""
Reranker Module - RAG Enhancement

重排序模块：对检索结果进行精细化排序
"""
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class RerankStrategy(Enum):
    """重排序策略"""
    CROSS_ENCODER = "cross_encoder"    # 交叉编码器
    BERT_SCORE = "bert_score"         # BERT-Score
    MMR = "mmr"                        # 最大边际相关
    DIVERSITY = "diversity"           # 多样性排序


@dataclass
class RerankResult:
    """重排序结果"""
    doc_id: str
    chunk_id: str
    content: str
    original_score: float
    rerank_score: float
    rank: int
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "original_score": self.original_score,
            "rerank_score": self.rerank_score,
            "rank": self.rank,
            "metadata": self.metadata or {}
        }


@dataclass
class RerankConfig:
    """重排序配置"""
    strategy: RerankStrategy = RerankStrategy.CROSS_ENCODER
    top_k: int = 5                    # 返回数量
    diversity_weight: float = 0.3     # 多样性权重
    similarity_weight: float = 0.7    # 相似度权重
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class BaseReranker(ABC):
    """重排序器基类"""
    
    @abstractmethod
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RerankResult]:
        """重排序"""
        pass
    
    @abstractmethod
    def compute_score(self, query: str, document: str) -> float:
        """计算单文档分数"""
        pass


class CrossEncoderReranker(BaseReranker):
    """交叉编码器重排序器"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """加载模型"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            
        except ImportError:
            print("Warning: transformers not installed, using mock reranker")
    
    def _get_scores_batch(
        self, 
        query: str, 
        documents: List[str]
    ) -> List[float]:
        """批量计算分数"""
        if self._model is None:
            self._load_model()
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            # 编码
            inputs = self._tokenizer(
                query,
                documents,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 推理
            with torch.no_grad():
                scores = self._model(**inputs).logits.squeeze(-1)
            
            # 转为Python浮点数
            return scores.tolist()
            
        except Exception as e:
            print(f"Error in cross-encoder: {e}")
            # 返回占位分数
            return [0.5] * len(documents)
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RerankResult]:
        """交叉编码器重排序"""
        if not candidates:
            return []
        
        # 提取文档内容
        documents = [
            c.get("content", "") for c in candidates
        ]
        original_scores = [
            c.get("score", 0.5) for c in candidates
        ]
        
        # 计算新分数
        rerank_scores = self._get_scores_batch(query, documents)
        
        # 创建结果
        results = []
        for i, (candidate, orig_score, rerank_score) in enumerate(
            zip(candidates, original_scores, rerank_scores)
        ):
            results.append(RerankResult(
                doc_id=candidate.get("doc_id", f"doc_{i}"),
                chunk_id=candidate.get("chunk_id", f"chunk_{i}"),
                content=candidate.get("content", ""),
                original_score=orig_score,
                rerank_score=rerank_score,
                rank=0,
                metadata=candidate.get("metadata", {})
            ))
        
        # 按重排序分数排序
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # 更新排名
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results[:top_k]
    
    def compute_score(self, query: str, document: str) -> float:
        """计算单文档分数"""
        scores = self._get_scores_batch(query, [document])
        return scores[0] if scores else 0.0


class MMRReranker(BaseReranker):
    """最大边际相关(MMR)重排序器"""
    
    def __init__(self, lambda_param: float = 0.5):
        """
        Args:
            lambda_param: 相似度与多样性之间的权衡参数
                - 值越大，相似度权重越高
                - 值越小，多样性权重越高
        """
        self.lambda_param = lambda_param
    
    def _compute_similarity(
        self, 
        query_embedding: np.ndarray, 
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """计算余弦相似度"""
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(doc_embeddings, axis=1)
        
        query_norm = max(query_norm, 1e-10)
        doc_norms = np.maximum(doc_norms, 1e-10)
        
        return np.dot(doc_embeddings, query_norm) / (doc_norms * query_norm)
    
    def _compute_diversity(
        self, 
        doc_embeddings: np.ndarray,
        selected_indices: List[int]
    ) -> np.ndarray:
        """计算与已选择文档的多样性（最小相似度）"""
        if not selected_indices:
            return np.zeros(len(doc_embeddings))
        
        selected_embeddings = doc_embeddings[selected_indices]
        
        # 计算每个文档与已选文档的最大相似度
        similarities = np.dot(doc_embeddings, selected_embeddings.T)
        max_similarities = np.max(similarities, axis=1)
        
        # 多样性 = 1 - 最大相似度
        return 1 - max_similarities
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RerankResult]:
        """MMR重排序"""
        if not candidates:
            return []
        
        # 提取嵌入
        contents = [c.get("content", "") for c in candidates]
        doc_embeddings = self._get_embeddings(contents)
        query_embedding = self._get_embeddings([query])[0]
        
        # 计算初始相似度
        similarities = self._compute_similarity(query_embedding, doc_embeddings)
        
        # MMR选择
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        for _ in range(min(top_k, len(candidates))):
            if not remaining_indices:
                break
            
            remaining_embeddings = doc_embeddings[remaining_indices]
            
            # 计算多样性
            diversities = self._compute_diversity(
                doc_embeddings, 
                selected_indices
            )
            diversities = diversities[remaining_indices]
            
            # MMR分数
            remaining_similarities = similarities[remaining_indices]
            mmr_scores = (
                self.lambda_param * remaining_similarities +
                (1 - self.lambda_param) * diversities
            )
            
            # 选择最高MMR分数的文档
            best_idx = np.argmax(mmr_scores)
            best_position = remaining_indices[best_idx]
            
            selected_indices.append(best_position)
            remaining_indices.remove(best_position)
        
        # 创建结果
        results = []
        for rank, idx in enumerate(selected_indices):
            candidate = candidates[idx]
            results.append(RerankResult(
                doc_id=candidate.get("doc_id", f"doc_{idx}"),
                chunk_id=candidate.get("chunk_id", f"chunk_{idx}"),
                content=candidate.get("content", ""),
                original_score=candidate.get("score", 0.5),
                rerank_score=similarities[idx],
                rank=rank + 1,
                metadata=candidate.get("metadata", {})
            ))
        
        return results
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """获取嵌入"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return model.encode(texts, normalize_embeddings=True)
        except ImportError:
            np.random.seed(42)
            return np.random.randn(len(texts), 384).astype(np.float32)
    
    def compute_score(self, query: str, document: str) -> float:
        """计算分数"""
        embeddings = self._get_embeddings([query, document])
        return float(np.dot(embeddings[0], embeddings[1]))


class DiversityReranker(BaseReranker):
    """多样性重排序器"""
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RerankResult]:
        """按文档来源多样性重排序"""
        if not candidates:
            return []
        
        # 按doc_id分组
        doc_groups: Dict[str, List[int]] = {}
        for i, c in enumerate(candidates):
            doc_id = c.get("doc_id", "default")
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(i)
        
        # 选择不同来源的文档
        selected = []
        remaining = list(range(len(candidates)))
        
        for doc_id, indices in doc_groups.items():
            if not remaining:
                break
            # 选择该来源中分数最高的
            best_idx = max(indices, key=lambda i: candidates[i].get("score", 0))
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        selected.sort(key=lambda i: candidates[i].get("score", 0), reverse=True)
        
        results = []
        for rank, idx in enumerate(selected[:top_k]):
            candidate = candidates[idx]
            results.append(RerankResult(
                doc_id=candidate.get("doc_id", f"doc_{idx}"),
                chunk_id=candidate.get("chunk_id", f"chunk_{idx}"),
                content=candidate.get("content", ""),
                original_score=candidate.get("score", 0.5),
                rerank_score=candidate.get("score", 0.5),
                rank=rank + 1,
                metadata=candidate.get("metadata", {})
            ))
        
        return results
    
    def compute_score(self, query: str, document: str) -> float:
        return 0.5


class RerankerPipeline:
    """重排序管道"""
    
    def __init__(self, config: Optional[RerankConfig] = None):
        self.config = config or RerankConfig()
        self._rerankers: Dict[RerankStrategy, BaseReranker] = {}
        
        # 初始化重排序器
        self._init_rerankers()
    
    def _init_rerankers(self):
        """初始化重排序器"""
        self._rerankers = {
            RerankStrategy.CROSS_ENCODER: CrossEncoderReranker(
                self.config.model_name
            ),
            RerankStrategy.MMR: MMRReranker(),
            RerankStrategy.DIVERSITY: DiversityReranker()
        }
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """执行重排序"""
        top_k = top_k or self.config.top_k
        
        reranker = self._rerankers.get(
            self.config.strategy,
            self._rerankers[RerankStrategy.CROSS_ENCODER]
        )
        
        return reranker.rerank(query, candidates, top_k)
    
    def rerank_with_diversity(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        diversity_weight: float = 0.3
    ) -> List[RerankResult]:
        """带多样性的重排序"""
        # 第一步：交叉编码器排序
        cross_reranker = self._rerankers[RerankStrategy.CROSS_ENCODER]
        initial_results = cross_reranker.rerank(query, candidates, top_k * 2)
        
        # 第二步：MMR优化
        mmr_reranker = self._rerankers[RerankStrategy.MMR]
        mmr_reranker.lambda_param = 1 - diversity_weight
        
        reranked = mmr_reranker.rerank(
            query,
            [r.to_dict() for r in initial_results],
            top_k
        )
        
        return reranked


def create_reranker(
    strategy: str = "cross_encoder",
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
) -> BaseReranker:
    """创建重排序器"""
    strategy_map = {
        "cross_encoder": CrossEncoderReranker(model_name),
        "mmr": MMRReranker(),
        "diversity": DiversityReranker()
    }
    return strategy_map.get(strategy, strategy_map["cross_encoder"])
