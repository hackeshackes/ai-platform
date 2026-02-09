"""
RAG向量存储 - Phase 3
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from uuid import uuid4

class VectorStoreType(Enum):
    CHROMADB = "chromadb"
    MILVUS = "milvus"

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: Dict = None
    created_at = datetime.utcnow()

class VectorStore:
    def __init__(self, collection_name: str, embedding_dim: int = 768):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.chunks: Dict[str, DocumentChunk] = {}
    
    async def add_texts(self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None) -> List[str]:
        chunk_ids = []
        for i, text in enumerate(texts):
            chunk_id = ids[i] if ids else str(uuid4())
            metadata = metadatas[i] if metadatas else {}
            self.chunks[chunk_id] = DocumentChunk(chunk_id=chunk_id, text=text, metadata=metadata)
            chunk_ids.append(chunk_id)
        return chunk_ids
    
    async def search(self, query: str, k: int = 5) -> List[Dict]:
        results = []
        query_lower = query.lower()
        for chunk in self.chunks.values():
            score = self._similarity(query, chunk.text)
            results.append({"chunk_id": chunk.chunk_id, "text": chunk.text, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
    
    def _similarity(self, query: str, text: str) -> float:
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        if not text_words:
            return 0.0
        intersection = query_words & text_words
        union = query_words | text_words
        return len(intersection) / len(union) if union else 0.0

vector_store = VectorStore("ai-platform-docs", 768)
