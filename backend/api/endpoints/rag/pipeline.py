"""
RAG API端点 v2.0 Phase 3
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict
from pydantic import BaseModel
from datetime import datetime

from backend.rag.pipeline import rag_pipeline, RAGConfig, RetrieverType, RerankerType
from backend.core.auth import get_current_user

router = APIRouter()

class CreateCollectionModel(BaseModel):
    name: str
    description: Optional[str] = None

class AddDocumentsModel(BaseModel):
    documents: List[str]
    metadatas: Optional[List[Dict]] = None

class QueryModel(BaseModel):
    question: str
    top_k: int = 5
    use_reranker: bool = False
    system_prompt: Optional[str] = None

@router.post("/collections")
async def create_collection(
    request: CreateCollectionModel,
    current_user = Depends(get_current_user)
):
    """
    创建RAG文档集合
    
    v2.0 Phase 3: RAG
    """
    collection = await rag_pipeline.create_collection(
        name=request.name,
        description=request.description or ""
    )
    return collection

@router.get("/collections")
async def list_collections():
    """
    列出RAG文档集合
    
    v2.0 Phase 3: RAG
    """
    collections = await rag_pipeline.list_collections()
    return {"collections": collections}

@router.post("/collections/{collection_id}/documents")
async def add_documents(
    collection_id: str,
    request: AddDocumentsModel,
    current_user = Depends(get_current_user)
):
    """
    添加文档到集合
    
    v2.0 Phase 3: RAG
    """
    result = await rag_pipeline.add_documents(
        collection_id=collection_id,
        documents=request.documents,
        metadatas=request.metadatas
    )
    return result

@router.post("/collections/{collection_id}/query")
async def query_collection(
    collection_id: str,
    request: QueryModel
):
    """
    执行RAG查询
    
    v2.0 Phase 3: RAG
    """
    # 验证集合存在
    collections = await rag_pipeline.list_collections()
    collection_ids = [c["collection_id"] for c in collections]
    
    if collection_id not in collection_ids:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    # 配置
    config = RAGConfig(
        retriever=RetrieverType.VECTOR,
        reranker=RerankerType.CROSS_ENCODER if request.use_reranker else RerankerType.NONE,
        top_k=request.top_k,
        system_prompt=request.system_prompt or "请根据以下文档回答问题。"
    )
    
    # 查询
    result = await rag_pipeline.query(
        collection_id=collection_id,
        question=request.question,
        config=config
    )
    
    return {
        "answer": result.answer,
        "sources": result.sources,
        "metadata": result.metadata
    }

@router.get("/collections/{collection_id}")
async def get_collection(collection_id: str):
    """
    获取集合详情
    
    v2.0 Phase 3: RAG
    """
    collections = await rag_pipeline.list_collections()
    for c in collections:
        if c["collection_id"] == collection_id:
            return c
    raise HTTPException(status_code=404, detail="Collection not found")
