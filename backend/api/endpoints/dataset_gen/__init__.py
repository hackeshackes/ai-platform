"""
Dataset Generation API Endpoints v3

数据集生成API
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

router = APIRouter(tags=["Dataset Generation"])

# 模拟数据
_generation_jobs: Dict[str, Dict] = {}
_templates: List[Dict] = []

class DatasetType(str, Enum):
    SYNTHETIC = "synthetic"
    AUGMENTATION = "augmentation"
    TRANSLATION = "translation"
    QUALITY_FILTER = "quality_filter"

class GenerationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# 初始化模板
_templates = [
    {
        "id": "qa-synthetic",
        "name": "QA合成数据",
        "type": "synthetic",
        "description": "基于种子知识生成问答对",
        "parameters": ["topic", "num_samples", "language"]
    },
    {
        "id": "code-synthetic",
        "name": "代码合成数据",
        "type": "synthetic",
        "description": "生成代码-注释对",
        "parameters": ["language", "complexity", "num_samples"]
    },
    {
        "id": "paraphrase-augment",
        "name": "改写增强",
        "type": "augmentation",
        "description": "使用LLM改写原文增强数据多样性",
        "parameters": ["augment_factor", "preserve_meaning"]
    },
    {
        "id": "back-translate",
        "name": "回译增强",
        "type": "augmentation",
        "description": "通过多语言回译增强数据",
        "parameters": ["languages", "rounds"]
    },
    {
        "id": "translate",
        "name": "多语言翻译",
        "type": "translation",
        "description": "将数据翻译成多种语言",
        "parameters": ["source_lang", "target_langs"]
    },
    {
        "id": "quality-filter",
        "name": "质量过滤",
        "type": "quality_filter",
        "description": "基于规则和模型过滤低质量数据",
        "parameters": ["rules", "min_length", "max_length"]
    }
]

# 生成任务配置
class GenerationConfig(BaseModel):
    name: str
    type: DatasetType
    source_model: str
    target_model: Optional[str] = None
    size: int = 1000
    topic: Optional[str] = None
    language: str = "zh"
    quality_filter: bool = True
    parameters: Dict[str, Any] = {}

@router.get("/generate/templates")
async def get_generation_templates():
    """获取数据集生成模板"""
    return {
        "templates": _templates,
        "total": len(_templates)
    }

@router.get("/generate/types")
async def get_generation_types():
    """获取支持的生成类型"""
    return {
        "types": [
            {"id": "synthetic", "name": "合成数据生成"},
            {"id": "augmentation", "name": "数据增强"},
            {"id": "translation", "name": "多语言翻译"},
            {"id": "quality_filter", "name": "质量过滤"}
        ]
    }

@router.post("/generate")
async def create_generation_job(config: GenerationConfig):
    """创建数据集生成任务"""
    import uuid
    
    job_id = f"gen_{uuid.uuid4().hex[:8]}"
    job = {
        "job_id": job_id,
        "name": config.name,
        "type": config.type.value,
        "source_model": config.source_model,
        "target_model": config.target_model,
        "size": config.size,
        "status": GenerationStatus.PENDING.value,
        "progress": 0.0,
        "created_at": datetime.utcnow().isoformat(),
        "parameters": config.parameters
    }
    
    _generation_jobs[job_id] = job
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "生成任务已创建"
    }

@router.get("/generate")
async def list_generation_jobs(status: Optional[str] = None):
    """列出生成任务"""
    jobs = list(_generation_jobs.values())
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    
    return {
        "total": len(jobs),
        "jobs": jobs
    }

@router.get("/generate/{job_id}")
async def get_generation_job(job_id: str):
    """获取生成任务详情"""
    job = _generation_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.delete("/generate/{job_id}")
async def delete_generation_job(job_id: str):
    """删除生成任务"""
    if job_id in _generation_jobs:
        del _generation_jobs[job_id]
        return {"message": "任务已删除"}
    raise HTTPException(status_code=404, detail="Job not found")

@router.get("/health")
async def dataset_gen_health():
    """健康检查"""
    return {
        "status": "healthy",
        "templates": len(_templates),
        "active_jobs": len([j for j in _generation_jobs.values() if j["status"] == "running"])
    }
