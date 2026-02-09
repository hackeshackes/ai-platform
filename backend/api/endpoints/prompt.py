"""
Prompt Management API端点 v2.4
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

# 直接导入模块
import importlib.util
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
module_path = os.path.join(backend_dir, 'prompt/manager.py')

spec = importlib.util.spec_from_file_location("prompt_module", module_path)
module = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(module)
    prompt_manager = module.prompt_manager
    PromptType = module.PromptType
    PromptStatus = module.PromptStatus
except Exception as e:
    print(f"Failed to import prompt module: {e}")
    prompt_manager = None
    PromptType = None
    PromptStatus = None

from api.endpoints.auth import get_current_user

router = APIRouter()

# Pydantic Models
class CreateTemplateModel(BaseModel):
    name: str
    description: str
    prompt_type: str
    template: str
    parameters: Optional[List[Dict]] = None
    examples: Optional[List[Dict]] = None

class UpdateTemplateModel(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    template: Optional[str] = None
    parameters: Optional[List[Dict]] = None

class CreatePromptModel(BaseModel):
    name: str
    description: str
    template_id: str
    prompt_type: str
    tags: Optional[List[str]] = None

class UpdatePromptModel(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    tags: Optional[List[str]] = None

class CreateVersionModel(BaseModel):
    template: str
    parameters: Optional[List[Dict]] = None
    changelog: str = ""

class TestPromptModel(BaseModel):
    test_input: Dict
    expected_output: Optional[str] = None
    latency_ms: float = 0.0
    token_count: int = 0

class EvaluateResultModel(BaseModel):
    result_id: str
    score: float

@router.get("/templates")
async def list_templates(
    prompt_type: Optional[str] = None,
    created_by: Optional[str] = None
):
    """
    列出Prompt模板
    
    v2.4: Prompt Management
    """
    ptype = PromptType(prompt_type) if prompt_type else None
    
    templates = prompt_manager.list_templates(
        prompt_type=ptype,
        created_by=created_by
    )
    
    return {
        "total": len(templates),
        "templates": [
            {
                "template_id": t.template_id,
                "name": t.name,
                "description": t.description,
                "type": t.prompt_type.value,
                "parameters_count": len(t.parameters),
                "examples_count": len(t.examples),
                "created_by": t.created_by,
                "created_at": t.created_at.isoformat()
            }
            for t in templates
        ]
    }

@router.post("/templates")
async def create_template(request: CreateTemplateModel):
    """
    创建Prompt模板
    
    v2.4: Prompt Management
    """
    try:
        ptype = PromptType(request.prompt_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid prompt type: {request.prompt_type}")
    
    template = prompt_manager.create_template(
        name=request.name,
        description=request.description,
        prompt_type=ptype,
        template=request.template,
        parameters=request.parameters,
        examples=request.examples
    )
    
    return {
        "template_id": template.template_id,
        "name": template.name,
        "message": "Template created"
    }

@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """
    获取模板详情
    
    v2.4: Prompt Management
    """
    template = prompt_manager.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {
        "template_id": template.template_id,
        "name": template.name,
        "description": template.description,
        "type": template.prompt_type.value,
        "template": template.template,
        "parameters": [
            {
                "name": p.name,
                "type": p.type,
                "required": p.required,
                "default": p.default,
                "description": p.description
            }
            for p in template.parameters
        ],
        "examples": template.examples,
        "created_by": template.created_by,
        "created_at": template.created_at.isoformat()
    }

@router.put("/templates/{template_id}")
async def update_template(template_id: str, request: UpdateTemplateModel):
    """
    更新模板
    
    v2.4: Prompt Management
    """
    result = prompt_manager.update_template(
        template_id=template_id,
        name=request.name,
        description=request.description,
        template=request.template,
        parameters=request.parameters
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {"message": "Template updated"}

@router.delete("/templates/{template_id}")
async def delete_template(template_id: str):
    """
    删除模板
    
    v2.4: Prompt Management
    """
    result = prompt_manager.delete_template(template_id)
    if not result:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {"message": "Template deleted"}

@router.get("/templates/summary")
async def get_template_summary():
    """
    获取模板统计
    
    v2.4: Prompt Management
    """
    summary = prompt_manager.get_template_summary()
    return summary

@router.get("")
async def list_prompts(
    status: Optional[str] = None,
    prompt_type: Optional[str] = None,
    tags: Optional[str] = None
):
    """
    列出Prompts
    
    v2.4: Prompt Management
    """
    pstatus = PromptStatus(status) if status else None
    ptype = PromptType(prompt_type) if prompt_type else None
    tag_list = tags.split(",") if tags else None
    
    prompts = prompt_manager.list_prompts(
        status=pstatus,
        prompt_type=ptype,
        tags=tag_list
    )
    
    return {
        "total": len(prompts),
        "prompts": [
            {
                "prompt_id": p.prompt_id,
                "name": p.name,
                "description": p.description,
                "type": p.prompt_type.value,
                "status": p.status.value,
                "version": p.current_version,
                "tags": p.tags,
                "metrics": p.metrics,
                "created_by": p.created_by,
                "created_at": p.created_at.isoformat()
            }
            for p in prompts
        ]
    }

@router.get("/health")
async def prompt_health():
    """
    Prompt Management健康检查
    
    v2.4: Prompt Management
    """
    return {
        "status": "healthy",
        "templates": len(prompt_manager.templates),
        "prompts": len(prompt_manager.prompts)
    }

@router.post("")
async def create_prompt(request: CreatePromptModel):
    """
    创建Prompt
    
    v2.4: Prompt Management
    """
    try:
        ptype = PromptType(request.prompt_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid prompt type: {request.prompt_type}")
    
    try:
        prompt = prompt_manager.create_prompt(
            name=request.name,
            description=request.description,
            template_id=request.template_id,
            prompt_type=ptype,
            tags=request.tags
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "prompt_id": prompt.prompt_id,
        "name": prompt.name,
        "message": "Prompt created"
    }

@router.get("/{prompt_id}")
async def get_prompt(prompt_id: str):
    """
    获取Prompt详情
    
    v2.4: Prompt Management
    """
    prompt = prompt_manager.get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    return {
        "prompt_id": prompt.prompt_id,
        "name": prompt.name,
        "description": prompt.description,
        "template_id": prompt.template_id,
        "type": prompt.prompt_type.value,
        "status": prompt.status.value,
        "tags": prompt.tags,
        "current_version": prompt.current_version,
        "metrics": prompt.metrics,
        "versions_count": len(prompt.versions),
        "created_by": prompt.created_by,
        "created_at": prompt.created_at.isoformat()
    }

@router.put("/{prompt_id}")
async def update_prompt(prompt_id: str, request: UpdatePromptModel):
    """
    更新Prompt
    
    v2.4: Prompt Management
    """
    status = PromptStatus(request.status) if request.status else None
    
    result = prompt_manager.update_prompt(
        prompt_id=prompt_id,
        name=request.name,
        description=request.description,
        status=status,
        tags=request.tags
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    return {"message": "Prompt updated"}

@router.delete("/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """
    删除Prompt
    
    v2.4: Prompt Management
    """
    result = prompt_manager.delete_prompt(prompt_id)
    if not result:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    return {"message": "Prompt deleted"}

@router.post("/{prompt_id}/versions")
async def create_version(prompt_id: str, request: CreateVersionModel):
    """
    创建新版本
    
    v2.4: Prompt Management
    """
    version = prompt_manager.create_version(
        prompt_id=prompt_id,
        template=request.template,
        parameters=request.parameters,
        changelog=request.changelog
    )
    
    if not version:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    return {
        "version_id": version.version_id,
        "version": version.version,
        "message": "Version created"
    }

@router.get("/{prompt_id}/versions/{version}")
async def get_version(prompt_id: str, version: int):
    """
    获取特定版本
    
    v2.4: Prompt Management
    """
    prompt = prompt_manager.get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    v = [v for v in prompt.versions if v.version == version]
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    
    return {
        "version_id": v[0].version_id,
        "version": v[0].version,
        "template": v[0].template,
        "changelog": v[0].changelog,
        "created_by": v[0].created_by,
        "created_at": v[0].created_at.isoformat()
    }

@router.post("/{prompt_id}/test")
async def test_prompt(prompt_id: str, request: TestPromptModel):
    """
    测试Prompt
    
    v2.4: Prompt Management
    """
    result = prompt_manager.add_test_result(
        prompt_id=prompt_id,
        test_input=request.test_input,
        test_output="Mock output for testing",  # 实际会由LLM生成
        expected_output=request.expected_output,
        latency_ms=request.latency_ms,
        token_count=request.token_count
    )
    
    return {
        "result_id": result.result_id,
        "version": result.version,
        "message": "Test recorded"
    }

@router.post("/{prompt_id}/evaluate")
async def evaluate_result(prompt_id: str, request: EvaluateResultModel):
    """
    评估测试结果
    
    v2.4: Prompt Management
    """
    result = prompt_manager.evaluate_test_result(
        result_id=request.result_id,
        prompt_id=prompt_id,
        score=request.score
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return {"message": "Result evaluated"}

@router.get("/{prompt_id}/results")
async def get_test_results(prompt_id: str):
    """
    获取测试结果
    
    v2.4: Prompt Management
    """
    results = prompt_manager.get_test_results(prompt_id)
    
    return {
        "total": len(results),
        "results": [
            {
                "result_id": r.result_id,
                "version": r.version,
                "test_input": r.test_input,
                "test_output": r.test_output,
                "expected_output": r.expected_output,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "token_count": r.token_count,
                "created_at": r.created_at.isoformat()
            }
            for r in results
        ]
    }

@router.get("/summary")
async def get_prompt_summary():
    """
    获取Prompt统计
    
    v2.4: Prompt Management
    """
    summary = prompt_manager.get_prompt_summary()
    return summary
