"""
Agents Factory API - Agent工厂API端点
提供模板管理、Agent创建、部署和回滚的REST API接口
"""

from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import logging
import asyncio

from agents.factory.models import (
    AgentTemplate,
    CreateAgentRequest,
    DeployAgentRequest,
    RollbackRequest,
    BatchCreateResponse,
    DeployResponse,
    RollbackResponse,
    FactoryStatus
)
from agents.factory.template_manager import template_manager
from agents.factory.template_engine import TemplateEngine, TemplateRenderError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents/factory", tags=["Agent Factory"])


# ============== 请求/响应模型 ==============

class CreateTemplateRequest(BaseModel):
    """创建模板请求"""
    name: str = Field(..., min_length=1, max_length=100, description="模板名称")
    description: str = Field(..., min_length=1, max_length=500, description="模板描述")
    version: str = Field(default="1.0.0", description="版本号")
    capabilities: List[str] = Field(..., description="能力列表")
    system_prompt: str = Field(..., min_length=10, description="系统提示词")
    tools: List[str] = Field(default_factory=list, description="工具列表")
    config: Dict[str, Any] = Field(default_factory=dict, description="配置")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class UpdateTemplateRequest(BaseModel):
    """更新模板请求"""
    description: str = Field(None, description="模板描述")
    capabilities: List[str] = Field(None, description="能力列表")
    system_prompt: str = Field(None, description="系统提示词")
    tools: List[str] = Field(None, description="工具列表")
    config: Dict[str, Any] = Field(None, description="配置")
    metadata: Dict[str, Any] = Field(None, description="元数据")
    changelog: str = Field(default="", description="变更日志")
    updated_by: str = Field(default="api", description="更新者")


class TemplateResponse(BaseModel):
    """模板响应"""
    success: bool
    template: Dict[str, Any]
    message: str = ""


class TemplateListResponse(BaseModel):
    """模板列表响应"""
    success: bool
    templates: List[Dict[str, Any]]
    total: int


class RenderTemplateRequest(BaseModel):
    """渲染模板请求"""
    template_id: str = Field(..., description="模板ID")
    variables: Dict[str, Any] = Field(default_factory=dict, description="模板变量")


class RollbackTemplateRequest(BaseModel):
    """回滚模板请求"""
    template_id: str = Field(..., description="模板ID")
    target_version: str = Field(..., description="目标版本")


class ImportTemplateRequest(BaseModel):
    """导入模板请求"""
    yaml_content: str = Field(..., description="YAML内容")
    overwrite: bool = Field(default=False, description="是否覆盖")


class BatchCreateRequest(BaseModel):
    """批量创建请求"""
    template_id: str = Field(..., description="模板ID")
    base_name: str = Field(..., min_length=1, max_length=100, description="基础名称")
    count: int = Field(default=1, ge=1, le=10, description="创建数量")
    variables: Dict[str, Any] = Field(default_factory=dict, description="模板变量")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


# ============== 模板管理API ==============

@router.post("/templates", response_model=TemplateResponse)
async def create_template(request: CreateTemplateRequest):
    """
    创建Agent模板
    
    创建一个新的Agent模板定义。
    """
    try:
        template = template_manager.create_template(
            name=request.name,
            description=request.description,
            version=request.version,
            capabilities=request.capabilities,
            system_prompt=request.system_prompt,
            tools=request.tools,
            config=request.config,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "template": template.model_dump(),
            "message": f"Template '{request.name}' created successfully"
        }
    except Exception as e:
        logger.error(f"Create template error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates", response_model=TemplateListResponse)
async def list_templates():
    """
    列出所有Agent模板
    
    查询所有可用的Agent模板。
    """
    try:
        templates = template_manager.list_templates()
        return {
            "success": True,
            "templates": [t.model_dump() for t in templates],
            "total": len(templates)
        }
    except Exception as e:
        logger.error(f"List templates error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/{template_id}", response_model=TemplateResponse)
async def get_template(template_id: str):
    """
    获取模板详情
    """
    try:
        template = template_manager.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return {
            "success": True,
            "template": template.model_dump()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get template error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/templates/{template_id}", response_model=TemplateResponse)
async def update_template(template_id: str, request: UpdateTemplateRequest):
    """
    更新模板
    """
    try:
        updates = {k: v for k, v in request.model_dump().items() if v is not None}
        
        template = template_manager.update_template(template_id, **updates)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return {
            "success": True,
            "template": template.model_dump(),
            "message": f"Template '{template_id}' updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update template error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/templates/{template_id}")
async def delete_template(template_id: str):
    """
    删除模板
    """
    try:
        success = template_manager.delete_template(template_id)
        if not success:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return {
            "success": True,
            "message": f"Template '{template_id}' deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete template error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/{template_id}/versions")
async def get_template_versions(template_id: str):
    """
    获取模板版本历史
    """
    try:
        versions = template_manager.get_versions(template_id)
        if not versions:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return {
            "success": True,
            "template_id": template_id,
            "versions": versions
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get versions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates/{template_id}/rollback", response_model=TemplateResponse)
async def rollback_template(template_id: str, request: RollbackTemplateRequest):
    """
    回滚模板到指定版本
    """
    try:
        template = template_manager.rollback_template(
            template_id,
            request.target_version
        )
        
        return {
            "success": True,
            "template": template.model_dump(),
            "message": f"Template rolled back to version {request.target_version}"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Rollback template error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates/render")
async def render_template(request: RenderTemplateRequest):
    """
    渲染模板
    
    使用变量渲染模板，返回配置结果。
    """
    try:
        template = template_manager.get_template(request.template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        engine = TemplateEngine()
        validated_vars = engine.validate_variables(template, request.variables)
        rendered = engine.render_template(template, validated_vars)
        
        return {
            "success": True,
            "rendered": rendered
        }
    except TemplateRenderError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Render template error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/templates/import")
async def import_template(request: ImportTemplateRequest):
    """
    导入模板
    """
    try:
        template = template_manager.import_template(
            request.yaml_content,
            overwrite=request.overwrite
        )
        
        return {
            "success": True,
            "template": template.model_dump(),
            "message": f"Template '{template.name}' imported successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Import template error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/{template_id}/export")
async def export_template(template_id: str):
    """
    导出模板
    """
    try:
        yaml_content = template_manager.export_template(template_id)
        if not yaml_content:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return {
            "success": True,
            "yaml_content": yaml_content
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export template error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Agent创建API ==============

@router.post("/create", response_model=BatchCreateResponse)
async def create_agent(request: CreateAgentRequest):
    """
    创建Agent实例
    
    根据模板创建单个或多个Agent实例。
    """
    try:
        instances, errors = await template_manager.create_batch(
            template_id=request.template_id,
            base_name=request.name,
            count=request.count,
            variables=request.variables,
            metadata=request.metadata
        )
        
        return {
            "success": len(errors) == 0,
            "agents": [i.model_dump() for i in instances],
            "total_count": len(instances),
            "failed_count": len(errors),
            "errors": errors
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Create agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchCreateResponse)
async def batch_create_agents(request: BatchCreateRequest):
    """
    批量创建Agent实例
    """
    try:
        instances, errors = await template_manager.create_batch(
            template_id=request.template_id,
            base_name=request.base_name,
            count=request.count,
            variables=request.variables,
            metadata=request.metadata
        )
        
        return {
            "success": len(errors) == 0,
            "agents": [i.model_dump() for i in instances],
            "total_count": len(instances),
            "failed_count": len(errors),
            "errors": errors
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Batch create error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/instances")
async def list_instances(status: str = None):
    """
    列出Agent实例
    """
    try:
        instances = template_manager.list_instances(status=status)
        return {
            "success": True,
            "instances": [i.model_dump() for i in instances],
            "total": len(instances)
        }
    except Exception as e:
        logger.error(f"List instances error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/instances/{instance_id}")
async def get_instance(instance_id: str):
    """
    获取Agent实例详情
    """
    try:
        instance = template_manager.get_instance(instance_id)
        if not instance:
            raise HTTPException(status_code=404, detail="Instance not found")
        
        return {
            "success": True,
            "instance": instance.model_dump()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get instance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Agent部署API ==============

@router.post("/deploy", response_model=DeployResponse)
async def deploy_agents(request: DeployAgentRequest):
    """
    部署Agent实例
    
    将Agent实例部署到指定环境。
    """
    try:
        deployed = []
        failed = []
        
        for agent_id in request.agent_ids:
            instance = template_manager.get_instance(agent_id)
            if instance:
                # 更新状态为部署中
                template_manager.update_instance_status(agent_id, "deploying")
                
                # 模拟部署逻辑（实际场景中会调用部署服务）
                deployed.append(agent_id)
                
                # 更新状态为运行中
                template_manager.update_instance_status(agent_id, "running")
            else:
                failed.append(agent_id)
        
        return {
            "success": len(failed) == 0,
            "deployed_agents": deployed,
            "failed_agents": failed,
            "message": f"Deployed {len(deployed)} agents, {len(failed)} failed"
        }
    except Exception as e:
        logger.error(f"Deploy agents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_agents(agent_ids: List[str]):
    """
    停止Agent实例
    """
    try:
        stopped = []
        failed = []
        
        for agent_id in agent_ids:
            instance = template_manager.update_instance_status(agent_id, "stopped")
            if instance:
                stopped.append(agent_id)
            else:
                failed.append(agent_id)
        
        return {
            "success": len(failed) == 0,
            "stopped_agents": stopped,
            "failed_agents": failed,
            "message": f"Stopped {len(stopped)} agents"
        }
    except Exception as e:
        logger.error(f"Stop agents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 回滚API ==============

@router.post("/rollback", response_model=RollbackResponse)
async def rollback_agent(request: RollbackRequest):
    """
    回滚Agent实例
    """
    try:
        instance = template_manager.get_instance(request.agent_id)
        if not instance:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        previous_version = instance.template_version
        
        # 获取目标版本的模板配置
        versions = template_manager.get_versions(instance.template_id)
        target_config = None
        for v in versions:
            if v['version'] == request.target_version:
                target_config = v
                break
        
        if not target_config:
            raise HTTPException(status_code=404, detail="Target version not found")
        
        # 更新实例配置
        instance.config['template_version'] = request.target_version
        instance.config['rollbacked_from'] = previous_version
        
        return {
            "success": True,
            "agent_id": request.agent_id,
            "previous_version": previous_version,
            "current_version": request.target_version,
            "message": f"Agent rolled back from {previous_version} to {request.target_version}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rollback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== 状态监控API ==============

@router.get("/status", response_model=FactoryStatus)
async def get_factory_status():
    """
    获取工厂状态
    
    返回Agent工厂的整体状态统计。
    """
    try:
        stats = template_manager.get_statistics()
        return FactoryStatus(**stats)
    except Exception as e:
        logger.error(f"Get factory status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    健康检查
    """
    return {
        "status": "healthy",
        "service": "agents_factory",
        "timestamp": datetime.now().isoformat()
    }
