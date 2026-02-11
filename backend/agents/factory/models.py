"""
Template Models - Agent模板Pydantic模型定义
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime


class AgentTemplateConfig(BaseModel):
    """Agent模板配置模型"""
    model: str = Field(default="gpt-4", description="模型名称")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(default=4096, gt=0, description="最大token数")
    stream: bool = Field(default=False, description="是否流式输出")
    timeout: int = Field(default=30, gt=0, description="超时时间(秒)")


class AgentTemplate(BaseModel):
    """Agent模板主模型"""
    name: str = Field(..., min_length=1, max_length=100, description="模板名称")
    description: str = Field(..., min_length=1, max_length=500, description="模板描述")
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$", description="版本号")
    capabilities: List[str] = Field(..., description="能力列表")
    config: AgentTemplateConfig = Field(default_factory=AgentTemplateConfig, description="配置")
    tools: List[str] = Field(default_factory=list, description="可用工具列表")
    system_prompt: str = Field(..., min_length=10, description="系统提示词")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    @field_validator('capabilities')
    @classmethod
    def validate_capabilities(cls, v):
        if not v:
            raise ValueError("At least one capability is required")
        return v
    
    @field_validator('tools')
    @classmethod
    def validate_tools(cls, v):
        allowed_tools = [
            'web_search', 'knowledge_graph', 'rag_enhanced',
            'file_read', 'file_write', 'code_execution',
            'memory', 'reasoning', 'analysis', 'summarization',
            'translation', 'image_generation', 'data_analysis'
        ]
        for tool in v:
            if tool not in allowed_tools:
                raise ValueError(f"Invalid tool: {tool}. Allowed: {allowed_tools}")
        return v


class TemplateVersion(BaseModel):
    """模板版本模型"""
    version: str = Field(..., description="版本号")
    template: AgentTemplate = Field(..., description="模板内容")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    created_by: str = Field(default="system", description="创建者")
    changelog: str = Field(default="", description="变更日志")


class CreateAgentRequest(BaseModel):
    """创建Agent请求"""
    template_id: str = Field(..., description="模板ID")
    name: str = Field(..., min_length=1, max_length=100, description="Agent名称")
    count: int = Field(default=1, ge=1, le=10, description="创建数量")
    variables: Dict[str, Any] = Field(default_factory=dict, description="模板变量")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class DeployAgentRequest(BaseModel):
    """部署Agent请求"""
    agent_ids: List[str] = Field(..., min_length=1, description="Agent ID列表")
    environment: str = Field(default="production", description="部署环境")
    replicas: int = Field(default=1, ge=1, le=10, description="副本数")
    resources: Dict[str, Any] = Field(default_factory=dict, description="资源配置")


class RollbackRequest(BaseModel):
    """回滚请求"""
    agent_id: str = Field(..., description="Agent ID")
    target_version: str = Field(..., description="目标版本")


class AgentInstance(BaseModel):
    """Agent实例模型"""
    id: str = Field(..., description="实例ID")
    name: str = Field(..., description="实例名称")
    template_id: str = Field(..., description="模板ID")
    template_version: str = Field(..., description="模板版本")
    config: Dict[str, Any] = Field(..., description="配置")
    status: str = Field(default="created", description="状态")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    deployed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class BatchCreateResponse(BaseModel):
    """批量创建响应"""
    success: bool
    agents: List[AgentInstance]
    total_count: int
    failed_count: int
    errors: List[str] = Field(default_factory=list)


class DeployResponse(BaseModel):
    """部署响应"""
    success: bool
    deployed_agents: List[str]
    failed_agents: List[str]
    message: str


class RollbackResponse(BaseModel):
    """回滚响应"""
    success: bool
    agent_id: str
    previous_version: str
    current_version: str
    message: str


class TemplateListItem(BaseModel):
    """模板列表项"""
    id: str
    name: str
    description: str
    version: str
    capabilities: List[str]
    created_at: datetime
    updated_at: datetime


class FactoryStatus(BaseModel):
    """工厂状态模型"""
    total_templates: int
    total_agents: int
    running_agents: int
    stopped_agents: int
    active_sessions: int
