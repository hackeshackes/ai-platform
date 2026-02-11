"""
Agents Factory - Agent工厂模块

提供企业级Agent模板管理和批量创建功能。

核心组件:
- TemplateEngine: 模板引擎，支持Jinja2渲染
- TemplateManager: 模板管理器，CRUD和版本控制
- models: Pydantic数据模型

使用示例:
    from agents.factory.template_manager import template_manager
    
    # 列出模板
    templates = template_manager.list_templates()
    
    # 创建Agent
    instance = template_manager.create_agent(
        template_id="research_agent",
        name="my-research-agent"
    )
"""

from .models import (
    AgentTemplate,
    AgentTemplateConfig,
    TemplateVersion,
    CreateAgentRequest,
    DeployAgentRequest,
    RollbackRequest,
    AgentInstance,
    BatchCreateResponse,
    FactoryStatus
)

from .template_engine import (
    TemplateEngine,
    TemplateRenderError,
    create_engine
)

from .template_manager import (
    TemplateManager,
    template_manager,
    get_manager
)

__all__ = [
    # Models
    "AgentTemplate",
    "AgentTemplateConfig",
    "TemplateVersion",
    "CreateAgentRequest",
    "DeployAgentRequest",
    "RollbackRequest",
    "AgentInstance",
    "BatchCreateResponse",
    "FactoryStatus",
    
    # Engine
    "TemplateEngine",
    "TemplateRenderError",
    "create_engine",
    
    # Manager
    "TemplateManager",
    "template_manager",
    "get_manager"
]
