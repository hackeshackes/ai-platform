"""
Template Manager - Agent模板管理器
负责模板的存储、版本控制和生命周期管理
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import shutil
import uuid
import yaml
import logging
import asyncio
from threading import Lock

from .models import (
    AgentTemplate,
    TemplateVersion,
    TemplateListItem,
    AgentInstance
)
from .template_engine import TemplateEngine, TemplateRenderError

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Agent模板管理器
    
    核心功能：
    - 模板CRUD操作
    - 版本控制
    - 模板存储管理
    - 模板导入导出
    """
    
    def __init__(self, templates_dir: str = None):
        """
        初始化模板管理器
        
        Args:
            templates_dir: 模板存储目录路径
        """
        self.templates_dir = Path(templates_dir) if templates_dir else \
            Path(__file__).parent / "templates"
        
        self._templates: Dict[str, AgentTemplate] = {}
        self._versions: Dict[str, List[TemplateVersion]] = {}
        self._instances: Dict[str, AgentInstance] = {}
        self._lock = Lock()
        self._engine = TemplateEngine()
        
        # 确保模板目录存在
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载内置模板
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """加载内置模板"""
        builtin_templates = [
            'research_agent.yaml',
            'coding_agent.yaml',
            'chat_agent.yaml',
            'support_agent.yaml'
        ]
        
        for template_file in builtin_templates:
            template_path = self.templates_dir / template_file
            if template_path.exists():
                try:
                    self._load_template_file(template_path)
                    logger.info(f"Loaded builtin template: {template_file}")
                except Exception as e:
                    logger.error(f"Failed to load {template_file}: {e}")
    
    def _load_template_file(self, path: Path) -> AgentTemplate:
        """加载单个模板文件"""
        content = path.read_text(encoding='utf-8')
        template = self._engine.parse_template(content)
        return self._register_template(template)
    
    def _register_template(self, template: AgentTemplate) -> AgentTemplate:
        """注册模板"""
        with self._lock:
            self._templates[template.name] = template
            if template.name not in self._versions:
                self._versions[template.name] = []
            
            # 添加初始版本
            version_entry = TemplateVersion(
                version=template.version,
                template=template,
                created_by="system",
                changelog="Initial version"
            )
            self._versions[template.name].append(version_entry)
            
            return template
    
    def create_template(
        self,
        name: str,
        description: str,
        version: str,
        capabilities: List[str],
        system_prompt: str,
        tools: List[str] = None,
        config: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> AgentTemplate:
        """
        创建新模板
        
        Args:
            name: 模板名称
            description: 描述
            version: 版本号
            capabilities: 能力列表
            system_prompt: 系统提示词
            tools: 工具列表
            config: 配置
            metadata: 元数据
            
        Returns:
            创建的模板对象
        """
        # 构建模板数据
        template_data = {
            'name': name,
            'description': description,
            'version': version,
            'capabilities': capabilities,
            'system_prompt': system_prompt,
            'tools': tools or [],
            'config': config or {},
            'metadata': metadata or {}
        }
        
        template = self._engine.parse_template(yaml.dump(template_data))
        return self._register_template(template)
    
    def get_template(self, template_id: str) -> Optional[AgentTemplate]:
        """获取模板"""
        return self._templates.get(template_id)
    
    def list_templates(self) -> List[TemplateListItem]:
        """列出所有模板"""
        items = []
        for template_id, template in self._templates.items():
            versions = self._versions.get(template_id, [])
            latest_version = versions[-1] if versions else None
            updated_at = latest_version.created_at if latest_version else datetime.now()
            
            items.append(TemplateListItem(
                id=template_id,
                name=template.name,
                description=template.description,
                version=template.version,
                capabilities=template.capabilities,
                created_at=updated_at,
                updated_at=updated_at
            ))
        return items
    
    def update_template(
        self,
        template_id: str,
        **updates
    ) -> Optional[AgentTemplate]:
        """
        更新模板
        
        Args:
            template_id: 模板ID
            updates: 更新字段
            
        Returns:
            更新后的模板
        """
        if template_id not in self._templates:
            return None
        
        with self._lock:
            old_template = self._templates[template_id]
            old_data = old_template.model_dump()
            old_data.update(updates)
            
            # 创建新版本
            new_template = AgentTemplate(**old_data)
            self._templates[template_id] = new_template
            
            # 添加版本记录
            version_entry = TemplateVersion(
                version=new_template.version,
                template=new_template,
                created_by=updates.get('updated_by', 'system'),
                changelog=updates.get('changelog', '')
            )
            self._versions[template_id].append(version_entry)
            
            logger.info(f"Updated template: {template_id}")
            return new_template
    
    def delete_template(self, template_id: str) -> bool:
        """删除模板"""
        if template_id not in self._templates:
            return False
        
        with self._lock:
            del self._templates[template_id]
            if template_id in self._versions:
                del self._versions[template_id]
            
            logger.info(f"Deleted template: {template_id}")
            return True
    
    def get_versions(self, template_id: str) -> List[Dict[str, Any]]:
        """获取模板版本历史"""
        versions = self._versions.get(template_id, [])
        return [
            {
                'version': v.version,
                'created_at': v.created_at.isoformat(),
                'created_by': v.created_by,
                'changelog': v.changelog
            }
            for v in versions
        ]
    
    def rollback_template(
        self,
        template_id: str,
        target_version: str
    ) -> Optional[AgentTemplate]:
        """
        回滚模板到指定版本
        
        Args:
            template_id: 模板ID
            target_version: 目标版本号
            
        Returns:
            回滚后的模板
        """
        versions = self._versions.get(template_id, [])
        target = None
        
        for v in versions:
            if v.version == target_version:
                target = v
                break
        
        if not target:
            raise ValueError(f"Version not found: {target_version}")
        
        # 创建新版本指向旧模板
        rollback_template = AgentTemplate(
            name=target.template.name,
            description=target.template.description,
            version=f"{target.template.version}.rollback",
            capabilities=target.template.capabilities,
            system_prompt=target.template.system_prompt,
            tools=target.template.tools,
            config=target.template.config.model_dump(),
            metadata={**target.template.metadata, 'rollbacked_from': target.version}
        )
        
        return self._register_template(rollback_template)
    
    def create_agent(
        self,
        template_id: str,
        name: str,
        variables: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> AgentInstance:
        """
        创建Agent实例
        
        Args:
            template_id: 模板ID
            name: Agent名称
            variables: 模板变量
            metadata: 元数据
            
        Returns:
            AgentInstance对象
            
        Raises:
            ValueError: 模板不存在
        """
        template = self._templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        # 验证变量
        validated_vars = self._engine.validate_variables(template, variables)
        
        # 渲染配置
        rendered = self._engine.render_template(template, validated_vars)
        
        # 创建实例
        instance = AgentInstance(
            id=str(uuid.uuid4()),
            name=name,
            template_id=template_id,
            template_version=template.version,
            config=rendered,
            status="created",
            metadata=metadata or {}
        )
        
        with self._lock:
            self._instances[instance.id] = instance
        
        logger.info(f"Created agent instance: {instance.id} from template: {template_id}")
        return instance
    
    async def create_batch(
        self,
        template_id: str,
        base_name: str,
        count: int,
        variables: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> List[AgentInstance]:
        """
        批量创建Agent实例
        
        Args:
            template_id: 模板ID
            base_name: 基础名称
            count: 创建数量
            variables: 模板变量
            metadata: 元数据
            
        Returns:
            创建的Agent实例列表
        """
        instances = []
        errors = []
        
        for i in range(count):
            agent_name = f"{base_name}_{i+1}" if count > 1 else base_name
            agent_vars = {
                **(variables or {}),
                'name': agent_name,
                'index': i + 1
            }
            
            try:
                instance = self.create_agent(
                    template_id=template_id,
                    name=agent_name,
                    variables=agent_vars,
                    metadata=metadata
                )
                instances.append(instance)
            except Exception as e:
                errors.append(f"Failed to create {agent_name}: {e}")
                logger.error(f"Batch create error: {e}")
        
        logger.info(f"Batch created {len(instances)} agents from template: {template_id}")
        return instances, errors
    
    def get_instance(self, instance_id: str) -> Optional[AgentInstance]:
        """获取Agent实例"""
        return self._instances.get(instance_id)
    
    def list_instances(self, status: str = None) -> List[AgentInstance]:
        """列出Agent实例"""
        instances = list(self._instances.values())
        if status:
            instances = [i for i in instances if i.status == status]
        return instances
    
    def update_instance_status(
        self,
        instance_id: str,
        status: str
    ) -> Optional[AgentInstance]:
        """更新实例状态"""
        instance = self._instances.get(instance_id)
        if not instance:
            return None
        
        instance.status = status
        if status == "running":
            instance.deployed_at = datetime.now()
        
        return instance
    
    def export_template(self, template_id: str) -> Optional[str]:
        """导出模板为YAML"""
        template = self._templates.get(template_id)
        if not template:
            return None
        
        return yaml.dump(template.model_dump(), allow_unicode=True)
    
    def import_template(
        self,
        yaml_content: str,
        overwrite: bool = False
    ) -> AgentTemplate:
        """导入模板"""
        template = self._engine.parse_template(yaml_content)
        
        if template.name in self._templates and not overwrite:
            raise ValueError(f"Template already exists: {template.name}")
        
        return self._register_template(template)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        instances = list(self._instances.values())
        return {
            'total_templates': len(self._templates),
            'total_agents': len(instances),
            'running_agents': sum(1 for i in instances if i.status == "running"),
            'stopped_agents': sum(1 for i in instances if i.status == "stopped"),
            'created_agents': sum(1 for i in instances if i.status == "created")
        }


# 全局模板管理器实例
template_manager = TemplateManager()


def get_manager() -> TemplateManager:
    """获取模板管理器实例"""
    return template_manager
