"""
Template Engine - Agent模板引擎
支持Jinja2模板渲染和配置验证
"""

from typing import Dict, Any, List, Optional
from jinja2 import Environment, BaseLoader, TemplateSyntaxError, UndefinedError
import yaml
import re
import json
import logging

from .models import AgentTemplate, AgentTemplateConfig

logger = logging.getLogger(__name__)


class TemplateEngine:
    """
    Agent模板引擎
    
    核心功能：
    - 模板解析与验证
    - Jinja2模板渲染
    - 模板变量管理
    """
    
    def __init__(self):
        self.jinja_env = Environment(
            loader=BaseLoader(),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        self._add_custom_filters()
    
    def _add_custom_filters(self):
        """添加自定义Jinja2过滤器"""
        self.jinja_env.filters['to_json'] = self._to_json_filter
        self.jinja_env.filters['upper'] = str.upper
        self.jinja_env.filters['lower'] = str.lower
        self.jinja_env.filters['capitalize'] = str.capitalize
        self.jinja_env.filters['first'] = lambda x: x[0] if x else ""
        self.jinja_env.filters['last'] = lambda x: x[-1] if x else ""
        self.jinja_env.filters['length'] = len
        self.jinja_env.filters['default'] = lambda x, d: x if x else d
    
    def _to_json_filter(self, value: Any) -> str:
        """将值转换为JSON字符串"""
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    
    def parse_template(self, template_content: str) -> AgentTemplate:
        """
        解析YAML模板内容
        
        Args:
            template_content: YAML格式的模板内容
            
        Returns:
            解析后的AgentTemplate对象
            
        Raises:
            ValueError: 模板格式错误
        """
        try:
            data = yaml.safe_load(template_content)
            if not data:
                raise ValueError("Empty template content")
            
            return AgentTemplate(**data)
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parse error: {e}")
            raise ValueError(f"Invalid YAML format: {e}")
        except Exception as e:
            logger.error(f"Template parse error: {e}")
            raise ValueError(f"Template validation failed: {e}")
    
    def render_template(
        self,
        template: AgentTemplate,
        variables: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        渲染模板
        
        Args:
            template: AgentTemplate对象
            variables: 模板变量
            
        Returns:
            渲染后的配置字典
            
        Raises:
            TemplateRenderError: 渲染失败
        """
        if variables is None:
            variables = {}
        
        try:
            # 构建渲染上下文
            context = {
                'name': template.name,
                'description': template.description,
                'version': template.version,
                'capabilities': template.capabilities,
                'config': template.config.model_dump(),
                'tools': template.tools,
                'system_prompt': template.system_prompt,
                'metadata': template.metadata,
                **variables
            }
            
            # 渲染system_prompt
            prompt_template = self.jinja_env.from_string(template.system_prompt)
            rendered_prompt = prompt_template.render(**context)
            
            # 构建渲染后的配置
            rendered_config = {
                'name': variables.get('name', template.name),
                'description': variables.get('description', template.description),
                'model': template.config.model,
                'temperature': template.config.temperature,
                'max_tokens': template.config.max_tokens,
                'stream': template.config.stream,
                'timeout': template.config.timeout,
                'capabilities': template.capabilities,
                'tools': template.tools,
                'system_prompt': rendered_prompt,
                'template_id': template.name,
                'template_version': template.version,
                'metadata': {**template.metadata, **variables.get('metadata', {})}
            }
            
            logger.info(f"Template '{template.name}' rendered successfully")
            return rendered_config
            
        except TemplateSyntaxError as e:
            logger.error(f"Template syntax error: {e}")
            raise TemplateRenderError(f"Template syntax error: {e}")
        except UndefinedError as e:
            logger.error(f"Undefined variable error: {e}")
            raise TemplateRenderError(f"Undefined variable: {e}")
        except Exception as e:
            logger.error(f"Template render error: {e}")
            raise TemplateRenderError(f"Failed to render template: {e}")
    
    def render_system_prompt(
        self,
        system_prompt: str,
        variables: Dict[str, Any] = None
    ) -> str:
        """
        渲染系统提示词
        
        Args:
            system_prompt: 系统提示词模板
            variables: 模板变量
            
        Returns:
            渲染后的提示词
        """
        if variables is None:
            variables = {}
        
        try:
            template = self.jinja_env.from_string(system_prompt)
            return template.render(**variables)
        except Exception as e:
            logger.error(f"System prompt render error: {e}")
            raise TemplateRenderError(f"Failed to render system prompt: {e}")
    
    def validate_variables(
        self,
        template: AgentTemplate,
        variables: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        验证模板变量
        
        Args:
            template: AgentTemplate对象
            variables: 待验证的变量
            
        Returns:
            验证通过或修正后的变量
            
        Raises:
            ValidationError: 验证失败
        """
        if variables is None:
            variables = {}
        
        validated = {}
        required_vars = ['name']
        
        # 检查必需变量
        for var in required_vars:
            if var not in variables:
                raise ValidationError(f"Missing required variable: {var}")
            validated[var] = variables[var]
        
        # 验证变量类型和格式
        validated.update(variables)
        
        # 验证名称格式
        if 'name' in validated:
            name = validated['name']
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
                raise ValidationError(
                    f"Invalid name format: {name}. "
                    "Must start with letter and contain only a-z, A-Z, 0-9, _, -"
                )
        
        logger.info(f"Variables validated: {list(validated.keys())}")
        return validated
    
    def merge_config(
        self,
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        合并配置
        
        Args:
            base_config: 基础配置
            override_config: 覆盖配置
            
        Returns:
            合并后的配置
        """
        merged = base_config.copy()
        merged.update(override_config)
        return merged


class TemplateRenderError(Exception):
    """模板渲染错误"""
    pass


class ValidationError(Exception):
    """验证错误"""
    pass


# 全局模板引擎实例
template_engine = TemplateEngine()


def create_engine() -> TemplateEngine:
    """创建模板引擎实例"""
    return TemplateEngine()
