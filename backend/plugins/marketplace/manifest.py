"""
Plugin Manifest - Plugin清单解析
负责解析和验证plugin.yaml清单文件
"""

import yaml
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PluginCategory(str, Enum):
    """Plugin分类枚举"""
    TOOL = "tool"           # 工具类
    AGENT = "agent"         # Agent类
    INTEGRATION = "integration"  # 集成类
    UI = "ui"              # UI组件
    VISUALIZATION = "visualization"  # 可视化
    DATA_SOURCE = "data_source"  # 数据源


@dataclass
class HookConfig:
    """Hook配置"""
    on_install: Optional[str] = None
    on_uninstall: Optional[str] = None
    on_activate: Optional[str] = None
    on_deactivate: Optional[str] = None
    on_update: Optional[str] = None


@dataclass
class PluginManifest:
    """Plugin清单数据类"""
    name: str
    version: str
    display_name: str
    description: str
    author: str
    category: PluginCategory
    
    # 可选字段
    tags: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    hooks: HookConfig = field(default_factory=HookConfig)
    min_platform_version: str = "1.0.0"
    entry_point: str = ""
    homepage: str = ""
    repository: str = ""
    license: str = "MIT"
    icon: Optional[str] = None
    screenshots: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'name': self.name,
            'version': self.version,
            'display_name': self.display_name,
            'description': self.description,
            'author': self.author,
            'category': self.category.value if isinstance(self.category, PluginCategory) else self.category,
            'tags': self.tags,
            'permissions': self.permissions,
            'dependencies': self.dependencies,
            'hooks': {
                'on_install': self.hooks.on_install,
                'on_uninstall': self.hooks.on_uninstall,
                'on_activate': self.hooks.on_activate,
                'on_deactivate': self.hooks.on_deactivate,
                'on_update': self.hooks.on_update,
            },
            'min_platform_version': self.min_platform_version,
            'entry_point': self.entry_point,
            'homepage': self.homepage,
            'repository': self.repository,
            'license': self.license,
            'icon': self.icon,
            'screenshots': self.screenshots,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PluginManifest':
        """从字典创建"""
        hooks_data = data.get('hooks', {})
        hooks = HookConfig(
            on_install=hooks_data.get('on_install'),
            on_uninstall=hooks_data.get('on_uninstall'),
            on_activate=hooks_data.get('on_activate'),
            on_deactivate=hooks_data.get('on_deactivate'),
            on_update=hooks_data.get('on_update'),
        )
        
        category = data.get('category', 'tool')
        if isinstance(category, str):
            try:
                category = PluginCategory(category)
            except ValueError:
                category = PluginCategory.TOOL
        
        return cls(
            name=data.get('name', ''),
            version=data.get('version', '1.0.0'),
            display_name=data.get('display_name', data.get('name', '')),
            description=data.get('description', ''),
            author=data.get('author', 'unknown'),
            category=category,
            tags=data.get('tags', []),
            permissions=data.get('permissions', []),
            dependencies=data.get('dependencies', {}),
            hooks=hooks,
            min_platform_version=data.get('min_platform_version', '1.0.0'),
            entry_point=data.get('entry_point', ''),
            homepage=data.get('homepage', ''),
            repository=data.get('repository', ''),
            license=data.get('license', 'MIT'),
            icon=data.get('icon'),
            screenshots=data.get('screenshots', []),
        )


class ManifestParser:
    """Plugin清单解析器"""
    
    MANIFEST_FILENAME = "plugin.yaml"
    
    @staticmethod
    def parse(manifest_path: str) -> PluginManifest:
        """
        解析Plugin清单文件
        
        Args:
            manifest_path: 清单文件路径 (plugin.yaml)
        
        Returns:
            PluginManifest对象
        
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 解析错误
        """
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = yaml.safe_load(content)
            
            if not isinstance(data, dict):
                raise ValueError("Invalid manifest format: expected dictionary")
            
            manifest = PluginManifest.from_dict(data)
            
            # 验证必需字段
            errors = ManifestValidator.validate(manifest)
            if errors:
                raise ValueError(f"Manifest validation failed: {', '.join(errors)}")
            
            return manifest
        
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse manifest: {e}")
    
    @staticmethod
    def parse_from_string(yaml_content: str) -> PluginManifest:
        """
        从字符串解析Plugin清单
        
        Args:
            yaml_content: YAML字符串
        
        Returns:
            PluginManifest对象
        """
        data = yaml.safe_load(yaml_content)
        
        if not isinstance(data, dict):
            raise ValueError("Invalid manifest format: expected dictionary")
        
        manifest = PluginManifest.from_dict(data)
        
        errors = ManifestValidator.validate(manifest)
        if errors:
            raise ValueError(f"Manifest validation failed: {', '.join(errors)}")
        
        return manifest
    
    @staticmethod
    def find_manifest(plugin_dir: str) -> Optional[str]:
        """
        在目录中查找Plugin清单文件
        
        Args:
            plugin_dir: Plugin目录路径
        
        Returns:
            清单文件路径，未找到返回None
        """
        manifest_path = os.path.join(plugin_dir, ManifestParser.MANIFEST_FILENAME)
        
        if os.path.exists(manifest_path):
            return manifest_path
        
        for root, dirs, files in os.walk(plugin_dir):
            if ManifestParser.MANIFEST_FILENAME in files:
                return os.path.join(root, ManifestParser.MANIFEST_FILENAME)
        
        return None


class ManifestValidator:
    """Plugin清单验证器"""
    
    REQUIRED_FIELDS = ['name', 'version', 'display_name', 'description', 'author', 'category']
    OPTIONAL_FIELDS = [
        'tags', 'permissions', 'dependencies', 'hooks', 'min_platform_version',
        'entry_point', 'homepage', 'repository', 'license', 'icon', 'screenshots'
    ]
    
    @staticmethod
    def validate(manifest: PluginManifest) -> List[str]:
        """
        验证Plugin清单
        
        Args:
            manifest: PluginManifest对象
        
        Returns:
            错误列表，空列表表示验证通过
        """
        errors = []
        
        for field_name in ManifestValidator.REQUIRED_FIELDS:
            value = getattr(manifest, field_name, None)
            if not value or (isinstance(value, str) and not value.strip()):
                errors.append(f"Required field '{field_name}' is missing or empty")
        
        # 验证版本格式
        if manifest.version:
            if not ManifestValidator._is_valid_version(manifest.version):
                errors.append(f"Invalid version format: {manifest.version}")
        
        # 验证分类
        if manifest.category:
            try:
                if isinstance(manifest.category, str):
                    PluginCategory(manifest.category)
            except ValueError:
                valid_categories = [c.value for c in PluginCategory]
                errors.append(f"Invalid category: {manifest.category}. Valid categories: {valid_categories}")
        
        # 验证权限格式
        for permission in manifest.permissions:
            if not ManifestValidator._is_valid_permission(permission):
                errors.append(f"Invalid permission format: {permission}")
        
        # 验证依赖版本
        for dep, version in manifest.dependencies.items():
            if not ManifestValidator._is_valid_dependency(version):
                errors.append(f"Invalid dependency version: {dep}={version}")
        
        # 验证Hook函数名
        hooks = manifest.hooks
        for hook_name, hook_func in [
            ('on_install', hooks.on_install),
            ('on_uninstall', hooks.on_uninstall),
            ('on_activate', hooks.on_activate),
            ('on_deactivate', hooks.on_deactivate),
            ('on_update', hooks.on_update),
        ]:
            if hook_func and not ManifestValidator._is_valid_function_name(hook_func):
                errors.append(f"Invalid {hook_name} function name: {hook_func}")
        
        return errors
    
    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """验证版本格式"""
        import re
        pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$'
        return bool(re.match(pattern, version))
    
    @staticmethod
    def _is_valid_permission(permission: str) -> bool:
        """验证权限格式"""
        import re
        pattern = r'^[a-z][a-z0-9._-]*$'
        return bool(re.match(pattern, permission))
    
    @staticmethod
    def _is_valid_dependency(version_spec: str) -> bool:
        """验证依赖版本规格"""
        import re
        valid_patterns = [
            r'^>=?\d+(\.\d+)*$',
            r'^<=?\d+(\.\d+)*$',
            r'^==?\d+(\.\d+)*$',
            r'^!=?\d+(\.\d+)*$',
            r'^~=?\d+(\.\d+)*$',
            r'^>=?\d+(\.\d+)*,<=?\d+(\.\d+)*$',
        ]
        return any(re.match(pattern, version_spec) for pattern in valid_patterns)
    
    @staticmethod
    def _is_valid_function_name(name: str) -> bool:
        """验证函数名格式"""
        import re
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        return bool(re.match(pattern, name))
