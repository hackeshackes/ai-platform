"""
Plugin Package - 插件系统

提供完整的Plugin市场功能:
- Plugin注册表 (registry.py)
- Plugin安装/卸载 (manager.py)
- Plugin市场浏览 (market.py)
- Plugin沙箱执行 (sandbox.py)
"""

from .registry import PluginRegistry, PluginMetadata, registry
from .manager import PluginManager, get_plugin_manager
from .market import PluginMarket
from .sandbox import PluginSandbox, SandboxConfig, ExecutionResult, plugin_sandbox

__all__ = [
    'PluginRegistry',
    'PluginMetadata',
    'registry',
    'PluginManager',
    'get_plugin_manager',
    'PluginMarket',
    'PluginSandbox',
    'SandboxConfig',
    'ExecutionResult',
    'plugin_sandbox',
]
