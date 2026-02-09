"""
插件系统 v2.2
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import importlib
import inspect

@dataclass
class Plugin:
    """插件"""
    plugin_id: str
    name: str
    version: str
    description: str
    plugin_type: str  # metric, storage, integration, ui
    entry_point: str
    settings: Dict = field(default_factory=dict)
    enabled: bool = True
    loaded_at: Optional[datetime] = None

@dataclass
class PluginHook:
    """插件钩子"""
    hook_id: str
    plugin_id: str
    hook_name: str
    callback: Callable
    priority: int = 0

class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[PluginHook]] = {}  # hook_name -> hooks
        self.hooks_cache: Dict[str, List[Callable]] = {}  # hook_name -> callbacks
        self.loaded_modules: Dict[str, Any] = {}
    
    # 插件管理
    def register_plugin(
        self,
        name: str,
        version: str,
        description: str,
        plugin_type: str,
        entry_point: str,
        settings: Optional[Dict] = None
    ) -> Plugin:
        """注册插件"""
        plugin = Plugin(
            plugin_id=str(uuid4()),
            name=name,
            version=version,
            description=description,
            plugin_type=plugin_type,
            entry_point=entry_point,
            settings=settings or {}
        )
        
        self.plugins[plugin.plugin_id] = plugin
        return plugin
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """获取插件"""
        return self.plugins.get(plugin_id)
    
    def list_plugins(
        self,
        plugin_type: Optional[str] = None,
        enabled: Optional[bool] = None
    ) -> List[Plugin]:
        """列出插件"""
        plugins = list(self.plugins.values())
        
        if plugin_type:
            plugins = [p for p in plugins if p.plugin_type == plugin_type]
        if enabled is not None:
            plugins = [p for p in plugins if p.enabled == enabled]
        
        return plugins
    
    def enable_plugin(self, plugin_id: str) -> bool:
        """启用插件"""
        plugin = self.plugins.get(plugin_id)
        if plugin:
            plugin.enabled = True
            return True
        return False
    
    def disable_plugin(self, plugin_id: str) -> bool:
        """禁用插件"""
        plugin = self.plugins.get(plugin_id)
        if plugin:
            plugin.enabled = False
            return True
        return False
    
    def update_settings(self, plugin_id: str, settings: Dict) -> bool:
        """更新插件设置"""
        plugin = self.plugins.get(plugin_id)
        if plugin:
            plugin.settings.update(settings)
            return True
        return False
    
    # 钩子系统
    def register_hook(
        self,
        plugin_id: str,
        hook_name: str,
        callback: Callable,
        priority: int = 0
    ) -> PluginHook:
        """注册钩子"""
        hook = PluginHook(
            hook_id=str(uuid4()),
            plugin_id=plugin_id,
            hook_name=hook_name,
            callback=callback,
            priority=priority
        )
        
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        
        self.hooks[hook_name].append(hook)
        # 按优先级排序
        self.hooks[hook_name].sort(key=lambda h: h.priority, reverse=True)
        
        # 清除缓存
        if hook_name in self.hooks_cache:
            del self.hooks_cache[hook_name]
        
        return hook
    
    def unregister_hook(self, hook_id: str) -> bool:
        """取消注册钩子"""
        for hook_name, hooks in self.hooks.items():
            for hook in hooks:
                if hook.hook_id == hook_id:
                    hooks.remove(hook)
                    if hook_name in self.hooks_cache:
                        del self.hooks_cache[hook_name]
                    return True
        return False
    
    def call_hooks(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """调用所有钩子"""
        # 检查缓存
        if hook_name not in self.hooks_cache:
            if hook_name in self.hooks:
                self.hooks_cache[hook_name] = [h.callback for h in self.hooks[hook_name] if h.plugin_id in [p.plugin_id for p in self.plugins.values() if p.enabled]]
            else:
                self.hooks_cache[hook_name] = []
        
        results = []
        for callback in self.hooks_cache[hook_name]:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Hook error: {e}")
        
        return results
    
    def emit_event(self, event_name: str, data: Dict) -> None:
        """发射事件"""
        self.call_hooks(f"event_{event_name}", data)
    
    # 动态加载
    def load_plugin_module(self, plugin_id: str) -> Any:
        """动态加载插件模块"""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        if plugin.plugin_id in self.loaded_modules:
            return self.loaded_modules[plugin.plugin_id]
        
        # 动态导入
        module_path, class_name = plugin.entry_point.rsplit(".", 1)
        module = importlib.import_module(module_path)
        
        cls = getattr(module, class_name)
        instance = cls()
        
        self.loaded_modules[plugin.plugin_id] = instance
        plugin.loaded_at = datetime.utcnow()
        
        return instance
    
    # 插件市场
    def get_marketplace_plugins(self) -> List[Dict]:
        """获取市场插件列表"""
        return [
            {
                "name": "mlflow-tracker",
                "version": "1.0.0",
                "description": "MLflow实验跟踪集成",
                "type": "integration",
                "installed": any(p.name == "mlflow-tracker" for p in self.plugins.values())
            },
            {
                "name": "weave-trace",
                "version": "1.0.0",
                "description": "Weights & Biases可视化",
                "type": "integration",
                "installed": any(p.name == "weave-trace" for p in self.plugins.values())
            },
            {
                "name": "custom-metric",
                "version": "1.0.0",
                "description": "自定义评估指标",
                "type": "metric",
                "installed": any(p.name == "custom-metric" for p in self.plugins.values())
            },
            {
                "name": "s3-storage",
                "version": "1.0.0",
                "description": "S3存储后端",
                "type": "storage",
                "installed": any(p.name == "s3-storage" for p in self.plugins.values())
            }
        ]
    
    def install_marketplace_plugin(self, name: str) -> Plugin:
        """安装市场插件"""
        marketplace = {
            "mlflow-tracker": {
                "version": "1.0.0",
                "description": "MLflow实验跟踪集成",
                "type": "integration",
                "entry_point": "backend.plugins.mlflow:MLFlowTracker"
            },
            "weave-trace": {
                "version": "1.0.0",
                "description": "Weights & Biases可视化",
                "type": "integration",
                "entry_point": "backend.plugins.weave:WandBTracker"
            },
            "custom-metric": {
                "version": "1.0.0",
                "description": "自定义评估指标",
                "type": "metric",
                "entry_point": "backend.plugins.metrics:CustomMetric"
            },
            "s3-storage": {
                "version": "1.0.0",
                "description": "S3存储后端",
                "type": "storage",
                "entry_point": "backend.plugins.storage:S3Storage"
            }
        }
        
        if name not in marketplace:
            raise ValueError(f"Plugin {name} not in marketplace")
        
        info = marketplace[name]
        return self.register_plugin(
            name=name,
            version=info["version"],
            description=info["description"],
            plugin_type=info["type"],
            entry_point=info["entry_point"]
        )

# PluginManager实例
plugin_manager = PluginManager()

# 注册内置钩子
def example_hook(data: Dict) -> Dict:
    """示例钩子"""
    print(f"Hook called with: {data}")
    return {"processed": True}

# 初始化时注册示例钩子
plugin_manager.register_hook(
    plugin_id="builtin",
    hook_name="pre_training",
    callback=example_hook,
    priority=0
)
