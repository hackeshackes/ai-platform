"""
Plugin Marketplace 模块 v2.4
对标: VS Code Marketplace, Jenkins Plugins
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4

class PluginCategory(str, Enum):
    """插件分类"""
    INTEGRATION = "integration"
    VISUALIZATION = "visualization"
    NOTIFICATION = "notification"
    MONITORING = "monitoring"
    SECURITY = "security"
    UTILITY = "utility"
    CUSTOM = "custom"

class PluginStatus(str, Enum):
    """插件状态"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    INSTALLED = "installed"

class InstallStatus(str, Enum):
    """安装状态"""
    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    UPDATING = "updating"
    ERROR = "error"

@dataclass
class Plugin:
    """插件"""
    plugin_id: str
    name: str
    description: str
    category: PluginCategory
    version: str
    status: PluginStatus
    author: str
    repository_url: Optional[str] = None
    homepage_url: Optional[str] = None
    license: str = "MIT"
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PluginVersion:
    """插件版本"""
    version_id: str
    plugin_id: str
    version: str
    changelog: str = ""
    download_url: Optional[str] = None
    file_size: int = 0
    checksum: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PluginReview:
    """插件评论"""
    review_id: str
    plugin_id: str
    user_id: str
    rating: int  # 1-5
    title: str
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PluginInstallation:
    """插件安装"""
    installation_id: str
    plugin_id: str
    version: str
    status: InstallStatus
    config: Dict = field(default_factory=dict)
    enabled: bool = True
    error_message: Optional[str] = None
    installed_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PluginHook:
    """插件钩子"""
    hook_id: str
    plugin_id: str
    hook_name: str
    handler: str
    priority: int = 0

class PluginMarketplaceEngine:
    """插件市场引擎 v2.4"""
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.versions: Dict[str, List[PluginVersion]] = {}
        self.reviews: List[PluginReview] = []
        self.installations: Dict[str, PluginInstallation] = {}
        self.hooks: List[PluginHook] = []
        
        # 初始化示例插件
        self._init_sample_plugins()
    
    def _init_sample_plugins(self):
        """初始化示例插件"""
        # 示例插件
        plugin = Plugin(
            plugin_id="slack-integration",
            name="Slack Integration",
            description="Slack通知集成",
            category=PluginCategory.INTEGRATION,
            version="1.0.0",
            status=PluginStatus.APPROVED,
            author="AI Platform Team",
            homepage_url="https://github.com/ai-platform/slack-integration",
            license="MIT",
            tags=["notification", "slack", "integration"],
            config_schema={
                "webhook_url": {"type": "string", "required": True},
                "channel": {"type": "string", "required": True}
            },
            metrics={"downloads": 1000, "rating": 4.5}
        )
        self.plugins[plugin.plugin_id] = plugin
        self.versions[plugin.plugin_id] = [
            PluginVersion(
                version_id="v1",
                plugin_id=plugin.plugin_id,
                version="1.0.0",
                changelog="Initial release"
            )
        ]
        
        # 数据可视化插件
        plugin2 = Plugin(
            plugin_id="advanced-viz",
            name="Advanced Visualization",
            description="高级数据可视化工具",
            category=PluginCategory.VISUALIZATION,
            version="2.0.0",
            status=PluginStatus.APPROVED,
            author="Viz Team",
            tags=["visualization", "charts", "dashboard"],
            metrics={"downloads": 500, "rating": 4.8}
        )
        self.plugins[plugin2.plugin_id] = plugin2
    
    # ==================== 插件管理 ====================
    
    def register_plugin(
        self,
        name: str,
        description: str,
        category: PluginCategory,
        version: str,
        author: str,
        repository_url: Optional[str] = None,
        homepage_url: Optional[str] = None,
        license: str = "MIT",
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        config_schema: Optional[Dict] = None
    ) -> Plugin:
        """注册插件"""
        plugin = Plugin(
            plugin_id=name.lower().replace(" ", "-"),
            name=name,
            description=description,
            category=category,
            version=version,
            status=PluginStatus.DRAFT,
            author=author,
            repository_url=repository_url,
            homepage_url=homepage_url,
            license=license,
            tags=tags or [],
            dependencies=dependencies or [],
            config_schema=config_schema or {}
        )
        
        self.plugins[plugin.plugin_id] = plugin
        self.versions[plugin.plugin_id] = []
        
        return plugin
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """获取插件"""
        return self.plugins.get(plugin_id)
    
    def list_plugins(
        self,
        category: Optional[PluginCategory] = None,
        status: Optional[PluginStatus] = None,
        search: Optional[str] = None
    ) -> List[Plugin]:
        """列出插件"""
        plugins = list(self.plugins.values())
        
        if category:
            plugins = [p for p in plugins if p.category == category]
        if status:
            plugins = [p for p in plugins if p.status == status]
        if search:
            search_lower = search.lower()
            plugins = [
                p for p in plugins
                if search_lower in p.name.lower() or search_lower in p.description.lower()
            ]
        
        return plugins
    
    def update_plugin(
        self,
        plugin_id: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config_schema: Optional[Dict] = None
    ) -> bool:
        """更新插件"""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return False
        
        if description:
            plugin.description = description
        if tags:
            plugin.tags = tags
        if config_schema:
            plugin.config_schema = config_schema
        
        plugin.updated_at = datetime.utcnow()
        return True
    
    def submit_for_review(self, plugin_id: str) -> bool:
        """提交审核"""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return False
        
        plugin.status = PluginStatus.PENDING_REVIEW
        plugin.updated_at = datetime.utcnow()
        return True
    
    def approve_plugin(self, plugin_id: str) -> bool:
        """批准插件"""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return False
        
        plugin.status = PluginStatus.APPROVED
        plugin.updated_at = datetime.utcnow()
        return True
    
    def reject_plugin(self, plugin_id: str, reason: str) -> bool:
        """拒绝插件"""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return False
        
        plugin.status = PluginStatus.REJECTED
        plugin.metrics["rejection_reason"] = reason
        plugin.updated_at = datetime.utcnow()
        return True
    
    # ==================== 版本管理 ====================
    
    def add_version(
        self,
        plugin_id: str,
        version: str,
        changelog: str = "",
        download_url: Optional[str] = None,
        file_size: int = 0
    ) -> PluginVersion:
        """添加版本"""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        v = PluginVersion(
            version_id=str(uuid4()),
            plugin_id=plugin_id,
            version=version,
            changelog=changelog,
            download_url=download_url,
            file_size=file_size
        )
        
        if plugin_id not in self.versions:
            self.versions[plugin_id] = []
        self.versions[plugin_id].append(v)
        
        # 更新插件版本
        plugin.version = version
        plugin.updated_at = datetime.utcnow()
        
        return v
    
    def get_versions(self, plugin_id: str) -> List[PluginVersion]:
        """获取版本列表"""
        return self.versions.get(plugin_id, [])
    
    # ==================== 评论管理 ====================
    
    def add_review(
        self,
        plugin_id: str,
        user_id: str,
        rating: int,
        title: str,
        content: str
    ) -> PluginReview:
        """添加评论"""
        review = PluginReview(
            review_id=str(uuid4()),
            plugin_id=plugin_id,
            user_id=user_id,
            rating=rating,
            title=title,
            content=content
        )
        
        self.reviews.append(review)
        
        # 更新插件评分
        plugin = self.plugins.get(plugin_id)
        if plugin:
            ratings = [
                r.rating for r in self.reviews if r.plugin_id == plugin_id
            ]
            plugin.metrics["rating"] = sum(ratings) / len(ratings) if ratings else 0
        
        return review
    
    def get_reviews(self, plugin_id: str) -> List[PluginReview]:
        """获取评论"""
        return [r for r in self.reviews if r.plugin_id == plugin_id]
    
    # ==================== 插件安装 ====================
    
    def install_plugin(
        self,
        plugin_id: str,
        version: str = "latest",
        config: Optional[Dict] = None
    ) -> PluginInstallation:
        """安装插件"""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        # 检查是否已安装
        if plugin_id in self.installations:
            existing = self.installations[plugin_id]
            if existing.status == InstallStatus.INSTALLING:
                raise ValueError("Plugin is already being installed")
        
        installation = PluginInstallation(
            installation_id=str(uuid4()),
            plugin_id=plugin_id,
            version=version,
            status=InstallStatus.INSTALLING,
            config=config or {}
        )
        
        self.installations[installation.installation_id] = installation
        
        # 模拟安装过程
        installation.status = InstallStatus.INSTALLED
        installation.installed_at = datetime.utcnow()
        
        return installation
    
    def uninstall_plugin(self, plugin_id: str) -> bool:
        """卸载插件"""
        if plugin_id in self.installations:
            del self.installations[plugin_id]
            return True
        return False
    
    def get_installed_plugins(self) -> List[PluginInstallation]:
        """获取已安装插件"""
        return list(self.installations.values())
    
    def update_plugin_config(
        self,
        plugin_id: str,
        config: Dict
    ) -> bool:
        """更新插件配置"""
        if plugin_id not in self.installations:
            return False
        
        self.installations[plugin_id].config = config
        self.installations[plugin_id].updated_at = datetime.utcnow()
        return True
    
    def enable_plugin(self, plugin_id: str, enabled: bool) -> bool:
        """启用/禁用插件"""
        if plugin_id not in self.installations:
            return False
        
        self.installations[plugin_id].enabled = enabled
        self.installations[plugin_id].updated_at = datetime.utcnow()
        return True
    
    # ==================== 插件钩子 ====================
    
    def register_hook(
        self,
        plugin_id: str,
        hook_name: str,
        handler: str,
        priority: int = 0
    ) -> PluginHook:
        """注册钩子"""
        hook = PluginHook(
            hook_id=str(uuid4()),
            plugin_id=plugin_id,
            hook_name=hook_name,
            handler=handler,
            priority=priority
        )
        
        self.hooks.append(hook)
        return hook
    
    def get_hooks(self, hook_name: str) -> List[PluginHook]:
        """获取钩子"""
        return [h for h in self.hooks if h.hook_name == hook_name]
    
    def trigger_hook(self, hook_name: str, data: Dict) -> List[Dict]:
        """触发钩子"""
        hooks = sorted(
            [h for h in self.hooks if h.hook_name == hook_name],
            key=lambda h: h.priority
        )
        
        results = []
        for hook in hooks:
            results.append({
                "plugin_id": hook.plugin_id,
                "handler": hook.handler,
                "result": f"Executed {hook.handler}"
            })
        
        return results
    
    # ==================== 统计信息 ====================
    
    def get_summary(self) -> Dict:
        """获取统计"""
        approved = [p for p in self.plugins.values() if p.status == PluginStatus.APPROVED]
        installed = [i for i in self.installations.values()]
        
        return {
            "total_plugins": len(self.plugins),
            "approved_plugins": len(approved),
            "pending_review": len([p for p in self.plugins.values() if p.status == PluginStatus.PENDING_REVIEW]),
            "installed_plugins": len(installed),
            "total_reviews": len(self.reviews),
            "plugins_by_category": {
                c.value: len([p for p in self.plugins.values() if p.category == c])
                for c in PluginCategory
            }
        }

# PluginMarketplaceEngine实例
plugin_marketplace_engine = PluginMarketplaceEngine()
