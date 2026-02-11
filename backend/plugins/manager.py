"""
Plugin Manager - Plugin管理器
负责Plugin的安装、卸载、更新和生命周期管理
"""

import os
import shutil
import subprocess
import zipfile
import hashlib
import json
import tempfile
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import logging

from .registry import PluginRegistry, PluginMetadata

logger = logging.getLogger(__name__)


class PluginManager:
    """Plugin管理器"""
    
    def __init__(self, 
                 registry: PluginRegistry,
                 plugins_dir: str = "./data/plugins/installed",
                 cache_dir: str = "./data/plugins/cache"):
        self.registry = registry
        self.plugins_dir = plugins_dir
        self.cache_dir = cache_dir
        self.installation_history: Dict[str, Dict] = {}
        
        os.makedirs(plugins_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        self._load_history()
    
    def _load_history(self) -> None:
        """加载安装历史"""
        history_file = os.path.join(self.plugins_dir, ".history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.installation_history = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
    
    def _save_history(self) -> None:
        """保存安装历史"""
        history_file = os.path.join(self.plugins_dir, ".history.json")
        with open(history_file, 'w') as f:
            json.dump(self.installation_history, f, indent=2)
    
    def _calculate_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def install_from_source(self, 
                           source_path: str,
                           progress_callback: Optional[Callable] = None) -> Dict:
        """从源码安装Plugin"""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        
        # 读取plugin.json
        metadata_file = source / "plugin.json"
        if not metadata_file.exists():
            raise ValueError("plugin.json not found in source directory")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
        
        metadata = PluginMetadata.from_dict(metadata_dict)
        
        # 验证元数据
        errors = self.registry.validate_metadata(metadata)
        if errors:
            raise ValueError(f"Invalid metadata: {', '.join(errors)}")
        
        # 生成plugin_id
        if not metadata.plugin_id:
            metadata.plugin_id = self.registry.generate_plugin_id(metadata.name)
        
        # 目标目录
        target_dir = os.path.join(self.plugins_dir, metadata.plugin_id)
        if os.path.exists(target_dir):
            raise ValueError(f"Plugin {metadata.plugin_id} already installed")
        
        # 复制文件
        shutil.copytree(source, target_dir)
        
        # 创建checksum
        checksum_file = os.path.join(target_dir, ".checksum")
        with open(checksum_file, 'w') as f:
            f.write(self._calculate_checksum(metadata_file))
        
        # 注册plugin
        self.registry.register(metadata)
        
        # 记录历史
        self.installation_history[metadata.plugin_id] = {
            'installed_at': datetime.now().isoformat(),
            'source': source_path,
            'version': metadata.version
        }
        self._save_history()
        
        logger.info(f"Installed plugin: {metadata.name} v{metadata.version}")
        return {'status': 'success', 'plugin_id': metadata.plugin_id}
    
    def install_from_url(self, 
                        url: str,
                        progress_callback: Optional[Callable] = None) -> Dict:
        """从URL安装Plugin"""
        import urllib.request
        import urllib.error
        
        cache_path = os.path.join(self.cache_dir, f"plugin_{hash(url)}.zip")
        
        try:
            # 下载
            logger.info(f"Downloading plugin from {url}")
            urllib.request.urlretrieve(url, cache_path)
            
            # 提取到临时目录
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(cache_path, 'r') as zf:
                zf.extractall(temp_dir)
            
            # 安装
            result = self.install_from_source(temp_dir, progress_callback)
            
            # 清理临时目录
            shutil.rmtree(temp_dir)
            
            return result
        except urllib.error.URLError as e:
            raise ValueError(f"Failed to download plugin: {e}")
    
    def install_from_market(self, plugin_id: str) -> Dict:
        """从市场安装已注册的Plugin"""
        metadata = self.registry.get(plugin_id)
        if not metadata:
            raise ValueError(f"Plugin not found in registry: {plugin_id}")
        
        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if os.path.exists(target_dir):
            raise ValueError(f"Plugin already installed: {plugin_id}")
        
        # 标记为从市场安装
        os.makedirs(target_dir, exist_ok=True)
        
        # 保存元数据
        metadata_file = os.path.join(target_dir, "plugin.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # 记录历史
        self.installation_history[plugin_id] = {
            'installed_at': datetime.now().isoformat(),
            'source': 'market',
            'version': metadata.version
        }
        self._save_history()
        
        logger.info(f"Installed plugin from market: {plugin_id}")
        return {'status': 'success', 'plugin_id': plugin_id}
    
    def uninstall(self, plugin_id: str, remove_data: bool = True) -> Dict:
        """卸载Plugin"""
        target_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.exists(target_dir):
            raise ValueError(f"Plugin not installed: {plugin_id}")
        
        # 注销
        self.registry.unregister(plugin_id)
        
        # 删除文件
        if remove_data:
            shutil.rmtree(target_dir)
        else:
            # 重命名以保留数据
            backup_dir = os.path.join(self.plugins_dir, f".{plugin_id}_backup")
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            os.rename(target_dir, backup_dir)
        
        # 清理历史
        if plugin_id in self.installation_history:
            del self.installation_history[plugin_id]
            self._save_history()
        
        logger.info(f"Uninstalled plugin: {plugin_id}")
        return {'status': 'success', 'plugin_id': plugin_id}
    
    def update(self, plugin_id: str, new_version: str = None) -> Dict:
        """更新Plugin"""
        installed_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.exists(installed_dir):
            raise ValueError(f"Plugin not installed: {plugin_id}")
        
        metadata_file = os.path.join(installed_dir, "plugin.json")
        with open(metadata_file, 'r') as f:
            current_metadata = PluginMetadata.from_dict(json.load(f))
        
        # 获取最新元数据
        latest_metadata = self.registry.get(plugin_id)
        if not latest_metadata:
            raise ValueError(f"Plugin not found in registry: {plugin_id}")
        
        if new_version:
            # 查找指定版本
            # 这里简化处理，实际应该查询版本历史
            pass
        
        # 更新
        shutil.copy(
            os.path.join(self.plugins_dir, plugin_id, "plugin.json"),
            metadata_file
        )
        
        logger.info(f"Updated plugin: {plugin_id}")
        return {'status': 'success', 'plugin_id': plugin_id}
    
    def list_installed(self) -> List[Dict]:
        """列出已安装的Plugin"""
        installed = []
        for plugin_id in os.listdir(self.plugins_dir):
            if plugin_id.startswith('.'):
                continue
            
            plugin_dir = os.path.join(self.plugins_dir, plugin_id)
            metadata_file = os.path.join(plugin_dir, "plugin.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = PluginMetadata.from_dict(json.load(f))
                
                history = self.installation_history.get(plugin_id, {})
                installed.append({
                    'metadata': metadata.to_dict(),
                    'installed_at': history.get('installed_at'),
                    'source': history.get('source', 'market')
                })
        
        return installed
    
    def get_plugin_path(self, plugin_id: str) -> Optional[str]:
        """获取Plugin安装路径"""
        plugin_dir = os.path.join(self.plugins_dir, plugin_id)
        if os.path.exists(plugin_dir):
            return plugin_dir
        return None
    
    def is_installed(self, plugin_id: str) -> bool:
        """检查Plugin是否已安装"""
        plugin_dir = os.path.join(self.plugins_dir, plugin_id)
        return os.path.exists(plugin_dir)
    
    def enable(self, plugin_id: str) -> Dict:
        """启用Plugin"""
        plugin_dir = os.path.join(self.plugins_dir, plugin_id)
        if not os.path.exists(plugin_dir):
            raise ValueError(f"Plugin not installed: {plugin_id}")
        
        enabled_file = os.path.join(plugin_dir, ".enabled")
        with open(enabled_file, 'w') as f:
            f.write(datetime.now().isoformat())
        
        logger.info(f"Enabled plugin: {plugin_id}")
        return {'status': 'success', 'plugin_id': plugin_id}
    
    def disable(self, plugin_id: str) -> Dict:
        """禁用Plugin"""
        plugin_dir = os.path.join(self.plugins_dir, plugin_id)
        enabled_file = os.path.join(plugin_dir, ".enabled")
        
        if os.path.exists(enabled_file):
            os.remove(enabled_file)
        
        logger.info(f"Disabled plugin: {plugin_id}")
        return {'status': 'success', 'plugin_id': plugin_id}
    
    def is_enabled(self, plugin_id: str) -> bool:
        """检查Plugin是否启用"""
        plugin_dir = os.path.join(self.plugins_dir, plugin_id)
        enabled_file = os.path.join(plugin_dir, ".enabled")
        return os.path.exists(enabled_file)


# 全局管理器实例（延迟初始化）
plugin_manager = None

def get_plugin_manager() -> PluginManager:
    """获取全局插件管理器实例"""
    global plugin_manager
    if plugin_manager is None:
        plugin_manager = PluginManager(registry)
    return plugin_manager
