"""
Plugin Publisher - Plugin发布器
负责Plugin的发布、版本管理和打包
"""

import os
import json
import zipfile
import hashlib
import shutil
import tarfile
import tempfile
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import re

from .models import PublishResult, PluginInfo
from .manifest import PluginManifest, ManifestParser, ManifestValidator
from .registry import PluginRegistry

logger = logging.getLogger(__name__)


class VersionManager:
    """版本管理器"""
    
    @staticmethod
    def parse_version(version: str) -> tuple:
        """
        解析版本号为整数元组
        
        Args:
            version: 版本字符串
        
        Returns:
            (major, minor, patch, prerelease, build)
        """
        # 移除v前缀
        version = version.lstrip('v')
        
        # 分割主版本和预发布/构建元数据
        parts = version.split('-')
        version_parts = parts[0].split('.')
        
        major = int(version_parts[0]) if len(version_parts) > 0 else 0
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        patch = int(version_parts[2]) if len(version_parts) > 2 else 0
        
        prerelease = parts[1] if len(parts) > 1 else ""
        build = parts[2] if len(parts) > 2 else ""
        
        return (major, minor, patch, prerelease, build)
    
    @staticmethod
    def compare_versions(v1: str, v2: str) -> int:
        """
        比较两个版本
        
        Args:
            v1: 版本1
            v2: 版本2
        
        Returns:
            -1 if v1 < v2, 0 if equal, 1 if v1 > v2
        """
        p1 = VersionManager.parse_version(v1)
        p2 = VersionManager.parse_version(v2)
        
        # 比较主版本
        for i in range(3):
            if p1[i] > p2[i]:
                return 1
            if p1[i] < p2[i]:
                return -1
        
        # 比较预发布
        if p1[3] and not p2[3]:
            return -1
        if not p1[3] and p2[3]:
            return 1
        if p1[3] > p2[3]:
            return 1
        if p1[3] < p2[3]:
            return -1
        
        return 0
    
    @staticmethod
    def bump_version(version: str, bump_type: str = "patch") -> str:
        """
        递增版本号
        
        Args:
            version: 当前版本
            bump_type: 递增类型 (major, minor, patch)
        
        Returns:
            新版本号
        """
        parts = VersionManager.parse_version(version)
        major, minor, patch, prerelease, build = parts
        
        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        
        new_version = f"{major}.{minor}.{patch}"
        if prerelease:
            new_version = f"{new_version}-{prerelease}"
        if build:
            new_version = f"{new_version}+{build}"
        
        return new_version


class PluginPublisher:
    """Plugin发布器"""
    
    def __init__(self,
                 registry: PluginRegistry,
                 publish_dir: str = "./data/plugins/published",
                 package_dir: str = "./data/plugins/packages"):
        self.registry = registry
        self.publish_dir = publish_dir
        self.package_dir = package_dir
        self.versions: Dict[str, Dict[str, Any]] = {}
        self.versions_file = os.path.join(publish_dir, "versions.json")
        
        os.makedirs(publish_dir, exist_ok=True)
        os.makedirs(package_dir, exist_ok=True)
        self._load_versions()
    
    def _load_versions(self) -> None:
        """加载版本历史"""
        if os.path.exists(self.versions_file):
            try:
                with open(self.versions_file, 'r', encoding='utf-8') as f:
                    self.versions = json.load(f)
                logger.info(f"Loaded version history for {len(self.versions)} plugins")
            except Exception as e:
                logger.error(f"Failed to load versions: {e}")
    
    def _save_versions(self) -> None:
        """保存版本历史"""
        with open(self.versions_file, 'w', encoding='utf-8') as f:
            json.dump(self.versions, f, indent=2, ensure_ascii=False)
    
    def validate_plugin(self, plugin_dir: str) -> tuple:
        """
        验证Plugin是否可发布
        
        Args:
            plugin_dir: Plugin目录
        
        Returns:
            (成功标志, PluginManifest或错误列表)
        """
        # 查找manifest
        manifest_path = ManifestParser.find_manifest(plugin_dir)
        if not manifest_path:
            return (False, ["No plugin.yaml found"])
        
        # 解析manifest
        try:
            manifest = ManifestParser.parse(manifest_path)
        except Exception as e:
            return (False, [f"Failed to parse manifest: {e}"])
        
        # 验证manifest
        errors = ManifestValidator.validate(manifest)
        if errors:
            return (False, errors)
        
        # 检查必要文件
        if manifest.entry_point:
            entry_path = os.path.join(plugin_dir, manifest.entry_point)
            if not os.path.exists(entry_path):
                return (False, [f"Entry point not found: {manifest.entry_point}"])
        
        return (True, manifest)
    
    def publish(self,
               plugin_dir: str,
               version: str = None,
               is_latest: bool = True,
               publish_notes: str = None) -> PublishResult:
        """
        发布Plugin
        
        Args:
            plugin_dir: Plugin目录
            version: 版本号
            is_latest: 是否设为最新版本
            publish_notes: 发布说明
        
        Returns:
            PublishResult
        """
        # 验证plugin
        valid, manifest_or_errors = self.validate_plugin(plugin_dir)
        
        if not valid:
            return PublishResult(
                success=False,
                message="Plugin validation failed",
                errors=manifest_or_errors
            )
        
        manifest = manifest_or_errors
        
        # 生成plugin id
        plugin_id = self.registry.generate_plugin_id(manifest.name)
        
        # 检查版本
        if version:
            manifest.version = version
        
        # 检查是否重复发布
        if plugin_id in self.versions:
            existing_versions = self.versions[plugin_id].get('versions', {})
            if manifest.version in existing_versions:
                return PublishResult(
                    success=False,
                    plugin_id=plugin_id,
                    version=manifest.version,
                    message=f"Version {manifest.version} already exists",
                    errors=[f"Version {manifest.version} has already been published"]
                )
        
        # 创建package
        package_path, checksum = self._create_package(plugin_id, manifest.version, plugin_dir)
        
        if not package_path:
            return PublishResult(
                success=False,
                plugin_id=plugin_id,
                message="Failed to create package",
                errors=["Failed to create plugin package"]
            )
        
        # 保存版本信息
        if plugin_id not in self.versions:
            self.versions[plugin_id] = {
                'name': manifest.name,
                'display_name': manifest.display_name,
                'description': manifest.description,
                'author': manifest.author,
                'category': manifest.category.value if hasattr(manifest.category, 'value') else manifest.category,
                'versions': {},
                'latest_version': manifest.version,
                'published_at': datetime.now().isoformat(),
            }
        
        self.versions[plugin_id]['versions'][manifest.version] = {
            'package_path': package_path,
            'checksum': checksum,
            'published_at': datetime.now().isoformat(),
            'publish_notes': publish_notes or "",
            'is_latest': is_latest,
        }
        
        if is_latest:
            # 取消之前的latest标记
            for v, info in self.versions[plugin_id]['versions'].items():
                if v != manifest.version:
                    info['is_latest'] = False
        
        self.versions[plugin_id]['latest_version'] = manifest.version
        self._save_versions()
        
        # 注册到registry
        self.registry.register(manifest, plugin_id)
        
        logger.info(f"Published plugin: {plugin_id} v{manifest.version}")
        
        return PublishResult(
            success=True,
            plugin_id=plugin_id,
            version=manifest.version,
            message=f"Plugin published successfully",
            errors=[]
        )
    
    def unpublish(self, plugin_id: str, version: str = None) -> PublishResult:
        """
        取消发布Plugin
        
        Args:
            plugin_id: Plugin ID
            version: 版本号，不指定则移除所有版本
        
        Returns:
            PublishResult
        """
        if plugin_id not in self.versions:
            return PublishResult(
                success=False,
                plugin_id=plugin_id,
                message="Plugin not found",
                errors=[f"Plugin {plugin_id} not found"]
            )
        
        plugin_info = self.versions[plugin_id]
        
        if version:
            if version not in plugin_info['versions']:
                return PublishResult(
                    success=False,
                    plugin_id=plugin_id,
                    message="Version not found",
                    errors=[f"Version {version} not found"]
                )
            
            # 删除版本package
            version_info = plugin_info['versions'][version]
            package_path = version_info.get('package_path')
            if package_path and os.path.exists(package_path):
                os.remove(package_path)
            
            del plugin_info['versions'][version]
            
            # 更新latest
            if version_info.get('is_latest') and plugin_info['versions']:
                new_latest = list(plugin_info['versions'].keys())[-1]
                plugin_info['versions'][new_latest]['is_latest'] = True
                plugin_info['latest_version'] = new_latest
            
            self._save_versions()
            
            return PublishResult(
                success=True,
                plugin_id=plugin_id,
                version=version,
                message=f"Version {version} unpublished"
            )
        else:
            # 删除所有版本
            for v, info in plugin_info['versions'].items():
                package_path = info.get('package_path')
                if package_path and os.path.exists(package_path):
                    os.remove(package_path)
            
            del self.versions[plugin_id]
            self._save_versions()
            
            # 从registry中移除
            self.registry.unregister(plugin_id)
            
            return PublishResult(
                success=True,
                plugin_id=plugin_id,
                message=f"Plugin {plugin_id} unpublished completely"
            )
    
    def update_version(self,
                      plugin_id: str,
                      new_version: str,
                      plugin_dir: str = None) -> PublishResult:
        """
        更新Plugin版本
        
        Args:
            plugin_id: Plugin ID
            new_version: 新版本号
            plugin_dir: 新版本目录
        
        Returns:
            PublishResult
        """
        if plugin_id not in self.versions:
            return PublishResult(
                success=False,
                plugin_id=plugin_id,
                message="Plugin not found",
                errors=[f"Plugin {plugin_id} not found"]
            )
        
        # 比较版本
        current_latest = self.versions[plugin_id]['latest_version']
        cmp = VersionManager.compare_versions(new_version, current_latest)
        
        if cmp < 0:
            return PublishResult(
                success=False,
                plugin_id=plugin_id,
                version=new_version,
                message="New version must be greater than current version",
                errors=[f"{new_version} is less than current version {current_latest}"]
            )
        
        # 如果提供了新目录，重新发布
        if plugin_dir:
            return self.publish(plugin_dir, version=new_version)
        else:
            # 只更新版本信息
            self.versions[plugin_id]['versions'][new_version] = {
                'package_path': self.versions[plugin_id]['versions'][current_latest]['package_path'],
                'checksum': self.versions[plugin_id]['versions'][current_latest]['checksum'],
                'published_at': datetime.now().isoformat(),
                'publish_notes': f"Version bump to {new_version}",
                'is_latest': True,
            }
            
            # 取消之前的latest
            for v, info in self.versions[plugin_id]['versions'].items():
                if v != new_version:
                    info['is_latest'] = False
            
            self.versions[plugin_id]['latest_version'] = new_version
            self._save_versions()
            
            return PublishResult(
                success=True,
                plugin_id=plugin_id,
                version=new_version,
                message=f"Plugin updated to version {new_version}"
            )
    
    def get_versions(self, plugin_id: str) -> Optional[Dict]:
        """
        获取Plugin的所有版本
        
        Args:
            plugin_id: Plugin ID
        
        Returns:
            版本信息字典
        """
        return self.versions.get(plugin_id)
    
    def get_latest_version(self, plugin_id: str) -> Optional[str]:
        """
        获取Plugin的最新版本号
        
        Args:
            plugin_id: Plugin ID
        
        Returns:
            最新版本号
        """
        if plugin_id in self.versions:
            return self.versions[plugin_id].get('latest_version')
        return None
    
    def get_package_info(self, plugin_id: str, version: str = None) -> Optional[Dict]:
        """
        获取Package信息
        
        Args:
            plugin_id: Plugin ID
            version: 版本号，不指定则获取latest
        
        Returns:
            Package信息
        """
        if plugin_id not in self.versions:
            return None
        
        plugin_info = self.versions[plugin_id]
        
        if not version:
            version = plugin_info['latest_version']
        
        if version not in plugin_info['versions']:
            return None
        
        return plugin_info['versions'][version]
    
    def list_published(self,
                       category: str = None,
                       author: str = None,
                       page: int = 1,
                       page_size: int = 20) -> Dict:
        """
        列出已发布的Plugin
        
        Args:
            category: 分类过滤
            author: 作者过滤
            page: 页码
            page_size: 每页数量
        
        Returns:
            Plugin列表和分页信息
        """
        results = []
        
        for plugin_id, info in self.versions.items():
            if category and info.get('category') != category:
                continue
            if author and info.get('author') != author:
                continue
            
            results.append({
                'plugin_id': plugin_id,
                'name': info['name'],
                'display_name': info['display_name'],
                'description': info['description'],
                'author': info['author'],
                'category': info['category'],
                'latest_version': info['latest_version'],
                'versions_count': len(info['versions']),
                'published_at': info['published_at'],
            })
        
        # 分页
        total = len(results)
        start = (page - 1) * page_size
        end = start + page_size
        
        return {
            'plugins': results[start:end],
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total': total,
                'total_pages': (total + page_size - 1) // page_size
            }
        }
    
    def _create_package(self,
                       plugin_id: str,
                       version: str,
                       plugin_dir: str) -> tuple:
        """
        创建Plugin安装包
        
        Args:
            plugin_id: Plugin ID
            version: 版本号
            plugin_dir: Plugin目录
        
        Returns:
            (package_path, checksum)
        """
        package_name = f"{plugin_id}-{version}"
        package_path = os.path.join(self.package_dir, f"{package_name}.zip")
        
        try:
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(plugin_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, plugin_dir)
                        zipf.write(file_path, arcname)
            
            # 计算checksum
            checksum = self._calculate_checksum(package_path)
            
            logger.info(f"Created package: {package_path}")
            return package_path, checksum
        
        except Exception as e:
            logger.error(f"Failed to create package: {e}")
            return None, None
    
    def _calculate_checksum(self, file_path: str) -> str:
        """
        计算文件checksum
        
        Args:
            file_path: 文件路径
        
        Returns:
            SHA256 checksum
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def download_package(self, plugin_id: str, version: str = None) -> Optional[str]:
        """
        获取Package下载路径
        
        Args:
            plugin_id: Plugin ID
            version: 版本号
        
        Returns:
            Package路径，不存在返回None
        """
        package_info = self.get_package_info(plugin_id, version)
        if package_info:
            return package_info.get('package_path')
        return None
