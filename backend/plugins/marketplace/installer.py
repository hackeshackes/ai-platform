"""
Plugin Installer - Plugin安装器
负责Plugin的安装、卸载、更新和依赖管理
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import venv

from .models import InstallResult, UninstallResult, InstallationInfo
from .manifest import PluginManifest, ManifestParser
from .registry import PluginRegistry

logger = logging.getLogger(__name__)


class DependencyManager:
    """依赖管理器"""
    
    @staticmethod
    def parse_dependencies(dependencies: Dict[str, str]) -> List[Tuple[str, str]]:
        """
        解析依赖列表
        
        Args:
            dependencies: 依赖字典 {package: version_spec}
        
        Returns:
            依赖列表 [(package, version_spec), ...]
        """
        return [(pkg, ver) for pkg, ver in dependencies.items()]
    
    @staticmethod
    def check_dependencies(dependencies: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """
        检查依赖是否已满足
        
        Args:
            dependencies: 依赖字典
        
        Returns:
            (满足的依赖, 缺失的依赖)
        """
        satisfied = []
        missing = []
        
        for pkg, version_spec in dependencies.items():
            try:
                import pkg_resources
                pkg_resources.working_set.require(f"{pkg}{version_spec}")
                satisfied.append(f"{pkg}{version_spec}")
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
                missing.append(f"{pkg}{version_spec}")
        
        return satisfied, missing
    
    @staticmethod
    def install_dependencies(dependencies: Dict[str, str],
                           install_path: str,
                           python_path: str = None) -> Tuple[bool, List[str]]:
        """
        安装依赖
        
        Args:
            dependencies: 依赖字典
            install_path: 安装路径
            python_path: Python路径
        
        Returns:
            (是否成功, 安装的依赖列表)
        """
        if not dependencies:
            return True, []
        
        pip_path = python_path or sys.executable
        
        cmd = [pip_path, "install", "-r", "-"]
        
        req_content = "\n".join(f"{pkg}{ver}" for pkg, ver in dependencies.items())
        
        try:
            result = subprocess.run(
                cmd,
                input=req_content,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=install_path
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to install dependencies: {result.stderr}")
                return False, []
            
            installed = list(dependencies.keys())
            logger.info(f"Installed dependencies: {installed}")
            return True, installed
        
        except subprocess.TimeoutExpired:
            logger.error("Dependency installation timed out")
            return False, []
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False, []
    
    @staticmethod
    def uninstall_dependencies(dependencies: Dict[str, str],
                               python_path: str = None) -> bool:
        """
        卸载依赖
        
        Args:
            dependencies: 依赖字典
            python_path: Python路径
        
        Returns:
            是否成功
        """
        if not dependencies:
            return True
        
        pip_path = python_path or sys.executable
        
        packages = list(dependencies.keys())
        
        try:
            result = subprocess.run(
                [pip_path, "uninstall", "-y"] + packages,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to uninstall some dependencies: {result.stderr}")
                return False
            
            logger.info(f"Uninstalled dependencies: {packages}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to uninstall dependencies: {e}")
            return False


class PluginInstaller:
    """Plugin安装器"""
    
    def __init__(self,
                 registry: PluginRegistry,
                 plugins_dir: str = "./data/plugins/installed",
                 python_path: str = None):
        self.registry = registry
        self.plugins_dir = plugins_dir
        self.python_path = python_path or sys.executable
        self.installations: Dict[str, InstallationInfo] = {}
        self.installations_file = os.path.join(plugins_dir, "installations.json")
        
        os.makedirs(plugins_dir, exist_ok=True)
        self._load_installations()
    
    def _load_installations(self) -> None:
        """加载安装信息"""
        if os.path.exists(self.installations_file):
            try:
                with open(self.installations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for inst_data in data.get('installations', []):
                        info = InstallationInfo(**inst_data)
                        self.installations[info.plugin_id] = info
                logger.info(f"Loaded {len(self.installations)} installed plugins")
            except Exception as e:
                logger.error(f"Failed to load installations: {e}")
    
    def _save_installations(self) -> None:
        """保存安装信息"""
        data = {
            'version': '1.0.0',
            'updated_at': datetime.now().isoformat(),
            'installations': [inst.to_dict() for inst in self.installations.values()]
        }
        with open(self.installations_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def install(self,
               plugin_id: str,
               source: str = None,
               version: str = None,
               install_path: str = None) -> InstallResult:
        """
        安装Plugin
        
        Args:
            plugin_id: Plugin ID
            source: 源路径或URL
            version: 指定版本
            install_path: 安装路径
        
        Returns:
            InstallResult
        """
        # 检查是否已安装
        if plugin_id in self.installations:
            existing = self.installations[plugin_id]
            return InstallResult(
                success=False,
                plugin_id=plugin_id,
                message=f"Plugin already installed (version: {existing.version})",
                errors=[f"Plugin {plugin_id} is already installed"]
            )
        
        # 获取plugin信息
        plugin = self.registry.get(plugin_id)
        if not plugin:
            return InstallResult(
                success=False,
                plugin_id=plugin_id,
                message="Plugin not found in registry",
                errors=[f"Plugin {plugin_id} not found"]
            )
        
        # 解析manifest
        if not source:
            source = os.path.join(self.plugins_dir, plugin_id)
        
        manifest_path = ManifestParser.find_manifest(source)
        if not manifest_path:
            return InstallResult(
                success=False,
                plugin_id=plugin_id,
                message="Plugin manifest not found",
                errors=["plugin.yaml not found"]
            )
        
        try:
            manifest = ManifestParser.parse(manifest_path)
        except Exception as e:
            return InstallResult(
                success=False,
                plugin_id=plugin_id,
                message=f"Failed to parse manifest: {e}",
                errors=[str(e)]
            )
        
        # 创建安装目录
        if not install_path:
            install_path = os.path.join(self.plugins_dir, plugin_id)
        
        os.makedirs(install_path, exist_ok=True)
        
        # 复制plugin文件
        try:
            self._copy_plugin_files(source, install_path)
        except Exception as e:
            return InstallResult(
                success=False,
                plugin_id=plugin_id,
                message=f"Failed to copy plugin files: {e}",
                errors=[str(e)]
            )
        
        # 安装依赖
        if manifest.dependencies:
            success, installed = DependencyManager.install_dependencies(
                manifest.dependencies,
                install_path,
                self.python_path
            )
            if not success:
                # 清理已复制的文件
                if os.path.exists(install_path):
                    shutil.rmtree(install_path)
                return InstallResult(
                    success=False,
                    plugin_id=plugin_id,
                    message="Failed to install dependencies",
                    errors=["Dependency installation failed"]
                )
        
        # 创建安装信息
        installation = InstallationInfo(
            plugin_id=plugin_id,
            version=manifest.version,
            installed_at=datetime.now().isoformat(),
            install_path=install_path,
            is_active=True,
            config={}
        )
        self.installations[plugin_id] = installation
        
        # 保存安装信息
        self._save_installations()
        
        # 更新registry中的状态
        self.registry.update(plugin_id, status="installed", installed_version=manifest.version)
        
        # 执行on_install hook
        if manifest.hooks.on_install:
            self._execute_hook(manifest.hooks.on_install, install_path, "on_install")
        
        logger.info(f"Installed plugin: {plugin_id} v{manifest.version}")
        
        return InstallResult(
            success=True,
            plugin_id=plugin_id,
            version=manifest.version,
            message=f"Plugin installed successfully",
            dependencies=list(manifest.dependencies.keys())
        )
    
    def uninstall(self, plugin_id: str, remove_dependencies: bool = True) -> UninstallResult:
        """
        卸载Plugin
        
        Args:
            plugin_id: Plugin ID
            remove_dependencies: 是否移除依赖
        
        Returns:
            UninstallResult
        """
        if plugin_id not in self.installations:
            return UninstallResult(
                success=False,
                plugin_id=plugin_id,
                message="Plugin not installed",
                errors=[f"Plugin {plugin_id} is not installed"]
            )
        
        installation = self.installations[plugin_id]
        
        # 执行on_uninstall hook
        plugin = self.registry.get(plugin_id)
        if plugin:
            manifest_path = os.path.join(installation.install_path, "plugin.yaml")
            if os.path.exists(manifest_path):
                try:
                    manifest = ManifestParser.parse(manifest_path)
                    if manifest.hooks.on_uninstall:
                        self._execute_hook(manifest.hooks.on_uninstall, installation.install_path, "on_uninstall")
                except Exception as e:
                    logger.warning(f"Failed to execute on_uninstall hook: {e}")
        
        # 卸载依赖
        if remove_dependencies and plugin:
            if plugin.dependencies:
                DependencyManager.uninstall_dependencies(
                    plugin.dependencies,
                    self.python_path
                )
        
        # 移除plugin文件
        if os.path.exists(installation.install_path):
            shutil.rmtree(installation.install_path)
        
        # 移除安装信息
        del self.installations[plugin_id]
        self._save_installations()
        
        # 更新registry中的状态
        self.registry.update(plugin_id, status="available", installed_version=None)
        
        logger.info(f"Uninstalled plugin: {plugin_id}")
        
        return UninstallResult(
            success=True,
            plugin_id=plugin_id,
            message="Plugin uninstalled successfully"
        )
    
    def update(self, plugin_id: str, source: str = None) -> InstallResult:
        """
        更新Plugin
        
        Args:
            plugin_id: Plugin ID
            source: 新版本源路径
        
        Returns:
            InstallResult
        """
        if plugin_id not in self.installations:
            return InstallResult(
                success=False,
                plugin_id=plugin_id,
                message="Plugin not installed",
                errors=[f"Plugin {plugin_id} is not installed"]
            )
        
        current_version = self.installations[plugin_id].version
        
        # 先卸载旧版本
        uninstall_result = self.uninstall(plugin_id, remove_dependencies=False)
        if not uninstall_result.success:
            return InstallResult(
                success=False,
                plugin_id=plugin_id,
                message=f"Failed to uninstall old version: {uninstall_result.message}",
                errors=uninstall_result.errors
            )
        
        # 安装新版本
        install_result = self.install(plugin_id, source=source)
        
        if not install_result.success:
            # 回滚：重新安装旧版本
            self.install(plugin_id, source=self.installations.get(plugin_id, {}).get('install_path') if False else None)
            return InstallResult(
                success=False,
                plugin_id=plugin_id,
                message=f"Failed to install new version, rolled back",
                errors=install_result.errors
            )
        
        logger.info(f"Updated plugin: {plugin_id} from {current_version} to {install_result.version}")
        
        return InstallResult(
            success=True,
            plugin_id=plugin_id,
            version=install_result.version,
            message=f"Plugin updated from {current_version} to {install_result.version}"
        )
    
    def is_installed(self, plugin_id: str) -> bool:
        """检查Plugin是否已安装"""
        return plugin_id in self.installations
    
    def get_installation(self, plugin_id: str) -> Optional[InstallationInfo]:
        """获取安装信息"""
        return self.installations.get(plugin_id)
    
    def list_installed(self) -> List[InstallationInfo]:
        """列出所有已安装的Plugin"""
        return list(self.installations.values())
    
    def activate(self, plugin_id: str) -> bool:
        """
        激活Plugin
        
        Args:
            plugin_id: Plugin ID
        
        Returns:
            是否成功
        """
        if plugin_id not in self.installations:
            return False
        
        installation = self.installations[plugin_id]
        installation.is_active = True
        self._save_installations()
        
        # 执行on_activate hook
        plugin = self.registry.get(plugin_id)
        if plugin:
            manifest_path = os.path.join(installation.install_path, "plugin.yaml")
            if os.path.exists(manifest_path):
                try:
                    manifest = ManifestParser.parse(manifest_path)
                    if manifest.hooks.on_activate:
                        self._execute_hook(manifest.hooks.on_activate, installation.install_path, "on_activate")
                except Exception as e:
                    logger.warning(f"Failed to execute on_activate hook: {e}")
        
        logger.info(f"Activated plugin: {plugin_id}")
        return True
    
    def deactivate(self, plugin_id: str) -> bool:
        """
        停用Plugin
        
        Args:
            plugin_id: Plugin ID
        
        Returns:
            是否成功
        """
        if plugin_id not in self.installations:
            return False
        
        installation = self.installations[plugin_id]
        installation.is_active = False
        self._save_installations()
        
        # 执行on_deactivate hook
        plugin = self.registry.get(plugin_id)
        if plugin:
            manifest_path = os.path.join(installation.install_path, "plugin.yaml")
            if os.path.exists(manifest_path):
                try:
                    manifest = ManifestParser.parse(manifest_path)
                    if manifest.hooks.on_deactivate:
                        self._execute_hook(manifest.hooks.on_deactivate, installation.install_path, "on_deactivate")
                except Exception as e:
                    logger.warning(f"Failed to execute on_deactivate hook: {e}")
        
        logger.info(f"Deactivated plugin: {plugin_id}")
        return True
    
    def _copy_plugin_files(self, source: str, dest: str) -> None:
        """
        复制Plugin文件
        
        Args:
            source: 源目录
            dest: 目标目录
        """
        if os.path.isdir(source):
            shutil.copytree(source, dest, dirs_exist_ok=False)
        else:
            os.makedirs(dest, exist_ok=True)
            shutil.copy2(source, os.path.join(dest, os.path.basename(source)))
    
    def _execute_hook(self, hook_func: str, plugin_path: str, hook_name: str) -> bool:
        """
        执行hook函数
        
        Args:
            hook_func: hook函数名
            plugin_path: plugin路径
            hook_name: hook名称
        
        Returns:
            是否成功
        """
        hook_file = os.path.join(plugin_path, "hooks.py")
        
        if not os.path.exists(hook_file):
            logger.warning(f"Hook file not found: {hook_file}")
            return False
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("hooks", hook_file)
            hooks_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hooks_module)
            
            if hasattr(hooks_module, hook_func):
                hook_func_ref = getattr(hooks_module, hook_func)
                hook_func_ref()
                logger.info(f"Executed {hook_name} for {plugin_path}")
                return True
            else:
                logger.warning(f"Hook function {hook_func} not found in hooks.py")
                return False
        
        except Exception as e:
            logger.error(f"Failed to execute hook {hook_func}: {e}")
            return False
