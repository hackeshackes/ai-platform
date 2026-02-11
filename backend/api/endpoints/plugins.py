"""
Plugin API Endpoints - Plugin API端点
提供RESTful API接口
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from flask import Blueprint, request, jsonify, send_file
import logging

from .registry import PluginRegistry, PluginMetadata
from .manager import PluginManager
from .market import PluginMarket
from .sandbox import PluginSandbox, SandboxConfig

logger = logging.getLogger(__name__)

# 创建Blueprint
plugins_bp = Blueprint('plugins', __name__, url_prefix='/api/v1/plugins')

# 全局实例（由app初始化时注入）
registry: Optional[PluginRegistry] = None
manager: Optional[PluginManager] = None
market: Optional[PluginMarket] = None
sandbox: Optional[PluginSandbox] = None


def init_app(app):
    """初始化插件模块"""
    global registry, manager, market, sandbox
    
    registry = PluginRegistry()
    manager = PluginManager(registry)
    market = PluginMarket(registry, manager)
    sandbox = PluginSandbox()
    
    logger.info("Plugin API initialized")


def require_plugins(func):
    """依赖检查装饰器"""
    def wrapper(*args, **kwargs):
        if not all([registry, manager, market]):
            return jsonify({'error': 'Plugin system not initialized'}), 500
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


# ============ 市场API ============

@plugins_bp.route('/market', methods=['GET'])
@require_plugins
def browse_market():
    """浏览市场"""
    category = request.args.get('category')
    sort_by = request.args.get('sort_by', 'popularity')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 20))
    
    result = market.browse_market(
        category=category,
        sort_by=sort_by,
        page=page,
        page_size=page_size
    )
    
    return jsonify(result)


@plugins_bp.route('/search', methods=['GET'])
@require_plugins
def search_plugins():
    """搜索插件"""
    query = request.args.get('q', '')
    category = request.args.get('category')
    min_rating = request.args.get('min_rating', type=float)
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 20))
    
    result = market.search_market(
        query=query,
        category=category,
        min_rating=min_rating,
        page=page,
        page_size=page_size
    )
    
    return jsonify(result)


@plugins_bp.route('/market/<plugin_id>', methods=['GET'])
@require_plugins
def get_market_plugin(plugin_id: str):
    """获取市场插件详情"""
    details = market.get_plugin_details(plugin_id)
    
    if not details:
        return jsonify({'error': 'Plugin not found'}), 404
    
    return jsonify(details)


@plugins_bp.route('/featured', methods=['GET'])
@require_plugins
def get_featured():
    """获取推荐插件"""
    return jsonify({'plugins': market.get_featured_plugins()})


# ============ 已安装插件API ============

@plugins_bp.route('/installed', methods=['GET'])
@require_plugins
def list_installed():
    """列出已安装插件"""
    installed = manager.list_installed()
    return jsonify({'plugins': installed})


@plugins_bp.route('/installed/<plugin_id>', methods=['GET'])
@require_plugins
def get_installed_plugin(plugin_id: str):
    """获取已安装插件详情"""
    path = manager.get_plugin_path(plugin_id)
    
    if not path:
        return jsonify({'error': 'Plugin not installed'}), 404
    
    metadata = registry.get(plugin_id)
    is_enabled = manager.is_enabled(plugin_id)
    
    return jsonify({
        'plugin_id': plugin_id,
        'metadata': metadata.to_dict() if metadata else None,
        'path': path,
        'enabled': is_enabled
    })


# ============ 安装/卸载API ============

@plugins_bp.route('/install', methods=['POST'])
@require_plugins
def install_plugin():
    """安装插件"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    source_type = data.get('type', 'market')  # market, url, source
    
    try:
        if source_type == 'market':
            plugin_id = data.get('plugin_id')
            if not plugin_id:
                return jsonify({'error': 'plugin_id required'}), 400
            result = manager.install_from_market(plugin_id)
        
        elif source_type == 'url':
            url = data.get('url')
            if not url:
                return jsonify({'error': 'url required'}), 400
            result = manager.install_from_url(url)
        
        elif source_type == 'source':
            source_path = data.get('source_path')
            if not source_path:
                return jsonify({'error': 'source_path required'}), 400
            result = manager.install_from_source(source_path)
        
        else:
            return jsonify({'error': f'Invalid source type: {source_type}'}), 400
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Install failed: {e}")
        return jsonify({'error': str(e)}), 400


@plugins_bp.route('/uninstall', methods=['POST'])
@require_plugins
def uninstall_plugin():
    """卸载插件"""
    data = request.get_json()
    
    if not data or 'plugin_id' not in data:
        return jsonify({'error': 'plugin_id required'}), 400
    
    plugin_id = data['plugin_id']
    remove_data = data.get('remove_data', True)
    
    try:
        result = manager.uninstall(plugin_id, remove_data=remove_data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Uninstall failed: {e}")
        return jsonify({'error': str(e)}), 400


@plugins_bp.route('/enable', methods=['POST'])
@require_plugins
def enable_plugin():
    """启用插件"""
    data = request.get_json()
    
    if not data or 'plugin_id' not in data:
        return jsonify({'error': 'plugin_id required'}), 400
    
    try:
        result = manager.enable(data['plugin_id'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@plugins_bp.route('/disable', methods=['POST'])
@require_plugins
def disable_plugin():
    """禁用插件"""
    data = request.get_json()
    
    if not data or 'plugin_id' not in data:
        return jsonify({'error': 'plugin_id required'}), 400
    
    try:
        result = manager.disable(data['plugin_id'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ============ 发布API ============

@plugins_bp.route('/publish', methods=['POST'])
@require_plugins
def publish_plugin():
    """发布插件到市场"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    required_fields = ['name', 'version', 'description', 'author', 'category']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    try:
        # 创建元数据
        metadata = PluginMetadata(
            plugin_id=data.get('plugin_id', ''),
            name=data['name'],
            version=data['version'],
            description=data['description'],
            author=data['author'],
            category=data['category'],
            tags=data.get('tags', []),
            dependencies=data.get('dependencies', {}),
            permissions=data.get('permissions', []),
            entry_point=data.get('entry_point', 'plugin.py'),
            homepage=data.get('homepage', ''),
            repository=data.get('repository', ''),
            license=data.get('license', 'MIT')
        )
        
        # 验证
        errors = registry.validate_metadata(metadata)
        if errors:
            return jsonify({'error': 'Validation failed', 'details': errors}), 400
        
        # 生成plugin_id
        if not metadata.plugin_id:
            metadata.plugin_id = registry.generate_plugin_id(metadata.name)
        
        # 注册到市场
        registry.register(metadata)
        
        return jsonify({
            'status': 'success',
            'plugin_id': metadata.plugin_id,
            'message': 'Plugin published successfully'
        })
    
    except Exception as e:
        logger.error(f"Publish failed: {e}")
        return jsonify({'error': str(e)}), 400


# ============ 评分API ============

@plugins_bp.route('/rate', methods=['POST'])
@require_plugins
def rate_plugin():
    """评分插件"""
    data = request.get_json()
    
    required_fields = ['plugin_id', 'user_id', 'rating']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    try:
        result = market.rate_plugin(
            plugin_id=data['plugin_id'],
            user_id=data['user_id'],
            rating=data['rating'],
            review=data.get('review')
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


# ============ 沙箱执行API ============

@plugins_bp.route('/execute', methods=['POST'])
@require_plugins
def execute_plugin():
    """在沙箱中执行代码"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Request body required'}), 400
    
    code = data.get('code')
    plugin_id = data.get('plugin_id')
    function_name = data.get('function', 'main')
    context = data.get('context', {})
    
    try:
        if plugin_id:
            # 执行已安装的插件
            result = sandbox.execute_plugin(
                plugin_id=plugin_id,
                function_name=function_name,
                args=data.get('args', []),
                kwargs=data.get('kwargs', {})
            )
        elif code:
            # 执行临时代码
            result = sandbox.execute_python(
                code=code,
                plugin_id=data.get('plugin_id', 'anonymous'),
                context=context
            )
        else:
            return jsonify({'error': 'code or plugin_id required'}), 400
        
        return jsonify({
            'success': result.success,
            'output': result.output,
            'error': result.error,
            'return_code': result.return_code,
            'execution_time_ms': result.execution_time_ms
        })
    
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return jsonify({'error': str(e)}), 400


# ============ 下载API ============

@plugins_bp.route('/download/<plugin_id>', methods=['GET'])
@require_plugins
def download_plugin(plugin_id: str):
    """下载插件"""
    try:
        download_info = market.download_plugin(plugin_id)
        plugin_dir = manager.get_plugin_path(plugin_id)
        
        if not plugin_dir:
            return jsonify({'error': 'Plugin not found'}), 404
        
        # 返回插件目录下的文件列表或打包下载
        import zipfile
        import os
        import tempfile
        
        zip_path = os.path.join(tempfile.gettempdir(), f"{plugin_id}.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(plugin_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, plugin_dir)
                    zf.write(file_path, arcname)
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"{plugin_id}.zip"
        )
    
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return jsonify({'error': str(e)}), 400


# ============ 统计API ============

@plugins_bp.route('/stats', methods=['GET'])
@require_plugins
def get_stats():
    """获取市场统计"""
    categories = registry.get_categories()
    installed_count = len(manager.list_installed())
    
    return jsonify({
        'total_plugins': len(registry.plugins),
        'installed_plugins': installed_count,
        'categories': categories
    })
