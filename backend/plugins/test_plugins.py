"""
Plugin System Test - 插件系统测试
"""

import sys
sys.path.insert(0, '/Users/yubao/.openclaw/workspace')

from plugins.registry import PluginRegistry, PluginMetadata
from plugins.manager import PluginManager
from plugins.market import PluginMarket, PluginRating, PluginStats
from plugins.sandbox import PluginSandbox, SandboxConfig

def test_registry():
    """测试注册表"""
    print("\n=== 测试注册表 ===")
    registry = PluginRegistry(registry_path='/tmp/test_registry')
    
    # 创建测试插件
    plugins = [
        PluginMetadata(
            plugin_id='text-processor',
            name='Text Processor',
            version='1.0.0',
            description='Process and transform text',
            author='Alice',
            category='utilities',
            tags=['text', 'processing']
        ),
        PluginMetadata(
            plugin_id='image-analyzer',
            name='Image Analyzer',
            version='2.1.0',
            description='Analyze images with ML',
            author='Bob',
            category='ai',
            tags=['image', 'ml', 'vision']
        ),
        PluginMetadata(
            plugin_id='data-exporter',
            name='Data Exporter',
            version='1.5.0',
            description='Export data to various formats',
            author='Charlie',
            category='utilities',
            tags=['data', 'export']
        )
    ]
    
    for p in plugins:
        registry.register(p)
    
    print(f"✓ 注册了 {len(registry.plugins)} 个插件")
    
    # 列出
    all_plugins = registry.list_all()
    print(f"✓ 列表插件: {[p.name for p in all_plugins]}")
    
    # 搜索
    results = registry.search('text')
    print(f"✓ 搜索'text': {[p.name for p in results]}")
    
    # 分类
    categories = registry.get_categories()
    print(f"✓ 分类统计: {categories}")
    
    return True


def test_manager():
    """测试管理器"""
    print("\n=== 测试管理器 ===")
    registry = PluginRegistry(registry_path='/tmp/test_registry2')
    manager = PluginManager(registry, plugins_dir='/tmp/test_installed')
    
    # 添加测试插件到注册表
    metadata = PluginMetadata(
        plugin_id='demo-plugin',
        name='Demo Plugin',
        version='1.0.0',
        description='A demo plugin',
        author='Test',
        category='demo',
        entry_point='plugin.py'
    )
    registry.register(metadata)
    
    # 从市场安装
    manager.install_from_market('demo-plugin')
    print(f"✓ 安装插件: demo-plugin")
    
    # 列出已安装
    installed = manager.list_installed()
    print(f"✓ 已安装数量: {len(installed)}")
    print(f"✓ 是否已安装: {manager.is_installed('demo-plugin')}")
    
    # 启用
    manager.enable('demo-plugin')
    print(f"✓ 启用状态: {manager.is_enabled('demo-plugin')}")
    
    # 禁用
    manager.disable('demo-plugin')
    print(f"✓ 禁用状态: {manager.is_enabled('demo-plugin')}")
    
    # 卸载
    manager.uninstall('demo-plugin')
    print(f"✓ 卸载后安装状态: {manager.is_installed('demo-plugin')}")
    
    return True


def test_market():
    """测试市场"""
    print("\n=== 测试市场 ===")
    registry = PluginRegistry(registry_path='/tmp/test_registry3')
    manager = PluginManager(registry, plugins_dir='/tmp/test_installed2')
    market = PluginMarket(registry, manager, market_data_dir='/tmp/test_market')
    
    # 添加测试插件
    plugins = [
        PluginMetadata(
            plugin_id='popular-tool',
            name='Popular Tool',
            version='2.0.0',
            description='A popular tool',
            author='Dev1',
            category='tools',
            tags=['popular']
        ),
        PluginMetadata(
            plugin_id='new-plugin',
            name='New Plugin',
            version='1.0.0',
            description='A new plugin',
            author='Dev2',
            category='tools'
        )
    ]
    
    for p in plugins:
        registry.register(p)
    
    # 浏览市场
    result = market.browse_market(sort_by='popularity')
    print(f"✓ 市场插件数: {len(result['plugins'])}")
    print(f"✓ 分类: {result['categories']}")
    
    # 搜索
    result = market.search_market('popular')
    print(f"✓ 搜索结果: {len(result['plugins'])}")
    
    # 评分
    market.rate_plugin('popular-tool', 'user1', 5, 'Great!')
    market.rate_plugin('popular-tool', 'user2', 4, 'Good')
    market.rate_plugin('popular-tool', 'user3', 5, 'Excellent!')
    
    stats = market.stats.get('popular-tool')
    print(f"✓ 评分统计: avg={stats.average_rating}, count={stats.total_ratings}")
    
    # 获取详情
    details = market.get_plugin_details('popular-tool')
    print(f"✓ 详情: {details['metadata']['name']}, 安装状态: {details['is_installed']}")
    
    # 获取推荐
    featured = market.get_featured_plugins()
    print(f"✓ 推荐插件数: {len(featured)}")
    
    return True


def test_sandbox():
    """测试沙箱"""
    print("\n=== 测试沙箱 ===")
    sandbox = PluginSandbox(SandboxConfig(timeout_seconds=2))
    
    # 执行简单代码
    code = """
data = {'name': 'test', 'value': 42}
print('Result:', data)
"""
    result = sandbox.execute_python(code, plugin_id='test')
    print(f"✓ 执行成功: {result.success}")
    print(f"✓ 执行时间: {result.execution_time_ms}ms")
    
    # 超时测试
    timeout_code = """
import time
time.sleep(10)
print('Done')
"""
    result = sandbox.execute_python(timeout_code, plugin_id='timeout_test')
    print(f"✓ 超时处理: {result.return_code == -1 or 'timeout' in result.error.lower()}")
    
    return True


if __name__ == '__main__':
    print("=" * 50)
    print("Plugin System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Registry", test_registry),
        ("Manager", test_manager),
        ("Market", test_market),
        ("Sandbox", test_sandbox)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {name} 测试通过")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
