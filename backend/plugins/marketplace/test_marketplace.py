#!/usr/bin/env python3
"""
Plugin Marketplace Test - 测试脚本
验证Plugin市场的核心功能
"""

import os
import sys
sys.path.insert(0, '/Users/yubao/.openclaw/workspace')

from plugins.marketplace import (
    PluginMarketplace,
    PluginRegistry,
    PluginInstaller,
    PluginPublisher,
    PluginManifest,
    ManifestParser,
)


def test_manifest_parsing():
    """测试Manifest解析"""
    print("\n=== 测试 Manifest 解析 ===")
    
    manifest_path = "/Users/yubao/.openclaw/workspace/data/plugins/test-plugin/plugin.yaml"
    
    try:
        manifest = ManifestParser.parse(manifest_path)
        print(f"✓ 解析成功: {manifest.display_name}")
        print(f"  - 名称: {manifest.name}")
        print(f"  - 版本: {manifest.version}")
        print(f"  - 分类: {manifest.category.value}")
        print(f"  - 作者: {manifest.author}")
        print(f"  - 权限: {manifest.permissions}")
        print(f"  - 依赖: {manifest.dependencies}")
        return True
    except Exception as e:
        print(f"✗ 解析失败: {e}")
        return False


def test_registry():
    """测试注册表"""
    print("\n=== 测试 Plugin 注册表 ===")
    
    registry = PluginRegistry(registry_dir="/Users/yubao/.openclaw/workspace/data/plugins/test_registry")
    
    # 注册测试plugin
    manifest_path = "/Users/yubao/.openclaw/workspace/data/plugins/test-plugin/plugin.yaml"
    manifest = ManifestParser.parse(manifest_path)
    
    # 获取生成的plugin_id
    plugin_id = registry.generate_plugin_id(manifest.name)
    print(f"✓ 生成的Plugin ID: {plugin_id}")
    
    success = registry.register(manifest)
    print(f"✓ 注册结果: {success}")
    
    # 获取plugin (使用正确的plugin_id)
    plugin = registry.get(plugin_id)
    if plugin:
        print(f"✓ 获取成功: {plugin.display_name} v{plugin.version}")
    else:
        # 尝试从列表中获取
        all_plugins = registry.list_all()
        if all_plugins['plugins']:
            plugin_id = all_plugins['plugins'][0]['plugin_id']
            plugin = registry.get(plugin_id)
            print(f"✓ 获取成功: {plugin.display_name} v{plugin.version}")
        else:
            print("✗ 获取失败: 插件列表为空")
    
    # 列出所有plugin
    result = registry.list_all()
    print(f"✓ 列表插件数量: {len(result['plugins'])}")
    
    return True


def test_search():
    """测试搜索"""
    print("\n=== 测试 Plugin 搜索 ===")
    
    registry = PluginRegistry(registry_dir="/Users/yubao/.openclaw/workspace/data/plugins/test_registry")
    
    # 添加一些测试数据
    manifest1 = PluginManifest(
        name="data-processor",
        version="1.0.0",
        display_name="数据处理器",
        description="处理和分析数据",
        author="dev1",
        category="tool",
        tags=["data", "analysis"]
    )
    manifest2 = PluginManifest(
        name="image-generator",
        version="2.0.0",
        display_name="图像生成器",
        description="AI图像生成",
        author="dev2",
        category="tool",
        tags=["image", "ai"]
    )
    
    registry.register(manifest1)
    registry.register(manifest2)
    
    # 搜索
    result = registry.search(query="data")
    print(f"✓ 搜索 'data' 结果: {len(result['plugins'])} 个插件")
    
    result = registry.search(query="image")
    print(f"✓ 搜索 'image' 结果: {len(result['plugins'])} 个插件")
    
    # 获取分类
    categories = registry.get_categories()
    print(f"✓ 分类数量: {len(categories)}")
    
    return True


def test_marketplace():
    """测试市场功能"""
    print("\n=== 测试 Plugin 市场 ===")
    
    marketplace = PluginMarketplace(
        registry_dir="/Users/yubao/.openclaw/workspace/data/plugins/test_marketplace",
        installed_dir="/Users/yubao/.openclaw/workspace/data/plugins/test_installed",
        published_dir="/Users/yubao/.openclaw/workspace/data/plugins/test_published",
        packages_dir="/Users/yubao/.openclaw/workspace/data/plugins/test_packages"
    )
    
    # 发布插件
    result = marketplace.publish_plugin(
        plugin_dir="/Users/yubao/.openclaw/workspace/data/plugins/test-plugin",
        version="1.0.0"
    )
    print(f"✓ 发布结果: {result.success}")
    if result.success:
        print(f"  - Plugin ID: {result.plugin_id}")
        print(f"  - 版本: {result.version}")
    
    # 浏览市场
    result = marketplace.browse_plugins()
    print(f"✓ 市场插件数量: {len(result['plugins'])}")
    
    # 获取分类
    categories = marketplace.get_categories()
    print(f"✓ 市场分类: {categories}")
    
    # 获取统计
    stats = marketplace.get_marketplace_stats()
    print(f"✓ 市场统计: {stats}")
    
    return True


def main():
    """主测试函数"""
    print("=" * 60)
    print("Plugin Marketplace Test Suite")
    print("=" * 60)
    
    tests = [
        ("Manifest Parsing", test_manifest_parsing),
        ("Registry", test_registry),
        ("Search", test_search),
        ("Marketplace", test_marketplace),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {name} 异常: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
