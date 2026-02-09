#!/usr/bin/env python3
"""v2.3 核心逻辑测试"""
import asyncio
import sys
import os

# 添加backend到路径
sys.path.insert(0, '/Users/yubao/.openclaw/projects/ai-platform/backend')

print("=" * 60)
print("AI Platform v2.3 核心逻辑测试")
print("=" * 60)

# 测试1: AI Gateway
print("\n[1] 测试 AI Gateway...")
try:
    from backend.gateway.gateway import ai_gateway, ProviderType
    print("  ✅ Gateway模块导入成功")
    
    # 测试功能
    providers = ai_gateway.list_providers()
    print(f"     - 提供商数量: {len(providers)}")
    
    routes = ai_gateway.list_routes()
    print(f"     - 路由数量: {len(routes)}")
    
    usage = ai_gateway.get_usage_stats()
    print(f"     - 使用统计: {usage}")
except Exception as e:
    print(f"  ❌ Gateway测试失败: {e}")

# 测试2: AI Assistant
print("\n[2] 测试 AI Assistant...")
try:
    from backend.assistant.assistant import ai_assistant
    print("  ✅ Assistant模块导入成功")
    
    async def test_assistant():
        # 测试功能
        conv = await ai_assistant.create_conversation(user_id="test_user")
        print(f"     - 对话创建成功: {conv.conversation_id}")
        
        # 健康状态
        health = {
            "conversations": len(ai_assistant.conversations),
            "knowledge_entries": len(ai_assistant.knowledge_base),
            "rules": len(ai_assistant.rule_engine.rules)
        }
        print(f"     - 健康状态: {health}")
    
    asyncio.run(test_assistant())
except Exception as e:
    print(f"  ❌ Assistant测试失败: {e}")

# 测试3: Judge Builder
print("\n[3] 测试 Judge Builder...")
try:
    from backend.judges.builder import judge_builder
    print("  ✅ Judge Builder模块导入成功")
    
    # 测试功能
    judges = judge_builder.list_judges()
    print(f"     - 评估器数量: {len(judges)}")
    
    templates = judge_builder.get_templates()
    print(f"     - 模板数量: {len(templates)}")
    
    # 内置评估器
    builtin = [j for j in judges if j.created_by == "system"]
    print(f"     - 内置评估器: {len(builtin)}")
except Exception as e:
    print(f"  ❌ Judge Builder测试失败: {e}")

# 测试4: Ray Data
print("\n[4] 测试 Ray Data...")
try:
    from backend.ray.manager import ray_manager
    print("  ✅ Ray Manager模块导入成功")
    
    # 测试功能
    clusters = ray_manager.list_clusters()
    print(f"     - 集群数量: {len(clusters)}")
    
    datasets = ray_manager.list_datasets()
    print(f"     - 数据集数量: {len(datasets)}")
    
    # 演示集群
    demo = ray_manager.get_cluster("demo-cluster")
    if demo:
        print(f"     - 演示集群: {demo.name} ({demo.status})")
except Exception as e:
    print(f"  ❌ Ray Data测试失败: {e}")

# 测试5: Performance
print("\n[5] 测试 Performance...")
try:
    from backend.optimization.performance import performance_optimizer
    print("  ✅ Performance模块导入成功")
    
    # 测试功能
    stats = performance_optimizer.get_cache_stats()
    print(f"     - 缓存统计: {stats}")
    
    summary = performance_optimizer.get_performance_summary()
    print(f"     - 性能摘要: {summary}")
except Exception as e:
    print(f"  ❌ Performance测试失败: {e}")

print("\n" + "=" * 60)
print("v2.3 核心逻辑测试完成!")
print("=" * 60)
