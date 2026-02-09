#!/usr/bin/env python3
"""v2.3 功能测试"""
import asyncio
import sys
sys.path.insert(0, '/Users/yubao/.openclaw/projects/ai-platform/backend')

print("=" * 60)
print("AI Platform v2.3 功能测试")
print("=" * 60)

# AI Gateway 测试
print("\n[1] AI Gateway 功能测试...")
from backend.gateway.gateway import ai_gateway

# 创建路由
route = ai_gateway.create_route(
    name="test-route",
    patterns=["^/chat/.*"],
    provider_id=list(ai_gateway.providers.keys())[0]
)
print(f"  ✅ 创建路由: {route.name}")

# 获取提供商
provider = ai_gateway.get_provider(list(ai_gateway.providers.keys())[0])
print(f"  ✅ 提供商: {provider.name} ({provider.provider_type.value})")

# AI Assistant 测试
print("\n[2] AI Assistant 功能测试...")
from backend.assistant.assistant import ai_assistant

async def test_assistant():
    # 创建对话
    conv = await ai_assistant.create_conversation(user_id="test")
    print(f"  ✅ 创建对话: {conv.conversation_id}")
    
    # 发送消息
    response = await ai_assistant.chat(conv.conversation_id, "帮我训练模型")
    print(f"  ✅ 发送消息: {response.message[:50]}...")
    
    # 诊断问题
    diagnosis = await ai_assistant.diagnose("训练 loss 不下降")
    print(f"  ✅ 诊断结果: {diagnosis['category']}")
    
    # 知识搜索
    results = ai_assistant.search_knowledge("GPU")
    print(f"  ✅ 知识搜索: 找到 {len(results)} 条结果")

asyncio.run(test_assistant())

# Judge Builder 测试
print("\n[3] Judge Builder 功能测试...")
from backend.judges.builder import judge_builder

# 测试评估器
judge = judge_builder.list_judges()[0]
print(f"  ✅ 评估器: {judge.name} ({judge.judge_type})")
print(f"     - 标准数: {len(judge.criteria)}")

# Ray Data 测试
print("\n[4] Ray Data 功能测试...")
from backend.ray.manager import ray_manager

# 获取集群
cluster = ray_manager.list_clusters()[0]
print(f"  ✅ 集群: {cluster.name}")
print(f"     - 状态: {cluster.status}")
print(f"     - 资源: {cluster.resources}")

# 获取数据集
dataset = ray_manager.list_datasets()[0]
print(f"  ✅ 数据集: {dataset.name}")
print(f"     - 格式: {dataset.format}")
print(f"     - 大小: {dataset.size_bytes / 1024 / 1024:.2f} MB")

print("\n" + "=" * 60)
print("v2.3 功能测试完成!")
print("=" * 60)
