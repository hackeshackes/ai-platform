"""
Quick Start Example - V9 Adaptive Learning
快速开始示例 - V9 自适应学习
"""

import asyncio
from backend.agents import AdaptiveLearner, Interaction


async def main():
    """主函数示例"""
    
    # 1. 创建学习引擎
    learner = AdaptiveLearner(agent_id="agent-001")
    print(f"Created learner for agent: {learner.agent_id}")
    
    # 2. 创建示例交互
    interaction = Interaction(
        text="帮我分析最近的销售数据，找出增长趋势",
        context={
            "session_id": "session-001",
            "user_id": "user-001"
        },
        actions=[
            {
                "type": "tool_call",
                "tool": "query_sales",
                "input": {"period": "recent"},
                "success": True
            },
            {
                "type": "tool_call",
                "tool": "analyze_trend",
                "input": {"data": "sales"},
                "success": True
            }
        ]
    )
    
    # 3. 学习
    result = await learner.learn_from_interaction(interaction)
    print(f"Learning result: success={result.success}, pattern_id={result.pattern_id}")
    
    # 4. 获取学习状态
    status = await learner.get_learning_status()
    print(f"Learning count: {status['learning_count']}")
    print(f"Success rate: {status['evaluation']['success_rate']:.2%}")
    
    # 5. 批量学习
    interactions = [
        Interaction(text="查询库存状态", context={}, actions=[]),
        Interaction(text="生成销售报告", context={}, actions=[]),
        Interaction(text="分析客户反馈", context={}, actions=[])
    ]
    
    results = await learner.batch_learn(interactions)
    print(f"Batch learning: {len(results)} interactions processed")
    
    # 6. 获取推荐策略
    new_interaction = Interaction(
        text="预测下季度销售额",
        context={}
    )
    recommendation = await learner.get_recommended_strategy(new_interaction)
    print(f"Recommended strategy: {recommendation['recommended_strategy']}")
    
    print("\nAdaptive learning demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
