"""
推荐系统示例 - examples.py

使用示例和演示代码
"""

from typing import Dict, List, Any
from datetime import datetime


def demo_basic_recommendation():
    """基础推荐演示"""
    from .hybrid_recommender import HybridRecommender, RecommendationRequest
    from .user_profile import UserProfile, UserBehavior
    from .item_features import ItemFeatures, ItemAttributes
    from .collaborative_filtering import CollaborativeFiltering, UserItemInteraction
    from .content_based import ContentBasedRecommender
    from .metrics import RecommenderMetrics
    
    # 创建组件
    cf = CollaborativeFiltering()
    content_rec = ContentBasedRecommender()
    metrics = RecommenderMetrics()
    item_features = ItemFeatures()
    
    # 创建混合推荐器
    recommender = HybridRecommender()
    recommender.set_components(
        cf_recommender=cf,
        content_recommender=content_rec,
        item_features=item_features
    )
    
    # 添加示例数据
    _add_demo_data(cf, content_rec, item_features)
    
    # 执行推荐
    request = RecommendationRequest(
        user_id="user_001",
        item_type="agent",
        top_k=5,
        diversity_weight=0.1
    )
    
    result = recommender.recommend(request)
    
    print("推荐结果:")
    for item in result.items:
        print(f"  - {item['item_id']}: {item['score']:.4f}")
    
    print(f"\n响应时间: {result.response_time:.2f}ms")
    print(f"多样性分数: {result.metadata.get('diversity_score', 0)}")
    
    return result


def demo_user_profile():
    """用户画像演示"""
    from .user_profile import UserProfile, UserBehavior
    
    # 创建用户画像
    profile = UserProfile("user_001")
    
    # 添加行为
    behaviors = [
        UserBehavior(
            user_id="user_001",
            timestamp=datetime.now(),
            action_type="use",
            item_id="agent_001",
            item_type="agent",
            duration=120,
            metadata={"scenario": "customer_service", "tags": ["chatbot", "nlp"]}
        ),
        UserBehavior(
            user_id="user_001",
            timestamp=datetime.now(),
            action_type="like",
            item_id="pipeline_001",
            item_type="pipeline",
            duration=60,
            metadata={"complexity": "medium", "task_type": "data_processing"}
        ),
        UserBehavior(
            user_id="user_001",
            timestamp=datetime.now(),
            action_type="click",
            item_id="template_001",
            item_type="template",
            metadata={"industry": "finance", "frequency": "daily"}
        )
    ]
    
    for behavior in behaviors:
        profile.add_behavior(behavior)
    
    # 分析行为
    analysis = profile.analyze_behaviors()
    print("行为分析:", analysis)
    
    # 提取偏好
    preferences = profile.extract_preferences()
    print("技能水平:", preferences.skill_level)
    print("复杂度偏好:", preferences.preferred_complexity)
    
    # 构建特征向量
    features = profile.build_feature_vector()
    print(f"特征向量维度: {len(features)}")
    
    return profile


def demo_collaborative_filtering():
    """协同过滤演示"""
    from .collaborative_filtering import CollaborativeFiltering, UserItemInteraction
    
    cf = CollaborativeFiltering(k=10)
    
    # 添加交互数据
    interactions = [
        UserItemInteraction(user_id="user_001", item_id="item_001", rating=1.0),
        UserItemInteraction(user_id="user_001", item_id="item_002", rating=0.8),
        UserItemInteraction(user_id="user_002", item_id="item_001", rating=0.9),
        UserItemInteraction(user_id="user_002", item_id="item_003", rating=1.0),
        UserItemInteraction(user_id="user_003", item_id="item_002", rating=0.7),
        UserItemInteraction(user_id="user_003", item_id="item_003", rating=0.8),
    ]
    
    cf.add_interactions_batch(interactions)
    
    # 计算相似度
    cf.compute_user_similarity()
    cf.compute_item_similarity()
    
    # 推荐
    recommendations = cf.recommend_user_based("user_001", top_k=3)
    print("用户协同推荐:", recommendations)
    
    # 矩阵分解
    cf.train_matrix_factorization()
    mf_recommendations = cf.recommend_matrix_factorization("user_001", top_k=3)
    print("矩阵分解推荐:", mf_recommendations)
    
    # 统计
    print("统计:", cf.get_stats())
    
    return cf


def demo_content_based():
    """基于内容推荐演示"""
    from .content_based import ContentBasedRecommender
    
    recommender = ContentBasedRecommender()
    
    # 添加物品
    items = [
        {
            'id': 'agent_001',
            'name': '智能客服机器人',
            'description': '基于NLP的智能客服系统，支持多轮对话和情感分析',
            'categories': ['agent', 'chatbot'],
            'tags': ['nlp', 'conversation', 'customer_service'],
            'metadata': {'complexity': 'medium'}
        },
        {
            'id': 'agent_002',
            'name': '代码审查助手',
            'description': 'AI驱动的代码审查工具，自动检测代码异味和安全漏洞',
            'categories': ['agent', 'developer_tool'],
            'tags': ['code', 'review', 'security'],
            'metadata': {'complexity': 'high'}
        },
        {
            'id': 'pipeline_001',
            'name': '数据处理流水线',
            'description': '自动化数据清洗、转换和加载流程',
            'categories': ['pipeline', 'data'],
            'tags': ['etl', 'data_processing', 'automation'],
            'metadata': {'complexity': 'medium'}
        },
        {
            'id': 'template_001',
            'name': '财务报表模板',
            'description': '企业级财务报表模板，包含利润表、资产负债表和现金流量表',
            'categories': ['template', 'finance'],
            'tags': ['reporting', 'finance', 'excel'],
            'metadata': {'industry': 'finance'}
        },
        {
            'id': 'tutorial_001',
            'name': '快速入门教程',
            'description': '从零开始学习AI助手的使用教程',
            'categories': ['tutorial', 'beginner'],
            'tags': ['learning', 'getting_started', 'basics'],
            'metadata': {'difficulty': 'beginner'}
        }
    ]
    
    recommender.add_items_batch(items)
    
    # 查询推荐
    results = recommender.recommend_by_query("智能客服", top_k=3)
    print("查询推荐:", results)
    
    # 查找相似物品
    similar = recommender.find_similar_items("agent_001", top_k=3)
    print("相似物品:", similar)
    
    # 按偏好推荐
    preferences = {'nlp': 0.8, 'conversation': 0.7, 'code': 0.3}
    pref_results = recommender.recommend_by_preferences(preferences, top_k=3)
    print("偏好推荐:", pref_results)
    
    return recommender


def demo_metrics():
    """指标计算演示"""
    from .metrics import RecommenderMetrics, InteractionRecord, EvaluationResult
    
    metrics = RecommenderMetrics()
    
    # 添加交互记录
    interactions = [
        InteractionRecord(user_id="u1", item_id="i1", recommended=True, 
                         timestamp=20240101, action="click", reward=0.5),
        InteractionRecord(user_id="u1", item_id="i2", recommended=True,
                         timestamp=20240101, action="use", reward=0.8),
        InteractionRecord(user_id="u1", item_id="i3", recommended=True,
                         timestamp=20240101, action="like", reward=1.0),
        InteractionRecord(user_id="u2", item_id="i1", recommended=True,
                         timestamp=20240101, action="click", reward=0.5),
        InteractionRecord(user_id="u2", item_id="i4", recommended=False,
                         timestamp=20240101, action="click", reward=0.3),
    ]
    
    for interaction in interactions:
        metrics.log_interaction(interaction)
    
    # 添加推荐记录
    metrics.log_recommendation("u1", ["i1", "i2", "i3", "i4", "i5"])
    metrics.log_recommendation("u2", ["i1", "i2", "i3"])
    
    # 计算指标
    accuracy = metrics.calculate_accuracy()
    precision = metrics.calculate_precision_at_k(3)
    recall = metrics.calculate_recall_at_k(3)
    diversity = metrics.calculate_diversity()
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率@3: {precision:.4f}")
    print(f"召回率@3: {recall:.4f}")
    print(f"多样性: {diversity:.4f}")
    
    # 全面评估
    result = metrics.evaluate_full()
    print(f"F1分数: {result.f1_score:.4f}")
    print(f"覆盖率: {result.coverage:.4f}")
    
    return metrics


def demo_full_workflow():
    """完整工作流演示"""
    print("=" * 50)
    print("推荐系统完整演示")
    print("=" * 50)
    
    # 1. 用户画像
    print("\n1. 用户画像演示")
    profile = demo_user_profile()
    
    # 2. 协同过滤
    print("\n2. 协同过滤演示")
    cf = demo_collaborative_filtering()
    
    # 3. 内容推荐
    print("\n3. 内容推荐演示")
    content_rec = demo_content_based()
    
    # 4. 指标计算
    print("\n4. 指标计算演示")
    metrics = demo_metrics()
    
    # 5. 完整推荐
    print("\n5. 完整推荐演示")
    result = demo_basic_recommendation()
    
    return {
        'profile': profile,
        'cf': cf,
        'content': content_rec,
        'metrics': metrics,
        'result': result
    }


def _add_demo_data(cf, content_rec, item_features):
    """添加演示数据"""
    from .item_features import ItemAttributes
    
    # 添加物品
    items = [
        ItemAttributes(
            item_id="agent_001",
            item_type="agent",
            name="智能客服机器人",
            description="基于NLP的智能客服系统",
            tags={"nlp", "chatbot", "customer_service"},
            categories={"agent"},
            scenarios={"customer_service", "support"},
            capabilities={"dialogue", "sentiment_analysis"}
        ),
        ItemAttributes(
            item_id="agent_002",
            item_type="agent",
            name="代码审查助手",
            description="AI驱动的代码审查工具",
            tags={"code", "review", "security"},
            categories={"agent", "developer_tool"},
            scenarios={"code_review", "development"}
        ),
        ItemAttributes(
            item_id="pipeline_001",
            item_type="pipeline",
            name="数据处理流水线",
            description="自动化数据处理流程",
            tags={"etl", "data_processing"},
            categories={"pipeline"},
            complexity="medium",
            task_types={"data_cleaning", "transform"}
        ),
        ItemAttributes(
            item_id="template_001",
            item_type="template",
            name="财务报表模板",
            description="企业级财务报表模板",
            tags={"finance", "reporting"},
            categories={"template"},
            industry="finance"
        ),
        ItemAttributes(
            item_id="tutorial_001",
            item_type="tutorial",
            name="快速入门教程",
            description="从零开始学习",
            tags={"learning", "basics"},
            categories={"tutorial"},
            difficulty="beginner"
        )
    ]
    
    for item in items:
        item_features.add_item(item)


if __name__ == "__main__":
    # 运行演示
    demo_full_workflow()
