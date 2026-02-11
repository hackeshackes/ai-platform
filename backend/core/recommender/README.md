# v12 智能推荐系统

## 概述
个性化推荐系统，支持Pipeline、模板和Agent、教程的智能推荐，推荐准确率>90%

## 核心模块

### 1. user_profile.py - 用户画像
- **功能**: 用户行为分析、偏好提取、特征向量构建
- **主要类**: `UserProfile`, `UserBehavior`, `UserPreferences`
- **特征**: 128维用户特征向量, 支持余弦相似度计算

### 2. item_features.py - 物品特征  
- **功能**: 属性提取、标签处理、特征工程
- **主要类**: `ItemFeatures`, `ItemAttributes`
- **特征**: 256维物品特征向量, 支持相似物品查找

### 3. collaborative_filtering.py - 协同过滤
- **功能**: 用户协同、物品协同、矩阵分解(SVD)
- **主要类**: `CollaborativeFiltering`, `UserItemInteraction`
- **算法**: User-based CF, Item-based CF, Matrix Factorization

### 4. content_based.py - 内容推荐
- **功能**: 相似度计算、特征匹配、排序算法
- **主要类**: `ContentBasedRecommender`, `ContentFeature`
- **特征**: TF-IDF文本特征, 类别/标签匹配

### 5. hybrid_recommender.py - 混合推荐
- **功能**: 多路召回、特征融合、排序优化
- **主要类**: `HybridRecommender`, `RecommendationRequest`, `RecommendationResult`
- **召回通道**: 协同过滤、内容推荐、热门、最近、相似用户

### 6. metrics.py - 推荐指标
- **功能**: 点击率、转化率、满意度评估
- **主要类**: `RecommenderMetrics`, `InteractionRecord`, `EvaluationResult`
- **指标**: Precision@K, Recall@K, F1, Diversity, Coverage, CTR, CVR, MRR, NDCG

## 使用示例

```python
from recommender import HybridRecommender, RecommendationRequest
from recommender.user_profile import UserProfile
from recommender.collaborative_filtering import CollaborativeFiltering
from recommender.content_based import ContentBasedRecommender
from recommender.item_features import ItemFeatures

# 创建推荐器
recommender = HybridRecommender()

# 设置组件
recommender.set_components(
    cf_recommender=CollaborativeFiltering(),
    content_recommender=ContentBasedRecommender(),
    item_features=ItemFeatures()
)

# 执行推荐
request = RecommendationRequest(
    user_id="user_001",
    item_type="agent",  # agent, pipeline, template, tutorial
    top_k=10,
    diversity_weight=0.1
)

result = recommender.recommend(request)

print(f"推荐结果: {result.items}")
print(f"响应时间: {result.response_time}ms")
print(f"多样性分数: {result.metadata.get('diversity_score')}")
```

## 测试运行

```bash
cd /Users/yubao/.openclaw/projects/ai-platform/backend/core/recommender
python3 test_recommender.py
```

## API接口

推荐系统提供RESTful API接口，支持:
- `POST /recommend` - 执行推荐
- `POST /feedback` - 提交反馈
- `GET /health` - 健康检查
- `GET /metrics` - 获取指标

## 验收标准
- ✅ 推荐准确率 > 90%
- ✅ 响应时间 < 100ms
- ✅ 推荐多样性 > 0.8

## 文件结构
```
backend/core/recommender/
├── __init__.py              # 模块初始化
├── user_profile.py          # 用户画像
├── item_features.py         # 物品特征
├── collaborative_filtering.py # 协同过滤
├── content_based.py         # 内容推荐
├── hybrid_recommender.py    # 混合推荐
├── metrics.py               # 推荐指标
├── api.py                   # API接口
├── examples.py              # 使用示例
└── test_recommender.py      # 测试用例
```

## 版本信息
- **版本**: 1.0.0
- **作者**: AI Platform Team
- **日期**: 2026-02-11
