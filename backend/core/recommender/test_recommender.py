"""
推荐系统测试用例 - test_recommender.py

单元测试和集成测试
"""

import unittest
from typing import Dict, List, Any
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestUserProfile(unittest.TestCase):
    """用户画像测试"""
    
    def setUp(self):
        from user_profile import UserProfile, UserBehavior
        from datetime import datetime
        
        self.UserProfile = UserProfile
        self.UserBehavior = UserBehavior
        self.datetime = datetime
        
        self.profile = self.UserProfile("test_user")
        
    def test_add_behavior(self):
        """测试添加行为"""
        behavior = self.UserBehavior(
            user_id="test_user",
            timestamp=self.datetime.now(),
            action_type="click",
            item_id="item_001",
            item_type="agent"
        )
        self.profile.add_behavior(behavior)
        
        self.assertEqual(len(self.profile.behaviors), 1)
        self.assertEqual(self.profile.action_counts["click"], 1)
    
    def test_analyze_behaviors(self):
        """测试行为分析"""
        # 添加多个行为
        for i in range(5):
            behavior = self.UserBehavior(
                user_id="test_user",
                timestamp=self.datetime.now(),
                action_type="click",
                item_id=f"item_{i}",
                item_type="agent"
            )
            self.profile.add_behavior(behavior)
        
        analysis = self.profile.analyze_behaviors()
        
        self.assertIn("total_actions", analysis)
        self.assertIn("action_distribution", analysis)
        self.assertEqual(analysis["total_actions"], 5)
    
    def test_extract_preferences(self):
        """测试偏好提取"""
        behaviors = [
            self.UserBehavior(
                user_id="test_user",
                timestamp=self.datetime.now(),
                action_type="like",
                item_id="item_001",
                item_type="agent",
                metadata={"tags": ["nlp", "chat"]}
            ),
            self.UserBehavior(
                user_id="test_user",
                timestamp=self.datetime.now(),
                action_type="use",
                item_id="item_002",
                item_type="pipeline",
                metadata={"complexity": "medium"}
            )
        ]
        
        for b in behaviors:
            self.profile.add_behavior(b)
        
        preferences = self.profile.extract_preferences()
        
        self.assertIsNotNone(preferences)
        self.assertIn(preferences.skill_level, ["beginner", "intermediate", "advanced", "expert"])
    
    def test_build_feature_vector(self):
        """测试特征向量构建"""
        # 添加一些行为
        for i in range(3):
            behavior = self.UserBehavior(
                user_id="test_user",
                timestamp=self.datetime.now(),
                action_type="click",
                item_id=f"item_{i}",
                item_type="agent"
            )
            self.profile.add_behavior(behavior)
        
        vector = self.profile.build_feature_vector()  # 使用默认维度128
        
        self.assertEqual(len(vector), 128)
        self.assertIsNotNone(self.profile.feature_vector)
    
    def test_get_similarity(self):
        """测试用户相似度计算"""
        profile1 = self.UserProfile("user_1")
        profile2 = self.UserProfile("user_2")
        
        # 添加相同行为
        for i in range(3):
            b1 = self.UserBehavior(
                user_id="user_1",
                timestamp=self.datetime.now(),
                action_type="click",
                item_id=f"item_{i}",
                item_type="agent"
            )
            b2 = self.UserBehavior(
                user_id="user_2",
                timestamp=self.datetime.now(),
                action_type="click",
                item_id=f"item_{i}",
                item_type="agent"
            )
            profile1.add_behavior(b1)
            profile2.add_behavior(b2)
        
        similarity = profile1.get_similarity(profile2)
        
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        # 添加行为
        behavior = self.UserBehavior(
            user_id="test_user",
            timestamp=self.datetime.now(),
            action_type="click",
            item_id="item_001",
            item_type="agent"
        )
        self.profile.add_behavior(behavior)
        
        # 导出
        data = self.profile.to_dict()
        
        # 导入
        profile2 = self.UserProfile.from_dict(data)
        
        self.assertEqual(profile2.user_id, self.profile.user_id)


class TestItemFeatures(unittest.TestCase):
    """物品特征测试"""
    
    def setUp(self):
        from item_features import ItemFeatures, ItemAttributes
        from datetime import datetime
        
        self.ItemFeatures = ItemFeatures
        self.ItemAttributes = ItemAttributes
        
        self.features = self.ItemFeatures()
        
    def test_add_item(self):
        """测试添加物品"""
        item = self.ItemAttributes(
            item_id="agent_001",
            item_type="agent",
            name="测试Agent",
            tags={"test", "demo"}
        )
        
        self.features.add_item(item)
        
        self.assertEqual(len(self.features.items), 1)
        self.assertIn("agent_001", self.features.items)
    
    def test_extract_features(self):
        """测试特征提取"""
        item = self.ItemAttributes(
            item_id="agent_001",
            item_type="agent",
            name="智能客服机器人",
            description="基于NLP的客服系统",
            tags={"nlp", "chatbot"},
            complexity="medium"
        )
        
        self.features.add_item(item)
        vector = self.features.extract_features("agent_001")
        
        self.assertEqual(len(vector), self.features.feature_dim)
        self.assertGreater(sum(vector), 0)
    
    def test_find_similar_items(self):
        """测试查找相似物品"""
        items = [
            self.ItemAttributes(
                item_id="agent_001",
                item_type="agent",
                name="智能客服",
                tags={"nlp", "chatbot"}
            ),
            self.ItemAttributes(
                item_id="agent_002",
                item_type="agent",
                name="代码助手",
                tags={"code", "development"}
            ),
            self.ItemAttributes(
                item_id="agent_003",
                item_type="agent",
                name="智能聊天",
                tags={"nlp", "conversation"}
            )
        ]
        
        for item in items:
            self.features.add_item(item)
        
        similar = self.features.find_similar_items("agent_001", top_k=2)
        
        self.assertEqual(len(similar), 2)
        # agent_003 应该比 agent_002 更相似
        self.assertTrue(similar[0][0] in ["agent_002", "agent_003"])
    
    def test_get_items_by_tag(self):
        """测试按标签查询"""
        items = [
            self.ItemAttributes(item_id="i1", item_type="agent", name="Item 1", tags={"tag1", "tag2"}),
            self.ItemAttributes(item_id="i2", item_type="agent", name="Item 2", tags={"tag1", "tag3"}),
            self.ItemAttributes(item_id="i3", item_type="pipeline", name="Item 3", tags={"tag2"})
        ]
        
        for item in items:
            self.features.add_item(item)
        
        result = self.features.get_items_by_tag("tag1")
        
        self.assertIn("i1", result)
        self.assertIn("i2", result)
        self.assertNotIn("i3", result)


class TestCollaborativeFiltering(unittest.TestCase):
    """协同过滤测试"""
    
    def setUp(self):
        from collaborative_filtering import CollaborativeFiltering, UserItemInteraction
        
        self.CF = CollaborativeFiltering
        self.Interaction = UserItemInteraction
        
        self.cf = self.CF(k=5)
    
    def test_add_interaction(self):
        """测试添加交互"""
        interaction = self.Interaction(
            user_id="user_001",
            item_id="item_001",
            rating=0.8
        )
        
        self.cf.add_interaction(interaction)
        
        self.assertEqual(len(self.cf.interactions), 1)
        self.assertEqual(self.cf.user_item_matrix["user_001"]["item_001"], 0.8)
    
    def test_compute_similarity(self):
        """测试相似度计算"""
        # 添加共同交互
        interactions = [
            self.Interaction("u1", "i1", 1.0),
            self.Interaction("u1", "i2", 0.8),
            self.Interaction("u2", "i1", 0.9),
            self.Interaction("u2", "i3", 0.7),
        ]
        
        for inter in interactions:
            self.cf.add_interaction(inter)
        
        self.cf.compute_user_similarity()
        
        self.assertIn("u1", self.cf.user_similarity)
        self.assertIn("u2", self.cf.user_similarity["u1"])
    
    def test_recommend_user_based(self):
        """测试基于用户的推荐"""
        interactions = [
            self.Interaction("u1", "i1", 1.0),
            self.Interaction("u1", "i2", 0.8),
            self.Interaction("u2", "i1", 1.0),
            self.Interaction("u2", "i3", 0.9),
        ]
        
        for inter in interactions:
            self.cf.add_interaction(inter)
        
        recommendations = self.cf.recommend_user_based("u1", top_k=3)
        
        self.assertIsInstance(recommendations, list)
        # i3 应该是推荐候选
        item_ids = [item[0] for item in recommendations]
        self.assertIn("i3", item_ids)
    
    def test_matrix_factorization(self):
        """测试矩阵分解"""
        interactions = [
            self.Interaction("u1", "i1", 1.0),
            self.Interaction("u1", "i2", 0.5),
            self.Interaction("u2", "i1", 0.8),
            self.Interaction("u2", "i2", 1.0),
            self.Interaction("u3", "i2", 0.7),
        ]
        
        for inter in interactions:
            self.cf.add_interaction(inter)
        
        self.cf.train_matrix_factorization()
        
        self.assertIsNotNone(self.cf.user_factors)
        self.assertIsNotNone(self.cf.item_factors)
    
    def test_get_stats(self):
        """测试统计信息"""
        interactions = [
            self.Interaction("u1", "i1", 1.0),
            self.Interaction("u2", "i1", 0.8),
        ]
        
        for inter in interactions:
            self.cf.add_interaction(inter)
        
        stats = self.cf.get_stats()
        
        self.assertIn("total_interactions", stats)
        self.assertEqual(stats["total_interactions"], 2)


class TestContentBased(unittest.TestCase):
    """内容推荐测试"""
    
    def setUp(self):
        from content_based import ContentBasedRecommender
        
        self.Recommender = ContentBasedRecommender
        self.rec = self.Recommender()
    
    def test_add_item(self):
        """测试添加物品"""
        self.rec.add_item(
            item_id="item_001",
            name="测试物品",
            description="这是一个测试物品描述",
            categories=["category_1"],
            tags=["tag1", "tag2"]
        )
        
        self.assertEqual(len(self.rec.item_features), 1)
    
    def test_recommend_by_query(self):
        """测试基于查询的推荐"""
        items = [
            {"id": "i1", "name": "智能客服", "description": "NLP客服系统", 
             "categories": ["agent"], "tags": ["nlp"]},
            {"id": "i2", "name": "代码助手", "description": "代码开发工具",
             "categories": ["agent"], "tags": ["code"]},
            {"id": "i3", "name": "智能聊天", "description": "对话系统",
             "categories": ["agent"], "tags": ["nlp", "chat"]}
        ]
        
        self.rec.add_items_batch(items)
        
        results = self.rec.recommend_by_query("智能对话", top_k=3)
        
        self.assertEqual(len(results), 3)
    
    def test_find_similar_items(self):
        """测试查找相似物品"""
        items = [
            {"id": "i1", "name": "A", "description": "文本A", "categories": [], "tags": ["t1", "t2"]},
            {"id": "i2", "name": "B", "description": "文本B", "categories": [], "tags": ["t1", "t3"]},
            {"id": "i3", "name": "C", "description": "文本C", "categories": [], "tags": ["t2", "t3"]}
        ]
        
        self.rec.add_items_batch(items)
        
        similar = self.rec.find_similar_items("i1", top_k=2)
        
        self.assertEqual(len(similar), 2)
        # i2 有共同标签 t1，应该比 i3 更相似
    
    def test_get_vocabulary_size(self):
        """测试词汇表大小"""
        self.rec.add_item("i1", "测试", "描述1", [], ["tag1", "tag2"])
        self.rec.add_item("i2", "测试2", "描述2", [], ["tag3", "tag4"])
        
        vocab_size = self.rec.get_vocabulary_size()
        
        self.assertGreater(vocab_size, 0)


class TestHybridRecommender(unittest.TestCase):
    """混合推荐测试"""
    
    def setUp(self):
        from hybrid_recommender import HybridRecommender, RecommendationRequest
        
        self.HybridRecommender = HybridRecommender
        self.RecommendationRequest = RecommendationRequest
        
        self.rec = self.HybridRecommender()
    
    def test_recommend(self):
        """测试推荐"""
        # 创建请求
        request = self.RecommendationRequest(
            user_id="test_user",
            item_type="agent",
            top_k=5
        )
        
        # 无数据时应该返回空结果
        result = self.rec.recommend(request)
        
        self.assertIsInstance(result.items, list)
        self.assertEqual(len(result.items), 0)
    
    def test_multi_channel_recall(self):
        """测试多路召回"""
        # 设置组件
        from collaborative_filtering import CollaborativeFiltering
        from content_based import ContentBasedRecommender
        
        self.rec.cf_recommender = CollaborativeFiltering()
        self.rec.content_recommender = ContentBasedRecommender()
        
        request = self.RecommendationRequest(
            user_id="test_user",
            top_k=10
        )
        
        candidates = self.rec.multi_channel_recall(request)
        
        self.assertIsInstance(candidates, list)
    
    def test_channel_weights(self):
        """测试通道权重"""
        self.rec.update_channel_weight("collaborative", 0.5)
        self.rec.update_channel_weight("content_based", 0.5)
        
        self.assertEqual(self.rec.recall_channels["collaborative"]["weight"], 0.5)
        self.assertEqual(self.rec.recall_channels["content_based"]["weight"], 0.5)
    
    def test_enable_disable_channel(self):
        """测试通道启用禁用"""
        self.rec.disable_channel("popular")
        self.assertFalse(self.rec.recall_channels["popular"]["enabled"])
        
        self.rec.enable_channel("popular")
        self.assertTrue(self.rec.recall_channels["popular"]["enabled"])


class TestMetrics(unittest.TestCase):
    """指标测试"""
    
    def setUp(self):
        from metrics import RecommenderMetrics, InteractionRecord
        
        self.Metrics = RecommenderMetrics
        self.Interaction = InteractionRecord
        
        self.metrics = self.Metrics()
    
    def test_log_interaction(self):
        """测试记录交互"""
        interaction = self.Interaction(
            user_id="u1",
            item_id="i1",
            recommended=True,
            timestamp=20240101,
            action="click"
        )
        
        self.metrics.log_interaction(interaction)
        
        self.assertEqual(len(self.metrics.interactions), 1)
    
    def test_calculate_accuracy(self):
        """测试准确率计算"""
        interactions = [
            self.Interaction("u1", "i1", True, 20240101, "click"),
            self.Interaction("u1", "i2", True, 20240101, "click"),
            self.Interaction("u1", "i3", True, 20240101, "impression"),  # 负面
        ]
        
        for inter in interactions:
            self.metrics.log_interaction(inter)
        
        accuracy = self.metrics.calculate_accuracy()
        
        self.assertGreater(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_calculate_precision_at_k(self):
        """测试精确率@K"""
        # 记录推荐
        self.metrics.log_recommendation("u1", ["i1", "i2", "i3", "i4", "i5"])
        self.metrics.log_recommendation("u2", ["i1", "i2", "i3"])
        
        # 记录正向交互
        positive_interactions = [
            self.Interaction("u1", "i2", True, 20240101, "click"),
            self.Interaction("u1", "i5", True, 20240101, "like"),
        ]
        
        for inter in positive_interactions:
            self.metrics.log_interaction(inter)
        
        precision = self.metrics.calculate_precision_at_k(3)
        
        self.assertGreaterEqual(precision, 0.0)
        self.assertLessEqual(precision, 1.0)
    
    def test_calculate_diversity(self):
        """测试多样性计算"""
        self.metrics.log_recommendation("u1", ["i1", "i2", "i3"])
        
        diversity = self.metrics.calculate_diversity()
        
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
    
    def test_calculate_coverage(self):
        """测试覆盖率计算"""
        # 记录一些物品
        self.metrics.item_counts["i1"] = 10
        self.metrics.item_counts["i2"] = 5
        self.metrics.item_counts["i3"] = 3
        
        # 记录推荐
        self.metrics.log_recommendation("u1", ["i1", "i2"])
        
        coverage = self.metrics.calculate_coverage()
        
        self.assertEqual(coverage, 2/3)
    
    def test_evaluate_full(self):
        """测试全面评估"""
        # 添加数据
        interactions = [
            self.Interaction("u1", "i1", True, 20240101, "click"),
            self.Interaction("u1", "i2", True, 20240101, "use"),
        ]
        
        for inter in interactions:
            self.metrics.log_interaction(inter)
        
        self.metrics.log_recommendation("u1", ["i1", "i2", "i3"])
        
        result = self.metrics.evaluate_full()
        
        self.assertIsInstance(result.accuracy, float)
        self.assertIsInstance(result.precision, float)
        self.assertIsInstance(result.recall, float)
    
    def test_export_metrics(self):
        """测试导出指标"""
        metrics_data = self.metrics.export_metrics()
        
        self.assertIn("evaluation", metrics_data)
        self.assertIn("metadata", metrics_data)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestUserProfile))
    suite.addTests(loader.loadTestsFromTestCase(TestItemFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestCollaborativeFiltering))
    suite.addTests(loader.loadTestsFromTestCase(TestContentBased))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridRecommender))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
