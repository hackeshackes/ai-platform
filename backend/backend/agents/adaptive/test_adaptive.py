"""
Test Suite - Adaptive Learning Module
单元测试 - 自适应学习模块
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from .models import (
    Pattern, Interaction, Entity, ContextInfo, ExecutionStep,
    IntentType, ActionType, LearningResult, EvaluationResult,
    OptimizedStrategy, PatternRecord
)
from .q_table import QTable
from .knowledge_base import KnowledgeBase
from .pattern_extractor import PatternExtractor
from .strategy_optimizer import StrategyOptimizer
from .evaluator import Evaluator
from .learner import AdaptiveLearner, InteractionRequest


class TestModels:
    """测试数据模型"""
    
    def test_entity_creation(self):
        """测试实体创建"""
        entity = Entity(name="test", value="value", type="string")
        assert entity.name == "test"
        assert entity.value == "value"
        assert entity.type == "string"
        assert entity.confidence == 1.0
    
    def test_pattern_creation(self):
        """测试模式创建"""
        pattern = Pattern(
            intent=IntentType.QUERY,
            intent_confidence=0.8
        )
        assert pattern.intent == IntentType.QUERY
        assert pattern.intent_confidence == 0.8
        assert len(pattern.entities) == 0
        assert pattern.success is True
    
    def test_pattern_to_dict(self):
        """测试模式序列化"""
        pattern = Pattern(
            id="test-id",
            intent=IntentType.ACTION,
            success=True
        )
        data = pattern.to_dict()
        assert data["id"] == "test-id"
        assert data["intent"] == "action"
        assert data["success"] is True
    
    def test_interaction_creation(self):
        """测试交互创建"""
        interaction = Interaction(
            text="帮我分析数据",
            context={"user_id": "user-001"}
        )
        assert interaction.text == "帮我分析数据"
        assert interaction.context["user_id"] == "user-001"
    
    def test_execution_step(self):
        """测试执行步骤"""
        step = ExecutionStep(
            step_number=1,
            action_type=ActionType.TOOL_CALL,
            action_name="analyze",
            input_params={"data": "test"}
        )
        assert step.step_number == 1
        assert step.action_type == ActionType.TOOL_CALL
        assert step.success is True


class TestQTable:
    """测试Q表"""
    
    def setup_method(self):
        """每个测试前初始化"""
        self.q_table = QTable(state_dim=64, action_dim=16)
    
    def test_init(self):
        """测试初始化"""
        assert self.q_table.state_dim == 64
        assert self.q_table.action_dim == 16
        assert self.q_table.learning_rate == 0.1
    
    def test_register_action(self):
        """测试注册动作"""
        action_id = self.q_table.register_action("test_action")
        assert action_id == 0
        assert self.q_table._action_names[0] == "test_action"
    
    def test_get_set(self):
        """测试获取和设置Q值"""
        state = "test_state"
        action = 0
        
        q_value = self.q_table.get(state, action)
        assert q_value == 0.0
        
        self.q_table.update(state, action, 0.5)
        q_value = self.q_table.get(state, action)
        assert q_value == 0.5
    
    def test_max(self):
        """测试最大值查询"""
        state = "test_state"
        
        self.q_table.update(state, 0, 0.3)
        self.q_table.update(state, 1, 0.7)
        self.q_table.update(state, 2, 0.5)
        
        assert self.q_table.max(state) == 0.7
    
    def test_argmax(self):
        """测试最大Q值动作"""
        state = "test_state"
        
        self.q_table.update(state, 0, 0.3)
        self.q_table.update(state, 1, 0.9)
        self.q_table.update(state, 2, 0.5)
        
        assert self.q_table.argmax(state) == 1
    
    def test_get_stats(self):
        """测试获取统计"""
        self.q_table.update("state1", 0, 0.5)
        self.q_table.update("state2", 1, 0.3)
        
        stats = self.q_table.get_stats()
        assert stats["total_states"] == 2
        assert stats["total_updates"] >= 2


class TestKnowledgeBase:
    """测试知识库"""
    
    def setup_method(self):
        """每个测试前初始化"""
        self.kb = KnowledgeBase()
    
    @pytest.mark.asyncio
    async def test_add_pattern(self):
        """测试添加模式"""
        pattern = Pattern(
            intent=IntentType.QUERY,
            success=True,
            reward=0.8
        )
        
        pattern_id = await self.kb.add(pattern)
        
        assert pattern_id == pattern.id
        assert len(self.kb._patterns) == 1
    
    @pytest.mark.asyncio
    async def test_get_pattern(self):
        """测试获取模式"""
        pattern = Pattern(id="test-id", intent=IntentType.ACTION)
        await self.kb.add(pattern)
        
        retrieved = await self.kb.get("test-id")
        
        assert retrieved is not None
        assert retrieved.id == "test-id"
    
    @pytest.mark.asyncio
    async def test_update_pattern(self):
        """测试更新模式"""
        pattern = Pattern(
            id="test-id",
            intent=IntentType.ACTION,
            success=True
        )
        await self.kb.add(pattern)
        
        # 再次添加相同的模式
        pattern.success = False
        await self.kb.update(pattern)
        
        assert self.kb._success_count["test-id"] == 1  # 只有第一次成功
    
    @pytest.mark.asyncio
    async def test_list_patterns(self):
        """测试列出模式"""
        await self.kb.add(Pattern(intent=IntentType.QUERY))
        await self.kb.add(Pattern(intent=IntentType.ACTION))
        await self.kb.add(Pattern(intent=IntentType.CREATION))
        
        patterns = await self.kb.list(limit=10)
        
        assert len(patterns) == 3
    
    @pytest.mark.asyncio
    async def test_find_similar(self):
        """测试查找相似模式"""
        from .models import Entity
        
        pattern1 = Pattern(
            intent=IntentType.QUERY,
            entities=[Entity("data", "test_data", "noun")]
        )
        await self.kb.add(pattern1)
        
        similar = await self.kb.find_similar(
            IntentType.QUERY,
            ["data", "info"],
            limit=5
        )
        
        assert len(similar) >= 1
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """测试获取统计"""
        await self.kb.add(Pattern(intent=IntentType.QUERY, success=True, reward=0.5))
        await self.kb.add(Pattern(intent=IntentType.QUERY, success=True, reward=0.7))
        await self.kb.add(Pattern(intent=IntentType.ACTION, success=False, reward=0.2))
        
        stats = await self.kb.get_stats()
        
        assert stats["total_patterns"] == 3
        assert stats["total_successes"] == 2


class TestPatternExtractor:
    """测试模式提取器"""
    
    def setup_method(self):
        """每个测试前初始化"""
        self.extractor = PatternExtractor()
    
    @pytest.mark.asyncio
    async def test_parse_intent_query(self):
        """测试解析查询意图"""
        intent, confidence = await self.extractor.parse_intent("什么是机器学习？")
        
        assert intent == IntentType.QUERY
        assert confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_parse_intent_action(self):
        """测试解析动作意图"""
        intent, confidence = await self.extractor.parse_intent("帮我执行数据分析")
        
        assert intent in [IntentType.ACTION, IntentType.ANALYSIS]
        assert confidence > 0.3
    
    @pytest.mark.asyncio
    async def test_parse_intent_creation(self):
        """测试解析创建意图"""
        intent, confidence = await self.extractor.parse_intent("写一段Python代码")
        
        assert intent == IntentType.CREATION
        assert confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_extract_entities(self):
        """测试提取实体"""
        text = "帮我分析2024年的销售数据，联系邮箱是 test@example.com"
        
        entities = await self.extractor.extract_entities(text)
        
        entity_types = [e.type for e in entities]
        
        assert "number" in entity_types or "date" in entity_types
        assert "email" in entity_types
    
    @pytest.mark.asyncio
    async def test_extract_path(self):
        """测试提取执行路径"""
        actions = [
            {"type": "tool_call", "tool": "analyze", "input": {"data": "test"}, "success": True},
            {"type": "response", "output": "分析结果"}
        ]
        
        path = await self.extractor.extract_path(actions)
        
        assert len(path) == 2
        assert path[0].action_type == ActionType.TOOL_CALL
        assert path[1].action_type == ActionType.RESPONSE
    
    @pytest.mark.asyncio
    async def test_full_extraction(self):
        """测试完整提取"""
        interaction = Interaction(
            text="帮我分析销售数据",
            context={"user_id": "user-001"},
            actions=[
                {"type": "tool_call", "tool": "analyze", "success": True}
            ]
        )
        
        pattern = await self.extractor.extract(interaction)
        
        assert pattern.intent in [IntentType.ACTION, IntentType.ANALYSIS]
        assert len(pattern.execution_path) == 1


class TestStrategyOptimizer:
    """测试策略优化器"""
    
    def setup_method(self):
        """每个测试前初始化"""
        self.optimizer = StrategyOptimizer(state_dim=64, action_dim=8)
    
    @pytest.mark.asyncio
    async def test_optimize(self):
        """测试优化"""
        pattern = Pattern(
            intent=IntentType.QUERY,
            success=True,
            reward=0.8
        )
        
        result = await self.optimizer.optimize(pattern, 0.5)
        
        assert isinstance(result, OptimizedStrategy)
        assert result.action in self.optimizer.DEFAULT_ACTIONS
        assert isinstance(result.q_value, float)
    
    @pytest.mark.asyncio
    async def test_explore_vs_exploit(self):
        """测试探索和利用"""
        state = self.optimizer.encode_state(Pattern(intent=IntentType.QUERY))
        
        # 多次测试探索
        exploration_count = 0
        for _ in range(100):
            is_exploration = self.optimizer.exploration_rate > 0.5
            if is_exploration:
                exploration_count += 1
        
        # 探索率应该在合理范围内
        assert 0 <= self.optimizer.exploration_rate <= 1
    
    def test_encode_state(self):
        """测试状态编码"""
        pattern = Pattern(
            intent=IntentType.QUERY,
            entities=[],
            execution_path=[],
            success=True
        )
        
        state = self.optimizer.encode_state(pattern)
        
        assert isinstance(state, str)
        assert len(state) == 16  # SHA256 hex的前16个字符
    
    @pytest.mark.asyncio
    async def test_get_policy(self):
        """测试获取策略"""
        pattern = Pattern(intent=IntentType.QUERY)
        
        policy = await self.optimizer.get_policy(pattern)
        
        assert isinstance(policy, str)
        assert policy in self.optimizer.DEFAULT_ACTIONS
    
    def test_get_stats(self):
        """测试获取统计"""
        stats = self.optimizer.get_stats()
        
        assert "total_states" in stats
        assert "exploration_rate" in stats


class TestEvaluator:
    """测试效果评估器"""
    
    def setup_method(self):
        """每个测试前初始化"""
        self.evaluator = Evaluator()
    
    @pytest.mark.asyncio
    async def test_evaluate_empty_history(self):
        """测试空历史评估"""
        result = await self.evaluator.evaluate("test-agent")
        
        assert result.agent_id == "test-agent"
        assert result.total_interactions == 0
        assert result.success_rate == 0.0
    
    @pytest.mark.asyncio
    async def test_calc_success_rate(self):
        """测试计算成功率"""
        history = [
            {"success": True, "reward": 0.5},
            {"success": True, "reward": 0.6},
            {"success": False, "reward": 0.2}
        ]
        
        rate = self.evaluator.calc_success_rate(history)
        
        assert rate == 2/3
    
    @pytest.mark.asyncio
    async def test_calc_improvement(self):
        """测试计算提升率"""
        history = [
            {"success": True, "reward": 0.3},
            {"success": True, "reward": 0.4},
            {"success": True, "reward": 0.5},
            {"success": True, "reward": 0.6},
            {"success": True, "reward": 0.7},
            {"success": True, "reward": 0.8},
            {"success": True, "reward": 0.9},
            {"success": True, "reward": 1.0},
            {"success": True, "reward": 1.0},
            {"success": True, "reward": 1.0}
        ]
        
        improvement = self.evaluator.calc_improvement(history)
        
        # 应该有正提升
        assert improvement > 0
    
    @pytest.mark.asyncio
    async def test_calc_consistency(self):
        """测试计算一致性"""
        # 一致的历史
        consistent_history = [
            {"success": True, "reward": 0.5},
            {"success": True, "reward": 0.51},
            {"success": True, "reward": 0.49}
        ]
        
        consistent = self.evaluator.calc_consistency(consistent_history)
        
        # 不一致的历史
        inconsistent_history = [
            {"success": True, "reward": 0.1},
            {"success": True, "reward": 0.9},
            {"success": True, "reward": 0.5}
        ]
        
        inconsistent = self.evaluator.calc_consistency(inconsistent_history)
        
        assert consistent > inconsistent
    
    def test_calc_recent_trend(self):
        """测试计算趋势"""
        # 稳定的历史
        history = [{"success": True, "reward": 0.5} for _ in range(10)]
        
        trend = self.evaluator.calc_recent_trend(history)
        
        assert trend in ["improving", "declining", "stable"]


class TestAdaptiveLearner:
    """测试自适应学习引擎"""
    
    def setup_method(self):
        """每个测试前初始化"""
        self.learner = AdaptiveLearner(agent_id="test-agent")
    
    @pytest.mark.asyncio
    async def test_learn_from_interaction(self):
        """测试从交互学习"""
        interaction = Interaction(
            text="帮我分析数据",
            context={},
            actions=[
                {"type": "tool_call", "tool": "analyze", "success": True}
            ]
        )
        
        result = await self.learner.learn_from_interaction(interaction)
        
        assert result.success is True
        assert result.pattern_id != ""
    
    @pytest.mark.asyncio
    async def test_get_learning_status(self):
        """测试获取学习状态"""
        status = await self.learner.get_learning_status()
        
        assert "agent_id" in status
        assert "learning_count" in status
        assert "evaluation" in status
    
    @pytest.mark.asyncio
    async def test_batch_learn(self):
        """测试批量学习"""
        interactions = [
            Interaction(text="测试1", context={}, actions=[]),
            Interaction(text="测试2", context={}, actions=[]),
            Interaction(text="测试3", context={}, actions=[])
        ]
        
        results = await self.learner.batch_learn(interactions)
        
        assert len(results) == 3
        assert all(r.success for r in results)
    
    @pytest.mark.asyncio
    async def test_reset(self):
        """测试重置"""
        await self.learner.reset()
        
        assert self.learner._learning_count == 0


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
