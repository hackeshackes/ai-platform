"""
Test Suite - API Endpoints (Simplified)
API端点单元测试 - 简化版
"""

import pytest
from unittest.mock import patch, MagicMock

# 测试导入
def test_import_endpoints():
    """测试导入API端点"""
    from .api.endpoints import router
    assert router is not None


def test_import_models():
    """测试导入模型"""
    from .models import (
        Pattern, Interaction, Entity, ContextInfo,
        IntentType, ActionType, LearningResult
    )
    assert Pattern is not None
    assert Interaction is not None
    assert IntentType is not None


def test_import_learner():
    """测试导入Learner"""
    from .learner import AdaptiveLearner
    assert AdaptiveLearner is not None


def test_import_extractor():
    """测试导入提取器"""
    from .pattern_extractor import PatternExtractor
    assert PatternExtractor is not None


def test_import_optimizer():
    """测试导入优化器"""
    from .strategy_optimizer import StrategyOptimizer
    assert StrategyOptimizer is not None


def test_import_evaluator():
    """测试导入评估器"""
    from .evaluator import Evaluator
    assert Evaluator is not None


def test_import_qtable():
    """测试导入Q表"""
    from .q_table import QTable
    assert QTable is not None


def test_import_knowledge_base():
    """测试导入知识库"""
    from .knowledge_base import KnowledgeBase
    assert KnowledgeBase is not None


class TestIntentTypes:
    """测试意图类型"""
    
    def test_all_intent_types_exist(self):
        """测试所有意图类型存在"""
        from .models import IntentType
        
        assert IntentType.QUERY.value == "query"
        assert IntentType.ACTION.value == "action"
        assert IntentType.CREATION.value == "creation"
        assert IntentType.ANALYSIS.value == "analysis"
        assert IntentType.LEARNING.value == "learning"
        assert IntentType.UNKNOWN.value == "unknown"


class TestActionTypes:
    """测试动作类型"""
    
    def test_all_action_types_exist(self):
        """测试所有动作类型存在"""
        from .models import ActionType
        
        assert ActionType.TOOL_CALL.value == "tool_call"
        assert ActionType.REASONING.value == "reasoning"
        assert ActionType.RESPONSE.value == "response"
        assert ActionType.ERROR.value == "error"


class TestLearnerClass:
    """测试Learner类"""
    
    def test_learner_init(self):
        """测试Learner初始化"""
        from .learner import AdaptiveLearner
        
        learner = AdaptiveLearner("test-agent")
        
        assert learner.agent_id == "test-agent"
        assert learner._learning_count == 0
    
    def test_learner_has_components(self):
        """测试Learner包含所有组件"""
        from .learner import AdaptiveLearner
        
        learner = AdaptiveLearner("test-agent")
        
        assert hasattr(learner, 'knowledge_base')
        assert hasattr(learner, 'extractor')
        assert hasattr(learner, 'optimizer')
        assert hasattr(learner, 'evaluator')


class TestQTableClass:
    """测试QTable类"""
    
    def test_qtable_init_default(self):
        """测试默认初始化"""
        from .q_table import QTable
        
        qt = QTable()
        
        assert qt.state_dim == 128
        assert qt.action_dim == 64
        assert qt.learning_rate == 0.1
    
    def test_qtable_init_custom(self):
        """测试自定义初始化"""
        from .q_table import QTable
        
        qt = QTable(state_dim=64, action_dim=16, learning_rate=0.2)
        
        assert qt.state_dim == 64
        assert qt.action_dim == 16
        assert qt.learning_rate == 0.2


class TestKnowledgeBaseClass:
    """测试KnowledgeBase类"""
    
    def test_kb_init(self):
        """测试初始化"""
        from .knowledge_base import KnowledgeBase
        
        kb = KnowledgeBase()
        
        assert kb._patterns == {}
    
    def test_kb_with_storage(self):
        """测试带存储路径初始化"""
        from .knowledge_base import KnowledgeBase
        
        kb = KnowledgeBase(storage_path="/tmp/test_kb.json")
        
        assert kb.storage_path == "/tmp/test_kb.json"


class TestPatternExtractorClass:
    """测试PatternExtractor类"""
    
    def test_extractor_init(self):
        """测试初始化"""
        from .pattern_extractor import PatternExtractor
        
        extractor = PatternExtractor()
        
        assert hasattr(extractor, 'INTENT_KEYWORDS')
        assert hasattr(extractor, 'ENTITY_TYPES')


class TestEvaluatorClass:
    """测试Evaluator类"""
    
    def test_evaluator_init(self):
        """测试初始化"""
        from .evaluator import Evaluator
        
        evaluator = Evaluator()
        
        assert evaluator.knowledge_base is None
        assert evaluator._evaluation_history == {}


class TestRoutingPrefix:
    """测试路由前缀"""
    
    def test_router_has_correct_prefix(self):
        """测试路由有正确的前缀"""
        from .api.endpoints import router
        
        assert router.prefix == "/adaptive"


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
