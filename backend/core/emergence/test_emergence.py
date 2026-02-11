"""
Emergence Engine Tests - 涌现引擎测试用例
"""

import unittest
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capability_detector import CapabilityDetector, EmergenceType, CapabilityLevel
from self_organization import SelfOrganization, PlasticityRule, NetworkType
from creative_generator import CreativeGenerator, InnovationType
from emergence_monitor import EmergenceMonitor, SafetyLevel, ImpactLevel


class TestCapabilityDetector(unittest.TestCase):
    """能力检测器测试"""
    
    def setUp(self):
        self.detector = CapabilityDetector()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(len(self.detector.known_patterns), 0)
        self.assertEqual(len(self.detector.behavior_history), 0)
    
    def test_novelty_detection(self):
        """测试新颖性检测"""
        # 第一次行为应该被识别为新颖
        behaviors = ['action:query', 'decision:respond']
        is_novel, score = self.detector._check_novelty(behaviors)
        self.assertTrue(is_novel)
        self.assertEqual(score, 1.0)
    
    def test_capability_detection(self):
        """测试能力检测"""
        class MockModel:
            pass
        
        model = MockModel()
        interaction = {
            'actions': ['analyze', 'query'],
            'decisions': ['select'],
            'strategies': ['context_aware']
        }
        
        capability = self.detector.detect(model, interaction)
        
        # 应该检测到新能力
        self.assertIsNotNone(capability)
        self.assertIn('capability_', capability.name)
        self.assertIsInstance(capability.emergence_type, EmergenceType)
        self.assertGreater(capability.confidence, 0.0)
    
    def test_boundary_exploration(self):
        """测试边界探索"""
        behaviors = ['action:test', 'action:verify']
        boundaries = self.detector._explore_boundaries(behaviors)
        
        self.assertIsInstance(boundaries, dict)
        self.assertGreater(len(boundaries), 0)
    
    def test_signature_creation(self):
        """测试签名创建"""
        behaviors = ['action:test']
        emergence_type = EmergenceType.BEHAVIORAL
        
        signature = self.detector._create_signature(behaviors, emergence_type)
        
        self.assertIsInstance(signature, str)
        self.assertEqual(len(signature), 16)
    
    def test_batch_detection(self):
        """测试批量检测"""
        class MockModel:
            pass
        
        model = MockModel()
        interactions = [
            {'actions': ['a1', 'a2']},
            {'actions': ['b1', 'b2']},
            {'actions': ['a1', 'a2']}  # 重复
        ]
        
        capabilities = self.detector.batch_detect(model, interactions)
        
        # 应该有2个不同能力
        self.assertEqual(len(capabilities), 2)


class TestSelfOrganization(unittest.TestCase):
    """自组织系统测试"""
    
    def setUp(self):
        self.organizer = SelfOrganization()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.organizer)
        self.assertEqual(len(self.organizer.neurons), 0)
        self.assertEqual(len(self.organizer.synapses), 0)
    
    def test_add_neuron(self):
        """测试添加神经元"""
        neuron = self.organizer.add_neuron(layer=0, position=(1, 2))
        
        self.assertIsNotNone(neuron)
        self.assertEqual(neuron.layer, 0)
        self.assertIn(neuron.neuron_id, self.organizer.neurons)
    
    def test_add_synapse(self):
        """测试添加突触"""
        n1 = self.organizer.add_neuron(layer=0, position=(1, 1))
        n2 = self.organizer.add_neuron(layer=1, position=(2, 2))
        
        synapse = self.organizer.add_synapse(n1.neuron_id, n2.neuron_id, weight=0.5)
        
        self.assertIsNotNone(synapse)
        self.assertEqual(synapse.weight, 0.5)
        self.assertEqual(synapse.plasticity_rule, PlasticityRule.HEBBIAN)
    
    def test_organization(self):
        """测试组织过程"""
        # 添加一些神经元和连接
        for i in range(5):
            self.organizer.add_neuron(layer=i % 2, position=(i, i))
        
        for i in range(4):
            self.organizer.add_synapse(f"neuron_{i}", f"neuron_{i+1}")
        
        # 执行自组织
        structure = self.organizer.organize()
        
        self.assertIsNotNone(structure)
        self.assertIsInstance(structure.network_type, NetworkType)
        self.assertGreater(len(structure.layers), 0)
    
    def test_hebbian_learning(self):
        """测试Hebbian学习"""
        # 添加神经元到organizer
        pre = self.organizer.add_neuron(layer=0, position=(1, 1))
        post = self.organizer.add_neuron(layer=1, position=(2, 2))
        
        delta_w = self.organizer._hebbian_rule(pre, post)
        
        # Hebbian规则应该产生正权重变化
        self.assertIsInstance(delta_w, float)
    
    def test_get_statistics(self):
        """测试获取统计"""
        # 添加一些数据
        for i in range(3):
            self.organizer.add_neuron(layer=i, position=(i, i))
        
        stats = self.organizer.get_statistics()
        
        self.assertIn('total_neurons', stats)
        self.assertIn('total_connections', stats)
        self.assertEqual(stats['total_neurons'], 3)
    
    def test_reset(self):
        """测试重置"""
        self.organizer.add_neuron(0, (1, 1))
        self.organizer.add_synapse('n1', 'n2')
        
        self.organizer.reset()
        
        self.assertEqual(len(self.organizer.neurons), 0)
        self.assertEqual(len(self.organizer.synapses), 0)


class TestCreativeGenerator(unittest.TestCase):
    """创意生成器测试"""
    
    def setUp(self):
        self.generator = CreativeGenerator()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.generator)
        self.assertEqual(len(self.generator.idea_history), 0)
    
    def test_problem_classification(self):
        """测试问题分类"""
        self.assertEqual(self.generator._classify_problem({'text': 'optimize performance'}), 'optimization')
        self.assertEqual(self.generator._classify_problem({'text': 'design a new system'}), 'design')
        self.assertEqual(self.generator._classify_problem({'text': 'solve the puzzle'}), 'problem_solving')
    
    def test_mode_selection(self):
        """测试模式选择"""
        from creative_generator import CreativeMode
        self.assertEqual(self.generator._select_mode('optimization'), CreativeMode.CONVERGENT)
        self.assertEqual(self.generator._select_mode('design'), CreativeMode.LATERAL)
        self.assertEqual(self.generator._select_mode('problem_solving'), CreativeMode.ANALOGICAL)
    
    def test_approach_generation(self):
        """测试方法生成"""
        approach = self.generator._generate_approach('test problem')
        
        self.assertIsInstance(approach, str)
        self.assertGreater(len(approach), 0)
    
    def test_problem_solving(self):
        """测试问题解决"""
        problem = {'description': '提高效率,降低成本'}
        constraints = {'budget': 'limited'}
        
        solutions = self.generator.solve(problem, constraints)
        
        self.assertIsInstance(solutions, list)
        self.assertGreater(len(solutions), 0)
        
        for sol in solutions:
            self.assertIsInstance(sol.solution_id, str)
            self.assertIsInstance(sol.innovation_type, InnovationType)
            self.assertGreater(sol.confidence, 0.0)
    
    def test_fitness_calculation(self):
        """测试适应度计算"""
        idea = {
            'approach': '这是一个测试方法，很长很长' * 10,
            'raw_novelty': 0.7
        }
        constraints = {}
        
        fitness = self.generator._calculate_fitness(idea, constraints)
        
        self.assertGreater(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
    
    def test_art_creation(self):
        """测试艺术创作"""
        art = self.generator.create_artistic_content('自然', '抽象')
        
        self.assertIsNotNone(art)
        self.assertIn('elements', art)
        self.assertIn('creativity_score', art)
    
    def test_strategy_discovery(self):
        """测试策略发现"""
        context = {'market': 'growing', 'competition': 'intense'}
        
        result = self.generator.discover_strategy(context)
        
        self.assertIn('strategies', result)
        self.assertIn('opportunities', result)


class TestEmergenceMonitor(unittest.TestCase):
    """涌现监控测试"""
    
    def setUp(self):
        self.monitor = EmergenceMonitor()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(len(self.monitor.events), 0)
        self.assertEqual(len(self.monitor.capability_metrics), 0)
    
    def test_behavior_tracking(self):
        """测试行为跟踪"""
        behavior = {
            'type': 'behavioral',
            'actions': ['query', 'respond'],
            'description': '对话行为'
        }
        
        event = self.monitor.track(behavior)
        
        # 可能检测到涌现
        if event:
            self.assertIsInstance(event.event_id, str)
            self.assertGreater(event.confidence, 0.0)
    
    def test_safety_check(self):
        """测试安全检查"""
        from emergence_monitor import EmergenceEvent
        
        event = EmergenceEvent(
            event_id='test_1',
            timestamp=1234567890,
            event_type='test',
            capability_name='test_cap',
            emergence_type='behavioral',
            confidence=0.9,
            impact=ImpactLevel.MODERATE,
            safety=SafetyLevel.SAFE,
            description='Test event'
        )
        
        result = self.monitor._safety_check(event)
        
        self.assertIn('level', result)
        self.assertIn('issues', result)
    
    def test_impact_assessment(self):
        """测试影响评估"""
        from emergence_monitor import EmergenceEvent
        
        event = EmergenceEvent(
            event_id='test_2',
            timestamp=1234567890,
            event_type='test',
            capability_name='test_cap',
            emergence_type='cognitive',
            confidence=0.7,
            impact=ImpactLevel.MODERATE,
            safety=SafetyLevel.SAFE,
            description='Test event'
        )
        
        result = self.monitor._assess_impact(event)
        
        self.assertIn('level', result)
        self.assertIn('score', result)
    
    def test_callback_registration(self):
        """测试回调注册"""
        callback_calls = []
        
        def callback(event):
            callback_calls.append(event)
        
        self.monitor.register_callback(callback)
        
        # 触发一个事件
        behavior = {'type': 'creative', 'description': '创新行为'}
        self.monitor.track(behavior)
        
        # 回调应该被调用
        # 注意：这取决于是否检测到涌现
    
    def test_event_history(self):
        """测试事件历史"""
        # 添加一些事件
        for i in range(5):
            self.monitor.track({'type': 'behavioral', 'description': f'行为{i}'})
        
        history = self.monitor.get_event_history(limit=10)
        
        self.assertLessEqual(len(history), 10)
    
    def test_safety_summary(self):
        """测试安全摘要"""
        summary = self.monitor.get_safety_summary()
        
        self.assertIn('total_events', summary)
        self.assertIn('safety_distribution', summary)
    
    def test_impact_summary(self):
        """测试影响摘要"""
        summary = self.monitor.get_impact_summary()
        
        self.assertIn('total_events', summary)
        self.assertIn('impact_distribution', summary)
    
    def test_report_generation(self):
        """测试报告生成"""
        report = self.monitor.generate_report()
        
        self.assertIn('summary', report)
        self.assertIn('safety', report)
        self.assertIn('impact', report)
        self.assertIn('recommendations', report)
    
    def test_capability_status(self):
        """测试能力状态"""
        # 先创建一些数据
        self.monitor.track({'type': 'skill', 'description': '工具使用'})
        
        # 获取状态
        status = self.monitor.get_capability_status('unknown')
        self.assertIsNone(status)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流程"""
        # 初始化所有组件
        detector = CapabilityDetector()
        organizer = SelfOrganization()
        generator = CreativeGenerator()
        monitor = EmergenceMonitor()
        
        # 1. 检测能力
        capability = detector.detect(None, {
            'actions': ['learn', 'adapt', 'evolve'],
            'decisions': ['strategy'],
            'learning': 'continuous'
        })
        
        # 2. 自组织
        structure = organizer.organize()
        
        # 3. 创意生成
        solutions = generator.solve({'description': '优化涌现能力'})
        
        # 4. 监控
        monitor.track({
            'capability': capability.name if capability else None,
            'structure': structure.metadata,
            'solutions': [s.solution_id for s in solutions]
        })
        
        # 验证结果
        report = monitor.generate_report()
        
        self.assertIn('summary', report)
        self.assertIsInstance(report['summary']['total_events'], int)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestCapabilityDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestSelfOrganization))
    suite.addTests(loader.loadTestsFromTestCase(TestCreativeGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestEmergenceMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回结果
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
