"""
Emergence Engine Examples - 涌现引擎使用示例
"""

from typing import Dict, List
from capability_detector import CapabilityDetector, EmergenceType, CapabilityLevel
from self_organization import SelfOrganization, PlasticityRule
from creative_generator import CreativeGenerator, InnovationType
from emergence_monitor import EmergenceMonitor


def example_capability_detection():
    """能力检测示例"""
    detector = CapabilityDetector()
    
    # 模拟模型
    class MockModel:
        pass
    
    model = MockModel()
    
    # 模拟交互数据
    interactions = [
        {
            'actions': ['analyze', 'query', 'respond'],
            'decisions': ['select_best_response'],
            'strategies': ['context_aware']
        },
        {
            'actions': ['search', 'retrieve', 'summarize'],
            'decisions': ['filter_relevant'],
            'learning': 'pattern_recognition'
        },
        {
            'actions': ['plan', 'execute', 'verify'],
            'decisions': ['optimize_approach'],
            'strategies': ['step_by_step']
        }
    ]
    
    # 检测能力涌现
    capabilities = []
    for interaction in interactions:
        capability = detector.detect(model, interaction)
        if capability:
            capabilities.append(capability)
            print(f"检测到能力: {capability.name}")
            print(f"  类型: {capability.emergence_type.value}")
            print(f"  置信度: {capability.confidence:.2f}")
            print(f"  行为: {capability.behaviors}")
    
    return capabilities


def example_self_organization():
    """自组织示例"""
    organizer = SelfOrganization()
    
    # 创建初始神经元
    for i in range(10):
        organizer.add_neuron(layer=i % 3, position=(i, i))
    
    # 创建初始连接
    for i in range(9):
        organizer.add_synapse(f"neuron_{i}", f"neuron_{i+1}", 
                              weight=0.5, 
                              plasticity_rule=PlasticityRule.HEBBIAN)
    
    # 执行自组织
    for _ in range(100):
        structure = organizer.organize()
    
    # 获取统计信息
    stats = organizer.get_statistics()
    print(f"网络统计:")
    print(f"  神经元数: {stats['total_neurons']}")
    print(f"  连接数: {stats['total_connections']}")
    print(f"  网络类型: {stats['network_type']}")
    
    return stats


def example_creative_generation():
    """创意生成示例"""
    generator = CreativeGenerator()
    
    # 定义问题
    problem = {
        'description': '提高用户参与度的解决方案',
        'current_metrics': {'engagement': 0.3, 'retention': 0.4},
        'target_metrics': {'engagement': 0.6, 'retention': 0.7}
    }
    
    constraints = {
        'budget': 'limited',
        'timeline': '3 months',
        'feasibility': 'high'
    }
    
    # 创造性解决问题
    solutions = generator.solve(problem, constraints)
    
    print(f"生成了 {len(solutions)} 个解决方案:")
    for sol in solutions[:3]:
        print(f"\n{sol.description}")
        print(f"  创新类型: {sol.innovation_type.value}")
        print(f"  新颖度: {sol.novelty_score:.2f}")
        print(f"  适用性: {sol.applicability:.2f}")
        print(f"  置信度: {sol.confidence:.2f}")
        print(f"  步骤: {sol.steps}")
    
    return solutions


def example_emergence_monitoring():
    """涌现监控示例"""
    monitor = EmergenceMonitor()
    
    # 注册事件回调
    def on_emergence(event):
        print(f"涌现事件: {event.description}")
        print(f"  安全级别: {event.safety.value}")
        print(f"  影响级别: {event.impact.value}")
    
    monitor.register_callback(on_emergence)
    
    # 模拟代理行为
    behaviors = [
        {'type': 'behavioral', 'actions': ['query', 'respond'], 'description': '对话行为'},
        {'type': 'cognitive', 'actions': ['analyze', 'infer'], 'description': '推理行为'},
        {'type': 'skill', 'actions': ['use_tool', 'execute'], 'description': '工具使用'},
        {'type': 'creative', 'description': '创新解决方案'},
    ]
    
    events = []
    for behavior in behaviors:
        event = monitor.track(behavior)
        if event:
            events.append(event)
    
    # 生成报告
    report = monitor.generate_report()
    print(f"\n监控报告:")
    print(f"  总事件数: {report['summary']['total_events']}")
    print(f"  安全分布: {report['safety']['safety_distribution']}")
    print(f"  影响分布: {report['impact']['impact_distribution']}")
    print(f"  建议: {report['recommendations']}")
    
    return events, report


def example_integration():
    """集成示例"""
    # 初始化所有组件
    detector = CapabilityDetector()
    organizer = SelfOrganization()
    generator = CreativeGenerator()
    monitor = EmergenceMonitor()
    
    # 模拟完整流程
    # 1. 检测到新行为
    interaction = {
        'actions': ['plan', 'adapt', 'learn'],
        'decisions': ['strategy_selection'],
        'learning': 'continuous_improvement'
    }
    
    capability = detector.detect(None, interaction)
    
    # 2. 如果检测到能力，进行自组织
    if capability:
        structure = organizer.organize()
        
        # 3. 生成创新解决方案
        problem = {'description': '优化新发现的能力'}
        solutions = generator.solve(problem)
        
        # 4. 监控整个过程
        monitor.track({
            'capability': capability.name,
            'structure': structure.metadata,
            'solutions': [s.solution_id for s in solutions]
        })
    
    return {
        'capability': capability.name if capability else None,
        'structure_type': organizer.structure.network_type.value,
        'solutions_count': len(solutions) if 'solutions' in dir() else 0
    }


if __name__ == '__main__':
    print("=" * 50)
    print("涌现引擎使用示例")
    print("=" * 50)
    
    print("\n1. 能力检测示例:")
    example_capability_detection()
    
    print("\n2. 自组织示例:")
    example_self_organization()
    
    print("\n3. 创意生成示例:")
    example_creative_generation()
    
    print("\n4. 涌现监控示例:")
    example_emergence_monitoring()
    
    print("\n5. 集成示例:")
    example_integration()
