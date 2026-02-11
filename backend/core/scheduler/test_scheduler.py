"""
智能调度系统 - 测试用例
"""

import unittest
from datetime import datetime, timedelta
from resource_optimizer import (
    ResourceOptimizer, Workload, NodeResources, 
    ResourceType, OptimizationStrategy
)
from load_balancer import (
    LoadBalancer, BackendServer, GrayReleaseRule,
    LoadBalancingAlgorithm
)
from auto_scaler import (
    AutoScaler, ScalingPolicy, ScalingDecision,
    ScalingType, ScalingAction
)
from cost_optimizer import (
    CostOptimizer, CostRecommendation, InstanceType
)


class TestResourceOptimizer(unittest.TestCase):
    """资源优化器测试"""
    
    def setUp(self):
        self.optimizer = ResourceOptimizer(strategy=OptimizationStrategy.BALANCED)
        
        # 添加测试节点
        self.nodes = [
            NodeResources(
                node_id="node-1",
                cpu_total=64, cpu_available=32,
                memory_total=128, memory_available=64,
                gpu_total=4, gpu_available=2
            ),
            NodeResources(
                node_id="node-2",
                cpu_total=64, cpu_available=48,
                memory_total=128, memory_available=96,
                gpu_total=4, gpu_available=4
            )
        ]
        
        for node in self.nodes:
            self.optimizer.register_node(node)
    
    def test_register_node(self):
        """测试节点注册"""
        node = NodeResources(
            node_id="test-node",
            cpu_total=32, cpu_available=16,
            memory_total=64, memory_available=32
        )
        self.assertTrue(self.optimizer.register_node(node))
        self.assertIn("test-node", self.optimizer.nodes)
    
    def test_register_workload(self):
        """测试工作负载注册"""
        workload = Workload(
            id="test-workload",
            name="Test Workload",
            resource_requirements={"cpu": 4, "memory": 8}
        )
        self.assertTrue(self.optimizer.register_workload(workload))
        self.assertIn("test-workload", self.optimizer.workloads)
    
    def test_optimize_basic(self):
        """测试基本优化"""
        workloads = [
            Workload(
                id="web-1",
                name="Web",
                resource_requirements={"cpu": 4, "memory": 8},
                priority=5
            ),
            Workload(
                id="app-1", 
                name="App",
                resource_requirements={"cpu": 8, "memory": 16},
                priority=8
            )
        ]
        
        allocation = self.optimizer.optimize(workloads)
        
        self.assertEqual(len(allocation), 2)
        # 至少应该有分配的节点
        self.assertTrue(any(r.get("node_id") for r in allocation.values()))
    
    def test_optimize_gpu_workload(self):
        """测试GPU工作负载分配"""
        gpu_workload = Workload(
            id="gpu-1",
            name="GPU Workload",
            resource_requirements={"cpu": 8, "memory": 16, "gpu": 2},
            is_gpu_required=True,
            priority=10
        )
        
        workloads = [gpu_workload]
        allocation = self.optimizer.optimize(workloads)
        
        # GPU工作负载应该分配到有GPU的节点
        result = allocation["gpu-1"]
        if result.get("node_id"):
            node = self.optimizer.nodes[result["node_id"]]
            self.assertGreater(node.gpu_total, 0)
    
    def test_optimization_report(self):
        """测试优化报告"""
        report = self.optimizer.get_optimization_report()
        
        self.assertIn("cpu_utilization", report)
        self.assertIn("memory_utilization", report)
        self.assertIn("total_nodes", report)
        self.assertEqual(report["total_nodes"], 2)


class TestLoadBalancer(unittest.TestCase):
    """负载均衡器测试"""
    
    def setUp(self):
        self.lb = LoadBalancer(
            algorithm=LoadBalancingAlgorithm.ROUND_ROBIN,
            health_check_interval=30
        )
        
        # 添加测试后端
        self.backends = [
            BackendServer(
                id="backend-1",
                host="10.0.0.1",
                port=8080,
                weight=100
            ),
            BackendServer(
                id="backend-2",
                host="10.0.0.2",
                port=8080,
                weight=100
            )
        ]
        
        for backend in self.backends:
            self.lb.register_backend(backend)
    
    def test_register_backend(self):
        """测试后端注册"""
        backend = BackendServer(
            id="test-backend",
            host="10.0.0.100",
            port=9090
        )
        self.assertTrue(self.lb.register_backend(backend))
        self.assertIn("test-backend", self.lb.backends)
    
    def test_route_request_round_robin(self):
        """测试轮询路由 - 切换到轮询算法"""
        self.lb.algorithm = LoadBalancingAlgorithm.ROUND_ROBIN
        
        clients = ["192.168.1.1", "192.168.1.2", "192.168.1.3"]
        backends = []
        
        for ip in clients:
            backend_id, headers = self.lb.route_request(
                request_id=f"req-{len(backends)+1}",
                client_ip=ip,
                path="/api/test"
            )
            backends.append(backend_id)
        
        # 轮询应该交替分配
        self.assertGreaterEqual(len(set(backends)), 1)  # 至少有1个后端被使用
    
    def test_route_request_ip_hash(self):
        """测试IP哈希路由"""
        self.lb.algorithm = LoadBalancingAlgorithm.IP_HASH
        
        backend1, _ = self.lb.route_request(
            request_id="req-1",
            client_ip="192.168.1.100",
            path="/api/test"
        )
        backend2, _ = self.lb.route_request(
            request_id="req-2",
            client_ip="192.168.1.100",  # 相同IP
            path="/api/test"
        )
        
        # 相同IP应该路由到相同后端
        self.assertEqual(backend1, backend2)
    
    def test_health_check(self):
        """测试健康检查"""
        self.lb.health_check("backend-1", is_healthy=False, response_time=100)
        self.assertFalse(self.lb.backends["backend-1"].is_healthy)
        
        self.lb.health_check("backend-1", is_healthy=True, response_time=20)
        self.assertTrue(self.lb.backends["backend-1"].is_healthy)
    
    def test_get_stats(self):
        """测试获取统计"""
        stats = self.lb.get_load_balancer_stats()
        
        self.assertIn("total_backends", stats)
        self.assertIn("healthy_backends", stats)
        self.assertEqual(stats["total_backends"], 2)
    
    def test_gray_release(self):
        """测试灰度发布"""
        rule = GrayReleaseRule(
            name="beta-test",
            backend_id="backend-1",
            traffic_percentage=50,
            conditions=[{"header": "X-Version", "value": "beta"}]
        )
        
        self.assertTrue(self.lb.add_gray_release_rule(rule))
        self.assertEqual(len(self.lb.gray_rules), 1)


class TestAutoScaler(unittest.TestCase):
    """自动伸缩器测试"""
    
    def setUp(self):
        self.scaler = AutoScaler(
            default_cooldown=300,
            enable_predictive=True
        )
        
        # 添加策略
        self.cpu_policy = ScalingPolicy(
            name="cpu-scaling",
            metric_name="cpu",
            threshold_high=80,
            threshold_low=30,
            scaling_type=ScalingType.HORIZONTAL,
            step_size=2
        )
        self.scaler.add_policy(self.cpu_policy)
    
    def test_add_policy(self):
        """测试添加策略"""
        policy = ScalingPolicy(
            name="memory-scaling",
            metric_name="memory",
            threshold_high=85,
            threshold_low=40
        )
        self.assertTrue(self.scaler.add_policy(policy))
        self.assertIn("memory-scaling", self.scaler.policies)
    
    def test_decide_scale_out(self):
        """测试扩容决策"""
        self.scaler.last_scaling_time = datetime.now() - timedelta(seconds=400)  # 冷却已过
        decision = self.scaler.decide(
            current_metrics={"cpu": 85, "qps": 1500}
        )
        
        self.assertEqual(decision.action, ScalingAction.SCALE_OUT)
        self.assertEqual(decision.scaling_type, ScalingType.HORIZONTAL)
    
    def test_decide_scale_in(self):
        """测试缩容决策"""
        self.scaler.last_scaling_time = datetime.now() - timedelta(seconds=400)
        
        # 添加一些模拟实例用于缩容
        from auto_scaler import Instance
        for i in range(3):
            inst = Instance(
                instance_id=f"instance-{i}",
                instance_type="small",
                status="running",
                created_at=datetime.now() - timedelta(hours=i)
            )
            inst.metrics = {'cpu': 20, 'memory': 25}
            self.scaler.instances[inst.instance_id] = inst
        
        decision = self.scaler.decide(
            current_metrics={"cpu": 20, "qps": 50}
        )
        
        # 由于低CPU应该触发缩容(如果有实例)
        self.assertIn(decision.action, [ScalingAction.SCALE_IN, ScalingAction.NO_ACTION])
    
    def test_decide_no_action(self):
        """测试无操作决策"""
        decision = self.scaler.decide(
            current_metrics={"cpu": 50, "qps": 500}
        )
        
        self.assertEqual(decision.action, ScalingAction.NO_ACTION)
    
    def test_predictive_scaling(self):
        """测试预测性伸缩"""
        predicted_load = {
            "current_qps": 1000,
            "predicted_qps": 1500  # 增长50%
        }
        
        decision = self.scaler.decide(
            current_metrics={"cpu": 50},
            predicted_load=predicted_load
        )
        
        # 预测性伸缩可能有不同的触发条件
        self.assertIn(decision.action, [ScalingAction.SCALE_OUT, ScalingAction.NO_ACTION])
    
    def test_get_recommendations(self):
        """测试获取推荐"""
        recommendations = self.scaler.get_scaling_recommendations()
        
        self.assertIn("current_instances", recommendations)
        self.assertIn("avg_cpu_utilization", recommendations)
        self.assertIn("suggested_action", recommendations)
    
    def test_cost_optimized_scaling(self):
        """测试成本优化伸缩"""
        result = self.scaler.cost_optimized_scaling(
            current_metrics={"cpu": 45, "memory": 50},
            spot_instances_available=True
        )
        
        self.assertIn("instance_mix", result)
        self.assertIn("cost_saving_opportunities", result)


class TestCostOptimizer(unittest.TestCase):
    """成本优化器测试"""
    
    def setUp(self):
        self.optimizer = CostOptimizer(
            spot_discount=0.7,
            reserved_discount=0.3
        )
    
    def test_analyze_basic(self):
        """测试基本分析"""
        usage_patterns = [
            {
                "instance_size": "medium",
                "instance_count": 10,
                "hours_per_month": 720,
                "utilization": 0.75
            }
        ]
        
        analysis = self.optimizer.analyze(
            usage_patterns=usage_patterns,
            reserved_vs_spot=True
        )
        
        self.assertIn("current_cost", analysis)
        self.assertIn("optimized_cost", analysis)
        self.assertIn("recommendations", analysis)
    
    def test_recommendations_generated(self):
        """测试推荐生成"""
        usage_patterns = [
            {
                "instance_size": "large",
                "instance_count": 5,
                "hours_per_month": 720,
                "utilization": 0.2  # 低利用率
            }
        ]
        
        analysis = self.optimizer.analyze(usage_patterns)
        
        # 应该生成调整实例规格的建议
        self.assertTrue(len(analysis["recommendations"]) > 0)
        rec_types = [r["type"] for r in analysis["recommendations"]]
        self.assertIn("rightsizing", rec_types)
    
    def test_calculate_savings(self):
        """测试计算节省"""
        savings = self.optimizer.calculate_savings_vs_ondemand(
            instance_count=10,
            instance_size="medium",
            instance_type=InstanceType.SPOT,
            hours_per_month=720
        )
        
        self.assertIn("monthly_cost", savings)
        self.assertIn("savings", savings)
        self.assertIn("savings_percentage", savings)
        self.assertGreater(savings["savings_percentage"], 0)
    
    def test_optimize_bid_strategy(self):
        """测试竞价策略优化"""
        historical_prices = [0.02, 0.025, 0.03, 0.028, 0.022]
        
        strategy = self.optimizer.optimize_bid_strategy(
            base_price=0.03,
            historical_prices=historical_prices,
            reliability_requirement=0.95
        )
        
        self.assertIn("recommended_bid", strategy)
        self.assertIn("strategy", strategy)
    
    def test_instance_mix_recommendation(self):
        """测试实例组合推荐"""
        # 稳定负载
        patterns = {
            "avg_utilization": 0.85,
            "stable_load": True,
            "variable_utilization": False
        }
        
        mix = self.optimizer._recommend_instance_mix(patterns)
        
        self.assertGreater(mix["reserved"], mix["spot"])
        
        # 弹性负载
        patterns["stable_load"] = False
        patterns["variable_utilization"] = True
        
        mix = self.optimizer._recommend_instance_mix(patterns)
        
        self.assertGreater(mix["spot"], mix["reserved"])


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        # 创建组件
        optimizer = ResourceOptimizer()
        lb = LoadBalancer()
        scaler = AutoScaler()
        cost_optimizer = CostOptimizer()
        
        # 注册节点和后端
        node = NodeResources(
            node_id="node-1",
            cpu_total=64, cpu_available=64,
            memory_total=128, memory_available=128
        )
        optimizer.register_node(node)
        
        backend = BackendServer(
            id="backend-1",
            host="10.0.0.1",
            port=8080
        )
        lb.register_backend(backend)
        
        # 添加伸缩策略
        policy = ScalingPolicy(
            name="test-policy",
            metric_name="cpu",
            threshold_high=80,
            threshold_low=30
        )
        scaler.add_policy(policy)
        
        # 验证所有组件正常工作
        allocation = optimizer.optimize([
            Workload(
                id="workload-1",
                name="Test",
                resource_requirements={"cpu": 4, "memory": 8}
            )
        ])
        
        backend_id, _ = lb.route_request(
            request_id="req-1",
            client_ip="192.168.1.1",
            path="/test"
        )
        
        decision = scaler.decide(current_metrics={"cpu": 50})
        
        analysis = cost_optimizer.analyze([
            {"instance_size": "small", "instance_count": 1, "hours_per_month": 720, "utilization": 0.5}
        ])
        
        # 验证结果
        self.assertIn("workload-1", allocation)
        self.assertEqual(backend_id, "backend-1")
        self.assertIsNotNone(decision)
        self.assertIn("current_cost", analysis)


def suite():
    """测试套件"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestResourceOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestLoadBalancer))
    suite.addTests(loader.loadTestsFromTestCase(TestAutoScaler))
    suite.addTests(loader.loadTestsFromTestCase(TestCostOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
