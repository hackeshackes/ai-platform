"""
智能调度系统 - 使用示例
"""

from resource_optimizer import ResourceOptimizer, Workload, NodeResources, ResourceType, OptimizationStrategy
from load_balancer import LoadBalancer, BackendServer, LoadBalancingAlgorithm
from auto_scaler import AutoScaler, ScalingPolicy, ScalingType
from cost_optimizer import CostOptimizer, InstanceType


def example_resource_optimizer():
    """资源优化器使用示例"""
    print("=" * 60)
    print("资源优化器示例")
    print("=" * 60)
    
    # 创建优化器
    optimizer = ResourceOptimizer(strategy=OptimizationStrategy.BALANCED)
    
    # 注册节点
    nodes = [
        NodeResources(
            node_id="node-1",
            cpu_total=64, cpu_available=32,
            memory_total=128, memory_available=64,
            gpu_total=4, gpu_available=2,
            network_bandwidth=10000
        ),
        NodeResources(
            node_id="node-2",
            cpu_total=64, cpu_available=48,
            memory_total=128, memory_available=96,
            gpu_total=4, gpu_available=4,
            network_bandwidth=10000
        ),
        NodeResources(
            node_id="node-3",
            cpu_total=32, cpu_available=16,
            memory_total=64, memory_available=32,
            gpu_total=0, gpu_available=0,
            network_bandwidth=5000
        )
    ]
    
    for node in nodes:
        optimizer.register_node(node)
    
    # 注册工作负载
    workloads = [
        Workload(
            id="web-1",
            name="Web Server",
            resource_requirements={"cpu": 4, "memory": 8},
            priority=8
        ),
        Workload(
            id="ml-1",
            name="ML Inference",
            resource_requirements={"cpu": 8, "memory": 16, "gpu": 2},
            priority=10,
            is_gpu_required=True
        ),
        Workload(
            id="db-1",
            name="Database",
            resource_requirements={"cpu": 8, "memory": 32},
            priority=9
        )
    ]
    
    for workload in workloads:
        optimizer.register_workload(workload)
    
    # 执行优化
    allocation = optimizer.optimize(
        workloads=workloads,
        constraints={"cpu": "max", "cost": "min"}
    )
    
    print("\n分配结果:")
    for workload_id, result in allocation.items():
        print(f"  {workload_id}: {result}")
    
    # 获取报告
    report = optimizer.get_optimization_report()
    print(f"\n优化报告:")
    for key, value in report.items():
        print(f"  {key}: {value}")


def example_load_balancer():
    """负载均衡器使用示例"""
    print("\n" + "=" * 60)
    print("负载均衡器示例")
    print("=" * 60)
    
    # 创建负载均衡器
    lb = LoadBalancer(
        algorithm=LoadBalancingAlgorithm.ADAPTIVE,
        health_check_interval=30
    )
    
    # 注册后端
    backends = [
        BackendServer(
            id="backend-1",
            host="10.0.0.1",
            port=8080,
            weight=100,
            max_connections=1000
        ),
        BackendServer(
            id="backend-2",
            host="10.0.0.2",
            port=8080,
            weight=100,
            max_connections=1000
        ),
        BackendServer(
            id="backend-3",
            host="10.0.0.3",
            port=8080,
            weight=150,
            max_connections=1500
        )
    ]
    
    for backend in backends:
        lb.register_backend(backend)
    
    # 模拟请求路由
    requests = [
        {"client_ip": "192.168.1.100", "path": "/api/users", "session_id": "sess-123"},
        {"client_ip": "192.168.1.101", "path": "/api/products"},
        {"client_ip": "192.168.1.102", "path": "/api/orders"}
    ]
    
    print("\n请求路由:")
    for req in requests:
        backend_id, headers = lb.route_request(
            request_id=f"req-{requests.index(req) + 1}",
            client_ip=req["client_ip"],
            path=req["path"],
            session_id=req.get("session_id")
        )
        print(f"  {req['path']} -> {backend_id}")
    
    # 获取统计
    stats = lb.get_load_balancer_stats()
    print(f"\n负载均衡统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_auto_scaler():
    """自动伸缩器使用示例"""
    print("\n" + "=" * 60)
    print("自动伸缩器示例")
    print("=" * 60)
    
    # 创建伸缩器
    scaler = AutoScaler(
        default_cooldown=300,
        enable_predictive=True
    )
    
    # 添加策略
    policies = [
        ScalingPolicy(
            name="cpu-scaling",
            metric_name="cpu",
            threshold_high=80,
            threshold_low=30,
            scaling_type=ScalingType.HORIZONTAL,
            step_size=2,
            cooldown_seconds=300
        ),
        ScalingPolicy(
            name="memory-scaling",
            metric_name="memory",
            threshold_high=85,
            threshold_low=40,
            scaling_type=ScalingType.HORIZONTAL,
            step_size=1
        )
    ]
    
    for policy in policies:
        scaler.add_policy(policy)
    
    # 模拟当前指标
    scenarios = [
        {"cpu": 85, "qps": 1200, "memory": 60},
        {"cpu": 25, "qps": 80, "memory": 35},
        {"cpu": 45, "qps": 500, "memory": 50}
    ]
    
    print("\n伸缩决策:")
    for i, metrics in enumerate(scenarios):
        decision = scaler.decide(
            current_metrics=metrics,
            target_response_time=100
        )
        print(f"\n场景 {i + 1} (指标: {metrics}):")
        print(f"  决策: {decision.action.value}")
        print(f"  原因: {decision.reason}")
    
    # 获取推荐
    recommendations = scaler.get_scaling_recommendations()
    print(f"\n伸缩推荐: {recommendations}")


def example_cost_optimizer():
    """成本优化器使用示例"""
    print("\n" + "=" * 60)
    print("成本优化器示例")
    print("=" * 60)
    
    # 创建优化器
    cost_optimizer = CostOptimizer(
        spot_discount=0.7,
        reserved_discount=0.3
    )
    
    # 使用模式分析
    usage_patterns = [
        {"instance_size": "medium", "instance_count": 10, "hours_per_month": 720, "utilization": 0.85},
        {"instance_size": "large", "instance_count": 5, "hours_per_month": 720, "utilization": 0.25}
    ]
    
    print("\n成本分析:")
    analysis = cost_optimizer.analyze(
        usage_patterns=usage_patterns,
        reserved_vs_spot=True
    )
    
    for key, value in analysis.items():
        print(f"  {key}: {value}")


def main():
    """运行所有示例"""
    print("智能调度系统 - 使用示例\n")
    
    example_resource_optimizer()
    example_load_balancer()
    example_auto_scaler()
    example_cost_optimizer()
    
    print("\n所有示例运行完成!")


if __name__ == "__main__":
    main()
