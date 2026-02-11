"""
AIOps 使用示例

展示各模块的基本用法
"""

import json
from datetime import datetime, timedelta
from .anomaly_detector import AnomalyDetector
from .root_cause_analyzer import RootCauseAnalyzer
from .auto_recovery import AutoRecovery
from .predictive_maintenance import PredictiveMaintenance


def example_anomaly_detection():
    """异常检测示例"""
    print("=" * 60)
    print("异常检测示例")
    print("=" * 60)

    # 创建检测器
    detector = AnomalyDetector()

    # 模拟CPU使用率数据
    cpu_values = [50, 52, 51, 48, 55, 60, 75, 85, 92, 95, 98]

    print("\n逐步检测CPU使用率异常:")
    for i, cpu in enumerate(cpu_values):
        result = detector.detect_realtime({"cpu": cpu})
        print(f"  CPU: {cpu}% -> 状态: {result['status']}, 异常数: {result['anomaly_count']}")

    # 批量检测
    print("\n批量检测多个指标:")
    metrics = {
        "cpu": 85,
        "memory": 90,
        "latency": 350,
        "error_rate": 1.5,
    }
    result = detector.detect_realtime(metrics)
    print(f"  状态: {result['status']}")
    print(f"  检测到 {result['anomaly_count']} 个异常")
    for anomaly in result["anomalies"]:
        print(f"    - {anomaly['description']}")

    # 获取健康度评分
    health = detector.get_health_score()
    print(f"\n系统健康度: {health['score']} ({health['status']})")


def example_root_cause_analysis():
    """根因分析示例"""
    print("\n" + "=" * 60)
    print("根因分析示例")
    print("=" * 60)

    # 创建分析器
    analyzer = RootCauseAnalyzer()

    # 分析高延迟问题
    print("\n分析高延迟问题:")
    result = analyzer.analyze(
        symptom="high_latency",
        time_range="1h",
        affected_services=["api-gateway", "order-service"]
    )

    print(f"  根因ID: {result.id}")
    print(f"  可能原因: {result.node_name} ({result.node_type.value})")
    print(f"  置信度: {result.confidence:.2%}")
    print(f"  描述: {result.description}")
    print(f"  建议操作:")
    for action in result.suggested_actions:
        print(f"    - {action}")

    # 获取依赖拓扑
    print("\n获取依赖拓扑:")
    topology = analyzer.get_dependency_topology()
    print(f"  服务节点数: {len(topology['nodes'])}")
    print(f"  依赖关系数: {len(topology['edges'])}")

    # 查询受影响服务
    print("\n查询mysql-master的影响范围:")
    affected = analyzer.graph.get_all_affected("mysql-master")
    print(f"  上游依赖: {list(affected['upstream'])}")
    print(f"  下游影响: {list(affected['downstream'])}")


def example_auto_recovery():
    """自动恢复示例"""
    print("\n" + "=" * 60)
    print("自动恢复示例")
    print("=" * 60)

    # 创建恢复系统
    recovery = AutoRecovery()

    # 创建故障事件
    print("\n创建故障事件:")
    incident = recovery.create_incident(
        title="API Gateway CPU使用率过高",
        description="CPU使用率达到95%,可能导致服务响应变慢",
        severity="high",
        service="api-gateway",
        metrics={"cpu": 95, "memory": 70},
        context={"symptom": "high_cpu"}
    )
    print(f"  事件ID: {incident.id}")
    print(f"  状态: {incident.status.value}")

    # 生成恢复计划
    print("\n生成恢复计划:")
    plan = recovery.generate_recovery_plan(incident)
    print(f"  计划ID: {plan.id}")
    print(f"  包含 {len(plan.actions)} 个恢复动作:")
    for action in plan.actions:
        print(f"    {action.order}. {action.name} ({action.strategy.value})")

    # 执行自动恢复
    print("\n执行自动恢复:")
    result = recovery.execute_recovery(incident.id, strategy="auto_fix")
    print(f"  状态: {result.status.value}")
    print(f"  耗时: {result.total_time_ms:.2f}ms")
    print(f"  成功率: {result.success_rate:.1%}")

    # 获取恢复统计
    print("\n获取恢复统计:")
    stats = recovery.get_statistics()
    print(f"  总事件数: {stats['total_incidents']}")
    print(f"  自动恢复成功: {stats['auto_recovered']}")
    print(f"  自动恢复率: {stats['auto_recovery_rate']:.1f}%")


def example_predictive_maintenance():
    """预测性维护示例"""
    print("\n" + "=" * 60)
    print("预测性维护示例")
    print("=" * 60)

    # 创建预测系统
    predictor = PredictiveMaintenance()

    # 添加历史数据
    print("\n添加历史数据:")
    base_time = datetime.now()
    for i in range(168):  # 一周的数据 (每小时一个点)
        timestamp = base_time - timedelta(hours=167 - i)
        cpu_value = 50 + i * 0.3 + (i % 24) * 2  # 趋势上升 + 日周期
        predictor.add_metric_data("cpu", cpu_value, timestamp)
        predictor.add_metric_data("memory", 60 + i * 0.2 + (i % 24) * 3, timestamp)

    # 预测未来状态
    print("\n预测未来24小时状态:")
    predictions = predictor.predict(
        metrics={"cpu": 85, "memory": 80},
        hours_ahead=24
    )

    for pred in predictions:
        print(f"  {pred.target}:")
        print(f"    预警级别: {pred.alert_level.value}")
        print(f"    预测值: {pred.value:.1f}")
        print(f"    置信度: {pred.confidence:.2%}")
        print(f"    描述: {pred.description}")

    # 容量预测
    print("\n容量预测:")
    resources = {
        "cpu": {
            "current": 70,
            "threshold": 100,
            "unit": "%",
            "daily_growth": 0.02,
        },
        "memory": {
            "current": 75,
            "threshold": 100,
            "unit": "%",
            "daily_growth": 0.015,
        },
        "disk": {
            "current": 500,
            "threshold": 1000,
            "unit": "GB",
            "daily_growth": 5,
        },
    }
    forecasts = predictor.forecast_capacity(resources, days_ahead=30)

    for fc in forecasts:
        print(f"  {fc.resource}:")
        print(f"    当前使用: {fc.current_value}{fc.unit}")
        print(f"    预测值: {fc.predicted_value:.1f}{fc.unit}")
        print(f"    到达容量天数: {fc.days_until_capacity}")
        print(f"    预警级别: {fc.alert_level.value}")

    # 获取资源趋势
    print("\n获取CPU趋势预测:")
    trend = predictor.predict_resource_trend("cpu", hours_ahead=24)
    print(f"  当前趋势: {trend['current_trend']}")
    print(f"  每小时增长: {trend['slope_per_hour']:.2f}%")
    print(f"  置信度: {trend['confidence']:.2%}")
    print(f"  24小时后预测值: {trend['predicted_value_in_24h']:.1f}%")

    # 获取系统健康度预测
    print("\n系统健康度预测:")
    health = predictor.get_system_health_prediction()
    print(f"  健康度评分: {health['score']}")
    print(f"  状态: {health['status']}")
    print(f"  活跃预警数: {health['active_alerts']}")


def example_full_workflow():
    """完整工作流示例"""
    print("\n" + "=" * 60)
    print("完整工作流示例")
    print("=" * 60)

    # 初始化所有组件
    anomaly_detector = AnomalyDetector()
    root_cause_analyzer = RootCauseAnalyzer()
    auto_recovery = AutoRecovery()
    predictor = PredictiveMaintenance()

    # 1. 监控检测到异常
    print("\n1. 异常检测:")
    metrics = {"cpu": 92, "memory": 85, "latency": 800}
    anomaly_result = anomaly_detector.detect_realtime(metrics)
    print(f"   检测到 {anomaly_result['anomaly_count']} 个异常")
    print(f"   状态: {anomaly_result['status']}")

    # 2. 根因分析
    print("\n2. 根因分析:")
    root_cause = root_cause_analyzer.analyze(
        symptom="high_cpu",
        time_range="1h"
    )
    print(f"   可能根因: {root_cause.node_name}")
    print(f"   置信度: {root_cause.confidence:.2%}")

    # 3. 自动恢复
    print("\n3. 自动恢复:")
    incident = auto_recovery.create_incident(
        title="CPU使用率过高导致性能下降",
        description=f"检测到CPU使用率: {metrics['cpu']}%",
        severity="high",
        service="api-gateway",
        metrics=metrics,
        context={"root_cause_id": root_cause.id}
    )
    recovery_result = auto_recovery.execute_recovery(incident.id, strategy="auto_fix")
    print(f"   恢复状态: {recovery_result.status.value}")
    print(f"   成功率: {recovery_result.success_rate:.1%}")

    # 4. 预测性维护
    print("\n4. 预测性维护:")
    predictions = predictor.predict(metrics, hours_ahead=24)
    for pred in predictions:
        print(f"   {pred.target}: {pred.alert_level.value} - {pred.description}")


def run_all_examples():
    """运行所有示例"""
    print("\n" + "#" * 60)
    print("# AIOps 智能运维系统使用示例")
    print("#" * 60)

    example_anomaly_detection()
    example_root_cause_analysis()
    example_auto_recovery()
    example_predictive_maintenance()
    example_full_workflow()

    print("\n" + "#" * 60)
    print("# 所有示例运行完成")
    print("#" * 60)


if __name__ == "__main__":
    run_all_examples()
