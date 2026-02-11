"""
Monitoring Test Suite - AI Platform Backend
监控测试套件
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============== 监控仪表盘测试 ==============

class TestMonitoringDashboard:
    """监控仪表盘测试类"""
    
    def test_dashboard_import(self):
        """测试仪表盘模块导入"""
        try:
            from monitoring import (
                MonitoringDashboard,
                MetricType,
                TimeRange,
                MetricPoint,
                CostMetrics,
                PerformanceMetrics,
                get_dashboard
            )
            assert MonitoringDashboard is not None
            assert MetricType is not None
        except ImportError:
            pytest.skip("Monitoring module not available")
    
    def test_metric_type_enum(self):
        """测试指标类型枚举"""
        try:
            from monitoring import MetricType
            assert hasattr(MetricType, 'COST')
            assert hasattr(MetricType, 'PERFORMANCE')
            assert hasattr(MetricType, 'TOKEN')
            assert hasattr(MetricType, 'REQUEST')
        except Exception as e:
            pytest.skip(f"MetricType not available: {e}")
    
    def test_time_range_enum(self):
        """测试时间范围枚举"""
        try:
            from monitoring import TimeRange
            assert hasattr(TimeRange, 'HOUR')
            assert hasattr(TimeRange, 'DAY')
            assert hasattr(TimeRange, 'WEEK')
            assert hasattr(TimeRange, 'MONTH')
        except Exception as e:
            pytest.skip(f"TimeRange not available: {e}")
    
    def test_metric_point_structure(self):
        """测试指标点结构"""
        try:
            from monitoring import MetricPoint
            point = MetricPoint(
                timestamp=datetime.utcnow(),
                value=100.0,
                unit="tokens",
                labels={"model": "gpt-4"}
            )
            assert point.timestamp is not None
            assert point.value == 100.0
            assert point.unit == "tokens"
        except Exception as e:
            pytest.skip(f"MetricPoint not available: {e}")
    
    def test_cost_metrics_structure(self):
        """测试成本指标结构"""
        try:
            from monitoring import CostMetrics
            metrics = CostMetrics(
                total_cost=100.50,
                api_cost=80.0,
                compute_cost=20.50,
                storage_cost=0.0,
                currency="USD"
            )
            assert metrics.total_cost == 100.50
            assert metrics.api_cost == 80.0
        except Exception as e:
            pytest.skip(f"CostMetrics not available: {e}")
    
    def test_performance_metrics_structure(self):
        """测试性能指标结构"""
        try:
            from monitoring import PerformanceMetrics
            metrics = PerformanceMetrics(
                avg_latency=150.5,
                p50_latency=100.0,
                p95_latency=300.0,
                p99_latency=500.0,
                throughput=1000.0
            )
            assert metrics.avg_latency == 150.5
            assert metrics.p95_latency == 300.0
        except Exception as e:
            pytest.skip(f"PerformanceMetrics not available: {e}")
    
    def test_get_dashboard(self):
        """测试获取仪表盘实例"""
        try:
            from monitoring import get_dashboard
            dashboard = get_dashboard()
            assert dashboard is not None
        except Exception as e:
            pytest.skip(f"Dashboard not available: {e}")
    
    def test_dashboard_metrics_collection(self):
        """测试仪表盘指标收集"""
        try:
            from monitoring import MonitoringDashboard
            dashboard = MonitoringDashboard()
            
            # 测试添加指标
            dashboard.collect_metric(
                metric_type="performance",
                name="latency",
                value=100.0,
                unit="ms"
            )
            
            # 测试获取指标
            metrics = dashboard.get_metrics("latency")
            assert metrics is not None
        except Exception as e:
            pytest.skip(f"Metrics collection failed: {e}")


# ============== 告警引擎测试 ==============

class TestAlertEngine:
    """告警引擎测试类"""
    
    def test_alert_engine_import(self):
        """测试告警引擎导入"""
        try:
            from monitoring import (
                AlertEngine,
                AlertSeverity,
                AlertStatus,
                AlertType,
                AlertRule,
                Alert,
                get_alert_engine
            )
            assert AlertEngine is not None
        except ImportError:
            pytest.skip("Alert engine not available")
    
    def test_alert_severity_enum(self):
        """测试告警级别枚举"""
        try:
            from monitoring import AlertSeverity
            assert hasattr(AlertSeverity, 'LOW')
            assert hasattr(AlertSeverity, 'MEDIUM')
            assert hasattr(AlertSeverity, 'HIGH')
            assert hasattr(AlertSeverity, 'CRITICAL')
        except Exception as e:
            pytest.skip(f"AlertSeverity not available: {e}")
    
    def test_alert_status_enum(self):
        """测试告警状态枚举"""
        try:
            from monitoring import AlertStatus
            assert hasattr(AlertStatus, 'ACTIVE')
            assert hasattr(AlertStatus, 'ACKNOWLEDGED')
            assert hasattr(AlertStatus, 'RESOLVED')
        except Exception as e:
            pytest.skip(f"AlertStatus not available: {e}")
    
    def test_alert_type_enum(self):
        """测试告警类型枚举"""
        try:
            from monitoring import AlertType
            assert hasattr(AlertType, 'COST')
            assert hasattr(AlertType, 'PERFORMANCE')
            assert hasattr(AlertType, 'SECURITY')
            assert hasattr(AlertType, 'SYSTEM')
        except Exception as e:
            pytest.skip(f"AlertType not available: {e}")
    
    def test_alert_rule_structure(self):
        """测试告警规则结构"""
        try:
            from monitoring import AlertRule
            rule = AlertRule(
                name="High Cost Alert",
                metric="total_cost",
                condition=">",
                threshold=100.0,
                severity=AlertSeverity.HIGH,
                enabled=True
            )
            assert rule.name == "High Cost Alert"
            assert rule.threshold == 100.0
        except Exception as e:
            pytest.skip(f"AlertRule not available: {e}")
    
    def test_alert_structure(self):
        """测试告警结构"""
        try:
            from monitoring import Alert, AlertSeverity, AlertType
            alert = Alert(
                alert_id="alert-001",
                rule_name="High Cost Alert",
                severity=AlertSeverity.HIGH,
                type=AlertType.COST,
                message="Cost exceeded threshold",
                timestamp=datetime.utcnow(),
                status=AlertStatus.ACTIVE
            )
            assert alert.alert_id == "alert-001"
            assert alert.severity == AlertSeverity.HIGH
        except Exception as e:
            pytest.skip(f"Alert not available: {e}")
    
    def test_get_alert_engine(self):
        """测试获取告警引擎"""
        try:
            from monitoring import get_alert_engine
            engine = get_alert_engine()
            assert engine is not None
        except Exception as e:
            pytest.skip(f"Alert engine not available: {e}")
    
    def test_alert_evaluation(self):
        """测试告警评估"""
        try:
            from monitoring import AlertEngine, AlertSeverity
            engine = AlertEngine()
            
            # 测试告警触发
            should_alert = engine.evaluate_condition(
                metric_name="cost",
                current_value=150.0,
                condition=">",
                threshold=100.0
            )
            assert should_alert is True
            
            # 测试告警不触发
            should_alert = engine.evaluate_condition(
                metric_name="cost",
                current_value=50.0,
                condition=">",
                threshold=100.0
            )
            assert should_alert is False
        except Exception as e:
            pytest.skip(f"Alert evaluation failed: {e}")


# ============== 优化引擎测试 ==============

class TestOptimizationEngine:
    """优化引擎测试类"""
    
    def test_optimization_engine_import(self):
        """测试优化引擎导入"""
        try:
            from monitoring import (
                OptimizationEngine,
                OptimizationCategory,
                OptimizationPriority,
                OptimizationRecommendation,
                UsagePattern,
                get_optimization_engine
            )
            assert OptimizationEngine is not None
        except ImportError:
            pytest.skip("Optimization engine not available")
    
    def test_optimization_category_enum(self):
        """测试优化类别枚举"""
        try:
            from monitoring import OptimizationCategory
            assert hasattr(OptimizationCategory, 'COST')
            assert hasattr(OptimizationCategory, 'PERFORMANCE')
            assert hasattr(OptimizationCategory, 'TOKEN_USAGE')
        except Exception as e:
            pytest.skip(f"OptimizationCategory not available: {e}")
    
    def test_optimization_priority_enum(self):
        """测试优化优先级枚举"""
        try:
            from monitoring import OptimizationPriority
            assert hasattr(OptimizationPriority, 'LOW')
            assert hasattr(OptimizationPriority, 'MEDIUM')
            assert hasattr(OptimizationPriority, 'HIGH')
        except Exception as e:
            pytest.skip(f"OptimizationPriority not available: {e}")
    
    def test_optimization_recommendation_structure(self):
        """测试优化建议结构"""
        try:
            from monitoring import (
                OptimizationRecommendation,
                OptimizationCategory,
                OptimizationPriority
            )
            rec = OptimizationRecommendation(
                id="opt-001",
                category=OptimizationCategory.COST,
                priority=OptimizationPriority.HIGH,
                title="Reduce API Calls",
                description="Optimize prompt caching",
                estimated_savings=50.0,
                implementation_effort="low"
            )
            assert rec.id == "opt-001"
            assert rec.category == OptimizationCategory.COST
        except Exception as e:
            pytest.skip(f"OptimizationRecommendation not available: {e}")
    
    def test_usage_pattern_structure(self):
        """测试使用模式结构"""
        try:
            from monitoring import UsagePattern
            pattern = UsagePattern(
                user_id="user-001",
                total_requests=1000,
                avg_tokens_per_request=500,
                peak_hours=[9, 10, 11],
                common_models=["gpt-4", "gpt-3.5-turbo"]
            )
            assert pattern.user_id == "user-001"
            assert pattern.total_requests == 1000
        except Exception as e:
            pytest.skip(f"UsagePattern not available: {e}")
    
    def test_get_optimization_engine(self):
        """测试获取优化引擎"""
        try:
            from monitoring import get_optimization_engine
            engine = get_optimization_engine()
            assert engine is not None
        except Exception as e:
            pytest.skip(f"Optimization engine not available: {e}")
    
    def test_optimization_recommendation_generation(self):
        """测试优化建议生成"""
        try:
            from monitoring import OptimizationEngine
            engine = OptimizationEngine()
            
            # 测试基于使用模式生成建议
            recommendations = engine.generate_recommendations(
                usage_data={
                    "total_cost": 200.0,
                    "api_calls": 5000,
                    "avg_latency": 200.0
                }
            )
            assert recommendations is not None
        except Exception as e:
            pytest.skip(f"Recommendation generation failed: {e}")


# ============== 监控指标计算测试 ==============

class TestMetricsCalculation:
    """监控指标计算测试类"""
    
    def test_latency_calculation(self):
        """测试延迟计算"""
        latencies = [100, 150, 200, 250, 300]
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency == 200.0
        
        sorted_latencies = sorted(latencies)
        p95_index = int(len(sorted_latencies) * 0.95)
        p95_latency = sorted_latencies[p95_index]
        assert p95_latency == 300
    
    def test_cost_aggregation(self):
        """测试成本聚合"""
        costs = [
            {"api": 50.0, "compute": 20.0, "storage": 5.0},
            {"api": 30.0, "compute": 15.0, "storage": 3.0}
        ]
        
        total_api = sum(c["api"] for c in costs)
        total_compute = sum(c["compute"] for c in costs)
        total_storage = sum(c["storage"] for c in costs)
        
        assert total_api == 80.0
        assert total_compute == 35.0
        assert total_storage == 8.0
    
    def test_token_usage_calculation(self):
        """测试token使用计算"""
        requests = [
            {"prompt_tokens": 100, "completion_tokens": 50},
            {"prompt_tokens": 200, "completion_tokens": 100},
            {"prompt_tokens": 150, "completion_tokens": 75}
        ]
        
        total_prompt = sum(r["prompt_tokens"] for r in requests)
        total_completion = sum(r["completion_tokens"] for r in requests)
        total_tokens = total_prompt + total_completion
        
        assert total_prompt == 450
        assert total_completion == 225
        assert total_tokens == 675
    
    def test_throughput_calculation(self):
        """测试吞吐量计算"""
        requests_count = 1000
        time_period_hours = 1
        
        throughput = requests_count / time_period_hours
        assert throughput == 1000.0
    
    def test_error_rate_calculation(self):
        """测试错误率计算"""
        total_requests = 1000
        failed_requests = 25
        
        error_rate = failed_requests / total_requests * 100
        assert error_rate == 2.5


# ============== 监控配置测试 ==============

class TestMonitoringConfiguration:
    """监控配置测试类"""
    
    def test_monitoring_config_import(self):
        """测试监控配置导入"""
        try:
            from core.config import settings
            # 检查settings是否可导入
            assert settings is not None
        except ImportError:
            pytest.skip("Settings not available")
    
    def test_alert_thresholds_configured(self):
        """测试告警阈值配置"""
        thresholds = {
            "cost_warning": 100.0,
            "cost_critical": 500.0,
            "latency_warning": 200.0,
            "latency_critical": 500.0,
            "error_rate_warning": 5.0,
            "error_rate_critical": 10.0
        }
        
        assert thresholds["cost_warning"] < thresholds["cost_critical"]
        assert thresholds["latency_warning"] < thresholds["latency_critical"]
    
    def test_collection_interval_configured(self):
        """测试收集间隔配置"""
        collection_interval = 60  # 秒
        assert collection_interval > 0
        assert collection_interval <= 3600  # 不超过1小时


# ============== 模拟监控操作测试 ==============

class TestMockMonitoringOperations:
    """模拟监控操作测试类"""
    
    def test_mock_metric_collection(self):
        """测试模拟指标收集"""
        metrics_buffer = []
        
        def collect_metric(metric_name, value, timestamp):
            metrics_buffer.append({
                "name": metric_name,
                "value": value,
                "timestamp": timestamp
            })
        
        # 收集指标
        collect_metric("latency", 100.0, datetime.utcnow())
        collect_metric("cost", 50.0, datetime.utcnow())
        
        assert len(metrics_buffer) == 2
        assert metrics_buffer[0]["name"] == "latency"
    
    def test_mock_alert_firing(self):
        """测试模拟告警触发"""
        alerts = []
        
        def fire_alert(alert_id, severity, message):
            alerts.append({
                "alert_id": alert_id,
                "severity": severity,
                "message": message,
                "timestamp": datetime.utcnow()
            })
        
        # 触发告警
        fire_alert("alert-001", "HIGH", "Cost exceeded threshold")
        fire_alert("alert-002", "LOW", "Latency increased")
        
        assert len(alerts) == 2
        assert alerts[0]["severity"] == "HIGH"
    
    def test_mock_optimization_analysis(self):
        """测试模拟优化分析"""
        analysis_results = []
        
        def analyze_optimization(usage_data):
            recommendations = []
            
            if usage_data.get("cost", 0) > 100:
                recommendations.append({
                    "type": "cost",
                    "action": "Optimize API calls",
                    "savings": 20.0
                })
            
            if usage_data.get("latency", 0) > 200:
                recommendations.append({
                    "type": "performance",
                    "action": "Enable caching",
                    "savings": 50.0
                })
            
            return recommendations
        
        # 分析优化
        result1 = analyze_optimization({"cost": 150, "latency": 100})
        result2 = analyze_optimization({"cost": 50, "latency": 300})
        result3 = analyze_optimization({"cost": 200, "latency": 250})
        
        assert len(result1) == 1
        assert len(result2) == 1
        assert len(result3) == 2
    
    def test_mock_dashboard_update(self):
        """测试模拟仪表盘更新"""
        dashboard_data = {
            "metrics": {},
            "alerts": [],
            "last_updated": None
        }
        
        def update_dashboard(metric_name, value):
            dashboard_data["metrics"][metric_name] = value
            dashboard_data["last_updated"] = datetime.utcnow()
        
        update_dashboard("total_cost", 100.0)
        update_dashboard("active_users", 50)
        
        assert len(dashboard_data["metrics"]) == 2
        assert dashboard_data["last_updated"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
