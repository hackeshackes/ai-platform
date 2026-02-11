"""
AIOps API 路由

提供REST API接口:
- POST /api/v1/aiops/anomaly/detect - 异常检测
- POST /api/v1/aiops/root-cause/analyze - 根因分析
- POST /api/v1/aiops/auto-recovery/execute - 自动恢复
- GET /api/v1/aiops/predictive/maintenance - 预测性维护
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify
from functools import wraps

from .anomaly_detector import AnomalyDetector
from .root_cause_analyzer import RootCauseAnalyzer
from .auto_recovery import AutoRecovery
from .predictive_maintenance import PredictiveMaintenance

logger = logging.getLogger(__name__)


def create_app(config: Optional[Dict] = None) -> Flask:
    """
    创建AIOps Flask应用

    Args:
        config: 配置字典

    Returns:
        Flask应用实例
    """
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # 初始化核心组件
    anomaly_detector = AnomalyDetector(config)
    root_cause_analyzer = RootCauseAnalyzer(config)
    auto_recovery = AutoRecovery(config)
    predictive_maintenance = PredictiveMaintenance(config)

    # 请求日志装饰器
    def log_request(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            logger.info(f"API请求: {request.method} {request.path}")
            return f(*args, **kwargs)
        return decorated_function

    # ==================== 健康检查 ====================

    @app.route("/health", methods=["GET"])
    @log_request
    def health_check():
        """健康检查"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "aiops-core",
        })

    # ==================== 异常检测 API ====================

    @app.route("/api/v1/aiops/anomaly/detect", methods=["POST"])
    @log_request
    def detect_anomaly():
        """
        异常检测

        请求体:
        {
            "metrics": {"cpu": 85, "memory": 90},
            "threshold": 0.8
        }

        返回:
        {
            "status": "warning",
            "anomaly_count": 2,
            "anomalies": [...],
            "timestamp": "..."
        }
        """
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "请求体不能为空"}), 400

            metrics = data.get("metrics", {})
            threshold = data.get("threshold", 0.8)

            if not metrics:
                return jsonify({"error": "metrics不能为空"}), 400

            # 执行检测
            result = anomaly_detector.detect_realtime(metrics)

            return jsonify(result)

        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/aiops/anomaly/history", methods=["GET"])
    @log_request
    def get_anomaly_history():
        """获取异常历史"""
        try:
            limit = request.args.get("limit", 50, type=int)
            health_score = anomaly_detector.get_health_score()
            metrics_summary = anomaly_detector.get_metrics_summary()

            return jsonify({
                "health_score": health_score,
                "metrics_summary": metrics_summary,
                "recent_anomalies": [
                    a.to_dict()
                    for a in list(anomaly_detector.anomaly_history)[-limit:]
                ],
            })

        except Exception as e:
            logger.error(f"获取异常历史失败: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/aiops/anomaly/train", methods=["POST"])
    @log_request
    def train_anomaly_detector():
        """训练异常检测模型"""
        try:
            data = request.get_json()
            historical_data = data.get("historical_data", {})

            if not historical_data:
                return jsonify({"error": "historical_data不能为空"}), 400

            anomaly_detector.train(historical_data)

            return jsonify({
                "status": "success",
                "message": "模型训练完成",
            })

        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            return jsonify({"error": str(e)}), 500

    # ==================== 根因分析 API ====================

    @app.route("/api/v1/aiops/root-cause/analyze", methods=["POST"])
    @log_request
    def analyze_root_cause():
        """
        根因分析

        请求体:
        {
            "symptom": "high_latency",
            "time_range": "1h",
            "affected_services": ["api-gateway", "user-service"]
        }

        返回:
        {
            "id": "rc_20231201_120000_0001",
            "node_id": "mysql-master",
            "confidence": 0.85,
            ...
        }
        """
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "请求体不能为空"}), 400

            symptom = data.get("symptom", "unknown")
            time_range = data.get("time_range", "1h")
            affected_services = data.get("affected_services")

            # 执行根因分析
            result = root_cause_analyzer.analyze(
                symptom=symptom,
                time_range=time_range,
                affected_services=affected_services,
            )

            return jsonify({
                "analysis_time_ms": result.analysis_time_ms,
                **result.to_dict(),
            })

        except Exception as e:
            logger.error(f"根因分析失败: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/aiops/root-cause/topology", methods=["GET"])
    @log_request
    def get_dependency_topology():
        """获取依赖拓扑"""
        try:
            topology = root_cause_analyzer.get_dependency_topology()
            return jsonify(topology)

        except Exception as e:
            logger.error(f"获取依赖拓扑失败: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/aiops/root-cause/affected", methods=["POST"])
    @log_request
    def get_affected_services():
        """获取受影响的服务"""
        try:
            data = request.get_json()
            node_id = data.get("node_id")

            if not node_id:
                return jsonify({"error": "node_id不能为空"}), 400

            affected = root_cause_analyzer.graph.get_all_affected(node_id)

            return jsonify({
                "node_id": node_id,
                "upstream": list(affected["upstream"]),
                "downstream": list(affected["downstream"]),
                "total_affected": list(affected["total_affected"]),
            })

        except Exception as e:
            logger.error(f"获取受影响服务失败: {e}")
            return jsonify({"error": str(e)}), 500

    # ==================== 自动恢复 API ====================

    @app.route("/api/v1/aiops/auto-recovery/execute", methods=["POST"])
    @log_request
    def execute_recovery():
        """
        执行自动恢复

        请求体:
        {
            "incident_id": "inc_123",
            "strategy": "auto_fix"
        }

        或者创建新事件并执行恢复:
        {
            "title": "CPU使用率过高",
            "description": "CPU使用率达到95%",
            "severity": "high",
            "service": "api-gateway",
            "metrics": {"cpu": 95, "memory": 80},
            "strategy": "auto_fix"
        }
        """
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "请求体不能为空"}), 400

            # 检查是创建新事件还是使用现有事件
            if "incident_id" in data:
                incident_id = data["incident_id"]
                strategy = data.get("strategy", "auto_fix")

                result = auto_recovery.execute_recovery(incident_id, strategy)
            else:
                # 创建新事件
                incident = auto_recovery.create_incident(
                    title=data.get("title", "自动创建的故障事件"),
                    description=data.get("description", ""),
                    severity=data.get("severity", "medium"),
                    service=data.get("service", "unknown"),
                    metrics=data.get("metrics", {}),
                    context=data.get("context", {}),
                )

                strategy = data.get("strategy", "auto_fix")
                result = auto_recovery.execute_recovery(incident.id, strategy)

            return jsonify({
                "incident_id": result.incident_id,
                "status": result.status.value,
                "total_time_ms": result.total_time_ms,
                "success_rate": result.success_rate,
                "actions": [a.to_dict() for a in result.actions],
            })

        except Exception as e:
            logger.error(f"自动恢复执行失败: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/aiops/auto-recovery/incident", methods=["POST"])
    @log_request
    def create_incident():
        """创建故障事件"""
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "请求体不能为空"}), 400

            incident = auto_recovery.create_incident(
                title=data.get("title"),
                description=data.get("description"),
                severity=data.get("severity"),
                service=data.get("service"),
                metrics=data.get("metrics", {}),
                context=data.get("context", {}),
            )

            return jsonify({
                "incident_id": incident.id,
                "status": incident.status.value,
                "created_at": incident.created_at.isoformat(),
            })

        except Exception as e:
            logger.error(f"创建事件失败: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/aiops/auto-recovery/incident/<incident_id>", methods=["GET"])
    @log_request
    def get_incident(incident_id: str):
        """获取事件状态"""
        try:
            status = auto_recovery.get_incident_status(incident_id)

            if not status:
                return jsonify({"error": "事件不存在"}), 404

            return jsonify(status)

        except Exception as e:
            logger.error(f"获取事件状态失败: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/aiops/auto-recovery/rollback", methods=["POST"])
    @log_request
    def execute_gray_rollback():
        """执行灰度回滚"""
        try:
            data = request.get_json()

            incident_id = data.get("incident_id")
            percentage = data.get("percentage", 10)

            if not incident_id:
                return jsonify({"error": "incident_id不能为空"}), 400

            result = auto_recovery.execute_gray_rollback(incident_id, percentage)

            return jsonify(result)

        except Exception as e:
            logger.error(f"灰度回滚失败: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/aiops/auto-recovery/statistics", methods=["GET"])
    @log_request
    def get_recovery_statistics():
        """获取恢复统计"""
        try:
            stats = auto_recovery.get_statistics()
            return jsonify(stats)

        except Exception as e:
            logger.error(f"获取统计失败: {e}")
            return jsonify({"error": str(e)}), 500

    # ==================== 预测性维护 API ====================

    @app.route("/api/v1/aiops/predictive/analyze", methods=["POST"])
    @log_request
    def predict_analyze():
        """
        预测分析

        请求体:
        {
            "metrics": {"cpu": 80, "memory": 75},
            "anomaly_scores": {"cpu": 0.3, "memory": 0.2},
            "hours_ahead": 24
        }

        返回:
        {
            "predictions": [...],
            "timestamp": "..."
        }
        """
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "请求体不能为空"}), 400

            metrics = data.get("metrics", {})
            anomaly_scores = data.get("anomaly_scores", {})
            hours_ahead = data.get("hours_ahead", 24)

            if not metrics:
                return jsonify({"error": "metrics不能为空"}), 400

            predictions = predictive_maintenance.predict(
                metrics=metrics,
                anomaly_scores=anomaly_scores,
                hours_ahead=hours_ahead,
            )

            return jsonify({
                "predictions": [p.to_dict() for p in predictions],
                "timestamp": datetime.now().isoformat(),
            })

        except Exception as e:
            logger.error(f"预测分析失败: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/aiops/predictive/capacity", methods=["POST"])
    @log_request
    def forecast_capacity():
        """
        容量预测

        请求体:
        {
            "resources": {
                "cpu": {
                    "current": 60,
                    "threshold": 100,
                    "unit": "%",
                    "daily_growth": 0.02
                },
                "memory": {
                    "current": 70,
                    "threshold": 100,
                    "unit": "%",
                    "daily_growth": 0.01
                }
            },
            "days_ahead": 30
        }
        """
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "请求体不能为空"}), 400

            resources = data.get("resources", {})
            days_ahead = data.get("days_ahead", 30)

            if not resources:
                return jsonify({"error": "resources不能为空"}), 400

            forecasts = predictive_maintenance.forecast_capacity(
                resources=resources,
                days_ahead=days_ahead,
            )

            return jsonify({
                "forecasts": [f.to_dict() for f in forecasts],
                "timestamp": datetime.now().isoformat(),
            })

        except Exception as e:
            logger.error(f"容量预测失败: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/aiops/predictive/trend/<metric>", methods=["GET"])
    @log_request
    def get_resource_trend(metric: str):
        """获取资源趋势"""
        try:
            hours_ahead = request.args.get("hours", 24, type=int)

            trend = predictive_maintenance.predict_resource_trend(
                metric=metric,
                hours_ahead=hours_ahead,
            )

            return jsonify(trend)

        except Exception as e:
            logger.error(f"获取资源趋势失败: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/aiops/predictive/alerts", methods=["GET"])
    @log_request
    def get_predictive_alerts():
        """获取预测性预警"""
        try:
            alerts = predictive_maintenance.check_alerts()

            return jsonify({
                "alerts": [a.to_dict() for a in alerts],
                "timestamp": datetime.now().isoformat(),
            })

        except Exception as e:
            logger.error(f"获取预警失败: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/v1/aiops/predictive/health", methods=["GET"])
    @log_request
    def get_health_prediction():
        """获取系统健康度预测"""
        try:
            health = predictive_maintenance.get_system_health_prediction()
            return jsonify(health)

        except Exception as e:
            logger.error(f"获取健康度预测失败: {e}")
            return jsonify({"error": str(e)}), 500

    # ==================== 综合仪表板 API ====================

    @app.route("/api/v1/aiops/dashboard", methods=["GET"])
    @log_request
    def get_dashboard():
        """获取综合仪表板数据"""
        try:
            return jsonify({
                "timestamp": datetime.now().isoformat(),
                "anomaly": {
                    "health_score": anomaly_detector.get_health_score(),
                },
                "recovery": {
                    "statistics": auto_recovery.get_statistics(),
                },
                "predictive": {
                    "alerts": predictive_maintenance.check_alerts(),
                    "statistics": predictive_maintenance.get_statistics(),
                },
            })

        except Exception as e:
            logger.error(f"获取仪表板数据失败: {e}")
            return jsonify({"error": str(e)}), 500

    return app


# 便捷启动函数
def run_server(host: str = "0.0.0.0", port: int = 8080, debug: bool = False):
    """启动AIOps服务器"""
    app = create_app()
    app.run(host=host, port=port, debug=debug)
