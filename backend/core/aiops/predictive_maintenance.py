"""
预测性维护模块

功能：
- 故障预测模型
- 资源趋势预测
- 容量规划
- 提前24小时预警
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import hashlib

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """预警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class PredictionType(Enum):
    """预测类型"""
    FAILURE = "failure"           # 故障预测
    TREND = "trend"               # 趋势预测
    CAPACITY = "capacity"         # 容量预测
    ANOMALY = "anomaly"           # 异常预测


@dataclass
class Prediction:
    """预测结果"""
    id: str
    prediction_type: PredictionType
    target: str
    value: float
    confidence: float  # 0-1
    timestamp: datetime
    prediction_time: datetime  # 预测时间点
    alert_level: AlertLevel
    description: str
    suggested_actions: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "prediction_type": self.prediction_type.value,
            "target": self.target,
            "value": self.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "prediction_time": self.prediction_time.isoformat(),
            "alert_level": self.alert_level.value,
            "description": self.description,
            "suggested_actions": self.suggested_actions,
            "metadata": self.metadata,
        }


@dataclass
class CapacityForecast:
    """容量预测"""
    resource: str
    current_value: float
    predicted_value: float
    unit: str
    threshold: float
    utilization_rate: float
    days_until_capacity: int
    recommended_capacity: float
    alert_level: AlertLevel

    def to_dict(self) -> Dict:
        return {
            "resource": self.resource,
            "current_value": self.current_value,
            "predicted_value": self.predicted_value,
            "unit": self.unit,
            "threshold": self.threshold,
            "utilization_rate": self.utilization_rate,
            "days_until_capacity": self.days_until_capacity,
            "recommended_capacity": self.recommended_capacity,
            "alert_level": self.alert_level.value,
        }


class TrendAnalyzer:
    """趋势分析器"""

    def __init__(self, window_size: int = 24):
        """
        初始化趋势分析器

        Args:
            window_size: 分析窗口大小
        """
        self.window_size = window_size
        self.history: Dict[str, deque] = {}

    def add_data_point(self, metric: str, value: float, timestamp: datetime):
        """添加数据点"""
        if metric not in self.history:
            self.history[metric] = deque(maxlen=self.window_size * 7)  # 存储一周数据

        self.history[metric].append((timestamp, value))

    def compute_trend(self, metric: str) -> Tuple[float, float, bool]:
        """
        计算趋势

        Returns:
            (slope, r_squared, is_trending_up)
        """
        if metric not in self.history or len(self.history[metric]) < 10:
            return (0, 0, False)

        # 提取时间序列
        data = list(self.history[metric])
        timestamps = [(t - data[0][0]).total_seconds() / 3600 for t, _ in data]  # 转换为小时
        values = [v for _, v in data]

        # 线性回归
        if len(values) < 2:
            return (0, 0, False)

        n = len(values)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_x2 = sum(x * x for x in timestamps)
        sum_y2 = sum(y * y for y in values)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return (0, 0, False)

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # 计算R平方
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in values)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(timestamps, values))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return (slope, r_squared, slope > 0)

    def predict_future(self, metric: str, hours_ahead: int = 24) -> Optional[float]:
        """
        预测未来值

        Args:
            metric: 指标名称
            hours_ahead: 预测多少小时后的值

        Returns:
            预测值
        """
        slope, r_squared, _ = self.compute_trend(metric)

        if r_squared < 0.5:  # 置信度不够
            return None

        if metric not in self.history or not self.history[metric]:
            return None

        # 获取最后一个值
        last_value = self.history[metric][-1][1]

        # 预测
        predicted_value = last_value + slope * hours_ahead

        return max(0, predicted_value)  # 确保非负


class FailurePredictor:
    """故障预测器"""

    def __init__(self):
        self.threshold_config = {
            "cpu": {"warning": 80, "critical": 90, "failure": 95},
            "memory": {"warning": 85, "critical": 90, "failure": 95},
            "disk": {"warning": 80, "critical": 90, "failure": 95},
            "latency": {"warning": 500, "critical": 1000, "failure": 2000},
            "error_rate": {"warning": 5, "critical": 10, "failure": 20},
        }

        # 预测权重
        self.weights = {
            "current_value": 0.3,
            "trend": 0.3,
            "rate_of_change": 0.2,
            "anomaly_score": 0.2,
        }

    def predict_failure_probability(self, metric: str, current_value: float,
                                     trend_slope: float, anomaly_score: float) -> float:
        """
        计算故障概率

        Args:
            metric: 指标名称
            current_value: 当前值
            trend_slope: 趋势斜率
            anomaly_score: 异常分数 (0-1)

        Returns:
            故障概率 (0-1)
        """
        thresholds = self.threshold_config.get(metric, {})
        if not thresholds:
            return 0.0

        # 标准化当前值
        normalized_value = min(current_value / thresholds["critical"], 1.0)

        # 标准化趋势
        normalized_trend = min(abs(trend_slope) / 10, 1.0)  # 假设每小时变化10%为最大

        # 综合评分
        probability = (
            self.weights["current_value"] * normalized_value +
            self.weights["trend"] * normalized_trend +
            self.weights["rate_of_change"] * normalized_trend +
            self.weights["anomaly_score"] * anomaly_score
        )

        return min(probability, 1.0)

    def predict_time_to_failure(self, metric: str, current_value: float,
                                 trend_slope: float) -> Optional[float]:
        """
        预测距离故障的时间

        Args:
            metric: 指标名称
            current_value: 当前值
            trend_slope: 趋势斜率 (每小时变化量)

        Returns:
            距离故障的小时数
        """
        thresholds = self.threshold_config.get(metric, {})
        if not thresholds or "failure" not in thresholds:
            return None

        failure_threshold = thresholds["failure"]
        warning_threshold = thresholds["warning"]

        if trend_slope <= 0:
            return None  # 没有上升趋势

        # 计算到故障阈值的时间
        time_to_failure = (failure_threshold - current_value) / trend_slope

        # 计算到预警阈值的时间
        time_to_warning = (warning_threshold - current_value) / trend_slope

        return max(0, time_to_failure)


class CapacityPlanner:
    """容量规划器"""

    def __init__(self):
        self.growth_rate_config = {
            "cpu": {"growth_factor": 1.1, "planning_period": 30},  # 每月增长10%, 规划30天
            "memory": {"growth_factor": 1.15, "planning_period": 30},
            "disk": {"growth_factor": 1.2, "planning_period": 90},
        }

    def calculate_required_capacity(self, resource: str, current_usage: float,
                                     daily_growth_rate: float,
                                     days_ahead: int = 30) -> Tuple[float, int]:
        """
        计算所需容量

        Args:
            resource: 资源类型
            current_usage: 当前使用量
            daily_growth_rate: 日增长率
            days_ahead: 提前多少天规划

        Returns:
            (所需容量, 到达容量的天数)
        """
        # 计算预测值
        growth_factor = (1 + daily_growth_rate) ** days_ahead
        predicted_usage = current_usage * growth_factor

        # 添加安全边际 (20%)
        required_capacity = predicted_usage * 1.2

        # 计算到达80%容量的时间
        target_threshold = required_capacity * 0.8
        if daily_growth_rate > 0:
            days_until_capacity = int(np.log(target_threshold / current_usage) / np.log(1 + daily_growth_rate))
        else:
            days_until_capacity = 999

        return (required_capacity, days_until_capacity)

    def recommend_scaling(self, resource: str, current_replicas: int,
                           current_usage: float, threshold: float) -> Dict:
        """
        推荐扩缩容策略

        Returns:
            扩缩容建议
        """
        utilization = current_usage / threshold * 100

        if utilization > 90:
            action = "scale_up"
            recommended_replicas = int(current_replicas * 1.5)
            reason = "当前使用率过高,需要立即扩容"
        elif utilization > 80:
            action = "scale_up"
            recommended_replicas = int(current_replicas * 1.25)
            reason = "使用率偏高,建议扩容"
        elif utilization < 30:
            action = "scale_down"
            recommended_replicas = int(current_replicas * 0.7)
            reason = "使用率过低,建议缩容"
        else:
            action = "keep"
            recommended_replicas = current_replicas
            reason = "当前容量合适,维持现状"

        return {
            "action": action,
            "current_replicas": current_replicas,
            "recommended_replicas": recommended_replicas,
            "utilization_percent": utilization,
            "reason": reason,
        }


class PredictiveMaintenance:
    """
    预测性维护系统

    功能：
    - 故障预测
    - 资源趋势预测
    - 容量规划
    - 提前24小时预警
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化预测性维护系统

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.trend_analyzer = TrendAnalyzer(window_size=self.config.get("window_size", 24))
        self.failure_predictor = FailurePredictor()
        self.capacity_planner = CapacityPlanner()

        # 预测历史
        self.predictions: List[Prediction] = []
        self.capacity_forecasts: List[CapacityForecast] = []

        # 配置
        self.early_warning_hours = self.config.get("early_warning_hours", 24)
        self.prediction_interval = self.config.get("prediction_interval", 3600)  # 每小时预测一次

        # 预测计数器
        self._prediction_counter = 0

    def _generate_prediction_id(self) -> str:
        """生成预测ID"""
        self._prediction_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"pred_{timestamp}_{self._prediction_counter:04d}"

    def add_metric_data(self, metric: str, value: float, timestamp: Optional[datetime] = None):
        """添加指标数据"""
        timestamp = timestamp or datetime.now()
        self.trend_analyzer.add_data_point(metric, value, timestamp)

    def predict(self, metrics: Dict[str, float], anomaly_scores: Optional[Dict[str, float]] = None,
                hours_ahead: int = 24) -> List[Prediction]:
        """
        执行预测

        Args:
            metrics: 当前指标
            anomaly_scores: 异常分数 (可选)
            hours_ahead: 预测多少小时后的状态

        Returns:
            预测列表
        """
        predictions = []
        anomaly_scores = anomaly_scores or {}

        for metric, current_value in metrics.items():
            # 获取趋势
            slope, r_squared, is_upward = self.trend_analyzer.compute_trend(metric)

            # 预测未来值
            predicted_value = self.trend_analyzer.predict_future(metric, hours_ahead)
            if predicted_value is None:
                predicted_value = current_value

            # 计算故障概率
            anomaly_score = anomaly_scores.get(metric, 0)
            failure_probability = self.failure_predictor.predict_failure_probability(
                metric, current_value, slope, anomaly_score
            )

            # 预测时间到故障
            time_to_failure = self.failure_predictor.predict_time_to_failure(
                metric, current_value, slope
            )

            # 确定预警级别
            alert_level, description = self._determine_alert(
                metric, current_value, predicted_value, failure_probability,
                time_to_failure, r_squared
            )

            if alert_level != AlertLevel.INFO:
                prediction = Prediction(
                    id=self._generate_prediction_id(),
                    prediction_type=PredictionType.FAILURE,
                    target=metric,
                    value=predicted_value,
                    confidence=r_squared,
                    timestamp=datetime.now(),
                    prediction_time=datetime.now() + timedelta(hours=hours_ahead),
                    alert_level=alert_level,
                    description=description,
                    suggested_actions=self._get_suggested_actions(
                        metric, current_value, predicted_value, alert_level
                    ),
                    metadata={
                        "current_value": current_value,
                        "slope": slope,
                        "failure_probability": failure_probability,
                        "time_to_failure_hours": time_to_failure,
                    },
                )

                predictions.append(prediction)
                self.predictions.append(prediction)

        logger.info(f"生成 {len(predictions)} 个预测")

        return predictions

    def _determine_alert(self, metric: str, current_value: float,
                         predicted_value: float, failure_probability: float,
                         time_to_failure: Optional[float], confidence: float) -> Tuple[AlertLevel, str]:
        """确定预警级别"""
        # 高置信度预测
        if confidence < 0.5:
            return (AlertLevel.INFO, f"预测置信度不足 ({confidence:.2%})")

        # 根据故障概率确定级别
        if failure_probability > 0.8:
            level = AlertLevel.CRITICAL
            level_desc = "严重"
        elif failure_probability > 0.5:
            level = AlertLevel.WARNING
            level_desc = "警告"
        elif failure_probability > 0.3:
            level = AlertLevel.WARNING
            level_desc = "注意"
        else:
            return (AlertLevel.INFO, "系统运行正常")

        # 检查是否会在预警时间内达到故障阈值
        if time_to_failure and time_to_failure <= self.early_warning_hours:
            level = AlertLevel.CRITICAL
            level_desc = "紧急"

        description = f"[{level_desc}] {metric} 预测故障概率: {failure_probability:.1%}"

        if time_to_failure:
            description += f", 预计 {time_to_failure:.1f} 小时后达到阈值"

        return (level, description)

    def _get_suggested_actions(self, metric: str, current_value: float,
                                predicted_value: float, alert_level: AlertLevel) -> List[str]:
        """获取建议的操作"""
        actions = {
            "cpu": [
                "检查CPU使用率异常增长的原因",
                "考虑增加实例数量",
                "优化代码性能",
                "启用自动扩缩容",
            ],
            "memory": [
                "检查内存泄漏",
                "增加容器内存限制",
                "优化内存使用",
                "考虑扩容",
            ],
            "disk": [
                "清理临时文件和日志",
                "增加磁盘空间",
                "实施数据归档策略",
            ],
            "latency": [
                "检查数据库查询性能",
                "增加缓存层",
                "优化API响应",
                "扩容后端服务",
            ],
            "error_rate": [
                "检查错误日志",
                "回滚最近的变更",
                "检查依赖服务",
            ],
        }

        return actions.get(metric, ["监控系统状态", "准备应急预案"])

    def forecast_capacity(self, resources: Dict[str, Dict],
                          days_ahead: int = 30) -> List[CapacityForecast]:
        """
        预测容量需求

        Args:
            resources: 资源字典 {resource_name: {current: x, threshold: y, unit: z, daily_growth: d}}
            days_ahead: 预测天数

        Returns:
            容量预测列表
        """
        forecasts = []

        for resource, config in resources.items():
            current = config.get("current", 0)
            threshold = config.get("threshold", 100)
            unit = config.get("unit", "%")
            daily_growth = config.get("daily_growth", 0.01)

            # 计算所需容量
            required_capacity, days_until = self.capacity_planner.calculate_required_capacity(
                resource, current, daily_growth, days_ahead
            )

            # 计算利用率
            utilization = current / threshold * 100

            # 确定预警级别
            if days_until < 7:
                alert = AlertLevel.CRITICAL
            elif days_until < 14:
                alert = AlertLevel.WARNING
            elif days_until < 30:
                alert = AlertLevel.WARNING
            else:
                alert = AlertLevel.INFO

            forecast = CapacityForecast(
                resource=resource,
                current_value=current,
                predicted_value=required_capacity,
                unit=unit,
                threshold=threshold,
                utilization_rate=utilization,
                days_until_capacity=days_until,
                recommended_capacity=required_capacity,
                alert_level=alert,
            )

            forecasts.append(forecast)
            self.capacity_forecasts.append(forecast)

        return forecasts

    def get_capacity_recommendation(self, resource: str, current_replicas: int,
                                     current_usage: float, threshold: float) -> Dict:
        """获取容量推荐"""
        return self.capacity_planner.recommend_scaling(
            resource, current_replicas, current_usage, threshold
        )

    def predict_resource_trend(self, metric: str, hours_ahead: int = 24) -> Dict:
        """
        预测资源趋势

        Args:
            metric: 指标名称
            hours_ahead: 预测小时数

        Returns:
            趋势预测字典
        """
        slope, r_squared, is_upward = self.trend_analyzer.compute_trend(metric)
        predicted_value = self.trend_analyzer.predict_future(metric, hours_ahead)

        return {
            "metric": metric,
            "current_trend": "upward" if is_upward else "downward" if slope < 0 else "stable",
            "slope_per_hour": slope,
            "confidence": r_squared,
            "predicted_value_in_24h": predicted_value,
            "hours_ahead": hours_ahead,
        }

    def check_alerts(self) -> List[Prediction]:
        """检查所有预警"""
        # 只返回最近24小时内的未处理预警
        cutoff = datetime.now() - timedelta(hours=24)
        recent_predictions = [
            p for p in self.predictions
            if p.timestamp > cutoff and p.alert_level != AlertLevel.INFO
        ]

        return recent_predictions

    def get_system_health_prediction(self) -> Dict:
        """获取系统健康度预测"""
        # 基于所有预测计算整体健康度
        if not self.predictions:
            return {
                "score": 100,
                "status": "healthy",
                "active_alerts": 0,
                "predictions_24h": 0,
            }

        # 计算活跃预警数量
        active_alerts = self.check_alerts()

        # 计算平均故障概率
        avg_failure_prob = np.mean([
            p.metadata.get("failure_probability", 0)
            for p in self.predictions[-100:]
        ])

        # 计算健康度评分
        score = max(0, 100 - avg_failure_prob_prob * 100)

        return {
            "score": score,
            "status": "critical" if score < 40 else "warning" if score < 70 else "healthy",
            "active_alerts": len(active_alerts),
            "predictions_24h": len(active_alerts),
        }

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "total_predictions": len(self.predictions),
            "critical_alerts": len([p for p in self.predictions if p.alert_level == AlertLevel.CRITICAL]),
            "warning_alerts": len([p for p in self.predictions if p.alert_level == AlertLevel.WARNING]),
            "capacity_forecasts": len(self.capacity_forecasts),
        }
