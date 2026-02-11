"""
异常检测模块

功能：
- 实时指标监控
- 统计异常检测
- 机器学习异常检测
- 检测准确率目标: >99%
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


class Severity(Enum):
    """异常严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """异常类型"""
    SPIKE = "spike"           # 突增
    DROP = "drop"             # 突降
    GRADUAL = "gradual"       # 渐变
    SEASONAL = "seasonal"     # 季节性异常
    TREND = "trend"           # 趋势异常
    VARIANCE = "variance"     # 方差异常


@dataclass
class Anomaly:
    """异常事件"""
    id: str
    metric: str
    value: float
    expected_value: float
    severity: Severity
    anomaly_type: AnomalyType
    score: float  # 0-1, 越高越异常
    timestamp: datetime
    description: str
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "metric": self.metric,
            "value": self.value,
            "expected_value": self.expected_value,
            "severity": self.severity.value,
            "anomaly_type": self.anomaly_type.value,
            "score": self.score,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class MetricData:
    """指标数据"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict = field(default_factory=dict)


class StatisticalDetector:
    """统计异常检测器"""

    def __init__(self, window_size: int = 60, threshold_std: float = 3.0):
        """
        初始化统计检测器

        Args:
            window_size: 滑动窗口大小
            threshold_std: 标准差阈值
        """
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.histories: Dict[str, deque] = {}

    def update_history(self, metric: str, value: float):
        """更新历史数据"""
        if metric not in self.histories:
            self.histories[metric] = deque(maxlen=self.window_size)
        self.histories[metric].append(value)

    def detect_zscore(self, metric: str, value: float) -> Optional[Tuple[bool, float, float, float]]:
        """
        Z-Score异常检测

        Returns:
            (is_anomaly, score, mean, std)
        """
        if metric not in self.histories or len(self.histories[metric]) < 10:
            return None

        history = list(self.histories[metric])
        mean = np.mean(history)
        std = np.std(history)

        if std == 0:
            return None

        zscore = abs(value - mean) / std
        is_anomaly = zscore > self.threshold_std

        return (is_anomaly, zscore, mean, std)

    def detect_iqr(self, metric: str, value: float, multiplier: float = 1.5) -> Optional[Tuple[bool, float]]:
        """
        IQR(四分位距)异常检测

        Returns:
            (is_anomaly, score)
        """
        if metric not in self.histories or len(self.histories[metric]) < 10:
            return None

        history = list(self.histories[metric])
        q1 = np.percentile(history, 25)
        q3 = np.percentile(history, 75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        is_anomaly = value < lower_bound or value > upper_bound

        # 计算异常分数
        if is_anomaly:
            if value < lower_bound:
                score = (lower_bound - value) / iqr
            else:
                score = (value - upper_bound) / iqr
            score = min(score, 10) / 10  # 归一化到0-1
        else:
            score = 0

        return (is_anomaly, score)


class MLDetector:
    """机器学习异常检测器"""

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化ML检测器

        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.is_trained = False
        self.threshold = 0.8

        # 简化的 Isolation Forest 模拟
        self.trees = []
        self.tree_depth = 10
        self.num_trees = 100

        # 训练数据统计
        self.mean = {}
        self.std = {}

    def _build_tree(self, data: np.ndarray, depth: int = 0) -> Dict:
        """构建隔离树"""
        if depth >= self.tree_depth or len(data) <= 1:
            return {"leaf": True, "size": len(data)}

        feature_idx = np.random.randint(0, data.shape[1])
        min_val = data[:, feature_idx].min()
        max_val = data[:, feature_idx].max()

        if min_val == max_val:
            return {"leaf": True, "size": len(data)}

        split_val = np.random.uniform(min_val, max_val)
        left_mask = data[:, feature_idx] < split_val
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": feature_idx,
            "split": split_val,
            "left": self._build_tree(data[left_mask], depth + 1),
            "right": self._build_tree(data[right_mask], depth + 1),
        }

    def _path_length(self, tree: Dict, sample: np.ndarray, depth: int = 0) -> float:
        """计算样本路径长度"""
        if tree["leaf"]:
            return depth + self._c(tree["size"])

        feature_idx = tree["feature"]
        if sample[feature_idx] < tree["split"]:
            return self._path_length(tree["left"], sample, depth + 1)
        else:
            return self._path_length(tree["right"], sample, depth + 1)

    def _c(self, n: int) -> float:
        """平均路径长度近似"""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)

    def fit(self, data: Dict[str, List[float]]):
        """训练模型"""
        # 标准化数据
        self.mean = {k: np.mean(v) for k, v in data.items()}
        self.std = {k: np.std(v) + 1e-6 for k, v in data.items()}

        # 构建特征矩阵
        max_len = max(len(v) for v in data.values())
        features = []
        for i in range(max_len):
            row = []
            for k in data.keys():
                if i < len(data[k]):
                    row.append((data[k][i] - self.mean[k]) / self.std[k])
                else:
                    row.append(0)
            features.append(row)

        X = np.array(features)

        # 构建隔离森林
        np.random.seed(42)
        for _ in range(self.num_trees):
            sample_idx = np.random.choice(len(X), size=min(len(X), 256), replace=False)
            tree = self._build_tree(X[sample_idx])
            self.trees.append(tree)

        self.is_trained = True
        logger.info("ML异常检测器训练完成")

    def predict(self, metrics: Dict[str, float]) -> Tuple[bool, float]:
        """
        预测异常

        Args:
            metrics: 指标字典

        Returns:
            (is_anomaly, score)
        """
        if not self.is_trained or not self.trees:
            # 使用统计方法作为后备
            return (False, 0.0)

        # 标准化输入
        features = []
        for k in self.mean.keys():
            if k in metrics:
                val = (metrics[k] - self.mean[k]) / self.std[k]
            else:
                val = 0
            features.append(val)

        sample = np.array(features)

        # 计算平均路径长度
        path_lengths = []
        for tree in self.trees:
            path_lengths.append(self._path_length(tree, sample))

        avg_path_length = np.mean(path_lengths)
        c = self._c(256)  # 假设样本大小为256

        # 计算异常分数
        score = 2 ** (-avg_path_length / c)
        score = min(max(score, 0), 1)

        return (score > self.threshold, score)


class AnomalyDetector:
    """
    综合异常检测器

    整合统计检测和机器学习检测
    目标检测准确率: >99%
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化异常检测器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.statistical_detector = StatisticalDetector(
            window_size=self.config.get("window_size", 60),
            threshold_std=self.config.get("threshold_std", 3.0),
        )
        self.ml_detector = MLDetector()

        # 阈值配置
        self.zscore_threshold = self.config.get("zscore_threshold", 3.0)
        self.iqr_multiplier = self.config.get("iqr_multiplier", 1.5)
        self.ml_threshold = self.config.get("ml_threshold", 0.8)

        # 严重程度阈值
        self.severity_thresholds = self.config.get("severity_thresholds", {
            "critical": 0.95,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.3,
        })

        # 指标阈值配置
        self.metric_thresholds = self.config.get("metric_thresholds", {
            "cpu": {"critical": 95, "high": 85, "medium": 70, "low": 50},
            "memory": {"critical": 95, "high": 85, "medium": 70, "low": 60},
            "disk": {"critical": 95, "high": 85, "medium": 75, "low": 60},
            "latency": {"critical": 1000, "high": 500, "medium": 200, "low": 100},
            "error_rate": {"critical": 10, "high": 5, "medium": 2, "low": 1},
            "requests_per_second": {"critical": 10000, "high": 5000, "medium": 2000, "low": 500},
        })

        # 异常历史
        self.anomaly_history: deque = deque(maxlen=1000)
        self._anomaly_counter = 0

    def _generate_anomaly_id(self, metric: str, timestamp: datetime) -> str:
        """生成异常ID"""
        self._anomaly_counter += 1
        unique_str = f"{metric}_{timestamp.isoformat()}_{self._anomaly_counter}"
        return f"anomaly_{hashlib.md5(unique_str.encode()).hexdigest()[:12]}"

    def _get_severity(self, metric: str, value: float, score: float) -> Severity:
        """确定严重程度"""
        # 首先根据分数判断
        if score >= self.severity_thresholds.get("critical", 0.95):
            return Severity.CRITICAL
        elif score >= self.severity_thresholds.get("high", 0.8):
            return Severity.HIGH
        elif score >= self.severity_thresholds.get("medium", 0.6):
            return Severity.MEDIUM
        elif score >= self.severity_thresholds.get("low", 0.3):
            return Severity.LOW
        else:
            return Severity.LOW

    def _get_anomaly_type(self, metric: str, value: float, expected: float) -> AnomalyType:
        """确定异常类型"""
        if value > expected * 1.5:
            return AnomalyType.SPIKE
        elif value < expected * 0.5:
            return AnomalyType.DROP
        elif value > expected:
            return AnomalyType.GRADUAL
        else:
            return AnomalyType.DROP

    def _describe_anomaly(self, metric: str, value: float, expected: float,
                          severity: Severity, anomaly_type: AnomalyType) -> str:
        """生成异常描述"""
        pct_change = ((value - expected) / expected * 100) if expected != 0 else 0

        type_desc = {
            AnomalyType.SPIKE: "突增",
            AnomalyType.DROP: "突降",
            AnomalyType.GRADUAL: "渐变",
            AnomalyType.SEASONAL: "季节性",
            AnomalyType.TREND: "趋势",
            AnomalyType.VARIANCE: "方差",
        }.get(anomaly_type, "异常")

        severity_desc = {
            Severity.CRITICAL: "严重",
            Severity.HIGH: "高",
            Severity.MEDIUM: "中",
            Severity.LOW: "低",
        }.get(severity, "未知")

        return f"[{severity_desc}]{type_desc}: {metric} 当前值 {value:.2f}, 预期值 {expected:.2f}, 变化 {pct_change:+.1f}%"

    def detect(self, metrics: Dict[str, float], timestamp: Optional[datetime] = None,
               threshold: float = 0.8) -> List[Anomaly]:
        """
        检测异常

        Args:
            metrics: 指标字典, 例如 {"cpu": 85, "memory": 90}
            timestamp: 时间戳, 默认为当前时间
            threshold: 检测阈值, 默认为0.8

        Returns:
            异常列表
        """
        timestamp = timestamp or datetime.now()
        anomalies = []

        for metric, value in metrics.items():
            # 更新统计检测器的历史
            self.statistical_detector.update_history(metric, value)

            # Z-Score检测
            zscore_result = self.statistical_detector.detect_zscore(metric, value)
            zscore_anomaly = False
            zscore_score = 0

            if zscore_result:
                zscore_anomaly, zscore_raw, mean, std = zscore_result
                if zscore_anomaly:
                    zscore_score = min(zscore_raw / self.zscore_threshold, 1.0)

            # IQR检测
            iqr_result = self.statistical_detector.detect_iqr(metric, value, self.iqr_multiplier)
            iqr_anomaly = False
            iqr_score = 0

            if iqr_result:
                iqr_anomaly, iqr_score = iqr_result

            # ML检测
            ml_anomaly, ml_score = self.ml_detector.predict({metric: value})

            # 综合评分 (加权平均)
            weights = {"statistical": 0.4, "ml": 0.6}
            combined_score = (
                weights["statistical"] * max(zscore_score, iqr_score) +
                weights["ml"] * ml_score
            )

            # 最终判断
            is_anomaly = combined_score >= threshold

            if is_anomaly:
                severity = self._get_severity(metric, value, combined_score)
                expected = zscore_result[2] if zscore_result else value
                anomaly_type = self._get_anomaly_type(metric, value, expected)
                description = self._describe_anomaly(metric, value, expected, severity, anomaly_type)

                anomaly = Anomaly(
                    id=self._generate_anomaly_id(metric, timestamp),
                    metric=metric,
                    value=value,
                    expected_value=expected,
                    severity=severity,
                    anomaly_type=anomaly_type,
                    score=combined_score,
                    timestamp=timestamp,
                    description=description,
                    metadata={
                        "zscore": zscore_raw if zscore_result else None,
                        "iqr_score": iqr_score,
                        "ml_score": ml_score,
                        "weights": weights,
                    },
                )

                anomalies.append(anomaly)
                self.anomaly_history.append(anomaly)

                logger.info(f"检测到异常: {description}")

        return anomalies

    def detect_realtime(self, metrics: Dict[str, float]) -> Dict:
        """
        实时检测 (API调用接口)

        Args:
            metrics: 指标字典

        Returns:
            检测结果字典
        """
        anomalies = self.detect(metrics, threshold=0.8)

        has_critical = any(a.severity == Severity.CRITICAL for a in anomalies)
        has_high = any(a.severity == Severity.HIGH for a in anomalies)

        return {
            "status": "critical" if has_critical else "warning" if has_high else "healthy",
            "anomaly_count": len(anomalies),
            "anomalies": [a.to_dict() for a in anomalies],
            "timestamp": datetime.now().isoformat(),
        }

    def train(self, historical_data: Dict[str, List[float]]):
        """
        训练ML检测器

        Args:
            historical_data: 历史数据字典
        """
        self.ml_detector.fit(historical_data)

    def get_metrics_summary(self) -> Dict:
        """获取指标摘要"""
        summary = {}
        for metric in self.statistical_detector.histories:
            history = list(self.statistical_detector.histories[metric])
            if history:
                summary[metric] = {
                    "count": len(history),
                    "mean": float(np.mean(history)),
                    "std": float(np.std(history)),
                    "min": float(np.min(history)),
                    "max": float(np.max(history)),
                    "recent": history[-10:],
                }
        return summary

    def get_health_score(self) -> Dict:
        """获取系统健康度评分"""
        recent_anomalies = [
            a for a in self.anomaly_history
            if a.timestamp > datetime.now() - timedelta(hours=1)
        ]

        if not recent_anomalies:
            return {
                "score": 100,
                "status": "healthy",
                "anomaly_count_1h": 0,
            }

        # 根据异常数量和严重程度计算健康度
        severity_weights = {
            Severity.CRITICAL: 25,
            Severity.HIGH: 15,
            Severity.MEDIUM: 8,
            Severity.LOW: 3,
        }

        penalty = sum(severity_weights.get(a.severity, 5) for a in recent_anomalies)
        health_score = max(0, 100 - penalty)

        return {
            "score": health_score,
            "status": "critical" if health_score < 40 else "warning" if health_score < 70 else "healthy",
            "anomaly_count_1h": len(recent_anomalies),
        }
