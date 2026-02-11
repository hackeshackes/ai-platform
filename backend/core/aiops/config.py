"""
AIOps 配置文件

包含所有模块的配置参数
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AnomalyDetectorConfig:
    """异常检测配置"""
    # 统计检测器配置
    window_size: int = 60  # 滑动窗口大小
    threshold_std: float = 3.0  # 标准差阈值

    # 检测阈值
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    ml_threshold: float = 0.8

    # 严重程度阈值
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "critical": 0.95,
        "high": 0.8,
        "medium": 0.6,
        "low": 0.3,
    })

    # 指标阈值
    metric_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "cpu": {"critical": 95, "high": 85, "medium": 70, "low": 50},
        "memory": {"critical": 95, "high": 85, "medium": 70, "low": 60},
        "disk": {"critical": 95, "high": 85, "medium": 75, "low": 60},
        "latency": {"critical": 1000, "high": 500, "medium": 200, "low": 100},
        "error_rate": {"critical": 10, "high": 5, "medium": 2, "low": 1},
        "requests_per_second": {"critical": 10000, "high": 5000, "medium": 2000, "low": 500},
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "threshold_std": self.threshold_std,
            "zscore_threshold": self.zscore_threshold,
            "iqr_multiplier": self.iqr_multiplier,
            "ml_threshold": self.ml_threshold,
            "severity_thresholds": self.severity_thresholds,
            "metric_thresholds": self.metric_thresholds,
        }


@dataclass
class RootCauseAnalyzerConfig:
    """根因分析配置"""
    # 相关性阈值
    correlation_threshold: float = 0.7
    confidence_threshold: float = 0.6

    # 分析超时
    analysis_timeout_ms: int = 60000  # 60秒

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_threshold": self.correlation_threshold,
            "confidence_threshold": self.confidence_threshold,
            "analysis_timeout_ms": self.analysis_timeout_ms,
        }


@dataclass
class AutoRecoveryConfig:
    """自动恢复配置"""
    # 是否启用自动恢复
    auto_recovery_enabled: bool = True

    # 最大重试次数
    max_retry_count: int = 3

    # 回滚阈值 (错误率超过此值自动回滚)
    rollback_threshold: float = 0.7

    # 执行超时
    execution_timeout_ms: int = 300000  # 5分钟

    # 灰度回滚配置
    gray_rollback_enabled: bool = True
    default_rollback_percentage: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auto_recovery_enabled": self.auto_recovery_enabled,
            "max_retry_count": self.max_retry_count,
            "rollback_threshold": self.rollback_threshold,
            "execution_timeout_ms": self.execution_timeout_ms,
            "gray_rollback_enabled": self.gray_rollback_enabled,
            "default_rollback_percentage": self.default_rollback_percentage,
        }


@dataclass
class PredictiveMaintenanceConfig:
    """预测性维护配置"""
    # 预警时间 (提前多少小时预警)
    early_warning_hours: int = 24

    # 预测间隔 (秒)
    prediction_interval: int = 3600

    # 窗口大小
    window_size: int = 24

    # 容量规划配置
    capacity_planning_days: int = 30
    safety_margin: float = 1.2  # 20%安全边际

    def to_dict(self) -> Dict[str, Any]:
        return {
            "early_warning_hours": self.early_warning_hours,
            "prediction_interval": self.prediction_interval,
            "window_size": self.window_size,
            "capacity_planning_days": self.capacity_planning_days,
            "safety_margin": self.safety_margin,
        }


@dataclass
class APIConfig:
    """API配置"""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False

    # CORS配置
    cors_enabled: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])

    # 认证配置
    auth_enabled: bool = False
    api_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "cors_enabled": self.cors_enabled,
            "cors_origins": self.cors_origins,
            "auth_enabled": self.auth_enabled,
            "api_key": "***" if self.api_key else None,
        }


@dataclass
class LoggingConfig:
    """日志配置"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_size_mb: int = 100
    backup_count: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "format": self.format,
            "file_path": self.file_path,
            "max_size_mb": self.max_size_mb,
            "backup_count": self.backup_count,
        }


@dataclass
class Config:
    """主配置类"""
    anomaly_detector: AnomalyDetectorConfig = field(default_factory=AnomalyDetectorConfig)
    root_cause_analyzer: RootCauseAnalyzerConfig = field(default_factory=RootCauseAnalyzerConfig)
    auto_recovery: AutoRecoveryConfig = field(default_factory=AutoRecoveryConfig)
    predictive_maintenance: PredictiveMaintenanceConfig = field(default_factory=PredictiveMaintenanceConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """从字典创建配置"""
        config = cls()

        if "anomaly_detector" in config_dict:
            ad = config_dict["anomaly_detector"]
            config.anomaly_detector = AnomalyDetectorConfig(
                window_size=ad.get("window_size", config.anomaly_detector.window_size),
                threshold_std=ad.get("threshold_std", config.anomaly_detector.threshold_std),
                zscore_threshold=ad.get("zscore_threshold", config.anomaly_detector.zscore_threshold),
                iqr_multiplier=ad.get("iqr_multiplier", config.anomaly_detector.iqr_multiplier),
                ml_threshold=ad.get("ml_threshold", config.anomaly_detector.ml_threshold),
                severity_thresholds=ad.get("severity_thresholds", config.anomaly_detector.severity_thresholds),
                metric_thresholds=ad.get("metric_thresholds", config.anomaly_detector.metric_thresholds),
            )

        if "root_cause_analyzer" in config_dict:
            rc = config_dict["root_cause_analyzer"]
            config.root_cause_analyzer = RootCauseAnalyzerConfig(
                correlation_threshold=rc.get("correlation_threshold", config.root_cause_analyzer.correlation_threshold),
                confidence_threshold=rc.get("confidence_threshold", config.root_cause_analyzer.confidence_threshold),
            )

        if "auto_recovery" in config_dict:
            ar = config_dict["auto_recovery"]
            config.auto_recovery = AutoRecoveryConfig(
                auto_recovery_enabled=ar.get("auto_recovery_enabled", config.auto_recovery.auto_recovery_enabled),
                max_retry_count=ar.get("max_retry_count", config.auto_recovery.max_retry_count),
                rollback_threshold=ar.get("rollback_threshold", config.auto_recovery.rollback_threshold),
            )

        if "predictive_maintenance" in config_dict:
            pm = config_dict["predictive_maintenance"]
            config.predictive_maintenance = PredictiveMaintenanceConfig(
                early_warning_hours=pm.get("early_warning_hours", config.predictive_maintenance.early_warning_hours),
                prediction_interval=pm.get("prediction_interval", config.predictive_maintenance.prediction_interval),
                window_size=pm.get("window_size", config.predictive_maintenance.window_size),
            )

        if "api" in config_dict:
            api = config_dict["api"]
            config.api = APIConfig(
                host=api.get("host", config.api.host),
                port=api.get("port", config.api.port),
                debug=api.get("debug", config.api.debug),
            )

        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_detector": self.anomaly_detector.to_dict(),
            "root_cause_analyzer": self.root_cause_analyzer.to_dict(),
            "auto_recovery": self.auto_recovery.to_dict(),
            "predictive_maintenance": self.predictive_maintenance.to_dict(),
            "api": self.api.to_dict(),
            "logging": self.logging.to_dict(),
        }


# 默认配置实例
default_config = Config()


def load_config(config_path: str) -> Config:
    """
    从文件加载配置

    Args:
        config_path: 配置文件路径 (JSON或YAML)

    Returns:
        Config实例
    """
    import json
    import yaml

    if config_path.endswith(".json"):
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path}")

    return Config.from_dict(config_dict)


def save_config(config: Config, config_path: str):
    """
    保存配置到文件

    Args:
        config: Config实例
        config_path: 配置文件路径
    """
    import json
    import yaml

    config_dict = config.to_dict()

    if config_path.endswith(".json"):
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path}")
