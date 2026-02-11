#!/usr/bin/env python3
"""
监控器 - monitor.py

功能:
- 实时监控
- 告警通知
- 日志收集
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from logging.handlers import RotatingFileHandler


class AlertLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class Metric:
    """监控指标"""
    name: str
    value: float
    unit: str
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Alert:
    """告警"""
    id: str
    name: str
    level: AlertLevel
    message: str
    deployment_id: str
    metric_name: str
    current_value: float
    threshold: float
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    notify_channels: List[str] = field(default_factory=list)


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: str
    level: str
    message: str
    deployment_id: str
    source: str = ""
    trace_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict] = []
        self.notification_channels: Dict[str, Callable] = {}
        self.alert_callbacks: List[Callable] = []
        
        # 配置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志"""
        log_path = self.config.get("log_path", "/var/log/ai-platform")
        self.logger = logging.getLogger("monitor")
        self.logger.setLevel(logging.DEBUG)
        
        # 文件处理器
        fh = RotatingFileHandler(
            f"{log_path}/monitor.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        fh.setLevel(logging.DEBUG)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def add_alert_rule(self, rule: Dict):
        """添加告警规则"""
        self.alert_rules.append(rule)
    
    def add_notification_channel(self, channel: str, callback: Callable):
        """添加通知渠道"""
        self.notification_channels[channel] = callback
    
    def on_alert(self, callback: Callable):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    def check_rule(self, metric: Metric) -> Optional[Alert]:
        """检查指标是否触发告警"""
        for rule in self.alert_rules:
            if rule["metric"] != metric.name:
                continue
            
            operator = rule.get("operator", "gt")
            threshold = rule["threshold"]
            
            should_alert = False
            if operator == "gt" and metric.value > threshold:
                should_alert = True
            elif operator == "gte" and metric.value >= threshold:
                should_alert = True
            elif operator == "lt" and metric.value < threshold:
                should_alert = True
            elif operator == "lte" and metric.value <= threshold:
                should_alert = True
            elif operator == "eq" and metric.value == threshold:
                should_alert = True
            
            if should_alert:
                return self._create_alert(metric, rule)
        
        return None
    
    def _create_alert(self, metric: Metric, rule: Dict) -> Alert:
        """创建告警"""
        import uuid
        
        alert_id = str(uuid.uuid4())[:8]
        
        level = AlertLevel(rule.get("level", "warning"))
        
        alert = Alert(
            id=alert_id,
            name=rule["name"],
            level=level,
            message=f"{metric.name} = {metric.value}{metric.unit}, 阈值: {rule['threshold']}",
            deployment_id=metric.labels.get("deployment_id", "unknown"),
            metric_name=metric.name,
            current_value=metric.value,
            threshold=rule["threshold"],
            notify_channels=rule.get("notify_channels", ["console"])
        )
        
        self.alerts[alert_id] = alert
        self._trigger_alert(alert)
        
        return alert
    
    def _trigger_alert(self, alert: Alert):
        """触发告警通知"""
        self.logger.warning(f"[ALERT] {alert.name}: {alert.message}")
        
        # 调用回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"告警回调失败: {e}")
        
        # 发送到通知渠道
        for channel in alert.notify_channels:
            if channel in self.notification_channels:
                try:
                    self.notification_channels[channel](alert)
                except Exception as e:
                    self.logger.error(f"通知发送失败 ({channel}): {e}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        if alert_id not in self.alerts:
            return False
        
        self.alerts[alert_id].status = AlertStatus.ACKNOWLEDGED
        self.alerts[alert_id].acknowledged_at = datetime.now().isoformat()
        return True
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        if alert_id not in self.alerts:
            return False
        
        self.alerts[alert_id].status = AlertStatus.RESOLVED
        self.alerts[alert_id].resolved_at = datetime.now().isoformat()
        return True
    
    def get_active_alerts(self, deployment_id: str = None) -> List[Alert]:
        """获取活跃告警"""
        alerts = [
            a for a in self.alerts.values() 
            if a.status == AlertStatus.ACTIVE
        ]
        
        if deployment_id:
            alerts = [a for a in alerts if a.deployment_id == deployment_id]
        
        return alerts
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            a for a in self.alerts.values()
            if datetime.fromisoformat(a.created_at) > cutoff
        ]


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.metrics: Dict[str, deque] = {}
        self.collectors: List[Callable] = []
        self._running = False
        
        # 保留时间（秒）
        self.retention_seconds = self.config.get("retention_seconds", 3600)
    
    def register_collector(self, collector: Callable):
        """注册指标收集器"""
        self.collectors.append(collector)
    
    async def start(self, interval: int = 10):
        """启动指标收集"""
        self._running = True
        print("[MetricsCollector] 启动指标收集...")
        
        while self._running:
            try:
                await self._collect_all()
            except Exception as e:
                print(f"[MetricsCollector] 收集错误: {e}")
            
            await asyncio.sleep(interval)
    
    def stop(self):
        """停止指标收集"""
        self._running = False
    
    async def _collect_all(self):
        """收集所有指标"""
        for collector in self.collectors:
            try:
                metrics = await collector()
                for metric in metrics:
                    self._store_metric(metric)
            except Exception as e:
                print(f"[MetricsCollector] 收集器错误: {e}")
    
    def _store_metric(self, metric: Metric):
        """存储指标"""
        if metric.name not in self.metrics:
            self.metrics[metric.name] = deque(maxlen=1000)
        
        self.metrics[metric.name].append(metric)
        
        # 清理过期指标
        self._cleanup_expired()
    
    def _cleanup_expired(self):
        """清理过期指标"""
        cutoff = datetime.now() - timedelta(seconds=self.retention_seconds)
        
        for metric_name in self.metrics:
            valid_metrics = []
            for metric in self.metrics[metric_name]:
                if datetime.fromisoformat(metric.timestamp) > cutoff:
                    valid_metrics.append(metric)
            self.metrics[metric_name] = deque(valid_metrics, maxlen=1000)
    
    def get_metric(self, name: str, limit: int = 100) -> List[Metric]:
        """获取指标"""
        if name not in self.metrics:
            return []
        return list(self.metrics[name])[-limit:]
    
    def get_metric_summary(self, name: str, minutes: int = 5) -> Dict:
        """获取指标摘要"""
        metrics = self.get_metric(name)
        if not metrics:
            return {}
        
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent = [m for m in metrics if datetime.fromisoformat(m.timestamp) > cutoff]
        
        if not recent:
            return {}
        
        values = [m.value for m in recent]
        
        return {
            "name": name,
            "count": len(recent),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": recent[-1].value,
            "unit": recent[-1].unit
        }


class LogCollector:
    """日志收集器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logs: Dict[str, deque] = {}
        self._running = False
        self.log_callbacks: List[Callable] = []
        
        # 日志保留时间（秒）
        self.retention_seconds = self.config.get("retention_seconds", 3600)
    
    async def start(self):
        """启动日志收集"""
        self._running = True
        print("[LogCollector] 启动日志收集...")
    
    def stop(self):
        """停止日志收集"""
        self._running = False
    
    def add_log(self, log: LogEntry):
        """添加日志"""
        if log.deployment_id not in self.logs:
            self.logs[log.deployment_id] = deque(maxlen=10000)
        
        self.logs[log.deployment_id].append(log)
        
        # 清理过期日志
        self._cleanup_expired()
        
        # 触发回调
        for callback in self.log_callbacks:
            try:
                callback(log)
            except Exception:
                pass
    
    def on_log(self, callback: Callable):
        """添加日志回调"""
        self.log_callbacks.append(callback)
    
    def _cleanup_expired(self):
        """清理过期日志"""
        cutoff = datetime.now() - timedelta(seconds=self.retention_seconds)
        
        for deployment_id in self.logs:
            valid_logs = []
            for log in self.logs[deployment_id]:
                if datetime.fromisoformat(log.timestamp) > cutoff:
                    valid_logs.append(log)
            self.logs[deployment_id] = deque(valid_logs, maxlen=10000)
    
    def get_logs(
        self, 
        deployment_id: str = None, 
        level: str = None,
        since: datetime = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """获取日志"""
        logs = []
        
        for dep_id, dep_logs in self.logs.items():
            if deployment_id and dep_id != deployment_id:
                continue
            
            for log in dep_logs:
                if level and log.level != level:
                    continue
                if since and datetime.fromisoformat(log.timestamp) < since:
                    continue
                logs.append(log)
        
        return logs[-limit:]
    
    def search_logs(self, query: str, deployment_id: str = None, limit: int = 100) -> List[LogEntry]:
        """搜索日志"""
        results = []
        query_lower = query.lower()
        
        for dep_id, dep_logs in self.logs.items():
            if deployment_id and dep_id != deployment_id:
                continue
            
            for log in dep_logs:
                if query_lower in log.message.lower():
                    results.append(log)
        
        return results[-limit:]


class Monitor:
    """监控系统主类"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.alert_manager = AlertManager(config)
        self.metrics_collector = MetricsCollector(config)
        self.log_collector = LogCollector(config)
        self._running = False
        
        # 设置默认告警规则
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """设置默认告警规则"""
        default_rules = [
            {
                "name": "CPU使用率过高",
                "metric": "cpu_percent",
                "operator": "gt",
                "threshold": 80,
                "level": "warning",
                "notify_channels": ["console", "email"]
            },
            {
                "name": "CPU使用率严重过高",
                "metric": "cpu_percent",
                "operator": "gt",
                "threshold": 95,
                "level": "critical",
                "notify_channels": ["console", "sms"]
            },
            {
                "name": "内存使用率过高",
                "metric": "memory_percent",
                "operator": "gt",
                "threshold": 85,
                "level": "warning",
                "notify_channels": ["console"]
            },
            {
                "name": "错误率过高",
                "metric": "error_rate",
                "operator": "gt",
                "threshold": 1.0,
                "level": "error",
                "notify_channels": ["console", "email"]
            },
            {
                "name": "延迟过高",
                "metric": "latency_p99",
                "operator": "gt",
                "threshold": 1000,
                "level": "warning",
                "notify_channels": ["console"]
            }
        ]
        
        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)
    
    async def start(self, metrics_interval: int = 10):
        """启动监控"""
        self._running = True
        print("[Monitor] 启动监控系统...")
        
        # 启动各组件
        await asyncio.gather(
            self.metrics_collector.start(metrics_interval),
            self.log_collector.start(),
            return_exceptions=True
        )
    
    def stop(self):
        """停止监控"""
        self._running = False
        self.metrics_collector.stop()
        self.log_collector.stop()
        print("[Monitor] 停止监控系统")
    
    def record_metric(self, metric: Metric):
        """记录指标"""
        self.metrics_collector._store_metric(metric)
        
        # 检查告警
        alert = self.alert_manager.check_rule(metric)
        if alert:
            print(f"[Monitor] 告警触发: {alert.name}")
    
    def record_log(self, log: LogEntry):
        """记录日志"""
        self.log_collector.add_log(log)
    
    def get_status(self, deployment_id: str = None) -> Dict:
        """获取监控状态"""
        active_alerts = self.alert_manager.get_active_alerts(deployment_id)
        
        return {
            "running": self._running,
            "active_alerts": len(active_alerts),
            "alerts": [a.__dict__ for a in active_alerts],
            "metrics_count": sum(len(m) for m in self.metrics_collector.metrics.values()),
            "logs_count": sum(len(l) for l in self.log_collector.logs.values())
        }
    
    def get_dashboard_data(self, deployment_id: str = None) -> Dict:
        """获取仪表盘数据"""
        return {
            "metrics": {
                name: self.metrics_collector.get_metric_summary(name)
                for name in self.metrics_collector.metrics
            },
            "alerts": {
                "active": len(self.alert_manager.get_active_alerts(deployment_id)),
                "recent": [
                    a.__dict__ for a in self.alert_manager.get_alert_history(1)
                ]
            },
            "logs": {
                "recent": [
                    l.__dict__ for l in self.log_collector.get_logs(deployment_id, limit=10)
                ]
            }
        }


# 默认系统监控收集器
async def default_system_collector():
    """默认系统指标收集器"""
    import psutil
    
    metrics = []
    
    # CPU
    metrics.append(Metric(
        name="cpu_percent",
        value=psutil.cpu_percent(interval=1),
        unit="%"
    ))
    
    # 内存
    mem = psutil.virtual_memory()
    metrics.append(Metric(
        name="memory_percent",
        value=mem.percent,
        unit="%"
    ))
    
    metrics.append(Metric(
        name="memory_used",
        value=mem.used / 1024 / 1024,
        unit="MB"
    ))
    
    # 磁盘
    disk = psutil.disk_usage('/')
    metrics.append(Metric(
        name="disk_percent",
        value=disk.percent,
        unit="%"
    ))
    
    # 网络
    net = psutil.net_io_counters()
    metrics.append(Metric(
        name="network_bytes_sent",
        value=net.bytes_sent,
        unit="bytes"
    ))
    metrics.append(Metric(
        name="network_bytes_recv",
        value=net.bytes_recv,
        unit="bytes"
    ))
    
    return metrics


if __name__ == "__main__":
    import json
    
    async def main():
        # 创建监控
        monitor = Monitor({"log_path": "./logs"})
        
        # 注册默认收集器
        monitor.metrics_collector.register_collector(default_system_collector)
        
        # 打印仪表盘数据
        await asyncio.sleep(2)
        print(json.dumps(monitor.get_dashboard_data(), indent=2, default=str))
    
    asyncio.run(main())
