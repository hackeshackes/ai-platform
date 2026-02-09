"""
监控告警系统 - Phase 2
"""
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from uuid import uuid4
import asyncio

class AlertSeverity(Enum):
    """告警级别"""
    CRITICAL = "critical"  # 严重
    HIGH = "high"          # 高
    MEDIUM = "medium"      # 中
    LOW = "low"            # 低
    INFO = "info"          # 信息

class AlertStatus(Enum):
    """告警状态"""
    FIRING = "firing"      # 触发中
    RESOLVED = "resolved"   # 已解决
    PENDING = "pending"      # 待确认
    SILENCED = "silenced"   # 已静默

@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    condition: str  # e.g., "gpu_utilization > 90"
    threshold: float
    duration_seconds: int  # 持续时间
    severity: AlertSeverity
    channels: List[str]  # webhook, email, slack
    enabled: bool = True

@dataclass
class Alert:
    """告警实例"""
    alert_id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    value: float
    threshold: float
    message: str
    fired_at: datetime
    resolved_at: Optional[datetime] = None
    labels: Dict = field(default_factory=dict)

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.channels: Dict[str, Callable] = {}
        self._register_default_rules()
        self._register_default_channels()
    
    def _register_default_rules(self):
        """注册默认告警规则"""
        default_rules = [
            AlertRule(
                rule_id="gpu-util-high",
                name="GPU利用率过高",
                condition="gpu_utilization > 90",
                threshold=90,
                duration_seconds=300,
                severity=AlertSeverity.HIGH,
                channels=["webhook", "slack"]
            ),
            AlertRule(
                rule_id="gpu-memory-high",
                name="GPU内存过高",
                condition="gpu_memory_usage > 90",
                threshold=90,
                duration_seconds=300,
                severity=AlertSeverity.HIGH,
                channels=["webhook", "email"]
            ),
            AlertRule(
                rule_id="task-failed",
                name="任务失败",
                condition="task_status == failed",
                threshold=1,
                duration_seconds=0,
                severity=AlertSeverity.HIGH,
                channels=["webhook", "email"]
            ),
            AlertRule(
                rule_id="queue-backlog",
                name="队列积压",
                condition="queue_size > 100",
                threshold=100,
                duration_seconds=600,
                severity=AlertSeverity.MEDIUM,
                channels=["webhook"]
            ),
            AlertRule(
                rule_id="disk-space-low",
                name="磁盘空间不足",
                condition="disk_usage > 85",
                threshold=85,
                duration_seconds=3600,
                severity=AlertSeverity.MEDIUM,
                channels=["webhook", "email"]
            ),
            AlertRule(
                rule_id="api-latency-high",
                name="API延迟过高",
                condition="api_latency_ms > 1000",
                threshold=1000,
                duration_seconds=300,
                severity=AlertSeverity.LOW,
                channels=["slack"]
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.rule_id] = rule
    
    def _register_default_channels(self):
        """注册默认通知渠道"""
        self.channels["webhook"] = self._send_webhook
        self.channels["email"] = self._send_email
        self.channels["slack"] = self._send_slack
    
    async def _send_webhook(self, alert: Alert):
        """发送Webhook通知"""
        # requests.post(webhook_url, json=alert)
        pass
    
    async def _send_email(self, alert: Alert):
        """发送邮件通知"""
        # smtplib发送邮件
        pass
    
    async def _send_slack(self, alert: Alert):
        """发送Slack通知"""
        # slack_sdk发送消息
        pass
    
    async def check_condition(
        self,
        rule_id: str,
        value: float,
        labels: Optional[Dict] = None
    ) -> Optional[Alert]:
        """检查条件是否触发告警"""
        rule = self.rules.get(rule_id)
        if not rule or not rule.enabled:
            return None
        
        # 评估条件
        triggered = False
        if rule.condition.endswith(">") and value > rule.threshold:
            triggered = True
        elif rule.condition.endswith("<") and value < rule.threshold:
            triggered = True
        elif rule.condition.endswith("==") and value == rule.threshold:
            triggered = True
        
        if triggered:
            return await self.fire_alert(rule, value, labels or {})
        return None
    
    async def fire_alert(
        self,
        rule: AlertRule,
        value: float,
        labels: Dict
    ) -> Alert:
        """触发告警"""
        alert = Alert(
            alert_id=str(uuid4()),
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.FIRING,
            value=value,
            threshold=rule.threshold,
            message=f"{rule.name}: 当前值 {value} > 阈值 {rule.threshold}",
            fired_at=datetime.utcnow(),
            labels=labels
        )
        
        self.alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # 发送通知
        for channel in rule.channels:
            if channel in self.channels:
                await self.channels[channel](alert)
        
        return alert
    
    async def resolve_alert(self, alert_id: str):
        """解决告警"""
        alert = self.alerts.get(alert_id)
        if not alert:
            raise ValueError(f"Alert {alert_id} not found")
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
    
    def silence_alert(self, alert_id: str, duration_hours: int = 24):
        """静默告警"""
        alert = self.alerts.get(alert_id)
        if alert:
            alert.status = AlertStatus.SILENCED
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """获取活跃告警"""
        alerts = [
            a for a in self.alerts.values()
            if a.status == AlertStatus.FIRING
        ]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts
    
    def get_alert_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Alert]:
        """获取告警历史"""
        history = self.alert_history
        
        if start_time:
            history = [a for a in history if a.fired_at >= start_time]
        if end_time:
            history = [a for a in history if a.fired_at <= end_time]
        
        return history[-limit:]
    
    def add_custom_rule(self, rule: AlertRule):
        """添加自定义规则"""
        self.rules[rule.rule_id] = rule
    
    def get_rules(self) -> List[Dict]:
        """获取所有规则"""
        return [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "condition": r.condition,
                "threshold": r.threshold,
                "severity": r.severity.value,
                "enabled": r.enabled
            }
            for r in self.rules.values()
        ]

# 告警管理器实例
alert_manager = AlertManager()
