"""
Alert Engine - AI Platform v4

告警引擎 - 实时监控异常并发送告警通知
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel
import uuid
import asyncio


class AlertSeverity(Enum):
    """告警级别"""
    CRITICAL = "critical"  # 立即处理
    WARNING = "warning"    # 需要关注
    INFO = "info"          # 信息性


class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertType(Enum):
    """告警类型"""
    COST_SPIKE = "cost_spike"
    HIGH_LATENCY = "high_latency"
    ERROR_RATE = "error_rate"
    TOKEN_USAGE = "token_usage"
    PROVIDER_DOWN = "provider_down"
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMIT = "rate_limit"


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    alert_type: AlertType
    condition: str  # e.g., "cost > 100"
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 5
    message_template: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "alert_type": self.alert_type.value,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "enabled": self.enabled,
            "cooldown_minutes": self.cooldown_minutes,
            "message_template": self.message_template
        }


@dataclass
class Alert:
    """告警实例"""
    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus
    title: str
    message: str
    current_value: float
    threshold: float
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    provider: Optional[str] = None
    model: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "message": self.message,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "triggered_at": self.triggered_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
            "provider": self.provider,
            "model": self.model
        }


class AlertEngine:
    """
    告警引擎 - 监控指标并在异常时触发告警
    
    功能:
    - 灵活的告警规则配置
    - 多级别告警支持
    - 告警抑制和聚合
    - 多渠道通知
    """
    
    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: List[Alert] = []
        self._alert_callbacks: List[Callable] = []
        self._cooldowns: Dict[str, datetime] = {}
        self._notification_channels: Dict[str, Any] = {}
        
    # ============ 规则管理 ============
    
    async def create_rule(self, rule: AlertRule) -> AlertRule:
        """创建告警规则"""
        self._rules[rule.rule_id] = rule
        return rule
    
    async def update_rule(self, rule_id: str, **kwargs) -> Optional[AlertRule]:
        """更新告警规则"""
        if rule_id not in self._rules:
            return None
        
        rule = self._rules[rule_id]
        for key, value in kwargs.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        return rule
    
    async def delete_rule(self, rule_id: str) -> bool:
        """删除告警规则"""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False
    
    async def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """获取告警规则"""
        return self._rules.get(rule_id)
    
    async def list_rules(self, enabled_only: bool = False) -> List[AlertRule]:
        """列出所有告警规则"""
        rules = list(self._rules.values())
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        return rules
    
    # ============ 内置规则 ============
    
    async def create_default_rules(self):
        """创建默认告警规则"""
        default_rules = [
            AlertRule(
                rule_id="cost_hourly_high",
                name="Hourly Cost Alert",
                alert_type=AlertType.COST_SPIKE,
                condition="cost > threshold",
                threshold=10.0,
                severity=AlertSeverity.WARNING,
                cooldown_minutes=60,
                message_template="Hourly cost ${current_value:.2f} exceeds threshold ${threshold:.2f}"
            ),
            AlertRule(
                rule_id="cost_daily_budget",
                name="Daily Cost Budget",
                alert_type=AlertType.COST_SPIKE,
                condition="cost > threshold",
                threshold=100.0,
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=360,
                message_template="Daily cost ${current_value:.2f} exceeds budget ${threshold:.2f}"
            ),
            AlertRule(
                rule_id="latency_p95",
                name="High P95 Latency",
                alert_type=AlertType.HIGH_LATENCY,
                condition="latency_p95 > threshold",
                threshold=5000.0,
                severity=AlertSeverity.WARNING,
                cooldown_minutes=15,
                message_template="P95 latency ${current_value:.0f}ms exceeds threshold ${threshold:.0f}ms"
            ),
            AlertRule(
                rule_id="error_rate_high",
                name="High Error Rate",
                alert_type=AlertType.ERROR_RATE,
                condition="error_rate > threshold",
                threshold=5.0,
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=10,
                message_template="Error rate ${current_value:.1f}% exceeds threshold ${threshold:.1f}%"
            ),
            AlertRule(
                rule_id="tokens_daily_limit",
                name="Daily Token Limit",
                alert_type=AlertType.TOKEN_USAGE,
                condition="tokens > threshold",
                threshold=1000000,
                severity=AlertSeverity.WARNING,
                cooldown_minutes=360,
                message_template="Daily tokens ${current_value:,} exceeds threshold ${threshold:,}"
            ),
            AlertRule(
                rule_id="provider_unavailable",
                name="Provider Unavailable",
                alert_type=AlertType.PROVIDER_DOWN,
                condition="provider_errors > threshold",
                threshold=10,
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=5,
                message_template="Provider ${provider} has ${current_value} consecutive errors"
            )
        ]
        
        for rule in default_rules:
            if rule.rule_id not in self._rules:
                await self.create_rule(rule)
    
    # ============ 告警触发 ============
    
    async def check_and_alert(
        self,
        alert_type: AlertType,
        current_value: float,
        threshold: float,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> List[Alert]:
        """
        检查指标并触发告警
        
        Args:
            alert_type: 告警类型
            current_value: 当前值
            threshold: 阈值
            provider: 提供商
            model: 模型
            metadata: 附加信息
            
        Returns:
            触发的告警列表
        """
        triggered_alerts = []
        
        # 查找匹配规则
        matching_rules = [
            r for r in self._rules.values()
            if r.enabled and r.alert_type == alert_type
        ]
        
        for rule in matching_rules:
            # 检查是否应该触发
            should_trigger = self._evaluate_condition(
                rule.condition, current_value, threshold
            )
            
            if should_trigger:
                # 检查冷却时间
                cooldown_key = f"{rule.rule_id}:{provider or 'global'}"
                if self._in_cooldown(cooldown_key, rule.cooldown_minutes):
                    continue
                
                # 创建告警
                alert = await self._create_alert(
                    rule=rule,
                    current_value=current_value,
                    threshold=threshold,
                    provider=provider,
                    model=model,
                    metadata=metadata
                )
                triggered_alerts.append(alert)
                
                # 设置冷却
                self._set_cooldown(cooldown_key)
        
        return triggered_alerts
    
    def _evaluate_condition(self, condition: str, current: float, threshold: float) -> bool:
        """评估触发条件"""
        if condition == "cost > threshold" or condition == ">":
            return current > threshold
        elif condition == "cost < threshold" or condition == "<":
            return current < threshold
        return False
    
    def _in_cooldown(self, key: str, cooldown_minutes: int) -> bool:
        """检查是否在冷却期"""
        if key not in self._cooldowns:
            return False
        last_triggered = self._cooldowns[key]
        elapsed = (datetime.now() - last_triggered).total_seconds() / 60
        return elapsed < cooldown_minutes
    
    def _set_cooldown(self, key: str):
        """设置冷却时间"""
        self._cooldowns[key] = datetime.now()
    
    async def _create_alert(
        self,
        rule: AlertRule,
        current_value: float,
        threshold: float,
        provider: Optional[str],
        model: Optional[str],
        metadata: Optional[Dict]
    ) -> Alert:
        """创建告警实例"""
        # 生成消息
        message = rule.message_template
        message = message.replace("${current_value}", f"{current_value:.2f}")
        message = message.replace("${threshold}", f"{threshold:.2f}")
        if provider:
            message = message.replace("${provider}", provider)
        if model:
            message = message.replace("${model}", model)
        
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            alert_type=rule.alert_type,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            title=f"{rule.name}",
            message=message,
            current_value=current_value,
            threshold=threshold,
            triggered_at=datetime.now(),
            metadata=metadata or {},
            provider=provider,
            model=model
        )
        
        self._alerts.append(alert)
        
        # 触发回调
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")
        
        # 发送通知
        await self._send_notification(alert)
        
        return alert
    
    async def _send_notification(self, alert: Alert):
        """发送告警通知"""
        for channel, handler in self._notification_channels.items():
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                print(f"Notification error on channel {channel}: {e}")
    
    # ============ 告警管理 ============
    
    async def acknowledge_alert(self, alert_id: str) -> Optional[Alert]:
        """确认告警"""
        for alert in self._alerts:
            if alert.alert_id == alert_id and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                return alert
        return None
    
    async def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        """解决告警"""
        for alert in self._alerts:
            if alert.alert_id == alert_id and alert.status != AlertStatus.RESOLVED:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                return alert
        return None
    
    async def suppress_alert(self, alert_id: str) -> Optional[Alert]:
        """抑制告警"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.SUPPRESSED
                return alert
        return None
    
    async def list_alerts(
        self,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[Alert]:
        """列出告警"""
        alerts = self._alerts
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # 按时间倒序
        alerts = sorted(alerts, key=lambda a: a.triggered_at, reverse=True)
        
        return alerts[:limit]
    
    async def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return await self.list_alerts(status=AlertStatus.ACTIVE)
    
    async def get_alert_stats(self) -> Dict[str, Any]:
        """获取告警统计"""
        stats = {
            "total": len(self._alerts),
            "active": 0,
            "acknowledged": 0,
            "resolved": 0,
            "by_severity": {},
            "by_type": {},
            "by_provider": {}
        }
        
        for alert in self._alerts:
            status_key = alert.status.value
            stats[status_key] = stats.get(status_key, 0) + 1
            
            # 按严重级别
            sev_key = alert.severity.value
            stats["by_severity"][sev_key] = stats["by_severity"].get(sev_key, 0) + 1
            
            # 按类型
            type_key = alert.alert_type.value
            stats["by_type"][type_key] = stats["by_type"].get(type_key, 0) + 1
            
            # 按提供商
            if alert.provider:
                stats["by_provider"][alert.provider] = stats["by_provider"].get(alert.provider, 0) + 1
        
        return stats
    
    # ============ 回调和通知 ============
    
    def add_callback(self, callback: Callable):
        """添加告警回调"""
        self._alert_callbacks.append(callback)
    
    def add_notification_channel(self, channel_name: str, handler: Any):
        """添加通知渠道"""
        self._notification_channels[channel_name] = handler
    
    def remove_notification_channel(self, channel_name: str):
        """移除通知渠道"""
        if channel_name in self._notification_channels:
            del self._notification_channels[channel_name]


# 创建全局告警引擎实例
alert_engine = AlertEngine()


def get_alert_engine() -> AlertEngine:
    """获取告警引擎实例"""
    return alert_engine
