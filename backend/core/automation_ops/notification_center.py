"""
通知中心 (Notification Center)
================================

功能:
- 多渠道通知
- 告警规则管理
- 升级策略
- 静默期管理

作者: AI Platform Team
版本: 1.0.0
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class NotificationChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    PHONE = "phone"
    WEBHOOK = "webhook"
    DINGTALK = "dingtalk"
    WECHAT = "wechat"
    PUSH = "push"
    TELEGRAM = "telegram"


class NotificationStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class AlertRule:
    id: str
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    level: AlertLevel
    channels: List[NotificationChannel]
    enabled: bool = True
    description: str = ""
    cooldown: int = 300
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def should_trigger(self) -> bool:
        if not self.enabled:
            return False
        if self.last_triggered:
            elapsed = (datetime.now() - self.last_triggered).total_seconds()
            if elapsed < self.cooldown:
                return False
        return True
    
    def record_trigger(self) -> None:
        self.last_triggered = datetime.now()
        self.trigger_count += 1


@dataclass
class EscalationPolicy:
    id: str
    name: str
    rules: List[Dict[str, Any]] = field(default_factory=list)
    default_timeout: int = 900
    max_escalation_level: int = 5
    description: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class SilencePeriod:
    id: str
    name: str
    start_time: datetime
    end_time: datetime
    reason: str = ""
    created_by: str = ""
    match_labels: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def is_active(self) -> bool:
        now = datetime.now()
        return self.start_time <= now <= self.end_time
    
    def matches(self, labels: Dict[str, str]) -> bool:
        for key, value in self.match_labels.items():
            if labels.get(key) != value:
                return False
        return True


@dataclass
class Notification:
    id: str
    alert_name: str
    level: AlertLevel
    title: str
    message: str
    channels: List[NotificationChannel]
    labels: Dict[str, str] = field(default_factory=dict)
    status: NotificationStatus = NotificationStatus.PENDING
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


class NotificationCenter:
    """通知中心"""
    
    def __init__(
        self,
        enable_metrics: bool = True,
        default_channels: Optional[List[NotificationChannel]] = None,
        max_retries: int = 3,
        retry_interval: int = 60
    ):
        self.enable_metrics = enable_metrics
        self.default_channels = default_channels or [NotificationChannel.EMAIL]
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        
        self._alert_rules: Dict[str, AlertRule] = {}
        self._escalation_policies: Dict[str, EscalationPolicy] = {}
        self._silence_periods: Dict[str, SilencePeriod] = {}
        self._notifications: List[Notification] = []
        self._channel_handlers: Dict[NotificationChannel, Callable] = {}
        
        self._metrics = {
            "total_notifications": 0,
            "sent_notifications": 0,
            "failed_notifications": 0,
            "alerts_triggered": 0
        }
        
        logger.info("Notification Center initialized")
    
    def register_channel(self, channel: NotificationChannel, handler: Callable[[Notification], Any]) -> None:
        self._channel_handlers[channel] = handler
        logger.info(f"Registered handler for channel: {channel.value}")
    
    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        level: AlertLevel,
        channels: List[NotificationChannel],
        enabled: bool = True,
        description: str = "",
        cooldown: int = 300,
        **kwargs
    ) -> str:
        rule = AlertRule(
            id=str(uuid.uuid4()),
            name=name,
            condition=condition,
            level=level,
            channels=channels,
            enabled=enabled,
            description=description,
            cooldown=cooldown,
            **kwargs
        )
        self._alert_rules[rule.id] = rule
        return rule.id
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        if rule_id in self._alert_rules:
            del self._alert_rules[rule_id]
            return True
        return False
    
    def get_alert_rules(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": r.id,
                "name": r.name,
                "level": r.level.value,
                "channels": [c.value for c in r.channels],
                "enabled": r.enabled,
                "description": r.description,
                "cooldown": r.cooldown,
                "last_triggered": r.last_triggered.isoformat() if r.last_triggered else None,
                "trigger_count": r.trigger_count
            }
            for r in self._alert_rules.values()
        ]
    
    def set_escalation_policy(self, policy: EscalationPolicy) -> str:
        self._escalation_policies[policy.id] = policy
        return policy.id
    
    def add_silence_period(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        reason: str = "",
        match_labels: Optional[Dict[str, str]] = None,
        created_by: str = ""
    ) -> str:
        silence = SilencePeriod(
            id=str(uuid.uuid4()),
            name=name,
            start_time=start_time,
            end_time=end_time,
            reason=reason,
            created_by=created_by,
            match_labels=match_labels or {}
        )
        self._silence_periods[silence.id] = silence
        return silence.id
    
    def remove_silence_period(self, silence_id: str) -> bool:
        if silence_id in self._silence_periods:
            del self._silence_periods[silence_id]
            return True
        return False
    
    def is_silenced(self, labels: Dict[str, str] = None) -> bool:
        for silence in self._silence_periods.values():
            if silence.is_active():
                if not labels or silence.matches(labels):
                    return True
        return False
    
    async def send_alert(
        self,
        alert_name: str,
        level: AlertLevel,
        title: str,
        message: str,
        channels: Optional[List[NotificationChannel]] = None,
        labels: Optional[Dict[str, str]] = None,
        escalation_policy_id: Optional[str] = None,
        **kwargs
    ) -> Notification:
        if self.is_silenced(labels):
            logger.info(f"Alert '{alert_name}' silenced by silence period")
            notification = Notification(
                id=str(uuid.uuid4()),
                alert_name=alert_name,
                level=level,
                title=title,
                message=message,
                channels=channels or self.default_channels,
                labels=labels or {},
                metadata={"silenced": True, **kwargs}
            )
            self._notifications.append(notification)
            return notification
        
        notification = Notification(
            id=str(uuid.uuid4()),
            alert_name=alert_name,
            level=level,
            title=title,
            message=message,
            channels=channels or self.default_channels,
            labels=labels or {},
            metadata={"escalation_policy_id": escalation_policy_id, **kwargs}
        )
        
        self._notifications.append(notification)
        self._metrics["total_notifications"] += 1
        
        asyncio.create_task(self._send_notification(notification))
        
        if escalation_policy_id:
            asyncio.create_task(self._check_escalation(notification, escalation_policy_id))
        
        logger.info(f"Alert sent: {alert_name} [{level.value}]")
        
        return notification
    
    async def _send_notification(self, notification: Notification) -> None:
        try:
            notification.status = NotificationStatus.SENT
            notification.sent_at = datetime.now()
            
            sent_channels = []
            failed_channels = []
            
            for channel in notification.channels:
                handler = self._channel_handlers.get(channel)
                
                if handler:
                    try:
                        await handler(notification)
                        sent_channels.append(channel.value)
                        if self.enable_metrics:
                            self._metrics["sent_notifications"] += 1
                    except Exception as e:
                        failed_channels.append(channel.value)
                        logger.error(f"Failed to send via {channel.value}: {e}")
                else:
                    logger.info(f"[{channel.value}] {notification.title}: {notification.message}")
                    sent_channels.append(channel.value)
            
            notification.metadata["sent_channels"] = sent_channels
            notification.metadata["failed_channels"] = failed_channels
            
            if failed_channels:
                notification.status = NotificationStatus.FAILED
                notification.error = f"Failed channels: {', '.join(failed_channels)}"
                if notification.retry_count < notification.max_retries:
                    notification.retry_count += 1
                    asyncio.create_task(self._retry_notification(notification))
            else:
                notification.status = NotificationStatus.DELIVERED
                notification.delivered_at = datetime.now()
        
        except Exception as e:
            notification.status = NotificationStatus.FAILED
            notification.error = str(e)
            logger.exception(f"Notification send error: {e}")
    
    async def _retry_notification(self, notification: Notification) -> None:
        await asyncio.sleep(self.retry_interval)
        if notification.retry_count < notification.max_retries:
            await self._send_notification(notification)
        else:
            self._metrics["failed_notifications"] += 1
    
    async def _check_escalation(self, notification: Notification, policy_id: str) -> None:
        policy = self._escalation_policies.get(policy_id)
        if not policy:
            return
        
        timeout = policy.default_timeout
        await asyncio.sleep(timeout)
        
        current_level = notification.level.value
        next_config = {
            "level": current_level + 1,
            "channels": [NotificationChannel.PHONE],
            "timeout": timeout
        }
        
        await self._send_escalation(notification, next_config)
    
    async def _send_escalation(self, notification: Notification, config: Dict[str, Any]) -> None:
        escalation_notification = Notification(
            id=str(uuid.uuid4()),
            alert_name=f"[ESCALATED] {notification.alert_name}",
            level=notification.level,
            title=f"升级告警: {notification.title}",
            message=notification.message,
            channels=config.get("channels", [NotificationChannel.PHONE]),
            labels=notification.labels,
            metadata={"escalation_level": config.get("level"), "original_notification_id": notification.id}
        )
        
        self._notifications.append(escalation_notification)
        asyncio.create_task(self._send_notification(escalation_notification))
    
    def acknowledge_notification(self, notification_id: str) -> bool:
        for notification in self._notifications:
            if notification.id == notification_id:
                notification.status = NotificationStatus.ACKNOWLEDGED
                return True
        return False
    
    def get_notifications(
        self,
        status: Optional[NotificationStatus] = None,
        alert_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        results = self._notifications
        
        if status:
            results = [n for n in results if n.status == status]
        if alert_name:
            results = [n for n in results if n.alert_name == alert_name]
        
        results = results[-limit:]
        
        return [
            {
                "id": n.id,
                "alert_name": n.alert_name,
                "level": n.level.value,
                "title": n.title,
                "status": n.status.value,
                "sent_at": n.sent_at.isoformat() if n.sent_at else None,
                "error": n.error
            }
            for n in results
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self._metrics,
            "active_rules": len(self._alert_rules),
            "active_silence_periods": sum(1 for s in self._silence_periods.values() if s.is_active()),
            "pending_notifications": len([n for n in self._notifications if n.status == NotificationStatus.PENDING])
        }
    
    def evaluate_rules(self, data: Dict[str, Any]) -> List[AlertRule]:
        """评估告警规则"""
        triggered = []
        
        for rule in self._alert_rules.values():
            if rule.should_trigger():
                try:
                    if rule.condition(data):
                        rule.record_trigger()
                        triggered.append(rule)
                except Exception as e:
                    logger.error(f"Rule evaluation error: {e}")
        
        return triggered
