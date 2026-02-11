"""
告警管理模块 - AlertManager

支持规则配置、告警触发和通知
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重程度"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(Enum):
    """告警状态"""
    FIRING = "firing"
    PENDING = "pending"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    condition: str  # 例如: "cpu_usage > 80"
    severity: AlertSeverity
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    duration_seconds: int = 60  # 持续时间
    enabled: bool = True
    # 计算属性
    metric_name: str = ""
    operator: str = ""
    threshold: float = 0.0
    
    def __post_init__(self):
        """解析条件表达式"""
        if self.condition:
            parts = self.condition.split()
            if len(parts) >= 3:
                self.metric_name = parts[0]
                self.operator = parts[1]
                try:
                    self.threshold = float(parts[2].rstrip('%'))
                except ValueError:
                    self.threshold = 0.0


@dataclass
class Alert:
    """告警实例"""
    rule: AlertRule
    status: AlertStatus
    current_value: float
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    fingerprint: str = ""
    evaluation_count: int = 0
    
    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = f"{self.rule.name}-{self.rule.severity.value}"


@dataclass
class AlertNotification:
    """告警通知"""
    alert: Alert
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sent: bool = False
    channels: List[str] = field(default_factory=list)


class AlertManager:
    """
    告警管理器
    
    管理告警规则、触发评估和通知发送
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化告警管理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._initialized = False
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._notifications: List[AlertNotification] = []
        
        # 通知配置
        self._notification_channels = self.config.get('channels', {})
        self._max_history_size = self.config.get('max_history_size', 1000)
        self._evaluation_interval = self.config.get('evaluation_interval', 15)
        
        # 回调函数
        self._on_alert_callbacks: List[Callable] = []
        
        # 加载默认规则
        self._load_default_rules()
    
    def _load_default_rules(self) -> None:
        """加载默认告警规则"""
        default_rules = [
            AlertRule(
                name="HighCPU",
                condition="cpu_usage > 80",
                severity=AlertSeverity.WARNING,
                message="CPU使用率过高",
                duration_seconds=60
            ),
            AlertRule(
                name="HighMemory",
                condition="memory_usage > 80",
                severity=AlertSeverity.WARNING,
                message="内存使用率过高",
                duration_seconds=60
            ),
            AlertRule(
                name="HighLatency",
                condition="api_latency_p95 > 1.0",
                severity=AlertSeverity.WARNING,
                message="API响应延迟过高",
                duration_seconds=120
            ),
            AlertRule(
                name="HighErrorRate",
                condition="error_rate > 1.0",
                severity=AlertSeverity.CRITICAL,
                message="错误率过高",
                duration_seconds=60
            ),
            AlertRule(
                name="DatabaseDown",
                condition="database_status == 0",
                severity=AlertSeverity.CRITICAL,
                message="数据库连接失败",
                duration_seconds=10
            ),
            AlertRule(
                name="RedisDown",
                condition="redis_status == 0",
                severity=AlertSeverity.WARNING,
                message="Redis连接失败",
                duration_seconds=30
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    async def initialize(self) -> None:
        """初始化告警管理器"""
        self._initialized = True
        logger.info("告警管理器初始化完成")
        
        # 启动评估循环
        asyncio.create_task(self._evaluation_loop())
    
    async def shutdown(self) -> None:
        """关闭告警管理器"""
        self._initialized = False
        self._active_alerts.clear()
        self._notifications.clear()
        logger.info("告警管理器已关闭")
    
    # ==================== 规则管理 ====================
    
    def add_rule(self, rule: AlertRule) -> None:
        """
        添加告警规则
        
        Args:
            rule: AlertRule实例
        """
        self._rules[rule.name] = rule
        logger.debug(f"已添加告警规则: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        移除告警规则
        
        Args:
            rule_name: 规则名称
            
        Returns:
            是否成功移除
        """
        if rule_name in self._rules:
            del self._rules[rule_name]
            return True
        return False
    
    def get_rule(self, rule_name: str) -> Optional[AlertRule]:
        """
        获取告警规则
        
        Args:
            rule_name: 规则名称
            
        Returns:
            AlertRule实例或None
        """
        return self._rules.get(rule_name)
    
    def list_rules(self) -> List[Dict[str, Any]]:
        """
        列出所有告警规则
        
        Returns:
            规则列表
        """
        return [
            {
                "name": rule.name,
                "condition": rule.condition,
                "severity": rule.severity.value,
                "message": rule.message,
                "enabled": rule.enabled,
                "duration_seconds": rule.duration_seconds
            }
            for rule in self._rules.values()
        ]
    
    def load_rules_from_file(self, file_path: str) -> None:
        """
        从文件加载告警规则
        
        Args:
            file_path: 规则文件路径（JSON或YAML）
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"告警规则文件不存在: {file_path}")
            return
        
        try:
            if path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
            else:
                import yaml
                with open(path, 'r', encoding='utf-8') as f:
                    rules_data = yaml.safe_load(f)
            
            for rule_data in rules_data:
                rule = AlertRule(
                    name=rule_data['name'],
                    condition=rule_data['condition'],
                    severity=AlertSeverity(rule_data.get('severity', 'warning')),
                    message=rule_data['message'],
                    duration_seconds=rule_data.get('duration_seconds', 60),
                    enabled=rule_data.get('enabled', True)
                )
                self.add_rule(rule)
            
            logger.info(f"从文件加载了 {len(rules_data)} 条告警规则")
            
        except Exception as e:
            logger.error(f"加载告警规则失败: {e}")
    
    # ==================== 告警评估 ====================
    
    async def check_all(self, metrics: Dict[str, Any]) -> List[Alert]:
        """
        检查所有规则
        
        Args:
            metrics: 性能指标字典
            
        Returns:
            触发的告警列表
        """
        triggered = []
        
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            
            alert = await self._evaluate_rule(rule, metrics)
            if alert:
                triggered.append(alert)
        
        return triggered
    
    async def _evaluate_rule(self, rule: AlertRule, metrics: Dict[str, Any]) -> Optional[Alert]:
        """
        评估单个规则
        
        Args:
            rule: 告警规则
            metrics: 指标数据
            
        Returns:
            Alert实例或None
        """
        # 获取指标值
        value = self._get_metric_value(metrics, rule.metric_name)
        
        if value is None:
            return None
        
        # 计算触发条件
        triggered = self._check_condition(value, rule.operator, rule.threshold)
        
        alert_key = rule.name
        
        if triggered:
            if alert_key not in self._active_alerts:
                # 新告警
                alert = Alert(
                    rule=rule,
                    status=AlertStatus.FIRING,
                    current_value=value,
                    started_at=datetime.utcnow(),
                    labels=rule.labels.copy(),
                    annotations=rule.annotations.copy()
                )
                self._active_alerts[alert_key] = alert
                
                # 发送通知
                await self._send_notification(alert)
                
                logger.warning(f"告警触发: {rule.name} (值: {value})")
                return alert
            else:
                # 更新现有告警
                alert = self._active_alerts[alert_key]
                alert.current_value = value
                alert.evaluation_count += 1
                
                # 检查是否应该升级或降级
                return None
        else:
            if alert_key in self._active_alerts:
                # 告警恢复
                alert = self._active_alerts[alert_key]
                alert.status = AlertStatus.RESOLVED
                alert.ended_at = datetime.utcnow()
                
                # 移除非活跃告警
                del self._active_alerts[alert_key]
                
                # 添加到历史
                self._add_to_history(alert)
                
                logger.info(f"告警恢复: {rule.name}")
                return None
        
        return None
    
    def _get_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """
        获取指标值
        
        Args:
            metrics: 指标字典
            metric_name: 指标名称
            
        Returns:
            指标值或None
        """
        # 直接查找
        if metric_name in metrics:
            value = metrics[metric_name]
            if isinstance(value, (int, float)):
                return float(value)
        
        # 嵌套查找
        for section in ['system', 'api', 'database']:
            if section in metrics and metric_name in metrics[section]:
                value = metrics[section][metric_name]
                if isinstance(value, (int, float)):
                    return float(value)
        
        return None
    
    def _check_condition(self, value: float, operator: str, threshold: float) -> bool:
        """
        检查条件
        
        Args:
            value: 当前值
            operator: 操作符
            threshold: 阈值
            
        Returns:
            是否触发
        """
        if operator == '>':
            return value > threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<':
            return value < threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold
        elif operator == '!=':
            return value != threshold
        else:
            logger.warning(f"未知的操作符: {operator}")
            return False
    
    async def _evaluation_loop(self) -> None:
        """告警评估循环"""
        while self._initialized:
            try:
                await asyncio.sleep(self._evaluation_interval)
                # 评估逻辑在check_all中调用
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"告警评估出错: {e}")
    
    # ==================== 通知管理 ====================
    
    async def _send_notification(self, alert: Alert) -> None:
        """
        发送告警通知
        
        Args:
            alert: 告警实例
        """
        notification = AlertNotification(
            alert=alert,
            severity=alert.rule.severity,
            message=alert.rule.message,
            channels=[]
        )
        
        # 获取告警级别对应的通知渠道
        severity_channels = self._notification_channels.get(
            alert.rule.severity.value, 
            self._notification_channels.get('default', [])
        )
        
        for channel in severity_channels:
            try:
                await self._send_to_channel(channel, alert)
                notification.channels.append(channel)
            except Exception as e:
                logger.error(f"发送通知到 {channel} 失败: {e}")
        
        notification.sent = len(notification.channels) > 0
        self._notifications.append(notification)
        
        # 调用回调
        for callback in self._on_alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
    
    async def _send_to_channel(self, channel: str, alert: Alert) -> None:
        """
        发送通知到指定渠道
        
        Args:
            channel: 渠道名称
            alert: 告警实例
        """
        if channel == 'log':
            logger.warning(f"[ALERT] {alert.rule.name}: {alert.rule.message}")
        elif channel == 'webhook':
            # Webhook通知（可扩展）
            pass
        # 可以添加更多渠道
    
    def on_alert(self, callback: Callable) -> None:
        """
        注册告警回调
        
        Args:
            callback: 回调函数
        """
        self._on_alert_callbacks.append(callback)
    
    # ==================== 历史管理 ====================
    
    def _add_to_history(self, alert: Alert) -> None:
        """添加到历史记录"""
        self._alert_history.append(alert)
        
        # 限制历史大小
        if len(self._alert_history) > self._max_history_size:
            self._alert_history = self._alert_history[-self._max_history_size:]
    
    def get_alert_history(self, 
                         since: Optional[datetime] = None,
                         status: Optional[AlertStatus] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取告警历史
        
        Args:
            since: 开始时间
            status: 告警状态
            limit: 最大数量
            
        Returns:
            告警历史列表
        """
        history = self._alert_history
        
        if since:
            history = [a for a in history if a.started_at >= since]
        
        if status:
            history = [a for a in history if a.status == status]
        
        return [
            {
                "name": alert.rule.name,
                "severity": alert.rule.severity.value,
                "status": alert.status.value,
                "message": alert.rule.message,
                "current_value": alert.current_value,
                "started_at": alert.started_at.isoformat(),
                "ended_at": alert.ended_at.isoformat() if alert.ended_at else None,
                "duration_seconds": (alert.ended_at - alert.started_at).total_seconds() if alert.ended_at else None
            }
            for alert in history[-limit:]
        ]
    
    # ==================== 状态查询 ====================
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        获取活跃告警
        
        Returns:
            活跃告警列表
        """
        return [
            {
                "name": alert.rule.name,
                "severity": alert.rule.severity.value,
                "status": alert.status.value,
                "message": alert.rule.message,
                "current_value": alert.current_value,
                "started_at": alert.started_at.isoformat(),
                "evaluation_count": alert.evaluation_count
            }
            for alert in self._active_alerts.values()
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取告警管理器状态
        
        Returns:
            状态信息
        """
        return {
            "initialized": self._initialized,
            "rules_count": len(self._rules),
            "active_alerts_count": len(self._active_alerts),
            "history_count": len(self._alert_history),
            "notifications_count": len(self._notifications),
            "rules": self.list_rules()
        }
    
    # ==================== 手动控制 ====================
    
    async def force_fire_alert(self, rule_name: str, message: str) -> Optional[Alert]:
        """
        手动触发告警
        
        Args:
            rule_name: 规则名称
            message: 告警消息
            
        Returns:
            Alert实例或None
        """
        rule = self._rules.get(rule_name)
        if not rule:
            return None
        
        alert = Alert(
            rule=rule,
            status=AlertStatus.FIRING,
            current_value=0,
            labels=rule.labels.copy(),
            annotations={"manual": True, "message": message}
        )
        
        self._active_alerts[rule_name] = alert
        await self._send_notification(alert)
        
        return alert
    
    async def resolve_alert(self, rule_name: str) -> bool:
        """
        手动解决告警
        
        Args:
            rule_name: 规则名称
            
        Returns:
            是否成功
        """
        if rule_name in self._active_alerts:
            alert = self._active_alerts[rule_name]
            alert.status = AlertStatus.RESOLVED
            alert.ended_at = datetime.utcnow()
            
            del self._active_alerts[rule_name]
            self._add_to_history(alert)
            
            return True
        
        return False
    
    def silence_alert(self, rule_name: str, duration_seconds: int = 3600) -> bool:
        """
        静默告警
        
        Args:
            rule_name: 规则名称
            duration_seconds: 静默时长
            
        Returns:
            是否成功
        """
        rule = self._rules.get(rule_name)
        if rule:
            rule.enabled = False
            return True
        return False
