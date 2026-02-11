"""
事件管理器 - Incident Manager
事件检测、事件分类、事件升级、事件响应
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

from .health_checker import HealthReport, HealthStatus


logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """事件严重级别"""
    P1_CRITICAL = "P1_CRITICAL"  # 影响核心业务，需要立即处理
    P2_HIGH = "P2_HIGH"          # 影响较大，需要尽快处理
    P3_MEDIUM = "P3_MEDIUM"      # 有影响，但可以稍后处理
    P4_LOW = "P4_LOW"            # 影响较小，可以计划处理
    P5_INFO = "P5_INFO"          # 信息级别，无需紧急处理


class IncidentStatus(Enum):
    """事件状态"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"


class IncidentCategory(Enum):
    """事件类别"""
    SERVICE_DOWN = "service_down"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    SECURITY_INCIDENT = "security_incident"
    CONFIGURATION_ERROR = "configuration_error"
    CAPACITY_ISSUE = "capacity_issue"


@dataclass
class Incident:
    """事件"""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    category: IncidentCategory
    created_at: datetime
    updated_at: datetime
    detected_by: str  # 检测源
    affected_services: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    assignee: Optional[str] = None
    escalation_level: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Dict] = field(default_factory=list)
    related_incidents: List[str] = field(default_factory=list)
    
    def add_timeline_event(
        self, event_type: str, message: str, user: Optional[str] = None
    ):
        """添加时间线事件"""
        self.timeline.append({
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'message': message,
            'user': user
        })
        self.updated_at = datetime.now()
    
    def escalate(self, new_level: int, reason: str):
        """升级事件"""
        self.escalation_level = new_level
        self.add_timeline_event('escalation', f"Escalated to level {new_level}: {reason}")


@dataclass
class EscalationPolicy:
    """升级策略"""
    level: int
    after_minutes: int
    notify_roles: List[str]
    auto_actions: List[str] = field(default_factory=list)
    require_approval: bool = False


class IncidentDetector:
    """事件检测器"""
    
    def __init__(self):
        self.detection_rules: Dict[str, Callable] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.cooldown_period = 300  # 5分钟冷却
    
    def register_detection_rule(
        self, rule_name: str, rule_func: Callable[[HealthReport], Optional[Incident]]
    ):
        """注册检测规则"""
        self.detection_rules[rule_name] = rule_func
    
    async def detect_from_report(
        self, report: HealthReport
    ) -> List[Incident]:
        """从健康报告检测事件"""
        incidents = []
        
        for rule_name, rule_func in self.detection_rules.items():
            # 检查冷却期
            if rule_name in self.alert_cooldowns:
                if datetime.now() < self.alert_cooldowns[rule_name]:
                    continue
            
            try:
                if asyncio.iscoroutinefunction(rule_func):
                    incident = await rule_func(report)
                else:
                    incident = rule_func(report)
                
                if incident:
                    incidents.append(incident)
                    # 设置冷却期
                    self.alert_cooldowns[rule_name] = datetime.now() + timedelta(
                        seconds=self.cooldown_period
                    )
            except Exception as e:
                logger.error(f"Detection rule {rule_name} failed: {e}")
        
        return incidents
    
    def clear_cooldown(self, rule_name: str):
        """清除冷却期"""
        self.alert_cooldowns.pop(rule_name, None)


class IncidentClassifier:
    """事件分类器"""
    
    # 分类规则映射
    CLASSIFICATION_RULES = {
        HealthStatus.DOWN: {
            'category': IncidentCategory.SERVICE_DOWN,
            'severity': IncidentSeverity.P1_CRITICAL
        },
        HealthStatus.CRITICAL: {
            'category': IncidentCategory.RESOURCE_EXHAUSTION,
            'severity': IncidentSeverity.P2_HIGH
        },
        HealthStatus.WARNING: {
            'category': IncidentCategory.PERFORMANCE_DEGRADATION,
            'severity': IncidentSeverity.P3_MEDIUM
        }
    }
    
    def classify(
        self,
        source: str,
        status: HealthStatus,
        message: str,
        **kwargs
    ) -> tuple[IncidentCategory, IncidentSeverity]:
        """分类事件"""
        rule = self.CLASSIFICATION_RULES.get(status, {
            'category': IncidentCategory.PERFORMANCE_DEGRADATION,
            'severity': IncidentSeverity.P4_LOW
        })
        
        return rule['category'], rule['severity']
    
    def calculate_severity(
        self,
        affected_users: int,
        revenue_impact: bool,
        is_core_service: bool
    ) -> IncidentSeverity:
        """计算严重级别"""
        if is_core_service and revenue_impact:
            return IncidentSeverity.P1_CRITICAL
        elif affected_users > 1000:
            return IncidentSeverity.P2_HIGH
        elif affected_users > 100:
            return IncidentSeverity.P3_MEDIUM
        else:
            return IncidentSeverity.P4_LOW


class IncidentManager:
    """事件管理器"""
    
    def __init__(self, config=None):
        self.config = config
        self.incidents: Dict[str, Incident] = {}
        self.active_incidents: Dict[str, Incident] = {}
        self.resolved_incidents: Dict[str, Incident] = {}
        self.detector = IncidentDetector()
        self.classifier = IncidentClassifier()
        self.escalation_policies: List[EscalationPolicy] = []
        self.notification_callbacks: List[Callable] = []
        self._setup_default_escalation()
        self._setup_default_detection_rules()
    
    def _setup_default_escalation(self):
        """设置默认升级策略"""
        self.escalation_policies = [
            EscalationPolicy(
                level=0,
                after_minutes=0,
                notify_roles=['on_call']
            ),
            EscalationPolicy(
                level=1,
                after_minutes=15,
                notify_roles=['on_call', 'team_lead']
            ),
            EscalationPolicy(
                level=2,
                after_minutes=30,
                notify_roles=['on_call', 'team_lead', 'manager']
            ),
            EscalationPolicy(
                level=3,
                after_minutes=60,
                notify_roles=['on_call', 'team_lead', 'manager', 'director']
            )
        ]
    
    def _setup_default_detection_rules(self):
        """设置默认检测规则"""
        self.detector.register_detection_rule(
            'service_down', self._detect_service_down
        )
        self.detector.register_detection_rule(
            'resource_critical', self._detect_resource_critical
        )
        self.detector.register_detection_rule(
            'performance_degradation', self._detect_performance_issue
        )
        self.detector.register_detection_rule(
            'dependency_failure', self._detect_dependency_failure
        )
    
    def _detect_service_down(
        self, report: HealthReport
    ) -> Optional[Incident]:
        """检测服务宕机"""
        for service in report.services:
            if service.status == HealthStatus.DOWN:
                category, severity = self.classifier.classify(
                    service.service_name, service.status, service.message
                )
                return self._create_incident(
                    title=f"Service Down: {service.service_name}",
                    description=service.message,
                    severity=severity,
                    category=category,
                    detected_by='health_checker',
                    affected_services=[service.service_name]
                )
        return None
    
    def _detect_resource_critical(
        self, report: HealthReport
    ) -> Optional[Incident]:
        """检测资源临界"""
        critical_resources = [
            r for r in report.resources
            if r.status == HealthStatus.CRITICAL
        ]
        
        if critical_resources:
            resource = critical_resources[0]
            category, severity = self.classifier.classify(
                resource.resource_type, resource.status, resource.message
            )
            return self._create_incident(
                title=f"Critical Resource: {resource.resource_type}",
                description=resource.message,
                severity=severity,
                category=category,
                detected_by='health_checker',
                affected_components=[resource.resource_type]
            )
        return None
    
    def _detect_performance_issue(
        self, report: HealthReport
    ) -> Optional[Incident]:
        """检测性能问题"""
        warning_resources = [
            r for r in report.resources
            if r.status == HealthStatus.WARNING
        ]
        
        if warning_resources:
            resource = warning_resources[0]
            category, severity = self.classifier.classify(
                resource.resource_type, resource.status, resource.message
            )
            return self._create_incident(
                title=f"Performance Issue: {resource.resource_type}",
                description=resource.message,
                severity=severity,
                category=IncidentCategory.PERFORMANCE_DEGRADATION,
                detected_by='health_checker',
                affected_components=[resource.resource_type]
            )
        return None
    
    def _detect_dependency_failure(
        self, report: HealthReport
    ) -> Optional[Incident]:
        """检测依赖故障"""
        failed_deps = [
            d for d in report.dependencies
            if d.status in [HealthStatus.DOWN, HealthStatus.CRITICAL]
        ]
        
        if failed_deps:
            dep = failed_deps[0]
            return self._create_incident(
                title=f"Dependency Failure: {dep.dependency_name}",
                description=dep.message,
                severity=IncidentSeverity.P2_HIGH,
                category=IncidentCategory.DEPENDENCY_FAILURE,
                detected_by='health_checker',
                affected_components=[dep.dependency_name]
            )
        return None
    
    def _create_incident(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        category: IncidentCategory,
        detected_by: str,
        affected_services: List[str] = None,
        affected_components: List[str] = None
    ) -> Incident:
        """创建事件"""
        incident_id = str(uuid.uuid4())[:8]
        
        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.DETECTED,
            category=category,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            detected_by=detected_by,
            affected_services=affected_services or [],
            affected_components=affected_components or []
        )
        
        incident.add_timeline_event('created', f"Incident detected by {detected_by}")
        
        self.incidents[incident_id] = incident
        self.active_incidents[incident_id] = incident
        
        return incident
    
    async def process_health_report(self, report: HealthReport) -> List[Incident]:
        """处理健康报告"""
        incidents = await self.detector.detect_from_report(report)
        
        for incident in incidents:
            self._notify_incident(incident)
        
        return incidents
    
    def _notify_incident(self, incident: Incident):
        """通知事件"""
        for callback in self.notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(incident))
                else:
                    callback(incident)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")
    
    def register_notification_callback(self, callback: Callable):
        """注册通知回调"""
        self.notification_callbacks.append(callback)
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """获取事件"""
        return self.incidents.get(incident_id)
    
    def get_active_incidents(self) -> List[Incident]:
        """获取活跃事件"""
        return list(self.active_incidents.values())
    
    def get_resolved_incidents(
        self, since: Optional[datetime] = None
    ) -> List[Incident]:
        """获取已解决事件"""
        incidents = list(self.resolved_incidents.values())
        if since:
            incidents = [i for i in incidents if i.updated_at >= since]
        return incidents
    
    def update_incident_status(
        self,
        incident_id: str,
        status: IncidentStatus,
        user: Optional[str] = None,
        message: Optional[str] = None
    ) -> bool:
        """更新事件状态"""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False
        
        old_status = incident.status
        incident.status = status
        incident.add_timeline_event(
            'status_change',
            f"Status changed from {old_status} to {status}",
            user=user
        )
        
        if message:
            incident.add_timeline_event('comment', message, user=user)
        
        # 更新索引
        if status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
            self.active_incidents.pop(incident_id, None)
            self.resolved_incidents[incident_id] = incident
        elif status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
            self.active_incidents[incident_id] = incident
        
        return True
    
    def assign_incident(
        self,
        incident_id: str,
        assignee: str,
        user: Optional[str] = None
    ) -> bool:
        """分配事件"""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False
        
        incident.assignee = assignee
        incident.add_timeline_event('assignment', f"Assigned to {assignee}", user=user)
        return True
    
    def escalate_incident(
        self,
        incident_id: str,
        reason: str,
        user: Optional[str] = None
    ) -> bool:
        """升级事件"""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False
        
        new_level = incident.escalation_level + 1
        incident.escalate(new_level, reason)
        
        # 触发升级策略
        policy = self._get_escalation_policy(new_level)
        if policy:
            self._notify_escalation(incident, policy)
        
        return True
    
    def _get_escalation_policy(self, level: int) -> Optional[EscalationPolicy]:
        """获取升级策略"""
        for policy in self.escalation_policies:
            if policy.level == level:
                return policy
        return None
    
    def _notify_escalation(
        self, incident: Incident, policy: EscalationPolicy
    ):
        """通知升级"""
        message = (
            f"INCIDENT ESCALATED: {incident.incident_id}\n"
            f"Title: {incident.title}\n"
            f"Level: {policy.level}\n"
            f"Notify: {', '.join(policy.notify_roles)}"
        )
        
        for role in policy.notify_roles:
            self._send_notification(role, message)
    
    def _send_notification(self, role: str, message: str):
        """发送通知"""
        # 实际实现应该根据角色选择不同的通知渠道
        logger.info(f"[{role}] Notification: {message}")
    
    def resolve_incident(
        self,
        incident_id: str,
        resolution: str,
        root_cause: Optional[str] = None,
        user: Optional[str] = None
    ) -> bool:
        """解决事件"""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False
        
        incident.resolution = resolution
        incident.root_cause = root_cause
        incident.add_timeline_event('resolution', resolution, user=user)
        
        if root_cause:
            incident.add_timeline_event('root_cause', root_cause, user=user)
        
        return self.update_incident_status(
            incident_id, IncidentStatus.RESOLVED, user=user
        )
    
    def get_incident_stats(self) -> Dict[str, Any]:
        """获取事件统计"""
        active_by_severity = {}
        for incident in self.active_incidents.values():
            severity = incident.severity.value
            active_by_severity[severity] = active_by_severity.get(severity, 0) + 1
        
        return {
            'total_active': len(self.active_incidents),
            'total_resolved': len(self.resolved_incidents),
            'total': len(self.incidents),
            'active_by_severity': active_by_severity,
            'avg_resolution_time_minutes': self._calculate_avg_resolution_time()
        }
    
    def _calculate_avg_resolution_time(self) -> float:
        """计算平均解决时间"""
        resolved = list(self.resolved_incidents.values())
        if not resolved:
            return 0
        
        total_time = sum(
            (i.updated_at - i.created_at).total_seconds()
            for i in resolved
        )
        return total_time / len(resolved) / 60  # 转换为分钟


# 创建全局事件管理器实例
_incident_manager: Optional[IncidentManager] = None


def get_incident_manager() -> IncidentManager:
    """获取全局事件管理器"""
    global _incident_manager
    if _incident_manager is None:
        _incident_manager = IncidentManager()
    return _incident_manager
