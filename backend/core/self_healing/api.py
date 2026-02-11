"""
API接口 - Self Healing API
提供REST API接口供外部调用
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from .health_checker import get_health_checker, run_health_check, HealthReport
from .incident_manager import get_incident_manager, Incident, IncidentStatus
from .fix_engine import get_fix_engine, FixRecord
from .runbook_automation import get_runbook_automation, Runbook, RunbookExecution


logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """API响应"""
    success: bool
    message: str
    data: Optional[Dict] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'success': self.success,
            'message': self.message,
            'data': self.data,
            'error': self.error
        }


class SelfHealingAPI:
    """自愈系统API"""
    
    def __init__(self):
        self.health_checker = get_health_checker()
        self.incident_manager = get_incident_manager()
        self.fix_engine = get_fix_engine()
        self.runbook_automation = get_runbook_automation()
    
    # ============ 健康检查相关API ============
    
    async def health_check_all(self) -> APIResponse:
        """执行所有健康检查"""
        try:
            report = await run_health_check()
            return APIResponse(
                success=True,
                message="Health check completed",
                data=self._format_health_report(report)
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return APIResponse(
                success=False,
                message="Health check failed",
                error=str(e)
            )
    
    def get_health_status(self) -> APIResponse:
        """获取健康状态"""
        report = self.health_checker.get_latest_report()
        if report:
            return APIResponse(
                success=True,
                message="Health status retrieved",
                data=self._format_health_report(report)
            )
        else:
            return APIResponse(
                success=False,
                message="No health report available"
            )
    
    def get_health_history(self, since: Optional[str] = None) -> APIResponse:
        """获取健康历史"""
        since_dt = None
        if since:
            since_dt = datetime.fromisoformat(since)
        
        history = self.health_checker.get_health_history(since_dt)
        return APIResponse(
            success=True,
            message="Health history retrieved",
            data={
                'reports': [self._format_health_report(r) for r in history]
            }
        )
    
    def _format_health_report(self, report: HealthReport) -> Dict:
        """格式化健康报告"""
        return {
            'report_id': report.report_id,
            'timestamp': report.timestamp.isoformat(),
            'overall_status': report.overall_status.value,
            'summary': report.summary,
            'recommendations': report.recommendations,
            'services': [
                {
                    'name': s.service_name,
                    'status': s.status.value,
                    'message': s.message,
                    'response_time_ms': s.response_time,
                    'details': s.details
                }
                for s in report.services
            ],
            'dependencies': [
                {
                    'name': d.dependency_name,
                    'type': d.dependency_type,
                    'status': d.status.value,
                    'message': d.message,
                    'response_time_ms': d.response_time
                }
                for d in report.dependencies
            ],
            'resources': [
                {
                    'type': r.resource_type,
                    'status': r.status.value,
                    'usage_percent': r.usage,
                    'message': r.message,
                    'details': r.details
                }
                for r in report.resources
            ]
        }
    
    # ============ 事件管理相关API ============
    
    def get_active_incidents(self) -> APIResponse:
        """获取活跃事件"""
        incidents = self.incident_manager.get_active_incidents()
        return APIResponse(
            success=True,
            message="Active incidents retrieved",
            data={
                'incidents': [self._format_incident(i) for i in incidents],
                'count': len(incidents)
            }
        )
    
    def get_incident(self, incident_id: str) -> APIResponse:
        """获取事件详情"""
        incident = self.incident_manager.get_incident(incident_id)
        if incident:
            return APIResponse(
                success=True,
                message="Incident retrieved",
                data=self._format_incident(incident)
            )
        else:
            return APIResponse(
                success=False,
                message="Incident not found"
            )
    
    def update_incident_status(
        self,
        incident_id: str,
        status: str,
        user: Optional[str] = None,
        message: Optional[str] = None
    ) -> APIResponse:
        """更新事件状态"""
        try:
            new_status = IncidentStatus(status)
        except ValueError:
            return APIResponse(
                success=False,
                message=f"Invalid status: {status}"
            )
        
        success = self.incident_manager.update_incident_status(
            incident_id, new_status, user=user, message=message
        )
        
        if success:
            return APIResponse(
                success=True,
                message=f"Incident {incident_id} status updated"
            )
        else:
            return APIResponse(
                success=False,
                message="Failed to update incident"
            )
    
    def assign_incident(
        self,
        incident_id: str,
        assignee: str,
        user: Optional[str] = None
    ) -> APIResponse:
        """分配事件"""
        success = self.incident_manager.assign_incident(incident_id, assignee, user)
        if success:
            return APIResponse(
                success=True,
                message=f"Incident {incident_id} assigned to {assignee}"
            )
        else:
            return APIResponse(
                success=False,
                message="Failed to assign incident"
            )
    
    def escalate_incident(
        self,
        incident_id: str,
        reason: str,
        user: Optional[str] = None
    ) -> APIResponse:
        """升级事件"""
        success = self.incident_manager.escalate_incident(incident_id, reason, user)
        if success:
            return APIResponse(
                success=True,
                message=f"Incident {incident_id} escalated"
            )
        else:
            return APIResponse(
                success=False,
                message="Failed to escalate incident"
            )
    
    def resolve_incident(
        self,
        incident_id: str,
        resolution: str,
        root_cause: Optional[str] = None,
        user: Optional[str] = None
    ) -> APIResponse:
        """解决事件"""
        success = self.incident_manager.resolve_incident(
            incident_id, resolution, root_cause, user
        )
        if success:
            return APIResponse(
                success=True,
                message=f"Incident {incident_id} resolved"
            )
        else:
            return APIResponse(
                success=False,
                message="Failed to resolve incident"
            )
    
    def get_incident_stats(self) -> APIResponse:
        """获取事件统计"""
        stats = self.incident_manager.get_incident_stats()
        return APIResponse(
            success=True,
            message="Incident statistics retrieved",
            data=stats
        )
    
    def _format_incident(self, incident: Incident) -> Dict:
        """格式化事件"""
        return {
            'incident_id': incident.incident_id,
            'title': incident.title,
            'description': incident.description,
            'severity': incident.severity.value,
            'status': incident.status.value,
            'category': incident.category.value,
            'created_at': incident.created_at.isoformat(),
            'updated_at': incident.updated_at.isoformat(),
            'detected_by': incident.detected_by,
            'affected_services': incident.affected_services,
            'affected_components': incident.affected_components,
            'escalation_level': incident.escalation_level,
            'assignee': incident.assignee,
            'resolution': incident.resolution,
            'root_cause': incident.root_cause,
            'timeline': incident.timeline,
            'metadata': incident.metadata
        }
    
    # ============ 修复引擎相关API ============
    
    async def trigger_fix(self, incident_id: str) -> APIResponse:
        """触发修复"""
        incident = self.incident_manager.get_incident(incident_id)
        if not incident:
            return APIResponse(
                success=False,
                message=f"Incident {incident_id} not found"
            )
        
        try:
            record = await self.fix_engine.process_incident(incident)
            return APIResponse(
                success=True,
                message=f"Fix triggered for incident {incident_id}",
                data=self._format_fix_record(record)
            )
        except Exception as e:
            logger.error(f"Fix trigger failed: {e}")
            return APIResponse(
                success=False,
                message="Failed to trigger fix",
                error=str(e)
            )
    
    def get_fix_record(self, record_id: str) -> APIResponse:
        """获取修复记录"""
        record = self.fix_engine.get_fix_record(record_id)
        if record:
            return APIResponse(
                success=True,
                message="Fix record retrieved",
                data=self._format_fix_record(record)
            )
        else:
            return APIResponse(
                success=False,
                message="Fix record not found"
            )
    
    def get_fix_history(self, since: Optional[str] = None) -> APIResponse:
        """获取修复历史"""
        since_dt = None
        if since:
            since_dt = datetime.fromisoformat(since)
        
        history = self.fix_engine.get_fix_history(since_dt)
        return APIResponse(
            success=True,
            message="Fix history retrieved",
            data={
                'records': [self._format_fix_record(r) for r in history]
            }
        )
    
    def get_fix_stats(self) -> APIResponse:
        """获取修复统计"""
        stats = self.fix_engine.get_fix_stats()
        return APIResponse(
            success=True,
            message="Fix statistics retrieved",
            data=stats
        )
    
    def _format_fix_record(self, record: FixRecord) -> Dict:
        """格式化修复记录"""
        return {
            'record_id': record.record_id,
            'incident_id': record.incident_id,
            'fix_type': record.fix_type.value,
            'strategy': record.strategy,
            'status': record.status.value,
            'started_at': record.started_at.isoformat(),
            'completed_at': record.completed_at.isoformat() if record.completed_at else None,
            'actions': [
                {
                    'action_id': a.action_id,
                    'type': a.fix_type.value,
                    'target': a.target,
                    'status': a.status.value,
                    'duration_seconds': a.duration_seconds(),
                    'error': a.error
                }
                for a in record.actions
            ],
            'verification_result': record.verification_result,
            'error_count': record.error_count,
            'rollback_performed': record.rollback_performed
        }
    
    # ============ 手册自动化相关API ============
    
    def get_runbooks(self) -> APIResponse:
        """获取所有手册"""
        runbooks = self.runbook_automation.get_all_runbooks()
        return APIResponse(
            success=True,
            message="Runbooks retrieved",
            data={
                'runbooks': [
                    {
                        'runbook_id': rb.runbook_id,
                        'name': rb.name,
                        'description': rb.description,
                        'category': rb.category,
                        'version': rb.version,
                        'steps_count': len(rb.steps)
                    }
                    for rb in runbooks
                ]
            }
        )
    
    def get_runbook(self, runbook_id: str) -> APIResponse:
        """获取手册详情"""
        runbook = self.runbook_automation.get_runbook(runbook_id)
        if runbook:
            return APIResponse(
                success=True,
                message="Runbook retrieved",
                data={
                    'runbook_id': runbook.runbook_id,
                    'name': runbook.name,
                    'description': runbook.description,
                    'category': runbook.category,
                    'version': runbook.version,
                    'steps': [
                        {
                            'step_id': s.step_id,
                            'name': s.name,
                            'type': s.step_type.value,
                            'description': s.description
                        }
                        for s in runbook.steps
                    ]
                }
            )
        else:
            return APIResponse(
                success=False,
                message="Runbook not found"
            )
    
    async def execute_runbook(
        self,
        runbook_id: str,
        incident_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """执行手册"""
        try:
            execution = await self.runbook_automation.execute_runbook(
                runbook_id,
                incident_id=incident_id,
                variables=variables
            )
            return APIResponse(
                success=True,
                message=f"Runbook {runbook_id} execution started",
                data=self._format_execution(execution)
            )
        except Exception as e:
            logger.error(f"Runbook execution failed: {e}")
            return APIResponse(
                success=False,
                message="Failed to execute runbook",
                error=str(e)
            )
    
    def get_execution(self, execution_id: str) -> APIResponse:
        """获取执行记录"""
        execution = self.runbook_automation.get_execution(execution_id)
        if execution:
            return APIResponse(
                success=True,
                message="Execution retrieved",
                data=self._format_execution(execution)
            )
        else:
            return APIResponse(
                success=False,
                message="Execution not found"
            )
    
    def get_execution_history(
        self, runbook_id: Optional[str] = None
    ) -> APIResponse:
        """获取执行历史"""
        history = self.runbook_automation.get_execution_history(runbook_id)
        return APIResponse(
            success=True,
            message="Execution history retrieved",
            data={
                'executions': [self._format_execution(e) for e in history]
            }
        )
    
    def approve_runbook_step(
        self,
        execution_id: str,
        step_id: str,
        approved: bool,
        approver: str
    ) -> APIResponse:
        """审批手册步骤"""
        success = self.runbook_automation.approve_step(
            execution_id, step_id, approved, approver
        )
        if success:
            return APIResponse(
                success=True,
                message=f"Step {step_id} approval processed"
            )
        else:
            return APIResponse(
                success=False,
                message="Failed to process approval"
            )
    
    def get_runbook_stats(self) -> APIResponse:
        """获取手册统计"""
        stats = self.runbook_automation.get_runbook_stats()
        return APIResponse(
            success=True,
            message="Runbook statistics retrieved",
            data=stats
        )
    
    def _format_execution(self, execution: RunbookExecution) -> Dict:
        """格式化执行记录"""
        return {
            'execution_id': execution.execution_id,
            'runbook_id': execution.runbook_id,
            'incident_id': execution.incident_id,
            'status': execution.status.value,
            'started_at': execution.started_at.isoformat(),
            'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
            'current_step': execution.current_step,
            'variables': execution.variables,
            'steps_completed': len(execution.step_executions)
        }
    
    # ============ 综合API ============
    
    async def full_diagnosis(self) -> APIResponse:
        """完整诊断"""
        results = {
            'health': None,
            'incidents': [],
            'stats': {}
        }
        
        # 执行健康检查
        health_report = await run_health_check()
        results['health'] = self._format_health_report(health_report)
        
        # 处理检测到的事件
        incidents = await self.incident_manager.process_health_report(health_report)
        results['incidents'] = [self._format_incident(i) for i in incidents]
        
        # 自动修复
        for incident in incidents:
            fix_record = await self.fix_engine.process_incident(incident)
            results.setdefault('fixes', []).append(self._format_fix_record(fix_record))
        
        # 统计
        results['stats'] = {
            'health_status': health_report.overall_status.value,
            'incidents_detected': len(incidents),
            'fixes_applied': len(results.get('fixes', []))
        }
        
        return APIResponse(
            success=True,
            message="Full diagnosis completed",
            data=results
        )
    
    def get_dashboard(self) -> APIResponse:
        """获取仪表盘数据"""
        return APIResponse(
            success=True,
            message="Dashboard data retrieved",
            data={
                'health': self.health_checker.get_latest_report().__dict__ if self.health_checker.get_latest_report() else None,
                'incidents': self.incident_manager.get_incident_stats(),
                'fixes': self.fix_engine.get_fix_stats(),
                'runbooks': self.runbook_automation.get_runbook_stats()
            }
        )


# 创建全局API实例
_api: Optional[SelfHealingAPI] = None


def get_api() -> SelfHealingAPI:
    """获取全局API实例"""
    global _api
    if _api is None:
        _api = SelfHealingAPI()
    return _api
