"""
测试用例 - Self Healing Tests
自愈系统的单元测试和集成测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHealthChecker:
    """健康检查器测试"""
    
    def test_health_status_enum(self):
        """测试健康状态枚举"""
        from self_healing.health_checker import HealthStatus
        
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.DOWN.value == "down"
        assert HealthStatus.UNKNOWN.value == "unknown"
    
    def test_service_health_creation(self):
        """测试服务健康状态创建"""
        from self_healing.health_checker import ServiceHealth, HealthStatus
        
        health = ServiceHealth(
            service_name="test_service",
            status=HealthStatus.HEALTHY,
            message="Service is running",
            last_check=datetime.now(),
            response_time=100.0
        )
        
        assert health.service_name == "test_service"
        assert health.status == HealthStatus.HEALTHY
        assert health.response_time == 100.0
    
    def test_resource_health_creation(self):
        """测试资源健康状态创建"""
        from self_healing.health_checker import ResourceHealth, HealthStatus
        
        health = ResourceHealth(
            resource_type="cpu",
            status=HealthStatus.WARNING,
            usage=85.0,
            message="CPU usage is high",
            last_check=datetime.now()
        )
        
        assert health.resource_type == "cpu"
        assert health.usage == 85.0
    
    def test_health_report_creation(self):
        """测试健康报告创建"""
        from self_healing.health_checker import (
            HealthReport, HealthStatus, ServiceHealth, ResourceHealth
        )
        
        report = HealthReport(
            report_id="test_report",
            timestamp=datetime.now(),
            overall_status=HealthStatus.HEALTHY,
            services=[],
            dependencies=[],
            resources=[],
            summary={'total_services': 0},
            recommendations=[]
        )
        
        assert report.report_id == "test_report"
        assert report.overall_status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_health_checker_initialization(self):
        """测试健康检查器初始化"""
        from self_healing.health_checker import HealthChecker
        
        checker = HealthChecker()
        
        assert checker is not None
        assert len(checker.service_checks) > 0
        assert len(checker.resource_checks) > 0
    
    @pytest.mark.asyncio
    async def test_cpu_check(self):
        """测试CPU检查"""
        from self_healing.health_checker import HealthChecker
        
        checker = HealthChecker()
        result = checker._check_cpu()
        
        assert result.resource_type == "cpu"
        assert result.usage >= 0
        assert result.usage <= 100
    
    @pytest.mark.asyncio
    async def test_memory_check(self):
        """测试内存检查"""
        from self_healing.health_checker import HealthChecker
        
        checker = HealthChecker()
        result = checker._check_memory()
        
        assert result.resource_type == "memory"
        assert result.usage >= 0
        assert result.usage <= 100
    
    @pytest.mark.asyncio
    async def test_disk_check(self):
        """测试磁盘检查"""
        from self_healing.health_checker import HealthChecker
        
        checker = HealthChecker()
        result = checker._check_disk()
        
        assert result.resource_type == "disk"
        assert result.usage >= 0
        assert result.usage <= 100
    
    def test_overall_status_calculation(self):
        """测试总体状态计算"""
        from self_healing.health_checker import (
            HealthChecker, HealthStatus, ServiceHealth
        )
        
        checker = HealthChecker()
        
        # 所有健康
        services = [
            ServiceHealth(
                service_name="s1",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.now(),
                response_time=10.0
            ),
            ServiceHealth(
                service_name="s2",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.now(),
                response_time=10.0
            )
        ]
        
        status = checker._calculate_overall_status(services, [], [])
        assert status == HealthStatus.HEALTHY
        
        # 包含警告
        services.append(
            ServiceHealth(
                service_name="s3",
                status=HealthStatus.WARNING,
                message="Warning",
                last_check=datetime.now(),
                response_time=10.0
            )
        )
        
        status = checker._calculate_overall_status(services, [], [])
        assert status == HealthStatus.WARNING
        
        # 包含严重
        services.append(
            ServiceHealth(
                service_name="s4",
                status=HealthStatus.CRITICAL,
                message="Critical",
                last_check=datetime.now(),
                response_time=10.0
            )
        )
        
        status = checker._calculate_overall_status(services, [], [])
        assert status == HealthStatus.CRITICAL


class TestIncidentManager:
    """事件管理器测试"""
    
    def test_incident_severity_enum(self):
        """测试事件严重级别枚举"""
        from self_healing.incident_manager import IncidentSeverity
        
        assert IncidentSeverity.P1_CRITICAL.value == "P1_CRITICAL"
        assert IncidentSeverity.P2_HIGH.value == "P2_HIGH"
    
    def test_incident_status_enum(self):
        """测试事件状态枚举"""
        from self_healing.incident_manager import IncidentStatus
        
        assert IncidentStatus.DETECTED.value == "detected"
        assert IncidentStatus.INVESTIGATING.value == "investigating"
        assert IncidentStatus.RESOLVED.value == "resolved"
    
    def test_incident_creation(self):
        """测试事件创建"""
        from self_healing.incident_manager import (
            Incident, IncidentSeverity, IncidentStatus, IncidentCategory
        )
        
        incident = Incident(
            incident_id="test_001",
            title="Test Incident",
            description="This is a test",
            severity=IncidentSeverity.P2_HIGH,
            status=IncidentStatus.DETECTED,
            category=IncidentCategory.SERVICE_DOWN,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            detected_by="test"
        )
        
        assert incident.incident_id == "test_001"
        assert incident.severity == IncidentSeverity.P2_HIGH
        assert len(incident.timeline) == 0
        
        # 添加时间线事件
        incident.add_timeline_event('test', 'Test event')
        assert len(incident.timeline) == 1
    
    def test_incident_escalation(self):
        """测试事件升级"""
        from self_healing.incident_manager import (
            Incident, IncidentSeverity, IncidentStatus, IncidentCategory
        )
        
        incident = Incident(
            incident_id="test_002",
            title="Test Incident",
            description="This is a test",
            severity=IncidentSeverity.P2_HIGH,
            status=IncidentStatus.DETECTED,
            category=IncidentCategory.SERVICE_DOWN,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            detected_by="test"
        )
        
        assert incident.escalation_level == 0
        
        incident.escalate(1, "High priority")
        assert incident.escalation_level == 1
        assert len(incident.timeline) == 1
    
    def test_incident_manager_initialization(self):
        """测试事件管理器初始化"""
        from self_healing.incident_manager import IncidentManager
        
        manager = IncidentManager()
        
        assert manager is not None
        assert len(manager.incidents) == 0
        assert len(manager.active_incidents) == 0
        assert len(manager.escalation_policies) > 0
    
    def test_get_active_incidents(self):
        """测试获取活跃事件"""
        from self_healing.incident_manager import (
            IncidentManager, Incident, IncidentSeverity, IncidentStatus, IncidentCategory
        )
        
        manager = IncidentManager()
        
        # 创建测试事件
        for i in range(3):
            manager._create_incident(
                title=f"Test Incident {i}",
                description=f"Description {i}",
                severity=IncidentSeverity.P3_MEDIUM,
                category=IncidentCategory.PERFORMANCE_DEGRADATION,
                detected_by="test"
            )
        
        active = manager.get_active_incidents()
        assert len(active) == 3
    
    def test_update_incident_status(self):
        """测试更新事件状态"""
        from self_healing.incident_manager import (
            IncidentManager, Incident, IncidentSeverity, IncidentStatus, IncidentCategory
        )
        
        manager = IncidentManager()
        
        manager._create_incident(
            title="Test Incident",
            description="Test",
            severity=IncidentSeverity.P3_MEDIUM,
            category=IncidentCategory.PERFORMANCE_DEGRADATION,
            detected_by="test"
        )
        
        incident_id = list(manager.incidents.keys())[0]
        
        success = manager.update_incident_status(
            incident_id,
            IncidentStatus.INVESTIGATING,
            user="engineer"
        )
        
        assert success == True
        
        incident = manager.get_incident(incident_id)
        assert incident.status == IncidentStatus.INVESTIGATING
    
    def test_assign_incident(self):
        """测试分配事件"""
        from self_healing.incident_manager import (
            IncidentManager, Incident, IncidentSeverity, IncidentStatus, IncidentCategory
        )
        
        manager = IncidentManager()
        
        manager._create_incident(
            title="Test Incident",
            description="Test",
            severity=IncidentSeverity.P3_MEDIUM,
            category=IncidentCategory.PERFORMANCE_DEGRADATION,
            detected_by="test"
        )
        
        incident_id = list(manager.incidents.keys())[0]
        
        success = manager.assign_incident(incident_id, "on_call")
        assert success == True
        
        incident = manager.get_incident(incident_id)
        assert incident.assignee == "on_call"
    
    def test_resolve_incident(self):
        """测试解决事件"""
        from self_healing.incident_manager import (
            IncidentManager, Incident, IncidentSeverity, IncidentStatus, IncidentCategory
        )
        
        manager = IncidentManager()
        
        manager._create_incident(
            title="Test Incident",
            description="Test",
            severity=IncidentSeverity.P3_MEDIUM,
            category=IncidentCategory.PERFORMANCE_DEGRADATION,
            detected_by="test"
        )
        
        incident_id = list(manager.incidents.keys())[0]
        
        success = manager.resolve_incident(
            incident_id,
            resolution="Fixed by restarting service",
            root_cause="Memory leak"
        )
        
        assert success == True
        
        incident = manager.get_incident(incident_id)
        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolution == "Fixed by restarting service"
        assert incident.root_cause == "Memory leak"
    
    def test_incident_stats(self):
        """测试事件统计"""
        from self_healing.incident_manager import (
            IncidentManager, Incident, IncidentSeverity, IncidentStatus, IncidentCategory
        )
        
        manager = IncidentManager()
        
        # 创建活跃事件
        for i in range(3):
            manager._create_incident(
                title=f"Active Incident {i}",
                description="Test",
                severity=IncidentSeverity.P3_MEDIUM,
                category=IncidentCategory.PERFORMANCE_DEGRADATION,
                detected_by="test"
            )
        
        stats_before = manager.get_incident_stats()
        assert stats_before['total_active'] == 3
        
        # 解决一个事件
        incident_id = list(manager.incidents.keys())[0]
        manager.resolve_incident(incident_id, "Fixed")
        
        stats_after = manager.get_incident_stats()
        assert stats_after['total_active'] + stats_after['total_resolved'] == 3


class TestFixEngine:
    """修复引擎测试"""
    
    def test_fix_status_enum(self):
        """测试修复状态枚举"""
        from self_healing.fix_engine import FixStatus
        
        assert FixStatus.PENDING.value == "pending"
        assert FixStatus.IN_PROGRESS.value == "in_progress"
        assert FixStatus.SUCCESS.value == "success"
        assert FixStatus.FAILED.value == "failed"
    
    def test_fix_type_enum(self):
        """测试修复类型枚举"""
        from self_healing.fix_engine import FixType
        
        assert FixType.RESTART_SERVICE.value == "restart_service"
        assert FixType.CLEAR_CACHE.value == "clear_cache"
        assert FixType.SCALE_UP.value == "scale_up"
    
    def test_fix_action_creation(self):
        """测试修复动作创建"""
        from self_healing.fix_engine import FixAction, FixType, FixStatus
        
        action = FixAction(
            action_id="action_001",
            fix_type=FixType.RESTART_SERVICE,
            target="web_server"
        )
        
        assert action.action_id == "action_001"
        assert action.fix_type == FixType.RESTART_SERVICE
        assert action.status == FixStatus.PENDING
        
        # 测试开始
        action.start()
        assert action.status == FixStatus.IN_PROGRESS
        assert action.started_at is not None
        
        # 测试完成
        action.complete(True, result={'success': True})
        assert action.status == FixStatus.SUCCESS
        assert action.completed_at is not None
    
    def test_fix_record_creation(self):
        """测试修复记录创建"""
        from self_healing.fix_engine import (
            FixRecord, FixType, FixStatus, FixAction
        )
        
        action = FixAction(
            action_id="action_001",
            fix_type=FixType.RESTART_SERVICE,
            target="web_server"
        )
        action.complete(True)
        
        record = FixRecord(
            record_id="record_001",
            incident_id="incident_001",
            fix_type=FixType.RESTART_SERVICE,
            strategy="restart_service",
            status=FixStatus.SUCCESS,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            actions=[action],
            verification_result=None
        )
        
        assert record.record_id == "record_001"
        assert len(record.actions) == 1
        assert record.status == FixStatus.SUCCESS
    
    def test_fix_engine_initialization(self):
        """测试修复引擎初始化"""
        from self_healing.fix_engine import FixEngine
        from self_healing.config import Config
        
        config = Config()
        engine = FixEngine(config=config)
        
        assert engine is not None
        assert len(engine.strategy_to_fix_types) > 0


class TestRunbookAutomation:
    """手册自动化测试"""
    
    def test_step_status_enum(self):
        """测试步骤状态枚举"""
        from self_healing.runbook_automation import StepStatus
        
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.RUNNING.value == "running"
        assert StepStatus.SUCCESS.value == "success"
        assert StepStatus.FAILED.value == "failed"
    
    def test_step_type_enum(self):
        """测试步骤类型枚举"""
        from self_healing.runbook_automation import StepType
        
        assert StepType.COMMAND.value == "command"
        assert StepType.CHECK.value == "check"
        assert StepType.APPROVAL.value == "approval"
    
    def test_runbook_creation(self):
        """测试手册创建"""
        from self_healing.runbook_automation import Runbook, RunbookStep, StepType
        
        step = RunbookStep(
            step_id="step_1",
            name="Check Service",
            step_type=StepType.CHECK,
            description="Check if service is running",
            command="systemctl status {{service_name}}"
        )
        
        runbook = Runbook(
            runbook_id="test_runbook",
            name="Test Runbook",
            description="A test runbook",
            category="testing",
            steps=[step]
        )
        
        assert runbook.runbook_id == "test_runbook"
        assert len(runbook.steps) == 1
        
        # 测试获取步骤
        found_step = runbook.get_step("step_1")
        assert found_step is not None
        assert found_step.name == "Check Service"
    
    def test_step_execution_creation(self):
        """测试步骤执行创建"""
        from self_healing.runbook_automation import StepExecution, StepStatus
        
        execution = StepExecution(
            execution_id="exec_001",
            step_id="step_1",
            status=StepStatus.RUNNING,
            started_at=datetime.now(),
            completed_at=None,
            input_data={'command': 'test'},
            output_data=None,
            error=None,
            logs=[]
        )
        
        assert execution.execution_id == "exec_001"
        assert execution.status == StepStatus.RUNNING
        
        # 添加日志
        execution.add_log("Test log message")
        assert len(execution.logs) == 1
    
    def test_runbook_automation_initialization(self):
        """测试手册自动化初始化"""
        from self_healing.runbook_automation import RunbookAutomation
        
        automation = RunbookAutomation()
        
        assert automation is not None
        assert len(automation.runbooks) > 0
        assert len(automation.step_executors) > 0
    
    def test_default_runbooks(self):
        """测试默认手册"""
        from self_healing.runbook_automation import RunbookAutomation
        
        automation = RunbookAutomation()
        
        assert 'service_restart' in automation.runbooks
        assert 'memory_cleanup' in automation.runbooks
        assert 'disk_cleanup' in automation.runbooks
        assert 'database_recovery' in automation.runbooks
        
        # 检查手册结构
        service_rb = automation.runbooks['service_restart']
        assert len(service_rb.steps) > 0


class TestConfig:
    """配置测试"""
    
    def test_resource_thresholds_creation(self):
        """测试资源阈值创建"""
        from self_healing.config import ResourceThresholds
        
        thresholds = ResourceThresholds(
            cpu_warning=70.0,
            cpu_critical=90.0,
            memory_warning=75.0,
            memory_critical=90.0
        )
        
        assert thresholds.cpu_warning == 70.0
        assert thresholds.cpu_critical == 90.0
    
    def test_fix_strategy_creation(self):
        """测试修复策略创建"""
        from self_healing.config import FixStrategy
        
        strategy = FixStrategy(
            failure_type="service_down",
            auto_fix=True,
            max_attempts=3,
            strategy="restart_service"
        )
        
        assert strategy.failure_type == "service_down"
        assert strategy.auto_fix == True
        assert strategy.max_attempts == 3
    
    def test_config_initialization(self):
        """测试配置初始化"""
        from self_healing.config import Config
        
        config = Config()
        
        assert config is not None
        assert len(config.services) > 0
        assert len(config.fix_strategies) > 0
    
    def test_get_strategy(self):
        """测试获取策略"""
        from self_healing.config import Config
        
        config = Config()
        
        strategy = config.get_strategy('service_down')
        assert strategy is not None
        assert strategy.strategy == 'restart_service'
        
        strategy = config.get_strategy('memory_leak')
        assert strategy is not None
        assert strategy.strategy == 'clear_cache_restart'
    
    def test_get_service(self):
        """测试获取服务"""
        from self_healing.config import Config
        
        config = Config()
        
        service = config.get_service('web_server')
        assert service is not None
        assert service.name == 'web_server'
        
        service = config.get_service('nonexistent')
        assert service is None


class TestAPI:
    """API测试"""
    
    def test_api_response_creation(self):
        """测试API响应创建"""
        from self_healing.api import APIResponse
        
        response = APIResponse(
            success=True,
            message="Test successful",
            data={'key': 'value'}
        )
        
        assert response.success == True
        assert response.message == "Test successful"
        assert response.data['key'] == 'value'
    
    def test_api_initialization(self):
        """测试API初始化"""
        from self_healing.api import SelfHealingAPI
        
        api = SelfHealingAPI()
        
        assert api is not None
        assert api.health_checker is not None
        assert api.incident_manager is not None
        assert api.fix_engine is not None
        assert api.runbook_automation is not None


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
