"""
手册自动化 - Runbook Automation
运维手册、步骤执行、条件判断、人工审批
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

from .incident_manager import Incident, IncidentStatus


logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_CONDITION = "waiting_condition"


class StepType(Enum):
    """步骤类型"""
    COMMAND = "command"
    SCRIPT = "script"
    API_CALL = "api_call"
    CHECK = "check"
    APPROVAL = "approval"
    NOTIFICATION = "notification"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    LOOP = "loop"
    WAIT = "wait"


@dataclass
class StepCondition:
    """步骤条件"""
    field: str
    operator: str  # eq, ne, gt, lt, ge, le, contains, in
    value: Any
    next_step_on_true: Optional[str] = None
    next_step_on_false: Optional[str] = None


@dataclass
class RunbookStep:
    """运维手册步骤"""
    step_id: str
    name: str
    step_type: StepType
    description: str
    command: Optional[str] = None
    script: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_method: str = "GET"
    api_body: Optional[Dict] = None
    check_condition: Optional[StepCondition] = None
    approval_role: Optional[str] = None
    notification_channel: Optional[str] = None
    wait_seconds: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[str] = None
    timeout: int = 300  # 秒
    retry_count: int = 0
    on_failure: str = "stop"  # stop, continue, rollback
    next_steps: List[str] = field(default_factory=list)
    parallel_with: List[str] = field(default_factory=list)


@dataclass
class StepExecution:
    """步骤执行记录"""
    execution_id: str
    step_id: str
    status: StepStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    input_data: Dict[str, Any]
    output_data: Optional[Dict]
    error: Optional[str]
    logs: List[str]
    retry_attempt: int = 0
    
    def add_log(self, message: str):
        """添加日志"""
        self.logs.append(f"[{datetime.now().isoformat()}] {message}")
    
    def duration_seconds(self) -> float:
        """获取执行时长"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0


@dataclass
class RunbookExecution:
    """手册执行记录"""
    execution_id: str
    runbook_id: str
    incident_id: Optional[str]
    status: StepStatus
    started_at: datetime
    completed_at: Optional[datetime]
    current_step: Optional[str]
    step_executions: Dict[str, StepExecution]
    variables: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def add_step_execution(self, execution: StepExecution):
        """添加步骤执行记录"""
        self.step_executions[execution.step_id] = execution
    
    def get_step_result(self, step_id: str) -> Optional[StepExecution]:
        """获取步骤结果"""
        return self.step_executions.get(step_id)
    
    def set_variable(self, name: str, value: Any):
        """设置变量"""
        self.variables[name] = value
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """获取变量"""
        return self.variables.get(name, default)


@dataclass
class Runbook:
    """运维手册"""
    runbook_id: str
    name: str
    description: str
    category: str
    steps: List[RunbookStep]
    prerequisites: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    is_active: bool = True
    
    def get_step(self, step_id: str) -> Optional[RunbookStep]:
        """获取步骤"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_next_steps(self, current_step_id: str) -> List[RunbookStep]:
        """获取下一步骤"""
        current = self.get_step(current_step_id)
        if not current:
            return []
        
        next_steps = []
        for step_id in current.next_steps:
            step = self.get_step(step_id)
            if step:
                next_steps.append(step)
        return next_steps


class RunbookAutomation:
    """手册自动化引擎"""
    
    def __init__(self):
        self.runbooks: Dict[str, Runbook] = {}
        self.executions: Dict[str, RunbookExecution] = {}
        self.step_executors: Dict[StepType, Callable] = {}
        self.approval_callbacks: List[Callable] = {}
        self._setup_default_runbooks()
        self._setup_default_executors()
    
    def _setup_default_runbooks(self):
        """设置默认手册"""
        self.runbooks = {
            'service_restart': self._create_service_restart_runbook(),
            'memory_cleanup': self._create_memory_cleanup_runbook(),
            'disk_cleanup': self._create_disk_cleanup_runbook(),
            'database_recovery': self._create_database_recovery_runbook()
        }
    
    def _create_service_restart_runbook(self) -> Runbook:
        """创建服务重启手册"""
        return Runbook(
            runbook_id='service_restart',
            name='Service Restart Runbook',
            description='Standard procedure for restarting a failed service',
            category='service_recovery',
            steps=[
                RunbookStep(
                    step_id='step_1',
                    name='Check Service Status',
                    step_type=StepType.CHECK,
                    description='Verify current service status',
                    command='systemctl status {{service_name}}',
                    expected_output='active/inactive',
                    on_failure='continue'
                ),
                RunbookStep(
                    step_id='step_2',
                    name='Stop Service',
                    step_type=StepType.COMMAND,
                    description='Stop the service gracefully',
                    command='systemctl stop {{service_name}}',
                    on_failure='continue'
                ),
                RunbookStep(
                    step_id='step_3',
                    name='Wait for Graceful Shutdown',
                    step_type=StepType.WAIT,
                    description='Wait for service to stop completely',
                    wait_seconds=10,
                    on_failure='continue'
                ),
                RunbookStep(
                    step_id='step_4',
                    name='Start Service',
                    step_type=StepType.COMMAND,
                    description='Start the service',
                    command='systemctl start {{service_name}}',
                    on_failure='rollback'
                ),
                RunbookStep(
                    step_id='step_5',
                    name='Verify Service Health',
                    step_type=StepType.CHECK,
                    description='Verify service is healthy',
                    command='systemctl status {{service_name}}',
                    expected_output='active (running)',
                    timeout=60,
                    on_failure='rollback'
                )
            ],
            variables={'service_name': ''}
        )
    
    def _create_memory_cleanup_runbook(self) -> Runbook:
        """创建内存清理手册"""
        return Runbook(
            runbook_id='memory_cleanup',
            name='Memory Cleanup Runbook',
            description='Procedure for cleaning up memory leaks',
            category='memory_issue',
            steps=[
                RunbookStep(
                    step_id='step_1',
                    name='Check Memory Usage',
                    step_type=StepType.CHECK,
                    description='Check current memory usage',
                    command='free -m',
                    on_failure='stop'
                ),
                RunbookStep(
                    step_id='step_2',
                    name='Drop Caches',
                    step_type=StepType.COMMAND,
                    description='Drop system caches',
                    command='sync; echo 3 > /proc/sys/vm/drop_caches',
                    on_failure='continue'
                ),
                RunbookStep(
                    step_id='step_3',
                    name='Restart Cache Service',
                    step_type=StepType.COMMAND,
                    description='Restart the cache service',
                    command='systemctl restart redis',
                    on_failure='rollback'
                ),
                RunbookStep(
                    step_id='step_4',
                    name='Verify Memory',
                    step_type=StepType.CHECK,
                    description='Verify memory has been freed',
                    command='free -m',
                    on_failure='continue'
                )
            ]
        )
    
    def _create_disk_cleanup_runbook(self) -> Runbook:
        """创建磁盘清理手册"""
        return Runbook(
            runbook_id='disk_cleanup',
            name='Disk Cleanup Runbook',
            description='Procedure for cleaning up disk space',
            category='disk_issue',
            steps=[
                RunbookStep(
                    step_id='step_1',
                    name='Check Disk Usage',
                    step_type=StepType.CHECK,
                    description='Check current disk usage',
                    command='df -h',
                    on_failure='stop'
                ),
                RunbookStep(
                    step_id='step_2',
                    name='Find Large Files',
                    step_type=StepType.COMMAND,
                    description='Find large log files',
                    command='find /var/log -name "*.log" -size +100M',
                    on_failure='continue'
                ),
                RunbookStep(
                    step_id='step_3',
                    name='Archive Old Logs',
                    step_type=StepType.COMMAND,
                    description='Archive logs older than 7 days',
                    command='find /var/log -name "*.log" -mtime +7 -exec gzip {} \\;',
                    on_failure='continue'
                ),
                RunbookStep(
                    step_id='step_4',
                    name='Clear Temp Files',
                    step_type=StepType.COMMAND,
                    description='Clear temporary files',
                    command='rm -rf /tmp/*',
                    on_failure='continue'
                ),
                RunbookStep(
                    step_id='step_5',
                    name='Verify Disk Space',
                    step_type=StepType.CHECK,
                    description='Verify disk space has been freed',
                    command='df -h',
                    expected_output='usage_below_80%',
                    on_failure='continue'
                )
            ]
        )
    
    def _create_database_recovery_runbook(self) -> Runbook:
        """创建数据库恢复手册"""
        return Runbook(
            runbook_id='database_recovery',
            name='Database Recovery Runbook',
            description='Procedure for database recovery',
            category='database_issue',
            steps=[
                RunbookStep(
                    step_id='step_1',
                    name='Check Database Status',
                    step_type=StepType.CHECK,
                    description='Check if database is running',
                    command='pg_isready -h localhost',
                    on_failure='continue'
                ),
                RunbookStep(
                    step_id='step_2',
                    name='Check Connection Pool',
                    step_type=StepType.CHECK,
                    description='Check active connections',
                    command='SELECT count(*) FROM pg_stat_activity',
                    on_failure='continue'
                ),
                RunbookStep(
                    step_id='step_3',
                    name='Terminate Idle Connections',
                    step_type=StepType.COMMAND,
                    description='Terminate long-running idle connections',
                    command="SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND query_start < NOW() - INTERVAL '10 minutes'",
                    approval_role='dba',
                    on_failure='continue'
                ),
                RunbookStep(
                    step_id='step_4',
                    name='Restart Database',
                    step_type=StepType.COMMAND,
                    description='Restart the database service',
                    command='systemctl restart postgresql',
                    approval_role='dba',
                    on_failure='rollback'
                ),
                RunbookStep(
                    step_id='step_5',
                    name='Verify Database',
                    step_type=StepType.CHECK,
                    description='Verify database is accepting connections',
                    command='pg_isready -h localhost',
                    expected_output='accepting connections',
                    timeout=60,
                    on_failure='rollback'
                )
            ],
            prerequisites=['backup_created']
        )
    
    def _setup_default_executors(self):
        """设置默认执行器"""
        self.step_executors[StepType.COMMAND] = self._execute_command
        self.step_executors[StepType.CHECK] = self._execute_check
        self.step_executors[StepType.NOTIFICATION] = self._execute_notification
        self.step_executors[StepType.WAIT] = self._execute_wait
        self.step_executors[StepType.APPROVAL] = self._execute_approval
        self.step_executors[StepType.API_CALL] = self._execute_api_call
    
    def register_runbook(self, runbook: Runbook):
        """注册手册"""
        self.runbooks[runbook.runbook_id] = runbook
    
    def register_step_executor(self, step_type: StepType, executor: Callable):
        """注册步骤执行器"""
        self.step_executors[step_type] = executor
    
    async def execute_runbook(
        self,
        runbook_id: str,
        incident_id: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        start_step: Optional[str] = None
    ) -> RunbookExecution:
        """执行手册"""
        runbook = self.runbooks.get(runbook_id)
        if not runbook:
            raise ValueError(f"Runbook {runbook_id} not found")
        
        execution_id = str(uuid.uuid4())[:8]
        
        execution = RunbookExecution(
            execution_id=execution_id,
            runbook_id=runbook_id,
            incident_id=incident_id,
            status=StepStatus.RUNNING,
            started_at=datetime.now(),
            completed_at=None,
            current_step=start_step or (runbook.steps[0].step_id if runbook.steps else None),
            step_executions={},
            variables=variables or dict(runbook.variables),
            metadata={}
        )
        
        self.executions[execution_id] = execution
        
        # 执行步骤
        await self._execute_steps(execution, runbook)
        
        return execution
    
    async def _execute_steps(
        self, execution: RunbookExecution, runbook: Runbook
    ):
        """执行所有步骤"""
        current_step_id = execution.current_step
        
        while current_step_id and execution.status == StepStatus.RUNNING:
            step = runbook.get_step(current_step_id)
            if not step:
                logger.error(f"Step {current_step_id} not found in runbook")
                break
            
            # 执行步骤
            result = await self._execute_step(execution, step, runbook)
            
            # 更新执行状态
            execution.current_step = current_step_id
            
            if result.status == StepStatus.SUCCESS:
                # 找到下一个步骤
                if step.next_steps:
                    current_step_id = step.next_steps[0]
                elif step.step_id == runbook.steps[-1].step_id:
                    execution.status = StepStatus.SUCCESS
                    execution.completed_at = datetime.now()
                else:
                    current_index = next(
                        (i for i, s in enumerate(runbook.steps) if s.step_id == step.step_id),
                        -1
                    )
                    if current_index < len(runbook.steps) - 1:
                        current_step_id = runbook.steps[current_index + 1].step_id
                    else:
                        execution.status = StepStatus.SUCCESS
                        execution.completed_at = datetime.now()
            elif result.status == StepStatus.FAILED:
                if step.on_failure == 'stop':
                    execution.status = StepStatus.FAILED
                    execution.completed_at = datetime.now()
                elif step.on_failure == 'continue':
                    current_index = next(
                        (i for i, s in enumerate(runbook.steps) if s.step_id == step.step_id),
                        -1
                    )
                    if current_index < len(runbook.steps) - 1:
                        current_step_id = runbook.steps[current_index + 1].step_id
            elif result.status == StepStatus.WAITING_APPROVAL:
                execution.status = StepStatus.WAITING_APPROVAL
                return
    
    async def _execute_step(
        self,
        execution: RunbookExecution,
        step: RunbookStep,
        runbook: Runbook
    ) -> StepExecution:
        """执行单个步骤"""
        execution_record = StepExecution(
            execution_id=str(uuid.uuid4())[:8],
            step_id=step.step_id,
            status=StepStatus.RUNNING,
            started_at=datetime.now(),
            completed_at=None,
            input_data={'command': step.command, 'parameters': step.parameters},
            output_data=None,
            error=None,
            logs=[]
        )
        
        # 替换变量
        command = self._render_template(step.command or '', execution.variables)
        
        # 获取执行器
        executor = self.step_executors.get(step.step_type)
        if not executor:
            execution_record.status = StepStatus.FAILED
            execution_record.error = f"No executor for step type {step.step_type}"
            return execution_record
        
        try:
            output = await executor(command, step.parameters, execution)
            execution_record.output_data = output
            execution_record.status = StepStatus.SUCCESS
            
        except Exception as e:
            execution_record.error = str(e)
            execution_record.status = StepStatus.FAILED
        
        execution_record.completed_at = datetime.now()
        execution.add_step_execution(execution_record)
        
        return execution_record
    
    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """渲染模板"""
        import re
        result = template
        for key, value in variables.items():
            result = result.replace(f'{{{{{key}}}}}', str(value))
        return result
    
    async def _execute_command(
        self, command: str, params: Dict, execution: RunbookExecution
    ) -> Dict:
        """执行命令"""
        import subprocess
        execution_record.add_log(f"Executing command: {command}")
        
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            return {
                'return_code': proc.returncode,
                'stdout': stdout.decode(),
                'stderr': stderr.decode()
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def _execute_check(
        self, command: str, params: Dict, execution: RunbookExecution
    ) -> Dict:
        """执行检查"""
        result = await self._execute_command(command, params, execution)
        execution_record.add_log(f"Check result: {result}")
        return result
    
    async def _execute_notification(
        self, message: str, params: Dict, execution: RunbookExecution
    ) -> Dict:
        """执行通知"""
        channel = params.get('notification_channel', 'default')
        execution_record.add_log(f"Sending notification to {channel}")
        return {'sent': True, 'channel': channel}
    
    async def _execute_wait(
        self, command: str, params: Dict, execution: RunbookExecution
    ) -> Dict:
        """执行等待"""
        wait_time = params.get('wait_seconds', 5)
        execution_record.add_log(f"Waiting for {wait_time} seconds")
        await asyncio.sleep(wait_time)
        return {'waited': wait_time}
    
    async def _execute_approval(
        self, command: str, params: Dict, execution: RunbookExecution
    ) -> Dict:
        """执行审批"""
        role = params.get('approval_role', 'admin')
        execution_record.add_log(f"Waiting for approval from {role}")
        return {'waiting': True, 'role': role}
    
    async def _execute_api_call(
        self, endpoint: str, params: Dict, execution: RunbookExecution
    ) -> Dict:
        """执行API调用"""
        import aiohttp
        method = params.get('api_method', 'GET')
        body = params.get('api_body')
        
        async with aiohttp.ClientSession() as session:
            if method == 'GET':
                async with session.get(endpoint) as response:
                    return {'status': response.status, 'body': await response.json()}
            elif method == 'POST':
                async with session.post(endpoint, json=body) as response:
                    return {'status': response.status, 'body': await response.json()}
        return {'error': 'Unknown method'}
    
    def get_runbook(self, runbook_id: str) -> Optional[Runbook]:
        """获取手册"""
        return self.runbooks.get(runbook_id)
    
    def get_all_runbooks(self) -> List[Runbook]:
        """获取所有手册"""
        return list(self.runbooks.values())
    
    def get_execution(self, execution_id: str) -> Optional[RunbookExecution]:
        """获取执行记录"""
        return self.executions.get(execution_id)
    
    def get_execution_history(
        self, runbook_id: Optional[str] = None
    ) -> List[RunbookExecution]:
        """获取执行历史"""
        executions = list(self.executions.values())
        if runbook_id:
            executions = [e for e in executions if e.runbook_id == runbook_id]
        return executions
    
    def approve_step(
        self,
        execution_id: str,
        step_id: str,
        approved: bool,
        approver: str
    ) -> bool:
        """审批步骤"""
        execution = self.executions.get(execution_id)
        if not execution:
            return False
        
        step_execution = execution.get_step_result(step_id)
        if not step_execution:
            return False
        
        if approved:
            step_execution.status = StepStatus.SUCCESS
            step_execution.output_data = {'approved': True, 'approver': approver}
        else:
            step_execution.status = StepStatus.FAILED
            step_execution.output_data = {'approved': False, 'approver': approver}
        
        return True
    
    def get_runbook_stats(self) -> Dict[str, Any]:
        """获取手册统计"""
        total = len(self.executions)
        successful = sum(1 for e in self.executions.values() if e.status == StepStatus.SUCCESS)
        failed = sum(1 for e in self.executions.values() if e.status == StepStatus.FAILED)
        
        return {
            'total_executions': total,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total * 100) if total > 0 else 0
        }


# 创建全局手册自动化实例
_runbook_automation: Optional[RunbookAutomation] = None


def get_runbook_automation() -> RunbookAutomation:
    """获取全局手册自动化实例"""
    global _runbook_automation
    if _runbook_automation is None:
        _runbook_automation = RunbookAutomation()
    return _runbook_automation
