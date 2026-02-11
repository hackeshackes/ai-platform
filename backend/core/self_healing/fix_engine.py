"""
修复引擎 - Fix Engine
策略匹配、自动修复、修复验证、修复记录
"""

import asyncio
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from .config import Config, FixStrategy
from .incident_manager import Incident, IncidentSeverity, IncidentManager


logger = logging.getLogger(__name__)


class FixStatus(Enum):
    """修复状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    REQUIRES_APPROVAL = "requires_approval"
    SKIPPED = "skipped"


class FixType(Enum):
    """修复类型"""
    RESTART_SERVICE = "restart_service"
    CLEAR_CACHE = "clear_cache"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    RATE_LIMIT = "rate_limit"
    SWITCH_ROUTE = "switch_route"
    CLEANUP_LOGS = "cleanup_logs"
    EXPAND_DISK = "expand_disk"
    OPTIMIZE_INDEX = "optimize_index"
    RESTART_INSTANCE = "restart_instance"
    ROLLBACK = "rollback"


@dataclass
class FixAction:
    """修复动作"""
    action_id: str
    fix_type: FixType
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: FixStatus = FixStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    rollback_action: Optional[str] = None
    
    def start(self):
        """开始执行"""
        self.status = FixStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def complete(self, success: bool, result: Dict = None, error: str = None):
        """完成执行"""
        self.completed_at = datetime.now()
        if success:
            self.status = FixStatus.SUCCESS
        else:
            self.status = FixStatus.FAILED
        self.result = result
        self.error = error
    
    def duration_seconds(self) -> float:
        """获取执行时长"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0


@dataclass
class FixRecord:
    """修复记录"""
    record_id: str
    incident_id: str
    fix_type: FixType
    strategy: str
    status: FixStatus
    started_at: datetime
    completed_at: Optional[datetime]
    actions: List[FixAction]
    verification_result: Optional[Dict]
    error_count: int = 0
    rollback_performed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """验证结果"""
    success: bool
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class FixExecutor:
    """修复执行器"""
    
    def __init__(self):
        self.executors: Dict[FixType, Callable] = {}
        self.verifiers: Dict[str, Callable] = {}
        self._setup_default_executors()
    
    def _setup_default_executors(self):
        """设置默认执行器"""
        self.executors[FixType.RESTART_SERVICE] = self._restart_service
        self.executors[FixType.CLEAR_CACHE] = self._clear_cache
        self.executors[FixType.SCALE_UP] = self._scale_up
        self.executors[FixType.SCALE_DOWN] = self._scale_down
        self.executors[FixType.RATE_LIMIT] = self._apply_rate_limit
        self.executors[FixType.SWITCH_ROUTE] = self._switch_route
        self.executors[FixType.CLEANUP_LOGS] = self._cleanup_logs
        self.executors[FixType.EXPAND_DISK] = self._expand_disk
        self.executors[FixType.OPTIMIZE_INDEX] = self._optimize_index
        self.executors[FixType.RESTART_INSTANCE] = self._restart_instance
        self.executors[FixType.ROLLBACK] = self._rollback
    
    def register_executor(self, fix_type: FixType, executor_func: Callable):
        """注册执行器"""
        self.executors[fix_type] = executor_func
    
    def register_verifier(self, verifier_name: str, verifier_func: Callable):
        """注册验证器"""
        self.verifiers[verifier_name] = verifier_func
    
    async def execute(self, action: FixAction) -> FixAction:
        """执行修复动作"""
        executor = self.executors.get(action.fix_type)
        if not executor:
            action.complete(False, error=f"No executor for {action.fix_type}")
            return action
        
        action.start()
        logger.info(f"Executing fix action: {action.fix_type} on {action.target}")
        
        try:
            if asyncio.iscoroutinefunction(executor):
                result = await executor(action.target, action.parameters)
            else:
                result = executor(action.target, action.parameters)
            
            action.complete(True, result=result)
            logger.info(f"Fix action completed successfully: {action.action_id}")
            
        except Exception as e:
            action.complete(False, error=str(e))
            logger.error(f"Fix action failed: {action.action_id}, error: {e}")
        
        return action
    
    async def _restart_service(self, target: str, params: Dict) -> Dict:
        """重启服务"""
        # 模拟重启服务
        await asyncio.sleep(1)
        return {'action': 'restart', 'target': target, 'success': True}
    
    async def _clear_cache(self, target: str, params: Dict) -> Dict:
        """清理缓存"""
        cache_pattern = params.get('pattern', '*')
        await asyncio.sleep(0.5)
        return {'action': 'clear_cache', 'target': target, 'pattern': cache_pattern}
    
    async def _scale_up(self, target: str, params: Dict) -> Dict:
        """扩容"""
        replicas = params.get('replicas', 2)
        await asyncio.sleep(1)
        return {'action': 'scale_up', 'target': target, 'replicas': replicas}
    
    async def _scale_down(self, target: str, params: Dict) -> Dict:
        """缩容"""
        replicas = params.get('replicas', 1)
        await asyncio.sleep(1)
        return {'action': 'scale_down', 'target': target, 'replicas': replicas}
    
    async def _apply_rate_limit(self, target: str, params: Dict) -> Dict:
        """应用限流"""
        rate = params.get('rate', '1000/min')
        await asyncio.sleep(0.5)
        return {'action': 'rate_limit', 'target': target, 'rate': rate}
    
    async def _switch_route(self, target: str, params: Dict) -> Dict:
        """切换路由"""
        route = params.get('route', 'backup')
        await asyncio.sleep(1)
        return {'action': 'switch_route', 'target': target, 'route': route}
    
    async def _cleanup_logs(self, target: str, params: Dict) -> Dict:
        """清理日志"""
        max_age = params.get('max_age_days', 7)
        await asyncio.sleep(1)
        return {'action': 'cleanup_logs', 'target': target, 'max_age_days': max_age}
    
    async def _expand_disk(self, target: str, params: Dict) -> Dict:
        """扩容磁盘"""
        size_gb = params.get('size_gb', 100)
        await asyncio.sleep(2)
        return {'action': 'expand_disk', 'target': target, 'size_gb': size_gb}
    
    async def _optimize_index(self, target: str, params: Dict) -> Dict:
        """优化索引"""
        index_name = params.get('index_name', 'default')
        await asyncio.sleep(1)
        return {'action': 'optimize_index', 'target': target, 'index_name': index_name}
    
    async def _restart_instance(self, target: str, params: Dict) -> Dict:
        """重启实例"""
        await asyncio.sleep(2)
        return {'action': 'restart_instance', 'target': target}
    
    async def _rollback(self, target: str, params: Dict) -> Dict:
        """回滚"""
        version = params.get('version', 'previous')
        await asyncio.sleep(2)
        return {'action': 'rollback', 'target': target, 'version': version}


class FixEngine:
    """修复引擎"""
    
    def __init__(self, config: Config = None, incident_manager: IncidentManager = None):
        self.config = config
        self.incident_manager = incident_manager
        self.executor = FixExecutor()
        self.fix_records: Dict[str, FixRecord] = {}
        self.pending_fixes: Dict[str, FixRecord] = {}
        self.cooldowns: Dict[str, datetime] = {}
        self.max_retry_count = 3
        self._setup_strategy_fix_mapping()
    
    def _setup_strategy_fix_mapping(self):
        """设置策略到修复类型的映射"""
        self.strategy_to_fix_types = {
            'restart_service': [FixType.RESTART_SERVICE],
            'clear_cache_restart': [FixType.CLEAR_CACHE, FixType.RESTART_SERVICE],
            'scale_up_rate_limit': [FixType.SCALE_UP, FixType.RATE_LIMIT],
            'switch_network_route': [FixType.SWITCH_ROUTE],
            'cleanup_logs_expand_disk': [FixType.CLEANUP_LOGS, FixType.EXPAND_DISK],
            'optimize_index_rate_limit': [FixType.OPTIMIZE_INDEX, FixType.RATE_LIMIT]
        }
    
    async def process_incident(self, incident: Incident) -> FixRecord:
        """处理事件"""
        # 检查是否在冷却期
        if self._is_in_cooldown(incident.category.value):
            logger.info(f"Incident {incident.incident_id} is in cooldown, skipping")
            return self._create_skipped_record(incident)
        
        # 获取修复策略
        strategy = self.config.get_strategy(incident.category.value) if self.config else None
        
        if not strategy:
            logger.warning(f"No strategy found for {incident.category.value}")
            return self._create_skipped_record(incident)
        
        # 检查是否需要审批
        if strategy.requires_approval:
            return await self._request_approval(incident, strategy)
        
        # 检查是否自动修复
        if not strategy.auto_fix:
            logger.info(f"Auto-fix disabled for {incident.category.value}")
            return self._create_skipped_record(incident)
        
        # 执行修复
        return await self._execute_fix(incident, strategy)
    
    def _is_in_cooldown(self, failure_type: str) -> bool:
        """检查是否在冷却期"""
        if failure_type in self.cooldowns:
            if datetime.now() < self.cooldowns[failure_type]:
                return True
        return False
    
    def _set_cooldown(self, failure_type: str, cooldown_period: int):
        """设置冷却期"""
        self.cooldowns[failure_type] = datetime.now() + timedelta(seconds=cooldown_period)
    
    async def _execute_fix(
        self, incident: Incident, strategy: FixStrategy
    ) -> FixRecord:
        """执行修复"""
        record_id = str(uuid.uuid4())[:8]
        fix_types = self.strategy_to_fix_types.get(strategy.strategy, [])
        
        if not fix_types:
            fix_types = [FixType.RESTART_SERVICE]  # 默认修复
        
        record = FixRecord(
            record_id=record_id,
            incident_id=incident.incident_id,
            fix_type=fix_types[0],
            strategy=strategy.strategy,
            status=FixStatus.IN_PROGRESS,
            started_at=datetime.now(),
            completed_at=None,
            actions=[],
            verification_result=None
        )
        
        self.fix_records[record_id] = record
        
        # 逐个执行修复动作
        for fix_type in fix_types:
            action = FixAction(
                action_id=str(uuid.uuid4())[:8],
                fix_type=fix_type,
                target=self._get_target(incident)
            )
            
            record.actions.append(action)
            await self.executor.execute(action)
            
            # 如果动作失败，检查是否需要回滚
            if action.status == FixStatus.FAILED:
                if strategy.rollback_strategy:
                    await self._rollback_fix(record, strategy.rollback_strategy)
                break
        
        # 检查是否所有动作都成功
        all_success = all(
            a.status == FixStatus.SUCCESS for a in record.actions
        )
        
        if all_success:
            record.status = FixStatus.SUCCESS
            # 设置冷却期
            self._set_cooldown(incident.category.value, strategy.cooldown_period)
        else:
            record.status = FixStatus.FAILED
            record.error_count = len(
                [a for a in record.actions if a.status == FixStatus.FAILED]
            )
        
        record.completed_at = datetime.now()
        
        # 验证修复结果
        record.verification_result = await self._verify_fix(incident, record)
        
        # 更新事件状态
        if self.incident_manager:
            if record.verification_result.success:
                self.incident_manager.resolve_incident(
                    incident.incident_id,
                    f"Auto-fixed using {strategy.strategy}"
                )
            else:
                self.incident_manager.escalate_incident(
                    incident.incident_id,
                    f"Auto-fix failed: {record.verification_result.message}"
                )
        
        return record
    
    def _get_target(self, incident: Incident) -> str:
        """获取修复目标"""
        if incident.affected_services:
            return incident.affected_services[0]
        elif incident.affected_components:
            return incident.affected_components[0]
        return 'unknown'
    
    async def _verify_fix(
        self, incident: Incident, record: FixRecord
    ) -> VerificationResult:
        """验证修复"""
        # 等待一下让系统恢复
        await asyncio.sleep(2)
        
        # 检查事件是否已解决
        if self.incident_manager:
            current_incident = self.incident_manager.get_incident(incident.incident_id)
            if current_incident and current_incident.status.value == 'resolved':
                return VerificationResult(
                    success=True,
                    message="Incident is resolved",
                    metrics={'incident_status': 'resolved'}
                )
        
        return VerificationResult(
            success=record.status == FixStatus.SUCCESS,
            message="Fix completed",
            metrics={'actions_completed': len(record.actions)}
        )
    
    async def _request_approval(
        self, incident: Incident, strategy: FixStrategy
    ) -> FixRecord:
        """请求审批"""
        record = FixRecord(
            record_id=str(uuid.uuid4())[:8],
            incident_id=incident.incident_id,
            fix_type=FixType.RESTART_SERVICE,
            strategy=strategy.strategy,
            status=FixStatus.REQUIRES_APPROVAL,
            started_at=datetime.now(),
            completed_at=None,
            actions=[],
            verification_result=None
        )
        
        self.fix_records[record_id] = record
        return record
    
    async def _rollback_fix(
        self, record: FixRecord, rollback_strategy: str
    ) -> bool:
        """回滚修复"""
        rollback_action = FixAction(
            action_id=str(uuid.uuid4())[:8],
            fix_type=FixType.ROLLBACK,
            target='system'
        )
        
        await self.executor.execute(rollback_action)
        
        if rollback_action.status == FixStatus.SUCCESS:
            record.status = FixStatus.ROLLED_BACK
            record.rollback_performed = True
            return True
        
        return False
    
    def _create_skipped_record(self, incident: Incident) -> FixRecord:
        """创建跳过的记录"""
        return FixRecord(
            record_id=str(uuid.uuid4())[:8],
            incident_id=incident.incident_id,
            fix_type=FixType.RESTART_SERVICE,
            strategy='skipped',
            status=FixStatus.SKIPPED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            actions=[],
            verification_result=VerificationResult(
                success=True,
                message="Skipped due to cooldown or policy"
            )
        )
    
    def get_fix_record(self, record_id: str) -> Optional[FixRecord]:
        """获取修复记录"""
        return self.fix_records.get(record_id)
    
    def get_fix_history(
        self, since: Optional[datetime] = None
    ) -> List[FixRecord]:
        """获取修复历史"""
        records = list(self.fix_records.values())
        if since:
            records = [r for r in records if r.started_at >= since]
        return records
    
    def get_fix_stats(self) -> Dict[str, Any]:
        """获取修复统计"""
        total = len(self.fix_records)
        successful = sum(
            1 for r in self.fix_records.values()
            if r.status == FixStatus.SUCCESS
        )
        failed = sum(
            1 for r in self.fix_records.values()
            if r.status == FixStatus.FAILED
        )
        
        return {
            'total_fixes': total,
            'successful': successful,
            'failed': failed,
            'skipped': sum(
                1 for r in self.fix_records.values()
                if r.status == FixStatus.SKIPPED
            ),
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'avg_fix_time_seconds': self._calculate_avg_fix_time()
        }
    
    def _calculate_avg_fix_time(self) -> float:
        """计算平均修复时间"""
        completed = [
            r for r in self.fix_records.values()
            if r.completed_at and r.status in [FixStatus.SUCCESS, FixStatus.FAILED]
        ]
        
        if not completed:
            return 0
        
        total_time = sum(
            (r.completed_at - r.started_at).total_seconds()
            for r in completed
        )
        return total_time / len(completed)
    
    def clear_cooldown(self, failure_type: str):
        """清除冷却期"""
        self.cooldowns.pop(failure_type, None)


# 创建全局修复引擎实例
_fix_engine: Optional[FixEngine] = None


def get_fix_engine() -> FixEngine:
    """获取全局修复引擎"""
    global _fix_engine
    if _fix_engine is None:
        from .config import get_config
        from .incident_manager import get_incident_manager
        
        _fix_engine = FixEngine(
            config=get_config(),
            incident_manager=get_incident_manager()
        )
    return _fix_engine
