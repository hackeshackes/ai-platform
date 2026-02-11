"""
安全审计模块
记录和查询安全相关事件
"""

import json
import os
import threading
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AuditAction(str, Enum):
    """审计动作类型"""
    # 数据操作
    DATA_CREATE = "data_create"
    DATA_READ = "data_read"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    
    # 认证
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGE = "password_change"
    
    # 权限
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_REVOKE = "permission_revoke"
    ROLE_ASSIGN = "role_assign"
    ROLE_REMOVE = "role_remove"
    
    # 系统操作
    CONFIG_CHANGE = "config_change"
    BACKUP = "backup"
    RESTORE = "restore"
    
    # 安全事件
    SECURITY_ALERT = "security_alert"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class AuditResult(str, Enum):
    """审计结果"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class AuditEvent:
    """审计事件模型"""

    def __init__(
        self,
        user_id: str,
        action: str,
        resource: str,
        resource_type: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        result: bool = True,
        result_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        self.event_id = self._generate_event_id()
        self.user_id = user_id
        self.action = action
        self.resource = resource
        self.resource_type = resource_type
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.result = result
        self.result_code = result_code
        self.details = details or {}
        self.timestamp = timestamp or datetime.utcnow()
        self.request_id = request_id
        self.session_id = session_id

    def _generate_event_id(self) -> str:
        """生成唯一事件ID"""
        return f"audit_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{id(self)}"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "resource_type": self.resource_type,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "result": self.result,
            "result_code": self.result_code,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "request_id": self.request_id,
            "session_id": self.session_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """从字典创建"""
        event = cls(
            user_id=data.get("user_id", ""),
            action=data.get("action", ""),
            resource=data.get("resource", ""),
            resource_type=data.get("resource_type"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            result=data.get("result", True),
            result_code=data.get("result_code"),
            details=data.get("details"),
            request_id=data.get("request_id"),
            session_id=data.get("session_id")
        )
        if "timestamp" in data and data["timestamp"]:
            if isinstance(data["timestamp"], str):
                event.timestamp = datetime.fromisoformat(data["timestamp"])
            else:
                event.timestamp = data["timestamp"]
        if "event_id" in data:
            event.event_id = data["event_id"]
        return event


class AuditLogger:
    """审计日志记录器"""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        async_mode: bool = True
    ):
        self.storage_path = storage_path or "/tmp/audit_logs"
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.async_mode = async_mode

        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()
        self._current_file_size = 0
        self._file_index = 0

        self._ensure_storage_dir()
        self._init_current_file()

    def _ensure_storage_dir(self) -> None:
        """确保存储目录存在"""
        os.makedirs(self.storage_path, exist_ok=True)

    def _get_current_file_path(self) -> str:
        """获取当前日志文件路径"""
        return os.path.join(
            self.storage_path,
            f"audit_{self._file_index:03d}.jsonl"
        )

    def _init_current_file(self) -> None:
        """初始化当前日志文件"""
        self._current_file_path = self._get_current_file_path()
        if os.path.exists(self._current_file_path):
            self._current_file_size = os.path.getsize(self._current_file_path)
        else:
            self._current_file_size = 0

    def _rotate_file_if_needed(self) -> None:
        """必要时轮转日志文件"""
        if self._current_file_size >= self.max_file_size:
            self._file_index += 1
            self._init_current_file()

    def log(self, event: AuditEvent) -> str:
        """
        记录审计事件

        Args:
            event: 审计事件

        Returns:
            事件ID
        """
        if self.async_mode:
            with self._lock:
                self._events.append(event)
                self._process_events()
        else:
            self._write_event(event)

        return event.event_id

    def _process_events(self) -> None:
        """处理待记录的事件"""
        while self._events:
            event = self._events.pop(0)
            self._write_event(event)

    def _write_event(self, event: AuditEvent) -> None:
        """写入事件到文件"""
        try:
            self._rotate_file_if_needed()

            with self._lock:
                with open(self._current_file_path, "a", encoding="utf-8") as f:
                    line = json.dumps(event.to_dict(), ensure_ascii=False)
                    f.write(line + "\n")
                    self._current_file_size += len(line) + 1
        except Exception:
            pass  # 静默处理写入错误

    def log_event(
        self,
        user_id: str,
        action: str,
        resource: str,
        resource_type: Optional[str] = None,
        ip_address: Optional[str] = None,
        result: bool = True,
        result_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """便捷的事件记录方法"""
        event = AuditEvent(
            user_id=user_id,
            action=action,
            resource=resource,
            resource_type=resource_type,
            ip_address=ip_address,
            result=result,
            result_code=result_code,
            details=details,
            **kwargs
        )
        return self.log(event)

    def query(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        result: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        查询审计日志

        Args:
            user_id: 用户ID过滤
            action: 动作过滤
            resource: 资源过滤
            resource_type: 资源类型过滤
            start_time: 开始时间
            end_time: 结束时间
            result: 结果过滤
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            符合条件的审计事件列表
        """
        events = []

        # 扫描所有日志文件
        for i in range(self._file_index + 1):
            file_path = os.path.join(self.storage_path, f"audit_{i:03d}.jsonl")
            if not os.path.exists(file_path):
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event_data = json.loads(line.strip())
                        event = AuditEvent.from_dict(event_data)

                        # 应用过滤条件
                        if user_id and event.user_id != user_id:
                            continue
                        if action and event.action != action:
                            continue
                        if resource and event.resource != resource:
                            continue
                        if resource_type and event.resource_type != resource_type:
                            continue
                        if result is not None and event.result != result:
                            continue
                        if start_time and event.timestamp < start_time:
                            continue
                        if end_time and event.timestamp > end_time:
                            continue

                        events.append(event.to_dict())
                    except (json.JSONDecodeError, KeyError):
                        continue

        # 应用分页
        return events[offset:offset + limit]

    def get_user_activity(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取用户活动记录"""
        return self.query(user_id=user_id, limit=limit)

    def get_resource_history(
        self,
        resource: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取资源访问历史"""
        return self.query(resource=resource, limit=limit)

    def get_failed_attempts(
        self,
        start_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取失败的访问尝试"""
        return self.query(result=False, start_time=start_time, limit=limit)

    def get_security_events(
        self,
        start_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取安全相关事件"""
        security_actions = [
            AuditAction.SECURITY_ALERT.value,
            AuditAction.SUSPICIOUS_ACTIVITY.value,
            AuditAction.UNAUTHORIZED_ACCESS.value,
            AuditAction.LOGIN_FAILED.value
        ]

        events = []
        for action in security_actions:
            events.extend(
                self.query(
                    action=action,
                    start_time=start_time,
                    limit=limit
                )
            )

        # 按时间排序
        events.sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )

        return events[:limit]

    def export(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json"
    ) -> str:
        """
        导出审计日志

        Args:
            start_time: 开始时间
            end_time: 结束时间
            format: 导出格式 (json, csv)

        Returns:
            导出文件路径
        """
        events = self.query(start_time=start_time, end_time=end_time, limit=10000)

        export_path = os.path.join(
            self.storage_path,
            f"audit_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

        if format == "json":
            with open(f"{export_path}.json", "w", encoding="utf-8") as f:
                json.dump(events, f, ensure_ascii=False, indent=2)
            return f"{export_path}.json"

        elif format == "csv":
            import csv
            with open(f"{export_path}.csv", "w", encoding="utf-8", newline="") as f:
                if events:
                    writer = csv.DictWriter(f, fieldnames=events[0].keys())
                    writer.writeheader()
                    writer.writerows(events)
            return f"{export_path}.csv"

        return ""

    def get_stats(self) -> Dict[str, Any]:
        """获取审计日志统计"""
        events = self.query(limit=10000)

        action_counts: Dict[str, int] = {}
        user_counts: Dict[str, int] = {}
        result_counts = {"success": 0, "failure": 0}

        for event in events:
            action = event.get("action", "unknown")
            user = event.get("user_id", "unknown")
            result = "success" if event.get("result") else "failure"

            action_counts[action] = action_counts.get(action, 0) + 1
            user_counts[user] = user_counts.get(user, 0) + 1
            result_counts[result] += 1

        return {
            "total_events": len(events),
            "action_distribution": action_counts,
            "user_activity": user_counts,
            "result_distribution": result_counts
        }
