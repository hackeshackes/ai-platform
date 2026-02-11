"""
Quota Manager - 配额管理器

提供API配额管理功能，支持：
- 用户级别配额
- API级别配额
- 配额周期管理（每日/每周/每月）
- 配额预警
"""
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import time
import threading
import hashlib

class QuotaPeriod(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"

@dataclass
class QuotaConfig:
    """配额配置"""
    limit: int  # 配额上限
    period: QuotaPeriod = QuotaPeriod.DAILY
    reset_on_period_end: bool = True
    warning_threshold: float = 0.8  # 80%预警阈值

@dataclass
class QuotaUsage:
    """配额使用情况"""
    key: str
    used: int
    limit: int
    remaining: int
    period: str
    reset_at: float
    percentage: float
    is_exceeded: bool

@dataclass
class QuotaRule:
    """配额规则"""
    name: str
    key_pattern: str  # 支持通配符
    config: QuotaConfig
    priority: int = 0
    enabled: bool = True
    description: str = ""

class QuotaManager:
    """配额管理器"""
    
    def __init__(self):
        self.usage: Dict[str, Dict[str, int]] = {}  # key -> {period -> count}
        self.period_keys: Dict[str, str] = {}  # key -> current_period_key
        self.rules: Dict[str, QuotaRule] = {}
        self.default_config = QuotaConfig(limit=1000)
        self.lock = threading.Lock()
    
    def _get_period_key(self, period: QuotaPeriod) -> str:
        """获取当前周期key"""
        now = datetime.now(timezone.utc)
        
        if period == QuotaPeriod.DAILY:
            return now.strftime("%Y-%m-%d")
        elif period == QuotaPeriod.WEEKLY:
            # ISO周
            return f"{now.year}-W{now.isocalendar()[1]:02d}"
        elif period == QuotaPeriod.MONTHLY:
            return now.strftime("%Y-%m")
        elif period == QuotaPeriod.YEARLY:
            return str(now.year)
        return now.strftime("%Y-%m-%d")
    
    def _get_period_end(self, period: QuotaPeriod) -> float:
        """获取周期结束时间戳"""
        now = datetime.now(timezone.utc)
        
        if period == QuotaPeriod.DAILY:
            tomorrow = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif period == QuotaPeriod.WEEKLY:
            # 下周一
            days_ahead = 7 - now.weekday()
            next_monday = (now + timedelta(days=days_ahead)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            tomorrow = next_monday
        elif period == QuotaPeriod.MONTHLY:
            next_month = now.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=32)
            next_month = next_month.replace(day=1)
            tomorrow = next_month
        elif period == QuotaPeriod.YEARLY:
            next_year = now.replace(year=now.year + 1, month=1, day=1)
            tomorrow = next_year
        else:
            tomorrow = now + timedelta(days=1)
        
        return tomorrow.timestamp()
    
    def _match_rule(self, key: str) -> Optional[QuotaRule]:
        """匹配配额规则"""
        matched = None
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            if self._match_pattern(key, rule.key_pattern):
                if matched is None or rule.priority > matched.priority:
                    matched = rule
        return matched
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """检查key是否匹配模式"""
        if pattern == "*" or pattern == key:
            return True
        
        # 支持简单通配符
        if pattern.endswith("*"):
            return key.startswith(pattern[:-1])
        if pattern.startswith("*"):
            return key.endswith(pattern[1:])
        
        return key == pattern
    
    def consume(self, key: str, amount: int = 1) -> QuotaUsage:
        """消耗配额"""
        with self.lock:
            rule = self._match_rule(key)
            if rule:
                config = rule.config
            else:
                config = self.default_config
            
            period_key = self._get_period_key(config.period)
            storage_key = f"{key}:{period_key}"
            
            if storage_key not in self.usage:
                self.usage[storage_key] = {}
            
            current = self.usage[storage_key].get("count", 0)
            new_count = current + amount
            
            limit = config.limit
            remaining = max(0, limit - new_count)
            is_exceeded = new_count > limit
            
            self.usage[storage_key] = {
                "count": new_count,
                "period": period_key,
                "updated_at": time.time()
            }
            
            # 保存period key用于查询
            self.period_keys[key] = period_key
            
            percentage = (new_count / limit * 100) if limit > 0 else 100
            reset_at = self._get_period_end(config.period)
            
            return QuotaUsage(
                key=key,
                used=new_count,
                limit=limit,
                remaining=remaining,
                period=period_key,
                reset_at=reset_at,
                percentage=percentage,
                is_exceeded=is_exceeded
            )
    
    def get_usage(self, key: str) -> QuotaUsage:
        """获取配额使用情况"""
        with self.lock:
            rule = self._match_rule(key)
            if rule:
                config = rule.config
            else:
                config = self.default_config
            
            period_key = self._get_period_key(config.period)
            storage_key = f"{key}:{period_key}"
            
            current = self.usage.get(storage_key, {}).get("count", 0)
            limit = config.limit
            remaining = max(0, limit - current)
            
            percentage = (current / limit * 100) if limit > 0 else 100
            reset_at = self._get_period_end(config.period)
            
            return QuotaUsage(
                key=key,
                used=current,
                limit=limit,
                remaining=remaining,
                period=period_key,
                reset_at=reset_at,
                percentage=percentage,
                is_exceeded=current > limit
            )
    
    def is_allowed(self, key: str, amount: int = 1) -> bool:
        """检查是否允许请求"""
        usage = self.get_usage(key)
        return (usage.used + amount) <= usage.limit
    
    def reset(self, key: str, period: Optional[QuotaPeriod] = None):
        """重置配额"""
        with self.lock:
            if period:
                period_key = self._get_period_key(period)
                storage_key = f"{key}:{period_key}"
                self.usage.pop(storage_key, None)
            else:
                # 重置所有period
                keys_to_remove = [k for k in self.usage if k.startswith(f"{key}:")]
                for k in keys_to_remove:
                    del self.usage[k]
                self.period_keys.pop(key, None)
    
    def add_rule(self, rule: QuotaRule):
        """添加配额规则"""
        rule_id = hashlib.md5(f"{rule.name}:{rule.key_pattern}".encode()).hexdigest()[:12]
        self.rules[rule_id] = rule
    
    def list_rules(self) -> List[Dict]:
        """列出所有规则"""
        return [
            {
                "id": rid,
                "name": r.name,
                "key_pattern": r.key_pattern,
                "limit": r.config.limit,
                "period": r.config.period.value,
                "priority": r.priority,
                "enabled": r.enabled,
                "description": r.description
            }
            for rid, r in self.rules.items()
        ]
    
    def delete_rule(self, rule_id: str) -> bool:
        """删除规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False
    
    def get_all_usage(self) -> List[QuotaUsage]:
        """获取所有配额使用情况"""
        with self.lock:
            usages = []
            seen_keys = set()
            
            for storage_key, data in self.usage.items():
                key = storage_key.split(":")[0]
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                usages.append(self.get_usage(key))
            
            return usages
    
    def cleanup_expired(self):
        """清理过期数据"""
        with self.lock:
            now = time.time()
            expired_keys = []
            
            for storage_key, data in self.usage.items():
                updated_at = data.get("updated_at", 0)
                # 清理7天前的数据
                if now - updated_at > 7 * 24 * 3600:
                    expired_keys.append(storage_key)
            
            for key in expired_keys:
                del self.usage[key]

# 全局配额管理器实例
quota_manager = QuotaManager()
