"""
自动化运维平台 - 配置文件
=============================

提供统一的配置管理:
- 默认配置
- 环境变量支持
- 配置验证

作者: AI Platform Team
版本: 1.0.0
"""

import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class Environment(Enum):
    """运行环境"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class PipelineConfig:
    """流水线配置"""
    max_concurrent_tasks: int = 10
    default_timeout: int = 300
    default_retries: int = 3
    progress_callback_enabled: bool = True


@dataclass
class SchedulerConfig:
    """调度器配置"""
    timezone: str = "UTC"
    check_interval: int = 10
    max_history: int = 1000
    enable_metrics: bool = True
    cleanup_interval: int = 3600


@dataclass
class WorkflowConfig:
    """工作流配置"""
    max_concurrent_steps: int = 10
    default_timeout: int = 86400
    enable_audit: bool = True


@dataclass
class NotificationConfig:
    """通知配置"""
    enable_metrics: bool = True
    default_channels: List[str] = field(default_factory=lambda: ["email"])
    max_retries: int = 3
    retry_interval: int = 60


@dataclass
class ServerConfig:
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1


@dataclass
class DatabaseConfig:
    """数据库配置"""
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str = "automation_ops"
    username: str = ""
    password: str = ""
    pool_size: int = 5


@dataclass
class RedisConfig:
    """Redis配置"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    key_prefix: str = "automation_ops:"


@dataclass
class Config:
    """主配置类"""
    environment: Environment = Environment.DEVELOPMENT
    
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    
    server: ServerConfig = field(default_factory=ServerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    
    # 自定义配置
    custom: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量加载配置"""
        env = Environment(os.getenv("ENVIRONMENT", "development").lower())
        
        pipeline = PipelineConfig(
            max_concurrent_tasks=int(os.getenv("MAX_CONCURRENT_TASKS", 10)),
            default_timeout=int(os.getenv("DEFAULT_TIMEOUT", 300)),
            default_retries=int(os.getenv("DEFAULT_RETRIES", 3))
        )
        
        scheduler = SchedulerConfig(
            timezone=os.getenv("TIMEZONE", "UTC"),
            check_interval=int(os.getenv("CHECK_INTERVAL", 10)),
            max_history=int(os.getenv("MAX_HISTORY", 1000))
        )
        
        server = ServerConfig(
            host=os.getenv("SERVER_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVER_PORT", 8000)),
            debug=os.getenv("SERVER_DEBUG", "false").lower() == "true"
        )
        
        database = DatabaseConfig(
            type=os.getenv("DB_TYPE", "sqlite"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            database=os.getenv("DB_NAME", "automation_ops"),
            username=os.getenv("DB_USERNAME", ""),
            password=os.getenv("DB_PASSWORD", "")
        )
        
        redis = RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            password=os.getenv("REDIS_PASSWORD", ""),
            key_prefix=os.getenv("REDIS_KEY_PREFIX", "automation_ops:")
        )
        
        return cls(
            environment=env,
            pipeline=pipeline,
            scheduler=scheduler,
            server=server,
            database=database,
            redis=redis
        )
    
    @classmethod
    def from_file(cls, path: str) -> "Config":
        """从文件加载配置"""
        with open(path, "r") as f:
            data = json.load(f)
        
        return cls(**data)
    
    def to_file(self, path: str) -> None:
        """保存配置到文件"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "environment": self.environment.value,
            "pipeline": {
                "max_concurrent_tasks": self.pipeline.max_concurrent_tasks,
                "default_timeout": self.pipeline.default_timeout,
                "default_retries": self.pipeline.default_retries
            },
            "scheduler": {
                "timezone": self.scheduler.timezone,
                "check_interval": self.scheduler.check_interval,
                "max_history": self.scheduler.max_history
            },
            "workflow": {
                "max_concurrent_steps": self.workflow.max_concurrent_steps,
                "default_timeout": self.workflow.default_timeout,
                "enable_audit": self.workflow.enable_audit
            },
            "notification": {
                "enable_metrics": self.notification.enable_metrics,
                "default_channels": self.notification.default_channels,
                "max_retries": self.notification.max_retries
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "debug": self.server.debug
            },
            "database": {
                "type": self.database.type,
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
                "key_prefix": self.redis.key_prefix
            },
            "custom": self.custom
        }


# 全局配置实例
_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置"""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """设置全局配置"""
    global _config
    _config = config


# 常用配置模板

def development_config() -> Config:
    """开发环境配置"""
    return Config(
        environment=Environment.DEVELOPMENT,
        pipeline=PipelineConfig(max_concurrent_tasks=5),
        server=ServerConfig(debug=True)
    )


def production_config() -> Config:
    """生产环境配置"""
    return Config(
        environment=Environment.PRODUCTION,
        pipeline=PipelineConfig(max_concurrent_tasks=50),
        scheduler=SchedulerConfig(check_interval=5),
        server=ServerConfig(workers=4)
    )


def testing_config() -> Config:
    """测试环境配置"""
    return Config(
        environment=Environment.TESTING,
        pipeline=PipelineConfig(max_concurrent_tasks=2),
        scheduler=SchedulerConfig(check_interval=1)
    )
