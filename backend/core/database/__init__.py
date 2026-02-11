"""
数据库模块
提供SQLite和PostgreSQL数据库支持，包括连接池管理和读写分离
"""

from .pool import (
    PostgresPool,
    PoolConfig,
    get_pool,
    init_pool,
    close_pool,
    execute_write,
    execute_read,
    execute_read_one
)

from .config import (
    DatabaseConfig,
    config
)

from .adapter import (
    DatabaseAdapter,
    DatabaseType,
    DatabaseConfig as DBConfig,
    get_adapter,
    execute as db_execute,
    execute_one as db_execute_one,
    execute_many as db_execute_many
)

__all__ = [
    # 连接池
    'PostgresPool',
    'PoolConfig',
    'get_pool',
    'init_pool',
    'close_pool',
    'execute_write',
    'execute_read',
    'execute_read_one',
    
    # 配置
    'DatabaseConfig',
    'config',
    
    # 适配器
    'DatabaseAdapter',
    'DatabaseType',
    'DBConfig',
    'get_adapter',
    'db_execute',
    'db_execute_one',
    'db_execute_many',
]
