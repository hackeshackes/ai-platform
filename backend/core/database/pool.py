"""
PostgreSQL连接池管理
支持读写分离和连接池配置
"""
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import time
import os


@dataclass
class PoolConfig:
    """连接池配置"""
    min_connections: int = 5
    max_connections: int = 50
    host: str = "/tmp"
    port: int = 5432
    database: str = "aiplatform"
    user: str = "yubao"
    password: str = ""
    connection_timeout: int = 30
    idle_timeout: int = 600  # 10分钟
    max_retries: int = 3


class PostgresPool:
    """
    PostgreSQL连接池类
    支持读写分离：写操作使用主库，读操作可使用只读副本
    """
    
    def __init__(self, config: Optional[PoolConfig] = None):
        self.config = config or PoolConfig()
        self._main_pool: Optional[pool.SimpleConnectionPool] = None
        self._read_pool: Optional[pool.SimpleConnectionPool] = None
        self._lock = threading.Lock()
        self._initialized = False
        
        # 统计信息
        self._stats = {
            "main_pool": {"active": 0, "idle": 0, "created": 0},
            "read_pool": {"active": 0, "idle": 0, "created": 0}
        }
    
    def _create_connection_kwargs(self, is_read: bool = False) -> Dict[str, Any]:
        """创建连接参数"""
        kwargs = {
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.database,
            "user": self.config.user,
            "password": self.config.password or "",
            "connect_timeout": self.config.connection_timeout,
            "options": "-c statement_timeout=30000"  # 30秒查询超时
        }
        return kwargs
    
    def _init_pools(self):
        """初始化连接池（线程安全）"""
        with self._lock:
            if self._initialized:
                return
            
            try:
                # 主库连接池（读写）
                self._main_pool = pool.SimpleConnectionPool(
                    self.config.min_connections,
                    self.config.max_connections,
                    **self._create_connection_kwargs(is_read=False)
                )
                
                # 只读副本连接池（读操作）
                # 如果有只读副本，使用不同的连接参数
                read_host = os.getenv("PGREAD_HOST", self.config.host)
                if read_host != self.config.host:
                    self._read_pool = pool.SimpleConnectionPool(
                        self.config.min_connections,
                        self.config.max_connections,
                        **self._create_connection_kwargs(is_read=True)
                    )
                else:
                    # 没有只读副本时，使用主库
                    self._read_pool = self._main_pool
                
                self._initialized = True
                print(f"✅ PostgreSQL pools initialized: main={self.config.host}, read={read_host}")
                
            except Exception as e:
                print(f"❌ Failed to initialize PostgreSQL pools: {e}")
                raise
    
    @property
    def main_pool(self) -> pool.SimpleConnectionPool:
        """获取主库连接池"""
        if not self._initialized:
            self._init_pools()
        return self._main_pool
    
    @property
    def read_pool(self) -> pool.SimpleConnectionPool:
        """获取只读连接池"""
        if not self._initialized:
            self._init_pools()
        return self._read_pool
    
    @contextmanager
    def get_write_connection(self):
        """
        获取写连接（主库）
        用于INSERT、UPDATE、DELETE操作
        """
        conn = None
        try:
            conn = self.main_pool.getconn()
            conn.autocommit = False
            yield conn
        finally:
            if conn:
                conn.close()
                self.main_pool.putconn(conn)
    
    @contextmanager
    def get_read_connection(self):
        """
        获取读连接（主库或只读副本）
        用于SELECT查询
        """
        conn = None
        try:
            conn = self.read_pool.getconn()
            yield conn
        finally:
            if conn:
                conn.close()
                self.read_pool.putconn(conn)
    
    def execute_write(
        self, 
        query: str, 
        params: Tuple = None, 
        fetch: bool = False
    ) -> List[Tuple]:
        """执行写操作"""
        with self.get_write_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            if fetch:
                return cursor.fetchall()
            return []
    
    def execute_read(self, query: str, params: Tuple = None) -> List[Tuple]:
        """执行读操作"""
        with self.get_read_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_read_one(self, query: str, params: Tuple = None) -> Optional[Tuple]:
        """执行读操作（单条结果）"""
        results = self.execute_read(query, params)
        return results[0] if results else None
    
    def execute_write_one(
        self, 
        query: str, 
        params: Tuple = None, 
        fetch: bool = False
    ) -> Optional[Tuple]:
        """执行写操作（单条结果）"""
        results = self.execute_write(query, params, fetch=fetch)
        return results[0] if results else None
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """批量执行"""
        with self.get_write_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
    
    def transaction(self):
        """
        获取事务上下文
        使用方式:
            with pool.transaction() as conn:
                conn.execute(...)
        """
        return self.get_write_connection()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查主库
            with self.get_read_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                main_healthy = True
        except Exception as e:
            main_healthy = False
            print(f"Main pool health check failed: {e}")
        
        return {
            "main_pool": {
                "healthy": main_healthy,
                **self._stats["main_pool"]
            },
            "read_pool": {
                "healthy": main_healthy,
                **self._stats["read_pool"]
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计"""
        return {
            "config": {
                "min_connections": self.config.min_connections,
                "max_connections": self.config.max_connections,
                "host": self.config.host,
                "database": self.config.database
            },
            "initialized": self._initialized,
            "health": self.health_check()
        }
    
    def close(self):
        """关闭所有连接池"""
        if self._main_pool:
            self._main_pool.closeall()
        if self._read_pool and self._read_pool != self._main_pool:
            self._read_pool.closeall()
        self._initialized = False
        print("✅ PostgreSQL pools closed")


# 全局连接池实例
_pool: Optional[PostgresPool] = None

def get_pool() -> PostgresPool:
    """获取全局连接池"""
    global _pool
    if _pool is None:
        _pool = PostgresPool()
    return _pool

def init_pool(config: Optional[PoolConfig] = None):
    """初始化全局连接池"""
    global _pool
    _pool = PostgresPool(config)
    _pool._init_pools()
    return _pool

def close_pool():
    """关闭全局连接池"""
    global _pool
    if _pool:
        _pool.close()
        _pool = None

# 便捷函数
def execute_write(query: str, params: Tuple = None, fetch: bool = False):
    """执行写操作"""
    return get_pool().execute_write(query, params, fetch)

def execute_read(query: str, params: Tuple = None):
    """执行读操作"""
    return get_pool().execute_read(query, params)

def execute_read_one(query: str, params: Tuple = None):
    """执行读操作（单条）"""
    return get_pool().execute_read_one(query, params)
