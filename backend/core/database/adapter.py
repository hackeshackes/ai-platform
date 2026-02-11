"""
SQLite适配器
提供与PostgreSQL兼容的SQLite后端，支持在两种数据库之间切换
"""
import sqlite3
import threading
import queue
import time
from typing import Optional, Generator, Dict, Any, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum


class DatabaseType(Enum):
    """数据库类型"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


@dataclass
class DatabaseConfig:
    """数据库配置"""
    db_type: DatabaseType = DatabaseType.SQLITE
    sqlite_path: str = "data/ai_platform.db"
    
    # PostgreSQL配置
    pg_host: str = "/tmp"
    pg_port: int = 5432
    pg_database: str = "aiplatform"
    pg_user: str = "yubao"
    pg_password: str = ""
    
    # 连接池配置
    pool_min: int = 5
    pool_max: int = 50
    connection_timeout: int = 30


class SQLiteConnectionPool:
    """SQLite连接池"""
    
    def __init__(
        self,
        database: str = "data/ai_platform.db",
        min_size: int = 5,
        max_size: int = 20,
        timeout: int = 30
    ):
        self.database = database
        self.min_size = min_size
        self.max_size = max_size
        self.timeout = timeout
        
        self._pool: queue.Queue = queue.Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._active = 0
        self._created = 0
        self._idle = 0
        self._init_pool()
    
    def _init_pool(self):
        """初始化连接池"""
        for _ in range(self.min_size):
            conn = self._create_connection()
            if conn:
                self._pool.put(conn)
                self._idle += 1
    
    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """创建连接"""
        try:
            conn = sqlite3.connect(
                self.database, 
                check_same_thread=False, 
                timeout=30
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")
            self._created += 1
            return conn
        except Exception as e:
            print(f"Failed to create SQLite connection: {e}")
            return None
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """获取连接"""
        try:
            conn = self._pool.get(timeout=self.timeout)
            self._idle -= 1
            self._active += 1
            try:
                yield conn
            finally:
                self._pool.put(conn)
                self._idle += 1
                self._active -= 1
        except queue.Empty:
            if self._created < self._max_size:
                conn = self._create_connection()
                if conn:
                    self._idle += 1
                    self._created += 1
                    try:
                        yield conn
                    finally:
                        self._pool.put(conn)
                    return
            raise RuntimeError("Connection pool exhausted")
    
    def execute(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """执行查询"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.fetchall()
    
    def execute_one(self, query: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
        """执行查询（单条结果）"""
        results = self.execute(query, params)
        return results[0] if results else None
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """批量执行"""
        with self.get_connection() as conn:
            cursor = conn.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
    
    def close(self):
        """关闭连接池"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "active": self._active,
            "idle": self._idle,
            "created": self._created,
            "max_size": self._max_size,
            "utilization": self._active / self._max_size if self._max_size > 0 else 0
        }


class DatabaseAdapter:
    """
    数据库适配器
    提供统一的接口，支持SQLite和PostgreSQL之间的切换
    """
    
    _instance: Optional['DatabaseAdapter'] = None
    _lock = threading.Lock()
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._sqlite_pool: Optional[SQLiteConnectionPool] = None
        self._pg_pool = None  # PostgreSQL pool
        self._initialized = False
    
    @classmethod
    def get_instance(cls, config: Optional[DatabaseConfig] = None) -> 'DatabaseAdapter':
        """获取单例"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance
    
    def initialize(self):
        """初始化数据库"""
        if self._initialized:
            return
        
        if self.config.db_type == DatabaseType.SQLITE:
            self._init_sqlite()
        elif self.config.db_type == DatabaseType.POSTGRESQL:
            self._init_postgresql()
        
        self._initialized = True
    
    def _init_sqlite(self):
        """初始化SQLite"""
        self._sqlite_pool = SQLiteConnectionPool(
            database=self.config.sqlite_path,
            min_size=self.config.pool_min,
            max_size=self.config.pool_max,
            timeout=self.config.connection_timeout
        )
        print(f"✅ SQLite pool initialized: {self.config.sqlite_path}")
    
    def _init_postgresql(self):
        """初始化PostgreSQL"""
        try:
            from core.database.pool import PostgresPool, PoolConfig
            
            pg_config = PoolConfig(
                min_connections=self.config.pool_min,
                max_connections=self.config.pool_max,
                host=self.config.pg_host,
                port=self.config.pg_port,
                database=self.config.pg_database,
                connection_timeout=self.config.connection_timeout
            )
            
            self._pg_pool = PostgresPool(pg_config)
            print(f"✅ PostgreSQL pool initialized: {self.config.pg_host}")
            
        except ImportError:
            print("⚠️  PostgreSQL pool not available, falling back to SQLite")
            self.config.db_type = DatabaseType.SQLITE
            self._init_sqlite()
    
    def switch_database(self, db_type: DatabaseType):
        """切换数据库类型"""
        self.close()
        self.config.db_type = db_type
        self._initialized = False
        self.initialize()
    
    def execute(
        self, 
        query: str, 
        params: Tuple = (),
        fetch: bool = True
    ) -> List[Tuple]:
        """执行查询（统一接口）"""
        if not self._initialized:
            self.initialize()
        
        if self.config.db_type == DatabaseType.SQLITE:
            return self._execute_sqlite(query, params, fetch)
        else:
            return self._execute_postgresql(query, params, fetch)
    
    def _execute_sqlite(
        self, 
        query: str, 
        params: Tuple,
        fetch: bool
    ) -> List[Tuple]:
        """执行SQLite查询"""
        if self._sqlite_pool is None:
            self._init_sqlite()
        
        results = self._sqlite_pool.execute(query, params)
        
        if not fetch:
            return []
        
        return [tuple(row) for row in results]
    
    def _execute_postgresql(
        self, 
        query: str, 
        params: Tuple,
        fetch: bool
    ) -> List[Tuple]:
        """执行PostgreSQL查询"""
        if self._pg_pool is None:
            self._init_postgresql()
        
        if fetch:
            return self._pg_pool.execute_read(query, params)
        else:
            self._pg_pool.execute_write(query, params)
            return []
    
    def execute_one(self, query: str, params: Tuple = ()) -> Optional[Tuple]:
        """执行查询（单条结果）"""
        results = self.execute(query, params)
        return results[0] if results else None
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """批量执行"""
        if not self._initialized:
            self.initialize()
        
        if self.config.db_type == DatabaseType.SQLITE:
            return self._sqlite_pool.execute_many(query, params_list)
        else:
            return self._pg_pool.execute_many(query, params_list)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._initialized:
            self.initialize()
        
        return {
            "db_type": self.config.db_type.value,
            "sqlite": self._sqlite_pool.get_stats() if self._sqlite_pool else None,
            "postgresql": self._pg_pool.get_stats() if self._pg_pool else None
        }
    
    def close(self):
        """关闭连接"""
        if self._sqlite_pool:
            self._sqlite_pool.close()
            self._sqlite_pool = None
        
        if self._pg_pool:
            self._pg_pool.close()
            self._pg_pool = None
        
        self._initialized = False


# 全局适配器实例
_adapter: Optional[DatabaseAdapter] = None

def get_adapter(config: Optional[DatabaseConfig] = None) -> DatabaseAdapter:
    """获取全局适配器"""
    global _adapter
    if _adapter is None:
        _adapter = DatabaseAdapter.get_instance(config)
    return _adapter

def execute(query: str, params: Tuple = (), fetch: bool = True) -> List[Tuple]:
    """执行查询"""
    return get_adapter().execute(query, params, fetch)

def execute_one(query: str, params: Tuple = ()) -> Optional[Tuple]:
    """执行查询（单条）"""
    return get_adapter().execute_one(query, params)

def execute_many(query: str, params_list: List[Tuple]) -> int:
    """批量执行"""
    return get_adapter().execute_many(query, params_list)
