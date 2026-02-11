"""
数据库配置
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    """数据库配置"""
    # SQLite (当前)
    sqlite_path: str = "data/ai_platform.db"
    
    # PostgreSQL (目标)
    pg_host: str = "/tmp"
    pg_port: int = 5432
    pg_database: str = "aiplatform"
    pg_user: str = "yubao"
    pg_password: str = ""
    
    # 连接池
    pool_min: int = 5
    pool_max: int = 20
    
    # 模式选择
    use_postgres: bool = False  # 切换到PostgreSQL时设为True
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """从环境变量加载配置"""
        import os
        return cls(
            pg_host=os.getenv("PGHOST", "/tmp"),
            pg_port=int(os.getenv("PGPORT", "5432")),
            pg_database=os.getenv("PGDATABASE", "aiplatform"),
            pg_user=os.getenv("PGUSER", "yubao"),
            pg_password=os.getenv("PGPASSWORD", ""),
            use_postgres=os.getenv("USE_POSTGRES", "").lower() == "true"
        )

# 默认配置
config = DatabaseConfig()
