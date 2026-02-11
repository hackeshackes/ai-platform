#!/usr/bin/env python3
"""
SQLite到PostgreSQL迁移脚本

Usage:
    python scripts/migrate_sqlite_to_pg.py [--dry-run] [--tables users,projects,...]
"""
import sqlite3
import psycopg2
from psycopg2 import sql
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import argparse
import sys
import os
import json
from datetime import datetime

# 添加backend到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database.pool import PostgresPool, PoolConfig, get_pool


@dataclass
class MigrationConfig:
    """迁移配置"""
    sqlite_path: str = "data/ai_platform.db"
    pg_config: PoolConfig = None
    batch_size: int = 1000
    dry_run: bool = False
    tables: List[str] = None


# SQLite到PostgreSQL类型映射
TYPE_MAPPING = {
    "INTEGER": "INTEGER",
    "INT": "INTEGER",
    "BIGINT": "BIGINT",
    "SMALLINT": "SMALLINT",
    "REAL": "REAL",
    "FLOAT": "REAL",
    "DOUBLE": "DOUBLE PRECISION",
    "TEXT": "TEXT",
    "VARCHAR": "VARCHAR",
    "CHAR": "CHAR",
    "BOOLEAN": "BOOLEAN",
    "BOOL": "BOOLEAN",
    "DATETIME": "TIMESTAMP",
    "DATE": "DATE",
    "TIME": "TIME",
    "BLOB": "BYTEA",
    "JSON": "JSONB",
    "NUMERIC": "NUMERIC",
    "DECIMAL": "DECIMAL",
}


def get_sqlite_type(sqlite_type: str) -> str:
    """获取SQLite类型映射"""
    sqlite_type = sqlite_type.upper().strip()
    return TYPE_MAPPING.get(sqlite_type, "TEXT")


class SQLiteSchemaReader:
    """SQLite schema读取器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """连接SQLite"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
    
    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()
    
    @contextmanager
    def get_connection(self):
        """获取连接上下文"""
        if not self.conn:
            self.connect()
        try:
            yield self.conn
        finally:
            pass
    
    def get_tables(self) -> List[str]:
        """获取所有表名"""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """获取表结构"""
        # 获取列信息
        cursor = self.conn.execute(f"PRAGMA table_info({table_name})")
        columns = []
        for row in cursor.fetchall():
            col_name = row['name']
            col_type = row['type']
            col_primary = row['pk'] == 1
            col_nullable = row['notnull'] == 0
            col_default = row['dflt_value']
            
            columns.append({
                "name": col_name,
                "type": get_sqlite_type(col_type),
                "primary_key": col_primary,
                "nullable": col_nullable,
                "default": col_default
            })
        
        # 获取索引信息
        cursor = self.conn.execute(f"PRAGMA index_list({table_name})")
        indexes = []
        for row in cursor.fetchall():
            if row['unique']:
                indexes.append({
                    "name": row['name'],
                    "columns": self.get_index_columns(table_name, row['name']),
                    "unique": True
                })
        
        # 获取外键信息
        cursor = self.conn.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = []
        for row in cursor.fetchall():
            foreign_keys.append({
                "table": row['table'],
                "from": row['from'],
                "to": row['to']
            })
        
        return {
            "name": table_name,
            "columns": columns,
            "indexes": indexes,
            "foreign_keys": foreign_keys
        }
    
    def get_index_columns(self, table_name: str, index_name: str) -> List[str]:
        """获取索引列"""
        cursor = self.conn.execute(
            f"PRAGMA index_xinfo({index_name})"
        )
        return [row[2] for row in cursor.fetchall() if row[2]]
    
    def get_table_data(
        self, 
        table_name: str, 
        offset: int = 0, 
        limit: int = 1000
    ) -> List[Tuple]:
        """获取表数据（分页）"""
        cursor = self.conn.execute(
            f"SELECT * FROM {table_name} LIMIT ? OFFSET ?",
            (limit, offset)
        )
        return cursor.fetchall()
    
    def get_table_count(self, table_name: str) -> int:
        """获取表记录数"""
        cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]


class PostgresSchemaWriter:
    """PostgreSQL schema写入器"""
    
    def __init__(self, pool: PostgresPool):
        self.pool = pool
        self.conn = None
    
    @contextmanager
    def get_connection(self):
        """获取写连接"""
        with self.pool.get_write_connection() as conn:
            yield conn
    
    def create_table(self, schema: Dict[str, Any], if_not_exists: bool = True):
        """创建表"""
        table_name = schema['name']
        columns_def = []
        
        for col in schema['columns']:
            col_def = f"{col['name']} {col['type']}"
            
            if col['primary_key']:
                col_def += " PRIMARY KEY"
            
            if not col['nullable']:
                col_def += " NOT NULL"
            
            if col['default'] is not None:
                col_def += f" DEFAULT {col['default']}"
            
            columns_def.append(col_def)
        
        # 外键约束
        for fk in schema['foreign_keys']:
            col_def = f"FOREIGN KEY ({fk['from']}) REFERENCES {fk['table']}({fk['to']})"
            columns_def.append(col_def)
        
        # 构建CREATE TABLE语句
        if_exists = "IF NOT EXISTS " if if_not_exists else ""
        query = f"""
        CREATE TABLE {if_exists}\"{table_name}\" (
            {', '.join(columns_def)}
        )
        """
        
        if self.pool.config.dry_run:
            print(f"[DRY-RUN] {query}")
            return
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
    
    def create_indexes(self, schema: Dict[str, Any]):
        """创建索引"""
        table_name = schema['name']
        
        for idx in schema['indexes']:
            idx_name = f"idx_{table_name}_{'_'.join(idx['columns'])}"
            columns = ', '.join(idx['columns'])
            
            query = f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ({columns})'
            
            if self.pool.config.dry_run:
                print(f"[DRY-RUN] {query}")
                continue
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
    
    def drop_table(self, table_name: str):
        """删除表"""
        query = f'DROP TABLE IF EXISTS "{table_name}" CASCADE'
        
        if self.pool.config.dry_run:
            print(f"[DRY-RUN] {query}")
            return
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
    
    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        )
        """
        
        with self.pool.get_read_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (table_name,))
            return cursor.fetchone()[0]


class DataMigrator:
    """数据迁移器"""
    
    def __init__(
        self, 
        sqlite_path: str, 
        pg_pool: PostgresPool,
        batch_size: int = 1000,
        dry_run: bool = False
    ):
        self.sqlite_reader = SQLiteSchemaReader(sqlite_path)
        self.pg_writer = PostgresSchemaWriter(pg_pool)
        self.pg_pool = pg_pool
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.stats = {
            "tables_migrated": 0,
            "rows_migrated": 0,
            "errors": []
        }
    
    def migrate_table(
        self, 
        table_name: str,
        recreate: bool = False
    ) -> Dict[str, Any]:
        """迁移单个表"""
        print(f"\n📦 Migrating table: {table_name}")
        
        # 读取SQLite schema
        schema = self.sqlite_reader.get_table_schema(table_name)
        
        # 检查目标表是否存在
        if self.pg_writer.table_exists(table_name):
            if recreate:
                print(f"  🔄 Dropping existing table: {table_name}")
                self.pg_writer.drop_table(table_name)
            else:
                print(f"  ⚠️  Table {table_name} already exists, skipping schema creation")
        
        # 创建表
        if not self.pg_writer.table_exists(table_name):
            print(f"  📐 Creating table: {table_name}")
            self.pg_writer.create_table(schema)
        
        # 创建索引
        print(f"  🔍 Creating indexes: {table_name}")
        self.pg_writer.create_indexes(schema)
        
        # 迁移数据
        total_rows = self.sqlite_reader.get_table_count(table_name)
        print(f"  📊 Total rows to migrate: {total_rows}")
        
        if total_rows == 0:
            print(f"  ✅ No data to migrate")
            self.stats["tables_migrated"] += 1
            return {"status": "success", "rows": 0}
        
        offset = 0
        migrated_rows = 0
        
        while offset < total_rows:
            # 读取数据
            rows = self.sqlite_reader.get_table_data(table_name, offset, self.batch_size)
            
            if not rows:
                break
            
            # 准备插入数据
            columns = [desc[0] for desc in rows[0].description]
            placeholders = ', '.join(['%s'] * len(columns))
            query = f'INSERT INTO "{table_name}" ({", ".join(columns)}) VALUES ({placeholders})'
            
            data = [tuple(row) for row in rows]
            
            if self.dry_run:
                print(f"  [DRY-RUN] INSERT {len(data)} rows into {table_name}")
            else:
                # 执行批量插入
                try:
                    with self.pg_pool.get_write_connection() as conn:
                        cursor = conn.cursor()
                        cursor.executemany(query, data)
                        conn.commit()
                except Exception as e:
                    # 如果批量插入失败，尝试逐行插入
                    print(f"  ⚠️  Batch insert failed: {e}, trying row by row")
                    for row in data:
                        try:
                            with self.pg_pool.get_write_connection() as conn:
                                cursor = conn.cursor()
                                cursor.execute(query, row)
                                conn.commit()
                        except Exception as row_e:
                            self.stats["errors"].append({
                                "table": table_name,
                                "row": row,
                                "error": str(row_e)
                            })
            
            migrated_rows += len(rows)
            offset += self.batch_size
            
            # 进度显示
            progress = (migrated_rows / total_rows) * 100
            print(f"  📈 Progress: {migrated_rows}/{total_rows} ({progress:.1f}%)")
        
        self.stats["tables_migrated"] += 1
        self.stats["rows_migrated"] += migrated_rows
        
        print(f"  ✅ Migrated {migrated_rows} rows")
        
        return {
            "status": "success",
            "rows": migrated_rows
        }
    
    def migrate_all(self, tables: List[str] = None, recreate: bool = False) -> Dict[str, Any]:
        """迁移所有表"""
        print("=" * 60)
        print("SQLite to PostgreSQL Migration")
        print("=" * 60)
        
        # 连接SQLite
        self.sqlite_reader.connect()
        
        try:
            # 获取所有表
            if tables is None:
                tables = self.sqlite_reader.get_tables()
            
            print(f"\n📋 Tables to migrate: {tables}")
            
            for table in tables:
                try:
                    self.migrate_table(table, recreate=recreate)
                except Exception as e:
                    print(f"  ❌ Error migrating {table}: {e}")
                    self.stats["errors"].append({
                        "table": table,
                        "error": str(e)
                    })
            
            print("\n" + "=" * 60)
            print("Migration Summary")
            print("=" * 60)
            print(f"  Tables migrated: {self.stats['tables_migrated']}")
            print(f"  Rows migrated: {self.stats['rows_migrated']}")
            print(f"  Errors: {len(self.stats['errors'])}")
            
            if self.stats['errors']:
                print("\n❌ Errors:")
                for error in self.stats['errors'][:10]:  # 只显示前10个
                    print(f"  - {error['table']}: {error['error']}")
            
            return self.stats
        
        finally:
            self.sqlite_reader.close()


def create_tables_postgres(pool: PostgresPool):
    """直接创建PostgreSQL表（如果没有SQLite源）"""
    
    tables = {
        "users": """
            CREATE TABLE IF NOT EXISTS "users" (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255),
                full_name VARCHAR(255),
                is_active BOOLEAN DEFAULT TRUE,
                is_superuser BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "projects": """
            CREATE TABLE IF NOT EXISTS "projects" (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                owner_id INTEGER REFERENCES "users"(id),
                status VARCHAR(50) DEFAULT 'active',
                config JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "agents": """
            CREATE TABLE IF NOT EXISTS "agents" (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                type VARCHAR(100),
                config JSONB DEFAULT '{}',
                project_id INTEGER REFERENCES "projects"(id),
                owner_id INTEGER REFERENCES "users"(id),
                status VARCHAR(50) DEFAULT 'inactive',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "datasets": """
            CREATE TABLE IF NOT EXISTS "datasets" (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                project_id INTEGER REFERENCES "projects"(id),
                data_type VARCHAR(50),
                schema JSONB DEFAULT '{}',
                source_path VARCHAR(500),
                size_bytes BIGINT DEFAULT 0,
                row_count INTEGER DEFAULT 0,
                owner_id INTEGER REFERENCES "users"(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "models": """
            CREATE TABLE IF NOT EXISTS "models" (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                version VARCHAR(50),
                model_type VARCHAR(100),
                framework VARCHAR(100),
                project_id INTEGER REFERENCES "projects"(id),
                owner_id INTEGER REFERENCES "users"(id),
                status VARCHAR(50) DEFAULT 'draft',
                config JSONB DEFAULT '{}',
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "experiments": """
            CREATE TABLE IF NOT EXISTS "experiments" (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                project_id INTEGER REFERENCES "projects"(id),
                model_id INTEGER REFERENCES "models"(id),
                status VARCHAR(50) DEFAULT 'running',
                config JSONB DEFAULT '{}',
                metrics JSONB DEFAULT '{}',
                owner_id INTEGER REFERENCES "users"(id),
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "tasks": """
            CREATE TABLE IF NOT EXISTS "tasks" (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                task_type VARCHAR(100),
                project_id INTEGER REFERENCES "projects"(id),
                agent_id INTEGER REFERENCES "agents"(id),
                status VARCHAR(50) DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                input_data JSONB DEFAULT '{}',
                output_data JSONB DEFAULT '{}',
                error_message TEXT,
                owner_id INTEGER REFERENCES "users"(id),
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "knowledge_graphs": """
            CREATE TABLE IF NOT EXISTS "knowledge_graphs" (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                project_id INTEGER REFERENCES "projects"(id),
                owner_id INTEGER REFERENCES "users"(id),
                graph_data JSONB DEFAULT '{}',
                schema_def JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "audit_logs": """
            CREATE TABLE IF NOT EXISTS "audit_logs" (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES "users"(id),
                action VARCHAR(100),
                resource_type VARCHAR(100),
                resource_id VARCHAR(255),
                details JSONB DEFAULT '{}',
                ip_address VARCHAR(45),
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
    }
    
    # 创建索引
    indexes = [
        'CREATE INDEX IF NOT EXISTS "idx_users_email" ON "users" (email)',
        'CREATE INDEX IF NOT EXISTS "idx_projects_owner" ON "projects" (owner_id)',
        'CREATE INDEX IF NOT EXISTS "idx_agents_project" ON "agents" (project_id)',
        'CREATE INDEX IF NOT EXISTS "idx_datasets_project" ON "datasets" (project_id)',
        'CREATE INDEX IF NOT EXISTS "idx_models_project" ON "models" (project_id)',
        'CREATE INDEX IF NOT EXISTS "idx_experiments_project" ON "experiments" (project_id)',
        'CREATE INDEX IF NOT EXISTS "idx_tasks_project" ON "tasks" (project_id)',
        'CREATE INDEX IF NOT EXISTS "idx_tasks_status" ON "tasks" (status)',
        'CREATE INDEX IF NOT EXISTS "idx_audit_logs_user" ON "audit_logs" (user_id)',
        'CREATE INDEX IF NOT EXISTS "idx_audit_logs_created" ON "audit_logs" (created_at)',
    ]
    
    with pool.get_write_connection() as conn:
        cursor = conn.cursor()
        
        # 创建表
        for table_name, create_sql in tables.items():
            print(f"Creating table: {table_name}")
            cursor.execute(create_sql)
        
        # 创建索引
        for idx_sql in indexes:
            print(f"Creating index: {idx_sql}")
            cursor.execute(idx_sql)
        
        conn.commit()
    
    print("✅ All tables and indexes created")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Migrate SQLite to PostgreSQL')
    parser.add_argument(
        '--sqlite', 
        type=str, 
        default='data/ai_platform.db',
        help='SQLite database path'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--tables',
        type=str,
        help='Comma-separated list of tables to migrate'
    )
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='Drop and recreate tables'
    )
    parser.add_argument(
        '--create-only',
        action='store_true',
        help='Only create schema without migrating data'
    )
    
    args = parser.parse_args()
    
    # 解析表列表
    tables = None
    if args.tables:
        tables = [t.strip() for t in args.tables.split(',')]
    
    # 创建PostgreSQL连接池
    pg_config = PoolConfig(
        min_connections=5,
        max_connections=50,
        host='/tmp',
        database='aiplatform',
        user='yubao'
    )
    
    pool = PostgresPool(pg_config)
    
    try:
        if args.create_only:
            # 只创建表结构
            create_tables_postgres(pool)
        else:
            # 执行迁移
            migrator = DataMigrator(
                sqlite_path=args.sqlite,
                pg_pool=pool,
                batch_size=1000,
                dry_run=args.dry_run
            )
            migrator.migrate_all(tables=tables, recreate=args.recreate)
    
    finally:
        pool.close()


if __name__ == '__main__':
    main()
