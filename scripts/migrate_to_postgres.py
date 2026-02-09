#!/usr/bin/env python3
"""
PostgreSQL 迁移脚本 - 从SQLite迁移到PostgreSQL
"""
import sqlite3
import psycopg2
import json
import os
from datetime import datetime

# 配置
SQLITE_DB = "ai_platform.db"
PG_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "dbname": os.getenv("PG_DB", "ai_platform"),
    "user": os.getenv("PG_USER", "aiplatform"),
    "password": os.getenv("PG_PASSWORD", "aiplatform123")
}

def migrate_users(sqlite_conn, pg_conn):
    """迁移用户"""
    cursor = sqlite_conn.cursor()
    pg_cursor = pg_conn.cursor()
    
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    
    for user in users:
        pg_cursor.execute("""
            INSERT INTO users (id, username, email, password_hash, role, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, user)
    
    pg_conn.commit()
    print(f"✅ 迁移 {len(users)} 用户")

def migrate_projects(sqlite_conn, pg_conn):
    """迁移项目"""
    cursor = sqlite_conn.cursor()
    pg_cursor = pg_conn.cursor()
    
    cursor.execute("SELECT * FROM projects")
    projects = cursor.fetchall()
    
    for project in projects:
        config = project[5]  # config字段
        if isinstance(config, str):
            config = json.loads(config)
        
        pg_cursor.execute("""
            INSERT INTO projects (id, name, description, owner_id, status, config, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (project[0], project[1], project[2], project[3], project[4], json.dumps(config), project[6], project[7]))
    
    pg_conn.commit()
    print(f"✅ 迁移 {len(projects)} 项目")

def migrate_experiments(sqlite_conn, pg_conn):
    """迁移实验"""
    cursor = sqlite_conn.cursor()
    pg_cursor = pg_conn.cursor()
    
    cursor.execute("SELECT * FROM experiments")
    experiments = cursor.fetchall()
    
    for exp in experiments:
        config = exp[6] if len(exp) > 6 else None
        metrics = exp[7] if len(exp) > 7 else None
        
        if isinstance(config, str):
            config = json.loads(config)
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        
        pg_cursor.execute("""
            INSERT INTO experiments (id, name, project_id, base_model, task_type, status, config, metrics, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (exp[0], exp[1], exp[2], exp[3], exp[4], exp[5], json.dumps(config) if config else None, json.dumps(metrics) if metrics else None, exp[8], exp[9]))
    
    pg_conn.commit()
    print(f"✅ 迁移 {len(experiments)} 实验")

def migrate_datasets(sqlite_conn, pg_conn):
    """迁移数据集"""
    cursor = sqlite_conn.cursor()
    pg_cursor = pg_conn.cursor()
    
    cursor.execute("SELECT * FROM datasets")
    datasets = cursor.fetchall()
    
    for ds in datasets:
        quality_report = ds[9] if len(ds) > 9 else None
        if isinstance(quality_report, str):
            quality_report = json.loads(quality_report)
        
        pg_cursor.execute("""
            INSERT INTO datasets (id, name, project_id, file_path, size, format, version, status, quality_report, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (ds[0], ds[1], ds[2], ds[3], ds[4], ds[5], ds[6], ds[7], json.dumps(quality_report) if quality_report else None, ds[10]))
    
    pg_conn.commit()
    print(f"✅ 迁移 {len(datasets)} 数据集")

def migrate_models(sqlite_conn, pg_conn):
    """迁移模型"""
    cursor = sqlite_conn.cursor()
    pg_cursor = pg_conn.cursor()
    
    cursor.execute("SELECT * FROM models")
    models = cursor.fetchall()
    
    for model in models:
        metrics = model[7] if len(model) > 7 else None
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        
        pg_cursor.execute("""
            INSERT INTO models (id, name, project_id, version, framework, file_path, size, metrics, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (model[0], model[1], model[2], model[3], model[4], model[5], model[6], json.dumps(metrics) if metrics else None, model[8]))
    
    pg_conn.commit()
    print(f"✅ 迁移 {len(models)} 模型")

def main():
    """主迁移函数"""
    print("=" * 50)
    print("AI Platform PostgreSQL 迁移")
    print("=" * 50)
    
    # 检查SQLite数据库
    if not os.path.exists(SQLITE_DB):
        print(f"❌ SQLite数据库 {SQLITE_DB} 不存在")
        return
    
    # 连接SQLite
    sqlite_conn = sqlite3.connect(SQLITE_DB)
    print(f"✅ 连接到 SQLite: {SQLITE_DB}")
    
    try:
        # 连接PostgreSQL
        pg_conn = psycopg2.connect(**PG_CONFIG)
        print(f"✅ 连接到 PostgreSQL: {PG_CONFIG['dbname']}")
        
        # 执行迁移
        print("\n开始迁移...")
        migrate_users(sqlite_conn, pg_conn)
        migrate_projects(sqlite_conn, pg_conn)
        migrate_experiments(sqlite_conn, pg_conn)
        migrate_datasets(sqlite_conn, pg_conn)
        migrate_models(sqlite_conn, pg_conn)
        
        print("\n" + "=" * 50)
        print("✅ 迁移完成!")
        print("=" * 50)
        
        pg_conn.close()
        
    except Exception as e:
        print(f"❌ 迁移失败: {e}")
        raise
    finally:
        sqlite_conn.close()

if __name__ == "__main__":
    main()
