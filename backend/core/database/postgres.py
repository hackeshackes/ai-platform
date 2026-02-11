"""
PostgreSQL数据库连接
"""
import psycopg2
from psycopg2 import pool
from typing import Optional
import os

# 连接池
_pg_pool: Optional[pool.SimpleConnectionPool] = None

def get_postgres_pool(min_conn: int = 5, max_conn: int = 20) -> pool.SimpleConnectionPool:
    """获取PostgreSQL连接池"""
    global _pg_pool
    
    if _pg_pool is None:
        _pg_pool = pool.SimpleConnectionPool(
            min_conn,
            max_conn,
            host='/tmp',
            database='aiplatform',
            user=os.getenv('USER', 'yubao')
        )
    
    return _pg_pool

def get_connection():
    """获取连接"""
    pool = get_postgres_pool()
    return pool.getconn()

def put_connection(conn):
    """归还连接"""
    pool = get_postgres_pool()
    pool.putconn(conn)

def execute_query(query: str, params: tuple = None) -> list:
    """执行查询"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.fetchall()
    finally:
        put_connection(conn)

def execute_one(query: str, params: tuple = None) -> tuple:
    """执行查询(单条结果)"""
    results = execute_query(query, params)
    return results[0] if results else None
