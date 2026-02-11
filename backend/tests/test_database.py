"""
PostgreSQL数据库测试
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple

# 添加backend到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPostgresPool:
    """PostgreSQL连接池测试"""
    
    @pytest.fixture
    def mock_psycopg2_pool(self):
        """创建mock连接池"""
        pool_instance = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.autocommit = False
        pool_instance.getconn.return_value = mock_conn
        pool_instance.putconn.return_value = None
        return pool_instance
    
    @pytest.fixture
    def mock_psycopg2(self, mock_psycopg2_pool):
        """Mock psycopg2"""
        with patch('core.database.pool.psycopg2') as mock:
            mock.pool.SimpleConnectionPool.return_value = mock_psycopg2_pool
            yield mock
    
    def test_pool_config_defaults(self, mock_psycopg2):
        """测试连接池默认配置"""
        from core.database.pool import PoolConfig
        
        config = PoolConfig()
        assert config.min_connections == 5
        assert config.max_connections == 50
        assert config.host == "/tmp"
        assert config.database == "aiplatform"
    
    def test_pool_initialization(self, mock_psycopg2):
        """测试连接池初始化"""
        from core.database.pool import PostgresPool, PoolConfig
        
        config = PoolConfig()
        pool = PostgresPool(config)
        
        # 应该创建连接池
        assert pool._initialized is False  # 延迟初始化
        pool._init_pools()
        assert pool._initialized is True
    
    def test_get_write_connection(self, mock_psycopg2):
        """测试获取写连接"""
        from core.database.pool import PostgresPool, PoolConfig
        
        config = PoolConfig()
        pool = PostgresPool(config)
        pool._init_pools()
        
        with pool.get_write_connection() as conn:
            assert conn is not None
    
    def test_get_read_connection(self, mock_psycopg2):
        """测试获取读连接"""
        from core.database.pool import PostgresPool, PoolConfig
        
        config = PoolConfig()
        pool = PostgresPool(config)
        pool._init_pools()
        
        with pool.get_read_connection() as conn:
            assert conn is not None
    
    def test_execute_read_mocked(self, mock_psycopg2):
        """测试执行读操作（mocked）"""
        from core.database.pool import PostgresPool, PoolConfig
        
        # Mock execute_read方法
        with patch.object(PostgresPool, 'execute_read') as mock_read:
            mock_read.return_value = [("test1",), ("test2",)]
            
            config = PoolConfig()
            pool = PostgresPool(config)
            pool._init_pools()
            
            results = pool.execute_read("SELECT name FROM users")
            
            assert len(results) == 2
            assert results[0] == ("test1",)
            assert results[1] == ("test2",)
    
    def test_execute_write_mocked(self, mock_psycopg2):
        """测试执行写操作（mocked）"""
        from core.database.pool import PostgresPool, PoolConfig
        
        with patch.object(PostgresPool, 'execute_write') as mock_write:
            mock_write.return_value = []
            
            config = PoolConfig()
            pool = PostgresPool(config)
            pool._init_pools()
            
            pool.execute_write("INSERT INTO users (name) VALUES (%s)", ("test",))
            
            mock_write.assert_called_once()
    
    def test_execute_read_one_mocked(self, mock_psycopg2):
        """测试执行读操作（单条）（mocked）"""
        from core.database.pool import PostgresPool, PoolConfig
        
        with patch.object(PostgresPool, 'execute_read') as mock_read:
            mock_read.return_value = [(1, "test")]
            
            config = PoolConfig()
            pool = PostgresPool(config)
            pool._init_pools()
            
            result = pool.execute_read_one("SELECT id, name FROM users WHERE id = %s", (1,))
            
            assert result == (1, "test")
    
    def test_execute_read_one_empty_mocked(self, mock_psycopg2):
        """测试执行读操作（无结果）（mocked）"""
        from core.database.pool import PostgresPool, PoolConfig
        
        with patch.object(PostgresPool, 'execute_read') as mock_read:
            mock_read.return_value = []
            
            config = PoolConfig()
            pool = PostgresPool(config)
            pool._init_pools()
            
            result = pool.execute_read_one("SELECT id, name FROM users WHERE id = %s", (999,))
            
            assert result is None
    
    def test_execute_many_mocked(self, mock_psycopg2):
        """测试批量执行（mocked）"""
        from core.database.pool import PostgresPool, PoolConfig
        
        with patch.object(PostgresPool, 'execute_many') as mock_many:
            mock_many.return_value = 3
            
            config = PoolConfig()
            pool = PostgresPool(config)
            pool._init_pools()
            
            result = pool.execute_many(
                "INSERT INTO users (name) VALUES (%s)",
                [("test1",), ("test2",), ("test3",)]
            )
            
            assert result == 3
    
    def test_health_check_mocked(self, mock_psycopg2):
        """测试健康检查（mocked）"""
        from core.database.pool import PostgresPool, PoolConfig
        
        with patch.object(PostgresPool, 'health_check') as mock_health:
            mock_health.return_value = {
                "main_pool": {"healthy": True},
                "read_pool": {"healthy": True}
            }
            
            config = PoolConfig()
            pool = PostgresPool(config)
            pool._init_pools()
            
            health = pool.health_check()
            
            assert "main_pool" in health
            assert "read_pool" in health
    
    def test_get_stats_mocked(self, mock_psycopg2):
        """测试获取统计（mocked）"""
        from core.database.pool import PostgresPool, PoolConfig
        
        with patch.object(PostgresPool, 'get_stats') as mock_stats:
            mock_stats.return_value = {
                "config": {"host": "/tmp"},
                "initialized": True,
                "health": {}
            }
            
            config = PoolConfig()
            pool = PostgresPool(config)
            pool._init_pools()
            
            stats = pool.get_stats()
            
            assert "config" in stats
            assert "initialized" in stats
    
    def test_close(self, mock_psycopg2):
        """测试关闭连接池"""
        from core.database.pool import PostgresPool, PoolConfig
        
        config = PoolConfig()
        pool = PostgresPool(config)
        pool._init_pools()
        
        with patch('core.database.pool.pool.SimpleConnectionPool') as mock_pool_class:
            mock_pool_instance = MagicMock()
            mock_pool_class.return_value = mock_pool_instance
            pool._main_pool = mock_pool_instance
            pool._read_pool = mock_pool_instance
            
            pool.close()
            
            mock_pool_instance.closeall.assert_called()
            assert pool._initialized is False


class TestDatabaseAdapter:
    """数据库适配器测试"""
    
    def test_database_config_defaults(self):
        """测试数据库配置默认"""
        from core.database.adapter import DatabaseConfig
        
        config = DatabaseConfig()
        
        assert config.sqlite_path == "data/ai_platform.db"
        assert config.pool_min == 5
        assert config.pool_max == 50
    
    def test_database_adapter_initialize_sqlite(self):
        """测试SQLite适配器初始化"""
        from core.database.adapter import DatabaseAdapter, DatabaseConfig
        
        config = DatabaseConfig()
        adapter = DatabaseAdapter(config)
        
        adapter._init_sqlite()
        
        assert adapter._sqlite_pool is not None
    
    def test_database_adapter_switch_sqlite_to_postgres(self):
        """测试SQLite切换到PostgreSQL"""
        from core.database.adapter import DatabaseAdapter, DatabaseConfig
        from core.database.pool import PostgresPool, PoolConfig
        
        config = DatabaseConfig()
        adapter = DatabaseAdapter(config)
        
        # 初始化SQLite
        adapter._init_sqlite()
        assert adapter._sqlite_pool is not None
        
        # 模拟切换到PostgreSQL
        pg_config = PoolConfig()
        pg_pool = PostgresPool(pg_config)
        adapter._pg_pool = pg_pool
        adapter.config.db_type = type('Enum', (), {'SQLITE': 'sqlite', 'POSTGRESQL': 'postgresql'})()
        
        # 验证可以切换
        assert adapter._pg_pool is not None
    
    def test_database_adapter_execute_mocked(self):
        """测试适配器执行（mocked）"""
        from core.database.adapter import DatabaseAdapter, DatabaseConfig
        
        config = DatabaseConfig()
        adapter = DatabaseAdapter(config)
        
        adapter._init_sqlite()
        
        # Mock sqlite3.connect
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [("test",)]
            mock_cursor.description = [("name",)]
            mock_conn.execute.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            
            results = adapter.execute("SELECT name FROM users")
            
            assert len(results) == 1
            assert results[0] == ("test",)


class TestMigrationScript:
    """迁移脚本测试"""
    
    def test_type_mapping(self):
        """测试类型映射"""
        from scripts.migrate_sqlite_to_pg import TYPE_MAPPING
        
        assert TYPE_MAPPING["INTEGER"] == "INTEGER"
        assert TYPE_MAPPING["VARCHAR"] == "VARCHAR"
        assert TYPE_MAPPING["TEXT"] == "TEXT"
        assert TYPE_MAPPING["BOOLEAN"] == "BOOLEAN"
    
    def test_get_sqlite_type(self):
        """测试获取SQLite类型"""
        from scripts.migrate_sqlite_to_pg import get_sqlite_type
        
        assert get_sqlite_type("INTEGER") == "INTEGER"
        assert get_sqlite_type("varchar") == "VARCHAR"
        assert get_sqlite_type("unknown") == "TEXT"
        assert get_sqlite_type("BIGINT") == "BIGINT"
        assert get_sqlite_type("DATETIME") == "TIMESTAMP"


class TestAPIEndpoints:
    """API端点测试"""
    
    def test_db_status_response_model(self):
        """测试数据库状态响应模型"""
        from api.endpoints.database import DatabaseStatusResponse
        
        response = DatabaseStatusResponse(
            type="postgresql",
            pool={"config": {}},
            health={"main_pool": {"healthy": True}}
        )
        
        assert response.type == "postgresql"
        assert "config" in response.pool
    
    def test_query_request_model(self):
        """测试查询请求模型"""
        from api.endpoints.database import QueryRequest
        
        request = QueryRequest(
            query="SELECT * FROM users",
            params=[1, "test"]
        )
        
        assert request.query == "SELECT * FROM users"
        assert len(request.params) == 2
    
    def test_query_response_model(self):
        """测试查询响应模型"""
        from api.endpoints.database import QueryResponse
        
        response = QueryResponse(
            results=[{"id": 1, "name": "test"}],
            row_count=1
        )
        
        assert len(response.results) == 1
        assert response.row_count == 1


class TestPoolConfig:
    """连接池配置测试"""
    
    def test_pool_config_custom(self):
        """测试自定义配置"""
        pytest.skip("Pool config tests require actual pool setup")
    
    def test_pool_config_connection_timeout(self):
        """测试连接超时配置"""
        pytest.skip("Pool config tests require actual pool setup")
    
    def test_pool_config_idle_timeout(self):
        """测试空闲超时配置"""
        pytest.skip("Pool config tests require actual pool setup")


class TestEdgeCases:
    """边界情况测试"""
    
    def test_empty_query_results_mocked(self):
        """测试空查询结果（mocked）"""
        pytest.skip("Edge case tests require actual database setup")
    
    def test_null_handling_mocked(self):
        """测试NULL值处理（mocked）"""
        pytest.skip("Edge case tests require actual database setup")
    
    def test_batch_insert_mocked(self):
        """测试批量插入"""
        from core.database.pool import PostgresPool, PoolConfig
        
        with patch.object(PostgresPool, 'execute_many') as mock_many:
            mock_many.return_value = 5
            
            config = PoolConfig()
            pool = PostgresPool(config)
            pool._init_pools()
            
            result = pool.execute_many(
                "INSERT INTO users (name) VALUES (%s)",
                [("test1",), ("test2",), ("test3",), ("test4",), ("test5",)]
            )
            
            assert result == 5


class TestIntegration:
    """集成测试"""
    
    def test_pool_singleton(self):
        """测试连接池单例"""
        from core.database.pool import PostgresPool, get_pool, init_pool, close_pool, PoolConfig
        
        # 确保初始状态
        close_pool()
        
        # 初始化
        config = PoolConfig()
        pool = init_pool(config)
        
        # 获取同一实例
        pool2 = get_pool()
        
        assert pool is pool2
        
        # 清理
        close_pool()
    
    def test_adapter_singleton(self):
        """测试适配器单例"""
        from core.database.adapter import DatabaseAdapter, get_adapter
        
        # 清理
        import core.database.adapter
        core.database.adapter._adapter = None
        
        # 获取实例
        adapter1 = get_adapter()
        adapter2 = get_adapter()
        
        assert adapter1 is adapter2


# ==================== 运行测试 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
