# PostgreSQL迁移指南

本文档描述AI Platform从SQLite迁移到PostgreSQL的完整过程。

## 目录

1. [概述](#概述)
2. [迁移前提](#迁移前提)
3. [迁移步骤](#迁移步骤)
4. [配置说明](#配置说明)
5. [API端点](#api端点)
6. [验证与测试](#验证与测试)
7. [回滚策略](#回滚策略)
8. [性能优化](#性能优化)

## 概述

### 迁移目标

- 从SQLite迁移到PostgreSQL
- 实现连接池管理
- 支持读写分离
- 保持SQLite兼容（可选切换）

### 迁移收益

- **并发支持**: PostgreSQL支持多连接并发访问
- **可靠性**: 更好的数据完整性和事务支持
- **扩展性**: 易于水平扩展
- **性能**: 连接池减少连接开销

## 迁移前提

### 系统要求

- PostgreSQL 13+
- Python 3.8+
- psycopg2-binary 2.9+

### 准备工作

1. **安装PostgreSQL**

```bash
# macOS
brew install postgresql@15

# 启动PostgreSQL
brew services start postgresql@15

# 创建数据库
createdb aiplatform
```

2. **安装Python依赖**

```bash
pip install psycopg2-binary
```

3. **备份SQLite数据**

```bash
cp data/ai_platform.db data/ai_platform.db.backup
```

## 迁移步骤

### 步骤1: 创建PostgreSQL表结构

```bash
# 创建表结构（不迁移数据）
python scripts/migrate_sqlite_to_pg.py --create-only
```

### 步骤2: 执行数据迁移

```bash
# 完整迁移
python scripts/migrate_sqlite_to_pg.py --sqlite data/ai_platform.db

# 模拟运行（不实际执行）
python scripts/migrate_sqlite_to_pg.py --dry-run --sqlite data/ai_platform.db

# 只迁移指定表
python scripts/migrate_sqlite_to_pg.py --tables users,projects,agents

# 重新创建表（删除现有数据）
python scripts/migrate_sqlite_to_pg.py --recreate --tables users
```

### 步骤3: 更新配置文件

```bash
# 设置环境变量
export PGHOST=/tmp
export PGDATABASE=aiplatform
export PGUSER=yubao

# 启用PostgreSQL
export USE_POSTGRES=true
```

### 步骤4: 重启应用

```bash
# 重启后端服务
python -m uvicorn main:app --reload
```

## 配置说明

### 连接池配置

```python
from core.database.pool import PoolConfig, PostgresPool

config = PoolConfig(
    min_connections=5,      # 最小连接数
    max_connections=50,     # 最大连接数
    host="/tmp",            # PostgreSQL socket目录
    port=5432,              # 端口
    database="aiplatform",  # 数据库名
    user="yubao",           # 用户名
    password="",            # 密码
    connection_timeout=30   # 连接超时(秒)
)

pool = PostgresPool(config)
```

### 环境变量配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| PGHOST | PostgreSQL主机 | /tmp |
| PGPORT | PostgreSQL端口 | 5432 |
| PGDATABASE | 数据库名 | aiplatform |
| PGUSER | 用户名 | yubao |
| PGPASSWORD | 密码 | 空 |
| PGREAD_HOST | 只读副本主机 | 同PGHOST |
| USE_POSTGRES | 启用PostgreSQL | false |

### 数据库适配器配置

```python
from core.database.adapter import DatabaseAdapter, DatabaseConfig, DatabaseType

# 使用SQLite
config = DatabaseConfig(db_type=DatabaseType.SQLITE)

# 使用PostgreSQL
config = DatabaseConfig(db_type=DatabaseType.POSTGRESQL)

adapter = DatabaseAdapter(config)
adapter.initialize()
```

## API端点

### 数据库管理

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/db/status` | GET | 获取数据库状态 |
| `/api/v1/db/health` | GET | 健康检查 |
| `/api/v1/db/stats` | GET | 统计信息 |
| `/api/v1/db/tables` | GET | 列出所有表 |
| `/api/v1/db/tables/{name}` | GET | 获取表信息 |

### 业务数据查询

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/db/users` | GET | 列出用户 |
| `/api/v1/db/projects` | GET | 列出项目 |
| `/api/v1/db/agents` | GET | 列出Agents |
| `/api/v1/db/datasets` | GET | 列出数据集 |
| `/api/v1/db/models` | GET | 列出模型 |
| `/api/v1/db/experiments` | GET | 列出实验 |
| `/api/v1/db/tasks` | GET | 列出任务 |
| `/api/v1/db/audit-logs` | GET | 列出审计日志 |

### 示例请求

```bash
# 获取数据库状态
curl http://localhost:8000/api/v1/db/status

# 获取数据库健康
curl http://localhost:8000/api/v1/db/health

# 列出项目
curl http://localhost:8000/api/v1/db/projects?limit=10&offset=0

# 筛选任务
curl http://localhost:8000/api/v1/db/tasks?status=running&project_id=1
```

## 验证与测试

### 1. 测试数据库连接

```bash
# 检查健康状态
curl http://localhost:8000/api/v1/db/health

# 预期响应
{
  "status": "healthy",
  "main_pool": true,
  "read_pool": true
}
```

### 2. 验证数据迁移

```bash
# 验证用户数量
psql -h /tmp -d aiplatform -c "SELECT COUNT(*) FROM users;"

# 验证项目数量
psql -h /tmp -d aiplatform -c "SELECT COUNT(*) FROM projects;"
```

### 3. 运行测试

```bash
# 运行数据库测试
pytest tests/test_database.py -v

# 运行覆盖率测试
pytest tests/test_database.py --cov=core.database --cov-report=html
```

### 4. 性能测试

```bash
# 使用wrk进行压力测试
wrk -t4 -c100 -d30s http://localhost:8000/api/v1/db/projects
```

## 回滚策略

### 回滚到SQLite

1. 修改环境变量

```bash
export USE_POSTGRES=false
unset PGHOST
```

2. 重启应用

```bash
# 重启服务
pkill -f uvicorn
python -m uvicorn main:app --reload
```

3. 从PostgreSQL导出数据（可选）

```bash
# 导出数据
pg_dump -h /tmp -d aiplatform > aiplatform_backup.sql

# 导入到SQLite
sqlite3 data/ai_platform.db < aiplatform_backup.sql
```

### 使用数据库适配器

```python
from core.database.adapter import DatabaseAdapter, DatabaseType

adapter = DatabaseAdapter.get_instance()

# 切换到SQLite
adapter.switch_database(DatabaseType.SQLITE)

# 切换到PostgreSQL
adapter.switch_database(DatabaseType.POSTGRESQL)
```

## 性能优化

### 连接池配置

```python
# 生产环境推荐配置
config = PoolConfig(
    min_connections=10,
    max_connections=100,
    connection_timeout=30,
    idle_timeout=300  # 5分钟空闲超时
)
```

### 数据库优化

```sql
-- 创建常用索引
CREATE INDEX idx_tasks_status ON tasks (status);
CREATE INDEX idx_tasks_project ON tasks (project_id);
CREATE INDEX idx_experiments_project ON experiments (project_id);

-- 分析表以优化查询
ANALYZE users;
ANALYZE projects;
ANALYZE tasks;
```

### 监控连接池

```python
# 获取连接池状态
stats = pool.get_stats()

# 监控指标
print(f"活动连接: {stats['active']}")
print(f"空闲连接: {stats['idle']}")
print(f"总连接数: {stats['created']}")
print(f"利用率: {stats['utilization']}")
```

## 常见问题

### Q1: 连接池耗尽

**问题**: `Connection pool exhausted`错误

**解决方案**:
```python
# 增加连接池大小
config = PoolConfig(max_connections=100)

# 或减少连接使用时间
with pool.get_connection() as conn:
    # 尽快完成操作
    pass
```

### Q2: PostgreSQL连接失败

**问题**: 无法连接到PostgreSQL

**解决方案**:
```bash
# 检查PostgreSQL服务
pg_isready -h /tmp

# 检查数据库是否存在
psql -h /tmp -l

# 检查权限
psql -h /tmp -d aiplatform -c "SELECT 1;"
```

### Q3: 数据迁移失败

**问题**: 迁移过程中出错

**解决方案**:
```bash
# 使用dry-run查看问题
python scripts/migrate_sqlite_to_pg.py --dry-run

# 检查SQLite数据
sqlite3 data/ai_platform.db "PRAGMA integrity_check;"

# 手动修复问题后重新迁移
python scripts/migrate_sqlite_to_pg.py --recreate --tables problematic_table
```

## 迁移检查清单

- [ ] PostgreSQL已安装并运行
- [ ] 数据库和用户已创建
- [ ] Python依赖已安装
- [ ] SQLite数据已备份
- [ ] 表结构已创建
- [ ] 数据已迁移
- [ ] 应用配置已更新
- [ ] 健康检查通过
- [ ] 功能测试通过
- [ ] 回滚方案已测试

## 相关文件

- `core/database/pool.py` - PostgreSQL连接池
- `core/database/config.py` - 数据库配置
- `core/database/adapter.py` - SQLite适配器
- `scripts/migrate_sqlite_to_pg.py` - 迁移脚本
- `api/endpoints/database.py` - API端点
- `tests/test_database.py` - 测试用例
