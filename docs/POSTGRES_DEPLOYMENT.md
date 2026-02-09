# AI Platform v2.1 PostgreSQL 生产环境部署

## 快速启动

### 1. 启动数据库服务

```bash
# 方式一: 使用Docker Compose
docker-compose -f docker-compose.postgres.yml up -d

# 方式二: 手动启动
# 确保 PostgreSQL 15+ 和 Redis 7+ 已安装
```

### 2. 初始化数据库

```bash
# 方式一: Docker初始化 (自动执行)
# init.sql 会在容器首次启动时自动执行

# 方式二: 手动初始化
psql -U aiplatform -d ai_platform -f docker/postgres/init.sql
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 设置数据库密码
```

### 4. 迁移数据 (从SQLite)

```bash
python scripts/migrate_to_postgres.py
```

### 5. 启动应用

```bash
# 开发模式
python backend/main.py

# 或使用Docker
docker-compose up -d
```

## Docker部署

```bash
# 构建并启动所有服务
docker-compose -f docker-compose.yml up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| PG_HOST | PostgreSQL主机 | localhost |
| PG_PORT | PostgreSQL端口 | 5432 |
| PG_DB | 数据库名 | ai_platform |
| PG_USER | 用户名 | aiplatform |
| PG_PASSWORD | 密码 | aiplatform123 |
| REDIS_HOST | Redis主机 | localhost |
| REDIS_PORT | Redis端口 | 6379 |

## 健康检查

```bash
# 检查PostgreSQL
psql -U aiplatform -d ai_platform -c "SELECT 1"

# 检查Redis
redis-cli ping

# 检查应用
curl http://localhost:8000/health
```
