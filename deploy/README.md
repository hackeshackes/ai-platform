# Docker 部署指南

## 快速启动

### 方式一：启动核心服务

```bash
# 启动数据库和缓存
docker-compose up -d postgres redis

# 验证服务
docker-compose ps
```

### 方式二：启动所有服务（包括ML工具）

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

## 服务列表

| 服务 | 端口 | 地址 | 说明 |
|------|------|------|------|
| PostgreSQL | 5432 | localhost:5432 | 主数据库 |
| Redis | 6379 | localhost:6379 | 缓存/消息队列 |
| MLflow | 5000 | localhost:5000 | 实验跟踪 |
| Ollama | 11434 | localhost:11434 | 本地推理 |
| Label Studio | 8080 | localhost:8080 | 数据标注 |
| 后端API | 8000 | localhost:8000 | API服务 |
| 前端 | 3000 | localhost:3000 | Web界面 |

## 环境变量

创建 `.env` 文件：

```bash
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_platform

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT
SECRET_KEY=your-secret-key

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
```

## GPU 支持

如果需要GPU加速，确保安装NVIDIA Container Toolkit：

```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 资源管理

### 查看资源使用

```bash
docker stats
```

### 清理资源

```bash
# 清理未使用的镜像
docker system prune -a

# 清理所有数据卷
docker-compose down -v
```

## 故障排查

### PostgreSQL 连接失败

```bash
# 检查日志
docker-compose logs postgres

# 等待健康检查
docker-compose wait postgres
```

### GPU 不可用

```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查Docker GPU支持
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

## 生产部署建议

1. **使用外部数据库**
   ```yaml
   postgres:
     # 使用外部托管的PostgreSQL
     external: true
   ```

2. **配置资源限制**
   ```yaml
   services:
     backend:
       deploy:
         resources:
           limits:
             cpus: '2'
             memory: 2G
   ```

3. **启用HTTPS**
   使用Nginx反向代理 + Let's Encrypt证书
