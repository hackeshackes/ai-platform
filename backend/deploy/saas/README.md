# AI Platform SaaS一键部署系统

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg">
  <img src="https://img.shields.io/badge/python-3.11+-green.svg">
  <img src="https://img.shields.io/badge/docker-ready-blue.svg">
</p>

## 简介

🚀 **3分钟内完成SaaS应用部署** - 自动化部署系统，支持一键部署、自动扩容、监控告警、CDN配置。

## 核心功能

### 1. 一键部署 (deployer.py)
- Agent/Pipeline一键创建
- 自动域名配置
- 自动SSL证书申请
- 支持Docker Compose、Kubernetes、Serverless

### 2. 资源管理 (resource_manager.py)
- 自动扩容/缩容
- 负载均衡
- 健康检查
- 资源监控

### 3. 监控告警 (monitor.py)
- 实时指标收集
- 多级别告警
- 多种通知渠道
- 日志收集分析

### 4. CDN管理 (cdn_manager.py)
- 自动CDN配置
- 缓存策略管理
- 带宽优化
- 缓存预热/清除

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置系统

编辑 `config.yaml`：
```yaml
app:
  host: "0.0.0.0"
  port: 8080

docker:
  socket: "/var/run/docker.sock"
```

### 3. 启动服务

```bash
# 开发模式
python api.py

# Docker模式
docker-compose up -d
```

### 4. 使用API

#### 一键部署

```bash
curl -X POST http://localhost:8080/api/v1/deploy/one-click \
  -H "Content-Type: application/json" \
  -d '{
    "type": "agent",
    "name": "my-agent",
    "config": {
      "image": "nginx:latest",
      "replicas": 2,
      "cpu_limit": "1000m",
      "memory_limit": "1Gi",
      "domain": "my-agent.example.com",
      "ssl_enabled": true
    }
  }'
```

#### 查询状态

```bash
curl http://localhost:8080/api/v1/deploy/{id}/status
```

#### 扩容

```bash
curl -X POST http://localhost:8080/api/v1/deploy/{id}/scale \
  -H "Content-Type: application/json" \
  -d '{"replicas": 3}'
```

## API参考

### 部署接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/deploy/one-click` | 一键部署 |
| GET | `/api/v1/deploy/{id}/status` | 查询部署状态 |
| POST | `/api/v1/deploy/{id}/scale` | 扩容 |
| DELETE | `/api/v1/deploy/{id}` | 删除部署 |
| GET | `/api/v1/deploy/list` | 列出所有部署 |

### 监控接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/monitor/status` | 监控状态 |
| GET | `/api/v1/monitor/metrics` | 指标数据 |
| GET | `/api/v1/monitor/alerts` | 告警列表 |
| POST | `/api/v1/monitor/alerts/{id}/ack` | 确认告警 |
| GET | `/api/v1/monitor/logs` | 日志查询 |

### CDN接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/cdn/configure` | 配置CDN |
| POST | `/api/v1/cdn/purge` | 清除缓存 |
| POST | `/api/v1/cdn/warm` | 预热缓存 |
| GET | `/api/v1/cdn/status` | CDN状态 |

## 部署流程

```
用户点击"一键部署"
    ↓
系统自动执行:
1. 创建容器 ⏱️ ~30s
2. 配置网络 ⏱️ ~10s
3. 设置域名 ⏱️ ~10s
4. 申请SSL ⏱️ ~30s
5. 配置CDN ⏱️ ~20s
6. 启动监控 ⏱️ ~10s
    ↓
部署完成 (3分钟内)
```

## 验收标准

- ✅ 部署时间 < 3分钟
- ✅ 可用性 > 99.9%
- ✅ 自动扩容时间 < 1分钟

## 项目结构

```
backend/deploy/saas/
├── __init__.py          # 包初始化
├── deployer.py          # 部署器
├── resource_manager.py  # 资源管理
├── monitor.py           # 监控器
├── cdn_manager.py       # CDN管理
├── api.py               # API接口
├── config.yaml          # 配置文件
├── Dockerfile           # Docker镜像
├── docker-compose.yml   # Docker编排
└── README.md            # 文档
```

## 使用Docker运行

```bash
# 构建镜像
docker build -t ai-platform-deployer .

# 运行容器
docker run -d \
  -p 8080:8080 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v ./logs:/var/log/ai-platform \
  ai-platform-deployer

# 或使用docker-compose
docker-compose up -d
```

## 监控集成

### Prometheus指标

默认暴露Prometheus格式指标：`http://localhost:8080/metrics`

### Grafana仪表盘

启动后访问 http://localhost:3000，使用 admin/admin123 登录。

## 配置说明

### 扩容策略

```yaml
deploy:
  scaling:
    min_replicas: 1
    max_replicas: 10
    scale_up_threshold: 80
    scale_down_threshold: 30
```

### 告警规则

```yaml
monitor:
  alerts:
    cpu_threshold_warning: 80
    memory_threshold_warning: 85
    error_rate_threshold: 1.0
```

## 故障排查

### 查看日志

```bash
# Docker日志
docker logs ai-platform-deployer

# 文件日志
tail -f ./logs/app.log
```

### 健康检查

```bash
curl http://localhost:8080/health
```

## 许可证

MIT License
