# API Gateway v6

API网关模块，提供流量控制和限流功能。

## 核心功能

### 1. 网关路由器 (`backend/gateway/router.py`)

提供动态路由配置、请求转发和负载均衡。

**主要组件：**
- `Route`: 路由配置类
- `GatewayRouter`: 主路由器

**功能：**
- 添加/更新/删除路由
- 路径匹配（支持路径参数）
- 负载均衡权重
- 健康检查支持

### 2. 请求限流器 (`backend/gateway/ratelimit.py`)

提供多种限流算法。

**支持的算法：**
- **Token Bucket**: 令牌桶算法，支持突发流量
- **Sliding Window**: 滑动窗口，更平滑的限流
- **Fixed Window**: 固定窗口，简单易用

### 3. 配额管理器 (`backend/gateway/quota.py`)

提供API配额管理。

**功能：**
- 用户级别配额
- API级别配额
- 配额周期（每日/每周/每月/每年）
- 配额预警

### 4. 网关中间件 (`backend/gateway/middleware.py`)

提供各种中间件：
- `RateLimitMiddleware`: 限流中间件
- `QuotaMiddleware`: 配额检查中间件
- `TrafficStatsMiddleware`: 流量统计中间件
- `AnomalyDetectionMiddleware`: 异常检测中间件
- `RequestLoggingMiddleware`: 请求日志中间件

## API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/gateway/route` | POST | 添加路由 |
| `/api/v1/gateway/routes` | GET | 列出路由 |
| `/api/v1/gateway/route/{id}` | PUT | 更新路由 |
| `/api/v1/gateway/route/{id}` | DELETE | 删除路由 |
| `/api/v1/gateway/limit` | POST | 设置限流 |
| `/api/v1/gateway/limit/{key}` | GET | 获取限流状态 |
| `/api/v1/gateway/limit/{key}` | DELETE | 重置限流 |
| `/api/v1/gateway/quota/rule` | POST | 添加配额规则 |
| `/api/v1/gateway/quota/rules` | GET | 列出配额规则 |
| `/api/v1/gateway/quota` | GET | 查询配额使用 |
| `/api/v1/gateway/usage` | GET | 获取使用统计 |
| `/api/v1/gateway/alerts` | GET | 获取告警 |
| `/api/v1/gateway/health` | GET | 健康检查 |

## 使用示例

### 添加路由

```python
import requests

response = requests.post("/api/v1/gateway/route", json={
    "path": "/api/v1/users/*",
    "target_url": "http://user-service:8080",
    "methods": ["GET", "POST"],
    "rate_limit": 100,
    "description": "User service"
})
```

### 设置限流

```python
response = requests.post("/api/v1/gateway/limit", json={
    "key": "user:123",
    "requests": 50,
    "window_seconds": 60,
    "algorithm": "token_bucket"
})
```

### 查询配额

```python
response = requests.get("/api/v1/gateway/quota", params={
    "key": "user:123"
})
```

## 配置中间件

```python
from fastapi import FastAPI
from backend.gateway.middleware import GatewayMiddleware

app = FastAPI()
middleware = GatewayMiddleware()

# 添加中间件
app.add_middleware(middleware.add_rate_limit(default_requests=100))
app.add_middleware(middleware.add_quota(default_quota=1000))
app.add_middleware(middleware.add_traffic_stats())
```

## 运行测试

```bash
python3 backend/gateway/test_gateway.py
```
