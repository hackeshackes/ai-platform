# AI Platform v2.0 API文档

## 概述

AI Platform v2.0 API 基于RESTful设计，支持JSON格式请求/响应。

## 基础URL

| 环境 | URL |
|------|-----|
| 开发环境 | `http://localhost:8000` |
| 生产环境 | `https://api.ai-platform.com` |

## 认证

所有API端点(除健康检查和认证)需要JWT Token。

```bash
# 获取Token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"

# 使用Token
curl http://localhost:8000/api/v1/projects \
  -H "Authorization: Bearer <token>"
```

## 响应格式

### 成功响应

```json
{
  "success": true,
  "data": {...},
  "meta": {
    "total": 100,
    "page": 1,
    "limit": 20
  }
}
```

### 错误响应

```json
{
  "success": false,
  "error": "错误信息"
}
```

## API端点

### 认证模块

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/v1/auth/token` | 获取访问令牌 |
| POST | `/api/v1/auth/refresh` | 刷新令牌 |
| GET | `/api/v1/auth/me` | 获取当前用户 |
| POST | `/api/v1/auth/logout` | 退出登录 |

### 项目模块

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/projects` | 获取项目列表 |
| POST | `/api/v1/projects` | 创建项目 |
| GET | `/api/v1/projects/{id}` | 获取项目详情 |
| PUT | `/api/v1/projects/{id}` | 更新项目 |
| DELETE | `/api/v1/projects/{id}` | 删除项目 |
| GET | `/api/v1/projects/stats` | 获取项目统计 |

### 任务模块

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/tasks` | 获取任务列表 |
| GET | `/api/v1/tasks/{id}` | 获取任务详情 |
| PUT | `/api/v1/tasks/{id}` | 更新任务 |
| DELETE | `/api/v1/tasks/{id}` | 删除任务 |
| GET | `/api/v1/tasks/{id}/logs` | 获取任务日志 |

### 数据集模块

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/datasets` | 获取数据集列表 |
| POST | `/api/v1/datasets` | 创建数据集 |
| GET | `/api/v1/datasets/{id}` | 获取数据集详情 |
| DELETE | `/api/v1/datasets/{id}` | 删除数据集 |

### 版本管理模块 (v2.0)

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/datasets/versions` | 获取版本列表 |
| POST | `/api/v1/datasets/versions` | 创建版本 |
| GET | `/api/v1/datasets/versions/{id}` | 获取版本详情 |
| DELETE | `/api/v1/datasets/versions/{id}` | 删除版本 |

### 质量检查模块 (v2.0)

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/v1/datasets/quality/check` | 检查数据集质量 |
| GET | `/api/v1/datasets/quality/quick` | 快速检查 |

### 模型模块

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/models` | 获取模型列表 |
| GET | `/api/v1/models/{id}` | 获取模型详情 |

### 训练模块

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/training/models` | 获取可用模型 |
| GET | `/api/v1/training/templates` | 获取训练模板 |
| POST | `/api/v1/training/submit` | 提交训练任务 |
| GET | `/api/v1/training/jobs` | 获取训练任务列表 |
| GET | `/api/v1/training/jobs/{id}` | 获取任务状态 |

### 推理模块

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/inference/models` | 获取推理模型 |
| POST | `/api/v1/inference/generate` | 执行推理 |
| GET | `/api/v1/inference/history` | 获取推理历史 |

### GPU监控模块

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/gpu` | 获取GPU状态 |

### 指标模块

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/metrics/loss` | 获取Loss曲线 |
| GET | `/api/v1/metrics/gpu` | 获取GPU指标 |

### 用户模块 (v2.0)

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/users` | 获取用户列表 |
| POST | `/api/v1/users` | 创建用户 |
| GET | `/api/v1/users/{id}` | 获取用户详情 |
| PUT | `/api/v1/users/{id}` | 更新用户 |
| DELETE | `/api/v1/users/{id}` | 删除用户 |

### 权限模块 (v2.0)

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/permissions/roles` | 获取角色列表 |
| POST | `/api/v1/permissions/roles` | 创建角色 |
| GET | `/api/v1/permissions/roles/{id}` | 获取角色详情 |
| PUT | `/api/v1/permissions/roles/{id}` | 更新角色 |
| DELETE | `/api/v1/permissions/roles/{id}` | 删除角色 |
| POST | `/api/v1/permissions/roles/init` | 初始化默认角色 |
| POST | `/api/v1/permissions/check` | 检查权限 |

### 设置模块

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/v1/settings/system` | 获取系统设置 |
| PUT | `/api/v1/settings/system` | 更新系统设置 |
| GET | `/api/v1/settings/storage` | 获取存储设置 |

## 缓存策略 (v2.0)

### 缓存键

| 资源 | 键格式 | TTL |
|------|--------|-----|
| 项目列表 | `projects:list:{user_id}` | 5分钟 |
| 任务状态 | `tasks:detail:{task_id}` | 30秒 |
| GPU指标 | `gpu:metrics` | 5秒 |
| 系统配置 | `system:config` | 24小时 |

### 缓存失效

```python
from core.decorators import invalidate_cache

@invalidate_cache("project_list", "project_detail")
async def update_project(id: int, data: dict):
    ...
```

## WebSocket API (v2.0)

```javascript
// 连接WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// 订阅训练进度
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'training',
  resource_id: 'task-123'
}));

// 接收消息
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data); // {type: 'progress', task_id: 'task-123', progress: 50}
};
```

## 错误码

| 错误码 | 说明 |
|--------|------|
| 400 | 请求错误 |
| 401 | 未认证 |
| 403 | 无权限 |
| 404 | 资源不存在 |
| 422 | 参数验证错误 |
| 500 | 服务器错误 |

---

**文档版本**: 2.0.0  
**更新日期**: 2026-02-09
