# AI Platform API 文档

**版本**: 1.0.0  
**更新**: 2026-02-08

---

## 目录

1. [认证](#认证)
2. [项目管理](#项目管理)
3. [任务管理](#任务管理)
4. [数据集管理](#数据集管理)
5. [模型管理](#模型管理)
6. [GPU监控](#gpu监控)
7. [训练指标](#训练指标)
8. [训练任务](#训练任务)
9. [推理服务](#推理服务)
10. [系统设置](#系统设置)

---

## 认证

### POST /api/v1/auth/token
登录获取访问Token。

**请求**:
```bash
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded

username=admin&password=admin123
```

**响应** (200):
```json
{
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 86400
}
```

**错误响应** (401):
```json
{
    "detail": "Incorrect username or password"
}
```

### GET /api/v1/auth/me
获取当前登录用户信息。

**请求**:
```bash
GET /api/v1/auth/me
Authorization: Bearer <token>
```

**响应** (200):
```json
{
    "id": 1,
    "username": "admin",
    "email": "admin@ai-platform.com",
    "is_active": true,
    "created_at": "2026-02-08T12:00:00"
}
```

---

## 项目管理

### GET /api/v1/projects
获取项目列表。

**请求**:
```bash
GET /api/v1/projects?skip=0&limit=100
Authorization: Bearer <token>
```

**响应** (200):
```json
{
    "total": 2,
    "projects": [
        {
            "id": 1,
            "name": "LLM Fine-tuning Demo",
            "description": "Llama 2 微调示例项目",
            "owner_id": 1,
            "status": "active",
            "created_at": "2026-02-08T12:00:00"
        }
    ]
}
```

### POST /api/v1/projects
创建新项目。

**请求**:
```bash
POST /api/v1/projects
Authorization: Bearer <token>
Content-Type: application/json

{
    "name": "新项目名称",
    "description": "项目描述（可选）"
}
```

**响应** (201):
```json
{
    "id": 3,
    "name": "新项目名称",
    "description": "项目描述",
    "owner_id": 1,
    "status": "active",
    "created_at": "2026-02-08T12:00:00"
}
```

### GET /api/v1/projects/{id}
获取项目详情。

### PUT /api/v1/projects/{id}
更新项目。

### DELETE /api/v1/projects/{id}
删除项目。

---

## 任务管理

### GET /api/v1/tasks
获取任务列表。

**参数**:
- `status`: 状态筛选 (pending/running/completed/failed)
- `skip`: 跳过的数量 (默认0)
- `limit`: 返回数量 (默认100)

**响应**:
```json
{
    "total": 5,
    "tasks": [
        {
            "id": 1,
            "name": "training-task-1",
            "type": "training",
            "project_id": 1,
            "status": "running",
            "progress": 45.5,
            "created_at": "2026-02-08T12:00:00"
        }
    ]
}
```

---

## 数据集管理

### GET /api/v1/datasets
获取数据集列表。

**响应**:
```json
{
    "total": 2,
    "datasets": [
        {
            "id": 1,
            "name": "alpaca-zh",
            "description": "中文Alpaca指令数据集",
            "project_id": 1,
            "format": "jsonl",
            "size": 52428800,
            "row_count": 52000,
            "status": "ready",
            "created_at": "2026-02-08T12:00:00"
        }
    ]
}
```

---

## 模型管理

### GET /api/v1/models
获取模型列表。

**响应**:
```json
{
    "total": 2,
    "models": [
        {
            "id": 1,
            "name": "Llama-2-7b-chat-finetuned",
            "base_model": "meta-llama/Llama-2-7b-chat-hf",
            "framework": "transformers",
            "size": "14GB",
            "status": "ready",
            "metrics": {
                "final_loss": 0.2834,
                "bleu_score": 45.2
            }
        }
    ]
}
```

---

## GPU监控

### GET /api/v1/gpu
获取GPU实时状态。

**响应**:
```json
{
    "total_gpus": 1,
    "total_memory_mb": 16384,
    "used_memory_mb": 4096,
    "avg_utilization": 45.0,
    "metrics": [
        {
            "gpu_id": 0,
            "name": "NVIDIA GeForce RTX 4090",
            "total_memory_mb": 16384,
            "used_memory_mb": 4096,
            "utilization_percent": 45,
            "temperature_c": 58,
            "power_watts": 320.5
        }
    ]
}
```

---

## 训练指标

### GET /api/v1/metrics/loss
获取Loss曲线数据。

**参数**:
- `experiment_id`: 实验ID (可选)
- `steps`: 数据点数量 (默认100)

**响应**:
```json
{
    "experiment_id": "demo-exp-001",
    "total_steps": 100,
    "metrics": {
        "initial_loss": 2.0405,
        "final_loss": 0.2834,
        "best_loss": 0.2512
    },
    "data": [
        {"step": 1, "loss": 2.0405, "epoch": 0},
        {"step": 2, "loss": 2.0150, "epoch": 0}
    ]
}
```

---

## 训练任务

### GET /api/v1/training/models
获取可用训练模型。

**响应**:
```json
{
    "models": [
        {
            "id": "llama2-7b",
            "name": "Llama-2-7b-chat-hf",
            "provider": "meta",
            "size": "7B",
            "type": "base"
        }
    ]
}
```

### GET /api/v1/training/templates
获取训练模板。

**响应**:
```json
{
    "templates": [
        {
            "id": "lora",
            "name": "LoRA (低秩适配)",
            "description": "轻量级微调，适合消费级GPU",
            "min_gpu_memory": "8GB"
        }
    ]
}
```

### POST /api/v1/training/submit
提交训练任务。

**请求**:
```json
{
    "model_id": "llama2-7b",
    "dataset_id": 1,
    "template_id": "lora",
    "project_id": 1,
    "experiment_name": "我的实验",
    "hyperparameters": {
        "learning_rate": 2e-4,
        "epochs": 3
    }
}
```

---

## 推理服务

### GET /api/v1/inference/models
获取可用推理模型。

**响应**:
```json
{
    "models": [
        {
            "id": "llama2-7b-chat",
            "name": "Llama-2-7b-chat-hf",
            "provider": "meta",
            "size": "7B",
            "status": "ready"
        }
    ]
}
```

### POST /api/v1/inference/generate
执行推理。

**请求**:
```json
{
    "model_id": "llama2-7b-chat",
    "prompt": "什么是机器学习？",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
}
```

**响应**:
```json
{
    "id": "abc12345",
    "model": "llama2-7b-chat",
    "prompt": "什么是机器学习？",
    "output": "机器学习是...",
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 100,
        "total_tokens": 110
    },
    "latency_ms": 123.45
}
```

---

## 系统设置

### GET /api/v1/settings/system
获取系统配置。

**响应**:
```json
{
    "site_name": "AI Platform",
    "site_description": "大模型全生命周期管理平台",
    "version": "1.0.0",
    "language": "zh-CN",
    "theme": "light",
    "features": {
        "gpu_monitoring": true,
        "distributed_training": true,
        "model_registry": true,
        "inference_service": true
    }
}
```

### GET /api/v1/settings/storage
获取存储配置。

**响应**:
```json
{
    "max_dataset_size_gb": 10,
    "max_model_size_gb": 50,
    "default_storage_path": "/data",
    "used_storage_gb": 2.5,
    "total_storage_gb": 100
}
```

---

## 错误处理

### HTTP状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 201 | 创建成功 |
| 400 | 请求错误 |
| 401 | 未认证 |
| 403 | 无权限 |
| 404 | 资源不存在 |
| 500 | 服务器错误 |

### 错误响应格式

```json
{
    "detail": "错误描述信息"
}
```

---

*文档版本: 1.0.0*  
*最后更新: 2026-02-08*
