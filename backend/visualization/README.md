# AI Platform v5 - 训练可视化模块

## 概述

本模块提供训练过程可视化功能，支持 Loss 曲线、GPU 监控、评估指标图表等。

## 文件结构

```
backend/
├── visualization/
│   ├── __init__.py          # 模块初始化
│   ├── charts.py            # 图表生成器
│   └── realtime.py          # 实时数据处理
└── api/
    └── endpoints/
        └── visualization.py # API端点
```

## API 端点

### 训练作业管理

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/visualization/training/jobs` | 创建训练作业 |
| GET | `/api/v1/visualization/training/jobs` | 列出训练作业 |
| GET | `/api/v1/visualization/training/jobs/{job_id}` | 获取作业详情 |
| DELETE | `/api/v1/visualization/training/jobs/{job_id}` | 删除作业 |
| PUT | `/api/v1/visualization/training/jobs/{job_id}/status` | 更新状态 |

### 指标管理

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/visualization/training/{job_id}/metrics` | 添加训练指标 |
| GET | `/api/v1/visualization/training/{job_id}/metrics/history` | 获取指标历史 |

### 可视化图表

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/v1/visualization/training/{job_id}/loss` | Loss 曲线 |
| GET | `/api/v1/visualization/training/{job_id}/gpu` | GPU 监控 |
| GET | `/api/v1/visualization/training/{job_id}/metrics/chart` | 评估指标 |
| GET | `/api/v1/visualization/training/{job_id}/learning-rate` | 学习率曲线 |

### 仪表盘

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/v1/visualization/dashboard` | 可视化仪表盘 |
| GET | `/api/v1/visualization/dashboard/job/{job_id}` | 指定作业仪表盘 |

### 实时流

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/v1/visualization/training/{job_id}/stream` | SSE 实时流 |

## 使用示例

### 1. 创建训练作业

```bash
curl -X POST "http://localhost:8000/api/v1/visualization/training/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "BERT Fine-tuning",
    "model_name": "bert-base-uncased",
    "total_epochs": 10,
    "total_steps": 10000
  }'
```

### 2. 添加训练指标

```bash
curl -X POST "http://localhost:8000/api/v1/visualization/training/job-001/metrics" \
  -H "Content-Type: application/json" \
  -d '{
    "step": 100,
    "epoch": 1,
    "train_loss": 0.85,
    "val_loss": 0.92,
    "learning_rate": 2e-5,
    "gpu_utilization": 75.5,
    "gpu_memory": 6144,
    "accuracy": 0.78,
    "f1": 0.76
  }'
```

### 3. 获取 Loss 曲线

```bash
curl "http://localhost:8000/api/v1/visualization/training/job-001/loss?smooth=true&smooth_factor=0.1"
```

### 4. 获取 GPU 监控

```bash
curl "http://localhost:8000/api/v1/visualization/training/job-001/gpu"
```

### 5. SSE 实时流

```javascript
const eventSource = new EventSource('http://localhost:8000/api/v1/visualization/training/job-001/stream');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('New metrics:', data);
};

eventSource.addEventListener('metrics', (event) => {
  const metrics = JSON.parse(event.data);
  // 更新图表
});
```

## 前端集成

### Chart.js 示例

```javascript
async function loadLossChart(jobId) {
  const response = await fetch(`/api/v1/visualization/training/${jobId}/loss`);
  const { data } = await response.json();
  
  new Chart(ctx, {
    type: data.type,
    data: data.data,
    options: data.options
  });
}
```

### Recharts 示例

```javascript
function LossChart({ data }) {
  return (
    <LineChart data={data.data.datasets[0].data}>
      {data.data.datasets.map((dataset, index) => (
        <Line
          key={index}
          dataKey={dataset.label}
          stroke={dataset.borderColor}
        />
      ))}
    </LineChart>
  );
}
```

## 技术栈

- **后端**: FastAPI + SSE (Server-Sent Events)
- **前端**: Chart.js / Recharts
- **数据存储**: 内存存储 (可扩展为时序数据库)
- **图表格式**: Chart.js 配置格式

## 扩展说明

### 自定义图表配置

```python
from backend.visualization.charts import LossChartConfig

config = LossChartConfig(
    title="Custom Loss",
    colors=["#ff0000", "#00ff00"],
    smooth_factor=0.2
)
```

### 添加新的指标类型

在 `TrainingMetrics` 类中添加新字段：

```python
@dataclass
class TrainingMetrics:
    step: int
    epoch: int
    # 添加新指标
    new_metric: Optional[float] = None
```
