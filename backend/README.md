# v10 监控体系

完整的系统监控体系，包括健康检查、性能指标、日志追踪和告警系统。

## 目录结构

```
backend/
├── core/
│   ├── __init__.py          # 模块初始化
│   ├── monitoring.py        # 监控主模块
│   ├── health.py            # 健康检查
│   ├── metrics.py           # 性能指标
│   ├── logger.py            # 结构化日志
│   ├── alerts.py            # 告警管理
│   └── tracing.py           # 链路追踪
├── api/
│   ├── __init__.py
│   └── endpoints/
│       ├── __init__.py
│       └── monitoring.py    # 监控API端点
├── config/
│   └── monitoring.yaml      # 配置文件
├── requirements.txt         # 依赖
└── tests/
    └── test_monitoring.py   # 单元测试
```

## 核心功能

### 1. 健康检查 (HealthChecker)

```python
from core.health import HealthChecker, HealthStatus

# 初始化
checker = HealthChecker()
await checker.initialize()

# 运行检查
result = await checker.run_all_checks()
# {
#     "status": "healthy",
#     "checks": {
#         "cpu": {"status": "healthy", "message": "CPU使用率正常: 25%"},
#         "memory": {...},
#         "disk": {...}
#     }
# }

# 注册自定义检查
async def custom_check():
    return HealthCheckResult(
        name="custom",
        status=HealthStatus.HEALTHY,
        message="OK"
    )
checker.register_check("custom", custom_check)
```

### 2. 性能指标 (MetricsCollector)

```python
from core.metrics import MetricsCollector

collector = MetricsCollector()
await collector.initialize()

# 计数器
collector.counter_inc("api_requests", 1.0)

# 瞬时值
collector.gauge_set("cpu_usage", 45.5)

# 直方图
collector.histogram_observe("request_latency", 0.25)

# 收集所有指标
metrics = await collector.collect_all()

# Prometheus格式导出
prometheus_output = collector.to_prometheus_format()
```

### 3. 结构化日志 (StructuredLogger)

```python
from core.logger import StructuredLogger, get_logger

logger = StructuredLogger()
logger.initialize()

# 基本日志
logger.info("用户登录成功", user_id="12345")
logger.error("请求失败", error_code=500)

# 带事件日志
logger.log_request("GET", "/api/users", 200, 0.05)
logger.log_database_query("SELECT *", 0.01, success=True)
```

### 4. 告警系统 (AlertManager)

```python
from core.alerts import AlertManager, AlertRule, AlertSeverity

manager = AlertManager()
await manager.initialize()

# 添加规则
rule = AlertRule(
    name="HighCPU",
    condition="cpu_usage > 80",
    severity=AlertSeverity.WARNING,
    message="CPU使用率过高"
)
manager.add_rule(rule)

# 检查告警
metrics = {"cpu_usage": 85}
alerts = await manager.check_all(metrics)

# 获取活跃告警
active = manager.get_active_alerts()
```

### 5. 链路追踪 (TracingManager)

```python
from core.tracing import TracingManager, SpanKind

tracer = TracingManager()
await tracer.initialize()

# 使用上下文管理器
with tracer.trace("process_order", SpanKind.INTERNAL) as span:
    span.set_attribute("order_id", "12345")
    span.add_event("validation_complete")
    
    with tracer.span("save_to_db", SpanKind.CLIENT):
        # 数据库操作
        pass

# 获取追踪ID
trace_id = tracer.get_current_trace_id()
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/monitoring/health` | GET | 健康检查 |
| `/monitoring/metrics` | GET | 性能指标 |
| `/monitoring/metrics/prometheus` | GET | Prometheus格式指标 |
| `/monitoring/alerts` | GET | 活跃告警 |
| `/monitoring/alerts/history` | GET | 告警历史 |
| `/monitoring/alerts/rules` | GET | 告警规则列表 |
| `/monitoring/logs` | GET | 日志查询 |
| `/monitoring/tracing/{trace_id}` | GET | 链路追踪详情 |
| `/monitoring/status` | GET | 监控系统状态 |

## 配置

配置文件 `config/monitoring.yaml`:

```yaml
service_name: "ai-platform"

logging:
  level: "INFO"
  json_format: true

alerts:
  evaluation_interval: 15
  channels:
    critical:
      - log
      - webhook

tracing:
  sampling_rate: 1.0
```

## 运行测试

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/test_monitoring.py -v --cov

# 查看覆盖率报告
pytest tests/test_monitoring.py --cov=backend --cov-report=html
```

## 与Prometheus集成

1. 启用Prometheus指标端点
2. 配置Prometheus抓取任务

```yaml
scrape_configs:
  - job_name: 'ai-platform'
    metrics_path: /monitoring/metrics/prometheus
    static_configs:
      - targets: ['localhost:8000']
```

## 告警规则

默认规则包括：
- `HighCPU`: CPU使用率 > 80%
- `HighMemory`: 内存使用率 > 85%
- `HighLatency`: API P95延迟 > 1秒
- `HighErrorRate`: 错误率 > 1%
- `DatabaseDown`: 数据库连接失败

## 扩展

### 添加自定义健康检查

```python
async def check_external_service():
    try:
        await ping("external-service")
        return HealthCheckResult(
            name="external",
            status=HealthStatus.HEALTHY,
            message="External service OK"
        )
    except Exception as e:
        return HealthCheckResult(
            name="external",
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )

system.health.register_check("external", check_external_service)
```

### 添加自定义导出器

```python
def custom_exporter(spans):
    for span in spans:
        # 发送到自定义系统
        send_to_otel(span)

tracing_manager.register_exporter(custom_exporter)
```

## 依赖

- `psutil>=5.9.0` - 系统监控
- `structlog>=23.0.0` - 结构化日志
- `pyyaml>=6.0` - YAML配置
- `fastapi>=0.100.0` - API框架
- `pytest>=7.0.0` - 测试框架
