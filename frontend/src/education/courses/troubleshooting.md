# 故障排除 - 1小时

解决常见问题、分析日志和使用调试工具。

## 1. 常见问题

### 1.1 Agent相关问题

```
问题：Agent响应超时
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因分析：
  • 模型响应时间长
  • 网络延迟
  • Agent配置不当

解决方案：
  1. 检查模型配置
     - 降低max_tokens
     - 调整temperature
  2. 增加超时时间
     timeout: 60000  // 60秒
  3. 启用流式输出
     stream: true
```

```
问题：Agent记忆丢失
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因分析：
  • Redis连接断开
  • 内存清理策略
  • Session过期

解决方案：
  1. 检查Redis连接状态
     redis-cli ping
  2. 调整TTL配置
     memory:
       short_term:
         ttl: 7200  // 2小时
```

### 1.2 Pipeline相关问题

```
问题：Pipeline执行失败
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
排查步骤：
  1. 查看执行日志
  2. 检查节点配置
  3. 验证输入数据
  4. 检查依赖服务
```

```
问题：节点连接超时
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因分析：
  • 下游服务不可用
  • 网络问题
  • 资源不足

解决方案：
  1. 增加超时配置
     timeout: 30000
  2. 添加重试机制
     retry:
       max_attempts: 3
```

### 1.3 性能问题

```
问题：响应延迟高
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
诊断步骤：
  1. 检查模型延迟
     metrics show --latency
  2. 检查队列积压
     metrics show --queue
  3. 检查资源使用
     metrics show --resources

优化建议：
  • 启用批处理
  • 增加缓存
  • 扩容节点
```

## 2. 日志分析

### 2.1 日志级别

| 级别 | 用途 | 示例 |
|-----|------|-----|
| DEBUG | 开发调试 | 变量值、函数调用 |
| INFO | 正常流程 | 请求开始/结束 |
| WARN | 异常情况 | 重试、超时 |
| ERROR | 错误 | 异常、失败 |

### 2.2 日志查看

```bash
# 查看最近100条日志
logs --tail 100

# 按级别过滤
logs --level error

# 按关键词搜索
logs --grep "timeout"

# 跟踪实时日志
logs --follow

# 按时间范围
logs --since "2024-01-01" --until "2024-01-02"
```

### 2.3 日志格式

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "ERROR",
  "trace_id": "abc123",
  "span_id": "def456",
  "message": "Agent execution failed",
  "context": {
    "agent_id": "agent-001",
    "node_id": "processor-001",
    "error": "TimeoutError"
  }
}
```

### 2.4 日志分析脚本

```bash
#!/bin/bash
# 分析错误日志

echo "=== 错误统计 ==="
grep -h '"level":"ERROR"' logs/*.json | \
  jq -r '.context.error // "Unknown"' | \
  sort | uniq -c | sort -rn

echo "=== 延迟分布 ==="
grep -h '"duration"' logs/*.json | \
  jq -r '.duration' | \
  awk '{sum+=$1; count++; if($1>max) max=$1} END {print "Avg:", sum/count, "Max:", max}'

echo "=== 慢请求 ==="
grep -h '"duration"' logs/*.json | \
  jq 'select(.duration > 5000)' | \
  jq -s 'sort_by(.duration) | .[-5:]'
```

## 3. 调试工具

### 3.1 调试命令

```bash
# 启动调试模式
debug --agent agent-001

# 查看Agent状态
status agent-001

# 内存分析
memory --profile agent-001

# 性能追踪
trace --id trace-001

# 模拟请求
test --input '{"text":"hello"}' --agent agent-001
```

### 3.2 监控面板

```typescript
// 实时监控配置
const monitorConfig = {
  refreshInterval: 5000,
  metrics: [
    'cpu_usage',
    'memory_usage',
    'request_count',
    'error_count',
    'latency_p50',
    'latency_p95',
    'latency_p99'
  ],
  alerts: [
    {
      condition: 'error_rate > 0.05',
      severity: 'warning'
    },
    {
      condition: 'latency_p95 > 3000',
      severity: 'warning'
    }
  ]
};
```

### 3.3 诊断命令

```bash
# 系统健康检查
health check

# 依赖服务状态
health check --dependencies

# 数据库连接测试
health check --database

# 缓存连接测试
health check --cache

# 网络连通性测试
health check --network
```

### 3.4 常用调试场景

#### 场景1：Agent无响应
```bash
# 1. 检查Agent状态
status agent-001

# 2. 查看运行日志
logs --agent agent-001 --tail 50

# 3. 检查资源使用
top --pid $(pgrep -f agent-001)

# 4. 重启Agent
restart agent-001
```

#### 场景2：Pipeline执行慢
```bash
# 1. 查看执行历史
pipeline history pipeline-001

# 2. 分析慢节点
pipeline analyze pipeline-001

# 3. 跟踪执行
pipeline trace pipeline-001 --execution-id xxx
```

## 4. 故障排查流程

### 标准流程
```
1. 收集信息
   ├── 错误信息
   ├── 日志
   └── 监控数据
   
2. 分析原因
   ├── 日志分析
   ├── 模式识别
   └── 关联分析
   
3. 确定方案
   ├── 临时方案
   └── 永久方案
   
4. 实施解决
   ├── 备份
   ├── 执行
   └── 验证
   
5. 总结记录
   ├── 故障报告
   └── 改进措施
```

## 总结

本课程涵盖：
- ✅ 常见问题诊断
- ✅ 日志分析方法
- ✅ 调试工具使用
- ✅ 故障排查流程

---
*课程时长：1小时*
