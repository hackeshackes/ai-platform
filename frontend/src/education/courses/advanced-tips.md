# 高级技巧 - 2小时

掌握性能优化、调试技巧和开发最佳实践。

## 1. 性能优化

### 1.1 延迟优化

```
延迟来源分析：
┌────────────────────────────────────────────────────┐
│  总延迟 = 模型推理 + 数据处理 + 网络传输 + 排队等待  │
└────────────────────────────────────────────────────┘

优化策略：
├── 模型推理优化
│   ├── 模型量化 (FP16/INT8)
│   ├── 批处理优化
│   └── KV Cache优化
├── 数据处理优化
│   ├── 异步处理
│   ├── 数据预取
│   └── 缓存策略
└── 系统优化
    ├── 连接池
    └── 负载均衡
```

### 1.2 批处理优化

```typescript
interface BatchConfig {
  max_batch_size: number;
  max_wait_time: number;  // ms
  batching_strategy: 'dynamic' | 'static';
}

class BatchProcessor {
  private queue: Request[] = [];
  private timer: NodeJS.Timer | null = null;
  
  async process(request: Request): Promise<Response> {
    // 添加到队列
    this.queue.push(request);
    
    // 触发批处理
    if (this.queue.length >= this.config.max_batch_size) {
      return this.executeBatch();
    }
    
    // 设置定时器
    if (!this.timer) {
      this.timer = setTimeout(
        () => this.executeBatch(),
        this.config.max_wait_time
      );
    }
    
    // 返回Promise等待结果
    return this.createPendingRequest(request);
  }
  
  private async executeBatch(): Promise<void> {
    const batch = this.queue.splice(0, this.config.max_batch_size);
    clearTimeout(this.timer!);
    this.timer = null;
    
    // 批量推理
    const results = await this.model.batchInference(
      batch.map(r => r.input)
    );
    
    // 分发结果
    batch.forEach((request, i) => {
      request.resolve(results[i]);
    });
  }
}
```

### 1.3 缓存策略

```typescript
// 多级缓存架构
class CacheManager {
  // L1: 内存缓存 (热点数据)
  private l1Cache: Map<string, CacheEntry> = new Map();
  
  // L2: Redis缓存 (分布式)
  private l2Cache: Redis;
  
  // L3: 持久化缓存 (磁盘)
  private l3Cache: DiskCache;
  
  async get(key: string): Promise<any> {
    // L1检查
    const l1Result = this.l1Cache.get(key);
    if (l1Result && !this.isExpired(l1Result)) {
      return l1Result.value;
    }
    
    // L2检查
    const l2Result = await this.l2Cache.get(key);
    if (l2Result) {
      // 回填L1
      this.l1Cache.set(key, l2Result);
      return l2Result.value;
    }
    
    // L3检查
    const l3Result = await this.l3Cache.get(key);
    if (l3Result) {
      // 回填L1和L2
      this.l1Cache.set(key, l3Result);
      await this.l2Cache.set(key, l3Result);
      return l3Result.value;
    }
    
    return null;
  }
}
```

### 1.4 内存优化

```typescript
// 内存池管理
class MemoryPool {
  private pool: ArrayBuffer[] = [];
  private readonly CHUNK_SIZE = 1024 * 1024; // 1MB
  
  allocate(): ArrayBuffer {
    return this.pool.pop() || new ArrayBuffer(this.CHUNK_SIZE);
  }
  
  release(buffer: ArrayBuffer): void {
    if (buffer.byteLength === this.CHUNK_SIZE) {
      this.pool.push(buffer);
    }
  }
}

// 大对象分块处理
async function processLargeObject(obj: any): Promise<void> {
  const chunks = chunk(JSON.stringify(obj), 10000);
  
  for (const chunk of chunks) {
    await processChunk(chunk);
  }
}
```

## 2. 调试技巧

### 2.1 日志系统

```typescript
// 分级日志
enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3
}

class Logger {
  private level: LogLevel;
  private context: Record<string, any>;
  
  debug(message: string, data?: any): void {
    if (this.level <= LogLevel.DEBUG) {
      this.write('DEBUG', message, data);
    }
  }
  
  info(message: string, data?: any): void {
    if (this.level <= LogLevel.INFO) {
      this.write('INFO', message, data);
    }
  }
  
  private write(level: string, message: string, data?: any): void {
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      context: this.context,
      data,
      trace_id: this.getTraceId()
    };
    
    console.log(JSON.stringify(entry));
  }
}
```

### 2.2 链路追踪

```typescript
// 分布式追踪
class Tracer {
  private static readonly SPAN_NAME = 'ai.pipeline';
  
  startSpan(name: string): Span {
    const span = {
      name,
      startTime: Date.now(),
      spans: [],
      attributes: {}
    };
    
    return span;
  }
  
  endSpan(span: Span): void {
    span.duration = Date.now() - span.startTime;
    this.exportSpan(span);
  }
  
  async withSpan<T>(
    name: string,
    fn: () => Promise<T>
  ): Promise<T> {
    const span = this.startSpan(name);
    try {
      return await fn();
    } finally {
      this.endSpan(span);
    }
  }
}

// 使用示例
const result = await tracer.withSpan('process_message', async () => {
  const sentiment = await analyzeSentiment(text);
  const entities = await extractEntities(text);
  return { sentiment, entities };
});
```

### 2.3 断点调试

```typescript
// 条件断点
class Debugger {
  private breakpoints: Map<string, BreakpointCondition[]> = new Map();
  
  checkBreakpoint(nodeId: string, state: any): boolean {
    const conditions = this.breakpoints.get(nodeId);
    if (!conditions) return false;
    
    return conditions.every(condition => 
      this.evaluateCondition(condition, state)
    );
  }
  
  private evaluateCondition(
    condition: BreakpointCondition, 
    state: any
  ): boolean {
    return new Function('state', condition.expression)(state);
  }
  
  // 状态快照
  snapshot(name: string): void {
    console.log(`[Snapshot: ${name}]`, JSON.stringify(
      this.getCurrentState(), 
      null, 
      2
    ));
  }
}
```

### 2.4 调试面板

```typescript
// 开发调试面板
class DebugPanel {
  private state: DebugState = {
    logs: [],
    metrics: {},
    trace: [],
    snapshots: []
  };
  
  // 实时指标监控
  monitorMetrics(): void {
    setInterval(() => {
      this.state.metrics = {
        memory_usage: process.memoryUsage(),
        cpu_usage: process.cpuUsage(),
        active_requests: this.getActiveRequestCount(),
        queue_length: this.getQueueLength()
      };
    }, 1000);
  }
  
  // 日志查看
  getRecentLogs(level?: LogLevel): LogEntry[] {
    return this.state.logs.filter(
      log => !level || log.level === level
    );
  }
  
  // 状态回放
  replay(traceId: string): void {
    const trace = this.state.trace.filter(
      t => t.trace_id === traceId
    );
    this.playback(trace);
  }
}
```

## 3. 最佳实践

### 3.1 代码规范

```
项目结构规范：
├── src/
│   ├── agents/          # Agent定义
│   │   ├── base/        # 基类
│   │   └── custom/     # 自定义Agent
│   ├── pipelines/       # Pipeline定义
│   ├── components/      # 可复用组件
│   ├── services/       # 服务层
│   ├── utils/          # 工具函数
│   └── types/          # 类型定义
├── tests/
│   ├── unit/           # 单元测试
│   ├── integration/   # 集成测试
│   └── e2e/            # 端到端测试
├── docs/               # 文档
└── scripts/            # 构建脚本
```

### 3.2 配置管理

```typescript
// 环境区分配置
interface Config {
  development: {
    logLevel: 'debug';
    apiEndpoint: string;
    enableDebugger: true;
  };
  staging: {
    logLevel: 'info';
    apiEndpoint: string;
    enableDebugger: false;
  };
  production: {
    logLevel: 'error';
    apiEndpoint: string;
    enableDebugger: false;
  };
}

// 配置加载
function loadConfig(): Config {
  const env = process.env.NODE_ENV || 'development';
  return config[env];
}
```

### 3.3 错误处理最佳实践

```typescript
// 统一错误处理
class AppError extends Error {
  constructor(
    public code: string,
    public message: string,
    public statusCode: number,
    public details?: any
  ) {
    super(message);
    this.name = 'AppError';
  }
}

// 错误处理中间件
async function errorHandler(
  error: Error,
  context: RequestContext
): Promise<Response> {
  if (error instanceof AppError) {
    return {
      status: error.statusCode,
      body: {
        code: error.code,
        message: error.message,
        details: error.details
      }
    };
  }
  
  // 未知错误
  logger.error('Unknown error', { error, context });
  
  return {
    status: 500,
    body: {
      code: 'INTERNAL_ERROR',
      message: 'An unexpected error occurred'
    }
  };
}
```

### 3.4 性能监控

```typescript
// 性能指标收集
class MetricsCollector {
  private counters: Map<string, number> = new Map();
  private histograms: Map<string, number[]> = new Map();
  
  increment(metric: string): void {
    this.counters.set(metric, (this.counters.get(metric) || 0) + 1);
  }
  
  observe(metric: string, value: number): void {
    const histogram = this.histograms.get(metric) || [];
    histogram.push(value);
    this.histograms.set(metric, histogram);
  }
  
  getSummary(metric: string): MetricSummary {
    const values = this.histograms.get(metric) || [];
    return {
      min: Math.min(...values),
      max: Math.max(...values),
      avg: values.reduce((a, b) => a + b, 0) / values.length,
      p50: percentile(values, 50),
      p95: percentile(values, 95),
      p99: percentile(values, 99)
    };
  }
}
```

## 实践任务

1. **任务1**: 实现一个批处理优化器，提升吞吐量
2. **任务2**: 配置完整的日志和链路追踪系统
3. **任务3**: 编写性能监控仪表盘

## 总结

本课程学习了：
- ✅ 性能优化（批处理、缓存、内存）
- ✅ 调试技巧（日志、追踪、断点）
- ✅ 开发最佳实践（代码规范、配置管理、错误处理）

---
*课程时长：2小时*
