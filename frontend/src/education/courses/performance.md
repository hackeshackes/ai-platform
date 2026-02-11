# 性能优化 - 2小时

深入学习缓存策略、并发优化和资源管理。

## 1. 缓存策略

### 1.1 多级缓存架构

```
┌────────────────────────────────────────────────────────────┐
│                      缓存层级                               │
├────────────┬─────────────┬─────────────┬─────────────────┤
│   L1      │     L2      │     L3      │      L4         │
│ 内存缓存   │   Redis     │   磁盘缓存   │   CDN静态资源   │
├────────────┼─────────────┼─────────────┼─────────────────┤
│ 极快(~1ms) │ 快(~10ms)   │ 中(~50ms)   │ 慢(~100ms+)    │
│ 容量小     │ 容量中      │ 容量大      │ 容量极大        │
│ 成本高     │ 成本中      │ 成本低      │ 成本极低        │
└────────────┴─────────────┴─────────────┴─────────────────┘
```

### 1.2 缓存实现

```typescript
interface CacheConfig {
  maxSize: number;           // 最大条目数
  ttl: number;               // 过期时间(ms)
  strategy: 'lru' | 'lfu';   // 淘汰策略
}

class CacheManager {
  private l1: Map<string, CacheEntry> = new Map();
  private l2: RedisCache;
  
  async get(key: string): Promise<any> {
    // L1检查
    const l1Value = this.l1.get(key);
    if (l1Value && !this.isExpired(l1Value)) {
      return l1Value.value;
    }
    
    // L2检查
    const l2Value = await this.l2.get(key);
    if (l2Value) {
      // 回填L1
      this.l1.set(key, l2Value);
      return l2Value.value;
    }
    
    return null;
  }
  
  async set(key: string, value: any, ttl?: number): Promise<void> {
    // 同时写入L1和L2
    const entry = { value, expiry: Date.now() + ttl };
    this.l1.set(key, entry);
    await this.l2.set(key, entry, ttl);
  }
}
```

### 1.3 缓存策略配置

```yaml
cache:
  # Agent响应缓存
  agent_responses:
    enabled: true
    ttl: 3600
    max_size: 10000
    
  # 知识库检索缓存
  knowledge_retrieval:
    enabled: true
    ttl: 7200
    max_size: 50000
    
  # 模型输出缓存
  model_outputs:
    enabled: true
    ttl: 3600
    max_size: 100000
    cache_key: "hash(${input})"
```

### 1.4 缓存失效

```typescript
// 主动失效
async function invalidateCache(pattern: string): Promise<void> {
  const keys = await this.redis.keys(pattern);
  await this.redis.del(keys);
}

// 延迟失效（防止缓存击穿）
async function invalidateWithDelay(
  key: string, 
  delay: 1000
): Promise<void> {
  setTimeout(async () => {
    await this.cache.del(key);
  }, delay);
}

// 批量失效
async function invalidateBatch(keys: string[]): Promise<void> {
  await Promise.all([
    this.l1.delete(...keys),
    this.l2.del(...keys)
  ]);
}
```

## 2. 并发优化

### 2.1 并发模型

```
并发处理模式：
┌─────────────────────────────────────────────────────────┐
│                     并发策略                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   同步模式          异步模式           并行模式           │
│   ────────         ────────          ────────           │
│   await A          Promise.all       Worker线程          │
│   await B          [A, B]            [A, B]并行          │
│   await C          同时发起           独立执行            │
│                                                          │
│   优点：简单          优点：快          优点：CPU密集     │
│   缺点：慢            缺点：资源多       缺点：复杂       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 2.2 批处理优化

```typescript
interface BatchConfig {
  maxBatchSize: number;
  maxWaitTime: number;  // ms
  concurrency: number;
}

class BatchProcessor {
  private queue: BatchItem[] = [];
  private processing: Set<Promise<any>> = new Set();
  
  async add(item: BatchItem): Promise<any> {
    return new Promise((resolve, reject) => {
      this.queue.push({ item, resolve, reject });
      this.processQueue();
    });
  }
  
  private async processQueue(): Promise<void> {
    while (this.queue.length > 0) {
      const batch = this.queue.splice(0, this.config.maxBatchSize);
      
      const promises = batch.map(b => 
        this.processBatchItem(b.item)
          .then(b.resolve)
          .catch(b.reject)
      );
      
      await Promise.all(promises);
    }
  }
  
  private async processBatchItem(item: BatchItem): Promise<any> {
    // 单项处理逻辑
  }
}
```

### 2.3 连接池管理

```typescript
class ConnectionPool {
  private pool: Pool;
  private config: PoolConfig;
  
  constructor(config: PoolConfig) {
    this.pool = createPool({
      host: config.host,
      port: config.port,
      max: config.maxConnections,
      idleTimeoutMillis: config.idleTimeout,
      connectionTimeoutMillis: config.connectionTimeout
    });
  }
  
  async acquire(): Promise<PoolConnection> {
    return this.pool.connect();
  }
  
  release(conn: PoolConnection): void {
    conn.release();
  }
  
  async close(): Promise<void> {
    await this.pool.end();
  }
}
```

### 2.4 限流策略

```typescript
interface RateLimitConfig {
  windowMs: number;      // 时间窗口
  maxRequests: number;    // 最大请求数
  keyPrefix: string;
}

class RateLimiter {
  private windows: Map<string, number[]> = new Map();
  
  async checkLimit(key: string): Promise<boolean> {
    const now = Date.now();
    const window = this.windows.get(key) || [];
    
    // 清理过期请求
    const valid = window.filter(t => now - t < this.config.windowMs);
    
    if (valid.length >= this.config.maxRequests) {
      return false;
    }
    
    valid.push(now);
    this.windows.set(key, valid);
    return true;
  }
  
  getRetryAfter(key: string): number {
    const window = this.windows.get(key) || [];
    if (window.length < this.config.maxRequests) return 0;
    
    const oldest = Math.min(...window);
    return this.config.windowMs - (Date.now() - oldest);
  }
}
```

## 3. 资源管理

### 3.1 内存管理

```typescript
// 内存池
class MemoryPool {
  private pool: Map<number, Buffer[]> = new Map();
  private readonly CHUNK_SIZE = 1024 * 1024; // 1MB
  
  allocate(size: number): Buffer {
    const chunks = Math.ceil(size / this.CHUNK_SIZE);
    const buffers: Buffer[] = [];
    
    for (let i = 0; i < chunks; i++) {
      const available = this.pool.get(i) || [];
      if (available.length > 0) {
        buffers.push(available.pop()!);
      } else {
        buffers.push(Buffer.alloc(this.CHUNK_SIZE));
      }
    }
    
    return Buffer.concat(buffers);
  }
  
  release(buffer: Buffer): void {
    const chunks = Math.ceil(buffer.length / this.CHUNK_SIZE);
    for (let i = 0; i < chunks; i++) {
      const chunk = buffer.subarray(
        i * this.CHUNK_SIZE, 
        Math.min((i + 1) * this.CHUNK_SIZE, buffer.length)
      );
      
      const available = this.pool.get(i) || [];
      available.push(chunk);
      this.pool.set(i, available);
    }
  }
}
```

### 3.2 CPU资源管理

```typescript
// 工作队列
class WorkerQueue {
  private queue: Task[] = [];
  private workers: Worker[];
  private running: number = 0;
  
  constructor(workerCount: number) {
    this.workers = Array.from(
      { length: workerCount }, 
      () => new Worker()
    );
  }
  
  async add(task: Task): Promise<any> {
    return new Promise((resolve, reject) => {
      this.queue.push({ task, resolve, reject });
      this.processQueue();
    });
  }
  
  private async processQueue(): Promise<void> {
    if (this.running >= this.workers.length) return;
    if (this.queue.length === 0) return;
    
    const worker = this.workers[this.running++];
    const { task, resolve, reject } = this.queue.shift()!;
    
    try {
      const result = await worker.run(task);
      resolve(result);
    } catch (error) {
      reject(error);
    } finally {
      this.running--;
      this.processQueue();
    }
  }
}
```

### 3.3 资源监控

```typescript
interface ResourceMetrics {
  cpu: {
    usage: number;
    cores: number;
  };
  memory: {
    used: number;
    total: number;
    limit: number;
  };
  disk: {
    read: number;
    write: number;
  };
  network: {
    in: number;
    out: number;
  };
}

class ResourceMonitor {
  private metrics: ResourceMetrics[] = [];
  
  async collectMetrics(): Promise<ResourceMetrics> {
    const cpu = await this.getCpuUsage();
    const memory = await this.getMemoryUsage();
    const disk = await this.getDiskUsage();
    const network = await this.getNetworkUsage();
    
    return { cpu, memory, disk, network };
  }
  
  getAverageMetrics(): ResourceMetrics {
    const count = this.metrics.length;
    if (count === 0) return null;
    
    return {
      cpu: { 
        usage: this.avg(m => m.cpu.usage), 
        cores: this.metrics[0].cpu.cores 
      },
      memory: this.avgObj(m => m.memory),
      disk: this.avgObj(m => m.disk),
      network: this.avgObj(m => m.network)
    };
  }
}
```

### 3.4 自动扩缩容

```typescript
interface ScalingConfig {
  minReplicas: number;
  maxReplicas: number;
  scaleUpThreshold: number;
  scaleDownThreshold: number;
  cooldown: number;
}

class AutoScaler {
  private lastScaleTime: number = 0;
  
  async shouldScale(metrics: ResourceMetrics): Promise<{
    action: 'up' | 'down' | 'none';
    replicas: number;
  }> {
    const cpuUsage = metrics.cpu.usage;
    const currentReplicas = this.getCurrentReplicas();
    
    // 扩容
    if (cpuUsage > this.config.scaleUpThreshold) {
      if (currentReplicas < this.config.maxReplicas) {
        return {
          action: 'up',
          replicas: Math.min(
            currentReplicas + 1,
            this.config.maxReplicas
          )
        };
      }
    }
    
    // 缩容
    if (cpuUsage < this.config.scaleDownThreshold) {
      if (currentReplicas > this.config.minReplicas) {
        return {
          action: 'down',
          replicas: Math.max(
            currentReplicas - 1,
            this.config.minReplicas
          )
        };
      }
    }
    
    return { action: 'none', replicas: currentReplicas };
  }
}
```

## 总结

本课程涵盖：
- ✅ 多级缓存策略
- ✅ 并发优化技术
- ✅ 资源管理方案
- ✅ 自动扩缩容

---
*课程时长：2小时*
