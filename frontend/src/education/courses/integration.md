# 集成开发 - 3小时

学习API集成、Webhook和第三方服务集成。

## 1. API集成

### 1.1 API客户端封装

```typescript
// api/client.ts
class APIClient {
  private baseUrl: string;
  private headers: Record<string, string>;
  
  constructor(config: APIConfig) {
    this.baseUrl = config.baseUrl;
    this.headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${config.apiKey}`,
      ...config.customHeaders
    };
  }
  
  async request<T>(
    method: string,
    path: string,
    data?: any
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const options: RequestInit = {
      method,
      headers: this.headers,
      body: data ? JSON.stringify(data) : undefined
    };
    
    const response = await fetch(url, options);
    
    if (!response.ok) {
      throw new APIError(
        response.status,
        response.statusText,
        await response.json()
      );
    }
    
    return response.json();
  }
  
  get<T>(path: string): Promise<T> {
    return this.request('GET', path);
  }
  
  post<T>(path: string, data?: any): Promise<T> {
    return this.request('POST', path, data);
  }
  
  put<T>(path: string, data?: any): Promise<T> {
    return this.request('PUT', path, data);
  }
  
  delete<T>(path: string): Promise<T> {
    return this.request('DELETE', path);
  }
}
```

### 1.2 认证配置

```typescript
// auth/token-manager.ts
class TokenManager {
  private token: string | null = null;
  private refreshToken: string | null = null;
  private tokenExpiry: number = 0;
  
  async getToken(): Promise<string> {
    if (this.token && Date.now() < this.tokenExpiry) {
      return this.token;
    }
    
    await this.refreshAccessToken();
    return this.token!;
  }
  
  private async refreshAccessToken(): Promise<void> {
    const response = await this.client.post('/auth/refresh', {
      refresh_token: this.refreshToken
    });
    
    this.token = response.access_token;
    this.refreshToken = response.refresh_token;
    this.tokenExpiry = Date.now() + response.expires_in * 1000;
  }
}

// OAuth2集成
class OAuthClient {
  async getAuthorizationUrl(state: string): Promise<string> {
    const params = new URLSearchParams({
      client_id: this.config.clientId,
      redirect_uri: this.config.redirectUri,
      response_type: 'code',
      scope: this.config.scopes.join(' '),
      state
    });
    
    return `${this.config.authUrl}?${params.toString()}`;
  }
  
  async exchangeCode(code: string): Promise<TokenResponse> {
    return this.client.post('/oauth/token', {
      grant_type: 'authorization_code',
      code,
      client_id: this.config.clientId,
      client_secret: this.config.clientSecret,
      redirect_uri: this.config.redirectUri
    });
  }
}
```

### 1.3 API版本管理

```typescript
// api/versioning.ts
class VersionedAPIClient {
  private clients: Map<string, APIClient> = new Map();
  
  constructor(
    private defaultVersion: string,
    private versions: Record<string, APIClient>
  ) {
    this.clients = new Map(Object.entries(versions));
  }
  
  getClient(version?: string): APIClient {
    const v = version || this.defaultVersion;
    const client = this.clients.get(v);
    if (!client) {
      throw new Error(`API version ${v} not supported`);
    }
    return client;
  }
  
  // 兼容旧版本
  async request<T>(
    method: string,
    path: string,
    data?: any,
    version?: string
  ): Promise<T> {
    const client = this.getClient(version);
    return client.request(method, path, data);
  }
}
```

### 1.4 错误重试

```typescript
// api/retry.ts
async function withRetry<T>(
  fn: () => Promise<T>,
  config: RetryConfig
): Promise<T> {
  let lastError: Error | null = null;
  
  for (let attempt = 0; attempt < config.maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      
      if (!this.isRetryable(error)) {
        throw error;
      }
      
      const delay = this.calculateDelay(attempt, config);
      await this.sleep(delay);
    }
  }
  
  throw lastError;
}

private calculateDelay(
  attempt: number,
  config: RetryConfig
): number {
  if (config.strategy === 'fixed') {
    return config.delay;
  }
  
  if (config.strategy === 'exponential') {
    return Math.min(
      config.delay * Math.pow(2, attempt),
      config.maxDelay
    );
  }
  
  if (config.strategy === 'jitter') {
    return Math.random() * config.delay;
  }
  
  return config.delay;
}
```

## 2. Webhook

### 2.1 Webhook配置

```typescript
// webhook/config.ts
interface WebhookConfig {
  url: string;
  events: WebhookEvent[];
  secret: string;
  retryPolicy: {
    maxAttempts: number;
    delay: number;
  };
}

type WebhookEvent = 
  | 'agent.created'
  | 'agent.updated'
  | 'pipeline.completed'
  | 'pipeline.failed'
  | 'message.received';

// 注册Webhook
async function registerWebhook(
  config: WebhookConfig
): Promise<WebhookRegistration> {
  return apiClient.post('/webhooks', {
    url: config.url,
    events: config.events,
    secret: config.secret,
    retry_policy: config.retryPolicy
  });
}
```

### 2.2 Webhook签名验证

```typescript
// webhook/verify.ts
class WebhookVerifier {
  async verify(
    payload: string,
    signature: string,
    secret: string
  ): Promise<boolean> {
    const expectedSignature = this.generateSignature(
      payload,
      secret
    );
    
    return crypto.timingSafeEqual(
      Buffer.from(signature),
      Buffer.from(expectedSignature)
    );
  }
  
  private generateSignature(
    payload: string,
    secret: string
  ): string {
    return crypto
      .createHmac('sha256', secret)
      .update(payload)
      .digest('hex');
  }
}
```

### 2.3 Webhook处理

```typescript
// webhook/handler.ts
class WebhookHandler {
  constructor(
    private verifier: WebhookVerifier,
    private handlers: Map<string, EventHandler>
  ) {}
  
  async handleRequest(
    req: IncomingMessage,
    body: Buffer
  ): Promise<Response> {
    // 验证签名
    const signature = req.headers['x-webhook-signature'];
    const isValid = await this.verifier.verify(
      body.toString(),
      signature as string,
      process.env.WEBHOOK_SECRET!
    );
    
    if (!isValid) {
      throw new Error('Invalid signature');
    }
    
    // 解析事件
    const event = JSON.parse(body.toString());
    const handler = this.handlers.get(event.type);
    
    if (!handler) {
      return { status: 200 };
    }
    
    // 处理事件
    await handler(event.data);
    
    return { status: 200 };
  }
}
```

### 2.4 发送Webhook

```typescript
// webhook/sender.ts
class WebhookSender {
  async send(
    event: string,
    data: any,
    webhook: WebhookConfig
  ): Promise<WebhookResult> {
    const payload = JSON.stringify({
      event,
      timestamp: new Date().toISOString(),
      data
    });
    
    const signature = this.generateSignature(
      payload,
      webhook.secret
    );
    
    for (let attempt = 0; attempt < webhook.retryPolicy.maxAttempts; attempt++) {
      try {
        const response = await fetch(webhook.url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Webhook-Signature': signature,
            'X-Webhook-Event': event
          },
          body: payload
        });
        
        return {
          success: response.ok,
          status: response.status,
          attempt
        };
      } catch (error) {
        if (attempt === webhook.retryPolicy.maxAttempts - 1) {
          throw error;
        }
        await this.sleep(webhook.retryPolicy.delay);
      }
    }
  }
}
```

## 3. 第三方集成

### 3.1 常用集成

```typescript
// integrations/index.ts
interface Integration {
  name: string;
  version: string;
  initialize(config: any): Promise<void>;
  execute(action: string, params: any): Promise<any>;
}

// 集成注册
const integrations: Map<string, Integration> = new Map();

function register(integration: Integration): void {
  integrations.set(integration.name, integration);
}

// 使用集成
async function useIntegration(
  name: string,
  action: string,
  params: any
): Promise<any> {
  const integration = integrations.get(name);
  if (!integration) {
    throw new Error(`Integration ${name} not found`);
  }
  
  return integration.execute(action, params);
}
```

### 3.2 数据库集成

```typescript
// integrations/database.ts
class DatabaseIntegration {
  private pool: Pool;
  
  async connect(config: DatabaseConfig): Promise<void> {
    this.pool = createPool({
      host: config.host,
      port: config.port,
      database: config.database,
      user: config.user,
      password: config.password,
      max: config.poolSize
    });
  }
  
  async query<T>(sql: string, params?: any[]): Promise<T[]> {
    const result = await this.pool.query(sql, params);
    return result.rows;
  }
  
  async transaction<T>(
    fn: (client: PoolClient) => Promise<T>
  ): Promise<T> {
    const client = await this.pool.connect();
    try {
      await client.query('BEGIN');
      const result = await fn(client);
      await client.query('COMMIT');
      return result;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }
}
```

### 3.3 消息队列集成

```typescript
// integrations/messaging.ts
class MessagingIntegration {
  private consumer: Consumer;
  private producer: Producer;
  
  async connect(config: MQConfig): Promise<void> {
    this.consumer = createConsumer({
      brokers: config.brokers,
      groupId: config.groupId
    });
    
    this.producer = createProducer({
      brokers: config.brokers
    });
  }
  
  async publish(topic: string, message: any): Promise<void> {
    await this.producer.send({
      topic,
      messages: [{
        key: message.id,
        value: JSON.stringify(message)
      }]
    });
  }
  
  async subscribe(
    topic: string,
    handler: (message: any) => Promise<void>
  ): Promise<void> {
    await this.consumer.subscribe({ topic });
    
    await this.consumer.run({
      eachMessage: async ({ message }) => {
        const data = JSON.parse(message.value!.toString());
        await handler(data);
      }
    });
  }
}
```

### 3.4 监控集成

```typescript
// integrations/monitoring.ts
class MonitoringIntegration {
  private metrics: MetricsClient;
  private traces: Tracer;
  
  async connect(config: MonitoringConfig): Promise<void> {
    this.metrics = createMetrics({
      endpoint: config.metricsEndpoint,
      apiKey: config.apiKey
    });
    
    this.traces = createTracer({
      endpoint: config.tracesEndpoint,
      serviceName: config.serviceName
    });
  }
  
  recordMetric(name: string, value: number, tags?: Record<string, string>): void {
    this.metrics.record(name, value, tags);
  }
  
  startSpan(name: string): Span {
    return this.traces.startSpan(name);
  }
  
  async flush(): Promise<void> {
    await Promise.all([
      this.metrics.flush(),
      this.traces.flush()
    ]);
  }
}
```

## 4. 集成最佳实践

### 4.1 集成配置

```yaml
# integrations.yaml
integrations:
  # 数据库
  database:
    enabled: true
    type: postgresql
    host: ${DB_HOST}
    port: 5432
    database: ${DB_NAME}
    pool_size: 10
    
  # 消息队列
  messaging:
    enabled: true
    type: kafka
    brokers:
      - ${KAFKA_BROKER}
    group_id: ai-platform
    
  # 监控
  monitoring:
    enabled: true
    provider: datadog
    api_key: ${DATADOG_API_KEY}
    
  # 日志
  logging:
    enabled: true
    provider: elk
    hosts:
      - ${ELK_HOST}
```

### 4.2 集成测试

```typescript
describe('Integration Tests', () => {
  it('should connect to database', async () => {
    const db = new DatabaseIntegration();
    await db.connect(testConfig.database);
    
    const result = await db.query('SELECT 1 as test');
    expect(result[0].test).toBe(1);
  });
  
  it('should send and receive messages', async () => {
    const messaging = new MessagingIntegration();
    await messaging.connect(testConfig.messaging);
    
    const received: any[] = [];
    await messaging.subscribe('test-topic', 
      (msg) => received.push(msg)
    );
    
    await messaging.publish('test-topic', { 
      id: 'test-123' 
    });
    
    await wait(1000);
    expect(received.length).toBe(1);
  });
});
```

## 实践任务

1. **任务1**: 实现一个API客户端，包含认证和重试
2. **任务2**: 创建Webhook处理系统
3. **任务3**: 集成数据库和消息队列

## 总结

本课程涵盖：
- ✅ API客户端封装和认证
- ✅ Webhook配置和处理
- ✅ 第三方服务集成
- ✅ 集成最佳实践

---
*课程时长：3小时*
