# 模板使用 - 30分钟

选择合适模板、自定义配置和部署运行。

## 1. 选择模板

### 1.1 模板类型

```
模板分类：
┌─────────────────────────────────────────────────────────┐
│  Agent模板                                               │
├─────────────────────────────────────────────────────────┤
│  • 对话型Agent模板                                       │
│  • 任务型Agent模板                                       │
│  • 知识问答模板                                          │
│  • 代码助手模板                                          │
├─────────────────────────────────────────────────────────┤
│  Pipeline模板                                           │
├─────────────────────────────────────────────────────────┤
│  • 文本处理Pipeline                                      │
│  • 客服对话Pipeline                                      │
│  • 数据抽取Pipeline                                      │
│  • RAG检索Pipeline                                       │
├─────────────────────────────────────────────────────────┤
│  项目模板                                                │
├─────────────────────────────────────────────────────────┤
│  • 企业级应用模板                                        │
│  • 原型开发模板                                          │
│  • 生产部署模板                                          │
└─────────────────────────────────────────────────────────┘
```

### 1.2 模板选择指南

| 需求 | 推荐模板 | 复杂度 |
|-----|---------|-------|
| 简单客服 | 对话型Agent | ⭐ |
| 复杂问答 | 知识库+RAG | ⭐⭐ |
| 自动化流程 | 任务型Pipeline | ⭐⭐⭐ |
| 代码助手 | 代码型Agent | ⭐⭐ |
| 完整应用 | 项目模板 | ⭐⭐⭐⭐ |

### 1.3 模板浏览

```bash
# 列出所有模板
template list

# 查看模板详情
template show chatbot-agent

# 搜索模板
template search "客服"

# 模板分类查看
template list --category agent
template list --category pipeline
template list --category project
```

## 2. 自定义配置

### 2.1 配置模板

```yaml
# template-config.yaml
template: chatbot-agent
version: "1.0.0"

# 自定义配置
config:
  # Agent配置
  agent:
    name: "my-chatbot"
    model: "gpt-4"
    temperature: 0.7
    
  # 技能配置
  skills:
    - intent_recognition
    - sentiment_analysis
    - knowledge_retrieval
    
  # 记忆配置
  memory:
    type: redis
    ttl: 3600
    
  # UI配置
  ui:
    theme: "dark"
    welcome_message: "你好，有什么可以帮助你？"
```

### 2.2 环境变量配置

```bash
# .env.template
# 模型配置
MODEL_NAME=gpt-4
MODEL_API_KEY=${OPENAI_API_KEY}
MODEL_ENDPOINT=https://api.openai.com/v1

# 数据库配置
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379

# 应用配置
APP_ENV=development
APP_PORT=3000
LOG_LEVEL=info
```

### 2.3 高级配置

```typescript
// advanced.config.ts
export default {
  // 性能配置
  performance: {
    maxConcurrentRequests: 100,
    requestTimeout: 30000,
    batchSize: 10,
    batchTimeout: 1000,
    cacheEnabled: true,
    cacheTTL: 3600
  },
  
  // 监控配置
  monitoring: {
    metricsEnabled: true,
    tracingEnabled: true,
    logLevel: 'info',
    alertThreshold: {
      errorRate: 0.05,
      latencyP95: 3000,
      memoryUsage: 0.8
    }
  },
  
  // 安全配置
  security: {
    authentication: 'bearer',
    rateLimiting: {
      windowMs: 60000,
      maxRequests: 100
    },
    corsOrigins: ['http://localhost:3000']
  }
};
```

### 2.4 模板变量

```yaml
# variables.yaml
variables:
  app_name:
    description: "应用名称"
    type: string
    required: true
    default: "my-app"
    
  model_provider:
    description: "模型提供商"
    type: enum
    options:
      - openai
      - anthropic
      - local
    default: "openai"
    
  enable_features:
    description: "启用的功能"
    type: array
    items:
      type: string
      enum:
        - streaming
        - caching
        - monitoring
```

## 3. 部署运行

### 3.1 本地开发

```bash
# 初始化模板
template init my-chatbot --template chatbot-agent

# 进入目录
cd my-chatbot

# 安装依赖
npm install

# 配置环境变量
cp .env.template .env
# 编辑 .env 文件

# 启动开发服务器
npm run dev

# 运行测试
npm test
```

### 3.2 Docker部署

```dockerfile
# Dockerfile
FROM node:18-alpine

WORKDIR /app

# 复制依赖文件
COPY package*.json ./

# 安装依赖
RUN npm ci --only=production

# 复制应用代码
COPY . .

# 构建
RUN npm run build

# 暴露端口
EXPOSE 3000

# 启动
CMD ["npm", "start"]
```

```bash
# 构建镜像
docker build -t my-chatbot:latest .

# 运行容器
docker run -d \
  --name my-chatbot \
  -p 3000:3000 \
  -v $(pwd)/config:/app/config \
  my-chatbot:latest
```

### 3.3 Kubernetes部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-chatbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-chatbot
  template:
    metadata:
      labels:
        app: my-chatbot
    spec:
      containers:
      - name: chatbot
        image: my-chatbot:latest
        ports:
        - containerPort: 3000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secrets
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 3.4 部署检查

```bash
# 健康检查
curl http://localhost:3000/health

# 监控指标
curl http://localhost:3000/metrics

# 查看日志
kubectl logs -f deployment/my-chatbot

# 扩缩容
kubectl scale deployment my-chatbot --replicas=5
```

## 4. 模板自定义示例

### 4.1 创建自定义模板

```bash
# 从现有项目创建模板
template init my-custom-template \
  --from /path/to/existing/project \
  --output /templates/my-custom
```

### 4.2 模板结构

```
my-custom-template/
├── template.yaml        # 模板定义
├── config.yaml          # 默认配置
├── README.md           # 模板说明
├── template/           # 模板文件
│   ├── src/
│   ├── tests/
│   └── package.json
└── variables.yaml      # 变量定义
```

## 实践任务

1. **任务1**: 使用对话型Agent模板创建客服机器人
2. **任务2**: 自定义模板配置，添加特定技能
3. **任务3**: 使用Docker部署到本地环境

## 总结

本课程涵盖：
- ✅ 模板类型和选择
- ✅ 自定义配置方法
- ✅ 多环境部署
- ✅ 自定义模板创建

---
*课程时长：30分钟*
