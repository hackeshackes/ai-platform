# 实战项目 - 4小时

完整的项目规划、实现、测试和部署流程。

## 1. 项目规划

### 1.1 项目概述

**项目名称**: 智能客服系统

**项目目标**: 构建一个基于AI Platform的智能客服系统，能够：
- 自动回复常见问题
- 智能转接人工客服
- 知识库检索增强
- 多轮对话上下文管理

### 1.2 功能需求

```
功能清单：
┌─────────────────────────────────────────────────────────────┐
│ 核心功能                                                     │
├─────────────────────────────────────────────────────────────┤
│  ✅ 自动问答                                                 │
│  ✅ 意图识别                                                 │
│  ✅ 知识库检索                                               │
│  ✅ 人工转接                                                 │
│  ✅ 对话历史管理                                             │
├─────────────────────────────────────────────────────────────┤
│ 扩展功能                                                     │
├─────────────────────────────────────────────────────────────┤
│  ⭕ 情感分析                                                 │
│  ⭕ 满意度评价                                               │
│  ⭕ 报表统计                                                 │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                     前端层 (React)                          │
├─────────────────────────────────────────────────────────────┤
│                     API网关                                  │
├─────────────────────────────────────────────────────────────┤
│                  AI Platform                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐    │
│  │ 对话Agent   │ │ 知识库      │ │ Pipeline编排        │    │
│  └─────────────┘ └─────────────┘ └─────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                  数据层 (PostgreSQL/Redis/ES)                │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 项目结构

```
smart-customer-service/
├── src/
│   ├── api/              # API接口
│   ├── agents/           # Agent定义
│   ├── pipelines/        # Pipeline定义
│   ├── services/        # 业务服务
│   ├── models/          # 数据模型
│   └── utils/           # 工具函数
├── tests/               # 测试
├── config/              # 配置
├── scripts/             # 脚本
├── docs/                # 文档
├── Dockerfile
└── README.md
```

## 2. 完整实现

### 2.1 Agent实现

```typescript
// agents/ChatAgent.ts
import { BaseAgent, AgentConfig } from './base';

interface ChatConfig extends AgentConfig {
  maxHistoryLength: number;
  enableKnowledgeBase: boolean;
}

export class ChatAgent extends BaseAgent {
  constructor(
    private config: ChatConfig,
    private knowledgeService: KnowledgeService
  ) {
    super(config);
  }
  
  async process(input: string, context: AgentContext): Promise<AgentResponse> {
    // 1. 获取对话历史
    const history = await this.getConversationHistory(
      context.conversationId
    );
    
    // 2. 意图识别
    const intent = await this.recognizeIntent(input, history);
    
    // 3. 根据意图处理
    switch (intent.type) {
      case 'faq':
        return this.handleFAQ(intent);
      case 'order':
        return this.handleOrderQuery(intent);
      case 'handover':
        return this.initiateHandover(intent);
      default:
        return this.handleGeneral(input, history);
    }
  }
  
  private async handleFAQ(intent: Intent): Promise<AgentResponse> {
    const knowledge = await this.knowledgeService.search(
      intent.query
    );
    
    const response = await this.generateResponse(
      knowledge,
      intent.context
    );
    
    return {
      type: 'text',
      content: response,
      metadata: { source: 'knowledge_base' }
    };
  }
}
```

### 2.2 Pipeline实现

```typescript
// pipelines/ConversationPipeline.ts
import { Pipeline, Node, Edge } from './base';

export class ConversationPipeline extends Pipeline {
  constructor(
    private intentClassifier: IntentClassifier,
    private chatAgent: ChatAgent
  ) {
    super('conversation');
    this.buildPipeline();
  }
  
  private buildPipeline(): void {
    const input = new Node('input', 'InputNode');
    const classify = new Node('classify', 'AgentNode', {
      agent: this.intentClassifier
    });
    const process = new Node('process', 'AgentNode', {
      agent: this.chatAgent
    });
    const output = new Node('output', 'OutputNode');
    
    this.addNode(input);
    this.addNode(classify);
    this.addNode(process);
    this.addNode(output);
    
    this.addEdge(input, 'text', classify, 'input');
    this.addEdge(classify, 'intent', process, 'intent');
    this.addEdge(process, 'response', output, 'input');
  }
  
  async execute(input: ConversationInput): Promise<ConversationResult> {
    return super.execute(input);
  }
}
```

### 2.3 服务层实现

```typescript
// services/ConversationService.ts
import { EventEmitter } from 'events';

export class ConversationService extends EventEmitter {
  constructor(
    private agent: ChatAgent,
    private store: ConversationStore
  ) {
    super();
  }
  
  async handleMessage(
    conversationId: string,
    message: UserMessage
  ): Promise<BotResponse> {
    // 验证会话
    const session = await this.validateSession(conversationId);
    
    // 保存用户消息
    await this.store.addMessage(conversationId, {
      role: 'user',
      content: message.text,
      timestamp: new Date()
    });
    
    // 处理消息
    const response = await this.agent.process(
      message.text,
      { conversationId, userId: session.userId }
    );
    
    // 保存响应
    await this.store.addMessage(conversationId, {
      role: 'assistant',
      content: response.content,
      timestamp: new Date()
    });
    
    return response;
  }
  
  private async validateSession(conversationId: string): Promise<Session> {
    let session = await this.store.getSession(conversationId);
    if (!session) {
      session = await this.store.createSession(conversationId);
    }
    return session;
  }
}
```

### 2.4 API层实现

```typescript
// api/routes/conversation.ts
import { Router, Request, Response } from 'express';

export function createConversationRoutes(
  service: ConversationService
): Router {
  const router = Router();
  
  router.post('/:conversationId/messages', async (req, res) => {
    try {
      const { conversationId } = req.params;
      const { text } = req.body;
      
      const response = await service.handleMessage(
        conversationId,
        { text }
      );
      
      res.json({ success: true, data: response });
    } catch (error) {
      res.status(500).json({ success: false, error: error.message });
    }
  });
  
  router.get('/:conversationId/history', async (req, res) => {
    const messages = await service.getHistory(
      req.params.conversationId
    );
    res.json({ success: true, data: messages });
  });
  
  return router;
}
```

## 3. 测试部署

### 3.1 测试用例

```typescript
// tests/conversation.test.ts
describe('ConversationService', () => {
  it('should handle user message', async () => {
    mockAgent.process.mockResolvedValue({
      type: 'text',
      content: 'Hello!'
    });
    
    const response = await service.handleMessage(
      'conv-123',
      { text: 'Hi' }
    );
    
    expect(response.content).toBe('Hello!');
    expect(mockAgent.process).toHaveBeenCalled();
  });
});
```

### 3.2 Docker部署

```dockerfile
# Dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
```

```bash
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - postgres
      - redis
```

### 3.3 K8s部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smart-customer-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: smart-customer-service
  template:
    metadata:
      labels:
        app: smart-customer-service
    spec:
      containers:
      - name: app
        image: smart-customer-service:latest
        ports:
        - containerPort: 3000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
```

### 3.4 监控配置

```yaml
# monitoring.yaml
metrics:
  enabled: true
  endpoint: /metrics
  
alerts:
  - condition: error_rate > 0.05
    severity: warning
  - condition: latency_p95 > 3000
    severity: warning
```

## 4. 项目总结

### 完成情况

| 模块 | 状态 | 说明 |
|-----|------|-----|
| Agent开发 | ✅ 完成 | 意图识别、知识库集成 |
| Pipeline编排 | ✅ 完成 | 对话流程控制 |
| API开发 | ✅ 完成 | RESTful接口 |
| 测试 | ✅ 完成 | 单元+集成测试 |
| 部署 | ✅ 完成 | Docker+K8s |

### 经验总结

1. **设计阶段**: 明确需求，合理划分模块
2. **开发阶段**: 遵循规范，注重测试
3. **部署阶段**: 渐进发布，做好监控

### 后续优化

- 性能优化：批处理、缓存
- 功能扩展：多渠道接入
- 体验提升：个性化推荐

## 实践任务

1. **任务1**: 按照本项目结构创建完整项目
2. **任务2**: 实现对话Agent核心逻辑
3. **任务3**: 编写测试并部署到K8s

---
*课程时长：4小时*
