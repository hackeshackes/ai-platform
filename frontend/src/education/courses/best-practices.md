# 最佳实践 - 2小时

学习项目结构、代码规范和测试策略。

## 1. 项目结构

### 推荐目录结构
```
ai-platform-project/
├── src/
│   ├── agents/              # Agent模块
│   │   ├── base/
│   │   ├── conversation/
│   │   ├── task/
│   │   └── index.ts
│   ├── pipelines/           # Pipeline模块
│   │   ├── base/
│   │   ├── nodes/
│   │   ├── templates/
│   │   └── index.ts
│   ├── components/          # React组件
│   ├── services/            # 服务层
│   │   ├── api/
│   │   ├── storage/
│   │   └── monitoring/
│   ├── utils/               # 工具函数
│   ├── types/               # 类型定义
│   ├── hooks/               # React Hooks
│   ├── config/              # 配置
│   ├── app.ts
│   └── main.ts
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
├── static/
├── package.json
├── tsconfig.json
└── README.md
```

## 2. 代码规范

### TypeScript规范
```typescript
// 类型定义优先
type AgentStatus = 'idle' | 'running' | 'error';

// 导出类型
export type PipelineConfig = {
  nodes: Node[];
  edges: Edge[];
};

// 使用泛型约束
function processItems<T extends Processable>(items: T[]): T[] {
  return items.map(item => item.process());
}
```

### 函数设计原则
```typescript
// 单一职责
async function createAgent(input: CreateAgentInput): Promise<Agent> {
  validateInput(input);
  const agent = await repository.create(input);
  await eventEmitter.emit('agent.created', agent);
  return agent;
}

// 参数对象化
interface UpdateAgentOptions {
  name?: string;
  skills?: string[];
  memory?: MemoryConfig;
}
```

### 错误处理
```typescript
// Result模式
type Result<T, E = Error> = 
  | { ok: true; value: T }
  | { ok: false; error: E };

// 统一错误处理
class AppError extends Error {
  constructor(
    public code: string,
    public statusCode: number,
    message: string
  ) {
    super(message);
  }
}
```

## 3. 测试策略

### 测试金字塔
```
        ┌───────┐
       /   E2E   \      10%
      /───────────\
     /   集成测试   \    30%
    /───────────────\
   /     单元测试     \  60%
  /───────────────────\
```

### 单元测试
```typescript
describe('Agent', () => {
  let agent: Agent;
  
  beforeEach(() => {
    agent = new Agent({
      model: 'gpt-4',
      skills: []
    });
  });
  
  it('should process message', async () => {
    const response = await agent.process('Hello');
    expect(response).toBeDefined();
  });
  
  it('should handle errors gracefully', async () => {
    await expect(
      agent.process('invalid input')
    ).rejects.toThrow();
  });
});
```

### 集成测试
```typescript
describe('Pipeline Integration', () => {
  it('should execute full pipeline', async () => {
    const pipeline = new Pipeline(config);
    const result = await pipeline.execute({
      trigger: { type: 'webhook' }
    });
    
    expect(result.status).toBe('success');
    expect(result.output).toBeDefined();
  });
});
```

### 测试覆盖要求
- 单元测试覆盖：>80%
- 集成测试覆盖：>60%
- 关键路径E2E测试：100%

## 4. 代码审查清单

### 提交前检查
- [ ] TypeScript编译无错误
- [ ] 单元测试全部通过
- [ ] 代码格式符合规范
- [ ] 无console.log调试代码
- [ ] 必要的注释已添加

### 审查要点
- [ ] 单一职责原则
- [ ] 依赖注入
- [ ] 错误处理完整
- [ ] 性能考虑
- [ ] 安全性检查

## 总结

本课程涵盖：
- ✅ 项目目录结构
- ✅ TypeScript编码规范
- ✅ 测试策略和实践
- ✅ 代码审查流程

---
*课程时长：2小时*
