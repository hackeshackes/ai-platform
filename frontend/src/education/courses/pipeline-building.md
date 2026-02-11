# Pipeline构建 - 1小时

学习Pipeline设计理念、节点连接、条件分支和错误处理。

## 1. Pipeline概念

Pipeline是AI Platform中用于编排复杂工作流的核心组件。

### 1.1 核心概念

```
Pipeline = 节点(Node) + 边(Edge) + 配置(Config)
```

**节点(Node)**: Pipeline中的基本处理单元
- 触发器节点：启动Pipeline
- 处理节点：数据转换和处理
- 条件节点：逻辑判断
- 输出节点：返回结果

**边(Edge)**: 节点之间的数据流动路径
- 数据流：传递处理结果
- 控制流：决定执行路径

### 1.2 Pipeline类型

| 类型 | 描述 | 适用场景 |
|-----|------|---------|
| 同步Pipeline | 阻塞式执行，立即返回结果 | 简单任务处理 |
| 异步Pipeline | 非阻塞执行，支持长时间任务 | 批处理、大模型推理 |
| 定时Pipeline | 按计划自动执行 | 数据同步、定期报告 |
| 事件Pipeline | 响应事件触发 | 实时处理、Webhook |

### 1.3 Pipeline结构示例

```yaml
pipeline:
  name: text-processing-pipeline
  version: "1.0.0"
  type: synchronous
  
  nodes:
    - id: trigger
      type: webhook
      config:
        method: POST
        
    - id: text_preprocessor
      type: processor
      input: trigger.output
      config:
        operations:
          - normalize
          - remove_special_chars
          
    - id: sentiment_analyzer
      type: agent
      input: text_preprocessor.output
      agent_id: sentiment-agent
      
    - id: result_formatter
      type: formatter
      input: sentiment_analyzer.output
      config:
        format: json
```

## 2. 节点连接

### 2.1 节点类型详解

#### 触发器节点

```yaml
# Webhook触发器
- id: webhook_trigger
  type: webhook
  config:
    path: /api/webhook
    method: POST
    authentication: bearer
    
# 定时触发器
- id: scheduled_trigger
  type: schedule
  config:
    cron: "0 0 * * *"  # 每天0点
    timezone: Asia/Shanghai
    
# 事件触发器
- id: event_trigger
  type: event
  config:
    event_type: message.created
    filters:
      channel: support
```

#### 处理节点

```yaml
# 数据转换
- id: data_transformer
  type: transformer
  input: previous_node.output
  config:
    mappings:
      - field: text
        transform: trim
      - field: word_count
        transform: count_words
        
# 条件分支
- id: router
  type: router
  input: transformer.output
  config:
    routes:
      - condition: "length(text) > 1000"
        output: long_text_handler
      - condition: "length(text) <= 1000"
        output: short_text_handler
```

#### Agent节点

```yaml
- id: main_agent
  type: agent
  input: trigger.output
  agent_id: my-agent-001
  config:
    timeout: 30000
    retry: 3
    fallback: error_handler
```

### 2.2 连接语法

```yaml
# 简单连接
nodes:
  - id: node_a
    type: ...
  - id: node_b
    type: ...
    input: node_a.output  # 连接到node_a
    
# 条件连接
- id: node_c
  type: ...
  input:
    on_success: node_a.output
    on_failure: fallback_node.output
```

### 2.3 数据传递

```typescript
interface NodeOutput {
  data: Record<string, any>;
  metadata: {
    node_id: string;
    timestamp: number;
    execution_time: number;
  };
}

// 在下一个节点中访问
const inputData = context.input.previous_node.data;
```

## 3. 条件分支

### 3.1 条件表达式

```yaml
config:
  conditions:
    # 简单条件
    - name: is_long_text
      expression: "length(input.text) > 1000"
      
    # 复杂条件
    - name: is_vip_user
      expression: "user.tier == 'vip' && user.score > 1000"
      
    # 正则匹配
    - name: contains_keywords
      expression: "input.text =~ /urgent|important/"
```

### 3.2 分支路由

```yaml
- id: content_router
  type: router
  input: trigger.output
  config:
    branches:
      - name: urgent_path
        condition: "contains_keywords('urgent')"
        priority: 1
        
      - name: vip_path
        condition: "is_vip_user"
        priority: 2
        
      - name: default_path
        condition: "true"
        priority: 99
```

### 3.3 并行处理

```yaml
- id: parallel_processor
  type: parallel
  input: trigger.output
  config:
    branches:
      - id: sentiment_analysis
        type: agent
        agent_id: sentiment-agent
        
      - id: keyword_extraction
        type: processor
        config:
          operation: extract_keywords
          
      - id: entity_recognition
        type: processor
        config:
          operation: extract_entities
          
  merge_strategy: all_completed
  timeout: 60000
```

## 4. 错误处理

### 4.1 错误类型

```
Pipeline错误分类：
├── 节点执行错误
│   ├── Agent超时
│   ├── 模型调用失败
│   └── 数据转换异常
├── 业务逻辑错误
│   ├── 条件不满足
│   └── 验证失败
└── 系统错误
    ├── 资源不足
    ├── 网络异常
    └── 服务不可用
```

### 4.2 错误处理配置

```yaml
pipeline:
  error_handling:
    # 全局错误策略
    strategy: continue_on_error
    
    # 重试配置
    retry:
      max_attempts: 3
      backoff: exponential
      initial_delay: 1000
      multiplier: 2
      
    # 错误分支
    on_error:
      - error_handler_node
      
    # 超时配置
    timeout: 300000  # 5分钟
```

### 4.3 错误处理节点

```yaml
# 错误处理器
- id: error_handler
  type: error_handler
  input: failed_node.error
  config:
    error_types:
      - validation_error
      - timeout_error
    actions:
      - log_error
      - notify_admin
      - fallback_response
      
# 回退节点
- id: fallback_response
  type: response
  config:
    status: 200
    body:
      success: false
      message: "服务暂时不可用，请稍后重试"
```

### 4.4 完整示例

```yaml
pipeline:
  name: robust-pipeline
  version: "1.0.0"
  
  nodes:
    # 触发器
    - id: trigger
      type: webhook
      config:
        method: POST
        
    # 主处理路径
    - id: main_processor
      type: agent
      input: trigger.output
      agent_id: main-agent
      error_output: error_handler
      
    # 错误处理
    - id: error_handler
      type: error_handler
      config:
        actions:
          - log
          - alert
          
    # 最终输出
    - id: response
      type: response
      input: 
        on_success: main_processor.output
        on_error: error_handler.output
        
  error_handling:
    retry:
      max_attempts: 3
      backoff: linear
      delay: 2000
```

## 实践任务

1. **任务1**: 创建一个同步Pipeline，包含至少4个节点
2. **任务2**: 实现条件分支，根据输入类型选择不同处理路径
3. **任务3**: 配置完整的错误处理和重试机制

## 总结

本课程学习了：
- ✅ Pipeline核心概念和类型
- ✅ 节点类型和连接方式
- ✅ 条件分支和路由逻辑
- ✅ 错误处理和重试机制

---
*课程时长：1小时*
