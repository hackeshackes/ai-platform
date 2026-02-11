# Agent Collaboration Network v1.0

AI Platform v8 - Agent协作网络模块

## 功能概述

提供多Agent协同工作能力，支持：
- Agent间通信和消息传递
- 任务分解和调度
- 工作流编排和执行
- 共识决策机制
- 进度追踪和监控

## 协作模式

```python
from backend.agents.collaboration import CollaborationMode

# 顺序执行 - Agent按顺序依次执行任务
SEQUENTIAL = "sequential"

# 并行执行 - Agent同时执行独立任务
PARALLEL = "parallel"

# 层级协作 - 监督者协调工作者
HIERARCHICAL = "hierarchical"

# 共识决策 - 通过投票达成共识
CONSENSUS = "consensus"
```

## 快速开始

### 1. 初始化编排器

```python
import asyncio
from backend.agents.collaboration import get_orchestrator

async def main():
    orchestrator = get_orchestrator()
    await orchestrator.initialize()
```

### 2. 创建协作会话

```python
session = await orchestrator.create_collaboration_session(
    name="research_project",
    description="Research team collaboration",
    mode="hierarchical",
    agent_ids=["researcher", "analyst", "writer"]
)
session_id = session["session_id"]
```

### 3. Agent加入会话

```python
await orchestrator.join_session(
    session_id=session_id,
    agent_id="researcher",
    role="worker",
    metadata={"capabilities": ["web_search", "data_analysis"]}
)
```

### 4. 分配任务

```python
await orchestrator.assign_task(
    session_id=session_id,
    task_name="收集数据",
    description="从多个来源收集市场数据",
    assigned_agent="researcher",
    priority=1
)
```

### 5. 执行协作

```python
result = await orchestrator.execute_collaboration(session_id)
```

## API端点

### REST API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/agents/collaboration/session` | POST | 创建协作会话 |
| `/api/v1/agents/collaboration/sessions` | GET | 列出所有会话 |
| `/api/v1/agents/collaboration/session/{id}` | GET | 获取会话详情 |
| `/api/v1/agents/collaboration/session/{id}/join` | POST | Agent加入会话 |
| `/api/v1/agents/collaboration/session/{id}/task` | POST | 分配任务 |
| `/api/v1/agents/collaboration/session/{id}/execute` | POST | 执行协作 |
| `/api/v1/agents/collaboration/session/{id}/result` | GET | 获取结果 |
| `/api/v1/agents/collaboration/session/{id}/progress` | GET | 获取进度 |

### 使用示例

```bash
# 创建协作会话
curl -X POST http://localhost:8000/api/v1/agents/collaboration/session \
  -H "Content-Type: application/json" \
  -d '{"name":"team-project","agents":["researcher","analyst"],"mode":"hierarchical"}'

# Agent加入
curl -X POST http://localhost:8000/api/v1/agents/collaboration/session/{id}/join \
  -d '{"agent_id":"researcher","role":"worker"}'

# 执行协作
curl -X POST http://localhost:8000/api/v1/agents/collaboration/session/{id}/execute

# 获取结果
curl http://localhost:8000/api/v1/agents/collaboration/session/{id}/result
```

## 模块结构

```
backend/agents/collaboration/
├── __init__.py           # 模块导出
├── models.py             # 数据模型定义
├── communication.py      # Agent间通信
├── workflow.py           # 工作流引擎
├── task_decomposer.py    # 任务分解器
├── consensus.py          # 共识机制
└── orchestrator.py       # 协作编排器
```

## 核心组件

### 1. CommunicationManager
管理Agent间的消息传递和事件通知。

### 2. WorkflowEngine
执行和管理工作流，支持顺序、并行和层级执行模式。

### 3. TaskDecomposer
将复杂任务分解为可执行的子任务。

### 4. ConsensusManager
处理多Agent间的共识决策。

### 5. CollaborationOrchestrator
协调整个协作过程，管理会话、任务和Agent。

## 测试

```bash
# 运行协作测试
PYTHONPATH=/Users/yubao/.openclaw/workspace python3 backend/agents/collaboration/test_collaboration.py
```

## 工作流定义示例

```yaml
workflow:
  name: research_report
  mode: hierarchical
  agents:
    - id: researcher
      role: info_gatherer
    - id: analyst
      role: pattern_recognizer
    - id: writer
      role: report_generator
  tasks:
    - name: 收集数据
      assigned_agent: researcher
    - name: 分析数据
      assigned_agent: analyst
      dependencies: ["收集数据"]
    - name: 生成报告
      assigned_agent: writer
      dependencies: ["分析数据"]
```
