# 快速入门 - 5分钟

欢迎来到AI Platform学习中心！本课程将在5分钟内带你了解平台核心概念，并完成第一个Agent和Pipeline的创建。

## 1. 什么是AI Platform

AI Platform是一个企业级AI应用开发平台，提供完整的Agent创建、Pipeline编排和部署能力。

### 核心特性
- **可视化开发**: 拖拽式构建复杂AI工作流
- **多模型支持**: 集成主流大语言模型
- **企业级特性**: 权限管理、审计日志、高可用部署

### 架构概览
```
┌─────────────────────────────────────────┐
│           AI Platform                    │
├─────────────┬─────────────┬─────────────┤
│   Agent     │  Pipeline   │  Deployment │
│   Studio    │  Builder    │   Manager   │
├─────────────┴─────────────┴─────────────┤
│         模型层 (LLM, Embedding, etc.)    │
├─────────────────────────────────────────┤
│         基础设施层                       │
└─────────────────────────────────────────┘
```

## 2. 创建第一个Agent

### 步骤1：访问Agent Studio
1. 登录AI Platform控制台
2. 点击左侧菜单「Agent Studio」
3. 点击「创建Agent」按钮

### 步骤2：配置Agent基本信息
```
Agent名称：my-first-agent
Agent描述：这是一个演示Agent
Agent类型：对话型
```

### 步骤3：配置技能
选择以下内置技能：
- 文本生成
- 问答能力
- 代码解释

### 步骤4：保存并测试
点击「保存」后，在右侧测试面板输入：
```
你好，请介绍一下你自己
```

## 3. 运行第一个Pipeline

### Pipeline概念
Pipeline是由多个节点组成的自动化工作流，用于编排复杂的AI任务。

### 创建步骤

1. **创建Pipeline**
   - 进入「Pipeline Builder」
   - 点击「新建Pipeline」

2. **添加节点**
   ```
   节点1: 触发器 (Webhook)
   节点2: Agent处理 (my-first-agent)
   节点3: 结果输出 (HTTP响应)
   ```

3. **连接节点**
   - 拖拽节点边缘建立连接
   - 节点1 → 节点2 → 节点3

4. **部署运行**
   - 点击「部署」
   - 触发测试请求

### 测试代码
```bash
curl -X POST https://api.aip.com/pipeline/my-first-pipeline \
  -H "Content-Type: application/json" \
  -d '{"input": "你好，AI Platform！"}'
```

## 恭喜完成！

你已经掌握了：
- ✅ AI Platform核心概念
- ✅ 创建和配置Agent
- ✅ 构建和运行Pipeline

### 下一步
继续学习「Agent创建」课程，深入了解Agent的高级配置。

---
*课程时长：约5分钟*
