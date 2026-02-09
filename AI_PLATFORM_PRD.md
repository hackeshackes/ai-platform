# AI Platform 功能需求文档

**版本：** V1.1  
**日期：** 2026-02-08  
**项目：** 大模型全生命周期管理平台

---

## 一、功能模块总览

| 序号 | 模块 | 功能描述 | 优先级 |
|------|------|----------|--------|
| 1 | Dashboard | 仪表盘、统计概览 | P0 |
| 2 | Projects | 项目管理 | P0 |
| 3 | Experiments | 实验管理 | P0 |
| 4 | Tasks | 任务调度 | P0 |
| 5 | Training | 创建训练 | P0 |
| 6 | Inference | 推理服务 | P0 |
| 7 | Datasets | 数据集管理 | P0 |
| 8 | Models | 模型管理 | P0 |
| 9 | Settings | 系统设置 | P1 |

---

## 二、详细功能需求

### 2.1 Dashboard（仪表盘）

**页面路径：** `/dashboard`

**功能清单：**
- [ ] **统计卡片（4个）**
  - 项目总数
  - 实验总数
  - 训练任务数
  - 部署服务数
  
- [ ] **最近任务列表**
  - 显示最近5个任务
  - 任务名称、状态、时间
  
- [ ] **系统状态**
  - GPU 使用率
  - 存储空间
  - 运行中服务数

**UI要求：**
- 4个卡片并排显示
- 卡片带数值和趋势箭头
- 最近任务使用表格展示

---

### 2.2 Projects（项目管理）

**页面路径：** `/projects`

**功能清单：**
- [ ] **项目列表表格**
  - 项目名称
  - 描述
  - 状态（活跃/归档）
  - 实验数
  - 创建时间
  - 操作（编辑、删除）
  
- [ ] **创建项目**
  - 项目名称（必填）
  - 描述（可选）
  - 默认状态：活跃

- [ ] **筛选功能**
  - 按状态筛选（全部/活跃/归档）
  - 搜索项目名称

---

### 2.3 Experiments（实验管理）

**页面路径：** `/experiments`

**功能清单：**
- [ ] **实验列表表格**
  - 实验名称
  - 基础模型
  - 任务类型（微调/训练/蒸馏）
  - 状态（运行中/已完成/等待中）
  - Loss 值
  - 创建时间
  - 操作（查看详情、停止）
  
- [ ] **创建实验**
  - 实验名称
  - 选择基础模型
  - 选择任务类型
  
- [ ] **实验详情**
  - 训练进度
  - Loss 曲线图
  - 训练日志

---

### 2.4 Tasks（任务调度）

**页面路径：** `/tasks`

**功能清单：**
- [ ] **任务列表**
  - 任务名称
  - 类型（训练/推理/数据处理）
  - GPU 使用
  - 进度（0-100%）
  - 状态（运行中/已完成/失败）
  - 开始时间
  - 操作（启动/停止/查看日志）
  
- [ ] **任务操作**
  - 启动任务
  - 停止任务
  - 查看日志
  - 删除任务

---

### 2.5 Training（创建训练）

**页面路径：** `/training`

**功能清单（3步向导）：**

**Step 1: 基础配置**
- [ ] 任务名称
- [ ] 选择项目
- [ ] 任务描述

**Step 2: 模型选择**
- [ ] 基础模型（Llama-2/Qwen/Baichuan等）
- [ ] 训练方法（Full/Lora/QLoRA）
- [ ] 选择数据集

**Step 3: 参数配置**
- [ ] 学习率（默认1e-4）
- [ ] 训练轮数 epochs（默认3）
- [ ] 批次大小 batch size（默认4）
- [ ] Warmup Steps
- [ ] GPU数量
- [ ] 训练策略（None/DeepSpeed/FSDP）

**提交：**
- [ ] 创建训练任务
- [ ] 跳转到任务列表

---

### 2.6 Inference（推理服务）

**页面路径：** `/inference`

**功能清单：**
- [ ] **部署服务表单**
  - 服务名称
  - 选择模型
  - 副本数（1-10）
  
- [ ] **服务列表**
  - 服务名称
  - 模型版本
  - 状态（运行中/已停止）
  - 副本数
  - 操作（启动/停止/删除）

- [ ] **API测试**
  - 输入文本框
  - 生成按钮
  - 输出结果展示

---

### 2.7 Datasets（数据集管理）

**页面路径：** `/datasets`

**功能清单：**
- [ ] **数据集列表**
  - 数据集名称
  - 类型
  - 格式
  - 大小
  - 行数
  - 标注状态
  - 操作（查看/删除）
  
- [ ] **创建数据集**
  - 数据集名称
  - 类型选择
  - 上传文件（CSV/JSON）
  
- [ ] **筛选和搜索**
  - 按标注状态筛选
  - 搜索数据集名称

---

### 2.8 Models（模型管理）

**页面路径：** `/models`

**功能清单：**
- [ ] **模型列表**
  - 模型名称
  - 基础模型
  - 参数量
  - 量化级别
  - 阶段（生产/预发布）
  - 操作（部署/下载/删除）
  
- [ ] **注册模型**
  - 模型名称
  - 选择基础模型
  - 参数量
  - 量化选择（FP16/INT8/INT4）
  - 模型文件上传

---

### 2.9 Settings（系统设置）

**页面路径：** `/settings`

**功能清单：**

**General（通用设置）**
- [ ] 站点名称
- [ ] 默认语言

**Notifications（通知设置）**
- [ ] 任务完成通知
- [ ] 失败告警

**Security（安全设置）**
- [ ] API密钥管理
- [ ] 访问控制

---

## 三、数据结构

### 3.1 Projects 表
```sql
CREATE TABLE projects (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  status TEXT DEFAULT 'active',
  experiment_count INTEGER DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

### 3.2 Experiments 表
```sql
CREATE TABLE experiments (
  id INTEGER PRIMARY KEY,
  project_id INTEGER,
  name TEXT NOT NULL,
  base_model TEXT,
  task_type TEXT,
  status TEXT,
  loss_value REAL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (project_id) REFERENCES projects(id)
)
```

### 3.3 Tasks 表
```sql
CREATE TABLE tasks (
  id INTEGER PRIMARY KEY,
  experiment_id INTEGER,
  name TEXT NOT NULL,
  type TEXT,
  gpu INTEGER,
  progress INTEGER DEFAULT 0,
  status TEXT,
  started_at DATETIME,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id)
)
```

### 3.4 Datasets 表
```sql
CREATE TABLE datasets (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT,
  format TEXT,
  size TEXT,
  rows INTEGER,
  annotation_status TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

### 3.5 Models 表
```sql
CREATE TABLE models (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  base_model TEXT,
  params TEXT,
  quantization TEXT,
  stage TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

---

## 四、API 接口

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | /api/v1/projects | 获取项目列表 |
| POST | /api/v1/projects | 创建项目 |
| GET | /api/v1/experiments | 获取实验列表 |
| POST | /api/v1/experiments | 创建实验 |
| GET | /api/v1/tasks | 获取任务列表 |
| POST | /api/v1/tasks | 创建任务 |
| GET | /api/v1/datasets | 获取数据集列表 |
| POST | /api/v1/datasets | 创建数据集 |
| GET | /api/v1/models | 获取模型列表 |
| POST | /api/v1/models | 注册模型 |

---

## 五、设计规范

### 5.1 配色方案
- 主色：#0066CC（华润蓝）
- 辅助色：#F5A623（华润橙）
- 背景色：#F5F5F5
- 文字色：#333333

### 5.2 组件库
- 使用 Ant Design 5.x
- 表格、表单、卡片、步骤条

### 5.3 响应式
- 支持 375px - 1920px

---

**文档结束**
