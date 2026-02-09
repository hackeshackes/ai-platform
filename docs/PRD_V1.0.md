# AI Platform V1.0 产品需求文档 (PRD)

**版本：** V1.0  
**日期：** 2026-02-08  
**项目：** AI Platform - 大模型全生命周期管理平台  
**规划周期：** 短期（1-2个月）

---

## 一、项目概述

### 1.1 产品愿景

AI Platform 是一个面向开发者和企业的**大模型全生命周期管理平台**，提供从数据准备、模型训练、实验管理到推理部署的一站式解决方案。

### 1.2 核心价值

- **简化工作流**：从数据到部署的端到端流程
- **降低门槛**：可视化界面，无需命令行
- **成本优化**：高效微调和量化技术
- **实时监控**：训练过程可视化

### 1.3 当前状态

| 模块 | 完成度 | 说明 |
|------|--------|------|
| 前端UI | 90% | 9个页面框架完成，中文化基本就绪 |
| 后端API | 85% | 11个API模块就绪 |
| 数据层 | 0% | 使用内存存储，需要持久化 |
| 认证 | 50% | 基础Token认证，需要完善 |

---

## 二、功能需求

### 2.1 用户故事

#### US-001: 数据打通
**作为** 开发者  
**我希望** 前端能够调用真实后端API  
**以便于** 看到真实数据，而不只是模拟数据

**验收标准：**
- [ ] 前端所有列表页面使用真实API数据
- [ ] API响应时间 < 500ms
- [ ] 错误时有友好的用户提示

#### US-002: 数据持久化
**作为** 管理员  
**我希望** 数据能够永久存储  
**以便于** 重启服务后数据不丢失

**验收标准：**
- [ ] 使用SQLite数据库存储
- [ ] 每次CRUD操作后数据正确保存
- [ ] 支持至少1000条记录

#### US-003: 项目管理
**作为** 用户  
**我希望** 创建和管理项目  
**以便于** 组织我的AI实验

**验收标准：**
- [ ] 创建项目（名称、描述）
- [ ] 查看项目列表
- [ ] 编辑/删除项目
- [ ] 按状态筛选（全部/活跃/归档）
- [ ] 搜索项目名称

#### US-004: 实验管理
**作为** 开发者  
**我希望** 跟踪我的模型实验  
**以便于** 对比不同实验结果

**验收标准：**
- [ ] 创建实验（选择项目、基础模型）
- [ ] 查看实验列表
- [ ] 查看实验详情（Loss曲线）
- [ ] 停止运行中的实验
- [ ] 按模型和类型筛选

#### US-005: 任务调度
**作为** 开发者  
**我希望** 查看和管理训练任务  
**以便于** 监控训练进度

**验收标准：**
- [ ] 查看任务列表（类型、进度、GPU）
- [ ] 启动/停止任务
- [ ] 查看训练日志
- [ ] 任务状态实时更新
- [ ] 按状态筛选

#### US-006: 训练任务创建
**作为** 开发者  
**我希望** 创建训练任务  
**以便于** 开始模型微调

**验收标准：**
- [ ] 3步向导（基础配置、模型选择、参数配置）
- [ ] 支持选择项目、模型、数据集
- [ ] 配置学习率、Epochs、Batch Size
- [ ] 提交后跳转到任务列表

#### US-007: 推理服务部署
**作为** 开发者  
**我希望** 部署模型为推理服务  
**以便于** 提供API调用

**验收标准：**
- [ ] 选择模型部署为服务
- [ ] 配置副本数
- [ ] 查看服务列表
- [ ] 启动/停止服务
- [ ] API测试工具

#### US-008: 数据集管理
**作为** 数据工程师  
**我希望** 管理数据集  
**以便于** 准备训练数据

**验收标准：**
- [ ] 查看数据集列表（类型、格式、大小）
- [ ] 上传数据集文件
- [ ] 按标注状态筛选
- [ ] 查看数据集详情

#### US-009: 模型管理
**作为** ML工程师  
**我希望** 注册和管理模型  
**以便于** 版本控制我的模型

**验收标准：**
- [ ] 注册新模型
- [ ] 查看模型列表（参数量、量化级别）
- [ ] 部署模型到推理服务
- [ ] 下载模型文件
- [ ] 按阶段筛选（生产/预发布）

#### US-010: Loss曲线可视化
**作为** 开发者  
**我希望** 查看实验的Loss曲线  
**以便于** 监控训练过程

**验收标准：**
- [ ] 在实验详情页面显示Loss曲线
- [ ] 支持缩放和拖拽
- [ ] 显示step和loss数值
- [ ] 平滑曲线显示

#### US-011: GPU实时监控
**作为** 运维工程师  
**我希望** 监控GPU使用情况  
**以便于** 了解资源使用状态

**验收标准：**
- [ ] 显示GPU使用率
- [ ] 显示显存占用
- [ ] 实时数据更新（每5秒）
- [ ] GPU不足告警

#### US-012: 训练日志查看
**作为** 开发者  
**我希望** 查看训练日志  
**以便于** 调试训练问题

**验收标准：**
- [ ] 实时日志流
- [ ] ERROR级别高亮
- [ ] 日志搜索
- [ ] 分页查看

#### US-013: 系统设置
**作为** 管理员  
**我希望** 配置系统参数  
**以便于** 定制平台行为

**验收标准：**
- [ ] 通用设置（站点名称、时区）
- [ ] 通知设置（任务完成/失败/GPU警告）
- [ ] 安全设置（修改密码、API密钥）
- [ ] 语言切换（中/英）

---

## 三、功能模块

### 3.1 Dashboard（仪表盘）

**页面路径：** `/dashboard`

**功能清单：**
- [x] **统计卡片（4个）**
  - [x] 项目总数
  - [x] 实验总数
  - [x] 训练任务数
  - [x] 部署服务数
  
- [x] **最近任务列表**
  - [x] 显示最近5个任务
  - [x] 任务名称、状态、时间
  
- [ ] **系统状态**
  - [ ] GPU 使用率（实时）
  - [ ] 存储空间
  - [ ] 运行中服务数

**UI要求：**
- 4个卡片并排显示
- 卡片带数值和趋势箭头
- 最近任务使用表格展示

### 3.2 Projects（项目管理）

**页面路径：** `/projects`

**数据模型：**
```python
class Project(Base):
    id: int
    name: str          # 项目名称
    description: str   # 描述
    status: str        # active/archived
    user_id: int       # 创建者
    created_at: datetime
```

**功能清单：**
- [x] **项目列表表格**
  - [x] 项目名称
  - [x] 描述
  - [x] 状态（活跃/归档）
  - [x] 实验数
  - [x] 创建时间
  - [x] 操作（编辑、删除）
  
- [x] **创建项目**
  - [x] 项目名称（必填）
  - [x] 描述（可选）
  - [x] 默认状态：活跃

- [x] **筛选功能**
  - [x] 按状态筛选（全部/活跃/归档）
  - [x] 搜索项目名称

### 3.3 Experiments（实验管理）

**页面路径：** `/experiments`

**数据模型：**
```python
class Experiment(Base):
    id: int
    project_id: int     # 关联项目
    name: str          # 实验名称
    base_model: str     # 基础模型
    task_type: str     # fine_tuning/training/distillation
    status: str        # pending/running/completed/failed
    loss_value: float  # 最新Loss值
    created_at: datetime
```

**功能清单：**
- [x] **实验列表表格**
  - [x] 实验名称
  - [x] 基础模型
  - [x] 任务类型（微调/训练/蒸馏）
  - [x] 状态（运行中/已完成/等待中）
  - [x] Loss 值
  - [x] 创建时间
  - [x] 操作（查看详情、停止）
  
- [x] **筛选功能**
  - [x] 按基础模型筛选
  - [x] 按任务类型筛选

- [ ] **实验详情**
  - [x] 训练进度
  - [x] **Loss 曲线图**（新）
  - [ ] 训练日志（新）

### 3.4 Tasks（任务调度）

**页面路径：** `/tasks`

**数据模型：**
```python
class Task(Base):
    id: int
    experiment_id: int  # 关联实验
    name: str          # 任务名称
    type: str          # training/inference/data/optimization
    gpu: int           # GPU数量
    progress: int       # 进度 0-100
    status: str        # pending/running/completed/failed
    logs: str          # 训练日志
    started_at: datetime
```

**功能清单：**
- [x] **任务列表**
  - [x] 任务名称
  - [x] 类型（训练/推理/数据处理）
  - [x] GPU 使用
  - [x] 进度（0-100%）
  - [x] 状态（运行中/已完成/失败）
  - [x] 开始时间
  - [x] 操作（启动、停止、查看日志、删除）

- [ ] **实时更新**
  - [x] 状态实时更新（每3秒轮询）
  - [x] 进度实时更新

- [ ] **日志查看**
  - [x] **实时日志流**（新）
  - [x] **ERROR高亮**（新）
  - [x] **日志搜索**（新）

### 3.5 Training（创建训练）

**页面路径：** `/training`

**功能清单：**
- [x] **Step 1: 基础配置**
  - [x] 任务名称
  - [x] 选择项目
  - [x] 描述
  
- [x] **Step 2: 模型选择**
  - [x] 基础模型（Llama-2/Qwen/Baichuan等）
  - [x] 训练方法（Full/Lora/QLoRA）
  - [x] 选择数据集
  
- [x] **Step 3: 参数配置**
  - [x] 学习率（默认1e-4）
  - [x] 训练轮数 epochs（默认3）
  - [x] 批次大小 batch size（默认4）
  - [x] Warmup Steps
  - [x] GPU数量
  - [x] 训练策略（None/DeepSpeed/FSDP）

- [x] **提交**
  - [x] 创建训练任务
  - [x] 跳转到任务列表

### 3.6 Inference（推理服务）

**页面路径：** `/inference`

**数据模型：**
```python
class InferenceService(Base):
    id: int
    name: str          # 服务名称
    model_id: int      # 关联模型
    replicas: int      # 副本数
    status: str        # running/stopped
    endpoint: str       # API端点
    created_at: datetime
```

**功能清单：**
- [x] **部署服务表单**
  - [x] 服务名称
  - [x] 选择模型
  - [x] 副本数（1-10）
  
- [x] **服务列表**
  - [x] 服务名称
  - [x] 模型版本
  - [x] 状态（运行中/已停止）
  - [x] 副本数
  - [x] 操作（启动、停止、删除）

- [x] **API测试**
  - [x] 输入文本框
  - [x] 生成按钮
  - [x] 输出结果展示

### 3.7 Datasets（数据集管理）

**页面路径：** `/datasets`

**数据模型：**
```python
class Dataset(Base):
    id: int
    name: str          # 数据集名称
    type: str          # qa/dialog/instructions/general
    format: str         # json/csv/jsonl
    size: str           # 文件大小
    rows: int          # 行数
    annotation_status: str  # annotated/annotating/notAnnotated
    file_path: str      # 文件路径
    created_at: datetime
```

**功能清单：**
- [x] **数据集列表**
  - [x] 数据集名称
  - [x] 类型
  - [x] 格式
  - [x] 大小
  - [x] 行数
  - [x] 标注状态
  - [x] 操作（查看、删除）
  
- [x] **创建数据集**
  - [x] 数据集名称
  - [x] 类型选择
  - [x] 格式选择
  - [x] 上传文件（CSV/JSON/JSONL）
  
- [x] **筛选和搜索**
  - [x] 按标注状态筛选
  - [x] 搜索数据集名称

### 3.8 Models（模型管理）

**页面路径：** `/models`

**数据模型：**
```python
class Model(Base):
    id: int
    name: str          # 模型名称
    base_model: str     # 基础模型
    params: str         # 参数量
    quantization: str   # fp16/int8/int4
    stage: str         # production/staging
    file_path: str      # 模型文件路径
    created_at: datetime
```

**功能清单：**
- [x] **模型列表**
  - [x] 模型名称
  - [x] 基础模型
  - [x] 参数量
  - [x] 量化级别
  - [x] 阶段（生产/预发布）
  - [x] 操作（部署、下载、详情）
  
- [x] **注册模型**
  - [x] 模型名称
  - [x] 选择基础模型
  - [x] 参数量
  - [x] 量化选择（FP16/INT8/INT4）
  - [x] 模型文件上传

### 3.9 Settings（系统设置）

**页面路径：** `/settings`

**功能清单：**

- [x] **General（通用设置）**
  - [x] 站点名称
  - [x] 默认语言
  - [x] 时区选择
  
- [x] **Notifications（通知设置）**
  - [x] 任务完成通知
  - [x] 失败告警
  - [x] GPU不足警告
  - [x] 存储空间不足
  - [x] 邮件通知开关
  
- [x] **Security（安全设置）**
  - [x] 修改密码
  - [x] API密钥管理
  - [x] 会话超时设置

---

## 四、数据流设计

### 4.1 API接口

#### 4.1.1 认证模块
| 方法 | 路径 | 功能 |
|------|------|------|
| POST | /api/v1/auth/token | 获取Token |
| POST | /api/v1/auth/refresh | 刷新Token |
| POST | /api/v1/auth/logout | 登出 |

#### 4.1.2 项目管理
| 方法 | 路径 | 功能 |
|------|------|------|
| GET | /api/v1/projects | 获取项目列表 |
| GET | /api/v1/projects/{id} | 获取项目详情 |
| POST | /api/v1/projects | 创建项目 |
| PUT | /api/v1/projects/{id} | 更新项目 |
| DELETE | /api/v1/projects/{id} | 删除项目 |

#### 4.1.3 实验管理
| 方法 | 路径 | 功能 |
|------|------|------|
| GET | /api/v1/experiments | 获取实验列表 |
| GET | /api/v1/experiments/{id} | 获取实验详情 |
| GET | /api/v1/experiments/{id}/loss | 获取Loss数据 |
| POST | /api/v1/experiments | 创建实验 |
| PUT | /api/v1/experiments/{id} | 更新实验 |
| DELETE | /api/v1/experiments/{id} | 删除实验 |
| POST | /api/v1/experiments/{id}/stop | 停止实验 |

#### 4.1.4 任务调度
| 方法 | 路径 | 功能 |
|------|------|------|
| GET | /api/v1/tasks | 获取任务列表 |
| GET | /api/v1/tasks/{id} | 获取任务详情 |
| GET | /api/v1/tasks/{id}/logs | 获取任务日志 |
| POST | /api/v1/tasks | 创建任务 |
| POST | /api/v1/tasks/{id}/start | 启动任务 |
| POST | /api/v1/tasks/{id}/stop | 停止任务 |
| DELETE | /api/v1/tasks/{id} | 删除任务 |

#### 4.1.5 数据集管理
| 方法 | 路径 | 功能 |
|------|------|------|
| GET | /api/v1/datasets | 获取数据集列表 |
| GET | /api/v1/datasets/{id} | 获取数据集详情 |
| POST | /api/v1/datasets | 创建数据集 |
| POST | /api/v1/datasets/upload | 上传数据集文件 |
| DELETE | /api/v1/datasets/{id} | 删除数据集 |

#### 4.1.6 模型管理
| 方法 | 路径 | 功能 |
|------|------|------|
| GET | /api/v1/models | 获取模型列表 |
| GET | /api/v1/models/{id} | 获取模型详情 |
| POST | /api/v1/models | 注册模型 |
| POST | /api/v1/models/{id}/deploy | 部署模型 |
| POST | /api/v1/models/{id}/download | 下载模型 |
| DELETE | /api/v1/models/{id} | 删除模型 |

#### 4.1.7 推理服务
| 方法 | 路径 | 功能 |
|------|------|------|
| GET | /api/v1/inference/services | 获取服务列表 |
| GET | /api/v1/inference/services/{id} | 获取服务详情 |
| POST | /api/v1/inference/services | 创建服务 |
| POST | /api/v1/inference/services/{id}/start | 启动服务 |
| POST | /api/v1/inference/services/{id}/stop | 停止服务 |
| DELETE | /api/v1/inference/services/{id} | 删除服务 |
| POST | /api/v1/inference/services/{id}/test | API测试 |

#### 4.1.8 系统监控
| 方法 | 路径 | 功能 |
|------|------|------|
| GET | /api/v1/monitoring/gpu | GPU状态 |
| GET | /api/v1/monitoring/storage | 存储状态 |

### 4.2 数据库模型

```python
# backend/database/models.py

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True)
    hashed_password = Column(String(256), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Project(Base):
    __tablename__ = 'projects'
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    status = Column(String(20), default='active')
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)

class Experiment(Base):
    __tablename__ = 'experiments'
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    project_id = Column(Integer, ForeignKey('projects.id'))
    base_model = Column(String(100))
    task_type = Column(String(50))
    status = Column(String(20), default='pending')
    loss_value = Column(Float)
    loss_history = Column(JSON)  # [{step, loss, epoch}, ...]
    created_at = Column(DateTime, default=datetime.utcnow)

class Task(Base):
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    type = Column(String(50))
    gpu = Column(Integer, default=0)
    progress = Column(Integer, default=0)
    status = Column(String(20), default='pending')
    logs = Column(Text)
    started_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    type = Column(String(50))
    format = Column(String(20))
    size = Column(String(20))
    rows = Column(Integer)
    annotation_status = Column(String(20))
    file_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)

class Model(Base):
    __tablename__ = 'models'
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    base_model = Column(String(100))
    params = Column(String(50))
    quantization = Column(String(20))
    stage = Column(String(20), default='staging')
    file_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)

class InferenceService(Base):
    __tablename__ = 'inference_services'
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    model_id = Column(Integer, ForeignKey('models.id'))
    replicas = Column(Integer, default=1)
    status = Column(String(20), default='stopped')
    endpoint = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
```

---

## 五、技术架构

### 5.1 前端技术栈

| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 框架 | React | 18.x | UI组件库 |
| 构建 | Vite | 5.x | 开发构建 |
| 路由 | React Router | 6.x | SPA路由 |
| UI库 | Ant Design | 5.x | 组件库 |
| 图表 | Recharts/ECharts | 最新 | Loss曲线 |
| HTTP | Axios | 最新 | API调用 |
| 状态 | React Context | 内置 | 状态管理 |
| 国际化 | 自定义Hook | - | 中英切换 |
| 类型 | TypeScript | 5.x | 类型安全 |

### 5.2 后端技术栈

| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 框架 | FastAPI | 0.11.x | API框架 |
| 数据库 | SQLite | 最新 | 数据持久化 |
| ORM | SQLAlchemy | 2.x | ORM映射 |
| 认证 | Python-Jose | 最新 | JWT认证 |
| 验证 | Pydantic | 2.x | 数据验证 |
| 日志 | Python logging | 内置 | 日志记录 |
| CORS | FastAPI CORS | 最新 | 跨域支持 |

### 5.3 部署架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户浏览器                        │
└─────────────────────┬───────────────────────────┘
                      │ HTTPS
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   Nginx (反向代理)                     │
│                  SSL证书 termination                  │
└─────────────────────┬───────────────────────────┘
                      │
          ┌──────────┴──────────┐
          ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│   React Front   │  │  FastAPI Backend │
│   (Vite)       │  │   (Python)      │
│   :3000        │  │   :8000         │
└─────────────────┘  └────────┬────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │    SQLite       │
                      │   (cms.db)      │
                      └─────────────────┘
```

---

## 六、用户体验

### 6.1 加载状态

**全局加载状态：**
- 页面切换时显示Spin加载动画
- API请求时显示加载指示器
- 大数据量时显示骨架屏

**组件加载：**
```tsx
import { Spin, Skeleton } from 'antd'

// API加载
{loading ? <Spin tip="加载中..." /> : <Content />}

// 骨架屏
<Skeleton active paragraph={{ rows: 4 }} />
```

### 6.2 错误处理

**错误边界：**
```tsx
<ErrorBoundary fallback={<ErrorPage />}>
  <App />
</ErrorBoundary>
```

**Toast提示：**
```tsx
import { message } from 'antd'

// 成功
message.success('操作成功')

// 错误
message.error(error.message)
```

### 6.3 响应式断点

| 断点 | 宽度 | 布局 |
|------|------|------|
| xs | <576px | 单列 |
| sm | ≥576px | 单列 |
| md | ≥768px | 双列 |
| lg | ≥992px | 三列 |
| xl | ≥1200px | 四列 |

---

## 七、测试需求

### 7.1 测试覆盖

| 类别 | 覆盖率 | 说明 |
|------|--------|------|
| 单元测试 | >60% | 核心逻辑 |
| API测试 | 100% | 所有端点 |
| E2E测试 | 核心流程 | 用户流程 |

### 7.2 测试用例

```python
# tests/test_projects.py
def test_create_project():
    # 创建项目
    response = client.post('/api/v1/projects', json={
        'name': 'Test Project',
        'description': 'Test Description'
    })
    assert response.status_code == 200
    assert response.json()['name'] == 'Test Project'

def test_list_projects():
    # 列表查询
    response = client.get('/api/v1/projects')
    assert response.status_code == 200
    assert len(response.json()) >= 0
```

### 7.3 E2E测试

```typescript
// tests/e2e.spec.ts
import { test, expect } from '@playwright/test'

test('create project workflow', async ({ page }) => {
  await page.goto('/projects')
  await page.click('text=创建项目')
  await page.fill('input[name="name"]', 'E2E Test')
  await page.click('button:has-text("确定")')
  await expect(page.locator('text=E2E Test')).toBeVisible()
})
```

---

## 八、性能要求

### 8.1 性能指标

| 指标 | 要求 | 说明 |
|------|------|------|
| 首屏加载 | <3s | FCP |
| API响应 | <500ms | 核心接口 |
| 列表渲染 | <1s | 100条数据 |
| 图表渲染 | <2s | Loss曲线 |
| 并发用户 | >50 | 同时在线 |

### 8.2 优化策略

- 代码分割（React.lazy）
- 路由懒加载
- API请求缓存
- 数据库索引
- 连接池配置

---

## 九、安全要求

### 9.1 认证授权

- JWT Token认证
- Token刷新机制
- 密码加密存储
- API限流保护

### 9.2 数据安全

- HTTPS加密传输
- SQL注入防护
- XSS攻击防护
- CORS跨域控制

---

## 十、实施计划

### 10.1 时间线

| 周次 | 里程碑 | 交付物 |
|------|--------|--------|
| Week 1 | 数据库设计 | SQLite数据库，ORM模型 |
| Week 1-2 | API打通 | 前端API集成，认证系统 |
| Week 2-3 | 核心CRUD | Projects/Experiments/Tasks API |
| Week 3-4 | 核心功能 | Loss曲线，GPU监控，日志 |
| Week 4-5 | 用户体验 | 加载状态，错误处理 |
| Week 5-6 | 测试完善 | 单元测试，E2E测试 |
| Week 6-7 | 文档完善 | 用户手册，技术文档 |
| Week 7-8 | V1.0发布 | 稳定版本 |

### 10.2 依赖关系

```
数据库设计 → API开发 → 前端集成
                        ↓
核心功能 ←───────────────┘
                        ↓
用户体验 → 测试 → 文档 → 发布
```

---

## 十一、验收标准

### 11.1 功能验收

- [ ] 所有用户故事完成
- [ ] API测试覆盖率100%
- [ ] 核心E2E测试通过
- [ ] 文档完整度>80%

### 11.2 性能验收

- [ ] API响应时间<500ms
- [ ] 首屏加载<3s
- [ ] 无内存泄漏

### 11.3 质量验收

- [ ] 严重Bug数量=0
- [ ] 严重安全漏洞=0
- [ ] 浏览器兼容性达标

---

## 附录

### A. 术语表

| 术语 | 说明 |
|------|------|
| LLM | Large Language Model，大语言模型 |
| LoRA | Low-Rank Adaptation，低秩适应 |
| QLoRA | Quantized LoRA，量化低秩适应 |
| PEFT | Parameter-Efficient Fine-Tuning，参数高效微调 |
| Loss | 训练损失值 |
| Epoch | 训练轮数 |
| Batch | 批次大小 |

### B. 参考资料

- [FastAPI文档](https://fastapi.tiangolo.com/)
- [React文档](https://react.dev/)
- [Ant Design](https://ant.design/)
- [SQLAlchemy文档](https://docs.sqlalchemy.org/)

---

**文档负责人：** AI Platform团队  
**评审日期：** 2026-02-08  
**版本：** V1.0
