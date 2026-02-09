# AI Platform - 大模型全生命周期管理平台

## 📊 项目概述

AI Platform 是一个用于管理大语言模型训练、推理和部署的完整平台。

**核心特性：**
- 🖥️ GPU实时监控 - 竞品普遍缺失，核心差异化功能
- 📈 Loss可视化 - ECharts专业图表
- 🚀 轻量级 - 开箱即用，比MLflow更简单
- 🇨🇳 中文本地化 - 国际化竞品的中文支持弱

## ✅ 已完成功能

### 核心功能
- **用户认证** - JWT登录/注册/Token管理
- **项目管理** - 创建、编辑、删除项目
- **任务管理** - 训练/推理任务状态追踪
- **数据集管理** - 上传、质量报告
- **模型管理** - 模型版本、评估指标

### 监控功能
- **GPU监控** - 实时显存、利用率、温度
- **Loss曲线** - ECharts可视化、缩放拖拽
- **任务日志** - 实时日志查看

### 训练与推理
- **训练任务** - 4步向导提交训练
- **推理服务** - 在线推理、推理历史

## 🛠️ 技术栈

| 层级 | 技术 | 版本 |
|------|------|------|
| **前端框架** | React | 18.x |
| **前端语言** | TypeScript | 5.x |
| **构建工具** | Vite | 5.x |
| **UI组件库** | Ant Design | 5.x |
| **图表库** | ECharts | 6.x |
| **HTTP客户端** | Axios | - |
| **后端框架** | FastAPI | - |
| **后端语言** | Python | 3.14 |
| **数据库** | SQLite | - |
| **认证** | JWT (PyJWT) | - |

## 📁 项目结构

```
ai-platform/
├── frontend/                 # 前端项目 (React + TypeScript)
│   ├── src/
│   │   ├── api/            # API客户端封装
│   │   │   └── client.ts    # Axios配置 + API方法
│   │   ├── components/      # 共享组件
│   │   ├── locales/         # 国际化 (中英文)
│   │   ├── pages/          # 页面组件 (11个)
│   │   │   ├── Login.tsx    # 登录页
│   │   │   ├── Dashboard.tsx # 仪表盘
│   │   │   ├── Projects.tsx # 项目管理
│   │   │   ├── Tasks.tsx    # 任务管理
│   │   │   ├── Datasets.tsx # 数据集管理
│   │   │   ├── Models.tsx   # 模型管理
│   │   │   ├── GPU.tsx     # GPU监控
│   │   │   ├── LossChart.tsx # Loss曲线
│   │   │   ├── Training.tsx # 训练任务
│   │   │   ├── Inference.tsx # 推理服务
│   │   │   └── Settings.tsx # 系统设置
│   │   ├── App.tsx        # 主应用组件
│   │   └── main.tsx       # 入口文件
│   ├── package.json       # 前端依赖
│   └── vite.config.ts    # Vite配置
│
├── backend/                  # 后端项目 (FastAPI)
│   ├── api/
│   │   ├── endpoints/      # API端点 (11个)
│   │   │   ├── auth.py      # 认证模块
│   │   │   ├── projects.py  # 项目管理
│   │   │   ├── tasks.py    # 任务管理
│   │   │   ├── datasets.py  # 数据集管理
│   │   │   ├── models.py   # 模型管理
│   │   │   ├── gpu.py      # GPU监控
│   │   │   ├── metrics.py  # 训练指标
│   │   │   ├── training.py  # 训练任务
│   │   │   ├── inference.py # 推理服务
│   │   │   └── settings.py # 系统设置
│   │   └── routes.py     # 路由聚合
│   ├── main.py           # FastAPI入口
│   └── models.py         # 数据模型
│
└── docs/                    # 项目文档
    ├── API.md              # API文档
    ├── DEPLOYMENT.md       # 部署文档
    ├── USER_MANUAL.md      # 用户手册
    ├── DEVELOPMENT.md     # 开发文档
    ├── ROADMAP.md         # 路线图
    └── V1.1_PLAN.md      # v1.1规划
```

## 🚀 快速开始

### 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|-----------|-----------|
| Node.js | 18.x | 20.x |
| Python | 3.10 | 3.14 |
| npm | 9.x | 10.x |
| pip | 23.x | 24.x |

### 安装步骤

#### 1. 克隆项目

```bash
git clone <repository-url>
cd ai-platform
```

#### 2. 后端安装

```bash
cd backend

# 创建虚拟环境 (推荐)
python3 -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

**requirements.txt 内容：**
```
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
python-jose>=3.3.0
passlib>=1.7.4
python-multipart>=0.0.6
pynvml>=12.0.0
```

#### 3. 前端安装

```bash
cd frontend

# 安装依赖
npm install
```

**package.json 关键依赖：**
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.22.0",
    "antd": "^5.15.0",
    "axios": "^1.6.0",
    "echarts": "^6.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "typescript": "^5.3.0",
    "vite": "^5.1.0"
  }
}
```

### 启动服务

#### 1. 启动后端

```bash
cd backend
source venv/bin/activate  # 如果使用虚拟环境
python3 main.py
```

**输出：**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### 2. 启动前端

```bash
cd frontend
npm run dev
```

**输出：**
```
VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

### 访问平台

1. 打开浏览器访问: **http://localhost:3000**
2. 使用测试账号登录

### 测试账号

| 用户名 | 密码 | 角色 |
|--------|------|------|
| admin | admin123 | 管理员 |

## 📡 API端点

| 模块 | 端点 | 方法 | 说明 |
|------|------|------|------|
| **认证** | /api/v1/auth/token | POST | 登录获取Token |
| | /api/v1/auth/me | GET | 获取当前用户 |
| **项目** | /api/v1/projects | GET | 项目列表 |
| | /api/v1/projects | POST | 创建项目 |
| **任务** | /api/v1/tasks | GET | 任务列表 |
| **数据集** | /api/v1/datasets | GET | 数据集列表 |
| **模型** | /api/v1/models | GET | 模型列表 |
| **GPU** | /api/v1/gpu | GET | GPU状态 |
| **指标** | /api/v1/metrics/loss | GET | Loss曲线 |
| **训练** | /api/v1/training/models | GET | 训练模型 |
| | /api/v1/training/submit | POST | 提交训练 |
| **推理** | /api/v1/inference/models | GET | 推理模型 |
| | /api/v1/inference/generate | POST | 推理生成 |
| **设置** | /api/v1/settings/system | GET | 系统设置 |
| | /api/v1/settings/storage | GET | 存储设置 |

### 认证示例

```bash
# 获取Token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"

# 响应
{
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "bearer",
    "expires_in": 86400
}

# 使用Token访问API
curl http://localhost:8000/api/v1/projects \
  -H "Authorization: Bearer <your_token>"
```

## 🧪 测试

### 后端测试

```bash
cd backend
pytest tests/           # 运行测试
pytest --cov=api tests/  # 带覆盖率
```

### 前端测试

```bash
cd frontend
npm run test            # 运行测试
npm run test -- --coverage  # 带覆盖率
```

## 📦 构建部署

### 前端构建

```bash
cd frontend
npm run build
```

构建产物在 `dist/` 目录，可部署到Nginx、CDN等。

### Docker部署

```bash
# 使用Docker Compose
docker-compose up -d

# 或分别构建
docker build -t ai-platform-backend ./backend
docker build -t ai-platform-frontend ./frontend
```

## 📚 文档

| 文档 | 说明 |
|------|------|
| [API.md](docs/API.md) | REST API接口说明 |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | 环境配置与部署 |
| [USER_MANUAL.md](docs/USER_MANUAL.md) | 平台使用指南 |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | 开发规范与指南 |
| [ROADMAP.md](docs/ROADMAP.md) | 版本规划 |
| [V1.1_PLAN.md](docs/V1.1_PLAN.md) | v1.1开发规划 |

## 🎯 v1.1规划

| 优先级 | 功能 | 工期 |
|--------|------|------|
| P0 | 数据集版本控制 | 3天 |
| P0 | 多用户支持 | 5天 |
| P1 | 数据质量检查 | 3天 |
| P1 | 权限管理(RBAC) | 4天 |

**v1.1预计上线**: 2026-02-23

## 🤝 贡献

1. Fork项目
2. 创建分支 (`git checkout -b feature/xxx`)
3. 提交更改 (`git commit -m 'Add xxx'`)
4. 推送到分支 (`git push origin feature/xxx`)
5. 创建Pull Request

## 📄 许可证

MIT License

## 📅 更新日志

### v1.0.0 (2026-02-08)
- ✅ 完成所有核心功能开发
- ✅ JWT认证系统 (登录/注册/Token)
- ✅ 11个前端页面
- ✅ 11个后端API模块
- ✅ GPU实时监控
- ✅ Loss曲线可视化
- ✅ 5份项目文档

---

**维护者**: AI Development Team  
**项目地址**: [GitHub Repository]
