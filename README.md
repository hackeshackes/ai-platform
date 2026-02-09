# AI Platform - 大模型全生命周期管理平台

<div align="center">

![AI Platform](https://img.shields.io/badge/AI-Platform-blue?style=for-the-badge)
![React](https://img.shields.io/badge/React-18.x-61DAFB?style=flat-square&logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=flat-square&logo=typescript)
![FastAPI](https://img.shields.io/badge/FastAPI-Python-009688?style=flat-square&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**一个用于管理大语言模型训练、推理和部署的完整平台**

[English](README.md) | [中文](README_CN.md)

</div>

---

## 📊 项目概述

AI Platform 是一个专为大语言模型(LLM)设计的全生命周期管理平台，提供从数据准备到模型部署的完整解决方案。

### ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🖥️ **GPU实时监控** | 竞品普遍缺失，核心差异化功能 |
| 📈 **Loss可视化** | ECharts专业图表，支持缩放拖拽 |
| 🚀 **轻量级** | 开箱即用，比MLflow更简单 |
| 🇨🇳 **中文本地化** | 国际化竞品的中文支持弱 |
| 🔐 **安全认证** | JWT双令牌认证体系 |

---

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

---

## 🛠️ 技术栈

### 前端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| React | 18.x | UI框架 |
| TypeScript | 5.x | 类型安全 |
| Vite | 5.x | 构建工具 |
| Ant Design | 5.x | UI组件库 |
| ECharts | 6.x | 图表可视化 |
| Axios | - | HTTP客户端 |
| React Router | 6.x | 路由管理 |

### 后端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| FastAPI | - | Web框架 |
| Python | 3.14 | 编程语言 |
| SQLite | - | 数据库 |
| SQLAlchemy | 2.x | ORM |
| Pydantic | 2.x | 数据验证 |
| PyJWT | - | 认证 |
| Passlib | - | 密码加密 |
| PyNVML | 12.x | GPU监控 |

---

## 📁 项目结构

```
ai-platform/
├── frontend/                 # 前端项目
│   ├── src/
│   │   ├── api/            # API客户端
│   │   │   └── client.ts    # Axios封装
│   │   ├── components/      # 共享组件
│   │   ├── locales/        # 国际化
│   │   ├── pages/          # 页面组件
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
│   │   │   ├── Settings.tsx # 系统设置
│   │   │   ├── DatasetVersions.tsx # 版本管理
│   │   │   └── DataQuality.tsx # 数据质量
│   │   ├── App.tsx        # 主应用
│   │   └── main.tsx       # 入口
│   ├── package.json       # 依赖配置
│   └── vite.config.ts    # Vite配置
│
├── backend/                  # 后端项目
│   ├── api/
│   │   ├── endpoints/      # API端点
│   │   │   ├── auth.py      # 认证
│   │   │   ├── projects.py  # 项目
│   │   │   ├── tasks.py    # 任务
│   │   │   ├── datasets.py  # 数据集
│   │   │   ├── models.py   # 模型
│   │   │   ├── gpu.py      # GPU监控
│   │   │   ├── metrics.py  # 指标
│   │   │   ├── training.py  # 训练
│   │   │   ├── inference.py # 推理
│   │   │   ├── settings.py # 设置
│   │   │   ├── versions.py # 版本(v1.1)
│   │   │   ├── quality.py # 质量(v1.1)
│   │   │   └── users.py   # 用户(v1.1)
│   │   └── routes.py     # 路由
│   ├── core/
│   │   └── quality_checker.py # 质量检查
│   ├── main.py           # FastAPI入口
│   └── models.py         # 数据模型
│
├── docs/                    # 文档
│   ├── API.md              # API文档
│   ├── DEPLOYMENT.md      # 部署文档
│   ├── USER_MANUAL.md     # 用户手册
│   ├── DEVELOPMENT.md     # 开发文档
│   ├── ROADMAP.md         # 路线图
│   └── TEST_REPORT.md     # 测试报告
│
├── requirements.txt        # Python依赖
├── .gitignore             # Git忽略
└── README.md              # 说明文档
```

---

## 🚀 快速开始

### 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|-----------|-----------|
| Node.js | 18.x | 20.x |
| Python | 3.10 | 3.14 |
| npm | 9.x | 10.x |
| pip | 23.x | 24.x |

### 前置条件

- Node.js 和 npm 已安装
- Python 3.10+ 已安装
- Git 已安装

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/hackeshackes/ai-platform.git
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
```txt
# AI Platform Backend Dependencies
# Generated: 2026-02-08

# Web Framework
fastapi>=0.109.0
uvicorn[standard]>=0.27.0

# Data Validation
pydantic>=2.5.0
email-validator>=2.1.0

# Authentication
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6

# Database
sqlalchemy>=2.0.0

# GPU Monitoring (optional)
pynvml>=12.0.0

# File Upload
python-magic>=0.4.27

# Configuration
pyyaml>=6.0.0

# Development
pytest>=7.4.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0
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

---

## 📖 使用手册

### 登录认证

1. 打开浏览器访问 `http://localhost:3000`
2. 输入用户名和密码
3. 点击"登录"按钮
4. 登录成功后自动跳转到仪表盘

### 项目管理

1. 点击左侧菜单"项目管理"
2. 点击"新建项目"按钮
3. 填写项目名称和描述
4. 点击"创建"

### GPU监控

1. 点击左侧菜单"GPU监控"
2. 查看实时显存使用情况
3. 监控GPU利用率和温度

### 训练任务

1. 点击左侧菜单"训练任务"
2. 点击"新建训练任务"
3. 选择模型、数据集
4. 配置训练参数
5. 提交训练

### 推理服务

1. 点击左侧菜单"推理服务"
2. 选择推理模型
3. 输入提示词
4. 点击"生成"

---

## 📡 API端点

### 认证模块

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/auth/token` | POST | 登录获取Token |
| `/api/v1/auth/me` | GET | 获取当前用户 |

### 核心模块

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/projects` | GET/POST | 项目列表/创建 |
| `/api/v1/tasks` | GET | 任务列表 |
| `/api/v1/datasets` | GET | 数据集列表 |
| `/api/v1/models` | GET | 模型列表 |

### 监控模块

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/gpu` | GET | GPU状态 |
| `/api/v1/metrics/loss` | GET | Loss曲线 |

### 训练与推理

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/training/models` | GET | 训练模型列表 |
| `/api/v1/training/submit` | POST | 提交训练任务 |
| `/api/v1/inference/models` | GET | 推理模型列表 |
| `/api/v1/inference/generate` POST | 推理生成 |

### 设置

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/settings/system` | GET | 系统配置 |
| `/api/v1/settings/storage` | GET | 存储配置 |

### API使用示例

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

---

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

---

## 📦 构建部署

### 前端构建

```bash
cd frontend
npm run build
```

构建产物在 `dist/` 目录。

### Docker部署

```bash
# 使用Docker Compose
docker-compose up -d

# 或分别构建
docker build -t ai-platform-backend ./backend
docker build -t ai-platform-frontend ./frontend
```

### Nginx配置示例

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # 前端静态文件
    location / {
        root /var/www/ai-platform/dist;
        try_files $uri $uri/ /index.html;
    }
    
    # 后端API代理
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📚 文档

| 文档 | 说明 |
|------|------|
| [API.md](docs/API.md) | REST API接口说明 |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | 环境配置与部署 |
| [USER_MANUAL.md](docs/USER_MANUAL.md) | 平台使用指南 |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | 开发规范与指南 |
| [ROADMAP.md](docs/ROADMAP.md) | 版本规划 |
| [TEST_REPORT.md](docs/TEST_REPORT.md) | 测试报告 |

---

## 🎯 v1.1规划

| 优先级 | 功能 | 工期 | 状态 |
|--------|------|------|------|
| P0 | 数据集版本控制 | 3天 | ✅ 完成 |
| P0 | 多用户支持 | 5天 | 🔄 进行中 |
| P1 | 数据质量检查 | 3天 | ✅ 完成 |
| P1 | 权限管理(RBAC) | 4天 | ⏳ 待开发 |

**v1.1预计上线**: 2026-02-23

---

## 🤝 贡献

1. Fork项目
2. 创建分支 (`git checkout -b feature/xxx`)
3. 提交更改 (`git commit -m 'Add xxx'`)
4. 推送到分支 (`git push origin feature/xxx`)
5. 创建Pull Request

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 📅 更新日志

### v1.0.0 (2026-02-08)
- ✅ 完成所有核心功能开发
- ✅ JWT认证系统
- ✅ 11个前端页面
- ✅ 11个后端API模块
- ✅ GPU实时监控
- ✅ Loss曲线可视化
- ✅ 6份项目文档

---

<div align="center">

**维护者**: AI Development Team  
**项目地址**: https://github.com/hackeshackes/ai-platform

⭐ 如果项目对你有帮助，欢迎Star！

</div>

---

## 🚀 v2.0 开发中

### Phase 1: 生产化基础 (进行中)

| 功能 | 状态 | 说明 |
|------|------|------|
| PostgreSQL升级 | 🔄 进行中 | Schema设计完成，迁移中 |
| Redis缓存 | 🔄 进行中 | CacheManager已创建 |
| Celery任务队列 | ✅ 完成 | 4种任务类型已定义 |
| Alembic迁移 | 🔄 进行中 | 迁移配置已完成 |

### v2.0 技术升级

| 组件 | 当前 | v2.0升级 |
|------|------|----------|
| 数据库 | SQLite | PostgreSQL 15 |
| 缓存 | 无 | Redis 7 |
| 任务队列 | 同步 | Celery 5 |
| ORM | SQLAlchemy | SQLAlchemy 2.x + Alembic |

### v2.0 资源

- [v2.0规划](docs/V2.0_PLAN.md)
- [v2.0详细设计](docs/V2.0_DETAILED_DESIGN.md)
- [PostgreSQL Schema](docs/SCHEMA.md)

---

**维护者**: AI Development Team  
**项目地址**: https://github.com/hackeshackes/ai-platform  
**v2.0开发分支**: main
