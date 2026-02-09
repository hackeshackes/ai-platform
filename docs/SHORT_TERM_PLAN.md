# AI Platform 短期规划（1-2个月）

**版本：** V1.0  
**日期：** 2026-02-08  
**规划周期：** 1-2个月

---

## 一、当前状态评估

### 1.1 已完成

| 模块 | 完成度 | 说明 |
|------|--------|------|
| 前端框架 | 90% | 9个页面UI完成，中文化基本就绪 |
| 后端框架 | 85% | 11个API模块就绪，认证完成 |
| 基础设施 | 70% | Docker配置完成，待部署 |
| 文档 | 60% | PRD、技术选型文档完成 |

### 1.2 待完善

| 模块 | 优先级 | 说明 |
|------|--------|------|
| 前后台数据连接 | P0 | 前端使用模拟数据 |
| 数据持久化 | P0 | 后端使用内存存储 |
| 核心功能 | P1 | Loss曲线、GPU监控、日志查看 |
| 用户体验 | P2 | 加载状态、错误提示 |

---

## 二、短期目标

### 2.1 核心目标

1. **前后台数据打通** - 实现真实API连接
2. **数据持久化** - 从内存存储迁移到SQLite
3. **核心功能完善** - 补充缺失功能
4. **稳定运行** - Bug修复和性能优化

### 2.2 关键指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| API响应时间 | <500ms | 核心接口 |
| 前端加载时间 | <3s | 首屏加载 |
| Bug数量 | <10 | 严重级别 |
| 测试覆盖率 | >60% | 核心功能 |

---

## 三、实施计划

### 阶段一：前后台数据打通（Week 1-2）

#### 3.1.1 前端API配置

**任务列表：**

```markdown
- [ ] 创建API配置层
  - [ ] 定义API_BASE_URL常量
  - [ ] 配置axios实例
  - [ ] 实现请求/响应拦截器
  - [ ] 添加错误处理

- [ ] 实现API调用
  - [ ] GET /api/v1/projects - 项目列表
  - [ ] GET /api/v1/experiments - 实验列表
  - [ ] GET /api/v1/tasks - 任务列表
  - [ ] GET /api/v1/datasets - 数据集列表
  - [ ] GET /api/v1/models - 模型列表
  - [ ] POST /api/v1/auth/token - 登录认证
```

**实现方案：**

```typescript
// frontend/src/api/index.ts
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 30000,
})

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// 响应拦截器
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.detail || '请求失败'
    console.error('API Error:', message)
    return Promise.reject(error)
  }
)

export default api
```

#### 3.1.2 后端数据持久化

**任务列表：**

```markdown
- [ ] 数据库设计
  - [ ] users表 - 用户认证
  - [ ] projects表 - 项目管理
  - [ ] experiments表 - 实验管理
  - [ ] tasks表 - 任务调度
  - [ ] datasets表 - 数据集管理
  - [ ] models表 - 模型管理

- [ ] ORM集成
  - [ ] 选择SQLAlchemy或Tortoise-ORM
  - [ ] 创建数据库模型
  - [ ] 实现CRUD操作
  - [ ] 数据迁移脚本

- [ ] API对接
  - [ ] 更新现有API使用数据库
  - [ ] 添加数据验证
  - [ ] 实现分页和过滤
```

**数据库模型设计：**

```python
# backend/database/models.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
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
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 阶段二：核心功能完善（Week 3-4）

#### 3.2.1 Loss曲线图

**任务列表：**

```markdown
- [ ] 后端API
  - [ ] GET /api/v1/experiments/{id}/loss - 获取Loss数据
  - [ ] 返回格式：[{step: number, loss: number, epoch: number}]

- [ ] 前端实现
  - [ ] 选择图表库（Recharts/ECharts）
  - [ ] 集成Experiments详情页
  - [ ] 添加平滑曲线
  - [ ] 支持缩放和拖拽

- [ ] 数据处理
  - [ ] 实现实时数据更新
  - [ ] 添加加载状态
  - [ ] 错误边界处理
```

**实现方案：**

```tsx
// frontend/src/pages/ExperimentDetail.tsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface LossData {
  step: number
  loss: number
  epoch: number
}

const LossChart: React.FC<{ experimentId: number }> = ({ experimentId }) => {
  const [data, setData] = useState<LossData[]>([])
  const [loading, setLoading] = useState(true)
  
  useEffect(() => {
    const fetchLoss = async () => {
      try {
        const response = await api.get(`/experiments/${experimentId}/loss`)
        setData(response.data)
      } catch (error) {
        console.error('Failed to fetch loss data:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchLoss()
  }, [experimentId])
  
  if (loading) return <Spin tip="加载中..." />
  
  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="step" label="Step" />
        <YAxis label="Loss" />
        <Tooltip />
        <Line type="monotone" dataKey="loss" stroke="#1890ff" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}
```

#### 3.2.2 GPU实时监控

**任务列表：**

```markdown
- [ ] 后端集成
  - [ ] 使用nvidia-ml-py3获取GPU状态
  - [ ] 创建 /api/v1/gpu/status 端点
  - [ ] 实现GPU信息聚合

- [ ] 前端实现
  - [ ] 创建GPU监控卡片组件
  - [ ] 实现实时数据轮询
  - [ ] 添加GPU使用率图表
  - [ ] 显存使用可视化

- [ ] 告警机制
  - [ ] 设置GPU使用阈值
  - [ ] 超出阈值时显示警告
  - [ ] 可配置的告警规则
```

#### 3.2.3 训练日志查看

**任务列表：**

```markdown
- [ ] 日志系统设计
  - [ ] 定义日志格式
  - [ ] 实现日志存储
  - [ ] 日志级别（INFO/WARNING/ERROR）

- [ ] 后端API
  - [ ] GET /api/v1/tasks/{id}/logs - 获取日志
  - [ ] 支持分页
  - [ ] 支持关键字搜索

- [ ] 前端实现
  - [ ] 创建日志查看器组件
  - [ ] 实现实时日志流（SSE）
  - [ ] 日志高亮（ERROR级别）
  - [ ] 自动滚动到底部
```

### 阶段三：用户体验优化（Week 5-6）

#### 3.3.1 加载状态

**任务列表：**

```markdown
- [ ] 全局加载状态
  - [ ] 页面切换加载动画
  - [ ] API请求加载状态
  - [ ] 大数据量分页加载

- [ ] 骨架屏
  - [ ] Dashboard骨架屏
  - [ ] 列表页骨架屏
  - [ ] 详情页骨架屏

- [ ] 进度指示
  - [ ] 文件上传进度
  - [ ] 任务创建进度
  - [ ] 批量操作进度
```

#### 3.3.2 错误处理

**任务列表：**

```markdown
- [ ] 错误边界
  - [ ] React Error Boundary实现
  - [ ] 全局错误捕获
  - [ ] 错误恢复机制

- [ ] 用户提示
  - [ ] Toast消息组件
  - [ ] 成功提示
  - [ ] 警告提示
  - [ ] 错误详情展示

- [ ] 离线支持
  - [ ] Service Worker配置
  - [ ] 离线页面
  - [ ] 请求重试机制
```

#### 3.3.3 响应式布局

**任务列表：**

```markdown
- [ ] 移动端适配
  - [ ] 768px断点适配
  - [ ] 导航栏响应式
  - [ ] 表格滚动处理

- [ ] 平板适配
  - [ ] 1024px断点
  - [ ] 卡片布局调整
  - [ ] 侧边栏折叠

- [ ] 大屏优化
  - [ ] 1440px+适配
  - [ ] Dashboard多列显示
  - [ ] 数据可视化优化
```

### 阶段四：测试与文档（Week 7-8）

#### 3.4.1 测试

**任务列表：**

```markdown
- [ ] 单元测试
  - [ ] 前端组件测试（Jest + React Testing Library）
  - [ ] 后端API测试（pytest）
  - [ ] 工具函数测试

- [ ] 集成测试
  - [ ] API端点测试
  - [ ] 用户流程测试
  - [ ] 数据一致性测试

- [ ] E2E测试
  - [ ] Playwright配置
  - [ ] 核心用户流程
  - [ ] 跨浏览器测试
```

#### 3.4.2 文档

**任务列表：**

```markdown
- [ ] 用户手册
  - [ ] 快速开始指南
  - [ ] 功能说明文档
  - [ ] 常见问题解答

- [ ] 技术文档
  - [ ] API接口文档
  - [ ] 架构设计文档
  - [ ] 部署指南

- [ ] 更新日志
  - [ ] 版本发布记录
  - [ ] 功能变更说明
  - [ ] 已知问题列表
```

---

## 四、里程碑

### 4.1 时间线

| 周次 | 里程碑 | 交付物 |
|------|--------|--------|
| Week 1 | 数据打通完成 | 前端API集成，数据库模型 |
| Week 2 | 核心API就绪 | CRUD API，认证系统 |
| Week 3 | 功能完善完成 | Loss图表，GPU监控，日志查看 |
| Week 4 | UI优化完成 | 加载状态，错误处理 |
| Week 5 | 测试完成 | 测试用例，CI/CD |
| Week 6 | 文档完成 | 用户手册，技术文档 |
| Week 7 | Beta发布 | 内部测试版本 |
| Week 8 | 稳定版本 | V1.0发布 |

### 4.2 验收标准

- [ ] 所有API响应时间 < 500ms
- [ ] 前端首屏加载 < 3s
- [ ] 严重Bug数量 = 0
- [ ] 测试覆盖率 > 60%
- [ ] 文档完整度 > 80%

---

## 五、资源需求

### 5.1 人力投入

| 角色 | 投入时间 | 职责 |
|------|----------|------|
| 前端开发 | 2人 | API集成，UI优化，测试 |
| 后端开发 | 1人 | 数据库，API，文档 |
| 全栈 | 1人 | 架构设计，技术决策 |

### 5.2 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| 前端框架 | React 18 | UI组件库 |
| 图表库 | Recharts/ECharts | 数据可视化 |
| 测试框架 | Jest + React Testing Library | 单元测试 |
| E2E测试 | Playwright | 端到端测试 |
| 后端框架 | FastAPI | API开发 |
| 数据库 | SQLite -> PostgreSQL | 数据持久化 |
| ORM | SQLAlchemy | 数据库操作 |

---

## 六、风险与应对

### 6.1 技术风险

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 数据库性能 | 低 | 添加索引，查询优化 |
| 前端性能 | 中 | 代码分割，懒加载 |
| API稳定性 | 中 | 限流，熔断，降级 |
| 数据一致性 | 低 | 事务管理，幂等设计 |

### 6.2 进度风险

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 需求变更 | 中 | 敏捷开发，快速响应 |
| 技术难点 | 中 | 技术预研，寻求支持 |
| 人力不足 | 低 | 任务优先级排序 |

---

## 七、预算估算

### 7.1 基础设施

| 项目 | 月成本 | 说明 |
|------|--------|------|
| 开发服务器 | ¥500 | GPU服务器（可选） |
| 云存储 | ¥100 | 附件和模型文件 |
| 域名/SSL | ¥50 | HTTPS证书 |
| **合计** | **¥650** | |

### 7.2 人力成本

| 角色 | 人天 | 日薪 | 小计 |
|------|------|------|------|
| 前端开发 | 40 | ¥1500 | ¥60,000 |
| 后端开发 | 30 | ¥1500 | ¥45,000 |
| 技术顾问 | 5 | ¥2000 | ¥10,000 |
| **合计** | | | **¥115,000** |

---

## 八、总结

### 8.1 预期成果

1. **V1.0稳定版本发布**
2. **完整的前后端数据流**
3. **核心功能完善（Loss曲线、GPU监控、日志查看）**
4. **60%+测试覆盖率**
5. **完整的用户和技术文档**

### 8.2 后续规划

短期规划完成后，将进入中期规划，重点关注：
1. **vLLM推理引擎集成**
2. **PEFT高效微调支持**
3. **RAG知识库**
4. **Agent框架**

---

**规划负责人：** AI Platform团队  
**评审日期：** 2026-02-08  
**版本：** V1.0
