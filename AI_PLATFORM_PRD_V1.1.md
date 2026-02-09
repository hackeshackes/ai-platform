# AI Platform 产品需求文档 (PRD)

**版本：** V1.1  
**日期：** 2026-02-08  
**基于：** 团队讨论纪要（MLEngineer, CrawlerAgent）  
**状态：** 评审通过

---

## 一、团队讨论共识

### 1.1 技术架构评估

| 组件 | 原方案 | 团队建议 | 决策 |
|------|--------|----------|------|
| 数据库 | SQLite | SQLite+MVP，后续Prisma迁移 | ✅ 原方案 |
| ORM | - | 预留Prisma接口 | ✅ 新增 |
| 实时通信 | WebSocket | WebSocket+SSE | ✅ 确认 |
| GPU监控 | - | pynvml集成 | ✅ 新增 |

### 1.2 功能优先级调整

| 功能 | 原优先级 | 新优先级 | 理由 |
|------|----------|----------|------|
| GPU实时监控 | P1 | **P0** | 差异化功能 |
| 数据集版本 | P2 | **P1** | 数据质量保证 |
| 权限管理(RBAC) | P1 | **P2** | MVP非必需 |
| 任务队列 | - | **P1** | 并发需求 |

---

## 二、功能需求（V1.1更新）

### 2.1 新增功能

#### 2.1.1 数据质量报告

```python
class DataQualityChecker:
    """数据集质量检查"""
    def check_null_ratio(self, df, threshold=0.1):
        """检查空值比例"""
        
    def check_duplicate(self, df):
        """检查重复数据"""
        
    def check_format(self, df, expected_format):
        """检查数据格式"""
        
    def generate_report(self, df):
        """生成质量报告"""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'null_check': {...},
            'duplicates': count,
            'memory_usage': 'MB'
        }
```

#### 2.1.2 GPU监控增强

```python
import pynvml

class GPUMonitor:
    """GPU监控器"""
    def get_usage(self):
        """获取GPU使用率"""
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        return {
            'gpu_util': util.gpu,
            'memory_used': mem.used // 1024**2,
            'memory_total': mem.total // 1024**2,
            'memory_fragmentation': self.calc_frag(mem)
        }
```

---

## 三、技术规格（V1.1更新）

### 3.1 数据模型（更新）

```sql
-- 用户表 (V1.1无变化)
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 数据集表 (新增version字段)
CREATE TABLE datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    project_id INTEGER NOT NULL,
    file_path VARCHAR(500),
    size INTEGER,
    format VARCHAR(50),
    version VARCHAR(50) DEFAULT 'v1',  -- 新增
    quality_report JSON,                 -- 新增
    status VARCHAR(50) DEFAULT 'uploaded',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

-- 任务表 (新增progress字段)
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    project_id INTEGER NOT NULL,
    type VARCHAR(50),
    status VARCHAR(50) DEFAULT 'pending',
    progress FLOAT DEFAULT 0.0,         -- 新增
    gpu_usage JSON,                      -- 新增
    logs TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);
```

### 3.2 API接口（更新）

| 模块 | 接口 | 方法 | V1.0 | V1.1新增 |
|------|------|------|-------|----------|
| 质量 | `/api/datasets/:id/quality` | GET | - | ✅ |
| GPU | `/api/metrics/gpu/detail` | GET | - | ✅ |
| 版本 | `/api/datasets/:id/versions` | GET | - | ✅ |

---

## 四、开发计划（更新）

### Phase 1: 基础设施打通（第1-3周）

| 周次 | 任务 | V1.0 | V1.1更新 | 状态 |
|------|------|-------|----------|------|
| **Week 1** | API客户端配置 | ✅ | - | ✅ 完成 |
| | SQLite数据模型 | ✅ | 增加quality_report, version | ✅ 完成 |
| | JWT认证 | ✅ | - | ⏳ 进行中 |
| **Week 2** | Session管理 | ✅ | - | ⏳ 待开始 |
| | 数据质量检查器 | - | ✅ 新增 | ⏳ 待开始 |
| **Week 3** | Projects CRUD | ✅ | - | ⏳ 待开始 |
| | GPU监控模块 | - | ✅ pynvml集成 | ⏳ 待开始 |

### Phase 2: 核心功能增强（第4-6周）

| 周次 | 任务 | V1.0 | V1.1更新 |
|------|------|-------|----------|
| **Week 4** | Loss曲线图 | ✅ | - |
| | GPU实时监控 | P1→P0 | ✅ 增强 |
| **Week 5** | 任务状态实时 | ✅ | - |
| | 任务队列管理 | - | ✅ 新增 |
| **Week 6** | 推理API对接 | ✅ | - |

---

## 五、风险与对策（V1.1新增）

| 风险 | 级别 | 对策 |
|------|------|------|
| SQLite并发限制 | 中 | MVP阶段用锁，后续迁移PostgreSQL |
| GPU监控兼容 | 低 | 备用方案：CPU监控fallback |
| 任务队列丢失 | 中 | Celery+Redis持久化 |

---

## 六、验收标准（V1.1更新）

### 6.1 功能验收

- [ ] API客户端集成到所有页面
- [ ] 用户登录/注册/登出正常
- [ ] Projects CRUD功能完整
- [ ] 数据集上传+质量报告生成
- [ ] GPU监控在有GPU环境正常显示
- [ ] 训练任务状态实时更新

### 6.2 性能验收

- [ ] API响应时间 < 500ms
- [ ] 页面加载时间 < 2s
- [ ] WebSocket连接稳定
- [ ] 支持10+并发任务

---

**文档版本**: V1.1  
**更新日期**: 2026-02-08  
**更新内容**: 基于团队讨论增加GPU监控、数据质量功能
