# AI Platform v2.0 PostgreSQL Schema

## 创建数据库

```sql
-- 创建数据库和用户
CREATE USER aiplatform WITH PASSWORD 'your_password';
CREATE DATABASE ai_platform OWNER aiplatform;
GRANT ALL PRIVILEGES ON DATABASE ai_platform TO aiplatform;

-- 连接数据库
\c ai_platform

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
```

## 表结构

```sql
-- 用户表
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP WITH TIME ZONE
);

-- 项目表
CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    status VARCHAR(50) DEFAULT 'active',
    config JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 实验表
CREATE TABLE experiments (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    base_model VARCHAR(255),
    task_type VARCHAR(50),
    status VARCHAR(50) DEFAULT 'pending',
    config JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 数据集表
CREATE TABLE datasets (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    size BIGINT DEFAULT 0,
    format VARCHAR(50),
    version VARCHAR(50) DEFAULT 'v1',
    status VARCHAR(50) DEFAULT 'uploaded',
    quality_report JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 数据集版本表
CREATE TABLE dataset_versions (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    dataset_id INTEGER REFERENCES datasets(id) ON DELETE CASCADE,
    version VARCHAR(20) NOT NULL,
    parent_version_id INTEGER REFERENCES dataset_versions(id),
    commit_message TEXT,
    file_hash VARCHAR(64),
    row_count INTEGER DEFAULT 0,
    file_size BIGINT DEFAULT 0,
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 模型表
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50),
    framework VARCHAR(100),
    file_path VARCHAR(500),
    size BIGINT DEFAULT 0,
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 任务表
CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    project_id INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    experiment_id INTEGER REFERENCES experiments(id) ON DELETE SET NULL,
    model_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    config JSONB DEFAULT '{}',
    logs TEXT,
    progress FLOAT DEFAULT 0.0,
    gpu_usage JSONB DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- 角色表
CREATE TABLE roles (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB NOT NULL DEFAULT '[]',
    is_system BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 项目权限表
CREATE TABLE project_permissions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    role_id INTEGER REFERENCES roles(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, project_id)
);

-- 操作审计表
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(36),
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 通知表
CREATE TABLE notifications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT,
    data JSONB DEFAULT '{}',
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Celery任务表
CREATE TABLE celery_tasks (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    args JSONB DEFAULT '[]',
    kwargs JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'PENDING',
    result JSONB,
    traceback TEXT,
    runtime FLOAT,
    worker VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- 索引
CREATE INDEX idx_users_uuid ON users(uuid);
CREATE INDEX idx_projects_uuid ON projects(uuid);
CREATE INDEX idx_experiments_uuid ON experiments(uuid);
CREATE INDEX idx_datasets_uuid ON datasets(uuid);
CREATE INDEX idx_models_uuid ON models(uuid);
CREATE INDEX idx_tasks_uuid ON tasks(uuid);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX idx_notifications_unread ON notifications(user_id, is_read) WHERE is_read = FALSE;
```

## 初始化数据

```sql
-- 插入默认角色
INSERT INTO roles (name, description, permissions, is_system) VALUES
('admin', '超级管理员，拥有所有权限', '["*"]', TRUE),
('developer', '开发者，可访问训练和推理', '["projects:read", "projects:write", "tasks:read", "tasks:write", "datasets:read", "datasets:write", "models:read", "models:write", "training:read", "training:write", "inference:read", "inference:write"]', TRUE),
('viewer', '查看者，仅有读取权限', '["projects:read", "tasks:read", "datasets:read", "models:read", "training:read", "inference:read"]', TRUE);

-- 插入默认管理员用户 (密码: admin123)
INSERT INTO users (username, email, password_hash, role) VALUES
('admin', 'admin@ai-platform.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4eVCeM9e1a5t2E3a', 'admin');
```
