"""
Alembic迁移脚本 - 初始化数据库
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """创建所有表"""
    
    # 创建UUID扩展
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
    
    # 用户表
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', postgresql.UUID(), nullable=False),
        sa.Column('username', sa.String(50), nullable=False),
        sa.Column('email', sa.String(100), nullable=False),
        sa.Column('password_hash', sa.String(256), nullable=False),
        sa.Column('role', sa.String(20), server_default='user', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=True, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email')
    )
    
    # 项目表
    op.create_table(
        'projects',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', postgresql.UUID(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('owner_id', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(50), server_default='active', nullable=False),
        sa.Column('config', postgresql.JSON(), server_default='{}', nullable=False),
        sa.Column('tags', postgresql.JSON(), server_default='[]', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid'),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='SET NULL')
    )
    
    # 实验表
    op.create_table(
        'experiments',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', postgresql.UUID(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('base_model', sa.String(255), nullable=True),
        sa.Column('task_type', sa.String(50), nullable=True),
        sa.Column('status', sa.String(50), server_default='pending', nullable=False),
        sa.Column('config', postgresql.JSON(), server_default='{}', nullable=False),
        sa.Column('metrics', postgresql.JSON(), server_default='{}', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE')
    )
    
    # 数据集表
    op.create_table(
        'datasets',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', postgresql.UUID(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('file_path', sa.String(500), nullable=True),
        sa.Column('size', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('format', sa.String(50), nullable=True),
        sa.Column('version', sa.String(50), server_default='v1', nullable=False),
        sa.Column('status', sa.String(50), server_default='uploaded', nullable=False),
        sa.Column('quality_report', postgresql.JSON(), server_default='{}', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE')
    )
    
    # 数据集版本表
    op.create_table(
        'dataset_versions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', postgresql.UUID(), nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=False),
        sa.Column('version', sa.String(20), nullable=False),
        sa.Column('parent_version_id', sa.Integer(), nullable=True),
        sa.Column('commit_message', sa.Text(), nullable=True),
        sa.Column('file_hash', sa.String(64), nullable=True),
        sa.Column('row_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('file_size', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid'),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['parent_version_id'], ['dataset_versions.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL')
    )
    
    # 模型表
    op.create_table(
        'models',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', postgresql.UUID(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('version', sa.String(50), nullable=True),
        sa.Column('framework', sa.String(100), nullable=True),
        sa.Column('file_path', sa.String(500), nullable=True),
        sa.Column('size', sa.BigInteger(), server_default='0', nullable=False),
        sa.Column('metrics', postgresql.JSON(), server_default='{}', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE')
    )
    
    # 任务表
    op.create_table(
        'tasks',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', postgresql.UUID(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=True),
        sa.Column('experiment_id', sa.Integer(), nullable=True),
        sa.Column('model_id', sa.Integer(), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), server_default='pending', nullable=False),
        sa.Column('priority', sa.Integer(), server_default='5', nullable=False),
        sa.Column('config', postgresql.JSON(), server_default='{}', nullable=False),
        sa.Column('logs', sa.Text(), nullable=True),
        sa.Column('progress', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('gpu_usage', postgresql.JSON(), server_default='{}', nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['model_id'], ['models.id'], ondelete='SET NULL')
    )
    
    # 角色表
    op.create_table(
        'roles',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('uuid', postgresql.UUID(), nullable=False),
        sa.Column('name', sa.String(50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('permissions', postgresql.JSON(), server_default='[]', nullable=False),
        sa.Column('is_system', sa.Boolean(), server_default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('uuid'),
        sa.UniqueConstraint('name')
    )
    
    # 项目权限表
    op.create_table(
        'project_permissions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=False),
        sa.Column('role_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['role_id'], ['roles.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id', 'project_id')
    )
    
    # 审计日志表
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=True),
        sa.Column('resource_id', sa.String(36), nullable=True),
        sa.Column('details', postgresql.JSON(), server_default='{}', nullable=False),
        sa.Column('ip_address', sa.ARRAY(sa.String), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL')
    )
    
    # 通知表
    op.create_table(
        'notifications',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('data', postgresql.JSON(), server_default='{}', nullable=False),
        sa.Column('is_read', sa.Boolean(), server_default=False, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    
    # 创建索引
    op.create_index('idx_users_uuid', 'users', ['uuid'], unique=True)
    op.create_index('idx_projects_uuid', 'projects', ['uuid'], unique=True)
    op.create_index('idx_experiments_uuid', 'experiments', ['uuid'], unique=True)
    op.create_index('idx_datasets_uuid', 'datasets', ['uuid'], unique=True)
    op.create_index('idx_models_uuid', 'models', ['uuid'], unique=True)
    op.create_index('idx_tasks_uuid', 'tasks', ['uuid'], unique=True)
    op.create_index('idx_projects_owner', 'projects', ['owner_id'])
    op.create_index('idx_experiments_project', 'experiments', ['project_id'])
    op.create_index('idx_datasets_project', 'datasets', ['project_id'])
    op.create_index('idx_tasks_project', 'tasks', ['project_id'])
    op.create_index('idx_tasks_status', 'tasks', ['status'])
    op.create_index('idx_audit_logs_user', 'audit_logs', ['user_id'])
    op.create_index('idx_notifications_unread', 'notifications', ['user_id', 'is_read'])
    
    # 插入默认角色
    op.execute("""
        INSERT INTO roles (uuid, name, description, permissions, is_system) VALUES
        ('00000000-0000-0000-0000-000000000001', 'admin', '超级管理员，拥有所有权限', '["*"]', true),
        ('00000000-0000-0000-0000-000000000002', 'developer', '开发者，可访问训练和推理', '["projects:read", "projects:write", "tasks:read", "tasks:write", "datasets:read", "datasets:write", "models:read", "models:write", "training:read", "training:write", "inference:read", "inference:write"]', true),
        ('00000000-0000-0000-0000-000000000003', 'viewer', '查看者，仅有读取权限', '["projects:read", "tasks:read", "datasets:read", "models:read", "training:read", "inference:read"]', true)
    """)

def downgrade():
    """删除所有表"""
    op.drop_table('notifications')
    op.drop_table('audit_logs')
    op.drop_table('project_permissions')
    op.drop_table('roles')
    op.drop_table('tasks')
    op.drop_table('models')
    op.drop_table('dataset_versions')
    op.drop_table('datasets')
    op.drop_table('experiments')
    op.drop_table('projects')
    op.drop_table('users')
