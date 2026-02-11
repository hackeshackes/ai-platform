# AI Platform v6 - 企业级SSO与多租户

## 概述

本模块实现企业级单点登录(SSO)集成和多租户隔离功能。

## 核心功能

### 1. SSO集成
- **Okta SSO** - Okta身份管理集成
- **Azure AD SSO** - Microsoft Azure Active Directory集成

### 2. 多租户支持
- 租户资源隔离
- 租户级权限管理
- 租户级配额管理

## 文件结构

```
backend/
├── auth/
│   └── sso.py              # SSO处理器
├── tenant/
│   ├── __init__.py
│   ├── manager.py          # 租户管理器
│   └── middleware.py       # 租户中间件
└── api/
    └── endpoints/
        ├── sso.py          # SSO API端点
        └── tenant.py       # 租户API端点
```

## API端点

### SSO认证
| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/auth/sso/okta` | POST | Okta SSO登录 |
| `/api/v1/auth/sso/azure` | POST | Azure AD SSO登录 |
| `/api/v1/auth/sso/url` | GET | 获取SSO授权URL |
| `/api/v1/auth/sso/providers` | GET | 获取支持的SSO提供商 |

### 租户管理
| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/tenant/me` | GET | 获取当前租户 |
| `/api/v1/tenant/users` | GET | 租户用户列表 |
| `/api/v1/tenant/quota` | GET/POST | 获取/设置配额 |
| `/api/v1/tenant/settings` | GET/PUT | 获取/更新设置 |
| `/api/v1/tenant/usage` | GET | 资源使用情况 |
| `/api/v1/tenant/members/{user_id}/role` | POST | 更新成员角色 |
| `/api/v1/tenant/members/{user_id}` | DELETE | 移除成员 |
| `/api/v1/tenant/members/{user_id}/invite` | POST | 邀请成员 |

## 环境变量配置

### Okta SSO
```bash
OKTA_CLIENT_ID=your_client_id
OKTA_CLIENT_SECRET=your_client_secret
OKTA_ISSUER=https://dev-xxxxxx.okta.com/oauth2/default
OKTA_REDIRECT_URI=http://localhost:8000/api/v1/auth/sso/okta/callback
```

### Azure AD SSO
```bash
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
AZURE_TENANT_ID=your_tenant_id
AZURE_REDIRECT_URI=http://localhost:8000/api/v1/auth/sso/azure/callback
```

## 使用示例

### Okta SSO登录
```bash
curl -X POST http://localhost:8000/api/v1/auth/sso/okta \
  -H "Content-Type: application/json" \
  -d '{
    "code": "authorization_code_from_okta",
    "state": "optional_state_param"
  }'
```

### 设置租户配额
```bash
curl -X POST http://localhost:8000/api/v1/tenant/quota \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "max_users": 50,
    "max_projects": 20,
    "max_storage_gb": 500
  }'
```

## 配额管理

每个租户支持以下配额限制：
- `max_users` - 最大用户数
- `max_projects` - 最大项目数
- `max_storage_gb` - 最大存储空间(GB)
- `api_calls_per_month` - 每月API调用次数

## 中间件

- `TenantMiddleware` - 自动注入租户上下文
- `QuotaCheckMiddleware` - 配额检查
- `TenantSettingsMiddleware` - 应用租户设置
