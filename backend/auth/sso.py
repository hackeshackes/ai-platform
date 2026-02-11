"""
SSO Handler - SSO认证处理器

支持：
- Okta SSO集成
- Azure AD SSO集成
- 统一SSO接口
"""

import secrets
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import httpx
import jwt
from fastapi import HTTPException, status

from core.config import settings
from auth.jwt_handler import jwt_handler, TokenType
from db.database import db


class SSOProvider(str, Enum):
    """SSO提供商枚举"""
    OKTA = "okta"
    AZURE_AD = "azure"


@dataclass
class SSOUserInfo:
    """SSO用户信息"""
    sub: str  # 唯一标识符
    email: str
    name: Optional[str] = None
    email_verified: bool = True
    groups: list = field(default_factory=list)
    tenant_id: Optional[str] = None
    raw_claims: Dict[str, Any] = field(default_factory=dict)


class OktaSSOHandler:
    """Okta SSO处理器"""
    
    def __init__(self):
        self.client_id = settings.okta_client_id
        self.client_secret = settings.okta_client_secret
        self.issuer = settings.okta_issuer
        self.redirect_uri = settings.okta_redirect_uri
    
    def get_authorization_url(self, state: str) -> str:
        """生成Okta授权URL"""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": "openid profile email groups",
            "state": state
        }
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.issuer}/v1/authorize?{query_string}"
    
    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """使用授权码交换令牌"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.issuer}/v1/token",
                data={
                    "grant_type": "authorization_code",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": self.redirect_uri
                }
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange code for tokens"
                )
            return response.json()
    
    async def get_user_info(self, access_token: str) -> SSOUserInfo:
        """获取Okta用户信息"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.issuer}/v1/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get user info"
                )
            claims = response.json()
            return SSOUserInfo(
                sub=claims.get("sub", ""),
                email=claims.get("email", ""),
                name=claims.get("name"),
                email_verified=claims.get("email_verified", True),
                groups=claims.get("groups", []),
                raw_claims=claims
            )


class AzureADSSOHandler:
    """Azure AD SSO处理器"""
    
    def __init__(self):
        self.client_id = settings.azure_client_id
        self.client_secret = settings.azure_client_secret
        self.tenant_id = settings.azure_tenant_id
        self.redirect_uri = settings.azure_redirect_uri
        self.graph_endpoint = "https://graph.microsoft.com/v1.0"
    
    def get_authorization_url(self, state: str) -> str:
        """生成Azure AD授权URL"""
        base_url = f"https://login.microsoftonline.com/{self.tenant_id}"
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": "openid profile email User.Read",
            "state": state,
            "response_mode": "query"
        }
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{base_url}/oauth2/v2.0/authorize?{query_string}"
    
    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """使用授权码交换令牌"""
        base_url = f"https://login.microsoftonline.com/{self.tenant_id}"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/oauth2/v2.0/token",
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "grant_type": "authorization_code"
                }
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange code for tokens"
                )
            return response.json()
    
    async def get_user_info(self, access_token: str) -> SSOUserInfo:
        """获取Azure AD用户信息"""
        async with httpx.AsyncClient() as client:
            # 获取用户信息
            user_response = await client.get(
                f"{self.graph_endpoint}/me",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            # 获取用户组成员
            groups_response = await client.get(
                f"{self.graph_endpoint}/me/memberOf",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if user_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get user info"
                )
            
            user_data = user_response.json()
            groups = []
            if groups_response.status_code == 200:
                groups = [g.get("displayName", "") for g in groups_response.json().get("value", [])]
            
            return SSOUserInfo(
                sub=user_data.get("id", ""),
                email=user_data.get("mail", user_data.get("userPrincipalName", "")),
                name=user_data.get("displayName"),
                email_verified=True,
                groups=groups,
                raw_claims=user_data
            )


class SSOHandler:
    """统一SSO处理器"""
    
    def __init__(self):
        self.okta_handler = OktaSSOHandler()
        self.azure_handler = AzureADSSOHandler()
    
    def get_auth_url(self, provider: SSOProvider, state: str) -> str:
        """获取指定提供商的授权URL"""
        if provider == SSOProvider.OKTA:
            return self.okta_handler.get_authorization_url(state)
        elif provider == SSOProvider.AZURE_AD:
            return self.azure_handler.get_authorization_url(state)
        else:
            raise ValueError(f"Unsupported SSO provider: {provider}")
    
    async def handle_callback(
        self,
        provider: SSOProvider,
        code: str,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """处理SSO回调"""
        # 交换令牌
        if provider == SSOProvider.OKTA:
            tokens = await self.okta_handler.exchange_code_for_tokens(code)
            user_info = await self.okta_handler.get_user_info(tokens["access_token"])
        elif provider == SSOProvider.AZURE_AD:
            tokens = await self.azure_handler.exchange_code_for_tokens(code)
            user_info = await self.azure_handler.get_user_info(tokens["access_token"])
        else:
            raise ValueError(f"Unsupported SSO provider: {provider}")
        
        # 查找或创建用户
        user = await self._find_or_create_sso_user(user_info, provider, tenant_id)
        
        # 生成JWT令牌
        access_token = jwt_handler.create_access_token(
            user_id=str(user["_id"]),
            email=user["email"],
            tenant_id=str(user.get("tenant_id", "")),
            roles=user.get("roles", [])
        )
        refresh_token = jwt_handler.create_refresh_token(
            user_id=str(user["_id"]),
            tenant_id=str(user.get("tenant_id", ""))
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": jwt_handler.access_token_expire_seconds,
            "user": {
                "user_id": str(user["_id"]),
                "email": user["email"],
                "name": user.get("name"),
                "tenant_id": str(user.get("tenant_id", "")),
                "roles": user.get("roles", [])
            }
        }
    
    async def _find_or_create_sso_user(
        self,
        user_info: SSOUserInfo,
        provider: SSOProvider,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """查找或创建SSO用户"""
        # 查找现有用户
        existing_user = await db.users.find_one({
            "sso_identifiers": {
                "$elemMatch": {
                    "provider": provider.value,
                    "sub": user_info.sub
                }
            }
        })
        
        if existing_user:
            # 更新令牌信息
            await db.users.update_one(
                {"_id": existing_user["_id"]},
                {
                    "$set": {
                        "last_login_at": datetime.now(timezone.utc),
                        f"sso_tokens.{provider.value}": {
                            "access_token": "***",  # 不存储实际令牌
                            "updated_at": datetime.now(timezone.utc)
                        }
                    }
                }
            )
            return existing_user
        
        # 确定租户
        target_tenant_id = tenant_id
        if not target_tenant_id:
            # 如果没有指定租户，创建新租户
            tenant = await db.tenants.insert_one({
                "name": user_info.name or user_info.email.split("@")[0],
                "created_at": datetime.now(timezone.utc),
                "quota": {
                    "max_users": 10,
                    "max_projects": 5,
                    "max_storage_gb": 100,
                    "api_calls_per_month": 10000
                },
                "settings": {
                    "allow_custom_domains": False,
                    "sso_provider": provider.value
                }
            })
            target_tenant_id = str(tenant.inserted_id)
        
        # 创建新用户
        new_user = {
            "email": user_info.email,
            "name": user_info.name,
            "password_hash": None,  # SSO用户无需密码
            "tenant_id": target_tenant_id,
            "roles": ["user"],
            "is_active": True,
            "is_verified": user_info.email_verified,
            "created_at": datetime.now(timezone.utc),
            "last_login_at": datetime.now(timezone.utc),
            "sso_identifiers": [{
                "provider": provider.value,
                "sub": user_info.sub
            }],
            "sso_tokens": {
                provider.value: {
                    "access_token": "***",
                    "updated_at": datetime.now(timezone.utc)
                }
            },
            "profile": {
                "avatar_url": None,
                "department": user_info.raw_claims.get("department"),
                "job_title": user_info.raw_claims.get("jobTitle")
            }
        }
        
        result = await db.users.insert_one(new_user)
        new_user["_id"] = result.inserted_id
        return new_user


# 全局SSO处理器实例
sso_handler = SSOHandler()
