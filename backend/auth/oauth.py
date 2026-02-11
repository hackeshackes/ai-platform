"""
OAuth 2.0 Handler - OAuth 2.0处理器

支持：
- GitHub OAuth
- Google OAuth
- Microsoft OAuth

功能：
- OAuth授权URL生成
- 访问令牌交换
- 用户信息获取
"""
import httpx
import secrets
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from ..core.config import settings
from ..core.exceptions import OAuthError


class OAuthProvider(Enum):
    GITHUB = "github"
    GOOGLE = "google"
    MICROSOFT = "microsoft"


@dataclass
class OAuthUserInfo:
    """OAuth用户信息"""
    provider: OAuthProvider
    provider_id: str
    email: Optional[str]
    name: Optional[str]
    avatar_url: Optional[str]
    raw_info: Dict[str, Any]


class OAuthBaseHandler(ABC):
    """OAuth处理器基类"""
    
    def __init__(self, provider: OAuthProvider):
        self.provider = provider
        self.client_id = self._get_client_id()
        self.client_secret = self._get_client_secret()
        self.redirect_uri = settings.OAUTH_REDIRECT_URI
    
    @abstractmethod
    def _get_client_id(self) -> str:
        """获取客户端ID"""
        pass
    
    @abstractmethod
    def _get_client_secret(self) -> str:
        """获取客户端密钥"""
        pass
    
    @abstractmethod
    def get_authorization_url(self, state: str) -> str:
        """获取授权URL"""
        pass
    
    @abstractmethod
    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """交换授权码为令牌"""
        pass
    
    @abstractmethod
    async def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """获取用户信息"""
        pass
    
    def generate_state(self) -> str:
        """生成安全的state参数防止CSRF"""
        return secrets.token_urlsafe(32)
    
    def verify_state(self, state: str, stored_state: str) -> bool:
        """验证state参数"""
        return secrets.compare_digest(state, stored_state)


class GitHubOAuthHandler(OAuthBaseHandler):
    """GitHub OAuth处理器"""
    
    def __init__(self):
        super().__init__(OAuthProvider.GITHUB)
        self.auth_url = "https://github.com/login/oauth/authorize"
        self.token_url = "https://github.com/login/oauth/access_token"
        self.user_api_url = "https://api.github.com/user"
    
    def _get_client_id(self) -> str:
        return settings.GITHUB_OAUTH_CLIENT_ID or ""
    
    def _get_client_secret(self) -> str:
        return settings.GITHUB_OAUTH_CLIENT_SECRET or ""
    
    def get_authorization_url(self, state: str) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "read:user user:email",
            "state": state
        }
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.auth_url}?{query_string}"
    
    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": self.redirect_uri
                },
                headers={"Accept": "application/json"}
            )
            
            if response.status_code != 200:
                raise OAuthError(f"Failed to exchange code: {response.text}")
            
            return response.json()
    
    async def get_user_info(self, access_token: str) -> OAuthUserInfo:
        async with httpx.AsyncClient() as client:
            # 获取用户信息
            user_response = await client.get(
                self.user_api_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json"
                }
            )
            
            if user_response.status_code != 200:
                raise OAuthError(f"Failed to get user info: {user_response.text}")
            
            user_data = user_response.json()
            
            # 获取邮箱（如果需要）
            email = user_data.get("email")
            if not email:
                email_response = await client.get(
                    "https://api.github.com/user/emails",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/json"
                    }
                )
                
                if email_response.status_code == 200:
                    emails = email_response.json()
                    primary_email = next(
                        (e for e in emails if e.get("primary")),
                        emails[0]
                    )
                    email = primary_email.get("email")
            
            return OAuthUserInfo(
                provider=self.provider,
                provider_id=str(user_data.get("id")),
                email=email,
                name=user_data.get("name"),
                avatar_url=user_data.get("avatar_url"),
                raw_info=user_data
            )


class GoogleOAuthHandler(OAuthBaseHandler):
    """Google OAuth处理器"""
    
    def __init__(self):
        super().__init__(OAuthProvider.GOOGLE)
        self.auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"
        self.user_api_url = "https://www.googleapis.com/oauth2/v2/userinfo"
    
    def _get_client_id(self) -> str:
        return settings.GOOGLE_OAUTH_CLIENT_ID or ""
    
    def _get_client_secret(self) -> str:
        return settings.GOOGLE_OAUTH_CLIENT_SECRET or ""
    
    def get_authorization_url(self, state: str) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "offline",
            "prompt": "consent"
        }
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.auth_url}?{query_string}"
    
    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "grant_type": "authorization_code"
                }
            )
            
            if response.status_code != 200:
                raise OAuthError(f"Failed to exchange code: {response.text}")
            
            return response.json()
    
    async def get_user_info(self, access_token: str) -> OAuthUserInfo:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.user_api_url,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if response.status_code != 200:
                raise OAuthError(f"Failed to get user info: {response.text}")
            
            user_data = response.json()
            
            return OAuthUserInfo(
                provider=self.provider,
                provider_id=user_data.get("id"),
                email=user_data.get("email"),
                name=user_data.get("name"),
                avatar_url=user_data.get("picture"),
                raw_info=user_data
            )


class MicrosoftOAuthHandler(OAuthBaseHandler):
    """Microsoft OAuth处理器"""
    
    def __init__(self):
        super().__init__(OAuthProvider.MICROSOFT)
        self.tenant_id = settings.MICROSOFT_OAUTH_TENANT_ID or "common"
        self.auth_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/authorize"
        self.token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        self.user_api_url = "https://graph.microsoft.com/v1.0/me"
    
    def _get_client_id(self) -> str:
        return settings.MICROSOFT_OAUTH_CLIENT_ID or ""
    
    def _get_client_secret(self) -> str:
        return settings.MICROSOFT_OAUTH_CLIENT_SECRET or ""
    
    def get_authorization_url(self, state: str) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": "openid email profile User.Read",
            "state": state
        }
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.auth_url}?{query_string}"
    
    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "grant_type": "authorization_code"
                }
            )
            
            if response.status_code != 200:
                raise OAuthError(f"Failed to exchange code: {response.text}")
            
            return response.json()
    
    async def get_user_info(self, access_token: str) -> OAuthUserInfo:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.user_api_url,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if response.status_code != 200:
                raise OAuthError(f"Failed to get user info: {response.text}")
            
            user_data = response.json()
            
            # 获取邮箱信息
            email = user_data.get("mail") or user_data.get("userPrincipalName")
            
            # 获取显示名称
            name = user_data.get("displayName")
            
            return OAuthUserInfo(
                provider=self.provider,
                provider_id=user_data.get("id"),
                email=email,
                name=name,
                avatar_url=None,  # Microsoft Graph需要额外请求
                raw_info=user_data
            )


class OAuthHandler:
    """OAuth处理器工厂"""
    
    _handlers = {
        OAuthProvider.GITHUB: GitHubOAuthHandler,
        OAuthProvider.GOOGLE: GoogleOAuthHandler,
        OAuthProvider.MICROSOFT: MicrosoftOAuthHandler
    }
    
    @classmethod
    def get_handler(cls, provider: str) -> OAuthBaseHandler:
        """
        获取指定提供商的处理器
        
        Args:
            provider: 提供商名称 (github, google, microsoft)
            
        Returns:
            OAuthBaseHandler: OAuth处理器实例
            
        Raises:
            ValueError: 不支持的提供商
        """
        try:
            provider_enum = OAuthProvider(provider.lower())
            handler_class = cls._handlers.get(provider_enum)
            if not handler_class:
                raise ValueError(f"Unsupported OAuth provider: {provider}")
            return handler_class()
        except ValueError:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
    
    @classmethod
    def get_authorization_url(cls, provider: str, state: str) -> str:
        """获取指定提供商的授权URL"""
        handler = cls.get_handler(provider)
        return handler.get_authorization_url(state)
    
    @classmethod
    async def handle_callback(
        cls,
        provider: str,
        code: str,
        state: str,
        stored_state: str
    ) -> OAuthUserInfo:
        """
        处理OAuth回调
        
        Args:
            provider: 提供商名称
            code: 授权码
            state: 回调中的state参数
            stored_state: 存储的state参数
            
        Returns:
            OAuthUserInfo: 用户信息
            
        Raises:
            OAuthError: OAuth处理失败
        """
        handler = cls.get_handler(provider)
        
        # 验证state
        if not handler.verify_state(state, stored_state):
            raise OAuthError("Invalid state parameter")
        
        # 交换令牌
        tokens = await handler.exchange_code_for_tokens(code)
        access_token = tokens.get("access_token")
        
        if not access_token:
            raise OAuthError("No access token received")
        
        # 获取用户信息
        return await handler.get_user_info(access_token)
