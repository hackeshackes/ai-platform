#!/usr/bin/env python3
"""
CDN管理器 - cdn_manager.py

功能:
- 自动CDN配置
- 缓存更新
- 带宽优化
"""

import asyncio
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse


class CDNProvider(Enum):
    CLOUDFLARE = "cloudflare"
    AWS_CLOUDFRONT = "aws_cloudfront"
    AZURE_CDN = "azure_cdn"
    ALIYUN_CDN = "aliyun_cdn"
    CUSTOM = "custom"


class CacheType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    API = "api"
    MEDIA = "media"


@dataclass
class CacheRule:
    """缓存规则"""
    id: str
    name: str
    patterns: List[str]  # URL patterns
    cache_type: CacheType
    ttl: int  # 秒
    priority: int = 0
    headers: Dict[str, str] = field(default_factory=dict)
    query_string_handling: str = "include_all"  # include_all, exclude_all, whitelist
    query_whitelist: List[str] = field(default_factory=list)
    compression: bool = True
    serve_stale: bool = False
    
    def matches_url(self, url: str) -> bool:
        """检查URL是否匹配规则"""
        for pattern in self.patterns:
            if pattern in url or self._wildcard_match(url, pattern):
                return True
        return False
    
    def _wildcard_match(self, url: str, pattern: str) -> bool:
        """通配符匹配"""
        # 简单实现：支持 * 通配符
        import fnmatch
        return fnmatch.fnmatch(url, pattern)


@dataclass
class CDNConfig:
    """CDN配置"""
    provider: CDNProvider = CDNProvider.CLOUDFLARE
    domain: str = ""
    origin_server: str = ""
    ssl_enabled: bool = True
    ssl_type: str = "letsencrypt"  # letsencrypt, custom, shared
    geo_blocking: List[str] = field(default_factory=list)  # 阻止的国家代码
    ip_blacklist: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None  # 请求/秒
    cache_rules: List[CacheRule] = field(default_factory=list)
    optimization: Dict = field(default_factory=dict)
    
    def get_endpoint(self) -> str:
        """获取CDN访问地址"""
        protocol = "https" if self.ssl_enabled else "http"
        return f"{protocol}://cdn.{self.domain}"


@dataclass
class CachePurgeRequest:
    """缓存清除请求"""
    id: str
    urls: List[str]
    purge_type: str = "url"  # url, tag, wildcard
    tags: List[str] = field(default_factory=list)
    "pending"
    status: str = created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


class CDNManager:
    """CDN管理器"""
    
    def __init__(self, config: Dict = None):
 config or {}
               self.config = self.cdn_configs: Dict[str, CDNConfig] = {}
        self.purge_requests: Dict[str, CachePurgeRequest] = {}
        self.cache_stats:] = {}
        Dict[str, Dict self._running = False
        
        # 初始化默认缓存规则
        self._setup_default_cache_rules()
    
    def _setup_default_cache_rules(self):
        """设置默认缓存规则"""
        self.default_rules = [
            CacheRule(
                id="static-assets",
                name="静态资源",
                patterns=["*.css", "*.js", "*.ico", "*.png", "*.jpg", "*.svg"],
                cache_type=CacheType.STATIC,
                ttl=86400,  # 1天
                priority=10
            ),
            CacheRule(
                id="api-responses",
                name="API响应",
                patterns=["/api/*"],
                cache_type=CacheType.API,
                 # 不缓存 ttl=0,
                priority=100
            ),
            CacheRule(
                id="media-files",
                name="媒体文件",
                patterns=["*.mp4", "*.mp3", "*.pdf", "*.zip"],
                cache_type=CacheType.MEDIA,
                ttl=604800,  # 7天
                priority=5
            ),
            CacheRule(
                id="html-pages",
                name="HTML页面",
                patterns=["*.html", "*.htm"],
                cache_type=CacheType.DYNAMIC,
                ttl=3600,  # 1小时
                priority=20,
                serve_stale=True
            )
        ]
    
    def add_cdn_config(self, domain: str, config: CDNConfig):
        """添加CDN配置"""
        self.cdn_configs[domain] = config
        self.cache_stats[domain] = {
            "hits": 0,
            "misses": 0,
            "bandwidth_bytes": 0,
            "requests": 0,
            "cache_ratio": 0.0
        }
    
    async def configure_cdn(self, domain: str, config: CDNConfig) -> Dict:
        """配置CDN"""
        print(f"[CDNManager] 配置CDN: {domain}")
        
        self.add_cdn_config(domain, config)
        
        # 应用缓存规则
        for rule in self.default_rules:
            await self._apply_cache_rule(domain, rule)
        
        # 配置SSL
        if config.ssl_enabled:
            await self._configure_ssl(domain)
        
        # 配置源站
        await self._configure_origin(domain, config.origin_server)
        
        return {
            "success": True,
            "domain": domain,
            "endpoint": config.get_endpoint()
        }
    
    async def _apply_cache_rule(self, domain: str, rule: CacheRule):
        """应用缓存规则"""
        print(f"[CDNManager] 应用缓存规则: {rule.name}")
        # 模拟API调用
        await asyncio.sleep(0.1)
        return True
    
    async def _configure_ssl(self, domain: str):
        """配置SSL证书"""
        print(f"[CDNManager] 配置SSL: {domain}")
        await asyncio.sleep(0.2)
        return True
    
    async def _configure_origin(self, domain: str, origin: str):
        """配置源站"""
        print(f"[CDNManager] 配置源站: {origin}")
        await asyncio.sleep(0.1)
        return True
    
    async def create_cache_rule(
        self,
        domain: str,
        name: str,
        patterns: List[str],
        cache_type: CacheType,
        ttl: int,
        **kwargs
    ) -> Dict:
        """创建缓存规则"""
        import uuid
        
        rule = CacheRule(
            id=str(uuid.uuid4())[:8],
            name=name,
            patterns=patterns,
            cache_type=cache_type,
            ttl=ttl,
            **kwargs
        )
        
        if domain not in self.cdn_configs:
            return {"success": False, "error": "CDN配置不存在"}
        
        self.cdn_configs[domain].cache_rules.append(rule)
        
        await self._apply_cache_rule(domain, rule)
        
        return {
            "success": True,
            "rule_id": rule.id
        }
    
    async def purge_cache(
        self,
        domain: str,
        urls: List[str] = None,
        tags: List[str] = None,
        purge_type: str = "url"
    ) -> Dict:
        """清除缓存"""
        import uuid
        
        purge_id = str(uuid.uuid4())[:8]
        
        request = CachePurgeRequest(
            id=purge_id,
            urls=urls or [],
            purge_type=purge_type,
            tags=tags or []
        )
        
        self.purge_requests[purge_id] = request
        
        print(f"[CDNManager] 清除缓存: {purge_id}")
        
        # 模拟异步清除
        await asyncio.sleep(0.5)
        
        request.status = "completed"
        request.completed_at = datetime.now().isoformat()
        
        # 更新缓存统计
        if domain in self.cache_stats:
            self.cache_stats[domain]["hits"] = 0
            self.cache_stats[domain]["misses"] = 0
        
        return {
            "success": True,
            "purge_id": purge_id,
            "urls_purged": len(urls) if urls else 0
        }
    
    async def purge_all(self, domain: str) -> Dict:
        """清除所有缓存"""
        return await self.purge_cache(domain, purge_type="wildcard")
    
    async def warm_cache(self, domain: str, urls: List[str]) -> Dict:
        """预热缓存"""
        print(f"[CDNManager] 预热缓存: {len(urls)} URLs")
        
        # 模拟预热
        await asyncio.sleep(0.3)
        
        # 更新统计
        if domain in self.cache_stats:
            self.cache_stats[domain]["hits"] += len(urls)
        
        return {
            "success": True,
            "urls_warmed": len(urls)
        }
    
    def get_cache_rules(self, domain: str) -> List[Dict]:
        """获取缓存规则"""
        if domain not in self.cdn_configs:
            return []
        
        return [rule.__dict__ for rule in self.cdn_configs[domain].cache_rules]
    
    def get_cache_status(self, domain: str) -> Dict:
        """获取缓存状态"""
        stats = self.cache_stats.get(domain, {})
        
        total = stats.get("hits", 0) + stats.get("misses", 0)
        cache_ratio = stats.get("hits", 0) / total * 100 if total > 0 else 0
        
        return {
            "domain": domain,
            "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0),
            "cache_ratio": f"{cache_ratio:.2f}%",
            "bandwidth_bytes": stats.get("bandwidth_bytes", 0),
            "bandwidth_mb": round(stats.get("bandwidth_bytes", 0) / 1024 / 1024, 2),
            "requests": stats.get("requests", 0)
        }
    
    def get_purge_status(self, purge_id: str) -> Optional[Dict]:
        """获取清除状态"""
        if purge_id not in self.purge_requests:
            return None
        
        request = self.purge_requests[purge_id]
        return request.__dict__
    
    async def optimize_bandwidth(self, domain: str) -> Dict:
        """优化带宽"""
        print(f"[CDNManager] 优化带宽: {domain}")
        
        config = self.cdn_configs.get(domain)
        if not config:
            return {"success": False, "error": "CDN配置不存在"}
        
        optimization = {
            "compression_enabled": True,
            "minify_js": True,
            "minify_css": True,
            "minify_html": True,
            "image_optimization": {
                "format": "auto",
                "quality": 85,
                "resize": True
            },
            "http2_push": True,
            "tls_1_3": True
        }
        
        config.optimization = optimization
        
        return {
            "success": True,
            "optimization": optimization
        }
    
    async def get_analytics(
        self,
        domain: str,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> Dict:
        """获取CDN分析数据"""
        # 模拟数据
        return {
            "domain": domain,
            "period": {
                "start": start_time or datetime.now().isoformat(),
                "end": end_time or datetime.now().isoformat()
            },
            "total_requests": 1000000,
            "cache_hit_ratio": 0.95,
            "bandwidth_gb": 50.5,
            "top_urls": [
                {"url": "/static/app.js", "requests": 100000},
                {"url": "/static/style.css", "requests": 80000},
                {"url": "/api/health", "requests": 50000}
            ],
            "status_codes": {
                "200": 950000,
                "304": 30000,
                "404": 10000,
                "500": 10000
            },
            "geography": {
                "asia": 0.6,
                "europe": 0.25,
                "americas": 0.15
            }
        }
    
    def get_cdn_list(self) -> List[Dict]:
        """获取所有CDN配置"""
        return [
            {
                "domain": domain,
                "endpoint": config.get_endpoint(),
                "ssl_enabled": config.ssl_enabled,
                "cache_rules_count": len(config.cache_rules)
            }
            for domain, config in self.cdn_configs.items()
        ]


# CLI入口
if __name__ == "__main__":
    import json
    
    async def main():
        cdn_manager = CDNManager()
        
        # 配置CDN
        config = CDNConfig(
            provider=CDNProvider.CLOUDFLARE,
            domain="example.com",
            origin_server="origin.example.com",
            ssl_enabled=True
        )
        
        result = await cdn_manager.configure_cdn("example.com", config)
        print(json.dumps(result, indent=2))
        
        # 获取缓存状态
        status = cdn_manager.get_cache_status("example.com")
        print(json.dumps(status, indent=2))
    
    asyncio.run(main())
