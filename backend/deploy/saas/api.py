#!/usr/bin/env python3
"""
API接口 - api.py

RESTful API接口:
- 一键部署
- 部署状态查询
- 扩容/缩容
- 监控数据
"""

import asyncio
import json
from typing import Dict, Any
from aiohttp import web
from datetime import datetime


class DeploymentAPI:
    """部署API"""
    
    def __init__(self, deployer, resource_manager, monitor, cdn_manager):
        self.deployer = deployer
        self.resource_manager = resource_manager
        self.monitor = monitor
        self.cdn_manager = cdn_manager
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        # 部署接口
        self.app.router.add_post('/api/v1/deploy/one-click', self.one_click_deploy)
        self.app.router.add_get('/api/v1/deploy/{id}/status', self.get_deploy_status)
        self.app.router.add_post('/api/v1/deploy/{id}/scale', self.scale_deployment)
        self.app.router.add_delete('/api/v1/deploy/{id}', self.delete_deployment)
        self.app.router.add_get('/api/v1/deploy/list', self.list_deployments)
        
        # 监控接口
        self.app.router.add_get('/api/v1/monitor/status', self.get_monitor_status)
        self.app.router.add_get('/api/v1/monitor/metrics', self.get_metrics)
        self.app.router.add_get('/api/v1/monitor/alerts', self.get_alerts)
        self.app.router.add_get('/api/v1/monitor/logs', self.get_logs)
        self.app.router.add_post('/api/v1/monitor/alerts/{id}/ack', self.acknowledge_alert)
        self.app.router.add_post('/api/v1/monitor/alerts/{id}/resolve', self.resolve_alert)
        
        # CDN接口
        self.app.router.add_post('/api/v1/cdn/configure', self.configure_cdn)
        self.app.router.add_post('/api/v1/cdn/purge', self.purge_cache)
        self.app.router.add_post('/api/v1/cdn/warm', self.warm_cache)
        self.app.router.add_get('/api/v1/cdn/status', self.get_cdn_status)
        self.app.router.add_get('/api/v1/cdn/analytics', self.get_cdn_analytics)
        
        # 健康检查
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/', self.index)
    
    async def one_click_deploy(self, request: web.Request) -> web.Response:
        """一键部署"""
        try:
            data = await request.json()
            
            # 验证必填字段
            required = ['type', 'name', 'config']
            for field in required:
                if field not in data:
                    return web.json_response({
                        "success": False,
                        "error": f"缺少必填字段: {field}"
                    }, status=400)
            
            # 导入部署配置类
            from deployer import DeployType, DeploymentConfig
            
            config = DeploymentConfig(
                name=data['name'],
                deploy_type=DeployType(data['type']),
                image=data['config'].get('image', 'nginx:latest'),
                replicas=data['config'].get('replicas', 1),
                cpu_limit=data['config'].get('cpu_limit', '1000m'),
                memory_limit=data['config'].get('memory_limit', '1Gi'),
                port=data['config'].get('port', 8080),
                domain=data['config'].get('domain'),
                ssl_enabled=data['config'].get('ssl_enabled', True),
                health_check_path=data['config'].get('health_check_path', '/health'),
                env_vars=data['config'].get('env_vars'),
                volumes=data['config'].get('volumes'),
                deploy_strategy=data['config'].get('deploy_strategy', 'rolling')
            )
            
            result = await self.deployer.deploy_one_click(config)
            
            return web.json_response(result, status=201 if result['success'] else 500)
            
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def get_deploy_status(self, request: web.Request) -> web.Response:
        """查询部署状态"""
        deployment_id = request.match_info['id']
        status = self.deployer.get_deployment_status(deployment_id)
        
        if not status:
            return web.json_response({
                "success": False,
                "error": "部署不存在"
            }, status=404)
        
        return web.json_response(status)
    
    async def scale_deployment(self, request: web.Request) -> web.Response:
        """扩容"""
        deployment_id = request.match_info['id']
        
        try:
            data = await request.json()
            replicas = data.get('replicas', 1)
            
            result = await self.deployer.scale_deployment(deployment_id, replicas)
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def delete_deployment(self, request: web.Request) -> web.Response:
        """删除部署"""
        deployment_id = request.match_info['id']
        result = await self.deployer.delete_deployment(deployment_id)
        return web.json_response(result)
    
    async def list_deployments(self, request: web.Request) -> web.Response:
        """列出所有部署"""
        deployments = {
            deployment_id: {
                "id": deployment_id,
                "name": info["name"],
                "type": info["type"],
                "status": info["status"]
            }
            for deployment_id, info in self.deployer.deployments.items()
        }
        
        return web.json_response({
            "count": len(deployments),
            "deployments": deployments
        })
    
    async def get_monitor_status(self, request: web.Request) -> web.Response:
        """获取监控状态"""
        deployment_id = request.query.get('deployment_id')
        status = self.monitor.get_status(deployment_id)
        return web.json_response(status)
    
    async def get_metrics(self, request: web.Request) -> web.Response:
        """获取指标数据"""
        deployment_id = request.query.get('deployment_id')
        metric_name = request.query.get('metric')
        limit = int(request.query.get('limit', 100))
        
        if metric_name:
            metrics = self.monitor.metrics_collector.get_metric(metric_name, limit)
            return web.json_response([m.__dict__ for m in metrics])
        
        dashboard = self.monitor.get_dashboard_data(deployment_id)
        return web.json_response(dashboard)
    
    async def get_alerts(self, request: web.Request) -> web.Response:
        """获取告警"""
        deployment_id = request.query.get('deployment_id')
        hours = int(request.query.get('hours', 24))
        
        alerts = self.monitor.alert_manager.get_alert_history(hours)
        return web.json_response([a.__dict__ for a in alerts])
    
    async def acknowledge_alert(self, request: web.Request) -> web.Response:
        """确认告警"""
        alert_id = request.match_info['id']
        success = self.monitor.alert_manager.acknowledge_alert(alert_id)
        
        return web.json_response({"success": success})
    
    async def resolve_alert(self, request: web.Request) -> web.Response:
        """解决告警"""
        alert_id = request.match_info['id']
        success = self.monitor.alert_manager.resolve_alert(alert_id)
        
        return web.json_response({"success": success})
    
    async def get_logs(self, request: web.Request) -> web.Response:
        """获取日志"""
        deployment_id = request.query.get('deployment_id')
        level = request.query.get('level')
        limit = int(request.query.get('limit', 100))
        
        logs = self.monitor.log_collector.get_logs(deployment_id, level, limit=limit)
        return web.json_response([l.__dict__ for l in logs])
    
    async def configure_cdn(self, request: web.Request) -> web.Response:
        """配置CDN"""
        try:
            data = await request.json()
            
            from cdn_manager import CDNProvider, CDNConfig
            
            config = CDNConfig(
                provider=CDNProvider(data.get('provider', 'cloudflare')),
                domain=data['domain'],
                origin_server=data['origin_server'],
                ssl_enabled=data.get('ssl_enabled', True)
            )
            
            result = await self.cdn_manager.configure_cdn(data['domain'], config)
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def purge_cache(self, request: web.Request) -> web.Response:
        """清除缓存"""
        try:
            data = await request.json()
            domain = data.get('domain')
            urls = data.get('urls')
            tags = data.get('tags')
            
            result = await self.cdn_manager.purge_cache(domain, urls, tags)
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def warm_cache(self, request: web.Request) -> web.Response:
        """预热缓存"""
        try:
            data = await request.json()
            domain = data.get('domain')
            urls = data.get('urls')
            
            result = await self.cdn_manager.warm_cache(domain, urls)
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def get_cdn_status(self, request: web.Request) -> web.Response:
        """获取CDN状态"""
        domain = request.query.get('domain')
        
        if domain:
            status = self.cdn_manager.get_cache_status(domain)
            rules = self.cdn_manager.get_cache_rules(domain)
            return web.json_response({"status": status, "rules": rules})
        
        return web.json_response({
            "cdns": self.cdn_manager.get_cdn_list()
        })
    
    async def get_cdn_analytics(self, request: web.Request) -> web.Response:
        """获取CDN分析"""
        domain = request.query.get('domain')
        
        if not domain:
            return web.json_response({
                "error": "缺少domain参数"
            }, status=400)
        
        analytics = await self.cdn_manager.get_analytics(domain)
        return web.json_response(analytics)
    
    async def health_check(self, request: web.Request) -> web.Response:
        """健康检查"""
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        })
    
    async def index(self, request: web.Request) -> web.Response:
        """API文档"""
        return web.json_response({
            "name": "AI Platform SaaS Deployment API",
            "version": "1.0.0",
            "endpoints": {
                "deploy": {
                    "POST /api/v1/deploy/one-click": "一键部署",
                    "GET /api/v1/deploy/{id}/status": "查询部署状态",
                    "POST /api/v1/deploy/{id}/scale": "扩容",
                    "DELETE /api/v1/deploy/{id}": "删除部署",
                    "GET /api/v1/deploy/list": "列出所有部署"
                },
                "monitor": {
                    "GET /api/v1/monitor/status": "监控状态",
                    "GET /api/v1/monitor/metrics": "指标数据",
                    "GET /api/v1/monitor/alerts": "告警列表",
                    "POST /api/v1/monitor/alerts/{id}/ack": "确认告警",
                    "POST /api/v1/monitor/alerts/{id}/resolve": "解决告警",
                    "GET /api/v1/monitor/logs": "日志查询"
                },
                "cdn": {
                    "POST /api/v1/cdn/configure": "配置CDN",
                    "POST /api/v1/cdn/purge": "清除缓存",
                    "POST /api/v1/cdn/warm": "预热缓存",
                    "GET /api/v1/cdn/status": "CDN状态",
                    "GET /api/v1/cdn/analytics": "CDN分析"
                },
                "health": {
                    "GET /health": "健康检查"
                }
            }
        })
    
    async def start(self, host: str = '0.0.0.0', port: int = 8080):
        """启动API服务"""
        print(f"[API] 启动API服务: http://{host}:{port}")
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        try:
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            await runner.cleanup()
    
    def create_app(self):
        """创建应用"""
        return self.app


# 独立运行
async def main():
    from deployer import Deployer
    from resource_manager import ResourceManager
    from monitor import Monitor
    from cdn_manager import CDNManager
    
    # 初始化组件
    deployer = Deployer()
    resource_manager = ResourceManager()
    monitor = Monitor()
    cdn_manager = CDNManager()
    
    # 创建API
    api = DeploymentAPI(deployer, resource_manager, monitor, cdn_manager)
    
    # 启动服务
    await api.start(port=8080)


if __name__ == "__main__":
    asyncio.run(main())
