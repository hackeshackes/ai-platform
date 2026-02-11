"""
深空探测系统 API - Deep Space System API
==========================================

RESTful API 接口

作者: AI Platform Team
版本: 1.0.0
"""

import json
import logging
from typing import Dict, List, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

from .navigation import DeepSpaceNavigator, RouteMethod
from .planet_explorer import PlanetExplorer
from .seti_integration import SETIAnalyzer
from .communication import DeepSpaceCommunicator
from .config import get_config

logger = logging.getLogger(__name__)

# 全局实例
_navigator: Optional[DeepSpaceNavigator] = None
_explorer: Optional[PlanetExplorer] = None
_seti: Optional[SETIAnalyzer] = None
_communicator: Optional[DeepSpaceCommunicator] = None

def get_navigator() -> DeepSpaceNavigator:
    """获取导航器实例"""
    global _navigator
    if _navigator is None:
        _navigator = DeepSpaceNavigator()
    return _navigator

def get_explorer(planet: str) -> PlanetExplorer:
    """获取行星探测器实例"""
    return PlanetExplorer(planet)

def get_seti() -> SETIAnalyzer:
    """获取SETI分析器实例"""
    global _seti
    if _seti is None:
        _seti = SETIAnalyzer()
    return _seti

def get_communicator() -> DeepSpaceCommunicator:
    """获取通信器实例"""
    global _communicator
    if _communicator is None:
        _communicator = DeepSpaceCommunicator()
    return _communicator


class DeepSpaceAPIHandler(BaseHTTPRequestHandler):
    """API请求处理器"""
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        logger.info(f"[API] {args[0]}")
    
    def send_json_response(self, status: int, data: Dict):
        """发送JSON响应"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))
    
    def parse_body(self) -> Dict:
        """解析请求体"""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            return json.loads(body.decode('utf-8'))
        return {}
    
    def do_GET(self):
        """处理GET请求"""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        try:
            if path == '/api/v1/health':
                self._handle_health()
            elif path == '/api/v1/navigation/status':
                self._handle_navigation_status()
            elif path == '/api/v1/explorer/status':
                self._handle_explorer_status(query)
            elif path == '/api/v1/seti/status':
                self._handle_seti_status()
            elif path == '/api/v1/communication/status':
                self._handle_communication_status()
            elif path == '/api/v1/systems/summary':
                self._handle_systems_summary()
            else:
                self.send_json_response(404, {'error': 'Not found'})
        except Exception as e:
            logger.error(f"API错误: {e}")
            self.send_json_response(500, {'error': str(e)})
    
    def do_POST(self):
        """处理POST请求"""
        parsed = urlparse(self.path)
        path = parsed.path
        body = self.parse_body()
        
        try:
            if path == '/api/v1/navigation/route':
                self._handle_plan_route(body)
            elif path == '/api/v1/explorer/analyze':
                self._handle_analyze_surface(body)
            elif path == '/api/v1/explorer/resources':
                self._handle_scan_resources(body)
            elif path == '/api/v1/explorer/landing':
                self._handle_select_site(body)
            elif path == '/api/v1/explorer/samples':
                self._handle_collect_samples(body)
            elif path == '/api/v1/explorer/full':
                self._handle_full_exploration(body)
            elif path == '/api/v1/seti/scan':
                self._handle_seti_scan(body)
            elif path == '/api/v1/seti/anomaly':
                self._handle_detect_anomaly(body)
            elif path == '/api/v1/communication/send':
                self._handle_send_message(body)
            elif path == '/api/v1/communication/link':
                self._handle_configure_link(body)
            else:
                self.send_json_response(404, {'error': 'Not found'})
        except Exception as e:
            logger.error(f"API错误: {e}")
            self.send_json_response(500, {'error': str(e)})
    
    def _handle_health(self):
        """健康检查"""
        self.send_json_response(200, {
            'status': 'healthy',
            'timestamp': self._get_timestamp(),
            'version': '1.0.0'
        })
    
    def _handle_navigation_status(self):
        """获取导航状态"""
        navigator = get_navigator()
        status = navigator.getNavigationStatus()
        self.send_json_response(200, status)
    
    def _handle_explorer_status(self, query: Dict):
        """获取探测器状态"""
        planet = query.get('planet', ['mars'])[0]
        explorer = get_explorer(planet)
        self.send_json_response(200, {
            'planet': planet,
            'status': 'operational'
        })
    
    def _handle_seti_status(self):
        """获取SETI系统状态"""
        seti = get_seti()
        status = seti.get_system_status()
        self.send_json_response(200, status)
    
    def _handle_communication_status(self):
        """获取通信系统状态"""
        comm = get_communicator()
        status = comm.get_communication_status()
        self.send_json_response(200, status)
    
    def _handle_systems_summary(self):
        """获取所有系统摘要"""
        self.send_json_response(200, {
            'timestamp': self._get_timestamp(),
            'systems': {
                'navigation': get_navigator().getNavigationStatus(),
                'seti': get_seti().get_system_status(),
                'communication': get_communicator().get_communication_status()
            }
        })
    
    def _handle_plan_route(self, body: Dict):
        """规划航线"""
        origin = body.get('origin', 'earth')
        destination = body.get('destination', 'mars')
        method = body.get('method', 'optimal')
        
        navigator = get_navigator()
        route = navigator.plan_route(
            origin=origin,
            destination=destination,
            method=method
        )
        
        self.send_json_response(200, {
            'origin': route.origin,
            'destination': route.destination,
            'method': route.method.value,
            'total_distance_au': route.total_distance,
            'total_time_days': route.total_time,
            'fuel_required_kg': route.fuel_required,
            'confidence': route.confidence,
            'success_rate': route.estimated_success_rate,
            'maneuvers': route.maneuvers,
            'trajectory_points': len(route.trajectory)
        })
    
    def _handle_analyze_surface(self, body: Dict):
        """分析表面"""
        planet = body.get('planet', 'mars')
        explorer = get_explorer(planet)
        result = explorer.analyze_surface()
        self.send_json_response(200, result)
    
    def _handle_scan_resources(self, body: Dict):
        """扫描资源"""
        planet = body.get('planet', 'mars')
        explorer = get_explorer(planet)
        result = explorer.scan_resources()
        self.send_json_response(200, result)
    
    def _handle_select_site(self, body: Dict):
        """选择着陆点"""
        planet = body.get('planet', 'mars')
        explorer = get_explorer(planet)
        result = explorer.select_site()
        self.send_json_response(200, result)
    
    def _handle_collect_samples(self, body: Dict):
        """采集样本"""
        planet = body.get('planet', 'mars')
        location = body.get('location', (0.0, 0.0))
        count = body.get('count', 5)
        
        explorer = get_explorer(planet)
        result = explorer.collect_samples(
            location=tuple(location),
            count=count
        )
        self.send_json_response(200, result)
    
    def _handle_full_exploration(self, body: Dict):
        """完整探索"""
        planet = body.get('planet', 'mars')
        target_location = body.get('target_location', None)
        
        explorer = get_explorer(planet)
        result = explorer.full_exploration(
            target_location=tuple(target_location) if target_location else None
        )
        self.send_json_response(200, result)
    
    def _handle_seti_scan(self, body: Dict):
        """SETI扫描"""
        seti = get_seti()
        result = seti.scan(data=body.get('data'))
        self.send_json_response(200, result)
    
    def _handle_detect_anomaly(self, body: Dict):
        """检测异常"""
        seti = get_seti()
        result = seti.detect_anomaly()
        self.send_json_response(200, result)
    
    def _handle_send_message(self, body: Dict):
        """发送消息"""
        content = body.get('content', '')
        destination = body.get('destination', 'unknown')
        priority = body.get('priority', 3)
        
        comm = get_communicator()
        result = comm.send_message(
            content=content,
            destination=destination,
            priority=priority
        )
        self.send_json_response(200, result)
    
    def _handle_configure_link(self, body: Dict):
        """配置通信链路"""
        distance = body.get('distance', 1e11)
        data_rate = body.get('data_rate', 1e6)
        
        comm = get_communicator()
        result = comm.configure_link(
            target_distance=distance,
            required_data_rate=data_rate
        )
        self.send_json_response(200, result)
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"


class APIServer:
    """API服务器"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.running = False
    
    def start(self):
        """启动服务器"""
        self.server = HTTPServer((self.host, self.port), DeepSpaceAPIHandler)
        self.running = True
        
        logger.info(f"深空探测系统 API 服务器启动: http://{self.host}:{self.port}")
        
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """停止服务器"""
        self.running = False
        if self.server:
            self.server.shutdown()
            logger.info("API服务器已停止")
    
    def start_background(self):
        """后台启动"""
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread


# 便捷函数
def create_api_server(host: str = '0.0.0.0', port: int = 8080) -> APIServer:
    """创建API服务器"""
    return APIServer(host, port)


# 简单的CLI界面
def run_cli():
    """运行CLI界面"""
    import sys
    
    print("=" * 60)
    print("    深空探测系统 - Deep Space Exploration System")
    print("    v1.0.0")
    print("=" * 60)
    print()
    
    while True:
        command = input("deep_space> ").strip()
        
        if command in ['exit', 'quit', 'q']:
            print("再见！")
            break
        elif command == 'status':
            print("\n系统状态:")
            print(f"  导航系统: {get_navigator().getNavigationStatus()}")
            print(f"  SETI系统: {get_seti().get_system_status()}")
            print(f"  通信系统: {get_communicator().get_communication_status()}")
        elif command.startswith('route '):
            parts = command.split()[1:]
            if len(parts) >= 2:
                route = get_navigator().plan_route(parts[0], parts[1])
                print(f"\n航线规划结果:")
                print(f"  从 {route.origin} 到 {route.destination}")
                print(f"  距离: {route.total_distance:.2f} AU")
                print(f"  时间: {route.total_time:.1f} 天")
                print(f"  燃料: {route.fuel_required:.0f} kg")
                print(f"  成功率: {route.estimated_success_rate*100:.1f}%")
        elif command.startswith('explore '):
            planet = command.split()[1]
            print(f"\n开始对 {planet} 的完整探索...")
            explorer = get_explorer(planet)
            result = explorer.full_exploration()
            print(f"  发现地形类型: {result['terrain_analysis']['terrain_types_found']}")
            print(f"  检测到资源: {result['resource_survey']['resource_types']}")
            print(f"  推荐着陆点: {result['landing_site_selection']['recommended_site']['name'] if result['landing_site_selection']['recommended_site'] else '无'}")
        elif command.startswith('seti '):
            print("\n执行SETI扫描...")
            result = get_seti().scan()
            print(f"  扫描信号数: {result['scan_summary']['signals_scanned']}")
            print(f"  检测到异常: {result['scan_summary']['anomalies_detected']}")
            print(f"  发现模式: {result['scan_summary']['patterns_found']}")
            print(f"  评估文明: {result['scan_summary']['civilizations_assessed']}")
        elif command.startswith('send '):
            parts = command.split(' ', 1)
            if len(parts) == 2:
                result = get_communicator().send_message(parts[1])
                print(f"\n消息已发送:")
                print(f"  ID: {result['message_id']}")
                print(f"  通道: {result['channel']}")
                print(f"  延迟: {result['delay_info']['one_way_delay_s']:.1f} 秒")
        elif command == 'help':
            print("\n可用命令:")
            print("  status           - 显示系统状态")
            print("  route <from> <to> - 规划航线")
            print("  explore <planet>  - 完整探索行星")
            print("  seti              - 执行SETI扫描")
            print("  send <message>    - 发送消息")
            print("  help              - 显示帮助")
            print("  exit              - 退出")
        else:
            print(f"未知命令: {command}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='深空探测系统')
    parser.add_argument('--cli', action='store_true', help='运行CLI界面')
    parser.add_argument('--server', action='store_true', help='启动API服务器')
    parser.add_argument('--port', type=int, default=8080, help='API服务器端口')
    parser.add_argument('--host', default='0.0.0.0', help='API服务器主机')
    
    args = parser.parse_args()
    
    if args.cli:
        run_cli()
    elif args.server:
        server = create_api_server(args.host, args.port)
        server.start()
    else:
        run_cli()
