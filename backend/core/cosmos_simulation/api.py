"""
API Module - API接口模块

提供宇宙模拟器的REST API接口。
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time


@dataclass
class APIResponse:
    """API响应"""
    success: bool
    data: Any
    message: str
    error: Optional[str] = None


class CosmosAPI:
    """
    宇宙模拟器API接口
    
    提供模拟器的主要功能接口。
    """
    
    def __init__(self, simulation=None, host: str = "localhost", port: int = 8000):
        """
        初始化API接口
        
        Args:
            simulation: 模拟器实例
            host: 主机地址
            port: 端口
        """
        self.simulation = simulation
        self.host = host
        self.port = port
        self._running = False
    
    def create_simulation(self, config: Optional[Dict] = None) -> APIResponse:
        """创建模拟"""
        from .cosmos_simulation import CosmosSimulation
        
        try:
            self.simulation = CosmosSimulation(config)
            return APIResponse(
                success=True,
                data={"simulation_id": id(self.simulation)},
                message="Simulation created successfully",
            )
        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                message="Failed to create simulation",
                error=str(e),
            )
    
    def set_initial_conditions(self, 
                                z: float = 1000, 
                                **kwargs) -> APIResponse:
        """设置初始条件"""
        if not self.simulation:
            return APIResponse(
                success=False,
                data=None,
                message="No simulation created",
                error="Create simulation first",
            )
        
        try:
            result = self.simulation.set_initial_conditions(z, **kwargs)
            return APIResponse(
                success=True,
                data=result,
                message=f"Initial conditions set at z={z}",
            )
        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                message="Failed to set initial conditions",
                error=str(e),
            )
    
    def evolve(self,
               end_redshift: float = 0,
               time_step: str = "1Gyr",
               **kwargs) -> APIResponse:
        """运行演化"""
        if not self.simulation:
            return APIResponse(
                success=False,
                data=None,
                message="No simulation created",
                error="Create simulation first",
            )
        
        try:
            result = self.simulation.evolve(end_redshift, time_step, **kwargs)
            return APIResponse(
                success=True,
                data=result,
                message=f"Evolved to z={end_redshift}",
            )
        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                message="Failed to evolve simulation",
                error=str(e),
            )
    
    def get_state(self,
                  redshift: float = 0,
                  scale: str = "galaxy") -> APIResponse:
        """获取宇宙状态"""
        if not self.simulation:
            return APIResponse(
                success=False,
                data=None,
                message="No simulation created",
                error="Create simulation first",
            )
        
        try:
            state = self.simulation.get_state(redshift, scale)
            return APIResponse(
                success=True,
                data=state,
                message=f"State at z={redshift}, scale={scale}",
            )
        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                message="Failed to get state",
                error=str(e),
            )
    
    def get_cosmology_params(self) -> APIResponse:
        """获取宇宙学参数"""
        if not self.simulation:
            return APIResponse(
                success=False,
                data=None,
                message="No simulation created",
                error="Create simulation first",
            )
        
        try:
            params = self.simulation.cosmology.get_cosmological_parameters()
            return APIResponse(
                success=True,
                data=params,
                message="Cosmological parameters",
            )
        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                message="Failed to get parameters",
                error=str(e),
            )
    
    def compute_distance(self, z: float) -> APIResponse:
        """计算宇宙学距离"""
        if not self.simulation:
            return APIResponse(
                success=False,
                data=None,
                message="No simulation created",
                error="Create simulation first",
            )
        
        try:
            cosmology = self.simulation.cosmology
            distances = {
                "z": z,
                "luminosity_distance": cosmology.compute_luminosity_distance(z),
                "angular_diameter_distance": cosmology.compute_angular_diameter_distance(z),
                "lookback_time": cosmology.compute_lookback_time(z),
                "distance_modulus": cosmology.compute_distance_modulus(z),
            }
            return APIResponse(
                success=True,
                data=distances,
                message=f"Distances at z={z}",
            )
        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                message="Failed to compute distances",
                error=str(e),
            )
    
    def create_galaxy(self,
                      mass: float,
                      z: float = 0) -> APIResponse:
        """创建星系"""
        if not self.simulation:
            return APIResponse(
                success=False,
                data=None,
                message="No simulation created",
                error="Create simulation first",
            )
        
        try:
            galaxy = self.simulation.galaxy_formation.create_collapsed_halo(mass, z)
            return APIResponse(
                success=True,
                data={
                    "halo_mass": galaxy.mass,
                    "halo_radius": galaxy.radius,
                    "concentration": galaxy.concentration,
                },
                message=f"Galaxy created with mass={mass}",
            )
        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                message="Failed to create galaxy",
                error=str(e),
            )
    
    def create_star(self, 
                    mass: float,
                    metallicity: float = 0.02) -> APIResponse:
        """创建恒星"""
        if not self.simulation:
            return APIResponse(
                success=False,
                data=None,
                message="No simulation created",
                error="Create simulation first",
            )
        
        try:
            star = self.simulation.stellar_evolution.create_star(mass, metallicity)
            return APIResponse(
                success=True,
                data={
                    "mass": star.mass,
                    "radius": star.radius,
                    "luminosity": star.luminosity,
                    "temperature": star.temperature,
                    "lifetime": star.lifetime,
                    "spectral_type": star.stellar_type.value,
                },
                message=f"Star created with mass={mass}",
            )
        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                message="Failed to create star",
                error=str(e),
            )
    
    def compute_cmb(self) -> APIResponse:
        """计算CMB功率谱"""
        if not self.simulation:
            return APIResponse(
                success=False,
                data=None,
                message="No simulation created",
                error="Create simulation first",
            )
        
        try:
            cmb = self.simulation.cosmology.compute_cmb_power_spectrum()
            return APIResponse(
                success=True,
                data={
                    "ell": cmb.ell.tolist(),
                    "C_ell_tt": cmb.C_ell_tt.tolist(),
                    "C_ell_ee": cmb.C_ell_ee.tolist(),
                },
                message="CMB power spectrum computed",
            )
        except Exception as e:
            return APIResponse(
                success=False,
                data=None,
                message="Failed to compute CMB",
                error=str(e),
            )
    
    def get_status(self) -> APIResponse:
        """获取模拟器状态"""
        status = {
            "running": self._running,
            "simulation_exists": self.simulation is not None,
            "host": self.host,
            "port": self.port,
        }
        
        if self.simulation:
            status["simulation"] = {
                "has_initial_conditions": hasattr(self.simulation, 'initial_conditions'),
                "has_evolution": hasattr(self.simulation, 'evolution'),
            }
        
        return APIResponse(
            success=True,
            data=status,
            message="API status",
        )
    
    def start_server(self):
        """启动API服务器"""
        self._running = True
        
        server = HTTPServer((self.host, self.port), self._create_handler())
        print(f"API server running at http://{self.host}:{self.port}")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self.stop_server()
    
    def stop_server(self):
        """停止API服务器"""
        self._running = False
    
    def _create_handler(self):
        """创建请求处理器"""
        api = self
        
        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = api.get_status()
                self.wfile.write(json.dumps(response.__dict__).encode())
            
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode())
                
                command = data.get('command')
                
                if command == 'create_simulation':
                    response = api.create_simulation(data.get('config'))
                elif command == 'set_initial_conditions':
                    response = api.set_initial_conditions(**data)
                elif command == 'evolve':
                    response = api.evolve(**data)
                elif command == 'get_state':
                    response = api.get_state(**data)
                elif command == 'create_galaxy':
                    response = api.create_galaxy(**data)
                elif command == 'create_star':
                    response = api.create_star(**data)
                else:
                    response = APIResponse(
                        success=False,
                        data=None,
                        message=f"Unknown command: {command}",
                    )
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response.__dict__).encode())
        
        return RequestHandler
    
    def run_in_background(self):
        """在后台线程中运行服务器"""
        thread = threading.Thread(target=self.start_server, daemon=True)
        thread.start()
        return thread


# 便捷函数
def quick_simulation(config: Optional[Dict] = None) -> "CosmosSimulation":
    """快速创建和运行简单模拟"""
    from .cosmos_simulation import CosmosSimulation
    
    sim = CosmosSimulation(config)
    sim.set_initial_conditions(100)
    sim.evolve(0, "1Gyr")
    return sim
