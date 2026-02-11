"""
气候模型API接口
提供RESTful API服务
"""

from typing import Dict, List, Optional
from datetime import datetime
import json

from .climate_model import ClimateModel


class ClimateModelAPI:
    """
    气候模型API接口类
    
    提供以下功能:
    - 创建和管理气候模型实例
    - 运行模拟
    - 获取预测结果
    - 查询系统状态
    """
    
    def __init__(self, resolution: str = "1km"):
        """
        初始化API
        
        Args:
            resolution: 模拟分辨率
        """
        self.resolution = resolution
        self.model = None
        self.status = "ready"
        
    def create_model(self, resolution: Optional[str] = None,
                     grid_size: int = 360,
                     lat_bands: int = 180) -> Dict:
        """
        创建气候模型实例
        
        Returns:
            模型信息
        """
        if resolution:
            self.resolution = resolution
            
        self.model = ClimateModel(
            resolution=self.resolution,
            grid_size=grid_size,
            lat_bands=lat_bands
        )
        
        self.status = "initialized"
        
        return {
            'status': 'success',
            'message': '气候模型已创建',
            'resolution': self.resolution,
            'grid_size': grid_size,
            'lat_bands': lat_bands
        }
        
    def set_scenario(self, scenario: str) -> Dict:
        """
               Args:
            设置气候情景
        
 scenario: RCP2.6, RCP4.5, RCP6.0, RCP8.5
            
        Returns:
            设置结果
        """
        if not self.model:
            return {'status': 'error', 'message': '模型未创建'}
            
        try:
            self.model.set_scenario(scenario)
            return {
                'status': 'success',
                'message': f'情景已设置为: {scenario}',
                'scenario': scenario
            }
        except ValueError as e:
            return {'status': 'error', 'message': str(e)}
            
    def run_simulation(self, start_year: int = 2020,
                       end_year: int = 2100,
                       verbose: bool = False) -> Dict:
        """
        运行气候模拟
        
        Args:
            start_year: 起始年份
            end_year: 结束年份
            verbose: 是否显示详细信息
            
        Returns:
            模拟结果摘要
        """
        if not self.model:
            return {'status': 'error', 'message': '模型未创建'}
            
        if self.status == "running":
            return {'status': 'error', 'message': '模拟已在运行中'}
            
        self.status = "running"
        
        results = self.model.run(
            start_year=start_year,
            end_year=end_year,
            verbose=verbose
        )
        
        self.status = "completed"
        
        # 提取关键结果
        summary = {
            'status': 'success',
            'scenario': results['scenario'],
            'period': f"{start_year} - {end_year}",
            'years_simulated': end_year - start_year + 1,
            'final_temperature': round(results['history']['temperature'][-1], 2),
            'final_co2': round(results['history']['co2'][-1], 1),
            'final_sea_level': round(results['history']['sea_level'][-1], 3),
            'temperature_change': round(
                results['history']['temperature'][-1] - results['history']['temperature'][0], 
                2
            )
        }
        
        return summary
        
    def get_prediction(self, variable: str = "temperature",
                        region: str = "global",
                        year: int = 2100) -> Dict:
        """
        获取气候预测
        
        Args:
            variable: 变量名
            region: 区域
            year: 目标年份
            
        Returns:
            预测数据
        """
        if not self.model:
            return {'status': 'error', 'message': '模型未创建'}
            
        prediction = self.model.get_prediction(
            variable=variable,
            region=region,
            year=year
        )
        
        return prediction
        
    def get_state(self) -> Dict:
        """获取系统状态"""
        if not self.model:
            return {
                'status': 'not_initialized',
                'message': '模型尚未创建'
            }
            
        return self.model.get_state()
        
    def get_feedback_analysis(self) -> Dict:
        """获取反馈机制分析"""
        if not self.model:
            return {'status': 'error', 'message': '模型未创建'}
            
        return self.model.get_feedback_analysis()
        
    def export_results(self, filepath: str) -> Dict:
        """导出模拟结果"""
        if not self.model:
            return {'status': 'error', 'message': '模型未创建'}
            
        try:
            result = self.model.export_results(filepath)
            return {
                'status': 'success',
                'message': result,
                'filepath': filepath
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def list_scenarios(self) -> Dict:
        """列出可用情景"""
        from . import config
        
        return {
            'scenarios': config.SCENARIOS,
            'default': 'RCP8.5'
        }
        
    def get_variables(self) -> Dict:
        """列出可用变量"""
        return {
            'variables': [
                {'name': 'temperature', 'unit': '°C', 'description': '全球平均气温'},
                {'name': 'co2', 'unit': 'ppm', 'description': '大气CO2浓度'},
                {'name': 'sea_level', 'unit': 'm', 'description': '全球平均海平面'},
                {'name': 'precipitation', 'unit': 'mm/day', 'description': '全球平均降水量'}
            ]
        }


# Flask API 路由示例 (需要Flask框架)
def create_flask_app():
    """
    创建Flask应用
    
    使用示例:
        from climate_model.api import create_flask_app
        app = create_flask_app()
        app.run()
    """
    try:
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        api = ClimateModelAPI()
        
        @app.route('/api/climate/model', methods=['POST'])
        def create():
            data = request.json or {}
            return jsonify(api.create_model(
                data.get('resolution'),
                data.get('grid_size', 360),
                data.get('lat_bands', 180)
            ))
            
        @app.route('/api/climate/scenario', methods=['POST'])
        def set_scenario():
            data = request.json
            return jsonify(api.set_scenario(data.get('scenario', 'RCP8.5')))
            
        @app.route('/api/climate/run', methods=['POST'])
        def run():
            data = request.json or {}
            return jsonify(api.run_simulation(
                data.get('start_year', 2020),
                data.get('end_year', 2100),
                data.get('verbose', False)
            ))
            
        @app.route('/api/climate/prediction', methods=['GET'])
        def prediction():
            return jsonify(api.get_prediction(
                request.args.get('variable', 'temperature'),
                request.args.get('region', 'global'),
                int(request.args.get('year', 2100))
            ))
            
        @app.route('/api/climate/state', methods=['GET'])
        def state():
            return jsonify(api.get_state())
            
        @app.route('/api/climate/feedback', methods=['GET'])
        def feedback():
            return jsonify(api.get_feedback_analysis())
            
        @app.route('/api/climate/scenarios', methods=['GET'])
        def scenarios():
            return jsonify(api.list_scenarios())
            
        @app.route('/api/climate/variables', methods=['GET'])
        def variables():
            return jsonify(api.get_variables())
            
        return app
        
    except ImportError:
        print("Flask未安装，无法创建Web服务")
        return None
