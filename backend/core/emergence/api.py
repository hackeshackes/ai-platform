"""
Emergence API - API接口
提供涌现引擎的RESTful API
"""

from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify
from .capability_detector import CapabilityDetector
from .self_organization import SelfOrganization
from .creative_generator import CreativeGenerator
from .emergence_monitor import EmergenceMonitor


class EmergenceAPI:
    """
    涌现引擎API接口
    提供RESTful API访问所有核心功能
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.app = Flask(__name__)
        
        self.detector = CapabilityDetector()
        self.organizer = SelfOrganization()
        self.generator = CreativeGenerator()
        self.monitor = EmergenceMonitor()
        
        self._setup_routes()
        self._storage: Dict[str, Any] = {}
        
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.route('/api/capability/detect', methods=['POST'])
        def detect_capability():
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            model = data.get('model')
            interaction_data = data.get('interaction_data', {})
            
            capability = self.detector.detect(model, interaction_data)
            
            if capability:
                return jsonify({
                    'success': True,
                    'capability': {
                        'name': capability.name,
                        'type': capability.emergence_type.value,
                        'level': capability.level.value,
                        'confidence': capability.confidence,
                        'signature': capability.signature,
                        'behaviors': capability.behaviors,
                        'boundaries': capability.boundaries
                    }
                })
            return jsonify({'success': False, 'message': 'No new capability detected'})
        
        @self.app.route('/api/capability/batch-detect', methods=['POST'])
        def batch_detect_capability():
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            model = data.get('model')
            interactions = data.get('interactions', [])
            
            capabilities = self.detector.batch_detect(model, interactions)
            
            return jsonify({
                'success': True,
                'count': len(capabilities),
                'capabilities': [
                    {'name': c.name, 'type': c.emergence_type.value, 'confidence': c.confidence}
                    for c in capabilities
                ]
            })
        
        @self.app.route('/api/organize', methods=['POST'])
        def organize_network():
            structure = self.organizer.organize()
            
            return jsonify({
                'success': True,
                'structure': {
                    'type': structure.network_type.value,
                    'layers': list(structure.layers.keys()),
                    'neuron_count': len(structure.neurons),
                    'connection_count': len(structure.connections),
                    'metadata': structure.metadata
                }
            })
        
        @self.app.route('/api/organize/statistics', methods=['GET'])
        def get_organize_stats():
            return jsonify(self.organizer.get_statistics())
        
        @self.app.route('/api/creative/solve', methods=['POST'])
        def solve_problem():
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            problem = data.get('problem', {})
            constraints = data.get('constraints', {})
            
            solutions = self.generator.solve(problem, constraints)
            
            return jsonify({
                'success': True,
                'count': len(solutions),
                'solutions': [
                    {
                        'id': s.solution_id,
                        'description': s.description,
                        'type': s.innovation_type.value,
                        'confidence': s.confidence,
                        'novelty': s.novelty_score,
                        'applicability': s.applicability,
                        'steps': s.steps
                    }
                    for s in solutions
                ]
            })
        
        @self.app.route('/api/creative/strategy', methods=['POST'])
        def discover_strategy():
            data = request.get_json() or {}
            context = data.get('context', {})
            
            strategy = self.generator.discover_strategy(context)
            return jsonify({'success': True, 'strategy': strategy})
        
        @self.app.route('/api/creative/art', methods=['POST'])
        def create_art():
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            theme = data.get('theme', 'abstract')
            style = data.get('style', 'modern')
            
            art = self.generator.create_artistic_content(theme, style)
            return jsonify({'success': True, 'art': art})
        
        @self.app.route('/api/monitor/track', methods=['POST'])
        def track_behavior():
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            event = self.monitor.track(data)
            
            if event:
                return jsonify({
                    'success': True,
                    'event': event.to_dict()
                })
            return jsonify({'success': False, 'message': 'No emergence detected'})
        
        @self.app.route('/api/monitor/events', methods=['GET'])
        def get_events():
            limit = request.args.get('limit', 100, type=int)
            return jsonify({'success': True, 'events': self.monitor.get_event_history(limit)})
        
        @self.app.route('/api/monitor/safety', methods=['GET'])
        def get_safety_summary():
            return jsonify({'success': True, 'safety': self.monitor.get_safety_summary()})
        
        @self.app.route('/api/monitor/report', methods=['GET'])
        def generate_report():
            return jsonify({'success': True, 'report': self.monitor.generate_report()})
        
        @self.app.route('/api/capability/<name>/status', methods=['GET'])
        def get_capability_status(name):
            status = self.monitor.get_capability_status(name)
            if status:
                return jsonify({'success': True, 'status': status})
            return jsonify({'success': False, 'message': 'Capability not found'}), 404
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy', 'components': ['detector', 'organizer', 'generator', 'monitor']})
    
    def run(self, host: str = '0.0.0.0', port: int = 8080, debug: bool = False):
        """运行API服务器"""
        self.app.run(host=host, port=port, debug=debug)
    
    def get_flask_app(self):
        """获取Flask应用"""
        return self.app
