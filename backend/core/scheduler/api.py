"""
智能调度系统 - API接口

提供RESTful API接口
"""

from typing import Dict, List, Optional
from datetime import datetime
from flask import Flask, jsonify, request
import logging

logger = logging.getLogger(__name__)


def create_scheduler_api(resource_optimizer, load_balancer, auto_scaler, cost_optimizer):
    """创建调度系统API"""
    
    app = Flask(__name__)
    
    # ==================== 资源优化 API ====================
    
    @app.route('/api/v1/optimizer/allocate', methods=['POST'])
    def allocate_resources():
        """资源分配"""
        try:
            data = request.get_json()
            workloads = data.get('workloads', [])
            constraints = data.get('constraints', {})
            
            result = resource_optimizer.optimize(workloads, constraints)
            
            return jsonify({
                "status": "success",
                "allocation": result,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"资源分配失败: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/api/v1/optimizer/report', methods=['GET'])
    def get_optimization_report():
        """获取优化报告"""
        report = resource_optimizer.get_optimization_report()
        return jsonify({
            "status": "success",
            "report": report,
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/api/v1/optimizer/gpu-schedule', methods=['POST'])
    def optimize_gpu():
        """GPU调度"""
        try:
            data = request.get_json()
            gpu_workloads = data.get('workloads', [])
            
            result = resource_optimizer.optimize_gpu_schedule(gpu_workloads)
            
            return jsonify({
                "status": "success",
                "allocation": result,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"GPU调度失败: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    # ==================== 负载均衡 API ====================
    
    @app.route('/api/v1/loadbalancer/route', methods=['POST'])
    def route_request():
        """路由请求"""
        try:
            data = request.get_json()
            
            backend_id, headers = load_balancer.route_request(
                request_id=data.get('request_id', ''),
                client_ip=data.get('client_ip', ''),
                path=data.get('path', ''),
                headers=data.get('headers', {}),
                session_id=data.get('session_id'),
                user_id=data.get('user_id')
            )
            
            return jsonify({
                "status": "success",
                "backend_id": backend_id,
                "response_headers": headers,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"路由请求失败: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/api/v1/loadbalancer/stats', methods=['GET'])
    def get_lb_stats():
        """获取负载均衡统计"""
        stats = load_balancer.get_load_balancer_stats()
        return jsonify({
            "status": "success",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/api/v1/loadbalancer/backend/register', methods=['POST'])
    def register_backend():
        """注册后端"""
        try:
            data = request.get_json()
            
            from load_balancer import BackendServer
            backend = BackendServer(
                id=data['id'],
                host=data['host'],
                port=data['port'],
                weight=data.get('weight', 100),
                max_connections=data.get('max_connections', 1000)
            )
            
            load_balancer.register_backend(backend)
            
            return jsonify({
                "status": "success",
                "message": f"后端 {data['id']} 注册成功",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"后端注册失败: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/api/v1/loadbalancer/health', methods=['POST'])
    def health_check():
        """健康检查"""
        try:
            data = request.get_json()
            
            backend_id = data.get('backend_id')
            is_healthy = data.get('is_healthy', True)
            response_time = data.get('response_time', 0)
            
            load_balancer.health_check(backend_id, is_healthy, response_time)
            
            return jsonify({
                "status": "success",
                "message": f"健康检查完成: {backend_id}",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/api/v1/loadbalancer/gray', methods=['POST'])
    def add_gray_release():
        """添加灰度发布"""
        try:
            data = request.get_json()
            
            from load_balancer import GrayReleaseRule
            rule = GrayReleaseRule(
                name=data['name'],
                backend_id=data['backend_id'],
                traffic_percentage=data.get('traffic_percentage', 10),
                conditions=data.get('conditions', [])
            )
            
            load_balancer.add_gray_release_rule(rule)
            
            return jsonify({
                "status": "success",
                "message": f"灰度规则 {data['name']} 已添加",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"灰度发布失败: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    # ==================== 自动伸缩 API ====================
    
    @app.route('/api/v1/autoscaler/decide', methods=['POST'])
    def scaling_decide():
        """伸缩决策"""
        try:
            data = request.get_json()
            
            decision = auto_scaler.decide(
                current_metrics=data.get('metrics', {}),
                target_response_time=data.get('target_response_time'),
                predicted_load=data.get('predicted_load')
            )
            
            return jsonify({
                "status": "success",
                "decision": {
                    "action": decision.action.value,
                    "type": decision.scaling_type.value,
                    "reason": decision.reason,
                    "details": decision.details,
                    "confidence": decision.confidence
                },
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"伸缩决策失败: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/api/v1/autoscaler/recommendations', methods=['GET'])
    def get_scaling_recommendations():
        """获取伸缩推荐"""
        recommendations = auto_scaler.get_scaling_recommendations()
        return jsonify({
            "status": "success",
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/api/v1/autoscaler/cost-optimized', methods=['POST'])
    def cost_optimized_scaling():
        """成本优化伸缩"""
        try:
            data = request.get_json()
            
            result = auto_scaler.cost_optimized_scaling(
                current_metrics=data.get('metrics', {}),
                spot_instances_available=data.get('spot_available', True),
                reserved_discount=data.get('reserved_discount', 0.3)
            )
            
            return jsonify({
                "status": "success",
                "optimization": result,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"成本优化失败: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/api/v1/autoscaler/history', methods=['GET'])
    def get_scaling_history():
        """获取伸缩历史"""
        limit = request.args.get('limit', 100, type=int)
        history = auto_scaler.get_scaling_history(limit)
        return jsonify({
            "status": "success",
            "history": history,
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/api/v1/autoscaler/policy', methods=['POST'])
    def add_policy():
        """添加伸缩策略"""
        try:
            data = request.get_json()
            
            from auto_scaler import ScalingPolicy, ScalingType
            policy = ScalingPolicy(
                name=data['name'],
                metric_name=data['metric_name'],
                threshold_high=data['threshold_high'],
                threshold_low=data['threshold_low'],
                cooldown_seconds=data.get('cooldown_seconds', 300),
                scaling_type=ScalingType(data.get('scaling_type', 'horizontal')),
                step_size=data.get('step_size', 1)
            )
            
            auto_scaler.add_policy(policy)
            
            return jsonify({
                "status": "success",
                "message": f"策略 {data['name']} 已添加",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"添加策略失败: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    # ==================== 成本优化 API ====================
    
    @app.route('/api/v1/cost/analyze', methods=['POST'])
    def analyze_cost():
        """成本分析"""
        try:
            data = request.get_json()
            
            result = cost_optimizer.analyze(
                usage_patterns=data.get('usage_patterns', []),
                reserved_vs_spot=data.get('reserved_vs_spot', True)
            )
            
            return jsonify({
                "status": "success",
                "analysis": result,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"成本分析失败: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/api/v1/cost/report', methods=['GET'])
    def get_cost_report():
        """获取成本报告"""
        period = request.args.get('period', 'monthly')
        report = cost_optimizer.get_cost_report(period)
        return jsonify({
            "status": "success",
            "report": report,
            "timestamp": datetime.now().isoformat()
        })
    
    @app.route('/api/v1/cost/savings', methods=['POST'])
    def calculate_savings():
        """计算节省"""
        try:
            data = request.get_json()
            
            from cost_optimizer import InstanceType
            result = cost_optimizer.calculate_savings_vs_ondemand(
                instance_count=data['instance_count'],
                instance_size=data['instance_size'],
                instance_type=InstanceType(data['instance_type']),
                hours_per_month=data.get('hours_per_month', 720)
            )
            
            return jsonify({
                "status": "success",
                "savings": result,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"计算节省失败: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/api/v1/cost/bid-strategy', methods=['POST'])
    def optimize_bid():
        """优化竞价策略"""
        try:
            data = request.get_json()
            
            result = cost_optimizer.optimize_bid_strategy(
                base_price=data['base_price'],
                historical_prices=data.get('historical_prices', []),
                reliability_requirement=data.get('reliability_requirement', 0.95)
            )
            
            return jsonify({
                "status": "success",
                "strategy": result,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"优化竞价失败: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    # ==================== 健康检查 ====================
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """API健康检查"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        })
    
    return app


if __name__ == '__main__':
    app = create_scheduler_api(None, None, None, None)
    app.run(host='0.0.0.0', port=8080, debug=True)
