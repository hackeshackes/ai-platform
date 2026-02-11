"""
Cross-Domain Reasoning API
跨域推理系统 API 接口
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from flask import Flask, request, jsonify

from .knowledge_fusion import KnowledgeFusion, KnowledgeSource
from .transfer_learning import TransferLearning, DomainSpec
from .analogical_reasoning import AnalogicalReasoner
from .unified_reasoner import UnifiedReasoner, ReasoningContext

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """创建Flask应用"""
    app = Flask(__name__)
    
    # 初始化核心模块
    fusion_engine = KnowledgeFusion()
    transfer_engine = TransferLearning()
    analogy_engine = AnalogicalReasoner()
    reasoner = UnifiedReasoner()
    
    # ============== 知识融合 API ==============
    
    @app.route('/api/v1/fusion/add_source', methods=['POST'])
    def add_knowledge_source():
        """添加知识源"""
        data = request.json
        
        source = KnowledgeSource(
            source_id=data.get('source_id'),
            name=data.get('name'),
            domain=data.get('domain'),
            knowledge_type=data.get('knowledge_type', 'facts'),
            data=data.get('data', {}),
            metadata=data.get('metadata', {}),
            confidence=data.get('confidence', 0.8)
        )
        
        fusion_engine.add_knowledge_source(source)
        
        return jsonify({
            "success": True,
            "message": f"Knowledge source {source.name} added successfully"
        })
    
    @app.route('/api/v1/fusion/fuse', methods=['POST'])
    def fuse_knowledge():
        """融合知识"""
        data = request.json
        sources_data = data.get('sources', [])
        fusion_type = data.get('fusion_type', 'comprehensive')
        
        sources = []
        for s in sources_data:
            sources.append(KnowledgeSource(
                source_id=s.get('source_id'),
                name=s.get('name'),
                domain=s.get('domain'),
                knowledge_type=s.get('knowledge_type', 'facts'),
                data=s.get('data', {}),
                metadata=s.get('metadata', {}),
                confidence=s.get('confidence', 0.8)
            ))
        
        result = fusion_engine.fuse(sources, fusion_type)
        
        return jsonify({
            "success": result.success,
            "result": result.to_dict()
        })
    
    @app.route('/api/v1/fusion/stats', methods=['GET'])
    def get_fusion_stats():
        """获取融合统计"""
        return jsonify(fusion_engine.get_statistics())
    
    # ============== 迁移学习 API ==============
    
    @app.route('/api/v1/transfer/add_domain', methods=['POST'])
    def add_domain():
        """添加领域"""
        data = request.json
        
        domain = DomainSpec(
            domain_id=data.get('domain_id'),
            name=data.get('name'),
            domain_type=data.get('domain_type', 'intermediate'),
            features=data.get('features', {}),
            examples=data.get('examples', [])
        )
        
        transfer_engine.add_domain(domain)
        
        return jsonify({
            "success": True,
            "message": f"Domain {domain.name} added successfully"
        })
    
    @app.route('/api/v1/transfer/adapt', methods=['POST'])
    def adapt_domain():
        """执行域适应"""
        data = request.json
        
        result = transfer_engine.adapt(
            source_domain=data.get('source_domain'),
            target_domain=data.get('target_domain'),
            model=data.get('model'),
            method=data.get('method'),
            num_iterations=data.get('num_iterations', 100)
        )
        
        return jsonify({
            "success": result.success,
            "result": result.to_dict()
        })
    
    @app.route('/api/v1/transfer/transfer_knowledge', methods=['POST'])
    def transfer_knowledge():
        """直接迁移知识"""
        data = request.json
        
        result = transfer_engine.transfer_knowledge(
            source_domain=data.get('source_domain'),
            target_domain=data.get('target_domain'),
            knowledge_type=data.get('knowledge_type', 'rules')
        )
        
        return jsonify(result)
    
    @app.route('/api/v1/transfer/stats', methods=['GET'])
    def get_transfer_stats():
        """获取迁移统计"""
        return jsonify(transfer_engine.get_statistics())
    
    # ============== 类比推理 API ==============
    
    @app.route('/api/v1/analogy/add_knowledge', methods=['POST'])
    def add_domain_knowledge():
        """添加领域知识"""
        data = request.json
        
        analogy_engine.add_domain_knowledge(
            domain=data.get('domain'),
            knowledge=data.get('knowledge', {})
        )
        
        return jsonify({
            "success": True,
            "message": f"Knowledge for {data.get('domain')} added"
        })
    
    @app.route('/api/v1/analogy/find', methods=['POST'])
    def find_analogy():
        """寻找类比"""
        data = request.json
        
        analogies = analogy_engine.find_analogy(
            source_domain=data.get('source_domain'),
            target_domain=data.get('target_domain'),
            problem=data.get('problem'),
            analogy_type=data.get('analogy_type', 'structural'),
            num_results=data.get('num_results', 5)
        )
        
        return jsonify({
            "success": True,
            "analogies": [a.to_dict() for a in analogies]
        })
    
    @app.route('/api/v1/analogy/stats', methods=['GET'])
    def get_analogy_stats():
        """获取类比统计"""
        return jsonify(analogy_engine.get_statistics())
    
    # ============== 统一推理 API ==============
    
    @app.route('/api/v1/reasoning/add_rule', methods=['POST'])
    def add_rule():
        """添加推理规则"""
        data = request.json
        
        from .unified_reasoner import LogicRule, LogicOperator
        
        rule = LogicRule(
            rule_id=data.get('rule_id'),
            antecedent=data.get('antecedent'),
            consequent=data.get('consequent'),
            operator=LogicOperator(data.get('operator', 'and')),
            conditions=data.get('conditions', []),
            conclusion=data.get('conclusion'),
            confidence=data.get('confidence', 0.8),
            domain=data.get('domain', 'general')
        )
        
        reasoner.add_rule(rule)
        
        return jsonify({
            "success": True,
            "message": f"Rule {rule.rule_id} added"
        })
    
    @app.route('/api/v1/reasoning/reason', methods=['POST'])
    def execute_reasoning():
        """执行推理"""
        data = request.json
        
        contexts_data = data.get('contexts', [])
        contexts = []
        
        for ctx in contexts_data:
            contexts.append(ReasoningContext(
                context_id=ctx.get('context_id', 'default'),
                facts=ctx.get('facts', []),
                rules=ctx.get('rules', []),
                assumptions=ctx.get('assumptions', []),
                constraints=ctx.get('constraints', []),
                metadata=ctx.get('metadata', {})
            ))
        
        result = reasoner.reason(
            contexts=contexts,
            query=data.get('query', ''),
            reasoning_type=data.get('reasoning_type'),
            max_steps=data.get('max_steps', 10)
        )
        
        return jsonify({
            "success": result.success,
            "result": result.to_dict()
        })
    
    @app.route('/api/v1/reasoning/multi_source', methods=['POST'])
    def multi_source_reasoning():
        """多源推理"""
        data = request.json
        
        contexts_data = data.get('contexts', [])
        contexts = []
        
        for ctx in contexts_data:
            contexts.append(ReasoningContext(
                context_id=ctx.get('context_id', 'default'),
                facts=ctx.get('facts', []),
                rules=ctx.get('rules', []),
                assumptions=ctx.get('assumptions', []),
                constraints=ctx.get('constraints', [])
            ))
        
        from .unified_reasoner import ReasoningType
        
        reasoning_types = [
            ReasoningType(rt) for rt in data.get('reasoning_types', 
            ['deductive', 'causal', 'common_sense'])
        ]
        
        results = reasoner.multi_source_reasoning(
            contexts=contexts,
            query=data.get('query', ''),
            reasoning_types=reasoning_types
        )
        
        return jsonify({
            "success": True,
            "results": {k: v.to_dict() for k, v in results.items()}
        })
    
    @app.route('/api/v1/reasoning/stats', methods=['GET'])
    def get_reasoning_stats():
        """获取推理统计"""
        return jsonify(reasoner.get_statistics())
    
    # ============== 系统 API ==============
    
    @app.route('/api/v1/system/health', methods=['GET'])
    def health_check():
        """健康检查"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "modules": {
                "knowledge_fusion": fusion_engine.get_statistics(),
                "transfer_learning": transfer_engine.get_statistics(),
                "analogical_reasoning": analogy_engine.get_statistics(),
                "unified_reasoning": reasoner.get_statistics()
            }
        })
    
    @app.route('/api/v1/system/stats', methods=['GET'])
    def get_all_stats():
        """获取所有统计"""
        return jsonify({
            "knowledge_fusion": fusion_engine.get_statistics(),
            "transfer_learning": transfer_engine.get_statistics(),
            "analogical_reasoning": analogy_engine.get_statistics(),
            "unified_reasoning": reasoner.get_statistics()
        })
    
    return app


# CLI支持
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-Domain Reasoning API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', default=8080, type=int, help='Port to bind')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)
