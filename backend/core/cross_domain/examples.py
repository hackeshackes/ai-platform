"""
Cross-Domain Reasoning Examples
跨域推理系统使用示例
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any
from datetime import datetime


# ============== 知识融合示例 ==============

def example_knowledge_fusion():
    """知识融合示例"""
    from cross_domain.knowledge_fusion import KnowledgeFusion, KnowledgeSource
    
    fusion = KnowledgeFusion()
    
    # 添加医学知识源
    medical_source = KnowledgeSource(
        source_id="medical_kb_001",
        name="Medical Knowledge Base",
        domain="medicine",
        knowledge_type="ontology",
        data={
            "classes": {
                "disease": {"attributes": {"name": "Disease", "description": "Medical condition"}},
                "treatment": {"attributes": {"name": "Treatment", "description": "Therapeutic intervention"}},
                "symptom": {"attributes": {"name": "Symptom", "description": "Observable manifestation"}}
            },
            "relations": [
                {"from": "disease", "to": "treatment", "type": "treated_by"},
                {"from": "disease", "to": "symptom", "type": "has_symptom"}
            ]
        },
        confidence=0.9
    )
    
    # 添加生物学知识源
    biology_source = KnowledgeSource(
        source_id="biology_kb_001",
        name="Biology Knowledge Base",
        domain="biology",
        knowledge_type="ontology",
        data={
            "classes": {
                "organism": {"attributes": {"name": "Organism", "description": "Living entity"}},
                "process": {"attributes": {"name": "Biological Process", "description": "Cellular activity"}},
                "structure": {"attributes": {"name": "Structure", "description": "Physical form"}}
            },
            "relations": [
                {"from": "organism", "to": "process", "type": "performs"},
                {"from": "organism", "to": "structure", "type": "has_structure"}
            ]
        },
        confidence=0.85
    )
    
    # 添加化学知识源
    chemistry_source = KnowledgeSource(
        source_id="chemistry_kb_001",
        name="Chemistry Knowledge Base",
        domain="chemistry",
        knowledge_type="ontology",
        data={
            "classes": {
                "compound": {"attributes": {"name": "Compound", "description": "Chemical substance"}},
                "reaction": {"attributes": {"name": "Reaction", "description": "Chemical process"}},
                "property": {"attributes": {"name": "Property", "description": "Chemical characteristic"}}
            },
            "relations": [
                {"from": "compound", "to": "reaction", "type": "undergoes"},
                {"from": "compound", "to": "property", "type": "has_property"}
            ]
        },
        confidence=0.8
    )
    
    # 执行融合
    result = fusion.fuse(
        sources=[medical_source, biology_source, chemistry_source],
        fusion_type="comprehensive"
    )
    
    return {
        "success": result.success,
        "unified_entities": len(result.unified_knowledge.get("entities", {})),
        "conflicts_resolved": len(result.conflicts_resolved),
        "overall_confidence": result.confidence_scores.get("overall_confidence", 0)
    }


# ============== 迁移学习示例 ==============

def example_transfer_learning():
    """迁移学习示例"""
    from cross_domain.transfer_learning import TransferLearning, DomainSpec, AdaptationMethod
    
    transfer = TransferLearning()
    
    # 添加源领域（生物学）
    biology_domain = DomainSpec(
        domain_id="biology_001",
        name="Biology Domain",
        domain_type="source",
        features={
            "complexity": 0.7,
            "abstraction_level": 0.6,
            "data_density": 0.8,
            "temporal_dependency": 0.5
        },
        examples=[
            {"sample": "cell_division", "label": "growth"},
            {"sample": "protein_synthesis", "label": "function"}
        ],
        metadata={"source": "textbook"}
    )
    
    # 添加目标领域（医学）
    medicine_domain = DomainSpec(
        domain_id="medicine_001",
        name="Medicine Domain",
        domain_type="target",
        features={
            "complexity": 0.9,
            "abstraction_level": 0.7,
            "data_density": 0.6,
            "temporal_dependency": 0.8
        },
        examples=[
            {"sample": "tumor_growth", "label": "pathology"},
            {"sample": "drug_response", "label": "treatment"}
        ],
        metadata={"source": "clinical"}
    )
    
    transfer.add_domain(biology_domain)
    transfer.add_domain(medicine_domain)
    
    # 执行迁移
    result = transfer.adapt(
        source_domain="biology_001",
        target_domain="medicine_001",
        model={"type": "base_model"},
        method=AdaptationMethod.DOMAIN_ADVERSARIAL,
        num_iterations=100
    )
    
    return {
        "success": result.success,
        "knowledge_preservation": result.knowledge_preservation_score,
        "transfer_gain": result.transfer_gain_score,
        "target_accuracy": result.metrics.get("target_accuracy", 0)
    }


# ============== 类比推理示例 ==============

def example_analogical_reasoning():
    """类比推理示例"""
    from cross_domain.analogical_reasoning import AnalogicalReasoner
    
    reasoner = AnalogicalReasoner()
    
    # 添加电子学知识
    electronics_knowledge = {
        "circuit": {
            "components": ["resistor", "capacitor", "transistor"],
            "function": "control_current",
            "behavior": "Ohm's law"
        },
        "current": {
            "direction": "flow",
            "magnitude": "measured_in_amperes",
            "behavior": "I = V/R"
        }
    }
    
    # 添加生物学知识
    biology_knowledge = {
        "circulatory_system": {
            "components": ["heart", "blood_vessels", "blood"],
            "function": "transport_materials",
            "behavior": "pressure_driven"
        },
        "blood_flow": {
            "direction": "circular",
            "magnitude": "cardiac_output",
            "behavior": "pressure_gradient"
        }
    }
    
    reasoner.add_domain_knowledge("electronics", electronics_knowledge)
    reasoner.add_domain_knowledge("biology", biology_knowledge)
    
    # 寻找类比
    analogies = reasoner.find_analogy(
        source_domain="electronics",
        target_domain="biology",
        problem={"topic": "flow_systems"},
        analogy_type="structural",
        num_results=3
    )
    
    return {
        "success": len(analogies) > 0,
        "analogies_found": len(analogies),
        "top_analogy": analogies[0].to_dict() if analogies else None,
        "avg_similarity": sum(a.similarity_score for a in analogies) / len(analogies) if analogies else 0
    }


# ============== 统一推理示例 ==============

def example_unified_reasoning():
    """统一推理示例"""
    from cross_domain.unified_reasoner import UnifiedReasoner, ReasoningContext, ReasoningType
    
    reasoner = UnifiedReasoner()
    
    # 添加推理规则
    from cross_domain.unified_reasoner import LogicRule, LogicOperator
    
    rule1 = LogicRule(
        rule_id="rule_001",
        antecedent="infection",
        consequent="inflammation",
        operator=LogicOperator.IMPLIES,
        conditions=["pathogen_present", "immune_response"],
        conclusion="inflammatory_response",
        confidence=0.85,
        domain="medicine"
    )
    
    rule2 = LogicRule(
        rule_id="rule_002",
        antecedent="inflammation",
        consequent="symptom",
        operator=LogicOperator.IMPLIES,
        conditions=["tissue_damage"],
        conclusion="observable_symptoms",
        confidence=0.75,
        domain="medicine"
    )
    
    reasoner.add_rule(rule1)
    reasoner.add_rule(rule2)
    
    # 创建推理上下文
    context = ReasoningContext(
        context_id="medical_diagnosis",
        facts=[
            {"fact": "pathogen_detected", "confidence": 0.9},
            {"fact": "immune_cells_active", "confidence": 0.85},
            {"fact": "tissue_damage_observed", "confidence": 0.7}
        ],
        rules=[],
        assumptions=["immune_system_functional"],
        constraints=["acute_condition"]
    )
    
    # 执行推理
    result = reasoner.reason(
        contexts=[context],
        query="What symptoms might be present?",
        reasoning_type=ReasoningType.DEDUCTIVE,
        max_steps=5
    )
    
    return {
        "success": result.success,
        "conclusion": result.conclusion,
        "confidence": result.confidence,
        "reasoning_steps": len(result.reasoning_steps)
    }


# ============== 综合示例 ==============

def example_cross_domain_integration():
    """跨域集成示例"""
    from cross_domain.knowledge_fusion import KnowledgeFusion, KnowledgeSource
    from cross_domain.transfer_learning import TransferLearning, DomainSpec
    from cross_domain.analogical_reasoning import AnalogicalReasoner
    from cross_domain.unified_reasoner import UnifiedReasoner, ReasoningContext
    
    # 1. 知识融合
    fusion = KnowledgeFusion()
    
    medical_source = KnowledgeSource(
        source_id="med_source",
        name="Medical KB",
        domain="medicine",
        knowledge_type="facts",
        data={
            "disease": "heart_disease",
            "symptoms": ["chest_pain", "shortness_of_breath"],
            "treatment": "medication"
        },
        confidence=0.9
    )
    
    result = fusion.fuse(sources=[medical_source])
    
    # 2. 迁移学习
    transfer = TransferLearning()
    transfer.add_domain(DomainSpec(
        domain_id="biology",
        name="Biology",
        domain_type="source",
        features={"similarity": 0.7},
        examples=[{"case": "heart_function"}]
    ))
    transfer_result = transfer.adapt(
        source_domain="biology",
        target_domain="medicine"
    )
    
    # 3. 类比推理
    analogizer = AnalogicalReasoner()
    analogies = analogizer.find_analogy(
        source_domain="electronics",
        target_domain="biology",
        problem={"case": "pump_system"}
    )
    
    # 4. 统一推理
    reasoner = UnifiedReasoner()
    context = ReasoningContext(
        context_id="research",
        facts=[{"fact": "cross_domain_data_available"}],
        rules=[],
        assumptions=[],
        constraints=[]
    )
    reasoning_result = reasoner.reason(
        contexts=[context],
        query="Synthesize cross-domain insights"
    )
    
    return {
        "fusion_success": result.success,
        "transfer_success": transfer_result.success,
        "analogies_found": len(analogies),
        "reasoning_success": reasoning_result.success,
        "integration_score": (
            (1 if result.success else 0) +
            (1 if transfer_result.success else 0) +
            (1 if reasoning_result.success else 0)
        ) / 3
    }


# 运行示例
if __name__ == "__main__":
    print("=" * 60)
    print("Cross-Domain Reasoning System - Examples")
    print("=" * 60)
    
    print("\n1. Knowledge Fusion Example:")
    fusion_result = example_knowledge_fusion()
    print(f"   Success: {fusion_result['success']}")
    print(f"   Entities: {fusion_result['unified_entities']}")
    
    print("\n2. Transfer Learning Example:")
    transfer_result = example_transfer_learning()
    print(f"   Success: {transfer_result['success']}")
    print(f"   Preservation: {transfer_result['knowledge_preservation']:.2f}")
    
    print("\n3. Analogical Reasoning Example:")
    analogy_result = example_analogical_reasoning()
    print(f"   Success: {analogy_result['success']}")
    print(f"   Analogies: {analogy_result['analogies_found']}")
    
    print("\n4. Unified Reasoning Example:")
    reasoning_result = example_unified_reasoning()
    print(f"   Success: {reasoning_result['success']}")
    print(f"   Confidence: {reasoning_result['confidence']:.2f}")
    
    print("\n5. Cross-Domain Integration Example:")
    integration = example_cross_domain_integration()
    print(f"   Integration Score: {integration['integration_score']:.2f}")
    
    print("\n" + "=" * 60)
