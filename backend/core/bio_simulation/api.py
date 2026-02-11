"""
API接口模块 - Bio Simulation API
================================
提供REST API接口用于访问生物模拟功能
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class APIResponse:
    """API响应"""
    success: bool
    data: Any
    message: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "error": self.error
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class BioSimulationAPI:
    """
    生物模拟系统API
    
    提供所有模块的统一API接口
    """
    
    def __init__(self):
        """初始化API"""
        self.protein_folding = None
        self.genomics = None
        self.drug_discovery = None
        self.cell_simulation = None
        self._initialize_modules()
    
    def _initialize_modules(self):
        """初始化所有模块"""
        from protein_folding import ProteinFolding
        from genomics import GenomicsAnalyzer
        from drug_discovery import DrugDiscovery
        from cell_simulation import CellSimulation
        
        self.protein_folding = ProteinFolding()
        self.genomics = GenomicsAnalyzer()
        self.drug_discovery = DrugDiscovery()
        self.cell_simulation = CellSimulation()
    
    # ==================== 蛋白质折叠API ====================
    
    def predict_protein_structure(self, 
                                  sequence: str,
                                  options: Optional[Dict] = None) -> APIResponse:
        """
        预测蛋白质3D结构
        
        POST /api/protein/predict
        
        Args:
            sequence: 氨基酸序列
            options: 预测选项
            
        Returns:
            预测的蛋白质结构
        """
        try:
            structure = self.protein_folding.predict(sequence)
            
            return APIResponse(
                success=True,
                data={
                    "sequence": structure.sequence,
                    "model_id": structure.model_id,
                    "confidence": structure.confidence_score,
                    "secondary_structure": structure.secondary_structure_string,
                    "pdb_content": structure.to_pdb(),
                    "num_residues": len(structure.residues),
                    "num_atoms": len(structure.atoms)
                },
                message="Structure prediction completed"
            )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    def refine_structure(self,
                         structure_data: Dict,
                         temperature: float = 300.0) -> APIResponse:
        """优化蛋白质结构"""
        try:
            from protein_folding import ProteinStructure
            structure = ProteinStructure(**structure_data)
            refined = self.protein_folding.run_molecular_dynamics(
                structure, temperature=temperature
            )
            return APIResponse(
                success=True,
                data={"model_id": refined.model_id, "pdb_content": refined.to_pdb()},
                message="Structure refinement completed"
            )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    def search_conformations(self,
                             structure_data: Dict,
                             num_conformations: int = 10) -> APIResponse:
        """搜索构象"""
        try:
            from protein_folding import ProteinStructure
            structure = ProteinStructure(**structure_data)
            conformations = self.protein_folding.search_conformations(
                structure, num_conformations
            )
            return APIResponse(
                success=True,
                data={"num_conformations": len(conformations)},
                message="Conformation search completed"
            )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    # ==================== 基因组分析API ====================
    
    def detect_variants(self,
                        sequence: str,
                        reference: Optional[str] = None,
                        min_quality: int = 30) -> APIResponse:
        """
        检测基因组变异
        
        POST /api/genomics/variants
        
        Args:
            sequence: 输入序列
            reference: 参考序列
            min_quality: 最低质量分数
            
        Returns:
            检测到的变异列表
        """
        try:
            variants = self.genomics.detect_variants(sequence, reference, min_quality)
            
            return APIResponse(
                success=True,
                data={
                    "num_variants": len(variants),
                    "variants": [
                        {
                            "id": v.id,
                            "chromosome": v.chromosome,
                            "position": v.position,
                            "reference": v.reference,
                            "alternate": v.alternate,
                            "type": v.variant_type.value,
                            "effect": v.effect.value,
                            "quality": v.quality,
                            "allele_frequency": v.allele_frequency
                        }
                        for v in variants
                    ]
                },
                message="Variant detection completed"
            )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    def analyze_gene_expression(self,
                                expression_data: List[Dict[str, Any]],
                                method: str = "TPM") -> APIResponse:
        """分析基因表达"""
        try:
            results = self.genomics.analyze_expression(expression_data, method)
            
            return APIResponse(
                success=True,
                data={
                    "method": method,
                    "num_genes": len(results),
                    "results": [
                        {
                            "gene_id": r.gene_id,
                            "gene_name": r.gene_name,
                            "expression": r.expression_level,
                            "normalized": r.normalized_expression
                        }
                        for r in results
                    ]
                },
                message="Expression analysis completed"
            )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    def classify_variants(self,
                          variants: List[Dict]) -> APIResponse:
        """分类变异致病性"""
        try:
            from genomics import GenomicVariant, VariantType, VariantEffect
            
            results = []
            for v_data in variants:
                variant = GenomicVariant(**v_data)
                classification = self.genomics.classify_pathogenicity(variant)
                results.append({
                    "variant_id": variant.id,
                    "classification": classification.value,
                    "score": variant.pathogenicity_score
                })
            
            return APIResponse(
                success=True,
                data={"classifications": results},
                message="Variant classification completed"
            )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    # ==================== 药物发现API ====================
    
    def screen_drug_candidates(self,
                               target_protein: Dict,
                               library: str = "zinc",
                               num_candidates: int = 100) -> APIResponse:
        """
        虚拟筛选候选药物
        
        POST /api/drug/screen
        
        Args:
            target_protein: 靶点蛋白质结构
            library: 分子库
            num_candidates: 返回候选数量
            
        Returns:
            筛选出的候选药物列表
        """
        try:
            candidates = self.drug_discovery.screen(
                target_protein, library, num_candidates
            )
            
            return APIResponse(
                success=True,
                data={
                    "library": library,
                    "num_screened": num_candidates,
                    "num_hits": len(candidates),
                    "candidates": [
                        {
                            "id": c.molecule_id,
                            "name": c.name,
                            "smiles": c.smiles,
                            "molecular_weight": c.molecular_weight,
                            "logp": c.logp,
                            "binding_affinity": c.binding_affinity,
                            "druglikeness": c.druglikeness,
                            "score": c.score
                        }
                        for c in candidates[:20]
                    ]
                },
                message="Virtual screening completed"
            )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    def predict_side_effects(self,
                             molecule: Dict) -> APIResponse:
        """预测药物副作用"""
        try:
            from drug_discovery import MoleculeCandidate
            mol = MoleculeCandidate(**molecule)
            predictions = self.drug_discovery.predict_side_effects(mol)
            
            return APIResponse(
                success=True,
                data={
                    "molecule_id": mol.molecule_id,
                    "predicted_side_effects": mol.predicted_side_effects,
                    "probabilities": mol.side_effect_probabilities
                },
                message="Side effect prediction completed"
            )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    def predict_drug_efficacy(self,
                              molecule: Dict,
                              target: str) -> APIResponse:
        """预测药物疗效"""
        try:
            from drug_discovery import MoleculeCandidate
            mol = MoleculeCandidate(**molecule)
            efficacy, confidence = self.drug_discovery.predict_efficacy(mol, target)
            
            return APIResponse(
                success=True,
                data={
                    "target": target,
                    "efficacy": efficacy,
                    "confidence": confidence
                },
                message="Efficacy prediction completed"
            )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    # ==================== 细胞模拟API ====================
    
    def create_cell(self,
                    cell_type: str = "eukaryote",
                    num_cells: int = 1) -> APIResponse:
        """
        创建细胞
        
        POST /api/cell/create
        
        Args:
            cell_type: 细胞类型
            num_cells: 创建数量
            
        Returns:
            创建的细胞信息
        """
        try:
            from cell_simulation import CellType
            
            cell_type_enum = CellType(cell_type) if cell_type in [ct.value for ct in CellType] else CellType.EUKARYOTE
            
            cells = []
            for _ in range(num_cells):
                cell = self.cell_simulation.create_cell(cell_type_enum)
                cells.append({
                    "cell_id": cell.cell_id,
                    "cell_type": cell.cell_type.value,
                    "radius": cell.radius,
                    "atp": cell.atp
                })
            
            return APIResponse(
                success=True,
                data={"cells": cells},
                message=f"Created {num_cells} cells"
            )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    def simulate_cells(self,
                       duration: float = 10.0,
                       dt: float = 0.01) -> APIResponse:
        """运行细胞模拟"""
        try:
            results = self.cell_simulation.simulate(duration, dt)
            
            return APIResponse(
                success=True,
                data={
                    "duration": duration,
                    "summary": results["summary"],
                    "num_timepoints": len(results["time_points"]),
                    "num_divisions": len(results["division_events"])
                },
                message="Cell simulation completed"
            )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    def get_pathway_simulation(self,
                               pathway_name: str,
                               time_steps: int = 100) -> APIResponse:
        """获取信号通路模拟结果"""
        try:
            from cell_simulation import CellType
            cell = self.cell_simulation.create_cell(CellType.EUKARYOTE)
            
            if pathway_name in cell.signaling_pathways:
                pathway = cell.signaling_pathways[pathway_name]
                results = pathway.simulate(time_steps)
                
                return APIResponse(
                    success=True,
                    data={
                        "pathway": pathway.name,
                        "components": list(results.keys()),
                        "simulation_results": results
                    },
                    message="Pathway simulation completed"
                )
            else:
                return APIResponse(
                    success=False,
                    data=None,
                    error=f"Pathway '{pathway_name}' not found"
                )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    def get_metabolic_flux(self,
                           network_id: str = "central_metabolism") -> APIResponse:
        """获取代谢通量"""
        try:
            from cell_simulation import CellType
            cell = self.cell_simulation.create_cell(CellType.EUKARYOTE)
            
            if cell.metabolic_network:
                results = cell.metabolic_network.simulate(time_steps=500)
                
                return APIResponse(
                    success=True,
                    data={
                        "network": cell.metabolic_network.name,
                        "metabolites": list(results.keys()),
                        "atp_balance": cell.metabolic_network.get_atp_balance()
                    },
                    message="Metabolic flux analysis completed"
                )
            else:
                return APIResponse(success=False, data=None, error="No metabolic network")
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    # ==================== 批量操作API ====================
    
    def run_complete_analysis(self,
                             sequence: str,
                             target: str) -> APIResponse:
        """
        运行完整的分析流程
        
        Args:
            sequence: 蛋白质序列
            target: 药物靶点
            
        Returns:
            完整分析结果
        """
        try:
            # 1. 蛋白质结构预测
            structure_result = self.predict_protein_structure(sequence)
            
            # 2. 变异检测
            variant_result = self.detect_variants(sequence)
            
            # 3. 药物筛选
            drug_result = self.screen_drug_candidates(
                structure_result.data, num_candidates=50
            )
            
            return APIResponse(
                success=True,
                data={
                    "structure_prediction": structure_result.data,
                    "variant_detection": variant_result.data,
                    "drug_screening": drug_result.data
                },
                message="Complete analysis pipeline finished"
            )
        except Exception as e:
            return APIResponse(success=False, data=None, error=str(e))
