"""
测试用例 - Bio Simulation Tests
==============================
提供各模块的单元测试和集成测试
"""

import unittest
import numpy as np
from typing import List, Dict, Any


class TestProteinFolding(unittest.TestCase):
    """蛋白质折叠模块测试"""
    
    def setUp(self):
        from protein_folding import ProteinFolding, AminoAcid
        self.folder = ProteinFolding()
        self.valid_sequence = "MALWMRLLPLLALLALWGPDPAA"
        self.long_sequence = "A" * 100
    
    def test_sequence_validation(self):
        """测试序列验证"""
        # 有效序列
        result = self.folder.predict(self.valid_sequence)
        self.assertEqual(len(result.sequence), len(self.valid_sequence))
        
        # 无效序列 (包含非标准氨基酸)
        with self.assertRaises(ValueError):
            self.folder.predict("MALWMRLXPLL")  # X不是标准氨基酸
    
    def test_structure_prediction(self):
        """测试结构预测"""
        structure = self.folder.predict(self.valid_sequence)
        
        self.assertIsNotNone(structure.model_id)
        self.assertIsNotNone(structure.confidence_score)
        self.assertGreater(structure.confidence_score, 0)
        self.assertGreater(len(structure.residues), 0)
        self.assertEqual(len(structure.residues), len(self.valid_sequence))
    
    def test_confidence_score(self):
        """测试置信度分数"""
        structure = self.folder.predict(self.valid_sequence)
        
        # 置信度应该在0-100之间
        self.assertGreaterEqual(structure.confidence_score, 0)
        self.assertLessEqual(structure.confidence_score, 100)
        
        # 如果有plDDT数据
        if structure.predicted_plddt is not None:
            self.assertEqual(len(structure.predicted_plddt), len(self.valid_sequence))
    
    def test_secondary_structure(self):
        """测试二级结构预测"""
        structure = self.folder.predict(self.valid_sequence)
        ss = structure.secondary_structure_string
        
        # 应该只包含H, E, C
        for char in ss:
            self.assertIn(char, ['H', 'E', 'C'])
        
        self.assertEqual(len(ss), len(self.valid_sequence))
    
    def test_pdb_generation(self):
        """测试PDB文件生成"""
        structure = self.folder.predict(self.valid_sequence)
        pdb = structure.to_pdb()
        
        # 应该包含HEADER和END
        self.assertIn("HEADER", pdb)
        self.assertIn("END", pdb)
        self.assertIn("ATOM", pdb)
    
    def test_molecular_dynamics(self):
        """测试分子动力学优化"""
        structure = self.folder.predict(self.valid_sequence)
        refined = self.folder.run_molecular_dynamics(structure, temperature=300.0)
        
        self.assertIsNotNone(refined)
        self.assertEqual(len(refined.atoms), len(structure.atoms))
    
    def test_conformation_search(self):
        """测试构象搜索"""
        structure = self.folder.predict(self.valid_sequence)
        conformations = self.folder.search_conformations(structure, num_conformations=5)
        
        self.assertEqual(len(conformations), 5)
        for conf in conformations:
            self.assertEqual(len(conf.atoms), len(structure.atoms))
    
    def test_amino_acid_properties(self):
        """测试氨基酸属性"""
        from protein_folding import AminoAcid
        
        # 分子量
        mw = AminoAcid.get_molecular_weight("ACDEFGHIKLMNPQRSTVWY")
        self.assertGreater(mw, 0)
        
        # 全名
        self.assertEqual(AminoAcid.get_full_name("A"), "Alanine")
        self.assertEqual(AminoAcid.get_full_name("W"), "Tryptophan")


class TestGenomics(unittest.TestCase):
    """基因组分析模块测试"""
    
    def setUp(self):
        from genomics import GenomicsAnalyzer
        self.analyzer = GenomicsAnalyzer()
    
    def test_variant_detection(self):
        """测试变异检测"""
        sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        reference = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        
        variants = self.analyzer.detect_variants(sequence, reference)
        
        # 应该有变异检测结果
        self.assertIsInstance(variants, list)
    
    def test_sequence_translation(self):
        """测试序列翻译"""
        dna = "ATGCGATCGATCGATCG"
        protein = self.analyzer.translate_sequence(dna)
        
        self.assertIsInstance(protein, str)
        self.assertEqual(len(protein), len(dna) // 3)
    
    def test_gc_content(self):
        """测试GC含量计算"""
        gc_high = "GCGCGCGCGC"
        gc_low = "ATATATATAT"
        
        high_content = self.analyzer.calculate_gc_content(gc_high)
        low_content = self.analyzer.calculate_gc_content(gc_low)
        
        self.assertGreater(high_content, 0.5)
        self.assertLess(low_content, 0.5)
    
    def test_variant_classification(self):
        """测试变异分类"""
        from genomics import GenomicVariant, VariantType, VariantEffect
        
        variant = GenomicVariant(
            chromosome="1",
            position=100,
            reference="A",
            alternate="G",
            variant_type=VariantType.SNV,
            effect=VariantEffect.MISSENSE,
            quality=60,
            gnomad_af=0.0001
        )
        
        classification = self.analyzer.classify_pathogenicity(variant)
        self.assertIsNotNone(classification)
    
    def test_expression_analysis(self):
        """测试表达分析"""
        expression_data = [
            {"gene_id": "BRCA1", "gene_name": "BRCA1", "count": 1500, "length": 8000},
            {"gene_id": "TP53", "gene_name": "TP53", "count": 2500, "length": 2000}
        ]
        
        results = self.analyzer.analyze_expression(expression_data, method="TPM")
        
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertGreaterEqual(r.normalized_expression, 0)
    
    def test_variant_metrics(self):
        """测试变异指标计算"""
        from genomics import GenomicVariant, VariantType, VariantEffect
        
        variants = [
            GenomicVariant(chromosome="1", position=100, reference="A", alternate="G",
                          variant_type=VariantType.SNV, effect=VariantEffect.MISSENSE),
            GenomicVariant(chromosome="1", position=200, reference="C", alternate="T",
                          variant_type=VariantType.SNV, effect=VariantEffect.SYNONYMOUS),
        ]
        
        metrics = self.analyzer.calculate_variant_metrics(variants)
        
        self.assertEqual(metrics["total_variants"], 2)
        self.assertIn("SNV", metrics["variant_type_counts"])


class TestDrugDiscovery(unittest.TestCase):
    """药物发现模块测试"""
    
    def setUp(self):
        from drug_discovery import DrugDiscovery
        self.discovery = DrugDiscovery()
    
    def test_library_loading(self):
        """测试分子库加载"""
        count = self.discovery.load_molecule_library("zinc")
        self.assertEqual(count, 1000)
        
        count = self.discovery.load_molecule_library("chembl")
        self.assertEqual(count, 500)
    
    def test_molecule_properties(self):
        """测试分子属性"""
        from drug_discovery import MoleculeCandidate
        
        mol = MoleculeCandidate(
            molecule_id="TEST_001",
            name="Test Compound",
            smiles="CCO",
            molecular_weight=46.07,
            logp=-0.31,
            hydrogen_bond_donors=1,
            hydrogen_bond_acceptors=1
        )
        
        self.assertTrue(mol.druglikeness)
        self.assertIn("C", mol.molecular_formula)
    
    def test_binding_site_detection(self):
        """测试结合位点检测"""
        target_protein = {"model_id": "test"}
        sites = self.discovery.find_binding_sites(target_protein)
        
        self.assertGreater(len(sites), 0)
        for site in sites:
            self.assertIsNotNone(site.site_id)
            self.assertGreater(site.volume, 0)
    
    def test_molecular_docking(self):
        """测试分子对接"""
        from drug_discovery import MoleculeCandidate
        
        mol = MoleculeCandidate(
            molecule_id="TEST_001",
            name="Test",
            smiles="CCO",
            molecular_weight=300,
            logp=2.0
        )
        
        result = self.discovery.dock_molecule(mol, {}, None)
        
        self.assertIsNotNone(result.binding_energy)
        self.assertLess(result.binding_energy, 0)
    
    def test_virtual_screening(self):
        """测试虚拟筛选"""
        target_protein = {"model_id": "test"}
        candidates = self.discovery.screen(target_protein, num_candidates=50)
        
        self.assertLessEqual(len(candidates), 50)
        for mol in candidates:
            self.assertIsNotNone(mol.binding_affinity)
            self.assertLess(mol.binding_affinity, 0)
    
    def test_side_effect_prediction(self):
        """测试副作用预测"""
        from drug_discovery import MoleculeCandidate
        
        mol = MoleculeCandidate(
            molecule_id="TEST_001",
            name="Test",
            smiles="CCO",
            molecular_weight=400,
            logp=3.0,
            hydrogen_bond_donors=2
        )
        
        predictions = self.discovery.predict_side_effects(mol)
        
        self.assertIsInstance(predictions, dict)
        self.assertEqual(len(mol.predicted_side_effects), 5)
    
    def test_efficacy_prediction(self):
        """测试疗效预测"""
        from drug_discovery import MoleculeCandidate
        
        mol = MoleculeCandidate(
            molecule_id="TEST_001",
            name="Test",
            smiles="CCO",
            molecular_weight=300,
            binding_affinity=-9.0
        )
        
        efficacy, confidence = self.discovery.predict_efficacy(mol, "EGFR")
        
        self.assertGreaterEqual(efficacy, 0)
        self.assertLessEqual(efficacy, 1)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)


class TestCellSimulation(unittest.TestCase):
    """细胞模拟模块测试"""
    
    def setUp(self):
        from cell_simulation import CellSimulation, CellType
        self.sim = CellSimulation()
    
    def test_cell_creation(self):
        """测试细胞创建"""
        cell = self.sim.create_cell(CellType.EUKARYOTE)
        
        self.assertIsNotNone(cell.cell_id)
        self.assertEqual(cell.cell_type, CellType.EUKARYOTE)
        self.assertGreater(cell.radius, 0)
        self.assertGreater(cell.atp, 0)
    
    def test_cell_update(self):
        """测试细胞更新"""
        cell = self.sim.create_cell(CellType.EUKARYOTE)
        initial_atp = cell.atp
        
        cell.update(dt=0.1)
        
        # ATP应该减少
        self.assertLessEqual(cell.atp, initial_atp)
    
    def test_signaling_pathway(self):
        """测试信号通路"""
        from cell_simulation import PathwayComponentType
        
        cell = self.sim.create_cell(CellType.EUKARYOTE)
        
        # 检查MAPK通路
        self.assertIn("MAPK", cell.signaling_pathways)
        
        pathway = cell.signaling_pathways["MAPK"]
        self.assertEqual(len(pathway.components), 4)  # RAF, MEK, ERK, ELK1
        
        # 模拟通路
        results = pathway.simulate(time_steps=50)
        self.assertEqual(len(results), 4)
    
    def test_metabolic_network(self):
        """测试代谢网络"""
        cell = self.sim.create_cell(CellType.EUKARYOTE)
        
        self.assertIsNotNone(cell.metabolic_network)
        self.assertGreater(len(cell.metabolic_network.reactions), 0)
        self.assertGreater(len(cell.metabolic_network.metabolites), 0)
    
    def test_cell_simulation_run(self):
        """测试细胞模拟运行"""
        # 创建多个细胞
        for _ in range(3):
            self.sim.create_cell(CellType.EUKARYOTE)
        
        # 运行模拟
        results = self.sim.simulate(duration=0.1, dt=0.001)
        
        self.assertIn("summary", results)
        self.assertEqual(results["summary"]["total_cells"], 3)
        self.assertGreaterEqual(results["summary"]["alive_cells"], 0)


class TestAPI(unittest.TestCase):
    """API测试"""
    
    def setUp(self):
        from api import BioSimulationAPI
        self.api = BioSimulationAPI()
    
    def test_protein_prediction_api(self):
        """测试蛋白质预测API"""
        result = self.api.predict_protein_structure("MALWMRLLPLL")
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data["model_id"])
        self.assertIsNotNone(result.data["confidence"])
    
    def test_variant_detection_api(self):
        """测试变异检测API"""
        result = self.api.detect_variants("ATGCGATCG")
        
        self.assertTrue(result.success)
        self.assertIn("num_variants", result.data)
    
    def test_drug_screening_api(self):
        """测试药物筛选API"""
        result = self.api.screen_drug_candidates({"model_id": "test"}, num_candidates=10)
        
        self.assertTrue(result.success)
        self.assertIn("candidates", result.data)
    
    def test_cell_creation_api(self):
        """测试细胞创建API"""
        result = self.api.create_cell("eukaryote", num_cells=2)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.data["cells"]), 2)
    
    def test_cell_simulation_api(self):
        """测试细胞模拟API"""
        self.api.create_cell("eukaryote", num_cells=1)
        result = self.api.simulate_cells(duration=0.1)
        
        self.assertTrue(result.success)
        self.assertIn("summary", result.data)
    
    def test_api_response_format(self):
        """测试API响应格式"""
        from api import APIResponse
        
        response = APIResponse(success=True, data={"test": "value"})
        
        self.assertTrue(response.success)
        self.assertEqual(response.data["test"], "value")
        
        json_str = response.to_json()
        self.assertIn("success", json_str)
        self.assertIn("test", json_str)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestProteinFolding))
    suite.addTests(loader.loadTestsFromTestCase(TestGenomics))
    suite.addTests(loader.loadTestsFromTestCase(TestDrugDiscovery))
    suite.addTests(loader.loadTestsFromTestCase(TestCellSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestAPI))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回结果
    return result


if __name__ == "__main__":
    run_tests()
