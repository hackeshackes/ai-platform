"""
使用示例 - Bio Simulation Examples
==================================
提供各模块的使用示例和演示代码
"""

from typing import List, Dict


def example_protein_folding():
    """
    蛋白质折叠示例
    
    演示如何使用ProteinFolding类进行蛋白质3D结构预测
    """
    from protein_folding import ProteinFolding
    
    # 初始化折叠器
    folder = ProteinFolding()
    
    # 示例氨基酸序列 (胰岛素B链部分)
    sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"
    
    print("=" * 60)
    print("蛋白质折叠预测示例")
    print("=" * 60)
    print(f"输入序列: {sequence}")
    print(f"序列长度: {len(sequence)}")
    print()
    
    # 预测结构
    structure = folder.predict(sequence)
    
    print("预测结果:")
    print(f"  模型ID: {structure.model_id}")
    print(f"  置信度: {structure.confidence_score:.2f}%")
    print(f"  残基数: {len(structure.residues)}")
    print(f"  原子数: {len(structure.atoms)}")
    print()
    
    print("二级结构分布:")
    ss = structure.secondary_structure_string
    print(f"  α-螺旋 (H): {ss.count('H')} ({ss.count('H')/len(ss)*100:.1f}%)")
    print(f"  β-折叠 (E): {ss.count('E')} ({ss.count('E')/len(ss)*100:.1f}%)")
    print(f"  无规卷曲 (C): {ss.count('C')} ({ss.count('C')/len(ss)*100:.1f}%)")
    print()
    
    # 分子动力学优化
    print("运行分子动力学优化...")
    refined = folder.run_molecular_dynamics(structure, temperature=300.0)
    print(f"优化后置信度: {refined.confidence_score:.2f}%")
    print()
    
    # 构象搜索
    print("搜索构象空间...")
    conformations = folder.search_conformations(structure, num_conformations=5)
    print(f"生成了 {len(conformations)} 个构象")
    print()
    
    # 保存PDB
    pdb_content = structure.to_pdb()
    print("PDB文件预览 (前10行):")
    for line in pdb_content.split('\n')[:10]:
        print(f"  {line}")


def example_genomics_analysis():
    """
    基因组分析示例
    
    演示如何使用GenomicsAnalyzer进行基因组分析
    """
    from genomics import GenomicsAnalyzer, VariantType, VariantEffect
    
    print("=" * 60)
    print("基因组分析示例")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = GenomicsAnalyzer()
    
    # 示例DNA序列
    sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    reference = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    
    print(f"输入序列: {sequence}")
    print()
    
    # 检测变异
    print("检测变异...")
    variants = analyzer.detect_variants(sequence, reference)
    print(f"检测到 {len(variants)} 个变异")
    print()
    
    for variant in variants[:5]:
        print(f"  变异: {variant.chromosome}:{variant.position}")
        print(f"    类型: {variant.variant_type.value}")
        print(f"    效应: {variant.effect.value}")
        print(f"    质量: {variant.quality:.1f}")
        print()
    
    # 序列翻译
    print("DNA序列翻译:")
    protein = analyzer.translate_sequence(sequence)
    print(f"  蛋白质序列: {protein}")
    print()
    
    # GC含量
    gc_content = analyzer.calculate_gc_content(sequence)
    print(f"GC含量: {gc_content*100:.1f}%")
    print()
    
    # 表达分析
    print("基因表达分析...")
    expression_data = [
        {"gene_id": "BRCA1", "gene_name": "BRCA1", "count": 1500, "length": 8000},
        {"gene_id": "TP53", "gene_name": "TP53", "count": 2500, "length": 2000},
        {"gene_id": "EGFR", "gene_name": "EGFR", "count": 800, "length": 12000}
    ]
    
    results = analyzer.analyze_expression(expression_data, method="TPM")
    for r in results:
        print(f"  {r.gene_name}: {r.normalized_expression:.2f} TPM")
    print()
    
    # 变异指标
    print("变异检测指标:")
    if variants:
        metrics = analyzer.calculate_variant_metrics(variants)
        print(f"  总变异数: {metrics['total_variants']}")
        print(f"  转换/颠换比: {metrics['ti_tv_ratio']:.2f}")


def example_drug_discovery():
    """
    药物发现示例
    
    演示如何使用DrugDiscovery进行虚拟筛选
    """
    from drug_discovery import DrugDiscovery, MoleculeCandidate
    
    print("=" * 60)
    print("药物发现示例")
    print("=" * 60)
    
    # 初始化引擎
    discovery = DrugDiscovery()
    
    # 加载分子库
    print("加载分子库...")
    num_molecules = discovery.load_molecule_library("zinc")
    print(f"加载了 {num_molecules} 个分子")
    print()
    
    # 模拟靶点蛋白质
    target_protein = {"model_id": "target_001", "sequence": "MKLP..."}
    
    # 虚拟筛选
    print("运行虚拟筛选...")
    candidates = discovery.screen(target_protein, library="zinc", num_candidates=100)
    print(f"筛选出 {len(candidates)} 个候选药物")
    print()
    
    print("排名前10的候选药物:")
    for i, mol in enumerate(candidates[:10], 1):
        print(f"  {i}. {mol.name}")
        print(f"     分子量: {mol.molecular_weight:.1f}")
        print(f"     LogP: {mol.logp:.2f}")
        print(f"     结合亲和力: {mol.binding_affinity:.2f} kcal/mol")
        print(f"     类药性: {'是' if mol.druglikeness else '否'}")
        print(f"     评分: {mol.score:.2f}")
        print()
    
    # 分析命中化合物
    print("命中化合物分析...")
    analysis = discovery.analyze_lead_compounds(candidates)
    print(f"  总候选数: {analysis['total_candidates']}")
    print(f"  类药化合物数: {analysis['druglike_count']}")
    print(f"  平均分子量: {analysis['avg_molecular_weight']:.1f}")
    print()
    
    # 预测单个分子副作用
    print("副作用预测...")
    if candidates:
        mol = candidates[0]
        side_effects = discovery.predict_side_effects(mol)
        print(f"  {mol.name} 预测的副作用:")
        for se in mol.predicted_side_effects[:5]:
            print(f"    - {se}")
        print()
        
        # 疗效预测
        print("疗效预测...")
        efficacy, confidence = discovery.predict_efficacy(mol, "EGFR")
        print(f"  靶点: EGFR")
        print(f"  预测疗效: {efficacy:.2f}")
        print(f"  置信度: {confidence:.2f}")


def example_cell_simulation():
    """
    细胞模拟示例
    
    演示如何使用CellSimulation进行细胞行为模拟
    """
    from cell_simulation import CellSimulation, CellType
    
    print("=" * 60)
    print("细胞模拟示例")
    print("=" * 60)
    
    # 初始化模拟器
    sim = CellSimulation()
    
    # 创建细胞
    print("创建细胞...")
    cell = sim.create_cell(CellType.EUKARYOTE)
    print(f"  细胞ID: {cell.cell_id}")
    print(f"  细胞类型: {cell.cell_type.value}")
    print(f"  半径: {cell.radius:.2f} μm")
    print(f"  ATP: {cell.atp:.2f} mM")
    print()
    
    # 查看信号通路
    print("信号通路:")
    for pathway_name, pathway in cell.signaling_pathways.items():
        print(f"  {pathway.name}: {len(pathway.components)} 个组件")
        print(f"    输入: {pathway.input_components}")
        print(f"    输出: {pathway.output_components}")
    print()
    
    # 模拟信号通路
    print("MAPK通路模拟...")
    if "MAPK" in cell.signaling_pathways:
        pathway = cell.signaling_pathways["MAPK"]
        results = pathway.simulate(time_steps=50)
        for comp_id, values in results.items():
            final_state = values[-1]
            print(f"  {comp_id}: 最终激活状态 = {final_state:.2f}")
    print()
    
    # 运行细胞模拟
    print("运行细胞模拟 (10小时)...")
    results = sim.simulate(duration=10.0, dt=0.01)
    
    print("模拟结果摘要:")
    summary = results["summary"]
    print(f"  存活细胞数: {summary['alive_cells']}")
    print(f"  总分裂次数: {summary['total_divisions']}")
    print(f"  最终ATP: {summary['final_atp']:.2f} mM")
    print(f"  最终ROS: {summary['final_ros']:.2f}")
    print(f"  分裂事件: {len(results['division_events'])}")
    print()


def example_api_usage():
    """
    API使用示例
    
    演示如何使用BioSimulationAPI
    """
    from api import BioSimulationAPI
    
    print("=" * 60)
    print("API使用示例")
    print("=" * 60)
    
    # 初始化API
    api = BioSimulationAPI()
    
    # 蛋白质折叠
    print("1. 蛋白质结构预测...")
    protein_result = api.predict_protein_structure("MALWMRLLPLLALLALWGPDPAA")
    if protein_result.success:
        print(f"   置信度: {protein_result.data['confidence']:.2f}%")
    print()
    
    # 基因组分析
    print("2. 基因组变异检测...")
    variant_result = api.detect_variants("ATGCGATCGATCGATCG")
    if variant_result.success:
        print(f"   检测到 {variant_result.data['num_variants']} 个变异")
    print()
    
    # 药物筛选
    print("3. 药物虚拟筛选...")
    drug_result = api.screen_drug_candidates({"model_id": "test"}, num_candidates=10)
    if drug_result.success:
        print(f"   筛选出 {drug_result.data['num_hits']} 个命中")
    print()
    
    # 细胞模拟
    print("4. 细胞模拟...")
    cell_result = api.create_cell("eukaryote", num_cells=3)
    if cell_result.success:
        print(f"   创建了 {len(cell_result.data['cells'])} 个细胞")
    
    sim_result = api.simulate_cells(duration=1.0)
    if sim_result.success:
        print(f"   存活细胞: {sim_result.data['summary']['alive_cells']}")
    print()
    
    # 完整分析流程
    print("5. 运行完整分析流程...")
    pipeline_result = api.run_complete_analysis(
        sequence="MALWMRLLPLLALLALWGPDPAA",
        target="EGFR"
    )
    if pipeline_result.success:
        print("   流程完成!")
        print(f"   结构置信度: {pipeline_result.data['structure_prediction']['confidence']:.2f}%")
        print(f"   检测变异: {pipeline_result.data['variant_detection']['num_variants']} 个")
        print(f"   命中药物: {pipeline_result.data['drug_screening']['num_hits']} 个")


def run_all_examples():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("BioSimulation 系统使用示例")
    print("=" * 70 + "\n")
    
    # 选择要运行的示例
    examples = [
        ("蛋白质折叠", example_protein_folding),
        ("基因组分析", example_genomics_analysis),
        ("药物发现", example_drug_discovery),
        ("细胞模拟", example_cell_simulation),
        ("API使用", example_api_usage)
    ]
    
    for name, func in examples:
        try:
            func()
            print("\n" + "-" * 60 + "\n")
        except Exception as e:
            print(f"示例运行出错: {e}")
            print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    run_all_examples()
