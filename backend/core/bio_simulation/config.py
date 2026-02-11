"""
配置模块 - Bio Simulation Configuration
======================================
系统配置和参数管理
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ProteinFoldingConfig:
    """蛋白质折叠配置"""
    # 模型参数
    model_type: str = "alphafold_v2"  # alphafold_v1, alphafold_v2, custom
    use_amber_relax: bool = True
    use_templates: bool = True
    
    # 预测参数
    max_sequence_length: int = 2500
    num_recycles: int = 3
    recycle_at: bool = True
    
    # 输出参数
    output_format: str = "pdb"  # pdb, mmcif
    save_msa: bool = True
    
    # 精度阈值
    min_confidence_threshold: float = 70.0
    high_confidence_threshold: float = 90.0


@dataclass
class GenomicsConfig:
    """基因组分析配置"""
    # 参考基因组
    reference_genome: str = "GRCh38"
    annotation_file: str = "gencode.v44.gtf"
    
    # 变异检测参数
    min_variant_quality: int = 30
    min_variant_reads: int = 5
    variant_allele_frequency_threshold: float = 0.05
    
    # 表达分析
    normalization_method: str = "TPM"  # TPM, FPKM, CPM
    min_gene_expression: float = 1.0
    
    # 变异分类
    pathogenicity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "benign": 0.2,
        "likely_benign": 0.4,
        "uncertain": 0.6,
        "likely_pathogenic": 0.8,
        "pathogenic": 0.9
    })


@dataclass
class DrugDiscoveryConfig:
    """药物发现配置"""
    # 分子库
    molecule_library: str = "zinc"  # zinc, chembl, pubchem, custom
    library_path: Optional[str] = None
    
    # 对接参数
    docking_engine: str = "autodock_vina"
    search_space_size: int = 1000000
    num_poses: int = 9
    exhaustiveness: int = 32
    
    # 筛选参数
    min_binding_affinity: float = -7.0  # kcal/mol
    max_rmse: float = 2.0
    min_similarity: float = 0.7
    
    # 预测参数
    use_side_effect_prediction: bool = True
    use_efficacy_prediction: bool = True


@dataclass
class CellSimulationConfig:
    """细胞模拟配置"""
    # 模拟参数
    simulation_type: str = "discrete"  # discrete, continuous, hybrid
    time_step: float = 0.001  # 秒
    max_simulation_time: float = 3600  # 秒
    
    # 细胞参数
    cell_radius: float = 10.0  # 微米
    membrane_permeability: float = 1e-6  # cm/s
    
    # 代谢参数
    default_atp_production: float = 30.0  # ATP/秒
    basal_metabolic_rate: float = 0.5
    
    # 信号通路
    max_pathway_components: int = 100
    pathway_simulation_depth: int = 10


@dataclass
class BioSimulationConfig:
    """主配置类"""
    protein_folding: ProteinFoldingConfig = field(default_factory=ProteinFoldingConfig)
    genomics: GenomicsConfig = field(default_factory=GenomicsConfig)
    drug_discovery: DrugDiscoveryConfig = field(default_factory=DrugDiscoveryConfig)
    cell_simulation: CellSimulationConfig = field(default_factory=CellSimulationConfig)
    
    # 全局设置
    num_threads: int = 4
    use_gpu: bool = True
    gpu_memory_limit: int = 16384  # MB
    cache_dir: Optional[str] = None
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "BioSimulationConfig":
        """从环境变量加载配置"""
        config = cls()
        
        # GPU设置
        if os.getenv("USE_GPU"):
            config.use_gpu = os.getenv("USE_GPU").lower() == "true"
        
        if os.getenv("GPU_MEMORY_LIMIT"):
            config.gpu_memory_limit = int(os.getenv("GPU_MEMORY_LIMIT"))
        
        # 线程数
        if os.getenv("NUM_THREADS"):
            config.num_threads = int(os.getenv("NUM_THREADS"))
        
        # 缓存目录
        config.cache_dir = os.getenv("CACHE_DIR")
        
        return config


# 默认配置实例
DEFAULT_CONFIG = BioSimulationConfig.from_env()
