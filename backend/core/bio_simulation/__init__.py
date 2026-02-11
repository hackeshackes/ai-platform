"""
生物模拟系统 - Bio Simulation System
=====================================
提供蛋白质折叠、基因组分析、药物发现和细胞模拟功能

模块:
- ProteinFolding: 蛋白质3D结构预测
- GenomicsAnalyzer: 基因组序列分析和突变检测
- DrugDiscovery: 药物筛选和分子对接
- CellSimulation: 细胞信号通路和代谢网络模拟

作者: AI Platform Team
版本: 1.0.0
"""

from .protein_folding import ProteinFolding, ProteinStructure
from .genomics import GenomicsAnalyzer, GenomicVariant
from .drug_discovery import DrugDiscovery, MoleculeCandidate
from .cell_simulation import CellSimulation, SignalingPathway

__version__ = "1.0.0"
__all__ = [
    'ProteinFolding',
    'ProteinStructure',
    'GenomicsAnalyzer',
    'GenomicVariant',
    'DrugDiscovery',
    'MoleculeCandidate',
    'CellSimulation',
    'SignalingPathway',
]
