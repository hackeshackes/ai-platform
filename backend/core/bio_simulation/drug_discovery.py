"""
药物发现模块 - Drug Discovery Module
=====================================
实现分子对接、药物筛选和副作用预测
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum
from datetime import datetime
import numpy as np


class MoleculeType(Enum):
    """分子类型"""
    SMALL_MOLECULE = "small_molecule"
    PEPTIDE = "peptide"
    ANTIBODY = "antibody"
    NATURAL_PRODUCT = "natural_product"


class BindingMode(Enum):
    """结合模式"""
    COVALENT = "covalent"
    NON_COVALENT = "non_covalent"
    ALLOSTERIC = "allosteric"
    ORTHOSTERIC = "orthosteric"


@dataclass
class MoleculeCandidate:
    """分子候选药物"""
    # 基本信息
    molecule_id: str
    name: str
    smiles: str
    molecule_type: MoleculeType = MoleculeType.SMALL_MOLECULE
    
    # 物理化学性质
    molecular_weight: float = 0.0
    logp: float = 0.0
    hydrogen_bond_donors: int = 0
    hydrogen_bond_acceptors: int = 0
    topological_polar_surface_area: float = 0.0
    rotatable_bonds: int = 0
    num_rings: int = 0
    fraction_sp3: float = 0.0
    
    # 药物相似性
    lipinski_rule_of_five_violations: int = 0
    druglikeness_score: float = 0.0
    
    # 结合信息
    binding_affinity: Optional[float] = None
    binding_mode: Optional[BindingMode] = None
    binding_site: Optional[str] = None
    
    # ADMET预测
    predicted_adsorption: str = "Unknown"
    predicted_distribution: str = "Unknown"
    predicted_metabolism: str = "Unknown"
    predicted_excretion: str = "Unknown"
    predicted_toxicity: str = "Unknown"
    
    # 副作用预测
    predicted_side_effects: List[str] = field(default_factory=list)
    side_effect_probabilities: Dict[str, float] = field(default_factory=dict)
    
    # 药效预测
    predicted_efficacy: float = 0.0
    efficacy_confidence: float = 0.0
    
    # 来源
    source_library: str = ""
    library_id: str = ""
    
    # 元数据
    created_at: str = ""
    score: float = 0.0
    
    @property
    def molecular_formula(self) -> str:
        formula = "C"
        formula += f"H{self.molecular_weight:.0f}" if self.molecular_weight else ""
        formula += f"O{self.hydrogen_bond_acceptors}"
        formula += f"N{self.hydrogen_bond_donors}"
        return formula
    
    @property
    def druglikeness(self) -> bool:
        violations = 0
        if self.molecular_weight > 500:
            violations += 1
        if self.logp > 5:
            violations += 1
        if self.hydrogen_bond_donors > 5:
            violations += 1
        if self.hydrogen_bond_acceptors > 10:
            violations += 1
        return violations < 2
    
    def to_sdf(self) -> str:
        lines = [
            f"{self.molecule_id}",
            f"  DrugDiscovery  {datetime.now().strftime('%Y%m%d')}",
            f"",
            f"  0  0  0  0  0  0  0  0  0  0999 V2000",
            f"    0.0000    0.0000    0.0000 O 0  0  0  0  0  0  0  0  0  0  0  0",
            f"M  END",
            f">  <NAME>",
            f"{self.name}",
            f"",
            f">  <SMILES>",
            f"{self.smiles}",
            f"",
            f">  <BINDING_AFFINITY>",
            f"{self.binding_affinity}" if self.binding_affinity else "N/A",
            f"",
            f"$$$$"
        ]
        return "\n".join(lines)


@dataclass
class DockingResult:
    """分子对接结果"""
    molecule_id: str
    molecule_name: str
    binding_energy: float
    estimated_ki: float
    docking_score: float = 0.0
    inter_score: float = 0.0
    intra_score: float = 0.0
    binding_pose: Optional[np.ndarray] = None
    binding_site_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    hydrogen_bonds: List[Dict] = field(default_factory=list)
    hydrophobic_contacts: List[Dict] = field(default_factory=list)
    pi_interactions: List[Dict] = field(default_factory=list)
    rmsd_to_reference: Optional[float] = None
    ligand_efficiency: float = 0.0
    comments: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "molecule_id": self.molecule_id,
            "molecule_name": self.molecule_name,
            "binding_energy": self.binding_energy,
            "estimated_ki": self.estimated_ki,
            "docking_score": self.docking_score,
            "hydrogen_bonds": len(self.hydrogen_bonds),
            "hydrophobic_contacts": len(self.hydrophobic_contacts),
            "ligand_efficiency": self.ligand_efficiency
        }


@dataclass
class BindingSite:
    """蛋白质结合位点"""
    site_id: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    volume: float
    residue_ids: List[str] = field(default_factory=list)
    residue_names: List[str] = field(default_factory=list)
    druggability_score: float = 0.0
    hydrophobicity: float = 0.0
    polarity: float = 0.0
    detection_method: str = ""
    protein_id: str = ""


class DrugDiscovery:
    """
    药物发现引擎
    
    提供分子对接、虚拟筛选和药物性质预测功能
    """
    
    COMMON_SIDE_EFFECTS = [
        "nausea", "headache", "dizziness", "fatigue", "insomnia",
        "diarrhea", "constipation", "rash", "dry_mouth", "blurred_vision",
        "cardiac_arrhythmia", "hepatotoxicity", "nephrotoxicity",
        "myelosuppression", "hypersensitivity"
    ]
    
    COMMON_TARGETS = [
        "EGFR", "BRCA1", "BRCA2", "TP53", "ALK", "BRAF", "KRAS",
        "HER2", "VEGFA", "TNF", "IL6", "COX2", "ACE2", "GLP1R"
    ]
    
    def __init__(self, config=None):
        self.config = config
        self.molecule_library = []
        self.target_proteins = {}
    
    def load_molecule_library(self, library: str = "zinc") -> int:
        if library == "zinc":
            self.molecule_library = self._generate_mock_library(1000, "zinc")
        elif library == "chembl":
            self.molecule_library = self._generate_mock_library(500, "chembl")
        elif library == "pubchem":
            self.molecule_library = self._generate_mock_library(2000, "pubchem")
        else:
            self.molecule_library = self._generate_mock_library(100, library)
        return len(self.molecule_library)
    
    def _generate_mock_library(self, size: int, library_name: str = "custom") -> List[MoleculeCandidate]:
        molecules = []
        for i in range(size):
            mw = np.random.uniform(200, 600)
            logp = np.random.uniform(-2, 5)
            hbd = np.random.randint(0, 5)
            hba = np.random.randint(1, 10)
            
            mol = MoleculeCandidate(
                molecule_id=f"MOL_{i:05d}",
                name=f"Compound_{i}",
                smiles=self._generate_random_smiles(),
                molecular_weight=mw,
                logp=logp,
                hydrogen_bond_donors=hbd,
                hydrogen_bond_acceptors=hba,
                topological_polar_surface_area=np.random.uniform(50, 150),
                rotatable_bonds=np.random.randint(0, 10),
                num_rings=np.random.randint(0, 5),
                druglikeness_score=np.random.uniform(0.3, 0.9),
                lipinski_rule_of_five_violations=np.random.randint(0, 2),
                source_library=library_name,
                created_at=datetime.now().isoformat()
            )
            molecules.append(mol)
        return molecules
    
    def _generate_random_smiles(self) -> str:
        atoms = ['C', 'C', 'C', 'N', 'O', 'S', 'F', 'Cl']
        bonds = ['', '=', '#']
        smiles = np.random.choice(atoms)
        for _ in range(np.random.randint(3, 20)):
            bond = np.random.choice(bonds)
            atom = np.random.choice(atoms)
            smiles += bond + atom
        return smiles
    
    def find_binding_sites(self, protein_structure: Any, min_volume: float = 100.0) -> List[BindingSite]:
        sites = []
        num_sites = np.random.randint(1, 5)
        
        for i in range(num_sites):
            site = BindingSite(
                site_id=f"site_{i+1}",
                center=(np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10)),
                size=(np.random.uniform(5, 15), np.random.uniform(5, 15), np.random.uniform(5, 15)),
                volume=np.random.uniform(min_volume, 1000),
                druggability_score=np.random.uniform(0.4, 0.9),
                hydrophobicity=np.random.uniform(0.2, 0.8),
                polarity=np.random.uniform(0.1, 0.5),
                detection_method="fpocket"
            )
            sites.append(site)
        return sorted(sites, key=lambda s: s.druggability_score, reverse=True)
    
    def dock_molecule(self, molecule: MoleculeCandidate, protein_structure: Any, 
                      binding_site: Optional[BindingSite] = None) -> DockingResult:
        binding_energy = np.random.uniform(-12, -5)
        
        if molecule.druglikeness:
            binding_energy -= np.random.uniform(0, 2)
        
        if binding_site:
            binding_energy -= binding_site.druggability_score * 3
        
        ki = np.exp(binding_energy / (0.001987 * 298)) * 1e9
        
        result = DockingResult(
            molecule_id=molecule.molecule_id,
            molecule_name=molecule.name,
            binding_energy=binding_energy,
            estimated_ki=ki,
            docking_score=-binding_energy,
            inter_score=np.random.uniform(-40, -10),
            intra_score=np.random.uniform(-5, 5),
            hydrogen_bonds=[{"donor": "ASP", "acceptor": "LIG", "distance": np.random.uniform(2.5, 3.5)}
                           for _ in range(np.random.randint(0, 4))],
            hydrophobic_contacts=[{"residue": np.random.choice(["LEU", "ILE", "VAL"]),
                                   "distance": np.random.uniform(3.5, 5.0)}
                                  for _ in range(np.random.randint(2, 8))],
            ligand_efficiency=binding_energy / max(molecule.molecular_weight / 100, 1),
            comments="Virtual screening result"
        )
        
        molecule.binding_affinity = binding_energy
        molecule.druglikeness_score = result.ligand_efficiency
        return result
    
    def screen(self, target_protein: Any, library: str = "zinc", 
               num_candidates: int = 100) -> List[MoleculeCandidate]:
        if not self.molecule_library:
            self.load_molecule_library(library)
        
        binding_sites = self.find_binding_sites(target_protein)
        
        if not binding_sites:
            binding_sites = [BindingSite(site_id="default", center=(0, 0, 0),
                                         size=(15, 15, 15), volume=500, druggability_score=0.7)]
        
        best_site = binding_sites[0]
        results = []
        
        for mol in self.molecule_library[:num_candidates]:
            docking_result = self.dock_molecule(mol, target_protein, best_site)
            if docking_result.binding_energy < -7.0:
                mol.score = docking_result.docking_score
                results.append(mol)
        
        results.sort(key=lambda m: m.score, reverse=True)
        return results[:num_candidates]
    
    def predict_side_effects(self, molecule: MoleculeCandidate) -> Dict[str, float]:
        probabilities = {}
        
        for side_effect in self.COMMON_SIDE_EFFECTS:
            prob = 0.1
            if molecule.molecular_weight > 500:
                prob += 0.05
            if molecule.molecular_weight > 700:
                prob += 0.05
            if molecule.logp > 3:
                prob += 0.03
            if molecule.logp > 5:
                prob += 0.03
            if molecule.hydrogen_bond_donors > 3:
                prob += 0.02
            if molecule.hydrogen_bond_acceptors > 7:
                prob += 0.02
            prob += np.random.uniform(-0.05, 0.05)
            probabilities[side_effect] = max(0, min(1, prob))
        
        molecule.predicted_side_effects = [
            se for se, prob in sorted(probabilities.items(), key=lambda x: -x[1])[:5]
        ]
        molecule.side_effect_probabilities = probabilities
        return probabilities
    
    def predict_efficacy(self, molecule: MoleculeCandidate, target: str) -> Tuple[float, float]:
        base_score = 0.5
        
        if molecule.binding_affinity is not None:
            if molecule.binding_affinity < -10:
                base_score += 0.3
            elif molecule.binding_affinity < -7:
                base_score += 0.1
        
        if molecule.druglikeness:
            base_score += 0.1
        
        if 1 < molecule.logp < 3:
            base_score += 0.05
        
        confidence = 0.6
        if molecule.binding_affinity is not None:
            confidence += 0.1
        if molecule.druglikeness:
            confidence += 0.1
        
        molecule.predicted_efficacy = min(1.0, base_score)
        molecule.efficacy_confidence = min(0.9, confidence)
        
        return (min(1.0, base_score), min(0.9, confidence))
    
    def analyze_lead_compounds(self, candidates: List[MoleculeCandidate]) -> Dict[str, Any]:
        """分析候选化合物"""
        if not candidates:
            return {"error": "No candidates provided"}
        
        analysis = {
            "total_candidates": len(candidates),
            "avg_binding_affinity": np.mean([c.binding_affinity for c in candidates 
                                             if c.binding_affinity]),
            "avg_molecular_weight": np.mean([c.molecular_weight for c in candidates]),
            "avg_logp": np.mean([c.logp for c in candidates]),
            "druglike_count": len([c for c in candidates if c.druglikeness]),
            "high_quality_count": len([c for c in candidates if c.druglikeness_score > 0.3]),
            "top_candidates": []
        }
        
        for mol in sorted(candidates, key=lambda m: m.score, reverse=True)[:10]:
            analysis["top_candidates"].append({
                "id": mol.molecule_id,
                "name": mol.name,
                "binding_affinity": mol.binding_affinity,
                "druglikeness": mol.druglikeness,
                "score": mol.score
            })
        
        return analysis
