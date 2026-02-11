"""
细胞模拟模块 - Cell Simulation Module
=====================================
实现细胞信号通路、代谢网络和细胞行为的模拟
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum
import numpy as np


class CellType(Enum):
    """细胞类型"""
    EUKARYOTE = "eukaryote"
    PROKARYOTE = "prokaryote"
    NEURON = "neuron"
    MUSCLE = "muscle_cell"
    IMMUNE = "immune_cell"
    STEM = "stem_cell"
    CANCER = "cancer_cell"


class CellState(Enum):
    """细胞状态"""
    ALIVE = "alive"
    DIVIDING = "dividing"
    APOPTOSIS = "apoptosis"
    NECROSIS = "necrosis"
    SENESCENT = "senescent"
    QUIESCENT = "quiescent"


class PathwayComponentType(Enum):
    """通路组件类型"""
    RECEPTOR = "receptor"
    LIGAND = "ligand"
    KINASE = "kinase"
    PHOSPHATASE = "phosphatase"
    TRANSCRIPTION_FACTOR = "transcription_factor"
    SECOND_MESSENGER = "second_messenger"
    G_PROTEIN = "g_protein"
    ADAPTOR = "adaptor"


class MetabolicReactionType(Enum):
    """代谢反应类型"""
    OXIDATION = "oxidation"
    REDUCTION = "reduction"
    HYDROLYSIS = "hydrolysis"
    SYNTHESIS = "synthesis"
    TRANSPORT = "transport"
    ISOMERIZATION = "isomerization"


@dataclass
class PathwayComponent:
    """信号通路组件"""
    component_id: str
    name: str
    component_type: PathwayComponentType
    
    concentration: float = 0.0
    phosphorylation_state: float = 0.0
    upstream_components: List[str] = field(default_factory=list)
    downstream_components: List[str] = field(default_factory=list)
    activation_rate: float = 1.0
    deactivation_rate: float = 0.1
    kd: float = 1.0
    cellular_location: str = "cytoplasm"
    gene_id: Optional[str] = None
    protein_id: Optional[str] = None
    
    def is_active(self, threshold: float = 0.5) -> bool:
        return self.concentration > threshold or self.phosphorylation_state > threshold


@dataclass
class SignalingPathway:
    """信号通路"""
    pathway_id: str
    name: str
    description: str
    components: Dict[str, PathwayComponent] = field(default_factory=dict)
    input_components: List[str] = field(default_factory=list)
    output_components: List[str] = field(default_factory=list)
    pathway_type: str = ""
    organism: str = "Homo sapiens"
    is_active: bool = True
    
    def add_component(self, component: PathwayComponent):
        self.components[component.component_id] = component
    
    def get_activated_components(self, threshold: float = 0.5) -> List[str]:
        return [cid for cid, comp in self.components.items() if comp.is_active(threshold)]
    
    def simulate(self, time_steps: int = 100, dt: float = 0.01) -> Dict[str, List[float]]:
        results = {comp.component_id: [] for comp in self.components.values()}
        
        for t in range(time_steps):
            for comp in self.components.values():
                input_signal = 0.0
                for upstream_id in comp.upstream_components:
                    if upstream_id in self.components:
                        upstream_comp = self.components[upstream_id]
                        if upstream_comp.is_active():
                            input_signal += upstream_comp.phosphorylation_state
                
                d_activation = comp.activation_rate * input_signal - comp.deactivation_rate * comp.phosphorylation_state
                comp.phosphorylation_state = max(0, min(1, comp.phosphorylation_state + d_activation * dt))
                results[comp.component_id].append(comp.phosphorylation_state)
        
        return results


@dataclass
class MetabolicReaction:
    """代谢反应"""
    reaction_id: str
    reaction_name: str
    reaction_type: MetabolicReactionType
    substrates: List[Tuple[str, float]] = field(default_factory=list)
    products: List[Tuple[str, float]] = field(default_factory=list)
    enzyme: Optional[str] = None
    enzyme_gene: Optional[str] = None
    kcat: float = 10.0
    km: float = 1.0
    location: str = "cytoplasm"
    delta_g: float = 0.0
    atp_cost: float = 0.0
    
    def calculate_rate(self, substrate_concentrations: Dict[str, float]) -> float:
        if not substrate_concentrations:
            return 0.0
        v = self.kcat
        for substrate, coef in self.substrates:
            conc = substrate_concentrations.get(substrate, 0)
            v *= conc / (self.km + conc)
        return v


@dataclass
class Metabolite:
    """代谢物"""
    metabolite_id: str
    name: str
    formula: str
    initial_concentration: float = 0.0
    current_concentration: float = 0.0
    molecular_weight: float = 0.0
    charge: int = 0
    is_variable: bool = True
    compartment: str = "cytoplasm"
    
    def update(self, delta: float, dt: float):
        self.current_concentration = max(0, self.current_concentration + delta * dt)


@dataclass
class MetabolicNetwork:
    """代谢网络"""
    network_id: str
    name: str
    metabolites: Dict[str, Metabolite] = field(default_factory=dict)
    reactions: Dict[str, MetabolicReaction] = field(default_factory=dict)
    metabolite_reactions: Dict[str, List[str]] = field(default_factory=dict)
    atp_concentration: float = 5.0
    nadh_concentration: float = 0.5
    nad_concentration: float = 1.0
    
    def add_metabolite(self, metabolite: Metabolite):
        self.metabolites[metabolite.metabolite_id] = metabolite
        self.current_concentration = metabolite.initial_concentration
    
    def add_reaction(self, reaction: MetabolicReaction):
        self.reactions[reaction.reaction_id] = reaction
        for substrate, _ in reaction.substrates:
            if substrate not in self.metabolite_reactions:
                self.metabolite_reactions[substrate] = []
            self.metabolite_reactions[substrate].append(reaction.reaction_id)
        for product, _ in reaction.products:
            if product not in self.metabolite_reactions:
                self.metabolite_reactions[product] = []
            self.metabolite_reactions[product].append(reaction.reaction_id)
    
    def simulate(self, time_steps: int = 1000, dt: float = 0.001) -> Dict[str, List[float]]:
        results = {mid: [] for mid in self.metabolites}
        
        for t in range(time_steps):
            reaction_rates = {}
            substrate_concs = {mid: m.current_concentration for mid, m in self.metabolites.items()}
            
            for rid, reaction in self.reactions.items():
                rate = reaction.calculate_rate(substrate_concs)
                reaction_rates[rid] = rate
            
            for rid, reaction in self.reactions.items():
                rate = reaction_rates[rid]
                for substrate, coef in reaction.substrates:
                    if substrate in self.metabolites:
                        self.metabolites[substrate].update(-coef * rate, dt)
                for product, coef in reaction.products:
                    if product in self.metabolites:
                        self.metabolites[product].update(coef * rate, dt)
            
            for mid, m in self.metabolites.items():
                results[mid].append(m.current_concentration)
        
        return results


@dataclass
class Cell:
    """单个细胞"""
    cell_id: str
    cell_type: CellType
    state: CellState = CellState.ALIVE
    age: float = 0.0
    generation: int = 0
    radius: float = 10.0
    volume: float = 1000.0
    membrane_potential: float = -70.0
    nucleus_size: float = 0.3
    mitochondria_count: int = 100
    ribosome_count: int = 10000
    atp: float = 5.0
    nadh: float = 0.5
    ros_level: float = 0.1
    signaling_pathways: Dict[str, SignalingPathway] = field(default_factory=dict)
    metabolic_network: Optional[MetabolicNetwork] = None
    chromosome_count: int = 46
    gene_expression_levels: Dict[str, float] = field(default_factory=dict)
    
    def update(self, dt: float):
        self.age += dt
        basal_atp_consumption = 0.5 * self.volume * dt
        self.atp = max(0, self.atp - basal_atp_consumption / 10)
        if self.ros_level > 0.5:
            self.state = CellState.APOPTOSIS
        if self.atp > 3.0 and np.random.random() < 0.01 * dt:
            self._divide()
    
    def _divide(self):
        self.generation += 1
        self.state = CellState.DIVIDING
        self.atp /= 2
        self.radius *= 0.8 ** (1/3)
        self.state = CellState.ALIVE
    
    def receive_signal(self, pathway_id: str, signal_strength: float):
        if pathway_id in self.signaling_pathways:
            pathway = self.signaling_pathways[pathway_id]
            results = pathway.simulate(time_steps=10)
            output_activation = np.mean([results[comp][-1] for comp in pathway.output_components if comp in results])
            if output_activation > 0.7:
                self.atp += signal_strength * 0.1
    
    def is_alive(self) -> bool:
        return self.state in [CellState.ALIVE, CellState.DIVIDING, CellState.QUIESCENT]


class CellSimulation:
    """细胞模拟器"""
    
    COMMON_PATHWAYS = {
        "MAPK": ["RAF", "MEK", "ERK", "ELK1"],
        "PI3K_AKT": ["PI3K", "AKT", "MTOR", "FOXO"],
        "JAK_STAT": ["JAK", "STAT", "SOCS"],
        "WNT": ["WNT", "FZD", "DKK", "beta-catenin", "TCF"]
    }
    
    def __init__(self, config=None):
        self.config = config
        self.cells: List[Cell] = []
        self.time = 0.0
    
    def create_cell(self, cell_type: CellType = CellType.EUKARYOTE) -> Cell:
        cell = Cell(
            cell_id=f"cell_{len(self.cells)}",
            cell_type=cell_type,
            radius=np.random.uniform(8, 15),
            atp=np.random.uniform(4, 6),
            metabolic_network=self._create_default_metabolism()
        )
        for pathway_name, components in self.COMMON_PATHWAYS.items():
            cell.signaling_pathways[pathway_name] = self._create_signaling_pathway(pathway_name, components)
        self.cells.append(cell)
        return cell
    
    def _create_signaling_pathway(self, name: str, components: List[str]) -> SignalingPathway:
        pathway = SignalingPathway(pathway_id=name.lower(), name=name, description=f"{name} signaling pathway", pathway_type=name)
        prev_comp = None
        for i, comp_name in enumerate(components):
            comp_type = PathwayComponentType.KINASE
            if i == 0:
                comp_type = PathwayComponentType.RECEPTOR
            elif i == len(components) - 1:
                comp_type = PathwayComponentType.TRANSCRIPTION_FACTOR
            comp = PathwayComponent(component_id=comp_name.lower(), name=comp_name, component_type=comp_type, concentration=np.random.uniform(0.1, 1.0))
            if prev_comp:
                comp.upstream_components.append(prev_comp.component_id)
                prev_comp.downstream_components.append(comp.component_id)
            pathway.add_component(comp)
            prev_comp = comp
        if prev_comp:
            pathway.output_components.append(prev_comp.component_id)
        return pathway
    
    def _create_default_metabolism(self) -> MetabolicNetwork:
        network = MetabolicNetwork(network_id="central_metabolism", name="Central Carbon Metabolism")
        metabolites = [
            ("glucose", "Glucose", "C6H12O6", 5.0),
            ("g6p", "Glucose-6-Phosphate", "C6H13O9P", 0.0),
            ("f6p", "Fructose-6-Phosphate", "C6H13O9P", 0.0),
            ("f16bp", "Fructose-1,6-Bisphosphate", "C6H14O12P2", 0.0),
            ("atp", "ATP", "C10H16N5O13P3", 5.0),
            ("adp", "ADP", "C10H15N5O10P2", 0.0),
            ("nad", "NAD+", "C21H27N7O14P2", 1.0),
            ("nadh", "NADH", "C21H28N7O14P2", 0.5),
            ("pyruvate", "Pyruvate", "C3H4O3", 0.0),
            ("lactate", "Lactate", "C3H6O3", 0.0)
        ]
        for mid, name, formula, conc in metabolites:
            met = Metabolite(metabolite_id=mid, name=name, formula=formula, initial_concentration=conc, molecular_weight=180.0)
            network.add_metabolite(met)
        reactions = [
            ("hexokinase", "Hexokinase", MetabolicReactionType.SYNTHESIS, [("glucose", 1), ("atp", 1)], [("g6p", 1), ("adp", 1)]),
            ("pgi", "Phosphoglucose Isomerase", MetabolicReactionType.ISOMERIZATION, [("g6p", 1)], [("f6p", 1)]),
            ("pfk", "Phosphofructokinase", MetabolicReactionType.SYNTHESIS, [("f6p", 1), ("atp", 1)], [("f16bp", 1), ("adp", 1)]),
            ("pk", "Pyruvate Kinase", MetabolicReactionType.SYNTHESIS, [("f16bp", 1), ("adp", 1)], [("pyruvate", 1), ("atp", 1)]),
            ("ldh", "Lactate Dehydrogenase", MetabolicReactionType.REDUCTION, [("pyruvate", 1), ("nadh", 1)], [("lactate", 1), ("nad", 1)])
        ]
        for rid, name, rtype, substrates, products in reactions:
            rxn = MetabolicReaction(reaction_id=rid, reaction_name=name, reaction_type=rtype, substrates=substrates, products=products, kcat=np.random.uniform(5, 20), km=np.random.uniform(0.5, 2.0))
            network.add_reaction(rxn)
        return network
    
    def simulate(self, duration: float = 10.0, dt: float = 0.01) -> Dict[str, Any]:
        time_steps = int(duration / dt)
        results = {"time_points": [], "cell_states": [], "atp_levels": [], "ros_levels": [], "division_events": []}
        
        for t in range(time_steps):
            current_time = t * dt
            self.time = current_time
            for cell in self.cells:
                cell.update(dt)
                if t % 100 == 0:
                    results["time_points"].append(current_time)
                    results["cell_states"].append(cell.state.value)
                    results["atp_levels"].append(cell.atp)
                    results["ros_levels"].append(cell.ros_level)
                if cell.state == CellState.DIVIDING:
                    results["division_events"].append({"time": current_time, "cell": cell.cell_id, "generation": cell.generation})
        
        results["summary"] = {
            "total_cells": len(self.cells),
            "alive_cells": len([c for c in self.cells if c.is_alive()]),
            "total_divisions": sum(c.generation for c in self.cells),
            "final_atp": np.mean([c.atp for c in self.cells]),
            "final_ros": np.mean([c.ros_level for c in self.cells])
        }
        return results
