"""
基因组分析模块 - Genomics Module
================================
实现基因组序列分析、突变检测和变异分类
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum
import numpy as np
import hashlib
import re


class VariantType(Enum):
    """变异类型"""
    SNV = "SNV"
    MNV = "MNV"
    INSERTION = "INS"
    DELETION = "DEL"
    INVERSION = "INV"
    DUPLICATION = "DUP"
    CNV = "CNV"
    STR = "STR"


class VariantEffect(Enum):
    """变异效应"""
    SYNONYMOUS = "synonymous_variant"
    MISSENSE = "missense_variant"
    NONSENSE = "stop_gained"
    FRAMESHIFT = "frameshift_variant"
    SPLICE_SITE = "splice_site_variant"
    INTRONIC = "intronic_variant"
    UPSTREAM = "upstream_gene_variant"
    DOWNSTREAM = "downstream_gene_variant"
    INTERGENIC = "intergenic_variant"
    REGULATORY = "regulatory_region_variant"
    INFRAME_INSERTION = "inframe_insertion"
    INFRAME_DELETION = "inframe_deletion"


class PathogenicityClass(Enum):
    """致病性分类"""
    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely Pathogenic"
    UNCERTAIN = "Uncertain Significance"
    LIKELY_BENIGN = "Likely Benign"
    BENIGN = "Benign"


@dataclass
class GenomicVariant:
    """基因组变异"""
    chromosome: str
    position: int
    reference: str
    alternate: str
    variant_type: VariantType
    effect: VariantEffect
    quality: float = 0.0
    read_depth: int = 0
    allele_frequency: float = 0.0
    pathogenicity_score: float = 0.0
    pathogenicity_class: Optional[PathogenicityClass] = None
    gene: Optional[str] = None
    transcript: Optional[str] = None
    protein_change: Optional[str] = None
    codon_change: Optional[str] = None
    gnomad_af: Optional[float] = None
    thousand_genomes_af: Optional[float] = None
    filter_status: str = "PASS"
    raw_data: Dict = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        return f"{self.chromosome}:{self.position}:{self.reference}:{self.alternate}"
    
    @property
    def length(self) -> int:
        return max(len(self.reference), len(self.alternate))
    
    def to_vcf_record(self) -> str:
        return (
            f"{self.chromosome}\t{self.position}\t.\t"
            f"{self.reference}\t{self.alternate}\t"
            f"{self.quality:.1f}\t{self.filter_status}\t"
            f"AF={self.allele_frequency:.4f};DP={self.read_depth}"
        )
    
    def is_transition(self) -> bool:
        if len(self.reference) != 1 or len(self.alternate) != 1:
            return False
        transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        return (self.reference, self.alternate) in transitions


@dataclass
class Gene:
    """基因"""
    name: str
    chromosome: str
    start: int
    end: int
    strand: str = "+"
    gene_type: str = "protein_coding"
    description: str = ""
    
    @property
    def length(self) -> int:
        return self.end - self.start + 1
    
    def contains_position(self, position: int) -> bool:
        return self.start <= position <= self.end


@dataclass
class Transcript:
    """转录本"""
    transcript_id: str
    gene_name: str
    chromosome: str
    start: int
    end: int
    strand: str
    exons: List[Tuple[int, int]] = field(default_factory=list)
    coding_start: Optional[int] = None
    coding_end: Optional[int] = None
    
    def get_cds_regions(self) -> List[Tuple[int, int]]:
        if self.coding_start is None or self.coding_end is None:
            return []
        cds_regions = []
        for exon_start, exon_end in self.exons:
            if exon_end < self.coding_start or exon_start > self.coding_end:
                continue
            cds_start = max(exon_start, self.coding_start)
            cds_end = min(exon_end, self.coding_end)
            if cds_start <= cds_end:
                cds_regions.append((cds_start, cds_end))
        return cds_regions


@dataclass
class ExpressionData:
    """基因表达数据"""
    gene_id: str
    gene_name: str
    expression_level: float
    normalized_expression: float
    expression_unit: str = "TPM"
    sample_id: str = ""
    
    def to_log2(self, pseudo_count: float = 1.0) -> float:
        return np.log2(self.expression_level + pseudo_count)


class GenomicsAnalyzer:
    """基因组分析器"""
    
    CODON_TABLE = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }
    
    def __init__(self, config=None):
        self.config = config
        self.reference_genome = {}
        self.gene_annotation = {}
    
    def load_reference_genome(self, fasta_path: str):
        sequence = ""
        chromosome = ""
        with open(fasta_path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    if chromosome:
                        self.reference_genome[chromosome] = sequence.upper()
                    chromosome = line[1:].strip().split()[0]
                    sequence = ""
                else:
                    sequence += line.strip().upper()
        if chromosome:
            self.reference_genome[chromosome] = sequence.upper()
    
    def detect_variants(self, sequence: str, reference: Optional[str] = None, 
                        min_quality: int = 30) -> List[GenomicVariant]:
        """检测序列中的变异"""
        variants = []
        if reference is None:
            return self._simulate_variant_detection(sequence, min_quality)
        
        for i, (ref_base, alt_base) in enumerate(zip(reference, sequence)):
            if ref_base != alt_base and alt_base in 'ACGT':
                quality = np.random.uniform(40, 99)
                if quality >= min_quality:
                    variant = GenomicVariant(
                        chromosome="1", position=i + 1,
                        reference=ref_base, alternate=alt_base,
                        variant_type=VariantType.SNV,
                        effect=self._predict_variant_effect(ref_base, alt_base),
                        quality=quality,
                        read_depth=np.random.randint(10, 100),
                        allele_frequency=np.random.uniform(0.3, 1.0),
                        gnomad_af=np.random.uniform(0, 0.01) if np.random.random() > 0.7 else None
                    )
                    variants.append(variant)
        return variants
    
    def _simulate_variant_detection(self, sequence: str, min_quality: int) -> List[GenomicVariant]:
        variants = []
        # 确保至少有一个变异，且不超过10个
        max_variants = max(1, len(sequence) // 100)
        num_variants = np.random.randint(1, min(10, max_variants + 1))
        
        for _ in range(num_variants):
            position = np.random.randint(1, len(sequence))
            ref_base = sequence[position - 1]
            alt_base = np.random.choice([b for b in 'ACGT' if b != ref_base])
            quality = np.random.uniform(min_quality, 99)
            
            variant = GenomicVariant(
                chromosome="1", position=position,
                reference=ref_base, alternate=alt_base,
                variant_type=VariantType.SNV,
                effect=self._predict_variant_effect(ref_base, alt_base),
                quality=quality,
                read_depth=np.random.randint(10, 100),
                allele_frequency=np.random.uniform(0.3, 1.0),
                gnomad_af=np.random.uniform(0, 0.01) if np.random.random() > 0.7 else None,
                gene=self._predict_affected_gene(position)
            )
            variants.append(variant)
        return sorted(variants, key=lambda v: v.position)
    
    def _predict_variant_effect(self, ref: str, alt: str) -> VariantEffect:
        return VariantEffect.INTRONIC
    
    def _predict_affected_gene(self, position: int) -> Optional[str]:
        genes = {'BRCA1': (43000001, 43125482), 'TP53': (7661779, 7687550), 'EGFR': (55000000, 55270000)}
        for gene, (start, end) in genes.items():
            if start <= position <= end:
                return gene
        return None
    
    def translate_sequence(self, sequence: str) -> str:
        protein = ""
        sequence = sequence.upper()
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            if codon in self.CODON_TABLE:
                aa = self.CODON_TABLE[codon]
                if aa == '*':
                    break
                protein += aa
        return protein
    
    def predict_protein_change(self, ref_codon: str, alt_codon: str) -> Tuple[str, str]:
        ref_aa = self.CODON_TABLE.get(ref_codon.upper(), 'X')
        alt_aa = self.CODON_TABLE.get(alt_codon.upper(), 'X')
        position = 1
        return f"{ref_aa}{position}{alt_aa}", f"{ref_codon}>{alt_codon}"
    
    def analyze_expression(self, expression_data: List[Dict[str, Any]], 
                           method: str = "TPM") -> List[ExpressionData]:
        results = []
        total_counts = sum(item.get('count', 0) for item in expression_data)
        
        for item in expression_data:
            raw_count = item.get('count', 0)
            
            if method == "TPM":
                normalized = (raw_count / total_counts) * 1e6 if total_counts > 0 else 0
            elif method == "FPKM":
                gene_length = item.get('length', 1000)
                normalized = (raw_count / total_counts) * 1e9 / gene_length if total_counts > 0 else 0
            elif method == "CPM":
                normalized = (raw_count / total_counts) * 1e6 if total_counts > 0 else 0
            else:
                normalized = raw_count
            
            exp_data = ExpressionData(
                gene_id=item.get('gene_id', ''),
                gene_name=item.get('gene_name', ''),
                expression_level=raw_count,
                normalized_expression=normalized,
                expression_unit=method,
                sample_id=item.get('sample_id', '')
            )
            results.append(exp_data)
        return results
    
    def classify_pathogenicity(self, variant: GenomicVariant) -> PathogenicityClass:
        score = 0.0
        if variant.variant_type == VariantType.SNV:
            if variant.effect == VariantEffect.NONSENSE:
                score += 0.4
            elif variant.effect == VariantEffect.FRAMESHIFT:
                score += 0.4
            elif variant.effect == VariantEffect.MISSENSE:
                score += 0.1
        
        if variant.gnomad_af is not None:
            if variant.gnomad_af < 0.0001:
                score += 0.3
            elif variant.gnomad_af < 0.01:
                score += 0.1
            elif variant.gnomad_af > 0.05:
                score -= 0.3
        
        if variant.quality > 50:
            score += 0.1
        
        if score >= 0.8:
            return PathogenicityClass.PATHOGENIC
        elif score >= 0.6:
            return PathogenicityClass.LIKELY_PATHOGENIC
        elif score >= 0.4:
            return PathogenicityClass.UNCERTAIN
        elif score >= 0.2:
            return PathogenicityClass.LIKELY_BENIGN
        else:
            return PathogenicityClass.BENIGN
    
    def calculate_variant_metrics(self, variants: List[GenomicVariant]) -> Dict[str, Any]:
        total = len(variants)
        if total == 0:
            return {"error": "No variants provided"}
        
        type_counts = {}
        effect_counts = {}
        transition_count = 0
        
        for v in variants:
            type_counts[v.variant_type.value] = type_counts.get(v.variant_type.value, 0) + 1
            effect_counts[v.effect.value] = effect_counts.get(v.effect.value, 0) + 1
            if v.is_transition():
                transition_count += 1
        
        ti_tv_ratio = transition_count / (total - transition_count) if total > transition_count else float('inf')
        
        return {
            "total_variants": total,
            "variant_type_counts": type_counts,
            "effect_counts": effect_counts,
            "transitions": transition_count,
            "transversions": total - transition_count,
            "ti_tv_ratio": ti_tv_ratio,
            "avg_quality": np.mean([v.quality for v in variants]),
            "high_quality_count": len([v for v in variants if v.quality > 50])
        }
    
    def find_motifs(self, sequence: str, motif_pattern: str) -> List[Dict[str, Any]]:
        matches = re.finditer(motif_pattern, sequence.upper())
        results = []
        for i, match in enumerate(matches):
            results.append({
                "motif_id": f"motif_{i+1}",
                "match": match.group(),
                "start": match.start() + 1,
                "end": match.end()
            })
        return results
    
    def calculate_gc_content(self, sequence: str) -> float:
        if not sequence:
            return 0.0
        gc_count = sequence.upper().count('G') + sequence.upper().count('C')
        return gc_count / len(sequence)
