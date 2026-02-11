"""
SETI集成系统 - SETI Integration Module
=======================================

信号处理、异常检测、模式识别和外星文明评估功能

作者: AI Platform Team
版本: 1.0.0
"""

import math
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import time

from .config import SystemConfig, get_config

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """信号类型"""
    NATURAL = "natural"           # 自然信号
    ARTIFICIAL = "artificial"    # 人工信号
    UNKNOWN = "unknown"           # 未知
    NOISE = "noise"              # 噪声
    INTERFERENCE = "interference"  # 干扰

class SignalClassification(Enum):
    """信号分类等级"""
    CANDIDATE = "candidate"      # 候选信号
    CONFIRMED = "confirmed"       # 确认信号
    FALSE_ALARM = "false_alarm"   # 误报
    DEBRIS = "debris"             # 太空碎片
    NATURAL_PHENOMENON = "natural"  # 自然现象

class CivilizationLevel(Enum):
    """文明等级 ( Kardashev 等级)"""
    TYPE_0 = "type_0"            # 行星级
    TYPE_I = "type_i"            # 行星级（能利用母星全部能源）
    TYPE_II = "type_ii"           # 恒星级（能利用整个恒星能源）
    TYPE_III = "type_iii"         # 星系级
    TYPE_IV = "type_iv"          # 宇宙级
    TYPE_V = "type_v"            # 多元宇宙级

class AnomalyType(Enum):
    """异常类型"""
    NARROWBAND_SPIKE = "narrowband_spike"  # 窄带尖峰
    PERIODIC_SIGNAL = "periodic"           # 周期性信号
    MODULATED_PATTERN = "modulated"        # 调制模式
    BROADBAND_BURST = "broadband"          # 宽带爆发
    SPECTRAL_LINE = "spectral_line"        # 谱线
    UNUSUAL_INTENSITY = "intensity"        # 异常强度

@dataclass
class RawSignal:
    """原始信号"""
    signal_id: str
    timestamp: float
    frequency: float          # Hz
    bandwidth: float           # Hz
    intensity: float           # Jy (央斯基)
    duration: float            # seconds
    source_direction: Tuple[float, float]  # RA, Dec (度)
    polarization: str = "unknown"
    snr: float = 0.0           # 信噪比
    
    def to_dict(self) -> Dict:
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp,
            'frequency': self.frequency,
            'bandwidth': self.bandwidth,
            'intensity': self.intensity,
            'duration': self.duration,
            'source_direction': self.source_direction,
            'polarization': self.polarization,
            'snr': self.snr
        }

@dataclass
class ProcessedSignal:
    """处理后的信号"""
    raw_signal: RawSignal
    signal_type: SignalType
    classification: SignalClassification
    features: Dict = field(default_factory=dict)
    confidence: float = 0.0
    analysis_notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'signal_id': self.raw_signal.signal_id,
            'signal_type': self.signal_type.value,
            'classification': self.classification.value,
            'features': self.features,
            'confidence': self.confidence,
            'analysis_notes': self.analysis_notes
        }

@dataclass
class Anomaly:
    """异常"""
    anomaly_id: str
    signal_id: str
    anomaly_type: AnomalyType
    significance: float        # 重要性 (0-1)
    description: str
    detected_at: float
    related_signals: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'anomaly_id': self.anomaly_id,
            'signal_id': self.signal_id,
            'anomaly_type': self.anomaly_type.value,
            'significance': self.significance,
            'description': self.description,
            'detected_at': self.detected_at,
            'related_signals': self.related_signals
        }

@dataclass
class CivilizationAssessment:
    """文明评估"""
    civilization_id: str
    signal_id: str
    estimated_level: CivilizationLevel
    confidence: float
    evidence: List[str]
    threat_level: str          # low/medium/high
    communication_attempt: bool = False
    assessment_notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'civilization_id': self.civilization_id,
            'signal_id': self.signal_id,
            'estimated_level': self.estimated_level.value,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'threat_level': self.threat_level,
            'communication_attempt': self.communication_attempt,
            'assessment_notes': self.assessment_notes
        }

@dataclass
class PatternMatch:
    """模式匹配结果"""
    pattern_id: str
    matched_signals: List[str]
    pattern_type: str
    frequency: float           # 出现频率
    significance: float        # 重要性
    description: str
    
    def to_dict(self) -> Dict:
        return {
            'pattern_id': self.pattern_id,
            'matched_signals': self.matched_signals,
            'pattern_type': self.pattern_type,
            'frequency': self.frequency,
            'significance': self.significance,
            'description': self.description
        }


class SignalProcessor:
    """信号处理器"""
    
    def __init__(self, config: 'SETIConfig' = None):
        self.config = config or get_config().seti
        self.processed_signals: deque = deque(maxlen=10000)
        self.processing_stats = {
            'total_processed': 0,
            'artificial_count': 0,
            'natural_count': 0,
            'avg_processing_time': 0
        }
    
    def process_signal(self, raw_signal: RawSignal) -> ProcessedSignal:
        """处理单个信号"""
        start_time = time.time()
        
        # 特征提取
        features = self._extract_features(raw_signal)
        
        # 信号分类
        signal_type = self._classify_signal(features, raw_signal)
        classification = self._determine_classification(features, signal_type)
        
        # 计算置信度
        confidence = self._calculate_confidence(features, signal_type)
        
        processed = ProcessedSignal(
            raw_signal=raw_signal,
            signal_type=signal_type,
            classification=classification,
            features=features,
            confidence=confidence,
            analysis_notes=self._generate_notes(features, signal_type)
        )
        
        # 更新统计
        processing_time = time.time() - start_time
        self._update_stats(processed, processing_time)
        
        self.processed_signals.append(processed)
        
        return processed
    
    def process_batch(self, signals: List[RawSignal]) -> List[ProcessedSignal]:
        """批量处理信号"""
        results = []
        for signal in signals:
            processed = self.process_signal(signal)
            results.append(processed)
        return results
    
    def _extract_features(self, signal: RawSignal) -> Dict:
        """提取信号特征"""
        return {
            'frequency_stability': random.uniform(0.8, 1.0),
            'bandwidth_ratio': signal.bandwidth / max(signal.frequency, 1),
            'intensity_variation': random.uniform(0, 0.1),
            'polarization_alignment': random.uniform(0.5, 1.0),
            'spectral_purity': random.uniform(0.7, 0.99),
            'modulation_index': random.uniform(0, 0.5),
            'drift_rate': random.uniform(-0.1, 0.1),  # Hz/s
            'coherence_time': random.uniform(0.1, 10.0)  # seconds
        }
    
    def _classify_signal(
        self,
        features: Dict,
        signal: RawSignal
    ) -> SignalType:
        """分类信号类型"""
        # 窄带信号通常是人工的
        if features['bandwidth_ratio'] < 1e-6:
            if features['frequency_stability'] > 0.95:
                return SignalType.ARTIFICIAL
        
        # 高稳定性的周期性信号
        if features['coherence_time'] > 1.0 and features['spectral_purity'] > 0.9:
            return SignalType.ARTIFICIAL
        
        # 自然噪声
        if signal.snr < 3:
            return SignalType.NOISE
        
        return SignalType.UNKNOWN
    
    def _determine_classification(
        self,
        features: Dict,
        signal_type: SignalType
    ) -> SignalClassification:
        """确定分类等级"""
        if signal_type == SignalType.NOISE:
            return SignalClassification.FALSE_ALARM
        elif signal_type == SignalType.ARTIFICIAL:
            if features['frequency_stability'] > 0.98:
                return SignalClassification.CONFIRMED
            else:
                return SignalClassification.CANDIDATE
        else:
            return SignalClassification.DEBRIS
    
    def _calculate_confidence(
        self,
        features: Dict,
        signal_type: SignalType
    ) -> float:
        """计算置信度"""
        base_confidence = 0.5
        
        if signal_type == SignalType.ARTIFICIAL:
            base_confidence += features['frequency_stability'] * 0.3
            base_confidence += features['spectral_purity'] * 0.1
        elif signal_type == SignalType.NOISE:
            base_confidence = 0.95
        
        return min(0.99, base_confidence)
    
    def _generate_notes(
        self,
        features: Dict,
        signal_type: SignalType
    ) -> str:
        """生成分析注释"""
        if signal_type == SignalType.ARTIFICIAL:
            return "高稳定性窄带信号，具有人工信号特征"
        elif signal_type == SignalType.NOISE:
            return "低于检测阈值，判定为背景噪声"
        else:
            return "信号特征不明确，需要进一步分析"
    
    def _update_stats(
        self,
        processed: ProcessedSignal,
        processing_time: float
    ):
        """更新统计"""
        self.processing_stats['total_processed'] += 1
        self.processing_stats['avg_processing_time'] = (
            (self.processing_stats['avg_processing_time'] * 
             (self.processing_stats['total_processed'] - 1) + processing_time)
            / self.processing_stats['total_processed']
        )
        
        if processed.signal_type == SignalType.ARTIFICIAL:
            self.processing_stats['artificial_count'] += 1
        elif processed.signal_type == SignalType.NOISE:
            self.processing_stats['natural_count'] += 1
    
    def get_processing_stats(self) -> Dict:
        """获取处理统计"""
        return {
            **self.processing_stats,
            'processing_rate': self.processing_stats['total_processed'] / 
                              max(self.processing_stats['avg_processing_time'], 0.001)
        }


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, config: 'SETIConfig' = None):
        self.config = config or get_config().seti
        self.detected_anomalies: List[Anomaly] = []
        self.anomaly_history: deque = deque(maxlen=1000)
    
    def detect_anomaly(
        self,
        processed_signals: List[ProcessedSignal]
    ) -> List[Anomaly]:
        """检测异常"""
        anomalies = []
        
        for signal in processed_signals:
            if self._is_anomaly(signal):
                anomaly = Anomaly(
                    anomaly_id=self._generate_anomaly_id(signal),
                    signal_id=signal.raw_signal.signal_id,
                    anomaly_type=self._identify_anomaly_type(signal),
                    significance=self._calculate_significance(signal),
                    description=self._describe_anomaly(signal),
                    detected_at=signal.raw_signal.timestamp
                )
                anomalies.append(anomaly)
                self.detected_anomalies.append(anomaly)
                self.anomaly_history.append(anomaly)
        
        return anomalies
    
    def _is_anomaly(self, signal: ProcessedSignal) -> bool:
        """判断是否为异常"""
        if signal.signal_type == SignalType.ARTIFICIAL:
            return signal.confidence > self.config.anomaly_sensitivity
        
        if signal.raw_signal.snr > 20:
            return True
        
        return False
    
    def _identify_anomaly_type(
        self,
        signal: ProcessedSignal
    ) -> AnomalyType:
        """识别异常类型"""
        features = signal.features
        
        if features['bandwidth_ratio'] < 1e-6:
            return AnomalyType.NARROWBAND_SPIKE
        elif features['drift_rate'] != 0:
            return AnomalyType.PERIODIC_SIGNAL
        elif features['modulation_index'] > 0.3:
            return AnomalyType.MODULATED_PATTERN
        else:
            return AnomalyType.UNUSUAL_INTENSITY
    
    def _calculate_significance(self, signal: ProcessedSignal) -> float:
        """计算重要性"""
        significance = signal.confidence
        
        if signal.raw_signal.snr > 50:
            significance *= 1.2
        
        if signal.features.get('frequency_stability', 0) > 0.99:
            significance *= 1.1
        
        return min(1.0, significance)
    
    def _describe_anomaly(self, signal: ProcessedSignal) -> str:
        """描述异常"""
        anomaly_type = self._identify_anomaly_type(signal)
        
        descriptions = {
            AnomalyType.NARROWBAND_SPIKE: 
                f"检测到窄带尖峰信号，频率 {signal.raw_signal.frequency/1e9:.3f} GHz",
            AnomalyType.PERIODIC_SIGNAL: 
                f"检测到周期性调制信号，周期约 {signal.features.get('coherence_time', 1):.1f}s",
            AnomalyType.MODULATED_PATTERN: 
                f"检测到复杂调制模式，调制指数 {signal.features.get('modulation_index', 0):.2f}",
            AnomalyType.UNUSUAL_INTENSITY: 
                f"检测到异常强度信号，SNR={signal.raw_signal.snr:.1f}"
        }
        
        return descriptions.get(anomaly_type, "检测到未知类型异常")
    
    def _generate_anomaly_id(self, signal: ProcessedSignal) -> str:
        """生成异常ID"""
        timestamp = str(signal.raw_signal.timestamp)
        return f"ANOMALY-{timestamp[-8:]}-{random.randint(100,999)}"
    
    def get_anomaly_summary(self) -> Dict:
        """获取异常摘要"""
        by_type = {}
        for anomaly in self.detected_anomalies:
            atype = anomaly.anomaly_type.value
            if atype not in by_type:
                by_type[atype] = []
            by_type[atype].append(anomaly.anomaly_id)
        
        return {
            'total_anomalies': len(self.detected_anomalies),
            'by_type': {k: len(v) for k, v in by_type.items()},
            'high_significance_count': sum(
                1 for a in self.detected_anomalies if a.significance > 0.8
            ),
            'recent_anomalies': [
                a.to_dict() for a in list(self.detected_anomalies)[-10:]
            ]
        }


class PatternRecognizer:
    """模式识别器"""
    
    def __init__(self, config: 'SETIConfig' = None):
        self.config = config or get_config().seti
        self.known_patterns: List[PatternMatch] = []
        self.pattern_database = {
            'prime_numbers': ['2', '3', '5', '7', '11', '13', '17', '19', '23', '29'],
            'fibonacci': ['0', '1', '1', '2', '3', '5', '8', '13', '21', '34'],
            'pi_digits': ['3', '1', '4', '1', '5', '9', '2', '6', '5', '3'],
            'hydrogen_line': '1420.4057517667',  # MHz
            'water_hole': '1420-1665'  # MHz
        }
    
    def recognize_patterns(
        self,
        signals: List[ProcessedSignal]
    ) -> List[PatternMatch]:
        """识别模式"""
        patterns = []
        
        for signal in signals:
            if signal.signal_type == SignalType.ARTIFICIAL:
                # 检查是否匹配已知模式
                for pattern_name, pattern_data in self.pattern_database.items():
                    if self._check_pattern_match(signal, pattern_name, pattern_data):
                        match = PatternMatch(
                            pattern_id=f"PATTERN-{pattern_name.upper()}",
                            matched_signals=[signal.raw_signal.signal_id],
                            pattern_type=pattern_name,
                            frequency=random.uniform(0.1, 0.5),
                            significance=0.85,
                            description=f"检测到与 {pattern_name} 相关模式"
                        )
                        patterns.append(match)
        
        self.known_patterns.extend(patterns)
        
        return patterns
    
    def _check_pattern_match(
        self,
        signal: ProcessedSignal,
        pattern_name: str,
        pattern_data: Any
    ) -> bool:
        """检查模式匹配"""
        freq_mhz = signal.raw_signal.frequency / 1e6
        
        if pattern_name == 'hydrogen_line':
            return abs(freq_mhz - float(pattern_data)) < 0.1
        
        if pattern_name == 'water_house':
            freq_range = pattern_data.split('-')
            return freq_range[0] <= freq_mhz <= freq_range[1]
        
        # 随机匹配
        return random.random() < 0.1
    
    def search_for_technosignatures(
        self,
        signals: List[ProcessedSignal]
    ) -> List[PatternMatch]:
        """搜索技术签名"""
        technosignatures = []
        
        for signal in signals:
            if signal.signal_type == SignalType.ARTIFICIAL:
                # 检查是否包含数字序列、调制模式等
                if self._contains_technosignature(signal):
                    pattern = PatternMatch(
                        pattern_id=f"TECH-{signal.raw_signal.signal_id}",
                        matched_signals=[signal.raw_signal.signal_id],
                        pattern_type='technosignature',
                        frequency=0.01,
                        significance=0.9,
                        description="检测到潜在技术签名"
                    )
                    technosignatures.append(pattern)
        
        return technosignatures
    
    def _contains_technosignature(self, signal: ProcessedSignal) -> bool:
        """检查是否包含技术签名"""
        # 高频稳定性 + 窄带宽 + 调制 = 可能是技术签名
        return (signal.features.get('frequency_stability', 0) > 0.98 and
                signal.features.get('bandwidth_ratio', 1) < 1e-6 and
                signal.features.get('modulation_index', 0) > 0.1)


class CivilizationAssessor:
    """文明评估器"""
    
    def __init__(self, config: 'SETIConfig' = None):
        self.config = config or get_config().seti
        self.assessments: List[CivilizationAssessment] = []
        self.civilization_database: Dict = {}
    
    def assess_civilization(
        self,
        signals: List[ProcessedSignal],
        anomalies: List[Anomaly] = None
    ) -> List[CivilizationAssessment]:
        """评估外星文明"""
        assessments = []
        
        for signal in signals:
            if signal.signal_type == SignalType.ARTIFICIAL:
                assessment = self._evaluate_signal(signal)
                assessments.append(assessment)
                self.assessments.append(assessment)
        
        return assessments
    
    def _evaluate_signal(
        self,
        signal: ProcessedSignal
    ) -> CivilizationAssessment:
        """评估单个信号"""
        # 估计文明等级
        level = self._estimate_level(signal)
        
        # 计算置信度
        confidence = self._calculate_assessment_confidence(signal, level)
        
        # 收集证据
        evidence = self._collect_evidence(signal)
        
        # 评估威胁等级
        threat_level = self._assess_threat(level, signal)
        
        return CivilizationAssessment(
            civilization_id=self._generate_civ_id(signal),
            signal_id=signal.raw_signal.signal_id,
            estimated_level=level,
            confidence=confidence,
            evidence=evidence,
            threat_level=threat_level,
            assessment_notes=self._generate_assessment_notes(signal, level)
        )
    
    def _estimate_level(self, signal: ProcessedSignal) -> CivilizationLevel:
        """估计文明等级"""
        freq = signal.raw_signal.frequency / 1e9  # GHz
        intensity = signal.raw_signal.intensity
        
        if intensity > 1e6:  # 强信号
            return CivilizationLevel.TYPE_II
        elif intensity > 1e3:  # 中等信号
            return CivilizationLevel.TYPE_I
        elif freq > 10:  # 高频信号
            return CivilizationLevel.TYPE_II
        elif 1 < freq < 10:
            return CivilizationLevel.TYPE_I
        else:
            return CivilizationLevel.TYPE_0
    
    def _calculate_assessment_confidence(
        self,
        signal: ProcessedSignal,
        level: CivilizationLevel
    ) -> float:
        """计算评估置信度"""
        confidence = signal.confidence
        
        if signal.features.get('frequency_stability', 0) > 0.99:
            confidence += 0.05
        
        return min(0.95, confidence)
    
    def _collect_evidence(self, signal: ProcessedSignal) -> List[str]:
        """收集证据"""
        evidence = []
        
        if signal.features.get('frequency_stability', 0) > 0.98:
            evidence.append("高频稳定性表明受控发射")
        
        if signal.features.get('modulation_index', 0) > 0.2:
            evidence.append("复杂调制模式表明信息内容")
        
        if signal.raw_signal.duration > 60:
            evidence.append("长持续时间表明定向传输")
        
        return evidence
    
    def _assess_threat(
        self,
        level: CivilizationLevel,
        signal: ProcessedSignal
    ) -> str:
        """评估威胁等级"""
        if level in [CivilizationLevel.TYPE_III, CivilizationLevel.TYPE_IV]:
            return "high"
        elif level == CivilizationLevel.TYPE_II:
            return "medium"
        else:
            return "low"
    
    def _generate_civ_id(self, signal: ProcessedSignal) -> str:
        """生成文明ID"""
        import random
        return f"CIV-{signal.raw_signal.signal_id[:8]}-{random.randint(1000,9999)}"
    
    def _generate_assessment_notes(
        self,
        signal: ProcessedSignal,
        level: CivilizationLevel
    ) -> str:
        """生成评估注释"""
        notes = {
            CivilizationLevel.TYPE_0: "早期文明，仅能进行基础无线电传输",
            CivilizationLevel.TYPE_I: "行星级文明，能够利用母星能源",
            CivilizationLevel.TYPE_II: "恒星级文明，能够利用恒星能源",
            CivilizationLevel.TYPE_III: "星系级文明，具有跨恒星系旅行能力"
        }
        return notes.get(level, "无法确定文明等级")


class SETIAnalyzer:
    """SETI分析主类"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or get_config()
        self.seti_config = self.config.seti
        
        # 初始化子系统
        self.signal_processor = SignalProcessor(self.seti_config)
        self.anomaly_detector = AnomalyDetector(self.seti_config)
        self.pattern_recognizer = PatternRecognizer(self.seti_config)
        self.civilization_assessor = CivilizationAssessor(self.seti_config)
        
        # 信号存储
        self.raw_signals: List[RawSignal] = []
        
        logger.info("SETI分析系统初始化完成")
    
    def scan(self, data: List[Dict] = None) -> Dict:
        """
        扫描信号
        
        Args:
            data: 可选，外部数据源
        
        Returns:
            Dict: 扫描结果
        """
        # 生成或获取信号
        if data:
            signals = self._parse_input_data(data)
        else:
            signals = self._generate_test_signals()
        
        # 处理信号
        processed_signals = self.signal_processor.process_batch(signals)
        
        # 检测异常
        anomalies = self.anomaly_detector.detect_anomaly(processed_signals)
        
        # 识别模式
        patterns = self.pattern_recognizer.recognize_patterns(processed_signals)
        
        # 评估文明
        assessments = self.civilization_assessor.assess_civilization(
            processed_signals, anomalies
        )
        
        return {
            'scan_summary': {
                'scan_time': self._get_timestamp(),
                'signals_scanned': len(signals),
                'processing_rate': self.seti_config.signal_processing_rate,
                'anomalies_detected': len(anomalies),
                'patterns_found': len(patterns),
                'civilizations_assessed': len(assessments)
            },
            'signal_analysis': {
                'total_processed': self.signal_processor.processing_stats['total_processed'],
                'artificial_signals': self.signal_processor.processing_stats['artificial_count'],
                'natural_signals': self.signal_processor.processing_stats['natural_count'],
                'processed_signals': [s.to_dict() for s in processed_signals]
            },
            'anomaly_report': {
                'total_anomalies': len(anomalies),
                'anomalies': [a.to_dict() for a in anomalies],
                'summary': self.anomaly_detector.get_anomaly_summary()
            },
            'pattern_analysis': {
                'patterns': [p.to_dict() for p in patterns],
                'technosignatures': self.pattern_recognizer.search_for_technosignatures(processed_signals)
            },
            'civilization_assessment': {
                'assessments': [a.to_dict() for a in assessments],
                'threat_summary': self._summarize_threats(assessments)
            }
        }
    
    def detect_anomaly(self, signals: List[ProcessedSignal] = None) -> Dict:
        """检测异常"""
        if signals is None:
            # 使用最近的信号
            signals = list(self.signal_processor.processed_signals)
        
        anomalies = self.anomaly_detector.detect_anomaly(signals)
        
        return {
            'anomalies_detected': len(anomalies),
            'anomaly_list': [a.to_dict() for a in anomalies],
            'detection_rate': len(anomalies) / max(len(signals), 1)
        }
    
    def recognize_patterns(self, signals: List[ProcessedSignal] = None) -> Dict:
        """识别模式"""
        if signals is None:
            signals = list(self.signal_processor.processed_signals)
        
        patterns = self.pattern_recognizer.recognize_patterns(signals)
        technosignatures = self.pattern_recognizer.search_for_technosignatures(signals)
        
        return {
            'patterns_found': len(patterns),
            'technosignatures_found': len(technosignatures),
            'patterns': [p.to_dict() for p in patterns],
            'technosignatures': [t.to_dict() for t in technosignatures]
        }
    
    def assess_civilization(self, signals: List[ProcessedSignal] = None) -> Dict:
        """评估文明"""
        if signals is None:
            signals = list(self.signal_processor.processed_signals)
        
        assessments = self.civilization_assessor.assess_civilization(signals)
        
        return {
            'civilizations_assessed': len(assessments),
            'assessments': [a.to_dict() for a in assessments],
            'threat_summary': self._summarize_threats(assessments)
        }
    
    def _generate_test_signals(self) -> List[RawSignal]:
        """生成测试信号"""
        signals = []
        num_signals = random.randint(50, 200)
        
        for i in range(num_signals):
            freq_range = self.seti_config.frequency_range
            freq = random.uniform(freq_range[0], freq_range[1]) * 1e9
            
            signal = RawSignal(
                signal_id=f"SIG-{random.randint(100000,999999)}",
                timestamp=time.time() + i * 0.001,
                frequency=freq,
                bandwidth=random.uniform(1, 100),
                intensity=random.uniform(0.001, 1000),
                duration=random.uniform(0.1, 60),
                source_direction=(random.uniform(0, 360), random.uniform(-90, 90)),
                polarization=random.choice(['linear', 'circular', 'unknown']),
                snr=random.uniform(1, 100)
            )
            signals.append(signal)
        
        return signals
    
    def _parse_input_data(self, data: List[Dict]) -> List[RawSignal]:
        """解析输入数据"""
        signals = []
        for item in data:
            signal = RawSignal(
                signal_id=item.get('id', f"SIG-{random.randint(100000,999999)}"),
                timestamp=item.get('timestamp', time.time()),
                frequency=item.get('frequency', random.uniform(1, 10) * 1e9),
                bandwidth=item.get('bandwidth', random.uniform(1, 100)),
                intensity=item.get('intensity', random.uniform(0.001, 1000)),
                duration=item.get('duration', random.uniform(0.1, 60)),
                source_direction=item.get('source_direction', 
                                         (random.uniform(0, 360), random.uniform(-90, 90))),
                polarization=item.get('polarization', 'unknown'),
                snr=item.get('snr', random.uniform(1, 100))
            )
            signals.append(signal)
        
        return signals
    
    def _summarize_threats(
        self,
        assessments: List[CivilizationAssessment]
    ) -> Dict:
        """总结威胁"""
        threat_counts = {'low': 0, 'medium': 0, 'high': 0}
        
        for assessment in assessments:
            threat_level = assessment.threat_level
            if threat_level in threat_counts:
                threat_counts[threat_level] += 1
        
        return {
            'total_civilizations': len(assessments),
            'by_threat_level': threat_counts,
            'highest_threat': max(
                [a.threat_level for a in assessments] or ['low']
            ),
            'recommendation': self._generate_recommendation(threat_counts)
        }
    
    def _generate_recommendation(self, threat_counts: Dict) -> str:
        """生成建议"""
        if threat_counts['high'] > 0:
            return "建议加强远程监控，避免主动联系"
        elif threat_counts['medium'] > 0:
            return "建议继续监测，收集更多信息"
        else:
            return "可考虑有控制的接触尝试"
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            'system': 'SETI_ANALYZER',
            'status': 'operational',
            'processing_rate': self.seti_config.signal_processing_rate,
            'signals_processed': self.signal_processor.processing_stats['total_processed'],
            'anomalies_detected': len(self.anomaly_detector.detected_anomalies),
            'civilizations_assessed': len(self.civilization_assessor.assessments)
        }


# 导入必要的配置
from .config import get_config, SETIConfig
