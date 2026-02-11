"""
根因分析模块

功能：
- 依赖图分析
- 时序相关性分析
- 故障传播链分析
- 定位时间目标: <1分钟
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import heapq

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """节点类型"""
    SERVICE = "service"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    LOAD_BALANCER = "load_balancer"
    MICROSERVICE = "microservice"
    EXTERNAL = "external"


class RelationshipType(Enum):
    """关系类型"""
    DEPENDS_ON = "depends_on"        # 依赖
    CALLS = "calls"                  # 调用
    CONTAINS = "contains"            # 包含
    SCALES_WITH = "scales_with"     # 扩缩容关联
    AFFECTS = "affects"              # 影响


@dataclass
class DependencyNode:
    """依赖图节点"""
    id: str
    name: str
    node_type: NodeType
    metadata: Dict = field(default_factory=dict)
    health_score: float = 1.0  # 0-1, 1为健康
    parent_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.node_type.value,
            "metadata": self.metadata,
            "health_score": self.health_score,
            "parent_ids": self.parent_ids,
        }


@dataclass
class DependencyEdge:
    """依赖图边"""
    source: str
    target: str
    relationship: RelationshipType
    weight: float = 1.0  # 影响权重
    latency_ms: float = 0  # 调用延迟
    error_rate: float = 0  # 错误率

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship.value,
            "weight": self.weight,
            "latency_ms": self.latency_ms,
            "error_rate": self.error_rate,
        }


@dataclass
class TimeSeriesPoint:
    """时间序列点"""
    timestamp: datetime
    value: float


@dataclass
class RootCause:
    """根因分析结果"""
    id: str
    node_id: str
    node_name: str
    node_type: NodeType
    confidence: float  # 0-1, 置信度
    description: str
    evidence: List[str] = field(default_factory=list)
    affected_services: List[str] = field(default_factory=list)
    related_anomalies: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    analysis_time_ms: float = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "node_name": self.node_name,
            "node_type": self.node_type.value,
            "confidence": self.confidence,
            "description": self.description,
            "evidence": self.evidence,
            "affected_services": self.affected_services,
            "related_anomalies": self.related_anomalies,
            "suggested_actions": self.suggested_actions,
            "analysis_time_ms": self.analysis_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class DependencyGraph:
    """依赖图管理器"""

    def __init__(self):
        self.nodes: Dict[str, DependencyNode] = {}
        self.edges: List[DependencyEdge] = []
        self.adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # node_id -> [(target, weight)]
        self.reverse_adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    def add_node(self, node: DependencyNode):
        """添加节点"""
        self.nodes[node.id] = node

    def add_edge(self, edge: DependencyEdge):
        """添加边"""
        self.edges.append(edge)
        self.adjacency[edge.source].append((edge.target, edge.weight))
        self.reverse_adjacency[edge.target].append((edge.source, edge.weight))

    def get_upstream(self, node_id: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """获取上游依赖 (被当前节点依赖的节点)"""
        if visited is None:
            visited = set()

        if node_id in visited:
            return visited

        visited.add(node_id)
        for upstream_id, _ in self.reverse_adjacency.get(node_id, []):
            self.get_upstream(upstream_id, visited)

        return visited - {node_id}

    def get_downstream(self, node_id: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """获取下游影响 (依赖当前节点的节点)"""
        if visited is None:
            visited = set()

        if node_id in visited:
            return visited

        visited.add(node_id)
        for downstream_id, _ in self.adjacency.get(node_id, []):
            self.get_downstream(downstream_id, visited)

        return visited - {node_id}

    def get_all_affected(self, node_id: str) -> Dict[str, Set[str]]:
        """获取所有受影响的服务"""
        upstream = self.get_upstream(node_id)
        downstream = self.get_downstream(node_id)

        return {
            "upstream": upstream,
            "downstream": downstream,
            "total_affected": upstream | downstream,
        }

    def build_default_graph(self):
        """构建默认的微服务依赖图"""
        # 添加节点
        nodes = [
            ("api-gateway", "API Gateway", NodeType.LOAD_BALANCER),
            ("user-service", "用户服务", NodeType.MICROSERVICE),
            ("order-service", "订单服务", NodeType.MICROSERVICE),
            ("payment-service", "支付服务", NodeType.MICROSERVICE),
            ("inventory-service", "库存服务", NodeType.MICROSERVICE),
            ("notification-service", "通知服务", NodeType.MICROSERVICE),
            ("redis-cache", "Redis缓存", NodeType.CACHE),
            ("mysql-master", "MySQL主库", NodeType.DATABASE),
            ("mysql-slave", "MySQL从库", NodeType.DATABASE),
            ("rabbitmq", "消息队列", NodeType.QUEUE),
            ("elasticsearch", "搜索引擎", NodeType.EXTERNAL),
        ]

        for node_id, name, node_type in nodes:
            self.add_node(DependencyNode(
                id=node_id,
                name=name,
                node_type=node_type,
            ))

        # 添加边 (source -> target 表示 source 依赖 target)
        edges = [
            ("api-gateway", "user-service", RelationshipType.CALLS, 1.0),
            ("api-gateway", "order-service", RelationshipType.CALLS, 1.0),
            ("api-gateway", "payment-service", RelationshipType.CALLS, 1.0),
            ("user-service", "redis-cache", RelationshipType.DEPENDS_ON, 0.8),
            ("user-service", "mysql-master", RelationshipType.DEPENDS_ON, 0.9),
            ("order-service", "user-service", RelationshipType.DEPENDS_ON, 0.7),
            ("order-service", "inventory-service", RelationshipType.CALLS, 0.9),
            ("order-service", "mysql-master", RelationshipType.DEPENDS_ON, 0.9),
            ("order-service", "rabbitmq", RelationshipType.DEPENDS_ON, 0.6),
            ("payment-service", "order-service", RelationshipType.DEPENDS_ON, 0.8),
            ("payment-service", "mysql-master", RelationshipType.DEPENDS_ON, 0.7),
            ("inventory-service", "mysql-master", RelationshipType.DEPENDS_ON, 0.9),
            ("inventory-service", "redis-cache", RelationshipType.DEPENDS_ON, 0.5),
            ("notification-service", "rabbitmq", RelationshipType.DEPENDS_ON, 0.6),
            ("notification-service", "elasticsearch", RelationshipType.CALLS, 0.5),
            ("mysql-slave", "mysql-master", RelationshipType.CONTAINS, 1.0),
        ]

        for source, target, rel, weight in edges:
            self.add_edge(DependencyEdge(
                source=source,
                target=target,
                relationship=rel,
                weight=weight,
            ))


class TimeSeriesAnalyzer:
    """时序相关性分析器"""

    def __init__(self):
        self.time_series: Dict[str, List[TimeSeriesPoint]] = {}

    def add_time_series(self, name: str, points: List[TimeSeriesPoint]):
        """添加时间序列"""
        self.time_series[name] = points

    def compute_correlation(self, series1: str, series2: str) -> float:
        """计算两个时间序列的相关系数"""
        if series1 not in self.time_series or series2 not in self.time_series:
            return 0.0

        # 对齐时间戳
        ts1_dict = {p.timestamp: p.value for p in self.time_series[series1]}
        ts2_dict = {p.timestamp: p.value for p in self.time_series[series2]}

        common_times = set(ts1_dict.keys()) & set(ts2_dict.keys())
        if len(common_times) < 10:
            return 0.0

        values1 = [ts1_dict[t] for t in sorted(common_times)]
        values2 = [ts2_dict[t] for t in sorted(common_times)]

        # Pearson相关系数
        mean1 = np.mean(values1)
        mean2 = np.mean(values2)

        numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2))
        denominator = np.sqrt(sum((v1 - mean1) ** 2 for v1 in values1)) * \
                      np.sqrt(sum((v2 - mean2) ** 2 for v2 in values2))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def compute_all_correlations(self) -> Dict[Tuple[str, str], float]:
        """计算所有时间序列对的相关性"""
        correlations = {}
        names = list(self.time_series.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                corr = self.compute_correlation(names[i], names[j])
                correlations[(names[i], names[j])] = corr

        return correlations

    def detect_causal_direction(self, series1: str, series2: str) -> Tuple[str, str, float]:
        """
        检测因果方向 (简化的Granger因果检验)

        Returns:
            (cause, effect, confidence)
        """
        if series1 not in self.time_series or series2 not in self.time_series:
            return (series1, series2, 0.0)

        ts1 = self.time_series[series1]
        ts2 = self.time_series[series2]

        # 如果series1的变化先于series2的变化, 则series1可能是因
        # 简化为: 检查哪个序列变化更剧烈
        if len(ts1) < 2 or len(ts2) < 2:
            return (series1, series2, 0.5)

        # 计算变化率
        changes1 = self._compute_changes([p.value for p in ts1])
        changes2 = self._compute_changes([p.value for p in ts2])

        avg_change1 = np.mean([abs(c) for c in changes1]) if changes1 else 0
        avg_change2 = np.mean([abs(c) for c in changes2]) if changes2 else 0

        if avg_change1 > avg_change2 * 1.5:
            return (series1, series2, min(avg_change1 / max(avg_change2, 1), 1.0))
        elif avg_change2 > avg_change1 * 1.5:
            return (series2, series1, min(avg_change2 / max(avg_change1, 1), 1.0))
        else:
            return (series1, series2, 0.5)

    def _compute_changes(self, values: List[float]) -> List[float]:
        """计算变化率"""
        if len(values) < 2:
            return []
        return [values[i] - values[i-1] for i in range(1, len(values))]


class FaultPropagationAnalyzer:
    """故障传播链分析器"""

    def __init__(self, dependency_graph: DependencyGraph):
        self.graph = dependency_graph
        self.propagation_history: List[Dict] = []

    def build_propagation_chain(self, initial_fault_node: str,
                                 affected_nodes: List[str]) -> List[Dict]:
        """
        构建故障传播链

        Args:
            initial_fault_node: 初始故障节点
            affected_nodes: 受影响的节点列表

        Returns:
            传播链列表
        """
        chain = []
        visited = set()

        # 找到从初始故障节点到各受影响节点的路径
        for affected in affected_nodes:
            path = self._find_shortest_path(initial_fault_node, affected)
            if path:
                chain.append({
                    "path": path,
                    "affected_node": affected,
                    "hop_count": len(path) - 1,
                })
                visited.update(path)

        # 按跳数排序
        chain.sort(key=lambda x: x["hop_count"])

        # 记录传播链
        self.propagation_history.append({
            "initial_fault": initial_fault_node,
            "chain": chain,
            "timestamp": datetime.now(),
        })

        return chain

    def _find_shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """使用BFS找最短路径"""
        if start == end:
            return [start]

        queue = [(start, [start])]
        visited = {start}

        while queue:
            node, path = queue.pop(0)

            for neighbor, _ in self.graph.adjacency.get(node, []):
                if neighbor == end:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def calculate_propagation_probability(self, source: str, target: str) -> float:
        """计算从源节点到目标节点的传播概率"""
        path = self._find_shortest_path(source, target)
        if not path:
            return 0.0

        # 计算路径上的权重乘积
        probability = 1.0
        for i in range(len(path) - 1):
            for edge in self.graph.edges:
                if edge.source == path[i] and edge.target == path[i + 1]:
                    probability *= (1 - edge.error_rate) * edge.weight
                    break

        return probability


class RootCauseAnalyzer:
    """
    综合根因分析器

    整合依赖图分析、时序相关分析和故障传播链分析
    定位时间目标: <1分钟
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化根因分析器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.graph = DependencyGraph()
        self.time_series_analyzer = TimeSeriesAnalyzer()

        # 构建默认依赖图
        self.graph.build_default_graph()

        self.fault_propagation_analyzer = FaultPropagationAnalyzer(self.graph)

        # 异常记录
        self.anomaly_records: Dict[str, Dict] = {}
        self._root_cause_counter = 0

        # 配置
        self.correlation_threshold = self.config.get("correlation_threshold", 0.7)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)

    def _generate_root_cause_id(self) -> str:
        """生成根因ID"""
        self._root_cause_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"rc_{timestamp}_{self._root_cause_counter:04d}"

    def register_anomaly(self, anomaly_id: str, node_id: str,
                         metric_name: str, metric_value: float,
                         timestamp: datetime):
        """注册异常事件"""
        self.anomaly_records[anomaly_id] = {
            "node_id": node_id,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "timestamp": timestamp,
        }

    def analyze(self, symptom: str, time_range: str = "1h",
                affected_services: Optional[List[str]] = None) -> RootCause:
        """
        执行根因分析

        Args:
            symptom: 症状描述 (例如 "high_latency", "service_down")
            time_range: 时间范围 (例如 "1h", "30m", "24h")
            affected_services: 受影响的服务列表

        Returns:
            RootCause 根因分析结果
        """
        start_time = datetime.now()

        # 解析时间范围
        time_range_map = {
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(hours=24),
        }
        time_delta = time_range_map.get(time_range, timedelta(hours=1))
        start_timestamp = datetime.now() - time_delta

        # 根据症状匹配可能的原因
        symptom_causes = self._match_symptom_to_causes(symptom)

        # 分析每个候选原因
        candidate_scores = []
        for cause in symptom_causes:
            if cause["node_id"] not in self.graph.nodes:
                continue

            # 获取该节点的上游依赖
            affected = self.graph.get_all_affected(cause["node_id"])

            # 计算置信度
            confidence = self._calculate_confidence(
                cause, affected, affected_services
            )

            if confidence >= self.correlation_threshold:
                candidate_scores.append((cause, confidence))

        # 排序并选择最佳根因
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        if candidate_scores:
            best_cause, best_confidence = candidate_scores[0]
            root_cause = self._build_root_cause(
                best_cause, best_confidence, symptom, time_delta,
                affected_services, start_time
            )
        else:
            # 没有找到明确的根因,返回默认分析结果
            root_cause = self._build_unknown_root_cause(symptom, start_time)

        analysis_time = (datetime.now() - start_time).total_seconds() * 1000
        root_cause.analysis_time_ms = analysis_time

        logger.info(f"根因分析完成: {root_cause.id}, 耗时: {analysis_time:.2f}ms")

        return root_cause

    def _match_symptom_to_causes(self, symptom: str) -> List[Dict]:
        """匹配症状到可能的原因"""
        symptom_map = {
            "high_latency": [
                {"node_id": "api-gateway", "reason": "网关负载过高"},
                {"node_id": "mysql-master", "reason": "数据库查询慢"},
                {"node_id": "redis-cache", "reason": "缓存未命中率高"},
                {"node_id": "rabbitmq", "reason": "消息队列积压"},
            ],
            "service_down": [
                {"node_id": "api-gateway", "reason": "网关不可用"},
                {"node_id": "user-service", "reason": "用户服务崩溃"},
                {"node_id": "mysql-master", "reason": "数据库连接失败"},
            ],
            "high_error_rate": [
                {"node_id": "payment-service", "reason": "支付服务异常"},
                {"node_id": "order-service", "reason": "订单服务错误"},
                {"node_id": "mysql-master", "reason": "数据库事务失败"},
            ],
            "cpu_high": [
                {"node_id": "api-gateway", "reason": "网关CPU使用率高"},
                {"node_id": "user-service", "reason": "用户服务CPU负载高"},
                {"node_id": "mysql-master", "reason": "数据库CPU负载高"},
            ],
            "memory_high": [
                {"node_id": "api-gateway", "reason": "网关内存使用率高"},
                {"node_id": "user-service", "reason": "用户服务内存泄漏"},
                {"node_id": "redis-cache", "reason": "缓存内存不足"},
            ],
        }

        return symptom_map.get(symptom, [
            {"node_id": "api-gateway", "reason": "需要进一步诊断"},
            {"node_id": "mysql-master", "reason": "数据库可能存在问题"},
        ])

    def _calculate_confidence(self, cause: Dict, affected: Dict,
                               affected_services: Optional[List[str]]) -> float:
        """计算置信度"""
        confidence = 0.5  # 基础置信度

        # 如果受影响服务列表匹配,增加置信度
        if affected_services:
            affected_set = set(affected.get("downstream", set()))
            matched = affected_set & set(affected_services)
            confidence += len(matched) * 0.1

        # 根据节点类型调整
        node = self.graph.nodes.get(cause["node_id"])
        if node:
            if node.node_type == NodeType.DATABASE:
                confidence += 0.2  # 数据库通常是关键点
            elif node.node_type == NodeType.CACHE:
                confidence += 0.1

        return min(confidence, 1.0)

    def _build_root_cause(self, cause: Dict, confidence: float, symptom: str,
                          time_delta: timedelta, affected_services: Optional[List[str]],
                          start_time: datetime) -> RootCause:
        """构建根因分析结果"""
        node_id = cause["node_id"]
        node = self.graph.nodes.get(node_id)

        affected = self.graph.get_all_affected(node_id)

        evidence = [
            f"症状 '{symptom}' 匹配到 {node.name if node else node_id}",
            f"该节点影响 {len(affected['downstream'])} 个下游服务",
            f"置信度: {confidence:.2%}",
        ]

        suggested_actions = self._get_suggested_actions(node_id, symptom)

        return RootCause(
            id=self._generate_root_cause_id(),
            node_id=node_id,
            node_name=node.name if node else node_id,
            node_type=node.node_type if node else NodeType.MICROSERVICE,
            confidence=confidence,
            description=cause["reason"],
            evidence=evidence,
            affected_services=list(affected.get("downstream", set())),
            suggested_actions=suggested_actions,
        )

    def _build_unknown_root_cause(self, symptom: str, start_time: datetime) -> RootCause:
        """构建未知根因结果"""
        return RootCause(
            id=self._generate_root_cause_id(),
            node_id="unknown",
            node_name="待进一步诊断",
            node_type=NodeType.MICROSERVICE,
            confidence=0.3,
            description=f"未能明确确定'{symptom}'的根本原因",
            evidence=[
                "症状不在已知模式中",
                "建议收集更多监控数据",
            ],
            affected_services=[],
            suggested_actions=[
                "检查最近的服务部署记录",
                "查看详细的错误日志",
                "联系技术支持团队",
            ],
        )

    def _get_suggested_actions(self, node_id: str, symptom: str) -> List[str]:
        """获取建议的操作"""
        action_map = {
            "mysql-master": [
                "检查数据库连接池配置",
                "优化慢查询",
                "考虑增加从库分散读压力",
            ],
            "redis-cache": [
                "检查缓存命中率",
                "增加缓存内存",
                "检查缓存过期策略",
            ],
            "rabbitmq": [
                "检查消息积压情况",
                "增加消费者数量",
                "清理死信队列",
            ],
            "api-gateway": [
                "检查网关负载均衡配置",
                "增加网关实例数量",
                "检查限流配置",
            ],
        }

        return action_map.get(node_id, [
            "检查该服务的日志",
            "查看监控指标趋势",
            "考虑重启服务",
        ])

    def add_service_dependency(self, source: str, target: str,
                                 relationship: str = "depends_on",
                                 weight: float = 1.0):
        """添加服务依赖关系"""
        try:
            rel_type = RelationshipType(relationship)
            self.graph.add_edge(DependencyEdge(
                source=source,
                target=target,
                relationship=rel_type,
                weight=weight,
            ))
        except ValueError:
            logger.warning(f"未知的依赖关系类型: {relationship}")

    def get_dependency_topology(self) -> Dict:
        """获取依赖拓扑结构"""
        return {
            "nodes": {k: v.to_dict() for k, v in self.graph.nodes.items()},
            "edges": [e.to_dict() for e in self.graph.edges],
        }

    def analyze_from_anomalies(self, anomalies: List[Dict]) -> List[RootCause]:
        """从异常列表进行根因分析"""
        root_causes = []

        for anomaly in anomalies:
            symptom = anomaly.get("metric", "unknown")
            root_cause = self.analyze(symptom, "1h")
            root_cause.related_anomalies = [anomaly.get("id", "")]
            root_causes.append(root_cause)

        return root_causes
