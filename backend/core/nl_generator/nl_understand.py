"""
自然语言理解模块 - Natural Language Understanding

负责意图识别、实体提取和槽位填充
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class IntentType(Enum):
    """意图类型枚举"""
    CREATE_PIPELINE = "create_pipeline"
    UPDATE_PIPELINE = "update_pipeline"
    DELETE_PIPELINE = "delete_pipeline"
    QUERY_PIPELINE = "query_pipeline"
    CREATE_AGENT = "create_agent"
    UPDATE_AGENT = "update_agent"
    DELETE_AGENT = "delete_agent"
    QUERY_AGENT = "query_agent"
    EXECUTE_TASK = "execute_task"
    UNKNOWN = "unknown"


class EntityType(Enum):
    """实体类型枚举"""
    PIPELINE_NAME = "pipeline_name"
    AGENT_NAME = "agent_name"
    NODE_TYPE = "node_type"
    CONNECTION = "connection"
    CONDITION = "condition"
    PARAMETER = "parameter"
    DATA_SOURCE = "data_source"
    OUTPUT_FORMAT = "output_format"


@dataclass
class Entity:
    """实体类"""
    type: EntityType
    value: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class Slot:
    """槽位类"""
    name: str
    value: Optional[str] = None
    required: bool = False
    filled: bool = False


@dataclass
class UnderstandingResult:
    """理解结果类"""
    intent: IntentType
    confidence: float
    entities: List[Entity]
    slots: Dict[str, Slot]
    raw_query: str = ""
    suggestions: List[str] = field(default_factory=list)


class NLUnderstand:
    """自然语言理解器"""
    
    def __init__(self):
        self._init_intent_patterns()
        self._init_entity_patterns()
    
    def _init_intent_patterns(self):
        """初始化意图识别模式"""
        # Agent相关模式 - 优先级更高，放在前面
        self.intent_patterns = {
            IntentType.CREATE_AGENT: [
                r"(?:创建|新建|制作|构建|开发)\s*(?:一个|个)?\s*(?:智能体|助手|机器人|agent)",
                r"(?:创建|新建|制作|构建|开发)\s*(?:一个|个)?\s*(?!.*(?:pipeline|流水线|管道))",
                r"我想\s*(?:创建|新建|制作|构建)\s*(?:一个|个)?\s*(?:智能体|助手|机器人|agent)",
                r"帮我\s*(?:创建|新建|制作|构建)\s*(?:一个|个)?\s*(?:智能体|助手|机器人|agent)",
            ],
            IntentType.UPDATE_AGENT: [
                r"(?:修改|更新|编辑|调整|更改)\s*(?:智能体|助手|机器人|agent)?",
            ],
            IntentType.DELETE_AGENT: [
                r"(?:删除|移除|去掉)\s*(?:智能体|助手|机器人|agent)?",
            ],
            IntentType.QUERY_AGENT: [
                r"(?:查询|查看|获取|列出)\s*(?:智能体|助手|机器人|agent)?",
            ],
            IntentType.CREATE_PIPELINE: [
                r"(?:创建|新建|制作|构建|开发)\s*(?:一个|个)?\s*(?:pipeline|流水线|管道)",
                r"我想\s*(?:创建|新建|制作|构建)\s*(?:一个|个)?\s*(?:pipeline|流水线|管道)",
                r"帮我\s*(?:创建|新建|制作|构建)\s*(?:一个|个)?\s*(?:pipeline|流水线|管道)",
                r"(?:创建一个|新建一个|制作一个|构建一个|开发一个).*(?:pipeline|流水线|管道)",
            ],
            IntentType.UPDATE_PIPELINE: [
                r"(?:修改|更新|编辑|调整|更改)\s*(?:pipeline|流水线|管道)?",
            ],
            IntentType.DELETE_PIPELINE: [
                r"(?:删除|移除|去掉)\s*(?:pipeline|流水线|管道)?",
            ],
            IntentType.QUERY_PIPELINE: [
                r"(?:查询|查看|获取|列出)\s*(?:pipeline|流水线|管道)?",
            ],
            IntentType.EXECUTE_TASK: [
                r"(?:执行|运行|启动|开始)\s*(?:任务|作业)?",
                r"帮我\s*(?:执行|运行|启动|开始)",
            ],
        }
    
    def _init_entity_patterns(self):
        """初始化实体识别模式"""
        self.entity_patterns = {
            EntityType.PIPELINE_NAME: [
                r"(?:叫做|名称|名字|名为)\s*([^\s,，。]+)",
                r"([^\s,，。]+)\s*(?:流水线|管道|pipeline)",
            ],
            EntityType.AGENT_NAME: [
                r"(?:叫做|名称|名字|名为)\s*([^\s,，。]+)",
                r"([^\s,，。]+)\s*(?:智能体|助手|机器人|agent)",
            ],
            EntityType.NODE_TYPE: [
                r"(?:节点|模块|组件)?\s*(?:类型)?\s*(?:是|为)?\s*(input|output|processor|llm|condition|filter|transform)",
            ],
            EntityType.CONNECTION: [
                r"(?:连接|关联|对接)\s*(?:到|于)?\s*([^\s,，。]+)",
                r"([^\s,，。]+)\s*(?:之后|后面|然后)",
            ],
            EntityType.PARAMETER: [
                r"(?:参数|属性|设置|配置)?\s*(?:是|为|叫|名叫)?\s*([^\s,，。]+)",
                r"--(\w+)\s*[:：]?\s*([^\s,，。]+)",
            ],
            EntityType.DATA_SOURCE: [
                r"(?:数据|输入)?\s*(?:来源|源)?\s*(?:是|为)?\s*(api|数据库|file|文件|http|https)",
            ],
            EntityType.OUTPUT_FORMAT: [
                r"(?:输出|结果)?\s*(?:格式|形式)?\s*(?:是|为)?\s*(json|xml|csv|markdown|text)",
            ],
        }
    
    def understand(self, query: str) -> UnderstandingResult:
        """
        理解用户输入的自然语言
        
        Args:
            query: 用户输入的自然语言查询
            
        Returns:
            UnderstandingResult: 理解结果
        """
        result = UnderstandingResult(
            intent=IntentType.UNKNOWN,
            confidence=0.0,
            entities=[],
            slots={},
            raw_query=query,
            suggestions=[]
        )
        
        # 1. 意图识别
        intent, intent_confidence = self._recognize_intent(query)
        result.intent = intent
        result.confidence = intent_confidence
        
        # 2. 实体提取
        entities = self._extract_entities(query)
        result.entities = entities
        
        # 3. 槽位填充
        slots = self._fill_slots(intent, entities)
        result.slots = slots
        
        # 4. 生成建议
        if result.confidence < 0.7:
            result.suggestions = self._generate_suggestions(result)
        
        return result
    
    def _recognize_intent(self, query: str) -> Tuple[IntentType, float]:
        """
        识别用户意图
        
        Args:
            query: 用户输入
            
        Returns:
            Tuple[IntentType, float]: 意图类型和置信度
        """
        best_intent = IntentType.UNKNOWN
        best_confidence = 0.0
        
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    # 计算匹配分数
                    match_count = len(re.findall(pattern, query_lower))
                    confidence = min(0.6 + (match_count * 0.1), 0.95)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent
        
        return best_intent, best_confidence
    
    def _extract_entities(self, query: str) -> List[Entity]:
        """
        提取实体
        
        Args:
            query: 用户输入
            
        Returns:
            List[Entity]: 实体列表
        """
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query.lower())
                for match in matches:
                    if match.groups():
                        value = match.group(1) or match.group(0)
                    else:
                        value = match.group(0)
                    
                    entity = Entity(
                        type=entity_type,
                        value=value.strip(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8
                    )
                    entities.append(entity)
        
        return entities
    
    def _fill_slots(self, intent: IntentType, entities: List[Entity]) -> Dict[str, Slot]:
        """
        填充槽位
        
        Args:
            intent: 意图类型
            entities: 实体列表
            
        Returns:
            Dict[str, Slot]: 槽位字典
        """
        # 根据意图类型定义必要槽位
        slot_templates = {
            IntentType.CREATE_PIPELINE: [
                Slot("pipeline_name", required=True),
                Slot("nodes", required=False),
                Slot("connections", required=False),
                Slot("parameters", required=False),
            ],
            IntentType.CREATE_AGENT: [
                Slot("agent_name", required=True),
                Slot("skills", required=False),
                Slot("memory", required=False),
                Slot("personality", required=False),
            ],
            IntentType.UPDATE_PIPELINE: [
                Slot("pipeline_name", required=True),
                Slot("changes", required=False),
            ],
            IntentType.EXECUTE_TASK: [
                Slot("task_name", required=True),
                Slot("parameters", required=False),
            ],
        }
        
        slot_dict = {}
        slot_template = slot_templates.get(intent, [])
        
        for slot in slot_template:
            # 尝试从实体中填充槽位
            for entity in entities:
                if self._match_slot(entity, slot.name):
                    slot.value = entity.value
                    slot.filled = True
                    break
            slot_dict[slot.name] = slot
        
        return slot_dict
    
    def _match_slot(self, entity: Entity, slot_name: str) -> bool:
        """匹配实体和槽位"""
        slot_entity_map = {
            "pipeline_name": [EntityType.PIPELINE_NAME],
            "agent_name": [EntityType.AGENT_NAME],
            "nodes": [EntityType.NODE_TYPE],
            "connections": [EntityType.CONNECTION],
            "parameters": [EntityType.PARAMETER],
        }
        
        return entity.type in slot_entity_map.get(slot_name, [])
    
    def _generate_suggestions(self, result: UnderstandingResult) -> List[str]:
        """生成建议"""
        suggestions = []
        
        if result.intent == IntentType.UNKNOWN:
            suggestions = [
                "请尝试使用更明确的指令，如'创建一个pipeline'",
                "可以说'创建一个名为xxx的agent'",
            ]
        elif result.slots:
            unfilled_required = [s for s in result.slots.values() 
                               if s.required and not s.filled]
            if unfilled_required:
                slot_names = ", ".join([s.name for s in unfilled_required])
                suggestions.append(f"请提供{slot_names}")
        
        return suggestions
    
    def add_custom_intent(self, intent: IntentType, patterns: List[str]):
        """添加自定义意图模式"""
        if intent not in self.intent_patterns:
            self.intent_patterns[intent] = []
        self.intent_patterns[intent].extend(patterns)
    
    def add_custom_entity(self, entity_type: EntityType, pattern: str):
        """添加自定义实体模式"""
        if entity_type not in self.entity_patterns:
            self.entity_patterns[entity_type] = []
        self.entity_patterns[entity_type].append(pattern)
    
    def train(self, dataset: List[Tuple[str, IntentType]]):
        """
        训练模型（基于规则的学习）
        
        Args:
            dataset: 训练数据集 [(query, intent), ...]
        """
        for query, intent in dataset:
            # 简单规则：从query中提取关键词
            keywords = self._extract_keywords(query)
            for keyword in keywords:
                pattern = rf"{keyword}"
                self.add_custom_intent(intent, [pattern])
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 移除停用词
        stopwords = {"的", "是", "在", "有", "和", "与", "或", "一个", "帮我", "我想"}
        words = text.split()
        keywords = [w for w in words if w not in stopwords and len(w) > 1]
        return keywords
