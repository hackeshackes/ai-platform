"""
实体识别模块 - NER (Named Entity Recognition)
支持多种NER方法：规则匹配、预训练模型、LLM
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re
from enum import Enum


class EntityType(Enum):
    """标准实体类型"""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    CUSTOM = "CUSTOM"
    EMAIL = "EMAIL"
    PHONE = "PHONE"


@dataclass
class EntityMention:
    """实体提及"""
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence
        }


class NERBase:
    """NER基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    def extract(self, text: str) -> List[EntityMention]:
        raise NotImplementedError


class RuleBasedNER(NERBase):
    """基于规则的NER"""
    
    def __init__(self):
        super().__init__("rule_based_ner")
        
        # 预定义实体类型和正则模式
        self.patterns = {
            EntityType.DATE.value: [
                r'\d{4}年\d{1,2}月\d{1,2}日',
                r'\d{4}-\d{2}-\d{2}',
                r'\d{1,2}月\d{1,2}日',
                r'今天|明天|昨天|上周|本周|下周',
                r'\d{1,2}年'
            ],
            EntityType.TIME.value: [
                r'\d{1,2}:\d{2}(:\d{2})?',
                r'\d{1,2}点\d{1,2}分',
                r'上午|下午|晚上|凌晨'
            ],
            EntityType.MONEY.value: [
                r'[\$¥€£]\d+(,\d{3})*(\.\d+)?',
                r'\d+(,\d{3})*(元|美元|欧元|英镑)',
                r'\d+(\.\d+)?(万|亿|千)'
            ],
            EntityType.PERSON.value: [
                r'[王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许邓冯韩曹曾彭萧蔡潘田董袁于余叶蒋杜苏魏程吕丁任沈钟徐姚卢傅钟姜崔谭陆汪范金石廖贾夏韦傅方邹熊孟秦白][\u4e00-\u9fa5]{1,2}'
            ],
            EntityType.ORGANIZATION.value: [
                r'[\u4e00-\u9fa5]*(公司|集团|企业|机构|组织|协会|学会|大学|学院|医院|银行|基金|证券)'
            ],
            EntityType.LOCATION.value: [
                r'[\u4e00-\u9fa5]*(省|市|区|县|街道|路|乡|镇|村)',
                r'[北京上海广州深圳杭州南京武汉成都西安]'
            ],
            EntityType.EMAIL.value: [
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            ],
            EntityType.PHONE.value: [
                r'1[3-9]\d{9}',
                r'\d{3,4}-\d{7,8}'
            ]
        }
        
        # 自定义词典
        self.custom_dict: Dict[str, str] = {}
    
    def add_custom_entity(self, entity_text: str, entity_type: str):
        """添加自定义实体"""
        self.custom_dict[entity_text] = entity_type
    
    def extract(self, text: str) -> List[EntityMention]:
        """提取实体"""
        mentions = []
        
        # 基于自定义词典的匹配
        for entity_text, entity_type in self.custom_dict.items():
            pattern = re.escape(entity_text)
            for match in re.finditer(pattern, text):
                mentions.append(EntityMention(
                    text=match.group(),
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0
                ))
        
        # 基于正则的匹配
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    mentions.append(EntityMention(
                        text=match.group(),
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    ))
        
        # 合并重叠提及，保留置信度最高的
        mentions = self._merge_overlapping(mentions)
        
        return sorted(mentions, key=lambda x: x.start)
    
    def _merge_overlapping(self, mentions: List[EntityMention]) -> List[EntityMention]:
        """合并重叠的实体提及"""
        if not mentions:
            return []
        
        sorted_mentions = sorted(mentions, key=lambda x: (x.start, -x.end))
        merged = []
        
        for mention in sorted_mentions:
            if not merged:
                merged.append(mention)
                continue
            
            last = merged[-1]
            if mention.start < last.end:
                # 有重叠，保留置信度高的
                if mention.confidence > last.confidence:
                    merged[-1] = mention
            else:
                merged.append(mention)
        
        return merged


class NERModel:
    """NER模型包装类"""
    
    def __init__(self, model_name: str = "bert-base-chinese"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
    
    def load(self):
        """加载模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self._model.eval()
        except ImportError:
            raise ImportError("transformers库未安装，请运行: pip install transformers")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")
    
    def extract(self, text: str) -> List[EntityMention]:
        """使用模型提取实体"""
        if self._model is None:
            self.load()
        
        from transformers import pipeline
        
        # 使用pipeline进行NER
        ner_pipeline = pipeline(
            "ner", 
            model=self._model, 
            tokenizer=self._tokenizer,
            aggregation_strategy="simple"
        )
        
        results = ner_pipeline(text)
        
        mentions = []
        for item in results:
            mentions.append(EntityMention(
                text=item["word"],
                entity_type=item["entity_group"],
                start=item["start"],
                end=item["end"],
                confidence=item["score"]
            ))
        
        return mentions


class LLMNER(NERBase):
    """基于LLM的NER"""
    
    def __init__(self, llm_client=None):
        super().__init__("llm_ner")
        self.llm_client = llm_client
    
    def extract(self, text: str) -> List[EntityMention]:
        """使用LLM提取实体"""
        if self.llm_client is None:
            return []
        
        prompt = f"""
请从以下文本中提取命名实体，以JSON格式返回：

文本：{text}

返回格式：
[
  {{"text": "实体文本", "type": "实体类型", "start": 起始位置, "end": 结束位置}}
]

支持的实体类型：PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PRODUCT, EVENT

只返回JSON数组，不要其他内容。
"""
        
        try:
            response = self.llm_client.generate(prompt)
            
            # 解析JSON响应
            import json
            entities = json.loads(response)
            
            mentions = []
            for entity in entities:
                mentions.append(EntityMention(
                    text=entity.get("text", ""),
                    entity_type=entity.get("type", "CUSTOM"),
                    start=entity.get("start", 0),
                    end=entity.get("end", 0),
                    confidence=0.95
                ))
            
            return mentions
        except Exception as e:
            print(f"LLM NER失败: {str(e)}")
            return []


class NERPipeline:
    """NER流水线 - 组合多种NER方法"""
    
    def __init__(self):
        self.ner_methods: List[Tuple[str, NERBase]] = []
        self._add_default_methods()
    
    def _add_default_methods(self):
        """添加默认NER方法"""
        rule_ner = RuleBasedNER()
        self.add_ner_method("rule", rule_ner, priority=10)
    
    def add_ner_method(self, name: str, ner: NERBase, priority: int = 0):
        """添加NER方法"""
        self.ner_methods.append((name, ner, priority))
        self.ner_methods.sort(key=lambda x: x[2], reverse=True)
    
    def extract(self, text: str, 
                entity_types: List[str] = None) -> List[EntityMention]:
        """提取实体"""
        all_mentions = []
        
        for name, ner, priority in self.ner_methods:
            mentions = ner.extract(text)
            
            if entity_types:
                mentions = [m for m in mentions if m.entity_type in entity_types]
            
            all_mentions.extend(mentions)
        
        # 合并重叠提及
        all_mentions = self._merge_mentions(all_mentions)
        
        return sorted(all_mentions, key=lambda x: x.start)
    
    def _merge_mentions(self, mentions: List[EntityMention]) -> List[EntityMention]:
        """合并多个NER方法的实体提及"""
        if not mentions:
            return []
        
        sorted_mentions = sorted(mentions, key=lambda x: (x.start, -x.end))
        merged = []
        
        for mention in sorted_mentions:
            if not merged:
                merged.append(mention)
                continue
            
            last = merged[-1]
            if mention.start < last.end:
                # 有重叠，保留置信度高的
                if mention.confidence > last.confidence:
                    merged[-1] = mention
            else:
                merged.append(mention)
        
        return merged
    
    def batch_extract(self, texts: List[str],
                     entity_types: List[str] = None) -> List[List[EntityMention]]:
        """批量提取实体"""
        return [self.extract(text, entity_types) for text in texts]


# 全局NER流水线实例
_ner_pipeline = None


def get_ner_pipeline() -> NERPipeline:
    """获取NER流水线实例"""
    global _ner_pipeline
    if _ner_pipeline is None:
        _ner_pipeline = NERPipeline()
    return _ner_pipeline


def extract_entities(text: str, 
                    entity_types: List[str] = None) -> List[EntityMention]:
    """便捷的实体提取函数"""
    return get_ner_pipeline().extract(text, entity_types)


def batch_extract_entities(texts: List[str],
                          entity_types: List[str] = None) -> List[List[EntityMention]]:
    """批量实体提取"""
    return get_ner_pipeline().batch_extract(texts, entity_types)
