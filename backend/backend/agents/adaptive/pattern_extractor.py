"""
Pattern Extractor - Interaction Pattern Analysis
模式提取器 - 交互模式分析
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging

from .models import (
    Interaction,
    Pattern,
    Entity,
    ContextInfo,
    ExecutionStep,
    IntentType,
    ActionType
)

logger = logging.getLogger(__name__)


class PatternExtractor:
    """
    交互模式提取器
    
    从交互数据中提取有意义的模式
    """
    
    # 意图关键词映射
    INTENT_KEYWORDS = {
        IntentType.QUERY: [
            "什么", "谁", "哪里", "如何", "怎样", "多少", "为什么",
            "what", "who", "where", "how", "why", "which"
        ],
        IntentType.ACTION: [
            "做", "执行", "运行", "启动", "停止", "创建", "删除", "更新",
            "do", "execute", "run", "start", "stop", "create", "delete", "update"
        ],
        IntentType.CREATION: [
            "写", "生成", "制作", "创作", "编写", "建立",
            "write", "generate", "create", "make", "build"
        ],
        IntentType.ANALYSIS: [
            "分析", "比较", "评估", "检查", "审查", "诊断",
            "analyze", "compare", "evaluate", "check", "review", "diagnose"
        ],
        IntentType.LEARNING: [
            "学习", "理解", "掌握", "熟悉", "了解",
            "learn", "understand", "master", "know"
        ]
    }
    
    # 常见实体类型
    ENTITY_TYPES = {
        "number": r"\d+(\.\d+)?",
        "date": r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "url": r"https?://[^\s]+",
        "file_path": r"/[^\s]+|/[a-zA-Z0-9_./-]+",
        "time": r"\d{1,2}:\d{2}(:\d{2})?"
    }
    
    def __init__(self):
        """初始化模式提取器"""
        self._init_patterns()
    
    def _init_patterns(self) -> None:
        """初始化正则表达式模式"""
        self._intent_patterns = {
            IntentType.QUERY: re.compile(r"\?|？", re.IGNORECASE),
            IntentType.ACTION: re.compile(
                r"^(帮我|请|能不能|是否|能否)?\s*(做|执行|运行|启动|停止|创建|删除|更新)",
                re.IGNORECASE
            ),
            IntentType.CREATION: re.compile(
                r"(写|生成|制作|创作|编写|建立|帮我生成|帮我写)",
                re.IGNORECASE
            ),
            IntentType.ANALYSIS: re.compile(
                r"(分析|比较|评估|检查|审查|诊断)",
                re.IGNORECASE
            ),
            IntentType.LEARNING: re.compile(
                r"(学习|理解|掌握|熟悉|了解)",
                re.IGNORECASE
            )
        }
    
    async def extract(self, interaction: Interaction) -> Pattern:
        """
        从交互中提取模式
        
        Args:
            interaction: 交互对象
            
        Returns:
            提取的模式
        """
        # 解析意图
        intent, confidence = await self.parse_intent(interaction.text)
        
        # 提取实体
        entities = await self.extract_entities(interaction.text)
        
        # 分析上下文
        context = await self.analyze_context(interaction.context)
        
        # 提取执行路径
        execution_path = await self.extract_path(interaction.actions)
        
        # 创建模式对象
        pattern = Pattern(
            intent=intent,
            intent_confidence=confidence,
            entities=entities,
            context=context,
            execution_path=execution_path,
            success=True,
            reward=0.0
        )
        
        logger.info(f"Extracted pattern: {pattern.id}, intent: {intent.value}")
        return pattern
    
    async def parse_intent(self, text: str) -> Tuple[IntentType, float]:
        """
        解析用户意图
        
        Args:
            text: 用户输入文本
            
        Returns:
            (意图类型, 置信度)
        """
        # 清理文本
        clean_text = text.strip()
        
        # 检查每个意图类型
        best_intent = IntentType.UNKNOWN
        best_confidence = 0.0
        
        for intent_type, keywords in self.INTENT_KEYWORDS.items():
            confidence = self._calculate_intent_confidence(clean_text, keywords)
            
            # 检查正则模式
            pattern = self._intent_patterns.get(intent_type)
            if pattern and pattern.search(clean_text):
                confidence = max(confidence, 0.8)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_intent = intent_type
        
        # 如果没有匹配，返回UNKNOWN
        if best_confidence < 0.3:
            best_intent = IntentType.UNKNOWN
            best_confidence = 0.3
        
        return best_intent, best_confidence
    
    def _calculate_intent_confidence(self, text: str, keywords: List[str]) -> float:
        """
        计算意图置信度
        
        Args:
            text: 输入文本
            keywords: 关键词列表
            
        Returns:
            置信度分数
        """
        text_lower = text.lower()
        matched_keywords = sum(1 for kw in keywords if kw.lower() in text_lower)
        
        if matched_keywords == 0:
            return 0.0
        
        # 考虑关键词密度
        keyword_density = matched_keywords / len(keywords)
        
        # 考虑关键词位置
        position_bonus = 0.0
        for kw in keywords:
            if kw.lower() in text_lower[:50]:  # 开头位置
                position_bonus = 0.1
                break
        
        return min(1.0, keyword_density + position_bonus)
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        entities = []
        
        # 提取数字
        numbers = re.findall(self.ENTITY_TYPES["number"], text)
        for num in numbers:
            entities.append(Entity(
                name=num,
                value=num,
                type="number",
                confidence=0.95
            ))
        
        # 提取日期
        dates = re.findall(self.ENTITY_TYPES["date"], text)
        for date in dates:
            entities.append(Entity(
                name=date,
                value=date,
                type="date",
                confidence=0.9
            ))
        
        # 提取邮箱
        emails = re.findall(self.ENTITY_TYPES["email"], text)
        for email in emails:
            entities.append(Entity(
                name=email,
                value=email,
                type="email",
                confidence=0.95
            ))
        
        # 提取URL
        urls = re.findall(self.ENTITY_TYPES["url"], text)
        for url in urls:
            entities.append(Entity(
                name=url[:50] + "..." if len(url) > 50 else url,
                value=url,
                type="url",
                confidence=0.95
            ))
        
        # 提取文件路径
        file_paths = re.findall(self.ENTITY_TYPES["file_path"], text)
        for path in file_paths:
            entities.append(Entity(
                name=path,
                value=path,
                type="file_path",
                confidence=0.9
            ))
        
        # 提取名词短语（简单实现）
        noun_phrases = self._extract_noun_phrases(text)
        for phrase in noun_phrases:
            entities.append(Entity(
                name=phrase,
                value=phrase,
                type="noun_phrase",
                confidence=0.7
            ))
        
        return entities
    
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """
        提取名词短语
        
        Args:
            text: 输入文本
            
        Returns:
            名词短语列表
        """
        # 简单实现：提取连续的中文字符或英文字符组合
        chinese_pattern = re.findall(r"[\u4e00-\u9fa5]{2,}", text)
        english_pattern = re.findall(r"\b[a-zA-Z]{3,}\b", text)
        
        return chinese_pattern + english_pattern
    
    async def analyze_context(self, context: Dict[str, Any]) -> Optional[ContextInfo]:
        """
        分析上下文信息
        
        Args:
            context: 上下文字典
            
        Returns:
            上下文信息对象
        """
        if not context:
            return None
        
        return ContextInfo(
            session_id=context.get("session_id", ""),
            user_id=context.get("user_id"),
            timestamp=datetime.fromisoformat(context["timestamp"]) 
                      if "timestamp" in context else datetime.now(),
            metadata=context.get("metadata", {})
        )
    
    async def extract_path(self, actions: List[Dict[str, Any]]) -> List[ExecutionStep]:
        """
        提取执行路径
        
        Args:
            actions: 动作列表
            
        Returns:
            执行步骤列表
        """
        steps = []
        
        for idx, action in enumerate(actions):
            # 确定动作类型
            action_type_str = action.get("type", "unknown")
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                action_type = ActionType.RESPONSE
            
            step = ExecutionStep(
                step_number=idx + 1,
                action_type=action_type,
                action_name=action.get("name", action.get("tool", "unknown")),
                input_params=action.get("input", action.get("parameters", {})),
                output=action.get("output"),
                duration_ms=action.get("duration_ms", 0),
                success=action.get("success", True),
                error_message=action.get("error")
            )
            steps.append(step)
        
        return steps
    
    async def batch_extract(self, interactions: List[Interaction]) -> List[Pattern]:
        """
        批量提取模式
        
        Args:
            interactions: 交互列表
            
        Returns:
            模式列表
        """
        patterns = []
        
        for interaction in interactions:
            pattern = await self.extract(interaction)
            patterns.append(pattern)
        
        logger.info(f"Batch extracted {len(patterns)} patterns")
        return patterns
