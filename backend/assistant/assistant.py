"""
AI Assistant模块 v2.3
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import json

@dataclass
class Conversation:
    """对话"""
    conversation_id: str
    user_id: str
    messages: List[Dict] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class AssistantResponse:
    """助手响应"""
    response_id: str
    conversation_id: str
    message: str
    suggestions: List[str] = field(default_factory=list)
    actions: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)

class AIAssistant:
    """AI Assistant - 智能助手"""
    
    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}
        self.knowledge_base: Dict[str, str] = {}
        self.rule_engine = RuleEngine()
        
        # 内置知识
        self._init_knowledge_base()
        
        # 内置规则
        self._init_rules()
    
    def _init_knowledge_base(self):
        """初始化知识库"""
        self.knowledge_base = {
            "training_failed": "训练失败可能原因:\n1. 数据格式错误\n2. 内存不足\n3. 超参数不合适\n4. 模型结构问题\n\n建议检查日志和监控指标。",
            
            "model_deployment": "模型部署步骤:\n1. 确保模型已注册\n2. 选择推理服务\n3. 配置资源\n4. 启动服务\n5. 测试验证",
            
            "data_quality": "数据质量检查:\n1. 完整性检查\n2. 一致性检查\n3. 准确性检查\n4. 唯一性检查",
            
            "gpu_usage": "GPU使用优化:\n1. 调整batch size\n2. 混合精度训练\n3. 模型并行\n4. 梯度累积"
        }
    
    def _init_rules(self):
        """初始化规则引擎"""
        # 问题诊断规则
        self.rule_engine.add_rule(
            name="training_issue",
            patterns=["训练", "失败", "error", "oom", "gpu"],
            category="training",
            response_template="检测到训练问题，建议:\n1. 检查GPU内存使用\n2. 减少batch size\n3. 查看错误日志"
        )
        
        self.rule_engine.add_rule(
            name="deployment_issue",
            patterns=["部署", "服务", "启动", "端口"],
            category="deployment",
            response_template="部署相关问题，建议:\n1. 检查服务状态\n2. 验证配置文件\n3. 查看端口占用"
        )
        
        self.rule_engine.add_rule(
            name="data_issue",
            patterns=["数据", "格式", "导入", "清洗"],
            category="data",
            response_template="数据问题，建议:\n1. 检查数据格式\n2. 验证数据完整性\n3. 运行质量检查"
        )
        
        self.rule_engine.add_rule(
            name="model_issue",
            patterns=["模型", "评估", "指标", "精度"],
            category="model",
            response_template="模型问题，建议:\n1. 查看评估指标\n2. 检查数据分布\n3. 调整模型参数"
        )
    
    # 对话管理
    async def create_conversation(self, user_id: str) -> Conversation:
        """创建对话"""
        conversation = Conversation(
            conversation_id=str(uuid4()),
            user_id=user_id,
            messages=[]
        )
        
        # 添加系统消息
        conversation.messages.append({
            "role": "system",
            "content": "我是AI Platform智能助手，可以帮助你解决问题、诊断错误、提供建议。",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        self.conversations[conversation.conversation_id] = conversation
        return conversation
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取对话"""
        return self.conversations.get(conversation_id)
    
    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Conversation]:
        """列出对话"""
        conversations = list(self.conversations.values())
        
        if user_id:
            conversations = [c for c in conversations if c.user_id == user_id]
        
        conversations.sort(key=lambda c: c.updated_at, reverse=True)
        return conversations[:limit]
    
    # 消息处理
    async def chat(
        self,
        conversation_id: str,
        message: str,
        context: Optional[Dict] = None
    ) -> AssistantResponse:
        """处理对话消息"""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # 添加用户消息
        conversation.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # 分析消息
        analysis = self._analyze_message(message)
        
        # 生成响应
        response_text = await self._generate_response(
            message=message,
            conversation=conversation,
            analysis=analysis,
            context=context or {}
        )
        
        # 生成建议
        suggestions = self._generate_suggestions(analysis)
        
        # 生成操作
        actions = self._generate_actions(analysis)
        
        # 构建响应
        response = AssistantResponse(
            response_id=str(uuid4()),
            conversation_id=conversation_id,
            message=response_text,
            suggestions=suggestions,
            actions=actions,
            confidence=analysis.get("confidence", 0.8),
            sources=analysis.get("sources", [])
        )
        
        # 添加助手消息
        conversation.messages.append({
            "role": "assistant",
            "content": response.message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        conversation.updated_at = datetime.utcnow()
        
        return response
    
    def _analyze_message(self, message: str) -> Dict[str, Any]:
        """分析消息"""
        message_lower = message.lower()
        
        # 匹配规则
        matched_rules = self.rule_engine.match(message_lower)
        
        # 提取关键词
        keywords = []
        keyword_patterns = ["训练", "部署", "数据", "模型", "GPU", "错误", "性能"]
        for pattern in keyword_patterns:
            if pattern in message:
                keywords.append(pattern)
        
        # 确定类别
        category = "general"
        for rule in matched_rules:
            if rule.category != "general":
                category = rule.category
                break
        
        # 匹配知识库
        sources = []
        for key, content in self.knowledge_base.items():
            if any(kw in message_lower for kw in key.split("_")):
                sources.append(key)
        
        return {
            "keywords": keywords,
            "category": category,
            "rules": [r.name for r in matched_rules],
            "sources": sources,
            "confidence": min(0.9, 0.5 + len(matched_rules) * 0.2 + len(sources) * 0.1)
        }
    
    async def _generate_response(
        self,
        message: str,
        conversation: Conversation,
        analysis: Dict,
        context: Dict
    ) -> str:
        """生成响应"""
        # 如果匹配到规则，使用规则响应
        if analysis.get("rules"):
            for rule in self.rule_engine.rules:
                if rule.name in analysis["rules"]:
                    return rule.response_template
        
        # 如果匹配到知识库，返回知识
        if analysis.get("sources"):
            key = analysis["sources"][0]
            if key in self.knowledge_base:
                return self.knowledge_base[key]
        
        # 默认响应
        return f"我理解您的问题。\n\n关于\"{message[:50]}...\":\n\n为了更好地帮助您，请提供更多详细信息，例如：\n1. 具体错误信息\n2. 相关的配置或日志\n3. 期望的结果\n\n您也可以尝试查看相关文档或联系技术支持。"
    
    def _generate_suggestions(self, analysis: Dict) -> List[str]:
        """生成建议"""
        suggestions = []
        category = analysis.get("category", "general")
        
        if category == "training":
            suggestions = [
                "查看GPU使用情况",
                "检查数据格式",
                "调整学习率",
                "减少batch size"
            ]
        elif category == "deployment":
            suggestions = [
                "检查服务状态",
                "验证配置文件",
                "查看端口占用",
                "检查日志"
            ]
        elif category == "data":
            suggestions = [
                "运行数据质量检查",
                "查看数据分布",
                "检查缺失值",
                "验证数据格式"
            ]
        elif category == "model":
            suggestions = [
                "查看评估指标",
                "检查混淆矩阵",
                "分析特征重要性",
                "调整模型参数"
            ]
        else:
            suggestions = [
                "查看系统文档",
                "搜索相关问题",
                "联系技术支持",
                "提交工单"
            ]
        
        return suggestions[:4]
    
    def _generate_actions(self, analysis: Dict) -> List[Dict]:
        """生成可执行操作"""
        category = analysis.get("category", "general")
        
        actions = []
        
        if category == "training":
            actions = [
                {"type": "link", "label": "查看GPU监控", "target": "/gpu"},
                {"type": "link", "label": "查看训练日志", "target": "/training"},
                {"type": "action", "label": "诊断训练问题", "command": "diagnose_training"}
            ]
        elif category == "deployment":
            actions = [
                {"type": "link", "label": "查看服务状态", "target": "/inference"},
                {"type": "action", "label": "重启服务", "command": "restart_service"},
                {"type": "link", "label": "查看日志", "target": "/logs"}
            ]
        elif category == "data":
            actions = [
                {"type": "link", "label": "数据质量检查", "target": "/quality"},
                {"type": "link", "label": "数据集管理", "target": "/datasets"}
            ]
        elif category == "model":
            actions = [
                {"type": "link", "label": "模型评估", "target": "/evaluation"},
                {"type": "link", "label": "模型对比", "target": "/models"}
            ]
        
        return actions
    
    # 问题诊断
    async def diagnose(self, problem: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """诊断问题"""
        analysis = self._analyze_message(problem.lower())
        
        # 收集相关信息
        diagnosis = {
            "problem": problem,
            "category": analysis["category"],
            "keywords": analysis["keywords"],
            "possible_causes": [],
            "solutions": [],
            "recommendations": []
        }
        
        # 基于类别添加诊断结果
        if analysis["category"] == "training":
            diagnosis["possible_causes"] = [
                "GPU内存不足",
                "数据格式错误",
                "超参数不合适",
                "模型结构问题"
            ]
            diagnosis["solutions"] = [
                "减少batch size",
                "检查数据格式",
                "调整学习率",
                "使用混合精度"
            ]
            diagnosis["recommendations"] = [
                "查看GPU监控面板",
                "检查训练日志",
                "验证数据质量"
            ]
        elif analysis["category"] == "deployment":
            diagnosis["possible_causes"] = [
                "服务配置错误",
                "端口被占用",
                "依赖缺失",
                "资源不足"
            ]
            diagnosis["solutions"] = [
                "检查配置文件",
                "释放端口",
                "安装依赖",
                "增加资源"
            ]
        elif analysis["category"] == "data":
            diagnosis["possible_causes"] = [
                "数据格式不匹配",
                "缺失值过多",
                "数据分布异常",
                "编码问题"
            ]
            diagnosis["solutions"] = [
                "转换数据格式",
                "处理缺失值",
                "检查数据分布",
                "统一编码"
            ]
        else:
            diagnosis["recommendations"] = [
                "查看系统日志",
                "搜索相关文档",
                "联系技术支持"
            ]
        
        return diagnosis
    
    # 知识库管理
    def add_knowledge(self, key: str, content: str):
        """添加知识"""
        self.knowledge_base[key] = content
    
    def search_knowledge(self, query: str) -> List[Dict]:
        """搜索知识"""
        results = []
        query_lower = query.lower()
        
        for key, content in self.knowledge_base.items():
            if query_lower in key or query_lower in content.lower():
                results.append({
                    "key": key,
                    "content": content[:200],
                    "relevance": 1.0
                })
        
        return results

@dataclass
class AssistantRule:
    """助手规则"""
    name: str
    patterns: List[str]
    category: str
    response_template: str

class RuleEngine:
    """规则引擎"""
    
    def __init__(self):
        self.rules: List[AssistantRule] = []
    
    def add_rule(
        self,
        name: str,
        patterns: List[str],
        category: str,
        response_template: str
    ):
        """添加规则"""
        rule = AssistantRule(
            name=name,
            patterns=patterns,
            category=category,
            response_template=response_template
        )
        self.rules.append(rule)
    
    def match(self, text: str) -> List[AssistantRule]:
        """匹配规则"""
        matched = []
        for rule in self.rules:
            for pattern in rule.patterns:
                if pattern in text:
                    matched.append(rule)
                    break
        return matched

# AIAssistant实例
ai_assistant = AIAssistant()
