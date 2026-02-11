"""
数据脱敏引擎 (Data Masking Engine)
支持多种脱敏策略：正则替换、哈希、截断、加密
"""

import re
import hashlib
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass


class MaskingStrategy(Enum):
    """脱敏策略枚举"""
    REDACT = "redact"           # 完全替换为 ***
    HASH = "hash"               # SHA256哈希
    PARTIAL = "partial"         # 部分掩码 (如 138****8888)
    MASK_CHAR = "mask_char"     # 用指定字符替换
    TRUNCATE = "truncate"       # 截断
    EMAIL = "email"             # 邮箱脱敏
    PHONE = "phone"             # 手机号脱敏
    CARD = "card"               # 银行卡号脱敏
    ID_CARD = "id_card"         # 身份证号脱敏


@dataclass
class MaskingRule:
    """脱敏规则配置"""
    field_name: str
    strategy: MaskingStrategy
    preserve_chars: int = 4      # 保留字符数
    mask_char: str = "*"         # 掩码字符
    salt: str = ""               # 哈希盐值


class DataMaskingEngine:
    """数据脱敏引擎"""
    
    def __init__(self):
        self.rules: List[MaskingRule] = []
        self.builtin_patterns = {
            MaskingStrategy.EMAIL: re.compile(r'^[\w.-]+@[\w.-]+\.\w+$'),
            MaskingStrategy.PHONE: re.compile(r'^1[3-9]\d{9}$'),
            MaskingStrategy.CARD: re.compile(r'^\d{16,19}$'),
            MaskingStrategy.ID_CARD: re.compile(r'^\d{17}[\dXx]$'),
        }
    
    def add_rule(self, rule: MaskingRule) -> None:
        """添加脱敏规则"""
        self.rules.append(rule)
    
    def add_rules_from_config(self, config: Dict[str, Any]) -> None:
        """从配置字典添加规则"""
        for field, settings in config.items():
            if isinstance(settings, dict):
                strategy = MaskingStrategy(settings.get('strategy', 'redact'))
                self.add_rule(MaskingRule(
                    field_name=field,
                    strategy=strategy,
                    preserve_chars=settings.get('preserve_chars', 4),
                    mask_char=settings.get('mask_char', '*'),
                    salt=settings.get('salt', '')
                ))
    
    def mask_value(self, value: Any, strategy: MaskingStrategy, 
                   preserve_chars: int = 4, mask_char: str = "*") -> Any:
        """根据策略脱敏单个值"""
        if value is None:
            return None
        
        str_value = str(value)
        
        if not str_value:
            return value
        
        if strategy == MaskingStrategy.REDACT:
            return mask_char * len(str_value)
        
        elif strategy == MaskingStrategy.HASH:
            return hashlib.sha256(f"{str_value}".encode()).hexdigest()
        
        elif strategy == MaskingStrategy.PARTIAL:
            if len(str_value) <= preserve_chars:
                return mask_char * len(str_value)
            start = len(str_value) - preserve_chars
            return mask_char * start + str_value[start:]
        
        elif strategy == MaskingStrategy.MASK_CHAR:
            return mask_char * len(str_value)
        
        elif strategy == MaskingStrategy.TRUNCATE:
            return str_value[:preserve_chars] + mask_char * (len(str_value) - preserve_chars)
        
        elif strategy == MaskingStrategy.EMAIL:
            if '@' in str_value:
                parts = str_value.split('@')
                local = parts[0]
                domain = parts[1] if len(parts) > 1 else ''
                masked_local = local[0] + mask_char * (len(local) - 1) if len(local) > 1 else mask_char
                return f"{masked_local}@{domain}"
            return mask_char * len(str_value)
        
        elif strategy == MaskingStrategy.PHONE:
            if len(str_value) >= 7:
                return str_value[:3] + mask_char * 4 + str_value[7:]
            return mask_char * len(str_value)
        
        elif strategy == MaskingStrategy.CARD:
            if len(str_value) >= 4:
                return mask_char * (len(str_value) - 4) + str_value[-4:]
            return mask_char * len(str_value)
        
        elif strategy == MaskingStrategy.ID_CARD:
            if len(str_value) >= 8:
                return str_value[:4] + mask_char * (len(str_value) - 8) + str_value[-4:]
            return mask_char * len(str_value)
        
        return str_value
    
    def mask_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """对整个文档应用脱敏规则"""
        masked = document.copy()
        
        for rule in self.rules:
            if rule.field_name in masked:
                masked[rule.field_name] = self.mask_value(
                    masked[rule.field_name],
                    rule.strategy,
                    rule.preserve_chars,
                    rule.mask_char
                )
        
        return masked
    
    def mask_list(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量脱敏文档列表"""
        return [self.mask_document(doc) for doc in documents]
    
    def auto_detect_and_mask(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """自动检测字段类型并脱敏"""
        masked = document.copy()
        
        for field, value in document.items():
            if value is None or not str(value):
                continue
            
            str_value = str(value)
            
            # 检测邮箱
            if self.builtin_patterns[MaskingStrategy.EMAIL].match(str_value):
                masked[field] = self.mask_value(value, MaskingStrategy.EMAIL)
            # 检测手机号
            elif self.builtin_patterns[MaskingStrategy.PHONE].match(str_value):
                masked[field] = self.mask_value(value, MaskingStrategy.PHONE)
            # 检测银行卡
            elif self.builtin_patterns[MaskingStrategy.CARD].match(str_value):
                masked[field] = self.mask_value(value, MaskingStrategy.CARD)
            # 检测身份证
            elif self.builtin_patterns[MaskingStrategy.ID_CARD].match(str_value):
                masked[field] = self.mask_value(value, MaskingStrategy.ID_CARD)
            # 敏感字段名模式匹配
            elif any(pattern in field.lower() for pattern in ['password', 'secret', 'token']):
                masked[field] = self.mask_value(value, MaskingStrategy.REDACT)
        
        return masked


# 默认实例
masking_engine = DataMaskingEngine()


def get_masking_engine() -> DataMaskingEngine:
    """获取全局脱敏引擎实例"""
    return masking_engine
