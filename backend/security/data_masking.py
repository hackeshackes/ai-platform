"""
数据脱敏模块
支持多种敏感数据的脱敏处理
"""

import re
from enum import Enum
from typing import Any, Callable, Dict, Optional


class MaskingType(str, Enum):
    """脱敏类型枚举"""
    EMAIL = "email"
    PHONE = "phone"
    ID_CARD = "id_card"
    NAME = "name"
    CREDIT_CARD = "credit_card"
    CUSTOM = "custom"


class DataMasking:
    """数据脱敏处理器"""

    # 内置脱敏策略
    MASKING_STRATEGIES: Dict[MaskingType, Callable[[str], str]] = {
        MaskingType.EMAIL: lambda s: self._mask_email(s),
        MaskingType.PHONE: lambda s: self._mask_phone(s),
        MaskingType.ID_CARD: lambda s: self._mask_id_card(s),
        MaskingType.NAME: lambda s: self._mask_name(s),
        MaskingType.CREDIT_CARD: lambda s: self._mask_credit_card(s),
    }

    def __init__(self):
        self.custom_strategies: Dict[str, Callable[[str], str]] = {}

    def _mask_email(self, email: str) -> str:
        """邮箱脱敏: user@example.com -> u***@example.com"""
        if not email or "@" not in email:
            return "***@***.***"
        
        local, domain = email.rsplit("@", 1)
        masked_local = local[0] + "***" if local else "***"
        return f"{masked_local}@{domain}"

    def _mask_phone(self, phone: str) -> str:
        """手机号脱敏: 13812345678 -> 138****5678"""
        if not phone or len(phone) < 7:
            return "***"
        
        # 保留前3后4位
        return f"{phone[:3]}{'*' * 4}{phone[-4:]}"

    def _mask_id_card(self, id_card: str) -> str:
        """身份证脱敏: 110101199001011234 -> 1101***********1234"""
        if not id_card or len(id_card) < 8:
            return "***"
        
        return f"{id_card[:4]}{'*' * (len(id_card) - 8)}{id_card[-4:]}"

    def _mask_name(self, name: str) -> str:
        """姓名脱敏: 张三丰 -> 张**"""
        if not name or len(name) < 2:
            return "***"
        
        if len(name) == 2:
            return f"{name[0]}*"
        return f"{name[0]}{'*' * (len(name) - 1)}"

    def _mask_credit_card(self, card: str) -> str:
        """信用卡脱敏: 6222021234567890 -> 6222********7890"""
        if not card or len(card) < 8:
            return "***"
        
        # 保留前4后4位
        return f"{card[:4]}{'*' * (len(card) - 8)}{card[-4:]}"

    def register_custom_strategy(
        self, 
        name: str, 
        strategy: Callable[[str], str]
    ) -> None:
        """注册自定义脱敏策略"""
        self.custom_strategies[name] = strategy

    def mask(
        self,
        data: str,
        mask_type: MaskingType,
        custom_pattern: Optional[str] = None,
        replace_char: str = "*"
    ) -> str:
        """
        执行数据脱敏

        Args:
            data: 原始数据
            mask_type: 脱敏类型
            custom_pattern: 自定义正则表达式（仅CUSTOM类型使用）
            replace_char: 替换字符

        Returns:
            脱敏后的数据
        """
        # 自定义脱敏
        if mask_type == MaskingType.CUSTOM:
            if custom_pattern:
                return self._mask_custom(data, custom_pattern, replace_char)
            return "***"
        
        # 使用内置策略
        strategy = self.MASKING_STRATEGIES.get(mask_type)
        if strategy:
            return strategy(data)
        
        return "***"

    def _mask_custom(
        self,
        data: str,
        pattern: str,
        replace_char: str
    ) -> str:
        """自定义正则脱敏"""
        try:
            return re.sub(pattern, replace_char * 3, data)
        except re.error:
            return "***"

    def mask_dict(
        self,
        data: Dict[str, Any],
        fields_config: Dict[str, MaskingType]
    ) -> Dict[str, Any]:
        """
        批量脱敏字典数据

        Args:
            data: 原始字典
            fields_config: 字段配置 {字段名: 脱敏类型}

        Returns:
            脱敏后的字典
        """
        result = data.copy()
        for field, mask_type in fields_config.items():
            if field in result and isinstance(result[field], str):
                result[field] = self.mask(result[field], mask_type)
        return result

    def mask_json(
        self,
        json_data: Dict[str, Any],
        sensitive_fields: Dict[str, MaskingType]
    ) -> Dict[str, Any]:
        """
        递归脱敏JSON数据

        Args:
            json_data: 原始JSON数据（支持嵌套）
            sensitive_fields: 敏感字段配置

        Returns:
            脱敏后的数据
        """
        def process(obj: Any) -> Any:
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if key in sensitive_fields and isinstance(value, str):
                        result[key] = self.mask(value, sensitive_fields[key])
                    elif isinstance(value, (dict, list)):
                        result[key] = process(value)
                    else:
                        result[key] = value
                return result
            elif isinstance(obj, list):
                return [process(item) for item in obj]
            return obj

        return process(json_data)
