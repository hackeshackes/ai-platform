"""
数据匿名化模块
实现多种数据匿名化技术
"""

import random
import re
import string
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


class Anonymizer:
    """
    数据匿名化处理器

    支持:
    - K-匿名性
    - L-多样性
    - 泛化
    - 扰动
    - 伪名化
    """

    def __init__(self):
        self._first_names_male = [
            "张", "李", "王", "刘", "陈", "杨", "黄", "赵", "周", "吴",
            "徐", "孙", "马", "朱", "胡", "郭", "何", "高", "林", "罗"
        ]
        self._first_names_female = [
            "张", "李", "王", "刘", "陈", "杨", "黄", "赵", "周", "吴",
            "徐", "孙", "马", "朱", "胡", "郭", "何", "高", "林", "罗"
        ]
        self._last_names = ["伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "军", "洋"]

    def _generate_name(self, gender: Optional[str] = None) -> str:
        """生成随机姓名"""
        if gender == "male":
            first = random.choice(self._first_names_male)
        elif gender == "female":
            first = random.choice(self._first_names_female)
        else:
            first = random.choice(self._first_names_male + self._first_names_female)

        last = random.choice(self._last_names)
        return f"{first}{last}"

    def _generate_phone(self) -> str:
        """生成随机手机号"""
        prefixes = ["130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
                   "150", "151", "152", "153", "155", "156", "157", "158", "159",
                   "180", "181", "182", "183", "185", "186", "187", "188", "189"]
        prefix = random.choice(prefixes)
        suffix = "".join(random.choices(string.digits, k=8))
        return f"{prefix}{suffix}"

    def _generate_id_card(self) -> str:
        """生成随机身份证号"""
        # 地区码
        regions = ["110101", "310101", "440301", "330102", "510104"]
        region = random.choice(regions)

        # 生日 (18年前)
        start = datetime(1960, 1, 1)
        end = datetime(2005, 12, 31)
        birth = start + timedelta(days=random.randint(0, (end - start).days))
        birthday = birth.strftime("%Y%m%d")

        # 顺序码
        seq = f"{random.randint(1, 999):03d}"

        # 校验码
        factors = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        check_codes = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
        total = sum(int(d) * f for d, f in zip(region + birthday + seq, factors))
        check_code = check_codes[total % 11]

        return f"{region}{birthday}{seq}{check_code}"

    def _generate_email(self, name: str = None) -> str:
        """生成随机邮箱"""
        if name is None:
            name = self._generate_name().lower()

        domains = ["example.com", "test.org", "demo.net", "sample.io"]
        domain = random.choice(domains)

        # 生成随机用户名
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"{name.lower()}.{suffix}@{domain}"

    def _generate_credit_card(self) -> str:
        """生成随机信用卡号"""
        prefixes = ["4", "5", "37", "6011"]
        prefix = random.choice(prefixes)

        # 生成符合Luhn算法的卡号
        length = random.choice([15, 16])
        remaining = length - len(prefix) - 1

        body = "".join(random.choices(string.digits, k=remaining))

        # 计算校验位
        def luhn_digit(s: str) -> str:
            total = 0
            reversed_digits = list(map(int, s[::-1]))
            for i, digit in enumerate(reversed_digits):
                if i % 2 == 0:
                    digit *= 2
                    if digit > 9:
                        digit -= 9
                total += digit
            return str((10 - total % 10) % 10)

        check_digit = luhn_digit(prefix + body)
        return f"{prefix}{body}{check_digit}"

    def pseudonymize(
        self,
        value: str,
        salt: Optional[str] = None
    ) -> str:
        """
        伪名化（保持一致性）

        Args:
            value: 原始值
            salt: 盐值（确保同一值产生相同伪名）

        Returns:
            伪名
        """
        import hashlib

        combined = f"{salt or ''}{value}".encode("utf-8")
        hash_value = hashlib.sha256(combined).hexdigest()

        # 生成易读的伪名（保留字母和数字）
        return hash_value[:16]

    def anonymize_field(
        self,
        value: Any,
        field_type: str,
        gender: Optional[str] = None
    ) -> Any:
        """
        匿名化单个字段

        Args:
            value: 原始值
            field_type: 字段类型 (name, phone, email, id_card, etc.)
            gender: 性别（用于姓名）

        Returns:
            匿名化后的值
        """
        if value is None:
            return None

        field_type = field_type.lower()

        if field_type == "name":
            return self._generate_name(gender)
        elif field_type == "phone":
            return self._generate_phone()
        elif field_type == "email":
            return self._generate_email()
        elif field_type == "id_card":
            return self._generate_id_card()
        elif field_type == "credit_card":
            return self._generate_credit_card()
        elif field_type == "address":
            return f"某省某市某区{random.randint(1, 100}号"
        elif field_type == "ip":
            return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.xxx"
        elif field_type == "uuid":
            return str(uuid.uuid4())
        else:
            return value

    def generalize_number(
        self,
        value: int,
        precision: int = 10
    ) -> int:
        """
        泛化数值

        Args:
            value: 原始值
            precision: 精度（向下取整的步长）

        Returns:
            泛化后的值
        """
        return (value // precision) * precision

    def generalize_range(
        self,
        value: int,
        lower: int,
        upper: int,
        bucket_count: int = 5
    ) -> Tuple[int, int]:
        """
        将数值泛化为范围区间

        Args:
            value: 原始值
            lower: 下界
            upper: 上界
            bucket_count: 桶数量

        Returns:
            (区间下限, 区间上限)
        """
        step = (upper - lower) // bucket_count
        bucket = max(0, min(bucket_count - 1, (value - lower) // step))
        return (
            lower + bucket * step,
            lower + (bucket + 1) * step
        )

    def generalize_age(
        self,
        age: int,
        buckets: Optional[List[Tuple[int, int]]] = None
    ) -> str:
        """
        泛化年龄为年龄段

        Args:
            age: 原始年龄
            buckets: 年龄段定义

        Returns:
            年龄段字符串
        """
        buckets = buckets or [
            (0, 17, "未成年"),
            (18, 30, "青年"),
            (31, 50, "中年"),
            (51, 65, "中老年"),
            (66, 100, "老年"),
            (101, 200, "长寿")
        ]

        for lower, upper, label in buckets:
            if lower <= age <= upper:
                return label
        return "未知"

    def generalize_date(
        self,
        date: datetime,
        level: str = "year"
    ) -> str:
        """
        泛化日期

        Args:
            date: 原始日期
            level: 泛化级别 (year, quarter, month, week)

        Returns:
            泛化后的日期
        """
        if level == "year":
            return str(date.year)
        elif level == "quarter":
            quarter = (date.month - 1) // 3 + 1
            return f"{date.year}Q{quarter}"
        elif level == "month":
            return date.strftime("%Y-%m")
        elif level == "week":
            return date.strftime("%Y-W%W")
        else:
            return date.strftime("%Y-%m-%d")

    def generalize_location(
        self,
        location: str,
        levels: int = 2
    ) -> str:
        """
        泛化地理位置

        Args:
            location: 原始位置 (如 "北京市朝阳区xxx路")
            levels: 保留层级数

        Returns:
            泛化后的位置
        """
        parts = location.replace("省", ",").replace("市", ",").replace("区", ",").replace("县", ",").split(",")
        parts = [p for p in parts if p.strip()]

        if len(parts) <= levels:
            return location

        return "".join(parts[:levels]) + "..."

    def add_noise(
        self,
        value: float,
        noise_level: float = 0.1,
        distribution: str = "laplace"
    ) -> float:
        """
        添加噪声（差分隐私）

        Args:
            value: 原始值
            noise_level: 噪声级别（相对于值的比例）
            distribution: 分布类型 (laplace, gaussian)

        Returns:
            添加噪声后的值
        """
        scale = abs(value) * noise_level

        if distribution == "laplace":
            # 拉普拉斯分布
            noise = random.laplace(0, scale)
        else:
            # 高斯分布
            noise = random.gauss(0, scale)

        return round(value + noise, 2)

    def suppress(
        self,
        value: Any,
        probability: float = 0.5
    ) -> Any:
        """
        抑制（随机删除）

        Args:
            value: 原始值
            probability: 抑制概率

        Returns:
            抑制后的值（可能被替换为None或通配符）
        """
        if random.random() < probability:
            return "*" * len(str(value)) if isinstance(value, str) else None
        return value

    def shuffle(self, data: List[Any]) -> List[Any]:
        """
        打乱数据顺序（保持值不变但破坏顺序关联）

        Args:
            data: 数据列表

        Returns:
            打乱后的列表
        """
        shuffled = data.copy()
        random.shuffle(shuffled)
        return shuffled

    def k_anonymize(
        self,
        records: List[Dict[str, Any]],
        quasi_identifiers: List[str],
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        K-匿名化处理

        Args:
            records: 记录列表
            quasi_identifiers: 准标识符列表
            k: K值

        Returns:
            K-匿名化后的记录
        """
        # 统计每个准标识符组合的出现次数
        groups: Dict[Tuple, List[int]] = {}

        for i, record in enumerate(records):
            key = tuple(record.get(qi) for qi in quasi_identifiers)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        # 对小于K的组进行泛化处理
        for key, indices in groups.items():
            if len(indices) < k:
                # 需要泛化这些记录
                for idx in indices:
                    for qi in quasi_identifiers:
                        value = records[idx].get(qi)
                        if isinstance(value, int):
                            records[idx][qi] = self.generalize_number(value, 100)
                        elif isinstance(value, str) and re.match(r"\d+", value):
                            records[idx][qi] = "***"

        return records

    def anonymize_dataset(
        self,
        data: List[Dict[str, Any]],
        field_mapping: Dict[str, str],
        gender_field: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        批量匿名化数据集

        Args:
            data: 数据列表
            field_mapping: 字段映射 {原始字段: 类型}
            gender_field: 性别字段名

        Returns:
          匿名化后的数据
        """
        result = []

        for record in data:
            new_record = {}

            for field, anon_type in field_mapping.items():
                value = record.get(field)
                gender = record.get(gender_field) if gender_field else None

                if field == gender_field:
                    # 性别字段保持不变或泛化
                    continue

                new_record[field] = self.anonymize_field(value, anon_type, gender)

            # 复制其他字段
            for key, value in record.items():
                if key not in field_mapping:
                    new_record[key] = value

            result.append(new_record)

        return result

    def create_synthetic_record(
        self,
        schema: Dict[str, str],
        reference: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建合成记录

        Args:
            schema: 模式定义 {字段: 类型}
            reference: 参考数据（用于保持统计特性）

        Returns:
            合成记录
        """
        record = {}

        for field, field_type in schema.items():
            if reference and field in reference:
                # 基于参考数据添加噪声
                if isinstance(reference[field], (int, float)):
                    record[field] = self.add_noise(reference[field], 0.1)
                else:
                    record[field] = self.anonymize_field(reference[field], field_type)
            else:
                record[field] = self.anonymize_field(None, field_type)

        return record
