"""
加密模块
实现AES-256对称加密和数据哈希
"""

import hashlib
import os
import secrets
from base64 import b64decode, b64encode
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


class EncryptionManager:
    """
    AES-256加密管理器

    支持:
    - AES-256-GCM authenticated encryption
    - 密钥派生 (PBKDF2)
    - 数据完整性验证
    """

    def __init__(
        self,
        key: Optional[bytes] = None,
        key_length: int = 32,  # 256 bits
        nonce_length: int = 12,  # GCM recommended
        tag_length: int = 16,  # GCM tag length
        kdf_iterations: int = 100000
    ):
        """
        初始化加密管理器

        Args:
            key: 加密密钥 (32 bytes for AES-256)
            key_length: 密钥长度 (bytes)
            nonce_length: nonce长度
            tag_length: 认证标签长度
            kdf_iterations: PBKDF2迭代次数
        """
        self.key_length = key_length
        self.nonce_length = nonce_length
        self.tag_length = tag_length
        self.kdf_iterations = kdf_iterations

        if key:
            self._key = key[:key_length]
        else:
            self._key = secrets.token_bytes(key_length)

    def _derive_key(
        self,
        password: str,
        salt: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """
        从密码派生密钥

        Args:
            password: 密码
            salt: 盐值

        Returns:
            (派生密钥, 盐值)
        """
        if salt is None:
            salt = secrets.token_bytes(16)

        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self.kdf_iterations,
            dklen=self.key_length
        )

        return key, salt

    def generate_key(self) -> bytes:
        """生成新的加密密钥"""
        return secrets.token_bytes(self.key_length)

    def generate_key_from_password(self, password: str) -> Tuple[bytes, bytes]:
        """
        从密码生成密钥

        Returns:
            (密钥, 盐值)
        """
        return self._derive_key(password)

    def set_key(self, key: bytes) -> None:
        """设置加密密钥"""
        self._key = key[:self.key_length]

    def encrypt(
        self,
        data: str,
        nonce: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        加密数据

        Args:
            data: 明文数据
            nonce: 可选nonce

        Returns:
            {
                "ciphertext": Base64编码的密文,
                "nonce": Base64编码的nonce,
                "tag": Base64编码的认证标签
            }
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            if nonce is None:
                nonce = secrets.token_bytes(self.nonce_length)

            aesgcm = AESGCM(self._key)
            ciphertext = aesgcm.encrypt(nonce, data.encode("utf-8"), None)

            return {
                "ciphertext": b64encode(ciphertext).decode("ascii"),
                "nonce": b64encode(nonce).decode("ascii"),
                "tag": b64encode(ciphertext[-self.tag_length:]).decode("ascii")
            }
        except ImportError:
            # Fallback: 简单XOR加密（不推荐生产使用）
            return self._fallback_encrypt(data, nonce)

    def decrypt(
        self,
        ciphertext: str,
        nonce: str,
        tag: Optional[str] = None
    ) -> str:
        """
        解密数据

        Args:
            ciphertext: Base64编码的密文
            nonce: Base64编码的nonce
            tag: Base64编码的认证标签

        Returns:
            明文数据
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce_bytes = b64decode(nonce)
            ciphertext_bytes = b64decode(ciphertext)

            aesgcm = AESGCM(self._key)
            plaintext = aesgcm.decrypt(nonce_bytes, ciphertext_bytes, None)

            return plaintext.decode("utf-8")
        except ImportError:
            # Fallback
            return self._fallback_decrypt(ciphertext, nonce)

    def _fallback_encrypt(
        self,
        data: str,
        nonce: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """简单XOR加密（仅作为后备）"""
        if nonce is None:
            nonce = secrets.token_bytes(self.nonce_length)

        key_stream = self._key
        data_bytes = data.encode("utf-8")
        ciphertext = bytes(
            a ^ b for a, b in zip(data_bytes, key_stream * ((len(data_bytes) // len(key_stream)) + 1))
        )

        return {
            "ciphertext": b64encode(ciphertext).decode("ascii"),
            "nonce": b64encode(nonce).decode("ascii"),
            "tag": ""
        }

    def _fallback_decrypt(self, ciphertext: str, nonce: str) -> str:
        """简单XOR解密"""
        nonce_bytes = b64decode(nonce)
        ciphertext_bytes = b64decode(ciphertext)
        key_stream = self._key

        plaintext = bytes(
            a ^ b for a, b in zip(ciphertext_bytes, key_stream * ((len(ciphertext_bytes) // len(key_stream)) + 1))
        )

        return plaintext.decode("utf-8")

    def encrypt_file(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        加密文件

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径

        Returns:
            加密结果信息
        """
        if output_path is None:
            output_path = input_path + ".enc"

        nonce = secrets.token_bytes(self.nonce_length)

        with open(input_path, "rb") as f:
            data = f.read()

        result = self.encrypt(data.decode("latin-1"), nonce)

        # 写入加密文件
        with open(output_path, "w", encoding="utf-8") as f:
            import json
            json.dump(result, f, indent=2)

        return {
            "input_path": input_path,
            "output_path": output_path,
            "nonce": result["nonce"],
            "size": len(data)
        }

    def decrypt_file(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        解密文件

        Args:
            input_path: 加密文件路径
            output_path: 输出文件路径

        Returns:
            解密后的文件内容
        """
        import json

        with open(input_path, "r", encoding="utf-8") as f:
            encrypted_data = json.load(f)

        plaintext = self.decrypt(
            encrypted_data["ciphertext"],
            encrypted_data["nonce"],
            encrypted_data.get("tag")
        )

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(plaintext)

        return plaintext

    # ============= 哈希功能 =============

    def hash(
        self,
        data: str,
        algorithm: str = "sha256"
    ) -> str:
        """
        计算数据哈希

        Args:
            data: 输入数据
            algorithm: 哈希算法 (sha256, sha512, blake2b)

        Returns:
            十六进制哈希值
        """
        if algorithm == "sha256":
            return hashlib.sha256(data.encode("utf-8")).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(data.encode("utf-8")).hexdigest()
        elif algorithm == "blake2b":
            return hashlib.blake2b(data.encode("utf-8")).hexdigest()
        elif algorithm == "blake2s":
            return hashlib.blake2s(data.encode("utf-8")).hexdigest()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def hash_file(self, path: str, algorithm: str = "sha256") -> str:
        """
        计算文件哈希

        Args:
            path: 文件路径
            algorithm: 哈希算法

        Returns:
            十六进制哈希值
        """
        hash_obj = hashlib.new(algorithm)

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    def hash_password(
        self,
        password: str,
        salt: Optional[bytes] = None
    ) -> Tuple[str, str]:
        """
        安全地哈希密码

        Args:
            password: 密码
            salt: 可选盐值

        Returns:
            (哈希值, 盐值)
        """
        if salt is None:
            salt = secrets.token_bytes(32)

        # 使用scrypt（比PBKDF2更安全）
        import hashlib

        hashed = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=16384,
            r=8,
            p=1,
            dklen=64
        )

        return b64encode(hashed).decode("ascii"), b64encode(salt).decode("ascii")

    def verify_password(
        self,
        password: str,
        password_hash: str,
        salt: str
    ) -> bool:
        """
        验证密码

        Args:
            password: 待验证密码
            password_hash: 存储的哈希值
            salt: 盐值

        Returns:
            是否匹配
        """
        import hashlib

        salt_bytes = b64decode(salt)
        hashed = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt_bytes,
            n=16384,
            r=8,
            p=1,
            dklen=64
        )

        return secrets.compare_digest(
            hashed,
            b64decode(password_hash)
        )

    def generate_token(self, length: int = 32) -> str:
        """生成安全随机令牌"""
        return secrets.token_urlsafe(length)

    def secure_compare(self, a: str, b: str) -> bool:
        """安全字符串比较（防止时序攻击）"""
        if len(a) != len(b):
            return False
        return secrets.compare_digest(a, b)

    def encrypt_dict(
        self,
        data: Dict[str, Any],
        fields_to_encrypt: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        加密字典中的特定字段

        Args:
            data: 原始字典
            fields_to_encrypt: 要加密的字段列表

        Returns:
            加密后的字典
        """
        fields_to_encrypt = fields_to_encrypt or []
        result = data.copy()

        for field in fields_to_encrypt:
            if field in result:
                if isinstance(result[field], str):
                    encrypted = self.encrypt(result[field])
                    result[field] = encrypted
                elif isinstance(result[field], dict):
                    result[field] = self.encrypt_dict(result[field], fields_to_encrypt)

        return result

    def decrypt_dict(
        self,
        data: Dict[str, Any],
        fields_to_decrypt: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        解密字典中的特定字段

        Args:
            data: 加密的字典
            fields_to_decrypt: 要解密的字段列表

        Returns:
            解密后的字典
        """
        fields_to_decrypt = fields_to_decrypt or []
        result = data.copy()

        for field in fields_to_decrypt:
            if field in result:
                if isinstance(result[field], dict):
                    decrypted = self.decrypt(
                        result[field]["ciphertext"],
                        result[field]["nonce"],
                        result[field].get("tag")
                    )
                    result[field] = decrypted

        return result
