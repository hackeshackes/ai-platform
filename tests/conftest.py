"""
测试配置
"""
import pytest
import sys
import os

# 添加backend到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

@pytest.fixture
def mock_cache():
    """模拟缓存"""
    class MockCache:
        def __init__(self):
            self.data = {}
        
        def get(self, key):
            return self.data.get(key)
        
        def set(self, key, value, ttl_key=None):
            self.data[key] = value
        
        def delete(self, key):
            self.data.pop(key, None)
        
        def invalidate_pattern(self, pattern):
            keys_to_delete = [k for k in self.data.keys() if pattern in k]
            for k in keys_to_delete:
                del self.data[k]
    
    return MockCache()

@pytest.fixture
def mock_db():
    """模拟数据库"""
    class MockDB:
        def __init__(self):
            self.data = {}
        
        def query(self, model):
            return MockQuery(self.data)
    
    return MockDB()

class MockQuery:
    def __init__(self, data):
        self.data = data
        self.filters = []
    
    def filter(self, *args):
        return self
    
    def all(self):
        return list(self.data.values()) if self.data else []
    
    def first(self):
        return list(self.data.values())[0] if self.data else None
