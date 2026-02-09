"""
Feature Store测试
"""
import pytest
from backend.feature_store.store import FeatureStore, Feature, FeatureGroup

@pytest.fixture
def feature_store():
    """测试用的FeatureStore"""
    fs = FeatureStore()
    return fs

@pytest.fixture
def sample_feature_group_data():
    """测试用的特征组数据"""
    return {
        "name": "user_features",
        "description": "用户特征",
        "owner_id": "user123",
        "features": [
            {"name": "age", "dtype": "int32", "description": "用户年龄"},
            {"name": "gender", "dtype": "string", "description": "性别"},
            {"name": "income", "dtype": "float64", "description": "收入"}
        ],
        "source_type": "batch"
    }

class TestFeatureStore:
    """FeatureStore测试类"""
    
    @pytest.mark.asyncio
    async def test_create_feature_group(self, feature_store, sample_feature_group_data):
        """测试创建特征组"""
        group = await feature_store.create_feature_group(
            name=sample_feature_group_data["name"],
            description=sample_feature_group_data["description"],
            owner_id=sample_feature_group_data["owner_id"],
            features=sample_feature_group_data["features"],
            source_type=sample_feature_group_data["source_type"]
        )
        
        assert group is not None
        assert group.name == "user_features"
        assert len(group.features) == 3
        assert group.owner_id == "user123"
    
    @pytest.mark.asyncio
    async def test_get_feature_group(self, feature_store, sample_feature_group_data):
        """测试获取特征组"""
        # 先创建
        created = await feature_store.create_feature_group(
            name=sample_feature_group_data["name"],
            description=sample_feature_group_data["description"],
            owner_id=sample_feature_group_data["owner_id"],
            features=sample_feature_group_data["features"]
        )
        
        # 再获取
        retrieved = await feature_store.get_feature_group(created.group_id)
        
        assert retrieved is not None
        assert retrieved.group_id == created.group_id
        assert retrieved.name == "user_features"
    
    @pytest.mark.asyncio
    async def test_list_feature_groups(self, feature_store, sample_feature_group_data):
        """测试列出特征组"""
        # 创建多个特征组
        for i in range(3):
            await feature_store.create_feature_group(
                name=f"feature_group_{i}",
                description=f"描述{i}",
                owner_id="user123",
                features=sample_feature_group_data["features"]
            )
        
        # 列出
        groups = await feature_store.list_feature_groups(owner_id="user123")
        
        assert len(groups) == 3
    
    @pytest.mark.asyncio
    async def test_add_features(self, feature_store, sample_feature_group_data):
        """测试添加特征"""
        # 创建特征组
        group = await feature_store.create_feature_group(
            name=sample_feature_group_data["name"],
            description=sample_feature_group_data["description"],
            owner_id=sample_feature_group_data["owner_id"],
            features=sample_feature_group_data["features"][:2]  # 只创建2个特征
        )
        
        # 添加特征
        new_features = [
            {"name": "new_feature", "dtype": "float64", "description": "新特征"}
        ]
        
        updated_group = await feature_store.add_features(
            group.group_id,
            new_features
        )
        
        assert len(updated_group.features) == 3
    
    @pytest.mark.asyncio
    async def test_get_schema(self, feature_store, sample_feature_group_data):
        """测试获取模式"""
        # 创建特征组
        group = await feature_store.create_feature_group(
            name=sample_feature_group_data["name"],
            description=sample_feature_group_data["description"],
            owner_id=sample_feature_group_data["owner_id"],
            features=sample_feature_group_data["features"]
        )
        
        # 获取模式
        schema = await feature_store.get_schema(group.group_id)
        
        assert schema is not None
        assert "features" in schema
        assert len(schema["features"]) == 3

class TestFeatureValidation:
    """特征验证测试"""
    
    def test_feature_creation(self):
        """测试特征创建"""
        feature = Feature(
            name="test",
            dtype="int32",
            description="测试特征",
            version=1
        )
        
        assert feature.name == "test"
        assert feature.dtype == "int32"
        assert feature.version == 1
    
    def test_feature_group_structure(self):
        """测试特征组结构"""
        features = [
            Feature(name="f1", dtype="int32"),
            Feature(name="f2", dtype="float64")
        ]
        
        group = FeatureGroup(
            group_id="test123",
            name="test_group",
            description="测试",
            owner_id="user1",
            features=features
        )
        
        assert len(group.features) == 2
        assert group.created_at is not None
