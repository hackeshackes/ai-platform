"""
AutoML Feature - 自动化特征工程
Automated Feature Engineering Module for AutoML

提供自动化特征工程功能:
- 特征生成
- 特征选择
- 特征编码
- 特征重要性分析
- 特征管道自动化
"""
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from uuid import uuid4
import json
import re
import warnings
from pathlib import Path
from collections import defaultdict

# 抑制sklearn弃用警告
warnings.filterwarnings('ignore', category=FutureWarning)

class FeatureType(Enum):
    """特征类型"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BINARY = "binary"

class TransformationType(Enum):
    """特征转换类型"""
    STANDARD_SCALER = "standard_scaler"
    MIN_MAX_SCALER = "min_max_scaler"
    ROBUST_SCALER = "robust_scaler"
    LABEL_ENCODER = "label_encoder"
    ONE_HOT = "one_hot"
    TARGET_ENCODER = "target_encoder"
    BINNING = "binning"
    LOG_TRANSFORM = "log_transform"
    POLYNOMIAL = "polynomial"
    INTERACTION = "interaction"

@dataclass
class FeatureInfo:
    """
    特征信息
    
    Attributes:
        name: 特征名称
        feature_type: 特征类型
        dtype: 数据类型
        missing_rate: 缺失率
        cardinality: 基数(唯一值数量)
        importance: 重要性分数
        transformed_name: 转换后的特征名
        transformation: 应用的转换
    """
    name: str
    feature_type: FeatureType
    dtype: str
    missing_rate: float = 0.0
    cardinality: int = 0
    importance: float = 0.0
    transformed_name: str = ""
    transformation: Optional[str] = None
    statistics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "feature_type": self.feature_type.value,
            "dtype": self.dtype,
            "missing_rate": self.missing_rate,
            "cardinality": self.cardinality,
            "importance": self.importance,
            "transformed_name": self.transformed_name,
            "transformation": self.transformation,
            "statistics": self.statistics
        }

@dataclass
class GeneratedFeature:
    """
    生成的新特征
    
    Attributes:
        feature_id: 特征ID
        source_features: 源特征列表
        name: 新特征名称
        description: 特征描述
        transformation: 转换类型
        importance: 重要性分数
        dtype: 数据类型
    """
    feature_id: str
    source_features: List[str]
    name: str
    description: str
    transformation: TransformationType
    importance: float = 0.0
    dtype: str = "float64"
    
    def to_dict(self) -> Dict:
        return {
            "feature_id": self.feature_id,
            "source_features": self.source_features,
            "name": self.name,
            "description": self.description,
            "transformation": self.transformation.value,
            "importance": self.importance,
            "dtype": self.dtype
        }

@dataclass
class FeatureEngineeringResult:
    """
    特征工程结果
    
    Attributes:
        result_id: 结果ID
        original_features: 原始特征信息
        generated_features: 生成的新特征
        selected_features: 选择的最终特征
        feature_pipeline: 特征处理管道
        feature_importance: 特征重要性排名
        total_features: 原始特征数
        final_features: 最终特征数
        created_at: 创建时间
    """
    result_id: str
    original_features: List[FeatureInfo]
    generated_features: List[GeneratedFeature]
    selected_features: List[str]
    feature_pipeline: Dict[str, Any]
    feature_importance: Dict[str, float]
    total_features: int
    final_features: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "original_features": [f.to_dict() for f in self.original_features],
            "generated_features": [f.to_dict() for f in self.generated_features],
            "selected_features": self.selected_features,
            "feature_pipeline": self.feature_pipeline,
            "feature_importance": self.feature_importance,
            "total_features": self.total_features,
            "final_features": self.final_features,
            "created_at": self.created_at.isoformat()
        }


class FeatureEngineer:
    """
    自动化特征工程师
    
    提供完整的特征工程流水线自动化。
    
    Usage:
        engineer = FeatureEngineer()
        
        result = await engineer.engineer(
            X=dataframe,
            y=target,
            task_type="classification",
            time_budget=300
        )
        
        print(f"Generated {len(result.generated_features)} new features")
        print(f"Final feature set: {result.selected_features}")
    """
    
    def __init__(self, verbose: bool = True):
        """初始化特征工程师"""
        self.verbose = verbose
        self.results: Dict[str, FeatureEngineeringResult] = {}
    
    async def engineer(
        self,
        X: Any,
        y: Optional[Any] = None,
        task_type: str = "classification",
        time_budget: int = 300,
        generate_features: bool = True,
        select_features: bool = True,
        target_col: Optional[str] = None,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        datetime_cols: Optional[List[str]] = None,
        max_new_features: int = 50,
        max_selected_features: int = 100
    ) -> FeatureEngineeringResult:
        """
        执行特征工程
        
        Args:
            X: 输入数据 (DataFrame或array-like)
            y: 目标变量 (可选)
            task_type: 任务类型 (classification/regression)
            time_budget: 时间预算(秒)
            generate_features: 是否生成新特征
            select_features: 是否进行特征选择
            target_col: 目标列名
            categorical_cols: 分类特征列
            numerical_cols: 数值特征列
            datetime_cols: 日期时间特征列
            max_new_features: 最大生成新特征数
            max_selected_features: 最大选择特征数
            
        Returns:
            FeatureEngineeringResult: 特征工程结果
        """
        import pandas as pd
        import numpy as np
        
        result_id = str(uuid4())
        
        if self.verbose:
            print(f"[Feature] Starting feature engineering: {result_id}")
        
        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            if hasattr(X, "values"):
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)
        
        # 分析原始特征
        original_features = self._analyze_features(X)
        
        if self.verbose:
            print(f"[Feature] Analyzed {len(original_features)} original features")
        
        generated_features = []
        
        # 生成新特征
        if generate_features:
            if self.verbose:
                print(f"[Feature] Generating new features...")
            
            generated_features = await self._generate_features(
                X, original_features, y,
                max_features=max_new_features,
                task_type=task_type
            )
            
            if self.verbose:
                print(f"[Feature] Generated {len(generated_features)} new features")
        
        # 合并所有特征
        all_features = list(original_features)
        
        # 如果有生成特征，合并
        if generated_features:
            for gf in generated_features:
                feat_info = FeatureInfo(
                    name=gf.name,
                    feature_type=FeatureType.NUMERICAL,
                    dtype=gf.dtype,
                    transformation=gf.transformation.value
                )
                all_features.append(feat_info)
        
        # 特征选择
        selected_features = list(X.columns)
        
        if select_features and y is not None:
            if self.verbose:
                print(f"[Feature] Selecting features...")
            
            selected_features = await self._select_features(
                X, y, all_features, 
                max_features=max_selected_features,
                task_type=task_type
            )
            
            if self.verbose:
                print(f"[Feature] Selected {len(selected_features)} features")
        
        # 计算特征重要性
        feature_importance = {}
        if y is not None:
            try:
                feature_importance = self._calculate_importance(
                    X, y, selected_features, task_type
                )
            except Exception as e:
                if self.verbose:
                    print(f"[Feature] Importance calculation failed: {e}")
        
        # 构建特征管道
        feature_pipeline = self._build_pipeline(
            original_features, generated_features, selected_features
        )
        
        result = FeatureEngineeringResult(
            result_id=result_id,
            original_features=original_features,
            generated_features=generated_features,
            selected_features=selected_features,
            feature_pipeline=feature_pipeline,
            feature_importance=feature_importance,
            total_features=len(original_features),
            final_features=len(selected_features)
        )
        
        self.results[result_id] = result
        
        if self.verbose:
            print(f"[Feature] Done! Original: {len(original_features)}, Final: {len(selected_features)}")
        
        return result
    
    def _analyze_features(self, X: Any) -> List[FeatureInfo]:
        """分析数据集中的特征"""
        import pandas as pd
        import numpy as np
        
        features = []
        
        for col in X.columns:
            series = X[col]
            
            # 检测特征类型
            if pd.api.types.is_numeric_dtype(series):
                feature_type = FeatureType.NUMERICAL
            elif pd.api.types.is_datetime64_any_dtype(series):
                feature_type = FeatureType.DATETIME
            elif series.dtype == "object" or str(series.dtype).startswith("category"):
                # 检查是否是二值特征
                unique_vals = series.dropna().unique()
                if len(unique_vals) == 2:
                    feature_type = FeatureType.BINARY
                else:
                    feature_type = FeatureType.CATEGORICAL
            else:
                feature_type = FeatureType.CATEGORICAL
            
            # 计算统计信息
            missing_rate = series.isna().mean()
            cardinality = series.nunique()
            
            if feature_type == FeatureType.NUMERICAL:
                stats = {
                    "mean": series.mean() if not pd.isna(series.mean()) else 0,
                    "std": series.std() if not pd.isna(series.std()) else 0,
                    "min": series.min() if not pd.isna(series.min()) else 0,
                    "max": series.max() if not pd.isna(series.max()) else 0
                }
            else:
                stats = {
                    "top_value": str(series.mode().iloc[0]) if len(series.mode()) > 0 else "",
                    "unique_count": cardinality
                }
            
            features.append(FeatureInfo(
                name=col,
                feature_type=feature_type,
                dtype=str(series.dtype),
                missing_rate=missing_rate,
                cardinality=cardinality,
                statistics=stats
            ))
        
        return features
    
    async def _generate_features(
        self,
        X: Any,
        features: List[FeatureInfo],
        y: Optional[Any],
        max_features: int = 50,
        task_type: str = "classification"
    ) -> List[GeneratedFeature]:
        """生成新特征"""
        import pandas as pd
        import numpy as np
        
        generated = []
        df = X.copy()
        
        # 按特征类型分组
        numerical_cols = [f.name for f in features if f.feature_type == FeatureType.NUMERICAL]
        categorical_cols = [f.name for f in features if f.feature_type == FeatureType.CATEGORICAL]
        
        # 1. 数值特征的数学运算
        if len(numerical_cols) >= 2:
            # 生成特征组合
            for i, col1 in enumerate(numerical_cols[:5]):
                for col2 in numerical_cols[i+1:6]:
                    if len(generated) >= max_features:
                        break
                    
                    # 加法
                    new_name = f"{col1}_plus_{col2}"
                    df[new_name] = df[col1] + df[col2]
                    generated.append(GeneratedFeature(
                        feature_id=str(uuid4()),
                        source_features=[col1, col2],
                        name=new_name,
                        description=f"Sum of {col1} and {col2}",
                        transformation=TransformationType.INTERACTION,
                        dtype="float64"
                    ))
                    
                    # 乘法
                    if len(generated) >= max_features:
                        break
                    
                    new_name = f"{col1}_times_{col2}"
                    df[new_name] = df[col1] * df[col2]
                    generated.append(GeneratedFeature(
                        feature_id=str(uuid4()),
                        source_features=[col1, col2],
                        name=new_name,
                        description=f"Product of {col1} and {col2}",
                        transformation=TransformationType.INTERACTION,
                        dtype="float64"
                    ))
        
        # 2. 聚合特征 (如果有分组列)
        if categorical_cols and numerical_cols:
            agg_col = categorical_cols[0]
            num_col = numerical_cols[0]
            
            # 分组均值
            new_name = f"{agg_col}_mean_{num_col}"
            df[new_name] = df.groupby(agg_col)[num_col].transform("mean")
            generated.append(GeneratedFeature(
                feature_id=str(uuid4()),
                source_features=[agg_col, num_col],
                name=new_name,
                description=f"Mean of {num_col} grouped by {agg_col}",
                transformation=TransformationType.TARGET_ENCODER,
                dtype="float64"
            ))
            
            # 分组计数
            new_name = f"{agg_col}_count"
            df[new_name] = df.groupby(agg_col)[num_col].transform("count")
            generated.append(GeneratedFeature(
                feature_id=str(uuid4()),
                source_features=[agg_col],
                name=new_name,
                description=f"Count of samples in {agg_col} group",
                transformation=TransformationType.INTERACTION,
                dtype="int64"
            ))
        
        # 3. 多项式特征 (前3个数值特征)
        if len(numerical_cols) >= 1:
            poly_cols = numerical_cols[:3]
            for col in poly_cols:
                if len(generated) >= max_features:
                    break
                
                # 平方
                new_name = f"{col}_squared"
                df[new_name] = df[col] ** 2
                generated.append(GeneratedFeature(
                    feature_id=str(uuid4()),
                    source_features=[col],
                    name=new_name,
                    description=f"Square of {col}",
                    transformation=TransformationType.POLYNOMIAL,
                    dtype="float64"
                ))
        
        # 4. 对数变换 (正值数值特征)
        for col in numerical_cols[:3]:
            if len(generated) >= max_features:
                break
            
            if (df[col] > 0).all():
                new_name = f"{col}_log"
                df[new_name] = np.log1p(df[col])
                generated.append(GeneratedFeature(
                    feature_id=str(uuid4()),
                    source_features=[col],
                    name=new_name,
                    description=f"Log transform of {col}",
                    transformation=TransformationType.LOG_TRANSFORM,
                    dtype="float64"
                ))
        
        # 5. 交叉特征
        if categorical_cols and len(categorical_cols) >= 2:
            cat1, cat2 = categorical_cols[:2]
            new_name = f"{cat1}_x_{cat2}"
            df[new_name] = df[cat1].astype(str) + "_" + df[cat2].astype(str)
            generated.append(GeneratedFeature(
                feature_id=str(uuid4()),
                source_features=[cat1, cat2],
                name=new_name,
                description=f"Cross product of {cat1} and {cat2}",
                transformation=TransformationType.INTERACTION,
                dtype="object"
            ))
        
        return generated[:max_features]
    
    async def _select_features(
        self,
        X: Any,
        y: Any,
        features: List[FeatureInfo],
        max_features: int = 100,
        task_type: str = "classification"
    ) -> List[str]:
        """选择最重要的特征"""
        import pandas as pd
        import numpy as np
        from sklearn.feature_selection import (
            SelectKBest, f_classif, f_regression,
            mutual_info_classif, mutual_info_regression,
            RFE
        )
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        df = X.copy()
        
        # 移除目标列和ID列
        exclude_cols = []
        for col in df.columns:
            if df[col].dtype == "object":
                # 对分类列进行标签编码
                try:
                    df[col] = pd.factorize(df[col])[0]
                except:
                    exclude_cols.append(col)
        
        # 处理缺失值
        df = df.fillna(0)
        
        # 处理无穷值
        df = df.replace([np.inf, -np.inf], 0)
        
        # 确保所有数据为数值型
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 选择特征
        all_cols = [f for f in df.columns if f not in exclude_cols]
        
        if len(all_cols) <= max_features:
            return all_cols
        
        try:
            # 使用随机森林进行特征重要性选择
            if task_type == "classification":
                estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                scoring = mutual_info_classif
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                scoring = mutual_info_regression
            
            # 训练估计器获取重要性
            estimator.fit(df[all_cols], y)
            importances = estimator.feature_importances_
            
            # 选择top-k
            indices = np.argsort(importances)[-max_features:]
            selected = [all_cols[i] for i in indices]
            
        except Exception as e:
            if self.verbose:
                print(f"[Feature] Selection failed, using variance: {e}")
            # 回退到简单方法
            variances = df[all_cols].var()
            selected = variances.nlargest(max_features).index.tolist()
        
        return selected
    
    def _calculate_importance(
        self,
        X: Any,
        y: Any,
        features: List[str],
        task_type: str
    ) -> Dict[str, float]:
        """计算特征重要性"""
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        df = X[features].copy()
        
        # 预处理
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.factorize(df[col])[0]
        
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        try:
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            model.fit(df, y)
            importance = dict(zip(features, model.feature_importances_))
            
        except Exception as e:
            importance = {f: 1.0/len(features) for f in features}
        
        # 排序
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return importance
    
    def _build_pipeline(
        self,
        original_features: List[FeatureInfo],
        generated_features: List[GeneratedFeature],
        selected_features: List[str]
    ) -> Dict[str, Any]:
        """构建特征处理管道"""
        return {
            "pipeline_id": str(uuid4()),
            "original_features": [f.name for f in original_features],
            "generated_features": [f.to_dict() for f in generated_features],
            "selected_features": selected_features,
            "transformations": {
                "numerical": ["standard_scaler"],
                "categorical": ["label_encoder", "one_hot"],
                "datetime": ["year", "month", "day"]
            },
            "created_at": datetime.utcnow().isoformat()
        }
    
    def apply_pipeline(
        self,
        pipeline: Dict[str, Any],
        X: Any
    ) -> Any:
        """应用特征管道到新数据"""
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
        
        df = X.copy()
        
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(X)
        
        # 应用生成特征
        generated = pipeline.get("generated_features", [])
        for feat in generated:
            if feat["name"] not in df.columns:
                # 重新计算生成特征
                source = feat["source_features"]
                if len(source) == 2:
                    col1, col2 = source
                    if feat["transformation"] == "interaction":
                        df[feat["name"]] = df[col1] * df[col2]
                elif len(source) == 1:
                    col = source[0]
                    if feat["transformation"] == "polynomial":
                        df[feat["name"]] = df[col] ** 2
                    elif feat["transformation"] == "log_transform":
                        df[feat["name"]] = np.log1p(df[col].clip(lower=0))
        
        # 选择特征
        selected = pipeline.get("selected_features", [])
        available_cols = [c for c in selected if c in df.columns]
        
        return df[available_cols]
    
    def get_result(self, result_id: str) -> Optional[FeatureEngineeringResult]:
        """获取特征工程结果"""
        return self.results.get(result_id)
    
    def list_results(self, limit: int = 20) -> List[Dict]:
        """列出所有特征工程结果"""
        result = []
        for res_id, res in self.results.items():
            result.append({
                "result_id": res_id,
                "total_features": res.total_features,
                "generated_features": len(res.generated_features),
                "final_features": res.final_features,
                "created_at": res.created_at.isoformat()
            })
        
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result[:limit]


# 默认特征工程师实例
default_engineer = FeatureEngineer(verbose=True)
