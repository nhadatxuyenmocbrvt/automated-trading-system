"""
Pipeline xử lý và lựa chọn đặc trưng.
Module này cung cấp lớp FeatureSelectionPipeline để kết hợp các phương pháp
lựa chọn đặc trưng khác nhau vào một quy trình xử lý và lựa chọn đồng nhất.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any, Callable, Set
import logging
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import get_scorer
from joblib import dump, load
import time
import os
from pathlib import Path
import json
from datetime import datetime

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import get_logger
from config.system_config import MODEL_DIR

# Thiết lập logger
logger = get_logger("feature_selector")

# Import các module lựa chọn đặc trưng
from data_processors.feature_engineering.feature_selector.statistical_methods import (
    correlation_selector, chi_squared_selector, anova_selector, mutual_info_selector
)
from data_processors.feature_engineering.feature_selector.importance_methods import (
    tree_importance_selector, random_forest_importance_selector, 
    boosting_importance_selector, shap_importance_selector
)
from data_processors.feature_engineering.feature_selector.dimensionality_reduction import (
    pca_reducer, lda_reducer
)
from data_processors.feature_engineering.feature_selector.wrapper_methods import (
    forward_selection, backward_elimination, 
    recursive_feature_elimination, sequential_feature_selector
)

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Lớp transformer cho scikit-learn để lựa chọn đặc trưng.
    """
    
    def __init__(
        self,
        method: str,
        params: Dict[str, Any] = None,
        target_column: Optional[str] = None
    ):
        """
        Khởi tạo FeatureSelector.
        
        Args:
            method: Tên phương pháp lựa chọn ('correlation', 'chi_squared', 'tree', etc.)
            params: Tham số cho phương pháp lựa chọn
            target_column: Tên cột mục tiêu
        """
        self.method = method
        self.params = params or {}
        self.target_column = target_column
        self.selected_features = None
        self.is_fitted = False
    
    def _get_selector_function(self) -> Callable:
        """
        Lấy hàm lựa chọn đặc trưng dựa trên tên phương pháp.
        
        Returns:
            Hàm lựa chọn đặc trưng
        """
        selector_map = {
            # Statistical methods
            'correlation': correlation_selector,
            'chi_squared': chi_squared_selector,
            'anova': anova_selector,
            'mutual_info': mutual_info_selector,
            
            # Importance methods
            'tree': tree_importance_selector,
            'random_forest': random_forest_importance_selector,
            'boosting': boosting_importance_selector,
            'shap': shap_importance_selector,
            
            # Wrapper methods
            'forward': forward_selection,
            'backward': backward_elimination,
            'rfe': recursive_feature_elimination,
            'sequential': sequential_feature_selector
        }
        
        if self.method not in selector_map:
            logger.error(f"Phương pháp lựa chọn '{self.method}' không hỗ trợ")
            raise ValueError(f"Phương pháp lựa chọn '{self.method}' không hỗ trợ")
        
        return selector_map[self.method]
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureSelector':
        """
        Học các đặc trưng cần lựa chọn.
        
        Args:
            X: DataFrame chứa các đặc trưng
            y: Series chứa biến mục tiêu
            
        Returns:
            Self
        """
        # Đảm bảo X là một DataFrame
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)
                logger.warning("X không phải DataFrame, đã chuyển đổi sang DataFrame")
        
        # Chuẩn bị dữ liệu
        df = X.copy()
        if y is not None:
            # Thêm cột target vào DataFrame
            if self.target_column is None:
                self.target_column = 'target'
            df[self.target_column] = y
        
        # Kiểm tra target_column
        if self.target_column is not None and self.target_column not in df.columns:
            logger.error(f"Cột mục tiêu '{self.target_column}' không tồn tại trong DataFrame")
            raise ValueError(f"Cột mục tiêu '{self.target_column}' không tồn tại trong DataFrame")
        
        # Lấy hàm lựa chọn đặc trưng
        selector_func = self._get_selector_function()
        
        # Thực hiện lựa chọn đặc trưng
        try:
            params = self.params.copy()
            
            # Thêm target_column vào params nếu cần
            if self.target_column is not None and 'target_column' not in params:
                params['target_column'] = self.target_column
            
            self.selected_features = selector_func(df, **params)
            self.is_fitted = True
            
            if not self.selected_features:
                logger.warning("Không có đặc trưng nào được chọn")
            else:
                logger.info(f"Đã chọn {len(self.selected_features)} đặc trưng")
        
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện lựa chọn đặc trưng: {str(e)}")
            raise
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng lựa chọn đặc trưng đã học.
        
        Args:
            X: DataFrame chứa các đặc trưng
            
        Returns:
            DataFrame chỉ chứa các đặc trưng đã chọn
        """
        # Kiểm tra đã fit chưa
        if not self.is_fitted:
            logger.error("transform() gọi trước khi fit()")
            raise ValueError("transform() gọi trước khi fit()")
        
        # Kiểm tra có đặc trưng nào được chọn không
        if not self.selected_features:
            logger.warning("Không có đặc trưng nào được chọn, trả về DataFrame gốc")
            return X
        
        # Đảm bảo X là một DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Kiểm tra các đặc trưng đã chọn có tồn tại trong X không
        missing_features = [f for f in self.selected_features if f not in X.columns]
        if missing_features:
            logger.warning(f"Các đặc trưng đã chọn không tồn tại trong DataFrame: {missing_features}")
        
        # Lọc các đặc trưng tồn tại
        valid_features = [f for f in self.selected_features if f in X.columns]
        
        # Trả về DataFrame chỉ chứa các đặc trưng đã chọn
        return X[valid_features]

class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Lớp transformer cho scikit-learn để biến đổi đặc trưng.
    """
    
    def __init__(
        self,
        method: str,
        params: Dict[str, Any] = None,
        target_column: Optional[str] = None
    ):
        """
        Khởi tạo FeatureTransformer.
        
        Args:
            method: Tên phương pháp biến đổi ('pca', 'lda', etc.)
            params: Tham số cho phương pháp biến đổi
            target_column: Tên cột mục tiêu
        """
        self.method = method
        self.params = params or {}
        self.target_column = target_column
        self.transformer_info = None
        self.is_fitted = False
    
    def _get_transformer_function(self) -> Callable:
        """
        Lấy hàm biến đổi đặc trưng dựa trên tên phương pháp.
        
        Returns:
            Hàm biến đổi đặc trưng
        """
        transformer_map = {
            'pca': pca_reducer,
            'lda': lda_reducer
        }
        
        if self.method not in transformer_map:
            logger.error(f"Phương pháp biến đổi '{self.method}' không hỗ trợ")
            raise ValueError(f"Phương pháp biến đổi '{self.method}' không hỗ trợ")
        
        return transformer_map[self.method]
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureTransformer':
        """
        Học biến đổi đặc trưng.
        
        Args:
            X: DataFrame chứa các đặc trưng
            y: Series chứa biến mục tiêu
            
        Returns:
            Self
        """
        # Đảm bảo X là một DataFrame
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)
                logger.warning("X không phải DataFrame, đã chuyển đổi sang DataFrame")
        
        # Chuẩn bị dữ liệu
        df = X.copy()
        if y is not None:
            # Thêm cột target vào DataFrame
            if self.target_column is None:
                self.target_column = 'target'
            df[self.target_column] = y
        
        # Kiểm tra target_column
        if self.method == 'lda' and self.target_column is None:
            logger.error("LDA cần cột mục tiêu")
            raise ValueError("LDA cần cột mục tiêu")
        
        if self.target_column is not None and self.target_column not in df.columns:
            logger.error(f"Cột mục tiêu '{self.target_column}' không tồn tại trong DataFrame")
            raise ValueError(f"Cột mục tiêu '{self.target_column}' không tồn tại trong DataFrame")
        
        # Lấy hàm biến đổi đặc trưng
        transformer_func = self._get_transformer_function()
        
        # Thực hiện biến đổi đặc trưng
        try:
            params = self.params.copy()
            
            # Thêm target_column vào params nếu cần
            if self.target_column is not None and 'target_column' not in params:
                params['target_column'] = self.target_column
            
            # Thêm tham số để trả về thông tin biến đổi
            params['return_components'] = True
            params['fit'] = True
            
            result_df, transformer_info = transformer_func(df, **params)
            
            self.transformer_info = transformer_info
            self.is_fitted = True
            
            logger.info(f"Đã học biến đổi {self.method} thành công")
        
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện biến đổi đặc trưng: {str(e)}")
            raise
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng biến đổi đặc trưng đã học.
        
        Args:
            X: DataFrame chứa các đặc trưng
            
        Returns:
            DataFrame sau khi biến đổi
        """
        # Kiểm tra đã fit chưa
        if not self.is_fitted:
            logger.error("transform() gọi trước khi fit()")
            raise ValueError("transform() gọi trước khi fit()")
        
        # Đảm bảo X là một DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Lấy hàm biến đổi đặc trưng
        transformer_func = self._get_transformer_function()
        
        # Thực hiện biến đổi đặc trưng
        try:
            params = self.params.copy()
            
            # Thêm target_column vào params nếu cần
            if self.target_column is not None and 'target_column' not in params:
                params['target_column'] = self.target_column
            
            # Thêm tham số để áp dụng biến đổi đã học
            params['fit'] = False
            if self.method == 'pca':
                params['pca_model'] = self.transformer_info["model"]
            elif self.method == 'lda':
                params['lda_model'] = self.transformer_info["model"]
            
            result_df = transformer_func(X, **params)
            return result_df
        
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng biến đổi đặc trưng: {str(e)}")
            raise

class FeatureSelectionPipeline:
    """
    Pipeline lựa chọn đặc trưng kết hợp nhiều phương pháp.
    """
    
    def __init__(
        self,
        name: str = None,
        target_column: str = None,
        problem_type: str = 'auto',
        save_dir: Optional[Path] = None
    ):
        """
        Khởi tạo FeatureSelectionPipeline.
        
        Args:
            name: Tên pipeline
            target_column: Tên cột mục tiêu
            problem_type: Loại bài toán ('classification', 'regression', 'auto')
            save_dir: Thư mục lưu trữ pipeline
        """
        self.name = name or f"feature_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.target_column = target_column
        self.problem_type = problem_type
        
        if save_dir is None:
            self.save_dir = MODEL_DIR / 'feature_pipelines' / self.name
        else:
            self.save_dir = save_dir / self.name
        
        # Tạo thư mục lưu trữ
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo pipeline
        self.steps = []
        self.pipeline = None
        self.original_columns = None
        self.selected_columns = None
        self.is_fitted = False
        
        # Thông tin về pipeline
        self.pipeline_info = {
            "name": self.name,
            "target_column": self.target_column,
            "problem_type": self.problem_type,
            "steps": [],
            "created_at": datetime.now().isoformat(),
            "last_fit": None,
            "input_shape": None,
            "output_shape": None,
            "performance": {}
        }
        
        logger.info(f"Đã khởi tạo FeatureSelectionPipeline: {self.name}")
    
    def add_selector(
        self,
        method: str,
        params: Dict[str, Any] = None,
        name: Optional[str] = None
    ) -> 'FeatureSelectionPipeline':
        """
        Thêm một bước lựa chọn đặc trưng vào pipeline.
        
        Args:
            method: Tên phương pháp lựa chọn
            params: Tham số cho phương pháp lựa chọn
            name: Tên bước (mặc định là {method}_selector)
            
        Returns:
            Self
        """
        # Tạo tên bước nếu không được cung cấp
        if name is None:
            name = f"{method}_selector"
        
        # Đảm bảo tên bước là duy nhất
        base_name = name
        counter = 1
        while any(step[0] == name for step in self.steps):
            name = f"{base_name}_{counter}"
            counter += 1
        
        # Thêm bước vào pipeline
        params = params or {}
        if self.target_column is not None and 'target_column' not in params:
            params['target_column'] = self.target_column
        
        self.steps.append(
            (name, FeatureSelector(method=method, params=params, target_column=self.target_column))
        )
        
        # Thêm thông tin bước vào pipeline_info
        self.pipeline_info["steps"].append({
            "name": name,
            "type": "selector",
            "method": method,
            "params": params
        })
        
        logger.info(f"Đã thêm bước lựa chọn {method} vào pipeline: {self.name}")
        
        return self
    
    def add_transformer(
        self,
        method: str,
        params: Dict[str, Any] = None,
        name: Optional[str] = None
    ) -> 'FeatureSelectionPipeline':
        """
        Thêm một bước biến đổi đặc trưng vào pipeline.
        
        Args:
            method: Tên phương pháp biến đổi
            params: Tham số cho phương pháp biến đổi
            name: Tên bước (mặc định là {method}_transformer)
            
        Returns:
            Self
        """
        # Tạo tên bước nếu không được cung cấp
        if name is None:
            name = f"{method}_transformer"
        
        # Đảm bảo tên bước là duy nhất
        base_name = name
        counter = 1
        while any(step[0] == name for step in self.steps):
            name = f"{base_name}_{counter}"
            counter += 1
        
        # Thêm bước vào pipeline
        params = params or {}
        if self.target_column is not None and 'target_column' not in params:
            params['target_column'] = self.target_column
        
        self.steps.append(
            (name, FeatureTransformer(method=method, params=params, target_column=self.target_column))
        )
        
        # Thêm thông tin bước vào pipeline_info
        self.pipeline_info["steps"].append({
            "name": name,
            "type": "transformer",
            "method": method,
            "params": params
        })
        
        logger.info(f"Đã thêm bước biến đổi {method} vào pipeline: {self.name}")
        
        return self
    
    def add_custom_step(
        self,
        name: str,
        transformer: BaseEstimator,
        step_info: Dict[str, Any] = None
    ) -> 'FeatureSelectionPipeline':
        """
        Thêm một bước tùy chỉnh vào pipeline.
        
        Args:
            name: Tên bước
            transformer: Đối tượng transformer (phải có fit và transform)
            step_info: Thông tin về bước (tùy chọn)
            
        Returns:
            Self
        """
        # Đảm bảo tên bước là duy nhất
        base_name = name
        counter = 1
        while any(step[0] == name for step in self.steps):
            name = f"{base_name}_{counter}"
            counter += 1
        
        # Thêm bước vào pipeline
        self.steps.append((name, transformer))
        
        # Thêm thông tin bước vào pipeline_info
        self.pipeline_info["steps"].append({
            "name": name,
            "type": "custom",
            **(step_info or {})
        })
        
        logger.info(f"Đã thêm bước tùy chỉnh {name} vào pipeline: {self.name}")
        
        return self
    
    def build(self) -> 'FeatureSelectionPipeline':
        """
        Xây dựng pipeline từ các bước đã thêm.
        
        Returns:
            Self
        """
        if not self.steps:
            logger.warning("Không có bước nào trong pipeline")
            return self
        
        # Xây dựng pipeline
        self.pipeline = Pipeline(self.steps)
        
        logger.info(f"Đã xây dựng pipeline với {len(self.steps)} bước")
        
        return self
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'FeatureSelectionPipeline':
        """
        Học pipeline lựa chọn đặc trưng.
        
        Args:
            X: DataFrame chứa các đặc trưng
            y: Series chứa biến mục tiêu
            
        Returns:
            Self
        """
        # Đảm bảo X là một DataFrame
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)
                logger.warning("X không phải DataFrame, đã chuyển đổi sang DataFrame")
        
        # Đảm bảo pipeline đã được xây dựng
        if self.pipeline is None:
            logger.info("Pipeline chưa được xây dựng, đang xây dựng...")
            self.build()
        
        # Lưu các cột ban đầu
        self.original_columns = X.columns.tolist()
        
        # Bắt đầu đo thời gian
        start_time = time.time()
        
        # Học pipeline
        try:
            self.pipeline.fit(X, y)
            self.is_fitted = True
            
            # Lưu các cột đã chọn
            transformed_X = self.pipeline.transform(X)
            if isinstance(transformed_X, pd.DataFrame):
                self.selected_columns = transformed_X.columns.tolist()
            else:
                # Nếu kết quả không phải DataFrame, tạo tên cột mới
                self.selected_columns = [f"feature_{i}" for i in range(transformed_X.shape[1])]
            
            # Cập nhật thông tin pipeline
            self.pipeline_info["last_fit"] = datetime.now().isoformat()
            self.pipeline_info["input_shape"] = X.shape
            self.pipeline_info["output_shape"] = transformed_X.shape
            
            # Kết thúc đo thời gian
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.info(f"Đã học pipeline thành công trong {execution_time:.2f} giây")
            logger.info(f"Giảm từ {len(self.original_columns)} thành {len(self.selected_columns)} đặc trưng")
            
        except Exception as e:
            logger.error(f"Lỗi khi học pipeline: {str(e)}")
            raise
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng pipeline lựa chọn đặc trưng.
        
        Args:
            X: DataFrame chứa các đặc trưng
            
        Returns:
            DataFrame sau khi lựa chọn và biến đổi đặc trưng
        """
        # Kiểm tra đã fit chưa
        if not self.is_fitted:
            logger.error("transform() gọi trước khi fit()")
            raise ValueError("transform() gọi trước khi fit()")
        
        # Đảm bảo X là một DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Áp dụng pipeline
        try:
            transformed_X = self.pipeline.transform(X)
            
            # Chuyển đổi sang DataFrame nếu cần
            if not isinstance(transformed_X, pd.DataFrame):
                if self.selected_columns and len(self.selected_columns) == transformed_X.shape[1]:
                    transformed_X = pd.DataFrame(transformed_X, index=X.index, columns=self.selected_columns)
                else:
                    transformed_X = pd.DataFrame(transformed_X, index=X.index)
            
            return transformed_X
            
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng pipeline: {str(e)}")
            raise
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Học và áp dụng pipeline lựa chọn đặc trưng.
        
        Args:
            X: DataFrame chứa các đặc trưng
            y: Series chứa biến mục tiêu
            
        Returns:
            DataFrame sau khi lựa chọn và biến đổi đặc trưng
        """
        return self.fit(X, y).transform(X)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Any,
        scoring: Union[str, List[str]] = None,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Đánh giá hiệu suất của pipeline lựa chọn đặc trưng.
        
        Args:
            X: DataFrame chứa các đặc trưng
            y: Series chứa biến mục tiêu
            model: Mô hình đánh giá
            scoring: Phương pháp đánh giá ('r2', 'accuracy', 'f1', etc. hoặc danh sách các phương pháp)
            cv: Số lượng fold cho cross-validation
            
        Returns:
            Dict chứa điểm đánh giá
        """
        # Kiểm tra đã fit chưa
        if not self.is_fitted:
            logger.error("evaluate() gọi trước khi fit()")
            raise ValueError("evaluate() gọi trước khi fit()")
        
        # Áp dụng pipeline để lấy đặc trưng đã chọn
        X_transformed = self.transform(X)
        
        # Xác định loại bài toán
        if self.problem_type == 'auto':
            is_categorical = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)
            is_discrete_numeric = pd.api.types.is_numeric_dtype(y) and len(y.unique()) <= 20
            problem_type = 'classification' if (is_categorical or is_discrete_numeric) else 'regression'
        else:
            problem_type = self.problem_type
        
        # Chọn đối tượng cross-validator phù hợp
        if problem_type == 'classification':
            cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_obj = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Đánh giá với các phương pháp khác nhau
        scores = {}
        
        if isinstance(scoring, list):
            # Nhiều phương pháp đánh giá
            for score_name in scoring:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_scores = cross_val_score(model, X_transformed, y, cv=cv_obj, scoring=score_name)
                scores[score_name] = np.mean(cv_scores)
        else:
            # Một phương pháp đánh giá
            score_name = scoring or ('accuracy' if problem_type == 'classification' else 'r2')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(model, X_transformed, y, cv=cv_obj, scoring=score_name)
            scores[score_name] = np.mean(cv_scores)
        
        # So sánh với đánh giá trên tất cả các đặc trưng
        if len(self.original_columns) > len(self.selected_columns):
            logger.info("Đánh giá trên tất cả các đặc trưng gốc để so sánh...")
            
            baseline_scores = {}
            
            if isinstance(scoring, list):
                for score_name in scoring:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cv_scores = cross_val_score(model, X, y, cv=cv_obj, scoring=score_name)
                    baseline_scores[f"baseline_{score_name}"] = np.mean(cv_scores)
            else:
                score_name = scoring or ('accuracy' if problem_type == 'classification' else 'r2')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_scores = cross_val_score(model, X, y, cv=cv_obj, scoring=score_name)
                baseline_scores[f"baseline_{score_name}"] = np.mean(cv_scores)
            
            # Thêm điểm baseline vào kết quả
            scores.update(baseline_scores)
        
        # Thêm thông tin về số lượng đặc trưng
        scores["n_features"] = len(self.selected_columns)
        scores["original_n_features"] = len(self.original_columns)
        
        # Cập nhật thông tin hiệu suất của pipeline
        self.pipeline_info["performance"].update(scores)
        
        # In kết quả
        logger.info(f"Đánh giá pipeline: {scores}")
        
        return scores
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Lấy mức độ quan trọng của các đặc trưng.
        
        Returns:
            Dict chứa mức độ quan trọng của từng đặc trưng
        """
        if not self.is_fitted:
            logger.error("get_feature_importance() gọi trước khi fit()")
            raise ValueError("get_feature_importance() gọi trước khi fit()")
        
        if not self.selected_columns:
            logger.warning("Không có đặc trưng nào được chọn")
            return {}
        
        # Tạo dict đơn giản với các đặc trưng được chọn
        feature_importance = {feature: 1.0 for feature in self.selected_columns}
        
        # Nếu có bước cuối là tree hoặc random_forest hoặc boosting, lấy feature_importances_
        last_step_name, last_step = self.steps[-1]
        
        if isinstance(last_step, FeatureSelector) and last_step.method in ['tree', 'random_forest', 'boosting']:
            if hasattr(last_step, 'importance_scores_'):
                # Nếu có thuộc tính importance_scores_, sử dụng nó
                for i, feature in enumerate(self.selected_columns):
                    feature_importance[feature] = last_step.importance_scores_[i]
        
        return feature_importance
    
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Lưu pipeline vào file.
        
        Args:
            path: Đường dẫn thư mục lưu trữ (mặc định là self.save_dir)
            
        Returns:
            Đường dẫn thư mục đã lưu
        """
        if path is None:
            path = self.save_dir
        
        # Đảm bảo thư mục tồn tại
        path.mkdir(parents=True, exist_ok=True)
        
        # Lưu pipeline
        if self.pipeline is not None:
            pipeline_path = path / f"{self.name}_pipeline.joblib"
            dump(self.pipeline, pipeline_path)
            logger.info(f"Đã lưu pipeline vào {pipeline_path}")
        
        # Lưu thông tin pipeline
        info_path = path / f"{self.name}_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(self.pipeline_info, f, indent=4, ensure_ascii=False)
        
        # Lưu các cột
        columns_path = path / f"{self.name}_columns.json"
        columns_data = {
            "original_columns": self.original_columns,
            "selected_columns": self.selected_columns
        }
        with open(columns_path, 'w', encoding='utf-8') as f:
            json.dump(columns_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Đã lưu pipeline {self.name} vào {path}")
        
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path], name: Optional[str] = None) -> 'FeatureSelectionPipeline':
        """
        Tải pipeline từ file.
        
        Args:
            path: Đường dẫn thư mục chứa pipeline
            name: Tên pipeline (mặc định là lấy từ thư mục)
            
        Returns:
            Pipeline đã tải
        """
        path = Path(path)
        
        # Nếu không cung cấp tên, lấy từ tên thư mục
        if name is None:
            name = path.name
        
        # Tìm file pipeline
        pipeline_file = list(path.glob(f"{name}_pipeline.joblib"))
        if not pipeline_file:
            pipeline_file = list(path.glob("*_pipeline.joblib"))
        
        if not pipeline_file:
            logger.error(f"Không tìm thấy file pipeline trong {path}")
            raise FileNotFoundError(f"Không tìm thấy file pipeline trong {path}")
        
        pipeline_path = pipeline_file[0]
        
        # Tìm file thông tin
        info_file = list(path.glob(f"{name}_info.json"))
        if not info_file:
            info_file = list(path.glob("*_info.json"))
        
        if not info_file:
            logger.warning(f"Không tìm thấy file thông tin trong {path}")
            info_path = None
        else:
            info_path = info_file[0]
        
        # Tìm file columns
        columns_file = list(path.glob(f"{name}_columns.json"))
        if not columns_file:
            columns_file = list(path.glob("*_columns.json"))
        
        if not columns_file:
            logger.warning(f"Không tìm thấy file columns trong {path}")
            columns_path = None
        else:
            columns_path = columns_file[0]
        
        # Tải pipeline
        pipeline_obj = load(pipeline_path)
        
        # Tạo instance mới
        instance = cls(name=name)
        instance.pipeline = pipeline_obj
        instance.steps = pipeline_obj.steps
        instance.is_fitted = True
        
        # Tải thông tin pipeline
        if info_path is not None:
            with open(info_path, 'r', encoding='utf-8') as f:
                instance.pipeline_info = json.load(f)
                
                # Cập nhật các thuộc tính từ thông tin
                if "target_column" in instance.pipeline_info:
                    instance.target_column = instance.pipeline_info["target_column"]
                
                if "problem_type" in instance.pipeline_info:
                    instance.problem_type = instance.pipeline_info["problem_type"]
        
        # Tải thông tin columns
        if columns_path is not None:
            with open(columns_path, 'r', encoding='utf-8') as f:
                columns_data = json.load(f)
                
                if "original_columns" in columns_data:
                    instance.original_columns = columns_data["original_columns"]
                
                if "selected_columns" in columns_data:
                    instance.selected_columns = columns_data["selected_columns"]
        
        logger.info(f"Đã tải pipeline {name} từ {path}")
        
        return instance

def create_selection_pipeline(
    selection_methods: List[str],
    target_column: str,
    problem_type: str = 'auto',
    n_features: Optional[int] = None,
    name: Optional[str] = None
) -> FeatureSelectionPipeline:
    """
    Tạo pipeline lựa chọn đặc trưng với các phương pháp đã chọn.
    
    Args:
        selection_methods: Danh sách các phương pháp lựa chọn
        target_column: Tên cột mục tiêu
        problem_type: Loại bài toán ('classification', 'regression', 'auto')
        n_features: Số lượng đặc trưng cần chọn
        name: Tên pipeline
        
    Returns:
        FeatureSelectionPipeline đã cấu hình
    """
    # Tạo tên pipeline nếu không được cung cấp
    if name is None:
        methods_str = "_".join(selection_methods)
        name = f"pipeline_{methods_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Tạo pipeline
    pipeline = FeatureSelectionPipeline(
        name=name,
        target_column=target_column,
        problem_type=problem_type
    )
    
    # Thêm các phương pháp lựa chọn
    for method in selection_methods:
        if method in ['pca', 'lda']:
            # Phương pháp giảm chiều
            params = {}
            if n_features is not None:
                params['n_components'] = n_features
            
            pipeline.add_transformer(method=method, params=params)
        else:
            # Phương pháp lựa chọn
            params = {}
            if n_features is not None:
                if method in ['correlation', 'chi_squared', 'anova', 'mutual_info', 'tree', 'random_forest', 'boosting', 'shap']:
                    params['k'] = n_features
                elif method in ['forward', 'rfe', 'sequential']:
                    params['n_features_to_select'] = n_features
                elif method == 'backward':
                    params['min_features'] = n_features
            
            pipeline.add_selector(method=method, params=params)
    
    # Xây dựng pipeline
    pipeline.build()
    
    logger.info(f"Đã tạo pipeline lựa chọn đặc trưng: {name}")
    
    return pipeline

def evaluate_feature_selection(
    df: pd.DataFrame,
    target_column: str,
    selection_methods: List[str],
    n_features_list: List[int],
    scoring: Union[str, List[str]] = None,
    problem_type: str = 'auto',
    cv: int = 5,
    model: Optional[Any] = None
) -> pd.DataFrame:
    """
    Đánh giá hiệu suất của các phương pháp lựa chọn đặc trưng với số lượng đặc trưng khác nhau.
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu
        selection_methods: Danh sách các phương pháp lựa chọn
        n_features_list: Danh sách số lượng đặc trưng cần đánh giá
        scoring: Phương pháp đánh giá ('r2', 'accuracy', 'f1', etc. hoặc danh sách)
        problem_type: Loại bài toán ('classification', 'regression', 'auto')
        cv: Số lượng fold cho cross-validation
        model: Mô hình đánh giá (None để tự động chọn)
        
    Returns:
        DataFrame chứa kết quả đánh giá
    """
    # Kiểm tra dữ liệu đầu vào
    if target_column not in df.columns:
        logger.error(f"Cột mục tiêu '{target_column}' không tồn tại trong DataFrame")
        raise ValueError(f"Cột mục tiêu '{target_column}' không tồn tại trong DataFrame")
    
    # Xác định loại bài toán
    if problem_type == 'auto':
        is_categorical = pd.api.types.is_categorical_dtype(df[target_column]) or pd.api.types.is_object_dtype(df[target_column])
        is_discrete_numeric = pd.api.types.is_numeric_dtype(df[target_column]) and len(df[target_column].unique()) <= 20
        problem_type = 'classification' if (is_categorical or is_discrete_numeric) else 'regression'
    
    # Khởi tạo mô hình nếu chưa được cung cấp
    if model is None:
        if problem_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Chuẩn bị dữ liệu
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Chuẩn bị kết quả
    results = []
    
    # Đánh giá trên dữ liệu gốc
    pipeline_baseline = FeatureSelectionPipeline(
        name="baseline",
        target_column=target_column,
        problem_type=problem_type
    )
    pipeline_baseline.original_columns = X.columns.tolist()
    pipeline_baseline.selected_columns = X.columns.tolist()
    pipeline_baseline.is_fitted = True
    
    baseline_scores = pipeline_baseline.evaluate(X, y, model, scoring, cv)
    
    # Thêm kết quả baseline
    results.append({
        "method": "baseline",
        "n_features": len(X.columns),
        **{k: v for k, v in baseline_scores.items() if k != 'n_features' and k != 'original_n_features'}
    })
    
    # Đánh giá từng phương pháp và số lượng đặc trưng
    for method in selection_methods:
        for n_features in n_features_list:
            if n_features >= len(X.columns):
                continue
            
            logger.info(f"Đánh giá {method} với {n_features} đặc trưng...")
            
            try:
                # Tạo pipeline
                pipeline = create_selection_pipeline(
                    selection_methods=[method],
                    target_column=target_column,
                    problem_type=problem_type,
                    n_features=n_features,
                    name=f"{method}_{n_features}"
                )
                
                # Học và đánh giá pipeline
                pipeline.fit(X, y)
                scores = pipeline.evaluate(X, y, model, scoring, cv)
                
                # Thêm vào kết quả
                results.append({
                    "method": method,
                    "n_features": n_features,
                    **{k: v for k, v in scores.items() if k != 'n_features' and k != 'original_n_features'}
                })
                
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá {method} với {n_features} đặc trưng: {str(e)}")
    
    # Tạo DataFrame kết quả
    results_df = pd.DataFrame(results)
    
    return results_df