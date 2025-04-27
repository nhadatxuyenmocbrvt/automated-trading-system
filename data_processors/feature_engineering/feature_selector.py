"""
Lựa chọn và đánh giá đặc trưng cho mô hình giao dịch.
File này cung cấp các lớp và phương thức để đánh giá tầm quan trọng của đặc trưng,
lựa chọn đặc trưng, và tối ưu hóa tập đặc trưng đầu vào cho các mô hình.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Set
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import os
import sys
import time
from datetime import datetime

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import setup_logger
from config.constants import ErrorCode

class FeatureSelector:
    """
    Lớp chính để lựa chọn và đánh giá các đặc trưng.
    """
    
    def __init__(
        self,
        method: str = 'correlation',  # 'correlation', 'mutual_info', 'random_forest', 'xgboost', 'pca', 'recursive'
        n_features: Optional[int] = None,  # Số lượng đặc trưng cần chọn (None để tự động)
        threshold: float = 0.05,  # Ngưỡng quan trọng/tương quan tối thiểu
        cv_folds: int = 5,  # Số fold cross-validation
        correlation_method: str = 'pearson',  # 'pearson', 'spearman', 'kendall'
        use_target_encoding: bool = False,  # Mã hóa biến categorical dựa trên target
        handle_collinearity: bool = True,  # Xử lý đa cộng tuyến giữa các đặc trưng
        collinearity_threshold: float = 0.9,  # Ngưỡng tương quan để xác định đa cộng tuyến
        save_dir: Optional[str] = None,  # Thư mục lưu kết quả
        random_state: int = 42  # Seed để đảm bảo tính tái lập
    ):
        """
        Khởi tạo bộ chọn đặc trưng.
        
        Args:
            method: Phương pháp lựa chọn đặc trưng
                'correlation': Dựa trên tương quan với biến mục tiêu
                'mutual_info': Dựa trên thông tin tương hỗ
                'random_forest': Sử dụng feature importance từ Random Forest
                'xgboost': Sử dụng feature importance từ XGBoost
                'pca': Principal Component Analysis
                'recursive': Recursive Feature Elimination
            n_features: Số lượng đặc trưng cần chọn (None để tự động)
            threshold: Ngưỡng quan trọng/tương quan tối thiểu
            cv_folds: Số fold cross-validation
            correlation_method: Phương pháp tính tương quan
            use_target_encoding: Mã hóa biến categorical dựa trên target
            handle_collinearity: Xử lý đa cộng tuyến giữa các đặc trưng
            collinearity_threshold: Ngưỡng tương quan để xác định đa cộng tuyến
            save_dir: Thư mục lưu kết quả
            random_state: Seed để đảm bảo tính tái lập
        """
        self.logger = setup_logger("feature_selector")
        
        self.method = method
        self.n_features = n_features
        self.threshold = threshold
        self.cv_folds = cv_folds
        self.correlation_method = correlation_method
        self.use_target_encoding = use_target_encoding
        self.handle_collinearity = handle_collinearity
        self.collinearity_threshold = collinearity_threshold
        self.random_state = random_state
        
        # Thiết lập thư mục lưu kết quả
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.save_dir = None
        
        # Lưu trữ kết quả lựa chọn đặc trưng
        self.selected_features = []
        self.feature_importance = {}
        self.feature_importance_df = None
        
        # Lưu trữ thông tin khử đa cộng tuyến
        self.collinear_features = {}
        
        # Khởi tạo các model (nếu cần)
        self._init_models()
        
        self.logger.info(f"Đã khởi tạo FeatureSelector với phương pháp {self.method}")
    
    def _init_models(self) -> None:
        """
        Khởi tạo các mô hình ML sử dụng trong quá trình lựa chọn đặc trưng.
        """
        self.model = None
        
        if self.method == 'random_forest':
            try:
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                self.rf_regressor = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                self.logger.info("Đã khởi tạo Random Forest model")
            except ImportError:
                self.logger.error("Không thể import RandomForest. Vui lòng cài đặt: pip install scikit-learn")
                self.method = 'correlation'
                self.logger.warning("Chuyển sang phương pháp correlation")
        
        elif self.method == 'xgboost':
            try:
                import xgboost as xgb
                self.xgb_regressor = xgb.XGBRegressor(n_estimators=100, random_state=self.random_state)
                self.xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=self.random_state)
                self.logger.info("Đã khởi tạo XGBoost model")
            except ImportError:
                self.logger.error("Không thể import XGBoost. Vui lòng cài đặt: pip install xgboost")
                self.method = 'correlation'
                self.logger.warning("Chuyển sang phương pháp correlation")
        
        elif self.method == 'mutual_info':
            try:
                from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
                self.mi_regression = mutual_info_regression
                self.mi_classif = mutual_info_classif
                self.logger.info("Đã khởi tạo Mutual Info selector")
            except ImportError:
                self.logger.error("Không thể import mutual_info. Vui lòng cài đặt: pip install scikit-learn")
                self.method = 'correlation'
                self.logger.warning("Chuyển sang phương pháp correlation")
        
        elif self.method == 'recursive':
            try:
                from sklearn.feature_selection import RFE
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                
                # Khởi tạo mô hình cơ sở cho RFE
                self.rf_regressor = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
                
                # Số đặc trưng cho RFE
                n_features_to_select = self.n_features if self.n_features else 10
                
                # Khởi tạo RFE
                self.rfe_regressor = RFE(estimator=self.rf_regressor, n_features_to_select=n_features_to_select)
                self.rfe_classifier = RFE(estimator=self.rf_classifier, n_features_to_select=n_features_to_select)
                
                self.logger.info(f"Đã khởi tạo RFE selector với {n_features_to_select} đặc trưng")
            except ImportError:
                self.logger.error("Không thể import RFE. Vui lòng cài đặt: pip install scikit-learn")
                self.method = 'correlation'
                self.logger.warning("Chuyển sang phương pháp correlation")
        
        elif self.method == 'pca':
            try:
                from sklearn.decomposition import PCA
                
                # Số thành phần cho PCA
                n_components = self.n_features if self.n_features else 'mle'
                
                self.pca = PCA(n_components=n_components, random_state=self.random_state)
                self.logger.info(f"Đã khởi tạo PCA selector với {n_components} thành phần")
            except ImportError:
                self.logger.error("Không thể import PCA. Vui lòng cài đặt: pip install scikit-learn")
                self.method = 'correlation'
                self.logger.warning("Chuyển sang phương pháp correlation")
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        categorical_columns: Optional[List[str]] = None,
        min_features_per_group: Optional[int] = None,
        problem_type: str = 'regression'  # 'regression' hoặc 'classification'
    ) -> List[str]:
        """
        Lựa chọn các đặc trưng quan trọng từ DataFrame.
        
        Args:
            X: DataFrame chứa các đặc trưng đầu vào
            y: Series chứa biến mục tiêu
            feature_groups: Dictionary ánh xạ tên nhóm với danh sách các đặc trưng thuộc nhóm đó
            categorical_columns: Danh sách các cột categorical
            min_features_per_group: Số đặc trưng tối thiểu lấy từ mỗi nhóm (nếu feature_groups được chỉ định)
            problem_type: Loại bài toán ('regression' hoặc 'classification')
            
        Returns:
            Danh sách các đặc trưng đã chọn
        """
        if X.empty or y.empty:
            self.logger.warning("DataFrame rỗng, không thể lựa chọn đặc trưng")
            return []
        
        start_time = time.time()
        self.logger.info(f"Bắt đầu quá trình lựa chọn đặc trưng với {X.shape[1]} đặc trưng đầu vào")
        
        # Xử lý các cột categorical
        X_processed = self._preprocess_features(X, y, categorical_columns, problem_type)
        
        # Kiểm tra và xác định lại lượng feature cần chọn
        n_features = self.n_features
        if n_features is None:
            n_features = min(50, X_processed.shape[1])  # Mặc định tối đa 50 hoặc tất cả đặc trưng
        elif n_features > X_processed.shape[1]:
            n_features = X_processed.shape[1]
            self.logger.warning(f"Số đặc trưng yêu cầu lớn hơn số đặc trưng có sẵn. Chọn tất cả {n_features} đặc trưng.")
        
        # Xác định có xử lý theo nhóm không
        if feature_groups and min_features_per_group is not None:
            self.logger.info(f"Xử lý lựa chọn đặc trưng theo {len(feature_groups)} nhóm")
            selected_features = self._select_features_by_group(
                X_processed, y, feature_groups, min_features_per_group, problem_type
            )
        else:
            self.logger.info("Xử lý lựa chọn đặc trưng trên toàn bộ tập dữ liệu")
            selected_features = self._select_features_from_all(X_processed, y, n_features, problem_type)
        
        # Khử đa cộng tuyến nếu cần
        if self.handle_collinearity and len(selected_features) > 1:
            selected_features = self._remove_collinear_features(X_processed[selected_features])
        
        # Đảm bảo đặc trưng chọn là tập con của đặc trưng đầu vào
        selected_features = [f for f in selected_features if f in X.columns]
        
        # Lưu danh sách đặc trưng đã chọn
        self.selected_features = selected_features
        
        # Lưu kết quả nếu có thư mục lưu trữ
        if self.save_dir:
            self._save_results()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Đã chọn {len(selected_features)} đặc trưng trong {elapsed_time:.2f} giây")
        
        return selected_features
    
    def _preprocess_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_columns: Optional[List[str]] = None,
        problem_type: str = 'regression'
    ) -> pd.DataFrame:
        """
        Tiền xử lý dữ liệu trước khi lựa chọn đặc trưng.
        
        Args:
            X: DataFrame chứa các đặc trưng đầu vào
            y: Series chứa biến mục tiêu
            categorical_columns: Danh sách các cột categorical
            problem_type: Loại bài toán ('regression' hoặc 'classification')
            
        Returns:
            DataFrame đã được tiền xử lý
        """
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        X_processed = X.copy()
        
        # Loại bỏ các cột có quá nhiều giá trị thiếu (>50%)
        missing_ratio = X_processed.isnull().mean()
        high_missing = missing_ratio[missing_ratio > 0.5].index.tolist()
        
        if high_missing:
            X_processed = X_processed.drop(columns=high_missing)
            self.logger.info(f"Đã loại bỏ {len(high_missing)} cột có >50% giá trị thiếu: {high_missing}")
        
        # Tự động phát hiện các cột categorical nếu không được chỉ định
        if categorical_columns is None:
            categorical_columns = []
            for col in X_processed.columns:
                if X_processed[col].dtype == 'object' or X_processed[col].dtype.name == 'category':
                    categorical_columns.append(col)
                elif X_processed[col].nunique() < 10 and X_processed[col].dtype.kind in 'ifu':
                    # Biến số có ít giá trị khác nhau có thể là categorical
                    categorical_columns.append(col)
        
        # Xử lý các cột categorical
        for col in categorical_columns:
            if col not in X_processed.columns:
                continue
                
            if self.use_target_encoding and len(y) > 0:
                # Target encoding (mean encoding)
                if problem_type == 'regression':
                    # Đối với bài toán hồi quy, sử dụng giá trị trung bình của target
                    target_means = y.groupby(X_processed[col]).mean()
                    X_processed[col] = X_processed[col].map(target_means)
                else:
                    # Đối với bài toán phân loại, sử dụng tỷ lệ của class dương
                    if len(y.unique()) == 2:
                        # Giả định 1 là lớp dương
                        pos_ratio = y.groupby(X_processed[col]).mean()
                        X_processed[col] = X_processed[col].map(pos_ratio)
                    else:
                        # Nhiều lớp, sử dụng one-hot encoding
                        try:
                            dummies = pd.get_dummies(X_processed[col], prefix=col, drop_first=True)
                            X_processed = pd.concat([X_processed.drop(columns=[col]), dummies], axis=1)
                        except Exception as e:
                            self.logger.error(f"Lỗi khi one-hot encoding cột {col}: {e}")
                            # Loại bỏ cột nếu không thể xử lý
                            X_processed = X_processed.drop(columns=[col])
            else:
                # One-hot encoding cho các biến categorical
                try:
                    dummies = pd.get_dummies(X_processed[col], prefix=col, drop_first=True)
                    X_processed = pd.concat([X_processed.drop(columns=[col]), dummies], axis=1)
                except Exception as e:
                    self.logger.error(f"Lỗi khi one-hot encoding cột {col}: {e}")
                    # Loại bỏ cột nếu không thể xử lý
                    X_processed = X_processed.drop(columns=[col])
        
        # Điền các giá trị thiếu
        X_processed = X_processed.fillna(X_processed.mean())
        
        # Loại bỏ các cột có phương sai bằng 0
        zero_var_cols = [col for col in X_processed.columns if X_processed[col].var() == 0]
        if zero_var_cols:
            X_processed = X_processed.drop(columns=zero_var_cols)
            self.logger.info(f"Đã loại bỏ {len(zero_var_cols)} cột có phương sai bằng 0")
        
        # Loại bỏ các feature có tên mà không cần sử dụng (như timestamp, id,...)
        exclude_patterns = ['id', 'timestamp', 'date', 'time', 'key', 'index']
        exclude_cols = [col for col in X_processed.columns if any(pattern in col.lower() for pattern in exclude_patterns)]
        
        if exclude_cols:
            X_processed = X_processed.drop(columns=exclude_cols)
            self.logger.info(f"Đã loại bỏ {len(exclude_cols)} cột có pattern loại trừ: {exclude_cols}")
        
        return X_processed
    
    def _select_features_from_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        problem_type: str = 'regression'
    ) -> List[str]:
        """
        Lựa chọn đặc trưng từ toàn bộ tập dữ liệu.
        
        Args:
            X: DataFrame chứa các đặc trưng đầu vào đã tiền xử lý
            y: Series chứa biến mục tiêu
            n_features: Số lượng đặc trưng cần chọn
            problem_type: Loại bài toán ('regression' hoặc 'classification')
            
        Returns:
            Danh sách các đặc trưng đã chọn
        """
        if self.method == 'correlation':
            return self._select_by_correlation(X, y, n_features)
        elif self.method == 'mutual_info':
            return self._select_by_mutual_info(X, y, n_features, problem_type)
        elif self.method == 'random_forest':
            return self._select_by_random_forest(X, y, n_features, problem_type)
        elif self.method == 'xgboost':
            return self._select_by_xgboost(X, y, n_features, problem_type)
        elif self.method == 'recursive':
            return self._select_by_recursive_elimination(X, y, n_features, problem_type)
        elif self.method == 'pca':
            return self._select_by_pca(X, n_features)
        else:
            self.logger.warning(f"Phương pháp {self.method} không được hỗ trợ. Sử dụng correlation")
            return self._select_by_correlation(X, y, n_features)
    
    def _select_features_by_group(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_groups: Dict[str, List[str]],
        min_features_per_group: int,
        problem_type: str = 'regression'
    ) -> List[str]:
        """
        Lựa chọn đặc trưng từ các nhóm đặc trưng.
        
        Args:
            X: DataFrame chứa các đặc trưng đầu vào đã tiền xử lý
            y: Series chứa biến mục tiêu
            feature_groups: Dictionary ánh xạ tên nhóm với danh sách các đặc trưng thuộc nhóm đó
            min_features_per_group: Số đặc trưng tối thiểu lấy từ mỗi nhóm
            problem_type: Loại bài toán ('regression' hoặc 'classification')
            
        Returns:
            Danh sách các đặc trưng đã chọn
        """
        selected_features = []
        all_group_features = {}
        
        # Lặp qua từng nhóm đặc trưng
        for group_name, group_features in feature_groups.items():
            # Lọc các đặc trưng có trong X
            valid_features = [f for f in group_features if f in X.columns]
            
            if not valid_features:
                self.logger.warning(f"Nhóm {group_name} không có đặc trưng hợp lệ, bỏ qua")
                continue
                
            # Tạo subset X chỉ với các đặc trưng trong nhóm
            X_group = X[valid_features]
            
            # Số đặc trưng cần chọn từ nhóm này
            n_to_select = min(min_features_per_group, len(valid_features))
            
            # Chọn đặc trưng từ nhóm
            self.logger.info(f"Lựa chọn đặc trưng từ nhóm {group_name} với {len(valid_features)} đặc trưng")
            group_selected = self._select_features_from_all(X_group, y, n_to_select, problem_type)
            
            # Lưu tất cả thông tin lựa chọn đặc trưng cho nhóm
            all_group_features[group_name] = {
                'all_features': valid_features,
                'selected_features': group_selected,
                'n_features': len(valid_features),
                'n_selected': len(group_selected)
            }
            
            # Thêm vào danh sách kết quả
            selected_features.extend(group_selected)
        
        # Lưu thông tin lựa chọn đặc trưng theo nhóm
        self.group_selection_info = all_group_features
        
        # Loại bỏ các đặc trưng trùng lặp
        selected_features = list(dict.fromkeys(selected_features))
        
        return selected_features
    
    def _select_by_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int
    ) -> List[str]:
        """
        Lựa chọn đặc trưng dựa trên hệ số tương quan với biến mục tiêu.
        
        Args:
            X: DataFrame chứa các đặc trưng đầu vào đã tiền xử lý
            y: Series chứa biến mục tiêu
            n_features: Số lượng đặc trưng cần chọn
            
        Returns:
            Danh sách các đặc trưng đã chọn
        """
        # Tính hệ số tương quan giữa các đặc trưng và biến mục tiêu
        correlation = {}
        for col in X.columns:
            try:
                corr = X[col].corr(y, method=self.correlation_method)
                if pd.notnull(corr):
                    correlation[col] = abs(corr)  # Lấy giá trị tuyệt đối
            except Exception as e:
                self.logger.warning(f"Không thể tính tương quan cho cột {col}: {e}")
                correlation[col] = 0.0
        
        # Lưu thông tin tương quan
        self.feature_importance = correlation
        self.feature_importance_df = pd.DataFrame({
            'feature': list(correlation.keys()),
            'importance': list(correlation.values())
        }).sort_values('importance', ascending=False)
        
        # Lọc các đặc trưng có tương quan lớn hơn ngưỡng
        significant_features = [col for col, corr in correlation.items() if corr >= self.threshold]
        
        # Sắp xếp các đặc trưng theo độ lớn của tương quan
        sorted_features = sorted(correlation.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Chọn n đặc trưng có tương quan cao nhất
        selected_features = [feature for feature, _ in sorted_features[:n_features]]
        
        # Nếu không đủ đặc trưng đạt ngưỡng, lấy thêm từ danh sách sắp xếp
        if len(significant_features) < n_features:
            self.logger.info(f"Chỉ có {len(significant_features)} đặc trưng đạt ngưỡng tương quan {self.threshold}")
            # Thêm những đặc trưng chưa có trong significant_features
            for feature, _ in sorted_features:
                if feature not in significant_features and len(significant_features) < n_features:
                    significant_features.append(feature)
        
        self.logger.info(f"Đã chọn {len(selected_features)} đặc trưng bằng phương pháp tương quan")
        return selected_features
    
    def _select_by_mutual_info(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        problem_type: str = 'regression'
    ) -> List[str]:
        """
        Lựa chọn đặc trưng dựa trên mutual information.
        
        Args:
            X: DataFrame chứa các đặc trưng đầu vào đã tiền xử lý
            y: Series chứa biến mục tiêu
            n_features: Số lượng đặc trưng cần chọn
            problem_type: Loại bài toán ('regression' hoặc 'classification')
            
        Returns:
            Danh sách các đặc trưng đã chọn
        """
        # Kiểm tra đã khởi tạo mutual_info
        if not hasattr(self, 'mi_regression') or not hasattr(self, 'mi_classif'):
            self.logger.error("Mutual information không được khởi tạo đúng")
            return self._select_by_correlation(X, y, n_features)
        
        try:
            # Chọn phương pháp mutual info phù hợp với loại bài toán
            if problem_type == 'regression':
                mi_func = self.mi_regression
            else:
                mi_func = self.mi_classif
            
            # Tính mutual information
            mi_scores = mi_func(X.values, y.values, random_state=self.random_state)
            
            # Tạo dict ánh xạ tên đặc trưng với điểm số
            mi_dict = dict(zip(X.columns, mi_scores))
            
            # Lưu thông tin mutual information
            self.feature_importance = mi_dict
            self.feature_importance_df = pd.DataFrame({
                'feature': list(mi_dict.keys()),
                'importance': list(mi_dict.values())
            }).sort_values('importance', ascending=False)
            
            # Lọc các đặc trưng có MI lớn hơn ngưỡng
            significant_features = [col for col, score in mi_dict.items() if score >= self.threshold]
            
            # Sắp xếp các đặc trưng theo độ lớn của MI
            sorted_features = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Chọn n đặc trưng có MI cao nhất
            selected_features = [feature for feature, _ in sorted_features[:n_features]]
            
            self.logger.info(f"Đã chọn {len(selected_features)} đặc trưng bằng phương pháp mutual information")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính toán mutual information: {e}")
            # Fallback sang phương pháp tương quan
            return self._select_by_correlation(X, y, n_features)
    
    def _select_by_random_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        problem_type: str = 'regression'
    ) -> List[str]:
        """
        Lựa chọn đặc trưng dựa trên feature importance từ Random Forest.
        
        Args:
            X: DataFrame chứa các đặc trưng đầu vào đã tiền xử lý
            y: Series chứa biến mục tiêu
            n_features: Số lượng đặc trưng cần chọn
            problem_type: Loại bài toán ('regression' hoặc 'classification')
            
        Returns:
            Danh sách các đặc trưng đã chọn
        """
        # Kiểm tra đã khởi tạo Random Forest
        if not hasattr(self, 'rf_regressor') or not hasattr(self, 'rf_classifier'):
            self.logger.error("Random Forest không được khởi tạo đúng")
            return self._select_by_correlation(X, y, n_features)
        
        try:
            # Chọn model phù hợp với loại bài toán
            if problem_type == 'regression':
                model = self.rf_regressor
            else:
                model = self.rf_classifier
            
            # Huấn luyện mô hình
            model.fit(X.values, y.values)
            
            # Lấy feature importance
            importance = model.feature_importances_
            
            # Tạo dict ánh xạ tên đặc trưng với importance
            importance_dict = dict(zip(X.columns, importance))
            
            # Lưu thông tin feature importance
            self.feature_importance = importance_dict
            self.feature_importance_df = pd.DataFrame({
                'feature': list(importance_dict.keys()),
                'importance': list(importance_dict.values())
            }).sort_values('importance', ascending=False)
            
            # Lọc các đặc trưng có importance lớn hơn ngưỡng
            significant_features = [col for col, imp in importance_dict.items() if imp >= self.threshold]
            
            # Sắp xếp các đặc trưng theo độ lớn của importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Chọn n đặc trưng có importance cao nhất
            selected_features = [feature for feature, _ in sorted_features[:n_features]]
            
            self.logger.info(f"Đã chọn {len(selected_features)} đặc trưng bằng phương pháp Random Forest")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Lỗi khi huấn luyện Random Forest: {e}")
            # Fallback sang phương pháp tương quan
            return self._select_by_correlation(X, y, n_features)
    
    def _select_by_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        problem_type: str = 'regression'
    ) -> List[str]:
        """
        Lựa chọn đặc trưng dựa trên feature importance từ XGBoost.
        
        Args:
            X: DataFrame chứa các đặc trưng đầu vào đã tiền xử lý
            y: Series chứa biến mục tiêu
            n_features: Số lượng đặc trưng cần chọn
            problem_type: Loại bài toán ('regression' hoặc 'classification')
            
        Returns:
            Danh sách các đặc trưng đã chọn
        """
        # Kiểm tra đã khởi tạo XGBoost
        if not hasattr(self, 'xgb_regressor') or not hasattr(self, 'xgb_classifier'):
            self.logger.error("XGBoost không được khởi tạo đúng")
            return self._select_by_correlation(X, y, n_features)
        
        try:
            # Chọn model phù hợp với loại bài toán
            if problem_type == 'regression':
                model = self.xgb_regressor
            else:
                model = self.xgb_classifier
            
            # Huấn luyện mô hình
            model.fit(X.values, y.values)
            
            # Lấy feature importance
            importance = model.feature_importances_
            
            # Tạo dict ánh xạ tên đặc trưng với importance
            importance_dict = dict(zip(X.columns, importance))
            
            # Lưu thông tin feature importance
            self.feature_importance = importance_dict
            self.feature_importance_df = pd.DataFrame({
                'feature': list(importance_dict.keys()),
                'importance': list(importance_dict.values())
            }).sort_values('importance', ascending=False)
            
            # Lọc các đặc trưng có importance lớn hơn ngưỡng
            significant_features = [col for col, imp in importance_dict.items() if imp >= self.threshold]
            
            # Sắp xếp các đặc trưng theo độ lớn của importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Chọn n đặc trưng có importance cao nhất
            selected_features = [feature for feature, _ in sorted_features[:n_features]]
            
            self.logger.info(f"Đã chọn {len(selected_features)} đặc trưng bằng phương pháp XGBoost")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Lỗi khi huấn luyện XGBoost: {e}")
            # Fallback sang phương pháp tương quan
            return self._select_by_correlation(X, y, n_features)
    
    def _select_by_recursive_elimination(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int,
        problem_type: str = 'regression'
    ) -> List[str]:
        """
        Lựa chọn đặc trưng dựa trên Recursive Feature Elimination.
        
        Args:
            X: DataFrame chứa các đặc trưng đầu vào đã tiền xử lý
            y: Series chứa biến mục tiêu
            n_features: Số lượng đặc trưng cần chọn
            problem_type: Loại bài toán ('regression' hoặc 'classification')
            
        Returns:
            Danh sách các đặc trưng đã chọn
        """
        # Kiểm tra đã khởi tạo RFE
        if not hasattr(self, 'rfe_regressor') or not hasattr(self, 'rfe_classifier'):
            self.logger.error("RFE không được khởi tạo đúng")
            return self._select_by_correlation(X, y, n_features)
        
        try:
            # Chọn model phù hợp với loại bài toán
            if problem_type == 'regression':
                rfe = self.rfe_regressor
            else:
                rfe = self.rfe_classifier
            
            # Cập nhật số lượng đặc trưng cần chọn
            if hasattr(rfe, 'n_features_to_select'):
                rfe.n_features_to_select = min(n_features, X.shape[1])
            
            # Huấn luyện RFE
            rfe.fit(X.values, y.values)
            
            # Lấy các đặc trưng được chọn (mask là một mảng boolean)
            selected_mask = rfe.support_
            
            # Lấy ranking của từng đặc trưng (1 là đặc trưng được chọn)
            ranking = rfe.ranking_
            
            # Tạo dict ánh xạ tên đặc trưng với ranking (đảo ngược để các giá trị thấp có importance cao)
            max_rank = max(ranking)
            importance_dict = {col: (max_rank - rank + 1) / max_rank for col, rank in zip(X.columns, ranking)}
            
            # Lưu thông tin feature importance
            self.feature_importance = importance_dict
            self.feature_importance_df = pd.DataFrame({
                'feature': list(importance_dict.keys()),
                'importance': list(importance_dict.values()),
                'ranking': ranking,
                'selected': selected_mask
            }).sort_values('importance', ascending=False)
            
            # Lấy danh sách các đặc trưng được chọn
            selected_features = X.columns[selected_mask].tolist()
            
            self.logger.info(f"Đã chọn {len(selected_features)} đặc trưng bằng phương pháp RFE")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thực hiện RFE: {e}")
            # Fallback sang phương pháp tương quan
            return self._select_by_correlation(X, y, n_features)
    
    def _select_by_pca(
        self,
        X: pd.DataFrame,
        n_features: int
    ) -> List[str]:
        """
        Lựa chọn đặc trưng dựa trên Principal Component Analysis.
        PCA không chọn trực tiếp các đặc trưng gốc mà tạo ra các thành phần chính.
        Phương pháp này chọn các đặc trưng gốc đóng góp nhiều nhất vào các thành phần chính.
        
        Args:
            X: DataFrame chứa các đặc trưng đầu vào đã tiền xử lý
            n_features: Số lượng đặc trưng cần chọn
            
        Returns:
            Danh sách các đặc trưng đã chọn (từ các đặc trưng gốc)
        """
        # Kiểm tra đã khởi tạo PCA
        if not hasattr(self, 'pca'):
            self.logger.error("PCA không được khởi tạo đúng")
            return X.columns[:n_features].tolist()  # Fallback đơn giản
        
        try:
            # Cập nhật số lượng thành phần chính
            if isinstance(self.pca.n_components, int):
                self.pca.n_components = min(n_features, X.shape[1], X.shape[0])
            
            # Chạy PCA
            pca_result = self.pca.fit_transform(X.values)
            
            # Lấy loading matrix (feature importance cho mỗi thành phần)
            components = self.pca.components_
            
            # Lấy explained variance
            explained_variance = self.pca.explained_variance_ratio_
            
            # Tính toán tổng đóng góp của mỗi đặc trưng đến phương sai được giải thích
            # (tính trung bình có trọng số của các giá trị tuyệt đối của loadings)
            feature_importance = np.zeros(X.shape[1])
            for i, variance in enumerate(explained_variance):
                feature_importance += np.abs(components[i, :]) * variance
            
            # Chuẩn hóa để tổng bằng 1
            feature_importance = feature_importance / np.sum(feature_importance)
            
            # Tạo dict ánh xạ tên đặc trưng với importance
            importance_dict = dict(zip(X.columns, feature_importance))
            
            # Lưu thông tin feature importance
            self.feature_importance = importance_dict
            self.feature_importance_df = pd.DataFrame({
                'feature': list(importance_dict.keys()),
                'importance': list(importance_dict.values())
            }).sort_values('importance', ascending=False)
            
            # Sắp xếp các đặc trưng theo độ lớn của importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Chọn n đặc trưng có importance cao nhất
            selected_features = [feature for feature, _ in sorted_features[:n_features]]
            
            # Lưu thêm thông tin về các thành phần chính
            self.pca_info = {
                'n_components': self.pca.n_components,
                'explained_variance': self.pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_).tolist(),
                'components': self.pca.components_.tolist() if hasattr(self.pca, 'components_') else None
            }
            
            self.logger.info(f"Đã chọn {len(selected_features)} đặc trưng bằng phương pháp PCA")
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thực hiện PCA: {e}")
            # Fallback đơn giản: chọn n đặc trưng đầu tiên
            return X.columns[:n_features].tolist()
    
    def _remove_collinear_features(self, X: pd.DataFrame) -> List[str]:
        """
        Loại bỏ các đặc trưng có tương quan cao với nhau (đa cộng tuyến).
        
        Args:
            X: DataFrame chứa các đặc trưng đã chọn
            
        Returns:
            Danh sách các đặc trưng sau khi loại bỏ đa cộng tuyến
        """
        # Tính ma trận tương quan
        corr_matrix = X.corr().abs()
        
        # Tạo ma trận tam giác trên (không xét các phần tử trên đường chéo)
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Tìm các cặp đặc trưng có tương quan cao hơn ngưỡng
        collinear_pairs = []
        for col in upper_triangle.columns:
            # Lấy các cột có tương quan với col lớn hơn ngưỡng
            collinear_cols = upper_triangle.index[upper_triangle[col] > self.collinearity_threshold].tolist()
            
            for collinear_col in collinear_cols:
                collinear_pairs.append((col, collinear_col, upper_triangle.loc[collinear_col, col]))
        
        # Lưu trữ thông tin về các đặc trưng đa cộng tuyến
        self.collinear_features = {
            'pairs': collinear_pairs,
            'threshold': self.collinearity_threshold
        }
        
        # Nếu không có đặc trưng đa cộng tuyến, trả về danh sách ban đầu
        if not collinear_pairs:
            return X.columns.tolist()
        
        # Sắp xếp các cặp theo độ tương quan giảm dần
        collinear_pairs = sorted(collinear_pairs, key=lambda x: x[2], reverse=True)
        
        # Tính tổng tương quan của mỗi đặc trưng với các đặc trưng khác
        feature_correlation_sum = {}
        for f1, f2, corr in collinear_pairs:
            if f1 not in feature_correlation_sum:
                feature_correlation_sum[f1] = 0
            if f2 not in feature_correlation_sum:
                feature_correlation_sum[f2] = 0
            
            feature_correlation_sum[f1] += corr
            feature_correlation_sum[f2] += corr
        
        # Ưu tiên loại bỏ đặc trưng có tổng tương quan cao hơn
        features_to_drop = set()
        
        # Lặp qua từng cặp và loại bỏ một đặc trưng
        for f1, f2, _ in collinear_pairs:
            # Nếu cả hai đặc trưng đều chưa bị loại
            if f1 not in features_to_drop and f2 not in features_to_drop:
                # Loại bỏ đặc trưng có tổng tương quan cao hơn
                if feature_correlation_sum.get(f1, 0) >= feature_correlation_sum.get(f2, 0):
                    features_to_drop.add(f1)
                else:
                    features_to_drop.add(f2)
        
        # Lọc danh sách đặc trưng cuối cùng
        selected_features = [f for f in X.columns if f not in features_to_drop]
        
        self.logger.info(f"Đã loại bỏ {len(features_to_drop)} đặc trưng đa cộng tuyến: {features_to_drop}")
        return selected_features
    
    def _save_results(self) -> None:
        """
        Lưu kết quả lựa chọn đặc trưng vào thư mục đã chỉ định.
        """
        if not self.save_dir:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Lưu danh sách đặc trưng đã chọn
        with open(self.save_dir / f"selected_features_{timestamp}.txt", 'w') as f:
            for feature in self.selected_features:
                f.write(f"{feature}\n")
        
        # Lưu feature importance
        if self.feature_importance_df is not None:
            self.feature_importance_df.to_csv(self.save_dir / f"feature_importance_{timestamp}.csv", index=False)
        
        # Lưu thông tin đặc trưng đa cộng tuyến nếu có
        if self.collinear_features:
            with open(self.save_dir / f"collinear_features_{timestamp}.txt", 'w') as f:
                f.write(f"Threshold: {self.collinear_features['threshold']}\n\n")
                f.write("Collinear feature pairs:\n")
                for pair in self.collinear_features['pairs']:
                    f.write(f"{pair[0]} - {pair[1]}: {pair[2]:.4f}\n")
        
        # Lưu thông tin đầy đủ
        info = {
            'timestamp': timestamp,
            'method': self.method,
            'n_features': len(self.selected_features),
            'selected_features': self.selected_features,
            'threshold': self.threshold,
            'feature_importance': self.feature_importance,
            'handle_collinearity': self.handle_collinearity,
            'collinearity_threshold': self.collinearity_threshold
        }
        
        with open(self.save_dir / f"feature_selection_info_{timestamp}.pkl", 'wb') as f:
            pickle.dump(info, f)
        
        self.logger.info(f"Đã lưu kết quả lựa chọn đặc trưng vào {self.save_dir}")
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> None:
        """
        Vẽ biểu đồ tầm quan trọng của đặc trưng.
        
        Args:
            top_n: Số lượng đặc trưng quan trọng nhất để hiển thị
            figsize: Kích thước của biểu đồ
            save_path: Đường dẫn để lưu biểu đồ, None nếu không lưu
        """
        if self.feature_importance_df is None or self.feature_importance_df.empty:
            self.logger.warning("Không có thông tin về tầm quan trọng của đặc trưng để vẽ biểu đồ")
            return
        
        try:
            # Lấy top_n đặc trưng quan trọng nhất
            top_features = self.feature_importance_df.sort_values('importance', ascending=False).head(top_n)
            
            # Tạo biểu đồ
            plt.figure(figsize=figsize)
            ax = sns.barplot(x='importance', y='feature', data=top_features)
            
            # Đánh dấu các đặc trưng đã chọn
            selected_mask = top_features['feature'].isin(self.selected_features)
            bars = ax.patches
            
            for i, (mask, bar) in enumerate(zip(selected_mask, bars)):
                if mask:
                    bar.set_color('green')
                else:
                    bar.set_color('gray')
            
            # Thêm nhãn và tiêu đề
            plt.title(f"Top {top_n} Feature Importance (Method: {self.method})")
            plt.xlabel('Importance Score')
            plt.ylabel('Feature Name')
            plt.tight_layout()
            
            # Thêm chú thích
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='Selected'),
                Patch(facecolor='gray', label='Not Selected')
            ]
            plt.legend(handles=legend_elements, loc='lower right')
            
            # Lưu biểu đồ nếu có đường dẫn
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Đã lưu biểu đồ tầm quan trọng của đặc trưng vào {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ biểu đồ: {e}")
    
    def plot_correlation_matrix(
        self,
        X: pd.DataFrame,
        features: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 12),
        save_path: Optional[str] = None
    ) -> None:
        """
        Vẽ ma trận tương quan của các đặc trưng.
        
        Args:
            X: DataFrame chứa các đặc trưng
            features: Danh sách các đặc trưng cần vẽ, None để sử dụng đặc trưng đã chọn
            figsize: Kích thước của biểu đồ
            save_path: Đường dẫn để lưu biểu đồ, None nếu không lưu
        """
        if features is None:
            features = self.selected_features
        
        if not features:
            self.logger.warning("Không có đặc trưng nào để vẽ ma trận tương quan")
            return
        
        try:
            # Lọc các đặc trưng tồn tại trong X
            valid_features = [f for f in features if f in X.columns]
            
            if not valid_features:
                self.logger.warning("Không có đặc trưng nào tồn tại trong dữ liệu để vẽ ma trận tương quan")
                return
            
            # Tính ma trận tương quan
            corr_matrix = X[valid_features].corr()
            
            # Vẽ heatmap
            plt.figure(figsize=figsize)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            
            sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap=cmap,
                vmax=1.0,
                vmin=-1.0,
                center=0,
                square=True,
                linewidths=.5,
                annot=True,
                fmt=".2f",
                cbar_kws={"shrink": .5}
            )
            
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            
            # Lưu biểu đồ nếu có đường dẫn
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Đã lưu ma trận tương quan vào {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ ma trận tương quan: {e}")
    
    def evaluate_feature_set(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: Optional[List[str]] = None,
        model_type: str = 'random_forest',
        cv: int = 5,
        problem_type: str = 'regression',
        eval_metric: str = 'rmse'
    ) -> Dict[str, float]:
        """
        Đánh giá hiệu suất của tập đặc trưng.
        
        Args:
            X: DataFrame chứa các đặc trưng
            y: Series chứa biến mục tiêu
            features: Danh sách các đặc trưng cần đánh giá, None để sử dụng đặc trưng đã chọn
            model_type: Loại mô hình để đánh giá ('random_forest', 'xgboost', 'linear')
            cv: Số fold cross-validation
            problem_type: Loại bài toán ('regression' hoặc 'classification')
            eval_metric: Metric đánh giá ('rmse', 'mae', 'r2' cho regression; 'accuracy', 'f1', 'auc' cho classification)
            
        Returns:
            Dict chứa kết quả đánh giá
        """
        if features is None:
            features = self.selected_features
        
        if not features:
            self.logger.warning("Không có đặc trưng nào để đánh giá")
            return {'error': 'No features to evaluate'}
        
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            
            # Lọc các đặc trưng tồn tại trong X
            valid_features = [f for f in features if f in X.columns]
            
            if not valid_features:
                self.logger.warning("Không có đặc trưng nào tồn tại trong dữ liệu để đánh giá")
                return {'error': 'No valid features in data'}
            
            # Chọn model
            if model_type == 'random_forest':
                if problem_type == 'regression':
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                else:
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            
            elif model_type == 'xgboost':
                try:
                    import xgboost as xgb
                    if problem_type == 'regression':
                        model = xgb.XGBRegressor(n_estimators=100, random_state=self.random_state)
                    else:
                        model = xgb.XGBClassifier(n_estimators=100, random_state=self.random_state)
                except ImportError:
                    self.logger.warning("XGBoost không được cài đặt, sử dụng Random Forest thay thế")
                    if problem_type == 'regression':
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                    else:
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            
            elif model_type == 'linear':
                if problem_type == 'regression':
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                else:
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            
            else:
                self.logger.warning(f"Không hỗ trợ model_type: {model_type}, sử dụng Random Forest thay thế")
                if problem_type == 'regression':
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
                else:
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            
            # Chọn metric
            if problem_type == 'regression':
                if eval_metric == 'rmse':
                    scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
                    scoring = 'neg_root_mean_squared_error'
                elif eval_metric == 'mae':
                    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
                    scoring = 'neg_mean_absolute_error'
                elif eval_metric == 'r2':
                    scorer = make_scorer(r2_score)
                    scoring = 'r2'
                else:
                    self.logger.warning(f"Không hỗ trợ eval_metric: {eval_metric} cho regression, sử dụng rmse thay thế")
                    scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
                    scoring = 'neg_root_mean_squared_error'
            else:  # classification
                if eval_metric == 'accuracy':
                    scorer = make_scorer(accuracy_score)
                    scoring = 'accuracy'
                elif eval_metric == 'f1':
                    if len(np.unique(y)) > 2:  # multiclass
                        scorer = make_scorer(lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'))
                        scoring = 'f1_weighted'
                    else:
                        scorer = make_scorer(f1_score)
                        scoring = 'f1'
                elif eval_metric == 'auc':
                    if len(np.unique(y)) > 2:  # multiclass
                        self.logger.warning("AUC không hỗ trợ tốt cho multiclass, chuyển sang accuracy")
                        scorer = make_scorer(accuracy_score)
                        scoring = 'accuracy'
                    else:
                        scorer = make_scorer(roc_auc_score)
                        scoring = 'roc_auc'
                else:
                    self.logger.warning(f"Không hỗ trợ eval_metric: {eval_metric} cho classification, sử dụng accuracy thay thế")
                    scorer = make_scorer(accuracy_score)
                    scoring = 'accuracy'
            
            # Thực hiện cross-validation
            cv_scores = cross_val_score(model, X[valid_features], y, cv=cv, scoring=scoring)
            
            # Chuyển đổi kết quả
            if problem_type == 'regression' and (eval_metric == 'rmse' or eval_metric == 'mae'):
                # neg_mean_squared_error và neg_mean_absolute_error trả về giá trị âm
                cv_scores = -cv_scores
            
            # Tính toán và trả về kết quả
            result = {
                'mean_score': float(np.mean(cv_scores)),
                'std_score': float(np.std(cv_scores)),
                'cv_scores': cv_scores.tolist(),
                'num_features': len(valid_features),
                'model_type': model_type,
                'problem_type': problem_type,
                'eval_metric': eval_metric
            }
            
            self.logger.info(f"Kết quả đánh giá: {eval_metric}={result['mean_score']:.4f} ± {result['std_score']:.4f} với {len(valid_features)} đặc trưng")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi đánh giá tập đặc trưng: {e}")
            return {'error': str(e)}
    
    def compare_feature_sets(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_sets: Dict[str, List[str]],
        model_type: str = 'random_forest',
        cv: int = 5,
        problem_type: str = 'regression',
        eval_metric: str = 'rmse',
        plot_result: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        So sánh hiệu suất của nhiều tập đặc trưng khác nhau.
        
        Args:
            X: DataFrame chứa các đặc trưng
            y: Series chứa biến mục tiêu
            feature_sets: Dictionary ánh xạ tên tập đặc trưng với danh sách các đặc trưng
            model_type: Loại mô hình để đánh giá
            cv: Số fold cross-validation
            problem_type: Loại bài toán ('regression' hoặc 'classification')
            eval_metric: Metric đánh giá
            plot_result: Vẽ biểu đồ so sánh kết quả
            figsize: Kích thước của biểu đồ
            save_path: Đường dẫn để lưu biểu đồ, None nếu không lưu
            
        Returns:
            DataFrame chứa kết quả so sánh
        """
        if not feature_sets:
            self.logger.warning("Không có tập đặc trưng nào để so sánh")
            return pd.DataFrame()
        
        # Đánh giá từng tập đặc trưng
        results = []
        
        for name, features in feature_sets.items():
            result = self.evaluate_feature_set(
                X, y, features, model_type, cv, problem_type, eval_metric
            )
            
            if 'error' in result:
                self.logger.warning(f"Lỗi khi đánh giá tập đặc trưng {name}: {result['error']}")
                continue
                
            result['feature_set'] = name
            results.append(result)
        
        # Tạo DataFrame kết quả
        if not results:
            self.logger.warning("Không có kết quả đánh giá nào")
            return pd.DataFrame()
            
        result_df = pd.DataFrame(results)
        
        # Sắp xếp kết quả theo metric đánh giá
        if problem_type == 'regression' and eval_metric in ['rmse', 'mae']:
            # Đối với RMSE và MAE, giá trị thấp hơn tốt hơn
            result_df = result_df.sort_values('mean_score', ascending=True)
        else:
            # Đối với R2, accuracy, F1, AUC, giá trị cao hơn tốt hơn
            result_df = result_df.sort_values('mean_score', ascending=False)
        
        # Vẽ biểu đồ so sánh nếu cần
        if plot_result and len(result_df) > 0:
            try:
                plt.figure(figsize=figsize)
                
                # Tạo thanh lỗi
                plt.errorbar(
                    result_df['feature_set'],
                    result_df['mean_score'],
                    yerr=result_df['std_score'],
                    fmt='o',
                    capsize=5,
                    elinewidth=2,
                    markeredgewidth=2
                )
                
                # Thêm nhãn và tiêu đề
                plt.title(f"Feature Set Comparison ({model_type}, {eval_metric})")
                plt.xlabel('Feature Set')
                plt.ylabel(eval_metric.upper())
                
                # Chỉnh sửa format nhãn trục y nếu cần
                if problem_type == 'regression' and eval_metric in ['rmse', 'mae']:
                    plt.gca().set_ylim(bottom=0)  # RMSE và MAE không âm
                
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Thêm giá trị cụ thể lên đồ thị
                for i, row in result_df.iterrows():
                    plt.text(
                        i,
                        row['mean_score'],
                        f"{row['mean_score']:.4f}",
                        ha='center',
                        va='bottom' if problem_type == 'regression' and eval_metric in ['rmse', 'mae'] else 'top',
                        fontweight='bold'
                    )
                
                # Lưu biểu đồ nếu có đường dẫn
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    self.logger.info(f"Đã lưu biểu đồ so sánh tập đặc trưng vào {save_path}")
                
                plt.show()
                
            except Exception as e:
                self.logger.error(f"Lỗi khi vẽ biểu đồ so sánh: {e}")
        
        return result_df
    
    def feature_stability_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_iterations: int = 10,
        sample_fraction: float = 0.8,
        n_features: Optional[int] = None,
        problem_type: str = 'regression',
        plot_result: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Phân tích độ ổn định của việc lựa chọn đặc trưng bằng bootstrap.
        
        Args:
            X: DataFrame chứa các đặc trưng
            y: Series chứa biến mục tiêu
            n_iterations: Số lần lặp lại bootstrap
            sample_fraction: Tỷ lệ mẫu cho mỗi lần bootstrap
            n_features: Số lượng đặc trưng cần chọn mỗi lần
            problem_type: Loại bài toán ('regression' hoặc 'classification')
            plot_result: Vẽ biểu đồ kết quả
            figsize: Kích thước của biểu đồ
            save_path: Đường dẫn để lưu biểu đồ, None nếu không lưu
            
        Returns:
            DataFrame chứa kết quả phân tích độ ổn định
        """
        if X.empty or y.empty:
            self.logger.warning("DataFrame rỗng, không thể phân tích độ ổn định")
            return pd.DataFrame()
        
        # Lưu phương pháp và số lượng đặc trưng hiện tại
        original_method = self.method
        original_n_features = self.n_features
        
        # Cập nhật số lượng đặc trưng nếu được chỉ định
        if n_features is not None:
            self.n_features = n_features
        
        # Kết quả lựa chọn đặc trưng cho mỗi lần lặp
        all_selected_features = []
        
        self.logger.info(f"Bắt đầu phân tích độ ổn định với {n_iterations} lần lặp (bootstrap {sample_fraction*100:.0f}%)")
        
        for i in range(n_iterations):
            # Tạo bootstrap sample
            n_samples = int(len(X) * sample_fraction)
            indices = np.random.choice(len(X), size=n_samples, replace=True)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
            
            # Lựa chọn đặc trưng trên mẫu bootstrap
            selected = self.select_features(X_sample, y_sample, problem_type=problem_type)
            all_selected_features.append(selected)
            
            self.logger.debug(f"Bootstrap {i+1}/{n_iterations}: Đã chọn {len(selected)} đặc trưng")
        
        # Đếm số lần xuất hiện của mỗi đặc trưng
        all_features = set()
        for features in all_selected_features:
            all_features.update(features)
        
        feature_counts = {feature: 0 for feature in all_features}
        
        for features in all_selected_features:
            for feature in features:
                feature_counts[feature] += 1
        
        # Tính tỷ lệ xuất hiện
        feature_frequency = {feature: count / n_iterations for feature, count in feature_counts.items()}
        
        # Tạo DataFrame kết quả
        stability_df = pd.DataFrame({
            'feature': list(feature_frequency.keys()),
            'frequency': list(feature_frequency.values()),
            'count': [feature_counts[f] for f in feature_frequency.keys()]
        }).sort_values('frequency', ascending=False)
        
        # Khôi phục phương pháp và số lượng đặc trưng
        self.method = original_method
        self.n_features = original_n_features
        
        # Vẽ biểu đồ nếu cần
        if plot_result and len(stability_df) > 0:
            try:
                plt.figure(figsize=figsize)
                
                # Giới hạn số đặc trưng hiển thị
                top_features = min(30, len(stability_df))
                plot_df = stability_df.head(top_features)
                
                # Vẽ biểu đồ cột
                ax = sns.barplot(x='frequency', y='feature', data=plot_df)
                
                # Đánh dấu các đặc trưng trong tập đã chọn cuối cùng
                if self.selected_features:
                    bars = ax.patches
                    for i, feature in enumerate(plot_df['feature']):
                        if i < len(bars) and feature in self.selected_features:
                            bars[i].set_color('green')
                
                # Thêm nhãn và tiêu đề
                plt.title(f"Feature Selection Stability Analysis ({self.method})")
                plt.xlabel('Selection Frequency')
                plt.ylabel('Feature Name')
                plt.xlim(0, 1.05)
                plt.grid(True, linestyle='--', alpha=0.7, axis='x')
                plt.tight_layout()
                
                # Thêm giá trị cụ thể lên đồ thị
                for i, row in plot_df.iterrows():
                    if i < top_features:
                        plt.text(
                            row['frequency'] + 0.02,
                            i,
                            f"{row['frequency']:.2f} ({row['count']}/{n_iterations})",
                            va='center',
                            fontsize=8
                        )
                
                # Thêm chú thích
                if self.selected_features:
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='green', label='In Final Selection'),
                        Patch(facecolor='#1f77b4', label='Not in Final Selection')
                    ]
                    plt.legend(handles=legend_elements, loc='lower right')
                
                # Lưu biểu đồ nếu có đường dẫn
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    self.logger.info(f"Đã lưu biểu đồ phân tích độ ổn định vào {save_path}")
                
                plt.show()
                
            except Exception as e:
                self.logger.error(f"Lỗi khi vẽ biểu đồ phân tích độ ổn định: {e}")
        
        self.logger.info(f"Đã hoàn thành phân tích độ ổn định cho {len(stability_df)} đặc trưng")
        return stability_df
    
    def get_feature_groups(self, X: pd.DataFrame, pattern_dict: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
        """
        Nhóm các đặc trưng dựa trên pattern trong tên.
        
        Args:
            X: DataFrame chứa các đặc trưng
            pattern_dict: Dictionary ánh xạ tên nhóm với list các pattern
            
        Returns:
            Dictionary ánh xạ tên nhóm với danh sách các đặc trưng
        """
        # Pattern mặc định nếu không được chỉ định
        if pattern_dict is None:
            pattern_dict = {
                'price': ['price', 'open', 'high', 'low', 'close', 'volume', 'ohlc', 'vwap'],
                'technical': ['rsi', 'macd', 'ma', 'ema', 'sma', 'momentum', 'volatility', 'atr', 'bollinger', 'adx'],
                'sentiment': ['sentiment', 'emotion', 'news', 'social', 'positive', 'negative', 'neutral'],
                'onchain': ['onchain', 'blockchain', 'transaction', 'address', 'wallet', 'hash', 'mempool'],
                'time': ['hour', 'day', 'week', 'month', 'year', 'time', 'date', 'weekday']
            }
        
        # Kết quả
        feature_groups = {group: [] for group in pattern_dict.keys()}
        feature_groups['other'] = []  # Nhóm default cho các đặc trưng không thuộc nhóm nào
        
        # Phân loại các đặc trưng vào các nhóm
        for feature in X.columns:
            feature_lower = feature.lower()
            assigned = False
            
            for group, patterns in pattern_dict.items():
                if any(pattern.lower() in feature_lower for pattern in patterns):
                    feature_groups[group].append(feature)
                    assigned = True
                    break
            
            if not assigned:
                feature_groups['other'].append(feature)
        
        # Loại bỏ các nhóm rỗng
        feature_groups = {group: features for group, features in feature_groups.items() if features}
        
        # Log kết quả
        for group, features in feature_groups.items():
            self.logger.info(f"Nhóm {group}: {len(features)} đặc trưng")
        
        return feature_groups


# Factory để tạo các FeatureSelector với cấu hình khác nhau
class FeatureSelectorFactory:
    """
    Factory để tạo các FeatureSelector với cấu hình khác nhau.
    """
    
    @staticmethod
    def create_correlation_selector(
        threshold: float = 0.05,
        correlation_method: str = 'pearson',
        handle_collinearity: bool = True
    ) -> FeatureSelector:
        """
        Tạo bộ chọn đặc trưng dựa trên tương quan.
        
        Args:
            threshold: Ngưỡng tương quan tối thiểu
            correlation_method: Phương pháp tính tương quan
            handle_collinearity: Xử lý đa cộng tuyến
            
        Returns:
            FeatureSelector đã cấu hình
        """
        return FeatureSelector(
            method='correlation',
            threshold=threshold,
            correlation_method=correlation_method,
            handle_collinearity=handle_collinearity
        )
    
    @staticmethod
    def create_model_based_selector(
        model_type: str = 'random_forest',  # 'random_forest' or 'xgboost'
        n_features: int = 20,
        threshold: float = 0.01,
        handle_collinearity: bool = True
    ) -> FeatureSelector:
        """
        Tạo bộ chọn đặc trưng dựa trên importance từ mô hình.
        
        Args:
            model_type: Loại mô hình ('random_forest', 'xgboost')
            n_features: Số lượng đặc trưng cần chọn
            threshold: Ngưỡng importance tối thiểu
            handle_collinearity: Xử lý đa cộng tuyến
            
        Returns:
            FeatureSelector đã cấu hình
        """
        return FeatureSelector(
            method=model_type,
            n_features=n_features,
            threshold=threshold,
            handle_collinearity=handle_collinearity
        )
    
    @staticmethod
    def create_advanced_selector(
        method: str = 'recursive',  # 'recursive' or 'mutual_info'
        n_features: int = 15,
        threshold: float = 0.01,
        handle_collinearity: bool = True,
        save_dir: Optional[str] = './feature_selection_results'
    ) -> FeatureSelector:
        """
        Tạo bộ chọn đặc trưng nâng cao.
        
        Args:
            method: Phương pháp lựa chọn ('recursive', 'mutual_info')
            n_features: Số lượng đặc trưng cần chọn
            threshold: Ngưỡng importance tối thiểu
            handle_collinearity: Xử lý đa cộng tuyến
            save_dir: Thư mục lưu kết quả
            
        Returns:
            FeatureSelector đã cấu hình
        """
        return FeatureSelector(
            method=method,
            n_features=n_features,
            threshold=threshold,
            handle_collinearity=handle_collinearity,
            save_dir=save_dir
        )