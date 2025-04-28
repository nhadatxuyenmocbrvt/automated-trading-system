"""
Phương pháp bọc (wrapper methods) cho lựa chọn đặc trưng.
Module này cung cấp các phương pháp bọc như Forward Selection, Backward Elimination,
Recursive Feature Elimination và Sequential Feature Selector để lựa chọn đặc trưng
dựa trên hiệu suất của mô hình.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import logging
import warnings
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import time

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import get_logger

# Thiết lập logger
logger = get_logger("feature_selector")

def forward_selection(
    df: pd.DataFrame,
    target_column: str,
    max_features: Optional[int] = None,
    min_features: int = 1,
    scoring: str = 'r2',
    problem_type: str = 'auto',
    cv: int = 5,
    model: Optional[Any] = None,
    exclude_columns: List[str] = [],
    verbose: bool = False
) -> List[str]:
    """
    Lựa chọn đặc trưng bằng phương pháp Forward Selection.
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu
        max_features: Số lượng đặc trưng tối đa cần chọn
        min_features: Số lượng đặc trưng tối thiểu cần chọn
        scoring: Phương pháp đánh giá ('r2', 'accuracy', 'f1', etc.)
        problem_type: Loại bài toán ('classification', 'regression', 'auto')
        cv: Số lượng fold cho cross-validation
        model: Mô hình sử dụng cho đánh giá (mặc định là LinearRegression hoặc LogisticRegression)
        exclude_columns: Danh sách cột cần loại trừ khỏi quá trình lựa chọn
        verbose: Hiển thị thông tin chi tiết
        
    Returns:
        Danh sách tên các cột đặc trưng được chọn
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        if target_column not in df.columns:
            logger.error(f"Cột mục tiêu '{target_column}' không tồn tại trong DataFrame")
            return []
        
        # Loại bỏ các cột không muốn xem xét
        exclude_columns.append(target_column)  # Loại trừ cả cột mục tiêu
        feature_df = df.drop(columns=exclude_columns, errors='ignore')
        
        # Chỉ giữ lại các cột số
        numeric_cols = feature_df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            logger.warning("Không có cột số nào để thực hiện Forward Selection")
            return []
        
        feature_df = feature_df[numeric_cols]
        
        # Kiểm tra kiểu dữ liệu của target và xác định loại bài toán
        if problem_type == 'auto':
            is_categorical = pd.api.types.is_categorical_dtype(df[target_column]) or pd.api.types.is_object_dtype(df[target_column])
            is_discrete_numeric = pd.api.types.is_numeric_dtype(df[target_column]) and len(df[target_column].unique()) <= 20
            problem_type = 'classification' if (is_categorical or is_discrete_numeric) else 'regression'
        
        # Chuẩn bị dữ liệu
        X = feature_df.values
        y = df[target_column].values
        
        # Khởi tạo mô hình nếu chưa được cung cấp
        if model is None:
            if problem_type == 'classification':
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                model = LinearRegression()
        
        # Xác định số lượng đặc trưng tối đa
        if max_features is None:
            max_features = len(numeric_cols)
        else:
            max_features = min(max_features, len(numeric_cols))
        
        # Thực hiện Forward Selection
        features_selected = []  # Danh sách đặc trưng đã chọn
        features_to_select = numeric_cols.copy()  # Danh sách đặc trưng cần xem xét
        
        best_score = -np.inf
        
        # Bắt đầu đo thời gian
        start_time = time.time()
        
        for i in range(max_features):
            best_new_score = -np.inf
            best_feature = None
            
            # Thử từng đặc trưng chưa được chọn
            for feature in features_to_select:
                current_features = features_selected + [feature]
                X_subset = df[current_features].values
                
                # Đánh giá mô hình với đặc trưng hiện tại
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_scores = cross_val_score(
                        clone(model), X_subset, y, 
                        cv=cv, scoring=scoring
                    )
                current_score = np.mean(cv_scores)
                
                # Cập nhật nếu tìm thấy đặc trưng tốt hơn
                if current_score > best_new_score:
                    best_new_score = current_score
                    best_feature = feature
            
            # Thêm đặc trưng tốt nhất
            if best_feature is not None:
                features_selected.append(best_feature)
                features_to_select.remove(best_feature)
                
                if verbose:
                    logger.info(f"Đã thêm đặc trưng {best_feature} (điểm: {best_new_score:.4f})")
                
                # Cập nhật điểm số tốt nhất
                best_score = best_new_score
            
            # Điều kiện dừng nếu không còn đặc trưng hoặc điểm không cải thiện
            if len(features_to_select) == 0:
                break
        
        # Kết thúc đo thời gian
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Đảm bảo có ít nhất min_features đặc trưng
        if len(features_selected) < min_features and len(numeric_cols) >= min_features:
            remaining_features = [f for f in numeric_cols if f not in features_selected]
            features_selected.extend(remaining_features[:min_features - len(features_selected)])
        
        logger.info(f"Forward Selection đã chọn {len(features_selected)}/{len(numeric_cols)} đặc trưng trong {execution_time:.2f} giây")
        
        return features_selected
    
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện Forward Selection: {str(e)}")
        return []

def backward_elimination(
    df: pd.DataFrame,
    target_column: str,
    min_features: int = 1,
    threshold: float = 0.0,
    scoring: str = 'r2',
    problem_type: str = 'auto',
    cv: int = 5,
    model: Optional[Any] = None,
    exclude_columns: List[str] = [],
    verbose: bool = False
) -> List[str]:
    """
    Lựa chọn đặc trưng bằng phương pháp Backward Elimination.
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu
        min_features: Số lượng đặc trưng tối thiểu cần giữ lại
        threshold: Ngưỡng giảm hiệu suất chấp nhận được khi loại bỏ đặc trưng
        scoring: Phương pháp đánh giá ('r2', 'accuracy', 'f1', etc.)
        problem_type: Loại bài toán ('classification', 'regression', 'auto')
        cv: Số lượng fold cho cross-validation
        model: Mô hình sử dụng cho đánh giá (mặc định là LinearRegression hoặc LogisticRegression)
        exclude_columns: Danh sách cột cần loại trừ khỏi quá trình lựa chọn
        verbose: Hiển thị thông tin chi tiết
        
    Returns:
        Danh sách tên các cột đặc trưng được chọn
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        if target_column not in df.columns:
            logger.error(f"Cột mục tiêu '{target_column}' không tồn tại trong DataFrame")
            return []
        
        # Loại bỏ các cột không muốn xem xét
        exclude_columns.append(target_column)  # Loại trừ cả cột mục tiêu
        feature_df = df.drop(columns=exclude_columns, errors='ignore')
        
        # Chỉ giữ lại các cột số
        numeric_cols = feature_df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            logger.warning("Không có cột số nào để thực hiện Backward Elimination")
            return []
        
        feature_df = feature_df[numeric_cols]
        
        # Kiểm tra kiểu dữ liệu của target và xác định loại bài toán
        if problem_type == 'auto':
            is_categorical = pd.api.types.is_categorical_dtype(df[target_column]) or pd.api.types.is_object_dtype(df[target_column])
            is_discrete_numeric = pd.api.types.is_numeric_dtype(df[target_column]) and len(df[target_column].unique()) <= 20
            problem_type = 'classification' if (is_categorical or is_discrete_numeric) else 'regression'
        
        # Chuẩn bị dữ liệu
        X = feature_df.values
        y = df[target_column].values
        
        # Khởi tạo mô hình nếu chưa được cung cấp
        if model is None:
            if problem_type == 'classification':
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                model = LinearRegression()
        
        # Bắt đầu với tất cả các đặc trưng
        features_selected = numeric_cols.copy()
        
        # Đánh giá mô hình với tất cả các đặc trưng
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_scores = cross_val_score(
                clone(model), feature_df.values, y, 
                cv=cv, scoring=scoring
            )
        best_score = np.mean(cv_scores)
        
        if verbose:
            logger.info(f"Điểm ban đầu với {len(features_selected)} đặc trưng: {best_score:.4f}")
        
        # Bắt đầu đo thời gian
        start_time = time.time()
        
        # Lặp cho đến khi đạt số lượng đặc trưng tối thiểu
        while len(features_selected) > min_features:
            worst_score = best_score
            worst_feature = None
            
            # Thử loại bỏ từng đặc trưng
            for feature in features_selected:
                features_subset = [f for f in features_selected if f != feature]
                X_subset = df[features_subset].values
                
                # Đánh giá mô hình khi loại bỏ đặc trưng
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_scores = cross_val_score(
                        clone(model), X_subset, y, 
                        cv=cv, scoring=scoring
                    )
                current_score = np.mean(cv_scores)
                
                # Tìm đặc trưng có tác động nhỏ nhất (hoặc tiêu cực) khi loại bỏ
                if current_score > worst_score or (best_score - current_score) <= threshold:
                    worst_score = current_score
                    worst_feature = feature
            
            # Loại bỏ đặc trưng nếu tìm thấy
            if worst_feature is not None:
                features_selected.remove(worst_feature)
                
                if verbose:
                    logger.info(f"Đã loại bỏ đặc trưng {worst_feature} (điểm mới: {worst_score:.4f})")
                
                # Cập nhật điểm số tốt nhất
                best_score = worst_score
            else:
                # Không thể loại bỏ thêm đặc trưng mà không giảm hiệu suất quá ngưỡng
                break
        
        # Kết thúc đo thời gian
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"Backward Elimination đã chọn {len(features_selected)}/{len(numeric_cols)} đặc trưng trong {execution_time:.2f} giây")
        
        return features_selected
    
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện Backward Elimination: {str(e)}")
        return []

def recursive_feature_elimination(
    df: pd.DataFrame,
    target_column: str,
    n_features_to_select: Optional[int] = None,
    step: float = 0.1,
    problem_type: str = 'auto',
    cv: int = 5,
    scoring: Optional[str] = None,
    model: Optional[Any] = None,
    exclude_columns: List[str] = [],
    verbose: bool = False
) -> List[str]:
    """
    Lựa chọn đặc trưng bằng phương pháp Recursive Feature Elimination (RFE).
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu
        n_features_to_select: Số lượng đặc trưng cần chọn (mặc định là sqrt(n_features))
        step: Số đặc trưng (nếu > 1) hoặc tỷ lệ đặc trưng (nếu <= 1) để loại bỏ mỗi bước
        problem_type: Loại bài toán ('classification', 'regression', 'auto')
        cv: Số lượng fold cho cross-validation với RFECV (chỉ khi scoring được cung cấp)
        scoring: Phương pháp đánh giá cho RFECV (None để sử dụng RFE thông thường)
        model: Mô hình sử dụng cho đánh giá (mặc định là LinearRegression hoặc LogisticRegression)
        exclude_columns: Danh sách cột cần loại trừ khỏi quá trình lựa chọn
        verbose: Hiển thị thông tin chi tiết
        
    Returns:
        Danh sách tên các cột đặc trưng được chọn
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        if target_column not in df.columns:
            logger.error(f"Cột mục tiêu '{target_column}' không tồn tại trong DataFrame")
            return []
        
        # Loại bỏ các cột không muốn xem xét
        exclude_columns.append(target_column)  # Loại trừ cả cột mục tiêu
        feature_df = df.drop(columns=exclude_columns, errors='ignore')
        
        # Chỉ giữ lại các cột số
        numeric_cols = feature_df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            logger.warning("Không có cột số nào để thực hiện RFE")
            return []
        
        feature_df = feature_df[numeric_cols]
        
        # Kiểm tra kiểu dữ liệu của target và xác định loại bài toán
        if problem_type == 'auto':
            is_categorical = pd.api.types.is_categorical_dtype(df[target_column]) or pd.api.types.is_object_dtype(df[target_column])
            is_discrete_numeric = pd.api.types.is_numeric_dtype(df[target_column]) and len(df[target_column].unique()) <= 20
            problem_type = 'classification' if (is_categorical or is_discrete_numeric) else 'regression'
        
        # Chuẩn bị dữ liệu
        X = feature_df.values
        y = df[target_column].values
        
        # Khởi tạo mô hình nếu chưa được cung cấp
        if model is None:
            if problem_type == 'classification':
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                model = LinearRegression()
        
        # Xác định số lượng đặc trưng cần chọn
        if n_features_to_select is None:
            n_features_to_select = int(np.sqrt(len(numeric_cols)))
        
        # Giới hạn số lượng đặc trưng
        n_features_to_select = max(1, min(n_features_to_select, len(numeric_cols)))
        
        # Bắt đầu đo thời gian
        start_time = time.time()
        
        # Sử dụng RFECV (RFE với Cross-Validation) nếu có scoring
        if scoring is not None:
            from sklearn.feature_selection import RFECV
            
            rfe = RFECV(
                estimator=model,
                step=step,
                cv=cv,
                scoring=scoring,
                min_features_to_select=n_features_to_select,
                verbose=1 if verbose else 0
            )
        else:
            # Sử dụng RFE thông thường
            rfe = RFE(
                estimator=model,
                n_features_to_select=n_features_to_select,
                step=step,
                verbose=1 if verbose else 0
            )
        
        # Thực hiện RFE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rfe.fit(X, y)
        
        # Lấy kết quả
        selected_mask = rfe.support_
        selected_features = [numeric_cols[i] for i, selected in enumerate(selected_mask) if selected]
        
        # Kết thúc đo thời gian
        end_time = time.time()
        execution_time = end_time - start_time
        
        if scoring is not None and hasattr(rfe, 'grid_scores_'):
            best_score = np.max(rfe.grid_scores_)
            logger.info(f"RFECV đã chọn {len(selected_features)}/{len(numeric_cols)} đặc trưng (điểm tốt nhất: {best_score:.4f})")
        else:
            logger.info(f"RFE đã chọn {len(selected_features)}/{len(numeric_cols)} đặc trưng")
        
        logger.info(f"Thời gian thực hiện: {execution_time:.2f} giây")
        
        return selected_features
    
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện RFE: {str(e)}")
        return []

def sequential_feature_selector(
    df: pd.DataFrame,
    target_column: str,
    n_features_to_select: Optional[int] = None,
    direction: str = 'forward',  # 'forward' hoặc 'backward'
    scoring: str = 'r2',
    problem_type: str = 'auto',
    cv: int = 5,
    model: Optional[Any] = None,
    exclude_columns: List[str] = [],
    verbose: bool = False
) -> List[str]:
    """
    Lựa chọn đặc trưng bằng phương pháp Sequential Feature Selector.
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu
        n_features_to_select: Số lượng đặc trưng cần chọn (mặc định là sqrt(n_features))
        direction: Hướng lựa chọn ('forward' hoặc 'backward')
        scoring: Phương pháp đánh giá ('r2', 'accuracy', 'f1', etc.)
        problem_type: Loại bài toán ('classification', 'regression', 'auto')
        cv: Số lượng fold cho cross-validation
        model: Mô hình sử dụng cho đánh giá (mặc định là LinearRegression hoặc LogisticRegression)
        exclude_columns: Danh sách cột cần loại trừ khỏi quá trình lựa chọn
        verbose: Hiển thị thông tin chi tiết
        
    Returns:
        Danh sách tên các cột đặc trưng được chọn
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        if target_column not in df.columns:
            logger.error(f"Cột mục tiêu '{target_column}' không tồn tại trong DataFrame")
            return []
        
        # Loại bỏ các cột không muốn xem xét
        exclude_columns.append(target_column)  # Loại trừ cả cột mục tiêu
        feature_df = df.drop(columns=exclude_columns, errors='ignore')
        
        # Chỉ giữ lại các cột số
        numeric_cols = feature_df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            logger.warning("Không có cột số nào để thực hiện Sequential Feature Selection")
            return []
        
        feature_df = feature_df[numeric_cols]
        
        # Kiểm tra kiểu dữ liệu của target và xác định loại bài toán
        if problem_type == 'auto':
            is_categorical = pd.api.types.is_categorical_dtype(df[target_column]) or pd.api.types.is_object_dtype(df[target_column])
            is_discrete_numeric = pd.api.types.is_numeric_dtype(df[target_column]) and len(df[target_column].unique()) <= 20
            problem_type = 'classification' if (is_categorical or is_discrete_numeric) else 'regression'
        
        # Chuẩn bị dữ liệu
        X = feature_df.values
        y = df[target_column].values
        
        # Khởi tạo mô hình nếu chưa được cung cấp
        if model is None:
            if problem_type == 'classification':
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                model = LinearRegression()
        
        # Xác định số lượng đặc trưng cần chọn
        if n_features_to_select is None:
            n_features_to_select = int(np.sqrt(len(numeric_cols)))
        
        # Giới hạn số lượng đặc trưng
        n_features_to_select = max(1, min(n_features_to_select, len(numeric_cols)))
        
        # Kiểm tra hướng lựa chọn
        if direction not in ['forward', 'backward']:
            logger.warning(f"Hướng '{direction}' không hợp lệ, sử dụng 'forward' thay thế")
            direction = 'forward'
        
        # Bắt đầu đo thời gian
        start_time = time.time()
        
        # Thực hiện Sequential Feature Selector
        sfs = SequentialFeatureSelector(
            estimator=model,
            n_features_to_select=n_features_to_select,
            direction=direction,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sfs.fit(X, y)
        
        # Lấy kết quả
        selected_mask = sfs.get_support()
        selected_features = [numeric_cols[i] for i, selected in enumerate(selected_mask) if selected]
        
        # Kết thúc đo thời gian
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"Sequential Feature Selector ({direction}) đã chọn {len(selected_features)}/{len(numeric_cols)} đặc trưng")
        logger.info(f"Thời gian thực hiện: {execution_time:.2f} giây")
        
        return selected_features
    
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện Sequential Feature Selection: {str(e)}")
        return []