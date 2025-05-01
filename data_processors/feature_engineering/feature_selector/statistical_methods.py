"""
Phương pháp thống kê cho lựa chọn đặc trưng.
Module này cung cấp các phương pháp thống kê để đánh giá
và lựa chọn đặc trưng như tương quan, chi-squared, ANOVA,
và mutual information.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import logging
from scipy import stats
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, mutual_info_regression
import warnings

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import get_logger

# Thiết lập logger
logger = get_logger("feature_selector")

def calculate_feature_correlation(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    method: str = 'pearson',
    threshold: float = 0.0
) -> pd.DataFrame:
    """
    Tính toán ma trận tương quan giữa các đặc trưng hoặc với biến mục tiêu.
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu (None để tính tương quan giữa tất cả các cột)
        method: Phương pháp tính tương quan ('pearson', 'kendall', 'spearman')
        threshold: Ngưỡng tối thiểu của hệ số tương quan để giữ lại
        
    Returns:
        DataFrame chứa ma trận tương quan đã lọc theo ngưỡng
    """
    # Kiểm tra dữ liệu đầu vào
    if df.empty:
        logger.warning("DataFrame rỗng, không thể tính toán tương quan")
        return pd.DataFrame()
    
    # Nếu có target_column, tính tương quan với target
    if target_column is not None and target_column in df.columns:
        # Chỉ tính tương quan giữa các biến đặc trưng và biến mục tiêu
        corr_with_target = pd.DataFrame()
        
        # Lấy cột mục tiêu
        target = df[target_column]
        
        # Kiểm tra kiểu dữ liệu của target
        is_numeric_target = pd.api.types.is_numeric_dtype(target)
        
        # Duyệt qua từng cột đặc trưng
        for col in df.columns:
            if col == target_column:
                continue
            
            # Kiểm tra kiểu dữ liệu của cột
            is_numeric_feature = pd.api.types.is_numeric_dtype(df[col])
            
            # Nếu cả target và feature đều là số, tính tương quan
            if is_numeric_feature and is_numeric_target:
                # Loại bỏ các giá trị NaN
                valid_data = pd.concat([df[col], target], axis=1).dropna()
                
                if len(valid_data) > 1:  # Cần ít nhất 2 điểm dữ liệu để tính tương quan
                    correlation = valid_data[col].corr(valid_data[target_column], method=method)
                    corr_with_target.loc[col, 'correlation'] = correlation
            
        # Sắp xếp theo giá trị tuyệt đối của tương quan giảm dần
        corr_with_target['abs_corr'] = corr_with_target['correlation'].abs()
        corr_with_target = corr_with_target.sort_values('abs_corr', ascending=False)
        
        # Loại bỏ các tương quan thấp hơn ngưỡng
        corr_with_target = corr_with_target[corr_with_target['abs_corr'] >= threshold]
        
        # Xóa cột tạm thời
        if 'abs_corr' in corr_with_target.columns:
            corr_with_target.drop('abs_corr', axis=1, inplace=True)
        
        return corr_with_target
    
    else:
        # Tính ma trận tương quan giữa tất cả các cột
        try:
            numeric_df = df.select_dtypes(include=['number'])
            
            if numeric_df.empty:
                logger.warning("Không có cột số nào để tính toán tương quan")
                return pd.DataFrame()
            
            corr_matrix = numeric_df.corr(method=method)
            
            # Loại bỏ các tương quan thấp hơn ngưỡng (giữ lại đường chéo)
            mask = np.abs(corr_matrix) >= threshold
            np.fill_diagonal(mask.values, True)  # Giữ đường chéo
            
            filtered_corr = corr_matrix.where(mask)
            
            return filtered_corr
        
        except Exception as e:
            logger.error(f"Lỗi khi tính ma trận tương quan: {str(e)}")
            return pd.DataFrame()

def correlation_selector(
    df: pd.DataFrame,
    target_column: str,
    k: Optional[int] = None,
    threshold: float = 0.0,
    method: str = 'pearson',
    exclude_columns: List[str] = [],
    **kwargs    
) -> List[str]:
    """
    Lựa chọn đặc trưng dựa trên tương quan với biến mục tiêu.
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu
        k: Số lượng đặc trưng cần chọn (None để chọn theo ngưỡng)
        threshold: Ngưỡng tối thiểu của tương quan tuyệt đối để chọn đặc trưng
        method: Phương pháp tính tương quan ('pearson', 'kendall', 'spearman')
        exclude_columns: Danh sách cột cần loại trừ khỏi việc lựa chọn
        
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
        
        # Lấy cột mục tiêu
        target = df[target_column]
        
        # Kiểm tra kiểu dữ liệu của target
        is_numeric_target = pd.api.types.is_numeric_dtype(target)
        if not is_numeric_target:
            logger.warning(f"Cột mục tiêu '{target_column}' không phải kiểu số, không thể tính tương quan")
            return []
        
        # Tính tương quan với target
        correlation_data = []
        
        for col in feature_df.columns:
            # Chỉ xem xét các cột số
            if not pd.api.types.is_numeric_dtype(feature_df[col]):
                continue
            
            # Loại bỏ các giá trị NaN
            valid_data = pd.concat([feature_df[col], target], axis=1).dropna()
            
            if len(valid_data) > 1:  # Cần ít nhất 2 điểm dữ liệu để tính tương quan
                correlation = valid_data[col].corr(valid_data[target_column], method=method)
                correlation_data.append((col, correlation, abs(correlation)))
        
        # Sắp xếp theo giá trị tuyệt đối của tương quan giảm dần
        correlation_data.sort(key=lambda x: x[2], reverse=True)
        
        # Chọn đặc trưng
        if k is not None:
            # Chọn top-k đặc trưng
            k = min(k, len(correlation_data))
            selected_features = [item[0] for item in correlation_data[:k]]
        else:
            # Chọn theo ngưỡng
            selected_features = [item[0] for item in correlation_data if item[2] >= threshold]
        
        logger.info(f"Đã chọn {len(selected_features)} đặc trưng dựa trên tương quan {method}")
        return selected_features
    
    except Exception as e:
        logger.error(f"Lỗi khi chọn đặc trưng dựa trên tương quan: {str(e)}")
        return []

def chi_squared_selector(
    df: pd.DataFrame,
    target_column: str,
    k: Optional[int] = None,
    threshold: float = 0.05,
    exclude_columns: List[str] = []
) -> List[str]:
    """
    Lựa chọn đặc trưng dựa trên kiểm định Chi-squared cho biến phân loại.
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu (phải là biến phân loại)
        k: Số lượng đặc trưng cần chọn (None để chọn theo ngưỡng p-value)
        threshold: Ngưỡng p-value để chọn đặc trưng (nhỏ hơn ngưỡng sẽ được chọn)
        exclude_columns: Danh sách cột cần loại trừ khỏi việc lựa chọn
        
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
            logger.warning("Không có cột số nào để thực hiện kiểm định Chi-squared")
            return []
        
        X = feature_df[numeric_cols]
        y = df[target_column]
        
        # Đảm bảo tất cả giá trị không âm cho Chi-squared
        X_non_negative = X.copy()
        for col in X.columns:
            min_val = X[col].min()
            if min_val < 0:
                X_non_negative[col] = X[col] - min_val  # Dịch chuyển để tất cả giá trị >= 0
        
        # Thực hiện kiểm định Chi-squared
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chi2_stats, p_values = chi2(X_non_negative, y)
        
        # Tạo DataFrame kết quả
        result_df = pd.DataFrame({
            'feature': numeric_cols,
            'chi2_stat': chi2_stats,
            'p_value': p_values
        })
        
        # Sắp xếp theo thống kê Chi-squared giảm dần
        result_df = result_df.sort_values('chi2_stat', ascending=False)
        
        # Chọn đặc trưng
        if k is not None:
            # Chọn top-k đặc trưng
            k = min(k, len(numeric_cols))
            selected_features = result_df['feature'].tolist()[:k]
        else:
            # Chọn theo ngưỡng p-value
            selected_features = result_df[result_df['p_value'] < threshold]['feature'].tolist()
        
        logger.info(f"Đã chọn {len(selected_features)} đặc trưng dựa trên kiểm định Chi-squared")
        return selected_features
    
    except Exception as e:
        logger.error(f"Lỗi khi chọn đặc trưng dựa trên kiểm định Chi-squared: {str(e)}")
        return []

def anova_selector(
    df: pd.DataFrame,
    target_column: str,
    k: Optional[int] = None,
    threshold: float = 0.05,
    exclude_columns: List[str] = []
) -> List[str]:
    """
    Lựa chọn đặc trưng dựa trên kiểm định ANOVA (F-test).
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu (phải là biến phân loại)
        k: Số lượng đặc trưng cần chọn (None để chọn theo ngưỡng p-value)
        threshold: Ngưỡng p-value để chọn đặc trưng (nhỏ hơn ngưỡng sẽ được chọn)
        exclude_columns: Danh sách cột cần loại trừ khỏi việc lựa chọn
        
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
            logger.warning("Không có cột số nào để thực hiện kiểm định ANOVA")
            return []
        
        X = feature_df[numeric_cols]
        y = df[target_column]
        
        # Thực hiện ANOVA F-test
        f_stats, p_values = f_classif(X, y)
        
        # Tạo DataFrame kết quả
        result_df = pd.DataFrame({
            'feature': numeric_cols,
            'f_stat': f_stats,
            'p_value': p_values
        })
        
        # Sắp xếp theo thống kê F giảm dần
        result_df = result_df.sort_values('f_stat', ascending=False)
        
        # Chọn đặc trưng
        if k is not None:
            # Chọn top-k đặc trưng
            k = min(k, len(numeric_cols))
            selected_features = result_df['feature'].tolist()[:k]
        else:
            # Chọn theo ngưỡng p-value
            selected_features = result_df[result_df['p_value'] < threshold]['feature'].tolist()
        
        logger.info(f"Đã chọn {len(selected_features)} đặc trưng dựa trên kiểm định ANOVA")
        return selected_features
    
    except Exception as e:
        logger.error(f"Lỗi khi chọn đặc trưng dựa trên kiểm định ANOVA: {str(e)}")
        return []

def mutual_info_selector(
    df: pd.DataFrame,
    target_column: str,
    k: Optional[int] = None,
    threshold: float = 0.0,
    discrete_target: bool = None,
    exclude_columns: List[str] = []
) -> List[str]:
    """
    Lựa chọn đặc trưng dựa trên Mutual Information.
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu
        k: Số lượng đặc trưng cần chọn (None để chọn theo ngưỡng)
        threshold: Ngưỡng tối thiểu của Mutual Information để chọn đặc trưng
        discrete_target: Target là biến rời rạc hay liên tục (None để tự động phát hiện)
        exclude_columns: Danh sách cột cần loại trừ khỏi việc lựa chọn
        
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
            logger.warning("Không có cột số nào để tính Mutual Information")
            return []
        
        X = feature_df[numeric_cols]
        y = df[target_column]
        
        # Tự động phát hiện loại target nếu không được chỉ định
        if discrete_target is None:
            # Nếu target là kiểu category hoặc object, coi là rời rạc
            # Hoặc nếu target là số và có <= 20 giá trị duy nhất, coi là rời rạc
            is_categorical = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)
            is_discrete_numeric = pd.api.types.is_numeric_dtype(y) and len(y.unique()) <= 20
            discrete_target = is_categorical or is_discrete_numeric
        
        # Tính Mutual Information
        if discrete_target:
            # Cho bài toán phân loại
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            # Cho bài toán hồi quy
            mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Tạo DataFrame kết quả
        result_df = pd.DataFrame({
            'feature': numeric_cols,
            'mi_score': mi_scores
        })
        
        # Sắp xếp theo điểm Mutual Information giảm dần
        result_df = result_df.sort_values('mi_score', ascending=False)
        
        # Chọn đặc trưng
        if k is not None:
            # Chọn top-k đặc trưng
            k = min(k, len(numeric_cols))
            selected_features = result_df['feature'].tolist()[:k]
        else:
            # Chọn theo ngưỡng
            selected_features = result_df[result_df['mi_score'] >= threshold]['feature'].tolist()
        
        logger.info(f"Đã chọn {len(selected_features)} đặc trưng dựa trên Mutual Information")
        return selected_features
    
    except Exception as e:
        logger.error(f"Lỗi khi chọn đặc trưng dựa trên Mutual Information: {str(e)}")
        return []