"""
Tiện ích tiền xử lý dữ liệu cho tạo đặc trưng.
File này cung cấp các hàm để chuẩn hóa, biến đổi, và làm sạch dữ liệu
trước khi tạo đặc trưng, giúp cải thiện chất lượng và hiệu suất của mô hình.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
import warnings

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import setup_logger

# Logger
logger = setup_logger("feature_preprocessing")

def normalize_features(
    df: pd.DataFrame, 
    columns: List[str],
    method: str = "zscore",
    fit: bool = True,
    fitted_params: Optional[Dict[str, Dict[str, float]]] = None,
    epsilon: float = 1e-8,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]]:
    """
    Chuẩn hóa các đặc trưng trong DataFrame.
    
    Args:
        df: DataFrame cần chuẩn hóa
        columns: Danh sách các cột cần chuẩn hóa
        method: Phương pháp chuẩn hóa ("zscore", "minmax", "robust", "decimal")
        fit: Học các tham số mới hay không
        fitted_params: Tham số đã học trước đó
        epsilon: Số nhỏ để tránh chia cho 0
        **kwargs: Các tham số bổ sung
    
    Returns:
        DataFrame đã chuẩn hóa, hoặc tuple (DataFrame, fitted_params) nếu fit=True
    """
    # Tạo bản sao của DataFrame
    result_df = df.copy()
    
    # Kiểm tra các cột tồn tại
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logger.warning("Không có cột hợp lệ để chuẩn hóa")
        if fit:
            return result_df, {}
        return result_df
    
    if fit:
        # Khởi tạo dict cho tham số mới
        new_params = {}
    else:
        # Kiểm tra xem đã cung cấp fitted_params chưa
        if fitted_params is None:
            logger.warning("Không thể chuẩn hóa mà không có fitted_params khi fit=False")
            return result_df
    
    # Chuẩn hóa từng cột
    for col in valid_columns:
        try:
            # Bỏ qua cột nếu không phải kiểu số
            if not np.issubdtype(df[col].dtype, np.number):
                logger.warning(f"Bỏ qua cột {col} vì không phải kiểu số")
                continue
            
            # Lấy dữ liệu không null
            non_null_mask = df[col].notnull()
            col_data = df.loc[non_null_mask, col].values
            
            if len(col_data) == 0:
                logger.warning(f"Bỏ qua cột {col} vì tất cả giá trị là null")
                continue
            
            if method == "zscore":
                # Z-score normalization: (x - mean) / std
                if fit:
                    mean = np.mean(col_data)
                    std = np.std(col_data)
                    # Tránh chia cho 0
                    if std < epsilon:
                        std = epsilon
                    new_params[col] = {"mean": mean, "std": std}
                else:
                    mean = fitted_params[col]["mean"]
                    std = fitted_params[col]["std"]
                
                result_df.loc[non_null_mask, col] = (df.loc[non_null_mask, col] - mean) / std
            
            elif method == "minmax":
                # Min-max normalization: (x - min) / (max - min)
                if fit:
                    min_val = np.min(col_data)
                    max_val = np.max(col_data)
                    # Tránh chia cho 0
                    if max_val - min_val < epsilon:
                        max_val = min_val + epsilon
                    new_params[col] = {"min": min_val, "max": max_val}
                else:
                    min_val = fitted_params[col]["min"]
                    max_val = fitted_params[col]["max"]
                
                result_df.loc[non_null_mask, col] = (df.loc[non_null_mask, col] - min_val) / (max_val - min_val)
            
            elif method == "robust":
                # Robust normalization: (x - median) / IQR
                if fit:
                    median = np.median(col_data)
                    q1 = np.percentile(col_data, 25)
                    q3 = np.percentile(col_data, 75)
                    iqr = q3 - q1
                    # Tránh chia cho 0
                    if iqr < epsilon:
                        iqr = epsilon
                    new_params[col] = {"median": median, "iqr": iqr}
                else:
                    median = fitted_params[col]["median"]
                    iqr = fitted_params[col]["iqr"]
                
                result_df.loc[non_null_mask, col] = (df.loc[non_null_mask, col] - median) / iqr
            
            elif method == "decimal":
                # Decimal scaling: x / 10^j where j is smallest integer such that max(|x|) < 1
                if fit:
                    max_abs = np.max(np.abs(col_data))
                    j = int(np.ceil(np.log10(max_abs))) if max_abs > 0 else 0
                    new_params[col] = {"j": j}
                else:
                    j = fitted_params[col]["j"]
                
                scale_factor = 10 ** j
                result_df.loc[non_null_mask, col] = df.loc[non_null_mask, col] / scale_factor
            
            else:
                logger.warning(f"Phương pháp chuẩn hóa không hợp lệ: {method}")
                continue
            
            logger.debug(f"Đã chuẩn hóa cột {col} bằng phương pháp {method}")
            
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn hóa cột {col}: {e}")
    
    if fit:
        return result_df, new_params
    return result_df

def standardize_features(
    df: pd.DataFrame, 
    columns: List[str],
    center: bool = True,
    scale: bool = True,
    fit: bool = True,
    fitted_params: Optional[Dict[str, Dict[str, float]]] = None,
    epsilon: float = 1e-8,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]]:
    """
    Tiêu chuẩn hóa các đặc trưng trong DataFrame.
    Khác với normalize_features, hàm này cho phép tùy chỉnh việc trừ trung bình và chia độ lệch chuẩn.
    
    Args:
        df: DataFrame cần tiêu chuẩn hóa
        columns: Danh sách các cột cần tiêu chuẩn hóa
        center: Có trừ trung bình không
        scale: Có chia độ lệch chuẩn không
        fit: Học các tham số mới hay không
        fitted_params: Tham số đã học trước đó
        epsilon: Số nhỏ để tránh chia cho 0
        **kwargs: Các tham số bổ sung
    
    Returns:
        DataFrame đã tiêu chuẩn hóa, hoặc tuple (DataFrame, fitted_params) nếu fit=True
    """
    # Tạo bản sao của DataFrame
    result_df = df.copy()
    
    # Kiểm tra các cột tồn tại
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logger.warning("Không có cột hợp lệ để tiêu chuẩn hóa")
        if fit:
            return result_df, {}
        return result_df
    
    if fit:
        # Khởi tạo dict cho tham số mới
        new_params = {}
    else:
        # Kiểm tra xem đã cung cấp fitted_params chưa
        if fitted_params is None:
            logger.warning("Không thể tiêu chuẩn hóa mà không có fitted_params khi fit=False")
            return result_df
    
    # Tiêu chuẩn hóa từng cột
    for col in valid_columns:
        try:
            # Bỏ qua cột nếu không phải kiểu số
            if not np.issubdtype(df[col].dtype, np.number):
                logger.warning(f"Bỏ qua cột {col} vì không phải kiểu số")
                continue
            
            # Lấy dữ liệu không null
            non_null_mask = df[col].notnull()
            col_data = df.loc[non_null_mask, col].values
            
            if len(col_data) == 0:
                logger.warning(f"Bỏ qua cột {col} vì tất cả giá trị là null")
                continue
            
            # Tính toán các tham số
            if fit:
                mean = np.mean(col_data) if center else 0
                std = np.std(col_data) if scale else 1
                # Tránh chia cho 0
                if std < epsilon:
                    std = epsilon
                new_params[col] = {"mean": mean, "std": std}
            else:
                mean = fitted_params[col]["mean"]
                std = fitted_params[col]["std"]
            
            # Áp dụng tiêu chuẩn hóa
            result_df.loc[non_null_mask, col] = (df.loc[non_null_mask, col] - mean) / std
            
            logger.debug(f"Đã tiêu chuẩn hóa cột {col} (center={center}, scale={scale})")
            
        except Exception as e:
            logger.error(f"Lỗi khi tiêu chuẩn hóa cột {col}: {e}")
    
    if fit:
        return result_df, new_params
    return result_df

def min_max_scale(
    df: pd.DataFrame, 
    columns: List[str],
    min_val: float = 0,
    max_val: float = 1,
    fit: bool = True,
    fitted_params: Optional[Dict[str, Dict[str, float]]] = None,
    epsilon: float = 1e-8,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]]:
    """
    Áp dụng min-max scaling cho các đặc trưng trong DataFrame.
    
    Args:
        df: DataFrame cần xử lý
        columns: Danh sách các cột cần xử lý
        min_val: Giá trị nhỏ nhất sau khi biến đổi
        max_val: Giá trị lớn nhất sau khi biến đổi
        fit: Học các tham số mới hay không
        fitted_params: Tham số đã học trước đó
        epsilon: Số nhỏ để tránh chia cho 0
        **kwargs: Các tham số bổ sung
    
    Returns:
        DataFrame đã xử lý, hoặc tuple (DataFrame, fitted_params) nếu fit=True
    """
    # Tạo bản sao của DataFrame
    result_df = df.copy()
    
    # Kiểm tra các cột tồn tại
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logger.warning("Không có cột hợp lệ để áp dụng min-max scaling")
        if fit:
            return result_df, {}
        return result_df
    
    if fit:
        # Khởi tạo dict cho tham số mới
        new_params = {}
    else:
        # Kiểm tra xem đã cung cấp fitted_params chưa
        if fitted_params is None:
            logger.warning("Không thể áp dụng min-max scaling mà không có fitted_params khi fit=False")
            return result_df
    
    # Xử lý từng cột
    for col in valid_columns:
        try:
            # Bỏ qua cột nếu không phải kiểu số
            if not np.issubdtype(df[col].dtype, np.number):
                logger.warning(f"Bỏ qua cột {col} vì không phải kiểu số")
                continue
            
            # Lấy dữ liệu không null
            non_null_mask = df[col].notnull()
            col_data = df.loc[non_null_mask, col].values
            
            if len(col_data) == 0:
                logger.warning(f"Bỏ qua cột {col} vì tất cả giá trị là null")
                continue
            
            # Tính toán các tham số
            if fit:
                feat_min = np.min(col_data)
                feat_max = np.max(col_data)
                # Tránh chia cho 0
                if feat_max - feat_min < epsilon:
                    feat_max = feat_min + epsilon
                new_params[col] = {"feat_min": feat_min, "feat_max": feat_max}
            else:
                feat_min = fitted_params[col]["feat_min"]
                feat_max = fitted_params[col]["feat_max"]
            
            # Áp dụng min-max scaling: (x - min) / (max - min) * (max_val - min_val) + min_val
            normalized = (df.loc[non_null_mask, col] - feat_min) / (feat_max - feat_min)
            result_df.loc[non_null_mask, col] = normalized * (max_val - min_val) + min_val
            
            logger.debug(f"Đã áp dụng min-max scaling cho cột {col} (phạm vi [{min_val}, {max_val}])")
            
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng min-max scaling cho cột {col}: {e}")
    
    if fit:
        return result_df, new_params
    return result_df

def handle_extreme_values(
    df: pd.DataFrame,
    columns: List[str] = None,
    method: str = "winsorize",
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    log_transform_columns: List[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Xử lý các giá trị cực đại/cực tiểu trong DataFrame.
    
    Args:
        df: DataFrame cần xử lý
        columns: Danh sách các cột cần xử lý (None để tự phát hiện)
        method: Phương pháp xử lý ('winsorize', 'clip', 'log')
        lower_quantile: Phân vị dưới cho winsorize
        upper_quantile: Phân vị trên cho winsorize
        log_transform_columns: Các cột cần áp dụng log transform
        **kwargs: Các tham số bổ sung
    
    Returns:
        DataFrame đã xử lý giá trị cực đoan
    """
    result_df = df.copy()
    
    # Nếu không cung cấp columns, tự phát hiện các cột số
    if columns is None:
        columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Nếu không cung cấp log_transform_columns, mặc định là các cột về khối lượng
    if log_transform_columns is None:
        log_transform_columns = [col for col in columns if any(substr in col.lower() for substr in ['volume', 'vol', 'quantity', 'amount'])]
    
    # Xử lý từng cột
    for col in columns:
        if col not in result_df.columns:
            logger.warning(f"Cột {col} không tồn tại trong DataFrame")
            continue
        
        # Lấy dữ liệu không null
        non_null_mask = result_df[col].notnull()
        col_data = result_df.loc[non_null_mask, col]
        
        try:
            if method == "winsorize":
                # Winsorize (cắt bớt giá trị ngoại lệ)
                lower_bound = col_data.quantile(lower_quantile)
                upper_bound = col_data.quantile(upper_quantile)
                
                # Lưu bản sao của dữ liệu gốc trước khi winsorize
                original_data = col_data.copy()
                
                # Áp dụng winsorize
                winsorized_data = col_data.clip(lower=lower_bound, upper=upper_bound)
                result_df.loc[non_null_mask, col] = winsorized_data
                
                # Tạo cột marker để đánh dấu các giá trị đã được winsorize
                result_df[f"{col}_winsorized"] = 0
                outliers_mask = (original_data < lower_bound) | (original_data > upper_bound)
                result_df.loc[outliers_mask, f"{col}_winsorized"] = 1
                
                logger.debug(f"Đã winsorize cột {col} trong phạm vi [{lower_bound}, {upper_bound}]")
                
            elif method == "clip":
                # Clip theo số độ lệch chuẩn
                std_multiplier = kwargs.get("std_multiplier", 3.0)
                mean = col_data.mean()
                std = col_data.std()
                
                lower_bound = mean - std_multiplier * std
                upper_bound = mean + std_multiplier * std
                
                result_df.loc[non_null_mask, col] = col_data.clip(lower=lower_bound, upper=upper_bound)
                logger.debug(f"Đã clip cột {col} trong phạm vi [{lower_bound}, {upper_bound}]")
                
            # Log transform được áp dụng sau winsorize/clip nếu cột thuộc log_transform_columns
            if col in log_transform_columns:
                # Đảm bảo giá trị không âm trước khi áp dụng log
                min_val = result_df.loc[non_null_mask, col].min()
                offset = abs(min_val) + 1.0 if min_val < 0 else 1.0
                
                # Áp dụng log transform
                result_df.loc[non_null_mask, f"{col}_log"] = np.log1p(result_df.loc[non_null_mask, col] + offset - 1.0)
                logger.debug(f"Đã áp dụng log transform cho cột {col}")
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý giá trị cực đoan cho cột {col}: {e}")
    
    return result_df

def log_transform(
    df: pd.DataFrame, 
    columns: List[str],
    base: float = np.e,
    offset: float = 1.0,
    **kwargs
) -> pd.DataFrame:
    """
    Áp dụng biến đổi logarithm cho các đặc trưng.
    
    Args:
        df: DataFrame cần xử lý
        columns: Danh sách các cột cần xử lý
        base: Cơ số logarithm (e, 10, 2)
        offset: Giá trị cộng thêm để tránh log(0)
        **kwargs: Các tham số bổ sung
    
    Returns:
        DataFrame đã biến đổi
    """
    # Tạo bản sao của DataFrame
    result_df = df.copy()
    
    # Kiểm tra các cột tồn tại
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logger.warning("Không có cột hợp lệ để áp dụng biến đổi logarithm")
        return result_df
    
    # Xử lý từng cột
    for col in valid_columns:
        try:
            # Bỏ qua cột nếu không phải kiểu số
            if not np.issubdtype(df[col].dtype, np.number):
                logger.warning(f"Bỏ qua cột {col} vì không phải kiểu số")
                continue
            
            # Lấy dữ liệu không null
            non_null_mask = df[col].notnull()
            
            # Kiểm tra giá trị âm
            min_val = df.loc[non_null_mask, col].min()
            if min_val < 0:
                logger.warning(f"Cột {col} có giá trị âm, điều chỉnh offset thành {abs(min_val) + offset}")
                adjusted_offset = abs(min_val) + offset
            else:
                adjusted_offset = offset
            
            # Áp dụng biến đổi logarithm
            if base == np.e:
                result_df.loc[non_null_mask, col] = np.log(df.loc[non_null_mask, col] + adjusted_offset)
            elif base == 10:
                result_df.loc[non_null_mask, col] = np.log10(df.loc[non_null_mask, col] + adjusted_offset)
            elif base == 2:
                result_df.loc[non_null_mask, col] = np.log2(df.loc[non_null_mask, col] + adjusted_offset)
            else:
                result_df.loc[non_null_mask, col] = np.log(df.loc[non_null_mask, col] + adjusted_offset) / np.log(base)
            
            logger.debug(f"Đã áp dụng biến đổi logarithm (base={base}) cho cột {col}")
            
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng biến đổi logarithm cho cột {col}: {e}")
    
    return result_df

def power_transform(
    df: pd.DataFrame, 
    columns: List[str],
    power: float = 0.5,
    offset: float = 0.0,
    **kwargs
) -> pd.DataFrame:
    """
    Áp dụng biến đổi lũy thừa cho các đặc trưng.
    
    Args:
        df: DataFrame cần xử lý
        columns: Danh sách các cột cần xử lý
        power: Số mũ (0.5 cho căn bậc hai, -1 cho nghịch đảo)
        offset: Giá trị cộng thêm
        **kwargs: Các tham số bổ sung
    
    Returns:
        DataFrame đã biến đổi
    """
    # Tạo bản sao của DataFrame
    result_df = df.copy()
    
    # Kiểm tra các cột tồn tại
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logger.warning("Không có cột hợp lệ để áp dụng biến đổi lũy thừa")
        return result_df
    
    # Xử lý từng cột
    for col in valid_columns:
        try:
            # Bỏ qua cột nếu không phải kiểu số
            if not np.issubdtype(df[col].dtype, np.number):
                logger.warning(f"Bỏ qua cột {col} vì không phải kiểu số")
                continue
            
            # Lấy dữ liệu không null
            non_null_mask = df[col].notnull()
            
            # Kiểm tra giá trị dữ liệu
            data = df.loc[non_null_mask, col].values
            
            # Nếu power < 0 hoặc không phải số nguyên, cần đảm bảo dữ liệu > 0
            if power < 0 or (power < 1 and power > 0):
                min_val = np.min(data)
                if min_val <= 0:
                    adjusted_offset = abs(min_val) + 1.0 + offset
                    logger.warning(f"Cột {col} có giá trị <= 0, điều chỉnh offset thành {adjusted_offset}")
                else:
                    adjusted_offset = offset
            else:
                adjusted_offset = offset
            
            # Áp dụng biến đổi lũy thừa
            result_df.loc[non_null_mask, col] = np.power(df.loc[non_null_mask, col] + adjusted_offset, power)
            
            logger.debug(f"Đã áp dụng biến đổi lũy thừa (power={power}) cho cột {col}")
            
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng biến đổi lũy thừa cho cột {col}: {e}")
    
    return result_df

def winsorize(
    df: pd.DataFrame, 
    columns: List[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    fit: bool = True,
    fitted_params: Optional[Dict[str, Dict[str, float]]] = None,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]]:
    """
    Áp dụng winsorization để cắt bớt các giá trị ngoại lai.
    
    Args:
        df: DataFrame cần xử lý
        columns: Danh sách các cột cần xử lý
        lower_quantile: Phân vị dưới (0-1)
        upper_quantile: Phân vị trên (0-1)
        fit: Học các tham số mới hay không
        fitted_params: Tham số đã học trước đó
        **kwargs: Các tham số bổ sung
    
    Returns:
        DataFrame đã xử lý, hoặc tuple (DataFrame, fitted_params) nếu fit=True
    """
    # Tạo bản sao của DataFrame
    result_df = df.copy()
    
    # Kiểm tra các cột tồn tại
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logger.warning("Không có cột hợp lệ để áp dụng winsorization")
        if fit:
            return result_df, {}
        return result_df
    
    if fit:
        # Khởi tạo dict cho tham số mới
        new_params = {}
    else:
        # Kiểm tra xem đã cung cấp fitted_params chưa
        if fitted_params is None:
            logger.warning("Không thể áp dụng winsorization mà không có fitted_params khi fit=False")
            return result_df
    
    # Xử lý từng cột
    for col in valid_columns:
        try:
            # Bỏ qua cột nếu không phải kiểu số
            if not np.issubdtype(df[col].dtype, np.number):
                logger.warning(f"Bỏ qua cột {col} vì không phải kiểu số")
                continue
            
            # Lấy dữ liệu không null
            non_null_mask = df[col].notnull()
            col_data = df.loc[non_null_mask, col].values
            
            if len(col_data) == 0:
                logger.warning(f"Bỏ qua cột {col} vì tất cả giá trị là null")
                continue
            
            # Tính toán các giới hạn
            if fit:
                lower_bound = np.quantile(col_data, lower_quantile)
                upper_bound = np.quantile(col_data, upper_quantile)
                new_params[col] = {"lower_bound": lower_bound, "upper_bound": upper_bound}
            else:
                lower_bound = fitted_params[col]["lower_bound"]
                upper_bound = fitted_params[col]["upper_bound"]
            
            # Áp dụng winsorization
            winsorized_data = np.copy(col_data)
            winsorized_data[winsorized_data < lower_bound] = lower_bound
            winsorized_data[winsorized_data > upper_bound] = upper_bound
            
            result_df.loc[non_null_mask, col] = winsorized_data
            
            logger.debug(f"Đã áp dụng winsorization cho cột {col} (phạm vi [{lower_bound}, {upper_bound}])")
            
        except Exception as e:
            logger.error(f"Lỗi khi áp dụng winsorization cho cột {col}: {e}")
    
    if fit:
        return result_df, new_params
    return result_df

def polynomial_features(
    df: pd.DataFrame, 
    columns: List[str],
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Tạo các đặc trưng đa thức từ các đặc trưng hiện có.
    
    Args:
        df: DataFrame cần xử lý
        columns: Danh sách các cột cần xử lý
        degree: Bậc đa thức
        interaction_only: Chỉ tạo các thành phần tương tác
        include_bias: Thêm cột hằng số 1
        **kwargs: Các tham số bổ sung
    
    Returns:
        DataFrame với các đặc trưng đa thức mới
    """
    # Tạo bản sao của DataFrame
    result_df = df.copy()
    
    # Kiểm tra các cột tồn tại
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logger.warning("Không có cột hợp lệ để tạo đặc trưng đa thức")
        return result_df
    
    try:
        from sklearn.preprocessing import PolynomialFeatures
        import numpy as np
        
        # Lấy dữ liệu đầu vào
        X = df[valid_columns].values
        
        # Tạo biến đổi đa thức
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        
        # Áp dụng biến đổi
        X_poly = poly.fit_transform(X)
        
        # Tạo tên cho các đặc trưng mới
        feature_names = poly.get_feature_names_out(valid_columns)
        
        # Thêm các đặc trưng mới vào DataFrame
        for i, name in enumerate(feature_names):
            # Bỏ qua cột đầu tiên nếu là bias
            if i == 0 and include_bias:
                continue
                
            # Chuẩn hóa tên đặc trưng
            clean_name = name.replace(" ", "").replace("^", "_pow_")
            
            # Kiểm tra trùng lặp tên
            if clean_name in result_df.columns:
                clean_name = f"poly_{clean_name}"
            
            # Thêm đặc trưng mới
            result_df[clean_name] = X_poly[:, i]
        
        logger.info(f"Đã tạo {X_poly.shape[1] - 1 - include_bias} đặc trưng đa thức mới")
        
    except ImportError:
        logger.error("Không thể tạo đặc trưng đa thức (scikit-learn chưa được cài đặt)")
    except Exception as e:
        logger.error(f"Lỗi khi tạo đặc trưng đa thức: {e}")
    
    return result_df

def bin_features(
    df: pd.DataFrame, 
    columns: List[str],
    n_bins: int = 10,
    strategy: str = "quantile", 
    labels: Optional[List[str]] = None,
    fit: bool = True,
    fitted_params: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]]:
    """
    Chia các đặc trưng thành các bin.
    
    Args:
        df: DataFrame cần xử lý
        columns: Danh sách các cột cần xử lý
        n_bins: Số lượng bin
        strategy: Chiến lược tạo bin ('uniform', 'quantile', 'kmeans')
        labels: Nhãn cho các bin
        fit: Học các tham số mới hay không
        fitted_params: Tham số đã học trước đó
        **kwargs: Các tham số bổ sung
    
    Returns:
        DataFrame đã xử lý, hoặc tuple (DataFrame, fitted_params) nếu fit=True
    """
    # Tạo bản sao của DataFrame
    result_df = df.copy()
    
    # Kiểm tra các cột tồn tại
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        logger.warning("Không có cột hợp lệ để chia thành các bin")
        if fit:
            return result_df, {}
        return result_df
    
    if fit:
        # Khởi tạo dict cho tham số mới
        new_params = {}
    else:
        # Kiểm tra xem đã cung cấp fitted_params chưa
        if fitted_params is None:
            logger.warning("Không thể chia thành các bin mà không có fitted_params khi fit=False")
            return result_df
    
    try:
        from sklearn.preprocessing import KBinsDiscretizer
        
        # Xử lý từng cột
        for col in valid_columns:
            try:
                # Bỏ qua cột nếu không phải kiểu số
                if not np.issubdtype(df[col].dtype, np.number):
                    logger.warning(f"Bỏ qua cột {col} vì không phải kiểu số")
                    continue
                
                # Lấy dữ liệu không null
                non_null_mask = df[col].notnull()
                X = df.loc[non_null_mask, col].values.reshape(-1, 1)
                
                if len(X) == 0:
                    logger.warning(f"Bỏ qua cột {col} vì tất cả giá trị là null")
                    continue
                
                if fit:
                    # Tạo và fit bộ rời rạc hóa
                    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                    kbd.fit(X)
                    new_params[col] = {"kbd": kbd}
                else:
                    kbd = fitted_params[col]["kbd"]
                
                # Áp dụng rời rạc hóa
                X_binned = kbd.transform(X).flatten()
                
                # Tạo tên cột mới
                new_col = f"{col}_bin"
                
                # Thêm cột mới với các bin
                result_df.loc[non_null_mask, new_col] = X_binned
                
                # Thêm thông tin nhãn nếu có
                if labels is not None and len(labels) == n_bins:
                    result_df.loc[non_null_mask, new_col] = result_df.loc[non_null_mask, new_col].map(
                        {i: labels[i] for i in range(n_bins)}
                    )
                
                logger.debug(f"Đã chia cột {col} thành {n_bins} bin")
                
            except Exception as e:
                logger.error(f"Lỗi khi chia cột {col} thành các bin: {e}")
                
    except ImportError:
        logger.error("Không thể chia thành các bin (scikit-learn chưa được cài đặt)")
        # Trả về DataFrame gốc
        if fit:
            return result_df, {}
        return result_df
    
    if fit:
        return result_df, new_params
    return result_df

def normalize_technical_indicators(
    df: pd.DataFrame, 
    indicators_config: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Chuẩn hóa các chỉ báo kỹ thuật trong DataFrame.
    
    Args:
        df: DataFrame chứa các chỉ báo kỹ thuật
        indicators_config: Cấu hình cho từng loại chỉ báo
            Ví dụ: {
                'rsi': {'min': 0, 'max': 100},
                'macd': {'method': 'zscore'},
                'volume': {'method': 'log'},
                'atr': {'method': 'percent_of_price', 'price_col': 'close'}
            }
        **kwargs: Tham số bổ sung
    
    Returns:
        DataFrame đã chuẩn hóa
    """
    result_df = df.copy()
    
    # Sử dụng cấu hình mặc định nếu không được cung cấp
    if indicators_config is None:
        indicators_config = {
            'rsi': {'min': 0, 'max': 100},
            'macd': {'method': 'zscore'},
            'volume': {'method': 'log'},
            'obv': {'method': 'zscore'},
            'adx': {'method': 'minmax', 'min': 0, 'max': 100},
            'atr': {'method': 'zscore'}
        }
    
    # Xử lý từng loại chỉ báo
    for indicator_type, config in indicators_config.items():
        # Tìm các cột chứa tên chỉ báo
        indicator_cols = [col for col in result_df.columns if indicator_type.lower() in col.lower()]
        
        if not indicator_cols:
            continue
            
        # Lấy phương pháp chuẩn hóa
        method = config.get('method', 'zscore')
        
        # Xử lý theo từng loại chỉ báo
        if indicator_type.lower() == 'rsi':
            # RSI luôn nằm trong khoảng [0, 100]
            for col in indicator_cols:
                result_df[col] = np.clip(result_df[col], 0, 100)
                
                # Nếu cần chuẩn hóa về [0, 1]
                if config.get('normalize', False):
                    result_df[col] = result_df[col] / 100
        
        elif indicator_type.lower() == 'volume':
            # Log transform cho volume
            for col in indicator_cols:
                # Thêm 1 để tránh log(0)
                result_df[f"{col}_log"] = np.log1p(result_df[col])
                
                # Tính volume tương đối
                if 'window' in config:
                    window = config['window']
                    vol_mean = result_df[col].rolling(window=window).mean()
                    vol_mean = vol_mean.replace(0, np.nan)  # Tránh chia cho 0
                    result_df[f"{col}_rel"] = result_df[col] / vol_mean
        
        elif method == 'zscore':
            # Chuẩn hóa Z-score
            for col in indicator_cols:
                # Bỏ qua nếu tất cả là NaN
                if result_df[col].isna().all():
                    continue
                    
                # Tính z-score
                mean = result_df[col].mean()
                std = result_df[col].std()
                
                # Tránh chia cho 0
                if std < 1e-8:
                    std = 1e-8
                
                result_df[f"{col}_std"] = (result_df[col] - mean) / std
        
        elif method == 'minmax':
            # Min-max scaling
            for col in indicator_cols:
                # Bỏ qua nếu tất cả là NaN
                if result_df[col].isna().all():
                    continue
                
                # Lấy min, max từ config hoặc từ dữ liệu
                min_val = config.get('min', result_df[col].min())
                max_val = config.get('max', result_df[col].max())
                
                # Tránh chia cho 0
                if max_val - min_val < 1e-8:
                    max_val = min_val + 1e-8
                
                # Chuẩn hóa
                result_df[f"{col}_norm"] = (result_df[col] - min_val) / (max_val - min_val)
        
        elif method == 'percent_of_price':
            # Chuẩn hóa theo phần trăm giá
            price_col = config.get('price_col', 'close')
            if price_col in result_df.columns:
                for col in indicator_cols:
                    # Tránh chia cho 0
                    price = result_df[price_col].replace(0, np.nan)
                    result_df[f"{col}_pct"] = (result_df[col] / price) * 100

        # Thêm đoạn code sau:
        elif indicator_type.lower() == 'macd':
            # Xử lý chỉ báo MACD một cách đặc biệt
            for col in indicator_cols:
                # Kiểm tra giá trị không hợp lệ và cực đoan
                invalid_mask = ~np.isfinite(result_df[col])
                if invalid_mask.any():
                    # Thay thế các giá trị không hợp lệ bằng 0
                    logger.warning(f"Thay thế {invalid_mask.sum()} giá trị không hợp lệ trong {col}")
                    result_df.loc[invalid_mask, col] = 0
                
                # Nếu cần chuẩn hóa về phạm vi cụ thể
                if config.get('normalize', False):
                    # Tính toán thống kê mô tả
                    median = result_df[col].median()
                    q1 = result_df[col].quantile(0.25)
                    q3 = result_df[col].quantile(0.75)
                    iqr = q3 - q1
                    
                    if iqr > 1e-8:  # Tránh chia cho giá trị quá nhỏ
                        # Phát hiện và xử lý ngoại lệ
                        lower_bound = median - 3 * iqr
                        upper_bound = median + 3 * iqr
                        
                        # Đếm số lượng ngoại lệ
                        outliers = (result_df[col] < lower_bound) | (result_df[col] > upper_bound)
                        if outliers.any():
                            logger.warning(f"Phát hiện {outliers.sum()} ngoại lệ trong {col}")
                            
                            # Giới hạn giá trị trong phạm vi [lower_bound, upper_bound]
                            result_df.loc[outliers, col] = result_df.loc[outliers, col].clip(lower=lower_bound, upper=upper_bound)
                        
                        # Áp dụng chuẩn hóa min-max sau khi xử lý ngoại lệ
                        min_val = result_df[col].min()
                        max_val = result_df[col].max()
                        range_val = max_val - min_val
                        
                        if range_val > 1e-8:
                            norm_col = f"{col}_norm"
                            result_df[norm_col] = (result_df[col] - min_val) / range_val
                            logger.debug(f"Đã chuẩn hóa {col} về phạm vi [0, 1]")
                        else:
                            # Nếu range quá nhỏ, đặt tất cả giá trị về 0.5
                            norm_col = f"{col}_norm"
                            result_df[norm_col] = 0.5
                    else:
                        # Nếu IQR quá nhỏ, không chuẩn hóa
                        logger.warning(f"Không thể chuẩn hóa {col} do biến thiên quá nhỏ")

    return result_df

def standardize_technical_indicators(
    df: pd.DataFrame,
    indicators_config: Optional[Dict[str, Dict[str, Any]]] = None,
    window: int = 100,
    auto_detect: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Chuẩn hóa các chỉ báo kỹ thuật để có trung bình 0 và độ lệch chuẩn 1.
    
    Args:
        df: DataFrame chứa các chỉ báo kỹ thuật
        indicators_config: Cấu hình cho từng loại chỉ báo
            Ví dụ: {
                'macd': {'cols': ['macd', 'macd_signal'], 'method': 'zscore'},
                'volume': {'cols': ['volume'], 'method': 'log'},
                'obv': {'cols': ['obv'], 'method': 'rolling_zscore'},
                'adx': {'cols': ['adx'], 'method': 'minmax'}
            }
        window: Cỡ cửa sổ cho chuẩn hóa rolling
        auto_detect: Tự động phát hiện các chỉ báo để chuẩn hóa
        **kwargs: Các tham số bổ sung
    
    Returns:
        DataFrame đã chuẩn hóa
    """
    result_df = df.copy()
    
    # Cấu hình mặc định nếu không được cung cấp
    if indicators_config is None:
        indicators_config = {
            'macd': {'method': 'zscore'},
            'obv': {'method': 'rolling_zscore'},
            'volume': {'method': 'log'},
            'adx': {'method': 'minmax_01'},
            'rsi': {'method': 'minmax_01'},
            'cci': {'method': 'robust'},
            'atr': {'method': 'rolling_zscore'}
        }
    
    # Tự động phát hiện các cột cho mỗi loại chỉ báo
    if auto_detect:
        for indicator_type, config in indicators_config.items():
            if 'cols' not in config:
                # Tìm tất cả các cột có chứa tên chỉ báo
                indicator_cols = [col for col in result_df.columns if indicator_type.lower() in col.lower()]
                config['cols'] = indicator_cols
    
    # Xử lý từng loại chỉ báo
    for indicator_type, config in indicators_config.items():
        # Lấy phương pháp chuẩn hóa
        method = config.get('method', 'zscore')
        cols = config.get('cols', [])
        
        if not cols:
            continue
        
        logger.info(f"Chuẩn hóa {len(cols)} cột của loại chỉ báo {indicator_type} bằng phương pháp {method}")
        
        for col in cols:
            if col not in result_df.columns:
                continue
                
            # Bỏ qua nếu cột đã được chuẩn hóa
            if col.endswith('_std') or col.endswith('_norm') or col.endswith('_scaled'):
                continue
                
            # Xử lý giá trị NaN/Inf
            invalid_mask = ~np.isfinite(result_df[col])
            if invalid_mask.any():
                valid_data = result_df.loc[~invalid_mask, col]
                if not valid_data.empty:
                    result_df.loc[invalid_mask, col] = valid_data.median()
                else:
                    result_df.loc[invalid_mask, col] = 0
            
            # Áp dụng phương pháp chuẩn hóa
            if method == 'zscore':
                # Standard z-score: (x - mean) / std
                mean = result_df[col].mean()
                std = result_df[col].std()
                
                if std > 1e-8:  # Tránh chia cho 0
                    result_df[f"{col}_std"] = (result_df[col] - mean) / std
                else:
                    result_df[f"{col}_std"] = 0
                    logger.warning(f"Không thể áp dụng z-score cho cột {col} do std quá nhỏ")
            
            elif method == 'rolling_zscore':
                # Rolling z-score trong cửa sổ
                rolling_mean = result_df[col].rolling(window=window, min_periods=1).mean()
                rolling_std = result_df[col].rolling(window=window, min_periods=1).std()
                
                # Tránh chia cho 0
                rolling_std = rolling_std.replace(0, np.nan).fillna(1e-8)
                
                result_df[f"{col}_std"] = (result_df[col] - rolling_mean) / rolling_std
                
                # Xử lý các giá trị không hợp lệ
                result_df[f"{col}_std"] = result_df[f"{col}_std"].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            elif method == 'log':
                # Log transform
                if (result_df[col] <= 0).any():
                    # Đảm bảo giá trị dương trước khi áp dụng log
                    min_val = result_df[col].min()
                    offset = abs(min_val) + 1 if min_val <= 0 else 1
                    result_df[f"{col}_log"] = np.log1p(result_df[col] + offset - 1)
                else:
                    result_df[f"{col}_log"] = np.log1p(result_df[col])
            
            elif method == 'minmax_01':
                # Min-max scaling to [0, 1]
                min_val = result_df[col].min()
                max_val = result_df[col].max()
                
                if max_val - min_val > 1e-8:  # Tránh chia cho 0
                    result_df[f"{col}_norm"] = (result_df[col] - min_val) / (max_val - min_val)
                else:
                    result_df[f"{col}_norm"] = 0.5
                    logger.warning(f"Không thể áp dụng min-max cho cột {col} do phạm vi quá nhỏ")
            
            elif method == 'minmax_11':
                # Min-max scaling to [-1, 1]
                min_val = result_df[col].min()
                max_val = result_df[col].max()
                
                if max_val - min_val > 1e-8:  # Tránh chia cho 0
                    result_df[f"{col}_norm"] = 2 * (result_df[col] - min_val) / (max_val - min_val) - 1
                else:
                    result_df[f"{col}_norm"] = 0
                    logger.warning(f"Không thể áp dụng min-max cho cột {col} do phạm vi quá nhỏ")
            
            elif method == 'robust':
                # Robust scaling: (x - median) / IQR
                median = result_df[col].median()
                q1 = result_df[col].quantile(0.25)
                q3 = result_df[col].quantile(0.75)
                iqr = q3 - q1
                
                if iqr > 1e-8:  # Tránh chia cho 0
                    result_df[f"{col}_scaled"] = (result_df[col] - median) / iqr
                else:
                    result_df[f"{col}_scaled"] = 0
                    logger.warning(f"Không thể áp dụng robust scaling cho cột {col} do IQR quá nhỏ")
    
    return result_df

def detect_and_fix_indicator_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'zscore',
    threshold: float = 3.0,
    **kwargs
) -> pd.DataFrame:
    """
    Phát hiện và xử lý các giá trị ngoại lệ trong các chỉ báo kỹ thuật.
    
    Args:
        df: DataFrame chứa các chỉ báo
        columns: Danh sách các cột cần xử lý (None để tự phát hiện)
        method: Phương pháp phát hiện ('zscore', 'iqr')
        threshold: Ngưỡng xác định ngoại lệ
        **kwargs: Tham số bổ sung
    
    Returns:
        DataFrame đã xử lý ngoại lệ
    """
    result_df = df.copy()
    
    # Nếu không cung cấp columns, tự phát hiện các chỉ báo
    if columns is None:
        # Các tiền tố thường gặp của các chỉ báo kỹ thuật
        indicator_prefixes = [
            'rsi', 'macd', 'obv', 'volume', 'adx', 'atr', 'cci', 
            'cmf', 'mfi', 'vo', 'bbw', 'stddev', 'supertrend'
        ]
        
        columns = []
        for col in result_df.columns:
            for prefix in indicator_prefixes:
                if prefix in col.lower() and col not in columns:
                    columns.append(col)
                    break
    
    # Xử lý từng cột
    for col in columns:
        if col not in result_df.columns or not np.issubdtype(result_df[col].dtype, np.number):
            continue
            
        # Lấy dữ liệu không null
        non_null_mask = result_df[col].notnull()
        col_data = result_df.loc[non_null_mask, col]
        
        if len(col_data) == 0:
            continue
            
        if method == 'zscore':
            # Z-score method
            mean = col_data.mean()
            std = col_data.std()
            
            if std > 0:  # Tránh chia cho 0
                z_scores = np.abs((col_data - mean) / std)
                outliers = z_scores > threshold
                
                # Thay thế ngoại lệ bằng giá trị giới hạn
                if outliers.any():
                    upper_limit = mean + threshold * std
                    lower_limit = mean - threshold * std
                    
                    result_df.loc[(non_null_mask) & (result_df[col] > upper_limit), col] = upper_limit
                    result_df.loc[(non_null_mask) & (result_df[col] < lower_limit), col] = lower_limit
        
        elif method == 'iqr':
            # IQR method
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            result_df.loc[(non_null_mask) & (result_df[col] > upper_bound), col] = upper_bound
            result_df.loc[(non_null_mask) & (result_df[col] < lower_bound), col] = lower_bound
    
    return result_df

def add_trend_strength_indicator(
    df: pd.DataFrame,
    price_col: str = 'close',
    ema_period: int = 89,
    slope_period: int = 5,
    normalize: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Thêm chỉ báo độ mạnh xu hướng dựa trên độ dốc (slope) của EMA.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_col: Tên cột giá
        ema_period: Kích thước cửa sổ cho EMA
        slope_period: Kích thước cửa sổ cho việc tính độ dốc
        normalize: Chuẩn hóa kết quả
        **kwargs: Tham số bổ sung
    
    Returns:
        DataFrame với các chỉ báo xu hướng mới
    """
    result_df = df.copy()
    
    # Kiểm tra xem cột giá tồn tại không
    if price_col not in result_df.columns:
        logger.warning(f"Cột giá {price_col} không tồn tại trong DataFrame")
        return result_df
    
    # Tính EMA
    ema_col = f"ema_{ema_period}"
    if ema_col not in result_df.columns:
        result_df[ema_col] = result_df[price_col].ewm(span=ema_period, adjust=False).mean()
    
    # Tính slope (độ dốc)
    slope = result_df[ema_col].diff(slope_period) / slope_period
    
    # Tính acceleration (gia tốc) - đạo hàm bậc 2
    acceleration = slope.diff()
    
    # Chuẩn hóa nếu được yêu cầu
    if normalize:
        # Sử dụng cửa sổ 100 period để chuẩn hóa
        window = kwargs.get('window', 100)
        
        # Đối với slope
        slope_rolling_std = slope.rolling(window=window, min_periods=1).std()
        slope_rolling_mean = slope.rolling(window=window, min_periods=1).mean()
        
        # Tránh chia cho 0
        slope_rolling_std = slope_rolling_std.replace(0, np.nan)
        
        # Z-score chuẩn hóa
        norm_slope = (slope - slope_rolling_mean) / slope_rolling_std
        
        # Xử lý các giá trị không hợp lệ
        norm_slope = norm_slope.replace([np.inf, -np.inf], np.nan)
        norm_slope = norm_slope.fillna(0)
        
        # Đối với acceleration
        acc_rolling_std = acceleration.rolling(window=window, min_periods=1).std()
        acc_rolling_mean = acceleration.rolling(window=window, min_periods=1).mean()
        
        # Tránh chia cho 0
        acc_rolling_std = acc_rolling_std.replace(0, np.nan)
        
        # Z-score chuẩn hóa
        norm_acceleration = (acceleration - acc_rolling_mean) / acc_rolling_std
        
        # Xử lý các giá trị không hợp lệ
        norm_acceleration = norm_acceleration.replace([np.inf, -np.inf], np.nan)
        norm_acceleration = norm_acceleration.fillna(0)
        
        # Đặt tên các cột kết quả
        result_df["trend_strength"] = norm_slope
        result_df["trend_acceleration"] = norm_acceleration
    else:
        # Đặt tên các cột kết quả không chuẩn hóa
        result_df["trend_slope"] = slope
        result_df["trend_acceleration"] = acceleration
    
    return result_df

def add_volatility_zscore(
    df: pd.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    atr_period: int = 14,
    zscore_window: int = 50,
    **kwargs
) -> pd.DataFrame:
    """
    Thêm chỉ báo Z-score của biến động (ATR).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        high_col: Tên cột giá cao
        low_col: Tên cột giá thấp
        close_col: Tên cột giá đóng cửa
        atr_period: Kích thước cửa sổ cho ATR
        zscore_window: Kích thước cửa sổ cho việc tính Z-score
        **kwargs: Tham số bổ sung
    
    Returns:
        DataFrame với các chỉ báo biến động mới
    """
    result_df = df.copy()
    
    # Kiểm tra xem các cột tồn tại không
    required_cols = [high_col, low_col, close_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]
    if missing_cols:
        logger.warning(f"Các cột sau không tồn tại trong DataFrame: {missing_cols}")
        return result_df
    
    # Kiểm tra xem ATR đã được tính toán chưa
    atr_col = f"atr_{atr_period}"
    if atr_col not in result_df.columns:
        # Tính True Range
        high = result_df[high_col]
        low = result_df[low_col]
        close = result_df[close_col]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Tính ATR
        result_df[atr_col] = tr.ewm(span=atr_period, min_periods=atr_period, adjust=False).mean()
    
    # Tính ATR percent (ATR/Close)
    atr_pct_col = f"atr_pct_{atr_period}"
    if atr_pct_col not in result_df.columns:
        # Tránh chia cho 0
        close_non_zero = result_df[close_col].replace(0, np.nan)
        result_df[atr_pct_col] = result_df[atr_col] / close_non_zero * 100
    
    # Tính Z-score của ATR
    # Tính trung bình và độ lệch chuẩn của ATR trong cửa sổ
    rolling_mean = result_df[atr_col].rolling(window=zscore_window, min_periods=1).mean()
    rolling_std = result_df[atr_col].rolling(window=zscore_window, min_periods=1).std()
    
    # Tránh chia cho 0
    rolling_std = rolling_std.replace(0, np.nan)
    
    # Tính Z-score
    volatility_zscore = (result_df[atr_col] - rolling_mean) / rolling_std
    
    # Xử lý các giá trị không hợp lệ
    volatility_zscore = volatility_zscore.replace([np.inf, -np.inf], np.nan)
    volatility_zscore = volatility_zscore.fillna(0)
    
    # Đặt tên cột kết quả
    result_df["volatility_zscore"] = volatility_zscore
    
    # Tính biến động tương đối so với quá khứ (percentile)
    def rolling_rank(x):
        ranks = x.rank(pct=True)
        return ranks.iloc[-1] if not ranks.empty else np.nan
    
    volatility_rank = result_df[atr_col].rolling(window=zscore_window, min_periods=1).apply(
        rolling_rank, 
        raw=False  # Cần False để xử lý Series
    )
    
    # Đặt tên cột kết quả
    result_df["volatility_rank"] = volatility_rank
    
    return result_df

def generate_labels(
    df: pd.DataFrame,
    price_col: str = 'close',
    window: int = 10,
    threshold: float = 0.01,
    **kwargs
) -> pd.DataFrame:
    """
    Tạo nhãn cho dữ liệu dựa trên biến động giá tương lai.
    
    Args:
        df: DataFrame với dữ liệu giá
        price_col: Tên cột giá
        window: Cửa sổ thời gian tương lai (số candle)
        threshold: Ngưỡng % thay đổi giá để xác định nhãn
        **kwargs: Tham số bổ sung
    
    Returns:
        DataFrame với cột nhãn mới
    """
    result_df = df.copy()
    
    # Kiểm tra xem cột giá tồn tại không
    if price_col not in result_df.columns:
        logger.warning(f"Cột giá {price_col} không tồn tại trong DataFrame")
        return result_df
    
    # Tính % thay đổi giá trong tương lai
    future_returns = result_df[price_col].shift(-window) / result_df[price_col] - 1
    
    # Tạo nhãn: 1 (mua) nếu lợi nhuận > threshold, -1 (bán) nếu lỗ > threshold, 0 (giữ nguyên) nếu khác
    result_df['label'] = 0
    result_df.loc[future_returns > threshold, 'label'] = 1  # Long signal
    result_df.loc[future_returns < -threshold, 'label'] = -1  # Short signal
    
    # Thêm các biến trợ giúp
    result_df['future_return_pct'] = future_returns * 100  # Dưới dạng %
    
    return result_df

def generate_labels(
    df: pd.DataFrame,
    price_col: str = 'close',
    window: int = 10,
    threshold: float = 0.01,
    **kwargs
) -> pd.DataFrame:
    """
    Tạo nhãn cho dữ liệu dựa trên biến động giá tương lai.
    
    Args:
        df: DataFrame với dữ liệu giá
        price_col: Tên cột giá
        window: Cửa sổ thời gian tương lai (số candle)
        threshold: Ngưỡng % thay đổi giá để xác định nhãn
        **kwargs: Tham số bổ sung
    
    Returns:
        DataFrame với cột nhãn mới
    """
    result_df = df.copy()
    
    # Kiểm tra xem cột giá tồn tại không
    if price_col not in result_df.columns:
        logger.warning(f"Cột giá {price_col} không tồn tại trong DataFrame")
        return result_df
    
    # Tính % thay đổi giá trong tương lai
    future_returns = result_df[price_col].shift(-window) / result_df[price_col] - 1
    
    # Tạo nhãn: 1 (mua) nếu lợi nhuận > threshold, -1 (bán) nếu lỗ > threshold, 0 (giữ nguyên) nếu khác
    result_df['label'] = 0
    result_df.loc[future_returns > threshold, 'label'] = 1  # Long signal
    result_df.loc[future_returns < -threshold, 'label'] = -1  # Short signal
    
    # Thêm các biến trợ giúp
    result_df['future_return_pct'] = future_returns * 100  # Dưới dạng %
    
    return result_df

def clean_technical_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Làm sạch và chuẩn hóa các chỉ báo kỹ thuật trong DataFrame.
    Hàm tổng hợp xử lý tất cả các vấn đề liên quan đến chỉ báo kỹ thuật.
    
    Args:
        df: DataFrame chứa các chỉ báo kỹ thuật
        config: Cấu hình cho việc xử lý
        **kwargs: Tham số bổ sung
    
    Returns:
        DataFrame đã xử lý
    """
    if config is None:
        config = {
            'fix_outliers': True,
            'outlier_method': 'zscore',
            'outlier_threshold': 3.0,
            'normalize_indicators': True,
            'standardize_indicators': True,
            'add_trend_strength': True,
            'add_volatility_zscore': True,
            'generate_labels': True,
            'label_window': 10,
            'label_threshold': 0.01
        }
    
    result_df = df.copy()
    
    # Bước 1: Sửa chữa các giá trị ngoại lệ
    if config.get('fix_outliers', True):
        from config.logging_config import setup_logger
        from data_processors.cleaners.outlier_detector import OutlierDetector
        
        logger = setup_logger("clean_technical_features")
        logger.info("Bắt đầu sửa chữa các giá trị ngoại lệ...")
        
        # Tạo detector
        detector = OutlierDetector(
            method=config.get('outlier_method', 'zscore'),
            threshold=config.get('outlier_threshold', 3.0),
            use_robust=config.get('use_robust', True)
        )
        
        # Phát hiện và sửa ngoại lệ
        result_df = detector.check_technical_indicators(result_df)
        
        # Sử dụng phương thức nâng cao nếu có
        if hasattr(detector, 'check_and_fix_technical_indicators_advanced'):
            result_df = detector.check_and_fix_technical_indicators_advanced(result_df)
        
        logger.info("Đã hoàn thành sửa chữa giá trị ngoại lệ")
    
    # Bước 2: Chuẩn hóa các chỉ báo
    if config.get('normalize_indicators', True):
        logger.info("Bắt đầu chuẩn hóa các chỉ báo kỹ thuật...")
        result_df = normalize_technical_indicators(
            result_df,
            indicators_config=config.get('indicators_config')
        )
        logger.info("Đã hoàn thành chuẩn hóa các chỉ báo")
    
    # Bước 2b: Standardize các chỉ báo
    if config.get('standardize_indicators', True):
        logger.info("Bắt đầu standardize các chỉ báo kỹ thuật...")
        result_df = standardize_technical_indicators(
            result_df,
            indicators_config=config.get('standardize_config'),
            window=config.get('standardize_window', 100),
            auto_detect=config.get('auto_detect', True)
        )
        logger.info("Đã hoàn thành standardize các chỉ báo")
    
    # Bước 3: Thêm trend_strength nếu cần
    if config.get('add_trend_strength', True):
        logger.info("Thêm chỉ báo độ mạnh xu hướng...")
        result_df = add_trend_strength_indicator(
            result_df,
            price_col=config.get('price_col', 'close'),
            ema_period=config.get('ema_period', 89),
            slope_period=config.get('slope_period', 5)
        )
        logger.info("Đã thêm chỉ báo độ mạnh xu hướng")
    
    # Bước 4: Thêm volatility_zscore nếu cần
    if config.get('add_volatility_zscore', True):
        logger.info("Thêm chỉ báo Z-score của biến động...")
        result_df = add_volatility_zscore(
            result_df,
            high_col=config.get('high_col', 'high'),
            low_col=config.get('low_col', 'low'),
            close_col=config.get('close_col', 'close'),
            atr_period=config.get('atr_period', 14),
            zscore_window=config.get('zscore_window', 50)
        )
        logger.info("Đã thêm chỉ báo Z-score của biến động")
    
    # Bước 5: Tạo nhãn nếu cần
    if config.get('generate_labels', False):
        logger.info("Tạo nhãn cho dữ liệu...")
        result_df = generate_labels(
            result_df,
            price_col=config.get('price_col', 'close'),
            window=config.get('label_window', 10),
            threshold=config.get('label_threshold', 0.01)
        )
        logger.info("Đã tạo nhãn cho dữ liệu")
    
    logger.info("Đã hoàn thành làm sạch và chuẩn hóa các chỉ báo kỹ thuật")
    return result_df

def add_trend_strength_indicator(
    df: pd.DataFrame,
    price_col: str = 'close',
    ema_period: int = 89,
    slope_period: int = 5,
    normalize: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Thêm chỉ báo độ mạnh xu hướng dựa trên độ dốc (slope) của EMA.
    
    Args:
        df: DataFrame chứa dữ liệu giá
        price_col: Tên cột giá
        ema_period: Kích thước cửa sổ cho EMA
        slope_period: Kích thước cửa sổ cho việc tính độ dốc
        normalize: Chuẩn hóa kết quả
        **kwargs: Tham số bổ sung
    
    Returns:
        DataFrame với các chỉ báo xu hướng mới
    """
    result_df = df.copy()
    
    # Kiểm tra xem cột giá tồn tại không
    if price_col not in result_df.columns:
        logger.warning(f"Cột giá {price_col} không tồn tại trong DataFrame")
        return result_df
    
    # Tính EMA
    ema_col = f"ema_{ema_period}"
    if ema_col not in result_df.columns:
        result_df[ema_col] = result_df[price_col].ewm(span=ema_period, adjust=False).mean()
    
    # Tính slope (độ dốc)
    slope = result_df[ema_col].diff(slope_period) / slope_period
    
    # Tính acceleration (gia tốc) - đạo hàm bậc 2
    acceleration = slope.diff()
    
    # Chuẩn hóa nếu được yêu cầu
    if normalize:
        # Sử dụng cửa sổ 100 period để chuẩn hóa
        window = kwargs.get('window', 100)
        
        # Đối với slope
        slope_rolling_std = slope.rolling(window=window, min_periods=1).std()
        slope_rolling_mean = slope.rolling(window=window, min_periods=1).mean()
        
        # Tránh chia cho 0
        slope_rolling_std = slope_rolling_std.replace(0, np.nan)
        
        # Z-score chuẩn hóa
        norm_slope = (slope - slope_rolling_mean) / slope_rolling_std
        
        # Xử lý các giá trị không hợp lệ
        norm_slope = norm_slope.replace([np.inf, -np.inf], np.nan)
        norm_slope = norm_slope.fillna(0)
        
        # Đối với acceleration
        acc_rolling_std = acceleration.rolling(window=window, min_periods=1).std()
        acc_rolling_mean = acceleration.rolling(window=window, min_periods=1).mean()
        
        # Tránh chia cho 0
        acc_rolling_std = acc_rolling_std.replace(0, np.nan)
        
        # Z-score chuẩn hóa
        norm_acceleration = (acceleration - acc_rolling_mean) / acc_rolling_std
        
        # Xử lý các giá trị không hợp lệ
        norm_acceleration = norm_acceleration.replace([np.inf, -np.inf], np.nan)
        norm_acceleration = norm_acceleration.fillna(0)
        
        # Đặt tên các cột kết quả
        result_df["trend_strength"] = norm_slope
        result_df["trend_acceleration"] = norm_acceleration
    else:
        # Đặt tên các cột kết quả không chuẩn hóa
        result_df["trend_slope"] = slope
        result_df["trend_acceleration"] = acceleration
    
    return result_df

def add_volatility_zscore(
    df: pd.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    atr_period: int = 14,
    zscore_window: int = 50,
    **kwargs
) -> pd.DataFrame:
    """
    Thêm chỉ báo Z-score của biến động (ATR).
    
    Args:
        df: DataFrame chứa dữ liệu giá
        high_col: Tên cột giá cao
        low_col: Tên cột giá thấp
        close_col: Tên cột giá đóng cửa
        atr_period: Kích thước cửa sổ cho ATR
        zscore_window: Kích thước cửa sổ cho việc tính Z-score
        **kwargs: Tham số bổ sung
    
    Returns:
        DataFrame với các chỉ báo biến động mới
    """
    result_df = df.copy()
    
    # Kiểm tra xem các cột tồn tại không
    required_cols = [high_col, low_col, close_col]
    missing_cols = [col for col in required_cols if col not in result_df.columns]
    if missing_cols:
        logger.warning(f"Các cột sau không tồn tại trong DataFrame: {missing_cols}")
        return result_df
    
    # Kiểm tra xem ATR đã được tính toán chưa
    atr_col = f"atr_{atr_period}"
    if atr_col not in result_df.columns:
        # Tính True Range
        high = result_df[high_col]
        low = result_df[low_col]
        close = result_df[close_col]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Tính ATR
        result_df[atr_col] = tr.ewm(span=atr_period, min_periods=atr_period, adjust=False).mean()
    
    # Tính ATR percent (ATR/Close)
    atr_pct_col = f"atr_pct_{atr_period}"
    if atr_pct_col not in result_df.columns:
        # Tránh chia cho 0
        close_non_zero = result_df[close_col].replace(0, np.nan)
        result_df[atr_pct_col] = result_df[atr_col] / close_non_zero * 100
    
    # Tính Z-score của ATR
    # Tính trung bình và độ lệch chuẩn của ATR trong cửa sổ
    rolling_mean = result_df[atr_col].rolling(window=zscore_window, min_periods=1).mean()
    rolling_std = result_df[atr_col].rolling(window=zscore_window, min_periods=1).std()
    
    # Tránh chia cho 0
    rolling_std = rolling_std.replace(0, np.nan)
    
    # Tính Z-score
    volatility_zscore = (result_df[atr_col] - rolling_mean) / rolling_std
    
    # Xử lý các giá trị không hợp lệ
    volatility_zscore = volatility_zscore.replace([np.inf, -np.inf], np.nan)
    volatility_zscore = volatility_zscore.fillna(0)
    
    # Đặt tên cột kết quả
    result_df["volatility_zscore"] = volatility_zscore
    
    # Tính biến động tương đối so với quá khứ (percentile)
    def rolling_rank(x):
        ranks = x.rank(pct=True)
        return ranks.iloc[-1] if not ranks.empty else np.nan
    
    volatility_rank = result_df[atr_col].rolling(window=zscore_window, min_periods=1).apply(
        rolling_rank, 
        raw=False  # Cần False để xử lý Series
    )
    
    # Đặt tên cột kết quả
    result_df["volatility_rank"] = volatility_rank
    
    return result_df