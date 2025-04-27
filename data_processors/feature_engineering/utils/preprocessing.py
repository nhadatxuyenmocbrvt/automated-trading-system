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