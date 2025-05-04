"""
Tiện ích kiểm tra và xác thực đặc trưng.
File này cung cấp các hàm để kiểm tra tính hợp lệ của các đặc trưng,
phát hiện dữ liệu bất thường, và đảm bảo chất lượng đặc trưng.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import warnings

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import setup_logger

# Logger
logger = setup_logger("feature_validation")

def validate_features(
    df: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
    check_nulls: bool = True,
    check_infinities: bool = True,
    check_range: bool = True,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    check_correlation: bool = False,
    correlation_threshold: float = 0.95,
    check_variance: bool = True,
    variance_threshold: float = 1e-8
) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Kiểm tra tính hợp lệ của các đặc trưng trong DataFrame.
    
    Args:
        df: DataFrame cần kiểm tra
        feature_list: Danh sách các đặc trưng cần kiểm tra (None để kiểm tra tất cả)
        check_nulls: Kiểm tra giá trị null
        check_infinities: Kiểm tra giá trị vô cùng
        check_range: Kiểm tra phạm vi giá trị
        min_val: Giá trị nhỏ nhất cho phép (None để không kiểm tra)
        max_val: Giá trị lớn nhất cho phép (None để không kiểm tra)
        check_correlation: Kiểm tra tương quan cao
        correlation_threshold: Ngưỡng tương quan
        check_variance: Kiểm tra phương sai nhỏ
        variance_threshold: Ngưỡng phương sai
    
    Returns:
        Tuple (is_valid, issues) với is_valid là True nếu hợp lệ, và issues là dict
        chứa danh sách vấn đề cho mỗi loại kiểm tra
    """
    # Nếu không cung cấp danh sách đặc trưng, sử dụng tất cả cột số
    if feature_list is None:
        feature_list = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Đảm bảo các đặc trưng tồn tại trong DataFrame
    existing_features = [f for f in feature_list if f in df.columns]
    missing_features = [f for f in feature_list if f not in df.columns]
    
    if missing_features:
        logger.warning(f"Các đặc trưng sau không tồn tại trong DataFrame: {missing_features}")
    
    if not existing_features:
        logger.error("Không có đặc trưng hợp lệ để kiểm tra")
        return False, {"missing_features": missing_features}
    
    # Khởi tạo dict lưu trữ vấn đề
    issues = {
        "null_values": [],
        "infinity_values": [],
        "out_of_range": [],
        "high_correlation": [],
        "low_variance": [],
        "missing_features": missing_features
    }
    
    # Kiểm tra giá trị null
    if check_nulls:
        for feature in existing_features:
            null_count = df[feature].isnull().sum()
            if null_count > 0:
                issues["null_values"].append(f"{feature} ({null_count} giá trị null)")
                logger.warning(f"Phát hiện {null_count} giá trị null trong đặc trưng {feature}")
    
    # Kiểm tra giá trị vô cùng
    if check_infinities:
        for feature in existing_features:
            inf_count = np.isinf(df[feature].to_numpy()).sum()
            if inf_count > 0:
                issues["infinity_values"].append(f"{feature} ({inf_count} giá trị vô cùng)")
                logger.warning(f"Phát hiện {inf_count} giá trị vô cùng trong đặc trưng {feature}")
    
    # Kiểm tra phạm vi giá trị
    if check_range and (min_val is not None or max_val is not None):
        for feature in existing_features:
            if min_val is not None:
                below_min = (df[feature] < min_val).sum()
                if below_min > 0:
                    issues["out_of_range"].append(f"{feature} ({below_min} giá trị < {min_val})")
                    logger.warning(f"Phát hiện {below_min} giá trị nhỏ hơn {min_val} trong đặc trưng {feature}")
            
            if max_val is not None:
                above_max = (df[feature] > max_val).sum()
                if above_max > 0:
                    issues["out_of_range"].append(f"{feature} ({above_max} giá trị > {max_val})")
                    logger.warning(f"Phát hiện {above_max} giá trị lớn hơn {max_val} trong đặc trưng {feature}")
    
    # Kiểm tra tương quan cao
    if check_correlation and len(existing_features) > 1:
        try:
            correlation_matrix = df[existing_features].corr()
            
            # Tạo mặt nạ cho tam giác trên (không bao gồm đường chéo)
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            
            # Lấy các cặp có tương quan cao
            high_corr_pairs = []
            for i, row in enumerate(correlation_matrix.index):
                for j, col in enumerate(correlation_matrix.columns):
                    if mask[i, j] and abs(correlation_matrix.iloc[i, j]) >= correlation_threshold:
                        high_corr_pairs.append(f"{row}-{col} ({correlation_matrix.iloc[i, j]:.2f})")
            
            issues["high_correlation"] = high_corr_pairs
            
            if high_corr_pairs:
                logger.warning(f"Phát hiện {len(high_corr_pairs)} cặp đặc trưng có tương quan cao (>= {correlation_threshold})")
                
        except Exception as e:
            logger.error(f"Lỗi khi tính toán ma trận tương quan: {e}")
    
    # Kiểm tra phương sai nhỏ
    if check_variance:
        for feature in existing_features:
            variance = df[feature].var()
            if variance < variance_threshold:
                issues["low_variance"].append(f"{feature} (variance = {variance:.8f})")
                logger.warning(f"Phát hiện đặc trưng {feature} có phương sai nhỏ ({variance:.8f})")
    
    # Kiểm tra tất cả các vấn đề
    all_issues = []
    for issue_type, issue_list in issues.items():
        if issue_type != "missing_features" and issue_list:
            all_issues.extend(issue_list)
    
    is_valid = len(all_issues) == 0
    
    if is_valid:
        logger.info(f"Tất cả {len(existing_features)} đặc trưng đều hợp lệ")
    else:
        logger.warning(f"Phát hiện {len(all_issues)} vấn đề trong {len(existing_features)} đặc trưng")
    
    return is_valid, issues

def check_feature_integrity(
    df: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
    reference_df: Optional[pd.DataFrame] = None,
    check_dtypes: bool = True,
    check_stats: bool = True,
    stats_tolerance: float = 0.1,
    check_missing: bool = True
) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Kiểm tra tính toàn vẹn của các đặc trưng bằng cách so sánh với reference_df nếu có,
    hoặc kiểm tra cấu trúc tổng quát nếu không có reference_df.
    
    Args:
        df: DataFrame cần kiểm tra
        feature_list: Danh sách các đặc trưng cần kiểm tra (None để kiểm tra tất cả)
        reference_df: DataFrame tham chiếu để so sánh (None để tự kiểm tra)
        check_dtypes: Kiểm tra kiểu dữ liệu
        check_stats: Kiểm tra thống kê (min, max, mean, std)
        stats_tolerance: Dung sai cho sự khác biệt thống kê
        check_missing: Kiểm tra tỷ lệ giá trị thiếu
    
    Returns:
        Tuple (is_intact, issues) với is_intact là True nếu toàn vẹn, và issues là dict
        chứa danh sách vấn đề cho mỗi loại kiểm tra
    """
    # Nếu không cung cấp danh sách đặc trưng, sử dụng tất cả cột số
    if feature_list is None:
        feature_list = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Đảm bảo các đặc trưng tồn tại trong DataFrame
    existing_features = [f for f in feature_list if f in df.columns]
    missing_features = [f for f in feature_list if f not in df.columns]
    
    if missing_features:
        logger.warning(f"Các đặc trưng sau không tồn tại trong DataFrame: {missing_features}")
    
    if not existing_features:
        logger.error("Không có đặc trưng hợp lệ để kiểm tra")
        return False, {"missing_features": missing_features}
    
    # Khởi tạo dict lưu trữ vấn đề
    issues = {
        "missing_features": missing_features,
        "dtype_mismatch": [],
        "stat_mismatch": [],
        "missing_rate_change": []
    }
    
    # Phần 1: Kiểm tra khi có reference_df
    if reference_df is not None:
        # Kiểm tra xem các đặc trưng có trong reference_df không
        ref_missing_features = [f for f in existing_features if f not in reference_df.columns]
        if ref_missing_features:
            logger.warning(f"Các đặc trưng sau không tồn tại trong reference_df: {ref_missing_features}")
            issues["ref_missing_features"] = ref_missing_features
        
        # Lọc các đặc trưng tồn tại trong cả hai DataFrame
        common_features = [f for f in existing_features if f in reference_df.columns]
        
        if not common_features:
            logger.error("Không có đặc trưng chung giữa hai DataFrame để so sánh")
            return False, issues
        
        # Kiểm tra kiểu dữ liệu
        if check_dtypes:
            for feature in common_features:
                if df[feature].dtype != reference_df[feature].dtype:
                    issue = f"{feature} (df: {df[feature].dtype}, ref: {reference_df[feature].dtype})"
                    issues["dtype_mismatch"].append(issue)
                    logger.warning(f"Kiểu dữ liệu không khớp cho {issue}")
        
        # Kiểm tra thống kê
        if check_stats:
            for feature in common_features:
                try:
                    # Tính các thống kê
                    df_stats = {
                        "min": df[feature].min(),
                        "max": df[feature].max(),
                        "mean": df[feature].mean(),
                        "std": df[feature].std()
                    }
                    
                    ref_stats = {
                        "min": reference_df[feature].min(),
                        "max": reference_df[feature].max(),
                        "mean": reference_df[feature].mean(),
                        "std": reference_df[feature].std()
                    }
                    
                    # So sánh các thống kê
                    stat_issues = []
                    for stat_name, df_value in df_stats.items():
                        ref_value = ref_stats[stat_name]
                        
                        # Tránh chia cho 0
                        if abs(ref_value) < 1e-10:
                            if abs(df_value) > stats_tolerance:
                                stat_issues.append(f"{stat_name}: {df_value} vs {ref_value}")
                        else:
                            # Tính phần trăm thay đổi
                            pct_change = abs((df_value - ref_value) / ref_value)
                            if pct_change > stats_tolerance:
                                stat_issues.append(f"{stat_name}: {df_value:.4f} vs {ref_value:.4f} ({pct_change:.2%})")
                    
                    if stat_issues:
                        issues["stat_mismatch"].append(f"{feature} ({', '.join(stat_issues)})")
                        logger.warning(f"Thống kê không khớp cho {feature}: {', '.join(stat_issues)}")
                        
                except Exception as e:
                    logger.error(f"Lỗi khi so sánh thống kê cho đặc trưng {feature}: {e}")
        
        # Kiểm tra tỷ lệ giá trị thiếu
        if check_missing:
            for feature in common_features:
                df_missing_rate = df[feature].isnull().mean()
                ref_missing_rate = reference_df[feature].isnull().mean()
                
                # Tính sự khác biệt tuyệt đối
                missing_diff = abs(df_missing_rate - ref_missing_rate)
                
                if missing_diff > stats_tolerance:
                    issue = f"{feature} (df: {df_missing_rate:.2%}, ref: {ref_missing_rate:.2%}, diff: {missing_diff:.2%})"
                    issues["missing_rate_change"].append(issue)
                    logger.warning(f"Tỷ lệ giá trị thiếu thay đổi lớn cho {issue}")
    
    # Phần 2: Tự kiểm tra khi không có reference_df
    else:
        # Kiểm tra các đặc trưng có giá trị tiêu chuẩn hay không
        for feature in existing_features:
            # Kiểm tra tỷ lệ giá trị thiếu
            if check_missing:
                missing_rate = df[feature].isnull().mean()
                if missing_rate > 0.2:  # Ngưỡng 20% giá trị thiếu
                    issues["missing_rate_change"].append(f"{feature} ({missing_rate:.2%} giá trị thiếu)")
                    logger.warning(f"Đặc trưng {feature} có {missing_rate:.2%} giá trị thiếu")
            
            # Kiểm tra phân phối (outlier, skewness)
            if check_stats:
                try:
                    # Tính các thống kê cơ bản
                    mean = df[feature].mean()
                    std = df[feature].std()
                    
                    # Kiểm tra outlier
                    z_scores = np.abs((df[feature] - mean) / std)
                    outlier_count = (z_scores > 3).sum()  # Z-score > 3
                    outlier_rate = outlier_count / len(df)
                    
                    if outlier_rate > 0.05:  # Ngưỡng 5% outlier
                        if "outliers" not in issues:
                            issues["outliers"] = []
                        issues["outliers"].append(f"{feature} ({outlier_count} outliers, {outlier_rate:.2%})")
                        logger.warning(f"Đặc trưng {feature} có {outlier_count} outliers ({outlier_rate:.2%})")
                
                except Exception as e:
                    logger.error(f"Lỗi khi kiểm tra phân phối cho đặc trưng {feature}: {e}")
    
    # Kiểm tra tất cả các vấn đề
    all_issues = []
    for issue_type, issue_list in issues.items():
        if issue_list:
            all_issues.extend(issue_list)
    
    is_intact = len(all_issues) == 0 or (len(all_issues) == len(issues.get("missing_features", [])))
    
    if is_intact:
        logger.info(f"Tất cả {len(existing_features)} đặc trưng đều toàn vẹn")
    else:
        logger.warning(f"Phát hiện {len(all_issues) - len(issues.get('missing_features', []))} vấn đề trong {len(existing_features)} đặc trưng")
    
    return is_intact, issues

def identify_redundant_features(
    df: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
    correlation_threshold: float = 0.95,
    variance_threshold: float = 1e-8
) -> Dict[str, List[str]]:
    """
    Xác định các đặc trưng dư thừa dựa trên tương quan và phương sai.
    
    Args:
        df: DataFrame cần kiểm tra
        feature_list: Danh sách các đặc trưng cần kiểm tra (None để kiểm tra tất cả)
        correlation_threshold: Ngưỡng tương quan
        variance_threshold: Ngưỡng phương sai
    
    Returns:
        Dict chứa danh sách đặc trưng dư thừa theo loại
    """
    # Nếu không cung cấp danh sách đặc trưng, sử dụng tất cả cột số
    if feature_list is None:
        feature_list = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Đảm bảo các đặc trưng tồn tại trong DataFrame
    existing_features = [f for f in feature_list if f in df.columns]
    
    if not existing_features:
        logger.error("Không có đặc trưng hợp lệ để kiểm tra")
        return {"zero_variance": [], "high_correlation": []}
    
    # Khởi tạo dict lưu trữ đặc trưng dư thừa
    redundant_features = {
        "zero_variance": [],
        "high_correlation": []
    }
    
    # Kiểm tra phương sai nhỏ
    for feature in existing_features:
        variance = df[feature].var()
        if variance < variance_threshold:
            redundant_features["zero_variance"].append(feature)
            logger.info(f"Đặc trưng {feature} có phương sai nhỏ ({variance:.8f})")
    
    # Kiểm tra tương quan cao
    if len(existing_features) > 1:
        try:
            # Tính ma trận tương quan
            correlation_matrix = df[existing_features].corr().abs()
            
            # Tạo mặt nạ cho tam giác trên (không bao gồm đường chéo)
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            
            # Tìm các cặp đặc trưng có tương quan cao
            high_corr_pairs = []
            for i, row in enumerate(correlation_matrix.index):
                for j, col in enumerate(correlation_matrix.columns):
                    if mask[i, j] and correlation_matrix.iloc[i, j] >= correlation_threshold:
                        high_corr_pairs.append((row, col, correlation_matrix.iloc[i, j]))
            
            # Đánh giá đặc trưng nào nên loại bỏ
            feature_scores = {feature: 0 for feature in existing_features}
            
            for f1, f2, corr in high_corr_pairs:
                # Tính điểm cho mỗi đặc trưng (số lần xuất hiện trong các cặp tương quan cao)
                feature_scores[f1] += 1
                feature_scores[f2] += 1
            
            # Lọc các đặc trưng có tương quan cao với nhiều đặc trưng khác
            for feature, score in feature_scores.items():
                if score > 0:
                    redundant_features["high_correlation"].append(f"{feature} (xuất hiện trong {score} cặp tương quan cao)")
                    logger.info(f"Đặc trưng {feature} có tương quan cao với {score} đặc trưng khác")
            
        except Exception as e:
            logger.error(f"Lỗi khi tính toán ma trận tương quan: {e}")
    
    return redundant_features

def check_feature_stability(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    feature_list: Optional[List[str]] = None,
    distribution_threshold: float = 0.1,
    use_ks_test: bool = False,
    alpha: float = 0.05
) -> Dict[str, List[str]]:
    """
    Kiểm tra tính ổn định của các đặc trưng giữa hai tập dữ liệu.
    
    Args:
        df1: DataFrame thứ nhất
        df2: DataFrame thứ hai
        feature_list: Danh sách các đặc trưng cần kiểm tra (None để kiểm tra tất cả)
        distribution_threshold: Ngưỡng cho sự thay đổi phân phối
        use_ks_test: Sử dụng Kolmogorov-Smirnov test
        alpha: Mức ý nghĩa cho KS test
    
    Returns:
        Dict chứa danh sách đặc trưng không ổn định theo loại
    """
    # Nếu không cung cấp danh sách đặc trưng, sử dụng các đặc trưng chung
    if feature_list is None:
        df1_cols = df1.select_dtypes(include=[np.number]).columns.tolist()
        df2_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
        feature_list = list(set(df1_cols) & set(df2_cols))
    
    # Đảm bảo các đặc trưng tồn tại trong cả hai DataFrame
    common_features = [f for f in feature_list if f in df1.columns and f in df2.columns]
    missing_features1 = [f for f in feature_list if f not in df1.columns]
    missing_features2 = [f for f in feature_list if f not in df2.columns]
    
    if missing_features1:
        logger.warning(f"Các đặc trưng sau không tồn tại trong DataFrame 1: {missing_features1}")
    
    if missing_features2:
        logger.warning(f"Các đặc trưng sau không tồn tại trong DataFrame 2: {missing_features2}")
    
    if not common_features:
        logger.error("Không có đặc trưng chung giữa hai DataFrame để kiểm tra")
        return {
            "missing_features1": missing_features1,
            "missing_features2": missing_features2,
            "unstable_features": []
        }
    
    # Khởi tạo dict lưu trữ đặc trưng không ổn định
    unstable_features = {
        "missing_features1": missing_features1,
        "missing_features2": missing_features2,
        "unstable_features": []
    }
    
    for feature in common_features:
        try:
            # Lấy chuỗi giá trị không null
            values1 = df1[feature].dropna().values
            values2 = df2[feature].dropna().values
            
            if len(values1) == 0 or len(values2) == 0:
                logger.warning(f"Đặc trưng {feature} có tất cả giá trị là null trong ít nhất một DataFrame")
                unstable_features["unstable_features"].append(f"{feature} (tất cả giá trị là null)")
                continue
            
            # Kiểm tra phân phối
            if use_ks_test:
                # Sử dụng Kolmogorov-Smirnov test
                try:
                    from scipy import stats
                    ks_stat, p_value = stats.ks_2samp(values1, values2)
                    
                    if p_value < alpha:
                        unstable_features["unstable_features"].append(f"{feature} (KS test: p-value = {p_value:.4f})")
                        logger.warning(f"Đặc trưng {feature} không ổn định theo KS test (p-value = {p_value:.4f})")
                except ImportError:
                    logger.warning("Không thể sử dụng KS test (scipy chưa được cài đặt)")
                    use_ks_test = False
            
            if not use_ks_test:
                # So sánh các thống kê cơ bản
                stats1 = {
                    "mean": np.mean(values1),
                    "std": np.std(values1),
                    "min": np.min(values1),
                    "max": np.max(values1),
                    "median": np.median(values1)
                }
                
                stats2 = {
                    "mean": np.mean(values2),
                    "std": np.std(values2),
                    "min": np.min(values2),
                    "max": np.max(values2),
                    "median": np.median(values2)
                }
                
                # Kiểm tra sự thay đổi
                unstable_stats = []
                for stat_name in ["mean", "median", "std"]:
                    val1 = stats1[stat_name]
                    val2 = stats2[stat_name]
                    
                    # Tránh chia cho 0
                    if abs(val1) < 1e-10:
                        if abs(val2) > distribution_threshold:
                            unstable_stats.append(f"{stat_name}: {val1:.4f} vs {val2:.4f}")
                    else:
                        # Tính phần trăm thay đổi
                        pct_change = abs((val2 - val1) / val1)
                        if pct_change > distribution_threshold:
                            unstable_stats.append(f"{stat_name}: {val1:.4f} vs {val2:.4f} ({pct_change:.2%})")
                
                if unstable_stats:
                    unstable_features["unstable_features"].append(f"{feature} ({', '.join(unstable_stats)})")
                    logger.warning(f"Đặc trưng {feature} không ổn định: {', '.join(unstable_stats)}")
        
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra tính ổn định cho đặc trưng {feature}: {e}")
    
    return unstable_features