"""
Phương pháp lựa chọn đặc trưng dựa trên mức độ quan trọng.
Module này cung cấp các phương pháp sử dụng các mô hình cây quyết định,
rừng ngẫu nhiên, boosting và SHAP để đánh giá mức độ quan trọng của đặc trưng.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import logging
import warnings
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import get_logger

# Thiết lập logger
logger = get_logger("feature_selector")

def tree_importance_selector(
    df: pd.DataFrame,
    target_column: str,
    k: Optional[int] = None,
    threshold: float = 0.0,
    problem_type: str = 'auto',
    params: Dict[str, Any] = None,
    exclude_columns: List[str] = []
) -> List[str]:
    """
    Lựa chọn đặc trưng dựa trên mức độ quan trọng từ mô hình cây quyết định.
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu
        k: Số lượng đặc trưng cần chọn (None để chọn theo ngưỡng)
        threshold: Ngưỡng tối thiểu của mức độ quan trọng để chọn đặc trưng
        problem_type: Loại bài toán ('classification', 'regression', 'auto')
        params: Tham số cho mô hình cây quyết định
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
            logger.warning("Không có cột số nào để tính mức độ quan trọng")
            return []
        
        X = feature_df[numeric_cols]
        y = df[target_column]
        
        # Xác định loại bài toán
        if problem_type == 'auto':
            # Tự động phát hiện loại bài toán
            is_categorical = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)
            is_discrete_numeric = pd.api.types.is_numeric_dtype(y) and len(y.unique()) <= 20
            problem_type = 'classification' if (is_categorical or is_discrete_numeric) else 'regression'
        
        # Các tham số mặc định
        default_params = {
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        }
        
        # Kết hợp với tham số đã cung cấp (nếu có)
        if params:
            default_params.update(params)
        
        # Khởi tạo mô hình cây quyết định
        if problem_type == 'classification':
            model = DecisionTreeClassifier(**default_params)
        else:
            model = DecisionTreeRegressor(**default_params)
        
        # Huấn luyện mô hình
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        
        # Lấy mức độ quan trọng của đặc trưng
        importance_scores = model.feature_importances_
        
        # Tạo DataFrame kết quả
        result_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance': importance_scores
        })
        
        # Sắp xếp theo mức độ quan trọng giảm dần
        result_df = result_df.sort_values('importance', ascending=False)
        
        # Chọn đặc trưng
        if k is not None:
            # Chọn top-k đặc trưng
            k = min(k, len(numeric_cols))
            selected_features = result_df['feature'].tolist()[:k]
        else:
            # Chọn theo ngưỡng
            selected_features = result_df[result_df['importance'] >= threshold]['feature'].tolist()
        
        logger.info(f"Đã chọn {len(selected_features)} đặc trưng dựa trên mức độ quan trọng từ mô hình cây quyết định")
        return selected_features
    
    except Exception as e:
        logger.error(f"Lỗi khi chọn đặc trưng dựa trên mô hình cây quyết định: {str(e)}")
        return []

def random_forest_importance_selector(
    df: pd.DataFrame,
    target_column: str,
    k: Optional[int] = None,
    threshold: float = 0.0,
    problem_type: str = 'auto',
    params: Dict[str, Any] = None,
    exclude_columns: List[str] = []
) -> List[str]:
    """
    Lựa chọn đặc trưng dựa trên mức độ quan trọng từ mô hình rừng ngẫu nhiên.
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu
        k: Số lượng đặc trưng cần chọn (None để chọn theo ngưỡng)
        threshold: Ngưỡng tối thiểu của mức độ quan trọng để chọn đặc trưng
        problem_type: Loại bài toán ('classification', 'regression', 'auto')
        params: Tham số cho mô hình rừng ngẫu nhiên
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
            logger.warning("Không có cột số nào để tính mức độ quan trọng")
            return []
        
        X = feature_df[numeric_cols]
        y = df[target_column]
        
        # Xác định loại bài toán
        if problem_type == 'auto':
            # Tự động phát hiện loại bài toán
            is_categorical = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)
            is_discrete_numeric = pd.api.types.is_numeric_dtype(y) and len(y.unique()) <= 20
            problem_type = 'classification' if (is_categorical or is_discrete_numeric) else 'regression'
        
        # Các tham số mặc định
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Kết hợp với tham số đã cung cấp (nếu có)
        if params:
            default_params.update(params)
        
        # Khởi tạo mô hình rừng ngẫu nhiên
        if problem_type == 'classification':
            model = RandomForestClassifier(**default_params)
        else:
            model = RandomForestRegressor(**default_params)
        
        # Huấn luyện mô hình
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        
        # Lấy mức độ quan trọng của đặc trưng
        importance_scores = model.feature_importances_
        
        # Tạo DataFrame kết quả
        result_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance': importance_scores
        })
        
        # Sắp xếp theo mức độ quan trọng giảm dần
        result_df = result_df.sort_values('importance', ascending=False)
        
        # Chọn đặc trưng
        if k is not None:
            # Chọn top-k đặc trưng
            k = min(k, len(numeric_cols))
            selected_features = result_df['feature'].tolist()[:k]
        else:
            # Chọn theo ngưỡng
            selected_features = result_df[result_df['importance'] >= threshold]['feature'].tolist()
        
        logger.info(f"Đã chọn {len(selected_features)} đặc trưng dựa trên mức độ quan trọng từ mô hình rừng ngẫu nhiên")
        return selected_features
    
    except Exception as e:
        logger.error(f"Lỗi khi chọn đặc trưng dựa trên mô hình rừng ngẫu nhiên: {str(e)}")
        return []

def boosting_importance_selector(
    df: pd.DataFrame,
    target_column: str,
    k: Optional[int] = None,
    threshold: float = 0.0,
    problem_type: str = 'auto',
    params: Dict[str, Any] = None,
    exclude_columns: List[str] = []
) -> List[str]:
    """
    Lựa chọn đặc trưng dựa trên mức độ quan trọng từ mô hình Gradient Boosting.
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu
        k: Số lượng đặc trưng cần chọn (None để chọn theo ngưỡng)
        threshold: Ngưỡng tối thiểu của mức độ quan trọng để chọn đặc trưng
        problem_type: Loại bài toán ('classification', 'regression', 'auto')
        params: Tham số cho mô hình Gradient Boosting
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
            logger.warning("Không có cột số nào để tính mức độ quan trọng")
            return []
        
        X = feature_df[numeric_cols]
        y = df[target_column]
        
        # Xác định loại bài toán
        if problem_type == 'auto':
            # Tự động phát hiện loại bài toán
            is_categorical = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)
            is_discrete_numeric = pd.api.types.is_numeric_dtype(y) and len(y.unique()) <= 20
            problem_type = 'classification' if (is_categorical or is_discrete_numeric) else 'regression'
        
        # Các tham số mặc định
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'random_state': 42
        }
        
        # Kết hợp với tham số đã cung cấp (nếu có)
        if params:
            default_params.update(params)
        
        # Khởi tạo mô hình Gradient Boosting
        if problem_type == 'classification':
            model = GradientBoostingClassifier(**default_params)
        else:
            model = GradientBoostingRegressor(**default_params)
        
        # Huấn luyện mô hình
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        
        # Lấy mức độ quan trọng của đặc trưng
        importance_scores = model.feature_importances_
        
        # Tạo DataFrame kết quả
        result_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance': importance_scores
        })
        
        # Sắp xếp theo mức độ quan trọng giảm dần
        result_df = result_df.sort_values('importance', ascending=False)
        
        # Chọn đặc trưng
        if k is not None:
            # Chọn top-k đặc trưng
            k = min(k, len(numeric_cols))
            selected_features = result_df['feature'].tolist()[:k]
        else:
            # Chọn theo ngưỡng
            selected_features = result_df[result_df['importance'] >= threshold]['feature'].tolist()
        
        logger.info(f"Đã chọn {len(selected_features)} đặc trưng dựa trên mức độ quan trọng từ mô hình Gradient Boosting")
        return selected_features
    
    except Exception as e:
        logger.error(f"Lỗi khi chọn đặc trưng dựa trên mô hình Gradient Boosting: {str(e)}")
        return []

def shap_importance_selector(
    df: pd.DataFrame,
    target_column: str,
    k: Optional[int] = None,
    threshold: float = 0.0,
    problem_type: str = 'auto',
    model_type: str = 'tree',
    params: Dict[str, Any] = None,
    exclude_columns: List[str] = []
) -> List[str]:
    """
    Lựa chọn đặc trưng dựa trên giá trị SHAP (SHapley Additive exPlanations).
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu
        k: Số lượng đặc trưng cần chọn (None để chọn theo ngưỡng)
        threshold: Ngưỡng tối thiểu của giá trị SHAP để chọn đặc trưng
        problem_type: Loại bài toán ('classification', 'regression', 'auto')
        model_type: Loại mô hình sử dụng để tính SHAP ('tree', 'linear', 'kernel')
        params: Tham số cho mô hình
        exclude_columns: Danh sách cột cần loại trừ khỏi việc lựa chọn
        
    Returns:
        Danh sách tên các cột đặc trưng được chọn
    """
    try:
        # Kiểm tra xem có thư viện shap không
        try:
            import shap
        except ImportError:
            logger.error("Thư viện SHAP chưa được cài đặt. Vui lòng cài đặt với 'pip install shap'")
            # Fallback sang random forest nếu không có SHAP
            logger.info("Sử dụng Random Forest Importance thay thế cho SHAP")
            return random_forest_importance_selector(df, target_column, k, threshold, problem_type, params, exclude_columns)
        
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
            logger.warning("Không có cột số nào để tính giá trị SHAP")
            return []
        
        X = feature_df[numeric_cols]
        y = df[target_column]
        
        # Xác định loại bài toán
        if problem_type == 'auto':
            # Tự động phát hiện loại bài toán
            is_categorical = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)
            is_discrete_numeric = pd.api.types.is_numeric_dtype(y) and len(y.unique()) <= 20
            problem_type = 'classification' if (is_categorical or is_discrete_numeric) else 'regression'
        
        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Khởi tạo và huấn luyện mô hình phù hợp
        if model_type == 'tree':
            # Sử dụng Random Forest
            if problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            
            # Tính toán giá trị SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Lấy giá trị tuyệt đối trung bình của SHAP cho mỗi đặc trưng
            if isinstance(shap_values, list):  # Cho bài toán phân loại đa lớp
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)
                shap_importance = np.abs(shap_values).mean(axis=0)
            else:
                shap_importance = np.abs(shap_values).mean(axis=0)
            
        elif model_type == 'linear':
            from sklearn.linear_model import Ridge, LogisticRegression
            
            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Sử dụng Linear Model
            if problem_type == 'classification':
                model = LogisticRegression(random_state=42)
            else:
                model = Ridge(random_state=42)
            
            # Huấn luyện mô hình
            model.fit(X_train_scaled, y_train)
            
            # Tính toán giá trị SHAP
            explainer = shap.LinearExplainer(model, X_train_scaled)
            shap_values = explainer.shap_values(X_test_scaled)
            
            # Lấy giá trị tuyệt đối trung bình của SHAP cho mỗi đặc trưng
            shap_importance = np.abs(shap_values).mean(axis=0)
            
        else:  # kernel
            # Sử dụng Kernel SHAP (chậm hơn nhưng đa năng)
            # Chỉ lấy một phần nhỏ của tập dữ liệu để tính SHAP
            max_samples = min(100, len(X_train))
            background = shap.sample(X_train, max_samples)
            
            # Sử dụng Random Forest làm mô hình nền
            if problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            
            # Chọn một số mẫu để tính SHAP
            test_samples = min(50, len(X_test))
            test_indices = np.random.choice(len(X_test), test_samples, replace=False)
            
            # Tính toán giá trị SHAP
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_test.iloc[test_indices])
            
            # Lấy giá trị tuyệt đối trung bình của SHAP cho mỗi đặc trưng
            if isinstance(shap_values, list):  # Cho bài toán phân loại đa lớp
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)
                shap_importance = np.abs(shap_values).mean(axis=0)
            else:
                shap_importance = np.abs(shap_values).mean(axis=0)
        
        # Tạo DataFrame kết quả
        result_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance': shap_importance
        })
        
        # Sắp xếp theo mức độ quan trọng giảm dần
        result_df = result_df.sort_values('importance', ascending=False)
        
        # Chọn đặc trưng
        if k is not None:
            # Chọn top-k đặc trưng
            k = min(k, len(numeric_cols))
            selected_features = result_df['feature'].tolist()[:k]
        else:
            # Chọn theo ngưỡng
            selected_features = result_df[result_df['importance'] >= threshold]['feature'].tolist()
        
        logger.info(f"Đã chọn {len(selected_features)} đặc trưng dựa trên giá trị SHAP")
        return selected_features
    
    except Exception as e:
        logger.error(f"Lỗi khi chọn đặc trưng dựa trên giá trị SHAP: {str(e)}")
        # Fallback sang random forest nếu có lỗi với SHAP
        logger.info("Sử dụng Random Forest Importance thay thế cho SHAP")
        return random_forest_importance_selector(df, target_column, k, threshold, problem_type, params, exclude_columns)

def permutation_importance_selector(
    df: pd.DataFrame,
    target_column: str,
    k: Optional[int] = None,
    threshold: float = 0.0,
    problem_type: str = 'auto',
    model_type: str = 'random_forest',
    n_repeats: int = 10,
    params: Dict[str, Any] = None,
    exclude_columns: List[str] = []
) -> List[str]:
    """
    Lựa chọn đặc trưng dựa trên Permutation Importance.
    
    Args:
        df: DataFrame chứa các đặc trưng và biến mục tiêu
        target_column: Tên cột mục tiêu
        k: Số lượng đặc trưng cần chọn (None để chọn theo ngưỡng)
        threshold: Ngưỡng tối thiểu của mức độ quan trọng để chọn đặc trưng
        problem_type: Loại bài toán ('classification', 'regression', 'auto')
        model_type: Loại mô hình sử dụng ('random_forest', 'gradient_boosting', 'decision_tree')
        n_repeats: Số lần lặp lại để tính Permutation Importance
        params: Tham số cho mô hình
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
            logger.warning("Không có cột số nào để tính Permutation Importance")
            return []
        
        X = feature_df[numeric_cols]
        y = df[target_column]
        
        # Xác định loại bài toán
        if problem_type == 'auto':
            # Tự động phát hiện loại bài toán
            is_categorical = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)
            is_discrete_numeric = pd.api.types.is_numeric_dtype(y) and len(y.unique()) <= 20
            problem_type = 'classification' if (is_categorical or is_discrete_numeric) else 'regression'
        
        # Các tham số mặc định
        default_params = {
            'random_state': 42
        }
        
        # Kết hợp với tham số đã cung cấp (nếu có)
        if params:
            default_params.update(params)
        
        # Khởi tạo mô hình phù hợp
        if model_type == 'random_forest':
            if problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, **default_params)
            else:
                model = RandomForestRegressor(n_estimators=100, **default_params)
        elif model_type == 'gradient_boosting':
            if problem_type == 'classification':
                model = GradientBoostingClassifier(n_estimators=100, **default_params)
            else:
                model = GradientBoostingRegressor(n_estimators=100, **default_params)
        else:  # decision_tree
            if problem_type == 'classification':
                model = DecisionTreeClassifier(**default_params)
            else:
                model = DecisionTreeRegressor(**default_params)
        
        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Huấn luyện mô hình
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        
        # Tính toán Permutation Importance
        perm_importance = permutation_importance(
            model, X_test, y_test, n_repeats=n_repeats, random_state=42
        )
        
        # Lấy giá trị trung bình của Permutation Importance
        importance_scores = perm_importance.importances_mean
        
        # Tạo DataFrame kết quả
        result_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance': importance_scores
        })
        
        # Sắp xếp theo mức độ quan trọng giảm dần
        result_df = result_df.sort_values('importance', ascending=False)
        
        # Chọn đặc trưng
        if k is not None:
            # Chọn top-k đặc trưng
            k = min(k, len(numeric_cols))
            selected_features = result_df['feature'].tolist()[:k]
        else:
            # Chọn theo ngưỡng
            selected_features = result_df[result_df['importance'] >= threshold]['feature'].tolist()
        
        logger.info(f"Đã chọn {len(selected_features)} đặc trưng dựa trên Permutation Importance")
        return selected_features
    
    except Exception as e:
        logger.error(f"Lỗi khi chọn đặc trưng dựa trên Permutation Importance: {str(e)}")
        return []