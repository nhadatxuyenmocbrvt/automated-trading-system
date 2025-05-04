"""
Phát hiện và xử lý ngoại lệ trong dữ liệu thị trường.
File này cung cấp các lớp và phương thức để phát hiện và xử lý các giá trị
ngoại lệ trong dữ liệu thị trường sử dụng nhiều phương pháp khác nhau.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from scipy import stats

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import setup_logger

class OutlierDetector:
    """
    Lớp để phát hiện và xử lý các giá trị ngoại lệ trong dữ liệu.
    """
    
    def __init__(
        self,
        method: str = 'z-score',
        threshold: float = 5.0,
        use_robust: bool = False,
        contamination: float = 0.05,
        eps: float = 0.5,
        min_samples: int = 5,
        n_neighbors: int = 20,
        kernel: str = 'rbf',
        nu: float = 0.01
    ):
        """
        Khởi tạo outlier detector.
        
        Args:
            method: Phương pháp phát hiện ngoại lệ ('z-score', 'iqr', 'isolation-forest', 
                  'dbscan', 'lof', 'one-class-svm')
            threshold: Ngưỡng để xác định ngoại lệ (cho z-score và iqr)
            use_robust: Sử dụng phương pháp robust (median, MAD) thay vì mean, std
            contamination: Tỷ lệ ngoại lệ dự kiến (dùng cho Isolation Forest và LOF)
            eps: Khoảng cách tối đa giữa các điểm trong DBSCAN
            min_samples: Số mẫu tối thiểu trong vùng lân cận cho DBSCAN
            n_neighbors: Số lượng hàng xóm cho LOF
            kernel: Kernel function cho One-Class SVM ('linear', 'poly', 'rbf', 'sigmoid')
            nu: Tham số nu cho One-Class SVM (upper bound trên tỷ lệ outlier)
        """
        self.logger = setup_logger("outlier_detector")
        
        self.method = method
        self.threshold = threshold
        self.use_robust = use_robust
        self.contamination = contamination
        self.eps = eps
        self.min_samples = min_samples
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.nu = nu
        
        self.models = {}  # Dictionary để lưu các model cho từng cột khi sử dụng per_column=True
        
        # Khởi tạo các mô hình phát hiện ngoại lệ nếu cần
        if method == 'isolation-forest':
            try:
                from sklearn.ensemble import IsolationForest
                self.model = IsolationForest(contamination=contamination, random_state=42)
            except ImportError:
                self.logger.warning("Không thể import sklearn. Chuyển sang sử dụng phương pháp z-score.")
                self.method = 'z-score'
        
        elif method == 'dbscan':
            try:
                from sklearn.cluster import DBSCAN
                self.model = DBSCAN(eps=eps, min_samples=min_samples)
            except ImportError:
                self.logger.warning("Không thể import sklearn. Chuyển sang sử dụng phương pháp z-score.")
                self.method = 'z-score'
        
        elif method == 'lof':
            try:
                from sklearn.neighbors import LocalOutlierFactor
                self.model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            except ImportError:
                self.logger.warning("Không thể import sklearn. Chuyển sang sử dụng phương pháp z-score.")
                self.method = 'z-score'
                
        elif method == 'one-class-svm':
            try:
                from sklearn import svm
                self.model = svm.OneClassSVM(kernel=kernel, nu=nu)
            except ImportError:
                self.logger.warning("Không thể import sklearn. Chuyển sang sử dụng phương pháp z-score.")
                self.method = 'z-score'
        
        self.logger.info(f"Đã khởi tạo OutlierDetector với phương pháp {self.method}")
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        per_column: bool = False,
        outlier_suffix: str = '_is_outlier',
        global_outlier_col: str = 'is_outlier'
    ) -> pd.DataFrame:
        """
        Phát hiện các ngoại lệ trong DataFrame.
        
        Args:
            df: DataFrame cần kiểm tra
            columns: Danh sách cột cần kiểm tra (None để kiểm tra tất cả các cột số)
            per_column: Nếu True, phát hiện ngoại lệ trên từng cột riêng biệt
                       Nếu False, phát hiện ngoại lệ trên toàn bộ dữ liệu
            outlier_suffix: Hậu tố để thêm vào tên cột khi tạo cột đánh dấu ngoại lệ riêng
            global_outlier_col: Tên cột để đánh dấu ngoại lệ chung
            
        Returns:
            DataFrame với các cột mới đánh dấu các ngoại lệ
        """
        if df.empty:
            self.logger.warning("DataFrame rỗng, không có gì để phát hiện")
            return df
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Xác định các cột cần kiểm tra
        if columns is None:
            # Lấy tất cả các cột số
            columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Chỉ lấy các cột tồn tại trong DataFrame
            columns = [col for col in columns if col in result_df.columns]
        
        self.logger.info(f"Bắt đầu phát hiện ngoại lệ cho {len(columns)} cột")
        
        # Tạo cột global để đánh dấu ngoại lệ
        result_df[global_outlier_col] = False
        
        # Tạo các cột đánh dấu ngoại lệ cho từng cột nếu per_column=True
        if per_column:
            for col in columns:
                result_df[f"{col}{outlier_suffix}"] = False
        
        # Dựa trên phương pháp đã chọn
        if self.method == 'z-score':
            self._detect_zscore_outliers(result_df, columns, per_column, outlier_suffix, global_outlier_col)
        
        elif self.method == 'iqr':
            self._detect_iqr_outliers(result_df, columns, per_column, outlier_suffix, global_outlier_col)
        
        elif self.method == 'isolation-forest':
            self._detect_isolation_forest_outliers(result_df, columns, per_column, outlier_suffix, global_outlier_col)
        
        elif self.method == 'dbscan':
            self._detect_dbscan_outliers(result_df, columns, per_column, outlier_suffix, global_outlier_col)
            
        elif self.method == 'lof':
            self._detect_lof_outliers(result_df, columns, per_column, outlier_suffix, global_outlier_col)
            
        elif self.method == 'one-class-svm':
            self._detect_one_class_svm_outliers(result_df, columns, per_column, outlier_suffix, global_outlier_col)
        
        else:
            self.logger.warning(f"Phương pháp không hợp lệ: {self.method}, sử dụng z-score")
            self._detect_zscore_outliers(result_df, columns, per_column, outlier_suffix, global_outlier_col)
        
        outlier_count = result_df[global_outlier_col].sum()
        self.logger.info(f"Đã phát hiện {outlier_count} ngoại lệ trong {len(result_df)} bản ghi")
        
        # Kiểm tra các chỉ báo kỹ thuật nếu có
        technical_columns = [col for col in result_df.columns if any(ind in col.lower()
                            for ind in ['rsi_', 'macd', 'stoch_', 'williams_r', 'obv', 'volume'])]
        if technical_columns:
            self.logger.info(f"Kiểm tra tính hợp lệ của {len(technical_columns)} chỉ báo kỹ thuật")
            result_df = self.check_technical_indicators(result_df)
            
        return result_df
    
    def _detect_zscore_outliers(self, df: pd.DataFrame, columns: List[str], 
                              per_column: bool = False, outlier_suffix: str = '_is_outlier',
                              global_outlier_col: str = 'is_outlier') -> None:
        """
        Phát hiện ngoại lệ bằng phương pháp Z-score.
        
        Args:
            df: DataFrame cần kiểm tra
            columns: Danh sách cột cần kiểm tra
            per_column: Nếu True, phát hiện ngoại lệ trên từng cột riêng biệt
            outlier_suffix: Hậu tố để thêm vào tên cột khi tạo cột đánh dấu ngoại lệ riêng
            global_outlier_col: Tên cột để đánh dấu ngoại lệ chung
        """
        for col in columns:
            if col not in df.columns:
                continue
                
            # Loại bỏ các giá trị NA
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            if self.use_robust:
                # Sử dụng median và MAD
                median = col_data.median()
                mad = stats.median_abs_deviation(col_data, nan_policy='omit')
                
                if mad == 0:
                    # Tránh chia cho 0
                    continue
                
                z_scores = 0.6745 * np.abs(col_data - median) / mad
            else:
                # Sử dụng mean và std
                mean = col_data.mean()
                std = col_data.std()
                
                if std == 0:
                    # Tránh chia cho 0
                    continue
                
                z_scores = np.abs((col_data - mean) / std)
            
            # Đánh dấu ngoại lệ
            outliers = z_scores > self.threshold
            
            # Cập nhật cột đánh dấu ngoại lệ riêng nếu cần
            if per_column:
                col_outlier_name = f"{col}{outlier_suffix}"
                df.loc[outliers.index, col_outlier_name] = True
            
            # Luôn cập nhật cột đánh dấu ngoại lệ chung
            df.loc[outliers.index, global_outlier_col] = True
    
    def _detect_iqr_outliers(self, df: pd.DataFrame, columns: List[str],
                            per_column: bool = False, outlier_suffix: str = '_is_outlier',
                            global_outlier_col: str = 'is_outlier') -> None:
        """
        Phát hiện ngoại lệ bằng phương pháp IQR (Interquartile Range).
        
        Args:
            df: DataFrame cần kiểm tra
            columns: Danh sách cột cần kiểm tra
            per_column: Nếu True, phát hiện ngoại lệ trên từng cột riêng biệt
            outlier_suffix: Hậu tố để thêm vào tên cột khi tạo cột đánh dấu ngoại lệ riêng
            global_outlier_col: Tên cột để đánh dấu ngoại lệ chung
        """
        for col in columns:
            if col not in df.columns:
                continue
                
            # Loại bỏ các giá trị NA
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Tính Q1, Q3 và IQR
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:
                # Tránh chia cho 0
                continue
            
            # Xác định ngưỡng trên và dưới
            lower_bound = q1 - self.threshold * iqr
            upper_bound = q3 + self.threshold * iqr
            
            # Đánh dấu ngoại lệ
            outliers = (col_data < lower_bound) | (col_data > upper_bound)
            
            # Cập nhật cột đánh dấu ngoại lệ riêng nếu cần
            if per_column:
                col_outlier_name = f"{col}{outlier_suffix}"
                df.loc[outliers.index, col_outlier_name] = True
            
            # Luôn cập nhật cột đánh dấu ngoại lệ chung
            df.loc[outliers.index, global_outlier_col] = True
    
    def _detect_isolation_forest_outliers(self, df: pd.DataFrame, columns: List[str],
                                      per_column: bool = False, outlier_suffix: str = '_is_outlier',
                                      global_outlier_col: str = 'is_outlier') -> None:
        """
        Phát hiện ngoại lệ bằng phương pháp Isolation Forest.
        
        Args:
            df: DataFrame cần kiểm tra
            columns: Danh sách cột cần kiểm tra
            per_column: Nếu True, phát hiện ngoại lệ trên từng cột riêng biệt
            outlier_suffix: Hậu tố để thêm vào tên cột khi tạo cột đánh dấu ngoại lệ riêng
            global_outlier_col: Tên cột để đánh dấu ngoại lệ chung
        """
        try:
            from sklearn.ensemble import IsolationForest
            
            if per_column:
                # Xử lý từng cột riêng biệt
                for col in columns:
                    if col not in df.columns:
                        continue
                        
                    # Chuẩn bị dữ liệu
                    X = df[[col]].copy()
                    
                    # Loại bỏ các hàng có giá trị NA
                    X = X.dropna()
                    
                    if len(X) == 0:
                        continue
                    
                    # Tạo model mới cho từng cột
                    if col not in self.models:
                        self.models[col] = IsolationForest(contamination=self.contamination, random_state=42)
                    
                    # Huấn luyện mô hình
                    self.models[col].fit(X)
                    
                    # Dự đoán ngoại lệ (-1 là ngoại lệ, 1 là bình thường)
                    predictions = self.models[col].predict(X)
                    
                    # Đánh dấu ngoại lệ
                    outliers = predictions == -1
                    
                    # Cập nhật cột đánh dấu ngoại lệ riêng
                    col_outlier_name = f"{col}{outlier_suffix}"
                    df.loc[X.index[outliers], col_outlier_name] = True
                    
                    # Cập nhật cột đánh dấu ngoại lệ chung
                    df.loc[X.index[outliers], global_outlier_col] = True
            else:
                # Chuẩn bị dữ liệu cho toàn bộ các cột
                X = df[columns].copy()
                
                # Loại bỏ các hàng có giá trị NA
                X = X.dropna()
                
                if len(X) == 0:
                    self.logger.warning("Không có dữ liệu hợp lệ để phát hiện ngoại lệ")
                    return
                
                # Huấn luyện mô hình
                self.model.fit(X)
                
                # Dự đoán ngoại lệ (-1 là ngoại lệ, 1 là bình thường)
                predictions = self.model.predict(X)
                
                # Đánh dấu ngoại lệ
                outliers = predictions == -1
                df.loc[X.index[outliers], global_outlier_col] = True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi sử dụng Isolation Forest: {e}")
            # Sử dụng phương pháp dự phòng
            self._detect_zscore_outliers(df, columns, per_column, outlier_suffix, global_outlier_col)
    
    def _detect_dbscan_outliers(self, df: pd.DataFrame, columns: List[str],
                                per_column: bool = False, outlier_suffix: str = '_is_outlier',
                                global_outlier_col: str = 'is_outlier') -> None:
        """
        Phát hiện ngoại lệ bằng phương pháp DBSCAN.
        
        Args:
            df: DataFrame cần kiểm tra
            columns: Danh sách cột cần kiểm tra
            per_column: Nếu True, phát hiện ngoại lệ trên từng cột riêng biệt
            outlier_suffix: Hậu tố để thêm vào tên cột khi tạo cột đánh dấu ngoại lệ riêng
            global_outlier_col: Tên cột để đánh dấu ngoại lệ chung
        """
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            
            if per_column:
                # Xử lý từng cột riêng biệt
                for col in columns:
                    if col not in df.columns:
                        continue
                        
                    # Chuẩn bị dữ liệu
                    X = df[[col]].copy()
                    
                    # Loại bỏ các hàng có giá trị NA
                    X = X.dropna()
                    
                    if len(X) == 0:
                        continue
                    
                    # Chuẩn hóa dữ liệu
                    X_scaled = StandardScaler().fit_transform(X)
                    
                    # Tạo model mới cho từng cột
                    if col not in self.models:
                        self.models[col] = DBSCAN(eps=self.eps, min_samples=self.min_samples)
                    
                    # Áp dụng DBSCAN
                    self.models[col].fit(X_scaled)
                    
                    # Các điểm có nhãn -1 là ngoại lệ
                    outliers = self.models[col].labels_ == -1
                    
                    # Cập nhật cột đánh dấu ngoại lệ riêng
                    col_outlier_name = f"{col}{outlier_suffix}"
                    df.loc[X.index[outliers], col_outlier_name] = True
                    
                    # Cập nhật cột đánh dấu ngoại lệ chung
                    df.loc[X.index[outliers], global_outlier_col] = True
            else:
                # Chuẩn bị dữ liệu cho toàn bộ các cột
                X = df[columns].copy()
                
                # Loại bỏ các hàng có giá trị NA
                X = X.dropna()
                
                if len(X) == 0:
                    self.logger.warning("Không có dữ liệu hợp lệ để phát hiện ngoại lệ")
                    return
                
                # Chuẩn hóa dữ liệu
                X_scaled = StandardScaler().fit_transform(X)
                
                # Áp dụng DBSCAN
                self.model.fit(X_scaled)
                
                # Các điểm có nhãn -1 là ngoại lệ
                outliers = self.model.labels_ == -1
                df.loc[X.index[outliers], global_outlier_col] = True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi sử dụng DBSCAN: {e}")
            # Sử dụng phương pháp dự phòng
            self._detect_zscore_outliers(df, columns, per_column, outlier_suffix, global_outlier_col)
    
    def _detect_lof_outliers(self, df: pd.DataFrame, columns: List[str],
                             per_column: bool = False, outlier_suffix: str = '_is_outlier',
                             global_outlier_col: str = 'is_outlier') -> None:
        """
        Phát hiện ngoại lệ bằng phương pháp Local Outlier Factor (LOF).
        
        Args:
            df: DataFrame cần kiểm tra
            columns: Danh sách cột cần kiểm tra
            per_column: Nếu True, phát hiện ngoại lệ trên từng cột riêng biệt
            outlier_suffix: Hậu tố để thêm vào tên cột khi tạo cột đánh dấu ngoại lệ riêng
            global_outlier_col: Tên cột để đánh dấu ngoại lệ chung
        """
        try:
            from sklearn.neighbors import LocalOutlierFactor
            from sklearn.preprocessing import StandardScaler
            
            if per_column:
                # Xử lý từng cột riêng biệt
                for col in columns:
                    if col not in df.columns:
                        continue
                        
                    # Chuẩn bị dữ liệu
                    X = df[[col]].copy()
                    
                    # Loại bỏ các hàng có giá trị NA
                    X = X.dropna()
                    
                    if len(X) == 0:
                        continue
                    
                    # Chuẩn hóa dữ liệu
                    X_scaled = StandardScaler().fit_transform(X)
                    
                    # Tạo model mới cho từng cột
                    lof = LocalOutlierFactor(n_neighbors=min(self.n_neighbors, len(X)-1), contamination=self.contamination)
                    
                    # Phát hiện ngoại lệ (-1 là ngoại lệ, 1 là bình thường)
                    predictions = lof.fit_predict(X_scaled)
                    
                    # Đánh dấu ngoại lệ
                    outliers = predictions == -1
                    
                    # Cập nhật cột đánh dấu ngoại lệ riêng
                    col_outlier_name = f"{col}{outlier_suffix}"
                    df.loc[X.index[outliers], col_outlier_name] = True
                    
                    # Cập nhật cột đánh dấu ngoại lệ chung
                    df.loc[X.index[outliers], global_outlier_col] = True
            else:
                # Chuẩn bị dữ liệu cho toàn bộ các cột
                X = df[columns].copy()
                
                # Loại bỏ các hàng có giá trị NA
                X = X.dropna()
                
                if len(X) == 0:
                    self.logger.warning("Không có dữ liệu hợp lệ để phát hiện ngoại lệ")
                    return
                
                # Chuẩn hóa dữ liệu
                X_scaled = StandardScaler().fit_transform(X)
                
                # Phát hiện ngoại lệ
                n_neighbors = min(self.n_neighbors, len(X)-1)
                lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=self.contamination)
                predictions = lof.fit_predict(X_scaled)
                
                # Đánh dấu ngoại lệ
                outliers = predictions == -1
                df.loc[X.index[outliers], global_outlier_col] = True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi sử dụng Local Outlier Factor: {e}")
            # Sử dụng phương pháp dự phòng
            self._detect_zscore_outliers(df, columns, per_column, outlier_suffix, global_outlier_col)
    
    def _detect_one_class_svm_outliers(self, df: pd.DataFrame, columns: List[str],
                                       per_column: bool = False, outlier_suffix: str = '_is_outlier',
                                       global_outlier_col: str = 'is_outlier') -> None:
        """
        Phát hiện ngoại lệ bằng phương pháp One-Class SVM.
        
        Args:
            df: DataFrame cần kiểm tra
            columns: Danh sách cột cần kiểm tra
            per_column: Nếu True, phát hiện ngoại lệ trên từng cột riêng biệt
            outlier_suffix: Hậu tố để thêm vào tên cột khi tạo cột đánh dấu ngoại lệ riêng
            global_outlier_col: Tên cột để đánh dấu ngoại lệ chung
        """
        try:
            from sklearn import svm
            from sklearn.preprocessing import StandardScaler
            
            if per_column:
                # Xử lý từng cột riêng biệt
                for col in columns:
                    if col not in df.columns:
                        continue
                        
                    # Chuẩn bị dữ liệu
                    X = df[[col]].copy()
                    
                    # Loại bỏ các hàng có giá trị NA
                    X = X.dropna()
                    
                    if len(X) == 0:
                        continue
                    
                    # Chuẩn hóa dữ liệu
                    X_scaled = StandardScaler().fit_transform(X)
                    
                    # Tạo model mới cho từng cột
                    one_class = svm.OneClassSVM(kernel=self.kernel, nu=self.nu)
                    
                    # Phát hiện ngoại lệ (1 là bình thường, -1 là ngoại lệ)
                    one_class.fit(X_scaled)
                    predictions = one_class.predict(X_scaled)
                    
                    # Đánh dấu ngoại lệ
                    outliers = predictions == -1
                    
                    # Cập nhật cột đánh dấu ngoại lệ riêng
                    col_outlier_name = f"{col}{outlier_suffix}"
                    df.loc[X.index[outliers], col_outlier_name] = True
                    
                    # Cập nhật cột đánh dấu ngoại lệ chung
                    df.loc[X.index[outliers], global_outlier_col] = True
            else:
                # Chuẩn bị dữ liệu cho toàn bộ các cột
                X = df[columns].copy()
                
                # Loại bỏ các hàng có giá trị NA
                X = X.dropna()
                
                if len(X) == 0:
                    self.logger.warning("Không có dữ liệu hợp lệ để phát hiện ngoại lệ")
                    return
                
                # Chuẩn hóa dữ liệu
                X_scaled = StandardScaler().fit_transform(X)
                
                # Phát hiện ngoại lệ
                one_class = svm.OneClassSVM(kernel=self.kernel, nu=self.nu)
                one_class.fit(X_scaled)
                predictions = one_class.predict(X_scaled)
                
                # Đánh dấu ngoại lệ
                outliers = predictions == -1
                df.loc[X.index[outliers], global_outlier_col] = True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi sử dụng One-Class SVM: {e}")
            # Sử dụng phương pháp dự phòng
            self._detect_zscore_outliers(df, columns, per_column, outlier_suffix, global_outlier_col)
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        replace_with: Optional[str] = None,
        per_column: bool = False,
        outlier_suffix: str = '_is_outlier',
        global_outlier_col: str = 'is_outlier',
        use_column_specific: bool = False
    ) -> pd.DataFrame:
        """
        Loại bỏ hoặc thay thế các ngoại lệ trong DataFrame.
        
        Args:
            df: DataFrame cần xử lý
            columns: Danh sách cột cần kiểm tra (None để kiểm tra tất cả các cột số)
            replace_with: Phương pháp thay thế ('mean', 'median', 'mode', 'interpolate', None để loại bỏ)
            per_column: Nếu True, phát hiện ngoại lệ trên từng cột riêng biệt
            outlier_suffix: Hậu tố để thêm vào tên cột khi tạo cột đánh dấu ngoại lệ riêng
            global_outlier_col: Tên cột để đánh dấu ngoại lệ chung
            use_column_specific: Nếu True, sử dụng các cột đánh dấu ngoại lệ riêng của từng cột 
                               thay vì cột chung để xác định ngoại lệ khi thay thế
            
        Returns:
            DataFrame đã xử lý
        """
        # Phát hiện ngoại lệ
        outlier_df = self.detect_outliers(df, columns, per_column, outlier_suffix, global_outlier_col)
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Xác định các cột cần xử lý
        if columns is None:
            # Lấy tất cả các cột số
            columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Chỉ lấy các cột tồn tại trong DataFrame
            columns = [col for col in columns if col in result_df.columns]
        
        if use_column_specific and per_column:
            # Xử lý riêng cho từng cột dựa trên cột đánh dấu riêng
            for col in columns:
                col_outlier_name = f"{col}{outlier_suffix}"
                if col_outlier_name not in outlier_df.columns:
                    continue
                
                # Lấy các hàng là ngoại lệ cho cột này
                outlier_rows = outlier_df[outlier_df[col_outlier_name]].index
                
                # Nếu không có ngoại lệ, bỏ qua
                if len(outlier_rows) == 0:
                    continue
                
                # Xử lý các ngoại lệ cho cột này
                self._replace_outliers_for_column(result_df, col, outlier_rows, replace_with)
                
            self.logger.info(f"Đã xử lý ngoại lệ cho từng cột riêng biệt")
        else:
            # Lấy các hàng là ngoại lệ theo cột chung
            outlier_rows = outlier_df[outlier_df[global_outlier_col]].index
            
            # Nếu không có ngoại lệ, trả về dữ liệu gốc
            if len(outlier_rows) == 0:
                self.logger.info("Không phát hiện ngoại lệ")
                return result_df
            
            # Xử lý các ngoại lệ
            if replace_with is None:
                # Loại bỏ các hàng có ngoại lệ
                result_df = result_df.drop(outlier_rows)
                self.logger.info(f"Đã loại bỏ {len(outlier_rows)} hàng có ngoại lệ")
            
            else:
                # Thay thế giá trị cho từng cột
                for col in columns:
                    self._replace_outliers_for_column(result_df, col, outlier_rows, replace_with)
        
        return result_df
    
    def _replace_outliers_for_column(self, df: pd.DataFrame, column: str, outlier_rows: pd.Index, 
                                   replace_with: Optional[str]) -> None:
        """
        Thay thế các ngoại lệ trong một cột cụ thể.
        
        Args:
            df: DataFrame cần xử lý
            column: Tên cột cần xử lý
            outlier_rows: Các hàng chứa ngoại lệ
            replace_with: Phương pháp thay thế
        """
        if replace_with == 'mean':
            # Thay thế bằng giá trị trung bình
            mean_val = df[column].mean()
            df.loc[outlier_rows, column] = mean_val
            
        elif replace_with == 'median':
            # Thay thế bằng giá trị trung vị
            median_val = df[column].median()
            df.loc[outlier_rows, column] = median_val
            
        elif replace_with == 'mode':
            # Thay thế bằng giá trị xuất hiện nhiều nhất
            mode_val = df[column].mode()[0]
            df.loc[outlier_rows, column] = mode_val
            
        elif replace_with == 'interpolate':
            # Đánh dấu các ngoại lệ là NA
            original_values = df.loc[outlier_rows, column].copy()
            df.loc[outlier_rows, column] = np.nan
            
            # Nội suy các giá trị NA
            df[column] = df[column].interpolate(method='linear')
            
            # Nếu còn giá trị NA ở đầu hoặc cuối, sử dụng ffill và bfill
            if df[column].isna().any():
                df[column] = df[column].fillna(method='ffill').fillna(method='bfill')
                
                # Nếu vẫn còn NA, khôi phục giá trị gốc
                still_na = df[column].isna()
                if still_na.any():
                    na_rows = outlier_rows[still_na.loc[outlier_rows]]
                    df.loc[na_rows, column] = original_values.loc[na_rows]
    
    def get_outlier_stats(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        per_column: bool = False,
        outlier_suffix: str = '_is_outlier',
        global_outlier_col: str = 'is_outlier'
    ) -> Dict[str, Any]:
        """
        Lấy thống kê về các ngoại lệ trong DataFrame.
        
        Args:
            df: DataFrame cần kiểm tra
            columns: Danh sách cột cần kiểm tra (None để kiểm tra tất cả các cột số)
            per_column: Nếu True, phát hiện ngoại lệ trên từng cột riêng biệt
            outlier_suffix: Hậu tố để thêm vào tên cột khi tạo cột đánh dấu ngoại lệ riêng
            global_outlier_col: Tên cột để đánh dấu ngoại lệ chung
            
        Returns:
            Dict với thống kê về ngoại lệ
        """
        # Phát hiện ngoại lệ
        outlier_df = self.detect_outliers(df, columns, per_column, outlier_suffix, global_outlier_col)
        
        # Lấy các hàng là ngoại lệ theo cột chung
        outlier_rows = outlier_df[outlier_df[global_outlier_col]]
        
        # Xác định các cột cần phân tích
        if columns is None:
            # Lấy tất cả các cột số
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Chỉ lấy các cột tồn tại trong DataFrame
            columns = [col for col in columns if col in df.columns]
        
        # Tạo kết quả
        result = {
            'total_samples': len(df),
            'total_outliers': len(outlier_rows),
            'outlier_percentage': len(outlier_rows) / len(df) * 100 if len(df) > 0 else 0,
            'outliers_per_column': {},
            'method': self.method,
            'threshold': self.threshold,
            'per_column_mode': per_column
        }
        
        # Tính số lượng ngoại lệ cho mỗi cột
        for col in columns:
            # Nếu đang sử dụng per_column, lấy thông tin từ cột đánh dấu riêng
            if per_column and f"{col}{outlier_suffix}" in outlier_df.columns:
                col_outliers = outlier_df[outlier_df[f"{col}{outlier_suffix}"]]
                outlier_count = len(col_outliers)
            else:
                # Thực hiện tính toán như trước
                if self.method == 'z-score':
                    try:
                        # Tính Z-score
                        if self.use_robust:
                            median = df[col].median()
                            mad = stats.median_abs_deviation(df[col].dropna(), nan_policy='omit')
                            
                            if mad == 0:
                                continue
                            
                            z_scores = 0.6745 * np.abs(df[col] - median) / mad
                        else:
                            mean = df[col].mean()
                            std = df[col].std()
                            
                            if std == 0:
                                continue
                            
                            z_scores = np.abs((df[col] - mean) / std)
                        
                        # Đếm ngoại lệ
                        outliers = z_scores > self.threshold
                        outlier_count = outliers.sum()
                    except Exception as e:
                        self.logger.warning(f"Không thể tính toán thống kê cho cột {col}: {e}")
                        outlier_count = 0
                else:
                    # Đối với các phương pháp khác, chúng ta chỉ đếm các hàng ngoại lệ chung
                    outlier_count = len(outlier_rows)
            
            # Lấy thống kê mô tả cho cột
            try:
                col_stats = {
                    'count': int(outlier_count),
                    'percentage': float(outlier_count / len(df) * 100) if len(df) > 0 else 0,
                    'min_value': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    'max_value': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                    'std': float(df[col].std()) if not pd.isna(df[col].std()) else None
                }
                
                # Thêm thống kê về các giá trị ngoại lệ nếu có
                if outlier_count > 0:
                    if per_column and f"{col}{outlier_suffix}" in outlier_df.columns:
                        outlier_values = df.loc[outlier_df[f"{col}{outlier_suffix}"], col]
                    else:
                        # Sử dụng Z-score hoặc IQR để xác định lại ngoại lệ cho thống kê
                        if self.method == 'z-score':
                            if self.use_robust:
                                median = df[col].median()
                                mad = stats.median_abs_deviation(df[col].dropna(), nan_policy='omit')
                                if mad > 0:
                                    z_scores = 0.6745 * np.abs(df[col] - median) / mad
                                    outliers = z_scores > self.threshold
                                    outlier_values = df.loc[outliers, col]
                                else:
                                    outlier_values = pd.Series()
                            else:
                                mean = df[col].mean()
                                std = df[col].std()
                                if std > 0:
                                    z_scores = np.abs((df[col] - mean) / std)
                                    outliers = z_scores > self.threshold
                                    outlier_values = df.loc[outliers, col]
                                else:
                                    outlier_values = pd.Series()
                        else:
                            # Sử dụng IQR cho các phương pháp khác
                            q1 = df[col].quantile(0.25)
                            q3 = df[col].quantile(0.75)
                            iqr = q3 - q1
                            if iqr > 0:
                                lower_bound = q1 - 1.5 * iqr
                                upper_bound = q3 + 1.5 * iqr
                                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                                outlier_values = df.loc[outliers, col]
                            else:
                                outlier_values = pd.Series()
                    
                    # Thêm thống kê về giá trị ngoại lệ
                    if not outlier_values.empty:
                        col_stats.update({
                            'outlier_min': float(outlier_values.min()) if not pd.isna(outlier_values.min()) else None,
                            'outlier_max': float(outlier_values.max()) if not pd.isna(outlier_values.max()) else None,
                            'outlier_mean': float(outlier_values.mean()) if not pd.isna(outlier_values.mean()) else None
                        })
                
                result['outliers_per_column'][col] = col_stats
                
            except Exception as e:
                self.logger.warning(f"Không thể tính toán thống kê mô tả cho cột {col}: {e}")
        
        return result

    def check_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kiểm tra tính hợp lệ của các chỉ báo kỹ thuật như RSI, Stochastic, MACD, v.v.
        
        Args:
            df: DataFrame chứa các chỉ báo kỹ thuật
            
        Returns:
            DataFrame đã được kiểm tra và sửa
        """
        result_df = df.copy()
        
        # 1. Kiểm tra các cột RSI (các cột có tên chứa 'rsi_')
        rsi_columns = [col for col in result_df.columns if 'rsi_' in col.lower() and not col.endswith('_is_outlier')]
        for col in rsi_columns:
            # Kiểm tra giá trị RSI nằm ngoài phạm vi [0, 100]
            invalid_mask = (result_df[col] < 0) | (result_df[col] > 100) | ~np.isfinite(result_df[col])
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                self.logger.warning(f"Phát hiện {invalid_count} giá trị không hợp lệ trong cột {col}")
                
                # Sửa giá trị không hợp lệ
                if '_norm_' in col:
                    # Nếu là RSI đã chuẩn hóa (0-1), sửa về 0.5
                    result_df.loc[invalid_mask, col] = 0.5
                else:
                    # Nếu là RSI chưa chuẩn hóa (0-100), sửa về 50
                    result_df.loc[invalid_mask, col] = 50
        
        # 2. Kiểm tra các cột MACD (không có phạm vi cố định, nhưng kiểm tra NaN, Inf)
        macd_columns = [col for col in result_df.columns if 'macd' in col.lower() and not col.endswith('_is_outlier')]
        for col in macd_columns:
            # Kiểm tra giá trị không hợp lệ (NaN, inf)
            invalid_mask = ~np.isfinite(result_df[col])
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                self.logger.warning(f"Phát hiện {invalid_count} giá trị không hợp lệ trong cột {col}")
                
                # Thay thế bằng giá trị trung bình có điều kiện
                valid_values = result_df.loc[~invalid_mask, col]
                if len(valid_values) > 0:
                    replacement = valid_values.mean()
                else:
                    replacement = 0  # Nếu không có giá trị hợp lệ nào
                
                result_df.loc[invalid_mask, col] = replacement
        
        # 3. Kiểm tra Stochastic Oscillator (giá trị 0-100)
        stoch_columns = [col for col in result_df.columns if ('stoch_' in col.lower() or 'stochastic' in col.lower()) 
                        and not col.endswith('_is_outlier')]
        for col in stoch_columns:
            invalid_mask = (result_df[col] < 0) | (result_df[col] > 100) | ~np.isfinite(result_df[col])
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                self.logger.warning(f"Phát hiện {invalid_count} giá trị không hợp lệ trong cột {col}")
                
                # Sửa giá trị không hợp lệ
                if '_norm_' in col:
                    result_df.loc[invalid_mask, col] = 0.5
                else:
                    result_df.loc[invalid_mask, col] = 50
        
        # 4. Kiểm tra Williams %R (giá trị -100 đến 0)
        williams_columns = [col for col in result_df.columns if 'williams_r' in col.lower() and not col.endswith('_is_outlier')]
        for col in williams_columns:
            if '_norm_' in col:
                # Kiểm tra Williams %R đã chuẩn hóa (0-1)
                invalid_mask = (result_df[col] < 0) | (result_df[col] > 1) | ~np.isfinite(result_df[col])
                if invalid_mask.sum() > 0:
                    self.logger.warning(f"Sửa {invalid_mask.sum()} giá trị Williams %R chuẩn hóa không hợp lệ")
                    result_df.loc[invalid_mask, col] = 0.5
            else:
                # Kiểm tra Williams %R thông thường (-100 đến 0)
                invalid_mask = (result_df[col] < -100) | (result_df[col] > 0) | ~np.isfinite(result_df[col])
                if invalid_mask.sum() > 0:
                    self.logger.warning(f"Sửa {invalid_mask.sum()} giá trị Williams %R không hợp lệ")
                    result_df.loc[invalid_mask, col] = -50
        
        # 5. Kiểm tra On-Balance Volume (không có giới hạn cụ thể, chỉ kiểm tra NaN, Inf)
        obv_columns = [col for col in result_df.columns if 'obv' in col.lower() and not col.endswith('_is_outlier')]
        for col in obv_columns:
            invalid_mask = ~np.isfinite(result_df[col])
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                self.logger.warning(f"Phát hiện {invalid_count} giá trị OBV không hợp lệ")
                
                # Xử lý các giá trị không hợp lệ
                # Sử dụng việc nội suy nếu có thể, hoặc thay thế bằng 0
                if invalid_mask.all():
                    # Nếu tất cả đều không hợp lệ, thay thế bằng 0
                    result_df[col] = 0
                else:
                    # Thử nội suy
                    temp_col = result_df[col].copy()
                    temp_col[invalid_mask] = np.nan
                    interpolated = temp_col.interpolate(method='linear')
                    
                    # Nếu vẫn còn NA sau nội suy (ở đầu hoặc cuối)
                    if interpolated.isna().any():
                        interpolated = interpolated.fillna(method='ffill').fillna(method='bfill')
                        
                        # Nếu vẫn còn NA, thay thế bằng 0
                        if interpolated.isna().any():
                            interpolated = interpolated.fillna(0)
                    
                    result_df[col] = interpolated
        
        # 6. Kiểm tra và xử lý các giá trị volume cực đại
        volume_columns = [col for col in result_df.columns if col.lower() == 'volume' or col.lower().endswith('_volume')]
        for col in volume_columns:
            # Kiểm tra giá trị âm (volume không thể âm)
            invalid_mask = (result_df[col] < 0) | ~np.isfinite(result_df[col])
            if invalid_mask.sum() > 0:
                self.logger.warning(f"Sửa {invalid_mask.sum()} giá trị {col} không hợp lệ (âm hoặc NaN/Inf)")
                result_df.loc[invalid_mask, col] = 0
            
            # Kiểm tra giá trị cực đại (outlier)
            mean_vol = result_df[col].mean()
            std_vol = result_df[col].std()
            if std_vol > 0:
                # Phát hiện outlier với z-score cao (> 10)
                z_scores = (result_df[col] - mean_vol) / std_vol
                extreme_mask = z_scores > 10
                if extreme_mask.sum() > 0:
                    self.logger.warning(f"Phát hiện {extreme_mask.sum()} giá trị {col} cực đại")
                    # Không thay đổi giá trị này, chỉ ghi nhận
        
        return result_df

class OutlierDetectionMethod:
    """
    Enum-like class để định nghĩa các phương pháp phát hiện ngoại lệ.
    """
    Z_SCORE = 'z-score'
    IQR = 'iqr'
    ISOLATION_FOREST = 'isolation-forest'
    DBSCAN = 'dbscan'
    LOCAL_OUTLIER_FACTOR = 'lof'
    ONE_CLASS_SVM = 'one-class-svm'
    
    @classmethod
    def get_all_methods(cls) -> List[str]:
        """Lấy danh sách tất cả các phương pháp."""
        return [cls.Z_SCORE, cls.IQR, cls.ISOLATION_FOREST, 
                cls.DBSCAN, cls.LOCAL_OUTLIER_FACTOR, cls.ONE_CLASS_SVM]
    
    @classmethod
    def get_statistical_methods(cls) -> List[str]:
        """Lấy danh sách các phương pháp thống kê."""
        return [cls.Z_SCORE, cls.IQR]
    
    @classmethod
    def get_machine_learning_methods(cls) -> List[str]:
        """Lấy danh sách các phương pháp machine learning."""
        return [cls.ISOLATION_FOREST, cls.DBSCAN, 
                cls.LOCAL_OUTLIER_FACTOR, cls.ONE_CLASS_SVM]


class OutlierFilterOps:
    """
    Enum-like class để định nghĩa các thao tác xử lý ngoại lệ.
    """
    REMOVE = 'remove'
    REPLACE_MEAN = 'mean'
    REPLACE_MEDIAN = 'median'
    REPLACE_MODE = 'mode'
    REPLACE_INTERPOLATE = 'interpolate'
    
    @classmethod
    def get_all_ops(cls) -> List[str]:
        """Lấy danh sách tất cả các thao tác."""
        return [cls.REMOVE, cls.REPLACE_MEAN, cls.REPLACE_MEDIAN, 
                cls.REPLACE_MODE, cls.REPLACE_INTERPOLATE]