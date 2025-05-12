"""
Xử lý dữ liệu thiếu trong dữ liệu thị trường.
File này cung cấp các lớp và phương thức để phát hiện và xử lý
các giá trị thiếu trong dữ liệu thị trường sử dụng nhiều phương pháp khác nhau.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import setup_logger

class MissingDataHandler:
    """
    Lớp để xử lý các giá trị thiếu trong dữ liệu.
    """
    
    def __init__(
        self,
        method: str = 'interpolate',
        knn_neighbors: int = 5,
        max_iter: int = 10,
        fill_value: Optional[Union[int, float, str]] = None,
        fallback_strategy: List[str] = None,
        auto_detect_time_series: bool = True,
        handle_categorical: bool = True,
        aggressive_cleaning: bool = True,
        ensure_no_nan: bool = True
    ):
        """
        Khởi tạo missing data handler.
        
        Args:
            method: Phương pháp xử lý ('mean', 'median', 'mode', 'constant', 'ffill', 'bfill',
                  'interpolate', 'knn', 'iterative', 'drop')
            knn_neighbors: Số lượng hàng xóm cho phương pháp KNN
            max_iter: Số lần lặp tối đa cho phương pháp iterative
            fill_value: Giá trị điền cho phương pháp 'constant'
            fallback_strategy: Danh sách các phương pháp backup theo thứ tự nếu phương pháp chính fail
                              (mặc định: ['interpolate', 'ffill', 'bfill', 'mean', 'median'])
            auto_detect_time_series: Tự động phát hiện và xử lý dữ liệu chuỗi thời gian
            handle_categorical: Tự động xử lý dữ liệu categorical khi áp dụng các phương pháp numeric
        """
        self.logger = logging.getLogger("MissingDataHandler")
        self.method = method
        self.fallback_strategy = fallback_strategy or ['interpolate', 'ffill', 'bfill', 'mean']
        self.aggressive_cleaning = aggressive_cleaning  # ✅ Đã thêm
        self.ensure_no_nan = ensure_no_nan              # ✅ Đã thêm
        self.knn_neighbors = knn_neighbors
        self.max_iter = max_iter
        self.fill_value = fill_value
        
        # Thiết lập fallback strategy
        self.fallback_strategy = fallback_strategy
        if self.fallback_strategy is None:
            self.fallback_strategy = ['interpolate', 'ffill', 'bfill', 'mean', 'median']
        
        # Xóa phương pháp chính khỏi danh sách fallback nếu có
        if self.method in self.fallback_strategy:
            self.fallback_strategy.remove(self.method)
        
        # Cấu hình xử lý nâng cao
        self.auto_detect_time_series = auto_detect_time_series
        self.handle_categorical = handle_categorical
        
        # Lưu trữ thông tin về chuyển đổi categorical
        self.category_mappings = {}
        self.original_dtypes = {}
        
        # Khởi tạo các imputer nếu cần
        self.imputer = None
        if method in ['mean', 'median', 'mode', 'constant', 'knn', 'iterative']:
            self._init_imputers()
        
        self.logger.info(f"Đã khởi tạo MissingDataHandler với phương pháp {self.method}")
        self.logger.debug(f"Fallback strategy: {', '.join(self.fallback_strategy)}")
    
    def _init_imputers(self) -> None:
        """
        Khởi tạo các imputer dựa trên phương pháp đã chọn.
        """
        try:
            from sklearn.impute import SimpleImputer
            
            if self.method == 'mean':
                self.imputer = SimpleImputer(strategy='mean')
            elif self.method == 'median':
                self.imputer = SimpleImputer(strategy='median')
            elif self.method == 'mode':
                self.imputer = SimpleImputer(strategy='most_frequent')
            elif self.method == 'constant':
                self.imputer = SimpleImputer(strategy='constant', fill_value=self.fill_value)
            elif self.method == 'knn':
                try:
                    from sklearn.impute import KNNImputer
                    self.imputer = KNNImputer(n_neighbors=self.knn_neighbors)
                except ImportError:
                    self.logger.warning("Không thể import KNNImputer. Chuyển sang sử dụng phương pháp fallback.")
                    self.method = self._use_fallback_method()
            elif self.method == 'iterative':
                try:
                    from sklearn.experimental import enable_iterative_imputer
                    from sklearn.impute import IterativeImputer
                    self.imputer = IterativeImputer(max_iter=self.max_iter, random_state=42)
                except ImportError:
                    self.logger.warning("Không thể import IterativeImputer. Chuyển sang sử dụng phương pháp fallback.")
                    self.method = self._use_fallback_method()
                    
        except ImportError:
            self.logger.warning("Không thể import sklearn. Chuyển sang sử dụng phương pháp fallback.")
            self.method = self._use_fallback_method()
    
    def _use_fallback_method(self) -> str:
        """
        Chọn phương pháp fallback đầu tiên từ danh sách.
        
        Returns:
            Phương pháp fallback
        """
        if not self.fallback_strategy:
            self.logger.warning("Không có fallback strategy nào được định nghĩa. Sử dụng 'interpolate' làm mặc định.")
            return 'interpolate'
        
        fallback = self.fallback_strategy[0]
        self.logger.info(f"Sử dụng phương pháp fallback: {fallback}")
        
        # Nếu fallback cũng cần imputer, thử khởi tạo lại
        if fallback in ['mean', 'median', 'mode', 'constant']:
            try:
                from sklearn.impute import SimpleImputer
                
                if fallback == 'mean':
                    self.imputer = SimpleImputer(strategy='mean')
                elif fallback == 'median':
                    self.imputer = SimpleImputer(strategy='median')
                elif fallback == 'mode':
                    self.imputer = SimpleImputer(strategy='most_frequent')
                elif fallback == 'constant':
                    self.imputer = SimpleImputer(strategy='constant', fill_value=self.fill_value)
            except ImportError:
                # Nếu vẫn không import được sklearn, sử dụng phương pháp không cần sklearn
                self.logger.warning(f"Không thể khởi tạo imputer cho {fallback}. Sử dụng 'interpolate'.")
                return 'interpolate'
        
        return fallback
    
    def _encode_categorical_columns(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Chuyển đổi các cột categorical sang numeric để xử lý bằng phương pháp KNN hoặc Iterative.
        
        Args:
            df: DataFrame cần xử lý
            columns: Danh sách cột cần xử lý
            
        Returns:
            Tuple (DataFrame đã chuyển đổi, danh sách các cột đã chuyển đổi)
        """
        encoded_df = df.copy()
        encoded_columns = []
        
        for col in columns:
            if col not in encoded_df.columns:
                continue
                
            # Kiểm tra nếu cột là categorical hoặc object
            if encoded_df[col].dtype.name == 'category' or encoded_df[col].dtype.name == 'object':
                # Lưu kiểu dữ liệu gốc
                self.original_dtypes[col] = encoded_df[col].dtype
                
                # Nếu là categorical, lưu các categories
                if encoded_df[col].dtype.name == 'category':
                    self.category_mappings[col] = dict(enumerate(encoded_df[col].cat.categories))
                else:
                    # Nếu là object, tạo mapping tạm thời
                    unique_values = encoded_df[col].dropna().unique()
                    self.category_mappings[col] = dict(enumerate(unique_values))
                
                # Chuyển đổi sang numeric
                # Sử dụng factorize để xử lý các giá trị NA tốt hơn
                codes, uniques = pd.factorize(encoded_df[col])
                encoded_df[col] = codes
                
                encoded_columns.append(col)
                self.logger.debug(f"Đã chuyển đổi cột {col} từ {self.original_dtypes[col]} sang numeric")
        
        return encoded_df, encoded_columns
    
    def _decode_categorical_columns(self, df: pd.DataFrame, encoded_columns: List[str]) -> pd.DataFrame:
        """
        Khôi phục các cột categorical từ dạng numeric về dạng ban đầu.
        
        Args:
            df: DataFrame đã xử lý
            encoded_columns: Danh sách các cột đã được chuyển đổi
            
        Returns:
            DataFrame với các cột đã được khôi phục
        """
        decoded_df = df.copy()
        
        for col in encoded_columns:
            if col not in decoded_df.columns or col not in self.category_mappings:
                continue
            
            # Chuyển từ code sang giá trị gốc
            decoded_df[col] = decoded_df[col].map(lambda x: self.category_mappings[col].get(int(x)) if pd.notna(x) and int(x) in self.category_mappings[col] else None)
            
            # Khôi phục kiểu dữ liệu
            if col in self.original_dtypes:
                if self.original_dtypes[col].name == 'category':
                    decoded_df[col] = decoded_df[col].astype('category')
                else:
                    # Cố gắng chuyển về đúng kiểu ban đầu
                    try:
                        decoded_df[col] = decoded_df[col].astype(self.original_dtypes[col])
                    except Exception as e:
                        self.logger.warning(f"Không thể chuyển đổi cột {col} về kiểu {self.original_dtypes[col]}: {e}")
            
            self.logger.debug(f"Đã khôi phục cột {col} về {self.original_dtypes.get(col, 'original type')}")
        
        return decoded_df
    
    def _detect_time_series(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Phát hiện xem DataFrame có phải là dữ liệu chuỗi thời gian không.
        
        Args:
            df: DataFrame cần kiểm tra
            
        Returns:
            Tuple (is_time_series, timestamp_column)
        """
        if not self.auto_detect_time_series:
            return False, None
        
        # Tìm cột timestamp
        timestamp_col = None
        
        # Kiểm tra tên cột phổ biến
        common_ts_names = ['timestamp', 'time', 'date', 'datetime']
        for col in common_ts_names:
            if col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]) or self._try_convert_to_datetime(df[col]):
                    timestamp_col = col
                    break
        
        # Nếu không tìm thấy, kiểm tra tất cả các cột datetime
        if timestamp_col is None:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    timestamp_col = col
                    break
        
        # Nếu vẫn không tìm thấy, thử chuyển đổi các cột object
        if timestamp_col is None:
            for col in df.select_dtypes(include=['object']).columns:
                if self._try_convert_to_datetime(df[col]):
                    timestamp_col = col
                    break
        
        return timestamp_col is not None, timestamp_col
    
    def _try_convert_to_datetime(self, series: pd.Series) -> bool:
        """
        Thử chuyển đổi một Series thành datetime.
        
        Args:
            series: Series cần chuyển đổi
            
        Returns:
            True nếu có thể chuyển đổi, False nếu không
        """
        try:
            pd.to_datetime(series, errors='raise')
            return True
        except:
            return False
    
    # Thêm vào file missing_data_handler.py trong lớp MissingDataHandler
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: float = 0.5,
        timestamp_col: Optional[str] = None,
        # Thêm tham số mới
        handle_leading_nan: bool = True,
        leading_nan_method: str = 'backfill',
        aggressive_cleaning: bool = True,  # Tham số mới để xử lý triệt để
        ensure_no_nan: bool = True,        # Đảm bảo không còn NaN nào
        extra_methods: Optional[List[str]] = None  # Các phương pháp bổ sung
    ) -> pd.DataFrame:
        """
        Xử lý các giá trị thiếu trong DataFrame.
        
        Args:
            df: DataFrame cần xử lý
            columns: Danh sách cột cần xử lý (None để xử lý tất cả các cột)
            threshold: Ngưỡng tỷ lệ giá trị thiếu để loại bỏ cột (0.0 - 1.0)
            timestamp_col: Tên cột thời gian (None để tự động phát hiện)
            handle_leading_nan: Xử lý các giá trị NaN ở đầu dữ liệu
            leading_nan_method: Phương pháp xử lý NaN đầu ('drop', 'backfill', 'custom_value')
            aggressive_cleaning: Xử lý triệt để các giá trị NaN
            ensure_no_nan: Đảm bảo không còn giá trị NaN nào sau khi xử lý
            extra_methods: Các phương pháp bổ sung để xử lý NaN
            
        Returns:
            DataFrame đã xử lý
        """
        if df.empty:
            self.logger.warning("DataFrame rỗng, không có gì để xử lý")
            return df
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Xác định các cột cần xử lý
        if columns is None:
            # Lấy tất cả các cột
            columns = result_df.columns.tolist()
        else:
            # Chỉ lấy các cột tồn tại trong DataFrame
            columns = [col for col in columns if col in result_df.columns]
        
        # Tính tỷ lệ giá trị thiếu cho mỗi cột
        missing_ratio = result_df[columns].isnull().mean()
        
        # Loại bỏ các cột có quá nhiều giá trị thiếu
        if threshold < 1.0:
            cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
            if cols_to_drop:
                self.logger.warning(f"Loại bỏ {len(cols_to_drop)} cột có quá nhiều giá trị thiếu: {cols_to_drop}")
                result_df.drop(columns=cols_to_drop, inplace=True)
                columns = [col for col in columns if col not in cols_to_drop]
        
        # Nếu không còn cột nào để xử lý
        if not columns:
            self.logger.warning("Không có cột nào để xử lý sau khi áp dụng ngưỡng")
            return result_df

        # Xử lý giá trị NaN cho từng cột còn lại
        for col in columns:
            if result_df[col].isna().sum() > 0:
                self.logger.info(f"Xử lý NaN cho cột {col}")
                
                # Áp dụng phương pháp xử lý NaN triệt để
                result_df[col] = result_df[col].interpolate(method='linear', limit_direction='both').fillna(method='ffill').fillna(method='bfill')
                result_df[col] = result_df[col].fillna(result_df[col].mean())
                
                # Đảm bảo không còn NaN nếu self.ensure_no_nan=True
                if self.ensure_no_nan and result_df[col].isna().sum() > 0:
                    self.logger.warning(f"Vẫn còn {result_df[col].isna().sum()} giá trị NaN trong cột {col}. Đang thay bằng 0.")
                    result_df[col] = result_df[col].fillna(0)  # Thay bằng giá trị mặc định là 0
        
        # Đếm lại số NaN sau xử lý
        remaining_nans = result_df.isna().sum().sum()
        if remaining_nans > 0:
            self.logger.warning(f"Sau khi xử lý vẫn còn {remaining_nans} giá trị NaN trong DataFrame.")
        
            return result_df
        
        # Xử lý NaN ở đầu dữ liệu nếu cần
        if handle_leading_nan:
            result_df = self.handle_leading_nan(
                result_df, 
                columns=columns, 
                method=leading_nan_method, 
                min_periods=5  # Số giá trị tối thiểu để tính giá trị thay thế
            )
            self.logger.info(f"Đã xử lý các giá trị NaN ở đầu dữ liệu bằng phương pháp {leading_nan_method}")
        
        # Tự động phát hiện chuỗi thời gian nếu cần
        is_time_series = False
        if timestamp_col is None and self.auto_detect_time_series:
            is_time_series, detected_timestamp_col = self._detect_time_series(result_df)
            if is_time_series:
                timestamp_col = detected_timestamp_col
                self.logger.info(f"Đã phát hiện dữ liệu chuỗi thời gian với cột timestamp: {timestamp_col}")
        elif timestamp_col is not None:
            is_time_series = True
            self.logger.info(f"Sử dụng cột timestamp: {timestamp_col}")
        
        # Xử lý dữ liệu ban đầu
        if is_time_series and timestamp_col in result_df.columns:
            # Xử lý chuỗi thời gian
            self.logger.info("Áp dụng phương pháp xử lý chuỗi thời gian")
            result_df = self._handle_time_series_missing(result_df, columns, timestamp_col)
        else:
            # Xử lý dữ liệu tabular
            self.logger.info("Áp dụng phương pháp xử lý dữ liệu bảng")
            result_df = self._handle_tabular_missing(result_df, columns)
        
        # THÊM ĐOẠN CODE MỚI: Xử lý triệt để các giá trị NaN nếu cần
        if aggressive_cleaning:
            # Kiểm tra số lượng NaN sau khi xử lý ban đầu
            nan_count = result_df[columns].isna().sum().sum()
            
            if nan_count > 0:
                self.logger.info(f"Còn {nan_count} giá trị NaN sau khi xử lý ban đầu, tiến hành xử lý triệt để")
                
                # Xác định các cột số để xử lý
                numeric_cols = result_df[columns].select_dtypes(include=[np.number]).columns.tolist()
                
                # Sử dụng các phương pháp bổ sung nếu được cung cấp, hoặc sử dụng danh sách mặc định
                if extra_methods is None:
                    extra_methods = ['interpolate', 'ffill', 'bfill', 'mean']
                
                # Áp dụng từng phương pháp cho đến khi không còn NaN
                for method in extra_methods:
                    if method == 'interpolate':
                        # Nội suy tuyến tính
                        result_df[numeric_cols] = result_df[numeric_cols].interpolate(method='linear', limit_direction='both')
                    elif method == 'ffill':
                        # Forward fill
                        result_df[numeric_cols] = result_df[numeric_cols].fillna(method='ffill')
                    elif method == 'bfill':
                        # Backward fill
                        result_df[numeric_cols] = result_df[numeric_cols].fillna(method='bfill')
                    elif method == 'mean':
                        # Điền bằng giá trị trung bình
                        for col in numeric_cols:
                            if result_df[col].isna().any():
                                mean_val = result_df[col].mean()
                                if not np.isnan(mean_val):
                                    result_df[col] = result_df[col].fillna(mean_val)
                    
                    # Kiểm tra nếu đã xử lý hết NaN
                    nan_count = result_df[numeric_cols].isna().sum().sum()
                    if nan_count == 0:
                        self.logger.info(f"Đã xử lý hết các giá trị NaN sau khi áp dụng phương pháp {method}")
                        break
                
                # Xử lý các cột không phải số
                non_numeric_cols = [col for col in columns if col not in numeric_cols]
                if non_numeric_cols:
                    for col in non_numeric_cols:
                        if result_df[col].isna().any():
                            result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')
                            
                            # Nếu vẫn còn NaN, điền bằng giá trị phổ biến nhất
                            if result_df[col].isna().any() and len(result_df[col].dropna()) > 0:
                                mode_val = result_df[col].mode()[0]
                                result_df[col] = result_df[col].fillna(mode_val)
        
        # THÊM ĐOẠN CODE MỚI: Đảm bảo không còn giá trị NaN nào, thay thế các giá trị còn thiếu bằng các giá trị hợp lý
        if ensure_no_nan:
            # Kiểm tra còn giá trị NaN không
            remaining_nans = result_df[columns].isna().sum()
            total_nans = remaining_nans.sum()
            
            if total_nans > 0:
                self.logger.warning(f"Vẫn còn {total_nans} giá trị NaN sau khi xử lý, áp dụng biện pháp cuối cùng")
                
                # Xử lý theo từng cột
                for col, nan_count in remaining_nans.items():
                    if nan_count > 0:
                        # Xác định giá trị hợp lý cho từng loại cột
                        if col in ['open', 'high', 'low', 'close']:
                            # Điền giá trị cuối cùng đã biết cho các cột giá
                            if not result_df[col].isna().all():
                                last_known = result_df[col].dropna().iloc[-1]
                                result_df[col] = result_df[col].fillna(last_known)
                            else:
                                # Nếu cả cột đều là NaN, điền 0 (trường hợp hiếm)
                                result_df[col] = result_df[col].fillna(0)
                        elif col == 'volume':
                            # Điền 0 cho khối lượng
                            result_df[col] = result_df[col].fillna(0)
                        elif 'rsi' in col.lower():
                            # RSI giá trị nằm trong khoảng 0-100, điền 50 (trung lập)
                            result_df[col] = result_df[col].fillna(50)
                        elif 'macd' in col.lower():
                            # MACD điền 0 (không có xu hướng)
                            result_df[col] = result_df[col].fillna(0)
                        elif any(x in col.lower() for x in ['bb_', 'bollinger']):
                            # Bollinger Band điền giá trung bình
                            if 'middle' in col.lower():
                                # Middle band là SMA, điền trung bình toàn cục
                                result_df[col] = result_df[col].fillna(result_df[col].mean() if not np.isnan(result_df[col].mean()) else 0)
                            elif 'upper' in col.lower():
                                # Upper band điền giá trị lớn hơn middle
                                if 'bb_middle_20' in result_df.columns:
                                    middle_val = result_df['bb_middle_20'].mean()
                                    result_df[col] = result_df[col].fillna(middle_val * 1.05 if not np.isnan(middle_val) else 0)
                                else:
                                    result_df[col] = result_df[col].fillna(0)
                            elif 'lower' in col.lower():
                                # Lower band điền giá trị nhỏ hơn middle
                                if 'bb_middle_20' in result_df.columns:
                                    middle_val = result_df['bb_middle_20'].mean()
                                    result_df[col] = result_df[col].fillna(middle_val * 0.95 if not np.isnan(middle_val) else 0)
                                else:
                                    result_df[col] = result_df[col].fillna(0)
                            else:
                                # Các thành phần Bollinger khác điền 0
                                result_df[col] = result_df[col].fillna(0)
                        else:
                            # Các cột khác điền trung bình nếu là số, phổ biến nhất nếu là categorical
                            if pd.api.types.is_numeric_dtype(result_df[col]):
                                mean_val = result_df[col].mean()
                                if not np.isnan(mean_val):
                                    result_df[col] = result_df[col].fillna(mean_val)
                                else:
                                    result_df[col] = result_df[col].fillna(0)
                            else:
                                if len(result_df[col].dropna()) > 0:
                                    mode_val = result_df[col].mode()[0]
                                    result_df[col] = result_df[col].fillna(mode_val)
                                else:
                                    # Nếu không có giá trị hợp lệ, điền một giá trị đặc biệt
                                    if pd.api.types.is_string_dtype(result_df[col]):
                                        result_df[col] = result_df[col].fillna("unknown")
                                    else:
                                        result_df[col] = result_df[col].fillna(0)
                
                # Kiểm tra lại sau khi xử lý
                final_nans = result_df[columns].isna().sum().sum()
                if final_nans > 0:
                    self.logger.error(f"Vẫn còn {final_nans} giá trị NaN sau khi áp dụng biện pháp cuối cùng!")
                else:
                    self.logger.info("Đã xử lý triệt để tất cả các giá trị NaN")
        
        return result_df
        
    # Thêm vào sau phương thức handle_missing_values hiện có
    def handle_leading_nan(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'backfill',
        min_periods: int = 5  # Số lượng giá trị tối thiểu cần thiết để bắt đầu điền
    ) -> pd.DataFrame:
        """
        Xử lý riêng các giá trị NaN ở đầu dữ liệu cho chỉ báo kỹ thuật.
    
        Args:
            df: DataFrame cần xử lý
            columns: Danh sách cột cần xử lý (None để xử lý tất cả)
            method: Phương pháp xử lý ('backfill', 'zero', 'mean', 'median')
            min_periods: Số lượng giá trị tối thiểu không phải NaN để tính giá trị thay thế
        
        Returns:
            DataFrame đã xử lý
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
        result_df = df.copy()
    
        for col in columns:
            if col not in result_df.columns:
                continue
            
            # Kiểm tra xem có giá trị NaN ở đầu không
            leading_na_count = 0
            for i in range(len(result_df)):
                if pd.isna(result_df[col].iloc[i]):
                    leading_na_count += 1
                else:
                    break
                
            if leading_na_count == 0:
                continue
            
            # Áp dụng phương pháp xử lý tương ứng
            if method == 'backfill':
                # Lấy giá trị đầu tiên không phải NaN để điền
                if leading_na_count < len(result_df):
                    first_valid = result_df[col].iloc[leading_na_count]
                    result_df.loc[result_df.index[:leading_na_count], col] = first_valid
            elif method == 'zero':
                # Điền 0 vào các giá trị NaN ở đầu
                result_df.loc[result_df.index[:leading_na_count], col] = 0
            elif method == 'mean':
                # Tính giá trị trung bình của min_periods giá trị đầu tiên không phải NaN
                if leading_na_count + min_periods <= len(result_df):
                    valid_values = result_df[col].iloc[leading_na_count:leading_na_count+min_periods]
                    mean_value = valid_values.mean()
                    result_df.loc[result_df.index[:leading_na_count], col] = mean_value
            elif method == 'median':
                # Tính giá trị trung vị của min_periods giá trị đầu tiên không phải NaN
                if leading_na_count + min_periods <= len(result_df):
                    valid_values = result_df[col].iloc[leading_na_count:leading_na_count+min_periods]
                    median_value = valid_values.median()
                    result_df.loc[result_df.index[:leading_na_count], col] = median_value
                
            # Log kết quả
            self.logger.debug(f"Đã xử lý {leading_na_count} giá trị NaN ở đầu cho cột {col} bằng phương pháp {method}")
    
        return result_df
    
    def _handle_tabular_missing(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Xử lý dữ liệu thiếu cho dữ liệu dạng bảng.
        
        Args:
            df: DataFrame cần xử lý
            columns: Danh sách cột cần xử lý
            
        Returns:
            DataFrame đã xử lý
        """
        result_df = df.copy()
        
        # Xử lý các giá trị thiếu dựa trên phương pháp đã chọn
        if self.method == 'drop':
            # Loại bỏ các hàng có giá trị thiếu
            original_len = len(result_df)
            result_df = result_df.dropna(subset=columns)
            new_len = len(result_df)
            self.logger.info(f"Đã loại bỏ {original_len - new_len} hàng có giá trị thiếu")
        
        elif self.method in ['mean', 'median', 'mode', 'constant', 'knn', 'iterative']:
            # Sử dụng imputer
            try:
                # Chia thành các cột số và không phải số
                numeric_cols = result_df[columns].select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = [col for col in columns if col not in numeric_cols]
                
                # Xử lý riêng biệt cho từng loại dữ liệu
                if numeric_cols:
                    # Áp dụng imputer cho các cột số
                    result_df = self._apply_imputer(result_df, numeric_cols, [])
                
                # Xử lý các cột categorical
                if categorical_cols:
                    result_df = self._handle_categorical_missing(result_df, categorical_cols)
                
            except Exception as e:
                self.logger.error(f"Lỗi khi sử dụng imputer: {e}")
                # Sử dụng phương pháp dự phòng
                prev_method = self.method
                self.method = self._use_fallback_method()
                
                if self.method != prev_method:
                    self.logger.info(f"Chuyển sang phương pháp fallback: {self.method}")
                    return self._handle_tabular_missing(df, columns)
                else:
                    # Nếu không thể chuyển sang phương pháp khác
                    self._interpolate_missing(result_df, columns)
        
        elif self.method == 'ffill':
            # Forward fill
            result_df[columns] = result_df[columns].ffill()
            self.logger.info(f"Đã điền giá trị thiếu bằng phương pháp forward fill")
        
        elif self.method == 'bfill':
            # Backward fill
            result_df[columns] = result_df[columns].bfill()
            self.logger.info(f"Đã điền giá trị thiếu bằng phương pháp backward fill")
        
        elif self.method == 'interpolate':
            # Nội suy tuyến tính
            self._interpolate_missing(result_df, columns)
        
        # Kiểm tra xem còn giá trị thiếu không
        remaining_missing = result_df[columns].isnull().sum().sum()
        if remaining_missing > 0:
            # Có thể còn giá trị thiếu ở đầu hoặc cuối các chuỗi
            # Sử dụng ffill và bfill để xử lý
            result_df[columns] = result_df[columns].fillna(method='ffill').fillna(method='bfill')
            
            final_missing = result_df[columns].isnull().sum().sum()
            if final_missing > 0:
                self.logger.warning(f"Vẫn còn {final_missing} giá trị thiếu sau khi xử lý")
            else:
                self.logger.info("Đã xử lý tất cả các giá trị thiếu")
        else:
            self.logger.info("Đã xử lý tất cả các giá trị thiếu")
        
        return result_df
    
    def _handle_categorical_missing(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Xử lý dữ liệu thiếu cho các cột categorical.
        
        Args:
            df: DataFrame cần xử lý
            categorical_cols: Danh sách cột categorical cần xử lý
            
        Returns:
            DataFrame đã xử lý
        """
        result_df = df.copy()
        
        if self.method == 'mode':
            # Mode imputer works for categorical data
            categorical_imputer = None
            try:
                from sklearn.impute import SimpleImputer
                categorical_imputer = SimpleImputer(strategy='most_frequent')
            except ImportError:
                self.logger.warning("Không thể import SimpleImputer. Sử dụng phương pháp thay thế.")
                
            if categorical_imputer:
                for col in categorical_cols:
                    if result_df[col].isnull().any():
                        # Chuyển đổi thành category để xử lý
                        is_categorical = result_df[col].dtype.name == 'category'
                        original_dtype = result_df[col].dtype
                        
                        # Transform
                        col_values = result_df[col].values.reshape(-1, 1)
                        imputed_col = categorical_imputer.fit_transform(col_values)
                        result_df[col] = imputed_col.flatten()
                        
                        # Khôi phục kiểu dữ liệu
                        if is_categorical:
                            result_df[col] = result_df[col].astype(original_dtype)
            else:
                # Fallback to mode calculation
                for col in categorical_cols:
                    if result_df[col].isnull().any():
                        mode_val = result_df[col].mode()[0]
                        result_df[col].fillna(mode_val, inplace=True)
        
        elif self.method in ['knn', 'iterative'] and self.handle_categorical:
            # Encode categorical columns for KNN or Iterative imputer
            encoded_df, encoded_columns = self._encode_categorical_columns(result_df, categorical_cols)
            
            if encoded_columns:
                # Apply imputer to encoded columns
                encoded_df = self._apply_imputer(encoded_df, encoded_columns, [])
                
                # Decode back to original types
                result_df = self._decode_categorical_columns(encoded_df, encoded_columns)
        
        else:
            # For other methods, use ffill and bfill
            self.logger.info(f"Phương pháp {self.method} không áp dụng trực tiếp cho dữ liệu categorical. Sử dụng ffill/bfill.")
            for col in categorical_cols:
                # Apply forward fill and backward fill
                result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')
                
                # If still missing, use mode
                if result_df[col].isnull().any():
                    if len(result_df[col].dropna()) > 0:
                        mode_val = result_df[col].mode()[0]
                        result_df[col].fillna(mode_val, inplace=True)
        
        return result_df
    
    def _apply_imputer(self, df: pd.DataFrame, columns: List[str], 
                       encoded_columns: List[str]) -> pd.DataFrame:
        """
        Áp dụng imputer cho DataFrame.
        
        Args:
            df: DataFrame cần xử lý
            columns: Danh sách cột cần xử lý
            encoded_columns: Danh sách cột đã encode (để không bị decode lại)
            
        Returns:
            DataFrame đã xử lý
        """
        result_df = df.copy()
        
        if not self.imputer:
            self.logger.warning("Imputer chưa được khởi tạo. Sử dụng phương pháp thay thế.")
            self._interpolate_missing(result_df, columns)
            return result_df
        
        # Impute missing values
        try:
            imputed_values = self.imputer.fit_transform(result_df[columns])
            result_df[columns] = imputed_values
            self.logger.info(f"Đã điền {len(columns)} cột bằng phương pháp {self.method}")
        except Exception as e:
            self.logger.error(f"Lỗi khi áp dụng imputer: {e}")
            # Fallback to interpolate
            self._interpolate_missing(result_df, columns)
        
        return result_df
    
    def _handle_time_series_missing(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        timestamp_col: str
    ) -> pd.DataFrame:
        """
        Xử lý dữ liệu thiếu cho dữ liệu chuỗi thời gian.
        
        Args:
            df: DataFrame cần xử lý
            columns: Danh sách cột cần xử lý
            timestamp_col: Tên cột thời gian
            
        Returns:
            DataFrame đã xử lý
        """
        # Sắp xếp dữ liệu theo thời gian
        result_df = df.copy()
        
        # Đảm bảo timestamp là datetime
        if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
            result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])
        
        # Sắp xếp theo thời gian
        result_df = result_df.sort_values(timestamp_col)
        
        # Xử lý các giá trị thiếu dựa trên phương pháp đã chọn
        if self.method == 'interpolate':
            # Phương pháp nội suy tốt nhất cho chuỗi thời gian
            for col in columns:
                if col != timestamp_col and result_df[col].isnull().any():
                    # Áp dụng nội suy phù hợp với kiểu dữ liệu
                    if pd.api.types.is_numeric_dtype(result_df[col]):
                        result_df[col] = result_df[col].interpolate(method='time')
                    else:
                        # Với categorical data, sử dụng ffill/bfill
                        result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info(f"Đã nội suy dữ liệu chuỗi thời gian theo phương pháp time")
        
        elif self.method in ['ffill', 'bfill']:
            # Sử dụng forward fill hoặc backward fill
            fill_method = 'ffill' if self.method == 'ffill' else 'bfill'
            result_df[columns] = result_df[columns].fillna(method=fill_method)
            
            # Sau đó sử dụng phương pháp còn lại để xử lý các giá trị ở đầu hoặc cuối
            opposite_method = 'bfill' if self.method == 'ffill' else 'ffill'
            result_df[columns] = result_df[columns].fillna(method=opposite_method)
            
            self.logger.info(f"Đã điền dữ liệu chuỗi thời gian theo phương pháp {self.method}")
        
        elif self.method in ['mean', 'median', 'mode', 'constant', 'knn', 'iterative']:
            # Chia theo loại dữ liệu
            numeric_cols = [col for col in columns if col != timestamp_col and 
                           pd.api.types.is_numeric_dtype(result_df[col])]
            categorical_cols = [col for col in columns if col != timestamp_col and 
                               not pd.api.types.is_numeric_dtype(result_df[col])]
            
            # Xử lý dữ liệu số
            if numeric_cols:
                try:
                    # Áp dụng imputer cho các cột số
                    result_df = self._apply_imputer(result_df, numeric_cols, [])
                except Exception as e:
                    self.logger.error(f"Lỗi khi xử lý cột số: {e}")
                    # Fallback to time interpolation
                    for col in numeric_cols:
                        result_df[col] = result_df[col].interpolate(method='time')
            
            # Xử lý dữ liệu categorical
            if categorical_cols:
                result_df = self._handle_categorical_missing(result_df, categorical_cols)
        
        elif self.method == 'drop':
            # Loại bỏ các hàng có giá trị thiếu
            original_len = len(result_df)
            result_df = result_df.dropna(subset=[col for col in columns if col != timestamp_col])
            new_len = len(result_df)
            self.logger.info(f"Đã loại bỏ {original_len - new_len} hàng có giá trị thiếu")
        
        # Kiểm tra xem còn giá trị thiếu không
        remaining_missing = result_df[[col for col in columns if col != timestamp_col]].isnull().sum().sum()
        if remaining_missing > 0:
            # Có thể còn giá trị thiếu ở đầu hoặc cuối các chuỗi thời gian
            # Sử dụng ffill và bfill để xử lý
            fillable_cols = [col for col in columns if col != timestamp_col]
            result_df[fillable_cols] = result_df[fillable_cols].fillna(method='ffill').fillna(method='bfill')
            
            final_missing = result_df[fillable_cols].isnull().sum().sum()
            if final_missing > 0:
                self.logger.warning(f"Vẫn còn {final_missing} giá trị thiếu sau khi xử lý")
            else:
                self.logger.info("Đã xử lý tất cả các giá trị thiếu")
        else:
            self.logger.info("Đã xử lý tất cả các giá trị thiếu")
        
        return result_df
    
    def _interpolate_missing(self, df: pd.DataFrame, columns: List[str]) -> None:
        """
        Nội suy các giá trị thiếu.
        
        Args:
            df: DataFrame cần xử lý
            columns: Danh sách cột cần xử lý
        """
        # Xác định các cột số
        numeric_cols = df[columns].select_dtypes(include=[np.number]).columns.tolist()
        
        # Nội suy tuyến tính cho các cột số
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            self.logger.info(f"Đã nội suy {len(numeric_cols)} cột số")
        
        # Xử lý các cột không phải số
        non_numeric_cols = [col for col in columns if col not in numeric_cols]
        if non_numeric_cols:
            for col in non_numeric_cols:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            self.logger.info(f"Đã xử lý {len(non_numeric_cols)} cột không phải số bằng ffill/bfill")
    
    def get_missing_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Lấy thống kê về dữ liệu thiếu trong DataFrame.
        
        Args:
            df: DataFrame cần kiểm tra
            
        Returns:
            Dict với thống kê về dữ liệu thiếu
        """
        # Tính tổng số giá trị thiếu
        missing_count = df.isnull().sum().sum()
        
        # Tính tỷ lệ giá trị thiếu
        missing_ratio = missing_count / (df.shape[0] * df.shape[1]) if df.size > 0 else 0
        
        # Tính số lượng giá trị thiếu theo cột
        missing_by_column = df.isnull().sum().to_dict()
        
        # Tính tỷ lệ giá trị thiếu theo cột
        missing_ratio_by_column = (df.isnull().mean() * 100).to_dict()
        
        # Tính số lượng hàng có ít nhất một giá trị thiếu
        rows_with_missing = df.isnull().any(axis=1).sum()
        
        # Tính tỷ lệ hàng có ít nhất một giá trị thiếu
        rows_missing_ratio = rows_with_missing / df.shape[0] if df.shape[0] > 0 else 0
        
        # Kiểm tra nếu DataFrame là chuỗi thời gian
        is_time_series, timestamp_col = self._detect_time_series(df)
        
        # Tạo kết quả
        result = {
            'total_values': df.size,
            'missing_values': int(missing_count),
            'missing_percentage': float(missing_ratio * 100),
            'rows_with_missing': int(rows_with_missing),
            'rows_missing_percentage': float(rows_missing_ratio * 100),
            'missing_by_column': {col: int(count) for col, count in missing_by_column.items()},
            'missing_percentage_by_column': {col: float(ratio) for col, ratio in missing_ratio_by_column.items()},
            'is_time_series': is_time_series,
            'timestamp_column': timestamp_col if is_time_series else None,
            'primary_method': self.method,
            'fallback_methods': self.fallback_strategy
        }
        
        return result
    
    def visualize_missing_data(self, df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Tạo đồ thị trực quan hóa dữ liệu thiếu (missingno matrix).
        
        Args:
            df: DataFrame cần trực quan hóa
            figsize: Kích thước hình
        """
        try:
            import matplotlib.pyplot as plt
            import missingno as msno
            
            plt.figure(figsize=figsize)
            msno.matrix(df)
            plt.title('Ma trận dữ liệu thiếu')
            plt.tight_layout()
            plt.show()
            
            plt.figure(figsize=figsize)
            msno.bar(df)
            plt.title('Số lượng giá trị thiếu theo cột')
            plt.tight_layout()
            plt.show()
            
            plt.figure(figsize=figsize)
            msno.heatmap(df)
            plt.title('Tương quan dữ liệu thiếu giữa các cột')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("Không thể import matplotlib hoặc missingno. Không thể trực quan hóa.")
    
    def fill_time_series_gaps(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        freq: str = None,
        method: str = 'linear'
    ) -> pd.DataFrame:
        """
        Điền các khoảng trống trong dữ liệu chuỗi thời gian.
        
        Args:
            df: DataFrame cần xử lý
            timestamp_col: Tên cột thời gian
            freq: Tần suất dữ liệu ('1min', '1h', '1d', ...)
            method: Phương pháp nội suy ('linear', 'time', 'nearest', 'zero', 'slinear', 
                   'quadratic', 'cubic', 'spline', 'barycentric', 'polynomial')
            
        Returns:
            DataFrame đã điền khoảng trống
        """
        if df.empty:
            self.logger.warning("DataFrame rỗng, không có gì để xử lý")
            return df
        
        # Kiểm tra cột timestamp
        if timestamp_col not in df.columns:
            self.logger.error(f"Không tìm thấy cột {timestamp_col} trong DataFrame")
            return df
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Đảm bảo cột timestamp là datetime
        if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
            try:
                result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])
            except Exception as e:
                self.logger.error(f"Không thể chuyển đổi cột {timestamp_col} thành datetime: {e}")
                return df
        
        # Sắp xếp theo thời gian
        result_df = result_df.sort_values(timestamp_col)
        
        # Nếu không có freq, ước tính từ dữ liệu
        if freq is None:
            # Tính thời gian trung bình giữa các bản ghi
            time_diffs = result_df[timestamp_col].diff().dropna()
            
            if len(time_diffs) > 0:
                median_diff = time_diffs.median()
                # Chuyển đổi timedelta thành string freq
                total_seconds = median_diff.total_seconds()
                
                if total_seconds < 60:
                    freq = f"{int(round(total_seconds))}S"  # seconds
                elif total_seconds < 3600:
                    freq = f"{int(round(total_seconds / 60))}min"  # minutes
                elif total_seconds < 86400:
                    freq = f"{int(round(total_seconds / 3600))}H"  # hours
                else:
                    freq = f"{int(round(total_seconds / 86400))}D"  # days
                
                self.logger.info(f"Ước tính tần suất dữ liệu: {freq}")
            else:
                self.logger.warning("Không thể ước tính tần suất dữ liệu")
                return df
        
        # Tạo index thời gian đầy đủ
        full_index = pd.date_range(
            start=result_df[timestamp_col].min(),
            end=result_df[timestamp_col].max(),
            freq=freq
        )
        
        # Đặt timestamp làm index
        result_df = result_df.set_index(timestamp_col)
        
        # Reindex với đầy đủ các điểm thời gian
        result_df = result_df.reindex(full_index)
        
        # Nội suy các giá trị thiếu
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            result_df[numeric_cols] = result_df[numeric_cols].interpolate(method=method)
        
        # Điền các giá trị còn thiếu cho các cột khác
        non_numeric_cols = [col for col in result_df.columns if col not in numeric_cols]
        if non_numeric_cols:
            result_df[non_numeric_cols] = result_df[non_numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        # Reset index để đưa timestamp trở lại thành cột
        result_df = result_df.reset_index()
        result_df = result_df.rename(columns={'index': timestamp_col})
        
        self.logger.info(f"Đã điền {len(full_index) - len(df)} khoảng trống trong dữ liệu chuỗi thời gian")
        
        return result_df

    # Thêm phương thức mới để tạo cột target cho học có giám sát
    def create_target_columns(
        self,
        df: pd.DataFrame,
        price_column: str = 'close',
        target_types: List[str] = ['direction', 'return', 'volatility'],
        horizons: List[int] = [1, 3, 5, 10],
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Tạo các cột mục tiêu cho huấn luyện có giám sát.
    
        Args:
            df: DataFrame gốc
            price_column: Tên cột giá sử dụng làm cơ sở
            target_types: Loại mục tiêu ('direction', 'return', 'volatility')
            horizons: Các khung thời gian tương lai
            threshold: Ngưỡng cho target direction
        
        Returns:
            DataFrame có cột mục tiêu
        """
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        target_df = df.copy()
    
        # Tạo các cột mục tiêu
        target_columns_created = []
    
        for horizon in horizons:
            # Loại 1: Target hướng (lên/xuống)
            if 'direction' in target_types:
                col_name = f'target_direction_{horizon}'
                # Tính phần trăm thay đổi giá
                pct_change = target_df[price_column].pct_change(periods=horizon).shift(-horizon)
            
                # Áp dụng ngưỡng
                if threshold > 0:
                    # 1: tăng vượt ngưỡng, -1: giảm vượt ngưỡng, 0: dao động trong ngưỡng
                    target_df[col_name] = 0
                    target_df.loc[pct_change > threshold, col_name] = 1
                    target_df.loc[pct_change < -threshold, col_name] = -1
                else:
                    # Không có ngưỡng: 1 nếu tăng, 0 nếu giảm
                    target_df[col_name] = (pct_change > 0).astype(int)
            
                target_columns_created.append(col_name)
        
            # Loại 2: Target lợi nhuận (% thay đổi)
            if 'return' in target_types:
                col_name = f'target_return_{horizon}'
                target_df[col_name] = target_df[price_column].pct_change(periods=horizon).shift(-horizon)
                target_columns_created.append(col_name)
        
            # Loại 3: Target biến động
            if 'volatility' in target_types:
                col_name = f'target_volatility_{horizon}'
                # Tính biến động như độ lệch chuẩn của lợi nhuận
                rolling_returns = target_df[price_column].pct_change().rolling(window=horizon).std().shift(-horizon)
                target_df[col_name] = rolling_returns
                target_columns_created.append(col_name)
    
        self.logger.info(f"Đã tạo {len(target_columns_created)} cột mục tiêu: {', '.join(target_columns_created)}")
    
        return target_df

    # Thêm phương thức mới để xử lý vấn đề chỉ báo trùng thông tin
    def remove_redundant_features(
        self, 
        df: pd.DataFrame,
        correlation_threshold: float = 0.95,
        redundant_groups: Optional[List[List[str]]] = None,
        keep_representative: bool = True
    ) -> pd.DataFrame:
        """
        Loại bỏ các chỉ báo kỹ thuật trùng lặp thông tin.
    
        Args:
            df: DataFrame cần xử lý
            correlation_threshold: Ngưỡng tương quan để xác định đặc trưng trùng lặp
            redundant_groups: Danh sách các nhóm đặc trưng đã biết là trùng lặp
            keep_representative: Giữ lại một đại diện cho mỗi nhóm trùng lặp
        
        Returns:
            DataFrame đã loại bỏ chỉ báo trùng lặp
        """
        # Thiết lập nhóm mặc định nếu không cung cấp
        if redundant_groups is None:
                redundant_groups = [
                # MACD Group - giữ lại macd_histogram vì nó là sự kết hợp của hai chỉ báo khác
                ['macd_line', 'macd_signal', 'macd_histogram'],
            
                # ATR Group - giữ lại atr_pct_14 vì nó đã được chuẩn hóa theo phần trăm
                ['atr_14', 'atr_pct_14', 'atr_norm_14'],
            
                # Bollinger Bands Group - giữ lại bb_percent_b_20 vì nó cung cấp thông tin vị trí tương đối
                ['bb_middle_20', 'sma_20', 'bb_upper_20', 'bb_lower_20', 'bb_percent_b_20'],
            
                # Directional Movement Group - giữ lại adx_14 vì nó là chỉ báo tổng hợp
                ['plus_di_14', 'minus_di_14', 'adx_14']
            ]
    
        # Quy tắc chọn đại diện cho mỗi nhóm
        group_representatives = {
            'macd': 'macd_histogram',
            'atr': 'atr_pct_14',
            'bollinger': 'bb_percent_b_20',
            'directional': 'adx_14',
            'supertrend': 'supertrend_trend_10_3.0',
            'ema': 'ema_14'
        }
    
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        pruned_df = df.copy()
        columns_to_drop = []
    
        # Loại bỏ chỉ báo trùng lặp theo nhóm đã xác định
        if redundant_groups:
            for group in redundant_groups:
                # Kiểm tra xem nhóm có tồn tại trong DataFrame không
                existing_cols = [col for col in group if col in pruned_df.columns]
                if len(existing_cols) <= 1:
                    continue
            
                # Xác định chỉ báo đại diện
                if keep_representative:
                    # Tìm nhóm chỉ báo
                    group_key = None
                    for key, representative in group_representatives.items():
                        if any(key in col for col in existing_cols):
                            group_key = key
                            break
                
                    # Nếu tìm thấy đại diện, giữ lại
                    if group_key and group_representatives[group_key] in existing_cols:
                        representative = group_representatives[group_key]
                        cols_to_drop = [col for col in existing_cols if col != representative]
                    else:
                        # Nếu không, giữ lại cột đầu tiên
                        representative = existing_cols[0]
                        cols_to_drop = existing_cols[1:]
                else:
                    # Loại bỏ tất cả các cột trong nhóm
                    cols_to_drop = existing_cols
            
                columns_to_drop.extend(cols_to_drop)
    
        # Nếu cần, thực hiện phát hiện tương quan
        if correlation_threshold < 1.0:
            numeric_cols = [col for col in pruned_df.columns if col not in columns_to_drop 
                            and pd.api.types.is_numeric_dtype(pruned_df[col])]
        
            # Tính ma trận tương quan
            if len(numeric_cols) > 1:
                corr_matrix = pruned_df[numeric_cols].corr().abs()
            
                # Tìm các cặp cột có tương quan cao
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                corr_cols_to_drop = []
            
                for col in upper_tri.columns:
                    # Lấy các cột có tương quan cao với cột hiện tại
                    high_corr = upper_tri.index[upper_tri[col] > correlation_threshold].tolist()
                
                    if high_corr:
                        # Tìm nhóm chứa cột hiện tại
                        group_key = None
                        for key in group_representatives:
                            if key in col.lower():
                                group_key = key
                                break
                    
                        # Nếu cột là đại diện của một nhóm, không loại bỏ nó
                        if group_key and col == group_representatives.get(group_key):
                            # Thêm các cột khác vào danh sách loại bỏ
                            corr_cols_to_drop.extend(high_corr)
                        else:
                            # Ưu tiên giữ cột đại diện nếu có trong danh sách
                            rep_cols = [c for c in high_corr if any(c == group_representatives.get(key) for key in group_representatives)]
                            if rep_cols:
                                # Nếu có cột đại diện, loại bỏ cột hiện tại
                                corr_cols_to_drop.append(col)
                            else:
                                # Nếu không, loại bỏ các cột khác
                                corr_cols_to_drop.extend(high_corr)
            
                # Loại bỏ các cột trùng lặp
                columns_to_drop.extend(list(set(corr_cols_to_drop)))
    
        # Loại bỏ các cột trùng lặp
        if columns_to_drop:
            columns_to_drop = list(set(columns_to_drop))  # Loại bỏ trùng lặp
            pruned_df = pruned_df.drop(columns=columns_to_drop)
    
        return pruned_df

class MissingValueMethod:
    """
    Enum-like class để định nghĩa các phương pháp xử lý giá trị thiếu.
    """
    MEAN = 'mean'
    MEDIAN = 'median'
    MODE = 'mode'
    CONSTANT = 'constant'
    FFILL = 'ffill'
    BFILL = 'bfill'
    INTERPOLATE = 'interpolate'
    KNN = 'knn'
    ITERATIVE = 'iterative'
    DROP = 'drop'
    
    @classmethod
    def get_all_methods(cls) -> List[str]:
        """Lấy danh sách tất cả các phương pháp."""
        return [cls.MEAN, cls.MEDIAN, cls.MODE, cls.CONSTANT, 
                cls.FFILL, cls.BFILL, cls.INTERPOLATE, 
                cls.KNN, cls.ITERATIVE, cls.DROP]
    
    @classmethod
    def get_statistical_methods(cls) -> List[str]:
        """Lấy danh sách các phương pháp thống kê."""
        return [cls.MEAN, cls.MEDIAN, cls.MODE, cls.CONSTANT]
    
    @classmethod
    def get_time_series_methods(cls) -> List[str]:
        """Lấy danh sách các phương pháp dành cho chuỗi thời gian."""
        return [cls.FFILL, cls.BFILL, cls.INTERPOLATE]
    
    @classmethod
    def get_advanced_methods(cls) -> List[str]:
        """Lấy danh sách các phương pháp nâng cao."""
        return [cls.KNN, cls.ITERATIVE]