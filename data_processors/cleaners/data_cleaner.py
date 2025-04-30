"""
Làm sạch dữ liệu thị trường.
File này cung cấp các lớp và phương thức để làm sạch dữ liệu thị trường,
bao gồm việc loại bỏ nhiễu, chuẩn hóa, và chuyển đổi dữ liệu.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from pathlib import Path

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import setup_logger
from config.constants import ErrorCode
from data_processors.cleaners.outlier_detector import OutlierDetector
from data_processors.cleaners.missing_data_handler import MissingDataHandler

class DataCleaner:
    """
    Lớp chính để làm sạch dữ liệu thị trường.
    """
    
    def __init__(
        self,
        use_outlier_detection: bool = True,
        use_missing_data_handler: bool = True,
        normalize_method: str = 'z-score',  # 'z-score', 'min-max', 'robust'
        outlier_detector_kwargs: Dict = None,
        missing_data_handler_kwargs: Dict = None
    ):
        """
        Khởi tạo bộ làm sạch dữ liệu.
        
        Args:
            use_outlier_detection: Sử dụng phát hiện ngoại lệ
            use_missing_data_handler: Sử dụng xử lý dữ liệu thiếu
            normalize_method: Phương pháp chuẩn hóa mặc định ('z-score', 'min-max', 'robust')
            outlier_detector_kwargs: Tham số cho OutlierDetector
            missing_data_handler_kwargs: Tham số cho MissingDataHandler
        """
        self.logger = setup_logger("data_cleaner")
        
        # Khởi tạo các thành phần con
        self.use_outlier_detection = use_outlier_detection
        self.use_missing_data_handler = use_missing_data_handler
        self.normalize_method = normalize_method
        
        if self.use_outlier_detection:
            outlier_kwargs = outlier_detector_kwargs or {}
            self.outlier_detector = OutlierDetector(**outlier_kwargs)
            
        if self.use_missing_data_handler:
            missing_kwargs = missing_data_handler_kwargs or {}
            self.missing_data_handler = MissingDataHandler(**missing_kwargs)
        
        self.logger.info(f"Đã khởi tạo DataCleaner với normalize_method={normalize_method}")
        
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Tải dữ liệu từ file.
        
        Args:
            file_path: Đường dẫn đến file dữ liệu (hỗ trợ CSV, Parquet)
        
        Returns:
            DataFrame chứa dữ liệu
        """
        file_path = Path(file_path)
        
        try:
            if file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
                self.logger.info(f"Đã tải {len(df)} dòng dữ liệu từ {file_path}")
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                self.logger.info(f"Đã tải {len(df)} dòng dữ liệu từ {file_path}")
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {file_path.suffix}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải dữ liệu từ {file_path}: {e}")
            raise

    def save_data(self, df: pd.DataFrame, file_path: Union[str, Path]) -> None:
        """
        Lưu DataFrame vào file.
        
        Args:
            df: DataFrame cần lưu
            file_path: Đường dẫn file đầu ra (hỗ trợ CSV, Parquet)
        """
        file_path = Path(file_path)
        
        try:
            # Đảm bảo thư mục đầu ra tồn tại
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.suffix.lower() == '.parquet':
                df.to_parquet(file_path, index=False)
                self.logger.info(f"Đã lưu {len(df)} dòng dữ liệu vào {file_path}")
            elif file_path.suffix.lower() == '.csv':
                df.to_csv(file_path, index=False)
                self.logger.info(f"Đã lưu {len(df)} dòng dữ liệu vào {file_path}")
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {file_path.suffix}")
                
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu dữ liệu vào {file_path}: {e}")
            raise
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        columns_to_clean: Optional[List[str]] = None,
        drop_na: bool = False,
        normalize: bool = False,
        remove_outliers: bool = True,
        handle_missing: bool = True,
        remove_duplicates: bool = True
    ) -> pd.DataFrame:
        """
        Làm sạch DataFrame.
        
        Args:
            df: DataFrame cần làm sạch
            columns_to_clean: Danh sách cột cần làm sạch (None để làm sạch tất cả)
            drop_na: Loại bỏ các hàng có giá trị NA
            normalize: Chuẩn hóa dữ liệu số
            remove_outliers: Loại bỏ các ngoại lệ
            handle_missing: Xử lý dữ liệu thiếu
            remove_duplicates: Loại bỏ các bản ghi trùng lặp
            
        Returns:
            DataFrame đã làm sạch
        """
        if df.empty:
            self.logger.warning("DataFrame rỗng, không có gì để làm sạch")
            return df
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        cleaned_df = df.copy()
        
        # Xác định các cột cần làm sạch
        if columns_to_clean is None:
            # Lấy tất cả các cột số
            columns_to_clean = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Chỉ lấy các cột tồn tại trong DataFrame
            columns_to_clean = [col for col in columns_to_clean if col in cleaned_df.columns]
        
        self.logger.info(f"Bắt đầu làm sạch {len(columns_to_clean)} cột")
        
        # Loại bỏ các bản ghi trùng lặp
        if remove_duplicates:
            original_len = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            new_len = len(cleaned_df)
            if original_len > new_len:
                self.logger.info(f"Đã loại bỏ {original_len - new_len} bản ghi trùng lặp")
        
        # Xử lý dữ liệu thiếu
        if handle_missing and self.use_missing_data_handler:
            self.logger.debug(f"Xử lý dữ liệu thiếu cho {len(columns_to_clean)} cột")
            cleaned_df = self.missing_data_handler.handle_missing_values(
                cleaned_df, columns=columns_to_clean
            )
        
        # Loại bỏ các ngoại lệ
        if remove_outliers and self.use_outlier_detection:
            self.logger.debug(f"Loại bỏ ngoại lệ cho {len(columns_to_clean)} cột")
            cleaned_df = self.outlier_detector.remove_outliers(
                cleaned_df, columns=columns_to_clean
            )
        
        # Chuẩn hóa dữ liệu
        if normalize:
            self.logger.debug(f"Chuẩn hóa dữ liệu cho {len(columns_to_clean)} cột")
            cleaned_df = self._normalize_columns(cleaned_df, columns_to_clean)
        
        # Loại bỏ các hàng có giá trị NA
        if drop_na:
            original_len = len(cleaned_df)
            cleaned_df = cleaned_df.dropna(subset=columns_to_clean)
            new_len = len(cleaned_df)
            if original_len > new_len:
                self.logger.info(f"Đã loại bỏ {original_len - new_len} hàng có giá trị NA")
        
        self.logger.info(f"Đã hoàn thành làm sạch dữ liệu, kết quả có {len(cleaned_df)} hàng")
        return cleaned_df
    
    def _normalize_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = None
    ) -> pd.DataFrame:
        """
        Chuẩn hóa các cột trong DataFrame.
        
        Args:
            df: DataFrame cần chuẩn hóa
            columns: Danh sách cột cần chuẩn hóa
            method: Phương pháp chuẩn hóa ('z-score', 'min-max', 'robust', 'decimal-scaling')
                   Nếu None, sử dụng giá trị từ constructor
            
        Returns:
            DataFrame đã chuẩn hóa
        """
        normalized_df = df.copy()
        
        # Sử dụng method từ constructor nếu không được chỉ định
        if method is None:
            method = self.normalize_method
        
        # Log số liệu thống kê trước khi chuẩn hóa
        stats_before = {}
        for col in columns:
            if col in normalized_df.columns:
                stats_before[col] = {
                    'mean': normalized_df[col].mean(),
                    'std': normalized_df[col].std(),
                    'min': normalized_df[col].min(),
                    'max': normalized_df[col].max()
                }
        
        self.logger.debug(f"Chuẩn hóa dữ liệu bằng phương pháp {method}")
        
        for col in columns:
            if col not in normalized_df.columns:
                self.logger.warning(f"Cột {col} không tồn tại trong DataFrame, bỏ qua")
                continue
                
            # Kiểm tra kiểu dữ liệu - chỉ chuẩn hóa cột số
            if not pd.api.types.is_numeric_dtype(normalized_df[col]):
                self.logger.warning(f"Cột {col} không phải kiểu số (kiểu {normalized_df[col].dtype}), bỏ qua")
                continue
            
            # Bỏ qua cột nếu tất cả là NA
            if normalized_df[col].isna().all():
                self.logger.warning(f"Cột {col} chỉ chứa giá trị NA, bỏ qua")
                continue
                
            # Z-score normalization: (x - mean) / std
            if method == 'z-score':
                mean = normalized_df[col].mean()
                std = normalized_df[col].std()
                if std != 0:
                    normalized_df[col] = (normalized_df[col] - mean) / std
                else:
                    self.logger.warning(f"Độ lệch chuẩn của cột {col} bằng 0, không thể chuẩn hóa z-score")
            
            # Min-max normalization: (x - min) / (max - min)
            elif method == 'min-max':
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val > min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                else:
                    self.logger.warning(f"Min và max của cột {col} bằng nhau, không thể chuẩn hóa min-max")
            
            # Robust normalization: (x - median) / IQR
            elif method == 'robust':
                median = normalized_df[col].median()
                q1 = normalized_df[col].quantile(0.25)
                q3 = normalized_df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    normalized_df[col] = (normalized_df[col] - median) / iqr
                else:
                    self.logger.warning(f"IQR của cột {col} bằng 0, không thể chuẩn hóa robust")
            
            # Decimal scaling: x / 10^j where j is smallest integer such that max(|x|) < 1
            elif method == 'decimal-scaling':
                max_abs = normalized_df[col].abs().max()
                if max_abs > 0:
                    j = int(np.log10(max_abs)) + 1
                    normalized_df[col] = normalized_df[col] / (10 ** j)
                else:
                    self.logger.warning(f"Giá trị max abs của cột {col} bằng 0, không thể chuẩn hóa decimal-scaling")
            
            else:
                self.logger.warning(f"Phương pháp chuẩn hóa không hợp lệ: {method}, bỏ qua cột {col}")
        
        # Log số liệu thống kê sau khi chuẩn hóa
        stats_after = {}
        for col in columns:
            if col in normalized_df.columns and col in stats_before:
                stats_after[col] = {
                    'mean': normalized_df[col].mean(),
                    'std': normalized_df[col].std(),
                    'min': normalized_df[col].min(),
                    'max': normalized_df[col].max()
                }
                
                self.logger.debug(f"Cột {col} - Trước: {stats_before[col]} - Sau: {stats_after[col]}")
        
        return normalized_df
    
    def clean_ohlcv_data(
        self,
        df: pd.DataFrame,
        handle_gaps: bool = True,
        handle_negative_values: bool = True,
        verify_high_low: bool = True,
        verify_open_close: bool = True,
        flag_outliers_only: bool = False  # Flag outliers without removing them
    ) -> pd.DataFrame:
        """
        Làm sạch dữ liệu OHLCV (Open, High, Low, Close, Volume).
        
        Args:
            df: DataFrame chứa dữ liệu OHLCV
            handle_gaps: Xử lý các khoảng trống trong dữ liệu
            handle_negative_values: Xử lý giá trị âm
            verify_high_low: Kiểm tra high >= low
            verify_open_close: Kiểm tra open và close nằm trong khoảng [low, high]
            flag_outliers_only: Chỉ đánh dấu ngoại lệ mà không loại bỏ
            
        Returns:
            DataFrame đã làm sạch
        """
        if df.empty:
            self.logger.warning("DataFrame rỗng, không có gì để làm sạch")
            return df
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        cleaned_df = df.copy()
        
        # Kiểm tra các cột cần thiết
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in cleaned_df.columns]
        
        if missing_columns:
            self.logger.error(f"Thiếu các cột trong dữ liệu OHLCV: {missing_columns}")
            return df
        
        self.logger.info(f"Bắt đầu làm sạch dữ liệu OHLCV với {len(df)} dòng")
        
        # Đảm bảo các cột có kiểu dữ liệu đúng
        try:
            if not pd.api.types.is_datetime64_any_dtype(cleaned_df['timestamp']):
                original_ts = cleaned_df['timestamp'].copy()
                cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
                self.logger.info(f"Đã chuyển đổi cột timestamp từ {original_ts.dtype} sang datetime64")
            
            # Đảm bảo các cột giá trị là số
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if not pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    original_dtype = cleaned_df[col].dtype
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                    self.logger.info(f"Đã chuyển đổi cột {col} từ {original_dtype} sang numeric")
            
            # Thống kê số lượng NA sau khi chuyển đổi
            na_counts = cleaned_df[['open', 'high', 'low', 'close', 'volume']].isna().sum()
            na_cols = na_counts[na_counts > 0]
            if not na_cols.empty:
                self.logger.warning(f"Số lượng NA sau khi chuyển đổi kiểu dữ liệu: {na_cols.to_dict()}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi chuyển đổi kiểu dữ liệu: {e}")
            # Trả về dữ liệu gốc nếu có lỗi
            return df
        
        # Xóa các bản ghi có timestamp trùng lặp
        original_len = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(subset=['timestamp'])
        new_len = len(cleaned_df)
        if original_len > new_len:
            self.logger.info(f"Đã loại bỏ {original_len - new_len} dòng có timestamp trùng lặp ({original_len} -> {new_len})")
        
        # Xử lý giá trị âm
        if handle_negative_values:
            for col in ['open', 'high', 'low', 'close']:
                negative_mask = cleaned_df[col] < 0
                negative_count = negative_mask.sum()
                if negative_count > 0:
                    self.logger.warning(f"Phát hiện {negative_count} giá trị âm trong cột {col}")
                    cleaned_df.loc[negative_mask, col] = np.nan
            
            # Khối lượng không được âm
            negative_volume_mask = cleaned_df['volume'] < 0
            negative_volume_count = negative_volume_mask.sum()
            if negative_volume_count > 0:
                self.logger.warning(f"Phát hiện {negative_volume_count} giá trị âm trong cột volume")
                cleaned_df.loc[negative_volume_mask, 'volume'] = 0
        
        # Kiểm tra high >= low
        inconsistencies = []
        if verify_high_low:
            high_low_mask = cleaned_df['high'] < cleaned_df['low']
            high_low_count = high_low_mask.sum()
            if high_low_count > 0:
                self.logger.warning(f"Phát hiện {high_low_count} dòng có high < low")
                inconsistencies.append(f"{high_low_count} dòng high < low")
                
                # Hoán đổi giá trị high và low
                high_vals = cleaned_df.loc[high_low_mask, 'high'].copy()
                low_vals = cleaned_df.loc[high_low_mask, 'low'].copy()
                cleaned_df.loc[high_low_mask, 'high'] = low_vals
                cleaned_df.loc[high_low_mask, 'low'] = high_vals
                self.logger.info(f"Đã hoán đổi {high_low_count} cặp giá trị high và low")
        
        # Kiểm tra open và close nằm trong khoảng [low, high]
        if verify_open_close:
            # Kiểm tra open
            open_low_mask = cleaned_df['open'] < cleaned_df['low']
            open_low_count = open_low_mask.sum()
            if open_low_count > 0:
                self.logger.warning(f"Phát hiện {open_low_count} dòng có open < low")
                inconsistencies.append(f"{open_low_count} dòng open < low")
                cleaned_df.loc[open_low_mask, 'open'] = cleaned_df.loc[open_low_mask, 'low']
            
            open_high_mask = cleaned_df['open'] > cleaned_df['high']
            open_high_count = open_high_mask.sum()
            if open_high_count > 0:
                self.logger.warning(f"Phát hiện {open_high_count} dòng có open > high")
                inconsistencies.append(f"{open_high_count} dòng open > high")
                cleaned_df.loc[open_high_mask, 'open'] = cleaned_df.loc[open_high_mask, 'high']
            
            # Kiểm tra close
            close_low_mask = cleaned_df['close'] < cleaned_df['low']
            close_low_count = close_low_mask.sum()
            if close_low_count > 0:
                self.logger.warning(f"Phát hiện {close_low_count} dòng có close < low")
                inconsistencies.append(f"{close_low_count} dòng close < low")
                cleaned_df.loc[close_low_mask, 'close'] = cleaned_df.loc[close_low_mask, 'low']
            
            close_high_mask = cleaned_df['close'] > cleaned_df['high']
            close_high_count = close_high_mask.sum()
            if close_high_count > 0:
                self.logger.warning(f"Phát hiện {close_high_count} dòng có close > high")
                inconsistencies.append(f"{close_high_count} dòng close > high")
                cleaned_df.loc[close_high_mask, 'close'] = cleaned_df.loc[close_high_mask, 'high']
        
        if inconsistencies:
            self.logger.info(f"Đã sửa chữa những vấn đề sau: {', '.join(inconsistencies)}")
        
        # Xử lý các khoảng trống trong dữ liệu
        if handle_gaps and len(cleaned_df) > 1:
            # Sắp xếp theo timestamp
            cleaned_df = cleaned_df.sort_values('timestamp')
            
            # Tính toán khoảng thời gian trung bình giữa các điểm dữ liệu
            time_diffs = cleaned_df['timestamp'].diff()[1:].dt.total_seconds()
            median_diff = time_diffs.median()
            
            # Phát hiện các khoảng trống lớn (gấp 2 lần trung bình)
            gap_threshold = 2 * median_diff
            large_gaps = time_diffs[time_diffs > gap_threshold]
            
            if not large_gaps.empty:
                self.logger.info(f"Phát hiện {len(large_gaps)} khoảng trống lớn trong dữ liệu (ngưỡng: {gap_threshold:.2f}s)")
                self.logger.info(f"Khoảng trống lớn nhất: {large_gaps.max():.2f}s, nhỏ nhất: {large_gaps.min():.2f}s")
                
                # Có thể thêm logic để điền các khoảng trống ở đây nếu cần
                # Ví dụ: tạo các dòng mới và điền bằng nội suy
        
        # Phát hiện và đánh dấu ngoại lệ nếu cần
        if self.use_outlier_detection:
            cols_to_check = ['open', 'high', 'low', 'close', 'volume']
            
            if flag_outliers_only:
                # Chỉ đánh dấu ngoại lệ mà không loại bỏ
                outlier_df = self.outlier_detector.detect_outliers(
                    cleaned_df, columns=cols_to_check, per_column=True
                )
                
                # Lấy các cột đánh dấu ngoại lệ từ outlier_df và thêm vào cleaned_df
                outlier_cols = [col for col in outlier_df.columns if col.endswith('_is_outlier') or col == 'is_outlier']
                for col in outlier_cols:
                    cleaned_df[col] = outlier_df[col]
                
                outlier_count = outlier_df['is_outlier'].sum()
                self.logger.info(f"Đã đánh dấu {outlier_count} dòng có ngoại lệ trong dữ liệu OHLCV (không loại bỏ)")
            else:
                # Loại bỏ ngoại lệ
                original_len = len(cleaned_df)
                cleaned_df = self.outlier_detector.remove_outliers(
                    cleaned_df, columns=cols_to_check
                )
                new_len = len(cleaned_df)
                
                if original_len > new_len:
                    self.logger.info(f"Đã loại bỏ {original_len - new_len} dòng chứa ngoại lệ ({original_len} -> {new_len})")
        
        # Xử lý giá trị NA nếu có
        if self.use_missing_data_handler:
            na_before = cleaned_df[['open', 'high', 'low', 'close', 'volume']].isna().sum().sum()
            if na_before > 0:
                cleaned_df = self.missing_data_handler.handle_missing_values(
                    cleaned_df, columns=['open', 'high', 'low', 'close', 'volume']
                )
                na_after = cleaned_df[['open', 'high', 'low', 'close', 'volume']].isna().sum().sum()
                self.logger.info(f"Đã xử lý {na_before - na_after} giá trị NA trong dữ liệu OHLCV ({na_before} -> {na_after})")
        
        self.logger.info(f"Đã hoàn thành làm sạch dữ liệu OHLCV, kết quả có {len(cleaned_df)} dòng")
        return cleaned_df
    
    def clean_orderbook_data(
        self,
        df: pd.DataFrame,
        verify_price_levels: bool = True,
        verify_quantities: bool = True,
        flag_outliers_only: bool = False,  # Flag outliers without removing them
        min_quantity_threshold: float = 0.0  # Lọc các lệnh có khối lượng quá nhỏ
    ) -> pd.DataFrame:
        """
        Làm sạch dữ liệu orderbook.
        
        Args:
            df: DataFrame chứa dữ liệu orderbook
            verify_price_levels: Kiểm tra các mức giá
            verify_quantities: Kiểm tra khối lượng
            flag_outliers_only: Chỉ đánh dấu ngoại lệ mà không loại bỏ
            min_quantity_threshold: Ngưỡng khối lượng tối thiểu (lọc các lệnh quá nhỏ)
            
        Returns:
            DataFrame đã làm sạch
        """
        if df.empty:
            self.logger.warning("DataFrame rỗng, không có gì để làm sạch")
            return df
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        cleaned_df = df.copy()
        
        # Kiểm tra các cột cần thiết
        required_columns = ['timestamp', 'price', 'amount', 'side']
        missing_columns = [col for col in required_columns if col not in cleaned_df.columns]
        
        if missing_columns:
            self.logger.error(f"Thiếu các cột trong dữ liệu orderbook: {missing_columns}")
            return df
        
        self.logger.info(f"Bắt đầu làm sạch dữ liệu orderbook với {len(df)} dòng")
        
        # Đảm bảo các cột có kiểu dữ liệu đúng
        try:
            if not pd.api.types.is_datetime64_any_dtype(cleaned_df['timestamp']):
                original_ts = cleaned_df['timestamp'].copy()
                cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
                self.logger.info(f"Đã chuyển đổi cột timestamp từ {original_ts.dtype} sang datetime64")
            
            # Đảm bảo các cột giá trị là số
            for col in ['price', 'amount']:
                if not pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    original_dtype = cleaned_df[col].dtype
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                    self.logger.info(f"Đã chuyển đổi cột {col} từ {original_dtype} sang numeric")
            
            # Thống kê số lượng NA sau khi chuyển đổi
            na_counts = cleaned_df[['price', 'amount']].isna().sum()
            if na_counts.sum() > 0:
                self.logger.warning(f"Số lượng NA sau khi chuyển đổi kiểu dữ liệu: {na_counts.to_dict()}")
                
        except Exception as e:
            self.logger.error(f"Lỗi khi chuyển đổi kiểu dữ liệu: {e}")
            # Trả về dữ liệu gốc nếu có lỗi
            return df
        
        # Kiểm tra giá
        if verify_price_levels:
            # Giá phải dương
            non_positive_price_mask = cleaned_df['price'] <= 0
            non_positive_price_count = non_positive_price_mask.sum()
            if non_positive_price_count > 0:
                self.logger.warning(f"Phát hiện {non_positive_price_count} dòng có giá <= 0")
                cleaned_df = cleaned_df[~non_positive_price_mask]
                self.logger.info(f"Đã loại bỏ {non_positive_price_count} dòng có giá <= 0")
        
        # Kiểm tra khối lượng
        if verify_quantities:
            # Khối lượng phải dương
            non_positive_amount_mask = cleaned_df['amount'] <= 0
            non_positive_amount_count = non_positive_amount_mask.sum()
            if non_positive_amount_count > 0:
                self.logger.warning(f"Phát hiện {non_positive_amount_count} dòng có khối lượng <= 0")
                cleaned_df = cleaned_df[~non_positive_amount_mask]
                self.logger.info(f"Đã loại bỏ {non_positive_amount_count} dòng có khối lượng <= 0")
            
            # Kiểm tra ngưỡng khối lượng tối thiểu
            if min_quantity_threshold > 0:
                small_amount_mask = cleaned_df['amount'] < min_quantity_threshold
                small_amount_count = small_amount_mask.sum()
                if small_amount_count > 0:
                    self.logger.warning(f"Phát hiện {small_amount_count} dòng có khối lượng < {min_quantity_threshold}")
                    cleaned_df = cleaned_df[~small_amount_mask]
                    self.logger.info(f"Đã loại bỏ {small_amount_count} dòng có khối lượng nhỏ")
        
        # Đảm bảo side chỉ có 'bid' hoặc 'ask'
        valid_sides = ['bid', 'ask']
        if 'side' in cleaned_df.columns:
            invalid_side_mask = ~cleaned_df['side'].isin(valid_sides)
            invalid_side_count = invalid_side_mask.sum()
            if invalid_side_count > 0:
                self.logger.warning(f"Phát hiện {invalid_side_count} dòng có side không hợp lệ")
                
                # Kiểm tra các giá trị side không hợp lệ
                invalid_sides = cleaned_df.loc[invalid_side_mask, 'side'].unique()
                self.logger.debug(f"Các giá trị side không hợp lệ: {invalid_sides}")
                
                # Loại bỏ các dòng có side không hợp lệ
                cleaned_df = cleaned_df[~invalid_side_mask]
                self.logger.info(f"Đã loại bỏ {invalid_side_count} dòng có side không hợp lệ")
        
        # Phát hiện và đánh dấu ngoại lệ nếu cần
        if self.use_outlier_detection:
            # Phát hiện ngoại lệ cho cột price và amount
            cols_to_check = ['price', 'amount']
            
            # Phân tích riêng cho bid và ask
            if flag_outliers_only:
                # Chỉ đánh dấu ngoại lệ mà không loại bỏ
                for side in cleaned_df['side'].unique():
                    side_mask = cleaned_df['side'] == side
                    side_df = cleaned_df[side_mask].copy()
                    
                    if len(side_df) > 10:  # Cần đủ dữ liệu để phát hiện ngoại lệ
                        outlier_df = self.outlier_detector.detect_outliers(
                            side_df, columns=cols_to_check, per_column=True,
                            outlier_suffix=f'_{side}_is_outlier'
                        )
                        
                        # Lấy các cột đánh dấu ngoại lệ và cập nhật vào cleaned_df
                        outlier_cols = [col for col in outlier_df.columns 
                                       if col.endswith(f'_{side}_is_outlier') or col == 'is_outlier']
                        
                        for col in outlier_cols:
                            # Chỉ cập nhật cho các dòng thuộc side tương ứng
                            cleaned_df.loc[side_mask, col] = outlier_df[col]
                        
                        outlier_count = outlier_df['is_outlier'].sum()
                        self.logger.info(f"Đã đánh dấu {outlier_count} ngoại lệ trong dữ liệu {side} (không loại bỏ)")
            else:
                # Loại bỏ ngoại lệ
                for side in cleaned_df['side'].unique():
                    side_mask = cleaned_df['side'] == side
                    side_df = cleaned_df[side_mask].copy()
                    
                    if len(side_df) > 10:  # Cần đủ dữ liệu để phát hiện ngoại lệ
                        original_len = len(side_df)
                        side_df_no_outliers = self.outlier_detector.remove_outliers(
                            side_df, columns=cols_to_check
                        )
                        new_len = len(side_df_no_outliers)
                        
                        if original_len > new_len:
                            self.logger.info(f"Đã loại bỏ {original_len - new_len} ngoại lệ từ dữ liệu {side}")
                            # Cập nhật lại cleaned_df bằng cách loại bỏ các dòng chứa ngoại lệ
                            outlier_indices = set(side_df.index) - set(side_df_no_outliers.index)
                            cleaned_df = cleaned_df.drop(outlier_indices)
        
        # Tính cost nếu chưa có
        if 'cost' not in cleaned_df.columns:
            cleaned_df['cost'] = cleaned_df['price'] * cleaned_df['amount']
            self.logger.debug("Đã tính toán cột 'cost' = price * amount")
            
        # Thống kê cuối cùng
        result_stats = {
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'removed_rows': len(df) - len(cleaned_df),
            'bid_count': (cleaned_df['side'] == 'bid').sum(),
            'ask_count': (cleaned_df['side'] == 'ask').sum(),
            'avg_bid_price': cleaned_df.loc[cleaned_df['side'] == 'bid', 'price'].mean() if 'bid' in cleaned_df['side'].values else None,
            'avg_ask_price': cleaned_df.loc[cleaned_df['side'] == 'ask', 'price'].mean() if 'ask' in cleaned_df['side'].values else None,
            'total_bid_volume': cleaned_df.loc[cleaned_df['side'] == 'bid', 'amount'].sum() if 'bid' in cleaned_df['side'].values else None,
            'total_ask_volume': cleaned_df.loc[cleaned_df['side'] == 'ask', 'amount'].sum() if 'ask' in cleaned_df['side'].values else None,
        }
        
        self.logger.info(f"Kết quả làm sạch dữ liệu orderbook: {len(df)} -> {len(cleaned_df)} dòng " +
                        f"(bids: {result_stats['bid_count']}, asks: {result_stats['ask_count']})")
        
        return cleaned_df
    
    def clean_trade_data(
        self,
        df: pd.DataFrame,
        verify_price: bool = True,
        verify_amount: bool = True,
        flag_outliers_only: bool = False,  # Flag outliers without removing them
        min_quantity_threshold: float = 0.0  # Lọc các giao dịch có khối lượng quá nhỏ
    ) -> pd.DataFrame:
        """
        Làm sạch dữ liệu giao dịch.
        
        Args:
            df: DataFrame chứa dữ liệu giao dịch
            verify_price: Kiểm tra giá
            verify_amount: Kiểm tra khối lượng
            flag_outliers_only: Chỉ đánh dấu ngoại lệ mà không loại bỏ
            min_quantity_threshold: Ngưỡng khối lượng tối thiểu (lọc các giao dịch quá nhỏ)
            
        Returns:
            DataFrame đã làm sạch
        """
        if df.empty:
            self.logger.warning("DataFrame rỗng, không có gì để làm sạch")
            return df
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        cleaned_df = df.copy()
        
        # Kiểm tra các cột cần thiết
        required_columns = ['timestamp', 'price', 'amount']
        missing_columns = [col for col in required_columns if col not in cleaned_df.columns]
        
        if missing_columns:
            self.logger.error(f"Thiếu các cột trong dữ liệu giao dịch: {missing_columns}")
            return df
        
        self.logger.info(f"Bắt đầu làm sạch dữ liệu giao dịch với {len(df)} dòng")
        
        # Đảm bảo các cột có kiểu dữ liệu đúng
        try:
            if not pd.api.types.is_datetime64_any_dtype(cleaned_df['timestamp']):
                original_ts = cleaned_df['timestamp'].copy()
                cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
                self.logger.info(f"Đã chuyển đổi cột timestamp từ {original_ts.dtype} sang datetime64")
            
            # Đảm bảo các cột giá trị là số
            for col in ['price', 'amount']:
                if not pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    original_dtype = cleaned_df[col].dtype
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                    self.logger.info(f"Đã chuyển đổi cột {col} từ {original_dtype} sang numeric")
            
            # Thống kê số lượng NA sau khi chuyển đổi
            na_counts = cleaned_df[['price', 'amount']].isna().sum()
            if na_counts.sum() > 0:
                self.logger.warning(f"Số lượng NA sau khi chuyển đổi kiểu dữ liệu: {na_counts.to_dict()}")
                
        except Exception as e:
            self.logger.error(f"Lỗi khi chuyển đổi kiểu dữ liệu: {e}")
            # Trả về dữ liệu gốc nếu có lỗi
            return df
        
        # Loại bỏ dòng bị trùng ID nếu có cột ID
        if 'id' in cleaned_df.columns:
            original_len = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(subset=['id'])
            new_len = len(cleaned_df)
            if original_len > new_len:
                self.logger.info(f"Đã loại bỏ {original_len - new_len} dòng có ID trùng lặp ({original_len} -> {new_len})")
        else:
            # Nếu không có ID, kiểm tra trùng lặp dựa trên timestamp, price, amount
            original_len = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(subset=['timestamp', 'price', 'amount'])
            new_len = len(cleaned_df)
            if original_len > new_len:
                self.logger.info(f"Đã loại bỏ {original_len - new_len} dòng trùng lặp dựa trên timestamp, price, amount ({original_len} -> {new_len})")
        
        # Kiểm tra giá
        if verify_price:
            # Giá phải dương
            non_positive_price_mask = cleaned_df['price'] <= 0
            non_positive_price_count = non_positive_price_mask.sum()
            if non_positive_price_count > 0:
                self.logger.warning(f"Phát hiện {non_positive_price_count} dòng có giá <= 0")
                # Loại bỏ các dòng có giá không hợp lệ
                cleaned_df = cleaned_df[~non_positive_price_mask]
                self.logger.info(f"Đã loại bỏ {non_positive_price_count} dòng có giá <= 0")
        
        # Kiểm tra khối lượng
        if verify_amount:
            # Khối lượng phải dương
            non_positive_amount_mask = cleaned_df['amount'] <= 0
            non_positive_amount_count = non_positive_amount_mask.sum()
            if non_positive_amount_count > 0:
                self.logger.warning(f"Phát hiện {non_positive_amount_count} dòng có khối lượng <= 0")
                # Loại bỏ các dòng có khối lượng không hợp lệ
                cleaned_df = cleaned_df[~non_positive_amount_mask]
                self.logger.info(f"Đã loại bỏ {non_positive_amount_count} dòng có khối lượng <= 0")
            
            # Kiểm tra ngưỡng khối lượng tối thiểu
            if min_quantity_threshold > 0:
                small_amount_mask = cleaned_df['amount'] < min_quantity_threshold
                small_amount_count = small_amount_mask.sum()
                if small_amount_count > 0:
                    self.logger.warning(f"Phát hiện {small_amount_count} dòng có khối lượng < {min_quantity_threshold}")
                    # Loại bỏ các dòng có khối lượng nhỏ
                    cleaned_df = cleaned_df[~small_amount_mask]
                    self.logger.info(f"Đã loại bỏ {small_amount_count} dòng có khối lượng nhỏ")
        
        # Đảm bảo dữ liệu được sắp xếp theo thời gian
        cleaned_df = cleaned_df.sort_values('timestamp')
        
        # Tính cost nếu chưa có
        if 'cost' not in cleaned_df.columns:
            cleaned_df['cost'] = cleaned_df['price'] * cleaned_df['amount']
            self.logger.debug("Đã tính toán cột 'cost' = price * amount")
        
        # Phát hiện và đánh dấu ngoại lệ nếu cần
        if self.use_outlier_detection:
            cols_to_check = ['price', 'amount']
            
            if flag_outliers_only:
                # Chỉ đánh dấu ngoại lệ mà không loại bỏ
                if 'side' in cleaned_df.columns:
                    # Phân tích riêng cho mỗi bên (buy/sell) nếu có thông tin side
                    for side in cleaned_df['side'].unique():
                        side_mask = cleaned_df['side'] == side
                        side_df = cleaned_df[side_mask].copy()
                        
                        if len(side_df) > 10:  # Cần đủ dữ liệu để phát hiện ngoại lệ
                            outlier_df = self.outlier_detector.detect_outliers(
                                side_df, columns=cols_to_check, per_column=True,
                                outlier_suffix=f'_{side}_is_outlier'
                            )
                            
                            # Lấy các cột đánh dấu ngoại lệ và cập nhật vào cleaned_df
                            outlier_cols = [col for col in outlier_df.columns 
                                          if col.endswith(f'_{side}_is_outlier') or col == 'is_outlier']
                            
                            for col in outlier_cols:
                                cleaned_df.loc[side_mask, col] = outlier_df[col]
                            
                            outlier_count = outlier_df['is_outlier'].sum()
                            self.logger.info(f"Đã đánh dấu {outlier_count} ngoại lệ trong dữ liệu {side} (không loại bỏ)")
                else:
                    # Nếu không có thông tin side, phân tích toàn bộ dữ liệu
                    outlier_df = self.outlier_detector.detect_outliers(
                        cleaned_df, columns=cols_to_check, per_column=True
                    )
                    
                    # Lấy các cột đánh dấu ngoại lệ và cập nhật vào cleaned_df
                    outlier_cols = [col for col in outlier_df.columns 
                                  if col.endswith('_is_outlier') or col == 'is_outlier']
                    
                    for col in outlier_cols:
                        cleaned_df[col] = outlier_df[col]
                    
                    outlier_count = outlier_df['is_outlier'].sum()
                    self.logger.info(f"Đã đánh dấu {outlier_count} ngoại lệ (không loại bỏ)")
            else:
                # Loại bỏ ngoại lệ
                if 'side' in cleaned_df.columns:
                    # Phân tích riêng cho mỗi bên (buy/sell) nếu có thông tin side
                    for side in cleaned_df['side'].unique():
                        side_mask = cleaned_df['side'] == side
                        side_df = cleaned_df[side_mask].copy()
                        
                        if len(side_df) > 10:  # Cần đủ dữ liệu để phát hiện ngoại lệ
                            original_len = len(side_df)
                            side_df_no_outliers = self.outlier_detector.remove_outliers(
                                side_df, columns=cols_to_check
                            )
                            new_len = len(side_df_no_outliers)
                            
                            if original_len > new_len:
                                self.logger.info(f"Đã loại bỏ {original_len - new_len} ngoại lệ từ dữ liệu {side}")
                                # Cập nhật lại cleaned_df bằng cách loại bỏ các dòng chứa ngoại lệ
                                outlier_indices = set(side_df.index) - set(side_df_no_outliers.index)
                                cleaned_df = cleaned_df.drop(outlier_indices)
                else:
                    # Nếu không có thông tin side, phân tích toàn bộ dữ liệu
                    original_len = len(cleaned_df)
                    cleaned_df = self.outlier_detector.remove_outliers(
                        cleaned_df, columns=cols_to_check
                    )
                    new_len = len(cleaned_df)
                    
                    if original_len > new_len:
                        self.logger.info(f"Đã loại bỏ {original_len - new_len} dòng chứa ngoại lệ ({original_len} -> {new_len})")
        
        # Thống kê cuối cùng
        result_stats = {
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'removed_rows': len(df) - len(cleaned_df),
            'avg_price': cleaned_df['price'].mean(),
            'total_volume': cleaned_df['amount'].sum(),
            'total_cost': cleaned_df['cost'].sum() if 'cost' in cleaned_df.columns else None,
            'time_range': f"{cleaned_df['timestamp'].min()} - {cleaned_df['timestamp'].max()}" if not cleaned_df.empty else None
        }
        
        self.logger.info(f"Kết quả làm sạch dữ liệu giao dịch: {len(df)} -> {len(cleaned_df)} dòng")
        return cleaned_df