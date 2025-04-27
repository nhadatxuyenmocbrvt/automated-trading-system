"""
Pipeline xử lý dữ liệu tích hợp cho hệ thống giao dịch.
File này cung cấp các lớp và phương thức để tạo pipeline xử lý dữ liệu hoàn chỉnh,
tích hợp các module làm sạch, tạo đặc trưng, và lựa chọn đặc trưng.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Set
import logging
from pathlib import Path
import pickle
import os
import sys
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import setup_logger
from config.constants import ErrorCode

# Import các module xử lý dữ liệu
from data_processors.cleaners.data_cleaner import DataCleaner
from data_processors.cleaners.outlier_detector import OutlierDetector
from data_processors.cleaners.missing_data_handler import MissingDataHandler
from data_processors.feature_engineering.technical_indicators import TechnicalIndicatorGenerator
from data_processors.feature_engineering.market_features import MarketFeatureGenerator
from data_processors.feature_engineering.sentiment_features import SentimentFeatureExtractor
from data_processors.feature_engineering.feature_selector import FeatureSelector

class DataPipeline:
    """
    Pipeline xử lý dữ liệu tích hợp cho hệ thống giao dịch.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        pipeline_name: str = "default_pipeline",
        output_dir: Optional[str] = None,
        debug_mode: bool = False
    ):
        """
        Khởi tạo pipeline xử lý dữ liệu.
        
        Args:
            config: Cấu hình cho pipeline (None để sử dụng cấu hình mặc định)
            pipeline_name: Tên của pipeline để lưu/tải
            output_dir: Thư mục lưu kết quả
            debug_mode: Chế độ debug
        """
        self.logger = setup_logger("data_pipeline")
        
        self.pipeline_name = pipeline_name
        self.debug_mode = debug_mode
        
        # Thiết lập thư mục lưu kết quả
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.output_dir = None
        
        # Khởi tạo cấu hình mặc định nếu không được cung cấp
        if config is None:
            config = self._get_default_config()
        
        self.config = config
        
        # Khởi tạo các thành phần của pipeline
        self._init_components()
        
        # Biến để theo dõi trạng thái và kết quả xử lý
        self.steps_executed = []
        self.execution_times = {}
        self.results_cache = {}
        self.feature_metadata = {}
        self.is_fitted = False
        
        self.logger.info(f"Đã khởi tạo DataPipeline '{pipeline_name}' với {len(config)} thành phần cấu hình")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Tạo cấu hình mặc định cho pipeline.
        
        Returns:
            Dictionary cấu hình mặc định
        """
        default_config = {
            # Cấu hình chung
            'general': {
                'handle_missing_data': True,
                'handle_outliers': True,
                'generate_technical_indicators': True,
                'generate_market_features': True,
                'process_sentiment': False,  # Tắt theo mặc định vì yêu cầu dữ liệu tin tức
                'select_features': True,
                'drop_na_after_processing': True,
                'add_date_features': True
            },
            
            # Cấu hình cho DataCleaner
            'data_cleaner': {
                'enabled': True,
                'normalize_method': 'z-score',
                'outlier_detector_kwargs': {
                    'method': 'z-score',
                    'threshold': 3.0,
                    'use_robust': True
                },
                'missing_data_handler_kwargs': {
                    'method': 'interpolate',
                    'handle_categorical': True
                }
            },
            
            # Cấu hình cho TechnicalIndicatorGenerator
            'technical_indicators': {
                'enabled': True,
                'indicators': [
                    'sma', 'ema', 'macd', 'rsi', 'bollinger_bands', 
                    'stochastic', 'atr', 'adx', 'obv', 'momentum'
                ],
                'windows': [5, 10, 20, 50, 100],
                'fillna': True
            },
            
            # Cấu hình cho MarketFeatureGenerator
            'market_features': {
                'enabled': True,
                'price_differences': True,
                'price_ratios': True,
                'volume_features': True,
                'volatility_features': True,
                'pattern_recognition': True,
                'custom_features': []
            },
            
            # Cấu hình cho SentimentFeatureExtractor
            'sentiment_features': {
                'enabled': False,
                'sentiment_method': 'lexicon',
                'language': 'en',
                'normalize_scores': True,
                'volume_weighted': True
            },
            
            # Cấu hình cho FeatureSelector
            'feature_selector': {
                'enabled': True,
                'method': 'random_forest',
                'n_features': 50,
                'threshold': 0.01,
                'handle_collinearity': True,
                'collinearity_threshold': 0.9
            },
            
            # Cấu hình cho các loại dữ liệu cụ thể
            'ohlcv_data': {
                'timestamp_column': 'timestamp',
                'ohlcv_columns': {
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                },
                'verify_high_low': True,
                'handle_gaps': True
            },
            
            'orderbook_data': {
                'timestamp_column': 'timestamp',
                'price_column': 'price',
                'amount_column': 'amount',
                'side_column': 'side',
                'min_quantity_threshold': 0.0
            },
            
            'trade_data': {
                'timestamp_column': 'timestamp',
                'price_column': 'price',
                'amount_column': 'amount',
                'side_column': 'side',
                'min_quantity_threshold': 0.0
            },
            
            # Cấu hình để xử lý dữ liệu tin tức/tâm lý
            'news_data': {
                'timestamp_column': 'timestamp',
                'text_column': 'content',
                'entity_column': 'entity',
                'source_column': 'source'
            }
        }
        
        return default_config
    
    def _init_components(self) -> None:
        """
        Khởi tạo các thành phần của pipeline dựa trên cấu hình.
        """
        # Khởi tạo DataCleaner nếu được kích hoạt
        if self.config['data_cleaner']['enabled']:
            self.data_cleaner = DataCleaner(
                normalize_method=self.config['data_cleaner']['normalize_method'],
                use_outlier_detection=self.config['general']['handle_outliers'],
                use_missing_data_handler=self.config['general']['handle_missing_data'],
                outlier_detector_kwargs=self.config['data_cleaner']['outlier_detector_kwargs'],
                missing_data_handler_kwargs=self.config['data_cleaner']['missing_data_handler_kwargs']
            )
        else:
            self.data_cleaner = None
        
        # Khởi tạo TechnicalIndicatorGenerator nếu được kích hoạt
        if self.config['technical_indicators']['enabled'] and self.config['general']['generate_technical_indicators']:
            self.technical_indicator_generator = TechnicalIndicatorGenerator(
                indicators=self.config['technical_indicators']['indicators'],
                windows=self.config['technical_indicators']['windows'],
                fillna=self.config['technical_indicators']['fillna']
            )
        else:
            self.technical_indicator_generator = None
        
        # Khởi tạo MarketFeatureGenerator nếu được kích hoạt
        if self.config['market_features']['enabled'] and self.config['general']['generate_market_features']:
            self.market_feature_generator = MarketFeatureGenerator(
                price_differences=self.config['market_features']['price_differences'],
                price_ratios=self.config['market_features']['price_ratios'],
                volume_features=self.config['market_features']['volume_features'],
                volatility_features=self.config['market_features']['volatility_features'],
                pattern_recognition=self.config['market_features']['pattern_recognition'],
                custom_features=self.config['market_features']['custom_features']
            )
        else:
            self.market_feature_generator = None
        
        # Khởi tạo SentimentFeatureExtractor nếu được kích hoạt
        if self.config['sentiment_features']['enabled'] and self.config['general']['process_sentiment']:
            self.sentiment_feature_extractor = SentimentFeatureExtractor(
                sentiment_method=self.config['sentiment_features']['sentiment_method'],
                language=self.config['sentiment_features']['language'],
                normalize_scores=self.config['sentiment_features']['normalize_scores'],
                volume_weighted=self.config['sentiment_features']['volume_weighted']
            )
        else:
            self.sentiment_feature_extractor = None
        
        # Khởi tạo FeatureSelector nếu được kích hoạt
        if self.config['feature_selector']['enabled'] and self.config['general']['select_features']:
            self.feature_selector = FeatureSelector(
                method=self.config['feature_selector']['method'],
                n_features=self.config['feature_selector']['n_features'],
                threshold=self.config['feature_selector']['threshold'],
                handle_collinearity=self.config['feature_selector']['handle_collinearity'],
                collinearity_threshold=self.config['feature_selector']['collinearity_threshold'],
                save_dir=self.output_dir / 'feature_selection' if self.output_dir else None
            )
        else:
            self.feature_selector = None
    
    def process_data(
        self,
        data: Dict[str, pd.DataFrame],
        target_column: Optional[str] = None,
        data_types: Optional[Dict[str, str]] = None,
        fit: bool = True,
        save_intermediate: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Xử lý dữ liệu qua pipeline.
        
        Args:
            data: Dictionary chứa các DataFrame dữ liệu (theo loại)
                Ví dụ: {'ohlcv': ohlcv_df, 'trades': trades_df, 'news': news_df}
            target_column: Tên cột mục tiêu (nếu có)
            data_types: Dictionary ánh xạ loại dữ liệu cho mỗi khóa trong data
                Ví dụ: {'ohlcv': 'ohlcv_data', 'trades': 'trade_data'}
            fit: Nếu True, sẽ fit pipeline với dữ liệu, ngược lại sẽ transform dữ liệu
            save_intermediate: Lưu các kết quả trung gian
            
        Returns:
            Dictionary chứa các DataFrame kết quả
        """
        start_time = time.time()
        self.steps_executed = []
        self.execution_times = {}
        self.results_cache = {}
        
        if not data:
            self.logger.warning("Không có dữ liệu đầu vào, không thể xử lý")
            return {}
        
        # Xác định loại dữ liệu cho mỗi DataFrame nếu không được chỉ định
        if data_types is None:
            data_types = {}
            for key in data.keys():
                if 'ohlcv' in key.lower() or 'price' in key.lower():
                    data_types[key] = 'ohlcv_data'
                elif 'orderbook' in key.lower() or 'book' in key.lower():
                    data_types[key] = 'orderbook_data'
                elif 'trade' in key.lower():
                    data_types[key] = 'trade_data'
                elif 'news' in key.lower() or 'sentiment' in key.lower():
                    data_types[key] = 'news_data'
                else:
                    data_types[key] = 'ohlcv_data'  # Mặc định xử lý như OHLCV
        
        self.logger.info(f"Bắt đầu xử lý {len(data)} DataFrame với pipeline '{self.pipeline_name}'")
        
        # Kết quả cuối cùng
        processed_data = {}
        
        # Xử lý từng DataFrame theo loại dữ liệu
        for key, df in data.items():
            if df.empty:
                self.logger.warning(f"DataFrame '{key}' rỗng, bỏ qua")
                continue
                
            data_type = data_types.get(key, 'ohlcv_data')
            
            # Tạo bản sao để tránh thay đổi dữ liệu gốc
            df_copy = df.copy()
            
            self.logger.info(f"Xử lý DataFrame '{key}' (loại: {data_type}) với {len(df_copy)} dòng")
            
            # Bước 1: Làm sạch dữ liệu
            df_cleaned = self._clean_data(df_copy, data_type)
            
            # Thêm đặc trưng thời gian nếu được cấu hình
            if self.config['general']['add_date_features']:
                df_cleaned = self._add_date_features(df_cleaned, data_type)
            
            # Bước 2: Tạo đặc trưng kỹ thuật nếu là dữ liệu OHLCV
            if data_type == 'ohlcv_data' and self.technical_indicator_generator is not None:
                df_with_indicators = self._generate_technical_indicators(df_cleaned, data_type)
            else:
                df_with_indicators = df_cleaned
            
            # Bước 3: Tạo đặc trưng thị trường
            if self.market_feature_generator is not None:
                df_with_market_features = self._generate_market_features(df_with_indicators, data_type)
            else:
                df_with_market_features = df_with_indicators
            
            # Bước 4: Xử lý đặc trưng tâm lý nếu là dữ liệu tin tức
            if data_type == 'news_data' and self.sentiment_feature_extractor is not None:
                df_with_sentiment = self._process_sentiment(df_with_market_features, data_type)
            else:
                df_with_sentiment = df_with_market_features
            
            # Lưu kết quả trung gian
            processed_data[key] = df_with_sentiment
            
            # Lưu kết quả trung gian nếu được yêu cầu
            if save_intermediate and self.output_dir:
                output_path = self.output_dir / f"{key}_processed.csv"
                df_with_sentiment.to_csv(output_path, index=False)
                self.logger.info(f"Đã lưu kết quả trung gian: {output_path}")
                
                # Lưu thêm metadata
                metadata = {
                    'original_shape': df.shape,
                    'processed_shape': df_with_sentiment.shape,
                    'steps_executed': self.steps_executed,
                    'execution_times': self.execution_times,
                    'columns_added': list(set(df_with_sentiment.columns) - set(df.columns)),
                    'columns_removed': list(set(df.columns) - set(df_with_sentiment.columns)),
                }
                
                metadata_path = self.output_dir / f"{key}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        # Bước 5: Kết hợp các DataFrame nếu cần
        if len(processed_data) > 1 and 'ohlcv' in processed_data:
            # Lấy DataFrame OHLCV làm chính
            main_df = processed_data['ohlcv']
            timestamp_col = self.config['ohlcv_data']['timestamp_column']
            
            # Kết hợp với các DataFrame khác
            for key, df in processed_data.items():
                if key != 'ohlcv':
                    # Xác định cột timestamp trong DataFrame hiện tại
                    df_timestamp_col = self.config.get(data_types.get(key, ''), {}).get('timestamp_column', timestamp_col)
                    
                    # Chuẩn bị để merge
                    df_to_merge = df.copy()
                    
                    # Đổi tên cột timestamp nếu cần
                    if df_timestamp_col != timestamp_col:
                        df_to_merge = df_to_merge.rename(columns={df_timestamp_col: timestamp_col})
                    
                    # Thêm tiền tố cho các cột (trừ timestamp) để tránh trùng tên
                    prefix = f"{key}_"
                    rename_dict = {col: f"{prefix}{col}" for col in df_to_merge.columns if col != timestamp_col}
                    df_to_merge = df_to_merge.rename(columns=rename_dict)
                    
                    # Merge vào main_df
                    main_df = pd.merge(main_df, df_to_merge, on=timestamp_col, how='left')
            
            processed_data['combined'] = main_df
        
        # Bước 6: Lựa chọn đặc trưng nếu được kích hoạt và có cột mục tiêu
        if self.feature_selector is not None and target_column is not None:
            # Chọn DataFrame chính để lựa chọn đặc trưng
            main_key = 'combined' if 'combined' in processed_data else list(processed_data.keys())[0]
            main_df = processed_data[main_key]
            
            # Kiểm tra cột mục tiêu có tồn tại không
            if target_column in main_df.columns:
                # Lựa chọn đặc trưng
                X = main_df.drop(columns=[target_column])
                y = main_df[target_column]
                
                # Xác định loại bài toán
                problem_type = 'regression'
                if y.dtype == 'bool' or y.dtype == 'object' or (pd.api.types.is_numeric_dtype(y) and len(y.unique()) <= 10):
                    problem_type = 'classification'
                
                # Fit hoặc transform
                if fit:
                    selected_features = self.feature_selector.select_features(X, y, problem_type=problem_type)
                    self.is_fitted = True
                else:
                    if not hasattr(self.feature_selector, 'selected_features') or not self.feature_selector.selected_features:
                        self.logger.warning("Feature selector chưa được fit, không thể transform")
                        selected_features = X.columns.tolist()
                    else:
                        selected_features = self.feature_selector.selected_features
                
                # Lọc các cột được chọn
                selected_cols = selected_features + [target_column]
                self.logger.info(f"Đã chọn {len(selected_features)} đặc trưng từ {X.shape[1]} đặc trưng ban đầu")
                
                # Cập nhật DataFrame với đặc trưng đã chọn
                for key in processed_data:
                    # Chỉ áp dụng cho DataFrame chứa tất cả các cột đã chọn
                    available_cols = [col for col in selected_cols if col in processed_data[key].columns]
                    if len(available_cols) > 0:
                        processed_data[key] = processed_data[key][available_cols]
                
                # Lưu metadata
                self.feature_metadata = {
                    'selected_features': selected_features,
                    'target_column': target_column,
                    'problem_type': problem_type,
                    'feature_importance': self.feature_selector.feature_importance if hasattr(self.feature_selector, 'feature_importance') else {}
                }
            else:
                self.logger.warning(f"Cột mục tiêu '{target_column}' không tồn tại trong dữ liệu, bỏ qua lựa chọn đặc trưng")
        
        # Bước 7: Loại bỏ các dòng NA nếu được cấu hình
        if self.config['general']['drop_na_after_processing']:
            for key in processed_data:
                original_rows = len(processed_data[key])
                processed_data[key] = processed_data[key].dropna()
                removed_rows = original_rows - len(processed_data[key])
                if removed_rows > 0:
                    self.logger.info(f"Đã loại bỏ {removed_rows} dòng có giá trị NA trong '{key}'")
        
        # Tính tổng thời gian xử lý
        elapsed_time = time.time() - start_time
        self.logger.info(f"Đã hoàn thành xử lý dữ liệu trong {elapsed_time:.2f} giây")
        
        return processed_data
    
    def _clean_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Làm sạch dữ liệu tùy theo loại dữ liệu.
        
        Args:
            df: DataFrame cần làm sạch
            data_type: Loại dữ liệu
            
        Returns:
            DataFrame đã làm sạch
        """
        start_time = time.time()
        
        if self.data_cleaner is None:
            self.logger.info("DataCleaner không được kích hoạt, bỏ qua bước làm sạch dữ liệu")
            return df
        
        try:
            # Xử lý tùy theo loại dữ liệu
            if data_type == 'ohlcv_data':
                # Lấy cấu hình OHLCV
                config = self.config['ohlcv_data']
                
                # Đổi tên các cột nếu cần
                rename_columns = {}
                for std_col, df_col in config['ohlcv_columns'].items():
                    if df_col in df.columns and df_col != std_col:
                        rename_columns[df_col] = std_col
                
                if rename_columns:
                    df = df.rename(columns=rename_columns)
                
                # Kiểm tra các cột cần thiết
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    self.logger.warning(f"Thiếu một số cột OHLCV cần thiết: {[col for col in required_cols if col not in df.columns]}")
                
                # Làm sạch dữ liệu OHLCV
                cleaned_df = self.data_cleaner.clean_ohlcv_data(
                    df=df,
                    handle_gaps=config.get('handle_gaps', True),
                    verify_high_low=config.get('verify_high_low', True),
                    verify_open_close=config.get('verify_open_close', True)
                )
                
            elif data_type == 'orderbook_data':
                # Lấy cấu hình orderbook
                config = self.config['orderbook_data']
                
                # Làm sạch dữ liệu orderbook
                cleaned_df = self.data_cleaner.clean_orderbook_data(
                    df=df,
                    verify_price_levels=True,
                    verify_quantities=True,
                    min_quantity_threshold=config.get('min_quantity_threshold', 0.0)
                )
                
            elif data_type == 'trade_data':
                # Lấy cấu hình trade
                config = self.config['trade_data']
                
                # Làm sạch dữ liệu trade
                cleaned_df = self.data_cleaner.clean_trade_data(
                    df=df,
                    verify_price=True,
                    verify_amount=True,
                    min_quantity_threshold=config.get('min_quantity_threshold', 0.0)
                )
                
            elif data_type == 'news_data':
                # Tin tức không cần làm sạch đặc biệt, chỉ làm sạch cơ bản
                columns_to_clean = [col for col in df.columns if col != self.config['news_data']['text_column']]
                cleaned_df = self.data_cleaner.clean_dataframe(
                    df=df,
                    columns_to_clean=columns_to_clean,
                    drop_na=False,
                    normalize=False,
                    remove_outliers=False,
                    handle_missing=self.config['general']['handle_missing_data']
                )
                
            else:
                # Làm sạch cơ bản cho các loại dữ liệu khác
                cleaned_df = self.data_cleaner.clean_dataframe(
                    df=df,
                    drop_na=False,
                    normalize=False,
                    remove_outliers=self.config['general']['handle_outliers'],
                    handle_missing=self.config['general']['handle_missing_data']
                )
                
            elapsed_time = time.time() - start_time
            self.steps_executed.append('clean_data')
            self.execution_times['clean_data'] = elapsed_time
            self.logger.info(f"Đã làm sạch dữ liệu {data_type} ({len(df)} -> {len(cleaned_df)} dòng) trong {elapsed_time:.2f} giây")
            
            # Lưu kết quả vào cache
            self.results_cache['cleaned_df'] = cleaned_df
            
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Lỗi khi làm sạch dữ liệu: {str(e)}")
            if self.debug_mode:
                import traceback
                self.logger.error(traceback.format_exc())
            return df
    
    def _add_date_features(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Thêm các đặc trưng liên quan đến thời gian (ngày trong tuần, giờ trong ngày, v.v.).
        
        Args:
            df: DataFrame cần xử lý
            data_type: Loại dữ liệu
            
        Returns:
            DataFrame với các đặc trưng thời gian bổ sung
        """
        # Xác định cột timestamp dựa trên loại dữ liệu
        timestamp_col = self.config.get(data_type, {}).get('timestamp_column', 'timestamp')
        
        if timestamp_col not in df.columns:
            self.logger.warning(f"Không tìm thấy cột timestamp '{timestamp_col}', bỏ qua việc thêm đặc trưng thời gian")
            return df
        
        try:
            # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
            result_df = df.copy()
            
            # Đảm bảo cột timestamp là datetime
            if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_col]):
                result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])
            
            # Thêm các đặc trưng thời gian
            # 1. Giờ trong ngày (0-23)
            result_df['hour'] = result_df[timestamp_col].dt.hour
            
            # 2. Ngày trong tuần (0=Monday, 6=Sunday)
            result_df['dayofweek'] = result_df[timestamp_col].dt.dayofweek
            
            # 3. Ngày trong tháng (1-31)
            result_df['day'] = result_df[timestamp_col].dt.day
            
            # 4. Tháng trong năm (1-12)
            result_df['month'] = result_df[timestamp_col].dt.month
            
            # 5. Năm
            result_df['year'] = result_df[timestamp_col].dt.year
            
            # 6. Là ngày cuối tuần (0=weekday, 1=weekend)
            result_df['is_weekend'] = result_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
            
            # 7. Là giờ giao dịch (7-19 = 1, khác = 0)
            result_df['is_trading_hour'] = result_df['hour'].apply(lambda x: 1 if 7 <= x <= 19 else 0)
            
            # 8. Quý trong năm (1-4)
            result_df['quarter'] = result_df[timestamp_col].dt.quarter
            
            # 9. Tuần trong năm (1-53)
            result_df['weekofyear'] = result_df[timestamp_col].dt.isocalendar().week.astype(int)
            
            self.steps_executed.append('add_date_features')
            self.logger.info(f"Đã thêm 9 đặc trưng thời gian vào DataFrame")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thêm đặc trưng thời gian: {str(e)}")
            if self.debug_mode:
                import traceback
                self.logger.error(traceback.format_exc())
            return df
    
    def _generate_technical_indicators(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Tạo các chỉ báo kỹ thuật cho dữ liệu OHLCV.
        
        Args:
            df: DataFrame cần xử lý
            data_type: Loại dữ liệu
            
        Returns:
            DataFrame với các chỉ báo kỹ thuật
        """
        start_time = time.time()
        
        if self.technical_indicator_generator is None:
            self.logger.info("TechnicalIndicatorGenerator không được kích hoạt, bỏ qua bước tạo chỉ báo kỹ thuật")
            return df
        
        if data_type != 'ohlcv_data':
            self.logger.info(f"Loại dữ liệu {data_type} không phải OHLCV, bỏ qua tạo chỉ báo kỹ thuật")
            return df
        
        # Kiểm tra các cột cần thiết
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            self.logger.warning(f"Thiếu một số cột OHLCV cần thiết: {[col for col in required_cols if col not in df.columns]}")
            return df
        
        try:
            # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
            result_df = df.copy()
            
            # Tạo các chỉ báo kỹ thuật
            if self.debug_mode:
                self.logger.debug(f"Bắt đầu tạo chỉ báo kỹ thuật với các chỉ báo: {self.config['technical_indicators']['indicators']}")
            
            result_df = self.technical_indicator_generator.generate_indicators(
                df=result_df,
                indicators=self.config['technical_indicators']['indicators'],
                custom_windows=self.config['technical_indicators']['windows']
            )
            
            # Đếm số lượng đặc trưng đã thêm
            new_columns = set(result_df.columns) - set(df.columns)
            
            elapsed_time = time.time() - start_time
            self.steps_executed.append('generate_technical_indicators')
            self.execution_times['generate_technical_indicators'] = elapsed_time
            self.logger.info(f"Đã tạo {len(new_columns)} chỉ báo kỹ thuật trong {elapsed_time:.2f} giây")
            
            # Lưu kết quả vào cache
            self.results_cache['df_with_indicators'] = result_df
            
            # Debug thêm thông tin về các chỉ báo đã thêm
            if self.debug_mode:
                self.logger.debug(f"Các chỉ báo đã thêm: {new_columns}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo các chỉ báo kỹ thuật: {str(e)}")
            if self.debug_mode:
                import traceback
                self.logger.error(traceback.format_exc())
            return df
    
    def _generate_market_features(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Tạo các đặc trưng thị trường cho DataFrame.
        
        Args:
            df: DataFrame cần xử lý
            data_type: Loại dữ liệu
            
        Returns:
            DataFrame với các đặc trưng thị trường
        """
        start_time = time.time()
        
        if self.market_feature_generator is None:
            self.logger.info("MarketFeatureGenerator không được kích hoạt, bỏ qua bước tạo đặc trưng thị trường")
            return df
        
        try:
            # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
            result_df = df.copy()
            
            # Xác định hàm tạo đặc trưng phù hợp với loại dữ liệu
            if data_type == 'ohlcv_data':
                # Kiểm tra các cột cần thiết
                required_cols = ['close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    self.logger.warning(f"Thiếu một số cột cần thiết: {[col for col in required_cols if col not in df.columns]}")
                    return df
                
                # Tạo đặc trưng từ dữ liệu OHLCV
                result_df = self.market_feature_generator.generate_price_features(
                    df=result_df,
                    price_column='close',
                    volume_column='volume'
                )
                
                # Tạo đặc trưng mô hình giá nếu được cấu hình
                if self.config['market_features']['pattern_recognition']:
                    result_df = self.market_feature_generator.detect_candlestick_patterns(result_df)
                
            elif data_type == 'orderbook_data':
                # Các đặc trưng orderbook
                config = self.config['orderbook_data']
                price_col = config.get('price_column', 'price')
                amount_col = config.get('amount_column', 'amount')
                side_col = config.get('side_column', 'side')
                
                if all(col in df.columns for col in [price_col, amount_col, side_col]):
                    result_df = self.market_feature_generator.generate_orderbook_features(
                        df=result_df,
                        price_column=price_col,
                        amount_column=amount_col,
                        side_column=side_col
                    )
                else:
                    self.logger.warning(f"Thiếu các cột orderbook cần thiết, bỏ qua tạo đặc trưng orderbook")
                
            elif data_type == 'trade_data':
                # Các đặc trưng giao dịch
                config = self.config['trade_data']
                price_col = config.get('price_column', 'price')
                amount_col = config.get('amount_column', 'amount')
                
                if all(col in df.columns for col in [price_col, amount_col]):
                    result_df = self.market_feature_generator.generate_trade_features(
                        df=result_df,
                        price_column=price_col,
                        amount_column=amount_col
                    )
                else:
                    self.logger.warning(f"Thiếu các cột trade cần thiết, bỏ qua tạo đặc trưng trade")
            
            else:
                # Các đặc trưng chung
                price_columns = [col for col in df.columns if 'price' in col.lower() or 'close' in col.lower()]
                volume_columns = [col for col in df.columns if 'volume' in col.lower() or 'amount' in col.lower()]
                
                if price_columns and volume_columns:
                    # Chọn cột đầu tiên tìm thấy
                    price_col = price_columns[0]
                    volume_col = volume_columns[0]
                    
                    result_df = self.market_feature_generator.generate_price_features(
                        df=result_df,
                        price_column=price_col,
                        volume_column=volume_col
                    )
            
            # Đếm số lượng đặc trưng đã thêm
            new_columns = set(result_df.columns) - set(df.columns)
            
            elapsed_time = time.time() - start_time
            self.steps_executed.append('generate_market_features')
            self.execution_times['generate_market_features'] = elapsed_time
            self.logger.info(f"Đã tạo {len(new_columns)} đặc trưng thị trường trong {elapsed_time:.2f} giây")
            
            # Lưu kết quả vào cache
            self.results_cache['df_with_market_features'] = result_df
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo các đặc trưng thị trường: {str(e)}")
            if self.debug_mode:
                import traceback
                self.logger.error(traceback.format_exc())
            return df
    
    def _process_sentiment(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Xử lý và trích xuất đặc trưng tâm lý từ dữ liệu tin tức.
        
        Args:
            df: DataFrame cần xử lý
            data_type: Loại dữ liệu
            
        Returns:
            DataFrame với các đặc trưng tâm lý
        """
        start_time = time.time()
        
        if self.sentiment_feature_extractor is None:
            self.logger.info("SentimentFeatureExtractor không được kích hoạt, bỏ qua bước xử lý tâm lý")
            return df
        
        if data_type != 'news_data':
            self.logger.info(f"Loại dữ liệu {data_type} không phải dữ liệu tin tức, bỏ qua xử lý tâm lý")
            return df
        
        try:
            # Lấy cấu hình
            config = self.config['news_data']
            text_column = config.get('text_column', 'content')
            timestamp_column = config.get('timestamp_column', 'timestamp')
            entity_column = config.get('entity_column', 'entity')
            source_column = config.get('source_column', 'source')
            
            # Kiểm tra cột văn bản
            if text_column not in df.columns:
                self.logger.warning(f"Không tìm thấy cột văn bản '{text_column}', bỏ qua xử lý tâm lý")
                return df
            
            # Xử lý đặc trưng tâm lý
            result_df = self.sentiment_feature_extractor.process_news_dataframe(
                df=df,
                text_column=text_column,
                timestamp_column=timestamp_column,
                entity_column=entity_column if entity_column in df.columns else None,
                source_column=source_column if source_column in df.columns else None
            )
            
            # Đếm số lượng đặc trưng đã thêm
            new_columns = set(result_df.columns) - set(df.columns)
            
            elapsed_time = time.time() - start_time
            self.steps_executed.append('process_sentiment')
            self.execution_times['process_sentiment'] = elapsed_time
            self.logger.info(f"Đã trích xuất {len(new_columns)} đặc trưng tâm lý trong {elapsed_time:.2f} giây")
            
            # Lưu kết quả vào cache
            self.results_cache['df_with_sentiment'] = result_df
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Lỗi khi trích xuất đặc trưng tâm lý: {str(e)}")
            if self.debug_mode:
                import traceback
                self.logger.error(traceback.format_exc())
            return df
    
    def save_pipeline(self, filepath: Optional[str] = None) -> str:
        """
        Lưu pipeline vào file.
        
        Args:
            filepath: Đường dẫn file để lưu (None để tạo tự động)
            
        Returns:
            Đường dẫn file đã lưu
        """
        if filepath is None:
            if self.output_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.output_dir / f"{self.pipeline_name}_{timestamp}.pkl"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = Path(f"./pipeline_{self.pipeline_name}_{timestamp}.pkl")
        else:
            filepath = Path(filepath)
        
        # Tạo thư mục cha nếu chưa tồn tại
        filepath.parent.mkdir(exist_ok=True, parents=True)
        
        # Dữ liệu cần lưu
        data_to_save = {
            'pipeline_name': self.pipeline_name,
            'config': self.config,
            'steps_executed': self.steps_executed,
            'execution_times': self.execution_times,
            'feature_metadata': self.feature_metadata,
            'is_fitted': self.is_fitted,
            'saved_at': datetime.now().isoformat()
        }
        
        # Nếu feature_selector đã fit, lưu selected_features
        if self.feature_selector is not None and hasattr(self.feature_selector, 'selected_features'):
            data_to_save['selected_features'] = self.feature_selector.selected_features
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
                
            self.logger.info(f"Đã lưu pipeline vào {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu pipeline: {str(e)}")
            if self.debug_mode:
                import traceback
                self.logger.error(traceback.format_exc())
            return ""
    
    @classmethod
    def load_pipeline(cls, filepath: str) -> 'DataPipeline':
        """
        Tải pipeline từ file.
        
        Args:
            filepath: Đường dẫn file
            
        Returns:
            Đối tượng DataPipeline đã tải
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Không tìm thấy file pipeline: {filepath}")
        
        logger = setup_logger("data_pipeline")
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Tạo pipeline mới với cấu hình đã lưu
            pipeline = cls(
                config=data['config'],
                pipeline_name=data['pipeline_name'],
                output_dir=filepath.parent,
                debug_mode=False
            )
            
            # Khôi phục trạng thái
            pipeline.steps_executed = data.get('steps_executed', [])
            pipeline.execution_times = data.get('execution_times', {})
            pipeline.feature_metadata = data.get('feature_metadata', {})
            pipeline.is_fitted = data.get('is_fitted', False)
            
            # Khôi phục selected_features nếu có
            if 'selected_features' in data and pipeline.feature_selector is not None:
                pipeline.feature_selector.selected_features = data['selected_features']
            
            logger.info(f"Đã tải pipeline từ {filepath} (lưu lúc {data.get('saved_at', 'unknown')})")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Lỗi khi tải pipeline: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def get_feature_metadata(self) -> Dict[str, Any]:
        """
        Lấy metadata cho các đặc trưng đã tạo.
        
        Returns:
            Dictionary chứa metadata
        """
        metadata = {
            'pipeline_name': self.pipeline_name,
            'steps_executed': self.steps_executed,
            'execution_times': self.execution_times,
            'feature_metadata': self.feature_metadata,
            'is_fitted': self.is_fitted
        }
        
        return metadata
    
    def visualize_feature_importance(
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
        if self.feature_selector is None or not hasattr(self.feature_selector, 'feature_importance_df'):
            self.logger.warning("Không có thông tin về tầm quan trọng của đặc trưng, không thể vẽ biểu đồ")
            return
        
        try:
            self.feature_selector.plot_feature_importance(
                top_n=top_n,
                figsize=figsize,
                save_path=save_path
            )
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ biểu đồ tầm quan trọng của đặc trưng: {str(e)}")
    
    def visualize_pipeline_steps(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> None:
        """
        Vẽ biểu đồ thời gian thực thi của các bước trong pipeline.
        
        Args:
            figsize: Kích thước của biểu đồ
            save_path: Đường dẫn để lưu biểu đồ, None nếu không lưu
        """
        if not self.execution_times:
            self.logger.warning("Không có thông tin về thời gian thực thi, không thể vẽ biểu đồ")
            return
        
        try:
            # Chuẩn bị dữ liệu
            steps = list(self.execution_times.keys())
            times = list(self.execution_times.values())
            
            # Vẽ biểu đồ
            plt.figure(figsize=figsize)
            bars = plt.barh(steps, times)
            
            # Thêm nhãn và tiêu đề
            plt.xlabel('Thời gian thực thi (giây)')
            plt.ylabel('Bước xử lý')
            plt.title(f'Thời gian thực thi các bước trong pipeline "{self.pipeline_name}"')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Thêm giá trị cụ thể lên đồ thị
            for i, (bar, time) in enumerate(zip(bars, times)):
                plt.text(bar.get_width() + 0.1, i, f"{time:.2f}s", va='center')
            
            plt.tight_layout()
            
            # Lưu biểu đồ nếu có đường dẫn
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Đã lưu biểu đồ thời gian thực thi vào {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ biểu đồ thời gian thực thi: {str(e)}")
    
    def generate_feature_report(
        self,
        data: Dict[str, pd.DataFrame],
        report_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tạo báo cáo về các đặc trưng đã tạo.
        
        Args:
            data: Dictionary chứa các DataFrame đã xử lý
            report_path: Đường dẫn để lưu báo cáo, None nếu không lưu
            
        Returns:
            Dictionary chứa báo cáo về đặc trưng
        """
        if not data:
            self.logger.warning("Không có dữ liệu để tạo báo cáo")
            return {}
        
        report = {
            'pipeline_name': self.pipeline_name,
            'created_at': datetime.now().isoformat(),
            'dataframes': {},
            'feature_metadata': self.feature_metadata,
            'execution_times': self.execution_times,
            'steps_executed': self.steps_executed,
            'config_summary': {
                'handle_missing_data': self.config['general']['handle_missing_data'],
                'handle_outliers': self.config['general']['handle_outliers'],
                'generate_technical_indicators': self.config['general']['generate_technical_indicators'],
                'generate_market_features': self.config['general']['generate_market_features'],
                'process_sentiment': self.config['general']['process_sentiment'],
                'select_features': self.config['general']['select_features']
            }
        }
        
        # Thông tin về từng DataFrame
        for key, df in data.items():
            # Thống kê cơ bản
            df_info = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_list': df.columns.tolist(),
                'missing_values': df.isna().sum().sum(),
                'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
                'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
            }
            
            # Thống kê chi tiết cho các cột số
            if df_info['numeric_columns']:
                df_info['numeric_stats'] = df[df_info['numeric_columns']].describe().to_dict()
            
            report['dataframes'][key] = df_info
        
        # Thông tin về đặc trưng đã chọn
        if self.feature_selector is not None and hasattr(self.feature_selector, 'selected_features'):
            report['selected_features'] = self.feature_selector.selected_features
            report['feature_importance'] = None
            
            if hasattr(self.feature_selector, 'feature_importance_df'):
                report['feature_importance'] = self.feature_selector.feature_importance_df.to_dict(orient='records')
        
        # Lưu báo cáo nếu có đường dẫn
        if report_path:
            try:
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                self.logger.info(f"Đã lưu báo cáo đặc trưng vào {report_path}")
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu báo cáo: {str(e)}")
        
        return report


class DataPipelineFactory:
    """
    Factory để tạo các DataPipeline với cấu hình khác nhau.
    """
    
    @staticmethod
    def create_basic_pipeline(output_dir: Optional[str] = None) -> DataPipeline:
        """
        Tạo pipeline xử lý dữ liệu cơ bản.
        
        Args:
            output_dir: Thư mục lưu kết quả
            
        Returns:
            DataPipeline đã cấu hình
        """
        # Cấu hình cơ bản
        config = {
            'general': {
                'handle_missing_data': True,
                'handle_outliers': True,
                'generate_technical_indicators': True,
                'generate_market_features': True,
                'process_sentiment': False,
                'select_features': False,
                'drop_na_after_processing': True,
                'add_date_features': True
            },
            
            'data_cleaner': {
                'enabled': True,
                'normalize_method': 'z-score',
                'outlier_detector_kwargs': {
                    'method': 'z-score',
                    'threshold': 3.0,
                    'use_robust': True
                },
                'missing_data_handler_kwargs': {
                    'method': 'interpolate',
                    'handle_categorical': True
                }
            },
            
            'technical_indicators': {
                'enabled': True,
                'indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands'],
                'windows': [5, 10, 20, 50],
                'fillna': True
            },
            
            'market_features': {
                'enabled': True,
                'price_differences': True,
                'price_ratios': True,
                'volume_features': True,
                'volatility_features': True,
                'pattern_recognition': False,
                'custom_features': []
            },
            
            'sentiment_features': {
                'enabled': False
            },
            
            'feature_selector': {
                'enabled': False
            }
        }
        
        return DataPipeline(config=config, pipeline_name="basic_pipeline", output_dir=output_dir)
    
    @staticmethod
    def create_advanced_pipeline(output_dir: Optional[str] = None) -> DataPipeline:
        """
        Tạo pipeline xử lý dữ liệu nâng cao.
        
        Args:
            output_dir: Thư mục lưu kết quả
            
        Returns:
            DataPipeline đã cấu hình
        """
        # Cấu hình nâng cao
        config = {
            'general': {
                'handle_missing_data': True,
                'handle_outliers': True,
                'generate_technical_indicators': True,
                'generate_market_features': True,
                'process_sentiment': True,
                'select_features': True,
                'drop_na_after_processing': True,
                'add_date_features': True
            },
            
            'data_cleaner': {
                'enabled': True,
                'normalize_method': 'robust',
                'outlier_detector_kwargs': {
                    'method': 'isolation-forest',
                    'threshold': 0.05,
                    'use_robust': True,
                    'contamination': 0.05
                },
                'missing_data_handler_kwargs': {
                    'method': 'knn',
                    'knn_neighbors': 5,
                    'handle_categorical': True
                }
            },
            
            'technical_indicators': {
                'enabled': True,
                'indicators': [
                    'sma', 'ema', 'wma', 'macd', 'rsi', 'bollinger_bands',
                    'stochastic', 'atr', 'adx', 'obv', 'momentum', 'ichimoku',
                    'vwap', 'roc', 'cci', 'williams_r', 'trix', 'keltner', 'donchian'
                ],
                'windows': [5, 10, 20, 50, 100, 200],
                'fillna': True
            },
            
            'market_features': {
                'enabled': True,
                'price_differences': True,
                'price_ratios': True,
                'volume_features': True,
                'volatility_features': True,
                'pattern_recognition': True,
                'custom_features': ['zigzag', 'support_resistance', 'liquidity']
            },
            
            'sentiment_features': {
                'enabled': True,
                'sentiment_method': 'vader',
                'language': 'en',
                'normalize_scores': True,
                'volume_weighted': True
            },
            
            'feature_selector': {
                'enabled': True,
                'method': 'random_forest',
                'n_features': 50,
                'threshold': 0.01,
                'handle_collinearity': True,
                'collinearity_threshold': 0.9
            }
        }
        
        return DataPipeline(config=config, pipeline_name="advanced_pipeline", output_dir=output_dir)
    
    @staticmethod
    def create_production_pipeline(output_dir: Optional[str] = None) -> DataPipeline:
        """
        Tạo pipeline xử lý dữ liệu tối ưu hóa cho môi trường sản xuất.
        
        Args:
            output_dir: Thư mục lưu kết quả
            
        Returns:
            DataPipeline đã cấu hình
        """
        # Cấu hình tối ưu hóa cho sản xuất
        config = {
            'general': {
                'handle_missing_data': True,
                'handle_outliers': True,
                'generate_technical_indicators': True,
                'generate_market_features': True,
                'process_sentiment': False,
                'select_features': True,
                'drop_na_after_processing': True,
                'add_date_features': True
            },
            
            'data_cleaner': {
                'enabled': True,
                'normalize_method': 'z-score',
                'outlier_detector_kwargs': {
                    'method': 'z-score',
                    'threshold': 3.0,
                    'use_robust': True
                },
                'missing_data_handler_kwargs': {
                    'method': 'interpolate',
                    'handle_categorical': True
                }
            },
            
            'technical_indicators': {
                'enabled': True,
                'indicators': [
                    'sma', 'ema', 'macd', 'rsi', 'bollinger_bands',
                    'stochastic', 'atr', 'adx', 'obv', 'momentum'
                ],
                'windows': [5, 10, 20, 50, 100],
                'fillna': True
            },
            
            'market_features': {
                'enabled': True,
                'price_differences': True,
                'price_ratios': True,
                'volume_features': True,
                'volatility_features': True,
                'pattern_recognition': False,
                'custom_features': []
            },
            
            'sentiment_features': {
                'enabled': False
            },
            
            'feature_selector': {
                'enabled': True,
                'method': 'correlation',
                'n_features': 30,
                'threshold': 0.01,
                'handle_collinearity': True,
                'collinearity_threshold': 0.9
            }
        }
        
        return DataPipeline(config=config, pipeline_name="production_pipeline", output_dir=output_dir)
    
    @staticmethod
    def create_custom_pipeline(
        config: Dict[str, Any],
        pipeline_name: str = "custom_pipeline",
        output_dir: Optional[str] = None
    ) -> DataPipeline:
        """
        Tạo pipeline xử lý dữ liệu với cấu hình tùy chỉnh.
        
        Args:
            config: Cấu hình tùy chỉnh
            pipeline_name: Tên của pipeline
            output_dir: Thư mục lưu kết quả
            
        Returns:
            DataPipeline đã cấu hình
        """
        return DataPipeline(config=config, pipeline_name=pipeline_name, output_dir=output_dir)