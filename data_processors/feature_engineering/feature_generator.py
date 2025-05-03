"""
Lớp chính để tạo đặc trưng từ dữ liệu thị trường.
File này cung cấp lớp FeatureGenerator để quản lý, tạo, và xuất các đặc trưng
từ các module con khác nhau, tạo thành một pipeline xử lý đặc trưng đầy đủ.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Callable, Set, Tuple
from pathlib import Path
import joblib
import json
from datetime import datetime
import concurrent.futures
from functools import partial

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import setup_logger
from config.system_config import BASE_DIR, MODEL_DIR
from data_processors.feature_engineering.utils.validation import validate_features, check_feature_integrity
from data_processors.feature_engineering.utils.preprocessing import normalize_features, standardize_features, min_max_scale
from data_processors.feature_engineering.feature_selector.statistical_methods import correlation_selector

# Các import mặc định khi các module đã được phát triển
try:
    from data_processors.feature_engineering.technical_indicators.trend_indicators import (
    exponential_moving_average as ema,
    simple_moving_average as sma
)
except ImportError:
    pass

try:
    from data_processors.feature_engineering.technical_indicators.momentum_indicators import (
    relative_strength_index as rsi,
    moving_average_convergence_divergence as macd
)
except ImportError:
    pass

try:
    from data_processors.feature_engineering.technical_indicators.volatility_indicators import (
    average_true_range as atr,
    bollinger_bands
)
except ImportError:
    pass

try:
    from data_processors.feature_engineering.technical_indicators.volume_indicators import (
    on_balance_volume as obv
)
except ImportError:
    pass

try:
    from data_processors.feature_engineering.technical_indicators.support_resistance import (
    detect_support_resistance as support_resistance_zones
)
except ImportError:
    pass


class FeatureGenerator:
    """
    Lớp chính để tạo và quản lý đặc trưng cho dữ liệu thị trường.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        feature_config: Optional[Dict[str, Any]] = None,
        max_workers: int = 4,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo trình tạo đặc trưng.
        
        Args:
            data_dir: Thư mục lưu trữ cấu hình và mô hình
            feature_config: Cấu hình đặc trưng
            max_workers: Số luồng tối đa cho xử lý song song
            logger: Logger hiện có, nếu có
        """
        # Thiết lập thư mục lưu trữ
        if data_dir is None:
            self.data_dir = MODEL_DIR / 'feature_engineering'
        else:
            self.data_dir = data_dir
        
        # Tạo thư mục nếu chưa tồn tại
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập logger
        if logger is None:
            self.logger = setup_logger("feature_generator")
        else:
            self.logger = logger
        
        # Thiết lập cấu hình đặc trưng
        self.feature_config = feature_config or {}
        
        # Số luồng tối đa cho xử lý song song
        self.max_workers = max_workers
        
        # Danh sách các đặc trưng đã đăng ký
        self.registered_features = {}
        
        # Danh sách các bộ tiền xử lý
        self.preprocessors = {}
        
        # Danh sách các biến đổi
        self.transformers = {}
        
        # Các bộ chọn lọc đặc trưng
        self.feature_selectors = {}
        
        # Trạng thái và cache
        self.is_fitted = False
        self.feature_info = {}
        
        self.logger.info("Đã khởi tạo FeatureGenerator")
    
    def register_feature(
        self,
        feature_name: str,
        feature_func: Callable,
        feature_params: Dict[str, Any] = None,
        dependencies: List[str] = None,
        feature_type: str = "technical",
        description: str = "",
        is_enabled: bool = True
    ) -> None:
        """
        Đăng ký một đặc trưng mới.
        
        Args:
            feature_name: Tên đặc trưng
            feature_func: Hàm để tính toán đặc trưng
            feature_params: Tham số cho hàm
            dependencies: Các cột dữ liệu cần thiết
            feature_type: Loại đặc trưng (technical, market, sentiment)
            description: Mô tả đặc trưng
            is_enabled: Bật/tắt đặc trưng
        """
        self.registered_features[feature_name] = {
            "name": feature_name,
            "function": feature_func,
            "params": feature_params or {},
            "dependencies": dependencies or [],
            "type": feature_type,
            "description": description,
            "is_enabled": is_enabled
        }
        
        self.logger.debug(f"Đã đăng ký đặc trưng: {feature_name}")
    
    def register_preprocessor(
        self,
        name: str,
        preprocessor_func: Callable,
        params: Dict[str, Any] = None,
        target_columns: List[str] = None,
        apply_by_default: bool = False
    ) -> None:
        """
        Đăng ký bộ tiền xử lý.
        
        Args:
            name: Tên bộ tiền xử lý
            preprocessor_func: Hàm tiền xử lý
            params: Tham số cho hàm
            target_columns: Các cột cần áp dụng
            apply_by_default: Áp dụng mặc định
        """
        self.preprocessors[name] = {
            "name": name,
            "function": preprocessor_func,
            "params": params or {},
            "target_columns": target_columns or [],
            "apply_by_default": apply_by_default,
            "fitted_params": None
        }
        
        self.logger.debug(f"Đã đăng ký bộ tiền xử lý: {name}")
    
    def register_transformer(
        self,
        name: str,
        transformer_func: Callable,
        params: Dict[str, Any] = None,
        target_columns: List[str] = None,
        is_stateful: bool = True
    ) -> None:
        """
        Đăng ký biến đổi.
        
        Args:
            name: Tên biến đổi
            transformer_func: Hàm biến đổi
            params: Tham số cho hàm
            target_columns: Các cột cần áp dụng
            is_stateful: Biến đổi có trạng thái
        """
        self.transformers[name] = {
            "name": name,
            "function": transformer_func,
            "params": params or {},
            "target_columns": target_columns or [],
            "is_stateful": is_stateful,
            "fitted_state": None
        }
        
        self.logger.debug(f"Đã đăng ký biến đổi: {name}")
    
    def register_feature_selector(
        self,
        name: str,
        selector_func: Callable,
        params: Dict[str, Any] = None,
        feature_types: List[str] = None,
        max_features: Optional[int] = None
    ) -> None:
        """
        Đăng ký bộ chọn lọc đặc trưng.
        
        Args:
            name: Tên bộ chọn lọc
            selector_func: Hàm chọn lọc
            params: Tham số cho hàm
            feature_types: Loại đặc trưng cần chọn lọc
            max_features: Số đặc trưng tối đa
        """
        self.feature_selectors[name] = {
            "name": name,
            "function": selector_func,
            "params": params or {},
            "feature_types": feature_types or ["technical", "market", "sentiment"],
            "max_features": max_features,
            "selected_features": None
        }
        
        self.logger.debug(f"Đã đăng ký bộ chọn lọc đặc trưng: {name}")
    
    def load_feature_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Tải cấu hình đặc trưng từ file.
        
        Args:
            config_path: Đường dẫn file cấu hình
        """
        if config_path is None:
            config_path = self.data_dir / "feature_config.json"
        else:
            config_path = Path(config_path)
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.feature_config = json.load(f)
            
            # Cập nhật trạng thái enabled của các đặc trưng
            for feature_name, config in self.feature_config.get("features", {}).items():
                if feature_name in self.registered_features:
                    self.registered_features[feature_name]["is_enabled"] = config.get("is_enabled", True)
                    self.registered_features[feature_name]["params"].update(config.get("params", {}))
            
            self.logger.info(f"Đã tải cấu hình đặc trưng từ {config_path}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải cấu hình đặc trưng: {e}")
    
    def save_feature_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Lưu cấu hình đặc trưng vào file.
        
        Args:
            config_path: Đường dẫn file cấu hình
        """
        if config_path is None:
            config_path = self.data_dir / "feature_config.json"
        else:
            config_path = Path(config_path)
        
        try:
            # Tạo cấu hình để lưu
            config_to_save = {
                "version": "1.0.0",
                "last_updated": datetime.now().isoformat(),
                "features": {},
                "preprocessors": {},
                "transformers": {},
                "feature_selectors": {}
            }
            
            # Lưu thông tin đặc trưng
            for name, feature_info in self.registered_features.items():
                config_to_save["features"][name] = {
                    "type": feature_info["type"],
                    "is_enabled": feature_info["is_enabled"],
                    "params": feature_info["params"],
                    "description": feature_info["description"]
                }
            
            # Lưu thông tin bộ tiền xử lý
            for name, preprocessor_info in self.preprocessors.items():
                config_to_save["preprocessors"][name] = {
                    "apply_by_default": preprocessor_info["apply_by_default"],
                    "params": preprocessor_info["params"],
                    "target_columns": preprocessor_info["target_columns"]
                }
            
            # Lưu thông tin biến đổi
            for name, transformer_info in self.transformers.items():
                config_to_save["transformers"][name] = {
                    "is_stateful": transformer_info["is_stateful"],
                    "params": transformer_info["params"],
                    "target_columns": transformer_info["target_columns"]
                }
            
            # Lưu thông tin bộ chọn lọc đặc trưng
            for name, selector_info in self.feature_selectors.items():
                config_to_save["feature_selectors"][name] = {
                    "params": selector_info["params"],
                    "feature_types": selector_info["feature_types"],
                    "max_features": selector_info["max_features"]
                }
            
            # Ghi file
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_to_save, f, indent=4)
            
            self.logger.info(f"Đã lưu cấu hình đặc trưng vào {config_path}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu cấu hình đặc trưng: {e}")
    
        """
        Đăng ký các đặc trưng mặc định phổ biến.
        """
    def register_default_features(self, all_indicators: bool = False) -> None:
        self.logger.info("Đăng ký các đặc trưng mặc định")

        if all_indicators:
            self._register_all_technical_indicators()

        
        # Đăng ký các đặc trưng khi các module con đã được phát triển
    
        try:
            self._register_default_technical_indicators()
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Không thể đăng ký technical indicators: {e}")
        
        try:
            self._register_default_market_features()
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Không thể đăng ký market features: {e}")
        
        try:
            self._register_default_sentiment_features()
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Không thể đăng ký sentiment features: {e}")
        
        # Đăng ký các bộ tiền xử lý mặc định
        self._register_default_preprocessors()
        
        # Đăng ký các biến đổi mặc định
        self._register_default_transformers()
        
        # Đăng ký các bộ chọn lọc đặc trưng mặc định
        self._register_default_feature_selectors()
    
    def _register_default_technical_indicators(self) -> None:
        """
        Đăng ký các chỉ báo kỹ thuật mặc định.
        """
        # Các đặc trưng này sẽ được triển khai khi module cụ thể đã phát triển
        pass
    
    def _register_default_market_features(self) -> None:
        """
        Đăng ký các đặc trưng thị trường mặc định.
        """
        # Các đặc trưng này sẽ được triển khai khi module cụ thể đã phát triển
        pass
    
    def _register_default_sentiment_features(self) -> None:
        """
        Đăng ký các đặc trưng tâm lý mặc định.
        """
        # Các đặc trưng này sẽ được triển khai khi module cụ thể đã phát triển
        pass
    
    def _register_default_preprocessors(self) -> None:
        """
        Đăng ký các bộ tiền xử lý mặc định.
        """
        # Đăng ký các bộ tiền xử lý như chuẩn hóa, xóa giá trị ngoại lai, etc.
        self.register_preprocessor(
            name="normalize",
            preprocessor_func=normalize_features,
            params={"method": "zscore"},
            apply_by_default=False
        )
        
        self.register_preprocessor(
            name="standardize",
            preprocessor_func=standardize_features,
            params={},
            apply_by_default=False
        )
        
        self.register_preprocessor(
            name="min_max_scale",
            preprocessor_func=min_max_scale,
            params={"min_val": 0, "max_val": 1},
            apply_by_default=False
        )
    
    def _register_default_transformers(self) -> None:
        """
        Đăng ký các biến đổi mặc định.
        """
        # Các biến đổi này sẽ được triển khai khi module cụ thể đã phát triển
        pass
    
    def _register_default_feature_selectors(self) -> None:
        """
        Đăng ký các bộ chọn lọc đặc trưng mặc định.
        """
        self.register_feature_selector(
            name="statistical_correlation",  
            selector_func=correlation_selector,
            params={
                "threshold": 0.05,  # Ngưỡng tương quan thấp hơn để chọn nhiều đặc trưng hơn
                "k": 30  # Chọn 30 đặc trưng tốt nhất nếu không có đủ đặc trưng vượt qua ngưỡng
            }
        )
        # Các bộ chọn lọc này sẽ được triển khai khi module cụ thể đã phát triển
        pass
    
    def _check_dependencies(self, df: pd.DataFrame, feature_info: Dict) -> bool:
        """
        Kiểm tra xem dataframe có chứa các cột phụ thuộc không.
        
        Args:
            df: DataFrame cần kiểm tra
            feature_info: Thông tin đặc trưng
            
        Returns:
            True nếu tất cả các phụ thuộc đều có mặt
        """
        dependencies = feature_info.get("dependencies", [])
        
        if not dependencies:
            return True
        
        missing_cols = [col for col in dependencies if col not in df.columns]
        
        if missing_cols:
            self.logger.warning(f"Thiếu các cột phụ thuộc cho {feature_info['name']}: {missing_cols}")
            return False
        
        return True
    
    def _calculate_feature(self, df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """
        Tính toán một đặc trưng cụ thể.
        
        Args:
            df: DataFrame chứa dữ liệu
            feature_name: Tên đặc trưng cần tính
            
        Returns:
            DataFrame với đặc trưng mới
        """
        if feature_name not in self.registered_features:
            self.logger.warning(f"Đặc trưng {feature_name} chưa được đăng ký")
            return df
        
        feature_info = self.registered_features[feature_name]
        
        if not feature_info["is_enabled"]:
            return df
        
        if not self._check_dependencies(df, feature_info):
            return df
        
        try:
            # Tạo bản sao để tránh thay đổi trực tiếp
            result_df = df.copy()
            
            # Tính toán đặc trưng
            feature_func = feature_info["function"]
            feature_params = feature_info["params"]
            
            # Gọi hàm tính toán đặc trưng
            result = feature_func(result_df, **feature_params)
            
            # Cập nhật DataFrame với kết quả mới
            if isinstance(result, pd.DataFrame):
                # Kết quả là DataFrame
                for col in result.columns:
                    if col not in df.columns:
                        result_df[col] = result[col]
            elif isinstance(result, pd.Series):
                # Kết quả là Series
                if result.name and result.name not in df.columns:
                    result_df[result.name] = result
                else:
                    result_df[feature_name] = result
            else:
                # Kết quả là giá trị đơn lẻ hoặc mảng
                result_df[feature_name] = result
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính toán đặc trưng {feature_name}: {e}")
            return df
    
    def calculate_features(
        self,
        df: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        Tính toán nhiều đặc trưng.
        
        Args:
            df: DataFrame chứa dữ liệu
            feature_names: Danh sách tên đặc trưng cần tính (None để tính tất cả)
            parallel: Tính toán song song
            
        Returns:
            DataFrame với các đặc trưng mới
        """
        if feature_names is None:
            # Tính tất cả các đặc trưng đã bật
            feature_names = [name for name, info in self.registered_features.items() if info["is_enabled"]]
        
        # Tạo bản sao để tránh thay đổi trực tiếp
        result_df = df.copy()
        
        if not feature_names:
            return result_df
        
        # Xây dựng đồ thị phụ thuộc và sắp xếp thứ tự tính toán
        dependencies_graph = {}
        for name in feature_names:
            if name in self.registered_features:
                dependencies = self.registered_features[name].get("dependencies", [])
                dependencies_graph[name] = [dep for dep in dependencies if dep in feature_names]
        
        # Sắp xếp topo để đảm bảo các phụ thuộc được tính trước
        sorted_features = self._topological_sort(dependencies_graph)
        
        if parallel and len(sorted_features) > 1:
            # Chia nhỏ danh sách tính năng thành các nhóm không phụ thuộc
            feature_groups = self._group_independent_features(dependencies_graph, sorted_features)
            
            # Tính toán từng nhóm, các tính năng trong một nhóm không phụ thuộc lẫn nhau
            for group in feature_groups:
                if len(group) == 1:
                    # Chỉ có một tính năng trong nhóm, không cần song song
                    feature_name = group[0]
                    result_df = self._calculate_feature(result_df, feature_name)
                else:
                    # Nhiều tính năng trong nhóm, tính toán song song
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(group))) as executor:
                        futures = []
                        for feature_name in group:
                            future = executor.submit(self._calculate_feature, result_df, feature_name)
                            futures.append((feature_name, future))
                        
                        # Thu thập kết quả
                        for feature_name, future in futures:
                            try:
                                feature_df = future.result()
                                # Thêm cột mới vào kết quả
                                for col in feature_df.columns:
                                    if col not in df.columns and col not in result_df.columns:
                                        result_df[col] = feature_df[col]
                            except Exception as e:
                                self.logger.error(f"Lỗi khi tính toán đặc trưng {feature_name}: {e}")
        else:
            # Tính tuần tự
            for feature_name in sorted_features:
                result_df = self._calculate_feature(result_df, feature_name)
        
        # Kiểm tra và ghi log các đặc trưng đã tạo
        created_features = [col for col in result_df.columns if col not in df.columns]
        self.logger.info(f"Đã tạo {len(created_features)} đặc trưng mới: {', '.join(created_features)}")
        
        return result_df
    
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """
        Sắp xếp topo để đảm bảo các phụ thuộc được tính trước.
        
        Args:
            graph: Đồ thị phụ thuộc
            
        Returns:
            Danh sách đã sắp xếp
        """
        # Khởi tạo
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node):
            # Phát hiện chu trình
            if node in temp_visited:
                raise ValueError(f"Phát hiện phụ thuộc vòng lặp với {node}")
            
            # Đã thăm nút này rồi
            if node in visited:
                return
            
            # Đánh dấu đang thăm
            temp_visited.add(node)
            
            # Thăm các nút liền kề
            for neighbor in graph.get(node, []):
                visit(neighbor)
            
            # Đã thăm xong
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        # Thăm từng nút
        for node in graph:
            if node not in visited:
                visit(node)
        
        # Kết quả là thứ tự ngược lại
        return list(reversed(result))
    
    def _group_independent_features(
        self,
        dependencies_graph: Dict[str, List[str]],
        sorted_features: List[str]
    ) -> List[List[str]]:
        """
        Nhóm các đặc trưng độc lập để tính toán song song.
        
        Args:
            dependencies_graph: Đồ thị phụ thuộc
            sorted_features: Danh sách đặc trưng đã sắp xếp
            
        Returns:
            Danh sách các nhóm đặc trưng không phụ thuộc lẫn nhau
        """
        result = []
        current_group = []
        computed_features = set()
        
        for feature in sorted_features:
            # Kiểm tra xem tính năng này có phụ thuộc vào các tính năng chưa tính không
            dependencies = dependencies_graph.get(feature, [])
            has_uncomputed_deps = any(dep not in computed_features for dep in dependencies)
            
            if has_uncomputed_deps:
                # Nếu có phụ thuộc chưa tính, phải tính các nhóm trước đó trước
                if current_group:
                    result.append(current_group)
                    for f in current_group:
                        computed_features.add(f)
                    current_group = []
            
            # Thêm vào nhóm hiện tại
            current_group.append(feature)
        
        # Thêm nhóm cuối cùng
        if current_group:
            result.append(current_group)
        
        return result
    
    def apply_preprocessors(
        self,
        df: pd.DataFrame,
        preprocessor_names: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Áp dụng các bộ tiền xử lý lên dữ liệu.
        
        Args:
            df: DataFrame cần xử lý
            preprocessor_names: Danh sách tên bộ tiền xử lý cần áp dụng
            target_columns: Các cột cần áp dụng
            fit: Học các tham số mới hay không
            
        Returns:
            DataFrame đã tiền xử lý
        """
        if preprocessor_names is None:
            # Áp dụng các bộ tiền xử lý mặc định
            preprocessor_names = [name for name, info in self.preprocessors.items() if info["apply_by_default"]]
        
        # Tạo bản sao để tránh thay đổi trực tiếp
        result_df = df.copy()
        
        for name in preprocessor_names:
            if name in self.preprocessors:
                preprocessor_info = self.preprocessors[name]
                
                # Xác định các cột cần áp dụng
                columns_to_apply = target_columns if target_columns is not None else preprocessor_info["target_columns"]
                
                # Nếu không có cột được chỉ định, áp dụng cho tất cả các cột số
                if not columns_to_apply:
                    columns_to_apply = result_df.select_dtypes(include=[np.number]).columns.tolist()
                
                try:
                    # Lấy hàm và tham số
                    preprocessor_func = preprocessor_info["function"]
                    params = preprocessor_info["params"].copy()
                    
                    # Áp dụng bộ tiền xử lý
                    if fit:
                        # Học và áp dụng
                        result_df, fitted_params = preprocessor_func(
                            result_df, columns_to_apply, fit=True, **params
                        )
                        self.preprocessors[name]["fitted_params"] = fitted_params
                    else:
                        # Áp dụng với các tham số đã học
                        fitted_params = self.preprocessors[name]["fitted_params"]
                        if fitted_params is not None:
                            result_df = preprocessor_func(
                                result_df, columns_to_apply, 
                                fit=False, fitted_params=fitted_params, **params
                            )
                        else:
                            self.logger.warning(f"Bộ tiền xử lý {name} chưa được fit")
                    
                    self.logger.debug(f"Đã áp dụng bộ tiền xử lý {name} cho {len(columns_to_apply)} cột")
                    
                except Exception as e:
                    self.logger.error(f"Lỗi khi áp dụng bộ tiền xử lý {name}: {e}")
        
        return result_df
    
    def apply_transformers(
        self,
        df: pd.DataFrame,
        transformer_names: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Áp dụng các biến đổi lên dữ liệu.
        
        Args:
            df: DataFrame cần xử lý
            transformer_names: Danh sách tên biến đổi cần áp dụng
            target_columns: Các cột cần áp dụng
            fit: Học các tham số mới hay không
            
        Returns:
            DataFrame đã biến đổi
        """
        if transformer_names is None:
            # Áp dụng tất cả các biến đổi
            transformer_names = list(self.transformers.keys())
        
        # Tạo bản sao để tránh thay đổi trực tiếp
        result_df = df.copy()
        
        for name in transformer_names:
            if name in self.transformers:
                transformer_info = self.transformers[name]
                
                # Xác định các cột cần áp dụng
                columns_to_apply = target_columns if target_columns is not None else transformer_info["target_columns"]
                
                # Nếu không có cột được chỉ định, áp dụng cho tất cả các cột số
                if not columns_to_apply:
                    columns_to_apply = result_df.select_dtypes(include=[np.number]).columns.tolist()
                
                try:
                    # Lấy hàm và tham số
                    transformer_func = transformer_info["function"]
                    params = transformer_info["params"].copy()
                    
                    # Áp dụng biến đổi
                    if fit and transformer_info["is_stateful"]:
                        # Học và áp dụng
                        result_df, fitted_state = transformer_func(
                            result_df, columns_to_apply, fit=True, **params
                        )
                        self.transformers[name]["fitted_state"] = fitted_state
                    else:
                        # Áp dụng với các tham số đã học
                        if transformer_info["is_stateful"]:
                            fitted_state = self.transformers[name]["fitted_state"]
                            if fitted_state is not None:
                                result_df = transformer_func(
                                    result_df, columns_to_apply, 
                                    fit=False, fitted_state=fitted_state, **params
                                )
                            else:
                                self.logger.warning(f"Biến đổi {name} chưa được fit")
                        else:
                            # Biến đổi không có trạng thái, luôn áp dụng trực tiếp
                            result_df = transformer_func(
                                result_df, columns_to_apply, **params
                            )
                    
                    self.logger.debug(f"Đã áp dụng biến đổi {name} cho {len(columns_to_apply)} cột")
                    
                except Exception as e:
                    self.logger.error(f"Lỗi khi áp dụng biến đổi {name}: {e}")
        
        return result_df
    
    def apply_feature_selection(
        self,
        df: pd.DataFrame,
        selector_name: str,
        target_column: str = 'close',  # Thêm giá trị mặc định là 'close'
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Áp dụng chọn lọc đặc trưng.
        
        Args:
            df: DataFrame cần xử lý
            selector_name: Tên bộ chọn lọc
            target_column: Cột mục tiêu (mặc định là 'close')
            fit: Học các tham số mới hay không
            
        Returns:
            DataFrame đã chọn lọc đặc trưng
        """
        if selector_name not in self.feature_selectors:
            self.logger.warning(f"Bộ chọn lọc {selector_name} không tồn tại")
            return df
        
        selector_info = self.feature_selectors[selector_name]
        
        try:
            # Lấy hàm và tham số
            selector_func = selector_info["function"]
            params = selector_info["params"].copy()
            
            # Thêm tham số target_column
            params["target_column"] = target_column
            
            # Áp dụng chọn lọc đặc trưng
            if fit:
                # Học và áp dụng
                selected_features, feature_scores = selector_func(df, **params)
                
                # Lưu danh sách tính năng đã chọn
                self.feature_selectors[selector_name]["selected_features"] = selected_features
                
                # Tạo DataFrame kết quả với các tính năng đã chọn
                if len(selected_features) > 0:
                    # Luôn giữ lại cột target_column
                    columns_to_keep = np.append(selected_features, target_column) if target_column not in selected_features else selected_features
                    result_df = df[columns_to_keep].copy()
                else:
                    # Nếu không có tính năng nào được chọn, trả về DataFrame ban đầu
                    result_df = df.copy()
            else:
                # Áp dụng với các đặc trưng đã chọn
                selected_features = self.feature_selectors[selector_name]["selected_features"]
                
                if selected_features is not None and len(selected_features) > 0:
                    # Luôn giữ lại cột target_column
                    columns_to_keep = np.append(selected_features, target_column) if target_column not in selected_features else selected_features
                    result_df = df[columns_to_keep].copy()
                else:
                    self.logger.warning(f"Bộ chọn lọc {selector_name} chưa được fit hoặc không chọn được tính năng nào")
                    result_df = df.copy()
            
            self.logger.info(f"Đã áp dụng chọn lọc đặc trưng {selector_name}, còn lại {result_df.shape[1]} đặc trưng")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Lỗi khi áp dụng chọn lọc đặc trưng {selector_name}: {str(e)}")
            return df
    
    def create_feature_pipeline(
        self,
        feature_names: Optional[List[str]] = None,
        preprocessor_names: Optional[List[str]] = None,
        transformer_names: Optional[List[str]] = None,
        feature_selector: Optional[str] = None,
        save_pipeline: bool = True,
        pipeline_name: Optional[str] = None
    ) -> None:
        """
        Tạo pipeline xử lý đặc trưng đầy đủ.
        
        Args:
            feature_names: Danh sách đặc trưng cần tạo
            preprocessor_names: Danh sách bộ tiền xử lý
            transformer_names: Danh sách biến đổi
            feature_selector: Tên bộ chọn lọc đặc trưng
            save_pipeline: Lưu pipeline hay không
            pipeline_name: Tên pipeline
        """
        pipeline_config = {
            "feature_names": feature_names,
            "preprocessor_names": preprocessor_names,
            "transformer_names": transformer_names,
            "feature_selector": feature_selector,
            "created_at": datetime.now().isoformat()
        }
        
        # Tạo tên pipeline nếu chưa có
        if pipeline_name is None:
            pipeline_name = f"feature_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.feature_config["pipelines"] = self.feature_config.get("pipelines", {})
        self.feature_config["pipelines"][pipeline_name] = pipeline_config
        
        self.logger.info(f"Đã tạo pipeline đặc trưng: {pipeline_name}")
        
        if save_pipeline:
            self.save_feature_config()
    
    def transform_data(
        self,
        df: pd.DataFrame,
        pipeline_name: Optional[str] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Áp dụng toàn bộ pipeline lên dữ liệu.
        
        Args:
            df: DataFrame cần xử lý
            pipeline_name: Tên pipeline cần áp dụng
            fit: Học các tham số mới hay không
            
        Returns:
            DataFrame đã xử lý
        """
        # Kiểm tra pipeline tồn tại
        if pipeline_name is not None:
            if "pipelines" not in self.feature_config or pipeline_name not in self.feature_config["pipelines"]:
                self.logger.warning(f"Pipeline {pipeline_name} không tồn tại")
                return df
            
            # Lấy cấu hình pipeline
            pipeline_config = self.feature_config["pipelines"][pipeline_name]
            feature_names = pipeline_config.get("feature_names")
            preprocessor_names = pipeline_config.get("preprocessor_names")
            transformer_names = pipeline_config.get("transformer_names")
            feature_selector = pipeline_config.get("feature_selector")
        else:
            # Sử dụng tất cả các đặc trưng đã đăng ký và bật
            feature_names = [name for name, info in self.registered_features.items() if info["is_enabled"]]
            # Sử dụng các bộ tiền xử lý mặc định
            preprocessor_names = [name for name, info in self.preprocessors.items() if info["apply_by_default"]]
            # Không áp dụng biến đổi
            transformer_names = None
            # Không áp dụng chọn lọc đặc trưng
            feature_selector = None
        
        # Tạo bản sao để tránh thay đổi trực tiếp
        result_df = df.copy()
        
        # Bước 1: Tính toán các đặc trưng
        if feature_names:
            result_df = self.calculate_features(result_df, feature_names, parallel=True)
        
        # Bước 2: Áp dụng các bộ tiền xử lý
        if preprocessor_names:
            result_df = self.apply_preprocessors(result_df, preprocessor_names, fit=fit)
        
        # Bước 3: Áp dụng các biến đổi
        if transformer_names:
            result_df = self.apply_transformers(result_df, transformer_names, fit=fit)
        
        # Bước 4: Áp dụng chọn lọc đặc trưng
        if feature_selector:
            result_df = self.apply_feature_selection(result_df, feature_selector, fit=fit)
        
        # Cập nhật trạng thái
        if fit:
            self.is_fitted = True
            
            # Lưu thông tin đặc trưng
            self.feature_info = {
                "num_features": result_df.shape[1],
                "feature_names": result_df.columns.tolist(),
                "numeric_features": result_df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_features": result_df.select_dtypes(exclude=[np.number]).columns.tolist(),
                "last_fit_time": datetime.now().isoformat()
            }
        
        return result_df
    
    def save_state(self, state_path: Optional[Union[str, Path]] = None) -> None:
        """
        Lưu trạng thái của FeatureGenerator.
        
        Args:
            state_path: Đường dẫn file trạng thái
        """
        if state_path is None:
            state_path = self.data_dir / "feature_generator_state.joblib"
        else:
            state_path = Path(state_path)
        
        # Tạo trạng thái để lưu
        state = {
            "preprocessors": {
                name: {"fitted_params": info["fitted_params"]}
                for name, info in self.preprocessors.items()
                if info["fitted_params"] is not None
            },
            "transformers": {
                name: {"fitted_state": info["fitted_state"]}
                for name, info in self.transformers.items()
                if info["is_stateful"] and info["fitted_state"] is not None
            },
            "feature_selectors": {
                name: {"selected_features": info["selected_features"]}
                for name, info in self.feature_selectors.items()
                if info["selected_features"] is not None
            },
            "feature_info": self.feature_info,
            "is_fitted": self.is_fitted,
            "version": "1.0.0"
        }
        
        try:
            # Lưu trạng thái
            joblib.dump(state, state_path)
            self.logger.info(f"Đã lưu trạng thái FeatureGenerator vào {state_path}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu trạng thái: {e}")
    
    def load_state(self, state_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Tải trạng thái của FeatureGenerator.
        
        Args:
            state_path: Đường dẫn file trạng thái
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        if state_path is None:
            state_path = self.data_dir / "feature_generator_state.joblib"
        else:
            state_path = Path(state_path)
        
        if not state_path.exists():
            self.logger.warning(f"File trạng thái {state_path} không tồn tại")
            return False
        
        try:
            # Tải trạng thái
            state = joblib.load(state_path)
            
            # Khôi phục trạng thái
            for name, preprocessor_state in state.get("preprocessors", {}).items():
                if name in self.preprocessors:
                    self.preprocessors[name]["fitted_params"] = preprocessor_state["fitted_params"]
            
            for name, transformer_state in state.get("transformers", {}).items():
                if name in self.transformers:
                    self.transformers[name]["fitted_state"] = transformer_state["fitted_state"]
            
            for name, selector_state in state.get("feature_selectors", {}).items():
                if name in self.feature_selectors:
                    self.feature_selectors[name]["selected_features"] = selector_state["selected_features"]
            
            self.feature_info = state.get("feature_info", {})
            self.is_fitted = state.get("is_fitted", False)
            
            self.logger.info(f"Đã tải trạng thái FeatureGenerator từ {state_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải trạng thái: {e}")
            return False
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về các đặc trưng đã đăng ký.
        
        Returns:
            Dict với thông tin đặc trưng
        """
        return {
            "registered_features": {
                name: {
                    "type": info["type"],
                    "is_enabled": info["is_enabled"],
                    "description": info["description"],
                    "dependencies": info["dependencies"]
                }
                for name, info in self.registered_features.items()
            },
            "preprocessors": list(self.preprocessors.keys()),
            "transformers": list(self.transformers.keys()),
            "feature_selectors": list(self.feature_selectors.keys()),
            "is_fitted": self.is_fitted,
            "feature_info": self.feature_info
        }

    def _register_all_technical_indicators(self) -> None:
        """
        Đăng ký toàn bộ technical indicators nếu có sẵn.
        """
        try:
            # Nhập các module cần thiết với xử lý lỗi rõ ràng
            try:
                from data_processors.feature_engineering.technical_indicators.trend_indicators import (
                    simple_moving_average as sma,
                    exponential_moving_average as ema,
                    bollinger_bands,
                    moving_average_convergence_divergence as macd
                )
            except ImportError as e:
                self.logger.warning(f"Lỗi import trend_indicators: {e}")
                sma = None
                ema = None
                bollinger_bands = None
                macd = None
            
            try:
                from data_processors.feature_engineering.technical_indicators.momentum_indicators import (
                    relative_strength_index as rsi
                )
            except ImportError as e:
                self.logger.warning(f"Lỗi import momentum_indicators: {e}")
                rsi = None
                if macd is None:  # Nếu macd chưa được import từ trend_indicators
                    try:
                        from data_processors.feature_engineering.technical_indicators.momentum_indicators import (
                            moving_average_convergence_divergence as macd
                        )
                    except ImportError:
                        macd = None
            
            try:
                from data_processors.feature_engineering.technical_indicators.volatility_indicators import (
                    average_true_range as atr
                )
            except ImportError as e:
                self.logger.warning(f"Lỗi import volatility_indicators: {e}")
                atr = None
                if bollinger_bands is None:  # Nếu bollinger_bands chưa được import từ trend_indicators
                    try:
                        from data_processors.feature_engineering.technical_indicators.volatility_indicators import (
                            bollinger_bands
                        )
                    except ImportError:
                        bollinger_bands = None
            
            try:
                from data_processors.feature_engineering.technical_indicators.volume_indicators import (
                    on_balance_volume as obv
                )
            except ImportError as e:
                self.logger.warning(f"Lỗi import volume_indicators: {e}")
                obv = None
            
            try:
                from data_processors.feature_engineering.technical_indicators.support_resistance import (
                    detect_support_resistance as support_resistance_zones
                )
            except ImportError as e:
                self.logger.warning(f"Lỗi import support_resistance: {e}")
                support_resistance_zones = None
            
            # Đăng ký các chỉ báo nếu chúng đã được import thành công
            if ema is not None:
                self.register_feature("trend_ema", ema, {"window": 14}, ["close"], "technical", "EMA 14")
            
            if sma is not None:
                self.register_feature("trend_sma", sma, {"window": 20}, ["close"], "technical", "SMA 20")
            
            if rsi is not None:
                self.register_feature("momentum_rsi", rsi, {"window": 14}, ["close"], "technical", "RSI 14")
            
            if macd is not None:
                self.register_feature("momentum_macd", macd, {}, ["close"], "technical", "MACD")
            
            if atr is not None:
                self.register_feature("volatility_atr", atr, {"window": 14}, ["high", "low", "close"], "technical", "ATR 14")
            
            if bollinger_bands is not None:
                self.register_feature("volatility_bbands", bollinger_bands, {"window": 20}, ["close"], "technical", "Bollinger Bands")
            
            if obv is not None:
                self.register_feature("volume_obv", obv, {}, ["close", "volume"], "technical", "OBV")
            
            if support_resistance_zones is not None:
                self.register_feature("support_resistance", support_resistance_zones, {}, ["high", "low"], "technical", "Hỗ trợ – Kháng cự")

            self.logger.info("Đã đăng ký tất cả technical indicators có sẵn")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi đăng ký indicators: {str(e)}")