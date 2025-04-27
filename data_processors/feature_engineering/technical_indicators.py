"""
Chỉ báo kỹ thuật cho dữ liệu thị trường.
File này cung cấp các lớp và phương thức để tính toán chỉ báo kỹ thuật
từ dữ liệu thị trường, bao gồm các chỉ báo cơ bản và nâng cao.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
import talib
from talib import abstract

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import setup_logger

class TechnicalIndicators:
    """
    Lớp chính để tính toán các chỉ báo kỹ thuật.
    """
    
    def __init__(
        self,
        use_talib: bool = True,
        custom_indicators: Optional[List[Callable]] = None
    ):
        """
        Khởi tạo bộ tính toán chỉ báo kỹ thuật.
        
        Args:
            use_talib: Sử dụng thư viện TA-Lib nếu khả dụng
            custom_indicators: Danh sách các hàm tính toán chỉ báo tùy chỉnh
        """
        self.logger = setup_logger("technical_indicators")
        
        self.use_talib = use_talib
        self.custom_indicators = custom_indicators or []
        
        # Kiểm tra xem TA-Lib có khả dụng không
        if self.use_talib:
            try:
                import talib
                self.talib_available = True
                self.logger.info("Đã khởi tạo TA-Lib thành công")
            except ImportError:
                self.talib_available = False
                self.logger.warning("Không thể import TA-Lib. Chuyển sang sử dụng cài đặt thay thế.")
                self.use_talib = False
        else:
            self.talib_available = False
        
        self.logger.info(f"Đã khởi tạo TechnicalIndicators với use_talib={self.use_talib}")
    
    def add_indicators(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Dict[str, Any]],
        ohlcv_columns: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Thêm các chỉ báo kỹ thuật vào DataFrame.
        
        Args:
            df: DataFrame cần thêm chỉ báo
            indicators: Dict chứa thông tin chỉ báo cần thêm
                      Định dạng: {
                          'indicator_name': {
                              'function': 'sma',
                              'params': {'timeperiod': 14},
                              'output_names': ['sma_14']
                          },
                          ...
                      }
            ohlcv_columns: Dict ánh xạ tên cột OHLCV ('open', 'high', 'low', 'close', 'volume')
                         với tên cột trong DataFrame. Mặc định là các tên cột chữ thường.
            
        Returns:
            DataFrame với các chỉ báo đã được thêm vào
        """
        if df.empty:
            self.logger.warning("DataFrame rỗng, không có gì để thêm chỉ báo")
            return df
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Chuẩn hóa tên cột OHLCV
        if ohlcv_columns is None:
            ohlcv_columns = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
        
        # Kiểm tra các cột cần thiết
        missing_columns = [col for col, mapped_col in ohlcv_columns.items() 
                          if mapped_col not in result_df.columns 
                          and col != 'volume']  # Volume có thể tùy chọn
        
        if missing_columns:
            self.logger.error(f"Thiếu các cột dữ liệu cần thiết: {missing_columns}")
            return df
        
        # Tính toán từng chỉ báo
        for indicator_name, indicator_config in indicators.items():
            try:
                self.logger.debug(f"Thêm chỉ báo: {indicator_name}")
                
                function_name = indicator_config.get('function')
                params = indicator_config.get('params', {})
                output_names = indicator_config.get('output_names', [indicator_name])
                
                # Xác định phương thức tính toán
                if self.use_talib and self.talib_available:
                    # Sử dụng TA-Lib
                    result_df = self._add_talib_indicator(
                        result_df, function_name, params, output_names, ohlcv_columns
                    )
                else:
                    # Sử dụng cài đặt thay thế
                    result_df = self._add_native_indicator(
                        result_df, function_name, params, output_names, ohlcv_columns
                    )
                
            except Exception as e:
                self.logger.error(f"Lỗi khi thêm chỉ báo {indicator_name}: {e}")
                continue
        
        # Thêm các chỉ báo tùy chỉnh
        if self.custom_indicators:
            for custom_indicator in self.custom_indicators:
                try:
                    result_df = custom_indicator(result_df, ohlcv_columns)
                except Exception as e:
                    self.logger.error(f"Lỗi khi thêm chỉ báo tùy chỉnh {custom_indicator.__name__}: {e}")
        
        return result_df
    
    def _add_talib_indicator(
        self,
        df: pd.DataFrame,
        function_name: str,
        params: Dict[str, Any],
        output_names: List[str],
        ohlcv_columns: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Thêm chỉ báo sử dụng TA-Lib.
        
        Args:
            df: DataFrame cần thêm chỉ báo
            function_name: Tên hàm TA-Lib
            params: Tham số cho hàm
            output_names: Tên cột đầu ra
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            DataFrame với chỉ báo đã được thêm vào
        """
        # Kiểm tra xem hàm có tồn tại trong TA-Lib không
        try:
            if hasattr(talib, function_name):
                func = getattr(talib, function_name)
            elif hasattr(abstract, function_name.upper()):
                func = getattr(abstract, function_name.upper())
            else:
                self.logger.warning(f"Hàm {function_name} không tồn tại trong TA-Lib. Chuyển sang cài đặt thay thế.")
                return self._add_native_indicator(df, function_name, params, output_names, ohlcv_columns)
            
            # Chuẩn bị dữ liệu đầu vào dựa trên abstract API hoặc gọi trực tiếp
            if hasattr(abstract, function_name.upper()):
                # Sử dụng abstract API
                inputs = {
                    'open': df[ohlcv_columns['open']].values,
                    'high': df[ohlcv_columns['high']].values,
                    'low': df[ohlcv_columns['low']].values,
                    'close': df[ohlcv_columns['close']].values
                }
                
                if 'volume' in ohlcv_columns and ohlcv_columns['volume'] in df.columns:
                    inputs['volume'] = df[ohlcv_columns['volume']].values
                
                # Lấy thông tin đầu ra
                info = abstract.Function(function_name.upper()).info
                output_count = len(info.get('output_names', [1]))
                
                # Gọi hàm
                result = func(inputs, **params)
                
                # Cập nhật kết quả vào DataFrame
                if output_count == 1 and not isinstance(result, tuple):
                    df[output_names[0]] = result
                else:
                    for i, output_value in enumerate(result):
                        if i < len(output_names):
                            df[output_names[i]] = output_value
                        else:
                            df[f"{function_name}_{i}"] = output_value
                
            else:
                # Gọi hàm trực tiếp
                # Xác định tham số đầu vào cần thiết
                import inspect
                sig = inspect.signature(func)
                param_names = [p for p in sig.parameters.keys()]
                
                # Chuẩn bị đầu vào
                inputs = {}
                if 'open' in param_names:
                    inputs['open'] = df[ohlcv_columns['open']].values
                if 'high' in param_names:
                    inputs['high'] = df[ohlcv_columns['high']].values
                if 'low' in param_names:
                    inputs['low'] = df[ohlcv_columns['low']].values
                if 'close' in param_names:
                    inputs['close'] = df[ohlcv_columns['close']].values
                if 'volume' in param_names and 'volume' in ohlcv_columns and ohlcv_columns['volume'] in df.columns:
                    inputs['volume'] = df[ohlcv_columns['volume']].values
                
                # Gọi hàm với các tham số phù hợp
                if inputs:
                    result = func(*list(inputs.values()), **params)
                    
                    # Xử lý kết quả đầu ra
                    if isinstance(result, tuple):
                        for i, output_value in enumerate(result):
                            if i < len(output_names):
                                df[output_names[i]] = output_value
                            else:
                                df[f"{function_name}_{i}"] = output_value
                    else:
                        df[output_names[0]] = result
                else:
                    # Chỉ truyền close nếu không có tham số cụ thể
                    result = func(df[ohlcv_columns['close']].values, **params)
                    
                    # Xử lý kết quả đầu ra
                    if isinstance(result, tuple):
                        for i, output_value in enumerate(result):
                            if i < len(output_names):
                                df[output_names[i]] = output_value
                            else:
                                df[f"{function_name}_{i}"] = output_value
                    else:
                        df[output_names[0]] = result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi sử dụng TA-Lib cho {function_name}: {e}")
            # Thử phương pháp cài đặt thay thế
            df = self._add_native_indicator(df, function_name, params, output_names, ohlcv_columns)
        
        return df
    
    def _add_native_indicator(
        self,
        df: pd.DataFrame,
        function_name: str,
        params: Dict[str, Any],
        output_names: List[str],
        ohlcv_columns: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Thêm chỉ báo sử dụng cài đặt Python thuần.
        
        Args:
            df: DataFrame cần thêm chỉ báo
            function_name: Tên hàm chỉ báo
            params: Tham số cho hàm
            output_names: Tên cột đầu ra
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            DataFrame với chỉ báo đã được thêm vào
        """
        # Ánh xạ tên hàm đến cài đặt nội bộ
        function_mapping = {
            'sma': self._sma,
            'ema': self._ema,
            'wma': self._wma,
            'rsi': self._rsi,
            'macd': self._macd,
            'bbands': self._bbands,
            'atr': self._atr,
            'stoch': self._stoch,
            'adx': self._adx,
            'cci': self._cci,
            'roc': self._roc,
            'willr': self._willr,
            'mom': self._mom,
            'obv': self._obv,
            'ad': self._ad,
            'aroon': self._aroon,
            'dmi': self._dmi,
            'kama': self._kama,
            'natr': self._natr,
            'ppo': self._ppo,
            'sar': self._sar,
            'stochf': self._stochf,
            'stochrsi': self._stochrsi,
            'trange': self._trange,
            'mfi': self._mfi,
            'ichimoku': self._ichimoku,
            'vwap': self._vwap,
            'supertrend': self._supertrend,
            'rvi': self._rvi,
            'zigzag': self._zigzag,
            'pivot_points': self._pivot_points,
            'heikin_ashi': self._heikin_ashi
        }
        
        if function_name.lower() in function_mapping:
            try:
                # Gọi hàm tương ứng
                impl_func = function_mapping[function_name.lower()]
                result = impl_func(df, params, ohlcv_columns)
                
                # Cập nhật kết quả vào DataFrame
                if isinstance(result, tuple):
                    for i, output_value in enumerate(result):
                        if i < len(output_names):
                            df[output_names[i]] = output_value
                        else:
                            df[f"{function_name}_{i}"] = output_value
                else:
                    df[output_names[0]] = result
                    
            except Exception as e:
                self.logger.error(f"Lỗi khi tính toán {function_name}: {e}")
        else:
            self.logger.warning(f"Không có cài đặt thay thế cho chỉ báo {function_name}")
        
        return df
    
    # Cài đặt các chỉ báo kỹ thuật
    
    def _sma(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Simple Moving Average (SMA).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 14)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 14)
        close = df[ohlcv_columns['close']].values
        
        # Tính SMA
        sma = np.zeros_like(close)
        sma[:] = np.nan
        
        for i in range(timeperiod - 1, len(close)):
            sma[i] = np.mean(close[i - timeperiod + 1:i + 1])
        
        return sma
    
    def _ema(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Exponential Moving Average (EMA).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 14)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 14)
        close = df[ohlcv_columns['close']].values
        
        # Tính EMA
        ema = np.zeros_like(close)
        ema[:] = np.nan
        
        # Phần sma đầu tiên
        if len(close) >= timeperiod:
            ema[timeperiod - 1] = np.mean(close[:timeperiod])
        
        # Hệ số alpha
        alpha = 2.0 / (timeperiod + 1)
        
        # Tính EMA cho các điểm còn lại
        for i in range(timeperiod, len(close)):
            ema[i] = alpha * close[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def _wma(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Weighted Moving Average (WMA).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 14)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 14)
        close = df[ohlcv_columns['close']].values
        
        # Tính WMA
        wma = np.zeros_like(close)
        wma[:] = np.nan
        
        # Trọng số
        weights = np.arange(1, timeperiod + 1)
        sum_weights = np.sum(weights)
        
        for i in range(timeperiod - 1, len(close)):
            wma[i] = np.sum(close[i - timeperiod + 1:i + 1] * weights) / sum_weights
        
        return wma
    
    def _rsi(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Relative Strength Index (RSI).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 14)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 14)
        close = df[ohlcv_columns['close']].values
        
        # Tính RSI
        rsi = np.zeros_like(close)
        rsi[:] = np.nan
        
        # Tính price change
        diff = np.diff(close)
        diff = np.insert(diff, 0, 0)
        
        # Tách thành gain và loss
        gain = np.where(diff > 0, diff, 0)
        loss = np.where(diff < 0, -diff, 0)
        
        # Tính avg gain và avg loss
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Tính first avg
        if len(close) >= timeperiod + 1:
            avg_gain[timeperiod] = np.mean(gain[1:timeperiod + 1])
            avg_loss[timeperiod] = np.mean(loss[1:timeperiod + 1])
        
        # Tính cho các giá trị tiếp theo
        for i in range(timeperiod + 1, len(close)):
            avg_gain[i] = (avg_gain[i - 1] * (timeperiod - 1) + gain[i]) / timeperiod
            avg_loss[i] = (avg_loss[i - 1] * (timeperiod - 1) + loss[i]) / timeperiod
        
        # Tính RS và RSI
        rs = np.zeros_like(close)
        for i in range(timeperiod, len(close)):
            if avg_loss[i] == 0:
                rs[i] = 100.0
            else:
                rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        
        return rsi
    
    def _macd(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tính Moving Average Convergence Divergence (MACD).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - fastperiod: Số kỳ của EMA nhanh (mặc định: 12)
                - slowperiod: Số kỳ của EMA chậm (mặc định: 26)
                - signalperiod: Số kỳ của đường tín hiệu (mặc định: 9)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Tuple (macd, signal, histogram)
        """
        fastperiod = params.get('fastperiod', 12)
        slowperiod = params.get('slowperiod', 26)
        signalperiod = params.get('signalperiod', 9)
        
        # Tính EMA nhanh và chậm
        ema_fast = self._ema(df, {'timeperiod': fastperiod}, ohlcv_columns)
        ema_slow = self._ema(df, {'timeperiod': slowperiod}, ohlcv_columns)
        
        # Tính MACD
        macd = ema_fast - ema_slow
        
        # Tính đường tín hiệu (EMA của MACD)
        temp_df = pd.DataFrame({'close': macd})
        signal = self._ema(temp_df, {'timeperiod': signalperiod}, {'close': 'close'})
        
        # Tính histogram
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def _bbands(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tính Bollinger Bands.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 20)
                - nbdevup: Số độ lệch chuẩn cho băng trên (mặc định: 2)
                - nbdevdn: Số độ lệch chuẩn cho băng dưới (mặc định: 2)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Tuple (upper, middle, lower)
        """
        timeperiod = params.get('timeperiod', 20)
        nbdevup = params.get('nbdevup', 2)
        nbdevdn = params.get('nbdevdn', 2)
        
        close = df[ohlcv_columns['close']].values
        
        # Tính SMA (middle band)
        middle = self._sma(df, {'timeperiod': timeperiod}, ohlcv_columns)
        
        # Tính độ lệch chuẩn
        stddev = np.zeros_like(close)
        stddev[:] = np.nan
        
        for i in range(timeperiod - 1, len(close)):
            stddev[i] = np.std(close[i - timeperiod + 1:i + 1], ddof=1)
        
        # Tính upper và lower bands
        upper = middle + nbdevup * stddev
        lower = middle - nbdevdn * stddev
        
        return upper, middle, lower
    
    def _atr(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Average True Range (ATR).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 14)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 14)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Tính True Range
        tr = np.zeros_like(close)
        tr[:] = np.nan
        
        # TR cho phần tử đầu tiên
        if len(close) > 0:
            tr[0] = high[0] - low[0]
        
        # TR cho các phần tử tiếp theo
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        
        # Tính ATR
        atr = np.zeros_like(close)
        atr[:] = np.nan
        
        # ATR đầu tiên
        if len(close) >= timeperiod:
            atr[timeperiod - 1] = np.mean(tr[:timeperiod])
        
        # ATR cho các phần tử tiếp theo
        for i in range(timeperiod, len(close)):
            atr[i] = (atr[i - 1] * (timeperiod - 1) + tr[i]) / timeperiod
        
        return atr
    
    def _stoch(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính Stochastic Oscillator.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - fastk_period: Số kỳ K nhanh (mặc định: 5)
                - slowk_period: Số kỳ K chậm (mặc định: 3)
                - slowd_period: Số kỳ D chậm (mặc định: 3)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Tuple (slowk, slowd)
        """
        fastk_period = params.get('fastk_period', 5)
        slowk_period = params.get('slowk_period', 3)
        slowd_period = params.get('slowd_period', 3)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Tính %K nhanh
        fastk = np.zeros_like(close)
        fastk[:] = np.nan
        
        for i in range(fastk_period - 1, len(close)):
            highest_high = np.max(high[i - fastk_period + 1:i + 1])
            lowest_low = np.min(low[i - fastk_period + 1:i + 1])
            
            if highest_high != lowest_low:
                fastk[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
            else:
                fastk[i] = 50  # Giá trị mặc định khi highest = lowest
        
        # Tính %K chậm (SMA của fastk)
        temp_df = pd.DataFrame({'close': fastk})
        slowk = self._sma(temp_df, {'timeperiod': slowk_period}, {'close': 'close'})
        
        # Tính %D (SMA của slowk)
        temp_df = pd.DataFrame({'close': slowk})
        slowd = self._sma(temp_df, {'timeperiod': slowd_period}, {'close': 'close'})
        
        return slowk, slowd
    
    def _adx(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Average Directional Index (ADX).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 14)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 14)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Tính +DM và -DM
        plus_dm = np.zeros_like(close)
        minus_dm = np.zeros_like(close)
        
        for i in range(1, len(close)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
            
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        # Tính True Range
        tr = np.zeros_like(close)
        tr[0] = np.nan
        
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        
        # Tính các chỉ số smoothed
        smoothed_plus_dm = np.zeros_like(close)
        smoothed_minus_dm = np.zeros_like(close)
        smoothed_tr = np.zeros_like(close)
        
        if len(close) >= timeperiod + 1:
            # Giá trị đầu tiên
            smoothed_plus_dm[timeperiod] = np.sum(plus_dm[1:timeperiod + 1])
            smoothed_minus_dm[timeperiod] = np.sum(minus_dm[1:timeperiod + 1])
            smoothed_tr[timeperiod] = np.sum(tr[1:timeperiod + 1])
            
            # Các giá trị tiếp theo
            for i in range(timeperiod + 1, len(close)):
                smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / timeperiod) + plus_dm[i]
                smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - (smoothed_minus_dm[i - 1] / timeperiod) + minus_dm[i]
                smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / timeperiod) + tr[i]
        
        # Tính +DI và -DI
        plus_di = np.zeros_like(close)
        minus_di = np.zeros_like(close)
        
        for i in range(timeperiod, len(close)):
            if smoothed_tr[i] > 0:
                plus_di[i] = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
                minus_di[i] = 100 * smoothed_minus_dm[i] / smoothed_tr[i]
            else:
                plus_di[i] = 0
                minus_di[i] = 0
        
        # Tính DX
        dx = np.zeros_like(close)
        dx[:] = np.nan
        
        for i in range(timeperiod, len(close)):
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            else:
                dx[i] = 0
        
        # Tính ADX
        adx = np.zeros_like(close)
        adx[:] = np.nan
        
        if len(close) >= 2 * timeperiod:
            # Giá trị đầu tiên
            adx[2 * timeperiod - 1] = np.mean(dx[timeperiod:2 * timeperiod])
            
            # Các giá trị tiếp theo
            for i in range(2 * timeperiod, len(close)):
                adx[i] = (adx[i - 1] * (timeperiod - 1) + dx[i]) / timeperiod
        
        return adx
    
    def _cci(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Commodity Channel Index (CCI).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 14)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 14)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Tính Typical Price
        tp = (high + low + close) / 3
        
        # Tính SMA của Typical Price
        tp_sma = np.zeros_like(tp)
        tp_sma[:] = np.nan
        
        for i in range(timeperiod - 1, len(tp)):
            tp_sma[i] = np.mean(tp[i - timeperiod + 1:i + 1])
        
        # Tính Mean Deviation
        mad = np.zeros_like(tp)
        mad[:] = np.nan
        
        for i in range(timeperiod - 1, len(tp)):
            mad[i] = np.mean(np.abs(tp[i - timeperiod + 1:i + 1] - tp_sma[i]))
        
        # Tính CCI
        cci = np.zeros_like(tp)
        cci[:] = np.nan
        
        for i in range(timeperiod - 1, len(tp)):
            if mad[i] > 0:
                cci[i] = (tp[i] - tp_sma[i]) / (0.015 * mad[i])
            else:
                cci[i] = 0  # Tránh chia cho 0
        
        return cci
    
    def _roc(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Rate of Change (ROC).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 10)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 10)
        close = df[ohlcv_columns['close']].values
        
        # Tính ROC
        roc = np.zeros_like(close)
        roc[:] = np.nan
        
        for i in range(timeperiod, len(close)):
            if close[i - timeperiod] != 0:
                roc[i] = ((close[i] - close[i - timeperiod]) / close[i - timeperiod]) * 100
            else:
                roc[i] = 0  # Tránh chia cho 0
        
        return roc
    
    def _willr(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Williams' %R.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 14)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 14)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Tính Williams %R
        willr = np.zeros_like(close)
        willr[:] = np.nan
        
        for i in range(timeperiod - 1, len(close)):
            highest_high = np.max(high[i - timeperiod + 1:i + 1])
            lowest_low = np.min(low[i - timeperiod + 1:i + 1])
            
            if highest_high != lowest_low:
                willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
            else:
                willr[i] = -50  # Giá trị mặc định khi highest = lowest
        
        return willr
    
    def _mom(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Momentum.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 10)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 10)
        close = df[ohlcv_columns['close']].values
        
        # Tính Momentum
        mom = np.zeros_like(close)
        mom[:] = np.nan
        
        for i in range(timeperiod, len(close)):
            mom[i] = close[i] - close[i - timeperiod]
        
        return mom
    
    def _obv(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính On Balance Volume (OBV).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số (không có tham số đặc biệt)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        close = df[ohlcv_columns['close']].values
        
        # Kiểm tra cột volume
        if 'volume' not in ohlcv_columns or ohlcv_columns['volume'] not in df.columns:
            self.logger.warning("Không tìm thấy cột volume cho chỉ báo OBV")
            return np.zeros_like(close)
        
        volume = df[ohlcv_columns['volume']].values
        
        # Tính OBV
        obv = np.zeros_like(close)
        
        # Giá trị đầu tiên
        obv[0] = volume[0]
        
        # Các giá trị tiếp theo
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        
        return obv
    
    def _ad(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Accumulation/Distribution Line (A/D).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số (không có tham số đặc biệt)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Kiểm tra cột volume
        if 'volume' not in ohlcv_columns or ohlcv_columns['volume'] not in df.columns:
            self.logger.warning("Không tìm thấy cột volume cho chỉ báo A/D")
            return np.zeros_like(close)
        
        volume = df[ohlcv_columns['volume']].values
        
        # Tính Money Flow Multiplier
        mfm = np.zeros_like(close)
        
        for i in range(len(close)):
            if high[i] != low[i]:
                mfm[i] = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            else:
                mfm[i] = 0
        
        # Tính Money Flow Volume
        mfv = mfm * volume
        
        # Tính A/D Line
        ad = np.zeros_like(close)
        ad[0] = mfv[0]
        
        for i in range(1, len(close)):
            ad[i] = ad[i - 1] + mfv[i]
        
        return ad
    
    def _aroon(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính Aroon Indicator.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 14)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Tuple (aroon_up, aroon_down)
        """
        timeperiod = params.get('timeperiod', 14)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        
        # Tính Aroon Up và Aroon Down
        aroon_up = np.zeros_like(high)
        aroon_down = np.zeros_like(high)
        aroon_up[:] = np.nan
        aroon_down[:] = np.nan
        
        for i in range(timeperiod, len(high)):
            # Tìm vị trí giá cao nhất và thấp nhất trong khoảng timeperiod
            period_high = high[i - timeperiod + 1:i + 1]
            period_low = low[i - timeperiod + 1:i + 1]
            
            high_idx = np.argmax(period_high)
            low_idx = np.argmin(period_low)
            
            # Tính Aroon
            aroon_up[i] = ((timeperiod - high_idx) / timeperiod) * 100
            aroon_down[i] = ((timeperiod - low_idx) / timeperiod) * 100
        
        return aroon_up, aroon_down
    
    def _dmi(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính Directional Movement Index (DMI: +DI, -DI).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 14)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Tuple (plus_di, minus_di)
        """
        timeperiod = params.get('timeperiod', 14)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Tính +DM và -DM
        plus_dm = np.zeros_like(close)
        minus_dm = np.zeros_like(close)
        
        for i in range(1, len(close)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
            
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        # Tính True Range
        tr = np.zeros_like(close)
        tr[0] = np.nan
        
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        
        # Tính các chỉ số smoothed
        smoothed_plus_dm = np.zeros_like(close)
        smoothed_minus_dm = np.zeros_like(close)
        smoothed_tr = np.zeros_like(close)
        
        if len(close) >= timeperiod + 1:
            # Giá trị đầu tiên
            smoothed_plus_dm[timeperiod] = np.sum(plus_dm[1:timeperiod + 1])
            smoothed_minus_dm[timeperiod] = np.sum(minus_dm[1:timeperiod + 1])
            smoothed_tr[timeperiod] = np.sum(tr[1:timeperiod + 1])
            
            # Các giá trị tiếp theo
            for i in range(timeperiod + 1, len(close)):
                smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / timeperiod) + plus_dm[i]
                smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - (smoothed_minus_dm[i - 1] / timeperiod) + minus_dm[i]
                smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / timeperiod) + tr[i]
        
        # Tính +DI và -DI
        plus_di = np.zeros_like(close)
        minus_di = np.zeros_like(close)
        plus_di[:] = np.nan
        minus_di[:] = np.nan
        
        for i in range(timeperiod, len(close)):
            if smoothed_tr[i] > 0:
                plus_di[i] = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
                minus_di[i] = 100 * smoothed_minus_dm[i] / smoothed_tr[i]
        
        return plus_di, minus_di
    
    def _kama(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Kaufman's Adaptive Moving Average (KAMA).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 10)
                - fast_period: Giai đoạn nhanh (mặc định: 2)
                - slow_period: Giai đoạn chậm (mặc định: 30)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 10)
        fast_period = params.get('fast_period', 2)
        slow_period = params.get('slow_period', 30)
        
        close = df[ohlcv_columns['close']].values
        
        # Tính Direction
        direction = np.abs(close[timeperiod:] - close[:-timeperiod])
        direction = np.insert(direction, 0, [0] * timeperiod)
        
        # Tính Volatility
        volatility = np.zeros_like(close)
        
        for i in range(timeperiod, len(close)):
            volatility[i] = np.sum(np.abs(close[i] - close[i-1]) for i in range(i - timeperiod + 1, i + 1))
        
        # Tính Efficiency Ratio
        er = np.zeros_like(close)
        
        for i in range(timeperiod, len(close)):
            if volatility[i] > 0:
                er[i] = direction[i] / volatility[i]
            else:
                er[i] = 0
        
        # Tính Smoothing Constant
        fast_sc = 2 / (fast_period + 1)
        slow_sc = 2 / (slow_period + 1)
        
        sc = np.zeros_like(close)
        sc[:] = np.nan
        
        for i in range(timeperiod, len(close)):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Tính KAMA
        kama = np.zeros_like(close)
        kama[:] = np.nan
        
        # Giá trị đầu tiên
        if len(close) >= timeperiod:
            kama[timeperiod - 1] = close[timeperiod - 1]
        
        # Các giá trị tiếp theo
        for i in range(timeperiod, len(close)):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        
        return kama
    
    def _natr(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Normalized Average True Range (NATR).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 14)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 14)
        
        close = df[ohlcv_columns['close']].values
        
        # Tính ATR
        atr = self._atr(df, {'timeperiod': timeperiod}, ohlcv_columns)
        
        # Tính NATR
        natr = np.zeros_like(close)
        natr[:] = np.nan
        
        for i in range(len(close)):
            if close[i] > 0:
                natr[i] = 100 * atr[i] / close[i]
        
        return natr
    
    def _ppo(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tính Percentage Price Oscillator (PPO).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - fastperiod: Số kỳ của EMA nhanh (mặc định: 12)
                - slowperiod: Số kỳ của EMA chậm (mặc định: 26)
                - signalperiod: Số kỳ của đường tín hiệu (mặc định: 9)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Tuple (ppo, signal, histogram)
        """
        fastperiod = params.get('fastperiod', 12)
        slowperiod = params.get('slowperiod', 26)
        signalperiod = params.get('signalperiod', 9)
        
        # Tính EMA nhanh và chậm
        ema_fast = self._ema(df, {'timeperiod': fastperiod}, ohlcv_columns)
        ema_slow = self._ema(df, {'timeperiod': slowperiod}, ohlcv_columns)
        
        # Tính PPO
        ppo = np.zeros_like(ema_fast)
        ppo[:] = np.nan
        
        for i in range(len(ema_fast)):
            if ema_slow[i] > 0:
                ppo[i] = 100 * (ema_fast[i] - ema_slow[i]) / ema_slow[i]
        
        # Tính đường tín hiệu (EMA của PPO)
        temp_df = pd.DataFrame({'close': ppo})
        signal = self._ema(temp_df, {'timeperiod': signalperiod}, {'close': 'close'})
        
        # Tính histogram
        histogram = ppo - signal
        
        return ppo, signal, histogram
    
    def _sar(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Parabolic SAR.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - acceleration: Hệ số tăng tốc (mặc định: 0.02)
                - maximum: Giá trị tối đa của hệ số tăng tốc (mặc định: 0.2)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        acceleration = params.get('acceleration', 0.02)
        maximum = params.get('maximum', 0.2)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        
        # Tính SAR
        sar = np.zeros_like(high)
        sar[:] = np.nan
        
        # Cần ít nhất 2 điểm dữ liệu
        if len(high) < 2:
            return sar
        
        # Xác định xu hướng ban đầu
        trend = 1 if high[1] > high[0] else -1
        
        # Đặt giá trị SAR đầu tiên
        if trend > 0:
            sar[1] = min(low[0], low[1])  # Up trend: SAR ở dưới
            ep = max(high[0], high[1])    # Extreme point
        else:
            sar[1] = max(high[0], high[1])  # Down trend: SAR ở trên
            ep = min(low[0], low[1])        # Extreme point
        
        # Hệ số AF ban đầu
        af = acceleration
        
        # Tính SAR cho các điểm tiếp theo
        for i in range(2, len(high)):
            # SAR cho điểm hiện tại
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            
            # Kiểm tra đảo chiều
            if trend > 0:  # Trong xu hướng tăng
                if low[i] < sar[i]:  # Đảo chiều: từ tăng sang giảm
                    trend = -1
                    sar[i] = ep  # SAR mới bằng EP trước đó (cao nhất của xu hướng tăng)
                    ep = low[i]  # EP mới là giá thấp hiện tại
                    af = acceleration  # Reset AF
                else:  # Tiếp tục xu hướng tăng
                    if high[i] > ep:  # Tạo EP mới
                        ep = high[i]
                        af = min(af + acceleration, maximum)  # Tăng AF, nhưng không quá maximum
                    # Đảm bảo SAR không vượt quá min(low) của 2 nến trước
                    sar[i] = min(sar[i], min(low[i - 1], low[i - 2]))
            
            else:  # Trong xu hướng giảm
                if high[i] > sar[i]:  # Đảo chiều: từ giảm sang tăng
                    trend = 1
                    sar[i] = ep  # SAR mới bằng EP trước đó (thấp nhất của xu hướng giảm)
                    ep = high[i]  # EP mới là giá cao hiện tại
                    af = acceleration  # Reset AF
                else:  # Tiếp tục xu hướng giảm
                    if low[i] < ep:  # Tạo EP mới
                        ep = low[i]
                        af = min(af + acceleration, maximum)  # Tăng AF, nhưng không quá maximum
                    # Đảm bảo SAR không thấp hơn max(high) của 2 nến trước
                    sar[i] = max(sar[i], max(high[i - 1], high[i - 2]))
        
        return sar
    
    def _stochf(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính Fast Stochastic Oscillator.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - fastk_period: Số kỳ K nhanh (mặc định: 5)
                - fastd_period: Số kỳ D nhanh (mặc định: 3)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Tuple (fastk, fastd)
        """
        fastk_period = params.get('fastk_period', 5)
        fastd_period = params.get('fastd_period', 3)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Tính %K nhanh
        fastk = np.zeros_like(close)
        fastk[:] = np.nan
        
        for i in range(fastk_period - 1, len(close)):
            highest_high = np.max(high[i - fastk_period + 1:i + 1])
            lowest_low = np.min(low[i - fastk_period + 1:i + 1])
            
            if highest_high != lowest_low:
                fastk[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)
            else:
                fastk[i] = 50  # Giá trị mặc định khi highest = lowest
        
        # Tính %D nhanh (SMA của fastk)
        temp_df = pd.DataFrame({'close': fastk})
        fastd = self._sma(temp_df, {'timeperiod': fastd_period}, {'close': 'close'})
        
        return fastk, fastd
    
    def _stochrsi(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính Stochastic RSI.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ cho RSI (mặc định: 14)
                - fastk_period: Số kỳ K nhanh (mặc định: 5)
                - fastd_period: Số kỳ D nhanh (mặc định: 3)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Tuple (fastk, fastd)
        """
        timeperiod = params.get('timeperiod', 14)
        fastk_period = params.get('fastk_period', 5)
        fastd_period = params.get('fastd_period', 3)
        
        # Tính RSI
        rsi = self._rsi(df, {'timeperiod': timeperiod}, ohlcv_columns)
        
        # Tính Stochastic của RSI
        fastk = np.zeros_like(rsi)
        fastk[:] = np.nan
        
        for i in range(timeperiod + fastk_period - 2, len(rsi)):
            window_rsi = rsi[i - fastk_period + 1:i + 1]
            window_rsi = window_rsi[~np.isnan(window_rsi)]  # Bỏ qua các giá trị NaN
            
            if len(window_rsi) > 0:
                min_rsi = np.min(window_rsi)
                max_rsi = np.max(window_rsi)
                
                if max_rsi != min_rsi:
                    fastk[i] = 100 * (rsi[i] - min_rsi) / (max_rsi - min_rsi)
                else:
                    fastk[i] = 50  # Giá trị mặc định khi min = max
        
        # Tính fastd (SMA của fastk)
        temp_df = pd.DataFrame({'close': fastk})
        fastd = self._sma(temp_df, {'timeperiod': fastd_period}, {'close': 'close'})
        
        return fastk, fastd
    
    def _trange(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính True Range.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số (không có tham số đặc biệt)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Tính True Range
        tr = np.zeros_like(close)
        tr[:] = np.nan
        
        # TR cho phần tử đầu tiên
        if len(close) > 0:
            tr[0] = high[0] - low[0]
        
        # TR cho các phần tử tiếp theo
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        
        return tr
    
    def _mfi(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Money Flow Index (MFI).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 14)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 14)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Kiểm tra cột volume
        if 'volume' not in ohlcv_columns or ohlcv_columns['volume'] not in df.columns:
            self.logger.warning("Không tìm thấy cột volume cho chỉ báo MFI")
            return np.zeros_like(close)
        
        volume = df[ohlcv_columns['volume']].values
        
        # Tính Typical Price
        tp = (high + low + close) / 3
        
        # Tính Money Flow
        mf = tp * volume
        
        # Tính Positive và Negative Money Flow
        pos_mf = np.zeros_like(close)
        neg_mf = np.zeros_like(close)
        
        for i in range(1, len(close)):
            if tp[i] > tp[i - 1]:
                pos_mf[i] = mf[i]
                neg_mf[i] = 0
            elif tp[i] < tp[i - 1]:
                pos_mf[i] = 0
                neg_mf[i] = mf[i]
            else:
                pos_mf[i] = 0
                neg_mf[i] = 0
        
        # Tính Money Flow Ratio và MFI
        mfi = np.zeros_like(close)
        mfi[:] = np.nan
        
        for i in range(timeperiod, len(close)):
            pos_sum = np.sum(pos_mf[i - timeperiod + 1:i + 1])
            neg_sum = np.sum(neg_mf[i - timeperiod + 1:i + 1])
            
            if neg_sum > 0:
                mfr = pos_sum / neg_sum
                mfi[i] = 100 - (100 / (1 + mfr))
            else:
                mfi[i] = 100  # Nếu neg_sum = 0, MFI = 100
        
        return mfi
    
    def _ichimoku(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Tính Ichimoku Cloud.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - tenkan_period: Kỳ hạn cho Tenkan-sen (mặc định: 9)
                - kijun_period: Kỳ hạn cho Kijun-sen (mặc định: 26)
                - senkou_b_period: Kỳ hạn cho Senkou Span B (mặc định: 52)
                - displacement: Độ trễ cho Senkou Span A và B (mặc định: 26)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Tuple (tenkan, kijun, senkou_a, senkou_b, chikou)
        """
        tenkan_period = params.get('tenkan_period', 9)
        kijun_period = params.get('kijun_period', 26)
        senkou_b_period = params.get('senkou_b_period', 52)
        displacement = params.get('displacement', 26)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Tính Tenkan-sen (Đường chuyển)
        tenkan = np.zeros_like(close)
        tenkan[:] = np.nan
        
        for i in range(tenkan_period - 1, len(close)):
            highest_high = np.max(high[i - tenkan_period + 1:i + 1])
            lowest_low = np.min(low[i - tenkan_period + 1:i + 1])
            tenkan[i] = (highest_high + lowest_low) / 2
        
        # Tính Kijun-sen (Đường cơ sở)
        kijun = np.zeros_like(close)
        kijun[:] = np.nan
        
        for i in range(kijun_period - 1, len(close)):
            highest_high = np.max(high[i - kijun_period + 1:i + 1])
            lowest_low = np.min(low[i - kijun_period + 1:i + 1])
            kijun[i] = (highest_high + lowest_low) / 2
        
        # Tính Senkou Span A (Mây A)
        senkou_a = np.zeros_like(close)
        senkou_a[:] = np.nan
        
        for i in range(kijun_period - 1, len(close)):
            if i + displacement < len(close):
                senkou_a[i + displacement] = (tenkan[i] + kijun[i]) / 2
        
        # Tính Senkou Span B (Mây B)
        senkou_b = np.zeros_like(close)
        senkou_b[:] = np.nan
        
        for i in range(senkou_b_period - 1, len(close)):
            if i + displacement < len(close):
                highest_high = np.max(high[i - senkou_b_period + 1:i + 1])
                lowest_low = np.min(low[i - senkou_b_period + 1:i + 1])
                senkou_b[i + displacement] = (highest_high + lowest_low) / 2
        
        # Tính Chikou Span (Đường trễ)
        chikou = np.zeros_like(close)
        chikou[:] = np.nan
        
        for i in range(displacement, len(close)):
            chikou[i - displacement] = close[i]
        
        return tenkan, kijun, senkou_a, senkou_b, chikou
    
    def _vwap(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Volume Weighted Average Price (VWAP).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: dữ liệu đầy đủ)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', None)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Kiểm tra cột volume
        if 'volume' not in ohlcv_columns or ohlcv_columns['volume'] not in df.columns:
            self.logger.warning("Không tìm thấy cột volume cho chỉ báo VWAP")
            return np.zeros_like(close)
        
        volume = df[ohlcv_columns['volume']].values
        
        # Tính Typical Price
        tp = (high + low + close) / 3
        
        # Tính VWAP
        vwap = np.zeros_like(close)
        vwap[:] = np.nan
        
        if timeperiod is None:
            # VWAP trên toàn bộ dữ liệu
            cum_tp_vol = np.zeros_like(close)
            cum_vol = np.zeros_like(close)
            
            cum_tp_vol[0] = tp[0] * volume[0]
            cum_vol[0] = volume[0]
            
            for i in range(1, len(close)):
                cum_tp_vol[i] = cum_tp_vol[i - 1] + tp[i] * volume[i]
                cum_vol[i] = cum_vol[i - 1] + volume[i]
                
                if cum_vol[i] > 0:
                    vwap[i] = cum_tp_vol[i] / cum_vol[i]
        else:
            # VWAP với timeperiod xác định
            for i in range(timeperiod - 1, len(close)):
                period_tp = tp[i - timeperiod + 1:i + 1]
                period_vol = volume[i - timeperiod + 1:i + 1]
                
                tp_vol_sum = np.sum(period_tp * period_vol)
                vol_sum = np.sum(period_vol)
                
                if vol_sum > 0:
                    vwap[i] = tp_vol_sum / vol_sum
        
        return vwap
    
    def _supertrend(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính SuperTrend.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - atr_period: Số kỳ cho ATR (mặc định: 10)
                - multiplier: Hệ số (mặc định: 3.0)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Tuple (supertrend, direction)
        """
        atr_period = params.get('atr_period', 10)
        multiplier = params.get('multiplier', 3.0)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Tính ATR
        atr = self._atr(df, {'timeperiod': atr_period}, ohlcv_columns)
        
        # Tính Basic UpperBand và LowerBand
        basic_upperband = np.zeros_like(close)
        basic_lowerband = np.zeros_like(close)
        
        for i in range(len(close)):
            basic_upperband[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
            basic_lowerband[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
        
        # Tính Final UpperBand và LowerBand
        final_upperband = np.zeros_like(close)
        final_lowerband = np.zeros_like(close)
        
        # Khởi tạo giá trị đầu tiên
        if len(close) > atr_period:
            final_upperband[atr_period] = basic_upperband[atr_period]
            final_lowerband[atr_period] = basic_lowerband[atr_period]
        
        # Tính Final Bands
        for i in range(atr_period + 1, len(close)):
            # Final Upperband
            if basic_upperband[i] < final_upperband[i - 1] or close[i - 1] > final_upperband[i - 1]:
                final_upperband[i] = basic_upperband[i]
            else:
                final_upperband[i] = final_upperband[i - 1]
            
            # Final Lowerband
            if basic_lowerband[i] > final_lowerband[i - 1] or close[i - 1] < final_lowerband[i - 1]:
                final_lowerband[i] = basic_lowerband[i]
            else:
                final_lowerband[i] = final_lowerband[i - 1]
        
        # Tính SuperTrend và Direction
        supertrend = np.zeros_like(close)
        direction = np.zeros_like(close)  # 1: uptrend, -1: downtrend
        
        if len(close) > atr_period:
            # Giá trị đầu tiên
            if close[atr_period] <= final_upperband[atr_period]:
                supertrend[atr_period] = final_upperband[atr_period]
                direction[atr_period] = -1
            else:
                supertrend[atr_period] = final_lowerband[atr_period]
                direction[atr_period] = 1
            
            # Các giá trị tiếp theo
            for i in range(atr_period + 1, len(close)):
                if supertrend[i - 1] == final_upperband[i - 1] and close[i] <= final_upperband[i]:
                    # Tiếp tục downtrend
                    supertrend[i] = final_upperband[i]
                    direction[i] = -1
                elif supertrend[i - 1] == final_upperband[i - 1] and close[i] > final_upperband[i]:
                    # Chuyển sang uptrend
                    supertrend[i] = final_lowerband[i]
                    direction[i] = 1
                elif supertrend[i - 1] == final_lowerband[i - 1] and close[i] >= final_lowerband[i]:
                    # Tiếp tục uptrend
                    supertrend[i] = final_lowerband[i]
                    direction[i] = 1
                elif supertrend[i - 1] == final_lowerband[i - 1] and close[i] < final_lowerband[i]:
                    # Chuyển sang downtrend
                    supertrend[i] = final_upperband[i]
                    direction[i] = -1
        
        return supertrend, direction
    
    def _rvi(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính Relative Vigor Index (RVI).
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - timeperiod: Số kỳ (mặc định: 10)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        timeperiod = params.get('timeperiod', 10)
        
        open_price = df[ohlcv_columns['open']].values
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Tính Vigor
        vigor = np.zeros_like(close)
        
        for i in range(len(close)):
            vigor[i] = (close[i] - open_price[i]) / (high[i] - low[i] if high[i] != low[i] else 1)
        
        # Tính RVI (SMA của Vigor)
        rvi = np.zeros_like(close)
        rvi[:] = np.nan
        
        for i in range(timeperiod - 1, len(close)):
            rvi[i] = np.mean(vigor[i - timeperiod + 1:i + 1])
        
        return rvi
    
    def _zigzag(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> np.ndarray:
        """
        Tính ZigZag.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - deviation: Phần trăm thay đổi tối thiểu (mặc định: 5.0)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Mảng kết quả
        """
        deviation = params.get('deviation', 5.0)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Tính ZigZag
        zigzag = np.zeros_like(close)
        zigzag[:] = np.nan
        
        # Khởi tạo
        if len(close) > 0:
            zigzag[0] = close[0]
            
            last_peak_idx = 0
            last_direction = 0  # 0: chưa xác định, 1: tăng, -1: giảm
            
            for i in range(1, len(close)):
                # Tính % thay đổi
                percent_change = abs(close[i] - close[last_peak_idx]) / close[last_peak_idx] * 100
                
                if percent_change >= deviation:
                    if close[i] > close[last_peak_idx]:
                        # Đỉnh mới cao hơn
                        if last_direction != 1:
                            # Đổi hướng từ giảm sang tăng
                            zigzag[last_peak_idx] = close[last_peak_idx]
                            last_direction = 1
                        
                        last_peak_idx = i
                        zigzag[i] = close[i]
                    
                    elif close[i] < close[last_peak_idx]:
                        # Đáy mới thấp hơn
                        if last_direction != -1:
                            # Đổi hướng từ tăng sang giảm
                            zigzag[last_peak_idx] = close[last_peak_idx]
                            last_direction = -1
                        
                        last_peak_idx = i
                        zigzag[i] = close[i]
        
        return zigzag
    
    def _pivot_points(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Tính Pivot Points.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số
                - method: Phương pháp ('standard', 'fibonacci', 'camarilla', 'woodie', 'demark')
                - timeperiod: Số kỳ cho mỗi pivot (mặc định: 1)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Tuple (pivot, r1, r2, r3, s1, s2, s3)
        """
        method = params.get('method', 'standard')
        timeperiod = params.get('timeperiod', 1)
        
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        open_price = df[ohlcv_columns['open']].values if 'open' in ohlcv_columns else None
        
        # Khởi tạo mảng kết quả
        pivot = np.zeros_like(close)
        r1 = np.zeros_like(close)
        r2 = np.zeros_like(close)
        r3 = np.zeros_like(close)
        s1 = np.zeros_like(close)
        s2 = np.zeros_like(close)
        s3 = np.zeros_like(close)
        
        pivot[:] = np.nan
        r1[:] = np.nan
        r2[:] = np.nan
        r3[:] = np.nan
        s1[:] = np.nan
        s2[:] = np.nan
        s3[:] = np.nan
        
        # Tính pivot points
        for i in range(timeperiod, len(close), timeperiod):
            prev_high = high[i - timeperiod]
            prev_low = low[i - timeperiod]
            prev_close = close[i - timeperiod]
            # Phải xác định prev_open cho phương pháp Demark
            prev_open = open_price[i - timeperiod] if open_price is not None else prev_close
            
            if method == 'standard':
                # Standard Pivot Points
                pivot[i] = (prev_high + prev_low + prev_close) / 3
                r1[i] = 2 * pivot[i] - prev_low
                s1[i] = 2 * pivot[i] - prev_high
                r2[i] = pivot[i] + (prev_high - prev_low)
                s2[i] = pivot[i] - (prev_high - prev_low)
                r3[i] = r1[i] + (prev_high - prev_low)
                s3[i] = s1[i] - (prev_high - prev_low)
            
            elif method == 'fibonacci':
                # Fibonacci Pivot Points
                pivot[i] = (prev_high + prev_low + prev_close) / 3
                r1[i] = pivot[i] + 0.382 * (prev_high - prev_low)
                s1[i] = pivot[i] - 0.382 * (prev_high - prev_low)
                r2[i] = pivot[i] + 0.618 * (prev_high - prev_low)
                s2[i] = pivot[i] - 0.618 * (prev_high - prev_low)
                r3[i] = pivot[i] + (prev_high - prev_low)
                s3[i] = pivot[i] - (prev_high - prev_low)
            
            elif method == 'camarilla':
                # Camarilla Pivot Points
                pivot[i] = (prev_high + prev_low + prev_close) / 3
                r1[i] = prev_close + 1.1 / 12.0 * (prev_high - prev_low)
                s1[i] = prev_close - 1.1 / 12.0 * (prev_high - prev_low)
                r2[i] = prev_close + 1.1 / 6.0 * (prev_high - prev_low)
                s2[i] = prev_close - 1.1 / 6.0 * (prev_high - prev_low)
                r3[i] = prev_close + 1.1 / 4.0 * (prev_high - prev_low)
                s3[i] = prev_close - 1.1 / 4.0 * (prev_high - prev_low)
            
            elif method == 'woodie':
                # Woodie Pivot Points
                if i > 0:  # Cần giá open của ngày hiện tại
                    current_open = open_price[i] if open_price is not None else close[i - 1]
                    pivot[i] = (prev_high + prev_low + 2 * current_open) / 4
                else:
                    pivot[i] = (prev_high + prev_low + 2 * prev_close) / 4
                
                r1[i] = 2 * pivot[i] - prev_low
                s1[i] = 2 * pivot[i] - prev_high
                r2[i] = pivot[i] + (prev_high - prev_low)
                s2[i] = pivot[i] - (prev_high - prev_low)
            
            elif method == 'demark':
                # Demark Pivot Points
                if prev_close < prev_open:
                    x = prev_high + 2 * prev_low + prev_close
                elif prev_close > prev_open:
                    x = 2 * prev_high + prev_low + prev_close
                else:
                    x = prev_high + prev_low + 2 * prev_close
                
                pivot[i] = x / 4
                r1[i] = x / 2 - prev_low
                s1[i] = x / 2 - prev_high
        
        return pivot, r1, r2, r3, s1, s2, s3
    
    def _heikin_ashi(self, df: pd.DataFrame, params: Dict[str, Any], ohlcv_columns: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Tính Heikin Ashi.
        
        Args:
            df: DataFrame chứa dữ liệu
            params: Tham số (không có tham số đặc biệt)
            ohlcv_columns: Dict ánh xạ tên cột OHLCV
            
        Returns:
            Tuple (ha_open, ha_high, ha_low, ha_close)
        """
        open_price = df[ohlcv_columns['open']].values
        high = df[ohlcv_columns['high']].values
        low = df[ohlcv_columns['low']].values
        close = df[ohlcv_columns['close']].values
        
        # Khởi tạo mảng kết quả
        ha_open = np.zeros_like(close)
        ha_high = np.zeros_like(close)
        ha_low = np.zeros_like(close)
        ha_close = np.zeros_like(close)
        
        # Giá trị đầu tiên
        ha_close[0] = (open_price[0] + high[0] + low[0] + close[0]) / 4
        ha_open[0] = (open_price[0] + close[0]) / 2
        ha_high[0] = high[0]
        ha_low[0] = low[0]
        
        # Các giá trị tiếp theo
        for i in range(1, len(close)):
            ha_close[i] = (open_price[i] + high[i] + low[i] + close[i]) / 4
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
            ha_high[i] = max(high[i], ha_open[i], ha_close[i])
            ha_low[i] = min(low[i], ha_open[i], ha_close[i])
        
        return ha_open, ha_high, ha_low, ha_close