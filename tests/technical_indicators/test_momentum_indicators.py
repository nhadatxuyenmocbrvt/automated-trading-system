"""
Unit tests cho module momentum_indicators trong technical_indicators.
"""

import unittest
import logging
import numpy as np
import pandas as pd

from data_processors.feature_engineering.technical_indicators.momentum_indicators import (
    relative_strength_index, stochastic_oscillator, commodity_channel_index,
    williams_r, rate_of_change, money_flow_index, true_strength_index
)

from tests.technical_indicators import logger
from tests.technical_indicators.test_data import (
    generate_sample_price_data, generate_trending_price_data
)

class TestMomentumIndicators(unittest.TestCase):
    """
    Test các hàm chỉ báo động lượng trong momentum_indicators.py.
    """
    
    def setUp(self):
        """
        Chuẩn bị dữ liệu cho các bài test.
        """
        logger.info("Thiết lập dữ liệu cho test_momentum_indicators.py")
        self.df = generate_sample_price_data(n_samples=100)
        self.trending_df = generate_trending_price_data(n_samples=100, trend_strength=0.01)
    
    def test_relative_strength_index(self):
        """
        Test hàm relative_strength_index.
        """
        logger.info("Thực hiện test_relative_strength_index")
        
        window = 14
        
        # Test với method='ema'
        result_df_ema = relative_strength_index(self.df, column='close', window=window, method='ema', prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column = f'test_rsi_{window}'
        self.assertIn(expected_column, result_df_ema.columns)
        logger.debug(f"RSI column '{expected_column}' tồn tại: OK")
        
        # Kiểm tra RSI nằm trong khoảng 0-100
        rsi_values = result_df_ema[expected_column].dropna()
        self.assertTrue(all(rsi_values >= 0))
        self.assertTrue(all(rsi_values <= 100))
        logger.debug("RSI nằm trong khoảng 0-100: OK")
        
        # Test với method='sma'
        result_df_sma = relative_strength_index(self.df, column='close', window=window, method='sma', prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        self.assertIn(expected_column, result_df_sma.columns)
        logger.debug(f"RSI với method='sma' column '{expected_column}' tồn tại: OK")
        
        # Kiểm tra RSI nằm trong khoảng 0-100
        rsi_values_sma = result_df_sma[expected_column].dropna()
        self.assertTrue(all(rsi_values_sma >= 0))
        self.assertTrue(all(rsi_values_sma <= 100))
        logger.debug("RSI với method='sma' nằm trong khoảng 0-100: OK")
        
        # Kiểm tra RSI với dữ liệu có xu hướng tăng
        result_df_trending = relative_strength_index(self.trending_df, column='close', window=window, prefix='')
        rsi_trending = result_df_trending[f'rsi_{window}'].dropna()
        
        # Với dữ liệu xu hướng tăng, RSI thường > 50
        self.assertGreater(rsi_trending.mean(), 50)
        logger.debug("Với dữ liệu xu hướng tăng, RSI trung bình > 50: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            relative_strength_index(pd.DataFrame(), column='close', window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_relative_strength_index")
    
    def test_stochastic_oscillator(self):
        """
        Test hàm stochastic_oscillator.
        """
        logger.info("Thực hiện test_stochastic_oscillator")
        
        k_period = 14
        d_period = 3
        
        result_df = stochastic_oscillator(
            self.df, k_period=k_period, d_period=d_period, smooth_k=1, prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        k_column = f'test_stoch_k_{k_period}'
        d_column = f'test_stoch_d_{k_period}_{d_period}'
        self.assertIn(k_column, result_df.columns)
        self.assertIn(d_column, result_df.columns)
        logger.debug(f"Stochastic Oscillator columns '{k_column}' và '{d_column}' tồn tại: OK")
        
        # Kiểm tra %K và %D nằm trong khoảng 0-100
        k_values = result_df[k_column].dropna()
        d_values = result_df[d_column].dropna()
        self.assertTrue(all(k_values >= 0))
        self.assertTrue(all(k_values <= 100))
        self.assertTrue(all(d_values >= 0))
        self.assertTrue(all(d_values <= 100))
        logger.debug("%K và %D nằm trong khoảng 0-100: OK")
        
        # Tính %K thủ công và so sánh
        high_max = self.df['high'].rolling(window=k_period).max()
        low_min = self.df['low'].rolling(window=k_period).min()
        
        # Tránh chia cho 0
        price_range = high_max - low_min
        price_range_safe = price_range.replace(0, np.nan)
        
        expected_k = 100 * ((self.df['close'] - low_min) / price_range_safe)
        
        # So sánh kết quả, bỏ qua NaN
        pd.testing.assert_series_equal(
            result_df[k_column].dropna(),
            expected_k.dropna(),
            check_dtype=False
        )
        logger.debug("%K = 100 * ((Close - Low Min) / (High Max - Low Min)): OK")
        
        # Tính %D thủ công (SMA của %K) và so sánh
        expected_d = expected_k.rolling(window=d_period).mean()
        pd.testing.assert_series_equal(
            result_df[d_column].dropna(),
            expected_d.dropna(),
            check_dtype=False
        )
        logger.debug("%D = SMA(%K): OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            stochastic_oscillator(pd.DataFrame(), k_period=k_period, d_period=d_period)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_stochastic_oscillator")
    
    def test_williams_r(self):
        """
        Test hàm williams_r.
        """
        logger.info("Thực hiện test_williams_r")
        
        window = 14
        
        result_df = williams_r(self.df, window=window, prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column = f'test_williams_r_{window}'
        self.assertIn(expected_column, result_df.columns)
        logger.debug(f"Williams %R column '{expected_column}' tồn tại: OK")
        
        # Kiểm tra Williams %R nằm trong khoảng -100 đến 0
        williams_values = result_df[expected_column].dropna()
        self.assertTrue(all(williams_values >= -100))
        self.assertTrue(all(williams_values <= 0))
        logger.debug("Williams %R nằm trong khoảng -100 đến 0: OK")
        
        # Tính Williams %R thủ công và so sánh
        high_max = self.df['high'].rolling(window=window).max()
        low_min = self.df['low'].rolling(window=window).min()
        
        # Tránh chia cho 0
        price_range = high_max - low_min
        price_range_safe = price_range.replace(0, np.nan)
        
        expected_williams = -100 * ((high_max - self.df['close']) / price_range_safe)
        
        # So sánh kết quả, bỏ qua NaN
        pd.testing.assert_series_equal(
            result_df[expected_column].dropna(),
            expected_williams.dropna(),
            check_dtype=False
        )
        logger.debug("Williams %R = -100 * ((High Max - Close) / (High Max - Low Min)): OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            williams_r(pd.DataFrame(), window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_williams_r")
    
    def test_commodity_channel_index(self):
        """
        Test hàm commodity_channel_index.
        """
        logger.info("Thực hiện test_commodity_channel_index")
        
        window = 20
        constant = 0.015
        
        result_df = commodity_channel_index(self.df, window=window, constant=constant, prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column = f'test_cci_{window}'
        self.assertIn(expected_column, result_df.columns)
        logger.debug(f"CCI column '{expected_column}' tồn tại: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            commodity_channel_index(pd.DataFrame(), window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_commodity_channel_index")
    
    def test_rate_of_change(self):
        """
        Test hàm rate_of_change.
        """
        logger.info("Thực hiện test_rate_of_change")
        
        window = 9
        
        # Test với percentage=True
        result_df_pct = rate_of_change(self.df, column='close', window=window, percentage=True, prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column_pct = f'test_roc_pct_{window}'
        self.assertIn(expected_column_pct, result_df_pct.columns)
        logger.debug(f"ROC percentage column '{expected_column_pct}' tồn tại: OK")
        
        # Tính ROC Percentage thủ công và so sánh
        price_n_ago = self.df['close'].shift(window)
        
        # Tránh chia cho 0
        price_n_ago_safe = price_n_ago.replace(0, np.nan)
        
        expected_roc_pct = (self.df['close'] - price_n_ago_safe) / price_n_ago_safe * 100
        
        # So sánh kết quả, bỏ qua NaN
        pd.testing.assert_series_equal(
            result_df_pct[expected_column_pct].dropna(),
            expected_roc_pct.dropna(),
            check_dtype=False
        )
        logger.debug("ROC % = (Close - Close n periods ago) / Close n periods ago * 100: OK")
        
        # Test với percentage=False
        result_df_abs = rate_of_change(self.df, column='close', window=window, percentage=False, prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column_abs = f'test_roc_{window}'
        self.assertIn(expected_column_abs, result_df_abs.columns)
        logger.debug(f"ROC absolute column '{expected_column_abs}' tồn tại: OK")
        
        # Tính ROC Absolute thủ công và so sánh
        expected_roc_abs = self.df['close'] - price_n_ago
        
        # So sánh kết quả, bỏ qua NaN
        pd.testing.assert_series_equal(
            result_df_abs[expected_column_abs].dropna(),
            expected_roc_abs.dropna(),
            check_dtype=False
        )
        logger.debug("ROC absolute = Close - Close n periods ago: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            rate_of_change(pd.DataFrame(), column='close', window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_rate_of_change")
    
    def test_money_flow_index(self):
        """
        Test hàm money_flow_index.
        """
        logger.info("Thực hiện test_money_flow_index")
        
        window = 14
        
        result_df = money_flow_index(self.df, window=window, prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column = f'test_mfi_{window}'
        self.assertIn(expected_column, result_df.columns)
        logger.debug(f"MFI column '{expected_column}' tồn tại: OK")
        
        # Kiểm tra MFI nằm trong khoảng 0-100
        mfi_values = result_df[expected_column].dropna()
        self.assertTrue(all(mfi_values >= 0))
        self.assertTrue(all(mfi_values <= 100))
        logger.debug("MFI nằm trong khoảng 0-100: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            money_flow_index(pd.DataFrame(), window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_money_flow_index")
    
    def test_true_strength_index(self):
        """
        Test hàm true_strength_index.
        """
        logger.info("Thực hiện test_true_strength_index")
        
        long_window = 25
        short_window = 13
        signal_window = 7
        
        result_df = true_strength_index(
            self.df, column='close', 
            long_window=long_window, short_window=short_window, signal_window=signal_window,
            prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        tsi_column = f'test_tsi_{long_window}_{short_window}'
        signal_column = f'test_tsi_signal_{signal_window}'
        self.assertIn(tsi_column, result_df.columns)
        self.assertIn(signal_column, result_df.columns)
        logger.debug(f"TSI columns '{tsi_column}' và '{signal_column}' tồn tại: OK")
        
        # Kiểm tra tính chất thống kê của TSI
        tsi_values = result_df[tsi_column].dropna()
        # TSI thường nằm trong khoảng -100 đến +100
        self.assertTrue(all(tsi_values >= -100))
        self.assertTrue(all(tsi_values <= 100))
        logger.debug("TSI nằm trong khoảng -100 đến +100: OK")
        
        # Signal là EMA của TSI
        signal_values = result_df[signal_column].dropna()
        expected_signal = tsi_values.ewm(span=signal_window, adjust=False).mean()
        pd.testing.assert_series_equal(
            signal_values,
            expected_signal.dropna(),
            check_dtype=False
        )
        logger.debug("Signal = EMA của TSI: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            true_strength_index(pd.DataFrame(), column='close', 
                               long_window=long_window, short_window=short_window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_true_strength_index")
    
    def tearDown(self):
        """
        Dọn dẹp sau khi test hoàn thành.
        """
        logger.info("Đã hoàn thành test_momentum_indicators.py")

if __name__ == '__main__':
    # Thiết lập logging khi chạy trực tiếp
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    unittest.main()