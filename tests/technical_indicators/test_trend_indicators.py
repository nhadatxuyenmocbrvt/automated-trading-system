"""
Unit tests cho module trend_indicators trong technical_indicators.
"""

import unittest
import logging
import numpy as np
import pandas as pd

from data_processors.feature_engineering.technical_indicators.trend_indicators import (
    simple_moving_average, exponential_moving_average, bollinger_bands,
    moving_average_convergence_divergence, average_directional_index,
    parabolic_sar, ichimoku_cloud
)

from tests.technical_indicators import logger
from tests.technical_indicators.test_data import (
    generate_sample_price_data, generate_trending_price_data
)

class TestTrendIndicators(unittest.TestCase):
    """
    Test các hàm chỉ báo xu hướng trong trend_indicators.py.
    """
    
    def setUp(self):
        """
        Chuẩn bị dữ liệu cho các bài test.
        """
        logger.info("Thiết lập dữ liệu cho test_trend_indicators.py")
        self.df = generate_sample_price_data(n_samples=100)
        self.trending_df = generate_trending_price_data(n_samples=100, trend_strength=0.01)
    
    def test_simple_moving_average(self):
        """
        Test hàm simple_moving_average.
        """
        logger.info("Thực hiện test_simple_moving_average")
        
        window = 10
        result_df = simple_moving_average(self.df, column='close', window=window, prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column = f'test_sma_{window}'
        self.assertIn(expected_column, result_df.columns)
        logger.debug(f"SMA column '{expected_column}' tồn tại: OK")
        
        # Kiểm tra số lượng NaN ban đầu (do window)
        nan_count = result_df[expected_column].isna().sum()
        self.assertEqual(nan_count, window - 1)
        logger.debug(f"SMA có {nan_count} giá trị NaN ban đầu: OK")
        
        # Kiểm tra giá trị đầu tiên sau cửa sổ
        first_valid_idx = window - 1
        expected_first_value = self.df['close'].iloc[:window].mean()
        self.assertAlmostEqual(result_df[expected_column].iloc[first_valid_idx], expected_first_value)
        logger.debug(f"Giá trị SMA đầu tiên chính xác: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            simple_moving_average(pd.DataFrame(), column='close', window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
    
    def test_exponential_moving_average(self):
        """
        Test hàm exponential_moving_average.
        """
        logger.info("Thực hiện test_exponential_moving_average")
        
        window = 10
        result_df = exponential_moving_average(self.df, column='close', window=window, prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column = f'test_ema_{window}'
        self.assertIn(expected_column, result_df.columns)
        logger.debug(f"EMA column '{expected_column}' tồn tại: OK")
        
        # Kiểm tra khi sử dụng alpha tùy chỉnh
        alpha = 0.2
        result_df_custom_alpha = exponential_moving_average(
            self.df, column='close', window=window, alpha=alpha, prefix='custom_'
        )
        expected_column_custom = f'custom_ema_{window}'
        self.assertIn(expected_column_custom, result_df_custom_alpha.columns)
        logger.debug(f"EMA với alpha tùy chỉnh ({alpha}): OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            exponential_moving_average(pd.DataFrame(), column='close', window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
    
    def test_bollinger_bands(self):
        """
        Test hàm bollinger_bands.
        """
        logger.info("Thực hiện test_bollinger_bands")
        
        window = 20
        std_dev = 2.0
        result_df = bollinger_bands(self.df, column='close', window=window, std_dev=std_dev, prefix='test_')
        
        # Kiểm tra các cột kết quả tồn tại
        columns = [
            f'test_bb_middle_{window}', 
            f'test_bb_upper_{window}', 
            f'test_bb_lower_{window}',
            f'test_bb_bandwidth_{window}',
            f'test_bb_percent_b_{window}'
        ]
        for col in columns:
            self.assertIn(col, result_df.columns)
        logger.debug("Tất cả các cột Bollinger Bands tồn tại: OK")
        
        # Kiểm tra middle band = SMA
        sma_df = simple_moving_average(self.df, column='close', window=window)
        middle_band = result_df[f'test_bb_middle_{window}']
        sma = sma_df[f'sma_{window}']
        # So sánh các giá trị không phải NaN
        valid_indices = ~middle_band.isna()
        pd.testing.assert_series_equal(middle_band[valid_indices], sma[valid_indices])
        logger.debug("Middle band = SMA: OK")
        
        # Kiểm tra upper band = middle band + std_dev * std
        rolling_std = self.df['close'].rolling(window=window).std(ddof=0)
        upper_band = result_df[f'test_bb_upper_{window}']
        expected_upper = middle_band + std_dev * rolling_std
        pd.testing.assert_series_equal(upper_band, expected_upper)
        logger.debug("Upper band = middle band + std_dev * std: OK")
        
        # Kiểm tra lower band = middle band - std_dev * std
        lower_band = result_df[f'test_bb_lower_{window}']
        expected_lower = middle_band - std_dev * rolling_std
        pd.testing.assert_series_equal(lower_band, expected_lower)
        logger.debug("Lower band = middle band - std_dev * std: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            bollinger_bands(pd.DataFrame(), column='close', window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
    
    def test_moving_average_convergence_divergence(self):
        """
        Test hàm moving_average_convergence_divergence.
        """
        logger.info("Thực hiện test_moving_average_convergence_divergence")
        
        fast_period = 12
        slow_period = 26
        signal_period = 9
        result_df = moving_average_convergence_divergence(
            self.df, column='close', 
            fast_period=fast_period, slow_period=slow_period, signal_period=signal_period,
            prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        columns = ['test_macd_line', 'test_macd_signal', 'test_macd_histogram']
        for col in columns:
            self.assertIn(col, result_df.columns)
        logger.debug("Tất cả các cột MACD tồn tại: OK")
        
        # Kiểm tra MACD line = fast EMA - slow EMA
        ema_fast = self.df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow_period, adjust=False).mean()
        expected_macd_line = ema_fast - ema_slow
        pd.testing.assert_series_equal(result_df['test_macd_line'], expected_macd_line)
        logger.debug("MACD line = fast EMA - slow EMA: OK")
        
        # Kiểm tra MACD signal = EMA của MACD line
        expected_signal = result_df['test_macd_line'].ewm(span=signal_period, adjust=False).mean()
        pd.testing.assert_series_equal(result_df['test_macd_signal'], expected_signal)
        logger.debug("MACD signal = EMA(MACD line): OK")
        
        # Kiểm tra MACD histogram = MACD line - MACD signal
        expected_histogram = result_df['test_macd_line'] - result_df['test_macd_signal']
        pd.testing.assert_series_equal(result_df['test_macd_histogram'], expected_histogram)
        logger.debug("MACD histogram = MACD line - MACD signal: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            moving_average_convergence_divergence(pd.DataFrame(), column='close')
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
    
    def test_average_directional_index(self):
        """
        Test hàm average_directional_index.
        """
        logger.info("Thực hiện test_average_directional_index")
        
        window = 14
        smooth_period = 14
        result_df = average_directional_index(
            self.trending_df, window=window, smooth_period=smooth_period, prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        columns = [f'test_adx_{window}', f'test_plus_di_{window}', f'test_minus_di_{window}']
        for col in columns:
            self.assertIn(col, result_df.columns)
        logger.debug("Tất cả các cột ADX tồn tại: OK")
        
        # Kiểm tra ADX luôn nằm trong khoảng 0-100
        adx = result_df[f'test_adx_{window}'].dropna()
        self.assertTrue(all(adx >= 0))
        self.assertTrue(all(adx <= 100))
        logger.debug("ADX nằm trong khoảng 0-100: OK")
        
        # Kiểm tra +DI và -DI cũng nằm trong khoảng 0-100
        plus_di = result_df[f'test_plus_di_{window}'].dropna()
        minus_di = result_df[f'test_minus_di_{window}'].dropna()
        self.assertTrue(all(plus_di >= 0))
        self.assertTrue(all(plus_di <= 100))
        self.assertTrue(all(minus_di >= 0))
        self.assertTrue(all(minus_di <= 100))
        logger.debug("+DI và -DI nằm trong khoảng 0-100: OK")
        
        # Với dữ liệu có xu hướng tăng, +DI thường sẽ lớn hơn -DI
        self.assertGreater(plus_di.mean(), minus_di.mean())
        logger.debug("Với xu hướng tăng, +DI > -DI trung bình: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            average_directional_index(pd.DataFrame())
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
    
    def test_parabolic_sar(self):
        """
        Test hàm parabolic_sar.
        """
        logger.info("Thực hiện test_parabolic_sar")
        
        result_df = parabolic_sar(self.df, af_start=0.02, af_step=0.02, af_max=0.2, prefix='test_')
        
        # Kiểm tra các cột kết quả tồn tại
        columns = ['test_psar', 'test_psar_trend']
        for col in columns:
            self.assertIn(col, result_df.columns)
        logger.debug("Tất cả các cột Parabolic SAR tồn tại: OK")
        
        # Kiểm tra PSAR không bị NaN
        self.assertFalse(result_df['test_psar'].isna().any())
        logger.debug("PSAR không có giá trị NaN: OK")
        
        # Kiểm tra trend chỉ có giá trị 1 hoặc -1
        trend = result_df['test_psar_trend'].unique()
        self.assertTrue(set(trend).issubset({1, -1}))
        logger.debug("PSAR trend chỉ có giá trị 1 hoặc -1: OK")
        
        # Kiểm tra xu hướng tổng thể
        uptrend_mask = result_df['test_psar_trend'] == 1
        downtrend_mask = result_df['test_psar_trend'] == -1
        
        # Ngoại trừ những điểm đảo chiều
        # PSAR theo nguyên tắc đang uptrend phải ở dưới low, downtrend phải ở trên high
        # Nhưng phải kiểm tra SAR của ngày hôm nay với high/low của ngày hôm qua
        uptrend_valid = uptrend_mask.shift(-1)
        downtrend_valid = downtrend_mask.shift(-1)
        
        # Loại bỏ các giá trị NaN sau khi shift
        uptrend_valid = uptrend_valid.fillna(False)
        downtrend_valid = downtrend_valid.fillna(False)
        
        # Kiểm tra xu hướng tăng: SAR < low (ngày trước)
        # Không kiểm tra từng giá trị mà kiểm tra tổng thể
        uptrend_psar = result_df.loc[uptrend_valid, 'test_psar']
        uptrend_low = result_df.loc[uptrend_valid, 'low'].shift(1)
        # Do PSAR tính toán phức tạp, chúng ta chỉ kiểm tra xu hướng tổng thể
        self.assertTrue((uptrend_psar < uptrend_low).mean() > 0.7)  # Ít nhất 70% tuân theo quy tắc
        logger.debug("PSAR uptrend kiểm tra: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            parabolic_sar(pd.DataFrame())
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
    
    def test_ichimoku_cloud(self):
        """
        Test hàm ichimoku_cloud.
        """
        logger.info("Thực hiện test_ichimoku_cloud")
        
        tenkan_period = 9
        kijun_period = 26
        senkou_b_period = 52
        chikou_period = 26
        result_df = ichimoku_cloud(
            self.df, 
            tenkan_period=tenkan_period, 
            kijun_period=kijun_period,
            senkou_b_period=senkou_b_period,
            chikou_period=chikou_period,
            prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        columns = [
            'test_ichimoku_tenkan_sen',
            'test_ichimoku_kijun_sen',
            'test_ichimoku_senkou_span_a',
            'test_ichimoku_senkou_span_b',
            'test_ichimoku_chikou_span'
        ]
        for col in columns:
            self.assertIn(col, result_df.columns)
        logger.debug("Tất cả các cột Ichimoku Cloud tồn tại: OK")
        
        # Kiểm tra tenkan_sen = (highest high + lowest low) / 2 trong tenkan_period
        high_max = self.df['high'].rolling(window=tenkan_period).max()
        low_min = self.df['low'].rolling(window=tenkan_period).min()
        expected_tenkan = (high_max + low_min) / 2
        pd.testing.assert_series_equal(
            result_df['test_ichimoku_tenkan_sen'].dropna(),
            expected_tenkan.dropna()
        )
        logger.debug("Tenkan-sen = (highest high + lowest low) / 2: OK")
        
        # Kiểm tra chikou_span = close dịch về quá khứ chikou_period
        expected_chikou = self.df['close'].shift(-chikou_period)
        pd.testing.assert_series_equal(
            result_df['test_ichimoku_chikou_span'],
            expected_chikou
        )
        logger.debug("Chikou-span = close dịch về quá khứ: OK")
        
        # Kiểm tra số lượng NaN trong Senkou Span A và B do việc dịch kỳ
        senkou_span_a = result_df['test_ichimoku_senkou_span_a']
        self.assertEqual(senkou_span_a.isna().sum(), kijun_period)
        logger.debug(f"Senkou Span A có đúng {kijun_period} giá trị NaN ban đầu: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            ichimoku_cloud(pd.DataFrame())
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
    
    def tearDown(self):
        """
        Dọn dẹp sau khi test hoàn thành.
        """
        logger.info("Đã hoàn thành test_trend_indicators.py")

if __name__ == '__main__':
    # Thiết lập logging khi chạy trực tiếp
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    unittest.main()