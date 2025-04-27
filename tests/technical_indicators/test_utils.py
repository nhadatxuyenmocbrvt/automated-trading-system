"""
Unit tests cho module utils trong technical_indicators.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data_processors.feature_engineering.technical_indicators.utils import (
    validate_price_data, get_candle_columns, prepare_price_data,
    exponential_weights, calculate_weighted_average, find_local_extrema,
    true_range, normalize_indicator, crossover, crossunder
)

from tests.technical_indicators import logger
from tests.technical_indicators.test_data import (
    generate_sample_price_data
)

class TestUtils(unittest.TestCase):
    """
    Test các hàm tiện ích trong utils.py.
    """
    
    def setUp(self):
        """
        Chuẩn bị dữ liệu cho các bài test.
        """
        logger.info("Thiết lập dữ liệu cho test_utils.py")
        self.df = generate_sample_price_data()
    
    def test_validate_price_data(self):
        """
        Test hàm validate_price_data.
        """
        logger.info("Thực hiện test_validate_price_data")
        
        # Test với DataFrame hợp lệ
        self.assertTrue(validate_price_data(self.df, ['open', 'high', 'low', 'close']))
        
        # Test với DataFrame thiếu một cột
        df_missing_col = self.df.drop('volume', axis=1)
        self.assertTrue(validate_price_data(df_missing_col, ['open', 'high', 'low']))
        self.assertFalse(validate_price_data(df_missing_col, ['open', 'high', 'low', 'volume']))
        
        # Test với DataFrame rỗng
        empty_df = pd.DataFrame()
        self.assertFalse(validate_price_data(empty_df))
        
        logger.debug("Hoàn thành test_validate_price_data")
    
    def test_get_candle_columns(self):
        """
        Test hàm get_candle_columns.
        """
        logger.info("Thực hiện test_get_candle_columns")
        
        # Test với DataFrame có tên cột tiêu chuẩn
        result = get_candle_columns(self.df)
        expected = {
            'open': 'open', 
            'high': 'high', 
            'low': 'low', 
            'close': 'close', 
            'volume': 'volume'
        }
        self.assertEqual(result, expected)
        
        # Test với DataFrame có tên cột viết hoa
        df_uppercase = self.df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        result = get_candle_columns(df_uppercase)
        expected = {
            'open': 'Open', 
            'high': 'High', 
            'low': 'Low', 
            'close': 'Close', 
            'volume': 'Volume'
        }
        self.assertEqual(result, expected)
        
        logger.debug("Hoàn thành test_get_candle_columns")
    
    def test_prepare_price_data(self):
        """
        Test hàm prepare_price_data.
        """
        logger.info("Thực hiện test_prepare_price_data")
        
        # Test với DataFrame đã có tên cột chuẩn
        result_df = prepare_price_data(self.df)
        self.assertEqual(list(result_df.columns), ['open', 'high', 'low', 'close', 'volume'])
        
        # Test với DataFrame có tên cột viết hoa
        df_uppercase = self.df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        result_df = prepare_price_data(df_uppercase)
        self.assertEqual(list(result_df.columns), ['open', 'high', 'low', 'close', 'volume'])
        
        logger.debug("Hoàn thành test_prepare_price_data")
    
    def test_exponential_weights(self):
        """
        Test hàm exponential_weights.
        """
        logger.info("Thực hiện test_exponential_weights")
        
        window = 5
        
        # Test với alpha mặc định
        weights = exponential_weights(window)
        self.assertEqual(len(weights), window)
        self.assertAlmostEqual(np.sum(weights), 1.0)
        
        # Kiểm tra xem weights có giảm dần không (từ cuối đến đầu)
        for i in range(1, len(weights)):
            self.assertGreater(weights[i-1], weights[i])
        
        # Test với alpha được chỉ định
        alpha = 0.3
        weights = exponential_weights(window, alpha)
        self.assertEqual(len(weights), window)
        self.assertAlmostEqual(np.sum(weights), 1.0)
        
        logger.debug("Hoàn thành test_exponential_weights")
    
    def test_calculate_weighted_average(self):
        """
        Test hàm calculate_weighted_average.
        """
        logger.info("Thực hiện test_calculate_weighted_average")
        
        series = pd.Series([1, 2, 3, 4, 5])
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Test với Series và weights
        result = calculate_weighted_average(series, weights)
        # Do cửa sổ trượt, kết quả sẽ có độ dài bằng độ dài của Series
        self.assertEqual(len(result), len(series))
        # Chỉ có giá trị từ vị trí thứ len(weights)-1 trở đi
        self.assertTrue(np.isnan(result.iloc[0]))
        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertTrue(np.isnan(result.iloc[2]))
        self.assertFalse(np.isnan(result.iloc[3]))
        self.assertFalse(np.isnan(result.iloc[4]))
        
        logger.debug("Hoàn thành test_calculate_weighted_average")
    
    def test_find_local_extrema(self):
        """
        Test hàm find_local_extrema.
        """
        logger.info("Thực hiện test_find_local_extrema")
        
        # Tạo dữ liệu với cực đại và cực tiểu rõ ràng
        series = pd.Series([1, 2, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1])
        
        # Test với method='simple'
        peaks, troughs = find_local_extrema(series, window=1, method='simple')
        # Cực đại ở các vị trí 2, 9
        self.assertTrue(peaks.iloc[2])
        self.assertTrue(peaks.iloc[9])
        # Cực tiểu ở các vị trí 5, 13
        self.assertTrue(troughs.iloc[5])
        self.assertTrue(troughs.iloc[13])
        
        # Test với method='robust'
        peaks, troughs = find_local_extrema(series, window=3, method='robust')
        # Cực đại ở vị trí 9
        self.assertTrue(peaks.iloc[9])
        # Cực tiểu ở vị trí 5 và 13
        self.assertTrue(troughs.iloc[5])
        self.assertTrue(troughs.iloc[13])
        
        logger.debug("Hoàn thành test_find_local_extrema")
    
    def test_true_range(self):
        """
        Test hàm true_range.
        """
        logger.info("Thực hiện test_true_range")
        
        high = pd.Series([120, 125, 130, 135, 140])
        low = pd.Series([110, 115, 120, 125, 130])
        close = pd.Series([115, 120, 125, 130, 135])
        
        tr = true_range(high, low, close)
        
        self.assertEqual(len(tr), len(high))
        self.assertEqual(tr.iloc[0], 10)  # high[0] - low[0]
        self.assertEqual(tr.iloc[1], 10)  # max(high[1]-low[1], high[1]-close[0], close[0]-low[1])
        
        logger.debug("Hoàn thành test_true_range")
    
    def test_normalize_indicator(self):
        """
        Test hàm normalize_indicator.
        """
        logger.info("Thực hiện test_normalize_indicator")
        
        # Tạo dữ liệu với phạm vi 0-100
        series = pd.Series([0, 25, 50, 75, 100])
        
        # Test với method='minmax'
        result = normalize_indicator(series, method='minmax')
        self.assertEqual(result.iloc[0], 0.0)  # min -> 0
        self.assertEqual(result.iloc[4], 1.0)  # max -> 1
        
        # Test với method='zscore'
        result = normalize_indicator(series, method='zscore')
        self.assertAlmostEqual(result.mean(), 0.0, places=10)  # Mean = 0
        self.assertAlmostEqual(result.std(), 1.0)  # StdDev = 1
        
        logger.debug("Hoàn thành test_normalize_indicator")
    
    def test_crossover(self):
        """
        Test hàm crossover.
        """
        logger.info("Thực hiện test_crossover")
        
        series1 = pd.Series([10, 20, 30, 25, 35])
        series2 = pd.Series([15, 25, 20, 30, 30])
        
        # Crossover xảy ra khi:
        # 1. series1[i-1] < series2[i-1]
        # 2. series1[i] >= series2[i]
        
        result = crossover(series1, series2)
        
        self.assertFalse(result.iloc[0])  # Không có dữ liệu trước đó
        self.assertFalse(result.iloc[1])  # 10 < 15, 20 < 25 -> không crossover
        self.assertTrue(result.iloc[2])   # 20 < 25, 30 > 20 -> crossover
        self.assertFalse(result.iloc[3])  # 30 > 20, 25 < 30 -> không crossover
        self.assertTrue(result.iloc[4])   # 25 < 30, 35 > 30 -> crossover
        
        logger.debug("Hoàn thành test_crossover")
    
    def test_crossunder(self):
        """
        Test hàm crossunder.
        """
        logger.info("Thực hiện test_crossunder")
        
        series1 = pd.Series([20, 30, 25, 35, 25])
        series2 = pd.Series([15, 25, 30, 30, 30])
        
        # Crossunder xảy ra khi:
        # 1. series1[i-1] > series2[i-1]
        # 2. series1[i] <= series2[i]
        
        result = crossunder(series1, series2)
        
        self.assertFalse(result.iloc[0])  # Không có dữ liệu trước đó
        self.assertFalse(result.iloc[1])  # 20 > 15, 30 > 25 -> không crossunder
        self.assertTrue(result.iloc[2])   # 30 > 25, 25 < 30 -> crossunder
        self.assertFalse(result.iloc[3])  # 25 < 30, 35 > 30 -> không crossunder
        self.assertTrue(result.iloc[4])   # 35 > 30, 25 < 30 -> crossunder
        
        logger.debug("Hoàn thành test_crossunder")
    
    def tearDown(self):
        """
        Dọn dẹp sau khi test hoàn thành.
        """
        logger.info("Đã hoàn thành test_utils.py")

if __name__ == '__main__':
    # Thiết lập logging khi chạy trực tiếp
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    unittest.main()