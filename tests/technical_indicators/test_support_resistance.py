"""
Unit tests cho module support_resistance trong technical_indicators.
"""

import unittest
import logging
import numpy as np
import pandas as pd

from data_processors.feature_engineering.technical_indicators.support_resistance import (
    detect_support_resistance, pivot_points, fibonacci_retracement, find_chart_patterns
)

from tests.technical_indicators import logger
from tests.technical_indicators.test_data import (
    generate_sample_price_data, generate_trending_price_data
)

class TestSupportResistance(unittest.TestCase):
    """
    Test các hàm chỉ báo hỗ trợ và kháng cự trong support_resistance.py.
    """
    
    def setUp(self):
        """
        Chuẩn bị dữ liệu cho các bài test.
        """
        logger.info("Thiết lập dữ liệu cho test_support_resistance.py")
        self.df = generate_sample_price_data(n_samples=200)
        self.trending_df = generate_trending_price_data(n_samples=200, trend_strength=0.01)
    
    def test_detect_support_resistance(self):
        """
        Test hàm detect_support_resistance.
        """
        logger.info("Thực hiện test_detect_support_resistance")
        
        window = 10
        min_touches = 2
        tolerance = 0.02
        
        result_df = detect_support_resistance(
            self.df, price_column='close', 
            window=window, min_touches=min_touches, tolerance=tolerance,
            prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        self.assertIn('test_is_peak', result_df.columns)
        self.assertIn('test_is_trough', result_df.columns)
        logger.debug("Các cột is_peak và is_trough tồn tại: OK")
        
        # Kiểm tra các mức hỗ trợ và kháng cự có được tìm thấy
        self.assertTrue('test_resistance_levels' in result_df.attrs or 'test_support_levels' in result_df.attrs)
        logger.debug("Các mức hỗ trợ/kháng cự được tìm thấy: OK")
        
        # Kiểm tra các thuộc tính is_peak và is_trough không thể đồng thời là True
        overlap = (result_df['test_is_peak'] & result_df['test_is_trough']).sum()
        self.assertEqual(overlap, 0)
        logger.debug("Không có điểm nào vừa là peak vừa là trough: OK")
        
        # Kiểm tra next_resistance > giá hiện tại
        if 'test_next_resistance' in result_df.columns:
            self.assertGreater(result_df['test_next_resistance'].iloc[-1], result_df['close'].iloc[-1])
            logger.debug("next_resistance > giá hiện tại: OK")
        
        # Kiểm tra next_support < giá hiện tại
        if 'test_next_support' in result_df.columns:
            self.assertLess(result_df['test_next_support'].iloc[-1], result_df['close'].iloc[-1])
            logger.debug("next_support < giá hiện tại: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            detect_support_resistance(pd.DataFrame(), price_column='close', window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
    
    def test_pivot_points(self):
        """
        Test hàm pivot_points.
        """
        logger.info("Thực hiện test_pivot_points")
        
        # Test pivot_points với phương pháp standard
        result_df_standard = pivot_points(
            self.df, high_column='high', low_column='low', close_column='close',
            method='standard', prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        standard_columns = [
            'test_pivot', 'test_support1', 'test_support2', 'test_support3',
            'test_resistance1', 'test_resistance2', 'test_resistance3'
        ]
        for col in standard_columns:
            self.assertIn(col, result_df_standard.columns)
        logger.debug("Tất cả các cột Pivot Points Standard tồn tại: OK")
        
        # Tính pivot point standard thủ công và so sánh
        # Pivot = (High + Low + Close) / 3
        expected_pivot = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        pd.testing.assert_series_equal(
            result_df_standard['test_pivot'],
            expected_pivot,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("Pivot = (High + Low + Close) / 3: OK")
        
        # Tính Support1 = (2 * Pivot) - High
        expected_s1 = (2 * expected_pivot) - self.df['high']
        pd.testing.assert_series_equal(
            result_df_standard['test_support1'],
            expected_s1,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("Support1 = (2 * Pivot) - High: OK")
        
        # Tính Resistance1 = (2 * Pivot) - Low
        expected_r1 = (2 * expected_pivot) - self.df['low']
        pd.testing.assert_series_equal(
            result_df_standard['test_resistance1'],
            expected_r1,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("Resistance1 = (2 * Pivot) - Low: OK")
        
        # Test pivot_points với phương pháp fibonacci
        result_df_fib = pivot_points(
            self.df, high_column='high', low_column='low', close_column='close',
            method='fibonacci', prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        fib_columns = [
            'test_pivot_fib', 'test_support1_fib', 'test_support2_fib', 'test_support3_fib',
            'test_resistance1_fib', 'test_resistance2_fib', 'test_resistance3_fib'
        ]
        for col in fib_columns:
            self.assertIn(col, result_df_fib.columns)
        logger.debug("Tất cả các cột Pivot Points Fibonacci tồn tại: OK")
        
        # Test pivot_points với phương pháp woodie
        result_df_woodie = pivot_points(
            self.df, high_column='high', low_column='low', close_column='close',
            method='woodie', prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        woodie_columns = [
            'test_pivot_woodie', 'test_support1_woodie', 'test_support2_woodie',
            'test_resistance1_woodie', 'test_resistance2_woodie'
        ]
        for col in woodie_columns:
            self.assertIn(col, result_df_woodie.columns)
        logger.debug("Tất cả các cột Pivot Points Woodie tồn tại: OK")
        
        # Tính pivot point woodie thủ công và so sánh
        # Pivot Woodie = (High + Low + 2 * Close) / 4
        expected_pivot_woodie = (self.df['high'] + self.df['low'] + 2 * self.df['close']) / 4
        pd.testing.assert_series_equal(
            result_df_woodie['test_pivot_woodie'],
            expected_pivot_woodie,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("Pivot Woodie = (High + Low + 2 * Close) / 4: OK")
        
        # Test pivot_points với phương pháp camarilla
        result_df_camarilla = pivot_points(
            self.df, high_column='high', low_column='low', close_column='close',
            method='camarilla', prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        camarilla_columns = [
            'test_resistance4_camarilla', 'test_resistance3_camarilla', 'test_resistance2_camarilla', 'test_resistance1_camarilla',
            'test_support1_camarilla', 'test_support2_camarilla', 'test_support3_camarilla', 'test_support4_camarilla'
        ]
        for col in camarilla_columns:
            self.assertIn(col, result_df_camarilla.columns)
        logger.debug("Tất cả các cột Pivot Points Camarilla tồn tại: OK")
        
        # Test pivot_points với phương pháp demark
        result_df_demark = pivot_points(
            self.df, high_column='high', low_column='low', close_column='close',
            method='demark', prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        demark_columns = ['test_pivot_demark', 'test_support1_demark', 'test_resistance1_demark']
        for col in demark_columns:
            self.assertIn(col, result_df_demark.columns)
        logger.debug("Tất cả các cột Pivot Points DeMark tồn tại: OK")
        
        # Kiểm tra với phương pháp không hợp lệ
        with self.assertRaises(ValueError):
            pivot_points(self.df, method='invalid_method')
        logger.debug("Kiểm tra xử lý lỗi với phương pháp không hợp lệ: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            pivot_points(pd.DataFrame())
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
    
    def test_fibonacci_retracement(self):
        """
        Test hàm fibonacci_retracement.
        """
        logger.info("Thực hiện test_fibonacci_retracement")
        
        window = 100
        
        # Test với trend='up'
        result_df_up = fibonacci_retracement(
            self.df, high_column='high', low_column='low',
            trend='up', window=window, prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        fib_columns = [
            'test_fib_trend', 'test_fib_swing_high', 'test_fib_swing_low',
            'test_fib_0', 'test_fib_236', 'test_fib_382', 'test_fib_5',
            'test_fib_618', 'test_fib_786', 'test_fib_1',
            'test_fib_ext_1618', 'test_fib_ext_2618'
        ]
        for col in fib_columns:
            self.assertIn(col, result_df_up.columns)
        logger.debug("Tất cả các cột Fibonacci Retracement tồn tại: OK")
        
        # Kiểm tra các giá trị Fibonacci có đúng thứ tự
        self.assertTrue(
            result_df_up['test_fib_0'].iloc[0] <
            result_df_up['test_fib_236'].iloc[0] <
            result_df_up['test_fib_382'].iloc[0] <
            result_df_up['test_fib_5'].iloc[0] <
            result_df_up['test_fib_618'].iloc[0] <
            result_df_up['test_fib_786'].iloc[0] <
            result_df_up['test_fib_1'].iloc[0] <
            result_df_up['test_fib_ext_1618'].iloc[0] <
            result_df_up['test_fib_ext_2618'].iloc[0]
        )
        logger.debug("Giá trị Fibonacci cho xu hướng tăng có thứ tự đúng: OK")
        
        # Kiểm tra swing_low và swing_high
        high_window = self.df['high'].tail(window)
        low_window = self.df['low'].tail(window)
        
        self.assertEqual(result_df_up['test_fib_swing_high'].iloc[0], high_window.max())
        self.assertEqual(result_df_up['test_fib_swing_low'].iloc[0], low_window.min())
        logger.debug("swing_high = max(high) và swing_low = min(low): OK")
        
        # Test với trend='down'
        result_df_down = fibonacci_retracement(
            self.df, high_column='high', low_column='low',
            trend='down', window=window, prefix='test_'
        )
        
        # Kiểm tra các giá trị Fibonacci có đúng thứ tự (đảo ngược vì xu hướng giảm)
        self.assertTrue(
            result_df_down['test_fib_0'].iloc[0] >
            result_df_down['test_fib_236'].iloc[0] >
            result_df_down['test_fib_382'].iloc[0] >
            result_df_down['test_fib_5'].iloc[0] >
            result_df_down['test_fib_618'].iloc[0] >
            result_df_down['test_fib_786'].iloc[0] >
            result_df_down['test_fib_1'].iloc[0] >
            result_df_down['test_fib_ext_1618'].iloc[0] >
            result_df_down['test_fib_ext_2618'].iloc[0]
        )
        logger.debug("Giá trị Fibonacci cho xu hướng giảm có thứ tự đúng: OK")
        
        # Test với trend='auto'
        result_df_auto = fibonacci_retracement(
            self.trending_df, high_column='high', low_column='low',
            trend='auto', window=window, prefix='test_'
        )
        
        # Kiểm tra trend được xác định
        self.assertIn(result_df_auto['test_fib_trend'].iloc[0], ['up', 'down'])
        logger.debug("Trend được xác định tự động: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            fibonacci_retracement(pd.DataFrame(), high_column='high', low_column='low')
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
    
    def test_find_chart_patterns(self):
        """
        Test hàm find_chart_patterns.
        """
        logger.info("Thực hiện test_find_chart_patterns")
        
        window = 50
        tolerance = 0.03
        
        result_df = find_chart_patterns(
            self.df, price_column='close', 
            window=window, tolerance=tolerance,
            prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        pattern_columns = [
            'test_double_top', 'test_double_bottom', 'test_head_shoulders',
            'test_inv_head_shoulders', 'test_triangle'
        ]
        for col in pattern_columns:
            self.assertIn(col, result_df.columns)
        logger.debug("Tất cả các cột Chart Patterns tồn tại: OK")
        
        # Các giá trị của các cột mẫu hình nên là boolean
        for column in pattern_columns:
            self.assertTrue(pd.api.types.is_bool_dtype(result_df[column]))
        logger.debug("Tất cả các cột Chart Patterns có kiểu dữ liệu boolean: OK")
        
        # Khi áp dụng cho dữ liệu xu hướng, có thể phát hiện mẫu hình triangle
        result_df_trending = find_chart_patterns(
            self.trending_df, price_column='close', 
            window=window, tolerance=tolerance
        )
        
        # Không đảm bảo rằng mẫu hình sẽ được phát hiện, chỉ kiểm tra hàm không bị lỗi
        self.assertIn('triangle', result_df_trending.columns)
        logger.debug("Phát hiện mẫu hình trên dữ liệu xu hướng: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            find_chart_patterns(pd.DataFrame(), price_column='close', window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
    
    def tearDown(self):
        """
        Dọn dẹp sau khi test hoàn thành.
        """
        logger.info("Đã hoàn thành test_support_resistance.py")

if __name__ == '__main__':
    # Thiết lập logging khi chạy trực tiếp
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    unittest.main()