"""
Unit tests cho module support_resistance trong technical_indicators.
"""

import unittest
import numpy as np
import pandas as pd

from data_processors.feature_engineering.technical_indicators.support_resistance import (
    detect_support_resistance, pivot_points, fibonacci_retracement, find_chart_patterns
)

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
        self.df = generate_sample_price_data(n_samples=200)
        self.trending_df = generate_trending_price_data(n_samples=200, trend_strength=0.01)
    
    def test_detect_support_resistance(self):
        """
        Test hàm detect_support_resistance.
        """
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
        
        # Kiểm tra các mức hỗ trợ và kháng cự có được tìm thấy
        self.assertTrue('test_resistance_levels' in result_df.attrs or 'test_support_levels' in result_df.attrs)
        
        # Kiểm tra các thuộc tính is_peak và is_trough không thể đồng thời là True
        overlap = (result_df['test_is_peak'] & result_df['test_is_trough']).sum()
        self.assertEqual(overlap, 0)
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            detect_support_resistance(pd.DataFrame(), price_column='close', window=window)
    
    def test_pivot_points(self):
        """
        Test hàm pivot_points.
        """
        # Test pivot_points với phương pháp standard
        result_df_standard = pivot_points(
            self.df, high_column='high', low_column='low', close_column='close',
            method='standard', prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        self.assertIn('test_pivot', result_df_standard.columns)
        self.assertIn('test_support1', result_df_standard.columns)
        self.assertIn('test_support2', result_df_standard.columns)
        self.assertIn('test_support3', result_df_standard.columns)
        self.assertIn('test_resistance1', result_df_standard.columns)
        self.assertIn('test_resistance2', result_df_standard.columns)
        self.assertIn('test_resistance3', result_df_standard.columns)
        
        # Tính pivot point standard thủ công và so sánh
        # Pivot = (High + Low + Close) / 3
        expected_pivot = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        pd.testing.assert_series_equal(
            result_df_standard['test_pivot'],
            expected_pivot,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        
        # Tính Support1 = (2 * Pivot) - High
        expected_s1 = (2 * expected_pivot) - self.df['high']
        pd.testing.assert_series_equal(
            result_df_standard['test_support1'],
            expected_s1,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        
        # Tính Resistance1 = (2 * Pivot) - Low
        expected_r1 = (2 * expected_pivot) - self.df['low']
        pd.testing.assert_series_equal(
            result_df_standard['test_resistance1'],
            expected_r1,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        
        # Test pivot_points với phương pháp fibonacci
        result_df_fib = pivot_points(
            self.df, high_column='high', low_column='low', close_column='close',
            method='fibonacci', prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        self.assertIn('test_pivot_fib', result_df_fib.columns)
        self.assertIn('test_support1_fib', result_df_fib.columns)
        
        # Test pivot_points với phương pháp woodie
        result_df_woodie = pivot_points(
            self.df, high_column='high', low_column='low', close_column='close',
            method='woodie', prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        self.assertIn('test_pivot_woodie', result_df_woodie.columns)
        
        # Test pivot_points với phương pháp camarilla
        result_df_camarilla = pivot_points(
            self.df, high_column='high', low_column='low', close_column='close',
            method='camarilla', prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        self.assertIn('test_resistance1_camarilla', result_df_camarilla.columns)
        
        # Test pivot_points với phương pháp demark
        result_df_demark = pivot_points(
            self.df, high_column='high', low_column='low', close_column='close',
            method='demark', prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        self.assertIn('test_pivot_demark', result_df_demark.columns)
        
        # Kiểm tra với phương pháp không hợp lệ
        with self.assertRaises(ValueError):
            pivot_points(self.df, method='invalid_method')
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            pivot_points(pd.DataFrame())
    
    def test_fibonacci_retracement(self):
        """
        Test hàm fibonacci_retracement.
        """
        window = 100
        
        # Test với trend='up'
        result_df_up = fibonacci_retracement(
            self.df, high_column='high', low_column='low',
            trend='up', window=window, prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        self.assertIn('test_fib_trend', result_df_up.columns)
        self.assertIn('test_fib_swing_high', result_df_up.columns)
        self.assertIn('test_fib_swing_low', result_df_up.columns)
        self.assertIn('test_fib_0', result_df_up.columns)
        self.assertIn('test_fib_236', result_df_up.columns)
        self.assertIn('test_fib_382', result_df_up.columns)
        self.assertIn('test_fib_5', result_df_up.columns)
        self.assertIn('test_fib_618', result_df_up.columns)
        self.assertIn('test_fib_786', result_df_up.columns)
        self.assertIn('test_fib_1', result_df_up.columns)
        self.assertIn('test_fib_ext_1618', result_df_up.columns)
        self.assertIn('test_fib_ext_2618', result_df_up.columns)
        
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
        
        # Test với trend='auto'
        result_df_auto = fibonacci_retracement(
            self.trending_df, high_column='high', low_column='low',
            trend='auto', window=window, prefix='test_'
        )
        
        # Kiểm tra trend được xác định tự động
        self.assertEqual(result_df_auto['test_fib_trend'].iloc[0], 'up')
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            fibonacci_retracement(pd.DataFrame(), high_column='high', low_column='low')
    
    def test_find_chart_patterns(self):
        """
        Test hàm find_chart_patterns.
        """
        window = 50
        tolerance = 0.03
        
        result_df = find_chart_patterns(
            self.df, price_column='close', 
            window=window, tolerance=tolerance,
            prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        self.assertIn('test_double_top', result_df.columns)
        self.assertIn('test_double_bottom', result_df.columns)
        self.assertIn('test_head_shoulders', result_df.columns)
        self.assertIn('test_inv_head_shoulders', result_df.columns)
        self.assertIn('test_triangle', result_df.columns)
        
        # Các giá trị của các cột mẫu hình nên là boolean
        for column in ['test_double_top', 'test_double_bottom', 'test_head_shoulders', 
                      'test_inv_head_shoulders', 'test_triangle']:
            self.assertTrue(pd.api.types.is_bool_dtype(result_df[column]))
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            find_chart_patterns(pd.DataFrame(), price_column='close', window=window)

if __name__ == '__main__':
    unittest.main()