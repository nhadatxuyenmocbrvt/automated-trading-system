"""
Unit tests cho module volume_indicators trong technical_indicators.
"""

import unittest
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data_processors.feature_engineering.technical_indicators.volume_indicators import (
    on_balance_volume, accumulation_distribution_line, chaikin_money_flow,
    volume_weighted_average_price, ease_of_movement, volume_oscillator,
    money_flow_index, price_volume_trend
)

from tests.technical_indicators import logger
from tests.technical_indicators.test_data import (
    generate_sample_price_data, generate_trending_price_data
)

class TestVolumeIndicators(unittest.TestCase):
    """
    Test các hàm chỉ báo khối lượng trong volume_indicators.py.
    """
    
    def setUp(self):
        """
        Chuẩn bị dữ liệu cho các bài test.
        """
        logger.info("Thiết lập dữ liệu cho test_volume_indicators.py")
        self.df = generate_sample_price_data(n_samples=100)
        self.trending_df = generate_trending_price_data(n_samples=100, trend_strength=0.01)
    
    def test_on_balance_volume(self):
        """
        Test hàm on_balance_volume.
        """
        logger.info("Thực hiện test_on_balance_volume")
        
        result_df = on_balance_volume(self.df, close_column='close', volume_column='volume', prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column = 'test_obv'
        self.assertIn(expected_column, result_df.columns)
        logger.debug(f"OBV column '{expected_column}' tồn tại: OK")
        
        # Tính OBV thủ công và so sánh
        # OBV = OBV trước + (Volume * Direction)
        close_diff = self.df['close'].diff()
        obv_direction = pd.Series(0, index=self.df.index)
        obv_direction.loc[close_diff > 0] = 1
        obv_direction.loc[close_diff < 0] = -1
        
        expected_obv = (self.df['volume'] * obv_direction).cumsum()
        pd.testing.assert_series_equal(
            result_df[expected_column],
            expected_obv,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("OBV = Σ(Volume * Direction): OK")
        
        # Kiểm tra với dữ liệu xu hướng tăng
        result_trending_df = on_balance_volume(self.trending_df)
        # Với dữ liệu xu hướng tăng, OBV thường tăng lên
        obv_trending = result_trending_df['obv']
        # So sánh OBV cuối cùng với OBV đầu tiên trong phần có giá trị (không phải NaN)
        self.assertGreater(obv_trending.iloc[-1], obv_trending.iloc[1])
        logger.debug("Với dữ liệu xu hướng tăng, OBV tăng lên: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            on_balance_volume(pd.DataFrame())
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_on_balance_volume")
    
    def test_accumulation_distribution_line(self):
        """
        Test hàm accumulation_distribution_line.
        """
        logger.info("Thực hiện test_accumulation_distribution_line")
        
        result_df = accumulation_distribution_line(self.df, prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column = 'test_ad_line'
        self.assertIn(expected_column, result_df.columns)
        logger.debug(f"A/D Line column '{expected_column}' tồn tại: OK")
        
        # Tính A/D Line thủ công và so sánh
        # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        high_low_diff = self.df['high'] - self.df['low']
        
        # Tránh chia cho 0
        high_low_diff = high_low_diff.replace(0, np.nan)
        
        money_flow_multiplier = ((self.df['close'] - self.df['low']) - 
                                (self.df['high'] - self.df['close'])) / high_low_diff
        
        # Money Flow Volume = Money Flow Multiplier * Volume
        money_flow_volume = money_flow_multiplier * self.df['volume']
        
        # A/D Line = Previous A/D Line + Money Flow Volume
        expected_ad_line = money_flow_volume.cumsum()
        
        # So sánh A/D Line, bỏ qua giá trị NaN
        ad_line = result_df[expected_column].dropna()
        expected = expected_ad_line.dropna()
        pd.testing.assert_series_equal(
            ad_line,
            expected,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("A/D Line = Σ(Money Flow Volume): OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            accumulation_distribution_line(pd.DataFrame())
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_accumulation_distribution_line")
    
    def test_chaikin_money_flow(self):
        """
        Test hàm chaikin_money_flow.
        """
        logger.info("Thực hiện test_chaikin_money_flow")
        
        window = 20
        result_df = chaikin_money_flow(self.df, window=window, prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column = f'test_cmf_{window}'
        self.assertIn(expected_column, result_df.columns)
        logger.debug(f"CMF column '{expected_column}' tồn tại: OK")
        
        # Tính CMF thủ công và so sánh
        # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        high_low_diff = self.df['high'] - self.df['low']
        
        # Tránh chia cho 0
        high_low_diff = high_low_diff.replace(0, np.nan)
        
        money_flow_multiplier = ((self.df['close'] - self.df['low']) - 
                                (self.df['high'] - self.df['close'])) / high_low_diff
        
        # Money Flow Volume = Money Flow Multiplier * Volume
        money_flow_volume = money_flow_multiplier * self.df['volume']
        
        # CMF = Sum(Money Flow Volume, n) / Sum(Volume, n)
        sum_money_flow_volume = money_flow_volume.rolling(window=window).sum()
        sum_volume = self.df['volume'].rolling(window=window).sum()
        
        expected_cmf = sum_money_flow_volume / sum_volume
        
        # So sánh CMF, bỏ qua giá trị NaN
        cmf = result_df[expected_column].dropna()
        expected = expected_cmf.dropna()
        pd.testing.assert_series_equal(
            cmf,
            expected,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("CMF = Sum(Money Flow Volume, n) / Sum(Volume, n): OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            chaikin_money_flow(pd.DataFrame(), window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_chaikin_money_flow")
    
    def test_volume_weighted_average_price(self):
        """
        Test hàm volume_weighted_average_price.
        """
        logger.info("Thực hiện test_volume_weighted_average_price")
        
        # Test với window=None (VWAP từ đầu dữ liệu)
        result_df = volume_weighted_average_price(self.df, window=None, prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column = 'test_vwap'
        self.assertIn(expected_column, result_df.columns)
        logger.debug(f"VWAP column '{expected_column}' tồn tại: OK")
        
        # Tính VWAP thủ công và so sánh
        # Typical Price = (High + Low + Close) / 3
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        # Volume * Typical Price
        vol_price = typical_price * self.df['volume']
        
        # VWAP = Cumulative(Volume * Typical Price) / Cumulative(Volume)
        cumsum_vol_price = vol_price.cumsum()
        cumsum_vol = self.df['volume'].cumsum()
        
        expected_vwap = cumsum_vol_price / cumsum_vol
        
        # So sánh VWAP, bỏ qua giá trị NaN
        vwap = result_df[expected_column].dropna()
        expected = expected_vwap.dropna()
        pd.testing.assert_series_equal(
            vwap,
            expected,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("VWAP = Cum(Volume * Typical Price) / Cum(Volume): OK")
        
        # Test với window chỉ định
        window = 20
        result_df_window = volume_weighted_average_price(self.df, window=window, prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column_window = f'test_vwap_{window}'
        self.assertIn(expected_column_window, result_df_window.columns)
        logger.debug(f"VWAP với window={window} column '{expected_column_window}' tồn tại: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            volume_weighted_average_price(pd.DataFrame())
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_volume_weighted_average_price")
    
    def test_ease_of_movement(self):
        """
        Test hàm ease_of_movement.
        """
        logger.info("Thực hiện test_ease_of_movement")
        
        window = 14
        divisor = 10000
        result_df = ease_of_movement(self.df, window=window, divisor=divisor, prefix='test_')
        
        # Kiểm tra các cột kết quả tồn tại
        self.assertIn('test_eom', result_df.columns)
        self.assertIn(f'test_eom_ma_{window}', result_df.columns)
        logger.debug(f"EOM columns 'test_eom' và 'test_eom_ma_{window}' tồn tại: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            ease_of_movement(pd.DataFrame(), window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_ease_of_movement")
    
    def test_volume_oscillator(self):
        """
        Test hàm volume_oscillator.
        """
        logger.info("Thực hiện test_volume_oscillator")
        
        short_window = 5
        long_window = 10
        
        # Test với percentage=True
        result_df_pct = volume_oscillator(
            self.df, volume_column='volume', 
            short_window=short_window, long_window=long_window,
            percentage=True, prefix='test_'
        )
        
        # Kiểm tra cột kết quả tồn tại
        expected_column_pct = f'test_vo_pct_{short_window}_{long_window}'
        self.assertIn(expected_column_pct, result_df_pct.columns)
        logger.debug(f"VO percentage column '{expected_column_pct}' tồn tại: OK")
        
        # Tính Volume Oscillator Percentage thủ công và so sánh
        short_vol_sma = self.df['volume'].rolling(window=short_window).mean()
        long_vol_sma = self.df['volume'].rolling(window=long_window).mean()
        
        expected_vo_pct = ((short_vol_sma - long_vol_sma) / long_vol_sma) * 100
        
        # So sánh VO Percentage, bỏ qua giá trị NaN
        vo_pct = result_df_pct[expected_column_pct].dropna()
        expected = expected_vo_pct.dropna()
        pd.testing.assert_series_equal(
            vo_pct,
            expected,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("VO % = ((Short SMA - Long SMA) / Long SMA) * 100: OK")
        
        # Test với percentage=False
        result_df_abs = volume_oscillator(
            self.df, volume_column='volume', 
            short_window=short_window, long_window=long_window,
            percentage=False, prefix='test_'
        )
        
        # Kiểm tra cột kết quả tồn tại
        expected_column_abs = f'test_vo_{short_window}_{long_window}'
        self.assertIn(expected_column_abs, result_df_abs.columns)
        logger.debug(f"VO absolute column '{expected_column_abs}' tồn tại: OK")
        
        # Tính Volume Oscillator Absolute thủ công và so sánh
        expected_vo_abs = short_vol_sma - long_vol_sma
        
        # So sánh VO Absolute, bỏ qua giá trị NaN
        vo_abs = result_df_abs[expected_column_abs].dropna()
        expected = expected_vo_abs.dropna()
        pd.testing.assert_series_equal(
            vo_abs,
            expected,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("VO absolute = Short SMA - Long SMA: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            volume_oscillator(pd.DataFrame(), volume_column='volume',
                             short_window=short_window, long_window=long_window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_volume_oscillator")
    
    def test_price_volume_trend(self):
        """
        Test hàm price_volume_trend.
        """
        logger.info("Thực hiện test_price_volume_trend")
        
        result_df = price_volume_trend(self.df, close_column='close', volume_column='volume', prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column = 'test_pvt'
        self.assertIn(expected_column, result_df.columns)
        logger.debug(f"PVT column '{expected_column}' tồn tại: OK")
        
        # Tính PVT thủ công và so sánh
        close_pct_change = self.df['close'].pct_change()
        pvt_step = self.df['volume'] * close_pct_change
        expected_pvt = pvt_step.cumsum()
        
        # So sánh PVT, bỏ qua giá trị NaN
        pvt = result_df[expected_column].dropna()
        expected = expected_pvt.dropna()
        pd.testing.assert_series_equal(
            pvt,
            expected,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("PVT = Σ(Volume * Price % Change): OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            price_volume_trend(pd.DataFrame(), close_column='close', volume_column='volume')
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_price_volume_trend")
    
    def tearDown(self):
        """
        Dọn dẹp sau khi test hoàn thành.
        """
        logger.info("Đã hoàn thành test_volume_indicators.py")

if __name__ == '__main__':
    # Thiết lập logging khi chạy trực tiếp
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    unittest.main()