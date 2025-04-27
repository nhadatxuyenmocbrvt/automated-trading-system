"""
Unit tests cho module volatility_indicators trong technical_indicators.
"""

import unittest
import logging
import numpy as np
import pandas as pd

from data_processors.feature_engineering.technical_indicators.volatility_indicators import (
    average_true_range, bollinger_bandwidth, keltner_channel,
    donchian_channel, ulcer_index, standard_deviation,
    historical_volatility, volatility_ratio
)

from tests.technical_indicators import logger
from tests.technical_indicators.test_data import (
    generate_sample_price_data, generate_volatile_price_data
)

class TestVolatilityIndicators(unittest.TestCase):
    """
    Test các hàm chỉ báo biến động trong volatility_indicators.py.
    """
    
    def setUp(self):
        """
        Chuẩn bị dữ liệu cho các bài test.
        """
        logger.info("Thiết lập dữ liệu cho test_volatility_indicators.py")
        self.df = generate_sample_price_data(n_samples=100)
        self.volatile_df = generate_volatile_price_data(n_samples=100, volatility=0.03)
    
    def test_average_true_range(self):
        """
        Test hàm average_true_range.
        """
        logger.info("Thực hiện test_average_true_range")
        
        window = 14
        
        # Test với method='ema'
        result_df_ema = average_true_range(self.df, window=window, method='ema', prefix='test_')
        
        # Kiểm tra các cột kết quả tồn tại
        self.assertIn(f'test_atr_{window}', result_df_ema.columns)
        self.assertIn(f'test_atr_pct_{window}', result_df_ema.columns)
        logger.debug(f"ATR columns 'test_atr_{window}' và 'test_atr_pct_{window}' tồn tại: OK")
        
        # Kiểm tra ATR luôn dương
        atr_values = result_df_ema[f'test_atr_{window}'].dropna()
        self.assertTrue(all(atr_values >= 0))
        logger.debug("ATR luôn dương: OK")
        
        # Kiểm tra ATR phần trăm nằm trong khoảng hợp lý
        atr_pct_values = result_df_ema[f'test_atr_pct_{window}'].dropna()
        self.assertTrue(all(atr_pct_values >= 0))
        self.assertTrue(all(atr_pct_values < 10))  # Thường dưới 10%
        logger.debug("ATR phần trăm nằm trong khoảng hợp lý: OK")
        
        # Test với method='sma'
        result_df_sma = average_true_range(self.df, window=window, method='sma', prefix='test_')
        self.assertIn(f'test_atr_{window}', result_df_sma.columns)
        logger.debug(f"ATR với method='sma' tồn tại: OK")
        
        # Test với dữ liệu biến động cao
        result_df_volatile = average_true_range(self.volatile_df, window=window, prefix='')
        atr_volatile = result_df_volatile[f'atr_{window}'].dropna()
        atr_normal = result_df_ema[f'test_atr_{window}'].dropna()
        
        # ATR của dữ liệu biến động cao thường cao hơn ATR của dữ liệu bình thường
        self.assertGreater(atr_volatile.mean(), atr_normal.mean())
        logger.debug("ATR của dữ liệu biến động cao > ATR của dữ liệu bình thường: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            average_true_range(pd.DataFrame(), window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_average_true_range")
    
    def test_bollinger_bandwidth(self):
        """
        Test hàm bollinger_bandwidth.
        """
        logger.info("Thực hiện test_bollinger_bandwidth")
        
        window = 20
        std_dev = 2.0
        
        result_df = bollinger_bandwidth(self.df, column='close', window=window, std_dev=std_dev, prefix='test_')
        
        # Kiểm tra các cột kết quả tồn tại
        self.assertIn(f'test_bbw_{window}', result_df.columns)
        self.assertIn(f'test_bbw_percentb_{window}', result_df.columns)
        logger.debug(f"Bollinger Bandwidth columns 'test_bbw_{window}' và 'test_bbw_percentb_{window}' tồn tại: OK")
        
        # Tính Bollinger Bandwidth thủ công và so sánh
        # Tính middle band (SMA của giá đóng cửa)
        middle_band = self.df['close'].rolling(window=window).mean()
        
        # Tính độ lệch chuẩn
        std = self.df['close'].rolling(window=window).std(ddof=0)
        
        # Tính upper và lower bands
        upper_band = middle_band + std_dev * std
        lower_band = middle_band - std_dev * std
        
        # Tính bandwidth = (Upper - Lower) / Middle
        expected_bandwidth = (upper_band - lower_band) / middle_band
        
        # So sánh bandwidth, bỏ qua giá trị NaN
        bandwidth = result_df[f'test_bbw_{window}'].dropna()
        expected = expected_bandwidth.dropna()
        pd.testing.assert_series_equal(
            bandwidth,
            expected,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("BBW = (Upper - Lower) / Middle: OK")
        
        # Kiểm tra %B nằm trong khoảng 0-1 cho giá trong băng Bollinger
        percentb = result_df[f'test_bbw_percentb_{window}'].dropna()
        # percentB có thể vượt quá khoảng [0, 1] khi giá vượt ra ngoài băng Bollinger
        # Phần lớn giá trị nên nằm trong khoảng này
        percent_in_range = ((percentb >= 0) & (percentb <= 1)).mean()
        self.assertGreater(percent_in_range, 0.7)  # Ít nhất 70% nằm trong khoảng
        logger.debug("%B phần lớn nằm trong khoảng 0-1: OK")
        
        # Test với dữ liệu biến động cao
        result_df_volatile = bollinger_bandwidth(self.volatile_df, column='close', window=window)
        bbw_volatile = result_df_volatile[f'bbw_{window}'].dropna()
        bbw_normal = result_df[f'test_bbw_{window}'].dropna()
        
        # Bandwidth của dữ liệu biến động cao thường lớn hơn bandwidth của dữ liệu bình thường
        self.assertGreater(bbw_volatile.mean(), bbw_normal.mean())
        logger.debug("BBW của dữ liệu biến động cao > BBW của dữ liệu bình thường: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            bollinger_bandwidth(pd.DataFrame(), column='close', window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_bollinger_bandwidth")
    
    def test_keltner_channel(self):
        """
        Test hàm keltner_channel.
        """
        logger.info("Thực hiện test_keltner_channel")
        
        window = 20
        atr_window = 10
        atr_multiplier = 2.0
        
        result_df = keltner_channel(
            self.df, window=window, atr_window=atr_window, 
            atr_multiplier=atr_multiplier, prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        kc_columns = [
            f'test_kc_middle_{window}', 
            f'test_kc_upper_{window}', 
            f'test_kc_lower_{window}',
            f'test_kc_width_{window}',
            f'test_kc_position_{window}'
        ]
        for col in kc_columns:
            self.assertIn(col, result_df.columns)
        logger.debug("Tất cả các cột Keltner Channel tồn tại: OK")
        
        # Kiểm tra middle line là EMA của close
        middle_line = result_df[f'test_kc_middle_{window}'].dropna()
        expected_middle = self.df['close'].ewm(span=window, adjust=False).mean().dropna()
        pd.testing.assert_series_equal(
            middle_line,
            expected_middle,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("Middle line = EMA của close: OK")
        
        # Kiểm tra position nằm trong khoảng 0-1 cho giá trong kênh Keltner
        position = result_df[f'test_kc_position_{window}'].dropna()
        
        # Giá trị position có thể vượt ra ngoài khoảng [0, 1] nếu giá vượt ra ngoài kênh
        # Nên chỉ kiểm tra rằng phần lớn giá trị nằm trong khoảng [0, 1]
        percent_in_range = ((position >= 0) & (position <= 1)).mean()
        self.assertGreater(percent_in_range, 0.7)  # Ít nhất 70% giá trị nằm trong khoảng
        logger.debug("Position phần lớn nằm trong khoảng 0-1: OK")
        
        # Test với dữ liệu biến động cao
        result_df_volatile = keltner_channel(
            self.volatile_df, window=window, atr_window=atr_window, atr_multiplier=atr_multiplier
        )
        kc_width_volatile = result_df_volatile[f'kc_width_{window}'].dropna()
        kc_width_normal = result_df[f'test_kc_width_{window}'].dropna()
        
        # Width của kênh Keltner cho dữ liệu biến động cao thường lớn hơn dữ liệu bình thường
        self.assertGreater(kc_width_volatile.mean(), kc_width_normal.mean())
        logger.debug("KC width của dữ liệu biến động cao > KC width của dữ liệu bình thường: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            keltner_channel(pd.DataFrame(), window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_keltner_channel")
    
    def test_donchian_channel(self):
        """
        Test hàm donchian_channel.
        """
        logger.info("Thực hiện test_donchian_channel")
        
        window = 20
        
        result_df = donchian_channel(self.df, window=window, prefix='test_')
        
        # Kiểm tra các cột kết quả tồn tại
        dc_columns = [
            f'test_dc_upper_{window}', 
            f'test_dc_lower_{window}', 
            f'test_dc_middle_{window}',
            f'test_dc_width_{window}',
            f'test_dc_position_{window}'
        ]
        for col in dc_columns:
            self.assertIn(col, result_df.columns)
        logger.debug("Tất cả các cột Donchian Channel tồn tại: OK")
        
        # Tính Donchian Channel thủ công và so sánh
        # Upper bound = Highest high trong cửa sổ
        expected_upper = self.df['high'].rolling(window=window).max()
        upper_bound = result_df[f'test_dc_upper_{window}']
        pd.testing.assert_series_equal(
            upper_bound.dropna(),
            expected_upper.dropna(),
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("Upper bound = Highest high trong cửa sổ: OK")
        
        # Lower bound = Lowest low trong cửa sổ
        expected_lower = self.df['low'].rolling(window=window).min()
        lower_bound = result_df[f'test_dc_lower_{window}']
        pd.testing.assert_series_equal(
            lower_bound.dropna(),
            expected_lower.dropna(),
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("Lower bound = Lowest low trong cửa sổ: OK")
        
        # Middle line = (Upper + Lower) / 2
        expected_middle = (expected_upper + expected_lower) / 2
        middle_line = result_df[f'test_dc_middle_{window}']
        pd.testing.assert_series_equal(
            middle_line.dropna(),
            expected_middle.dropna(),
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("Middle line = (Upper + Lower) / 2: OK")
        
        # Test với dữ liệu biến động cao
        result_df_volatile = donchian_channel(self.volatile_df, window=window)
        dc_width_volatile = result_df_volatile[f'dc_width_{window}'].dropna()
        dc_width_normal = result_df[f'test_dc_width_{window}'].dropna()
        
        # Width của kênh Donchian cho dữ liệu biến động cao thường lớn hơn dữ liệu bình thường
        self.assertGreater(dc_width_volatile.mean(), dc_width_normal.mean())
        logger.debug("DC width của dữ liệu biến động cao > DC width của dữ liệu bình thường: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            donchian_channel(pd.DataFrame(), window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_donchian_channel")
    
    def test_ulcer_index(self):
        """
        Test hàm ulcer_index.
        """
        logger.info("Thực hiện test_ulcer_index")
        
        window = 14
        
        result_df = ulcer_index(self.df, column='close', window=window, prefix='test_')
        
        # Kiểm tra cột kết quả tồn tại
        expected_column = f'test_ui_{window}'
        self.assertIn(expected_column, result_df.columns)
        logger.debug(f"Ulcer Index column '{expected_column}' tồn tại: OK")
        
        # Kiểm tra Ulcer Index luôn dương
        ui_values = result_df[expected_column].dropna()
        self.assertTrue(all(ui_values >= 0))
        logger.debug("Ulcer Index luôn dương: OK")
        
        # Test với dữ liệu biến động cao
        result_df_volatile = ulcer_index(self.volatile_df, column='close', window=window)
        ui_volatile = result_df_volatile[f'ui_{window}'].dropna()
        ui_normal = result_df[expected_column].dropna()
        
        # Ulcer Index của dữ liệu biến động cao thường cao hơn dữ liệu bình thường
        self.assertGreater(ui_volatile.mean(), ui_normal.mean())
        logger.debug("UI của dữ liệu biến động cao > UI của dữ liệu bình thường: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            ulcer_index(pd.DataFrame(), column='close', window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_ulcer_index")
    
    def test_standard_deviation(self):
        """
        Test hàm standard_deviation.
        """
        logger.info("Thực hiện test_standard_deviation")
        
        window = 20
        trading_periods = 252
        
        result_df = standard_deviation(
            self.df, column='close', window=window, 
            trading_periods=trading_periods, prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        self.assertIn(f'test_stddev_{window}', result_df.columns)
        self.assertIn(f'test_annvol_{window}', result_df.columns)
        logger.debug(f"Standard Deviation columns 'test_stddev_{window}' và 'test_annvol_{window}' tồn tại: OK")
        
        # Tính Standard Deviation thủ công và so sánh
        returns = self.df['close'].pct_change()
        expected_stddev = returns.rolling(window=window).std()
        
        # So sánh StdDev, bỏ qua giá trị NaN
        stddev = result_df[f'test_stddev_{window}'].dropna()
        expected = expected_stddev.dropna()
        pd.testing.assert_series_equal(
            stddev,
            expected,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("StdDev = Standard Deviation of Returns: OK")
        
        # Tính Annualized Volatility thủ công và so sánh
        expected_annvol = expected_stddev * np.sqrt(trading_periods)
        
        # So sánh Annualized Volatility, bỏ qua giá trị NaN
        annvol = result_df[f'test_annvol_{window}'].dropna()
        expected = expected_annvol.dropna()
        pd.testing.assert_series_equal(
            annvol,
            expected,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("Annualized Vol = StdDev * sqrt(trading_periods): OK")
        
        # Test với dữ liệu biến động cao
        result_df_volatile = standard_deviation(
            self.volatile_df, column='close', window=window, trading_periods=trading_periods
        )
        annvol_volatile = result_df_volatile[f'annvol_{window}'].dropna()
        annvol_normal = result_df[f'test_annvol_{window}'].dropna()
        
        # Annualized Volatility của dữ liệu biến động cao thường lớn hơn dữ liệu bình thường
        self.assertGreater(annvol_volatile.mean(), annvol_normal.mean())
        logger.debug("AnnVol của dữ liệu biến động cao > AnnVol của dữ liệu bình thường: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            standard_deviation(pd.DataFrame(), column='close', window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_standard_deviation")
    
    def test_historical_volatility(self):
        """
        Test hàm historical_volatility.
        """
        logger.info("Thực hiện test_historical_volatility")
        
        window = 20
        trading_periods = 252
        
        result_df = historical_volatility(
            self.df, column='close', window=window, 
            trading_periods=trading_periods, prefix='test_'
        )
        
        # Kiểm tra các cột kết quả tồn tại
        self.assertIn(f'test_hvol_{window}', result_df.columns)
        self.assertIn(f'test_hvol_pct_{window}', result_df.columns)
        logger.debug(f"Historical Volatility columns 'test_hvol_{window}' và 'test_hvol_pct_{window}' tồn tại: OK")
        
        # Tính Historical Volatility thủ công và so sánh
        log_returns = np.log(self.df['close'] / self.df['close'].shift(1))
        expected_rolling_std = log_returns.rolling(window=window).std()
        expected_hvol = expected_rolling_std * np.sqrt(trading_periods)
        
        # So sánh Historical Volatility, bỏ qua giá trị NaN
        hvol = result_df[f'test_hvol_{window}'].dropna()
        expected = expected_hvol.dropna()
        pd.testing.assert_series_equal(
            hvol,
            expected,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("HVol = StdDev(Log Returns) * sqrt(trading_periods): OK")
        
        # Kiểm tra Historical Volatility Percentage = HVol * 100
        hvol_pct = result_df[f'test_hvol_pct_{window}'].dropna()
        expected_pct = expected_hvol.dropna() * 100
        pd.testing.assert_series_equal(
            hvol_pct,
            expected_pct,
            check_dtype=False  # Bỏ qua kiểm tra kiểu dữ liệu
        )
        logger.debug("HVol % = HVol * 100: OK")
        
        # Test với dữ liệu biến động cao
        result_df_volatile = historical_volatility(
            self.volatile_df, column='close', window=window, trading_periods=trading_periods
        )
        hvol_volatile = result_df_volatile[f'hvol_{window}'].dropna()
        hvol_normal = result_df[f'test_hvol_{window}'].dropna()
        
        # Historical Volatility của dữ liệu biến động cao thường lớn hơn dữ liệu bình thường
        self.assertGreater(hvol_volatile.mean(), hvol_normal.mean())
        logger.debug("HVol của dữ liệu biến động cao > HVol của dữ liệu bình thường: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            historical_volatility(pd.DataFrame(), column='close', window=window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_historical_volatility")
    
    def test_volatility_ratio(self):
        """
        Test hàm volatility_ratio.
        """
        logger.info("Thực hiện test_volatility_ratio")
        
        short_window = 5
        long_window = 20
        
        result_df = volatility_ratio(
            self.df, column='close', 
            short_window=short_window, long_window=long_window,
            prefix='test_'
        )
        
        # Kiểm tra cột kết quả tồn tại
        expected_column = f'test_vol_ratio_{short_window}_{long_window}'
        self.assertIn(expected_column, result_df.columns)
        logger.debug(f"Volatility Ratio column '{expected_column}' tồn tại: OK")
        
        # Tính Volatility Ratio thủ công và so sánh
        returns = self.df['close'].pct_change()
        short_vol = returns.rolling(window=short_window).std()
        long_vol = returns.rolling(window=long_window).std()
        
        expected_vol_ratio = short_vol / long_vol
        
        # So sánh Volatility Ratio, bỏ qua giá trị NaN và vô cùng
        vol_ratio = result_df[expected_column].replace([np.inf, -np.inf], np.nan).dropna()
        expected = expected_vol_ratio.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Có thể có sự khác biệt nhỏ do làm tròn, vì vậy kiểm tra giá trị gần đúng
        pd.testing.assert_series_equal(
            vol_ratio,
            expected,
            check_dtype=False,  # Bỏ qua kiểm tra kiểu dữ liệu
            atol=1e-10  # Dung sai tuyệt đối
        )
        logger.debug("Vol Ratio = Short-term Vol / Long-term Vol: OK")
        
        # Kiểm tra khi sử dụng DataFrame không hợp lệ
        with self.assertRaises(ValueError):
            volatility_ratio(pd.DataFrame(), column='close',
                            short_window=short_window, long_window=long_window)
        logger.debug("Kiểm tra xử lý lỗi với dữ liệu không hợp lệ: OK")
        
        logger.debug("Hoàn thành test_volatility_ratio")
    
    def tearDown(self):
        """
        Dọn dẹp sau khi test hoàn thành.
        """
        logger.info("Đã hoàn thành test_volatility_indicators.py")

if __name__ == '__main__':
    # Thiết lập logging khi chạy trực tiếp
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    unittest.main()