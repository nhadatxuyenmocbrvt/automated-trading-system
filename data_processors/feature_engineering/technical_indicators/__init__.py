"""
Module chỉ báo kỹ thuật.
Cung cấp nhiều chỉ báo kỹ thuật để phân tích dữ liệu thị trường.
"""

# Import để cung cấp API ngắn gọn cho người dùng
from data_processors.feature_engineering.technical_indicators.trend_indicators import (
    simple_moving_average, exponential_moving_average, bollinger_bands, 
    moving_average_convergence_divergence, average_directional_index,
    parabolic_sar, ichimoku_cloud, supertrend
)

from data_processors.feature_engineering.technical_indicators.momentum_indicators import (
    relative_strength_index, stochastic_oscillator, commodity_channel_index,
    williams_r, rate_of_change, money_flow_index, true_strength_index
)

from data_processors.feature_engineering.technical_indicators.volume_indicators import (
    on_balance_volume, accumulation_distribution_line, chaikin_money_flow,
    money_flow_index, volume_weighted_average_price, ease_of_movement,
    volume_oscillator, price_volume_trend, accumulated_distribution_volume
)

from data_processors.feature_engineering.technical_indicators.volatility_indicators import (
    average_true_range, bollinger_bandwidth, keltner_channel,
    donchian_channel, ulcer_index, standard_deviation, historical_volatility,
    volatility_ratio
)

from data_processors.feature_engineering.technical_indicators.support_resistance import (
    detect_support_resistance, pivot_points, fibonacci_retracement
)

# Hàm tiện ích hữu ích
from data_processors.feature_engineering.technical_indicators.utils import (
    crossover, crossunder, normalize_indicator, true_range,
    find_local_extrema, prepare_price_data
)

# Định nghĩa các alias ngắn gọn cho các hàm phổ biến
sma = simple_moving_average
ema = exponential_moving_average
macd = moving_average_convergence_divergence
rsi = relative_strength_index
bbands = bollinger_bands
adx = average_directional_index
psar = parabolic_sar
atr = average_true_range
obv = on_balance_volume
stoch = stochastic_oscillator
cci = commodity_channel_index
mfi = money_flow_index
vwap = volume_weighted_average_price
sr = detect_support_resistance
st = supertrend
vo = volume_oscillator
pvt = price_volume_trend
adv = accumulated_distribution_volume
bb_width = bollinger_bandwidth
hv = historical_volatility
kc = keltner_channel
dc = donchian_channel

__all__ = [
    # Trend
    'simple_moving_average', 'exponential_moving_average', 'bollinger_bands',
    'moving_average_convergence_divergence', 'average_directional_index',
    'parabolic_sar', 'ichimoku_cloud', 'supertrend',
    
    # Momentum
    'relative_strength_index', 'stochastic_oscillator', 'commodity_channel_index',
    'williams_r', 'rate_of_change', 'money_flow_index', 'true_strength_index',
    
    # Volume
    'on_balance_volume', 'accumulation_distribution_line', 'chaikin_money_flow',
    'money_flow_index', 'volume_weighted_average_price', 'ease_of_movement',
    'volume_oscillator', 'price_volume_trend', 'accumulated_distribution_volume',
    
    # Volatility
    'average_true_range', 'bollinger_bandwidth', 'keltner_channel',
    'donchian_channel', 'ulcer_index', 'standard_deviation',
    'historical_volatility', 'volatility_ratio',
    
    # Support/Resistance
    'detect_support_resistance', 'pivot_points', 'fibonacci_retracement',
    
    # Utils
    'crossover', 'crossunder', 'normalize_indicator', 'true_range',
    'find_local_extrema', 'prepare_price_data',
    
    # Aliases
    'sma', 'ema', 'macd', 'rsi', 'bbands', 'adx', 'psar', 'atr', 
    'obv', 'stoch', 'cci', 'mfi', 'vwap', 'sr', 'st', 'vo', 'pvt',
    'adv', 'bb_width', 'hv', 'kc', 'dc'
]