"""
Module đặc trưng thị trường.
Mô-đun này cung cấp các đặc trưng chuyên biệt dựa trên dữ liệu thị trường,
bao gồm đặc trưng giá cả, khối lượng, biến động, thanh khoản và sổ lệnh.
"""

# Import các module con
from data_processors.feature_engineering.market_features.price_features import (
    calculate_returns, calculate_log_returns, calculate_rsi_features,
    calculate_price_momentum, calculate_price_ratios, calculate_price_channels,
    calculate_price_crossovers, calculate_price_divergence
)

from data_processors.feature_engineering.market_features.volatility_features import (
    calculate_volatility_features, calculate_relative_volatility,
    calculate_volatility_ratio, calculate_volatility_patterns,
    calculate_volatility_regime, calculate_garch_features
)

from data_processors.feature_engineering.market_features.volume_features import (
    calculate_volume_features, calculate_relative_volume,
    calculate_volume_price_correlation, calculate_volume_oscillations,
    calculate_obv_features, calculate_vwap_features
)

from data_processors.feature_engineering.market_features.orderbook_features import (
    calculate_orderbook_imbalance, calculate_orderbook_features,
    calculate_market_pressure, calculate_order_flow, calculate_liquidity_distribution,
    calculate_support_resistance_levels
)

from data_processors.feature_engineering.market_features.liquidity_features import (
    calculate_liquidity_features, calculate_amihud_illiquidity,
    calculate_market_depth, calculate_spread_analysis,
    calculate_market_impact, calculate_slippage_estimation
)

from data_processors.feature_engineering.market_features.custom_features import (
    calculate_mean_reversion_features, calculate_trend_strength_features,
    calculate_price_pattern_features, calculate_market_regime_features,
    calculate_event_impact_features, calculate_correlation_features
)

# Danh sách các hàm quan trọng để export
__all__ = [
    # Price features
    'calculate_returns', 'calculate_log_returns', 'calculate_rsi_features',
    'calculate_price_momentum', 'calculate_price_ratios', 'calculate_price_channels',
    'calculate_price_crossovers', 'calculate_price_divergence',
    
    # Volatility features
    'calculate_volatility_features', 'calculate_relative_volatility',
    'calculate_volatility_ratio', 'calculate_volatility_patterns',
    'calculate_volatility_regime', 'calculate_garch_features',
    
    # Volume features
    'calculate_volume_features', 'calculate_relative_volume',
    'calculate_volume_price_correlation', 'calculate_volume_oscillations',
    'calculate_obv_features', 'calculate_vwap_features',
    
    # Orderbook features
    'calculate_orderbook_imbalance', 'calculate_orderbook_features',
    'calculate_market_pressure', 'calculate_order_flow', 'calculate_liquidity_distribution',
    'calculate_support_resistance_levels',
    
    # Liquidity features
    'calculate_liquidity_features', 'calculate_amihud_illiquidity',
    'calculate_market_depth', 'calculate_spread_analysis',
    'calculate_market_impact', 'calculate_slippage_estimation',
    
    # Custom features
    'calculate_mean_reversion_features', 'calculate_trend_strength_features',
    'calculate_price_pattern_features', 'calculate_market_regime_features',
    'calculate_event_impact_features', 'calculate_correlation_features'
]