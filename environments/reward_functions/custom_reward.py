"""
Hàm phần thưởng tùy chỉnh.
File này cung cấp các hàm tính toán phần thưởng tùy chỉnh,
kết hợp nhiều yếu tố như lợi nhuận, rủi ro, và các chỉ số khác.
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional
import pandas as pd
import logging

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import PositionSide

# Thiết lập logger
logger = get_logger("reward_functions")

def calculate_custom_reward(
    navs: List[float],
    balances: List[float],
    positions: List[List[Dict[str, Any]]],
    current_pnl: float,
    performance_metrics: Optional[Dict[str, Any]] = None,
    profit_weight: float = 0.6,
    risk_weight: float = 0.3,
    drawdown_penalty_weight: float = 0.1,
    win_rate_bonus_weight: float = 0.1,
    holding_time_penalty: float = 0.001,
    **kwargs
) -> float:
    """
    Tính toán phần thưởng tùy chỉnh kết hợp nhiều yếu tố.
    
    Args:
        navs: Danh sách NAV (Net Asset Value) theo từng bước
        balances: Danh sách số dư theo từng bước
        positions: Danh sách vị thế theo từng bước
        current_pnl: P&L (Profit and Loss) hiện tại
        performance_metrics: Các chỉ số hiệu suất bổ sung
        profit_weight: Trọng số cho thành phần lợi nhuận
        risk_weight: Trọng số cho thành phần rủi ro
        drawdown_penalty_weight: Trọng số cho phạt drawdown
        win_rate_bonus_weight: Trọng số cho thưởng tỷ lệ thắng
        holding_time_penalty: Phạt cho thời gian nắm giữ vị thế dài
        **kwargs: Các tham số khác
        
    Returns:
        Giá trị phần thưởng tùy chỉnh
    """
    # Kiểm tra tham số
    if len(navs) < 2:
        return 0.0
    
    # Thành phần lợi nhuận - dựa trên thay đổi NAV
    prev_nav = navs[-2]
    current_nav = navs[-1]
    
    if prev_nav <= 0:
        logger.warning("Giá trị NAV trước đó không hợp lệ (≤ 0)")
        return 0.0
    
    profit_component = (current_nav - prev_nav) / prev_nav
    
    # Thành phần rủi ro - dựa trên độ biến động
    risk_component = 0.0
    if len(navs) > 5:
        # Tính toán lợi tức trong 5 bước gần nhất
        recent_returns = []
        for i in range(1, min(6, len(navs))):
            if navs[-i-1] > 0:
                recent_returns.append((navs[-i] - navs[-i-1]) / navs[-i-1])
        
        if recent_returns:
            # Độ lệch chuẩn của lợi tức gần đây
            volatility = np.std(recent_returns)
            
            # Phạt độ biến động cao
            risk_component = -volatility
    
    # Phạt cho drawdown - khuyến khích ổn định
    drawdown_penalty = 0.0
    if performance_metrics and 'max_drawdown' in performance_metrics:
        max_drawdown = performance_metrics['max_drawdown']
        drawdown_penalty = -max_drawdown
    
    # Thưởng cho tỷ lệ thắng cao
    win_rate_bonus = 0.0
    if performance_metrics and 'win_count' in performance_metrics and 'trade_count' in performance_metrics:
        win_count = performance_metrics['win_count']
        trade_count = performance_metrics['trade_count']
        
        if trade_count > 0:
            win_rate = win_count / trade_count
            # Thưởng phi tuyến cho tỷ lệ thắng
            win_rate_bonus = np.tanh(win_rate * 2 - 1)  # Âm khi < 0.5, dương khi > 0.5
    
    # Phạt cho việc nắm giữ vị thế quá lâu
    holding_time_component = 0.0
    current_positions = positions[-1] if positions else []
    
    if current_positions:
        total_holding_time = 0
        for pos in current_positions:
            if 'entry_index' in pos:
                current_idx = len(navs) - 1
                entry_idx = pos['entry_index']
                holding_time = current_idx - entry_idx
                total_holding_time += holding_time
        
        # Phạt thời gian nắm giữ dài
        holding_time_component = -holding_time_penalty * total_holding_time
    
    # Tính toán phần thưởng cuối cùng
    reward = (
        profit_weight * profit_component +
        risk_weight * risk_component +
        drawdown_penalty_weight * drawdown_penalty +
        win_rate_bonus_weight * win_rate_bonus +
        holding_time_component
    )
    
    return reward

def calculate_position_based_reward(
    navs: List[float],
    balances: List[float],
    positions: List[List[Dict[str, Any]]],
    current_pnl: float,
    current_price: float,
    atr_value: Optional[float] = None,
    trend_direction: Optional[int] = None,
    target_ratio: float = 2.0,  # Target profit / stop loss ratio
    position_size_penalty: float = 0.01,  # Penalty for excessive position size
    **kwargs
) -> float:
    """
    Tính toán phần thưởng dựa trên các quyết định vị thế và quan hệ với xu hướng.
    
    Args:
        navs: Danh sách NAV (Net Asset Value) theo từng bước
        balances: Danh sách số dư theo từng bước
        positions: Danh sách vị thế theo từng bước
        current_pnl: P&L (Profit and Loss) hiện tại
        current_price: Giá hiện tại
        atr_value: Giá trị Average True Range (đơn vị biến động trung bình)
        trend_direction: Hướng xu hướng (1 = lên, -1 = xuống, 0 = đi ngang)
        target_ratio: Tỷ lệ mục tiêu lợi nhuận / dừng lỗ
        position_size_penalty: Phạt cho kích thước vị thế quá lớn
        **kwargs: Các tham số khác
        
    Returns:
        Giá trị phần thưởng dựa trên vị thế
    """
    # Thành phần lợi nhuận cơ bản
    if len(navs) < 2:
        return 0.0
    
    prev_nav = navs[-2]
    current_nav = navs[-1]
    
    if prev_nav <= 0:
        logger.warning("Giá trị NAV trước đó không hợp lệ (≤ 0)")
        return 0.0
    
    # Tính lợi tức
    profit_component = (current_nav - prev_nav) / prev_nav
    
    # Thành phần vị thế
    position_component = 0.0
    current_positions = positions[-1] if positions else []
    
    # Thưởng/phạt dựa trên vị thế và xu hướng
    if current_positions and trend_direction is not None:
        for pos in current_positions:
            # Kiểm tra xem vị thế có phù hợp với xu hướng không
            position_side = 1 if pos.get('side') == PositionSide.LONG.value else -1
            
            # Thưởng nếu vị thế cùng chiều với xu hướng
            if position_side == trend_direction:
                position_component += 0.01
            else:
                position_component -= 0.01
            
            # Phạt nếu kích thước vị thế quá lớn so với số dư
            if 'size' in pos and balances and balances[-1] > 0:
                size_ratio = pos['size'] / balances[-1]
                if size_ratio > 0.5:  # Vị thế > 50% số dư
                    position_component -= position_size_penalty * (size_ratio - 0.5)
    
    # Thành phần quản lý rủi ro
    risk_management_component = 0.0
    if atr_value is not None and atr_value > 0 and current_positions:
        for pos in current_positions:
            if 'entry_price' in pos and 'stoploss' in pos and pos['stoploss'] is not None:
                entry_price = pos['entry_price']
                stoploss = pos['stoploss']
                
                # Tính khoảng cách dừng lỗ (đơn vị ATR)
                if pos.get('side') == PositionSide.LONG.value:
                    stop_distance = (entry_price - stoploss) / atr_value
                else:
                    stop_distance = (stoploss - entry_price) / atr_value
                
                # Thưởng nếu dừng lỗ hợp lý (1-3 ATR)
                if 1.0 <= stop_distance <= 3.0:
                    risk_management_component += 0.01
                else:
                    risk_management_component -= 0.01
                
                # Kiểm tra tỷ lệ mục tiêu lợi nhuận / dừng lỗ
                if 'takeprofit' in pos and pos['takeprofit'] is not None:
                    takeprofit = pos['takeprofit']
                    
                    if pos.get('side') == PositionSide.LONG.value:
                        tp_distance = (takeprofit - entry_price) / atr_value
                        sl_distance = (entry_price - stoploss) / atr_value
                    else:
                        tp_distance = (entry_price - takeprofit) / atr_value
                        sl_distance = (stoploss - entry_price) / atr_value
                    
                    if sl_distance > 0:
                        actual_ratio = tp_distance / sl_distance
                        
                        # Thưởng nếu tỷ lệ gần với target_ratio
                        ratio_diff = abs(actual_ratio - target_ratio)
                        if ratio_diff < 0.5:
                            risk_management_component += 0.02
    
    # Tính toán phần thưởng cuối cùng
    reward = profit_component + position_component + risk_management_component
    
    return reward

def calculate_adaptive_reward(
    navs: List[float],
    balances: List[float],
    positions: List[List[Dict[str, Any]]],
    current_pnl: float,
    market_volatility: Optional[float] = None,
    trade_frequency_target: int = 1,  # Target trades per 100 steps
    episode_step: int = 0,
    **kwargs
) -> float:
    """
    Tính toán phần thưởng thích ứng dựa trên độ biến động thị trường và tần suất giao dịch.
    
    Args:
        navs: Danh sách NAV (Net Asset Value) theo từng bước
        balances: Danh sách số dư theo từng bước
        positions: Danh sách vị thế theo từng bước
        current_pnl: P&L (Profit and Loss) hiện tại
        market_volatility: Độ biến động thị trường hiện tại
        trade_frequency_target: Tần suất giao dịch mục tiêu (giao dịch / 100 bước)
        episode_step: Bước hiện tại trong episode
        **kwargs: Các tham số khác
        
    Returns:
        Giá trị phần thưởng thích ứng
    """
    # Thành phần lợi nhuận cơ bản
    if len(navs) < 2:
        return 0.0
    
    prev_nav = navs[-2]
    current_nav = navs[-1]
    
    if prev_nav <= 0:
        logger.warning("Giá trị NAV trước đó không hợp lệ (≤ 0)")
        return 0.0
    
    # Tính lợi tức
    profit_component = (current_nav - prev_nav) / prev_nav
    
    # Thành phần điều chỉnh độ biến động
    volatility_component = 0.0
    if market_volatility is not None:
        # Trong thị trường biến động, thưởng lợi nhuận nhiều hơn
        # Trong thị trường ít biến động, giảm thưởng lợi nhuận
        volatility_factor = np.tanh(market_volatility * 10)  # Giới hạn trong khoảng [-1, 1]
        volatility_component = profit_component * volatility_factor
    
    # Thành phần điều chỉnh tần suất giao dịch
    frequency_component = 0.0
    if episode_step > 0:
        # Đếm số giao dịch đã thực hiện
        trade_count = 0
        for i in range(1, min(len(positions), episode_step + 1)):
            prev_pos_count = len(positions[i-1]) if i-1 < len(positions) else 0
            curr_pos_count = len(positions[i]) if i < len(positions) else 0
            
            # Đếm các thay đổi vị thế
            if curr_pos_count != prev_pos_count:
                trade_count += abs(curr_pos_count - prev_pos_count)
        
        # Tính tần suất giao dịch hiện tại
        current_frequency = (trade_count / episode_step) * 100
        
        # Điều chỉnh phần thưởng dựa trên sự khác biệt với tần suất mục tiêu
        frequency_diff = current_frequency - trade_frequency_target
        
        if abs(frequency_diff) < 0.5:
            # Gần với tần suất mục tiêu -> thưởng
            frequency_component = 0.01
        elif frequency_diff > 0:
            # Giao dịch quá nhiều -> phạt
            frequency_component = -0.01 * frequency_diff
        else:
            # Giao dịch quá ít -> phạt nhẹ hơn
            frequency_component = 0.005 * frequency_diff  # Âm nhưng nhỏ hơn
    
    # Tính toán phần thưởng cuối cùng
    reward = profit_component + volatility_component + frequency_component
    
    return reward

def calculate_multi_timeframe_reward(
    navs: List[float],
    balances: List[float],
    positions: List[List[Dict[str, Any]]],
    current_pnl: float,
    short_term_trend: Optional[int] = None,  # 1 = lên, -1 = xuống, 0 = đi ngang
    medium_term_trend: Optional[int] = None,
    long_term_trend: Optional[int] = None,
    alignment_bonus: float = 0.02,  # Thưởng khi các xu hướng thống nhất
    **kwargs
) -> float:
    """
    Tính toán phần thưởng dựa trên sự phù hợp với nhiều khung thời gian.
    
    Args:
        navs: Danh sách NAV (Net Asset Value) theo từng bước
        balances: Danh sách số dư theo từng bước
        positions: Danh sách vị thế theo từng bước
        current_pnl: P&L (Profit and Loss) hiện tại
        short_term_trend: Xu hướng ngắn hạn
        medium_term_trend: Xu hướng trung hạn
        long_term_trend: Xu hướng dài hạn
        alignment_bonus: Thưởng cho sự thống nhất xu hướng
        **kwargs: Các tham số khác
        
    Returns:
        Giá trị phần thưởng dựa trên nhiều khung thời gian
    """
    # Thành phần lợi nhuận cơ bản
    if len(navs) < 2:
        return 0.0
    
    prev_nav = navs[-2]
    current_nav = navs[-1]
    
    if prev_nav <= 0:
        logger.warning("Giá trị NAV trước đó không hợp lệ (≤ 0)")
        return 0.0
    
    # Tính lợi tức
    profit_component = (current_nav - prev_nav) / prev_nav
    
    # Thành phần xu hướng đa khung thời gian
    trend_component = 0.0
    
    # Kiểm tra sự thống nhất của các xu hướng
    if None not in [short_term_trend, medium_term_trend, long_term_trend]:
        # Sự thống nhất tuyệt đối (cả 3 xu hướng giống nhau và không phải đi ngang)
        if (short_term_trend == medium_term_trend == long_term_trend) and short_term_trend != 0:
            trend_component += alignment_bonus * 2
        
        # Sự thống nhất của 2 xu hướng và không mâu thuẫn với xu hướng còn lại
        elif (short_term_trend == medium_term_trend and long_term_trend != -short_term_trend) or \
             (short_term_trend == long_term_trend and medium_term_trend != -short_term_trend) or \
             (medium_term_trend == long_term_trend and short_term_trend != -medium_term_trend):
            trend_component += alignment_bonus
    
    # Thành phần vị thế theo xu hướng
    position_trend_component = 0.0
    current_positions = positions[-1] if positions else []
    
    if current_positions:
        # Đánh giá vị thế theo từng khung thời gian
        for pos in current_positions:
            position_side = 1 if pos.get('side') == PositionSide.LONG.value else -1
            
            # Trọng số cho từng khung thời gian (dài hạn quan trọng hơn)
            short_weight = 0.2
            medium_weight = 0.3
            long_weight = 0.5
            
            if short_term_trend is not None:
                # Thưởng/phạt dựa trên việc đi theo xu hướng ngắn hạn
                if position_side == short_term_trend:
                    position_trend_component += short_weight * 0.01
                elif short_term_trend != 0:  # Không phạt nếu đi ngang
                    position_trend_component -= short_weight * 0.01
            
            if medium_term_trend is not None:
                # Thưởng/phạt dựa trên việc đi theo xu hướng trung hạn
                if position_side == medium_term_trend:
                    position_trend_component += medium_weight * 0.01
                elif medium_term_trend != 0:  # Không phạt nếu đi ngang
                    position_trend_component -= medium_weight * 0.01
            
            if long_term_trend is not None:
                # Thưởng/phạt dựa trên việc đi theo xu hướng dài hạn
                if position_side == long_term_trend:
                    position_trend_component += long_weight * 0.01
                elif long_term_trend != 0:  # Không phạt nếu đi ngang
                    position_trend_component -= long_weight * 0.01
    
    # Tính toán phần thưởng cuối cùng
    reward = profit_component + trend_component + position_trend_component
    
    return reward