"""
Hàm phần thưởng theo lợi nhuận.
File này định nghĩa các hàm tính toán phần thưởng dựa trên lợi nhuận
trong quá trình giao dịch, tập trung vào tối đa hóa lợi nhuận.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import logging

from config.logging_config import get_logger

def calculate_profit_reward(
    nav_history: List[float],
    balance_history: List[float],
    position_history: List[List[Dict[str, Any]]],
    current_pnl: float,
    pnl_threshold: float = 0.0,
    consecutive_factor: float = 0.1,
    drawdown_penalty: float = 0.5,
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Tính toán phần thưởng dựa trên lợi nhuận.
    
    Args:
        nav_history: Lịch sử giá trị tài sản ròng
        balance_history: Lịch sử số dư
        position_history: Lịch sử vị thế
        current_pnl: Lợi nhuận hiện tại
        pnl_threshold: Ngưỡng lợi nhuận tối thiểu để có phần thưởng dương
        consecutive_factor: Hệ số thưởng cho các lợi nhuận liên tiếp
        drawdown_penalty: Hệ số phạt cho drawdown
        logger: Logger tùy chỉnh
        
    Returns:
        Giá trị phần thưởng
    """
    # Thiết lập logger
    logger = logger or get_logger("profit_reward")
    
    # Kiểm tra dữ liệu đầu vào
    if len(nav_history) < 2:
        logger.warning("Không đủ dữ liệu để tính toán phần thưởng")
        return 0.0
    
    # Lấy giá trị NAV hiện tại và trước đó
    prev_nav = nav_history[-2]
    current_nav = nav_history[-1]
    
    # Tính phần thưởng cơ bản (% thay đổi NAV)
    if prev_nav <= 0:
        # Tránh chia cho 0
        base_reward = 0.0
    else:
        base_reward = (current_nav - prev_nav) / prev_nav
    
    # Kiểm tra ngưỡng lợi nhuận
    if current_pnl < pnl_threshold:
        base_reward = min(base_reward, 0.0)  # Chỉ cho phần thưởng âm nếu dưới ngưỡng
    
    # Tính toán phần thưởng bổ sung cho các lợi nhuận liên tiếp
    consecutive_reward = 0.0
    if len(nav_history) >= 3:
        # Đếm số lượt lợi nhuận liên tiếp
        consecutive_profits = 0
        for i in range(len(nav_history) - 2, 0, -1):
            if nav_history[i] > nav_history[i-1]:
                consecutive_profits += 1
            else:
                break
        
        # Thưởng cho các lợi nhuận liên tiếp
        if consecutive_profits > 0 and base_reward > 0:
            consecutive_reward = base_reward * consecutive_factor * min(consecutive_profits, 5)
    
    # Tính toán phạt cho drawdown
    drawdown_reward = 0.0
    if len(nav_history) >= 3:
        # Tìm NAV cao nhất trong lịch sử
        max_nav = max(nav_history[:-1])
        
        # Tính drawdown hiện tại
        if max_nav > 0:
            current_drawdown = max(0, (max_nav - current_nav) / max_nav)
            
            # Áp dụng phạt nếu đang trong drawdown
            if current_drawdown > 0:
                drawdown_reward = -current_drawdown * drawdown_penalty
    
    # Tổng hợp phần thưởng với giới hạn
    total_reward = np.clip(base_reward + consecutive_reward + drawdown_reward, -1.0, 1.0)
    
    logger.debug(f"Phần thưởng: cơ bản={base_reward:.4f}, liên tiếp={consecutive_reward:.4f}, drawdown={drawdown_reward:.4f}, tổng={total_reward:.4f}")
    
    return total_reward

def calculate_position_profit_reward(
    positions: List[Dict[str, Any]],
    current_price: float,
    reward_scaling: float = 1.0
) -> float:
    """
    Tính toán phần thưởng dựa trên lợi nhuận của từng vị thế.
    
    Args:
        positions: Danh sách các vị thế hiện tại
        current_price: Giá hiện tại
        reward_scaling: Hệ số tỷ lệ phần thưởng
        
    Returns:
        Giá trị phần thưởng
    """
    if not positions:
        return 0.0
    
    total_pnl = 0.0
    total_size = 0.0
    
    for pos in positions:
        # Lấy thông tin vị thế
        side = pos.get('side', 'long')
        size = pos.get('size', 0.0)
        entry_price = pos.get('entry_price', current_price)
        leverage = pos.get('leverage', 1.0)
        
        # Tính P&L
        if side.lower() == 'long':
            pnl = size * (current_price - entry_price) / entry_price * leverage
        else:  # short
            pnl = size * (entry_price - current_price) / entry_price * leverage
        
        total_pnl += pnl
        total_size += size
    
    # Tính phần thưởng
    if total_size > 0:
        # Phần thưởng là % lợi nhuận trên tổng kích thước vị thế
        reward = (total_pnl / total_size) * reward_scaling
    else:
        reward = 0.0
    
    return reward

def calculate_trade_completion_reward(
    trade_history: List[Dict[str, Any]],
    win_reward: float = 1.0,
    loss_penalty: float = -1.0
) -> float:
    """
    Tính toán phần thưởng dựa trên kết quả hoàn thành giao dịch.
    
    Args:
        trade_history: Lịch sử các giao dịch đã hoàn thành
        win_reward: Phần thưởng cho giao dịch thắng
        loss_penalty: Phạt cho giao dịch thua
        
    Returns:
        Giá trị phần thưởng
    """
    if not trade_history:
        return 0.0
    
    # Chỉ xem xét giao dịch mới nhất
    latest_trade = trade_history[-1]
    
    # Lấy thông tin P&L
    pnl = latest_trade.get('pnl', 0.0)
    
    # Tính phần thưởng
    if pnl > 0:
        reward = win_reward
    else:
        reward = loss_penalty
    
    return reward

def calculate_profit_factor_reward(
    trade_history: List[Dict[str, Any]],
    window_size: int = 10,
    min_trades: int = 3,
    scaling_factor: float = 2.0
) -> float:
    """
    Tính toán phần thưởng dựa trên profit factor (tổng lợi nhuận / tổng lỗ).
    
    Args:
        trade_history: Lịch sử các giao dịch đã hoàn thành
        window_size: Số lượng giao dịch gần nhất để tính toán
        min_trades: Số lượng giao dịch tối thiểu để tính toán
        scaling_factor: Hệ số tỷ lệ phần thưởng
        
    Returns:
        Giá trị phần thưởng
    """
    if len(trade_history) < min_trades:
        return 0.0
    
    # Lấy giao dịch trong cửa sổ
    recent_trades = trade_history[-window_size:]
    
    # Tính tổng lợi nhuận và tổng lỗ
    total_profit = sum(trade['pnl'] for trade in recent_trades if trade.get('pnl', 0) > 0)
    total_loss = sum(abs(trade['pnl']) for trade in recent_trades if trade.get('pnl', 0) < 0)
    
    # Tính profit factor
    if total_loss == 0:
        if total_profit > 0:
            profit_factor = scaling_factor  # Giá trị tối đa nếu không có lỗ
        else:
            profit_factor = 1.0  # Trung tính nếu không có lợi nhuận và không có lỗ
    else:
        profit_factor = total_profit / total_loss
    
    # Ánh xạ profit factor vào khoảng [0, scaling_factor]
    reward = min(profit_factor, scaling_factor)
    
    # Điều chỉnh: profit factor < 1 sẽ cho phần thưởng âm
    if profit_factor < 1.0:
        reward = profit_factor - 1.0  # Dải từ -1.0 đến 0.0
    
    return reward