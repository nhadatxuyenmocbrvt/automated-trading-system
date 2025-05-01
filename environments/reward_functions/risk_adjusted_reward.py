"""
Hàm phần thưởng điều chỉnh theo rủi ro.
File này cung cấp các hàm tính toán phần thưởng có điều chỉnh theo các chỉ số rủi ro.
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional
import pandas as pd
import logging
import math

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import PositionSide

# Thiết lập logger
logger = get_logger("reward_functions")

def calculate_risk_adjusted_reward(
    navs: List[float],
    balances: List[float],
    positions: List[List[Dict[str, Any]]],
    current_pnl: float,
    risk_free_rate: float = 0.0,
    window_size: int = 30,
    annualization_factor: int = 252,  # 252 trading days in a year
    **kwargs
) -> float:
    """
    Tính toán phần thưởng điều chỉnh theo rủi ro (Sharpe Ratio).
    
    Args:
        navs: Danh sách NAV (Net Asset Value) theo từng bước
        balances: Danh sách số dư theo từng bước
        positions: Danh sách vị thế theo từng bước
        current_pnl: P&L (Profit and Loss) hiện tại
        risk_free_rate: Lãi suất phi rủi ro hàng ngày
        window_size: Kích thước cửa sổ để tính toán
        annualization_factor: Hệ số quy đổi năm
        **kwargs: Các tham số khác
        
    Returns:
        Giá trị phần thưởng điều chỉnh theo rủi ro
    """
    # Kiểm tra tham số
    if len(navs) < 2:
        return 0.0
    
    # Lấy NAV hiện tại và NAV trước đó
    prev_nav = navs[-2]
    current_nav = navs[-1]
    
    # Tính lợi tức cơ bản
    if prev_nav <= 0:
        logger.warning("Giá trị NAV trước đó không hợp lệ (≤ 0)")
        return 0.0
    
    current_return = (current_nav - prev_nav) / prev_nav
    
    # Số lượng dữ liệu lịch sử cần sử dụng
    history_size = min(window_size, len(navs) - 1)
    
    if history_size < 2:
        # Không đủ dữ liệu lịch sử, trả về lợi tức hiện tại
        return current_return
    
    # Tính toán lợi tức lịch sử
    returns = []
    for i in range(1, history_size + 1):
        if i + 1 < len(navs):
            prev = navs[-i-1]
            curr = navs[-i]
            if prev > 0:
                returns.append((curr - prev) / prev)
    
    # Nếu không đủ dữ liệu lợi tức
    if len(returns) < 2:
        return current_return
    
    # Tính toán thống kê
    avg_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)  # ddof=1 for sample standard deviation
    
    # Tránh chia cho 0
    if std_return <= 0:
        # Nếu độ lệch chuẩn bằng 0, trả về dấu của (avg_return - risk_free_rate)
        if avg_return > risk_free_rate:
            return 1.0  # Lợi tức tốt hơn lãi suất phi rủi ro, không có biến động
        elif avg_return < risk_free_rate:
            return -1.0  # Lợi tức kém hơn lãi suất phi rủi ro, không có biến động
        else:
            return 0.0  # Bằng lãi suất phi rủi ro, không có biến động
    
    # Tính toán Sharpe Ratio hàng ngày
    daily_sharpe = (avg_return - risk_free_rate) / std_return
    
    # Điều chỉnh reward dựa trên Sharpe Ratio
    # Sử dụng hàm tanh để giới hạn phần thưởng
    reward = np.tanh(daily_sharpe)
    
    return reward

def calculate_sortino_reward(
    navs: List[float],
    balances: List[float],
    positions: List[List[Dict[str, Any]]],
    current_pnl: float,
    risk_free_rate: float = 0.0,
    window_size: int = 30,
    **kwargs
) -> float:
    """
    Tính toán phần thưởng dựa trên Sortino Ratio (chỉ xem xét downside risk).
    
    Args:
        navs: Danh sách NAV (Net Asset Value) theo từng bước
        balances: Danh sách số dư theo từng bước
        positions: Danh sách vị thế theo từng bước
        current_pnl: P&L (Profit and Loss) hiện tại
        risk_free_rate: Lãi suất phi rủi ro hàng ngày
        window_size: Kích thước cửa sổ để tính toán
        **kwargs: Các tham số khác
        
    Returns:
        Giá trị phần thưởng dựa trên Sortino Ratio
    """
    # Kiểm tra tham số
    if len(navs) < 2:
        return 0.0
    
    # Lấy NAV hiện tại và NAV trước đó
    prev_nav = navs[-2]
    current_nav = navs[-1]
    
    # Tính lợi tức cơ bản
    if prev_nav <= 0:
        logger.warning("Giá trị NAV trước đó không hợp lệ (≤ 0)")
        return 0.0
    
    current_return = (current_nav - prev_nav) / prev_nav
    
    # Số lượng dữ liệu lịch sử cần sử dụng
    history_size = min(window_size, len(navs) - 1)
    
    if history_size < 2:
        # Không đủ dữ liệu lịch sử, trả về lợi tức hiện tại
        return current_return
    
    # Tính toán lợi tức lịch sử
    returns = []
    for i in range(1, history_size + 1):
        if i + 1 < len(navs):
            prev = navs[-i-1]
            curr = navs[-i]
            if prev > 0:
                returns.append((curr - prev) / prev)
    
    # Nếu không đủ dữ liệu lợi tức
    if len(returns) < 2:
        return current_return
    
    # Tính toán thống kê
    avg_return = np.mean(returns)
    
    # Tính downside deviation (chỉ xem xét lợi tức âm)
    # Lấy các lợi tức thấp hơn MAR (Minimum Acceptable Return, thường là risk_free_rate)
    downside_returns = [r - risk_free_rate for r in returns if r < risk_free_rate]
    
    if not downside_returns:
        # Không có lợi tức âm, trả về giá trị lớn
        return 1.0
    
    # Tính downside deviation
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))
    
    # Tránh chia cho 0
    if downside_deviation <= 0:
        if avg_return > risk_free_rate:
            return 1.0
        elif avg_return < risk_free_rate:
            return -1.0
        else:
            return 0.0
    
    # Tính toán Sortino Ratio
    sortino_ratio = (avg_return - risk_free_rate) / downside_deviation
    
    # Điều chỉnh reward dựa trên Sortino Ratio
    # Sử dụng hàm tanh để giới hạn phần thưởng
    reward = np.tanh(sortino_ratio)
    
    return reward

def calculate_calmar_reward(
    navs: List[float],
    balances: List[float],
    positions: List[List[Dict[str, Any]]],
    current_pnl: float,
    window_size: int = 100,
    **kwargs
) -> float:
    """
    Tính toán phần thưởng dựa trên Calmar Ratio (tỷ lệ lợi tức trung bình trên drawdown tối đa).
    
    Args:
        navs: Danh sách NAV (Net Asset Value) theo từng bước
        balances: Danh sách số dư theo từng bước
        positions: Danh sách vị thế theo từng bước
        current_pnl: P&L (Profit and Loss) hiện tại
        window_size: Kích thước cửa sổ để tính toán
        **kwargs: Các tham số khác
        
    Returns:
        Giá trị phần thưởng dựa trên Calmar Ratio
    """
    # Kiểm tra tham số
    if len(navs) < 2:
        return 0.0
    
    # Lấy NAV hiện tại và NAV trước đó
    prev_nav = navs[-2]
    current_nav = navs[-1]
    
    # Tính lợi tức cơ bản
    if prev_nav <= 0:
        logger.warning("Giá trị NAV trước đó không hợp lệ (≤ 0)")
        return 0.0
    
    current_return = (current_nav - prev_nav) / prev_nav
    
    # Số lượng dữ liệu lịch sử cần sử dụng
    history_size = min(window_size, len(navs))
    
    if history_size < 10:  # Yêu cầu ít nhất 10 điểm dữ liệu
        return current_return
    
    # Lấy dữ liệu NAV trong cửa sổ
    nav_window = navs[-history_size:]
    
    # Tính toán lợi tức trung bình
    returns = []
    for i in range(1, len(nav_window)):
        prev = nav_window[i-1]
        curr = nav_window[i]
        if prev > 0:
            returns.append((curr - prev) / prev)
    
    avg_return = np.mean(returns) if returns else 0.0
    
    # Tính toán maximum drawdown
    # Maximum drawdown là sự sụt giảm tối đa từ đỉnh đến đáy trong một khoảng thời gian
    max_drawdown = 0.0
    peak = nav_window[0]
    
    for nav in nav_window:
        if nav > peak:
            peak = nav
        drawdown = (peak - nav) / peak if peak > 0 else 0.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Tránh chia cho 0
    if max_drawdown <= 0:
        if avg_return > 0:
            return 1.0
        elif avg_return < 0:
            return -1.0
        else:
            return 0.0
    
    # Tính toán Calmar Ratio
    calmar_ratio = avg_return / max_drawdown
    
    # Điều chỉnh reward dựa trên Calmar Ratio
    # Sử dụng hàm tanh để giới hạn phần thưởng
    reward = np.tanh(calmar_ratio)
    
    return reward

def calculate_combined_risk_reward(
    navs: List[float],
    balances: List[float],
    positions: List[List[Dict[str, Any]]],
    current_pnl: float,
    risk_free_rate: float = 0.0,
    sharpe_weight: float = 0.4,
    sortino_weight: float = 0.4,
    calmar_weight: float = 0.2,
    **kwargs
) -> float:
    """
    Tính toán phần thưởng kết hợp từ nhiều chỉ số rủi ro.
    
    Args:
        navs: Danh sách NAV (Net Asset Value) theo từng bước
        balances: Danh sách số dư theo từng bước
        positions: Danh sách vị thế theo từng bước
        current_pnl: P&L (Profit and Loss) hiện tại
        risk_free_rate: Lãi suất phi rủi ro hàng ngày
        sharpe_weight: Trọng số cho Sharpe Ratio
        sortino_weight: Trọng số cho Sortino Ratio
        calmar_weight: Trọng số cho Calmar Ratio
        **kwargs: Các tham số khác
        
    Returns:
        Giá trị phần thưởng kết hợp
    """
    # Tính toán từng loại reward
    sharpe_reward = calculate_risk_adjusted_reward(
        navs, balances, positions, current_pnl, risk_free_rate, **kwargs
    )
    
    sortino_reward = calculate_sortino_reward(
        navs, balances, positions, current_pnl, risk_free_rate, **kwargs
    )
    
    calmar_reward = calculate_calmar_reward(
        navs, balances, positions, current_pnl, **kwargs
    )
    
    # Tính toán reward kết hợp có trọng số
    combined_reward = (
        sharpe_weight * sharpe_reward +
        sortino_weight * sortino_reward +
        calmar_weight * calmar_reward
    )
    
    return combined_reward