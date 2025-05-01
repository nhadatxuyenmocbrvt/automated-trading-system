"""
Hàm phần thưởng theo lợi nhuận.
File này cung cấp các hàm tính toán phần thưởng dựa trên lợi nhuận từ giao dịch.
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

def calculate_profit_reward(
    navs: List[float],
    balances: List[float],
    positions: List[List[Dict[str, Any]]],
    current_pnl: float,
    use_log_returns: bool = False,
    scaling_factor: float = 1.0,
    penalize_inactivity: bool = False,
    inactivity_penalty: float = 0.0001,
    **kwargs
) -> float:
    """
    Tính toán phần thưởng dựa trên lợi nhuận.
    
    Args:
        navs: Danh sách NAV (Net Asset Value) theo từng bước
        balances: Danh sách số dư theo từng bước
        positions: Danh sách vị thế theo từng bước
        current_pnl: P&L (Profit and Loss) hiện tại
        use_log_returns: Sử dụng lợi tức logarit thay vì lợi tức tuyệt đối
        scaling_factor: Hệ số điều chỉnh phần thưởng (để tăng/giảm gradient)
        penalize_inactivity: Phạt khi không hoạt động
        inactivity_penalty: Mức phạt cho việc không hoạt động
        **kwargs: Các tham số khác
        
    Returns:
        Giá trị phần thưởng
    """
    # Kiểm tra tham số
    if len(navs) < 2:
        return 0.0
    
    # Tính tỷ lệ thay đổi NAV
    prev_nav = navs[-2]
    current_nav = navs[-1]
    
    if prev_nav <= 0:
        # Tránh chia cho 0
        logger.warning("Giá trị NAV trước đó không hợp lệ (≤ 0)")
        return 0.0
    
    # Tính phần thưởng dựa trên thay đổi tương đối của NAV
    if use_log_returns:
        # Sử dụng logarithmic returns (phù hợp hơn cho các mô hình)
        # log(current_nav / prev_nav) tương đương với log(current_nav) - log(prev_nav)
        # Điều này cho phép tập trung vào phần trăm thay đổi thay vì giá trị tuyệt đối
        reward = np.log(current_nav / prev_nav)
    else:
        # Sử dụng phần trăm thay đổi đơn giản
        reward = (current_nav - prev_nav) / prev_nav
    
    # Áp dụng hệ số điều chỉnh
    reward *= scaling_factor
    
    # Phạt khi không hoạt động (nếu được kích hoạt)
    if penalize_inactivity:
        # Kiểm tra xem có bất kỳ vị thế mở nào không
        current_positions = positions[-1] if positions else []
        
        if not current_positions:
            # Không có vị thế mở nào -> áp dụng mức phạt
            reward -= inactivity_penalty
    
    return reward

def calculate_profit_reward_with_risk_factor(
    navs: List[float],
    balances: List[float],
    positions: List[List[Dict[str, Any]]],
    current_pnl: float,
    volatility_window: int = 20,
    risk_aversion_factor: float = 1.0,
    **kwargs
) -> float:
    """
    Tính toán phần thưởng theo lợi nhuận có điều chỉnh theo rủi ro đơn giản.
    
    Args:
        navs: Danh sách NAV (Net Asset Value) theo từng bước
        balances: Danh sách số dư theo từng bước
        positions: Danh sách vị thế theo từng bước
        current_pnl: P&L (Profit and Loss) hiện tại
        volatility_window: Cửa sổ để tính toán độ biến động
        risk_aversion_factor: Hệ số ảnh hưởng của rủi ro (càng lớn càng né rủi ro)
        **kwargs: Các tham số khác
        
    Returns:
        Giá trị phần thưởng có điều chỉnh rủi ro
    """
    # Tính phần thưởng lợi nhuận cơ bản
    base_reward = calculate_profit_reward(navs, balances, positions, current_pnl)
    
    # Nếu không đủ dữ liệu để tính toán độ biến động
    if len(navs) < volatility_window + 1:
        return base_reward
    
    # Tính toán lợi tức theo từng bước
    returns = []
    for i in range(1, min(volatility_window + 1, len(navs))):
        prev = navs[-i-1]
        curr = navs[-i]
        if prev > 0:
            returns.append((curr - prev) / prev)
    
    # Nếu không đủ dữ liệu lợi tức
    if not returns:
        return base_reward
    
    # Tính độ biến động (độ lệch chuẩn của lợi tức)
    volatility = np.std(returns)
    
    # Điều chỉnh phần thưởng dựa trên rủi ro
    # Công thức: reward / (1 + risk_aversion_factor * volatility)
    if volatility > 0:
        risk_adjusted_reward = base_reward / (1 + risk_aversion_factor * volatility)
    else:
        risk_adjusted_reward = base_reward
    
    return risk_adjusted_reward

def calculate_trade_based_reward(
    navs: List[float],
    balances: List[float],
    positions: List[List[Dict[str, Any]]],
    current_pnl: float,
    last_action: Optional[Dict[str, Any]] = None,
    win_bonus: float = 0.1,
    loss_penalty: float = 0.2,
    **kwargs
) -> float:
    """
    Tính toán phần thưởng dựa trên kết quả của giao dịch cụ thể.
    
    Args:
        navs: Danh sách NAV (Net Asset Value) theo từng bước
        balances: Danh sách số dư theo từng bước
        positions: Danh sách vị thế theo từng bước
        current_pnl: P&L (Profit and Loss) hiện tại
        last_action: Thông tin về hành động cuối cùng
        win_bonus: Phần thưởng thêm cho giao dịch thắng
        loss_penalty: Phạt thêm cho giao dịch thua
        **kwargs: Các tham số khác
        
    Returns:
        Giá trị phần thưởng
    """
    # Tính phần thưởng lợi nhuận cơ bản
    base_reward = calculate_profit_reward(navs, balances, positions, current_pnl)
    
    # Nếu không có thông tin về hành động cuối cùng
    if last_action is None:
        return base_reward
    
    # Kiểm tra xem hành động cuối có phải là đóng vị thế không
    if last_action.get('action_type') in ['close', 'close_all']:
        # Lấy P&L từ hành động
        action_pnl = last_action.get('pnl', 0.0)
        
        # Điều chỉnh phần thưởng dựa trên kết quả giao dịch
        if action_pnl > 0:
            # Giao dịch thắng -> thêm thưởng
            base_reward += win_bonus
        elif action_pnl < 0:
            # Giao dịch thua -> thêm phạt
            base_reward -= loss_penalty
    
    return base_reward