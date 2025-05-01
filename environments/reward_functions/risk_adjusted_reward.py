"""
Hàm phần thưởng điều chỉnh theo rủi ro.
File này định nghĩa các hàm tính toán phần thưởng có điều chỉnh theo
các thông số rủi ro như volatility, drawdown, hoặc Sharpe ratio.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from math import sqrt

from config.logging_config import get_logger

def calculate_risk_adjusted_reward(
    nav_history: List[float],
    balance_history: List[float],
    position_history: List[List[Dict[str, Any]]],
    current_pnl: float,
    risk_free_rate: float = 0.0,
    window_size: int = 20,
    min_window_size: int = 5,
    volatility_penalty: float = 1.0,
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Tính toán phần thưởng điều chỉnh theo rủi ro dựa trên các chỉ số như Sharpe ratio.
    
    Args:
        nav_history: Lịch sử giá trị tài sản ròng
        balance_history: Lịch sử số dư
        position_history: Lịch sử vị thế
        current_pnl: Lợi nhuận hiện tại
        risk_free_rate: Lãi suất phi rủi ro (hàng ngày)
        window_size: Kích thước cửa sổ để tính toán
        min_window_size: Kích thước cửa sổ tối thiểu
        volatility_penalty: Hệ số phạt cho biến động
        logger: Logger tùy chỉnh
        
    Returns:
        Giá trị phần thưởng
    """
    # Thiết lập logger
    logger = logger or get_logger("risk_adjusted_reward")
    
    # Kiểm tra dữ liệu đầu vào
    if len(nav_history) < max(2, min_window_size):
        logger.warning(f"Không đủ dữ liệu để tính toán phần thưởng điều chỉnh theo rủi ro, cần ít nhất {max(2, min_window_size)} điểm dữ liệu")
        return 0.0
    
    # Lấy cửa sổ dữ liệu NAV
    window_size = min(window_size, len(nav_history))
    nav_window = nav_history[-window_size:]
    
    # Tính toán phần thưởng lợi nhuận cơ bản (% thay đổi NAV)
    prev_nav = nav_history[-2]
    current_nav = nav_history[-1]
    
    if prev_nav <= 0:
        # Tránh chia cho 0
        base_reward = 0.0
    else:
        base_reward = (current_nav - prev_nav) / prev_nav
    
    # Tính toán lợi nhuận trung bình
    returns = []
    for i in range(1, len(nav_window)):
        if nav_window[i-1] > 0:  # Tránh chia cho 0
            ret = (nav_window[i] - nav_window[i-1]) / nav_window[i-1]
            returns.append(ret)
    
    if not returns:
        logger.warning("Không có dữ liệu lợi nhuận để tính toán")
        return base_reward  # Trả về phần thưởng cơ bản nếu không có dữ liệu
    
    # Tính toán Sharpe ratio
    avg_return = np.mean(returns)
    std_return = np.std(returns) if len(returns) > 1 else 0.0001  # Tránh chia cho 0
    
    # Điều chỉnh theo kỳ hạn (giả sử dữ liệu theo ngày)
    daily_risk_free = risk_free_rate / 365.0
    
    if std_return == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (avg_return - daily_risk_free) / std_return
        
    # Điều chỉnh Sharpe ratio thành phần thưởng
    if sharpe_ratio >= 0:
        sharpe_reward = sharpe_ratio
    else:
        # Phần thưởng âm cho Sharpe âm, nhưng với penalty lớn hơn
        sharpe_reward = sharpe_ratio * 2.0
    
    # Tính toán drawdown
    drawdown_penalty = 0.0
    if len(nav_window) >= 3:
        drawdowns = []
        peak = nav_window[0]
        
        for nav in nav_window[1:]:
            if nav > peak:
                peak = nav
            drawdown = (peak - nav) / peak if peak > 0 else 0
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns) if drawdowns else 0
        drawdown_penalty = -max_drawdown * volatility_penalty
    
    # Tính toán phần thưởng cuối cùng
    # Trọng số: 40% base_reward, 40% sharpe_reward, 20% drawdown_penalty
    total_reward = 0.4 * base_reward + 0.4 * sharpe_reward + 0.2 * drawdown_penalty
    
    logger.debug(f"Phần thưởng điều chỉnh theo rủi ro: cơ bản={base_reward:.4f}, sharpe={sharpe_reward:.4f}, drawdown={drawdown_penalty:.4f}, tổng={total_reward:.4f}")
    
    return total_reward

def calculate_sortino_reward(
    nav_history: List[float],
    risk_free_rate: float = 0.0,
    window_size: int = 20,
    min_window_size: int = 5,
    scaling_factor: float = 2.0
) -> float:
    """
    Tính toán phần thưởng dựa trên tỷ số Sortino (chỉ tính downside risk).
    
    Args:
        nav_history: Lịch sử giá trị tài sản ròng
        risk_free_rate: Lãi suất phi rủi ro (hàng ngày)
        window_size: Kích thước cửa sổ để tính toán
        min_window_size: Kích thước cửa sổ tối thiểu
        scaling_factor: Hệ số tỷ lệ phần thưởng
        
    Returns:
        Giá trị phần thưởng
    """
    if len(nav_history) < max(2, min_window_size):
        return 0.0
    
    # Lấy cửa sổ dữ liệu NAV
    window_size = min(window_size, len(nav_history))
    nav_window = nav_history[-window_size:]
    
    # Tính toán lợi nhuận
    returns = []
    for i in range(1, len(nav_window)):
        if nav_window[i-1] > 0:  # Tránh chia cho 0
            ret = (nav_window[i] - nav_window[i-1]) / nav_window[i-1]
            returns.append(ret)
    
    if not returns:
        return 0.0
    
    # Tính toán Sortino ratio
    avg_return = np.mean(returns)
    
    # Tính downside deviation (chỉ tính các lợi nhuận âm)
    negative_returns = [ret for ret in returns if ret < 0]
    
    if not negative_returns:
        # Nếu không có lợi nhuận âm, giả định một downside deviation nhỏ
        downside_deviation = 0.0001
    else:
        downside_deviation = sqrt(sum(ret**2 for ret in negative_returns) / len(negative_returns))
    
    # Điều chỉnh theo kỳ hạn (giả sử dữ liệu theo ngày)
    daily_risk_free = risk_free_rate / 365.0
    
    if downside_deviation == 0:
        sortino_ratio = scaling_factor if avg_return > daily_risk_free else 0.0
    else:
        sortino_ratio = (avg_return - daily_risk_free) / downside_deviation
    
    # Điều chỉnh Sortino ratio thành phần thưởng
    if sortino_ratio >= 0:
        reward = min(sortino_ratio, scaling_factor)
    else:
        # Phần thưởng âm cho Sortino âm
        reward = max(sortino_ratio, -scaling_factor)
    
    return reward

def calculate_calmar_reward(
    nav_history: List[float],
    annualization_factor: float = 252.0,  # Số ngày giao dịch trong năm
    window_size: int = 60,
    min_window_size: int = 10,
    scaling_factor: float = 2.0
) -> float:
    """
    Tính toán phần thưởng dựa trên tỷ số Calmar (lợi nhuận trung bình / max drawdown).
    
    Args:
        nav_history: Lịch sử giá trị tài sản ròng
        annualization_factor: Hệ số annualization
        window_size: Kích thước cửa sổ để tính toán
        min_window_size: Kích thước cửa sổ tối thiểu
        scaling_factor: Hệ số tỷ lệ phần thưởng
        
    Returns:
        Giá trị phần thưởng
    """
    if len(nav_history) < max(2, min_window_size):
        return 0.0
    
    # Lấy cửa sổ dữ liệu NAV
    window_size = min(window_size, len(nav_history))
    nav_window = nav_history[-window_size:]
    
    # Tính tổng lợi nhuận
    first_nav = nav_window[0]
    last_nav = nav_window[-1]
    
    if first_nav <= 0:
        return 0.0
    
    total_return = (last_nav - first_nav) / first_nav
    
    # Annualize return
    period = window_size / annualization_factor
    annual_return = (1 + total_return) ** (1 / period) - 1 if period > 0 else 0
    
    # Tính toán maximum drawdown
    peak = nav_window[0]
    max_drawdown = 0.0
    
    for nav in nav_window[1:]:
        if nav > peak:
            peak = nav
        else:
            drawdown = (peak - nav) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
    
    # Tính Calmar ratio
    if max_drawdown == 0:
        calmar_ratio = scaling_factor if annual_return > 0 else 0.0
    else:
        calmar_ratio = annual_return / max_drawdown
    
    # Điều chỉnh Calmar ratio thành phần thưởng
    if calmar_ratio >= 0:
        reward = min(calmar_ratio, scaling_factor)
    else:
        # Phần thưởng âm cho Calmar âm, nhưng với penalty lớn hơn
        reward = max(calmar_ratio * 1.5, -scaling_factor)
    
    return reward

def calculate_risk_reward_profile_reward(
    nav_history: List[float],
    position_history: List[List[Dict[str, Any]]],
    win_rate_weight: float = 0.3,
    profit_factor_weight: float = 0.3,
    drawdown_weight: float = 0.4,
    window_size: int = 30,
    min_trades: int = 3
) -> float:
    """
    Tính toán phần thưởng dựa trên hồ sơ lợi nhuận-rủi ro tổng hợp.
    
    Args:
        nav_history: Lịch sử giá trị tài sản ròng
        position_history: Lịch sử vị thế
        win_rate_weight: Trọng số cho tỷ lệ thắng
        profit_factor_weight: Trọng số cho profit factor
        drawdown_weight: Trọng số cho drawdown
        window_size: Kích thước cửa sổ để tính toán
        min_trades: Số lượng giao dịch tối thiểu
        
    Returns:
        Giá trị phần thưởng
    """
    if len(nav_history) < 2 or len(position_history) < min_trades:
        return 0.0
    
    # Lấy dữ liệu trong cửa sổ
    window_size = min(window_size, len(nav_history))
    nav_window = nav_history[-window_size:]
    pos_window = position_history[-window_size:]
    
    # Tính tỷ lệ thắng
    win_count = 0
    loss_count = 0
    total_profit = 0.0
    total_loss = 0.0
    
    for positions in pos_window:
        for pos in positions:
            if 'closed' in pos and pos['closed']:
                pnl = pos.get('pnl', 0.0)
                if pnl > 0:
                    win_count += 1
                    total_profit += pnl
                else:
                    loss_count += 1
                    total_loss += abs(pnl)
    
    total_trades = win_count + loss_count
    
    if total_trades < min_trades:
        return 0.0
    
    # Tính các thành phần
    win_rate = win_count / total_trades if total_trades > 0 else 0.0
    
    profit_factor = total_profit / total_loss if total_loss > 0 else (2.0 if total_profit > 0 else 1.0)
    # Chuẩn hóa profit factor vào khoảng [0, 1]
    normalized_profit_factor = min(profit_factor / 2.0, 1.0)
    
    # Tính maximum drawdown
    peak = nav_window[0]
    max_drawdown = 0.0
    
    for nav in nav_window[1:]:
        if nav > peak:
            peak = nav
        else:
            drawdown = (peak - nav) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
    
    # Điểm drawdown (thấp hơn là tốt hơn)
    drawdown_score = 1.0 - min(max_drawdown * 2.0, 1.0)
    
    # Tính toán phần thưởng tổng hợp
    reward = (
        win_rate * win_rate_weight +
        normalized_profit_factor * profit_factor_weight +
        drawdown_score * drawdown_weight
    )
    
    # Điều chỉnh khoảng phần thưởng
    adjusted_reward = (reward * 2.0) - 1.0  # Ánh xạ [0, 1] thành [-1, 1]
    
    return adjusted_reward