"""
Hàm phần thưởng tùy chỉnh.
File này cung cấp các hàm tính toán phần thưởng tùy chỉnh, kết hợp
nhiều tiêu chí và cho phép người dùng tạo các hàm phần thưởng riêng.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import logging
import json

from config.logging_config import get_logger
from config.constants import PositionSide, OrderType

def calculate_custom_reward(
    nav_history: List[float],
    balance_history: List[float],
    position_history: List[List[Dict[str, Any]]],
    current_pnl: float,
    performance_metrics: Optional[Dict[str, Any]] = None,
    reward_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Tính toán phần thưởng tùy chỉnh dựa trên nhiều tiêu chí.
    
    Args:
        nav_history: Lịch sử giá trị tài sản ròng
        balance_history: Lịch sử số dư
        position_history: Lịch sử vị thế
        current_pnl: Lợi nhuận hiện tại
        performance_metrics: Các chỉ số hiệu suất hiện tại
        reward_config: Cấu hình phần thưởng tùy chỉnh
        logger: Logger tùy chỉnh
        
    Returns:
        Giá trị phần thưởng
    """
    # Thiết lập logger
    logger = logger or get_logger("custom_reward")
    
    # Kiểm tra dữ liệu đầu vào
    if len(nav_history) < 2:
        logger.warning("Không đủ dữ liệu để tính toán phần thưởng")
        return 0.0
    
    # Tải cấu hình mặc định nếu không được cung cấp
    if reward_config is None:
        reward_config = {
            "profit_weight": 0.4,
            "risk_weight": 0.3,
            "trade_frequency_weight": 0.1,
            "consistency_weight": 0.2,
            "drawdown_penalty": 0.5,
            "idle_penalty": 0.1,
            "over_trading_penalty": 0.2,
            "min_trades_for_metrics": 3,
            "lookback_window": 20
        }
    
    # Khởi tạo các thành phần phần thưởng
    profit_reward = 0.0
    risk_reward = 0.0
    trade_frequency_reward = 0.0
    consistency_reward = 0.0
    penalty_reward = 0.0
    
    # Lấy giá trị NAV hiện tại và trước đó
    prev_nav = nav_history[-2]
    current_nav = nav_history[-1]
    
    # 1. Tính phần thưởng lợi nhuận
    if prev_nav > 0:
        profit_reward = (current_nav - prev_nav) / prev_nav
    
    # 2. Tính phần thưởng rủi ro
    if performance_metrics and 'win_count' in performance_metrics and 'loss_count' in performance_metrics:
        win_count = performance_metrics.get('win_count', 0)
        loss_count = performance_metrics.get('loss_count', 0)
        total_trades = win_count + loss_count
        
        if total_trades >= reward_config.get('min_trades_for_metrics', 3):
            # Tính win rate
            win_rate = win_count / total_trades if total_trades > 0 else 0.0
            
            # Tính reward dựa trên win rate
            # Ánh xạ win rate [0, 1] thành [-0.5, 1.0]
            risk_reward = win_rate * 1.5 - 0.5
    
    # 3. Tính phần thưởng tần suất giao dịch
    recent_pos_changes = _count_position_changes(position_history, reward_config.get('lookback_window', 20))
    
    # Tần suất giao dịch lý tưởng là 1-3 thay đổi mỗi 10 bước
    ideal_frequency = 0.2  # 2 giao dịch / 10 bước
    actual_frequency = recent_pos_changes / min(len(position_history), reward_config.get('lookback_window', 20))
    
    if actual_frequency == 0:
        # Phạt cho việc không giao dịch
        trade_frequency_reward = -reward_config.get('idle_penalty', 0.1)
    elif actual_frequency > ideal_frequency * 3:
        # Phạt cho việc giao dịch quá nhiều
        trade_frequency_reward = -reward_config.get('over_trading_penalty', 0.2)
    else:
        # Thưởng cho tần suất giao dịch hợp lý
        # Ánh xạ tần suất [0, ideal_frequency*3] thành [0, 1]
        normalized_freq = min(actual_frequency / (ideal_frequency * 2), 1.0)
        trade_frequency_reward = normalized_freq
    
    # 4. Tính phần thưởng tính nhất quán
    consistency_reward = _calculate_consistency_reward(nav_history, reward_config.get('lookback_window', 20))
    
    # 5. Tính phần phạt cho drawdown lớn
    if performance_metrics and 'max_drawdown' in performance_metrics:
        max_drawdown = performance_metrics.get('max_drawdown', 0.0)
        if max_drawdown > 0.1:  # Phạt khi drawdown vượt quá 10%
            penalty_factor = min(max_drawdown * 2, 1.0)  # Giới hạn ở 50% drawdown
            penalty_reward = -penalty_factor * reward_config.get('drawdown_penalty', 0.5)
    
    # Tổng hợp phần thưởng với trọng số
    total_reward = (
        profit_reward * reward_config.get('profit_weight', 0.4) +
        risk_reward * reward_config.get('risk_weight', 0.3) +
        trade_frequency_reward * reward_config.get('trade_frequency_weight', 0.1) +
        consistency_reward * reward_config.get('consistency_weight', 0.2) +
        penalty_reward
    )
    
    logger.debug(f"Phần thưởng tùy chỉnh: lợi nhuận={profit_reward:.4f}, rủi ro={risk_reward:.4f}, tần suất={trade_frequency_reward:.4f}, nhất quán={consistency_reward:.4f}, phạt={penalty_reward:.4f}, tổng={total_reward:.4f}")
    
    return total_reward

def _count_position_changes(position_history: List[List[Dict[str, Any]]], window_size: int = 20) -> int:
    """
    Đếm số lượng thay đổi vị thế trong cửa sổ thời gian.
    
    Args:
        position_history: Lịch sử vị thế
        window_size: Kích thước cửa sổ
        
    Returns:
        Số lượng thay đổi vị thế
    """
    if len(position_history) < 2:
        return 0
    
    # Lấy cửa sổ dữ liệu
    window_size = min(window_size, len(position_history))
    window = position_history[-window_size:]
    
    # Đếm số lượng thay đổi
    changes = 0
    prev_positions = set()
    
    for positions in window:
        current_positions = set(pos.get('id', i) for i, pos in enumerate(positions))
        
        # Đếm vị thế mới mở
        new_positions = current_positions - prev_positions
        changes += len(new_positions)
        
        # Đếm vị thế đã đóng
        closed_positions = prev_positions - current_positions
        changes += len(closed_positions)
        
        prev_positions = current_positions
    
    return changes

def _calculate_consistency_reward(nav_history: List[float], window_size: int = 20) -> float:
    """
    Tính toán phần thưởng cho tính nhất quán của NAV.
    
    Args:
        nav_history: Lịch sử NAV
        window_size: Kích thước cửa sổ
        
    Returns:
        Giá trị phần thưởng nhất quán
    """
    if len(nav_history) < window_size:
        return 0.0
    
    # Lấy cửa sổ dữ liệu
    window = nav_history[-window_size:]
    
    # Tính các lợi nhuận theo từng bước
    returns = []
    for i in range(1, len(window)):
        if window[i-1] > 0:
            ret = (window[i] - window[i-1]) / window[i-1]
            returns.append(ret)
    
    if not returns:
        return 0.0
    
    # Tính độ lệch chuẩn của lợi nhuận
    std_dev = np.std(returns)
    
    # Reward cao cho độ lệch chuẩn thấp (nhất quán cao)
    # Ánh xạ std_dev [0, 0.1] thành [1, 0]
    consistency_reward = max(0.0, 1.0 - std_dev * 10.0)
    
    return consistency_reward

def create_multi_objective_reward(
    profit_fn: Callable,
    risk_fn: Callable,
    weights: Dict[str, float] = None
) -> Callable:
    """
    Tạo hàm phần thưởng nhiều mục tiêu bằng cách kết hợp các hàm riêng lẻ.
    
    Args:
        profit_fn: Hàm tính phần thưởng lợi nhuận
        risk_fn: Hàm tính phần thưởng rủi ro
        weights: Trọng số cho từng thành phần
        
    Returns:
        Hàm phần thưởng kết hợp
    """
    if weights is None:
        weights = {'profit': 0.6, 'risk': 0.4}
    
    def combined_reward(
        nav_history: List[float],
        balance_history: List[float],
        position_history: List[List[Dict[str, Any]]],
        current_pnl: float,
        **kwargs
    ) -> float:
        # Tính phần thưởng lợi nhuận
        profit_reward = profit_fn(
            nav_history=nav_history,
            balance_history=balance_history,
            position_history=position_history,
            current_pnl=current_pnl,
            **kwargs
        )
        
        # Tính phần thưởng rủi ro
        risk_reward = risk_fn(
            nav_history=nav_history,
            balance_history=balance_history,
            position_history=position_history,
            current_pnl=current_pnl,
            **kwargs
        )
        
        # Kết hợp với trọng số
        combined = (
            profit_reward * weights.get('profit', 0.6) +
            risk_reward * weights.get('risk', 0.4)
        )
        
        return combined
    
    return combined_reward

def load_reward_function(config_path: str) -> Callable:
    """
    Tải hàm phần thưởng tùy chỉnh từ file cấu hình.
    
    Args:
        config_path: Đường dẫn file cấu hình
        
    Returns:
        Hàm phần thưởng đã cấu hình
    """
    logger = get_logger("custom_reward")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        reward_type = config.get('type', 'custom')
        
        if reward_type == 'profit':
            from environments.reward_functions.profit_reward import calculate_profit_reward
            return lambda **kwargs: calculate_profit_reward(**kwargs, **config.get('params', {}))
        
        elif reward_type == 'risk_adjusted':
            from environments.reward_functions.risk_adjusted_reward import calculate_risk_adjusted_reward
            return lambda **kwargs: calculate_risk_adjusted_reward(**kwargs, **config.get('params', {}))
        
        elif reward_type == 'custom':
            return lambda **kwargs: calculate_custom_reward(**kwargs, reward_config=config.get('params', {}))
        
        elif reward_type == 'multi_objective':
            # Tải các hàm thành phần
            profit_fn_name = config.get('profit_function', 'profit_reward.calculate_profit_reward')
            risk_fn_name = config.get('risk_function', 'risk_adjusted_reward.calculate_risk_adjusted_reward')
            
            # Import động
            profit_module, profit_func = profit_fn_name.split('.')
            risk_module, risk_func = risk_fn_name.split('.')
            
            profit_mod = __import__(f'environments.reward_functions.{profit_module}', fromlist=[profit_func])
            risk_mod = __import__(f'environments.reward_functions.{risk_module}', fromlist=[risk_func])
            
            profit_fn = getattr(profit_mod, profit_func)
            risk_fn = getattr(risk_mod, risk_func)
            
            # Tạo hàm kết hợp
            return create_multi_objective_reward(
                profit_fn=profit_fn,
                risk_fn=risk_fn,
                weights=config.get('weights', {'profit': 0.6, 'risk': 0.4})
            )
        
        else:
            logger.warning(f"Loại phần thưởng không được hỗ trợ: {reward_type}, sử dụng phần thưởng tùy chỉnh mặc định")
            return calculate_custom_reward
            
    except Exception as e:
        logger.error(f"Lỗi khi tải hàm phần thưởng: {str(e)}")
        return calculate_custom_reward

def create_staged_reward(
    thresholds: List[float],
    reward_functions: List[Callable],
    transition_smoothing: bool = True
) -> Callable:
    """
    Tạo hàm phần thưởng phân giai đoạn, sử dụng các hàm khác nhau tùy theo thành tích.
    
    Args:
        thresholds: Danh sách ngưỡng NAV để chuyển đổi giữa các hàm
        reward_functions: Danh sách hàm phần thưởng tương ứng với mỗi giai đoạn
        transition_smoothing: Làm mịn chuyển đổi giữa các hàm
        
    Returns:
        Hàm phần thưởng phân giai đoạn
    """
    if len(thresholds) != len(reward_functions) - 1:
        raise ValueError("Số lượng ngưỡng phải bằng số lượng hàm - 1")
    
    def staged_reward(
        nav_history: List[float],
        balance_history: List[float],
        position_history: List[List[Dict[str, Any]]],
        current_pnl: float,
        **kwargs
    ) -> float:
        if len(nav_history) < 1:
            return 0.0
        
        # Lấy NAV hiện tại
        current_nav = nav_history[-1]
        initial_balance = balance_history[0]
        
        # Tính NAV tương đối (so với số dư ban đầu)
        relative_nav = current_nav / initial_balance if initial_balance > 0 else 1.0
        
        # Xác định giai đoạn hiện tại
        stage = 0
        for i, threshold in enumerate(thresholds):
            if relative_nav >= threshold:
                stage = i + 1
            else:
                break
        
        # Lấy hàm phần thưởng cho giai đoạn hiện tại
        reward_fn = reward_functions[stage]
        reward = reward_fn(
            nav_history=nav_history,
            balance_history=balance_history,
            position_history=position_history,
            current_pnl=current_pnl,
            **kwargs
        )
        
        # Làm mịn chuyển đổi nếu cần
        if transition_smoothing and stage > 0 and relative_nav <= thresholds[stage-1] * 1.1:
            # Trong vùng chuyển đổi (±10% quanh ngưỡng)
            prev_fn = reward_functions[stage-1]
            prev_reward = prev_fn(
                nav_history=nav_history,
                balance_history=balance_history,
                position_history=position_history,
                current_pnl=current_pnl,
                **kwargs
            )
            
            # Tính trọng số cho việc pha trộn
            threshold = thresholds[stage-1]
            blend_range = threshold * 0.1
            blend_weight = (relative_nav - threshold) / blend_range
            blend_weight = max(0.0, min(1.0, blend_weight))
            
            # Pha trộn phần thưởng
            reward = prev_reward * (1 - blend_weight) + reward * blend_weight
        
        return reward
    
    return staged_reward