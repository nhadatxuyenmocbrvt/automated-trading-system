"""
Tính toán mức độ rủi ro.
File này định nghĩa các phương pháp để tính toán và đánh giá mức độ rủi ro
cho các giao dịch dựa trên nhiều chỉ số và chiến lược quản lý rủi ro.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import math

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import ErrorCode, Timeframe, TIMEFRAME_TO_SECONDS

class RiskCalculator:
    """
    Lớp tính toán và đánh giá rủi ro.
    Cung cấp các phương pháp để tính toán các chỉ số rủi ro, phân tích xác suất,
    và đánh giá mức độ rủi ro cho các giao dịch.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.03,  # 3% hàng năm
        confidence_level: float = 0.95,  # 95% VaR
        drawdown_threshold: float = 0.15,  # 15% drawdown tối đa chấp nhận được
        default_currency: str = "USD",
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo RiskCalculator.
        
        Args:
            risk_free_rate: Lãi suất phi rủi ro hàng năm
            confidence_level: Mức độ tin cậy cho VaR
            drawdown_threshold: Ngưỡng drawdown tối đa chấp nhận được
            default_currency: Đơn vị tiền tệ mặc định
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("risk_calculator")
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.drawdown_threshold = drawdown_threshold
        self.default_currency = default_currency
        
        self.logger.info(f"Đã khởi tạo RiskCalculator với risk_free_rate={risk_free_rate}, confidence_level={confidence_level}")
    
    def calculate_max_drawdown(
        self,
        balance_history: List[float],
        window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Tính toán mức drawdown tối đa.
        
        Args:
            balance_history: Lịch sử số dư/NAV
            window: Kích thước cửa sổ tính toán (None để tính toàn bộ)
            
        Returns:
            Dict chứa thông tin drawdown
        """
        if not balance_history:
            self.logger.warning("Lịch sử số dư trống")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Lịch sử số dư trống"
                }
            }
        
        try:
            # Chuyển đổi sang numpy array để tính toán hiệu quả
            balance_array = np.array(balance_history)
            
            # Giới hạn cửa sổ tính toán nếu cần
            if window is not None and window > 0 and window < len(balance_array):
                balance_array = balance_array[-window:]
            
            # Tính peak running maximum
            running_max = np.maximum.accumulate(balance_array)
            
            # Tính drawdown theo công thức: (peak - current) / peak
            drawdown_array = (running_max - balance_array) / running_max
            
            # Tìm drawdown tối đa
            max_drawdown = np.max(drawdown_array)
            max_drawdown_idx = np.argmax(drawdown_array)
            
            # Tìm peak trước drawdown tối đa
            peak_idx = np.maximum.accumulate(np.arange(len(balance_array)))[max_drawdown_idx]
            peak_value = balance_array[peak_idx]
            valley_value = balance_array[max_drawdown_idx]
            
            # Tính thời gian hồi phục (số bước từ đáy đến khi vượt đỉnh)
            recovery_steps = None
            if max_drawdown_idx < len(balance_array) - 1:
                for i in range(max_drawdown_idx + 1, len(balance_array)):
                    if balance_array[i] >= peak_value:
                        recovery_steps = i - max_drawdown_idx
                        break
            
            # Tính drawdown hiện tại
            current_drawdown = (running_max[-1] - balance_array[-1]) / running_max[-1] if running_max[-1] > 0 else 0
            
            # Tính thống kê drawdown
            drawdown_mean = np.mean(drawdown_array)
            drawdown_median = np.median(drawdown_array)
            drawdown_std = np.std(drawdown_array)
            
            # Tính ngưỡng cảnh báo
            warning_threshold = drawdown_mean + 2 * drawdown_std
            
            result = {
                "status": "success",
                "max_drawdown": max_drawdown,
                "max_drawdown_idx": max_drawdown_idx,
                "peak_idx": peak_idx,
                "peak_value": peak_value,
                "valley_value": valley_value,
                "recovery_steps": recovery_steps,
                "current_drawdown": current_drawdown,
                "drawdown_mean": drawdown_mean,
                "drawdown_median": drawdown_median,
                "drawdown_std": drawdown_std,
                "warning_threshold": warning_threshold,
                "is_warning": current_drawdown > warning_threshold,
                "is_critical": current_drawdown > self.drawdown_threshold
            }
            
            self.logger.info(f"Đã tính max drawdown: {max_drawdown:.2%}, hiện tại: {current_drawdown:.2%}")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính drawdown: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính drawdown: {str(e)}"
                }
            }
    
    def calculate_value_at_risk(
        self,
        returns: List[float],
        position_size: float,
        time_horizon: int = 1,  # Số ngày
        method: str = "historical",  # "historical", "parametric", "monte_carlo"
        confidence_level: Optional[float] = None,
        num_simulations: int = 10000  # Cho Monte Carlo
    ) -> Dict[str, Any]:
        """
        Tính toán Value at Risk (VaR).
        
        Args:
            returns: Danh sách lợi nhuận phần trăm hàng ngày
            position_size: Kích thước vị thế
            time_horizon: Khoảng thời gian (ngày)
            method: Phương pháp tính VaR
            confidence_level: Mức độ tin cậy (None để sử dụng mặc định)
            num_simulations: Số lần mô phỏng cho Monte Carlo
            
        Returns:
            Dict chứa thông tin VaR
        """
        if not returns:
            self.logger.warning("Danh sách lợi nhuận trống")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Danh sách lợi nhuận trống"
                }
            }
        
        # Sử dụng mức tin cậy mặc định nếu không được chỉ định
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        try:
            # Chuyển đổi sang numpy array để tính toán hiệu quả
            returns_array = np.array(returns)
            
            # VaR theo phương pháp lịch sử
            if method == "historical":
                # Sắp xếp lợi nhuận tăng dần
                sorted_returns = np.sort(returns_array)
                
                # Tính chỉ số ở mức tin cậy
                index = int(len(sorted_returns) * (1 - confidence_level))
                index = max(0, min(index, len(sorted_returns) - 1))
                
                # Tính VaR
                var_percent = -sorted_returns[index]  # Dấu trừ vì VaR là giá trị dương
                var_amount = position_size * var_percent
                
                # Điều chỉnh cho time_horizon
                var_amount = var_amount * math.sqrt(time_horizon)
                
            # VaR theo phương pháp tham số
            elif method == "parametric":
                # Tính mean và std của returns
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                
                # Tính Z-score ở mức tin cậy
                z_score = -np.percentile(np.random.normal(0, 1, 10000), (1 - confidence_level) * 100)
                
                # Tính VaR
                var_percent = -mean_return + z_score * std_return
                var_amount = position_size * var_percent
                
                # Điều chỉnh cho time_horizon
                var_amount = var_amount * math.sqrt(time_horizon)
                
            # VaR theo phương pháp Monte Carlo
            elif method == "monte_carlo":
                # Tính mean và std của returns
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                
                # Mô phỏng lợi nhuận theo phân phối lognormal
                simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
                
                # Tính lợi nhuận tích lũy cho time_horizon
                cumulative_returns = np.zeros(num_simulations)
                for i in range(num_simulations):
                    cumulative_return = 1.0
                    for _ in range(time_horizon):
                        cumulative_return *= (1 + simulated_returns[i])
                    cumulative_returns[i] = cumulative_return - 1.0
                
                # Tính VaR
                var_percent = -np.percentile(cumulative_returns, (1 - confidence_level) * 100)
                var_amount = position_size * var_percent
                
            else:
                self.logger.warning(f"Phương pháp VaR không hợp lệ: {method}")
                return {
                    "status": "error",
                    "error": {
                        "code": ErrorCode.INVALID_PARAMETER.value,
                        "message": f"Phương pháp VaR không hợp lệ: {method}"
                    }
                }
            
            # Tính Conditional VaR (Expected Shortfall)
            if method == "historical":
                # Lấy các lợi nhuận tệ hơn VaR
                tail_returns = sorted_returns[:index+1]
                cvar_percent = -np.mean(tail_returns) if len(tail_returns) > 0 else var_percent
                cvar_amount = position_size * cvar_percent * math.sqrt(time_horizon)
                
            elif method == "parametric":
                # Tính Expected Shortfall cho phân phối normal
                z_score_es = np.exp(-0.5 * z_score**2) / (math.sqrt(2*math.pi) * (1 - confidence_level))
                cvar_percent = -mean_return + std_return * z_score_es
                cvar_amount = position_size * cvar_percent * math.sqrt(time_horizon)
                
            else:  # monte_carlo
                # Lấy các lợi nhuận tệ hơn VaR
                sorted_returns = np.sort(cumulative_returns)
                index = int(len(sorted_returns) * (1 - confidence_level))
                tail_returns = sorted_returns[:index+1]
                cvar_percent = -np.mean(tail_returns) if len(tail_returns) > 0 else var_percent
                cvar_amount = position_size * cvar_percent
            
            result = {
                "status": "success",
                "var_percent": var_percent,
                "var_amount": var_amount,
                "cvar_percent": cvar_percent,
                "cvar_amount": cvar_amount,
                "method": method,
                "confidence_level": confidence_level,
                "time_horizon": time_horizon,
                "position_size": position_size,
                "currency": self.default_currency
            }
            
            self.logger.info(f"Đã tính VaR ({method}): {var_amount:.2f} {self.default_currency} ({var_percent:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính VaR: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính VaR: {str(e)}"
                }
            }
    
    def calculate_sharpe_ratio(
        self,
        returns: List[float],
        timeframe: str = "1d",
        risk_free_rate: Optional[float] = None,
        annualize: bool = True
    ) -> Dict[str, Any]:
        """
        Tính toán Sharpe Ratio.
        
        Args:
            returns: Danh sách lợi nhuận phần trăm
            timeframe: Khung thời gian của returns
            risk_free_rate: Lãi suất phi rủi ro (None để sử dụng mặc định)
            annualize: Quy đổi về giá trị hàng năm
            
        Returns:
            Dict chứa thông tin Sharpe Ratio
        """
        if not returns:
            self.logger.warning("Danh sách lợi nhuận trống")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Danh sách lợi nhuận trống"
                }
            }
        
        # Sử dụng lãi suất phi rủi ro mặc định nếu không được chỉ định
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        try:
            # Chuyển đổi sang numpy array để tính toán hiệu quả
            returns_array = np.array(returns)
            
            # Tính mean và std của returns
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            # Tính số giao dịch trong năm
            if timeframe in TIMEFRAME_TO_SECONDS:
                seconds_per_interval = TIMEFRAME_TO_SECONDS[timeframe]
                periods_per_year = 365 * 24 * 60 * 60 / seconds_per_interval
            else:
                periods_per_year = {
                    "1m": 525600,    # 365 * 24 * 60
                    "5m": 105120,    # 365 * 24 * 12
                    "15m": 35040,    # 365 * 24 * 4
                    "30m": 17520,    # 365 * 24 * 2
                    "1h": 8760,      # 365 * 24
                    "4h": 2190,      # 365 * 6
                    "1d": 365,
                    "1w": 52,
                    "1M": 12,
                }.get(timeframe, 252)  # Mặc định 252 phiên giao dịch
            
            # Điều chỉnh lãi suất phi rủi ro theo khung thời gian
            daily_risk_free = risk_free_rate / periods_per_year
            
            # Tính Sharpe Ratio
            sharpe_ratio = (mean_return - daily_risk_free) / std_return if std_return > 0 else 0
            
            # Quy đổi về giá trị hàng năm nếu cần
            if annualize:
                sharpe_ratio = sharpe_ratio * math.sqrt(periods_per_year)
            
            # Tính Sortino Ratio (chỉ xét downside risk)
            negative_returns = returns_array[returns_array < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else std_return
            
            sortino_ratio = (mean_return - daily_risk_free) / downside_std if downside_std > 0 else 0
            
            # Quy đổi về giá trị hàng năm nếu cần
            if annualize:
                sortino_ratio = sortino_ratio * math.sqrt(periods_per_year)
            
            # Đánh giá kết quả
            sharpe_rating = "Xuất sắc" if sharpe_ratio > 2.0 else \
                            "Tốt" if sharpe_ratio > 1.0 else \
                            "Trung bình" if sharpe_ratio > 0.5 else \
                            "Yếu" if sharpe_ratio > 0 else \
                            "Rất yếu"
            
            result = {
                "status": "success",
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "mean_return": mean_return,
                "std_return": std_return,
                "downside_std": downside_std,
                "timeframe": timeframe,
                "periods_per_year": periods_per_year,
                "risk_free_rate": risk_free_rate,
                "daily_risk_free": daily_risk_free,
                "annualized": annualize,
                "sharpe_rating": sharpe_rating
            }
            
            self.logger.info(f"Đã tính Sharpe Ratio: {sharpe_ratio:.4f}, Sortino Ratio: {sortino_ratio:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính Sharpe Ratio: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính Sharpe Ratio: {str(e)}"
                }
            }
    
    def calculate_kelly_criterion(
        self,
        win_rate: float,
        win_loss_ratio: float,
        conservative_factor: float = 0.5
    ) -> Dict[str, Any]:
        """
        Tính toán Kelly Criterion để xác định kích thước vị thế tối ưu.
        
        Args:
            win_rate: Tỷ lệ giao dịch thắng (0-1)
            win_loss_ratio: Tỷ lệ giữa lợi nhuận trung bình và lỗ trung bình
            conservative_factor: Hệ số bảo thủ (0-1, thường dùng 0.5)
            
        Returns:
            Dict chứa thông tin Kelly Criterion
        """
        # Kiểm tra tham số
        if win_rate < 0 or win_rate > 1:
            self.logger.warning(f"Tỷ lệ thắng không hợp lệ: {win_rate}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Tỷ lệ thắng phải từ 0 đến 1"
                }
            }
        
        if win_loss_ratio <= 0:
            self.logger.warning(f"Tỷ lệ win/loss không hợp lệ: {win_loss_ratio}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Tỷ lệ win/loss phải > 0"
                }
            }
        
        try:
            # Tính Kelly Criterion
            # f* = (p * b - (1 - p)) / b
            # p: xác suất thắng
            # b: tỷ lệ win/loss
            # f*: phần tối ưu của vốn để đặt cược
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Áp dụng hệ số bảo thủ
            conservative_kelly = kelly_fraction * conservative_factor
            
            # Giới hạn giá trị
            if kelly_fraction < 0:
                # Kelly âm nghĩa là không nên giao dịch
                kelly_fraction = 0
                conservative_kelly = 0
                recommendation = "Không nên giao dịch"
            else:
                if kelly_fraction > 0.5:
                    recommendation = "Cơ hội đầu tư tốt, nhưng nên thận trọng"
                elif kelly_fraction > 0.25:
                    recommendation = "Cơ hội đầu tư khá, có thể cân nhắc"
                elif kelly_fraction > 0:
                    recommendation = "Cơ hội đầu tư nhỏ, cần thận trọng"
                else:
                    recommendation = "Không nên giao dịch"
            
            result = {
                "status": "success",
                "kelly_fraction": kelly_fraction,
                "conservative_kelly": conservative_kelly,
                "win_rate": win_rate,
                "win_loss_ratio": win_loss_ratio,
                "conservative_factor": conservative_factor,
                "recommendation": recommendation,
                "percent_of_capital": conservative_kelly * 100
            }
            
            self.logger.info(f"Đã tính Kelly Criterion: {kelly_fraction:.4f}, conservative: {conservative_kelly:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính Kelly Criterion: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính Kelly Criterion: {str(e)}"
                }
            }
    
    def calculate_risk_of_ruin(
        self,
        win_rate: float,
        win_loss_ratio: float,
        risk_per_trade: float,
        num_trades: int = 100
    ) -> Dict[str, Any]:
        """
        Tính toán Risk of Ruin (xác suất phá sản).
        
        Args:
            win_rate: Tỷ lệ giao dịch thắng (0-1)
            win_loss_ratio: Tỷ lệ giữa lợi nhuận trung bình và lỗ trung bình
            risk_per_trade: Phần trăm vốn rủi ro mỗi giao dịch
            num_trades: Số lượng giao dịch tính xác suất
            
        Returns:
            Dict chứa thông tin Risk of Ruin
        """
        # Kiểm tra tham số
        if win_rate < 0 or win_rate > 1:
            self.logger.warning(f"Tỷ lệ thắng không hợp lệ: {win_rate}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Tỷ lệ thắng phải từ 0 đến 1"
                }
            }
        
        if risk_per_trade <= 0 or risk_per_trade >= 1:
            self.logger.warning(f"Tỷ lệ rủi ro không hợp lệ: {risk_per_trade}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Tỷ lệ rủi ro phải > 0 và < 1"
                }
            }
        
        try:
            # Tính R (odds ratio) = (1 + win_loss_ratio) / (1 - win_loss_ratio)
            # Đảm bảo R > 0
            if win_loss_ratio >= 1:
                # Trường hợp win_loss_ratio >= 1, giao dịch có lợi nhuận dương
                R = (1 + win_loss_ratio) / 1
            else:
                R = (1 + win_loss_ratio) / (1 - win_loss_ratio)
            
            # Tính Risk of Ruin theo công thức (1 - edge)/(1 + edge)^num_trades
            edge = 2 * win_rate - 1  # Expected value per trade
            
            if edge > 0:
                single_trade_risk = (1 - edge) / (1 + edge)
                risk_of_ruin = single_trade_risk ** num_trades
            else:
                # Nếu edge <= 0, chắc chắn sẽ phá sản nếu giao dịch đủ lâu
                risk_of_ruin = 1.0
            
            # Tính Drawdown tối đa dự kiến
            # Sử dụng phương pháp ước lượng từ lý thuyết xác suất
            # Expected max drawdown ~ sqrt(num_trades) * sqrt(win_rate * (1 - win_rate)) * risk_per_trade
            expected_max_dd = math.sqrt(num_trades) * math.sqrt(win_rate * (1 - win_rate)) * risk_per_trade
            
            # Đánh giá mức độ rủi ro
            if risk_of_ruin < 0.01:
                risk_rating = "Rất thấp"
            elif risk_of_ruin < 0.05:
                risk_rating = "Thấp"
            elif risk_of_ruin < 0.1:
                risk_rating = "Trung bình"
            elif risk_of_ruin < 0.25:
                risk_rating = "Cao"
            else:
                risk_rating = "Rất cao"
            
            result = {
                "status": "success",
                "risk_of_ruin": risk_of_ruin,
                "risk_of_ruin_percent": risk_of_ruin * 100,
                "win_rate": win_rate,
                "win_loss_ratio": win_loss_ratio,
                "risk_per_trade": risk_per_trade,
                "num_trades": num_trades,
                "odds_ratio": R,
                "edge": edge,
                "risk_rating": risk_rating,
                "expected_max_drawdown": expected_max_dd,
                "expected_max_drawdown_percent": expected_max_dd * 100
            }
            
            self.logger.info(f"Đã tính Risk of Ruin: {risk_of_ruin:.6f} ({risk_rating})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính Risk of Ruin: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính Risk of Ruin: {str(e)}"
                }
            }
    
    def calculate_position_correlation(
        self,
        returns_data: Dict[str, List[float]],
        timeframe: str = "1d",
        min_correlation: float = 0.6
    ) -> Dict[str, Any]:
        """
        Tính toán ma trận tương quan giữa các vị thế.
        
        Args:
            returns_data: Dict với key là symbol và value là danh sách lợi nhuận
            timeframe: Khung thời gian của returns
            min_correlation: Ngưỡng tương quan tối thiểu để cảnh báo
            
        Returns:
            Dict chứa ma trận tương quan và cảnh báo
        """
        if not returns_data:
            self.logger.warning("Dữ liệu lợi nhuận trống")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Dữ liệu lợi nhuận trống"
                }
            }
        
        try:
            # Tạo DataFrame từ dữ liệu lợi nhuận
            returns_df = pd.DataFrame(returns_data)
            
            # Tính ma trận tương quan
            correlation_matrix = returns_df.corr()
            
            # Tìm các cặp có tương quan cao
            high_correlation_pairs = []
            
            # Lấy tam giác trên của ma trận tương quan
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    symbol1 = correlation_matrix.columns[i]
                    symbol2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]
                    
                    # Kiểm tra nếu tương quan vượt ngưỡng
                    if abs(corr_value) >= min_correlation:
                        high_correlation_pairs.append({
                            "symbol1": symbol1,
                            "symbol2": symbol2,
                            "correlation": corr_value,
                            "sign": "positive" if corr_value > 0 else "negative"
                        })
            
            # Sắp xếp theo mức độ tương quan giảm dần
            high_correlation_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            # Tính tỷ lệ tương quan cao
            total_pairs = len(correlation_matrix.columns) * (len(correlation_matrix.columns) - 1) / 2
            high_corr_ratio = len(high_correlation_pairs) / total_pairs if total_pairs > 0 else 0
            
            # Đánh giá mức độ đa dạng hóa
            if high_corr_ratio > 0.5:
                diversification_rating = "Kém"
            elif high_corr_ratio > 0.3:
                diversification_rating = "Trung bình"
            elif high_corr_ratio > 0.1:
                diversification_rating = "Khá"
            else:
                diversification_rating = "Tốt"
            
            result = {
                "status": "success",
                "correlation_matrix": correlation_matrix.to_dict(),
                "high_correlation_pairs": high_correlation_pairs,
                "timeframe": timeframe,
                "min_correlation": min_correlation,
                "num_symbols": len(returns_data),
                "total_pairs": total_pairs,
                "high_correlation_count": len(high_correlation_pairs),
                "high_correlation_ratio": high_corr_ratio,
                "diversification_rating": diversification_rating
            }
            
            self.logger.info(f"Đã tính tương quan: {len(high_correlation_pairs)} cặp có tương quan cao, mức đa dạng hóa: {diversification_rating}")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính tương quan: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính tương quan: {str(e)}"
                }
            }
    
    def calculate_risk_adjusted_return(
        self,
        returns: List[float],
        timeframe: str = "1d",
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
        target_max_drawdown: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Tính toán các chỉ số lợi nhuận điều chỉnh theo rủi ro.
        
        Args:
            returns: Danh sách lợi nhuận phần trăm
            timeframe: Khung thời gian của returns
            target_return: Lợi nhuận mục tiêu
            target_volatility: Độ biến động mục tiêu
            target_max_drawdown: Drawdown tối đa mục tiêu
            
        Returns:
            Dict chứa các chỉ số lợi nhuận điều chỉnh theo rủi ro
        """
        if not returns:
            self.logger.warning("Danh sách lợi nhuận trống")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Danh sách lợi nhuận trống"
                }
            }
        
        try:
            # Chuyển đổi sang numpy array để tính toán hiệu quả
            returns_array = np.array(returns)
            
            # Tính mean và std của returns
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            # Tính số giao dịch trong năm
            if timeframe in TIMEFRAME_TO_SECONDS:
                seconds_per_interval = TIMEFRAME_TO_SECONDS[timeframe]
                periods_per_year = 365 * 24 * 60 * 60 / seconds_per_interval
            else:
                periods_per_year = {
                    "1m": 525600,    # 365 * 24 * 60
                    "5m": 105120,    # 365 * 24 * 12
                    "15m": 35040,    # 365 * 24 * 4
                    "30m": 17520,    # 365 * 24 * 2
                    "1h": 8760,      # 365 * 24
                    "4h": 2190,      # 365 * 6
                    "1d": 365,
                    "1w": 52,
                    "1M": 12,
                }.get(timeframe, 252)  # Mặc định 252 phiên giao dịch
            
            # Quy đổi về giá trị hàng năm
            annual_return = (1 + mean_return) ** periods_per_year - 1
            annual_volatility = std_return * math.sqrt(periods_per_year)
            
            # Tính Sharpe Ratio
            daily_risk_free = self.risk_free_rate / periods_per_year
            sharpe_ratio = (mean_return - daily_risk_free) / std_return if std_return > 0 else 0
            
            # Quy đổi về giá trị hàng năm
            sharpe_ratio_annualized = sharpe_ratio * math.sqrt(periods_per_year)
            
            # Tính Sortino Ratio (chỉ xét downside risk)
            negative_returns = returns_array[returns_array < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else std_return
            
            sortino_ratio = (mean_return - daily_risk_free) / downside_std if downside_std > 0 else 0
            sortino_ratio_annualized = sortino_ratio * math.sqrt(periods_per_year)
            
            # Tính Calmar Ratio
            # Tính drawdown theo balance_history
            balance_history = [1.0]
            for r in returns_array:
                balance_history.append(balance_history[-1] * (1 + r))
            
            max_drawdown_result = self.calculate_max_drawdown(balance_history)
            
            if max_drawdown_result.get("status") == "success":
                max_drawdown = max_drawdown_result.get("max_drawdown", 0)
                
                # Tính Calmar Ratio = annual_return / max_drawdown
                calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            else:
                max_drawdown = 0
                calmar_ratio = 0
            
            # Tính Omega Ratio (tỷ lệ lợi nhuận tích cực/tiêu cực)
            # Threshold: 0 (lãi suất không rủi ro)
            threshold = daily_risk_free
            positive_returns = returns_array[returns_array > threshold]
            negative_returns = returns_array[returns_array <= threshold]
            
            positive_mean = np.mean(positive_returns) if len(positive_returns) > 0 else 0
            negative_mean = np.mean(negative_returns - threshold) if len(negative_returns) > 0 else 0
            
            positive_sum = np.sum(positive_returns - threshold) if len(positive_returns) > 0 else 0
            negative_sum = abs(np.sum(negative_returns - threshold)) if len(negative_returns) > 0 else 0
            
            omega_ratio = positive_sum / negative_sum if negative_sum > 0 else float('inf')
            
            # Đánh giá kết quả
            performance_rating = "Xuất sắc" if sharpe_ratio_annualized > 2.0 else \
                                "Tốt" if sharpe_ratio_annualized > 1.0 else \
                                "Trung bình" if sharpe_ratio_annualized > 0.5 else \
                                "Yếu" if sharpe_ratio_annualized > 0 else \
                                "Rất yếu"
            
            # So sánh với mục tiêu
            target_comparison = {}
            if target_return is not None:
                target_comparison["return"] = {
                    "target": target_return,
                    "actual": annual_return,
                    "achieved": annual_return >= target_return
                }
            
            if target_volatility is not None:
                target_comparison["volatility"] = {
                    "target": target_volatility,
                    "actual": annual_volatility,
                    "achieved": annual_volatility <= target_volatility
                }
            
            if target_max_drawdown is not None:
                target_comparison["max_drawdown"] = {
                    "target": target_max_drawdown,
                    "actual": max_drawdown,
                    "achieved": max_drawdown <= target_max_drawdown
                }
            
            result = {
                "status": "success",
                "mean_return": mean_return,
                "std_return": std_return,
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe_ratio,
                "sharpe_ratio_annualized": sharpe_ratio_annualized,
                "sortino_ratio": sortino_ratio,
                "sortino_ratio_annualized": sortino_ratio_annualized,
                "max_drawdown": max_drawdown,
                "calmar_ratio": calmar_ratio,
                "omega_ratio": omega_ratio,
                "timeframe": timeframe,
                "periods_per_year": periods_per_year,
                "downside_std": downside_std,
                "performance_rating": performance_rating,
                "target_comparison": target_comparison
            }
            
            self.logger.info(f"Đã tính chỉ số rủi ro-lợi nhuận: Sharpe={sharpe_ratio_annualized:.2f}, Sortino={sortino_ratio_annualized:.2f}, Calmar={calmar_ratio:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính chỉ số rủi ro-lợi nhuận: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính chỉ số rủi ro-lợi nhuận: {str(e)}"
                }
            }
    
    def assess_market_risk(
        self,
        volatility: float,
        trading_volume: float,
        average_volatility: float,
        average_volume: float,
        market_trend: str,  # "bullish", "bearish", "sideways"
        prev_volatility: Optional[float] = None,
        market_sentiment: Optional[str] = None,  # "positive", "negative", "neutral"
        key_levels_proximity: Optional[float] = None,  # 0-1, 1 = rất gần
        risk_events: List[str] = []  # Các sự kiện rủi ro
    ) -> Dict[str, Any]:
        """
        Đánh giá rủi ro thị trường dựa trên nhiều yếu tố.
        
        Args:
            volatility: Độ biến động hiện tại
            trading_volume: Khối lượng giao dịch hiện tại
            average_volatility: Độ biến động trung bình trong quá khứ
            average_volume: Khối lượng giao dịch trung bình trong quá khứ
            market_trend: Xu hướng thị trường
            prev_volatility: Độ biến động trước đó
            market_sentiment: Tâm lý thị trường
            key_levels_proximity: Độ gần với các mức hỗ trợ/kháng cự quan trọng
            risk_events: Danh sách các sự kiện rủi ro
            
        Returns:
            Dict chứa đánh giá rủi ro thị trường
        """
        try:
            # Tính biến động tương đối
            vol_ratio = volatility / average_volatility if average_volatility > 0 else 1
            
            # Tính khối lượng tương đối
            volume_ratio = trading_volume / average_volume if average_volume > 0 else 1
            
            # Tính biến động đang tăng/giảm
            vol_change = 0
            if prev_volatility is not None and prev_volatility > 0:
                vol_change = (volatility - prev_volatility) / prev_volatility
            
            # Đánh giá rủi ro dựa trên độ biến động
            volatility_risk = 0
            if vol_ratio > 2.0:
                volatility_risk = 4  # Rất cao
            elif vol_ratio > 1.5:
                volatility_risk = 3  # Cao
            elif vol_ratio > 1.2:
                volatility_risk = 2  # Trung bình
            elif vol_ratio > 1.0:
                volatility_risk = 1  # Thấp
            else:
                volatility_risk = 0  # Rất thấp
            
            # Điều chỉnh nếu biến động đang tăng nhanh
            if vol_change > 0.3:
                volatility_risk = min(volatility_risk + 2, 4)
            elif vol_change > 0.1:
                volatility_risk = min(volatility_risk + 1, 4)
            
            # Đánh giá rủi ro dựa trên khối lượng
            volume_risk = 0
            if volume_ratio > 3.0:
                volume_risk = 4  # Rất cao (khối lượng bất thường)
            elif volume_ratio > 2.0:
                volume_risk = 3  # Cao
            elif volume_ratio > 1.5:
                volume_risk = 2  # Trung bình
            elif volume_ratio > 1.0:
                volume_risk = 1  # Thấp
            else:
                volume_risk = 0  # Rất thấp
            
            # Đánh giá rủi ro dựa trên xu hướng
            trend_risk = 0
            if market_trend == "bearish":
                trend_risk = 3  # Cao trong thị trường giảm
            elif market_trend == "bullish":
                trend_risk = 1  # Thấp trong thị trường tăng
            else:  # sideways
                trend_risk = 2  # Trung bình trong thị trường sideway
            
            # Đánh giá rủi ro dựa trên tâm lý
            sentiment_risk = 0
            if market_sentiment == "negative":
                sentiment_risk = 3  # Cao khi tâm lý tiêu cực
            elif market_sentiment == "positive":
                sentiment_risk = 1  # Thấp khi tâm lý tích cực
            else:  # neutral
                sentiment_risk = 2  # Trung bình khi tâm lý trung lập
            
            # Đánh giá rủi ro dựa trên mức gần các mức hỗ trợ/kháng cự
            level_risk = 0
            if key_levels_proximity is not None:
                if key_levels_proximity > 0.8:
                    level_risk = 3  # Cao khi rất gần mức quan trọng
                elif key_levels_proximity > 0.5:
                    level_risk = 2  # Trung bình
                else:
                    level_risk = 1  # Thấp
            
            # Đánh giá rủi ro dựa trên các sự kiện
            event_risk = min(len(risk_events), 4)
            
            # Tính tổng điểm rủi ro (trọng số khác nhau cho mỗi thành phần)
            risk_factors = {
                "volatility": {"score": volatility_risk, "weight": 0.3},
                "volume": {"score": volume_risk, "weight": 0.15},
                "trend": {"score": trend_risk, "weight": 0.2},
                "sentiment": {"score": sentiment_risk, "weight": 0.15},
                "key_levels": {"score": level_risk, "weight": 0.1},
                "events": {"score": event_risk, "weight": 0.1}
            }
            
            weighted_risk_score = sum(factor["score"] * factor["weight"] for factor in risk_factors.values())
            
            # Xác định mức độ rủi ro tổng thể
            if weighted_risk_score > 3.0:
                risk_level = "Rất cao"
                risk_level_value = 4
            elif weighted_risk_score > 2.0:
                risk_level = "Cao"
                risk_level_value = 3
            elif weighted_risk_score > 1.0:
                risk_level = "Trung bình"
                risk_level_value = 2
            elif weighted_risk_score > 0.5:
                risk_level = "Thấp"
                risk_level_value = 1
            else:
                risk_level = "Rất thấp"
                risk_level_value = 0
            
            # Đưa ra khuyến nghị dựa trên mức độ rủi ro
            if risk_level_value >= 3:
                recommendation = "Không nên giao dịch hoặc giảm kích thước vị thế xuống rất thấp"
            elif risk_level_value == 2:
                recommendation = "Hãy thận trọng, giảm kích thước vị thế và sử dụng stop loss chặt chẽ"
            elif risk_level_value == 1:
                recommendation = "Có thể giao dịch nhưng nên thận trọng và theo dõi các yếu tố rủi ro"
            else:
                recommendation = "Điều kiện giao dịch thuận lợi, rủi ro thấp"
            
            # Tạo danh sách các yếu tố rủi ro được sắp xếp theo mức độ rủi ro
            sorted_risk_factors = sorted(
                [(name, details["score"]) for name, details in risk_factors.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            top_risk_factors = [factor[0] for factor in sorted_risk_factors if factor[1] >= 3]
            
            result = {
                "status": "success",
                "weighted_risk_score": weighted_risk_score,
                "risk_level": risk_level,
                "risk_level_value": risk_level_value,
                "recommendation": recommendation,
                "risk_factors": risk_factors,
                "top_risk_factors": top_risk_factors,
                "vol_ratio": vol_ratio,
                "volume_ratio": volume_ratio,
                "vol_change": vol_change,
                "market_trend": market_trend,
                "market_sentiment": market_sentiment,
                "key_levels_proximity": key_levels_proximity,
                "risk_events": risk_events
            }
            
            self.logger.info(f"Đã đánh giá rủi ro thị trường: {risk_level} (điểm: {weighted_risk_score:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi đánh giá rủi ro thị trường: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi đánh giá rủi ro thị trường: {str(e)}"
                }
            }