"""
Quản lý sụt giảm vốn.
File này định nghĩa các phương pháp để theo dõi, phân tích và xử lý drawdown
nhằm bảo vệ vốn và giảm thiểu rủi ro trong quá trình giao dịch.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import ErrorCode
from risk_management.risk_calculator import RiskCalculator

class DrawdownManager:
    """
    Lớp quản lý sụt giảm vốn (drawdown).
    Cung cấp các phương pháp để theo dõi, phân tích và đưa ra chiến lược ứng phó
    với các tình huống drawdown để bảo vệ vốn.
    """
    
    def __init__(
        self,
        initial_balance: float,
        max_acceptable_drawdown: float = 0.2,  # 20% drawdown tối đa chấp nhận được
        warning_threshold: float = 0.1,  # 10% drawdown ngưỡng cảnh báo
        critical_threshold: float = 0.15,  # 15% drawdown ngưỡng nguy hiểm
        recovery_factor: float = 0.5,  # Hệ số giảm rủi ro trong thời kỳ hồi phục
        lookback_window: int = 30,  # Cửa sổ nhìn lại (ngày)
        risk_calculator: Optional[RiskCalculator] = None,
        auto_adjust_risk: bool = True,
        position_sizing_factors: Optional[Dict[str, float]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo quản lý drawdown.
        
        Args:
            initial_balance: Số dư ban đầu
            max_acceptable_drawdown: Drawdown tối đa chấp nhận được
            warning_threshold: Ngưỡng drawdown để cảnh báo
            critical_threshold: Ngưỡng drawdown để hành động quyết liệt
            recovery_factor: Hệ số giảm rủi ro khi hồi phục
            lookback_window: Cửa sổ nhìn lại (ngày) để phân tích drawdown
            risk_calculator: Đối tượng RiskCalculator
            auto_adjust_risk: Tự động điều chỉnh rủi ro theo drawdown
            position_sizing_factors: Hệ số điều chỉnh kích thước vị thế cho các mức drawdown
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("drawdown_manager")
        self.initial_balance = initial_balance
        self.max_acceptable_drawdown = max_acceptable_drawdown
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.recovery_factor = recovery_factor
        self.lookback_window = lookback_window
        self.auto_adjust_risk = auto_adjust_risk
        
        # Mặc định cho position_sizing_factors nếu không được cung cấp
        if position_sizing_factors is None:
            self.position_sizing_factors = {
                "normal": 1.0,  # < warning_threshold
                "warning": 0.7,  # warning_threshold to critical_threshold
                "critical": 0.5,  # critical_threshold to max_acceptable_drawdown
                "emergency": 0.25,  # > max_acceptable_drawdown
                "recovery": 0.8   # Trong giai đoạn hồi phục
            }
        else:
            self.position_sizing_factors = position_sizing_factors
        
        # Khởi tạo risk_calculator nếu không được cung cấp
        if risk_calculator is None:
            from risk_management.risk_calculator import RiskCalculator
            self.risk_calculator = RiskCalculator()
        else:
            self.risk_calculator = risk_calculator
        
        # Lưu trữ lịch sử
        self.balance_history = [initial_balance]
        self.drawdown_history = [0.0]
        self.timestamp_history = [datetime.now()]
        self.max_balance = initial_balance
        self.current_drawdown = 0.0
        self.in_recovery_mode = False
        self.recovery_start_date = None
        self.recovery_target_balance = initial_balance
        
        self.logger.info(f"Đã khởi tạo DrawdownManager với max_drawdown={max_acceptable_drawdown:.2%}, auto_adjust_risk={auto_adjust_risk}")
    
    def update_balance(self, new_balance: float, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Cập nhật số dư và tính toán drawdown hiện tại.
        
        Args:
            new_balance: Số dư mới
            timestamp: Thời gian cập nhật (mặc định: datetime.now())
            
        Returns:
            Dict chứa thông tin drawdown cập nhật
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Thêm vào lịch sử
        self.balance_history.append(new_balance)
        self.timestamp_history.append(timestamp)
        
        # Cập nhật max_balance nếu cần
        if new_balance > self.max_balance:
            self.max_balance = new_balance
            
            # Nếu đang trong chế độ hồi phục và đã vượt qua recovery_target_balance
            if self.in_recovery_mode and new_balance >= self.recovery_target_balance:
                self.in_recovery_mode = False
                self.recovery_start_date = None
                self.logger.info(f"Đã hoàn thành hồi phục, quay lại chế độ giao dịch bình thường với số dư {new_balance:.2f}")
        
        # Tính drawdown hiện tại: (max_balance - current_balance) / max_balance
        if self.max_balance > 0:
            self.current_drawdown = (self.max_balance - new_balance) / self.max_balance
        else:
            self.current_drawdown = 0.0
        
        self.drawdown_history.append(self.current_drawdown)
        
        # Xác định mức độ drawdown
        drawdown_level = self.get_drawdown_level(self.current_drawdown)
        
        # Nếu không ở chế độ hồi phục và drawdown ở mức critical hoặc emergency
        if not self.in_recovery_mode and drawdown_level in ["critical", "emergency"]:
            self.enter_recovery_mode(new_balance)
        
        # Tính kích thước vị thế điều chỉnh (nếu cần)
        position_size_factor = self.calculate_position_size_factor()
        
        result = {
            "status": "success",
            "current_balance": new_balance,
            "max_balance": self.max_balance,
            "current_drawdown": self.current_drawdown,
            "drawdown_level": drawdown_level,
            "in_recovery_mode": self.in_recovery_mode,
            "position_size_factor": position_size_factor,
            "timestamp": timestamp
        }
        
        self.logger.info(f"Đã cập nhật drawdown: {self.current_drawdown:.2%}, level: {drawdown_level}, factor: {position_size_factor:.2f}")
        return result
    
    def get_drawdown_level(self, drawdown: float) -> str:
        """
        Xác định mức độ drawdown.
        
        Args:
            drawdown: Giá trị drawdown
            
        Returns:
            Mức độ drawdown ("normal", "warning", "critical", "emergency")
        """
        if drawdown >= self.max_acceptable_drawdown:
            return "emergency"
        elif drawdown >= self.critical_threshold:
            return "critical"
        elif drawdown >= self.warning_threshold:
            return "warning"
        else:
            return "normal"
    
    def calculate_position_size_factor(self) -> float:
        """
        Tính toán hệ số điều chỉnh kích thước vị thế dựa trên drawdown.
        
        Returns:
            Hệ số điều chỉnh kích thước vị thế
        """
        # Nếu không tự động điều chỉnh rủi ro, trả về 1.0
        if not self.auto_adjust_risk:
            return 1.0
        
        # Xác định mức độ drawdown
        drawdown_level = self.get_drawdown_level(self.current_drawdown)
        
        # Nếu đang trong chế độ hồi phục, sử dụng hệ số recovery
        if self.in_recovery_mode:
            return self.position_sizing_factors.get("recovery", 0.8)
        
        # Nếu không, sử dụng hệ số tương ứng với mức độ drawdown
        return self.position_sizing_factors.get(drawdown_level, 1.0)
    
    def enter_recovery_mode(self, current_balance: float) -> None:
        """
        Chuyển sang chế độ hồi phục sau khi trải qua drawdown đáng kể.
        
        Args:
            current_balance: Số dư hiện tại
        """
        self.in_recovery_mode = True
        self.recovery_start_date = datetime.now()
        
        # Thiết lập mục tiêu hồi phục (ví dụ: trở lại 90% max_balance)
        self.recovery_target_balance = self.max_balance * 0.9
        
        # Tính thời gian dự kiến để hồi phục
        estimated_daily_return = 0.005  # Giả sử lợi nhuận hàng ngày 0.5%
        expected_days_to_recover = np.log(self.recovery_target_balance / current_balance) / np.log(1 + estimated_daily_return)
        
        self.logger.info(f"Đã chuyển sang chế độ hồi phục. Mục tiêu: {self.recovery_target_balance:.2f}, " +
                         f"dự kiến: {int(expected_days_to_recover)} ngày")
    
    def analyze_drawdown(self, window: Optional[int] = None) -> Dict[str, Any]:
        """
        Phân tích chi tiết drawdown trong một khoảng thời gian.
        
        Args:
            window: Số ngày nhìn lại (None để sử dụng lookback_window mặc định)
            
        Returns:
            Dict chứa thông tin phân tích drawdown
        """
        # Sử dụng cửa sổ mặc định nếu không được chỉ định
        if window is None:
            window = self.lookback_window
        
        # Đảm bảo có đủ dữ liệu
        if len(self.balance_history) < 2:
            self.logger.warning("Không đủ dữ liệu lịch sử để phân tích drawdown")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.DATA_NOT_FOUND.value,
                    "message": "Không đủ dữ liệu lịch sử để phân tích drawdown"
                }
            }
        
        try:
            # Tạo DataFrame cho phân tích
            df = pd.DataFrame({
                'timestamp': self.timestamp_history,
                'balance': self.balance_history,
                'drawdown': self.drawdown_history
            })
            
            # Lọc theo cửa sổ thời gian
            if len(df) > window:
                df = df.tail(window)
            
            # Tính giá trị cao nhất, thấp nhất, trung bình của drawdown
            max_dd = df['drawdown'].max()
            min_dd = df['drawdown'].min()
            avg_dd = df['drawdown'].mean()
            std_dd = df['drawdown'].std()
            
            # Tìm drawdown kéo dài nhất
            consecutive_dd = 0
            max_consecutive_dd = 0
            recovery_count = 0
            
            for i in range(1, len(df)):
                if df['balance'].iloc[i] < df['balance'].iloc[i-1]:
                    consecutive_dd += 1
                else:
                    if consecutive_dd > 0:
                        recovery_count += 1
                    max_consecutive_dd = max(max_consecutive_dd, consecutive_dd)
                    consecutive_dd = 0
            
            # Tính tốc độ hồi phục trung bình (nếu có giai đoạn hồi phục)
            recovery_periods = []
            in_drawdown = False
            drawdown_start = 0
            
            for i in range(1, len(df)):
                if not in_drawdown and df['drawdown'].iloc[i] > 0.05:  # Ngưỡng 5% để xem là đang trong drawdown
                    in_drawdown = True
                    drawdown_start = i
                elif in_drawdown and df['drawdown'].iloc[i] < 0.01:  # Ngưỡng 1% để xem là đã hồi phục
                    in_drawdown = False
                    recovery_periods.append(i - drawdown_start)
            
            avg_recovery_period = np.mean(recovery_periods) if recovery_periods else 0
            
            # Tính các thống kê liên quan đến biến động drawdown
            dd_volatility = df['drawdown'].rolling(window=min(5, len(df))).std().mean()
            
            # Dự đoán drawdown trong tương lai
            if len(df) >= 10:
                # Sử dụng phân phối lịch sử để dự đoán
                dd_percentiles = np.percentile(df['drawdown'], [50, 75, 90, 95, 99])
                expected_max_dd = dd_percentiles[2]  # 90th percentile
            else:
                expected_max_dd = max_dd
            
            # Xác định mức độ rủi ro tổng thể
            if expected_max_dd > self.max_acceptable_drawdown:
                risk_level = "Cao"
            elif expected_max_dd > self.critical_threshold:
                risk_level = "Trung bình-cao"
            elif expected_max_dd > self.warning_threshold:
                risk_level = "Trung bình"
            else:
                risk_level = "Thấp"
            
            # Đưa ra khuyến nghị
            recommendations = []
            
            if expected_max_dd > self.max_acceptable_drawdown:
                recommendations.append("Giảm đáng kể kích thước vị thế (50% hoặc ít hơn)")
                recommendations.append("Xem xét tạm dừng giao dịch cho đến khi thị trường ổn định")
            elif expected_max_dd > self.critical_threshold:
                recommendations.append("Giảm kích thước vị thế (khoảng 30%)")
                recommendations.append("Sử dụng stop loss chặt chẽ hơn")
            elif expected_max_dd > self.warning_threshold:
                recommendations.append("Giảm kích thước vị thế nhẹ (khoảng 10-20%)")
                recommendations.append("Tăng cường quản lý rủi ro cho mỗi giao dịch")
            
            # Tạo một biểu đồ drawdown
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(df['timestamp'], df['balance'], label='Số dư')
            plt.title('Lịch sử số dư')
            plt.xlabel('Thời gian')
            plt.ylabel('Số dư')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(df['timestamp'], df['drawdown'] * 100, label='Drawdown (%)')
            plt.axhline(y=self.warning_threshold * 100, color='yellow', linestyle='--', label=f'Ngưỡng cảnh báo ({self.warning_threshold*100}%)')
            plt.axhline(y=self.critical_threshold * 100, color='orange', linestyle='--', label=f'Ngưỡng nguy hiểm ({self.critical_threshold*100}%)')
            plt.axhline(y=self.max_acceptable_drawdown * 100, color='red', linestyle='--', label=f'Ngưỡng tối đa ({self.max_acceptable_drawdown*100}%)')
            plt.title('Lịch sử Drawdown')
            plt.xlabel('Thời gian')
            plt.ylabel('Drawdown (%)')
            plt.legend()
            
            plt.tight_layout()
            
            # Chuyển biểu đồ thành base64 string để trả về
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            result = {
                "status": "success",
                "current_drawdown": self.current_drawdown,
                "max_drawdown": max_dd,
                "min_drawdown": min_dd,
                "avg_drawdown": avg_dd,
                "std_drawdown": std_dd,
                "drawdown_volatility": dd_volatility,
                "max_consecutive_drawdown": max_consecutive_dd,
                "recovery_count": recovery_count,
                "avg_recovery_period": avg_recovery_period,
                "expected_max_drawdown": expected_max_dd,
                "risk_level": risk_level,
                "in_recovery_mode": self.in_recovery_mode,
                "recovery_target_balance": self.recovery_target_balance if self.in_recovery_mode else None,
                "recovery_progress": (self.balance_history[-1] / self.recovery_target_balance) if self.in_recovery_mode else None,
                "recommendations": recommendations,
                "chart": chart_base64
            }
            
            self.logger.info(f"Đã phân tích drawdown: max={max_dd:.2%}, avg={avg_dd:.2%}, risk={risk_level}")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi phân tích drawdown: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi phân tích drawdown: {str(e)}"
                }
            }
    
    def get_position_size_recommendations(
        self,
        account_balance: float,
        normal_position_size: float,
        expected_win_rate: float = 0.5,
        expected_win_loss_ratio: float = 1.5
    ) -> Dict[str, Any]:
        """
        Đưa ra khuyến nghị về kích thước vị thế dựa trên drawdown.
        
        Args:
            account_balance: Số dư tài khoản hiện tại
            normal_position_size: Kích thước vị thế bình thường (không có drawdown)
            expected_win_rate: Tỷ lệ thắng kỳ vọng
            expected_win_loss_ratio: Tỷ lệ lợi nhuận/lỗ kỳ vọng
            
        Returns:
            Dict chứa các khuyến nghị về kích thước vị thế
        """
        try:
            # Tính kích thước vị thế điều chỉnh cho drawdown hiện tại
            position_size_factor = self.calculate_position_size_factor()
            adjusted_position_size = normal_position_size * position_size_factor
            
            # Tính kích thước vị thế theo Kelly Criterion
            kelly_result = self.risk_calculator.calculate_kelly_criterion(
                win_rate=expected_win_rate,
                win_loss_ratio=expected_win_loss_ratio,
                conservative_factor=0.5  # Sử dụng half-Kelly để thận trọng
            )
            
            if kelly_result.get("status") == "success":
                kelly_position_size = account_balance * kelly_result.get("conservative_kelly", 0.1)
                
                # Sử dụng kích thước nhỏ hơn giữa adjusted_position_size và kelly_position_size
                recommended_position_size = min(adjusted_position_size, kelly_position_size)
            else:
                recommended_position_size = adjusted_position_size
            
            # Tính max drawdown được chấp nhận cho vị thế này
            acceptable_loss_percent = self.max_acceptable_drawdown - self.current_drawdown
            max_loss_amount = account_balance * acceptable_loss_percent
            
            # Tính số lượng giao dịch trước khi hết hạn mức rủi ro
            if expected_win_rate > 0 and expected_win_loss_ratio > 0:
                expected_return_per_trade = expected_win_rate * expected_win_loss_ratio - (1 - expected_win_rate)
                if expected_return_per_trade > 0:
                    trades_to_recovery = 0  # Không cần hồi phục nếu kỳ vọng dương
                else:
                    trades_to_recovery = float('inf')  # Không thể hồi phục nếu kỳ vọng âm
            else:
                trades_to_recovery = float('inf')
            
            # Tạo các mức kích thước vị thế
            position_sizes = {
                "very_conservative": recommended_position_size * 0.5,
                "conservative": recommended_position_size * 0.7,
                "recommended": recommended_position_size,
                "aggressive": recommended_position_size * 1.2,  # Chỉ cho phép nếu drawdown thấp
            }
            
            # Nếu drawdown ở mức cao, không cho phép aggressive
            if self.current_drawdown >= self.warning_threshold:
                position_sizes["aggressive"] = "Không khuyến nghị khi drawdown cao"
            
            result = {
                "status": "success",
                "current_drawdown": self.current_drawdown,
                "drawdown_level": self.get_drawdown_level(self.current_drawdown),
                "position_size_factor": position_size_factor,
                "normal_position_size": normal_position_size,
                "recommended_position_size": recommended_position_size,
                "position_sizes": position_sizes,
                "max_acceptable_loss": max_loss_amount,
                "acceptable_loss_percent": acceptable_loss_percent,
                "kelly_criterion": kelly_result.get("kelly_fraction", 0),
                "kelly_position_size": kelly_position_size if kelly_result.get("status") == "success" else None,
                "expected_win_rate": expected_win_rate,
                "expected_win_loss_ratio": expected_win_loss_ratio,
                "in_recovery_mode": self.in_recovery_mode,
                "trades_to_recovery": trades_to_recovery
            }
            
            self.logger.info(f"Đã tính khuyến nghị kích thước vị thế: {recommended_position_size:.2f} (factor: {position_size_factor:.2f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính khuyến nghị kích thước vị thế: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính khuyến nghị kích thước vị thế: {str(e)}"
                }
            }
    
    def simulate_recovery(
        self,
        current_balance: float,
        expected_win_rate: float,
        expected_win_loss_ratio: float,
        position_size_percent: float,
        num_simulations: int = 1000,
        max_trades: int = 100
    ) -> Dict[str, Any]:
        """
        Mô phỏng quá trình hồi phục từ drawdown.
        
        Args:
            current_balance: Số dư hiện tại
            expected_win_rate: Tỷ lệ thắng kỳ vọng
            expected_win_loss_ratio: Tỷ lệ lợi nhuận/lỗ kỳ vọng
            position_size_percent: Phần trăm kích thước vị thế
            num_simulations: Số lần mô phỏng
            max_trades: Số giao dịch tối đa trong mô phỏng
            
        Returns:
            Dict chứa kết quả mô phỏng
        """
        try:
            # Tính số lần thắng và thua trong mỗi mô phỏng
            recovery_target = self.max_balance
            
            # Mảng lưu kết quả
            recovery_trades = []  # Số giao dịch cần để hồi phục
            final_balances = []  # Số dư cuối cùng
            success_count = 0  # Số mô phỏng thành công
            
            # Chạy mô phỏng
            for _ in range(num_simulations):
                balance = current_balance
                for trade in range(max_trades):
                    # Tính kích thước vị thế
                    position_size = balance * position_size_percent
                    
                    # Xác định kết quả giao dịch
                    is_win = np.random.random() < expected_win_rate
                    
                    # Cập nhật số dư
                    if is_win:
                        balance += position_size * expected_win_loss_ratio
                    else:
                        balance -= position_size
                    
                    # Kiểm tra nếu đã hồi phục
                    if balance >= recovery_target:
                        recovery_trades.append(trade + 1)
                        success_count += 1
                        break
                
                # Lưu số dư cuối cùng
                final_balances.append(balance)
                
                # Nếu không hồi phục sau max_trades, đánh dấu là không thành công
                if balance < recovery_target:
                    recovery_trades.append(float('inf'))
            
            # Tính thống kê
            success_rate = success_count / num_simulations
            
            # Tính số giao dịch trung bình để hồi phục (chỉ tính các mô phỏng thành công)
            successful_recovery_trades = [t for t in recovery_trades if t != float('inf')]
            avg_recovery_trades = np.mean(successful_recovery_trades) if successful_recovery_trades else float('inf')
            
            # Tính các phân vị cho số dư cuối cùng
            final_balance_percentiles = np.percentile(final_balances, [5, 25, 50, 75, 95])
            
            # Tính xác suất phá sản (số dư <= 0)
            ruin_probability = sum(1 for b in final_balances if b <= 0) / num_simulations
            
            # Vẽ biểu đồ phân phối số dư cuối cùng
            plt.figure(figsize=(10, 6))
            plt.hist(final_balances, bins=30, alpha=0.7)
            plt.axvline(x=current_balance, color='red', linestyle='--', label=f'Số dư hiện tại ({current_balance:.2f})')
            plt.axvline(x=recovery_target, color='green', linestyle='--', label=f'Mục tiêu hồi phục ({recovery_target:.2f})')
            plt.title('Phân phối số dư sau {max_trades} giao dịch')
            plt.xlabel('Số dư cuối cùng')
            plt.ylabel('Số lượng mô phỏng')
            plt.legend()
            
            # Chuyển biểu đồ thành base64 string để trả về
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            result = {
                "status": "success",
                "success_rate": success_rate,
                "avg_recovery_trades": avg_recovery_trades,
                "median_recovery_trades": np.median(successful_recovery_trades) if successful_recovery_trades else float('inf'),
                "ruin_probability": ruin_probability,
                "final_balance_mean": np.mean(final_balances),
                "final_balance_median": np.median(final_balances),
                "final_balance_percentiles": {
                    "p5": final_balance_percentiles[0],
                    "p25": final_balance_percentiles[1],
                    "p50": final_balance_percentiles[2],
                    "p75": final_balance_percentiles[3],
                    "p95": final_balance_percentiles[4]
                },
                "recovery_target": recovery_target,
                "current_balance": current_balance,
                "current_drawdown": self.current_drawdown,
                "max_trades": max_trades,
                "num_simulations": num_simulations,
                "expected_win_rate": expected_win_rate,
                "expected_win_loss_ratio": expected_win_loss_ratio,
                "position_size_percent": position_size_percent,
                "chart": chart_base64
            }
            
            self.logger.info(f"Đã mô phỏng hồi phục từ drawdown: success_rate={success_rate:.2%}, avg_trades={avg_recovery_trades:.1f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi mô phỏng hồi phục: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi mô phỏng hồi phục: {str(e)}"
                }
            }
    
    def calculate_max_safe_position_size(
        self,
        current_balance: float,
        risk_per_trade: float,
        stop_loss_percent: float,
        max_consecutive_losses: int = 5
    ) -> Dict[str, Any]:
        """
        Tính kích thước vị thế tối đa an toàn dựa trên khả năng chịu đựng chuỗi thua lỗ.
        
        Args:
            current_balance: Số dư hiện tại
            risk_per_trade: Phần trăm rủi ro trên mỗi giao dịch
            stop_loss_percent: Phần trăm dừng lỗ
            max_consecutive_losses: Số lần thua liên tiếp tối đa cần chịu đựng
            
        Returns:
            Dict chứa thông tin kích thước vị thế an toàn
        """
        try:
            # Tính lỗ tối đa có thể chấp nhận đươc
            available_risk = self.max_acceptable_drawdown - self.current_drawdown
            max_loss_amount = current_balance * available_risk
            
            # Tính lỗ trên mỗi giao dịch
            loss_per_trade = current_balance * risk_per_trade
            
            # Tính tổng lỗ sau max_consecutive_losses giao dịch
            total_loss = loss_per_trade * max_consecutive_losses
            
            # Điều chỉnh nếu vượt quá max_loss_amount
            if total_loss > max_loss_amount:
                adjusted_loss_per_trade = max_loss_amount / max_consecutive_losses
                adjusted_risk_per_trade = adjusted_loss_per_trade / current_balance
            else:
                adjusted_risk_per_trade = risk_per_trade
            
            # Tính kích thước vị thế dựa trên rủi ro điều chỉnh
            position_size = (adjusted_risk_per_trade * current_balance) / stop_loss_percent
            
            result = {
                "status": "success",
                "max_safe_position_size": position_size,
                "current_balance": current_balance,
                "current_drawdown": self.current_drawdown,
                "available_risk": available_risk,
                "max_loss_amount": max_loss_amount,
                "original_risk_per_trade": risk_per_trade,
                "adjusted_risk_per_trade": adjusted_risk_per_trade,
                "stop_loss_percent": stop_loss_percent,
                "max_consecutive_losses": max_consecutive_losses,
                "position_size_percent": position_size / current_balance
            }
            
            self.logger.info(f"Đã tính kích thước vị thế an toàn: {position_size:.2f} ({result['position_size_percent']:.2%} số dư)")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính kích thước vị thế an toàn: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính kích thước vị thế an toàn: {str(e)}"
                }
            }
    
    def save_drawdown_history(self, file_path: str) -> bool:
        """
        Lưu lịch sử drawdown vào file.
        
        Args:
            file_path: Đường dẫn file
            
        Returns:
            True nếu lưu thành công, False nếu không
        """
        try:
            # Tạo DataFrame từ lịch sử
            df = pd.DataFrame({
                'timestamp': self.timestamp_history,
                'balance': self.balance_history,
                'drawdown': self.drawdown_history
            })
            
            # Lưu vào file CSV
            df.to_csv(file_path, index=False)
            
            self.logger.info(f"Đã lưu lịch sử drawdown vào {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu lịch sử drawdown: {str(e)}")
            return False
    
    def load_drawdown_history(self, file_path: str) -> bool:
        """
        Tải lịch sử drawdown từ file.
        
        Args:
            file_path: Đường dẫn file
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            # Đọc file CSV
            df = pd.read_csv(file_path)
            
            # Chuyển đổi timestamp từ string sang datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Cập nhật lịch sử
            self.timestamp_history = df['timestamp'].tolist()
            self.balance_history = df['balance'].tolist()
            self.drawdown_history = df['drawdown'].tolist()
            
            # Cập nhật các biến liên quan
            self.max_balance = max(self.balance_history)
            self.current_drawdown = self.drawdown_history[-1] if self.drawdown_history else 0.0
            
            self.logger.info(f"Đã tải lịch sử drawdown từ {file_path}: {len(self.balance_history)} điểm dữ liệu")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải lịch sử drawdown: {str(e)}")
            return False