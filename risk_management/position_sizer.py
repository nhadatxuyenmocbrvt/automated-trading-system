"""
Định kích thước vị thế.
File này định nghĩa các phương pháp để tính toán kích thước vị thế tối ưu
dựa trên các chiến lược quản lý rủi ro khác nhau.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import MAX_RISK_PER_TRADE, MAX_LEVERAGE, ErrorCode
from risk_management.risk_calculator import RiskCalculator

class PositionSizer:
    """
    Lớp định kích thước vị thế.
    Cung cấp các phương pháp khác nhau để tính toán kích thước vị thế tối ưu
    nhằm quản lý rủi ro và tối đa hóa lợi nhuận kỳ vọng.
    """
    
    def __init__(
        self,
        account_balance: float,
        max_risk_per_trade: float = MAX_RISK_PER_TRADE,
        max_leverage: float = MAX_LEVERAGE,
        risk_calculator: Optional[RiskCalculator] = None,
        leverage_step: float = 0.1,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo định kích thước vị thế.
        
        Args:
            account_balance: Số dư tài khoản
            max_risk_per_trade: Rủi ro tối đa cho mỗi giao dịch (tỷ lệ phần trăm của số dư)
            max_leverage: Đòn bẩy tối đa
            risk_calculator: Đối tượng RiskCalculator (tùy chọn)
            leverage_step: Bước nhảy đòn bẩy khi tính toán
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("position_sizer")
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_leverage = max_leverage
        self.leverage_step = leverage_step
        
        # Khởi tạo risk calculator nếu không được cung cấp
        if risk_calculator is None:
            from risk_management.risk_calculator import RiskCalculator
            self.risk_calculator = RiskCalculator()
        else:
            self.risk_calculator = risk_calculator
            
        self.logger.info(f"Đã khởi tạo PositionSizer với account_balance={account_balance}, max_risk={max_risk_per_trade}, max_leverage={max_leverage}")
    
    def update_account_balance(self, new_balance: float) -> None:
        """
        Cập nhật số dư tài khoản.
        
        Args:
            new_balance: Số dư mới
        """
        self.account_balance = new_balance
        self.logger.debug(f"Đã cập nhật số dư tài khoản: {new_balance}")
    
    def calculate_position_size_fixed_risk(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_percent: Optional[float] = None,
        leverage: float = 1.0,
        fee_rate: float = 0.001,
        slippage_percent: float = 0.001
    ) -> Dict[str, Any]:
        """
        Tính kích thước vị thế dựa trên rủi ro cố định.
        
        Args:
            entry_price: Giá vào lệnh
            stop_loss_price: Giá dừng lỗ
            risk_percent: Phần trăm rủi ro (% số dư), None để sử dụng max_risk_per_trade
            leverage: Đòn bẩy
            fee_rate: Tỷ lệ phí giao dịch
            slippage_percent: Tỷ lệ trượt giá
            
        Returns:
            Dict chứa thông tin kích thước vị thế và các thông số liên quan
        """
        if risk_percent is None:
            risk_percent = self.max_risk_per_trade
        
        # Kiểm tra tham số
        if risk_percent <= 0 or risk_percent > 1:
            self.logger.warning(f"Tỷ lệ rủi ro không hợp lệ: {risk_percent}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Tỷ lệ rủi ro phải > 0 và <= 1"
                }
            }
        
        if entry_price <= 0 or stop_loss_price <= 0:
            self.logger.warning(f"Giá không hợp lệ: entry={entry_price}, stop_loss={stop_loss_price}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Giá phải lớn hơn 0"
                }
            }
        
        if leverage <= 0 or leverage > self.max_leverage:
            self.logger.warning(f"Đòn bẩy không hợp lệ: {leverage}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": f"Đòn bẩy phải > 0 và <= {self.max_leverage}"
                }
            }
        
        try:
            # Xác định loại lệnh (long/short)
            is_long = entry_price < stop_loss_price
            
            # Tính khoảng cách dừng lỗ (%)
            if is_long:
                stop_loss_distance_percent = abs(entry_price - stop_loss_price) / entry_price
            else:
                stop_loss_distance_percent = abs(entry_price - stop_loss_price) / entry_price
            
            # Tính tổng chi phí (phí + trượt giá) (%)
            total_cost_percent = fee_rate * 2 + slippage_percent  # phí vào, phí ra, trượt giá
            
            # Tính số tiền rủi ro
            risk_amount = self.account_balance * risk_percent
            
            # Tính kích thước vị thế (có tính đến đòn bẩy)
            position_size = (risk_amount / (stop_loss_distance_percent + total_cost_percent)) * leverage
            
            # Giới hạn kích thước vị thế không vượt quá số dư * đòn bẩy
            max_position_size = self.account_balance * leverage
            if position_size > max_position_size:
                position_size = max_position_size
                self.logger.warning(f"Kích thước vị thế đã bị giới hạn bởi số dư * đòn bẩy: {max_position_size}")
            
            # Tính số lượng coin tương ứng
            coin_amount = position_size / entry_price
            
            # Tính margin cần thiết
            required_margin = position_size / leverage
            
            # Tính rủi ro thực tế
            actual_risk_amount = position_size * (stop_loss_distance_percent + total_cost_percent) / leverage
            actual_risk_percent = actual_risk_amount / self.account_balance
            
            result = {
                "status": "success",
                "position_size": position_size,
                "coin_amount": coin_amount,
                "required_margin": required_margin,
                "risk_amount": risk_amount,
                "actual_risk_amount": actual_risk_amount,
                "actual_risk_percent": actual_risk_percent,
                "stop_loss_distance_percent": stop_loss_distance_percent,
                "total_cost_percent": total_cost_percent,
                "leverage": leverage,
                "is_long": is_long
            }
            
            self.logger.info(f"Đã tính kích thước vị thế: {position_size:.2f} ({coin_amount:.6f} coin)")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính kích thước vị thế: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính kích thước vị thế: {str(e)}"
                }
            }
    
    def calculate_position_size_risk_reward(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        risk_percent: Optional[float] = None,
        min_risk_reward_ratio: float = 2.0,
        leverage: float = 1.0,
        fee_rate: float = 0.001,
        slippage_percent: float = 0.001
    ) -> Dict[str, Any]:
        """
        Tính kích thước vị thế dựa trên tỷ lệ rủi ro/phần thưởng.
        
        Args:
            entry_price: Giá vào lệnh
            stop_loss_price: Giá dừng lỗ
            take_profit_price: Giá chốt lời
            risk_percent: Phần trăm rủi ro (% số dư), None để sử dụng max_risk_per_trade
            min_risk_reward_ratio: Tỷ lệ rủi ro/phần thưởng tối thiểu
            leverage: Đòn bẩy
            fee_rate: Tỷ lệ phí giao dịch
            slippage_percent: Tỷ lệ trượt giá
            
        Returns:
            Dict chứa thông tin kích thước vị thế và các thông số liên quan
        """
        if risk_percent is None:
            risk_percent = self.max_risk_per_trade
        
        # Kiểm tra tham số
        if min_risk_reward_ratio <= 0:
            self.logger.warning(f"Tỷ lệ rủi ro/phần thưởng không hợp lệ: {min_risk_reward_ratio}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Tỷ lệ rủi ro/phần thưởng phải > 0"
                }
            }
        
        try:
            # Xác định loại lệnh (long/short)
            is_long = entry_price < stop_loss_price
            
            # Tính khoảng cách dừng lỗ và chốt lời (%)
            if is_long:
                stop_loss_distance_percent = abs(entry_price - stop_loss_price) / entry_price
                take_profit_distance_percent = abs(take_profit_price - entry_price) / entry_price
            else:
                stop_loss_distance_percent = abs(entry_price - stop_loss_price) / entry_price
                take_profit_distance_percent = abs(entry_price - take_profit_price) / entry_price
            
            # Tính tỷ lệ rủi ro/phần thưởng
            risk_reward_ratio = take_profit_distance_percent / stop_loss_distance_percent
            
            # Kiểm tra tỷ lệ rủi ro/phần thưởng
            if risk_reward_ratio < min_risk_reward_ratio:
                self.logger.warning(f"Tỷ lệ rủi ro/phần thưởng ({risk_reward_ratio:.2f}) < tối thiểu ({min_risk_reward_ratio})")
                
                # Tính kích thước vị thế theo tỷ lệ
                scaling_factor = risk_reward_ratio / min_risk_reward_ratio
                reduced_risk_percent = risk_percent * scaling_factor
                
                self.logger.info(f"Giảm rủi ro từ {risk_percent:.2%} xuống {reduced_risk_percent:.2%} do tỷ lệ R/R thấp")
                
                # Tính kích thước vị thế với rủi ro giảm
                result = self.calculate_position_size_fixed_risk(
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    risk_percent=reduced_risk_percent,
                    leverage=leverage,
                    fee_rate=fee_rate,
                    slippage_percent=slippage_percent
                )
                
                # Thêm thông tin về tỷ lệ rủi ro/phần thưởng
                if result["status"] == "success":
                    result["risk_reward_ratio"] = risk_reward_ratio
                    result["min_risk_reward_ratio"] = min_risk_reward_ratio
                    result["scaling_factor"] = scaling_factor
                    result["original_risk_percent"] = risk_percent
                    result["reduced_risk_percent"] = reduced_risk_percent
                    result["take_profit_distance_percent"] = take_profit_distance_percent
                
                return result
            
            # Tính kích thước vị thế bình thường
            result = self.calculate_position_size_fixed_risk(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                risk_percent=risk_percent,
                leverage=leverage,
                fee_rate=fee_rate,
                slippage_percent=slippage_percent
            )
            
            # Thêm thông tin về tỷ lệ rủi ro/phần thưởng
            if result["status"] == "success":
                result["risk_reward_ratio"] = risk_reward_ratio
                result["min_risk_reward_ratio"] = min_risk_reward_ratio
                result["take_profit_distance_percent"] = take_profit_distance_percent
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính kích thước vị thế R/R: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính kích thước vị thế R/R: {str(e)}"
                }
            }
    
    def calculate_position_size_kelly(
        self,
        win_rate: float,
        reward_risk_ratio: float,
        risk_fraction: float = 1.0,
        max_risk_percent: Optional[float] = None,
        entry_price: float = 0.0,
        leverage: float = 1.0
    ) -> Dict[str, Any]:
        """
        Tính kích thước vị thế dựa trên tiêu chí Kelly.
        
        Args:
            win_rate: Tỷ lệ thắng (0-1)
            reward_risk_ratio: Tỷ lệ phần thưởng/rủi ro
            risk_fraction: Phần rủi ro của Kelly (thường <1 để thận trọng)
            max_risk_percent: Giới hạn rủi ro tối đa (% số dư)
            entry_price: Giá vào lệnh (nếu muốn tính số lượng coin)
            leverage: Đòn bẩy
            
        Returns:
            Dict chứa thông tin kích thước vị thế theo Kelly
        """
        if max_risk_percent is None:
            max_risk_percent = self.max_risk_per_trade
        
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
        
        if reward_risk_ratio <= 0:
            self.logger.warning(f"Tỷ lệ phần thưởng/rủi ro không hợp lệ: {reward_risk_ratio}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Tỷ lệ phần thưởng/rủi ro phải > 0"
                }
            }
        
        try:
            # Tính Kelly Criterion
            # f* = (p * b - (1 - p)) / b
            # p: xác suất thắng
            # b: tỷ lệ phần thưởng/rủi ro
            # f*: phần tối ưu của vốn để đặt cược
            kelly_fraction = (win_rate * reward_risk_ratio - (1 - win_rate)) / reward_risk_ratio
            
            # Điều chỉnh theo risk_fraction để thận trọng hơn
            adjusted_kelly = kelly_fraction * risk_fraction
            
            # Giới hạn tối đa
            if adjusted_kelly < 0:
                # Kelly âm nghĩa là không nên giao dịch
                kelly_position_size = 0
                self.logger.warning(f"Kelly âm ({kelly_fraction:.4f}): không nên giao dịch")
                actual_risk_percent = 0
            else:
                # Giới hạn theo max_risk_percent
                actual_risk_percent = min(adjusted_kelly, max_risk_percent)
                kelly_position_size = self.account_balance * actual_risk_percent * leverage
            
            # Tính số lượng coin nếu có entry_price
            coin_amount = kelly_position_size / entry_price if entry_price > 0 else 0
            
            # Tính margin cần thiết
            required_margin = kelly_position_size / leverage if kelly_position_size > 0 else 0
            
            result = {
                "status": "success",
                "position_size": kelly_position_size,
                "coin_amount": coin_amount,
                "required_margin": required_margin,
                "kelly_fraction": kelly_fraction,
                "adjusted_kelly": adjusted_kelly,
                "actual_risk_percent": actual_risk_percent,
                "risk_fraction": risk_fraction,
                "max_risk_percent": max_risk_percent,
                "win_rate": win_rate,
                "reward_risk_ratio": reward_risk_ratio,
                "leverage": leverage
            }
            
            self.logger.info(f"Đã tính kích thước vị thế Kelly: {kelly_position_size:.2f} ({actual_risk_percent:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính kích thước vị thế Kelly: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính kích thước vị thế Kelly: {str(e)}"
                }
            }
    
    def calculate_position_size_volatility(
        self,
        entry_price: float,
        atr_value: float,
        atr_multiplier: float = 2.0,
        risk_percent: Optional[float] = None,
        leverage: float = 1.0,
        stop_loss_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Tính kích thước vị thế dựa trên biến động (ATR).
        
        Args:
            entry_price: Giá vào lệnh
            atr_value: Giá trị ATR (Average True Range)
            atr_multiplier: Hệ số nhân ATR cho dừng lỗ
            risk_percent: Phần trăm rủi ro (% số dư), None để sử dụng max_risk_per_trade
            leverage: Đòn bẩy
            stop_loss_price: Giá dừng lỗ (tùy chọn, nếu không sẽ tính bằng ATR)
            
        Returns:
            Dict chứa thông tin kích thước vị thế dựa trên ATR
        """
        if risk_percent is None:
            risk_percent = self.max_risk_per_trade
        
        # Kiểm tra tham số
        if atr_value <= 0:
            self.logger.warning(f"Giá trị ATR không hợp lệ: {atr_value}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Giá trị ATR phải > 0"
                }
            }
        
        try:
            # Xác định stop loss nếu không được cung cấp
            calculated_stop_loss = False
            
            if stop_loss_price is None:
                # Tính stop loss dựa trên ATR
                # Giả định đây là lệnh long, nếu thực tế sẽ được điều chỉnh sau
                stop_loss_price = entry_price - (atr_value * atr_multiplier)
                calculated_stop_loss = True
            
            # Xác định loại lệnh (long/short)
            is_long = entry_price < stop_loss_price
            
            # Điều chỉnh stop loss nếu cần
            if calculated_stop_loss and not is_long:
                # Nếu là short, tính lại stop loss
                stop_loss_price = entry_price + (atr_value * atr_multiplier)
            
            # Tính kích thước vị thế dựa trên fixed risk
            result = self.calculate_position_size_fixed_risk(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                risk_percent=risk_percent,
                leverage=leverage
            )
            
            # Thêm thông tin về ATR
            if result["status"] == "success":
                result["atr_value"] = atr_value
                result["atr_multiplier"] = atr_multiplier
                result["calculated_stop_loss"] = calculated_stop_loss
                
                # Tính khoảng cách stop loss theo ATR
                result["stop_loss_atr_distance"] = abs(entry_price - stop_loss_price) / atr_value
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính kích thước vị thế theo ATR: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính kích thước vị thế theo ATR: {str(e)}"
                }
            }
    
    def optimize_position_leverage(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        risk_percent: Optional[float] = None,
        min_risk_reward_ratio: float = 2.0,
        max_position_size: Optional[float] = None,
        fee_rate: float = 0.001
    ) -> Dict[str, Any]:
        """
        Tối ưu hóa đòn bẩy cho vị thế.
        
        Args:
            entry_price: Giá vào lệnh
            stop_loss_price: Giá dừng lỗ
            take_profit_price: Giá chốt lời
            risk_percent: Phần trăm rủi ro (% số dư), None để sử dụng max_risk_per_trade
            min_risk_reward_ratio: Tỷ lệ rủi ro/phần thưởng tối thiểu
            max_position_size: Kích thước vị thế tối đa (tùy chọn)
            fee_rate: Tỷ lệ phí giao dịch
            
        Returns:
            Dict chứa thông tin đòn bẩy tối ưu và kích thước vị thế
        """
        if risk_percent is None:
            risk_percent = self.max_risk_per_trade
        
        if max_position_size is None:
            max_position_size = self.account_balance * self.max_leverage
        
        # Danh sách kết quả cho các mức đòn bẩy
        leverage_results = []
        
        # Thử các mức đòn bẩy khác nhau
        for leverage in np.arange(1.0, self.max_leverage + self.leverage_step, self.leverage_step):
            result = self.calculate_position_size_risk_reward(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                risk_percent=risk_percent,
                min_risk_reward_ratio=min_risk_reward_ratio,
                leverage=leverage,
                fee_rate=fee_rate
            )
            
            if result["status"] == "success":
                # Kiểm tra kích thước vị thế
                if result["position_size"] <= max_position_size:
                    # Tính điểm tối ưu
                    # Chúng ta muốn tối đa hóa: expected_return * risk_reward_ratio / required_margin
                    expected_return = result["position_size"] * result["risk_reward_ratio"] * result["actual_risk_percent"]
                    optimization_score = expected_return * result["risk_reward_ratio"] / result["required_margin"]
                    
                    result["optimization_score"] = optimization_score
                    leverage_results.append(result)
        
        # Nếu không có kết quả nào
        if not leverage_results:
            self.logger.warning("Không tìm được đòn bẩy tối ưu")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Không tìm được đòn bẩy tối ưu"
                }
            }
        
        # Sắp xếp theo điểm tối ưu
        leverage_results.sort(key=lambda x: x["optimization_score"], reverse=True)
        
        # Lấy kết quả tốt nhất
        best_result = leverage_results[0]
        best_result["leverages_tested"] = len(leverage_results)
        
        self.logger.info(f"Đã tìm thấy đòn bẩy tối ưu: {best_result['leverage']:.1f}x, score: {best_result['optimization_score']:.4f}")
        
        return best_result
    
    def calculate_max_drawdown_position_size(
        self,
        max_drawdown_percent: float,
        expected_win_rate: float,
        expected_loss_percent: float,
        max_consecutive_losses: int = 5,
        leverage: float = 1.0
    ) -> Dict[str, Any]:
        """
        Tính kích thước vị thế dựa trên khả năng chịu đựng drawdown tối đa.
        
        Args:
            max_drawdown_percent: Drawdown tối đa có thể chấp nhận (%)
            expected_win_rate: Tỷ lệ thắng kỳ vọng (0-1)
            expected_loss_percent: Tỷ lệ thua trung bình mỗi giao dịch
            max_consecutive_losses: Số lượng thua liên tiếp tối đa cần chịu đựng
            leverage: Đòn bẩy
            
        Returns:
            Dict chứa thông tin kích thước vị thế theo khả năng chịu drawdown
        """
        # Kiểm tra tham số
        if max_drawdown_percent <= 0 or max_drawdown_percent >= 1:
            self.logger.warning(f"Drawdown tối đa không hợp lệ: {max_drawdown_percent}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.INVALID_PARAMETER.value,
                    "message": "Drawdown tối đa phải > 0 và < 1"
                }
            }
        
        try:
            # Tính xác suất thua liên tiếp
            consecutive_loss_probability = (1 - expected_win_rate) ** max_consecutive_losses
            
            # Tính drawdown từ max_consecutive_losses
            drawdown_from_consecutive = 1 - ((1 - expected_loss_percent) ** max_consecutive_losses)
            
            # Tính risk_percent để đạt được drawdown mong muốn
            if drawdown_from_consecutive > 0:
                safe_risk_percent = max_drawdown_percent / drawdown_from_consecutive
            else:
                safe_risk_percent = self.max_risk_per_trade
            
            # Giới hạn tối đa
            risk_percent = min(safe_risk_percent, self.max_risk_per_trade)
            
            # Tính kích thước vị thế
            position_size = self.account_balance * risk_percent * leverage
            
            # Tính margin cần thiết
            required_margin = position_size / leverage
            
            result = {
                "status": "success",
                "position_size": position_size,
                "risk_percent": risk_percent,
                "required_margin": required_margin,
                "max_drawdown_percent": max_drawdown_percent,
                "expected_win_rate": expected_win_rate,
                "expected_loss_percent": expected_loss_percent,
                "max_consecutive_losses": max_consecutive_losses,
                "consecutive_loss_probability": consecutive_loss_probability,
                "drawdown_from_consecutive": drawdown_from_consecutive,
                "leverage": leverage
            }
            
            self.logger.info(f"Đã tính kích thước vị thế theo drawdown: {position_size:.2f} ({risk_percent:.2%})")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tính kích thước vị thế theo drawdown: {str(e)}")
            return {
                "status": "error",
                "error": {
                    "code": ErrorCode.UNKNOWN_ERROR.value,
                    "message": f"Lỗi khi tính kích thước vị thế theo drawdown: {str(e)}"
                }
            }