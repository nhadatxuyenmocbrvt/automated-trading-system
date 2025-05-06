"""
Hồ sơ rủi ro tích cực.
File này định nghĩa lớp AggressiveProfile thực hiện các chiến lược
quản lý rủi ro với mức chấp nhận rủi ro cao để đạt lợi nhuận lớn.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import PositionSide, OrderStatus
from risk_management.risk_calculator import RiskCalculator
from risk_management.position_sizer import PositionSizer
from risk_management.stop_loss import StopLossManager
from risk_management.take_profit import TakeProfitManager
from risk_management.drawdown_manager import DrawdownManager

class AggressiveProfile:
    """
    Lớp hồ sơ rủi ro tích cực.
    Định nghĩa các chiến lược quản lý rủi ro cho nhà đầu tư chấp nhận rủi ro cao,
    ưu tiên tối đa lợi nhuận và cơ hội tăng trưởng mạnh.
    """
    
    def __init__(
        self,
        capital: float = 10000.0,
        risk_per_trade: float = 0.03,  # 3% mỗi giao dịch
        max_drawdown: float = 0.25,    # 25% drawdown tối đa
        leverage_max: float = 5.0,     # Đòn bẩy tối đa 5x
        portfolio_volatility_target: float = 0.25,  # 25% biến động danh mục mục tiêu
        correlation_threshold: float = 0.8,    # Ngưỡng tương quan
        max_position_weight: float = 0.35,     # 35% tỷ trọng tối đa cho một tài sản
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo hồ sơ rủi ro tích cực.
        
        Args:
            capital: Vốn ban đầu
            risk_per_trade: Phần trăm rủi ro cho mỗi giao dịch
            max_drawdown: Phần trăm drawdown tối đa cho phép
            leverage_max: Đòn bẩy tối đa
            portfolio_volatility_target: Biến động danh mục mục tiêu
            correlation_threshold: Ngưỡng tương quan để xác định vị thế liên quan
            max_position_weight: Tỷ trọng tối đa cho một tài sản
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("aggressive_profile")
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.leverage_max = leverage_max
        self.portfolio_volatility_target = portfolio_volatility_target
        self.correlation_threshold = correlation_threshold
        self.max_position_weight = max_position_weight
        
        # Thiết lập các tham số mặc định cho hồ sơ tích cực
        self.asset_allocation = {
            "BTC": 0.40,     # Bitcoin: 40%
            "ETH": 0.30,     # Ethereum: 30%
            "STABLES": 0.05, # Stablecoin: 5%
            "OTHERS": 0.25   # Các altcoin khác: 25%
        }
        
        # Khởi tạo các thành phần quản lý rủi ro
        self._init_risk_components()
        
        self.logger.info(f"Đã khởi tạo hồ sơ rủi ro tích cực với risk_per_trade={risk_per_trade:.1%}, max_drawdown={max_drawdown:.1%}, leverage_max={leverage_max}x")
    
    def _init_risk_components(self) -> None:
        """
        Khởi tạo các thành phần quản lý rủi ro với tham số phù hợp cho hồ sơ tích cực.
        """
        # Position Sizer - chấp nhận rủi ro cao để tối đa lợi nhuận
        self.position_sizer = PositionSizer(
            risk_per_trade=self.risk_per_trade,
            position_sizing_method="risk_percentage",
            max_position_size_percent=0.20,  # Giới hạn 20% vốn cho mỗi vị thế
            min_position_size=0.001,         # Size tối thiểu
            use_kelly_criterion=True,        # Sử dụng công thức Kelly
            kelly_fraction=0.7               # Sử dụng 70% kết quả Kelly (để giảm rủi ro)
        )
        
        # Stop Loss Manager - stop loss rộng hơn để tránh bị stopped out sớm
        self.stop_loss_manager = StopLossManager(
            max_risk_percent=self.risk_per_trade,
            trailing_stop_enabled=True,
            trailing_stop_activation_percent=0.03,  # Kích hoạt trailing stop sau khi lãi 3%
            trailing_stop_distance_percent=0.02,    # Trailing stop ở khoảng cách 2%
            atr_multiplier=2.0                      # Sử dụng 2x ATR cho stop loss
        )
        
        # Take Profit Manager - để lợi nhuận chạy xa hơn
        self.take_profit_manager = TakeProfitManager(
            risk_reward_ratio=3.0,                   # Tỷ lệ R:R 3:1
            partial_take_profit_enabled=True,        # Bật take profit một phần
            partial_take_profit_levels=[1.5, 3.0, 5.0],  # Take tại 1.5R, 3.0R, 5.0R
            partial_take_profit_percents=[0.25, 0.25, 0.25]  # Đóng 25% tại mỗi mức
        )
        
        # Drawdown Manager - chấp nhận drawdown lớn hơn trước khi phản ứng
        self.drawdown_manager = DrawdownManager(
            max_drawdown_percent=self.max_drawdown,
            position_reduction_schedule=[
                (0.15, 0.25),  # Giảm 25% vị thế khi drawdown 15%
                (0.20, 0.5),   # Giảm 50% vị thế khi drawdown 20%
                (0.22, 0.75)   # Giảm 75% vị thế khi drawdown 22%
            ],
            cooldown_period_days=3,            # Cooldown 3 ngày sau khi kích hoạt
            emergency_action="reduce_leverage"  # Giảm đòn bẩy trong trường hợp khẩn cấp thay vì đóng vị thế
        )
        
        # Risk Calculator
        self.risk_calculator = RiskCalculator()
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        current_capital: float,
        current_holdings: Dict[str, Any] = None,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Tính toán kích thước vị thế theo hồ sơ tích cực.
        
        Args:
            symbol: Mã tài sản
            entry_price: Giá vào
            stop_loss_price: Giá stop loss
            current_capital: Vốn hiện tại
            current_holdings: Danh sách các vị thế hiện tại
            volatility: Biến động của tài sản (nếu có)
            win_rate: Tỷ lệ thắng lịch sử (nếu có)
            avg_win_loss_ratio: Tỷ lệ thắng/thua trung bình (nếu có)
            
        Returns:
            Dict chứa thông tin kích thước vị thế và các thông số liên quan
        """
        # Kiểm tra giá và stop loss
        if entry_price <= 0 or stop_loss_price <= 0:
            self.logger.warning(f"Giá không hợp lệ: entry={entry_price}, stop_loss={stop_loss_price}")
            return {
                "status": "error",
                "message": "Giá không hợp lệ",
                "position_size": 0.0
            }
        
        # Tính toán position size theo số lượng vốn có nguy cơ
        risk_amount = current_capital * self.risk_per_trade
        
        # Tính toán khoảng cách risk
        if entry_price > stop_loss_price:  # Long position
            risk_per_unit = entry_price - stop_loss_price
            side = "long"
        else:  # Short position
            risk_per_unit = stop_loss_price - entry_price
            side = "short"
        
        # Tránh chia cho 0
        if risk_per_unit <= 0:
            self.logger.warning(f"Khoảng cách risk không hợp lệ: {risk_per_unit}")
            return {
                "status": "error",
                "message": "Khoảng cách risk không hợp lệ",
                "position_size": 0.0
            }
        
        # Tính basic position size
        position_size = risk_amount / risk_per_unit
        
        # Điều chỉnh position size dựa trên biến động cao - profile tích cực thậm chí tăng size khi biến động cao
        if volatility is not None and volatility > 0:
            # Điều chỉnh position size theo biến động - tăng size khi biến động phù hợp cho cơ hội
            if volatility > 0.08:  # Biến động rất cao > 8%
                volatility_factor = 0.85  # Giảm 15%
            elif volatility > 0.05 and volatility <= 0.08:  # Biến động cao 5-8%
                volatility_factor = 1.1  # Tăng 10%
            elif volatility > 0.03 and volatility <= 0.05:  # Biến động vừa phải 3-5%
                volatility_factor = 1.2  # Tăng 20%
            else:  # Biến động thấp <= 3%
                volatility_factor = 1.0  # Giữ nguyên
                
            position_size *= volatility_factor
            self.logger.info(f"Điều chỉnh position size theo biến động {volatility:.1%}: factor={volatility_factor}")
        
        # Điều chỉnh theo công thức Kelly nếu có thông tin win rate & avg win/loss ratio
        if win_rate is not None and avg_win_loss_ratio is not None and win_rate > 0 and avg_win_loss_ratio > 0:
            # Công thức Kelly: f* = (p*b - q)/b, trong đó p=win_rate, q=1-p, b=avg_win_loss_ratio
            kelly_fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
            
            # Giới hạn kết quả Kelly (hệ số an toàn 0.7)
            kelly_fraction = max(0, min(kelly_fraction * 0.7, 0.5))
            
            kelly_position_size = current_capital * kelly_fraction / risk_per_unit
            
            # Lấy trung bình giữa position size thông thường và dựa theo Kelly
            position_size = (position_size + kelly_position_size) / 2
            
            self.logger.info(f"Điều chỉnh position size theo Kelly: win_rate={win_rate:.2f}, win_loss_ratio={avg_win_loss_ratio:.2f}, kelly={kelly_fraction:.2f}")
        
        # Áp dụng đòn bẩy (profile tích cực sử dụng đòn bẩy)
        max_leverage = min(self.leverage_max, 5.0)  # Giới hạn tối đa 5x
        
        # Điều chỉnh đòn bẩy dựa trên biến động
        if volatility is not None:
            # Giảm đòn bẩy khi biến động cao, tăng khi biến động thấp
            if volatility > 0.08:
                leverage = min(1.5, max_leverage)  # Giới hạn 1.5x khi biến động rất cao
            elif volatility > 0.05:
                leverage = min(2.5, max_leverage)  # Giới hạn 2.5x khi biến động cao
            elif volatility > 0.03:
                leverage = min(3.5, max_leverage)  # Giới hạn 3.5x khi biến động vừa phải
            else:
                leverage = max_leverage  # Đòn bẩy tối đa khi biến động thấp
        else:
            leverage = max_leverage
        
        # Tính position size với đòn bẩy
        leveraged_position_size = position_size * leverage
        
        # Kiểm tra tỷ trọng danh mục
        if current_holdings is not None:
            # Tính tổng giá trị danh mục
            total_portfolio_value = current_capital + sum(
                pos.get("value", 0) for pos in current_holdings.values()
            )
            
            # Kiểm tra vị thế mới có vượt quá tỷ trọng tối đa không
            new_position_value = leveraged_position_size * entry_price
            new_position_weight = new_position_value / total_portfolio_value
            
            if new_position_weight > self.max_position_weight:
                # Điều chỉnh position size để khớp với tỷ trọng tối đa
                adjusted_position_size = (total_portfolio_value * self.max_position_weight) / entry_price
                
                # Giảm đòn bẩy nếu cần
                if adjusted_position_size < position_size:
                    leverage = max(1.0, adjusted_position_size / position_size)
                    leveraged_position_size = position_size * leverage
                else:
                    leveraged_position_size = adjusted_position_size
                
                self.logger.info(f"Điều chỉnh position size để không vượt quá {self.max_position_weight:.1%} tỷ trọng: {leveraged_position_size}")
        
        # Đảm bảo position size không âm
        leveraged_position_size = max(0, leveraged_position_size)
        
        # Tính giá trị vị thế
        position_value = leveraged_position_size * entry_price
        
        # Tính mức rủi ro thực tế sau điều chỉnh
        actual_risk_amount = position_size * risk_per_unit  # Không tính đòn bẩy trong rủi ro
        actual_risk_percent = actual_risk_amount / current_capital if current_capital > 0 else 0
        
        return {
            "status": "success",
            "symbol": symbol,
            "side": side,
            "position_size": leveraged_position_size,
            "base_position_size": position_size,  # Position size trước khi áp dụng đòn bẩy
            "leverage": leverage,
            "position_value": position_value,
            "risk_amount": actual_risk_amount,
            "risk_percent": actual_risk_percent,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price
        }
    
    def calculate_take_profit_levels(
        self,
        entry_price: float,
        stop_loss_price: float,
        side: str,
        risk_reward_targets: List[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Tính toán các mức take profit cho hồ sơ tích cực.
        
        Args:
            entry_price: Giá vào
            stop_loss_price: Giá stop loss
            side: Phía vị thế ('long' hoặc 'short')
            risk_reward_targets: Danh sách các tỷ lệ R:R mục tiêu
            
        Returns:
            Danh sách các mức take profit
        """
        # Nếu không cung cấp targets, sử dụng mặc định cho hồ sơ tích cực
        if risk_reward_targets is None:
            risk_reward_targets = [1.5, 3.0, 5.0, 8.0]  # Mục tiêu xa hơn, để lợi nhuận chạy
        
        # Tính khoảng cách risk
        if side.lower() == "long":
            risk_distance = entry_price - stop_loss_price
        else:
            risk_distance = stop_loss_price - entry_price
        
        # Tránh chia cho 0 hoặc giá trị âm
        if risk_distance <= 0:
            self.logger.warning(f"Khoảng cách risk không hợp lệ: {risk_distance}")
            return []
        
        # Tính các mức take profit
        take_profit_levels = []
        
        for rr_target in risk_reward_targets:
            if side.lower() == "long":
                tp_price = entry_price + (risk_distance * rr_target)
                
                # Quyết định % đóng vị thế tại mức này
                if rr_target <= 1.5:
                    close_percent = 0.25  # Đóng 25% vị thế tại 1.5R
                elif rr_target <= 3.0:
                    close_percent = 0.25  # Đóng 25% vị thế tại 3.0R
                elif rr_target <= 5.0:
                    close_percent = 0.25  # Đóng 25% vị thế tại 5.0R
                else:
                    close_percent = 0.25  # Đóng 25% vị thế tại mức còn lại
            else:
                tp_price = entry_price - (risk_distance * rr_target)
                
                # Quyết định % đóng vị thế tại mức này
                if rr_target <= 1.5:
                    close_percent = 0.25  # Đóng 25% vị thế tại 1.5R
                elif rr_target <= 3.0:
                    close_percent = 0.25  # Đóng 25% vị thế tại 3.0R
                elif rr_target <= 5.0:
                    close_percent = 0.25  # Đóng 25% vị thế tại 5.0R
                else:
                    close_percent = 0.25  # Đóng 25% vị thế tại mức còn lại
            
            take_profit_levels.append({
                "price": tp_price,
                "risk_reward": rr_target,
                "close_percent": close_percent
            })
        
        return take_profit_levels
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
        support_resistance_levels: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Tính toán stop loss cho hồ sơ tích cực.
        
        Args:
            entry_price: Giá vào
            side: Phía vị thế ('long' hoặc 'short')
            atr: Chỉ số ATR hiện tại (nếu có)
            volatility: Biến động của tài sản (nếu có)
            support_resistance_levels: Danh sách các mức hỗ trợ/kháng cự (nếu có)
            
        Returns:
            Dict chứa thông tin stop loss
        """
        # Mặc định sử dụng stop loss rộng hơn cho profile tích cực
        if side.lower() == "long":
            # Cho vị thế Long, stop loss ban đầu 3.5% dưới giá vào
            default_stop_loss = entry_price * 0.965
        else:
            # Cho vị thế Short, stop loss ban đầu 3.5% trên giá vào
            default_stop_loss = entry_price * 1.035
        
        # Điều chỉnh dựa trên ATR nếu có
        if atr is not None and atr > 0:
            if side.lower() == "long":
                atr_stop_loss = entry_price - (atr * 2.0)  # 2.0x ATR cho profile tích cực
            else:
                atr_stop_loss = entry_price + (atr * 2.0)  # 2.0x ATR cho profile tích cực
                
            # Chọn stop loss hợp lý hơn giữa default và ATR-based
            if side.lower() == "long":
                stop_loss_price = max(default_stop_loss, atr_stop_loss)
            else:
                stop_loss_price = min(default_stop_loss, atr_stop_loss)
        else:
            stop_loss_price = default_stop_loss
        
        # Điều chỉnh dựa trên biến động nếu có
        if volatility is not None and volatility > 0:
            # Với profile tích cực, điều chỉnh stop loss theo biến động, rộng hơn khi biến động cao
            volatility_multiplier = min(2.5, max(1.2, 1.0 + volatility))  # Tăng multiplier khi volatility tăng
            
            if side.lower() == "long":
                vol_adjusted_stop = entry_price * (1 - 0.035 * volatility_multiplier)
                # Chọn stop loss xa hơn
                stop_loss_price = min(stop_loss_price, vol_adjusted_stop)
            else:
                vol_adjusted_stop = entry_price * (1 + 0.035 * volatility_multiplier)
                # Chọn stop loss xa hơn
                stop_loss_price = max(stop_loss_price, vol_adjusted_stop)
        
        # Điều chỉnh dựa trên các mức hỗ trợ/kháng cự nếu có
        if support_resistance_levels and len(support_resistance_levels) > 0:
            if side.lower() == "long":
                # Tìm mức hỗ trợ gần nhất dưới entry_price
                closest_support = None
                for level in sorted(support_resistance_levels, reverse=True):
                    if level < entry_price:
                        closest_support = level
                        break
                
                if closest_support is not None:
                    # Đặt stop loss ngay dưới mức hỗ trợ
                    sr_stop_loss = closest_support * 0.98
                    # Chỉ sử dụng nếu không quá xa entry price
                    if sr_stop_loss > entry_price * 0.90:  # Không quá xa entry (tối đa 10%)
                        stop_loss_price = max(stop_loss_price, sr_stop_loss)
            else:
                # Tìm mức kháng cự gần nhất trên entry_price
                closest_resistance = None
                for level in sorted(support_resistance_levels):
                    if level > entry_price:
                        closest_resistance = level
                        break
                
                if closest_resistance is not None:
                    # Đặt stop loss ngay trên mức kháng cự
                    sr_stop_loss = closest_resistance * 1.02
                    # Chỉ sử dụng nếu không quá xa entry price
                    if sr_stop_loss < entry_price * 1.10:  # Không quá xa entry (tối đa 10%)
                        stop_loss_price = min(stop_loss_price, sr_stop_loss)
        
        # Tính khoảng cách stop loss
        if side.lower() == "long":
            stop_distance = (entry_price - stop_loss_price) / entry_price
        else:
            stop_distance = (stop_loss_price - entry_price) / entry_price
        
        # Đảm bảo stop loss không xa quá 8% cho hồ sơ tích cực
        max_stop_distance = 0.08  # 8%
        
        if stop_distance > max_stop_distance:
            # Điều chỉnh stop loss nếu khoảng cách quá lớn
            if side.lower() == "long":
                stop_loss_price = entry_price * (1 - max_stop_distance)
            else:
                stop_loss_price = entry_price * (1 + max_stop_distance)
            
            stop_distance = max_stop_distance
        
        return {
            "stop_loss_price": stop_loss_price,
            "stop_distance_percent": stop_distance * 100,
            "stop_type": "percentage",
            "initial_stop": stop_loss_price,
            "trailing_activation_percent": 3.0,  # Kích hoạt trailing stop sau khi lãi 3%
            "trailing_distance_percent": 2.0     # Trailing stop ở khoảng cách 2%
        }
    
    def get_max_positions(self, current_drawdown: float = 0.0) -> int:
        """
        Lấy số lượng vị thế tối đa cho phép dựa trên drawdown hiện tại.
        
        Args:
            current_drawdown: Drawdown hiện tại (%)
            
        Returns:
            Số lượng vị thế tối đa
        """
        # Số lượng vị thế mặc định cho hồ sơ tích cực
        default_max_positions = 12
        
        # Điều chỉnh dựa trên drawdown - profile tích cực có thể mở nhiều vị thế hơn
        if current_drawdown <= 0.08:  # Drawdown < 8%
            return default_max_positions
        elif current_drawdown <= 0.12:  # Drawdown 8-12%
            return 10
        elif current_drawdown <= 0.18:  # Drawdown 12-18%
            return 8
        elif current_drawdown <= 0.22:  # Drawdown 18-22%
            return 5
        else:  # Drawdown > 22%
            return 3  # Vẫn cho phép một số vị thế ngay cả với drawdown cao
    
    def should_trade(
        self,
        current_conditions: Dict[str, Any],
        recent_losses: int = 0,
        consecutive_losses: int = 0
    ) -> Tuple[bool, str]:
        """
        Kiểm tra xem có nên giao dịch trong điều kiện hiện tại không.
        
        Args:
            current_conditions: Dict chứa các điều kiện thị trường hiện tại
            recent_losses: Số lần thua gần đây
            consecutive_losses: Số lần thua liên tiếp
            
        Returns:
            Tuple (bool, str) - (Nên giao dịch không, Lý do)
        """
        # Lấy các thông số từ điều kiện hiện tại
        market_volatility = current_conditions.get("market_volatility", 0.0)
        current_drawdown = current_conditions.get("current_drawdown", 0.0)
        market_trend = current_conditions.get("market_trend", "neutral")
        
        # Kiểm tra drawdown - profile tích cực vẫn có thể giao dịch với drawdown cao hơn
        if current_drawdown >= self.max_drawdown:
            return False, f"Drawdown hiện tại ({current_drawdown:.1%}) vượt quá giới hạn tối đa ({self.max_drawdown:.1%})"
        
        # Kiểm tra biến động thị trường - profile tích cực thích biến động cao
        if market_volatility > 0.10:  # Biến động > 10%
            # Chỉ tránh biến động cực đoan
            return False, f"Biến động thị trường quá cao ({market_volatility:.1%})"
        
        # Biến động cao là cơ hội cho profile tích cực
        if market_volatility > 0.05 and market_volatility <= 0.10:
            # Biến động cao (5-10%) nhưng chấp nhận được
            # Không cần return False ở đây, đây là điều kiện tốt cho profile tích cực
            pass
        
        # Kiểm tra lịch sử thua lỗ gần đây - profile tích cực chấp nhận nhiều lần thua hơn
        if consecutive_losses >= 5:
            return False, f"Đã thua {consecutive_losses} lần liên tiếp, nên nghỉ ngơi"
        
        if recent_losses >= 10:
            return False, f"Đã thua {recent_losses} lần gần đây, nên giảm tần suất giao dịch"
        
        # Kiểm tra xu hướng thị trường - profile tích cực có thể giao dịch trong nhiều điều kiện
        # Chỉ tránh thị trường giảm mạnh kết hợp với drawdown cao
        if market_trend == "strongly_bearish" and current_drawdown > 0.15:
            return False, "Thị trường đang trong xu hướng giảm mạnh kết hợp với drawdown cao, thận trọng khi mở vị thế mới"
        
        # Profile tích cực thích xu hướng mạnh, có thể tận dụng cơ hội
        if market_trend in ["strongly_bullish", "strongly_bearish"] and current_drawdown < 0.10:
            # Điều kiện tốt cho profile tích cực - xu hướng mạnh, drawdown thấp
            pass
        
        # Giảm giao dịch khi drawdown gần ngưỡng
        if current_drawdown >= self.max_drawdown * 0.9:
            # Chỉ giao dịch những cơ hội rất tốt
            if market_trend != "strongly_bullish" and market_volatility <= 0.03:
                return False, f"Drawdown gần ngưỡng cảnh báo ({current_drawdown:.1%}), chỉ giao dịch cơ hội tốt nhất"
        
        return True, "Điều kiện giao dịch phù hợp với hồ sơ tích cực"
    
    def get_risk_profile_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình đầy đủ của hồ sơ rủi ro tích cực.
        
        Returns:
            Dict chứa toàn bộ cấu hình
        """
        return {
            "name": "aggressive",
            "description": "Hồ sơ rủi ro tích cực, ưu tiên lợi nhuận tối đa",
            "risk_per_trade": self.risk_per_trade,
            "max_drawdown": self.max_drawdown,
            "leverage_max": self.leverage_max,
            "portfolio_volatility_target": self.portfolio_volatility_target,
            "correlation_threshold": self.correlation_threshold,
            "max_position_weight": self.max_position_weight,
            "asset_allocation": self.asset_allocation,
            "take_profit_levels": [1.5, 3.0, 5.0, 8.0],
            "stop_loss_config": {
                "initial_stop_percent": 0.035,
                "trailing_activation_percent": 0.03,
                "trailing_distance_percent": 0.02,
                "atr_multiplier": 2.0
            },
            "position_sizing": {
                "method": "risk_percentage",
                "max_position_size_percent": 0.20,
                "min_position_size": 0.001,
                "use_kelly_criterion": True,
                "kelly_fraction": 0.7
            },
            "drawdown_management": {
                "position_reduction_schedule": [
                    (0.15, 0.25),  # Giảm 25% vị thế khi drawdown 15%
                    (0.20, 0.5),   # Giảm 50% vị thế khi drawdown 20%
                    (0.22, 0.75)   # Giảm 75% vị thế khi drawdown 22%
                ],
                "cooldown_period_days": 3
            },
            "max_positions": 12,
            "max_consecutive_losses": 5,
            "max_daily_trades": 8
        }