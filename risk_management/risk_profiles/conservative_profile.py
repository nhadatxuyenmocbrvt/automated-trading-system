"""
Hồ sơ rủi ro thận trọng.
File này định nghĩa lớp ConservativeProfile thực hiện các chiến lược
quản lý rủi ro với mức độ bảo thủ và an toàn cao.
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

class ConservativeProfile:
    """
    Lớp hồ sơ rủi ro thận trọng.
    Định nghĩa các chiến lược quản lý rủi ro với mức độ bảo thủ và an toàn cao,
    phù hợp cho nhà đầu tư ưu tiên giữ vốn và ổn định.
    """
    
    def __init__(
        self,
        capital: float = 10000.0,
        risk_per_trade: float = 0.01,  # 1% mỗi giao dịch
        max_drawdown: float = 0.10,    # 10% drawdown tối đa
        leverage_max: float = 1.0,     # Không sử dụng đòn bẩy
        portfolio_volatility_target: float = 0.10,  # 10% biến động danh mục mục tiêu
        correlation_threshold: float = 0.6,    # Ngưỡng tương quan
        max_position_weight: float = 0.20,     # 20% tỷ trọng tối đa cho một tài sản
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo hồ sơ rủi ro thận trọng.
        
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
        self.logger = logger or get_logger("conservative_profile")
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.leverage_max = leverage_max
        self.portfolio_volatility_target = portfolio_volatility_target
        self.correlation_threshold = correlation_threshold
        self.max_position_weight = max_position_weight
        
        # Thiết lập các tham số mặc định cho hồ sơ thận trọng
        self.asset_allocation = {
            "BTC": 0.25,     # Bitcoin: 25%
            "ETH": 0.15,     # Ethereum: 15%
            "STABLES": 0.50, # Stablecoin: 50%
            "OTHERS": 0.10   # Các altcoin khác: 10%
        }
        
        # Khởi tạo các thành phần quản lý rủi ro
        self._init_risk_components()
        
        self.logger.info(f"Đã khởi tạo hồ sơ rủi ro thận trọng với risk_per_trade={risk_per_trade:.1%}, max_drawdown={max_drawdown:.1%}")
    
    def _init_risk_components(self) -> None:
        """
        Khởi tạo các thành phần quản lý rủi ro với tham số phù hợp cho hồ sơ thận trọng.
        """
        # Position Sizer - thận trọng với giới hạn size nhỏ
        self.position_sizer = PositionSizer(
            risk_per_trade=self.risk_per_trade,
            position_sizing_method="risk_percentage",
            max_position_size_percent=0.05,  # Giới hạn 5% vốn cho mỗi vị thế
            min_position_size=0.001          # Size tối thiểu
        )
        
        # Stop Loss Manager - stop loss chặt chẽ
        self.stop_loss_manager = StopLossManager(
            max_risk_percent=self.risk_per_trade,
            trailing_stop_enabled=True,
            trailing_stop_activation_percent=0.01,  # Kích hoạt trailing stop sau khi lãi 1%
            trailing_stop_distance_percent=0.01,    # Trailing stop ở khoảng cách 1%
            atr_multiplier=1.0                      # Sử dụng 1x ATR cho stop loss
        )
        
        # Take Profit Manager - take profit sớm, ưu tiên bảo toàn lợi nhuận
        self.take_profit_manager = TakeProfitManager(
            risk_reward_ratio=1.5,                   # Tỷ lệ R:R 1.5:1
            partial_take_profit_enabled=True,        # Bật take profit một phần
            partial_take_profit_levels=[0.5, 0.75],  # Take 50% tại 0.5R, 75% tại 0.75R
            partial_take_profit_percents=[0.3, 0.3]  # Đóng 30% vị thế tại mỗi mức
        )
        
        # Drawdown Manager - phản ứng nhanh với drawdown nhỏ
        self.drawdown_manager = DrawdownManager(
            max_drawdown_percent=self.max_drawdown,
            position_reduction_schedule=[
                (0.05, 0.2),  # Giảm 20% vị thế khi drawdown 5%
                (0.07, 0.5),  # Giảm 50% vị thế khi drawdown 7%
                (0.09, 1.0)   # Đóng toàn bộ vị thế khi drawdown 9%
            ],
            cooldown_period_days=7,           # Cooldown 7 ngày sau khi kích hoạt
            emergency_action="close_all"       # Đóng tất cả các vị thế trong trường hợp khẩn cấp
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
        volatility: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Tính toán kích thước vị thế theo hồ sơ thận trọng.
        
        Args:
            symbol: Mã tài sản
            entry_price: Giá vào
            stop_loss_price: Giá stop loss
            current_capital: Vốn hiện tại
            current_holdings: Danh sách các vị thế hiện tại
            volatility: Biến động của tài sản (nếu có)
            
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
        
        # Giảm position size dựa trên biến động nếu có
        if volatility is not None and volatility > 0:
            # Giảm position size khi biến động cao
            if volatility > 0.04:  # 4% biến động hàng ngày
                volatility_factor = 0.8  # Giảm 20%
            elif volatility > 0.03:  # 3% biến động hàng ngày
                volatility_factor = 0.9  # Giảm 10%
            else:
                volatility_factor = 1.0
                
            position_size *= volatility_factor
            self.logger.info(f"Điều chỉnh position size theo biến động {volatility:.1%}: factor={volatility_factor}")
        
        # Kiểm tra tỷ trọng danh mục
        if current_holdings is not None:
            # Tính tổng giá trị danh mục
            total_portfolio_value = current_capital + sum(
                pos.get("value", 0) for pos in current_holdings.values()
            )
            
            # Kiểm tra vị thế mới có vượt quá tỷ trọng tối đa không
            new_position_value = position_size * entry_price
            new_position_weight = new_position_value / total_portfolio_value
            
            if new_position_weight > self.max_position_weight:
                # Điều chỉnh position size để khớp với tỷ trọng tối đa
                adjusted_position_size = (total_portfolio_value * self.max_position_weight) / entry_price
                
                # Chọn giá trị nhỏ hơn giữa position size ban đầu và adjusted position size
                position_size = min(position_size, adjusted_position_size)
                
                self.logger.info(f"Điều chỉnh position size để không vượt quá {self.max_position_weight:.1%} tỷ trọng: {position_size}")
        
        # Đảm bảo position size không âm
        position_size = max(0, position_size)
        
        # Tính giá trị vị thế
        position_value = position_size * entry_price
        
        # Tính mức rủi ro thực tế sau điều chỉnh
        actual_risk_amount = position_size * risk_per_unit
        actual_risk_percent = actual_risk_amount / current_capital if current_capital > 0 else 0
        
        return {
            "status": "success",
            "symbol": symbol,
            "side": side,
            "position_size": position_size,
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
        Tính toán các mức take profit cho hồ sơ thận trọng.
        
        Args:
            entry_price: Giá vào
            stop_loss_price: Giá stop loss
            side: Phía vị thế ('long' hoặc 'short')
            risk_reward_targets: Danh sách các tỷ lệ R:R mục tiêu
            
        Returns:
            Danh sách các mức take profit
        """
        # Nếu không cung cấp targets, sử dụng mặc định cho hồ sơ thận trọng
        if risk_reward_targets is None:
            risk_reward_targets = [0.5, 0.8, 1.2, 1.5]  # Nhiều cấp take profit nhỏ
        
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
                if rr_target <= 0.5:
                    close_percent = 0.25  # Đóng 25% vị thế tại 0.5R
                elif rr_target <= 1.0:
                    close_percent = 0.35  # Đóng 35% vị thế tại 0.8-1.0R
                elif rr_target <= 1.5:
                    close_percent = 0.25  # Đóng 25% vị thế tại 1.2-1.5R
                else:
                    close_percent = 0.15  # Đóng 15% vị thế tại mức còn lại
            else:
                tp_price = entry_price - (risk_distance * rr_target)
                
                # Quyết định % đóng vị thế tại mức này
                if rr_target <= 0.5:
                    close_percent = 0.25  # Đóng 25% vị thế tại 0.5R
                elif rr_target <= 1.0:
                    close_percent = 0.35  # Đóng 35% vị thế tại 0.8-1.0R
                elif rr_target <= 1.5:
                    close_percent = 0.25  # Đóng 25% vị thế tại 1.2-1.5R
                else:
                    close_percent = 0.15  # Đóng 15% vị thế tại mức còn lại
            
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
        Tính toán stop loss cho hồ sơ thận trọng.
        
        Args:
            entry_price: Giá vào
            side: Phía vị thế ('long' hoặc 'short')
            atr: Chỉ số ATR hiện tại (nếu có)
            volatility: Biến động của tài sản (nếu có)
            support_resistance_levels: Danh sách các mức hỗ trợ/kháng cự (nếu có)
            
        Returns:
            Dict chứa thông tin stop loss
        """
        # Mặc định sử dụng stop loss cứng
        if side.lower() == "long":
            # Cho vị thế Long, stop loss ban đầu 2% dưới giá vào
            default_stop_loss = entry_price * 0.98
        else:
            # Cho vị thế Short, stop loss ban đầu 2% trên giá vào
            default_stop_loss = entry_price * 1.02
        
        # Điều chỉnh dựa trên ATR nếu có
        if atr is not None and atr > 0:
            if side.lower() == "long":
                atr_stop_loss = entry_price - (atr * 1.0)  # 1.0x ATR cho profile thận trọng
            else:
                atr_stop_loss = entry_price + (atr * 1.0)  # 1.0x ATR cho profile thận trọng
                
            # Chọn stop loss chặt hơn giữa default và ATR-based
            if side.lower() == "long":
                stop_loss_price = max(default_stop_loss, atr_stop_loss)
            else:
                stop_loss_price = min(default_stop_loss, atr_stop_loss)
        else:
            stop_loss_price = default_stop_loss
        
        # Điều chỉnh dựa trên biến động nếu có
        if volatility is not None and volatility > 0:
            # Cho biến động cao, đặt stop loss chặt hơn
            volatility_multiplier = min(1.5, max(0.8, 1.0 - volatility))  # Giảm multiplier khi volatility tăng
            
            if side.lower() == "long":
                vol_adjusted_stop = entry_price * (1 - 0.02 * volatility_multiplier)
                # Chọn stop loss chặt hơn
                stop_loss_price = max(stop_loss_price, vol_adjusted_stop)
            else:
                vol_adjusted_stop = entry_price * (1 + 0.02 * volatility_multiplier)
                # Chọn stop loss chặt hơn
                stop_loss_price = min(stop_loss_price, vol_adjusted_stop)
        
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
                    sr_stop_loss = closest_support * 0.99
                    # Chọn stop loss hợp lý nhất
                    if sr_stop_loss > entry_price * 0.95:  # Không quá xa entry (tối đa 5%)
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
                    sr_stop_loss = closest_resistance * 1.01
                    # Chọn stop loss hợp lý nhất
                    if sr_stop_loss < entry_price * 1.05:  # Không quá xa entry (tối đa 5%)
                        stop_loss_price = min(stop_loss_price, sr_stop_loss)
        
        # Tính khoảng cách stop loss
        if side.lower() == "long":
            stop_distance = (entry_price - stop_loss_price) / entry_price
        else:
            stop_distance = (stop_loss_price - entry_price) / entry_price
        
        # Đảm bảo stop loss không xa quá 3% cho hồ sơ thận trọng
        max_stop_distance = 0.03  # 3%
        
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
            "trailing_activation_percent": 1.0,  # Kích hoạt trailing stop sau khi lãi 1%
            "trailing_distance_percent": 1.0     # Trailing stop ở khoảng cách 1%
        }
    
    def get_max_positions(self, current_drawdown: float = 0.0) -> int:
        """
        Lấy số lượng vị thế tối đa cho phép dựa trên drawdown hiện tại.
        
        Args:
            current_drawdown: Drawdown hiện tại (%)
            
        Returns:
            Số lượng vị thế tối đa
        """
        # Số lượng vị thế mặc định cho hồ sơ thận trọng
        default_max_positions = 5
        
        # Điều chỉnh dựa trên drawdown
        if current_drawdown <= 0.03:  # Drawdown < 3%
            return default_max_positions
        elif current_drawdown <= 0.05:  # Drawdown 3-5%
            return 4
        elif current_drawdown <= 0.08:  # Drawdown 5-8%
            return 3
        elif current_drawdown <= 0.10:  # Drawdown 8-10%
            return 2
        else:  # Drawdown > 10%
            return 1  # Rất thận trọng khi drawdown lớn
    
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
        
        # Kiểm tra drawdown
        if current_drawdown >= self.max_drawdown:
            return False, f"Drawdown hiện tại ({current_drawdown:.1%}) vượt quá giới hạn tối đa ({self.max_drawdown:.1%})"
        
        # Kiểm tra biến động thị trường
        if market_volatility > 0.04:  # Biến động > 4%
            return False, f"Biến động thị trường quá cao ({market_volatility:.1%})"
        
        # Kiểm tra lịch sử thua lỗ gần đây
        if consecutive_losses >= 3:
            return False, f"Đã thua {consecutive_losses} lần liên tiếp, nên nghỉ ngơi"
        
        if recent_losses >= 5:
            return False, f"Đã thua {recent_losses} lần gần đây, nên giảm tần suất giao dịch"
        
        # Kiểm tra xu hướng thị trường
        if market_trend == "strongly_bearish":
            return False, "Thị trường đang trong xu hướng giảm mạnh, không nên mở vị thế mới"
        
        # Giảm giao dịch khi drawdown gần ngưỡng
        if current_drawdown >= self.max_drawdown * 0.8:
            # Chỉ giao dịch trong điều kiện rất tốt
            if market_trend != "strongly_bullish" and market_volatility > 0.02:
                return False, f"Drawdown gần ngưỡng cảnh báo ({current_drawdown:.1%}), chỉ giao dịch trong điều kiện tốt nhất"
        
        return True, "Điều kiện giao dịch phù hợp với hồ sơ thận trọng"
    
    def get_risk_profile_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình đầy đủ của hồ sơ rủi ro thận trọng.
        
        Returns:
            Dict chứa toàn bộ cấu hình
        """
        return {
            "name": "conservative",
            "description": "Hồ sơ rủi ro thận trọng, ưu tiên bảo toàn vốn",
            "risk_per_trade": self.risk_per_trade,
            "max_drawdown": self.max_drawdown,
            "leverage_max": self.leverage_max,
            "portfolio_volatility_target": self.portfolio_volatility_target,
            "correlation_threshold": self.correlation_threshold,
            "max_position_weight": self.max_position_weight,
            "asset_allocation": self.asset_allocation,
            "take_profit_levels": [0.5, 0.8, 1.2, 1.5],
            "stop_loss_config": {
                "initial_stop_percent": 0.02,
                "trailing_activation_percent": 0.01,
                "trailing_distance_percent": 0.01,
                "atr_multiplier": 1.0
            },
            "position_sizing": {
                "method": "risk_percentage",
                "max_position_size_percent": 0.05,
                "min_position_size": 0.001
            },
            "drawdown_management": {
                "position_reduction_schedule": [
                    (0.05, 0.2),  # Giảm 20% vị thế khi drawdown 5%
                    (0.07, 0.5),  # Giảm 50% vị thế khi drawdown 7%
                    (0.09, 1.0)   # Đóng toàn bộ vị thế khi drawdown 9%
                ],
                "cooldown_period_days": 7
            },
            "max_positions": 5,
            "max_consecutive_losses": 3,
            "max_daily_trades": 3
        }