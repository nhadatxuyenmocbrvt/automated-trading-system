"""
Bộ chọn chiến lược giao dịch.
File này định nghĩa lớp StrategySelector để lựa chọn chiến lược giao dịch phù hợp nhất
từ một tập hợp các chiến lược dựa trên nhiều tiêu chí như ưu tiên, hiệu suất, và ngữ cảnh thị trường.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from enum import Enum
from datetime import datetime
import random

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import get_logger
from config.system_config import get_system_config
from agent_manager.strategy_queue.strategy_manager import StrategyManager, StrategyStatus, StrategyType

class SelectionMethod(Enum):
    """
    Phương pháp lựa chọn chiến lược.
    Định nghĩa các phương pháp lựa chọn chiến lược từ danh sách.
    """
    HIGHEST_PRIORITY = "highest_priority"     # Chọn chiến lược có ưu tiên cao nhất
    BEST_PERFORMANCE = "best_performance"     # Chọn chiến lược có hiệu suất tốt nhất
    ROUND_ROBIN = "round_robin"               # Lần lượt chọn các chiến lược
    RANDOM = "random"                         # Chọn ngẫu nhiên
    WEIGHTED_RANDOM = "weighted_random"       # Chọn ngẫu nhiên có trọng số
    CONTEXT_BASED = "context_based"           # Chọn dựa trên ngữ cảnh thị trường
    HYBRID = "hybrid"                         # Kết hợp nhiều phương pháp
    MULTI_FACTOR = "multi_factor"             # Phân tích đa yếu tố
    CUSTOM = "custom"                         # Phương pháp tùy chỉnh

class StrategySelector:
    """
    Lớp bộ chọn chiến lược.
    Chịu trách nhiệm lựa chọn chiến lược giao dịch tốt nhất từ một danh sách
    chiến lược dựa trên các tiêu chí và phương pháp khác nhau.
    """
    
    def __init__(
        self,
        strategy_manager: StrategyManager,
        selection_method: Union[str, SelectionMethod] = SelectionMethod.HYBRID,
        max_selections: int = 5,
        use_market_context: bool = True,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo bộ chọn chiến lược.
        
        Args:
            strategy_manager: Quản lý chiến lược cung cấp danh sách chiến lược
            selection_method: Phương pháp lựa chọn chiến lược
            max_selections: Số lượng chiến lược tối đa được chọn mỗi lần
            use_market_context: Có sử dụng ngữ cảnh thị trường khi lựa chọn không
            logger: Logger tùy chỉnh
            **kwargs: Các tham số tùy chọn khác
        """
        # Thiết lập logger
        self.logger = logger or get_logger("strategy_selector")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập các thuộc tính
        self.strategy_manager = strategy_manager
        self.max_selections = max_selections
        self.use_market_context = use_market_context
        self.kwargs = kwargs
        
        # Thiết lập phương pháp lựa chọn
        if isinstance(selection_method, str):
            try:
                self.selection_method = SelectionMethod(selection_method)
            except ValueError:
                self.logger.warning(f"Phương pháp lựa chọn '{selection_method}' không hợp lệ, sử dụng 'hybrid'")
                self.selection_method = SelectionMethod.HYBRID
        else:
            self.selection_method = selection_method
        
        # Các biến theo dõi
        self.last_selected_strategies = []  # ID của các chiến lược được chọn gần đây nhất
        self.selection_history = []  # Lịch sử các lựa chọn
        self.selection_count = 0  # Số lần đã lựa chọn
        self.current_index = 0  # Chỉ số hiện tại cho round-robin
        
        # Trọng số cho các yếu tố (cho phương pháp multi-factor)
        self.factor_weights = {
            "priority": kwargs.get("priority_weight", 0.3),
            "performance": kwargs.get("performance_weight", 0.3),
            "recent_success": kwargs.get("recent_success_weight", 0.2),
            "market_fit": kwargs.get("market_fit_weight", 0.2)
        }
        
        # Hàm tùy chỉnh
        self.custom_selection_fn = kwargs.get("custom_selection_fn", None)
        
        self.logger.info(
            f"Đã khởi tạo {self.__class__.__name__} với phương pháp {self.selection_method.value}, "
            f"max_selections={max_selections}, use_market_context={use_market_context}"
        )
    
    def select_strategies(
        self,
        market_context: Optional[Dict[str, Any]] = None,
        count: Optional[int] = None,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Lựa chọn chiến lược phù hợp nhất dựa trên ngữ cảnh thị trường.
        
        Args:
            market_context: Thông tin ngữ cảnh thị trường
            count: Số lượng chiến lược cần chọn (mặc định là max_selections)
            filter_params: Tham số lọc chiến lược trước khi chọn
            
        Returns:
            Danh sách ID của các chiến lược được chọn
        """
        # Sử dụng max_selections nếu count không được chỉ định
        if count is None:
            count = self.max_selections
        
        # Đảm bảo count không vượt quá max_selections
        count = min(count, self.max_selections)
        
        # Lọc chiến lược theo tham số nếu có
        if filter_params:
            available_strategies = self.strategy_manager.filter_strategies(filter_params)
        else:
            # Mặc định chỉ lấy các chiến lược đang hoạt động
            default_filters = {"metadata.status": StrategyStatus.ACTIVE.value}
            available_strategies = self.strategy_manager.filter_strategies(default_filters)
        
        # Kiểm tra nếu không có chiến lược nào
        if not available_strategies:
            self.logger.warning("Không có chiến lược nào khả dụng để lựa chọn")
            return []
        
        # Tùy theo phương pháp lựa chọn
        if self.selection_method == SelectionMethod.HIGHEST_PRIORITY:
            selected_ids = self._select_highest_priority(available_strategies, count)
        elif self.selection_method == SelectionMethod.BEST_PERFORMANCE:
            selected_ids = self._select_best_performance(available_strategies, count)
        elif self.selection_method == SelectionMethod.ROUND_ROBIN:
            selected_ids = self._select_round_robin(available_strategies, count)
        elif self.selection_method == SelectionMethod.RANDOM:
            selected_ids = self._select_random(available_strategies, count)
        elif self.selection_method == SelectionMethod.WEIGHTED_RANDOM:
            selected_ids = self._select_weighted_random(available_strategies, count)
        elif self.selection_method == SelectionMethod.CONTEXT_BASED:
            selected_ids = self._select_context_based(available_strategies, market_context, count)
        elif self.selection_method == SelectionMethod.HYBRID:
            selected_ids = self._select_hybrid(available_strategies, market_context, count)
        elif self.selection_method == SelectionMethod.MULTI_FACTOR:
            selected_ids = self._select_multi_factor(available_strategies, market_context, count)
        elif self.selection_method == SelectionMethod.CUSTOM and self.custom_selection_fn is not None:
            try:
                selected_ids = self.custom_selection_fn(
                    self, available_strategies, market_context, count
                )
            except Exception as e:
                self.logger.error(f"Lỗi khi sử dụng hàm lựa chọn tùy chỉnh: {str(e)}")
                # Fallback to hybrid
                selected_ids = self._select_hybrid(available_strategies, market_context, count)
        else:
            # Mặc định là hybrid
            selected_ids = self._select_hybrid(available_strategies, market_context, count)
        
        # Cập nhật biến theo dõi
        self.last_selected_strategies = selected_ids
        self.selection_history.append({
            "timestamp": datetime.now().isoformat(),
            "selected_ids": selected_ids,
            "selection_method": self.selection_method.value,
            "market_context_used": market_context is not None
        })
        self.selection_count += 1
        
        self.logger.info(f"Đã chọn {len(selected_ids)} chiến lược: {selected_ids}")
        return selected_ids
    
    def _select_highest_priority(
        self,
        strategies: Dict[str, Dict[str, Any]],
        count: int
    ) -> List[str]:
        """
        Chọn chiến lược dựa trên ưu tiên cao nhất.
        
        Args:
            strategies: Dictionary các chiến lược
            count: Số lượng chiến lược cần chọn
            
        Returns:
            Danh sách ID của các chiến lược được chọn
        """
        # Lấy ưu tiên của các chiến lược
        strategy_priorities = []
        
        for strategy_id, strategy in strategies.items():
            metadata = strategy.get("metadata", {})
            priority = metadata.get("priority", 0)
            strategy_priorities.append((strategy_id, priority))
        
        # Sắp xếp theo ưu tiên giảm dần
        strategy_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Lấy các chiến lược có ưu tiên cao nhất
        selected_ids = [item[0] for item in strategy_priorities[:count]]
        
        return selected_ids
    
    def _select_best_performance(
        self,
        strategies: Dict[str, Dict[str, Any]],
        count: int
    ) -> List[str]:
        """
        Chọn chiến lược dựa trên hiệu suất tốt nhất.
        
        Args:
            strategies: Dictionary các chiến lược
            count: Số lượng chiến lược cần chọn
            
        Returns:
            Danh sách ID của các chiến lược được chọn
        """
        # Xếp hạng các chiến lược
        rankings = []
        
        for strategy_id, strategy in strategies.items():
            rank = self.strategy_manager.calculate_strategy_rank(strategy_id)
            rankings.append((strategy_id, rank))
        
        # Sắp xếp theo thứ hạng giảm dần
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Lấy các chiến lược có thứ hạng cao nhất
        selected_ids = [item[0] for item in rankings[:count]]
        
        return selected_ids
    
    def _select_round_robin(
        self,
        strategies: Dict[str, Dict[str, Any]],
        count: int
    ) -> List[str]:
        """
        Chọn chiến lược theo phương pháp lần lượt.
        
        Args:
            strategies: Dictionary các chiến lược
            count: Số lượng chiến lược cần chọn
            
        Returns:
            Danh sách ID của các chiến lược được chọn
        """
        # Lấy danh sách ID
        strategy_ids = list(strategies.keys())
        selected_ids = []
        
        # Nếu không đủ chiến lược
        if len(strategy_ids) <= count:
            return strategy_ids
        
        # Chọn chiến lược lần lượt
        for _ in range(count):
            # Đặt lại chỉ số nếu vượt quá
            if self.current_index >= len(strategy_ids):
                self.current_index = 0
            
            selected_ids.append(strategy_ids[self.current_index])
            self.current_index += 1
        
        return selected_ids
    
    def _select_random(
        self,
        strategies: Dict[str, Dict[str, Any]],
        count: int
    ) -> List[str]:
        """
        Chọn chiến lược ngẫu nhiên.
        
        Args:
            strategies: Dictionary các chiến lược
            count: Số lượng chiến lược cần chọn
            
        Returns:
            Danh sách ID của các chiến lược được chọn
        """
        # Lấy danh sách ID
        strategy_ids = list(strategies.keys())
        
        # Nếu không đủ chiến lược
        if len(strategy_ids) <= count:
            return strategy_ids
        
        # Chọn ngẫu nhiên không lặp lại
        selected_ids = random.sample(strategy_ids, count)
        
        return selected_ids
    
    def _select_weighted_random(
        self,
        strategies: Dict[str, Dict[str, Any]],
        count: int
    ) -> List[str]:
        """
        Chọn chiến lược ngẫu nhiên có trọng số.
        
        Args:
            strategies: Dictionary các chiến lược
            count: Số lượng chiến lược cần chọn
            
        Returns:
            Danh sách ID của các chiến lược được chọn
        """
        # Lấy danh sách ID và trọng số
        strategy_weights = []
        
        for strategy_id, strategy in strategies.items():
            # Tính trọng số dựa trên nhiều yếu tố
            weight = 1.0  # Trọng số mặc định
            
            # 1. Ưu tiên
            metadata = strategy.get("metadata", {})
            priority = metadata.get("priority", 0)
            weight += priority
            
            # 2. Hiệu suất gần đây
            performance = strategy.get("performance", {})
            latest = performance.get("latest", {})
            
            if "profit" in latest:
                weight += max(0, latest["profit"] * 2)
            
            if "win_rate" in latest:
                weight += latest["win_rate"]
            
            # Đảm bảo trọng số không âm
            weight = max(0.1, weight)
            
            strategy_weights.append((strategy_id, weight))
        
        # Nếu không đủ chiến lược
        if len(strategy_weights) <= count:
            return [sw[0] for sw in strategy_weights]
        
        # Tính tổng trọng số
        total_weight = sum(w for _, w in strategy_weights)
        
        # Chuẩn hóa trọng số
        if total_weight > 0:
            probabilities = [w / total_weight for _, w in strategy_weights]
        else:
            # Nếu tổng trọng số <= 0, sử dụng phân phối đều
            probabilities = [1.0 / len(strategy_weights)] * len(strategy_weights)
        
        # Chọn ngẫu nhiên có trọng số không lặp lại
        strategy_ids = [sw[0] for sw in strategy_weights]
        selected_ids = []
        
        for _ in range(min(count, len(strategy_ids))):
            if not strategy_ids:
                break
            
            # Chọn một chiến lược
            chosen_idx = np.random.choice(len(strategy_ids), p=probabilities)
            chosen_id = strategy_ids[chosen_idx]
            
            # Thêm vào danh sách đã chọn
            selected_ids.append(chosen_id)
            
            # Xóa khỏi danh sách để không chọn lại
            strategy_ids.pop(chosen_idx)
            probabilities.pop(chosen_idx)
            
            # Chuẩn hóa lại xác suất
            if probabilities:
                total_prob = sum(probabilities)
                if total_prob > 0:
                    probabilities = [p / total_prob for p in probabilities]
                else:
                    probabilities = [1.0 / len(probabilities)] * len(probabilities)
        
        return selected_ids
    
    def _select_context_based(
        self,
        strategies: Dict[str, Dict[str, Any]],
        market_context: Optional[Dict[str, Any]],
        count: int
    ) -> List[str]:
        """
        Chọn chiến lược dựa trên ngữ cảnh thị trường.
        
        Args:
            strategies: Dictionary các chiến lược
            market_context: Thông tin ngữ cảnh thị trường
            count: Số lượng chiến lược cần chọn
            
        Returns:
            Danh sách ID của các chiến lược được chọn
        """
        # Nếu không có thông tin ngữ cảnh, sử dụng phương pháp best_performance
        if not market_context:
            self.logger.warning("Không có thông tin ngữ cảnh thị trường, sử dụng phương pháp best_performance")
            return self._select_best_performance(strategies, count)
        
        # Tính điểm phù hợp cho từng chiến lược
        strategy_scores = []
        
        for strategy_id, strategy in strategies.items():
            score = self._calculate_market_fit(strategy, market_context)
            strategy_scores.append((strategy_id, score))
        
        # Sắp xếp theo điểm giảm dần
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Lấy các chiến lược có điểm cao nhất
        selected_ids = [item[0] for item in strategy_scores[:count]]
        
        return selected_ids
    
    def _select_hybrid(
        self,
        strategies: Dict[str, Dict[str, Any]],
        market_context: Optional[Dict[str, Any]],
        count: int
    ) -> List[str]:
        """
        Chọn chiến lược bằng phương pháp kết hợp.
        
        Args:
            strategies: Dictionary các chiến lược
            market_context: Thông tin ngữ cảnh thị trường
            count: Số lượng chiến lược cần chọn
            
        Returns:
            Danh sách ID của các chiến lược được chọn
        """
        # Kết hợp nhiều phương pháp:
        # 1. Tính điểm cho từng chiến lược
        # 2. Chọn các chiến lược có điểm cao nhất
        
        strategy_scores = {}
        
        # Bước 1: Xếp hạng theo ưu tiên
        priority_ids = self._select_highest_priority(strategies, len(strategies))
        for i, strategy_id in enumerate(priority_ids):
            priority_score = 1.0 - (i / len(priority_ids))  # Chuẩn hóa về [0, 1]
            strategy_scores[strategy_id] = strategy_scores.get(strategy_id, 0) + priority_score * 0.3
        
        # Bước 2: Xếp hạng theo hiệu suất
        performance_ids = self._select_best_performance(strategies, len(strategies))
        for i, strategy_id in enumerate(performance_ids):
            performance_score = 1.0 - (i / len(performance_ids))
            strategy_scores[strategy_id] = strategy_scores.get(strategy_id, 0) + performance_score * 0.3
        
        # Bước 3: Tính điểm phù hợp với thị trường nếu có ngữ cảnh
        if market_context:
            for strategy_id, strategy in strategies.items():
                market_fit = self._calculate_market_fit(strategy, market_context)
                strategy_scores[strategy_id] = strategy_scores.get(strategy_id, 0) + market_fit * 0.4
        else:
            # Nếu không có ngữ cảnh, phân bổ điểm cho các chiến lược ngẫu nhiên có trọng số
            weighted_ids = self._select_weighted_random(strategies, len(strategies))
            for i, strategy_id in enumerate(weighted_ids):
                random_score = 1.0 - (i / len(weighted_ids))
                strategy_scores[strategy_id] = strategy_scores.get(strategy_id, 0) + random_score * 0.4
        
        # Chuyển đổi điểm thành danh sách và sắp xếp
        scores_list = [(strategy_id, score) for strategy_id, score in strategy_scores.items()]
        scores_list.sort(key=lambda x: x[1], reverse=True)
        
        # Lấy các chiến lược có điểm cao nhất
        selected_ids = [item[0] for item in scores_list[:count]]
        
        return selected_ids
    
    def _select_multi_factor(
        self,
        strategies: Dict[str, Dict[str, Any]],
        market_context: Optional[Dict[str, Any]],
        count: int
    ) -> List[str]:
        """
        Chọn chiến lược dựa trên phân tích đa yếu tố.
        
        Args:
            strategies: Dictionary các chiến lược
            market_context: Thông tin ngữ cảnh thị trường
            count: Số lượng chiến lược cần chọn
            
        Returns:
            Danh sách ID của các chiến lược được chọn
        """
        # Tính điểm cho từng chiến lược dựa trên nhiều yếu tố
        strategy_scores = {}
        
        for strategy_id, strategy in strategies.items():
            # Khởi tạo điểm
            strategy_scores[strategy_id] = 0.0
            
            # 1. Yếu tố ưu tiên
            metadata = strategy.get("metadata", {})
            priority = metadata.get("priority", 0)
            normalized_priority = min(1.0, max(0.0, priority / 10.0))  # Chuẩn hóa về [0, 1]
            strategy_scores[strategy_id] += normalized_priority * self.factor_weights["priority"]
            
            # 2. Yếu tố hiệu suất
            performance = strategy.get("performance", {})
            latest = performance.get("latest", {})
            
            performance_score = 0.0
            performance_count = 0
            
            if "profit" in latest:
                normalized_profit = min(1.0, max(0.0, (latest["profit"] + 0.2) / 0.4))  # Chuẩn hóa về [0, 1]
                performance_score += normalized_profit
                performance_count += 1
            
            if "win_rate" in latest:
                performance_score += latest["win_rate"]
                performance_count += 1
            
            if "sharpe_ratio" in latest:
                normalized_sharpe = min(1.0, max(0.0, latest["sharpe_ratio"] / 3.0))
                performance_score += normalized_sharpe
                performance_count += 1
            
            if performance_count > 0:
                performance_score /= performance_count
                strategy_scores[strategy_id] += performance_score * self.factor_weights["performance"]
            
            # 3. Yếu tố thành công gần đây
            history = performance.get("history", [])
            if history:
                # Lấy 5 kết quả gần nhất
                recent_history = history[-5:]
                
                success_count = sum(1 for h in recent_history if h.get("profit", 0) > 0)
                recent_success_score = success_count / len(recent_history)
                
                strategy_scores[strategy_id] += recent_success_score * self.factor_weights["recent_success"]
            
            # 4. Yếu tố phù hợp với thị trường
            if market_context:
                market_fit = self._calculate_market_fit(strategy, market_context)
                strategy_scores[strategy_id] += market_fit * self.factor_weights["market_fit"]
        
        # Chuyển đổi điểm thành danh sách và sắp xếp
        scores_list = [(strategy_id, score) for strategy_id, score in strategy_scores.items()]
        scores_list.sort(key=lambda x: x[1], reverse=True)
        
        # Lấy các chiến lược có điểm cao nhất
        selected_ids = [item[0] for item in scores_list[:count]]
        
        return selected_ids
    
    def _calculate_market_fit(
        self,
        strategy: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> float:
        """
        Tính toán mức độ phù hợp của chiến lược với ngữ cảnh thị trường.
        
        Args:
            strategy: Thông tin chiến lược
            market_context: Thông tin ngữ cảnh thị trường
            
        Returns:
            Điểm phù hợp từ 0.0 đến 1.0
        """
        # Khởi tạo điểm
        fit_score = 0.5  # Điểm mặc định
        
        # Lấy thông tin về loại chiến lược
        strategy_type = strategy.get("type", StrategyType.CUSTOM.value)
        
        # Lấy thông tin về điều kiện áp dụng
        conditions = strategy.get("conditions", {})
        
        # Phân tích ngữ cảnh thị trường
        # 1. Xu hướng thị trường
        market_trend = market_context.get("trend", "sideways")
        # 2. Biến động thị trường
        volatility = market_context.get("volatility", 0.5)
        # 3. Khối lượng giao dịch
        volume = market_context.get("volume", "normal")
        # 4. Tâm lý thị trường
        sentiment = market_context.get("sentiment", "neutral")
        
        # Điều chỉnh điểm theo loại chiến lược và ngữ cảnh
        if strategy_type == StrategyType.TREND_FOLLOWING.value:
            # Chiến lược theo xu hướng phù hợp với thị trường có xu hướng rõ ràng
            if market_trend == "uptrend" or market_trend == "downtrend":
                fit_score += 0.3
            else:
                fit_score -= 0.2
            
            # Thích hợp với biến động trung bình đến cao
            if 0.3 <= volatility <= 0.8:
                fit_score += 0.2
        
        elif strategy_type == StrategyType.MEAN_REVERSION.value:
            # Chiến lược hồi quy về trung bình phù hợp với thị trường đi ngang
            if market_trend == "sideways":
                fit_score += 0.3
            else:
                fit_score -= 0.1
            
            # Thích hợp với biến động thấp đến trung bình
            if volatility <= 0.5:
                fit_score += 0.2
        
        elif strategy_type == StrategyType.BREAKOUT.value:
            # Chiến lược đột phá phù hợp với biến động cao và khối lượng lớn
            if volatility >= 0.6:
                fit_score += 0.3
            
            if volume == "high":
                fit_score += 0.2
        
        elif strategy_type == StrategyType.MOMENTUM.value:
            # Chiến lược momentum phù hợp với xu hướng mạnh và tâm lý tích cực
            if market_trend == "uptrend" and sentiment == "positive":
                fit_score += 0.4
            elif market_trend == "downtrend" and sentiment == "negative":
                fit_score += 0.4
        
        elif strategy_type == StrategyType.ARBITRAGE.value:
            # Chiến lược chênh lệch giá không phụ thuộc nhiều vào xu hướng thị trường
            fit_score += 0.1
        
        elif strategy_type == StrategyType.MARKET_MAKING.value:
            # Chiến lược tạo lập thị trường phù hợp với thị trường biến động thấp
            if volatility <= 0.3:
                fit_score += 0.4
            else:
                fit_score -= 0.2
        
        elif strategy_type == StrategyType.SENTIMENT.value:
            # Chiến lược theo tâm lý thị trường phù hợp với tâm lý rõ ràng
            if sentiment != "neutral":
                fit_score += 0.4
        
        # Kiểm tra các điều kiện cụ thể của chiến lược
        condition_matched = 0
        total_conditions = 0
        
        for condition_key, condition_value in conditions.items():
            total_conditions += 1
            
            # Kiểm tra từng điều kiện
            if condition_key in market_context:
                if isinstance(condition_value, list):
                    # Điều kiện dạng danh sách
                    if market_context[condition_key] in condition_value:
                        condition_matched += 1
                elif isinstance(condition_value, dict):
                    # Điều kiện dạng phạm vi
                    if condition_value.get("min") is not None and condition_value.get("max") is not None:
                        if condition_value["min"] <= market_context[condition_key] <= condition_value["max"]:
                            condition_matched += 1
                else:
                    # Điều kiện dạng giá trị đơn
                    if market_context[condition_key] == condition_value:
                        condition_matched += 1
        
        # Tính điểm từ các điều kiện
        if total_conditions > 0:
            condition_score = condition_matched / total_conditions
            fit_score = 0.7 * fit_score + 0.3 * condition_score
        
        # Đảm bảo điểm nằm trong khoảng [0, 1]
        fit_score = min(1.0, max(0.0, fit_score))
        
        return fit_score
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về quá trình lựa chọn.
        
        Returns:
            Dict chứa thống kê
        """
        stats = {
            "total_selections": self.selection_count,
            "last_selected": self.last_selected_strategies,
            "selection_method": self.selection_method.value,
            "factor_weights": self.factor_weights.copy()
        }
        
        # Thống kê về tần suất chọn
        if self.selection_history:
            strategy_frequency = {}
            
            for selection in self.selection_history:
                for strategy_id in selection["selected_ids"]:
                    strategy_frequency[strategy_id] = strategy_frequency.get(strategy_id, 0) + 1
            
            # Sắp xếp theo tần suất
            top_strategies = sorted(
                strategy_frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Top 10
            
            stats["top_selected_strategies"] = [
                {"id": strategy_id, "frequency": freq} for strategy_id, freq in top_strategies
            ]
        
        return stats
    
    def set_selection_method(self, method: Union[str, SelectionMethod]) -> bool:
        """
        Đặt phương pháp lựa chọn chiến lược.
        
        Args:
            method: Phương pháp lựa chọn mới
            
        Returns:
            True nếu đặt thành công, False nếu không
        """
        if isinstance(method, str):
            try:
                method = SelectionMethod(method)
            except ValueError:
                self.logger.warning(f"Phương pháp lựa chọn '{method}' không hợp lệ")
                return False
        
        self.selection_method = method
        self.logger.info(f"Đã đặt phương pháp lựa chọn mới: {self.selection_method.value}")
        return True
    
    def set_factor_weights(self, weights: Dict[str, float]) -> bool:
        """
        Đặt trọng số cho các yếu tố (cho phương pháp multi-factor).
        
        Args:
            weights: Dict chứa trọng số mới
            
        Returns:
            True nếu đặt thành công, False nếu không
        """
        # Kiểm tra tính hợp lệ
        for factor in weights:
            if factor not in self.factor_weights:
                self.logger.warning(f"Yếu tố '{factor}' không hợp lệ")
                return False
            
            if weights[factor] < 0:
                self.logger.warning(f"Trọng số không được âm: {factor}={weights[factor]}")
                return False
        
        # Cập nhật trọng số
        for factor, weight in weights.items():
            self.factor_weights[factor] = weight
        
        # Chuẩn hóa trọng số
        total_weight = sum(self.factor_weights.values())
        if total_weight > 0:
            for factor in self.factor_weights:
                self.factor_weights[factor] /= total_weight
        
        self.logger.info(f"Đã đặt trọng số mới cho các yếu tố: {self.factor_weights}")
        return True
    
    def set_max_selections(self, max_selections: int) -> bool:
        """
        Đặt số lượng chiến lược tối đa được chọn mỗi lần.
        
        Args:
            max_selections: Số lượng tối đa mới
            
        Returns:
            True nếu đặt thành công, False nếu không
        """
        if max_selections <= 0:
            self.logger.warning(f"Số lượng tối đa phải dương: {max_selections}")
            return False
        
        self.max_selections = max_selections
        self.logger.info(f"Đã đặt số lượng chiến lược tối đa mới: {self.max_selections}")
        return True