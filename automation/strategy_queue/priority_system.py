"""
Hệ thống ưu tiên chiến lược.
File này định nghĩa lớp PrioritySystem để tính toán và quản lý mức độ ưu tiên
của các chiến lược giao dịch, dựa trên hiệu suất, độ mới, và các yếu tố khác.
"""

import os
import sys
import logging
import time
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config


class PrioritySystem:
    """
    Hệ thống tính toán và quản lý ưu tiên cho các chiến lược giao dịch.
    Cung cấp các phương thức để đánh giá ưu tiên dựa trên hiệu suất,
    thời gian chờ, các điều kiện thị trường, và các yếu tố khác.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        performance_weight: float = 0.5,
        recency_weight: float = 0.2,
        waiting_weight: float = 0.1,
        execution_weight: float = 0.1,
        market_condition_weight: float = 0.1,
        retry_penalty: float = 0.3,
        failure_penalty: float = 0.5,
        performance_window: int = 10,
        max_waiting_bonus: float = 2.0,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo hệ thống ưu tiên.
        
        Args:
            config: Cấu hình hệ thống ưu tiên
            performance_weight: Trọng số của hiệu suất trong tính toán ưu tiên
            recency_weight: Trọng số của độ mới
            waiting_weight: Trọng số của thời gian chờ
            execution_weight: Trọng số của thời gian thực thi
            market_condition_weight: Trọng số của điều kiện thị trường
            retry_penalty: Phạt cho việc thử lại
            failure_penalty: Phạt cho việc thất bại
            performance_window: Số lượng thực thi gần nhất để tính hiệu suất
            max_waiting_bonus: Thưởng tối đa cho thời gian chờ
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("priority_system")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Nếu có config, sử dụng các giá trị từ config
        if config:
            self.performance_weight = config.get('performance_weight', performance_weight)
            self.recency_weight = config.get('recency_weight', recency_weight)
            self.waiting_weight = config.get('waiting_weight', waiting_weight)
            self.execution_weight = config.get('execution_weight', execution_weight)
            self.market_condition_weight = config.get('market_condition_weight', market_condition_weight)
            self.retry_penalty = config.get('retry_penalty', retry_penalty)
            self.failure_penalty = config.get('failure_penalty', failure_penalty)
            self.performance_window = config.get('performance_window', performance_window)
            self.max_waiting_bonus = config.get('max_waiting_bonus', max_waiting_bonus)
        else:
            # Sử dụng giá trị mặc định
            self.performance_weight = performance_weight
            self.recency_weight = recency_weight
            self.waiting_weight = waiting_weight
            self.execution_weight = execution_weight
            self.market_condition_weight = market_condition_weight
            self.retry_penalty = retry_penalty
            self.failure_penalty = failure_penalty
            self.performance_window = performance_window
            self.max_waiting_bonus = max_waiting_bonus
        
        # Lưu các tham số bổ sung
        self.kwargs = kwargs
        
        # Các thông số bên trong
        self.market_condition_cache = {}
        self.performance_history = {}
        self.last_update_time = time.time()
        
        # Tính tổng trọng số để đảm bảo tổng bằng 1
        total_weight = (
            self.performance_weight + self.recency_weight + self.waiting_weight +
            self.execution_weight + self.market_condition_weight
        )
        
        # Chuẩn hóa trọng số nếu tổng không bằng 1
        if abs(total_weight - 1.0) > 1e-6:
            self.performance_weight /= total_weight
            self.recency_weight /= total_weight
            self.waiting_weight /= total_weight
            self.execution_weight /= total_weight
            self.market_condition_weight /= total_weight
            
            self.logger.info("Đã chuẩn hóa trọng số ưu tiên để tổng bằng 1")
        
        self.logger.info(
            f"Đã khởi tạo PrioritySystem với trọng số: performance={self.performance_weight:.2f}, "
            f"recency={self.recency_weight:.2f}, waiting={self.waiting_weight:.2f}, "
            f"execution={self.execution_weight:.2f}, market_condition={self.market_condition_weight:.2f}"
        )
    
    def calculate_initial_priority(self, strategy_info: Dict[str, Any]) -> float:
        """
        Tính toán ưu tiên ban đầu cho chiến lược mới.
        
        Args:
            strategy_info: Thông tin chiến lược
            
        Returns:
            float: Giá trị ưu tiên
        """
        # Lấy các thông số từ chiến lược
        initial_priority = strategy_info.get('initial_priority', 0.0)
        strategy_type = strategy_info.get('type', 'unknown')
        
        # Nếu đã có ưu tiên ban đầu, sử dụng giá trị đó
        if initial_priority > 0:
            return initial_priority
        
        # Tính toán ưu tiên mặc định dựa trên loại chiến lược
        base_priority = 0.5  # Giá trị mặc định
        
        if strategy_type == 'high_frequency':
            base_priority = 0.8
        elif strategy_type == 'day_trading':
            base_priority = 0.7
        elif strategy_type == 'swing_trading':
            base_priority = 0.6
        elif strategy_type == 'position_trading':
            base_priority = 0.5
        elif strategy_type == 'experimental':
            base_priority = 0.3
        
        # Điều chỉnh dựa trên thị trường được giao dịch
        market = strategy_info.get('market', '').lower()
        
        market_adjustment = 0.0
        if 'crypto' in market:
            market_adjustment = 0.1  # Ưu tiên crypto cao hơn
        elif 'forex' in market:
            market_adjustment = 0.05
        elif 'stock' in market:
            market_adjustment = 0.0
        
        # Điều chỉnh dựa trên điều kiện thị trường hiện tại
        market_condition_score = self._get_market_condition_score(market)
        
        # Tính ưu tiên cuối cùng
        priority = base_priority + market_adjustment + market_condition_score * 0.2
        
        # Giới hạn trong khoảng [0, 1]
        priority = max(0.0, min(1.0, priority))
        
        self.logger.debug(
            f"Tính toán ưu tiên ban đầu cho chiến lược loại '{strategy_type}', "
            f"thị trường '{market}': {priority:.4f}"
        )
        
        return priority
    
    def recalculate_priority(self, strategy_item: Dict[str, Any]) -> float:
        """
        Tính lại ưu tiên cho chiến lược trong hàng đợi.
        
        Args:
            strategy_item: Thông tin chiến lược trong hàng đợi
            
        Returns:
            float: Ưu tiên mới
        """
        strategy_info = strategy_item['strategy_info']
        strategy_id = strategy_info['id']
        
        # Lấy các thông số từ strategy_item
        current_priority = strategy_item.get('priority', 0.5)
        added_time = strategy_item.get('added_time')
        last_execution = strategy_item.get('last_execution')
        executions = strategy_item.get('executions', 0)
        avg_performance = strategy_item.get('avg_performance', 0.0)
        success_rate = strategy_item.get('success_rate', 0.0)
        retry_count = strategy_item.get('retry_count', 0)
        
        # Đảm bảo added_time là datetime
        if added_time and not isinstance(added_time, datetime):
            try:
                added_time = datetime.fromisoformat(added_time)
            except (ValueError, TypeError):
                added_time = datetime.now() - timedelta(days=1)  # Giá trị mặc định
        
        # Đảm bảo last_execution là datetime
        if last_execution and not isinstance(last_execution, datetime):
            try:
                last_execution = datetime.fromisoformat(last_execution)
            except (ValueError, TypeError):
                last_execution = None
        
        # 1. Tính điểm hiệu suất
        performance_score = self._calculate_performance_score(avg_performance, success_rate, executions)
        
        # 2. Tính điểm mới
        recency_score = self._calculate_recency_score(last_execution)
        
        # 3. Tính điểm thời gian chờ
        waiting_score = self._calculate_waiting_score(added_time)
        
        # 4. Tính điểm thời gian thực thi
        execution_score = self._calculate_execution_score(strategy_item)
        
        # 5. Tính điểm điều kiện thị trường
        market_condition_score = self._calculate_market_condition_score(strategy_info)
        
        # 6. Điều chỉnh cho việc thử lại
        retry_adjustment = retry_count * self.retry_penalty
        
        # Tính ưu tiên tổng thể
        priority = (
            self.performance_weight * performance_score +
            self.recency_weight * recency_score +
            self.waiting_weight * waiting_score +
            self.execution_weight * execution_score +
            self.market_condition_weight * market_condition_score -
            retry_adjustment
        )
        
        # Giới hạn trong khoảng [0, 1]
        priority = max(0.0, min(1.0, priority))
        
        # Lưu vào history nếu có thay đổi đáng kể
        if abs(priority - current_priority) > 0.05:
            if strategy_id not in self.performance_history:
                self.performance_history[strategy_id] = []
            
            self.performance_history[strategy_id].append({
                'timestamp': datetime.now(),
                'old_priority': current_priority,
                'new_priority': priority,
                'performance_score': performance_score,
                'recency_score': recency_score,
                'waiting_score': waiting_score,
                'execution_score': execution_score,
                'market_condition_score': market_condition_score,
                'retry_adjustment': retry_adjustment
            })
            
            # Giữ historical record ở mức hợp lý
            if len(self.performance_history[strategy_id]) > 100:
                self.performance_history[strategy_id] = self.performance_history[strategy_id][-100:]
        
        return priority
    
    def calculate_priority_after_execution(
        self, 
        strategy_item: Dict[str, Any], 
        performance: float, 
        success: bool = True
    ) -> float:
        """
        Tính ưu tiên mới sau khi thực thi chiến lược.
        
        Args:
            strategy_item: Thông tin chiến lược
            performance: Hiệu suất của lần thực thi
            success: Thành công hay thất bại
            
        Returns:
            float: Ưu tiên mới
        """
        strategy_info = strategy_item['strategy_info']
        
        # Lấy ưu tiên hiện tại
        current_priority = strategy_item.get('priority', 0.5)
        
        if success:
            # Nếu thành công, điều chỉnh dựa trên hiệu suất
            if performance > 0.05:  # Hiệu suất rất tốt
                priority_adjustment = 0.2
            elif performance > 0.01:  # Hiệu suất tốt
                priority_adjustment = 0.1
            elif performance > 0:  # Hiệu suất tích cực
                priority_adjustment = 0.05
            elif performance > -0.01:  # Hiệu suất trung bình
                priority_adjustment = 0
            else:  # Hiệu suất kém
                priority_adjustment = -0.1
            
            # Áp dụng điều chỉnh
            new_priority = current_priority + priority_adjustment
            
            # Lấy số lần thực thi
            executions = strategy_item.get('executions', 1)
            
            # Chiến lược đã chạy nhiều lần cần ổn định hơn
            if executions > 10:
                # Tiệm cận dần về mức trung bình
                new_priority = new_priority * 0.9 + 0.5 * 0.1
        else:
            # Nếu thất bại, giảm ưu tiên
            new_priority = current_priority - self.failure_penalty
        
        # Kiểm tra loại chiến lược để điều chỉnh thêm
        strategy_type = strategy_info.get('type', '').lower()
        
        # Một số loại chiến lược cần ưu tiên đặc biệt
        if 'hedging' in strategy_type:
            # Chiến lược đối trọng luôn cần ưu tiên cao
            new_priority = max(new_priority, 0.7)
        elif 'stop_loss' in strategy_type:
            # Chiến lược dừng lỗ cần ưu tiên cao
            new_priority = max(new_priority, 0.8)
        
        # Giới hạn trong khoảng [0, 1]
        new_priority = max(0.0, min(1.0, new_priority))
        
        self.logger.debug(
            f"Ưu tiên sau thực thi: {current_priority:.2f} -> {new_priority:.2f} "
            f"(performance: {performance:.4f}, success: {success})"
        )
        
        return new_priority
    
    def calculate_retry_priority(self, failed_strategy: Dict[str, Any]) -> float:
        """
        Tính ưu tiên cho chiến lược thất bại khi thử lại.
        
        Args:
            failed_strategy: Thông tin chiến lược thất bại
            
        Returns:
            float: Ưu tiên mới
        """
        current_priority = failed_strategy.get('priority', 0.5)
        failure_count = failed_strategy.get('retry_count', 1)
        
        # Giảm ưu tiên dựa trên số lần thất bại
        new_priority = current_priority - (failure_count * self.failure_penalty)
        
        # Đảm bảo ưu tiên không quá thấp
        new_priority = max(0.1, new_priority)
        
        # Điều chỉnh dựa trên thời gian thất bại
        failure_time = failed_strategy.get('failure_time')
        
        if failure_time:
            # Nếu đã thất bại từ lâu, tăng ưu tiên một chút
            if isinstance(failure_time, str):
                try:
                    failure_time = datetime.fromisoformat(failure_time)
                except (ValueError, TypeError):
                    failure_time = datetime.now() - timedelta(days=1)
            
            days_since_failure = (datetime.now() - failure_time).total_seconds() / 86400
            
            if days_since_failure > 7:  # Hơn 1 tuần
                recovery_bonus = 0.3
            elif days_since_failure > 3:  # Hơn 3 ngày
                recovery_bonus = 0.2
            elif days_since_failure > 1:  # Hơn 1 ngày
                recovery_bonus = 0.1
            else:
                recovery_bonus = 0.0
            
            new_priority += recovery_bonus
        
        # Giới hạn trong khoảng [0, 1]
        new_priority = max(0.0, min(1.0, new_priority))
        
        return new_priority
    
    def adjust_priority_based_on_market(
        self, 
        strategy_id: str, 
        market_condition: Dict[str, Any]
    ) -> Optional[float]:
        """
        Điều chỉnh ưu tiên dựa trên điều kiện thị trường.
        
        Args:
            strategy_id: ID chiến lược
            market_condition: Thông tin điều kiện thị trường
            
        Returns:
            float: Điều chỉnh ưu tiên hoặc None nếu không có thay đổi
        """
        # Cập nhật cache điều kiện thị trường
        self.market_condition_cache.update(market_condition)
        
        # Lấy thông tin chiến lược từ history
        if strategy_id not in self.performance_history:
            return None
        
        strategy_history = self.performance_history[strategy_id]
        if not strategy_history:
            return None
        
        # Lấy ưu tiên hiện tại
        current_priority = strategy_history[-1].get('new_priority', 0.5)
        
        # Đánh giá điều kiện thị trường
        market_type = market_condition.get('market_type', 'normal')
        volatility = market_condition.get('volatility', 0.5)
        trend = market_condition.get('trend', 0.0)
        
        # Điều chỉnh dựa trên loại thị trường
        if market_type == 'highly_volatile':
            # Trong thị trường biến động cao, ưu tiên chiến lược đối trọng và dừng lỗ
            if 'hedging' in str(strategy_id).lower() or 'stop_loss' in str(strategy_id).lower():
                return min(1.0, current_priority + 0.3)
        
        elif market_type == 'trending':
            # Trong thị trường xu hướng, ưu tiên chiến lược theo xu hướng
            if 'trend' in str(strategy_id).lower():
                return min(1.0, current_priority + 0.2)
        
        elif market_type == 'ranging':
            # Trong thị trường dao động, ưu tiên chiến lược dao động
            if 'range' in str(strategy_id).lower() or 'oscillator' in str(strategy_id).lower():
                return min(1.0, current_priority + 0.2)
        
        # Không có thay đổi
        return None
    
    def bulk_update_priorities(self, market_condition: Dict[str, Any]) -> Dict[str, float]:
        """
        Cập nhật hàng loạt ưu tiên dựa trên điều kiện thị trường.
        
        Args:
            market_condition: Thông tin điều kiện thị trường
            
        Returns:
            Dict: {strategy_id: new_priority} cho các chiến lược có cập nhật
        """
        updates = {}
        
        # Cập nhật cache điều kiện thị trường
        self.market_condition_cache.update(market_condition)
        
        # Duyệt qua tất cả chiến lược trong history
        for strategy_id, history in self.performance_history.items():
            if not history:
                continue
            
            adjustment = self.adjust_priority_based_on_market(strategy_id, market_condition)
            
            if adjustment is not None:
                updates[strategy_id] = adjustment
        
        if updates:
            self.logger.info(f"Đã cập nhật ưu tiên hàng loạt cho {len(updates)} chiến lược")
        
        return updates
    
    def get_priority_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử ưu tiên của chiến lược.
        
        Args:
            strategy_id: ID chiến lược
            
        Returns:
            List: Lịch sử ưu tiên
        """
        return self.performance_history.get(strategy_id, [])
    
    def _calculate_performance_score(
        self, 
        avg_performance: float, 
        success_rate: float, 
        executions: int
    ) -> float:
        """
        Tính điểm hiệu suất dựa trên hiệu suất trung bình và tỷ lệ thành công.
        
        Args:
            avg_performance: Hiệu suất trung bình
            success_rate: Tỷ lệ thành công
            executions: Số lần thực thi
            
        Returns:
            float: Điểm hiệu suất [0, 1]
        """
        # Chiến lược chưa chạy bao giờ
        if executions == 0:
            return 0.5  # Giá trị trung bình
        
        # Tính điểm từ hiệu suất
        # Chuẩn hóa hiệu suất: [-0.1, 0.1] -> [0, 1]
        performance_normalized = min(1.0, max(0.0, (avg_performance + 0.1) / 0.2))
        
        # Tính điểm từ tỷ lệ thành công
        success_score = min(1.0, max(0.0, success_rate))
        
        # Trọng số của mỗi thành phần thay đổi theo số lần thực thi
        if executions < 5:
            # Ít dữ liệu, ưu tiên tỷ lệ thành công
            performance_weight = 0.3
            success_weight = 0.7
        else:
            # Nhiều dữ liệu, ưu tiên hiệu suất
            performance_weight = 0.7
            success_weight = 0.3
        
        # Tính điểm tổng hợp
        score = performance_weight * performance_normalized + success_weight * success_score
        
        return score
    
    def _calculate_recency_score(self, last_execution: Optional[datetime]) -> float:
        """
        Tính điểm mới dựa trên thời gian thực thi gần nhất.
        
        Args:
            last_execution: Thời gian thực thi gần nhất
            
        Returns:
            float: Điểm mới [0, 1]
        """
        if last_execution is None:
            return 1.0  # Chưa bao giờ chạy
        
        # Tính số giờ từ lần thực thi cuối
        hours_since_last = (datetime.now() - last_execution).total_seconds() / 3600
        
        # Điểm giảm dần theo thời gian, đạt 0 sau 24 giờ
        score = 1.0 - min(1.0, hours_since_last / 24.0)
        
        return score
    
    def _calculate_waiting_score(self, added_time: Optional[datetime]) -> float:
        """
        Tính điểm thời gian chờ.
        
        Args:
            added_time: Thời gian thêm vào hàng đợi
            
        Returns:
            float: Điểm thời gian chờ [0, 1]
        """
        if added_time is None:
            return 0.0
        
        # Tính số giờ đã chờ
        hours_waiting = (datetime.now() - added_time).total_seconds() / 3600
        
        # Điểm tăng dần theo thời gian chờ, đạt max sau 12 giờ
        score = min(self.max_waiting_bonus, hours_waiting / 12.0)
        
        # Chuẩn hóa về [0, 1]
        score = min(1.0, score / self.max_waiting_bonus)
        
        return score
    
    def _calculate_execution_score(self, strategy_item: Dict[str, Any]) -> float:
        """
        Tính điểm thời gian thực thi.
        
        Args:
            strategy_item: Thông tin chiến lược
            
        Returns:
            float: Điểm thời gian thực thi [0, 1]
        """
        # Tính trung bình thời gian thực thi
        avg_execution_time = strategy_item.get('avg_execution_time', 300.0)  # Mặc định 5 phút
        
        # Chiến lược thực thi nhanh được ưu tiên cao hơn
        # Chuẩn hóa: [0, 3600] -> [1, 0]
        score = 1.0 - min(1.0, avg_execution_time / 3600.0)
        
        return score
    
    def _calculate_market_condition_score(self, strategy_info: Dict[str, Any]) -> float:
        """
        Tính điểm điều kiện thị trường cho chiến lược.
        
        Args:
            strategy_info: Thông tin chiến lược
            
        Returns:
            float: Điểm điều kiện thị trường [0, 1]
        """
        # Lấy thông tin thị trường từ chiến lược
        market = strategy_info.get('market', '').lower()
        strategy_type = strategy_info.get('type', '').lower()
        
        # Lấy điều kiện thị trường hiện tại từ cache
        market_type = self.market_condition_cache.get('market_type', 'normal')
        volatility = self.market_condition_cache.get('volatility', 0.5)
        trend = self.market_condition_cache.get('trend', 0.0)
        
        # Tính điểm phù hợp
        score = 0.5  # Giá trị mặc định
        
        # Chiến lược dành riêng cho thị trường cụ thể
        if market_type == 'highly_volatile' and 'volatility' in strategy_type:
            score = 0.9
        elif market_type == 'trending' and 'trend' in strategy_type:
            score = 0.9
        elif market_type == 'ranging' and ('range' in strategy_type or 'oscillator' in strategy_type):
            score = 0.9
        elif market_type == 'normal':
            # Thị trường bình thường, ưu tiên các chiến lược tổng quát
            if 'general' in strategy_type or 'balanced' in strategy_type:
                score = 0.8
        
        # Điều chỉnh dựa trên thị trường được giao dịch
        market_match = False
        
        if 'crypto' in market and 'crypto' in self.market_condition_cache.get('market', ''):
            market_match = True
        elif 'forex' in market and 'forex' in self.market_condition_cache.get('market', ''):
            market_match = True
        elif 'stock' in market and 'stock' in self.market_condition_cache.get('market', ''):
            market_match = True
        
        if market_match:
            score += 0.2
        
        # Giới hạn trong khoảng [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score
    
    def _get_market_condition_score(self, market: str) -> float:
        """
        Lấy điểm điều kiện thị trường hiện tại cho một thị trường cụ thể.
        
        Args:
            market: Tên thị trường
            
        Returns:
            float: Điểm điều kiện thị trường [-1, 1]
        """
        # Mặc định điều kiện thị trường trung tính
        if not self.market_condition_cache:
            return 0.0
        
        market = market.lower()
        
        # Lấy thông tin thị trường từ cache
        if 'crypto' in market:
            volatility = self.market_condition_cache.get('crypto_volatility', 0.5)
            trend = self.market_condition_cache.get('crypto_trend', 0.0)
        elif 'forex' in market:
            volatility = self.market_condition_cache.get('forex_volatility', 0.3)
            trend = self.market_condition_cache.get('forex_trend', 0.0)
        elif 'stock' in market:
            volatility = self.market_condition_cache.get('stock_volatility', 0.2)
            trend = self.market_condition_cache.get('stock_trend', 0.0)
        else:
            volatility = self.market_condition_cache.get('volatility', 0.4)
            trend = self.market_condition_cache.get('trend', 0.0)
        
        # Tính điểm: Kết hợp xu hướng và biến động
        # Biến động cao và xu hướng rõ ràng -> điểm cao
        score = trend * 0.6 + (volatility - 0.5) * 0.4
        
        return score