"""
Điều phối viên các agent.
File này định nghĩa lớp AgentCoordinator để quản lý và điều phối
nhiều agent trong hệ thống, cho phép kết hợp quyết định từ nhiều agent
và cung cấp các cơ chế lựa chọn agent phù hợp cho từng tình huống.
"""

import os
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from enum import Enum
from pathlib import Path
import json

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from models.agents.base_agent import BaseAgent
from environments.base_environment import BaseEnvironment

class AgentSelectionMethod(Enum):
    """
    Phương pháp lựa chọn agent.
    Định nghĩa các phương pháp để chọn agent khi điều phối.
    """
    BEST_PERFORMANCE = "best_performance"  # Chọn agent có hiệu suất tốt nhất
    ROUND_ROBIN = "round_robin"            # Lần lượt chọn từng agent
    WEIGHTED_RANDOM = "weighted_random"    # Chọn ngẫu nhiên có trọng số
    CONTEXT_BASED = "context_based"        # Chọn dựa trên ngữ cảnh thị trường
    ENSEMBLE = "ensemble"                  # Kết hợp tất cả agent
    HYBRID = "hybrid"                      # Kết hợp nhiều phương pháp

class AgentCombinationMethod(Enum):
    """
    Phương pháp kết hợp quyết định từ nhiều agent.
    Định nghĩa các phương pháp để tổng hợp các quyết định.
    """
    MAJORITY_VOTE = "majority_vote"        # Bỏ phiếu theo đa số
    WEIGHTED_VOTE = "weighted_vote"        # Bỏ phiếu có trọng số
    AVERAGE_ACTION = "average_action"      # Lấy trung bình hành động
    HIGHEST_CONFIDENCE = "highest_confidence"  # Chọn agent tự tin nhất
    PERFORMANCE_BASED = "performance_based"    # Dựa trên hiệu suất gần đây
    CUSTOM = "custom"                      # Phương pháp tùy chỉnh

class AgentCoordinator:
    """
    Lớp điều phối viên các agent.
    Quản lý và điều phối nhiều agent, cung cấp cơ chế để lựa chọn
    agent phù hợp và kết hợp quyết định từ nhiều agent.
    """
    
    def __init__(
        self,
        agents: Optional[List[BaseAgent]] = None,
        env: Optional[BaseEnvironment] = None,
        selection_method: Union[str, AgentSelectionMethod] = AgentSelectionMethod.BEST_PERFORMANCE,
        combination_method: Union[str, AgentCombinationMethod] = AgentCombinationMethod.WEIGHTED_VOTE,
        name: str = "agent_coordinator",
        logger: Optional[logging.Logger] = None,
        max_agents: int = 10,
        performance_window: int = 100,
        save_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Khởi tạo điều phối viên agent.
        
        Args:
            agents: Danh sách các agent để quản lý
            env: Môi trường tương tác
            selection_method: Phương pháp lựa chọn agent
            combination_method: Phương pháp kết hợp quyết định
            name: Tên của điều phối viên
            logger: Logger tùy chỉnh
            max_agents: Số lượng agent tối đa có thể quản lý
            performance_window: Kích thước cửa sổ để theo dõi hiệu suất
            save_dir: Thư mục lưu trữ
            **kwargs: Các tham số tùy chọn khác
        """
        # Thiết lập logger
        self.logger = logger or get_logger("agent_coordinator")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập danh sách agent
        self.agents = agents or []
        if len(self.agents) > max_agents:
            self.logger.warning(f"Số lượng agent vượt quá giới hạn ({len(self.agents)} > {max_agents}), chỉ lấy {max_agents} agent đầu tiên")
            self.agents = self.agents[:max_agents]
        
        # Môi trường tương tác
        self.env = env
        
        # Phương pháp lựa chọn và kết hợp
        if isinstance(selection_method, str):
            try:
                self.selection_method = AgentSelectionMethod(selection_method)
            except ValueError:
                self.logger.warning(f"Phương pháp lựa chọn '{selection_method}' không hợp lệ, sử dụng 'best_performance'")
                self.selection_method = AgentSelectionMethod.BEST_PERFORMANCE
        else:
            self.selection_method = selection_method
        
        if isinstance(combination_method, str):
            try:
                self.combination_method = AgentCombinationMethod(combination_method)
            except ValueError:
                self.logger.warning(f"Phương pháp kết hợp '{combination_method}' không hợp lệ, sử dụng 'weighted_vote'")
                self.combination_method = AgentCombinationMethod.WEIGHTED_VOTE
        else:
            self.combination_method = combination_method
        
        # Thiết lập các thuộc tính khác
        self.name = name
        self.max_agents = max_agents
        self.performance_window = performance_window
        
        # Lưu các tham số khác
        self.kwargs = kwargs
        
        # Thư mục lưu trữ
        if save_dir is None:
            save_dir = self.system_config.get("agent.checkpoint_dir")
        self.save_dir = Path(save_dir) / self.name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo trọng số và hiệu suất của các agent
        self.agent_weights = {i: 1.0 for i in range(len(self.agents))}
        self.agent_performances = {i: [] for i in range(len(self.agents))}
        self.agent_selections = {i: 0 for i in range(len(self.agents))}
        
        # Biến theo dõi
        self.current_agent_idx = 0
        self.selection_history = []
        self.performance_history = []
        self.step_count = 0
        self.episode_count = 0
        
        # Hàm tùy chỉnh cho các phương pháp custom
        self.custom_selection_fn = kwargs.get("custom_selection_fn", None)
        self.custom_combination_fn = kwargs.get("custom_combination_fn", None)
        
        self.logger.info(
            f"Đã khởi tạo {self.__class__.__name__} với {len(self.agents)} agent, "
            f"selection_method={self.selection_method.value}, "
            f"combination_method={self.combination_method.value}"
        )
    
    def add_agent(self, agent: BaseAgent) -> bool:
        """
        Thêm agent vào hệ thống điều phối.
        
        Args:
            agent: Agent cần thêm
            
        Returns:
            True nếu thêm thành công, False nếu không
        """
        if len(self.agents) >= self.max_agents:
            self.logger.warning(f"Đã đạt giới hạn số lượng agent tối đa ({self.max_agents}), không thể thêm")
            return False
        
        # Kiểm tra xem agent đã tồn tại chưa
        for existing_agent in self.agents:
            if existing_agent.name == agent.name:
                self.logger.warning(f"Agent '{agent.name}' đã tồn tại, không thể thêm")
                return False
        
        # Thêm agent mới
        self.agents.append(agent)
        
        # Cập nhật các cấu trúc dữ liệu liên quan
        idx = len(self.agents) - 1
        self.agent_weights[idx] = 1.0
        self.agent_performances[idx] = []
        self.agent_selections[idx] = 0
        
        self.logger.info(f"Đã thêm agent '{agent.name}' vào hệ thống điều phối, tổng số agent: {len(self.agents)}")
        return True
    
    def remove_agent(self, agent_idx: int) -> bool:
        """
        Xóa agent khỏi hệ thống điều phối.
        
        Args:
            agent_idx: Chỉ số của agent cần xóa
            
        Returns:
            True nếu xóa thành công, False nếu không
        """
        if agent_idx < 0 or agent_idx >= len(self.agents):
            self.logger.warning(f"Chỉ số agent {agent_idx} không hợp lệ, phạm vi hợp lệ: 0-{len(self.agents)-1}")
            return False
        
        agent_name = self.agents[agent_idx].name
        
        # Xóa agent
        self.agents.pop(agent_idx)
        
        # Cập nhật các cấu trúc dữ liệu liên quan
        # Cần tạo lại các dict để đảm bảo chỉ số liên tục
        new_weights = {}
        new_performances = {}
        new_selections = {}
        
        for i, idx in enumerate(range(len(self.agents))):
            if i < agent_idx:
                # Các agent trước agent bị xóa giữ nguyên chỉ số
                new_weights[i] = self.agent_weights[i]
                new_performances[i] = self.agent_performances[i]
                new_selections[i] = self.agent_selections[i]
            else:
                # Các agent sau agent bị xóa giảm chỉ số đi 1
                new_weights[i] = self.agent_weights[i+1]
                new_performances[i] = self.agent_performances[i+1]
                new_selections[i] = self.agent_selections[i+1]
        
        self.agent_weights = new_weights
        self.agent_performances = new_performances
        self.agent_selections = new_selections
        
        # Cập nhật current_agent_idx nếu cần
        if self.current_agent_idx >= len(self.agents):
            self.current_agent_idx = 0
        
        self.logger.info(f"Đã xóa agent '{agent_name}' khỏi hệ thống điều phối, tổng số agent còn lại: {len(self.agents)}")
        return True
    
    def select_agent(self, state: np.ndarray, context: Optional[Dict[str, Any]] = None) -> int:
        """
        Chọn agent phù hợp dựa trên trạng thái hiện tại và ngữ cảnh.
        
        Args:
            state: Trạng thái hiện tại
            context: Thông tin ngữ cảnh bổ sung (tâm lý thị trường, biến động, etc.)
            
        Returns:
            Chỉ số của agent được chọn
        """
        if not self.agents:
            self.logger.warning("Không có agent nào để chọn")
            return -1
        
        # Chọn agent theo phương pháp đã cấu hình
        selected_idx = -1
        
        if self.selection_method == AgentSelectionMethod.BEST_PERFORMANCE:
            selected_idx = self._select_best_performance_agent()
        elif self.selection_method == AgentSelectionMethod.ROUND_ROBIN:
            selected_idx = self._select_round_robin_agent()
        elif self.selection_method == AgentSelectionMethod.WEIGHTED_RANDOM:
            selected_idx = self._select_weighted_random_agent()
        elif self.selection_method == AgentSelectionMethod.CONTEXT_BASED:
            selected_idx = self._select_context_based_agent(state, context)
        elif self.selection_method == AgentSelectionMethod.ENSEMBLE:
            # Với ENSEMBLE, -1 nghĩa là sẽ sử dụng tất cả các agent
            selected_idx = -1
        elif self.selection_method == AgentSelectionMethod.HYBRID:
            selected_idx = self._select_hybrid_agent(state, context)
        
        # Nếu là phương pháp CUSTOM
        if selected_idx == -1 and self.custom_selection_fn is not None:
            try:
                selected_idx = self.custom_selection_fn(self, state, context)
            except Exception as e:
                self.logger.error(f"Lỗi khi sử dụng hàm lựa chọn tùy chỉnh: {str(e)}")
                # Fallback to best performance
                selected_idx = self._select_best_performance_agent()
        
        # Kiểm tra kết quả lựa chọn
        if selected_idx == -1 or selected_idx >= len(self.agents):
            # Nếu là ENSEMBLE hoặc lỗi, trả về -1
            if self.selection_method == AgentSelectionMethod.ENSEMBLE:
                return -1
            
            # Nếu lỗi, sử dụng agent đầu tiên
            self.logger.warning(f"Lựa chọn agent không hợp lệ: {selected_idx}, sử dụng agent 0")
            selected_idx = 0
        
        # Cập nhật biến theo dõi
        self.current_agent_idx = selected_idx
        self.agent_selections[selected_idx] += 1
        self.selection_history.append(selected_idx)
        
        return selected_idx
    
    def act(self, state: np.ndarray, explore: bool = True, context: Optional[Dict[str, Any]] = None) -> Union[int, np.ndarray]:
        """
        Chọn hành động dựa trên trạng thái hiện tại.
        
        Args:
            state: Trạng thái hiện tại
            explore: Có thực hiện khám phá không
            context: Thông tin ngữ cảnh bổ sung
            
        Returns:
            Hành động được chọn
        """
        if not self.agents:
            self.logger.error("Không có agent nào để thực hiện hành động")
            # Trả về hành động mặc định (0)
            return 0
        
        # Chọn agent hoặc ensemble
        selected_idx = self.select_agent(state, context)
        
        # Kiểm tra phương pháp điều phối
        if selected_idx == -1 or self.selection_method == AgentSelectionMethod.ENSEMBLE:
            # Kết hợp hành động từ tất cả các agent
            return self.combine_actions(state, explore, context)
        
        # Sử dụng agent được chọn
        return self.agents[selected_idx].act(state, explore)
    
    def combine_actions(self, state: np.ndarray, explore: bool = True, context: Optional[Dict[str, Any]] = None) -> Union[int, np.ndarray]:
        """
        Kết hợp hành động từ nhiều agent.
        
        Args:
            state: Trạng thái hiện tại
            explore: Có thực hiện khám phá không
            context: Thông tin ngữ cảnh bổ sung
            
        Returns:
            Hành động được kết hợp
        """
        if not self.agents:
            self.logger.error("Không có agent nào để kết hợp hành động")
            return 0
        
        # Lấy hành động từ tất cả các agent
        actions = []
        confidences = []
        
        for agent in self.agents:
            try:
                # Thử lấy hành động và độ tự tin (nếu có)
                if hasattr(agent, 'act_with_confidence'):
                    action, confidence = agent.act_with_confidence(state, explore)
                else:
                    action = agent.act(state, explore)
                    # Giả định độ tự tin là 1.0 nếu không có
                    confidence = 1.0
                
                actions.append(action)
                confidences.append(confidence)
            except Exception as e:
                self.logger.warning(f"Lỗi khi lấy hành động từ agent '{agent.name}': {str(e)}")
                # Bỏ qua agent lỗi
        
        if not actions:
            self.logger.error("Không thể lấy hành động từ bất kỳ agent nào")
            return 0
        
        # Kết hợp hành động theo phương pháp đã cấu hình
        combined_action = None
        
        if self.combination_method == AgentCombinationMethod.MAJORITY_VOTE:
            combined_action = self._combine_majority_vote(actions)
        elif self.combination_method == AgentCombinationMethod.WEIGHTED_VOTE:
            combined_action = self._combine_weighted_vote(actions)
        elif self.combination_method == AgentCombinationMethod.AVERAGE_ACTION:
            combined_action = self._combine_average_action(actions)
        elif self.combination_method == AgentCombinationMethod.HIGHEST_CONFIDENCE:
            combined_action = self._combine_highest_confidence(actions, confidences)
        elif self.combination_method == AgentCombinationMethod.PERFORMANCE_BASED:
            combined_action = self._combine_performance_based(actions)
        elif self.combination_method == AgentCombinationMethod.CUSTOM and self.custom_combination_fn is not None:
            try:
                combined_action = self.custom_combination_fn(actions, confidences, self.agent_weights)
            except Exception as e:
                self.logger.error(f"Lỗi khi sử dụng hàm kết hợp tùy chỉnh: {str(e)}")
                # Fallback to majority vote
                combined_action = self._combine_majority_vote(actions)
        
        # Kiểm tra kết quả kết hợp
        if combined_action is None:
            self.logger.warning("Kết hợp hành động thất bại, sử dụng hành động đầu tiên")
            combined_action = actions[0]
        
        return combined_action
    
    def update_performance(self, agent_idx: int, reward: float) -> None:
        """
        Cập nhật hiệu suất của agent.
        
        Args:
            agent_idx: Chỉ số của agent
            reward: Phần thưởng nhận được
        """
        if agent_idx < 0 or agent_idx >= len(self.agents):
            self.logger.warning(f"Chỉ số agent {agent_idx} không hợp lệ, không cập nhật hiệu suất")
            return
        
        # Thêm phần thưởng vào lịch sử hiệu suất
        self.agent_performances[agent_idx].append(reward)
        
        # Giới hạn kích thước cửa sổ hiệu suất
        if len(self.agent_performances[agent_idx]) > self.performance_window:
            self.agent_performances[agent_idx].pop(0)
        
        # Cập nhật trọng số dựa trên hiệu suất
        self._update_agent_weights()
    
    def _update_agent_weights(self) -> None:
        """
        Cập nhật trọng số cho các agent dựa trên hiệu suất gần đây.
        """
        for idx in range(len(self.agents)):
            performances = self.agent_performances[idx]
            
            if not performances:
                # Nếu không có dữ liệu hiệu suất, giữ nguyên trọng số
                continue
            
            # Tính trọng số dựa trên hiệu suất trung bình
            avg_performance = np.mean(performances)
            
            # Điều chỉnh trọng số (sử dụng softmax hoặc normalization)
            # Ở đây sử dụng phương pháp đơn giản: trọng số tỉ lệ thuận với hiệu suất
            if avg_performance < 0:
                # Nếu hiệu suất âm, sử dụng trọng số tối thiểu
                self.agent_weights[idx] = 0.1
            else:
                # Nếu hiệu suất dương, tỉ lệ thuận với hiệu suất
                self.agent_weights[idx] = 1.0 + avg_performance
        
        # Chuẩn hóa trọng số
        total_weight = sum(self.agent_weights.values())
        if total_weight > 0:
            for idx in self.agent_weights:
                self.agent_weights[idx] /= total_weight
    
    def _select_best_performance_agent(self) -> int:
        """
        Chọn agent có hiệu suất tốt nhất.
        
        Returns:
            Chỉ số của agent được chọn
        """
        best_idx = 0
        best_performance = float('-inf')
        
        for idx in range(len(self.agents)):
            performances = self.agent_performances[idx]
            
            if not performances:
                # Nếu không có dữ liệu hiệu suất, bỏ qua
                continue
            
            avg_performance = np.mean(performances)
            
            if avg_performance > best_performance:
                best_performance = avg_performance
                best_idx = idx
        
        return best_idx
    
    def _select_round_robin_agent(self) -> int:
        """
        Chọn agent theo phương pháp round-robin.
        
        Returns:
            Chỉ số của agent được chọn
        """
        selected_idx = self.current_agent_idx
        
        # Chọn agent tiếp theo cho lần sau
        self.current_agent_idx = (self.current_agent_idx + 1) % len(self.agents)
        
        return selected_idx
    
    def _select_weighted_random_agent(self) -> int:
        """
        Chọn agent ngẫu nhiên có trọng số.
        
        Returns:
            Chỉ số của agent được chọn
        """
        weights = list(self.agent_weights.values())
        indices = list(range(len(self.agents)))
        
        # Chuẩn hóa trọng số
        total_weight = sum(weights)
        if total_weight <= 0:
            # Nếu tổng trọng số <= 0, sử dụng phân phối đều
            weights = [1.0] * len(self.agents)
            total_weight = len(self.agents)
        
        normalized_weights = [w / total_weight for w in weights]
        
        # Chọn ngẫu nhiên theo trọng số
        selected_idx = np.random.choice(indices, p=normalized_weights)
        
        return selected_idx
    
    def _select_context_based_agent(self, state: np.ndarray, context: Optional[Dict[str, Any]] = None) -> int:
        """
        Chọn agent dựa trên ngữ cảnh.
        
        Args:
            state: Trạng thái hiện tại
            context: Thông tin ngữ cảnh bổ sung
            
        Returns:
            Chỉ số của agent được chọn
        """
        if context is None or not context:
            # Nếu không có thông tin ngữ cảnh, fallback to weighted random
            return self._select_weighted_random_agent()
        
        # Tính điểm phù hợp cho từng agent
        scores = np.zeros(len(self.agents))
        
        # Ví dụ: Phân tích ngữ cảnh để tính điểm
        # 1. Kiểm tra độ biến động thị trường
        volatility = context.get('volatility', 0.0)
        # 2. Kiểm tra xu hướng thị trường
        trend = context.get('trend', 0.0)
        # 3. Kiểm tra tâm lý thị trường
        sentiment = context.get('sentiment', 0.0)
        
        for idx, agent in enumerate(self.agents):
            # Phân tích đặc điểm của agent và ngữ cảnh
            # Ví dụ: Agent DQN thích hợp với thị trường ít biến động
            if 'dqn' in agent.name.lower():
                scores[idx] += (1.0 - volatility) * 0.5  # Trọng số 0.5 cho yếu tố biến động
            
            # Agent PPO thích hợp với thị trường biến động
            elif 'ppo' in agent.name.lower():
                scores[idx] += volatility * 0.5
            
            # Các quy tắc khác dựa trên đặc điểm agent và ngữ cảnh
            # ...
            
            # Kết hợp với hiệu suất gần đây
            performances = self.agent_performances[idx]
            if performances:
                avg_performance = np.mean(performances)
                scores[idx] += avg_performance * 0.5  # Trọng số 0.5 cho hiệu suất
        
        # Chọn agent có điểm cao nhất
        if np.all(scores == 0):
            # Nếu tất cả điểm đều bằng 0, fallback to weighted random
            return self._select_weighted_random_agent()
        
        return np.argmax(scores)
    
    def _select_hybrid_agent(self, state: np.ndarray, context: Optional[Dict[str, Any]] = None) -> int:
        """
        Chọn agent theo phương pháp kết hợp nhiều phương pháp.
        
        Args:
            state: Trạng thái hiện tại
            context: Thông tin ngữ cảnh bổ sung
            
        Returns:
            Chỉ số của agent được chọn
        """
        # Hybrid có thể kết hợp nhiều phương pháp lựa chọn
        # Ví dụ: 50% thời gian sử dụng best performance, 30% context-based, 20% weighted random
        
        # Lấy random số từ 0 đến 1
        r = np.random.random()
        
        if r < 0.5:
            # 50% thời gian sử dụng best performance
            return self._select_best_performance_agent()
        elif r < 0.8:
            # 30% thời gian sử dụng context-based
            return self._select_context_based_agent(state, context)
        else:
            # 20% thời gian sử dụng weighted random
            return self._select_weighted_random_agent()
    
    def _combine_majority_vote(self, actions: List[Union[int, np.ndarray]]) -> Union[int, np.ndarray]:
        """
        Kết hợp hành động bằng bỏ phiếu đa số.
        
        Args:
            actions: Danh sách các hành động
            
        Returns:
            Hành động được bỏ phiếu nhiều nhất
        """
        # Kiểm tra kiểu hành động
        if isinstance(actions[0], np.ndarray):
            # Xử lý hành động dạng mảng
            # Đối với hành động liên tục, bỏ phiếu đa số không áp dụng được trực tiếp
            # Nên sử dụng trung bình cộng
            return np.mean(actions, axis=0)
        else:
            # Xử lý hành động rời rạc
            # Đếm số lượng mỗi hành động
            action_counts = {}
            for action in actions:
                if action not in action_counts:
                    action_counts[action] = 0
                action_counts[action] += 1
            
            # Tìm hành động được bỏ phiếu nhiều nhất
            max_count = 0
            most_voted_action = actions[0]  # Mặc định là hành động đầu tiên
            
            for action, count in action_counts.items():
                if count > max_count:
                    max_count = count
                    most_voted_action = action
            
            return most_voted_action
    
    def _combine_weighted_vote(self, actions: List[Union[int, np.ndarray]]) -> Union[int, np.ndarray]:
        """
        Kết hợp hành động bằng bỏ phiếu có trọng số.
        
        Args:
            actions: Danh sách các hành động
            
        Returns:
            Hành động được bỏ phiếu có trọng số nhiều nhất
        """
        # Kiểm tra kiểu hành động
        if isinstance(actions[0], np.ndarray):
            # Xử lý hành động dạng mảng
            # Lấy trung bình có trọng số
            weighted_sum = np.zeros_like(actions[0], dtype=float)
            total_weight = 0.0
            
            for idx, action in enumerate(actions):
                if idx < len(self.agents):
                    weight = self.agent_weights[idx]
                    weighted_sum += action * weight
                    total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return actions[0]  # Mặc định là hành động đầu tiên
        else:
            # Xử lý hành động rời rạc
            # Đếm số lượng có trọng số cho mỗi hành động
            weighted_counts = {}
            
            for idx, action in enumerate(actions):
                if idx < len(self.agents):
                    weight = self.agent_weights[idx]
                    
                    if action not in weighted_counts:
                        weighted_counts[action] = 0
                    
                    weighted_counts[action] += weight
            
            # Tìm hành động có tổng trọng số cao nhất
            max_weight = 0
            most_weighted_action = actions[0]  # Mặc định là hành động đầu tiên
            
            for action, weight in weighted_counts.items():
                if weight > max_weight:
                    max_weight = weight
                    most_weighted_action = action
            
            return most_weighted_action
    
    def _combine_average_action(self, actions: List[Union[int, np.ndarray]]) -> Union[int, np.ndarray]:
        """
        Kết hợp hành động bằng cách lấy trung bình.
        
        Args:
            actions: Danh sách các hành động
            
        Returns:
            Hành động trung bình
        """
        # Kiểm tra kiểu hành động
        if isinstance(actions[0], np.ndarray):
            # Xử lý hành động dạng mảng
            return np.mean(actions, axis=0)
        else:
            # Xử lý hành động rời rạc
            # Đối với hành động rời rạc, lấy trung bình không phải lúc nào cũng có ý nghĩa
            # Trong trường hợp này, làm tròn đến hành động gần nhất
            return int(round(np.mean(actions)))
    
    def _combine_highest_confidence(self, actions: List[Union[int, np.ndarray]], confidences: List[float]) -> Union[int, np.ndarray]:
        """
        Chọn hành động có độ tự tin cao nhất.
        
        Args:
            actions: Danh sách các hành động
            confidences: Danh sách độ tự tin tương ứng
            
        Returns:
            Hành động có độ tự tin cao nhất
        """
        if not confidences:
            # Nếu không có thông tin về độ tự tin, fallback to majority vote
            return self._combine_majority_vote(actions)
        
        # Tìm hành động có độ tự tin cao nhất
        max_confidence_idx = np.argmax(confidences)
        return actions[max_confidence_idx]
    
    def _combine_performance_based(self, actions: List[Union[int, np.ndarray]]) -> Union[int, np.ndarray]:
        """
        Kết hợp hành động dựa trên hiệu suất của các agent.
        
        Args:
            actions: Danh sách các hành động
            
        Returns:
            Hành động được kết hợp dựa trên hiệu suất
        """
        # Tương tự weighted vote, nhưng trọng số dựa trên hiệu suất
        # Kiểm tra kiểu hành động
        if isinstance(actions[0], np.ndarray):
            # Xử lý hành động dạng mảng
            weighted_sum = np.zeros_like(actions[0], dtype=float)
            total_weight = 0.0
            
            for idx, action in enumerate(actions):
                if idx < len(self.agents):
                    # Tính trọng số dựa trên hiệu suất
                    performances = self.agent_performances[idx]
                    if performances:
                        weight = np.mean(performances)
                        # Đảm bảo trọng số không âm
                        weight = max(0.1, weight + 1.0)
                    else:
                        weight = 0.1  # Trọng số mặc định nếu không có dữ liệu hiệu suất
                    
                    weighted_sum += action * weight
                    total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return actions[0]  # Mặc định là hành động đầu tiên
        else:
            # Xử lý hành động rời rạc
            weighted_counts = {}
            
            for idx, action in enumerate(actions):
                if idx < len(self.agents):
                    # Tính trọng số dựa trên hiệu suất
                    performances = self.agent_performances[idx]
                    if performances:
                        weight = np.mean(performances)
                        # Đảm bảo trọng số không âm
                        weight = max(0.1, weight + 1.0)
                    else:
                        weight = 0.1  # Trọng số mặc định nếu không có dữ liệu hiệu suất
                    
                    if action not in weighted_counts:
                        weighted_counts[action] = 0
                    
                    weighted_counts[action] += weight
            
            # Tìm hành động có tổng trọng số cao nhất
            max_weight = 0
            most_weighted_action = actions[0]  # Mặc định là hành động đầu tiên
            
            for action, weight in weighted_counts.items():
                if weight > max_weight:
                    max_weight = weight
                    most_weighted_action = action
            
            return most_weighted_action
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê hiệu suất của các agent.
        
        Returns:
            Dict chứa thống kê hiệu suất
        """
        stats = {}
        
        for idx, agent in enumerate(self.agents):
            performances = self.agent_performances[idx]
            selections = self.agent_selections[idx]
            weight = self.agent_weights[idx]
            
            if performances:
                avg_performance = np.mean(performances)
                min_performance = np.min(performances)
                max_performance = np.max(performances)
                std_performance = np.std(performances)
            else:
                avg_performance = 0.0
                min_performance = 0.0
                max_performance = 0.0
                std_performance = 0.0
            
            stats[agent.name] = {
                "avg_performance": avg_performance,
                "min_performance": min_performance,
                "max_performance": max_performance,
                "std_performance": std_performance,
                "selections": selections,
                "weight": weight
            }
        
        return stats
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Lưu trạng thái của điều phối viên.
        
        Args:
            path: Đường dẫn lưu trạng thái
        """
        if path is None:
            path = self.save_dir / f"{self.name}_state.json"
        
        # Tạo dict trạng thái
        state = {
            "name": self.name,
            "selection_method": self.selection_method.value,
            "combination_method": self.combination_method.value,
            "agent_weights": {str(k): v for k, v in self.agent_weights.items()},
            "agent_selections": {str(k): v for k, v in self.agent_selections.items()},
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "current_agent_idx": self.current_agent_idx,
            "selection_history": self.selection_history[-100:],  # Chỉ lưu 100 lựa chọn gần nhất
            "created_at": time.time()
        }
        
        # Lưu vào file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Đã lưu trạng thái điều phối viên tại {path}")
    
    def load(self, path: Union[str, Path]) -> bool:
        """
        Tải trạng thái của điều phối viên.
        
        Args:
            path: Đường dẫn tải trạng thái
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        path = Path(path)
        if not path.exists():
            self.logger.warning(f"Không tìm thấy file trạng thái tại {path}")
            return False
        
        try:
            # Đọc từ file
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Cập nhật trạng thái
            if "name" in state:
                self.name = state["name"]
            
            if "selection_method" in state:
                try:
                    self.selection_method = AgentSelectionMethod(state["selection_method"])
                except ValueError:
                    self.logger.warning(f"Phương pháp lựa chọn '{state['selection_method']}' không hợp lệ, giữ nguyên giá trị hiện tại")
            
            if "combination_method" in state:
                try:
                    self.combination_method = AgentCombinationMethod(state["combination_method"])
                except ValueError:
                    self.logger.warning(f"Phương pháp kết hợp '{state['combination_method']}' không hợp lệ, giữ nguyên giá trị hiện tại")
            
            if "agent_weights" in state:
                # Chuyển key từ string về int
                self.agent_weights = {int(k): v for k, v in state["agent_weights"].items()}
            
            if "agent_selections" in state:
                # Chuyển key từ string về int
                self.agent_selections = {int(k): v for k, v in state["agent_selections"].items()}
            
            if "step_count" in state:
                self.step_count = state["step_count"]
            
            if "episode_count" in state:
                self.episode_count = state["episode_count"]
            
            if "current_agent_idx" in state:
                self.current_agent_idx = state["current_agent_idx"]
            
            if "selection_history" in state:
                self.selection_history = state["selection_history"]
            
            self.logger.info(f"Đã tải trạng thái điều phối viên từ {path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải trạng thái điều phối viên: {str(e)}")
            return False