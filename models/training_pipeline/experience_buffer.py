"""
Bộ đệm kinh nghiệm cho huấn luyện agent.
File này định nghĩa các lớp quản lý bộ nhớ kinh nghiệm (Experience Replay),
được sử dụng để lưu trữ và lấy mẫu các trải nghiệm từ quá trình tương tác với môi trường.
"""

import random
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque, namedtuple

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger

# Định nghĩa kiểu dữ liệu Experience
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceBuffer:
    """
    Lớp quản lý bộ đệm kinh nghiệm cho các thuật toán Reinforcement Learning.
    Hỗ trợ nhiều loại bộ đệm khác nhau: tiêu chuẩn, có ưu tiên, có cân bằng.
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        state_dim: Optional[Union[int, Tuple[int, ...]]] = None,
        action_dim: Optional[Union[int, Tuple[int, ...]]] = None,
        buffer_type: str = "uniform",
        alpha: float = 0.6,  # Cho prioritized replay
        beta: float = 0.4,   # Cho importance sampling
        beta_increment: float = 0.001,  # Tăng beta theo thời gian
        epsilon: float = 1e-5,  # Epsilon nhỏ để tránh ưu tiên 0
        sample_consecutive: bool = False,  # Cho recurrent networks
        sequence_length: int = 10,  # Cho recurrent networks
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo bộ đệm kinh nghiệm.
        
        Args:
            buffer_size: Kích thước tối đa của bộ đệm
            state_dim: Kích thước không gian trạng thái
            action_dim: Kích thước không gian hành động
            buffer_type: Loại bộ đệm ('uniform', 'prioritized', 'balanced')
            alpha: Hệ số alpha cho prioritized replay (càng lớn càng ưu tiên)
            beta: Hệ số beta cho importance sampling (càng lớn càng ít bias)
            beta_increment: Tốc độ tăng beta theo thời gian
            epsilon: Giá trị nhỏ để tránh ưu tiên bằng 0
            sample_consecutive: Lấy mẫu tuần tự (cho mạng RNN)
            sequence_length: Độ dài chuỗi khi lấy mẫu tuần tự
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("experience_buffer")
        
        # Thiết lập các tham số
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_type = buffer_type.lower()
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.sample_consecutive = sample_consecutive
        self.sequence_length = sequence_length
        
        # Khởi tạo biến đếm
        self.current_size = 0
        self.next_idx = 0
        
        # Khởi tạo bộ đệm dựa trên loại
        if self.buffer_type == "uniform":
            # Bộ đệm thông thường với deque
            self.buffer = deque(maxlen=buffer_size)
            
        elif self.buffer_type == "prioritized":
            # Bộ đệm có ưu tiên
            # Dữ liệu được lưu trong list/numpy array, không dùng deque
            self.buffer = [None] * buffer_size
            # Mảng lưu độ ưu tiên
            self.priorities = np.zeros(buffer_size, dtype=np.float32)
            
        elif self.buffer_type == "balanced":
            # Bộ đệm cân bằng giữa các loại trải nghiệm
            # Thường dùng để cân bằng giữa các hành động đối với các bài toán có phân phối hành động mất cân bằng
            self.buffer = {}  # Dict với key là hành động/nhóm hành động
            self.action_counts = {}  # Đếm số trải nghiệm của mỗi hành động
            
        else:
            self.logger.warning(f"Loại bộ đệm không hợp lệ: {buffer_type}, sử dụng 'uniform'")
            self.buffer_type = "uniform"
            self.buffer = deque(maxlen=buffer_size)
        
        # Các tham số bổ sung
        self.kwargs = kwargs
        
        self.logger.info(
            f"Đã khởi tạo {self.__class__.__name__} với buffer_type={buffer_type}, "
            f"buffer_size={buffer_size}"
        )
    
    def add(
        self, 
        state: np.ndarray, 
        action: Union[int, np.ndarray], 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        priority: Optional[float] = None
    ) -> None:
        """
        Thêm trải nghiệm vào bộ đệm.
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái kế tiếp
            done: Đã kết thúc episode hay chưa
            priority: Độ ưu tiên (cho prioritized replay)
        """
        # Tạo đối tượng Experience
        experience = Experience(state, action, reward, next_state, done)
        
        if self.buffer_type == "uniform":
            # Thêm vào đuôi deque
            self.buffer.append(experience)
            self.current_size = len(self.buffer)
            
        elif self.buffer_type == "prioritized":
            # Thêm vào vị trí next_idx
            self.buffer[self.next_idx] = experience
            
            # Cập nhật ưu tiên
            if priority is not None:
                self.priorities[self.next_idx] = priority
            else:
                # Nếu không có ưu tiên, sử dụng ưu tiên cao nhất hiện tại
                self.priorities[self.next_idx] = np.max(self.priorities) if self.current_size > 0 else 1.0
            
            # Cập nhật chỉ số và kích thước
            self.next_idx = (self.next_idx + 1) % self.buffer_size
            self.current_size = min(self.current_size + 1, self.buffer_size)
            
        elif self.buffer_type == "balanced":
            # Xác định key cho action (nếu là mảng thì chuyển thành tuple để hash được)
            action_key = tuple(action) if isinstance(action, np.ndarray) else action
            
            # Khởi tạo buffer cho action này nếu chưa có
            if action_key not in self.buffer:
                self.buffer[action_key] = deque(maxlen=self.buffer_size // 10)  # Mỗi hành động lưu tối đa 1/10 buffer
                self.action_counts[action_key] = 0
            
            # Thêm vào buffer tương ứng
            self.buffer[action_key].append(experience)
            self.action_counts[action_key] += 1
            
            # Cập nhật kích thước tổng
            self.current_size = sum(len(buf) for buf in self.buffer.values())
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Cập nhật độ ưu tiên cho các trải nghiệm.
        Chỉ áp dụng cho loại buffer 'prioritized'.
        
        Args:
            indices: Danh sách chỉ số cần cập nhật
            priorities: Danh sách độ ưu tiên mới
        """
        if self.buffer_type != "prioritized":
            self.logger.warning("update_priorities chỉ áp dụng cho buffer loại 'prioritized'")
            return
        
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < self.buffer_size:
                self.priorities[idx] = priority + self.epsilon
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], Optional[List[int]], Optional[np.ndarray]]:
        """
        Lấy mẫu ngẫu nhiên từ bộ đệm.
        
        Args:
            batch_size: Kích thước batch cần lấy
            
        Returns:
            Tuple (samples, indices, weights)
            - samples: Danh sách các trải nghiệm đã lấy
            - indices: Chỉ số của các mẫu (cho prioritized)
            - weights: Trọng số importance sampling (cho prioritized)
        """
        if self.current_size == 0:
            # Buffer trống
            return [], None, None
        
        # Đảm bảo batch_size không vượt quá kích thước hiện tại
        batch_size = min(batch_size, self.current_size)
        
        if self.buffer_type == "uniform":
            # Lấy mẫu đơn giản từ deque
            if self.sample_consecutive:
                # Lấy mẫu các chuỗi trải nghiệm liên tiếp
                samples = self._sample_consecutive_uniform(batch_size)
            else:
                # Lấy mẫu ngẫu nhiên
                samples = random.sample(self.buffer, batch_size)
            
            return samples, None, None
            
        elif self.buffer_type == "prioritized":
            return self._sample_prioritized(batch_size)
            
        elif self.buffer_type == "balanced":
            return self._sample_balanced(batch_size)
        
        # Mặc định
        return [], None, None
    
    def _sample_consecutive_uniform(self, batch_size: int) -> List[List[Experience]]:
        """
        Lấy mẫu các chuỗi trải nghiệm liên tiếp từ bộ đệm thông thường.
        
        Args:
            batch_size: Số lượng chuỗi cần lấy
            
        Returns:
            Danh sách các chuỗi trải nghiệm, mỗi chuỗi có độ dài sequence_length
        """
        # Đảm bảo có đủ dữ liệu để lấy chuỗi
        if len(self.buffer) < self.sequence_length:
            return []
        
        samples = []
        for _ in range(batch_size):
            # Chọn vị trí bắt đầu ngẫu nhiên
            start_idx = random.randint(0, len(self.buffer) - self.sequence_length)
            # Lấy chuỗi liên tiếp
            sequence = list(self.buffer)[start_idx:start_idx + self.sequence_length]
            samples.append(sequence)
        
        return samples
    
    def _sample_prioritized(self, batch_size: int) -> Tuple[List[Experience], List[int], np.ndarray]:
        """
        Lấy mẫu từ bộ đệm có ưu tiên.
        
        Args:
            batch_size: Kích thước batch cần lấy
            
        Returns:
            Tuple (samples, indices, weights)
        """
        # Chỉ lấy mẫu từ phần đã có dữ liệu
        if self.current_size < self.buffer_size:
            priorities = self.priorities[:self.current_size]
        else:
            priorities = self.priorities
        
        # Tính xác suất lấy mẫu dựa trên ưu tiên
        probs = priorities ** self.alpha
        probs = probs / np.sum(probs)
        
        # Lấy mẫu indices
        indices = np.random.choice(
            len(probs), batch_size, replace=len(probs) < batch_size, p=probs
        )
        
        # Tính trọng số importance sampling
        # Công thức: w_i = (1/N * 1/P(i))^beta
        weights = (len(probs) * probs[indices]) ** (-self.beta)
        # Chuẩn hóa về [0, 1]
        weights = weights / np.max(weights)
        
        # Tăng beta theo thời gian
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Lấy mẫu tương ứng
        samples = [self.buffer[idx] for idx in indices]
        
        return samples, indices, weights
    
    def _sample_balanced(self, batch_size: int) -> Tuple[List[Experience], None, None]:
        """
        Lấy mẫu từ bộ đệm cân bằng.
        
        Args:
            batch_size: Kích thước batch cần lấy
            
        Returns:
            Tuple (samples, None, None)
        """
        # Kiểm tra xem có hành động nào được lưu chưa
        if not self.buffer:
            return [], None, None
        
        # Lấy danh sách hành động
        actions = list(self.buffer.keys())
        
        # Tính số mẫu cần lấy từ mỗi hành động
        samples_per_action = batch_size // len(actions)
        remaining = batch_size % len(actions)
        
        samples = []
        
        for action in actions:
            # Số mẫu cần lấy từ hành động này
            n_samples = min(samples_per_action, len(self.buffer[action]))
            if n_samples > 0:
                action_samples = random.sample(list(self.buffer[action]), n_samples)
                samples.extend(action_samples)
        
        # Bổ sung các mẫu còn thiếu từ các hành động ngẫu nhiên
        if remaining > 0 and len(samples) < batch_size:
            additional_samples = []
            all_experiences = [exp for buf in self.buffer.values() for exp in buf]
            
            if all_experiences:
                n_additional = min(batch_size - len(samples), len(all_experiences))
                additional_samples = random.sample(all_experiences, n_additional)
                samples.extend(additional_samples)
        
        return samples, None, None
    
    def get_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Lấy batch dữ liệu đã được chuyển đổi thành tensor numpy.
        
        Args:
            batch_size: Kích thước batch cần lấy
            
        Returns:
            Dict chứa batch dữ liệu dạng numpy arrays
        """
        # Lấy mẫu từ buffer
        samples, indices, weights = self.sample(batch_size)
        
        if not samples:
            return {}
        
        # Kiểm tra xem samples có phải là chuỗi liên tiếp không
        is_sequence = self.sample_consecutive and isinstance(samples[0], list)
        
        if is_sequence:
            # Xử lý dữ liệu chuỗi
            return self._process_sequence_batch(samples, indices, weights)
        else:
            # Xử lý dữ liệu thông thường
            return self._process_batch(samples, indices, weights)
    
    def _process_batch(
        self, 
        samples: List[Experience], 
        indices: Optional[List[int]], 
        weights: Optional[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Xử lý batch dữ liệu thông thường, chuyển đổi thành tensor numpy.
        
        Args:
            samples: Danh sách các trải nghiệm
            indices: Chỉ số của các mẫu
            weights: Trọng số importance sampling
            
        Returns:
            Dict chứa batch dữ liệu dạng numpy arrays
        """
        # Khởi tạo các mảng numpy
        states = np.array([exp.state for exp in samples])
        
        # Xử lý actions (có thể là int hoặc array)
        if isinstance(samples[0].action, np.ndarray):
            actions = np.array([exp.action for exp in samples])
        else:
            actions = np.array([exp.action for exp in samples], dtype=np.int32)
        
        rewards = np.array([exp.reward for exp in samples], dtype=np.float32)
        next_states = np.array([exp.next_state for exp in samples])
        dones = np.array([exp.done for exp in samples], dtype=np.float32)
        
        # Tạo dict kết quả
        batch = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        
        # Thêm indices và weights nếu có
        if indices is not None:
            batch['indices'] = np.array(indices)
        
        if weights is not None:
            batch['weights'] = weights
        
        return batch
    
    def _process_sequence_batch(
        self, 
        samples: List[List[Experience]], 
        indices: Optional[List[int]], 
        weights: Optional[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Xử lý batch dữ liệu chuỗi, chuyển đổi thành tensor numpy.
        
        Args:
            samples: Danh sách các chuỗi trải nghiệm
            indices: Chỉ số của các mẫu
            weights: Trọng số importance sampling
            
        Returns:
            Dict chứa batch dữ liệu dạng numpy arrays
        """
        # Kích thước batch và độ dài chuỗi
        batch_size = len(samples)
        seq_length = len(samples[0])
        
        # Khởi tạo các mảng numpy
        if isinstance(samples[0][0].state, np.ndarray):
            state_shape = samples[0][0].state.shape
            states = np.zeros((batch_size, seq_length, *state_shape), dtype=np.float32)
            next_states = np.zeros((batch_size, seq_length, *state_shape), dtype=np.float32)
        else:
            states = np.zeros((batch_size, seq_length), dtype=np.float32)
            next_states = np.zeros((batch_size, seq_length), dtype=np.float32)
        
        # Kiểm tra kiểu action
        if isinstance(samples[0][0].action, np.ndarray):
            action_shape = samples[0][0].action.shape
            actions = np.zeros((batch_size, seq_length, *action_shape), dtype=np.float32)
        else:
            actions = np.zeros((batch_size, seq_length), dtype=np.int32)
        
        rewards = np.zeros((batch_size, seq_length), dtype=np.float32)
        dones = np.zeros((batch_size, seq_length), dtype=np.float32)
        
        # Điền dữ liệu
        for i, sequence in enumerate(samples):
            for j, exp in enumerate(sequence):
                states[i, j] = exp.state
                
                if isinstance(exp.action, np.ndarray):
                    actions[i, j] = exp.action
                else:
                    actions[i, j] = exp.action
                
                rewards[i, j] = exp.reward
                next_states[i, j] = exp.next_state
                dones[i, j] = exp.done
        
        # Tạo dict kết quả
        batch = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        
        return batch
    
    def update_batch(self, indices: List[int], td_errors: List[float]) -> None:
        """
        Cập nhật độ ưu tiên cho một batch dựa trên TD errors.
        
        Args:
            indices: Chỉ số của các mẫu cần cập nhật
            td_errors: TD errors tương ứng
        """
        # Chỉ áp dụng cho prioritized replay
        if self.buffer_type != "prioritized":
            return
        
        # Chuyển đổi TD errors thành độ ưu tiên
        priorities = np.abs(td_errors) + self.epsilon
        
        # Cập nhật ưu tiên
        self.update_priorities(indices, priorities)
    
    def clear(self) -> None:
        """
        Xóa tất cả dữ liệu trong bộ đệm.
        """
        if self.buffer_type == "uniform":
            self.buffer.clear()
        elif self.buffer_type == "prioritized":
            self.buffer = [None] * self.buffer_size
            self.priorities = np.zeros(self.buffer_size, dtype=np.float32)
        elif self.buffer_type == "balanced":
            self.buffer.clear()
            self.action_counts.clear()
        
        self.current_size = 0
        self.next_idx = 0
    
    def get_size(self) -> int:
        """
        Lấy kích thước hiện tại của bộ đệm.
        
        Returns:
            Số lượng trải nghiệm hiện có trong bộ đệm
        """
        return self.current_size
    
    def is_full(self) -> bool:
        """
        Kiểm tra xem bộ đệm đã đầy chưa.
        
        Returns:
            True nếu bộ đệm đã đầy, False nếu chưa
        """
        return self.current_size >= self.buffer_size
    
    def get_all_transitions(self) -> List[Experience]:
        """
        Lấy tất cả các trải nghiệm trong bộ đệm.
        
        Returns:
            Danh sách tất cả trải nghiệm
        """
        if self.buffer_type == "uniform":
            return list(self.buffer)
        elif self.buffer_type == "prioritized":
            return [exp for exp in self.buffer[:self.current_size] if exp is not None]
        elif self.buffer_type == "balanced":
            return [exp for sublist in self.buffer.values() for exp in sublist]
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Lấy thông tin thống kê về bộ đệm.
        
        Returns:
            Dict chứa thông tin thống kê
        """
        stats = {
            "buffer_type": self.buffer_type,
            "buffer_size": self.buffer_size,
            "current_size": self.current_size,
            "buffer_usage": self.current_size / self.buffer_size if self.buffer_size > 0 else 0
        }
        
        if self.buffer_type == "prioritized":
            if self.current_size > 0:
                valid_priorities = self.priorities[:self.current_size]
                stats.update({
                    "min_priority": float(np.min(valid_priorities)),
                    "max_priority": float(np.max(valid_priorities)),
                    "mean_priority": float(np.mean(valid_priorities)),
                    "std_priority": float(np.std(valid_priorities)),
                    "beta": self.beta
                })
        
        elif self.buffer_type == "balanced":
            stats["action_counts"] = {str(key): count for key, count in self.action_counts.items()}
            stats["action_types"] = len(self.buffer)
        
        return stats
    
    def save_to_disk(self, filepath: str) -> bool:
        """
        Lưu bộ đệm vào ổ đĩa.
        
        Args:
            filepath: Đường dẫn file để lưu
            
        Returns:
            True nếu lưu thành công, False nếu không
        """
        try:
            # Tạo dict để lưu
            save_data = {
                "buffer_type": self.buffer_type,
                "buffer_size": self.buffer_size,
                "current_size": self.current_size,
                "next_idx": self.next_idx,
                "alpha": self.alpha,
                "beta": self.beta,
                "epsilon": self.epsilon
            }
            
            if self.buffer_type == "uniform":
                # Chuyển đổi trải nghiệm thành list of dicts
                experiences = []
                for exp in self.buffer:
                    experiences.append({
                        "state": exp.state,
                        "action": exp.action,
                        "reward": exp.reward,
                        "next_state": exp.next_state,
                        "done": exp.done
                    })
                save_data["experiences"] = experiences
                
            elif self.buffer_type == "prioritized":
                # Chuyển đổi trải nghiệm thành list of dicts
                experiences = []
                for i in range(self.current_size):
                    if self.buffer[i] is not None:
                        exp = self.buffer[i]
                        experiences.append({
                            "state": exp.state,
                            "action": exp.action,
                            "reward": exp.reward,
                            "next_state": exp.next_state,
                            "done": exp.done,
                            "priority": self.priorities[i]
                        })
                save_data["experiences"] = experiences
                
            elif self.buffer_type == "balanced":
                # Chuyển đổi trải nghiệm thành dict of lists
                experiences = {}
                for action_key, exp_list in self.buffer.items():
                    action_str = str(action_key)
                    experiences[action_str] = []
                    for exp in exp_list:
                        experiences[action_str].append({
                            "state": exp.state,
                            "action": exp.action,
                            "reward": exp.reward,
                            "next_state": exp.next_state,
                            "done": exp.done
                        })
                save_data["experiences"] = experiences
                save_data["action_counts"] = {str(k): v for k, v in self.action_counts.items()}
            
            # Lưu vào file
            np.save(filepath, save_data, allow_pickle=True)
            self.logger.info(f"Đã lưu bộ đệm vào {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu bộ đệm: {str(e)}")
            return False
    
    def load_from_disk(self, filepath: str) -> bool:
        """
        Tải bộ đệm từ ổ đĩa.
        
        Args:
            filepath: Đường dẫn file để tải
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            save_data = np.load(filepath, allow_pickle=True).item()
            
            # Kiểm tra phiên bản và kiểu buffer
            if save_data["buffer_type"] != self.buffer_type:
                self.logger.warning(
                    f"Kiểu buffer không khớp: {save_data['buffer_type']} vs {self.buffer_type}"
                )
                return False
            
            # Cập nhật các thuộc tính
            self.buffer_size = save_data["buffer_size"]
            self.current_size = save_data["current_size"]
            self.next_idx = save_data["next_idx"]
            self.alpha = save_data["alpha"]
            self.beta = save_data["beta"]
            self.epsilon = save_data["epsilon"]
            
            # Khởi tạo lại buffer
            if self.buffer_type == "uniform":
                self.buffer = deque(maxlen=self.buffer_size)
                # Tải lại các trải nghiệm
                for exp_dict in save_data["experiences"]:
                    exp = Experience(
                        exp_dict["state"],
                        exp_dict["action"],
                        exp_dict["reward"],
                        exp_dict["next_state"],
                        exp_dict["done"]
                    )
                    self.buffer.append(exp)
                
            elif self.buffer_type == "prioritized":
                self.buffer = [None] * self.buffer_size
                self.priorities = np.zeros(self.buffer_size, dtype=np.float32)
                
                # Tải lại các trải nghiệm
                for i, exp_dict in enumerate(save_data["experiences"]):
                    if i < self.buffer_size:
                        exp = Experience(
                            exp_dict["state"],
                            exp_dict["action"],
                            exp_dict["reward"],
                            exp_dict["next_state"],
                            exp_dict["done"]
                        )
                        self.buffer[i] = exp
                        self.priorities[i] = exp_dict["priority"]
                
            elif self.buffer_type == "balanced":
                self.buffer = {}
                self.action_counts = {}
                
                # Tải lại các trải nghiệm
                for action_str, exp_list in save_data["experiences"].items():
                    # Chuyển đổi action string về dạng ban đầu (int hoặc tuple)
                    try:
                        # Thử chuyển thành int
                        action_key = int(action_str)
                    except ValueError:
                        # Nếu không phải int, có thể là tuple
                        if action_str.startswith('(') and action_str.endswith(')'):
                            # Chuyển thành tuple of floats
                            action_key = tuple(float(x.strip()) for x in action_str[1:-1].split(','))
                        else:
                            # Giữ nguyên string
                            action_key = action_str
                    
                    # Khởi tạo buffer cho action này
                    self.buffer[action_key] = deque(maxlen=self.buffer_size // 10)
                    
                    # Tải các trải nghiệm
                    for exp_dict in exp_list:
                        exp = Experience(
                            exp_dict["state"],
                            exp_dict["action"],
                            exp_dict["reward"],
                            exp_dict["next_state"],
                            exp_dict["done"]
                        )
                        self.buffer[action_key].append(exp)
                
                # Tải lại action_counts
                self.action_counts = {
                    eval(k) if k.startswith('(') else (int(k) if k.isdigit() else k): v
                    for k, v in save_data["action_counts"].items()
                }
            
            self.logger.info(f"Đã tải bộ đệm từ {filepath} với {self.current_size} trải nghiệm")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải bộ đệm: {str(e)}")
            return False


class MultiAgentExperienceBuffer:
    """
    Bộ đệm kinh nghiệm cho multi-agent.
    Quản lý nhiều bộ đệm riêng biệt cho từng agent.
    """
    
    def __init__(
        self,
        num_agents: int,
        buffer_size: int = 10000,
        state_dim: Optional[Union[int, Tuple[int, ...]]] = None,
        action_dim: Optional[Union[int, Tuple[int, ...]]] = None,
        buffer_type: str = "uniform",
        shared_buffer: bool = False,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo bộ đệm kinh nghiệm đa agent.
        
        Args:
            num_agents: Số lượng agents
            buffer_size: Kích thước tối đa của bộ đệm
            state_dim: Kích thước không gian trạng thái
            action_dim: Kích thước không gian hành động
            buffer_type: Loại bộ đệm ('uniform', 'prioritized', 'balanced')
            shared_buffer: Sử dụng chung một bộ đệm cho tất cả agents
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("multi_agent_experience_buffer")
        
        # Thiết lập các tham số
        self.num_agents = num_agents
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_type = buffer_type
        self.shared_buffer = shared_buffer
        
        # Khởi tạo các bộ đệm
        if shared_buffer:
            # Sử dụng chung một bộ đệm
            self.buffers = {
                'shared': ExperienceBuffer(
                    buffer_size=buffer_size,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    buffer_type=buffer_type,
                    logger=logger,
                    **kwargs
                )
            }
        else:
            # Tạo bộ đệm riêng cho từng agent
            self.buffers = {
                agent_id: ExperienceBuffer(
                    buffer_size=buffer_size,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    buffer_type=buffer_type,
                    logger=logger,
                    **kwargs
                )
                for agent_id in range(num_agents)
            }
        
        self.logger.info(
            f"Đã khởi tạo {self.__class__.__name__} với {num_agents} agents, "
            f"shared_buffer={shared_buffer}, buffer_type={buffer_type}"
        )
    
    def add(
        self, 
        agent_id: int,
        state: np.ndarray, 
        action: Union[int, np.ndarray], 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        priority: Optional[float] = None
    ) -> None:
        """
        Thêm trải nghiệm vào bộ đệm của agent cụ thể.
        
        Args:
            agent_id: ID của agent
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái kế tiếp
            done: Đã kết thúc episode hay chưa
            priority: Độ ưu tiên (cho prioritized replay)
        """
        if self.shared_buffer:
            buffer = self.buffers['shared']
        else:
            if agent_id not in self.buffers:
                self.logger.warning(f"Agent ID {agent_id} không hợp lệ, sử dụng 0")
                agent_id = 0
            buffer = self.buffers[agent_id]
        
        buffer.add(state, action, reward, next_state, done, priority)
    
    def sample(self, agent_id: int, batch_size: int) -> Tuple[List[Experience], Optional[List[int]], Optional[np.ndarray]]:
        """
        Lấy mẫu ngẫu nhiên từ bộ đệm của agent cụ thể.
        
        Args:
            agent_id: ID của agent
            batch_size: Kích thước batch cần lấy
            
        Returns:
            Tuple (samples, indices, weights)
        """
        if self.shared_buffer:
            buffer = self.buffers['shared']
        else:
            if agent_id not in self.buffers:
                self.logger.warning(f"Agent ID {agent_id} không hợp lệ, sử dụng 0")
                agent_id = 0
            buffer = self.buffers[agent_id]
        
        return buffer.sample(batch_size)
    
    def get_batch(self, agent_id: int, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Lấy batch dữ liệu từ bộ đệm của agent cụ thể.
        
        Args:
            agent_id: ID của agent
            batch_size: Kích thước batch cần lấy
            
        Returns:
            Dict chứa batch dữ liệu
        """
        if self.shared_buffer:
            buffer = self.buffers['shared']
        else:
            if agent_id not in self.buffers:
                self.logger.warning(f"Agent ID {agent_id} không hợp lệ, sử dụng 0")
                agent_id = 0
            buffer = self.buffers[agent_id]
        
        return buffer.get_batch(batch_size)
    
    def update_priorities(self, agent_id: int, indices: List[int], priorities: List[float]) -> None:
        """
        Cập nhật độ ưu tiên cho bộ đệm của agent cụ thể.
        
        Args:
            agent_id: ID của agent
            indices: Danh sách chỉ số
            priorities: Danh sách độ ưu tiên
        """
        if self.shared_buffer:
            buffer = self.buffers['shared']
        else:
            if agent_id not in self.buffers:
                return
            buffer = self.buffers[agent_id]
        
        buffer.update_priorities(indices, priorities)
    
    def update_batch(self, agent_id: int, indices: List[int], td_errors: List[float]) -> None:
        """
        Cập nhật độ ưu tiên cho một batch của agent cụ thể.
        
        Args:
            agent_id: ID của agent
            indices: Chỉ số của các mẫu
            td_errors: TD errors tương ứng
        """
        if self.shared_buffer:
            buffer = self.buffers['shared']
        else:
            if agent_id not in self.buffers:
                return
            buffer = self.buffers[agent_id]
        
        buffer.update_batch(indices, td_errors)
    
    def get_size(self, agent_id: Optional[int] = None) -> Union[int, Dict[int, int]]:
        """
        Lấy kích thước hiện tại của bộ đệm.
        
        Args:
            agent_id: ID của agent (None để lấy của tất cả)
            
        Returns:
            Kích thước hoặc dict kích thước của từng agent
        """
        if agent_id is not None:
            if self.shared_buffer:
                return self.buffers['shared'].get_size()
            else:
                if agent_id not in self.buffers:
                    self.logger.warning(f"Agent ID {agent_id} không hợp lệ, sử dụng 0")
                    agent_id = 0
                return self.buffers[agent_id].get_size()
        else:
            if self.shared_buffer:
                return {'shared': self.buffers['shared'].get_size()}
            else:
                return {agent_id: buffer.get_size() for agent_id, buffer in self.buffers.items()}
    
    def clear(self, agent_id: Optional[int] = None) -> None:
        """
        Xóa bộ đệm của agent cụ thể hoặc tất cả.
        
        Args:
            agent_id: ID của agent (None để xóa tất cả)
        """
        if agent_id is not None:
            if self.shared_buffer:
                self.buffers['shared'].clear()
            else:
                if agent_id in self.buffers:
                    self.buffers[agent_id].clear()
        else:
            for buffer in self.buffers.values():
                buffer.clear()
    
    def save_to_disk(self, filepath: str) -> bool:
        """
        Lưu tất cả bộ đệm vào ổ đĩa.
        
        Args:
            filepath: Đường dẫn file để lưu
            
        Returns:
            True nếu lưu thành công, False nếu không
        """
        try:
            # Tạo thư mục nếu chưa tồn tại
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Lưu từng buffer vào file riêng
            for agent_id, buffer in self.buffers.items():
                agent_filepath = f"{filepath}_{agent_id}"
                buffer.save_to_disk(agent_filepath)
            
            # Lưu thông tin tổng quát
            metadata = {
                "num_agents": self.num_agents,
                "buffer_size": self.buffer_size,
                "buffer_type": self.buffer_type,
                "shared_buffer": self.shared_buffer,
                "agent_ids": list(self.buffers.keys())
            }
            np.save(f"{filepath}_metadata", metadata, allow_pickle=True)
            
            self.logger.info(f"Đã lưu tất cả bộ đệm vào {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu bộ đệm: {str(e)}")
            return False
    
    def load_from_disk(self, filepath: str) -> bool:
        """
        Tải tất cả bộ đệm từ ổ đĩa.
        
        Args:
            filepath: Đường dẫn file để tải
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            # Tải metadata
            metadata = np.load(f"{filepath}_metadata", allow_pickle=True).item()
            
            # Kiểm tra tính tương thích
            if metadata["buffer_type"] != self.buffer_type or metadata["shared_buffer"] != self.shared_buffer:
                self.logger.warning("Kiểu buffer hoặc chế độ shared không khớp")
                return False
            
            # Tải từng buffer
            success = True
            for agent_id in metadata["agent_ids"]:
                agent_filepath = f"{filepath}_{agent_id}"
                if agent_id in self.buffers:
                    buffer_success = self.buffers[agent_id].load_from_disk(agent_filepath)
                    success = success and buffer_success
                else:
                    self.logger.warning(f"Agent ID {agent_id} không tồn tại trong buffer hiện tại")
                    success = False
            
            if success:
                self.logger.info(f"Đã tải tất cả bộ đệm từ {filepath}")
            else:
                self.logger.warning(f"Có lỗi khi tải một số bộ đệm từ {filepath}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải bộ đệm: {str(e)}")
            return False