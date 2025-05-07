"""
Quản lý kinh nghiệm cho tái huấn luyện agent.
File này định nghĩa lớp ExperienceManager để quản lý, lưu trữ, và tận dụng kinh nghiệm
từ các quá trình huấn luyện và triển khai trước đó để cải thiện quá trình tái huấn luyện.
"""

import os
import sys
import time
import json
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Deque
from collections import deque, defaultdict
import random
import shutil
import h5py
import zlib
import hashlib
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config, MODEL_DIR

class SumTree:
    """
    Cấu trúc dữ liệu SumTree cho Prioritized Experience Replay.
    Cho phép lấy mẫu theo xác suất tỷ lệ với ưu tiên.
    """
    
    def __init__(self, capacity: int):
        """
        Khởi tạo SumTree với dung lượng nhất định.
        
        Args:
            capacity: Dung lượng tối đa của SumTree
        """
        self.capacity = capacity  # Số lá của cây
        self.tree = np.zeros(2 * capacity - 1)  # Cây nhị phân đầy đủ
        self.data = np.zeros(capacity, dtype=object)  # Dữ liệu kinh nghiệm
        self.n_entries = 0  # Số phần tử hiện tại
        self.write = 0  # Vị trí ghi hiện tại
    
    def _propagate(self, idx: int, change: float) -> None:
        """
        Lan truyền thay đổi giá trị lên nút cha.
        
        Args:
            idx: Chỉ số nút
            change: Lượng thay đổi
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """
        Tìm vị trí lá dựa trên giá trị s.
        
        Args:
            idx: Chỉ số nút bắt đầu
            s: Giá trị cần tìm
            
        Returns:
            Chỉ số lá
        """
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """
        Lấy tổng ưu tiên.
        
        Returns:
            Tổng ưu tiên
        """
        return self.tree[0]
    
    def add(self, p: float, data: Any) -> None:
        """
        Thêm dữ liệu mới với ưu tiên p.
        
        Args:
            p: Ưu tiên
            data: Dữ liệu kinh nghiệm
        """
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, p: float) -> None:
        """
        Cập nhật ưu tiên.
        
        Args:
            idx: Chỉ số nút
            p: Ưu tiên mới
        """
        change = p - self.tree[idx]
        
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Lấy phần tử theo ưu tiên s.
        
        Args:
            s: Giá trị ưu tiên (0 <= s <= total)
            
        Returns:
            Tuple (idx, priority, data)
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        return (idx, self.tree[idx], self.data[dataIdx])

class ExperienceManager:
    """
    Quản lý kinh nghiệm cho tái huấn luyện agent.
    
    Lớp này chịu trách nhiệm:
    1. Lưu trữ và quản lý kinh nghiệm (state, action, reward, next_state, done)
    2. Hỗ trợ transfer learning giữa các phiên bản model
    3. Lọc, ưu tiên và lấy mẫu kinh nghiệm hiệu quả
    4. Lưu và tải kinh nghiệm từ ổ đĩa
    """
    
    def __init__(
        self,
        agent_type: str,
        model_version: str,
        memory_size: int = 100000,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_increment: float = 0.001,
        per_epsilon: float = 0.01,
        use_prioritized: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        preserve_experience: bool = True,
        memory_compression: bool = True,
        import_legacy_experiences: bool = True,
        max_experiences_per_version: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo ExperienceManager.
        
        Args:
            agent_type: Loại agent ("dqn", "ppo", "a2c", v.v.)
            model_version: Phiên bản hiện tại của model
            memory_size: Kích thước bộ nhớ kinh nghiệm
            per_alpha: Hệ số alpha cho prioritized replay (độ ưu tiên)
            per_beta: Hệ số beta cho prioritized replay (trọng số IS)
            per_beta_increment: Tốc độ tăng beta
            per_epsilon: Giá trị epsilon nhỏ để tránh ưu tiên bằng 0
            use_prioritized: Sử dụng prioritized experience replay
            output_dir: Thư mục lưu trữ kinh nghiệm
            preserve_experience: Giữ lại kinh nghiệm giữa các phiên bản
            memory_compression: Nén dữ liệu kinh nghiệm khi lưu
            import_legacy_experiences: Nhập kinh nghiệm từ phiên bản cũ
            max_experiences_per_version: Số file kinh nghiệm tối đa cho mỗi phiên bản
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("experience_manager")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Lưu thông tin cơ bản
        self.agent_type = agent_type
        self.model_version = model_version
        self.memory_size = memory_size
        self.use_prioritized = use_prioritized
        self.preserve_experience = preserve_experience
        self.memory_compression = memory_compression
        self.import_legacy_experiences = import_legacy_experiences
        self.max_experiences_per_version = max_experiences_per_version
        
        # Tham số cho prioritized experience replay
        self.per_alpha = per_alpha  # Độ ưu tiên (α) quyết định mức độ ưu tiên dựa trên TD error
        self.per_beta = per_beta    # Trọng số importance-sampling (β) bù trừ cho độ thiên lệch
        self.per_beta_increment = per_beta_increment  # Tốc độ tăng beta theo thời gian
        self.per_epsilon = per_epsilon  # Epsilon nhỏ để đảm bảo tất cả kinh nghiệm đều có cơ hội bị lấy mẫu
        
        # Thiết lập thư mục lưu trữ
        if output_dir is None:
            output_dir = MODEL_DIR / 'retraining' / 'experiences'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo bộ nhớ kinh nghiệm
        if self.use_prioritized:
            # Prioritized experience replay
            self.memory = SumTree(memory_size)
            self.experience_count = 0
        else:
            # Uniform experience replay
            self.memory = deque(maxlen=memory_size)
        
        # Biến theo dõi
        self.add_count = 0
        self.sample_count = 0
        self.last_saved_time = 0
        self.last_loaded_time = 0
        self.imported_experiences = set()
        
        # Thống kê
        self.stats = {
            "total_added": 0,
            "total_sampled": 0,
            "positive_rewards": 0,
            "negative_rewards": 0,
            "zero_rewards": 0,
            "terminal_states": 0,
            "high_priority_samples": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Tạo mã hash cho model version để sử dụng trong tên file
        self.version_hash = hashlib.md5(f"{agent_type}_{model_version}".encode()).hexdigest()[:8]
        
        # Nhập kinh nghiệm từ các phiên bản cũ nếu được yêu cầu
        if self.import_legacy_experiences:
            self._import_legacy_experiences()
        
        self.logger.info(
            f"Đã khởi tạo ExperienceManager cho {self.agent_type} "
            f"phiên bản {self.model_version} với memory_size={self.memory_size}, "
            f"use_prioritized={self.use_prioritized}"
        )
    
    def add(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        additional_info: Optional[Dict[str, Any]] = None,
        priority: Optional[float] = None
    ) -> None:
        """
        Thêm trải nghiệm mới vào bộ nhớ.
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái kế tiếp
            done: Đã kết thúc episode chưa
            additional_info: Thông tin bổ sung về trải nghiệm
            priority: Ưu tiên cho trải nghiệm (chỉ dùng với prioritized ER)
        """
        # Tạo đối tượng trải nghiệm
        experience = {
            "state": np.array(state, dtype=np.float32),
            "action": action,
            "reward": reward,
            "next_state": np.array(next_state, dtype=np.float32),
            "done": done,
            "timestamp": time.time(),
            "info": additional_info or {}
        }
        
        # Thêm vào bộ nhớ
        if self.use_prioritized:
            # Nếu không cung cấp ưu tiên, sử dụng ưu tiên cao nhất
            if priority is None:
                if self.experience_count == 0:
                    max_priority = 1.0
                else:
                    max_priority = self.memory.total() / max(1, self.experience_count)
                priority = max_priority
            
            # Chuyển đổi ưu tiên theo alpha
            adjusted_priority = (abs(priority) + self.per_epsilon) ** self.per_alpha
            
            # Thêm vào Sum Tree
            self.memory.add(adjusted_priority, experience)
            self.experience_count = min(self.memory_size, self.experience_count + 1)
        else:
            # Thêm vào deque
            self.memory.append(experience)
        
        # Cập nhật thống kê
        self.stats["total_added"] += 1
        if reward > 0:
            self.stats["positive_rewards"] += 1
        elif reward < 0:
            self.stats["negative_rewards"] += 1
        else:
            self.stats["zero_rewards"] += 1
        
        if done:
            self.stats["terminal_states"] += 1
        
        self.add_count += 1
        self.stats["last_updated"] = datetime.now().isoformat()
        
        # Tự động lưu định kỳ (sau mỗi 1000 kinh nghiệm mới)
        if self.add_count % 1000 == 0:
            self.save()
    
    def add_batch(
        self,
        batch: List[Dict[str, Any]],
        priorities: Optional[List[float]] = None
    ) -> None:
        """
        Thêm một batch trải nghiệm vào bộ nhớ.
        
        Args:
            batch: Danh sách các trải nghiệm
            priorities: Danh sách ưu tiên tương ứng
        """
        for i, experience in enumerate(batch):
            priority = None
            if priorities is not None and i < len(priorities):
                priority = priorities[i]
            
            self.add(
                state=experience.get("state"),
                action=experience.get("action"),
                reward=experience.get("reward"),
                next_state=experience.get("next_state"),
                done=experience.get("done"),
                additional_info=experience.get("info"),
                priority=priority
            )
    
    def sample(
        self,
        batch_size: int,
        return_indices_and_weights: bool = False
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], List[int], np.ndarray]]:
        """
        Lấy mẫu ngẫu nhiên từ bộ nhớ kinh nghiệm.
        
        Args:
            batch_size: Kích thước batch cần lấy mẫu
            return_indices_and_weights: Có trả về indices và trọng số sampling không
            
        Returns:
            List các trải nghiệm hoặc Tuple (trải nghiệm, indices, weights)
        """
        # Kiểm tra số lượng mẫu có đủ không
        if self.use_prioritized:
            n_samples = min(self.experience_count, batch_size)
        else:
            n_samples = min(len(self.memory), batch_size)
        
        if n_samples == 0:
            if return_indices_and_weights:
                return [], [], np.array([])
            return []
        
        # Lấy mẫu
        if self.use_prioritized:
            # Lấy mẫu từ Sum Tree theo ưu tiên
            batch = []
            indices = []
            weights = np.zeros(n_samples, dtype=np.float32)
            
            # Tính tổng ưu tiên
            total_priority = self.memory.total()
            
            # Segment chia đều tổng ưu tiên
            segment = total_priority / n_samples
            
            # Tăng beta theo thời gian
            beta = min(1.0, self.per_beta + self.sample_count * self.per_beta_increment)
            
            for i in range(n_samples):
                # Chọn điểm ngẫu nhiên trong segment
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b)
                
                # Lấy mẫu từ Sum Tree
                idx, priority, data = self.memory.get(s)
                
                # Tính trọng số IS
                sample_prob = priority / total_priority
                weights[i] = (n_samples * sample_prob) ** (-beta)
                
                indices.append(idx)
                batch.append(data)
            
            # Chuẩn hóa trọng số
            weights = weights / weights.max()
            
        else:
            # Lấy mẫu đồng đều từ deque
            batch = random.sample(list(self.memory), n_samples)
            indices = [0] * n_samples  # Không có ý nghĩa với uniform sampling
            weights = np.ones(n_samples, dtype=np.float32)  # Trọng số đồng đều
        
        # Cập nhật thống kê
        self.sample_count += 1
        self.stats["total_sampled"] += n_samples
        
        if return_indices_and_weights:
            return batch, indices, weights
        
        return batch
    
    def sample_recent(
        self,
        batch_size: int,
        time_window: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Lấy mẫu trải nghiệm gần đây.
        
        Args:
            batch_size: Kích thước batch cần lấy mẫu
            time_window: Cửa sổ thời gian (giây), None để không giới hạn
            
        Returns:
            List các trải nghiệm gần đây
        """
        now = time.time()
        
        if self.use_prioritized:
            # Lấy tất cả trải nghiệm trong Sum Tree
            all_experiences = []
            total = self.memory.total()
            
            if total > 0:
                for i in range(self.experience_count):
                    s = random.uniform(0, total)
                    _, _, data = self.memory.get(s)
                    all_experiences.append(data)
            
            # Lọc theo thời gian nếu cần
            if time_window is not None:
                filtered_experiences = [
                    exp for exp in all_experiences 
                    if now - exp.get("timestamp", 0) <= time_window
                ]
            else:
                filtered_experiences = all_experiences
            
            # Sắp xếp theo thời gian
            filtered_experiences.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            # Trả về batch_size mẫu gần đây nhất
            return filtered_experiences[:batch_size]
            
        else:
            # Deque đã sắp xếp theo thứ tự thêm vào
            # Lấy batch_size mẫu cuối cùng
            experiences = list(self.memory)
            
            # Lọc theo thời gian nếu cần
            if time_window is not None:
                filtered_experiences = [
                    exp for exp in experiences 
                    if now - exp.get("timestamp", 0) <= time_window
                ]
            else:
                filtered_experiences = experiences
            
            # Sắp xếp theo thời gian
            filtered_experiences.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            # Trả về batch_size mẫu gần đây nhất
            return filtered_experiences[:batch_size]
    
    def sample_successful(
        self,
        batch_size: int,
        reward_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Lấy mẫu trải nghiệm thành công (reward cao).
        
        Args:
            batch_size: Kích thước batch cần lấy mẫu
            reward_threshold: Ngưỡng phần thưởng để xem là thành công
            
        Returns:
            List các trải nghiệm thành công
        """
        # Lấy tất cả trải nghiệm
        if self.use_prioritized:
            all_experiences = []
            total = self.memory.total()
            
            if total > 0:
                for i in range(self.experience_count):
                    s = random.uniform(0, total)
                    _, _, data = self.memory.get(s)
                    all_experiences.append(data)
        else:
            all_experiences = list(self.memory)
        
        # Lọc các trải nghiệm thành công
        successful_experiences = [
            exp for exp in all_experiences 
            if exp.get("reward", 0) > reward_threshold
        ]
        
        # Nếu không đủ, thêm các trải nghiệm reward cao nhất
        if len(successful_experiences) < batch_size and all_experiences:
            # Sắp xếp theo reward giảm dần
            remaining = sorted(
                [exp for exp in all_experiences if exp not in successful_experiences],
                key=lambda x: x.get("reward", 0),
                reverse=True
            )
            
            # Thêm vào cho đủ batch_size
            successful_experiences.extend(
                remaining[:batch_size - len(successful_experiences)]
            )
        
        # Chọn ngẫu nhiên nếu có nhiều hơn batch_size
        if len(successful_experiences) > batch_size:
            return random.sample(successful_experiences, batch_size)
        
        return successful_experiences
    
    def update_priorities(
        self,
        indices: List[int],
        priorities: List[float]
    ) -> None:
        """
        Cập nhật ưu tiên cho các mẫu.
        
        Args:
            indices: Danh sách chỉ số cần cập nhật
            priorities: Danh sách ưu tiên mới
        """
        if not self.use_prioritized:
            self.logger.warning("Không thể cập nhật ưu tiên khi không sử dụng prioritized experience replay")
            return
        
        for idx, priority in zip(indices, priorities):
            # Chuyển đổi ưu tiên theo alpha
            adjusted_priority = (abs(priority) + self.per_epsilon) ** self.per_alpha
            
            # Cập nhật trong Sum Tree
            self.memory.update(idx, adjusted_priority)
            
            # Cập nhật thống kê
            if adjusted_priority > 1.0:  # Ưu tiên cao
                self.stats["high_priority_samples"] += 1
    
    def clear(self) -> None:
        """
        Xóa tất cả trải nghiệm trong bộ nhớ.
        """
        if self.use_prioritized:
            self.memory = SumTree(self.memory_size)
            self.experience_count = 0
        else:
            self.memory.clear()
        
        # Đặt lại các biến theo dõi
        self.add_count = 0
        self.sample_count = 0
        
        # Đặt lại thống kê
        self.stats = {
            "total_added": 0,
            "total_sampled": 0,
            "positive_rewards": 0,
            "negative_rewards": 0,
            "zero_rewards": 0,
            "terminal_states": 0,
            "high_priority_samples": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        self.logger.info("Đã xóa tất cả trải nghiệm trong bộ nhớ")
    
    def save(
        self,
        file_path: Optional[Union[str, Path]] = None,
        compress: Optional[bool] = None
    ) -> str:
        """
        Lưu trải nghiệm vào file.
        
        Args:
            file_path: Đường dẫn lưu file (None để sử dụng đường dẫn mặc định)
            compress: Nén dữ liệu hay không (None để sử dụng cấu hình mặc định)
            
        Returns:
            Đường dẫn đã lưu
        """
        if file_path is None:
            # Tạo tên file dựa trên agent_type, model_version và timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{self.agent_type}_{self.version_hash}_{timestamp}.exp"
            file_path = self.output_dir / file_name
        
        # Sử dụng cấu hình nén mặc định nếu không chỉ định
        if compress is None:
            compress = self.memory_compression
        
        try:
            # Chuẩn bị dữ liệu để lưu
            save_data = {
                "agent_type": self.agent_type,
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat(),
                "stats": self.stats,
                "config": {
                    "memory_size": self.memory_size,
                    "use_prioritized": self.use_prioritized,
                    "per_alpha": self.per_alpha,
                    "per_beta": self.per_beta,
                    "per_beta_increment": self.per_beta_increment,
                    "per_epsilon": self.per_epsilon
                }
            }
            
            # Lưu trải nghiệm tùy theo loại bộ nhớ
            if self.use_prioritized:
                # Với prioritized replay, lưu cả dữ liệu và ưu tiên
                experiences = []
                priorities = []
                
                # Lấy tất cả trải nghiệm từ Sum Tree
                total = self.memory.total()
                
                if total > 0:
                    for i in range(self.experience_count):
                        s = random.uniform(0, total)
                        idx, priority, data = self.memory.get(s)
                        experiences.append(data)
                        priorities.append(priority)
                
                save_data["experiences"] = experiences
                save_data["priorities"] = priorities
                save_data["experience_count"] = self.experience_count
                
            else:
                # Với uniform replay, chỉ cần lưu dữ liệu
                save_data["experiences"] = list(self.memory)
            
            # Tạo thư mục nếu không tồn tại
            if isinstance(file_path, str):
                file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Lưu dữ liệu
            if compress:
                # Sử dụng pickle và nén zlib
                with open(file_path, 'wb') as f:
                    pickled_data = pickle.dumps(save_data)
                    compressed_data = zlib.compress(pickled_data)
                    f.write(compressed_data)
                
                file_size = file_path.stat().st_size
                self.logger.info(
                    f"Đã lưu trải nghiệm vào {file_path} (nén): "
                    f"{len(save_data['experiences'])} mẫu, {file_size/1024:.2f}KB"
                )
            else:
                # Lưu thẳng bằng pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(save_data, f)
                
                file_size = file_path.stat().st_size
                self.logger.info(
                    f"Đã lưu trải nghiệm vào {file_path}: "
                    f"{len(save_data['experiences'])} mẫu, {file_size/1024:.2f}KB"
                )
            
            # Cập nhật thời gian lưu
            self.last_saved_time = time.time()
            
            # Dọn dẹp các file cũ nếu quá nhiều
            self._cleanup_old_experience_files()
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu trải nghiệm: {str(e)}")
            return ""
    
    def load(
        self,
        file_path: Union[str, Path],
        replace_current: bool = False
    ) -> bool:
        """
        Tải trải nghiệm từ file.
        
        Args:
            file_path: Đường dẫn file
            replace_current: Thay thế trải nghiệm hiện tại hay gộp vào
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            # Kiểm tra file tồn tại
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"Không tìm thấy file trải nghiệm: {file_path}")
                return False
            
            # Đọc file
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # Thử giải nén
                try:
                    decompressed_data = zlib.decompress(data)
                    loaded_data = pickle.loads(decompressed_data)
                except (zlib.error, pickle.UnpicklingError):
                    # Nếu không phải dữ liệu nén, đọc trực tiếp
                    loaded_data = pickle.loads(data)
            except Exception as e:
                self.logger.error(f"Lỗi khi đọc file trải nghiệm: {str(e)}")
                return False
            
            # Kiểm tra dữ liệu
            if "experiences" not in loaded_data:
                self.logger.error("File không chứa dữ liệu trải nghiệm hợp lệ")
                return False
            
            # Lấy trải nghiệm từ file
            file_experiences = loaded_data["experiences"]
            file_priorities = loaded_data.get("priorities", None)
            file_experience_count = loaded_data.get("experience_count", len(file_experiences))
            
            # Kiểm tra tương thích
            file_use_prioritized = loaded_data.get("config", {}).get("use_prioritized", False)
            
            if file_use_prioritized != self.use_prioritized:
                self.logger.warning(
                    f"Khác loại bộ nhớ: File sử dụng prioritized={file_use_prioritized}, "
                    f"hiện tại sử dụng prioritized={self.use_prioritized}"
                )
                # Vẫn tiếp tục nhưng không sử dụng được ưu tiên từ file
                file_priorities = None
            
            # Xóa dữ liệu hiện tại nếu cần
            if replace_current:
                if self.use_prioritized:
                    self.memory = SumTree(self.memory_size)
                    self.experience_count = 0
                else:
                    self.memory.clear()
            
            # Thêm trải nghiệm vào bộ nhớ
            if self.use_prioritized:
                # Với prioritized replay, cần cả trải nghiệm và ưu tiên
                for i, exp in enumerate(file_experiences):
                    priority = None
                    if file_priorities and i < len(file_priorities):
                        priority = file_priorities[i]
                    
                    # Sử dụng priority mặc định nếu không có
                    if priority is None:
                        priority = 1.0
                    
                    # Thêm vào Sum Tree
                    self.memory.add(priority, exp)
                
                self.experience_count = min(self.memory_size, self.experience_count + len(file_experiences))
                
            else:
                # Với uniform replay, chỉ cần thêm trải nghiệm
                for exp in file_experiences:
                    self.memory.append(exp)
            
            # Cập nhật số lượng đã thêm
            old_total = self.stats["total_added"]
            self.stats["total_added"] += len(file_experiences)
            
            # Cập nhật thống kê khác từ file nếu là thay thế
            if replace_current:
                file_stats = loaded_data.get("stats", {})
                for key, value in file_stats.items():
                    if key in self.stats:
                        self.stats[key] = value
            
            # Cập nhật thời gian tải
            self.last_loaded_time = time.time()
            self.stats["last_updated"] = datetime.now().isoformat()
            
            # Thêm vào danh sách đã nhập
            self.imported_experiences.add(str(file_path))
            
            self.logger.info(
                f"Đã tải {len(file_experiences)} trải nghiệm từ {file_path}, "
                f"tổng cộng {self.stats['total_added']} trải nghiệm"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải trải nghiệm: {str(e)}")
            return False
    
    def merge_from_agent(
        self,
        agent_type: str,
        model_version: str,
        ratio: float = 0.5,
        max_samples: Optional[int] = None
    ) -> bool:
        """
        Gộp trải nghiệm từ một agent khác.
        
        Args:
            agent_type: Loại agent nguồn
            model_version: Phiên bản model nguồn
            ratio: Tỷ lệ trải nghiệm từ agent khác (0.0-1.0)
            max_samples: Số lượng mẫu tối đa từ agent khác
            
        Returns:
            True nếu gộp thành công, False nếu không
        """
        # Tìm các file kinh nghiệm của agent nguồn
        source_version_hash = hashlib.md5(f"{agent_type}_{model_version}".encode()).hexdigest()[:8]
        source_pattern = f"{agent_type}_{source_version_hash}_*.exp"
        source_files = list(self.output_dir.glob(source_pattern))
        
        if not source_files:
            self.logger.warning(f"Không tìm thấy file trải nghiệm nào cho {agent_type} phiên bản {model_version}")
            return False
        
        # Sắp xếp theo thời gian tạo (mới nhất trước)
        source_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Chỉ lấy file gần đây nhất
        source_file = source_files[0]
        
        # Đọc file
        try:
            with open(source_file, 'rb') as f:
                data = f.read()
            
            # Thử giải nén
            try:
                decompressed_data = zlib.decompress(data)
                loaded_data = pickle.loads(decompressed_data)
            except (zlib.error, pickle.UnpicklingError):
                # Nếu không phải dữ liệu nén, đọc trực tiếp
                loaded_data = pickle.loads(data)
            
            # Kiểm tra dữ liệu
            if "experiences" not in loaded_data:
                self.logger.error("File không chứa dữ liệu trải nghiệm hợp lệ")
                return False
            
            # Lấy trải nghiệm từ file
            source_experiences = loaded_data["experiences"]
            source_priorities = loaded_data.get("priorities", None)
            
            # Tính số lượng mẫu cần lấy
            if max_samples is None:
                # Dựa vào ratio và memory_size
                max_samples = int(self.memory_size * ratio)
            
            # Lấy ngẫu nhiên nếu có nhiều hơn cần thiết
            if len(source_experiences) > max_samples:
                indices = random.sample(range(len(source_experiences)), max_samples)
                merged_experiences = [source_experiences[i] for i in indices]
                merged_priorities = None
                if source_priorities:
                    merged_priorities = [source_priorities[i] for i in indices]
            else:
                merged_experiences = source_experiences
                merged_priorities = source_priorities
            
            # Thêm vào bộ nhớ hiện tại
            self.add_batch(merged_experiences, merged_priorities)
            
            self.logger.info(
                f"Đã gộp {len(merged_experiences)} trải nghiệm từ {agent_type} "
                f"phiên bản {model_version}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi gộp trải nghiệm: {str(e)}")
            return False
    
    def filter_by_reward(
        self,
        min_reward: Optional[float] = None,
        max_reward: Optional[float] = None
    ) -> "ExperienceManager":
        """
        Lọc trải nghiệm theo phần thưởng.
        
        Args:
            min_reward: Phần thưởng tối thiểu (None để không giới hạn)
            max_reward: Phần thưởng tối đa (None để không giới hạn)
            
        Returns:
            ExperienceManager mới chỉ chứa trải nghiệm thỏa mãn
        """
        # Tạo ExperienceManager mới
        filtered_manager = ExperienceManager(
            agent_type=self.agent_type,
            model_version=f"{self.model_version}_filtered",
            memory_size=self.memory_size,
            use_prioritized=self.use_prioritized,
            per_alpha=self.per_alpha,
            per_beta=self.per_beta,
            output_dir=self.output_dir,
            logger=self.logger
        )
        
        # Lấy tất cả trải nghiệm hiện tại
        if self.use_prioritized:
            experiences = []
            priorities = []
            
            total = self.memory.total()
            if total > 0:
                for i in range(self.experience_count):
                    s = random.uniform(0, total)
                    idx, priority, data = self.memory.get(s)
                    experiences.append(data)
                    priorities.append(priority)
        else:
            experiences = list(self.memory)
            priorities = None
        
        # Lọc theo điều kiện
        filtered_experiences = []
        filtered_priorities = [] if priorities else None
        
        for i, exp in enumerate(experiences):
            reward = exp.get("reward", 0)
            
            # Kiểm tra điều kiện
            if (min_reward is None or reward >= min_reward) and (max_reward is None or reward <= max_reward):
                filtered_experiences.append(exp)
                if filtered_priorities:
                    filtered_priorities.append(priorities[i])
        
        # Thêm vào ExperienceManager mới
        filtered_manager.add_batch(filtered_experiences, filtered_priorities)
        
        self.logger.info(
            f"Đã lọc {len(filtered_experiences)}/{len(experiences)} trải nghiệm "
            f"theo điều kiện reward: {min_reward} <= r <= {max_reward}"
        )
        
        return filtered_manager
    
    def filter_by_time(
        self,
        max_age: Optional[float] = None,
        min_age: Optional[float] = None
    ) -> "ExperienceManager":
        """
        Lọc trải nghiệm theo thời gian.
        
        Args:
            max_age: Tuổi tối đa (giây) (None để không giới hạn)
            min_age: Tuổi tối thiểu (giây) (None để không giới hạn)
            
        Returns:
            ExperienceManager mới chỉ chứa trải nghiệm thỏa mãn
        """
        now = time.time()
        
        # Tạo ExperienceManager mới
        filtered_manager = ExperienceManager(
            agent_type=self.agent_type,
            model_version=f"{self.model_version}_filtered",
            memory_size=self.memory_size,
            use_prioritized=self.use_prioritized,
            per_alpha=self.per_alpha,
            per_beta=self.per_beta,
            output_dir=self.output_dir,
            logger=self.logger
        )
        
        # Lấy tất cả trải nghiệm hiện tại
        if self.use_prioritized:
            experiences = []
            priorities = []
            
            total = self.memory.total()
            if total > 0:
                for i in range(self.experience_count):
                    s = random.uniform(0, total)
                    idx, priority, data = self.memory.get(s)
                    experiences.append(data)
                    priorities.append(priority)
        else:
            experiences = list(self.memory)
            priorities = None
        
        # Lọc theo điều kiện
        filtered_experiences = []
        filtered_priorities = [] if priorities else None
        
        for i, exp in enumerate(experiences):
            timestamp = exp.get("timestamp", 0)
            age = now - timestamp
            
            # Kiểm tra điều kiện
            if (max_age is None or age <= max_age) and (min_age is None or age >= min_age):
                filtered_experiences.append(exp)
                if filtered_priorities:
                    filtered_priorities.append(priorities[i])
        
        # Thêm vào ExperienceManager mới
        filtered_manager.add_batch(filtered_experiences, filtered_priorities)
        
        self.logger.info(
            f"Đã lọc {len(filtered_experiences)}/{len(experiences)} trải nghiệm "
            f"theo điều kiện thời gian: {min_age}s <= age <= {max_age}s"
        )
        
        return filtered_manager
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về bộ nhớ trải nghiệm.
        
        Returns:
            Dict chứa các thống kê
        """
        # Cập nhật các thống kê cơ bản
        current_stats = self.stats.copy()
        
        # Thêm thông tin hiện tại
        if self.use_prioritized:
            current_size = self.experience_count
        else:
            current_size = len(self.memory)
        
        current_stats.update({
            "memory_type": "prioritized" if self.use_prioritized else "uniform",
            "current_size": current_size,
            "memory_capacity": self.memory_size,
            "utilization": current_size / self.memory_size if self.memory_size > 0 else 0.0,
            "last_saved_time": self.last_saved_time,
            "last_loaded_time": self.last_loaded_time,
            "memory_stats_timestamp": datetime.now().isoformat(),
            "imported_experiences_count": len(self.imported_experiences)
        })
        
        return current_stats
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Xuất bộ nhớ trải nghiệm sang DataFrame.
        
        Returns:
            DataFrame chứa dữ liệu trải nghiệm
        """
        # Lấy tất cả trải nghiệm
        if self.use_prioritized:
            experiences = []
            priorities = []
            
            total = self.memory.total()
            if total > 0:
                for i in range(self.experience_count):
                    s = random.uniform(0, total)
                    idx, priority, data = self.memory.get(s)
                    experiences.append(data)
                    priorities.append(priority)
        else:
            experiences = list(self.memory)
            priorities = None
        
        # Tạo cấu trúc dữ liệu cơ bản
        data = {
            "timestamp": [],
            "reward": [],
            "done": []
        }
        
        # Thêm state và next_state nếu có
        state_sizes = set()
        next_state_sizes = set()
        
        for exp in experiences:
            state = exp.get("state")
            next_state = exp.get("next_state")
            
            if state is not None:
                if hasattr(state, "shape"):
                    state_sizes.add(len(state))
            
            if next_state is not None:
                if hasattr(next_state, "shape"):
                    next_state_sizes.add(len(next_state))
        
        # Tạo cột cho state và next_state
        if state_sizes:
            max_state_size = max(state_sizes)
            for i in range(max_state_size):
                data[f"state_{i}"] = []
        
        if next_state_sizes:
            max_next_state_size = max(next_state_sizes)
            for i in range(max_next_state_size):
                data[f"next_state_{i}"] = []
        
        # Thêm action
        data["action"] = []
        
        # Thêm priority nếu có
        if priorities:
            data["priority"] = []
        
        # Điền dữ liệu
        for i, exp in enumerate(experiences):
            # Trích xuất các trường cơ bản
            data["timestamp"].append(exp.get("timestamp", 0))
            data["reward"].append(exp.get("reward", 0))
            data["done"].append(exp.get("done", False))
            data["action"].append(exp.get("action", 0))
            
            # Trích xuất state
            state = exp.get("state")
            if state is not None and hasattr(state, "__iter__"):
                for j, val in enumerate(state):
                    if j < max_state_size:
                        data[f"state_{j}"].append(val)
            else:
                for j in range(max_state_size):
                    data[f"state_{j}"].append(None)
            
            # Trích xuất next_state
            next_state = exp.get("next_state")
            if next_state is not None and hasattr(next_state, "__iter__"):
                for j, val in enumerate(next_state):
                    if j < max_next_state_size:
                        data[f"next_state_{j}"].append(val)
            else:
                for j in range(max_next_state_size):
                    data[f"next_state_{j}"].append(None)
            
            # Thêm priority nếu có
            if priorities and i < len(priorities):
                data["priority"].append(priorities[i])
        
        # Tạo DataFrame
        df = pd.DataFrame(data)
        
        # Thêm thông tin metadata
        df.attrs["agent_type"] = self.agent_type
        df.attrs["model_version"] = self.model_version
        df.attrs["memory_type"] = "prioritized" if self.use_prioritized else "uniform"
        df.attrs["export_time"] = datetime.now().isoformat()
        
        return df
    
    def import_from_dataframe(
        self,
        df: pd.DataFrame,
        replace_current: bool = False
    ) -> bool:
        """
        Nhập trải nghiệm từ DataFrame.
        
        Args:
            df: DataFrame chứa dữ liệu trải nghiệm
            replace_current: Thay thế trải nghiệm hiện tại hay gộp vào
            
        Returns:
            True nếu nhập thành công, False nếu không
        """
        try:
            # Xóa dữ liệu hiện tại nếu cần
            if replace_current:
                if self.use_prioritized:
                    self.memory = SumTree(self.memory_size)
                    self.experience_count = 0
                else:
                    self.memory.clear()
            
            # Xác định các cột state và next_state
            state_columns = [col for col in df.columns if col.startswith("state_")]
            next_state_columns = [col for col in df.columns if col.startswith("next_state_")]
            
            # Sắp xếp theo thứ tự số
            state_columns.sort(key=lambda x: int(x.split("_")[1]))
            next_state_columns.sort(key=lambda x: int(x.split("_")[1]))
            
            # Biến đếm cho số lượng đã nhập
            imported_count = 0
            
            # Thêm từng trải nghiệm
            for _, row in df.iterrows():
                # Tạo state và next_state
                state = np.array([row[col] for col in state_columns], dtype=np.float32)
                next_state = np.array([row[col] for col in next_state_columns], dtype=np.float32)
                
                # Lấy các trường còn lại
                action = row.get("action", 0)
                reward = row.get("reward", 0)
                done = row.get("done", False)
                timestamp = row.get("timestamp", time.time())
                
                # Tạo thông tin bổ sung
                additional_info = {
                    "from_dataframe": True,
                    "imported_at": datetime.now().isoformat()
                }
                
                # Lấy priority nếu có
                priority = row.get("priority", None)
                
                # Tạo trải nghiệm
                experience = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                    "timestamp": timestamp,
                    "info": additional_info
                }
                
                # Thêm vào bộ nhớ
                if self.use_prioritized:
                    # Nếu không cung cấp ưu tiên, sử dụng ưu tiên cao nhất
                    if priority is None:
                        if self.experience_count == 0:
                            max_priority = 1.0
                        else:
                            max_priority = self.memory.total() / max(1, self.experience_count)
                        priority = max_priority
                    
                    # Chuyển đổi ưu tiên theo alpha
                    adjusted_priority = (abs(priority) + self.per_epsilon) ** self.per_alpha
                    
                    # Thêm vào Sum Tree
                    self.memory.add(adjusted_priority, experience)
                    self.experience_count = min(self.memory_size, self.experience_count + 1)
                else:
                    # Thêm vào deque
                    self.memory.append(experience)
                
                imported_count += 1
            
            # Cập nhật thống kê
            self.stats["total_added"] += imported_count
            self.stats["last_updated"] = datetime.now().isoformat()
            
            self.logger.info(f"Đã nhập {imported_count} trải nghiệm từ DataFrame")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi nhập trải nghiệm từ DataFrame: {str(e)}")
            return False
    
    def _import_legacy_experiences(self) -> int:
        """
        Nhập kinh nghiệm từ các phiên bản model cũ.
        
        Returns:
            Số lượng file kinh nghiệm đã nhập
        """
        if not self.preserve_experience:
            return 0
        
        # Tìm các file kinh nghiệm trong thư mục
        experience_files = list(self.output_dir.glob("*.exp"))
        
        if not experience_files:
            self.logger.info("Không tìm thấy file kinh nghiệm nào để nhập")
            return 0
        
        # Sắp xếp theo thời gian tạo (mới nhất trước)
        experience_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Kiểm tra nếu có file của phiên bản hiện tại
        current_version_files = [
            f for f in experience_files 
            if f"{self.agent_type}_{self.version_hash}" in f.name
        ]
        
        # Nhập file từ phiên bản hiện tại nếu có
        if current_version_files:
            self.logger.info(f"Tìm thấy {len(current_version_files)} file kinh nghiệm cho phiên bản hiện tại")
            self.load(current_version_files[0])
            
            # Chỉ nhập file đầu tiên (mới nhất)
            imported_count = 1
            self.imported_experiences.add(str(current_version_files[0]))
            
            return imported_count
        
        # Không tìm thấy file của phiên bản hiện tại, tìm từ các phiên bản trước
        legacy_files = [
            f for f in experience_files 
            if f"{self.agent_type}_" in f.name and f not in self.imported_experiences
        ]
        
        if not legacy_files:
            self.logger.info("Không tìm thấy file kinh nghiệm trước đó phù hợp")
            return 0
        
        # Giới hạn số lượng file cần nhập
        files_to_import = legacy_files[:self.max_experiences_per_version]
        
        # Nhập từ các file
        imported_count = 0
        
        for file_path in files_to_import:
            if self.load(file_path, replace_current=False):
                imported_count += 1
                self.imported_experiences.add(str(file_path))
        
        self.logger.info(f"Đã nhập {imported_count}/{len(files_to_import)} file kinh nghiệm từ các phiên bản trước")
        
        return imported_count
    
    def _cleanup_old_experience_files(self) -> int:
        """
        Dọn dẹp các file kinh nghiệm cũ.
        
        Returns:
            Số lượng file đã xóa
        """
        # Tìm các file kinh nghiệm của phiên bản hiện tại
        current_pattern = f"{self.agent_type}_{self.version_hash}_*.exp"
        current_files = list(self.output_dir.glob(current_pattern))
        
        # Nếu số lượng file ít hơn giới hạn, không cần dọn dẹp
        if len(current_files) <= self.max_experiences_per_version:
            return 0
        
        # Sắp xếp theo thời gian tạo (cũ nhất trước)
        current_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Xác định số lượng file cần xóa
        files_to_delete = current_files[:len(current_files) - self.max_experiences_per_version]
        
        # Xóa các file
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                deleted_count += 1
                self.logger.debug(f"Đã xóa file kinh nghiệm cũ: {file_path}")
            except Exception as e:
                self.logger.warning(f"Không thể xóa file {file_path}: {str(e)}")
        
        if deleted_count > 0:
            self.logger.info(f"Đã dọn dẹp {deleted_count} file kinh nghiệm cũ")
        
        return deleted_count


def create_experience_manager(
    agent_type: str,
    model_version: str,
    memory_size: int = 100000,
    use_prioritized: bool = False,
    preserve_experience: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None
) -> ExperienceManager:
    """
    Hàm tiện ích để tạo ExperienceManager.
    
    Args:
        agent_type: Loại agent ("dqn", "ppo", "a2c", v.v.)
        model_version: Phiên bản hiện tại của model
        memory_size: Kích thước bộ nhớ kinh nghiệm
        use_prioritized: Sử dụng prioritized experience replay
        preserve_experience: Giữ lại kinh nghiệm giữa các phiên bản
        output_dir: Thư mục lưu trữ kinh nghiệm
        logger: Logger tùy chỉnh
        
    Returns:
        ExperienceManager đã được khởi tạo
    """
    # Khởi tạo ExperienceManager
    experience_manager = ExperienceManager(
        agent_type=agent_type,
        model_version=model_version,
        memory_size=memory_size,
        use_prioritized=use_prioritized,
        preserve_experience=preserve_experience,
        output_dir=output_dir,
        logger=logger
    )
    
    return experience_manager