"""
Quản lý chiến lược giao dịch.
File này định nghĩa lớp StrategyManager để quản lý các chiến lược giao dịch,
bao gồm thêm/xóa/cập nhật chiến lược, lưu/tải chiến lược, và cập nhật hiệu suất.
"""

import os
import time
import logging
import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from enum import Enum
from datetime import datetime
import numpy as np

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import get_logger
from config.system_config import get_system_config, MODEL_DIR

class StrategyStatus(Enum):
    """
    Trạng thái của chiến lược.
    Định nghĩa các trạng thái có thể có của một chiến lược.
    """
    ACTIVE = "active"           # Chiến lược đang hoạt động
    INACTIVE = "inactive"       # Chiến lược không hoạt động
    PENDING = "pending"         # Chiến lược đang chờ kích hoạt
    COMPLETED = "completed"     # Chiến lược đã hoàn thành
    FAILED = "failed"           # Chiến lược thất bại
    PAUSED = "paused"           # Chiến lược tạm dừng
    TESTING = "testing"         # Chiến lược đang trong giai đoạn kiểm thử
    OPTIMIZING = "optimizing"   # Chiến lược đang được tối ưu hóa

class StrategyType(Enum):
    """
    Loại chiến lược.
    Định nghĩa các loại chiến lược có thể có.
    """
    TREND_FOLLOWING = "trend_following"     # Chiến lược theo xu hướng
    MEAN_REVERSION = "mean_reversion"       # Chiến lược hồi quy về trung bình
    BREAKOUT = "breakout"                  # Chiến lược đột phá
    MOMENTUM = "momentum"                  # Chiến lược momentum
    ARBITRAGE = "arbitrage"                # Chiến lược chênh lệch giá
    MARKET_MAKING = "market_making"        # Chiến lược tạo lập thị trường
    SENTIMENT = "sentiment"                # Chiến lược theo tâm lý thị trường
    HYBRID = "hybrid"                      # Chiến lược kết hợp
    CUSTOM = "custom"                      # Chiến lược tùy chỉnh

class StrategyManager:
    """
    Lớp quản lý chiến lược giao dịch.
    Chịu trách nhiệm quản lý các chiến lược, bao gồm thêm/xóa/cập nhật chiến lược,
    lưu/tải chiến lược, và cập nhật hiệu suất.
    """
    
    def __init__(
        self,
        strategies: Optional[Dict[str, Dict[str, Any]]] = None,
        save_dir: Optional[Union[str, Path]] = None,
        name: str = "strategy_manager",
        max_strategies: int = 100,
        auto_save: bool = True,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo quản lý chiến lược.
        
        Args:
            strategies: Dictionary các chiến lược ban đầu (id: thông tin chiến lược)
            save_dir: Thư mục lưu chiến lược
            name: Tên của quản lý chiến lược
            max_strategies: Số lượng chiến lược tối đa có thể quản lý
            auto_save: Tự động lưu sau mỗi thay đổi
            logger: Logger tùy chỉnh
            **kwargs: Các tham số tùy chọn khác
        """
        # Thiết lập logger
        self.logger = logger or get_logger("strategy_manager")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập các thuộc tính
        self.name = name
        self.max_strategies = max_strategies
        self.auto_save = auto_save
        self.kwargs = kwargs
        
        # Thư mục lưu trữ
        if save_dir is None:
            save_dir = MODEL_DIR / 'strategy_queue'
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo danh sách chiến lược
        self.strategies = strategies or {}
        
        # Các biến theo dõi
        self.last_update_time = time.time()
        self.last_loaded_time = 0
        self.last_saved_time = 0
        self.modification_count = 0
        
        # Tạo file name.json để lưu
        self.save_file = self.save_dir / f"{self.name}.json"
        
        # Nếu đã có file, tải dữ liệu
        if self.save_file.exists() and not self.strategies:
            self.load_strategies()
        # Nếu có strategies được truyền vào và không có file, lưu strategies
        elif self.strategies and not self.save_file.exists() and self.auto_save:
            self.save_strategies()
        
        self.logger.info(
            f"Đã khởi tạo {self.__class__.__name__} với {len(self.strategies)} chiến lược, "
            f"auto_save={self.auto_save}, max_strategies={self.max_strategies}"
        )
    
    def add_strategy(
        self,
        strategy_config: Dict[str, Any],
        strategy_id: Optional[str] = None,
        overwrite: bool = False
    ) -> Tuple[bool, str]:
        """
        Thêm chiến lược mới vào danh sách.
        
        Args:
            strategy_config: Cấu hình của chiến lược
            strategy_id: ID của chiến lược (tạo mới nếu không có)
            overwrite: Ghi đè nếu đã tồn tại
            
        Returns:
            Tuple (success, strategy_id)
        """
        # Kiểm tra số lượng chiến lược
        if len(self.strategies) >= self.max_strategies and strategy_id not in self.strategies:
            self.logger.warning(
                f"Đã đạt giới hạn số lượng chiến lược tối đa ({self.max_strategies})"
            )
            return False, ""
        
        # Tạo ID nếu không có
        if not strategy_id:
            strategy_id = str(uuid.uuid4())
        
        # Kiểm tra ID đã tồn tại
        if strategy_id in self.strategies and not overwrite:
            self.logger.warning(
                f"Chiến lược với ID '{strategy_id}' đã tồn tại. Sử dụng overwrite=True để ghi đè."
            )
            return False, strategy_id
        
        # Kiểm tra và chuẩn hóa cấu hình chiến lược
        if not self._validate_strategy_config(strategy_config):
            self.logger.error(f"Cấu hình chiến lược không hợp lệ: {strategy_config}")
            return False, strategy_id
        
        # Thêm metadata cho chiến lược
        if "metadata" not in strategy_config:
            strategy_config["metadata"] = {}
        
        strategy_config["metadata"].update({
            "id": strategy_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": strategy_config["metadata"].get("status", StrategyStatus.INACTIVE.value)
        })
        
        # Thêm vào danh sách
        self.strategies[strategy_id] = strategy_config
        
        # Cập nhật biến theo dõi
        self.last_update_time = time.time()
        self.modification_count += 1
        
        # Tự động lưu nếu được cấu hình
        if self.auto_save:
            self.save_strategies()
        
        self.logger.info(f"Đã thêm chiến lược với ID: {strategy_id}")
        return True, strategy_id
    
    def update_strategy(
        self,
        strategy_id: str,
        updates: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """
        Cập nhật chiến lược hiện có.
        
        Args:
            strategy_id: ID của chiến lược cần cập nhật
            updates: Các cập nhật cần áp dụng
            merge: Kết hợp với cấu hình hiện có thay vì thay thế hoàn toàn
            
        Returns:
            True nếu cập nhật thành công, False nếu không
        """
        # Kiểm tra chiến lược tồn tại
        if strategy_id not in self.strategies:
            self.logger.warning(f"Không tìm thấy chiến lược với ID: {strategy_id}")
            return False
        
        # Lấy cấu hình hiện tại
        current_config = self.strategies[strategy_id]
        
        if merge:
            # Cập nhật đệ quy các trường cần thiết
            self._recursive_update(current_config, updates)
        else:
            # Thay thế hoàn toàn, nhưng giữ lại một số metadata quan trọng
            if "metadata" in current_config:
                metadata = current_config["metadata"].copy()
                updates["metadata"] = updates.get("metadata", {})
                updates["metadata"]["id"] = strategy_id
                updates["metadata"]["created_at"] = metadata.get("created_at")
                
                # Cập nhật các trường metadata khác
                for key, value in metadata.items():
                    if key not in updates["metadata"] and key not in ["updated_at"]:
                        updates["metadata"][key] = value
            
            # Thay thế cấu hình
            current_config = updates
            self.strategies[strategy_id] = current_config
        
        # Cập nhật thời gian cập nhật
        if "metadata" in current_config:
            current_config["metadata"]["updated_at"] = datetime.now().isoformat()
        
        # Cập nhật biến theo dõi
        self.last_update_time = time.time()
        self.modification_count += 1
        
        # Tự động lưu nếu được cấu hình
        if self.auto_save:
            self.save_strategies()
        
        self.logger.info(f"Đã cập nhật chiến lược với ID: {strategy_id}")
        return True
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """
        Xóa chiến lược khỏi danh sách.
        
        Args:
            strategy_id: ID của chiến lược cần xóa
            
        Returns:
            True nếu xóa thành công, False nếu không
        """
        # Kiểm tra chiến lược tồn tại
        if strategy_id not in self.strategies:
            self.logger.warning(f"Không tìm thấy chiến lược với ID: {strategy_id}")
            return False
        
        # Xóa chiến lược
        del self.strategies[strategy_id]
        
        # Cập nhật biến theo dõi
        self.last_update_time = time.time()
        self.modification_count += 1
        
        # Tự động lưu nếu được cấu hình
        if self.auto_save:
            self.save_strategies()
        
        self.logger.info(f"Đã xóa chiến lược với ID: {strategy_id}")
        return True
    
    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin chiến lược theo ID.
        
        Args:
            strategy_id: ID của chiến lược
            
        Returns:
            Thông tin chiến lược nếu tồn tại, None nếu không
        """
        if strategy_id not in self.strategies:
            self.logger.warning(f"Không tìm thấy chiến lược với ID: {strategy_id}")
            return None
        
        return self.strategies[strategy_id]
    
    def get_all_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Lấy tất cả chiến lược.
        
        Returns:
            Dictionary các chiến lược (id: thông tin chiến lược)
        """
        return self.strategies.copy()
    
    def get_active_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Lấy tất cả chiến lược đang hoạt động.
        
        Returns:
            Dictionary các chiến lược đang hoạt động
        """
        active_strategies = {}
        
        for strategy_id, strategy_config in self.strategies.items():
            metadata = strategy_config.get("metadata", {})
            if metadata.get("status") == StrategyStatus.ACTIVE.value:
                active_strategies[strategy_id] = strategy_config
        
        return active_strategies
    
    def filter_strategies(
        self,
        filters: Dict[str, Any],
        match_all: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Lọc chiến lược theo các tiêu chí.
        
        Args:
            filters: Dictionary các tiêu chí lọc
            match_all: True nếu phải thỏa mãn tất cả tiêu chí, False nếu chỉ cần một
            
        Returns:
            Dictionary các chiến lược thỏa mãn
        """
        filtered_strategies = {}
        
        for strategy_id, strategy_config in self.strategies.items():
            match = True if match_all else False
            
            for key, value in filters.items():
                # Hỗ trợ lọc lồng nhau với dấu chấm
                # Ví dụ: "metadata.status" = "active"
                keys = key.split(".")
                curr_config = strategy_config
                
                # Duyệt qua các cấp của key
                for k in keys:
                    if isinstance(curr_config, dict) and k in curr_config:
                        curr_config = curr_config[k]
                    else:
                        curr_config = None
                        break
                
                # Kiểm tra giá trị
                if curr_config == value:
                    if not match_all:
                        match = True
                        break
                elif match_all:
                    match = False
                    break
            
            if match:
                filtered_strategies[strategy_id] = strategy_config
        
        return filtered_strategies
    
    def update_strategy_status(
        self,
        strategy_id: str,
        status: Union[str, StrategyStatus]
    ) -> bool:
        """
        Cập nhật trạng thái của chiến lược.
        
        Args:
            strategy_id: ID của chiến lược
            status: Trạng thái mới
            
        Returns:
            True nếu cập nhật thành công, False nếu không
        """
        # Chuyển đổi sang string nếu là enum
        if isinstance(status, StrategyStatus):
            status = status.value
        
        # Kiểm tra chiến lược tồn tại
        if strategy_id not in self.strategies:
            self.logger.warning(f"Không tìm thấy chiến lược với ID: {strategy_id}")
            return False
        
        # Kiểm tra trạng thái hợp lệ
        valid_statuses = [s.value for s in StrategyStatus]
        if status not in valid_statuses:
            self.logger.warning(
                f"Trạng thái không hợp lệ: {status}. "
                f"Trạng thái hợp lệ: {', '.join(valid_statuses)}"
            )
            return False
        
        # Cập nhật trạng thái
        if "metadata" not in self.strategies[strategy_id]:
            self.strategies[strategy_id]["metadata"] = {}
        
        self.strategies[strategy_id]["metadata"]["status"] = status
        self.strategies[strategy_id]["metadata"]["updated_at"] = datetime.now().isoformat()
        
        # Cập nhật biến theo dõi
        self.last_update_time = time.time()
        self.modification_count += 1
        
        # Tự động lưu nếu được cấu hình
        if self.auto_save:
            self.save_strategies()
        
        self.logger.info(f"Đã cập nhật trạng thái chiến lược {strategy_id} thành {status}")
        return True
    
    def update_strategy_performance(
        self,
        strategy_id: str,
        performance_metrics: Dict[str, Any]
    ) -> bool:
        """
        Cập nhật hiệu suất của chiến lược.
        
        Args:
            strategy_id: ID của chiến lược
            performance_metrics: Các chỉ số hiệu suất cần cập nhật
            
        Returns:
            True nếu cập nhật thành công, False nếu không
        """
        # Kiểm tra chiến lược tồn tại
        if strategy_id not in self.strategies:
            self.logger.warning(f"Không tìm thấy chiến lược với ID: {strategy_id}")
            return False
        
        # Cập nhật hiệu suất
        if "performance" not in self.strategies[strategy_id]:
            self.strategies[strategy_id]["performance"] = {}
        
        # Thêm timestamp
        performance_metrics["timestamp"] = datetime.now().isoformat()
        
        # Cập nhật và lưu lịch sử hiệu suất
        if "history" not in self.strategies[strategy_id]["performance"]:
            self.strategies[strategy_id]["performance"]["history"] = []
        
        # Giới hạn lịch sử hiệu suất
        max_history = self.kwargs.get("max_performance_history", 100)
        history = self.strategies[strategy_id]["performance"]["history"]
        
        # Thêm vào lịch sử
        history.append(performance_metrics)
        
        # Cắt bớt nếu vượt quá giới hạn
        if len(history) > max_history:
            self.strategies[strategy_id]["performance"]["history"] = history[-max_history:]
        
        # Cập nhật chỉ số hiệu suất gần nhất
        self.strategies[strategy_id]["performance"]["latest"] = performance_metrics
        
        # Cập nhật thời gian cập nhật
        if "metadata" in self.strategies[strategy_id]:
            self.strategies[strategy_id]["metadata"]["updated_at"] = datetime.now().isoformat()
        
        # Cập nhật biến theo dõi
        self.last_update_time = time.time()
        self.modification_count += 1
        
        # Tự động lưu nếu được cấu hình
        if self.auto_save:
            self.save_strategies()
        
        self.logger.debug(f"Đã cập nhật hiệu suất cho chiến lược {strategy_id}")
        return True
    
    def calculate_strategy_rank(
        self,
        strategy_id: str,
        custom_ranking_fn: Optional[Callable] = None
    ) -> float:
        """
        Tính toán thứ hạng của chiến lược dựa trên hiệu suất.
        
        Args:
            strategy_id: ID của chiến lược
            custom_ranking_fn: Hàm tính toán thứ hạng tùy chỉnh
            
        Returns:
            Thứ hạng của chiến lược (giá trị cao = thứ hạng cao)
        """
        # Kiểm tra chiến lược tồn tại
        if strategy_id not in self.strategies:
            self.logger.warning(f"Không tìm thấy chiến lược với ID: {strategy_id}")
            return 0.0
        
        strategy_config = self.strategies[strategy_id]
        
        # Sử dụng hàm tính toán tùy chỉnh nếu có
        if custom_ranking_fn is not None:
            try:
                return custom_ranking_fn(strategy_config)
            except Exception as e:
                self.logger.error(f"Lỗi khi sử dụng hàm tính toán thứ hạng tùy chỉnh: {str(e)}")
                # Tiếp tục với phương pháp mặc định
        
        # Phương pháp tính toán mặc định
        rank = 0.0
        
        # 1. Lấy điểm ưu tiên từ metadata
        metadata = strategy_config.get("metadata", {})
        priority = metadata.get("priority", 0)
        rank += priority * 10  # Trọng số cao cho ưu tiên
        
        # 2. Trạng thái ảnh hưởng đến thứ hạng
        status = metadata.get("status")
        if status == StrategyStatus.ACTIVE.value:
            rank += 50
        elif status == StrategyStatus.PAUSED.value:
            rank += 20
        elif status == StrategyStatus.PENDING.value:
            rank += 30
        elif status == StrategyStatus.TESTING.value:
            rank += 10
        elif status == StrategyStatus.COMPLETED.value or status == StrategyStatus.FAILED.value:
            rank -= 50  # Giảm thứ hạng cho các chiến lược đã hoàn thành hoặc thất bại
        
        # 3. Hiệu suất gần đây
        performance = strategy_config.get("performance", {})
        latest = performance.get("latest", {})
        
        # Thêm điểm cho các chỉ số hiệu suất
        if "profit" in latest:
            rank += latest["profit"] * 5
        
        if "win_rate" in latest:
            rank += latest["win_rate"] * 2
        
        if "sharpe_ratio" in latest:
            rank += latest["sharpe_ratio"] * 3
        
        if "max_drawdown" in latest:
            rank -= abs(latest["max_drawdown"]) * 2  # Giảm thứ hạng cho drawdown cao
        
        # 4. Tính toán từ lịch sử hiệu suất
        history = performance.get("history", [])
        if history:
            # Tính trung bình các chỉ số trong 10 lần cập nhật gần nhất
            recent_history = history[-10:]
            
            # Tính trung bình profit
            avg_profit = np.mean([h.get("profit", 0) for h in recent_history if "profit" in h])
            rank += avg_profit * 2
            
            # Tính xu hướng hiệu suất
            if len(recent_history) >= 2:
                profits = [h.get("profit", 0) for h in recent_history if "profit" in h]
                if len(profits) >= 2:
                    # Phát hiện xu hướng tăng/giảm
                    trend = np.polyfit(range(len(profits)), profits, 1)[0]
                    rank += trend * 10  # Xu hướng tăng được cộng điểm
        
        # 5. Thời gian tồn tại
        if "created_at" in metadata:
            try:
                created_time = datetime.fromisoformat(metadata["created_at"])
                age_days = (datetime.now() - created_time).days
                
                # Chiến lược tồn tại lâu được cộng điểm (tối đa 30 ngày)
                rank += min(age_days, 30) * 0.1
            except (ValueError, TypeError):
                pass
        
        return rank
    
    def rank_strategies(
        self,
        custom_ranking_fn: Optional[Callable] = None,
        filter_inactive: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Xếp hạng tất cả chiến lược dựa trên hiệu suất.
        
        Args:
            custom_ranking_fn: Hàm tính toán thứ hạng tùy chỉnh
            filter_inactive: Lọc bỏ các chiến lược không hoạt động
            
        Returns:
            Danh sách các tuple (strategy_id, rank) đã sắp xếp
        """
        rankings = []
        
        for strategy_id in self.strategies:
            # Kiểm tra trạng thái nếu cần lọc
            if filter_inactive:
                metadata = self.strategies[strategy_id].get("metadata", {})
                status = metadata.get("status")
                
                if status == StrategyStatus.INACTIVE.value or status == StrategyStatus.FAILED.value:
                    continue
            
            # Tính toán thứ hạng
            rank = self.calculate_strategy_rank(strategy_id, custom_ranking_fn)
            rankings.append((strategy_id, rank))
        
        # Sắp xếp theo thứ hạng giảm dần
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def save_strategies(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Lưu tất cả chiến lược vào file.
        
        Args:
            file_path: Đường dẫn file lưu (sử dụng file mặc định nếu không có)
            
        Returns:
            True nếu lưu thành công, False nếu không
        """
        if file_path is None:
            file_path = self.save_file
        
        try:
            # Chuẩn bị dữ liệu lưu
            save_data = {
                "name": self.name,
                "strategies": self.strategies,
                "last_update_time": time.time(),
                "last_saved_time": time.time(),
                "version": "1.0",
                "metadata": {
                    "strategy_count": len(self.strategies),
                    "saved_at": datetime.now().isoformat()
                }
            }
            
            # Lưu vào file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            # Cập nhật biến theo dõi
            self.last_saved_time = time.time()
            
            self.logger.info(f"Đã lưu {len(self.strategies)} chiến lược vào {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu chiến lược: {str(e)}")
            return False
    
    def load_strategies(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Tải chiến lược từ file.
        
        Args:
            file_path: Đường dẫn file tải (sử dụng file mặc định nếu không có)
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        if file_path is None:
            file_path = self.save_file
        
        if not os.path.exists(file_path):
            self.logger.warning(f"Không tìm thấy file chiến lược tại {file_path}")
            return False
        
        try:
            # Đọc từ file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Kiểm tra dữ liệu hợp lệ
            if "strategies" not in data:
                self.logger.error(f"Dữ liệu không hợp lệ trong file {file_path}")
                return False
            
            # Cập nhật dữ liệu
            self.strategies = data["strategies"]
            
            # Cập nhật tên nếu có
            if "name" in data:
                self.name = data["name"]
            
            # Cập nhật biến theo dõi
            self.last_loaded_time = time.time()
            self.last_update_time = time.time()
            
            self.logger.info(f"Đã tải {len(self.strategies)} chiến lược từ {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải chiến lược: {str(e)}")
            return False
    
    def export_strategies(
        self,
        strategy_ids: Optional[List[str]] = None,
        file_path: Optional[Union[str, Path]] = None
    ) -> bool:
        """
        Xuất một số chiến lược ra file riêng.
        
        Args:
            strategy_ids: Danh sách ID chiến lược cần xuất (tất cả nếu None)
            file_path: Đường dẫn file xuất
            
        Returns:
            True nếu xuất thành công, False nếu không
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.save_dir / f"exported_strategies_{timestamp}.json"
        
        # Lấy chiến lược cần xuất
        strategies_to_export = {}
        
        if strategy_ids is None:
            # Xuất tất cả chiến lược
            strategies_to_export = self.strategies
        else:
            # Xuất chiến lược theo danh sách ID
            for strategy_id in strategy_ids:
                if strategy_id in self.strategies:
                    strategies_to_export[strategy_id] = self.strategies[strategy_id]
                else:
                    self.logger.warning(f"Không tìm thấy chiến lược với ID: {strategy_id}")
        
        if not strategies_to_export:
            self.logger.warning("Không có chiến lược nào để xuất")
            return False
        
        try:
            # Chuẩn bị dữ liệu xuất
            export_data = {
                "name": f"exported_{self.name}",
                "strategies": strategies_to_export,
                "metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "source_manager": self.name,
                    "strategy_count": len(strategies_to_export)
                }
            }
            
            # Lưu vào file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Đã xuất {len(strategies_to_export)} chiến lược vào {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi xuất chiến lược: {str(e)}")
            return False
    
    def import_strategies(
        self,
        file_path: Union[str, Path],
        overwrite: bool = False
    ) -> Tuple[int, int]:
        """
        Nhập chiến lược từ file.
        
        Args:
            file_path: Đường dẫn file nhập
            overwrite: Ghi đè nếu đã tồn tại
            
        Returns:
            Tuple (số chiến lược đã nhập, số chiến lược thất bại)
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"Không tìm thấy file chiến lược tại {file_path}")
            return 0, 0
        
        try:
            # Đọc từ file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Kiểm tra dữ liệu hợp lệ
            if "strategies" not in data:
                self.logger.error(f"Dữ liệu không hợp lệ trong file {file_path}")
                return 0, 0
            
            # Nhập từng chiến lược
            imported_count = 0
            failed_count = 0
            
            for strategy_id, strategy_config in data["strategies"].items():
                success, _ = self.add_strategy(strategy_config, strategy_id, overwrite)
                
                if success:
                    imported_count += 1
                else:
                    failed_count += 1
            
            # Tự động lưu nếu được cấu hình
            if self.auto_save and imported_count > 0:
                self.save_strategies()
            
            self.logger.info(
                f"Đã nhập {imported_count} chiến lược, thất bại {failed_count} chiến lược từ {file_path}"
            )
            return imported_count, failed_count
        except Exception as e:
            self.logger.error(f"Lỗi khi nhập chiến lược: {str(e)}")
            return 0, 0
    
    def _validate_strategy_config(self, strategy_config: Dict[str, Any]) -> bool:
        """
        Kiểm tra tính hợp lệ của cấu hình chiến lược.
        
        Args:
            strategy_config: Cấu hình chiến lược cần kiểm tra
            
        Returns:
            True nếu hợp lệ, False nếu không
        """
        # Kiểm tra các trường bắt buộc
        required_fields = ["name", "description", "params"]
        
        for field in required_fields:
            if field not in strategy_config:
                self.logger.warning(f"Thiếu trường bắt buộc '{field}' trong cấu hình chiến lược")
                return False
        
        # Kiểm tra và chuẩn hóa loại chiến lược
        if "type" in strategy_config:
            strategy_type = strategy_config["type"]
            
            # Kiểm tra nếu là enum
            valid_types = [t.value for t in StrategyType]
            if strategy_type not in valid_types:
                self.logger.warning(
                    f"Loại chiến lược không hợp lệ: {strategy_type}. "
                    f"Loại hợp lệ: {', '.join(valid_types)}"
                )
                return False
        else:
            # Gán loại mặc định nếu không có
            strategy_config["type"] = StrategyType.CUSTOM.value
        
        # Kiểm tra params
        params = strategy_config["params"]
        if not isinstance(params, dict):
            self.logger.warning("Tham số phải là dictionary")
            return False
        
        # Có thể thêm các ràng buộc khác
        
        return True
    
    def _recursive_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Cập nhật đệ quy một dictionary bằng một dictionary khác.
        
        Args:
            target: Dictionary đích
            source: Dictionary nguồn
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Cập nhật đệ quy nếu cả hai là dict
                self._recursive_update(target[key], value)
            else:
                # Gán trực tiếp
                target[key] = value