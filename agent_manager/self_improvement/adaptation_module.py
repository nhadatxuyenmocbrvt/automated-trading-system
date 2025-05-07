"""
Module thích ứng cho agent.
File này định nghĩa lớp AdaptationModule để tự động điều chỉnh và tối ưu hóa
các tham số của agent dựa trên hiệu suất theo thời gian, thích ứng với
thay đổi của thị trường và môi trường giao dịch.
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import copy
import time

# Thêm thư mục gốc vào path để import được các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config

from models.agents.base_agent import BaseAgent
from models.agents.dqn_agent import DQNAgent
from models.agents.ppo_agent import PPOAgent
from models.agents.a2c_agent import A2CAgent

from environments.base_environment import BaseEnvironment
from environments.trading_gym.trading_env import TradingEnv

from agent_manager.self_improvement.agent_evaluator import AgentEvaluator

class AdaptationModule:
    """
    Lớp quản lý thích ứng tự động cho các agent.
    
    Cung cấp các phương thức để:
    1. Phân tích hiệu suất agent và xác định các tham số cần cải thiện
    2. Tự động điều chỉnh các siêu tham số dựa trên hiệu suất
    3. Theo dõi hiệu quả của các thay đổi tham số
    4. Phát hiện sự thay đổi trong thị trường và kích hoạt quá trình thích ứng
    5. Lưu và tải trạng thái thích ứng để tiếp tục quá trình cải thiện
    """
    
    def __init__(
        self,
        agent: Optional[BaseAgent] = None,
        evaluator: Optional[AgentEvaluator] = None,
        env: Optional[BaseEnvironment] = None,
        output_dir: Optional[Union[str, Path]] = None,
        adaptation_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo module thích ứng.
        
        Args:
            agent: Agent cần thích ứng
            evaluator: Công cụ đánh giá agent
            env: Môi trường để đánh giá hiệu suất
            output_dir: Thư mục đầu ra cho kết quả thích ứng
            adaptation_config: Cấu hình thích ứng bổ sung
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("adaptation_module")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập các thành phần chính
        self.agent = agent
        self.evaluator = evaluator
        self.env = env
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(self.system_config.get("model_dir", "./models")) / "adaptations" / timestamp
        else:
            self.output_dir = Path(output_dir)
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập cấu hình thích ứng mặc định
        self.default_config = {
            "adaptation_frequency": 5,            # Số lần đánh giá trước khi thích ứng
            "min_evaluations_required": 3,        # Số đánh giá tối thiểu trước khi bắt đầu thích ứng
            "exploration_factor": 0.2,            # Mức độ khám phá khi thử tham số mới
            "max_param_change": 0.3,              # Thay đổi tối đa cho mỗi tham số (%)
            "num_evaluations_per_adaptation": 3,  # Số lần đánh giá cho mỗi tham số mới
            "success_threshold": 0.05,            # Cải thiện tối thiểu để chấp nhận thay đổi (5%)
            "revert_failed_changes": True,        # Quay lại tham số cũ nếu thất bại
            "track_market_changes": True,         # Theo dõi sự thay đổi của thị trường
            "market_change_threshold": 0.1,       # Ngưỡng thay đổi thị trường để kích hoạt thích ứng
            "adapt_on_market_change": True,       # Thích ứng khi phát hiện thay đổi thị trường
            "adaptable_parameters": {             # Các tham số có thể thích ứng và phạm vi giá trị
                "learning_rate": {
                    "min": 0.0001,
                    "max": 0.01,
                    "type": "float",
                    "scale": "log"               # Thang logarit để tìm kiếm hiệu quả hơn
                },
                "gamma": {
                    "min": 0.9,
                    "max": 0.999,
                    "type": "float",
                    "scale": "linear"
                },
                "epsilon": {
                    "min": 0.01,
                    "max": 1.0,
                    "type": "float",
                    "scale": "linear"
                },
                "epsilon_decay": {
                    "min": 0.97,
                    "max": 0.999,
                    "type": "float",
                    "scale": "linear"
                },
                "batch_size": {
                    "min": 16,
                    "max": 256,
                    "type": "int",
                    "scale": "log",
                    "step": 16                    # Bước nhảy cho giá trị nguyên
                },
                "memory_size": {
                    "min": 1000,
                    "max": 100000,
                    "type": "int",
                    "scale": "log",
                    "step": 1000
                },
                "hidden_layers": {
                    "type": "list_int",
                    "options": [
                        [64],
                        [128],
                        [256],
                        [64, 64],
                        [128, 64],
                        [256, 128],
                        [128, 64, 32]
                    ]
                }
            },
            "importance_weights": {              # Trọng số tầm quan trọng của các metric
                "mean_reward": 1.0,
                "sharpe_ratio": 0.8,
                "sortino_ratio": 0.7,
                "win_rate": 0.6,
                "max_drawdown": 0.5,
                "volatility": 0.4
            },
            "optimization_goal": "mean_reward",   # Metric chính để tối ưu hóa
            "cooldown_period": 2,                # Số đánh giá chờ giữa các lần thích ứng
            "max_consecutive_failures": 3         # Số lần thất bại liên tiếp trước khi dừng thích ứng
        }
        
        # Cập nhật cấu hình thích ứng từ tham số nếu có
        self.adaptation_config = self.default_config.copy()
        if adaptation_config:
            # Cập nhật đệ quy để giữ lại các tham số mặc định nếu không được chỉ định
            self._recursive_update(self.adaptation_config, adaptation_config)
        
        # Khởi tạo các biến theo dõi
        self.adaptation_history = []   # Lịch sử các lần thích ứng
        self.evaluation_results = []   # Kết quả đánh giá 
        self.best_parameters = {}      # Tham số tốt nhất tìm được
        self.best_performance = None   # Hiệu suất tốt nhất
        self.current_exploration = self.adaptation_config["exploration_factor"]  # Mức độ khám phá hiện tại
        self.consecutive_failures = 0   # Số lần thất bại liên tiếp
        self.cooldown_counter = 0       # Bộ đếm thời gian chờ
        self.last_adaptation_time = None  # Thời gian thích ứng gần nhất
        
        # Lưu trạng thái ban đầu của agent
        self._save_initial_state()
        
        self.logger.info(f"Đã khởi tạo AdaptationModule cho {agent.__class__.__name__ if agent else 'chưa có agent'}")
    
    def _recursive_update(self, target_dict: dict, update_dict: dict) -> None:
        """
        Cập nhật đệ quy một dict với các giá trị từ dict khác, 
        giữ nguyên cấu trúc và các giá trị không được cập nhật.
        
        Args:
            target_dict: Dict đích cần cập nhật
            update_dict: Dict chứa các giá trị cập nhật
        """
        for k, v in update_dict.items():
            if isinstance(v, dict) and k in target_dict and isinstance(target_dict[k], dict):
                self._recursive_update(target_dict[k], v)
            else:
                target_dict[k] = v
    
    def _save_initial_state(self) -> None:
        """
        Lưu trạng thái ban đầu của agent để có thể quay lại nếu cần.
        """
        if self.agent is None:
            return
        
        # Lưu các tham số của agent
        self.initial_parameters = {}
        for param_name in self.adaptation_config["adaptable_parameters"].keys():
            if hasattr(self.agent, param_name):
                self.initial_parameters[param_name] = copy.deepcopy(getattr(self.agent, param_name))
        
        # Nếu không có tham số nào được tìm thấy, thử lấy từ config
        if not self.initial_parameters and hasattr(self.agent, "config"):
            for param_name in self.adaptation_config["adaptable_parameters"].keys():
                if param_name in self.agent.config:
                    self.initial_parameters[param_name] = copy.deepcopy(self.agent.config[param_name])
        
        # Lưu trạng thái ban đầu làm trạng thái tốt nhất
        self.best_parameters = self.initial_parameters.copy()
        
        self.logger.debug(f"Đã lưu trạng thái ban đầu với {len(self.initial_parameters)} tham số")
    
    def setup(
        self, 
        agent: Optional[BaseAgent] = None,
        evaluator: Optional[AgentEvaluator] = None,
        env: Optional[BaseEnvironment] = None
    ) -> bool:
        """
        Thiết lập các thành phần cần thiết cho module thích ứng.
        
        Args:
            agent: Agent cần thích ứng
            evaluator: Công cụ đánh giá agent
            env: Môi trường để đánh giá hiệu suất
            
        Returns:
            True nếu thiết lập thành công, False nếu không
        """
        # Cập nhật các thành phần nếu được cung cấp
        if agent is not None:
            self.agent = agent
            # Lưu trạng thái ban đầu của agent mới
            self._save_initial_state()
        
        if evaluator is not None:
            self.evaluator = evaluator
        
        if env is not None:
            self.env = env
        
        # Kiểm tra các thành phần bắt buộc
        if self.agent is None:
            self.logger.error("Không thể thiết lập AdaptationModule: Thiếu agent")
            return False
        
        if self.evaluator is None:
            self.logger.warning("Chưa có evaluator. Sẽ tạo một evaluator mới.")
            try:
                self.evaluator = AgentEvaluator(
                    agents=[{"id": "current_agent", "agent": self.agent}],
                    environments=[{"id": "current_env", "environment": self.env}] if self.env else [],
                    output_dir=self.output_dir / "evaluations",
                    logger=self.logger
                )
            except Exception as e:
                self.logger.error(f"Không thể tạo evaluator: {str(e)}")
                return False
        
        if self.env is None:
            self.logger.warning("Chưa có môi trường. Module thích ứng sẽ sử dụng môi trường được cung cấp khi đánh giá.")
        
        # Kiểm tra các thông tin cấu hình cần thiết
        self._validate_configuration()
        
        self.logger.info(f"Đã thiết lập AdaptationModule thành công cho {self.agent.__class__.__name__}")
        return True
    
    def _validate_configuration(self) -> None:
        """
        Kiểm tra và điều chỉnh cấu hình nếu cần thiết.
        """
        # Loại agent hiện tại
        agent_class = self.agent.__class__.__name__
        
        # Kiểm tra xem các tham số thích ứng có phù hợp với loại agent không
        adaptable_params = self.adaptation_config["adaptable_parameters"]
        
        if agent_class == "DQNAgent":
            # Đảm bảo các tham số DQN đặc thù có sẵn
            if "double_dqn" not in adaptable_params:
                adaptable_params["double_dqn"] = {
                    "type": "bool",
                    "options": [True, False]
                }
            if "dueling" not in adaptable_params:
                adaptable_params["dueling"] = {
                    "type": "bool",
                    "options": [True, False]
                }
            if "prioritized_replay" not in adaptable_params:
                adaptable_params["prioritized_replay"] = {
                    "type": "bool",
                    "options": [True, False]
                }
                
        elif agent_class == "PPOAgent":
            # Đảm bảo các tham số PPO đặc thù có sẵn
            if "clip_ratio" not in adaptable_params:
                adaptable_params["clip_ratio"] = {
                    "min": 0.1,
                    "max": 0.3,
                    "type": "float",
                    "scale": "linear"
                }
            if "entropy_coef" not in adaptable_params:
                adaptable_params["entropy_coef"] = {
                    "min": 0.0,
                    "max": 0.1,
                    "type": "float",
                    "scale": "linear"
                }
                
        elif agent_class == "A2CAgent":
            # Đảm bảo các tham số A2C đặc thù có sẵn
            if "entropy_coef" not in adaptable_params:
                adaptable_params["entropy_coef"] = {
                    "min": 0.0,
                    "max": 0.1,
                    "type": "float",
                    "scale": "linear"
                }
            if "value_coef" not in adaptable_params:
                adaptable_params["value_coef"] = {
                    "min": 0.5,
                    "max": 1.0,
                    "type": "float",
                    "scale": "linear"
                }
        
        # Ghi log thông tin các tham số có thể thích ứng
        self.logger.info(f"Đã xác nhận {len(adaptable_params)} tham số có thể thích ứng cho {agent_class}")
    
    def add_evaluation_result(self, evaluation_result: Dict[str, Any]) -> None:
        """
        Thêm kết quả đánh giá mới vào lịch sử.
        
        Args:
            evaluation_result: Dict chứa kết quả đánh giá từ AgentEvaluator
        """
        if not evaluation_result:
            self.logger.warning("Không thể thêm kết quả đánh giá rỗng")
            return
        
        # Thêm kết quả vào lịch sử
        self.evaluation_results.append({
            "timestamp": datetime.now().isoformat(),
            "result": evaluation_result
        })
        
        # Cập nhật hiệu suất tốt nhất nếu cần
        self._update_best_performance(evaluation_result)
        
        # Giảm bộ đếm cooldown nếu đang trong thời gian chờ
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        
        # Ghi log
        eval_metric = self._get_primary_metric_value(evaluation_result)
        self.logger.info(f"Đã thêm kết quả đánh giá mới: {self.adaptation_config['optimization_goal']} = {eval_metric:.4f}")
        
        # Kiểm tra xem có cần thích ứng không
        if self._should_adapt():
            self.adapt()
    
    def _get_primary_metric_value(self, evaluation_result: Dict[str, Any]) -> float:
        """
        Lấy giá trị của metric chính từ kết quả đánh giá.
        
        Args:
            evaluation_result: Dict chứa kết quả đánh giá
            
        Returns:
            Giá trị của metric chính hoặc 0.0 nếu không tìm thấy
        """
        metric_name = self.adaptation_config["optimization_goal"]
        
        # Tìm kiếm trong basic_metrics
        if "basic_metrics" in evaluation_result and metric_name in evaluation_result["basic_metrics"]:
            return float(evaluation_result["basic_metrics"][metric_name])
        
        # Tìm kiếm trong advanced_metrics
        if "advanced_metrics" in evaluation_result and metric_name in evaluation_result["advanced_metrics"]:
            return float(evaluation_result["advanced_metrics"][metric_name])
        
        # Trường hợp đặc biệt: mean_reward có thể được lưu ở cấp cao nhất
        if metric_name == "mean_reward" and "mean_reward" in evaluation_result:
            return float(evaluation_result["mean_reward"])
        
        # Không tìm thấy
        self.logger.warning(f"Không tìm thấy metric '{metric_name}' trong kết quả đánh giá")
        return 0.0
    
    def _update_best_performance(self, evaluation_result: Dict[str, Any]) -> bool:
        """
        Cập nhật thông tin hiệu suất tốt nhất nếu kết quả mới tốt hơn.
        
        Args:
            evaluation_result: Dict chứa kết quả đánh giá
            
        Returns:
            True nếu có cập nhật, False nếu không
        """
        metric_value = self._get_primary_metric_value(evaluation_result)
        
        # Lấy giá trị metric chính
        metric_name = self.adaptation_config["optimization_goal"]
        
        # Kiểm tra xem có phải là hiệu suất tốt nhất không
        if self.best_performance is None or self._is_better_performance(metric_value, self.best_performance, metric_name):
            old_value = self.best_performance
            self.best_performance = metric_value
            
            # Lưu tham số hiện tại là tham số tốt nhất
            self._save_current_parameters_as_best()
            
            # Ghi log
            if old_value is not None:
                improvement = ((metric_value - old_value) / abs(old_value)) * 100 if old_value != 0 else float('inf')
                self.logger.info(f"Đã cập nhật hiệu suất tốt nhất: {metric_name} = {metric_value:.4f} (+{improvement:.2f}%)")
            else:
                self.logger.info(f"Đã thiết lập hiệu suất tốt nhất ban đầu: {metric_name} = {metric_value:.4f}")
            
            return True
        
        return False
    
    def _save_current_parameters_as_best(self) -> None:
        """
        Lưu tham số hiện tại của agent làm tham số tốt nhất.
        """
        if self.agent is None:
            return
        
        # Lưu các tham số của agent
        self.best_parameters = {}
        for param_name in self.adaptation_config["adaptable_parameters"].keys():
            if hasattr(self.agent, param_name):
                self.best_parameters[param_name] = copy.deepcopy(getattr(self.agent, param_name))
        
        # Nếu không có tham số nào được tìm thấy, thử lấy từ config
        if not self.best_parameters and hasattr(self.agent, "config"):
            for param_name in self.adaptation_config["adaptable_parameters"].keys():
                if param_name in self.agent.config:
                    self.best_parameters[param_name] = copy.deepcopy(self.agent.config[param_name])
    
    def _is_better_performance(self, new_value: float, current_value: float, metric_name: str) -> bool:
        """
        Kiểm tra xem giá trị mới có tốt hơn giá trị hiện tại không.
        
        Args:
            new_value: Giá trị mới
            current_value: Giá trị hiện tại
            metric_name: Tên metric để xác định hướng so sánh
            
        Returns:
            True nếu giá trị mới tốt hơn, False nếu không
        """
        # Các metric mà giá trị càng lớn càng tốt
        greater_is_better = metric_name not in ["max_drawdown", "volatility"]
        
        if greater_is_better:
            return new_value > current_value
        else:
            return new_value < current_value
    
    def _should_adapt(self) -> bool:
        """
        Kiểm tra xem có nên thích ứng tại thời điểm hiện tại không.
        
        Returns:
            True nếu nên thích ứng, False nếu không
        """
        # Nếu đang trong thời gian chờ, không thích ứng
        if self.cooldown_counter > 0:
            return False
        
        # Kiểm tra số lượng đánh giá tối thiểu
        if len(self.evaluation_results) < self.adaptation_config["min_evaluations_required"]:
            return False
        
        # Kiểm tra tần suất thích ứng
        recent_adaptations = [
            a for a in self.adaptation_history
            if (datetime.now() - datetime.fromisoformat(a["timestamp"])).total_seconds() < 3600  # 1 giờ
        ]
        
        if recent_adaptations:
            # Nếu đã thích ứng gần đây, chỉ thích ứng theo tần suất
            evals_since_last_adaptation = len(self.evaluation_results) - recent_adaptations[-1].get("evaluation_idx", 0)
            return evals_since_last_adaptation >= self.adaptation_config["adaptation_frequency"]
        
        # Nếu chưa thích ứng gần đây, kiểm tra số lượng đánh giá đã đủ chưa
        return len(self.evaluation_results) >= self.adaptation_config["adaptation_frequency"]
    
    def adapt(self) -> Dict[str, Any]:
        """
        Thực hiện quá trình thích ứng cho agent.
        
        Returns:
            Dict chứa kết quả thích ứng
        """
        if self.agent is None or self.evaluator is None:
            self.logger.error("Không thể thích ứng: Thiếu agent hoặc evaluator")
            return {"success": False, "error": "Thiếu agent hoặc evaluator"}
        
        self.logger.info(f"Bắt đầu quá trình thích ứng cho {self.agent.__class__.__name__}")
        
        # Lưu trạng thái trước khi thích ứng
        previous_parameters = self._get_current_parameters()
        previous_performance = self.best_performance
        
        # Tạo tham số mới để thử nghiệm
        new_parameters = self._generate_new_parameters()
        
        # Áp dụng tham số mới
        self._apply_parameters(new_parameters)
        
        # Đánh giá tham số mới
        evaluation_results = []
        for _ in range(self.adaptation_config["num_evaluations_per_adaptation"]):
            # Đánh giá agent với tham số mới
            if self.env:
                eval_result = self.evaluator.evaluate_agent(
                    agent_id="current_agent", 
                    env_id="current_env"
                )
            else:
                # Nếu không có env, phải dùng env được cung cấp khi đánh giá
                self.logger.warning("Không có môi trường cố định. Sẽ sử dụng kết quả đánh giá gần nhất.")
                if self.evaluation_results:
                    eval_result = self.evaluation_results[-1]["result"]
                else:
                    self.logger.error("Không có kết quả đánh giá trước đó")
                    return {"success": False, "error": "Không có kết quả đánh giá"}
            
            if "error" in eval_result:
                self.logger.error(f"Lỗi khi đánh giá tham số mới: {eval_result['error']}")
                continue
            
            evaluation_results.append(eval_result)
        
        # Tính trung bình kết quả đánh giá
        if not evaluation_results:
            self.logger.error("Không có kết quả đánh giá nào thành công")
            
            # Quay lại tham số trước đó
            self._apply_parameters(previous_parameters)
            
            return {
                "success": False,
                "error": "Không có kết quả đánh giá nào thành công",
                "timestamp": datetime.now().isoformat(),
                "previous_parameters": previous_parameters,
                "attempted_parameters": new_parameters
            }
        
        # Lấy giá trị metric chính từ các đánh giá
        metric_values = [self._get_primary_metric_value(result) for result in evaluation_results]
        avg_metric_value = sum(metric_values) / len(metric_values)
        
        # Kiểm tra xem tham số mới có tốt hơn không
        metric_name = self.adaptation_config["optimization_goal"]
        is_better = self._is_better_performance(avg_metric_value, previous_performance, metric_name)
        
        # Tính phần trăm cải thiện
        improvement = 0
        if previous_performance != 0:
            raw_diff = avg_metric_value - previous_performance
            improvement = (raw_diff / abs(previous_performance)) * 100
        
        # Nếu cải thiện vượt ngưỡng, chấp nhận tham số mới
        if is_better and abs(improvement) >= self.adaptation_config["success_threshold"]:
            self.logger.info(
                f"Tham số mới tốt hơn: {metric_name} = {avg_metric_value:.4f} "
                f"(+{improvement:.2f}% so với {previous_performance:.4f})"
            )
            
            # Cập nhật hiệu suất tốt nhất và tham số tốt nhất
            self.best_performance = avg_metric_value
            self.best_parameters = new_parameters.copy()
            
            # Đặt lại bộ đếm thất bại
            self.consecutive_failures = 0
            
            # Tăng mức độ khám phá nếu thành công
            self.current_exploration = min(self.current_exploration * 1.1, self.adaptation_config["exploration_factor"] * 2)
            
            adaptation_result = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "evaluation_idx": len(self.evaluation_results) - 1,
                "previous_parameters": previous_parameters,
                "new_parameters": new_parameters,
                "previous_performance": previous_performance,
                "new_performance": avg_metric_value,
                "improvement": improvement,
                "metric": metric_name
            }
        else:
            self.logger.info(
                f"Tham số mới không đủ tốt: {metric_name} = {avg_metric_value:.4f} "
                f"({improvement:.2f}% so với {previous_performance:.4f})"
            )
            
            # Tăng bộ đếm thất bại
            self.consecutive_failures += 1
            
            # Giảm mức độ khám phá nếu thất bại
            self.current_exploration = max(self.current_exploration * 0.9, self.adaptation_config["exploration_factor"] * 0.5)
            
            # Quay lại tham số trước đó nếu được cấu hình
            if self.adaptation_config["revert_failed_changes"]:
                self._apply_parameters(previous_parameters)
                self.logger.info("Đã quay lại tham số trước đó")
            
            adaptation_result = {
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "evaluation_idx": len(self.evaluation_results) - 1,
                "attempted_parameters": new_parameters,
                "current_parameters": self._get_current_parameters(),
                "previous_performance": previous_performance,
                "attempted_performance": avg_metric_value,
                "difference": improvement,
                "metric": metric_name
            }
        
        # Thêm vào lịch sử thích ứng
        self.adaptation_history.append(adaptation_result)
        
        # Thiết lập thời gian chờ
        self.cooldown_counter = self.adaptation_config["cooldown_period"]
        self.last_adaptation_time = datetime.now()
        
        # Lưu kết quả thích ứng
        self._save_adaptation_result(adaptation_result)
        
        # Kiểm tra số lần thất bại liên tiếp
        if self.consecutive_failures >= self.adaptation_config["max_consecutive_failures"]:
            self.logger.warning(
                f"Đã đạt số lần thất bại liên tiếp tối đa ({self.consecutive_failures}). "
                f"Quay lại tham số tốt nhất."
            )
            
            # Quay lại tham số tốt nhất
            self._apply_parameters(self.best_parameters)
            
            # Đặt lại mức độ khám phá
            self.current_exploration = self.adaptation_config["exploration_factor"]
            
            # Đặt lại bộ đếm thất bại
            self.consecutive_failures = 0
        
        return adaptation_result
    
    def detect_market_change(self, market_data: pd.DataFrame, window_size: int = 100) -> bool:
        """
        Phát hiện sự thay đổi đáng kể trong thị trường.
        
        Args:
            market_data: DataFrame chứa dữ liệu thị trường
            window_size: Kích thước cửa sổ để phát hiện thay đổi
            
        Returns:
            True nếu phát hiện thay đổi, False nếu không
        """
        if not self.adaptation_config["track_market_changes"]:
            return False
        
        if market_data is None or len(market_data) < window_size * 2:
            self.logger.warning("Không đủ dữ liệu để phát hiện thay đổi thị trường")
            return False
        
        try:
            # Lấy cửa sổ dữ liệu gần đây và trước đó
            recent_window = market_data.iloc[-window_size:]
            previous_window = market_data.iloc[-window_size*2:-window_size]
            
            # Tính các chỉ số thị trường
            # 1. Biến động (độ lệch chuẩn của lợi nhuận)
            if 'close' in market_data.columns:
                recent_returns = recent_window['close'].pct_change().dropna()
                previous_returns = previous_window['close'].pct_change().dropna()
                
                recent_volatility = recent_returns.std()
                previous_volatility = previous_returns.std()
                
                volatility_change = abs(recent_volatility - previous_volatility) / previous_volatility
                
                # 2. Xu hướng (hệ số góc của đường hồi quy tuyến tính)
                recent_trend = np.polyfit(range(len(recent_window)), recent_window['close'].values, 1)[0]
                previous_trend = np.polyfit(range(len(previous_window)), previous_window['close'].values, 1)[0]
                
                trend_change = abs(recent_trend - previous_trend) / (abs(previous_trend) + 1e-10)
                
                # 3. Khối lượng giao dịch (nếu có)
                volume_change = 0
                if 'volume' in market_data.columns:
                    recent_volume = recent_window['volume'].mean()
                    previous_volume = previous_window['volume'].mean()
                    volume_change = abs(recent_volume - previous_volume) / previous_volume
                
                # Tính tổng thay đổi (có trọng số)
                total_change = (0.5 * volatility_change + 0.3 * trend_change + 0.2 * volume_change)
                
                # So sánh với ngưỡng
                threshold = self.adaptation_config["market_change_threshold"]
                if total_change > threshold:
                    self.logger.info(
                        f"Đã phát hiện thay đổi thị trường: {total_change:.4f} > {threshold} "
                        f"(Volatility: {volatility_change:.4f}, Trend: {trend_change:.4f}, Volume: {volume_change:.4f})"
                    )
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Lỗi khi phát hiện thay đổi thị trường: {str(e)}")
            return False
    
    def adapt_to_market_change(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Thích ứng với sự thay đổi thị trường.
        
        Args:
            market_data: DataFrame chứa dữ liệu thị trường
            
        Returns:
            Dict chứa kết quả thích ứng
        """
        if not self.adaptation_config["adapt_on_market_change"]:
            return {"success": False, "message": "Thích ứng tự động với thay đổi thị trường đã bị tắt"}
        
        # Phát hiện thay đổi thị trường
        if not self.detect_market_change(market_data):
            return {"success": False, "message": "Không phát hiện thay đổi đáng kể trong thị trường"}
        
        self.logger.info("Bắt đầu thích ứng với thay đổi thị trường")
        
        # Bỏ qua thời gian chờ
        self.cooldown_counter = 0
        
        # Tăng mức độ khám phá tạm thời
        original_exploration = self.current_exploration
        self.current_exploration = min(self.current_exploration * 1.5, 1.0)
        
        # Thực hiện thích ứng
        result = self.adapt()
        
        # Khôi phục mức độ khám phá
        self.current_exploration = original_exploration
        
        # Thêm thông tin về thay đổi thị trường
        result["triggered_by"] = "market_change"
        
        return result
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """
        Lấy tham số hiện tại của agent.
        
        Returns:
            Dict các tham số hiện tại
        """
        if self.agent is None:
            return {}
        
        # Lấy các tham số của agent
        current_parameters = {}
        for param_name in self.adaptation_config["adaptable_parameters"].keys():
            if hasattr(self.agent, param_name):
                current_parameters[param_name] = copy.deepcopy(getattr(self.agent, param_name))
        
        # Nếu không có tham số nào được tìm thấy, thử lấy từ config
        if not current_parameters and hasattr(self.agent, "config"):
            for param_name in self.adaptation_config["adaptable_parameters"].keys():
                if param_name in self.agent.config:
                    current_parameters[param_name] = copy.deepcopy(self.agent.config[param_name])
        
        return current_parameters
    
    def _generate_new_parameters(self) -> Dict[str, Any]:
        """
        Tạo tham số mới để thử nghiệm.
        
        Returns:
            Dict các tham số mới
        """
        # Lấy tham số hiện tại
        current_parameters = self._get_current_parameters()
        if not current_parameters:
            self.logger.warning("Không tìm thấy tham số hiện tại, sử dụng tham số ban đầu")
            current_parameters = self.initial_parameters.copy()
        
        # Tạo tham số mới dựa trên tham số hiện tại
        new_parameters = copy.deepcopy(current_parameters)
        adaptable_params = self.adaptation_config["adaptable_parameters"]
        
        # Chỉ thay đổi một số tham số (không phải tất cả)
        num_params = len(adaptable_params)
        params_to_change = max(1, int(num_params * self.current_exploration))
        
        # Chọn ngẫu nhiên tham số để thay đổi
        param_names = list(adaptable_params.keys())
        np.random.shuffle(param_names)
        selected_params = param_names[:params_to_change]
        
        for param_name in selected_params:
            if param_name not in adaptable_params:
                continue
                
            param_config = adaptable_params[param_name]
            param_type = param_config.get("type", "float")
            
            # Thay đổi tham số dựa trên loại
            if param_type == "float":
                new_parameters[param_name] = self._generate_float_param(
                    current_value=current_parameters.get(param_name, 0.0),
                    param_config=param_config
                )
            elif param_type == "int":
                new_parameters[param_name] = self._generate_int_param(
                    current_value=current_parameters.get(param_name, 0),
                    param_config=param_config
                )
            elif param_type == "bool":
                # Đối với bool, đơn giản là đảo ngược giá trị
                if np.random.random() < self.current_exploration:
                    new_parameters[param_name] = not current_parameters.get(param_name, False)
            elif param_type == "list_int":
                # Đối với list_int, chọn một giá trị từ danh sách options
                if "options" in param_config:
                    new_parameters[param_name] = copy.deepcopy(
                        np.random.choice(param_config["options"])
                    )
        
        self.logger.debug(f"Đã tạo tham số mới: {new_parameters}")
        return new_parameters
    
    def _generate_float_param(self, current_value: float, param_config: Dict[str, Any]) -> float:
        """
        Tạo giá trị float mới dựa trên giá trị hiện tại và cấu hình.
        
        Args:
            current_value: Giá trị hiện tại
            param_config: Cấu hình tham số
            
        Returns:
            Giá trị float mới
        """
        min_val = param_config.get("min", 0.0)
        max_val = param_config.get("max", 1.0)
        scale = param_config.get("scale", "linear")
        
        # Mức độ thay đổi dựa trên exploration factor
        max_change = self.adaptation_config["max_param_change"] * self.current_exploration
        
        if scale == "log":
            # Thang logarit
            log_min = np.log10(min_val) if min_val > 0 else -5
            log_max = np.log10(max_val)
            log_current = np.log10(current_value) if current_value > 0 else log_min
            
            # Tạo giá trị mới với thay đổi tương đối
            log_range = log_max - log_min
            log_change = np.random.uniform(-max_change, max_change) * log_range
            log_new = log_current + log_change
            
            # Đảm bảo giá trị nằm trong khoảng
            log_new = max(log_min, min(log_max, log_new))
            
            # Chuyển về không gian gốc
            new_value = 10 ** log_new
        else:
            # Thang tuyến tính
            # Tạo giá trị mới với thay đổi tương đối
            value_range = max_val - min_val
            change = np.random.uniform(-max_change, max_change) * value_range
            new_value = current_value + change
            
            # Đảm bảo giá trị nằm trong khoảng
            new_value = max(min_val, min(max_val, new_value))
        
        return new_value
    
    def _generate_int_param(self, current_value: int, param_config: Dict[str, Any]) -> int:
        """
        Tạo giá trị int mới dựa trên giá trị hiện tại và cấu hình.
        
        Args:
            current_value: Giá trị hiện tại
            param_config: Cấu hình tham số
            
        Returns:
            Giá trị int mới
        """
        min_val = int(param_config.get("min", 1))
        max_val = int(param_config.get("max", 100))
        scale = param_config.get("scale", "linear")
        step = int(param_config.get("step", 1))
        
        # Mức độ thay đổi dựa trên exploration factor
        max_change = self.adaptation_config["max_param_change"] * self.current_exploration
        
        if scale == "log":
            # Thang logarit
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            log_current = np.log10(current_value) if current_value > 0 else log_min
            
            # Tạo giá trị mới với thay đổi tương đối
            log_range = log_max - log_min
            log_change = np.random.uniform(-max_change, max_change) * log_range
            log_new = log_current + log_change
            
            # Đảm bảo giá trị nằm trong khoảng
            log_new = max(log_min, min(log_max, log_new))
            
            # Chuyển về không gian gốc và làm tròn thành số nguyên
            new_value = int(10 ** log_new)
        else:
            # Thang tuyến tính
            # Tạo giá trị mới với thay đổi tương đối
            value_range = max_val - min_val
            change = int(np.random.uniform(-max_change, max_change) * value_range)
            new_value = current_value + change
            
            # Đảm bảo giá trị nằm trong khoảng
            new_value = max(min_val, min(max_val, new_value))
        
        # Làm tròn theo step
        new_value = round(new_value / step) * step
        
        return new_value
    
    def _apply_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Áp dụng tham số cho agent.
        
        Args:
            parameters: Dict các tham số cần áp dụng
        """
        if self.agent is None or not parameters:
            return
        
        for param_name, param_value in parameters.items():
            if hasattr(self.agent, param_name):
                # Thiết lập trực tiếp thuộc tính
                setattr(self.agent, param_name, copy.deepcopy(param_value))
                self.logger.debug(f"Đã thiết lập {param_name} = {param_value}")
            elif hasattr(self.agent, "config") and param_name in self.agent.config:
                # Cập nhật trong config
                self.agent.config[param_name] = copy.deepcopy(param_value)
                self.logger.debug(f"Đã cập nhật config[{param_name}] = {param_value}")
            else:
                self.logger.warning(f"Không thể áp dụng tham số {param_name}: Thuộc tính không tồn tại")
    
    def _save_adaptation_result(self, result: Dict[str, Any]) -> None:
        """
        Lưu kết quả thích ứng vào file.
        
        Args:
            result: Dict chứa kết quả thích ứng
        """
        # Tạo thư mục results nếu chưa tồn tại
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Tạo tên file dựa trên timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = results_dir / f"adaptation_{timestamp}.json"
        
        # Lưu kết quả vào file JSON
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False, default=str)
            
            self.logger.debug(f"Đã lưu kết quả thích ứng vào {file_path}")
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu kết quả thích ứng: {str(e)}")
    
    def save_state(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """
        Lưu trạng thái của module thích ứng.
        
        Args:
            file_path: Đường dẫn file lưu trạng thái (None để tạo tự động)
            
        Returns:
            Đường dẫn file đã lưu
        """
        # Tạo đường dẫn mặc định nếu không được cung cấp
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.output_dir / f"adaptation_state_{timestamp}.json"
        else:
            file_path = Path(file_path)
        
        # Đảm bảo thư mục tồn tại
        file_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Tạo dict trạng thái
        state = {
            "timestamp": datetime.now().isoformat(),
            "adaptation_config": self.adaptation_config,
            "initial_parameters": self.initial_parameters,
            "best_parameters": self.best_parameters,
            "best_performance": self.best_performance,
            "current_exploration": self.current_exploration,
            "consecutive_failures": self.consecutive_failures,
            "cooldown_counter": self.cooldown_counter,
            "last_adaptation_time": self.last_adaptation_time.isoformat() if self.last_adaptation_time else None,
            "adaptation_history": self.adaptation_history,
            "agent_type": self.agent.__class__.__name__ if self.agent else None
        }
        
        # Lưu vào file JSON
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4, ensure_ascii=False, default=str)
            
            self.logger.info(f"Đã lưu trạng thái module thích ứng vào {file_path}")
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu trạng thái: {str(e)}")
            return ""
    
    def load_state(self, file_path: Union[str, Path]) -> bool:
        """
        Tải trạng thái của module thích ứng.
        
        Args:
            file_path: Đường dẫn file trạng thái
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f"File trạng thái không tồn tại: {file_path}")
            return False
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            # Cập nhật các thuộc tính từ trạng thái
            self.adaptation_config = state.get("adaptation_config", self.adaptation_config)
            self.initial_parameters = state.get("initial_parameters", {})
            self.best_parameters = state.get("best_parameters", {})
            self.best_performance = state.get("best_performance")
            self.current_exploration = state.get("current_exploration", self.adaptation_config["exploration_factor"])
            self.consecutive_failures = state.get("consecutive_failures", 0)
            self.cooldown_counter = state.get("cooldown_counter", 0)
            
            # Chuyển đổi timestamp thành datetime
            last_adaptation_time = state.get("last_adaptation_time")
            if last_adaptation_time:
                self.last_adaptation_time = datetime.fromisoformat(last_adaptation_time)
            else:
                self.last_adaptation_time = None
            
            self.adaptation_history = state.get("adaptation_history", [])
            
            # Kiểm tra loại agent
            agent_type = state.get("agent_type")
            if self.agent and agent_type and self.agent.__class__.__name__ != agent_type:
                self.logger.warning(
                    f"Loại agent hiện tại ({self.agent.__class__.__name__}) khác với "
                    f"loại agent trong trạng thái ({agent_type})"
                )
            
            self.logger.info(f"Đã tải trạng thái module thích ứng từ {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải trạng thái: {str(e)}")
            return False
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Lấy tham số tốt nhất đã tìm được.
        
        Returns:
            Dict các tham số tốt nhất
        """
        return self.best_parameters.copy()
    
    def apply_best_parameters(self) -> bool:
        """
        Áp dụng tham số tốt nhất cho agent.
        
        Returns:
            True nếu áp dụng thành công, False nếu không
        """
        if not self.best_parameters:
            self.logger.warning("Không có tham số tốt nhất để áp dụng")
            return False
        
        self._apply_parameters(self.best_parameters)
        self.logger.info("Đã áp dụng tham số tốt nhất cho agent")
        return True
    
    def reset_to_initial_parameters(self) -> bool:
        """
        Đặt lại tham số agent về trạng thái ban đầu.
        
        Returns:
            True nếu đặt lại thành công, False nếu không
        """
        if not self.initial_parameters:
            self.logger.warning("Không có tham số ban đầu để đặt lại")
            return False
        
        self._apply_parameters(self.initial_parameters)
        self.logger.info("Đã đặt lại tham số agent về trạng thái ban đầu")
        return True
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """
        Lấy tóm tắt về quá trình thích ứng.
        
        Returns:
            Dict chứa tóm tắt thích ứng
        """
        # Tính số lần thích ứng thành công/thất bại
        successful_adaptations = sum(1 for a in self.adaptation_history if a.get("success", False))
        
        # Tạo tóm tắt
        summary = {
            "total_adaptations": len(self.adaptation_history),
            "successful_adaptations": successful_adaptations,
            "failed_adaptations": len(self.adaptation_history) - successful_adaptations,
            "best_performance": self.best_performance,
            "initial_parameters": self.initial_parameters,
            "best_parameters": self.best_parameters,
            "current_parameters": self._get_current_parameters(),
            "current_exploration": self.current_exploration,
            "consecutive_failures": self.consecutive_failures,
            "last_adaptation_time": self.last_adaptation_time.isoformat() if self.last_adaptation_time else None
        }
        
        # Tính cải thiện tổng thể
        if self.best_performance is not None and self.initial_parameters:
            # Tìm hiệu suất ban đầu
            initial_performance = None
            for result in self.evaluation_results:
                eval_result = result.get("result", {})
                metric_name = self.adaptation_config["optimization_goal"]
                metric_value = self._get_primary_metric_value(eval_result)
                
                if initial_performance is None:
                    initial_performance = metric_value
                    break
            
            if initial_performance is not None and initial_performance != 0:
                improvement = (self.best_performance - initial_performance) / abs(initial_performance) * 100
                summary["overall_improvement"] = improvement
                summary["initial_performance"] = initial_performance
        
        return summary
    
    def generate_adaptation_report(self, output_format: str = "markdown") -> str:
        """
        Tạo báo cáo về quá trình thích ứng.
        
        Args:
            output_format: Định dạng đầu ra ("markdown", "html", "json")
            
        Returns:
            Nội dung báo cáo
        """
        # Lấy tóm tắt thích ứng
        summary = self.get_adaptation_summary()
        
        if output_format == "json":
            return json.dumps(summary, indent=4, ensure_ascii=False, default=str)
        
        elif output_format == "html":
            # Tạo báo cáo HTML
            html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo thích ứng Agent</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .success {{
            color: #27ae60;
        }}
        .failure {{
            color: #e74c3c;
        }}
        .improvement {{
            color: #2980b9;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>Báo cáo thích ứng Agent</h1>
    <p><strong>Thời gian báo cáo:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Tóm tắt thích ứng</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Giá trị</th>
        </tr>
        <tr>
            <td>Tổng số lần thích ứng</td>
            <td>{summary.get('total_adaptations', 0)}</td>
        </tr>
        <tr>
            <td>Số lần thành công</td>
            <td class="success">{summary.get('successful_adaptations', 0)}</td>
        </tr>
        <tr>
            <td>Số lần thất bại</td>
            <td class="failure">{summary.get('failed_adaptations', 0)}</td>
        </tr>"""
            
            # Thêm thông tin cải thiện nếu có
            if 'overall_improvement' in summary:
                html += f"""
        <tr>
            <td>Cải thiện tổng thể</td>
            <td class="improvement">{summary['overall_improvement']:.2f}%</td>
        </tr>
        <tr>
            <td>Hiệu suất ban đầu</td>
            <td>{summary['initial_performance']:.4f}</td>
        </tr>"""
                
            html += f"""
        <tr>
            <td>Hiệu suất tốt nhất</td>
            <td>{summary.get('best_performance', 'N/A')}</td>
        </tr>
        <tr>
            <td>Thích ứng gần nhất</td>
            <td>{summary.get('last_adaptation_time', 'N/A')}</td>
        </tr>
    </table>
    
    <h2>Tham số tốt nhất</h2>
    <table>
        <tr>
            <th>Tham số</th>
            <th>Giá trị ban đầu</th>
            <th>Giá trị tốt nhất</th>
            <th>Thay đổi</th>
        </tr>"""
            
            # Thêm thông tin tham số
            for param_name in summary.get('best_parameters', {}):
                initial_value = summary.get('initial_parameters', {}).get(param_name, 'N/A')
                best_value = summary.get('best_parameters', {}).get(param_name, 'N/A')
                
                # Tính phần trăm thay đổi nếu có thể
                if isinstance(initial_value, (int, float)) and isinstance(best_value, (int, float)) and initial_value != 0:
                    change = (best_value - initial_value) / abs(initial_value) * 100
                    change_text = f"{change:.2f}%"
                else:
                    change_text = "N/A"
                
                html += f"""
        <tr>
            <td>{param_name}</td>
            <td>{initial_value}</td>
            <td>{best_value}</td>
            <td>{change_text}</td>
        </tr>"""
            
            html += """
    </table>
    
    <h2>Lịch sử thích ứng</h2>
    <table>
        <tr>
            <th>Thời gian</th>
            <th>Kết quả</th>
            <th>Metric</th>
            <th>Giá trị trước</th>
            <th>Giá trị sau</th>
            <th>Cải thiện</th>
        </tr>"""
            
            # Thêm lịch sử thích ứng
            for adaptation in self.adaptation_history:
                timestamp = adaptation.get('timestamp', 'N/A')
                success = adaptation.get('success', False)
                metric = adaptation.get('metric', 'N/A')
                
                if success:
                    prev_perf = adaptation.get('previous_performance', 0)
                    new_perf = adaptation.get('new_performance', 0)
                    improvement = adaptation.get('improvement', 0)
                    
                    html += f"""
        <tr>
            <td>{timestamp}</td>
            <td class="success">Thành công</td>
            <td>{metric}</td>
            <td>{prev_perf:.4f}</td>
            <td>{new_perf:.4f}</td>
            <td class="improvement">{improvement:.2f}%</td>
        </tr>"""
                else:
                    prev_perf = adaptation.get('previous_performance', 0)
                    attempted_perf = adaptation.get('attempted_performance', 0)
                    difference = adaptation.get('difference', 0)
                    
                    html += f"""
        <tr>
            <td>{timestamp}</td>
            <td class="failure">Thất bại</td>
            <td>{metric}</td>
            <td>{prev_perf:.4f}</td>
            <td>{attempted_perf:.4f}</td>
            <td class="failure">{difference:.2f}%</td>
        </tr>"""
            
            html += """
    </table>
</body>
</html>"""
            
            return html
            
        else:  # markdown
            # Tạo báo cáo Markdown
            report = f"# Báo cáo thích ứng Agent\n\n"
            report += f"**Thời gian báo cáo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            report += "## Tóm tắt thích ứng\n\n"
            report += f"- **Tổng số lần thích ứng:** {summary.get('total_adaptations', 0)}\n"
            report += f"- **Số lần thành công:** {summary.get('successful_adaptations', 0)}\n"
            report += f"- **Số lần thất bại:** {summary.get('failed_adaptations', 0)}\n"
            
            # Thêm thông tin cải thiện nếu có
            if 'overall_improvement' in summary:
                report += f"- **Cải thiện tổng thể:** {summary['overall_improvement']:.2f}%\n"
                report += f"- **Hiệu suất ban đầu:** {summary['initial_performance']:.4f}\n"
                
            report += f"- **Hiệu suất tốt nhất:** {summary.get('best_performance', 'N/A')}\n"
            report += f"- **Thích ứng gần nhất:** {summary.get('last_adaptation_time', 'N/A')}\n\n"
            
            report += "## Tham số tốt nhất\n\n"
            report += "| Tham số | Giá trị ban đầu | Giá trị tốt nhất | Thay đổi |\n"
            report += "|---------|----------------|-----------------|----------|\n"
            
            # Thêm thông tin tham số
            for param_name in summary.get('best_parameters', {}):
                initial_value = summary.get('initial_parameters', {}).get(param_name, 'N/A')
                best_value = summary.get('best_parameters', {}).get(param_name, 'N/A')
                
                # Tính phần trăm thay đổi nếu có thể
                if isinstance(initial_value, (int, float)) and isinstance(best_value, (int, float)) and initial_value != 0:
                    change = (best_value - initial_value) / abs(initial_value) * 100
                    change_text = f"{change:.2f}%"
                else:
                    change_text = "N/A"
                
                report += f"| {param_name} | {initial_value} | {best_value} | {change_text} |\n"
            
            report += "\n## Lịch sử thích ứng\n\n"
            report += "| Thời gian | Kết quả | Metric | Giá trị trước | Giá trị sau | Cải thiện |\n"
            report += "|-----------|---------|--------|--------------|-------------|------------|\n"
            
            # Thêm lịch sử thích ứng
            for adaptation in self.adaptation_history:
                timestamp = adaptation.get('timestamp', 'N/A')
                success = adaptation.get('success', False)
                metric = adaptation.get('metric', 'N/A')
                
                if success:
                    prev_perf = adaptation.get('previous_performance', 0)
                    new_perf = adaptation.get('new_performance', 0)
                    improvement = adaptation.get('improvement', 0)
                    
                    report += f"| {timestamp} | Thành công | {metric} | {prev_perf:.4f} | {new_perf:.4f} | {improvement:.2f}% |\n"
                else:
                    prev_perf = adaptation.get('previous_performance', 0)
                    attempted_perf = adaptation.get('attempted_performance', 0)
                    difference = adaptation.get('difference', 0)
                    
                    report += f"| {timestamp} | Thất bại | {metric} | {prev_perf:.4f} | {attempted_perf:.4f} | {difference:.2f}% |\n"
            
            return report