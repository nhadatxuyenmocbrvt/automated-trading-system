"""
Meta-Learning cho hệ thống multi-agent.
File này định nghĩa lớp MetaLearner để cải thiện hiệu suất của các agent thông qua
các kỹ thuật meta-learning, phân tích và tổng hợp kiến thức từ nhiều agent.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import json
import random
import copy
from collections import defaultdict

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
from models.training_pipeline.hyperparameter_tuner import HyperparameterTuner

class MetaLearner:
    """
    Lớp Meta-Learner để cải thiện hiệu suất agent thông qua meta-learning.
    
    Cung cấp các phương thức để:
    1. Phân tích hiệu suất của nhiều agent với các cấu hình khác nhau
    2. Học các mẫu thành công từ các agent khác nhau
    3. Tạo các cấu hình agent mới dựa trên kinh nghiệm
    4. Kết hợp các điểm mạnh từ nhiều agent
    5. Thích ứng với điều kiện thị trường thay đổi
    """
    
    def __init__(
        self,
        agent_evaluator: Optional[AgentEvaluator] = None,
        hyperparameter_tuner: Optional[HyperparameterTuner] = None,
        agents_repository: Optional[Dict[str, BaseAgent]] = None,
        performance_history: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        market_condition_analyzer: Optional[Any] = None,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo Meta-Learner.
        
        Args:
            agent_evaluator: Đối tượng đánh giá agent
            hyperparameter_tuner: Bộ điều chỉnh siêu tham số
            agents_repository: Kho lưu trữ các agent (ID -> Agent)
            performance_history: Lịch sử hiệu suất của các agent
            market_condition_analyzer: Bộ phân tích điều kiện thị trường
            output_dir: Thư mục đầu ra để lưu kết quả
            config: Cấu hình meta-learner
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("meta_learner")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập các thành phần
        self.agent_evaluator = agent_evaluator or AgentEvaluator(logger=self.logger)
        
        # Hyperparameter tuner (nếu có)
        self.hyperparameter_tuner = hyperparameter_tuner
        
        # Kho lưu trữ agent và hiệu suất
        self.agents_repository = agents_repository or {}
        self.performance_history = performance_history or {}
        
        # Bộ phân tích điều kiện thị trường (nếu có)
        self.market_condition_analyzer = market_condition_analyzer
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            self.output_dir = Path(self.system_config.get("model_dir", "./models")) / "meta_learner"
        else:
            self.output_dir = Path(output_dir)
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập cấu hình mặc định
        self.default_config = {
            "meta_batch_size": 5,  # Số lượng agent sử dụng cho mỗi lần học
            "adaptation_rate": 0.2,  # Tốc độ thích ứng với môi trường mới
            "innovation_rate": 0.1,  # Tốc độ đổi mới cấu hình
            "crossover_probability": 0.7,  # Xác suất kết hợp các cấu hình
            "mutation_probability": 0.3,  # Xác suất đột biến cấu hình
            "elitism_ratio": 0.2,  # Tỉ lệ giữ lại cấu hình tốt nhất
            "generation_size": 10,  # Số lượng cấu hình mới mỗi thế hệ
            "max_generations": 5,  # Số thế hệ tối đa
            "performance_metrics": [
                "mean_reward",
                "sharpe_ratio",
                "win_rate",
                "max_drawdown"
            ],
            "metric_weights": {
                "mean_reward": 1.0,
                "sharpe_ratio": 0.8,
                "win_rate": 0.7,
                "max_drawdown": 0.6
            },
            "early_stopping": True,
            "early_stopping_patience": 2,
            "save_best_config": True,
            "ensemble_size": 3,  # Số lượng agent trong ensemble
            "market_condition_features": [
                "volatility",
                "trend_strength",
                "volume_profile"
            ]
        }
        
        # Cập nhật config từ tham số
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # Danh sách các cấu hình đã tạo và hiệu suất tương ứng
        self.generated_configs = []
        self.config_performance = {}
        
        # Mapping điều kiện thị trường -> cấu hình phù hợp
        self.market_condition_configs = defaultdict(list)
        
        # State hiện tại
        self.current_generation = 0
        self.best_config = None
        self.best_performance = None
        
        self.logger.info("Đã khởi tạo MetaLearner")
    
    def register_agent(self, agent: BaseAgent, agent_id: Optional[str] = None, 
                      record_config: bool = True) -> str:
        """
        Đăng ký một agent mới vào kho lưu trữ.
        
        Args:
            agent: Agent cần đăng ký
            agent_id: ID cho agent (None để tạo tự động)
            record_config: Ghi lại cấu hình của agent
            
        Returns:
            ID của agent đã đăng ký
        """
        if agent_id is None:
            agent_id = f"agent_{len(self.agents_repository) + 1}_{agent.__class__.__name__}"
        
        # Kiểm tra xem agent_id đã tồn tại chưa
        if agent_id in self.agents_repository:
            self.logger.warning(f"Agent ID '{agent_id}' đã tồn tại. Thêm hậu tố để tránh trùng lặp.")
            agent_id = f"{agent_id}_{datetime.now().strftime('%H%M%S')}"
        
        # Lưu agent vào kho
        self.agents_repository[agent_id] = agent
        
        # Khởi tạo lịch sử hiệu suất nếu cần
        if agent_id not in self.performance_history:
            self.performance_history[agent_id] = []
        
        # Ghi lại cấu hình của agent nếu được yêu cầu
        if record_config and hasattr(agent, 'config'):
            config = copy.deepcopy(agent.config)
            config["agent_type"] = agent.__class__.__name__
            
            self.generated_configs.append({
                "agent_id": agent_id,
                "config": config,
                "created_at": datetime.now().isoformat(),
                "generation": self.current_generation
            })
        
        # Đăng ký agent với agent_evaluator
        if self.agent_evaluator:
            self.agent_evaluator.register_agent(agent, agent_id)
        
        self.logger.info(f"Đã đăng ký agent '{agent_id}' ({agent.__class__.__name__}) vào kho lưu trữ")
        
        return agent_id
    
    def record_performance(
        self, 
        agent_id: str, 
        performance_data: Dict[str, Any],
        market_condition: Optional[str] = None
    ) -> None:
        """
        Ghi lại hiệu suất của một agent.
        
        Args:
            agent_id: ID của agent
            performance_data: Dữ liệu hiệu suất
            market_condition: Điều kiện thị trường (nếu có)
        """
        if agent_id not in self.agents_repository:
            self.logger.warning(f"Agent ID '{agent_id}' không tồn tại trong kho lưu trữ")
            return
        
        # Thêm timestamp và market_condition
        performance_data["timestamp"] = datetime.now().isoformat()
        if market_condition:
            performance_data["market_condition"] = market_condition
        
        # Ghi lại dữ liệu hiệu suất
        if agent_id not in self.performance_history:
            self.performance_history[agent_id] = []
        
        self.performance_history[agent_id].append(performance_data)
        
        # Cập nhật hiệu suất cấu hình nếu agent có trong danh sách cấu hình đã tạo
        for config_info in self.generated_configs:
            if config_info["agent_id"] == agent_id:
                config_id = self._get_config_hash(config_info["config"])
                
                if config_id not in self.config_performance:
                    self.config_performance[config_id] = []
                
                self.config_performance[config_id].append(performance_data)
                
                # Nếu có thông tin điều kiện thị trường, cập nhật mapping
                if market_condition:
                    self.market_condition_configs[market_condition].append(config_info["config"])
                
                break
        
        self.logger.debug(f"Đã ghi lại hiệu suất mới cho agent '{agent_id}'")
    
    def analyze_agent_configurations(
        self,
        metric_name: Optional[str] = None,
        top_n: int = 5,
        min_evaluations: int = 3
    ) -> Dict[str, Any]:
        """
        Phân tích cấu hình của các agent để tìm ra các mẫu thành công.
        
        Args:
            metric_name: Tên metric để đánh giá (None để sử dụng metric tổng hợp)
            top_n: Số lượng cấu hình hàng đầu để phân tích
            min_evaluations: Số lượng đánh giá tối thiểu để xem xét
            
        Returns:
            Dict chứa kết quả phân tích
        """
        # Metric mặc định nếu không được chỉ định
        if metric_name is None:
            metric_name = "mean_reward"
        
        # Thu thập dữ liệu hiệu suất của tất cả agent
        all_agent_performances = []
        
        for agent_id, performance_list in self.performance_history.items():
            if len(performance_list) < min_evaluations:
                continue
                
            # Lấy agent
            agent = self.agents_repository.get(agent_id)
            if agent is None or not hasattr(agent, 'config'):
                continue
                
            # Tính hiệu suất trung bình theo metric
            metric_values = []
            for perf in performance_list:
                # Tìm giá trị metric trong hiệu suất
                if metric_name in perf:
                    metric_values.append(perf[metric_name])
                elif "basic_metrics" in perf and metric_name in perf["basic_metrics"]:
                    metric_values.append(perf["basic_metrics"][metric_name])
                elif "advanced_metrics" in perf and metric_name in perf["advanced_metrics"]:
                    metric_values.append(perf["advanced_metrics"][metric_name])
            
            if not metric_values:
                continue
                
            avg_metric = np.mean(metric_values)
            
            # Thêm vào danh sách với cấu hình
            all_agent_performances.append({
                "agent_id": agent_id,
                "agent_type": agent.__class__.__name__,
                "config": copy.deepcopy(agent.config),
                "avg_performance": avg_metric,
                "num_evaluations": len(metric_values)
            })
        
        # Sắp xếp theo hiệu suất
        all_agent_performances.sort(key=lambda x: x["avg_performance"], reverse=True)
        
        # Lấy các cấu hình top_n
        top_configs = all_agent_performances[:top_n]
        
        # Phân tích các tham số phổ biến trong top_n cấu hình
        common_params = self._extract_common_parameters(top_configs)
        
        # Phân tích tác động của các siêu tham số đến hiệu suất
        param_impacts = self._analyze_parameter_impacts(all_agent_performances)
        
        # Phân tích theo loại agent
        agent_type_performance = {}
        for agent_perf in all_agent_performances:
            agent_type = agent_perf["agent_type"]
            if agent_type not in agent_type_performance:
                agent_type_performance[agent_type] = []
            agent_type_performance[agent_type].append(agent_perf["avg_performance"])
        
        # Tính hiệu suất trung bình cho mỗi loại agent
        agent_type_avg = {
            agent_type: np.mean(performances)
            for agent_type, performances in agent_type_performance.items()
            if performances
        }
        
        # Tạo kết quả
        result = {
            "metric": metric_name,
            "top_configs": top_configs,
            "common_parameters": common_params,
            "parameter_impacts": param_impacts,
            "agent_type_performance": agent_type_avg,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Đã phân tích {len(all_agent_performances)} cấu hình agent theo metric '{metric_name}'")
        
        return result
    
    def generate_new_configurations(
        self,
        base_agent_ids: Optional[List[str]] = None,
        num_configs: int = 5,
        generation_method: str = "evolutionary",
        target_market_condition: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Tạo các cấu hình mới dựa trên kinh nghiệm từ các agent hiện có.
        
        Args:
            base_agent_ids: Danh sách ID các agent làm cơ sở (None để sử dụng tất cả)
            num_configs: Số lượng cấu hình mới cần tạo
            generation_method: Phương pháp tạo ("evolutionary", "sampling", "gradient")
            target_market_condition: Điều kiện thị trường mục tiêu
            
        Returns:
            Danh sách các cấu hình mới
        """
        # Tăng thế hệ
        self.current_generation += 1
        
        # Xác định các agent base
        if base_agent_ids is None:
            base_agent_ids = list(self.agents_repository.keys())
        
        base_agents = []
        for agent_id in base_agent_ids:
            if agent_id in self.agents_repository:
                agent = self.agents_repository[agent_id]
                if hasattr(agent, 'config'):
                    base_agents.append((agent_id, agent))
        
        if not base_agents:
            self.logger.warning("Không có agent nào để tạo cấu hình mới")
            return []
        
        # Lấy phân tích cấu hình nếu có nhiều hơn 2 agent
        analysis = None
        if len(base_agents) >= 2:
            analysis = self.analyze_agent_configurations()
        
        # Chọn phương pháp tạo
        if generation_method == "evolutionary":
            new_configs = self._generate_evolutionary_configs(
                base_agents, analysis, num_configs, target_market_condition
            )
        elif generation_method == "sampling":
            new_configs = self._generate_sampled_configs(
                base_agents, analysis, num_configs, target_market_condition
            )
        elif generation_method == "gradient":
            new_configs = self._generate_gradient_configs(
                base_agents, analysis, num_configs, target_market_condition
            )
        else:
            self.logger.warning(f"Phương pháp tạo '{generation_method}' không được hỗ trợ. Sử dụng 'evolutionary'")
            new_configs = self._generate_evolutionary_configs(
                base_agents, analysis, num_configs, target_market_condition
            )
        
        # Lưu cấu hình mới vào danh sách
        for config in new_configs:
            self.generated_configs.append({
                "agent_id": config.get("agent_id", f"gen_{self.current_generation}_{len(self.generated_configs)}"),
                "config": config,
                "created_at": datetime.now().isoformat(),
                "generation": self.current_generation,
                "parent_ids": config.get("parent_ids", []),
                "generation_method": generation_method,
                "target_market_condition": target_market_condition
            })
        
        self.logger.info(f"Đã tạo {len(new_configs)} cấu hình mới bằng phương pháp '{generation_method}'")
        
        return new_configs
    
    def create_agent_from_config(
        self,
        config: Dict[str, Any],
        env: Optional[BaseEnvironment] = None,
        agent_id: Optional[str] = None
    ) -> Tuple[str, BaseAgent]:
        """
        Tạo một agent mới từ cấu hình.
        
        Args:
            config: Cấu hình agent
            env: Môi trường sử dụng cho agent (nếu cần)
            agent_id: ID cho agent mới (None để tạo tự động)
            
        Returns:
            Tuple (agent_id, agent)
        """
        # Xác định loại agent
        agent_type = config.get("agent_type")
        if agent_type is None:
            # Thử đoán loại agent từ cấu hình
            if "double_dqn" in config or "dueling" in config:
                agent_type = "DQNAgent"
            elif "clip_range" in config or "vf_coef" in config:
                agent_type = "PPOAgent"
            elif "actor_lr" in config or "critic_lr" in config:
                agent_type = "A2CAgent"
            else:
                # Mặc định là DQNAgent
                agent_type = "DQNAgent"
                self.logger.warning("Không xác định được loại agent, sử dụng mặc định là DQNAgent")
        
        # Lấy các tham số cần thiết
        state_dim = config.get("state_dim")
        action_dim = config.get("action_dim")
        
        # Nếu không có state_dim hoặc action_dim trong config nhưng có env
        if (state_dim is None or action_dim is None) and env is not None:
            if hasattr(env, "observation_space") and hasattr(env.observation_space, "shape"):
                state_dim = env.observation_space.shape
            
            if hasattr(env, "action_space"):
                if hasattr(env.action_space, "n"):
                    action_dim = env.action_space.n
                elif hasattr(env.action_space, "shape"):
                    action_dim = env.action_space.shape[0]
        
        if state_dim is None or action_dim is None:
            self.logger.error("Không thể tạo agent: thiếu state_dim hoặc action_dim trong cấu hình")
            return None, None
        
        # Tạo agent mới
        try:
            if agent_type == "DQNAgent":
                agent = DQNAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    env=env,
                    **{k: v for k, v in config.items() if k not in ["agent_type", "state_dim", "action_dim"]}
                )
            elif agent_type == "PPOAgent":
                agent = PPOAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    env=env,
                    **{k: v for k, v in config.items() if k not in ["agent_type", "state_dim", "action_dim"]}
                )
            elif agent_type == "A2CAgent":
                agent = A2CAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    env=env,
                    **{k: v for k, v in config.items() if k not in ["agent_type", "state_dim", "action_dim"]}
                )
            else:
                self.logger.error(f"Loại agent không được hỗ trợ: {agent_type}")
                return None, None
                
            # Đăng ký agent mới
            registered_id = self.register_agent(agent, agent_id)
            
            return registered_id, agent
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo agent từ cấu hình: {str(e)}")
            return None, None
    
    def create_ensemble_agent(
        self,
        agent_ids: List[str],
        aggregation_method: str = "weighted_average",
        weights: Optional[List[float]] = None,
        ensemble_id: Optional[str] = None
    ) -> Tuple[str, Any]:
        """
        Tạo một ensemble agent kết hợp từ nhiều agent.
        
        Args:
            agent_ids: Danh sách ID các agent thành phần
            aggregation_method: Phương pháp kết hợp kết quả
            weights: Trọng số cho từng agent (nếu sử dụng weighted_average)
            ensemble_id: ID cho ensemble agent (None để tạo tự động)
            
        Returns:
            Tuple (ensemble_id, ensemble_agent)
        """
        # Kiểm tra xem các agent có tồn tại không
        agents = []
        for agent_id in agent_ids:
            if agent_id in self.agents_repository:
                agents.append(self.agents_repository[agent_id])
            else:
                self.logger.warning(f"Agent ID '{agent_id}' không tồn tại trong kho lưu trữ")
        
        if not agents:
            self.logger.error("Không có agent nào để tạo ensemble")
            return None, None
        
        # Nếu không có weights, tạo weights đồng đều
        if weights is None:
            weights = [1.0 / len(agents)] * len(agents)
        elif len(weights) != len(agents):
            self.logger.warning("Số lượng trọng số không khớp với số lượng agent. Sử dụng trọng số đồng đều")
            weights = [1.0 / len(agents)] * len(agents)
        
        # Chuẩn hóa weights
        weights_sum = sum(weights)
        weights = [w / weights_sum for w in weights]
        
        # Tạo ensemble agent
        from agent_manager.ensemble_agent import EnsembleAgent
        
        ensemble_agent = EnsembleAgent(
            agents=agents,
            weights=weights,
            aggregation_method=aggregation_method,
            name=f"ensemble_{len(agents)}_agents"
        )
        
        # Đăng ký ensemble agent
        if ensemble_id is None:
            ensemble_id = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Đăng ký với meta_learner
        self.register_agent(ensemble_agent, ensemble_id)
        
        self.logger.info(f"Đã tạo ensemble agent '{ensemble_id}' từ {len(agents)} agents")
        
        return ensemble_id, ensemble_agent
    
    def adapt_to_market_condition(
        self,
        market_condition: str,
        base_agent_id: str,
        adaptation_rate: Optional[float] = None
    ) -> Tuple[str, BaseAgent]:
        """
        Thích ứng một agent với điều kiện thị trường cụ thể.
        
        Args:
            market_condition: Điều kiện thị trường cần thích ứng
            base_agent_id: ID của agent cơ sở để thích ứng
            adaptation_rate: Tốc độ thích ứng (None để sử dụng cấu hình mặc định)
            
        Returns:
            Tuple (adapted_agent_id, adapted_agent)
        """
        # Sử dụng adaptation_rate từ config nếu không được cung cấp
        if adaptation_rate is None:
            adaptation_rate = self.config["adaptation_rate"]
        
        # Kiểm tra xem agent cơ sở có tồn tại không
        if base_agent_id not in self.agents_repository:
            self.logger.error(f"Agent ID '{base_agent_id}' không tồn tại trong kho lưu trữ")
            return None, None
        
        base_agent = self.agents_repository[base_agent_id]
        
        # Kiểm tra xem đã từng phân tích điều kiện thị trường này chưa
        if market_condition not in self.market_condition_configs or not self.market_condition_configs[market_condition]:
            # Nếu chưa có thông tin, tạo một cấu hình mới dựa trên agent cơ sở
            self.logger.warning(f"Không có thông tin về điều kiện thị trường '{market_condition}'. Tạo cấu hình mới")
            
            # Tạo cấu hình mới
            new_configs = self.generate_new_configurations(
                base_agent_ids=[base_agent_id],
                num_configs=1,
                target_market_condition=market_condition
            )
            
            if not new_configs:
                self.logger.error("Không thể tạo cấu hình mới")
                return None, None
                
            # Tạo agent mới từ cấu hình
            adapted_agent_id, adapted_agent = self.create_agent_from_config(
                new_configs[0],
                env=getattr(base_agent, 'env', None),
                agent_id=f"{base_agent_id}_adapted_{market_condition}"
            )
            
            return adapted_agent_id, adapted_agent
        
        # Lấy các cấu hình tốt nhất cho điều kiện thị trường
        suitable_configs = self.market_condition_configs[market_condition]
        
        # Tạo cấu hình thích ứng bằng cách kết hợp cấu hình cơ sở với cấu hình phù hợp
        base_config = copy.deepcopy(base_agent.config) if hasattr(base_agent, 'config') else {}
        
        # Lấy cấu hình ngẫu nhiên từ các cấu hình phù hợp
        selected_config = random.choice(suitable_configs)
        
        # Kết hợp các cấu hình
        adapted_config = {}
        
        # Sao chép các tham số từ base_config
        for key, value in base_config.items():
            adapted_config[key] = value
        
        # Cập nhật một số tham số từ selected_config dựa trên adaptation_rate
        for key, value in selected_config.items():
            if key in base_config:
                # Nếu là tham số số học, thực hiện interpolate
                if isinstance(value, (int, float)) and isinstance(base_config[key], (int, float)):
                    adapted_config[key] = base_config[key] * (1 - adaptation_rate) + value * adaptation_rate
                    
                    # Đảm bảo giữ đúng kiểu
                    if isinstance(base_config[key], int):
                        adapted_config[key] = int(adapted_config[key])
                        
                # Đối với các tham số khác, có xác suất adaptation_rate để lấy giá trị từ selected_config
                elif random.random() < adaptation_rate:
                    adapted_config[key] = value
            else:
                # Thêm tham số mới với xác suất adaptation_rate
                if random.random() < adaptation_rate:
                    adapted_config[key] = value
        
        # Thêm thông tin về agent gốc và điều kiện thị trường
        adapted_config["agent_type"] = base_agent.__class__.__name__
        adapted_config["base_agent_id"] = base_agent_id
        adapted_config["market_condition"] = market_condition
        
        # Tạo agent mới từ cấu hình
        adapted_agent_id, adapted_agent = self.create_agent_from_config(
            adapted_config,
            env=getattr(base_agent, 'env', None),
            agent_id=f"{base_agent_id}_adapted_{market_condition}"
        )
        
        self.logger.info(
            f"Đã tạo agent thích ứng '{adapted_agent_id}' cho điều kiện thị trường '{market_condition}' "
            f"từ agent cơ sở '{base_agent_id}'"
        )
        
        return adapted_agent_id, adapted_agent
    
    def learn_from_best_performers(
        self,
        metric_name: Optional[str] = None,
        top_n: int = 3,
        num_new_configs: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Học từ các agent hoạt động tốt nhất và tạo các cấu hình mới.
        
        Args:
            metric_name: Tên metric để đánh giá (None để sử dụng metric tổng hợp)
            top_n: Số lượng agent tốt nhất để học
            num_new_configs: Số lượng cấu hình mới cần tạo
            
        Returns:
            Danh sách các cấu hình mới
        """
        # Phân tích cấu hình hiện tại
        analysis = self.analyze_agent_configurations(metric_name=metric_name, top_n=top_n)
        
        # Lấy ID của các agent tốt nhất
        top_agent_ids = [config["agent_id"] for config in analysis["top_configs"]]
        
        if not top_agent_ids:
            self.logger.warning("Không có agent nào để học")
            return []
        
        # Tạo cấu hình mới từ các agent tốt nhất
        new_configs = self.generate_new_configurations(
            base_agent_ids=top_agent_ids,
            num_configs=num_new_configs,
            generation_method="evolutionary"
        )
        
        self.logger.info(
            f"Đã học từ {len(top_agent_ids)} agent tốt nhất "
            f"và tạo {len(new_configs)} cấu hình mới"
        )
        
        return new_configs
    
    def optimize_for_market_condition(
        self,
        market_condition: str,
        base_agent_id: Optional[str] = None,
        num_iterations: int = 3,
        configs_per_iteration: int = 3,
        eval_episodes: int = 5
    ) -> Tuple[str, BaseAgent]:
        """
        Tối ưu hóa agent cho một điều kiện thị trường cụ thể.
        
        Args:
            market_condition: Điều kiện thị trường cần tối ưu
            base_agent_id: ID của agent cơ sở (None để chọn agent tốt nhất hiện tại)
            num_iterations: Số vòng lặp tối ưu
            configs_per_iteration: Số cấu hình mỗi vòng lặp
            eval_episodes: Số episode để đánh giá mỗi cấu hình
            
        Returns:
            Tuple (optimized_agent_id, optimized_agent)
        """
        # Nếu không có base_agent_id, tìm agent tốt nhất hiện tại
        if base_agent_id is None:
            # Lấy agent có hiệu suất tốt nhất
            best_agent_id = None
            best_performance = float('-inf')
            
            for agent_id, perf_list in self.performance_history.items():
                if not perf_list:
                    continue
                    
                # Lấy hiệu suất gần nhất
                latest_perf = perf_list[-1]
                
                # Tìm giá trị mean_reward
                mean_reward = None
                if "mean_reward" in latest_perf:
                    mean_reward = latest_perf["mean_reward"]
                elif "basic_metrics" in latest_perf and "mean_reward" in latest_perf["basic_metrics"]:
                    mean_reward = latest_perf["basic_metrics"]["mean_reward"]
                
                if mean_reward is not None and mean_reward > best_performance:
                    best_performance = mean_reward
                    best_agent_id = agent_id
            
            if best_agent_id is None:
                self.logger.error("Không tìm thấy agent nào để tối ưu")
                return None, None
                
            base_agent_id = best_agent_id
        
        # Kiểm tra xem agent cơ sở có tồn tại không
        if base_agent_id not in self.agents_repository:
            self.logger.error(f"Agent ID '{base_agent_id}' không tồn tại trong kho lưu trữ")
            return None, None
        
        base_agent = self.agents_repository[base_agent_id]
        
        # Tìm môi trường cho điều kiện thị trường này
        env = None
        if self.market_condition_analyzer and hasattr(self.market_condition_analyzer, 'create_env_for_condition'):
            env = self.market_condition_analyzer.create_env_for_condition(market_condition)
        else:
            # Sử dụng môi trường của agent cơ sở
            env = getattr(base_agent, 'env', None)
        
        if env is None:
            self.logger.warning("Không có môi trường cho quá trình tối ưu. Hiệu suất không thể đánh giá")
        
        # Lưu agent tốt nhất hiện tại
        best_agent_id = base_agent_id
        best_agent = base_agent
        best_metric = float('-inf')
        
        # Lặp qua các vòng tối ưu
        for iteration in range(num_iterations):
            self.logger.info(f"Bắt đầu vòng tối ưu {iteration+1}/{num_iterations}")
            
            # Tạo configs mới dựa trên agent tốt nhất hiện tại
            new_configs = self.generate_new_configurations(
                base_agent_ids=[best_agent_id],
                num_configs=configs_per_iteration,
                target_market_condition=market_condition
            )
            
            # Tạo và đánh giá agents từ configs mới
            for config in new_configs:
                agent_id, agent = self.create_agent_from_config(
                    config,
                    env=env,
                    agent_id=f"opt_{market_condition}_{iteration}_{config.get('id', random.randint(0, 9999))}"
                )
                
                if agent_id is None or agent is None:
                    continue
                
                # Đánh giá agent mới
                if env is not None and self.agent_evaluator is not None:
                    # Đảm bảo môi trường được đăng ký
                    env_id = f"env_{market_condition}"
                    try:
                        env_id = self.agent_evaluator.register_environment(env, env_id)
                    except:
                        pass
                        
                    # Đánh giá agent
                    eval_result = self.agent_evaluator.evaluate_agent(
                        agent_id=agent_id,
                        env_id=env_id,
                        num_episodes=eval_episodes
                    )
                    
                    # Ghi lại hiệu suất
                    self.record_performance(
                        agent_id=agent_id,
                        performance_data=eval_result,
                        market_condition=market_condition
                    )
                    
                    # Cập nhật agent tốt nhất
                    if "basic_metrics" in eval_result and "mean_reward" in eval_result["basic_metrics"]:
                        metric = eval_result["basic_metrics"]["mean_reward"]
                        
                        if metric > best_metric:
                            best_metric = metric
                            best_agent_id = agent_id
                            best_agent = agent
                            
                            self.logger.info(
                                f"Tìm thấy agent tốt hơn '{best_agent_id}' "
                                f"với mean_reward = {best_metric:.4f}"
                            )
            
            self.logger.info(f"Kết thúc vòng tối ưu {iteration+1}/{num_iterations}")
        
        # Tạo agent tối ưu cuối cùng
        optimized_agent_id = f"optimized_{market_condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Sao chép agent tốt nhất
        optimized_agent = copy.deepcopy(best_agent)
        
        # Đăng ký agent tối ưu
        self.register_agent(optimized_agent, optimized_agent_id)
        
        self.logger.info(
            f"Đã tối ưu hóa thành công agent cho điều kiện thị trường '{market_condition}'. "
            f"Agent tối ưu: '{optimized_agent_id}' với metric = {best_metric:.4f}"
        )
        
        return optimized_agent_id, optimized_agent
    
    def save_repository(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """
        Lưu trạng thái của kho lưu trữ agent vào file.
        
        Args:
            file_path: Đường dẫn file (None để tạo tự động)
            
        Returns:
            Đường dẫn file đã lưu
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.output_dir / f"meta_learner_repository_{timestamp}.json"
        else:
            file_path = Path(file_path)
        
        # Tạo thư mục cha nếu chưa tồn tại
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Chuẩn bị dữ liệu để lưu
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "generated_configs": self.generated_configs,
            "current_generation": self.current_generation,
            "performance_history": self.performance_history,
            # Không thể lưu trực tiếp các agents do có thể chứa các đối tượng không serializable
            "agent_ids": list(self.agents_repository.keys()),
            "agent_types": {
                agent_id: agent.__class__.__name__
                for agent_id, agent in self.agents_repository.items()
            }
        }
        
        # Lưu vào file
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False, default=str)
            
            self.logger.info(f"Đã lưu kho lưu trữ meta-learner vào {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu kho lưu trữ: {str(e)}")
            return ""
    
    def load_repository(self, file_path: Union[str, Path]) -> bool:
        """
        Tải trạng thái của kho lưu trữ agent từ file.
        
        Args:
            file_path: Đường dẫn file
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.error(f"File không tồn tại: {file_path}")
                return False
            
            # Đọc file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Cập nhật thông tin
            self.config = data.get("config", self.default_config)
            self.generated_configs = data.get("generated_configs", [])
            self.current_generation = data.get("current_generation", 0)
            self.performance_history = data.get("performance_history", {})
            
            # Lưu ý: không thể tải trực tiếp agents từ file JSON
            # Cần tạo lại agents từ cấu hình hoặc tải từ file mô hình
            
            self.logger.info(f"Đã tải kho lưu trữ meta-learner từ {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải kho lưu trữ: {str(e)}")
            return False
    
    def export_best_agent_configs(
        self,
        top_n: int = 5,
        metric_name: Optional[str] = None,
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Xuất cấu hình của các agent tốt nhất.
        
        Args:
            top_n: Số lượng agent tốt nhất để xuất
            metric_name: Tên metric để đánh giá (None để sử dụng metric tổng hợp)
            output_format: Định dạng đầu ra ("json", "yaml", "python")
            
        Returns:
            Dict chứa kết quả xuất
        """
        # Phân tích cấu hình để tìm agents tốt nhất
        analysis = self.analyze_agent_configurations(metric_name=metric_name, top_n=top_n)
        
        # Lấy configs
        best_configs = analysis.get("top_configs", [])
        
        if not best_configs:
            self.logger.warning("Không có cấu hình nào để xuất")
            return {"configs": [], "file_path": None}
        
        # Chuẩn bị dữ liệu xuất
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "metric": metric_name,
            "configs": best_configs
        }
        
        # Tạo file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"best_agent_configs_{timestamp}"
        
        # Xuất theo định dạng
        if output_format == "json":
            file_path = self.output_dir / f"{file_name}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False, default=str)
        
        elif output_format == "yaml":
            try:
                import yaml
                file_path = self.output_dir / f"{file_name}.yaml"
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)
            except ImportError:
                self.logger.warning("Thư viện yaml không được cài đặt. Sử dụng định dạng JSON thay thế")
                file_path = self.output_dir / f"{file_name}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=4, ensure_ascii=False, default=str)
        
        elif output_format == "python":
            file_path = self.output_dir / f"{file_name}.py"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# Cấu hình agents tốt nhất được tạo bởi MetaLearner\n")
                f.write(f"# Thời gian: {datetime.now().isoformat()}\n\n")
                
                f.write("best_configs = [\n")
                for config in best_configs:
                    f.write("    {\n")
                    for key, value in config.items():
                        # Format giá trị theo kiểu Python
                        if isinstance(value, str):
                            f.write(f"        '{key}': '{value}',\n")
                        else:
                            f.write(f"        '{key}': {value},\n")
                    f.write("    },\n")
                f.write("]\n")
        else:
            self.logger.warning(f"Định dạng đầu ra không được hỗ trợ: {output_format}. Sử dụng JSON thay thế")
            file_path = self.output_dir / f"{file_name}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False, default=str)
        
        self.logger.info(f"Đã xuất cấu hình của {len(best_configs)} agent tốt nhất vào {file_path}")
        
        return {
            "configs": best_configs,
            "file_path": str(file_path)
        }
    
    def generate_meta_learning_report(
        self,
        num_generations: Optional[int] = None,
        include_config_details: bool = True,
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Tạo báo cáo về quá trình meta-learning.
        
        Args:
            num_generations: Số thế hệ gần nhất để báo cáo (None để bao gồm tất cả)
            include_config_details: Bao gồm chi tiết cấu hình trong báo cáo
            output_format: Định dạng đầu ra ("markdown", "html", "json")
            
        Returns:
            Dict chứa báo cáo và đường dẫn file
        """
        # Xác định số thế hệ cần báo cáo
        if num_generations is None or num_generations > self.current_generation:
            num_generations = self.current_generation
        
        # Lọc cấu hình theo thế hệ
        start_generation = self.current_generation - num_generations + 1
        filtered_configs = [
            config for config in self.generated_configs
            if config["generation"] >= start_generation
        ]
        
        # Phân tích hiệu suất theo thế hệ
        generation_performance = {}
        
        for config in filtered_configs:
            gen = config["generation"]
            agent_id = config["agent_id"]
            
            if gen not in generation_performance:
                generation_performance[gen] = []
            
            # Tìm hiệu suất của agent này
            if agent_id in self.performance_history and self.performance_history[agent_id]:
                # Lấy hiệu suất gần nhất
                latest_perf = self.performance_history[agent_id][-1]
                
                # Tìm giá trị mean_reward
                mean_reward = None
                if "mean_reward" in latest_perf:
                    mean_reward = latest_perf["mean_reward"]
                elif "basic_metrics" in latest_perf and "mean_reward" in latest_perf["basic_metrics"]:
                    mean_reward = latest_perf["basic_metrics"]["mean_reward"]
                
                if mean_reward is not None:
                    generation_performance[gen].append((agent_id, mean_reward))
        
        # Tính hiệu suất trung bình của mỗi thế hệ
        avg_performance = {}
        best_agents = {}
        improvement = {}
        
        for gen, performances in generation_performance.items():
            if performances:
                rewards = [perf[1] for perf in performances]
                avg_performance[gen] = sum(rewards) / len(rewards)
                
                # Tìm agent tốt nhất
                best_agent_id, best_reward = max(performances, key=lambda x: x[1])
                best_agents[gen] = (best_agent_id, best_reward)
                
                # Tính sự cải thiện so với thế hệ trước
                if gen > start_generation and gen - 1 in avg_performance:
                    improvement[gen] = avg_performance[gen] - avg_performance[gen - 1]
        
        # Tạo báo cáo
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "current_generation": self.current_generation,
            "num_generations_reported": num_generations,
            "total_agents": len(self.agents_repository),
            "total_configs": len(self.generated_configs),
            "reported_configs": len(filtered_configs),
            "generation_performance": generation_performance,
            "avg_performance": avg_performance,
            "best_agents": best_agents,
            "improvement": improvement
        }
        
        # Thêm chi tiết cấu hình nếu được yêu cầu
        if include_config_details:
            report_data["config_details"] = filtered_configs
        
        # Tạo file báo cáo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"meta_learning_report_{timestamp}"
        
        # Xuất báo cáo theo định dạng
        file_path = None
        report_content = None
        
        if output_format == "markdown":
            file_path = self.output_dir / f"{file_name}.md"
            report_content = self._generate_markdown_report(report_data)
        elif output_format == "html":
            file_path = self.output_dir / f"{file_name}.html"
            report_content = self._generate_html_report(report_data)
        elif output_format == "json":
            file_path = self.output_dir / f"{file_name}.json"
            report_content = report_data
        else:
            self.logger.warning(f"Định dạng báo cáo không được hỗ trợ: {output_format}. Sử dụng JSON thay thế")
            file_path = self.output_dir / f"{file_name}.json"
            report_content = report_data
        
        # Lưu báo cáo
        try:
            if output_format == "json" or output_format not in ["markdown", "html"]:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(report_content, f, indent=4, ensure_ascii=False, default=str)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(report_content)
            
            self.logger.info(f"Đã tạo báo cáo meta-learning tại {file_path}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu báo cáo: {str(e)}")
            file_path = None
        
        return {
            "report": report_content,
            "file_path": str(file_path) if file_path else None
        }
    
    def _extract_common_parameters(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Trích xuất các tham số phổ biến từ danh sách cấu hình.
        
        Args:
            configs: Danh sách các cấu hình
            
        Returns:
            Dict chứa các tham số phổ biến và thống kê
        """
        if not configs:
            return {}
        
        # Thu thập tất cả các khóa
        all_keys = set()
        for config in configs:
            if "config" in config:
                all_keys.update(config["config"].keys())
            else:
                all_keys.update(config.keys())
        
        # Thu thập giá trị cho mỗi khóa
        param_values = {key: [] for key in all_keys}
        
        for config in configs:
            cfg = config.get("config", config)
            for key in all_keys:
                if key in cfg:
                    param_values[key].append(cfg[key])
        
        # Phân tích các tham số
        common_params = {}
        
        for key, values in param_values.items():
            if not values:
                continue
                
            # Nếu tất cả các giá trị giống nhau
            if all(v == values[0] for v in values):
                common_params[key] = {
                    "value": values[0],
                    "common": True,
                    "frequency": 1.0
                }
            else:
                # Tính tần suất của mỗi giá trị
                value_counts = {}
                for value in values:
                    if value not in value_counts:
                        value_counts[value] = 0
                    value_counts[value] += 1
                
                # Tìm giá trị phổ biến nhất
                most_common_value, count = max(value_counts.items(), key=lambda x: x[1])
                frequency = count / len(values)
                
                # Nếu là số, tính thêm thống kê
                if all(isinstance(v, (int, float)) for v in values):
                    mean_value = sum(values) / len(values)
                    min_value = min(values)
                    max_value = max(values)
                    std_value = (sum((v - mean_value) ** 2 for v in values) / len(values)) ** 0.5
                    
                    common_params[key] = {
                        "most_common": most_common_value,
                        "frequency": frequency,
                        "mean": mean_value,
                        "min": min_value,
                        "max": max_value,
                        "std": std_value,
                        "common": False
                    }
                else:
                    common_params[key] = {
                        "most_common": most_common_value,
                        "frequency": frequency,
                        "common": False
                    }
        
        return common_params
    
    def _analyze_parameter_impacts(self, configs_with_performance: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Phân tích tác động của các tham số đến hiệu suất.
        
        Args:
            configs_with_performance: Danh sách các cấu hình kèm hiệu suất
            
        Returns:
            Dict chứa tác động của mỗi tham số
        """
        if not configs_with_performance:
            return {}
        
        # Thu thập tất cả các khóa
        all_keys = set()
        for config_info in configs_with_performance:
            config = config_info.get("config", {})
            all_keys.update(config.keys())
        
        # Loại bỏ các khóa không phải tham số
        exclude_keys = {"agent_id", "agent_type", "parent_ids", "generation_method"}
        param_keys = all_keys - exclude_keys
        
        # Phân tích tác động cho mỗi tham số
        param_impacts = {}
        
        for key in param_keys:
            # Thu thập các cặp (giá trị tham số, hiệu suất)
            value_performance_pairs = []
            
            for config_info in configs_with_performance:
                config = config_info.get("config", {})
                performance = config_info.get("avg_performance")
                
                if key in config and performance is not None:
                    value = config[key]
                    # Chỉ xử lý các giá trị số
                    if isinstance(value, (int, float)):
                        value_performance_pairs.append((value, performance))
            
            # Nếu có ít nhất 3 cặp giá trị, tính tương quan
            if len(value_performance_pairs) >= 3:
                values = [pair[0] for pair in value_performance_pairs]
                performances = [pair[1] for pair in value_performance_pairs]
                
                # Tính hệ số tương quan Pearson
                mean_value = sum(values) / len(values)
                mean_perf = sum(performances) / len(performances)
                
                numerator = sum((v - mean_value) * (p - mean_perf) for v, p in zip(values, performances))
                denominator = (sum((v - mean_value) ** 2 for v in values) * sum((p - mean_perf) ** 2 for p in performances)) ** 0.5
                
                correlation = numerator / denominator if denominator != 0 else 0
                
                # Phân loại tác động
                impact = "neutral"
                if abs(correlation) >= 0.7:
                    impact = "strong_positive" if correlation > 0 else "strong_negative"
                elif abs(correlation) >= 0.3:
                    impact = "moderate_positive" if correlation > 0 else "moderate_negative"
                elif abs(correlation) >= 0.1:
                    impact = "weak_positive" if correlation > 0 else "weak_negative"
                
                param_impacts[key] = {
                    "correlation": correlation,
                    "impact": impact,
                    "sample_size": len(value_performance_pairs),
                    "value_range": [min(values), max(values)]
                }
        
        return param_impacts
    
    def _generate_evolutionary_configs(
        self, 
        base_agents: List[Tuple[str, BaseAgent]], 
        analysis: Optional[Dict[str, Any]], 
        num_configs: int,
        target_market_condition: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Tạo cấu hình mới sử dụng thuật toán tiến hóa.
        
        Args:
            base_agents: Danh sách các agent cơ sở (agent_id, agent)
            analysis: Kết quả phân tích cấu hình
            num_configs: Số lượng cấu hình cần tạo
            target_market_condition: Điều kiện thị trường mục tiêu
            
        Returns:
            Danh sách các cấu hình mới
        """
        new_configs = []
        
        # Lấy các tham số tác động tích cực từ phân tích
        positive_params = {}
        if analysis and "parameter_impacts" in analysis:
            for param, impact_info in analysis["parameter_impacts"].items():
                if impact_info["impact"] in ["strong_positive", "moderate_positive"]:
                    positive_params[param] = impact_info
        
        # Lấy cấu hình cơ sở
        base_configs = []
        for agent_id, agent in base_agents:
            if hasattr(agent, 'config'):
                config = copy.deepcopy(agent.config)
                config["agent_type"] = agent.__class__.__name__
                config["parent_ids"] = [agent_id]
                base_configs.append(config)
        
        if not base_configs:
            self.logger.warning("Không có cấu hình cơ sở để tạo cấu hình mới")
            return []
        
        # Nếu có thông tin về điều kiện thị trường, ưu tiên cấu hình từ điều kiện đó
        market_configs = []
        if target_market_condition and target_market_condition in self.market_condition_configs:
            market_configs = self.market_condition_configs[target_market_condition]
        
        # Sử dụng crossover để tạo cấu hình mới
        for i in range(num_configs):
            # Chọn ngẫu nhiên 2 cấu hình cha mẹ
            if random.random() < 0.2 and market_configs:
                # 20% cơ hội lấy một cấu hình từ market_configs
                parent1 = random.choice(base_configs)
                parent2 = random.choice(market_configs)
            else:
                # Chọn ngẫu nhiên từ base_configs
                parents = random.sample(base_configs, min(2, len(base_configs)))
                parent1 = parents[0]
                parent2 = parents[1] if len(parents) > 1 else parents[0]
            
            # Tạo cấu hình con bằng crossover
            child_config = {}
            
            # Danh sách tất cả các khóa từ cả hai cha mẹ
            all_keys = set(parent1.keys()) | set(parent2.keys())
            
            # Ưu tiên các tham số tích cực
            for key in all_keys:
                if key in ["agent_type", "parent_ids"]:
                    continue
                    
                # Xác định cách chọn giá trị cho từng tham số
                if key in positive_params:
                    # Đối với tham số tích cực, ưu tiên giá trị tốt
                    parent1_value = parent1.get(key)
                    parent2_value = parent2.get(key)
                    
                    if parent1_value is not None and parent2_value is not None:
                        # Nếu cả hai đều có giá trị
                        if isinstance(parent1_value, (int, float)) and isinstance(parent2_value, (int, float)):
                            # Đối với giá trị số, có thể nội suy hoặc chọn giá trị tốt hơn
                            impact = positive_params[key]["impact"]
                            correlation = positive_params[key]["correlation"]
                            
                            if (correlation > 0 and parent1_value > parent2_value) or (correlation < 0 and parent1_value < parent2_value):
                                better_value = parent1_value
                                worse_value = parent2_value
                            else:
                                better_value = parent2_value
                                worse_value = parent1_value
                            
                            # Có cơ hội nội suy giữa hai giá trị
                            if random.random() < 0.7:  # 70% cơ hội chọn giá trị tốt hơn
                                child_config[key] = better_value
                            elif random.random() < 0.8:  # 80% * 30% = 24% cơ hội nội suy
                                interpolation = random.uniform(0.3, 0.7)  # Nội suy 30-70%
                                child_config[key] = worse_value * (1 - interpolation) + better_value * interpolation
                                
                                # Đảm bảo giữ đúng kiểu
                                if isinstance(parent1_value, int) and isinstance(parent2_value, int):
                                    child_config[key] = int(child_config[key])
                            else:
                                # 6% cơ hội đột biến
                                range_width = abs(parent1_value - parent2_value)
                                mutation = random.uniform(-0.2, 0.2) * range_width
                                child_config[key] = better_value + mutation
                                
                                # Đảm bảo giữ đúng kiểu
                                if isinstance(parent1_value, int) and isinstance(parent2_value, int):
                                    child_config[key] = int(child_config[key])
                        else:
                            # Đối với giá trị không phải số, chọn ngẫu nhiên
                            child_config[key] = random.choice([parent1_value, parent2_value])
                    elif parent1_value is not None:
                        child_config[key] = parent1_value
                    elif parent2_value is not None:
                        child_config[key] = parent2_value
                else:
                    # Đối với các tham số khác, thực hiện crossover thông thường
                    if random.random() < self.config["crossover_probability"]:
                        # Thực hiện crossover
                        if key in parent1 and key in parent2:
                            # Nếu cả hai đều có tham số này
                            if random.random() < 0.5:
                                child_config[key] = parent1[key]
                            else:
                                child_config[key] = parent2[key]
                        elif key in parent1:
                            child_config[key] = parent1[key]
                        elif key in parent2:
                            child_config[key] = parent2[key]
                    else:
                        # Không thực hiện crossover, sử dụng giá trị từ parent1 nếu có
                        if key in parent1:
                            child_config[key] = parent1[key]
            
            # Đảm bảo agent_type được đặt
            child_config["agent_type"] = parent1.get("agent_type", "DQNAgent")
            
            # Theo dõi parent_ids
            child_config["parent_ids"] = list(set(
                parent1.get("parent_ids", []) + parent2.get("parent_ids", [])
            ))
            
            # Thực hiện đột biến
            self._apply_mutation(child_config, positive_params)
            
            # Thêm vào danh sách cấu hình mới
            new_configs.append(child_config)
        
        return new_configs
    
    def _apply_mutation(self, config: Dict[str, Any], positive_params: Dict[str, Dict[str, Any]]) -> None:
        """
        Áp dụng đột biến cho một cấu hình.
        
        Args:
            config: Cấu hình cần đột biến
            positive_params: Thông tin các tham số tác động tích cực
        """
        for key, value in list(config.items()):
            if key in ["agent_type", "parent_ids"]:
                continue
                
            # Xác suất đột biến
            if random.random() < self.config["mutation_probability"]:
                # Đột biến tham số
                if isinstance(value, (int, float)):
                    # Đối với tham số số, thay đổi trong khoảng ±20%
                    mutation_scale = 0.2
                    
                    # Nếu là tham số tích cực, tăng đột biến theo hướng tốt
                    if key in positive_params:
                        correlation = positive_params[key]["correlation"]
                        if correlation > 0:
                            # Tham số càng lớn càng tốt
                            mutation = random.uniform(0, mutation_scale) * abs(value)
                        else:
                            # Tham số càng nhỏ càng tốt
                            mutation = random.uniform(-mutation_scale, 0) * abs(value)
                    else:
                        # Đột biến ngẫu nhiên
                        mutation = random.uniform(-mutation_scale, mutation_scale) * abs(value)
                    
                    # Áp dụng đột biến
                    mutated_value = value + mutation
                    
                    # Đảm bảo giữ đúng kiểu
                    if isinstance(value, int):
                        mutated_value = int(mutated_value)
                    
                    config[key] = mutated_value
                elif isinstance(value, bool):
                    # Đối với boolean, chuyển giá trị
                    config[key] = not value
                elif isinstance(value, str) and isinstance(value, (list, tuple)):
                    # Đối với danh sách hoặc tuple, không làm gì
                    pass
    
    def _generate_sampled_configs(
        self, 
        base_agents: List[Tuple[str, BaseAgent]], 
        analysis: Optional[Dict[str, Any]], 
        num_configs: int,
        target_market_condition: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Tạo cấu hình mới bằng cách lấy mẫu xung quanh cấu hình tốt.
        
        Args:
            base_agents: Danh sách các agent cơ sở (agent_id, agent)
            analysis: Kết quả phân tích cấu hình
            num_configs: Số lượng cấu hình cần tạo
            target_market_condition: Điều kiện thị trường mục tiêu
            
        Returns:
            Danh sách các cấu hình mới
        """
        new_configs = []
        
        # Lấy common_parameters từ phân tích
        common_params = {}
        if analysis and "common_parameters" in analysis:
            common_params = analysis["common_parameters"]
        
        # Lấy cấu hình cơ sở
        base_configs = []
        for agent_id, agent in base_agents:
            if hasattr(agent, 'config'):
                config = copy.deepcopy(agent.config)
                config["agent_type"] = agent.__class__.__name__
                config["parent_ids"] = [agent_id]
                base_configs.append(config)
        
        if not base_configs:
            self.logger.warning("Không có cấu hình cơ sở để tạo cấu hình mới")
            return []
        
        # Nếu có thông tin về điều kiện thị trường, ưu tiên cấu hình từ điều kiện đó
        market_configs = []
        if target_market_condition and target_market_condition in self.market_condition_configs:
            market_configs = self.market_condition_configs[target_market_condition]
        
        # Lấy mẫu từ cấu hình cơ sở
        for i in range(num_configs):
            # Chọn một cấu hình cơ sở ngẫu nhiên
            if random.random() < 0.3 and market_configs:
                # 30% cơ hội lấy một cấu hình từ market_configs
                base_config = copy.deepcopy(random.choice(market_configs))
            else:
                # 70% cơ hội lấy từ base_configs
                base_config = copy.deepcopy(random.choice(base_configs))
            
            # Tạo cấu hình mới bằng cách lấy mẫu
            new_config = {}
            
            for key, value in base_config.items():
                if key in ["agent_type", "parent_ids"]:
                    new_config[key] = value
                    continue
                
                # Xác định phương pháp lấy mẫu cho từng tham số
                if key in common_params and isinstance(value, (int, float)):
                    param_info = common_params[key]
                    
                    if param_info.get("common", False):
                        # Nếu là tham số phổ biến, giữ nguyên
                        new_config[key] = value
                    else:
                        # Lấy mẫu từ phân phối dựa trên thống kê
                        if "mean" in param_info and "std" in param_info:
                            # Sử dụng phân phối normal
                            mean = param_info["mean"]
                            std = param_info["std"]
                            sampled_value = random.normalvariate(mean, std)
                            
                            # Đảm bảo giữ đúng kiểu
                            if isinstance(value, int):
                                sampled_value = int(sampled_value)
                                
                            new_config[key] = sampled_value
                        else:
                            # Không có đủ thông tin thống kê, thay đổi ngẫu nhiên
                            if isinstance(value, (int, float)):
                                mutation_scale = 0.3
                                mutation = random.uniform(-mutation_scale, mutation_scale) * abs(value)
                                
                                sampled_value = value + mutation
                                
                                # Đảm bảo giữ đúng kiểu
                                if isinstance(value, int):
                                    sampled_value = int(sampled_value)
                                    
                                new_config[key] = sampled_value
                            else:
                                new_config[key] = value
                else:
                    # Không có thông tin phân tích, giữ nguyên giá trị
                    new_config[key] = value
            
            # Thêm vào danh sách cấu hình mới
            new_configs.append(new_config)
        
        return new_configs
    
    def _generate_gradient_configs(
        self, 
        base_agents: List[Tuple[str, BaseAgent]], 
        analysis: Optional[Dict[str, Any]], 
        num_configs: int,
        target_market_condition: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Tạo cấu hình mới bằng cách điều chỉnh theo gradient.
        
        Args:
            base_agents: Danh sách các agent cơ sở (agent_id, agent)
            analysis: Kết quả phân tích cấu hình
            num_configs: Số lượng cấu hình cần tạo
            target_market_condition: Điều kiện thị trường mục tiêu
            
        Returns:
            Danh sách các cấu hình mới
        """
        new_configs = []
        
        # Lấy thông tin tác động tham số từ phân tích
        param_impacts = {}
        if analysis and "parameter_impacts" in analysis:
            param_impacts = analysis["parameter_impacts"]
        
        # Xác định agent tốt nhất để làm cơ sở
        best_agent_id = None
        best_agent = None
        best_performance = float('-inf')
        
        for agent_id, agent in base_agents:
            # Tìm hiệu suất của agent
            performance = float('-inf')
            
            if agent_id in self.performance_history and self.performance_history[agent_id]:
                # Lấy hiệu suất gần nhất
                latest_perf = self.performance_history[agent_id][-1]
                
                # Tìm giá trị mean_reward
                if "mean_reward" in latest_perf:
                    performance = latest_perf["mean_reward"]
                elif "basic_metrics" in latest_perf and "mean_reward" in latest_perf["basic_metrics"]:
                    performance = latest_perf["basic_metrics"]["mean_reward"]
            
            if performance > best_performance:
                best_performance = performance
                best_agent_id = agent_id
                best_agent = agent
        
        if best_agent is None:
            # Nếu không có thông tin hiệu suất, chọn ngẫu nhiên
            best_agent_id, best_agent = random.choice(base_agents)
        
        # Lấy cấu hình cơ sở
        if not hasattr(best_agent, 'config'):
            self.logger.warning("Agent tốt nhất không có cấu hình")
            return []
            
        base_config = copy.deepcopy(best_agent.config)
        base_config["agent_type"] = best_agent.__class__.__name__
        base_config["parent_ids"] = [best_agent_id]
        
        # Tạo các cấu hình mới theo gradient
        for i in range(num_configs):
            new_config = copy.deepcopy(base_config)
            
            # Điều chỉnh theo gradient cho từng tham số
            for key, value in list(new_config.items()):
                if key in ["agent_type", "parent_ids"]:
                    continue
                    
                # Chỉ điều chỉnh các tham số số
                if not isinstance(value, (int, float)):
                    continue
                
                # Nếu có thông tin tác động, điều chỉnh theo hướng tích cực
                if key in param_impacts:
                    impact_info = param_impacts[key]
                    correlation = impact_info["correlation"]
                    
                    # Tính gradient step
                    step_size = abs(value) * 0.1  # 10% của giá trị hiện tại
                    
                    # Điều chỉnh theo hướng tương quan
                    if correlation > 0:
                        # Tương quan dương, tăng giá trị
                        adjustment = random.uniform(0.05, 0.15) * step_size
                    elif correlation < 0:
                        # Tương quan âm, giảm giá trị
                        adjustment = -random.uniform(0.05, 0.15) * step_size
                    else:
                        # Không có tương quan rõ ràng, điều chỉnh ngẫu nhiên
                        adjustment = random.uniform(-0.1, 0.1) * step_size
                    
                    # Áp dụng điều chỉnh
                    adjusted_value = value + adjustment
                    
                    # Đảm bảo giữ đúng kiểu
                    if isinstance(value, int):
                        adjusted_value = int(adjusted_value)
                        
                    new_config[key] = adjusted_value
                else:
                    # Không có thông tin tác động, điều chỉnh nhỏ ngẫu nhiên
                    if random.random() < 0.3:  # 30% cơ hội điều chỉnh
                        adjustment = random.uniform(-0.05, 0.05) * abs(value)
                        adjusted_value = value + adjustment
                        
                        # Đảm bảo giữ đúng kiểu
                        if isinstance(value, int):
                            adjusted_value = int(adjusted_value)
                            
                        new_config[key] = adjusted_value
            
            # Thêm vào danh sách cấu hình mới
            new_configs.append(new_config)
        
        return new_configs
    
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Tạo hash duy nhất cho một cấu hình.
        
        Args:
            config: Cấu hình cần hash
            
        Returns:
            Chuỗi hash
        """
        import hashlib
        import json
        
        # Loại bỏ các khóa không cần thiết
        clean_config = {k: v for k, v in config.items() if k not in ["parent_ids", "agent_id"]}
        
        # Sắp xếp khóa để đảm bảo tính nhất quán
        config_str = json.dumps(clean_config, sort_keys=True)
        
        # Tạo hash
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """
        Tạo báo cáo markdown từ dữ liệu.
        
        Args:
            report_data: Dữ liệu báo cáo
            
        Returns:
            Chuỗi markdown
        """
        md = f"# Báo cáo Meta-Learning\n\n"
        md += f"**Thời gian:** {report_data.get('timestamp', 'Không xác định')}\n\n"
        
        # Thông tin chung
        md += "## 1. Thông tin chung\n\n"
        md += f"- **Thế hệ hiện tại:** {report_data.get('current_generation', 0)}\n"
        md += f"- **Số thế hệ báo cáo:** {report_data.get('num_generations_reported', 0)}\n"
        md += f"- **Tổng số agent:** {report_data.get('total_agents', 0)}\n"
        md += f"- **Tổng số cấu hình đã tạo:** {report_data.get('total_configs', 0)}\n"
        md += f"- **Số cấu hình trong báo cáo:** {report_data.get('reported_configs', 0)}\n\n"
        
        # Hiệu suất theo thế hệ
        avg_performance = report_data.get('avg_performance', {})
        best_agents = report_data.get('best_agents', {})
        improvement = report_data.get('improvement', {})
        
        if avg_performance:
            md += "## 2. Hiệu suất theo thế hệ\n\n"
            md += "| Thế hệ | Hiệu suất TB | Agent tốt nhất | Hiệu suất tốt nhất | Cải thiện |\n"
            md += "|--------|-------------|---------------|-------------------|----------|\n"
            
            for gen in sorted(avg_performance.keys()):
                avg = avg_performance.get(gen, 0)
                best_info = best_agents.get(gen, ("Unknown", 0))
                best_agent_id, best_reward = best_info
                imp = improvement.get(gen, 0)
                
                # Dấu + cho cải thiện dương
                imp_str = f"+{imp:.4f}" if imp > 0 else f"{imp:.4f}"
                
                md += f"| {gen} | {avg:.4f} | {best_agent_id} | {best_reward:.4f} | {imp_str} |\n"
            
            md += "\n"
        
        # Biểu đồ tiến trình (mô tả)
        if avg_performance:
            md += "## 3. Diễn biến hiệu suất\n\n"
            md += "### Hiệu suất trung bình theo thế hệ\n\n"
            
            gens = sorted(avg_performance.keys())
            avgs = [avg_performance[gen] for gen in gens]
            
            # Tạo "biểu đồ" ASCII đơn giản
            md += "```\n"
            max_avg = max(avgs)
            min_avg = min(avgs)
            range_avg = max_avg - min_avg
            
            if range_avg > 0:
                height = 10
                for y in range(height, 0, -1):
                    line = ""
                    for avg in avgs:
                        normalized_val = (avg - min_avg) / range_avg
                        bar_height = normalized_val * height
                        if bar_height >= y:
                            line += "█"
                        else:
                            line += " "
                    md += line + "\n"
                
                # Trục x
                md += "─" * len(gens) + "\n"
                
            md += f"Min: {min_avg:.4f}, Max: {max_avg:.4f}\n"
            md += "```\n\n"
            
            # Dữ liệu cải thiện
            if improvement:
                md += "### Cải thiện giữa các thế hệ\n\n"
                
                # Tính tổng cải thiện
                total_improvement = sum(improvement.values())
                avg_improvement = total_improvement / len(improvement) if improvement else 0
                
                md += f"- **Tổng cải thiện:** {total_improvement:.4f}\n"
                md += f"- **Cải thiện trung bình mỗi thế hệ:** {avg_improvement:.4f}\n\n"
        
        # Phân tích cấu hình
        if "config_details" in report_data and report_data["config_details"]:
            configs = report_data["config_details"]
            
            md += "## 4. Phân tích cấu hình\n\n"
            md += f"### 4.1. Phân bố theo loại agent\n\n"
            
            # Đếm các loại agent
            agent_types = {}
            for config in configs:
                agent_type = config.get("config", {}).get("agent_type", config.get("agent_type", "Unknown"))
                if agent_type not in agent_types:
                    agent_types[agent_type] = 0
                agent_types[agent_type] += 1
            
            # Hiển thị phân bố
            md += "| Loại agent | Số lượng | Tỉ lệ |\n"
            md += "|------------|----------|-------|\n"
            
            for agent_type, count in agent_types.items():
                percentage = count / len(configs) * 100
                md += f"| {agent_type} | {count} | {percentage:.1f}% |\n"
            
            md += "\n"
            
            # Chi tiết cấu hình
            md += "### 4.2. Chi tiết cấu hình đã tạo\n\n"
            
            # Giới hạn số lượng cấu hình hiển thị
            max_configs = min(10, len(configs))
            md += f"*Hiển thị {max_configs}/{len(configs)} cấu hình*\n\n"
            
            for i, config_info in enumerate(configs[:max_configs]):
                config = config_info.get("config", config_info)
                agent_type = config.get("agent_type", "Unknown")
                gen = config_info.get("generation", "Unknown")
                created_at = config_info.get("created_at", "Unknown")
                
                md += f"#### Cấu hình {i+1}: {agent_type} (Thế hệ {gen})\n\n"
                md += f"**Tạo lúc:** {created_at}\n\n"
                
                # Hiển thị các tham số chính
                md += "**Tham số chính:**\n\n"
                md += "```\n"
                
                # Chọn một số tham số quan trọng để hiển thị
                important_params = ["learning_rate", "gamma", "epsilon", "batch_size", "hidden_layers"]
                for param in important_params:
                    if param in config:
                        md += f"{param}: {config[param]}\n"
                
                md += "```\n\n"
        
        # Kết luận
        md += "## 5. Kết luận\n\n"
        
        # Phân tích xu hướng
        if improvement:
            positive_improvements = sum(1 for imp in improvement.values() if imp > 0)
            negative_improvements = sum(1 for imp in improvement.values() if imp < 0)
            
            if positive_improvements > negative_improvements:
                md += "Meta-learning đang cho thấy **xu hướng tích cực**. "
                md += f"Có {positive_improvements} thế hệ cải thiện hiệu suất và {negative_improvements} thế hệ giảm hiệu suất.\n\n"
            elif positive_improvements < negative_improvements:
                md += "Meta-learning đang gặp **khó khăn** trong việc cải thiện hiệu suất. "
                md += f"Có {negative_improvements} thế hệ giảm hiệu suất và chỉ {positive_improvements} thế hệ cải thiện.\n\n"
            else:
                md += "Meta-learning đang có **kết quả trung hòa**. "
                md += f"Số thế hệ cải thiện và giảm hiệu suất là bằng nhau ({positive_improvements}).\n\n"
        
        # Đề xuất
        md += "### Đề xuất tiếp theo\n\n"
        md += "1. Tiếp tục huấn luyện thêm thế hệ để cải thiện hiệu suất\n"
        md += "2. Tăng cường đa dạng cấu hình để khám phá không gian tham số rộng hơn\n"
        md += "3. Tập trung tối ưu hóa các tham số có tác động tích cực mạnh\n"
        
        return md
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """
        Tạo báo cáo HTML từ dữ liệu.
        
        Args:
            report_data: Dữ liệu báo cáo
            
        Returns:
            Chuỗi HTML
        """
        html = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo Meta-Learning</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #2980b9;
            margin-top: 30px;
        }
        h3 {
            color: #3498db;
        }
        h4 {
            color: #16a085;
            margin-top: 20px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .positive {
            color: #27ae60;
        }
        .negative {
            color: #e74c3c;
        }
        .neutral {
            color: #f39c12;
        }
        .conclusion {
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }
        .config-container {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        pre {
            background-color: #f8f9f9;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1>Báo cáo Meta-Learning</h1>
    <p><strong>Thời gian:</strong> """ + f"{report_data.get('timestamp', 'Không xác định')}" + """</p>
    
    <h2>1. Thông tin chung</h2>
    <ul>
        <li><strong>Thế hệ hiện tại:</strong> """ + f"{report_data.get('current_generation', 0)}" + """</li>
        <li><strong>Số thế hệ báo cáo:</strong> """ + f"{report_data.get('num_generations_reported', 0)}" + """</li>
        <li><strong>Tổng số agent:</strong> """ + f"{report_data.get('total_agents', 0)}" + """</li>
        <li><strong>Tổng số cấu hình đã tạo:</strong> """ + f"{report_data.get('total_configs', 0)}" + """</li>
        <li><strong>Số cấu hình trong báo cáo:</strong> """ + f"{report_data.get('reported_configs', 0)}" + """</li>
    </ul>
"""
        
        # Hiệu suất theo thế hệ
        avg_performance = report_data.get('avg_performance', {})
        best_agents = report_data.get('best_agents', {})
        improvement = report_data.get('improvement', {})
        
        if avg_performance:
            html += """
    <h2>2. Hiệu suất theo thế hệ</h2>
    <table>
        <tr>
            <th>Thế hệ</th>
            <th>Hiệu suất TB</th>
            <th>Agent tốt nhất</th>
            <th>Hiệu suất tốt nhất</th>
            <th>Cải thiện</th>
        </tr>
"""
            
            for gen in sorted(avg_performance.keys()):
                avg = avg_performance.get(gen, 0)
                best_info = best_agents.get(gen, ("Unknown", 0))
                best_agent_id, best_reward = best_info
                imp = improvement.get(gen, 0)
                
                # CSS class cho cải thiện
                imp_class = "positive" if imp > 0 else ("negative" if imp < 0 else "neutral")
                imp_str = f"+{imp:.4f}" if imp > 0 else f"{imp:.4f}"
                
                html += f"""        <tr>
            <td>{gen}</td>
            <td>{avg:.4f}</td>
            <td>{best_agent_id}</td>
            <td>{best_reward:.4f}</td>
            <td class="{imp_class}">{imp_str}</td>
        </tr>
"""
            
            html += """    </table>
"""
        
        # Phân tích cấu hình
        if "config_details" in report_data and report_data["config_details"]:
            configs = report_data["config_details"]
            
            html += """
    <h2>4. Phân tích cấu hình</h2>
    <h3>4.1. Phân bố theo loại agent</h3>
"""
            
            # Đếm các loại agent
            agent_types = {}
            for config in configs:
                agent_type = config.get("config", {}).get("agent_type", config.get("agent_type", "Unknown"))
                if agent_type not in agent_types:
                    agent_types[agent_type] = 0
                agent_types[agent_type] += 1
            
            # Hiển thị phân bố
            html += """    <table>
        <tr>
            <th>Loại agent</th>
            <th>Số lượng</th>
            <th>Tỉ lệ</th>
        </tr>
"""
            
            for agent_type, count in agent_types.items():
                percentage = count / len(configs) * 100
                html += f"""        <tr>
            <td>{agent_type}</td>
            <td>{count}</td>
            <td>{percentage:.1f}%</td>
        </tr>
"""
            
            html += """    </table>
    
    <h3>4.2. Chi tiết cấu hình đã tạo</h3>
"""
            
            # Giới hạn số lượng cấu hình hiển thị
            max_configs = min(10, len(configs))
            html += f"    <p><i>Hiển thị {max_configs}/{len(configs)} cấu hình</i></p>"
            
            for i, config_info in enumerate(configs[:max_configs]):
                config = config_info.get("config", config_info)
                agent_type = config.get("agent_type", "Unknown")
                gen = config_info.get("generation", "Unknown")
                created_at = config_info.get("created_at", "Unknown")
                
                html += f"""
    <h4>Cấu hình {i+1}: {agent_type} (Thế hệ {gen})</h4>
    <div class="config-container">
        <p><strong>Tạo lúc:</strong> {created_at}</p>
        <p><strong>Tham số chính:</strong></p>
        <pre>
"""
                
                # Chọn một số tham số quan trọng để hiển thị
                important_params = ["learning_rate", "gamma", "epsilon", "batch_size", "hidden_layers"]
                for param in important_params:
                    if param in config:
                        html += f"{param}: {config[param]}\n"
                
                html += """        </pre>
    </div>
"""
        
        # Kết luận
        html += """
    <h2>5. Kết luận</h2>
    <div class="conclusion">
"""
        
        # Phân tích xu hướng
        if improvement:
            positive_improvements = sum(1 for imp in improvement.values() if imp > 0)
            negative_improvements = sum(1 for imp in improvement.values() if imp < 0)
            
            if positive_improvements > negative_improvements:
                html += f"""        <p>Meta-learning đang cho thấy <strong>xu hướng tích cực</strong>. 
        Có {positive_improvements} thế hệ cải thiện hiệu suất và {negative_improvements} thế hệ giảm hiệu suất.</p>
"""
            elif positive_improvements < negative_improvements:
                html += f"""        <p>Meta-learning đang gặp <strong>khó khăn</strong> trong việc cải thiện hiệu suất. 
        Có {negative_improvements} thế hệ giảm hiệu suất và chỉ {positive_improvements} thế hệ cải thiện.</p>
"""
            else:
                html += f"""        <p>Meta-learning đang có <strong>kết quả trung hòa</strong>.
        Số thế hệ cải thiện và giảm hiệu suất là bằng nhau ({positive_improvements}).</p>
"""
        
        # Đề xuất
        html += """
        <h3>Đề xuất tiếp theo</h3>
        <ol>
            <li>Tiếp tục huấn luyện thêm thế hệ để cải thiện hiệu suất</li>
            <li>Tăng cường đa dạng cấu hình để khám phá không gian tham số rộng hơn</li>
            <li>Tập trung tối ưu hóa các tham số có tác động tích cực mạnh</li>
        </ol>
    </div>
</body>
</html>"""
        
        return html