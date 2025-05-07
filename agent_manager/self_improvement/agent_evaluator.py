"""
Đánh giá hiệu suất agent.
File này định nghĩa lớp AgentEvaluator để đánh giá hiệu suất, độ mạnh/yếu,
và các điểm cần cải thiện của các agent trong hệ thống giao dịch tự động.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime

# Thêm thư mục gốc vào path để import được các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config

from models.agents.base_agent import BaseAgent
from environments.base_environment import BaseEnvironment
from environments.trading_gym.trading_env import TradingEnv

from backtesting.performance_metrics import PerformanceMetrics
from backtesting.strategy_tester import StrategyTester

from logs.metrics.training_metrics import TrainingMetricsTracker

class AgentEvaluator:
    """
    Lớp đánh giá hiệu suất và chất lượng của agent.
    
    Cung cấp các phương thức để:
    1. Đánh giá hiệu suất agent trong môi trường cụ thể
    2. So sánh hiệu suất giữa các agent
    3. Phân tích điểm mạnh và điểm yếu của agent
    4. Đề xuất cải tiến cho agent
    5. Theo dõi sự phát triển của agent qua thời gian
    """
    
    def __init__(
        self,
        agents: Optional[List[BaseAgent]] = None,
        environments: Optional[List[BaseEnvironment]] = None,
        metrics: Optional[List[str]] = None,
        test_data: Optional[Dict[str, pd.DataFrame]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        evaluation_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo AgentEvaluator.
        
        Args:
            agents: Danh sách các agent cần đánh giá
            environments: Danh sách các môi trường để đánh giá
            metrics: Danh sách các metrics để sử dụng đánh giá
            test_data: Dữ liệu kiểm tra cho đánh giá
            output_dir: Thư mục lưu kết quả đánh giá
            evaluation_config: Cấu hình đánh giá bổ sung
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("agent_evaluator")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập các danh sách agent và môi trường
        self.agents = agents or []
        self.environments = environments or []
        
        # Thiết lập dữ liệu kiểm tra
        self.test_data = test_data or {}
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(self.system_config.get("model_dir", "./models")) / "evaluations" / timestamp
        else:
            self.output_dir = Path(output_dir)
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập metrics đánh giá
        self.default_metrics = [
            "total_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "volatility",
            "calmar_ratio",
            "average_trade",
            "recovery_factor"
        ]
        
        self.metrics = metrics or self.default_metrics
        
        # Thiết lập cấu hình đánh giá
        self.default_config = {
            "num_episodes": 10,
            "episode_steps": 1000,
            "render": False,
            "verbose": True,
            "save_results": True,
            "use_bootstrap": True,
            "bootstrap_samples": 100,
            "confidence_level": 0.95,
            "cross_validation": False,
            "cv_folds": 5,
            "comparison_baseline": None,
            "minimum_performance": {
                "sharpe_ratio": 0.5,
                "win_rate": 0.4,
                "total_return": 0.0
            },
            "evaluation_frequency": "weekly"
        }
        
        self.eval_config = self.default_config.copy()
        if evaluation_config:
            self.eval_config.update(evaluation_config)
        
        # Khởi tạo dict để lưu kết quả đánh giá
        self.evaluation_results = {}
        self.comparison_results = {}
        
        # Khởi tạo metrics tracker
        self.metrics_tracker = None
        
        # Khởi tạo PerformanceMetrics nếu có
        try:
            self.performance_metrics = PerformanceMetrics(logger=self.logger)
        except Exception as e:
            self.logger.warning(f"Không thể khởi tạo PerformanceMetrics: {str(e)}")
            self.performance_metrics = None
        
        self.logger.info(f"Đã khởi tạo AgentEvaluator với {len(self.agents)} agents và {len(self.environments)} môi trường")
    
    def register_agent(self, agent: BaseAgent, agent_id: Optional[str] = None) -> str:
        """
        Đăng ký một agent mới để đánh giá.
        
        Args:
            agent: Agent cần đăng ký
            agent_id: ID cho agent (None để tạo tự động)
            
        Returns:
            ID của agent đã đăng ký
        """
        if agent_id is None:
            agent_id = f"agent_{len(self.agents) + 1}_{agent.__class__.__name__}"
        
        # Kiểm tra xem agent_id đã tồn tại chưa
        existing_ids = [a.get("id") for a in self.agents if isinstance(a, dict) and "id" in a]
        if agent_id in existing_ids:
            self.logger.warning(f"Agent ID '{agent_id}' đã tồn tại. Thêm hậu tố để tránh trùng lặp.")
            agent_id = f"{agent_id}_{datetime.now().strftime('%H%M%S')}"
        
        # Lưu agent với ID
        self.agents.append({"id": agent_id, "agent": agent})
        
        self.logger.info(f"Đã đăng ký agent '{agent_id}' ({agent.__class__.__name__}) để đánh giá")
        
        return agent_id
    
    def register_environment(self, environment: BaseEnvironment, env_id: Optional[str] = None) -> str:
        """
        Đăng ký một môi trường mới để đánh giá.
        
        Args:
            environment: Môi trường cần đăng ký
            env_id: ID cho môi trường (None để tạo tự động)
            
        Returns:
            ID của môi trường đã đăng ký
        """
        if env_id is None:
            env_id = f"env_{len(self.environments) + 1}_{environment.__class__.__name__}"
        
        # Kiểm tra xem env_id đã tồn tại chưa
        existing_ids = [e.get("id") for e in self.environments if isinstance(e, dict) and "id" in e]
        if env_id in existing_ids:
            self.logger.warning(f"Environment ID '{env_id}' đã tồn tại. Thêm hậu tố để tránh trùng lặp.")
            env_id = f"{env_id}_{datetime.now().strftime('%H%M%S')}"
        
        # Lưu environment với ID
        self.environments.append({"id": env_id, "environment": environment})
        
        self.logger.info(f"Đã đăng ký môi trường '{env_id}' ({environment.__class__.__name__}) để đánh giá")
        
        return env_id
    
    def evaluate_agent(
        self,
        agent_id: str,
        env_id: str,
        num_episodes: Optional[int] = None,
        evaluation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Đánh giá một agent cụ thể trong một môi trường cụ thể.
        
        Args:
            agent_id: ID của agent cần đánh giá
            env_id: ID của môi trường để đánh giá
            num_episodes: Số lượng episodes đánh giá (None để sử dụng cấu hình mặc định)
            evaluation_params: Các tham số đánh giá bổ sung
            
        Returns:
            Dict chứa kết quả đánh giá
        """
        # Lấy agent và môi trường theo ID
        agent_info = next((a for a in self.agents if isinstance(a, dict) and a.get("id") == agent_id), None)
        env_info = next((e for e in self.environments if isinstance(e, dict) and e.get("id") == env_id), None)
        
        if agent_info is None:
            self.logger.error(f"Không tìm thấy agent với ID '{agent_id}'")
            return {"error": f"Agent ID '{agent_id}' không tồn tại"}
        
        if env_info is None:
            self.logger.error(f"Không tìm thấy môi trường với ID '{env_id}'")
            return {"error": f"Environment ID '{env_id}' không tồn tại"}
        
        agent = agent_info["agent"]
        env = env_info["environment"]
        
        # Thiết lập số episodes
        if num_episodes is None:
            num_episodes = self.eval_config["num_episodes"]
        
        # Kết hợp tham số đánh giá
        params = self.eval_config.copy()
        if evaluation_params:
            params.update(evaluation_params)
        
        # Tạo ID đánh giá
        eval_id = f"{agent_id}_{env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(
            f"Bắt đầu đánh giá agent '{agent_id}' trong môi trường '{env_id}' "
            f"với {num_episodes} episodes"
        )
        
        try:
            # Đánh giá agent
            rewards = agent.evaluate(
                env=env,
                num_episodes=num_episodes,
                max_steps=params["episode_steps"],
                render=params["render"]
            )
            
            # Tính toán các metrics
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            median_reward = np.median(rewards)
            
            # Tạo dict kết quả cơ bản
            results = {
                "eval_id": eval_id,
                "agent_id": agent_id,
                "env_id": env_id,
                "agent_class": agent.__class__.__name__,
                "env_class": env.__class__.__name__,
                "num_episodes": num_episodes,
                "timestamp": datetime.now().isoformat(),
                "basic_metrics": {
                    "mean_reward": float(mean_reward),
                    "std_reward": float(std_reward),
                    "min_reward": float(min_reward),
                    "max_reward": float(max_reward),
                    "median_reward": float(median_reward)
                },
                "episode_rewards": [float(r) for r in rewards]
            }
            
            # Thêm các performance metrics nâng cao nếu có
            if self.performance_metrics is not None and hasattr(env, "history"):
                try:
                    # Lấy lịch sử từ môi trường
                    env_history = env.history
                    
                    # Tính toán các metrics nâng cao
                    perf_metrics = self.performance_metrics.calculate_metrics(
                        returns=rewards,
                        balance_history=env_history.get("balances", []),
                        position_history=env_history.get("positions", []),
                        trade_history=getattr(env, "order_history", [])
                    )
                    
                    # Thêm vào kết quả
                    results["advanced_metrics"] = perf_metrics
                    
                except Exception as e:
                    self.logger.warning(f"Không thể tính toán performance metrics: {str(e)}")
            
            # Đánh giá so với ngưỡng tối thiểu
            minimum_performance = params["minimum_performance"]
            evaluation_passed = True
            
            # Kiểm tra từng metric so với ngưỡng
            for metric, threshold in minimum_performance.items():
                # Tìm giá trị metric từ kết quả
                metric_value = None
                
                if metric == "total_return":
                    metric_value = mean_reward
                elif "advanced_metrics" in results and metric in results["advanced_metrics"]:
                    metric_value = results["advanced_metrics"][metric]
                elif metric in results["basic_metrics"]:
                    metric_value = results["basic_metrics"][metric]
                
                if metric_value is not None:
                    if metric_value < threshold:
                        evaluation_passed = False
                        self.logger.warning(
                            f"Metric '{metric}' không đạt ngưỡng tối thiểu: "
                            f"{metric_value:.4f} < {threshold:.4f}"
                        )
            
            results["evaluation_passed"] = evaluation_passed
            
            # Lưu kết quả
            self.evaluation_results[eval_id] = results
            
            # Lưu kết quả đánh giá vào file nếu được yêu cầu
            if params["save_results"]:
                self._save_evaluation_results(eval_id, results)
            
            self.logger.info(
                f"Đã hoàn thành đánh giá agent '{agent_id}': Mean reward = {mean_reward:.4f}, "
                f"Đánh giá {'PASSED' if evaluation_passed else 'FAILED'}"
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Lỗi khi đánh giá agent '{agent_id}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"error": error_msg, "eval_id": eval_id}
    
    def evaluate_all_agents(
        self,
        env_id: Optional[str] = None,
        num_episodes: Optional[int] = None,
        evaluation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Đánh giá tất cả agents đã đăng ký trong một môi trường cụ thể.
        
        Args:
            env_id: ID của môi trường để đánh giá (None để đánh giá trên tất cả môi trường)
            num_episodes: Số lượng episodes đánh giá
            evaluation_params: Các tham số đánh giá bổ sung
            
        Returns:
            Dict ánh xạ agent_id -> kết quả đánh giá
        """
        results = {}
        
        # Xác định danh sách môi trường để đánh giá
        if env_id is not None:
            env_ids = [env_id]
        else:
            env_ids = [e.get("id") for e in self.environments if isinstance(e, dict) and "id" in e]
        
        if not env_ids:
            self.logger.error("Không có môi trường nào để đánh giá")
            return {"error": "Không có môi trường nào để đánh giá"}
        
        # Xác định danh sách agent để đánh giá
        agent_ids = [a.get("id") for a in self.agents if isinstance(a, dict) and "id" in a]
        
        if not agent_ids:
            self.logger.error("Không có agent nào để đánh giá")
            return {"error": "Không có agent nào để đánh giá"}
        
        # Lặp qua từng agent và môi trường
        for agent_id in agent_ids:
            agent_results = {}
            
            for env_id in env_ids:
                # Đánh giá agent trong môi trường
                eval_result = self.evaluate_agent(
                    agent_id=agent_id,
                    env_id=env_id,
                    num_episodes=num_episodes,
                    evaluation_params=evaluation_params
                )
                
                agent_results[env_id] = eval_result
            
            results[agent_id] = agent_results
        
        return results
    
    def compare_agents(
        self,
        agent_ids: List[str],
        env_id: str,
        metrics_to_compare: Optional[List[str]] = None,
        num_episodes: Optional[int] = None,
        comparison_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        So sánh hiệu suất của nhiều agents trong cùng một môi trường.
        
        Args:
            agent_ids: Danh sách ID các agent cần so sánh
            env_id: ID của môi trường để đánh giá
            metrics_to_compare: Danh sách các metrics để so sánh
            num_episodes: Số lượng episodes đánh giá
            comparison_params: Các tham số so sánh bổ sung
            
        Returns:
            Dict chứa kết quả so sánh
        """
        # Thiết lập metrics để so sánh
        if metrics_to_compare is None:
            metrics_to_compare = self.metrics
        
        # Tạo ID so sánh
        comparison_id = f"comparison_{env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(
            f"Bắt đầu so sánh {len(agent_ids)} agents trong môi trường '{env_id}' "
            f"dựa trên {len(metrics_to_compare)} metrics"
        )
        
        # Thiết lập tham số đánh giá
        eval_params = self.eval_config.copy()
        if comparison_params:
            eval_params.update(comparison_params)
        
        # Đánh giá từng agent
        agent_evaluations = {}
        
        for agent_id in agent_ids:
            eval_result = self.evaluate_agent(
                agent_id=agent_id,
                env_id=env_id,
                num_episodes=num_episodes,
                evaluation_params=eval_params
            )
            
            agent_evaluations[agent_id] = eval_result
        
        # Tạo bảng so sánh
        comparison_table = {}
        
        for metric in metrics_to_compare:
            metric_values = {}
            
            for agent_id, eval_result in agent_evaluations.items():
                # Tìm giá trị metric từ kết quả đánh giá
                metric_value = None
                
                if metric == "mean_reward" or metric == "total_return":
                    if "basic_metrics" in eval_result and "mean_reward" in eval_result["basic_metrics"]:
                        metric_value = eval_result["basic_metrics"]["mean_reward"]
                elif "advanced_metrics" in eval_result and metric in eval_result["advanced_metrics"]:
                    metric_value = eval_result["advanced_metrics"][metric]
                elif "basic_metrics" in eval_result and metric in eval_result["basic_metrics"]:
                    metric_value = eval_result["basic_metrics"][metric]
                
                if metric_value is not None:
                    metric_values[agent_id] = metric_value
            
            comparison_table[metric] = metric_values
        
        # Xác định agent tốt nhất cho từng metric
        best_agents = {}
        
        for metric, values in comparison_table.items():
            if not values:
                continue
                
            # Xác định agent tốt nhất cho metric này
            if metric in ["max_drawdown", "volatility"]:
                # Các metric càng nhỏ càng tốt
                best_agent_id = min(values.items(), key=lambda x: x[1])[0]
            else:
                # Các metric càng lớn càng tốt
                best_agent_id = max(values.items(), key=lambda x: x[1])[0]
            
            best_agents[metric] = best_agent_id
        
        # Tính điểm tổng thể cho mỗi agent
        overall_scores = {agent_id: 0 for agent_id in agent_ids}
        
        for metric, values in comparison_table.items():
            if not values:
                continue
                
            # Chuẩn hóa giá trị metric thành điểm từ 0-1
            min_value = min(values.values())
            max_value = max(values.values())
            
            if max_value == min_value:
                # Tránh chia cho 0
                normalized_values = {agent_id: 1.0 for agent_id in values}
            else:
                if metric in ["max_drawdown", "volatility"]:
                    # Các metric càng nhỏ càng tốt
                    normalized_values = {
                        agent_id: 1 - (value - min_value) / (max_value - min_value)
                        for agent_id, value in values.items()
                    }
                else:
                    # Các metric càng lớn càng tốt
                    normalized_values = {
                        agent_id: (value - min_value) / (max_value - min_value)
                        for agent_id, value in values.items()
                    }
            
            # Cộng điểm chuẩn hóa vào điểm tổng thể
            for agent_id, score in normalized_values.items():
                overall_scores[agent_id] += score
        
        # Xác định agent tốt nhất tổng thể
        best_overall_agent = max(overall_scores.items(), key=lambda x: x[1])[0]
        
        # Tạo kết quả so sánh
        comparison_result = {
            "comparison_id": comparison_id,
            "env_id": env_id,
            "agent_ids": agent_ids,
            "metrics_compared": metrics_to_compare,
            "timestamp": datetime.now().isoformat(),
            "comparison_table": comparison_table,
            "best_agents_by_metric": best_agents,
            "overall_scores": overall_scores,
            "best_overall_agent": best_overall_agent,
            "evaluations": agent_evaluations
        }
        
        # Lưu kết quả so sánh
        self.comparison_results[comparison_id] = comparison_result
        
        # Lưu kết quả so sánh vào file nếu được yêu cầu
        if eval_params.get("save_results", True):
            self._save_comparison_results(comparison_id, comparison_result)
        
        self.logger.info(
            f"Đã hoàn thành so sánh {len(agent_ids)} agents. "
            f"Agent tốt nhất tổng thể: {best_overall_agent}"
        )
        
        return comparison_result
    
    def analyze_agent_strengths_weaknesses(
        self,
        agent_id: str,
        evaluation_id: Optional[str] = None,
        metrics_importance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Phân tích điểm mạnh và điểm yếu của một agent dựa trên kết quả đánh giá.
        
        Args:
            agent_id: ID của agent cần phân tích
            evaluation_id: ID của đánh giá cụ thể (None để sử dụng đánh giá gần nhất)
            metrics_importance: Dict ánh xạ tên metrics -> trọng số tầm quan trọng
            
        Returns:
            Dict chứa phân tích điểm mạnh/yếu
        """
        # Tìm kết quả đánh giá
        if evaluation_id is not None:
            # Sử dụng đánh giá cụ thể
            if evaluation_id not in self.evaluation_results:
                self.logger.error(f"Không tìm thấy kết quả đánh giá với ID '{evaluation_id}'")
                return {"error": f"Evaluation ID '{evaluation_id}' không tồn tại"}
                
            evaluation = self.evaluation_results[evaluation_id]
            
            # Kiểm tra xem đánh giá có phải cho agent này không
            if evaluation.get("agent_id") != agent_id:
                self.logger.error(f"Evaluation ID '{evaluation_id}' không phải cho agent '{agent_id}'")
                return {"error": f"Evaluation ID '{evaluation_id}' không phải cho agent '{agent_id}'"}
        else:
            # Sử dụng đánh giá gần nhất
            agent_evaluations = [
                eval_result for eval_id, eval_result in self.evaluation_results.items()
                if eval_result.get("agent_id") == agent_id
            ]
            
            if not agent_evaluations:
                self.logger.error(f"Không tìm thấy kết quả đánh giá nào cho agent '{agent_id}'")
                return {"error": f"Không có đánh giá nào cho agent '{agent_id}'"}
            
            # Sắp xếp theo thời gian và lấy đánh giá gần nhất
            agent_evaluations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            evaluation = agent_evaluations[0]
            evaluation_id = evaluation.get("eval_id")
        
        self.logger.info(f"Phân tích điểm mạnh/yếu cho agent '{agent_id}' dựa trên đánh giá '{evaluation_id}'")
        
        # Thiết lập trọng số metrics
        default_metrics_importance = {
            "mean_reward": 1.0,
            "sharpe_ratio": 0.9,
            "win_rate": 0.8,
            "max_drawdown": 0.8,
            "volatility": 0.7,
            "sortino_ratio": 0.7,
            "calmar_ratio": 0.6,
            "average_trade": 0.5,
            "recovery_factor": 0.5,
            "profit_factor": 0.6
        }
        
        importance = default_metrics_importance.copy()
        if metrics_importance:
            importance.update(metrics_importance)
        
        # Thu thập tất cả các metrics từ đánh giá
        all_metrics = {}
        
        # Thêm các metrics cơ bản
        if "basic_metrics" in evaluation:
            all_metrics.update(evaluation["basic_metrics"])
        
        # Thêm các metrics nâng cao
        if "advanced_metrics" in evaluation:
            all_metrics.update(evaluation["advanced_metrics"])
        
        # Phân loại điểm mạnh/yếu dựa trên ngưỡng
        strengths = {}
        weaknesses = {}
        neutral = {}
        
        # Ngưỡng đánh giá
        thresholds = {
            "mean_reward": 0.0,              # Lớn hơn 0 là điểm mạnh
            "sharpe_ratio": 1.0,             # Lớn hơn 1 là điểm mạnh
            "sortino_ratio": 1.5,            # Lớn hơn 1.5 là điểm mạnh
            "win_rate": 0.55,                # Lớn hơn 55% là điểm mạnh
            "profit_factor": 1.2,            # Lớn hơn 1.2 là điểm mạnh
            "max_drawdown": -0.2,            # Nhỏ hơn 20% là điểm mạnh
            "volatility": 0.15,              # Nhỏ hơn 15% là điểm mạnh
            "calmar_ratio": 0.5,             # Lớn hơn 0.5 là điểm mạnh
            "average_trade": 0.0,            # Lớn hơn 0 là điểm mạnh
            "recovery_factor": 1.0           # Lớn hơn 1 là điểm mạnh
        }
        
        # Đánh giá từng metric
        for metric, value in all_metrics.items():
            if metric not in importance:
                continue
                
            # Lấy ngưỡng cho metric này
            threshold = thresholds.get(metric)
            if threshold is None:
                # Nếu không có ngưỡng cho metric này, bỏ qua
                continue
            
            # Phân loại theo metric và ngưỡng
            if metric in ["max_drawdown", "volatility"]:
                # Metrics mà giá trị càng nhỏ càng tốt
                if value < threshold:
                    strengths[metric] = value
                elif value > threshold * 1.5:  # Nếu vượt quá 150% ngưỡng, là điểm yếu
                    weaknesses[metric] = value
                else:
                    neutral[metric] = value
            else:
                # Metrics mà giá trị càng lớn càng tốt
                if value > threshold:
                    strengths[metric] = value
                elif value < threshold * 0.5:  # Nếu nhỏ hơn 50% ngưỡng, là điểm yếu
                    weaknesses[metric] = value
                else:
                    neutral[metric] = value
        
        # Tính điểm tổng thể
        strength_score = sum(importance.get(metric, 0.5) for metric in strengths)
        weakness_score = sum(importance.get(metric, 0.5) for metric in weaknesses)
        
        # Tạo phân tích
        analysis = {
            "agent_id": agent_id,
            "evaluation_id": evaluation_id,
            "timestamp": datetime.now().isoformat(),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "neutral": neutral,
            "strength_score": strength_score,
            "weakness_score": weakness_score,
            "overall_rating": strength_score - weakness_score
        }
        
        # Thêm đánh giá ngắn gọn
        if strength_score > weakness_score * 2:
            analysis["assessment"] = "Xuất sắc"
        elif strength_score > weakness_score:
            analysis["assessment"] = "Tốt"
        elif strength_score == weakness_score:
            analysis["assessment"] = "Trung bình"
        elif weakness_score > strength_score * 2:
            analysis["assessment"] = "Kém"
        else:
            analysis["assessment"] = "Cần cải thiện"
        
        self.logger.info(
            f"Đã hoàn thành phân tích cho agent '{agent_id}': "
            f"Strengths: {len(strengths)}, Weaknesses: {len(weaknesses)}, "
            f"Assessment: {analysis['assessment']}"
        )
        
        return analysis
    
    def suggest_improvements(
        self,
        agent_id: str,
        analysis: Optional[Dict[str, Any]] = None,
        improvement_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Đề xuất các cải tiến cho agent dựa trên phân tích điểm mạnh/yếu.
        
        Args:
            agent_id: ID của agent cần đề xuất cải tiến
            analysis: Phân tích điểm mạnh/yếu (None để tự động phân tích)
            improvement_areas: Danh sách các lĩnh vực cần tập trung cải thiện
            
        Returns:
            Dict chứa các đề xuất cải tiến
        """
        # Nếu không có phân tích, tự động phân tích
        if analysis is None:
            analysis = self.analyze_agent_strengths_weaknesses(agent_id)
            
            if "error" in analysis:
                return analysis
        
        # Kiểm tra xem phân tích có phải cho agent này không
        if analysis.get("agent_id") != agent_id:
            self.logger.error(f"Phân tích không phải cho agent '{agent_id}'")
            return {"error": f"Phân tích không phải cho agent '{agent_id}'"}
        
        # Lấy thông tin agent
        agent_info = next((a for a in self.agents if isinstance(a, dict) and a.get("id") == agent_id), None)
        
        if agent_info is None:
            self.logger.error(f"Không tìm thấy agent với ID '{agent_id}'")
            return {"error": f"Agent ID '{agent_id}' không tồn tại"}
        
        agent = agent_info["agent"]
        agent_class = agent.__class__.__name__
        
        self.logger.info(f"Đề xuất cải tiến cho agent '{agent_id}' ({agent_class})")
        
        # Xác định các lĩnh vực cần cải thiện
        weaknesses = analysis.get("weaknesses", {})
        
        if not weaknesses and not improvement_areas:
            return {
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Agent đang hoạt động tốt, không có điểm yếu rõ ràng để cải thiện",
                "improvements": []
            }
        
        # Chuẩn bị đề xuất cải tiến
        improvements = []
        
        # Đề xuất cải tiến dựa trên điểm yếu
        improvement_templates = {
            "mean_reward": {
                "description": "Cải thiện phần thưởng trung bình",
                "suggestions": [
                    "Điều chỉnh hàm phần thưởng để phản ánh tốt hơn mục tiêu tối ưu hóa",
                    "Thử nghiệm với tỷ lệ học khác nhau để tìm giá trị tối ưu",
                    "Tăng số lượng episode huấn luyện để agent học hiệu quả hơn"
                ]
            },
            "sharpe_ratio": {
                "description": "Cải thiện tỷ lệ Sharpe (cân bằng lợi nhuận và rủi ro)",
                "suggestions": [
                    "Thêm điều kiện dừng lỗ tự động để giảm biến động",
                    "Điều chỉnh chiến lược để cân bằng giữa lợi nhuận và rủi ro",
                    "Thêm thành phần kiểm soát rủi ro vào hàm phần thưởng"
                ]
            },
            "sortino_ratio": {
                "description": "Cải thiện tỷ lệ Sortino (giảm biến động tiêu cực)",
                "suggestions": [
                    "Tập trung vào giảm các giao dịch thua lỗ thay vì tối đa hóa lợi nhuận",
                    "Thêm phạt nặng cho các quyết định dẫn đến lỗ lớn trong hàm phần thưởng",
                    "Điều chỉnh chính sách khám phá để đưa ra ít quyết định rủi ro hơn"
                ]
            },
            "win_rate": {
                "description": "Cải thiện tỷ lệ thắng",
                "suggestions": [
                    "Điều chỉnh ngưỡng quyết định để tăng độ chính xác của dự đoán",
                    "Thêm bộ lọc xu hướng để chỉ giao dịch khi có tín hiệu mạnh",
                    "Tăng kích thước memory buffer để agent học từ nhiều tình huống hơn"
                ]
            },
            "max_drawdown": {
                "description": "Giảm drawdown tối đa",
                "suggestions": [
                    "Thêm cơ chế dừng lỗ tự động để giới hạn thua lỗ",
                    "Điều chỉnh kích thước vị thế để phù hợp với mức rủi ro chấp nhận được",
                    "Thêm cơ chế phòng thủ khi thị trường biến động mạnh"
                ]
            },
            "volatility": {
                "description": "Giảm biến động trong hiệu suất",
                "suggestions": [
                    "Đa dạng hóa chiến lược giao dịch để giảm phụ thuộc vào một mẫu hình",
                    "Thêm cơ chế cân bằng danh mục đầu tư tự động",
                    "Giảm kích thước vị thế để hạn chế tác động của các giao dịch đơn lẻ"
                ]
            },
            "average_trade": {
                "description": "Cải thiện lợi nhuận trung bình mỗi giao dịch",
                "suggestions": [
                    "Tối ưu hóa chiến lược chốt lời để nắm giữ các vị thế có lãi lâu hơn",
                    "Thêm bộ lọc để loại bỏ các tín hiệu giao dịch yếu",
                    "Điều chỉnh tham số đầu vào để tập trung vào cơ hội lợi nhuận cao hơn"
                ]
            }
        }
        
        # Xử lý từng lĩnh vực cần cải thiện
        areas_to_improve = set(improvement_areas or []) | set(weaknesses.keys())
        
        for area in areas_to_improve:
            if area in improvement_templates:
                template = improvement_templates[area]
                current_value = weaknesses.get(area, "N/A")
                
                improvement = {
                    "area": area,
                    "current_value": current_value,
                    "description": template["description"],
                    "suggestions": template["suggestions"]
                }
                
                # Thêm đề xuất cụ thể cho từng loại agent
                if agent_class == "DQNAgent":
                    if area in ["mean_reward", "win_rate"]:
                        improvement["agent_specific"] = [
                            "Tăng kích thước mạng neural network",
                            "Thử các biến thể như Double DQN hoặc Dueling DQN",
                            "Điều chỉnh tham số epsilon để cân bằng khám phá và khai thác"
                        ]
                elif agent_class == "PPOAgent":
                    if area in ["mean_reward", "win_rate"]:
                        improvement["agent_specific"] = [
                            "Điều chỉnh tham số clip epsilon để cân bằng cập nhật policy",
                            "Thử nghiệm với các giá trị entropy coefficient khác nhau",
                            "Tăng số bước huấn luyện mỗi batch để cải thiện hội tụ"
                        ]
                elif agent_class == "A2CAgent":
                    if area in ["mean_reward", "win_rate"]:
                        improvement["agent_specific"] = [
                            "Thêm cơ chế entropy regularization để khuyến khích khám phá",
                            "Điều chỉnh tỷ lệ giữa actor loss và critic loss",
                            "Tăng số worker để cải thiện đa dạng kinh nghiệm"
                        ]
                
                improvements.append(improvement)
        
        # Đề xuất chung
        general_improvements = [
            {
                "area": "feature_engineering",
                "description": "Cải thiện đặc trưng đầu vào",
                "suggestions": [
                    "Thêm các indicator kỹ thuật mới để cung cấp tín hiệu tốt hơn",
                    "Chuẩn hóa dữ liệu đầu vào để cải thiện quá trình học",
                    "Thử nghiệm với các cửa sổ thời gian khác nhau để tìm tối ưu"
                ]
            },
            {
                "area": "hyperparameter_tuning",
                "description": "Tối ưu hóa siêu tham số",
                "suggestions": [
                    "Thực hiện tìm kiếm lưới hoặc tìm kiếm ngẫu nhiên trên không gian siêu tham số",
                    "Thử nghiệm với các cấu trúc mạng neural network khác nhau",
                    "Điều chỉnh tỷ lệ học và hệ số giảm phần thưởng"
                ]
            }
        ]
        
        # Thêm đề xuất chung nếu không có đề xuất cụ thể
        if not improvements:
            improvements = general_improvements
        else:
            # Thêm đề xuất chung nếu không trùng với các đề xuất cụ thể
            for gen_imp in general_improvements:
                if not any(imp["area"] == gen_imp["area"] for imp in improvements):
                    improvements.append(gen_imp)
        
        # Tạo kết quả
        result = {
            "agent_id": agent_id,
            "agent_class": agent_class,
            "timestamp": datetime.now().isoformat(),
            "weaknesses": weaknesses,
            "improvements": improvements
        }
        
        self.logger.info(f"Đã đề xuất {len(improvements)} cải tiến cho agent '{agent_id}'")
        
        return result
    
    def track_agent_progress(
        self,
        agent_id: str,
        env_id: str,
        evaluations: Optional[List[Dict[str, Any]]] = None,
        metrics_to_track: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Theo dõi sự tiến bộ của agent qua nhiều lần đánh giá.
        
        Args:
            agent_id: ID của agent cần theo dõi
            env_id: ID của môi trường đánh giá
            evaluations: Danh sách các kết quả đánh giá (None để sử dụng từ lịch sử)
            metrics_to_track: Danh sách các metrics cần theo dõi
            
        Returns:
            Dict chứa thông tin về tiến bộ của agent
        """
        # Thiết lập metrics cần theo dõi
        if metrics_to_track is None:
            metrics_to_track = ["mean_reward", "win_rate", "sharpe_ratio", "max_drawdown"]
        
        # Nếu không có evaluations, lấy từ lịch sử
        if evaluations is None:
            evaluations = [
                eval_result for eval_id, eval_result in self.evaluation_results.items()
                if eval_result.get("agent_id") == agent_id and eval_result.get("env_id") == env_id
            ]
        
        if not evaluations:
            self.logger.error(f"Không tìm thấy kết quả đánh giá nào cho agent '{agent_id}' trên môi trường '{env_id}'")
            return {"error": f"Không có đánh giá nào cho agent '{agent_id}' trên môi trường '{env_id}'"}
        
        # Sắp xếp evaluations theo thời gian
        evaluations.sort(key=lambda x: x.get("timestamp", ""))
        
        # Trích xuất dữ liệu cho từng metric
        metrics_history = {metric: [] for metric in metrics_to_track}
        timestamps = []
        
        for eval_result in evaluations:
            timestamps.append(eval_result.get("timestamp", ""))
            
            # Lấy giá trị cho từng metric
            for metric in metrics_to_track:
                # Tìm giá trị metric từ kết quả đánh giá
                metric_value = None
                
                if metric == "mean_reward":
                    if "basic_metrics" in eval_result and "mean_reward" in eval_result["basic_metrics"]:
                        metric_value = eval_result["basic_metrics"]["mean_reward"]
                elif "advanced_metrics" in eval_result and metric in eval_result["advanced_metrics"]:
                    metric_value = eval_result["advanced_metrics"][metric]
                elif "basic_metrics" in eval_result and metric in eval_result["basic_metrics"]:
                    metric_value = eval_result["basic_metrics"][metric]
                
                metrics_history[metric].append(metric_value if metric_value is not None else None)
        
        # Tính toán sự thay đổi và xu hướng
        changes = {}
        trends = {}
        
        for metric, values in metrics_history.items():
            # Lọc bỏ các giá trị None
            valid_values = [v for v in values if v is not None]
            
            if len(valid_values) < 2:
                changes[metric] = None
                trends[metric] = "unknown"
                continue
            
            # Tính % thay đổi từ đầu đến cuối
            first_value = valid_values[0]
            last_value = valid_values[-1]
            
            if first_value == 0:
                # Tránh chia cho 0
                change_percent = float('inf') if last_value > 0 else (float('-inf') if last_value < 0 else 0)
            else:
                change_percent = (last_value - first_value) / abs(first_value) * 100
            
            changes[metric] = {
                "first_value": first_value,
                "last_value": last_value,
                "absolute_change": last_value - first_value,
                "percent_change": change_percent
            }
            
            # Xác định xu hướng
            if len(valid_values) >= 3:
                # Tính xu hướng đơn giản dựa trên độ dốc của đường thẳng hồi quy
                x = list(range(len(valid_values)))
                # Tính hệ số góc của đường thẳng hồi quy
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(valid_values)
                sum_xx = sum(xi * xi for xi in x)
                sum_xy = sum(xi * yi for xi, yi in zip(x, valid_values))
                
                # Công thức hồi quy tuyến tính
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                
                if slope > 0.05 * abs(first_value):
                    trends[metric] = "improving_strongly"
                elif slope > 0:
                    trends[metric] = "improving"
                elif slope < -0.05 * abs(first_value):
                    trends[metric] = "deteriorating_strongly"
                elif slope < 0:
                    trends[metric] = "deteriorating"
                else:
                    trends[metric] = "stable"
            else:
                # Nếu chỉ có 2 điểm, xác định xu hướng đơn giản
                if change_percent > 10:
                    trends[metric] = "improving_strongly"
                elif change_percent > 0:
                    trends[metric] = "improving"
                elif change_percent < -10:
                    trends[metric] = "deteriorating_strongly"
                elif change_percent < 0:
                    trends[metric] = "deteriorating"
                else:
                    trends[metric] = "stable"
        
        # Tạo kết quả
        result = {
            "agent_id": agent_id,
            "env_id": env_id,
            "tracking_period": {
                "start": timestamps[0] if timestamps else None,
                "end": timestamps[-1] if timestamps else None,
                "num_evaluations": len(evaluations)
            },
            "metrics_history": metrics_history,
            "timestamps": timestamps,
            "changes": changes,
            "trends": trends
        }
        
        # Thêm đánh giá tổng thể
        # Đếm số metrics cải thiện/giảm sút
        improving_metrics = sum(1 for trend in trends.values() if trend in ["improving", "improving_strongly"])
        deteriorating_metrics = sum(1 for trend in trends.values() if trend in ["deteriorating", "deteriorating_strongly"])
        
        if improving_metrics > deteriorating_metrics:
            result["overall_trend"] = "improving"
        elif improving_metrics < deteriorating_metrics:
            result["overall_trend"] = "deteriorating"
        else:
            result["overall_trend"] = "mixed"
        
        self.logger.info(
            f"Đã theo dõi tiến bộ của agent '{agent_id}' qua {len(evaluations)} đánh giá. "
            f"Xu hướng tổng thể: {result['overall_trend']}"
        )
        
        return result
    
    def _save_evaluation_results(self, eval_id: str, results: Dict[str, Any]) -> str:
        """
        Lưu kết quả đánh giá vào file.
        
        Args:
            eval_id: ID của đánh giá
            results: Kết quả đánh giá
            
        Returns:
            Đường dẫn file đã lưu
        """
        # Tạo thư mục evaluations nếu chưa tồn tại
        eval_dir = self.output_dir / "evaluations"
        eval_dir.mkdir(exist_ok=True, parents=True)
        
        # Tạo file path
        file_path = eval_dir / f"{eval_id}.json"
        
        # Lưu kết quả vào file JSON
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                import json
                json.dump(results, f, indent=4, ensure_ascii=False, default=str)
            
            self.logger.debug(f"Đã lưu kết quả đánh giá vào {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu kết quả đánh giá: {str(e)}")
            return ""
    
    def _save_comparison_results(self, comparison_id: str, results: Dict[str, Any]) -> str:
        """
        Lưu kết quả so sánh vào file.
        
        Args:
            comparison_id: ID của so sánh
            results: Kết quả so sánh
            
        Returns:
            Đường dẫn file đã lưu
        """
        # Tạo thư mục comparisons nếu chưa tồn tại
        comparison_dir = self.output_dir / "comparisons"
        comparison_dir.mkdir(exist_ok=True, parents=True)
        
        # Tạo file path
        file_path = comparison_dir / f"{comparison_id}.json"
        
        # Lưu kết quả vào file JSON
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                import json
                json.dump(results, f, indent=4, ensure_ascii=False, default=str)
            
            self.logger.debug(f"Đã lưu kết quả so sánh vào {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu kết quả so sánh: {str(e)}")
            return ""
    
    def _load_evaluation_results(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Tải kết quả đánh giá từ file.
        
        Args:
            file_path: Đường dẫn file
            
        Returns:
            Dict chứa kết quả đánh giá
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                import json
                results = json.load(f)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải kết quả đánh giá từ {file_path}: {str(e)}")
            return {}
    
    def _bootstrap_metrics(
        self,
        rewards: List[float],
        num_samples: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """
        Tính toán khoảng tin cậy cho các metrics bằng phương pháp bootstrap.
        
        Args:
            rewards: Danh sách phần thưởng
            num_samples: Số lượng mẫu bootstrap
            confidence_level: Mức độ tin cậy
            
        Returns:
            Dict chứa khoảng tin cậy cho các metrics
        """
        if len(rewards) < 2:
            self.logger.warning("Không đủ dữ liệu để thực hiện bootstrap")
            return {}
        
        try:
            # Tạo các mẫu bootstrap
            bootstrap_samples = []
            for _ in range(num_samples):
                # Lấy mẫu với thay thế
                sample = np.random.choice(rewards, size=len(rewards), replace=True)
                bootstrap_samples.append(sample)
            
            # Tính toán các metrics cho từng mẫu
            bootstrap_metrics = {
                "mean_reward": [],
                "std_reward": [],
                "median_reward": []
            }
            
            for sample in bootstrap_samples:
                bootstrap_metrics["mean_reward"].append(np.mean(sample))
                bootstrap_metrics["std_reward"].append(np.std(sample))
                bootstrap_metrics["median_reward"].append(np.median(sample))
            
            # Tính các phân vị cho khoảng tin cậy
            alpha = 1 - confidence_level
            lower_percentile = alpha / 2 * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            # Tính khoảng tin cậy cho từng metric
            confidence_intervals = {}
            
            for metric, values in bootstrap_metrics.items():
                lower_bound = np.percentile(values, lower_percentile)
                upper_bound = np.percentile(values, upper_percentile)
                mean_value = np.mean(values)
                
                confidence_intervals[metric] = {
                    "mean": mean_value,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "confidence_level": confidence_level
                }
            
            return confidence_intervals
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thực hiện bootstrap: {str(e)}")
            return {}
    
    def generate_report(
        self,
        agent_id: str,
        env_id: Optional[str] = None,
        evaluation_id: Optional[str] = None,
        include_analysis: bool = True,
        include_improvements: bool = True,
        include_progress: bool = True,
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Tạo báo cáo tổng hợp về agent.
        
        Args:
            agent_id: ID của agent cần báo cáo
            env_id: ID của môi trường đánh giá (None để sử dụng tất cả môi trường)
            evaluation_id: ID của đánh giá cụ thể (None để sử dụng đánh giá gần nhất)
            include_analysis: Bao gồm phân tích điểm mạnh/yếu
            include_improvements: Bao gồm đề xuất cải tiến
            include_progress: Bao gồm theo dõi tiến bộ
            output_format: Định dạng đầu ra ("markdown", "json", "html")
            
        Returns:
            Dict chứa báo cáo và đường dẫn file
        """
        # Kiểm tra agent
        agent_info = next((a for a in self.agents if isinstance(a, dict) and a.get("id") == agent_id), None)
        
        if agent_info is None:
            self.logger.error(f"Không tìm thấy agent với ID '{agent_id}'")
            return {"error": f"Agent ID '{agent_id}' không tồn tại"}
        
        agent = agent_info["agent"]
        agent_class = agent.__class__.__name__
        
        # Tìm kết quả đánh giá
        if evaluation_id is not None:
            # Sử dụng đánh giá cụ thể
            if evaluation_id not in self.evaluation_results:
                self.logger.error(f"Không tìm thấy kết quả đánh giá với ID '{evaluation_id}'")
                return {"error": f"Evaluation ID '{evaluation_id}' không tồn tại"}
                
            evaluations = [self.evaluation_results[evaluation_id]]
            
            # Kiểm tra xem đánh giá có phải cho agent này không
            if evaluations[0].get("agent_id") != agent_id:
                self.logger.error(f"Evaluation ID '{evaluation_id}' không phải cho agent '{agent_id}'")
                return {"error": f"Evaluation ID '{evaluation_id}' không phải cho agent '{agent_id}'"}
        else:
            # Sử dụng tất cả đánh giá cho agent này
            if env_id is not None:
                # Lọc theo môi trường
                evaluations = [
                    eval_result for eval_id, eval_result in self.evaluation_results.items()
                    if eval_result.get("agent_id") == agent_id and eval_result.get("env_id") == env_id
                ]
            else:
                # Tất cả môi trường
                evaluations = [
                    eval_result for eval_id, eval_result in self.evaluation_results.items()
                    if eval_result.get("agent_id") == agent_id
                ]
        
        if not evaluations:
            self.logger.error(f"Không tìm thấy kết quả đánh giá nào cho agent '{agent_id}'")
            return {"error": f"Không có đánh giá nào cho agent '{agent_id}'"}
        
        # Sắp xếp evaluations theo thời gian
        evaluations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        latest_evaluation = evaluations[0]
        
        # Thực hiện phân tích nếu được yêu cầu
        analysis = None
        if include_analysis:
            analysis = self.analyze_agent_strengths_weaknesses(
                agent_id=agent_id,
                evaluation_id=latest_evaluation.get("eval_id")
            )
        
        # Đề xuất cải tiến nếu được yêu cầu
        improvements = None
        if include_improvements and analysis:
            improvements = self.suggest_improvements(
                agent_id=agent_id,
                analysis=analysis
            )
        
        # Theo dõi tiến bộ nếu được yêu cầu
        progress = None
        if include_progress and len(evaluations) > 1:
            # Nếu có nhiều đánh giá cho cùng một môi trường
            if env_id is not None:
                progress = self.track_agent_progress(
                    agent_id=agent_id,
                    env_id=env_id,
                    evaluations=evaluations
                )
            else:
                # Nhóm đánh giá theo môi trường
                env_evaluations = {}
                for eval_result in evaluations:
                    env = eval_result.get("env_id")
                    if env not in env_evaluations:
                        env_evaluations[env] = []
                    env_evaluations[env].append(eval_result)
                
                # Theo dõi tiến bộ cho môi trường có nhiều đánh giá nhất
                most_evaluations_env = max(env_evaluations.items(), key=lambda x: len(x[1]))
                if len(most_evaluations_env[1]) > 1:
                    progress = self.track_agent_progress(
                        agent_id=agent_id,
                        env_id=most_evaluations_env[0],
                        evaluations=most_evaluations_env[1]
                    )
        
        # Tạo nội dung báo cáo
        if output_format == "markdown":
            report_content = self._generate_markdown_report(
                agent_id=agent_id,
                agent_class=agent_class,
                evaluations=evaluations,
                analysis=analysis,
                improvements=improvements,
                progress=progress
            )
        elif output_format == "html":
            report_content = self._generate_html_report(
                agent_id=agent_id,
                agent_class=agent_class,
                evaluations=evaluations,
                analysis=analysis,
                improvements=improvements,
                progress=progress
            )
        else:  # json
            report_content = {
                "agent_id": agent_id,
                "agent_class": agent_class,
                "evaluations": evaluations,
                "analysis": analysis,
                "improvements": improvements,
                "progress": progress,
                "timestamp": datetime.now().isoformat()
            }
        
        # Lưu báo cáo vào file
        report_dir = self.output_dir / "reports"
        report_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "markdown":
            file_path = report_dir / f"report_{agent_id}_{timestamp}.md"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report_content)
        elif output_format == "html":
            file_path = report_dir / f"report_{agent_id}_{timestamp}.html"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report_content)
        else:  # json
            file_path = report_dir / f"report_{agent_id}_{timestamp}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                import json
                json.dump(report_content, f, indent=4, ensure_ascii=False, default=str)
        
        self.logger.info(f"Đã tạo báo cáo cho agent '{agent_id}' tại {file_path}")
        
        return {
            "agent_id": agent_id,
            "report_format": output_format,
            "report_path": str(file_path),
            "content": report_content
        }
    
    def _generate_markdown_report(
        self,
        agent_id: str,
        agent_class: str,
        evaluations: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]] = None,
        improvements: Optional[Dict[str, Any]] = None,
        progress: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Tạo báo cáo dạng Markdown về agent.
        
        Args:
            agent_id: ID của agent
            agent_class: Tên lớp agent
            evaluations: Danh sách các kết quả đánh giá
            analysis: Phân tích điểm mạnh/yếu
            improvements: Đề xuất cải tiến
            progress: Thông tin về tiến bộ
            
        Returns:
            Nội dung báo cáo dạng Markdown
        """
        # Lấy đánh giá gần nhất
        latest_evaluation = evaluations[0]
        
        # Bắt đầu tạo báo cáo
        report = f"# Báo cáo đánh giá Agent: {agent_id}\n\n"
        report += f"**Thời gian báo cáo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Thông tin cơ bản
        report += "## 1. Thông tin chung\n\n"
        report += f"- **Agent ID:** {agent_id}\n"
        report += f"- **Loại Agent:** {agent_class}\n"
        report += f"- **Môi trường đánh giá:** {latest_evaluation.get('env_id', 'N/A')}\n"
        report += f"- **Đánh giá mới nhất:** {latest_evaluation.get('timestamp', 'N/A')}\n"
        report += f"- **Số lượng đánh giá:** {len(evaluations)}\n\n"
        
        # Kết quả đánh giá mới nhất
        report += "## 2. Kết quả đánh giá mới nhất\n\n"
        
        # Các metrics cơ bản
        basic_metrics = latest_evaluation.get("basic_metrics", {})
        if basic_metrics:
            report += "### 2.1. Metrics cơ bản\n\n"
            report += "| Metric | Giá trị |\n"
            report += "|--------|--------|\n"
            
            for metric, value in basic_metrics.items():
                report += f"| {metric} | {value:.4f} |\n"
            
            report += "\n"
        
        # Các metrics nâng cao
        advanced_metrics = latest_evaluation.get("advanced_metrics", {})
        if advanced_metrics:
            report += "### 2.2. Metrics nâng cao\n\n"
            report += "| Metric | Giá trị |\n"
            report += "|--------|--------|\n"
            
            for metric, value in advanced_metrics.items():
                report += f"| {metric} | {value:.4f} |\n"
            
            report += "\n"
        
        # Phân tích điểm mạnh/yếu
        if analysis:
            report += "## 3. Phân tích điểm mạnh và điểm yếu\n\n"
            report += f"**Đánh giá tổng thể:** {analysis.get('assessment', 'N/A')}\n\n"
            
            # Điểm mạnh
            strengths = analysis.get("strengths", {})
            if strengths:
                report += "### 3.1. Điểm mạnh\n\n"
                report += "| Metric | Giá trị |\n"
                report += "|--------|--------|\n"
                
                for metric, value in strengths.items():
                    report += f"| {metric} | {value:.4f} |\n"
                
                report += "\n"
            
            # Điểm yếu
            weaknesses = analysis.get("weaknesses", {})
            if weaknesses:
                report += "### 3.2. Điểm yếu\n\n"
                report += "| Metric | Giá trị |\n"
                report += "|--------|--------|\n"
                
                for metric, value in weaknesses.items():
                    report += f"| {metric} | {value:.4f} |\n"
                
                report += "\n"
        
        # Đề xuất cải tiến
        if improvements:
            report += "## 4. Đề xuất cải tiến\n\n"
            
            for i, improvement in enumerate(improvements.get("improvements", []), 1):
                area = improvement.get("area", "")
                description = improvement.get("description", "")
                
                report += f"### 4.{i}. {description}\n\n"
                
                if "current_value" in improvement:
                    report += f"- **Giá trị hiện tại của {area}:** {improvement['current_value']:.4f}\n\n"
                
                report += "**Đề xuất:**\n\n"
                
                for suggestion in improvement.get("suggestions", []):
                    report += f"- {suggestion}\n"
                
                # Thêm đề xuất cụ thể cho loại agent nếu có
                if "agent_specific" in improvement:
                    report += f"\n**Đề xuất cụ thể cho {agent_class}:**\n\n"
                    
                    for suggestion in improvement.get("agent_specific", []):
                        report += f"- {suggestion}\n"
                
                report += "\n"
        
        # Theo dõi tiến bộ
        if progress:
            report += "## 5. Tiến bộ qua thời gian\n\n"
            
            # Thời gian theo dõi
            tracking_period = progress.get("tracking_period", {})
            start_time = tracking_period.get("start", "N/A")
            end_time = tracking_period.get("end", "N/A")
            num_evaluations = tracking_period.get("num_evaluations", 0)
            
            report += f"**Thời gian theo dõi:** Từ {start_time} đến {end_time}\n"
            report += f"**Số lượng đánh giá:** {num_evaluations}\n"
            report += f"**Xu hướng tổng thể:** {progress.get('overall_trend', 'N/A')}\n\n"
            
            # Các metrics thay đổi
            changes = progress.get("changes", {})
            if changes:
                report += "### 5.1. Thay đổi các metrics\n\n"
                report += "| Metric | Giá trị đầu | Giá trị cuối | Thay đổi | % Thay đổi | Xu hướng |\n"
                report += "|--------|------------|-------------|----------|------------|----------|\n"
                
                for metric, change in changes.items():
                    if change is None:
                        continue
                        
                    first_value = change.get("first_value", 0)
                    last_value = change.get("last_value", 0)
                    absolute_change = change.get("absolute_change", 0)
                    percent_change = change.get("percent_change", 0)
                    trend = progress.get("trends", {}).get(metric, "unknown")
                    
                    report += f"| {metric} | {first_value:.4f} | {last_value:.4f} | {absolute_change:.4f} | {percent_change:.2f}% | {trend} |\n"
                
                report += "\n"
        
        # Kết luận
        report += "## 6. Kết luận và tóm tắt\n\n"
        
        # Nếu có phân tích
        if analysis:
            assessment = analysis.get("assessment", "")
            if assessment in ["Xuất sắc", "Tốt"]:
                report += "Agent đang hoạt động tốt, có thể được sử dụng trong môi trường thực tế. "
            elif assessment == "Trung bình":
                report += "Agent hiện tại có hiệu suất trung bình, cần cải thiện thêm trước khi sử dụng trong môi trường thực tế. "
            else:
                report += "Agent chưa đạt hiệu suất đủ tốt, cần cải thiện đáng kể trước khi sử dụng. "
        
        # Nếu có theo dõi tiến bộ
        if progress:
            overall_trend = progress.get("overall_trend", "")
            if overall_trend == "improving":
                report += "Xu hướng hiệu suất đang cải thiện qua thời gian. "
            elif overall_trend == "deteriorating":
                report += "Xu hướng hiệu suất đang giảm sút, cần xem xét lại quá trình huấn luyện. "
            elif overall_trend == "mixed":
                report += "Xu hướng hiệu suất có cả cải thiện và giảm sút ở các metrics khác nhau. "
        
        # Thêm đề xuất tổng thể
        report += "\n\n**Đề xuất tổng thể:**\n\n"
        
        if improvements and improvements.get("improvements"):
            report += "1. Tập trung cải thiện các điểm yếu đã nêu ở trên.\n"
            report += "2. Tiếp tục theo dõi hiệu suất và đánh giá thường xuyên.\n"
        else:
            report += "1. Tiếp tục theo dõi hiệu suất và đánh giá thường xuyên.\n"
            report += "2. Thử nghiệm agent trong môi trường thử nghiệm trước khi triển khai thực tế.\n"
        
        return report
    
    def _generate_html_report(
        self,
        agent_id: str,
        agent_class: str,
        evaluations: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]] = None,
        improvements: Optional[Dict[str, Any]] = None,
        progress: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Tạo báo cáo dạng HTML về agent.
        
        Args:
            agent_id: ID của agent
            agent_class: Tên lớp agent
            evaluations: Danh sách các kết quả đánh giá
            analysis: Phân tích điểm mạnh/yếu
            improvements: Đề xuất cải tiến
            progress: Thông tin về tiến bộ
            
        Returns:
            Nội dung báo cáo dạng HTML
        """
        # Lấy đánh giá gần nhất
        latest_evaluation = evaluations[0]
        
        # Tạo báo cáo HTML cơ bản
        html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo cáo đánh giá Agent: {agent_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2980b9;
            margin-top: 30px;
        }}
        h3 {{
            color: #3498db;
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
        .strength {{
            color: #27ae60;
        }}
        .weakness {{
            color: #e74c3c;
        }}
        .improving {{
            color: #27ae60;
        }}
        .deteriorating {{
            color: #e74c3c;
        }}
        .mixed {{
            color: #f39c12;
        }}
        .conclusion {{
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>Báo cáo đánh giá Agent: {agent_id}</h1>
    <p><strong>Thời gian báo cáo:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>1. Thông tin chung</h2>
    <ul>
        <li><strong>Agent ID:</strong> {agent_id}</li>
        <li><strong>Loại Agent:</strong> {agent_class}</li>
        <li><strong>Môi trường đánh giá:</strong> {latest_evaluation.get('env_id', 'N/A')}</li>
        <li><strong>Đánh giá mới nhất:</strong> {latest_evaluation.get('timestamp', 'N/A')}</li>
        <li><strong>Số lượng đánh giá:</strong> {len(evaluations)}</li>
    </ul>
    
    <h2>2. Kết quả đánh giá mới nhất</h2>
"""
        
        # Các metrics cơ bản
        basic_metrics = latest_evaluation.get("basic_metrics", {})
        if basic_metrics:
            html += """
    <h3>2.1. Metrics cơ bản</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Giá trị</th>
        </tr>
"""
            
            for metric, value in basic_metrics.items():
                html += f"""
        <tr>
            <td>{metric}</td>
            <td>{value:.4f}</td>
        </tr>"""
            
            html += """
    </table>"""
        
        # Các metrics nâng cao
        advanced_metrics = latest_evaluation.get("advanced_metrics", {})
        if advanced_metrics:
            html += """
    <h3>2.2. Metrics nâng cao</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Giá trị</th>
        </tr>
"""
            
            for metric, value in advanced_metrics.items():
                html += f"""
        <tr>
            <td>{metric}</td>
            <td>{value:.4f}</td>
        </tr>"""
            
            html += """
    </table>"""
        
        # Phân tích điểm mạnh/yếu
        if analysis:
            html += f"""
    <h2>3. Phân tích điểm mạnh và điểm yếu</h2>
    <p><strong>Đánh giá tổng thể:</strong> {analysis.get('assessment', 'N/A')}</p>
"""
            
            # Điểm mạnh
            strengths = analysis.get("strengths", {})
            if strengths:
                html += """
    <h3>3.1. Điểm mạnh</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Giá trị</th>
        </tr>
"""
                
                for metric, value in strengths.items():
                    html += f"""
        <tr>
            <td>{metric}</td>
            <td class="strength">{value:.4f}</td>
        </tr>"""
                
                html += """
    </table>"""
            
            # Điểm yếu
            weaknesses = analysis.get("weaknesses", {})
            if weaknesses:
                html += """
    <h3>3.2. Điểm yếu</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Giá trị</th>
        </tr>
"""
                
                for metric, value in weaknesses.items():
                    html += f"""
        <tr>
            <td>{metric}</td>
            <td class="weakness">{value:.4f}</td>
        </tr>"""
                
                html += """
    </table>"""
        
        # Đề xuất cải tiến
        if improvements:
            html += """
    <h2>4. Đề xuất cải tiến</h2>
"""
            
            for i, improvement in enumerate(improvements.get("improvements", []), 1):
                area = improvement.get("area", "")
                description = improvement.get("description", "")
                
                html += f"""
    <h3>4.{i}. {description}</h3>"""
                
                if "current_value" in improvement:
                    html += f"""
    <p><strong>Giá trị hiện tại của {area}:</strong> {improvement['current_value']:.4f}</p>"""
                
                html += """
    <p><strong>Đề xuất:</strong></p>
    <ul>"""
                
                for suggestion in improvement.get("suggestions", []):
                    html += f"""
        <li>{suggestion}</li>"""
                
                html += """
    </ul>"""
                
                # Thêm đề xuất cụ thể cho loại agent nếu có
                if "agent_specific" in improvement:
                    html += f"""
    <p><strong>Đề xuất cụ thể cho {agent_class}:</strong></p>
    <ul>"""
                    
                    for suggestion in improvement.get("agent_specific", []):
                        html += f"""
        <li>{suggestion}</li>"""
                    
                    html += """
    </ul>"""
        
        # Theo dõi tiến bộ
        if progress:
            html += """
    <h2>5. Tiến bộ qua thời gian</h2>
"""
            
            # Thời gian theo dõi
            tracking_period = progress.get("tracking_period", {})
            start_time = tracking_period.get("start", "N/A")
            end_time = tracking_period.get("end", "N/A")
            num_evaluations = tracking_period.get("num_evaluations", 0)
            overall_trend = progress.get('overall_trend', 'N/A')
            trend_class = ""
            
            if overall_trend == "improving":
                trend_class = "improving"
            elif overall_trend == "deteriorating":
                trend_class = "deteriorating"
            elif overall_trend == "mixed":
                trend_class = "mixed"
            
            html += f"""
    <p><strong>Thời gian theo dõi:</strong> Từ {start_time} đến {end_time}</p>
    <p><strong>Số lượng đánh giá:</strong> {num_evaluations}</p>
    <p><strong>Xu hướng tổng thể:</strong> <span class="{trend_class}">{overall_trend}</span></p>
"""
            
            # Các metrics thay đổi
            changes = progress.get("changes", {})
            if changes:
                html += """
    <h3>5.1. Thay đổi các metrics</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Giá trị đầu</th>
            <th>Giá trị cuối</th>
            <th>Thay đổi</th>
            <th>% Thay đổi</th>
            <th>Xu hướng</th>
        </tr>
"""
                
                for metric, change in changes.items():
                    if change is None:
                        continue
                        
                    first_value = change.get("first_value", 0)
                    last_value = change.get("last_value", 0)
                    absolute_change = change.get("absolute_change", 0)
                    percent_change = change.get("percent_change", 0)
                    trend = progress.get("trends", {}).get(metric, "unknown")
                    
                    trend_class = ""
                    if trend in ["improving", "improving_strongly"]:
                        trend_class = "improving"
                    elif trend in ["deteriorating", "deteriorating_strongly"]:
                        trend_class = "deteriorating"
                    
                    html += f"""
        <tr>
            <td>{metric}</td>
            <td>{first_value:.4f}</td>
            <td>{last_value:.4f}</td>
            <td>{absolute_change:.4f}</td>
            <td>{percent_change:.2f}%</td>
            <td class="{trend_class}">{trend}</td>
        </tr>"""
                
                html += """
    </table>"""
        
        # Kết luận
        html += """
    <h2>6. Kết luận và tóm tắt</h2>
    <div class="conclusion">
"""
        
        # Nếu có phân tích
        if analysis:
            assessment = analysis.get("assessment", "")
            if assessment in ["Xuất sắc", "Tốt"]:
                html += """
        <p>Agent đang hoạt động tốt, có thể được sử dụng trong môi trường thực tế.</p>"""
            elif assessment == "Trung bình":
                html += """
        <p>Agent hiện tại có hiệu suất trung bình, cần cải thiện thêm trước khi sử dụng trong môi trường thực tế.</p>"""
            else:
                html += """
        <p>Agent chưa đạt hiệu suất đủ tốt, cần cải thiện đáng kể trước khi sử dụng.</p>"""
        
        # Nếu có theo dõi tiến bộ
        if progress:
            overall_trend = progress.get("overall_trend", "")
            if overall_trend == "improving":
                html += """
        <p>Xu hướng hiệu suất đang cải thiện qua thời gian.</p>"""
            elif overall_trend == "deteriorating":
                html += """
        <p>Xu hướng hiệu suất đang giảm sút, cần xem xét lại quá trình huấn luyện.</p>"""
            elif overall_trend == "mixed":
                html += """
        <p>Xu hướng hiệu suất có cả cải thiện và giảm sút ở các metrics khác nhau.</p>"""
        
        # Thêm đề xuất tổng thể
        html += """
        <p><strong>Đề xuất tổng thể:</strong></p>
        <ol>"""
        
        if improvements and improvements.get("improvements"):
            html += """
            <li>Tập trung cải thiện các điểm yếu đã nêu ở trên.</li>
            <li>Tiếp tục theo dõi hiệu suất và đánh giá thường xuyên.</li>"""
        else:
            html += """
            <li>Tiếp tục theo dõi hiệu suất và đánh giá thường xuyên.</li>
            <li>Thử nghiệm agent trong môi trường thử nghiệm trước khi triển khai thực tế.</li>"""
        
        html += """
        </ol>
    </div>
</body>
</html>"""
        
        return html