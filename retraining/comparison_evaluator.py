"""
So sánh và đánh giá mô hình.
File này định nghĩa lớp ComparisonEvaluator để so sánh các phiên bản 
khác nhau của mô hình, đánh giá hiệu suất và đưa ra khuyến nghị về việc triển khai.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import concurrent.futures
from functools import partial
import time
import statistics

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger
from config.system_config import get_system_config
from config.constants import BacktestMetric, RetrainingTrigger
from retraining.performance_tracker import PerformanceTracker
from retraining.model_updater import ModelUpdater
from backtesting.evaluation.performance_evaluator import PerformanceEvaluator
from models.agents.base_agent import BaseAgent
from environments.base_environment import BaseEnvironment

class ComparisonEvaluator:
    """
    So sánh và đánh giá hiệu suất của các phiên bản mô hình khác nhau.
    
    Lớp này cung cấp các công cụ để:
    1. So sánh hiệu suất của các mô hình trên cùng dữ liệu kiểm tra
    2. Phân tích sự khác biệt giữa các mô hình trong các điều kiện thị trường khác nhau
    3. Thực hiện các bài kiểm tra thống kê để xác định ý nghĩa của sự khác biệt
    4. Đưa ra khuyến nghị về việc nên triển khai mô hình nào
    """
    
    def __init__(
        self,
        agent_type: str,
        base_version: str,
        output_dir: Optional[Union[str, Path]] = None,
        metrics_to_compare: Optional[List[str]] = None,
        significance_threshold: float = 0.05,
        min_improvement_threshold: float = 0.05,
        evaluation_episodes: int = 50,
        evaluation_environments: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo ComparisonEvaluator.
        
        Args:
            agent_type: Loại agent đang đánh giá (dqn, ppo, a2c, etc.)
            base_version: Phiên bản cơ sở để so sánh với các phiên bản khác
            output_dir: Thư mục đầu ra cho kết quả đánh giá
            metrics_to_compare: Danh sách các số liệu cần so sánh
            significance_threshold: Ngưỡng cho kiểm định thống kê (p-value)
            min_improvement_threshold: Ngưỡng cải thiện tối thiểu để đáng xem xét
            evaluation_episodes: Số lượng episode để đánh giá mỗi mô hình
            evaluation_environments: Danh sách môi trường đánh giá
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("comparison_evaluator")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập các thông số
        self.agent_type = agent_type
        self.base_version = base_version

        # Lưu thông số đánh giá
        self.significance_threshold = significance_threshold
        self.min_improvement_threshold = min_improvement_threshold
        self.evaluation_episodes = evaluation_episodes
        self.evaluation_environments = evaluation_environments or ["default"]
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            self.output_dir = Path(self.system_config.get("model_dir", "./models")) / 'evaluations' / agent_type
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập các số liệu đánh giá mặc định nếu không được cung cấp
        if metrics_to_compare is None:
            self.metrics_to_compare = [
                BacktestMetric.SHARPE_RATIO.value,
                BacktestMetric.SORTINO_RATIO.value,
                BacktestMetric.WIN_RATE.value,
                BacktestMetric.PROFIT_FACTOR.value,
                BacktestMetric.MAX_DRAWDOWN.value,
                BacktestMetric.EXPECTANCY.value,
                "roi",
                "volatility",
                "consistency",
                "recovery_factor"
            ]
        else:
            self.metrics_to_compare = metrics_to_compare
        
        # Khởi tạo ModelUpdater để quản lý các phiên bản
        self.model_updater = ModelUpdater(
            agent_type=agent_type,
            model_version=base_version,
            output_dir=self.output_dir.parent / "versions",
            logger=self.logger
        )
        
        # Từ điển lưu trữ tất cả các phiên bản mô hình cần so sánh
        self.models_to_compare = {
            base_version: {"agent": None, "description": "Base model", "version": base_version}
        }
        
        # Lưu trữ kết quả đánh giá
        self.evaluation_results = {}
        self.comparison_results = {}
        self.statistical_tests = {}
        
        # Lưu trữ môi trường đánh giá
        self.environments = {}
        
        self.logger.info(f"Đã khởi tạo ComparisonEvaluator cho agent {agent_type}, phiên bản cơ sở: {base_version}")
    
    def add_model_to_compare(
        self,
        version: str,
        agent: Optional[BaseAgent] = None,
        model_path: Optional[Union[str, Path]] = None,
        description: Optional[str] = None
    ) -> bool:
        """
        Thêm một mô hình để so sánh.
        
        Args:
            version: Phiên bản của mô hình
            agent: Instance của agent (nếu đã được tải)
            model_path: Đường dẫn đến file mô hình (nếu chưa tải)
            description: Mô tả về mô hình
            
        Returns:
            True nếu thêm thành công, False nếu không
        """
        try:
            # Kiểm tra xem phiên bản này đã có trong danh sách chưa
            if version in self.models_to_compare:
                # Nếu agent được cung cấp, cập nhật
                if agent is not None:
                    self.models_to_compare[version]["agent"] = agent
                    
                # Nếu description được cung cấp, cập nhật
                if description is not None:
                    self.models_to_compare[version]["description"] = description
                    
                self.logger.info(f"Đã cập nhật thông tin cho mô hình {version}")
                return True
            
            # Thêm mô hình mới
            model_info = {
                "agent": agent,
                "model_path": str(model_path) if model_path else None,
                "description": description or f"Model version {version}",
                "version": version
            }
            
            # Kiểm tra tính hợp lệ của model_path
            if model_path is not None and not Path(model_path).exists():
                self.logger.warning(f"File mô hình không tồn tại: {model_path}")
                model_info["warning"] = "Model path does not exist"
            
            # Đăng ký với model_updater nếu chưa có
            if version not in self.model_updater.versions:
                register_result = self.model_updater.register_new_version(
                    agent=agent,
                    version=version,
                    model_path=model_path,
                    metadata={"description": model_info["description"]}
                )
                
                if not register_result["success"]:
                    self.logger.warning(f"Không thể đăng ký phiên bản {version} với model_updater: {register_result['message']}")
            
            # Thêm vào danh sách mô hình
            self.models_to_compare[version] = model_info
            
            self.logger.info(f"Đã thêm mô hình {version} để so sánh")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thêm mô hình {version}: {str(e)}")
            return False
    
    def setup_evaluation_environment(
        self,
        env_name: str,
        env_creator: Callable[[], BaseEnvironment],
        env_config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> bool:
        """
        Thiết lập môi trường đánh giá.
        
        Args:
            env_name: Tên của môi trường
            env_creator: Hàm tạo môi trường
            env_config: Cấu hình cho môi trường
            description: Mô tả về môi trường
            
        Returns:
            True nếu thiết lập thành công, False nếu không
        """
        try:
            # Lưu thông tin môi trường
            env_info = {
                "creator": env_creator,
                "config": env_config or {},
                "description": description or f"Evaluation environment {env_name}",
                "name": env_name
            }
            
            # Thêm vào danh sách môi trường
            self.environments[env_name] = env_info
            
            self.logger.info(f"Đã thiết lập môi trường đánh giá {env_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thiết lập môi trường đánh giá {env_name}: {str(e)}")
            return False
    
    def load_models(self) -> Dict[str, bool]:
        """
        Tải tất cả các mô hình từ đường dẫn đã chỉ định.
        
        Returns:
            Dict map từ phiên bản đến trạng thái tải (True/False)
        """
        load_results = {}
        
        for version, model_info in self.models_to_compare.items():
            # Bỏ qua nếu đã có agent
            if model_info["agent"] is not None:
                load_results[version] = True
                continue
            
            # Tải mô hình từ model_updater
            load_result = self.model_updater.load_model(
                version=version,
                agent=model_info.get("agent")
            )
            
            if load_result["success"]:
                load_results[version] = True
                self.logger.info(f"Đã tải mô hình {version} thành công")
            else:
                load_results[version] = False
                self.logger.warning(f"Không thể tải mô hình {version}: {load_result['message']}")
        
        return load_results
    
    def evaluate_model(
        self,
        version: str,
        env_name: str,
        num_episodes: Optional[int] = None,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Đánh giá một mô hình cụ thể trên một môi trường.
        
        Args:
            version: Phiên bản của mô hình cần đánh giá
            env_name: Tên của môi trường đánh giá
            num_episodes: Số lượng episode để đánh giá
            render: Hiển thị môi trường trong quá trình đánh giá
            
        Returns:
            Dict chứa kết quả đánh giá
        """
        if version not in self.models_to_compare:
            self.logger.error(f"Phiên bản {version} không có trong danh sách mô hình để so sánh")
            return {
                "success": False,
                "message": f"Phiên bản {version} không có trong danh sách mô hình để so sánh"
            }
        
        if env_name not in self.environments:
            self.logger.error(f"Môi trường {env_name} không có trong danh sách môi trường đánh giá")
            return {
                "success": False,
                "message": f"Môi trường {env_name} không có trong danh sách môi trường đánh giá"
            }
        
        model_info = self.models_to_compare[version]
        env_info = self.environments[env_name]
        
        # Lấy agent
        agent = model_info["agent"]
        
        if agent is None:
            self.logger.error(f"Agent cho phiên bản {version} chưa được tải")
            return {
                "success": False,
                "message": f"Agent cho phiên bản {version} chưa được tải"
            }
        
        # Tạo môi trường mới
        try:
            env = env_info["creator"]()
            
            # Thiết lập các cấu hình môi trường
            for key, value in env_info["config"].items():
                if hasattr(env, key):
                    setattr(env, key, value)
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo môi trường {env_name}: {str(e)}")
            return {
                "success": False,
                "message": f"Lỗi khi tạo môi trường {env_name}: {str(e)}"
            }
        
        # Thiết lập số lượng episode
        if num_episodes is None:
            num_episodes = self.evaluation_episodes
        
        # Đánh giá agent
        start_time = time.time()
        
        try:
            # Sử dụng phương thức đánh giá của agent
            rewards = agent.evaluate(
                env=env,
                num_episodes=num_episodes,
                render=render
            )
            
            # Tính toán các số liệu
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            median_reward = np.median(rewards)
            
            # Tính các số liệu khác
            # Win rate: tỷ lệ các episode có reward > 0
            win_rate = np.mean([1 if r > 0 else 0 for r in rewards])
            
            # Consistency: tỷ lệ giữa trung vị và trung bình
            consistency = median_reward / mean_reward if mean_reward != 0 else 0
            
            # Volatility: độ lệch chuẩn chia cho trung bình tuyệt đối
            volatility = std_reward / abs(mean_reward) if mean_reward != 0 else float('inf')
            
            # Recovery factor: trung bình chia cho drawdown tối đa
            max_drawdown = self._calculate_max_drawdown(rewards)
            recovery_factor = abs(mean_reward) / max_drawdown if max_drawdown > 0 else float('inf')
            
            evaluation_time = time.time() - start_time
            
            # Lưu kết quả đánh giá
            result = {
                "success": True,
                "version": version,
                "env_name": env_name,
                "num_episodes": num_episodes,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "min_reward": min_reward,
                    "max_reward": max_reward,
                    "median_reward": median_reward,
                    BacktestMetric.WIN_RATE.value: win_rate,
                    "consistency": consistency,
                    "volatility": volatility,
                    BacktestMetric.MAX_DRAWDOWN.value: max_drawdown,
                    "recovery_factor": recovery_factor
                },
                "raw_rewards": rewards,
                "evaluation_time": evaluation_time
            }
            
            # Thêm vào kết quả đánh giá
            if version not in self.evaluation_results:
                self.evaluation_results[version] = {}
            
            self.evaluation_results[version][env_name] = result
            
            self.logger.info(
                f"Đánh giá {version} trên {env_name}: "
                f"Mean Reward: {mean_reward:.2f}, "
                f"Win Rate: {win_rate:.2f}, "
                f"Volatility: {volatility:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi đánh giá {version} trên {env_name}: {str(e)}")
            return {
                "success": False,
                "message": f"Lỗi khi đánh giá {version} trên {env_name}: {str(e)}",
                "version": version,
                "env_name": env_name
            }
    
    def evaluate_all_models(
        self,
        environments: Optional[List[str]] = None,
        num_episodes: Optional[int] = None,
        parallel: bool = True,
        max_workers: int = 4
    ) -> Dict[str, Dict[str, Any]]:
        """
        Đánh giá tất cả các mô hình trên tất cả các môi trường.
        
        Args:
            environments: Danh sách tên môi trường (None để sử dụng tất cả)
            num_episodes: Số lượng episode để đánh giá
            parallel: Sử dụng đánh giá song song
            max_workers: Số lượng worker tối đa khi đánh giá song song
            
        Returns:
            Dict chứa kết quả đánh giá
        """
        # Nếu không có danh sách môi trường, sử dụng tất cả
        if environments is None:
            environments = list(self.environments.keys())
        
        # Lọc các môi trường hợp lệ
        valid_environments = [env for env in environments if env in self.environments]
        
        if not valid_environments:
            self.logger.error("Không có môi trường hợp lệ nào để đánh giá")
            return {}
        
        # Tạo danh sách công việc đánh giá
        evaluation_tasks = []
        
        for version in self.models_to_compare.keys():
            for env_name in valid_environments:
                evaluation_tasks.append((version, env_name))
        
        results = {}
        
        # Đánh giá song song hoặc tuần tự
        if parallel and len(evaluation_tasks) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Tạo future cho mỗi công việc đánh giá
                futures = {
                    executor.submit(self.evaluate_model, version, env_name, num_episodes): (version, env_name)
                    for version, env_name in evaluation_tasks
                }
                
                # Xử lý kết quả
                for future in concurrent.futures.as_completed(futures):
                    version, env_name = futures[future]
                    try:
                        result = future.result()
                        
                        if version not in results:
                            results[version] = {}
                        
                        results[version][env_name] = result
                    except Exception as e:
                        self.logger.error(f"Lỗi khi đánh giá {version} trên {env_name}: {str(e)}")
        else:
            # Đánh giá tuần tự
            for version, env_name in evaluation_tasks:
                result = self.evaluate_model(version, env_name, num_episodes)
                
                if version not in results:
                    results[version] = {}
                
                results[version][env_name] = result
        
        return results
    
    def compare_models(
        self,
        target_version: Optional[str] = None,
        base_version: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        environments: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        So sánh hiệu suất giữa hai mô hình.
        
        Args:
            target_version: Phiên bản mô hình cần so sánh
            base_version: Phiên bản mô hình cơ sở (mặc định sử dụng self.base_version)
            metrics: Danh sách số liệu cần so sánh
            environments: Danh sách môi trường cần so sánh
            
        Returns:
            Dict chứa kết quả so sánh
        """
        # Thiết lập giá trị mặc định
        if target_version is None:
            # Nếu không chỉ định target_version, so sánh tất cả với base_version
            target_versions = [v for v in self.models_to_compare.keys() if v != self.base_version]
        else:
            target_versions = [target_version]
        
        if base_version is None:
            base_version = self.base_version
        
        if metrics is None:
            metrics = self.metrics_to_compare
        
        if environments is None:
            environments = list(self.environments.keys())
        
        # Kiểm tra xem có đủ dữ liệu để so sánh không
        if base_version not in self.evaluation_results:
            self.logger.error(f"Chưa có dữ liệu đánh giá cho phiên bản cơ sở {base_version}")
            return {
                "success": False,
                "message": f"Chưa có dữ liệu đánh giá cho phiên bản cơ sở {base_version}"
            }
        
        comparison_results = {}
        
        for target in target_versions:
            if target not in self.evaluation_results:
                self.logger.warning(f"Chưa có dữ liệu đánh giá cho phiên bản {target}, bỏ qua")
                continue
            
            # So sánh trên từng môi trường
            env_comparisons = {}
            
            for env_name in environments:
                if env_name not in self.evaluation_results[base_version]:
                    self.logger.warning(f"Chưa có dữ liệu đánh giá cho {base_version} trên {env_name}, bỏ qua")
                    continue
                
                if env_name not in self.evaluation_results[target]:
                    self.logger.warning(f"Chưa có dữ liệu đánh giá cho {target} trên {env_name}, bỏ qua")
                    continue
                
                # Lấy dữ liệu đánh giá
                base_eval = self.evaluation_results[base_version][env_name]
                target_eval = self.evaluation_results[target][env_name]
                
                # So sánh từng số liệu
                metric_comparisons = {}
                
                for metric in metrics:
                    if metric not in base_eval["metrics"] or metric not in target_eval["metrics"]:
                        continue
                    
                    base_value = base_eval["metrics"][metric]
                    target_value = target_eval["metrics"][metric]
                    
                    # Tính phần trăm thay đổi
                    if base_value != 0:
                        percent_change = (target_value - base_value) / abs(base_value) * 100
                    else:
                        percent_change = float('inf') if target_value > 0 else float('-inf') if target_value < 0 else 0
                    
                    # Xác định có cải thiện hay không
                    # Với một số số liệu, giá trị nhỏ hơn là tốt hơn
                    if metric in [BacktestMetric.MAX_DRAWDOWN.value, "volatility"]:
                        is_improvement = target_value < base_value
                    else:
                        is_improvement = target_value > base_value
                    
                    # Đánh giá mức độ cải thiện
                    if is_improvement:
                        if abs(percent_change) >= self.min_improvement_threshold * 100:
                            improvement_level = "significant"
                        else:
                            improvement_level = "minor"
                    else:
                        if abs(percent_change) >= self.min_improvement_threshold * 100:
                            improvement_level = "regression"
                        else:
                            improvement_level = "neutral"
                    
                    # Thực hiện kiểm định thống kê nếu có dữ liệu thô
                    statistical_test = None
                    p_value = None
                    
                    if "raw_rewards" in base_eval and "raw_rewards" in target_eval:
                        # T-test cho các mẫu độc lập
                        from scipy import stats
                        try:
                            t_stat, p_value = stats.ttest_ind(
                                base_eval["raw_rewards"],
                                target_eval["raw_rewards"],
                                equal_var=False  # Không giả định phương sai bằng nhau
                            )
                            
                            statistical_test = {
                                "test": "t-test",
                                "t_stat": float(t_stat),
                                "p_value": float(p_value),
                                "significant": p_value < self.significance_threshold
                            }
                        except Exception as e:
                            self.logger.warning(f"Lỗi khi thực hiện kiểm định thống kê: {str(e)}")
                    
                    # Lưu kết quả so sánh cho số liệu này
                    metric_comparisons[metric] = {
                        "base_value": base_value,
                        "target_value": target_value,
                        "absolute_diff": target_value - base_value,
                        "percent_change": percent_change,
                        "is_improvement": is_improvement,
                        "improvement_level": improvement_level,
                        "statistical_test": statistical_test
                    }
                
                # Tính tổng quan so sánh cho môi trường này
                improvements_count = sum(1 for m in metric_comparisons.values() if m["is_improvement"])
                regressions_count = sum(1 for m in metric_comparisons.values() if not m["is_improvement"])
                
                significant_improvements = sum(1 for m in metric_comparisons.values() 
                                               if m["is_improvement"] and m["improvement_level"] == "significant")
                
                significant_regressions = sum(1 for m in metric_comparisons.values() 
                                              if not m["is_improvement"] and m["improvement_level"] == "regression")
                
                # Đánh giá tổng thể cho môi trường
                if significant_improvements > significant_regressions:
                    overall_assessment = "improvement"
                elif significant_regressions > significant_improvements:
                    overall_assessment = "regression"
                else:
                    if improvements_count > regressions_count:
                        overall_assessment = "slight_improvement"
                    elif regressions_count > improvements_count:
                        overall_assessment = "slight_regression"
                    else:
                        overall_assessment = "neutral"
                
                # Lưu kết quả so sánh cho môi trường này
                env_comparisons[env_name] = {
                    "metrics": metric_comparisons,
                    "improvements_count": improvements_count,
                    "regressions_count": regressions_count,
                    "significant_improvements": significant_improvements,
                    "significant_regressions": significant_regressions,
                    "overall_assessment": overall_assessment
                }
            
            # Tính tổng quan so sánh cho tất cả các môi trường
            if env_comparisons:
                # Đếm số môi trường có cải thiện tổng thể
                improved_environments = sum(1 for env_comp in env_comparisons.values() 
                                           if env_comp["overall_assessment"] in ["improvement", "slight_improvement"])
                
                # Đếm số môi trường có xuống cấp tổng thể
                regressed_environments = sum(1 for env_comp in env_comparisons.values() 
                                            if env_comp["overall_assessment"] in ["regression", "slight_regression"])
                
                # Đánh giá tổng thể cho tất cả các môi trường
                if improved_environments > regressed_environments:
                    overall_verdict = "improvement"
                elif regressed_environments > improved_environments:
                    overall_verdict = "regression"
                else:
                    overall_verdict = "neutral"
                
                # Tính điểm cải thiện tổng thể
                improvement_score = 0
                total_environments = len(env_comparisons)
                
                for env_comp in env_comparisons.values():
                    if env_comp["overall_assessment"] == "improvement":
                        improvement_score += 1.0
                    elif env_comp["overall_assessment"] == "slight_improvement":
                        improvement_score += 0.5
                    elif env_comp["overall_assessment"] == "slight_regression":
                        improvement_score -= 0.5
                    elif env_comp["overall_assessment"] == "regression":
                        improvement_score -= 1.0
                
                normalized_score = improvement_score / total_environments if total_environments > 0 else 0
                
                # Lưu kết quả so sánh cho phiên bản này
                comparison_results[target] = {
                    "base_version": base_version,
                    "target_version": target,
                    "environments": env_comparisons,
                    "improved_environments": improved_environments,
                    "regressed_environments": regressed_environments,
                    "neutral_environments": total_environments - improved_environments - regressed_environments,
                    "total_environments": total_environments,
                    "overall_verdict": overall_verdict,
                    "improvement_score": normalized_score,
                    "timestamp": datetime.now().isoformat()
                }
            
        # Lưu tất cả kết quả so sánh
        if target_version is None:
            # Lưu kết quả so sánh tất cả
            self.comparison_results = comparison_results
        else:
            # Cập nhật kết quả so sánh cho phiên bản cụ thể
            for target, result in comparison_results.items():
                self.comparison_results[target] = result
        
        return comparison_results
    
    def get_recommendation(
        self,
        target_version: Optional[str] = None,
        min_improvement_score: float = 0.3,
        require_statistical_significance: bool = False
    ) -> Dict[str, Any]:
        """
        Đưa ra khuyến nghị về việc nên triển khai phiên bản mô hình nào.
        
        Args:
            target_version: Phiên bản cụ thể để đưa ra khuyến nghị
            min_improvement_score: Điểm cải thiện tối thiểu để khuyến nghị triển khai
            require_statistical_significance: Yêu cầu kiểm định thống kê có ý nghĩa
            
        Returns:
            Dict chứa khuyến nghị
        """
        if not self.comparison_results:
            self.logger.warning("Chưa có kết quả so sánh nào để đưa ra khuyến nghị")
            return {
                "success": False,
                "message": "Chưa có kết quả so sánh nào để đưa ra khuyến nghị",
                "recommendation": "no_data"
            }
        
        if target_version is not None:
            # Chỉ đưa ra khuyến nghị cho phiên bản cụ thể
            if target_version not in self.comparison_results:
                self.logger.warning(f"Chưa có kết quả so sánh cho phiên bản {target_version}")
                return {
                    "success": False,
                    "message": f"Chưa có kết quả so sánh cho phiên bản {target_version}",
                    "recommendation": "no_data"
                }
            
            versions_to_check = [target_version]
        else:
            # Đưa ra khuyến nghị cho tất cả các phiên bản đã so sánh
            versions_to_check = list(self.comparison_results.keys())
        
        # Tìm phiên bản có điểm cải thiện cao nhất
        best_version = None
        best_score = -float('inf')
        
        for version in versions_to_check:
            comparison = self.comparison_results[version]
            score = comparison.get("improvement_score", 0)
            
            # Kiểm tra xem có kiểm định thống kê có ý nghĩa không
            if require_statistical_significance:
                has_significance = False
                
                for env_name, env_comp in comparison.get("environments", {}).items():
                    for metric, metric_comp in env_comp.get("metrics", {}).items():
                        if metric_comp.get("statistical_test", {}).get("significant", False):
                            has_significance = True
                            break
                    
                    if has_significance:
                        break
                
                # Bỏ qua nếu không có kiểm định thống kê có ý nghĩa
                if not has_significance:
                    continue
            
            # Cập nhật phiên bản tốt nhất
            if score > best_score:
                best_version = version
                best_score = score
        
        # Đưa ra khuyến nghị
        if best_version is None:
            return {
                "success": True,
                "recommendation": "keep_current",
                "message": "Không có phiên bản nào cải thiện đáng kể",
                "base_version": self.base_version
            }
        
        if best_score < min_improvement_score:
            return {
                "success": True,
                "recommendation": "keep_current",
                "message": f"Cải thiện không đủ đáng kể (điểm: {best_score:.2f} < {min_improvement_score})",
                "best_version": best_version,
                "best_score": best_score,
                "base_version": self.base_version
            }
        
        # Khuyến nghị triển khai phiên bản tốt nhất
        return {
            "success": True,
            "recommendation": "deploy_new_version",
            "message": f"Khuyến nghị triển khai phiên bản {best_version} (điểm cải thiện: {best_score:.2f})",
            "version_to_deploy": best_version,
            "improvement_score": best_score,
            "base_version": self.base_version,
            "comparison": self.comparison_results[best_version]
        }
    
    def generate_comparison_report(
        self,
        output_path: Optional[Union[str, Path]] = None,
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Tạo báo cáo so sánh chi tiết.
        
        Args:
            output_path: Đường dẫn để lưu báo cáo
            include_plots: Bao gồm biểu đồ hay không
            
        Returns:
            Dict chứa báo cáo
        """
        if not self.comparison_results:
            self.logger.warning("Chưa có kết quả so sánh nào để tạo báo cáo")
            return {
                "success": False,
                "message": "Chưa có kết quả so sánh nào để tạo báo cáo"
            }
        
        # Lấy khuyến nghị
        recommendation = self.get_recommendation()
        
        # Tạo báo cáo
        report = {
            "agent_type": self.agent_type,
            "base_version": self.base_version,
            "models_compared": list(self.models_to_compare.keys()),
            "environments_used": list(self.environments.keys()),
            "metrics_compared": self.metrics_to_compare,
            "recommendation": recommendation,
            "comparison_results": self.comparison_results,
            "report_time": datetime.now().isoformat(),
            "plots": []
        }
        
        # Tạo biểu đồ (nếu cần)
        if include_plots:
            plot_dir = None
            if output_path:
                plot_dir = Path(output_path).parent / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Tạo biểu đồ so sánh hiệu suất
                for metric in self.metrics_to_compare:
                    plot_path = None
                    if plot_dir:
                        plot_path = plot_dir / f"{self.agent_type}_{metric}_comparison.png"
                    
                    fig = self.plot_metric_comparison(
                        metric=metric,
                        show_plot=False,
                        save_path=plot_path
                    )
                    
                    if fig and plot_path:
                        report["plots"].append(str(plot_path))
                        plt.close(fig)
                
                # Tạo biểu đồ cải thiện tổng thể
                improvement_path = None
                if plot_dir:
                    improvement_path = plot_dir / f"{self.agent_type}_improvement_scores.png"
                
                fig = self.plot_improvement_scores(
                    show_plot=False,
                    save_path=improvement_path
                )
                
                if fig and improvement_path:
                    report["plots"].append(str(improvement_path))
                    plt.close(fig)
                
            except Exception as e:
                self.logger.warning(f"Lỗi khi tạo biểu đồ: {str(e)}")
        
        # Lưu báo cáo nếu cần
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=4)
            
            self.logger.info(f"Đã lưu báo cáo so sánh tại {output_path}")
        
        return report
    
    def plot_metric_comparison(
        self,
        metric: str,
        versions: Optional[List[str]] = None,
        environments: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> Optional[plt.Figure]:
        """
        Vẽ biểu đồ so sánh một số liệu cụ thể giữa các phiên bản.
        
        Args:
            metric: Số liệu cần so sánh
            versions: Danh sách phiên bản cần so sánh
            environments: Danh sách môi trường cần so sánh
            figsize: Kích thước biểu đồ
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ hay không
            
        Returns:
            Figure của matplotlib hoặc None
        """
        if not self.evaluation_results:
            self.logger.warning(f"Chưa có kết quả đánh giá nào để vẽ biểu đồ so sánh {metric}")
            return None
        
        # Thiết lập giá trị mặc định
        if versions is None:
            versions = list(self.models_to_compare.keys())
        
        if environments is None:
            environments = list(self.environments.keys())
        
        # Lọc các phiên bản và môi trường có dữ liệu
        valid_versions = []
        for version in versions:
            if version in self.evaluation_results:
                valid_versions.append(version)
        
        valid_environments = []
        for env_name in environments:
            # Kiểm tra xem có ít nhất một phiên bản có dữ liệu cho môi trường này không
            has_data = False
            for version in valid_versions:
                if env_name in self.evaluation_results[version]:
                    has_data = True
                    break
            
            if has_data:
                valid_environments.append(env_name)
        
        if not valid_versions or not valid_environments:
            self.logger.warning(f"Không có dữ liệu hợp lệ để vẽ biểu đồ so sánh {metric}")
            return None
        
        # Chuẩn bị dữ liệu
        values = []
        labels = []
        
        for version in valid_versions:
            version_values = []
            
            for env_name in valid_environments:
                if env_name in self.evaluation_results[version]:
                    eval_result = self.evaluation_results[version][env_name]
                    
                    if metric in eval_result["metrics"]:
                        version_values.append(eval_result["metrics"][metric])
                    else:
                        version_values.append(np.nan)
                else:
                    version_values.append(np.nan)
            
            values.append(version_values)
            labels.append(version)
        
        # Tạo biểu đồ
        fig, ax = plt.subplots(figsize=figsize)
        
        # Thiết lập vị trí các cột
        num_versions = len(valid_versions)
        num_environments = len(valid_environments)
        bar_width = 0.8 / num_versions
        
        # Tạo các thanh
        for i, (version_values, version) in enumerate(zip(values, labels)):
            x = np.arange(num_environments) + i * bar_width - (num_versions - 1) * bar_width / 2
            bars = ax.bar(x, version_values, bar_width, label=version)
            
            # Thêm nhãn giá trị
            for bar, value in zip(bars, version_values):
                if not np.isnan(value):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Tùy chỉnh biểu đồ
        ax.set_title(f"So sánh {metric.replace('_', ' ').title()} giữa các phiên bản")
        ax.set_xticks(np.arange(num_environments))
        ax.set_xticklabels(valid_environments)
        ax.legend()
        
        # Định dạng giá trị là phần trăm cho một số metrics
        if metric in [BacktestMetric.WIN_RATE.value, BacktestMetric.MAX_DRAWDOWN.value]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Thêm đường baseline
        if self.base_version in labels:
            base_index = labels.index(self.base_version)
            base_values = values[base_index]
            
            for i, value in enumerate(base_values):
                if not np.isnan(value):
                    ax.axhline(y=value, xmin=i/num_environments, xmax=(i+1)/num_environments,
                              color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Đã lưu biểu đồ tại {save_path}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_improvement_scores(
        self,
        versions: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> Optional[plt.Figure]:
        """
        Vẽ biểu đồ điểm cải thiện của các phiên bản.
        
        Args:
            versions: Danh sách phiên bản cần so sánh
            figsize: Kích thước biểu đồ
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ hay không
            
        Returns:
            Figure của matplotlib hoặc None
        """
        if not self.comparison_results:
            self.logger.warning("Chưa có kết quả so sánh nào để vẽ biểu đồ điểm cải thiện")
            return None
        
        # Thiết lập giá trị mặc định
        if versions is None:
            versions = list(self.comparison_results.keys())
        
        # Lọc các phiên bản có dữ liệu
        valid_versions = [v for v in versions if v in self.comparison_results]
        
        if not valid_versions:
            self.logger.warning("Không có dữ liệu hợp lệ để vẽ biểu đồ điểm cải thiện")
            return None
        
        # Chuẩn bị dữ liệu
        scores = [self.comparison_results[v].get("improvement_score", 0) for v in valid_versions]
        verdicts = [self.comparison_results[v].get("overall_verdict", "neutral") for v in valid_versions]
        
        # Ánh xạ verdict thành màu sắc
        colors = []
        for verdict in verdicts:
            if verdict == "improvement":
                colors.append('green')
            elif verdict == "regression":
                colors.append('red')
            else:
                colors.append('gray')
        
        # Tạo biểu đồ
        fig, ax = plt.subplots(figsize=figsize)
        
        # Vẽ các thanh
        bars = ax.bar(valid_versions, scores, color=colors)
        
        # Thêm nhãn giá trị
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height if height >= 0 else 0,
                   f'{score:.2f}', ha='center', va='bottom')
        
        # Thêm đường ngưỡng cải thiện
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Tùy chỉnh biểu đồ
        ax.set_title(f"Điểm cải thiện của các phiên bản so với {self.base_version}")
        ax.set_ylabel("Điểm cải thiện")
        ax.set_xlabel("Phiên bản")
        
        # Thêm chú thích
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Cải thiện'),
            Patch(facecolor='gray', label='Trung tính'),
            Patch(facecolor='red', label='Xuống cấp')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Đã lưu biểu đồ tại {save_path}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _calculate_max_drawdown(self, rewards: List[float]) -> float:
        """
        Tính toán drawdown tối đa từ danh sách phần thưởng.
        
        Args:
            rewards: Danh sách phần thưởng
            
        Returns:
            Giá trị drawdown tối đa
        """
        # Tích lũy phần thưởng
        cumulative = np.cumsum(rewards)
        
        # Tính toán drawdown
        max_dd = 0
        peak = cumulative[0]
        
        for value in cumulative:
            if value > peak:
                peak = value
            elif peak - value > max_dd:
                max_dd = peak - value
        
        # Chuẩn hóa drawdown
        max_dd_pct = max_dd / abs(peak) if peak != 0 else 0
        
        return max_dd_pct


def create_comparison_evaluator(
    agent_type: str,
    base_version: str,
    output_dir: Optional[Union[str, Path]] = None,
    metrics_to_compare: Optional[List[str]] = None,
    evaluation_episodes: int = 50,
    logger: Optional[logging.Logger] = None
) -> ComparisonEvaluator:
    """
    Hàm tiện ích để tạo ComparisonEvaluator.
    
    Args:
        agent_type: Loại agent đang đánh giá
        base_version: Phiên bản cơ sở để so sánh
        output_dir: Thư mục đầu ra cho kết quả đánh giá
        metrics_to_compare: Danh sách các số liệu cần so sánh
        evaluation_episodes: Số lượng episode để đánh giá mỗi mô hình
        logger: Logger tùy chỉnh
        
    Returns:
        ComparisonEvaluator đã được cấu hình
    """
    return ComparisonEvaluator(
        agent_type=agent_type,
        base_version=base_version,
        output_dir=output_dir,
        metrics_to_compare=metrics_to_compare,
        evaluation_episodes=evaluation_episodes,
        logger=logger
    )