"""
Module điều chỉnh siêu tham số.
File này định nghĩa các phương pháp tối ưu hóa siêu tham số cho việc huấn luyện
agent, bao gồm Grid Search, Random Search, và Bayesian Optimization.
"""

import os
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime
import concurrent.futures
from functools import partial
import matplotlib.pyplot as plt
import copy
import itertools

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger
from config.system_config import get_system_config, MODEL_DIR
from environments.base_environment import BaseEnvironment
from models.agents.base_agent import BaseAgent
from models.training_pipeline.trainer import Trainer

# Try to import optional dependencies
try:
    from skopt import gp_minimize, forest_minimize, dummy_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterTuner:
    """
    Lớp cung cấp các phương pháp tối ưu hóa siêu tham số cho việc huấn luyện agent.
    Hỗ trợ Grid Search, Random Search, Bayesian Optimization (với skopt), và Tree-structured Parzen Estimator (với optuna).
    """
    
    def __init__(
        self,
        agent_class: type,
        env_class: type,
        param_space: Dict[str, Any],
        metric: str = "eval_reward_mean",
        metric_direction: str = "maximize",
        n_trials: int = 20,
        random_seed: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        n_jobs: int = 1,
        logger: Optional[logging.Logger] = None,
        experiment_name: Optional[str] = None,
        **kwargs
    ):
        """
        Khởi tạo Hyperparameter Tuner.
        
        Args:
            agent_class: Lớp agent (không phải instance)
            env_class: Lớp môi trường (không phải instance)
            param_space: Không gian siêu tham số cần tối ưu
            metric: Tên của metric dùng để đánh giá (eval_reward_mean, eval_reward_std, etc.)
            metric_direction: Hướng tối ưu hóa (maximize hoặc minimize)
            n_trials: Số lần thử nghiệm
            random_seed: Seed cho bộ sinh số ngẫu nhiên
            output_dir: Thư mục đầu ra cho kết quả
            n_jobs: Số lượng công việc chạy song song (-1 để sử dụng tất cả CPU)
            logger: Logger tùy chỉnh
            experiment_name: Tên thí nghiệm
        """
        # Thiết lập logger
        self.logger = logger or get_logger("hyperparameter_tuner")
        
        # Thiết lập các thành phần chính
        self.agent_class = agent_class
        self.env_class = env_class
        self.param_space = param_space
        self.metric = metric
        self.metric_direction = metric_direction
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        
        # Thiết lập random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Thiết lập thư mục đầu ra
        if experiment_name is None:
            experiment_name = f"hparam_tuning_{agent_class.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        
        if output_dir is None:
            self.output_dir = MODEL_DIR / 'hyperparam_tuning' / self.experiment_name
        else:
            self.output_dir = Path(output_dir) / self.experiment_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Các tham số bổ sung
        self.kwargs = kwargs
        
        # Kết quả tuning
        self.results = []
        self.best_params = None
        self.best_score = float('-inf') if metric_direction == "maximize" else float('inf')
        self.tuning_complete = False
        
        # Validator function cho param space
        self._validate_param_space()
        
        self.logger.info(f"Đã khởi tạo HyperparameterTuner cho {agent_class.__name__} trên {env_class.__name__}")
        self.logger.info(f"Thí nghiệm: {self.experiment_name}, Metric: {metric} ({metric_direction})")
    
    def _validate_param_space(self) -> None:
        """
        Kiểm tra tính hợp lệ của không gian tham số.
        """
        # Kiểm tra từng tham số
        for param_name, param_config in self.param_space.items():
            # Tất cả các param_config phải định nghĩa 'type'
            if 'type' not in param_config:
                raise ValueError(f"Param '{param_name}' phải chứa trường 'type'")
            
            param_type = param_config['type']
            
            # Kiểm tra loại parameter
            if param_type == 'categorical':
                if 'values' not in param_config:
                    raise ValueError(f"Param loại 'categorical' '{param_name}' phải có trường 'values'")
            elif param_type in ['int', 'float']:
                if 'min' not in param_config or 'max' not in param_config:
                    raise ValueError(f"Param loại '{param_type}' '{param_name}' phải có trường 'min' và 'max'")
                
                if param_type == 'int' and ('step' in param_config and not isinstance(param_config['step'], int)):
                    raise ValueError(f"'step' cho param loại 'int' '{param_name}' phải là số nguyên")
            else:
                raise ValueError(f"Loại param không hợp lệ: {param_type}. Chỉ hỗ trợ: 'categorical', 'int', 'float'")
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Đánh giá bộ tham số bằng cách huấn luyện một agent.
        
        Args:
            params: Dict các tham số cần đánh giá
            
        Returns:
            Dict kết quả đánh giá
        """
        try:
            # Tạo môi trường
            env_kwargs = self.kwargs.get("env_kwargs", {})
            env = self.env_class(**env_kwargs)
            
            # Tạo agent với các siêu tham số cụ thể
            agent_params = self.kwargs.get("agent_kwargs", {}).copy()
            agent_params.update(params)
            
            # Lọc lại các tham số hợp lệ cho agent
            agent = self.agent_class(**agent_params)
            
            # Tạo trainer
            trainer_kwargs = self.kwargs.get("trainer_kwargs", {})
            trainer = Trainer(
                agent=agent,
                env=env,
                output_dir=self.output_dir / f"trial_{len(self.results)}",
                experiment_name=f"{self.experiment_name}_trial_{len(self.results)}",
                **trainer_kwargs
            )
            
            # Thực hiện huấn luyện
            n_episodes = trainer_kwargs.get("num_episodes", 200)
            trainer.train()
            
            # Đánh giá agent
            n_eval_episodes = trainer_kwargs.get("num_eval_episodes", 10)
            eval_rewards = trainer.evaluate(num_episodes=n_eval_episodes)
            
            # Tính toán các metrics
            eval_reward_mean = np.mean(eval_rewards)
            eval_reward_std = np.std(eval_rewards)
            eval_reward_min = np.min(eval_rewards)
            eval_reward_max = np.max(eval_rewards)
            
            # Đóng môi trường
            env.close()
            
            # Tạo kết quả
            result = {
                "params": params.copy(),
                "eval_reward_mean": float(eval_reward_mean),
                "eval_reward_std": float(eval_reward_std),
                "eval_reward_min": float(eval_reward_min),
                "eval_reward_max": float(eval_reward_max),
                "completed": True,
                "error": None,
                "trial_id": len(self.results)
            }
            
            # Cập nhật best score nếu cần
            score = result[self.metric]
            if self.metric_direction == "maximize" and score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
            elif self.metric_direction == "minimize" and score < self.best_score:
                self.best_score = score
                self.best_params = params.copy()
            
            self.logger.info(
                f"Trial {len(self.results)}: {self.metric}={score:.4f}, "
                f"Mean={eval_reward_mean:.2f}, Min={eval_reward_min:.2f}, Max={eval_reward_max:.2f}"
            )
            
            return result
            
        except Exception as e:
            # Xử lý lỗi
            self.logger.error(f"Lỗi khi đánh giá params: {str(e)}")
            
            # Tạo kết quả lỗi
            result = {
                "params": params.copy(),
                "eval_reward_mean": float('nan'),
                "eval_reward_std": float('nan'),
                "eval_reward_min": float('nan'),
                "eval_reward_max": float('nan'),
                "completed": False,
                "error": str(e),
                "trial_id": len(self.results)
            }
            
            return result
    
    def _sample_param_value(self, param_name: str, param_config: Dict[str, Any]) -> Any:
        """
        Lấy mẫu một giá trị tham số từ không gian tham số.
        
        Args:
            param_name: Tên tham số
            param_config: Cấu hình tham số
            
        Returns:
            Giá trị tham số
        """
        param_type = param_config['type']
        
        if param_type == 'categorical':
            values = param_config['values']
            return np.random.choice(values)
        
        elif param_type == 'int':
            min_val = param_config['min']
            max_val = param_config['max']
            step = param_config.get('step', 1)
            
            if 'log_scale' in param_config and param_config['log_scale']:
                # Lấy mẫu theo thang log
                log_min = np.log10(max(min_val, 1e-10))
                log_max = np.log10(max_val)
                log_val = np.random.uniform(log_min, log_max)
                val = int(10 ** log_val)
                # Làm tròn theo step
                val = int(np.round(val / step) * step)
                # Đảm bảo nằm trong khoảng
                val = max(min_val, min(max_val, val))
                return val
            else:
                # Lấy mẫu đều
                possible_values = list(range(min_val, max_val + 1, step))
                return np.random.choice(possible_values)
        
        elif param_type == 'float':
            min_val = param_config['min']
            max_val = param_config['max']
            
            if 'log_scale' in param_config and param_config['log_scale']:
                # Lấy mẫu theo thang log
                log_min = np.log10(max(min_val, 1e-10))
                log_max = np.log10(max_val)
                log_val = np.random.uniform(log_min, log_max)
                val = 10 ** log_val
                # Đảm bảo nằm trong khoảng
                val = max(min_val, min(max_val, val))
                return val
            else:
                # Lấy mẫu đều
                return np.random.uniform(min_val, max_val)
        
        else:
            raise ValueError(f"Loại param không hợp lệ: {param_type}")
    
    def _generate_param_grid(self) -> List[Dict[str, Any]]:
        """
        Tạo lưới tham số đầy đủ cho grid search.
        
        Returns:
            Danh sách các bộ tham số
        """
        param_lists = {}
        
        for param_name, param_config in self.param_space.items():
            param_type = param_config['type']
            
            if param_type == 'categorical':
                param_lists[param_name] = param_config['values']
            
            elif param_type == 'int':
                min_val = param_config['min']
                max_val = param_config['max']
                step = param_config.get('step', 1)
                
                if 'log_scale' in param_config and param_config['log_scale']:
                    # Tạo giá trị theo thang log
                    num_values = param_config.get('num_values', 5)
                    log_min = np.log10(max(min_val, 1e-10))
                    log_max = np.log10(max_val)
                    log_values = np.linspace(log_min, log_max, num_values)
                    values = [int(10 ** log_val) for log_val in log_values]
                    # Làm tròn theo step
                    values = [int(np.round(val / step) * step) for val in values]
                    # Loại bỏ các giá trị trùng lặp
                    values = sorted(list(set(values)))
                    # Đảm bảo nằm trong khoảng
                    values = [max(min_val, min(max_val, val)) for val in values]
                    param_lists[param_name] = values
                else:
                    param_lists[param_name] = list(range(min_val, max_val + 1, step))
            
            elif param_type == 'float':
                min_val = param_config['min']
                max_val = param_config['max']
                
                if 'log_scale' in param_config and param_config['log_scale']:
                    # Tạo giá trị theo thang log
                    num_values = param_config.get('num_values', 5)
                    log_min = np.log10(max(min_val, 1e-10))
                    log_max = np.log10(max_val)
                    log_values = np.linspace(log_min, log_max, num_values)
                    values = [10 ** log_val for log_val in log_values]
                    param_lists[param_name] = values
                else:
                    num_values = param_config.get('num_values', 5)
                    param_lists[param_name] = list(np.linspace(min_val, max_val, num_values))
        
        # Tạo tích Descartes của tất cả các tham số
        param_names = list(param_lists.keys())
        param_values = list(param_lists.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Chuyển đổi thành danh sách dict
        param_dicts = []
        for combination in param_combinations:
            param_dict = {}
            for i, param_name in enumerate(param_names):
                param_dict[param_name] = combination[i]
            param_dicts.append(param_dict)
        
        return param_dicts
    
    def grid_search(self) -> Dict[str, Any]:
        """
        Thực hiện grid search trên không gian tham số.
        
        Returns:
            Dict chứa kết quả tối ưu
        """
        self.logger.info("Bắt đầu Grid Search")
        
        # Tạo lưới tham số
        param_grid = self._generate_param_grid()
        
        self.logger.info(f"Tạo lưới với {len(param_grid)} bộ tham số")
        
        # Hạn chế số lượng trials nếu cần
        if self.n_trials < len(param_grid):
            self.logger.warning(
                f"Số lượng grid points ({len(param_grid)}) vượt quá n_trials ({self.n_trials}). "
                f"Chỉ thực hiện {self.n_trials} trials đầu tiên."
            )
            param_grid = param_grid[:self.n_trials]
        
        # Đánh giá song song nếu có thể
        if self.n_jobs > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for params in param_grid:
                    future = executor.submit(self._evaluate_params, params)
                    futures.append(future)
                
                # Thu thập kết quả
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    self.results.append(result)
        else:
            # Đánh giá tuần tự
            for params in param_grid:
                result = self._evaluate_params(params)
                self.results.append(result)
        
        # Lưu kết quả
        self._save_results()
        
        # Đánh dấu hoàn thành
        self.tuning_complete = True
        
        # Vẽ biểu đồ kết quả
        self._plot_results()
        
        return self._get_best_result()
    
    def random_search(self) -> Dict[str, Any]:
        """
        Thực hiện random search trên không gian tham số.
        
        Returns:
            Dict chứa kết quả tối ưu
        """
        self.logger.info(f"Bắt đầu Random Search với {self.n_trials} trials")
        
        for trial in range(self.n_trials):
            # Lấy mẫu ngẫu nhiên các tham số
            params = {}
            for param_name, param_config in self.param_space.items():
                params[param_name] = self._sample_param_value(param_name, param_config)
            
            # Đánh giá bộ tham số
            result = self._evaluate_params(params)
            self.results.append(result)
            
            # Lưu kết quả theo tiến trình
            if (trial + 1) % max(1, self.n_trials // 10) == 0:
                self._save_results()
        
        # Lưu kết quả cuối cùng
        self._save_results()
        
        # Đánh dấu hoàn thành
        self.tuning_complete = True
        
        # Vẽ biểu đồ kết quả
        self._plot_results()
        
        return self._get_best_result()
    
    def bayesian_optimization(self) -> Dict[str, Any]:
        """
        Thực hiện Bayesian Optimization trên không gian tham số.
        
        Returns:
            Dict chứa kết quả tối ưu
        """
        if not SKOPT_AVAILABLE:
            self.logger.error("Không thể sử dụng Bayesian Optimization: thư viện 'scikit-optimize' không khả dụng")
            self.logger.info("Chuyển sang sử dụng Random Search")
            return self.random_search()
        
        self.logger.info(f"Bắt đầu Bayesian Optimization với {self.n_trials} trials")
        
        # Chuyển đổi param_space sang định dạng skopt
        dimensions = []
        dimension_names = []
        
        for param_name, param_config in self.param_space.items():
            param_type = param_config['type']
            
            if param_type == 'categorical':
                dimension = Categorical(param_config['values'], name=param_name)
            
            elif param_type == 'int':
                min_val = param_config['min']
                max_val = param_config['max']
                
                if 'log_scale' in param_config and param_config['log_scale']:
                    dimension = Integer(min_val, max_val, name=param_name, prior='log-uniform')
                else:
                    dimension = Integer(min_val, max_val, name=param_name)
            
            elif param_type == 'float':
                min_val = param_config['min']
                max_val = param_config['max']
                
                if 'log_scale' in param_config and param_config['log_scale']:
                    dimension = Real(min_val, max_val, name=param_name, prior='log-uniform')
                else:
                    dimension = Real(min_val, max_val, name=param_name)
            
            else:
                raise ValueError(f"Loại param không hợp lệ: {param_type}")
            
            dimensions.append(dimension)
            dimension_names.append(param_name)
        
        # Định nghĩa hàm mục tiêu
        @use_named_args(dimensions)
        def objective(**params):
            result = self._evaluate_params(params)
            self.results.append(result)
            
            # Lấy giá trị cần tối ưu
            value = result[self.metric]
            
            # Chuyển đổi nếu là maximize
            if self.metric_direction == "maximize":
                return -value
            else:
                return value
        
        # Thực hiện Bayesian Optimization
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=self.n_trials,
            random_state=self.random_seed,
            n_jobs=self.n_jobs if self.n_jobs > 0 else None,
            verbose=True
        )
        
        # Lưu kết quả
        self._save_results()
        
        # Đánh dấu hoàn thành
        self.tuning_complete = True
        
        # Vẽ biểu đồ kết quả
        self._plot_results()
        
        return self._get_best_result()
    
    def optuna_optimization(self) -> Dict[str, Any]:
        """
        Thực hiện tối ưu hóa sử dụng Optuna (TPE - Tree-structured Parzen Estimator).
        
        Returns:
            Dict chứa kết quả tối ưu
        """
        if not OPTUNA_AVAILABLE:
            self.logger.error("Không thể sử dụng Optuna: thư viện 'optuna' không khả dụng")
            self.logger.info("Chuyển sang sử dụng Random Search")
            return self.random_search()
        
        self.logger.info(f"Bắt đầu Optuna Optimization với {self.n_trials} trials")
        
        # Tạo study
        study_name = f"study_{self.experiment_name}"
        
        if self.metric_direction == "maximize":
            direction = "maximize"
        else:
            direction = "minimize"
        
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_seed)
        )
        
        # Định nghĩa hàm mục tiêu
        def objective(trial):
            # Lấy tham số từ không gian tham số
            params = {}
            
            for param_name, param_config in self.param_space.items():
                param_type = param_config['type']
                
                if param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
                
                elif param_type == 'int':
                    min_val = param_config['min']
                    max_val = param_config['max']
                    step = param_config.get('step', 1)
                    
                    if 'log_scale' in param_config and param_config['log_scale']:
                        params[param_name] = trial.suggest_int(param_name, min_val, max_val, log=True)
                    else:
                        params[param_name] = trial.suggest_int(param_name, min_val, max_val, step=step)
                
                elif param_type == 'float':
                    min_val = param_config['min']
                    max_val = param_config['max']
                    
                    if 'log_scale' in param_config and param_config['log_scale']:
                        params[param_name] = trial.suggest_float(param_name, min_val, max_val, log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            
            # Đánh giá bộ tham số
            result = self._evaluate_params(params)
            self.results.append(result)
            
            # Trả về giá trị mục tiêu
            return result[self.metric]
        
        # Thực hiện tối ưu hóa
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs if self.n_jobs > 0 else 1)
        
        # Lưu kết quả
        self._save_results()
        
        # Đánh dấu hoàn thành
        self.tuning_complete = True
        
        # Vẽ biểu đồ kết quả
        self._plot_results()
        
        # Lưu biểu đồ Optuna
        try:
            optuna_plots_dir = self.output_dir / "optuna_plots"
            optuna_plots_dir.mkdir(exist_ok=True)
            
            # Lưu biểu đồ tham số quan trọng
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(str(optuna_plots_dir / "param_importances.png"))
            
            # Lưu biểu đồ Parallel Coordinate
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(str(optuna_plots_dir / "parallel_coordinate.png"))
            
            # Lưu biểu đồ Slice
            fig = optuna.visualization.plot_slice(study)
            fig.write_image(str(optuna_plots_dir / "slice.png"))
            
            # Lưu biểu đồ Contour
            fig = optuna.visualization.plot_contour(study)
            fig.write_image(str(optuna_plots_dir / "contour.png"))
            
        except Exception as e:
            self.logger.warning(f"Không thể lưu biểu đồ Optuna: {str(e)}")
        
        return self._get_best_result()
    
    def _get_best_result(self) -> Dict[str, Any]:
        """
        Lấy kết quả tốt nhất từ các thử nghiệm.
        
        Returns:
            Dict chứa thông tin về kết quả tốt nhất
        """
        # Lọc các kết quả đã hoàn thành
        completed_results = [r for r in self.results if r["completed"]]
        
        if not completed_results:
            self.logger.warning("Không có kết quả nào hoàn thành")
            return {
                "best_params": None,
                "best_score": None,
                "all_results": self.results
            }
        
        # Sắp xếp kết quả theo metric
        if self.metric_direction == "maximize":
            sorted_results = sorted(completed_results, key=lambda r: r[self.metric], reverse=True)
        else:
            sorted_results = sorted(completed_results, key=lambda r: r[self.metric])
        
        # Lấy kết quả tốt nhất
        best_result = sorted_results[0]
        
        return {
            "best_params": best_result["params"],
            "best_score": best_result[self.metric],
            "best_result": best_result,
            "all_results": self.results
        }
    
    def _save_results(self) -> None:
        """
        Lưu kết quả vào file.
        """
        # Tạo DataFrame từ kết quả
        results_data = []
        for result in self.results:
            # Tạo một bản sao của params và thêm các metrics
            result_item = result["params"].copy()
            result_item.update({
                "eval_reward_mean": result["eval_reward_mean"],
                "eval_reward_std": result["eval_reward_std"],
                "eval_reward_min": result["eval_reward_min"],
                "eval_reward_max": result["eval_reward_max"],
                "completed": result["completed"],
                "error": result["error"],
                "trial_id": result["trial_id"]
            })
            results_data.append(result_item)
        
        # Chuyển thành DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Lưu vào CSV
        csv_path = self.output_dir / "tuning_results.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Lưu kết quả dạng JSON (đầy đủ hơn)
        json_path = self.output_dir / "tuning_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "experiment_name": self.experiment_name,
                "agent_class": self.agent_class.__name__,
                "env_class": self.env_class.__name__,
                "param_space": self.param_space,
                "metric": self.metric,
                "metric_direction": self.metric_direction,
                "n_trials": self.n_trials,
                "best_params": self.best_params,
                "best_score": float(self.best_score) if isinstance(self.best_score, np.number) else self.best_score,
                "results": self.results,
                "completed": self.tuning_complete,
                "timestamp": datetime.now().isoformat()
            }, f, indent=4, default=lambda o: str(o))
        
        self.logger.info(f"Đã lưu kết quả vào {csv_path} và {json_path}")
    
    def _plot_results(self) -> None:
        """
        Vẽ biểu đồ kết quả.
        """
        # Tạo thư mục plots
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Lọc các kết quả đã hoàn thành
        completed_results = [r for r in self.results if r["completed"]]
        
        if not completed_results:
            self.logger.warning("Không có kết quả nào để vẽ biểu đồ")
            return
        
        # 1. Biểu đồ tiến trình
        plt.figure(figsize=(10, 6))
        
        trial_ids = [r["trial_id"] for r in completed_results]
        scores = [r[self.metric] for r in completed_results]
        
        plt.plot(trial_ids, scores, 'o-', color='blue', alpha=0.7)
        
        # Highlight điểm tốt nhất
        if self.metric_direction == "maximize":
            best_idx = np.argmax(scores)
        else:
            best_idx = np.argmin(scores)
        
        best_trial_id = completed_results[best_idx]["trial_id"]
        best_score = completed_results[best_idx][self.metric]
        
        plt.scatter([best_trial_id], [best_score], color='red', s=100, zorder=5)
        plt.annotate(f"Best: {best_score:.4f}", 
                    (best_trial_id, best_score),
                    xytext=(10, 10),
                    textcoords='offset points',
                    color='red')
        
        # Thêm đường cho giá trị best hiện tại
        if self.metric_direction == "maximize":
            current_best = np.maximum.accumulate(scores)
        else:
            current_best = np.minimum.accumulate(scores)
        
        plt.plot(trial_ids, current_best, '--', color='green', alpha=0.7, label="Current Best")
        
        plt.xlabel("Trial ID")
        plt.ylabel(self.metric)
        plt.title(f"Optimization Progress: {self.experiment_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Lưu biểu đồ
        progress_plot_path = plots_dir / "optimization_progress.png"
        plt.savefig(progress_plot_path)
        plt.close()
        
        # 2. Biểu đồ phân phối điểm số
        plt.figure(figsize=(10, 6))
        
        plt.hist(scores, bins=min(len(scores) // 2 + 1, 20), alpha=0.7, color='blue')
        plt.axvline(best_score, color='red', linestyle='--', linewidth=2)
        plt.annotate(f"Best: {best_score:.4f}", 
                    (best_score, 0),
                    xytext=(5, 10),
                    textcoords='offset points',
                    color='red')
        
        plt.xlabel(self.metric)
        plt.ylabel("Count")
        plt.title(f"Score Distribution: {self.experiment_name}")
        plt.grid(True, alpha=0.3)
        
        # Lưu biểu đồ
        dist_plot_path = plots_dir / "score_distribution.png"
        plt.savefig(dist_plot_path)
        plt.close()
        
        # 3. Biểu đồ tham số quan trọng (nếu có nhiều hơn 5 trials)
        if len(completed_results) >= 5:
            # Tạo DataFrame cho phân tích
            param_names = list(self.param_space.keys())
            
            for param_name in param_names:
                # Bỏ qua tham số categorical
                param_config = self.param_space[param_name]
                if param_config['type'] == 'categorical':
                    continue
                
                plt.figure(figsize=(10, 6))
                
                param_values = [r["params"][param_name] for r in completed_results]
                
                # Vẽ scatter plot
                plt.scatter(param_values, scores, alpha=0.7, color='blue')
                
                # Thêm đường hồi quy để thấy xu hướng
                try:
                    z = np.polyfit(param_values, scores, 1)
                    p = np.poly1d(z)
                    plt.plot(sorted(param_values), p(sorted(param_values)), "r--", alpha=0.5)
                except:
                    pass
                
                # Highlight điểm tốt nhất
                plt.scatter([param_values[best_idx]], [scores[best_idx]], color='red', s=100, zorder=5)
                
                plt.xlabel(param_name)
                plt.ylabel(self.metric)
                plt.title(f"Impact of {param_name} on {self.metric}")
                plt.grid(True, alpha=0.3)
                
                # Lưu biểu đồ
                param_plot_path = plots_dir / f"param_impact_{param_name}.png"
                plt.savefig(param_plot_path)
                plt.close()
        
        self.logger.info(f"Đã tạo các biểu đồ tại {plots_dir}")
    
    def export_report(self) -> str:
        """
        Xuất báo cáo tối ưu hóa dạng Markdown.
        
        Returns:
            Đường dẫn file báo cáo
        """
        report_path = self.output_dir / "optimization_report.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            # Tiêu đề
            f.write(f"# Báo cáo Tối ưu hóa Siêu tham số: {self.experiment_name}\n\n")
            
            # Thông tin chung
            f.write("## Thông tin chung\n\n")
            f.write(f"- **Thời gian:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Agent:** {self.agent_class.__name__}\n")
            f.write(f"- **Môi trường:** {self.env_class.__name__}\n")
            f.write(f"- **Số trials đã thực hiện:** {len(self.results)}\n")
            f.write(f"- **Số trials hoàn thành:** {len([r for r in self.results if r['completed']])}\n")
            f.write(f"- **Metric:** {self.metric} ({self.metric_direction})\n")
            f.write(f"- **Trạng thái:** {'Hoàn thành' if self.tuning_complete else 'Đang thực hiện'}\n\n")
            
            # Kết quả tốt nhất
            best_result = self._get_best_result()
            
            f.write("## Kết quả tốt nhất\n\n")
            
            if best_result["best_params"] is not None:
                f.write(f"- **{self.metric}:** {best_result['best_score']:.4f}\n")
                f.write("- **Tham số tốt nhất:**\n\n")
                f.write("```json\n")
                json.dump(best_result["best_params"], f, indent=2, ensure_ascii=False)
                f.write("\n```\n\n")
                
                # Thông tin chi tiết
                best_detail = best_result["best_result"]
                f.write("### Chi tiết kết quả tốt nhất\n\n")
                f.write(f"- **Trial ID:** {best_detail['trial_id']}\n")
                f.write(f"- **Mean Reward:** {best_detail['eval_reward_mean']:.4f}\n")
                f.write(f"- **Std Reward:** {best_detail['eval_reward_std']:.4f}\n")
                f.write(f"- **Min Reward:** {best_detail['eval_reward_min']:.4f}\n")
                f.write(f"- **Max Reward:** {best_detail['eval_reward_max']:.4f}\n\n")
            else:
                f.write("*Chưa có kết quả*\n\n")
            
            # Không gian tham số
            f.write("## Không gian tham số\n\n")
            f.write("```json\n")
            json.dump(self.param_space, f, indent=2, ensure_ascii=False)
            f.write("\n```\n\n")
            
            # Biểu đồ
            plots_dir = self.output_dir / "plots"
            if plots_dir.exists():
                f.write("## Biểu đồ\n\n")
                
                progress_plot = plots_dir / "optimization_progress.png"
                if progress_plot.exists():
                    f.write("### Tiến trình tối ưu hóa\n\n")
                    f.write(f"![Optimization Progress](plots/optimization_progress.png)\n\n")
                
                dist_plot = plots_dir / "score_distribution.png"
                if dist_plot.exists():
                    f.write("### Phân phối điểm số\n\n")
                    f.write(f"![Score Distribution](plots/score_distribution.png)\n\n")
                
                # Biểu đồ tham số
                f.write("### Ảnh hưởng của tham số\n\n")
                
                param_plots = list(plots_dir.glob("param_impact_*.png"))
                if param_plots:
                    for plot in param_plots:
                        param_name = plot.stem.replace("param_impact_", "")
                        f.write(f"#### {param_name}\n\n")
                        f.write(f"![Impact of {param_name}](plots/{plot.name})\n\n")
                else:
                    f.write("*Không có đủ dữ liệu để phân tích ảnh hưởng của tham số*\n\n")
            
            # Optuna plots
            optuna_plots_dir = self.output_dir / "optuna_plots"
            if optuna_plots_dir.exists() and list(optuna_plots_dir.glob("*.png")):
                f.write("## Biểu đồ Optuna\n\n")
                
                for plot in optuna_plots_dir.glob("*.png"):
                    plot_name = plot.stem.replace("_", " ").title()
                    f.write(f"### {plot_name}\n\n")
                    f.write(f"![{plot_name}](optuna_plots/{plot.name})\n\n")
            
            # Kết luận
            f.write("## Kết luận\n\n")
            
            if self.tuning_complete and best_result["best_params"] is not None:
                f.write("Quá trình tối ưu hóa siêu tham số đã hoàn thành thành công. ")
                f.write(f"Bộ tham số tốt nhất đạt được {self.metric} là {best_result['best_score']:.4f}.\n\n")
                
                f.write("### Khuyến nghị\n\n")
                f.write("- Sử dụng bộ tham số tốt nhất cho các lần huấn luyện tiếp theo.\n")
                f.write("- Có thể tinh chỉnh thêm xung quanh các giá trị tham số tốt nhất.\n")
                f.write("- Thử nghiệm với môi trường khác để đánh giá tính tổng quát.\n\n")
            else:
                f.write("Quá trình tối ưu hóa siêu tham số vẫn đang tiếp tục. Báo cáo này được tạo làm mốc trung gian.\n\n")
        
        self.logger.info(f"Đã xuất báo cáo tối ưu hóa tại {report_path}")
        
        return str(report_path)
    
    def get_best_agent(self) -> BaseAgent:
        """
        Tạo agent mới với tham số tốt nhất.
        
        Returns:
            Instance của agent với tham số tốt nhất
        """
        if self.best_params is None:
            self.logger.warning("Chưa có tham số tốt nhất. Hãy chạy tối ưu hóa trước.")
            return None
        
        # Tạo agent với các siêu tham số tốt nhất
        agent_params = self.kwargs.get("agent_kwargs", {}).copy()
        agent_params.update(self.best_params)
        
        # Lọc lại các tham số hợp lệ cho agent
        agent = self.agent_class(**agent_params)
        
        return agent
    
    def train_best_agent(self, num_episodes: Optional[int] = None) -> Trainer:
        """
        Huấn luyện agent với tham số tốt nhất.
        
        Args:
            num_episodes: Số episodes huấn luyện (None để sử dụng giá trị mặc định)
            
        Returns:
            Trainer instance đã hoàn thành huấn luyện
        """
        if self.best_params is None:
            self.logger.warning("Chưa có tham số tốt nhất. Hãy chạy tối ưu hóa trước.")
            return None
        
        # Tạo môi trường
        env_kwargs = self.kwargs.get("env_kwargs", {})
        env = self.env_class(**env_kwargs)
        
        # Tạo agent với các siêu tham số tốt nhất
        agent = self.get_best_agent()
        
        # Tạo trainer
        trainer_kwargs = self.kwargs.get("trainer_kwargs", {}).copy()
        
        if num_episodes is not None:
            trainer_kwargs["num_episodes"] = num_episodes
        
        trainer = Trainer(
            agent=agent,
            env=env,
            output_dir=self.output_dir / "best_agent",
            experiment_name=f"{self.experiment_name}_best_agent",
            **trainer_kwargs
        )
        
        # Thực hiện huấn luyện
        trainer.train()
        
        # Xuất báo cáo
        trainer.export_report()
        
        return trainer