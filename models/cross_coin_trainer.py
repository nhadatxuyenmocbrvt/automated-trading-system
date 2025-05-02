"""
Module huấn luyện đa cặp tiền.
File này định nghĩa lớp CrossCoinTrainer để huấn luyện agent trên nhiều cặp tiền tệ,
cho phép tận dụng dữ liệu từ nhiều thị trường để tạo ra một mô hình tổng quát hơn.
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

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import get_logger
from config.system_config import get_system_config, MODEL_DIR
from environments.base_environment import BaseEnvironment
from environments.trading_gym.trading_env import TradingEnv
from models.agents.base_agent import BaseAgent
from models.training_pipeline.trainer import Trainer


class CrossCoinTrainer:
    """
    Lớp quản lý việc huấn luyện agent trên nhiều cặp tiền tệ.
    """
    
    def __init__(
        self,
        agent_class: type,
        env_class: type,
        symbols: List[str],
        data_sources: Dict[str, Union[pd.DataFrame, str, Path]],
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        use_common_features: bool = True,
        training_mode: str = "sequential",
        transfer_learning: bool = True,
        evaluation_interval: int = 10,
        n_eval_episodes: int = 5,
        market_specific_params: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Khởi tạo CrossCoinTrainer.
        
        Args:
            agent_class: Lớp agent (không phải instance)
            env_class: Lớp môi trường (không phải instance)
            symbols: Danh sách các cặp tiền tệ
            data_sources: Dict với key là symbol và value là DataFrame hoặc đường dẫn đến dữ liệu
            config: Cấu hình huấn luyện chung
            output_dir: Thư mục đầu ra cho kết quả huấn luyện
            logger: Logger tùy chỉnh
            experiment_name: Tên thí nghiệm
            use_tensorboard: Sử dụng TensorBoard để theo dõi
            use_common_features: Sử dụng các đặc trưng chung giữa các cặp tiền
            training_mode: Chế độ huấn luyện ("sequential", "parallel", "mixed")
            transfer_learning: Sử dụng transfer learning giữa các cặp tiền
            evaluation_interval: Số episodes giữa mỗi lần đánh giá
            n_eval_episodes: Số episodes đánh giá
            market_specific_params: Tham số riêng cho từng thị trường
        """
        # Thiết lập logger
        self.logger = logger or get_logger("cross_coin_trainer")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập các thành phần chính
        self.agent_class = agent_class
        self.env_class = env_class
        self.symbols = symbols
        
        # Thiết lập các tham số
        if experiment_name is None:
            experiment_name = f"cross_coin_{agent_class.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        
        self.use_tensorboard = use_tensorboard
        self.use_common_features = use_common_features
        self.training_mode = training_mode
        self.transfer_learning = transfer_learning
        self.evaluation_interval = evaluation_interval
        self.n_eval_episodes = n_eval_episodes
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            self.output_dir = MODEL_DIR / 'cross_coin_training' / self.experiment_name
        else:
            self.output_dir = Path(output_dir) / self.experiment_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập cấu hình mặc định nếu không được cung cấp
        if config is None:
            config = self._get_default_config()
        self.config = config
        
        # Tham số riêng cho từng thị trường
        self.market_specific_params = market_specific_params or {}
        
        # Các tham số bổ sung
        self.kwargs = kwargs
        
        # Tải dữ liệu
        self.data_sources = {}
        self._load_data_sources(data_sources)
        
        # Lịch sử huấn luyện
        self.history = {
            "episode_rewards": {},
            "episode_lengths": {},
            "losses": {},
            "eval_rewards": {},
            "timestamps": {}
        }
        
        # Khởi tạo các môi trường và agents
        self.environments = {}
        self.agents = {}
        self.trainers = {}
        
        # Trạng thái huấn luyện
        self.current_symbol = None
        self.training_complete = False
        
        # Khởi tạo môi trường và agent
        self._init_environments_and_agents()
        
        self.logger.info(f"Đã khởi tạo CrossCoinTrainer cho {len(symbols)} cặp tiền")
        self.logger.info(f"Kết quả huấn luyện sẽ được lưu tại {self.output_dir}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Lấy cấu hình mặc định cho huấn luyện.
        
        Returns:
            Dict cấu hình mặc định
        """
        return {
            "num_episodes_per_symbol": 200,
            "max_steps_per_episode": 1000,
            "eval_frequency": 10,
            "num_eval_episodes": 5,
            "save_frequency": 50,
            "learning_starts": 1000,
            "target_update_frequency": 1000,
            "update_frequency": 4,
            "render_eval": False,
            "render_frequency": 0,  # 0: không render, n: render mỗi n episodes
            "gamma": 0.99,
            "learning_rate": 0.001,
            "epsilon_schedule": {
                "start": 1.0,
                "end": 0.01,
                "decay_steps": 10000
            },
            "batch_size": 64
        }
    
    def _load_data_sources(self, data_sources: Dict[str, Union[pd.DataFrame, str, Path]]) -> None:
        """
        Tải dữ liệu từ các nguồn.
        
        Args:
            data_sources: Dict với key là symbol và value là DataFrame hoặc đường dẫn đến dữ liệu
        """
        for symbol, source in data_sources.items():
            if symbol not in self.symbols:
                self.logger.warning(f"Symbol {symbol} không nằm trong danh sách symbols, bỏ qua")
                continue
            
            try:
                if isinstance(source, pd.DataFrame):
                    # Đã là DataFrame
                    self.data_sources[symbol] = source
                    self.logger.info(f"Đã tải dữ liệu cho {symbol} từ DataFrame ({len(source)} dòng)")
                else:
                    # Là đường dẫn
                    path = Path(source)
                    if not path.exists():
                        self.logger.error(f"Không tìm thấy file dữ liệu cho {symbol}: {path}")
                        continue
                    
                    # Tải dữ liệu
                    if path.suffix == '.csv':
                        df = pd.read_csv(path)
                    elif path.suffix == '.parquet':
                        df = pd.read_parquet(path)
                    elif path.suffix == '.json':
                        df = pd.read_json(path)
                    else:
                        self.logger.error(f"Định dạng file không được hỗ trợ: {path.suffix}")
                        continue
                    
                    self.data_sources[symbol] = df
                    self.logger.info(f"Đã tải dữ liệu cho {symbol} từ {path} ({len(df)} dòng)")
            
            except Exception as e:
                self.logger.error(f"Lỗi khi tải dữ liệu cho {symbol}: {str(e)}")
    
    def _init_environments_and_agents(self) -> None:
        """
        Khởi tạo các môi trường và agent cho từng cặp tiền.
        """
        for symbol in self.symbols:
            if symbol not in self.data_sources:
                self.logger.warning(f"Không có dữ liệu cho {symbol}, bỏ qua")
                continue
            
            try:
                # Tạo thư mục cho symbol
                symbol_dir = self.output_dir / symbol.replace('/', '_')
                symbol_dir.mkdir(exist_ok=True)
                
                # Lấy tham số riêng cho thị trường này nếu có
                market_params = self.market_specific_params.get(symbol, {})
                
                # Kết hợp với tham số mặc định
                env_kwargs = self.kwargs.get("env_kwargs", {}).copy()
                env_kwargs.update(market_params.get("env_kwargs", {}))
                
                # Khởi tạo môi trường
                env = self.env_class(
                    data=self.data_sources[symbol],
                    symbol=symbol,
                    logger=self.logger,
                    **env_kwargs
                )
                
                # Tạo agent params
                agent_kwargs = self.kwargs.get("agent_kwargs", {}).copy()
                agent_kwargs.update(market_params.get("agent_kwargs", {}))
                
                # Kiểm tra và cập nhật giá trị state_dim và action_dim
                if hasattr(env, 'observation_space') and 'state_dim' not in agent_kwargs:
                    if hasattr(env.observation_space, 'shape'):
                        agent_kwargs['state_dim'] = env.observation_space.shape
                    
                if hasattr(env, 'action_space') and 'action_dim' not in agent_kwargs:
                    if hasattr(env.action_space, 'n'):
                        agent_kwargs['action_dim'] = env.action_space.n
                    elif hasattr(env.action_space, 'shape'):
                        agent_kwargs['action_dim'] = env.action_space.shape[0]
                
                # Khởi tạo agent
                agent = self.agent_class(**agent_kwargs)
                
                # Khởi tạo trainer
                trainer_kwargs = self.kwargs.get("trainer_kwargs", {}).copy()
                trainer_kwargs.update(market_params.get("trainer_kwargs", {}))
                
                trainer = Trainer(
                    agent=agent,
                    env=env,
                    output_dir=symbol_dir,
                    experiment_name=f"{self.experiment_name}_{symbol.replace('/', '_')}",
                    **trainer_kwargs
                )
                
                # Lưu vào danh sách
                self.environments[symbol] = env
                self.agents[symbol] = agent
                self.trainers[symbol] = trainer
                
                self.logger.info(f"Đã khởi tạo môi trường và agent cho {symbol}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi khởi tạo môi trường và agent cho {symbol}: {str(e)}")
    
    def train_sequential(self) -> Dict[str, Any]:
        """
        Huấn luyện tuần tự trên từng cặp tiền.
        
        Returns:
            Dict chứa lịch sử huấn luyện
        """
        self.logger.info(f"Bắt đầu huấn luyện tuần tự trên {len(self.trainers)} cặp tiền")
        
        # Lấy các tham số từ cấu hình
        num_episodes = self.config["num_episodes_per_symbol"]
        
        # Thời gian bắt đầu
        start_time = time.time()
        
        # Huấn luyện lần lượt trên từng cặp tiền
        for i, (symbol, trainer) in enumerate(self.trainers.items()):
            self.logger.info(f"[{i+1}/{len(self.trainers)}] Bắt đầu huấn luyện cho {symbol}")
            
            # Cập nhật symbol hiện tại
            self.current_symbol = symbol
            
            # Nếu là transfer learning và không phải cặp tiền đầu tiên
            if self.transfer_learning and i > 0:
                # Lấy symbol trước đó
                prev_symbol = list(self.trainers.keys())[i-1]
                prev_agent = self.agents[prev_symbol]
                
                # Sao chép trọng số từ agent trước sang agent hiện tại
                self._transfer_weights(prev_agent, self.agents[symbol])
                self.logger.info(f"Đã áp dụng transfer learning từ {prev_symbol} sang {symbol}")
            
            # Huấn luyện
            trainer.train()
            
            # Cập nhật lịch sử
            self.history["episode_rewards"][symbol] = trainer.history["episode_rewards"]
            self.history["episode_lengths"][symbol] = trainer.history["episode_lengths"]
            self.history["losses"][symbol] = trainer.history["losses"]
            self.history["eval_rewards"][symbol] = trainer.history["eval_rewards"]
            self.history["timestamps"][symbol] = trainer.history["timestamps"]
            
            # Lưu lịch sử huấn luyện tổng hợp
            self._save_combined_history()
            
            # Đánh giá cross-validation (kiểm tra chéo)
            if i > 0:
                self._cross_validate(symbol)
        
        # Đánh giá tổng hợp
        self._evaluate_all()
        
        # Đánh dấu hoàn thành
        self.training_complete = True
        
        # Lưu lịch sử huấn luyện cuối cùng
        self._save_combined_history()
        
        # Vẽ biểu đồ kết quả
        self._plot_combined_results()
        
        # Xuất báo cáo
        self.export_report()
        
        return self.history
    
    def train_parallel(self) -> Dict[str, Any]:
        """
        Huấn luyện song song trên tất cả các cặp tiền.
        
        Returns:
            Dict chứa lịch sử huấn luyện
        """
        self.logger.info(f"Bắt đầu huấn luyện song song trên {len(self.trainers)} cặp tiền")
        
        # Lấy các tham số từ cấu hình
        num_episodes = self.config["num_episodes_per_symbol"]
        
        # Thời gian bắt đầu
        start_time = time.time()
        
        # Xác định số lượng công việc song song
        n_jobs = min(len(self.trainers), os.cpu_count() or 4)
        
        # Huấn luyện song song
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Chuẩn bị các tham số cho hàm huấn luyện
            params = []
            for symbol, trainer in self.trainers.items():
                # Lưu trainer và môi trường vào disk để tránh lỗi serialization
                symbol_dir = self.output_dir / symbol.replace('/', '_')
                trainer_path = symbol_dir / "trainer.pkl"
                
                # Đảm bảo thư mục tồn tại
                symbol_dir.mkdir(exist_ok=True)
                
                # Lưu trainer
                trainer.save_config(str(trainer_path))
                
                # Thêm vào danh sách params
                params.append((symbol, str(trainer_path), num_episodes))
            
            # Chạy các công việc song song
            futures = [executor.submit(self._train_symbol_wrapper, param) for param in params]
            
            # Thu thập kết quả
            for future in concurrent.futures.as_completed(futures):
                try:
                    symbol, history = future.result()
                    
                    # Cập nhật lịch sử
                    self.history["episode_rewards"][symbol] = history["episode_rewards"]
                    self.history["episode_lengths"][symbol] = history["episode_lengths"]
                    self.history["losses"][symbol] = history["losses"]
                    self.history["eval_rewards"][symbol] = history["eval_rewards"]
                    self.history["timestamps"][symbol] = history["timestamps"]
                    
                    self.logger.info(f"Đã hoàn thành huấn luyện cho {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"Lỗi khi huấn luyện: {str(e)}")
        
        # Tải lại các agent đã huấn luyện
        self._reload_trained_agents()
        
        # Đánh giá tổng hợp
        self._evaluate_all()
        
        # Đánh dấu hoàn thành
        self.training_complete = True
        
        # Lưu lịch sử huấn luyện
        self._save_combined_history()
        
        # Vẽ biểu đồ kết quả
        self._plot_combined_results()
        
        # Xuất báo cáo
        self.export_report()
        
        return self.history
    
    def _train_symbol_wrapper(self, params: Tuple[str, str, int]) -> Tuple[str, Dict[str, Any]]:
        """
        Wrapper cho quá trình huấn luyện một symbol, được sử dụng trong song song.
        
        Args:
            params: Tuple chứa (symbol, trainer_path, num_episodes)
            
        Returns:
            Tuple (symbol, history)
        """
        try:
            symbol, trainer_path, num_episodes = params
            
            # Tải lại trainer
            trainer = Trainer.load_from_config(trainer_path)
            
            # Huấn luyện
            trainer.train()
            
            # Trả về kết quả
            return symbol, trainer.history
            
        except Exception as e:
            self.logger.error(f"Lỗi khi huấn luyện symbol {params[0]}: {str(e)}")
            raise
    
    def _reload_trained_agents(self) -> None:
        """
        Tải lại các agent đã huấn luyện sau quá trình huấn luyện song song.
        """
        for symbol in self.trainers.keys():
            try:
                # Đường dẫn lưu mô hình
                symbol_dir = self.output_dir / symbol.replace('/', '_')
                model_path = symbol_dir / "models" / "best_model"
                
                # Tải lại agent
                if model_path.exists():
                    self.agents[symbol].load(str(model_path))
                    self.logger.info(f"Đã tải lại agent cho {symbol} từ {model_path}")
                else:
                    self.logger.warning(f"Không tìm thấy mô hình cho {symbol} tại {model_path}")
                    
            except Exception as e:
                self.logger.error(f"Lỗi khi tải lại agent cho {symbol}: {str(e)}")
    
    def train_mixed(self) -> Dict[str, Any]:
        """
        Huấn luyện theo chế độ mixed (kết hợp tuần tự và song song).
        Huấn luyện tuần tự trên từng cặp tiền, nhưng mỗi cặp huấn luyện song song trên các phần dữ liệu.
        
        Returns:
            Dict chứa lịch sử huấn luyện
        """
        self.logger.info(f"Bắt đầu huấn luyện mixed trên {len(self.trainers)} cặp tiền")
        
        # Lấy các tham số từ cấu hình
        num_episodes = self.config["num_episodes_per_symbol"]
        
        # Thời gian bắt đầu
        start_time = time.time()
        
        # Huấn luyện lần lượt trên từng cặp tiền, nhưng song song trên các phần dữ liệu
        for i, (symbol, trainer) in enumerate(self.trainers.items()):
            self.logger.info(f"[{i+1}/{len(self.trainers)}] Bắt đầu huấn luyện cho {symbol}")
            
            # Cập nhật symbol hiện tại
            self.current_symbol = symbol
            
            # Nếu là transfer learning và không phải cặp tiền đầu tiên
            if self.transfer_learning and i > 0:
                # Lấy symbol trước đó
                prev_symbol = list(self.trainers.keys())[i-1]
                prev_agent = self.agents[prev_symbol]
                
                # Sao chép trọng số từ agent trước sang agent hiện tại
                self._transfer_weights(prev_agent, self.agents[symbol])
                self.logger.info(f"Đã áp dụng transfer learning từ {prev_symbol} sang {symbol}")
            
            # Chia dữ liệu thành các phần
            data = self.data_sources[symbol]
            data_chunks = self._split_data(data)
            
            # Khởi tạo các môi trường và agent cho các phần dữ liệu
            chunk_envs = []
            chunk_agents = []
            chunk_trainers = []
            
            for j, chunk in enumerate(data_chunks):
                # Tạo môi trường mới với phần dữ liệu
                env_kwargs = self.kwargs.get("env_kwargs", {}).copy()
                chunk_env = self.env_class(
                    data=chunk,
                    symbol=symbol,
                    logger=self.logger,
                    **env_kwargs
                )
                
                # Sao chép agent
                chunk_agent = self._clone_agent(self.agents[symbol])
                
                # Tạo trainer
                trainer_kwargs = self.kwargs.get("trainer_kwargs", {}).copy()
                chunk_dir = self.output_dir / symbol.replace('/', '_') / f"chunk_{j}"
                chunk_dir.mkdir(exist_ok=True, parents=True)
                
                chunk_trainer = Trainer(
                    agent=chunk_agent,
                    env=chunk_env,
                    output_dir=chunk_dir,
                    experiment_name=f"{self.experiment_name}_{symbol.replace('/', '_')}_chunk_{j}",
                    **trainer_kwargs
                )
                
                chunk_envs.append(chunk_env)
                chunk_agents.append(chunk_agent)
                chunk_trainers.append(chunk_trainer)
            
            # Huấn luyện song song trên các phần dữ liệu
            self._train_chunks_parallel(chunk_trainers)
            
            # Kết hợp các agent
            self._combine_agents(self.agents[symbol], chunk_agents)
            
            # Đánh giá kết quả
            eval_rewards = trainer.evaluate()
            
            # Cập nhật lịch sử
            self.history["episode_rewards"][symbol] = trainer.history["episode_rewards"]
            self.history["episode_lengths"][symbol] = trainer.history["episode_lengths"]
            self.history["losses"][symbol] = trainer.history["losses"]
            self.history["eval_rewards"][symbol] = [np.mean(eval_rewards)]
            self.history["timestamps"][symbol] = trainer.history["timestamps"]
            
            # Lưu lịch sử huấn luyện tổng hợp
            self._save_combined_history()
            
            # Đánh giá cross-validation (kiểm tra chéo)
            if i > 0:
                self._cross_validate(symbol)
        
        # Đánh giá tổng hợp
        self._evaluate_all()
        
        # Đánh dấu hoàn thành
        self.training_complete = True
        
        # Lưu lịch sử huấn luyện cuối cùng
        self._save_combined_history()
        
        # Vẽ biểu đồ kết quả
        self._plot_combined_results()
        
        # Xuất báo cáo
        self.export_report()
        
        return self.history
    
    def _split_data(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Chia DataFrame thành các phần cho huấn luyện song song.
        
        Args:
            data: DataFrame dữ liệu gốc
            
        Returns:
            Danh sách các DataFrame con
        """
        # Số lượng phần mặc định
        n_chunks = os.cpu_count() or 4
        
        # Giới hạn số lượng phần dựa trên kích thước dữ liệu
        n_chunks = min(n_chunks, len(data) // 1000)
        n_chunks = max(1, n_chunks)  # Ít nhất 1 phần
        
        # Chia dữ liệu
        chunk_size = len(data) // n_chunks
        chunks = []
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < n_chunks - 1 else len(data)
            chunks.append(data.iloc[start_idx:end_idx].copy())
        
        return chunks
    
    def _clone_agent(self, agent: BaseAgent) -> BaseAgent:
        """
        Tạo bản sao của agent.
        
        Args:
            agent: Agent cần sao chép
            
        Returns:
            Agent mới
        """
        # Sử dụng cùng tham số
        cloned_agent = self.agent_class(**agent.config)
        
        # Sao chép trọng số từ agent gốc
        self._transfer_weights(agent, cloned_agent)
        
        return cloned_agent
    
    def _train_chunks_parallel(self, trainers: List[Trainer]) -> None:
        """
        Huấn luyện song song trên các phần dữ liệu.
        
        Args:
            trainers: Danh sách các trainer cho từng phần dữ liệu
        """
        if len(trainers) == 1:
            # Nếu chỉ có 1 trainer, huấn luyện tuần tự
            trainers[0].train()
            return
        
        # Xác định số lượng công việc song song
        n_jobs = min(len(trainers), os.cpu_count() or 4)
        
        # Huấn luyện song song
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Chuẩn bị các tham số cho hàm huấn luyện
            trainer_paths = []
            for i, trainer in enumerate(trainers):
                # Lưu trainer vào disk để tránh lỗi serialization
                trainer_dir = trainer.output_dir
                trainer_path = trainer_dir / "trainer.pkl"
                
                # Lưu trainer
                trainer.save_config(str(trainer_path))
                trainer_paths.append((i, str(trainer_path)))
            
            # Chạy các công việc song song
            futures = [executor.submit(self._train_chunk_wrapper, param) for param in trainer_paths]
            
            # Thu thập kết quả
            for future in concurrent.futures.as_completed(futures):
                try:
                    chunk_idx = future.result()
                    self.logger.info(f"Đã hoàn thành huấn luyện cho chunk {chunk_idx}")
                    
                except Exception as e:
                    self.logger.error(f"Lỗi khi huấn luyện chunk: {str(e)}")
    
    def _train_chunk_wrapper(self, params: Tuple[int, str]) -> int:
        """
        Wrapper cho quá trình huấn luyện một phần dữ liệu, được sử dụng trong song song.
        
        Args:
            params: Tuple chứa (chunk_idx, trainer_path)
            
        Returns:
            chunk_idx
        """
        try:
            chunk_idx, trainer_path = params
            
            # Tải lại trainer
            trainer = Trainer.load_from_config(trainer_path)
            
            # Huấn luyện
            trainer.train()
            
            # Trả về kết quả
            return chunk_idx
            
        except Exception as e:
            self.logger.error(f"Lỗi khi huấn luyện chunk {params[0]}: {str(e)}")
            raise
    
    def _combine_agents(self, target_agent: BaseAgent, source_agents: List[BaseAgent]) -> None:
        """
        Kết hợp nhiều agents thành một.
        
        Args:
            target_agent: Agent đích
            source_agents: Danh sách các agent nguồn
        """
        if not source_agents:
            return
        
        try:
            # Lấy trọng số từ tất cả các agent
            all_weights = []
            for agent in source_agents:
                # Kiểm tra xem agent có phương thức get_weights không
                if hasattr(agent, "get_weights"):
                    all_weights.append(agent.get_weights())
                # Hoặc sử dụng cách tiếp cận dựa trên mạng network
                elif hasattr(agent, "q_network") and hasattr(agent.q_network, "get_weights"):
                    all_weights.append(agent.q_network.get_weights())
                elif hasattr(agent, "policy_network") and hasattr(agent.policy_network, "get_weights"):
                    all_weights.append(agent.policy_network.get_weights())
                elif hasattr(agent, "network") and hasattr(agent.network, "get_weights"):
                    all_weights.append(agent.network.get_weights())
            
            if not all_weights:
                self.logger.warning("Không thể lấy trọng số từ các agent nguồn")
                return
            
            # Tính trung bình của các trọng số
            avg_weights = []
            for i in range(len(all_weights[0])):
                # Tính trung bình cho mỗi lớp trọng số
                layer_weights = []
                for weights in all_weights:
                    layer_weights.append(weights[i])
                
                avg_weights.append(np.mean(layer_weights, axis=0))
            
            # Đặt trọng số cho agent đích
            if hasattr(target_agent, "set_weights"):
                target_agent.set_weights(avg_weights)
            elif hasattr(target_agent, "q_network") and hasattr(target_agent.q_network, "set_weights"):
                target_agent.q_network.set_weights(avg_weights)
                if hasattr(target_agent, "target_q_network"):
                    target_agent.target_q_network.set_weights(avg_weights)
            elif hasattr(target_agent, "policy_network") and hasattr(target_agent.policy_network, "set_weights"):
                target_agent.policy_network.set_weights(avg_weights)
            elif hasattr(target_agent, "network") and hasattr(target_agent.network, "set_weights"):
                target_agent.network.set_weights(avg_weights)
            
            self.logger.info(f"Đã kết hợp {len(source_agents)} agents")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi kết hợp các agents: {str(e)}")
    
    def _transfer_weights(self, source_agent: BaseAgent, target_agent: BaseAgent) -> None:
        """
        Chuyển trọng số từ source_agent sang target_agent.
        
        Args:
            source_agent: Agent nguồn
            target_agent: Agent đích
        """
        try:
            # Kiểm tra xem agent có phương thức transfer_weights không
            if hasattr(source_agent, "transfer_weights") and hasattr(target_agent, "transfer_weights"):
                source_agent.transfer_weights(target_agent)
                return
            
            # Hoặc sử dụng cách tiếp cận dựa trên mạng network
            if hasattr(source_agent, "get_weights") and hasattr(target_agent, "set_weights"):
                weights = source_agent.get_weights()
                target_agent.set_weights(weights)
            elif hasattr(source_agent, "q_network") and hasattr(target_agent, "q_network"):
                weights = source_agent.q_network.get_weights()
                target_agent.q_network.set_weights(weights)
                
                # Cập nhật target network nếu có
                if hasattr(source_agent, "target_q_network") and hasattr(target_agent, "target_q_network"):
                    target_weights = source_agent.target_q_network.get_weights()
                    target_agent.target_q_network.set_weights(target_weights)
            elif hasattr(source_agent, "policy_network") and hasattr(target_agent, "policy_network"):
                policy_weights = source_agent.policy_network.get_weights()
                target_agent.policy_network.set_weights(policy_weights)
                
                # Chuyển trọng số cho mạng value nếu có
                if hasattr(source_agent, "value_network") and hasattr(target_agent, "value_network"):
                    value_weights = source_agent.value_network.get_weights()
                    target_agent.value_network.set_weights(value_weights)
            elif hasattr(source_agent, "network") and hasattr(target_agent, "network"):
                weights = source_agent.network.get_weights()
                target_agent.network.set_weights(weights)
            else:
                self.logger.warning("Không thể chuyển trọng số giữa các agent")
                return
            
            self.logger.info("Đã chuyển trọng số thành công")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi chuyển trọng số: {str(e)}")
    
    def train(self) -> Dict[str, Any]:
        """
        Huấn luyện agent trên tất cả các cặp tiền dựa trên chế độ đã chọn.
        
        Returns:
            Dict chứa lịch sử huấn luyện
        """
        if self.training_mode == "sequential":
            return self.train_sequential()
        elif self.training_mode == "parallel":
            return self.train_parallel()
        elif self.training_mode == "mixed":
            return self.train_mixed()
        else:
            self.logger.error(f"Chế độ huấn luyện không hợp lệ: {self.training_mode}")
            self.logger.info("Chuyển sang chế độ huấn luyện tuần tự")
            return self.train_sequential()
    
    def _cross_validate(self, symbol: str) -> None:
        """
        Đánh giá chéo agent trên các cặp tiền khác.
        
        Args:
            symbol: Symbol hiện tại đang đánh giá
        """
        self.logger.info(f"Bắt đầu đánh giá chéo cho {symbol}")
        
        # Tạo thư mục cho cross validation
        cross_val_dir = self.output_dir / "cross_validation"
        cross_val_dir.mkdir(exist_ok=True)
        
        # Tạo DataFrame để lưu kết quả
        results = []
        
        # Lấy agent hiện tại
        current_agent = self.agents[symbol]
        
        # Đánh giá trên tất cả các môi trường đã huấn luyện trước đó
        for test_symbol, test_env in self.environments.items():
            if test_symbol == symbol:
                continue  # Bỏ qua đánh giá trên chính nó
            
            # Đánh giá
            rewards = []
            for _ in range(self.n_eval_episodes):
                state = test_env.reset()
                episode_reward = 0
                done = False
                step = 0
                
                while not done and step < self.config["max_steps_per_episode"]:
                    # Lấy hành động từ agent hiện tại
                    action = current_agent.act(state, explore=False)
                    
                    # Thực hiện hành động trong môi trường test
                    next_state, reward, done, info = test_env.step(action)
                    
                    # Cập nhật
                    state = next_state
                    episode_reward += reward
                    step += 1
                
                rewards.append(episode_reward)
            
            # Lưu kết quả
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            
            results.append({
                "train_symbol": symbol,
                "test_symbol": test_symbol,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "min_reward": np.min(rewards),
                "max_reward": np.max(rewards)
            })
            
            self.logger.info(f"Cross-val {symbol} -> {test_symbol}: Mean={mean_reward:.2f}, Std={std_reward:.2f}")
        
        # Lưu kết quả
        if results:
            df = pd.DataFrame(results)
            csv_path = cross_val_dir / f"{symbol}_cross_val.csv"
            df.to_csv(csv_path, index=False)
            
            self.logger.info(f"Đã lưu kết quả cross-validation cho {symbol} tại {csv_path}")
    
    def _evaluate_all(self) -> None:
        """
        Đánh giá tổng hợp tất cả các agent trên tất cả các môi trường.
        """
        self.logger.info("Bắt đầu đánh giá tổng hợp")
        
        # Tạo thư mục cho đánh giá tổng hợp
        eval_dir = self.output_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        # Tạo DataFrame để lưu kết quả
        results = []
        
        # Đánh giá từng agent trên từng môi trường
        for agent_symbol, agent in self.agents.items():
            for env_symbol, env in self.environments.items():
                # Đánh giá
                rewards = []
                for _ in range(self.n_eval_episodes):
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    step = 0
                    
                    while not done and step < self.config["max_steps_per_episode"]:
                        # Lấy hành động từ agent
                        action = agent.act(state, explore=False)
                        
                        # Thực hiện hành động trong môi trường
                        next_state, reward, done, info = env.step(action)
                        
                        # Cập nhật
                        state = next_state
                        episode_reward += reward
                        step += 1
                    
                    rewards.append(episode_reward)
                
                # Lưu kết quả
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                
                results.append({
                    "agent_symbol": agent_symbol,
                    "env_symbol": env_symbol,
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "min_reward": np.min(rewards),
                    "max_reward": np.max(rewards)
                })
                
                self.logger.info(f"Evaluation {agent_symbol} -> {env_symbol}: Mean={mean_reward:.2f}, Std={std_reward:.2f}")
        
        # Lưu kết quả
        if results:
            df = pd.DataFrame(results)
            csv_path = eval_dir / "all_evaluations.csv"
            df.to_csv(csv_path, index=False)
            
            # Tạo ma trận đánh giá
            pivot_df = df.pivot(index="agent_symbol", columns="env_symbol", values="mean_reward")
            pivot_path = eval_dir / "evaluation_matrix.csv"
            pivot_df.to_csv(pivot_path)
            
            # Vẽ heatmap
            self._plot_evaluation_heatmap(pivot_df)
            
            self.logger.info(f"Đã lưu kết quả đánh giá tổng hợp tại {eval_dir}")
    
    def _plot_evaluation_heatmap(self, pivot_df: pd.DataFrame) -> None:
        """
        Vẽ heatmap cho ma trận đánh giá.
        
        Args:
            pivot_df: DataFrame dạng pivot chứa kết quả đánh giá
        """
        try:
            # Tạo hình
            plt.figure(figsize=(10, 8))
            
            # Vẽ heatmap
            im = plt.imshow(pivot_df.values, cmap='YlGn')
            
            # Thêm colorbar
            plt.colorbar(im, label='Mean Reward')
            
            # Thêm nhãn trục
            plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=45, ha='right')
            plt.yticks(range(len(pivot_df.index)), pivot_df.index)
            
            # Thêm giá trị vào các ô
            for i in range(len(pivot_df.index)):
                for j in range(len(pivot_df.columns)):
                    value = pivot_df.values[i, j]
                    plt.text(j, i, f"{value:.2f}", ha='center', va='center',
                            color='white' if value < pivot_df.values.mean() else 'black')
            
            # Tiêu đề và nhãn
            plt.title("Cross-Market Evaluation Matrix")
            plt.xlabel("Environment Symbol")
            plt.ylabel("Agent Symbol")
            
            # Điều chỉnh layout
            plt.tight_layout()
            
            # Lưu hình
            eval_dir = self.output_dir / "evaluation"
            plt.savefig(eval_dir / "evaluation_heatmap.png")
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ heatmap đánh giá: {str(e)}")
    
    def _save_combined_history(self) -> None:
        """
        Lưu lịch sử huấn luyện tổng hợp.
        """
        # Tạo DataFrame cho từng loại metric
        for metric in ["episode_rewards", "episode_lengths", "losses", "eval_rewards"]:
            try:
                # Tạo DataFrame
                metric_data = {}
                
                for symbol, values in self.history[metric].items():
                    if values:  # Kiểm tra xem có dữ liệu không
                        metric_data[symbol] = values
                
                if not metric_data:
                    continue
                
                # Tạo DataFrame
                df = pd.DataFrame(metric_data)
                
                # Lưu vào file
                csv_path = self.output_dir / f"combined_{metric}.csv"
                df.to_csv(csv_path, index=True)
                
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu {metric}: {str(e)}")
        
        # Lưu cấu trúc lịch sử đầy đủ dưới dạng JSON
        try:
            # Chuyển đổi history thành dạng có thể serialize
            serializable_history = {}
            
            for key, value in self.history.items():
                if key == "timestamps":
                    # Chuyển timestamps thành chuỗi
                    serializable_history[key] = {
                        symbol: [str(t) for t in times] for symbol, times in value.items()
                    }
                else:
                    # Chuyển các giá trị numpy thành list
                    serializable_history[key] = {
                        symbol: [float(v) if isinstance(v, np.number) else v for v in values]
                        for symbol, values in value.items()
                    }
            
            # Lưu vào file
            json_path = self.output_dir / "training_history.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "experiment_name": self.experiment_name,
                    "symbols": self.symbols,
                    "config": self.config,
                    "history": serializable_history,
                    "training_complete": self.training_complete,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=4, default=str)
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu lịch sử huấn luyện: {str(e)}")
    
    def _plot_combined_results(self) -> None:
        """
        Vẽ biểu đồ kết quả tổng hợp.
        """
        # Tạo thư mục plots
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Biểu đồ rewards theo từng episode
        plt.figure(figsize=(12, 6))
        
        for symbol, rewards in self.history["episode_rewards"].items():
            if rewards:  # Kiểm tra xem có dữ liệu không
                plt.plot(rewards, label=symbol)
        
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Rewards Across Markets")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(plots_dir / "combined_rewards.png")
        plt.close()
        
        # 2. Biểu đồ độ dài episode
        plt.figure(figsize=(12, 6))
        
        for symbol, lengths in self.history["episode_lengths"].items():
            if lengths:  # Kiểm tra xem có dữ liệu không
                plt.plot(lengths, label=symbol)
        
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.title("Episode Lengths Across Markets")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(plots_dir / "combined_lengths.png")
        plt.close()
        
        # 3. Biểu đồ losses
        plt.figure(figsize=(12, 6))
        
        for symbol, losses in self.history["losses"].items():
            if losses:  # Kiểm tra xem có dữ liệu không
                plt.plot(losses, label=symbol)
        
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Training Losses Across Markets")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Sử dụng log scale nếu có sự khác biệt lớn
        max_loss = max([max(losses) if losses else 0 for losses in self.history["losses"].values()])
        min_loss = min([min(filter(lambda x: x > 0, losses)) if losses else 1 
                        for losses in self.history["losses"].values()])
        
        if max_loss > 10 * min_loss:
            plt.yscale('log')
        
        plt.savefig(plots_dir / "combined_losses.png")
        plt.close()
        
        # 4. Biểu đồ đánh giá
        plt.figure(figsize=(12, 6))
        
        for symbol, eval_rewards in self.history["eval_rewards"].items():
            if eval_rewards:  # Kiểm tra xem có dữ liệu không
                # Tính toán các episodes đánh giá
                eval_episodes = range(
                    self.evaluation_interval, 
                    self.evaluation_interval * (len(eval_rewards) + 1), 
                    self.evaluation_interval
                )
                
                plt.plot(eval_episodes, eval_rewards, 'o-', label=symbol)
        
        plt.xlabel("Episode")
        plt.ylabel("Evaluation Reward")
        plt.title("Evaluation Rewards Across Markets")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(plots_dir / "combined_evaluations.png")
        plt.close()
        
        self.logger.info(f"Đã tạo các biểu đồ kết quả tổng hợp tại {plots_dir}")
    
    def export_report(self) -> str:
        """
        Xuất báo cáo huấn luyện đa cặp tiền dạng Markdown.
        
        Returns:
            Đường dẫn file báo cáo
        """
        report_path = self.output_dir / "cross_coin_training_report.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            # Tiêu đề
            f.write(f"# Báo cáo Huấn luyện Đa Cặp Tiền: {self.experiment_name}\n\n")
            
            # Thông tin chung
            f.write("## Thông tin chung\n\n")
            f.write(f"- **Thời gian:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Agent:** {self.agent_class.__name__}\n")
            f.write(f"- **Môi trường:** {self.env_class.__name__}\n")
            f.write(f"- **Cặp tiền tệ:** {', '.join(self.symbols)}\n")
            f.write(f"- **Chế độ huấn luyện:** {self.training_mode}\n")
            f.write(f"- **Transfer learning:** {self.transfer_learning}\n")
            f.write(f"- **Trạng thái:** {'Hoàn thành' if self.training_complete else 'Đang huấn luyện'}\n\n")
            
            # Kết quả huấn luyện
            f.write("## Kết quả huấn luyện\n\n")
            
            # Bảng tổng quan
            f.write("### Bảng tổng quan\n\n")
            f.write("| Cặp tiền | Số episodes | Reward trung bình | Reward đánh giá |\n")
            f.write("|----------|-------------|-------------------|----------------|\n")
            
            for symbol in self.symbols:
                if symbol in self.history["episode_rewards"] and self.history["episode_rewards"][symbol]:
                    n_episodes = len(self.history["episode_rewards"][symbol])
                    mean_reward = np.mean(self.history["episode_rewards"][symbol])
                    
                    eval_reward = "N/A"
                    if symbol in self.history["eval_rewards"] and self.history["eval_rewards"][symbol]:
                        eval_reward = f"{self.history['eval_rewards'][symbol][-1]:.2f}"
                    
                    f.write(f"| {symbol} | {n_episodes} | {mean_reward:.2f} | {eval_reward} |\n")
            
            f.write("\n")
            
            # Biểu đồ
            plots_dir = self.output_dir / "plots"
            if plots_dir.exists():
                f.write("## Biểu đồ huấn luyện\n\n")
                
                # Biểu đồ rewards
                rewards_plot = plots_dir / "combined_rewards.png"
                if rewards_plot.exists():
                    f.write("### Rewards theo episode\n\n")
                    f.write(f"![Combined Rewards](plots/combined_rewards.png)\n\n")
                
                # Biểu đồ độ dài episode
                lengths_plot = plots_dir / "combined_lengths.png"
                if lengths_plot.exists():
                    f.write("### Độ dài episode\n\n")
                    f.write(f"![Combined Lengths](plots/combined_lengths.png)\n\n")
                
                # Biểu đồ losses
                losses_plot = plots_dir / "combined_losses.png"
                if losses_plot.exists():
                    f.write("### Losses\n\n")
                    f.write(f"![Combined Losses](plots/combined_losses.png)\n\n")
                
                # Biểu đồ đánh giá
                evals_plot = plots_dir / "combined_evaluations.png"
                if evals_plot.exists():
                    f.write("### Đánh giá\n\n")
                    f.write(f"![Combined Evaluations](plots/combined_evaluations.png)\n\n")
            
            # Đánh giá tổng hợp
            eval_dir = self.output_dir / "evaluation"
            if eval_dir.exists() and (eval_dir / "evaluation_heatmap.png").exists():
                f.write("## Đánh giá tổng hợp\n\n")
                f.write("### Ma trận đánh giá chéo\n\n")
                f.write(f"![Evaluation Heatmap](evaluation/evaluation_heatmap.png)\n\n")
                f.write("Ma trận đánh giá chéo hiển thị hiệu suất của mỗi agent (trục dọc) trên mỗi môi trường (trục ngang).\n")
                f.write("Giá trị cao hơn (màu xanh đậm hơn) thể hiện hiệu suất tốt hơn.\n\n")
            
            # Cấu hình huấn luyện
            f.write("## Cấu hình huấn luyện\n\n")
            f.write("```json\n")
            json.dump(self.config, f, indent=2, ensure_ascii=False)
            f.write("\n```\n\n")
            
            # Kết luận
            f.write("## Kết luận và nhận xét\n\n")
            
            if self.training_complete:
                # Đọc ma trận đánh giá nếu có
                eval_matrix_path = eval_dir / "evaluation_matrix.csv"
                conclusions = []
                
                if eval_matrix_path.exists():
                    try:
                        matrix_df = pd.read_csv(eval_matrix_path, index_col=0)
                        
                        # Tìm agent tốt nhất trên tất cả các thị trường
                        best_agent = matrix_df.mean(axis=1).idxmax()
                        conclusions.append(f"- Agent được huấn luyện trên {best_agent} có hiệu suất tốt nhất trên tất cả các thị trường.")
                        
                        # Tìm thị trường dễ dàng nhất
                        easiest_market = matrix_df.mean(axis=0).idxmax()
                        conclusions.append(f"- Thị trường {easiest_market} dường như dễ dự đoán nhất với hiệu suất cao nhất từ các agents.")
                        
                        # Tìm thị trường khó khăn nhất
                        hardest_market = matrix_df.mean(axis=0).idxmin()
                        conclusions.append(f"- Thị trường {hardest_market} có vẻ khó dự đoán nhất với hiệu suất thấp nhất từ các agents.")
                        
                        # Kiểm tra khả năng tổng quát hóa
                        diag_mean = np.mean(np.diag(matrix_df.values))
                        offdiag_mean = (matrix_df.values.sum() - np.sum(np.diag(matrix_df.values))) / (matrix_df.size - len(matrix_df))
                        
                        if diag_mean > offdiag_mean * 1.2:
                            conclusions.append("- Các agents có xu hướng hoạt động tốt nhất trên thị trường mà chúng được huấn luyện, cho thấy một số đặc điểm riêng biệt của từng thị trường.")
                        else:
                            conclusions.append("- Các agents thể hiện khả năng tổng quát hóa tốt giữa các thị trường, cho thấy có nhiều điểm tương đồng giữa các cặp tiền.")
                        
                        # Đánh giá transfer learning
                        if self.transfer_learning:
                            transfer_effectiveness = "hiệu quả" if offdiag_mean > diag_mean * 0.8 else "hạn chế"
                            conclusions.append(f"- Transfer learning có hiệu quả {transfer_effectiveness} trong việc chuyển giao kiến thức giữa các thị trường.")
                    
                    except Exception as e:
                        self.logger.error(f"Lỗi khi phân tích ma trận đánh giá: {str(e)}")
                
                if conclusions:
                    for conclusion in conclusions:
                        f.write(f"{conclusion}\n")
                else:
                    f.write("Quá trình huấn luyện đa cặp tiền đã hoàn thành. Đánh giá chi tiết về hiệu suất của từng agent trên các thị trường khác nhau có thể được tìm thấy trong thư mục evaluation.\n")
            else:
                f.write("Quá trình huấn luyện đa cặp tiền vẫn đang tiếp tục. Báo cáo này được tạo làm mốc trung gian.\n")
            
            f.write("\n### Các bước tiếp theo\n\n")
            f.write("- Phân tích sâu hơn về hiệu suất của từng cặp tiền\n")
            f.write("- Tối ưu hóa siêu tham số cho từng thị trường cụ thể\n")
            f.write("- Thử nghiệm với các loại mô hình khác để so sánh hiệu suất\n")
            f.write("- Đánh giá trên dữ liệu thời gian thực để kiểm tra độ tin cậy\n")
        
        self.logger.info(f"Đã xuất báo cáo huấn luyện đa cặp tiền tại {report_path}")
        
        return str(report_path)
    
    def get_best_agent(self, metric: str = "eval_rewards") -> Tuple[str, BaseAgent]:
        """
        Lấy agent tốt nhất dựa trên metric.
        
        Args:
            metric: Metric để đánh giá ('eval_rewards', 'episode_rewards')
            
        Returns:
            Tuple (symbol, agent)
        """
        best_score = float('-inf')
        best_symbol = None
        
        for symbol, agent in self.agents.items():
            if symbol in self.history[metric] and self.history[metric][symbol]:
                # Lấy giá trị cuối cùng của metric
                score = self.history[metric][symbol][-1]
                
                if score > best_score:
                    best_score = score
                    best_symbol = symbol
        
        if best_symbol is None:
            self.logger.warning(f"Không tìm thấy agent tốt nhất dựa trên {metric}")
            # Trả về agent đầu tiên nếu không tìm thấy
            if self.agents:
                first_symbol = next(iter(self.agents.keys()))
                return first_symbol, self.agents[first_symbol]
            return None, None
        
        return best_symbol, self.agents[best_symbol]
    
    def get_best_universal_agent(self) -> Tuple[str, BaseAgent]:
        """
        Lấy agent có hiệu suất tốt nhất trên tất cả các thị trường.
        
        Returns:
            Tuple (symbol, agent)
        """
        # Đọc ma trận đánh giá nếu có
        eval_matrix_path = self.output_dir / "evaluation" / "evaluation_matrix.csv"
        
        if not eval_matrix_path.exists():
            self.logger.warning("Không tìm thấy ma trận đánh giá, sử dụng get_best_agent thay thế")
            return self.get_best_agent()
        
        try:
            matrix_df = pd.read_csv(eval_matrix_path, index_col=0)
            
            # Tính điểm trung bình cho mỗi agent
            mean_scores = matrix_df.mean(axis=1)
            
            # Tìm agent tốt nhất
            best_symbol = mean_scores.idxmax()
            
            return best_symbol, self.agents[best_symbol]
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tìm agent tốt nhất: {str(e)}")
            return self.get_best_agent()
    
    def create_ensemble_agent(self, weights: Optional[Dict[str, float]] = None) -> BaseAgent:
        """
        Tạo một agent tổng hợp từ tất cả các agent đã huấn luyện.
        
        Args:
            weights: Dict trọng số cho từng agent (None để sử dụng trọng số bằng nhau)
            
        Returns:
            Agent tổng hợp
        """
        # Kiểm tra xem có đủ agent không
        if len(self.agents) < 2:
            self.logger.warning("Không đủ agents để tạo ensemble, trả về agent duy nhất")
            if self.agents:
                return next(iter(self.agents.values()))
            return None
        
        # Thiết lập trọng số mặc định nếu không được cung cấp
        if weights is None:
            weights = {symbol: 1.0 / len(self.agents) for symbol in self.agents.keys()}
        
        # Kiểm tra trọng số
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            # Chuẩn hóa trọng số
            weights = {symbol: w / total_weight for symbol, w in weights.items()}
        
        # Tạo agent mới với cùng cấu hình
        first_agent = next(iter(self.agents.values()))
        ensemble_agent = self._clone_agent(first_agent)
        
        # Kết hợp trọng số của tất cả các agent
        try:
            # Lấy tất cả các lớp trọng số
            all_weights = []
            
            # Xác định phương thức lấy trọng số
            if hasattr(first_agent, "get_weights"):
                get_weights_fn = lambda a: a.get_weights()
            elif hasattr(first_agent, "q_network") and hasattr(first_agent.q_network, "get_weights"):
                get_weights_fn = lambda a: a.q_network.get_weights()
            elif hasattr(first_agent, "policy_network") and hasattr(first_agent.policy_network, "get_weights"):
                get_weights_fn = lambda a: a.policy_network.get_weights()
            elif hasattr(first_agent, "network") and hasattr(first_agent.network, "get_weights"):
                get_weights_fn = lambda a: a.network.get_weights()
            else:
                self.logger.error("Không thể xác định phương thức lấy trọng số")
                return first_agent
            
            # Lấy trọng số từ agent đầu tiên để xác định cấu trúc
            first_weights = get_weights_fn(first_agent)
            
            # Khởi tạo trọng số tổng hợp với giá trị 0
            ensemble_weights = [np.zeros_like(w) for w in first_weights]
            
            # Tính trọng số tổng hợp
            for symbol, agent in self.agents.items():
                if symbol not in weights:
                    continue
                
                agent_weight = weights[symbol]
                agent_weights = get_weights_fn(agent)
                
                # Kiểm tra kích thước trọng số
                if len(agent_weights) != len(ensemble_weights):
                    self.logger.warning(f"Agent {symbol} có cấu trúc trọng số khác, bỏ qua")
                    continue
                
                # Cộng dồn trọng số
                for i in range(len(ensemble_weights)):
                    if agent_weights[i].shape != ensemble_weights[i].shape:
                        self.logger.warning(f"Agent {symbol} có kích thước lớp {i} khác, bỏ qua")
                        continue
                    
                    ensemble_weights[i] += agent_weights[i] * agent_weight
            
            # Đặt trọng số cho agent tổng hợp
            if hasattr(ensemble_agent, "set_weights"):
                ensemble_agent.set_weights(ensemble_weights)
            elif hasattr(ensemble_agent, "q_network") and hasattr(ensemble_agent.q_network, "set_weights"):
                ensemble_agent.q_network.set_weights(ensemble_weights)
                if hasattr(ensemble_agent, "target_q_network"):
                    ensemble_agent.target_q_network.set_weights(ensemble_weights)
            elif hasattr(ensemble_agent, "policy_network") and hasattr(ensemble_agent.policy_network, "set_weights"):
                ensemble_agent.policy_network.set_weights(ensemble_weights)
                if hasattr(ensemble_agent, "value_network"):
                    ensemble_agent.value_network.set_weights(ensemble_weights)
            elif hasattr(ensemble_agent, "network") and hasattr(ensemble_agent.network, "set_weights"):
                ensemble_agent.network.set_weights(ensemble_weights)
            
            self.logger.info("Đã tạo agent tổng hợp thành công")
            
            return ensemble_agent
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo agent tổng hợp: {str(e)}")
            return first_agent