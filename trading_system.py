"""
Hệ thống giao dịch tự động.
File này chứa lớp AutomatedTradingSystem là lớp trung tâm điều phối
các thành phần khác nhau của hệ thống.
"""

import os
import time
import logging
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config, DATA_DIR, MODEL_DIR
from config.security_config import get_security_config

# Import các module xử lý dữ liệu
from data_processors.data_pipeline import DataPipeline

# Import các module thu thập dữ liệu
try:
    from data_collectors.market_data.historical_data_collector import create_data_collector
    DATA_COLLECTORS_AVAILABLE = True
except ImportError:
    DATA_COLLECTORS_AVAILABLE = False

# Import môi trường huấn luyện khi có
try:
    from environments.trading_gym.trading_env import TradingEnv
    ENVIRONMENTS_AVAILABLE = True
except ImportError:
    ENVIRONMENTS_AVAILABLE = False

# Import các agent khi có
try:
    from models.agents.dqn_agent import DQNAgent
    from models.training_pipeline.trainer import Trainer
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

class AutomatedTradingSystem:
    """
    Lớp chính điều phối hệ thống giao dịch tự động.
    Quản lý luồng dữ liệu, huấn luyện và giao dịch.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data_dir: Optional[Path] = None,
        model_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo hệ thống giao dịch tự động.
        
        Args:
            config: Cấu hình hệ thống
            data_dir: Thư mục dữ liệu
            model_dir: Thư mục chứa mô hình
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("trading_system")
        
        # Lấy cấu hình hệ thống nếu không được cung cấp
        if config is None:
            config = get_system_config().get_all()
        self.config = config
        
        # Thiết lập thư mục dữ liệu
        if data_dir is None:
            data_dir = DATA_DIR
        self.data_dir = Path(data_dir)
        
        # Thiết lập thư mục mô hình
        if model_dir is None:
            model_dir = MODEL_DIR
        self.model_dir = Path(model_dir)
        
        # Đảm bảo thư mục tồn tại
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo các thành phần chính
        self.data_pipeline = None
        self.current_agent = None
        self.current_environment = None
        self.current_trainer = None
        
        # Trạng thái hệ thống
        self.is_collecting_data = False
        self.is_training = False
        self.is_trading = False
        
        self.logger.info("Đã khởi tạo hệ thống giao dịch tự động")
    
    async def collect_data(
        self,
        exchange_id: str,
        symbols: List[str],
        timeframe: str,
        days: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        futures: bool = False,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Path]:
        """
        Thu thập dữ liệu thị trường.
        
        Args:
            exchange_id: ID sàn giao dịch
            symbols: Danh sách cặp giao dịch
            timeframe: Khung thời gian
            days: Số ngày cần thu thập (tính từ hiện tại)
            start_date: Ngày bắt đầu (định dạng YYYY-MM-DD)
            end_date: Ngày kết thúc (định dạng YYYY-MM-DD)
            futures: Thu thập dữ liệu futures thay vì spot
            output_dir: Thư mục lưu dữ liệu
            
        Returns:
            Dict với key là symbol và value là đường dẫn file dữ liệu
        """
        if not DATA_COLLECTORS_AVAILABLE:
            self.logger.error("Không thể thu thập dữ liệu: Module thu thập dữ liệu không khả dụng")
            return {}
        
        # Thiết lập trạng thái
        self.is_collecting_data = True
        historical_collector = None  # Khởi tạo biến này ở đây để tránh lỗi
        
        try:
            # Khởi tạo data pipeline nếu chưa có
            if self.data_pipeline is None:
                self.data_pipeline = DataPipeline(
                    logger=self.logger
                )
            
            # Xác định thời gian thu thập
            from datetime import datetime, timedelta
            
            end_time = None
            start_time = None
            
            if end_date:
                end_time = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end_time = datetime.now()
            
            if start_date:
                start_time = datetime.strptime(start_date, "%Y-%m-%d")
            elif days:
                start_time = end_time - timedelta(days=days)
            else:
                # Mặc định lấy 30 ngày
                start_time = end_time - timedelta(days=30)
            
            # Thiết lập thư mục đầu ra
            if output_dir is None:
                output_dir = self.data_dir / "collected" / exchange_id
                
                # Thêm thông tin futures nếu có
                if futures:
                    output_dir = output_dir / "futures"
                else:
                    output_dir = output_dir / "spot"
                    
                # Thêm timeframe
                output_dir = output_dir / timeframe
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Thu thập dữ liệu
            self.logger.info(f"Bắt đầu thu thập dữ liệu từ {exchange_id} cho {len(symbols)} cặp tiền, khung thời gian {timeframe}")
            
            # Gọi phương thức thu thập dữ liệu từ data_pipeline
            collected_data = await self.data_pipeline.collect_data(
                exchange_id=exchange_id,
                symbols=symbols,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                is_futures=futures
            )
            
            # Lưu dữ liệu thu thập được
            saved_paths = {}
            
            if collected_data:
                # Lưu dữ liệu
                saved_results = self.data_pipeline.save_data(
                    collected_data,
                    output_dir=output_dir,
                    file_format='parquet',
                    include_metadata=True
                )
                
                # Chuyển đổi đường dẫn thành Path
                saved_paths = {symbol: Path(path) for symbol, path in saved_results.items()}
                
                self.logger.info(f"Đã thu thập và lưu dữ liệu thành công cho {len(saved_paths)} cặp tiền")
            else:
                self.logger.warning("Không có dữ liệu nào được thu thập")
            
            return saved_paths
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập dữ liệu: {str(e)}", exc_info=True)
            return {}
        finally:
            self.is_collecting_data = False
            # Đóng historical_collector nếu nó đã được khởi tạo
            try:
                if historical_collector is not None:
                    await historical_collector.exchange_connector.close()
            except Exception as e:
                self.logger.warning(f"Không thể đóng historical_collector: {str(e)}")
                pass
    
    def process_data(
        self,
        data_paths: Dict[str, Path],
        pipeline_name: Optional[str] = None,
        clean_data: bool = True,
        generate_features: bool = True,
         all_indicators: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """
        Xử lý dữ liệu thị trường.
        
        Args:
            data_paths: Dict với key là symbol và value là đường dẫn file dữ liệu
            pipeline_name: Tên pipeline xử lý dữ liệu
            clean_data: Làm sạch dữ liệu
            generate_features: Tạo đặc trưng
            all_indicators: Sử dụng tất cả các chỉ báo kỹ thuật có sẵn
            output_dir: Thư mục lưu dữ liệu đã xử lý
            
        Returns:
            Dict với key là symbol và value là đường dẫn file dữ liệu đã xử lý
        """
        try:
            # Khởi tạo data pipeline nếu chưa có
            if self.data_pipeline is None:
                self.data_pipeline = DataPipeline(
                    logger=self.logger
                )
            
            # Thiết lập thư mục đầu ra
            if output_dir is None:
                output_dir = self.data_dir / "processed"
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Tải dữ liệu
            self.logger.info(f"Bắt đầu xử lý dữ liệu cho {len(data_paths)} cặp tiền")
            
            # Tạo dict để lưu dữ liệu đã tải
            loaded_data = {}
            
            # Tải từng file dữ liệu
            for symbol, path in data_paths.items():
                if not path.exists():
                    self.logger.warning(f"Không tìm thấy file dữ liệu cho {symbol}: {path}")
                    continue
                
                # Xác định định dạng file
                file_format = None
                if path.suffix == '.csv':
                    file_format = 'csv'
                elif path.suffix == '.parquet':
                    file_format = 'parquet'
                elif path.suffix == '.json':
                    file_format = 'json'
                
                # Tải dữ liệu
                symbol_data = self.data_pipeline.load_data(
                    file_paths=path,
                    file_format=file_format
                )
                
                if symbol_data:
                    # Nếu symbol_data có nhiều symbols, chỉ lấy symbol đúng
                    if symbol in symbol_data:
                        loaded_data[symbol] = symbol_data[symbol]
                    else:
                        # Lấy symbol đầu tiên nếu không tìm thấy
                        first_symbol = next(iter(symbol_data.keys()))
                        loaded_data[symbol] = symbol_data[first_symbol]
                else:
                    self.logger.warning(f"Không thể tải dữ liệu cho {symbol}")
            
            # Kiểm tra xem có dữ liệu nào được tải không
            if not loaded_data:
                self.logger.error("Không có dữ liệu nào được tải")
                return {}
            
            # Xử lý dữ liệu
            processed_data = loaded_data
            
            # Làm sạch dữ liệu nếu cần
            if clean_data:
                processed_data = self.data_pipeline.clean_data(
                    data=processed_data,
                    clean_ohlcv=True
                )
            
            # Tạo đặc trưng nếu cần
            if generate_features:
                processed_data = self.data_pipeline.generate_features(
                    data=processed_data,
                    use_pipeline=pipeline_name,
                    all_indicators=all_indicators  # Truyền tham số này
                )
            
            # Lưu dữ liệu đã xử lý
            saved_results = self.data_pipeline.save_data(
                processed_data,
                output_dir=output_dir,
                file_format='parquet',
                include_metadata=True
            )
            
            # Chuyển đổi đường dẫn thành Path
            saved_paths = {symbol: Path(path) for symbol, path in saved_results.items()}
            
            self.logger.info(f"Đã xử lý và lưu dữ liệu thành công cho {len(saved_paths)} cặp tiền")
            
            return saved_paths
            
        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý dữ liệu: {str(e)}", exc_info=True)
            return {}
    
    async def train_agent(
        self,
        data_path: Union[str, Path],
        agent_type: str = "dqn",
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        num_episodes: int = 1000,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Tuple[bool, Path]:
        """
        Huấn luyện agent trên dữ liệu thị trường.
        
        Args:
            data_path: Đường dẫn file dữ liệu
            agent_type: Loại agent ("dqn", "ppo", "a2c")
            symbol: Cặp giao dịch
            timeframe: Khung thời gian
            num_episodes: Số episode huấn luyện
            output_dir: Thư mục lưu mô hình
            
        Returns:
            Tuple (success, model_path)
        """
        if not AGENTS_AVAILABLE or not ENVIRONMENTS_AVAILABLE:
            self.logger.error("Không thể huấn luyện: Module agent hoặc môi trường không khả dụng")
            return False, Path()
        
        # Thiết lập trạng thái
        self.is_training = True
        
        try:
            # Thiết lập thư mục đầu ra
            if output_dir is None:
                output_dir = self.model_dir / agent_type / symbol.replace('/', '_') / timeframe
            
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Tải dữ liệu
            self.logger.info(f"Bắt đầu huấn luyện agent {agent_type} cho {symbol}, khung thời gian {timeframe}")
            
            # Khởi tạo data pipeline nếu chưa có
            if self.data_pipeline is None:
                self.data_pipeline = DataPipeline(
                    logger=self.logger
                )
            
            # Tải dữ liệu
            data_path = Path(data_path)
            if not data_path.exists():
                self.logger.error(f"Không tìm thấy file dữ liệu: {data_path}")
                return False, Path()
            
            # Xác định định dạng file
            file_format = None
            if data_path.suffix == '.csv':
                file_format = 'csv'
            elif data_path.suffix == '.parquet':
                file_format = 'parquet'
            elif data_path.suffix == '.json':
                file_format = 'json'
            
            # Tải dữ liệu
            loaded_data = self.data_pipeline.load_data(
                file_paths=data_path,
                file_format=file_format
            )
            
            if not loaded_data:
                self.logger.error("Không thể tải dữ liệu để huấn luyện")
                return False, Path()
            
            # Lấy DataFrame cho symbol
            if symbol in loaded_data:
                df = loaded_data[symbol]
            else:
                # Lấy DataFrame đầu tiên nếu không tìm thấy symbol
                first_symbol = next(iter(loaded_data.keys()))
                df = loaded_data[first_symbol]
                self.logger.warning(f"Không tìm thấy dữ liệu cho {symbol}, sử dụng {first_symbol} thay thế")
            
            # Tạo môi trường
            env_kwargs = kwargs.get("env_kwargs", {})
            env = TradingEnv(
                data=df,
                symbol=symbol,
                timeframe=timeframe,
                **env_kwargs
            )
            
            # Lưu trữ môi trường hiện tại
            self.current_environment = env
            
            # Tạo agent dựa trên loại
            agent_kwargs = kwargs.get("agent_kwargs", {})
            
            # Chuẩn bị các tham số cần thiết cho agent
            if "state_dim" not in agent_kwargs and hasattr(env, "observation_space"):
                if hasattr(env.observation_space, "shape"):
                    agent_kwargs["state_dim"] = env.observation_space.shape
            
            if "action_dim" not in agent_kwargs and hasattr(env, "action_space"):
                if hasattr(env.action_space, "n"):
                    agent_kwargs["action_dim"] = env.action_space.n
                elif hasattr(env.action_space, "shape"):
                    agent_kwargs["action_dim"] = env.action_space.shape[0]
            
            # Tạo agent
            if agent_type.lower() == "dqn":
                from models.agents.dqn_agent import DQNAgent
                agent = DQNAgent(**agent_kwargs)
            elif agent_type.lower() == "ppo":
                from models.agents.ppo_agent import PPOAgent
                agent = PPOAgent(**agent_kwargs)
            elif agent_type.lower() == "a2c":
                from models.agents.a2c_agent import A2CAgent
                agent = A2CAgent(**agent_kwargs)
            else:
                self.logger.error(f"Loại agent không được hỗ trợ: {agent_type}")
                return False, Path()
            
            # Lưu trữ agent hiện tại
            self.current_agent = agent
            
            # Tạo trainer
            trainer_kwargs = kwargs.get("trainer_kwargs", {})
            trainer = Trainer(
                agent=agent,
                env=env,
                output_dir=output_dir,
                **trainer_kwargs
            )
            
            # Lưu trữ trainer hiện tại
            self.current_trainer = trainer
            
            # Huấn luyện agent
            self.logger.info(f"Bắt đầu huấn luyện agent {agent_type} với {num_episodes} episodes")
            
            # Thiết lập số episodes
            trainer.config["num_episodes"] = num_episodes
            
            # Huấn luyện
            training_history = trainer.train()
            
            # Đường dẫn đến mô hình tốt nhất
            model_path = output_dir / "models" / "best_model"
            
            if model_path.exists():
                self.logger.info(f"Đã hoàn thành huấn luyện. Mô hình tốt nhất được lưu tại {model_path}")
                return True, model_path
            else:
                self.logger.warning("Huấn luyện hoàn tất nhưng không tìm thấy file mô hình")
                # Tìm kiếm file mô hình khác
                model_files = list(output_dir.glob("**/*model*"))
                if model_files:
                    return True, model_files[0]
                return False, Path()
            
        except Exception as e:
            self.logger.error(f"Lỗi khi huấn luyện agent: {str(e)}", exc_info=True)
            return False, Path()
        finally:
            self.is_training = False
    
    def evaluate_agent(
        self,
        model_path: Union[str, Path],
        data_path: Union[str, Path],
        agent_type: str = "dqn",
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        num_episodes: int = 10,
        **kwargs
    ) -> Dict[str, float]:
        """
        Đánh giá hiệu suất của agent.
        
        Args:
            model_path: Đường dẫn file mô hình
            data_path: Đường dẫn file dữ liệu
            agent_type: Loại agent ("dqn", "ppo", "a2c")
            symbol: Cặp giao dịch
            timeframe: Khung thời gian
            num_episodes: Số episode đánh giá
            
        Returns:
            Dict chứa kết quả đánh giá
        """
        if not AGENTS_AVAILABLE or not ENVIRONMENTS_AVAILABLE:
            self.logger.error("Không thể đánh giá: Module agent hoặc môi trường không khả dụng")
            return {}
        
        try:
            # Tải dữ liệu
            self.logger.info(f"Bắt đầu đánh giá agent {agent_type} cho {symbol}, khung thời gian {timeframe}")
            
            # Khởi tạo data pipeline nếu chưa có
            if self.data_pipeline is None:
                self.data_pipeline = DataPipeline(
                    logger=self.logger
                )
            
            # Tải dữ liệu
            data_path = Path(data_path)
            if not data_path.exists():
                self.logger.error(f"Không tìm thấy file dữ liệu: {data_path}")
                return {}
            
            # Xác định định dạng file
            file_format = None
            if data_path.suffix == '.csv':
                file_format = 'csv'
            elif data_path.suffix == '.parquet':
                file_format = 'parquet'
            elif data_path.suffix == '.json':
                file_format = 'json'
            
            # Tải dữ liệu
            loaded_data = self.data_pipeline.load_data(
                file_paths=data_path,
                file_format=file_format
            )
            
            if not loaded_data:
                self.logger.error("Không thể tải dữ liệu để đánh giá")
                return {}
            
            # Lấy DataFrame cho symbol
            if symbol in loaded_data:
                df = loaded_data[symbol]
            else:
                # Lấy DataFrame đầu tiên nếu không tìm thấy symbol
                first_symbol = next(iter(loaded_data.keys()))
                df = loaded_data[first_symbol]
                self.logger.warning(f"Không tìm thấy dữ liệu cho {symbol}, sử dụng {first_symbol} thay thế")
            
            # Tạo môi trường
            env_kwargs = kwargs.get("env_kwargs", {})
            env = TradingEnv(
                data=df,
                symbol=symbol,
                timeframe=timeframe,
                **env_kwargs
            )
            
            # Tạo agent dựa trên loại
            agent_kwargs = kwargs.get("agent_kwargs", {})
            
            # Chuẩn bị các tham số cần thiết cho agent
            if "state_dim" not in agent_kwargs and hasattr(env, "observation_space"):
                if hasattr(env.observation_space, "shape"):
                    agent_kwargs["state_dim"] = env.observation_space.shape
            
            if "action_dim" not in agent_kwargs and hasattr(env, "action_space"):
                if hasattr(env.action_space, "n"):
                    agent_kwargs["action_dim"] = env.action_space.n
                elif hasattr(env.action_space, "shape"):
                    agent_kwargs["action_dim"] = env.action_space.shape[0]
            
            # Tạo agent
            if agent_type.lower() == "dqn":
                from models.agents.dqn_agent import DQNAgent
                agent = DQNAgent(**agent_kwargs)
            elif agent_type.lower() == "ppo":
                from models.agents.ppo_agent import PPOAgent
                agent = PPOAgent(**agent_kwargs)
            elif agent_type.lower() == "a2c":
                from models.agents.a2c_agent import A2CAgent
                agent = A2CAgent(**agent_kwargs)
            else:
                self.logger.error(f"Loại agent không được hỗ trợ: {agent_type}")
                return {}
            
            # Tải mô hình
            model_path = Path(model_path)
            if not model_path.exists():
                self.logger.error(f"Không tìm thấy file mô hình: {model_path}")
                return {}
            
            # Tải mô hình vào agent
            if not agent.load_model(model_path):
                self.logger.error(f"Không thể tải mô hình từ {model_path}")
                return {}
            
            # Đánh giá agent
            self.logger.info(f"Đánh giá agent trên {num_episodes} episodes")
            
            eval_rewards = agent.evaluate(env, num_episodes=num_episodes)
            
            # Tính toán các chỉ số đánh giá
            mean_reward = float(np.mean(eval_rewards))
            std_reward = float(np.std(eval_rewards))
            min_reward = float(np.min(eval_rewards))
            max_reward = float(np.max(eval_rewards))
            
            # Tạo dict kết quả
            results = {
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "min_reward": min_reward,
                "max_reward": max_reward,
                "num_episodes": num_episodes
            }
            
            self.logger.info(
                f"Kết quả đánh giá: Mean reward: {mean_reward:.2f} ± {std_reward:.2f}, "
                f"Min: {min_reward:.2f}, Max: {max_reward:.2f}"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Lỗi khi đánh giá agent: {str(e)}", exc_info=True)
            return {}
    
    def start_dashboard(self, port: int = 8501):
        """
        Khởi động dashboard Streamlit.
        
        Args:
            port: Cổng cho dashboard
        """
        try:
            import streamlit as st
        except ImportError:
            self.logger.error("Không thể khởi động dashboard: Streamlit chưa được cài đặt")
            print("Hãy cài đặt Streamlit với lệnh: pip install streamlit")
            return
        
        try:
            import subprocess
            from pathlib import Path
            
            # Đường dẫn tới app.py của Streamlit
            dashboard_path = Path(__file__).parent / "streamlit_dashboard" / "app.py"
            
            if not dashboard_path.exists():
                self.logger.error(f"Không tìm thấy file dashboard: {dashboard_path}")
                return
            
            # Chạy Streamlit
            self.logger.info(f"Khởi động dashboard tại http://localhost:{port}")
            subprocess.Popen(["streamlit", "run", str(dashboard_path), "--server.port", str(port)])
            
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi động dashboard: {str(e)}", exc_info=True)