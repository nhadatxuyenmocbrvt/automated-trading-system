"""
Hệ thống giao dịch tự động.
File này định nghĩa lớp AutomatedTradingSystem, trung tâm điều phối
các module khác nhau của hệ thống giao dịch.
"""

import os
import sys
import logging
import asyncio
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Import các module từ hệ thống
from config.system_config import get_system_config
from config.logging_config import get_logger
from config.security_config import get_security_config
from config.env import get_env

# Import các module xử lý dữ liệu
from data_processors.data_pipeline import DataPipeline

# Import module collectors
try:
    from data_collectors.market_data.historical_data_collector import create_data_collector
    DATA_COLLECTORS_AVAILABLE = True
except ImportError:
    DATA_COLLECTORS_AVAILABLE = False

# Import module environments
try:
    from environments.trading_gym.trading_env import TradingEnv
    ENVIRONMENTS_AVAILABLE = True
except ImportError:
    ENVIRONMENTS_AVAILABLE = False

# Import module agents
try:
    from models.agents.dqn_agent import DQNAgent
    from models.agents.ppo_agent import PPOAgent
    from models.agents.a2c_agent import A2CAgent
    from models.training_pipeline.trainer import Trainer
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

# Import module deployment
try:
    from deployment.trade_executor import TradeExecutor
    DEPLOYMENT_AVAILABLE = True
except ImportError:
    DEPLOYMENT_AVAILABLE = False

class AutomatedTradingSystem:
    """
    Lớp trung tâm quản lý toàn bộ hệ thống giao dịch tự động.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        mode: str = "development",
        verbose: bool = False
    ):
        """
        Khởi tạo hệ thống giao dịch tự động.
        
        Args:
            config_path: Đường dẫn file cấu hình
            data_dir: Thư mục dữ liệu
            model_dir: Thư mục lưu trữ mô hình
            log_dir: Thư mục lưu trữ logs
            mode: Chế độ hoạt động ("development", "testing", "production")
            verbose: Hiển thị thông tin chi tiết
        """
        # Thiết lập logger
        self.logger = get_logger("trading_system")
        
        # Thiết lập cấu hình hệ thống
        self.system_config = get_system_config()
        if config_path:
            try:
                self.system_config.load_from_file(config_path)
                self.logger.info(f"Đã tải cấu hình từ {config_path}")
            except Exception as e:
                self.logger.error(f"Lỗi khi tải cấu hình: {str(e)}")
        
        # Thiết lập chế độ hoạt động
        self.mode = mode
        self.verbose = verbose
        
        # Thiết lập đường dẫn thư mục
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(self.system_config.get("data_dir", "data"))
        
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = Path(self.system_config.get("model_dir", "saved_models"))
        
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path(self.system_config.get("log_dir", "logs"))
        
        # Đảm bảo các thư mục tồn tại
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo các thành phần
        self.data_pipeline = None
        self.environment = None
        self.agent = None
        self.trade_executor = None
        
        self.logger.info(f"Đã khởi tạo AutomatedTradingSystem ở chế độ {mode}")
    
    async def setup(self) -> None:
        """
        Thiết lập các thành phần của hệ thống.
        """
        # Khởi tạo data pipeline
        self.data_pipeline = DataPipeline(
            data_dir=self.data_dir,
            output_dir=self.data_dir / "processed",
            logger=self.logger
        )
        
        self.logger.info("Đã thiết lập các thành phần cơ bản của hệ thống")
    
    async def collect_data(
        self,
        exchange_id: str,
        symbols: List[str],
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        days: Optional[int] = None,
        is_futures: bool = False,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Thu thập dữ liệu từ sàn giao dịch.
        
        Args:
            exchange_id: ID sàn giao dịch
            symbols: Danh sách cặp giao dịch
            timeframe: Khung thời gian
            start_time: Thời gian bắt đầu
            end_time: Thời gian kết thúc
            days: Số ngày cần lấy (nếu không có start_time)
            is_futures: Là thị trường futures hay không
            api_key: API key (nếu cần)
            api_secret: API secret (nếu cần)
            
        Returns:
            Dict với key là symbol và value là DataFrame
        """
        if not DATA_COLLECTORS_AVAILABLE:
            self.logger.error("Không thể thu thập dữ liệu: Module data_collectors không khả dụng")
            return {}
        
        if self.data_pipeline is None:
            await self.setup()
        
        # Thiết lập thời gian
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None and days is not None:
            start_time = end_time - timedelta(days=days)
        
        self.logger.info(f"Thu thập dữ liệu từ {exchange_id} cho {len(symbols)} cặp giao dịch, khung thời gian {timeframe}")
        
        try:
            # Thu thập dữ liệu
            data = await self.data_pipeline.collect_data(
                exchange_id=exchange_id,
                symbols=symbols,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                is_futures=is_futures,
                api_key=api_key,
                api_secret=api_secret
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập dữ liệu: {str(e)}")
            return {}
    
    async def process_data(
        self,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        input_files: Optional[List[str]] = None,
        pipeline_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        clean_data: bool = True,
        generate_features: bool = True,
        save_results: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Xử lý dữ liệu thị trường.
        
        Args:
            data: Dict dữ liệu đầu vào
            input_files: Danh sách file đầu vào
            pipeline_name: Tên pipeline xử lý
            output_dir: Thư mục đầu ra
            clean_data: Làm sạch dữ liệu hay không
            generate_features: Tạo đặc trưng hay không
            save_results: Lưu kết quả hay không
            
        Returns:
            Dict với key là symbol và value là DataFrame đã xử lý
        """
        if self.data_pipeline is None:
            await self.setup()
        
        # Thiết lập pipeline
        steps = [
            {"name": "load_data", "enabled": input_files is not None},
            {"name": "clean_data", "enabled": clean_data},
            {"name": "generate_features", "enabled": generate_features},
            {"name": "save_data", "enabled": save_results}
        ]
        
        if pipeline_name is None:
            pipeline_name = f"custom_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.data_pipeline.register_pipeline(
                name=pipeline_name,
                steps=steps,
                description="Pipeline tùy chỉnh được tạo từ AutomatedTradingSystem"
            )
        
        if output_dir is not None:
            output_path = Path(output_dir)
        else:
            output_path = self.data_dir / "processed"
        
        self.logger.info(f"Xử lý dữ liệu với pipeline '{pipeline_name}'")
        
        try:
            # Xử lý dữ liệu
            processed_data = await self.data_pipeline.run_pipeline(
                pipeline_name=pipeline_name,
                input_data=data,
                input_files=input_files,
                output_dir=output_path,
                save_results=save_results
            )
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý dữ liệu: {str(e)}")
            return {}
    
    def setup_environment(
        self,
        data: Dict[str, pd.DataFrame],
        symbol: str,
        initial_balance: float = 10000.0,
        max_positions: int = 5,
        window_size: int = 100,
        reward_function: str = "profit",
        include_positions: bool = True,
        include_balance: bool = True
    ) -> bool:
        """
        Thiết lập môi trường huấn luyện.
        
        Args:
            data: Dict dữ liệu đã xử lý
            symbol: Cặp giao dịch cần sử dụng
            initial_balance: Số dư ban đầu
            max_positions: Số vị thế tối đa
            window_size: Kích thước cửa sổ dữ liệu
            reward_function: Hàm phần thưởng
            include_positions: Bao gồm thông tin vị thế trong không gian quan sát
            include_balance: Bao gồm thông tin số dư trong không gian quan sát
            
        Returns:
            True nếu thiết lập thành công, False nếu không
        """
        if not ENVIRONMENTS_AVAILABLE:
            self.logger.error("Không thể thiết lập môi trường: Module environments không khả dụng")
            return False
        
        if symbol not in data:
            self.logger.error(f"Không tìm thấy dữ liệu cho cặp giao dịch {symbol}")
            return False
        
        try:
            # Thiết lập môi trường
            self.environment = TradingEnv(
                data=data[symbol],
                symbol=symbol,
                initial_balance=initial_balance,
                max_positions=max_positions,
                window_size=window_size,
                reward_function=reward_function,
                include_positions=include_positions,
                include_balance=include_balance,
                logger=self.logger
            )
            
            self.logger.info(f"Đã thiết lập môi trường huấn luyện cho {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thiết lập môi trường huấn luyện: {str(e)}")
            return False
    
    def setup_agent(
        self,
        agent_type: str = "dqn",
        load_model: bool = False,
        model_path: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Thiết lập agent.
        
        Args:
            agent_type: Loại agent ('dqn', 'ppo', 'a2c')
            load_model: Tải mô hình có sẵn hay không
            model_path: Đường dẫn file mô hình
            **kwargs: Các tham số khác cho agent
            
        Returns:
            True nếu thiết lập thành công, False nếu không
        """
        if not AGENTS_AVAILABLE:
            self.logger.error("Không thể thiết lập agent: Module models không khả dụng")
            return False
        
        if self.environment is None:
            self.logger.error("Cần thiết lập môi trường trước khi thiết lập agent")
            return False
        
        try:
            # Thiết lập agent
            if agent_type.lower() == "dqn":
                self.agent = DQNAgent(
                    env=self.environment,
                    model_dir=self.model_dir,
                    **kwargs
                )
            elif agent_type.lower() == "ppo":
                self.agent = PPOAgent(
                    env=self.environment,
                    model_dir=self.model_dir,
                    **kwargs
                )
            elif agent_type.lower() == "a2c":
                self.agent = A2CAgent(
                    env=self.environment,
                    model_dir=self.model_dir,
                    **kwargs
                )
            else:
                self.logger.error(f"Loại agent không hợp lệ: {agent_type}")
                return False
            
            # Tải mô hình nếu cần
            if load_model and model_path:
                self.agent.load(model_path)
                self.logger.info(f"Đã tải mô hình từ {model_path}")
            
            self.logger.info(f"Đã thiết lập agent {agent_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thiết lập agent: {str(e)}")
            return False
    
    def train_agent(
        self,
        episodes: int = 1000,
        eval_interval: int = 100,
        save_interval: int = 100,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Huấn luyện agent.
        
        Args:
            episodes: Số episode huấn luyện
            eval_interval: Khoảng thời gian đánh giá
            save_interval: Khoảng thời gian lưu mô hình
            save_path: Đường dẫn lưu mô hình
            **kwargs: Các tham số khác cho huấn luyện
            
        Returns:
            Dict kết quả huấn luyện
        """
        if not AGENTS_AVAILABLE:
            self.logger.error("Không thể huấn luyện agent: Module models không khả dụng")
            return {}
        
        if self.agent is None or self.environment is None:
            self.logger.error("Cần thiết lập agent và môi trường trước khi huấn luyện")
            return {}
        
        try:
            # Khởi tạo trainer
            trainer = Trainer(
                agent=self.agent,
                env=self.environment,
                model_dir=self.model_dir,
                logger=self.logger
            )
            
            # Huấn luyện agent
            self.logger.info(f"Bắt đầu huấn luyện với {episodes} episodes")
            results = trainer.train(
                episodes=episodes,
                eval_interval=eval_interval,
                save_interval=save_interval,
                save_path=save_path,
                **kwargs
            )
            
            self.logger.info(f"Đã hoàn thành huấn luyện agent")
            return results
            
        except Exception as e:
            self.logger.error(f"Lỗi khi huấn luyện agent: {str(e)}")
            return {}
    
    def backtest(
        self,
        data: Optional[Dict[str, pd.DataFrame]] = None,
        symbol: Optional[str] = None,
        initial_balance: float = 10000.0,
        render_mode: str = "console",
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Chạy backtest.
        
        Args:
            data: Dict dữ liệu đã xử lý
            symbol: Cặp giao dịch cần sử dụng
            initial_balance: Số dư ban đầu
            render_mode: Chế độ hiển thị ('console', 'human', 'rgb_array')
            output_path: Đường dẫn lưu kết quả
            
        Returns:
            Dict kết quả backtest
        """
        if not ENVIRONMENTS_AVAILABLE or not AGENTS_AVAILABLE:
            self.logger.error("Không thể chạy backtest: Module environments hoặc models không khả dụng")
            return {}
        
        if self.agent is None:
            self.logger.error("Cần thiết lập agent trước khi chạy backtest")
            return {}
        
        if data is not None and symbol is not None:
            # Thiết lập môi trường mới với dữ liệu mới
            env_setup = self.setup_environment(
                data=data,
                symbol=symbol,
                initial_balance=initial_balance
            )
            
            if not env_setup:
                return {}
        
        if self.environment is None:
            self.logger.error("Cần thiết lập môi trường trước khi chạy backtest")
            return {}
        
        try:
            # Chạy backtest
            self.logger.info(f"Bắt đầu chạy backtest")
            
            # Đặt lại môi trường
            state = self.environment.reset()
            done = False
            total_reward = 0
            
            # Lưu lịch sử
            history = {
                "actions": [],
                "rewards": [],
                "balances": [self.environment.current_balance],
                "navs": [self.environment.current_nav],
                "timestamps": [self.environment.get_current_timestamp()]
            }
            
            # Chạy qua từng bước
            while not done:
                # Lấy hành động từ agent
                action = self.agent.act(state, deterministic=True)
                
                # Thực hiện hành động
                next_state, reward, done, info = self.environment.step(action)
                
                # Render nếu cần
                if render_mode != "none":
                    self.environment.render(mode=render_mode)
                
                # Cập nhật trạng thái
                state = next_state
                total_reward += reward
                
                # Lưu lịch sử
                history["actions"].append(action)
                history["rewards"].append(reward)
                history["balances"].append(info["balance"])
                history["navs"].append(info["nav"])
                history["timestamps"].append(info["timestamp"])
            
            # Đóng môi trường
            self.environment.close()
            
            # Tính toán kết quả
            results = {
                "total_reward": total_reward,
                "final_balance": history["balances"][-1],
                "final_nav": history["navs"][-1],
                "return_pct": (history["navs"][-1] / initial_balance - 1) * 100,
                "win_rate": self.environment.performance_metrics["win_count"] / max(1, self.environment.performance_metrics["trade_count"]),
                "max_drawdown": self.environment.performance_metrics["max_drawdown"],
                "trade_count": self.environment.performance_metrics["trade_count"],
                "history": history
            }
            
            # Lưu kết quả nếu cần
            if output_path:
                self.environment.save_history(output_path)
                self.logger.info(f"Đã lưu lịch sử backtest vào {output_path}")
            
            self.logger.info(f"Đã hoàn thành backtest với tỷ suất lợi nhuận {results['return_pct']:.2f}%")
            return results
            
        except Exception as e:
            self.logger.error(f"Lỗi khi chạy backtest: {str(e)}")
            return {}
    
    def setup_trade_executor(
        self,
        exchange_id: str,
        symbol: str,
        api_key: str,
        api_secret: str,
        is_futures: bool = False,
        test_mode: bool = True,
        **kwargs
    ) -> bool:
        """
        Thiết lập trình thực thi giao dịch.
        
        Args:
            exchange_id: ID sàn giao dịch
            symbol: Cặp giao dịch
            api_key: API key
            api_secret: API secret
            is_futures: Là thị trường futures hay không
            test_mode: Chạy ở chế độ test hay không
            **kwargs: Các tham số khác cho trình thực thi
            
        Returns:
            True nếu thiết lập thành công, False nếu không
        """
        if not DEPLOYMENT_AVAILABLE:
            self.logger.error("Không thể thiết lập trình thực thi: Module deployment không khả dụng")
            return False
        
        if self.agent is None:
            self.logger.error("Cần thiết lập agent trước khi thiết lập trình thực thi")
            return False
        
        try:
            # Thiết lập trình thực thi
            self.trade_executor = TradeExecutor(
                agent=self.agent,
                exchange_id=exchange_id,
                symbol=symbol,
                api_key=api_key,
                api_secret=api_secret,
                is_futures=is_futures,
                test_mode=test_mode,
                logger=self.logger,
                **kwargs
            )
            
            self.logger.info(f"Đã thiết lập trình thực thi giao dịch cho {symbol} trên {exchange_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thiết lập trình thực thi: {str(e)}")
            return False
    
    async def start_trading(
        self,
        max_trades: Optional[int] = None,
        timeout: Optional[int] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Bắt đầu giao dịch thực tế.
        
        Args:
            max_trades: Số giao dịch tối đa
            timeout: Thời gian tối đa (giây)
            stop_loss_pct: Phần trăm dừng lỗ
            take_profit_pct: Phần trăm chốt lời
            
        Returns:
            Dict kết quả giao dịch
        """
        if not DEPLOYMENT_AVAILABLE:
            self.logger.error("Không thể bắt đầu giao dịch: Module deployment không khả dụng")
            return {}
        
        if self.trade_executor is None:
            self.logger.error("Cần thiết lập trình thực thi trước khi bắt đầu giao dịch")
            return {}
        
        try:
            # Bắt đầu giao dịch
            self.logger.info(f"Bắt đầu giao dịch thực tế")
            
            if self.mode == "production":
                self.logger.warning("Giao dịch đang chạy ở chế độ PRODUCTION")
            else:
                self.logger.info(f"Giao dịch đang chạy ở chế độ {self.mode}")
            
            results = await self.trade_executor.run(
                max_trades=max_trades,
                timeout=timeout,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct
            )
            
            self.logger.info(f"Đã hoàn thành phiên giao dịch")
            return results
            
        except Exception as e:
            self.logger.error(f"Lỗi khi bắt đầu giao dịch: {str(e)}")
            return {}
    
    def save_system_state(self, file_path: Optional[str] = None) -> bool:
        """
        Lưu trạng thái của hệ thống.
        
        Args:
            file_path: Đường dẫn file trạng thái
            
        Returns:
            True nếu lưu thành công, False nếu không
        """
        if file_path is None:
            file_path = self.data_dir / "system_state.json"
        
        try:
            # Lưu cấu hình hệ thống
            self.system_config.save_to_file(file_path)
            
            # Lưu trạng thái data pipeline nếu có
            if self.data_pipeline is not None:
                self.data_pipeline.save_pipeline_state()
            
            # Lưu mô hình agent nếu có
            if self.agent is not None:
                agent_path = self.model_dir / "latest_agent.h5"
                self.agent.save(str(agent_path))
            
            self.logger.info(f"Đã lưu trạng thái hệ thống vào {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu trạng thái hệ thống: {str(e)}")
            return False
    
    def load_system_state(self, file_path: Optional[str] = None) -> bool:
        """
        Tải trạng thái của hệ thống.
        
        Args:
            file_path: Đường dẫn file trạng thái
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        if file_path is None:
            file_path = self.data_dir / "system_state.json"
        
        if not Path(file_path).exists():
            self.logger.warning(f"File trạng thái không tồn tại: {file_path}")
            return False
        
        try:
            # Tải cấu hình hệ thống
            self.system_config.load_from_file(file_path)
            
            # Tải trạng thái data pipeline nếu có
            if self.data_pipeline is not None:
                self.data_pipeline.load_pipeline_state()
            
            self.logger.info(f"Đã tải trạng thái hệ thống từ {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải trạng thái hệ thống: {str(e)}")
            return False