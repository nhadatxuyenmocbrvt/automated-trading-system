"""
Pipeline tái huấn luyện agent.
File này định nghĩa lớp RetrainingPipeline để quản lý quá trình tái huấn luyện
các agent theo lịch trình hoặc dựa vào kết quả đánh giá hiệu suất, tự động cải thiện
các mô hình khi phát hiện hiệu suất giảm sút hoặc thay đổi thị trường.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import concurrent.futures
import threading
import schedule
import copy
import numpy as np
import pandas as pd
import shutil

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config, MODEL_DIR
from config.constants import RetrainingTrigger, BacktestMetric

# Import các module liên quan đến retraining
from retraining.performance_tracker import PerformanceTracker, create_performance_tracker
from retraining.experience_manager import ExperienceManager
from retraining.model_updater import ModelUpdater
from retraining.comparison_evaluator import ComparisonEvaluator
from environments.base_environment import BaseEnvironment

# Import các module huấn luyện
from models.training_pipeline.trainer import Trainer
from models.agents.base_agent import BaseAgent
from models.agents.dqn_agent import DQNAgent
from models.agents.ppo_agent import PPOAgent
from models.agents.a2c_agent import A2CAgent

# Import các module đánh giá
from backtesting.backtester import Backtester
from backtesting.performance_metrics import PerformanceMetrics

# Import các module thu thập và xử lý dữ liệu
from data_processors.data_pipeline import DataPipeline
from data_collectors.market_data.historical_data_collector import create_data_collector

# Import các module tự thích nghi
from agent_manager.self_improvement.adaptation_module import AdaptationModule

class RetrainingPipeline:
    """
    Pipeline quản lý và thực hiện quá trình tái huấn luyện agent.
    
    Lớp này cung cấp các tính năng để:
    1. Theo dõi hiệu suất agent qua thời gian
    2. Quyết định khi nào cần tái huấn luyện dựa trên hiệu suất hoặc lịch trình
    3. Thu thập dữ liệu mới và chuẩn bị cho tái huấn luyện
    4. Thực hiện quy trình tái huấn luyện hoàn chỉnh
    5. Đánh giá và triển khai mô hình mới
    6. Ghi chép và báo cáo quá trình tái huấn luyện
    """
    
    def __init__(
        self,
        agent: Optional[BaseAgent] = None,
        agent_config: Optional[Dict[str, Any]] = None,
        agent_type: str = "dqn",
        model_version: str = "v1.0.0",
        environment_name: str = "trading_env",
        strategy_name: str = "default_strategy",
        symbols: List[str] = ["BTC/USDT"],
        performance_tracker: Optional[PerformanceTracker] = None,
        data_pipeline: Optional[DataPipeline] = None,
        trainer: Optional[Trainer] = None,
        adaptation_module: Optional[AdaptationModule] = None,
        retraining_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo pipeline tái huấn luyện.
        
        Args:
            agent: Agent cần được tái huấn luyện
            agent_config: Cấu hình agent (nếu không cung cấp agent)
            agent_type: Loại agent ("dqn", "ppo", "a2c", v.v.)
            model_version: Phiên bản hiện tại của mô hình
            environment_name: Tên môi trường huấn luyện
            strategy_name: Tên chiến lược giao dịch
            symbols: Danh sách các cặp tiền được giao dịch
            performance_tracker: Công cụ theo dõi hiệu suất
            data_pipeline: Pipeline xử lý dữ liệu
            trainer: Công cụ huấn luyện
            adaptation_module: Module tự thích nghi
            retraining_dir: Thư mục chứa dữ liệu tái huấn luyện
            config: Cấu hình bổ sung
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("retraining_pipeline")
        
        # Thiết lập cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Lưu thông tin cơ bản
        self.agent = agent
        self.agent_config = agent_config or {}
        self.agent_type = agent_type
        self.model_version = model_version
        self.environment_name = environment_name
        self.strategy_name = strategy_name
        self.symbols = symbols
        
        # Thiết lập thư mục tái huấn luyện
        if retraining_dir is None:
            self.retraining_dir = Path(self.system_config.get("retraining_dir", "./retraining"))
        else:
            self.retraining_dir = Path(retraining_dir)
        
        self.retraining_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập cấu hình mặc định
        self.default_config = {
            "retraining": {
                "schedule": {
                    "enabled": True,                # Bật tái huấn luyện theo lịch
                    "frequency": "weekly",          # Tần suất: "daily", "weekly", "monthly"
                    "day_of_week": 1,               # Thứ 2 (0=CN, 1=T2, ...)
                    "time": "02:00",                # Thời gian trong ngày (giờ:phút)
                    "timezone": "UTC",              # Múi giờ
                },
                "performance_based": {
                    "enabled": True,                # Bật tái huấn luyện dựa trên hiệu suất
                    "check_frequency": 24,          # Kiểm tra hiệu suất mỗi 24 giờ
                    "min_trades_required": 30,      # Số giao dịch tối thiểu để đánh giá
                    "metrics_to_track": [
                        BacktestMetric.SHARPE_RATIO.value,
                        BacktestMetric.WIN_RATE.value,
                        BacktestMetric.PROFIT_FACTOR.value,
                        "roi"
                    ],
                    "prioritize_recent_data": True, # Ưu tiên dữ liệu gần đây
                },
                "data_collection": {
                    "lookback_days": 60,            # Số ngày dữ liệu lịch sử
                    "min_data_points": 1000,        # Số điểm dữ liệu tối thiểu
                    "validate_data": True,          # Xác thực dữ liệu trước khi sử dụng
                    "include_recent": True,         # Bao gồm dữ liệu mới nhất
                    "timeframes": ["1h", "4h", "1d"], # Khung thời gian thu thập
                },
                "training": {
                    "epochs": 100,                  # Số epoch huấn luyện
                    "batch_size": 64,               # Kích thước batch
                    "learning_rate": 0.001,         # Tốc độ học
                    "optimizer": "adam",            # Trình tối ưu hóa
                    "loss": "mse",                  # Hàm mất mát
                    "validation_split": 0.2,        # Tỷ lệ dữ liệu xác thực
                    "early_stopping": True,         # Dừng sớm
                    "patience": 10,                 # Số epochs chờ trước khi dừng sớm
                    "save_checkpoints": True,       # Lưu các checkpoint
                    "checkpoint_frequency": 10,     # Tần suất lưu checkpoint
                    "use_adaptive_learning": True,  # Sử dụng tốc độ học thích ứng
                },
                "evaluation": {
                    "backtest_days": 30,            # Số ngày dữ liệu backtest
                    "compare_with_current": True,   # So sánh với mô hình hiện tại
                    "metrics": [                    # Các metrics đánh giá
                        BacktestMetric.SHARPE_RATIO.value,
                        BacktestMetric.SORTINO_RATIO.value,
                        BacktestMetric.MAX_DRAWDOWN.value,
                        BacktestMetric.WIN_RATE.value,
                        "roi",
                        "volatility"
                    ],
                    "improvement_threshold": 0.05,  # Ngưỡng cải thiện để chấp nhận mô hình mới (5%)
                    "evaluation_period": "recent",  # Giai đoạn đánh giá: "recent", "random", "stress"
                },
                "deployment": {
                    "auto_deploy": True,            # Tự động triển khai nếu tốt hơn
                    "keep_old_versions": 3,         # Số phiên bản cũ giữ lại
                    "rollback_enabled": True,       # Cho phép quay lại phiên bản cũ
                    "gradual_rollout": True,        # Triển khai dần dần
                    "validation_period": 24,        # Thời gian xác thực trước khi triển khai đầy đủ (giờ)
                },
                "notification": {
                    "enabled": True,                # Bật thông báo
                    "email": True,                  # Gửi email
                    "telegram": False,              # Gửi thông báo Telegram
                    "success_notification": True,   # Thông báo khi thành công
                    "failure_notification": True,   # Thông báo khi thất bại
                },
                "adaptation": {
                    "enabled": True,                # Bật tính năng tự thích nghi
                    "market_change_sensitivity": 0.3, # Độ nhạy với thay đổi thị trường
                    "gradual_adaptation": True,     # Thích nghi dần dần
                    "preserve_experience": True,    # Giữ lại kinh nghiệm
                }
            }
        }
        
        # Cập nhật cấu hình với cấu hình được cung cấp
        self.config = self.default_config.copy()
        if config:
            self._update_nested_dict(self.config, config)
        
        # Khởi tạo PerformanceTracker nếu chưa được cung cấp
        self.performance_tracker = performance_tracker
        if self.performance_tracker is None:
            try:
                self.performance_tracker = create_performance_tracker(
                    agent_name=self.agent_type,
                    model_version=self.model_version,
                    strategy_name=self.strategy_name,
                    symbols=self.symbols,
                    tracking_metrics=self.config["retraining"]["performance_based"]["metrics_to_track"],
                    retraining_dir=self.retraining_dir / "performance_data",
                    logger=self.logger
                )
                self.logger.info(f"Đã khởi tạo PerformanceTracker cho {self.agent_type}")
            except Exception as e:
                self.logger.warning(f"Không thể khởi tạo PerformanceTracker: {e}")
        
        # Khởi tạo DataPipeline nếu chưa được cung cấp
        self.data_pipeline = data_pipeline
        if self.data_pipeline is None:
            try:
                self.data_pipeline = DataPipeline(
                    data_dir=self.retraining_dir / "data",
                    output_dir=self.retraining_dir / "processed_data",
                    logger=self.logger
                )
                self.logger.info("Đã khởi tạo DataPipeline")
            except Exception as e:
                self.logger.warning(f"Không thể khởi tạo DataPipeline: {e}")
        
        # Khởi tạo Module tự thích nghi nếu chưa được cung cấp và tính năng được bật
        self.adaptation_module = adaptation_module
        if self.adaptation_module is None and self.config["retraining"]["adaptation"]["enabled"]:
            try:
                if self.agent:
                    self.adaptation_module = AdaptationModule(
                        agent=self.agent,
                        output_dir=self.retraining_dir / "adaptation",
                        logger=self.logger
                    )
                    self.logger.info("Đã khởi tạo AdaptationModule")
            except Exception as e:
                self.logger.warning(f"Không thể khởi tạo AdaptationModule: {e}")
        
        # Khởi tạo Trainer nếu chưa được cung cấp
        self.trainer = trainer
        
        # Khởi tạo các thành phần phụ
        self.model_updater = None
        self.experience_manager = None
        self.comparison_evaluator = None
        self._init_components()
        
        # Biến theo dõi trạng thái
        self.is_retraining = False
        self.last_retraining_time = None
        self.retraining_history = []
        self.current_job = None
        self.scheduler = None
        self.scheduler_thread = None
        
        # Khởi động lịch trình tái huấn luyện nếu được bật
        if self.config["retraining"]["schedule"]["enabled"]:
            self.setup_schedule()
        
        self.logger.info(f"Đã khởi tạo RetrainingPipeline cho {self.agent_type} phiên bản {self.model_version}")
    
    def _init_components(self) -> None:
        """
        Khởi tạo các thành phần phụ cho pipeline.
        """
        # Khởi tạo ExperienceManager
        try:
            self.experience_manager = ExperienceManager(
                agent_type=self.agent_type,
                model_version=self.model_version,
                output_dir=self.retraining_dir / "experience",
                preserve_experience=self.config["retraining"]["adaptation"]["preserve_experience"],
                logger=self.logger
            )
            self.logger.info("Đã khởi tạo ExperienceManager")
        except (ImportError, Exception) as e:
            self.logger.warning(f"Không thể khởi tạo ExperienceManager: {e}")
        
        # Khởi tạo ModelUpdater
        try:
            self.model_updater = ModelUpdater(
                agent_type=self.agent_type,
                model_version=self.model_version,
                output_dir=self.retraining_dir / "model_updates",
                keep_old_versions=self.config["retraining"]["deployment"]["keep_old_versions"],
                logger=self.logger
            )
            self.logger.info("Đã khởi tạo ModelUpdater")
        except (ImportError, Exception) as e:
            self.logger.warning(f"Không thể khởi tạo ModelUpdater: {e}")
        
        # Khởi tạo ComparisonEvaluator
        try:
            self.comparison_evaluator = ComparisonEvaluator(
                agent_type=self.agent_type,
                model_version=self.model_version,
                output_dir=self.retraining_dir / "evaluations",
                metrics=self.config["retraining"]["evaluation"]["metrics"],
                logger=self.logger
            )
            self.logger.info("Đã khởi tạo ComparisonEvaluator")
        except (ImportError, Exception) as e:
            self.logger.warning(f"Không thể khởi tạo ComparisonEvaluator: {e}")
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """
        Cập nhật từ điển lồng nhau.
        
        Args:
            d: Từ điển cần cập nhật
            u: Từ điển chứa các giá trị mới
            
        Returns:
            Từ điển đã cập nhật
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def setup_schedule(self) -> None:
        """
        Thiết lập lịch trình tái huấn luyện.
        """
        if not self.config["retraining"]["schedule"]["enabled"]:
            self.logger.info("Lịch trình tái huấn luyện bị tắt trong cấu hình")
            return
        
        # Lấy thông tin lịch trình
        schedule_config = self.config["retraining"]["schedule"]
        frequency = schedule_config.get("frequency", "weekly")
        day_of_week = schedule_config.get("day_of_week", 1)  # Thứ 2
        time_str = schedule_config.get("time", "02:00")
        
        # Hủy lịch trình hiện tại nếu có
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.logger.info("Hủy lịch trình hiện tại")
            self.scheduler_thread = None
            self.scheduler = None
        
        # Tạo lịch trình mới
        self.scheduler = schedule.Scheduler()
        
        # Thiết lập công việc theo tần suất
        job = None
        if frequency == "daily":
            job = self.scheduler.every().day.at(time_str).do(self.retraining_job)
            self.logger.info(f"Đã thiết lập tái huấn luyện hàng ngày lúc {time_str}")
        elif frequency == "weekly":
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            day_name = days[day_of_week % 7]
            job_getter = getattr(self.scheduler.every(), day_name)
            job = job_getter.at(time_str).do(self.retraining_job)
            self.logger.info(f"Đã thiết lập tái huấn luyện hàng tuần vào {day_name} lúc {time_str}")
        elif frequency == "monthly":
            # Không có hỗ trợ trực tiếp cho hàng tháng, phải tự cài đặt
            day_of_month = schedule_config.get("day_of_month", 1)
            
            def monthly_job():
                # Chỉ chạy nếu đúng ngày trong tháng
                if datetime.now().day == day_of_month:
                    self.retraining_job()
            
            job = self.scheduler.every().day.at(time_str).do(monthly_job)
            self.logger.info(f"Đã thiết lập tái huấn luyện hàng tháng vào ngày {day_of_month} lúc {time_str}")
        
        # Lưu công việc hiện tại
        if job:
            self.current_job = job
            
            # Khởi động luồng riêng cho scheduler
            def run_scheduler():
                while True:
                    self.scheduler.run_pending()
                    time.sleep(60)  # Kiểm tra mỗi phút
            
            self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            self.scheduler_thread.start()
    
    def retraining_job(self) -> Dict[str, Any]:
        """
        Công việc tái huấn luyện được lên lịch.
        
        Returns:
            Dict chứa kết quả tái huấn luyện
        """
        self.logger.info("Bắt đầu công việc tái huấn luyện theo lịch trình")
        
        # Kiểm tra xem có đang tái huấn luyện không
        if self.is_retraining:
            self.logger.warning("Đã có quá trình tái huấn luyện đang chạy. Bỏ qua công việc này.")
            return {
                "status": "skipped",
                "reason": "Đã có quá trình tái huấn luyện đang chạy",
                "timestamp": datetime.now().isoformat()
            }
        
        # Thực hiện tái huấn luyện
        result = self.retrain(reason="scheduled")
        
        return result
    
    def check_performance(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Kiểm tra hiệu suất và quyết định có nên tái huấn luyện không.
        
        Returns:
            Tuple (should_retrain, recommendation)
        """
        if not self.performance_tracker:
            self.logger.warning("Không có PerformanceTracker để kiểm tra hiệu suất")
            return False, None
        
        # Lấy khuyến nghị tái huấn luyện từ performance tracker
        recommendation = self.performance_tracker.get_retraining_recommendation()
        
        # Kiểm tra xem có nên tái huấn luyện không
        should_retrain = recommendation.get("should_retrain", False)
        
        if should_retrain:
            self.logger.info("PerformanceTracker khuyến nghị tái huấn luyện")
            
            # Kiểm tra số giao dịch tối thiểu
            # (Giả sử recommendation chứa thông tin về số lượng giao dịch)
            min_trades = self.config["retraining"]["performance_based"]["min_trades_required"]
            trades_count = recommendation.get("trades_count", 0)
            
            if trades_count < min_trades:
                self.logger.info(f"Số lượng giao dịch ({trades_count}) chưa đủ ngưỡng tối thiểu ({min_trades})")
                return False, recommendation
            
            return True, recommendation
        
        return False, recommendation
    
    def collect_new_data(self, lookback_days: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Thu thập dữ liệu mới cho tái huấn luyện.
        
        Args:
            lookback_days: Số ngày dữ liệu lịch sử (None để sử dụng giá trị từ cấu hình)
            
        Returns:
            Dict chứa dữ liệu mới, ánh xạ symbol -> DataFrame
        """
        if lookback_days is None:
            lookback_days = self.config["retraining"]["data_collection"]["lookback_days"]
        
        self.logger.info(f"Thu thập dữ liệu mới với lookback {lookback_days} ngày")
        
        # Kiểm tra xem có DataPipeline không
        if not self.data_pipeline:
            self.logger.error("Không có DataPipeline để thu thập dữ liệu")
            return {}
        
        # Lấy danh sách timeframes
        timeframes = self.config["retraining"]["data_collection"]["timeframes"]
        
        # Tính thời gian bắt đầu và kết thúc
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        all_data = {}
        
        # Thu thập dữ liệu cho mỗi timeframe
        for timeframe in timeframes:
            self.logger.info(f"Thu thập dữ liệu cho timeframe {timeframe}")
            
            try:
                # Sử dụng DataPipeline để thu thập dữ liệu
                # Lưu ý: run_pipeline là coroutine nên cần async/await
                # Nhưng vì hàm này không phải coroutine, nên chúng ta cần sử dụng
                # asyncio.run hoặc một event loop để chạy
                # Đoạn sau đây là pseudocode, bạn cần điều chỉnh cho phù hợp với môi trường

                # Giả định data_pipeline có phương thức load_data không bất đồng bộ
                if hasattr(self.data_pipeline, "load_data"):
                    # Tìm các file dữ liệu có sẵn
                    data_files = {}
                    for symbol in self.symbols:
                        file_pattern = f"{symbol.replace('/', '_')}_{timeframe}_*.parquet"
                        files = list(self.retraining_dir.glob(f"data/{file_pattern}"))
                        if files:
                            # Sử dụng file mới nhất
                            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                            data_files[symbol] = files[0]
                    
                    # Tải dữ liệu nếu có file
                    if data_files:
                        timeframe_data = self.data_pipeline.load_data(file_paths=data_files)
                        if timeframe_data:
                            for symbol, df in timeframe_data.items():
                                all_data[f"{symbol}_{timeframe}"] = df
                
                # Nếu không có đủ dữ liệu, thử thu thập trực tiếp
                if not all_data or len(all_data) < len(self.symbols):
                    self.logger.info("Không đủ dữ liệu có sẵn, thu thập trực tiếp")
                    
                    # Tạo data collector (giả định)
                    try:
                        # Hàm này là ví dụ, dựa trên file hiện có mà bạn đang xây dựng
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        data_collector = loop.run_until_complete(create_data_collector(
                            exchange_id=self.system_config.get("default_exchange", "binance"),
                            testnet=False
                        ))
                        
                        # Thu thập dữ liệu OHLCV
                        collected_data = loop.run_until_complete(data_collector.collect_all_symbols_ohlcv(
                            symbols=self.symbols,
                            timeframe=timeframe,
                            start_time=start_time,
                            end_time=end_time
                        ))
                        
                        # Đóng data collector
                        loop.run_until_complete(data_collector.exchange_connector.close())
                        loop.close()
                        
                        # Xử lý dữ liệu thu thập được
                        for symbol, df in collected_data.items():
                            if not df.empty:
                                all_data[f"{symbol}_{timeframe}"] = df
                                
                                # Lưu vào file để sử dụng sau này
                                output_dir = self.retraining_dir / "data"
                                output_dir.mkdir(parents=True, exist_ok=True)
                                file_name = f"{symbol.replace('/', '_')}_{timeframe}_{end_time.strftime('%Y%m%d')}.parquet"
                                df.to_parquet(output_dir / file_name, index=False)
                                
                                self.logger.info(f"Đã lưu {len(df)} dòng dữ liệu cho {symbol} ({timeframe})")
                    
                    except Exception as e:
                        self.logger.error(f"Lỗi khi tạo data collector: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi thu thập dữ liệu cho timeframe {timeframe}: {str(e)}")
        
        # Kiểm tra xem có đủ dữ liệu không
        min_data_points = self.config["retraining"]["data_collection"]["min_data_points"]
        if sum(len(df) for df in all_data.values()) < min_data_points:
            self.logger.warning(f"Không đủ dữ liệu cho tái huấn luyện (cần ít nhất {min_data_points} điểm dữ liệu)")
        
        return all_data
    
    def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Tiền xử lý dữ liệu cho tái huấn luyện.
        
        Args:
            data: Dict chứa dữ liệu thô, ánh xạ symbol -> DataFrame
            
        Returns:
            Dict chứa dữ liệu đã xử lý
        """
        if not data:
            self.logger.warning("Không có dữ liệu để tiền xử lý")
            return {}
        
        self.logger.info(f"Tiền xử lý dữ liệu cho {len(data)} bộ dữ liệu")
        
        # Kiểm tra xem có DataPipeline không
        if not self.data_pipeline:
            self.logger.error("Không có DataPipeline để tiền xử lý dữ liệu")
            return data
        
        try:
            # Làm sạch dữ liệu
            cleaned_data = self.data_pipeline.clean_data(
                data,
                clean_ohlcv=True,
                handle_leading_nan=True,
                leading_nan_method='backfill',
                min_periods=5,
                handle_extreme_volume=True
            )
            
            # Tạo đặc trưng
            featured_data = self.data_pipeline.generate_features(
                cleaned_data,
                all_indicators=True,
                clean_indicators=True
            )
            
            # Loại bỏ các chỉ báo trùng lặp
            pruned_data = self.data_pipeline.remove_redundant_indicators(
                featured_data,
                correlation_threshold=0.95
            )
            
            # Tạo các cột mục tiêu
            data_with_targets = self.data_pipeline.create_target_features(
                pruned_data,
                price_column="close",
                target_types=["direction", "return", "volatility"],
                horizons=[1, 3, 5, 10],
                threshold=0.001
            )
            
            self.logger.info("Đã hoàn thành tiền xử lý dữ liệu")
            return data_with_targets
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tiền xử lý dữ liệu: {str(e)}")
            return data
    
    def prepare_training_data(self, processed_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Chuẩn bị dữ liệu cho quá trình huấn luyện.
        
        Args:
            processed_data: Dict chứa dữ liệu đã xử lý, ánh xạ symbol -> DataFrame
            
        Returns:
            Dict chứa dữ liệu huấn luyện được tổ chức
        """
        if not processed_data:
            self.logger.warning("Không có dữ liệu đã xử lý để chuẩn bị cho huấn luyện")
            return {}
        
        self.logger.info(f"Chuẩn bị dữ liệu huấn luyện cho {len(processed_data)} bộ dữ liệu")
        
        # Tỷ lệ chia tập huấn luyện/kiểm thử
        train_test_split = self.config["retraining"]["training"]["validation_split"]
        
        # Dữ liệu được tổ chức
        organized_data = {}
        
        for key, df in processed_data.items():
            try:
                # Kiểm tra xem có cột mục tiêu không
                target_columns = [col for col in df.columns if col.startswith(("direction_", "return_", "volatility_"))]
                if not target_columns:
                    self.logger.warning(f"Không tìm thấy cột mục tiêu cho {key}, sử dụng cột 'close'")
                    if 'close' in df.columns:
                        # Tạo cột mục tiêu đơn giản nếu cần
                        df['direction_1'] = (df['close'].pct_change(1) > 0).astype(int)
                        target_columns = ['direction_1']
                    else:
                        self.logger.error(f"Không tìm thấy cột 'close' trong dữ liệu {key}")
                        continue
                
                # Chọn cột mục tiêu đầu tiên
                target_column = target_columns[0]
                
                # Loại bỏ hàng có giá trị NaN trong cột mục tiêu
                df_clean = df.dropna(subset=[target_column])
                
                if df_clean.empty:
                    self.logger.warning(f"Sau khi loại bỏ NaN, dữ liệu {key} rỗng")
                    continue
                
                # Chia tập huấn luyện/kiểm thử
                train_size = int(len(df_clean) * (1 - train_test_split))
                
                # Ưu tiên dữ liệu gần đây nếu được cấu hình
                if self.config["retraining"]["performance_based"]["prioritize_recent_data"]:
                    # Dữ liệu gần đây nhất vào tập kiểm thử
                    train_df = df_clean.iloc[:train_size].copy()
                    test_df = df_clean.iloc[train_size:].copy()
                else:
                    # Xáo trộn trước khi chia
                    df_shuffled = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
                    train_df = df_shuffled.iloc[:train_size].copy()
                    test_df = df_shuffled.iloc[train_size:].copy()
                
                # Phân tách đầu vào và mục tiêu
                # Loại bỏ các cột không cần thiết cho đầu vào
                ignored_columns = ['timestamp', 'open_time', 'close_time'] + target_columns
                feature_columns = [col for col in train_df.columns if col not in ignored_columns]
                
                X_train = train_df[feature_columns]
                y_train = train_df[target_column]
                
                X_test = test_df[feature_columns]
                y_test = test_df[target_column]
                
                # Tổ chức dữ liệu
                organized_data[key] = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'feature_columns': feature_columns,
                    'target_column': target_column,
                    'train_df': train_df,
                    'test_df': test_df
                }
                
                self.logger.info(f"Đã chuẩn bị dữ liệu huấn luyện cho {key}: {len(X_train)} mẫu huấn luyện, {len(X_test)} mẫu kiểm thử")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi chuẩn bị dữ liệu huấn luyện cho {key}: {str(e)}")
        
        return organized_data
    
    def create_new_agent(
        self,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        model_params: Optional[Dict[str, Any]] = None
    ) -> Optional[BaseAgent]:
        """
        Tạo một agent mới với các tham số được cung cấp.
        
        Args:
            state_dim: Kích thước không gian trạng thái
            action_dim: Kích thước không gian hành động
            model_params: Tham số mô hình
            
        Returns:
            Agent mới hoặc None nếu không thể tạo
        """
        if not model_params:
            model_params = {}
        
        # Kết hợp với cấu hình agent nếu có
        if self.agent_config:
            params = self.agent_config.copy()
            params.update(model_params)
        else:
            params = model_params
        
        # Thêm các tham số từ cấu hình huấn luyện
        training_config = self.config["retraining"]["training"]
        params["learning_rate"] = model_params.get("learning_rate", training_config.get("learning_rate", 0.001))
        
        # Tạo agent dựa trên loại
        try:
            if self.agent_type.lower() == "dqn":
                return DQNAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    name=f"dqn_agent_{self.model_version}_retrained",
                    logger=self.logger,
                    **params
                )
            elif self.agent_type.lower() == "ppo":
                return PPOAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    name=f"ppo_agent_{self.model_version}_retrained",
                    logger=self.logger,
                    **params
                )
            elif self.agent_type.lower() == "a2c":
                return A2CAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    name=f"a2c_agent_{self.model_version}_retrained",
                    logger=self.logger,
                    **params
                )
            else:
                self.logger.error(f"Loại agent không được hỗ trợ: {self.agent_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo agent mới: {str(e)}")
            return None
    
    def load_environment(self, symbol: str, data: pd.DataFrame) -> Optional[BaseEnvironment]:
        """
        Tải môi trường huấn luyện với dữ liệu cung cấp.
        
        Args:
            symbol: Cặp giao dịch
            data: DataFrame chứa dữ liệu
            
        Returns:
            Môi trường huấn luyện hoặc None nếu không thể tải
        """
        try:
            # Import động module môi trường
            import importlib
            env_module = importlib.import_module('environments.trading_gym.trading_env')
            TradingEnv = getattr(env_module, 'TradingEnv')
            
            # Tạo môi trường
            env = TradingEnv(
                symbol=symbol,
                data=data,
                initial_balance=10000,
                commission=0.001,
                window_size=20,
                logger=self.logger
            )
            
            return env
            
        except ImportError:
            self.logger.error("Không thể import module môi trường")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi tải môi trường: {str(e)}")
            return None
    
    def transfer_knowledge(self, source_agent: BaseAgent, target_agent: BaseAgent) -> bool:
        """
        Chuyển kiến thức từ agent cũ sang agent mới (transfer learning).
        
        Args:
            source_agent: Agent nguồn (cũ)
            target_agent: Agent đích (mới)
            
        Returns:
            True nếu chuyển thành công, False nếu không
        """
        if source_agent is None or target_agent is None:
            self.logger.warning("Không thể chuyển kiến thức: Thiếu agent nguồn hoặc đích")
            return False
        
        # Kiểm tra xem agent có cùng loại không
        if source_agent.__class__.__name__ != target_agent.__class__.__name__:
            self.logger.warning(f"Khác loại agent: {source_agent.__class__.__name__} -> {target_agent.__class__.__name__}")
            return False
        
        try:
            # Sao chép trọng số của mạng nơ-ron (nếu có)
            if hasattr(source_agent, 'q_network') and hasattr(target_agent, 'q_network'):
                # Đối với DQNAgent
                weights = source_agent.q_network.get_weights()
                target_agent.q_network.set_weights(weights)
                
                # Cập nhật mạng mục tiêu
                if hasattr(target_agent, 'update_target_network'):
                    target_agent.update_target_network()
                
                self.logger.info("Đã chuyển trọng số mạng Q-network")
                return True
                
            elif hasattr(source_agent, 'policy_network') and hasattr(target_agent, 'policy_network'):
                # Đối với PPOAgent hoặc A2CAgent
                policy_weights = source_agent.policy_network.get_weights()
                value_weights = source_agent.value_network.get_weights()
                
                target_agent.policy_network.set_weights(policy_weights)
                target_agent.value_network.set_weights(value_weights)
                
                self.logger.info("Đã chuyển trọng số mạng policy và value")
                return True
            
            else:
                self.logger.warning("Không tìm thấy mạng nơ-ron để chuyển trọng số")
                return False
                
        except Exception as e:
            self.logger.error(f"Lỗi khi chuyển kiến thức: {str(e)}")
            return False
    
    def train_agent(
        self,
        agent: BaseAgent,
        training_data: Dict[str, Dict[str, pd.DataFrame]],
        epochs: Optional[int] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        transfer_learning: bool = True
    ) -> Dict[str, Any]:
        """
        Huấn luyện agent với dữ liệu mới.
        
        Args:
            agent: Agent cần huấn luyện
            training_data: Dict chứa dữ liệu huấn luyện đã chuẩn bị
            epochs: Số epochs huấn luyện (None để sử dụng giá trị từ cấu hình)
            checkpoint_dir: Thư mục lưu checkpoint (None để tạo tự động)
            transfer_learning: Sử dụng transfer learning nếu có agent hiện tại
            
        Returns:
            Dict chứa kết quả huấn luyện
        """
        if not agent:
            self.logger.error("Không có agent để huấn luyện")
            return {"status": "error", "message": "Không có agent để huấn luyện"}
        
        if not training_data:
            self.logger.error("Không có dữ liệu huấn luyện")
            return {"status": "error", "message": "Không có dữ liệu huấn luyện"}
        
        # Lấy số epochs từ cấu hình nếu không được cung cấp
        if epochs is None:
            epochs = self.config["retraining"]["training"]["epochs"]
        
        # Thiết lập thư mục checkpoint
        if checkpoint_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = self.retraining_dir / "checkpoints" / f"{self.model_version}_{timestamp}"
        
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo cặp cờ để theo dõi tiến trình
        training_results = {}
        overall_success = True
        
        # Huấn luyện trên từng bộ dữ liệu
        for key, data in training_data.items():
            self.logger.info(f"Huấn luyện agent trên dữ liệu {key}")
            
            try:
                # Tạo môi trường huấn luyện nếu cần
                if self.environment_name == "trading_env":
                    # Nếu dữ liệu có train_df
                    if 'train_df' in data:
                        symbol = key.split('_')[0] if '_' in key else key
                        env = self.load_environment(symbol, data['train_df'])
                        
                        if env:
                            # Khởi tạo Trainer nếu chưa có
                            if self.trainer is None:
                                self.trainer = Trainer(
                                    agent=agent,
                                    env=env,
                                    output_dir=checkpoint_dir,
                                    logger=self.logger
                                )
                            else:
                                # Cập nhật agent và env cho trainer hiện có
                                self.trainer.agent = agent
                                self.trainer.env = env
                            
                            # Chạy huấn luyện
                            training_config = self.config["retraining"]["training"]
                            history = self.trainer.train(
                                env=env,
                                num_episodes=epochs,
                                save_freq=training_config.get("checkpoint_frequency", 10),
                                early_stopping=training_config.get("early_stopping", True),
                                eval_freq=training_config.get("eval_frequency", 10)
                            )
                            
                            # Lưu kết quả
                            training_results[key] = {
                                "status": "success",
                                "episodes": len(history.get("episode_rewards", [])),
                                "final_reward": history.get("episode_rewards", [])[-1] if history.get("episode_rewards") else None,
                                "mean_reward": np.mean(history.get("episode_rewards", [0])),
                                "history": {
                                    "rewards": history.get("episode_rewards", []),
                                    "losses": history.get("losses", [])
                                }
                            }
                    else:
                        # Huấn luyện bằng dữ liệu X_train, y_train
                        # Đây là cách tiếp cận thay thế nếu không có môi trường hoàn chỉnh
                        self.logger.info(f"Sử dụng phương pháp huấn luyện trực tiếp cho {key}")
                        
                        X_train = data['X_train']
                        y_train = data['y_train']
                        X_test = data['X_test']
                        y_test = data['y_test']
                        
                        # Huấn luyện dựa trên loại agent
                        if self.agent_type.lower() == "dqn":
                            # Mã giả cho việc huấn luyện DQN với dữ liệu có nhãn
                            for epoch in range(epochs):
                                losses = []
                                for i in range(0, len(X_train), 32):  # batch_size=32
                                    batch_X = X_train.iloc[i:i+32].values
                                    batch_y = y_train.iloc[i:i+32].values
                                    
                                    # Mô phỏng quá trình huấn luyện
                                    for j in range(len(batch_X)):
                                        state = batch_X[j]
                                        action = int(batch_y[j])  # Giả sử y là hành động
                                        reward = 1.0 if action == 1 else -1.0  # Giả định reward
                                        next_state = batch_X[j]  # Giả định next_state
                                        done = False
                                        
                                        # Lưu kinh nghiệm và học
                                        agent.remember(state, action, reward, next_state, done)
                                        loss_info = agent.learn()
                                        if loss_info and "loss" in loss_info:
                                            losses.append(loss_info["loss"])
                                
                                # Đánh giá trên tập kiểm thử
                                if epoch % 10 == 0:
                                    correct = 0
                                    for i in range(len(X_test)):
                                        state = X_test.iloc[i].values
                                        true_action = int(y_test.iloc[i])
                                        pred_action = agent.act(state, explore=False)
                                        if pred_action == true_action:
                                            correct += 1
                                    
                                    accuracy = correct / len(X_test)
                                    self.logger.info(f"Epoch {epoch}, Loss: {np.mean(losses):.4f}, Accuracy: {accuracy:.4f}")
                            
                            # Lưu kết quả
                            training_results[key] = {
                                "status": "success",
                                "epochs": epochs,
                                "final_loss": np.mean(losses) if losses else None,
                                "accuracy": accuracy,
                                "history": {
                                    "losses": losses
                                }
                            }
                        
                        elif self.agent_type.lower() in ["ppo", "a2c"]:
                            # Phương pháp huấn luyện cho agent chính sách
                            # (Code này chỉ là giả định, bạn cần điều chỉnh cho phù hợp)
                            self.logger.info(f"Agent loại {self.agent_type} cần môi trường hoàn chỉnh, đang sử dụng cách tiếp cận giả lập")
                            # Code giả lập...
                
            except Exception as e:
                self.logger.error(f"Lỗi khi huấn luyện agent trên dữ liệu {key}: {str(e)}")
                training_results[key] = {
                    "status": "error",
                    "message": str(e)
                }
                overall_success = False
        
        # Lưu agent đã huấn luyện
        try:
            model_path = checkpoint_dir / f"{agent.name}_final.h5"
            agent.save_model(path=model_path)
            self.logger.info(f"Đã lưu agent đã huấn luyện tại {model_path}")
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu agent đã huấn luyện: {str(e)}")
            overall_success = False
        
        # Tạo kết quả tổng thể
        result = {
            "status": "success" if overall_success else "partial_success",
            "agent_name": agent.name,
            "model_version": self.model_version,
            "checkpoint_dir": str(checkpoint_dir),
            "training_results": training_results,
            "completed_at": datetime.now().isoformat()
        }
        
        # Lưu kết quả huấn luyện
        result_path = checkpoint_dir / "training_result.json"
        try:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False, default=str)
            self.logger.info(f"Đã lưu kết quả huấn luyện tại {result_path}")
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu kết quả huấn luyện: {str(e)}")
        
        return result
    
    def evaluate_retrained_agent(
        self,
        agent: BaseAgent,
        test_data: Dict[str, pd.DataFrame],
        baseline_agent: Optional[BaseAgent] = None
    ) -> Dict[str, Any]:
        """
        Đánh giá hiệu suất của agent đã tái huấn luyện.
        
        Args:
            agent: Agent đã tái huấn luyện
            test_data: Dict chứa dữ liệu kiểm thử
            baseline_agent: Agent cơ sở (hiện tại) để so sánh
            
        Returns:
            Dict chứa kết quả đánh giá
        """
        if not agent:
            self.logger.error("Không có agent để đánh giá")
            return {"status": "error", "message": "Không có agent để đánh giá"}
        
        if not test_data:
            self.logger.error("Không có dữ liệu kiểm thử")
            return {"status": "error", "message": "Không có dữ liệu kiểm thử"}
        
        self.logger.info(f"Đánh giá agent {agent.name} trên {len(test_data)} bộ dữ liệu kiểm thử")
        
        # Khởi tạo đối tượng Backtester nếu có thể
        backtester = None
        try:
            backtester = Backtester(
                output_dir=self.retraining_dir / "evaluations" / f"{agent.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                logger=self.logger
            )
        except Exception as e:
            self.logger.warning(f"Không thể khởi tạo Backtester: {e}")
        
        # Kết quả đánh giá
        evaluation_results = {
            "agent": {
                "name": agent.name,
                "type": agent.__class__.__name__,
                "model_version": self.model_version
            },
            "symbols": {},
            "comparison": {}
        }
        
        # Đánh giá trên từng bộ dữ liệu
        for key, df in test_data.items():
            symbol = key.split('_')[0] if '_' in key else key
            timeframe = key.split('_')[1] if '_' in key and len(key.split('_')) > 1 else "1h"
            
            self.logger.info(f"Đánh giá trên dữ liệu {key}")
            
            try:
                # Đánh giá bằng Backtester
                if backtester:
                    # Đăng ký chiến lược đơn giản sử dụng agent
                    def agent_strategy(df, params):
                        # Xử lý dữ liệu để tạo trạng thái
                        state = df.iloc[-1][params.get("feature_columns", [])].values
                        
                        # Lấy hành động từ agent
                        action = params["agent"].act(state, explore=False)
                        
                        # Trả về tín hiệu giao dịch
                        if action == 1:  # Giả sử 1 là mua, 0 là không hành động
                            return {"action": "buy", "quantity": 1.0}
                        else:
                            return {"action": "hold"}
                    
                    backtester.register_strategy(
                        strategy_func=agent_strategy,
                        strategy_name=f"{agent.name}_{symbol}",
                        strategy_params={"agent": agent}
                    )
                    
                    # Tạo bản sao của dữ liệu kiểm thử
                    test_data_copy = {}
                    if isinstance(df, pd.DataFrame):
                        test_data_copy[symbol] = df
                    elif isinstance(df, dict) and 'test_df' in df:
                        test_data_copy[symbol] = df['test_df']
                    else:
                        test_data_copy[symbol] = df
                    
                    # Chạy backtest
                    backtest_result = backtester.run_backtest(
                        strategy_name=f"{agent.name}_{symbol}",
                        data=test_data_copy
                    )
                    
                    # Trích xuất kết quả
                    if backtest_result["status"] == "completed":
                        symbol_results = backtest_result["symbol_results"].get(symbol, {})
                        metrics = symbol_results.get("metrics", {})
                        
                        evaluation_results["symbols"][key] = {
                            "roi": symbol_results.get("roi", 0.0),
                            "final_balance": symbol_results.get("final_balance", 0.0),
                            "initial_balance": symbol_results.get("initial_balance", 0.0),
                            "metrics": metrics,
                            "status": "success"
                        }
                    else:
                        evaluation_results["symbols"][key] = {
                            "status": "error",
                            "message": "Backtest không thành công"
                        }
                
                # Đánh giá trực tiếp nếu không có Backtester
                else:
                    if isinstance(df, pd.DataFrame):
                        test_df = df
                    elif isinstance(df, dict) and 'test_df' in df:
                        test_df = df['test_df']
                    elif isinstance(df, dict) and 'X_test' in df and 'y_test' in df:
                        # Đánh giá dựa trên X_test và y_test
                        X_test = df['X_test']
                        y_test = df['y_test']
                        
                        # Đánh giá độ chính xác
                        correct = 0
                        for i in range(len(X_test)):
                            state = X_test.iloc[i].values
                            true_label = int(y_test.iloc[i])
                            pred_action = agent.act(state, explore=False)
                            if pred_action == true_label:
                                correct += 1
                        
                        accuracy = correct / len(X_test)
                        
                        evaluation_results["symbols"][key] = {
                            "accuracy": accuracy,
                            "samples": len(X_test),
                            "correct_predictions": correct,
                            "status": "success"
                        }
                        continue
                    else:
                        self.logger.warning(f"Không thể xác định dạng dữ liệu kiểm thử cho {key}")
                        test_df = df
                    
                    # Tính toán một số metrics cơ bản
                    total_samples = len(test_df)
                    correct_predictions = 0
                    predictions = []
                    
                    for i in range(len(test_df)):
                        # Giả sử có cột mục tiêu trong dữ liệu
                        target_columns = [col for col in test_df.columns if col.startswith(("direction_", "return_"))]
                        if target_columns:
                            target_column = target_columns[0]
                            true_label = int(test_df.iloc[i][target_column]) if target_column in test_df.columns else None
                            
                            # Lấy đặc trưng
                            feature_columns = [col for col in test_df.columns if col not in target_columns]
                            state = test_df.iloc[i][feature_columns].values
                            
                            # Dự đoán
                            pred_action = agent.act(state, explore=False)
                            predictions.append(pred_action)
                            
                            if true_label is not None and pred_action == true_label:
                                correct_predictions += 1
                    
                    if total_samples > 0:
                        accuracy = correct_predictions / total_samples
                    else:
                        accuracy = 0.0
                    
                    evaluation_results["symbols"][key] = {
                        "accuracy": accuracy,
                        "samples": total_samples,
                        "correct_predictions": correct_predictions,
                        "predictions": predictions[:10],  # Chỉ lưu 10 dự đoán đầu tiên
                        "status": "success"
                    }
                
                # So sánh với baseline nếu có
                if baseline_agent:
                    baseline_results = {}
                    
                    # Đánh giá tương tự với baseline_agent
                    # (Code tương tự như trên, nhưng với baseline_agent)
                    
                    evaluation_results["comparison"][key] = {
                        "baseline": baseline_results,
                        "improvement": {}  # Tính toán cải thiện
                    }
                
            except Exception as e:
                self.logger.error(f"Lỗi khi đánh giá trên dữ liệu {key}: {str(e)}")
                evaluation_results["symbols"][key] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Tính kết quả tổng thể
        overall_metrics = {}
        success_symbols = [s for s, r in evaluation_results["symbols"].items() if r.get("status") == "success"]
        
        if success_symbols:
            # Tính trung bình các metrics
            for metric in self.config["retraining"]["evaluation"]["metrics"]:
                values = []
                for symbol in success_symbols:
                    symbol_result = evaluation_results["symbols"][symbol]
                    if "metrics" in symbol_result and metric in symbol_result["metrics"]:
                        values.append(symbol_result["metrics"][metric])
                
                if values:
                    overall_metrics[metric] = sum(values) / len(values)
            
            # Tính độ chính xác tổng thể
            accuracies = [r.get("accuracy", 0.0) for s, r in evaluation_results["symbols"].items() if "accuracy" in r]
            if accuracies:
                overall_metrics["accuracy"] = sum(accuracies) / len(accuracies)
        
        evaluation_results["overall"] = {
            "metrics": overall_metrics,
            "success_symbols_count": len(success_symbols),
            "total_symbols_count": len(test_data)
        }
        
        # Đánh giá cải thiện tổng thể so với baseline
        if baseline_agent and "comparison" in evaluation_results:
            improvements = {}
            for metric, value in overall_metrics.items():
                baseline_value = 0.0  # Giá trị từ baseline nếu có
                # Tính cải thiện phần trăm
                if baseline_value != 0:
                    improvements[metric] = (value - baseline_value) / abs(baseline_value) * 100
            
            evaluation_results["overall"]["improvements"] = improvements
        
        # Lưu kết quả đánh giá
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = self.retraining_dir / "evaluations"
        eval_dir.mkdir(exist_ok=True, parents=True)
        
        eval_path = eval_dir / f"evaluation_{agent.name}_{timestamp}.json"
        try:
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, indent=4, ensure_ascii=False, default=str)
            self.logger.info(f"Đã lưu kết quả đánh giá tại {eval_path}")
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu kết quả đánh giá: {str(e)}")
        
        return evaluation_results
    
    def decide_deployment(
        self,
        evaluation_result: Dict[str, Any],
        improvement_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Quyết định có nên triển khai mô hình mới hay không.
        
        Args:
            evaluation_result: Dict chứa kết quả đánh giá
            improvement_threshold: Ngưỡng cải thiện để chấp nhận mô hình mới
            
        Returns:
            Dict chứa quyết định triển khai
        """
        if improvement_threshold is None:
            improvement_threshold = self.config["retraining"]["evaluation"]["improvement_threshold"]
        
        self.logger.info(f"Đánh giá triển khai với ngưỡng cải thiện {improvement_threshold:.2%}")
        
        # Kiểm tra kết quả đánh giá
        if not evaluation_result or "overall" not in evaluation_result:
            self.logger.error("Kết quả đánh giá không hợp lệ")
            return {
                "decision": "reject",
                "reason": "Kết quả đánh giá không hợp lệ",
                "timestamp": datetime.now().isoformat()
            }
        
        # Lấy metrics tổng thể
        overall_metrics = evaluation_result.get("overall", {}).get("metrics", {})
        improvements = evaluation_result.get("overall", {}).get("improvements", {})
        
        # Kiểm tra cải thiện
        is_better = False
        improved_metrics = []
        
        if improvements:
            # Nếu có so sánh trực tiếp với baseline
            for metric, value in improvements.items():
                if value >= improvement_threshold * 100:  # Chuyển ngưỡng thành phần trăm
                    improved_metrics.append({
                        "name": metric,
                        "improvement": f"{value:.2f}%"
                    })
            
            is_better = len(improved_metrics) > 0
            
        else:
            # So sánh với hiệu suất hiện tại
            current_performance = None
            if self.performance_tracker:
                latest_performance = self.performance_tracker.get_latest_performance()
                if "metrics" in latest_performance:
                    current_performance = latest_performance["metrics"]
            
            # Nếu có dữ liệu hiệu suất hiện tại
            if current_performance:
                for metric, new_value in overall_metrics.items():
                    if metric in current_performance:
                        current_value = current_performance[metric]
                        
                        # Tính cải thiện
                        if current_value != 0:
                            improvement = (new_value - current_value) / abs(current_value)
                            
                            # Đối với một số metrics, thấp hơn là tốt hơn
                            if metric in [BacktestMetric.MAX_DRAWDOWN.value, "volatility"]:
                                improvement = -improvement
                            
                            if improvement >= improvement_threshold:
                                improved_metrics.append({
                                    "name": metric,
                                    "improvement": f"{improvement:.2%}"
                                })
                
                is_better = len(improved_metrics) > 0
        
        # Quyết định triển khai
        if is_better and self.config["retraining"]["deployment"]["auto_deploy"]:
            decision = {
                "decision": "deploy",
                "reason": f"Cải thiện đạt ngưỡng. {len(improved_metrics)} metrics được cải thiện.",
                "improved_metrics": improved_metrics,
                "timestamp": datetime.now().isoformat()
            }
            self.logger.info(f"Quyết định: Triển khai mô hình mới. {len(improved_metrics)} metrics được cải thiện.")
        else:
            if not is_better:
                reason = "Không có metrics nào được cải thiện đáng kể."
            else:
                reason = "Tự động triển khai bị tắt trong cấu hình."
            
            decision = {
                "decision": "reject",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            self.logger.info(f"Quyết định: Không triển khai mô hình mới. {reason}")
        
        return decision
    
    def deploy_new_model(
        self,
        agent: BaseAgent,
        model_path: Union[str, Path],
        evaluation_result: Dict[str, Any],
        gradual_rollout: Optional[bool] = None,
        validation_period: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Triển khai mô hình mới sau khi tái huấn luyện.
        
        Args:
            agent: Agent mới để triển khai
            model_path: Đường dẫn đến model đã lưu
            evaluation_result: Dict chứa kết quả đánh giá
            gradual_rollout: Triển khai dần dần hay không
            validation_period: Thời gian xác thực trước khi triển khai đầy đủ (giờ)
            
        Returns:
            Dict chứa thông tin triển khai
        """
        if gradual_rollout is None:
            gradual_rollout = self.config["retraining"]["deployment"]["gradual_rollout"]
        
        if validation_period is None:
            validation_period = self.config["retraining"]["deployment"]["validation_period"]
        
        self.logger.info(f"Triển khai mô hình mới: {agent.name}")
        
        # Tạo phiên bản mới
        new_version = self._generate_new_version()
        
        # Thư mục triển khai
        deployment_dir = self.retraining_dir / "deployments" / new_version
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Lưu model vào thư mục triển khai
            new_model_path = deployment_dir / f"{agent.name}.h5"
            
            if isinstance(model_path, (str, Path)):
                # Sao chép file model
                shutil.copy(model_path, new_model_path)
            else:
                # Lưu model trực tiếp
                agent.save_model(path=new_model_path)
            
            # Lưu thông tin triển khai
            deployment_info = {
                "agent_name": agent.name,
                "agent_type": agent.__class__.__name__,
                "previous_version": self.model_version,
                "new_version": new_version,
                "deployed_at": datetime.now().isoformat(),
                "model_path": str(new_model_path),
                "evaluation_summary": {
                    "overall_metrics": evaluation_result.get("overall", {}).get("metrics", {}),
                    "improvements": evaluation_result.get("overall", {}).get("improvements", {})
                },
                "gradual_rollout": gradual_rollout,
                "validation_period": validation_period,
                "status": "pending" if gradual_rollout else "active"
            }
            
            # Lưu thông tin triển khai
            info_path = deployment_dir / "deployment_info.json"
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(deployment_info, f, indent=4, ensure_ascii=False, default=str)
            
            # Cập nhật ModelUpdater nếu có
            if self.model_updater:
                update_result = self.model_updater.register_new_version(
                    agent=agent,
                    version=new_version,
                    model_path=new_model_path,
                    metadata=deployment_info
                )
                
                if not update_result.get("success", False):
                    self.logger.warning(f"Không thể đăng ký phiên bản mới với ModelUpdater: {update_result.get('message')}")
            
            # Nếu triển khai dần dần, lên lịch xác thực sau validation_period
            if gradual_rollout:
                self.logger.info(f"Triển khai dần dần. Lên lịch xác thực sau {validation_period} giờ.")
                
                # Lên lịch xác thực
                schedule.every(validation_period).hours.do(
                    self.validate_gradual_deployment,
                    version=new_version,
                    deployment_dir=deployment_dir
                )
            else:
                # Triển khai đầy đủ ngay lập tức
                self.logger.info("Triển khai đầy đủ ngay lập tức")
                
                # Cập nhật phiên bản hiện tại
                self.model_version = new_version
                
                # Cập nhật agent
                self.agent = agent
                
                # Cập nhật trạng thái triển khai
                deployment_info["status"] = "active"
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(deployment_info, f, indent=4, ensure_ascii=False, default=str)
                
                # Cập nhật PerformanceTracker
                if self.performance_tracker:
                    self.performance_tracker.mark_retrained(
                        new_model_version=new_version,
                        reset_history=False
                    )
            
            # Gửi thông báo nếu được cấu hình
            if self.config["retraining"]["notification"]["enabled"] and self.config["retraining"]["notification"]["success_notification"]:
                self._send_notification(
                    title=f"Triển khai thành công: {agent.name} v{new_version}",
                    message=f"Mô hình mới đã được triển khai: {agent.name} phiên bản {new_version}.",
                    success=True
                )
            
            return {
                "status": "success",
                "agent_name": agent.name,
                "previous_version": self.model_version,
                "new_version": new_version,
                "deployment_dir": str(deployment_dir),
                "gradual_rollout": gradual_rollout,
                "deployment_info": deployment_info
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi khi triển khai mô hình mới: {str(e)}")
            
            # Gửi thông báo lỗi
            if self.config["retraining"]["notification"]["enabled"] and self.config["retraining"]["notification"]["failure_notification"]:
                self._send_notification(
                    title=f"Lỗi triển khai: {agent.name}",
                    message=f"Lỗi khi triển khai mô hình mới: {str(e)}",
                    success=False
                )
            
            return {
                "status": "error",
                "agent_name": agent.name,
                "message": f"Lỗi khi triển khai mô hình mới: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def validate_gradual_deployment(
        self,
        version: str,
        deployment_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Xác thực triển khai dần dần sau thời gian thử nghiệm.
        
        Args:
            version: Phiên bản cần xác thực
            deployment_dir: Thư mục triển khai
            
        Returns:
            Dict chứa kết quả xác thực
        """
        deployment_dir = Path(deployment_dir)
        self.logger.info(f"Xác thực triển khai dần dần cho phiên bản {version}")
        
        try:
            # Đọc thông tin triển khai
            info_path = deployment_dir / "deployment_info.json"
            if not info_path.exists():
                raise FileNotFoundError(f"Không tìm thấy file thông tin triển khai: {info_path}")
            
            with open(info_path, "r", encoding="utf-8") as f:
                deployment_info = json.load(f)
            
            # Kiểm tra xem triển khai có đang ở trạng thái pending không
            if deployment_info.get("status") != "pending":
                self.logger.warning(f"Triển khai không ở trạng thái pending: {deployment_info.get('status')}")
                return {
                    "status": "skipped",
                    "version": version,
                    "reason": f"Triển khai không ở trạng thái pending: {deployment_info.get('status')}"
                }
            
            # Kiểm tra hiệu suất trong thời gian thử nghiệm
            # Giả định rằng PerformanceTracker đã theo dõi hiệu suất của mô hình mới
            validation_passed = True
            validation_details = {}
            
            if self.performance_tracker:
                # Lấy báo cáo hiệu suất gần đây
                recent_report = self.performance_tracker.get_performance_summary(
                    days=1  # Chỉ xem hiệu suất 24 giờ gần đây
                )
                
                if recent_report and "metrics_summary" in recent_report:
                    # Kiểm tra có vấn đề nghiêm trọng nào không
                    serious_issues = []
                    
                    for metric, values in recent_report["metrics_summary"].items():
                        # Ví dụ: kiểm tra max_drawdown có vượt quá ngưỡng không
                        if metric == BacktestMetric.MAX_DRAWDOWN.value and values.get("max", 0.0) > 0.2:
                            serious_issues.append(f"Max drawdown vượt quá 20%: {values.get('max', 0.0):.2%}")
                        
                        # Ví dụ: kiểm tra win_rate có quá thấp không
                        if metric == BacktestMetric.WIN_RATE.value and values.get("mean", 0.0) < 0.4:
                            serious_issues.append(f"Win rate quá thấp: {values.get('mean', 0.0):.2%}")
                    
                    if serious_issues:
                        validation_passed = False
                        validation_details["issues"] = serious_issues
                
                # Thêm các kiểm tra hiệu suất khác nếu cần
            
            # Quyết định xác thực
            if validation_passed:
                # Cập nhật trạng thái triển khai thành active
                deployment_info["status"] = "active"
                deployment_info["validated_at"] = datetime.now().isoformat()
                
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(deployment_info, f, indent=4, ensure_ascii=False, default=str)
                
                # Cập nhật phiên bản hiện tại và agent
                previous_version = self.model_version
                self.model_version = version
                
                # Tải agent mới nếu cần
                model_path = deployment_info.get("model_path")
                if model_path and Path(model_path).exists() and self.agent:
                    try:
                        success = self.agent.load_model(model_path)
                        if success:
                            self.logger.info(f"Đã tải agent mới từ {model_path}")
                    except Exception as e:
                        self.logger.error(f"Lỗi khi tải agent mới: {e}")
                
                # Cập nhật PerformanceTracker
                if self.performance_tracker:
                    self.performance_tracker.mark_retrained(
                        new_model_version=version,
                        reset_history=False
                    )
                
                self.logger.info(f"Xác thực thành công. Đã triển khai đầy đủ phiên bản {version}.")
                
                # Gửi thông báo
                if self.config["retraining"]["notification"]["enabled"] and self.config["retraining"]["notification"]["success_notification"]:
                    self._send_notification(
                        title=f"Triển khai hoàn tất: v{version}",
                        message=f"Mô hình v{version} đã được xác thực và triển khai đầy đủ.",
                        success=True
                    )
                
                return {
                    "status": "success",
                    "version": version,
                    "previous_version": previous_version,
                    "validated_at": deployment_info["validated_at"]
                }
                
            else:
                # Không vượt qua xác thực, quay lại phiên bản cũ
                deployment_info["status"] = "rejected"
                deployment_info["rejected_at"] = datetime.now().isoformat()
                deployment_info["rejection_reason"] = validation_details
                
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(deployment_info, f, indent=4, ensure_ascii=False, default=str)
                
                self.logger.warning(f"Xác thực thất bại. Không triển khai phiên bản {version}.")
                
                # Gửi thông báo
                if self.config["retraining"]["notification"]["enabled"] and self.config["retraining"]["notification"]["failure_notification"]:
                    self._send_notification(
                        title=f"Triển khai thất bại: v{version}",
                        message=f"Mô hình v{version} không vượt qua xác thực: {serious_issues}",
                        success=False
                    )
                
                return {
                    "status": "rejected",
                    "version": version,
                    "rejection_reason": validation_details,
                    "rejected_at": deployment_info["rejected_at"]
                }
                
        except Exception as e:
            self.logger.error(f"Lỗi khi xác thực triển khai dần dần: {str(e)}")
            
            # Gửi thông báo
            if self.config["retraining"]["notification"]["enabled"] and self.config["retraining"]["notification"]["failure_notification"]:
                self._send_notification(
                    title=f"Lỗi xác thực: v{version}",
                    message=f"Lỗi khi xác thực mô hình v{version}: {str(e)}",
                    success=False
                )
            
            return {
                "status": "error",
                "version": version,
                "message": f"Lỗi khi xác thực: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def rollback_deployment(
        self,
        to_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Quay lại phiên bản cũ nếu có vấn đề với phiên bản mới.
        
        Args:
            to_version: Phiên bản cần quay lại (None để quay lại phiên bản gần nhất)
            
        Returns:
            Dict chứa kết quả rollback
        """
        if not self.config["retraining"]["deployment"]["rollback_enabled"]:
            self.logger.warning("Tính năng rollback bị tắt trong cấu hình")
            return {
                "status": "disabled",
                "message": "Tính năng rollback bị tắt trong cấu hình"
            }
        
        self.logger.info(f"Đang quay lại phiên bản cũ: {to_version if to_version else 'gần nhất'}")
        
        try:
            # Tìm phiên bản để quay lại
            target_version = to_version
            
            if not target_version:
                # Lấy danh sách phiên bản
                versions = []
                deployments_dir = self.retraining_dir / "deployments"
                
                if deployments_dir.exists():
                    for path in deployments_dir.iterdir():
                        if path.is_dir():
                            info_path = path / "deployment_info.json"
                            if info_path.exists():
                                try:
                                    with open(info_path, "r", encoding="utf-8") as f:
                                        info = json.load(f)
                                    
                                    if info.get("status") == "active" and info.get("new_version") != self.model_version:
                                        versions.append({
                                            "version": info.get("new_version"),
                                            "deployed_at": info.get("deployed_at"),
                                            "path": str(path)
                                        })
                                except Exception:
                                    continue
                
                # Sắp xếp theo thời gian giảm dần
                versions.sort(key=lambda x: x.get("deployed_at", ""), reverse=True)
                
                if versions:
                    target_version = versions[0].get("version")
                else:
                    self.logger.warning("Không tìm thấy phiên bản cũ để quay lại")
                    return {
                        "status": "error",
                        "message": "Không tìm thấy phiên bản cũ để quay lại"
                    }
            
            # Tìm thông tin và model path của phiên bản mục tiêu
            target_info = None
            target_model_path = None
            target_dir = self.retraining_dir / "deployments" / target_version
            
            if target_dir.exists():
                info_path = target_dir / "deployment_info.json"
                if info_path.exists():
                    with open(info_path, "r", encoding="utf-8") as f:
                        target_info = json.load(f)
                    
                    target_model_path = target_info.get("model_path")
                else:
                    self.logger.error(f"Không tìm thấy file thông tin cho phiên bản {target_version}")
                    return {
                        "status": "error",
                        "message": f"Không tìm thấy file thông tin cho phiên bản {target_version}"
                    }
            else:
                self.logger.error(f"Không tìm thấy thư mục cho phiên bản {target_version}")
                return {
                    "status": "error",
                    "message": f"Không tìm thấy thư mục cho phiên bản {target_version}"
                }
            
            # Kiểm tra model path
            if not target_model_path or not Path(target_model_path).exists():
                self.logger.error(f"Không tìm thấy file model cho phiên bản {target_version}: {target_model_path}")
                return {
                    "status": "error",
                    "message": f"Không tìm thấy file model cho phiên bản {target_version}"
                }
            
            # Lưu phiên bản hiện tại
            current_version = self.model_version
            
            # Tải model cũ
            if self.agent:
                try:
                    success = self.agent.load_model(target_model_path)
                    if not success:
                        self.logger.error(f"Không thể tải model từ {target_model_path}")
                        return {
                            "status": "error",
                            "message": f"Không thể tải model từ {target_model_path}"
                        }
                except Exception as e:
                    self.logger.error(f"Lỗi khi tải model: {str(e)}")
                    return {
                        "status": "error",
                        "message": f"Lỗi khi tải model: {str(e)}"
                    }
            
            # Cập nhật phiên bản
            self.model_version = target_version
            
            # Cập nhật PerformanceTracker
            if self.performance_tracker:
                self.performance_tracker.mark_retrained(
                    new_model_version=target_version,
                    reset_history=False
                )
            
            # Ghi log rollback
            rollback_info = {
                "timestamp": datetime.now().isoformat(),
                "from_version": current_version,
                "to_version": target_version,
                "reason": "Manual rollback" if to_version else "Automatic rollback",
                "model_path": target_model_path
            }
            
            # Lưu thông tin rollback
            rollback_dir = self.retraining_dir / "rollbacks"
            rollback_dir.mkdir(exist_ok=True, parents=True)
            rollback_path = rollback_dir / f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(rollback_path, "w", encoding="utf-8") as f:
                json.dump(rollback_info, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Đã quay lại phiên bản {target_version} thành công")
            
            # Gửi thông báo
            if self.config["retraining"]["notification"]["enabled"]:
                self._send_notification(
                    title=f"Rollback thành công: v{target_version}",
                    message=f"Đã quay lại phiên bản {target_version} từ phiên bản {current_version}.",
                    success=True
                )
            
            return {
                "status": "success",
                "from_version": current_version,
                "to_version": target_version,
                "rollback_info": rollback_info
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi khi quay lại phiên bản cũ: {str(e)}")
            
            # Gửi thông báo
            if self.config["retraining"]["notification"]["enabled"] and self.config["retraining"]["notification"]["failure_notification"]:
                self._send_notification(
                    title="Lỗi rollback",
                    message=f"Lỗi khi quay lại phiên bản cũ: {str(e)}",
                    success=False
                )
            
            return {
                "status": "error",
                "message": f"Lỗi khi quay lại phiên bản cũ: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def run_adaptation(self) -> Dict[str, Any]:
        """
        Chạy quá trình tự thích nghi của agent.
        
        Returns:
            Dict chứa kết quả thích nghi
        """
        if not self.config["retraining"]["adaptation"]["enabled"]:
            self.logger.info("Tính năng tự thích nghi bị tắt trong cấu hình")
            return {
                "status": "disabled",
                "message": "Tính năng tự thích nghi bị tắt trong cấu hình"
            }
        
        if not self.adaptation_module:
            self.logger.error("Không có AdaptationModule để thực hiện thích nghi")
            return {
                "status": "error",
                "message": "Không có AdaptationModule để thực hiện thích nghi"
            }
        
        self.logger.info("Bắt đầu quá trình tự thích nghi")
        
        try:
            # Cập nhật dữ liệu thị trường mới nhất
            market_data = None
            for symbol in self.symbols:
                # Giả định lấy dữ liệu từ thư mục data
                file_pattern = f"{symbol.replace('/', '_')}_1h_*.parquet"
                files = list(self.retraining_dir.glob(f"data/{file_pattern}"))
                if files:
                    # Sử dụng file mới nhất
                    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    try:
                        market_data = pd.read_parquet(files[0])
                        break
                    except Exception:
                        continue
            
            # Thực hiện thích nghi
            if market_data is not None:
                # Phát hiện sự thay đổi thị trường và thích nghi nếu cần
                market_change_detected = self.adaptation_module.detect_market_change(market_data)
                
                if market_change_detected:
                    self.logger.info("Phát hiện thay đổi thị trường, thực hiện thích nghi sâu")
                    adaptation_result = self.adaptation_module.adapt_to_market_change(market_data)
                else:
                    self.logger.info("Thực hiện thích nghi thông thường")
                    adaptation_result = self.adaptation_module.adapt()
                
                # Lưu trạng thái thích nghi
                self.adaptation_module.save_state()
                
                # Cập nhật PerformanceTracker với tham số mới
                if self.performance_tracker and adaptation_result.get("success", False):
                    # Lưu thay đổi tham số vào lịch sử hiệu suất
                    if "new_parameters" in adaptation_result:
                        param_changes = {
                            "timestamp": datetime.now().isoformat(),
                            "parameters": adaptation_result["new_parameters"],
                            "reason": "adaptation",
                            "agent_version": self.model_version
                        }
                        
                        # Cập nhật hiệu suất với thay đổi tham số
                        self.performance_tracker.update_performance({
                            "parameter_changes": param_changes
                        })
                
                return {
                    "status": "success",
                    "adaptation_result": adaptation_result,
                    "market_change_detected": market_change_detected,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Thích nghi thông thường nếu không có dữ liệu thị trường
                adaptation_result = self.adaptation_module.adapt()
                
                # Lưu trạng thái thích nghi
                self.adaptation_module.save_state()
                
                return {
                    "status": "success",
                    "adaptation_result": adaptation_result,
                    "market_change_detected": False,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Lỗi khi thực hiện tự thích nghi: {str(e)}")
            return {
                "status": "error",
                "message": f"Lỗi khi thực hiện tự thích nghi: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_new_version(self) -> str:
        """
        Tạo số phiên bản mới dựa trên phiên bản hiện tại.
        
        Returns:
            Chuỗi phiên bản mới
        """
        # Phân tích phiên bản hiện tại
        try:
            parts = self.model_version.split('.')
            
            if len(parts) >= 3:
                # Semantic versioning: major.minor.patch
                major = int(parts[0].strip('v'))
                minor = int(parts[1])
                patch = int(parts[2])
                
                # Tăng patch version
                patch += 1
                
                return f"v{major}.{minor}.{patch}"
            else:
                # Phiên bản đơn giản
                try:
                    # Thử phân tích phiên bản là "v1", "v2", ...
                    version_num = int(self.model_version.strip('v'))
                    return f"v{version_num + 1}"
                except ValueError:
                    # Nếu không thể phân tích, thêm timestamp
                    timestamp = datetime.now().strftime("%Y%m%d%H%M")
                    return f"{self.model_version}_{timestamp}"
                
        except Exception:
            # Nếu không thể phân tích, tạo phiên bản mới với timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            return f"v1.0.0_{timestamp}"
    
    def _send_notification(
        self,
        title: str,
        message: str,
        success: bool
    ) -> bool:
        """
        Gửi thông báo về quá trình tái huấn luyện.
        
        Args:
            title: Tiêu đề thông báo
            message: Nội dung thông báo
            success: Thành công hay thất bại
            
        Returns:
            True nếu gửi thành công, False nếu không
        """
        if not self.config["retraining"]["notification"]["enabled"]:
            return False
        
        try:
            notification_config = self.config["retraining"]["notification"]
            
            # Gửi email nếu được bật
            if notification_config.get("email", False):
                try:
                    # Mã giả cho việc gửi email
                    self.logger.info(f"[Email] {title}: {message}")
                    
                    # Ở đây sẽ có code để gửi email thực tế
                    # Ví dụ: send_email(to=email_addresses, subject=title, body=message)
                except Exception as e:
                    self.logger.error(f"Lỗi khi gửi thông báo email: {str(e)}")
            
            # Gửi thông báo Telegram nếu được bật
            if notification_config.get("telegram", False):
                try:
                    # Mã giả cho việc gửi thông báo Telegram
                    self.logger.info(f"[Telegram] {title}: {message}")
                    
                    # Ở đây sẽ có code để gửi thông báo Telegram thực tế
                    # Ví dụ: send_telegram(chat_id=telegram_chat_id, text=f"{title}\n\n{message}")
                except Exception as e:
                    self.logger.error(f"Lỗi khi gửi thông báo Telegram: {str(e)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi gửi thông báo: {str(e)}")
            return False
    
    def retrain(
        self,
        reason: str = "manual",
        lookback_days: Optional[int] = None,
        force: bool = False,
        transfer_learning: bool = True
    ) -> Dict[str, Any]:
        """
        Thực hiện quy trình tái huấn luyện đầy đủ.
        
        Args:
            reason: Lý do tái huấn luyện ("manual", "scheduled", "performance")
            lookback_days: Số ngày dữ liệu lịch sử (None để sử dụng giá trị từ cấu hình)
            force: Bắt buộc tái huấn luyện ngay cả khi đang có quá trình khác
            transfer_learning: Sử dụng transfer learning từ mô hình hiện tại
            
        Returns:
            Dict chứa kết quả tái huấn luyện
        """
        # Kiểm tra xem có đang tái huấn luyện không
        if self.is_retraining and not force:
            self.logger.warning("Đã có quá trình tái huấn luyện đang chạy. Bỏ qua yêu cầu mới.")
            return {
                "status": "skipped",
                "reason": "Đã có quá trình tái huấn luyện đang chạy",
                "timestamp": datetime.now().isoformat()
            }
        
        # Bắt đầu quá trình tái huấn luyện
        self.is_retraining = True
        self.last_retraining_time = datetime.now()
        
        # Tạo ID duy nhất cho quá trình tái huấn luyện
        retraining_id = f"retrain_{self.agent_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Thư mục lưu trữ cho quá trình tái huấn luyện này
        retraining_dir = self.retraining_dir / retraining_id
        retraining_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Bắt đầu quá trình tái huấn luyện: {retraining_id} (lý do: {reason})")
        
        # Khởi tạo kết quả
        result = {
            "id": retraining_id,
            "status": "running",
            "reason": reason,
            "agent_type": self.agent_type,
            "model_version": self.model_version,
            "start_time": self.last_retraining_time.isoformat(),
            "steps": {},
            "final_status": None
        }
        
        try:
            # Bước 1: Thu thập dữ liệu mới
            self.logger.info("Bước 1: Thu thập dữ liệu mới")
            start_time = time.time()
            
            new_data = self.collect_new_data(lookback_days=lookback_days)
            
            result["steps"]["collect_data"] = {
                "status": "success" if new_data else "error",
                "time_taken": time.time() - start_time,
                "data_collected": {symbol: len(df) for symbol, df in new_data.items()},
                "symbols": list(new_data.keys())
            }
            
            if not new_data:
                self.logger.error("Không thể thu thập dữ liệu mới. Dừng quá trình tái huấn luyện.")
                result["status"] = "error"
                result["error_message"] = "Không thể thu thập dữ liệu mới"
                result["final_status"] = "failed"
                self.is_retraining = False
                return result
            
            # Bước 2: Tiền xử lý dữ liệu
            self.logger.info("Bước 2: Tiền xử lý dữ liệu")
            start_time = time.time()
            
            processed_data = self.preprocess_data(new_data)
            
            result["steps"]["preprocess_data"] = {
                "status": "success" if processed_data else "error",
                "time_taken": time.time() - start_time,
                "processed_data_info": {symbol: len(df) for symbol, df in processed_data.items()}
            }
            
            if not processed_data:
                self.logger.error("Lỗi khi tiền xử lý dữ liệu. Dừng quá trình tái huấn luyện.")
                result["status"] = "error"
                result["error_message"] = "Lỗi khi tiền xử lý dữ liệu"
                result["final_status"] = "failed"
                self.is_retraining = False
                return result
            
            # Bước 3: Chuẩn bị dữ liệu huấn luyện
            self.logger.info("Bước 3: Chuẩn bị dữ liệu huấn luyện")
            start_time = time.time()
            
            training_data = self.prepare_training_data(processed_data)
            
            result["steps"]["prepare_training_data"] = {
                "status": "success" if training_data else "error",
                "time_taken": time.time() - start_time,
                "training_data_info": {symbol: len(data.get('X_train', [])) for symbol, data in training_data.items()}
            }
            
            if not training_data:
                self.logger.error("Lỗi khi chuẩn bị dữ liệu huấn luyện. Dừng quá trình tái huấn luyện.")
                result["status"] = "error"
                result["error_message"] = "Lỗi khi chuẩn bị dữ liệu huấn luyện"
                result["final_status"] = "failed"
                self.is_retraining = False
                return result
            
            # Bước 4: Tạo agent mới hoặc sao chép agent hiện tại
            self.logger.info("Bước 4: Tạo agent mới")
            start_time = time.time()
            
            # Lấy thông tin state_dim và action_dim từ dữ liệu huấn luyện
            example_data = next(iter(training_data.values()))
            
            # Xác định state_dim và action_dim
            if 'X_train' in example_data:
                state_dim = example_data['X_train'].shape[1]
                # Giả định action_dim từ cột mục tiêu
                if example_data['target_column'].startswith('direction_'):
                    action_dim = 2  # Binary classification
                else:
                    action_dim = 1  # Regression
            else:
                # Giá trị mặc định nếu không thể xác định
                state_dim = 20
                action_dim = 2
            
            # Tạo agent mới với tham số từ cấu hình
            new_agent = self.create_new_agent(
                state_dim=state_dim,
                action_dim=action_dim,
                model_params=self.config["retraining"]["training"]
            )
            
            result["steps"]["create_agent"] = {
                "status": "success" if new_agent else "error",
                "time_taken": time.time() - start_time,
                "agent_info": {
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "agent_type": self.agent_type
                }
            }
            
            if not new_agent:
                self.logger.error("Lỗi khi tạo agent mới. Dừng quá trình tái huấn luyện.")
                result["status"] = "error"
                result["error_message"] = "Lỗi khi tạo agent mới"
                result["final_status"] = "failed"
                self.is_retraining = False
                return result
            
            # Bước 5: Chuyển kiến thức từ agent cũ sang agent mới (nếu có)
            if transfer_learning and self.agent:
                self.logger.info("Bước 5: Chuyển kiến thức từ agent cũ")
                start_time = time.time()
                
                transfer_success = self.transfer_knowledge(self.agent, new_agent)
                
                result["steps"]["transfer_knowledge"] = {
                    "status": "success" if transfer_success else "skipped",
                    "time_taken": time.time() - start_time,
                    "transfer_success": transfer_success
                }
            
            # Bước 6: Huấn luyện agent mới
            self.logger.info("Bước 6: Huấn luyện agent mới")
            start_time = time.time()
            
            training_result = self.train_agent(
                agent=new_agent,
                training_data=training_data,
                epochs=self.config["retraining"]["training"]["epochs"],
                checkpoint_dir=retraining_dir / "checkpoints",
                transfer_learning=transfer_learning
            )
            
            result["steps"]["train_agent"] = {
                "status": training_result.get("status", "error"),
                "time_taken": time.time() - start_time,
                "training_summary": training_result
            }
            
            if training_result.get("status") not in ["success", "partial_success"]:
                self.logger.error("Lỗi khi huấn luyện agent mới. Dừng quá trình tái huấn luyện.")
                result["status"] = "error"
                result["error_message"] = "Lỗi khi huấn luyện agent mới"
                result["final_status"] = "failed"
                self.is_retraining = False
                return result
            
            # Bước 7: Đánh giá agent mới
            self.logger.info("Bước 7: Đánh giá agent mới")
            start_time = time.time()
            
            # Tạo dữ liệu kiểm thử
            test_data = {}
            for key, data in training_data.items():
                if 'test_df' in data:
                    test_data[key] = data['test_df']
                elif 'X_test' in data and 'y_test' in data:
                    test_data[key] = {
                        'X_test': data['X_test'],
                        'y_test': data['y_test']
                    }
            
            evaluation_result = self.evaluate_retrained_agent(
                agent=new_agent,
                test_data=test_data,
                baseline_agent=self.agent
            )
            
            result["steps"]["evaluate_agent"] = {
                "status": "success" if evaluation_result.get("overall") else "error",
                "time_taken": time.time() - start_time,
                "evaluation_summary": {
                    "overall_metrics": evaluation_result.get("overall", {}).get("metrics", {}),
                    "symbols_evaluated": list(evaluation_result.get("symbols", {}).keys())
                }
            }
            
            # Bước 8: Quyết định có triển khai hay không
            self.logger.info("Bước 8: Quyết định triển khai")
            start_time = time.time()
            
            deployment_decision = self.decide_deployment(evaluation_result)
            
            result["steps"]["decide_deployment"] = {
                "status": "success",
                "time_taken": time.time() - start_time,
                "decision": deployment_decision.get("decision"),
                "reason": deployment_decision.get("reason")
            }
            
            # Bước 9: Triển khai nếu được quyết định
            if deployment_decision.get("decision") == "deploy":
                self.logger.info("Bước 9: Triển khai mô hình mới")
                start_time = time.time()
                
                # Lấy đường dẫn model
                model_path = None
                if "training_summary" in result["steps"]["train_agent"]:
                    model_path = result["steps"]["train_agent"]["training_summary"].get("checkpoint_dir")
                    if model_path:
                        model_path = Path(model_path) / f"{new_agent.name}_final.h5"
                
                deployment_result = self.deploy_new_model(
                    agent=new_agent,
                    model_path=model_path,
                    evaluation_result=evaluation_result,
                    gradual_rollout=self.config["retraining"]["deployment"]["gradual_rollout"]
                )
                
                result["steps"]["deploy_model"] = {
                    "status": deployment_result.get("status", "error"),
                    "time_taken": time.time() - start_time,
                    "deployment_summary": deployment_result
                }
                
                # Cập nhật trạng thái cuối cùng
                if deployment_result.get("status") == "success":
                    result["final_status"] = "deployed"
                    
                    # Cập nhật agent và model_version nếu không triển khai dần dần
                    if not self.config["retraining"]["deployment"]["gradual_rollout"]:
                        self.agent = new_agent
                        self.model_version = deployment_result.get("new_version", self.model_version)
                else:
                    result["final_status"] = "trained_not_deployed"
            else:
                # Không triển khai
                result["final_status"] = "trained_not_deployed"
            
            # Cập nhật trạng thái chung
            result["status"] = "completed"
            result["end_time"] = datetime.now().isoformat()
            
            # Lưu kết quả
            result_path = retraining_dir / "retraining_result.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False, default=str)
            
            # Lưu vào lịch sử
            self.retraining_history.append(result)
            
            # Lưu kèm agent nếu không triển khai
            if result["final_status"] == "trained_not_deployed" and new_agent:
                try:
                    model_save_path = retraining_dir / f"{new_agent.name}_not_deployed.h5"
                    new_agent.save_model(path=model_save_path)
                    self.logger.info(f"Đã lưu agent không được triển khai tại {model_save_path}")
                except Exception as e:
                    self.logger.error(f"Lỗi khi lưu agent không được triển khai: {str(e)}")
            
            self.logger.info(f"Hoàn thành quá trình tái huấn luyện: {retraining_id}")
            
            # Gửi thông báo kết quả
            if self.config["retraining"]["notification"]["enabled"]:
                if result["final_status"] == "deployed":
                    self._send_notification(
                        title="Tái huấn luyện thành công và triển khai",
                        message=f"Đã hoàn thành tái huấn luyện và triển khai mô hình mới. ID: {retraining_id}",
                        success=True
                    )
                elif result["final_status"] == "trained_not_deployed":
                    self._send_notification(
                        title="Tái huấn luyện thành công nhưng không triển khai",
                        message=f"Đã hoàn thành tái huấn luyện nhưng mô hình mới không được triển khai. ID: {retraining_id}",
                        success=True
                    )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thực hiện tái huấn luyện: {str(e)}")
            
            # Cập nhật trạng thái
            result["status"] = "error"
            result["error_message"] = str(e)
            result["final_status"] = "failed"
            result["end_time"] = datetime.now().isoformat()
            
            # Lưu kết quả
            try:
                result_path = retraining_dir / "retraining_result.json"
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4, ensure_ascii=False, default=str)
                
                # Lưu vào lịch sử
                self.retraining_history.append(result)
            except Exception:
                pass
            
            # Gửi thông báo lỗi
            if self.config["retraining"]["notification"]["enabled"] and self.config["retraining"]["notification"]["failure_notification"]:
                self._send_notification(
                    title="Lỗi trong quá trình tái huấn luyện",
                    message=f"Lỗi: {str(e)}. ID: {retraining_id}",
                    success=False
                )
            
            # Đặt lại trạng thái
            self.is_retraining = False
            
            return result
    
    def get_retraining_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Lấy lịch sử tái huấn luyện.
        
        Args:
            limit: Số lượng bản ghi lịch sử trả về
            
        Returns:
            Danh sách lịch sử tái huấn luyện
        """
        history = sorted(
            self.retraining_history,
            key=lambda x: x.get("start_time", ""),
            reverse=True
        )
        return history[:limit]
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của quá trình tái huấn luyện.
        
        Returns:
            Dict chứa thông tin trạng thái
        """
        status = {
            "is_retraining": self.is_retraining,
            "last_retraining_time": self.last_retraining_time.isoformat() if self.last_retraining_time else None,
            "model_version": self.model_version,
            "agent_type": self.agent_type,
            "symbols": self.symbols
        }
        
        # Thêm thông tin từ retraining_history nếu có
        if self.retraining_history:
            latest = self.retraining_history[-1]
            status["latest_retraining"] = {
                "id": latest.get("id"),
                "status": latest.get("status"),
                "final_status": latest.get("final_status"),
                "reason": latest.get("reason"),
                "start_time": latest.get("start_time"),
                "end_time": latest.get("end_time")
            }
        
        # Thêm thông tin từ performance_tracker nếu có
        if self.performance_tracker:
            performance_recommendation = self.performance_tracker.get_retraining_recommendation()
            status["performance_recommendation"] = {
                "should_retrain": performance_recommendation.get("should_retrain", False),
                "recommendation_age": performance_recommendation.get("days_since_recommendation")
            }
        
        return status


def create_retraining_pipeline(
    agent_or_config: Union[BaseAgent, Dict[str, Any]],
    agent_type: str = "dqn",
    model_version: str = "v1.0.0",
    environment_name: str = "trading_env",
    strategy_name: str = "default_strategy",
    symbols: List[str] = ["BTC/USDT"],
    retraining_dir: Optional[Union[str, Path]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> RetrainingPipeline:
    """
    Hàm tiện ích để tạo RetrainingPipeline.
    
    Args:
        agent_or_config: Agent cần tái huấn luyện hoặc cấu hình agent
        agent_type: Loại agent ("dqn", "ppo", "a2c", v.v.)
        model_version: Phiên bản hiện tại của mô hình
        environment_name: Tên môi trường huấn luyện
        strategy_name: Tên chiến lược giao dịch
        symbols: Danh sách các cặp tiền được giao dịch
        retraining_dir: Thư mục chứa dữ liệu tái huấn luyện
        config_overrides: Các ghi đè cấu hình
        logger: Logger tùy chỉnh
        
    Returns:
        RetrainingPipeline đã được cấu hình
    """
    # Xác định agent và config
    agent = None
    agent_config = None
    
    if isinstance(agent_or_config, BaseAgent):
        agent = agent_or_config
        agent_config = agent.config
    elif isinstance(agent_or_config, dict):
        agent_config = agent_or_config
    else:
        raise ValueError("agent_or_config phải là BaseAgent hoặc Dict")
    
    # Tạo pipeline
    pipeline = RetrainingPipeline(
        agent=agent,
        agent_config=agent_config,
        agent_type=agent_type,
        model_version=model_version,
        environment_name=environment_name,
        strategy_name=strategy_name,
        symbols=symbols,
        retraining_dir=retraining_dir,
        config=config_overrides,
        logger=logger
    )
    
    return pipeline


def schedule_retraining(
    pipeline: RetrainingPipeline,
    frequency: str = "weekly",
    day_of_week: int = 1,
    time: str = "02:00",
    check_performance: bool = True,
    check_interval_hours: int = 24
) -> bool:
    """
    Lên lịch tái huấn luyện định kỳ.
    
    Args:
        pipeline: RetrainingPipeline cần lên lịch
        frequency: Tần suất tái huấn luyện ("daily", "weekly", "monthly")
        day_of_week: Thứ trong tuần (0=CN, 1=T2, ...)
        time: Thời gian trong ngày (giờ:phút)
        check_performance: Bật kiểm tra hiệu suất tự động
        check_interval_hours: Khoảng thời gian kiểm tra hiệu suất (giờ)
        
    Returns:
        True nếu lên lịch thành công, False nếu không
    """
    try:
        # Cập nhật cấu hình schedule
        pipeline.config["retraining"]["schedule"] = {
            "enabled": True,
            "frequency": frequency,
            "day_of_week": day_of_week,
            "time": time,
            "timezone": "UTC"
        }
        
        # Cập nhật cấu hình kiểm tra hiệu suất
        pipeline.config["retraining"]["performance_based"]["enabled"] = check_performance
        pipeline.config["retraining"]["performance_based"]["check_frequency"] = check_interval_hours
        
        # Thiết lập lịch trình
        pipeline.setup_schedule()
        
        # Thiết lập lịch trình kiểm tra hiệu suất nếu được bật
        if check_performance and pipeline.scheduler:
            # Lên lịch kiểm tra hiệu suất định kỳ
            def check_performance_job():
                should_retrain, recommendation = pipeline.check_performance()
                if should_retrain:
                    # Tự động tái huấn luyện
                    pipeline.retrain(reason="performance")
            
            # Thêm công việc kiểm tra hiệu suất
            pipeline.scheduler.every(check_interval_hours).hours.do(check_performance_job)
            pipeline.logger.info(f"Đã lên lịch kiểm tra hiệu suất mỗi {check_interval_hours} giờ")
        
        return True
        
    except Exception as e:
        if pipeline.logger:
            pipeline.logger.error(f"Lỗi khi lên lịch tái huấn luyện: {str(e)}")
        return False


def run_retraining_pipeline(
    agent_or_config: Union[BaseAgent, Dict[str, Any]],
    agent_type: str = "dqn",
    model_version: str = "v1.0.0",
    environment_name: str = "trading_env",
    strategy_name: str = "default_strategy",
    symbols: List[str] = ["BTC/USDT"],
    retraining_dir: Optional[Union[str, Path]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    schedule: bool = False,
    frequency: str = "weekly",
    run_now: bool = True
) -> Dict[str, Any]:
    """
    Chạy pipeline tái huấn luyện.
    
    Args:
        agent_or_config: Agent cần tái huấn luyện hoặc cấu hình agent
        agent_type: Loại agent ("dqn", "ppo", "a2c", v.v.)
        model_version: Phiên bản hiện tại của mô hình
        environment_name: Tên môi trường huấn luyện
        strategy_name: Tên chiến lược giao dịch
        symbols: Danh sách các cặp tiền được giao dịch
        retraining_dir: Thư mục chứa dữ liệu tái huấn luyện
        config_overrides: Các ghi đè cấu hình
        logger: Logger tùy chỉnh
        schedule: Lên lịch tái huấn luyện định kỳ
        frequency: Tần suất tái huấn luyện ("daily", "weekly", "monthly")
        run_now: Chạy tái huấn luyện ngay lập tức
        
    Returns:
        Dict chứa kết quả tái huấn luyện hoặc thông tin pipeline
    """
    # Tạo pipeline
    pipeline = create_retraining_pipeline(
        agent_or_config=agent_or_config,
        agent_type=agent_type,
        model_version=model_version,
        environment_name=environment_name,
        strategy_name=strategy_name,
        symbols=symbols,
        retraining_dir=retraining_dir,
        config_overrides=config_overrides,
        logger=logger
    )
    
    # Lên lịch nếu cần
    if schedule:
        schedule_retraining(
            pipeline=pipeline,
            frequency=frequency
        )
    
    # Chạy ngay nếu cần
    if run_now:
        return pipeline.retrain(reason="manual")
    else:
        return {
            "status": "created",
            "pipeline": pipeline,
            "agent_type": agent_type,
            "model_version": model_version,
            "scheduled": schedule
        }