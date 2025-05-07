"""
Theo dõi hiệu suất hệ thống giao dịch tự động.
File này định nghĩa các lớp và hàm để theo dõi và phân tích hiệu suất hệ thống giao dịch,
tự động phát hiện suy giảm hiệu suất và đưa ra đề xuất tái huấn luyện.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import json
import time
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import threading
import queue
import os
import signal
import warnings

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from automation.metrics.performance_metrics import AutomationPerformanceMetrics
from automation.metrics.efficiency_metrics import EfficiencyMetrics
from automation.metrics.evaluation_metrics import EvaluationMetrics

class PerformanceTracker:
    """
    Lớp theo dõi và phân tích hiệu suất hệ thống giao dịch tự động.
    Cung cấp các cơ chế theo dõi liên tục và phát hiện suy giảm hiệu suất.
    """
    
    def __init__(
        self,
        system_id: str,
        config_path: Optional[Union[str, Path]] = None,
        metrics_save_dir: Optional[Union[str, Path]] = None,
        alert_thresholds: Optional[Dict[str, float]] = None,
        tracking_interval: int = 3600,  # Theo dõi mỗi giờ
        check_interval: int = 60,      # Kiểm tra dữ liệu mỗi phút
        auto_retrain_threshold: float = 0.7,  # Ngưỡng tự động tái huấn luyện (0.0-1.0)
        min_trades_for_evaluation: int = 30,
        logger: Optional[logging.Logger] = None,
        callback_handler: Optional[Any] = None
    ):
        """
        Khởi tạo đối tượng PerformanceTracker.
        
        Args:
            system_id (str): ID duy nhất cho hệ thống giao dịch
            config_path (Union[str, Path], optional): Đường dẫn file cấu hình
            metrics_save_dir (Union[str, Path], optional): Thư mục lưu dữ liệu metrics
            alert_thresholds (Dict[str, float], optional): Ngưỡng cảnh báo cho các chỉ số
            tracking_interval (int, optional): Chu kỳ theo dõi (giây)
            check_interval (int, optional): Chu kỳ kiểm tra dữ liệu (giây)
            auto_retrain_threshold (float, optional): Ngưỡng tự động tái huấn luyện
            min_trades_for_evaluation (int, optional): Số giao dịch tối thiểu để đánh giá
            logger (logging.Logger, optional): Logger tùy chỉnh
            callback_handler (Any, optional): Đối tượng xử lý callback từ tracker
        """
        # Thiết lập logger
        self.logger = logger or get_logger(f"performance_tracker_{system_id}")
        
        # Lưu các tham số
        self.system_id = system_id
        self.tracking_interval = tracking_interval
        self.check_interval = check_interval
        self.auto_retrain_threshold = auto_retrain_threshold
        self.min_trades_for_evaluation = min_trades_for_evaluation
        self.callback_handler = callback_handler
        
        # Đường dẫn thư mục lưu metrics
        if metrics_save_dir is None:
            metrics_save_dir = Path("logs") / "performance" / system_id
        else:
            metrics_save_dir = Path(metrics_save_dir)
        
        self.metrics_save_dir = metrics_save_dir
        self.metrics_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Tải cấu hình
        self.config = self._load_config(config_path)
        
        # Thiết lập ngưỡng cảnh báo
        self.alert_thresholds = alert_thresholds or {
            "drawdown": 0.15,              # Cảnh báo khi drawdown vượt quá 15%
            "win_rate_decline": 0.15,      # Cảnh báo khi tỷ lệ thắng giảm 15%
            "sharpe_decline": 0.3,         # Cảnh báo khi Sharpe giảm 30%
            "volatility_increase": 0.5,    # Cảnh báo khi biến động tăng 50%
            "profit_factor_decline": 0.3,  # Cảnh báo khi profit factor giảm 30%
            "consecutive_losses": 5,       # Cảnh báo sau 5 giao dịch thua liên tiếp
            "efficiency_decline": 0.2      # Cảnh báo khi hiệu quả giảm 20%
        }
        
        # Khởi tạo các đối tượng metrics
        self.performance_metrics = AutomationPerformanceMetrics(
            equity_curve=pd.Series(dtype=float),  # Khởi tạo rỗng
            alert_thresholds=self.alert_thresholds,
            logger=self.logger
        )
        
        self.efficiency_metrics = EfficiencyMetrics(
            efficiency_threshold=0.7,
            resource_threshold=0.8,
            time_threshold=1.5,
            history_file=self.metrics_save_dir / "efficiency_history.json",
            logger=self.logger
        )
        
        self.evaluation_metrics = EvaluationMetrics(
            min_improvement_threshold=0.05,
            stability_threshold=0.7,
            version_history_file=self.metrics_save_dir / "version_history.json",
            logger=self.logger
        )
        
        # Dữ liệu theo dõi
        self.tracking_data = {
            "equity_curve": pd.Series(dtype=float),
            "trades": pd.DataFrame(),
            "metrics_history": [],
            "alerts": [],
            "retraining_recommendations": []
        }
        
        # Trạng thái theo dõi
        self.is_tracking = False
        self.tracking_thread = None
        self.stop_event = threading.Event()
        self.data_queue = queue.Queue()
        
        # Thông tin phiên bản hiện tại
        self.current_model_version = None
        self.current_model_info = None
        
        # Thời gian theo dõi
        self.start_time = None
        self.last_check_time = None
        self.last_save_time = None
        
        self.logger.info(f"Đã khởi tạo PerformanceTracker cho hệ thống {system_id}")
        
        # Tải dữ liệu lịch sử nếu có
        self._load_tracking_data()
    
    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Tải cấu hình từ file.
        
        Args:
            config_path (Union[str, Path], optional): Đường dẫn file cấu hình
            
        Returns:
            Dict[str, Any]: Cấu hình đã tải
        """
        # Cấu hình mặc định
        default_config = {
            "performance_check": {
                "enable_auto_recommendations": True,
                "metrics_to_track": [
                    "sharpe_ratio", "sortino_ratio", "win_rate", "profit_factor", 
                    "max_drawdown", "expectancy", "calmar_ratio"
                ],
                "metric_weights": {
                    "sharpe_ratio": 0.15,
                    "sortino_ratio": 0.15,
                    "win_rate": 0.10,
                    "profit_factor": 0.15,
                    "max_drawdown": 0.15,
                    "expectancy": 0.15,
                    "calmar_ratio": 0.15
                },
                "evaluation_windows": [30, 60, 90, 180],  # ngày
                "save_interval": 86400,  # 24 giờ
            },
            "alert_config": {
                "enable_alerts": True,
                "alert_channels": ["log", "email"],
                "email_recipients": [],
                "alert_cooldown": 3600  # 1 giờ
            },
            "auto_retrain": {
                "enable_auto_retrain": False,
                "retrain_cooldown": 604800,  # 7 ngày
                "min_trades_for_retrain": 100,
                "min_days_for_retrain": 30
            }
        }
        
        # Nếu không có file cấu hình, sử dụng cấu hình mặc định
        if config_path is None:
            return default_config
        
        # Tải cấu hình từ file
        config_path = Path(config_path)
        if not config_path.exists():
            self.logger.warning(f"Không tìm thấy file cấu hình tại {config_path}. Sử dụng cấu hình mặc định.")
            return default_config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Kết hợp với cấu hình mặc định
            for section in default_config:
                if section not in config:
                    config[section] = default_config[section]
                else:
                    for key in default_config[section]:
                        if key not in config[section]:
                            config[section][key] = default_config[section][key]
            
            self.logger.info(f"Đã tải cấu hình từ {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải file cấu hình: {str(e)}. Sử dụng cấu hình mặc định.")
            return default_config
    
    def _load_tracking_data(self) -> None:
        """
        Tải dữ liệu theo dõi từ file.
        """
        # Đường dẫn file dữ liệu
        data_file = self.metrics_save_dir / "tracking_data.json"
        
        if not data_file.exists():
            self.logger.info("Không tìm thấy file dữ liệu theo dõi hiệu suất. Khởi tạo mới.")
            return
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Khôi phục dữ liệu
            if "equity_data" in data and data["equity_data"]:
                self.tracking_data["equity_curve"] = pd.Series(
                    data["equity_data"]["values"],
                    index=pd.to_datetime(data["equity_data"]["timestamps"])
                )
            
            if "trades_data" in data and data["trades_data"]:
                self.tracking_data["trades"] = pd.DataFrame(data["trades_data"])
                
                # Chuyển đổi cột thời gian
                if "entry_time" in self.tracking_data["trades"].columns:
                    self.tracking_data["trades"]["entry_time"] = pd.to_datetime(self.tracking_data["trades"]["entry_time"])
                if "exit_time" in self.tracking_data["trades"].columns:
                    self.tracking_data["trades"]["exit_time"] = pd.to_datetime(self.tracking_data["trades"]["exit_time"])
            
            # Khôi phục lịch sử metrics
            if "metrics_history" in data:
                self.tracking_data["metrics_history"] = data["metrics_history"]
            
            # Khôi phục cảnh báo
            if "alerts" in data:
                self.tracking_data["alerts"] = []
                for alert in data["alerts"]:
                    if "timestamp" in alert:
                        alert["timestamp"] = datetime.fromisoformat(alert["timestamp"])
                    self.tracking_data["alerts"].append(alert)
            
            # Khôi phục đề xuất tái huấn luyện
            if "retraining_recommendations" in data:
                self.tracking_data["retraining_recommendations"] = []
                for rec in data["retraining_recommendations"]:
                    if "timestamp" in rec:
                        rec["timestamp"] = datetime.fromisoformat(rec["timestamp"])
                    self.tracking_data["retraining_recommendations"].append(rec)
            
            # Khôi phục thông tin phiên bản
            if "current_model_version" in data:
                self.current_model_version = data["current_model_version"]
            
            if "current_model_info" in data:
                self.current_model_info = data["current_model_info"]
            
            # Nếu có dữ liệu, cập nhật đối tượng performance_metrics
            if len(self.tracking_data["equity_curve"]) > 0:
                self.performance_metrics = AutomationPerformanceMetrics(
                    equity_curve=self.tracking_data["equity_curve"],
                    trades=self.tracking_data["trades"] if not self.tracking_data["trades"].empty else None,
                    alert_thresholds=self.alert_thresholds,
                    logger=self.logger
                )
            
            self.logger.info(f"Đã tải dữ liệu theo dõi hiệu suất từ {data_file}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải dữ liệu theo dõi hiệu suất: {str(e)}")
    
    def _save_tracking_data(self) -> None:
        """
        Lưu dữ liệu theo dõi vào file.
        """
        # Đường dẫn file dữ liệu
        data_file = self.metrics_save_dir / "tracking_data.json"
        
        try:
            # Chuẩn bị dữ liệu để lưu
            equity_data = {}
            if not self.tracking_data["equity_curve"].empty:
                equity_data = {
                    "timestamps": [t.isoformat() for t in self.tracking_data["equity_curve"].index],
                    "values": self.tracking_data["equity_curve"].values.tolist()
                }
            
            trades_data = []
            if not self.tracking_data["trades"].empty:
                # Chuyển đổi DataFrame thành list của dict
                trades_data = self.tracking_data["trades"].to_dict(orient="records")
                
                # Chuyển đổi các giá trị datetime thành string
                for trade in trades_data:
                    for key, value in trade.items():
                        if isinstance(value, pd.Timestamp) or isinstance(value, datetime):
                            trade[key] = value.isoformat()
            
            # Chuyển đổi các cảnh báo và đề xuất
            alerts = []
            for alert in self.tracking_data["alerts"]:
                alert_copy = alert.copy()
                if "timestamp" in alert_copy and isinstance(alert_copy["timestamp"], (datetime, pd.Timestamp)):
                    alert_copy["timestamp"] = alert_copy["timestamp"].isoformat()
                alerts.append(alert_copy)
            
            recommendations = []
            for rec in self.tracking_data["retraining_recommendations"]:
                rec_copy = rec.copy()
                if "timestamp" in rec_copy and isinstance(rec_copy["timestamp"], (datetime, pd.Timestamp)):
                    rec_copy["timestamp"] = rec_copy["timestamp"].isoformat()
                recommendations.append(rec_copy)
            
            # Tạo đối tượng dữ liệu
            data = {
                "system_id": self.system_id,
                "equity_data": equity_data,
                "trades_data": trades_data,
                "metrics_history": self.tracking_data["metrics_history"],
                "alerts": alerts,
                "retraining_recommendations": recommendations,
                "current_model_version": self.current_model_version,
                "current_model_info": self.current_model_info,
                "last_updated": datetime.now().isoformat()
            }
            
            # Lưu dữ liệu
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            self.last_save_time = datetime.now()
            self.logger.info(f"Đã lưu dữ liệu theo dõi hiệu suất vào {data_file}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu dữ liệu theo dõi hiệu suất: {str(e)}")
    
    def start_tracking(self) -> None:
        """
        Bắt đầu theo dõi hiệu suất.
        """
        if self.is_tracking:
            self.logger.warning("Đang theo dõi hiệu suất. Bỏ qua yêu cầu bắt đầu mới.")
            return
        
        # Đặt lại sự kiện dừng
        self.stop_event.clear()
        
        # Đặt các biến thời gian
        self.start_time = datetime.now()
        self.last_check_time = self.start_time
        self.last_save_time = self.start_time
        
        # Bắt đầu thread theo dõi
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
        self.is_tracking = True
        self.logger.info("Đã bắt đầu theo dõi hiệu suất")
    
    def stop_tracking(self) -> None:
        """
        Dừng theo dõi hiệu suất.
        """
        if not self.is_tracking:
            self.logger.warning("Không đang theo dõi hiệu suất. Bỏ qua yêu cầu dừng.")
            return
        
        # Đặt sự kiện dừng
        self.stop_event.set()
        
        # Đợi thread kết thúc
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=10)
        
        self.is_tracking = False
        self.logger.info("Đã dừng theo dõi hiệu suất")
        
        # Lưu dữ liệu
        self._save_tracking_data()
    
    def _tracking_loop(self) -> None:
        """
        Vòng lặp theo dõi hiệu suất.
        """
        try:
            while not self.stop_event.is_set():
                # Kiểm tra hàng đợi dữ liệu
                self._process_data_queue()
                
                # Kiểm tra hiệu suất
                current_time = datetime.now()
                if (current_time - self.last_check_time).total_seconds() >= self.tracking_interval:
                    self._check_performance()
                    self.last_check_time = current_time
                
                # Lưu dữ liệu định kỳ
                save_interval = self.config.get("performance_check", {}).get("save_interval", 86400)
                if (current_time - self.last_save_time).total_seconds() >= save_interval:
                    self._save_tracking_data()
                
                # Đợi một khoảng thời gian
                time.sleep(self.check_interval)
                
        except Exception as e:
            self.logger.error(f"Lỗi trong vòng lặp theo dõi hiệu suất: {str(e)}")
            self.is_tracking = False
    
    def _process_data_queue(self) -> None:
        """
        Xử lý dữ liệu từ hàng đợi.
        """
        # Xử lý tất cả các mục trong hàng đợi
        while not self.data_queue.empty():
            try:
                data_item = self.data_queue.get_nowait()
                
                # Xác định loại dữ liệu và xử lý
                if "type" in data_item:
                    if data_item["type"] == "equity":
                        self._process_equity_data(data_item)
                    elif data_item["type"] == "trade":
                        self._process_trade_data(data_item)
                    elif data_item["type"] == "metrics":
                        self._process_metrics_data(data_item)
                    elif data_item["type"] == "model_update":
                        self._process_model_update(data_item)
                    else:
                        self.logger.warning(f"Loại dữ liệu không được hỗ trợ: {data_item['type']}")
                
                # Đánh dấu mục đã xử lý
                self.data_queue.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Lỗi khi xử lý dữ liệu từ hàng đợi: {str(e)}")
    
    def _process_equity_data(self, data_item: Dict[str, Any]) -> None:
        """
        Xử lý dữ liệu giá trị vốn.
        
        Args:
            data_item (Dict[str, Any]): Dữ liệu giá trị vốn
        """
        # Trích xuất dữ liệu
        timestamp = data_item.get("timestamp", datetime.now())
        value = data_item.get("value", 0.0)
        
        # Đảm bảo timestamp là đối tượng datetime
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Cập nhật chuỗi giá trị vốn
        self.tracking_data["equity_curve"].loc[timestamp] = value
        
        # Sắp xếp theo thời gian
        self.tracking_data["equity_curve"].sort_index(inplace=True)
        
        # Cập nhật đối tượng performance_metrics
        self.performance_metrics = AutomationPerformanceMetrics(
            equity_curve=self.tracking_data["equity_curve"],
            trades=self.tracking_data["trades"] if not self.tracking_data["trades"].empty else None,
            alert_thresholds=self.alert_thresholds,
            logger=self.logger
        )
    
    def _process_trade_data(self, data_item: Dict[str, Any]) -> None:
        """
        Xử lý dữ liệu giao dịch.
        
        Args:
            data_item (Dict[str, Any]): Dữ liệu giao dịch
        """
        # Trích xuất dữ liệu
        trade_info = data_item.get("trade", {})
        
        # Kiểm tra dữ liệu
        if not trade_info:
            self.logger.warning("Dữ liệu giao dịch trống")
            return
        
        # Chuyển đổi thời gian
        for time_field in ["entry_time", "exit_time"]:
            if time_field in trade_info and isinstance(trade_info[time_field], str):
                trade_info[time_field] = datetime.fromisoformat(trade_info[time_field])
        
        # Thêm giao dịch vào DataFrame
        new_trade = pd.DataFrame([trade_info])
        
        if self.tracking_data["trades"].empty:
            self.tracking_data["trades"] = new_trade
        else:
            self.tracking_data["trades"] = pd.concat([self.tracking_data["trades"], new_trade], ignore_index=True)
        
        # Cập nhật đối tượng performance_metrics
        self.performance_metrics = AutomationPerformanceMetrics(
            equity_curve=self.tracking_data["equity_curve"],
            trades=self.tracking_data["trades"],
            alert_thresholds=self.alert_thresholds,
            logger=self.logger
        )
    
    def _process_metrics_data(self, data_item: Dict[str, Any]) -> None:
        """
        Xử lý dữ liệu metrics.
        
        Args:
            data_item (Dict[str, Any]): Dữ liệu metrics
        """
        # Trích xuất dữ liệu
        metrics = data_item.get("metrics", {})
        timestamp = data_item.get("timestamp", datetime.now())
        source = data_item.get("source", "unknown")
        
        # Kiểm tra dữ liệu
        if not metrics:
            self.logger.warning("Dữ liệu metrics trống")
            return
        
        # Đảm bảo timestamp là đối tượng datetime
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Thêm vào lịch sử metrics
        metrics_entry = {
            "timestamp": timestamp,
            "source": source,
            "metrics": metrics
        }
        
        self.tracking_data["metrics_history"].append(metrics_entry)
    
    def _process_model_update(self, data_item: Dict[str, Any]) -> None:
        """
        Xử lý cập nhật mô hình.
        
        Args:
            data_item (Dict[str, Any]): Thông tin cập nhật mô hình
        """
        # Trích xuất dữ liệu
        version = data_item.get("version")
        model_info = data_item.get("model_info", {})
        metrics = data_item.get("metrics", {})
        
        # Kiểm tra dữ liệu
        if not version:
            self.logger.warning("Thông tin phiên bản mô hình trống")
            return
        
        # Cập nhật thông tin phiên bản
        self.current_model_version = version
        self.current_model_info = model_info
        
        # Thêm phiên bản vào lịch sử đánh giá
        self.evaluation_metrics.add_model_version(
            version=version,
            model_info=model_info,
            metrics=metrics,
            replace_if_exists=True
        )
        
        self.logger.info(f"Đã cập nhật thông tin mô hình: {version}")
    
    def _check_performance(self) -> None:
        """
        Kiểm tra hiệu suất hệ thống và phát hiện suy giảm.
        """
        # Kiểm tra xem có đủ dữ liệu không
        if len(self.tracking_data["equity_curve"]) < 10:
            self.logger.debug("Không đủ dữ liệu để kiểm tra hiệu suất")
            return
        
        # Kiểm tra các cảnh báo hiệu suất
        metrics_update = self.performance_metrics.update_metrics(
            new_equity_point=self.tracking_data["equity_curve"].iloc[-1]
        )
        
        # Xử lý các cảnh báo mới
        if "alerts" in metrics_update and metrics_update["alerts"]:
            for alert in metrics_update["alerts"]:
                # Kiểm tra xem cảnh báo đã tồn tại chưa
                is_duplicate = False
                for existing_alert in self.tracking_data["alerts"]:
                    if (
                        existing_alert.get("type") == alert.get("type") and
                        (datetime.now() - existing_alert.get("timestamp", datetime.min)).total_seconds() < 
                        self.config.get("alert_config", {}).get("alert_cooldown", 3600)
                    ):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # Thêm vào lịch sử cảnh báo
                    self.tracking_data["alerts"].append(alert)
                    
                    # Gửi cảnh báo
                    self._send_alert(alert)
        
        # Kiểm tra đề xuất tái huấn luyện
        if self.config.get("performance_check", {}).get("enable_auto_recommendations", True):
            recommendation = self.performance_metrics.get_retraining_recommendation()
            
            if recommendation["should_retrain"]:
                # Kiểm tra xem đề xuất đã tồn tại chưa
                is_duplicate = False
                for existing_rec in self.tracking_data["retraining_recommendations"]:
                    if (datetime.now() - existing_rec.get("timestamp", datetime.min)).total_seconds() < 86400:  # 1 ngày
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # Thêm vào lịch sử đề xuất
                    self.tracking_data["retraining_recommendations"].append(recommendation)
                    
                    # Xử lý đề xuất tái huấn luyện
                    self._handle_retraining_recommendation(recommendation)
    
    def _send_alert(self, alert: Dict[str, Any]) -> None:
        """
        Gửi cảnh báo.
        
        Args:
            alert (Dict[str, Any]): Thông tin cảnh báo
        """
        # Kiểm tra có bật cảnh báo không
        if not self.config.get("alert_config", {}).get("enable_alerts", True):
            return
        
        # Ghi log cảnh báo
        self.logger.warning(f"CẢNH BÁO: {alert['message']} (Độ nghiêm trọng: {alert.get('severity', 'medium')})")
        
        # Kiểm tra các kênh cảnh báo
        alert_channels = self.config.get("alert_config", {}).get("alert_channels", ["log"])
        
        # Gọi callback nếu có
        if self.callback_handler:
            try:
                self.callback_handler.on_alert(alert)
            except Exception as e:
                self.logger.error(f"Lỗi khi gọi callback xử lý cảnh báo: {str(e)}")
    
    def _handle_retraining_recommendation(self, recommendation: Dict[str, Any]) -> None:
        """
        Xử lý đề xuất tái huấn luyện.
        
        Args:
            recommendation (Dict[str, Any]): Đề xuất tái huấn luyện
        """
        # Kiểm tra cấu hình tự động tái huấn luyện
        auto_retrain_config = self.config.get("auto_retrain", {})
        enable_auto_retrain = auto_retrain_config.get("enable_auto_retrain", False)
        
        # Ghi log đề xuất
        self.logger.info(f"Đề xuất tái huấn luyện: Độ tin cậy = {recommendation['confidence']:.2%}")
        for reason in recommendation.get("reasons", []):
            self.logger.info(f"- {reason}")
        
        # Kiểm tra xem có đủ dữ liệu để tái huấn luyện không
        min_trades = auto_retrain_config.get("min_trades_for_retrain", 100)
        min_days = auto_retrain_config.get("min_days_for_retrain", 30)
        
        trades_count = len(self.tracking_data["trades"])
        if self.tracking_data["equity_curve"].empty:
            days_count = 0
        else:
            first_date = self.tracking_data["equity_curve"].index[0]
            last_date = self.tracking_data["equity_curve"].index[-1]
            days_count = (last_date - first_date).days
        
        # Kiểm tra cooldown
        retrain_cooldown = auto_retrain_config.get("retrain_cooldown", 604800)  # 7 ngày mặc định
        last_retrain_time = None
        
        for rec in reversed(self.tracking_data["retraining_recommendations"]):
            if rec.get("action_taken") == "retrained":
                last_retrain_time = rec.get("timestamp")
                break
        
        cooldown_ok = True
        if last_retrain_time is not None:
            time_since_last_retrain = (datetime.now() - last_retrain_time).total_seconds()
            cooldown_ok = time_since_last_retrain >= retrain_cooldown
        
        # Đánh dấu khuyến nghị với trạng thái
        recommendation["has_enough_data"] = trades_count >= min_trades and days_count >= min_days
        recommendation["cooldown_ok"] = cooldown_ok
        
        # Quyết định tự động tái huấn luyện
        should_auto_retrain = (
            enable_auto_retrain and 
            recommendation["has_enough_data"] and 
            cooldown_ok and
            recommendation["confidence"] >= self.auto_retrain_threshold
        )
        
        if should_auto_retrain:
            self.logger.info("Bắt đầu quá trình tự động tái huấn luyện...")
            
            # Đánh dấu khuyến nghị
            recommendation["action_taken"] = "auto_retrain_initiated"
            recommendation["action_timestamp"] = datetime.now()
            
            # Gọi callback nếu có
            if self.callback_handler:
                try:
                    self.callback_handler.on_auto_retrain(recommendation)
                except Exception as e:
                    self.logger.error(f"Lỗi khi gọi callback tự động tái huấn luyện: {str(e)}")
        else:
            # Gọi callback thông báo đề xuất
            if self.callback_handler:
                try:
                    self.callback_handler.on_retrain_recommendation(recommendation)
                except Exception as e:
                    self.logger.error(f"Lỗi khi gọi callback đề xuất tái huấn luyện: {str(e)}")
    
    def add_equity_point(self, timestamp: Union[str, datetime], value: float) -> None:
        """
        Thêm điểm dữ liệu vốn vào hệ thống theo dõi.
        
        Args:
            timestamp (Union[str, datetime]): Thời gian
            value (float): Giá trị vốn
        """
        # Đảm bảo timestamp là đối tượng datetime
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Tạo dữ liệu
        data_item = {
            "type": "equity",
            "timestamp": timestamp,
            "value": value
        }
        
        # Thêm vào hàng đợi
        self.data_queue.put(data_item)
        
        # Nếu không đang theo dõi, xử lý ngay
        if not self.is_tracking:
            self._process_equity_data(data_item)
    
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Thêm giao dịch vào hệ thống theo dõi.
        
        Args:
            trade_data (Dict[str, Any]): Dữ liệu giao dịch
        """
        # Tạo dữ liệu
        data_item = {
            "type": "trade",
            "trade": trade_data
        }
        
        # Thêm vào hàng đợi
        self.data_queue.put(data_item)
        
        # Nếu không đang theo dõi, xử lý ngay
        if not self.is_tracking:
            self._process_trade_data(data_item)
    
    def update_model_info(self, version: str, model_info: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Cập nhật thông tin mô hình.
        
        Args:
            version (str): Phiên bản mô hình
            model_info (Dict[str, Any]): Thông tin mô hình
            metrics (Dict[str, Any], optional): Các chỉ số của mô hình
        """
        # Tạo dữ liệu
        data_item = {
            "type": "model_update",
            "version": version,
            "model_info": model_info,
            "metrics": metrics or {}
        }
        
        # Thêm vào hàng đợi
        self.data_queue.put(data_item)
        
        # Nếu không đang theo dõi, xử lý ngay
        if not self.is_tracking:
            self._process_model_update(data_item)
    
    def add_metrics(self, metrics: Dict[str, Any], source: str = "system", timestamp: Optional[datetime] = None) -> None:
        """
        Thêm dữ liệu metrics vào hệ thống theo dõi.
        
        Args:
            metrics (Dict[str, Any]): Dữ liệu metrics
            source (str, optional): Nguồn dữ liệu
            timestamp (datetime, optional): Thời gian
        """
        # Tạo dữ liệu
        data_item = {
            "type": "metrics",
            "metrics": metrics,
            "source": source,
            "timestamp": timestamp or datetime.now()
        }
        
        # Thêm vào hàng đợi
        self.data_queue.put(data_item)
        
        # Nếu không đang theo dõi, xử lý ngay
        if not self.is_tracking:
            self._process_metrics_data(data_item)
    
    def evaluate_current_performance(self) -> Dict[str, Any]:
        """
        Đánh giá hiệu suất hiện tại của hệ thống.
        
        Returns:
            Dict[str, Any]: Kết quả đánh giá
        """
        # Kiểm tra xem có đủ dữ liệu không
        if len(self.tracking_data["equity_curve"]) < 10:
            return {
                "status": "insufficient_data",
                "message": "Không đủ dữ liệu để đánh giá hiệu suất",
                "timestamp": datetime.now()
            }
        
        # Tính toán các chỉ số hiệu suất
        if not self.tracking_data["trades"].empty:
            from backtesting.performance_metrics import PerformanceMetrics
            
            perf_metrics = PerformanceMetrics(
                equity_curve=self.tracking_data["equity_curve"],
                trades=self.tracking_data["trades"]
            )
            
            metrics = perf_metrics.calculate_all_metrics()
        else:
            # Tính toán các chỉ số cơ bản nếu không có dữ liệu giao dịch
            equity = self.tracking_data["equity_curve"]
            returns = equity.pct_change().dropna()
            
            metrics = {
                "total_return": (equity.iloc[-1] / equity.iloc[0]) - 1 if len(equity) > 1 else 0,
                "volatility": returns.std() * np.sqrt(252) if len(returns) > 1 else 0,
                "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0,
                "max_drawdown": self._calculate_max_drawdown(equity)
            }
        
        # Lấy các cảnh báo gần đây
        recent_alerts = [
            alert for alert in self.tracking_data["alerts"]
            if (datetime.now() - alert.get("timestamp", datetime.min)).total_seconds() < 86400  # 24 giờ
        ]
        
        # Lấy đề xuất tái huấn luyện gần đây nhất
        latest_recommendation = None
        if self.tracking_data["retraining_recommendations"]:
            latest_recommendation = self.tracking_data["retraining_recommendations"][-1]
        
        # Đánh giá tổng thể
        overall_status = "stable"
        if latest_recommendation and latest_recommendation.get("should_retrain", False):
            if latest_recommendation.get("confidence", 0) > 0.8:
                overall_status = "critical"
            else:
                overall_status = "warning"
        elif len(recent_alerts) > 3:
            overall_status = "warning"
        
        # Tạo kết quả đánh giá
        evaluation = {
            "status": overall_status,
            "timestamp": datetime.now(),
            "metrics": metrics,
            "recent_alerts_count": len(recent_alerts),
            "recent_alerts": recent_alerts,
            "latest_recommendation": latest_recommendation,
            "data_summary": {
                "equity_points": len(self.tracking_data["equity_curve"]),
                "trades_count": len(self.tracking_data["trades"]),
                "first_date": self.tracking_data["equity_curve"].index[0] if not self.tracking_data["equity_curve"].empty else None,
                "last_date": self.tracking_data["equity_curve"].index[-1] if not self.tracking_data["equity_curve"].empty else None,
                "days_monitored": (self.tracking_data["equity_curve"].index[-1] - self.tracking_data["equity_curve"].index[0]).days 
                                  if len(self.tracking_data["equity_curve"]) > 1 else 0
            },
            "current_model": {
                "version": self.current_model_version,
                "info": self.current_model_info
            }
        }
        
        return evaluation
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """
        Tính toán drawdown tối đa.
        
        Args:
            equity (pd.Series): Chuỗi giá trị vốn
            
        Returns:
            float: Drawdown tối đa
        """
        if len(equity) < 2:
            return 0.0
            
        # Tính peak-to-trough cho mỗi điểm
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        
        return abs(drawdown.min())
    
    def mark_retrain_completed(self, version: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Đánh dấu quá trình tái huấn luyện đã hoàn thành.
        
        Args:
            version (str): Phiên bản mô hình mới
            metrics (Dict[str, Any], optional): Các chỉ số của mô hình mới
        """
        # Đánh dấu đề xuất tái huấn luyện gần nhất là đã hoàn thành
        if self.tracking_data["retraining_recommendations"]:
            latest_rec = self.tracking_data["retraining_recommendations"][-1]
            latest_rec["action_taken"] = "retrained"
            latest_rec["action_timestamp"] = datetime.now()
            latest_rec["new_model_version"] = version
            
            if metrics:
                latest_rec["new_model_metrics"] = metrics
        
        # Cập nhật thông tin mô hình hiện tại
        self.update_model_info(version=version, model_info={"retrained": True}, metrics=metrics)
        
        self.logger.info(f"Đã đánh dấu tái huấn luyện hoàn thành với phiên bản mới: {version}")
        
        # Lưu dữ liệu
        self._save_tracking_data()
    
    def get_retraining_data(self, days_back: int = 90) -> Dict[str, Any]:
        """
        Lấy dữ liệu cho quá trình tái huấn luyện.
        
        Args:
            days_back (int, optional): Số ngày dữ liệu cần lấy
            
        Returns:
            Dict[str, Any]: Dữ liệu cho tái huấn luyện
        """
        # Kiểm tra xem có đủ dữ liệu không
        if self.tracking_data["equity_curve"].empty:
            return {
                "status": "insufficient_data",
                "message": "Không có dữ liệu vốn để tái huấn luyện"
            }
        
        # Tính thời điểm bắt đầu
        end_date = self.tracking_data["equity_curve"].index[-1]
        start_date = end_date - timedelta(days=days_back)
        
        # Lọc dữ liệu
        equity_subset = self.tracking_data["equity_curve"][
            (self.tracking_data["equity_curve"].index >= start_date) &
            (self.tracking_data["equity_curve"].index <= end_date)
        ]
        
        trades_subset = pd.DataFrame()
        if not self.tracking_data["trades"].empty and 'exit_time' in self.tracking_data["trades"].columns:
            trades_subset = self.tracking_data["trades"][
                (self.tracking_data["trades"]['exit_time'] >= start_date) &
                (self.tracking_data["trades"]['exit_time'] <= end_date)
            ]
        
        # Tính các thống kê
        equity_stats = {}
        if not equity_subset.empty:
            returns = equity_subset.pct_change().dropna()
            equity_stats = {
                "total_return": (equity_subset.iloc[-1] / equity_subset.iloc[0]) - 1 if len(equity_subset) > 1 else 0,
                "volatility": returns.std() * np.sqrt(252) if len(returns) > 1 else 0,
                "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0,
                "max_drawdown": self._calculate_max_drawdown(equity_subset)
            }
        
        trades_stats = {}
        if not trades_subset.empty and 'profit' in trades_subset.columns:
            win_rate = (trades_subset['profit'] > 0).mean() if len(trades_subset) > 0 else 0
            avg_profit = trades_subset[trades_subset['profit'] > 0]['profit'].mean() if len(trades_subset[trades_subset['profit'] > 0]) > 0 else 0
            avg_loss = abs(trades_subset[trades_subset['profit'] < 0]['profit'].mean()) if len(trades_subset[trades_subset['profit'] < 0]) > 0 else 0
            
            trades_stats = {
                "win_rate": win_rate,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "profit_factor": avg_profit / avg_loss if avg_loss > 0 else float('inf'),
                "expectancy": win_rate * avg_profit - (1 - win_rate) * avg_loss
            }
        
        # Tạo kết quả
        result = {
            "status": "success",
            "start_date": start_date,
            "end_date": end_date,
            "days": days_back,
            "equity_points": len(equity_subset),
            "trades_count": len(trades_subset),
            "equity_stats": equity_stats,
            "trades_stats": trades_stats,
            "equity_curve": equity_subset.to_dict(),
            "trades": trades_subset.to_dict(orient="records") if not trades_subset.empty else []
        }
        
        return result
    
    def plot_performance_overview(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Tạo biểu đồ tổng quan về hiệu suất.
        
        Args:
            figsize (Tuple[int, int], optional): Kích thước hình
            
        Returns:
            plt.Figure: Đối tượng Figure của matplotlib
        """
        # Kiểm tra xem có đủ dữ liệu không
        if self.tracking_data["equity_curve"].empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Không có dữ liệu để tạo biểu đồ", 
                 horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Tạo hình
        fig, axs = plt.subplots(3, 2, figsize=figsize)
        
        # 1. Đường cong vốn
        ax1 = axs[0, 0]
        self.tracking_data["equity_curve"].plot(ax=ax1, color='blue', linewidth=2)
        ax1.set_title('Đường cong vốn')
        ax1.set_ylabel('Giá trị vốn')
        ax1.grid(True, alpha=0.3)
        
        # Thêm các điểm đánh dấu tái huấn luyện
        retrain_dates = []
        for rec in self.tracking_data["retraining_recommendations"]:
            if rec.get("action_taken") == "retrained" and "action_timestamp" in rec:
                timestamp = rec["action_timestamp"]
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                
                # Tìm điểm gần nhất trong equity_curve
                closest_idx = self.tracking_data["equity_curve"].index.get_indexer([timestamp], method='nearest')[0]
                if closest_idx >= 0 and closest_idx < len(self.tracking_data["equity_curve"]):
                    retrain_dates.append((
                        self.tracking_data["equity_curve"].index[closest_idx], 
                        self.tracking_data["equity_curve"].iloc[closest_idx],
                        rec.get("new_model_version", "Unknown")
                    ))
        
        for date, value, version in retrain_dates:
            ax1.scatter(date, value, color='red', s=100, zorder=5)
            ax1.annotate(f"Retrain: {version}", (date, value), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                       fontsize=8)
        
        # 2. Returns
        ax2 = axs[0, 1]
        returns = self.tracking_data["equity_curve"].pct_change().dropna()
        returns.plot(ax=ax2, color='green', alpha=0.6)
        ax2.set_title('Returns')
        ax2.set_ylabel('Return')
        ax2.grid(True, alpha=0.3)
        
        # Thêm đường trung bình trượt
        if len(returns) > 30:
            returns.rolling(window=30).mean().plot(ax=ax2, color='red', linewidth=2)
            ax2.legend(['Daily Return', '30-day MA'])
        
        # 3. Drawdown
        ax3 = axs[1, 0]
        if len(self.tracking_data["equity_curve"]) > 1:
            rolling_max = self.tracking_data["equity_curve"].cummax()
            drawdown = ((self.tracking_data["equity_curve"] - rolling_max) / rolling_max)
            drawdown.plot(ax=ax3, color='red', alpha=0.6)
            ax3.set_title('Drawdown')
            ax3.set_ylabel('Drawdown')
            ax3.grid(True, alpha=0.3)
            
            # Thêm ngưỡng cảnh báo
            if "drawdown" in self.alert_thresholds:
                ax3.axhline(y=-self.alert_thresholds["drawdown"], color='red', linestyle='--', alpha=0.7)
        
        # 4. Giao dịch thắng/thua theo thời gian
        ax4 = axs[1, 1]
        if not self.tracking_data["trades"].empty and 'profit' in self.tracking_data["trades"].columns:
            # Chuyển đổi thành chuỗi thời gian
            if 'exit_time' in self.tracking_data["trades"].columns:
                trade_times = self.tracking_data["trades"]['exit_time']
                profits = self.tracking_data["trades"]['profit']
                
                # Tạo Series theo thời gian
                if not isinstance(trade_times.iloc[0], (datetime, pd.Timestamp)):
                    # Chuyển đổi nếu không phải datetime
                    trade_times = pd.to_datetime(trade_times)
                
                trade_series = pd.Series(profits.values, index=trade_times)
                
                # Vẽ lợi nhuận giao dịch
                ax4.scatter(trade_series.index, trade_series.values, 
                         c=trade_series.values > 0, cmap='RdYlGn', alpha=0.6)
                ax4.set_title('Kết quả giao dịch')
                ax4.set_ylabel('Profit/Loss')
                ax4.grid(True, alpha=0.3)
                
                # Thêm đường trung bình trượt
                if len(trade_series) > 10:
                    trade_series.rolling(window=10).mean().plot(ax=ax4, color='black', linewidth=2)
        
        # 5. Cảnh báo
        ax5 = axs[2, 0]
        ax5.axis('off')  # Tắt trục
        
        # Tạo bảng thông tin cảnh báo
        alert_text = "CÁC CẢNH BÁO GẦN ĐÂY:\n\n"
        
        recent_alerts = sorted(
            [a for a in self.tracking_data["alerts"] 
             if (datetime.now() - a.get("timestamp", datetime.min)).total_seconds() < 7*86400],  # 7 ngày
            key=lambda x: x.get("timestamp", datetime.min),
            reverse=True
        )
        
        if recent_alerts:
            for i, alert in enumerate(recent_alerts[:5]):  # Hiển thị 5 cảnh báo gần nhất
                timestamp = alert.get("timestamp", datetime.now())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                
                severity = alert.get("severity", "medium").upper()
                severity_color = {
                    "LOW": "blue",
                    "MEDIUM": "orange",
                    "HIGH": "red"
                }.get(severity, "black")
                
                alert_text += f"{i+1}. [{timestamp.strftime('%Y-%m-%d %H:%M')}] "
                alert_text += f"[{severity}] {alert.get('message', 'Không có mô tả')}\n"
            
            if len(recent_alerts) > 5:
                alert_text += f"\n... và {len(recent_alerts) - 5} cảnh báo khác\n"
        else:
            alert_text += "Không có cảnh báo nào trong 7 ngày qua.\n"
        
        ax5.text(0.05, 0.95, alert_text, transform=ax5.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 6. Thông tin đề xuất tái huấn luyện
        ax6 = axs[2, 1]
        ax6.axis('off')  # Tắt trục
        
        # Tạo bảng thông tin đề xuất
        if self.tracking_data["retraining_recommendations"]:
            latest_rec = self.tracking_data["retraining_recommendations"][-1]
            
            # Tính số ngày từ đề xuất gần nhất
            rec_time = latest_rec.get("timestamp", datetime.now())
            if isinstance(rec_time, str):
                rec_time = datetime.fromisoformat(rec_time)
            
            days_ago = (datetime.now() - rec_time).days
            
            rec_text = "ĐỀ XUẤT TÁI HUẤN LUYỆN:\n\n"
            rec_text += f"Đề xuất gần nhất: {rec_time.strftime('%Y-%m-%d %H:%M')} ({days_ago} ngày trước)\n"
            rec_text += f"Cần tái huấn luyện: {'CÓ' if latest_rec.get('should_retrain', False) else 'KHÔNG'}\n"
            rec_text += f"Độ tin cậy: {latest_rec.get('confidence', 0):.2%}\n\n"
            
            rec_text += "LÝ DO:\n"
            for reason in latest_rec.get("reasons", []):
                rec_text += f"- {reason}\n"
            
            # Thông tin hành động
            if latest_rec.get("action_taken"):
                action = latest_rec.get("action_taken")
                action_time = latest_rec.get("action_timestamp")
                if isinstance(action_time, str):
                    action_time = datetime.fromisoformat(action_time)
                
                rec_text += f"\nHÀNH ĐỘNG: {action} ({action_time.strftime('%Y-%m-%d')})\n"
                
                if action == "retrained" and "new_model_version" in latest_rec:
                    rec_text += f"Phiên bản mới: {latest_rec['new_model_version']}\n"
        else:
            rec_text = "ĐỀ XUẤT TÁI HUẤN LUYỆN:\n\nChưa có đề xuất nào.\n"
        
        ax6.text(0.05, 0.95, rec_text, transform=ax6.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.suptitle(f"Tổng quan hiệu suất hệ thống: {self.system_id}", fontsize=16)
        plt.subplots_adjust(top=0.92)
        
        return fig