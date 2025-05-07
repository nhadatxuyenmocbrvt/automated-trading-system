"""
Theo dõi hiệu suất cho tái huấn luyện agent.
File này cung cấp các công cụ để theo dõi, phân tích và đánh giá
hiệu suất của agent trong thời gian thực, phát hiện sự xuống cấp hiệu suất,
và đưa ra quyết định khi nào nên tái huấn luyện agent.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import concurrent.futures
from functools import partial
import time

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import BacktestMetric, RetrainingTrigger
from config.system_config import get_system_config

# Import các module liên quan
try:
    from logs.metrics.trading_metrics import TradingMetricsTracker
    TRADING_METRICS_AVAILABLE = True
except ImportError:
    TRADING_METRICS_AVAILABLE = False

try:
    from backtesting.evaluation.performance_evaluator import PerformanceEvaluator
    PERFORMANCE_EVALUATOR_AVAILABLE = True
except ImportError:
    PERFORMANCE_EVALUATOR_AVAILABLE = False


class PerformanceTracker:
    """
    Theo dõi hiệu suất của agent và xác định khi nào cần tái huấn luyện.
    
    Lớp này theo dõi hiệu suất của agent trong môi trường thực, phát hiện sự xuống cấp
    hiệu suất theo thời gian, và đưa ra khuyến nghị về việc tái huấn luyện dựa trên
    các ngưỡng và điều kiện đã được cấu hình.
    """
    
    def __init__(
        self,
        agent_name: str,
        model_version: str,
        strategy_name: str,
        symbols: List[str],
        tracking_metrics: Optional[List[str]] = None,
        retraining_dir: Optional[Union[str, Path]] = None,
        baseline_metrics_path: Optional[Union[str, Path]] = None,
        evaluation_window: int = 14,  # Số ngày để đánh giá hiệu suất
        retraining_triggers: Optional[Dict[str, Dict[str, float]]] = None,
        update_frequency: int = 24,  # Số giờ giữa các lần cập nhật
        min_trades_for_evaluation: int = 30,  # Số giao dịch tối thiểu để đánh giá
        auto_save: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo PerformanceTracker.
        
        Args:
            agent_name: Tên của agent
            model_version: Phiên bản mô hình
            strategy_name: Tên chiến lược giao dịch
            symbols: Danh sách các cặp tiền được giao dịch
            tracking_metrics: Các số liệu cần theo dõi (None để sử dụng danh sách mặc định)
            retraining_dir: Thư mục lưu dữ liệu theo dõi hiệu suất và quyết định tái huấn luyện
            baseline_metrics_path: Đường dẫn đến file số liệu baseline
            evaluation_window: Số ngày dữ liệu để đánh giá hiệu suất
            retraining_triggers: Dict chứa các ngưỡng kích hoạt tái huấn luyện
            update_frequency: Số giờ giữa các lần cập nhật
            min_trades_for_evaluation: Số giao dịch tối thiểu để đánh giá
            auto_save: Tự động lưu trạng thái theo dõi
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("performance_tracker")
        
        # Thiết lập các thông tin cơ bản
        self.agent_name = agent_name
        self.model_version = model_version
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.evaluation_window = evaluation_window
        self.update_frequency = update_frequency
        self.min_trades_for_evaluation = min_trades_for_evaluation
        self.auto_save = auto_save
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập danh sách các số liệu theo dõi mặc định nếu không được cung cấp
        if tracking_metrics is None:
            self.tracking_metrics = [
                BacktestMetric.SHARPE_RATIO.value,
                BacktestMetric.SORTINO_RATIO.value,
                BacktestMetric.WIN_RATE.value,
                BacktestMetric.PROFIT_FACTOR.value,
                BacktestMetric.MAX_DRAWDOWN.value,
                BacktestMetric.EXPECTANCY.value,
                "roi",
                "volatility"
            ]
        else:
            self.tracking_metrics = tracking_metrics
            
        # Thiết lập thư mục lưu trữ
        if retraining_dir is None:
            retraining_dir = Path(self.system_config.get("retraining_dir", "./retraining"))
        self.retraining_dir = Path(retraining_dir) / f"{agent_name}_{model_version}"
        self.retraining_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập các ngưỡng kích hoạt tái huấn luyện
        # Nếu không có, sử dụng giá trị mặc định
        default_triggers = {
            RetrainingTrigger.DRAWDOWN.value: {
                "threshold": 0.1,  # Kích hoạt nếu max drawdown vượt quá 10%
                "period": 7,       # Trong vòng 7 ngày
                "weight": 0.3      # Trọng số khi tính tổng hợp
            },
            RetrainingTrigger.PROFIT_DECLINE.value: {
                "threshold": 0.15,  # Kích hoạt nếu lợi nhuận giảm 15%
                "period": 14,       # So với 14 ngày trước
                "weight": 0.25      # Trọng số khi tính tổng hợp
            },
            RetrainingTrigger.WIN_RATE_DECLINE.value: {
                "threshold": 0.1,   # Kích hoạt nếu win rate giảm 10%
                "period": 14,       # So với 14 ngày trước
                "weight": 0.2       # Trọng số khi tính tổng hợp
            },
            RetrainingTrigger.TRADE_FREQUENCY_DECLINE.value: {
                "threshold": 0.3,   # Kích hoạt nếu tần suất giao dịch giảm 30%
                "period": 7,        # So với 7 ngày trước
                "weight": 0.15      # Trọng số khi tính tổng hợp
            },
            RetrainingTrigger.MARKET_REGIME_CHANGE.value: {
                "threshold": 0.25,  # Kích hoạt nếu chỉ số thay đổi thị trường vượt quá 25%
                "period": 5,        # Trong vòng 5 ngày
                "weight": 0.1       # Trọng số khi tính tổng hợp
            }
        }
        
        self.retraining_triggers = retraining_triggers or default_triggers
        
        # Thiết lập path cho số liệu baseline
        self.baseline_metrics_path = baseline_metrics_path
        self.baseline_metrics = self._load_baseline_metrics()
        
        # Khởi tạo các biến theo dõi
        self.performance_history = []
        self.retraining_recommendations = []
        self.last_update_time = None
        self.current_sequence_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Theo dõi hiệu suất theo từng symbol
        self.symbol_performance = {symbol: [] for symbol in symbols}
        
        # Kiểm tra sự sẵn có của các module phụ thuộc
        if not TRADING_METRICS_AVAILABLE:
            self.logger.warning("Module TradingMetricsTracker không khả dụng. Một số tính năng sẽ bị giới hạn.")
        
        if not PERFORMANCE_EVALUATOR_AVAILABLE:
            self.logger.warning("Module PerformanceEvaluator không khả dụng. Một số tính năng sẽ bị giới hạn.")
        
        # Tải lịch sử hiệu suất nếu có
        self._load_history()
        
        self.logger.info(f"Đã khởi tạo PerformanceTracker cho {agent_name} phiên bản {model_version}")
    
    def _load_baseline_metrics(self) -> Dict[str, Any]:
        """
        Tải số liệu baseline từ file.
        
        Returns:
            Dict chứa số liệu baseline
        """
        if not self.baseline_metrics_path or not Path(self.baseline_metrics_path).exists():
            self.logger.warning("Không tìm thấy file số liệu baseline. Sẽ sử dụng giá trị mặc định.")
            # Tạo baseline mặc định
            baseline = {}
            for metric in self.tracking_metrics:
                # Giá trị mặc định cho từng loại metrics
                if metric == BacktestMetric.WIN_RATE.value:
                    baseline[metric] = 0.5
                elif metric == BacktestMetric.PROFIT_FACTOR.value:
                    baseline[metric] = 1.2
                elif metric == BacktestMetric.SHARPE_RATIO.value:
                    baseline[metric] = 1.0
                elif metric == BacktestMetric.SORTINO_RATIO.value:
                    baseline[metric] = 1.2
                elif metric == BacktestMetric.MAX_DRAWDOWN.value:
                    baseline[metric] = 0.15
                elif metric == BacktestMetric.EXPECTANCY.value:
                    baseline[metric] = 0.1
                elif metric == "roi":
                    baseline[metric] = 0.05
                elif metric == "volatility":
                    baseline[metric] = 0.2
                else:
                    baseline[metric] = 0.0
            return baseline
        
        try:
            with open(self.baseline_metrics_path, 'r', encoding='utf-8') as f:
                baseline = json.load(f)
            
            self.logger.info(f"Đã tải số liệu baseline từ {self.baseline_metrics_path}")
            return baseline
        
        except Exception as e:
            self.logger.error(f"Lỗi khi tải số liệu baseline: {e}")
            return {}
    
    def _load_history(self) -> None:
        """
        Tải lịch sử hiệu suất và quyết định tái huấn luyện từ file.
        """
        history_path = self.retraining_dir / f"{self.agent_name}_{self.model_version}_history.json"
        
        if not history_path.exists():
            self.logger.info("Không tìm thấy file lịch sử hiệu suất. Sẽ tạo mới.")
            return
        
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.performance_history = data.get("performance_history", [])
            self.retraining_recommendations = data.get("retraining_recommendations", [])
            
            # Lấy thời gian cập nhật cuối cùng
            if self.performance_history:
                last_performance = self.performance_history[-1]
                self.last_update_time = datetime.fromisoformat(last_performance.get("timestamp", datetime.now().isoformat()))
            
            # Tải hiệu suất theo symbol
            symbol_perf = data.get("symbol_performance", {})
            for symbol in self.symbols:
                if symbol in symbol_perf:
                    self.symbol_performance[symbol] = symbol_perf[symbol]
            
            self.logger.info(f"Đã tải lịch sử hiệu suất với {len(self.performance_history)} bản ghi")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải lịch sử hiệu suất: {e}")
    
    def save_history(self) -> None:
        """
        Lưu lịch sử hiệu suất và quyết định tái huấn luyện vào file.
        """
        history_path = self.retraining_dir / f"{self.agent_name}_{self.model_version}_history.json"
        
        try:
            data = {
                "agent_name": self.agent_name,
                "model_version": self.model_version,
                "strategy_name": self.strategy_name,
                "symbols": self.symbols,
                "performance_history": self.performance_history,
                "retraining_recommendations": self.retraining_recommendations,
                "symbol_performance": self.symbol_performance,
                "last_update": datetime.now().isoformat()
            }
            
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Đã lưu lịch sử hiệu suất vào {history_path}")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu lịch sử hiệu suất: {e}")
    
    def update_performance(
        self,
        metrics_data: Optional[Dict[str, Dict[str, float]]] = None,
        metrics_files: Optional[Dict[str, str]] = None,
        update_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Cập nhật hiệu suất với dữ liệu mới.
        
        Args:
            metrics_data: Dict chứa dữ liệu số liệu mới, ánh xạ symbol -> metrics
            metrics_files: Dict chứa đường dẫn đến file số liệu, ánh xạ symbol -> file path
            update_timestamp: Thời gian cập nhật (None để sử dụng thời gian hiện tại)
            
        Returns:
            Dict chứa thông tin cập nhật
        """
        if update_timestamp is None:
            update_timestamp = datetime.now()
        
        # Kiểm tra xem có cần cập nhật không
        if self.last_update_time and (update_timestamp - self.last_update_time).total_seconds() < self.update_frequency * 3600:
            self.logger.info(f"Chưa đến thời điểm cập nhật tiếp theo. Bỏ qua cập nhật hiệu suất.")
            return {
                "status": "skipped",
                "message": "Chưa đến thời điểm cập nhật tiếp theo",
                "last_update": self.last_update_time.isoformat(),
                "next_update": (self.last_update_time + timedelta(hours=self.update_frequency)).isoformat()
            }
        
        # Cập nhật dữ liệu từ file nếu được cung cấp
        if metrics_files:
            metrics_data = metrics_data or {}
            for symbol, file_path in metrics_files.items():
                if symbol in self.symbols:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            symbol_data = json.load(f)
                        
                        # Lấy metrics cần thiết
                        symbol_metrics = {}
                        if "performance_stats" in symbol_data:
                            # Cấu trúc từ TradingMetricsTracker
                            for metric in self.tracking_metrics:
                                if metric in symbol_data["performance_stats"]:
                                    symbol_metrics[metric] = symbol_data["performance_stats"][metric]
                        elif "metrics" in symbol_data:
                            # Cấu trúc từ PerformanceEvaluator
                            for metric in self.tracking_metrics:
                                if metric in symbol_data["metrics"]:
                                    symbol_metrics[metric] = symbol_data["metrics"][metric]
                        
                        metrics_data[symbol] = symbol_metrics
                        
                    except Exception as e:
                        self.logger.error(f"Lỗi khi đọc file metrics cho {symbol}: {e}")
        
        # Nếu không có dữ liệu metrics, thử lấy từ TradingMetricsTracker
        if not metrics_data and TRADING_METRICS_AVAILABLE:
            metrics_data = {}
            for symbol in self.symbols:
                try:
                    # Tạo đường dẫn file metrics mặc định
                    symbol_metrics_path = self.system_config.get("log_dir") / "trading" / symbol.replace('/', '_') / f"{symbol.replace('/', '_')}_{update_timestamp.strftime('%Y%m%d')}_metrics.json"
                    
                    if Path(symbol_metrics_path).exists():
                        with open(symbol_metrics_path, 'r', encoding='utf-8') as f:
                            symbol_data = json.load(f)
                        
                        symbol_metrics = {}
                        if "performance_stats" in symbol_data:
                            for metric in self.tracking_metrics:
                                if metric in symbol_data["performance_stats"]:
                                    symbol_metrics[metric] = symbol_data["performance_stats"][metric]
                        
                        metrics_data[symbol] = symbol_metrics
                
                except Exception as e:
                    self.logger.warning(f"Không thể lấy metrics tự động cho {symbol}: {e}")
        
        # Nếu vẫn không có dữ liệu, đưa ra cảnh báo
        if not metrics_data:
            self.logger.warning("Không có dữ liệu metrics nào để cập nhật hiệu suất")
            return {
                "status": "error",
                "message": "Không có dữ liệu metrics nào để cập nhật hiệu suất",
                "timestamp": update_timestamp.isoformat()
            }
        
        # Tính toán hiệu suất tổng hợp
        combined_metrics = self._combine_symbol_metrics(metrics_data)
        
        # Lưu hiệu suất mới
        performance_entry = {
            "timestamp": update_timestamp.isoformat(),
            "metrics": combined_metrics,
            "symbol_metrics": metrics_data
        }
        
        self.performance_history.append(performance_entry)
        
        # Cập nhật hiệu suất theo symbol
        for symbol, metrics in metrics_data.items():
            if symbol in self.symbol_performance:
                self.symbol_performance[symbol].append({
                    "timestamp": update_timestamp.isoformat(),
                    "metrics": metrics
                })
        
        # Cập nhật thời gian cập nhật cuối
        self.last_update_time = update_timestamp
        
        # Kiểm tra các điều kiện kích hoạt tái huấn luyện
        retraining_needed, triggers = self._check_retraining_triggers()
        
        # Lưu khuyến nghị tái huấn luyện nếu cần
        if retraining_needed:
            recommendation = {
                "timestamp": update_timestamp.isoformat(),
                "recommendation": "retrain",
                "triggers": triggers,
                "sequence_id": self.current_sequence_id,
                "metrics": combined_metrics
            }
            
            self.retraining_recommendations.append(recommendation)
            self.logger.info(f"Đề xuất tái huấn luyện với {len(triggers)} yếu tố kích hoạt")
        
        # Tự động lưu lịch sử nếu được cấu hình
        if self.auto_save:
            self.save_history()
        
        return {
            "status": "success",
            "timestamp": update_timestamp.isoformat(),
            "metrics": combined_metrics,
            "retraining_needed": retraining_needed,
            "triggers": triggers if retraining_needed else []
        }
    
    def _combine_symbol_metrics(self, symbol_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Tính toán hiệu suất tổng hợp từ số liệu của các symbol riêng lẻ.
        
        Args:
            symbol_metrics: Dict chứa số liệu từng symbol
            
        Returns:
            Dict chứa số liệu tổng hợp
        """
        if not symbol_metrics:
            return {}
        
        # Tính toán số liệu tổng hợp
        combined = {}
        symbol_count = len(symbol_metrics)
        
        for metric in self.tracking_metrics:
            values = []
            
            for symbol, metrics in symbol_metrics.items():
                if metric in metrics:
                    values.append(metrics[metric])
            
            if values:
                # Tùy theo loại metrics mà có phương pháp tổng hợp khác nhau
                if metric == BacktestMetric.MAX_DRAWDOWN.value:
                    # Lấy max drawdown lớn nhất giữa các symbol
                    combined[metric] = max(values)
                    
                elif metric in [BacktestMetric.SHARPE_RATIO.value, BacktestMetric.SORTINO_RATIO.value, 
                              BacktestMetric.PROFIT_FACTOR.value, BacktestMetric.EXPECTANCY.value]:
                    # Lấy giá trị trung bình cho các tỷ lệ
                    combined[metric] = sum(values) / len(values)
                    
                elif metric == BacktestMetric.WIN_RATE.value:
                    # Tính trung bình win rate
                    combined[metric] = sum(values) / len(values)
                    
                elif metric == "roi":
                    # Tính trung bình ROI
                    combined[metric] = sum(values) / len(values)
                    
                elif metric == "volatility":
                    # Tính trung bình volatility
                    combined[metric] = sum(values) / len(values)
                    
                else:
                    # Mặc định lấy giá trị trung bình
                    combined[metric] = sum(values) / len(values)
        
        return combined
    
    def _check_retraining_triggers(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Kiểm tra các điều kiện kích hoạt tái huấn luyện.
        
        Returns:
            Tuple (retraining_needed, activated_triggers)
        """
        if len(self.performance_history) < 2:
            return False, []
        
        # Lấy số liệu hiệu suất hiện tại
        current_performance = self.performance_history[-1]
        current_metrics = current_performance.get("metrics", {})
        current_time = datetime.fromisoformat(current_performance.get("timestamp"))
        
        activated_triggers = []
        
        # Kiểm tra từng điều kiện kích hoạt
        # 1. Drawdown trigger
        if RetrainingTrigger.DRAWDOWN.value in self.retraining_triggers:
            trigger_config = self.retraining_triggers[RetrainingTrigger.DRAWDOWN.value]
            threshold = trigger_config.get("threshold", 0.1)
            
            if BacktestMetric.MAX_DRAWDOWN.value in current_metrics:
                current_drawdown = current_metrics[BacktestMetric.MAX_DRAWDOWN.value]
                
                if current_drawdown > threshold:
                    activated_triggers.append({
                        "trigger": RetrainingTrigger.DRAWDOWN.value,
                        "threshold": threshold,
                        "current_value": current_drawdown,
                        "weight": trigger_config.get("weight", 0.3)
                    })
        
        # 2. Profit decline trigger
        if RetrainingTrigger.PROFIT_DECLINE.value in self.retraining_triggers:
            trigger_config = self.retraining_triggers[RetrainingTrigger.PROFIT_DECLINE.value]
            threshold = trigger_config.get("threshold", 0.15)
            period = trigger_config.get("period", 14)
            
            # Tìm số liệu trước đó
            previous_performance = None
            for perf in reversed(self.performance_history[:-1]):
                perf_time = datetime.fromisoformat(perf.get("timestamp"))
                if (current_time - perf_time).days >= period:
                    previous_performance = perf
                    break
            
            if previous_performance and "roi" in current_metrics:
                previous_metrics = previous_performance.get("metrics", {})
                if "roi" in previous_metrics and previous_metrics["roi"] > 0:
                    current_roi = current_metrics["roi"]
                    previous_roi = previous_metrics["roi"]
                    
                    roi_decline = (previous_roi - current_roi) / previous_roi
                    
                    if roi_decline > threshold:
                        activated_triggers.append({
                            "trigger": RetrainingTrigger.PROFIT_DECLINE.value,
                            "threshold": threshold,
                            "current_value": roi_decline,
                            "previous_roi": previous_roi,
                            "current_roi": current_roi,
                            "weight": trigger_config.get("weight", 0.25)
                        })
        
        # 3. Win rate decline trigger
        if RetrainingTrigger.WIN_RATE_DECLINE.value in self.retraining_triggers:
            trigger_config = self.retraining_triggers[RetrainingTrigger.WIN_RATE_DECLINE.value]
            threshold = trigger_config.get("threshold", 0.1)
            period = trigger_config.get("period", 14)
            
            # Tìm số liệu trước đó
            previous_performance = None
            for perf in reversed(self.performance_history[:-1]):
                perf_time = datetime.fromisoformat(perf.get("timestamp"))
                if (current_time - perf_time).days >= period:
                    previous_performance = perf
                    break
            
            if previous_performance and BacktestMetric.WIN_RATE.value in current_metrics:
                previous_metrics = previous_performance.get("metrics", {})
                if BacktestMetric.WIN_RATE.value in previous_metrics:
                    current_win_rate = current_metrics[BacktestMetric.WIN_RATE.value]
                    previous_win_rate = previous_metrics[BacktestMetric.WIN_RATE.value]
                    
                    win_rate_decline = previous_win_rate - current_win_rate
                    
                    if win_rate_decline > threshold:
                        activated_triggers.append({
                            "trigger": RetrainingTrigger.WIN_RATE_DECLINE.value,
                            "threshold": threshold,
                            "current_value": win_rate_decline,
                            "previous_win_rate": previous_win_rate,
                            "current_win_rate": current_win_rate,
                            "weight": trigger_config.get("weight", 0.2)
                        })
        
        # 4. Trade frequency decline
        if RetrainingTrigger.TRADE_FREQUENCY_DECLINE.value in self.retraining_triggers:
            # Bổ sung triển khai điều kiện này khi có thêm dữ liệu về tần suất giao dịch
            pass
        
        # 5. Market regime change
        if RetrainingTrigger.MARKET_REGIME_CHANGE.value in self.retraining_triggers:
            # Bổ sung triển khai điều kiện này khi có thêm dữ liệu về chế độ thị trường
            pass
        
        # Quyết định xem có nên tái huấn luyện hay không
        # Nếu có ít nhất một điều kiện kích hoạt, hoặc tổng trọng số vượt ngưỡng
        if activated_triggers:
            total_weight = sum(trigger.get("weight", 0) for trigger in activated_triggers)
            
            # Nếu tổng trọng số vượt quá 0.5, đề xuất tái huấn luyện
            return total_weight > 0.5 or len(activated_triggers) >= 2, activated_triggers
        
        return False, []
    
    def get_latest_performance(self) -> Dict[str, Any]:
        """
        Lấy số liệu hiệu suất mới nhất.
        
        Returns:
            Dict chứa số liệu hiệu suất mới nhất
        """
        if not self.performance_history:
            return {
                "status": "no_data",
                "message": "Chưa có dữ liệu hiệu suất nào được ghi nhận"
            }
        
        latest = self.performance_history[-1].copy()
        
        # Thêm đánh giá so với baseline
        baseline_comparison = self._compare_with_baseline(latest.get("metrics", {}))
        latest["baseline_comparison"] = baseline_comparison
        
        # Thêm thông tin về xu hướng
        trends = self._analyze_performance_trends()
        latest["trends"] = trends
        
        return latest
    
    def _compare_with_baseline(self, current_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        So sánh số liệu hiện tại với baseline.
        
        Args:
            current_metrics: Dict chứa số liệu hiện tại
            
        Returns:
            Dict chứa kết quả so sánh
        """
        comparison = {}
        
        for metric in self.tracking_metrics:
            if metric in current_metrics and metric in self.baseline_metrics:
                current = current_metrics[metric]
                baseline = self.baseline_metrics[metric]
                
                # Tính toán chênh lệch
                if baseline != 0:
                    relative_diff = (current - baseline) / abs(baseline)
                else:
                    relative_diff = float('inf') if current > 0 else float('-inf') if current < 0 else 0
                
                # Xác định liệu chênh lệch có tích cực hay không
                # Với một số metrics, giá trị cao hơn là tốt hơn
                if metric in [BacktestMetric.MAX_DRAWDOWN.value, "volatility"]:
                    # Với drawdown và volatility, thấp hơn là tốt hơn
                    is_better = current < baseline
                else:
                    # Với các metrics khác, cao hơn là tốt hơn
                    is_better = current > baseline
                
                comparison[metric] = {
                    "current": current,
                    "baseline": baseline,
                    "absolute_diff": current - baseline,
                    "relative_diff": relative_diff,
                    "is_better": is_better
                }
        
        return comparison
    
    def _analyze_performance_trends(self, window: int = 7) -> Dict[str, Dict[str, float]]:
        """
        Phân tích xu hướng hiệu suất theo thời gian.
        
        Args:
            window: Số ngày để phân tích xu hướng
            
        Returns:
            Dict chứa xu hướng của các metrics
        """
        if len(self.performance_history) < 2:
            return {}
        
        latest_timestamp = datetime.fromisoformat(self.performance_history[-1]["timestamp"])
        
        # Lọc dữ liệu trong khoảng thời gian window
        window_data = []
        for entry in reversed(self.performance_history):
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if (latest_timestamp - entry_time).days <= window:
                window_data.append(entry)
        
        if len(window_data) < 2:
            return {}
        
        # Đảo ngược để có thứ tự tăng dần theo thời gian
        window_data.reverse()
        
        # Phân tích xu hướng cho từng metric
        trends = {}
        
        for metric in self.tracking_metrics:
            values = [entry.get("metrics", {}).get(metric) for entry in window_data if metric in entry.get("metrics", {})]
            
            if len(values) >= 2:
                # Tính xu hướng tuyến tính
                x = np.arange(len(values))
                y = np.array(values)
                valid_indices = ~np.isnan(y)
                
                if np.sum(valid_indices) >= 2:
                    x_valid = x[valid_indices]
                    y_valid = y[valid_indices]
                    
                    try:
                        slope, intercept = np.polyfit(x_valid, y_valid, 1)
                        
                        # Tính biến động
                        std = np.std(y_valid)
                        mean = np.mean(y_valid)
                        cv = std / mean if mean != 0 else float('inf')
                        
                        # Xác định xu hướng
                        if metric in [BacktestMetric.MAX_DRAWDOWN.value, "volatility"]:
                            # Với drawdown và volatility, giảm là tốt
                            trend_direction = "improving" if slope < 0 else "deteriorating" if slope > 0 else "stable"
                        else:
                            # Với các metrics khác, tăng là tốt
                            trend_direction = "improving" if slope > 0 else "deteriorating" if slope < 0 else "stable"
                        
                        trends[metric] = {
                            "slope": slope,
                            "direction": trend_direction,
                            "volatility": cv,
                            "start_value": values[0],
                            "end_value": values[-1],
                            "change": values[-1] - values[0],
                            "percent_change": (values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else float('inf')
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Lỗi khi phân tích xu hướng cho {metric}: {e}")
        
        return trends
    
    def get_retraining_recommendation(self) -> Dict[str, Any]:
        """
        Lấy khuyến nghị tái huấn luyện mới nhất.
        
        Returns:
            Dict chứa khuyến nghị tái huấn luyện và thông tin liên quan
        """
        if not self.retraining_recommendations:
            return {
                "status": "no_recommendation",
                "message": "Chưa có khuyến nghị tái huấn luyện nào",
                "should_retrain": False
            }
        
        latest_recommendation = self.retraining_recommendations[-1].copy()
        
        # Kiểm tra xem khuyến nghị có quá cũ không
        recommendation_time = datetime.fromisoformat(latest_recommendation["timestamp"])
        time_since_recommendation = (datetime.now() - recommendation_time).total_seconds() / (24 * 3600)  # Số ngày
        
        should_retrain = time_since_recommendation <= 7  # Chỉ đề xuất tái huấn luyện nếu khuyến nghị trong vòng 7 ngày
        
        return {
            "status": "recommendation_available",
            "should_retrain": should_retrain,
            "recommendation": latest_recommendation,
            "days_since_recommendation": time_since_recommendation
        }
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Tạo bản tóm tắt hiệu suất trong khoảng thời gian nhất định.
        
        Args:
            days: Số ngày để tóm tắt
            
        Returns:
            Dict chứa bản tóm tắt hiệu suất
        """
        if not self.performance_history:
            return {
                "status": "no_data",
                "message": "Chưa có dữ liệu hiệu suất nào được ghi nhận"
            }
        
        current_time = datetime.now()
        
        # Lọc dữ liệu trong khoảng thời gian
        filtered_data = []
        for entry in self.performance_history:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if (current_time - entry_time).days <= days:
                filtered_data.append(entry)
        
        if not filtered_data:
            return {
                "status": "no_recent_data",
                "message": f"Không có dữ liệu hiệu suất nào trong {days} ngày gần đây"
            }
        
        # Tính toán tóm tắt cho từng metric
        metrics_summary = {}
        
        for metric in self.tracking_metrics:
            values = [entry.get("metrics", {}).get(metric) for entry in filtered_data if metric in entry.get("metrics", {})]
            
            if values:
                metrics_summary[metric] = {
                    "start": values[0],
                    "end": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "std": np.std(values),
                    "change": values[-1] - values[0],
                    "percent_change": (values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else float('inf')
                }
        
        # Tính toán tóm tắt theo symbol
        symbol_summary = {}
        
        for symbol in self.symbols:
            symbol_data = self.symbol_performance.get(symbol, [])
            filtered_symbol_data = []
            
            for entry in symbol_data:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if (current_time - entry_time).days <= days:
                    filtered_symbol_data.append(entry)
            
            if filtered_symbol_data:
                symbol_metrics = {}
                
                for metric in self.tracking_metrics:
                    values = [entry.get("metrics", {}).get(metric) for entry in filtered_symbol_data if metric in entry.get("metrics", {})]
                    
                    if values:
                        symbol_metrics[metric] = {
                            "start": values[0],
                            "end": values[-1],
                            "mean": sum(values) / len(values),
                            "change": values[-1] - values[0],
                            "percent_change": (values[-1] - values[0]) / abs(values[0]) if values[0] != 0 else float('inf')
                        }
                
                symbol_summary[symbol] = symbol_metrics
        
        # Tạo báo cáo tóm tắt
        summary = {
            "period": {
                "days": days,
                "start": filtered_data[0]["timestamp"],
                "end": filtered_data[-1]["timestamp"]
            },
            "data_points": len(filtered_data),
            "metrics_summary": metrics_summary,
            "symbol_summary": symbol_summary,
            "baseline_comparison": self._compare_with_baseline(filtered_data[-1].get("metrics", {})),
            "trends": self._analyze_performance_trends(window=min(days, 7))
        }
        
        return summary
    
    def plot_performance_trends(
        self, 
        metrics: Optional[List[str]] = None,
        days: int = 30,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> Optional[plt.Figure]:
        """
        Vẽ biểu đồ xu hướng hiệu suất.
        
        Args:
            metrics: Danh sách metrics cần vẽ (None để sử dụng tất cả)
            days: Số ngày dữ liệu để hiển thị
            figsize: Kích thước biểu đồ
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ hay không
            
        Returns:
            Figure của matplotlib hoặc None
        """
        if not self.performance_history:
            self.logger.warning("Không có dữ liệu hiệu suất để vẽ biểu đồ")
            return None
        
        # Sử dụng tất cả metrics nếu không có danh sách metrics được cung cấp
        if metrics is None:
            metrics = self.tracking_metrics
        
        current_time = datetime.now()
        
        # Lọc dữ liệu trong khoảng thời gian
        filtered_data = []
        for entry in self.performance_history:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if (current_time - entry_time).days <= days:
                filtered_data.append(entry)
        
        if not filtered_data:
            self.logger.warning(f"Không có dữ liệu hiệu suất nào trong {days} ngày gần đây")
            return None
        
        # Chuẩn bị dữ liệu
        data = {
            "timestamp": [datetime.fromisoformat(entry["timestamp"]) for entry in filtered_data]
        }
        
        for metric in metrics:
            data[metric] = [entry.get("metrics", {}).get(metric) for entry in filtered_data]
        
        # Tạo DataFrame
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        # Tạo biểu đồ
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
        
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            # Kiểm tra dữ liệu hợp lệ
            if metric not in df.columns or df[metric].isna().all():
                continue
            
            # Vẽ biểu đồ
            axes[i].plot(df.index, df[metric], marker='o', linestyle='-', markersize=4)
            
            # Vẽ đường baseline nếu có
            if metric in self.baseline_metrics:
                axes[i].axhline(y=self.baseline_metrics[metric], color='r', linestyle='--', alpha=0.7, 
                               label=f'Baseline: {self.baseline_metrics[metric]:.4f}')
            
            # Tùy chỉnh biểu đồ
            axes[i].set_title(f"{metric.replace('_', ' ').title()}")
            axes[i].grid(True, alpha=0.3)
            
            # Định dạng giá trị là phần trăm cho một số metrics
            if metric in [BacktestMetric.WIN_RATE.value, BacktestMetric.MAX_DRAWDOWN.value, "roi"]:
                axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # Thêm legend
            if metric in self.baseline_metrics:
                axes[i].legend()
            
            # Tính và vẽ đường xu hướng
            if len(df[metric].dropna()) >= 2:
                x = np.arange(len(df[metric].dropna()))
                y = df[metric].dropna().values
                
                try:
                    slope, intercept = np.polyfit(x, y, 1)
                    trend_x = np.array([x[0], x[-1]])
                    trend_y = slope * trend_x + intercept
                    
                    # Vẽ đường xu hướng
                    axes[i].plot(df[metric].dropna().index.values[[0, -1]], trend_y, 'g--', alpha=0.7, 
                               label=f'Trend: {"+" if slope > 0 else ""}{slope:.2e}')
                    axes[i].legend()
                    
                except Exception as e:
                    self.logger.warning(f"Lỗi khi tính xu hướng cho {metric}: {e}")
        
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
    
    def plot_symbol_comparison(
        self,
        metric: str,
        days: int = 30,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> Optional[plt.Figure]:
        """
        Vẽ biểu đồ so sánh hiệu suất giữa các symbol.
        
        Args:
            metric: Metric cần so sánh
            days: Số ngày dữ liệu để hiển thị
            figsize: Kích thước biểu đồ
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ hay không
            
        Returns:
            Figure của matplotlib hoặc None
        """
        current_time = datetime.now()
        
        # Tạo dữ liệu cho từng symbol
        symbol_data = {}
        
        for symbol in self.symbols:
            symbol_history = []
            
            for entry in self.symbol_performance.get(symbol, []):
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if (current_time - entry_time).days <= days:
                    symbol_history.append(entry)
            
            if symbol_history:
                timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in symbol_history]
                values = [entry.get("metrics", {}).get(metric) for entry in symbol_history]
                
                if len(timestamps) > 0 and not all(v is None for v in values):
                    symbol_data[symbol] = {
                        "timestamps": timestamps,
                        "values": values
                    }
        
        if not symbol_data:
            self.logger.warning(f"Không có dữ liệu để vẽ biểu đồ so sánh cho metric {metric}")
            return None
        
        # Tạo biểu đồ
        fig, ax = plt.subplots(figsize=figsize)
        
        for symbol, data in symbol_data.items():
            ax.plot(data["timestamps"], data["values"], marker='o', linestyle='-', markersize=4, label=symbol)
        
        # Vẽ đường baseline nếu có
        if metric in self.baseline_metrics:
            ax.axhline(y=self.baseline_metrics[metric], color='r', linestyle='--', alpha=0.7, 
                      label=f'Baseline: {self.baseline_metrics[metric]:.4f}')
        
        # Tùy chỉnh biểu đồ
        ax.set_title(f"So sánh {metric.replace('_', ' ').title()} giữa các symbol")
        ax.set_xlabel("Thời gian")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Định dạng giá trị là phần trăm cho một số metrics
        if metric in [BacktestMetric.WIN_RATE.value, BacktestMetric.MAX_DRAWDOWN.value, "roi"]:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
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
    
    def generate_performance_report(
        self,
        days: int = 30,
        output_path: Optional[Union[str, Path]] = None,
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Tạo báo cáo hiệu suất chi tiết.
        
        Args:
            days: Số ngày dữ liệu để đưa vào báo cáo
            output_path: Đường dẫn để lưu báo cáo
            include_plots: Bao gồm biểu đồ hay không
            
        Returns:
            Dict chứa báo cáo
        """
        if not self.performance_history:
            return {
                "status": "no_data",
                "message": "Chưa có dữ liệu hiệu suất nào được ghi nhận"
            }
        
        # Lấy bản tóm tắt hiệu suất
        summary = self.get_performance_summary(days=days)
        
        # Lấy khuyến nghị tái huấn luyện
        retraining_recommendation = self.get_retraining_recommendation()
        
        # Tạo báo cáo
        report = {
            "agent_name": self.agent_name,
            "model_version": self.model_version,
            "strategy_name": self.strategy_name,
            "report_time": datetime.now().isoformat(),
            "performance_summary": summary,
            "retraining_recommendation": retraining_recommendation,
            "plots": []
        }
        
        # Biểu đồ (nếu cần)
        if include_plots:
            # Thư mục để lưu biểu đồ
            plot_dir = None
            if output_path:
                plot_dir = Path(output_path).parent / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Vẽ biểu đồ xu hướng hiệu suất
                trends_path = None
                if plot_dir:
                    trends_path = plot_dir / f"{self.agent_name}_{self.model_version}_trends.png"
                
                fig = self.plot_performance_trends(
                    days=days,
                    show_plot=False,
                    save_path=trends_path
                )
                
                if fig and trends_path:
                    report["plots"].append(str(trends_path))
                    plt.close(fig)
                
                # Vẽ biểu đồ so sánh symbol cho một số metrics quan trọng
                for metric in [BacktestMetric.SHARPE_RATIO.value, BacktestMetric.WIN_RATE.value, "roi"]:
                    symbol_compare_path = None
                    if plot_dir:
                        symbol_compare_path = plot_dir / f"{self.agent_name}_{self.model_version}_{metric}_symbols.png"
                    
                    fig = self.plot_symbol_comparison(
                        metric=metric,
                        days=days,
                        show_plot=False,
                        save_path=symbol_compare_path
                    )
                    
                    if fig and symbol_compare_path:
                        report["plots"].append(str(symbol_compare_path))
                        plt.close(fig)
            
            except Exception as e:
                self.logger.warning(f"Lỗi khi tạo biểu đồ: {e}")
        
        # Lưu báo cáo nếu cần
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=4)
            
            self.logger.info(f"Đã lưu báo cáo hiệu suất tại {output_path}")
        
        return report
    
    def mark_retrained(
        self,
        new_model_version: str,
        reset_history: bool = False,
        keep_baseline: bool = True
    ) -> Dict[str, Any]:
        """
        Đánh dấu model đã được tái huấn luyện và cập nhật thông tin.
        
        Args:
            new_model_version: Phiên bản mới của model
            reset_history: Xóa lịch sử hiệu suất cũ hay không
            keep_baseline: Giữ nguyên baseline hay không
            
        Returns:
            Dict chứa thông tin trạng thái
        """
        # Lưu lịch sử hiện tại vào file
        old_history_path = self.retraining_dir / f"{self.agent_name}_{self.model_version}_history_archive.json"
        
        try:
            # Lưu lịch sử cũ
            data = {
                "agent_name": self.agent_name,
                "model_version": self.model_version,
                "strategy_name": self.strategy_name,
                "symbols": self.symbols,
                "performance_history": self.performance_history,
                "retraining_recommendations": self.retraining_recommendations,
                "symbol_performance": self.symbol_performance,
                "archive_time": datetime.now().isoformat(),
                "new_model_version": new_model_version
            }
            
            with open(old_history_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Đã lưu lịch sử hiệu suất cũ vào {old_history_path}")
            
            # Cập nhật model version
            old_model_version = self.model_version
            self.model_version = new_model_version
            
            # Nếu reset_history, xóa lịch sử hiệu suất cũ
            if reset_history:
                self.performance_history = []
                self.retraining_recommendations = []
                self.symbol_performance = {symbol: [] for symbol in self.symbols}
                self.last_update_time = None
            
            # Tạo thư mục mới cho phiên bản mới
            self.retraining_dir = self.retraining_dir.parent / f"{self.agent_name}_{new_model_version}"
            self.retraining_dir.mkdir(parents=True, exist_ok=True)
            
            # Nếu keep_baseline=False, sử dụng số liệu hiệu suất hiện tại làm baseline mới
            if not keep_baseline and self.performance_history:
                latest_metrics = self.performance_history[-1].get("metrics", {})
                if latest_metrics:
                    self.baseline_metrics = latest_metrics.copy()
                    
                    # Lưu baseline mới
                    baseline_path = self.retraining_dir / f"{self.agent_name}_{new_model_version}_baseline.json"
                    with open(baseline_path, 'w', encoding='utf-8') as f:
                        json.dump(self.baseline_metrics, f, indent=4, ensure_ascii=False)
                    
                    self.baseline_metrics_path = baseline_path
            
            # Tạo sequence ID mới
            self.current_sequence_id = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # Lưu lịch sử mới
            self.save_history()
            
            return {
                "status": "success",
                "old_model_version": old_model_version,
                "new_model_version": new_model_version,
                "history_reset": reset_history,
                "baseline_kept": keep_baseline,
                "archive_path": str(old_history_path)
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi khi đánh dấu model đã được tái huấn luyện: {e}")
            return {
                "status": "error",
                "message": f"Lỗi khi đánh dấu model đã được tái huấn luyện: {e}"
            }
    
    def get_historical_versions(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các phiên bản model trước đó.
        
        Returns:
            List các phiên bản model và thông tin liên quan
        """
        versions = []
        
        # Tìm các file lịch sử archive
        archive_pattern = f"{self.agent_name}_*_history_archive.json"
        archive_files = list(self.retraining_dir.parent.glob(archive_pattern))
        
        for archive_file in archive_files:
            try:
                with open(archive_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Lấy thông tin version
                model_version = data.get("model_version")
                archive_time = data.get("archive_time")
                new_model_version = data.get("new_model_version")
                
                if model_version and archive_time:
                    versions.append({
                        "model_version": model_version,
                        "archive_time": archive_time,
                        "new_model_version": new_model_version,
                        "archive_file": str(archive_file)
                    })
                
            except Exception as e:
                self.logger.warning(f"Lỗi khi đọc file archive {archive_file}: {e}")
        
        # Sắp xếp theo thời gian
        versions.sort(key=lambda x: x.get("archive_time", ""), reverse=True)
        
        return versions
    
    def compare_with_previous_version(
        self,
        previous_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        So sánh hiệu suất với phiên bản trước đó.
        
        Args:
            previous_version: Phiên bản trước để so sánh (None để tự động chọn)
            
        Returns:
            Dict chứa kết quả so sánh
        """
        # Nếu không có phiên bản trước được chỉ định, tìm phiên bản gần nhất
        if not previous_version:
            versions = self.get_historical_versions()
            if not versions:
                return {
                    "status": "no_previous_version",
                    "message": "Không tìm thấy phiên bản trước đó"
                }
            
            previous_version = versions[0]["model_version"]
        
        # Tìm file archive của phiên bản trước
        archive_file = self.retraining_dir.parent / f"{self.agent_name}_{previous_version}_history_archive.json"
        
        if not archive_file.exists():
            return {
                "status": "version_not_found",
                "message": f"Không tìm thấy dữ liệu lịch sử cho phiên bản {previous_version}"
            }
        
        try:
            # Đọc dữ liệu lịch sử
            with open(archive_file, 'r', encoding='utf-8') as f:
                previous_data = json.load(f)
            
            previous_history = previous_data.get("performance_history", [])
            
            if not previous_history or not self.performance_history:
                return {
                    "status": "insufficient_data",
                    "message": "Không đủ dữ liệu để so sánh"
                }
            
            # Lấy số liệu gần nhất của cả hai phiên bản
            current_metrics = self.performance_history[-1].get("metrics", {})
            previous_metrics = previous_history[-1].get("metrics", {})
            
            # So sánh các metrics
            comparison = {}
            
            for metric in self.tracking_metrics:
                if metric in current_metrics and metric in previous_metrics:
                    current = current_metrics[metric]
                    previous = previous_metrics[metric]
                    
                    # Tính toán chênh lệch
                    if previous != 0:
                        relative_diff = (current - previous) / abs(previous)
                    else:
                        relative_diff = float('inf') if current > 0 else float('-inf') if current < 0 else 0
                    
                    # Xác định liệu chênh lệch có tích cực hay không
                    if metric in [BacktestMetric.MAX_DRAWDOWN.value, "volatility"]:
                        # Với drawdown và volatility, thấp hơn là tốt hơn
                        is_better = current < previous
                    else:
                        # Với các metrics khác, cao hơn là tốt hơn
                        is_better = current > previous
                    
                    comparison[metric] = {
                        "current": current,
                        "previous": previous,
                        "absolute_diff": current - previous,
                        "relative_diff": relative_diff,
                        "is_better": is_better
                    }
            
            # Đánh giá tổng thể
            better_metrics_count = sum(1 for metric_data in comparison.values() if metric_data["is_better"])
            worse_metrics_count = sum(1 for metric_data in comparison.values() if not metric_data["is_better"])
            
            is_better_overall = better_metrics_count > worse_metrics_count
            
            return {
                "status": "success",
                "current_version": self.model_version,
                "previous_version": previous_version,
                "comparison": comparison,
                "better_metrics_count": better_metrics_count,
                "worse_metrics_count": worse_metrics_count,
                "total_metrics_compared": len(comparison),
                "is_better_overall": is_better_overall
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi khi so sánh với phiên bản trước: {e}")
            return {
                "status": "error",
                "message": f"Lỗi khi so sánh với phiên bản trước: {e}"
            }


def create_performance_tracker(
    agent_name: str,
    model_version: str,
    strategy_name: str,
    symbols: List[str],
    tracking_metrics: Optional[List[str]] = None,
    retraining_dir: Optional[Union[str, Path]] = None,
    baseline_metrics_path: Optional[Union[str, Path]] = None,
    evaluation_window: int = 14,
    retraining_triggers: Optional[Dict[str, Dict[str, float]]] = None,
    logger: Optional[logging.Logger] = None
) -> PerformanceTracker:
    """
    Hàm tiện ích để tạo PerformanceTracker.
    
    Args:
        agent_name: Tên của agent
        model_version: Phiên bản mô hình
        strategy_name: Tên chiến lược giao dịch
        symbols: Danh sách các cặp tiền được giao dịch
        tracking_metrics: Các số liệu cần theo dõi (None để sử dụng danh sách mặc định)
        retraining_dir: Thư mục lưu dữ liệu theo dõi hiệu suất và quyết định tái huấn luyện
        baseline_metrics_path: Đường dẫn đến file số liệu baseline
        evaluation_window: Số ngày dữ liệu để đánh giá hiệu suất
        retraining_triggers: Dict chứa các ngưỡng kích hoạt tái huấn luyện
        logger: Logger tùy chỉnh
        
    Returns:
        PerformanceTracker đã được cấu hình
    """
    tracker = PerformanceTracker(
        agent_name=agent_name,
        model_version=model_version,
        strategy_name=strategy_name,
        symbols=symbols,
        tracking_metrics=tracking_metrics,
        retraining_dir=retraining_dir,
        baseline_metrics_path=baseline_metrics_path,
        evaluation_window=evaluation_window,
        retraining_triggers=retraining_triggers,
        logger=logger
    )
    
    return tracker