"""
Chỉ số hiệu suất tự động cho quá trình tự động cải tiến.
File này định nghĩa các lớp và hàm để tính toán và theo dõi các chỉ số hiệu suất nâng cao
giúp hệ thống quyết định khi nào cần tái huấn luyện hoặc cải tiến các mô hình.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy import stats

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from backtesting.performance_metrics import PerformanceMetrics

class AutomationPerformanceMetrics:
    """
    Lớp tính toán các chỉ số hiệu suất nâng cao cho quá trình tự động cải tiến.
    Mở rộng từ PerformanceMetrics với các chỉ số bổ sung tập trung vào phát hiện
    thay đổi và xu hướng suy giảm hiệu suất để kích hoạt quá trình tái huấn luyện.
    """
    
    def __init__(
        self,
        equity_curve: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252,
        initial_capital: float = 10000,
        fee_rate: float = 0.001,
        window_sizes: Dict[str, int] = None,
        alert_thresholds: Dict[str, float] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo đối tượng AutomationPerformanceMetrics.
        
        Args:
            equity_curve (pd.Series): Chuỗi giá trị vốn theo thời gian
            trades (pd.DataFrame, optional): DataFrame chứa thông tin các giao dịch
            benchmark (pd.Series, optional): Chuỗi giá trị của benchmark theo thời gian
            risk_free_rate (float, optional): Lãi suất phi rủi ro hàng năm. Mặc định là 0.02 (2%)
            trading_days_per_year (int, optional): Số ngày giao dịch trong một năm. Mặc định là 252
            initial_capital (float, optional): Vốn ban đầu. Mặc định là 10,000
            fee_rate (float, optional): Tỷ lệ phí giao dịch. Mặc định là 0.001 (0.1%)
            window_sizes (Dict[str, int], optional): Kích thước cửa sổ cho các chỉ số khác nhau
            alert_thresholds (Dict[str, float], optional): Ngưỡng cảnh báo cho các chỉ số
            logger (logging.Logger, optional): Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("automation_performance_metrics")
        
        # Khởi tạo PerformanceMetrics cơ bản
        self.base_metrics = PerformanceMetrics(
            equity_curve=equity_curve,
            trades=trades,
            benchmark=benchmark,
            risk_free_rate=risk_free_rate,
            trading_days_per_year=trading_days_per_year,
            initial_capital=initial_capital,
            fee_rate=fee_rate
        )
        
        # Lưu các tham số
        self.equity_curve = equity_curve
        self.trades = trades
        self.benchmark = benchmark
        
        # Thiết lập cửa sổ cho các chỉ số
        self.window_sizes = window_sizes or {
            "short": 5,           # Cửa sổ ngắn (5 giao dịch/ngày)
            "medium": 20,         # Cửa sổ trung bình (1 tháng giao dịch)
            "long": 60,           # Cửa sổ dài (3 tháng giao dịch)
            "very_long": 120      # Cửa sổ rất dài (6 tháng giao dịch)
        }
        
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
        
        # Lịch sử cảnh báo
        self.alert_history = []
        
        # Lịch sử chỉ số theo thời gian
        self.metrics_history = {
            "timestamp": [],
            "rolling_sharpe": [],
            "rolling_sortino": [],
            "rolling_win_rate": [],
            "rolling_profit_factor": [],
            "rolling_drawdown": [],
            "rolling_volatility": [],
            "regime_change_indicator": []
        }
        
        # Khởi tạo tracking metrics
        self._initialize_tracking()
    
    def _initialize_tracking(self) -> None:
        """
        Khởi tạo theo dõi chỉ số theo thời gian.
        """
        # Nếu có dữ liệu giao dịch, tính các chỉ số ban đầu
        if self.trades is not None and len(self.trades) > 0:
            # Tính các chỉ số rolling ban đầu
            self._calculate_rolling_metrics()
    
    def _calculate_rolling_metrics(self) -> Dict[str, pd.Series]:
        """
        Tính toán các chỉ số trượt (rolling) qua thời gian.
        
        Returns:
            Dict[str, pd.Series]: Dictionary các chỉ số trượt theo thời gian
        """
        if self.trades is None or len(self.trades) < self.window_sizes["short"]:
            return {}
        
        # Đảm bảo trades đã được sắp xếp theo thời gian
        if 'entry_time' in self.trades.columns:
            trades_sorted = self.trades.sort_values('entry_time')
        else:
            trades_sorted = self.trades
        
        # Tính toán các chỉ số trượt
        rolling_metrics = {}
        
        # Cửa sổ ngắn (nhạy cảm với thay đổi gần đây)
        short_window = self.window_sizes["short"]
        
        # Win rate trượt
        rolling_metrics["rolling_win_rate_short"] = trades_sorted['profit'].rolling(
            window=short_window, min_periods=3).apply(
            lambda x: (x > 0).mean() if len(x) > 0 else np.nan
        )
        
        # Profit factor trượt
        def calc_profit_factor(x):
            wins = x[x > 0].sum()
            losses = abs(x[x < 0].sum())
            return wins / losses if losses != 0 else (float('inf') if wins > 0 else 0)
        
        rolling_metrics["rolling_profit_factor_short"] = trades_sorted['profit'].rolling(
            window=short_window, min_periods=3).apply(calc_profit_factor)
        
        # Cửa sổ trung bình
        medium_window = self.window_sizes["medium"]
        rolling_metrics["rolling_win_rate_medium"] = trades_sorted['profit'].rolling(
            window=medium_window, min_periods=5).apply(
            lambda x: (x > 0).mean() if len(x) > 0 else np.nan
        )
        rolling_metrics["rolling_profit_factor_medium"] = trades_sorted['profit'].rolling(
            window=medium_window, min_periods=5).apply(calc_profit_factor)
        
        # Tính hiệu quả (phần trăm thay đổi trong Win Rate)
        if len(rolling_metrics["rolling_win_rate_medium"]) > medium_window:
            past_win_rate = rolling_metrics["rolling_win_rate_medium"].shift(medium_window)
            current_win_rate = rolling_metrics["rolling_win_rate_medium"]
            rolling_metrics["win_rate_efficiency"] = (current_win_rate - past_win_rate) / past_win_rate
            
            past_pf = rolling_metrics["rolling_profit_factor_medium"].shift(medium_window)
            current_pf = rolling_metrics["rolling_profit_factor_medium"]
            rolling_metrics["profit_factor_efficiency"] = (current_pf - past_pf) / past_pf
        
        # Chỉ số phát hiện thay đổi chế độ (regime change)
        if 'equity' in self.trades.columns:
            # Sử dụng CUSUM (Cumulative Sum) để phát hiện thay đổi
            returns = self.trades['equity'].pct_change().dropna()
            rolling_metrics["regime_change_indicator"] = self._calculate_cusum(returns)
        
        return rolling_metrics
    
    def _calculate_cusum(self, returns: pd.Series, threshold: float = 1.0, drift: float = 0.0) -> pd.Series:
        """
        Tính CUSUM (Cumulative Sum) để phát hiện thay đổi chế độ.
        
        Args:
            returns (pd.Series): Chuỗi lợi nhuận (returns)
            threshold (float): Ngưỡng phát hiện
            drift (float): Hệ số dịch chuyển
            
        Returns:
            pd.Series: Chỉ số CUSUM
        """
        # Chuẩn hóa returns
        std_returns = (returns - returns.mean()) / returns.std()
        
        # Tính CUSUM+
        pos_cusum = np.zeros_like(std_returns)
        # Tính CUSUM-
        neg_cusum = np.zeros_like(std_returns)
        
        for i in range(1, len(std_returns)):
            # CUSUM dương
            pos_cusum[i] = max(0, pos_cusum[i-1] + std_returns.iloc[i] - drift)
            # CUSUM âm
            neg_cusum[i] = max(0, neg_cusum[i-1] - std_returns.iloc[i] - drift)
        
        # Kết hợp thành một chỉ số
        cusum = pd.Series(data=np.maximum(pos_cusum, neg_cusum), index=returns.index)
        
        # Phát hiện thay đổi khi vượt ngưỡng
        change_points = cusum > threshold
        
        return change_points
    
    def update_metrics(self, new_equity_point: float, new_trades: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Cập nhật chỉ số với dữ liệu mới.
        
        Args:
            new_equity_point (float): Giá trị vốn mới
            new_trades (pd.DataFrame, optional): Giao dịch mới để thêm vào
            
        Returns:
            Dict[str, Any]: Dictionary các chỉ số và cảnh báo mới
        """
        # Cập nhật equity curve
        self.equity_curve = pd.concat([
            self.equity_curve, 
            pd.Series([new_equity_point], index=[datetime.now()])
        ])
        
        # Cập nhật trades nếu có
        if new_trades is not None and len(new_trades) > 0:
            if self.trades is None:
                self.trades = new_trades
            else:
                self.trades = pd.concat([self.trades, new_trades], ignore_index=True)
        
        # Tính lại chỉ số cơ bản
        self.base_metrics = PerformanceMetrics(
            equity_curve=self.equity_curve,
            trades=self.trades,
            benchmark=self.benchmark,
            risk_free_rate=self.base_metrics.risk_free_rate,
            trading_days_per_year=self.base_metrics.trading_days_per_year,
            initial_capital=self.base_metrics.initial_capital,
            fee_rate=self.base_metrics.fee_rate
        )
        
        # Tính lại rolling metrics
        rolling_metrics = self._calculate_rolling_metrics()
        
        # Kiểm tra cảnh báo
        alerts = self.check_for_alerts(rolling_metrics)
        
        # Cập nhật lịch sử chỉ số
        self._update_metrics_history()
        
        return {
            "base_metrics": self.base_metrics.calculate_all_metrics(),
            "rolling_metrics": rolling_metrics,
            "alerts": alerts
        }
    
    def check_for_alerts(self, rolling_metrics: Optional[Dict[str, pd.Series]] = None) -> List[Dict[str, Any]]:
        """
        Kiểm tra các cảnh báo dựa trên ngưỡng đã thiết lập.
        
        Args:
            rolling_metrics (Dict[str, pd.Series], optional): Chỉ số trượt đã tính toán trước
            
        Returns:
            List[Dict[str, Any]]: Danh sách các cảnh báo mới
        """
        if rolling_metrics is None:
            rolling_metrics = self._calculate_rolling_metrics()
        
        new_alerts = []
        
        # Kiểm tra drawdown
        current_drawdown = self.base_metrics.max_drawdown()
        if current_drawdown > self.alert_thresholds["drawdown"]:
            alert = {
                "timestamp": datetime.now(),
                "type": "drawdown_alert",
                "message": f"Drawdown đã vượt ngưỡng: {current_drawdown:.2%}",
                "value": current_drawdown,
                "threshold": self.alert_thresholds["drawdown"],
                "severity": "high" if current_drawdown > self.alert_thresholds["drawdown"] * 1.5 else "medium"
            }
            new_alerts.append(alert)
        
        # Kiểm tra sự suy giảm win rate
        if "rolling_win_rate_short" in rolling_metrics and "rolling_win_rate_medium" in rolling_metrics:
            short_win_rate = rolling_metrics["rolling_win_rate_short"].iloc[-1] if len(rolling_metrics["rolling_win_rate_short"]) > 0 else None
            medium_win_rate = rolling_metrics["rolling_win_rate_medium"].iloc[-1] if len(rolling_metrics["rolling_win_rate_medium"]) > 0 else None
            
            if short_win_rate is not None and medium_win_rate is not None:
                win_rate_decline = (medium_win_rate - short_win_rate) / medium_win_rate if medium_win_rate > 0 else 0
                
                if win_rate_decline > self.alert_thresholds["win_rate_decline"]:
                    alert = {
                        "timestamp": datetime.now(),
                        "type": "win_rate_decline_alert",
                        "message": f"Tỷ lệ thắng giảm: {win_rate_decline:.2%}",
                        "value": win_rate_decline,
                        "threshold": self.alert_thresholds["win_rate_decline"],
                        "severity": "high" if win_rate_decline > self.alert_thresholds["win_rate_decline"] * 1.5 else "medium"
                    }
                    new_alerts.append(alert)
        
        # Kiểm tra profit factor
        if "rolling_profit_factor_short" in rolling_metrics and "rolling_profit_factor_medium" in rolling_metrics:
            short_pf = rolling_metrics["rolling_profit_factor_short"].iloc[-1] if len(rolling_metrics["rolling_profit_factor_short"]) > 0 else None
            medium_pf = rolling_metrics["rolling_profit_factor_medium"].iloc[-1] if len(rolling_metrics["rolling_profit_factor_medium"]) > 0 else None
            
            if short_pf is not None and medium_pf is not None and medium_pf > 0:
                pf_decline = (medium_pf - short_pf) / medium_pf
                
                if pf_decline > self.alert_thresholds["profit_factor_decline"]:
                    alert = {
                        "timestamp": datetime.now(),
                        "type": "profit_factor_decline_alert",
                        "message": f"Profit factor giảm: {pf_decline:.2%}",
                        "value": pf_decline,
                        "threshold": self.alert_thresholds["profit_factor_decline"],
                        "severity": "high" if pf_decline > self.alert_thresholds["profit_factor_decline"] * 1.5 else "medium"
                    }
                    new_alerts.append(alert)
        
        # Kiểm tra giao dịch thua liên tiếp
        if self.trades is not None and len(self.trades) >= self.alert_thresholds["consecutive_losses"]:
            recent_trades = self.trades.iloc[-int(self.alert_thresholds["consecutive_losses"]):]
            if all(recent_trades['profit'] < 0):
                alert = {
                    "timestamp": datetime.now(),
                    "type": "consecutive_losses_alert",
                    "message": f"Có {len(recent_trades)} giao dịch thua liên tiếp",
                    "value": len(recent_trades),
                    "threshold": self.alert_thresholds["consecutive_losses"],
                    "severity": "high"
                }
                new_alerts.append(alert)
        
        # Kiểm tra thay đổi chế độ (regime change)
        if "regime_change_indicator" in rolling_metrics:
            recent_changes = rolling_metrics["regime_change_indicator"].iloc[-10:] if len(rolling_metrics["regime_change_indicator"]) > 10 else rolling_metrics["regime_change_indicator"]
            if recent_changes.any():
                alert = {
                    "timestamp": datetime.now(),
                    "type": "regime_change_alert",
                    "message": "Phát hiện thay đổi chế độ thị trường",
                    "value": True,
                    "threshold": True,
                    "severity": "high"
                }
                new_alerts.append(alert)
        
        # Cập nhật lịch sử cảnh báo
        if new_alerts:
            self.alert_history.extend(new_alerts)
            for alert in new_alerts:
                self.logger.warning(f"Cảnh báo hiệu suất: {alert['message']} (Mức độ: {alert['severity']})")
        
        return new_alerts
    
    def _update_metrics_history(self) -> None:
        """
        Cập nhật lịch sử các chỉ số theo thời gian.
        """
        now = datetime.now()
        
        # Tính các chỉ số hiện tại
        base_metrics = self.base_metrics.calculate_all_metrics()
        rolling_metrics = self._calculate_rolling_metrics()
        
        # Cập nhật lịch sử
        self.metrics_history["timestamp"].append(now)
        self.metrics_history["rolling_sharpe"].append(base_metrics.get("sharpe_ratio", None))
        self.metrics_history["rolling_sortino"].append(base_metrics.get("sortino_ratio", None))
        
        # Các chỉ số từ rolling_metrics
        if "rolling_win_rate_short" in rolling_metrics and len(rolling_metrics["rolling_win_rate_short"]) > 0:
            self.metrics_history["rolling_win_rate"].append(rolling_metrics["rolling_win_rate_short"].iloc[-1])
        else:
            self.metrics_history["rolling_win_rate"].append(None)
        
        if "rolling_profit_factor_short" in rolling_metrics and len(rolling_metrics["rolling_profit_factor_short"]) > 0:
            self.metrics_history["rolling_profit_factor"].append(rolling_metrics["rolling_profit_factor_short"].iloc[-1])
        else:
            self.metrics_history["rolling_profit_factor"].append(None)
        
        self.metrics_history["rolling_drawdown"].append(base_metrics.get("max_drawdown", None))
        self.metrics_history["rolling_volatility"].append(base_metrics.get("volatility", None))
        
        # Chỉ số thay đổi chế độ
        if "regime_change_indicator" in rolling_metrics and len(rolling_metrics["regime_change_indicator"]) > 0:
            self.metrics_history["regime_change_indicator"].append(rolling_metrics["regime_change_indicator"].iloc[-1])
        else:
            self.metrics_history["regime_change_indicator"].append(False)
    
    def get_retraining_recommendation(self) -> Dict[str, Any]:
        """
        Đưa ra khuyến nghị về việc cần tái huấn luyện hay không.
        
        Returns:
            Dict[str, Any]: Khuyến nghị tái huấn luyện và các chỉ số liên quan
        """
        # Tính toán các chỉ số hiện tại
        current_metrics = self.base_metrics.calculate_all_metrics()
        rolling_metrics = self._calculate_rolling_metrics()
        
        # Kiểm tra các điều kiện cần tái huấn luyện
        conditions_met = 0
        conditions_total = 5
        reasons = []
        
        # 1. Drawdown vượt ngưỡng
        if current_metrics.get("max_drawdown", 0) > self.alert_thresholds["drawdown"]:
            conditions_met += 1
            reasons.append(f"Drawdown ({current_metrics['max_drawdown']:.2%}) vượt ngưỡng ({self.alert_thresholds['drawdown']:.2%})")
        
        # 2. Suy giảm win rate đáng kể
        if len(self.metrics_history["rolling_win_rate"]) > self.window_sizes["medium"]:
            recent_win_rate = self.metrics_history["rolling_win_rate"][-1]
            historical_win_rate = np.mean(self.metrics_history["rolling_win_rate"][-self.window_sizes["medium"]:-1])
            
            if recent_win_rate is not None and historical_win_rate is not None:
                win_rate_decline = (historical_win_rate - recent_win_rate) / historical_win_rate if historical_win_rate > 0 else 0
                
                if win_rate_decline > self.alert_thresholds["win_rate_decline"]:
                    conditions_met += 1
                    reasons.append(f"Tỷ lệ thắng giảm {win_rate_decline:.2%}")
        
        # 3. Suy giảm Sharpe ratio
        if len(self.metrics_history["rolling_sharpe"]) > self.window_sizes["medium"]:
            recent_sharpe = self.metrics_history["rolling_sharpe"][-1]
            historical_sharpe = np.mean([s for s in self.metrics_history["rolling_sharpe"][-self.window_sizes["medium"]:-1] if s is not None])
            
            if recent_sharpe is not None and historical_sharpe is not None and historical_sharpe > 0:
                sharpe_decline = (historical_sharpe - recent_sharpe) / historical_sharpe
                
                if sharpe_decline > self.alert_thresholds["sharpe_decline"]:
                    conditions_met += 1
                    reasons.append(f"Sharpe ratio giảm {sharpe_decline:.2%}")
        
        # 4. Phát hiện thay đổi chế độ thị trường
        if any(self.metrics_history["regime_change_indicator"][-10:]):
            conditions_met += 1
            reasons.append("Phát hiện thay đổi chế độ thị trường")
        
        # 5. Số lượng cảnh báo tích lũy
        recent_alerts = [alert for alert in self.alert_history 
                         if (datetime.now() - alert["timestamp"]).days <= 7]
        if len(recent_alerts) >= 3:
            conditions_met += 1
            reasons.append(f"Có {len(recent_alerts)} cảnh báo trong 7 ngày qua")
        
        # Tính độ tự tin cho khuyến nghị
        confidence = conditions_met / conditions_total
        
        # Quyết định khuyến nghị
        should_retrain = confidence >= 0.6  # Tái huấn luyện nếu 60% điều kiện được đáp ứng
        
        return {
            "should_retrain": should_retrain,
            "confidence": confidence,
            "reasons": reasons,
            "metrics": {
                "current": current_metrics,
                "rolling": {k: v.iloc[-1] if isinstance(v, pd.Series) and len(v) > 0 else v 
                           for k, v in rolling_metrics.items()}
            },
            "alert_count": len(recent_alerts),
            "timestamp": datetime.now()
        }
    
    def plot_performance_overview(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Tạo biểu đồ tổng quan về hiệu suất cho quá trình tự động cải tiến.
        
        Args:
            figsize (Tuple[int, int], optional): Kích thước hình. Mặc định là (15, 10).
            
        Returns:
            plt.Figure: Đối tượng Figure của matplotlib
        """
        fig, axs = plt.subplots(3, 2, figsize=figsize)
        
        # Chuyển đổi lịch sử chỉ số thành DataFrame
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.set_index("timestamp", inplace=True)
        
        # 1. Đường cong vốn
        self.base_metrics.plot_equity_curve(figsize=(15, 3))
        plt.close()  # Đóng hình vừa tạo, chỉ lấy nội dung
        
        ax1 = axs[0, 0]
        self.equity_curve.plot(ax=ax1, color='blue', linewidth=2)
        ax1.set_title('Đường cong vốn')
        ax1.set_ylabel('Giá trị vốn')
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Win Rate
        ax2 = axs[0, 1]
        if len(metrics_df) > 0:
            metrics_df['rolling_win_rate'].dropna().plot(ax=ax2, color='green', linewidth=2)
            ax2.set_title('Tỷ lệ thắng theo thời gian')
            ax2.set_ylabel('Tỷ lệ thắng')
            ax2.grid(True, alpha=0.3)
            
            # Thêm ngưỡng cảnh báo
            if len(metrics_df) > self.window_sizes["medium"]:
                historical_win_rate = metrics_df['rolling_win_rate'].dropna().rolling(self.window_sizes["medium"]).mean()
                warning_threshold = historical_win_rate * (1 - self.alert_thresholds["win_rate_decline"])
                warning_threshold.plot(ax=ax2, color='red', linestyle='--', alpha=0.7, label='Ngưỡng cảnh báo')
                ax2.legend()
        
        # 3. Rolling Sharpe & Sortino
        ax3 = axs[1, 0]
        if len(metrics_df) > 0:
            metrics_df['rolling_sharpe'].dropna().plot(ax=ax3, color='purple', linewidth=2, label='Sharpe')
            metrics_df['rolling_sortino'].dropna().plot(ax=ax3, color='darkred', linewidth=2, label='Sortino')
            ax3.set_title('Sharpe & Sortino theo thời gian')
            ax3.set_ylabel('Giá trị')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 4. Rolling Profit Factor
        ax4 = axs[1, 1]
        if len(metrics_df) > 0:
            metrics_df['rolling_profit_factor'].dropna().plot(ax=ax4, color='orange', linewidth=2)
            ax4.set_title('Profit Factor theo thời gian')
            ax4.set_ylabel('Profit Factor')
            ax4.grid(True, alpha=0.3)
            
            # Thêm ngưỡng cảnh báo
            if len(metrics_df) > self.window_sizes["medium"]:
                historical_pf = metrics_df['rolling_profit_factor'].dropna().rolling(self.window_sizes["medium"]).mean()
                warning_threshold = historical_pf * (1 - self.alert_thresholds["profit_factor_decline"])
                warning_threshold.plot(ax=ax4, color='red', linestyle='--', alpha=0.7, label='Ngưỡng cảnh báo')
                ax4.legend()
        
        # 5. Drawdown & Volatility
        ax5 = axs[2, 0]
        if len(metrics_df) > 0:
            metrics_df['rolling_drawdown'].dropna().plot(ax=ax5, color='red', linewidth=2, label='Drawdown')
            ax5.set_title('Drawdown theo thời gian')
            ax5.set_ylabel('Drawdown')
            ax5.grid(True, alpha=0.3)
            
            # Thêm ngưỡng cảnh báo
            ax5.axhline(y=self.alert_thresholds["drawdown"], color='red', linestyle='--', 
                     alpha=0.7, label=f'Ngưỡng cảnh báo ({self.alert_thresholds["drawdown"]:.2%})')
            ax5.legend()
        
        # 6. Cảnh báo & Đề xuất tái huấn luyện
        ax6 = axs[2, 1]
        ax6.axis('off')  # Tắt trục
        
        # Tạo bảng thông tin
        recommendation = self.get_retraining_recommendation()
        recent_alerts = [alert for alert in self.alert_history 
                         if (datetime.now() - alert["timestamp"]).days <= 7]
        
        info_text = (
            f"TÁI HUẤN LUYỆN: {'CẦN' if recommendation['should_retrain'] else 'KHÔNG CẦN'}\n"
            f"Độ tin cậy: {recommendation['confidence']:.2%}\n\n"
            f"LÝ DO:\n"
        )
        
        for reason in recommendation['reasons']:
            info_text += f"- {reason}\n"
        
        info_text += f"\nCẢNH BÁO GẦN ĐÂY ({len(recent_alerts)}):\n"
        
        for i, alert in enumerate(recent_alerts[:3]):
            info_text += f"- {alert['timestamp'].strftime('%d/%m/%Y')}: {alert['message']} [{alert['severity']}]\n"
        
        if len(recent_alerts) > 3:
            info_text += f"... và {len(recent_alerts) - 3} cảnh báo khác\n"
        
        ax6.text(0.05, 0.95, info_text, 
              transform=ax6.transAxes, 
              fontsize=10, 
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        return fig
    
    def save_metrics_history(self, path: Union[str, Path]) -> None:
        """
        Lưu lịch sử chỉ số vào file JSON.
        
        Args:
            path (Union[str, Path]): Đường dẫn lưu file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Chuẩn bị dữ liệu để lưu
        serializable_data = {
            "metrics_history": {
                "timestamp": [t.isoformat() for t in self.metrics_history["timestamp"]],
                "rolling_sharpe": [float(s) if s is not None else None for s in self.metrics_history["rolling_sharpe"]],
                "rolling_sortino": [float(s) if s is not None else None for s in self.metrics_history["rolling_sortino"]],
                "rolling_win_rate": [float(w) if w is not None else None for w in self.metrics_history["rolling_win_rate"]],
                "rolling_profit_factor": [float(p) if p is not None else None for p in self.metrics_history["rolling_profit_factor"]],
                "rolling_drawdown": [float(d) if d is not None else None for d in self.metrics_history["rolling_drawdown"]],
                "rolling_volatility": [float(v) if v is not None else None for v in self.metrics_history["rolling_volatility"]],
                "regime_change_indicator": [bool(r) for r in self.metrics_history["regime_change_indicator"]]
            },
            "alert_history": [
                {
                    "timestamp": alert["timestamp"].isoformat(),
                    "type": alert["type"],
                    "message": alert["message"],
                    "value": float(alert["value"]) if isinstance(alert["value"], (int, float)) else str(alert["value"]),
                    "threshold": float(alert["threshold"]) if isinstance(alert["threshold"], (int, float)) else str(alert["threshold"]),
                    "severity": alert["severity"]
                }
                for alert in self.alert_history
            ],
            "window_sizes": self.window_sizes,
            "alert_thresholds": self.alert_thresholds,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Đã lưu lịch sử chỉ số vào {path}")
    
    def load_metrics_history(self, path: Union[str, Path]) -> bool:
        """
        Tải lịch sử chỉ số từ file JSON.
        
        Args:
            path (Union[str, Path]): Đường dẫn file
            
        Returns:
            bool: True nếu tải thành công, False nếu không
        """
        path = Path(path)
        if not path.exists():
            self.logger.warning(f"Không tìm thấy file {path}")
            return False
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Khôi phục metrics_history
            self.metrics_history["timestamp"] = [datetime.fromisoformat(t) for t in data["metrics_history"]["timestamp"]]
            self.metrics_history["rolling_sharpe"] = data["metrics_history"]["rolling_sharpe"]
            self.metrics_history["rolling_sortino"] = data["metrics_history"]["rolling_sortino"]
            self.metrics_history["rolling_win_rate"] = data["metrics_history"]["rolling_win_rate"]
            self.metrics_history["rolling_profit_factor"] = data["metrics_history"]["rolling_profit_factor"]
            self.metrics_history["rolling_drawdown"] = data["metrics_history"]["rolling_drawdown"]
            self.metrics_history["rolling_volatility"] = data["metrics_history"]["rolling_volatility"]
            self.metrics_history["regime_change_indicator"] = data["metrics_history"]["regime_change_indicator"]
            
            # Khôi phục alert_history
            self.alert_history = [
                {
                    "timestamp": datetime.fromisoformat(alert["timestamp"]),
                    "type": alert["type"],
                    "message": alert["message"],
                    "value": alert["value"],
                    "threshold": alert["threshold"],
                    "severity": alert["severity"]
                }
                for alert in data["alert_history"]
            ]
            
            # Khôi phục các thông số khác
            if "window_sizes" in data:
                self.window_sizes = data["window_sizes"]
            
            if "alert_thresholds" in data:
                self.alert_thresholds = data["alert_thresholds"]
            
            self.logger.info(f"Đã tải lịch sử chỉ số từ {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải lịch sử chỉ số: {str(e)}")
            return False