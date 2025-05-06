"""
Đánh giá chiến lược giao dịch.
File này cung cấp các công cụ để phân tích và đánh giá chiến lược giao dịch
dựa trên kết quả backtest, bao gồm phân tích hiệu suất, phân tích rủi ro,
và so sánh giữa các chiến lược khác nhau.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import OrderStatus, OrderType, PositionStatus, PositionSide
from config.constants import Timeframe, BacktestMetric
from config.system_config import BACKTEST_DIR

# Import các module đánh giá khác
from backtesting.evaluation.performance_evaluator import PerformanceEvaluator
from backtesting.evaluation.risk_evaluator import RiskEvaluator

# Import các hồ sơ rủi ro
from risk_management.risk_profiles.conservative_profile import ConservativeProfile
from risk_management.risk_profiles.moderate_profile import ModerateProfile
from risk_management.risk_profiles.aggressive_profile import AggressiveProfile

class StrategyEvaluator:
    """
    Lớp đánh giá chiến lược giao dịch.
    Cung cấp phương thức phân tích và đánh giá chiến lược giao dịch
    dựa trên kết quả backtest để giúp xác định hiệu suất và xếp hạng
    các chiến lược khác nhau.
    """
    
    def __init__(
        self,
        backtest_results: Optional[Dict[str, Any]] = None,
        strategy_name: str = "unknown",
        output_dir: Optional[Path] = None,
        market_data: Optional[pd.DataFrame] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo đánh giá chiến lược.
        
        Args:
            backtest_results: Kết quả backtest
            strategy_name: Tên chiến lược
            output_dir: Thư mục đầu ra báo cáo đánh giá
            market_data: Dữ liệu thị trường dùng khi backtest
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("strategy_evaluator")
        
        # Lưu thông tin chiến lược
        self.strategy_name = strategy_name
        self.backtest_results = backtest_results
        self.market_data = market_data
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            self.output_dir = BACKTEST_DIR / 'evaluation' / strategy_name
        else:
            self.output_dir = Path(output_dir)
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo các evaluator khác
        self.performance_evaluator = PerformanceEvaluator(logger=self.logger)
        self.risk_evaluator = RiskEvaluator(logger=self.logger)
        
        # Biến lưu trữ kết quả đánh giá
        self.evaluation_results = {}
        
        # Định nghĩa danh sách các tiêu chí đánh giá
        self.evaluation_criteria = {
            'performance': [
                BacktestMetric.TOTAL_RETURN.value,
                BacktestMetric.ANNUALIZED_RETURN.value,
                BacktestMetric.SHARPE_RATIO.value,
                BacktestMetric.SORTINO_RATIO.value,
                BacktestMetric.CALMAR_RATIO.value,
                BacktestMetric.EXPECTANCY.value
            ],
            'risk': [
                BacktestMetric.MAX_DRAWDOWN.value,
                BacktestMetric.RISK_REWARD_RATIO.value,
                'volatility',
                'recovery_time',
                'max_consecutive_losses',
                'worst_trade'
            ],
            'consistency': [
                BacktestMetric.WIN_RATE.value,
                BacktestMetric.PROFIT_FACTOR.value,
                'profit_consistency',
                'trade_frequency',
                'average_holding_time',
                'trade_size_consistency'
            ],
            'adaptability': [
                'performance_in_bull',
                'performance_in_bear',
                'performance_in_sideways',
                'performance_in_high_volatility',
                'performance_in_low_volatility'
            ]
        }
        
        # Trọng số cho các tiêu chí
        self.criteria_weights = {
            'performance': 0.35,
            'risk': 0.30,
            'consistency': 0.20,
            'adaptability': 0.15
        }
        
        self.logger.info(f"Đã khởi tạo StrategyEvaluator cho chiến lược '{strategy_name}'")
    
    def set_backtest_results(self, backtest_results: Dict[str, Any]) -> None:
        """
        Cập nhật kết quả backtest.
        
        Args:
            backtest_results: Kết quả backtest mới
        """
        self.backtest_results = backtest_results
        self.logger.info(f"Đã cập nhật kết quả backtest cho chiến lược '{self.strategy_name}'")
    
    def set_market_data(self, market_data: pd.DataFrame) -> None:
        """
        Cập nhật dữ liệu thị trường.
        
        Args:
            market_data: Dữ liệu thị trường mới
        """
        self.market_data = market_data
        self.logger.info(f"Đã cập nhật dữ liệu thị trường cho chiến lược '{self.strategy_name}'")
    
    def evaluate_strategy(
        self,
        category: Optional[str] = None,
        detailed: bool = True,
        save_results: bool = True,
        risk_profile: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Đánh giá chiến lược giao dịch.
        
        Args:
            category: Loại đánh giá (performance, risk, consistency, adaptability, all)
            detailed: Thực hiện đánh giá chi tiết
            save_results: Lưu kết quả đánh giá
            risk_profile: Hồ sơ rủi ro để đánh giá (conservative, moderate, aggressive)
            
        Returns:
            Dict kết quả đánh giá
        """
        if self.backtest_results is None:
            self.logger.error("Không có kết quả backtest để đánh giá")
            return {"status": "error", "message": "Không có kết quả backtest để đánh giá"}
        
        # Xác định các loại đánh giá cần thực hiện
        if category is None or category.lower() == 'all':
            categories = list(self.evaluation_criteria.keys())
        else:
            if category.lower() in self.evaluation_criteria:
                categories = [category.lower()]
            else:
                self.logger.error(f"Loại đánh giá không hợp lệ: {category}")
                return {"status": "error", "message": f"Loại đánh giá không hợp lệ: {category}"}
        
        # Khởi tạo kết quả đánh giá
        evaluation_results = {
            "strategy_name": self.strategy_name,
            "evaluation_date": datetime.now().isoformat(),
            "metrics": {},
            "scores": {},
            "overall_score": 0.0,
            "risk_profile": risk_profile,
        }
        
        # Thực hiện đánh giá cho từng loại
        for category in categories:
            self.logger.info(f"Đang đánh giá chiến lược '{self.strategy_name}' - loại: {category}")
            
            if category == 'performance':
                metrics = self._evaluate_performance()
            elif category == 'risk':
                metrics = self._evaluate_risk(risk_profile)
            elif category == 'consistency':
                metrics = self._evaluate_consistency()
            elif category == 'adaptability':
                metrics = self._evaluate_adaptability()
            else:
                metrics = {}
            
            # Thêm metrics vào kết quả
            evaluation_results["metrics"][category] = metrics
            
            # Tính điểm cho loại đánh giá này
            score = self._calculate_category_score(category, metrics)
            evaluation_results["scores"][category] = score
        
        # Tính tổng điểm
        total_score = 0.0
        total_weight = 0.0
        
        for category, score in evaluation_results["scores"].items():
            weight = self.criteria_weights.get(category, 0.0)
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            evaluation_results["overall_score"] = total_score / total_weight
        
        # Đánh giá sự phù hợp với hồ sơ rủi ro
        if risk_profile is not None:
            risk_compatibility = self._evaluate_risk_profile_compatibility(risk_profile)
            evaluation_results["risk_profile_compatibility"] = risk_compatibility
        
        # Thêm xếp hạng tổng thể
        evaluation_results["rating"] = self._get_rating(evaluation_results["overall_score"])
        
        # Thêm đề xuất cải thiện
        evaluation_results["improvement_suggestions"] = self._suggest_improvements(evaluation_results)
        
        # Lưu kết quả đánh giá
        if save_results:
            self._save_evaluation_results(evaluation_results)
        
        # Lưu kết quả đánh giá vào biến instance
        self.evaluation_results = evaluation_results
        
        self.logger.info(f"Đã hoàn thành đánh giá chiến lược '{self.strategy_name}' - Điểm: {evaluation_results['overall_score']:.2f}/10, Xếp hạng: {evaluation_results['rating']}")
        
        return evaluation_results
    
    def _evaluate_performance(self) -> Dict[str, Any]:
        """
        Đánh giá hiệu suất của chiến lược.
        
        Returns:
            Dict chứa các chỉ số hiệu suất
        """
        # Trích xuất thông tin từ kết quả backtest
        metrics = self.backtest_results.get("metrics", {})
        trades = self.backtest_results.get("trades", {})
        balance_history = self.backtest_results.get("balance_history", {})
        
        # Tạo DataFrame từ lịch sử equity
        if balance_history:
            timestamps = balance_history.get("timestamp", [])
            equity = balance_history.get("equity", [])
            
            if timestamps and equity and len(timestamps) == len(equity):
                equity_df = pd.DataFrame({
                    "timestamp": pd.to_datetime(timestamps),
                    "equity": equity
                })
                equity_df.set_index("timestamp", inplace=True)
            else:
                equity_df = pd.DataFrame()
        else:
            equity_df = pd.DataFrame()
        
        # Sử dụng PerformanceEvaluator để tính các chỉ số hiệu suất
        performance_metrics = self.performance_evaluator.calculate_performance_metrics(
            trades=trades.get("closed_positions", []),
            metrics=metrics,
            equity_curve=equity_df
        )
        
        # Bổ sung thêm các chỉ số hiệu suất
        if equity_df.empty:
            annualized_volatility = None
        else:
            # Tính toán lợi nhuận hàng ngày
            daily_returns = equity_df["equity"].pct_change().dropna()
            
            if len(daily_returns) > 1:
                # Tính toán biến động hàng năm (annualized volatility)
                annualized_volatility = daily_returns.std() * np.sqrt(252)
            else:
                annualized_volatility = None
        
        # Bổ sung vào dict hiệu suất
        performance_metrics["annualized_volatility"] = annualized_volatility
        
        return performance_metrics
    
    def _evaluate_risk(self, risk_profile: Optional[str] = None) -> Dict[str, Any]:
        """
        Đánh giá rủi ro của chiến lược.
        
        Args:
            risk_profile: Hồ sơ rủi ro để so sánh
            
        Returns:
            Dict chứa các chỉ số rủi ro
        """
        # Trích xuất thông tin từ kết quả backtest
        metrics = self.backtest_results.get("metrics", {})
        trades = self.backtest_results.get("trades", {})
        balance_history = self.backtest_results.get("balance_history", {})
        closed_positions = trades.get("closed_positions", [])
        
        # Tạo DataFrame từ lịch sử equity
        if balance_history:
            timestamps = balance_history.get("timestamp", [])
            equity = balance_history.get("equity", [])
            
            if timestamps and equity and len(timestamps) == len(equity):
                equity_df = pd.DataFrame({
                    "timestamp": pd.to_datetime(timestamps),
                    "equity": equity
                })
                equity_df.set_index("timestamp", inplace=True)
            else:
                equity_df = pd.DataFrame()
        else:
            equity_df = pd.DataFrame()
        
        # Sử dụng RiskEvaluator để tính các chỉ số rủi ro
        risk_metrics = self.risk_evaluator.calculate_risk_metrics(
            trades=closed_positions,
            metrics=metrics,
            equity_curve=equity_df
        )
        
        # Tính toán thêm các chỉ số rủi ro chi tiết
        
        # 1. Tính số lần thua liên tiếp tối đa
        if closed_positions:
            # Sắp xếp các giao dịch theo thời gian
            sorted_trades = sorted(closed_positions, key=lambda x: x.get("exit_time", x.get("entry_time", "")))
            
            # Tìm chuỗi thua lỗ dài nhất
            max_losing_streak = 0
            current_streak = 0
            
            for trade in sorted_trades:
                if trade.get("realized_pnl", 0) < 0:
                    current_streak += 1
                    max_losing_streak = max(max_losing_streak, current_streak)
                else:
                    current_streak = 0
        else:
            max_losing_streak = 0
        
        # 2. Tìm giao dịch tệ nhất
        if closed_positions:
            worst_trade = min(closed_positions, key=lambda x: x.get("realized_pnl", 0))
            worst_trade_loss = worst_trade.get("realized_pnl", 0)
            worst_trade_percent = worst_trade.get("roi", 0) * 100  # Đổi sang phần trăm
        else:
            worst_trade_loss = 0
            worst_trade_percent = 0
        
        # 3. Tính thời gian phục hồi sau drawdown
        recovery_time_days = self.risk_evaluator.calculate_recovery_time(equity_df)
        
        # 4. Đánh giá phù hợp với hồ sơ rủi ro
        if risk_profile:
            risk_compatibility = self._evaluate_risk_profile_compatibility(risk_profile)
        else:
            risk_compatibility = None
        
        # Bổ sung thêm các chỉ số rủi ro vào kết quả
        risk_metrics.update({
            "max_consecutive_losses": max_losing_streak,
            "worst_trade_loss": worst_trade_loss,
            "worst_trade_percent": worst_trade_percent,
            "recovery_time_days": recovery_time_days,
            "risk_profile_compatibility": risk_compatibility
        })
        
        return risk_metrics
    
    def _evaluate_consistency(self) -> Dict[str, Any]:
        """
        Đánh giá tính nhất quán của chiến lược.
        
        Returns:
            Dict chứa các chỉ số tính nhất quán
        """
        # Trích xuất thông tin từ kết quả backtest
        metrics = self.backtest_results.get("metrics", {})
        trades = self.backtest_results.get("trades", {})
        balance_history = self.backtest_results.get("balance_history", {})
        closed_positions = trades.get("closed_positions", [])
        
        if not closed_positions:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "profit_consistency": 0.0,
                "trade_frequency": 0.0,
                "average_holding_time": 0.0,
                "trade_size_consistency": 0.0
            }
        
        # 1. Tính tỷ lệ lợi nhuận theo thời gian
        if balance_history:
            timestamps = balance_history.get("timestamp", [])
            equity = balance_history.get("equity", [])
            
            if timestamps and equity and len(timestamps) == len(equity):
                equity_df = pd.DataFrame({
                    "timestamp": pd.to_datetime(timestamps),
                    "equity": equity
                })
                equity_df.set_index("timestamp", inplace=True)
                
                # Tính lợi nhuận hàng ngày
                equity_df["daily_return"] = equity_df["equity"].pct_change()
                
                # Tính tỷ lệ ngày có lợi nhuận
                profitable_days = len(equity_df[equity_df["daily_return"] > 0])
                total_days = len(equity_df["daily_return"].dropna())
                
                if total_days > 0:
                    profit_consistency = profitable_days / total_days
                else:
                    profit_consistency = 0.0
            else:
                profit_consistency = 0.0
        else:
            profit_consistency = 0.0
        
        # 2. Tính tần suất giao dịch
        if closed_positions and len(closed_positions) >= 2:
            # Sắp xếp giao dịch theo thời gian
            sorted_trades = sorted(closed_positions, key=lambda x: pd.to_datetime(x.get("entry_time", "")))
            
            # Lấy thời gian bắt đầu và kết thúc
            start_time = pd.to_datetime(sorted_trades[0].get("entry_time", ""))
            end_time = pd.to_datetime(sorted_trades[-1].get("exit_time", ""))
            
            # Tính tổng số ngày
            total_days = (end_time - start_time).days
            
            if total_days > 0:
                trade_frequency = len(closed_positions) / total_days
            else:
                trade_frequency = len(closed_positions)
        else:
            trade_frequency = 0.0
        
        # 3. Tính thời gian giữ vị thế trung bình
        total_holding_time = 0
        trades_with_time = 0
        
        for trade in closed_positions:
            entry_time = pd.to_datetime(trade.get("entry_time", ""))
            exit_time = pd.to_datetime(trade.get("exit_time", ""))
            
            if pd.notna(entry_time) and pd.notna(exit_time):
                holding_time = (exit_time - entry_time).total_seconds() / 3600  # Giờ
                total_holding_time += holding_time
                trades_with_time += 1
        
        if trades_with_time > 0:
            average_holding_time = total_holding_time / trades_with_time
        else:
            average_holding_time = 0.0
        
        # 4. Tính độ nhất quán về kích thước giao dịch
        if closed_positions:
            position_sizes = [trade.get("position_size", 0) for trade in closed_positions]
            
            if position_sizes:
                mean_size = np.mean(position_sizes)
                std_size = np.std(position_sizes)
                
                if mean_size > 0:
                    # Coefficient of variation - càng thấp càng nhất quán
                    cv = std_size / mean_size
                    trade_size_consistency = 1.0 - min(cv, 1.0)  # Đảm bảo giá trị từ 0-1
                else:
                    trade_size_consistency = 0.0
            else:
                trade_size_consistency = 0.0
        else:
            trade_size_consistency = 0.0
        
        # Tạo dict kết quả
        consistency_metrics = {
            "win_rate": metrics.get(BacktestMetric.WIN_RATE.value, 0.0),
            "profit_factor": metrics.get(BacktestMetric.PROFIT_FACTOR.value, 0.0),
            "profit_consistency": profit_consistency,
            "trade_frequency": trade_frequency,
            "average_holding_time": average_holding_time,
            "trade_size_consistency": trade_size_consistency
        }
        
        return consistency_metrics
    
    def _evaluate_adaptability(self) -> Dict[str, Any]:
        """
        Đánh giá khả năng thích nghi của chiến lược với các điều kiện thị trường khác nhau.
        
        Returns:
            Dict chứa các chỉ số khả năng thích nghi
        """
        # Trích xuất thông tin từ kết quả backtest
        trades = self.backtest_results.get("trades", {})
        closed_positions = trades.get("closed_positions", [])
        
        # Kiểm tra xem có dữ liệu thị trường không
        if self.market_data is None or self.market_data.empty:
            self.logger.warning("Không có dữ liệu thị trường để đánh giá khả năng thích nghi")
            return {
                "performance_in_bull": None,
                "performance_in_bear": None,
                "performance_in_sideways": None,
                "performance_in_high_volatility": None,
                "performance_in_low_volatility": None
            }
        
        if not closed_positions:
            return {
                "performance_in_bull": 0.0,
                "performance_in_bear": 0.0,
                "performance_in_sideways": 0.0,
                "performance_in_high_volatility": 0.0,
                "performance_in_low_volatility": 0.0
            }
        
        # Phân loại điều kiện thị trường
        market_conditions = self._classify_market_conditions()
        
        # Tính ROI trung bình trong các điều kiện thị trường khác nhau
        roi_by_condition = {}
        
        for trade in closed_positions:
            # Lấy thời gian entry và exit
            entry_time = pd.to_datetime(trade.get("entry_time", ""))
            exit_time = pd.to_datetime(trade.get("exit_time", ""))
            
            if pd.isna(entry_time) or pd.isna(exit_time):
                continue
            
            # Tìm điều kiện thị trường vào thời điểm entry
            entry_date = entry_time.date()
            
            # Gán giao dịch vào các điều kiện thị trường
            for condition, dates in market_conditions.items():
                if entry_date in dates:
                    if condition not in roi_by_condition:
                        roi_by_condition[condition] = []
                    
                    roi_by_condition[condition].append(trade.get("roi", 0))
                    break
        
        # Tính ROI trung bình cho mỗi điều kiện
        adaptability_metrics = {}
        
        for condition in ["bull", "bear", "sideways", "high_volatility", "low_volatility"]:
            key = f"performance_in_{condition}"
            
            if condition in roi_by_condition and roi_by_condition[condition]:
                adaptability_metrics[key] = np.mean(roi_by_condition[condition])
            else:
                adaptability_metrics[key] = None
        
        return adaptability_metrics
    
    def _classify_market_conditions(self) -> Dict[str, List[datetime.date]]:
        """
        Phân loại các điều kiện thị trường dựa trên dữ liệu giá.
        
        Returns:
            Dict chứa các ngày tương ứng với mỗi điều kiện thị trường
        """
        # Đảm bảo dữ liệu thị trường có cột timestamp và close
        if self.market_data is None or "timestamp" not in self.market_data.columns or "close" not in self.market_data.columns:
            return {}
        
        # Chuẩn bị dữ liệu
        market_data = self.market_data.copy()
        market_data["timestamp"] = pd.to_datetime(market_data["timestamp"])
        market_data["date"] = market_data["timestamp"].dt.date
        market_data.set_index("timestamp", inplace=True)
        
        # Tính các chỉ số để phân loại
        
        # 1. Tính SMA 20 để xác định xu hướng
        market_data["sma20"] = market_data["close"].rolling(window=20).mean()
        
        # 2. Tính biến động 20 ngày
        market_data["volatility"] = market_data["close"].pct_change().rolling(window=20).std()
        
        # Phân loại các ngày theo điều kiện thị trường
        market_conditions = {
            "bull": [],
            "bear": [],
            "sideways": [],
            "high_volatility": [],
            "low_volatility": []
        }
        
        # Tính ngưỡng biến động
        median_volatility = market_data["volatility"].median()
        high_vol_threshold = market_data["volatility"].quantile(0.75)
        low_vol_threshold = market_data["volatility"].quantile(0.25)
        
        # Lọc bỏ dữ liệu NaN
        market_data = market_data.dropna(subset=["sma20", "volatility"])
        
        # Phân loại điều kiện thị trường
        for i in range(1, len(market_data)):
            current_row = market_data.iloc[i]
            prev_row = market_data.iloc[i-1]
            
            date = current_row.name.date()
            
            # Phân loại theo xu hướng
            if current_row["close"] > current_row["sma20"] and prev_row["close"] > prev_row["sma20"]:
                market_conditions["bull"].append(date)
            elif current_row["close"] < current_row["sma20"] and prev_row["close"] < prev_row["sma20"]:
                market_conditions["bear"].append(date)
            else:
                market_conditions["sideways"].append(date)
            
            # Phân loại theo biến động
            if current_row["volatility"] > high_vol_threshold:
                market_conditions["high_volatility"].append(date)
            elif current_row["volatility"] < low_vol_threshold:
                market_conditions["low_volatility"].append(date)
        
        return market_conditions
    
    def _calculate_category_score(self, category: str, metrics: Dict[str, Any]) -> float:
        """
        Tính điểm cho một loại đánh giá.
        
        Args:
            category: Loại đánh giá
            metrics: Các chỉ số đánh giá
            
        Returns:
            Điểm cho loại đánh giá (0-10)
        """
        if not metrics:
            return 0.0
        
        # Điểm tối đa cho mỗi chỉ số
        max_scores = {
            # Performance
            BacktestMetric.TOTAL_RETURN.value: 10.0,  # > 100% = 10 điểm
            BacktestMetric.ANNUALIZED_RETURN.value: 10.0,  # > 50% = 10 điểm
            BacktestMetric.SHARPE_RATIO.value: 10.0,  # > 3 = 10 điểm
            BacktestMetric.SORTINO_RATIO.value: 10.0,  # > 4 = 10 điểm
            BacktestMetric.CALMAR_RATIO.value: 10.0,  # > 5 = 10 điểm
            BacktestMetric.EXPECTANCY.value: 10.0,  # > 1.0 = 10 điểm
            "annualized_volatility": 10.0,  # < 0.1 = 10 điểm
            
            # Risk
            BacktestMetric.MAX_DRAWDOWN.value: 10.0,  # < 0.05 = 10 điểm
            BacktestMetric.RISK_REWARD_RATIO.value: 10.0,  # > 3 = 10 điểm
            "volatility": 10.0,  # < 0.1 = 10 điểm
            "recovery_time_days": 10.0,  # < 10 = 10 điểm
            "max_consecutive_losses": 10.0,  # < 3 = 10 điểm
            "worst_trade_percent": 10.0,  # > -2% = 10 điểm
            
            # Consistency
            BacktestMetric.WIN_RATE.value: 10.0,  # > 0.6 = 10 điểm
            BacktestMetric.PROFIT_FACTOR.value: 10.0,  # > 3 = 10 điểm
            "profit_consistency": 10.0,  # > 0.6 = 10 điểm
            "trade_frequency": 10.0,  # Tùy thuộc vào chiến lược
            "average_holding_time": 10.0,  # Tùy thuộc vào chiến lược
            "trade_size_consistency": 10.0,  # > 0.8 = 10 điểm
            
            # Adaptability
            "performance_in_bull": 10.0,  # > 0.1 = 10 điểm
            "performance_in_bear": 10.0,  # > 0.05 = 10 điểm
            "performance_in_sideways": 10.0,  # > 0.02 = 10 điểm
            "performance_in_high_volatility": 10.0,  # > 0.05 = 10 điểm
            "performance_in_low_volatility": 10.0,  # > 0.02 = 10 điểm
        }
        
        # Hàm tính điểm tùy theo chỉ số
        score_functions = {
            # Performance - Higher is better
            BacktestMetric.TOTAL_RETURN.value: lambda x: min(10.0, max(0.0, 10.0 * x / 1.0)) if x is not None else 0.0,
            BacktestMetric.ANNUALIZED_RETURN.value: lambda x: min(10.0, max(0.0, 10.0 * x / 0.5)) if x is not None else 0.0,
            BacktestMetric.SHARPE_RATIO.value: lambda x: min(10.0, max(0.0, 10.0 * x / 3.0)) if x is not None else 0.0,
            BacktestMetric.SORTINO_RATIO.value: lambda x: min(10.0, max(0.0, 10.0 * x / 4.0)) if x is not None else 0.0,
            BacktestMetric.CALMAR_RATIO.value: lambda x: min(10.0, max(0.0, 10.0 * x / 5.0)) if x is not None else 0.0,
            BacktestMetric.EXPECTANCY.value: lambda x: min(10.0, max(0.0, 10.0 * x / 1.0)) if x is not None else 0.0,
            "annualized_volatility": lambda x: min(10.0, max(0.0, 10.0 * (1.0 - x / 0.3))) if x is not None else 0.0,
            
            # Risk - Lower is better for some
            BacktestMetric.MAX_DRAWDOWN.value: lambda x: min(10.0, max(0.0, 10.0 * (1.0 - x / 0.3))) if x is not None else 0.0,
            BacktestMetric.RISK_REWARD_RATIO.value: lambda x: min(10.0, max(0.0, 10.0 * x / 3.0)) if x is not None else 0.0,
            "volatility": lambda x: min(10.0, max(0.0, 10.0 * (1.0 - x / 0.3))) if x is not None else 0.0,
            "recovery_time_days": lambda x: min(10.0, max(0.0, 10.0 * (1.0 - x / 60.0))) if x is not None else 0.0,
            "max_consecutive_losses": lambda x: min(10.0, max(0.0, 10.0 * (1.0 - x / 10.0))) if x is not None else 0.0,
            "worst_trade_percent": lambda x: min(10.0, max(0.0, 10.0 * (1.0 - abs(x) / 10.0))) if x is not None else 0.0,
            
            # Consistency
            BacktestMetric.WIN_RATE.value: lambda x: min(10.0, max(0.0, 10.0 * x / 0.6)) if x is not None else 0.0,
            BacktestMetric.PROFIT_FACTOR.value: lambda x: min(10.0, max(0.0, 10.0 * x / 3.0)) if x is not None else 0.0,
            "profit_consistency": lambda x: min(10.0, max(0.0, 10.0 * x / 0.6)) if x is not None else 0.0,
            "trade_frequency": lambda x: 5.0 if x is None else 5.0,  # Neutral score
            "average_holding_time": lambda x: 5.0 if x is None else 5.0,  # Neutral score
            "trade_size_consistency": lambda x: min(10.0, max(0.0, 10.0 * x / 0.8)) if x is not None else 0.0,
            
            # Adaptability
            "performance_in_bull": lambda x: min(10.0, max(0.0, 10.0 * x / 0.1)) if x is not None else 0.0,
            "performance_in_bear": lambda x: min(10.0, max(0.0, 10.0 * x / 0.05)) if x is not None else 0.0,
            "performance_in_sideways": lambda x: min(10.0, max(0.0, 10.0 * x / 0.02)) if x is not None else 0.0,
            "performance_in_high_volatility": lambda x: min(10.0, max(0.0, 10.0 * x / 0.05)) if x is not None else 0.0,
            "performance_in_low_volatility": lambda x: min(10.0, max(0.0, 10.0 * x / 0.02)) if x is not None else 0.0,
        }
        
        # Lấy danh sách chỉ số cho loại đánh giá
        criteria_list = self.evaluation_criteria.get(category, [])
        
        if not criteria_list:
            return 0.0
        
        # Tính điểm cho từng chỉ số
        total_score = 0.0
        valid_criteria = 0
        
        for criterion in criteria_list:
            if criterion in metrics and criterion in score_functions:
                value = metrics[criterion]
                
                if value is not None:
                    score = score_functions[criterion](value)
                    total_score += score
                    valid_criteria += 1
        
        # Tính điểm trung bình
        if valid_criteria > 0:
            avg_score = total_score / valid_criteria
        else:
            avg_score = 0.0
        
        return avg_score
    
    def _evaluate_risk_profile_compatibility(self, risk_profile: str) -> Dict[str, Any]:
        """
        Đánh giá mức độ phù hợp của chiến lược với hồ sơ rủi ro.
        
        Args:
            risk_profile: Tên hồ sơ rủi ro (conservative, moderate, aggressive)
            
        Returns:
            Dict chứa thông tin đánh giá
        """
        # Kiểm tra hồ sơ rủi ro hợp lệ
        if risk_profile not in ["conservative", "moderate", "aggressive"]:
            self.logger.warning(f"Hồ sơ rủi ro không hợp lệ: {risk_profile}")
            return {
                "profile": risk_profile,
                "compatibility_score": 0.0,
                "compatibility_rating": "Unknown",
                "reasons": ["Hồ sơ rủi ro không hợp lệ"]
            }
        
        # Trích xuất thông tin chiến lược từ kết quả backtest
        metrics = self.backtest_results.get("metrics", {})
        trades = self.backtest_results.get("trades", {})
        balance_history = self.backtest_results.get("balance_history", {})
        
        # Khởi tạo đối tượng hồ sơ rủi ro tương ứng
        if risk_profile == "conservative":
            profile = ConservativeProfile()
        elif risk_profile == "moderate":
            profile = ModerateProfile()
        else:  # aggressive
            profile = AggressiveProfile()
        
        # Lấy cấu hình hồ sơ rủi ro
        profile_config = profile.get_risk_profile_config()
        
        # So sánh các chỉ số của chiến lược với thông số của hồ sơ rủi ro
        compatibility_scores = []
        compatibility_reasons = []
        
        # 1. So sánh max_drawdown
        strategy_max_drawdown = abs(metrics.get(BacktestMetric.MAX_DRAWDOWN.value, 0.0))
        profile_max_drawdown = profile_config.get("max_drawdown", 0.15)
        
        if strategy_max_drawdown <= profile_max_drawdown:
            compatibility_scores.append(1.0)
            compatibility_reasons.append(f"Mức drawdown tối đa ({strategy_max_drawdown:.1%}) nằm trong giới hạn cho phép ({profile_max_drawdown:.1%})")
        else:
            ratio = profile_max_drawdown / strategy_max_drawdown
            compatibility_scores.append(max(0.0, ratio))
            compatibility_reasons.append(f"Mức drawdown tối đa ({strategy_max_drawdown:.1%}) vượt quá giới hạn cho phép ({profile_max_drawdown:.1%})")
        
        # 2. So sánh risk_per_trade
        # Tính average_loss_percent từ các giao dịch
        closed_positions = trades.get("closed_positions", [])
        losing_trades = [t for t in closed_positions if t.get("realized_pnl", 0) < 0]
        
        if losing_trades:
            avg_loss_percent = sum([abs(t.get("roi", 0)) for t in losing_trades]) / len(losing_trades)
        else:
            avg_loss_percent = 0.0
        
        profile_risk_per_trade = profile_config.get("risk_per_trade", 0.02)
        
        if avg_loss_percent <= profile_risk_per_trade:
            compatibility_scores.append(1.0)
            compatibility_reasons.append(f"Mức rủi ro trung bình trên mỗi giao dịch ({avg_loss_percent:.1%}) phù hợp với giới hạn ({profile_risk_per_trade:.1%})")
        else:
            ratio = profile_risk_per_trade / avg_loss_percent
            compatibility_scores.append(max(0.0, ratio))
            compatibility_reasons.append(f"Mức rủi ro trung bình trên mỗi giao dịch ({avg_loss_percent:.1%}) cao hơn giới hạn ({profile_risk_per_trade:.1%})")
        
        # 3. So sánh win_rate
        strategy_win_rate = metrics.get(BacktestMetric.WIN_RATE.value, 0.0)
        min_win_rate = 0.4  # Mức tối thiểu cho mọi hồ sơ
        
        if strategy_win_rate >= min_win_rate:
            compatibility_scores.append(1.0)
            compatibility_reasons.append(f"Tỷ lệ thắng ({strategy_win_rate:.1%}) đạt mức tối thiểu yêu cầu")
        else:
            ratio = strategy_win_rate / min_win_rate
            compatibility_scores.append(max(0.0, ratio))
            compatibility_reasons.append(f"Tỷ lệ thắng ({strategy_win_rate:.1%}) thấp hơn mức tối thiểu yêu cầu ({min_win_rate:.1%})")
        
        # 4. So sánh profit_factor
        strategy_profit_factor = metrics.get(BacktestMetric.PROFIT_FACTOR.value, 0.0)
        min_profit_factor = 1.2  # Mức tối thiểu cho mọi hồ sơ
        
        if strategy_profit_factor >= min_profit_factor:
            compatibility_scores.append(1.0)
            compatibility_reasons.append(f"Profit factor ({strategy_profit_factor:.2f}) đạt mức tối thiểu yêu cầu")
        else:
            ratio = strategy_profit_factor / min_profit_factor
            compatibility_scores.append(max(0.0, ratio))
            compatibility_reasons.append(f"Profit factor ({strategy_profit_factor:.2f}) thấp hơn mức tối thiểu yêu cầu ({min_profit_factor:.2f})")
        
        # 5. So sánh max_consecutive_losses
        if hasattr(self, "evaluation_results") and self.evaluation_results:
            max_consecutive_losses = self.evaluation_results.get("metrics", {}).get("risk", {}).get("max_consecutive_losses", 0)
        else:
            # Tính toán nếu chưa có kết quả đánh giá
            max_consecutive_losses = 0
            current_streak = 0
            
            for trade in sorted(closed_positions, key=lambda x: x.get("exit_time", "")):
                if trade.get("realized_pnl", 0) < 0:
                    current_streak += 1
                    max_consecutive_losses = max(max_consecutive_losses, current_streak)
                else:
                    current_streak = 0
        
        max_allowed_losses = 3 if risk_profile == "conservative" else (4 if risk_profile == "moderate" else 5)
        
        if max_consecutive_losses <= max_allowed_losses:
            compatibility_scores.append(1.0)
            compatibility_reasons.append(f"Số lần thua liên tiếp tối đa ({max_consecutive_losses}) trong giới hạn cho phép ({max_allowed_losses})")
        else:
            ratio = max_allowed_losses / max_consecutive_losses
            compatibility_scores.append(max(0.0, ratio))
            compatibility_reasons.append(f"Số lần thua liên tiếp tối đa ({max_consecutive_losses}) vượt quá giới hạn cho phép ({max_allowed_losses})")
        
        # Tính điểm tổng thể
        if compatibility_scores:
            compatibility_score = sum(compatibility_scores) / len(compatibility_scores)
        else:
            compatibility_score = 0.0
        
        # Xếp hạng mức độ phù hợp
        if compatibility_score >= 0.9:
            compatibility_rating = "Excellent"
        elif compatibility_score >= 0.7:
            compatibility_rating = "Good"
        elif compatibility_score >= 0.5:
            compatibility_rating = "Moderate"
        elif compatibility_score >= 0.3:
            compatibility_rating = "Poor"
        else:
            compatibility_rating = "Very Poor"
        
        return {
            "profile": risk_profile,
            "compatibility_score": compatibility_score,
            "compatibility_rating": compatibility_rating,
            "reasons": compatibility_reasons
        }
    
    def _get_rating(self, score: float) -> str:
        """
        Chuyển đổi điểm số thành xếp hạng.
        
        Args:
            score: Điểm số (0-10)
            
        Returns:
            Xếp hạng dưới dạng chuỗi
        """
        if score >= 9.0:
            return "A+ (Xuất sắc)"
        elif score >= 8.0:
            return "A (Rất tốt)"
        elif score >= 7.0:
            return "B+ (Tốt)"
        elif score >= 6.0:
            return "B (Khá tốt)"
        elif score >= 5.0:
            return "C+ (Trung bình khá)"
        elif score >= 4.0:
            return "C (Trung bình)"
        elif score >= 3.0:
            return "D+ (Dưới trung bình)"
        elif score >= 2.0:
            return "D (Kém)"
        else:
            return "F (Rất kém)"
    
    def _suggest_improvements(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """
        Đề xuất các cải thiện cho chiến lược dựa trên kết quả đánh giá.
        
        Args:
            evaluation_results: Kết quả đánh giá
            
        Returns:
            Danh sách các đề xuất cải thiện
        """
        suggestions = []
        
        # Lấy điểm từng hạng mục
        scores = evaluation_results.get("scores", {})
        metrics = evaluation_results.get("metrics", {})
        
        # 1. Kiểm tra hiệu suất
        performance_score = scores.get("performance", 0.0)
        if performance_score < 5.0:
            suggestions.append("Cải thiện hiệu suất tổng thể bằng cách tối ưu hóa quy tắc vào lệnh và chọn cơ hội giao dịch tốt hơn.")
        
        # 2. Kiểm tra rủi ro
        risk_score = scores.get("risk", 0.0)
        if risk_score < 5.0:
            risk_metrics = metrics.get("risk", {})
            
            # Kiểm tra drawdown
            max_drawdown = risk_metrics.get(BacktestMetric.MAX_DRAWDOWN.value)
            if max_drawdown and max_drawdown > 0.2:
                suggestions.append(f"Giảm drawdown tối đa (hiện tại: {max_drawdown:.1%}) bằng cách cải thiện quản lý rủi ro và đặt stop loss chặt chẽ hơn.")
            
            # Kiểm tra các lần thua liên tiếp
            max_consecutive_losses = risk_metrics.get("max_consecutive_losses")
            if max_consecutive_losses and max_consecutive_losses > 5:
                suggestions.append(f"Giảm số lần thua liên tiếp (hiện tại: {max_consecutive_losses}) bằng cách cải thiện bộ lọc vào lệnh và tối ưu hóa điểm vào lệnh.")
        
        # 3. Kiểm tra tính nhất quán
        consistency_score = scores.get("consistency", 0.0)
        if consistency_score < 5.0:
            consistency_metrics = metrics.get("consistency", {})
            
            # Kiểm tra tỷ lệ thắng
            win_rate = consistency_metrics.get(BacktestMetric.WIN_RATE.value)
            if win_rate and win_rate < 0.5:
                suggestions.append(f"Cải thiện tỷ lệ thắng (hiện tại: {win_rate:.1%}) bằng cách tối ưu hóa bộ lọc vào lệnh và điểm vào lệnh.")
            
            # Kiểm tra profit factor
            profit_factor = consistency_metrics.get(BacktestMetric.PROFIT_FACTOR.value)
            if profit_factor and profit_factor < 1.5:
                suggestions.append(f"Tăng profit factor (hiện tại: {profit_factor:.2f}) bằng cách điều chỉnh tỷ lệ risk/reward cao hơn và cải thiện chiến lược take profit.")
        
        # 4. Kiểm tra khả năng thích nghi
        adaptability_score = scores.get("adaptability", 0.0)
        if adaptability_score < 5.0:
            adaptability_metrics = metrics.get("adaptability", {})
            
            # Kiểm tra hiệu suất trong thị trường giảm
            bear_performance = adaptability_metrics.get("performance_in_bear")
            if bear_performance and bear_performance < 0:
                suggestions.append("Cải thiện hiệu suất trong thị trường giảm giá bằng cách phát triển chiến lược short hoặc bổ sung bộ lọc để tránh giao dịch trong xu hướng giảm mạnh.")
            
            # Kiểm tra hiệu suất trong biến động cao
            high_vol_performance = adaptability_metrics.get("performance_in_high_volatility")
            if high_vol_performance and high_vol_performance < 0:
                suggestions.append("Cải thiện hiệu suất trong thị trường biến động cao bằng cách điều chỉnh kích thước vị thế và sử dụng stop loss động.")
        
        # 5. Kiểm tra tổng thể
        overall_score = evaluation_results.get("overall_score", 0.0)
        if overall_score < 4.0:
            suggestions.append("Xem xét cải thiện toàn diện chiến lược bằng cách kết hợp với các chỉ báo hoặc phương pháp phân tích khác.")
        
        # 6. Kiểm tra sự phù hợp với hồ sơ rủi ro
        if "risk_profile_compatibility" in evaluation_results:
            compatibility = evaluation_results["risk_profile_compatibility"]
            profile = compatibility.get("profile")
            score = compatibility.get("compatibility_score", 0.0)
            
            if score < 0.6:
                suggestions.append(f"Điều chỉnh chiến lược để phù hợp hơn với hồ sơ rủi ro {profile} bằng cách tuân thủ các giới hạn rủi ro tương ứng.")
        
        # Nếu không có đề xuất nào, thêm một đề xuất chung
        if not suggestions:
            suggestions.append("Chiến lược đang hoạt động tốt. Xem xét tối ưu hóa thêm để đạt hiệu suất cao hơn.")
        
        return suggestions
    
    def _save_evaluation_results(self, evaluation_results: Dict[str, Any]) -> None:
        """
        Lưu kết quả đánh giá vào file.
        
        Args:
            evaluation_results: Kết quả đánh giá
        """
        # Tạo tên file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{self.strategy_name}_evaluation_{timestamp}.json"
        file_path = self.output_dir / file_name
        
        # Tạo bản sao để xử lý
        results_to_save = evaluation_results.copy()
        
        # Chuyển đổi các kiểu dữ liệu không hỗ trợ JSON
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (pd.DataFrame)):
                return obj.to_dict()
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            else:
                return obj
        
        # Xử lý từng khóa trong dict
        for key, value in results_to_save.items():
            if isinstance(value, dict):
                # Xử lý các dict con
                for sub_key, sub_value in value.items():
                    results_to_save[key][sub_key] = convert_to_serializable(sub_value)
            else:
                results_to_save[key] = convert_to_serializable(value)
        
        # Lưu vào file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results_to_save, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Đã lưu kết quả đánh giá vào {file_path}")
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu kết quả đánh giá: {str(e)}")
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Tạo báo cáo đánh giá toàn diện.
        
        Returns:
            Dict chứa thông tin báo cáo
        """
        if not self.evaluation_results:
            self.logger.error("Chưa có kết quả đánh giá để tạo báo cáo")
            return {"status": "error", "message": "Chưa có kết quả đánh giá để tạo báo cáo"}
        
        # Tạo báo cáo dựa trên kết quả đánh giá
        report = {
            "title": f"Báo cáo đánh giá chiến lược '{self.strategy_name}'",
            "generation_time": datetime.now().isoformat(),
            "strategy_name": self.strategy_name,
            "overall_rating": self.evaluation_results.get("rating", "N/A"),
            "overall_score": self.evaluation_results.get("overall_score", 0.0),
            "summary": self._generate_report_summary(),
            "performance_analysis": self._generate_performance_analysis(),
            "risk_analysis": self._generate_risk_analysis(),
            "consistency_analysis": self._generate_consistency_analysis(),
            "adaptability_analysis": self._generate_adaptability_analysis(),
            "risk_profile_analysis": self._generate_risk_profile_analysis(),
            "improvement_suggestions": self.evaluation_results.get("improvement_suggestions", []),
            "comparison_with_benchmarks": self._generate_benchmark_comparison()
        }
        
        # Lưu báo cáo
        self._save_report(report)
        
        return report
    
    def _generate_report_summary(self) -> str:
        """
        Tạo tóm tắt báo cáo.
        
        Returns:
            Chuỗi tóm tắt
        """
        if not self.evaluation_results:
            return "Không có dữ liệu đánh giá."
        
        overall_score = self.evaluation_results.get("overall_score", 0.0)
        rating = self.evaluation_results.get("rating", "N/A")
        
        scores = self.evaluation_results.get("scores", {})
        performance_score = scores.get("performance", 0.0)
        risk_score = scores.get("risk", 0.0)
        consistency_score = scores.get("consistency", 0.0)
        adaptability_score = scores.get("adaptability", 0.0)
        
        metrics = self.evaluation_results.get("metrics", {})
        performance_metrics = metrics.get("performance", {})
        
        # Lấy một số chỉ số quan trọng
        total_return = performance_metrics.get(BacktestMetric.TOTAL_RETURN.value, 0.0)
        max_drawdown = performance_metrics.get(BacktestMetric.MAX_DRAWDOWN.value, 0.0)
        sharpe_ratio = performance_metrics.get(BacktestMetric.SHARPE_RATIO.value, 0.0)
        
        summary = (
            f"Chiến lược '{self.strategy_name}' đạt điểm tổng thể {overall_score:.2f}/10, xếp hạng {rating}. "
            f"Chiến lược có tổng lợi nhuận {total_return:.2%}, drawdown tối đa {max_drawdown:.2%}, "
            f"và Sharpe Ratio {sharpe_ratio:.2f}. "
            f"Điểm từng hạng mục: Hiệu suất: {performance_score:.2f}, Rủi ro: {risk_score:.2f}, "
            f"Tính nhất quán: {consistency_score:.2f}, Khả năng thích nghi: {adaptability_score:.2f}. "
        )
        
        return summary
    
    def _generate_performance_analysis(self) -> Dict[str, Any]:
        """
        Tạo phân tích hiệu suất.
        
        Returns:
            Dict chứa thông tin phân tích hiệu suất
        """
        if not self.evaluation_results:
            return {"status": "error", "message": "Không có dữ liệu đánh giá."}
        
        metrics = self.evaluation_results.get("metrics", {})
        performance_metrics = metrics.get("performance", {})
        
        # Lấy các chỉ số chính về hiệu suất
        analysis = {
            "total_return": performance_metrics.get(BacktestMetric.TOTAL_RETURN.value, 0.0),
            "annualized_return": performance_metrics.get(BacktestMetric.ANNUALIZED_RETURN.value, 0.0),
            "sharpe_ratio": performance_metrics.get(BacktestMetric.SHARPE_RATIO.value, 0.0),
            "sortino_ratio": performance_metrics.get(BacktestMetric.SORTINO_RATIO.value, 0.0),
            "calmar_ratio": performance_metrics.get(BacktestMetric.CALMAR_RATIO.value, 0.0),
            "expectancy": performance_metrics.get(BacktestMetric.EXPECTANCY.value, 0.0),
            "annualized_volatility": performance_metrics.get("annualized_volatility", 0.0)
        }
        
        # Đánh giá các chỉ số
        analysis["return_assessment"] = self._assess_metric(
            analysis["total_return"],
            "Lợi nhuận tổng thể",
            thresholds=[0.05, 0.1, 0.2, 0.5, 1.0],
            assessments=["Rất kém", "Kém", "Trung bình", "Tốt", "Rất tốt", "Xuất sắc"],
            higher_is_better=True
        )
        
        analysis["sharpe_assessment"] = self._assess_metric(
            analysis["sharpe_ratio"],
            "Sharpe Ratio",
            thresholds=[0.5, 1.0, 1.5, 2.0, 3.0],
            assessments=["Rất kém", "Kém", "Trung bình", "Tốt", "Rất tốt", "Xuất sắc"],
            higher_is_better=True
        )
        
        analysis["sortino_assessment"] = self._assess_metric(
            analysis["sortino_ratio"],
            "Sortino Ratio",
            thresholds=[0.75, 1.25, 2.0, 2.5, 3.5],
            assessments=["Rất kém", "Kém", "Trung bình", "Tốt", "Rất tốt", "Xuất sắc"],
            higher_is_better=True
        )
        
        analysis["calmar_assessment"] = self._assess_metric(
            analysis["calmar_ratio"],
            "Calmar Ratio",
            thresholds=[0.5, 1.0, 2.0, 3.0, 5.0],
            assessments=["Rất kém", "Kém", "Trung bình", "Tốt", "Rất tốt", "Xuất sắc"],
            higher_is_better=True
        )
        
        # Tạo tóm tắt hiệu suất
        analysis["summary"] = (
            f"Chiến lược đạt tổng lợi nhuận {analysis['total_return']:.2%} "
            f"({analysis['annualized_return']:.2%} hàng năm), với Sharpe Ratio {analysis['sharpe_ratio']:.2f} "
            f"và Sortino Ratio {analysis['sortino_ratio']:.2f}. "
            f"Expectancy của chiến lược là {analysis['expectancy']:.4f}, thể hiện giá trị kỳ vọng trung bình "
            f"cho mỗi giao dịch. Biến động hàng năm ở mức {analysis['annualized_volatility']:.2%}."
        )
        
        return analysis
    
    def _generate_risk_analysis(self) -> Dict[str, Any]:
        """
        Tạo phân tích rủi ro.
        
        Returns:
            Dict chứa thông tin phân tích rủi ro
        """
        if not self.evaluation_results:
            return {"status": "error", "message": "Không có dữ liệu đánh giá."}
        
        metrics = self.evaluation_results.get("metrics", {})
        risk_metrics = metrics.get("risk", {})
        
        # Lấy các chỉ số chính về rủi ro
        analysis = {
            "max_drawdown": risk_metrics.get(BacktestMetric.MAX_DRAWDOWN.value, 0.0),
            "risk_reward_ratio": risk_metrics.get(BacktestMetric.RISK_REWARD_RATIO.value, 0.0),
            "recovery_time_days": risk_metrics.get("recovery_time_days", 0),
            "max_consecutive_losses": risk_metrics.get("max_consecutive_losses", 0),
            "worst_trade_loss": risk_metrics.get("worst_trade_loss", 0.0),
            "worst_trade_percent": risk_metrics.get("worst_trade_percent", 0.0)
        }
        
        # Đánh giá các chỉ số
        analysis["drawdown_assessment"] = self._assess_metric(
            analysis["max_drawdown"],
            "Drawdown tối đa",
            thresholds=[0.05, 0.1, 0.15, 0.2, 0.3],
            assessments=["Xuất sắc", "Rất tốt", "Tốt", "Trung bình", "Kém", "Rất kém"],
            higher_is_better=False
        )
        
        analysis["risk_reward_assessment"] = self._assess_metric(
            analysis["risk_reward_ratio"],
            "Tỷ lệ Risk/Reward",
            thresholds=[1.0, 1.5, 2.0, 2.5, 3.0],
            assessments=["Rất kém", "Kém", "Trung bình", "Tốt", "Rất tốt", "Xuất sắc"],
            higher_is_better=True
        )
        
        analysis["recovery_time_assessment"] = self._assess_metric(
            analysis["recovery_time_days"],
            "Thời gian phục hồi",
            thresholds=[10, 20, 30, 60, 90],
            assessments=["Xuất sắc", "Rất tốt", "Tốt", "Trung bình", "Kém", "Rất kém"],
            higher_is_better=False
        )
        
        analysis["consecutive_losses_assessment"] = self._assess_metric(
            analysis["max_consecutive_losses"],
            "Số lần thua liên tiếp tối đa",
            thresholds=[3, 5, 7, 10, 15],
            assessments=["Xuất sắc", "Rất tốt", "Tốt", "Trung bình", "Kém", "Rất kém"],
            higher_is_better=False
        )
        
        # Tạo tóm tắt rủi ro
        analysis["summary"] = (
            f"Chiến lược có drawdown tối đa {analysis['max_drawdown']:.2%} với thời gian phục hồi "
            f"{analysis['recovery_time_days']} ngày. Tỷ lệ risk/reward là {analysis['risk_reward_ratio']:.2f}. "
            f"Chiến lược đã trải qua tối đa {analysis['max_consecutive_losses']} lần thua liên tiếp, "
            f"với giao dịch thua lỗ lớn nhất là {analysis['worst_trade_percent']:.2%}."
        )
        
        return analysis
    
    def _generate_consistency_analysis(self) -> Dict[str, Any]:
        """
        Tạo phân tích tính nhất quán.
        
        Returns:
            Dict chứa thông tin phân tích tính nhất quán
        """
        if not self.evaluation_results:
            return {"status": "error", "message": "Không có dữ liệu đánh giá."}
        
        metrics = self.evaluation_results.get("metrics", {})
        consistency_metrics = metrics.get("consistency", {})
        
        # Lấy các chỉ số chính về tính nhất quán
        analysis = {
            "win_rate": consistency_metrics.get(BacktestMetric.WIN_RATE.value, 0.0),
            "profit_factor": consistency_metrics.get(BacktestMetric.PROFIT_FACTOR.value, 0.0),
            "profit_consistency": consistency_metrics.get("profit_consistency", 0.0),
            "trade_frequency": consistency_metrics.get("trade_frequency", 0.0),
            "average_holding_time": consistency_metrics.get("average_holding_time", 0.0),
            "trade_size_consistency": consistency_metrics.get("trade_size_consistency", 0.0)
        }
        
        # Đánh giá các chỉ số
        analysis["win_rate_assessment"] = self._assess_metric(
            analysis["win_rate"],
            "Tỷ lệ thắng",
            thresholds=[0.4, 0.45, 0.5, 0.55, 0.6],
            assessments=["Rất kém", "Kém", "Trung bình", "Tốt", "Rất tốt", "Xuất sắc"],
            higher_is_better=True
        )
        
        analysis["profit_factor_assessment"] = self._assess_metric(
            analysis["profit_factor"],
            "Profit Factor",
            thresholds=[1.1, 1.3, 1.5, 2.0, 3.0],
            assessments=["Rất kém", "Kém", "Trung bình", "Tốt", "Rất tốt", "Xuất sắc"],
            higher_is_better=True
        )
        
        analysis["profit_consistency_assessment"] = self._assess_metric(
            analysis["profit_consistency"],
            "Tính nhất quán lợi nhuận",
            thresholds=[0.4, 0.45, 0.5, 0.55, 0.6],
            assessments=["Rất kém", "Kém", "Trung bình", "Tốt", "Rất tốt", "Xuất sắc"],
            higher_is_better=True
        )
        
        # Tạo tóm tắt tính nhất quán
        analysis["summary"] = (
            f"Chiến lược có tỷ lệ thắng {analysis['win_rate']:.2%} và profit factor {analysis['profit_factor']:.2f}. "
            f"Tính nhất quán lợi nhuận ở mức {analysis['profit_consistency']:.2%}, với tần suất giao dịch trung bình "
            f"{analysis['trade_frequency']:.2f} giao dịch/ngày và thời gian giữ vị thế trung bình "
            f"{analysis['average_holding_time']:.2f} giờ."
        )
        
        return analysis
    
    def _generate_adaptability_analysis(self) -> Dict[str, Any]:
        """
        Tạo phân tích khả năng thích nghi.
        
        Returns:
            Dict chứa thông tin phân tích khả năng thích nghi
        """
        if not self.evaluation_results:
            return {"status": "error", "message": "Không có dữ liệu đánh giá."}
        
        metrics = self.evaluation_results.get("metrics", {})
        adaptability_metrics = metrics.get("adaptability", {})
        
        # Lấy các chỉ số chính về khả năng thích nghi
        analysis = {
            "performance_in_bull": adaptability_metrics.get("performance_in_bull", None),
            "performance_in_bear": adaptability_metrics.get("performance_in_bear", None),
            "performance_in_sideways": adaptability_metrics.get("performance_in_sideways", None),
            "performance_in_high_volatility": adaptability_metrics.get("performance_in_high_volatility", None),
            "performance_in_low_volatility": adaptability_metrics.get("performance_in_low_volatility", None)
        }
        
        # Xác định môi trường thị trường mạnh nhất và yếu nhất
        valid_performances = {k: v for k, v in analysis.items() if v is not None}
        
        if valid_performances:
            strongest_market = max(valid_performances, key=valid_performances.get)
            weakest_market = min(valid_performances, key=valid_performances.get)
            
            # Chuyển đổi tên kỹ thuật thành tên hiển thị
            market_display_names = {
                "performance_in_bull": "thị trường tăng",
                "performance_in_bear": "thị trường giảm",
                "performance_in_sideways": "thị trường đi ngang",
                "performance_in_high_volatility": "biến động cao",
                "performance_in_low_volatility": "biến động thấp"
            }
            
            analysis["strongest_market"] = {
                "type": market_display_names.get(strongest_market, strongest_market),
                "performance": valid_performances[strongest_market]
            }
            
            analysis["weakest_market"] = {
                "type": market_display_names.get(weakest_market, weakest_market),
                "performance": valid_performances[weakest_market]
            }
        else:
            analysis["strongest_market"] = None
            analysis["weakest_market"] = None
        
        # Đánh giá khả năng thích nghi tổng thể
        # Đếm số loại thị trường có hiệu suất tốt (ROI > 0)
        num_positive_markets = sum(1 for v in valid_performances.values() if v is not None and v > 0)
        num_markets = sum(1 for v in valid_performances.values() if v is not None)
        
        if num_markets > 0:
            adaptability_score = num_positive_markets / num_markets
            
            if adaptability_score >= 0.8:
                adaptability_rating = "Xuất sắc"
            elif adaptability_score >= 0.6:
                adaptability_rating = "Rất tốt"
            elif adaptability_score >= 0.4:
                adaptability_rating = "Tốt"
            elif adaptability_score >= 0.2:
                adaptability_rating = "Trung bình"
            else:
                adaptability_rating = "Kém"
                
            analysis["adaptability_score"] = adaptability_score
            analysis["adaptability_rating"] = adaptability_rating
        else:
            analysis["adaptability_score"] = None
            analysis["adaptability_rating"] = "Không thể đánh giá"
        
        # Tạo tóm tắt khả năng thích nghi
        if analysis["strongest_market"] and analysis["weakest_market"]:
            analysis["summary"] = (
                f"Chiến lược hoạt động tốt nhất trong {analysis['strongest_market']['type']} "
                f"(ROI: {analysis['strongest_market']['performance']:.2%}) và kém nhất trong "
                f"{analysis['weakest_market']['type']} (ROI: {analysis['weakest_market']['performance']:.2%}). "
                f"Khả năng thích nghi tổng thể được đánh giá ở mức {analysis['adaptability_rating']}."
            )
        else:
            analysis["summary"] = "Không đủ dữ liệu để đánh giá khả năng thích nghi."
        
        return analysis
    
    def _generate_risk_profile_analysis(self) -> Dict[str, Any]:
        """
        Tạo phân tích sự phù hợp với hồ sơ rủi ro.
        
        Returns:
            Dict chứa thông tin phân tích hồ sơ rủi ro
        """
        if not self.evaluation_results or "risk_profile_compatibility" not in self.evaluation_results:
            return {"status": "error", "message": "Không có dữ liệu đánh giá hồ sơ rủi ro."}
        
        compatibility = self.evaluation_results["risk_profile_compatibility"]
        
        profile = compatibility.get("profile")
        score = compatibility.get("compatibility_score", 0.0)
        rating = compatibility.get("compatibility_rating", "Unknown")
        reasons = compatibility.get("reasons", [])
        
        # Tạo đánh giá tóm tắt
        if profile == "conservative":
            profile_description = "hồ sơ thận trọng (conservative), tập trung vào bảo toàn vốn và giảm thiểu rủi ro"
        elif profile == "moderate":
            profile_description = "hồ sơ vừa phải (moderate), cân bằng giữa rủi ro và lợi nhuận"
        elif profile == "aggressive":
            profile_description = "hồ sơ tích cực (aggressive), ưu tiên lợi nhuận cao với rủi ro lớn hơn"
        else:
            profile_description = f"hồ sơ {profile}"
        
        # Tạo đề xuất
        if score >= 0.8:
            suggestion = f"Chiến lược rất phù hợp với {profile_description}."
        elif score >= 0.6:
            suggestion = f"Chiến lược phù hợp với {profile_description}."
        elif score >= 0.4:
            suggestion = f"Chiến lược có thể sử dụng với {profile_description} sau khi điều chỉnh một số tham số."
        else:
            suggestion = f"Chiến lược không phù hợp với {profile_description}. Xem xét sử dụng hồ sơ rủi ro khác hoặc điều chỉnh chiến lược."
        
        analysis = {
            "profile": profile,
            "compatibility_score": score,
            "compatibility_rating": rating,
            "profile_description": profile_description,
            "suggestion": suggestion,
            "reasons": reasons,
            "summary": (
                f"Chiến lược có mức độ phù hợp {score:.1%} ({rating}) với {profile_description}. {suggestion}"
            )
        }
        
        return analysis
    
    def _generate_benchmark_comparison(self) -> Dict[str, Any]:
        """
        Tạo so sánh với benchmark.
        
        Returns:
            Dict chứa thông tin so sánh benchmark
        """
        # Lưu ý: Phần này giả định không có dữ liệu benchmark thực tế,
        # nhưng có thể được mở rộng để so sánh với S&P500, BTC-HODL, v.v.
        
        # So sánh với chiến lược HODL
        # Giả sử có dữ liệu thị trường từ đối tượng self.market_data
        if self.market_data is None or self.market_data.empty:
            return {"status": "error", "message": "Không có dữ liệu thị trường để so sánh."}
        
        if not self.backtest_results:
            return {"status": "error", "message": "Không có kết quả backtest để so sánh."}
        
        try:
            # Tính lợi nhuận HODL
            market_data = self.market_data.copy()
            if "timestamp" in market_data.columns and "close" in market_data.columns:
                market_data["timestamp"] = pd.to_datetime(market_data["timestamp"])
                market_data = market_data.sort_values("timestamp")
                
                first_price = market_data["close"].iloc[0]
                last_price = market_data["close"].iloc[-1]
                
                hodl_return = (last_price - first_price) / first_price
                
                # Lấy lợi nhuận chiến lược
                strategy_return = self.backtest_results.get("roi", 0.0)
                
                # So sánh hiệu suất
                outperformance = strategy_return - hodl_return
                
                # Tính lợi nhuận theo thời gian
                first_date = market_data["timestamp"].iloc[0]
                last_date = market_data["timestamp"].iloc[-1]
                trading_days = (last_date - first_date).days
                
                if trading_days > 0:
                    # Tính lợi nhuận hàng năm
                    hodl_annual_return = ((1 + hodl_return) ** (365 / trading_days)) - 1
                    strategy_annual_return = self.backtest_results.get("metrics", {}).get(
                        BacktestMetric.ANNUALIZED_RETURN.value, 0.0
                    )
                    
                    annual_outperformance = strategy_annual_return - hodl_annual_return
                else:
                    hodl_annual_return = None
                    strategy_annual_return = None
                    annual_outperformance = None
                
                # Tạo kết quả so sánh
                comparison = {
                    "hodl_return": hodl_return,
                    "strategy_return": strategy_return,
                    "outperformance": outperformance,
                    "hodl_annual_return": hodl_annual_return,
                    "strategy_annual_return": strategy_annual_return,
                    "annual_outperformance": annual_outperformance,
                    "trading_days": trading_days
                }
                
                # Tạo đánh giá
                if outperformance > 0:
                    comparison["assessment"] = "Chiến lược vượt trội so với HODL"
                    comparison["outperformance_percent"] = f"+{outperformance:.2%}"
                else:
                    comparison["assessment"] = "Chiến lược kém hiệu quả hơn HODL"
                    comparison["outperformance_percent"] = f"{outperformance:.2%}"
                
                # Tạo tóm tắt
                comparison["summary"] = (
                    f"Trong {trading_days} ngày giao dịch, chiến lược đạt lợi nhuận {strategy_return:.2%} "
                    f"so với HODL {hodl_return:.2%}, {comparison['outperformance_percent']} điểm phần trăm. "
                    f"Lợi nhuận hàng năm của chiến lược là {strategy_annual_return:.2%} so với HODL {hodl_annual_return:.2%}."
                )
                
                return comparison
            else:
                return {"status": "error", "message": "Dữ liệu thị trường không có cột 'timestamp' hoặc 'close'."}
                
        except Exception as e:
            return {"status": "error", "message": f"Lỗi khi so sánh với benchmark: {str(e)}"}
    
    def _assess_metric(
        self,
        value: Optional[float],
        metric_name: str,
        thresholds: List[float],
        assessments: List[str],
        higher_is_better: bool = True
    ) -> Dict[str, Any]:
        """
        Đánh giá một chỉ số dựa trên ngưỡng.
        
        Args:
            value: Giá trị chỉ số
            metric_name: Tên chỉ số
            thresholds: Danh sách các ngưỡng để đánh giá
            assessments: Danh sách đánh giá tương ứng với ngưỡng
            higher_is_better: True nếu giá trị cao hơn là tốt hơn
            
        Returns:
            Dict chứa đánh giá
        """
        if value is None:
            return {
                "value": None,
                "assessment": "N/A",
                "description": f"Không có dữ liệu để đánh giá {metric_name}."
            }
        
        # Kiểm tra số lượng ngưỡng và đánh giá
        if len(assessments) != len(thresholds) + 1:
            self.logger.warning(f"Số lượng đánh giá ({len(assessments)}) phải bằng số lượng ngưỡng + 1 ({len(thresholds) + 1}).")
            return {
                "value": value,
                "assessment": "Error",
                "description": "Lỗi cấu hình đánh giá."
            }
        
        # Tìm đánh giá tương ứng dựa trên ngưỡng
        assessment_idx = 0
        
        if higher_is_better:
            # Cao hơn là tốt hơn
            for i, threshold in enumerate(thresholds):
                if value >= threshold:
                    assessment_idx = i + 1
        else:
            # Thấp hơn là tốt hơn
            for i, threshold in enumerate(thresholds):
                if value <= threshold:
                    assessment_idx = i + 1
                    break
        
        assessment = assessments[assessment_idx]
        
        # Tạo mô tả
        if higher_is_better:
            if assessment_idx == 0:
                description = f"{metric_name} ({value:.4g}) thấp hơn ngưỡng tối thiểu {thresholds[0]}."
            elif assessment_idx == len(thresholds):
                description = f"{metric_name} ({value:.4g}) vượt ngưỡng cao nhất {thresholds[-1]}."
            else:
                description = f"{metric_name} ({value:.4g}) nằm giữa {thresholds[assessment_idx-1]} và {thresholds[assessment_idx]}."
        else:
            if assessment_idx == 0:
                description = f"{metric_name} ({value:.4g}) cao hơn ngưỡng tối đa {thresholds[0]}."
            elif assessment_idx == len(thresholds):
                description = f"{metric_name} ({value:.4g}) thấp hơn ngưỡng thấp nhất {thresholds[-1]}."
            else:
                description = f"{metric_name} ({value:.4g}) nằm giữa {thresholds[assessment_idx-1]} và {thresholds[assessment_idx]}."
        
        return {
            "value": value,
            "assessment": assessment,
            "description": description
        }
    
    def _save_report(self, report: Dict[str, Any]) -> None:
        """
        Lưu báo cáo đánh giá.
        
        Args:
            report: Báo cáo đánh giá
        """
        # Tạo tên file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{self.strategy_name}_report_{timestamp}.json"
        file_path = self.output_dir / file_name
        
        # Xử lý các kiểu dữ liệu không hỗ trợ JSON
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (pd.DataFrame)):
                return obj.to_dict()
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            else:
                return obj
        
        # Xử lý từng khóa trong dict
        report_to_save = {}
        for key, value in report.items():
            if isinstance(value, dict):
                report_to_save[key] = {}
                for sub_key, sub_value in value.items():
                    report_to_save[key][sub_key] = convert_to_serializable(sub_value)
            else:
                report_to_save[key] = convert_to_serializable(value)
        
        # Lưu vào file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_to_save, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Đã lưu báo cáo đánh giá vào {file_path}")
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu báo cáo đánh giá: {str(e)}")
    
    def compare_strategies(
        self,
        strategies: List[Dict[str, Any]],
        metrics_to_compare: Optional[List[str]] = None,
        normalize_scores: bool = True
    ) -> Dict[str, Any]:
        """
        So sánh nhiều chiến lược.
        
        Args:
            strategies: Danh sách kết quả đánh giá chiến lược
            metrics_to_compare: Danh sách chỉ số cần so sánh
            normalize_scores: Chuẩn hóa điểm số để dễ so sánh
            
        Returns:
            Dict chứa kết quả so sánh
        """
        if not strategies:
            return {"status": "error", "message": "Không có chiến lược để so sánh"}
        
        # Nếu không chỉ định metrics, sử dụng danh sách mặc định
        if metrics_to_compare is None:
            metrics_to_compare = [
                BacktestMetric.TOTAL_RETURN.value,
                BacktestMetric.ANNUALIZED_RETURN.value,
                BacktestMetric.SHARPE_RATIO.value,
                BacktestMetric.MAX_DRAWDOWN.value,
                BacktestMetric.WIN_RATE.value,
                BacktestMetric.PROFIT_FACTOR.value
            ]
        
        # Tạo DataFrame so sánh
        comparison_data = []
        
        for strategy in strategies:
            strategy_name = strategy.get("strategy_name", "Unknown")
            overall_score = strategy.get("overall_score", 0.0)
            rating = strategy.get("rating", "N/A")
            
            row_data = {
                "strategy_name": strategy_name,
                "overall_score": overall_score,
                "rating": rating
            }
            
            # Thêm các chỉ số cụ thể
            for metric in metrics_to_compare:
                # Tìm giá trị metric từ các loại đánh giá khác nhau
                found = False
                
                for category in ["performance", "risk", "consistency", "adaptability"]:
                    if category in strategy.get("metrics", {}):
                        category_metrics = strategy["metrics"][category]
                        if metric in category_metrics:
                            row_data[metric] = category_metrics[metric]
                            found = True
                            break
                
                if not found:
                    row_data[metric] = None
            
            comparison_data.append(row_data)
        
        # Tạo DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Chuẩn hóa điểm số nếu được yêu cầu
        if normalize_scores and len(comparison_df) > 1:
            # Xác định các chỉ số cần đảo ngược (giá trị thấp là tốt)
            reverse_metrics = [BacktestMetric.MAX_DRAWDOWN.value]
            
            # Chuẩn hóa min-max
            for metric in metrics_to_compare:
                if metric in comparison_df.columns and pd.api.types.is_numeric_dtype(comparison_df[metric]):
                    min_val = comparison_df[metric].min()
                    max_val = comparison_df[metric].max()
                    
                    if max_val > min_val:
                        if metric in reverse_metrics:
                            # Đảo ngược giá trị (giá trị thấp là tốt)
                            comparison_df[f"{metric}_normalized"] = 1 - (comparison_df[metric] - min_val) / (max_val - min_val)
                        else:
                            # Giá trị cao là tốt
                            comparison_df[f"{metric}_normalized"] = (comparison_df[metric] - min_val) / (max_val - min_val)
                    else:
                        comparison_df[f"{metric}_normalized"] = 1.0
        
        # Xếp hạng chiến lược dựa trên điểm tổng thể
        ranked_strategies = comparison_df.sort_values("overall_score", ascending=False).copy()
        ranked_strategies["rank"] = np.arange(1, len(ranked_strategies) + 1)
        
        # Tổng hợp kết quả
        result = {
            "comparison_df": comparison_df,
            "ranked_strategies": ranked_strategies,
            "best_strategy": ranked_strategies.iloc[0].to_dict() if len(ranked_strategies) > 0 else None,
            "worst_strategy": ranked_strategies.iloc[-1].to_dict() if len(ranked_strategies) > 0 else None,
            "metrics_compared": metrics_to_compare
        }
        
        # Tạo tóm tắt
        if len(ranked_strategies) > 0:
            best_strategy = ranked_strategies.iloc[0]
            result["summary"] = (
                f"Đã so sánh {len(ranked_strategies)} chiến lược. "
                f"Chiến lược '{best_strategy['strategy_name']}' xếp hạng cao nhất với điểm {best_strategy['overall_score']:.2f}/10 "
                f"(xếp hạng {best_strategy['rating']})."
            )
        else:
            result["summary"] = "Không có dữ liệu để so sánh chiến lược."
        
        return result
    
    def visualize_strategy_results(
        self,
        show_equity_curve: bool = True,
        show_drawdown: bool = True,
        show_trades: bool = True,
        show_benchmarks: bool = True,
        save_figures: bool = True,
        figure_size: Tuple[int, int] = (12, 8)
    ) -> Dict[str, plt.Figure]:
        """
        Trực quan hóa kết quả chiến lược.
        
        Args:
            show_equity_curve: Hiển thị đường cong vốn
            show_drawdown: Hiển thị drawdown
            show_trades: Hiển thị các giao dịch
            show_benchmarks: Hiển thị benchmark
            save_figures: Lưu hình ảnh
            figure_size: Kích thước hình ảnh
            
        Returns:
            Dict chứa các đối tượng figure
        """
        if self.backtest_results is None:
            self.logger.error("Không có kết quả backtest để trực quan hóa")
            return {}
        
        # Thiết lập phong cách seaborn
        sns.set(style="whitegrid")
        
        # Lưu trữ các hình ảnh
        figures = {}
        
        # Trích xuất dữ liệu
        balance_history = self.backtest_results.get("balance_history", {})
        trades = self.backtest_results.get("trades", {})
        metrics = self.backtest_results.get("metrics", {})
        
        # 1. Hiển thị đường cong vốn
        if show_equity_curve and balance_history:
            timestamps = balance_history.get("timestamp", [])
            equity = balance_history.get("equity", [])
            
            if timestamps and equity and len(timestamps) == len(equity):
                equity_df = pd.DataFrame({
                    "timestamp": pd.to_datetime(timestamps),
                    "equity": equity
                })
                equity_df.set_index("timestamp", inplace=True)
                
                # Tạo figure
                fig, ax = plt.subplots(figsize=figure_size)
                
                # Vẽ đường cong vốn
                ax.plot(equity_df.index, equity_df["equity"], label="Equity", color="#1f77b4", linewidth=2)
                
                # Thêm benchmark nếu có dữ liệu thị trường
                if show_benchmarks and self.market_data is not None and not self.market_data.empty:
                    market_data = self.market_data.copy()
                    if "timestamp" in market_data.columns and "close" in market_data.columns:
                        market_data["timestamp"] = pd.to_datetime(market_data["timestamp"])
                        market_data = market_data.sort_values("timestamp")
                        market_data.set_index("timestamp", inplace=True)
                        
                        # Chuẩn hóa giá thị trường với giá trị ban đầu của equity
                        first_equity = equity_df["equity"].iloc[0]
                        first_price = market_data["close"].iloc[0]
                        
                        market_data["normalized_close"] = market_data["close"] / first_price * first_equity
                        
                        # Reindex để khớp với dữ liệu equity
                        market_reindexed = market_data.reindex(equity_df.index, method="ffill")
                        
                        # Vẽ benchmark
                        ax.plot(market_reindexed.index, market_reindexed["normalized_close"], 
                                label="Market (HODL)", color="#ff7f0e", linewidth=1.5, alpha=0.8, linestyle="--")
                
                # Thiết lập tiêu đề và nhãn
                ax.set_title(f"Đường cong vốn - {self.strategy_name}", fontsize=16)
                ax.set_xlabel("Thời gian", fontsize=12)
                ax.set_ylabel("Vốn", fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Thêm thông tin về hiệu suất
                final_equity = equity_df["equity"].iloc[-1]
                initial_equity = equity_df["equity"].iloc[0]
                total_return = (final_equity - initial_equity) / initial_equity
                
                # Thêm metrics key vào đồ thị
                max_drawdown = metrics.get(BacktestMetric.MAX_DRAWDOWN.value, 0.0)
                sharpe_ratio = metrics.get(BacktestMetric.SHARPE_RATIO.value, 0.0)
                
                text = (
                    f"Lợi nhuận: {total_return:.2%}\n"
                    f"Drawdown tối đa: {max_drawdown:.2%}\n"
                    f"Sharpe Ratio: {sharpe_ratio:.2f}"
                )
                
                # Đặt text ở góc trên bên phải
                ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # Định dạng trục x
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Lưu figure
                figures["equity_curve"] = fig
                
                if save_figures:
                    figure_path = self.output_dir / f"{self.strategy_name}_equity_curve.png"
                    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
                    self.logger.info(f"Đã lưu biểu đồ đường cong vốn vào {figure_path}")
        
        # 2. Hiển thị drawdown
        if show_drawdown and balance_history:
            timestamps = balance_history.get("timestamp", [])
            equity = balance_history.get("equity", [])
            
            if timestamps and equity and len(timestamps) == len(equity):
                equity_df = pd.DataFrame({
                    "timestamp": pd.to_datetime(timestamps),
                    "equity": equity
                })
                equity_df.set_index("timestamp", inplace=True)
                
                # Tính drawdown
                equity_df["peak"] = equity_df["equity"].cummax()
                equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]
                
                # Tạo figure
                fig, ax = plt.subplots(figsize=figure_size)
                
                # Vẽ drawdown
                ax.fill_between(equity_df.index, equity_df["drawdown"], 0, 
                                color="red", alpha=0.3, label="Drawdown")
                
                # Thiết lập tiêu đề và nhãn
                ax.set_title(f"Drawdown - {self.strategy_name}", fontsize=16)
                ax.set_xlabel("Thời gian", fontsize=12)
                ax.set_ylabel("Drawdown", fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Định dạng trục y thành phần trăm
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
                
                # Thêm thông tin về drawdown
                max_drawdown = metrics.get(BacktestMetric.MAX_DRAWDOWN.value, 0.0)
                max_drawdown_point = equity_df["drawdown"].min()
                max_drawdown_time = equity_df[equity_df["drawdown"] == max_drawdown_point].index[0]
                
                # Đánh dấu điểm drawdown tối đa
                ax.plot(max_drawdown_time, max_drawdown_point, 'ro', ms=10)
                ax.annotate(f'Max DD: {max_drawdown_point:.2%}', 
                            xy=(max_drawdown_time, max_drawdown_point),
                            xytext=(30, -30),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='->'),
                            fontsize=12)
                
                # Định dạng trục x
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Lưu figure
                figures["drawdown"] = fig
                
                if save_figures:
                    figure_path = self.output_dir / f"{self.strategy_name}_drawdown.png"
                    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
                    self.logger.info(f"Đã lưu biểu đồ drawdown vào {figure_path}")
        
        # 3. Hiển thị các giao dịch
        if show_trades and trades and self.market_data is not None:
            closed_positions = trades.get("closed_positions", [])
            
            if closed_positions and self.market_data is not None and not self.market_data.empty:
                # Chuẩn bị dữ liệu thị trường
                market_data = self.market_data.copy()
                if "timestamp" in market_data.columns and "close" in market_data.columns:
                    market_data["timestamp"] = pd.to_datetime(market_data["timestamp"])
                    market_data = market_data.sort_values("timestamp")
                    market_data.set_index("timestamp", inplace=True)
                    
                    # Chuẩn bị dữ liệu giao dịch
                    trades_df = pd.DataFrame(closed_positions)
                    
                    # Chuyển timestamp thành datetime
                    if "entry_time" in trades_df.columns:
                        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
                    if "exit_time" in trades_df.columns:
                        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
                    
                    # Tạo figure
                    fig, ax = plt.subplots(figsize=figure_size)
                    
                    # Vẽ giá thị trường
                    ax.plot(market_data.index, market_data["close"], color="black", linewidth=1, alpha=0.7)
                    
                    # Vẽ các điểm vào lệnh và ra lệnh
                    for _, trade in trades_df.iterrows():
                        try:
                            # Lấy thông tin giao dịch
                            entry_time = trade.get("entry_time")
                            exit_time = trade.get("exit_time")
                            entry_price = trade.get("entry_price")
                            exit_price = trade.get("exit_price")
                            side = trade.get("side", "").lower()
                            pnl = trade.get("realized_pnl", 0)
                            
                            # Chỉ vẽ nếu có đủ thông tin
                            if pd.notna(entry_time) and pd.notna(exit_time) and pd.notna(entry_price) and pd.notna(exit_price):
                                # Xác định màu sắc dựa trên loại lệnh và lợi nhuận
                                if side == "long":
                                    entry_marker = "^"  # Tam giác lên
                                    if pnl > 0:
                                        exit_marker = "o"  # Tròn
                                        line_color = "green"
                                    else:
                                        exit_marker = "X"  # X
                                        line_color = "red"
                                else:  # short
                                    entry_marker = "v"  # Tam giác xuống
                                    if pnl > 0:
                                        exit_marker = "o"  # Tròn
                                        line_color = "green"
                                    else:
                                        exit_marker = "X"  # X
                                        line_color = "red"
                                
                                # Vẽ điểm vào lệnh
                                ax.plot(entry_time, entry_price, marker=entry_marker, color=line_color, 
                                        markersize=10, markeredgecolor="black")
                                
                                # Vẽ điểm ra lệnh
                                ax.plot(exit_time, exit_price, marker=exit_marker, color=line_color, 
                                        markersize=10, markeredgecolor="black")
                                
                                # Vẽ đường nối
                                ax.plot([entry_time, exit_time], [entry_price, exit_price], 
                                        color=line_color, linewidth=1.5, alpha=0.7)
                        except Exception as e:
                            self.logger.warning(f"Lỗi khi vẽ giao dịch: {str(e)}")
                    
                    # Thiết lập tiêu đề và nhãn
                    ax.set_title(f"Giao dịch - {self.strategy_name}", fontsize=16)
                    ax.set_xlabel("Thời gian", fontsize=12)
                    ax.set_ylabel("Giá", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    
                    # Thêm chú thích
                    from matplotlib.lines import Line2D
                    
                    legend_elements = [
                        Line2D([0], [0], marker="^", color="w", markerfacecolor="green", 
                               markersize=10, markeredgecolor="black", label="Long Entry"),
                        Line2D([0], [0], marker="v", color="w", markerfacecolor="red", 
                               markersize=10, markeredgecolor="black", label="Short Entry"),
                        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", 
                               markersize=10, markeredgecolor="black", label="Profitable Exit"),
                        Line2D([0], [0], marker="X", color="w", markerfacecolor="red", 
                               markersize=10, markeredgecolor="black", label="Loss Exit")
                    ]
                    
                    ax.legend(handles=legend_elements, loc="upper left")
                    
                    # Định dạng trục x
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Lưu figure
                    figures["trades"] = fig
                    
                    if save_figures:
                        figure_path = self.output_dir / f"{self.strategy_name}_trades.png"
                        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
                        self.logger.info(f"Đã lưu biểu đồ giao dịch vào {figure_path}")
        
        return figures