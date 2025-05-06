#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module đánh giá hiệu suất chiến lược giao dịch.

Module này cung cấp các công cụ để đánh giá hiệu suất chiến lược giao dịch
sau khi thực hiện backtest, tính toán các chỉ số hiệu suất theo nhiều khía cạnh,
thực hiện so sánh chiến lược và đưa ra các báo cáo chi tiết.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta
import json
from pathlib import Path
import concurrent.futures
from functools import partial
import pickle
import warnings

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import BacktestMetric
from config.system_config import get_system_config, BACKTEST_DIR

# Import các module backtesting
try:
    from backtesting.performance_metrics import PerformanceMetrics
    PERFORMANCE_METRICS_AVAILABLE = True
except ImportError:
    PERFORMANCE_METRICS_AVAILABLE = False


class PerformanceEvaluator:
    """
    Đánh giá hiệu suất của chiến lược giao dịch.
    
    Lớp này cung cấp các phương thức để phân tích và đánh giá toàn diện
    hiệu suất của một hoặc nhiều chiến lược giao dịch, đưa ra nhận định
    về các điểm mạnh, điểm yếu và đề xuất cải tiến.
    """
    
    def __init__(
        self,
        results_dir: Optional[Path] = None,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252,
        benchmark_data: Optional[pd.DataFrame] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo đối tượng PerformanceEvaluator.
        
        Args:
            results_dir: Thư mục chứa kết quả backtest
            risk_free_rate: Lãi suất phi rủi ro hàng năm
            trading_days_per_year: Số ngày giao dịch trong một năm
            benchmark_data: Dữ liệu benchmark (nếu có)
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("performance_evaluator")
        
        # Thiết lập thư mục kết quả
        if results_dir is None:
            self.results_dir = BACKTEST_DIR
        else:
            self.results_dir = Path(results_dir)
        
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.benchmark_data = benchmark_data
        
        # Kiểm tra sự sẵn có của module PerformanceMetrics
        if not PERFORMANCE_METRICS_AVAILABLE:
            self.logger.warning("Module PerformanceMetrics không khả dụng. Một số tính năng sẽ bị giới hạn.")
        
        # Lưu trữ kết quả đã đánh giá
        self.evaluated_results = {}
        
        # Danh sách các chỉ số quan trọng
        self.key_metrics = [
            BacktestMetric.TOTAL_RETURN.value,
            BacktestMetric.ANNUALIZED_RETURN.value,
            BacktestMetric.SHARPE_RATIO.value,
            BacktestMetric.SORTINO_RATIO.value,
            BacktestMetric.MAX_DRAWDOWN.value,
            BacktestMetric.WIN_RATE.value,
            BacktestMetric.PROFIT_FACTOR.value,
            BacktestMetric.CALMAR_RATIO.value,
            BacktestMetric.EXPECTANCY.value,
            BacktestMetric.SYSTEM_QUALITY_NUMBER.value,
            "volatility"
        ]
        
        self.logger.info(f"Đã khởi tạo PerformanceEvaluator với results_dir={self.results_dir}")
    
    def load_backtest_result(self, result_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Tải kết quả backtest từ file.
        
        Args:
            result_path: Đường dẫn đến file kết quả
            
        Returns:
            Dict chứa kết quả backtest
        """
        result_path = Path(result_path)
        
        if not result_path.exists():
            self.logger.error(f"File kết quả không tồn tại: {result_path}")
            return {}
        
        try:
            if result_path.suffix == '.json':
                with open(result_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
            elif result_path.suffix == '.pickle':
                with open(result_path, 'rb') as f:
                    result = pickle.load(f)
            else:
                self.logger.error(f"Định dạng file không được hỗ trợ: {result_path.suffix}")
                return {}
            
            self.logger.info(f"Đã tải kết quả backtest từ {result_path}")
            return result
        
        except Exception as e:
            self.logger.error(f"Lỗi khi tải kết quả backtest: {e}")
            return {}
    
    def load_all_results(self, strategy_dir: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Tải tất cả kết quả backtest từ thư mục.
        
        Args:
            strategy_dir: Thư mục chứa kết quả của chiến lược
            
        Returns:
            Dict ánh xạ tên chiến lược -> kết quả
        """
        results = {}
        
        if strategy_dir is None:
            # Tìm tất cả thư mục con
            strategy_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        else:
            strategy_dir = Path(strategy_dir)
            if not strategy_dir.exists() or not strategy_dir.is_dir():
                self.logger.error(f"Thư mục chiến lược không tồn tại: {strategy_dir}")
                return {}
            strategy_dirs = [strategy_dir]
        
        for strat_dir in strategy_dirs:
            strategy_name = strat_dir.name
            
            # Tìm file combined_result.json
            combined_file = strat_dir / "combined_result.json"
            if combined_file.exists():
                result = self.load_backtest_result(combined_file)
                if result:
                    results[strategy_name] = result
                    continue
            
            # Nếu không có file combined_result.json, tìm kết quả từng symbol
            symbol_results = {}
            for result_file in strat_dir.glob("*_result.json"):
                symbol = result_file.stem.replace("_result", "")
                symbol_result = self.load_backtest_result(result_file)
                if symbol_result:
                    symbol_results[symbol] = symbol_result
            
            if symbol_results:
                # Tạo combined result tạm thời
                results[strategy_name] = {
                    "strategy_name": strategy_name,
                    "symbol_results": symbol_results,
                    "status": "success"
                }
        
        self.logger.info(f"Đã tải {len(results)} kết quả backtest")
        return results
    
    def evaluate_strategy(
        self,
        result: Dict[str, Any],
        strategy_name: Optional[str] = None,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Đánh giá hiệu suất của một chiến lược.
        
        Args:
            result: Kết quả backtest
            strategy_name: Tên chiến lược
            detailed: Đánh giá chi tiết hay không
            
        Returns:
            Dict chứa kết quả đánh giá
        """
        if not result or result.get("status") != "success":
            self.logger.error("Kết quả backtest không hợp lệ")
            return {
                "strategy_name": strategy_name or result.get("strategy_name", "unknown"),
                "status": "error",
                "message": "Kết quả backtest không hợp lệ"
            }
        
        # Lấy tên chiến lược
        strategy_name = strategy_name or result.get("strategy_name", "unknown")
        
        # Kết quả đánh giá
        evaluation = {
            "strategy_name": strategy_name,
            "status": "success",
            "evaluation_time": datetime.now().isoformat(),
            "metrics_summary": {},
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "detailed_metrics": {},
            "symbol_evaluations": {}
        }
        
        # Tổng hợp từ combined_result nếu có
        if "combined_result" in result and result["combined_result"]:
            combined = result["combined_result"]
            
            # Tổng hợp metrics
            metrics = combined.get("metrics", {})
            evaluation["metrics_summary"] = metrics.copy()
            
            # Thêm ROI và các thông tin khác
            evaluation["metrics_summary"]["roi"] = combined.get("roi", 0)
            evaluation["metrics_summary"]["initial_balance"] = combined.get("initial_balance", 0)
            evaluation["metrics_summary"]["final_balance"] = combined.get("final_balance", 0)
            evaluation["metrics_summary"]["symbols_count"] = combined.get("symbols_count", 0)
        
        # Đánh giá từng symbol
        if "symbol_results" in result and detailed:
            for symbol, symbol_result in result["symbol_results"].items():
                if isinstance(symbol_result, dict) and symbol_result.get("status") == "success":
                    symbol_eval = self._evaluate_symbol_result(symbol_result, symbol)
                    evaluation["symbol_evaluations"][symbol] = symbol_eval
        
        # Đánh giá điểm mạnh và điểm yếu
        self._analyze_strengths_weaknesses(evaluation)
        
        # Đề xuất cải tiến
        self._generate_recommendations(evaluation)
        
        # Lưu kết quả đánh giá
        self.evaluated_results[strategy_name] = evaluation
        
        return evaluation
    
    def _evaluate_symbol_result(
        self,
        symbol_result: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """
        Đánh giá kết quả của một symbol.
        
        Args:
            symbol_result: Kết quả backtest của symbol
            symbol: Tên symbol
            
        Returns:
            Dict chứa kết quả đánh giá
        """
        evaluation = {
            "symbol": symbol,
            "metrics": {},
            "strengths": [],
            "weaknesses": []
        }
        
        # Lấy metrics
        metrics = symbol_result.get("metrics", {})
        evaluation["metrics"] = metrics.copy()
        
        # Thêm ROI và các thông tin khác
        evaluation["metrics"]["roi"] = symbol_result.get("roi", 0)
        evaluation["metrics"]["initial_balance"] = symbol_result.get("initial_balance", 0)
        evaluation["metrics"]["final_balance"] = symbol_result.get("final_balance", 0)
        
        # Phân tích chi tiết hơn nếu có dữ liệu giao dịch
        trades = symbol_result.get("trades", [])
        if trades and len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            
            # Phân tích giao dịch theo thời gian
            if "entry_time" in trades_df.columns:
                trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
                trades_by_month = trades_df.groupby(trades_df["entry_time"].dt.to_period("M"))
                
                monthly_performance = {}
                for month, group in trades_by_month:
                    monthly_performance[str(month)] = {
                        "trades_count": len(group),
                        "win_rate": len(group[group["profit"] > 0]) / len(group) if len(group) > 0 else 0,
                        "avg_profit": group["profit"].mean(),
                        "total_profit": group["profit"].sum()
                    }
                
                evaluation["monthly_performance"] = monthly_performance
            
            # Phân tích lợi nhuận/lỗ
            evaluation["trade_stats"] = {
                "total_trades": len(trades_df),
                "winning_trades": len(trades_df[trades_df["profit"] > 0]),
                "losing_trades": len(trades_df[trades_df["profit"] < 0]),
                "avg_profit": trades_df[trades_df["profit"] > 0]["profit"].mean() if len(trades_df[trades_df["profit"] > 0]) > 0 else 0,
                "avg_loss": trades_df[trades_df["profit"] < 0]["profit"].mean() if len(trades_df[trades_df["profit"] < 0]) > 0 else 0,
                "max_profit": trades_df["profit"].max(),
                "max_loss": trades_df["profit"].min(),
                "profit_std": trades_df["profit"].std(),
                "consecutive_wins": self._get_max_consecutive(trades_df["profit"] > 0),
                "consecutive_losses": self._get_max_consecutive(trades_df["profit"] < 0)
            }
        
        # Phân tích balance history nếu có
        balance_history = symbol_result.get("balance_history", {})
        if isinstance(balance_history, dict) and "equity" in balance_history and len(balance_history["equity"]) > 0:
            # Chuyển đổi balance history thành DataFrame
            if isinstance(balance_history["equity"], list) and isinstance(balance_history.get("timestamp"), list):
                try:
                    balance_df = pd.DataFrame({
                        "timestamp": pd.to_datetime(balance_history["timestamp"]),
                        "equity": balance_history["equity"]
                    })
                    balance_df.set_index("timestamp", inplace=True)
                    
                    # Phân tích hiệu suất theo thời gian
                    if len(balance_df) > 1:
                        # Tính returns
                        balance_df["returns"] = balance_df["equity"].pct_change()
                        
                        # Tính các chỉ số bổ sung nếu có PerformanceMetrics
                        if PERFORMANCE_METRICS_AVAILABLE:
                            try:
                                perf_metrics = PerformanceMetrics(
                                    equity_curve=balance_df["equity"],
                                    risk_free_rate=self.risk_free_rate,
                                    trading_days_per_year=self.trading_days_per_year
                                )
                                
                                # Tính thêm các chỉ số
                                evaluation["metrics"]["volatility"] = perf_metrics.volatility()
                                evaluation["metrics"]["downside_volatility"] = perf_metrics.downside_volatility()
                                evaluation["metrics"]["skewness"] = perf_metrics.skewness()
                                evaluation["metrics"]["kurtosis"] = perf_metrics.kurtosis()
                                evaluation["metrics"]["value_at_risk"] = perf_metrics.value_at_risk()
                                
                                # Tính drawdown periods
                                dd_periods = perf_metrics.drawdown_periods()
                                if dd_periods:
                                    evaluation["drawdown_periods"] = dd_periods
                                
                                # Tính recovery periods
                                recovery_periods = perf_metrics.recovery_periods()
                                if recovery_periods:
                                    evaluation["recovery_periods"] = recovery_periods
                            
                            except Exception as e:
                                self.logger.warning(f"Lỗi khi tính toán metrics bổ sung: {e}")
                
                except Exception as e:
                    self.logger.warning(f"Lỗi khi xử lý balance history: {e}")
        
        # Phân tích điểm mạnh và điểm yếu của symbol
        self._analyze_symbol_strengths_weaknesses(evaluation)
        
        return evaluation
    
    def _get_max_consecutive(self, series: pd.Series) -> int:
        """
        Tính số lần liên tiếp lớn nhất của giá trị True.
        
        Args:
            series: Series chứa giá trị boolean
            
        Returns:
            Số lần liên tiếp lớn nhất
        """
        if len(series) == 0:
            return 0
            
        # Chuyển đổi series thành list các giá trị 0 và 1
        values = series.astype(int).tolist()
        
        # Tính số lần liên tiếp lớn nhất
        max_consecutive = 0
        current_consecutive = 0
        
        for value in values:
            if value == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _analyze_strengths_weaknesses(self, evaluation: Dict[str, Any]) -> None:
        """
        Phân tích điểm mạnh và điểm yếu của chiến lược.
        
        Args:
            evaluation: Dict chứa kết quả đánh giá
        """
        metrics = evaluation.get("metrics_summary", {})
        
        # Điểm mạnh
        strengths = []
        
        # ROI và lợi nhuận
        roi = metrics.get("roi", 0)
        if roi > 0.2:  # ROI > 20%
            strengths.append(f"Hiệu suất rất tốt với ROI {roi:.2%}")
        elif roi > 0.1:  # ROI > 10%
            strengths.append(f"Hiệu suất tốt với ROI {roi:.2%}")
        
        # Sharpe Ratio
        sharpe = metrics.get(BacktestMetric.SHARPE_RATIO.value, 0)
        if sharpe > 2:
            strengths.append(f"Sharpe Ratio rất tốt ({sharpe:.2f}), cho thấy hiệu quả cao so với rủi ro")
        elif sharpe > 1:
            strengths.append(f"Sharpe Ratio tốt ({sharpe:.2f}), cho thấy hiệu quả hợp lý so với rủi ro")
        
        # Win Rate
        win_rate = metrics.get(BacktestMetric.WIN_RATE.value, 0)
        if win_rate > 0.6:
            strengths.append(f"Tỷ lệ thắng cao ({win_rate:.2%})")
        
        # Profit Factor
        profit_factor = metrics.get(BacktestMetric.PROFIT_FACTOR.value, 0)
        if profit_factor > 2:
            strengths.append(f"Profit Factor rất tốt ({profit_factor:.2f})")
        elif profit_factor > 1.5:
            strengths.append(f"Profit Factor tốt ({profit_factor:.2f})")
        
        # SQN
        sqn = metrics.get(BacktestMetric.SYSTEM_QUALITY_NUMBER.value, 0)
        if sqn > 4:
            strengths.append(f"Chỉ số chất lượng hệ thống (SQN) xuất sắc ({sqn:.2f})")
        elif sqn > 2:
            strengths.append(f"Chỉ số chất lượng hệ thống (SQN) tốt ({sqn:.2f})")
        
        # Điểm yếu
        weaknesses = []
        
        # ROI và lợi nhuận
        if roi < 0:
            weaknesses.append(f"Chiến lược thua lỗ với ROI {roi:.2%}")
        elif roi < 0.05:
            weaknesses.append(f"Hiệu suất thấp với ROI chỉ {roi:.2%}")
        
        # Max Drawdown
        max_dd = metrics.get(BacktestMetric.MAX_DRAWDOWN.value, 0)
        if max_dd > 0.3:
            weaknesses.append(f"Max Drawdown rất cao ({max_dd:.2%}), cho thấy rủi ro lớn")
        elif max_dd > 0.2:
            weaknesses.append(f"Max Drawdown cao ({max_dd:.2%})")
        
        # Sharpe Ratio
        if sharpe < 0:
            weaknesses.append(f"Sharpe Ratio âm ({sharpe:.2f}), hiệu suất tệ hơn tài sản phi rủi ro")
        elif sharpe < 0.5:
            weaknesses.append(f"Sharpe Ratio thấp ({sharpe:.2f}), hiệu quả chưa tương xứng với rủi ro")
        
        # Win Rate
        if win_rate < 0.4:
            weaknesses.append(f"Tỷ lệ thắng thấp ({win_rate:.2%})")
        
        # Profit Factor
        if profit_factor < 1:
            weaknesses.append(f"Profit Factor dưới 1 ({profit_factor:.2f}), cho thấy chiến lược thua lỗ")
        
        # Biến động
        volatility = metrics.get("volatility", 0)
        if volatility > 0.3:
            weaknesses.append(f"Biến động rất cao ({volatility:.2%}), cho thấy rủi ro lớn")
        
        # Cập nhật kết quả đánh giá
        evaluation["strengths"] = strengths
        evaluation["weaknesses"] = weaknesses
    
    def _analyze_symbol_strengths_weaknesses(self, evaluation: Dict[str, Any]) -> None:
        """
        Phân tích điểm mạnh và điểm yếu của một symbol.
        
        Args:
            evaluation: Dict chứa kết quả đánh giá của symbol
        """
        metrics = evaluation.get("metrics", {})
        trade_stats = evaluation.get("trade_stats", {})
        
        # Điểm mạnh
        strengths = []
        
        # ROI và lợi nhuận
        roi = metrics.get("roi", 0)
        if roi > 0.2:  # ROI > 20%
            strengths.append(f"Hiệu suất rất tốt với ROI {roi:.2%}")
        elif roi > 0.1:  # ROI > 10%
            strengths.append(f"Hiệu suất tốt với ROI {roi:.2%}")
        
        # Sharpe Ratio
        sharpe = metrics.get(BacktestMetric.SHARPE_RATIO.value, 0)
        if sharpe > 2:
            strengths.append(f"Sharpe Ratio rất tốt ({sharpe:.2f})")
        elif sharpe > 1:
            strengths.append(f"Sharpe Ratio tốt ({sharpe:.2f})")
        
        # Win Rate
        win_rate = metrics.get(BacktestMetric.WIN_RATE.value, 0)
        if win_rate > 0.6:
            strengths.append(f"Tỷ lệ thắng cao ({win_rate:.2%})")
        
        # Profit Factor
        profit_factor = metrics.get(BacktestMetric.PROFIT_FACTOR.value, 0)
        if profit_factor > 2:
            strengths.append(f"Profit Factor rất tốt ({profit_factor:.2f})")
        
        # Consecutive wins
        consecutive_wins = trade_stats.get("consecutive_wins", 0)
        if consecutive_wins >= 5:
            strengths.append(f"Số lần thắng liên tiếp tối đa: {consecutive_wins}")
        
        # Điểm yếu
        weaknesses = []
        
        # ROI và lợi nhuận
        if roi < 0:
            weaknesses.append(f"Symbol thua lỗ với ROI {roi:.2%}")
        elif roi < 0.05:
            weaknesses.append(f"Hiệu suất thấp với ROI chỉ {roi:.2%}")
        
        # Max Drawdown
        max_dd = metrics.get(BacktestMetric.MAX_DRAWDOWN.value, 0)
        if max_dd > 0.3:
            weaknesses.append(f"Max Drawdown rất cao ({max_dd:.2%})")
        
        # Win Rate
        if win_rate < 0.4:
            weaknesses.append(f"Tỷ lệ thắng thấp ({win_rate:.2%})")
        
        # Consecutive losses
        consecutive_losses = trade_stats.get("consecutive_losses", 0)
        if consecutive_losses >= 5:
            weaknesses.append(f"Số lần thua liên tiếp tối đa: {consecutive_losses}")
        
        # Cập nhật kết quả đánh giá
        evaluation["strengths"] = strengths
        evaluation["weaknesses"] = weaknesses
    
    def _generate_recommendations(self, evaluation: Dict[str, Any]) -> None:
        """
        Đề xuất cải tiến cho chiến lược.
        
        Args:
            evaluation: Dict chứa kết quả đánh giá
        """
        recommendations = []
        metrics = evaluation.get("metrics_summary", {})
        weaknesses = evaluation.get("weaknesses", [])
        
        # Dựa vào điểm yếu
        for weakness in weaknesses:
            if "Max Drawdown" in weakness:
                recommendations.append("Cải thiện quản lý rủi ro để giảm Max Drawdown, ví dụ: thêm stop loss hoặc trailing stop")
            elif "Tỷ lệ thắng thấp" in weakness:
                recommendations.append("Cải thiện tín hiệu vào lệnh để tăng tỷ lệ thắng, xem xét thêm bộ lọc để giảm giao dịch giả")
            elif "Sharpe Ratio" in weakness and "thấp" in weakness:
                recommendations.append("Cân nhắc điều chỉnh chiến lược để cải thiện tỷ lệ lợi nhuận/rủi ro")
            elif "Biến động" in weakness and "cao" in weakness:
                recommendations.append("Giảm biến động bằng cách áp dụng kích thước vị thế thích ứng hoặc lọc các giai đoạn biến động cao")
        
        # Dựa vào chỉ số
        roi = metrics.get("roi", 0)
        win_rate = metrics.get(BacktestMetric.WIN_RATE.value, 0)
        profit_factor = metrics.get(BacktestMetric.PROFIT_FACTOR.value, 0)
        
        # Nếu ROI âm hoặc thấp
        if roi < 0.05:
            if win_rate > 0.5:
                # Tỷ lệ thắng tốt nhưng ROI thấp
                recommendations.append("Tỷ lệ thắng tốt nhưng ROI thấp, cân nhắc tăng tỷ lệ lợi nhuận/lỗ bằng cách điều chỉnh take profit cao hơn hoặc giảm stop loss")
            else:
                # Cả tỷ lệ thắng và ROI đều thấp
                recommendations.append("Cải thiện cơ bản chiến lược: xem xét lại logic tín hiệu, thời điểm vào lệnh, và quản lý vị thế")
        
        # Nếu profit factor thấp
        if profit_factor < 1.2 and profit_factor > 0:
            recommendations.append("Cải thiện Profit Factor bằng cách xem xét lại logic chốt lời/cắt lỗ, tăng kích thước vị thế cho giao dịch thắng và giảm cho giao dịch thua")
        
        # Đề xuất theo chiến lược
        symbol_evals = evaluation.get("symbol_evaluations", {})
        
        if len(symbol_evals) > 1:
            # Tìm symbol hiệu quả nhất
            best_symbol = None
            best_roi = -float('inf')
            
            for symbol, symbol_eval in symbol_evals.items():
                symbol_roi = symbol_eval.get("metrics", {}).get("roi", 0)
                if symbol_roi > best_roi:
                    best_roi = symbol_roi
                    best_symbol = symbol
            
            if best_symbol and best_roi > 0:
                recommendations.append(f"Hiệu suất tốt nhất đạt được với {best_symbol} (ROI: {best_roi:.2%}). Cân nhắc tối ưu hóa chiến lược cho các cặp tiền khác dựa trên cấu hình này.")
        
        # Đề xuất tối ưu hóa tham số
        if roi >= 0:
            recommendations.append("Cân nhắc tối ưu hóa các tham số chiến lược để cải thiện hiệu suất")
        
        # Lọc bỏ các đề xuất trùng lặp
        recommendations = list(set(recommendations))
        
        # Cập nhật kết quả đánh giá
        evaluation["recommendations"] = recommendations
    
    def evaluate_all_strategies(
        self,
        results: Optional[Dict[str, Dict[str, Any]]] = None,
        detailed: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Đánh giá hiệu suất của tất cả các chiến lược.
        
        Args:
            results: Dict ánh xạ tên chiến lược -> kết quả backtest
            detailed: Đánh giá chi tiết hay không
            
        Returns:
            Dict ánh xạ tên chiến lược -> kết quả đánh giá
        """
        if results is None:
            results = self.load_all_results()
        
        evaluations = {}
        
        for strategy_name, result in results.items():
            evaluation = self.evaluate_strategy(result, strategy_name, detailed)
            evaluations[strategy_name] = evaluation
        
        return evaluations
    
    def compare_strategies(
        self,
        evaluations: Optional[Dict[str, Dict[str, Any]]] = None,
        metrics: Optional[List[str]] = None,
        sort_by: str = BacktestMetric.SHARPE_RATIO.value,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        So sánh các chiến lược dựa trên các chỉ số đã đánh giá.
        
        Args:
            evaluations: Dict ánh xạ tên chiến lược -> kết quả đánh giá
            metrics: Danh sách chỉ số cần so sánh
            sort_by: Chỉ số để sắp xếp
            ascending: Sắp xếp tăng dần hay không
            
        Returns:
            DataFrame so sánh các chiến lược
        """
        if evaluations is None:
            if not self.evaluated_results:
                evaluations = self.evaluate_all_strategies()
            else:
                evaluations = self.evaluated_results
        
        if not evaluations:
            self.logger.error("Không có chiến lược nào để so sánh")
            return pd.DataFrame()
        
        # Thiết lập metrics mặc định
        if metrics is None:
            metrics = self.key_metrics
        
        # Dữ liệu so sánh
        comparison_data = []
        
        # Lặp qua từng chiến lược
        for strategy_name, evaluation in evaluations.items():
            if evaluation.get("status") != "success":
                continue
            
            strategy_metrics = evaluation.get("metrics_summary", {})
            
            row = {"strategy": strategy_name}
            
            # Thêm các metrics
            for metric in metrics:
                if metric in strategy_metrics:
                    row[metric] = strategy_metrics[metric]
            
            # Thêm số lượng điểm mạnh và điểm yếu
            row["strengths_count"] = len(evaluation.get("strengths", []))
            row["weaknesses_count"] = len(evaluation.get("weaknesses", []))
            
            comparison_data.append(row)
        
        # Tạo DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sắp xếp
        if sort_by in comparison_df.columns:
            comparison_df.sort_values(by=sort_by, ascending=ascending, inplace=True)
        
        return comparison_df
    
    def plot_strategies_comparison(
        self,
        comparison_df: Optional[pd.DataFrame] = None,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> Optional[plt.Figure]:
        """
        Vẽ biểu đồ so sánh các chiến lược.
        
        Args:
            comparison_df: DataFrame so sánh các chiến lược
            metrics: Danh sách chỉ số cần vẽ
            figsize: Kích thước biểu đồ
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ hay không
            
        Returns:
            Figure của matplotlib hoặc None
        """
        if comparison_df is None:
            comparison_df = self.compare_strategies()
        
        if comparison_df.empty:
            self.logger.error("Không có dữ liệu để vẽ biểu đồ")
            return None
        
        # Thiết lập metrics mặc định
        if metrics is None:
            # Chọn tối đa 6 metrics quan trọng nhất
            metrics = [
                BacktestMetric.SHARPE_RATIO.value,
                BacktestMetric.SORTINO_RATIO.value,
                BacktestMetric.MAX_DRAWDOWN.value,
                BacktestMetric.WIN_RATE.value,
                "roi",
                BacktestMetric.PROFIT_FACTOR.value
            ]
            # Lọc các metrics có trong DataFrame
            metrics = [m for m in metrics if m in comparison_df.columns]
            # Giới hạn số lượng metrics
            metrics = metrics[:min(6, len(metrics))]
        
        # Tạo biểu đồ
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric not in comparison_df.columns:
                self.logger.warning(f"Metric '{metric}' không có trong dữ liệu")
                continue
            
            # Tạo barplot
            sns.barplot(x='strategy', y=metric, data=comparison_df, ax=axes[i])
            
            # Tùy chỉnh biểu đồ
            axes[i].set_title(f"{metric.replace('_', ' ').title()}")
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            
            # Format giá trị thành phần trăm cho một số metrics
            if metric in ["roi", BacktestMetric.MAX_DRAWDOWN.value, BacktestMetric.WIN_RATE.value]:
                axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
            
            # Thêm giá trị trên mỗi cột
            for j, v in enumerate(comparison_df[metric]):
                if pd.notna(v):
                    if metric in ["roi", BacktestMetric.MAX_DRAWDOWN.value, BacktestMetric.WIN_RATE.value]:
                        text = f'{v:.1%}'
                    else:
                        text = f'{v:.2f}'
                    axes[i].text(j, v, text, ha='center', va='bottom', fontsize=9)
        
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
    
    def plot_strategy_metrics(
        self,
        strategy_name: str,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> Optional[plt.Figure]:
        """
        Vẽ biểu đồ các chỉ số hiệu suất của một chiến lược.
        
        Args:
            strategy_name: Tên chiến lược
            figsize: Kích thước biểu đồ
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ hay không
            
        Returns:
            Figure của matplotlib hoặc None
        """
        if strategy_name not in self.evaluated_results:
            self.logger.error(f"Chiến lược '{strategy_name}' chưa được đánh giá")
            return None
        
        evaluation = self.evaluated_results[strategy_name]
        if evaluation.get("status") != "success":
            self.logger.error(f"Đánh giá chiến lược '{strategy_name}' không thành công")
            return None
        
        symbol_evals = evaluation.get("symbol_evaluations", {})
        if not symbol_evals:
            self.logger.error(f"Không có dữ liệu symbol cho chiến lược '{strategy_name}'")
            return None
        
        # Tạo biểu đồ
        fig = plt.figure(figsize=figsize)
        
        # Số lượng đồ thị
        num_plots = min(len(symbol_evals), 9)  # Giới hạn 9 đồ thị
        cols = min(3, num_plots)
        rows = (num_plots + cols - 1) // cols
        
        # Vẽ đồ thị cho từng symbol
        for i, (symbol, symbol_eval) in enumerate(list(symbol_evals.items())[:num_plots]):
            ax = fig.add_subplot(rows, cols, i + 1)
            
            # Kiểm tra có balance history không
            if "balance_history" in symbol_eval:
                # Đã xử lý dữ liệu balance history
                balance_df = pd.DataFrame(symbol_eval["balance_history"])
                
                # Vẽ đường cong equity
                ax.plot(balance_df.index, balance_df["equity"], label='Equity')
                
                # Nếu có drawdown, vẽ thêm drawdown
                if "drawdown" in balance_df.columns:
                    ax2 = ax.twinx()
                    ax2.fill_between(balance_df.index, 0, -balance_df["drawdown"], color='red', alpha=0.3)
                    ax2.set_ylabel('Drawdown')
                    ax2.set_ylim(-max(balance_df["drawdown"]) * 1.5, 0)
            else:
                # Không có dữ liệu, hiển thị chỉ số quan trọng
                metrics = symbol_eval.get("metrics", {})
                if metrics:
                    text = f"ROI: {metrics.get('roi', 0):.2%}\n"
                    text += f"Sharpe: {metrics.get(BacktestMetric.SHARPE_RATIO.value, 0):.2f}\n"
                    text += f"Max DD: {metrics.get(BacktestMetric.MAX_DRAWDOWN.value, 0):.2%}\n"
                    text += f"Win Rate: {metrics.get(BacktestMetric.WIN_RATE.value, 0):.2%}"
                    
                    ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            
            ax.set_title(f"{symbol}")
        
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
    
    def generate_report(
        self,
        strategy_name: str,
        output_path: Optional[Union[str, Path]] = None,
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Tạo báo cáo đánh giá cho một chiến lược.
        
        Args:
            strategy_name: Tên chiến lược
            output_path: Đường dẫn để lưu báo cáo
            include_plots: Bao gồm biểu đồ hay không
            
        Returns:
            Dict chứa báo cáo
        """
        if strategy_name not in self.evaluated_results:
            self.logger.error(f"Chiến lược '{strategy_name}' chưa được đánh giá")
            return {"status": "error", "message": f"Chiến lược '{strategy_name}' chưa được đánh giá"}
        
        evaluation = self.evaluated_results[strategy_name]
        if evaluation.get("status") != "success":
            self.logger.error(f"Đánh giá chiến lược '{strategy_name}' không thành công")
            return {"status": "error", "message": f"Đánh giá chiến lược '{strategy_name}' không thành công"}
        
        # Tạo báo cáo
        report = {
            "strategy_name": strategy_name,
            "report_time": datetime.now().isoformat(),
            "status": "success",
            "summary": {},
            "detailed_metrics": {},
            "analysis": {},
            "recommendations": [],
            "plots": []
        }
        
        # Tóm tắt
        metrics = evaluation.get("metrics_summary", {})
        
        summary = {
            "roi": metrics.get("roi", 0),
            "annualized_return": metrics.get(BacktestMetric.ANNUALIZED_RETURN.value, 0),
            "sharpe_ratio": metrics.get(BacktestMetric.SHARPE_RATIO.value, 0),
            "max_drawdown": metrics.get(BacktestMetric.MAX_DRAWDOWN.value, 0),
            "win_rate": metrics.get(BacktestMetric.WIN_RATE.value, 0),
            "profit_factor": metrics.get(BacktestMetric.PROFIT_FACTOR.value, 0),
            "initial_balance": metrics.get("initial_balance", 0),
            "final_balance": metrics.get("final_balance", 0),
            "symbols_count": metrics.get("symbols_count", 0)
        }
        
        report["summary"] = summary
        
        # Chỉ số chi tiết
        report["detailed_metrics"] = metrics.copy()
        
        # Phân tích
        analysis = {
            "strengths": evaluation.get("strengths", []),
            "weaknesses": evaluation.get("weaknesses", []),
            "symbol_performance": {}
        }
        
        # Thêm thông tin về hiệu suất từng symbol
        symbol_evals = evaluation.get("symbol_evaluations", {})
        
        for symbol, symbol_eval in symbol_evals.items():
            analysis["symbol_performance"][symbol] = {
                "roi": symbol_eval.get("metrics", {}).get("roi", 0),
                "sharpe_ratio": symbol_eval.get("metrics", {}).get(BacktestMetric.SHARPE_RATIO.value, 0),
                "max_drawdown": symbol_eval.get("metrics", {}).get(BacktestMetric.MAX_DRAWDOWN.value, 0),
                "win_rate": symbol_eval.get("metrics", {}).get(BacktestMetric.WIN_RATE.value, 0),
                "strengths": symbol_eval.get("strengths", []),
                "weaknesses": symbol_eval.get("weaknesses", [])
            }
        
        report["analysis"] = analysis
        
        # Đề xuất
        report["recommendations"] = evaluation.get("recommendations", [])
        
        # Biểu đồ (nếu cần)
        if include_plots:
            # Thư mục để lưu biểu đồ
            plot_dir = None
            if output_path:
                plot_dir = Path(output_path).parent / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Vẽ biểu đồ chỉ số của từng symbol
                fig = self.plot_strategy_metrics(
                    strategy_name=strategy_name,
                    show_plot=False
                )
                
                if fig and plot_dir:
                    plot_path = plot_dir / f"{strategy_name}_metrics.png"
                    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    report["plots"].append(str(plot_path))
            
            except Exception as e:
                self.logger.warning(f"Lỗi khi tạo biểu đồ: {e}")
        
        # Lưu báo cáo nếu cần
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=4)
            
            self.logger.info(f"Đã lưu báo cáo tại {output_path}")
        
        return report
    
    def generate_strategies_ranking(
        self,
        evaluations: Optional[Dict[str, Dict[str, Any]]] = None,
        metrics: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Tạo bảng xếp hạng các chiến lược dựa trên nhiều chỉ số.
        
        Args:
            evaluations: Dict ánh xạ tên chiến lược -> kết quả đánh giá
            metrics: Danh sách chỉ số để xếp hạng
            weights: Trọng số cho từng chỉ số
            ascending: Sắp xếp tăng dần hay không
            
        Returns:
            DataFrame bảng xếp hạng
        """
        if evaluations is None:
            if not self.evaluated_results:
                evaluations = self.evaluate_all_strategies()
            else:
                evaluations = self.evaluated_results
        
        if not evaluations:
            self.logger.error("Không có chiến lược nào để xếp hạng")
            return pd.DataFrame()
        
        # Thiết lập metrics mặc định
        if metrics is None:
            metrics = [
                BacktestMetric.SHARPE_RATIO.value,
                BacktestMetric.SORTINO_RATIO.value,
                "roi",
                BacktestMetric.MAX_DRAWDOWN.value,
                BacktestMetric.WIN_RATE.value,
                BacktestMetric.PROFIT_FACTOR.value
            ]
        
        # Thiết lập trọng số mặc định
        if weights is None:
            weights = {
                BacktestMetric.SHARPE_RATIO.value: 0.25,
                BacktestMetric.SORTINO_RATIO.value: 0.15,
                "roi": 0.2,
                BacktestMetric.MAX_DRAWDOWN.value: 0.15,
                BacktestMetric.WIN_RATE.value: 0.1,
                BacktestMetric.PROFIT_FACTOR.value: 0.15
            }
        
        # Lọc các metric hợp lệ
        valid_metrics = [m for m in metrics if m in weights]
        
        # Nếu không có metrics hợp lệ, sử dụng Sharpe Ratio làm tiêu chí duy nhất
        if not valid_metrics:
            valid_metrics = [BacktestMetric.SHARPE_RATIO.value]
            weights = {BacktestMetric.SHARPE_RATIO.value: 1.0}
        
        # Chuẩn hóa trọng số
        total_weight = sum(weights[m] for m in valid_metrics)
        normalized_weights = {m: weights[m] / total_weight for m in valid_metrics}
        
        # Dữ liệu xếp hạng
        ranking_data = []
        
        # Lặp qua từng chiến lược
        for strategy_name, evaluation in evaluations.items():
            if evaluation.get("status") != "success":
                continue
            
            metrics_summary = evaluation.get("metrics_summary", {})
            
            row = {"strategy": strategy_name}
            
            # Thêm các metrics
            strategy_score = 0
            metrics_scores = {}
            
            for metric in valid_metrics:
                if metric in metrics_summary:
                    value = metrics_summary[metric]
                    
                    # Đảo giá trị cho một số metrics (nhỏ hơn là tốt hơn)
                    if metric in [BacktestMetric.MAX_DRAWDOWN.value]:
                        value = -value
                    
                    row[metric] = value
                    metrics_scores[metric] = value
            
            ranking_data.append(row)
        
        # Tạo DataFrame
        ranking_df = pd.DataFrame(ranking_data)
        
        if ranking_df.empty:
            return ranking_df
        
        # Chuẩn hóa các chỉ số
        normalized_df = ranking_df.copy()
        for metric in valid_metrics:
            if metric in ranking_df.columns:
                if ranking_df[metric].std() > 0:
                    normalized_df[f"{metric}_norm"] = (ranking_df[metric] - ranking_df[metric].mean()) / ranking_df[metric].std()
                else:
                    normalized_df[f"{metric}_norm"] = 0
        
        # Tính điểm tổng hợp
        total_scores = []
        
        for _, row in normalized_df.iterrows():
            score = 0
            for metric in valid_metrics:
                norm_col = f"{metric}_norm"
                if norm_col in normalized_df.columns:
                    score += row[norm_col] * normalized_weights[metric]
            total_scores.append(score)
        
        ranking_df["total_score"] = total_scores
        
        # Sắp xếp
        ranking_df.sort_values(by="total_score", ascending=not ascending, inplace=True)
        
        # Thêm cột xếp hạng
        ranking_df["rank"] = range(1, len(ranking_df) + 1)
        
        # Sắp xếp lại các cột
        cols = ["rank", "strategy", "total_score"] + [m for m in valid_metrics if m in ranking_df.columns]
        ranking_df = ranking_df[cols]
        
        return ranking_df