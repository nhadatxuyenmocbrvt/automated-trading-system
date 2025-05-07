"""
Chỉ số đánh giá cho quá trình tự động cải tiến.
File này định nghĩa các lớp và hàm để đánh giá so sánh hiệu suất giữa các phiên bản mô hình
nhằm tự động đưa ra quyết định về việc tái huấn luyện và cải tiến.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import os
from sklearn.metrics import confusion_matrix, classification_report
from itertools import combinations

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from backtesting.performance_metrics import PerformanceMetrics

class EvaluationMetrics:
    """
    Lớp đánh giá các chỉ số so sánh hiệu suất giữa các phiên bản mô hình.
    Cung cấp các công cụ để so sánh và đưa ra quyết định tự động về việc cải tiến mô hình.
    """
    
    def __init__(
        self,
        min_improvement_threshold: float = 0.05,
        confidence_interval: float = 0.95,
        min_evaluation_periods: int = 30,
        stability_threshold: float = 0.8,
        evaluation_metrics: List[str] = None,
        metric_weights: Dict[str, float] = None,
        version_history_file: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo đối tượng EvaluationMetrics.
        
        Args:
            min_improvement_threshold (float): Ngưỡng cải thiện tối thiểu để chấp nhận mô hình mới (%)
            confidence_interval (float): Khoảng tin cậy cho các đánh giá thống kê
            min_evaluation_periods (int): Số lượng giai đoạn đánh giá tối thiểu trước khi đưa ra quyết định
            stability_threshold (float): Ngưỡng ổn định cho các mô hình mới
            evaluation_metrics (List[str]): Danh sách các chỉ số đánh giá cần xem xét
            metric_weights (Dict[str, float]): Trọng số cho từng chỉ số đánh giá
            version_history_file (Union[str, Path]): File lưu lịch sử phiên bản
            logger (logging.Logger): Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("evaluation_metrics")
        
        # Lưu các tham số
        self.min_improvement_threshold = min_improvement_threshold
        self.confidence_interval = confidence_interval
        self.min_evaluation_periods = min_evaluation_periods
        self.stability_threshold = stability_threshold
        
        # Thiết lập các chỉ số đánh giá mặc định nếu không được cung cấp
        self.evaluation_metrics = evaluation_metrics or [
            "sharpe_ratio",     # Chỉ số Sharpe
            "sortino_ratio",    # Chỉ số Sortino
            "win_rate",         # Tỷ lệ thắng
            "profit_factor",    # Hệ số lợi nhuận
            "max_drawdown",     # Sụt giảm tối đa
            "expectancy",       # Kỳ vọng
            "total_return",     # Tổng lợi nhuận
            "volatility",       # Biến động
            "calmar_ratio"      # Chỉ số Calmar
        ]
        
        # Thiết lập trọng số cho các chỉ số đánh giá (phải tổng bằng 1)
        self.metric_weights = metric_weights or {
            "sharpe_ratio": 0.15,
            "sortino_ratio": 0.15,
            "win_rate": 0.10,
            "profit_factor": 0.15,
            "max_drawdown": 0.10,
            "expectancy": 0.10,
            "total_return": 0.15,
            "volatility": 0.05,
            "calmar_ratio": 0.05
        }
        
        # Kiểm tra tổng trọng số
        total_weight = sum(self.metric_weights.values())
        if abs(total_weight - 1.0) > 1e-10:
            self.logger.warning(f"Tổng trọng số ({total_weight}) không bằng 1.0. Tự động chuẩn hóa.")
            self.metric_weights = {k: v / total_weight for k, v in self.metric_weights.items()}
        
        # File lưu lịch sử phiên bản
        self.version_history_file = Path(version_history_file) if version_history_file else None
        
        # Lịch sử phiên bản
        self.version_history = {
            "versions": [],
            "model_info": [],
            "metrics": [],
            "timestamps": [],
            "comparison_results": []
        }
        
        # Trạng thái hiện tại
        self.current_model_version = None
        self.current_model_metrics = None
        
        # Tải lịch sử phiên bản nếu có
        if self.version_history_file and self.version_history_file.exists():
            self.load_version_history(self.version_history_file)
    
    def add_model_version(
        self,
        version: str,
        model_info: Dict[str, Any],
        metrics: Dict[str, Any],
        replace_if_exists: bool = False
    ) -> None:
        """
        Thêm phiên bản mô hình mới vào lịch sử.
        
        Args:
            version (str): Phiên bản mô hình (ví dụ: "v1.0.1")
            model_info (Dict[str, Any]): Thông tin về mô hình
            metrics (Dict[str, Any]): Các chỉ số hiệu suất của mô hình
            replace_if_exists (bool): Thay thế phiên bản nếu đã tồn tại
        """
        # Kiểm tra xem phiên bản đã tồn tại chưa
        if version in self.version_history["versions"] and not replace_if_exists:
            self.logger.warning(f"Phiên bản '{version}' đã tồn tại. Sử dụng replace_if_exists=True để ghi đè.")
            return
        elif version in self.version_history["versions"] and replace_if_exists:
            # Xóa phiên bản cũ
            idx = self.version_history["versions"].index(version)
            self.version_history["versions"].pop(idx)
            self.version_history["model_info"].pop(idx)
            self.version_history["metrics"].pop(idx)
            self.version_history["timestamps"].pop(idx)
            # Cập nhật kết quả so sánh
            self.version_history["comparison_results"] = [
                result for result in self.version_history["comparison_results"]
                if result["version_a"] != version and result["version_b"] != version
            ]
        
        # Thêm phiên bản mới
        self.version_history["versions"].append(version)
        self.version_history["model_info"].append(model_info)
        self.version_history["metrics"].append(metrics)
        self.version_history["timestamps"].append(datetime.now())
        
        # Cập nhật mô hình hiện tại
        self.current_model_version = version
        self.current_model_metrics = metrics
        
        # So sánh với phiên bản trước đó
        if len(self.version_history["versions"]) > 1:
            self._compare_with_all_previous_versions(version)
        
        self.logger.info(f"Đã thêm phiên bản mô hình mới: {version}")
        
        # Lưu lịch sử
        if self.version_history_file:
            self.save_version_history(self.version_history_file)
    
    def _compare_with_all_previous_versions(self, new_version: str) -> None:
        """
        So sánh phiên bản mới với tất cả các phiên bản trước đó.
        
        Args:
            new_version (str): Phiên bản mới cần so sánh
        """
        # Lấy chỉ số của phiên bản mới
        new_idx = self.version_history["versions"].index(new_version)
        new_metrics = self.version_history["metrics"][new_idx]
        
        # So sánh với tất cả phiên bản trước đó
        for i, version in enumerate(self.version_history["versions"]):
            if version == new_version:
                continue
            
            old_metrics = self.version_history["metrics"][i]
            
            # So sánh chi tiết
            comparison_result = self.compare_models(
                metrics_a=old_metrics,
                metrics_b=new_metrics,
                version_a=version,
                version_b=new_version
            )
            
            # Thêm kết quả so sánh vào lịch sử
            self.version_history["comparison_results"].append(comparison_result)
    
    def compare_models(
        self,
        metrics_a: Dict[str, Any],
        metrics_b: Dict[str, Any],
        version_a: Optional[str] = None,
        version_b: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        So sánh hiệu suất giữa hai mô hình.
        
        Args:
            metrics_a (Dict[str, Any]): Các chỉ số của mô hình A
            metrics_b (Dict[str, Any]): Các chỉ số của mô hình B
            version_a (str, optional): Phiên bản của mô hình A
            version_b (str, optional): Phiên bản của mô hình B
            
        Returns:
            Dict[str, Any]: Kết quả so sánh
        """
        version_a = version_a or "Model A"
        version_b = version_b or "Model B"
        
        # Chỉ so sánh các chỉ số được chọn
        metric_comparisons = {}
        weighted_score = 0.0
        
        for metric in self.evaluation_metrics:
            if metric in metrics_a and metric in metrics_b:
                # Lấy giá trị chỉ số
                value_a = metrics_a[metric]
                value_b = metrics_b[metric]
                
                # Tính % thay đổi
                if value_a != 0:
                    pct_change = (value_b - value_a) / abs(value_a)
                else:
                    pct_change = float('inf') if value_b > 0 else (float('-inf') if value_b < 0 else 0.0)
                
                # Xác định hướng tốt hơn (tăng hay giảm là tốt)
                better_if_higher = metric not in ["max_drawdown", "volatility"]
                is_better = (pct_change > 0) if better_if_higher else (pct_change < 0)
                
                # Chuẩn hóa giá trị thay đổi (phạm vi 0-1, 0.5 là trung tính)
                if better_if_higher:
                    # Đối với chỉ số càng cao càng tốt
                    normalized_improvement = 0.5 + (0.5 * min(pct_change, 1.0)) if pct_change > 0 else (0.5 + (0.5 * max(pct_change, -1.0)))
                else:
                    # Đối với chỉ số càng thấp càng tốt (đảo dấu)
                    normalized_improvement = 0.5 + (0.5 * min(-pct_change, 1.0)) if pct_change < 0 else (0.5 + (0.5 * max(-pct_change, -1.0)))
                
                # Tính điểm có trọng số
                weight = self.metric_weights.get(metric, 0.0)
                weighted_contribution = normalized_improvement * weight
                
                # Lưu kết quả so sánh
                metric_comparisons[metric] = {
                    "value_a": value_a,
                    "value_b": value_b,
                    "pct_change": pct_change,
                    "is_better": is_better,
                    "normalized_improvement": normalized_improvement,
                    "weight": weight,
                    "weighted_contribution": weighted_contribution
                }
                
                weighted_score += weighted_contribution
        
        # Xác định có nên khuyến nghị mô hình B không
        is_significant_improvement = weighted_score > 0.5 + self.min_improvement_threshold
        
        # Tạo kết quả tổng thể
        result = {
            "timestamp": datetime.now(),
            "version_a": version_a,
            "version_b": version_b,
            "metric_comparisons": metric_comparisons,
            "weighted_score": weighted_score,
            "is_significant_improvement": is_significant_improvement,
            "recommendation": "adopt_new_model" if is_significant_improvement else "keep_current_model"
        }
        
        self.logger.info(f"So sánh mô hình {version_a} vs {version_b}: Điểm cải thiện = {weighted_score:.4f}, Khuyến nghị: {result['recommendation']}")
        
        return result
    
    def evaluate_model_stability(
        self,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        trades: Optional[pd.DataFrame] = None,
        equity_curve: Optional[pd.Series] = None,
        min_trades: int = 100
    ) -> Dict[str, Any]:
        """
        Đánh giá độ ổn định của mô hình.
        
        Args:
            version (str, optional): Phiên bản mô hình cần đánh giá (None = mô hình hiện tại)
            metrics (Dict[str, Any], optional): Các chỉ số của mô hình
            trades (pd.DataFrame, optional): DataFrame chứa thông tin giao dịch
            equity_curve (pd.Series, optional): Chuỗi giá trị vốn
            min_trades (int, optional): Số lượng giao dịch tối thiểu cần đánh giá
            
        Returns:
            Dict[str, Any]: Kết quả đánh giá độ ổn định
        """
        if version is None and self.current_model_version is not None:
            version = self.current_model_version
            metrics = self.current_model_metrics
        elif version is not None and metrics is None:
            # Tìm metrics từ lịch sử nếu chỉ cung cấp phiên bản
            if version in self.version_history["versions"]:
                idx = self.version_history["versions"].index(version)
                metrics = self.version_history["metrics"][idx]
            else:
                self.logger.warning(f"Không tìm thấy phiên bản '{version}' trong lịch sử")
                return {"error": f"Không tìm thấy phiên bản '{version}'"}
        
        # Kiểm tra dữ liệu
        if metrics is None:
            self.logger.warning("Không có dữ liệu metrics để đánh giá độ ổn định")
            return {"error": "Không có dữ liệu metrics"}
        
        # Đánh giá độ ổn định dựa trên chuỗi vốn và giao dịch
        stability_scores = {}
        
        # 1. Đánh giá dựa trên metrics
        stability_scores["metrics_stability"] = self._evaluate_metrics_stability(metrics)
        
        # 2. Đánh giá dựa trên giao dịch (nếu có)
        if trades is not None and len(trades) >= min_trades:
            stability_scores["trades_stability"] = self._evaluate_trades_stability(trades)
        
        # 3. Đánh giá dựa trên chuỗi vốn (nếu có)
        if equity_curve is not None and len(equity_curve) >= 30:
            stability_scores["equity_stability"] = self._evaluate_equity_stability(equity_curve)
        
        # Tính điểm ổn định tổng thể
        weights = {
            "metrics_stability": 0.3,
            "trades_stability": 0.4,
            "equity_stability": 0.3
        }
        
        # Tính điểm trung bình có trọng số từ các điểm ổn định có sẵn
        available_scores = {k: v["score"] for k, v in stability_scores.items() if "score" in v}
        
        if available_scores:
            total_weight = sum(weights[k] for k in available_scores.keys())
            weighted_score = sum(weights[k] * v for k, v in available_scores.items()) / total_weight
        else:
            weighted_score = 0.0
        
        # Quyết định mô hình có ổn định không
        is_stable = weighted_score >= self.stability_threshold
        
        result = {
            "version": version,
            "stability_scores": stability_scores,
            "overall_stability_score": weighted_score,
            "is_stable": is_stable,
            "stability_threshold": self.stability_threshold,
            "timestamp": datetime.now()
        }
        
        self.logger.info(f"Đánh giá độ ổn định mô hình {version}: Điểm = {weighted_score:.4f}, Ổn định: {is_stable}")
        
        return result
    
    def _evaluate_metrics_stability(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Đánh giá độ ổn định dựa trên các chỉ số.
        
        Args:
            metrics (Dict[str, Any]): Các chỉ số của mô hình
            
        Returns:
            Dict[str, Any]: Đánh giá độ ổn định của chỉ số
        """
        # Các chỉ số ổn định cần xem xét
        stability_metrics = {
            "sharpe_ratio": (0.5, np.inf),     # Sharpe > 0.5 là tốt
            "sortino_ratio": (0.7, np.inf),    # Sortino > 0.7 là tốt
            "win_rate": (0.4, 0.7),            # Win rate 40-70% là cân bằng và ổn định
            "profit_factor": (1.2, 3.0),       # Profit factor 1.2-3.0 là ổn định
            "max_drawdown": (0, 0.25),         # Max drawdown < 25% là ổn định
            "expectancy": (0, np.inf),         # Expectancy > 0 là ổn định
            "calmar_ratio": (0.5, np.inf)      # Calmar > 0.5 là ổn định
        }
        
        scores = {}
        for metric, (min_val, max_val) in stability_metrics.items():
            if metric in metrics:
                value = metrics[metric]
                
                # Tính điểm ổn định (0-1)
                if min_val == 0 and max_val == np.inf:
                    # Chỉ cần > 0
                    score = 1.0 if value > 0 else 0.0
                elif min_val == 0:
                    # Càng thấp càng tốt, nhưng phải >= 0
                    score = 1.0 - min(value / max_val, 1.0)
                elif max_val == np.inf:
                    # Càng cao càng tốt, nhưng phải >= min_val
                    score = min(value / (min_val * 2), 1.0) if value >= min_val else (value / min_val * 0.5)
                else:
                    # Nằm trong khoảng [min_val, max_val] là tốt nhất
                    if value < min_val:
                        score = value / min_val * 0.5  # 0-0.5 nếu < min_val
                    elif value > max_val:
                        score = 0.5 * (1.0 - min((value - max_val) / max_val, 1.0))  # 0-0.5 nếu > max_val
                    else:
                        # 0.5-1.0 nếu nằm trong khoảng
                        position = (value - min_val) / (max_val - min_val)
                        # Hàm chuông (parabola) để đạt tối đa ở giữa khoảng
                        score = 0.5 + 0.5 * (1.0 - 4.0 * (position - 0.5)**2)
                
                scores[metric] = {
                    "value": value,
                    "acceptable_range": (min_val, max_val),
                    "stability_score": score
                }
        
        # Tính điểm ổn định tổng thể
        overall_score = sum(item["stability_score"] for item in scores.values()) / max(len(scores), 1)
        
        return {
            "score": overall_score,
            "metric_scores": scores,
            "is_stable": overall_score >= self.stability_threshold
        }
    
    def _evaluate_trades_stability(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Đánh giá độ ổn định dựa trên lịch sử giao dịch.
        
        Args:
            trades (pd.DataFrame): DataFrame chứa thông tin giao dịch
            
        Returns:
            Dict[str, Any]: Đánh giá độ ổn định của giao dịch
        """
        stability_scores = {}
        
        # 1. Tính điểm ổn định dựa trên phân phối lợi nhuận
        if 'profit' in trades.columns:
            profits = trades['profit']
            
            # Tính tỷ lệ thắng theo thời gian
            trades_count = len(trades)
            window_size = max(20, trades_count // 10)
            rolling_win_rate = profits.rolling(window=window_size).apply(lambda x: (x > 0).mean())
            
            # Tính độ biến động của tỷ lệ thắng
            win_rate_std = rolling_win_rate.std()
            win_rate_stability = 1.0 - min(win_rate_std * 5, 1.0)  # Biến động thấp = ổn định cao
            
            # Tính phân phối lợi nhuận
            profit_mean = profits.mean()
            profit_std = profits.std()
            profit_skew = profits.skew()
            profit_kurtosis = profits.kurtosis()
            
            # Tính hệ số biến thiên (CV)
            cv = abs(profit_std / profit_mean) if profit_mean != 0 else float('inf')
            cv_stability = 1.0 - min(cv / 3, 1.0)  # CV thấp = ổn định cao
            
            # Điểm phân phối lợi nhuận (favor positive skew)
            skew_score = 0.5 + min(profit_skew / 3, 0.5) if profit_skew > 0 else (0.5 - min(abs(profit_skew) / 3, 0.5))
            
            # Kurtosis quá cao hoặc quá thấp đều không tốt
            kurtosis_score = 1.0 - min(abs(profit_kurtosis - 3) / 5, 1.0)
            
            # Tính điểm tổng hợp
            profit_stability = 0.3 * win_rate_stability + 0.3 * cv_stability + 0.25 * skew_score + 0.15 * kurtosis_score
            
            stability_scores["profit_distribution"] = {
                "win_rate_stability": win_rate_stability,
                "cv_stability": cv_stability,
                "skew_score": skew_score,
                "kurtosis_score": kurtosis_score,
                "score": profit_stability
            }
        
        # 2. Tính điểm ổn định dựa trên các chuỗi chiến thắng/thua
        if 'profit' in trades.columns:
            # Xác định các chuỗi chiến thắng/thua
            wins = (profits > 0).astype(int)
            win_streaks = []
            loss_streaks = []
            
            current_streak = 1
            for i in range(1, len(wins)):
                if wins[i] == wins[i-1]:
                    current_streak += 1
                else:
                    if wins[i-1] == 1:
                        win_streaks.append(current_streak)
                    else:
                        loss_streaks.append(current_streak)
                    current_streak = 1
            
            # Thêm chuỗi cuối cùng
            if len(wins) > 0:
                if wins.iloc[-1] == 1:
                    win_streaks.append(current_streak)
                else:
                    loss_streaks.append(current_streak)
            
            # Tính các thống kê chuỗi
            max_win_streak = max(win_streaks) if win_streaks else 0
            max_loss_streak = max(loss_streaks) if loss_streaks else 0
            avg_win_streak = sum(win_streaks) / len(win_streaks) if win_streaks else 0
            avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0
            
            # Tính điểm ổn định chuỗi
            # Chuỗi thắng dài là tốt, chuỗi thua ngắn là tốt
            win_streak_score = min(avg_win_streak / 5, 1.0)
            loss_streak_score = 1.0 - min(avg_loss_streak / 5, 1.0)
            
            # Hệ số đặc biệt cho chuỗi thua dài quá mức
            exceptional_loss_penalty = 1.0 - min(max_loss_streak / 15, 0.9) if max_loss_streak > 8 else 0.0
            
            # Tính điểm tổng hợp
            streak_stability = 0.4 * win_streak_score + 0.4 * loss_streak_score - exceptional_loss_penalty
            streak_stability = max(0.0, min(1.0, streak_stability))
            
            stability_scores["streaks"] = {
                "max_win_streak": max_win_streak,
                "max_loss_streak": max_loss_streak,
                "avg_win_streak": avg_win_streak,
                "avg_loss_streak": avg_loss_streak,
                "win_streak_score": win_streak_score,
                "loss_streak_score": loss_streak_score,
                "exceptional_loss_penalty": exceptional_loss_penalty,
                "score": streak_stability
            }
        
        # 3. Tính điểm ổn định dựa trên tính thời vụ (seasonality)
        if 'entry_time' in trades.columns and 'profit' in trades.columns:
            try:
                trades_with_time = trades.copy()
                trades_with_time['entry_time'] = pd.to_datetime(trades_with_time['entry_time'])
                
                # Tính lợi nhuận theo ngày trong tuần
                trades_with_time['day_of_week'] = trades_with_time['entry_time'].dt.dayofweek
                day_profits = trades_with_time.groupby('day_of_week')['profit'].mean()
                
                # Tính lợi nhuận theo giờ trong ngày
                trades_with_time['hour'] = trades_with_time['entry_time'].dt.hour
                hour_profits = trades_with_time.groupby('hour')['profit'].mean()
                
                # Tính biến động ngày/giờ (thấp = ổn định cao)
                day_variation = day_profits.std() / day_profits.mean() if day_profits.mean() != 0 else float('inf')
                hour_variation = hour_profits.std() / hour_profits.mean() if hour_profits.mean() != 0 else float('inf')
                
                # Chuyển đổi thành điểm ổn định
                day_stability = 1.0 - min(day_variation, 1.0)
                hour_stability = 1.0 - min(hour_variation, 1.0)
                
                # Tính điểm tổng hợp
                seasonality_stability = 0.5 * day_stability + 0.5 * hour_stability
                
                stability_scores["seasonality"] = {
                    "day_stability": day_stability,
                    "hour_stability": hour_stability,
                    "score": seasonality_stability
                }
            except Exception as e:
                self.logger.warning(f"Không thể tính điểm ổn định thời vụ: {str(e)}")
        
        # Tính điểm ổn định tổng thể cho giao dịch
        stability_weights = {
            "profit_distribution": 0.5,
            "streaks": 0.3,
            "seasonality": 0.2
        }
        
        # Tính điểm trung bình có trọng số
        available_scores = {k: v["score"] for k, v in stability_scores.items() if "score" in v}
        
        if available_scores:
            total_weight = sum(stability_weights[k] for k in available_scores.keys())
            overall_score = sum(stability_weights[k] * v for k, v in available_scores.items()) / total_weight
        else:
            overall_score = 0.0
        
        return {
            "score": overall_score,
            "stability_measures": stability_scores,
            "is_stable": overall_score >= self.stability_threshold
        }
    
    def _evaluate_equity_stability(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """
        Đánh giá độ ổn định dựa trên chuỗi vốn.
        
        Args:
            equity_curve (pd.Series): Chuỗi giá trị vốn
            
        Returns:
            Dict[str, Any]: Đánh giá độ ổn định của chuỗi vốn
        """
        stability_measures = {}
        
        # 1. Tính hệ số tăng trưởng ổn định
        returns = equity_curve.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        
        # Tính các thông số xu hướng
        linear_trend = np.polyfit(range(len(equity_curve)), equity_curve.values, 1)[0]
        normalized_trend = linear_trend / equity_curve.iloc[0]
        
        # Tính R-squared của xu hướng tuyến tính
        trend_line = np.polyval(np.polyfit(range(len(equity_curve)), equity_curve.values, 1), range(len(equity_curve)))
        ss_tot = sum((equity_curve.values - equity_curve.values.mean()) ** 2)
        ss_res = sum((equity_curve.values - trend_line) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Tính độ biến động quanh xu hướng
        trend_deviation = np.mean(np.abs(equity_curve.values - trend_line) / trend_line)
        
        # Tính điểm độ ổn định xu hướng
        trend_score = min(r_squared, 1.0)
        volatility_score = 1.0 - min(trend_deviation * 10, 1.0)
        trend_direction_score = min(max(normalized_trend * 10, 0.0), 1.0) if normalized_trend > 0 else 0.0
        
        # Tính điểm tổng hợp cho xu hướng
        trend_stability = 0.4 * trend_score + 0.3 * volatility_score + 0.3 * trend_direction_score
        
        stability_measures["trend_stability"] = {
            "r_squared": r_squared,
            "trend_deviation": trend_deviation,
            "normalized_trend": normalized_trend,
            "trend_score": trend_score,
            "volatility_score": volatility_score,
            "trend_direction_score": trend_direction_score,
            "score": trend_stability
        }
        
        # 2. Tính độ ổn định dựa trên drawdown
        # Tính peak-to-trough cho mỗi điểm
        rolling_max = equity_curve.cummax()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Phân tích các giai đoạn drawdown
        drawdown_periods = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdowns):
            if not in_drawdown and dd < 0:
                # Bắt đầu drawdown mới
                in_drawdown = True
                start_idx = i
            elif in_drawdown and dd == 0:
                # Kết thúc drawdown
                in_drawdown = False
                drawdown_periods.append({
                    'start_idx': start_idx,
                    'end_idx': i,
                    'duration': i - start_idx,
                    'depth': abs(drawdowns[start_idx:i].min())
                })
                start_idx = None
        
        # Nếu vẫn đang trong drawdown tại điểm cuối
        if in_drawdown:
            drawdown_periods.append({
                'start_idx': start_idx,
                'end_idx': len(drawdowns) - 1,
                'duration': len(drawdowns) - 1 - start_idx,
                'depth': abs(drawdowns[start_idx:].min())
            })
        
        # Tính các thống kê về drawdown
        if drawdown_periods:
            avg_drawdown_depth = sum(p['depth'] for p in drawdown_periods) / len(drawdown_periods)
            avg_drawdown_duration = sum(p['duration'] for p in drawdown_periods) / len(drawdown_periods)
            max_drawdown_duration = max(p['duration'] for p in drawdown_periods)
            
            # Tính điểm ổn định drawdown
            depth_score = 1.0 - min(avg_drawdown_depth * 5, 1.0)
            duration_score = 1.0 - min(avg_drawdown_duration / len(equity_curve) * 10, 1.0)
            max_duration_penalty = min(max_drawdown_duration / len(equity_curve) * 2, 0.5)
            
            # Tính điểm tổng hợp cho drawdown
            drawdown_stability = 0.5 * depth_score + 0.5 * duration_score - max_duration_penalty
            drawdown_stability = max(0.0, drawdown_stability)
        else:
            # Không có drawdown = điểm tuyệt đối
            drawdown_stability = 1.0
        
        stability_measures["drawdown_stability"] = {
            "max_drawdown": max_drawdown,
            "avg_drawdown_depth": avg_drawdown_depth if drawdown_periods else 0,
            "avg_drawdown_duration": avg_drawdown_duration if drawdown_periods else 0,
            "max_drawdown_duration": max_drawdown_duration if drawdown_periods else 0,
            "score": drawdown_stability
        }
        
        # 3. Tính độ ổn định dựa trên phân phối lợi nhuận
        returns_stability = {}
        if len(returns) > 30:
            # Tính các thông số thống kê
            returns_std = returns.std()
            returns_mean = returns.mean()
            returns_skew = returns.skew()
            returns_kurt = returns.kurtosis()
            
            # Tính hệ số biến thiên (CV)
            cv = abs(returns_std / returns_mean) if returns_mean != 0 else float('inf')
            
            # Tính điểm ổn định
            cv_score = 1.0 - min(cv / 5, 1.0)
            sharpe_score = min(max(sharpe / 2, 0.0), 1.0)
            skew_score = 0.5 + min(returns_skew / 3, 0.5) if returns_skew > 0 else (0.5 - min(abs(returns_skew) / 3, 0.5))
            kurt_score = 1.0 - min(abs(returns_kurt - 3) / 5, 1.0)
            
            # Tính điểm tổng hợp cho phân phối lợi nhuận
            returns_stability = {
                "std": returns_std,
                "mean": returns_mean,
                "sharpe": sharpe,
                "skew": returns_skew,
                "kurtosis": returns_kurt,
                "cv": cv,
                "cv_score": cv_score,
                "sharpe_score": sharpe_score,
                "skew_score": skew_score,
                "kurt_score": kurt_score,
                "score": 0.3 * cv_score + 0.3 * sharpe_score + 0.2 * skew_score + 0.2 * kurt_score
            }
        
        # Tính điểm ổn định tổng thể cho chuỗi vốn
        measures = {
            "trend_stability": stability_measures["trend_stability"]["score"],
            "drawdown_stability": stability_measures["drawdown_stability"]["score"]
        }
        
        if returns_stability:
            measures["returns_stability"] = returns_stability["score"]
        
        # Tính điểm trung bình
        overall_score = sum(measures.values()) / len(measures)
        
        return {
            "score": overall_score,
            "stability_measures": stability_measures,
            "returns_stability": returns_stability if returns_stability else None,
            "is_stable": overall_score >= self.stability_threshold
        }
    
    def should_promote_model(self, version: str) -> Dict[str, Any]:
        """
        Xác định xem có nên khuyến khích sử dụng phiên bản mô hình mới không.
        
        Args:
            version (str): Phiên bản mô hình cần xem xét
            
        Returns:
            Dict[str, Any]: Kết quả đánh giá và khuyến nghị
        """
        if version not in self.version_history["versions"]:
            return {"error": f"Phiên bản '{version}' không tồn tại trong lịch sử"}
        
        # Tìm tất cả kết quả so sánh liên quan đến phiên bản này
        comparisons = [
            result for result in self.version_history["comparison_results"]
            if result["version_a"] == version or result["version_b"] == version
        ]
        
        if not comparisons:
            return {"error": f"Chưa có kết quả so sánh nào cho phiên bản '{version}'"}
        
        # Đếm số lần phiên bản này tốt hơn/kém hơn các phiên bản khác
        better_than = []
        worse_than = []
        
        for comp in comparisons:
            if comp["version_a"] == version and comp["recommendation"] == "keep_current_model":
                # Phiên bản này (A) không tốt hơn phiên bản khác (B)
                worse_than.append(comp["version_b"])
            elif comp["version_b"] == version and comp["recommendation"] == "adopt_new_model":
                # Phiên bản này (B) tốt hơn phiên bản khác (A)
                better_than.append(comp["version_a"])
            elif comp["version_a"] == version and comp["recommendation"] == "adopt_new_model":
                # Phiên bản khác (B) không tốt hơn phiên bản này (A)
                better_than.append(comp["version_b"])
            elif comp["version_b"] == version and comp["recommendation"] == "keep_current_model":
                # Phiên bản khác (A) tốt hơn phiên bản này (B)
                worse_than.append(comp["version_a"])
        
        # Tính số lượng phiên bản được so sánh
        compared_with = list(set(better_than + worse_than))
        num_compared = len(compared_with)
        
        if num_compared == 0:
            return {"error": f"Không có so sánh hợp lệ cho phiên bản '{version}'"}
        
        # Tính tỷ lệ tốt hơn/kém hơn
        num_better = len(set(better_than))
        num_worse = len(set(worse_than))
        
        better_ratio = num_better / num_compared
        worse_ratio = num_worse / num_compared
        
        # Xác định có nên khuyến khích sử dụng không
        should_promote = better_ratio > 0.7 and worse_ratio < 0.3
        
        # Tìm phiên bản tốt nhất hiện tại
        current_best = self._find_best_model_version()
        
        result = {
            "version": version,
            "num_compared": num_compared,
            "better_than": list(set(better_than)),
            "worse_than": list(set(worse_than)),
            "better_ratio": better_ratio,
            "worse_ratio": worse_ratio,
            "should_promote": should_promote,
            "current_best_version": current_best,
            "is_current_best": version == current_best,
            "recommendation": "promote" if should_promote else "do_not_promote"
        }
        
        return result
    
    def _find_best_model_version(self) -> Optional[str]:
        """
        Tìm phiên bản mô hình tốt nhất dựa trên kết quả so sánh.
        
        Returns:
            str: Phiên bản mô hình tốt nhất
        """
        if not self.version_history["versions"]:
            return None
        
        # Tính "điểm" cho mỗi phiên bản dựa trên số lần thắng/thua
        scores = {version: 0 for version in self.version_history["versions"]}
        
        for comp in self.version_history["comparison_results"]:
            if comp["recommendation"] == "adopt_new_model":
                # Phiên bản B tốt hơn phiên bản A
                scores[comp["version_b"]] += 1
                scores[comp["version_a"]] -= 1
            else:
                # Phiên bản A tốt hơn phiên bản B
                scores[comp["version_a"]] += 1
                scores[comp["version_b"]] -= 1
        
        # Tìm phiên bản có điểm cao nhất
        best_version = max(scores.items(), key=lambda x: x[1])[0]
        
        return best_version
    
    def get_model_performance_history(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy lịch sử hiệu suất của một hoặc tất cả các phiên bản mô hình.
        
        Args:
            version (str, optional): Phiên bản mô hình cần xem lịch sử (None = tất cả)
            
        Returns:
            Dict[str, Any]: Lịch sử hiệu suất
        """
        if version is not None and version not in self.version_history["versions"]:
            return {"error": f"Phiên bản '{version}' không tồn tại trong lịch sử"}
        
        if version is not None:
            # Lấy thông tin cho một phiên bản cụ thể
            idx = self.version_history["versions"].index(version)
            
            performance_history = {
                "version": version,
                "model_info": self.version_history["model_info"][idx],
                "metrics": self.version_history["metrics"][idx],
                "timestamp": self.version_history["timestamps"][idx],
                "comparisons": [
                    comp for comp in self.version_history["comparison_results"]
                    if comp["version_a"] == version or comp["version_b"] == version
                ]
            }
        else:
            # Lấy thông tin cho tất cả phiên bản
            performance_history = {
                "versions": self.version_history["versions"],
                "metrics": {
                    version: self.version_history["metrics"][i]
                    for i, version in enumerate(self.version_history["versions"])
                },
                "comparisons": self.version_history["comparison_results"]
            }
        
        return performance_history
    
    def get_metric_trends(self, metrics: List[str] = None) -> Dict[str, Any]:
        """
        Phân tích xu hướng của các chỉ số qua các phiên bản.
        
        Args:
            metrics (List[str], optional): Danh sách các chỉ số cần phân tích
            
        Returns:
            Dict[str, Any]: Phân tích xu hướng
        """
        if not self.version_history["versions"]:
            return {"error": "Không có dữ liệu phiên bản trong lịch sử"}
        
        metrics_to_analyze = metrics or self.evaluation_metrics
        
        # Tập hợp dữ liệu theo thời gian
        versions = self.version_history["versions"]
        timestamps = self.version_history["timestamps"]
        
        # Sắp xếp các phiên bản theo thời gian
        sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
        sorted_versions = [versions[i] for i in sorted_indices]
        
        # Tập hợp dữ liệu cho từng chỉ số
        metric_data = {}
        
        for metric in metrics_to_analyze:
            values = []
            valid_versions = []
            
            for i in sorted_indices:
                if metric in self.version_history["metrics"][i]:
                    values.append(self.version_history["metrics"][i][metric])
                    valid_versions.append(versions[i])
            
            if not values:
                continue
                
            # Tính xu hướng (hệ số góc)
            try:
                x = np.arange(len(values))
                slope, intercept = np.polyfit(x, values, 1)
                
                # Tính R-squared
                y_pred = slope * x + intercept
                r_squared = 1 - (sum((np.array(values) - y_pred) ** 2) / 
                                sum((np.array(values) - np.mean(values)) ** 2))
                
                # Xác định hướng xu hướng
                better_if_higher = metric not in ["max_drawdown", "volatility"]
                is_improving = (slope > 0) if better_if_higher else (slope < 0)
                
                metric_data[metric] = {
                    "versions": valid_versions,
                    "values": values,
                    "trend": {
                        "slope": slope,
                        "intercept": intercept,
                        "r_squared": r_squared,
                        "is_improving": is_improving
                    }
                }
            except:
                # Nếu không thể tính xu hướng
                metric_data[metric] = {
                    "versions": valid_versions,
                    "values": values,
                    "trend": None
                }
        
        return {
            "metrics": metric_data,
            "versions_chronological": sorted_versions
        }
    
    def save_version_history(self, path: Union[str, Path]) -> None:
        """
        Lưu lịch sử phiên bản vào file.
        
        Args:
            path (Union[str, Path]): Đường dẫn file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Chuyển đổi dữ liệu để lưu
        serializable_history = {
            "versions": self.version_history["versions"],
            "model_info": self.version_history["model_info"],
            "metrics": self.version_history["metrics"],
            "timestamps": [t.isoformat() for t in self.version_history["timestamps"]],
            "comparison_results": []
        }
        
        # Chuyển đổi kết quả so sánh
        for result in self.version_history["comparison_results"]:
            serializable_result = result.copy()
            serializable_result["timestamp"] = serializable_result["timestamp"].isoformat()
            serializable_history["comparison_results"].append(serializable_result)
        
        # Lưu thông tin cấu hình
        serializable_history["config"] = {
            "min_improvement_threshold": self.min_improvement_threshold,
            "confidence_interval": self.confidence_interval,
            "min_evaluation_periods": self.min_evaluation_periods,
            "stability_threshold": self.stability_threshold,
            "evaluation_metrics": self.evaluation_metrics,
            "metric_weights": self.metric_weights
        }
        
        # Lưu thông tin hiện tại
        serializable_history["current_model_version"] = self.current_model_version
        serializable_history["current_model_metrics"] = self.current_model_metrics
        serializable_history["last_updated"] = datetime.now().isoformat()
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=4, ensure_ascii=False)
        
        self.logger.info(f"Đã lưu lịch sử phiên bản vào {path}")
    
    def load_version_history(self, path: Union[str, Path]) -> bool:
        """
        Tải lịch sử phiên bản từ file.
        
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
            
            # Khôi phục dữ liệu
            self.version_history["versions"] = data["versions"]
            self.version_history["model_info"] = data["model_info"]
            self.version_history["metrics"] = data["metrics"]
            self.version_history["timestamps"] = [datetime.fromisoformat(t) for t in data["timestamps"]]
            
            # Khôi phục kết quả so sánh
            self.version_history["comparison_results"] = []
            for result in data["comparison_results"]:
                result_copy = result.copy()
                result_copy["timestamp"] = datetime.fromisoformat(result_copy["timestamp"])
                self.version_history["comparison_results"].append(result_copy)
            
            # Khôi phục cấu hình
            if "config" in data:
                self.min_improvement_threshold = data["config"].get("min_improvement_threshold", self.min_improvement_threshold)
                self.confidence_interval = data["config"].get("confidence_interval", self.confidence_interval)
                self.min_evaluation_periods = data["config"].get("min_evaluation_periods", self.min_evaluation_periods)
                self.stability_threshold = data["config"].get("stability_threshold", self.stability_threshold)
                self.evaluation_metrics = data["config"].get("evaluation_metrics", self.evaluation_metrics)
                self.metric_weights = data["config"].get("metric_weights", self.metric_weights)
            
            # Khôi phục thông tin hiện tại
            self.current_model_version = data.get("current_model_version")
            self.current_model_metrics = data.get("current_model_metrics")
            
            self.logger.info(f"Đã tải lịch sử phiên bản từ {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải lịch sử phiên bản: {str(e)}")
            return False
    
    def plot_version_comparison(
        self,
        versions: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Vẽ biểu đồ so sánh các phiên bản mô hình.
        
        Args:
            versions (List[str], optional): Danh sách các phiên bản cần so sánh
            metrics (List[str], optional): Danh sách các chỉ số cần so sánh
            figsize (Tuple[int, int], optional): Kích thước hình
            
        Returns:
            plt.Figure: Đối tượng Figure của matplotlib
        """
        if not self.version_history["versions"]:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Không có dữ liệu phiên bản trong lịch sử", 
                 horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Xác định phiên bản cần so sánh
        versions_to_compare = versions or self.version_history["versions"]
        metrics_to_compare = metrics or self.evaluation_metrics[:6]  # Lấy 6 chỉ số đầu tiên nếu không chỉ định
        
        # Lọc các phiên bản hợp lệ
        valid_versions = [v for v in versions_to_compare if v in self.version_history["versions"]]
        
        if not valid_versions:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Không có phiên bản hợp lệ để so sánh", 
                 horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Chuẩn bị dữ liệu cho biểu đồ
        version_indices = [self.version_history["versions"].index(v) for v in valid_versions]
        
        # Tính số hàng và cột cho subplots
        num_metrics = len(metrics_to_compare)
        num_cols = min(3, num_metrics)
        num_rows = (num_metrics + num_cols - 1) // num_cols
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
        if num_rows == 1 and num_cols == 1:
            axs = [axs]
        elif num_rows == 1 or num_cols == 1:
            axs = axs.flatten()
        
        # Vẽ biểu đồ cho từng chỉ số
        for i, metric in enumerate(metrics_to_compare):
            if i >= len(axs):
                break
                
            ax = axs[i]
            
            # Lấy giá trị cho từng phiên bản
            values = []
            labels = []
            
            for idx, version in zip(version_indices, valid_versions):
                metrics_data = self.version_history["metrics"][idx]
                if metric in metrics_data:
                    values.append(metrics_data[metric])
                    labels.append(version)
            
            if not values:
                ax.text(0.5, 0.5, f"Không có dữ liệu cho chỉ số {metric}", 
                     horizontalalignment='center', verticalalignment='center')
                continue
            
            # Xác định màu cho từng cột (phiên bản hiện tại = xanh)
            colors = ['green' if v == self.current_model_version else 'blue' for v in labels]
            
            # Vẽ biểu đồ cột
            bars = ax.bar(range(len(values)), values, color=colors)
            
            # Thêm nhãn giá trị
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Đặt nhãn
            ax.set_title(metric)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            # Thêm lưới
            ax.grid(True, alpha=0.3)
        
        # Ẩn các trục thừa
        for i in range(len(metrics_to_compare), len(axs)):
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle("So sánh các phiên bản mô hình", fontsize=16)
        
        return fig