#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module đánh giá rủi ro cho hệ thống backtesting.

Module này cung cấp các lớp và hàm để đánh giá rủi ro trong quá trình
backtesting, bao gồm các chỉ số về drawdown, volatility, value-at-risk,
và các chỉ số rủi ro khác.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO
import base64
import json

from config.constants import TRADING_DAYS_PER_YEAR
from config.logging_config import get_logger
from backtesting.performance_metrics import PerformanceMetrics
from risk_management.risk_calculator import RiskCalculator


class RiskEvaluator:
    """
    Đánh giá rủi ro cho các chiến lược giao dịch.
    
    Lớp này cung cấp các phương thức để đánh giá rủi ro của các chiến lược
    giao dịch, bao gồm phân tích drawdown, volatility, và các chỉ số rủi ro khác.
    """
    
    def __init__(
            self, 
            equity_curve: pd.Series, 
            trades: pd.DataFrame = None, 
            benchmark: pd.Series = None,
            risk_free_rate: float = 0.02,
            initial_capital: float = 10000,
            max_drawdown_threshold: float = 0.20,
            var_confidence_level: float = 0.95,
            risk_periods: list = [1, 5, 20, 60],  # Ngày, tuần, tháng, quý
            performance_metrics: Optional[PerformanceMetrics] = None,
            logger = None
        ):
        """
        Khởi tạo đối tượng RiskEvaluator.
        
        Args:
            equity_curve (pd.Series): Chuỗi giá trị vốn theo thời gian.
            trades (pd.DataFrame, optional): DataFrame chứa thông tin các giao dịch.
            benchmark (pd.Series, optional): Chuỗi giá trị của benchmark theo thời gian.
            risk_free_rate (float, optional): Lãi suất phi rủi ro hàng năm. Mặc định là 0.02 (2%).
            initial_capital (float, optional): Vốn ban đầu. Mặc định là 10,000.
            max_drawdown_threshold (float, optional): Ngưỡng drawdown tối đa chấp nhận được. Mặc định là 0.20 (20%).
            var_confidence_level (float, optional): Mức độ tin cậy cho VaR. Mặc định là 0.95 (95%).
            risk_periods (list, optional): Danh sách các khoảng thời gian để tính toán rủi ro.
            performance_metrics (PerformanceMetrics, optional): Đối tượng PerformanceMetrics đã được tính toán trước đó.
            logger: Logger tùy chỉnh
        """
        self.logger = logger or get_logger("risk_evaluator")
        self.equity_curve = equity_curve
        self.trades = trades
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        self.initial_capital = initial_capital
        self.max_drawdown_threshold = max_drawdown_threshold
        self.var_confidence_level = var_confidence_level
        self.risk_periods = risk_periods
        
        # Tạo đối tượng PerformanceMetrics nếu không được cung cấp
        if performance_metrics is None:
            self.performance_metrics = PerformanceMetrics(
                equity_curve=equity_curve,
                trades=trades,
                benchmark=benchmark,
                risk_free_rate=risk_free_rate,
                initial_capital=initial_capital
            )
        else:
            self.performance_metrics = performance_metrics
        
        # Khởi tạo RiskCalculator để sử dụng các phương thức tính toán rủi ro
        self.risk_calculator = RiskCalculator(
            risk_free_rate=risk_free_rate,
            confidence_level=var_confidence_level
        )
        
        # Tính toán các giá trị cơ bản
        self.returns = self.performance_metrics.returns
        self.log_returns = self.performance_metrics.log_returns
        self.cum_returns = self.performance_metrics.cum_returns
        
        # Khởi tạo các kết quả
        self.results = {}
        
    def evaluate_all_risks(self) -> Dict[str, Any]:
        """
        Đánh giá tất cả các loại rủi ro.
        
        Returns:
            Dict[str, Any]: Dictionary chứa tất cả các đánh giá rủi ro.
        """
        # Tính toán các chỉ số rủi ro cơ bản
        drawdown_analysis = self.analyze_drawdown()
        volatility_analysis = self.analyze_volatility()
        var_analysis = self.analyze_value_at_risk()
        tail_risk_analysis = self.analyze_tail_risk()
        distribution_analysis = self.analyze_returns_distribution()
        
        # Đánh giá rủi ro tổng hợp
        risk_assessment = self.get_overall_risk_assessment()
        
        # Tổng hợp kết quả
        self.results = {
            "drawdown_analysis": drawdown_analysis,
            "volatility_analysis": volatility_analysis,
            "var_analysis": var_analysis,
            "tail_risk_analysis": tail_risk_analysis,
            "distribution_analysis": distribution_analysis,
            "risk_assessment": risk_assessment
        }
        
        return self.results
    
    def analyze_drawdown(self) -> Dict[str, Any]:
        """
        Phân tích chi tiết drawdown.
        
        Returns:
            Dict[str, Any]: Kết quả phân tích drawdown.
        """
        # Lấy các giá trị drawdown từ PerformanceMetrics
        max_drawdown = self.performance_metrics.max_drawdown()
        max_drawdown_duration = self.performance_metrics.max_drawdown_duration()
        drawdown_periods = self.performance_metrics.drawdown_periods()
        recovery_periods = self.performance_metrics.recovery_periods()
        
        # Tính toán các chỉ số bổ sung
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        
        # Tính tần suất drawdown ở các mức khác nhau
        drawdown_frequency = {
            "5%+": sum(1 for d in drawdown if d <= -0.05) / len(drawdown),
            "10%+": sum(1 for d in drawdown if d <= -0.10) / len(drawdown),
            "15%+": sum(1 for d in drawdown if d <= -0.15) / len(drawdown),
            "20%+": sum(1 for d in drawdown if d <= -0.20) / len(drawdown),
        }
        
        # Tính thống kê về thời gian hồi phục
        if recovery_periods:
            avg_recovery_time = np.mean([period['duration_days'] for period in recovery_periods])
            max_recovery_time = max([period['duration_days'] for period in recovery_periods])
            min_recovery_time = min([period['duration_days'] for period in recovery_periods])
        else:
            avg_recovery_time = 0
            max_recovery_time = 0
            min_recovery_time = 0
        
        # Tính toán tỷ lệ thời gian trong drawdown
        days_in_drawdown = sum(1 for d in drawdown if d < 0)
        time_in_drawdown = days_in_drawdown / len(drawdown) if len(drawdown) > 0 else 0
        
        # Tính toán tỷ lệ thời gian trong drawdown nghiêm trọng (>10%)
        days_in_severe_drawdown = sum(1 for d in drawdown if d <= -0.10)
        time_in_severe_drawdown = days_in_severe_drawdown / len(drawdown) if len(drawdown) > 0 else 0
        
        # Đánh giá mức độ rủi ro drawdown
        if max_drawdown > self.max_drawdown_threshold * 1.25:
            drawdown_risk = "Rất cao"
        elif max_drawdown > self.max_drawdown_threshold:
            drawdown_risk = "Cao"
        elif max_drawdown > self.max_drawdown_threshold * 0.75:
            drawdown_risk = "Trung bình cao"
        elif max_drawdown > self.max_drawdown_threshold * 0.5:
            drawdown_risk = "Trung bình"
        else:
            drawdown_risk = "Thấp"
        
        # Vẽ biểu đồ drawdown underwater
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
        ax.set_title('Drawdown Underwater Chart')
        ax.set_ylabel('Drawdown')
        ax.set_xlabel('Thời gian')
        ax.grid(True, alpha=0.3)
        
        # Thêm các ngưỡng drawdown
        ax.axhline(y=-0.05, color='yellow', linestyle='--', alpha=0.5, label='5%')
        ax.axhline(y=-0.10, color='orange', linestyle='--', alpha=0.5, label='10%')
        ax.axhline(y=-0.20, color='red', linestyle='--', alpha=0.5, label='20%')
        
        plt.legend()
        plt.tight_layout()
        
        # Chuyển biểu đồ thành base64 string để trả về
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        drawdown_chart = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Tổng hợp kết quả
        result = {
            "max_drawdown": max_drawdown,
            "max_drawdown_percent": max_drawdown * 100,
            "max_drawdown_duration": max_drawdown_duration,
            "drawdown_periods_count": len(drawdown_periods),
            "recovery_periods_count": len(recovery_periods),
            "avg_recovery_time": avg_recovery_time,
            "max_recovery_time": max_recovery_time,
            "min_recovery_time": min_recovery_time,
            "time_in_drawdown": time_in_drawdown,
            "time_in_severe_drawdown": time_in_severe_drawdown,
            "drawdown_frequency": drawdown_frequency,
            "drawdown_risk": drawdown_risk,
            "top_drawdown_periods": drawdown_periods[:5] if len(drawdown_periods) > 5 else drawdown_periods,
            "drawdown_chart": drawdown_chart
        }
        
        self.logger.info(f"Đã phân tích drawdown: max={max_drawdown:.2%}, risk={drawdown_risk}")
        return result
    
    def analyze_volatility(self) -> Dict[str, Any]:
        """
        Phân tích biến động của lợi nhuận.
        
        Returns:
            Dict[str, Any]: Kết quả phân tích volatility.
        """
        # Tính các chỉ số volatility
        daily_volatility = self.returns.std()
        annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Tính downside volatility
        downside_returns = self.returns[self.returns < 0]
        daily_downside_volatility = downside_returns.std() if len(downside_returns) > 0 else 0
        annualized_downside_volatility = daily_downside_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Tính volatility theo các khoảng thời gian khác nhau
        period_volatility = {}
        for period in self.risk_periods:
            if len(self.returns) > period:
                rolling_vol = self.returns.rolling(window=period).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                period_volatility[f"{period}d"] = {
                    "mean": rolling_vol.mean(),
                    "max": rolling_vol.max(),
                    "min": rolling_vol.min(),
                    "current": rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0
                }
        
        # Tính volatility clustering (tự tương quan của bình phương lợi nhuận)
        squared_returns = self.returns ** 2
        if len(squared_returns) > 1:
            volatility_clustering = squared_returns.autocorr(lag=1)
        else:
            volatility_clustering = 0
        
        # Tính volatility của volatility
        if len(self.returns) > 30:
            rolling_vol_30d = self.returns.rolling(window=30).std()
            vol_of_vol = rolling_vol_30d.std() / rolling_vol_30d.mean() if rolling_vol_30d.mean() > 0 else 0
        else:
            vol_of_vol = 0
        
        # So sánh với benchmark nếu có
        benchmark_comparison = {}
        if self.benchmark is not None:
            benchmark_returns = self.benchmark.pct_change().dropna()
            benchmark_volatility = benchmark_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            relative_volatility = annualized_volatility / benchmark_volatility if benchmark_volatility > 0 else float('inf')
            
            benchmark_comparison = {
                "benchmark_volatility": benchmark_volatility,
                "relative_volatility": relative_volatility,
                "is_less_volatile": annualized_volatility < benchmark_volatility
            }
        
        # Tính volatility ratio (upside/downside)
        upside_returns = self.returns[self.returns > 0]
        upside_volatility = upside_returns.std() if len(upside_returns) > 0 else 0
        
        volatility_ratio = upside_volatility / daily_downside_volatility if daily_downside_volatility > 0 else float('inf')
        
        # Đánh giá mức độ rủi ro biến động
        if annualized_volatility > 0.30:  # >30%
            volatility_risk = "Rất cao"
        elif annualized_volatility > 0.20:  # >20%
            volatility_risk = "Cao"
        elif annualized_volatility > 0.15:  # >15%
            volatility_risk = "Trung bình cao"
        elif annualized_volatility > 0.10:  # >10%
            volatility_risk = "Trung bình"
        elif annualized_volatility > 0.05:  # >5%
            volatility_risk = "Thấp"
        else:
            volatility_risk = "Rất thấp"
        
        # Vẽ biểu đồ rolling volatility
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for period in [20, 60]:  # 1 tháng, 3 tháng
            if len(self.returns) > period:
                rolling_vol = self.returns.rolling(window=period).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                ax.plot(rolling_vol.index, rolling_vol * 100, label=f'{period} ngày')
        
        ax.set_title('Rolling Annualized Volatility')
        ax.set_ylabel('Volatility (%)')
        ax.set_xlabel('Thời gian')
        ax.grid(True, alpha=0.3)
        
        # Thêm các ngưỡng volatility
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10%')
        ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='20%')
        ax.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='30%')
        
        plt.legend()
        plt.tight_layout()
        
        # Chuyển biểu đồ thành base64 string để trả về
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        volatility_chart = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Tổng hợp kết quả
        result = {
            "daily_volatility": daily_volatility,
            "annualized_volatility": annualized_volatility,
            "daily_downside_volatility": daily_downside_volatility,
            "annualized_downside_volatility": annualized_downside_volatility,
            "volatility_ratio": volatility_ratio,
            "volatility_clustering": volatility_clustering,
            "vol_of_vol": vol_of_vol,
            "period_volatility": period_volatility,
            "benchmark_comparison": benchmark_comparison,
            "volatility_risk": volatility_risk,
            "volatility_chart": volatility_chart
        }
        
        self.logger.info(f"Đã phân tích volatility: annual={annualized_volatility:.2%}, risk={volatility_risk}")
        return result
    
    def analyze_value_at_risk(self) -> Dict[str, Any]:
        """
        Phân tích Value at Risk (VaR) và Expected Shortfall (ES).
        
        Returns:
            Dict[str, Any]: Kết quả phân tích VaR.
        """
        # Tính VaR theo các phương pháp khác nhau
        # 1. Historical VaR
        historical_var = self.performance_metrics.value_at_risk(self.var_confidence_level)
        
        # 2. Parametric VaR (giả sử phân phối chuẩn)
        z_score = stats.norm.ppf(1 - self.var_confidence_level)
        parametric_var = -self.returns.mean() + z_score * self.returns.std()
        
        # 3. Cornish-Fisher VaR (điều chỉnh theo skewness và kurtosis)
        skewness = stats.skew(self.returns.dropna())
        kurtosis = stats.kurtosis(self.returns.dropna())
        
        z_cf = z_score + (z_score**2 - 1) * skewness / 6 + (z_score**3 - 3*z_score) * kurtosis / 24 - (2*z_score**3 - 5*z_score) * skewness**2 / 36
        
        cornish_fisher_var = -self.returns.mean() + z_cf * self.returns.std()
        
        # Expected Shortfall (Conditional VaR)
        historical_es = self.performance_metrics.conditional_value_at_risk(self.var_confidence_level)
        
        # Tính VaR cho các khoảng thời gian khác nhau
        var_by_period = {}
        es_by_period = {}
        
        for period in self.risk_periods:
            if len(self.returns) > period:
                # Tính VaR theo lịch sử
                period_var = self.returns.iloc[-period:].quantile(1 - self.var_confidence_level)
                var_by_period[f"{period}d"] = abs(period_var)
                
                # Tính ES theo lịch sử
                period_returns = self.returns.iloc[-period:]
                period_var_cutoff = period_returns.quantile(1 - self.var_confidence_level)
                period_es = period_returns[period_returns <= period_var_cutoff].mean()
                es_by_period[f"{period}d"] = abs(period_es)
        
        # Chuyển đổi sang giá trị vốn
        latest_equity = self.equity_curve.iloc[-1]
        
        var_amount = {
            "historical": historical_var * latest_equity,
            "parametric": parametric_var * latest_equity,
            "cornish_fisher": cornish_fisher_var * latest_equity
        }
        
        es_amount = historical_es * latest_equity
        
        # Tính VaR contribution - chỉ có thể khi có dữ liệu giao dịch
        var_contribution = {}
        if self.trades is not None and len(self.trades) > 0:
            # Nếu có thông tin về symbol, tính VaR contribution cho mỗi symbol
            if 'symbol' in self.trades.columns:
                symbols = self.trades['symbol'].unique()
                for symbol in symbols:
                    symbol_trades = self.trades[self.trades['symbol'] == symbol]
                    symbol_returns = symbol_trades['profit'] / symbol_trades['size'] if 'size' in symbol_trades.columns else symbol_trades['profit']
                    
                    if len(symbol_returns) > 0:
                        symbol_var = abs(symbol_returns.quantile(1 - self.var_confidence_level)) if len(symbol_returns) > 1 else 0
                        var_contribution[symbol] = symbol_var
        
        # Đánh giá mức độ rủi ro VaR
        var_pct_of_capital = historical_var * 100  # Convert to percentage
        
        if var_pct_of_capital > 5:  # >5% mất vốn hàng ngày
            var_risk = "Rất cao"
        elif var_pct_of_capital > 3:  # >3%
            var_risk = "Cao"
        elif var_pct_of_capital > 2:  # >2%
            var_risk = "Trung bình cao"
        elif var_pct_of_capital > 1:  # >1%
            var_risk = "Trung bình"
        else:
            var_risk = "Thấp"
        
        # Vẽ biểu đồ so sánh VaR theo các phương pháp
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(['Historical', 'Parametric', 'Cornish-Fisher'], 
                    [historical_var, parametric_var, cornish_fisher_var],
                    alpha=0.7)
        
        # Thêm giá trị lên thanh
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2%}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')
        
        # Thêm ES vào biểu đồ
        ax.axhline(y=historical_es, color='red', linestyle='--', 
                 label=f'Expected Shortfall: {historical_es:.2%}')
        
        ax.set_title(f'Value at Risk ({self.var_confidence_level*100}% confidence)')
        ax.set_ylabel('Loss as % of Portfolio')
        ax.set_ylim(0, max(historical_var, parametric_var, cornish_fisher_var, historical_es) * 1.2)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Chuyển biểu đồ thành base64 string để trả về
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        var_chart = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Tổng hợp kết quả
        result = {
            "var": {
                "historical": historical_var,
                "parametric": parametric_var,
                "cornish_fisher": cornish_fisher_var
            },
            "var_amount": var_amount,
            "var_by_period": var_by_period,
            "es": historical_es,
            "es_amount": es_amount,
            "es_by_period": es_by_period,
            "var_contribution": var_contribution,
            "confidence_level": self.var_confidence_level,
            "var_risk": var_risk,
            "var_chart": var_chart
        }
        
        self.logger.info(f"Đã phân tích VaR: historical={historical_var:.2%}, ES={historical_es:.2%}, risk={var_risk}")
        return result
    
    def analyze_tail_risk(self) -> Dict[str, Any]:
        """
        Phân tích rủi ro đuôi (tail risk).
        
        Returns:
            Dict[str, Any]: Kết quả phân tích tail risk.
        """
        # Tính các chỉ số liên quan đến rủi ro đuôi
        # 1. Tail ratio
        tail_ratio = self.performance_metrics.tail_ratio(quantile=0.05)
        
        # 2. Maximum drawdown (là một chỉ số rủi ro đuôi)
        max_drawdown = self.performance_metrics.max_drawdown()
        
        # 3. Maximum loss
        max_loss = self.returns.min() if len(self.returns) > 0 else 0
        
        # 4. Worst 5 days
        worst_days = self.returns.nsmallest(5)
        
        # 5. Sortino ratio (sử dụng downside risk)
        sortino_ratio = self.performance_metrics.sortino_ratio()
        
        # 6. Worst drawdown periods
        drawdown_periods = self.performance_metrics.drawdown_periods()
        worst_drawdowns = sorted(drawdown_periods, key=lambda x: x['drawdown_amount'], reverse=True)[:5] if drawdown_periods else []
        
        # 7. Drawdown frequency (% thời gian trong drawdown)
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        
        time_in_drawdown = sum(1 for d in drawdown if d < 0) / len(drawdown) if len(drawdown) > 0 else 0
        
        # 8. Ulcer Index (căn bậc hai của tổng bình phương drawdown)
        squared_drawdowns = drawdown ** 2
        ulcer_index = np.sqrt(squared_drawdowns.mean())
        
        # 9. Pain Index (trung bình drawdown)
        pain_index = abs(drawdown.mean())
        
        # 10. Pain Ratio
        annualized_return = self.performance_metrics.annualized_return()
        pain_ratio = annualized_return / pain_index if pain_index > 0 else float('inf')
        
        # Đánh giá mức độ rủi ro đuôi
        # Kết hợp nhiều chỉ số để đưa ra đánh giá
        tail_risk_score = 0
        
        # Điểm cho max drawdown
        if max_drawdown > 0.3:
            tail_risk_score += 5
        elif max_drawdown > 0.2:
            tail_risk_score += 4
        elif max_drawdown > 0.15:
            tail_risk_score += 3
        elif max_drawdown > 0.1:
            tail_risk_score += 2
        elif max_drawdown > 0.05:
            tail_risk_score += 1
        
        # Điểm cho max loss
        if max_loss < -0.1:
            tail_risk_score += 5
        elif max_loss < -0.05:
            tail_risk_score += 4
        elif max_loss < -0.03:
            tail_risk_score += 3
        elif max_loss < -0.02:
            tail_risk_score += 2
        elif max_loss < -0.01:
            tail_risk_score += 1
        
        # Điểm cho ulcer index
        if ulcer_index > 0.1:
            tail_risk_score += 5
        elif ulcer_index > 0.05:
            tail_risk_score += 4
        elif ulcer_index > 0.03:
            tail_risk_score += 3
        elif ulcer_index > 0.02:
            tail_risk_score += 2
        elif ulcer_index > 0.01:
            tail_risk_score += 1
        
        # Đánh giá mức độ rủi ro dựa trên điểm
        if tail_risk_score >= 10:
            tail_risk = "Rất cao"
        elif tail_risk_score >= 7:
            tail_risk = "Cao"
        elif tail_risk_score >= 5:
            tail_risk = "Trung bình cao"
        elif tail_risk_score >= 3:
            tail_risk = "Trung bình"
        else:
            tail_risk = "Thấp"
        
        # Vẽ biểu đồ phân phối lợi nhuận tập trung vào đuôi
        fig, ax = plt.subplots(figsize=(10, 6))
        
        returns_to_plot = self.returns.dropna()
        sns.histplot(returns_to_plot, bins=50, kde=True, ax=ax)
        
        # Highlight các giá trị đuôi trái (âm)
        left_tail = returns_to_plot[returns_to_plot <= returns_to_plot.quantile(0.05)]
        if len(left_tail) > 0:
            sns.histplot(left_tail, bins=10, color='red', ax=ax, alpha=0.7, 
                       label=f'Left Tail (5%): {left_tail.mean():.2%} avg. loss')
        
        ax.set_title('Phân phối lợi nhuận với nhấn mạnh đuôi trái')
        ax.set_xlabel('Lợi nhuận hàng ngày')
        ax.set_ylabel('Tần suất')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Chuyển biểu đồ thành base64 string để trả về
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        tail_risk_chart = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Tổng hợp kết quả
        result = {
            "tail_ratio": tail_ratio,
            "max_drawdown": max_drawdown,
            "max_loss": max_loss,
            "worst_days": worst_days.to_dict(),
            "sortino_ratio": sortino_ratio,
            "worst_drawdowns": worst_drawdowns,
            "time_in_drawdown": time_in_drawdown,
            "ulcer_index": ulcer_index,
            "pain_index": pain_index,
            "pain_ratio": pain_ratio,
            "tail_risk_score": tail_risk_score,
            "tail_risk": tail_risk,
            "tail_risk_chart": tail_risk_chart
        }
        
        self.logger.info(f"Đã phân tích tail risk: score={tail_risk_score}, risk={tail_risk}")
        return result
    
    def analyze_returns_distribution(self) -> Dict[str, Any]:
        """
        Phân tích phân phối lợi nhuận.
        
        Returns:
            Dict[str, Any]: Kết quả phân tích phân phối lợi nhuận.
        """
        # Tính các thống kê mô tả
        returns_stats = self.returns.describe()
        
        # Tính skewness và kurtosis
        skewness = stats.skew(self.returns.dropna())
        kurtosis = stats.kurtosis(self.returns.dropna())
        
        # Kiểm định tính chuẩn
        jarque_bera_stat, jarque_bera_pval = stats.jarque_bera(self.returns.dropna())
        is_normal = jarque_bera_pval > 0.05
        
        # Tính tỷ lệ lợi nhuận dương/âm
        positive_returns = sum(1 for r in self.returns if r > 0) / len(self.returns) if len(self.returns) > 0 else 0
        negative_returns = sum(1 for r in self.returns if r < 0) / len(self.returns) if len(self.returns) > 0 else 0
        
        # Tính các phân vị
        percentiles = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percentiles[f"p{p}"] = self.returns.quantile(p/100)
        
        # So sánh với phân phối chuẩn
        normal_dist_comparison = {}
        if len(self.returns) > 0:
            returns_mean = self.returns.mean()
            returns_std = self.returns.std()
            
            # Tính lý thuyết vs thực tế cho các phân vị
            normal_dist_comparison = {
                "p1": {
                    "actual": self.returns.quantile(0.01),
                    "normal": returns_mean + stats.norm.ppf(0.01) * returns_std
                },
                "p5": {
                    "actual": self.returns.quantile(0.05),
                    "normal": returns_mean + stats.norm.ppf(0.05) * returns_std
                },
                "p95": {
                    "actual": self.returns.quantile(0.95),
                    "normal": returns_mean + stats.norm.ppf(0.95) * returns_std
                },
                "p99": {
                    "actual": self.returns.quantile(0.99),
                    "normal": returns_mean + stats.norm.ppf(0.99) * returns_std
                }
            }
        
        # Đánh giá mức độ tốt của phân phối
        distribution_score = 0
        
        # Skewness tốt khi dương (chuột sang phải)
        if skewness > 0.5:
            distribution_score += 2
        elif skewness > 0:
            distribution_score += 1
        elif skewness < -0.5:
            distribution_score -= 2
        elif skewness < 0:
            distribution_score -= 1
        
        # Kurtosis cao hơn có nghĩa là đuôi dày hơn (fat tails) - không tốt
        if kurtosis > 5:
            distribution_score -= 3
        elif kurtosis > 3:
            distribution_score -= 2
        elif kurtosis > 1:
            distribution_score -= 1
        
        # Tỷ lệ lợi nhuận dương/âm cao là tốt
        if positive_returns > 0.6:
            distribution_score += 3
        elif positive_returns > 0.55:
            distribution_score += 2
        elif positive_returns > 0.5:
            distribution_score += 1
        elif positive_returns < 0.4:
            distribution_score -= 3
        elif positive_returns < 0.45:
            distribution_score -= 2
        
        # Đánh giá dựa trên điểm
        if distribution_score >= 4:
            distribution_quality = "Rất tốt"
        elif distribution_score >= 2:
            distribution_quality = "Tốt"
        elif distribution_score >= 0:
            distribution_quality = "Trung bình"
        elif distribution_score >= -2:
            distribution_quality = "Kém"
        else:
            distribution_quality = "Rất kém"
        
        # Vẽ biểu đồ QQ-plot để so sánh với phân phối chuẩn
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Histogram với KDE
        sns.histplot(self.returns.dropna(), bins=50, kde=True, ax=ax1)
        ax1.set_title('Phân phối lợi nhuận')
        ax1.set_xlabel('Lợi nhuận hàng ngày')
        ax1.set_ylabel('Tần suất')
        
        # QQ-plot
        stats.probplot(self.returns.dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot vs. Phân phối chuẩn')
        
        plt.tight_layout()
        
        # Chuyển biểu đồ thành base64 string để trả về
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        distribution_chart = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Tổng hợp kết quả
        result = {
            "statistics": returns_stats.to_dict(),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "jarque_bera": {
                "statistic": jarque_bera_stat,
                "p_value": jarque_bera_pval,
                "is_normal": is_normal
            },
            "positive_returns_ratio": positive_returns,
            "negative_returns_ratio": negative_returns,
            "percentiles": percentiles,
            "normal_dist_comparison": normal_dist_comparison,
            "distribution_score": distribution_score,
            "distribution_quality": distribution_quality,
            "distribution_chart": distribution_chart
        }
        
        self.logger.info(f"Đã phân tích phân phối lợi nhuận: skew={skewness:.2f}, kurt={kurtosis:.2f}, quality={distribution_quality}")
        return result
    
    def calculate_stress_test(self, scenario_name: str = None, stress_percent: float = None, 
                            duration_days: int = None, recovery_rate: float = None) -> Dict[str, Any]:
        """
        Thực hiện stress test cho chiến lược.
        
        Args:
            scenario_name (str, optional): Tên kịch bản stress test có sẵn.
            stress_percent (float, optional): Phần trăm stress (giảm) nếu tạo kịch bản tùy chỉnh.
            duration_days (int, optional): Số ngày stress kéo dài nếu tạo kịch bản tùy chỉnh.
            recovery_rate (float, optional): Tốc độ phục hồi hàng ngày sau stress nếu tạo kịch bản tùy chỉnh.
            
        Returns:
            Dict[str, Any]: Kết quả stress test.
        """
        # Các kịch bản có sẵn
        predefined_scenarios = {
            "2008_crisis": {"stress_percent": 0.40, "duration_days": 60, "recovery_rate": 0.005},
            "2020_covid": {"stress_percent": 0.35, "duration_days": 30, "recovery_rate": 0.01},
            "2022_crypto_winter": {"stress_percent": 0.60, "duration_days": 90, "recovery_rate": 0.003},
            "average_correction": {"stress_percent": 0.15, "duration_days": 20, "recovery_rate": 0.007},
            "severe_correction": {"stress_percent": 0.25, "duration_days": 40, "recovery_rate": 0.005}
        }
        
        # Sử dụng kịch bản có sẵn hoặc tùy chỉnh
        if scenario_name and scenario_name in predefined_scenarios:
            scenario = predefined_scenarios[scenario_name]
            stress_percent = scenario["stress_percent"]
            duration_days = scenario["duration_days"]
            recovery_rate = scenario["recovery_rate"]
            scenario_label = scenario_name
        else:
            if stress_percent is None or duration_days is None:
                stress_percent = 0.25  # Mặc định 25%
                duration_days = 30     # Mặc định 30 ngày
            
            if recovery_rate is None:
                recovery_rate = 0.005  # Mặc định 0.5% mỗi ngày
                
            scenario_label = "custom"
        
        # Lấy giá trị cuối cùng của equity
        last_equity = self.equity_curve.iloc[-1]
        
        # Tạo chuỗi thời gian cho stress test
        stress_days = duration_days
        recovery_days = int(stress_percent / recovery_rate)  # Số ngày để phục hồi hoàn toàn
        simulation_days = stress_days + recovery_days
        
        dates = pd.date_range(start=self.equity_curve.index[-1] + pd.Timedelta(days=1), periods=simulation_days)
        
        # Tạo equity curve sau stress
        stressed_equity = []
        
        # Giai đoạn stress
        for i in range(stress_days):
            day_stress = (1 - stress_percent * (i + 1) / stress_days)
            stressed_equity.append(last_equity * day_stress)
        
        # Giai đoạn phục hồi
        recovery_start = stressed_equity[-1]
        for i in range(recovery_days):
            day_recovery = recovery_start * (1 + recovery_rate * (i + 1))
            # Không vượt quá giá trị ban đầu
            stressed_equity.append(min(day_recovery, last_equity))
        
        # Tạo Series
        stress_simulation = pd.Series(stressed_equity, index=dates)
        
        # Tính các chỉ số
        min_equity = min(stressed_equity)
        max_drawdown_amount = last_equity - min_equity
        max_drawdown_percent = max_drawdown_amount / last_equity
        
        full_recovery_days = 0
        for i, value in enumerate(stressed_equity):
            if value >= last_equity:
                full_recovery_days = i + 1
                break
        
        if full_recovery_days == 0 and stressed_equity[-1] < last_equity:
            full_recovery_days = None
        
        # Đánh giá mức độ ảnh hưởng
        capital_after_stress = self.initial_capital + (last_equity - self.initial_capital) * (1 - stress_percent)
        profit_loss_after_stress = capital_after_stress - self.initial_capital
        profit_loss_percent = profit_loss_after_stress / self.initial_capital
        
        if profit_loss_percent < -0.3:
            impact = "Rất nghiêm trọng"
        elif profit_loss_percent < -0.2:
            impact = "Nghiêm trọng"
        elif profit_loss_percent < -0.1:
            impact = "Đáng kể"
        elif profit_loss_percent < 0:
            impact = "Vừa phải"
        else:
            impact = "Nhẹ"
        
        # Vẽ biểu đồ stress test
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Vẽ equity thực tế
        ax.plot(self.equity_curve.index, self.equity_curve, label='Equity thực tế', color='blue')
        
        # Vẽ equity sau stress
        ax.plot(stress_simulation.index, stress_simulation, label='Equity sau stress', color='red', linestyle='--')
        
        # Thêm vạch đánh dấu điểm stress và phục hồi
        ax.axvline(x=dates[0], color='orange', linestyle='--', alpha=0.5, label='Bắt đầu stress')
        ax.axvline(x=dates[stress_days - 1], color='green', linestyle='--', alpha=0.5, label='Kết thúc stress')
        
        if full_recovery_days is not None:
            ax.axvline(x=dates[full_recovery_days - 1], color='purple', linestyle='--', alpha=0.5, label='Phục hồi hoàn toàn')
        
        ax.set_title(f'Stress Test Scenario: {scenario_label.replace("_", " ").title()}')
        ax.set_ylabel('Equity')
        ax.set_xlabel('Thời gian')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Chuyển biểu đồ thành base64 string để trả về
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        stress_test_chart = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Tổng hợp kết quả
        result = {
            "scenario": {
                "name": scenario_label,
                "stress_percent": stress_percent,
                "duration_days": duration_days,
                "recovery_rate": recovery_rate
            },
            "results": {
                "min_equity": min_equity,
                "max_drawdown_amount": max_drawdown_amount,
                "max_drawdown_percent": max_drawdown_percent,
                "full_recovery_days": full_recovery_days,
                "capital_after_stress": capital_after_stress,
                "profit_loss_after_stress": profit_loss_after_stress,
                "profit_loss_percent": profit_loss_percent
            },
            "impact": impact,
            "stress_test_chart": stress_test_chart
        }
        
        self.logger.info(f"Đã thực hiện stress test: scenario={scenario_label}, impact={impact}")
        return result
    
    def calculate_risk_return_metrics(self) -> Dict[str, Any]:
        """
        Tính toán các chỉ số rủi ro-lợi nhuận.
        
        Returns:
            Dict[str, Any]: Các chỉ số rủi ro-lợi nhuận.
        """
        # Sharpe Ratio
        sharpe_ratio = self.performance_metrics.sharpe_ratio()
        
        # Sortino Ratio
        sortino_ratio = self.performance_metrics.sortino_ratio()
        
        # Calmar Ratio
        calmar_ratio = self.performance_metrics.calmar_ratio()
        
        # Omega Ratio
        omega_ratio = self.performance_metrics.omega_ratio()
        
        # Information Ratio (nếu có benchmark)
        information_ratio = self.performance_metrics.information_ratio() if self.benchmark is not None else None
        
        # Tính toán Alpha và Beta (nếu có benchmark)
        alpha = None
        beta = None
        
        if self.benchmark is not None:
            beta = self.performance_metrics.beta()
            alpha = self.performance_metrics.alpha()
        
        # Treynor Ratio (nếu có benchmark)
        treynor_ratio = self.performance_metrics.treynor_ratio() if self.benchmark is not None else None
        
        # Tính các chỉ số bổ sung
        annualized_return = self.performance_metrics.annualized_return()
        annualized_volatility = self.returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        max_drawdown = self.performance_metrics.max_drawdown()
        
        # Tính Gain to Pain Ratio
        positive_returns = self.returns[self.returns > 0].sum()
        negative_returns = abs(self.returns[self.returns < 0].sum())
        gain_to_pain_ratio = positive_returns / negative_returns if negative_returns > 0 else float('inf')
        
        # Tính kỳ vọng toán học
        expected_return = annualized_return
        
        # Vẽ biểu đồ Risk-Return
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Vẽ điểm Risk-Return của chiến lược
        ax.scatter(annualized_volatility * 100, annualized_return * 100, s=100, color='blue', label='Chiến lược')
        
        # Vẽ điểm Risk-Return của benchmark nếu có
        if self.benchmark is not None:
            benchmark_returns = self.benchmark.pct_change().dropna()
            benchmark_annualized_return = (1 + benchmark_returns.mean()) ** TRADING_DAYS_PER_YEAR - 1
            benchmark_annualized_volatility = benchmark_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            
            ax.scatter(benchmark_annualized_volatility * 100, benchmark_annualized_return * 100, 
                     s=100, color='green', label='Benchmark')
        
        # Vẽ đường Risk-Free
        ax.axhline(y=self.risk_free_rate * 100, color='red', linestyle='--', label=f'Risk-Free ({self.risk_free_rate*100}%)')
        
        # Vẽ đường Sharpe Ratio
        x_range = np.linspace(0, max(annualized_volatility, 0.5) * 100, 100)
        y_range = self.risk_free_rate * 100 + sharpe_ratio * x_range
        ax.plot(x_range, y_range, color='gray', linestyle=':', label=f'Sharpe = {sharpe_ratio:.2f}')
        
        ax.set_title('Risk-Return Profile')
        ax.set_xlabel('Rủi ro (Annualized Volatility %)')
        ax.set_ylabel('Lợi nhuận (Annualized Return %)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Chuyển biểu đồ thành base64 string để trả về
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        risk_return_chart = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Đánh giá mức độ hiệu quả của chiến lược
        if sharpe_ratio > 2.0:
            performance_rating = "Xuất sắc"
        elif sharpe_ratio > 1.5:
            performance_rating = "Rất tốt"
        elif sharpe_ratio > 1.0:
            performance_rating = "Tốt"
        elif sharpe_ratio > 0.5:
            performance_rating = "Trung bình"
        elif sharpe_ratio > 0:
            performance_rating = "Yếu"
        else:
            performance_rating = "Rất yếu"
        
        # Tổng hợp kết quả
        result = {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "omega_ratio": omega_ratio,
            "information_ratio": information_ratio,
            "alpha": alpha,
            "beta": beta,
            "treynor_ratio": treynor_ratio,
            "gain_to_pain_ratio": gain_to_pain_ratio,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "max_drawdown": max_drawdown,
            "expected_return": expected_return,
            "performance_rating": performance_rating,
            "risk_return_chart": risk_return_chart
        }
        
        self.logger.info(f"Đã tính toán chỉ số rủi ro-lợi nhuận: Sharpe={sharpe_ratio:.2f}, Rating={performance_rating}")
        return result
    
    def get_overall_risk_assessment(self) -> Dict[str, Any]:
        """
        Cung cấp đánh giá rủi ro tổng thể.
        
        Returns:
            Dict[str, Any]: Đánh giá rủi ro tổng thể.
        """
        # Tính toán các chỉ số rủi ro cơ bản
        max_drawdown = self.performance_metrics.max_drawdown()
        annualized_volatility = self.returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        var_95 = self.performance_metrics.value_at_risk(confidence=0.95)
        sharpe_ratio = self.performance_metrics.sharpe_ratio()
        sortino_ratio = self.performance_metrics.sortino_ratio()
        
        # Tính thêm nếu chưa có trong performance_metrics
        # Calmar Ratio
        calmar_ratio = self.performance_metrics.calmar_ratio()
        
        # Tính Ulcer Index
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        squared_drawdowns = drawdown ** 2
        ulcer_index = np.sqrt(squared_drawdowns.mean())
        
        # Đánh giá từng loại rủi ro
        risk_scores = {}
        
        # 1. Đánh giá drawdown
        if max_drawdown > self.max_drawdown_threshold * 1.25:
            risk_scores["drawdown"] = {"score": 5, "level": "Rất cao"}
        elif max_drawdown > self.max_drawdown_threshold:
            risk_scores["drawdown"] = {"score": 4, "level": "Cao"}
        elif max_drawdown > self.max_drawdown_threshold * 0.75:
            risk_scores["drawdown"] = {"score": 3, "level": "Trung bình cao"}
        elif max_drawdown > self.max_drawdown_threshold * 0.5:
            risk_scores["drawdown"] = {"score": 2, "level": "Trung bình"}
        else:
            risk_scores["drawdown"] = {"score": 1, "level": "Thấp"}
        
        # 2. Đánh giá volatility
        if annualized_volatility > 0.30:
            risk_scores["volatility"] = {"score": 5, "level": "Rất cao"}
        elif annualized_volatility > 0.20:
            risk_scores["volatility"] = {"score": 4, "level": "Cao"}
        elif annualized_volatility > 0.15:
            risk_scores["volatility"] = {"score": 3, "level": "Trung bình cao"}
        elif annualized_volatility > 0.10:
            risk_scores["volatility"] = {"score": 2, "level": "Trung bình"}
        else:
            risk_scores["volatility"] = {"score": 1, "level": "Thấp"}
        
        # 3. Đánh giá VaR
        var_pct_of_capital = var_95 * 100
        if var_pct_of_capital > 5:
            risk_scores["var"] = {"score": 5, "level": "Rất cao"}
        elif var_pct_of_capital > 3:
            risk_scores["var"] = {"score": 4, "level": "Cao"}
        elif var_pct_of_capital > 2:
            risk_scores["var"] = {"score": 3, "level": "Trung bình cao"}
        elif var_pct_of_capital > 1:
            risk_scores["var"] = {"score": 2, "level": "Trung bình"}
        else:
            risk_scores["var"] = {"score": 1, "level": "Thấp"}
        
        # 4. Đánh giá risk-adjusted return
        if sharpe_ratio < 0:
            risk_scores["risk_return"] = {"score": 5, "level": "Rất cao"}
        elif sharpe_ratio < 0.5:
            risk_scores["risk_return"] = {"score": 4, "level": "Cao"}
        elif sharpe_ratio < 1.0:
            risk_scores["risk_return"] = {"score": 3, "level": "Trung bình cao"}
        elif sharpe_ratio < 1.5:
            risk_scores["risk_return"] = {"score": 2, "level": "Trung bình"}
        else:
            risk_scores["risk_return"] = {"score": 1, "level": "Thấp"}
        
        # 5. Đánh giá tail risk
        skewness = stats.skew(self.returns.dropna())
        kurtosis = stats.kurtosis(self.returns.dropna())
        
        tail_risk_score = 0
        
        # Skewness âm là không tốt
        if skewness < -0.5:
            tail_risk_score += 2
        elif skewness < 0:
            tail_risk_score += 1
        
        # Kurtosis cao là không tốt
        if kurtosis > 5:
            tail_risk_score += 3
        elif kurtosis > 3:
            tail_risk_score += 2
        elif kurtosis > 1:
            tail_risk_score += 1
        
        if tail_risk_score >= 4:
            risk_scores["tail_risk"] = {"score": 5, "level": "Rất cao"}
        elif tail_risk_score == 3:
            risk_scores["tail_risk"] = {"score": 4, "level": "Cao"}
        elif tail_risk_score == 2:
            risk_scores["tail_risk"] = {"score": 3, "level": "Trung bình cao"}
        elif tail_risk_score == 1:
            risk_scores["tail_risk"] = {"score": 2, "level": "Trung bình"}
        else:
            risk_scores["tail_risk"] = {"score": 1, "level": "Thấp"}
        
        # Tính điểm rủi ro tổng thể
        risk_weights = {
            "drawdown": 0.35,
            "volatility": 0.25,
            "var": 0.15,
            "risk_return": 0.15,
            "tail_risk": 0.10
        }
        
        overall_risk_score = sum(risk_scores[key]["score"] * risk_weights[key] for key in risk_weights)
        
        # Xác định mức độ rủi ro tổng thể
        if overall_risk_score >= 4.5:
            overall_risk_level = "Rất cao"
        elif overall_risk_score >= 3.5:
            overall_risk_level = "Cao"
        elif overall_risk_score >= 2.5:
            overall_risk_level = "Trung bình cao"
        elif overall_risk_score >= 1.5:
            overall_risk_level = "Trung bình"
        else:
            overall_risk_level = "Thấp"
        
        # Tính toán tất cả các chỉ số rủi ro-lợi nhuận
        risk_return_metrics = self.calculate_risk_return_metrics()
        
        # Vẽ biểu đồ radar cho chỉ số rủi ro
        categories = ['Drawdown', 'Volatility', 'VaR', 'Risk-Return', 'Tail Risk']
        
        # Chuẩn bị dữ liệu cho biểu đồ radar
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Khép kín đồ thị
        
        # Chuẩn hóa điểm về thang 0-1 (0 là tốt nhất, 1 là tệ nhất)
        normalized_scores = [
            (risk_scores["drawdown"]["score"] - 1) / 4, 
            (risk_scores["volatility"]["score"] - 1) / 4,
            (risk_scores["var"]["score"] - 1) / 4,
            (risk_scores["risk_return"]["score"] - 1) / 4,
            (risk_scores["tail_risk"]["score"] - 1) / 4
        ]
        normalized_scores += normalized_scores[:1]  # Khép kín đồ thị
        
        # Vẽ biểu đồ radar
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Thêm các đường tròn chỉ mức độ
        for i in range(5):
            level = (i + 1) / 5
            ax.plot(angles, [level] * (N + 1), '--', color='gray', alpha=0.3)
        
        # Vẽ biểu đồ radar chính
        ax.plot(angles, normalized_scores, linewidth=2, linestyle='solid', label='Mức độ rủi ro')
        ax.fill(angles, normalized_scores, alpha=0.25)
        
        # Thêm các trục và nhãn
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['Thấp', 'Trung bình', 'Cao', 'Rất cao'])
        ax.grid(True)
        
        plt.title('Đánh giá rủi ro tổng thể', size=15, y=1.1)
        plt.tight_layout()
        
        # Chuyển biểu đồ thành base64 string để trả về
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        risk_radar_chart = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Tạo các khuyến nghị quản lý rủi ro
        risk_recommendations = []
        
        # 1. Khuyến nghị về drawdown
        if risk_scores["drawdown"]["score"] >= 4:
            risk_recommendations.append("Cân nhắc giảm kích thước vị thế xuống 50% hoặc ít hơn để giảm thiểu rủi ro drawdown")
            risk_recommendations.append("Thiết lập hệ thống cảnh báo sớm khi drawdown vượt quá 10%")
        elif risk_scores["drawdown"]["score"] >= 3:
            risk_recommendations.append("Giới hạn kích thước vị thế ở mức 70-80% so với bình thường")
            risk_recommendations.append("Tối ưu hóa chiến lược stop loss để giảm thiểu drawdown")
        
        # 2. Khuyến nghị về volatility
        if risk_scores["volatility"]["score"] >= 4:
            risk_recommendations.append("Giảm đòn bẩy và phân bổ vốn vào các thị trường ít biến động hơn")
            risk_recommendations.append("Xem xét chiến lược hedging để giảm biến động danh mục đầu tư")
        elif risk_scores["volatility"]["score"] >= 3:
            risk_recommendations.append("Tăng cường đa dạng hóa các cặp giao dịch để giảm biến động")
        
        # 3. Khuyến nghị về VaR
        if risk_scores["var"]["score"] >= 4:
            risk_recommendations.append("Giới hạn vốn rủi ro hàng ngày không vượt quá 1-2% tổng vốn")
            risk_recommendations.append("Thực hiện stress test thường xuyên với các kịch bản thị trường khác nhau")
        elif risk_scores["var"]["score"] >= 3:
            risk_recommendations.append("Thiết lập hạn mức rủi ro hàng ngày và tuân thủ nghiêm ngặt")
        
        # 4. Khuyến nghị về Risk-Return
        if risk_scores["risk_return"]["score"] >= 4:
            risk_recommendations.append("Xem xét lại toàn bộ chiến lược giao dịch - lợi nhuận không tương xứng với rủi ro")
            risk_recommendations.append("Tạm dừng giao dịch và quay lại backtest cho đến khi cải thiện tỷ lệ Sharpe")
        elif risk_scores["risk_return"]["score"] >= 3:
            risk_recommendations.append("Tối ưu hóa tham số chiến lược để cải thiện tỷ lệ Sharpe")
        
        # 5. Khuyến nghị về Tail Risk
        if risk_scores["tail_risk"]["score"] >= 4:
            risk_recommendations.append("Thực hiện hedging cho các sự kiện biến động lớn (black swan events)")
            risk_recommendations.append("Giữ một phần vốn (20-30%) ở dạng tiền mặt để đối phó với sự kiện đuôi")
        elif risk_scores["tail_risk"]["score"] >= 3:
            risk_recommendations.append("Tránh tích tụ vị thế trong các thị trường có lịch sử đuôi dày")
        
        # Thêm khuyến nghị chung dựa trên mức độ rủi ro tổng thể
        if overall_risk_score >= 4.0:
            general_recommendation = "Chiến lược có rủi ro RẤT CAO. Cần tái cấu trúc hoàn toàn hoặc giảm mạnh kích thước vị thế."
        elif overall_risk_score >= 3.0:
            general_recommendation = "Chiến lược có rủi ro CAO. Cần thực hiện nhiều biện pháp quản lý rủi ro và giảm kích thước vị thế."
        elif overall_risk_score >= 2.0:
            general_recommendation = "Chiến lược có rủi ro TRUNG BÌNH. Cần giám sát chặt chẽ và cải thiện một số khía cạnh quản lý rủi ro."
        else:
            general_recommendation = "Chiến lược có rủi ro THẤP. Tiếp tục duy trì các biện pháp quản lý rủi ro hiện tại."
        
        # Tổng hợp kết quả
        result = {
            "risk_scores": risk_scores,
            "risk_weights": risk_weights,
            "overall_risk_score": overall_risk_score,
            "overall_risk_level": overall_risk_level,
            "risk_metrics": {
                "max_drawdown": max_drawdown,
                "annualized_volatility": annualized_volatility,
                "var_95": var_95,
                "cvar_95": self.performance_metrics.conditional_value_at_risk(confidence=0.95),
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "ulcer_index": ulcer_index
            },
            "risk_return_metrics": risk_return_metrics,
            "general_recommendation": general_recommendation,
            "risk_recommendations": risk_recommendations,
            "risk_radar_chart": risk_radar_chart
        }
        
        self.logger.info(f"Đã đánh giá rủi ro tổng thể: score={overall_risk_score:.2f}, level={overall_risk_level}")
        return result
    
    def generate_risk_report(self, include_charts: bool = True) -> str:
        """
        Tạo báo cáo rủi ro đầy đủ.
        
        Args:
            include_charts (bool, optional): Bao gồm các biểu đồ trong báo cáo. Mặc định là True.
            
        Returns:
            str: Báo cáo rủi ro dạng Markdown.
        """
        # Tính toán tất cả các chỉ số rủi ro nếu chưa có
        if not self.results:
            self.evaluate_all_risks()
        
        # Tạo báo cáo Markdown
        report = []
        
        # Tiêu đề báo cáo
        report.append("# Báo cáo đánh giá rủi ro chiến lược giao dịch\n")
        report.append(f"**Ngày tạo:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        
        # Thông tin tổng quan
        report.append("## 1. Tổng quan đánh giá rủi ro\n")
        
        risk_assessment = self.results.get("risk_assessment", self.get_overall_risk_assessment())
        overall_risk_level = risk_assessment["overall_risk_level"]
        overall_risk_score = risk_assessment["overall_risk_score"]
        
        report.append(f"**Mức độ rủi ro tổng thể:** {overall_risk_level} ({overall_risk_score:.2f}/5.0)\n")
        report.append(f"**Đánh giá chung:** {risk_assessment['general_recommendation']}\n")
        
        # Chỉ số rủi ro chính
        report.append("### Các chỉ số rủi ro chính:\n")
        
        metrics = risk_assessment["risk_metrics"]
        report.append(f"- **Maximum Drawdown:** {metrics['max_drawdown']:.2%}")
        report.append(f"- **Annualized Volatility:** {metrics['annualized_volatility']:.2%}")
        report.append(f"- **Value at Risk (95%):** {metrics['var_95']:.2%}")
        report.append(f"- **Conditional VaR (95%):** {metrics['cvar_95']:.2%}")
        report.append(f"- **Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}")
        report.append(f"- **Sortino Ratio:** {metrics['sortino_ratio']:.2f}")
        report.append(f"- **Calmar Ratio:** {metrics['calmar_ratio']:.2f}")
        report.append(f"- **Ulcer Index:** {metrics['ulcer_index']:.4f}\n")
        
        # Thêm biểu đồ radar nếu yêu cầu
        if include_charts and "risk_radar_chart" in risk_assessment:
            report.append("### Biểu đồ đánh giá rủi ro tổng thể:\n")
            report.append(f"![Risk Radar Chart](data:image/png;base64,{risk_assessment['risk_radar_chart']})\n")
        
        # Các khuyến nghị
        report.append("### Khuyến nghị quản lý rủi ro:\n")
        for recommendation in risk_assessment.get("risk_recommendations", []):
            report.append(f"- {recommendation}")
        report.append("\n")
        
        # Chi tiết drawdown
        report.append("## 2. Phân tích Drawdown\n")
        
        drawdown = self.results.get("drawdown_analysis", {})
        report.append(f"**Drawdown tối đa:** {drawdown.get('max_drawdown_percent', 0):.2f}%")
        report.append(f"**Thời gian drawdown tối đa:** {drawdown.get('max_drawdown_duration', 0)} ngày")
        report.append(f"**Tỷ lệ thời gian trong drawdown:** {drawdown.get('time_in_drawdown', 0):.2%}")
        report.append(f"**Mức độ rủi ro drawdown:** {drawdown.get('drawdown_risk', 'N/A')}\n")
        
        # Thêm biểu đồ drawdown nếu yêu cầu
        if include_charts and "drawdown_chart" in drawdown:
            report.append("### Biểu đồ drawdown:\n")
            report.append(f"![Drawdown Chart](data:image/png;base64,{drawdown['drawdown_chart']})\n")
        
        # Chi tiết phân tích biến động
        report.append("## 3. Phân tích Volatility\n")
        
        volatility = self.results.get("volatility_analysis", {})
        report.append(f"**Biến động hàng năm:** {volatility.get('annualized_volatility', 0):.2%}")
        report.append(f"**Biến động phía dưới hàng năm:** {volatility.get('annualized_downside_volatility', 0):.2%}")
        report.append(f"**Tỷ lệ biến động:** {volatility.get('volatility_ratio', 0):.2f}")
        report.append(f"**Mức độ rủi ro biến động:** {volatility.get('volatility_risk', 'N/A')}\n")
        
        # Thêm biểu đồ volatility nếu yêu cầu
        if include_charts and "volatility_chart" in volatility:
            report.append("### Biểu đồ biến động:\n")
            report.append(f"![Volatility Chart](data:image/png;base64,{volatility['volatility_chart']})\n")
        
        # Chi tiết Value at Risk
        report.append("## 4. Phân tích Value at Risk (VaR)\n")
        
        var_analysis = self.results.get("var_analysis", {})
        var_dict = var_analysis.get("var", {})
        
        report.append(f"**Confidence Level:** {var_analysis.get('confidence_level', 0.95) * 100}%")
        report.append(f"**Historical VaR:** {var_dict.get('historical', 0):.2%}")
        report.append(f"**Parametric VaR:** {var_dict.get('parametric', 0):.2%}")
        report.append(f"**Cornish-Fisher VaR:** {var_dict.get('cornish_fisher', 0):.2%}")
        report.append(f"**Expected Shortfall (ES):** {var_analysis.get('es', 0):.2%}")
        report.append(f"**Mức độ rủi ro VaR:** {var_analysis.get('var_risk', 'N/A')}\n")
        
        # Thêm biểu đồ VaR nếu yêu cầu
        if include_charts and "var_chart" in var_analysis:
            report.append("### Biểu đồ Value at Risk:\n")
            report.append(f"![VaR Chart](data:image/png;base64,{var_analysis['var_chart']})\n")
        
        # Chi tiết Tail Risk
        report.append("## 5. Phân tích Tail Risk\n")
        
        tail_risk = self.results.get("tail_risk_analysis", {})
        report.append(f"**Tail Ratio:** {tail_risk.get('tail_ratio', 0):.2f}")
        report.append(f"**Maximum Loss:** {tail_risk.get('max_loss', 0):.2%}")
        report.append(f"**Ulcer Index:** {tail_risk.get('ulcer_index', 0):.4f}")
        report.append(f"**Pain Index:** {tail_risk.get('pain_index', 0):.4f}")
        report.append(f"**Pain Ratio:** {tail_risk.get('pain_ratio', 0):.2f}")
        report.append(f"**Mức độ Tail Risk:** {tail_risk.get('tail_risk', 'N/A')}\n")
        
        # Thêm biểu đồ tail risk nếu yêu cầu
        if include_charts and "tail_risk_chart" in tail_risk:
            report.append("### Biểu đồ Tail Risk:\n")
            report.append(f"![Tail Risk Chart](data:image/png;base64,{tail_risk['tail_risk_chart']})\n")
        
        # Chi tiết phân phối lợi nhuận
        report.append("## 6. Phân tích phân phối lợi nhuận\n")
        
        distribution = self.results.get("distribution_analysis", {})
        report.append(f"**Skewness:** {distribution.get('skewness', 0):.2f}")
        report.append(f"**Kurtosis:** {distribution.get('kurtosis', 0):.2f}")
        report.append(f"**Tỷ lệ lợi nhuận dương:** {distribution.get('positive_returns_ratio', 0):.2%}")
        report.append(f"**Chất lượng phân phối:** {distribution.get('distribution_quality', 'N/A')}\n")
        
        # Thêm thông tin kiểm định Jarque-Bera
        jarque_bera = distribution.get("jarque_bera", {})
        if jarque_bera:
            report.append(f"**Jarque-Bera Test:** statistic={jarque_bera.get('statistic', 0):.2f}, p-value={jarque_bera.get('p_value', 0):.4f}")
            report.append(f"**Phân phối chuẩn:** {'Có' if jarque_bera.get('is_normal', False) else 'Không'}\n")
        
        # Thêm biểu đồ phân phối nếu yêu cầu
        if include_charts and "distribution_chart" in distribution:
            report.append("### Biểu đồ phân phối lợi nhuận:\n")
            report.append(f"![Distribution Chart](data:image/png;base64,{distribution['distribution_chart']})\n")
        
        # Tổng hợp các chỉ số rủi ro-lợi nhuận
        report.append("## 7. Tổng hợp chỉ số rủi ro-lợi nhuận\n")
        
        risk_return = risk_assessment.get("risk_return_metrics", {})
        report.append(f"**Lợi nhuận hàng năm:** {risk_return.get('annualized_return', 0):.2%}")
        report.append(f"**Biến động hàng năm:** {risk_return.get('annualized_volatility', 0):.2%}")
        report.append(f"**Sharpe Ratio:** {risk_return.get('sharpe_ratio', 0):.2f}")
        report.append(f"**Sortino Ratio:** {risk_return.get('sortino_ratio', 0):.2f}")
        report.append(f"**Calmar Ratio:** {risk_return.get('calmar_ratio', 0):.2f}")
        report.append(f"**Omega Ratio:** {risk_return.get('omega_ratio', 0):.2f}")
        report.append(f"**Gain to Pain Ratio:** {risk_return.get('gain_to_pain_ratio', 0):.2f}")
        
        # Thêm thông tin Alpha/Beta nếu có
        if risk_return.get("alpha") is not None:
            report.append(f"**Alpha:** {risk_return.get('alpha', 0):.4f}")
            report.append(f"**Beta:** {risk_return.get('beta', 0):.2f}")
            report.append(f"**Information Ratio:** {risk_return.get('information_ratio', 0):.2f}")
            report.append(f"**Treynor Ratio:** {risk_return.get('treynor_ratio', 0):.2f}")
        
        report.append(f"**Đánh giá hiệu suất:** {risk_return.get('performance_rating', 'N/A')}\n")
        
        # Thêm biểu đồ risk-return nếu yêu cầu
        if include_charts and "risk_return_chart" in risk_return:
            report.append("### Biểu đồ Risk-Return:\n")
            report.append(f"![Risk-Return Chart](data:image/png;base64,{risk_return['risk_return_chart']})\n")
        
        # Kết luận
        report.append("## 8. Kết luận và khuyến nghị\n")
        report.append(f"**Đánh giá tổng thể:** {risk_assessment['general_recommendation']}\n")
        
        report.append("**Khuyến nghị cụ thể:**\n")
        for recommendation in risk_assessment.get("risk_recommendations", []):
            report.append(f"- {recommendation}")
        
        # Xử lý tên file nếu cần
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report.append(f"\n---\nBáo cáo được tạo tự động bởi RiskEvaluator vào {timestamp}")
        
        return "\n".join(report)
    
    def save_risk_report(self, file_path: str = None, include_charts: bool = True) -> str:
        """
        Tạo và lưu báo cáo rủi ro.
        
        Args:
            file_path (str, optional): Đường dẫn file để lưu báo cáo. Nếu None, tạo tên file tự động.
            include_charts (bool, optional): Bao gồm các biểu đồ trong báo cáo. Mặc định là True.
            
        Returns:
            str: Đường dẫn file đã lưu.
        """
        # Tạo báo cáo
        report = self.generate_risk_report(include_charts=include_charts)
        
        # Xử lý tên file nếu không được cung cấp
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"risk_report_{timestamp}.md"
        
        # Lưu file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"Đã lưu báo cáo rủi ro vào {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu báo cáo rủi ro: {str(e)}")
            return None
    
    def export_risk_metrics_json(self, file_path: str = None) -> Dict[str, Any]:
        """
        Xuất các chỉ số rủi ro dưới dạng JSON.
        
        Args:
            file_path (str, optional): Đường dẫn file để lưu JSON. Nếu None, không lưu file.
            
        Returns:
            Dict[str, Any]: Dictionary chứa các chỉ số rủi ro.
        """
        # Tính toán tất cả các chỉ số rủi ro nếu chưa có
        if not self.results:
            self.evaluate_all_risks()
        
        # Loại bỏ các biểu đồ base64 để giảm kích thước
        metrics_json = {}
        
        for key, value in self.results.items():
            if isinstance(value, dict):
                metrics_json[key] = {k: v for k, v in value.items() if not k.endswith('_chart')}
            else:
                metrics_json[key] = value
        
        # Lưu file nếu được yêu cầu
        if file_path is not None:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics_json, f, indent=4)
                
                self.logger.info(f"Đã xuất chỉ số rủi ro vào {file_path}")
            except Exception as e:
                self.logger.error(f"Lỗi khi xuất chỉ số rủi ro: {str(e)}")
        
        return metrics_json
    
    def run_monte_carlo_simulation(
        self,
        num_simulations: int = 1000,
        days_forward: int = 252,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Thực hiện mô phỏng Monte Carlo để dự báo rủi ro tương lai.
        
        Args:
            num_simulations (int, optional): Số lần mô phỏng. Mặc định là 1000.
            days_forward (int, optional): Số ngày mô phỏng tương lai. Mặc định là 252 (1 năm giao dịch).
            confidence_level (float, optional): Mức độ tin cậy. Mặc định là 0.95 (95%).
            
        Returns:
            Dict[str, Any]: Kết quả mô phỏng.
        """
        # Tính các tham số thống kê từ dữ liệu lịch sử
        returns_mean = self.returns.mean()
        returns_std = self.returns.std()
        
        # Lấy giá trị cuối cùng của equity
        last_equity = self.equity_curve.iloc[-1]
        
        # Tạo mảng lưu kết quả
        simulated_paths = np.zeros((num_simulations, days_forward + 1))
        simulated_paths[:, 0] = last_equity
        
        # Thực hiện mô phỏng Monte Carlo
        for i in range(num_simulations):
            # Sinh ngẫu nhiên các lợi nhuận theo phân phối chuẩn
            random_returns = np.random.normal(returns_mean, returns_std, days_forward)
            
            # Tính equity theo thời gian
            for j in range(days_forward):
                simulated_paths[i, j + 1] = simulated_paths[i, j] * (1 + random_returns[j])
        
        # Tính các thống kê từ kết quả mô phỏng
        final_values = simulated_paths[:, -1]
        
        # Tính các phân vị
        percentiles = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percentiles[f"p{p}"] = np.percentile(final_values, p)
        
        # Tính các chỉ số rủi ro
        # 1. VaR
        var = last_equity - np.percentile(final_values, (1 - confidence_level) * 100)
        var_percent = var / last_equity
        
        # 2. Drawdown tối đa dự kiến
        max_drawdowns = []
        
        for i in range(num_simulations):
            # Tính drawdown cho mỗi đường mô phỏng
            cummax = np.maximum.accumulate(simulated_paths[i])
            drawdown = (simulated_paths[i] - cummax) / cummax
            max_drawdowns.append(abs(drawdown.min()))
        
        expected_max_drawdown = np.mean(max_drawdowns)
        worst_case_drawdown = np.percentile(max_drawdowns, 95)  # 95th percentile
        
        # 3. Xác suất lợi nhuận dương
        profit_probability = np.mean(final_values > last_equity)
        
        # 4. Xác suất drawdown nghiêm trọng
        severe_drawdown_threshold = self.max_drawdown_threshold
        severe_drawdown_probability = np.mean([dd > severe_drawdown_threshold for dd in max_drawdowns])
        
        # Vẽ biểu đồ mô phỏng
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Vẽ các đường mô phỏng (vẽ một số lượng có giới hạn để không quá rối)
        num_paths_to_plot = min(100, num_simulations)
        time_points = np.arange(days_forward + 1)
        
        for i in range(num_paths_to_plot):
            ax.plot(time_points, simulated_paths[i], alpha=0.1, color='blue')
        
        # Vẽ đường trung bình
        mean_path = np.mean(simulated_paths, axis=0)
        ax.plot(time_points, mean_path, color='red', linewidth=2, label='Trung bình')
        
        # Vẽ các đường phân vị
        percentile_5 = np.percentile(simulated_paths, 5, axis=0)
        percentile_95 = np.percentile(simulated_paths, 95, axis=0)
        
        ax.plot(time_points, percentile_5, color='orange', linewidth=1.5, linestyle='--', label='5th Percentile')
        ax.plot(time_points, percentile_95, color='green', linewidth=1.5, linestyle='--', label='95th Percentile')
        
        # Vẽ vùng khoảng tin cậy
        ax.fill_between(time_points, percentile_5, percentile_95, color='gray', alpha=0.2)
        
        ax.set_title(f'Monte Carlo Simulation ({num_simulations} paths, {days_forward} days)')
        ax.set_xlabel('Days')
        ax.set_ylabel('Equity')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Chuyển biểu đồ thành base64 string để trả về
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        simulation_chart = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Vẽ histogram của final values
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(final_values, bins=50, kde=True, ax=ax)
        
        # Vẽ các đường vertical cho các giá trị quan trọng
        ax.axvline(x=last_equity, color='red', linestyle='-', linewidth=1.5, 
                 label=f'Giá trị hiện tại: {last_equity:.2f}')
        
        ax.axvline(x=np.percentile(final_values, 5), color='orange', linestyle='--', linewidth=1.5,
                 label=f'5th Percentile: {np.percentile(final_values, 5):.2f}')
        
        ax.axvline(x=np.percentile(final_values, 95), color='green', linestyle='--', linewidth=1.5,
                 label=f'95th Percentile: {np.percentile(final_values, 95):.2f}')
        
        ax.set_title(f'Distribution of Final Equity Values after {days_forward} days')
        ax.set_xlabel('Equity')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Chuyển biểu đồ thành base64 string để trả về
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        distribution_chart = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Tổng hợp kết quả
        result = {
            "simulation_params": {
                "num_simulations": num_simulations,
                "days_forward": days_forward,
                "confidence_level": confidence_level
            },
            "final_values": {
                "mean": np.mean(final_values),
                "median": np.median(final_values),
                "min": np.min(final_values),
                "max": np.max(final_values),
                "std": np.std(final_values),
                "percentiles": percentiles
            },
            "risk_metrics": {
                "var": var,
                "var_percent": var_percent,
                "expected_max_drawdown": expected_max_drawdown,
                "worst_case_drawdown": worst_case_drawdown,
                "profit_probability": profit_probability,
                "severe_drawdown_probability": severe_drawdown_probability
            },
            "simulation_chart": simulation_chart,
            "distribution_chart": distribution_chart
        }
        
        self.logger.info(f"Đã thực hiện mô phỏng Monte Carlo: {num_simulations} paths, {days_forward} days")
        return result