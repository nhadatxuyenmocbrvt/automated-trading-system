#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module đo lường hiệu suất cho hệ thống backtesting.

Module này cung cấp các lớp và hàm để tính toán các chỉ số đánh giá hiệu suất 
giao dịch trong quá trình backtesting, bao gồm các chỉ số về lợi nhuận,
rủi ro, hiệu quả và các tỷ số đánh giá.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config.constants import TRADING_DAYS_PER_YEAR


class PerformanceMetrics:
    """
    Tính toán các chỉ số đánh giá hiệu suất giao dịch.
    
    Lớp này cung cấp các phương thức để tính toán nhiều loại chỉ số hiệu suất
    khác nhau, bao gồm lợi nhuận, rủi ro, hiệu quả và các tỷ số đánh giá.
    """
    
    def __init__(
            self, 
            equity_curve: pd.Series, 
            trades: pd.DataFrame = None, 
            benchmark: pd.Series = None,
            risk_free_rate: float = 0.02,
            trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
            initial_capital: float = 10000,
            fee_rate: float = 0.001
        ):
        """
        Khởi tạo đối tượng PerformanceMetrics.
        
        Args:
            equity_curve (pd.Series): Chuỗi giá trị vốn theo thời gian.
            trades (pd.DataFrame, optional): DataFrame chứa thông tin các giao dịch.
            benchmark (pd.Series, optional): Chuỗi giá trị của benchmark theo thời gian.
            risk_free_rate (float, optional): Lãi suất phi rủi ro hàng năm. Mặc định là 0.02 (2%).
            trading_days_per_year (int, optional): Số ngày giao dịch trong một năm. Mặc định theo cấu hình.
            initial_capital (float, optional): Vốn ban đầu. Mặc định là 10,000.
            fee_rate (float, optional): Tỷ lệ phí giao dịch. Mặc định là 0.001 (0.1%).
        """
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital
        self.trades = trades
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.fee_rate = fee_rate
        
        # Tính toán các giá trị quan trọng từ equity curve
        self.returns = self._calculate_returns()
        self.log_returns = np.log(1 + self.returns)
        self.cum_returns = (1 + self.returns).cumprod() - 1
        
        # Tính toán các giá trị quan trọng từ benchmark (nếu có)
        if self.benchmark is not None:
            self.benchmark_returns = self.benchmark.pct_change().dropna()
            self.benchmark_log_returns = np.log(1 + self.benchmark_returns)
            self.benchmark_cum_returns = (1 + self.benchmark_returns).cumprod() - 1
    
    def _calculate_returns(self) -> pd.Series:
        """
        Tính toán tỷ suất lợi nhuận từ equity curve.
        
        Returns:
            pd.Series: Chuỗi tỷ suất lợi nhuận theo thời gian.
        """
        return self.equity_curve.pct_change().dropna()
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Tính toán tất cả các chỉ số hiệu suất.
        
        Returns:
            Dict[str, Any]: Dictionary chứa tất cả các chỉ số hiệu suất.
        """
        return {
            # Chỉ số lợi nhuận
            "total_return": self.total_return(),
            "annualized_return": self.annualized_return(),
            "profit_factor": self.profit_factor(),
            "win_rate": self.win_rate(),
            "average_profit": self.average_profit(),
            "average_loss": self.average_loss(),
            "profit_to_loss_ratio": self.profit_to_loss_ratio(),
            
            # Chỉ số rủi ro
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_duration": self.max_drawdown_duration(),
            "drawdown_periods": self.drawdown_periods(),
            "recovery_periods": self.recovery_periods(),
            "volatility": self.volatility(),
            "downside_volatility": self.downside_volatility(),
            "value_at_risk": self.value_at_risk(),
            "conditional_value_at_risk": self.conditional_value_at_risk(),
            
            # Tỷ số đánh giá
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "information_ratio": self.information_ratio() if self.benchmark is not None else None,
            "treynor_ratio": self.treynor_ratio() if self.benchmark is not None else None,
            "omega_ratio": self.omega_ratio(),
            
            # Chỉ số hiệu quả
            "expectancy": self.expectancy(),
            "annualized_expectancy": self.annualized_expectancy(),
            "expectancy_ratio": self.expectancy_ratio(),
            "system_quality_number": self.system_quality_number(),
            
            # Thống kê giao dịch
            "total_trades": self.total_trades(),
            "average_trade_duration": self.average_trade_duration(),
            "average_bars_in_trades": self.average_bars_in_trades(),
            "trades_per_month": self.trades_per_month(),
            "trades_per_year": self.trades_per_year(),
            "annual_turnover": self.annual_turnover(),
            
            # Thống kê phân phối
            "skewness": self.skewness(),
            "kurtosis": self.kurtosis(),
            "tail_ratio": self.tail_ratio(),
            "jarque_bera": self.jarque_bera()
        }
    
    def summary(self) -> pd.DataFrame:
        """
        Tạo bảng tóm tắt các chỉ số hiệu suất quan trọng.
        
        Returns:
            pd.DataFrame: DataFrame chứa tóm tắt các chỉ số hiệu suất.
        """
        metrics = self.calculate_all_metrics()
        
        summary_dict = {
            "Lợi nhuận tổng (%)": metrics["total_return"] * 100,
            "Lợi nhuận hàng năm (%)": metrics["annualized_return"] * 100,
            "Tỷ lệ thắng (%)": metrics["win_rate"] * 100 if metrics["win_rate"] is not None else None,
            "Max Drawdown (%)": metrics["max_drawdown"] * 100,
            "Thời gian Drawdown tối đa (ngày)": metrics["max_drawdown_duration"],
            "Sharpe Ratio": metrics["sharpe_ratio"],
            "Sortino Ratio": metrics["sortino_ratio"],
            "Calmar Ratio": metrics["calmar_ratio"],
            "Profit Factor": metrics["profit_factor"],
            "Tỷ lệ lợi nhuận/lỗ": metrics["profit_to_loss_ratio"],
            "Tổng số giao dịch": metrics["total_trades"],
            "Thời gian giao dịch trung bình (ngày)": metrics["average_trade_duration"],
            "Biến động hàng năm (%)": metrics["volatility"] * 100,
            "VAR (95%)": metrics["value_at_risk"] * 100,
            "Độ lệch": metrics["skewness"],
            "Độ nhọn": metrics["kurtosis"],
            "Chỉ số chất lượng hệ thống (SQN)": metrics["system_quality_number"]
        }
        
        # Thêm các chỉ số liên quan đến benchmark nếu có
        if self.benchmark is not None:
            summary_dict.update({
                "Alpha": self.alpha(),
                "Beta": self.beta(),
                "Information Ratio": metrics["information_ratio"],
                "Treynor Ratio": metrics["treynor_ratio"]
            })
        
        return pd.DataFrame(list(summary_dict.items()), columns=["Chỉ số", "Giá trị"])
    
    #--------------------
    # Chỉ số lợi nhuận
    #--------------------
    
    def total_return(self) -> float:
        """
        Tính tổng lợi nhuận.
        
        Returns:
            float: Tổng lợi nhuận dưới dạng decimal (ví dụ: 0.25 = 25%).
        """
        if len(self.equity_curve) < 2:
            return 0.0
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
    
    def annualized_return(self) -> float:
        """
        Tính lợi nhuận hàng năm.
        
        Returns:
            float: Lợi nhuận hàng năm dưới dạng decimal.
        """
        if len(self.equity_curve) < 2:
            return 0.0
        
        total_days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if total_days <= 0:
            return 0.0
        
        years = total_days / 365.25
        return (1 + self.total_return()) ** (1 / max(years, 1/365.25)) - 1
    
    def profit_factor(self) -> float:
        """
        Tính profit factor (tỷ lệ giữa tổng lợi nhuận và tổng lỗ).
        
        Returns:
            float: Profit factor. Giá trị lớn hơn 1 là tốt.
        """
        if self.trades is None or len(self.trades) == 0:
            return 0.0
            
        gross_profit = self.trades[self.trades['profit'] > 0]['profit'].sum()
        gross_loss = abs(self.trades[self.trades['profit'] < 0]['profit'].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return gross_profit / gross_loss
    
    def win_rate(self) -> Optional[float]:
        """
        Tính tỷ lệ thắng.
        
        Returns:
            float: Tỷ lệ thắng (từ 0 đến 1) hoặc None nếu không có giao dịch.
        """
        if self.trades is None or len(self.trades) == 0:
            return None
            
        winning_trades = len(self.trades[self.trades['profit'] > 0])
        total_trades = len(self.trades)
        
        return winning_trades / total_trades if total_trades > 0 else 0.0
    
    def average_profit(self) -> Optional[float]:
        """
        Tính lợi nhuận trung bình của các giao dịch thắng.
        
        Returns:
            float: Lợi nhuận trung bình của các giao dịch thắng hoặc None nếu không có giao dịch thắng.
        """
        if self.trades is None:
            return None
            
        winning_trades = self.trades[self.trades['profit'] > 0]
        
        if len(winning_trades) == 0:
            return 0.0
            
        return winning_trades['profit'].mean()
    
    def average_loss(self) -> Optional[float]:
        """
        Tính lỗ trung bình của các giao dịch thua.
        
        Returns:
            float: Lỗ trung bình (giá trị dương) của các giao dịch thua hoặc None nếu không có giao dịch thua.
        """
        if self.trades is None:
            return None
            
        losing_trades = self.trades[self.trades['profit'] < 0]
        
        if len(losing_trades) == 0:
            return 0.0
            
        return abs(losing_trades['profit'].mean())
    
    def profit_to_loss_ratio(self) -> Optional[float]:
        """
        Tính tỷ lệ giữa lợi nhuận trung bình và lỗ trung bình.
        
        Returns:
            float: Tỷ lệ lợi nhuận/lỗ hoặc None nếu không có đủ dữ liệu.
        """
        avg_profit = self.average_profit()
        avg_loss = self.average_loss()
        
        if avg_profit is None or avg_loss is None or avg_loss == 0:
            return None
            
        return avg_profit / avg_loss
    
    #--------------------
    # Chỉ số rủi ro
    #--------------------
    
    def max_drawdown(self) -> float:
        """
        Tính drawdown tối đa.
        
        Returns:
            float: Drawdown tối đa dưới dạng decimal (ví dụ: 0.25 = 25%).
        """
        if len(self.equity_curve) < 2:
            return 0.0
            
        # Tính peak-to-trough cho mỗi điểm
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        
        return abs(drawdown.min())
    
    def max_drawdown_duration(self) -> int:
        """
        Tính thời gian của drawdown tối đa (số ngày từ peak đến khi phục hồi lại peak).
        
        Returns:
            int: Số ngày của drawdown tối đa.
        """
        if len(self.equity_curve) < 2:
            return 0
            
        # Tìm đỉnh và đáy
        equity = self.equity_curve
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        
        # Tìm drawdown tối đa
        max_dd = drawdown.min()
        
        # Nếu không có drawdown
        if max_dd == 0:
            return 0
            
        # Tìm thời gian của drawdown tối đa
        # Tìm peak trước drawdown tối đa
        dd_idx = drawdown[drawdown == max_dd].index[0]
        peak_idx = rolling_max[rolling_max.index <= dd_idx].idxmax()
        
        # Tìm thời điểm phục hồi (nếu có)
        recovery_idx = None
        if peak_idx is not None:
            peak_value = equity[peak_idx]
            # Tìm thời điểm sau đáy mà equity vượt qua giá trị peak
            recovery_candidates = equity[(equity.index > dd_idx) & (equity >= peak_value)]
            if not recovery_candidates.empty:
                recovery_idx = recovery_candidates.index[0]
        
        # Tính số ngày
        if recovery_idx is not None:
            return (recovery_idx - peak_idx).days
        else:
            # Nếu chưa phục hồi, tính đến ngày cuối cùng
            return (equity.index[-1] - peak_idx).days
    
    def drawdown_periods(self) -> List[Dict[str, Any]]:
        """
        Xác định tất cả các giai đoạn drawdown.
        
        Returns:
            List[Dict[str, Any]]: Danh sách các giai đoạn drawdown, mỗi giai đoạn là một dict với các thông tin:
                start_date, end_date, drawdown_amount, duration_days.
        """
        if len(self.equity_curve) < 2:
            return []
            
        equity = self.equity_curve
        drawdown_periods = []
        
        # Tính rolling max và drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        
        # Xác định các giai đoạn drawdown
        in_drawdown = False
        start_date = None
        peak_value = None
        
        for date, value in drawdown.items():
            if not in_drawdown and value < 0:
                # Bắt đầu drawdown mới
                in_drawdown = True
                start_date = date
                peak_value = rolling_max[date]
            elif in_drawdown and value == 0:
                # Kết thúc drawdown
                in_drawdown = False
                drawdown_periods.append({
                    'start_date': start_date,
                    'end_date': date,
                    'drawdown_amount': abs(drawdown[start_date:date].min()),
                    'duration_days': (date - start_date).days
                })
                start_date = None
                peak_value = None
        
        # Nếu vẫn đang trong drawdown tại thời điểm cuối cùng
        if in_drawdown:
            drawdown_periods.append({
                'start_date': start_date,
                'end_date': equity.index[-1],
                'drawdown_amount': abs(drawdown[start_date:].min()),
                'duration_days': (equity.index[-1] - start_date).days
            })
        
        return drawdown_periods
    
    def recovery_periods(self) -> List[Dict[str, Any]]:
        """
        Xác định các giai đoạn phục hồi (từ đáy của drawdown đến khi vượt qua đỉnh trước đó).
        
        Returns:
            List[Dict[str, Any]]: Danh sách các giai đoạn phục hồi, mỗi giai đoạn là một dict với các thông tin:
                bottom_date, recovery_date, recovery_amount, duration_days.
        """
        if len(self.equity_curve) < 2:
            return []
            
        equity = self.equity_curve
        recovery_periods = []
        
        # Tính rolling max và drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        
        # Tìm các đáy cục bộ
        is_trough = (drawdown.shift(1) < drawdown) & (drawdown.shift(-1) < drawdown)
        if len(is_trough) > 0:
            troughs = equity[is_trough]
            
            for trough_date, trough_value in troughs.items():
                # Tìm đỉnh trước đáy
                peak_candidates = rolling_max[rolling_max.index < trough_date]
                if not peak_candidates.empty:
                    peak_date = peak_candidates.idxmax()
                    peak_value = equity[peak_date]
                    
                    # Tìm thời điểm phục hồi
                    recovery_candidates = equity[(equity.index > trough_date) & (equity >= peak_value)]
                    if not recovery_candidates.empty:
                        recovery_date = recovery_candidates.index[0]
                        
                        recovery_periods.append({
                            'bottom_date': trough_date,
                            'recovery_date': recovery_date,
                            'recovery_amount': (equity[recovery_date] / trough_value) - 1,
                            'duration_days': (recovery_date - trough_date).days
                        })
        
        return recovery_periods
    
    def volatility(self, annualized: bool = True) -> float:
        """
        Tính độ biến động (volatility) của lợi nhuận.
        
        Args:
            annualized (bool, optional): Nếu True, trả về giá trị hàng năm. Mặc định là True.
            
        Returns:
            float: Độ biến động dưới dạng decimal.
        """
        if len(self.returns) < 2:
            return 0.0
            
        vol = self.returns.std()
        
        if annualized:
            vol = vol * np.sqrt(self.trading_days_per_year)
            
        return vol
    
    def downside_volatility(self, threshold: float = 0.0, annualized: bool = True) -> float:
        """
        Tính độ biến động phía dưới (downside volatility) của lợi nhuận.
        
        Args:
            threshold (float, optional): Ngưỡng để xác định lợi nhuận "không mong muốn". Mặc định là 0.
            annualized (bool, optional): Nếu True, trả về giá trị hàng năm. Mặc định là True.
            
        Returns:
            float: Độ biến động phía dưới dưới dạng decimal.
        """
        if len(self.returns) < 2:
            return 0.0
            
        downside_returns = self.returns[self.returns < threshold]
        
        if len(downside_returns) == 0:
            return 0.0
            
        downside_vol = downside_returns.std()
        
        if annualized:
            downside_vol = downside_vol * np.sqrt(self.trading_days_per_year)
            
        return downside_vol
    
    def value_at_risk(self, confidence: float = 0.95) -> float:
        """
        Tính Value at Risk (VaR) theo phương pháp lịch sử.
        
        Args:
            confidence (float, optional): Mức độ tin cậy. Mặc định là 0.95 (95%).
            
        Returns:
            float: VaR dưới dạng decimal.
        """
        if len(self.returns) < 2:
            return 0.0
            
        return abs(np.percentile(self.returns, 100 * (1 - confidence)))
    
    def conditional_value_at_risk(self, confidence: float = 0.95) -> float:
        """
        Tính Conditional Value at Risk (CVaR) - hay Expected Shortfall (ES).
        
        Args:
            confidence (float, optional): Mức độ tin cậy. Mặc định là 0.95 (95%).
            
        Returns:
            float: CVaR dưới dạng decimal.
        """
        if len(self.returns) < 2:
            return 0.0
            
        var = self.value_at_risk(confidence)
        return abs(self.returns[self.returns <= -var].mean())
    
    #--------------------
    # Tỷ số đánh giá
    #--------------------
    
    def sharpe_ratio(self, annualized: bool = True) -> float:
        """
        Tính Sharpe Ratio.
        
        Args:
            annualized (bool, optional): Nếu True, trả về giá trị hàng năm. Mặc định là True.
            
        Returns:
            float: Sharpe Ratio.
        """
        if len(self.returns) < 2:
            return 0.0
            
        excess_returns = self.returns - (self.risk_free_rate / self.trading_days_per_year)
        sharpe = excess_returns.mean() / self.returns.std()
        
        if annualized:
            sharpe = sharpe * np.sqrt(self.trading_days_per_year)
            
        return sharpe
    
    def sortino_ratio(self, annualized: bool = True, threshold: float = 0.0) -> float:
        """
        Tính Sortino Ratio.
        
        Args:
            annualized (bool, optional): Nếu True, trả về giá trị hàng năm. Mặc định là True.
            threshold (float, optional): Ngưỡng để xác định lợi nhuận "không mong muốn". Mặc định là 0.
            
        Returns:
            float: Sortino Ratio.
        """
        if len(self.returns) < 2:
            return 0.0
            
        excess_returns = self.returns - (self.risk_free_rate / self.trading_days_per_year)
        downside_vol = self.downside_volatility(threshold, annualized=False)
        
        if downside_vol == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
            
        sortino = excess_returns.mean() / downside_vol
        
        if annualized:
            sortino = sortino * np.sqrt(self.trading_days_per_year)
            
        return sortino
    
    def calmar_ratio(self) -> float:
        """
        Tính Calmar Ratio.
        
        Returns:
            float: Calmar Ratio.
        """
        annualized_return = self.annualized_return()
        max_dd = self.max_drawdown()
        
        if max_dd == 0:
            return float('inf') if annualized_return > 0 else 0.0
            
        return annualized_return / max_dd
    
    def information_ratio(self) -> Optional[float]:
        """
        Tính Information Ratio so với benchmark.
        
        Returns:
            float: Information Ratio hoặc None nếu không có benchmark.
        """
        if self.benchmark is None or len(self.returns) < 2 or len(self.benchmark_returns) < 2:
            return None
            
        # Đảm bảo rằng hai chuỗi cùng kích thước
        common_index = self.returns.index.intersection(self.benchmark_returns.index)
        if len(common_index) < 2:
            return None
            
        returns = self.returns.loc[common_index]
        benchmark_returns = self.benchmark_returns.loc[common_index]
        
        # Tính tracking error
        tracking_diff = returns - benchmark_returns
        tracking_error = tracking_diff.std() * np.sqrt(self.trading_days_per_year)
        
        if tracking_error == 0:
            return 0.0
            
        # Tính active return
        active_return = self.annualized_return() - (
                (1 + benchmark_returns).prod() ** (self.trading_days_per_year / len(benchmark_returns)) - 1
            )
            
        return active_return / tracking_error
    
    def treynor_ratio(self) -> Optional[float]:
        """
        Tính Treynor Ratio.
        
        Returns:
            float: Treynor Ratio hoặc None nếu không có benchmark.
        """
        if self.benchmark is None:
            return None
            
        beta = self.beta()
        
        if beta == 0:
            return float('inf') if self.annualized_return() > self.risk_free_rate else float('-inf')
            
        return (self.annualized_return() - self.risk_free_rate) / beta
    
    def omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Tính Omega Ratio.
        
        Args:
            threshold (float, optional): Ngưỡng lợi nhuận mong muốn. Mặc định là 0.
            
        Returns:
            float: Omega Ratio.
        """
        if len(self.returns) < 2:
            return 0.0
            
        returns_above = self.returns[self.returns > threshold].sum()
        returns_below = abs(self.returns[self.returns < threshold].sum())
        
        if returns_below == 0:
            return float('inf') if returns_above > 0 else 0.0
            
        return returns_above / returns_below
    
    def alpha(self) -> Optional[float]:
        """
        Tính Alpha so với benchmark (CAPM).
        
        Returns:
            float: Alpha hoặc None nếu không có benchmark.
        """
        if self.benchmark is None:
            return None
            
        beta = self.beta()
        
        benchmark_return = (
            (1 + self.benchmark_returns).prod() ** (252 / len(self.benchmark_returns)) - 1
        )
        
        return self.annualized_return() - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
    
    def beta(self) -> Optional[float]:
        """
        Tính Beta so với benchmark.
        
        Returns:
            float: Beta hoặc None nếu không có benchmark.
        """
        if self.benchmark is None or len(self.returns) < 2 or len(self.benchmark_returns) < 2:
            return None
            
        # Đảm bảo rằng hai chuỗi cùng kích thước
        common_index = self.returns.index.intersection(self.benchmark_returns.index)
        
        if len(common_index) < 2:
            return None
            
        returns = self.returns.loc[common_index]
        benchmark_returns = self.benchmark_returns.loc[common_index]
        
        # Tính covariance và variance
        covariance = returns.cov(benchmark_returns)
        variance = benchmark_returns.var()
        
        if variance == 0:
            return 0.0
            
        return covariance / variance
    
    #--------------------
    # Chỉ số hiệu quả
    #--------------------
    
    def expectancy(self) -> float:
        """
        Tính độ kỳ vọng (expectancy) của hệ thống.
        
        Returns:
            float: Độ kỳ vọng (kỳ vọng lợi nhuận trên mỗi giao dịch).
        """
        if self.trades is None or len(self.trades) == 0:
            return 0.0
            
        win_rate = self.win_rate()
        avg_win = self.average_profit()
        avg_loss = self.average_loss()
        
        if win_rate is None or avg_win is None or avg_loss is None:
            return 0.0
            
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    def annualized_expectancy(self) -> float:
        """
        Tính độ kỳ vọng hàng năm.
        
        Returns:
            float: Độ kỳ vọng hàng năm.
        """
        expectancy = self.expectancy()
        trades_per_year = self.trades_per_year()
        
        return expectancy * trades_per_year
    
    def expectancy_ratio(self) -> float:
        """
        Tính tỷ lệ kỳ vọng (expectancy ratio).
        
        Returns:
            float: Tỷ lệ kỳ vọng.
        """
        if self.trades is None or len(self.trades) == 0:
            return 0.0
            
        win_rate = self.win_rate()
        profit_to_loss_ratio = self.profit_to_loss_ratio()
        
        if win_rate is None or profit_to_loss_ratio is None:
            return 0.0
            
        return (win_rate * profit_to_loss_ratio) / (1 - win_rate)
    
    def system_quality_number(self) -> float:
        """
        Tính System Quality Number (SQN).
        
        Returns:
            float: System Quality Number.
        """
        if self.trades is None or len(self.trades) == 0:
            return 0.0
            
        # Tính trung bình và độ lệch chuẩn của lợi nhuận trên mỗi giao dịch
        mean_r = self.trades['profit'].mean()
        std_r = self.trades['profit'].std()
        
        if std_r == 0:
            return 0.0
            
        return (mean_r / std_r) * np.sqrt(len(self.trades))
    
    #--------------------
    # Thống kê giao dịch
    #--------------------
    
    def total_trades(self) -> int:
        """
        Tính tổng số giao dịch.
        
        Returns:
            int: Tổng số giao dịch.
        """
        if self.trades is None:
            return 0
            
        return len(self.trades)
    
    def average_trade_duration(self) -> Optional[float]:
        """
        Tính thời gian trung bình của mỗi giao dịch.
        
        Returns:
            float: Thời gian trung bình (số ngày) hoặc None nếu không có thông tin.
        """
        if self.trades is None or 'duration' not in self.trades.columns:
            return None
            
        return self.trades['duration'].mean()
    
    def average_bars_in_trades(self) -> Optional[float]:
        """
        Tính số thanh nến trung bình trong mỗi giao dịch.
        
        Returns:
            float: Số thanh nến trung bình hoặc None nếu không có thông tin.
        """
        if self.trades is None or 'bars' not in self.trades.columns:
            return None
            
        return self.trades['bars'].mean()
    
    def trades_per_month(self) -> float:
        """
        Tính số giao dịch trung bình mỗi tháng.
        
        Returns:
            float: Số giao dịch trung bình mỗi tháng.
        """
        if self.trades is None or len(self.trades) == 0:
            return 0.0
            
        if 'entry_time' not in self.trades.columns:
            # Ước tính từ số ngày giao dịch
            total_days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
            if total_days <= 0:
                return 0.0
                
            months = total_days / 30.4375  # Số ngày trung bình trong tháng
            return len(self.trades) / max(months, 1/30.4375)
        else:
            # Tính từ dữ liệu giao dịch
            first_trade = self.trades['entry_time'].min()
            last_trade = self.trades['entry_time'].max()
            
            if pd.isna(first_trade) or pd.isna(last_trade):
                return 0.0
                
            months = (last_trade - first_trade).days / 30.4375
            return len(self.trades) / max(months, 1/30.4375)
    
    def trades_per_year(self) -> float:
        """
        Tính số giao dịch trung bình mỗi năm.
        
        Returns:
            float: Số giao dịch trung bình mỗi năm.
        """
        return self.trades_per_month() * 12
    
    def annual_turnover(self) -> float:
        """
        Tính tỷ lệ quay vòng vốn hàng năm.
        
        Returns:
            float: Tỷ lệ quay vòng vốn hàng năm.
        """
        if self.trades is None or len(self.trades) == 0:
            return 0.0
            
        if 'volume' not in self.trades.columns:
            # Ước tính từ số lượng giao dịch
            return self.trades_per_year() * 2  # Mỗi giao dịch có vào và ra
        else:
            # Tính từ khối lượng giao dịch
            total_volume = self.trades['volume'].sum()
            years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
            average_equity = self.equity_curve.mean()
            
            if years <= 0 or average_equity <= 0:
                return 0.0
                
            return total_volume / (average_equity * max(years, 1/365.25))
    
    #--------------------
    # Thống kê phân phối
    #--------------------
    
    def skewness(self) -> float:
        """
        Tính độ lệch (skewness) của phân phối lợi nhuận.
        
        Returns:
            float: Độ lệch.
        """
        if len(self.returns) < 2:
            return 0.0
            
        return stats.skew(self.returns.dropna())
    
    def kurtosis(self) -> float:
        """
        Tính độ nhọn (kurtosis) của phân phối lợi nhuận.
        
        Returns:
            float: Độ nhọn.
        """
        if len(self.returns) < 2:
            return 0.0
            
        return stats.kurtosis(self.returns.dropna())
    
    def tail_ratio(self, quantile: float = 0.05) -> float:
        """
        Tính tỷ lệ đuôi (tail ratio) của phân phối lợi nhuận.
        
        Args:
            quantile (float, optional): Phần trăm đuôi để tính. Mặc định là 0.05 (5%).
            
        Returns:
            float: Tỷ lệ đuôi.
        """
        if len(self.returns) < 2:
            return 0.0
            
        upper_tail = np.abs(self.returns.quantile(1 - quantile))
        lower_tail = np.abs(self.returns.quantile(quantile))
        
        if lower_tail == 0:
            return float('inf') if upper_tail > 0 else 0.0
            
        return upper_tail / lower_tail
    
    def jarque_bera(self) -> Tuple[float, float]:
        """
        Tính kiểm định Jarque-Bera để kiểm tra tính chuẩn của phân phối.
        
        Returns:
            Tuple[float, float]: (giá trị thống kê, p-value).
        """
        if len(self.returns) < 2:
            return (0.0, 1.0)
            
        return stats.jarque_bera(self.returns.dropna())
    
    #--------------------
    # Trực quan hóa
    #--------------------
    
    def plot_equity_curve(self, figsize: Tuple[int, int] = (12, 6), 
                       include_benchmark: bool = True, 
                       include_drawdown: bool = True) -> plt.Figure:
        """
        Vẽ đường cong vốn (equity curve).
        
        Args:
            figsize (Tuple[int, int], optional): Kích thước hình. Mặc định là (12, 6).
            include_benchmark (bool, optional): Nếu True, vẽ cả benchmark. Mặc định là True.
            include_drawdown (bool, optional): Nếu True, vẽ cả drawdown. Mặc định là True.
            
        Returns:
            plt.Figure: Đối tượng Figure của matplotlib.
        """
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Chuẩn hóa equity curve và benchmark (nếu có)
        equity_normalized = self.equity_curve / self.equity_curve.iloc[0]
        ax1.plot(equity_normalized, label='Chiến lược', color='blue', linewidth=2)
        
        if include_benchmark and self.benchmark is not None:
            benchmark_normalized = self.benchmark / self.benchmark.iloc[0]
            ax1.plot(benchmark_normalized, label='Benchmark', color='green', linestyle='--', linewidth=1.5)
        
        ax1.set_xlabel('Thời gian')
        ax1.set_ylabel('Giá trị vốn (đã chuẩn hóa)')
        ax1.grid(True, alpha=0.3)
        
        if include_drawdown:
            ax2 = ax1.twinx()
            
            # Tính drawdown
            rolling_max = self.equity_curve.cummax()
            drawdown = (self.equity_curve - rolling_max) / rolling_max
            
            ax2.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_ylim(min(drawdown.min() * 1.5, -0.05), 0.01)
        
        plt.title('Đường cong vốn' + (' và Drawdown' if include_drawdown else ''))
        plt.legend(loc='best')
        plt.tight_layout()
        
        return fig
    
    def plot_returns_distribution(self, figsize: Tuple[int, int] = (12, 6), 
                               bins: int = 50) -> plt.Figure:
        """
        Vẽ phân phối lợi nhuận.
        
        Args:
            figsize (Tuple[int, int], optional): Kích thước hình. Mặc định là (12, 6).
            bins (int, optional): Số lượng bins. Mặc định là 50.
            
        Returns:
            plt.Figure: Đối tượng Figure của matplotlib.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.histplot(self.returns, bins=bins, kde=True, ax=ax)
        
        # Vẽ đường phân phối chuẩn để so sánh
        x = np.linspace(self.returns.min(), self.returns.max(), 1000)
        y = stats.norm.pdf(x, self.returns.mean(), self.returns.std())
        ax.plot(x, y, 'r-', alpha=0.7, label='Phân phối chuẩn')
        
        # Vẽ các chỉ số quan trọng
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(self.returns.mean(), color='green', linestyle='--', alpha=0.7, 
                label=f'Trung bình: {self.returns.mean():.4f}')
        
        # Vẽ VaR
        var_95 = self.value_at_risk(0.95)
        ax.axvline(-var_95, color='red', linestyle='--', alpha=0.7,
                label=f'VaR (95%): {var_95:.4f}')
        
        ax.set_xlabel('Lợi nhuận')
        ax.set_ylabel('Tần suất')
        ax.set_title('Phân phối lợi nhuận')
        
        # Thêm thông tin thống kê
        stats_text = (
            f'Trung bình: {self.returns.mean():.4f}\n'
            f'Độ lệch chuẩn: {self.returns.std():.4f}\n'
            f'Độ lệch (Skewness): {self.skewness():.4f}\n'
            f'Độ nhọn (Kurtosis): {self.kurtosis():.4f}'
        )
        
        plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.legend()
        plt.tight_layout()
        
        return fig
    
    def plot_drawdown_periods(self, top_n: int = 5, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Vẽ các giai đoạn drawdown lớn nhất.
        
        Args:
            top_n (int, optional): Số lượng giai đoạn drawdown lớn nhất để hiển thị. Mặc định là 5.
            figsize (Tuple[int, int], optional): Kích thước hình. Mặc định là (12, 6).
            
        Returns:
            plt.Figure: Đối tượng Figure của matplotlib.
        """
        periods = self.drawdown_periods()
        
        if not periods:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Không có giai đoạn drawdown', 
                 horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Sắp xếp theo mức độ drawdown
        periods.sort(key=lambda x: x['drawdown_amount'], reverse=True)
        top_periods = periods[:min(top_n, len(periods))]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, period in enumerate(top_periods):
            start_date = period['start_date']
            end_date = period['end_date']
            drawdown_amount = period['drawdown_amount']
            duration = period['duration_days']
            
            # Vẽ giai đoạn drawdown
            equity_slice = self.equity_curve[start_date:end_date]
            normalized_equity = equity_slice / equity_slice.iloc[0]
            
            ax.plot(equity_slice.index, normalized_equity, 
                 label=f'#{i+1}: {drawdown_amount:.2%} ({duration} ngày)')
        
        ax.set_xlabel('Thời gian')
        ax.set_ylabel('Giá trị vốn (đã chuẩn hóa)')
        ax.set_title(f'Top {len(top_periods)} giai đoạn drawdown lớn nhất')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        return fig
    
    def plot_monthly_returns_heatmap(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Vẽ heatmap lợi nhuận theo tháng và năm.
        
        Args:
            figsize (Tuple[int, int], optional): Kích thước hình. Mặc định là (12, 8).
            
        Returns:
            plt.Figure: Đối tượng Figure của matplotlib.
        """
        if len(self.returns) < 30:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Không đủ dữ liệu để vẽ heatmap lợi nhuận hàng tháng', 
                 horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Tính lợi nhuận hàng tháng
        monthly_returns = self.equity_curve.resample('M').last().pct_change().dropna()
        
        # Tạo DataFrame với chỉ số là năm và tháng
        monthly_returns.index = pd.MultiIndex.from_arrays(
            [monthly_returns.index.year, monthly_returns.index.month],
            names=['Năm', 'Tháng']
        )
        
        # Pivot để tạo bảng năm x tháng
        heatmap_data = monthly_returns.unstack(level='Tháng')
        
        # Vẽ heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        cmap = plt.cm.RdYlGn  # Red for negative, Green for positive
        sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap=cmap, center=0, ax=ax)
        
        ax.set_title('Lợi nhuận hàng tháng')
        ax.set_ylabel('Năm')
        ax.set_xlabel('Tháng')
        ax.set_xticklabels(['Th.1', 'Th.2', 'Th.3', 'Th.4', 'Th.5', 'Th.6', 
                          'Th.7', 'Th.8', 'Th.9', 'Th.10', 'Th.11', 'Th.12'])
        
        plt.tight_layout()
        
        return fig