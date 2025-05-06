#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module trực quan hóa hiệu suất cho kết quả backtest.

Module này cung cấp các lớp và hàm để tạo các biểu đồ hiệu suất cho
kết quả backtest, bao gồm đường cong lợi nhuận, biểu đồ drawdown,
phân phối lợi nhuận, và nhiều loại biểu đồ hiệu suất khác.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta
import json
from pathlib import Path
from matplotlib.ticker import FuncFormatter
from io import BytesIO
import base64
import warnings

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import BacktestMetric, TRADING_DAYS_PER_YEAR
from config.system_config import get_system_config, BACKTEST_DIR

# Cố gắng import các thư viện phụ thuộc bổ sung
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Kiểm tra tính sẵn có của module hiệu suất
try:
    from backtesting.performance_metrics import PerformanceMetrics
    PERFORMANCE_METRICS_AVAILABLE = True
except ImportError:
    PERFORMANCE_METRICS_AVAILABLE = False

# Style cho matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')

# Định nghĩa bảng màu
COLOR_PALETTE = {
    'equity': '#1f77b4',          # Xanh dương
    'benchmark': '#ff7f0e',       # Cam
    'drawdown': '#d62728',        # Đỏ
    'positive': '#2ca02c',        # Xanh lá
    'negative': '#d62728',        # Đỏ
    'neutral': '#7f7f7f',         # Xám
    'highlight': '#9467bd',       # Tím
    'secondary': '#8c564b',       # Nâu
    'accent': '#e377c2',          # Hồng
    'warning': '#ff7f0e',         # Cam
    'info': '#17becf'             # Xanh ngọc
}


class PerformanceCharts:
    """
    Lớp tạo các biểu đồ hiệu suất từ kết quả backtest.
    
    Lớp này cung cấp các phương thức để tạo và xuất các biểu đồ hiệu suất
    khác nhau cho kết quả backtest, giúp phân tích và đánh giá chiến lược giao dịch.
    """
    
    def __init__(
        self,
        equity_curve: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None,
        returns: Optional[pd.Series] = None,
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        initial_capital: float = 10000,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        style: str = 'darkgrid',
        logger = None
    ):
        """
        Khởi tạo đối tượng PerformanceCharts.
        
        Args:
            equity_curve: Series chứa giá trị tài sản theo thời gian
            trades: DataFrame chứa thông tin các giao dịch
            returns: Series chứa lợi nhuận phần trăm theo thời gian
            benchmark: Series chứa giá trị benchmark để so sánh
            risk_free_rate: Lãi suất phi rủi ro hàng năm
            initial_capital: Vốn ban đầu
            figsize: Kích thước mặc định cho biểu đồ
            dpi: Độ phân giải hình ảnh
            style: Phong cách biểu đồ ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
            logger: Logger tùy chỉnh
        """
        """
        Khởi tạo đối tượng PerformanceCharts.
        
        Args:
            equity_curve: Series chứa giá trị tài sản theo thời gian
            trades: DataFrame chứa thông tin các giao dịch
            returns: Series chứa lợi nhuận phần trăm theo thời gian
            benchmark: Series chứa giá trị benchmark để so sánh
            risk_free_rate: Lãi suất phi rủi ro hàng năm
            initial_capital: Vốn ban đầu
            figsize: Kích thước mặc định cho biểu đồ
            dpi: Độ phân giải hình ảnh
            style: Phong cách biểu đồ ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("performance_charts")
        
        # Lưu trữ dữ liệu
        self.equity_curve = equity_curve
        self.trades = trades
        self.returns = returns
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        self.initial_capital = initial_capital
        
        # Thiết lập biểu đồ
        self.figsize = figsize
        self.dpi = dpi
        
        # Thiết lập style
        if style in ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']:
            sns.set_style(style)
        
        # Thiết lập font và màu cho matplotlib
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
        
        # Đảm bảo các dữ liệu có định dạng đúng
        if self.equity_curve is not None and not isinstance(self.equity_curve.index, pd.DatetimeIndex):
            try:
                self.equity_curve.index = pd.to_datetime(self.equity_curve.index)
            except Exception as e:
                self.logger.warning(f"Không thể chuyển đổi index của equity_curve thành datetime: {e}")
        
        if self.benchmark is not None and not isinstance(self.benchmark.index, pd.DatetimeIndex):
            try:
                self.benchmark.index = pd.to_datetime(self.benchmark.index)
            except Exception as e:
                self.logger.warning(f"Không thể chuyển đổi index của benchmark thành datetime: {e}")
        
        # Tính toán returns nếu có equity_curve và không truyền returns
        if self.equity_curve is not None and self.returns is None:
            self.returns = self.equity_curve.pct_change().fillna(0)
        
        # Đảm bảo các cột datetime trong trades có định dạng đúng
        if self.trades is not None:
            for col in ['entry_time', 'exit_time']:
                if col in self.trades.columns and not pd.api.types.is_datetime64_any_dtype(self.trades[col]):
                    try:
                        self.trades[col] = pd.to_datetime(self.trades[col])
                    except Exception as e:
                        self.logger.warning(f"Không thể chuyển đổi cột {col} trong trades thành datetime: {e}")
        
        # Kiểm tra PerformanceMetrics
        self.performance_metrics = None
        if PERFORMANCE_METRICS_AVAILABLE and self.equity_curve is not None:
            try:
                self.performance_metrics = PerformanceMetrics(
                    equity_curve=self.equity_curve,
                    trades=self.trades,
                    benchmark=self.benchmark,
                    risk_free_rate=self.risk_free_rate,
                    initial_capital=self.initial_capital
                )
            except Exception as e:
                self.logger.warning(f"Không thể khởi tạo PerformanceMetrics: {e}")
        
        self.logger.info(f"Khởi tạo PerformanceCharts thành công")
    
    def set_data(
        self,
        equity_curve: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None,
        returns: Optional[pd.Series] = None,
        benchmark: Optional[pd.Series] = None
    ):
        """
        Cập nhật dữ liệu cho biểu đồ.
        
        Args:
            equity_curve: Series chứa giá trị tài sản theo thời gian
            trades: DataFrame chứa thông tin các giao dịch
            returns: Series chứa lợi nhuận phần trăm theo thời gian
            benchmark: Series chứa giá trị benchmark để so sánh
        """
        if equity_curve is not None:
            self.equity_curve = equity_curve
        
        if trades is not None:
            self.trades = trades
        
        if returns is not None:
            self.returns = returns
        elif equity_curve is not None and self.returns is None:
            # Tính toán returns từ equity_curve mới
            self.returns = self.equity_curve.pct_change().fillna(0)
        
        if benchmark is not None:
            self.benchmark = benchmark
        
        # Cập nhật PerformanceMetrics
        if PERFORMANCE_METRICS_AVAILABLE and self.equity_curve is not None:
            try:
                self.performance_metrics = PerformanceMetrics(
                    equity_curve=self.equity_curve,
                    trades=self.trades,
                    benchmark=self.benchmark,
                    risk_free_rate=self.risk_free_rate,
                    initial_capital=self.initial_capital
                )
            except Exception as e:
                self.logger.warning(f"Không thể khởi tạo PerformanceMetrics: {e}")
        
        self.logger.info(f"Cập nhật dữ liệu thành công")
    
    def _format_date_axis(self, ax, rotation=45):
        """
        Định dạng trục ngày tháng trên biểu đồ.
        
        Args:
            ax: Trục matplotlib
            rotation: Góc xoay của nhãn
        """
        # Định dạng trục x dựa trên độ dài dữ liệu
        if self.equity_curve is not None and len(self.equity_curve) > 0:
            try:
                date_range = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
                
                if date_range > 730:  # > 2 năm
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    ax.xaxis.set_major_locator(mdates.YearLocator())
                elif date_range > 180:  # > 6 tháng
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                elif date_range > 30:  # > 1 tháng
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                else:  # <= 1 tháng
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            except (AttributeError, TypeError) as e:
                # Xử lý trường hợp index không phải DatetimeIndex
                self.logger.warning(f"Không thể định dạng trục ngày tháng: {e}")
                # Thử chuyển đổi nếu có thể
                try:
                    if not isinstance(self.equity_curve.index, pd.DatetimeIndex):
                        self.equity_curve.index = pd.to_datetime(self.equity_curve.index)
                        self._format_date_axis(ax, rotation)  # Gọi lại hàm sau khi chuyển đổi
                        return
                except Exception:
                    pass
        
        try:
            plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')
        except Exception as e:
            self.logger.warning(f"Không thể định dạng nhãn trục x: {e}")
    
    def _format_pct_axis(self, ax, axis='y'):
        """
        Định dạng trục phần trăm trên biểu đồ.
        
        Args:
            ax: Trục matplotlib
            axis: Trục cần định dạng ('x', 'y')
        """
        formatter = FuncFormatter(lambda y, _: f'{y:.1%}')
        
        if axis.lower() == 'y':
            ax.yaxis.set_major_formatter(formatter)
        else:
            ax.xaxis.set_major_formatter(formatter)
    
    def create_performance_dashboard(
        self, 
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Tạo bảng điều khiển hiệu suất tổng hợp.
        
        Args:
            figsize: Kích thước biểu đồ, nếu None sẽ sử dụng giá trị mặc định
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Figure của matplotlib
        """
        if self.equity_curve is None:
            self.logger.error("Không có dữ liệu equity_curve để tạo bảng điều khiển")
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center')
            return fig
        
        # Tạo figure với 2x2 subplots
        fig = plt.figure(figsize=figsize or (self.figsize[0] * 2, self.figsize[1] * 2), dpi=self.dpi)
        
        # Định nghĩa GridSpec
        gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1])
        
        # 1. Equity Curve (hàng đầu, cột trái + phải)
        ax_equity = fig.add_subplot(gs[0, :])
        ax_equity.plot(self.equity_curve.index, self.equity_curve, 
                      label="Equity", color=COLOR_PALETTE['equity'], linewidth=2)
        
        # Thêm benchmark nếu có
        if self.benchmark is not None:
            # Điều chỉnh giá trị ban đầu của benchmark để so sánh
            if not self.benchmark.empty and not self.equity_curve.empty:
                # Lấy ngày bắt đầu chung
                start_date = max(self.benchmark.index[0], self.equity_curve.index[0])
                
                # Điều chỉnh benchmark
                benchmark_adjusted = self.benchmark.copy()
                initial_equity = self.equity_curve.loc[self.equity_curve.index >= start_date].iloc[0]
                initial_benchmark = self.benchmark.loc[self.benchmark.index >= start_date].iloc[0]
                
                if initial_benchmark != 0:
                    scale_factor = initial_equity / initial_benchmark
                    benchmark_adjusted = benchmark_adjusted * scale_factor
                
                ax_equity.plot(benchmark_adjusted.index, benchmark_adjusted, 
                             label="Benchmark", color=COLOR_PALETTE['benchmark'], 
                             linewidth=1.5, linestyle='--', alpha=0.8)
        
        # Thêm điểm giao dịch nếu có
        if self.trades is not None and not self.trades.empty:
            if 'entry_time' in self.trades.columns and 'exit_time' in self.trades.columns and 'profit' in self.trades.columns:
                # Giao dịch có lãi
                profitable_trades = self.trades[self.trades['profit'] > 0]
                if not profitable_trades.empty:
                    ax_equity.scatter(profitable_trades['exit_time'], 
                                    self.equity_curve.reindex(profitable_trades['exit_time']),
                                    marker='^', s=30, color=COLOR_PALETTE['positive'], alpha=0.7, 
                                    label="Profitable Trade")
                
                # Giao dịch lỗ
                losing_trades = self.trades[self.trades['profit'] < 0]
                if not losing_trades.empty:
                    ax_equity.scatter(losing_trades['exit_time'], 
                                    self.equity_curve.reindex(losing_trades['exit_time']),
                                    marker='v', s=30, color=COLOR_PALETTE['negative'], alpha=0.7, 
                                    label="Losing Trade")
        
        # Tính toán ROI
        roi = None
        if len(self.equity_curve) > 1:
            roi = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        
        # Tính Annualized Return
        annualized_return = None
        if len(self.equity_curve) > 1:
            start_date = self.equity_curve.index[0]
            end_date = self.equity_curve.index[-1]
            days = (end_date - start_date).days
            if days > 0:
                annualized_return = ((1 + roi) ** (365 / days)) - 1
        
        # Thêm chú thích hiệu suất
        if roi is not None:
            performance_text = f"Total Return: {roi:.2%}"
            if annualized_return is not None:
                performance_text += f"\nAnnualized: {annualized_return:.2%}"
            
            ax_equity.text(0.01, 0.97, performance_text,
                          transform=ax_equity.transAxes,
                          verticalalignment='top', horizontalalignment='left',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Định dạng trục
        self._format_date_axis(ax_equity)
        
        # Tùy chỉnh biểu đồ
        ax_equity.set_title("Equity Curve", fontsize=16)
        ax_equity.set_xlabel("Date", fontsize=12)
        ax_equity.set_ylabel("Equity", fontsize=12)
        ax_equity.grid(True, alpha=0.3)
        ax_equity.legend(loc='upper left')
        
        # 2. Drawdown (hàng dưới, cột trái)
        ax_drawdown = fig.add_subplot(gs[1, 0])
        
        # Tính toán drawdown
        rolling_max = self.equity_curve.cummax()
        drawdown_series = (self.equity_curve - rolling_max) / rolling_max
        
        ax_drawdown.fill_between(drawdown_series.index, 0, drawdown_series, 
                               color=COLOR_PALETTE['drawdown'], alpha=0.3)
        ax_drawdown.plot(drawdown_series.index, drawdown_series, 
                       color=COLOR_PALETTE['drawdown'], linewidth=1, alpha=0.5)
        
        # Đánh dấu drawdown tối đa
        max_drawdown_idx = drawdown_series.idxmin()
        max_drawdown = drawdown_series.min()
        
        ax_drawdown.scatter(max_drawdown_idx, max_drawdown, 
                          s=50, color=COLOR_PALETTE['highlight'], zorder=5,
                          marker='o', label=f"Max DD: {max_drawdown:.2%}")
        
        # Thêm các ngưỡng tham chiếu
        ax_drawdown.axhline(y=-0.1, color='orange', linestyle='--', alpha=0.5, label="-10%")
        ax_drawdown.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5, label="-20%")
        
        # Định dạng trục
        self._format_date_axis(ax_drawdown)
        self._format_pct_axis(ax_drawdown)
        
        # Tùy chỉnh biểu đồ
        ax_drawdown.set_title("Drawdown", fontsize=14)
        ax_drawdown.set_ylabel("Drawdown", fontsize=12)
        ax_drawdown.set_xlabel("Date", fontsize=12)
        ax_drawdown.grid(True, alpha=0.3)
        ax_drawdown.legend(loc='lower right', fontsize=10)
        
        # 3. Phân phối lợi nhuận hàng tháng (hàng dưới, cột phải)
        ax_monthly = fig.add_subplot(gs[1, 1])
        
        if self.returns is not None:
            # Tính toán lợi nhuận theo tháng
            monthly_returns = self.returns.groupby([
                lambda x: x.year,
                lambda x: x.month
            ]).apply(lambda x: (1 + x).prod() - 1)
            
            # Vẽ histogram với KDE
            if len(monthly_returns) > 1:
                sns.histplot(monthly_returns, bins=20, kde=True, ax=ax_monthly,
                           color=COLOR_PALETTE['equity'], alpha=0.7)
                
                # Thêm đường vertical tại giá trị 0
                ax_monthly.axvline(x=0, color='black', linestyle='--', alpha=0.7)
                
                # Tính các thống kê
                avg_monthly = monthly_returns.mean()
                positive_months = (monthly_returns > 0).sum() / len(monthly_returns)
                best_month = monthly_returns.max()
                worst_month = monthly_returns.min()
                
                # Thêm chú thích thống kê
                stats_text = (
                    f"Avg Monthly: {avg_monthly:.2%}\n"
                    f"Positive Months: {positive_months:.1%}\n"
                    f"Best Month: {best_month:.2%}\n"
                    f"Worst Month: {worst_month:.2%}"
                )
                
                ax_monthly.text(0.95, 0.95, stats_text, transform=ax_monthly.transAxes,
                              verticalalignment='top', horizontalalignment='right',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # Định dạng trục
                self._format_pct_axis(ax_monthly, axis='x')
                
                # Tùy chỉnh biểu đồ
                ax_monthly.set_title("Monthly Returns Distribution", fontsize=14)
                ax_monthly.set_xlabel("Monthly Return", fontsize=12)
                ax_monthly.set_ylabel("Frequency", fontsize=12)
                ax_monthly.grid(True, alpha=0.3)
            else:
                ax_monthly.text(0.5, 0.5, "Không đủ dữ liệu", ha='center', va='center', transform=ax_monthly.transAxes)
        else:
            ax_monthly.text(0.5, 0.5, "Không có dữ liệu returns", ha='center', va='center', transform=ax_monthly.transAxes)
        
        # Điều chỉnh khoảng cách giữa các subplot
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            try:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Đã lưu bảng điều khiển hiệu suất tại {save_path}")
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu bảng điều khiển hiệu suất: {e}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_equity_curve(
        self,
        include_benchmark: bool = True,
        include_trades: bool = True,
        log_scale: bool = False,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Vẽ biểu đồ đường cong lợi nhuận (equity curve).
        
        Args:
            include_benchmark: Bao gồm đường benchmark nếu có
            include_trades: Hiển thị các điểm giao dịch nếu có
            log_scale: Sử dụng thang đo logarit cho trục y
            figsize: Kích thước biểu đồ, nếu None sẽ sử dụng giá trị mặc định
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Figure của matplotlib
        """
        if self.equity_curve is None:
            self.logger.error("Không có dữ liệu equity_curve để vẽ biểu đồ")
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center')
            return fig
        
        fig, ax = plt.subplots(figsize=figsize or self.figsize, dpi=self.dpi)
        
        # Vẽ đường cong giá trị tài sản
        ax.plot(self.equity_curve.index, self.equity_curve, 
                label="Equity", color=COLOR_PALETTE['equity'], linewidth=2)
        
        # Vẽ benchmark nếu có
        if include_benchmark and self.benchmark is not None:
            # Điều chỉnh giá trị ban đầu của benchmark để so sánh
            if not self.benchmark.empty and not self.equity_curve.empty:
                # Lấy ngày bắt đầu chung
                start_date = max(self.benchmark.index[0], self.equity_curve.index[0])
                
                # Điều chỉnh benchmark
                benchmark_adjusted = self.benchmark.copy()
                initial_equity = self.equity_curve.loc[self.equity_curve.index >= start_date].iloc[0]
                initial_benchmark = self.benchmark.loc[self.benchmark.index >= start_date].iloc[0]
                
                if initial_benchmark != 0:
                    scale_factor = initial_equity / initial_benchmark
                    benchmark_adjusted = benchmark_adjusted * scale_factor
                
                ax.plot(benchmark_adjusted.index, benchmark_adjusted, 
                       label="Benchmark", color=COLOR_PALETTE['benchmark'], 
                       linewidth=1.5, linestyle='--', alpha=0.8)
        
        # Vẽ các điểm giao dịch
        if include_trades and self.trades is not None and not self.trades.empty:
            if 'entry_time' in self.trades.columns and 'exit_time' in self.trades.columns and 'profit' in self.trades.columns:
                # Giao dịch có lãi
                profitable_trades = self.trades[self.trades['profit'] > 0]
                if not profitable_trades.empty:
                    ax.scatter(profitable_trades['exit_time'], 
                              self.equity_curve.reindex(profitable_trades['exit_time']),
                              marker='^', s=50, color=COLOR_PALETTE['positive'], alpha=0.7, 
                              label="Profitable Trade")
                
                # Giao dịch lỗ
                losing_trades = self.trades[self.trades['profit'] < 0]
                if not losing_trades.empty:
                    ax.scatter(losing_trades['exit_time'], 
                              self.equity_curve.reindex(losing_trades['exit_time']),
                              marker='v', s=50, color=COLOR_PALETTE['negative'], alpha=0.7, 
                              label="Losing Trade")
        
        # Định dạng trục
        self._format_date_axis(ax)
        
        # Sử dụng thang đo logarit nếu cần
        if log_scale:
            ax.set_yscale('log')
        
        # Tùy chỉnh biểu đồ
        ax.set_title("Equity Curve", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Equity", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Tối ưu hóa không gian hiển thị
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            try:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Đã lưu biểu đồ equity curve tại {save_path}")
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu biểu đồ equity curve: {e}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_drawdown(
        self,
        underwater: bool = True,
        highlight_max: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Vẽ biểu đồ drawdown.
        
        Args:
            underwater: Vẽ biểu đồ underwater (drawdown âm) hoặc drawdown dương
            highlight_max: Đánh dấu drawdown tối đa
            figsize: Kích thước biểu đồ, nếu None sẽ sử dụng giá trị mặc định
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Figure của matplotlib
        """
        if self.equity_curve is None:
            self.logger.error("Không có dữ liệu equity_curve để vẽ biểu đồ drawdown")
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center')
            return fig
        
        fig, ax = plt.subplots(figsize=figsize or self.figsize, dpi=self.dpi)
        
        # Tính toán drawdown
        rolling_max = self.equity_curve.cummax()
        drawdown_series = (self.equity_curve - rolling_max) / rolling_max
        
        if underwater:
            # Vẽ biểu đồ underwater (drawdown âm)
            ax.fill_between(drawdown_series.index, 0, drawdown_series, 
                           color=COLOR_PALETTE['drawdown'], alpha=0.3)
            ax.plot(drawdown_series.index, drawdown_series, 
                   color=COLOR_PALETTE['drawdown'], linewidth=1, alpha=0.5)
        else:
            # Vẽ drawdown dương
            drawdown_series_positive = -drawdown_series
            ax.fill_between(drawdown_series_positive.index, 0, drawdown_series_positive, 
                           color=COLOR_PALETTE['drawdown'], alpha=0.3)
            ax.plot(drawdown_series_positive.index, drawdown_series_positive, 
                   color=COLOR_PALETTE['drawdown'], linewidth=1, alpha=0.5)
        
        # Đánh dấu drawdown tối đa
        if highlight_max:
            max_drawdown_idx = drawdown_series.idxmin()
            max_drawdown = drawdown_series.min()
            
            ax.scatter(max_drawdown_idx, max_drawdown, 
                      s=100, color=COLOR_PALETTE['highlight'], zorder=5,
                      marker='o', label=f"Max Drawdown: {max_drawdown:.2%}")
        
        # Thêm các ngưỡng tham chiếu
        if underwater:
            ax.axhline(y=-0.1, color='orange', linestyle='--', alpha=0.5, label="-10%")
            ax.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5, label="-20%")
            ax.axhline(y=-0.3, color='darkred', linestyle='--', alpha=0.5, label="-30%")
        else:
            ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label="10%")
            ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label="20%")
            ax.axhline(y=0.3, color='darkred', linestyle='--', alpha=0.5, label="30%")
        
        # Định dạng trục
        self._format_date_axis(ax)
        self._format_pct_axis(ax)
        
        # Tùy chỉnh biểu đồ
        if underwater:
            ax.set_title("Drawdown Underwater Chart", fontsize=16)
            ax.set_ylabel("Drawdown", fontsize=12)
        else:
            ax.set_title("Drawdown Chart", fontsize=16)
            ax.set_ylabel("Drawdown (%)", fontsize=12)
            
        ax.set_xlabel("Date", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        # Tối ưu hóa không gian hiển thị
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            try:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Đã lưu biểu đồ drawdown tại {save_path}")
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu biểu đồ drawdown: {e}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_monthly_returns(
        self,
        heatmap: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        cmap: str = 'RdYlGn'
    ) -> plt.Figure:
        """
        Vẽ biểu đồ lợi nhuận theo tháng.
        
        Args:
            heatmap: Vẽ dạng heatmap hoặc barplot
            figsize: Kích thước biểu đồ, nếu None sẽ sử dụng giá trị mặc định
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            cmap: Bảng màu cho heatmap
            
        Returns:
            Figure của matplotlib
        """
        if self.returns is None:
            self.logger.error("Không có dữ liệu returns để vẽ biểu đồ lợi nhuận theo tháng")
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center')
            return fig
        
        # Định dạng index thành datetime nếu chưa phải
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            try:
                self.returns.index = pd.to_datetime(self.returns.index)
            except Exception as e:
                self.logger.error(f"Không thể chuyển đổi index thành datetime: {e}")
                fig, ax = plt.subplots(figsize=figsize or self.figsize)
                ax.text(0.5, 0.5, "Lỗi định dạng dữ liệu", ha='center', va='center')
                return fig
        
        # Tính toán lợi nhuận theo tháng
        monthly_returns = self.returns.groupby([
            lambda x: x.year,
            lambda x: x.month
        ]).apply(lambda x: (1 + x).prod() - 1)
        
        # Chuyển đổi thành DataFrame với các cột là tháng và các hàng là năm
        monthly_returns_table = monthly_returns.unstack()
        
        # Điền giá trị NaN bằng 0
        monthly_returns_table = monthly_returns_table.fillna(0)
        
        # Đổi tên cột thành tên tháng
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_returns_table.columns = month_names[:len(monthly_returns_table.columns)]
        
        fig, ax = plt.subplots(figsize=figsize or self.figsize, dpi=self.dpi)
        
        if heatmap:
            # Sử dụng seaborn để vẽ heatmap
            sns.heatmap(monthly_returns_table, annot=True, fmt='.2%',
                       cmap=cmap, center=0, linewidths=1, ax=ax,
                       cbar_kws={'label': 'Monthly Return'})
            
            # Tùy chỉnh biểu đồ
            ax.set_title("Monthly Returns Heatmap", fontsize=16)
        else:
            # Tạo dữ liệu cho barplot
            years = monthly_returns_table.index.tolist()
            data = []
            
            for year in years:
                for i, month in enumerate(month_names[:len(monthly_returns_table.columns)]):
                    data.append({
                        'Year': year,
                        'Month': month,
                        'Return': monthly_returns_table.loc[year, month]
                    })
            
            df_plot = pd.DataFrame(data)
            
            # Vẽ barplot
            sns.barplot(
                x='Month', y='Return', hue='Year', data=df_plot, ax=ax,
                palette='Set2'
            )
            
            # Định dạng trục y thành phần trăm
            self._format_pct_axis(ax)
            
            # Tùy chỉnh biểu đồ
            ax.set_title("Monthly Returns by Year", fontsize=16)
            ax.set_xlabel("Month", fontsize=12)
            ax.set_ylabel("Return", fontsize=12)
            ax.legend(title="Year", loc='upper right')
        
        # Tối ưu hóa không gian hiển thị
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            try:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Đã lưu biểu đồ lợi nhuận theo tháng tại {save_path}")
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu biểu đồ lợi nhuận theo tháng: {e}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_returns_distribution(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        bins: int = 50,
        kde: bool = True,
        rug: bool = False,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Vẽ biểu đồ phân phối lợi nhuận.
        
        Args:
            figsize: Kích thước biểu đồ, nếu None sẽ sử dụng giá trị mặc định
            bins: Số lượng bins cho histogram
            kde: Hiển thị đường KDE (Kernel Density Estimation)
            rug: Hiển thị rug plot
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Figure của matplotlib
        """
        if self.returns is None:
            self.logger.error("Không có dữ liệu returns để vẽ biểu đồ phân phối")
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center')
            return fig
        
        fig, ax = plt.subplots(figsize=figsize or self.figsize, dpi=self.dpi)
        
        # Lọc bỏ các giá trị NaN
        returns_data = self.returns.dropna()
        
        # Vẽ histogram với KDE
        sns.histplot(returns_data, bins=bins, kde=kde, color=COLOR_PALETTE['equity'],
                    ax=ax, stat='density', alpha=0.7)
        
        # Thêm rug plot nếu cần
        if rug:
            sns.rugplot(returns_data, color=COLOR_PALETTE['equity'], alpha=0.3, ax=ax)
        
        # Vẽ đường vertical tại giá trị 0
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Nếu có PerformanceMetrics, thì lấy thêm một số thống kê
        if self.performance_metrics is not None:
            try:
                skewness = self.performance_metrics.skewness()
                kurtosis = self.performance_metrics.kurtosis()
                
                # Thêm chú thích thống kê
                stats_text = (
                    f"Mean: {returns_data.mean():.2%}\n"
                    f"Median: {returns_data.median():.2%}\n"
                    f"Std Dev: {returns_data.std():.2%}\n"
                    f"Skewness: {skewness:.2f}\n"
                    f"Kurtosis: {kurtosis:.2f}"
                )
                
                # Đặt vị trí chú thích tùy thuộc vào sự phân bố dữ liệu
                if skewness < 0:
                    x_pos, y_pos = 0.05, 0.95
                    va, ha = 'top', 'left'
                else:
                    x_pos, y_pos = 0.95, 0.95
                    va, ha = 'top', 'right'
                
                ax.text(x_pos, y_pos, stats_text, transform=ax.transAxes,
                       verticalalignment=va, horizontalalignment=ha,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            except Exception as e:
                self.logger.warning(f"Không thể thêm thống kê phân phối: {e}")
        
        # Định dạng trục x thành phần trăm
        self._format_pct_axis(ax, axis='x')
        
        # Tùy chỉnh biểu đồ
        ax.set_title("Returns Distribution", fontsize=16)
        ax.set_xlabel("Return", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Thêm các đường đánh dấu phân vị
        quantiles = returns_data.quantile([0.025, 0.975])
        ax.axvline(x=quantiles.iloc[0], color=COLOR_PALETTE['negative'], 
                  linestyle='--', alpha=0.7, label=f"2.5% VaR: {quantiles.iloc[0]:.2%}")
        ax.axvline(x=quantiles.iloc[1], color=COLOR_PALETTE['positive'], 
                  linestyle='--', alpha=0.7, label=f"97.5% VaR: {quantiles.iloc[1]:.2%}")
        
        ax.legend(loc='upper left')
        
        # Tối ưu hóa không gian hiển thị
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            try:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Đã lưu biểu đồ phân phối lợi nhuận tại {save_path}")
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu biểu đồ phân phối lợi nhuận: {e}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_rolling_metrics(
        self,
        metrics: List[str] = ['volatility', 'sharpe', 'drawdown'],
        window: int = 30,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Vẽ biểu đồ chỉ số theo cửa sổ trượt.
        
        Args:
            metrics: Danh sách các chỉ số cần vẽ
            window: Kích thước cửa sổ trượt (số ngày)
            figsize: Kích thước biểu đồ, nếu None sẽ sử dụng giá trị mặc định
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Figure của matplotlib
        """
        if self.returns is None:
            self.logger.error("Không có dữ liệu returns để vẽ biểu đồ chỉ số trượt")
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center')
            return fig
        
        # Giới hạn danh sách chỉ số
        valid_metrics = ['volatility', 'sharpe', 'sortino', 'drawdown', 'calmar', 'omega']
        metrics = [m for m in metrics if m in valid_metrics]
        
        if not metrics:
            self.logger.error(f"Không có chỉ số hợp lệ trong danh sách. Chỉ số hợp lệ: {valid_metrics}")
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Không có chỉ số hợp lệ", ha='center', va='center')
            return fig
        
        # Tính số hàng và cột cho subplots
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (self.figsize[0], self.figsize[1] * n_rows / 2),
                                dpi=self.dpi, squeeze=False)
        
        # Làm phẳng mảng axes để dễ truy cập
        axes = axes.flatten()
        
        # Vẽ từng chỉ số
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            if metric == 'volatility':
                # Tính volatility trượt
                rolling_vol = self.returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                ax.plot(rolling_vol.index, rolling_vol, color=COLOR_PALETTE['secondary'], linewidth=1.5)
                ax.set_title(f"Rolling Volatility ({window}d)", fontsize=14)
                ax.set_ylabel("Annualized Volatility", fontsize=12)
                self._format_pct_axis(ax)
                
                # Thêm các ngưỡng tham chiếu
                ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label="10%")
                ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label="20%")
                ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label="30%")
                
            elif metric == 'sharpe':
                # Tính Sharpe ratio trượt
                rolling_return = self.returns.rolling(window=window).mean() * TRADING_DAYS_PER_YEAR
                rolling_vol = self.returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                rolling_sharpe = (rolling_return - self.risk_free_rate) / rolling_vol
                
                ax.plot(rolling_sharpe.index, rolling_sharpe, color=COLOR_PALETTE['equity'], linewidth=1.5)
                ax.set_title(f"Rolling Sharpe Ratio ({window}d)", fontsize=14)
                ax.set_ylabel("Sharpe Ratio", fontsize=12)
                
                # Thêm các ngưỡng tham chiếu
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label="0")
                ax.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label="1")
                ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, label="2")
                
            elif metric == 'sortino':
                # Tính Sortino ratio trượt
                rolling_return = self.returns.rolling(window=window).mean() * TRADING_DAYS_PER_YEAR
                
                # Tính downside deviation
                downside_returns = self.returns.copy()
                downside_returns[downside_returns > 0] = 0
                rolling_downside_vol = downside_returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                
                # Xử lý trường hợp chia cho 0
                rolling_downside_vol = rolling_downside_vol.replace(0, np.nan)
                rolling_sortino = (rolling_return - self.risk_free_rate) / rolling_downside_vol
                
                ax.plot(rolling_sortino.index, rolling_sortino, color=COLOR_PALETTE['highlight'], linewidth=1.5)
                ax.set_title(f"Rolling Sortino Ratio ({window}d)", fontsize=14)
                ax.set_ylabel("Sortino Ratio", fontsize=12)
                
                # Thêm các ngưỡng tham chiếu
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label="0")
                ax.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label="1")
                ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, label="2")
                
            elif metric == 'drawdown':
                # Tính drawdown trượt
                equity_window = self.equity_curve.rolling(window=window)
                rolling_max = equity_window.max()
                rolling_drawdown = (self.equity_curve - rolling_max) / rolling_max
                
                ax.fill_between(rolling_drawdown.index, 0, rolling_drawdown, 
                               color=COLOR_PALETTE['drawdown'], alpha=0.3)
                ax.plot(rolling_drawdown.index, rolling_drawdown, 
                       color=COLOR_PALETTE['drawdown'], linewidth=1, alpha=0.7)
                
                ax.set_title(f"Rolling Drawdown ({window}d)", fontsize=14)
                ax.set_ylabel("Drawdown", fontsize=12)
                self._format_pct_axis(ax)
                
                # Thêm các ngưỡng tham chiếu
                ax.axhline(y=-0.1, color='orange', linestyle='--', alpha=0.5, label="-10%")
                ax.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5, label="-20%")
                
            elif metric == 'calmar':
                # Tính Calmar ratio trượt
                rolling_return = self.returns.rolling(window=window).mean() * TRADING_DAYS_PER_YEAR
                
                # Tính max drawdown trong cửa sổ
                equity_window = self.equity_curve.rolling(window=window)
                rolling_max = equity_window.max()
                rolling_drawdown = (self.equity_curve - rolling_max) / rolling_max
                rolling_max_drawdown = rolling_drawdown.rolling(window=window).min().abs()
                
                # Xử lý trường hợp chia cho 0
                rolling_max_drawdown = rolling_max_drawdown.replace(0, np.nan)
                rolling_calmar = rolling_return / rolling_max_drawdown
                
                ax.plot(rolling_calmar.index, rolling_calmar, color=COLOR_PALETTE['info'], linewidth=1.5)
                ax.set_title(f"Rolling Calmar Ratio ({window}d)", fontsize=14)
                ax.set_ylabel("Calmar Ratio", fontsize=12)
                
                # Thêm các ngưỡng tham chiếu
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label="0")
                ax.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label="1")
                ax.axhline(y=3, color='green', linestyle='--', alpha=0.5, label="3")
                
            elif metric == 'omega':
                # Tính Omega ratio trượt
                threshold = 0  # Có thể điều chỉnh ngưỡng này
                
                def rolling_omega(returns, window, threshold):
                    result = pd.Series(index=returns.index)
                    
                    for i in range(window - 1, len(returns)):
                        window_returns = returns.iloc[i - window + 1:i + 1]
                        gains = window_returns[window_returns > threshold] - threshold
                        losses = threshold - window_returns[window_returns < threshold]
                        
                        if losses.sum() == 0:
                            result.iloc[i] = np.nan
                        else:
                            result.iloc[i] = gains.sum() / losses.sum()
                    
                    return result
                
                rolling_omega_ratio = rolling_omega(self.returns, window, threshold)
                
                ax.plot(rolling_omega_ratio.index, rolling_omega_ratio, color=COLOR_PALETTE['accent'], linewidth=1.5)
                ax.set_title(f"Rolling Omega Ratio ({window}d)", fontsize=14)
                ax.set_ylabel("Omega Ratio", fontsize=12)
                
                # Thêm ngưỡng tham chiếu
                ax.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label="1")
            
            # Định dạng trục
            self._format_date_axis(ax)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')
        
        # Ẩn các trục không sử dụng
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        # Tối ưu hóa không gian hiển thị
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            try:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Đã lưu biểu đồ chỉ số trượt tại {save_path}")
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu biểu đồ chỉ số trượt: {e}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_metrics_summary(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Vẽ biểu đồ tóm tắt các chỉ số hiệu suất quan trọng.
        
        Args:
            figsize: Kích thước biểu đồ, nếu None sẽ sử dụng giá trị mặc định
            metrics: Danh sách các chỉ số cần hiển thị, nếu None sẽ hiển thị các chỉ số mặc định
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Figure của matplotlib
        """
        if self.equity_curve is None:
            self.logger.error("Không có dữ liệu equity_curve để vẽ biểu đồ tóm tắt")
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center')
            return fig
        
        # Danh sách chỉ số mặc định
        if metrics is None:
            metrics = {
                'Total Return': None,                         # Tổng lợi nhuận
                'Annualized Return': None,                    # Lợi nhuận hàng năm
                'Sharpe Ratio': None,                         # Sharpe Ratio
                'Sortino Ratio': None,                        # Sortino Ratio
                'Max Drawdown': None,                         # Drawdown tối đa
                'Win Rate': None,                             # Tỷ lệ thắng
                'Profit Factor': None,                        # Hệ số lợi nhuận
                'Volatility': None,                           # Biến động
                'Calmar Ratio': None,                         # Calmar Ratio
                'Avg. Trade': None                            # Lợi nhuận trung bình mỗi giao dịch
            }
        
        # Tính toán các chỉ số
        if self.performance_metrics is not None:
            try:
                metrics['Total Return'] = self.performance_metrics.total_return()
                metrics['Annualized Return'] = self.performance_metrics.annualized_return()
                metrics['Sharpe Ratio'] = self.performance_metrics.sharpe_ratio()
                metrics['Sortino Ratio'] = self.performance_metrics.sortino_ratio()
                metrics['Max Drawdown'] = self.performance_metrics.max_drawdown()
                metrics['Volatility'] = self.performance_metrics.volatility()
                metrics['Calmar Ratio'] = self.performance_metrics.calmar_ratio()
                
                if self.trades is not None and len(self.trades) > 0:
                    metrics['Win Rate'] = len(self.trades[self.trades['profit'] > 0]) / len(self.trades)
                    
                    profit_sum = self.trades[self.trades['profit'] > 0]['profit'].sum()
                    loss_sum = abs(self.trades[self.trades['profit'] < 0]['profit'].sum())
                    if loss_sum > 0:
                        metrics['Profit Factor'] = profit_sum / loss_sum
                    else:
                        metrics['Profit Factor'] = float('inf')
                    
                    metrics['Avg. Trade'] = self.trades['profit'].mean()
            except Exception as e:
                self.logger.warning(f"Không thể tính toán một số chỉ số: {e}")
        
        # Nếu không có PerformanceMetrics, tính toán thủ công
        else:
            try:
                # Tính Total Return
                if self.equity_curve is not None and len(self.equity_curve) > 1:
                    metrics['Total Return'] = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
                
                # Tính Annualized Return
                if self.equity_curve is not None and len(self.equity_curve) > 1:
                    start_date = self.equity_curve.index[0]
                    end_date = self.equity_curve.index[-1]
                    days = (end_date - start_date).days
                    if days > 0:
                        metrics['Annualized Return'] = ((1 + metrics['Total Return']) ** (365 / days)) - 1
                
                # Tính Volatility
                if self.returns is not None:
                    metrics['Volatility'] = self.returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                
                # Tính Max Drawdown
                if self.equity_curve is not None:
                    rolling_max = self.equity_curve.cummax()
                    drawdown = (self.equity_curve - rolling_max) / rolling_max
                    metrics['Max Drawdown'] = drawdown.min()
                
                # Tính Sharpe Ratio
                if metrics['Annualized Return'] is not None and metrics['Volatility'] is not None:
                    metrics['Sharpe Ratio'] = (metrics['Annualized Return'] - self.risk_free_rate) / metrics['Volatility']
                
                # Tính Calmar Ratio
                if metrics['Annualized Return'] is not None and metrics['Max Drawdown'] is not None and metrics['Max Drawdown'] < 0:
                    metrics['Calmar Ratio'] = metrics['Annualized Return'] / abs(metrics['Max Drawdown'])
                
                # Tính các chỉ số liên quan đến giao dịch
                if self.trades is not None and len(self.trades) > 0:
                    metrics['Win Rate'] = len(self.trades[self.trades['profit'] > 0]) / len(self.trades)
                    
                    profit_sum = self.trades[self.trades['profit'] > 0]['profit'].sum()
                    loss_sum = abs(self.trades[self.trades['profit'] < 0]['profit'].sum())
                    if loss_sum > 0:
                        metrics['Profit Factor'] = profit_sum / loss_sum
                    else:
                        metrics['Profit Factor'] = float('inf')
                    
                    metrics['Avg. Trade'] = self.trades['profit'].mean()
            except Exception as e:
                self.logger.warning(f"Không thể tính toán một số chỉ số thủ công: {e}")
        
        # Loại bỏ các chỉ số None
        metrics = {k: v for k, v in metrics.items() if v is not None}
        
        # Tạo hình
        fig, ax = plt.subplots(figsize=figsize or (self.figsize[0], self.figsize[1] * 0.8), dpi=self.dpi)
        
        # Nếu không có chỉ số nào, hiển thị thông báo
        if not metrics:
            ax.text(0.5, 0.5, "Không có đủ dữ liệu để tính toán chỉ số", ha='center', va='center', transform=ax.transAxes)
            plt.tight_layout()
            
            if save_path:
                try:
                    plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                except Exception as e:
                    self.logger.error(f"Lỗi khi lưu biểu đồ tóm tắt: {e}")
            
            if show_plot:
                plt.show()
            
            return fig
        
        # Tạo bảng
        cell_text = []
        colors = []
        
        # Thêm các hàng vào bảng
        for metric_name, value in metrics.items():
            # Định dạng giá trị
            if metric_name in ['Total Return', 'Annualized Return', 'Max Drawdown', 'Win Rate']:
                formatted_value = f"{value:.2%}"
            elif metric_name in ['Volatility']:
                formatted_value = f"{value:.2%}"
            elif metric_name in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Profit Factor']:
                formatted_value = f"{value:.2f}"
            elif metric_name in ['Avg. Trade']:
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = f"{value}"
            
            cell_text.append([metric_name, formatted_value])
            
            # Màu sắc dựa trên giá trị
            if metric_name in ['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Sortino Ratio', 'Win Rate', 'Profit Factor', 'Calmar Ratio', 'Avg. Trade']:
                if value > 0:
                    colors.append(['#f0f0f0', '#d8f0d8'])  # Xanh nhạt cho giá trị dương
                else:
                    colors.append(['#f0f0f0', '#f0d8d8'])  # Đỏ nhạt cho giá trị âm
            elif metric_name in ['Max Drawdown', 'Volatility']:
                if abs(value) < 0.1:
                    colors.append(['#f0f0f0', '#d8f0d8'])  # Xanh nhạt cho giá trị thấp
                elif abs(value) < 0.2:
                    colors.append(['#f0f0f0', '#f0f0d8'])  # Vàng nhạt cho giá trị trung bình
                else:
                    colors.append(['#f0f0f0', '#f0d8d8'])  # Đỏ nhạt cho giá trị cao
            else:
                colors.append(['#f0f0f0', '#f0f0f0'])  # Màu mặc định
        
        # Vẽ bảng
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(
            cellText=cell_text,
            colLabels=["Metric", "Value"],
            colWidths=[0.6, 0.3],
            cellLoc='center',
            loc='center',
            cellColours=colors
        )
        
        # Định dạng bảng
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Đặt tiêu đề
        ax.set_title("Performance Metrics Summary", fontsize=16, pad=20)
        
        # Tối ưu hóa không gian hiển thị
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            try:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Đã lưu biểu đồ tóm tắt chỉ số tại {save_path}")
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu biểu đồ tóm tắt chỉ số: {e}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_trade_analysis(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Vẽ biểu đồ phân tích giao dịch.
        
        Args:
            figsize: Kích thước biểu đồ, nếu None sẽ sử dụng giá trị mặc định
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Figure của matplotlib
        """
        if self.trades is None:
            self.logger.error("Không có dữ liệu trades để vẽ biểu đồ phân tích giao dịch")
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center')
            return fig
        
        # Kiểm tra cấu trúc của DataFrame trades
        required_columns = ['entry_time', 'exit_time', 'profit']
        if not all(col in self.trades.columns for col in required_columns):
            self.logger.error(f"DataFrame trades thiếu một số cột cần thiết: {required_columns}")
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Dữ liệu trades không đúng định dạng", ha='center', va='center')
            return fig
        
        # Tạo figure với 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize or (self.figsize[0] * 1.5, self.figsize[1] * 1.5), dpi=self.dpi)
        
        # 1. Biểu đồ phân bố lợi nhuận giao dịch
        ax1 = axes[0, 0]
        sns.histplot(self.trades['profit'], bins=30, kde=True, ax=ax1,
                    color=COLOR_PALETTE['equity'], alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax1.set_title("Trade Profit Distribution", fontsize=14)
        ax1.set_xlabel("Profit", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. Biểu đồ lợi nhuận giao dịch theo thời gian
        ax2 = axes[0, 1]
        ax2.scatter(self.trades['exit_time'], self.trades['profit'],
                   color=[COLOR_PALETTE['positive'] if p > 0 else COLOR_PALETTE['negative'] for p in self.trades['profit']],
                   alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        self._format_date_axis(ax2)
        ax2.set_title("Trade Profit Over Time", fontsize=14)
        ax2.set_xlabel("Exit Time", fontsize=12)
        ax2.set_ylabel("Profit", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. Biểu đồ thống kê thắng/thua
        ax3 = axes[1, 0]
        
        # Tính thống kê
        win_count = len(self.trades[self.trades['profit'] > 0])
        loss_count = len(self.trades[self.trades['profit'] < 0])
        win_rate = win_count / len(self.trades) if len(self.trades) > 0 else 0
        
        avg_win = self.trades[self.trades['profit'] > 0]['profit'].mean() if win_count > 0 else 0
        avg_loss = self.trades[self.trades['profit'] < 0]['profit'].mean() if loss_count > 0 else 0
        
        # Vẽ barchart
        stats = pd.DataFrame({
            'Category': ['Win Rate', 'Loss Rate', 'Avg Win', 'Avg Loss'],
            'Value': [win_rate, 1 - win_rate, avg_win, avg_loss],
            'Color': [COLOR_PALETTE['positive'], COLOR_PALETTE['negative'], 
                     COLOR_PALETTE['positive'], COLOR_PALETTE['negative']]
        })
        
        sns.barplot(x='Category', y='Value', data=stats, palette=stats['Color'], ax=ax3)
        
        # Format các nhãn
        for i, v in enumerate(stats['Value']):
            if i < 2:  # Win Rate và Loss Rate
                text = f"{v:.1%}"
            else:  # Avg Win và Avg Loss
                text = f"{v:.2f}"
            ax3.text(i, v, text, ha='center', va='bottom')
        
        ax3.set_title("Win/Loss Statistics", fontsize=14)
        ax3.set_xlabel("")
        ax3.set_ylabel("Value", fontsize=12)
        ax3.grid(True, axis='y', alpha=0.3)
        
        # 4. Biểu đồ lợi nhuận tích lũy từ các giao dịch
        ax4 = axes[1, 1]
        
        # Tính lợi nhuận tích lũy
        self.trades = self.trades.sort_values('exit_time')
        self.trades['cumulative_profit'] = self.trades['profit'].cumsum()
        
        ax4.plot(self.trades['exit_time'], self.trades['cumulative_profit'],
                color=COLOR_PALETTE['equity'], linewidth=2)
        self._format_date_axis(ax4)
        ax4.set_title("Cumulative Profit", fontsize=14)
        ax4.set_xlabel("Exit Time", fontsize=12)
        ax4.set_ylabel("Cumulative Profit", fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Thêm thông tin chi tiết
        text = (
            f"Total Trades: {len(self.trades)}\n"
            f"Win Rate: {win_rate:.1%}\n"
            f"Profit Factor: {abs(self.trades[self.trades['profit'] > 0]['profit'].sum()) / abs(self.trades[self.trades['profit'] < 0]['profit'].sum()) if abs(self.trades[self.trades['profit'] < 0]['profit'].sum()) > 0 else float('inf'):.2f}\n"
            f"Avg Win: {avg_win:.2f}\n"
            f"Avg Loss: {avg_loss:.2f}\n"
            f"Max Win: {self.trades['profit'].max():.2f}\n"
            f"Max Loss: {self.trades['profit'].min():.2f}"
        )
        
        ax4.text(0.05, 0.05, text, transform=ax4.transAxes,
               verticalalignment='bottom', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Tối ưu hóa không gian hiển thị
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            try:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Đã lưu biểu đồ phân tích giao dịch tại {save_path}")
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu biểu đồ phân tích giao dịch: {e}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_hold_time_analysis(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> plt.Figure:
        """
        Vẽ biểu đồ phân tích thời gian nắm giữ giao dịch.
        
        Args:
            figsize: Kích thước biểu đồ, nếu None sẽ sử dụng giá trị mặc định
            save_path: Đường dẫn để lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Figure của matplotlib
        """
        if self.trades is None:
            self.logger.error("Không có dữ liệu trades để vẽ biểu đồ phân tích thời gian nắm giữ")
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center')
            return fig
        
        # Kiểm tra cấu trúc của DataFrame trades
        required_columns = ['entry_time', 'exit_time', 'profit']
        if not all(col in self.trades.columns for col in required_columns):
            self.logger.error(f"DataFrame trades thiếu một số cột cần thiết: {required_columns}")
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            ax.text(0.5, 0.5, "Dữ liệu trades không đúng định dạng", ha='center', va='center')
            return fig
        
        # Tạo figure với 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize or (self.figsize[0] * 1.5, self.figsize[1] * 1.5), dpi=self.dpi)
        
        # Tính thời gian nắm giữ
        self.trades['hold_time'] = (pd.to_datetime(self.trades['exit_time']) - 
                                   pd.to_datetime(self.trades['entry_time']))
        
        # Convert timedelta thành giờ
        self.trades['hold_time_hours'] = self.trades['hold_time'].dt.total_seconds() / 3600
        
        # 1. Phân phối thời gian nắm giữ
        ax1 = axes[0, 0]
        sns.histplot(self.trades['hold_time_hours'], bins=30, kde=True, ax=ax1,
                    color=COLOR_PALETTE['equity'], alpha=0.7)
        
        # Thêm các thống kê về thời gian nắm giữ
        avg_hold = self.trades['hold_time_hours'].mean()
        median_hold = self.trades['hold_time_hours'].median()
        max_hold = self.trades['hold_time_hours'].max()
        min_hold = self.trades['hold_time_hours'].min()
        
        stats_text = (
            f"Avg: {avg_hold:.1f} hrs\n"
            f"Median: {median_hold:.1f} hrs\n"
            f"Min: {min_hold:.1f} hrs\n"
            f"Max: {max_hold:.1f} hrs"
        )
        
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax1.set_title("Hold Time Distribution (Hours)", fontsize=14)
        ax1.set_xlabel("Hold Time (Hours)", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. Thời gian nắm giữ vs lợi nhuận
        ax2 = axes[0, 1]
        ax2.scatter(self.trades['hold_time_hours'], self.trades['profit'],
                  color=[COLOR_PALETTE['positive'] if p > 0 else COLOR_PALETTE['negative'] for p in self.trades['profit']],
                  alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        
        # Thêm đường xu hướng
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                self.trades['hold_time_hours'], self.trades['profit'])
            
            x_line = np.linspace(self.trades['hold_time_hours'].min(), self.trades['hold_time_hours'].max(), 100)
            y_line = slope * x_line + intercept
            
            # Vẽ đường trendline
            ax2.plot(x_line, y_line, color='black', linestyle='-', alpha=0.7,
                   label=f'y = {slope:.4f}x + {intercept:.4f}, R²: {r_value**2:.3f}')
            ax2.legend(loc='upper right', fontsize=10)
        except Exception as e:
            self.logger.warning(f"Không thể vẽ đường xu hướng: {e}")
        
        ax2.set_title("Hold Time vs. Profit", fontsize=14)
        ax2.set_xlabel("Hold Time (Hours)", fontsize=12)
        ax2.set_ylabel("Profit", fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. Phân tích thời gian nắm giữ theo loại giao dịch (thắng/thua)
        ax3 = axes[1, 0]
        
        # Tạo DataFrame cho win/loss
        winning_trades = self.trades[self.trades['profit'] > 0]
        losing_trades = self.trades[self.trades['profit'] < 0]
        
        # Tính thống kê
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            # Vẽ boxplot
            trade_types = []
            hold_times = []
            
            for _, row in winning_trades.iterrows():
                trade_types.append('Winning')
                hold_times.append(row['hold_time_hours'])
                
            for _, row in losing_trades.iterrows():
                trade_types.append('Losing')
                hold_times.append(row['hold_time_hours'])
                
            box_data = pd.DataFrame({
                'Trade Type': trade_types,
                'Hold Time (Hours)': hold_times
            })
            
            sns.boxplot(x='Trade Type', y='Hold Time (Hours)', data=box_data, ax=ax3, 
                       palette=[COLOR_PALETTE['positive'], COLOR_PALETTE['negative']])
            
            # Thêm violin plot
            sns.stripplot(x='Trade Type', y='Hold Time (Hours)', data=box_data, ax=ax3,
                         size=4, color='black', alpha=0.3)
            
            # Thêm thống kê
            avg_win_time = winning_trades['hold_time_hours'].mean()
            avg_loss_time = losing_trades['hold_time_hours'].mean()
            
            text = (
                f"Winning Avg: {avg_win_time:.1f} hrs\n"
                f"Losing Avg: {avg_loss_time:.1f} hrs\n"
                f"Ratio: {avg_win_time / avg_loss_time if avg_loss_time > 0 else float('inf'):.2f}x"
            )
            
            ax3.text(0.05, 0.95, text, transform=ax3.transAxes,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax3.text(0.5, 0.5, "Không đủ dữ liệu", ha='center', va='center', transform=ax3.transAxes)
        
        ax3.set_title("Hold Time by Trade Outcome", fontsize=14)
        ax3.set_ylabel("Hold Time (Hours)", fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. Phân tích thời gian nắm giữ theo thời gian
        ax4 = axes[1, 1]
        
        ax4.scatter(self.trades['exit_time'], self.trades['hold_time_hours'],
                  color=[COLOR_PALETTE['positive'] if p > 0 else COLOR_PALETTE['negative'] for p in self.trades['profit']],
                  alpha=0.7)
        
        # Thêm đường xu hướng
        try:
            from scipy import stats
            
            # Chuyển đổi timestamp sang số
            exit_times_num = mdates.date2num(pd.to_datetime(self.trades['exit_time']))
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                exit_times_num, self.trades['hold_time_hours'])
            
            x_line = np.linspace(min(exit_times_num), max(exit_times_num), 100)
            y_line = slope * x_line + intercept
            
            # Chuyển số ngược lại thành timestamp cho việc vẽ
            x_dates = mdates.num2date(x_line)
            
            # Vẽ đường trendline
            ax4.plot(x_dates, y_line, color='black', linestyle='-', alpha=0.7,
                   label=f'Trend: {"↑" if slope > 0 else "↓"} R²: {r_value**2:.3f}')
            ax4.legend(loc='upper right', fontsize=10)
        except Exception as e:
            self.logger.warning(f"Không thể vẽ đường xu hướng cho thời gian: {e}")
        
        self._format_date_axis(ax4)
        ax4.set_title("Hold Time Over Calendar Time", fontsize=14)
        ax4.set_xlabel("Exit Time", fontsize=12)
        ax4.set_ylabel("Hold Time (Hours)", fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Tối ưu hóa không gian hiển thị
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            try:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Đã lưu biểu đồ phân tích thời gian nắm giữ tại {save_path}")
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu biểu đồ phân tích thời gian nắm giữ: {e}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        
        return fig