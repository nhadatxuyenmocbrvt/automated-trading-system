#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module biểu đồ hiệu suất cho dashboard Streamlit.

Module này cung cấp các lớp và hàm để tạo biểu đồ trực quan hóa
hiệu suất giao dịch, giúp người dùng theo dõi và đánh giá
hiệu quả của chiến lược giao dịch.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
import os

# Import các module từ hệ thống
from config.constants import OrderStatus, PositionSide, OrderType, BacktestMetric
from logs.metrics.trading_metrics import TradingMetricsTracker, MultiSymbolTradingMetricsTracker
from backtesting.performance_metrics import PerformanceMetrics


class PerformanceCharts:
    """
    Lớp tạo và quản lý các biểu đồ hiệu suất cho dashboard Streamlit.
    
    Cung cấp nhiều loại biểu đồ trực quan hóa khác nhau cho
    hiệu suất giao dịch, bao gồm đường cong vốn, phân phối lợi nhuận,
    phân tích drawdown, so sánh hiệu suất, và nhiều loại khác.
    """
    
    def __init__(
        self,
        metrics_data: Optional[Union[Dict, pd.DataFrame, TradingMetricsTracker, MultiSymbolTradingMetricsTracker]] = None,
        symbol: Optional[str] = None,
        strategy_name: Optional[str] = None,
        theme: str = "streamlit",
        height: int = 500,
        width: int = 800
    ):
        """
        Khởi tạo đối tượng PerformanceCharts.
        
        Args:
            metrics_data: Dữ liệu số liệu (Dict, DataFrame, hoặc TradingMetricsTracker)
            symbol: Cặp giao dịch (nếu có)
            strategy_name: Tên chiến lược (nếu có)
            theme: Theme màu cho biểu đồ ('streamlit', 'plotly', 'dark', 'light')
            height: Chiều cao mặc định cho biểu đồ
            width: Chiều rộng mặc định cho biểu đồ
        """
        self.metrics_data = metrics_data
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.theme = theme
        self.height = height
        self.width = width
        
        # Thiết lập theme và style
        self._setup_theme()
        
        # Khởi tạo các DataFrame cho hiển thị biểu đồ
        self.equity_df = None
        self.trades_df = None
        self.drawdown_df = None
        self.returns_df = None
        self.portfolio_df = None
        
        # Xử lý dữ liệu metrics
        if metrics_data is not None:
            self._process_metrics_data()
    
    def _setup_theme(self):
        """
        Thiết lập theme và style cho biểu đồ.
        """
        if self.theme == "dark":
            plt.style.use('dark_background')
            self.colors = {
                'primary': '#3366FF',
                'secondary': '#FF6B6B',
                'positive': '#4CAF50',
                'negative': '#F44336',
                'neutral': '#9E9E9E',
                'background': '#121212',
                'text': '#FFFFFF'
            }
        else:  # light theme
            plt.style.use('default')
            self.colors = {
                'primary': '#1F77B4',
                'secondary': '#FF7F0E',
                'positive': '#2E7D32',
                'negative': '#C62828',
                'neutral': '#607D8B',
                'background': '#FFFFFF',
                'text': '#000000'
            }
        
        # Thiết lập style cho seaborn
        sns.set(style="whitegrid" if self.theme != "dark" else "darkgrid")
    
    def _process_metrics_data(self):
        """
        Xử lý dữ liệu metrics và chuyển đổi thành các DataFrame cần thiết.
        """
        if isinstance(self.metrics_data, TradingMetricsTracker):
            # Lấy dữ liệu từ TradingMetricsTracker
            self._process_trading_metrics_tracker()
        
        elif isinstance(self.metrics_data, MultiSymbolTradingMetricsTracker):
            # Lấy dữ liệu từ MultiSymbolTradingMetricsTracker
            self._process_multi_symbol_tracker()
        
        elif isinstance(self.metrics_data, dict):
            # Lấy dữ liệu từ Dict (giả định là kết quả từ save_metrics)
            self._process_metrics_dict()
        
        elif isinstance(self.metrics_data, pd.DataFrame):
            # Đã là DataFrame, giả định là equity curve
            self.equity_df = self.metrics_data
        
        elif isinstance(self.metrics_data, str) or isinstance(self.metrics_data, Path):
            # Đường dẫn tới file metrics json
            self._load_metrics_from_file(self.metrics_data)
    
    def _process_trading_metrics_tracker(self):
        """
        Xử lý dữ liệu từ đối tượng TradingMetricsTracker.
        """
        tracker = self.metrics_data
        
        # Lấy thông tin symbol và strategy nếu chưa có
        if self.symbol is None:
            self.symbol = tracker.symbol
        
        if self.strategy_name is None:
            self.strategy_name = tracker.strategy_name
        
        # Xử lý dữ liệu vốn
        if tracker.metrics_history["capital_history"]:
            equity_data = []
            for entry in tracker.metrics_history["capital_history"]:
                equity_data.append({
                    "timestamp": entry["timestamp"],
                    "capital": entry["capital"],
                    "event": entry.get("event", "update")
                })
            
            if equity_data:
                self.equity_df = pd.DataFrame(equity_data)
                self.equity_df["timestamp"] = pd.to_datetime(self.equity_df["timestamp"])
                self.equity_df.set_index("timestamp", inplace=True)
                self.equity_df.sort_index(inplace=True)
                self.equity_df = self.equity_df[~self.equity_df.index.duplicated(keep='last')]
        
        # Xử lý dữ liệu giao dịch
        if tracker.metrics_history["profit_loss_history"]:
            trades_data = []
            for trade in tracker.metrics_history["profit_loss_history"]:
                # Tìm thông tin thời gian giao dịch
                duration_info = next((d for d in tracker.metrics_history["trade_durations"] 
                                if d["position_id"] == trade["position_id"]), None)
                
                trade_info = {
                    "timestamp": trade["timestamp"],
                    "position_id": trade["position_id"],
                    "profit_loss": trade["profit_loss"],
                    "profit_loss_percent": trade.get("profit_loss_percent", 0),
                    "is_win": any(w["position_id"] == trade["position_id"] and w["is_win"] 
                                for w in tracker.metrics_history["win_history"]),
                    "duration": duration_info["duration"] if duration_info else 0,
                    "reason": trade.get("reason", "manual")
                }
                trades_data.append(trade_info)
            
            if trades_data:
                self.trades_df = pd.DataFrame(trades_data)
                self.trades_df["timestamp"] = pd.to_datetime(self.trades_df["timestamp"])
                self.trades_df.sort_values("timestamp", inplace=True)
        
        # Xử lý dữ liệu drawdown
        if tracker.metrics_history["drawdowns"]:
            self.drawdown_df = pd.DataFrame(tracker.metrics_history["drawdowns"])
            for time_col in ["peak_time", "trough_time", "recovery_time"]:
                if time_col in self.drawdown_df.columns:
                    self.drawdown_df[time_col] = pd.to_datetime(self.drawdown_df[time_col])
        
        # Tính toán returns từ equity curve
        if self.equity_df is not None and len(self.equity_df) > 1:
            self.returns_df = pd.DataFrame({
                "daily_return": self.equity_df["capital"].pct_change()
            })
            self.returns_df.dropna(inplace=True)
    
    def _process_multi_symbol_tracker(self):
        """
        Xử lý dữ liệu từ đối tượng MultiSymbolTradingMetricsTracker.
        """
        multi_tracker = self.metrics_data
        
        # Lấy thông tin strategy nếu chưa có
        if self.strategy_name is None:
            self.strategy_name = multi_tracker.strategy_name
        
        # Xử lý dữ liệu danh mục
        if multi_tracker.portfolio_history:
            portfolio_data = []
            for entry in multi_tracker.portfolio_history:
                data = {
                    "timestamp": entry["timestamp"],
                    "total_capital": entry["total_capital"]
                }
                # Thêm dữ liệu cho từng symbol
                for symbol, symbol_data in entry["symbols"].items():
                    data[f"{symbol}_capital"] = symbol_data["capital"]
                    data[f"{symbol}_allocation"] = symbol_data["allocation"]
                    data[f"{symbol}_actual_allocation"] = symbol_data["actual_allocation"]
                
                portfolio_data.append(data)
            
            if portfolio_data:
                self.portfolio_df = pd.DataFrame(portfolio_data)
                self.portfolio_df["timestamp"] = pd.to_datetime(self.portfolio_df["timestamp"])
                self.portfolio_df.set_index("timestamp", inplace=True)
                self.portfolio_df.sort_index(inplace=True)
                self.portfolio_df = self.portfolio_df[~self.portfolio_df.index.duplicated(keep='last')]
        
        # Sử dụng dữ liệu danh mục làm dữ liệu vốn
        if self.portfolio_df is not None:
            self.equity_df = pd.DataFrame({
                "capital": self.portfolio_df["total_capital"]
            })
        
        # Tính toán returns từ equity curve
        if self.equity_df is not None and len(self.equity_df) > 1:
            self.returns_df = pd.DataFrame({
                "daily_return": self.equity_df["capital"].pct_change()
            })
            self.returns_df.dropna(inplace=True)
    
    def _process_metrics_dict(self):
        """
        Xử lý dữ liệu từ Dict (giả định là kết quả từ save_metrics).
        """
        data = self.metrics_data
        
        # Lấy thông tin cơ bản
        if self.symbol is None and "symbol" in data:
            self.symbol = data["symbol"]
        
        if self.strategy_name is None and "strategy_name" in data:
            self.strategy_name = data["strategy_name"]
        
        # Xử lý dữ liệu vốn
        if "metrics_history" in data and "capital_history" in data["metrics_history"]:
            equity_data = data["metrics_history"]["capital_history"]
            if equity_data:
                self.equity_df = pd.DataFrame(equity_data)
                self.equity_df["timestamp"] = pd.to_datetime(self.equity_df["timestamp"])
                self.equity_df.set_index("timestamp", inplace=True)
                self.equity_df.sort_index(inplace=True)
                self.equity_df = self.equity_df[~self.equity_df.index.duplicated(keep='last')]
        
        # Xử lý dữ liệu giao dịch
        if "metrics_history" in data and "profit_loss_history" in data["metrics_history"]:
            trades_data = []
            for trade in data["metrics_history"]["profit_loss_history"]:
                # Tìm thông tin thời gian giao dịch nếu có
                duration_info = None
                if "trade_durations" in data["metrics_history"]:
                    duration_info = next((d for d in data["metrics_history"]["trade_durations"] 
                                    if d["position_id"] == trade["position_id"]), None)
                
                is_win = False
                if "win_history" in data["metrics_history"]:
                    is_win = any(w["position_id"] == trade["position_id"] and w["is_win"] 
                              for w in data["metrics_history"]["win_history"])
                
                trade_info = {
                    "timestamp": trade["timestamp"],
                    "position_id": trade["position_id"],
                    "profit_loss": trade["profit_loss"],
                    "profit_loss_percent": trade.get("profit_loss_percent", 0),
                    "is_win": is_win,
                    "duration": duration_info["duration"] if duration_info else 0,
                    "reason": trade.get("reason", "manual")
                }
                trades_data.append(trade_info)
            
            if trades_data:
                self.trades_df = pd.DataFrame(trades_data)
                self.trades_df["timestamp"] = pd.to_datetime(self.trades_df["timestamp"])
                self.trades_df.sort_values("timestamp", inplace=True)
        
        # Xử lý dữ liệu drawdown
        if "metrics_history" in data and "drawdowns" in data["metrics_history"]:
            self.drawdown_df = pd.DataFrame(data["metrics_history"]["drawdowns"])
            for time_col in ["peak_time", "trough_time", "recovery_time"]:
                if time_col in self.drawdown_df.columns:
                    self.drawdown_df[time_col] = pd.to_datetime(self.drawdown_df[time_col])
        
        # Xử lý dữ liệu danh mục (nếu có)
        if "portfolio_history" in data:
            self.portfolio_df = pd.DataFrame(data["portfolio_history"])
            self.portfolio_df["timestamp"] = pd.to_datetime(self.portfolio_df["timestamp"])
            self.portfolio_df.set_index("timestamp", inplace=True)
            self.portfolio_df.sort_index(inplace=True)
            self.portfolio_df = self.portfolio_df[~self.portfolio_df.index.duplicated(keep='last')]
        
        # Tính toán returns từ equity curve
        if self.equity_df is not None and len(self.equity_df) > 1:
            self.returns_df = pd.DataFrame({
                "daily_return": self.equity_df["capital"].pct_change()
            })
            self.returns_df.dropna(inplace=True)
    
    def _load_metrics_from_file(self, file_path: Union[str, Path]):
        """
        Tải dữ liệu metrics từ file JSON.
        
        Args:
            file_path: Đường dẫn tới file metrics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Xử lý dữ liệu
            self.metrics_data = data
            self._process_metrics_dict()
            
        except Exception as e:
            st.error(f"Không thể tải dữ liệu metrics từ file: {str(e)}")
    
    def load_data(
        self,
        metrics_data: Union[Dict, pd.DataFrame, TradingMetricsTracker, MultiSymbolTradingMetricsTracker, str, Path]
    ):
        """
        Tải dữ liệu metrics mới.
        
        Args:
            metrics_data: Dữ liệu metrics mới
        """
        self.metrics_data = metrics_data
        self._process_metrics_data()
    
    def plot_equity_curve(
        self,
        use_plotly: bool = True,
        include_trades: bool = True,
        include_drawdown: bool = True,
        display_metrics: bool = True,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Vẽ đường cong vốn (equity curve).
        
        Args:
            use_plotly: Sử dụng Plotly thay vì Matplotlib
            include_trades: Hiển thị các giao dịch trên biểu đồ
            include_drawdown: Hiển thị drawdown
            display_metrics: Hiển thị các chỉ số performance
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            height: Chiều cao của biểu đồ
            width: Chiều rộng của biểu đồ
            
        Returns:
            Đối tượng go.Figure của Plotly hoặc plt.Figure của Matplotlib
        """
        if self.equity_df is None or len(self.equity_df) < 2:
            st.warning("Không có đủ dữ liệu để vẽ đường cong vốn.")
            return None
        
        # Lọc dữ liệu theo phạm vi thời gian nếu có
        equity_df = self.equity_df.copy()
        trades_df = None if self.trades_df is None else self.trades_df.copy()
        
        if date_range is not None:
            start_date, end_date = date_range
            equity_df = equity_df[(equity_df.index >= start_date) & (equity_df.index <= end_date)]
            
            if trades_df is not None:
                trades_df = trades_df[(trades_df["timestamp"] >= start_date) & (trades_df["timestamp"] <= end_date)]
        
        if use_plotly:
            # Sử dụng Plotly để vẽ biểu đồ
            height = height or self.height
            width = width or self.width
            
            # Tạo subplots nếu cần hiển thị drawdown
            fig = make_subplots(
                rows=2 if include_drawdown else 1,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3] if include_drawdown else [1]
            )
            
            # Thêm đường cong vốn
            fig.add_trace(
                go.Scatter(
                    x=equity_df.index,
                    y=equity_df["capital"],
                    mode="lines",
                    name="Vốn",
                    line=dict(color=self.colors["primary"], width=2)
                ),
                row=1, col=1
            )
            
            # Thêm marker cho các giao dịch nếu cần
            if include_trades and trades_df is not None and len(trades_df) > 0:
                # Chia các giao dịch thành thắng và thua
                winning_trades = trades_df[trades_df["is_win"] == True]
                losing_trades = trades_df[trades_df["is_win"] == False]
                
                # Thêm marker cho giao dịch thắng
                if len(winning_trades) > 0:
                    # Tìm giá trị vốn tương ứng với thời điểm giao dịch
                    win_capitals = []
                    for timestamp in winning_trades["timestamp"]:
                        idx = equity_df.index.get_indexer([timestamp], method='nearest')[0]
                        if idx >= 0 and idx < len(equity_df):
                            win_capitals.append(equity_df.iloc[idx]["capital"])
                        else:
                            win_capitals.append(None)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=winning_trades["timestamp"],
                            y=win_capitals,
                            mode="markers",
                            name="Thắng",
                            marker=dict(
                                color=self.colors["positive"],
                                size=8,
                                symbol="triangle-up",
                                line=dict(color="white", width=1)
                            ),
                            hovertemplate="<b>Thắng</b><br>Thời gian: %{x}<br>P&L: %{text}<extra></extra>",
                            text=[f"{pl:.2f} ({pl_pct:.2f}%)" for pl, pl_pct in zip(
                                winning_trades["profit_loss"], winning_trades["profit_loss_percent"])]
                        ),
                        row=1, col=1
                    )
                
                # Thêm marker cho giao dịch thua
                if len(losing_trades) > 0:
                    # Tìm giá trị vốn tương ứng với thời điểm giao dịch
                    loss_capitals = []
                    for timestamp in losing_trades["timestamp"]:
                        idx = equity_df.index.get_indexer([timestamp], method='nearest')[0]
                        if idx >= 0 and idx < len(equity_df):
                            loss_capitals.append(equity_df.iloc[idx]["capital"])
                        else:
                            loss_capitals.append(None)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=losing_trades["timestamp"],
                            y=loss_capitals,
                            mode="markers",
                            name="Thua",
                            marker=dict(
                                color=self.colors["negative"],
                                size=8,
                                symbol="triangle-down",
                                line=dict(color="white", width=1)
                            ),
                            hovertemplate="<b>Thua</b><br>Thời gian: %{x}<br>P&L: %{text}<extra></extra>",
                            text=[f"{pl:.2f} ({pl_pct:.2f}%)" for pl, pl_pct in zip(
                                losing_trades["profit_loss"], losing_trades["profit_loss_percent"])]
                        ),
                        row=1, col=1
                    )
            
            # Thêm drawdown nếu cần
            if include_drawdown:
                # Tính drawdown
                rolling_max = equity_df["capital"].cummax()
                drawdown_series = (equity_df["capital"] - rolling_max) / rolling_max * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=equity_df.index,
                        y=drawdown_series,
                        mode="lines",
                        name="Drawdown",
                        fill="tozeroy",
                        fillcolor=f"rgba{tuple(int(self.colors['negative'][1:][i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}",
                        line=dict(color=self.colors["negative"], width=1)
                    ),
                    row=2, col=1
                )
            
            # Thêm chỉ số hiệu suất nếu cần
            if display_metrics:
                # Tính các chỉ số
                initial_capital = equity_df.iloc[0]["capital"]
                final_capital = equity_df.iloc[-1]["capital"]
                total_return = (final_capital / initial_capital - 1) * 100
                
                # Tính max drawdown
                rolling_max = equity_df["capital"].cummax()
                drawdown = (equity_df["capital"] - rolling_max) / rolling_max
                max_drawdown = abs(drawdown.min()) * 100
                
                # Thêm annotation
                metrics_text = (
                    f"Vốn ban đầu: {initial_capital:.2f}<br>"
                    f"Vốn hiện tại: {final_capital:.2f}<br>"
                    f"Tổng lợi nhuận: {total_return:.2f}%<br>"
                    f"Max Drawdown: {max_drawdown:.2f}%"
                )
                
                fig.add_annotation(
                    text=metrics_text,
                    align="left",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.98,
                    bordercolor=self.colors["neutral"],
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="rgba(250, 250, 250, 0.8)" if self.theme != "dark" else "rgba(50, 50, 50, 0.8)",
                    font=dict(size=10)
                )
            
            # Cập nhật layout
            title = f"Đường cong vốn" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
            
            fig.update_layout(
                title=title,
                template="plotly" + ("_dark" if self.theme == "dark" else "_white"),
                height=height,
                width=width,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified",
                xaxis=dict(title="Thời gian"),
                yaxis=dict(title="Vốn"),
                yaxis2=dict(title="Drawdown (%)", tickformat=".2f") if include_drawdown else None
            )
            
            return fig
        
        else:
            # Sử dụng Matplotlib để vẽ biểu đồ
            height = height or (8 if include_drawdown else 6)
            width = width or 12
            
            if include_drawdown:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            else:
                fig, ax1 = plt.subplots(figsize=(width, height))
            
            # Vẽ đường cong vốn
            ax1.plot(equity_df.index, equity_df["capital"], color=self.colors["primary"], linewidth=2, label="Vốn")
            
            # Thêm marker cho các giao dịch nếu cần
            if include_trades and trades_df is not None and len(trades_df) > 0:
                # Chia các giao dịch thành thắng và thua
                winning_trades = trades_df[trades_df["is_win"] == True]
                losing_trades = trades_df[trades_df["is_win"] == False]
                
                # Thêm marker cho giao dịch thắng
                if len(winning_trades) > 0:
                    # Tìm giá trị vốn tương ứng với thời điểm giao dịch
                    for timestamp in winning_trades["timestamp"]:
                        idx = equity_df.index.get_indexer([timestamp], method='nearest')[0]
                        if idx >= 0 and idx < len(equity_df):
                            ax1.scatter(timestamp, equity_df.iloc[idx]["capital"], 
                                      marker="^", color=self.colors["positive"], s=50, zorder=3)
                
                # Thêm marker cho giao dịch thua
                if len(losing_trades) > 0:
                    # Tìm giá trị vốn tương ứng với thời điểm giao dịch
                    for timestamp in losing_trades["timestamp"]:
                        idx = equity_df.index.get_indexer([timestamp], method='nearest')[0]
                        if idx >= 0 and idx < len(equity_df):
                            ax1.scatter(timestamp, equity_df.iloc[idx]["capital"], 
                                      marker="v", color=self.colors["negative"], s=50, zorder=3)
            
            # Định dạng trục x
            ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Thêm grid
            ax1.grid(alpha=0.3)
            
            # Thêm title và label
            title = f"Đường cong vốn" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
            ax1.set_title(title)
            ax1.set_ylabel("Vốn")
            
            # Thêm drawdown nếu cần
            if include_drawdown:
                # Tính drawdown
                rolling_max = equity_df["capital"].cummax()
                drawdown_series = (equity_df["capital"] - rolling_max) / rolling_max * 100
                
                ax2.fill_between(
                    equity_df.index, 
                    0, 
                    drawdown_series, 
                    color=self.colors["negative"], 
                    alpha=0.3
                )
                ax2.plot(
                    equity_df.index, 
                    drawdown_series, 
                    color=self.colors["negative"], 
                    linewidth=1
                )
                
                ax2.set_ylabel("Drawdown (%)")
                ax2.grid(alpha=0.3)
                
                # Định dạng trục y cho drawdown (giá trị âm, hiển thị dương)
                ax2.set_ylim(min(drawdown_series.min() * 1.1, -1), 1)
                
                # For better display, invert the y-axis as drawdowns are negative
                ax2.invert_yaxis()
            
            # Thêm chỉ số hiệu suất nếu cần
            if display_metrics:
                # Tính các chỉ số
                initial_capital = equity_df.iloc[0]["capital"]
                final_capital = equity_df.iloc[-1]["capital"]
                total_return = (final_capital / initial_capital - 1) * 100
                
                # Tính max drawdown
                rolling_max = equity_df["capital"].cummax()
                drawdown = (equity_df["capital"] - rolling_max) / rolling_max
                max_drawdown = abs(drawdown.min()) * 100
                
                # Thêm text box
                metrics_text = (
                    f"Vốn ban đầu: {initial_capital:.2f}\n"
                    f"Vốn hiện tại: {final_capital:.2f}\n"
                    f"Tổng lợi nhuận: {total_return:.2f}%\n"
                    f"Max Drawdown: {max_drawdown:.2f}%"
                )
                
                props = dict(boxstyle='round', facecolor='white' if self.theme != "dark" else 'gray', alpha=0.7)
                ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=9,
                        verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            return fig
    
    def plot_drawdown_chart(
        self,
        use_plotly: bool = True,
        top_n: int = 5,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Vẽ biểu đồ drawdown và hiển thị các giai đoạn drawdown lớn nhất.
        
        Args:
            use_plotly: Sử dụng Plotly thay vì Matplotlib
            top_n: Số lượng giai đoạn drawdown lớn nhất để hiển thị
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            height: Chiều cao của biểu đồ
            width: Chiều rộng của biểu đồ
            
        Returns:
            Đối tượng go.Figure của Plotly hoặc plt.Figure của Matplotlib
        """
        if self.equity_df is None or len(self.equity_df) < 2:
            st.warning("Không có đủ dữ liệu để vẽ biểu đồ drawdown.")
            return None
        
        # Lọc dữ liệu theo phạm vi thời gian nếu có
        equity_df = self.equity_df.copy()
        
        if date_range is not None:
            start_date, end_date = date_range
            equity_df = equity_df[(equity_df.index >= start_date) & (equity_df.index <= end_date)]
        
        # Tính drawdown
        rolling_max = equity_df["capital"].cummax()
        drawdown_series = (equity_df["capital"] - rolling_max) / rolling_max * 100
        
        # Xác định các giai đoạn drawdown
        drawdown_periods = []
        in_drawdown = False
        start_idx = None
        peak_value = None
        
        for i, (date, value) in enumerate(drawdown_series.items()):
            if not in_drawdown and value < 0:
                # Bắt đầu giai đoạn drawdown mới
                in_drawdown = True
                start_idx = i
                peak_value = rolling_max[i]
                
            elif in_drawdown and value == 0:
                # Kết thúc giai đoạn drawdown
                end_idx = i
                
                if start_idx is not None:
                    trough_idx = drawdown_series.iloc[start_idx:end_idx].idxmin()
                    trough_value = equity_df.loc[trough_idx, "capital"]
                    drawdown_amount = abs(drawdown_series.iloc[start_idx:end_idx].min())
                    
                    drawdown_periods.append({
                        "start_date": drawdown_series.index[start_idx],
                        "end_date": drawdown_series.index[end_idx],
                        "trough_date": trough_idx,
                        "peak_value": peak_value,
                        "trough_value": trough_value,
                        "drawdown_percent": drawdown_amount,
                        "duration_days": (drawdown_series.index[end_idx] - drawdown_series.index[start_idx]).days
                    })
                
                in_drawdown = False
                start_idx = None
                peak_value = None
        
        # Nếu vẫn đang trong drawdown tại cuối chuỗi
        if in_drawdown and start_idx is not None:
            end_idx = len(drawdown_series) - 1
            trough_idx = drawdown_series.iloc[start_idx:end_idx+1].idxmin()
            trough_value = equity_df.loc[trough_idx, "capital"]
            drawdown_amount = abs(drawdown_series.iloc[start_idx:end_idx+1].min())
            
            drawdown_periods.append({
                "start_date": drawdown_series.index[start_idx],
                "end_date": drawdown_series.index[end_idx],
                "trough_date": trough_idx,
                "peak_value": peak_value,
                "trough_value": trough_value,
                "drawdown_percent": drawdown_amount,
                "duration_days": (drawdown_series.index[end_idx] - drawdown_series.index[start_idx]).days
            })
        
        # Sắp xếp các giai đoạn drawdown theo mức độ
        drawdown_periods.sort(key=lambda x: x["drawdown_percent"], reverse=True)
        top_drawdowns = drawdown_periods[:min(top_n, len(drawdown_periods))]
        
        if use_plotly:
            # Sử dụng Plotly để vẽ biểu đồ
            height = height or self.height
            width = width or self.width
            
            # Tạo subplots
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Thêm đường cong vốn
            fig.add_trace(
                go.Scatter(
                    x=equity_df.index,
                    y=equity_df["capital"],
                    mode="lines",
                    name="Vốn",
                    line=dict(color=self.colors["primary"], width=2)
                ),
                row=1, col=1
            )
            
            # Thêm drawdown
            fig.add_trace(
                go.Scatter(
                    x=equity_df.index,
                    y=drawdown_series,
                    mode="lines",
                    name="Drawdown",
                    fill="tozeroy",
                    fillcolor=f"rgba{tuple(int(self.colors['negative'][1:][i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}",
                    line=dict(color=self.colors["negative"], width=1)
                ),
                row=2, col=1
            )
            
            # Đánh dấu các giai đoạn drawdown lớn nhất
            for i, dd in enumerate(top_drawdowns):
                # Vẽ vùng drawdown trên biểu đồ equity
                fig.add_trace(
                    go.Scatter(
                        x=[dd["start_date"], dd["trough_date"], dd["end_date"]],
                        y=[equity_df.loc[dd["start_date"], "capital"], 
                           equity_df.loc[dd["trough_date"], "capital"], 
                           equity_df.loc[dd["end_date"], "capital"]],
                        mode="lines+markers",
                        name=f"DD #{i+1}: -{dd['drawdown_percent']:.2f}%",
                        line=dict(color=self.colors["secondary"], dash="dot", width=1.5),
                        marker=dict(size=8),
                        hovertemplate=(
                            "<b>Drawdown #%d</b><br>" +
                            "Start: %{x[0]}<br>" +
                            "Trough: %{x[1]}<br>" +
                            "End: %{x[2]}<br>" +
                            "Drawdown: %.2f%%<br>" +
                            "Duration: %d days<extra></extra>"
                        ) % (i+1, dd["drawdown_percent"], dd["duration_days"])
                    ),
                    row=1, col=1
                )
                
                # Đánh dấu trên biểu đồ drawdown
                fig.add_trace(
                    go.Scatter(
                        x=[dd["trough_date"]],
                        y=[drawdown_series.loc[dd["trough_date"]]],
                        mode="markers",
                        marker=dict(
                            color=self.colors["secondary"],
                            size=10,
                            symbol="circle",
                            line=dict(color="white", width=2)
                        ),
                        name=f"DD #{i+1} Trough",
                        showlegend=False,
                        hovertemplate=(
                            "<b>Peak Drawdown #%d</b><br>" +
                            "Date: %{x}<br>" +
                            "Drawdown: %.2f%%<extra></extra>"
                        ) % (i+1, dd["drawdown_percent"])
                    ),
                    row=2, col=1
                )
            
            # Cập nhật layout
            title = f"Phân tích Drawdown" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
            
            fig.update_layout(
                title=title,
                template="plotly" + ("_dark" if self.theme == "dark" else "_white"),
                height=height,
                width=width,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified",
                xaxis2=dict(title="Thời gian"),
                yaxis=dict(title="Vốn"),
                yaxis2=dict(title="Drawdown (%)", tickformat=".2f")
            )
            
            return fig
        
        else:
            # Sử dụng Matplotlib để vẽ biểu đồ
            height = height or 10
            width = width or 12
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            
            # Vẽ đường cong vốn
            ax1.plot(equity_df.index, equity_df["capital"], color=self.colors["primary"], linewidth=2, label="Vốn")
            
            # Đánh dấu các giai đoạn drawdown lớn nhất
            for i, dd in enumerate(top_drawdowns):
                # Vẽ vùng drawdown
                ax1.plot(
                    [dd["start_date"], dd["trough_date"], dd["end_date"]],
                    [equity_df.loc[dd["start_date"], "capital"], 
                     equity_df.loc[dd["trough_date"], "capital"], 
                     equity_df.loc[dd["end_date"], "capital"]],
                    color=self.colors["secondary"],
                    linestyle=":",
                    marker="o",
                    linewidth=1.5,
                    label=f"DD #{i+1}: -{dd['drawdown_percent']:.2f}%"
                )
            
            # Vẽ drawdown
            ax2.fill_between(
                equity_df.index, 
                0, 
                drawdown_series, 
                color=self.colors["negative"], 
                alpha=0.3
            )
            ax2.plot(
                equity_df.index, 
                drawdown_series, 
                color=self.colors["negative"], 
                linewidth=1
            )
            
            # Đánh dấu điểm drawdown tối đa
            for i, dd in enumerate(top_drawdowns):
                ax2.scatter(
                    dd["trough_date"],
                    drawdown_series.loc[dd["trough_date"]],
                    color=self.colors["secondary"],
                    s=80,
                    zorder=5,
                    edgecolor="white"
                )
            
            # Định dạng trục x
            ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Thêm grid
            ax1.grid(alpha=0.3)
            ax2.grid(alpha=0.3)
            
            # Thêm title và label
            title = f"Phân tích Drawdown" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
            ax1.set_title(title)
            ax1.set_ylabel("Vốn")
            ax2.set_ylabel("Drawdown (%)")
            
            # For better display, invert the y-axis as drawdowns are negative
            ax2.invert_yaxis()
            
            # Thêm legend
            ax1.legend(loc="upper left")
            
            plt.tight_layout()
            return fig
    
    def plot_returns_distribution(
        self,
        use_plotly: bool = True,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        bins: int = 50,
        display_stats: bool = True,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Vẽ biểu đồ phân phối lợi nhuận.
        
        Args:
            use_plotly: Sử dụng Plotly thay vì Matplotlib
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            bins: Số lượng bins cho histogram
            display_stats: Hiển thị các thống kê
            height: Chiều cao của biểu đồ
            width: Chiều rộng của biểu đồ
            
        Returns:
            Đối tượng go.Figure của Plotly hoặc plt.Figure của Matplotlib
        """
        if self.returns_df is None or len(self.returns_df) < 2:
            if self.equity_df is not None and len(self.equity_df) > 1:
                # Tính returns từ equity curve
                self.returns_df = pd.DataFrame({
                    "daily_return": self.equity_df["capital"].pct_change()
                })
                self.returns_df.dropna(inplace=True)
            else:
                st.warning("Không có đủ dữ liệu để vẽ phân phối lợi nhuận.")
                return None
        
        # Lọc dữ liệu theo phạm vi thời gian nếu có
        returns_df = self.returns_df.copy()
        
        if date_range is not None:
            start_date, end_date = date_range
            returns_df = returns_df[(returns_df.index >= start_date) & (returns_df.index <= end_date)]
        
        if use_plotly:
            # Sử dụng Plotly để vẽ biểu đồ
            height = height or self.height
            width = width or self.width
            
            # Tính các thống kê
            mean_return = returns_df["daily_return"].mean()
            std_return = returns_df["daily_return"].std()
            skew = returns_df["daily_return"].skew()
            kurtosis = returns_df["daily_return"].kurtosis()
            sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
            min_return = returns_df["daily_return"].min()
            max_return = returns_df["daily_return"].max()
            
            # Tạo histogram
            fig = go.Figure()
            
            fig.add_trace(
                go.Histogram(
                    x=returns_df["daily_return"] * 100,  # Chuyển thành phần trăm
                    nbinsx=bins,
                    name="Returns",
                    marker_color=self.colors["primary"],
                    opacity=0.7
                )
            )
            
            # Thêm đường phân phối chuẩn
            x_range = np.linspace(
                min(returns_df["daily_return"].min() * 100, -3 * std_return * 100),
                max(returns_df["daily_return"].max() * 100, 3 * std_return * 100),
                1000
            )
            y_normal = (1 / (std_return * 100 * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x_range - mean_return * 100) / (std_return * 100)) ** 2
            )
            
            # Scale the normal distribution to match the histogram
            scaling_factor = len(returns_df) * (x_range[1] - x_range[0])
            y_normal = y_normal * scaling_factor
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_normal,
                    mode="lines",
                    name="Phân phối chuẩn",
                    line=dict(color=self.colors["secondary"], width=2)
                )
            )
            
            # Thêm các đường tham chiếu
            fig.add_vline(
                x=mean_return * 100,
                line_dash="dash",
                line_color=self.colors["positive"],
                annotation_text=f"Mean: {mean_return*100:.2f}%",
                annotation_position="top right"
            )
            
            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color=self.colors["neutral"],
                annotation_text="0%",
                annotation_position="bottom right"
            )
            
            # Thêm thông tin thống kê nếu cần
            if display_stats:
                stats_text = (
                    f"Mean: {mean_return*100:.2f}%<br>"
                    f"Std: {std_return*100:.2f}%<br>"
                    f"Skew: {skew:.2f}<br>"
                    f"Kurtosis: {kurtosis:.2f}<br>"
                    f"Sharpe (annualized): {sharpe:.2f}<br>"
                    f"Min: {min_return*100:.2f}%<br>"
                    f"Max: {max_return*100:.2f}%"
                )
                
                fig.add_annotation(
                    text=stats_text,
                    align="left",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.98,
                    bordercolor=self.colors["neutral"],
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="rgba(250, 250, 250, 0.8)" if self.theme != "dark" else "rgba(50, 50, 50, 0.8)",
                    font=dict(size=10)
                )
            
            # Cập nhật layout
            title = f"Phân phối lợi nhuận" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
            
            fig.update_layout(
                title=title,
                template="plotly" + ("_dark" if self.theme == "dark" else "_white"),
                height=height,
                width=width,
                xaxis=dict(title="Lợi nhuận (%)"),
                yaxis=dict(title="Tần suất"),
                bargap=0.1
            )
            
            return fig
        
        else:
            # Sử dụng Matplotlib để vẽ biểu đồ
            height = height or 6
            width = width or 10
            
            fig, ax = plt.subplots(figsize=(width, height))
            
            # Tính các thống kê
            mean_return = returns_df["daily_return"].mean()
            std_return = returns_df["daily_return"].std()
            skew = returns_df["daily_return"].skew()
            kurtosis = returns_df["daily_return"].kurtosis()
            sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
            min_return = returns_df["daily_return"].min()
            max_return = returns_df["daily_return"].max()
            
            # Vẽ histogram
            n, bins_values, patches = ax.hist(
                returns_df["daily_return"] * 100,  # Chuyển thành phần trăm
                bins=bins,
                color=self.colors["primary"],
                alpha=0.7
            )
            
            # Thêm đường phân phối chuẩn
            x_range = np.linspace(
                min(returns_df["daily_return"].min() * 100, -3 * std_return * 100),
                max(returns_df["daily_return"].max() * 100, 3 * std_return * 100),
                1000
            )
            y_normal = (1 / (std_return * 100 * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x_range - mean_return * 100) / (std_return * 100)) ** 2
            )
            
            # Scale the normal distribution to match the histogram
            bin_width = bins_values[1] - bins_values[0]
            scaling_factor = len(returns_df) * bin_width
            y_normal = y_normal * scaling_factor
            
            ax.plot(x_range, y_normal, color=self.colors["secondary"], linewidth=2, label="Phân phối chuẩn")
            
            # Thêm các đường tham chiếu
            ax.axvline(mean_return * 100, color=self.colors["positive"], linestyle="--", 
                      label=f"Mean: {mean_return*100:.2f}%")
            ax.axvline(0, color=self.colors["neutral"], linestyle="--")
            
            # Thêm thông tin thống kê nếu cần
            if display_stats:
                stats_text = (
                    f"Mean: {mean_return*100:.2f}%\n"
                    f"Std: {std_return*100:.2f}%\n"
                    f"Skew: {skew:.2f}\n"
                    f"Kurtosis: {kurtosis:.2f}\n"
                    f"Sharpe (annualized): {sharpe:.2f}\n"
                    f"Min: {min_return*100:.2f}%\n"
                    f"Max: {max_return*100:.2f}%"
                )
                
                props = dict(boxstyle='round', facecolor='white' if self.theme != "dark" else 'gray', alpha=0.7)
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=props)
            
            # Thêm grid
            ax.grid(alpha=0.3)
            
            # Thêm title và label
            title = f"Phân phối lợi nhuận" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
            ax.set_title(title)
            ax.set_xlabel("Lợi nhuận (%)")
            ax.set_ylabel("Tần suất")
            
            # Thêm legend
            ax.legend()
            
            plt.tight_layout()
            return fig
    
    def plot_monthly_returns_heatmap(
        self,
        use_plotly: bool = True,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Vẽ heatmap lợi nhuận theo tháng và năm.
        
        Args:
            use_plotly: Sử dụng Plotly thay vì Matplotlib
            height: Chiều cao của biểu đồ
            width: Chiều rộng của biểu đồ
            
        Returns:
            Đối tượng go.Figure của Plotly hoặc plt.Figure của Matplotlib
        """
        if self.equity_df is None or len(self.equity_df) < 30:
            st.warning("Không có đủ dữ liệu để vẽ heatmap lợi nhuận hàng tháng.")
            return None
        
        # Tính lợi nhuận hàng tháng
        monthly_returns = self.equity_df["capital"].resample('M').last().pct_change().dropna()
        
        # Tạo DataFrame với chỉ số là năm và tháng
        monthly_returns.index = pd.MultiIndex.from_arrays(
            [monthly_returns.index.year, monthly_returns.index.month],
            names=['Year', 'Month']
        )
        
        # Pivot để tạo bảng năm x tháng
        heatmap_data = monthly_returns.unstack(level='Month')
        
        if use_plotly:
            # Sử dụng Plotly để vẽ biểu đồ
            height = height or self.height
            width = width or self.width
            
            # Chuẩn bị dữ liệu cho heatmap
            years = heatmap_data.index.tolist()
            months = list(range(1, 13))
            
            z_data = []
            for year in years:
                month_data = []
                for month in months:
                    if month in heatmap_data.columns:
                        value = heatmap_data.loc[year, month]
                        month_data.append(100 * value if not pd.isna(value) else None)
                    else:
                        month_data.append(None)
                z_data.append(month_data)
            
            # Tạo heatmap
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=month_names,
                y=years,
                colorscale='RdYlGn',  # Red for negative, Green for positive
                zmid=0,  # Centered around zero
                text=[[f"{z:.2f}%" if z is not None else "" for z in row] for row in z_data],
                hovertemplate="Năm: %{y}<br>Tháng: %{x}<br>Lợi nhuận: %{z:.2f}%<extra></extra>"
            ))
            
            # Cập nhật layout
            title = f"Lợi nhuận hàng tháng" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
            
            fig.update_layout(
                title=title,
                template="plotly" + ("_dark" if self.theme == "dark" else "_white"),
                height=height,
                width=width,
                xaxis=dict(title="Tháng"),
                yaxis=dict(title="Năm")
            )
            
            return fig
        
        else:
            # Sử dụng Matplotlib để vẽ biểu đồ
            height = height or 8
            width = width or 10
            
            fig, ax = plt.subplots(figsize=(width, height))
            
            # Heatmap sử dụng seaborn
            heatmap = sns.heatmap(
                heatmap_data * 100,  # Chuyển thành phần trăm
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",  # Red for negative, Green for positive
                center=0,
                ax=ax
            )
            
            # Thêm title và label
            title = f"Lợi nhuận hàng tháng" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
            ax.set_title(title)
            ax.set_ylabel("Năm")
            ax.set_xlabel("Tháng")
            
            # Cập nhật labels cho tháng
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticklabels(month_labels)
            
            plt.tight_layout()
            return fig
    
    def plot_trade_analysis(
        self,
        use_plotly: bool = True,
        breakdown_by: str = "profit_loss",
        trade_count: int = 100,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Vẽ biểu đồ phân tích các giao dịch (trade analysis).
        
        Args:
            use_plotly: Sử dụng Plotly thay vì Matplotlib
            breakdown_by: Phân tích theo tiêu chí ('profit_loss', 'duration', 'reason')
            trade_count: Số lượng giao dịch gần nhất để phân tích
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            height: Chiều cao của biểu đồ
            width: Chiều rộng của biểu đồ
            
        Returns:
            Đối tượng go.Figure của Plotly hoặc plt.Figure của Matplotlib
        """
        if self.trades_df is None or len(self.trades_df) == 0:
            st.warning("Không có dữ liệu giao dịch để phân tích.")
            return None
        
        # Lọc dữ liệu theo phạm vi thời gian nếu có
        trades_df = self.trades_df.copy()
        
        if date_range is not None:
            start_date, end_date = date_range
            trades_df = trades_df[(trades_df["timestamp"] >= start_date) & (trades_df["timestamp"] <= end_date)]
        
        # Lấy các giao dịch gần nhất
        trades_df = trades_df.sort_values("timestamp", ascending=False).head(trade_count)
        
        # Phân loại giao dịch
        if breakdown_by == "profit_loss":
            win_trades = trades_df[trades_df["profit_loss"] > 0]
            loss_trades = trades_df[trades_df["profit_loss"] <= 0]
            
            win_percent = len(win_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
            
            avg_win = win_trades["profit_loss"].mean() if len(win_trades) > 0 else 0
            avg_loss = loss_trades["profit_loss"].mean() if len(loss_trades) > 0 else 0
            
            profit_factor = abs(win_trades["profit_loss"].sum() / loss_trades["profit_loss"].sum()) if loss_trades["profit_loss"].sum() != 0 else float('inf')
            
            if use_plotly:
                # Sử dụng Plotly để vẽ biểu đồ
                height = height or self.height
                width = width or self.width
                
                # Tạo subplots
                fig = make_subplots(
                    rows=2, 
                    cols=2,
                    specs=[
                        [{"type": "pie"}, {"type": "bar"}],
                        [{"type": "scatter", "colspan": 2}, None]
                    ],
                    subplot_titles=("Tỷ lệ thắng/thua", "Thống kê lợi nhuận/lỗ", "Biến động lợi nhuận"),
                    vertical_spacing=0.1,
                    horizontal_spacing=0.1
                )
                
                # 1. Biểu đồ tròn thắng/thua
                fig.add_trace(
                    go.Pie(
                        labels=["Thắng", "Thua"],
                        values=[len(win_trades), len(loss_trades)],
                        marker=dict(colors=[self.colors["positive"], self.colors["negative"]]),
                        textinfo="percent+label",
                        hole=0.3,
                        hoverinfo="label+percent+value"
                    ),
                    row=1, col=1
                )
                
                # 2. Biểu đồ cột lợi nhuận trung bình
                fig.add_trace(
                    go.Bar(
                        x=["Lợi nhuận TB", "Lỗ TB", "Profit Factor"],
                        y=[avg_win, avg_loss, profit_factor],
                        marker=dict(
                            color=[self.colors["positive"], self.colors["negative"], self.colors["primary"]]
                        ),
                        text=[f"{avg_win:.2f}", f"{avg_loss:.2f}", f"{profit_factor:.2f}"],
                        textposition="auto"
                    ),
                    row=1, col=2
                )
                
                # 3. Biểu đồ giao dịch theo thời gian
                fig.add_trace(
                    go.Scatter(
                        x=trades_df["timestamp"],
                        y=trades_df["profit_loss"],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=trades_df["profit_loss"].apply(
                                lambda x: self.colors["positive"] if x > 0 else self.colors["negative"]
                            ),
                            symbol=trades_df["profit_loss"].apply(
                                lambda x: "triangle-up" if x > 0 else "triangle-down"
                            ),
                            line=dict(width=1, color="white")
                        ),
                        hovertemplate=(
                            "<b>Giao dịch</b><br>" +
                            "Thời gian: %{x}<br>" +
                            "P&L: %{y:.2f}<br>" +
                            "P&L %%: %{text}<extra></extra>"
                        ),
                        text=trades_df["profit_loss_percent"].apply(lambda x: f"{x:.2f}%")
                    ),
                    row=2, col=1
                )
                
                # Thêm đường tham chiếu 0
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color=self.colors["neutral"],
                    row=2, col=1
                )
                
                # Cập nhật layout
                title = f"Phân tích giao dịch" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
                
                fig.update_layout(
                    title=title,
                    template="plotly" + ("_dark" if self.theme == "dark" else "_white"),
                    height=height,
                    width=width,
                    showlegend=False,
                    annotations=[
                        dict(
                            text=f"Win rate: {win_percent:.1f}%",
                            x=0.5, y=0.5,
                            xref="x1", yref="paper",
                            showarrow=False,
                            font=dict(size=14)
                        )
                    ]
                )
                
                # Cập nhật trục
                fig.update_xaxes(title_text="Thời gian", row=2, col=1)
                fig.update_yaxes(title_text="Lợi nhuận", row=2, col=1)
                
                return fig
                
            else:
                # Sử dụng Matplotlib để vẽ biểu đồ
                height = height or 12
                width = width or 15
                
                fig, axs = plt.subplots(2, 2, figsize=(width, height), 
                                       gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
                
                # 1. Biểu đồ tròn thắng/thua
                axs[0, 0].pie(
                    [len(win_trades), len(loss_trades)],
                    labels=["Thắng", "Thua"],
                    colors=[self.colors["positive"], self.colors["negative"]],
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops=dict(width=0.3)
                )
                axs[0, 0].set_title("Tỷ lệ thắng/thua")
                
                # Thêm win rate ở giữa
                axs[0, 0].text(0, 0, f"{win_percent:.1f}%", ha='center', va='center', fontsize=12)
                
                # 2. Biểu đồ cột lợi nhuận trung bình
                bars = axs[0, 1].bar(
                    ["Lợi nhuận TB", "Lỗ TB", "Profit Factor"],
                    [avg_win, avg_loss, profit_factor],
                    color=[self.colors["positive"], self.colors["negative"], self.colors["primary"]]
                )
                
                # Thêm giá trị trên mỗi cột
                for bar in bars:
                    height = bar.get_height()
                    axs[0, 1].text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f'{height:.2f}',
                        ha='center', va='bottom'
                    )
                
                axs[0, 1].set_title("Thống kê lợi nhuận/lỗ")
                axs[0, 1].grid(alpha=0.3)
                
                # 3. Biểu đồ giao dịch theo thời gian
                for i, row in trades_df.iterrows():
                    marker = '^' if row["profit_loss"] > 0 else 'v'
                    color = self.colors["positive"] if row["profit_loss"] > 0 else self.colors["negative"]
                    axs[1, 0].scatter(row["timestamp"], row["profit_loss"], marker=marker, color=color, s=100)
                
                axs[1, 0].axhline(y=0, color=self.colors["neutral"], linestyle='--', alpha=0.7)
                axs[1, 0].set_title("Biến động lợi nhuận")
                axs[1, 0].set_xlabel("Thời gian")
                axs[1, 0].set_ylabel("Lợi nhuận")
                axs[1, 0].grid(alpha=0.3)
                
                # Định dạng trục thời gian
                axs[1, 0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                plt.setp(axs[1, 0].xaxis.get_majorticklabels(), rotation=45)
                
                # Remove the unused subplot
                fig.delaxes(axs[1, 1])
                
                # Thêm title
                title = f"Phân tích giao dịch" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
                plt.suptitle(title, fontsize=16)
                
                plt.tight_layout()
                return fig
        
        elif breakdown_by == "duration":
            # Phân loại theo thời gian nắm giữ
            trades_df["duration_group"] = pd.cut(
                trades_df["duration"],
                bins=[0, 1, 4, 24, 72, float('inf')],
                labels=["<1h", "1-4h", "4-24h", "1-3d", ">3d"]
            )
            
            duration_groups = trades_df.groupby("duration_group")
            
            if use_plotly:
                # Sử dụng Plotly để vẽ biểu đồ
                height = height or self.height
                width = width or self.width
                
                # Tạo subplots
                fig = make_subplots(
                    rows=2, 
                    cols=2,
                    specs=[
                        [{"type": "pie"}, {"type": "bar"}],
                        [{"type": "scatter", "colspan": 2}, None]
                    ],
                    subplot_titles=("Phân bố thời gian nắm giữ", "Lợi nhuận theo thời gian nắm giữ", "Lợi nhuận vs Thời gian nắm giữ"),
                    vertical_spacing=0.1,
                    horizontal_spacing=0.1
                )
                
                # 1. Biểu đồ tròn phân bố thời gian nắm giữ
                duration_counts = duration_groups.size()
                
                fig.add_trace(
                    go.Pie(
                        labels=duration_counts.index,
                        values=duration_counts.values,
                        textinfo="percent+label",
                        hole=0.3,
                        hoverinfo="label+percent+value"
                    ),
                    row=1, col=1
                )
                
                # 2. Biểu đồ cột lợi nhuận trung bình theo thời gian nắm giữ
                avg_profit_by_duration = duration_groups["profit_loss"].mean()
                
                fig.add_trace(
                    go.Bar(
                        x=avg_profit_by_duration.index,
                        y=avg_profit_by_duration.values,
                        marker=dict(
                            color=[self.colors["positive"] if x > 0 else self.colors["negative"] 
                                  for x in avg_profit_by_duration.values]
                        ),
                        text=[f"{x:.2f}" for x in avg_profit_by_duration.values],
                        textposition="auto"
                    ),
                    row=1, col=2
                )
                
                # 3. Biểu đồ scatter lợi nhuận vs thời gian nắm giữ
                fig.add_trace(
                    go.Scatter(
                        x=trades_df["duration"],
                        y=trades_df["profit_loss"],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=trades_df["profit_loss"].apply(
                                lambda x: self.colors["positive"] if x > 0 else self.colors["negative"]
                            ),
                            symbol=trades_df["profit_loss"].apply(
                                lambda x: "triangle-up" if x > 0 else "triangle-down"
                            ),
                            line=dict(width=1, color="white")
                        ),
                        hovertemplate=(
                            "<b>Giao dịch</b><br>" +
                            "Thời gian nắm giữ: %{x:.2f}h<br>" +
                            "P&L: %{y:.2f}<extra></extra>"
                        )
                    ),
                    row=2, col=1
                )
                
                # Thêm đường tham chiếu 0
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color=self.colors["neutral"],
                    row=2, col=1
                )
                
                # Cập nhật layout
                title = f"Phân tích thời gian nắm giữ" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
                
                fig.update_layout(
                    title=title,
                    template="plotly" + ("_dark" if self.theme == "dark" else "_white"),
                    height=height,
                    width=width,
                    showlegend=False
                )
                
                # Cập nhật trục
                fig.update_xaxes(title_text="Thời gian nắm giữ (giờ)", row=2, col=1)
                fig.update_yaxes(title_text="Lợi nhuận", row=2, col=1)
                
                return fig
                
            else:
                # Sử dụng Matplotlib để vẽ biểu đồ
                height = height or 12
                width = width or 15
                
                fig, axs = plt.subplots(2, 2, figsize=(width, height), 
                                       gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
                
                # 1. Biểu đồ tròn phân bố thời gian nắm giữ
                duration_counts = duration_groups.size()
                
                axs[0, 0].pie(
                    duration_counts.values,
                    labels=duration_counts.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops=dict(width=0.3)
                )
                axs[0, 0].set_title("Phân bố thời gian nắm giữ")
                
                # 2. Biểu đồ cột lợi nhuận trung bình theo thời gian nắm giữ
                avg_profit_by_duration = duration_groups["profit_loss"].mean()
                
                bars = axs[0, 1].bar(
                    avg_profit_by_duration.index,
                    avg_profit_by_duration.values,
                    color=[self.colors["positive"] if x > 0 else self.colors["negative"] 
                          for x in avg_profit_by_duration.values]
                )
                
                # Thêm giá trị trên mỗi cột
                for bar in bars:
                    height = bar.get_height()
                    axs[0, 1].text(
                        bar.get_x() + bar.get_width()/2.,
                        height if height > 0 else height - 1,
                        f'{height:.2f}',
                        ha='center', va='bottom' if height > 0 else 'top'
                    )
                
                axs[0, 1].set_title("Lợi nhuận theo thời gian nắm giữ")
                axs[0, 1].grid(alpha=0.3)
                
                # 3. Biểu đồ scatter lợi nhuận vs thời gian nắm giữ
                for i, row in trades_df.iterrows():
                    marker = '^' if row["profit_loss"] > 0 else 'v'
                    color = self.colors["positive"] if row["profit_loss"] > 0 else self.colors["negative"]
                    axs[1, 0].scatter(row["duration"], row["profit_loss"], marker=marker, color=color, s=100)
                
                axs[1, 0].axhline(y=0, color=self.colors["neutral"], linestyle='--', alpha=0.7)
                axs[1, 0].set_title("Lợi nhuận vs Thời gian nắm giữ")
                axs[1, 0].set_xlabel("Thời gian nắm giữ (giờ)")
                axs[1, 0].set_ylabel("Lợi nhuận")
                axs[1, 0].grid(alpha=0.3)
                
                # Remove the unused subplot
                fig.delaxes(axs[1, 1])
                
                # Thêm title
                title = f"Phân tích thời gian nắm giữ" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
                plt.suptitle(title, fontsize=16)
                
                plt.tight_layout()
                return fig
        
        elif breakdown_by == "reason":
            # Phân loại theo lý do đóng vị thế
            if "reason" not in trades_df.columns:
                trades_df["reason"] = "manual"  # Mặc định nếu không có thông tin
            
            reason_groups = trades_df.groupby("reason")
            
            if use_plotly:
                # Sử dụng Plotly để vẽ biểu đồ
                height = height or self.height
                width = width or self.width
                
                # Tạo subplots
                fig = make_subplots(
                    rows=2, 
                    cols=2,
                    specs=[
                        [{"type": "pie"}, {"type": "bar"}],
                        [{"type": "bar", "colspan": 2}, None]
                    ],
                    subplot_titles=("Phân bố lý do đóng vị thế", "Lợi nhuận trung bình theo lý do", "Win rate theo lý do"),
                    vertical_spacing=0.1,
                    horizontal_spacing=0.1
                )
                
                # 1. Biểu đồ tròn phân bố lý do đóng vị thế
                reason_counts = reason_groups.size()
                
                fig.add_trace(
                    go.Pie(
                        labels=reason_counts.index,
                        values=reason_counts.values,
                        textinfo="percent+label",
                        hole=0.3,
                        hoverinfo="label+percent+value"
                    ),
                    row=1, col=1
                )
                
                # 2. Biểu đồ cột lợi nhuận trung bình theo lý do
                avg_profit_by_reason = reason_groups["profit_loss"].mean()
                
                fig.add_trace(
                    go.Bar(
                        x=avg_profit_by_reason.index,
                        y=avg_profit_by_reason.values,
                        marker=dict(
                            color=[self.colors["positive"] if x > 0 else self.colors["negative"] 
                                  for x in avg_profit_by_reason.values]
                        ),
                        text=[f"{x:.2f}" for x in avg_profit_by_reason.values],
                        textposition="auto"
                    ),
                    row=1, col=2
                )
                
                # 3. Biểu đồ cột win rate theo lý do
                win_rate_by_reason = {}
                for reason, group in reason_groups:
                    wins = len(group[group["profit_loss"] > 0])
                    total = len(group)
                    win_rate_by_reason[reason] = wins / total * 100 if total > 0 else 0
                
                fig.add_trace(
                    go.Bar(
                        x=list(win_rate_by_reason.keys()),
                        y=list(win_rate_by_reason.values()),
                        marker=dict(color=self.colors["primary"]),
                        text=[f"{x:.1f}%" for x in win_rate_by_reason.values()],
                        textposition="auto"
                    ),
                    row=2, col=1
                )
                
                # Cập nhật layout
                title = f"Phân tích lý do đóng vị thế" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
                
                fig.update_layout(
                    title=title,
                    template="plotly" + ("_dark" if self.theme == "dark" else "_white"),
                    height=height,
                    width=width,
                    showlegend=False
                )
                
                # Cập nhật trục
                fig.update_yaxes(title_text="Win rate (%)", row=2, col=1)
                
                return fig
                
            else:
                # Sử dụng Matplotlib để vẽ biểu đồ
                height = height or 12
                width = width or 15
                
                fig, axs = plt.subplots(2, 2, figsize=(width, height), 
                                       gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
                
                # 1. Biểu đồ tròn phân bố lý do đóng vị thế
                reason_counts = reason_groups.size()
                
                axs[0, 0].pie(
                    reason_counts.values,
                    labels=reason_counts.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops=dict(width=0.3)
                )
                axs[0, 0].set_title("Phân bố lý do đóng vị thế")
                
                # 2. Biểu đồ cột lợi nhuận trung bình theo lý do
                avg_profit_by_reason = reason_groups["profit_loss"].mean()
                
                bars = axs[0, 1].bar(
                    avg_profit_by_reason.index,
                    avg_profit_by_reason.values,
                    color=[self.colors["positive"] if x > 0 else self.colors["negative"] 
                          for x in avg_profit_by_reason.values]
                )
                
                # Thêm giá trị trên mỗi cột
                for bar in bars:
                    height = bar.get_height()
                    axs[0, 1].text(
                        bar.get_x() + bar.get_width()/2.,
                        height if height > 0 else height - 0.5,
                        f'{height:.2f}',
                        ha='center', va='bottom' if height > 0 else 'top'
                    )
                
                axs[0, 1].set_title("Lợi nhuận trung bình theo lý do")
                axs[0, 1].grid(alpha=0.3)
                
                # 3. Biểu đồ cột win rate theo lý do
                win_rate_by_reason = {}
                for reason, group in reason_groups:
                    wins = len(group[group["profit_loss"] > 0])
                    total = len(group)
                    win_rate_by_reason[reason] = wins / total * 100 if total > 0 else 0
                
                bars = axs[1, 0].bar(
                    win_rate_by_reason.keys(),
                    win_rate_by_reason.values(),
                    color=self.colors["primary"]
                )
                
                # Thêm giá trị trên mỗi cột
                for bar in bars:
                    height = bar.get_height()
                    axs[1, 0].text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f'{height:.1f}%',
                        ha='center', va='bottom'
                    )
                
                axs[1, 0].set_title("Win rate theo lý do")
                axs[1, 0].set_ylabel("Win rate (%)")
                axs[1, 0].grid(alpha=0.3)
                
                # Remove the unused subplot
                fig.delaxes(axs[1, 1])
                
                # Thêm title
                title = f"Phân tích lý do đóng vị thế" + (f" - {self.symbol}" if self.symbol else "") + (f" ({self.strategy_name})" if self.strategy_name else "")
                plt.suptitle(title, fontsize=16)
                
                plt.tight_layout()
                return fig
    
    def plot_portfolio_allocation(
        self,
        use_plotly: bool = True,
        display_changes: bool = True,
        date: Optional[datetime] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Vẽ biểu đồ phân bổ danh mục (portfolio allocation).
        
        Args:
            use_plotly: Sử dụng Plotly thay vì Matplotlib
            display_changes: Hiển thị thay đổi phân bổ theo thời gian
            date: Ngày cụ thể để hiển thị phân bổ (None để hiển thị gần nhất)
            height: Chiều cao của biểu đồ
            width: Chiều rộng của biểu đồ
            
        Returns:
            Đối tượng go.Figure của Plotly hoặc plt.Figure của Matplotlib
        """
        if self.portfolio_df is None or len(self.portfolio_df) == 0:
            st.warning("Không có dữ liệu danh mục để phân tích.")
            return None
        
        # Lọc dữ liệu theo ngày nếu cung cấp
        portfolio_df = self.portfolio_df.copy()
        
        if date is not None:
            # Tìm ngày gần nhất
            idx = portfolio_df.index.get_indexer([date], method='nearest')[0]
            portfolio_df = portfolio_df.iloc[[idx]]
        else:
            # Lấy dữ liệu gần nhất
            portfolio_df = portfolio_df.iloc[[-1]]
        
        # Lấy tất cả các cột chứa thông tin vốn của từng symbol
        symbol_columns = [col for col in portfolio_df.columns if col.endswith('_capital')]
        symbols = [col.replace('_capital', '') for col in symbol_columns]
        
        if len(symbol_columns) == 0:
            st.warning("Không có thông tin phân bổ vốn theo symbol.")
            return None
        
        # Tạo dữ liệu phân bổ
        allocation_data = {}
        
        for symbol, col in zip(symbols, symbol_columns):
            allocation_data[symbol] = portfolio_df[col].iloc[0]
        
        # Tạo dữ liệu thay đổi phân bổ nếu cần
        allocation_changes = {}
        
        if display_changes and len(self.portfolio_df) > 1:
            # Lấy phân bổ trước đó (1 tuần hoặc 10 điểm dữ liệu, tùy thuộc giá trị nào nhỏ hơn)
            days_ago = 7
            lookback_points = min(10, len(self.portfolio_df) - 1)
            
            if len(self.portfolio_df) > lookback_points:
                previous_df = self.portfolio_df.iloc[-lookback_points-1]
                
                for symbol, col in zip(symbols, symbol_columns):
                    current = allocation_data[symbol]
                    previous = previous_df[col] if col in previous_df else 0
                    change = (current - previous) / previous * 100 if previous > 0 else float('inf')
                    allocation_changes[symbol] = change
        
        if use_plotly:
            # Sử dụng Plotly để vẽ biểu đồ
            height = height or self.height
            width = width or self.width
            
            # Tạo subplots
            fig = make_subplots(
                rows=1, 
                cols=2 if display_changes and allocation_changes else 1,
                specs=[
                    [{"type": "pie"}, {"type": "bar"} if display_changes and allocation_changes else None]
                ],
                subplot_titles=(
                    "Phân bổ danh mục hiện tại", 
                    "Thay đổi phân bổ (7 ngày)" if display_changes and allocation_changes else None
                )
            )
            
            # 1. Biểu đồ tròn phân bổ hiện tại
            fig.add_trace(
                go.Pie(
                    labels=list(allocation_data.keys()),
                    values=list(allocation_data.values()),
                    textinfo="percent+label",
                    hole=0.3,
                    hoverinfo="label+percent+value+text",
                    text=[f"{val:.2f}" for val in allocation_data.values()]
                ),
                row=1, col=1
            )
            
            # 2. Biểu đồ cột thay đổi phân bổ
            if display_changes and allocation_changes:
                fig.add_trace(
                    go.Bar(
                        x=list(allocation_changes.keys()),
                        y=list(allocation_changes.values()),
                        marker=dict(
                            color=[self.colors["positive"] if x > 0 else self.colors["negative"] 
                                  for x in allocation_changes.values()]
                        ),
                        text=[f"{x:.1f}%" if x != float('inf') else "New" for x in allocation_changes.values()],
                        textposition="auto"
                    ),
                    row=1, col=2
                )
                
                # Thêm đường tham chiếu 0
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color=self.colors["neutral"],
                    row=1, col=2
                )
            
            # Cập nhật layout
            title = f"Phân bổ danh mục" + (f" - {datetime.strftime(portfolio_df.index[0], '%Y-%m-%d')}" if portfolio_df.index[0] else "")
            
            fig.update_layout(
                title=title,
                template="plotly" + ("_dark" if self.theme == "dark" else "_white"),
                height=height,
                width=width,
                showlegend=False
            )
            
            # Cập nhật trục nếu có biểu đồ thay đổi
            if display_changes and allocation_changes:
                fig.update_yaxes(title_text="Thay đổi (%)", row=1, col=2)
            
            return fig
            
        else:
            # Sử dụng Matplotlib để vẽ biểu đồ
            height = height or 8
            width = width or 12
            
            if display_changes and allocation_changes:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height))
            else:
                fig, ax1 = plt.subplots(figsize=(width, height))
            
            # 1. Biểu đồ tròn phân bổ hiện tại
            ax1.pie(
                list(allocation_data.values()),
                labels=list(allocation_data.keys()),
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(width=0.3)
            )
            ax1.set_title("Phân bổ danh mục hiện tại")
            
            # Thêm tổng vốn ở giữa
            total_capital = sum(allocation_data.values())
            ax1.text(0, 0, f"{total_capital:.2f}", ha='center', va='center', fontsize=12)
            
            # 2. Biểu đồ cột thay đổi phân bổ
            if display_changes and allocation_changes:
                bars = ax2.bar(
                    allocation_changes.keys(),
                    allocation_changes.values(),
                    color=[self.colors["positive"] if x > 0 else self.colors["negative"] 
                          for x in allocation_changes.values()]
                )
                
                # Thêm giá trị trên mỗi cột
                for bar in bars:
                    height = bar.get_height()
                    if height != float('inf'):
                        ax2.text(
                            bar.get_x() + bar.get_width()/2.,
                            height if height > 0 else height - 5,
                            f'{height:.1f}%',
                            ha='center', va='bottom' if height > 0 else 'top'
                        )
                    else:
                        ax2.text(
                            bar.get_x() + bar.get_width()/2.,
                            0,
                            'New',
                            ha='center', va='bottom'
                        )
                
                ax2.axhline(y=0, color=self.colors["neutral"], linestyle='--', alpha=0.7)
                ax2.set_title("Thay đổi phân bổ (7 ngày)")
                ax2.set_ylabel("Thay đổi (%)")
                ax2.grid(alpha=0.3)
            
            # Thêm title
            title = f"Phân bổ danh mục" + (f" - {datetime.strftime(portfolio_df.index[0], '%Y-%m-%d')}" if portfolio_df.index[0] else "")
            plt.suptitle(title, fontsize=16)
            
            plt.tight_layout()
            return fig
    
    def plot_performance_comparison(
        self,
        benchmark_data: Union[pd.DataFrame, Dict[str, float], List[float]],
        benchmark_name: str = "Benchmark",
        use_plotly: bool = True,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        normalize: bool = True,
        include_metrics: bool = True,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> go.Figure:
        """
        Vẽ biểu đồ so sánh hiệu suất với benchmark.
        
        Args:
            benchmark_data: Dữ liệu benchmark (DataFrame, Dict, hoặc List)
            benchmark_name: Tên benchmark
            use_plotly: Sử dụng Plotly thay vì Matplotlib
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            normalize: Chuẩn hóa dữ liệu về 100 tại thời điểm bắt đầu
            include_metrics: Hiển thị các chỉ số so sánh
            height: Chiều cao của biểu đồ
            width: Chiều rộng của biểu đồ
            
        Returns:
            Đối tượng go.Figure của Plotly hoặc plt.Figure của Matplotlib
        """
        if self.equity_df is None or len(self.equity_df) < 2:
            st.warning("Không có đủ dữ liệu vốn để so sánh với benchmark.")
            return None
        
        # Chuyển đổi benchmark_data sang DataFrame
        if isinstance(benchmark_data, dict):
            # Dict với key là timestamp và value là giá trị
            benchmark_df = pd.DataFrame({
                'value': pd.Series(benchmark_data)
            })
            benchmark_df.index = pd.to_datetime(benchmark_df.index)
            
        elif isinstance(benchmark_data, list):
            # List các giá trị theo thứ tự thời gian
            if len(self.equity_df) < len(benchmark_data):
                # Cắt bớt benchmark_data nếu dài hơn
                benchmark_data = benchmark_data[:len(self.equity_df)]
            elif len(self.equity_df) > len(benchmark_data):
                # Báo lỗi nếu benchmark_data ngắn hơn
                st.warning("Dữ liệu benchmark ngắn hơn dữ liệu vốn. Sẽ chỉ so sánh phần chồng nhau.")
            
            benchmark_df = pd.DataFrame({
                'value': benchmark_data
            }, index=self.equity_df.index[:len(benchmark_data)])
            
        elif isinstance(benchmark_data, pd.DataFrame):
            # Đã là DataFrame
            benchmark_df = benchmark_data.copy()
            if 'value' not in benchmark_df.columns:
                # Giả định cột đầu tiên là giá trị
                benchmark_df = benchmark_df.rename(columns={benchmark_df.columns[0]: 'value'})
            
            # Đảm bảo index là datetime
            if not isinstance(benchmark_df.index, pd.DatetimeIndex):
                st.warning("Index của benchmark_data không phải là datetime. Sẽ sử dụng index của equity_df.")
                benchmark_df.index = self.equity_df.index[:len(benchmark_df)]
        else:
            st.error("Định dạng benchmark_data không được hỗ trợ.")
            return None
        
        # Lọc dữ liệu theo phạm vi thời gian nếu có
        equity_df = self.equity_df.copy()
        
        if date_range is not None:
            start_date, end_date = date_range
            equity_df = equity_df[(equity_df.index >= start_date) & (equity_df.index <= end_date)]
            benchmark_df = benchmark_df[(benchmark_df.index >= start_date) & (benchmark_df.index <= end_date)]
        
        # Đồng bộ hóa chỉ mục
        common_dates = equity_df.index.intersection(benchmark_df.index)
        if len(common_dates) == 0:
            st.error("Không có dữ liệu chồng nhau giữa equity và benchmark.")
            return None
        
        equity_df = equity_df.loc[common_dates]
        benchmark_df = benchmark_df.loc[common_dates]
        
        # Chuẩn hóa dữ liệu nếu cần
        if normalize:
            equity_normalized = equity_df["capital"] / equity_df["capital"].iloc[0] * 100
            benchmark_normalized = benchmark_df["value"] / benchmark_df["value"].iloc[0] * 100
        else:
            equity_normalized = equity_df["capital"]
            benchmark_normalized = benchmark_df["value"]
        
        # Tính các chỉ số so sánh
        strategy_return = (equity_df["capital"].iloc[-1] / equity_df["capital"].iloc[0]) - 1
        benchmark_return = (benchmark_df["value"].iloc[-1] / benchmark_df["value"].iloc[0]) - 1
        outperformance = strategy_return - benchmark_return
        
        # Tính returns hàng ngày
        strategy_daily_returns = equity_df["capital"].pct_change().dropna()
        benchmark_daily_returns = benchmark_df["value"].pct_change().dropna()
        
        # Tính beta
        cov = strategy_daily_returns.cov(benchmark_daily_returns)
        var = benchmark_daily_returns.var()
        beta = cov / var if var > 0 else 0
        
        # Tính alpha
        risk_free_rate = 0.02 / 252  # Giả định 2% hàng năm
        alpha = strategy_daily_returns.mean() - (risk_free_rate + beta * (benchmark_daily_returns.mean() - risk_free_rate))
        alpha_annualized = alpha * 252
        
        # Tính correlation
        correlation = strategy_daily_returns.corr(benchmark_daily_returns)
        
        if use_plotly:
            # Sử dụng Plotly để vẽ biểu đồ
            height = height or self.height
            width = width or self.width
            
            # Tạo subplots
            if include_metrics:
                fig = make_subplots(
                    rows=2, 
                    cols=2,
                    specs=[
                        [{"colspan": 2}, None],
                        [{"type": "bar"}, {"type": "bar"}]
                    ],
                    subplot_titles=(
                        "So sánh hiệu suất", 
                        "Lợi nhuận tổng",
                        "Chỉ số so sánh"
                    ),
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3]
                )
            else:
                fig = go.Figure()
            
            # 1. Biểu đồ đường so sánh hiệu suất
            fig.add_trace(
                go.Scatter(
                    x=equity_df.index,
                    y=equity_normalized,
                    mode="lines",
                    name=f"Chiến lược {self.strategy_name}" if self.strategy_name else "Chiến lược",
                    line=dict(color=self.colors["primary"], width=2)
                ),
                row=1 if include_metrics else None, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=benchmark_df.index,
                    y=benchmark_normalized,
                    mode="lines",
                    name=benchmark_name,
                    line=dict(color=self.colors["secondary"], width=2)
                ),
                row=1 if include_metrics else None, col=1
            )
            
            if include_metrics:
                # 2. Biểu đồ cột so sánh lợi nhuận tổng
                fig.add_trace(
                    go.Bar(
                        x=["Chiến lược", benchmark_name],
                        y=[strategy_return * 100, benchmark_return * 100],
                        marker=dict(
                            color=[self.colors["primary"], self.colors["secondary"]]
                        ),
                        text=[f"{strategy_return*100:.2f}%", f"{benchmark_return*100:.2f}%"],
                        textposition="auto"
                    ),
                    row=2, col=1
                )
                
                # 3. Biểu đồ cột các chỉ số so sánh
                fig.add_trace(
                    go.Bar(
                        x=["Alpha", "Beta", "Correlation", "Outperformance"],
                        y=[alpha_annualized * 100, beta, correlation, outperformance * 100],
                        marker=dict(color=self.colors["neutral"]),
                        text=[f"{alpha_annualized*100:.2f}%", f"{beta:.2f}", f"{correlation:.2f}", f"{outperformance*100:.2f}%"],
                        textposition="auto"
                    ),
                    row=2, col=2
                )
            
            # Cập nhật layout
            title = f"So sánh hiệu suất với {benchmark_name}" + (f" - {self.symbol}" if self.symbol else "")
            
            fig.update_layout(
                title=title,
                template="plotly" + ("_dark" if self.theme == "dark" else "_white"),
                height=height,
                width=width,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02 if include_metrics else 1.1,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified"
            )
            
            # Cập nhật trục
            if include_metrics:
                fig.update_xaxes(title_text="Thời gian", row=1, col=1)
                fig.update_yaxes(title_text="Giá trị" + (" (chuẩn hóa)" if normalize else ""), row=1, col=1)
                fig.update_yaxes(title_text="Lợi nhuận (%)", row=2, col=1)
            else:
                fig.update_xaxes(title_text="Thời gian")
                fig.update_yaxes(title_text="Giá trị" + (" (chuẩn hóa)" if normalize else ""))
            
            return fig
            
        else:
            # Sử dụng Matplotlib để vẽ biểu đồ
            if include_metrics:
                height = height or 10
                width = width or 15
                
                fig = plt.figure(figsize=(width, height))
                gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
                ax1 = fig.add_subplot(gs[0, :])
                ax2 = fig.add_subplot(gs[1, 0])
                ax3 = fig.add_subplot(gs[1, 1])
            else:
                height = height or 6
                width = width or 10
                
                fig, ax1 = plt.subplots(figsize=(width, height))
            
            # 1. Biểu đồ đường so sánh hiệu suất
            ax1.plot(
                equity_df.index, 
                equity_normalized,
                color=self.colors["primary"],
                linewidth=2,
                label=f"Chiến lược {self.strategy_name}" if self.strategy_name else "Chiến lược"
            )
            
            ax1.plot(
                benchmark_df.index, 
                benchmark_normalized,
                color=self.colors["secondary"],
                linewidth=2,
                label=benchmark_name
            )
            
            # Định dạng trục thời gian
            ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Thêm grid
            ax1.grid(alpha=0.3)
            
            # Thêm legend
            ax1.legend()
            
            # Thêm title và label
            ax1.set_title("So sánh hiệu suất")
            ax1.set_xlabel("Thời gian")
            ax1.set_ylabel("Giá trị" + (" (chuẩn hóa)" if normalize else ""))
            
            if include_metrics:
                # 2. Biểu đồ cột so sánh lợi nhuận tổng
                bars = ax2.bar(
                    ["Chiến lược", benchmark_name],
                    [strategy_return * 100, benchmark_return * 100],
                    color=[self.colors["primary"], self.colors["secondary"]]
                )
                
                # Thêm giá trị trên mỗi cột
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width()/2.,
                        height if height > 0 else height - 1,
                        f'{height:.2f}%',
                        ha='center', va='bottom' if height > 0 else 'top'
                    )
                
                ax2.set_title("Lợi nhuận tổng")
                ax2.set_ylabel("Lợi nhuận (%)")
                ax2.grid(alpha=0.3)
                
                # 3. Biểu đồ cột các chỉ số so sánh
                bars = ax3.bar(
                    ["Alpha", "Beta", "Correlation", "Outperf"],
                    [alpha_annualized * 100, beta, correlation, outperformance * 100],
                    color=self.colors["neutral"]
                )
                
                # Thêm giá trị trên mỗi cột
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(
                        bar.get_x() + bar.get_width()/2.,
                        height if height > 0 else height - 0.05,
                        f'{height:.2f}' + ("%" if bar.get_x() in [0, 3] else ""),
                        ha='center', va='bottom' if height > 0 else 'top'
                    )
                
                ax3.set_title("Chỉ số so sánh")
                ax3.grid(alpha=0.3)
            
            # Thêm title tổng
            title = f"So sánh hiệu suất với {benchmark_name}" + (f" - {self.symbol}" if self.symbol else "")
            plt.suptitle(title, fontsize=16)
            
            plt.tight_layout()
            return fig
    
    def create_performance_summary_dashboard(
        self,
        use_plotly: bool = True,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        benchmark_data: Optional[Union[pd.DataFrame, Dict[str, float], List[float]]] = None,
        benchmark_name: str = "Benchmark",
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> Dict[str, go.Figure]:
        """
        Tạo bảng tổng hợp các biểu đồ hiệu suất cho dashboard.
        
        Args:
            use_plotly: Sử dụng Plotly thay vì Matplotlib
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            benchmark_data: Dữ liệu benchmark nếu có
            benchmark_name: Tên benchmark
            height: Chiều cao của biểu đồ
            width: Chiều rộng của biểu đồ
            
        Returns:
            Dict chứa các đối tượng biểu đồ
        """
        # Kiểm tra dữ liệu
        if self.equity_df is None or len(self.equity_df) == 0:
            st.warning("Không có dữ liệu để tạo dashboard.")
            return {}
        
        # Tạo các biểu đồ
        dashboard = {}
        
        # 1. Đường cong vốn
        dashboard["equity_curve"] = self.plot_equity_curve(
            use_plotly=use_plotly,
            include_trades=True,
            include_drawdown=True,
            display_metrics=True,
            date_range=date_range,
            height=height,
            width=width
        )
        
        # 2. Phân phối lợi nhuận
        dashboard["returns_distribution"] = self.plot_returns_distribution(
            use_plotly=use_plotly,
            date_range=date_range,
            display_stats=True,
            height=height,
            width=width
        )
        
        # 3. Heatmap lợi nhuận theo tháng
        dashboard["monthly_returns"] = self.plot_monthly_returns_heatmap(
            use_plotly=use_plotly,
            height=height,
            width=width
        )
        
        # 4. Phân tích giao dịch
        if self.trades_df is not None and len(self.trades_df) > 0:
            dashboard["trade_analysis"] = self.plot_trade_analysis(
                use_plotly=use_plotly,
                breakdown_by="profit_loss",
                date_range=date_range,
                height=height,
                width=width
            )
        
        # 5. Drawdown
        dashboard["drawdown"] = self.plot_drawdown_chart(
            use_plotly=use_plotly,
            top_n=3,
            date_range=date_range,
            height=height,
            width=width
        )
        
        # 6. So sánh với benchmark nếu có
        if benchmark_data is not None:
            dashboard["benchmark_comparison"] = self.plot_performance_comparison(
                benchmark_data=benchmark_data,
                benchmark_name=benchmark_name,
                use_plotly=use_plotly,
                date_range=date_range,
                normalize=True,
                include_metrics=True,
                height=height,
                width=width
            )
        
        # 7. Phân bổ danh mục nếu có
        if self.portfolio_df is not None and len(self.portfolio_df) > 0:
            dashboard["portfolio_allocation"] = self.plot_portfolio_allocation(
                use_plotly=use_plotly,
                display_changes=True,
                height=height,
                width=width
            )
        
        return dashboard
    
    def export_performance_report(
        self,
        output_path: Optional[Union[str, Path]] = None,
        include_benchmark: bool = False,
        benchmark_data: Optional[Union[pd.DataFrame, Dict[str, float], List[float]]] = None,
        benchmark_name: str = "Benchmark",
        date_range: Optional[Tuple[datetime, datetime]] = None,
        format: str = "html"
    ) -> str:
        """
        Xuất báo cáo hiệu suất.
        
        Args:
            output_path: Đường dẫn đầu ra (None để tạo đường dẫn mặc định)
            include_benchmark: Có bao gồm so sánh với benchmark không
            benchmark_data: Dữ liệu benchmark nếu có
            benchmark_name: Tên benchmark
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            format: Định dạng báo cáo ("html", "pdf", "json")
            
        Returns:
            Đường dẫn tới file báo cáo
        """
        if self.equity_df is None or len(self.equity_df) == 0:
            st.error("Không có dữ liệu để xuất báo cáo.")
            return ""
        
        # Tạo đường dẫn mặc định nếu không được cung cấp
        if output_path is None:
            current_date = datetime.now().strftime("%Y%m%d")
            symbol_str = f"_{self.symbol.replace('/', '_')}" if self.symbol else ""
            strategy_str = f"_{self.strategy_name}" if self.strategy_name else ""
            filename = f"performance_report{symbol_str}{strategy_str}_{current_date}.{format}"
            
            output_path = Path("./reports") / filename
        else:
            output_path = Path(output_path)
        
        # Tạo thư mục nếu chưa tồn tại
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Tạo dashboard với tất cả các biểu đồ
        dashboard = self.create_performance_summary_dashboard(
            use_plotly=True,  # Sử dụng Plotly cho export
            date_range=date_range,
            benchmark_data=benchmark_data if include_benchmark else None,
            benchmark_name=benchmark_name
        )
        
        if format == "html":
            # Tạo báo cáo HTML
            import plotly.io as pio
            
            # Tính toán các chỉ số hiệu suất
            metrics = self._calculate_performance_metrics(date_range)
            
            # Tạo HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Báo cáo hiệu suất giao dịch</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: {'#121212' if self.theme == 'dark' else '#ffffff'};
                        color: {'#ffffff' if self.theme == 'dark' else '#000000'};
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                    }}
                    h1, h2, h3 {{
                        color: {'#ffffff' if self.theme == 'dark' else '#333333'};
                    }}
                    .header {{
                        text-align: center;
                        margin-bottom: 30px;
                    }}
                    .metrics-container {{
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-between;
                        margin-bottom: 30px;
                    }}
                    .metric-box {{
                        background-color: {'#1e1e1e' if self.theme == 'dark' else '#f5f5f5'};
                        border-radius: 5px;
                        padding: 15px;
                        margin-bottom: 15px;
                        width: calc(25% - 20px);
                        box-sizing: border-box;
                    }}
                    .metric-title {{
                        font-size: 14px;
                        margin-bottom: 5px;
                        color: {'#aaaaaa' if self.theme == 'dark' else '#666666'};
                    }}
                    .metric-value {{
                        font-size: 24px;
                        font-weight: bold;
                    }}
                    .positive {{
                        color: {self.colors["positive"]};
                    }}
                    .negative {{
                        color: {self.colors["negative"]};
                    }}
                    .chart-container {{
                        margin-bottom: 40px;
                    }}
                    .footer {{
                        text-align: center;
                        margin-top: 50px;
                        padding-top: 20px;
                        border-top: 1px solid {'#333333' if self.theme == 'dark' else '#dddddd'};
                        font-size: 12px;
                        color: {'#aaaaaa' if self.theme == 'dark' else '#888888'};
                    }}
                    @media (max-width: 768px) {{
                        .metric-box {{
                            width: calc(50% - 15px);
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Báo cáo hiệu suất giao dịch</h1>
                        <h2>{f"{self.symbol}" if self.symbol else ""} {f"- {self.strategy_name}" if self.strategy_name else ""}</h2>
                        <p>Thời gian báo cáo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div class="metrics-container">
            """
            
            # Thêm các metric vào báo cáo
            for key, value in metrics.items():
                if key in ["total_return", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "win_rate", 
                          "profit_factor", "expectancy", "max_drawdown", "roi"]:
                    
                    # Định dạng giá trị
                    formatted_value = ""
                    css_class = ""
                    
                    if key in ["total_return", "roi", "win_rate"]:
                        formatted_value = f"{value:.2f}%"
                        css_class = "positive" if value > 0 else "negative"
                    elif key == "max_drawdown":
                        formatted_value = f"{value:.2f}%"
                        css_class = "negative"
                    elif key == "profit_factor":
                        formatted_value = f"{value:.2f}x"
                        css_class = "positive" if value > 1 else "negative"
                    else:
                        formatted_value = f"{value:.2f}"
                        css_class = "positive" if value > 0 else "negative"
                    
                    # Định dạng tiêu đề
                    title_map = {
                        "total_return": "Tổng lợi nhuận",
                        "sharpe_ratio": "Sharpe Ratio",
                        "sortino_ratio": "Sortino Ratio",
                        "calmar_ratio": "Calmar Ratio",
                        "win_rate": "Win Rate",
                        "profit_factor": "Profit Factor",
                        "expectancy": "Expectancy",
                        "max_drawdown": "Max Drawdown",
                        "roi": "ROI"
                    }
                    
                    title = title_map.get(key, key)
                    
                    html_content += f"""
                    <div class="metric-box">
                        <div class="metric-title">{title}</div>
                        <div class="metric-value {css_class}">{formatted_value}</div>
                    </div>
                    """
            
            html_content += """
                    </div>
                    
                    <div class="charts">
            """
            
            # Thêm các biểu đồ
            for chart_name, fig in dashboard.items():
                if fig is not None:
                    chart_title_map = {
                        "equity_curve": "Đường cong vốn",
                        "returns_distribution": "Phân phối lợi nhuận",
                        "monthly_returns": "Lợi nhuận theo tháng",
                        "trade_analysis": "Phân tích giao dịch",
                        "drawdown": "Phân tích Drawdown",
                        "benchmark_comparison": f"So sánh với {benchmark_name}",
                        "portfolio_allocation": "Phân bổ danh mục"
                    }
                    
                    chart_title = chart_title_map.get(chart_name, chart_name)
                    
                    html_content += f"""
                        <div class="chart-container">
                            <h2>{chart_title}</h2>
                            <div id="{chart_name}_chart"></div>
                            <script>
                                var plotlyData = {pio.to_json(fig)};
                                Plotly.newPlot('{chart_name}_chart', plotlyData.data, plotlyData.layout);
                            </script>
                        </div>
                    """
            
            html_content += """
                    </div>
                    
                    <div class="footer">
                        <p>Báo cáo được tạo bởi hệ thống Automated Trading System</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Ghi file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            return str(output_path)
            
        elif format == "pdf":
            # Xuất báo cáo PDF
            try:
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import letter, A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                
                # Bước 1: Tạo file PDF
                doc = SimpleDocTemplate(str(output_path), pagesize=A4)
                styles = getSampleStyleSheet()
                
                # Tùy chỉnh style
                styles.add(ParagraphStyle(name='Title',
                                         parent=styles['Heading1'],
                                         alignment=1,  # Center
                                         fontSize=16))
                
                styles.add(ParagraphStyle(name='Subtitle',
                                        parent=styles['Heading2'],
                                        alignment=1,  # Center
                                        fontSize=14))
                
                styles.add(ParagraphStyle(name='Normal_Center',
                                         parent=styles['Normal'],
                                         alignment=1))
                
                # Bước 2: Tạo nội dung
                elements = []
                
                # Tiêu đề
                elements.append(Paragraph("Báo cáo hiệu suất giao dịch", styles['Title']))
                
                if self.symbol or self.strategy_name:
                    subtitle = ""
                    if self.symbol:
                        subtitle += self.symbol
                    if self.strategy_name:
                        subtitle += f" - {self.strategy_name}"
                    elements.append(Paragraph(subtitle, styles['Subtitle']))
                
                elements.append(Paragraph(f"Thời gian báo cáo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                         styles['Normal_Center']))
                elements.append(Spacer(1, 0.5*inch))
                
                # Tính toán metrics
                metrics = self._calculate_performance_metrics(date_range)
                
                # Thêm metrics vào bảng
                data = []
                data.append(["Chỉ số", "Giá trị"])
                
                metrics_to_show = [
                    ("total_return", "Tổng lợi nhuận", lambda x: f"{x:.2f}%"),
                    ("roi", "ROI", lambda x: f"{x:.2f}%"),
                    ("sharpe_ratio", "Sharpe Ratio", lambda x: f"{x:.2f}"),
                    ("sortino_ratio", "Sortino Ratio", lambda x: f"{x:.2f}"),
                    ("calmar_ratio", "Calmar Ratio", lambda x: f"{x:.2f}"),
                    ("win_rate", "Win Rate", lambda x: f"{x:.2f}%"),
                    ("profit_factor", "Profit Factor", lambda x: f"{x:.2f}x"),
                    ("expectancy", "Expectancy", lambda x: f"{x:.2f}"),
                    ("max_drawdown", "Max Drawdown", lambda x: f"{x:.2f}%")
                ]
                
                for key, title, formatter in metrics_to_show:
                    if key in metrics:
                        data.append([title, formatter(metrics[key])])
                
                # Tạo bảng metrics
                t = Table(data, colWidths=[2.5*inch, 1.5*inch])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(t)
                elements.append(Spacer(1, 0.5*inch))
                
                # Lưu các biểu đồ thành file tạm và thêm vào báo cáo
                import tempfile
                import os
                
                temp_dir = tempfile.gettempdir()
                
                for chart_name, fig in dashboard.items():
                    if fig is not None:
                        chart_title_map = {
                            "equity_curve": "Đường cong vốn",
                            "returns_distribution": "Phân phối lợi nhuận",
                            "monthly_returns": "Lợi nhuận theo tháng",
                            "trade_analysis": "Phân tích giao dịch",
                            "drawdown": "Phân tích Drawdown",
                            "benchmark_comparison": f"So sánh với {benchmark_name}",
                            "portfolio_allocation": "Phân bổ danh mục"
                        }
                        
                        chart_title = chart_title_map.get(chart_name, chart_name)
                        
                        # Lưu biểu đồ thành file tạm
                        temp_file = os.path.join(temp_dir, f"{chart_name}.png")
                        fig.write_image(temp_file, width=700, height=400)
                        
                        # Thêm tiêu đề biểu đồ
                        elements.append(Paragraph(chart_title, styles['Heading2']))
                        elements.append(Spacer(1, 0.2*inch))
                        
                        # Thêm hình ảnh
                        img = Image(temp_file, width=7*inch, height=4*inch)
                        elements.append(img)
                        elements.append(Spacer(1, 0.5*inch))
                
                # Thêm footer
                elements.append(Spacer(1, 0.5*inch))
                elements.append(Paragraph("Báo cáo được tạo bởi hệ thống Automated Trading System", 
                                         styles['Normal_Center']))
                
                # Tạo PDF
                doc.build(elements)
                
                # Xóa các file tạm
                for chart_name in dashboard.keys():
                    temp_file = os.path.join(temp_dir, f"{chart_name}.png")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                return str(output_path)
                
            except Exception as e:
                st.error(f"Không thể tạo báo cáo PDF: {str(e)}")
                # Thử xuất dưới dạng HTML nếu không tạo được PDF
                return self.export_performance_report(output_path.with_suffix(".html"), 
                                                     include_benchmark, benchmark_data, 
                                                     benchmark_name, date_range, "html")
        
        elif format == "json":
            # Xuất dữ liệu dưới dạng JSON
            metrics = self._calculate_performance_metrics(date_range)
            
            # Tạo cấu trúc dữ liệu
            report_data = {
                "metadata": {
                    "symbol": self.symbol,
                    "strategy_name": self.strategy_name,
                    "generated_at": datetime.now().isoformat(),
                    "date_range": {
                        "start": date_range[0].isoformat() if date_range and date_range[0] else None,
                        "end": date_range[1].isoformat() if date_range and date_range[1] else None
                    } if date_range else None
                },
                "metrics": metrics,
                "equity_data": self.equity_df["capital"].to_dict() if self.equity_df is not None else None,
                "trades_data": self.trades_df.to_dict('records') if self.trades_df is not None else None
            }
            
            # Thêm dữ liệu benchmark nếu có
            if include_benchmark and benchmark_data is not None:
                # Chuyển đổi benchmark_data thành dict
                if isinstance(benchmark_data, pd.DataFrame):
                    benchmark_dict = benchmark_data["value"].to_dict() if "value" in benchmark_data.columns else benchmark_data[benchmark_data.columns[0]].to_dict()
                elif isinstance(benchmark_data, dict):
                    benchmark_dict = benchmark_data
                else:  # list
                    benchmark_dict = {i: val for i, val in enumerate(benchmark_data)}
                
                report_data["benchmark"] = {
                    "name": benchmark_name,
                    "data": benchmark_dict
                }
            
            # Ghi file
            with open(output_path, "w", encoding="utf-8") as f:
                import json
                json.dump(report_data, f, indent=4, ensure_ascii=False)
            
            return str(output_path)
        
        else:
            st.error(f"Định dạng {format} không được hỗ trợ.")
            return ""
    
    def _calculate_performance_metrics(self, date_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, float]:
        """
        Tính toán các chỉ số hiệu suất.
        
        Args:
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            
        Returns:
            Dict chứa các chỉ số hiệu suất
        """
        if self.equity_df is None or len(self.equity_df) < 2:
            return {}
        
        # Lọc dữ liệu theo phạm vi thời gian nếu có
        equity_df = self.equity_df.copy()
        
        if date_range is not None:
            start_date, end_date = date_range
            equity_df = equity_df[(equity_df.index >= start_date) & (equity_df.index <= end_date)]
        
        # Tính lợi nhuận
        initial_capital = equity_df["capital"].iloc[0]
        final_capital = equity_df["capital"].iloc[-1]
        total_return = (final_capital / initial_capital - 1) * 100
        
        # Tính returns hàng ngày
        daily_returns = equity_df["capital"].pct_change().dropna()
        
        # Tính drawdown
        rolling_max = equity_df["capital"].cummax()
        drawdown = (equity_df["capital"] - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) * 100
        
        # Các chỉ số hiệu suất
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        
        # Sharpe Ratio (giả định risk-free rate = 2% hàng năm)
        risk_free_rate = 0.02 / 252  # Giả định 252 ngày giao dịch mỗi năm
        sharpe_ratio = (mean_return - risk_free_rate) / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Sortino Ratio (chỉ tính downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (mean_return - risk_free_rate) * 252 / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar Ratio
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365
        annualized_return = (1 + total_return / 100) ** (1 / years) - 1 if years > 0 else 0
        calmar_ratio = annualized_return / (max_drawdown / 100) if max_drawdown > 0 else 0
        
        # Win Rate và Profit Factor từ trades_df
        win_rate = 0
        profit_factor = 0
        expectancy = 0
        
        if self.trades_df is not None and len(self.trades_df) > 0:
            # Lọc giao dịch theo phạm vi thời gian
            trades_df = self.trades_df.copy()
            
            if date_range is not None:
                start_date, end_date = date_range
                trades_df = trades_df[(trades_df["timestamp"] >= start_date) & (trades_df["timestamp"] <= end_date)]
            
            if len(trades_df) > 0:
                # Win Rate
                winning_trades = trades_df[trades_df["profit_loss"] > 0]
                win_rate = len(winning_trades) / len(trades_df) * 100
                
                # Profit Factor
                total_profit = winning_trades["profit_loss"].sum() if len(winning_trades) > 0 else 0
                losing_trades = trades_df[trades_df["profit_loss"] <= 0]
                total_loss = abs(losing_trades["profit_loss"].sum()) if len(losing_trades) > 0 else 0
                
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                # Expectancy
                avg_win = winning_trades["profit_loss"].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades["profit_loss"].mean() if len(losing_trades) > 0 else 0
                
                win_probability = len(winning_trades) / len(trades_df)
                loss_probability = 1 - win_probability
                
                expectancy = (win_probability * avg_win) + (loss_probability * avg_loss)
        
        # Tạo dict kết quả
        metrics = {
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "roi": total_return,  # Alias
            "annualized_return": annualized_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "volatility": std_return * 100 * np.sqrt(252),  # Annualized
            "mean_daily_return": mean_return * 100,
            "trading_days": len(daily_returns)
        }
        
        return metrics


class StreamlitPerformanceCharts(PerformanceCharts):
    """
    Lớp mở rộng từ PerformanceCharts, tích hợp trực tiếp với Streamlit.
    
    Cung cấp các phương thức để hiển thị biểu đồ hiệu suất trực tiếp 
    trong Streamlit dashboard mà không cần phải xử lý riêng biệt.
    """
    
    def __init__(
        self,
        metrics_data: Optional[Union[Dict, pd.DataFrame, TradingMetricsTracker, MultiSymbolTradingMetricsTracker]] = None,
        symbol: Optional[str] = None,
        strategy_name: Optional[str] = None,
        theme: str = "streamlit"
    ):
        """
        Khởi tạo đối tượng StreamlitPerformanceCharts.
        
        Args:
            metrics_data: Dữ liệu số liệu
            symbol: Cặp giao dịch
            strategy_name: Tên chiến lược
            theme: Theme màu
        """
        super().__init__(metrics_data, symbol, strategy_name, theme)
    
    def display_equity_curve(
        self,
        include_trades: bool = True,
        include_drawdown: bool = True,
        display_metrics: bool = True,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        use_container_width: bool = True
    ):
        """
        Hiển thị đường cong vốn trong Streamlit.
        
        Args:
            include_trades: Hiển thị các giao dịch trên biểu đồ
            include_drawdown: Hiển thị drawdown
            display_metrics: Hiển thị các chỉ số performance
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            use_container_width: Sử dụng chiều rộng container
        """
        st.subheader("Đường cong vốn")
        
        fig = self.plot_equity_curve(
            use_plotly=True,
            include_trades=include_trades,
            include_drawdown=include_drawdown,
            display_metrics=display_metrics,
            date_range=date_range
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=use_container_width)
        else:
            st.warning("Không có đủ dữ liệu để hiển thị đường cong vốn.")
    
    def display_drawdown_chart(
        self,
        top_n: int = 3,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        use_container_width: bool = True
    ):
        """
        Hiển thị biểu đồ drawdown trong Streamlit.
        
        Args:
            top_n: Số lượng giai đoạn drawdown lớn nhất để hiển thị
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            use_container_width: Sử dụng chiều rộng container
        """
        st.subheader("Phân tích Drawdown")
        
        fig = self.plot_drawdown_chart(
            use_plotly=True,
            top_n=top_n,
            date_range=date_range
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=use_container_width)
        else:
            st.warning("Không có đủ dữ liệu để hiển thị biểu đồ drawdown.")
    
    def display_returns_distribution(
        self,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        bins: int = 50,
        display_stats: bool = True,
        use_container_width: bool = True
    ):
        """
        Hiển thị phân phối lợi nhuận trong Streamlit.
        
        Args:
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            bins: Số lượng bins cho histogram
            display_stats: Hiển thị các thống kê
            use_container_width: Sử dụng chiều rộng container
        """
        st.subheader("Phân phối lợi nhuận")
        
        fig = self.plot_returns_distribution(
            use_plotly=True,
            date_range=date_range,
            bins=bins,
            display_stats=display_stats
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=use_container_width)
        else:
            st.warning("Không có đủ dữ liệu để hiển thị phân phối lợi nhuận.")
    
    def display_monthly_returns_heatmap(
        self,
        use_container_width: bool = True
    ):
        """
        Hiển thị heatmap lợi nhuận hàng tháng trong Streamlit.
        
        Args:
            use_container_width: Sử dụng chiều rộng container
        """
        st.subheader("Lợi nhuận theo tháng")
        
        fig = self.plot_monthly_returns_heatmap(use_plotly=True)
        
        if fig:
            st.plotly_chart(fig, use_container_width=use_container_width)
        else:
            st.warning("Không có đủ dữ liệu để hiển thị heatmap lợi nhuận hàng tháng.")
    
    def display_trade_analysis(
        self,
        breakdown_by: str = "profit_loss",
        trade_count: int = 100,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        use_container_width: bool = True
    ):
        """
        Hiển thị phân tích giao dịch trong Streamlit.
        
        Args:
            breakdown_by: Phân tích theo tiêu chí ('profit_loss', 'duration', 'reason')
            trade_count: Số lượng giao dịch gần nhất để phân tích
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            use_container_width: Sử dụng chiều rộng container
        """
        st.subheader("Phân tích giao dịch")
        
        # Tạo radio button để chọn loại phân tích
        breakdown_options = {
            "profit_loss": "Lợi nhuận/Lỗ",
            "duration": "Thời gian nắm giữ",
            "reason": "Lý do đóng vị thế"
        }
        
        selected_breakdown = st.radio(
            "Phân tích theo:",
            options=list(breakdown_options.keys()),
            format_func=lambda x: breakdown_options[x],
            key="trade_analysis_breakdown"
        )
        
        fig = self.plot_trade_analysis(
            use_plotly=True,
            breakdown_by=selected_breakdown,
            trade_count=trade_count,
            date_range=date_range
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=use_container_width)
        else:
            st.warning("Không có dữ liệu giao dịch để phân tích.")
    
    def display_portfolio_allocation(
        self,
        display_changes: bool = True,
        date: Optional[datetime] = None,
        use_container_width: bool = True
    ):
        """
        Hiển thị phân bổ danh mục trong Streamlit.
        
        Args:
            display_changes: Hiển thị thay đổi phân bổ theo thời gian
            date: Ngày cụ thể để hiển thị phân bổ (None để hiển thị gần nhất)
            use_container_width: Sử dụng chiều rộng container
        """
        st.subheader("Phân bổ danh mục")
        
        fig = self.plot_portfolio_allocation(
            use_plotly=True,
            display_changes=display_changes,
            date=date
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=use_container_width)
        else:
            st.warning("Không có dữ liệu danh mục để phân tích.")
    
    def display_performance_comparison(
        self,
        benchmark_data: Union[pd.DataFrame, Dict[str, float], List[float]],
        benchmark_name: str = "Benchmark",
        date_range: Optional[Tuple[datetime, datetime]] = None,
        normalize: bool = True,
        include_metrics: bool = True,
        use_container_width: bool = True
    ):
        """
        Hiển thị so sánh hiệu suất trong Streamlit.
        
        Args:
            benchmark_data: Dữ liệu benchmark
            benchmark_name: Tên benchmark
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            normalize: Chuẩn hóa dữ liệu về 100 tại thời điểm bắt đầu
            include_metrics: Hiển thị các chỉ số so sánh
            use_container_width: Sử dụng chiều rộng container
        """
        st.subheader(f"So sánh với {benchmark_name}")
        
        fig = self.plot_performance_comparison(
            benchmark_data=benchmark_data,
            benchmark_name=benchmark_name,
            use_plotly=True,
            date_range=date_range,
            normalize=normalize,
            include_metrics=include_metrics
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=use_container_width)
        else:
            st.warning(f"Không có đủ dữ liệu để so sánh với {benchmark_name}.")
    
    def display_metrics_summary(self, date_range: Optional[Tuple[datetime, datetime]] = None):
        """
        Hiển thị tóm tắt các chỉ số hiệu suất trong Streamlit.
        
        Args:
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
        """
        st.subheader("Tóm tắt hiệu suất")
        
        metrics = self._calculate_performance_metrics(date_range)
        
        if not metrics:
            st.warning("Không có đủ dữ liệu để hiển thị các chỉ số hiệu suất.")
            return
        
        # Tạo layout 3 cột
        col1, col2, col3 = st.columns(3)
        
        # Các chỉ số chính
        col1.metric(
            "Tổng lợi nhuận",
            f"{metrics.get('total_return', 0):.2f}%",
            delta=None
        )
        
        col2.metric(
            "Win Rate",
            f"{metrics.get('win_rate', 0):.2f}%",
            delta=None
        )
        
        col3.metric(
            "Profit Factor",
            f"{metrics.get('profit_factor', 0):.2f}x",
            delta=None
        )
        
        # Tạo layout 3 cột khác
        col4, col5, col6 = st.columns(3)
        
        col4.metric(
            "Sharpe Ratio",
            f"{metrics.get('sharpe_ratio', 0):.2f}",
            delta=None
        )
        
        col5.metric(
            "Max Drawdown",
            f"{metrics.get('max_drawdown', 0):.2f}%",
            delta=None
        )
        
        col6.metric(
            "Expectancy",
            f"{metrics.get('expectancy', 0):.2f}",
            delta=None
        )
        
        # Hiển thị các chỉ số khác trong expander
        with st.expander("Xem thêm chỉ số"):
            # Tạo layout 2 cột
            col7, col8 = st.columns(2)
            
            col7.metric(
                "Annualized Return",
                f"{metrics.get('annualized_return', 0):.2f}%",
                delta=None
            )
            
            col8.metric(
                "Volatility (Annualized)",
                f"{metrics.get('volatility', 0):.2f}%",
                delta=None
            )
            
            col7.metric(
                "Sortino Ratio",
                f"{metrics.get('sortino_ratio', 0):.2f}",
                delta=None
            )
            
            col8.metric(
                "Calmar Ratio",
                f"{metrics.get('calmar_ratio', 0):.2f}",
                delta=None
            )
            
            col7.metric(
                "Số ngày giao dịch",
                f"{metrics.get('trading_days', 0)}",
                delta=None
            )
            
            col8.metric(
                "Lợi nhuận trung bình hàng ngày",
                f"{metrics.get('mean_daily_return', 0):.2f}%",
                delta=None
            )
    
    def display_full_dashboard(
        self,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        benchmark_data: Optional[Union[pd.DataFrame, Dict[str, float], List[float]]] = None,
        benchmark_name: str = "Benchmark"
    ):
        """
        Hiển thị đầy đủ dashboard hiệu suất trong Streamlit.
        
        Args:
            date_range: Phạm vi thời gian (bắt đầu, kết thúc)
            benchmark_data: Dữ liệu benchmark
            benchmark_name: Tên benchmark
        """
        # Tiêu đề
        title = "Dashboard hiệu suất giao dịch"
        if self.symbol:
            title += f" - {self.symbol}"
        if self.strategy_name:
            title += f" ({self.strategy_name})"
        
        st.title(title)
        
        # Kiểm tra dữ liệu
        if self.equity_df is None or len(self.equity_df) == 0:
            st.warning("Không có dữ liệu để hiển thị dashboard.")
            return
        
        # Hiển thị phạm vi thời gian nếu có
        if date_range:
            start_date, end_date = date_range
            st.caption(f"Phạm vi thời gian: {start_date.strftime('%Y-%m-%d')} đến {end_date.strftime('%Y-%m-%d')}")
        
        # 1. Hiển thị tóm tắt các chỉ số hiệu suất
        self.display_metrics_summary(date_range)
        
        # 2. Hiển thị đường cong vốn
        self.display_equity_curve(
            include_trades=True,
            include_drawdown=True,
            display_metrics=True,
            date_range=date_range
        )
        
        # Tạo tabs cho các biểu đồ khác
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Phân tích giao dịch", 
            "Drawdown", 
            "Phân phối lợi nhuận", 
            "Lợi nhuận theo tháng",
            "Danh mục"
        ])
        
        # Tab 1: Phân tích giao dịch
        with tab1:
            if self.trades_df is not None and len(self.trades_df) > 0:
                self.display_trade_analysis(
                    breakdown_by="profit_loss",
                    date_range=date_range
                )
            else:
                st.warning("Không có dữ liệu giao dịch để phân tích.")
        
        # Tab 2: Drawdown
        with tab2:
            self.display_drawdown_chart(
                top_n=3,
                date_range=date_range
            )
        
        # Tab 3: Phân phối lợi nhuận
        with tab3:
            self.display_returns_distribution(
                date_range=date_range,
                display_stats=True
            )
        
        # Tab 4: Lợi nhuận theo tháng
        with tab4:
            self.display_monthly_returns_heatmap()
        
        # Tab 5: Danh mục
        with tab5:
            if self.portfolio_df is not None and len(self.portfolio_df) > 0:
                self.display_portfolio_allocation(display_changes=True)
            else:
                st.warning("Không có dữ liệu danh mục để phân tích.")
        
        # Hiển thị so sánh với benchmark nếu có
        if benchmark_data is not None:
            st.markdown("---")
            self.display_performance_comparison(
                benchmark_data=benchmark_data,
                benchmark_name=benchmark_name,
                date_range=date_range,
                normalize=True,
                include_metrics=True
            )
        
        # Hiển thị nút xuất báo cáo
        st.markdown("---")
        export_col1, export_col2 = st.columns([1, 3])
        
        report_format = export_col1.selectbox(
            "Định dạng báo cáo:",
            options=["html", "pdf", "json"],
            index=0
        )
        
        if export_col2.button("Xuất báo cáo"):
            with st.spinner("Đang tạo báo cáo..."):
                report_path = self.export_performance_report(
                    output_path=None,
                    include_benchmark=benchmark_data is not None,
                    benchmark_data=benchmark_data,
                    benchmark_name=benchmark_name,
                    date_range=date_range,
                    format=report_format
                )
                
                if report_path:
                    # Đọc file báo cáo
                    with open(report_path, "rb") as f:
                        report_data = f.read()
                    
                    # Hiển thị nút tải xuống
                    st.download_button(
                        label=f"Tải xuống báo cáo ({report_format.upper()})",
                        data=report_data,
                        file_name=os.path.basename(report_path),
                        mime={"html": "text/html", "pdf": "application/pdf", "json": "application/json"}[report_format]
                    )
                else:
                    st.error("Không thể tạo báo cáo.")


def load_metrics_from_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Hàm tiện ích để tải dữ liệu metrics từ file JSON.
    
    Args:
        file_path: Đường dẫn tới file metrics
        
    Returns:
        Dict chứa dữ liệu metrics
    """
    try:
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
        
    except Exception as e:
        st.error(f"Không thể tải dữ liệu metrics từ file: {str(e)}")
        return {}


def create_performance_charts(
    metrics_data: Optional[Union[Dict, pd.DataFrame, TradingMetricsTracker, MultiSymbolTradingMetricsTracker, str, Path]] = None,
    symbol: Optional[str] = None,
    strategy_name: Optional[str] = None,
    theme: str = "streamlit",
    for_streamlit: bool = False
) -> Union[PerformanceCharts, StreamlitPerformanceCharts]:
    """
    Hàm tiện ích để tạo đối tượng PerformanceCharts.
    
    Args:
        metrics_data: Dữ liệu metrics
        symbol: Cặp giao dịch
        strategy_name: Tên chiến lược
        theme: Theme màu
        for_streamlit: Tạo đối tượng StreamlitPerformanceCharts nếu True
        
    Returns:
        Đối tượng PerformanceCharts hoặc StreamlitPerformanceCharts
    """
    if for_streamlit:
        return StreamlitPerformanceCharts(metrics_data, symbol, strategy_name, theme)
    else:
        return PerformanceCharts(metrics_data, symbol, strategy_name, theme)