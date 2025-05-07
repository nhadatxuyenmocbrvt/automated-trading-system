"""
Hiển thị các chỉ số và số liệu thống kê cho dashboard Streamlit.
File này cung cấp các chức năng tạo và hiển thị các chỉ số hiệu suất,
rủi ro, huấn luyện và giao dịch trên giao diện dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import datetime
from pathlib import Path
import json
import altair as alt
import base64
from io import BytesIO

# Import các module từ hệ thống
from config.constants import BacktestMetric, OrderStatus, PositionSide, OrderType
from backtesting.evaluation.risk_evaluator import RiskEvaluator
from backtesting.evaluation.performance_evaluator import PerformanceEvaluator
from logs.metrics.training_metrics import TrainingMetricsTracker
from logs.metrics.trading_metrics import TradingMetricsTracker
from logs.logger import get_system_logger

# Khởi tạo logger
logger = get_system_logger("metrics_display")


class MetricsDisplay:
    """
    Lớp hiển thị các chỉ số (metrics) trên dashboard.
    Cung cấp các phương thức để hiển thị các loại chỉ số khác nhau
    với nhiều định dạng hiển thị: card, gauge, chart...
    """
    
    def __init__(self, 
                theme: str = "light",
                color_scheme: Optional[Dict[str, str]] = None):
        """
        Khởi tạo MetricsDisplay.
        
        Args:
            theme: Chủ đề hiển thị ('light' hoặc 'dark')
            color_scheme: Dict ánh xạ loại dữ liệu đến màu sắc
        """
        self.theme = theme
        
        # Màu sắc mặc định
        self.color_scheme = color_scheme or {
            "profit": "#0ECB81",         # Xanh lá
            "loss": "#F6465D",           # Đỏ
            "neutral": "#9DA8AF",        # Xám
            "primary": "#1E88E5",        # Xanh dương
            "secondary": "#7B1FA2",      # Tím
            "warning": "#FF9800",        # Cam
            "info": "#00B8D4",           # Xanh lơ
            "background": "#FFFFFF",     # Trắng
            "text": "#1E2026",           # Đen đậm
            "gradient_start": "#00B8D4", # Xanh lơ
            "gradient_end": "#1E88E5"    # Xanh dương
        }
        
        # Điều chỉnh màu sắc dựa trên theme
        if theme == "dark":
            self.color_scheme.update({
                "background": "#1E2026",
                "text": "#EAECEF"
            })
    
    def display_metric_card(self, 
                           title: str, 
                           value: Union[float, int, str], 
                           suffix: str = "",
                           prefix: str = "",
                           delta: Optional[float] = None,
                           delta_suffix: str = "%",
                           color: Optional[str] = None,
                           help_text: Optional[str] = None,
                           format_string: Optional[str] = None) -> None:
        """
        Hiển thị một chỉ số dạng card.
        
        Args:
            title: Tiêu đề
            value: Giá trị
            suffix: Hậu tố (%, $, etc.)
            prefix: Tiền tố ($, etc.)
            delta: Giá trị thay đổi
            delta_suffix: Hậu tố cho delta
            color: Màu sắc (None để tự động)
            help_text: Văn bản trợ giúp
            format_string: Định dạng cho giá trị (ví dụ: "{:.2f}")
        """
        # Định dạng giá trị nếu cần
        if isinstance(value, (int, float)) and format_string:
            formatted_value = format_string.format(value)
        elif isinstance(value, float):
            formatted_value = f"{value:.2f}"
        else:
            formatted_value = str(value)
            
        # Thêm prefix và suffix
        display_value = f"{prefix}{formatted_value}{suffix}"
        
        # Xác định màu sắc
        if color is None:
            if isinstance(value, (int, float)) and delta is not None:
                color = self.color_scheme["profit"] if delta > 0 else (
                    self.color_scheme["loss"] if delta < 0 else self.color_scheme["neutral"]
                )
            else:
                color = self.color_scheme["primary"]
        
        # Hiển thị metric với delta nếu có
        if delta is not None:
            st.metric(
                label=title,
                value=display_value,
                delta=f"{delta:.2f}{delta_suffix}",
                delta_color="normal",
                help=help_text
            )
        else:
            st.metric(
                label=title,
                value=display_value,
                help=help_text
            )
    
    def display_metric_gauge(self,
                           title: str,
                           value: float,
                           min_value: float = 0.0,
                           max_value: float = 1.0,
                           green_threshold: float = 0.7,
                           yellow_threshold: float = 0.4,
                           suffix: str = "%",
                           help_text: Optional[str] = None) -> None:
        """
        Hiển thị một chỉ số dạng đồng hồ đo.
        
        Args:
            title: Tiêu đề
            value: Giá trị (0-1 hoặc tỷ lệ phần trăm)
            min_value: Giá trị tối thiểu
            max_value: Giá trị tối đa
            green_threshold: Ngưỡng cho màu xanh
            yellow_threshold: Ngưỡng cho màu vàng
            suffix: Hậu tố (%, $, etc.)
            help_text: Văn bản trợ giúp
        """
        # Chuẩn hóa giá trị nếu cần
        normalized_value = value
        if max_value != 1.0:
            normalized_value = (value - min_value) / (max_value - min_value)
        
        # Giới hạn giá trị từ 0 đến 1
        normalized_value = max(0, min(1, normalized_value))
        
        # Xác định màu sắc
        if normalized_value >= green_threshold:
            color = self.color_scheme["profit"]
        elif normalized_value >= yellow_threshold:
            color = self.color_scheme["warning"]
        else:
            color = self.color_scheme["loss"]
        
        # Hiển thị tiêu đề
        st.subheader(title)
        if help_text:
            st.caption(help_text)
        
        # Tạo biểu đồ với Plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={
                'suffix': suffix,
                'font': {'color': color, 'size': 24}
            },
            gauge={
                'axis': {'range': [min_value, max_value]},
                'bar': {'color': color},
                'steps': [
                    {'range': [min_value, min_value + (max_value - min_value) * yellow_threshold], 'color': self.color_scheme["loss"]},
                    {'range': [min_value + (max_value - min_value) * yellow_threshold, 
                             min_value + (max_value - min_value) * green_threshold], 'color': self.color_scheme["warning"]},
                    {'range': [min_value + (max_value - min_value) * green_threshold, max_value], 'color': self.color_scheme["profit"]}
                ],
                'threshold': {
                    'line': {'color': color, 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        # Điều chỉnh layout
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": self.color_scheme["text"]}
        )
        
        # Hiển thị
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    def display_metric_progress(self,
                              title: str,
                              value: float,
                              min_value: float = 0.0,
                              max_value: float = 1.0,
                              green_threshold: float = 0.7,
                              yellow_threshold: float = 0.4,
                              help_text: Optional[str] = None) -> None:
        """
        Hiển thị một chỉ số dạng thanh tiến trình.
        
        Args:
            title: Tiêu đề
            value: Giá trị
            min_value: Giá trị tối thiểu
            max_value: Giá trị tối đa
            green_threshold: Ngưỡng cho màu xanh
            yellow_threshold: Ngưỡng cho màu vàng
            help_text: Văn bản trợ giúp
        """
        # Chuẩn hóa giá trị
        normalized_value = (value - min_value) / (max_value - min_value)
        normalized_value = max(0, min(1, normalized_value))
        
        # Xác định màu sắc
        if normalized_value >= green_threshold:
            color = self.color_scheme["profit"]
        elif normalized_value >= yellow_threshold:
            color = self.color_scheme["warning"]
        else:
            color = self.color_scheme["loss"]
        
        # Hiển thị tiêu đề
        st.subheader(title)
        if help_text:
            st.caption(help_text)
        
        # Hiển thị progress bar
        st.progress(normalized_value)
        
        # Hiển thị giá trị
        st.markdown(f"<p style='text-align: right; color: {color};'>{value:.2f} / {max_value:.2f}</p>", 
                  unsafe_allow_html=True)
    
    def display_performance_metrics(self, metrics: Dict[str, Any], columns: int = 3) -> None:
        """
        Hiển thị nhiều chỉ số hiệu suất cùng một lúc.
        
        Args:
            metrics: Dict chứa các chỉ số
            columns: Số cột để hiển thị
        """
        # Tạo layout cột
        cols = st.columns(columns)
        
        # Định nghĩa các chỉ số được hiển thị
        display_metrics = {
            BacktestMetric.TOTAL_RETURN.value: {"title": "Lợi nhuận", "suffix": "%", "format": "{:.2f}"},
            BacktestMetric.ANNUALIZED_RETURN.value: {"title": "Lợi nhuận hàng năm", "suffix": "%", "format": "{:.2f}"},
            BacktestMetric.SHARPE_RATIO.value: {"title": "Sharpe Ratio", "format": "{:.2f}"},
            BacktestMetric.SORTINO_RATIO.value: {"title": "Sortino Ratio", "format": "{:.2f}"},
            BacktestMetric.CALMAR_RATIO.value: {"title": "Calmar Ratio", "format": "{:.2f}"},
            BacktestMetric.MAX_DRAWDOWN.value: {"title": "Max Drawdown", "suffix": "%", "format": "{:.2f}"},
            BacktestMetric.WIN_RATE.value: {"title": "Win Rate", "suffix": "%", "format": "{:.2f}"},
            BacktestMetric.PROFIT_FACTOR.value: {"title": "Profit Factor", "format": "{:.2f}"},
            BacktestMetric.EXPECTANCY.value: {"title": "Expectancy", "format": "{:.2f}"},
            "volatility": {"title": "Biến động", "suffix": "%", "format": "{:.2f}"},
            "roi": {"title": "ROI", "suffix": "%", "format": "{:.2f}"}
        }
        
        # Hiển thị các chỉ số
        i = 0
        for key, value in metrics.items():
            if key in display_metrics:
                metric_info = display_metrics[key]
                
                # Chuyển đổi giá trị phần trăm nếu cần
                display_value = value
                if key in [BacktestMetric.TOTAL_RETURN.value, 
                           BacktestMetric.ANNUALIZED_RETURN.value,
                           BacktestMetric.MAX_DRAWDOWN.value, 
                           BacktestMetric.WIN_RATE.value,
                           "volatility", "roi"] and isinstance(value, (int, float)):
                    display_value = value * 100
                
                with cols[i % columns]:
                    self.display_metric_card(
                        title=metric_info["title"],
                        value=display_value,
                        suffix=metric_info.get("suffix", ""),
                        format_string=metric_info.get("format")
                    )
                i += 1
        
        # Thêm các card trống nếu không chia đều được
        remaining = columns - (i % columns)
        if remaining < columns:
            for _ in range(remaining):
                with cols[(i + _) % columns]:
                    st.write("")
    
    def display_risk_metrics(self, risk_metrics: Dict[str, Any], columns: int = 3) -> None:
        """
        Hiển thị các chỉ số rủi ro.
        
        Args:
            risk_metrics: Dict chứa các chỉ số rủi ro
            columns: Số cột để hiển thị
        """
        # Tạo layout cột
        cols = st.columns(columns)
        
        # Định nghĩa các chỉ số được hiển thị
        display_metrics = {
            "var": {"title": "Value at Risk (95%)", "suffix": "%", "format": "{:.2f}"},
            "es": {"title": "Expected Shortfall", "suffix": "%", "format": "{:.2f}"},
            "max_drawdown": {"title": "Max Drawdown", "suffix": "%", "format": "{:.2f}"},
            "volatility": {"title": "Biến động", "suffix": "%", "format": "{:.2f}"},
            "downside_volatility": {"title": "Downside Vol", "suffix": "%", "format": "{:.2f}"},
            "volatility_ratio": {"title": "Vol Ratio", "format": "{:.2f}"},
            "skewness": {"title": "Skewness", "format": "{:.2f}"},
            "kurtosis": {"title": "Kurtosis", "format": "{:.2f}"},
            "tail_ratio": {"title": "Tail Ratio", "format": "{:.2f}"},
            "ulcer_index": {"title": "Ulcer Index", "format": "{:.4f}"},
            "pain_index": {"title": "Pain Index", "format": "{:.4f}"},
            "pain_ratio": {"title": "Pain Ratio", "format": "{:.2f}"}
        }
        
        # Hiển thị các chỉ số
        i = 0
        for key, metric_info in display_metrics.items():
            # Lấy giá trị, có thể ở nhiều cấp trong dictionary
            value = None
            
            # Xử lý các trường hợp đặc biệt
            if key == "var" and "var" in risk_metrics and "historical" in risk_metrics["var"]:
                value = risk_metrics["var"]["historical"] * 100
            elif key in risk_metrics:
                value = risk_metrics[key]
                # Chuyển đổi phần trăm
                if key in ["max_drawdown", "volatility", "downside_volatility", "es"] and isinstance(value, (int, float)):
                    value = value * 100
            
            # Chỉ hiển thị nếu có giá trị
            if value is not None:
                with cols[i % columns]:
                    self.display_metric_card(
                        title=metric_info["title"],
                        value=value,
                        suffix=metric_info.get("suffix", ""),
                        format_string=metric_info.get("format")
                    )
                i += 1
        
        # Thêm các card trống nếu không chia đều được
        remaining = columns - (i % columns)
        if remaining < columns:
            for _ in range(remaining):
                with cols[(i + _) % columns]:
                    st.write("")
    
    def display_trading_stats(self, trading_stats: Dict[str, Any], columns: int = 3) -> None:
        """
        Hiển thị các thống kê giao dịch.
        
        Args:
            trading_stats: Dict chứa các thống kê giao dịch
            columns: Số cột để hiển thị
        """
        # Tạo layout cột
        cols = st.columns(columns)
        
        # Định nghĩa các chỉ số được hiển thị
        display_metrics = {
            "total_trades": {"title": "Tổng số giao dịch", "format": "{:d}"},
            "winning_trades": {"title": "Giao dịch thắng", "format": "{:d}"},
            "losing_trades": {"title": "Giao dịch thua", "format": "{:d}"},
            "win_rate": {"title": "Win Rate", "suffix": "%", "format": "{:.2f}"},
            "avg_profit": {"title": "Lãi trung bình", "format": "{:.2f}"},
            "avg_loss": {"title": "Lỗ trung bình", "format": "{:.2f}"},
            "profit_factor": {"title": "Profit Factor", "format": "{:.2f}"},
            "avg_trade_duration": {"title": "Thời gian giao dịch TB", "suffix": "h", "format": "{:.2f}"},
            "consecutive_wins": {"title": "Thắng liên tiếp max", "format": "{:d}"},
            "consecutive_losses": {"title": "Thua liên tiếp max", "format": "{:d}"},
            "largest_win": {"title": "Lãi lớn nhất", "format": "{:.2f}"},
            "largest_loss": {"title": "Lỗ lớn nhất", "format": "{:.2f}"},
            "total_fees": {"title": "Tổng phí", "format": "{:.2f}"}
        }
        
        # Hiển thị các chỉ số
        i = 0
        for key, metric_info in display_metrics.items():
            # Lấy giá trị, có thể nằm ở các vị trí khác nhau trong dict
            value = None
            
            if key in trading_stats:
                value = trading_stats[key]
            elif key == "consecutive_wins" and "trade_stats" in trading_stats:
                value = trading_stats["trade_stats"].get("consecutive_wins")
            elif key == "consecutive_losses" and "trade_stats" in trading_stats:
                value = trading_stats["trade_stats"].get("consecutive_losses")
            elif key == "largest_win" and "trade_stats" in trading_stats:
                value = trading_stats["trade_stats"].get("max_profit")
            elif key == "largest_loss" and "trade_stats" in trading_stats:
                value = trading_stats["trade_stats"].get("max_loss")
            
            # Chuyển đổi giá trị phần trăm nếu cần
            if key == "win_rate" and value is not None and isinstance(value, (int, float)):
                value = value * 100
            
            # Chỉ hiển thị nếu có giá trị
            if value is not None:
                with cols[i % columns]:
                    self.display_metric_card(
                        title=metric_info["title"],
                        value=value,
                        suffix=metric_info.get("suffix", ""),
                        format_string=metric_info.get("format")
                    )
                i += 1
        
        # Thêm các card trống nếu không chia đều được
        remaining = columns - (i % columns)
        if remaining < columns:
            for _ in range(remaining):
                with cols[(i + _) % columns]:
                    st.write("")
    
    def display_training_stats(self, training_stats: Dict[str, Any], columns: int = 3) -> None:
        """
        Hiển thị các thống kê huấn luyện.
        
        Args:
            training_stats: Dict chứa các thống kê huấn luyện
            columns: Số cột để hiển thị
        """
        # Tạo layout cột
        cols = st.columns(columns)
        
        # Định nghĩa các chỉ số được hiển thị
        display_metrics = {
            "mean_reward": {"title": "Phần thưởng TB", "format": "{:.2f}"},
            "max_reward": {"title": "Phần thưởng max", "format": "{:.2f}"},
            "mean_last_10": {"title": "Phần thưởng TB (10 cuối)", "format": "{:.2f}"},
            "mean_last_100": {"title": "Phần thưởng TB (100 cuối)", "format": "{:.2f}"},
            "total_episodes": {"title": "Tổng số episode", "format": "{:d}"},
            "total_steps": {"title": "Tổng số step", "format": "{:d}"},
            "win_rate": {"title": "Win Rate", "suffix": "%", "format": "{:.2f}"},
            "mean_win_rate": {"title": "Win Rate TB", "suffix": "%", "format": "{:.2f}"},
            "mean_loss": {"title": "Loss TB", "format": "{:.4f}"},
            "final_win_rate": {"title": "Win Rate cuối", "suffix": "%", "format": "{:.2f}"}
        }
        
        # Hiển thị các chỉ số
        i = 0
        for key, metric_info in display_metrics.items():
            # Lấy giá trị
            value = None
            
            # Tìm kiếm ở nhiều vị trí trong dict
            if key in training_stats:
                value = training_stats[key]
            elif "summary" in training_stats and key in training_stats["summary"]:
                value = training_stats["summary"][key]
            
            # Chuyển đổi giá trị phần trăm nếu cần
            if key in ["win_rate", "mean_win_rate", "final_win_rate"] and value is not None and isinstance(value, (int, float)):
                value = value * 100
            
            # Chỉ hiển thị nếu có giá trị
            if value is not None:
                with cols[i % columns]:
                    self.display_metric_card(
                        title=metric_info["title"],
                        value=value,
                        suffix=metric_info.get("suffix", ""),
                        format_string=metric_info.get("format")
                    )
                i += 1
        
        # Thêm các card trống nếu không chia đều được
        remaining = columns - (i % columns)
        if remaining < columns:
            for _ in range(remaining):
                with cols[(i + _) % columns]:
                    st.write("")
    
    def display_profit_loss_chart(self, 
                                profit_loss_data: pd.DataFrame,
                                cumulative: bool = True,
                                title: str = "Lợi nhuận/Lỗ theo thời gian",
                                height: int = 400) -> None:
        """
        Hiển thị biểu đồ lợi nhuận/lỗ theo thời gian.
        
        Args:
            profit_loss_data: DataFrame chứa dữ liệu lợi nhuận/lỗ
            cumulative: Hiển thị tích lũy hay không
            title: Tiêu đề biểu đồ
            height: Chiều cao biểu đồ
        """
        st.subheader(title)
        
        if profit_loss_data.empty:
            st.info("Không có dữ liệu để hiển thị")
            return
        
        # Đảm bảo timestamp là index nếu chưa phải
        if 'timestamp' in profit_loss_data.columns:
            profit_loss_data = profit_loss_data.set_index('timestamp')
        
        # Chuẩn bị dữ liệu
        plot_data = profit_loss_data.copy()
        
        # Xác định cột dữ liệu
        value_column = None
        for col in ["profit_loss", "profit", "pnl", "return"]:
            if col in plot_data.columns:
                value_column = col
                break
        
        if value_column is None:
            st.error("Không tìm thấy cột dữ liệu lợi nhuận/lỗ")
            return
        
        # Tính giá trị tích lũy nếu cần
        if cumulative and f"cum_{value_column}" not in plot_data.columns:
            plot_data[f"cum_{value_column}"] = plot_data[value_column].cumsum()
        
        # Chọn cột để vẽ
        y_column = f"cum_{value_column}" if cumulative else value_column
        
        # Tạo biểu đồ với Plotly
        fig = px.line(
            plot_data, 
            y=y_column,
            title=title,
            labels={y_column: "Lợi nhuận/Lỗ"},
            height=height
        )
        
        # Thêm điểm vào đường
        fig.update_traces(mode='lines+markers')
        
        # Tô màu cho vùng dưới đường
        fig.update_traces(
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(color=self.color_scheme["primary"]),
            marker=dict(color=self.color_scheme["primary"])
        )
        
        # Thêm đường tham chiếu 0
        fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Breakeven",
            annotation_position="bottom right"
        )
        
        # Điều chỉnh layout
        fig.update_layout(
            xaxis_title="Thời gian",
            yaxis_title="Lợi nhuận/Lỗ",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=self.color_scheme["text"]),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Hiển thị biểu đồ
        st.plotly_chart(fig, use_container_width=True)
    
    def display_equity_chart(self, 
                           equity_data: pd.DataFrame,
                           title: str = "Đường cong vốn",
                           show_drawdown: bool = True,
                           height: int = 400) -> None:
        """
        Hiển thị biểu đồ đường cong vốn.
        
        Args:
            equity_data: DataFrame chứa dữ liệu vốn
            title: Tiêu đề biểu đồ
            show_drawdown: Hiển thị drawdown hay không
            height: Chiều cao biểu đồ
        """
        st.subheader(title)
        
        if equity_data.empty:
            st.info("Không có dữ liệu để hiển thị")
            return
        
        # Đảm bảo timestamp là index nếu chưa phải
        if 'timestamp' in equity_data.columns:
            equity_data = equity_data.set_index('timestamp')
        
        # Chuẩn bị dữ liệu
        plot_data = equity_data.copy()
        
        # Xác định cột dữ liệu
        value_column = None
        for col in ["equity", "capital", "balance", "portfolio_value"]:
            if col in plot_data.columns:
                value_column = col
                break
        
        if value_column is None:
            st.error("Không tìm thấy cột dữ liệu vốn")
            return
        
        # Tính drawdown nếu cần
        if show_drawdown:
            # Tính peak-to-trough cho mỗi điểm
            rolling_max = plot_data[value_column].cummax()
            plot_data["drawdown"] = (plot_data[value_column] - rolling_max) / rolling_max * 100
        
        # Tạo subplots nếu hiển thị drawdown
        if show_drawdown:
            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )
            
            # Thêm đường cong vốn
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data[value_column],
                    name="Vốn",
                    line=dict(color=self.color_scheme["primary"]),
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # Thêm drawdown
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["drawdown"],
                    name="Drawdown",
                    fill='tozeroy',
                    line=dict(color=self.color_scheme["loss"]),
                    mode='lines'
                ),
                row=2, col=1
            )
            
            # Thêm đường tham chiếu 0 cho drawdown
            fig.add_shape(
                type="line",
                x0=plot_data.index[0],
                y0=0,
                x1=plot_data.index[-1],
                y1=0,
                line=dict(color="gray", width=1, dash="dash"),
                row=2, col=1
            )
        else:
            # Chỉ hiển thị đường cong vốn
            fig = go.Figure()
            
            # Thêm đường cong vốn
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data[value_column],
                    name="Vốn",
                    line=dict(color=self.color_scheme["primary"]),
                    mode='lines'
                )
            )
        
        # Điều chỉnh layout
        fig.update_layout(
            title=title,
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=self.color_scheme["text"]),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Thời gian"),
            yaxis=dict(title="Vốn")
        )
        
        if show_drawdown:
            fig.update_layout(
                yaxis2=dict(title="Drawdown (%)", autorange="reversed")
            )
        
        # Hiển thị biểu đồ
        st.plotly_chart(fig, use_container_width=True)
    
    def display_trade_distribution(self, 
                                 trades_data: pd.DataFrame,
                                 title: str = "Phân phối giao dịch",
                                 height: int = 400) -> None:
        """
        Hiển thị biểu đồ phân phối giao dịch.
        
        Args:
            trades_data: DataFrame chứa dữ liệu giao dịch
            title: Tiêu đề biểu đồ
            height: Chiều cao biểu đồ
        """
        st.subheader(title)
        
        if trades_data.empty:
            st.info("Không có dữ liệu để hiển thị")
            return
        
        # Chuẩn bị dữ liệu
        plot_data = trades_data.copy()
        
        # Xác định cột dữ liệu
        value_column = None
        for col in ["profit_loss", "profit", "pnl"]:
            if col in plot_data.columns:
                value_column = col
                break
        
        if value_column is None:
            st.error("Không tìm thấy cột dữ liệu lợi nhuận/lỗ")
            return
        
        # Tạo phân loại wins/losses
        plot_data["result"] = plot_data[value_column].apply(
            lambda x: "Thắng" if x > 0 else ("Thua" if x < 0 else "Hòa")
        )
        
        # Tính số lượng wins/losses
        win_count = len(plot_data[plot_data["result"] == "Thắng"])
        loss_count = len(plot_data[plot_data["result"] == "Thua"])
        
        # Tạo layout 2 cột
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Tạo biểu đồ histogram với Plotly
            fig = px.histogram(
                plot_data,
                x=value_column,
                color="result",
                color_discrete_map={
                    "Thắng": self.color_scheme["profit"],
                    "Thua": self.color_scheme["loss"],
                    "Hòa": self.color_scheme["neutral"]
                },
                title="Phân phối lợi nhuận/lỗ",
                height=height-50
            )
            
            # Thêm đường KDE
            kde_fig = ff.create_distplot(
                [plot_data[value_column].values],
                ["KDE"],
                show_hist=False,
                show_rug=False
            )
            fig.add_trace(kde_fig.data[0])
            
            # Thêm đường tham chiếu 0
            fig.add_vline(
                x=0, 
                line_dash="dash", 
                line_color="gray",
                annotation_text="Breakeven",
                annotation_position="top right"
            )
            
            # Điều chỉnh layout
            fig.update_layout(
                xaxis_title="Lợi nhuận/Lỗ",
                yaxis_title="Số lượng giao dịch",
                hovermode="x unified",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=self.color_scheme["text"]),
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Hiển thị biểu đồ
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tạo biểu đồ pie chart
            fig = px.pie(
                names=["Thắng", "Thua"],
                values=[win_count, loss_count],
                title="Tỷ lệ thắng/thua",
                color=["Thắng", "Thua"],
                color_discrete_map={
                    "Thắng": self.color_scheme["profit"],
                    "Thua": self.color_scheme["loss"]
                },
                height=height-50
            )
            
            # Điều chỉnh layout
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=self.color_scheme["text"]),
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            
            # Hiển thị biểu đồ
            st.plotly_chart(fig, use_container_width=True)
            
            # Hiển thị thêm thống kê
            if len(plot_data) > 0:
                wins = plot_data[plot_data[value_column] > 0]
                losses = plot_data[plot_data[value_column] < 0]
                
                st.metric("Win Rate", f"{win_count/len(plot_data):.2%}")
                
                if len(wins) > 0:
                    st.metric("Lãi trung bình", f"{wins[value_column].mean():.2f}")
                
                if len(losses) > 0:
                    st.metric("Lỗ trung bình", f"{losses[value_column].mean():.2f}")
                
                profit_factor = abs(wins[value_column].sum() / losses[value_column].sum()) if len(losses) > 0 and losses[value_column].sum() != 0 else float('inf')
                st.metric("Profit Factor", f"{profit_factor:.2f}")
    
    def display_drawdown_chart(self, 
                             equity_data: pd.DataFrame,
                             title: str = "Phân tích Drawdown",
                             height: int = 400) -> None:
        """
        Hiển thị biểu đồ phân tích drawdown.
        
        Args:
            equity_data: DataFrame chứa dữ liệu vốn
            title: Tiêu đề biểu đồ
            height: Chiều cao biểu đồ
        """
        st.subheader(title)
        
        if equity_data.empty:
            st.info("Không có dữ liệu để hiển thị")
            return
        
        # Đảm bảo timestamp là index nếu chưa phải
        if 'timestamp' in equity_data.columns:
            equity_data = equity_data.set_index('timestamp')
        
        # Chuẩn bị dữ liệu
        plot_data = equity_data.copy()
        
        # Xác định cột dữ liệu
        value_column = None
        for col in ["equity", "capital", "balance", "portfolio_value"]:
            if col in plot_data.columns:
                value_column = col
                break
        
        if value_column is None:
            st.error("Không tìm thấy cột dữ liệu vốn")
            return
        
        # Tạo biểu đồ
        fig = go.Figure()
        
        # Tính drawdown
        rolling_max = plot_data[value_column].cummax()
        drawdown = (plot_data[value_column] - rolling_max) / rolling_max * 100
        
        # Thêm drawdown
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=drawdown,
                name="Drawdown",
                fill='tozeroy',
                fillcolor=f"rgba({int(self.color_scheme['loss'][1:3], 16)}, {int(self.color_scheme['loss'][3:5], 16)}, {int(self.color_scheme['loss'][5:7], 16)}, 0.3)",
                line=dict(color=self.color_scheme["loss"]),
                mode='lines'
            )
        )
        
        # Tìm top drawdowns
        # Tạo danh sách các đợt drawdown
        drawdown_periods = []
        current_drawdown = None
        
        for i, (date, dd_value) in enumerate(drawdown.items()):
            if dd_value < 0 and current_drawdown is None:
                # Bắt đầu một drawdown mới
                current_drawdown = {
                    "start": date,
                    "start_value": plot_data[value_column].iloc[i],
                    "min_date": date,
                    "min_value": plot_data[value_column].iloc[i],
                    "min_drawdown": dd_value
                }
            elif dd_value < 0 and current_drawdown is not None:
                # Drawdown đang tiếp tục
                if dd_value < current_drawdown["min_drawdown"]:
                    # Cập nhật giá trị thấp nhất
                    current_drawdown["min_date"] = date
                    current_drawdown["min_value"] = plot_data[value_column].iloc[i]
                    current_drawdown["min_drawdown"] = dd_value
            elif dd_value == 0 and current_drawdown is not None:
                # Kết thúc drawdown
                current_drawdown["end"] = date
                current_drawdown["end_value"] = plot_data[value_column].iloc[i]
                current_drawdown["duration"] = (current_drawdown["end"] - current_drawdown["start"]).days
                drawdown_periods.append(current_drawdown)
                current_drawdown = None
        
        # Thêm drawdown hiện tại nếu chưa kết thúc
        if current_drawdown is not None:
            current_drawdown["end"] = drawdown.index[-1]
            current_drawdown["end_value"] = plot_data[value_column].iloc[-1]
            current_drawdown["duration"] = (current_drawdown["end"] - current_drawdown["start"]).days
            drawdown_periods.append(current_drawdown)
        
        # Sắp xếp theo độ sâu
        drawdown_periods.sort(key=lambda x: x["min_drawdown"])
        
        # Thêm annotations cho top 3 drawdowns
        for i, period in enumerate(drawdown_periods[:3]):
            fig.add_trace(
                go.Scatter(
                    x=[period["min_date"]],
                    y=[period["min_drawdown"]],
                    mode="markers+text",
                    marker=dict(size=10, color=self.color_scheme["warning"]),
                    text=[f"DD #{i+1}"],
                    textposition="bottom center",
                    showlegend=False
                )
            )
        
        # Thêm đường tham chiếu 0
        fig.add_shape(
            type="line",
            x0=plot_data.index[0],
            y0=0,
            x1=plot_data.index[-1],
            y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        # Điều chỉnh layout
        fig.update_layout(
            title=title,
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=self.color_scheme["text"]),
            yaxis=dict(title="Drawdown (%)", autorange="reversed"),
            xaxis=dict(title="Thời gian"),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Hiển thị biểu đồ
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị thông tin về top drawdowns
        if drawdown_periods:
            st.subheader("Top Drawdowns")
            
            # Tạo DataFrame cho drawdowns
            dd_df = pd.DataFrame([
                {
                    "Thứ hạng": i+1,
                    "Bắt đầu": period["start"].strftime("%Y-%m-%d"),
                    "Kết thúc": period["end"].strftime("%Y-%m-%d"),
                    "Thời gian (ngày)": period["duration"],
                    "Drawdown (%)": f"{period['min_drawdown']:.2f}%",
                    "Drawdown (giá trị)": f"{period['start_value'] - period['min_value']:.2f}"
                }
                for i, period in enumerate(sorted(drawdown_periods, key=lambda x: x["min_drawdown"])[:5])
            ])
            
            st.dataframe(dd_df, use_container_width=True)
    
    def display_training_rewards_chart(self, 
                                     rewards_data: pd.DataFrame,
                                     title: str = "Phần thưởng huấn luyện",
                                     height: int = 400) -> None:
        """
        Hiển thị biểu đồ phần thưởng huấn luyện.
        
        Args:
            rewards_data: DataFrame chứa dữ liệu phần thưởng
            title: Tiêu đề biểu đồ
            height: Chiều cao biểu đồ
        """
        st.subheader(title)
        
        if rewards_data.empty:
            st.info("Không có dữ liệu để hiển thị")
            return
        
        # Chuẩn bị dữ liệu
        plot_data = rewards_data.copy()
        
        # Đảm bảo có cột episode
        if "episode" not in plot_data.columns:
            plot_data["episode"] = range(1, len(plot_data) + 1)
        
        # Xác định cột dữ liệu
        value_column = None
        for col in ["reward", "episode_reward", "return"]:
            if col in plot_data.columns:
                value_column = col
                break
        
        if value_column is None:
            st.error("Không tìm thấy cột dữ liệu phần thưởng")
            return
        
        # Tạo biểu đồ
        fig = go.Figure()
        
        # Thêm đường phần thưởng
        fig.add_trace(
            go.Scatter(
                x=plot_data["episode"],
                y=plot_data[value_column],
                name="Phần thưởng",
                line=dict(color=self.color_scheme["primary"], width=1),
                mode='lines+markers',
                marker=dict(size=3),
                opacity=0.7
            )
        )
        
        # Tính trung bình trượt nếu đủ dữ liệu
        if len(plot_data) >= 10:
            window_size = min(100, max(10, len(plot_data) // 10))
            plot_data[f"{value_column}_rolling"] = plot_data[value_column].rolling(window=window_size).mean()
            
            # Thêm đường trung bình trượt
            fig.add_trace(
                go.Scatter(
                    x=plot_data["episode"],
                    y=plot_data[f"{value_column}_rolling"],
                    name=f"Trung bình trượt ({window_size} episodes)",
                    line=dict(color=self.color_scheme["secondary"], width=2),
                    mode='lines'
                )
            )
        
        # Thêm đường tham chiếu 0
        fig.add_shape(
            type="line",
            x0=plot_data["episode"].min(),
            y0=0,
            x1=plot_data["episode"].max(),
            y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        # Điều chỉnh layout
        fig.update_layout(
            title=title,
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=self.color_scheme["text"]),
            yaxis=dict(title="Phần thưởng"),
            xaxis=dict(title="Episode"),
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Hiển thị biểu đồ
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị thống kê
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Phần thưởng TB",
                f"{plot_data[value_column].mean():.2f}",
                help="Giá trị trung bình của phần thưởng"
            )
        
        with col2:
            st.metric(
                "Phần thưởng Max",
                f"{plot_data[value_column].max():.2f}",
                help="Giá trị lớn nhất của phần thưởng"
            )
        
        with col3:
            # Tính phần thưởng trung bình 10 episode cuối
            last_10_mean = plot_data[value_column].tail(10).mean()
            overall_mean = plot_data[value_column].mean()
            delta = ((last_10_mean / overall_mean) - 1) * 100 if overall_mean != 0 else 0
            
            st.metric(
                "10 episode cuối (TB)",
                f"{last_10_mean:.2f}",
                f"{delta:.1f}%",
                help="Phần thưởng trung bình của 10 episode gần nhất"
            )
        
        with col4:
            # Tính phần thưởng trung bình 100 episode cuối nếu đủ dữ liệu
            if len(plot_data) >= 100:
                last_100_mean = plot_data[value_column].tail(100).mean()
                delta = ((last_100_mean / overall_mean) - 1) * 100 if overall_mean != 0 else 0
                
                st.metric(
                    "100 episode cuối (TB)",
                    f"{last_100_mean:.2f}",
                    f"{delta:.1f}%",
                    help="Phần thưởng trung bình của 100 episode gần nhất"
                )
    
    def display_trade_history_table(self, 
                                  trades_data: pd.DataFrame,
                                  title: str = "Lịch sử giao dịch",
                                  limit: int = 10) -> None:
        """
        Hiển thị bảng lịch sử giao dịch.
        
        Args:
            trades_data: DataFrame chứa dữ liệu giao dịch
            title: Tiêu đề bảng
            limit: Số lượng giao dịch hiển thị
        """
        st.subheader(title)
        
        if trades_data.empty:
            st.info("Không có dữ liệu để hiển thị")
            return
        
        # Chuẩn bị dữ liệu
        plot_data = trades_data.copy()
        
        # Xác định các cột cần thiết
        required_columns = ["timestamp", "entry_price", "exit_price", "profit_loss", "side"]
        missing_columns = [col for col in required_columns if col not in plot_data.columns]
        
        # Kiểm tra các cột
        if missing_columns:
            alternative_columns = {
                "timestamp": ["entry_time", "time", "date", "datetime"],
                "entry_price": ["open_price", "entry"],
                "exit_price": ["close_price", "exit"],
                "profit_loss": ["profit", "pnl", "return"],
                "side": ["direction", "position", "type"]
            }
            
            # Tìm cột thay thế
            for col in missing_columns:
                for alt_col in alternative_columns[col]:
                    if alt_col in plot_data.columns:
                        plot_data[col] = plot_data[alt_col]
                        break
        
        # Kiểm tra lại các cột cần thiết
        missing_columns = [col for col in required_columns if col not in plot_data.columns]
        if missing_columns:
            st.error(f"Không tìm thấy các cột cần thiết: {', '.join(missing_columns)}")
            return
        
        # Đảm bảo timestamp đúng định dạng
        if plot_data["timestamp"].dtype == 'object':
            try:
                plot_data["timestamp"] = pd.to_datetime(plot_data["timestamp"])
            except:
                pass
        
        # Sắp xếp dữ liệu theo thời gian (mới nhất trước)
        plot_data = plot_data.sort_values("timestamp", ascending=False)
        
        # Giới hạn số lượng giao dịch
        display_data = plot_data.head(limit)
        
        # Định dạng lại dữ liệu
        display_df = pd.DataFrame()
        display_df["Thời gian"] = display_data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        display_df["Hướng"] = display_data["side"].apply(
            lambda x: "Mua" if x.lower() in ["buy", "long"] else "Bán" if x.lower() in ["sell", "short"] else x
        )
        display_df["Giá vào"] = display_data["entry_price"].round(2)
        display_df["Giá ra"] = display_data["exit_price"].round(2)
        display_df["Lãi/Lỗ"] = display_data["profit_loss"].round(2)
        
        # Thêm cột kết quả
        display_df["Kết quả"] = display_data["profit_loss"].apply(
            lambda x: "Thắng" if x > 0 else ("Thua" if x < 0 else "Hòa")
        )
        
        # Tùy chỉnh hiển thị
        st.dataframe(
            display_df,
            column_config={
                "Lãi/Lỗ": st.column_config.NumberColumn(
                    "Lãi/Lỗ",
                    format="%.2f",
                    help="Lãi/lỗ của giao dịch",
                ),
                "Kết quả": st.column_config.Column(
                    "Kết quả",
                    help="Kết quả của giao dịch",
                    width="small"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Hiển thị nút xem thêm
        if len(plot_data) > limit:
            st.button("Xem thêm giao dịch")
    
    def display_position_table(self, 
                             positions_data: pd.DataFrame,
                             title: str = "Vị thế hiện tại") -> None:
        """
        Hiển thị bảng vị thế hiện tại.
        
        Args:
            positions_data: DataFrame chứa dữ liệu vị thế
            title: Tiêu đề bảng
        """
        st.subheader(title)
        
        if positions_data.empty:
            st.info("Không có vị thế nào đang mở")
            return
        
        # Chuẩn bị dữ liệu
        plot_data = positions_data.copy()
        
        # Xác định các cột cần thiết
        required_columns = ["symbol", "side", "entry_price", "current_price", "quantity", "unrealized_pnl"]
        missing_columns = [col for col in required_columns if col not in plot_data.columns]
        
        # Kiểm tra các cột
        if missing_columns:
            alternative_columns = {
                "symbol": ["asset", "pair", "ticker"],
                "side": ["direction", "position", "type"],
                "entry_price": ["open_price", "entry"],
                "current_price": ["price", "mark_price", "market_price"],
                "quantity": ["size", "amount", "volume"],
                "unrealized_pnl": ["pnl", "profit_loss", "floating_pnl"]
            }
            
            # Tìm cột thay thế
            for col in missing_columns:
                for alt_col in alternative_columns[col]:
                    if alt_col in plot_data.columns:
                        plot_data[col] = plot_data[alt_col]
                        break
        
        # Kiểm tra lại các cột cần thiết
        missing_columns = [col for col in required_columns if col not in plot_data.columns]
        if missing_columns:
            st.error(f"Không tìm thấy các cột cần thiết: {', '.join(missing_columns)}")
            return
        
        # Tính ROI nếu chưa có
        if "roi" not in plot_data.columns:
            plot_data["roi"] = (plot_data["current_price"] - plot_data["entry_price"]) / plot_data["entry_price"]
            # Đảo dấu cho short
            plot_data.loc[plot_data["side"].str.lower().isin(["sell", "short"]), "roi"] *= -1
        
        # Định dạng lại dữ liệu
        display_df = pd.DataFrame()
        display_df["Cặp tiền"] = plot_data["symbol"]
        display_df["Hướng"] = plot_data["side"].apply(
            lambda x: "Mua" if x.lower() in ["buy", "long"] else "Bán" if x.lower() in ["sell", "short"] else x
        )
        display_df["Giá vào"] = plot_data["entry_price"].round(5)
        display_df["Giá hiện tại"] = plot_data["current_price"].round(5)
        display_df["Khối lượng"] = plot_data["quantity"].round(5)
        display_df["Lãi/Lỗ"] = plot_data["unrealized_pnl"].round(2)
        display_df["ROI (%)"] = (plot_data["roi"] * 100).round(2)
        
        # Tùy chỉnh hiển thị
        st.dataframe(
            display_df,
            column_config={
                "Lãi/Lỗ": st.column_config.NumberColumn(
                    "Lãi/Lỗ",
                    format="%.2f",
                    help="Lãi/lỗ chưa thực hiện của vị thế",
                ),
                "ROI (%)": st.column_config.NumberColumn(
                    "ROI (%)",
                    format="%.2f%%",
                    help="Tỷ suất sinh lời của vị thế",
                )
            },
            hide_index=True,
            use_container_width=True
        )
    
    def display_strategy_performance_comparison(self,
                                              strategy_performance: Dict[str, Dict[str, Any]],
                                              title: str = "So sánh hiệu suất chiến lược",
                                              metrics: Optional[List[str]] = None,
                                              height: int = 500) -> None:
        """
        Hiển thị biểu đồ so sánh hiệu suất các chiến lược.
        
        Args:
            strategy_performance: Dict với key là tên chiến lược và value là dict chỉ số
            title: Tiêu đề biểu đồ
            metrics: Danh sách các chỉ số cần so sánh
            height: Chiều cao biểu đồ
        """
        st.subheader(title)
        
        if not strategy_performance:
            st.info("Không có dữ liệu để hiển thị")
            return
        
        # Metrics mặc định
        default_metrics = [
            BacktestMetric.TOTAL_RETURN.value,
            BacktestMetric.SHARPE_RATIO.value,
            BacktestMetric.SORTINO_RATIO.value,
            BacktestMetric.MAX_DRAWDOWN.value,
            BacktestMetric.WIN_RATE.value,
            BacktestMetric.PROFIT_FACTOR.value
        ]
        
        # Sử dụng metrics được cung cấp hoặc mặc định
        metrics_to_display = metrics or default_metrics
        
        # Tạo DataFrame từ dữ liệu chiến lược
        strategies = []
        for strategy_name, perf in strategy_performance.items():
            strategy_data = {"strategy": strategy_name}
            
            for metric in metrics_to_display:
                # Lấy giá trị metric từ hiệu suất
                if metric in perf:
                    value = perf[metric]
                    
                    # Chuyển đổi phần trăm nếu cần
                    if metric in [BacktestMetric.TOTAL_RETURN.value, 
                                  BacktestMetric.MAX_DRAWDOWN.value, 
                                  BacktestMetric.WIN_RATE.value] and isinstance(value, (int, float)):
                        value = value * 100
                    
                    strategy_data[metric] = value
                else:
                    strategy_data[metric] = 0
            
            strategies.append(strategy_data)
        
        # Tạo DataFrame
        df = pd.DataFrame(strategies)
        
        # Tạo biểu đồ radar cho mỗi chiến lược
        fig = go.Figure()
        
        # Tên hiển thị cho các metrics
        metric_names = {
            BacktestMetric.TOTAL_RETURN.value: "Lợi nhuận (%)",
            BacktestMetric.ANNUALIZED_RETURN.value: "Lợi nhuận hàng năm (%)",
            BacktestMetric.SHARPE_RATIO.value: "Sharpe Ratio",
            BacktestMetric.SORTINO_RATIO.value: "Sortino Ratio",
            BacktestMetric.CALMAR_RATIO.value: "Calmar Ratio",
            BacktestMetric.MAX_DRAWDOWN.value: "Max Drawdown (%)",
            BacktestMetric.WIN_RATE.value: "Win Rate (%)",
            BacktestMetric.PROFIT_FACTOR.value: "Profit Factor"
        }
        
        # Chuẩn hóa dữ liệu cho biểu đồ radar
        normalized_df = df.copy()
        for metric in metrics_to_display:
            if metric != "strategy":
                # Tính min-max scaling
                min_val = normalized_df[metric].min()
                max_val = normalized_df[metric].max()
                
                if max_val > min_val:
                    # Chuẩn hóa
                    normalized_df[metric] = (normalized_df[metric] - min_val) / (max_val - min_val)
                    
                    # Đảo ngược cho max_drawdown (giá trị thấp là tốt)
                    if metric == BacktestMetric.MAX_DRAWDOWN.value:
                        normalized_df[metric] = 1 - normalized_df[metric]
        
        # Thêm các trace radar
        for i, strategy in enumerate(df["strategy"].unique()):
            strategy_data = normalized_df[normalized_df["strategy"] == strategy]
            
            # Chuẩn bị dữ liệu radar
            r = [strategy_data[metric].values[0] for metric in metrics_to_display if metric != "strategy"]
            theta = [metric_names.get(metric, metric) for metric in metrics_to_display if metric != "strategy"]
            
            # Đảm bảo radar khép kín
            r.append(r[0])
            theta.append(theta[0])
            
            # Thêm trace
            fig.add_trace(go.Scatterpolar(
                r=r,
                theta=theta,
                fill='toself',
                name=strategy,
                opacity=0.7
            ))
        
        # Điều chỉnh layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=title,
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=self.color_scheme["text"]),
            margin=dict(l=80, r=80, t=30, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        # Hiển thị biểu đồ radar
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị bảng so sánh
        st.subheader("Bảng so sánh chi tiết")
        
        # Đổi tên cột cho dễ đọc
        display_df = df.copy()
        display_df.columns = [metric_names.get(col, col) if col != "strategy" else "Chiến lược" for col in display_df.columns]
        
        # Hiển thị bảng
        st.dataframe(
            display_df.set_index("Chiến lược"),
            use_container_width=True
        )
    
    def display_risk_gauge_panel(self, risk_metrics: Dict[str, Any]) -> None:
        """
        Hiển thị panel các đồng hồ đo rủi ro.
        
        Args:
            risk_metrics: Dict chứa các chỉ số rủi ro
        """
        st.subheader("Đánh giá rủi ro")
        
        # Tạo các gauge cho các chỉ số rủi ro chính
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # VaR gauge
            var_value = None
            if "var" in risk_metrics and "historical" in risk_metrics["var"]:
                var_value = risk_metrics["var"]["historical"] * 100
            elif "var_95" in risk_metrics:
                var_value = risk_metrics["var_95"] * 100
            
            if var_value is not None:
                self.display_metric_gauge(
                    title="Value at Risk (95%)",
                    value=var_value,
                    min_value=0.0,
                    max_value=10.0,
                    green_threshold=0.3,  # <3% là tốt
                    yellow_threshold=0.7,  # 3-7% là trung bình
                    suffix="%",
                    help_text="Mức tổn thất tối đa với độ tin cậy 95%"
                )
        
        with col2:
            # Max Drawdown gauge
            max_dd_value = None
            if "max_drawdown" in risk_metrics:
                max_dd_value = risk_metrics["max_drawdown"] * 100
            elif "max_drawdown_percent" in risk_metrics:
                max_dd_value = risk_metrics["max_drawdown_percent"]
            
            if max_dd_value is not None:
                self.display_metric_gauge(
                    title="Maximum Drawdown",
                    value=max_dd_value,
                    min_value=0.0,
                    max_value=50.0,
                    green_threshold=0.2,  # <10% là tốt
                    yellow_threshold=0.6,  # 10-30% là trung bình
                    suffix="%",
                    help_text="Sụt giảm tối đa từ đỉnh đến đáy"
                )
        
        with col3:
            # Volatility gauge
            vol_value = None
            if "volatility" in risk_metrics:
                vol_value = risk_metrics["volatility"] * 100
            elif "annualized_volatility" in risk_metrics:
                vol_value = risk_metrics["annualized_volatility"] * 100
            
            if vol_value is not None:
                self.display_metric_gauge(
                    title="Biến động",
                    value=vol_value,
                    min_value=0.0,
                    max_value=50.0,
                    green_threshold=0.3,  # <15% là tốt
                    yellow_threshold=0.6,  # 15-30% là trung bình
                    suffix="%",
                    help_text="Biến động hàng năm của danh mục"
                )
        
        # Risk Rating
        risk_rating = "N/A"
        rating_explanation = ""
        
        # Xác định risk rating
        if "risk_assessment" in risk_metrics and "overall_risk_level" in risk_metrics["risk_assessment"]:
            risk_rating = risk_metrics["risk_assessment"]["overall_risk_level"]
            if "general_recommendation" in risk_metrics["risk_assessment"]:
                rating_explanation = risk_metrics["risk_assessment"]["general_recommendation"]
        else:
            # Tự đánh giá dựa trên max_drawdown và volatility
            if max_dd_value is not None and vol_value is not None:
                risk_score = 0
                
                # Điểm cho max_drawdown
                if max_dd_value > 30:
                    risk_score += 5
                elif max_dd_value > 20:
                    risk_score += 4
                elif max_dd_value > 15:
                    risk_score += 3
                elif max_dd_value > 10:
                    risk_score += 2
                elif max_dd_value > 5:
                    risk_score += 1
                
                # Điểm cho volatility
                if vol_value > 30:
                    risk_score += 5
                elif vol_value > 20:
                    risk_score += 4
                elif vol_value > 15:
                    risk_score += 3
                elif vol_value > 10:
                    risk_score += 2
                elif vol_value > 5:
                    risk_score += 1
                
                # Xác định risk rating
                if risk_score >= 9:
                    risk_rating = "Rất cao"
                    rating_explanation = "Chiến lược có mức độ rủi ro rất cao, cần thận trọng và quản lý rủi ro chặt chẽ."
                elif risk_score >= 7:
                    risk_rating = "Cao"
                    rating_explanation = "Chiến lược có mức độ rủi ro cao, cần áp dụng các biện pháp quản lý rủi ro."
                elif risk_score >= 5:
                    risk_rating = "Trung bình cao"
                    rating_explanation = "Chiến lược có mức độ rủi ro trung bình cao, cần theo dõi các chỉ số rủi ro thường xuyên."
                elif risk_score >= 3:
                    risk_rating = "Trung bình"
                    rating_explanation = "Chiến lược có mức độ rủi ro trung bình, cần cân bằng giữa rủi ro và lợi nhuận."
                else:
                    risk_rating = "Thấp"
                    rating_explanation = "Chiến lược có mức độ rủi ro thấp, phù hợp với người giao dịch an toàn."
        
        # Hiển thị đánh giá rủi ro
        st.info(f"**Đánh giá rủi ro: {risk_rating}**\n\n{rating_explanation}")
        
        # Hiển thị các khuyến nghị rủi ro nếu có
        if "risk_assessment" in risk_metrics and "risk_recommendations" in risk_metrics["risk_assessment"]:
            recommendations = risk_metrics["risk_assessment"]["risk_recommendations"]
            
            if recommendations:
                st.subheader("Khuyến nghị quản lý rủi ro")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
    
    def display_performance_summary(self, performance_metrics: Dict[str, Any]) -> None:
        """
        Hiển thị tóm tắt hiệu suất.
        
        Args:
            performance_metrics: Dict chứa các chỉ số hiệu suất
        """
        st.subheader("Tóm tắt hiệu suất")
        
        # Tạo layout cho các KPI chính
        col1, col2, col3, col4 = st.columns(4)
        
        # ROI
        roi_value = None
        if "roi" in performance_metrics:
            roi_value = performance_metrics["roi"] * 100
        elif "total_return" in performance_metrics:
            roi_value = performance_metrics["total_return"] * 100
        
        if roi_value is not None:
            with col1:
                self.display_metric_card(
                    title="Lợi nhuận",
                    value=roi_value,
                    suffix="%",
                    format_string="{:.2f}",
                    color=self.color_scheme["profit"] if roi_value > 0 else self.color_scheme["loss"]
                )
        
        # Sharpe Ratio
        sharpe_value = None
        if BacktestMetric.SHARPE_RATIO.value in performance_metrics:
            sharpe_value = performance_metrics[BacktestMetric.SHARPE_RATIO.value]
        
        if sharpe_value is not None:
            with col2:
                # Xác định màu dựa trên giá trị Sharpe
                if sharpe_value > 2:
                    color = self.color_scheme["profit"]
                elif sharpe_value > 1:
                    color = self.color_scheme["warning"]
                else:
                    color = self.color_scheme["loss"]
                
                self.display_metric_card(
                    title="Sharpe Ratio",
                    value=sharpe_value,
                    format_string="{:.2f}",
                    color=color
                )
        
        # Win Rate
        win_rate_value = None
        if BacktestMetric.WIN_RATE.value in performance_metrics:
            win_rate_value = performance_metrics[BacktestMetric.WIN_RATE.value] * 100
        
        if win_rate_value is not None:
            with col3:
                # Xác định màu dựa trên giá trị Win Rate
                if win_rate_value > 60:
                    color = self.color_scheme["profit"]
                elif win_rate_value > 50:
                    color = self.color_scheme["warning"]
                else:
                    color = self.color_scheme["loss"]
                
                self.display_metric_card(
                    title="Win Rate",
                    value=win_rate_value,
                    suffix="%",
                    format_string="{:.2f}",
                    color=color
                )
        
        # Profit Factor
        profit_factor_value = None
        if BacktestMetric.PROFIT_FACTOR.value in performance_metrics:
            profit_factor_value = performance_metrics[BacktestMetric.PROFIT_FACTOR.value]
        
        if profit_factor_value is not None:
            with col4:
                # Xác định màu dựa trên Profit Factor
                if profit_factor_value > 2:
                    color = self.color_scheme["profit"]
                elif profit_factor_value > 1:
                    color = self.color_scheme["warning"]
                else:
                    color = self.color_scheme["loss"]
                
                self.display_metric_card(
                    title="Profit Factor",
                    value=profit_factor_value,
                    format_string="{:.2f}",
                    color=color
                )
        
        # Thêm hàng thứ hai với các chỉ số phụ
        col1, col2, col3, col4 = st.columns(4)
        
        # Max Drawdown
        max_dd_value = None
        if BacktestMetric.MAX_DRAWDOWN.value in performance_metrics:
            max_dd_value = performance_metrics[BacktestMetric.MAX_DRAWDOWN.value] * 100
        
        if max_dd_value is not None:
            with col1:
                # Xác định màu dựa trên Max Drawdown
                if max_dd_value < 10:
                    color = self.color_scheme["profit"]
                elif max_dd_value < 20:
                    color = self.color_scheme["warning"]
                else:
                    color = self.color_scheme["loss"]
                
                self.display_metric_card(
                    title="Max Drawdown",
                    value=max_dd_value,
                    suffix="%",
                    format_string="{:.2f}",
                    color=color,
                    help_text="Sụt giảm tối đa từ đỉnh đến đáy"
                )
        
        # Calmar Ratio
        calmar_value = None
        if BacktestMetric.CALMAR_RATIO.value in performance_metrics:
            calmar_value = performance_metrics[BacktestMetric.CALMAR_RATIO.value]
        
        if calmar_value is not None:
            with col2:
                # Xác định màu dựa trên Calmar Ratio
                if calmar_value > 1:
                    color = self.color_scheme["profit"]
                elif calmar_value > 0.5:
                    color = self.color_scheme["warning"]
                else:
                    color = self.color_scheme["loss"]
                
                self.display_metric_card(
                    title="Calmar Ratio",
                    value=calmar_value,
                    format_string="{:.2f}",
                    color=color,
                    help_text="Tỷ lệ lợi nhuận/drawdown"
                )
        
        # Annualized Return
        annual_return_value = None
        if BacktestMetric.ANNUALIZED_RETURN.value in performance_metrics:
            annual_return_value = performance_metrics[BacktestMetric.ANNUALIZED_RETURN.value] * 100
        
        if annual_return_value is not None:
            with col3:
                self.display_metric_card(
                    title="Lợi nhuận hàng năm",
                    value=annual_return_value,
                    suffix="%",
                    format_string="{:.2f}",
                    color=self.color_scheme["profit"] if annual_return_value > 0 else self.color_scheme["loss"],
                    help_text="Lợi nhuận quy đổi theo năm"
                )
        
        # Volatility
        volatility_value = None
        if "volatility" in performance_metrics:
            volatility_value = performance_metrics["volatility"] * 100
        
        if volatility_value is not None:
            with col4:
                # Xác định màu dựa trên volatility
                if volatility_value < 15:
                    color = self.color_scheme["profit"]
                elif volatility_value < 30:
                    color = self.color_scheme["warning"]
                else:
                    color = self.color_scheme["loss"]
                
                self.display_metric_card(
                    title="Biến động",
                    value=volatility_value,
                    suffix="%",
                    format_string="{:.2f}",
                    color=color,
                    help_text="Biến động hàng năm của danh mục"
                )
    
    def display_portfolio_allocation(self, 
                                   allocation_data: Dict[str, float],
                                   title: str = "Phân bổ danh mục",
                                   height: int = 400) -> None:
        """
        Hiển thị biểu đồ phân bổ danh mục.
        
        Args:
            allocation_data: Dict với key là symbol và value là phần trăm phân bổ
            title: Tiêu đề biểu đồ
            height: Chiều cao biểu đồ
        """
        st.subheader(title)
        
        if not allocation_data:
            st.info("Không có dữ liệu phân bổ danh mục")
            return
        
        # Chuyển đổi dữ liệu
        allocation_items = sorted(allocation_data.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in allocation_items]
        values = [item[1] for item in allocation_items]
        
        # Tạo biểu đồ
        fig = go.Figure()
        
        # Thêm pie chart
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                textinfo='label+percent',
                marker=dict(
                    # Tạo bảng màu tự động dựa vào số lượng mục
                    colors=[self.color_scheme[color] for color in ["primary", "secondary", "info", "warning", "neutral"]][:len(labels)]
                )
            )
        )
        
        # Điều chỉnh layout
        fig.update_layout(
            title=title,
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=self.color_scheme["text"]),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        # Hiển thị biểu đồ
        st.plotly_chart(fig, use_container_width=True)
    
    def display_indicators_panel(self, 
                               price_data: pd.DataFrame, 
                               indicators: List[str],
                               title: str = "Chỉ báo kỹ thuật",
                               height: int = 600) -> None:
        """
        Hiển thị panel các chỉ báo kỹ thuật.
        
        Args:
            price_data: DataFrame chứa dữ liệu giá
            indicators: Danh sách các chỉ báo cần hiển thị
            title: Tiêu đề panel
            height: Chiều cao biểu đồ
        """
        st.subheader(title)
        
        if price_data.empty:
            st.info("Không có dữ liệu giá để hiển thị")
            return
        
        if not indicators:
            st.info("Không có chỉ báo nào được chọn")
            return
        
        # Danh sách các chỉ báo hỗ trợ
        supported_indicators = {
            "sma": "SMA (Simple Moving Average)",
            "ema": "EMA (Exponential Moving Average)",
            "rsi": "RSI (Relative Strength Index)",
            "macd": "MACD (Moving Average Convergence Divergence)",
            "bb": "Bollinger Bands",
            "atr": "ATR (Average True Range)",
            "stoch": "Stochastic Oscillator",
            "obv": "OBV (On-Balance Volume)"
        }
        
        # Lọc các chỉ báo được hỗ trợ
        selected_indicators = [ind for ind in indicators if ind in supported_indicators]
        
        if not selected_indicators:
            st.warning("Các chỉ báo được chọn không được hỗ trợ")
            return
        
        # Chuẩn bị dữ liệu
        plot_data = price_data.copy()
        
        # Đảm bảo timestamp là index nếu chưa phải
        if 'timestamp' in plot_data.columns:
            plot_data = plot_data.set_index('timestamp')
        
        # Xác định các cột OHLCV
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in plot_data.columns]
        
        if missing_columns:
            # Kiểm tra các tên cột thay thế phổ biến
            alternative_columns = {
                "open": ["Open", "OPEN", "open_price"],
                "high": ["High", "HIGH", "high_price"],
                "low": ["Low", "LOW", "low_price"],
                "close": ["Close", "CLOSE", "close_price"],
                "volume": ["Volume", "VOLUME", "vol"]
            }
            
            # Tìm cột thay thế
            for col in missing_columns:
                for alt_col in alternative_columns[col]:
                    if alt_col in plot_data.columns:
                        plot_data[col] = plot_data[alt_col]
                        break
        
        # Kiểm tra lại các cột cần thiết
        missing_columns = [col for col in ["open", "high", "low", "close"] if col not in plot_data.columns]
        if missing_columns:
            st.error(f"Không tìm thấy các cột dữ liệu giá cần thiết: {', '.join(missing_columns)}")
            return
        
        # Tính toán các chỉ báo được chọn
        # SMA
        if "sma" in selected_indicators:
            plot_data["sma_20"] = plot_data["close"].rolling(window=20).mean()
            plot_data["sma_50"] = plot_data["close"].rolling(window=50).mean()
        
        # EMA
        if "ema" in selected_indicators:
            plot_data["ema_20"] = plot_data["close"].ewm(span=20, adjust=False).mean()
            plot_data["ema_50"] = plot_data["close"].ewm(span=50, adjust=False).mean()
        
        # RSI
        if "rsi" in selected_indicators:
            delta = plot_data["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            plot_data["rsi"] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        if "bb" in selected_indicators:
            window = 20
            std = 2
            sma = plot_data["close"].rolling(window=window).mean()
            std_dev = plot_data["close"].rolling(window=window).std()
            plot_data["bb_upper"] = sma + (std * std_dev)
            plot_data["bb_middle"] = sma
            plot_data["bb_lower"] = sma - (std * std_dev)
        
        # MACD
        if "macd" in selected_indicators:
            fast = 12
            slow = 26
            signal = 9
            plot_data["macd_fast"] = plot_data["close"].ewm(span=fast, adjust=False).mean()
            plot_data["macd_slow"] = plot_data["close"].ewm(span=slow, adjust=False).mean()
            plot_data["macd"] = plot_data["macd_fast"] - plot_data["macd_slow"]
            plot_data["macd_signal"] = plot_data["macd"].ewm(span=signal, adjust=False).mean()
            plot_data["macd_histogram"] = plot_data["macd"] - plot_data["macd_signal"]
        
        # ATR
        if "atr" in selected_indicators:
            window = 14
            high_low = plot_data["high"] - plot_data["low"]
            high_close = abs(plot_data["high"] - plot_data["close"].shift())
            low_close = abs(plot_data["low"] - plot_data["close"].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            plot_data["atr"] = true_range.rolling(window=window).mean()
        
        # Stochastic Oscillator
        if "stoch" in selected_indicators:
            k_period = 14
            d_period = 3
            low_min = plot_data["low"].rolling(window=k_period).min()
            high_max = plot_data["high"].rolling(window=k_period).max()
            plot_data["stoch_k"] = 100 * ((plot_data["close"] - low_min) / (high_max - low_min))
            plot_data["stoch_d"] = plot_data["stoch_k"].rolling(window=d_period).mean()
        
        # OBV
        if "obv" in selected_indicators and "volume" in plot_data.columns:
            plot_data["obv"] = (np.sign(plot_data["close"].diff()) * plot_data["volume"]).fillna(0).cumsum()
        
        # Sắp xếp lại subplot
        fig = make_subplots(
            rows=len(selected_indicators) + 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=["Biểu đồ giá"] + [supported_indicators[ind] for ind in selected_indicators],
            row_heights=[0.4] + [0.6 / len(selected_indicators)] * len(selected_indicators)
        )
        
        # Thêm candlestick
        fig.add_trace(
            go.Candlestick(
                x=plot_data.index,
                open=plot_data["open"],
                high=plot_data["high"],
                low=plot_data["low"],
                close=plot_data["close"],
                name="OHLC"
            ),
            row=1, col=1
        )
        
        # Thêm các chỉ báo
        row_idx = 2
        
        # SMA
        if "sma" in selected_indicators:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["close"],
                    name="Close",
                    line=dict(color=self.color_scheme["primary"], width=1),
                    opacity=0.5
                ),
                row=row_idx, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["sma_20"],
                    name="SMA 20",
                    line=dict(color=self.color_scheme["profit"], width=1.5)
                ),
                row=row_idx, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["sma_50"],
                    name="SMA 50",
                    line=dict(color=self.color_scheme["secondary"], width=1.5)
                ),
                row=row_idx, col=1
            )
            
            row_idx += 1
        
        # EMA
        if "ema" in selected_indicators:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["close"],
                    name="Close",
                    line=dict(color=self.color_scheme["primary"], width=1),
                    opacity=0.5
                ),
                row=row_idx, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["ema_20"],
                    name="EMA 20",
                    line=dict(color=self.color_scheme["profit"], width=1.5)
                ),
                row=row_idx, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["ema_50"],
                    name="EMA 50",
                    line=dict(color=self.color_scheme["secondary"], width=1.5)
                ),
                row=row_idx, col=1
            )
            
            row_idx += 1
        
        # RSI
        if "rsi" in selected_indicators:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["rsi"],
                    name="RSI",
                    line=dict(color=self.color_scheme["primary"], width=1.5)
                ),
                row=row_idx, col=1
            )
            
            # Thêm đường tham chiếu 70 và 30
            fig.add_shape(
                type="line",
                x0=plot_data.index[0],
                y0=70,
                x1=plot_data.index[-1],
                y1=70,
                line=dict(color=self.color_scheme["loss"], width=1, dash="dash"),
                row=row_idx, col=1
            )
            
            fig.add_shape(
                type="line",
                x0=plot_data.index[0],
                y0=30,
                x1=plot_data.index[-1],
                y1=30,
                line=dict(color=self.color_scheme["profit"], width=1, dash="dash"),
                row=row_idx, col=1
            )
            
            fig.update_yaxes(range=[0, 100], row=row_idx, col=1)
            
            row_idx += 1
        
        # Bollinger Bands
        if "bb" in selected_indicators:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["close"],
                    name="Close",
                    line=dict(color=self.color_scheme["primary"], width=1.5)
                ),
                row=row_idx, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["bb_upper"],
                    name="Upper Band",
                    line=dict(color=self.color_scheme["loss"], width=1, dash="dash")
                ),
                row=row_idx, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["bb_middle"],
                    name="Middle Band",
                    line=dict(color=self.color_scheme["neutral"], width=1, dash="dash")
                ),
                row=row_idx, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["bb_lower"],
                    name="Lower Band",
                    line=dict(color=self.color_scheme["profit"], width=1, dash="dash")
                ),
                row=row_idx, col=1
            )
            
            row_idx += 1
        
        # MACD
        if "macd" in selected_indicators:
            # MACD Line
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["macd"],
                    name="MACD",
                    line=dict(color=self.color_scheme["primary"], width=1.5)
                ),
                row=row_idx, col=1
            )
            
            # Signal Line
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["macd_signal"],
                    name="Signal",
                    line=dict(color=self.color_scheme["secondary"], width=1.5)
                ),
                row=row_idx, col=1
            )
            
            # Histogram
            colors = np.where(plot_data["macd_histogram"] >= 0, self.color_scheme["profit"], self.color_scheme["loss"])
            
            fig.add_trace(
                go.Bar(
                    x=plot_data.index, 
                    y=plot_data["macd_histogram"],
                    name="Histogram",
                    marker_color=colors
                ),
                row=row_idx, col=1
            )
            
            # Đường tham chiếu 0
            fig.add_shape(
                type="line",
                x0=plot_data.index[0],
                y0=0,
                x1=plot_data.index[-1],
                y1=0,
                line=dict(color="gray", width=1, dash="dash"),
                row=row_idx, col=1
            )
            
            row_idx += 1
        
        # ATR
        if "atr" in selected_indicators:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["atr"],
                    name="ATR",
                    line=dict(color=self.color_scheme["primary"], width=1.5)
                ),
                row=row_idx, col=1
            )
            
            row_idx += 1
        
        # Stochastic Oscillator
        if "stoch" in selected_indicators:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["stoch_k"],
                    name="%K",
                    line=dict(color=self.color_scheme["primary"], width=1.5)
                ),
                row=row_idx, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["stoch_d"],
                    name="%D",
                    line=dict(color=self.color_scheme["secondary"], width=1.5)
                ),
                row=row_idx, col=1
            )
            
            # Thêm đường tham chiếu 80 và 20
            fig.add_shape(
                type="line",
                x0=plot_data.index[0],
                y0=80,
                x1=plot_data.index[-1],
                y1=80,
                line=dict(color=self.color_scheme["loss"], width=1, dash="dash"),
                row=row_idx, col=1
            )
            
            fig.add_shape(
                type="line",
                x0=plot_data.index[0],
                y0=20,
                x1=plot_data.index[-1],
                y1=20,
                line=dict(color=self.color_scheme["profit"], width=1, dash="dash"),
                row=row_idx, col=1
            )
            
            fig.update_yaxes(range=[0, 100], row=row_idx, col=1)
            
            row_idx += 1
        
        # OBV
        if "obv" in selected_indicators and "obv" in plot_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data["obv"],
                    name="OBV",
                    line=dict(color=self.color_scheme["primary"], width=1.5)
                ),
                row=row_idx, col=1
            )
            
            row_idx += 1
        
        # Điều chỉnh layout
        fig.update_layout(
            title=title,
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=self.color_scheme["text"]),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Điều chỉnh các trục y
        for i in range(1, row_idx):
            fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
                row=i,
                col=1
            )
        
        # Hiển thị biểu đồ
        st.plotly_chart(fig, use_container_width=True)


# Các hàm helper để sử dụng MetricsDisplay

def create_metrics_display(theme: str = "light") -> MetricsDisplay:
    """
    Tạo đối tượng MetricsDisplay với chủ đề được chỉ định.
    
    Args:
        theme: Chủ đề hiển thị ('light' hoặc 'dark')
        
    Returns:
        Đối tượng MetricsDisplay
    """
    return MetricsDisplay(theme=theme)


def display_full_dashboard(
    metrics_data: Dict[str, Any],
    equity_data: Optional[pd.DataFrame] = None,
    trades_data: Optional[pd.DataFrame] = None,
    theme: str = "light"
) -> None:
    """
    Hiển thị dashboard đầy đủ với các chỉ số và biểu đồ.
    
    Args:
        metrics_data: Dict chứa các chỉ số hiệu suất và rủi ro
        equity_data: DataFrame chứa dữ liệu vốn
        trades_data: DataFrame chứa dữ liệu giao dịch
        theme: Chủ đề hiển thị ('light' hoặc 'dark')
    """
    # Tạo đối tượng MetricsDisplay
    display = MetricsDisplay(theme=theme)
    
    # Hiển thị tóm tắt hiệu suất
    display.display_performance_summary(metrics_data)
    
    # Hiển thị tabs
    tab1, tab2, tab3 = st.tabs(["Biểu đồ hiệu suất", "Phân tích rủi ro", "Thống kê giao dịch"])
    
    with tab1:
        # Biểu đồ hiệu suất
        if equity_data is not None:
            display.display_equity_chart(equity_data, show_drawdown=True)
        
        # Biểu đồ phân phối giao dịch nếu có dữ liệu
        if trades_data is not None:
            display.display_trade_distribution(trades_data)
    
    with tab2:
        # Panel đồng hồ đo rủi ro
        if "risk_metrics" in metrics_data:
            display.display_risk_gauge_panel(metrics_data["risk_metrics"])
        
        # Biểu đồ drawdown nếu có dữ liệu
        if equity_data is not None:
            display.display_drawdown_chart(equity_data)
        
        # Hiển thị các chỉ số rủi ro chi tiết
        if "risk_metrics" in metrics_data:
            display.display_risk_metrics(metrics_data["risk_metrics"])
    
    with tab3:
        # Hiển thị thống kê giao dịch
        if "trading_stats" in metrics_data:
            display.display_trading_stats(metrics_data["trading_stats"])
        
        # Hiển thị bảng lịch sử giao dịch nếu có dữ liệu
        if trades_data is not None:
            display.display_trade_history_table(trades_data)


# Hàm tiện ích bổ sung

def display_training_dashboard(
    training_data: Dict[str, Any],
    rewards_data: Optional[pd.DataFrame] = None,
    theme: str = "light"
) -> None:
    """
    Hiển thị dashboard huấn luyện với các chỉ số và biểu đồ.
    
    Args:
        training_data: Dict chứa các chỉ số huấn luyện
        rewards_data: DataFrame chứa dữ liệu phần thưởng theo episode
        theme: Chủ đề hiển thị ('light' hoặc 'dark')
    """
    # Tạo đối tượng MetricsDisplay
    display = MetricsDisplay(theme=theme)
    
    # Hiển thị thống kê huấn luyện
    display.display_training_stats(training_data)
    
    # Hiển thị biểu đồ phần thưởng nếu có dữ liệu
    if rewards_data is not None:
        display.display_training_rewards_chart(rewards_data)


def display_trading_dashboard(
    trading_data: Dict[str, Any],
    equity_data: Optional[pd.DataFrame] = None,
    trades_data: Optional[pd.DataFrame] = None,
    positions_data: Optional[pd.DataFrame] = None,
    price_data: Optional[pd.DataFrame] = None,
    indicators: Optional[List[str]] = None,
    theme: str = "light"
) -> None:
    """
    Hiển thị dashboard giao dịch với các chỉ số và biểu đồ.
    
    Args:
        trading_data: Dict chứa các chỉ số giao dịch
        equity_data: DataFrame chứa dữ liệu vốn
        trades_data: DataFrame chứa dữ liệu giao dịch
        positions_data: DataFrame chứa dữ liệu vị thế hiện tại
        price_data: DataFrame chứa dữ liệu giá
        indicators: Danh sách các chỉ báo cần hiển thị
        theme: Chủ đề hiển thị ('light' hoặc 'dark')
    """
    # Tạo đối tượng MetricsDisplay
    display = MetricsDisplay(theme=theme)
    
    # Hiển thị tóm tắt hiệu suất
    display.display_performance_summary(trading_data)
    
    # Hiển thị tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Biểu đồ hiệu suất", "Vị thế & Giao dịch", "Phân tích rủi ro", "Chỉ báo kỹ thuật"])
    
    with tab1:
        # Biểu đồ hiệu suất
        if equity_data is not None:
            display.display_equity_chart(equity_data, show_drawdown=True)
        
        # Biểu đồ phân phối giao dịch nếu có dữ liệu
        if trades_data is not None:
            display.display_trade_distribution(trades_data)
    
    with tab2:
        # Hiển thị bảng vị thế hiện tại nếu có dữ liệu
        if positions_data is not None:
            display.display_position_table(positions_data)
        
        # Hiển thị bảng lịch sử giao dịch nếu có dữ liệu
        if trades_data is not None:
            display.display_trade_history_table(trades_data)
        
        # Hiển thị thống kê giao dịch
        if "trading_stats" in trading_data:
            display.display_trading_stats(trading_data["trading_stats"])
    
    with tab3:
        # Panel đồng hồ đo rủi ro
        if "risk_metrics" in trading_data:
            display.display_risk_gauge_panel(trading_data["risk_metrics"])
        
        # Biểu đồ drawdown nếu có dữ liệu
        if equity_data is not None:
            display.display_drawdown_chart(equity_data)
        
        # Hiển thị các chỉ số rủi ro chi tiết
        if "risk_metrics" in trading_data:
            display.display_risk_metrics(trading_data["risk_metrics"])
    
    with tab4:
        # Hiển thị panel chỉ báo kỹ thuật nếu có dữ liệu
        if price_data is not None and indicators:
            display.display_indicators_panel(price_data, indicators)
        else:
            st.info("Không có dữ liệu giá hoặc chỉ báo để hiển thị")


def display_strategy_comparison(
    strategies_data: Dict[str, Dict[str, Any]],
    metrics: Optional[List[str]] = None,
    theme: str = "light"
) -> None:
    """
    Hiển thị so sánh hiệu suất các chiến lược.
    
    Args:
        strategies_data: Dict với key là tên chiến lược và value là Dict chỉ số
        metrics: Danh sách các chỉ số cần so sánh
        theme: Chủ đề hiển thị ('light' hoặc 'dark')
    """
    # Tạo đối tượng MetricsDisplay
    display = MetricsDisplay(theme=theme)
    
    # Hiển thị biểu đồ so sánh
    display.display_strategy_performance_comparison(strategies_data, metrics=metrics)


# Chỉ chạy khi file này được chạy trực tiếp
if __name__ == "__main__":
    # Cài đặt các biến môi trường mô phỏng
    import sys
    import os
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    
    # Thêm thư mục gốc vào sys.path
    project_dir = Path(__file__).parent.parent.parent
    sys.path.append(str(project_dir))
    
    # Tạo ứng dụng Streamlit mô phỏng
    st.set_page_config(page_title="Metrics Display Demo", layout="wide")
    
    # Tạo dữ liệu giả
    # Dữ liệu vốn
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    equity = [100000]
    for i in range(1, len(dates)):
        equity.append(equity[-1] * (1 + np.random.normal(0.0005, 0.01)))
    equity_data = pd.DataFrame({
        'timestamp': dates,
        'equity': equity
    })
    
    # Dữ liệu giao dịch
    trades = []
    for i in range(100):
        entry_date = dates[np.random.randint(0, len(dates)-30)]
        exit_date = entry_date + pd.Timedelta(days=np.random.randint(1, 30))
        side = np.random.choice(['buy', 'sell'])
        entry_price = np.random.uniform(100, 200)
        exit_price = entry_price * (1 + np.random.normal(0.01, 0.05))
        quantity = np.random.uniform(0.1, 1.0)
        profit = (exit_price - entry_price) * quantity if side == 'buy' else (entry_price - exit_price) * quantity
        
        trades.append({
            'timestamp': exit_date,
            'entry_time': entry_date,
            'exit_time': exit_date,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'profit_loss': profit
        })
    
    trades_data = pd.DataFrame(trades)
    
    # Dữ liệu metrics
    metrics_data = {
        "roi": 0.15,
        BacktestMetric.TOTAL_RETURN.value: 0.15,
        BacktestMetric.ANNUALIZED_RETURN.value: 0.12,
        BacktestMetric.SHARPE_RATIO.value: 1.8,
        BacktestMetric.SORTINO_RATIO.value: 2.1,
        BacktestMetric.CALMAR_RATIO.value: 1.2,
        BacktestMetric.MAX_DRAWDOWN.value: 0.1,
        BacktestMetric.WIN_RATE.value: 0.65,
        BacktestMetric.PROFIT_FACTOR.value: 1.7,
        "volatility": 0.12,
        "risk_metrics": {
            "var": {
                "historical": 0.02,
                "parametric": 0.025
            },
            "es": 0.03,
            "max_drawdown": 0.1,
            "volatility": 0.12,
            "downside_volatility": 0.08,
            "volatility_ratio": 1.2,
            "skewness": 0.3,
            "kurtosis": 3.5,
            "tail_ratio": 0.9,
            "ulcer_index": 0.05,
            "pain_index": 0.04,
            "pain_ratio": 3.0
        },
        "trading_stats": {
            "total_trades": 100,
            "winning_trades": 65,
            "losing_trades": 35,
            "win_rate": 0.65,
            "avg_profit": 150,
            "avg_loss": -100,
            "profit_factor": 1.7,
            "avg_trade_duration": 5.0,
            "consecutive_wins": 8,
            "consecutive_losses": 4,
            "largest_win": 500,
            "largest_loss": -350,
            "total_fees": 250
        }
    }
    
    # Hiển thị demo
    st.title("Dashboard Metrics Display Demo")
    
    display = MetricsDisplay()
    
    tab1, tab2, tab3 = st.tabs(["Hiệu suất & Rủi ro", "Các loại chỉ số", "Biểu đồ"])
    
    with tab1:
        display_full_dashboard(metrics_data, equity_data, trades_data)
    
    with tab2:
        st.subheader("Hiển thị các loại chỉ số")
        
        st.write("### Chỉ số dạng Card")
        cols = st.columns(4)
        with cols[0]:
            display.display_metric_card("Lợi nhuận", 15.5, suffix="%")
        with cols[1]:
            display.display_metric_card("Sharpe Ratio", 1.8)
        with cols[2]:
            display.display_metric_card("Win Rate", 65, suffix="%")
        with cols[3]:
            display.display_metric_card("Số giao dịch", 100)
        
        st.write("### Chỉ số dạng Gauge")
        cols = st.columns(3)
        with cols[0]:
            display.display_metric_gauge("Win Rate", 0.65, min_value=0, max_value=1)
        with cols[1]:
            display.display_metric_gauge("Max Drawdown", 0.1, min_value=0, max_value=0.5)
        with cols[2]:
            display.display_metric_gauge("VaR (95%)", 0.02, min_value=0, max_value=0.1)
    
    with tab3:
        st.subheader("Hiển thị các loại biểu đồ")
        
        st.write("### Biểu đồ đường cong vốn")
        display.display_equity_chart(equity_data)
        
        st.write("### Biểu đồ phân phối giao dịch")
        display.display_trade_distribution(trades_data)
        
        st.write("### Bảng lịch sử giao dịch")
        display.display_trade_history_table(trades_data)