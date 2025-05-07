"""
Module trực quan hóa giao dịch cho dashboard Streamlit của hệ thống giao dịch tự động.
File này cung cấp các lớp và hàm để hiển thị biểu đồ giao dịch, thống kê hiệu suất,
và trực quan hóa danh mục đầu tư dưới dạng các thành phần tương tác trong ứng dụng Streamlit.
"""

import os
import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import json
import io
import base64
import time

# Import các module từ hệ thống
from logs.logger import get_logger
from config.constants import PositionSide, OrderType, OrderStatus, PositionStatus
from config.constants import Timeframe, ErrorCode, BacktestMetric
from config.system_config import get_system_config

# Import các module backtesting
try:
    from backtesting.visualization.backtest_visualizer import BacktestVisualizer
    BACKTEST_VISUALIZER_AVAILABLE = True
except ImportError:
    BACKTEST_VISUALIZER_AVAILABLE = False

# Import các module xử lý dữ liệu
try:
    from data_processors.feature_engineering.technical_indicator import trend_indicators, momentum_indicators, volume_indicators
    INDICATORS_AVAILABLE = True
except ImportError:
    INDICATORS_AVAILABLE = False


class TradeVisualization:
    """
    Lớp chính cung cấp các phương thức trực quan hóa giao dịch cho Streamlit dashboard.
    
    Cung cấp các chức năng:
    1. Hiển thị biểu đồ giá và giao dịch (candlestick, đường, v.v.)
    2. Hiển thị danh sách giao dịch và thống kê
    3. Hiển thị các chỉ số hiệu suất
    4. Trực quan hóa danh mục đầu tư
    5. Trực quan hóa so sánh chiến lược
    6. Bảng điều khiển tương tác để lọc và tùy chỉnh hiển thị
    """
    
    def __init__(
        self,
        style: str = "light",
        use_plotly: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo TradeVisualization.
        
        Args:
            style: Phong cách biểu đồ ('light', 'dark', 'streamlit')
            use_plotly: Sử dụng Plotly (True) hoặc Matplotlib (False)
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("trade_visualization")
        
        # Thiết lập cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập phong cách biểu đồ
        self.style = style
        self.use_plotly = use_plotly
        self._set_style()
        
        # Khởi tạo BacktestVisualizer nếu có sẵn
        if BACKTEST_VISUALIZER_AVAILABLE:
            self.backtest_visualizer = BacktestVisualizer(
                style='dark' if style == 'dark' else 'default',
                interactive=use_plotly,
                logger=self.logger
            )
        
        self.logger.info(f"Đã khởi tạo TradeVisualization với style={style}, use_plotly={use_plotly}")
    
    def _set_style(self) -> None:
        """
        Thiết lập phong cách biểu đồ.
        """
        if not self.use_plotly:
            # Thiết lập phong cách Matplotlib
            if self.style == "dark":
                plt.style.use('dark_background')
            elif self.style == "light":
                plt.style.use('default')
            elif self.style == "streamlit":
                # Phong cách phù hợp với Streamlit
                plt.style.use('fivethirtyeight')
            else:
                # Phong cách mặc định
                plt.style.use('default')
                
            # Thiết lập phông chữ và kích thước
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.size': 10,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16
            })
            
            # Thiết lập màu sắc cho seaborn
            sns.set_palette("deep")
    
    def plot_price_chart(
        self,
        price_data: pd.DataFrame,
        trades_data: Optional[pd.DataFrame] = None,
        symbol: str = "",
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        chart_type: str = "candlestick",
        show_volume: bool = True,
        show_indicators: List[str] = ["sma", "ema", "bollinger"],
        height: int = 600,
        width: int = 1000,
        allow_download: bool = False
    ) -> Any:
        """
        Vẽ biểu đồ giá và đánh dấu các giao dịch lên đó.
        
        Args:
            price_data: DataFrame chứa dữ liệu giá (OHLCV)
            trades_data: DataFrame chứa dữ liệu giao dịch
            symbol: Mã token/cặp giao dịch
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc
            chart_type: Loại biểu đồ ('candlestick', 'line', 'ohlc', 'area')
            show_volume: Hiển thị khối lượng
            show_indicators: Danh sách các chỉ báo cần hiển thị
            height: Chiều cao biểu đồ
            width: Chiều rộng biểu đồ
            allow_download: Cho phép tải xuống biểu đồ
            
        Returns:
            Đối tượng figure hoặc None
        """
        # Kiểm tra dữ liệu đầu vào
        if price_data is None or len(price_data) == 0:
            st.error("Không có dữ liệu giá để hiển thị!")
            return None
            
        # Chuyển đổi index thành datetime nếu chưa phải
        if not isinstance(price_data.index, pd.DatetimeIndex):
            # Kiểm tra xem có cột timestamp/datetime không
            if 'timestamp' in price_data.columns:
                price_data = price_data.set_index('timestamp')
            elif 'datetime' in price_data.columns:
                price_data = price_data.set_index('datetime')
            elif 'date' in price_data.columns:
                price_data = price_data.set_index('date')
            else:
                # Nếu không có cột thời gian, sử dụng chỉ số làm index
                price_data = price_data.copy()
                price_data.index = pd.to_datetime(price_data.index)
        
        # Kiểm tra các cột OHLCV cần thiết
        required_columns = ['open', 'high', 'low', 'close']
        if show_volume:
            required_columns.append('volume')
            
        # Chuẩn hóa tên cột (viết thường)
        price_data.columns = [col.lower() for col in price_data.columns]
        
        # Kiểm tra các cột tồn tại
        missing_columns = [col for col in required_columns if col not in price_data.columns]
        if missing_columns:
            # Thử chuyển đổi tên cột
            column_mapping = {}
            if 'o' in price_data.columns and 'open' not in price_data.columns:
                column_mapping['o'] = 'open'
            if 'h' in price_data.columns and 'high' not in price_data.columns:
                column_mapping['h'] = 'high'
            if 'l' in price_data.columns and 'low' not in price_data.columns:
                column_mapping['l'] = 'low'
            if 'c' in price_data.columns and 'close' not in price_data.columns:
                column_mapping['c'] = 'close'
            if 'v' in price_data.columns and 'volume' not in price_data.columns:
                column_mapping['v'] = 'volume'
                
            if column_mapping:
                price_data = price_data.rename(columns=column_mapping)
                # Kiểm tra lại các cột cần thiết
                missing_columns = [col for col in required_columns if col not in price_data.columns]
        
        if missing_columns:
            st.error(f"Dữ liệu giá thiếu các cột: {missing_columns}")
            return None
            
        # Lọc theo thời gian
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            price_data = price_data[price_data.index >= start_date]
            
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            price_data = price_data[price_data.index <= end_date]
            
        # Tính toán các chỉ báo kỹ thuật nếu cần
        indicators_data = {}
        if show_indicators and len(show_indicators) > 0:
            if INDICATORS_AVAILABLE:
                # Sử dụng module technical_indicator nếu có sẵn
                if "sma" in show_indicators:
                    indicators_data["SMA 20"] = trend_indicators.sma(price_data['close'], window=20)
                    indicators_data["SMA 50"] = trend_indicators.sma(price_data['close'], window=50)
                
                if "ema" in show_indicators:
                    indicators_data["EMA 20"] = trend_indicators.ema(price_data['close'], window=20)
                    indicators_data["EMA 50"] = trend_indicators.ema(price_data['close'], window=50)
                
                if "bollinger" in show_indicators:
                    bb_upper, bb_middle, bb_lower = trend_indicators.bollinger_bands(price_data['close'])
                    indicators_data["BB Upper"] = bb_upper
                    indicators_data["BB Lower"] = bb_lower
                
                if "rsi" in show_indicators:
                    indicators_data["RSI"] = momentum_indicators.rsi(price_data['close'])
                
                if "macd" in show_indicators:
                    macd_line, signal_line, histogram = momentum_indicators.macd(price_data['close'])
                    indicators_data["MACD"] = macd_line
                    indicators_data["Signal"] = signal_line
                
                if "volume_profile" in show_indicators and "volume" in price_data.columns:
                    indicators_data["Volume Profile"] = volume_indicators.volume_profile(price_data)
            else:
                # Tính toán thủ công
                if "sma" in show_indicators:
                    indicators_data["SMA 20"] = price_data['close'].rolling(window=20).mean()
                    indicators_data["SMA 50"] = price_data['close'].rolling(window=50).mean()
                    
                if "ema" in show_indicators:
                    indicators_data["EMA 20"] = price_data['close'].ewm(span=20, adjust=False).mean()
                    indicators_data["EMA 50"] = price_data['close'].ewm(span=50, adjust=False).mean()
                    
                if "bollinger" in show_indicators:
                    sma20 = price_data['close'].rolling(window=20).mean()
                    std20 = price_data['close'].rolling(window=20).std()
                    indicators_data["BB Upper"] = sma20 + (std20 * 2)
                    indicators_data["BB Lower"] = sma20 - (std20 * 2)
                    
                if "rsi" in show_indicators:
                    delta = price_data['close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    rs = avg_gain / avg_loss
                    indicators_data["RSI"] = 100 - (100 / (1 + rs))
        
        try:
            if self.use_plotly:
                return self._plot_price_chart_plotly(
                    price_data=price_data,
                    trades_data=trades_data,
                    symbol=symbol,
                    chart_type=chart_type,
                    show_volume=show_volume,
                    indicators_data=indicators_data,
                    height=height,
                    width=width,
                    allow_download=allow_download
                )
            else:
                return self._plot_price_chart_matplotlib(
                    price_data=price_data,
                    trades_data=trades_data,
                    symbol=symbol,
                    chart_type=chart_type,
                    show_volume=show_volume,
                    indicators_data=indicators_data,
                    height=height,
                    width=width,
                    allow_download=allow_download
                )
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ biểu đồ giá: {e}")
            st.error(f"Lỗi khi vẽ biểu đồ giá: {e}")
            return None
    
    def _plot_price_chart_plotly(
        self,
        price_data: pd.DataFrame,
        trades_data: Optional[pd.DataFrame],
        symbol: str,
        chart_type: str,
        show_volume: bool,
        indicators_data: Dict[str, pd.Series],
        height: int,
        width: int,
        allow_download: bool
    ) -> go.Figure:
        """
        Vẽ biểu đồ giá sử dụng Plotly.
        
        Args: (xem phương thức plot_price_chart)
            
        Returns:
            Đối tượng Figure của plotly
        """
        # Xác định số lượng subplots
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.8, 0.2],
                subplot_titles=(f"{symbol} - Biểu đồ giá", "Khối lượng")
            )
        else:
            fig = make_subplots(
                rows=1, cols=1, 
                subplot_titles=(f"{symbol} - Biểu đồ giá",)
            )
            
        # Thêm dữ liệu nến (candlestick) hoặc các loại biểu đồ khác
        if chart_type == "candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=price_data.index,
                    open=price_data['open'],
                    high=price_data['high'],
                    low=price_data['low'],
                    close=price_data['close'],
                    name="Giá",
                    increasing_line_color='#26a69a', 
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
        elif chart_type == "line":
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data['close'],
                    mode='lines',
                    name="Giá đóng cửa",
                    line=dict(color='#2196f3', width=2)
                ),
                row=1, col=1
            )
        elif chart_type == "ohlc":
            fig.add_trace(
                go.Ohlc(
                    x=price_data.index,
                    open=price_data['open'],
                    high=price_data['high'],
                    low=price_data['low'],
                    close=price_data['close'],
                    name="Giá",
                    increasing_line_color='#26a69a', 
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
        elif chart_type == "area":
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=price_data['close'],
                    mode='lines',
                    fill='tozeroy',
                    name="Giá đóng cửa",
                    line=dict(color='#2196f3', width=1),
                    fillcolor='rgba(33, 150, 243, 0.2)'
                ),
                row=1, col=1
            )
            
        # Thêm khối lượng nếu cần
        if show_volume and 'volume' in price_data.columns:
            colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' for i, row in price_data.iterrows()]
            
            fig.add_trace(
                go.Bar(
                    x=price_data.index,
                    y=price_data['volume'],
                    name="Khối lượng",
                    marker_color=colors,
                    opacity=0.8
                ),
                row=2, col=1
            )
            
        # Thêm các chỉ báo kỹ thuật
        for name, data in indicators_data.items():
            # Phân biệt màu sắc cho các chỉ báo
            if "SMA" in name:
                color = '#7e57c2'  # Tím
                dash = 'solid'
            elif "EMA" in name:
                color = '#fb8c00'  # Cam
                dash = 'solid'
            elif "BB Upper" in name:
                color = '#43a047'  # Xanh lá
                dash = 'dash'
            elif "BB Lower" in name:
                color = '#43a047'  # Xanh lá
                dash = 'dash'
            elif "RSI" in name:
                color = '#e53935'  # Đỏ
                dash = 'solid'
            elif "MACD" in name:
                color = '#3f51b5'  # Xanh dương
                dash = 'solid'
            elif "Signal" in name:
                color = '#ff9800'  # Cam đậm
                dash = 'dash'
            else:
                color = '#757575'  # Xám
                dash = 'solid'
                
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=data,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=1, dash=dash)
                ),
                row=1, col=1
            )
            
        # Thêm đánh dấu giao dịch nếu có
        if trades_data is not None and len(trades_data) > 0:
            # Đảm bảo các cột cần thiết tồn tại
            required_columns = ['entry_time', 'exit_time', 'side', 'entry_price', 'exit_price']
            missing_columns = [col for col in required_columns if col not in trades_data.columns]
            
            if not missing_columns:
                # Chuẩn hóa dữ liệu
                if not pd.api.types.is_datetime64_any_dtype(trades_data['entry_time']):
                    trades_data['entry_time'] = pd.to_datetime(trades_data['entry_time'])
                if not pd.api.types.is_datetime64_any_dtype(trades_data['exit_time']):
                    trades_data['exit_time'] = pd.to_datetime(trades_data['exit_time'])
                
                # Đánh dấu điểm vào lệnh
                buy_entries = trades_data[trades_data['side'] == 'LONG']
                sell_entries = trades_data[trades_data['side'] == 'SHORT']
                
                # Thêm điểm vào lệnh mua
                if len(buy_entries) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_entries['entry_time'],
                            y=buy_entries['entry_price'],
                            mode='markers',
                            name='Vào lệnh MUA',
                            marker=dict(
                                color='green',
                                size=10,
                                symbol='triangle-up',
                                line=dict(width=2, color='darkgreen')
                            ),
                            hovertemplate=(
                                '<b>Vào lệnh MUA</b><br>' +
                                'Thời gian: %{x}<br>' +
                                'Giá: %{y:.8f}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        row=1, col=1
                    )
                    
                    # Thêm điểm thoát lệnh mua
                    fig.add_trace(
                        go.Scatter(
                            x=buy_entries['exit_time'],
                            y=buy_entries['exit_price'],
                            mode='markers',
                            name='Thoát lệnh MUA',
                            marker=dict(
                                color='lightgreen',
                                size=10,
                                symbol='circle',
                                line=dict(width=2, color='darkgreen')
                            ),
                            hovertemplate=(
                                '<b>Thoát lệnh MUA</b><br>' +
                                'Thời gian: %{x}<br>' +
                                'Giá: %{y:.8f}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        row=1, col=1
                    )
                
                # Thêm điểm vào lệnh bán
                if len(sell_entries) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_entries['entry_time'],
                            y=sell_entries['entry_price'],
                            mode='markers',
                            name='Vào lệnh BÁN',
                            marker=dict(
                                color='red',
                                size=10,
                                symbol='triangle-down',
                                line=dict(width=2, color='darkred')
                            ),
                            hovertemplate=(
                                '<b>Vào lệnh BÁN</b><br>' +
                                'Thời gian: %{x}<br>' +
                                'Giá: %{y:.8f}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        row=1, col=1
                    )
                    
                    # Thêm điểm thoát lệnh bán
                    fig.add_trace(
                        go.Scatter(
                            x=sell_entries['exit_time'],
                            y=sell_entries['exit_price'],
                            mode='markers',
                            name='Thoát lệnh BÁN',
                            marker=dict(
                                color='lightcoral',
                                size=10,
                                symbol='circle',
                                line=dict(width=2, color='darkred')
                            ),
                            hovertemplate=(
                                '<b>Thoát lệnh BÁN</b><br>' +
                                'Thời gian: %{x}<br>' +
                                'Giá: %{y:.8f}<br>' +
                                '<extra></extra>'
                            )
                        ),
                        row=1, col=1
                    )
                    
                # Vẽ đường kết nối giữa điểm vào lệnh và thoát lệnh
                for _, trade in trades_data.iterrows():
                    color = 'green' if trade['side'] == 'LONG' else 'red'
                    color = 'rgba(0, 128, 0, 0.5)' if trade['side'] == 'LONG' else 'rgba(255, 0, 0, 0.5)'
                    
                    if not pd.isna(trade['entry_time']) and not pd.isna(trade['exit_time']):
                        fig.add_shape(
                            type="line",
                            x0=trade['entry_time'],
                            y0=trade['entry_price'],
                            x1=trade['exit_time'],
                            y1=trade['exit_price'],
                            line=dict(
                                color=color,
                                width=1,
                                dash="dot",
                            ),
                            row=1, col=1
                        )
        
        # Cập nhật layout
        template = "plotly_white" if self.style != "dark" else "plotly_dark"
        fig.update_layout(
            height=height,
            width=width,
            template=template,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        # Thiết lập các trục
        fig.update_yaxes(title_text="Giá", row=1, col=1)
        if show_volume:
            fig.update_yaxes(title_text="Khối lượng", row=2, col=1)
            
        # Thiết lập grid
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)' if self.style != "dark" else 'rgba(255,255,255,0.1)'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)' if self.style != "dark" else 'rgba(255,255,255,0.1)'
        )
        
        # Hiển thị biểu đồ trên Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Thêm chức năng tải xuống nếu cần
        if allow_download:
            self._add_download_button(fig, f"{symbol}_price_chart.html", "Tải xuống biểu đồ")
            
        return fig
    
    def _plot_price_chart_matplotlib(
        self,
        price_data: pd.DataFrame,
        trades_data: Optional[pd.DataFrame],
        symbol: str,
        chart_type: str,
        show_volume: bool,
        indicators_data: Dict[str, pd.Series],
        height: int,
        width: int,
        allow_download: bool
    ) -> plt.Figure:
        """
        Vẽ biểu đồ giá sử dụng Matplotlib.
        
        Args: (xem phương thức plot_price_chart)
            
        Returns:
            Đối tượng Figure của matplotlib
        """
        # Tính toán kích thước trong inch (1 px = 1/96 inch)
        figsize = (width / 96, height / 96)
        
        # Xác định số lượng subplots
        if show_volume:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
            axes = [ax1, ax2]
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
            axes = [ax1]
            
        # Vẽ biểu đồ giá
        if chart_type == "candlestick":
            # Vẽ nến từ dữ liệu OHLC
            up = price_data[price_data.close >= price_data.open]
            down = price_data[price_data.close < price_data.open]
            
            # Nến tăng
            ax1.bar(up.index, up.close - up.open, bottom=up.open, width=0.7, color='green', alpha=0.8)
            ax1.bar(up.index, up.high - up.close, bottom=up.close, width=0.1, color='green', alpha=0.8)
            ax1.bar(up.index, up.low - up.open, bottom=up.open, width=0.1, color='green', alpha=0.8)
            
            # Nến giảm
            ax1.bar(down.index, down.close - down.open, bottom=down.open, width=0.7, color='red', alpha=0.8)
            ax1.bar(down.index, down.high - down.open, bottom=down.open, width=0.1, color='red', alpha=0.8)
            ax1.bar(down.index, down.low - down.close, bottom=down.close, width=0.1, color='red', alpha=0.8)
            
        elif chart_type == "line":
            ax1.plot(price_data.index, price_data.close, label='Giá đóng cửa', color='blue', linewidth=2)
            
        elif chart_type == "ohlc":
            from mplfinance.original_flavor import candlestick_ohlc
            import matplotlib.dates as mdates
            
            # Chuyển đổi dữ liệu sang định dạng OHLC
            ohlc = price_data.reset_index()
            ohlc['date_num'] = mdates.date2num(ohlc['timestamp'] if 'timestamp' in ohlc.columns else ohlc.index)
            ohlc = ohlc[['date_num', 'open', 'high', 'low', 'close']]
            
            # Vẽ biểu đồ OHLC
            candlestick_ohlc(ax1, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
        elif chart_type == "area":
            ax1.fill_between(price_data.index, 0, price_data.close, alpha=0.3, color='blue')
            ax1.plot(price_data.index, price_data.close, color='blue', linewidth=2)
            
        # Thêm các chỉ báo kỹ thuật
        for name, data in indicators_data.items():
            # Phân biệt màu sắc cho các chỉ báo
            if "SMA" in name:
                color = 'purple'
                linestyle = '-'
            elif "EMA" in name:
                color = 'orange'
                linestyle = '-'
            elif "BB Upper" in name:
                color = 'green'
                linestyle = '--'
            elif "BB Lower" in name:
                color = 'green'
                linestyle = '--'
            elif "RSI" in name:
                color = 'red'
                linestyle = '-'
            elif "MACD" in name:
                color = 'blue'
                linestyle = '-'
            elif "Signal" in name:
                color = 'orange'
                linestyle = '--'
            else:
                color = 'gray'
                linestyle = '-'
                
            ax1.plot(price_data.index, data, label=name, color=color, linestyle=linestyle, linewidth=1.5)
            
        # Thêm khối lượng nếu cần
        if show_volume and 'volume' in price_data.columns:
            # Màu cho khối lượng dựa trên giá tăng/giảm
            colors = ['green' if close >= open else 'red' for close, open in zip(price_data.close, price_data.open)]
            ax2.bar(price_data.index, price_data.volume, color=colors, alpha=0.8, width=0.8)
            ax2.set_ylabel('Khối lượng')
            
        # Thêm đánh dấu giao dịch nếu có
        if trades_data is not None and len(trades_data) > 0:
            # Đảm bảo các cột cần thiết tồn tại
            required_columns = ['entry_time', 'exit_time', 'side', 'entry_price', 'exit_price']
            missing_columns = [col for col in required_columns if col not in trades_data.columns]
            
            if not missing_columns:
                # Chuẩn hóa dữ liệu
                if not pd.api.types.is_datetime64_any_dtype(trades_data['entry_time']):
                    trades_data['entry_time'] = pd.to_datetime(trades_data['entry_time'])
                if not pd.api.types.is_datetime64_any_dtype(trades_data['exit_time']):
                    trades_data['exit_time'] = pd.to_datetime(trades_data['exit_time'])
                
                # Đánh dấu điểm vào lệnh
                buy_entries = trades_data[trades_data['side'] == 'LONG']
                sell_entries = trades_data[trades_data['side'] == 'SHORT']
                
                # Thêm điểm vào lệnh mua
                if len(buy_entries) > 0:
                    ax1.scatter(
                        buy_entries['entry_time'], 
                        buy_entries['entry_price'], 
                        marker='^', 
                        color='green', 
                        s=100,
                        label='Vào lệnh MUA'
                    )
                    
                    # Thêm điểm thoát lệnh mua
                    ax1.scatter(
                        buy_entries['exit_time'], 
                        buy_entries['exit_price'], 
                        marker='o', 
                        color='lightgreen', 
                        s=100,
                        label='Thoát lệnh MUA'
                    )
                
                # Thêm điểm vào lệnh bán
                if len(sell_entries) > 0:
                    ax1.scatter(
                        sell_entries['entry_time'], 
                        sell_entries['entry_price'], 
                        marker='v', 
                        color='red', 
                        s=100,
                        label='Vào lệnh BÁN'
                    )
                    
                    # Thêm điểm thoát lệnh bán
                    ax1.scatter(
                        sell_entries['exit_time'], 
                        sell_entries['exit_price'], 
                        marker='o', 
                        color='lightcoral', 
                        s=100,
                        label='Thoát lệnh BÁN'
                    )
                    
                # Vẽ đường kết nối giữa điểm vào lệnh và thoát lệnh
                for _, trade in trades_data.iterrows():
                    color = 'green' if trade['side'] == 'LONG' else 'red'
                    if not pd.isna(trade['entry_time']) and not pd.isna(trade['exit_time']):
                        ax1.plot(
                            [trade['entry_time'], trade['exit_time']], 
                            [trade['entry_price'], trade['exit_price']], 
                            color=color, 
                            linestyle='--', 
                            alpha=0.5,
                            linewidth=1
                        )
                    
        # Thiết lập tiêu đề và nhãn
        ax1.set_title(f"{symbol} - Biểu đồ giá")
        ax1.set_ylabel('Giá')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Định dạng trục x
        if len(price_data) > 0:
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
            axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
        # Điều chỉnh layout
        plt.tight_layout()
        
        # Hiển thị biểu đồ trên Streamlit
        st.pyplot(fig)
        
        # Thêm chức năng tải xuống nếu cần
        if allow_download:
            self._add_download_button_matplotlib(fig, f"{symbol}_price_chart.png", "Tải xuống biểu đồ")
            
        return fig
    
    def plot_portfolio_performance(
        self,
        portfolio_data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        title: str = "Hiệu suất danh mục đầu tư",
        metrics: List[str] = ["equity", "drawdown", "returns"],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        height: int = 800,
        add_stats: bool = True,
        allow_download: bool = False
    ) -> Any:
        """
        Vẽ biểu đồ hiệu suất danh mục đầu tư.
        
        Args:
            portfolio_data: DataFrame chứa dữ liệu hiệu suất danh mục
            benchmark_data: DataFrame chứa dữ liệu chuẩn so sánh
            title: Tiêu đề biểu đồ
            metrics: Danh sách các chỉ số cần hiển thị
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc
            height: Chiều cao biểu đồ
            add_stats: Thêm thống kê tóm tắt
            allow_download: Cho phép tải xuống biểu đồ
            
        Returns:
            Đối tượng figure hoặc None
        """
        # Kiểm tra dữ liệu đầu vào
        if portfolio_data is None or len(portfolio_data) == 0:
            st.error("Không có dữ liệu danh mục đầu tư để hiển thị!")
            return None
            
        # Chuyển đổi index thành datetime nếu chưa phải
        if not isinstance(portfolio_data.index, pd.DatetimeIndex):
            # Kiểm tra xem có cột timestamp/datetime không
            if 'timestamp' in portfolio_data.columns:
                portfolio_data = portfolio_data.set_index('timestamp')
            elif 'datetime' in portfolio_data.columns:
                portfolio_data = portfolio_data.set_index('datetime')
            elif 'date' in portfolio_data.columns:
                portfolio_data = portfolio_data.set_index('date')
            else:
                # Nếu không có cột thời gian, sử dụng chỉ số làm index
                portfolio_data = portfolio_data.copy()
                portfolio_data.index = pd.to_datetime(portfolio_data.index)
                
        # Chuẩn hóa tên cột (viết thường)
        portfolio_data.columns = [col.lower() for col in portfolio_data.columns]
        
        # Lọc theo thời gian
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            portfolio_data = portfolio_data[portfolio_data.index >= start_date]
            
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            portfolio_data = portfolio_data[portfolio_data.index <= end_date]
            
        # Chuẩn bị dữ liệu chuẩn so sánh nếu có
        if benchmark_data is not None:
            # Chuẩn hóa benchmark_data
            if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                if 'timestamp' in benchmark_data.columns:
                    benchmark_data = benchmark_data.set_index('timestamp')
                elif 'datetime' in benchmark_data.columns:
                    benchmark_data = benchmark_data.set_index('datetime')
                elif 'date' in benchmark_data.columns:
                    benchmark_data = benchmark_data.set_index('date')
                    
            # Chuẩn hóa tên cột
            benchmark_data.columns = [col.lower() for col in benchmark_data.columns]
            
            # Lọc theo thời gian
            if start_date is not None:
                benchmark_data = benchmark_data[benchmark_data.index >= start_date]
            if end_date is not None:
                benchmark_data = benchmark_data[benchmark_data.index <= end_date]
                
            # Chuẩn hóa giá trị
            benchmark_col = 'close' if 'close' in benchmark_data.columns else benchmark_data.columns[0]
            benchmark_equity = benchmark_data[benchmark_col]
            # Chuẩn hóa về giá trị ban đầu
            benchmark_norm = benchmark_equity / benchmark_equity.iloc[0]
        else:
            benchmark_norm = None
            
        # Chuẩn bị các chỉ số cần hiển thị
        plot_data = {}
        
        # Chỉ số tài sản (equity)
        if "equity" in metrics:
            if "equity" in portfolio_data.columns:
                plot_data["equity"] = portfolio_data["equity"]
                # Chuẩn hóa về giá trị ban đầu
                plot_data["equity_norm"] = portfolio_data["equity"] / portfolio_data["equity"].iloc[0]
            elif "balance" in portfolio_data.columns:
                plot_data["equity"] = portfolio_data["balance"]
                plot_data["equity_norm"] = portfolio_data["balance"] / portfolio_data["balance"].iloc[0]
            elif "value" in portfolio_data.columns:
                plot_data["equity"] = portfolio_data["value"]
                plot_data["equity_norm"] = portfolio_data["value"] / portfolio_data["value"].iloc[0]
                
        # Drawdown
        if "drawdown" in metrics:
            if "drawdown" in portfolio_data.columns:
                plot_data["drawdown"] = portfolio_data["drawdown"]
            else:
                # Tính drawdown từ equity
                if "equity" in plot_data:
                    rolling_max = plot_data["equity"].cummax()
                    plot_data["drawdown"] = (plot_data["equity"] - rolling_max) / rolling_max
                    
        # Lợi nhuận hàng ngày
        if "returns" in metrics:
            if "returns" in portfolio_data.columns:
                plot_data["returns"] = portfolio_data["returns"]
            else:
                # Tính lợi nhuận từ equity
                if "equity" in plot_data:
                    plot_data["returns"] = plot_data["equity"].pct_change().fillna(0)
                    
        try:
            if self.use_plotly:
                return self._plot_portfolio_performance_plotly(
                    plot_data=plot_data,
                    benchmark_norm=benchmark_norm,
                    title=title,
                    metrics=metrics,
                    height=height,
                    add_stats=add_stats,
                    allow_download=allow_download
                )
            else:
                return self._plot_portfolio_performance_matplotlib(
                    plot_data=plot_data,
                    benchmark_norm=benchmark_norm,
                    title=title,
                    metrics=metrics,
                    height=height,
                    add_stats=add_stats,
                    allow_download=allow_download
                )
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ biểu đồ hiệu suất danh mục: {e}")
            st.error(f"Lỗi khi vẽ biểu đồ hiệu suất danh mục: {e}")
            return None
    
    def _plot_portfolio_performance_plotly(
        self,
        plot_data: Dict[str, pd.Series],
        benchmark_norm: Optional[pd.Series],
        title: str,
        metrics: List[str],
        height: int,
        add_stats: bool,
        allow_download: bool
    ) -> go.Figure:
        """
        Vẽ biểu đồ hiệu suất danh mục đầu tư sử dụng Plotly.
        
        Args: (xem phương thức plot_portfolio_performance)
            
        Returns:
            Đối tượng Figure của plotly
        """
        # Xác định số lượng subplots
        n_plots = 0
        if "equity" in metrics and "equity" in plot_data:
            n_plots += 1
        if "drawdown" in metrics and "drawdown" in plot_data:
            n_plots += 1
        if "returns" in metrics and "returns" in plot_data:
            n_plots += 1
            
        if n_plots == 0:
            st.error("Không có chỉ số nào để hiển thị!")
            return None
            
        # Tạo subplots
        fig = make_subplots(
            rows=n_plots, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=self._get_subplot_titles(metrics, plot_data)
        )
        
        # Vẽ các chỉ số
        current_plot = 1
        
        # Vẽ equity
        if "equity" in metrics and "equity" in plot_data:
            # Vẽ equity
            fig.add_trace(
                go.Scatter(
                    x=plot_data["equity"].index,
                    y=plot_data["equity_norm"],
                    mode='lines',
                    name='Danh mục',
                    line=dict(color='#2196f3', width=2)
                ),
                row=current_plot, col=1
            )
            
            # Vẽ benchmark nếu có
            if benchmark_norm is not None:
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_norm.index,
                        y=benchmark_norm,
                        mode='lines',
                        name='Chuẩn',
                        line=dict(color='#757575', width=1.5, dash='dash')
                    ),
                    row=current_plot, col=1
                )
                
            current_plot += 1
            
        # Vẽ drawdown
        if "drawdown" in metrics and "drawdown" in plot_data:
            fig.add_trace(
                go.Scatter(
                    x=plot_data["drawdown"].index,
                    y=plot_data["drawdown"] * 100,  # Chuyển sang phần trăm
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='#ef5350', width=1.5),
                    fillcolor='rgba(239, 83, 80, 0.2)'
                ),
                row=current_plot, col=1
            )
            
            current_plot += 1
            
        # Vẽ returns
        if "returns" in metrics and "returns" in plot_data:
            colors = ['#26a69a' if r >= 0 else '#ef5350' for r in plot_data["returns"]]
            
            fig.add_trace(
                go.Bar(
                    x=plot_data["returns"].index,
                    y=plot_data["returns"] * 100,  # Chuyển sang phần trăm
                    name='Lợi nhuận hàng ngày',
                    marker_color=colors
                ),
                row=current_plot, col=1
            )
            
        # Cập nhật layout
        template = "plotly_white" if self.style != "dark" else "plotly_dark"
        fig.update_layout(
            title=title,
            height=height,
            template=template,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=60, b=10)
        )
        
        # Thiết lập các trục
        current_plot = 1
        if "equity" in metrics and "equity" in plot_data:
            fig.update_yaxes(title_text="Giá trị chuẩn hóa", row=current_plot, col=1)
            current_plot += 1
            
        if "drawdown" in metrics and "drawdown" in plot_data:
            fig.update_yaxes(title_text="Drawdown (%)", row=current_plot, col=1)
            current_plot += 1
            
        if "returns" in metrics and "returns" in plot_data:
            fig.update_yaxes(title_text="Lợi nhuận (%)", row=current_plot, col=1)
            
        # Thiết lập grid
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)' if self.style != "dark" else 'rgba(255,255,255,0.1)'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)' if self.style != "dark" else 'rgba(255,255,255,0.1)'
        )
        
        # Hiển thị thống kê nếu cần
        if add_stats and "equity" in plot_data:
            stats = self._calculate_performance_stats(plot_data["equity"])
            
            # Hiển thị bảng thống kê
            cols = st.columns(4)
            cols[0].metric("Tổng lợi nhuận", f"{stats['total_return']:.2%}")
            cols[1].metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
            cols[2].metric("Max Drawdown", f"{stats['max_drawdown']:.2%}")
            cols[3].metric("Volatility", f"{stats['volatility']:.2%}")
            
        # Hiển thị biểu đồ trên Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Thêm chức năng tải xuống nếu cần
        if allow_download:
            self._add_download_button(fig, "portfolio_performance.html", "Tải xuống biểu đồ")
            
        return fig
    
    def _plot_portfolio_performance_matplotlib(
        self,
        plot_data: Dict[str, pd.Series],
        benchmark_norm: Optional[pd.Series],
        title: str,
        metrics: List[str],
        height: int,
        add_stats: bool,
        allow_download: bool
    ) -> plt.Figure:
        """
        Vẽ biểu đồ hiệu suất danh mục đầu tư sử dụng Matplotlib.
        
        Args: (xem phương thức plot_portfolio_performance)
            
        Returns:
            Đối tượng Figure của matplotlib
        """
        # Xác định số lượng subplots
        n_plots = 0
        if "equity" in metrics and "equity" in plot_data:
            n_plots += 1
        if "drawdown" in metrics and "drawdown" in plot_data:
            n_plots += 1
        if "returns" in metrics and "returns" in plot_data:
            n_plots += 1
            
        if n_plots == 0:
            st.error("Không có chỉ số nào để hiển thị!")
            return None
            
        # Tính toán kích thước trong inch (1 px = 1/96 inch)
        figsize = (10, height / 96)
        
        # Tạo figure và axes
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True, gridspec_kw={'hspace': 0.3})
        
        # Đảm bảo axes là list kể cả khi chỉ có 1 metric
        if n_plots == 1:
            axes = [axes]
            
        # Vẽ các chỉ số
        current_plot = 0
        
        # Vẽ equity
        if "equity" in metrics and "equity" in plot_data:
            ax = axes[current_plot]
            
            # Vẽ equity
            ax.plot(
                plot_data["equity_norm"].index,
                plot_data["equity_norm"],
                label='Danh mục',
                color='blue',
                linewidth=2
            )
            
            # Vẽ benchmark nếu có
            if benchmark_norm is not None:
                ax.plot(
                    benchmark_norm.index,
                    benchmark_norm,
                    label='Chuẩn',
                    color='gray',
                    linestyle='--',
                    linewidth=1.5
                )
                
            ax.set_title("Giá trị danh mục đầu tư (chuẩn hóa)")
            ax.set_ylabel("Giá trị chuẩn hóa")
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            current_plot += 1
            
        # Vẽ drawdown
        if "drawdown" in metrics and "drawdown" in plot_data:
            ax = axes[current_plot]
            
            ax.fill_between(
                plot_data["drawdown"].index,
                0,
                plot_data["drawdown"] * 100,  # Chuyển sang phần trăm
                color='red',
                alpha=0.3,
                label='Drawdown'
            )
            
            ax.plot(
                plot_data["drawdown"].index,
                plot_data["drawdown"] * 100,
                color='red',
                linewidth=1,
                alpha=0.7
            )
            
            ax.set_title("Drawdown")
            ax.set_ylabel("Drawdown (%)")
            ax.grid(True, alpha=0.3)
            
            # Hiển thị mức độ drawdown
            max_dd = plot_data["drawdown"].min() * 100
            ax.axhline(max_dd, color='black', linestyle=':', alpha=0.6)
            ax.text(
                plot_data["drawdown"].index[0],
                max_dd * 1.1,
                f'Max DD: {max_dd:.2f}%',
                fontsize=9,
                verticalalignment='bottom'
            )
            
            current_plot += 1
            
        # Vẽ returns
        if "returns" in metrics and "returns" in plot_data:
            ax = axes[current_plot]
            
            # Tạo colors array cho các thanh
            colors = ['green' if r >= 0 else 'red' for r in plot_data["returns"]]
            
            ax.bar(
                plot_data["returns"].index,
                plot_data["returns"] * 100,  # Chuyển sang phần trăm
                color=colors,
                alpha=0.7,
                label='Lợi nhuận hàng ngày'
            )
            
            ax.set_title("Lợi nhuận hàng ngày")
            ax.set_ylabel("Lợi nhuận (%)")
            ax.grid(True, alpha=0.3)
            
        # Định dạng trục x
        if len(plot_data) > 0:
            for ax in axes:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
        # Thiết lập tiêu đề
        fig.suptitle(title, fontsize=16)
        
        # Điều chỉnh layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Hiển thị thống kê nếu cần
        if add_stats and "equity" in plot_data:
            stats = self._calculate_performance_stats(plot_data["equity"])
            
            # Hiển thị bảng thống kê
            cols = st.columns(4)
            cols[0].metric("Tổng lợi nhuận", f"{stats['total_return']:.2%}")
            cols[1].metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
            cols[2].metric("Max Drawdown", f"{stats['max_drawdown']:.2%}")
            cols[3].metric("Volatility", f"{stats['volatility']:.2%}")
            
        # Hiển thị biểu đồ trên Streamlit
        st.pyplot(fig)
        
        # Thêm chức năng tải xuống nếu cần
        if allow_download:
            self._add_download_button_matplotlib(fig, "portfolio_performance.png", "Tải xuống biểu đồ")
            
        return fig
    
    def display_trade_details(
        self,
        trades_data: pd.DataFrame,
        symbol: Optional[str] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        include_metrics: bool = True,
        allow_download: bool = True
    ) -> pd.DataFrame:
        """
        Hiển thị bảng chi tiết các giao dịch và thống kê.
        
        Args:
            trades_data: DataFrame chứa dữ liệu giao dịch
            symbol: Lọc theo mã token/cặp giao dịch
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc
            include_metrics: Hiển thị thống kê giao dịch
            allow_download: Cho phép tải xuống dữ liệu
            
        Returns:
            DataFrame đã được lọc
        """
        # Kiểm tra dữ liệu đầu vào
        if trades_data is None or len(trades_data) == 0:
            st.error("Không có dữ liệu giao dịch để hiển thị!")
            return pd.DataFrame()
            
        # Chuẩn hóa các cột thời gian
        for col in ['entry_time', 'exit_time']:
            if col in trades_data.columns and not pd.api.types.is_datetime64_any_dtype(trades_data[col]):
                trades_data[col] = pd.to_datetime(trades_data[col])
                
        # Lọc theo symbol
        filtered_data = trades_data
        if symbol is not None and 'symbol' in trades_data.columns:
            filtered_data = filtered_data[filtered_data['symbol'] == symbol]
            
        # Lọc theo thời gian
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if 'entry_time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['entry_time'] >= start_date]
                
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            if 'entry_time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['entry_time'] <= end_date]
                
        # Sắp xếp theo thời gian
        if 'entry_time' in filtered_data.columns:
            filtered_data = filtered_data.sort_values('entry_time', ascending=False)
            
        # Tính toán thời gian giao dịch nếu chưa có
        if 'entry_time' in filtered_data.columns and 'exit_time' in filtered_data.columns and 'duration' not in filtered_data.columns:
            filtered_data['duration'] = (filtered_data['exit_time'] - filtered_data['entry_time']).dt.total_seconds() / 3600  # Hours
            
        # Chuyển đổi các cột giá & lợi nhuận
        numeric_cols = ['entry_price', 'exit_price', 'profit', 'profit_pct', 'fees']
        for col in numeric_cols:
            if col in filtered_data.columns and not pd.api.types.is_numeric_dtype(filtered_data[col]):
                filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')
                
        # Hiển thị thống kê giao dịch nếu cần
        if include_metrics and len(filtered_data) > 0:
            self._display_trade_metrics(filtered_data)
            
        # Hiển thị bảng giao dịch
        if len(filtered_data) > 0:
            st.subheader(f"Chi tiết giao dịch ({len(filtered_data)} giao dịch)")
            
            # Chọn các cột cần hiển thị
            display_columns = []
            
            # Cột cơ bản
            basic_cols = ['id', 'symbol', 'side', 'entry_time', 'exit_time', 'duration', 
                          'entry_price', 'exit_price', 'quantity', 'profit', 'profit_pct', 'fees']
            
            for col in basic_cols:
                if col in filtered_data.columns:
                    display_columns.append(col)
                    
            # Thêm các cột khác
            additional_cols = [col for col in filtered_data.columns if col not in basic_cols and col not in display_columns]
            display_columns.extend(additional_cols)
            
            # Tạo bản sao để định dạng
            display_data = filtered_data[display_columns].copy()
            
            # Định dạng các cột thời gian
            for col in ['entry_time', 'exit_time']:
                if col in display_data.columns:
                    display_data[col] = display_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
            # Định dạng các cột lợi nhuận
            if 'profit' in display_data.columns:
                display_data['profit'] = display_data['profit'].map(lambda x: f"{x:.6f}")
                
            if 'profit_pct' in display_data.columns:
                display_data['profit_pct'] = display_data['profit_pct'].map(lambda x: f"{x:.2%}")
                
            # Định dạng thời gian giao dịch
            if 'duration' in display_data.columns:
                display_data['duration'] = display_data['duration'].map(lambda x: f"{x:.2f} h")
                
            # Hiển thị bảng với định dạng màu sắc
            st.dataframe(
                display_data.style.apply(
                    lambda row: ['background-color: rgba(0, 255, 0, 0.1)' if row['profit'] > 0 else 'background-color: rgba(255, 0, 0, 0.1)' for _ in row] 
                    if 'profit' in row else [''] * len(row),
                    axis=1
                ),
                use_container_width=True
            )
            
            # Thêm nút tải xuống
            if allow_download:
                csv = filtered_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="trades.csv" class="btn">Tải xuống dữ liệu giao dịch (CSV)</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("Không có giao dịch nào trong khoảng thời gian đã chọn.")
            
        return filtered_data
    
    def _display_trade_metrics(self, trades_data: pd.DataFrame) -> None:
        """
        Hiển thị thống kê giao dịch.
        
        Args:
            trades_data: DataFrame chứa dữ liệu giao dịch
        """
        # Tính toán các chỉ số cơ bản
        total_trades = len(trades_data)
        
        # Tính số lượng giao dịch thắng/thua
        if 'profit' in trades_data.columns:
            winning_trades = len(trades_data[trades_data['profit'] > 0])
            losing_trades = len(trades_data[trades_data['profit'] <= 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Tính tổng lợi nhuận
            total_profit = trades_data['profit'].sum()
            avg_profit = trades_data['profit'].mean()
            
            # Tính lợi nhuận theo loại giao dịch
            if winning_trades > 0:
                avg_win = trades_data[trades_data['profit'] > 0]['profit'].mean()
                max_win = trades_data['profit'].max()
            else:
                avg_win = 0
                max_win = 0
                
            if losing_trades > 0:
                avg_loss = trades_data[trades_data['profit'] <= 0]['profit'].mean()
                max_loss = trades_data['profit'].min()
            else:
                avg_loss = 0
                max_loss = 0
                
            # Tính profit factor
            total_win = trades_data[trades_data['profit'] > 0]['profit'].sum()
            total_loss = abs(trades_data[trades_data['profit'] < 0]['profit'].sum())
            profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
            
            # Tính trung bình thời gian giao dịch
            if 'duration' in trades_data.columns:
                avg_duration = trades_data['duration'].mean()
            else:
                avg_duration = None
                
            # Tính tỷ lệ lợi nhuận trên rủi ro
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                
            # Hiển thị thống kê
            st.subheader("Thống kê giao dịch")
            
            # Hiển thị các chỉ số chính
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Tổng giao dịch", f"{total_trades}")
            col2.metric("Tỷ lệ thắng", f"{win_rate:.2%}")
            col3.metric("Tổng lợi nhuận", f"{total_profit:.6f}")
            col4.metric("Profit Factor", f"{profit_factor:.2f}")
            
            # Hiển thị thống kê chi tiết
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Số giao dịch thắng", f"{winning_trades}")
            col1.metric("Lợi nhuận TB", f"{avg_profit:.6f}")
            col1.metric("Lợi nhuận TB (Thắng)", f"{avg_win:.6f}")
            col1.metric("Lợi nhuận tối đa", f"{max_win:.6f}")
            
            col2.metric("Số giao dịch thua", f"{losing_trades}")
            col2.metric("Tổng phí", f"{trades_data['fees'].sum():.6f}" if 'fees' in trades_data.columns else "N/A")
            col2.metric("Thua lỗ TB", f"{avg_loss:.6f}")
            col2.metric("Thua lỗ tối đa", f"{max_loss:.6f}")
            
            col3.metric("TG giao dịch TB", f"{avg_duration:.2f} h" if avg_duration is not None else "N/A")
            col3.metric("Tỷ lệ lợi nhuận/rủi ro", f"{win_loss_ratio:.2f}")
            
            # Hiển thị thông tin theo loại giao dịch (long/short)
            if 'side' in trades_data.columns:
                long_trades = trades_data[trades_data['side'] == 'LONG']
                short_trades = trades_data[trades_data['side'] == 'SHORT']
                
                long_count = len(long_trades)
                short_count = len(short_trades)
                
                if long_count > 0:
                    long_win_rate = len(long_trades[long_trades['profit'] > 0]) / long_count
                    long_profit = long_trades['profit'].sum()
                else:
                    long_win_rate = 0
                    long_profit = 0
                    
                if short_count > 0:
                    short_win_rate = len(short_trades[short_trades['profit'] > 0]) / short_count
                    short_profit = short_trades['profit'].sum()
                else:
                    short_win_rate = 0
                    short_profit = 0
                    
                col4.metric("Giao dịch Long", f"{long_count}")
                col4.metric("Tỷ lệ thắng Long", f"{long_win_rate:.2%}")
                col4.metric("Lợi nhuận Long", f"{long_profit:.6f}")
                
                col3.metric("Giao dịch Short", f"{short_count}")
                col3.metric("Tỷ lệ thắng Short", f"{short_win_rate:.2%}")
                col3.metric("Lợi nhuận Short", f"{short_profit:.6f}")
                
            # Vẽ biểu đồ phân phối lợi nhuận
            self._plot_trade_profit_distribution(trades_data)
            
            # Phân tích giao dịch theo khung thời gian
            self._plot_trade_time_analysis(trades_data)
    
    def _plot_trade_profit_distribution(self, trades_data: pd.DataFrame) -> None:
        """
        Vẽ biểu đồ phân phối lợi nhuận giao dịch.
        
        Args:
            trades_data: DataFrame chứa dữ liệu giao dịch
        """
        if 'profit' not in trades_data.columns or len(trades_data) < 2:
            return
            
        st.subheader("Phân phối lợi nhuận")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Vẽ histogram phân phối lợi nhuận
            if self.use_plotly:
                fig = px.histogram(
                    trades_data, 
                    x='profit', 
                    nbins=20,
                    color_discrete_sequence=['#2196f3'],
                    title="Phân phối lợi nhuận"
                )
                fig.update_layout(
                    xaxis_title="Lợi nhuận",
                    yaxis_title="Số lượng giao dịch",
                    template="plotly_white" if self.style != "dark" else "plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.hist(trades_data['profit'], bins=20, alpha=0.7, color='blue')
                ax.axvline(0, color='red', linestyle='--', alpha=0.7)
                ax.set_title("Phân phối lợi nhuận")
                ax.set_xlabel("Lợi nhuận")
                ax.set_ylabel("Số lượng giao dịch")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
        with col2:
            # Vẽ pie chart tỷ lệ thắng/thua
            winning_trades = len(trades_data[trades_data['profit'] > 0])
            losing_trades = len(trades_data[trades_data['profit'] <= 0])
            
            if self.use_plotly:
                fig = px.pie(
                    values=[winning_trades, losing_trades],
                    names=['Thắng', 'Thua'],
                    color_discrete_sequence=['#26a69a', '#ef5350'],
                    title="Tỷ lệ thắng/thua"
                )
                fig.update_traces(textinfo='percent+label')
                fig.update_layout(
                    template="plotly_white" if self.style != "dark" else "plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie(
                    [winning_trades, losing_trades],
                    labels=['Thắng', 'Thua'],
                    colors=['green', 'red'],
                    autopct='%1.1f%%',
                    shadow=True,
                    startangle=90
                )
                ax.set_title("Tỷ lệ thắng/thua")
                ax.axis('equal')
                st.pyplot(fig)
                
        # Vẽ biểu đồ phân phối lợi nhuận theo % nếu có profit_pct
        if 'profit_pct' in trades_data.columns and len(trades_data) > 0:
            if self.use_plotly:
                fig = px.histogram(
                    trades_data, 
                    x='profit_pct', 
                    nbins=20,
                    color_discrete_sequence=['#3f51b5'],
                    title="Phân phối lợi nhuận (%)"
                )
                fig.update_layout(
                    xaxis_title="Lợi nhuận (%)",
                    yaxis_title="Số lượng giao dịch",
                    xaxis_tickformat='.2%',
                    template="plotly_white" if self.style != "dark" else "plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(trades_data['profit_pct'], bins=20, alpha=0.7, color='purple')
                ax.axvline(0, color='red', linestyle='--', alpha=0.7)
                ax.set_title("Phân phối lợi nhuận (%)")
                ax.set_xlabel("Lợi nhuận (%)")
                ax.set_ylabel("Số lượng giao dịch")
                ax.grid(True, alpha=0.3)
                
                # Định dạng trục x dưới dạng phần trăm
                from matplotlib.ticker import FuncFormatter
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
                
                st.pyplot(fig)
    
    def _plot_trade_time_analysis(self, trades_data: pd.DataFrame) -> None:
        """
        Phân tích giao dịch theo khung thời gian.
        
        Args:
            trades_data: DataFrame chứa dữ liệu giao dịch
        """
        if 'entry_time' not in trades_data.columns or len(trades_data) < 2:
            return
            
        st.subheader("Phân tích theo thời gian")
        
        # Chuyển đổi entry_time thành datetime nếu chưa phải
        if not pd.api.types.is_datetime64_any_dtype(trades_data['entry_time']):
            trades_data['entry_time'] = pd.to_datetime(trades_data['entry_time'])
        
        # Thêm các cột thời gian
        trades_data['year'] = trades_data['entry_time'].dt.year
        trades_data['month'] = trades_data['entry_time'].dt.month
        trades_data['day_of_week'] = trades_data['entry_time'].dt.day_name()
        trades_data['hour'] = trades_data['entry_time'].dt.hour
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Phân tích theo ngày trong tuần
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            day_counts = trades_data['day_of_week'].value_counts().reindex(day_order).fillna(0)
            day_profits = trades_data.groupby('day_of_week')['profit'].sum().reindex(day_order).fillna(0)
            
            if self.use_plotly:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Bar(
                        x=day_counts.index,
                        y=day_counts.values,
                        name="Số giao dịch",
                        marker_color='blue'
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=day_profits.index,
                        y=day_profits.values,
                        mode='lines+markers',
                        name="Lợi nhuận",
                        line=dict(color='green', width=2),
                        marker=dict(size=8)
                    ),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title="Phân tích theo ngày trong tuần",
                    template="plotly_white" if self.style != "dark" else "plotly_dark",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                fig.update_yaxes(title_text="Số giao dịch", secondary_y=False)
                fig.update_yaxes(title_text="Lợi nhuận", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax1 = plt.subplots(figsize=(5, 4))
                
                ax1.bar(day_counts.index, day_counts.values, color='blue', alpha=0.7)
                ax1.set_xlabel("Ngày trong tuần")
                ax1.set_ylabel("Số giao dịch", color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                
                ax2 = ax1.twinx()
                ax2.plot(day_profits.index, day_profits.values, 'g-', marker='o')
                ax2.set_ylabel("Lợi nhuận", color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                
                plt.title("Phân tích theo ngày trong tuần")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
        with col2:
            # Phân tích theo giờ trong ngày
            hour_counts = trades_data['hour'].value_counts().sort_index()
            hour_profits = trades_data.groupby('hour')['profit'].sum()
            
            if self.use_plotly:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Bar(
                        x=hour_counts.index,
                        y=hour_counts.values,
                        name="Số giao dịch",
                        marker_color='blue'
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=hour_profits.index,
                        y=hour_profits.values,
                        mode='lines+markers',
                        name="Lợi nhuận",
                        line=dict(color='green', width=2),
                        marker=dict(size=8)
                    ),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title="Phân tích theo giờ trong ngày",
                    template="plotly_white" if self.style != "dark" else "plotly_dark",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                fig.update_yaxes(title_text="Số giao dịch", secondary_y=False)
                fig.update_yaxes(title_text="Lợi nhuận", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax1 = plt.subplots(figsize=(5, 4))
                
                ax1.bar(hour_counts.index, hour_counts.values, color='blue', alpha=0.7)
                ax1.set_xlabel("Giờ trong ngày")
                ax1.set_ylabel("Số giao dịch", color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                
                ax2 = ax1.twinx()
                ax2.plot(hour_profits.index, hour_profits.values, 'g-', marker='o')
                ax2.set_ylabel("Lợi nhuận", color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                
                plt.title("Phân tích theo giờ trong ngày")
                plt.tight_layout()
                st.pyplot(fig)
                
        # Phân tích theo tháng
        if len(trades_data['month'].unique()) > 1:
            month_counts = trades_data['month'].value_counts().sort_index()
            month_profits = trades_data.groupby('month')['profit'].sum()
            
            # Tạo tên tháng
            month_names = {
                1: 'Tháng 1', 2: 'Tháng 2', 3: 'Tháng 3', 4: 'Tháng 4',
                5: 'Tháng 5', 6: 'Tháng 6', 7: 'Tháng 7', 8: 'Tháng 8',
                9: 'Tháng 9', 10: 'Tháng 10', 11: 'Tháng 11', 12: 'Tháng 12'
            }
            
            month_counts.index = [month_names[m] for m in month_counts.index]
            month_profits.index = [month_names[m] for m in month_profits.index]
            
            if self.use_plotly:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Bar(
                        x=month_counts.index,
                        y=month_counts.values,
                        name="Số giao dịch",
                        marker_color='blue'
                    ),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=month_profits.index,
                        y=month_profits.values,
                        mode='lines+markers',
                        name="Lợi nhuận",
                        line=dict(color='green', width=2),
                        marker=dict(size=8)
                    ),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title="Phân tích theo tháng",
                    template="plotly_white" if self.style != "dark" else "plotly_dark",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                fig.update_yaxes(title_text="Số giao dịch", secondary_y=False)
                fig.update_yaxes(title_text="Lợi nhuận", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax1 = plt.subplots(figsize=(10, 4))
                
                ax1.bar(month_counts.index, month_counts.values, color='blue', alpha=0.7)
                ax1.set_xlabel("Tháng")
                ax1.set_ylabel("Số giao dịch", color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                
                ax2 = ax1.twinx()
                ax2.plot(month_profits.index, month_profits.values, 'g-', marker='o')
                ax2.set_ylabel("Lợi nhuận", color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                
                plt.title("Phân tích theo tháng")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
        # Phân tích hiệu suất theo thời gian
        if len(trades_data) >= 10:
            # Tính toán lợi nhuận tích lũy theo thời gian
            trades_sorted = trades_data.sort_values('entry_time')
            trades_sorted['cumulative_profit'] = trades_sorted['profit'].cumsum()
            
            # Tính toán số giao dịch tích lũy
            trades_sorted['trade_count'] = range(1, len(trades_sorted) + 1)
            
            if self.use_plotly:
                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=trades_sorted['entry_time'],
                        y=trades_sorted['cumulative_profit'],
                        mode='lines',
                        name="Lợi nhuận tích lũy",
                        line=dict(color='green', width=2)
                    )
                )
                
                fig.update_layout(
                    title="Lợi nhuận tích lũy theo thời gian",
                    xaxis_title="Thời gian",
                    yaxis_title="Lợi nhuận tích lũy",
                    template="plotly_white" if self.style != "dark" else "plotly_dark",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(10, 4))
                
                ax.plot(trades_sorted['entry_time'], trades_sorted['cumulative_profit'], 'g-', linewidth=2)
                ax.set_xlabel("Thời gian")
                ax.set_ylabel("Lợi nhuận tích lũy")
                ax.grid(True, alpha=0.3)
                
                plt.title("Lợi nhuận tích lũy theo thời gian")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    
    def plot_performance_comparison(
        self,
        strategies_data: Dict[str, pd.DataFrame],
        metrics: List[str] = ["cumulative_returns", "drawdown", "monthly_returns"],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        height: int = 800,
        normalize: bool = True,
        allow_download: bool = False
    ) -> Any:
        """
        Vẽ biểu đồ so sánh hiệu suất giữa các chiến lược.
        
        Args:
            strategies_data: Dict[tên chiến lược, DataFrame dữ liệu]
            metrics: Danh sách các chỉ số cần so sánh
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc
            height: Chiều cao biểu đồ
            normalize: Chuẩn hóa về 1 tại thời điểm bắt đầu
            allow_download: Cho phép tải xuống biểu đồ
            
        Returns:
            Đối tượng figure hoặc None
        """
        # Kiểm tra dữ liệu đầu vào
        if not strategies_data or len(strategies_data) == 0:
            st.error("Không có dữ liệu chiến lược để hiển thị!")
            return None
            
        # Chuẩn bị dữ liệu
        processed_data = {}
        
        for strategy_name, data in strategies_data.items():
            # Xử lý dữ liệu equity
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data = data.set_index('timestamp')
                elif 'datetime' in data.columns:
                    data = data.set_index('datetime')
                elif 'date' in data.columns:
                    data = data.set_index('date')
                else:
                    data = data.copy()
                    data.index = pd.to_datetime(data.index)
                    
            # Lọc theo thời gian
            if start_date is not None:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                data = data[data.index >= start_date]
                
            if end_date is not None:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                data = data[data.index <= end_date]
                
            # Xác định cột equity
            equity_col = None
            for col_name in ['equity', 'balance', 'value', 'close']:
                if col_name in data.columns:
                    equity_col = col_name
                    break
                    
            if equity_col is None:
                # Sử dụng cột đầu tiên
                equity_col = data.columns[0]
                
            # Tính toán các chỉ số
            equity = data[equity_col]
            
            if normalize:
                equity_norm = equity / equity.iloc[0]
            else:
                equity_norm = equity
                
            # Tính drawdown
            peak = equity.cummax()
            drawdown = (equity - peak) / peak
            
            # Tính lợi nhuận hàng ngày
            daily_returns = equity.pct_change().fillna(0)
            
            # Tích lũy các chỉ số
            processed_data[strategy_name] = {
                'equity': equity,
                'equity_norm': equity_norm,
                'drawdown': drawdown,
                'daily_returns': daily_returns
            }
            
        try:
            if self.use_plotly:
                return self._plot_performance_comparison_plotly(
                    processed_data=processed_data,
                    metrics=metrics,
                    height=height,
                    normalize=normalize,
                    allow_download=allow_download
                )
            else:
                return self._plot_performance_comparison_matplotlib(
                    processed_data=processed_data,
                    metrics=metrics,
                    height=height,
                    normalize=normalize,
                    allow_download=allow_download
                )
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ biểu đồ so sánh hiệu suất: {e}")
            st.error(f"Lỗi khi vẽ biểu đồ so sánh hiệu suất: {e}")
            return None
    
    def _plot_performance_comparison_plotly(
        self,
        processed_data: Dict[str, Dict[str, pd.Series]],
        metrics: List[str],
        height: int,
        normalize: bool,
        allow_download: bool
    ) -> go.Figure:
        """
        Vẽ biểu đồ so sánh hiệu suất sử dụng Plotly.
        
        Args: (xem phương thức plot_performance_comparison)
            
        Returns:
            Đối tượng Figure của plotly
        """
        # Xác định số lượng subplots
        n_plots = 0
        if "cumulative_returns" in metrics:
            n_plots += 1
        if "drawdown" in metrics:
            n_plots += 1
        if "monthly_returns" in metrics:
            n_plots += 1
        if "daily_returns" in metrics:
            n_plots += 1
            
        if n_plots == 0:
            st.error("Không có chỉ số nào để hiển thị!")
            return None
            
        # Tạo tiêu đề cho mỗi subplot
        subplot_titles = []
        if "cumulative_returns" in metrics:
            subplot_titles.append("Lợi nhuận tích lũy")
        if "drawdown" in metrics:
            subplot_titles.append("Drawdown")
        if "monthly_returns" in metrics:
            subplot_titles.append("Lợi nhuận theo tháng")
        if "daily_returns" in metrics:
            subplot_titles.append("Lợi nhuận hàng ngày")
            
        # Tạo subplots
        fig = make_subplots(
            rows=n_plots, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles
        )
        
        # Các màu cho các chiến lược
        colors = px.colors.qualitative.Plotly
        
        # Vẽ các chỉ số
        current_plot = 1
        
        # Vẽ lợi nhuận tích lũy
        if "cumulative_returns" in metrics:
            for i, (strategy_name, data) in enumerate(processed_data.items()):
                color = colors[i % len(colors)]
                
                fig.add_trace(
                    go.Scatter(
                        x=data["equity_norm"].index,
                        y=data["equity_norm"],
                        mode='lines',
                        name=strategy_name,
                        line=dict(color=color, width=2)
                    ),
                    row=current_plot, col=1
                )
                
            current_plot += 1
            
        # Vẽ drawdown
        if "drawdown" in metrics:
            for i, (strategy_name, data) in enumerate(processed_data.items()):
                color = colors[i % len(colors)]
                
                fig.add_trace(
                    go.Scatter(
                        x=data["drawdown"].index,
                        y=data["drawdown"] * 100,  # Chuyển sang phần trăm
                        mode='lines',
                        name=strategy_name,
                        line=dict(color=color, width=2)
                    ),
                    row=current_plot, col=1
                )
                
            current_plot += 1
            
        # Vẽ lợi nhuận theo tháng
        if "monthly_returns" in metrics:
            # Tính lợi nhuận theo tháng cho mỗi chiến lược
            monthly_returns = {}
            
            for strategy_name, data in processed_data.items():
                # Resample dữ liệu theo tháng
                monthly = data["equity"].resample('M').last()
                monthly_return = monthly.pct_change().fillna(0)
                monthly_returns[strategy_name] = monthly_return
                
            # Tìm khoảng thời gian chung
            all_months = set()
            for returns in monthly_returns.values():
                all_months.update(returns.index)
                
            all_months = sorted(all_months)
            
            # Tạo DataFrame chung
            monthly_df = pd.DataFrame(index=all_months)
            
            for strategy_name, returns in monthly_returns.items():
                monthly_df[strategy_name] = returns
                
            # Fill NA
            monthly_df = monthly_df.fillna(0)
            
            # Vẽ biểu đồ cột
            for i, strategy_name in enumerate(processed_data.keys()):
                color = colors[i % len(colors)]
                
                fig.add_trace(
                    go.Bar(
                        x=monthly_df.index,
                        y=monthly_df[strategy_name] * 100,  # Chuyển sang phần trăm
                        name=strategy_name,
                        marker_color=color,
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=current_plot, col=1
                )
                
            current_plot += 1
            
        # Vẽ lợi nhuận hàng ngày
        if "daily_returns" in metrics:
            for i, (strategy_name, data) in enumerate(processed_data.items()):
                color = colors[i % len(colors)]
                
                fig.add_trace(
                    go.Scatter(
                        x=data["daily_returns"].index,
                        y=data["daily_returns"] * 100,  # Chuyển sang phần trăm
                        mode='lines',
                        name=strategy_name,
                        line=dict(color=color, width=1),
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=current_plot, col=1
                )
                
        # Cập nhật layout
        template = "plotly_white" if self.style != "dark" else "plotly_dark"
        fig.update_layout(
            title="So sánh hiệu suất chiến lược",
            height=height,
            template=template,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=60, b=10)
        )
        
        # Thiết lập các trục
        current_plot = 1
        if "cumulative_returns" in metrics:
            fig.update_yaxes(title_text="Giá trị" + (" (chuẩn hóa)" if normalize else ""), row=current_plot, col=1)
            current_plot += 1
            
        if "drawdown" in metrics:
            fig.update_yaxes(title_text="Drawdown (%)", row=current_plot, col=1)
            current_plot += 1
            
        if "monthly_returns" in metrics:
            fig.update_yaxes(title_text="Lợi nhuận (%)", row=current_plot, col=1)
            current_plot += 1
            
        if "daily_returns" in metrics:
            fig.update_yaxes(title_text="Lợi nhuận (%)", row=current_plot, col=1)
            
        # Thiết lập grid
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)' if self.style != "dark" else 'rgba(255,255,255,0.1)'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)' if self.style != "dark" else 'rgba(255,255,255,0.1)'
        )
        
        # Hiển thị bảng so sánh
        self._display_strategy_comparison_table(processed_data)
        
        # Hiển thị biểu đồ trên Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Thêm chức năng tải xuống nếu cần
        if allow_download:
            self._add_download_button(fig, "performance_comparison.html", "Tải xuống biểu đồ")
            
        return fig
    
    def _plot_performance_comparison_matplotlib(
        self,
        processed_data: Dict[str, Dict[str, pd.Series]],
        metrics: List[str],
        height: int,
        normalize: bool,
        allow_download: bool
    ) -> plt.Figure:
        """
        Vẽ biểu đồ so sánh hiệu suất sử dụng Matplotlib.
        
        Args: (xem phương thức plot_performance_comparison)
            
        Returns:
            Đối tượng Figure của matplotlib
        """
        # Xác định số lượng subplots
        n_plots = 0
        if "cumulative_returns" in metrics:
            n_plots += 1
        if "drawdown" in metrics:
            n_plots += 1
        if "monthly_returns" in metrics:
            n_plots += 1
        if "daily_returns" in metrics:
            n_plots += 1
            
        if n_plots == 0:
            st.error("Không có chỉ số nào để hiển thị!")
            return None
            
        # Tính toán kích thước trong inch (1 px = 1/96 inch)
        figsize = (10, height / 96)
        
        # Tạo figure và axes
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True, gridspec_kw={'hspace': 0.3})
        
        # Đảm bảo axes là list kể cả khi chỉ có 1 metric
        if n_plots == 1:
            axes = [axes]
            
        # Các màu cho các chiến lược
        colors = plt.cm.tab10.colors
        
        # Vẽ các chỉ số
        current_plot = 0
        
        # Vẽ lợi nhuận tích lũy
        if "cumulative_returns" in metrics:
            ax = axes[current_plot]
            
            for i, (strategy_name, data) in enumerate(processed_data.items()):
                color = colors[i % len(colors)]
                
                ax.plot(
                    data["equity_norm"].index,
                    data["equity_norm"],
                    label=strategy_name,
                    color=color,
                    linewidth=2
                )
                
            ax.set_title("Lợi nhuận tích lũy")
            ax.set_ylabel("Giá trị" + (" (chuẩn hóa)" if normalize else ""))
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            current_plot += 1
            
        # Vẽ drawdown
        if "drawdown" in metrics:
            ax = axes[current_plot]
            
            for i, (strategy_name, data) in enumerate(processed_data.items()):
                color = colors[i % len(colors)]
                
                ax.plot(
                    data["drawdown"].index,
                    data["drawdown"] * 100,  # Chuyển sang phần trăm
                    label=strategy_name,
                    color=color,
                    linewidth=2
                )
                
            ax.set_title("Drawdown")
            ax.set_ylabel("Drawdown (%)")
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            current_plot += 1
            
        # Vẽ lợi nhuận theo tháng
        if "monthly_returns" in metrics:
            ax = axes[current_plot]
            
            # Tính lợi nhuận theo tháng cho mỗi chiến lược
            monthly_returns = {}
            
            for strategy_name, data in processed_data.items():
                # Resample dữ liệu theo tháng
                monthly = data["equity"].resample('M').last()
                monthly_return = monthly.pct_change().fillna(0)
                monthly_returns[strategy_name] = monthly_return
                
            # Tìm khoảng thời gian chung
            all_months = set()
            for returns in monthly_returns.values():
                all_months.update(returns.index)
                
            all_months = sorted(all_months)
            
            # Tạo DataFrame chung
            monthly_df = pd.DataFrame(index=all_months)
            
            for strategy_name, returns in monthly_returns.items():
                monthly_df[strategy_name] = returns
                
            # Fill NA
            monthly_df = monthly_df.fillna(0)
            
            # Vẽ biểu đồ cột
            bar_width = 0.8 / len(processed_data)
            offset = -0.4 + bar_width / 2
            
            for i, strategy_name in enumerate(processed_data.keys()):
                color = colors[i % len(colors)]
                position = np.arange(len(monthly_df.index)) + offset + i * bar_width
                
                ax.bar(
                    position,
                    monthly_df[strategy_name] * 100,  # Chuyển sang phần trăm
                    width=bar_width,
                    label=strategy_name,
                    color=color,
                    alpha=0.7
                )
                
            ax.set_title("Lợi nhuận theo tháng")
            ax.set_ylabel("Lợi nhuận (%)")
            ax.set_xticks(np.arange(len(monthly_df.index)))
            ax.set_xticklabels([d.strftime('%Y-%m') for d in monthly_df.index], rotation=90)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            current_plot += 1
            
        # Vẽ lợi nhuận hàng ngày
        if "daily_returns" in metrics:
            ax = axes[current_plot]
            
            for i, (strategy_name, data) in enumerate(processed_data.items()):
                color = colors[i % len(colors)]
                
                ax.plot(
                    data["daily_returns"].index,
                    data["daily_returns"] * 100,  # Chuyển sang phần trăm
                    label=strategy_name,
                    color=color,
                    linewidth=1,
                    alpha=0.7
                )
                
            ax.set_title("Lợi nhuận hàng ngày")
            ax.set_ylabel("Lợi nhuận (%)")
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
        # Định dạng trục x
        for ax in axes:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
        # Thiết lập tiêu đề
        fig.suptitle("So sánh hiệu suất chiến lược", fontsize=16)
        
        # Điều chỉnh layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Hiển thị bảng so sánh
        self._display_strategy_comparison_table(processed_data)
        
        # Hiển thị biểu đồ trên Streamlit
        st.pyplot(fig)
        
        # Thêm chức năng tải xuống nếu cần
        if allow_download:
            self._add_download_button_matplotlib(fig, "performance_comparison.png", "Tải xuống biểu đồ")
            
        return fig
    
    def _display_strategy_comparison_table(self, processed_data: Dict[str, Dict[str, pd.Series]]) -> None:
        """
        Hiển thị bảng so sánh các chiến lược.
        
        Args:
            processed_data: Dict[tên chiến lược, Dict[tên chỉ số, dữ liệu]]
        """
        # Tính toán các chỉ số cho mỗi chiến lược
        comparison_data = []
        
        for strategy_name, data in processed_data.items():
            # Tính các chỉ số
            stats = self._calculate_performance_stats(data['equity'])
            
            # Thêm vào danh sách
            comparison_data.append({
                'Chiến lược': strategy_name,
                'Lợi nhuận': stats['total_return'],
                'Sharpe Ratio': stats['sharpe_ratio'],
                'Max Drawdown': stats['max_drawdown'],
                'Volatility': stats['volatility'],
                'Lợi nhuận năm': stats['annual_return'],
                'Profit Factor': stats.get('profit_factor', 0),
                'Số giao dịch': stats.get('total_trades', 0)
            })
            
        # Tạo DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Định dạng các cột
        df['Lợi nhuận'] = df['Lợi nhuận'].map(lambda x: f"{x:.2%}")
        df['Lợi nhuận năm'] = df['Lợi nhuận năm'].map(lambda x: f"{x:.2%}")
        df['Max Drawdown'] = df['Max Drawdown'].map(lambda x: f"{x:.2%}")
        df['Volatility'] = df['Volatility'].map(lambda x: f"{x:.2%}")
        df['Sharpe Ratio'] = df['Sharpe Ratio'].map(lambda x: f"{x:.2f}")
        df['Profit Factor'] = df['Profit Factor'].map(lambda x: f"{x:.2f}")
        
        # Hiển thị tiêu đề
        st.subheader("So sánh hiệu suất")
        
        # Hiển thị bảng
        st.dataframe(df, use_container_width=True)
        
        # Vẽ biểu đồ cột so sánh các chỉ số chính
        st.subheader("So sánh các chỉ số")
        
        # Chuẩn bị dữ liệu cho biểu đồ cột
        metrics_to_compare = ['Lợi nhuận', 'Sharpe Ratio', 'Max Drawdown', 'Volatility']
        strategies = df['Chiến lược'].tolist()
        
        if self.use_plotly:
            # Tạo biểu đồ cột Plotly cho từng metric
            for metric in metrics_to_compare:
                fig = go.Figure()
                
                # Chuyển đổi các giá trị từ chuỗi về số
                values = []
                for val in df[metric]:
                    if isinstance(val, str) and '%' in val:
                        values.append(float(val.strip('%')) / 100)
                    else:
                        try:
                            values.append(float(val))
                        except:
                            values.append(0)
                
                # Xác định màu sắc
                if metric in ['Lợi nhuận', 'Sharpe Ratio']:
                    colors = ['green' if v > 0 else 'red' for v in values]
                elif metric in ['Max Drawdown', 'Volatility']:
                    colors = ['red' if v < 0 else 'green' for v in values]
                else:
                    colors = px.colors.qualitative.Plotly
                
                # Thêm cột
                fig.add_trace(
                    go.Bar(
                        x=strategies,
                        y=values,
                        marker_color=colors,
                        text=[f"{v:.2%}" if metric in ['Lợi nhuận', 'Max Drawdown', 'Volatility'] else f"{v:.2f}" for v in values],
                        textposition='auto'
                    )
                )
                
                fig.update_layout(
                    title=f"So sánh {metric}",
                    xaxis_title="Chiến lược",
                    yaxis_title=metric,
                    template="plotly_white" if self.style != "dark" else "plotly_dark"
                )
                
                if metric in ['Lợi nhuận', 'Max Drawdown', 'Volatility']:
                    fig.update_yaxes(tickformat='.2%')
                    
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Sử dụng Matplotlib
            for metric in metrics_to_compare:
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Chuyển đổi các giá trị từ chuỗi về số
                values = []
                for val in df[metric]:
                    if isinstance(val, str) and '%' in val:
                        values.append(float(val.strip('%')) / 100)
                    else:
                        try:
                            values.append(float(val))
                        except:
                            values.append(0)
                
                # Xác định màu sắc
                if metric in ['Lợi nhuận', 'Sharpe Ratio']:
                    colors = ['green' if v > 0 else 'red' for v in values]
                elif metric in ['Max Drawdown', 'Volatility']:
                    colors = ['red' if v < 0 else 'green' for v in values]
                else:
                    colors = plt.cm.tab10.colors
                
                # Vẽ cột
                bars = ax.bar(strategies, values, color=colors)
                
                # Thêm giá trị trên mỗi cột
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{height:.2%}" if metric in ['Lợi nhuận', 'Max Drawdown', 'Volatility'] else f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom'
                    )
                    
                ax.set_title(f"So sánh {metric}")
                ax.set_xlabel("Chiến lược")
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
                
                if metric in ['Lợi nhuận', 'Max Drawdown', 'Volatility']:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2%}'.format(x)))
                    
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    
    def plot_drawdown_chart(
        self,
        equity_data: pd.DataFrame,
        title: str = "Phân tích Drawdown",
        top_n: int = 5,
        height: int = 500,
        allow_download: bool = False
    ) -> Any:
        """
        Vẽ biểu đồ phân tích drawdown.
        
        Args:
            equity_data: DataFrame chứa dữ liệu equity
            title: Tiêu đề biểu đồ
            top_n: Số lượng drawdown lớn nhất cần hiển thị
            height: Chiều cao biểu đồ
            allow_download: Cho phép tải xuống biểu đồ
            
        Returns:
            Đối tượng figure hoặc None
        """
        # Kiểm tra dữ liệu đầu vào
        if equity_data is None or len(equity_data) == 0:
            st.error("Không có dữ liệu equity để hiển thị!")
            return None
            
        # Chuyển đổi index thành datetime nếu chưa phải
        if not isinstance(equity_data.index, pd.DatetimeIndex):
            # Kiểm tra xem có cột timestamp/datetime không
            if 'timestamp' in equity_data.columns:
                equity_data = equity_data.set_index('timestamp')
            elif 'datetime' in equity_data.columns:
                equity_data = equity_data.set_index('datetime')
            elif 'date' in equity_data.columns:
                equity_data = equity_data.set_index('date')
            else:
                # Nếu không có cột thời gian, sử dụng chỉ số làm index
                equity_data = equity_data.copy()
                equity_data.index = pd.to_datetime(equity_data.index)
                
        # Chuẩn hóa tên cột (viết thường)
        equity_data.columns = [col.lower() for col in equity_data.columns]
            
        # Xác định cột equity
        equity_col = None
        for col_name in ['equity', 'balance', 'value', 'close']:
            if col_name in equity_data.columns:
                equity_col = col_name
                break
                
        if equity_col is None:
            # Sử dụng cột đầu tiên
            equity_col = equity_data.columns[0]
            
        equity_series = equity_data[equity_col]
        
        # Tính drawdown
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        
        # Xác định các giai đoạn drawdown lớn nhất
        drawdown_periods = []
        in_drawdown = False
        start_idx = None
        
        for i, (date, value) in enumerate(drawdown.items()):
            if not in_drawdown and value < 0:
                # Bắt đầu giai đoạn drawdown mới
                in_drawdown = True
                start_idx = i
            elif in_drawdown and value == 0:
                # Kết thúc giai đoạn drawdown
                end_idx = i
                in_drawdown = False
                
                # Tính drawdown tối đa trong giai đoạn
                dd_period = drawdown.iloc[start_idx:end_idx]
                max_dd = dd_period.min()
                max_dd_idx = dd_period.idxmin()
                
                # Thêm vào danh sách
                drawdown_periods.append({
                    'start_date': drawdown.index[start_idx],
                    'worst_date': max_dd_idx,
                    'end_date': drawdown.index[end_idx],
                    'max_drawdown': max_dd,
                    'duration': (drawdown.index[end_idx] - drawdown.index[start_idx]).days,
                    'recovered': True
                })
        
        # Nếu vẫn đang trong drawdown tại thời điểm cuối cùng
        if in_drawdown:
            end_idx = len(drawdown) - 1
            dd_period = drawdown.iloc[start_idx:end_idx+1]
            max_dd = dd_period.min()
            max_dd_idx = dd_period.idxmin()
            
            drawdown_periods.append({
                'start_date': drawdown.index[start_idx],
                'worst_date': max_dd_idx,
                'end_date': drawdown.index[end_idx],
                'max_drawdown': max_dd,
                'duration': (drawdown.index[end_idx] - drawdown.index[start_idx]).days,
                'recovered': False
            })
        
        # Sắp xếp theo mức độ drawdown (từ lớn đến nhỏ)
        drawdown_periods.sort(key=lambda x: x['max_drawdown'])
        top_periods = drawdown_periods[:min(top_n, len(drawdown_periods))]
        
        try:
            if self.use_plotly:
                return self._plot_drawdown_chart_plotly(
                    equity_series=equity_series,
                    drawdown=drawdown,
                    top_periods=top_periods,
                    title=title,
                    height=height,
                    allow_download=allow_download
                )
            else:
                return self._plot_drawdown_chart_matplotlib(
                    equity_series=equity_series,
                    drawdown=drawdown,
                    top_periods=top_periods,
                    title=title,
                    height=height,
                    allow_download=allow_download
                )
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ biểu đồ phân tích drawdown: {e}")
            st.error(f"Lỗi khi vẽ biểu đồ phân tích drawdown: {e}")
            return None
    
    def _plot_drawdown_chart_plotly(
        self,
        equity_series: pd.Series,
        drawdown: pd.Series,
        top_periods: List[Dict[str, Any]],
        title: str,
        height: int,
        allow_download: bool
    ) -> go.Figure:
        """
        Vẽ biểu đồ phân tích drawdown sử dụng Plotly.
        
        Args: (xem phương thức plot_drawdown_chart)
            
        Returns:
            Đối tượng Figure của plotly
        """
        # Tạo biểu đồ với 2 phần
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=(title, "Chi tiết Drawdown")
        )
        
        # Vẽ đường cong vốn
        fig.add_trace(
            go.Scatter(
                x=equity_series.index,
                y=equity_series,
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Vẽ drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)'
            ),
            row=2, col=1
        )
        
        # Các màu cho các giai đoạn drawdown
        colors = px.colors.qualitative.Plotly
        
        # Vẽ các giai đoạn drawdown lớn nhất
        for i, period in enumerate(top_periods):
            color = colors[i % len(colors)]
            
            # Thêm vùng đánh dấu
            fig.add_vrect(
                x0=period['start_date'],
                x1=period['end_date'],
                fillcolor=color,
                opacity=0.15,
                line_width=0,
                annotation_text=f"DD #{i+1}",
                annotation_position="top left",
                annotation=dict(font_size=10),
                row=1, col=1
            )
            
            fig.add_vrect(
                x0=period['start_date'],
                x1=period['end_date'],
                fillcolor=color,
                opacity=0.15,
                line_width=0,
                row=2, col=1
            )
            
            # Đánh dấu đỉnh, đáy, và điểm hồi phục trên equity
            fig.add_trace(
                go.Scatter(
                    x=[period['start_date']],
                    y=[equity_series[period['start_date']]],
                    mode='markers',
                    marker=dict(color=color, size=12, symbol='triangle-up'),
                    name=f"DD#{i+1} Start",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[period['worst_date']],
                    y=[equity_series[period['worst_date']]],
                    mode='markers',
                    marker=dict(color=color, size=12, symbol='triangle-down'),
                    name=f"DD#{i+1} Bottom",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[period['end_date']],
                    y=[equity_series[period['end_date']]],
                    mode='markers',
                    marker=dict(color=color, size=12, symbol='circle'),
                    name=f"DD#{i+1} Recovery",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Đánh dấu điểm drawdown tối đa
            fig.add_trace(
                go.Scatter(
                    x=[period['worst_date']],
                    y=[drawdown[period['worst_date']] * 100],
                    mode='markers+text',
                    marker=dict(color=color, size=12, symbol='triangle-down'),
                    text=f"{drawdown[period['worst_date']]:.2%}",
                    textposition="bottom center",
                    name=f"DD#{i+1}: {period['max_drawdown']:.2%} ({period['duration']} ngày)",
                ),
                row=2, col=1
            )
            
        # Hiển thị thông tin về các drawdown lớn nhất
        dd_info = pd.DataFrame(top_periods)
        dd_info['max_drawdown'] = dd_info['max_drawdown'].map(lambda x: f"{x:.2%}")
        dd_info['start_date'] = dd_info['start_date'].dt.strftime('%Y-%m-%d')
        dd_info['worst_date'] = dd_info['worst_date'].dt.strftime('%Y-%m-%d')
        dd_info['end_date'] = dd_info['end_date'].dt.strftime('%Y-%m-%d')
        
        dd_info = dd_info.rename(columns={
            'max_drawdown': 'Drawdown',
            'start_date': 'Bắt đầu',
            'worst_date': 'Đáy',
            'end_date': 'Kết thúc',
            'duration': 'Thời gian (ngày)',
            'recovered': 'Đã phục hồi'
        })
        
        # Cập nhật layout
        template = "plotly_white" if self.style != "dark" else "plotly_dark"
        fig.update_layout(
            height=height,
            template=template,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=60, b=10)
        )
        
        # Thiết lập các trục
        fig.update_yaxes(title_text="Giá trị vốn", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Thời gian", row=2, col=1)
        
        # Thiết lập grid
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)' if self.style != "dark" else 'rgba(255,255,255,0.1)'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)' if self.style != "dark" else 'rgba(255,255,255,0.1)'
        )
        
        # Hiển thị bảng thông tin drawdown
        st.subheader(f"Top {len(dd_info)} Drawdown lớn nhất")
        st.dataframe(dd_info, use_container_width=True)
        
        # Hiển thị biểu đồ trên Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Thêm chức năng tải xuống nếu cần
        if allow_download:
            self._add_download_button(fig, "drawdown_analysis.html", "Tải xuống biểu đồ")
            
        return fig
    
    def _plot_drawdown_chart_matplotlib(
        self,
        equity_series: pd.Series,
        drawdown: pd.Series,
        top_periods: List[Dict[str, Any]],
        title: str,
        height: int,
        allow_download: bool
    ) -> plt.Figure:
        """
        Vẽ biểu đồ phân tích drawdown sử dụng Matplotlib.
        
        Args: (xem phương thức plot_drawdown_chart)
            
        Returns:
            Đối tượng Figure của matplotlib
        """
        # Tính toán kích thước trong inch (1 px = 1/96 inch)
        figsize = (10, height / 96)
        
        # Tạo biểu đồ với 2 phần
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1.5, 1]})
        
        # Vẽ đường cong vốn và đánh dấu các giai đoạn drawdown lớn
        ax1.plot(equity_series.index, equity_series, label='Equity', color='blue', linewidth=2)
        
        # Các màu cho các giai đoạn drawdown
        colors = plt.cm.tab10.colors
        
        # Vẽ các giai đoạn drawdown lớn nhất
        patches = []
        for i, period in enumerate(top_periods):
            color = colors[i % len(colors)]
            
            # Vùng drawdown
            ax1.axvspan(period['start_date'], period['end_date'], alpha=0.2, color=color)
            
            # Đánh dấu đỉnh, đáy và điểm hồi phục
            ax1.scatter(period['start_date'], equity_series[period['start_date']], color=color, s=100, marker='^')
            ax1.scatter(period['worst_date'], equity_series[period['worst_date']], color=color, s=100, marker='v')
            ax1.scatter(period['end_date'], equity_series[period['end_date']], color=color, s=100, marker='o')
            
            # Tạo patch cho legend
            patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.5)
            patches.append(patch)
        
        # Vẽ drawdown
        ax2.fill_between(drawdown.index, 0, drawdown * 100, color='red', alpha=0.5)
        ax2.plot(drawdown.index, drawdown * 100, color='red', linewidth=1)
        
        # Đánh dấu các drawdown lớn nhất
        for i, period in enumerate(top_periods):
            color = colors[i % len(colors)]
            ax2.axvspan(period['start_date'], period['end_date'], alpha=0.2, color=color)
            
            # Đánh dấu điểm drawdown tối đa
            ax2.scatter(period['worst_date'], drawdown[period['worst_date']] * 100, color=color, s=100, marker='v')
            
            # Ghi chú mức drawdown tối đa
            ax2.annotate(
                f"{drawdown[period['worst_date']]:.2%}",
                xy=(period['worst_date'], drawdown[period['worst_date']] * 100),
                xytext=(10, -20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=color)
            )
        
        # Thiết lập ax1 (equity curve)
        ax1.set_title(title, fontsize=14)
        ax1.set_ylabel('Giá trị vốn')
        ax1.grid(True, alpha=0.3)
        
        # Tạo legend cho các giai đoạn drawdown
        legend_labels = [f"DD #{i+1}: {period['max_drawdown']:.2%} ({period['duration']} ngày)" 
                        for i, period in enumerate(top_periods)]
        ax1.legend([*patches], legend_labels, loc='best')
        
        # Thiết lập ax2 (drawdown)
        ax2.set_title("Chi tiết Drawdown", fontsize=12)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Thời gian')
        ax2.grid(True, alpha=0.3)
        
        # Định dạng trục x
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.tight_layout()
        
        # Hiển thị bảng thông tin drawdown
        dd_info = pd.DataFrame(top_periods)
        dd_info['max_drawdown'] = dd_info['max_drawdown'].map(lambda x: f"{x:.2%}")
        dd_info['start_date'] = dd_info['start_date'].dt.strftime('%Y-%m-%d')
        dd_info['worst_date'] = dd_info['worst_date'].dt.strftime('%Y-%m-%d')
        dd_info['end_date'] = dd_info['end_date'].dt.strftime('%Y-%m-%d')
        
        dd_info = dd_info.rename(columns={
            'max_drawdown': 'Drawdown',
            'start_date': 'Bắt đầu',
            'worst_date': 'Đáy',
            'end_date': 'Kết thúc',
            'duration': 'Thời gian (ngày)',
            'recovered': 'Đã phục hồi'
        })
        
        st.subheader(f"Top {len(dd_info)} Drawdown lớn nhất")
        st.dataframe(dd_info, use_container_width=True)
        
        # Hiển thị biểu đồ trên Streamlit
        st.pyplot(fig)
        
        # Thêm chức năng tải xuống nếu cần
        if allow_download:
            self._add_download_button_matplotlib(fig, "drawdown_analysis.png", "Tải xuống biểu đồ")
            
        return fig
    
    def plot_rolling_stats(
        self,
        equity_data: pd.DataFrame,
        title: str = "Các chỉ số hiệu suất theo thời gian",
        window: int = 30,
        metrics: List[str] = ["returns", "volatility", "sharpe", "win_rate"],
        height: int = 600,
        allow_download: bool = False
    ) -> Any:
        """
        Vẽ biểu đồ các chỉ số hiệu suất được tính theo cửa sổ trượt.
        
        Args:
            equity_data: DataFrame chứa dữ liệu equity và giao dịch
            title: Tiêu đề biểu đồ
            window: Kích thước cửa sổ trượt (số ngày)
            metrics: Danh sách các chỉ số cần hiển thị
            height: Chiều cao biểu đồ
            allow_download: Cho phép tải xuống biểu đồ
            
        Returns:
            Đối tượng figure hoặc None
        """
        # Kiểm tra dữ liệu đầu vào
        if equity_data is None or len(equity_data) == 0:
            st.error("Không có dữ liệu để hiển thị!")
            return None
            
        # Chuyển đổi index thành datetime nếu chưa phải
        if not isinstance(equity_data.index, pd.DatetimeIndex):
            # Kiểm tra xem có cột timestamp/datetime không
            if 'timestamp' in equity_data.columns:
                equity_data = equity_data.set_index('timestamp')
            elif 'datetime' in equity_data.columns:
                equity_data = equity_data.set_index('datetime')
            elif 'date' in equity_data.columns:
                equity_data = equity_data.set_index('date')
            else:
                # Nếu không có cột thời gian, sử dụng chỉ số làm index
                equity_data = equity_data.copy()
                equity_data.index = pd.to_datetime(equity_data.index)
                
        # Chuẩn hóa tên cột (viết thường)
        equity_data.columns = [col.lower() for col in equity_data.columns]
            
        # Xác định cột equity
        equity_col = None
        for col_name in ['equity', 'balance', 'value', 'close']:
            if col_name in equity_data.columns:
                equity_col = col_name
                break
                
        if equity_col is None:
            # Sử dụng cột đầu tiên
            equity_col = equity_data.columns[0]
            
        # Lấy series dữ liệu equity
        equity_series = equity_data[equity_col]
        
        # Tính lợi nhuận hàng ngày
        daily_returns = equity_series.pct_change().fillna(0)
        
        # Tạo DataFrame cho các chỉ số
        df_metrics = pd.DataFrame(index=daily_returns.index)
        
        # Tính các chỉ số theo cửa sổ trượt
        # 1. Lợi nhuận tích lũy
        if 'returns' in metrics:
            df_metrics['returns'] = daily_returns.rolling(window=window).apply(
                lambda x: (1 + x).prod() - 1, raw=True
            )
        
        # 2. Độ biến động (volatility)
        if 'volatility' in metrics:
            df_metrics['volatility'] = daily_returns.rolling(window=window).std() * np.sqrt(252)
        
        # 3. Sharpe ratio
        if 'sharpe' in metrics:
            risk_free_rate = 0.02 / 252  # Lãi suất phi rủi ro hàng ngày
            df_metrics['sharpe'] = (daily_returns.rolling(window=window).mean() - risk_free_rate) / \
                                daily_returns.rolling(window=window).std() * np.sqrt(252)
        
        # 4. Maximum drawdown
        if 'drawdown' in metrics:
            # Tính rolling max và drawdown tại mỗi thời điểm
            rolling_df = pd.DataFrame(index=equity_series.index)
            rolling_df['equity'] = equity_series
            
            # Tính drawdown cho mỗi window
            result = []
            for i in range(len(rolling_df) - window + 1):
                window_df = rolling_df.iloc[i:i+window]
                peak = window_df['equity'].max()
                current = window_df['equity'].iloc[-1]
                dd = (current - peak) / peak
                result.append(dd)
                
            # Điền NA cho các vị trí đầu
            na_values = [np.nan] * (len(rolling_df) - len(result))
            all_values = na_values + result
            df_metrics['drawdown'] = all_values
        
        # 5. Tỷ lệ thắng
        if 'win_rate' in metrics and 'trades' in equity_data.columns:
            trades = equity_data['trades']
            
            # Tính tỷ lệ thắng trong một cửa sổ thời gian
            wins = equity_data.get('wins', None)
            
            if wins is not None:
                # Nếu có sẵn cột wins
                win_rate = wins.rolling(window=window).sum() / trades.rolling(window=window).sum()
                win_rate = win_rate.fillna(0)
                df_metrics['win_rate'] = win_rate
        
        # Loại bỏ các chỉ số không tồn tại
        metrics = [m for m in metrics if m in df_metrics.columns]
        
        if not metrics:
            st.error("Không có chỉ số nào để hiển thị!")
            return None
            
        try:
            if self.use_plotly:
                return self._plot_rolling_stats_plotly(
                    df_metrics=df_metrics,
                    metrics=metrics,
                    window=window,
                    title=title,
                    height=height,
                    allow_download=allow_download
                )
            else:
                return self._plot_rolling_stats_matplotlib(
                    df_metrics=df_metrics,
                    metrics=metrics,
                    window=window,
                    title=title,
                    height=height,
                    allow_download=allow_download
                )
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ biểu đồ các chỉ số hiệu suất: {e}")
            st.error(f"Lỗi khi vẽ biểu đồ các chỉ số hiệu suất: {e}")
            return None
    
    def _plot_rolling_stats_plotly(
        self,
        df_metrics: pd.DataFrame,
        metrics: List[str],
        window: int,
        title: str,
        height: int,
        allow_download: bool
    ) -> go.Figure:
        """
        Vẽ biểu đồ các chỉ số hiệu suất sử dụng Plotly.
        
        Args: (xem phương thức plot_rolling_stats)
            
        Returns:
            Đối tượng Figure của plotly
        """
        # Xác định số lượng subplots
        n_plots = len(metrics)
        
        # Tạo tiêu đề cho mỗi subplot
        subplot_titles = []
        for metric in metrics:
            if metric == 'returns':
                subplot_titles.append(f"Lợi nhuận ({window} ngày)")
            elif metric == 'volatility':
                subplot_titles.append(f"Độ biến động ({window} ngày)")
            elif metric == 'sharpe':
                subplot_titles.append(f"Sharpe Ratio ({window} ngày)")
            elif metric == 'drawdown':
                subplot_titles.append(f"Max Drawdown ({window} ngày)")
            elif metric == 'win_rate':
                subplot_titles.append(f"Tỷ lệ thắng ({window} ngày)")
                
        # Tạo subplots
        fig = make_subplots(
            rows=n_plots, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles
        )
        
        # Vẽ các chỉ số
        for i, metric in enumerate(metrics):
            row = i + 1
            
            # Định dạng giá trị
            if metric in ['returns', 'volatility', 'drawdown', 'win_rate']:
                y_values = df_metrics[metric] * 100  # Chuyển sang phần trăm
                y_title = "%"
            else:
                y_values = df_metrics[metric]
                y_title = ""
                
            # Đường chỉ số
            fig.add_trace(
                go.Scatter(
                    x=df_metrics.index,
                    y=y_values,
                    mode='lines',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=f'rgb({50+i*30}, {100+i*30}, {200-i*20})', width=2)
                ),
                row=row, col=1
            )
            
            # Thêm đường ngang tại 0 nếu cần
            if metric in ['returns', 'sharpe', 'drawdown']:
                fig.add_shape(
                    type="line",
                    x0=df_metrics.index[0], 
                    x1=df_metrics.index[-1],
                    y0=0, 
                    y1=0,
                    line=dict(color="black", width=1, dash="dash"),
                    row=row, col=1
                )
            
            # Đường giá trị trung bình
            avg_value = df_metrics[metric].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=df_metrics.index,
                    y=[avg_value * 100 if metric in ['returns', 'volatility', 'drawdown', 'win_rate'] else avg_value] * len(df_metrics),
                    mode='lines',
                    name=f'Avg {metric}',
                    line=dict(color='blue', width=1, dash='dot'),
                    showlegend=False
                ),
                row=row, col=1
            )
            
            # Thêm annotation cho giá trị trung bình
            if metric in ['returns', 'volatility', 'drawdown', 'win_rate']:
                avg_text = f'Trung bình: {avg_value:.2%}'
            else:
                avg_text = f'Trung bình: {avg_value:.2f}'
            
            fig.add_annotation(
                x=0.02,
                y=0.95,
                xref=f"x{row}" if row > 1 else "x",
                yref=f"y{row}" if row > 1 else "y",
                text=avg_text,
                showarrow=False,
                xanchor='left',
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                font=dict(size=10)
            )
            
        # Cập nhật layout
        template = "plotly_white" if self.style != "dark" else "plotly_dark"
        fig.update_layout(
            title=title,
            height=height,
            template=template,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            margin=dict(l=10, r=10, t=60, b=10)
        )
        
        # Thiết lập grid
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)' if self.style != "dark" else 'rgba(255,255,255,0.1)'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)' if self.style != "dark" else 'rgba(255,255,255,0.1)'
        )
        
        # Hiển thị biểu đồ trên Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Thêm chức năng tải xuống nếu cần
        if allow_download:
            self._add_download_button(fig, "rolling_stats.html", "Tải xuống biểu đồ")
            
        return fig
    
    def _plot_rolling_stats_matplotlib(
        self,
        df_metrics: pd.DataFrame,
        metrics: List[str],
        window: int,
        title: str,
        height: int,
        allow_download: bool
    ) -> plt.Figure:
        """
        Vẽ biểu đồ các chỉ số hiệu suất sử dụng Matplotlib.
        
        Args: (xem phương thức plot_rolling_stats)
            
        Returns:
            Đối tượng Figure của matplotlib
        """
        # Xác định số lượng subplots
        n_plots = len(metrics)
        
        # Tính toán kích thước trong inch (1 px = 1/96 inch)
        figsize = (10, height / 96)
        
        # Tạo figure và axes
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True, gridspec_kw={'hspace': 0.3})
        
        # Đảm bảo axes là list kể cả khi chỉ có 1 metric
        if n_plots == 1:
            axes = [axes]
            
        # Vẽ các chỉ số
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Định dạng metric name
            metric_name = metric.replace('_', ' ').title()
            
            # Định dạng giá trị
            if metric in ['returns', 'volatility', 'drawdown', 'win_rate']:
                y_values = df_metrics[metric] * 100  # Chuyển sang phần trăm
                y_title = "%"
            else:
                y_values = df_metrics[metric]
                y_title = ""
                
            # Vẽ đường
            ax.plot(df_metrics.index, y_values, label=f'{metric_name} ({window}d)', linewidth=2)
            
            # Thêm đường ngang tại 0 nếu cần
            if metric in ['returns', 'sharpe', 'drawdown']:
                ax.axhline(0, color='black', linestyle='--', alpha=0.3)
            
            # Thêm vùng tô màu theo điều kiện
            if metric == 'returns':
                ax.fill_between(df_metrics.index, 0, y_values, 
                               where=y_values >= 0, color='green', alpha=0.3)
                ax.fill_between(df_metrics.index, 0, y_values, 
                               where=y_values < 0, color='red', alpha=0.3)
            
            elif metric == 'drawdown':
                ax.fill_between(df_metrics.index, 0, y_values, color='red', alpha=0.3)
            
            elif metric == 'sharpe':
                ax.fill_between(df_metrics.index, 0, y_values, 
                               where=y_values >= 0, color='green', alpha=0.3)
                ax.fill_between(df_metrics.index, 0, y_values, 
                               where=y_values < 0, color='red', alpha=0.3)
                
                # Thêm đường tham chiếu cho Sharpe tốt (>1)
                ax.axhline(1, color='green', linestyle='--', alpha=0.5)
            
            # Thiết lập trục y
            if metric == 'returns':
                ax.set_ylabel(f'Lợi nhuận ({y_title})')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}%'.format(x)))
            elif metric == 'volatility':
                ax.set_ylabel(f'Độ biến động ({y_title})')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}%'.format(x)))
            elif metric == 'sharpe':
                ax.set_ylabel('Sharpe Ratio')
            elif metric == 'drawdown':
                ax.set_ylabel(f'Drawdown ({y_title})')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}%'.format(x)))
            elif metric == 'win_rate':
                ax.set_ylabel(f'Tỷ lệ thắng ({y_title})')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}%'.format(x)))
            
            # Tính giá trị trung bình và hiển thị
            avg_value = df_metrics[metric].mean()
            if metric in ['returns', 'volatility', 'drawdown', 'win_rate']:
                avg_value_display = avg_value * 100
            else:
                avg_value_display = avg_value
                
            ax.axhline(avg_value_display, color='blue', linestyle=':', alpha=0.7)
            
            # Thêm chú thích giá trị trung bình
            if metric in ['returns', 'volatility', 'drawdown', 'win_rate']:
                text = f'Trung bình: {avg_value:.2%}'
            else:
                text = f'Trung bình: {avg_value:.2f}'
            
            ax.annotate(text, xy=(0.02, 0.05), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
                
            # Thêm legend và grid
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
        # Thiết lập trục x cho subplot cuối cùng
        axes[-1].set_xlabel('Thời gian')
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        
        # Thiết lập tiêu đề chính
        fig.suptitle(title, fontsize=14)
        
        # Điều chỉnh layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        
        # Hiển thị biểu đồ trên Streamlit
        st.pyplot(fig)
        
        # Thêm chức năng tải xuống nếu cần
        if allow_download:
            self._add_download_button_matplotlib(fig, "rolling_stats.png", "Tải xuống biểu đồ")
            
        return fig
    
    def create_dashboard_widgets(
        self,
        price_data: pd.DataFrame,
        trades_data: Optional[pd.DataFrame] = None,
        portfolio_data: Optional[pd.DataFrame] = None,
        metrics_data: Optional[Dict[str, Any]] = None,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Tạo các widget điều khiển cho dashboard.
        
        Args:
            price_data: DataFrame chứa dữ liệu giá
            trades_data: DataFrame chứa dữ liệu giao dịch
            portfolio_data: DataFrame chứa dữ liệu danh mục đầu tư
            metrics_data: Dict chứa dữ liệu các chỉ số
            symbols: Danh sách các mã token/cặp giao dịch
            timeframes: Danh sách các khung thời gian
            start_date: Ngày bắt đầu mặc định
            end_date: Ngày kết thúc mặc định
            
        Returns:
            Tuple gồm (filters, selectors)
        """
        st.sidebar.header("Điều khiển Dashboard")
        
        # Khởi tạo các dict lưu trữ widgets
        filters = {}
        selectors = {}
        
        # Widget chọn mã token/cặp giao dịch
        if symbols is None and price_data is not None and 'symbol' in price_data.columns:
            symbols = sorted(price_data['symbol'].unique())
            
        if symbols is not None and len(symbols) > 0:
            selectors['symbol'] = st.sidebar.selectbox(
                "Chọn cặp giao dịch",
                options=symbols,
                index=0
            )
            
        # Widget chọn khung thời gian
        if timeframes is None:
            timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
            
        selectors['timeframe'] = st.sidebar.selectbox(
            "Khung thời gian",
            options=timeframes,
            index=timeframes.index("1d") if "1d" in timeframes else 0
        )
        
        # Widget chọn khoảng thời gian
        st.sidebar.subheader("Khoảng thời gian")
        
        # Xác định min/max date từ dữ liệu
        min_date = None
        max_date = None
        
        # Từ price_data
        if price_data is not None and isinstance(price_data.index, pd.DatetimeIndex):
            min_date = price_data.index.min()
            max_date = price_data.index.max()
        elif price_data is not None and 'timestamp' in price_data.columns:
            min_date = pd.to_datetime(price_data['timestamp']).min()
            max_date = pd.to_datetime(price_data['timestamp']).max()
            
        # Từ portfolio_data
        if portfolio_data is not None and isinstance(portfolio_data.index, pd.DatetimeIndex):
            if min_date is None or portfolio_data.index.min() < min_date:
                min_date = portfolio_data.index.min()
            if max_date is None or portfolio_data.index.max() > max_date:
                max_date = portfolio_data.index.max()
                
        # Từ trades_data
        if trades_data is not None and 'entry_time' in trades_data.columns:
            entry_times = pd.to_datetime(trades_data['entry_time'])
            if min_date is None or entry_times.min() < min_date:
                min_date = entry_times.min()
            if max_date is None or entry_times.max() > max_date:
                max_date = entry_times.max()
                
        # Thiết lập giá trị mặc định
        if start_date is None and min_date is not None:
            start_date = min_date
        if end_date is None and max_date is not None:
            end_date = max_date
            
        # Nếu vẫn chưa có giá trị, sử dụng giá trị mặc định
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
            
        # Widget chọn ngày
        filters['date_range'] = st.sidebar.date_input(
            "Chọn khoảng thời gian",
            value=(start_date.date(), end_date.date()),
            min_value=min_date.date() if min_date is not None else None,
            max_value=max_date.date() if max_date is not None else None
        )
        
        # Widget chọn loại biểu đồ
        selectors['chart_type'] = st.sidebar.selectbox(
            "Loại biểu đồ",
            options=["candlestick", "line", "ohlc", "area"],
            index=0,
            format_func=lambda x: {
                "candlestick": "Nến (Candlestick)",
                "line": "Đường (Line)",
                "ohlc": "OHLC",
                "area": "Vùng (Area)"
            }.get(x, x)
        )
        
        # Widget hiển thị chỉ báo
        st.sidebar.subheader("Chỉ báo kỹ thuật")
        
        filters['show_volume'] = st.sidebar.checkbox("Hiển thị khối lượng", value=True)
        
        selectors['indicators'] = []
        if st.sidebar.checkbox("SMA", value=True):
            selectors['indicators'].append("sma")
        if st.sidebar.checkbox("EMA", value=True):
            selectors['indicators'].append("ema")
        if st.sidebar.checkbox("Bollinger Bands", value=True):
            selectors['indicators'].append("bollinger")
        if st.sidebar.checkbox("RSI", value=False):
            selectors['indicators'].append("rsi")
        if st.sidebar.checkbox("MACD", value=False):
            selectors['indicators'].append("macd")
            
        # Widget hiển thị giao dịch
        if trades_data is not None:
            filters['show_trades'] = st.sidebar.checkbox("Hiển thị giao dịch", value=True)
            
        # Widget phân tích hiệu suất
        st.sidebar.subheader("Phân tích hiệu suất")
        
        if portfolio_data is not None:
            selectors['metrics'] = []
            if st.sidebar.checkbox("Đường cong vốn", value=True):
                selectors['metrics'].append("equity")
            if st.sidebar.checkbox("Drawdown", value=True):
                selectors['metrics'].append("drawdown")
            if st.sidebar.checkbox("Lợi nhuận theo tháng", value=False):
                selectors['metrics'].append("monthly_returns")
            if st.sidebar.checkbox("Lợi nhuận hàng ngày", value=False):
                selectors['metrics'].append("daily_returns")
                
        # Widget cài đặt biểu đồ
        st.sidebar.subheader("Cài đặt biểu đồ")
        
        selectors['height'] = st.sidebar.slider("Chiều cao biểu đồ", min_value=300, max_value=1000, value=600, step=50)
        
        filters['allow_download'] = st.sidebar.checkbox("Cho phép tải xuống biểu đồ", value=True)
        
        return filters, selectors
    
    def display_metrics_cards(
        self,
        metrics: Dict[str, Union[float, int, str]]
    ) -> None:
        """
        Hiển thị các chỉ số quan trọng dưới dạng card.
        
        Args:
            metrics: Dict chứa các chỉ số cần hiển thị
        """
        # Xác định số lượng cột
        n_cols = min(4, len(metrics))
        
        if n_cols == 0:
            return
            
        cols = st.columns(n_cols)
        
        # Hiển thị các chỉ số
        for i, (metric_name, value) in enumerate(metrics.items()):
            col_idx = i % n_cols
            
            # Định dạng giá trị
            if isinstance(value, float):
                if metric_name.lower() in ['total_return', 'win_rate', 'max_drawdown', 'volatility', 'annual_return']:
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
                
            # Định dạng tên metric
            display_name = metric_name.replace('_', ' ').title()
            
            # Hiển thị metric
            cols[col_idx].metric(display_name, formatted_value)
    
    def display_backtest_report(
        self,
        strategy_name: str,
        backtest_result: Dict[str, Any],
        equity_data: Optional[pd.DataFrame] = None,
        trades_data: Optional[pd.DataFrame] = None,
        allow_download: bool = True
    ) -> None:
        """
        Hiển thị báo cáo backtest tổng hợp.
        
        Args:
            strategy_name: Tên chiến lược
            backtest_result: Dict chứa kết quả backtest
            equity_data: DataFrame chứa dữ liệu equity
            trades_data: DataFrame chứa dữ liệu giao dịch
            allow_download: Cho phép tải xuống báo cáo
        """
        # Kiểm tra dữ liệu đầu vào
        if backtest_result is None:
            st.error("Không có dữ liệu backtest để hiển thị!")
            return
            
        # Thiết lập tiêu đề
        st.header(f"Báo cáo Backtest - {strategy_name}")
        
        # Hiển thị tổng quan
        st.subheader("Tổng quan")
        
        # Lấy thông tin cơ bản
        metrics = {}
        
        # Từ kết quả tổng hợp
        if "combined_result" in backtest_result:
            result = backtest_result["combined_result"]
            metrics["total_return"] = result.get("roi", 0)
            metrics["initial_balance"] = result.get("initial_balance", 0)
            metrics["final_balance"] = result.get("final_balance", 0)
            
            # Thêm các metrics khác
            if "metrics" in result:
                for key, value in result["metrics"].items():
                    metrics[key] = value
        
        # Từ kết quả theo symbol
        elif "symbol_results" in backtest_result:
            # Tính tổng hợp từ các symbol
            total_initial_balance = 0
            total_final_balance = 0
            metrics_sum = {}
            metrics_count = {}
            
            for symbol, symbol_result in backtest_result["symbol_results"].items():
                if isinstance(symbol_result, dict) and symbol_result.get("status") == "success":
                    total_initial_balance += symbol_result.get("initial_balance", 0)
                    total_final_balance += symbol_result.get("final_balance", 0)
                    
                    # Tính tổng các metrics
                    if "metrics" in symbol_result:
                        for metric, value in symbol_result["metrics"].items():
                            if metric not in metrics_sum:
                                metrics_sum[metric] = 0
                                metrics_count[metric] = 0
                            
                            metrics_sum[metric] += value
                            metrics_count[metric] += 1
            
            # Tính trung bình các metrics
            for metric, total in metrics_sum.items():
                if metrics_count[metric] > 0:
                    metrics[metric] = total / metrics_count[metric]
            
            # Tính ROI tổng hợp
            if total_initial_balance > 0:
                metrics["total_return"] = (total_final_balance - total_initial_balance) / total_initial_balance
            else:
                metrics["total_return"] = 0
                
            metrics["initial_balance"] = total_initial_balance
            metrics["final_balance"] = total_final_balance
            
        # Hiển thị các chỉ số
        self.display_metrics_cards(metrics)
        
        # Hiển thị thông tin cấu hình
        st.subheader("Cấu hình")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Tham số chiến lược:**")
            if "strategy_params" in backtest_result:
                for key, value in backtest_result["strategy_params"].items():
                    st.write(f"- {key}: {value}")
            else:
                st.write("Không có thông tin")
                
        with col2:
            st.write("**Tham số backtest:**")
            if "backtest_params" in backtest_result:
                for key, value in backtest_result["backtest_params"].items():
                    st.write(f"- {key}: {value}")
            else:
                st.write("Không có thông tin")
                
        # Hiển thị đường cong vốn nếu có dữ liệu
        if equity_data is not None:
            st.subheader("Đường cong vốn")
            
            self.plot_price_chart(
                price_data=equity_data,
                chart_type="line",
                title="Đường cong vốn",
                show_volume=False,
                allow_download=allow_download
            )
            
            # Hiển thị phân tích drawdown
            st.subheader("Phân tích Drawdown")
            
            self.plot_drawdown_chart(
                equity_data=equity_data,
                title="Phân tích Drawdown",
                allow_download=allow_download
            )
            
        # Hiển thị thống kê giao dịch nếu có dữ liệu
        if trades_data is not None:
            st.subheader("Thống kê giao dịch")
            
            self.display_trade_details(
                trades_data=trades_data,
                include_metrics=True,
                allow_download=allow_download
            )
            
        # Hiển thị hiệu suất theo thời gian
        if equity_data is not None:
            st.subheader("Hiệu suất theo thời gian")
            
            self.plot_rolling_stats(
                equity_data=equity_data,
                title="Các chỉ số hiệu suất theo thời gian",
                allow_download=allow_download
            )
            
        # Nút tải xuống báo cáo đầy đủ
        if allow_download and BACKTEST_VISUALIZER_AVAILABLE:
            st.subheader("Tải xuống báo cáo")
            
            # Kiểm tra xem đã có sẵn BacktestVisualizer chưa
            if not hasattr(self, 'backtest_visualizer'):
                self.backtest_visualizer = BacktestVisualizer(
                    style='dark' if self.style == 'dark' else 'default',
                    interactive=self.use_plotly,
                    logger=self.logger
                )
                
            # Tạm thời lưu dữ liệu vào backtest_visualizer
            if equity_data is not None:
                self.backtest_visualizer.load_equity_curve(equity_data, strategy_name)
                
            if trades_data is not None:
                self.backtest_visualizer.load_trade_history(trades_data, strategy_name)
                
            # Tạo và lưu báo cáo
            if st.button("Tạo báo cáo HTML"):
                with st.spinner("Đang tạo báo cáo..."):
                    try:
                        # Tạo báo cáo
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        report_path = Path(f"./reports/backtest_report_{strategy_name}_{timestamp}.html")
                        report_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        output_path = self.backtest_visualizer.create_backtest_report(
                            strategy_name=strategy_name,
                            output_path=report_path,
                            format='html'
                        )
                        
                        if output_path:
                            # Tạo nút tải xuống
                            with open(output_path, "rb") as f:
                                report_data = f.read()
                                
                            b64 = base64.b64encode(report_data).decode()
                            href = f'<a href="data:text/html;base64,{b64}" download="backtest_report_{strategy_name}.html" class="btn">Tải xuống báo cáo HTML</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        else:
                            st.error("Không thể tạo báo cáo. Vui lòng kiểm tra logs để biết thêm chi tiết.")
                            
                    except Exception as e:
                        st.error(f"Lỗi khi tạo báo cáo: {e}")
                        
    def plot_equity_curve(
        self,
        price_data: pd.DataFrame,
        chart_type: str = "line",
        title: str = "Đường cong vốn",
        show_volume: bool = False,
        benchmark_data: Optional[pd.DataFrame] = None,
        allow_download: bool = False
    ) -> Any:
        """
        Vẽ biểu đồ đường cong vốn (equity curve).
        
        Args:
            price_data: DataFrame chứa dữ liệu equity
            chart_type: Loại biểu đồ ('line', 'area')
            title: Tiêu đề biểu đồ
            show_volume: Hiển thị khối lượng
            benchmark_data: DataFrame chứa dữ liệu chuẩn so sánh
            allow_download: Cho phép tải xuống biểu đồ
            
        Returns:
            Đối tượng figure hoặc None
        """
        # Kiểm tra dữ liệu đầu vào
        if price_data is None or len(price_data) == 0:
            st.error("Không có dữ liệu equity để hiển thị!")
            return None
            
        # Chuyển đổi index thành datetime nếu chưa phải
        if not isinstance(price_data.index, pd.DatetimeIndex):
            # Kiểm tra xem có cột timestamp/datetime không
            if 'timestamp' in price_data.columns:
                price_data = price_data.set_index('timestamp')
            elif 'datetime' in price_data.columns:
                price_data = price_data.set_index('datetime')
            elif 'date' in price_data.columns:
                price_data = price_data.set_index('date')
            else:
                # Nếu không có cột thời gian, sử dụng chỉ số làm index
                price_data = price_data.copy()
                price_data.index = pd.to_datetime(price_data.index)
                
        # Chuẩn hóa tên cột (viết thường)
        price_data.columns = [col.lower() for col in price_data.columns]
        
        # Xác định cột equity
        equity_col = None
        for col_name in ['equity', 'balance', 'value', 'close']:
            if col_name in price_data.columns:
                equity_col = col_name
                break
                
        if equity_col is None:
            # Sử dụng cột đầu tiên
            equity_col = price_data.columns[0]
            
        # Tạo dữ liệu OHLC cho equity
        ohlc_data = pd.DataFrame(index=price_data.index)
        ohlc_data['open'] = price_data[equity_col]
        ohlc_data['high'] = price_data[equity_col]
        ohlc_data['low'] = price_data[equity_col]
        ohlc_data['close'] = price_data[equity_col]
        
        if 'volume' in price_data.columns and show_volume:
            ohlc_data['volume'] = price_data['volume']
        
        # Vẽ biểu đồ
        if self.use_plotly:
            # Sử dụng Plotly
            fig = go.Figure()
            
            if chart_type == "line":
                fig.add_trace(
                    go.Scatter(
                        x=ohlc_data.index,
                        y=ohlc_data['close'],
                        mode='lines',
                        name='Equity',
                        line=dict(color='blue', width=2)
                    )
                )
            elif chart_type == "area":
                fig.add_trace(
                    go.Scatter(
                        x=ohlc_data.index,
                        y=ohlc_data['close'],
                        mode='lines',
                        name='Equity',
                        fill='tozeroy',
                        fillcolor='rgba(0, 0, 255, 0.2)',
                        line=dict(color='blue', width=2)
                    )
                )
                
            # Thêm benchmark nếu có
            if benchmark_data is not None:
                if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                    if 'timestamp' in benchmark_data.columns:
                        benchmark_data = benchmark_data.set_index('timestamp')
                    elif 'datetime' in benchmark_data.columns:
                        benchmark_data = benchmark_data.set_index('datetime')
                    elif 'date' in benchmark_data.columns:
                        benchmark_data = benchmark_data.set_index('date')
                        
                # Chuẩn hóa tên cột
                benchmark_data.columns = [col.lower() for col in benchmark_data.columns]
                
                # Xác định cột benchmark
                benchmark_col = None
                for col_name in ['close', 'value', 'price']:
                    if col_name in benchmark_data.columns:
                        benchmark_col = col_name
                        break
                        
                if benchmark_col is None:
                    benchmark_col = benchmark_data.columns[0]
                    
                # Chuẩn hóa benchmark để so sánh
                benchmark_start = benchmark_data[benchmark_col].iloc[0]
                equity_start = ohlc_data['close'].iloc[0]
                
                normalized_benchmark = benchmark_data[benchmark_col] / benchmark_start * equity_start
                
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_data.index,
                        y=normalized_benchmark,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='gray', width=1.5, dash='dash')
                    )
                )
                
            # Cập nhật layout
            template = "plotly_white" if self.style != "dark" else "plotly_dark"
            fig.update_layout(
                title=title,
                xaxis_title="Thời gian",
                yaxis_title="Giá trị vốn",
                template=template,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode="x unified",
                margin=dict(l=10, r=10, t=60, b=10)
            )
            
            # Hiển thị biểu đồ trên Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            # Thêm chức năng tải xuống nếu cần
            if allow_download:
                self._add_download_button(fig, "equity_curve.html", "Tải xuống biểu đồ")
                
            return fig
        else:
            # Sử dụng Matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == "line":
                ax.plot(ohlc_data.index, ohlc_data['close'], label='Equity', color='blue', linewidth=2)
            elif chart_type == "area":
                ax.fill_between(ohlc_data.index, 0, ohlc_data['close'], alpha=0.3, color='blue')
                ax.plot(ohlc_data.index, ohlc_data['close'], label='Equity', color='blue', linewidth=2)
                
            # Thêm benchmark nếu có
            if benchmark_data is not None:
                if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                    if 'timestamp' in benchmark_data.columns:
                        benchmark_data = benchmark_data.set_index('timestamp')
                    elif 'datetime' in benchmark_data.columns:
                        benchmark_data = benchmark_data.set_index('datetime')
                    elif 'date' in benchmark_data.columns:
                        benchmark_data = benchmark_data.set_index('date')
                        
                # Chuẩn hóa tên cột
                benchmark_data.columns = [col.lower() for col in benchmark_data.columns]
                
                # Xác định cột benchmark
                benchmark_col = None
                for col_name in ['close', 'value', 'price']:
                    if col_name in benchmark_data.columns:
                        benchmark_col = col_name
                        break
                        
                if benchmark_col is None:
                    benchmark_col = benchmark_data.columns[0]
                    
                # Chuẩn hóa benchmark để so sánh
                benchmark_start = benchmark_data[benchmark_col].iloc[0]
                equity_start = ohlc_data['close'].iloc[0]
                
                normalized_benchmark = benchmark_data[benchmark_col] / benchmark_start * equity_start
                
                ax.plot(benchmark_data.index, normalized_benchmark, label='Benchmark', color='gray', linestyle='--', linewidth=1.5)
                
            # Cập nhật layout
            ax.set_title(title)
            ax.set_xlabel("Thời gian")
            ax.set_ylabel("Giá trị vốn")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            # Định dạng trục x
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            plt.tight_layout()
            
            # Hiển thị biểu đồ trên Streamlit
            st.pyplot(fig)
            
            # Thêm chức năng tải xuống nếu cần
            if allow_download:
                self._add_download_button_matplotlib(fig, "equity_curve.png", "Tải xuống biểu đồ")
                
            return fig
    
    def _get_subplot_titles(self, metrics: List[str], plot_data: Dict[str, pd.Series]) -> List[str]:
        """
        Lấy tiêu đề cho các subplot.
        
        Args:
            metrics: Danh sách các chỉ số
            plot_data: Dữ liệu cho các biểu đồ
            
        Returns:
            Danh sách tiêu đề cho các subplot
        """
        titles = []
        
        for metric in metrics:
            if metric == "equity" and "equity" in plot_data:
                titles.append("Đường cong vốn")
            elif metric == "drawdown" and "drawdown" in plot_data:
                titles.append("Drawdown")
            elif metric == "returns" and "returns" in plot_data:
                titles.append("Lợi nhuận hàng ngày")
            elif metric == "cumulative_returns" and "equity_norm" in plot_data:
                titles.append("Lợi nhuận tích lũy")
            elif metric == "monthly_returns":
                titles.append("Lợi nhuận theo tháng")
            elif metric == "daily_returns":
                titles.append("Lợi nhuận hàng ngày")
                
        return titles
    
    def _calculate_performance_stats(self, equity_series: pd.Series) -> Dict[str, float]:
        """
        Tính toán các chỉ số hiệu suất từ dữ liệu equity.
        
        Args:
            equity_series: Series chứa dữ liệu equity
            
        Returns:
            Dict chứa các chỉ số hiệu suất
        """
        stats = {}
        
        # Tính lợi nhuận tổng
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
        stats['total_return'] = total_return
        
        # Tính lợi nhuận hàng ngày
        daily_returns = equity_series.pct_change().fillna(0)
        
        # Tính volatility
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        stats['volatility'] = volatility
        
        # Tính max drawdown
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        stats['max_drawdown'] = max_drawdown
        
        # Tính Sharpe ratio (giả sử lãi suất phi rủi ro là 2%)
        risk_free_rate = 0.02 / 252  # Daily
        sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std() * np.sqrt(252)
        stats['sharpe_ratio'] = sharpe_ratio
        
        # Tính Sortino ratio (lấy chỉ các returns âm cho downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (daily_returns.mean() - risk_free_rate) / downside_deviation * np.sqrt(252) if downside_deviation != 0 else 0
        else:
            sortino_ratio = float('inf')  # Không có returns âm
        stats['sortino_ratio'] = sortino_ratio
        
        # Tính lợi nhuận hàng năm
        days = (equity_series.index[-1] - equity_series.index[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0
        stats['annual_return'] = annual_return
        
        # Tính Calmar ratio (lợi nhuận hàng năm / max drawdown)
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else float('inf')
        stats['calmar_ratio'] = calmar_ratio
        
        # Tính giá trị lớn nhất, nhỏ nhất
        max_value = equity_series.max()
        min_value = equity_series.min()
        stats['max_value'] = max_value
        stats['min_value'] = min_value
        
        return stats
    
    def _add_download_button(
        self, 
        fig: go.Figure, 
        filename: str, 
        button_text: str
    ) -> None:
        """
        Thêm nút tải xuống cho biểu đồ Plotly.
        
        Args:
            fig: Đối tượng Figure của plotly
            filename: Tên file
            button_text: Chữ trên nút tải xuống
        """
        # Tạo HTML cho biểu đồ
        html_data = fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        # Tạo HTML đầy đủ
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <title>{filename}</title>
        </head>
        <body>
            {html_data}
        </body>
        </html>
        """
        
        # Mã hóa dữ liệu
        b64 = base64.b64encode(full_html.encode()).decode()
        
        # Tạo nút tải xuống
        href = f'<a href="data:text/html;base64,{b64}" download="{filename}" class="btn">{button_text}</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def _add_download_button_matplotlib(
        self, 
        fig: plt.Figure, 
        filename: str, 
        button_text: str
    ) -> None:
        """
        Thêm nút tải xuống cho biểu đồ Matplotlib.
        
        Args:
            fig: Đối tượng Figure của matplotlib
            filename: Tên file
            button_text: Chữ trên nút tải xuống
        """
        # Lưu biểu đồ vào buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Mã hóa dữ liệu
        b64 = base64.b64encode(buf.read()).decode()
        
        # Tạo nút tải xuống
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}" class="btn">{button_text}</a>'
        st.markdown(href, unsafe_allow_html=True)
        
    def create_live_dashboard(
        self,
        price_data_source: Callable,
        trades_data_source: Optional[Callable] = None,
        portfolio_data_source: Optional[Callable] = None,
        auto_update_interval: float = 60.0,  # Giây
        max_points: int = 1000
    ) -> None:
        """
        Tạo dashboard trực tiếp với dữ liệu thời gian thực.
        
        Args:
            price_data_source: Hàm lấy dữ liệu giá mới nhất
            trades_data_source: Hàm lấy dữ liệu giao dịch mới nhất
            portfolio_data_source: Hàm lấy dữ liệu danh mục mới nhất
            auto_update_interval: Thời gian giữa các lần cập nhật tự động (giây)
            max_points: Số lượng điểm dữ liệu tối đa hiển thị
        """
        # Tiêu đề dashboard
        st.title("Dashboard Giao dịch Trực tiếp")
        
        # Tạo trạng thái session nếu chưa có
        if "last_update" not in st.session_state:
            st.session_state.last_update = datetime.now()
        if "price_data" not in st.session_state:
            st.session_state.price_data = None
        if "trades_data" not in st.session_state:
            st.session_state.trades_data = None
        if "portfolio_data" not in st.session_state:
            st.session_state.portfolio_data = None
            
        # Widget điều khiển
        col1, col2, col3 = st.columns(3)
        
        with col1:
            auto_update = st.checkbox("Tự động cập nhật", value=True)
            
        with col2:
            update_interval = st.slider(
                "Thời gian cập nhật (giây)",
                min_value=5.0,
                max_value=300.0,
                value=auto_update_interval,
                step=5.0,
                disabled=not auto_update
            )
            
        with col3:
            if st.button("Cập nhật ngay"):
                # Cập nhật dữ liệu ngay lập tức
                try:
                    st.session_state.price_data = price_data_source(max_points)
                    if trades_data_source:
                        st.session_state.trades_data = trades_data_source(max_points)
                    if portfolio_data_source:
                        st.session_state.portfolio_data = portfolio_data_source(max_points)
                    st.session_state.last_update = datetime.now()
                except Exception as e:
                    st.error(f"Lỗi khi cập nhật dữ liệu: {e}")
                    
        # Hiển thị thời gian cập nhật gần nhất
        st.caption(f"Cập nhật gần nhất: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
                    
        # Kiểm tra xem có cần cập nhật tự động không
        if auto_update and (datetime.now() - st.session_state.last_update).total_seconds() >= update_interval:
            try:
                st.session_state.price_data = price_data_source(max_points)
                if trades_data_source:
                    st.session_state.trades_data = trades_data_source(max_points)
                if portfolio_data_source:
                    st.session_state.portfolio_data = portfolio_data_source(max_points)
                st.session_state.last_update = datetime.now()
            except Exception as e:
                st.error(f"Lỗi khi cập nhật dữ liệu tự động: {e}")
                
        # Tạo tabs hiển thị
        tab1, tab2, tab3 = st.tabs(["Biểu đồ giá", "Giao dịch", "Danh mục đầu tư"])
        
        with tab1:
            # Hiển thị biểu đồ giá
            if st.session_state.price_data is not None:
                symbols = sorted(st.session_state.price_data['symbol'].unique()) if 'symbol' in st.session_state.price_data.columns else None
                
                if symbols and len(symbols) > 0:
                    selected_symbol = st.selectbox("Chọn cặp giao dịch", options=symbols)
                    symbol_data = st.session_state.price_data[st.session_state.price_data['symbol'] == selected_symbol]
                else:
                    symbol_data = st.session_state.price_data
                    
                # Lọc dữ liệu trades nếu có
                symbol_trades = None
                if st.session_state.trades_data is not None and 'symbol' in st.session_state.trades_data.columns:
                    symbol_trades = st.session_state.trades_data[st.session_state.trades_data['symbol'] == selected_symbol]
                    
                # Hiển thị biểu đồ
                self.plot_price_chart(
                    price_data=symbol_data,
                    trades_data=symbol_trades,
                    symbol=selected_symbol if symbols else "",
                    chart_type="candlestick",
                    show_volume=True,
                    show_indicators=["sma", "ema", "bollinger"]
                )
                
        with tab2:
            # Hiển thị giao dịch
            if st.session_state.trades_data is not None and len(st.session_state.trades_data) > 0:
                self.display_trade_details(
                    trades_data=st.session_state.trades_data,
                    include_metrics=True
                )
            else:
                st.info("Không có dữ liệu giao dịch để hiển thị.")
                
        with tab3:
            # Hiển thị danh mục đầu tư
            if st.session_state.portfolio_data is not None:
                self.plot_portfolio_performance(
                    portfolio_data=st.session_state.portfolio_data,
                    metrics=["equity", "drawdown"]
                )
            else:
                st.info("Không có dữ liệu danh mục đầu tư để hiển thị.")
                
        # Tự động cập nhật biểu đồ
        if auto_update:
            st.empty()
            time.sleep(1)  # Tránh quá nhiều cập nhật
            st.rerun()