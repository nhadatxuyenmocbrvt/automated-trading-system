#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dashboard giao dịch thời gian thực cho Automated Trading System.

File này cung cấp giao diện đồ họa để theo dõi hiệu suất giao dịch,
phân tích lợi nhuận/rủi ro, và điều khiển các hoạt động giao dịch
thông qua Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import json
from typing import Dict, List, Tuple, Optional, Any, Union

# Thêm đường dẫn gốc của dự án vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import các module nội bộ
from logs.metrics.trading_metrics import TradingMetricsTracker, MultiSymbolTradingMetricsTracker
from logs.logger import get_system_logger, SystemLogger
from config.system_config import get_system_config, SystemConfig
from config.constants import BacktestMetric
from deployment.exchange_api.account_manager import AccountManager
from deployment.exchange_api.position_tracker import PositionTracker
from risk_management.risk_calculator import RiskCalculator
from backtesting.visualization.performance_charts import PerformanceCharts
from backtesting.performance_metrics import PerformanceMetrics
from streamlit_dashboard.components.sidebar import create_sidebar, create_account_section
from streamlit_dashboard.components.metrics_display import display_trading_metrics


# Khởi tạo logger
logger = get_system_logger("trading_dashboard")

# Lấy cấu hình hệ thống
SYSTEM_CONFIG = get_system_config()

def load_trading_data(strategy_name: str, symbol: str = 'all', days: int = 30) -> pd.DataFrame:
    """
    Tải dữ liệu giao dịch từ file log
    
    Args:
        strategy_name: Tên của chiến lược
        symbol: Cặp tiền (all hoặc tên cặp cụ thể)
        days: Số ngày dữ liệu gần nhất cần lấy
        
    Returns:
        DataFrame chứa dữ liệu giao dịch
    """
    try:
        # Xác định đường dẫn file log
        log_dir = SYSTEM_CONFIG.get("log_dir", "./logs")
        
        if symbol == 'all':
            # Tải từ file combined
            log_path = os.path.join(
                log_dir, 
                "trading", 
                f"{strategy_name.replace('/', '_')}_combined_metrics.json"
            )
        else:
            # Tải từ file specific symbol
            log_path = os.path.join(
                log_dir, 
                "trading", 
                symbol.replace('/', '_'),
                f"{strategy_name.replace('/', '_')}_metrics.json"
            )
        
        if not os.path.exists(log_path):
            logger.warning(f"Không tìm thấy file log: {log_path}")
            return pd.DataFrame()
        
        # Đọc dữ liệu từ file JSON
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Kiểm tra cấu trúc dữ liệu
        if "metrics_history" not in data:
            logger.warning(f"Dữ liệu không đúng cấu trúc: {log_path}")
            return pd.DataFrame()
        
        # Lấy lịch sử giao dịch
        trades_data = data["metrics_history"]["trades"]
        
        if not trades_data:
            logger.warning(f"Không có dữ liệu giao dịch: {log_path}")
            return pd.DataFrame()
        
        # Chuyển đổi thành DataFrame
        df = pd.DataFrame(trades_data)
        
        # Chuyển đổi các cột thời gian
        for col in ['entry_time', 'exit_time', 'timestamp']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Lọc dữ liệu theo số ngày
        if 'exit_time' in df.columns:  # Dùng exit_time để lọc
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['exit_time'] >= cutoff_date]
        elif 'entry_time' in df.columns:  # Hoặc dùng entry_time nếu không có exit_time
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['entry_time'] >= cutoff_date]
        
        return df
    
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu giao dịch: {e}")
        return pd.DataFrame()

def load_balance_history(strategy_name: str, days: int = 30) -> pd.DataFrame:
    """
    Tải dữ liệu lịch sử số dư từ file log
    
    Args:
        strategy_name: Tên của chiến lược
        days: Số ngày dữ liệu gần nhất cần lấy
        
    Returns:
        DataFrame chứa lịch sử số dư
    """
    try:
        # Xác định đường dẫn file log
        log_dir = SYSTEM_CONFIG.get("log_dir", "./logs")
        log_path = os.path.join(
            log_dir, 
            "trading", 
            f"{strategy_name.replace('/', '_')}_metrics.json"
        )
        
        if not os.path.exists(log_path):
            logger.warning(f"Không tìm thấy file log: {log_path}")
            return pd.DataFrame()
        
        # Đọc dữ liệu từ file JSON
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Kiểm tra cấu trúc dữ liệu
        if "metrics_history" not in data or "capital_history" not in data["metrics_history"]:
            logger.warning(f"Dữ liệu không đúng cấu trúc: {log_path}")
            return pd.DataFrame()
        
        # Lấy lịch sử số dư
        balance_data = data["metrics_history"]["capital_history"]
        
        if not balance_data:
            logger.warning(f"Không có dữ liệu lịch sử số dư: {log_path}")
            return pd.DataFrame()
        
        # Chuyển đổi thành DataFrame
        df = pd.DataFrame(balance_data)
        
        # Chuyển đổi cột timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Lọc dữ liệu theo số ngày
        if 'timestamp' in df.columns:
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff_date]
        
        # Sắp xếp dữ liệu theo thời gian
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        return df
    
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu lịch sử số dư: {e}")
        return pd.DataFrame()

def load_active_positions(strategy_name: str) -> pd.DataFrame:
    """
    Tải dữ liệu vị thế đang mở
    
    Args:
        strategy_name: Tên của chiến lược
        
    Returns:
        DataFrame chứa dữ liệu vị thế đang mở
    """
    try:
        # Xác định đường dẫn file
        log_dir = SYSTEM_CONFIG.get("log_dir", "./logs")
        positions_path = os.path.join(
            log_dir, 
            "trading", 
            f"{strategy_name.replace('/', '_')}_active_positions.json"
        )
        
        if not os.path.exists(positions_path):
            logger.warning(f"Không tìm thấy file vị thế đang mở: {positions_path}")
            return pd.DataFrame()
        
        # Đọc dữ liệu từ file JSON
        with open(positions_path, 'r', encoding='utf-8') as f:
            positions = json.load(f)
        
        if not positions:
            logger.info(f"Không có vị thế đang mở: {positions_path}")
            return pd.DataFrame()
        
        # Chuyển đổi thành DataFrame
        df = pd.DataFrame(positions.values())
        
        # Chuyển đổi các cột thời gian
        for col in ['entry_time', 'timestamp']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Tính thời gian đã mở
        if 'entry_time' in df.columns:
            df['duration'] = datetime.now() - df['entry_time']
            df['duration_hours'] = df['duration'].dt.total_seconds() / 3600
        
        # Tính lãi/lỗ hiện tại
        if 'current_price' in df.columns and 'entry_price' in df.columns and 'quantity' in df.columns:
            df['unrealized_pnl'] = df.apply(
                lambda x: (x['current_price'] - x['entry_price']) * x['quantity'] 
                if x['side'].lower() == 'long' else 
                (x['entry_price'] - x['current_price']) * x['quantity'],
                axis=1
            )
            
            df['unrealized_pnl_pct'] = df.apply(
                lambda x: ((x['current_price'] - x['entry_price']) / x['entry_price']) * 100
                if x['side'].lower() == 'long' else
                ((x['entry_price'] - x['current_price']) / x['entry_price']) * 100,
                axis=1
            )
        
        return df
    
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu vị thế đang mở: {e}")
        return pd.DataFrame()

def load_available_strategies() -> List[str]:
    """
    Tải danh sách các chiến lược đang triển khai
    
    Returns:
        Danh sách tên các chiến lược
    """
    try:
        # Xác định đường dẫn thư mục chứa thông tin triển khai
        deployment_dir = os.path.join(
            SYSTEM_CONFIG.get("base_dir", "./"), 
            "deployment", 
            "active_strategies"
        )
        
        if not os.path.exists(deployment_dir):
            logger.warning(f"Không tìm thấy thư mục triển khai: {deployment_dir}")
            
            # Thử tìm trong thư mục backtest
            backtest_dir = os.path.join(
                SYSTEM_CONFIG.get("base_dir", "./"),
                "backtesting"
            )
            
            if os.path.exists(backtest_dir):
                strategies = [d for d in os.listdir(backtest_dir) 
                             if os.path.isdir(os.path.join(backtest_dir, d)) 
                             and not d.startswith("__")]
                return strategies
            
            return []
        
        # Lấy danh sách các thư mục con (mỗi thư mục là một chiến lược)
        strategies = [d for d in os.listdir(deployment_dir) 
                     if os.path.isdir(os.path.join(deployment_dir, d))
                     and not d.startswith("__")]
        
        return strategies
    
    except Exception as e:
        logger.error(f"Lỗi khi tải danh sách chiến lược: {e}")
        return []

def load_available_symbols(strategy_name: str = None) -> List[str]:
    """
    Tải danh sách các cặp tiền đang được giao dịch
    
    Args:
        strategy_name: Tên chiến lược (nếu cần lọc theo chiến lược)
        
    Returns:
        Danh sách cặp tiền
    """
    try:
        if strategy_name is None:
            # Lấy danh sách từ cấu hình hệ thống
            symbols = SYSTEM_CONFIG.get("trading_symbols", ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        else:
            # Đọc từ file cấu hình chiến lược
            config_path = os.path.join(
                SYSTEM_CONFIG.get("base_dir", "./"),
                "deployment",
                "active_strategies",
                strategy_name,
                "config.json"
            )
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                symbols = config.get("symbols", ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
            else:
                # Fallback to default symbols
                symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        return symbols
    except Exception as e:
        logger.error(f"Lỗi khi tải danh sách cặp tiền: {e}")
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

def calculate_performance_metrics(trades_df: pd.DataFrame, balance_df: pd.DataFrame) -> Dict[str, float]:
    """
    Tính toán các chỉ số hiệu suất
    
    Args:
        trades_df: DataFrame chứa dữ liệu giao dịch
        balance_df: DataFrame chứa dữ liệu số dư
        
    Returns:
        Dict chứa các chỉ số hiệu suất
    """
    metrics = {}
    
    try:
        # Nếu không có dữ liệu, trả về metrics mặc định
        if trades_df.empty or balance_df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'max_drawdown_percent': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'avg_trade': 0,
                'roi': 0,
                'annualized_return': 0
            }
        
        # Tính tổng số giao dịch
        total_trades = len(trades_df)
        metrics['total_trades'] = total_trades
        
        # Tính số giao dịch thắng/thua
        if 'profit' in trades_df.columns:
            profit_col = 'profit'
        elif 'pnl' in trades_df.columns:
            profit_col = 'pnl'
        else:
            # Tìm cột chứa thông tin lợi nhuận
            profit_cols = [col for col in trades_df.columns if 'profit' in col.lower() or 'pnl' in col.lower()]
            profit_col = profit_cols[0] if profit_cols else None
        
        if profit_col is not None:
            # Tính tỷ lệ thắng
            win_trades = trades_df[trades_df[profit_col] > 0]
            loss_trades = trades_df[trades_df[profit_col] <= 0]
            
            winning_trades = len(win_trades)
            losing_trades = len(loss_trades)
            
            metrics['winning_trades'] = winning_trades
            metrics['losing_trades'] = losing_trades
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            metrics['win_rate'] = win_rate
            
            # Tính lợi nhuận trung bình
            if winning_trades > 0:
                avg_profit = win_trades[profit_col].mean()
                metrics['avg_profit'] = avg_profit
            else:
                metrics['avg_profit'] = 0
            
            # Tính lỗ trung bình
            if losing_trades > 0:
                avg_loss = abs(loss_trades[profit_col].mean())
                metrics['avg_loss'] = avg_loss
            else:
                metrics['avg_loss'] = 0
            
            # Tính profit factor
            total_profit = win_trades[profit_col].sum()
            total_loss = abs(loss_trades[profit_col].sum())
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            metrics['profit_factor'] = profit_factor
            
            # Tính trung bình mỗi giao dịch
            avg_trade = trades_df[profit_col].mean()
            metrics['avg_trade'] = avg_trade
        
        # Tính ROI
        if 'capital' in balance_df.columns:
            first_balance = balance_df['capital'].iloc[0]
            last_balance = balance_df['capital'].iloc[-1]
            
            roi = (last_balance / first_balance) - 1
            metrics['roi'] = roi
            
            # Tính thời gian (ngày)
            if 'timestamp' in balance_df.columns:
                start_date = balance_df['timestamp'].iloc[0]
                end_date = balance_df['timestamp'].iloc[-1]
                days = (end_date - start_date).days
                
                # Tính annualized return
                if days > 0:
                    annualized_return = ((1 + roi) ** (365 / days)) - 1
                    metrics['annualized_return'] = annualized_return
                else:
                    metrics['annualized_return'] = 0
        
        # Tính drawdown
        if 'capital' in balance_df.columns:
            equity = balance_df['capital'].values
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            
            max_drawdown = drawdown.max()
            max_drawdown_amount = (peak - equity).max()
            
            metrics['max_drawdown'] = max_drawdown_amount
            metrics['max_drawdown_percent'] = max_drawdown
        
        # Tính Sharpe và Sortino nếu có ít nhất 2 điểm dữ liệu
        if 'capital' in balance_df.columns and len(balance_df) > 1:
            # Tính returns
            returns = balance_df['capital'].pct_change().dropna()
            
            # Sharpe Ratio (annualized)
            risk_free_rate = 0.02 / 252  # 2% hàng năm
            if len(returns) > 0 and returns.std() > 0:
                sharpe = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
                metrics['sharpe_ratio'] = sharpe
            else:
                metrics['sharpe_ratio'] = 0
            
            # Sortino Ratio (annualized)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                sortino = (returns.mean() - risk_free_rate) / negative_returns.std() * np.sqrt(252)
                metrics['sortino_ratio'] = sortino
            else:
                metrics['sortino_ratio'] = 0
        
        return metrics
    
    except Exception as e:
        logger.error(f"Lỗi khi tính toán metrics hiệu suất: {e}")
        return metrics

def plot_equity_curve(balance_df: pd.DataFrame, trades_df: pd.DataFrame = None) -> go.Figure:
    """
    Tạo biểu đồ đường cong vốn (equity curve)
    
    Args:
        balance_df: DataFrame chứa dữ liệu số dư
        trades_df: DataFrame chứa dữ liệu giao dịch (tùy chọn)
        
    Returns:
        Đối tượng biểu đồ Plotly
    """
    try:
        # Kiểm tra dữ liệu
        if balance_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Không có dữ liệu để hiển thị",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Tạo biểu đồ
        fig = go.Figure()
        
        # Thêm đường cong vốn
        if 'timestamp' in balance_df.columns and 'capital' in balance_df.columns:
            fig.add_trace(go.Scatter(
                x=balance_df['timestamp'],
                y=balance_df['capital'],
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ))
        
        # Thêm điểm giao dịch nếu có
        if trades_df is not None and not trades_df.empty:
            # Tìm cột thời gian và lợi nhuận
            time_col = None
            profit_col = None
            
            for col in trades_df.columns:
                if 'exit' in col.lower() and 'time' in col.lower():
                    time_col = col
                    break
            
            if time_col is None and 'timestamp' in trades_df.columns:
                time_col = 'timestamp'
            
            for col in trades_df.columns:
                if 'profit' in col.lower() or 'pnl' in col.lower():
                    profit_col = col
                    break
            
            if time_col is not None and profit_col is not None:
                # Lọc giao dịch thắng
                win_trades = trades_df[trades_df[profit_col] > 0]
                # Lọc giao dịch thua
                loss_trades = trades_df[trades_df[profit_col] <= 0]
                
                # Với mỗi giao dịch, tìm điểm tương ứng trên equity curve
                if not win_trades.empty:
                    win_equity = []
                    for idx, trade in win_trades.iterrows():
                        trade_time = trade[time_col]
                        # Tìm điểm gần nhất trên equity curve
                        closest_idx = balance_df[balance_df['timestamp'] >= trade_time].index[0] if len(balance_df[balance_df['timestamp'] >= trade_time]) > 0 else balance_df.index[-1]
                        win_equity.append(balance_df.loc[closest_idx, 'capital'])
                    
                    fig.add_trace(go.Scatter(
                        x=win_trades[time_col],
                        y=win_equity,
                        mode='markers',
                        name='Giao dịch thắng',
                        marker=dict(color='green', size=8, symbol='triangle-up')
                    ))
                
                if not loss_trades.empty:
                    loss_equity = []
                    for idx, trade in loss_trades.iterrows():
                        trade_time = trade[time_col]
                        # Tìm điểm gần nhất trên equity curve
                        closest_idx = balance_df[balance_df['timestamp'] >= trade_time].index[0] if len(balance_df[balance_df['timestamp'] >= trade_time]) > 0 else balance_df.index[-1]
                        loss_equity.append(balance_df.loc[closest_idx, 'capital'])
                    
                    fig.add_trace(go.Scatter(
                        x=loss_trades[time_col],
                        y=loss_equity,
                        mode='markers',
                        name='Giao dịch thua',
                        marker=dict(color='red', size=8, symbol='triangle-down')
                    ))
        
        # Cập nhật layout
        fig.update_layout(
            title='Đường cong vốn',
            xaxis_title='Thời gian',
            yaxis_title='Vốn',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Thêm đường tham chiếu tại vốn ban đầu
        if not balance_df.empty and 'capital' in balance_df.columns:
            initial_capital = balance_df['capital'].iloc[0]
            fig.add_shape(
                type="line",
                x0=balance_df['timestamp'].iloc[0],
                y0=initial_capital,
                x1=balance_df['timestamp'].iloc[-1],
                y1=initial_capital,
                line=dict(color="gray", width=1, dash="dash")
            )
        
        return fig
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ đường cong vốn: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Lỗi khi tạo biểu đồ: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def plot_drawdown_chart(balance_df: pd.DataFrame) -> go.Figure:
    """
    Tạo biểu đồ drawdown
    
    Args:
        balance_df: DataFrame chứa dữ liệu số dư
        
    Returns:
        Đối tượng biểu đồ Plotly
    """
    try:
        # Kiểm tra dữ liệu
        if balance_df.empty or 'capital' not in balance_df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Không có dữ liệu để hiển thị",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Tính drawdown
        equity = balance_df['capital'].values
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100  # Đổi thành phần trăm
        
        # Tạo DataFrame mới chứa drawdown
        dd_df = pd.DataFrame({
            'timestamp': balance_df['timestamp'],
            'drawdown': drawdown
        })
        
        # Tạo biểu đồ
        fig = go.Figure()
        
        # Thêm biểu đồ drawdown
        fig.add_trace(go.Scatter(
            x=dd_df['timestamp'],
            y=dd_df['drawdown'],
            fill='tozeroy',
            mode='lines',
            line=dict(color='red', width=2),
            name='Drawdown',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        # Thêm các ngưỡng tham chiếu
        fig.add_shape(
            type="line",
            x0=dd_df['timestamp'].iloc[0],
            y0=-5,
            x1=dd_df['timestamp'].iloc[-1],
            y1=-5,
            line=dict(color="orange", width=1, dash="dash"),
            name="5% Drawdown"
        )
        
        fig.add_shape(
            type="line",
            x0=dd_df['timestamp'].iloc[0],
            y0=-10,
            x1=dd_df['timestamp'].iloc[-1],
            y1=-10,
            line=dict(color="red", width=1, dash="dash"),
            name="10% Drawdown"
        )
        
        # Đánh dấu drawdown tối đa
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown[max_dd_idx]
        max_dd_time = dd_df['timestamp'].iloc[max_dd_idx]
        
        fig.add_trace(go.Scatter(
            x=[max_dd_time],
            y=[max_dd],
            mode='markers+text',
            marker=dict(color='black', size=10),
            text=[f"{max_dd:.1f}%"],
            textposition="bottom center",
            name=f"Max Drawdown: {max_dd:.1f}%"
        ))
        
        # Cập nhật layout
        fig.update_layout(
            title='Biểu đồ Drawdown',
            xaxis_title='Thời gian',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            yaxis=dict(range=[min(drawdown) * 1.1, 0.5])  # Đảm bảo đủ không gian hiển thị
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ drawdown: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Lỗi khi tạo biểu đồ: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def plot_trade_distribution(trades_df: pd.DataFrame) -> go.Figure:
    """
    Tạo biểu đồ phân phối lợi nhuận giao dịch
    
    Args:
        trades_df: DataFrame chứa dữ liệu giao dịch
        
    Returns:
        Đối tượng biểu đồ Plotly
    """
    try:
        # Kiểm tra dữ liệu
        if trades_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Không có dữ liệu để hiển thị",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Tìm cột lợi nhuận
        profit_col = None
        for col in trades_df.columns:
            if 'profit' in col.lower() or 'pnl' in col.lower():
                profit_col = col
                break
        
        if profit_col is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Không tìm thấy cột lợi nhuận trong dữ liệu",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Tạo biểu đồ phân phối
        fig = go.Figure()
        
        # Thêm histogram
        fig.add_trace(go.Histogram(
            x=trades_df[profit_col],
            nbinsx=30,
            marker_color=trades_df[profit_col].apply(
                lambda x: 'green' if x > 0 else 'red'
            ),
            name="Phân phối lợi nhuận",
            opacity=0.7
        ))
        
        # Thêm đường tham chiếu 0
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="black", width=2, dash="dash")
        )
        
        # Tính toán thống kê
        mean_profit = trades_df[profit_col].mean()
        median_profit = trades_df[profit_col].median()
        
        # Thêm đường trung bình
        fig.add_shape(
            type="line",
            x0=mean_profit,
            y0=0,
            x1=mean_profit,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="blue", width=2, dash="solid"),
            name="Trung bình"
        )
        
        # Thêm annotation cho trung bình
        fig.add_annotation(
            x=mean_profit,
            y=0.95,
            xref="x",
            yref="paper",
            text=f"Trung bình: {mean_profit:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=40,
            ay=-40
        )
        
        # Cập nhật layout
        fig.update_layout(
            title='Phân phối lợi nhuận giao dịch',
            xaxis_title='Lợi nhuận',
            yaxis_title='Số lượng giao dịch',
            bargap=0.05,
            showlegend=False
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ phân phối lợi nhuận: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Lỗi khi tạo biểu đồ: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def plot_monthly_returns(balance_df: pd.DataFrame) -> go.Figure:
    """
    Tạo biểu đồ heatmap lợi nhuận theo tháng
    
    Args:
        balance_df: DataFrame chứa dữ liệu số dư
        
    Returns:
        Đối tượng biểu đồ Plotly
    """
    try:
        # Kiểm tra dữ liệu
        if balance_df.empty or 'capital' not in balance_df.columns or 'timestamp' not in balance_df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Không có dữ liệu để hiển thị",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Chuyển index thành datetime
        df = balance_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Chuyển thành giá trị cuối ngày
        daily_balance = df['capital'].resample('D').last().dropna()
        
        # Tính lợi nhuận hàng ngày
        daily_returns = daily_balance.pct_change().dropna()
        
        # Tính lợi nhuận theo tháng
        monthly_returns = daily_returns.groupby([
            lambda x: x.year,
            lambda x: x.month
        ]).apply(lambda x: (1 + x).prod() - 1)
        
        # Reshape dữ liệu để tạo heatmap
        monthly_data = []
        
        for (year, month), value in monthly_returns.items():
            monthly_data.append({
                'Year': year,
                'Month': month,
                'Return': value * 100  # Đổi thành phần trăm
            })
        
        monthly_df = pd.DataFrame(monthly_data)
        
        # Tạo pivot table
        if not monthly_df.empty:
            pivot_df = monthly_df.pivot("Year", "Month", "Return")
            
            # Thay tên cột thành tên tháng
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot_df.columns = [month_names[i-1] for i in pivot_df.columns]
            
            # Tạo biểu đồ heatmap
            fig = go.Figure(data=go.Heatmap(
                z=pivot_df.values,
                x=pivot_df.columns,
                y=pivot_df.index,
                colorscale='RdYlGn',
                zmid=0,
                text=[[f"{val:.2f}%" for val in row] for row in pivot_df.values],
                texttemplate="%{text}",
                textfont={"size": 11},
                hovertemplate='Năm: %{y}<br>Tháng: %{x}<br>Lợi nhuận: %{z:.2f}%<extra></extra>'
            ))
            
            # Cập nhật layout
            fig.update_layout(
                title='Lợi nhuận theo tháng (%)',
                xaxis_title='Tháng',
                yaxis_title='Năm',
                height=400,
                margin=dict(l=30, r=30, t=50, b=30)
            )
        else:
            # Không đủ dữ liệu, tạo biểu đồ trống
            fig = go.Figure()
            fig.add_annotation(
                text="Không đủ dữ liệu để hiển thị lợi nhuận theo tháng",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return fig
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ lợi nhuận theo tháng: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Lỗi khi tạo biểu đồ: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def display_risk_metrics(trades_df: pd.DataFrame, balance_df: pd.DataFrame):
    """
    Hiển thị các chỉ số rủi ro
    
    Args:
        trades_df: DataFrame chứa dữ liệu giao dịch
        balance_df: DataFrame chứa dữ liệu số dư
    """
    st.subheader("Đánh giá rủi ro")
    
    if trades_df.empty or balance_df.empty:
        st.warning("Không đủ dữ liệu để đánh giá rủi ro")
        return
    
    try:
        # Tính các chỉ số về rủi ro
        
        # 1. Tính drawdown
        if 'capital' in balance_df.columns:
            equity = balance_df['capital'].values
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak
            max_drawdown = drawdown.max() * 100  # Đổi thành phần trăm
        else:
            max_drawdown = 0
        
        # 2. Tính biến động (volatility)
        if 'capital' in balance_df.columns and len(balance_df) > 1:
            returns = balance_df['capital'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized và đổi thành phần trăm
        else:
            volatility = 0
        
        # 3. Tính Value at Risk (VaR)
        if 'capital' in balance_df.columns and len(balance_df) > 10:
            returns = balance_df['capital'].pct_change().dropna()
            var_95 = abs(np.percentile(returns, 5)) * 100  # VaR 95% confidence level, đổi thành phần trăm
            var_99 = abs(np.percentile(returns, 1)) * 100  # VaR 99% confidence level, đổi thành phần trăm
        else:
            var_95 = 0
            var_99 = 0
        
        # 4. Tính Conditional Value at Risk (CVaR) / Expected Shortfall
        if 'capital' in balance_df.columns and len(balance_df) > 10:
            returns = balance_df['capital'].pct_change().dropna()
            cvar_95 = abs(returns[returns < np.percentile(returns, 5)].mean()) * 100  # đổi thành phần trăm
        else:
            cvar_95 = 0
        
        # 5. Tính tỷ lệ Margin-to-Equity
        # Giả lập vì không có dữ liệu thực tế
        margin_to_equity = 25.0  # %
        
        # 6. Tính tỷ lệ Risk of Ruin
        # Giả lập công thức đơn giản R = (1-W/L)^N
        # W = tỷ lệ thắng, L = 1-W, N = số lần rủi ro 2R
        risk_to_reward = 1.5  # Tỷ lệ rủi ro/phần thưởng trung bình
        win_rate = len(trades_df[trades_df['profit'] > 0]) / len(trades_df) if 'profit' in trades_df.columns and len(trades_df) > 0 else 0.5
        risk_of_ruin = (1 - (win_rate * risk_to_reward) / (1 - win_rate)) ** 20 * 100
        risk_of_ruin = max(0, min(100, risk_of_ruin))  # Giới hạn trong khoảng 0-100%
        
        # Hiển thị các chỉ số rủi ro
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            st.metric("Biến động (Volatility)", f"{volatility:.2f}%")
            
        with col2:
            st.metric("Value at Risk (95%)", f"{var_95:.2f}%")
            st.metric("Expected Shortfall", f"{cvar_95:.2f}%")
            
        with col3:
            st.metric("Margin-to-Equity", f"{margin_to_equity:.2f}%")
            st.metric("Risk of Ruin", f"{risk_of_ruin:.2f}%")
        
        # Thêm biểu đồ radar cho chỉ số rủi ro
        st.subheader("Chỉ số rủi ro")
        
        # Chuẩn hóa các chỉ số về thang điểm 0-10
        normalized_risk = {
            'Drawdown': min(10, max_drawdown / 3),  # 30% drawdown = 10 điểm
            'Volatility': min(10, volatility / 5),  # 50% volatility = 10 điểm
            'VaR': min(10, var_95 / 3),             # 30% VaR = 10 điểm
            'Margin Usage': min(10, margin_to_equity / 10),  # 100% margin = 10 điểm
            'Risk of Ruin': min(10, risk_of_ruin / 10)  # 100% RoR = 10 điểm
        }
        
        # Chuẩn bị dữ liệu cho biểu đồ radar
        categories = list(normalized_risk.keys())
        values = list(normalized_risk.values())
        
        # Thêm điểm đầu vào cuối để tạo biểu đồ radar kín
        categories.append(categories[0])
        values.append(values[0])
        
        # Tạo biểu đồ radar
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Risk Score',
            line_color='red',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Lỗi khi tính toán chỉ số rủi ro: {e}")
        logger.error(f"Lỗi khi tính toán chỉ số rủi ro: {e}")

def main():
    """
    Hàm chính cho dashboard giao dịch
    """
    st.set_page_config(
        page_title="Trading Dashboard - Automated Trading System",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Dashboard Giao Dịch")
    
    # Tạo sidebar
    create_sidebar()
    create_account_section()
    
    # Tải danh sách chiến lược
    strategies = load_available_strategies()
    
    if not strategies:
        st.warning("Không tìm thấy chiến lược nào đang hoạt động.")
        st.info("Vui lòng triển khai ít nhất một chiến lược trước khi sử dụng dashboard.")
        return
    
    # Tạo các bộ lọc
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_strategy = st.selectbox(
            "Chọn chiến lược",
            options=strategies
        )
    
    # Tải danh sách cặp tiền
    symbols = load_available_symbols(selected_strategy)
    
    with col2:
        selected_symbol = st.selectbox(
            "Chọn cặp tiền",
            options=["Tất cả"] + symbols
        )
        
        # Chuyển đổi tên cặp
        symbol_filter = selected_symbol if selected_symbol != "Tất cả" else "all"
    
    with col3:
        time_options = {
            "24 giờ qua": 1,
            "7 ngày gần đây": 7,
            "30 ngày gần đây": 30,
            "90 ngày gần đây": 90,
            "Tất cả": 365
        }
        
        selected_time = st.selectbox(
            "Khoảng thời gian",
            options=list(time_options.keys())
        )
        
        time_days = time_options[selected_time]
    
    # Tải dữ liệu
    trades_data = load_trading_data(selected_strategy, symbol_filter, time_days)
    balance_data = load_balance_history(selected_strategy, time_days)
    
    # Kiểm tra dữ liệu
    if trades_data.empty and balance_data.empty:
        st.warning(f"Không có dữ liệu giao dịch cho chiến lược {selected_strategy} trong khoảng thời gian đã chọn.")
        return
    
    # Tính toán các chỉ số hiệu suất
    metrics = calculate_performance_metrics(trades_data, balance_data)
    
    # Hiển thị overview
    st.subheader("Tổng quan hiệu suất")
    
    # Hiển thị các chỉ số chính
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    
    with overview_col1:
        roi = metrics.get('roi', 0) * 100
        roi_color = 'normal' if roi == 0 else ('off' if roi < 0 else 'normal')
        st.metric("Tổng lợi nhuận", f"{roi:.2f}%", delta=None, delta_color=roi_color)
    
    with overview_col2:
        win_rate = metrics.get('win_rate', 0) * 100
        st.metric("Tỷ lệ thắng", f"{win_rate:.2f}%")
    
    with overview_col3:
        profit_factor = metrics.get('profit_factor', 0)
        profit_factor_display = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
        st.metric("Profit Factor", profit_factor_display)
    
    with overview_col4:
        total_trades = metrics.get('total_trades', 0)
        winning_trades = metrics.get('winning_trades', 0)
        losing_trades = metrics.get('losing_trades', 0)
        st.metric("Tổng giao dịch", f"{total_trades}", f"{winning_trades} thắng / {losing_trades} thua")
    
    # Hiển thị biểu đồ đường cong vốn
    st.subheader("Đường cong vốn")
    equity_fig = plot_equity_curve(balance_data, trades_data)
    st.plotly_chart(equity_fig, use_container_width=True)
    
    # Hiển thị biểu đồ drawdown
    drawdown_fig = plot_drawdown_chart(balance_data)
    st.plotly_chart(drawdown_fig, use_container_width=True)
    
    # Hiển thị chỉ số chi tiết
    with st.expander("Xem thêm chỉ số chi tiết"):
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
            st.metric("Lợi nhuận hàng năm", f"{metrics.get('annualized_return', 0) * 100:.2f}%")
            
        with detail_col2:
            st.metric("Avg Profit", f"{metrics.get('avg_profit', 0):.2f}")
            st.metric("Avg Loss", f"{metrics.get('avg_loss', 0):.2f}")
            st.metric("Avg Trade", f"{metrics.get('avg_trade', 0):.2f}")
            
        with detail_col3:
            avg_profit = metrics.get('avg_profit', 0)
            avg_loss = metrics.get('avg_loss', 0)
            if avg_loss != 0:
                risk_reward = avg_profit / avg_loss
            else:
                risk_reward = float('inf')
            
            risk_reward_display = f"{risk_reward:.2f}" if risk_reward != float('inf') else "∞"
            
            st.metric("Risk-Reward Ratio", risk_reward_display)
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown_percent', 0) * 100:.2f}%")
            st.metric("Recovery Factor", f"{metrics.get('recovery_factor', 0):.2f}")
    
    # Tạo tab cho các phân tích khác nhau
    tabs = st.tabs(["Phân tích giao dịch", "Phân tích thời gian", "Quản lý rủi ro", "Vị thế đang mở"])
    
    # Tab 1: Phân tích giao dịch
    with tabs[0]:
        if not trades_data.empty:
            # Hiển thị biểu đồ phân phối lợi nhuận
            st.subheader("Phân phối lợi nhuận giao dịch")
            dist_fig = plot_trade_distribution(trades_data)
            st.plotly_chart(dist_fig, use_container_width=True)
            
            # Hiển thị thống kê giao dịch
            st.subheader("Thống kê giao dịch")
            
            if 'symbol' in trades_data.columns:
                # Phân tích theo cặp tiền
                symbol_stats = {}
                
                for symbol in trades_data['symbol'].unique():
                    symbol_data = trades_data[trades_data['symbol'] == symbol]
                    
                    # Tìm cột lợi nhuận
                    profit_col = None
                    for col in symbol_data.columns:
                        if 'profit' in col.lower() or 'pnl' in col.lower():
                            profit_col = col
                            break
                    
                    if profit_col is not None:
                        win_trades = len(symbol_data[symbol_data[profit_col] > 0])
                        loss_trades = len(symbol_data[symbol_data[profit_col] <= 0])
                        total_trades = len(symbol_data)
                        
                        win_rate = win_trades / total_trades if total_trades > 0 else 0
                        avg_profit = symbol_data[symbol_data[profit_col] > 0][profit_col].mean() if win_trades > 0 else 0
                        avg_loss = abs(symbol_data[symbol_data[profit_col] <= 0][profit_col].mean()) if loss_trades > 0 else 0
                        
                        total_profit = symbol_data[profit_col].sum()
                        profit_factor = (symbol_data[symbol_data[profit_col] > 0][profit_col].sum() / 
                                        abs(symbol_data[symbol_data[profit_col] <= 0][profit_col].sum())) if loss_trades > 0 else float('inf')
                        
                        symbol_stats[symbol] = {
                            'total_trades': total_trades,
                            'win_rate': win_rate,
                            'avg_profit': avg_profit,
                            'avg_loss': avg_loss,
                            'total_profit': total_profit,
                            'profit_factor': profit_factor
                        }
                
                # Tạo DataFrame từ thống kê
                symbol_stats_df = pd.DataFrame.from_dict(symbol_stats, orient='index')
                
                # Thêm cột phần trăm
                symbol_stats_df['win_rate_pct'] = symbol_stats_df['win_rate'] * 100
                
                # Định dạng lại DataFrame để hiển thị
                display_df = symbol_stats_df.copy()
                display_df.reset_index(inplace=True)
                display_df.rename(columns={
                    'index': 'Cặp tiền',
                    'total_trades': 'Tổng giao dịch',
                    'win_rate_pct': 'Tỷ lệ thắng (%)',
                    'avg_profit': 'Lợi nhuận TB',
                    'avg_loss': 'Lỗ TB',
                    'total_profit': 'Tổng lợi nhuận',
                    'profit_factor': 'Profit Factor'
                }, inplace=True)
                
                # Định dạng các cột số
                display_df['Tỷ lệ thắng (%)'] = display_df['Tỷ lệ thắng (%)'].round(2)
                display_df['Lợi nhuận TB'] = display_df['Lợi nhuận TB'].round(2)
                display_df['Lỗ TB'] = display_df['Lỗ TB'].round(2)
                display_df['Tổng lợi nhuận'] = display_df['Tổng lợi nhuận'].round(2)
                display_df['Profit Factor'] = display_df['Profit Factor'].apply(
                    lambda x: round(x, 2) if x != float('inf') else "∞"
                )
                
                # Hiển thị DataFrame
                st.dataframe(display_df, use_container_width=True)
                
                # Biểu đồ so sánh win rate và profit factor
                if len(symbol_stats) > 1:
                    st.subheader("So sánh cặp tiền")
                    
                    comp_fig = go.Figure()
                    
                    # Thêm bar chart cho win rate
                    comp_fig.add_trace(go.Bar(
                        x=list(symbol_stats.keys()),
                        y=[stats['win_rate'] * 100 for stats in symbol_stats.values()],
                        name='Tỷ lệ thắng (%)',
                        marker_color='green',
                        opacity=0.7
                    ))
                    
                    # Thêm bar chart cho profit factor
                    comp_fig.add_trace(go.Bar(
                        x=list(symbol_stats.keys()),
                        y=[min(stats['profit_factor'], 5) for stats in symbol_stats.values()],
                        name='Profit Factor',
                        marker_color='blue',
                        opacity=0.7
                    ))
                    
                    # Thêm line chart cho tổng lợi nhuận
                    comp_fig.add_trace(go.Scatter(
                        x=list(symbol_stats.keys()),
                        y=[stats['total_profit'] for stats in symbol_stats.values()],
                        name='Tổng lợi nhuận',
                        mode='lines+markers',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Cập nhật layout
                    comp_fig.update_layout(
                        title='So sánh hiệu suất các cặp tiền',
                        xaxis_title='Cặp tiền',
                        yaxis_title='Giá trị',
                        barmode='group',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(comp_fig, use_container_width=True)
            
            # Hiển thị lịch sử giao dịch
            st.subheader("Lịch sử giao dịch")
            
            # Chọn các cột để hiển thị
            display_cols = []
            rename_map = {}
            
            # Tìm cột lợi nhuận
            profit_col = None
            for col in trades_data.columns:
                if 'profit' in col.lower() or 'pnl' in col.lower():
                    profit_col = col
                    display_cols.append(profit_col)
                    rename_map[profit_col] = 'Lợi nhuận'
                    break
            
            # Tìm các cột thời gian
            entry_time_col = None
            exit_time_col = None
            
            for col in trades_data.columns:
                if 'entry' in col.lower() and 'time' in col.lower():
                    entry_time_col = col
                    display_cols.append(entry_time_col)
                    rename_map[entry_time_col] = 'Thời gian mở'
                
                if 'exit' in col.lower() and 'time' in col.lower():
                    exit_time_col = col
                    display_cols.append(exit_time_col)
                    rename_map[exit_time_col] = 'Thời gian đóng'
            
            # Thêm các cột khác
            if 'symbol' in trades_data.columns:
                display_cols.append('symbol')
                rename_map['symbol'] = 'Cặp tiền'
            
            if 'side' in trades_data.columns:
                display_cols.append('side')
                rename_map['side'] = 'Hướng'
            
            if 'exit_price' in trades_data.columns:
                display_cols.append('exit_price')
                rename_map['exit_price'] = 'Giá đóng'
            
            if 'quantity' in trades_data.columns:
                display_cols.append('quantity')
                rename_map['quantity'] = 'Khối lượng'
            
            # Tạo bản sao và chọn các cột cần hiển thị
            trades_display = trades_data[display_cols].copy()
            
            # Đổi tên cột
            trades_display.rename(columns=rename_map, inplace=True)
            
            # Sắp xếp theo thời gian giao dịch (mới nhất lên đầu)
            if exit_time_col and exit_time_col in trades_data.columns:
                trades_display = trades_display.sort_values(rename_map[exit_time_col], ascending=False)
            elif entry_time_col and entry_time_col in trades_data.columns:
                trades_display = trades_display.sort_values(rename_map[entry_time_col], ascending=False)
            
            # Thêm tính toán lợi nhuận phần trăm nếu có đủ dữ liệu
            if 'entry_price' in trades_data.columns and 'exit_price' in trades_data.columns:
                if 'side' in trades_data.columns:
                    trades_display['Lợi nhuận (%)'] = trades_data.apply(
                        lambda x: ((x['exit_price'] - x['entry_price']) / x['entry_price'] * 100) if x['side'].lower() == 'long'
                        else ((x['entry_price'] - x['exit_price']) / x['entry_price'] * 100),
                        axis=1
                    ).round(2)
                else:
                    trades_display['Lợi nhuận (%)'] = ((trades_data['exit_price'] - trades_data['entry_price']) / trades_data['entry_price'] * 100).round(2)
            
            # Hiển thị DataFrame
            st.dataframe(trades_display, use_container_width=True)
            
        else:
            st.info("Không có dữ liệu giao dịch để hiển thị.")
    
    # Tab 2: Phân tích thời gian
    with tabs[1]:
        if not balance_data.empty:
            # Hiển thị biểu đồ lợi nhuận theo tháng
            st.subheader("Lợi nhuận theo tháng")
            monthly_fig = plot_monthly_returns(balance_data)
            st.plotly_chart(monthly_fig, use_container_width=True)
            
            # Thêm phân tích theo ngày trong tuần nếu có đủ dữ liệu
            if 'timestamp' in balance_data.columns and 'capital' in balance_data.columns and len(balance_data) > 7:
                st.subheader("Hiệu suất theo ngày trong tuần")
                
                # Chuẩn bị dữ liệu
                df = balance_data.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Tính lợi nhuận hàng ngày
                daily_balance = df['capital'].resample('D').last().dropna()
                daily_returns = daily_balance.pct_change().dropna()
                
                # Thêm ngày trong tuần
                daily_returns = pd.DataFrame(daily_returns)
                daily_returns['day_of_week'] = daily_returns.index.dayofweek
                
                # Chuyển đổi số thành tên ngày
                day_names = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
                daily_returns['day_name'] = daily_returns['day_of_week'].apply(lambda x: day_names[x])
                
                # Tính lợi nhuận trung bình theo ngày
                day_performance = daily_returns.groupby('day_name')[0].agg(['mean', 'std', 'count'])
                day_performance['mean'] = day_performance['mean'] * 100  # Đổi thành phần trăm
                day_performance['std'] = day_performance['std'] * 100    # Đổi thành phần trăm
                
                # Sắp xếp theo thứ tự ngày trong tuần
                day_performance = day_performance.reindex(day_names)
                
                # Tạo biểu đồ
                dow_fig = go.Figure()
                
                # Thêm bar chart cho lợi nhuận trung bình
                dow_fig.add_trace(go.Bar(
                    x=day_performance.index,
                    y=day_performance['mean'],
                    error_y=dict(
                        type='data',
                        array=day_performance['std'] / np.sqrt(day_performance['count']),
                        visible=True
                    ),
                    name='Lợi nhuận trung bình',
                    marker_color=day_performance['mean'].apply(
                        lambda x: 'green' if x > 0 else 'red'
                    )
                ))
                
                # Thêm scatter cho số lượng ngày
                dow_fig.add_trace(go.Scatter(
                    x=day_performance.index,
                    y=day_performance['count'],
                    mode='markers+text',
                    marker=dict(size=day_performance['count'] / day_performance['count'].max() * 20 + 5),
                    text=day_performance['count'],
                    textposition="top center",
                    yaxis='y2',
                    name='Số ngày'
                ))
                
                # Cập nhật layout
                dow_fig.update_layout(
                    title='Hiệu suất theo ngày trong tuần',
                    xaxis_title='Ngày',
                    yaxis=dict(
                        title='Lợi nhuận trung bình (%)',
                        titlefont=dict(color='green'),
                        tickfont=dict(color='green')
                    ),
                    yaxis2=dict(
                        title='Số ngày',
                        titlefont=dict(color='blue'),
                        tickfont=dict(color='blue'),
                        anchor='x',
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(dow_fig, use_container_width=True)
            
            # Thêm phân tích theo giờ trong ngày nếu có dữ liệu giao dịch
            if not trades_data.empty:
                # Tìm cột thời gian và lợi nhuận
                time_col = None
                profit_col = None
                
                for col in trades_data.columns:
                    if ('exit' in col.lower() or 'entry' in col.lower()) and 'time' in col.lower():
                        time_col = col
                        break
                
                for col in trades_data.columns:
                    if 'profit' in col.lower() or 'pnl' in col.lower():
                        profit_col = col
                        break
                
                if time_col is not None and profit_col is not None:
                    st.subheader("Hiệu suất theo giờ trong ngày")
                    
                    # Chuẩn bị dữ liệu
                    hour_data = trades_data.copy()
                    hour_data['hour'] = pd.to_datetime(hour_data[time_col]).dt.hour
                    
                    # Tính hiệu suất theo giờ
                    hour_performance = hour_data.groupby('hour')[profit_col].agg(['mean', 'sum', 'count'])
                    
                    # Tạo biểu đồ
                    hour_fig = go.Figure()
                    
                    # Thêm bar chart cho lợi nhuận trung bình
                    hour_fig.add_trace(go.Bar(
                        x=hour_performance.index,
                        y=hour_performance['mean'],
                        name='Lợi nhuận TB/giao dịch',
                        marker_color=hour_performance['mean'].apply(
                            lambda x: 'green' if x > 0 else 'red'
                        )
                    ))
                    
                    # Thêm line chart cho tổng lợi nhuận
                    hour_fig.add_trace(go.Scatter(
                        x=hour_performance.index,
                        y=hour_performance['sum'],
                        mode='lines+markers',
                        name='Tổng lợi nhuận',
                        line=dict(color='blue', width=2),
                        yaxis='y2'
                    ))
                    
                    # Cập nhật layout
                    hour_fig.update_layout(
                        title='Hiệu suất theo giờ trong ngày',
                        xaxis=dict(
                            title='Giờ',
                            tickmode='array',
                            tickvals=list(range(24)),
                            ticktext=[f"{h}:00" for h in range(24)]
                        ),
                        yaxis=dict(
                            title='Lợi nhuận TB/giao dịch',
                            titlefont=dict(color='green'),
                            tickfont=dict(color='green')
                        ),
                        yaxis2=dict(
                            title='Tổng lợi nhuận',
                            titlefont=dict(color='blue'),
                            tickfont=dict(color='blue'),
                            anchor='x',
                            overlaying='y',
                            side='right'
                        ),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(hour_fig, use_container_width=True)
                    
                    # Thêm bảng số lượng giao dịch theo giờ
                    st.subheader("Số lượng giao dịch theo giờ")
                    
                    # Tạo heatmap cho số lượng giao dịch theo giờ và ngày trong tuần
                    if 'timestamp' in trades_data.columns or time_col is not None:
                        # Sử dụng cột thời gian đã tìm thấy
                        hour_data['day_of_week'] = pd.to_datetime(hour_data[time_col]).dt.dayofweek
                        hour_data['day_name'] = hour_data['day_of_week'].apply(lambda x: day_names[x])
                        
                        # Tạo bảng pivot
                        hour_day_pivot = pd.pivot_table(
                            hour_data, 
                            values=profit_col, 
                            index='day_name', 
                            columns='hour', 
                            aggfunc='count',
                            fill_value=0
                        )
                        
                        # Sắp xếp theo thứ tự ngày trong tuần
                        hour_day_pivot = hour_day_pivot.reindex(day_names)
                        
                        # Tạo heatmap
                        heatmap_fig = go.Figure(data=go.Heatmap(
                            z=hour_day_pivot.values,
                            x=[f"{h}:00" for h in hour_day_pivot.columns],
                            y=hour_day_pivot.index,
                            colorscale='Blues',
                            showscale=True,
                            text=hour_day_pivot.values,
                            texttemplate="%{text}",
                            textfont={"size": 10},
                            colorbar=dict(title='Số giao dịch')
                        ))
                        
                        # Cập nhật layout
                        heatmap_fig.update_layout(
                            title='Số lượng giao dịch theo giờ và ngày trong tuần',
                            xaxis_title='Giờ',
                            yaxis_title='Ngày trong tuần'
                        )
                        
                        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        else:
            st.info("Không có đủ dữ liệu để phân tích theo thời gian.")
    
    # Tab 3: Quản lý rủi ro
    with tabs[2]:
        # Hiển thị các chỉ số rủi ro
        display_risk_metrics(trades_data, balance_data)
        
        # Hiển thị bảng định cỡ vị thế và đề xuất quản lý rủi ro
        st.subheader("Đề xuất quản lý rủi ro")
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            # Tính toán và hiển thị đề xuất kích thước vị thế
            account_size = 10000  # Giả lập kích thước tài khoản
            
            risk_per_trade_pct = st.slider(
                "Rủi ro mỗi giao dịch (%)",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.1
            )
            
            st.write("##### Đề xuất kích thước vị thế")
            
            risk_per_trade = account_size * risk_per_trade_pct / 100
            
            position_sizes = pd.DataFrame({
                "Stop Loss (%)": [1.0, 2.0, 3.0, 5.0, 10.0],
                "Kích thước vị thế ($)": [
                    risk_per_trade / 0.01,
                    risk_per_trade / 0.02,
                    risk_per_trade / 0.03,
                    risk_per_trade / 0.05,
                    risk_per_trade / 0.10
                ],
                "Số cặp BTCUSDT": [
                    (risk_per_trade / 0.01) / 50000,
                    (risk_per_trade / 0.02) / 50000,
                    (risk_per_trade / 0.03) / 50000,
                    (risk_per_trade / 0.05) / 50000,
                    (risk_per_trade / 0.10) / 50000
                ],
                "Số cặp ETHUSDT": [
                    (risk_per_trade / 0.01) / 2500,
                    (risk_per_trade / 0.02) / 2500,
                    (risk_per_trade / 0.03) / 2500,
                    (risk_per_trade / 0.05) / 2500,
                    (risk_per_trade / 0.10) / 2500
                ]
            })
            
            # Định dạng lại các cột
            position_sizes["Kích thước vị thế ($)"] = position_sizes["Kích thước vị thế ($)"].round(2)
            position_sizes["Số cặp BTCUSDT"] = position_sizes["Số cặp BTCUSDT"].round(4)
            position_sizes["Số cặp ETHUSDT"] = position_sizes["Số cặp ETHUSDT"].round(4)
            
            st.dataframe(position_sizes, use_container_width=True)
            
            st.info(f"Mức rủi ro mỗi giao dịch: ${risk_per_trade:.2f}")
        
        with risk_col2:
            # Hiển thị đề xuất quản lý rủi ro dựa trên drawdown
            max_dd = metrics.get('max_drawdown_percent', 0) * 100
            
            st.write("##### Đề xuất dựa trên drawdown")
            
            if max_dd > 20:
                st.error(f"⚠️ Max drawdown hiện tại ({max_dd:.2f}%) vượt ngưỡng an toàn (20%)")
                st.markdown("""
                **Đề xuất:**
                - Giảm kích thước vị thế xuống 50%
                - Thắt chặt tiêu chí vào lệnh
                - Tăng mức stop loss
                - Tạm dừng giao dịch nếu drawdown đạt 25%
                """)
            elif max_dd > 10:
                st.warning(f"⚠️ Max drawdown hiện tại ({max_dd:.2f}%) đang ở mức cảnh báo (10-20%)")
                st.markdown("""
                **Đề xuất:**
                - Giảm kích thước vị thế xuống 75%
                - Tránh mở nhiều vị thế cùng lúc
                - Xem xét điều chỉnh tham số chiến lược
                """)
            else:
                st.success(f"✅ Max drawdown hiện tại ({max_dd:.2f}%) ở mức an toàn (< 10%)")
                st.markdown("""
                **Đề xuất:**
                - Duy trì mức rủi ro hiện tại
                - Có thể tăng kích thước vị thế nếu win rate > 60%
                """)
            
            # Đề xuất dựa trên win rate và profit factor
            win_rate = metrics.get('win_rate', 0) * 100
            profit_factor = metrics.get('profit_factor', 0)
            
            st.write("##### Đề xuất dựa trên hiệu suất")
            
            if win_rate < 40:
                st.error(f"⚠️ Win rate thấp ({win_rate:.2f}%)")
                st.markdown("""
                **Đề xuất:**
                - Xem xét lại chiến lược vào lệnh
                - Thêm bộ lọc để giảm tín hiệu giả
                - Giảm tần suất giao dịch
                """)
            elif win_rate > 60:
                st.success(f"✅ Win rate cao ({win_rate:.2f}%)")
                st.markdown("""
                **Đề xuất:**
                - Có thể tăng kích thước vị thế
                - Tối ưu hóa chiến lược chốt lời
                """)
            
            if profit_factor < 1.2:
                st.error(f"⚠️ Profit factor thấp ({profit_factor:.2f})")
                st.markdown("""
                **Đề xuất:**
                - Cải thiện tỷ lệ R:R (nắm giữ lâu hơn, chốt lời xa hơn)
                - Cắt lỗ sớm hơn
                - Tránh giao dịch trong giai đoạn biến động thấp
                """)
            elif profit_factor > 2.0:
                st.success(f"✅ Profit factor cao ({profit_factor:.2f})")
                st.markdown("""
                **Đề xuất:**
                - Tối ưu hóa chiến lược để tận dụng ưu thế
                - Xem xét tăng đòn bẩy một cách có kiểm soát
                """)
    
    # Tab 4: Vị thế đang mở
    with tabs[3]:
        st.subheader("Vị thế đang mở")
        
        # Tải các vị thế đang mở
        active_positions = load_active_positions(selected_strategy)
        
        if not active_positions.empty:
            # Hiển thị các vị thế đang mở
            st.write(f"Có {len(active_positions)} vị thế đang mở")
            
            # Chọn các cột để hiển thị
            display_cols = []
            rename_map = {}
            
            # Tìm các cột cần thiết
            for col, new_name in [
                ('symbol', 'Cặp tiền'),
                ('side', 'Hướng'),
                ('entry_price', 'Giá vào'),
                ('current_price', 'Giá hiện tại'),
                ('quantity', 'Khối lượng'),
                ('unrealized_pnl', 'Lợi nhuận'),
                ('unrealized_pnl_pct', 'Lợi nhuận (%)'),
                ('entry_time', 'Thời gian mở'),
                ('duration_hours', 'Thời gian nắm giữ (giờ)')
            ]:
                if col in active_positions.columns:
                    display_cols.append(col)
                    rename_map[col] = new_name
            
            # Tạo bản sao và chọn các cột cần hiển thị
            positions_display = active_positions[display_cols].copy()
            
            # Đổi tên cột
            positions_display.rename(columns=rename_map, inplace=True)
            
            # Định dạng lại các cột số
            for col in positions_display.columns:
                if col == 'Lợi nhuận (%)':
                    positions_display[col] = positions_display[col].round(2)
                elif col == 'Lợi nhuận':
                    positions_display[col] = positions_display[col].round(4)
                elif col == 'Thời gian nắm giữ (giờ)':
                    positions_display[col] = positions_display[col].round(1)
            
            # Hiển thị DataFrame
            st.dataframe(positions_display, use_container_width=True)
            
            # Thêm biểu đồ lợi nhuận vị thế
            if 'unrealized_pnl' in active_positions.columns and 'symbol' in active_positions.columns:
                st.subheader("Phân tích vị thế đang mở")
                
                # Tạo biểu đồ
                positions_fig = go.Figure()
                
                # Thêm bar chart cho lợi nhuận
                positions_fig.add_trace(go.Bar(
                    x=active_positions['symbol'],
                    y=active_positions['unrealized_pnl'],
                    name='Lợi nhuận',
                    marker_color=active_positions['unrealized_pnl'].apply(
                        lambda x: 'green' if x > 0 else 'red'
                    )
                ))
                
                # Cập nhật layout
                positions_fig.update_layout(
                    title='Lợi nhuận vị thế đang mở',
                    xaxis_title='Cặp tiền',
                    yaxis_title='Lợi nhuận',
                    hovermode='x unified'
                )
                
                st.plotly_chart(positions_fig, use_container_width=True)
                
                # Thêm biểu đồ pie chart cho phân bổ vị thế
                if 'quantity' in active_positions.columns and 'entry_price' in active_positions.columns:
                    # Tính giá trị vị thế
                    active_positions['position_value'] = active_positions['quantity'] * active_positions['entry_price']
                    
                    # Tạo biểu đồ pie chart
                    pie_fig = go.Figure(data=[go.Pie(
                        labels=active_positions['symbol'],
                        values=active_positions['position_value'],
                        hole=.3,
                        textinfo='label+percent',
                        marker=dict(
                            colors=px.colors.qualitative.Pastel
                        )
                    )])
                    
                    # Cập nhật layout
                    pie_fig.update_layout(
                        title='Phân bổ vốn theo cặp tiền',
                        showlegend=False
                    )
                    
                    st.plotly_chart(pie_fig, use_container_width=True)
            
            # Thêm phần quản lý vị thế
            st.subheader("Quản lý vị thế")
            
            management_col1, management_col2 = st.columns(2)
            
            with management_col1:
                st.write("##### Điều chỉnh Stop Loss")
                
                for idx, position in active_positions.iterrows():
                    symbol = position['symbol']
                    entry_price = position['entry_price']
                    current_price = position['current_price']
                    
                    # Tính các mức stop loss đề xuất
                    if position['side'].lower() == 'long':
                        sl_tight = entry_price * 0.98
                        sl_medium = entry_price * 0.95
                        sl_loose = entry_price * 0.90
                        sl_current = current_price * 0.97
                    else:  # short
                        sl_tight = entry_price * 1.02
                        sl_medium = entry_price * 1.05
                        sl_loose = entry_price * 1.10
                        sl_current = current_price * 1.03
                    
                    # Hiển thị expander cho mỗi vị thế
                    with st.expander(f"Stop Loss - {symbol}"):
                        st.write(f"Giá vào: {entry_price}")
                        st.write(f"Giá hiện tại: {current_price}")
                        st.write("---")
                        st.write("**Mức Stop Loss đề xuất:**")
                        
                        # Hiển thị các mức stop loss
                        sl_options = {
                            "Chặt (2%)": sl_tight,
                            "Trung bình (5%)": sl_medium,
                            "Rộng (10%)": sl_loose,
                            "Theo giá hiện tại (3%)": sl_current
                        }
                        
                        for name, value in sl_options.items():
                            st.write(f"{name}: {value:.2f}")
                        
                        # Nút áp dụng
                        if st.button(f"Áp dụng Stop Loss cho {symbol}"):
                            st.success(f"Đã áp dụng Stop Loss cho {symbol}")
            
            with management_col2:
                st.write("##### Điều chỉnh Take Profit")
                
                for idx, position in active_positions.iterrows():
                    symbol = position['symbol']
                    entry_price = position['entry_price']
                    current_price = position['current_price']
                    
                    # Tính các mức take profit đề xuất
                    if position['side'].lower() == 'long':
                        tp_tight = entry_price * 1.02
                        tp_medium = entry_price * 1.05
                        tp_loose = entry_price * 1.10
                        tp_current = current_price * 1.03
                    else:  # short
                        tp_tight = entry_price * 0.98
                        tp_medium = entry_price * 0.95
                        tp_loose = entry_price * 0.90
                        tp_current = current_price * 0.97
                    
                    # Hiển thị expander cho mỗi vị thế
                    with st.expander(f"Take Profit - {symbol}"):
                        st.write(f"Giá vào: {entry_price}")
                        st.write(f"Giá hiện tại: {current_price}")
                        st.write("---")
                        st.write("**Mức Take Profit đề xuất:**")
                        
                        # Hiển thị các mức take profit
                        tp_options = {
                            "Nhanh (2%)": tp_tight,
                            "Trung bình (5%)": tp_medium,
                            "Xa (10%)": tp_loose,
                            "Theo giá hiện tại (3%)": tp_current
                        }
                        
                        for name, value in tp_options.items():
                            st.write(f"{name}: {value:.2f}")
                        
                        # Nút áp dụng
                        if st.button(f"Áp dụng Take Profit cho {symbol}"):
                            st.success(f"Đã áp dụng Take Profit cho {symbol}")
            
            # Thêm nút đóng tất cả vị thế
            if st.button("Đóng tất cả vị thế", type="primary"):
                st.warning("⚠️ Bạn có chắc chắn muốn đóng tất cả vị thế?")
                
                confirm_col1, confirm_col2 = st.columns(2)
                
                with confirm_col1:
                    if st.button("Xác nhận đóng tất cả"):
                        st.success("Đã đóng tất cả vị thế!")
                
                with confirm_col2:
                    if st.button("Hủy"):
                        st.info("Đã hủy thao tác")
        
        else:
            st.info("Không có vị thế nào đang mở")
            
            # Hiển thị phần mở vị thế mới nếu không có vị thế đang mở
            st.subheader("Mở vị thế mới")
            
            # Tạo form mở vị thế
            with st.form("open_position_form"):
                form_col1, form_col2 = st.columns(2)
                
                with form_col1:
                    new_symbol = st.selectbox(
                        "Cặp tiền",
                        options=symbols
                    )
                    
                    new_side = st.radio(
                        "Hướng",
                        options=["Long", "Short"],
                        horizontal=True
                    )
                    
                    new_quantity = st.number_input(
                        "Khối lượng",
                        min_value=0.001,
                        step=0.001,
                        value=0.01
                    )
                
                with form_col2:
                    new_price = st.number_input(
                        "Giá vào",
                        min_value=0.01,
                        step=0.01,
                        value=50000.0 if new_symbol == "BTCUSDT" else 2500.0
                    )
                    
                    new_sl_pct = st.slider(
                        "Stop Loss (%)",
                        min_value=1.0,
                        max_value=10.0,
                        value=5.0,
                        step=0.5
                    )
                    
                    new_tp_pct = st.slider(
                        "Take Profit (%)",
                        min_value=1.0,
                        max_value=20.0,
                        value=10.0,
                        step=0.5
                    )
                
                # Tính giá trị vị thế
                position_value = new_price * new_quantity
                
                # Hiển thị thông tin
                st.info(f"Giá trị vị thế: ${position_value:.2f}")
                
                # Nút xác nhận
                submit_button = st.form_submit_button("Mở vị thế")
            
            if submit_button:
                st.success(f"Đã mở vị thế {new_side} {new_symbol}: {new_quantity} @ {new_price}")
                st.info("Vui lòng làm mới trang để xem vị thế mới")
    
    # Thêm phần điều khiển giao dịch
    st.subheader("Điều khiển giao dịch")
    
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        if st.button("🚀 Bật tự động giao dịch", use_container_width=True):
            st.success("Đã bật chế độ tự động giao dịch")
    
    with control_col2:
        if st.button("⏸️ Tạm dừng giao dịch", use_container_width=True):
            st.warning("Đã tạm dừng giao dịch")
    
    with control_col3:
        if st.button("⚠️ Dừng khẩn cấp", use_container_width=True):
            st.error("Đã dừng khẩn cấp và đóng tất cả vị thế!")

if __name__ == "__main__":
    main()