"""
Ứng dụng Streamlit chính cho Automated Trading System.
File này cung cấp dashboard chính để hiển thị thông tin và 
điều khiển hệ thống giao dịch tự động, tích hợp các thành phần
từ các module khác.
"""

import os
import sys
import time
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

# Thêm thư mục gốc vào đường dẫn để có thể import các module
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Import các module cần thiết từ hệ thống
from config.system_config import get_system_config, LOG_DIR
from config.env import get_env, get_env_manager
from logs.logger import get_logger, get_system_logger
from logs.metrics.training_metrics import load_training_metrics
from logs.metrics.trading_metrics import load_trading_metrics, calculate_performance_metrics
from logs.metrics.system_metrics import get_system_stats

# Import các thành phần UI
from streamlit_dashboard.components.sidebar import render_sidebar
from streamlit_dashboard.components.metrics_display import (
    render_main_metrics, 
    render_trading_performance_metrics,
    render_agent_metrics
)
from streamlit_dashboard.charts.performance_charts import (
    plot_equity_curve,
    plot_drawdown_chart,
    plot_return_distribution
)
from streamlit_dashboard.charts.risk_visualization import (
    plot_risk_metrics,
    plot_risk_heatmap
)
from streamlit_dashboard.charts.trade_visualization import (
    plot_trade_history,
    plot_trade_distribution
)

# Tạo logger
logger = get_system_logger("dashboard")

# Cấu hình trang
st.set_page_config(
    page_title="Automated Trading System Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    """Tải CSS tùy chỉnh cho dashboard."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2rem;
            font-weight: bold;
            color: #0078ff;
            margin-bottom: 1rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #0066cc;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .alert-box {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .alert-success {
            background-color: #d4edda;
            color: #155724;
        }
        .alert-warning {
            background-color: #fff3cd;
            color: #856404;
        }
        .alert-danger {
            background-color: #f8d7da;
            color: #721c24;
        }
        .card {
            padding: 15px;
            border-radius: 5px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }
        .metrics-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #666;
        }
        .small-font {
            font-size: 0.8rem;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .status-active {
            background-color: #28a745;
        }
        .status-inactive {
            background-color: #dc3545;
        }
        .status-warning {
            background-color: #ffc107;
        }
        /* Định dạng cho giao diện dark mode */
        @media (prefers-color-scheme: dark) {
            .metric-container {
                background-color: #262730;
            }
            .metric-label {
                color: #ccc;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def load_system_config():
    """Tải cấu hình hệ thống."""
    system_config = get_system_config()
    return system_config.get_all()

def load_trading_data():
    """Tải dữ liệu giao dịch gần đây từ các file log."""
    try:
        # Lấy danh sách file log giao dịch
        trading_log_dir = LOG_DIR / "trading"
        log_files = list(trading_log_dir.glob("trading_*_*.log"))
        
        # Sắp xếp file theo thời gian (mới nhất trước)
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not log_files:
            return pd.DataFrame()
        
        # Đọc file log mới nhất
        latest_log = log_files[0]
        
        # Phân tích log để tạo DataFrame
        trades = []
        with open(latest_log, 'r', encoding='utf-8') as f:
            for line in f:
                if "Đã mở vị thế" in line or "Đã đóng vị thế" in line:
                    try:
                        parts = line.split(" - ")
                        timestamp = parts[0].strip()
                        
                        # Trích xuất thông tin giao dịch
                        if "Đã mở vị thế" in line:
                            # Ví dụ: Đã mở vị thế LONG 0.01 @ 27350.5, đòn bẩy: 5x
                            trade_info = parts[-1].strip()
                            trade_parts = trade_info.split()
                            
                            direction = trade_parts[3]  # LONG hoặc SHORT
                            size = float(trade_parts[4])
                            price = float(trade_parts[6].rstrip(','))
                            
                            leverage = 1.0
                            if len(trade_parts) > 8 and "đòn bẩy:" in trade_info:
                                leverage = float(trade_parts[-1].rstrip('x'))
                            
                            trades.append({
                                'timestamp': timestamp,
                                'action': 'open',
                                'direction': direction,
                                'size': size,
                                'price': price,
                                'leverage': leverage,
                                'pnl': None,
                                'pnl_percent': None
                            })
                        
                        elif "Đã đóng vị thế" in line:
                            # Ví dụ: Đã đóng vị thế tại 27450.5, P&L: 100.0 (0.37%)
                            trade_info = parts[-1].strip()
                            trade_parts = trade_info.split(', ')
                            
                            price_part = trade_parts[0].split()
                            price = float(price_part[-1])
                            
                            pnl_parts = trade_parts[1].split(': ')
                            pnl = float(pnl_parts[1])
                            
                            pnl_percent_part = trade_parts[2].strip('()')
                            pnl_percent = float(pnl_percent_part.rstrip('%'))
                            
                            trades.append({
                                'timestamp': timestamp,
                                'action': 'close',
                                'direction': None,  # Không biết hướng từ thông tin đóng
                                'size': None,       # Không biết size từ thông tin đóng
                                'price': price,
                                'leverage': None,
                                'pnl': pnl,
                                'pnl_percent': pnl_percent
                            })
                    except Exception as e:
                        logger.warning(f"Lỗi khi phân tích dòng log: {e}", extra={"component": "dashboard"})
                        continue
        
        # Tạo DataFrame từ danh sách giao dịch
        df = pd.DataFrame(trades)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu giao dịch: {e}", extra={"component": "dashboard"}, exc_info=True)
        return pd.DataFrame()

def load_agent_metrics():
    """Tải thông tin về các agent đã huấn luyện."""
    try:
        agent_dir = ROOT_DIR / "saved_models"
        agent_info = []
        
        # Kiểm tra các thư mục agent
        for agent_path in agent_dir.glob("*"):
            if agent_path.is_dir():
                agent_name = agent_path.name
                
                # Tìm file metrics gần đây nhất
                metrics_files = list(agent_path.glob("metrics_*.json"))
                metrics_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Tìm file model gần đây nhất
                model_files = list(agent_path.glob("model_*.pt"))
                model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                if metrics_files:
                    # Đọc metrics
                    with open(metrics_files[0], 'r') as f:
                        metrics = json.load(f)
                    
                    # Lấy thông tin cơ bản
                    last_update = datetime.fromtimestamp(metrics_files[0].stat().st_mtime)
                    episodes = metrics.get("total_episodes", 0)
                    
                    # Tính các metrics hiệu suất
                    avg_reward = metrics.get("average_reward", 0)
                    win_rate = metrics.get("win_rate", 0) * 100
                    sharpe = metrics.get("sharpe_ratio", 0)
                    
                    agent_info.append({
                        "name": agent_name,
                        "last_update": last_update,
                        "episodes": episodes,
                        "avg_reward": avg_reward,
                        "win_rate": win_rate,
                        "sharpe": sharpe,
                        "model_path": str(model_files[0]) if model_files else "N/A"
                    })
        
        return pd.DataFrame(agent_info)
    
    except Exception as e:
        logger.error(f"Lỗi khi tải metrics agent: {e}", extra={"component": "dashboard"}, exc_info=True)
        return pd.DataFrame()

def check_system_status():
    """Kiểm tra trạng thái hệ thống."""
    try:
        # Lấy thông tin hệ thống
        stats = get_system_stats()
        
        # Kiểm tra các thành phần
        components_status = {
            "API Server": {
                "status": "active" if stats["api_server"]["running"] else "inactive",
                "message": f"Hoạt động bình thường" if stats["api_server"]["running"] else "Không hoạt động",
                "last_check": datetime.now().strftime("%H:%M:%S")
            },
            "Data Collector": {
                "status": "active" if stats["data_collector"]["running"] else "inactive",
                "message": f"Đang thu thập: {stats['data_collector']['symbols_count']} cặp" if stats["data_collector"]["running"] else "Không hoạt động",
                "last_check": datetime.now().strftime("%H:%M:%S")
            },
            "Trading Agent": {
                "status": "active" if stats["trading_agent"]["running"] else "inactive",
                "message": f"Đang giao dịch: {stats['trading_agent']['active_pairs']} cặp" if stats["trading_agent"]["running"] else "Không hoạt động",
                "last_check": datetime.now().strftime("%H:%M:%S")
            },
            "Risk Manager": {
                "status": "active" if stats["risk_manager"]["running"] else "inactive",
                "message": f"Hoạt động bình thường" if stats["risk_manager"]["running"] else "Không hoạt động",
                "last_check": datetime.now().strftime("%H:%M:%S")
            },
            "Database": {
                "status": "active" if stats["database"]["connected"] else "inactive",
                "message": f"Kết nối OK" if stats["database"]["connected"] else "Mất kết nối",
                "last_check": datetime.now().strftime("%H:%M:%S")
            }
        }
        
        # Xác định trạng thái hệ thống tổng thể
        if all(c["status"] == "active" for c in components_status.values()):
            overall_status = "active"
            overall_message = "Tất cả các hệ thống hoạt động bình thường"
        elif any(c["status"] == "inactive" for c in components_status.values()):
            overall_status = "warning"
            inactive_components = [name for name, c in components_status.items() if c["status"] == "inactive"]
            overall_message = f"Một số thành phần không hoạt động: {', '.join(inactive_components)}"
        else:
            overall_status = "inactive"
            overall_message = "Hệ thống không hoạt động"
        
        return {
            "overall": {
                "status": overall_status,
                "message": overall_message,
                "last_check": datetime.now().strftime("%H:%M:%S")
            },
            "components": components_status,
            "resources": {
                "cpu_usage": stats["resources"]["cpu_percent"],
                "memory_usage": stats["resources"]["memory_percent"],
                "disk_usage": stats["resources"]["disk_percent"]
            }
        }
    
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra trạng thái hệ thống: {e}", extra={"component": "dashboard"}, exc_info=True)
        return {
            "overall": {
                "status": "unknown",
                "message": f"Lỗi khi kiểm tra trạng thái: {str(e)}",
                "last_check": datetime.now().strftime("%H:%M:%S")
            },
            "components": {},
            "resources": {
                "cpu_usage": 0,
                "memory_usage": 0,
                "disk_usage": 0
            }
        }

def load_recent_logs(n_entries=100):
    """Tải các dòng log gần đây."""
    try:
        log_files = list(LOG_DIR.glob("*.log"))
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not log_files:
            return []
        
        # Đọc file log mới nhất
        latest_log = log_files[0]
        
        # Đọc n dòng cuối cùng
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-n_entries:] if len(lines) > n_entries else lines
    
    except Exception as e:
        logger.error(f"Lỗi khi tải log gần đây: {e}", extra={"component": "dashboard"}, exc_info=True)
        return []

def render_main_dashboard():
    """Hiển thị dashboard chính."""
    # Header
    st.markdown('<div class="main-header">📈 Automated Trading System</div>', unsafe_allow_html=True)
    
    # Tải dữ liệu
    system_config = load_system_config()
    trading_data = load_trading_data()
    agent_data = load_agent_metrics()
    system_status = check_system_status()
    
    # Hiển thị trạng thái hệ thống
    status_color = {
        "active": "green",
        "warning": "orange",
        "inactive": "red",
        "unknown": "gray"
    }
    
    st.markdown(
        f"""<div class="alert-box alert-{'success' if system_status['overall']['status'] == 'active' else 'warning' if system_status['overall']['status'] == 'warning' else 'danger'}">
            <span style="color: {status_color[system_status['overall']['status']]}; font-size: 1.2rem;">●</span>
            <strong>Trạng thái hệ thống:</strong> {system_status['overall']['message']}
            <span style="float: right;" class="small-font">Cập nhật lúc: {system_status['overall']['last_check']}</span>
        </div>""",
        unsafe_allow_html=True
    )
    
    # Tabs chính
    tabs = st.tabs(["Tổng quan", "Giao dịch", "Agent", "Hệ thống", "Logs"])
    
    # Tab 1: Tổng quan
    with tabs[0]:
        # Hiển thị chỉ số chính
        col1, col2, col3, col4 = st.columns(4)
        
        # Tính toán các metrics từ dữ liệu giao dịch
        if not trading_data.empty:
            closed_trades = trading_data[trading_data['action'] == 'close']
            total_trades = len(closed_trades)
            
            if total_trades > 0:
                profitable_trades = len(closed_trades[closed_trades['pnl'] > 0])
                win_rate = (profitable_trades / total_trades) * 100
                avg_pnl = closed_trades['pnl'].mean()
                avg_pnl_percent = closed_trades['pnl_percent'].mean()
            else:
                win_rate = 0
                avg_pnl = 0
                avg_pnl_percent = 0
        else:
            total_trades = 0
            win_rate = 0
            avg_pnl = 0
            avg_pnl_percent = 0
        
        with col1:
            st.markdown(
                f"""<div class="card">
                    <div class="metric-label">Tổng số giao dịch</div>
                    <div class="metric-value">{total_trades}</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""<div class="card">
                    <div class="metric-label">Tỷ lệ thắng</div>
                    <div class="metric-value">{win_rate:.2f}%</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""<div class="card">
                    <div class="metric-label">P&L trung bình</div>
                    <div class="metric-value">{avg_pnl:.2f}</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                f"""<div class="card">
                    <div class="metric-label">P&L % trung bình</div>
                    <div class="metric-value">{avg_pnl_percent:.2f}%</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        # Biểu đồ hiệu suất
        st.markdown('<div class="sub-header">Hiệu suất giao dịch</div>', unsafe_allow_html=True)
        
        if not trading_data.empty:
            # Tạo dữ liệu cho biểu đồ P&L tích lũy
            closed_trades = trading_data[trading_data['action'] == 'close'].copy()
            
            if not closed_trades.empty:
                # Sắp xếp theo thời gian
                closed_trades = closed_trades.sort_values('timestamp')
                
                # Tính P&L tích lũy
                closed_trades['cumulative_pnl'] = closed_trades['pnl'].cumsum()
                
                # Vẽ biểu đồ
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=closed_trades['timestamp'],
                    y=closed_trades['cumulative_pnl'],
                    mode='lines',
                    name='P&L tích lũy',
                    line=dict(color='#0078ff', width=2)
                ))
                
                fig.update_layout(
                    title='P&L tích lũy theo thời gian',
                    xaxis_title='Thời gian',
                    yaxis_title='P&L',
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Biểu đồ phân phối lợi nhuận
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=closed_trades['pnl_percent'],
                        marker_color='#0078ff',
                        opacity=0.7,
                        name='Phân phối lợi nhuận'
                    ))
                    
                    fig_dist.update_layout(
                        title='Phân phối lợi nhuận (%)',
                        xaxis_title='Lợi nhuận (%)',
                        yaxis_title='Số lượng giao dịch',
                        height=300,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    # Tính drawdown
                    cumulative = closed_trades['pnl'].cumsum()
                    running_max = cumulative.cummax()
                    drawdown = (cumulative - running_max) / running_max * 100
                    
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(
                        x=closed_trades['timestamp'],
                        y=drawdown,
                        mode='lines',
                        name='Drawdown',
                        line=dict(color='#ff3b30', width=2),
                        fill='tozeroy'
                    ))
                    
                    fig_dd.update_layout(
                        title='Drawdown (%)',
                        xaxis_title='Thời gian',
                        yaxis_title='Drawdown (%)',
                        height=300,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig_dd, use_container_width=True)
            else:
                st.info("Chưa có giao dịch nào được đóng. Không có dữ liệu hiệu suất để hiển thị.")
        else:
            st.info("Không có dữ liệu giao dịch. Hãy bắt đầu giao dịch để xem hiệu suất.")
        
        # Thông tin hệ thống
        st.markdown('<div class="sub-header">Tài nguyên hệ thống</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_color = "#00b894" if system_status["resources"]["cpu_usage"] < 70 else "#fdcb6e" if system_status["resources"]["cpu_usage"] < 90 else "#ff7675"
            st.markdown(
                f"""<div class="card">
                    <div class="metric-label">CPU</div>
                    <div class="metric-value" style="color: {cpu_color};">{system_status["resources"]["cpu_usage"]}%</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        with col2:
            memory_color = "#00b894" if system_status["resources"]["memory_usage"] < 70 else "#fdcb6e" if system_status["resources"]["memory_usage"] < 90 else "#ff7675"
            st.markdown(
                f"""<div class="card">
                    <div class="metric-label">Bộ nhớ</div>
                    <div class="metric-value" style="color: {memory_color};">{system_status["resources"]["memory_usage"]}%</div>
                </div>""",
                unsafe_allow_html=True
            )
        
        with col3:
            disk_color = "#00b894" if system_status["resources"]["disk_usage"] < 70 else "#fdcb6e" if system_status["resources"]["disk_usage"] < 90 else "#ff7675"
            st.markdown(
                f"""<div class="card">
                    <div class="metric-label">Đĩa</div>
                    <div class="metric-value" style="color: {disk_color};">{system_status["resources"]["disk_usage"]}%</div>
                </div>""",
                unsafe_allow_html=True
            )
    
    # Tab 2: Giao dịch
    with tabs[1]:
        st.markdown('<div class="sub-header">Lịch sử giao dịch</div>', unsafe_allow_html=True)
        
        if not trading_data.empty:
            # Filter cho tab giao dịch
            st.markdown("#### Bộ lọc")
            col1, col2 = st.columns(2)
            
            with col1:
                date_filter = st.date_input(
                    "Khoảng thời gian",
                    value=(
                        trading_data['timestamp'].min().date(),
                        datetime.now().date()
                    ),
                    key="date_filter"
                )
            
            with col2:
                action_filter = st.multiselect(
                    "Loại giao dịch",
                    options=["open", "close"],
                    default=["open", "close"],
                    format_func=lambda x: "Mở vị thế" if x == "open" else "Đóng vị thế",
                    key="action_filter"
                )
            
            # Lọc dữ liệu
            filtered_data = trading_data.copy()
            
            if len(date_filter) == 2:
                start_date, end_date = date_filter
                end_date = end_date + timedelta(days=1)  # Để bao gồm cả ngày cuối
                filtered_data = filtered_data[(filtered_data['timestamp'].dt.date >= start_date) & 
                                             (filtered_data['timestamp'].dt.date <= end_date)]
            
            if action_filter:
                filtered_data = filtered_data[filtered_data['action'].isin(action_filter)]
            
            # Hiển thị dữ liệu
            if not filtered_data.empty:
                # Định dạng dữ liệu để hiển thị
                display_data = filtered_data.copy()
                display_data['timestamp'] = display_data['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
                display_data['action'] = display_data['action'].map({"open": "Mở vị thế", "close": "Đóng vị thế"})
                
                # Định dạng cột
                renamed_columns = {
                    'timestamp': 'Thời gian',
                    'action': 'Hành động',
                    'direction': 'Hướng',
                    'size': 'Kích thước',
                    'price': 'Giá',
                    'leverage': 'Đòn bẩy',
                    'pnl': 'P&L',
                    'pnl_percent': 'P&L %'
                }
                
                # Hiển thị bảng dữ liệu
                st.dataframe(
                    display_data.rename(columns=renamed_columns),
                    use_container_width=True,
                    height=400
                )
                
                # Thống kê giao dịch
                st.markdown("#### Thống kê giao dịch")
                
                # Chỉ tính toán metrics cho các giao dịch đã đóng
                closed_trades = filtered_data[filtered_data['action'] == 'close']
                
                if not closed_trades.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Tổng số giao dịch đã đóng", len(closed_trades))
                    
                    with col2:
                        profitable_trades = len(closed_trades[closed_trades['pnl'] > 0])
                        win_rate = (profitable_trades / len(closed_trades)) * 100
                        st.metric("Tỷ lệ thắng", f"{win_rate:.2f}%")
                    
                    with col3:
                        avg_pnl = closed_trades['pnl'].mean()
                        st.metric("P&L trung bình", f"{avg_pnl:.2f}")
                    
                    with col4:
                        avg_pnl_percent = closed_trades['pnl_percent'].mean()
                        st.metric("P&L % trung bình", f"{avg_pnl_percent:.2f}%")
                    
                    # Tính các metrics bổ sung
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        max_profit = closed_trades['pnl'].max()
                        st.metric("Lợi nhuận lớn nhất", f"{max_profit:.2f}")
                    
                    with col2:
                        max_loss = closed_trades['pnl'].min()
                        st.metric("Thua lỗ lớn nhất", f"{max_loss:.2f}")
                    
                    with col3:
                        profit_sum = closed_trades[closed_trades['pnl'] > 0]['pnl'].sum()
                        st.metric("Tổng lợi nhuận", f"{profit_sum:.2f}")
                    
                    with col4:
                        loss_sum = closed_trades[closed_trades['pnl'] < 0]['pnl'].sum()
                        st.metric("Tổng thua lỗ", f"{loss_sum:.2f}")
                    
                    # Tính Profit Factor
                    if loss_sum != 0:
                        profit_factor = abs(profit_sum / loss_sum)
                    else:
                        profit_factor = float('inf')
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Profit Factor", f"{profit_factor:.2f}")
                    
                    with col2:
                        if len(closed_trades) > 0:
                            expected_return = (win_rate / 100 * closed_trades[closed_trades['pnl'] > 0]['pnl'].mean() + 
                                          (1 - win_rate / 100) * closed_trades[closed_trades['pnl'] < 0]['pnl'].mean())
                        else:
                            expected_return = 0
                        st.metric("Kỳ vọng toán", f"{expected_return:.2f}")
                    
                    with col3:
                        net_profit = closed_trades['pnl'].sum()
                        st.metric("Lợi nhuận ròng", f"{net_profit:.2f}")
                    
                    with col4:
                        roi = (closed_trades['pnl_percent'].mean() * len(closed_trades))
                        st.metric("ROI tổng", f"{roi:.2f}%")
                else:
                    st.info("Không có giao dịch nào đã đóng trong khoảng thời gian đã chọn.")
            else:
                st.info("Không có dữ liệu giao dịch phù hợp với bộ lọc đã chọn.")
        else:
            st.info("Không có dữ liệu giao dịch để hiển thị.")
    
    # Tab 3: Agent
    with tabs[2]:
        st.markdown('<div class="sub-header">Thông tin Agent</div>', unsafe_allow_html=True)
        
        if not agent_data.empty:
            # Hiển thị danh sách agent
            st.markdown("#### Danh sách Agent đã huấn luyện")
            
            # Định dạng dữ liệu để hiển thị
            display_data = agent_data.copy()
            display_data['last_update'] = display_data['last_update'].dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Đổi tên cột để hiển thị
            renamed_columns = {
                'name': 'Tên Agent',
                'last_update': 'Cập nhật lần cuối',
                'episodes': 'Số episode',
                'avg_reward': 'Phần thưởng TB',
                'win_rate': 'Tỷ lệ thắng (%)',
                'sharpe': 'Tỷ số Sharpe',
                'model_path': 'Đường dẫn mô hình'
            }
            
            # Hiển thị bảng dữ liệu
            st.dataframe(
                display_data.rename(columns=renamed_columns),
                use_container_width=True,
                height=300
            )
            
            # Chọn agent để xem chi tiết
            selected_agent = st.selectbox(
                "Chọn Agent để xem chi tiết",
                options=agent_data['name'].tolist(),
                key="agent_select"
            )
            
            if selected_agent:
                # Lấy thông tin agent đã chọn
                agent_info = agent_data[agent_data['name'] == selected_agent].iloc[0]
                
                st.markdown(f"#### Chi tiết Agent: {selected_agent}")
                
                # Hiển thị các thông số chi tiết
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Số episode đã huấn luyện", agent_info['episodes'])
                
                with col2:
                    st.metric("Phần thưởng trung bình", f"{agent_info['avg_reward']:.4f}")
                
                with col3:
                    st.metric("Tỷ lệ thắng", f"{agent_info['win_rate']:.2f}%")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Tỷ số Sharpe", f"{agent_info['sharpe']:.4f}")
                
                with col2:
                    st.metric("Cập nhật lần cuối", agent_info['last_update'])
                
                with col3:
                    st.text("Đường dẫn mô hình")
                    st.code(agent_info['model_path'], language="text")
                
                # Nút triển khai agent
                col1, col2 = st.columns(2)
                
                with col1:
                    st.button(
                        "Triển khai Agent này",
                        type="primary",
                        key="deploy_agent_button",
                        help="Triển khai agent này để giao dịch thực"
                    )
                
                with col2:
                    st.button(
                        "Tiếp tục huấn luyện",
                        key="continue_training_button",
                        help="Tiếp tục huấn luyện agent này"
                    )
                
                # Thông tin về quá trình huấn luyện
                st.markdown("#### Biểu đồ huấn luyện")
                
                # Giả lập dữ liệu cho biểu đồ huấn luyện
                # Trong thực tế, bạn sẽ tải dữ liệu huấn luyện thực từ file log hoặc metrics
                episodes = list(range(1, agent_info['episodes'] + 1))
                reward_data = [agent_info['avg_reward'] * (1 + 0.2 * (i / agent_info['episodes']) + 0.05 * (0.5 - (i % 30) / 30)) for i in episodes]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=episodes,
                    y=reward_data,
                    mode='lines',
                    name='Phần thưởng',
                    line=dict(color='#0078ff', width=2)
                ))
                
                fig.update_layout(
                    title='Phần thưởng theo episode',
                    xaxis_title='Episode',
                    yaxis_title='Phần thưởng',
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu agent nào được tìm thấy. Hãy huấn luyện agent để xem thông tin.")
    
    # Tab 4: Hệ thống
    with tabs[3]:
        st.markdown('<div class="sub-header">Trạng thái hệ thống</div>', unsafe_allow_html=True)
        
        # Hiển thị thông tin tài nguyên hệ thống
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=system_status["resources"]["cpu_usage"],
                title={"text": "CPU Usage"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0078ff"},
                    "steps": [
                        {"range": [0, 50], "color": "#e8f5e9"},
                        {"range": [50, 80], "color": "#ffe0b2"},
                        {"range": [80, 100], "color": "#ffcdd2"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90
                    }
                }
            ))
            
            cpu_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(cpu_gauge, use_container_width=True)
        
        with col2:
            memory_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=system_status["resources"]["memory_usage"],
                title={"text": "Memory Usage"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0078ff"},
                    "steps": [
                        {"range": [0, 50], "color": "#e8f5e9"},
                        {"range": [50, 80], "color": "#ffe0b2"},
                        {"range": [80, 100], "color": "#ffcdd2"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90
                    }
                }
            ))
            
            memory_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(memory_gauge, use_container_width=True)
        
        with col3:
            disk_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=system_status["resources"]["disk_usage"],
                title={"text": "Disk Usage"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0078ff"},
                    "steps": [
                        {"range": [0, 50], "color": "#e8f5e9"},
                        {"range": [50, 80], "color": "#ffe0b2"},
                        {"range": [80, 100], "color": "#ffcdd2"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90
                    }
                }
            ))
            
            disk_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(disk_gauge, use_container_width=True)
        
        # Trạng thái các thành phần
        st.markdown("#### Các thành phần hệ thống")
        
        for component_name, component in system_status["components"].items():
            status_icon = "🟢" if component["status"] == "active" else "🔴"
            st.markdown(
                f"""<div class="card">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <strong>{status_icon} {component_name}</strong><br>
                            <span class="small-font">{component["message"]}</span>
                        </div>
                        <div class="small-font">
                            Kiểm tra lúc: {component["last_check"]}
                        </div>
                    </div>
                </div>""",
                unsafe_allow_html=True
            )
        
        # Cấu hình hệ thống
        st.markdown("#### Cấu hình hệ thống")
        
        # Hiển thị tab cho các nhóm cấu hình
        config_tabs = st.tabs(["Cấu hình chung", "Giao dịch", "Agent", "Hệ thống"])
        
        with config_tabs[0]:
            # Cấu hình chung
            st.json({
                "version": system_config["version"],
                "build_date": system_config["build_date"],
                "environment": system_config["environment"],
                "debug_mode": system_config["debug_mode"]
            })
        
        with config_tabs[1]:
            # Cấu hình giao dịch
            st.json(system_config["trading"])
        
        with config_tabs[2]:
            # Cấu hình agent
            st.json(system_config["agent"])
        
        with config_tabs[3]:
            # Cấu hình hệ thống chi tiết
            st.json({
                "max_threads": system_config["max_threads"],
                "max_processes": system_config["max_processes"],
                "request_timeout": system_config["request_timeout"],
                "max_retries": system_config["max_retries"],
                "memory_limit": system_config["memory_limit"],
                "data_storage_format": system_config["data_storage_format"]
            })
    
    # Tab 5: Logs
    with tabs[4]:
        st.markdown('<div class="sub-header">Logs hệ thống</div>', unsafe_allow_html=True)
        
        # Filter logs
        col1, col2 = st.columns(2)
        
        with col1:
            log_type = st.selectbox(
                "Loại log",
                options=["main", "trading", "training", "api", "system", "error"],
                format_func=lambda x: {
                    "main": "Log chính",
                    "trading": "Giao dịch",
                    "training": "Huấn luyện",
                    "api": "API",
                    "system": "Hệ thống",
                    "error": "Lỗi"
                }[x],
                key="log_type_select"
            )
        
        with col2:
            log_level = st.selectbox(
                "Mức độ log",
                options=["ALL", "INFO", "WARNING", "ERROR", "CRITICAL"],
                key="log_level_select"
            )
        
        # Tải log
        log_entries = load_recent_logs(200)
        
        # Lọc log theo loại và mức độ
        filtered_logs = []
        
        for log in log_entries:
            # Kiểm tra loại log
            if log_type == "main" or f"[{log_type.upper()}]" in log:
                # Kiểm tra mức độ log
                if log_level == "ALL" or f"[{log_level}]" in log:
                    filtered_logs.append(log)
        
        # Hiển thị log
        st.text_area(
            "Log entries",
            value="".join(filtered_logs),
            height=500,
            key="log_text_area"
        )
        
        # Nút làm mới log
        col1, col2 = st.columns([1, 4])
        with col1:
            st.button(
                "Làm mới",
                key="refresh_logs_button"
            )

# Render sidebar
def render_sidebar():
    """Hiển thị sidebar."""
    st.sidebar.markdown("## Automated Trading System")
    
    # Thông tin phiên bản hệ thống
    system_config = load_system_config()
    st.sidebar.markdown(f"**Phiên bản:** {system_config['version']}")
    st.sidebar.markdown(f"**Môi trường:** {system_config['environment']}")
    
    # Thêm các phần khác cho sidebar
    st.sidebar.markdown("---")
    
    # Menu điều hướng
    st.sidebar.markdown("### Điều hướng")
    page = st.sidebar.radio(
        "Đi tới:",
        options=["Dashboard", "Quản lý Agent", "Quản lý Giao dịch", "Cài đặt Hệ thống", "Giám sát"],
        format_func=lambda x: {
            "Dashboard": "📊 Dashboard",
            "Quản lý Agent": "🤖 Quản lý Agent",
            "Quản lý Giao dịch": "📈 Quản lý Giao dịch",
            "Cài đặt Hệ thống": "⚙️ Cài đặt Hệ thống",
            "Giám sát": "👁️ Giám sát"
        }[x],
    )
    
    # Hiển thị trang tương ứng (hiện tại chỉ có Dashboard)
    if page != "Dashboard":
        st.sidebar.info(f"Trang {page} đang trong quá trình phát triển. Vui lòng quay lại sau.")
    
    # Phần điều khiển nhanh
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Điều khiển nhanh")
    
    # Nút khởi động/dừng hệ thống
    system_running = st.sidebar.checkbox("Hệ thống đang chạy", value=True, key="system_running")
    
    if system_running:
        if st.sidebar.button("Dừng hệ thống", key="stop_system"):
            # Logic dừng hệ thống
            st.sidebar.warning("Đang dừng hệ thống...")
            time.sleep(1)
            st.sidebar.success("Hệ thống đã dừng thành công!")
            st.sidebar.checkbox("Hệ thống đang chạy", value=False, key="system_running_updated")
    else:
        if st.sidebar.button("Khởi động hệ thống", key="start_system"):
            # Logic khởi động hệ thống
            st.sidebar.info("Đang khởi động hệ thống...")
            time.sleep(1)
            st.sidebar.success("Hệ thống đã khởi động thành công!")
            st.sidebar.checkbox("Hệ thống đang chạy", value=True, key="system_running_updated")
    
    # Nút thiết lập lại cấu hình
    if st.sidebar.button("Tải lại cấu hình", key="reload_config"):
        st.sidebar.info("Đang tải lại cấu hình...")
        time.sleep(1)
        st.sidebar.success("Cấu hình đã được tải lại thành công!")
    
    # Thông tin phần dưới
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Hỗ trợ")
    st.sidebar.markdown("📧 Email: support@example.com")
    st.sidebar.markdown("🔗 Documentation: [Link](https://docs.example.com)")
    
    # Thông tin cuối trang
    st.sidebar.markdown("---")
    st.sidebar.markdown("Automated Trading System © 2024")
    st.sidebar.markdown("*Version 0.1.0*")

def main():
    """Hàm chính của ứng dụng Streamlit."""
    # Tải CSS
    load_css()
    
    # Render sidebar
    render_sidebar()
    
    # Render dashboard chính
    render_main_dashboard()

if __name__ == "__main__":
    main()