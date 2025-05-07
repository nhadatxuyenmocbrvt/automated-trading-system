"""
Dashboard trang giám sát hệ thống.
File này cung cấp giao diện người dùng để giám sát tình trạng,
hiệu suất, và sức khỏe của hệ thống giao dịch tự động.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import datetime
from datetime import timedelta
import psutil
import socket
import platform
from typing import Dict, List, Any, Optional, Tuple, Union

# Import các module từ hệ thống
from logs.metrics.system_metrics import get_system_metrics, SystemMetricsCollector
from logs.logger import get_system_logger
from config.constants import SystemStatus, OrderStatus, Exchange, Timeframe
from config.env import get_env
from streamlit_dashboard.components.sidebar import render_sidebar

# Khởi tạo logger
logger = get_system_logger("dashboard_system_monitor")

# Tiêu đề trang
st.set_page_config(
    page_title="ATS - Giám sát hệ thống",
    page_icon="📊",
    layout="wide"
)

def create_system_overview():
    """Tạo phần tổng quan hệ thống."""
    
    st.header("Tổng quan hệ thống")
    
    # Lấy số liệu từ bộ thu thập
    metrics_collector = get_system_metrics()
    system_summary = metrics_collector.get_system_summary()
    
    # Tạo các card thông tin
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="CPU Usage",
            value=f"{system_summary['cpu']['usage']:.1f}%",
            delta=f"{system_summary['cpu']['usage'] - system_summary['cpu']['average']:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Memory Usage",
            value=f"{system_summary['memory']['usage']:.1f}%",
            delta=f"{system_summary['memory']['usage'] - system_summary['memory']['average']:.1f}%"
        )
    
    with col3:
        st.metric(
            label="Disk Usage",
            value=f"{system_summary['disk']['usage']:.1f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Network Latency",
            value=f"{system_summary['network']['latency']:.1f}ms",
            delta=f"{system_summary['network']['latency'] - system_summary['network']['average']:.1f}ms",
            delta_color="inverse"
        )
    
    st.divider()
    
    return system_summary

def create_system_health_cards(system_summary: Dict[str, Any]):
    """
    Tạo các thẻ trạng thái sức khỏe hệ thống.
    
    Args:
        system_summary: Tóm tắt thông tin hệ thống
    """
    st.subheader("Sức khỏe hệ thống")
    
    # Định nghĩa ngưỡng cảnh báo
    thresholds = {
        "cpu": {"warning": 70, "critical": 90},
        "memory": {"warning": 80, "critical": 90},
        "disk": {"warning": 80, "critical": 90},
        "network": {"warning": 100, "critical": 200}  # ms
    }
    
    # Tạo các thẻ trạng thái
    cols = st.columns(4)
    
    # CPU Health
    cpu_usage = system_summary['cpu']['usage']
    cpu_status = "Tốt"
    cpu_color = "green"
    
    if cpu_usage >= thresholds["cpu"]["critical"]:
        cpu_status = "Nghiêm trọng"
        cpu_color = "red"
    elif cpu_usage >= thresholds["cpu"]["warning"]:
        cpu_status = "Cảnh báo"
        cpu_color = "orange"
    
    cols[0].markdown(
        f"""
        <div style="padding:10px;border-radius:5px;border:1px solid {cpu_color};">
            <h3 style="color:{cpu_color};margin:0;">CPU Status: {cpu_status}</h3>
            <p>Hiện tại: {cpu_usage:.1f}%</p>
            <p>Trung bình: {system_summary['cpu']['average']:.1f}%</p>
            <p>Per Core: {', '.join([f"{x:.0f}%" for x in system_summary['cpu']['per_core'][:4]])}{'...' if len(system_summary['cpu']['per_core']) > 4 else ''}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Memory Health
    mem_usage = system_summary['memory']['usage']
    mem_status = "Tốt"
    mem_color = "green"
    
    if mem_usage >= thresholds["memory"]["critical"]:
        mem_status = "Nghiêm trọng"
        mem_color = "red"
    elif mem_usage >= thresholds["memory"]["warning"]:
        mem_status = "Cảnh báo"
        mem_color = "orange"
    
    mem_details = system_summary['memory']['details']
    cols[1].markdown(
        f"""
        <div style="padding:10px;border-radius:5px;border:1px solid {mem_color};">
            <h3 style="color:{mem_color};margin:0;">Memory Status: {mem_status}</h3>
            <p>Hiện tại: {mem_usage:.1f}%</p>
            <p>Tổng: {mem_details['total']:.1f} GB</p>
            <p>Khả dụng: {mem_details['available']:.1f} GB</p>
            <p>Đã dùng: {mem_details['used']:.1f} GB</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Disk Health
    disk_usage = system_summary['disk']['usage']
    disk_status = "Tốt"
    disk_color = "green"
    
    if disk_usage >= thresholds["disk"]["critical"]:
        disk_status = "Nghiêm trọng"
        disk_color = "red"
    elif disk_usage >= thresholds["disk"]["warning"]:
        disk_status = "Cảnh báo"
        disk_color = "orange"
    
    disk_details = system_summary['disk']['details']
    cols[2].markdown(
        f"""
        <div style="padding:10px;border-radius:5px;border:1px solid {disk_color};">
            <h3 style="color:{disk_color};margin:0;">Disk Status: {disk_status}</h3>
            <p>Hiện tại: {disk_usage:.1f}%</p>
            <p>Tổng: {disk_details['total']:.1f} GB</p>
            <p>Khả dụng: {disk_details['free']:.1f} GB</p>
            <p>Đã dùng: {disk_details['used']:.1f} GB</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Network Health
    net_latency = system_summary['network']['latency']
    net_status = "Tốt"
    net_color = "green"
    
    if net_latency >= thresholds["network"]["critical"]:
        net_status = "Nghiêm trọng"
        net_color = "red"
    elif net_latency >= thresholds["network"]["warning"]:
        net_status = "Cảnh báo"
        net_color = "orange"
    
    net_stats = system_summary['network']['stats']
    cols[3].markdown(
        f"""
        <div style="padding:10px;border-radius:5px;border:1px solid {net_color};">
            <h3 style="color:{net_color};margin:0;">Network Status: {net_status}</h3>
            <p>Độ trễ: {net_latency:.1f} ms</p>
            <p>Băng thông gửi: {net_stats['bandwidth']['sent']:.2f} MB/s</p>
            <p>Băng thông nhận: {net_stats['bandwidth']['received']:.2f} MB/s</p>
            <p>Kết nối: {net_stats['connections']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()

def create_system_info_section(system_summary: Dict[str, Any]):
    """
    Tạo phần thông tin hệ thống.
    
    Args:
        system_summary: Tóm tắt thông tin hệ thống
    """
    st.subheader("Thông tin hệ thống")
    
    system_info = system_summary['system']
    
    # Tạo layout 2 cột
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Thông tin cơ bản")
        info_table = {
            "Hostname": system_info["hostname"],
            "Platform": system_info["platform"],
            "Python Version": system_info["python_version"],
            "CPU Cores": system_info["cpu_count"],
            "Uptime": f"{system_info['uptime'] / 3600:.1f} giờ",
            "Thời gian hệ thống": system_info["timestamp"]
        }
        
        st.table(pd.DataFrame(list(info_table.items()), columns=["Thuộc tính", "Giá trị"]))
    
    with col2:
        st.markdown("#### Thông tin môi trường")
        
        # Lấy một số thông tin từ biến môi trường
        env_info = {
            "Môi trường": get_env("TRADING_ENV", "development").upper(),
            "Trạng thái": get_env("SYSTEM_STATUS", "unknown").upper(),
            "Số vị thế tối đa": get_env("MAX_OPEN_POSITIONS", "5"),
            "Đòn bẩy mặc định": get_env("DEFAULT_LEVERAGE", "1.0"),
            "Rủi ro mỗi giao dịch": f"{float(get_env('RISK_PER_TRADE', '0.02')) * 100:.1f}%",
            "Bộ nhớ đệm logs": get_env("LOG_BUFFER_SIZE", "1000"),
            "Tần suất backup": get_env("BACKUP_FREQUENCY", "1 ngày")
        }
        
        st.table(pd.DataFrame(list(env_info.items()), columns=["Biến", "Giá trị"]))
    
    st.divider()

def create_resource_usage_charts(system_summary: Dict[str, Any]):
    """
    Tạo biểu đồ sử dụng tài nguyên.
    
    Args:
        system_summary: Tóm tắt thông tin hệ thống
    """
    st.subheader("Sử dụng tài nguyên theo thời gian")
    
    # Tạo dữ liệu giả lập cho biểu đồ
    # Trong thực tế, dữ liệu này sẽ được lấy từ cơ sở dữ liệu
    chart_times = [datetime.datetime.now() - timedelta(minutes=i) for i in range(30, 0, -1)]
    
    # CPU Usage
    cpu_data = []
    for _ in range(30):
        base_value = system_summary['cpu']['usage']
        variation = np.random.normal(0, 5)  # Random variation
        cpu_value = max(0, min(100, base_value + variation))  # Keep between 0-100
        cpu_data.append(cpu_value)
    
    # Memory Usage
    memory_data = []
    for _ in range(30):
        base_value = system_summary['memory']['usage']
        variation = np.random.normal(0, 3)  # Less variation for memory
        memory_value = max(0, min(100, base_value + variation))  # Keep between 0-100
        memory_data.append(memory_value)
    
    # Network Latency
    latency_data = []
    for _ in range(30):
        base_value = system_summary['network']['latency']
        variation = np.random.normal(0, 10)  # More variation for latency
        latency_value = max(0, base_value + variation)  # Keep positive
        latency_data.append(latency_value)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': chart_times,
        'CPU (%)': cpu_data,
        'Memory (%)': memory_data,
        'Network (ms)': latency_data
    })
    
    # Sử dụng tabs để hiển thị các biểu đồ
    tab1, tab2, tab3 = st.tabs(["CPU & Memory", "Network", "Combined"])
    
    with tab1:
        fig = px.line(df, x="time", y=["CPU (%)", "Memory (%)"],
                     title="CPU & Memory Usage Over Time",
                     labels={"value": "Usage (%)", "time": "Time"},
                     line_shape="spline")
        
        # Thêm vùng cảnh báo
        fig.add_shape(
            type="rect",
            x0=df["time"].min(),
            x1=df["time"].max(),
            y0=70,
            y1=100,
            fillcolor="rgba(255,0,0,0.1)",
            line=dict(width=0),
            layer="below"
        )
        
        fig.update_layout(
            height=400,
            legend_title_text="Metric",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.line(df, x="time", y="Network (ms)",
                     title="Network Latency Over Time",
                     labels={"Network (ms)": "Latency (ms)", "time": "Time"},
                     line_shape="spline")
        
        # Thêm vùng cảnh báo
        fig.add_shape(
            type="rect",
            x0=df["time"].min(),
            x1=df["time"].max(),
            y0=100,
            y1=df["Network (ms)"].max() * 1.2,
            fillcolor="rgba(255,0,0,0.1)",
            line=dict(width=0),
            layer="below"
        )
        
        fig.update_layout(
            height=400,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Combined chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add CPU and Memory lines
        fig.add_trace(
            go.Scatter(x=df["time"], y=df["CPU (%)"], name="CPU (%)"),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=df["time"], y=df["Memory (%)"], name="Memory (%)"),
            secondary_y=False
        )
        
        # Add Network line on secondary y-axis
        fig.add_trace(
            go.Scatter(x=df["time"], y=df["Network (ms)"], name="Network (ms)"),
            secondary_y=True
        )
        
        # Set titles
        fig.update_layout(
            title_text="System Resources Over Time",
            height=400,
            hovermode="x unified"
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Usage (%)", secondary_y=False)
        fig.update_yaxes(title_text="Latency (ms)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()

def create_exchange_api_health(system_summary: Dict[str, Any]):
    """
    Tạo phần sức khỏe API sàn giao dịch.
    
    Args:
        system_summary: Tóm tắt thông tin hệ thống
    """
    st.subheader("Sức khỏe API sàn giao dịch")
    
    # Dữ liệu sàn giao dịch từ system_summary
    exchange_data = system_summary.get("exchange", {})
    
    if not exchange_data:
        # Tạo dữ liệu mẫu nếu không có dữ liệu thực
        exchange_data = {
            "binance": {
                "api_name": "binance",
                "total_requests": 1250,
                "success_count": 1180,
                "error_count": 70,
                "success_rate": 94.4,
                "avg_latency": 87.5,
                "rate_limit_hits": 5,
                "authentication_errors": 0,
                "timeout_errors": 8
            },
            "bybit": {
                "api_name": "bybit",
                "total_requests": 825,
                "success_count": 798,
                "error_count": 27,
                "success_rate": 96.7,
                "avg_latency": 120.3,
                "rate_limit_hits": 2,
                "authentication_errors": 1,
                "timeout_errors": 3
            }
        }
    
    # Hiển thị bảng thông tin
    exchange_info = []
    for name, data in exchange_data.items():
        # Xác định trạng thái
        status = "Tốt"
        status_color = "green"
        
        if data.get("success_rate", 100) < 90:
            status = "Sự cố"
            status_color = "red"
        elif data.get("success_rate", 100) < 95:
            status = "Cảnh báo"
            status_color = "orange"
        
        exchange_info.append({
            "Sàn": name.capitalize(),
            "Tổng yêu cầu": data.get("total_requests", 0),
            "Tỷ lệ thành công": f"{data.get('success_rate', 100):.1f}%",
            "Độ trễ TB": f"{data.get('avg_latency', 0):.1f} ms",
            "Rate Limit": data.get("rate_limit_hits", 0),
            "Timeout": data.get("timeout_errors", 0),
            "Trạng thái": f"<span style='color:{status_color}'>{status}</span>"
        })
    
    # Tạo DataFrame và hiển thị
    if exchange_info:
        df = pd.DataFrame(exchange_info)
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # Hiển thị biểu đồ cho mỗi sàn
        for name, data in exchange_data.items():
            with st.expander(f"Chi tiết sàn {name.capitalize()}"):
                # Tạo 2 cột
                col1, col2 = st.columns(2)
                
                with col1:
                    # Biểu đồ tỷ lệ thành công/lỗi
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=["Thành công", "Lỗi"],
                            values=[data.get("success_count", 0), data.get("error_count", 0)],
                            hole=.4,
                            marker_colors=["#28a745", "#dc3545"]
                        )
                    ])
                    
                    fig.update_layout(
                        title_text=f"Tỷ lệ yêu cầu {name.capitalize()}",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Biểu đồ phân loại lỗi
                    error_types = {
                        "Rate Limit": data.get("rate_limit_hits", 0),
                        "Authentication": data.get("authentication_errors", 0),
                        "Timeout": data.get("timeout_errors", 0),
                        "Khác": data.get("error_count", 0) - 
                                data.get("rate_limit_hits", 0) - 
                                data.get("authentication_errors", 0) - 
                                data.get("timeout_errors", 0)
                    }
                    
                    # Loại bỏ các loại lỗi có giá trị 0
                    error_types = {k: v for k, v in error_types.items() if v > 0}
                    
                    if error_types:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(error_types.keys()),
                                y=list(error_types.values()),
                                marker_color="#dc3545"
                            )
                        ])
                        
                        fig.update_layout(
                            title_text=f"Phân loại lỗi {name.capitalize()}",
                            height=300,
                            yaxis_title="Số lượng"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Không có thông tin lỗi chi tiết.")
    else:
        st.info("Không có dữ liệu API sàn giao dịch.")
    
    st.divider()

def create_log_activity_section():
    """Tạo phần hoạt động logs."""
    st.subheader("Hoạt động Logs Gần Đây")
    
    # Trong thực tế, logs sẽ được lấy từ hệ thống log
    # Đây là ví dụ dữ liệu giả lập
    sample_logs = [
        {"timestamp": datetime.datetime.now() - timedelta(minutes=5), 
         "level": "INFO", 
         "component": "system_monitor", 
         "message": "Hệ thống hoạt động bình thường"},
        {"timestamp": datetime.datetime.now() - timedelta(minutes=15), 
         "level": "WARNING", 
         "component": "exchange_api", 
         "message": "Độ trễ API tăng trên sàn Binance"},
        {"timestamp": datetime.datetime.now() - timedelta(minutes=45), 
         "level": "ERROR", 
         "component": "order_execution", 
         "message": "Lỗi thực thi lệnh trên cặp BTC/USDT"},
        {"timestamp": datetime.datetime.now() - timedelta(hours=2), 
         "level": "INFO", 
         "component": "model_training", 
         "message": "Hoàn thành huấn luyện mô hình DQN mới"},
        {"timestamp": datetime.datetime.now() - timedelta(hours=4), 
         "level": "WARNING", 
         "component": "risk_management", 
         "message": "Đạt ngưỡng rủi ro 80% cho cặp ETH/USDT"},
        {"timestamp": datetime.datetime.now() - timedelta(hours=6), 
         "level": "INFO", 
         "component": "system_startup", 
         "message": "Hệ thống khởi động thành công với 10 cặp giao dịch"}
    ]
    
    # Tạo DataFrame
    logs_df = pd.DataFrame(sample_logs)
    
    # Format thời gian
    logs_df["timestamp"] = logs_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Định nghĩa màu cho cấp độ log
    def log_color(level):
        colors = {
            "INFO": "green",
            "WARNING": "orange",
            "ERROR": "red",
            "CRITICAL": "darkred"
        }
        return f"<span style='color:{colors.get(level, 'black')}'>{level}</span>"
    
    # Áp dụng màu
    logs_df["level"] = logs_df["level"].apply(log_color)
    
    # Hiển thị bảng logs
    st.markdown(logs_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Nút để xem thêm logs
    if st.button("Xem thêm logs"):
        st.info("Chức năng xem logs chi tiết sẽ được hiển thị ở đây.")
    
    st.divider()

def create_active_trades_section():
    """Tạo phần giao dịch đang hoạt động."""
    st.subheader("Giao dịch đang hoạt động")
    
    # Trong thực tế, thông tin giao dịch sẽ được lấy từ hệ thống quản lý giao dịch
    # Đây là ví dụ dữ liệu giả lập
    sample_trades = [
        {"symbol": "BTC/USDT", "type": "LONG", "entry_price": 57850.25, "current_price": 58100.50, "profit_loss": 0.43, "time": "2h 15m"},
        {"symbol": "ETH/USDT", "type": "SHORT", "entry_price": 3245.75, "current_price": 3210.25, "profit_loss": 1.09, "time": "45m"},
        {"symbol": "SOL/USDT", "type": "LONG", "entry_price": 142.30, "current_price": 140.85, "profit_loss": -1.02, "time": "3h 30m"},
        {"symbol": "XRP/USDT", "type": "LONG", "entry_price": 0.5725, "current_price": 0.5812, "profit_loss": 1.52, "time": "1h 10m"}
    ]
    
    # Tạo DataFrame
    trades_df = pd.DataFrame(sample_trades)
    
    # Định dạng P&L với màu sắc
    def format_pnl(pnl):
        color = "green" if pnl >= 0 else "red"
        return f"<span style='color:{color}'>{pnl:+.2f}%</span>"
    
    trades_df["profit_loss"] = trades_df["profit_loss"].apply(format_pnl)
    
    # Định dạng loại giao dịch
    def format_type(type_str):
        color = "green" if type_str == "LONG" else "red"
        return f"<span style='color:{color}'>{type_str}</span>"
    
    trades_df["type"] = trades_df["type"].apply(format_type)
    
    # Hiển thị bảng giao dịch
    st.markdown(trades_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Liên kết tới trang giao dịch chi tiết
    st.markdown("[Xem tất cả giao dịch đang hoạt động →](#)")
    
    st.divider()

def create_system_actions():
    """Tạo phần hành động hệ thống."""
    st.subheader("Hành động hệ thống")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Làm mới dữ liệu", type="primary"):
            st.rerun()
    
    with col2:
        if st.button("Kiểm tra sức khỏe", help="Chạy kiểm tra sức khỏe hệ thống toàn diện"):
            st.info("Đang chạy kiểm tra sức khỏe hệ thống...")
            with st.spinner("Đang kiểm tra..."):
                # Giả lập thời gian xử lý
                time.sleep(2)
            st.success("Kiểm tra sức khỏe hoàn tất. Không phát hiện vấn đề.")
    
    with col3:
        if st.button("Xóa bộ nhớ đệm", help="Xóa bộ nhớ đệm hệ thống"):
            st.info("Đang xóa bộ nhớ đệm hệ thống...")
            with st.spinner("Đang xóa..."):
                # Giả lập thời gian xử lý
                time.sleep(1)
            st.success("Đã xóa bộ nhớ đệm thành công.")

def main():
    """Hàm chính của trang giám sát hệ thống."""
    
    # Render sidebar
    sidebar, selected_page = render_sidebar("system_monitor")
    
    # Set tiêu đề trang
    st.title("Giám sát hệ thống")
    
    # Hiển thị thời gian hiện tại
    st.markdown(f"**Cập nhật lần cuối:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Tạo các phần nội dung
    system_summary = create_system_overview()
    create_system_health_cards(system_summary)
    
    # Chia thành 2 tab chính
    tab1, tab2 = st.tabs(["Tài nguyên hệ thống", "Giao dịch & Logs"])
    
    with tab1:
        create_system_info_section(system_summary)
        create_resource_usage_charts(system_summary)
        create_exchange_api_health(system_summary)
    
    with tab2:
        create_active_trades_section()
        create_log_activity_section()
    
    # Phần hành động hệ thống
    create_system_actions()

if __name__ == "__main__":
    main()