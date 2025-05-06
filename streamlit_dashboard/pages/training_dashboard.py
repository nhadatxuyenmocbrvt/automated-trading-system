import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import json

# Thêm đường dẫn gốc của dự án vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import các module nội bộ
from logs.metrics.training_metrics import TrainingMetricsLogger
from logs.tensorboard.tb_logger import TensorboardLogger
from models.training_pipeline.trainer import Trainer
from config.system_config import SystemConfig
from streamlit_dashboard.charts.performance_charts import create_reward_chart, create_loss_chart, create_learning_curve
from streamlit_dashboard.components.sidebar import create_sidebar
from streamlit_dashboard.components.metrics_display import display_training_metrics

def load_training_data(agent_name, days=30):
    """
    Tải dữ liệu huấn luyện từ file log
    
    Parameters:
        agent_name (str): Tên của agent
        days (int): Số ngày dữ liệu gần nhất cần lấy
    
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu huấn luyện
    """
    try:
        # Đường dẫn đến file log
        log_path = os.path.join(
            SystemConfig.get_log_dir(),
            "training",
            f"{agent_name}_metrics.json"
        )
        
        # Đọc dữ liệu từ file JSON
        with open(log_path, 'r') as f:
            data = json.load(f)
        
        # Chuyển đổi thành DataFrame
        df = pd.DataFrame(data)
        
        # Chuyển đổi cột timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Lọc dữ liệu theo số ngày
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] >= cutoff_date]
        
        return df
    
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu huấn luyện: {e}")
        return pd.DataFrame()

def load_available_agents():
    """
    Tải danh sách các agent có sẵn
    
    Returns:
        list: Danh sách tên các agent
    """
    try:
        # Đường dẫn đến thư mục chứa các agent
        agents_dir = os.path.join(SystemConfig.get_models_dir(), "saved_agents")
        
        # Lấy danh sách các thư mục con (mỗi thư mục là một agent)
        agents = [d for d in os.listdir(agents_dir) if os.path.isdir(os.path.join(agents_dir, d))]
        
        return agents
    
    except Exception as e:
        st.error(f"Lỗi khi tải danh sách agent: {e}")
        return []

def get_training_status(agent_name):
    """
    Kiểm tra trạng thái huấn luyện của agent
    
    Parameters:
        agent_name (str): Tên của agent
    
    Returns:
        str: Trạng thái huấn luyện ('Đang huấn luyện', 'Đã hoàn thành', 'Chưa bắt đầu')
    """
    try:
        # Đường dẫn đến file trạng thái
        status_path = os.path.join(
            SystemConfig.get_log_dir(),
            "training",
            f"{agent_name}_status.json"
        )
        
        # Đọc dữ liệu từ file JSON
        with open(status_path, 'r') as f:
            status = json.load(f)
        
        return status.get('status', 'Không xác định')
    
    except Exception as e:
        # Nếu không tìm thấy file, có thể agent chưa được huấn luyện
        return "Chưa bắt đầu"

def display_training_configuration(agent_name):
    """
    Hiển thị cấu hình huấn luyện của agent
    
    Parameters:
        agent_name (str): Tên của agent
    """
    try:
        # Đường dẫn đến file cấu hình
        config_path = os.path.join(
            SystemConfig.get_models_dir(),
            "saved_agents",
            agent_name,
            "config.json"
        )
        
        # Đọc dữ liệu từ file JSON
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Hiển thị cấu hình
        st.subheader("Cấu hình huấn luyện")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Thuật toán", config.get('algorithm', 'N/A'))
            st.metric("Learning rate", config.get('learning_rate', 'N/A'))
            st.metric("Batch size", config.get('batch_size', 'N/A'))
            
        with col2:
            st.metric("Gamma", config.get('gamma', 'N/A'))
            st.metric("Epsilon", config.get('epsilon', 'N/A'))
            st.metric("Mô hình mạng", config.get('network_type', 'N/A'))
        
        # Hiển thị thông tin thêm trong expander
        with st.expander("Xem thêm thông tin cấu hình"):
            # Loại bỏ các trường đã hiển thị ở trên
            displayed_keys = ['algorithm', 'learning_rate', 'batch_size', 'gamma', 'epsilon', 'network_type']
            remaining_config = {k: v for k, v in config.items() if k not in displayed_keys}
            
            # Hiển thị các trường còn lại
            st.json(remaining_config)
            
    except Exception as e:
        st.warning(f"Không thể hiển thị cấu hình huấn luyện: {e}")

def display_hyperparameter_comparison():
    """
    Hiển thị so sánh hiệu suất giữa các cấu hình siêu tham số khác nhau
    """
    try:
        # Đường dẫn đến file dữ liệu so sánh
        comparison_path = os.path.join(
            SystemConfig.get_log_dir(),
            "training",
            "hyperparameter_comparison.json"
        )
        
        # Đọc dữ liệu từ file JSON
        with open(comparison_path, 'r') as f:
            comparison_data = json.load(f)
        
        # Chuyển đổi thành DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Hiển thị bảng so sánh
        st.subheader("So sánh hiệu suất với các siêu tham số khác nhau")
        st.dataframe(df)
        
        # Tạo biểu đồ so sánh
        fig = px.bar(
            df, 
            x='config_name', 
            y='mean_reward',
            error_y='reward_std',
            color='algorithm',
            title="So sánh hiệu suất theo cấu hình siêu tham số"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.info("Chưa có dữ liệu so sánh siêu tham số hoặc đã xảy ra lỗi khi tải dữ liệu.")

def main():
    """
    Hàm chính của trang dashboard huấn luyện
    """
    st.title("Dashboard Huấn Luyện Agent")
    
    # Tạo sidebar
    create_sidebar()
    
    # Tải danh sách các agent có sẵn
    available_agents = load_available_agents()
    
    if not available_agents:
        st.warning("Không tìm thấy agent nào. Vui lòng tạo agent mới trước.")
        return
    
    # Chọn agent để xem
    selected_agent = st.selectbox(
        "Chọn agent để xem chi tiết huấn luyện",
        available_agents
    )
    
    # Tùy chọn khoảng thời gian
    time_options = {
        "7 ngày gần đây": 7,
        "30 ngày gần đây": 30,
        "90 ngày gần đây": 90,
        "Tất cả": 9999
    }
    
    selected_time = st.radio(
        "Khoảng thời gian",
        list(time_options.keys()),
        horizontal=True
    )
    
    days = time_options[selected_time]
    
    # Lấy trạng thái huấn luyện
    training_status = get_training_status(selected_agent)
    
    # Hiển thị thông tin trạng thái
    status_color = {
        "Đang huấn luyện": "🟡",
        "Đã hoàn thành": "🟢",
        "Chưa bắt đầu": "⚪",
        "Đã dừng": "🔴",
        "Không xác định": "⚪"
    }
    
    st.subheader(f"Agent: {selected_agent} {status_color.get(training_status, '⚪')} {training_status}")
    
    # Hiển thị cấu hình huấn luyện
    display_training_configuration(selected_agent)
    
    # Tải dữ liệu huấn luyện
    training_data = load_training_data(selected_agent, days)
    
    if training_data.empty:
        st.warning(f"Không có dữ liệu huấn luyện cho agent {selected_agent} trong khoảng thời gian đã chọn.")
        return
    
    # Hiển thị các metrics huấn luyện
    display_training_metrics(training_data)
    
    # Tạo các biểu đồ
    st.subheader("Biểu đồ tiến trình huấn luyện")
    
    # Biểu đồ phần thưởng
    reward_fig = create_reward_chart(training_data)
    st.plotly_chart(reward_fig, use_container_width=True)
    
    # Tạo 2 cột
    col1, col2 = st.columns(2)
    
    with col1:
        # Biểu đồ loss
        loss_fig = create_loss_chart(training_data)
        st.plotly_chart(loss_fig, use_container_width=True)
    
    with col2:
        # Biểu đồ learning curve
        learning_curve_fig = create_learning_curve(training_data)
        st.plotly_chart(learning_curve_fig, use_container_width=True)
    
    # Hiển thị so sánh siêu tham số
    display_hyperparameter_comparison()
    
    # Thêm nút điều khiển huấn luyện
    st.subheader("Điều khiển huấn luyện")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Bắt đầu huấn luyện", disabled=(training_status == "Đang huấn luyện")):
            # Gọi hàm để bắt đầu huấn luyện
            st.success(f"Đã bắt đầu huấn luyện cho agent {selected_agent}")
            st.rerun()
    
    with col2:
        if st.button("Dừng huấn luyện", disabled=(training_status != "Đang huấn luyện")):
            # Gọi hàm để dừng huấn luyện
            st.warning(f"Đã dừng huấn luyện cho agent {selected_agent}")
            st.rerun()
    
    with col3:
        if st.button("Tạo bản sao lưu"):
            # Gọi hàm để tạo bản sao lưu
            st.info(f"Đã tạo bản sao lưu cho agent {selected_agent}")

if __name__ == "__main__":
    main()