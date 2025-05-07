"""
Thành phần thanh bên cho dashboard Streamlit.
File này cung cấp các chức năng tạo và quản lý thanh bên (sidebar),
bao gồm điều hướng, lọc, và tùy chọn cấu hình cho dashboard.
"""

import os
import streamlit as st
import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
import pandas as pd
from pathlib import Path

# Import các module từ hệ thống
from config.constants import SystemStatus, OrderStatus, Exchange, Timeframe, Indicator, AgentType
from config.env import get_env, set_env
from logs.logger import get_system_logger
from streamlit_dashboard.components.controls import create_date_filter, create_dropdown_filter

# Khởi tạo logger
logger = get_system_logger("dashboard_sidebar")


class SidebarManager:
    """
    Lớp quản lý thanh bên cho dashboard Streamlit.
    Cung cấp các phương thức tạo và quản lý các thành phần trong thanh bên.
    """
    
    def __init__(self, 
                app_name: str = "ATS Dashboard", 
                page_mapping: Optional[Dict[str, str]] = None,
                logo_path: Optional[str] = None):
        """
        Khởi tạo SidebarManager.
        
        Args:
            app_name: Tên ứng dụng
            page_mapping: Dict ánh xạ từ tên hiển thị đến tên thực tế của trang
            logo_path: Đường dẫn đến logo (tương đối từ thư mục gốc)
        """
        self.app_name = app_name
        
        # Ánh xạ trang mặc định
        self.page_mapping = page_mapping or {
            "Tổng quan hệ thống": "system_monitor",
            "Theo dõi huấn luyện": "training_dashboard",
            "Theo dõi giao dịch": "trading_dashboard",
            "Cài đặt hệ thống": "settings"
        }
        
        # Đường dẫn logo
        project_dir = Path(__file__).parent.parent.parent
        self.logo_path = logo_path or project_dir / "streamlit_dashboard" / "assets" / "logo.png"
        
        # Trạng thái hệ thống
        self.system_status = self._get_system_status()
        
        # Môi trường hiện tại
        self.current_env = get_env("TRADING_ENV", "development")
        
        # Thông tin người dùng
        self.user_info = self._get_user_info()
    
    def _get_system_status(self) -> SystemStatus:
        """
        Lấy trạng thái hiện tại của hệ thống.
        
        Returns:
            Trạng thái hệ thống
        """
        try:
            # Trong thực tế cần lấy trạng thái từ dịch vụ quản lý hệ thống
            # Đây chỉ là một giả lập đơn giản
            status_str = get_env("SYSTEM_STATUS", SystemStatus.RUNNING.value)
            return SystemStatus(status_str)
        except Exception as e:
            logger.error(f"Lỗi khi lấy trạng thái hệ thống: {str(e)}")
            return SystemStatus.ERROR
    
    def _get_user_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin người dùng hiện tại.
        
        Returns:
            Dict thông tin người dùng
        """
        # Trong thực tế, lấy thông tin từ session hoặc DB
        # Đây là thông tin mặc định
        return {
            "username": get_env("CURRENT_USER", "admin"),
            "role": "admin",
            "last_login": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _get_available_coins(self) -> List[str]:
        """
        Lấy danh sách các coin có sẵn.
        
        Returns:
            Danh sách các coin
        """
        # Trong thực tế, lấy từ dịch vụ dữ liệu hoặc cấu hình
        asset_list = get_env("ASSET_LIST", "").split(",")
        coins = []
        
        for asset in asset_list:
            if "/" in asset:
                base_coin = asset.split("/")[0]
                if base_coin not in coins:
                    coins.append(base_coin)
        
        # Thêm một số coin phổ biến nếu danh sách rỗng
        if not coins:
            coins = ["BTC", "ETH", "BNB", "XRP", "SOL", "ADA", "DOT", "DOGE"]
        
        return coins
    
    def _get_timeframe_options(self) -> List[str]:
        """
        Lấy danh sách các tùy chọn khung thời gian.
        
        Returns:
            Danh sách các khung thời gian
        """
        return [tf.value for tf in Timeframe]
    
    def _get_exchange_options(self) -> List[str]:
        """
        Lấy danh sách các sàn giao dịch đang hoạt động.
        
        Returns:
            Danh sách các sàn giao dịch
        """
        from config.constants import ACTIVE_EXCHANGES
        return [ex.value for ex in ACTIVE_EXCHANGES]
    
    def _get_agent_options(self) -> List[str]:
        """
        Lấy danh sách các loại agent.
        
        Returns:
            Danh sách các loại agent
        """
        return [agent.value for agent in AgentType]
    
    def _get_indicator_options(self) -> List[str]:
        """
        Lấy danh sách các chỉ báo kỹ thuật.
        
        Returns:
            Danh sách các chỉ báo
        """
        return [ind.value for ind in Indicator]
    
    def render_header(self):
        """Hiển thị phần đầu của thanh bên."""
        st.sidebar.image(self.logo_path, width=100)
        st.sidebar.title(self.app_name)
        
        # Hiển thị trạng thái
        status_color = {
            SystemStatus.RUNNING: "green",
            SystemStatus.INITIALIZING: "blue",
            SystemStatus.PAUSED: "orange",
            SystemStatus.STOPPING: "red",
            SystemStatus.STOPPED: "red",
            SystemStatus.ERROR: "red",
            SystemStatus.MAINTENANCE: "orange"
        }
        
        status = self.system_status
        color = status_color.get(status, "gray")
        
        st.sidebar.markdown(
            f"""
            <div style='display: flex; align-items: center;'>
                <div style='width: 15px; height: 15px; border-radius: 50%; 
                           background-color: {color}; margin-right: 10px;'></div>
                <div>Trạng thái: <b>{status.value}</b></div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Hiển thị môi trường
        env_color = {
            "development": "blue",
            "testing": "orange",
            "production": "green"
        }
        
        env_color_value = env_color.get(self.current_env, "gray")
        
        st.sidebar.markdown(
            f"""
            <div style='display: flex; align-items: center;'>
                <div style='width: 15px; height: 15px; border-radius: 50%; 
                           background-color: {env_color_value}; margin-right: 10px;'></div>
                <div>Môi trường: <b>{self.current_env.upper()}</b></div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.sidebar.divider()
    
    def render_navigation(self) -> str:
        """
        Hiển thị phần điều hướng của thanh bên.
        
        Returns:
            Tên trang được chọn
        """
        st.sidebar.subheader("Điều hướng")
        
        # Tạo radio buttons cho điều hướng
        page_names = list(self.page_mapping.keys())
        selected_page_name = st.sidebar.radio("", page_names, index=0)
        
        # Lấy tên thực tế của trang từ ánh xạ
        selected_page = self.page_mapping.get(selected_page_name, "system_monitor")
        
        st.sidebar.divider()
        
        return selected_page
    
    def render_system_filters(self, on_change: Optional[Callable] = None):
        """
        Hiển thị bộ lọc hệ thống trong thanh bên.
        
        Args:
            on_change: Hàm callback khi giá trị thay đổi
        """
        st.sidebar.subheader("Bộ lọc hệ thống")
        
        # Bộ lọc sàn giao dịch
        exchange_options = self._get_exchange_options()
        selected_exchange = st.sidebar.selectbox(
            "Sàn giao dịch",
            options=exchange_options,
            index=0,
            on_change=on_change if on_change else None
        )
        
        # Lưu vào session state
        st.session_state.selected_exchange = selected_exchange
        
        # Bộ lọc khung thời gian
        timeframe_options = self._get_timeframe_options()
        selected_timeframe = st.sidebar.selectbox(
            "Khung thời gian",
            options=timeframe_options,
            index=timeframe_options.index("1h") if "1h" in timeframe_options else 0,
            on_change=on_change if on_change else None
        )
        
        # Lưu vào session state
        st.session_state.selected_timeframe = selected_timeframe
        
        # Bộ lọc coin
        coin_options = self._get_available_coins()
        selected_coin = st.sidebar.selectbox(
            "Coin",
            options=coin_options,
            index=0,
            on_change=on_change if on_change else None
        )
        
        # Lưu vào session state
        st.session_state.selected_coin = selected_coin
        
        st.sidebar.divider()
    
    def render_date_filter(self, on_change: Optional[Callable] = None):
        """
        Hiển thị bộ lọc ngày trong thanh bên.
        
        Args:
            on_change: Hàm callback khi giá trị thay đổi
        """
        st.sidebar.subheader("Bộ lọc thời gian")
        
        # Tùy chọn nhanh
        date_options = {
            "Hôm nay": 0,
            "7 ngày qua": 7, 
            "30 ngày qua": 30,
            "90 ngày qua": 90,
            "Tùy chỉnh": -1
        }
        
        selected_date_option = st.sidebar.selectbox(
            "Khoảng thời gian",
            options=list(date_options.keys()),
            index=1,  # 7 ngày qua
            on_change=on_change if on_change else None
        )
        
        # Lưu vào session state
        st.session_state.selected_date_option = selected_date_option
        
        # Xử lý tùy chọn tùy chỉnh
        days = date_options[selected_date_option]
        
        if days >= 0:
            end_date = datetime.datetime.now().date()
            start_date = end_date - datetime.timedelta(days=days)
            
            # Hiển thị thông tin ngày đã chọn
            st.sidebar.markdown(f"**Từ:** {start_date.strftime('%d/%m/%Y')}")
            st.sidebar.markdown(f"**Đến:** {end_date.strftime('%d/%m/%Y')}")
            
            # Lưu vào session state
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date
        else:
            # Cho phép người dùng chọn ngày tùy chỉnh
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "Từ ngày",
                    value=datetime.datetime.now().date() - datetime.timedelta(days=7),
                    on_change=on_change if on_change else None
                )
            
            with col2:
                end_date = st.date_input(
                    "Đến ngày",
                    value=datetime.datetime.now().date(),
                    on_change=on_change if on_change else None
                )
            
            # Lưu vào session state
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date
        
        st.sidebar.divider()
    
    def render_training_filters(self, on_change: Optional[Callable] = None):
        """
        Hiển thị bộ lọc huấn luyện trong thanh bên.
        
        Args:
            on_change: Hàm callback khi giá trị thay đổi
        """
        st.sidebar.subheader("Bộ lọc huấn luyện")
        
        # Bộ lọc loại agent
        agent_options = self._get_agent_options()
        selected_agent = st.sidebar.selectbox(
            "Loại agent",
            options=agent_options,
            index=0,
            on_change=on_change if on_change else None
        )
        
        # Lưu vào session state
        st.session_state.selected_agent = selected_agent
        
        # Danh sách mô hình được huấn luyện
        # Trong thực tế, lấy từ dịch vụ quản lý mô hình
        model_options = ["latest", "best_reward", "best_sharpe", "stable"]
        selected_model = st.sidebar.selectbox(
            "Mô hình",
            options=model_options,
            index=0,
            on_change=on_change if on_change else None
        )
        
        # Lưu vào session state
        st.session_state.selected_model = selected_model
        
        st.sidebar.divider()
    
    def render_trading_filters(self, on_change: Optional[Callable] = None):
        """
        Hiển thị bộ lọc giao dịch trong thanh bên.
        
        Args:
            on_change: Hàm callback khi giá trị thay đổi
        """
        st.sidebar.subheader("Bộ lọc giao dịch")
        
        # Bộ lọc trạng thái lệnh
        order_status_options = [status.value for status in OrderStatus]
        
        selected_order_status = st.sidebar.multiselect(
            "Trạng thái lệnh",
            options=order_status_options,
            default=["filled", "open", "partially_filled"],
            on_change=on_change if on_change else None
        )
        
        # Lưu vào session state
        st.session_state.selected_order_status = selected_order_status
        
        # Bộ lọc chỉ báo
        indicator_options = self._get_indicator_options()
        selected_indicators = st.sidebar.multiselect(
            "Chỉ báo kỹ thuật",
            options=indicator_options,
            default=["sma", "ema", "rsi"],
            on_change=on_change if on_change else None
        )
        
        # Lưu vào session state
        st.session_state.selected_indicators = selected_indicators
        
        st.sidebar.divider()
    
    def render_system_controls(self):
        """Hiển thị các điều khiển hệ thống trong thanh bên."""
        st.sidebar.subheader("Điều khiển hệ thống")
        
        # Kiểm tra quyền admin
        if self.user_info.get("role") == "admin":
            # Các nút điều khiển cho admin
            system_action = st.sidebar.selectbox(
                "Hành động hệ thống",
                options=["Chọn hành động", "Khởi động", "Tạm dừng", "Tiếp tục", "Dừng", "Khởi động lại", "Bảo trì"]
            )
            
            if system_action != "Chọn hành động":
                if st.sidebar.button(f"Thực hiện: {system_action}"):
                    self._handle_system_action(system_action)
            
            # Tùy chọn nâng cao
            with st.sidebar.expander("Tùy chọn nâng cao"):
                # Tùy chỉnh số lượng vị thế mở tối đa
                max_positions = st.number_input(
                    "Số vị thế mở tối đa",
                    min_value=1,
                    max_value=50,
                    value=int(get_env("MAX_OPEN_POSITIONS", "5"))
                )
                
                # Tùy chỉnh đòn bẩy mặc định
                default_leverage = st.number_input(
                    "Đòn bẩy mặc định",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(get_env("DEFAULT_LEVERAGE", "1.0")),
                    step=0.1
                )
                
                # Tùy chỉnh rủi ro mỗi giao dịch
                risk_per_trade = st.number_input(
                    "Rủi ro mỗi giao dịch (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=float(get_env("RISK_PER_TRADE", "2.0")) * 100,
                    step=0.1
                ) / 100.0
                
                if st.button("Lưu cấu hình"):
                    self._save_advanced_config(max_positions, default_leverage, risk_per_trade)
        else:
            # Các điều khiển cơ bản cho người dùng thông thường
            st.sidebar.info("Chỉ người dùng admin có thể điều khiển hệ thống.")
        
        st.sidebar.divider()
    
    def render_user_section(self):
        """Hiển thị phần thông tin người dùng trong thanh bên."""
        st.sidebar.subheader("Thông tin người dùng")
        
        st.sidebar.text(f"Người dùng: {self.user_info['username']}")
        st.sidebar.text(f"Vai trò: {self.user_info['role']}")
        st.sidebar.text(f"Đăng nhập lần cuối: {self.user_info['last_login']}")
        
        # Nút đăng xuất
        if st.sidebar.button("Đăng xuất"):
            self._handle_logout()
    
    def _handle_system_action(self, action: str):
        """
        Xử lý hành động hệ thống.
        
        Args:
            action: Hành động được chọn
        """
        action_mapping = {
            "Khởi động": SystemStatus.RUNNING,
            "Tạm dừng": SystemStatus.PAUSED,
            "Tiếp tục": SystemStatus.RUNNING,
            "Dừng": SystemStatus.STOPPED,
            "Khởi động lại": SystemStatus.INITIALIZING,
            "Bảo trì": SystemStatus.MAINTENANCE
        }
        
        if action in action_mapping:
            new_status = action_mapping[action]
            
            # Trong thực tế, gọi API để thay đổi trạng thái hệ thống
            # Đây chỉ là mô phỏng
            set_env("SYSTEM_STATUS", new_status.value)
            self.system_status = new_status
            
            # Hiển thị thông báo
            st.sidebar.success(f"Đã thực hiện hành động: {action}")
            logger.info(f"Thay đổi trạng thái hệ thống thành {new_status.value}")
            
            # Reload ứng dụng để cập nhật trạng thái
            st.experimental_rerun()
    
    def _save_advanced_config(self, max_positions: int, default_leverage: float, risk_per_trade: float):
        """
        Lưu cấu hình nâng cao.
        
        Args:
            max_positions: Số vị thế mở tối đa
            default_leverage: Đòn bẩy mặc định
            risk_per_trade: Rủi ro mỗi giao dịch
        """
        # Lưu cấu hình vào biến môi trường
        set_env("MAX_OPEN_POSITIONS", str(max_positions))
        set_env("DEFAULT_LEVERAGE", str(default_leverage))
        set_env("RISK_PER_TRADE", str(risk_per_trade))
        
        # Hiển thị thông báo
        st.sidebar.success("Đã lưu cấu hình thành công!")
        logger.info("Đã cập nhật cấu hình nâng cao của hệ thống")
    
    def _handle_logout(self):
        """Xử lý đăng xuất."""
        # Trong thực tế, xóa session và chuyển đến trang đăng nhập
        st.session_state.clear()
        st.experimental_rerun()
    
    def render_footer(self):
        """Hiển thị phần chân của thanh bên."""
        st.sidebar.divider()
        
        current_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        st.sidebar.markdown(f"<small>Cập nhật lần cuối: {current_time}</small>", unsafe_allow_html=True)
        st.sidebar.markdown("<small>© 2025 Automated Trading System</small>", unsafe_allow_html=True)
    
    def render_full_sidebar(self, page: Optional[str] = None) -> str:
        """
        Hiển thị toàn bộ thanh bên.
        
        Args:
            page: Trang hiện tại, nếu None sẽ hiển thị điều hướng
            
        Returns:
            Trang được chọn (nếu page=None)
        """
        self.render_header()
        
        # Hiển thị điều hướng nếu không có trang cụ thể
        selected_page = page
        if page is None:
            selected_page = self.render_navigation()
        
        # Hiển thị các bộ lọc dựa trên trang
        if selected_page in ["system_monitor", "settings"]:
            self.render_system_filters()
        elif selected_page == "training_dashboard":
            self.render_system_filters()
            self.render_date_filter()
            self.render_training_filters()
        elif selected_page == "trading_dashboard":
            self.render_system_filters()
            self.render_date_filter()
            self.render_trading_filters()
        
        # Hiển thị điều khiển hệ thống
        self.render_system_controls()
        
        # Hiển thị thông tin người dùng
        self.render_user_section()
        
        # Hiển thị chân trang
        self.render_footer()
        
        return selected_page


# Hàm helper để tạo thanh bên
def create_sidebar() -> SidebarManager:
    """
    Tạo và cấu hình thanh bên.
    
    Returns:
        SidebarManager đã được cấu hình
    """
    sidebar = SidebarManager(app_name="ATS Dashboard")
    return sidebar


def render_sidebar(current_page: Optional[str] = None) -> Tuple[SidebarManager, str]:
    """
    Hiển thị thanh bên và trả về manager và trang được chọn.
    
    Args:
        current_page: Trang hiện tại (nếu có)
    
    Returns:
        Tuple (SidebarManager, selected_page)
    """
    sidebar = create_sidebar()
    selected_page = sidebar.render_full_sidebar(current_page)
    return sidebar, selected_page


# Chỉ chạy khi file này được chạy trực tiếp
if __name__ == "__main__":
    # Cài đặt các biến môi trường mô phỏng
    import sys
    import os
    
    # Thêm thư mục gốc vào sys.path
    project_dir = Path(__file__).parent.parent.parent
    sys.path.append(str(project_dir))
    
    # Mô phỏng môi trường
    os.environ["TRADING_ENV"] = "development"
    os.environ["SYSTEM_STATUS"] = "running"
    os.environ["MAX_OPEN_POSITIONS"] = "5"
    os.environ["DEFAULT_LEVERAGE"] = "1.0"
    os.environ["RISK_PER_TRADE"] = "0.02"
    
    # Tạo ứng dụng Streamlit mô phỏng
    st.set_page_config(page_title="ATS Dashboard", layout="wide")
    
    # Tạo và hiển thị thanh bên
    sidebar, selected_page = render_sidebar()
    
    # Hiển thị trang chính
    st.title(f"Trang chính: {selected_page}")
    st.write("Nội dung trang sẽ được hiển thị ở đây dựa trên trang được chọn.")
    
    # Hiển thị các giá trị đã chọn
    st.subheader("Các giá trị đã chọn")
    for key, value in st.session_state.items():
        if key.startswith("selected_"):
            st.write(f"{key}: {value}")