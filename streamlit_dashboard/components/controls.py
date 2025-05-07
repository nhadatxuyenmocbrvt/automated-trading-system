"""
Các thành phần điều khiển tái sử dụng cho dashboard Streamlit.
File này cung cấp các hàm và lớp để tạo và quản lý các điều khiển UI
nhất quán và dễ tùy chỉnh trong toàn bộ dashboard.
"""

import streamlit as st
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import altair as alt
from pathlib import Path

# Import các module từ hệ thống
from config.constants import (SystemStatus, OrderStatus, Exchange, Timeframe, 
                             Indicator, AgentType, RiskProfile)
from config.env import get_env, set_env
from logs.logger import get_system_logger

# Khởi tạo logger
logger = get_system_logger("dashboard_controls")

class ControlStyle:
    """Lớp định nghĩa các kiểu định dạng cho các điều khiển."""
    
    # Màu sắc chính
    PRIMARY_COLOR = "#1E88E5"
    SECONDARY_COLOR = "#26A69A"
    WARNING_COLOR = "#FFC107"
    DANGER_COLOR = "#EF5350"
    SUCCESS_COLOR = "#66BB6A"
    INFO_COLOR = "#29B6F6"
    
    # Các biến thể màu
    LIGHT_PRIMARY = "#BBDEFB"
    DARK_PRIMARY = "#1565C0"
    
    # Màu nền và văn bản
    BACKGROUND_COLOR = "#F5F5F5"
    TEXT_COLOR = "#212121"
    MUTED_TEXT_COLOR = "#757575"
    
    # CSS chung
    CARD_STYLE = """
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    background-color: white;
    margin-bottom: 1rem;
    """
    
    BUTTON_STYLE = f"""
    background-color: {PRIMARY_COLOR};
    color: white;
    border: none;
    border-radius: 0.25rem;
    padding: 0.5rem 1rem;
    font-weight: bold;
    cursor: pointer;
    transition: background-color 0.3s;
    """
    
    BUTTON_HOVER_STYLE = f"""
    background-color: {DARK_PRIMARY};
    """
    
    # Các kiểu đặc biệt
    METRIC_STYLE_POSITIVE = f"""
    color: {SUCCESS_COLOR};
    font-weight: bold;
    font-size: 1.5rem;
    """
    
    METRIC_STYLE_NEGATIVE = f"""
    color: {DANGER_COLOR};
    font-weight: bold;
    font-size: 1.5rem;
    """
    
    METRIC_STYLE_NEUTRAL = f"""
    color: {TEXT_COLOR};
    font-weight: bold;
    font-size: 1.5rem;
    """
    
    # Các thẻ (tags)
    TAG_STYLE = """
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 1rem;
    font-size: 0.75rem;
    font-weight: bold;
    margin-right: 0.5rem;
    """
    
    @classmethod
    def get_tag_style(cls, tag_type: str) -> str:
        """
        Lấy style cho tag dựa trên loại.
        
        Args:
            tag_type: Loại tag (success, warning, danger, info)
            
        Returns:
            CSS style cho tag
        """
        tag_styles = {
            'success': f"background-color: {cls.SUCCESS_COLOR}; color: white;",
            'warning': f"background-color: {cls.WARNING_COLOR}; color: #212121;",
            'danger': f"background-color: {cls.DANGER_COLOR}; color: white;",
            'info': f"background-color: {cls.INFO_COLOR}; color: white;",
            'primary': f"background-color: {cls.PRIMARY_COLOR}; color: white;",
            'secondary': f"background-color: {cls.SECONDARY_COLOR}; color: white;",
        }
        
        return cls.TAG_STYLE + tag_styles.get(tag_type, "")
    
    @classmethod
    def apply_custom_style(cls):
        """Áp dụng CSS tùy chỉnh cho dashboard."""
        st.markdown(
            f"""
            <style>
                .stButton > button {{{cls.BUTTON_STYLE}}}
                .stButton > button:hover {{{cls.BUTTON_HOVER_STYLE}}}
                .metric-positive {{{cls.METRIC_STYLE_POSITIVE}}}
                .metric-negative {{{cls.METRIC_STYLE_NEGATIVE}}}
                .metric-neutral {{{cls.METRIC_STYLE_NEUTRAL}}}
                .card {{{cls.CARD_STYLE}}}
                .tag-success {{{cls.get_tag_style('success')}}}
                .tag-warning {{{cls.get_tag_style('warning')}}}
                .tag-danger {{{cls.get_tag_style('danger')}}}
                .tag-info {{{cls.get_tag_style('info')}}}
                .tag-primary {{{cls.get_tag_style('primary')}}}
                .tag-secondary {{{cls.get_tag_style('secondary')}}}
            </style>
            """,
            unsafe_allow_html=True
        )


# Bộ lọc ngày và thời gian
def create_date_filter(
    key_prefix: str = "date",
    default_days: int = 7,
    allow_custom: bool = True,
    show_time: bool = False,
    on_change: Optional[Callable] = None
) -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Tạo bộ lọc ngày với các tùy chọn nhanh.
    
    Args:
        key_prefix: Tiền tố cho session state keys
        default_days: Số ngày mặc định
        allow_custom: Cho phép tùy chỉnh khoảng thời gian
        show_time: Hiển thị lựa chọn giờ phút
        on_change: Hàm callback khi giá trị thay đổi
        
    Returns:
        Tuple (start_datetime, end_datetime)
    """
    date_options = {
        "Hôm nay": 0,
        "Hôm qua": 1,
        "7 ngày qua": 7, 
        "30 ngày qua": 30,
        "90 ngày qua": 90,
        "Năm nay": 365,
    }
    
    if allow_custom:
        date_options["Tùy chỉnh"] = -1
    
    # Tạo khóa duy nhất cho session state
    option_key = f"{key_prefix}_option"
    start_key = f"{key_prefix}_start"
    end_key = f"{key_prefix}_end"
    
    # Khởi tạo giá trị mặc định nếu chưa có trong session state
    if option_key not in st.session_state:
        st.session_state[option_key] = list(date_options.keys())[2]  # 7 ngày qua
    
    if start_key not in st.session_state or end_key not in st.session_state:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=default_days)
        
        st.session_state[start_key] = start_date
        st.session_state[end_key] = end_date
    
    # Hàm cập nhật ngày khi lựa chọn thay đổi
    def update_dates():
        option = st.session_state[option_key]
        days = date_options[option]
        
        end_date = datetime.datetime.now()
        
        if days == 0:  # Hôm nay
            start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif days > 0:
            start_date = end_date - datetime.timedelta(days=days)
        else:
            # Tùy chỉnh - giữ nguyên giá trị hiện tại
            return
        
        st.session_state[start_key] = start_date
        st.session_state[end_key] = end_date
        
        # Gọi callback nếu có
        if on_change is not None:
            on_change()
    
    # Tạo dropdown cho tùy chọn
    selected_option = st.selectbox(
        "Khoảng thời gian",
        options=list(date_options.keys()),
        index=list(date_options.keys()).index(st.session_state[option_key]),
        key=option_key,
        on_change=update_dates
    )
    
    # Xử lý tùy chọn tùy chỉnh
    if selected_option == "Tùy chỉnh":
        cols = st.columns(2)
        
        with cols[0]:
            if show_time:
                start_date = st.datetime_input(
                    "Từ",
                    value=st.session_state[start_key],
                    key=f"{start_key}_input",
                    on_change=on_change
                )
            else:
                start_date = st.date_input(
                    "Từ",
                    value=st.session_state[start_key].date(),
                    key=f"{start_key}_input",
                    on_change=on_change
                )
                # Chuyển đổi từ date sang datetime
                if isinstance(start_date, datetime.date) and not isinstance(start_date, datetime.datetime):
                    start_date = datetime.datetime.combine(start_date, datetime.time.min)
            
            st.session_state[start_key] = start_date
        
        with cols[1]:
            if show_time:
                end_date = st.datetime_input(
                    "Đến",
                    value=st.session_state[end_key],
                    key=f"{end_key}_input",
                    on_change=on_change
                )
            else:
                end_date = st.date_input(
                    "Đến",
                    value=st.session_state[end_key].date(),
                    key=f"{end_key}_input",
                    on_change=on_change
                )
                # Chuyển đổi từ date sang datetime
                if isinstance(end_date, datetime.date) and not isinstance(end_date, datetime.datetime):
                    end_date = datetime.datetime.combine(end_date, datetime.time.max)
            
            st.session_state[end_key] = end_date
    else:
        # Sử dụng giá trị đã tính toán
        start_date = st.session_state[start_key]
        end_date = st.session_state[end_key]
    
    return start_date, end_date


# Bộ lọc dropdown
def create_dropdown_filter(
    label: str,
    options: List[Any],
    default_index: int = 0,
    key: Optional[str] = None,
    help_text: Optional[str] = None,
    on_change: Optional[Callable] = None,
    format_func: Optional[Callable[[Any], str]] = None
) -> Any:
    """
    Tạo bộ lọc dropdown.
    
    Args:
        label: Nhãn cho dropdown
        options: Danh sách các tùy chọn
        default_index: Chỉ số mặc định
        key: Khóa cho session state
        help_text: Văn bản trợ giúp
        on_change: Hàm callback khi giá trị thay đổi
        format_func: Hàm định dạng hiển thị tùy chọn
        
    Returns:
        Giá trị được chọn
    """
    return st.selectbox(
        label,
        options=options,
        index=default_index,
        key=key,
        help=help_text,
        on_change=on_change,
        format_func=format_func
    )


# Bộ lọc đa lựa chọn
def create_multiselect_filter(
    label: str,
    options: List[Any],
    default: Optional[List[Any]] = None,
    key: Optional[str] = None,
    help_text: Optional[str] = None,
    on_change: Optional[Callable] = None,
    format_func: Optional[Callable[[Any], str]] = None
) -> List[Any]:
    """
    Tạo bộ lọc đa lựa chọn.
    
    Args:
        label: Nhãn cho multiselect
        options: Danh sách các tùy chọn
        default: Các giá trị mặc định
        key: Khóa cho session state
        help_text: Văn bản trợ giúp
        on_change: Hàm callback khi giá trị thay đổi
        format_func: Hàm định dạng hiển thị tùy chọn
        
    Returns:
        Danh sách giá trị được chọn
    """
    return st.multiselect(
        label,
        options=options,
        default=default or [],
        key=key,
        help=help_text,
        on_change=on_change,
        format_func=format_func
    )


# Bộ lọc khoảng số
def create_number_range_filter(
    label: str,
    min_value: float,
    max_value: float,
    default_min: Optional[float] = None,
    default_max: Optional[float] = None,
    step: float = 1.0,
    key_prefix: str = "range",
    help_text: Optional[str] = None,
    on_change: Optional[Callable] = None,
    format: Optional[str] = None
) -> Tuple[float, float]:
    """
    Tạo bộ lọc khoảng số.
    
    Args:
        label: Nhãn cho bộ lọc
        min_value: Giá trị tối thiểu cho phép
        max_value: Giá trị tối đa cho phép
        default_min: Giá trị tối thiểu mặc định
        default_max: Giá trị tối đa mặc định
        step: Bước nhảy
        key_prefix: Tiền tố cho session state keys
        help_text: Văn bản trợ giúp
        on_change: Hàm callback khi giá trị thay đổi
        format: Định dạng hiển thị
        
    Returns:
        Tuple (min_value, max_value) được chọn
    """
    # Tạo khóa duy nhất cho session state
    min_key = f"{key_prefix}_min"
    max_key = f"{key_prefix}_max"
    
    # Khởi tạo giá trị mặc định nếu chưa có trong session state
    if min_key not in st.session_state:
        st.session_state[min_key] = default_min if default_min is not None else min_value
    
    if max_key not in st.session_state:
        st.session_state[max_key] = default_max if default_max is not None else max_value
    
    st.write(label)
    
    if help_text:
        st.caption(help_text)
    
    cols = st.columns(2)
    
    with cols[0]:
        min_val = st.number_input(
            "Tối thiểu",
            min_value=min_value,
            max_value=max_value,
            value=st.session_state[min_key],
            step=step,
            key=min_key,
            on_change=on_change,
            format=format
        )
    
    with cols[1]:
        max_val = st.number_input(
            "Tối đa",
            min_value=min_value,
            max_value=max_value,
            value=st.session_state[max_key],
            step=step,
            key=max_key,
            on_change=on_change,
            format=format
        )
    
    # Đảm bảo giá trị min <= max
    if min_val > max_val:
        st.warning(f"Giá trị tối thiểu ({min_val}) không thể lớn hơn giá trị tối đa ({max_val})")
        min_val = max_val
        st.session_state[min_key] = min_val
    
    return min_val, max_val


# Thanh trượt khoảng
def create_range_slider(
    label: str,
    min_value: float,
    max_value: float,
    default_min: Optional[float] = None,
    default_max: Optional[float] = None,
    step: float = 1.0,
    key: Optional[str] = None,
    help_text: Optional[str] = None,
    on_change: Optional[Callable] = None,
    format: Optional[str] = None
) -> Tuple[float, float]:
    """
    Tạo thanh trượt khoảng số.
    
    Args:
        label: Nhãn cho thanh trượt
        min_value: Giá trị tối thiểu cho phép
        max_value: Giá trị tối đa cho phép
        default_min: Giá trị tối thiểu mặc định
        default_max: Giá trị tối đa mặc định
        step: Bước nhảy
        key: Khóa cho session state
        help_text: Văn bản trợ giúp
        on_change: Hàm callback khi giá trị thay đổi
        format: Định dạng hiển thị
        
    Returns:
        Tuple (min_value, max_value) được chọn
    """
    if default_min is None:
        default_min = min_value
    
    if default_max is None:
        default_max = max_value
    
    return st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=(default_min, default_max),
        step=step,
        key=key,
        help=help_text,
        on_change=on_change,
        format=format
    )


# Nút công tắc (toggle)
def create_toggle(
    label: str,
    default: bool = False,
    key: Optional[str] = None,
    help_text: Optional[str] = None,
    on_change: Optional[Callable] = None
) -> bool:
    """
    Tạo nút công tắc (toggle).
    
    Args:
        label: Nhãn cho công tắc
        default: Giá trị mặc định
        key: Khóa cho session state
        help_text: Văn bản trợ giúp
        on_change: Hàm callback khi giá trị thay đổi
        
    Returns:
        Giá trị boolean của công tắc
    """
    return st.checkbox(
        label,
        value=default,
        key=key,
        help=help_text,
        on_change=on_change
    )


# Nhóm nút radio
def create_radio_group(
    label: str,
    options: List[Any],
    default_index: int = 0,
    key: Optional[str] = None,
    help_text: Optional[str] = None,
    on_change: Optional[Callable] = None,
    format_func: Optional[Callable[[Any], str]] = None,
    horizontal: bool = False
) -> Any:
    """
    Tạo nhóm nút radio.
    
    Args:
        label: Nhãn cho nhóm nút
        options: Danh sách các tùy chọn
        default_index: Chỉ số mặc định
        key: Khóa cho session state
        help_text: Văn bản trợ giúp
        on_change: Hàm callback khi giá trị thay đổi
        format_func: Hàm định dạng hiển thị tùy chọn
        horizontal: Hiển thị theo chiều ngang
        
    Returns:
        Giá trị được chọn
    """
    if horizontal:
        cols = st.columns(len(options))
        
        # Tạo khóa duy nhất cho session state
        radio_key = key or f"radio_{label}".replace(" ", "_").lower()
        
        # Khởi tạo giá trị mặc định nếu chưa có trong session state
        if radio_key not in st.session_state:
            st.session_state[radio_key] = options[default_index]
        
        selected_value = st.session_state[radio_key]
        
        for i, (col, option) in enumerate(zip(cols, options)):
            with col:
                display_value = format_func(option) if format_func else option
                
                if st.button(
                    display_value,
                    key=f"{radio_key}_{i}",
                    type="primary" if option == selected_value else "secondary"
                ):
                    st.session_state[radio_key] = option
                    selected_value = option
                    
                    # Gọi callback nếu có
                    if on_change is not None:
                        on_change()
        
        if help_text:
            st.caption(help_text)
        
        return selected_value
    else:
        return st.radio(
            label,
            options=options,
            index=default_index,
            key=key,
            help=help_text,
            on_change=on_change,
            format_func=format_func,
            horizontal=True
        )


# Bộ lọc chọn giai đoạn thời gian
def create_timeframe_filter(
    default: Optional[str] = None,
    show_label: bool = True,
    key: Optional[str] = None,
    on_change: Optional[Callable] = None
) -> str:
    """
    Tạo bộ lọc chọn giai đoạn thời gian.
    
    Args:
        default: Giá trị mặc định
        show_label: Hiển thị nhãn
        key: Khóa cho session state
        on_change: Hàm callback khi giá trị thay đổi
        
    Returns:
        Giá trị khung thời gian được chọn
    """
    timeframe_options = [tf.value for tf in Timeframe]
    
    if default is None:
        default = "1h"
    
    default_index = timeframe_options.index(default) if default in timeframe_options else 3  # 1h
    
    label = "Khung thời gian" if show_label else ""
    
    return st.selectbox(
        label,
        options=timeframe_options,
        index=default_index,
        key=key or "timeframe_filter",
        on_change=on_change
    )


# Bộ lọc chọn coin
def create_coin_filter(
    coins: Optional[List[str]] = None,
    default: Optional[str] = None,
    show_label: bool = True,
    key: Optional[str] = None,
    on_change: Optional[Callable] = None
) -> str:
    """
    Tạo bộ lọc chọn coin.
    
    Args:
        coins: Danh sách các coin
        default: Giá trị mặc định
        show_label: Hiển thị nhãn
        key: Khóa cho session state
        on_change: Hàm callback khi giá trị thay đổi
        
    Returns:
        Giá trị coin được chọn
    """
    # Lấy danh sách coin từ biến môi trường nếu không được cung cấp
    if coins is None:
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
    
    if default is None:
        default = "BTC"
    
    default_index = coins.index(default) if default in coins else 0
    
    label = "Coin" if show_label else ""
    
    return st.selectbox(
        label,
        options=coins,
        index=default_index,
        key=key or "coin_filter",
        on_change=on_change
    )


# Bộ lọc chọn sàn giao dịch
def create_exchange_filter(
    exchanges: Optional[List[str]] = None,
    default: Optional[str] = None,
    show_label: bool = True,
    key: Optional[str] = None,
    on_change: Optional[Callable] = None
) -> str:
    """
    Tạo bộ lọc chọn sàn giao dịch.
    
    Args:
        exchanges: Danh sách các sàn giao dịch
        default: Giá trị mặc định
        show_label: Hiển thị nhãn
        key: Khóa cho session state
        on_change: Hàm callback khi giá trị thay đổi
        
    Returns:
        Giá trị sàn giao dịch được chọn
    """
    # Lấy danh sách sàn giao dịch
    if exchanges is None:
        exchanges = [ex.value for ex in Exchange]
    
    if default is None:
        default = get_env("DEFAULT_EXCHANGE", "binance")
    
    default_index = exchanges.index(default) if default in exchanges else 0
    
    label = "Sàn giao dịch" if show_label else ""
    
    return st.selectbox(
        label,
        options=exchanges,
        index=default_index,
        key=key or "exchange_filter",
        on_change=on_change
    )


# Bộ lọc cho cặp giao dịch
def create_trading_pair_filter(
    exchange: Optional[str] = None,
    default: Optional[str] = None,
    show_label: bool = True,
    key: Optional[str] = None,
    on_change: Optional[Callable] = None
) -> str:
    """
    Tạo bộ lọc chọn cặp giao dịch.
    
    Args:
        exchange: Sàn giao dịch để lọc cặp giao dịch
        default: Giá trị mặc định
        show_label: Hiển thị nhãn
        key: Khóa cho session state
        on_change: Hàm callback khi giá trị thay đổi
        
    Returns:
        Giá trị cặp giao dịch được chọn
    """
    # Lấy danh sách cặp giao dịch
    # Trong thực tế, cần lấy từ database hoặc API
    # Đây là danh sách mẫu
    trading_pairs = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT", 
        "ADA/USDT", "DOT/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT"
    ]
    
    # Lọc theo sàn giao dịch nếu được chỉ định
    if exchange:
        # Trong thực tế, cần lấy từ API sàn giao dịch
        pass
    
    if default is None:
        default = get_env("DEFAULT_SYMBOL", "BTC/USDT")
    
    default_index = trading_pairs.index(default) if default in trading_pairs else 0
    
    label = "Cặp giao dịch" if show_label else ""
    
    return st.selectbox(
        label,
        options=trading_pairs,
        index=default_index,
        key=key or "trading_pair_filter",
        on_change=on_change
    )


# Bộ lọc cho chỉ báo kỹ thuật
def create_indicator_filter(
    multi_select: bool = True,
    default: Optional[Union[str, List[str]]] = None,
    show_label: bool = True,
    key: Optional[str] = None,
    on_change: Optional[Callable] = None
) -> Union[str, List[str]]:
    """
    Tạo bộ lọc chọn chỉ báo kỹ thuật.
    
    Args:
        multi_select: Cho phép chọn nhiều chỉ báo
        default: Giá trị mặc định
        show_label: Hiển thị nhãn
        key: Khóa cho session state
        on_change: Hàm callback khi giá trị thay đổi
        
    Returns:
        Giá trị hoặc danh sách chỉ báo được chọn
    """
    indicator_options = [ind.value for ind in Indicator]
    
    label = "Chỉ báo kỹ thuật" if show_label else ""
    
    if multi_select:
        if default is None or not isinstance(default, list):
            default = ["sma", "ema", "rsi"]
        
        return st.multiselect(
            label,
            options=indicator_options,
            default=default,
            key=key or "indicator_multi_filter",
            on_change=on_change
        )
    else:
        if default is None or isinstance(default, list):
            default = "sma"
        
        default_index = indicator_options.index(default) if default in indicator_options else 0
        
        return st.selectbox(
            label,
            options=indicator_options,
            index=default_index,
            key=key or "indicator_filter",
            on_change=on_change
        )


# Bộ lọc cho hồ sơ rủi ro
def create_risk_profile_filter(
    default: Optional[str] = None,
    show_label: bool = True,
    key: Optional[str] = None,
    on_change: Optional[Callable] = None
) -> str:
    """
    Tạo bộ lọc chọn hồ sơ rủi ro.
    
    Args:
        default: Giá trị mặc định
        show_label: Hiển thị nhãn
        key: Khóa cho session state
        on_change: Hàm callback khi giá trị thay đổi
        
    Returns:
        Giá trị hồ sơ rủi ro được chọn
    """
    risk_profile_options = [profile.value for profile in RiskProfile]
    
    if default is None:
        default = "moderate"
    
    default_index = risk_profile_options.index(default) if default in risk_profile_options else 1  # moderate
    
    label = "Hồ sơ rủi ro" if show_label else ""
    
    return st.selectbox(
        label,
        options=risk_profile_options,
        index=default_index,
        key=key or "risk_profile_filter",
        on_change=on_change
    )


# Bộ lọc loại agent
def create_agent_type_filter(
    default: Optional[str] = None,
    show_label: bool = True,
    key: Optional[str] = None,
    on_change: Optional[Callable] = None
) -> str:
    """
    Tạo bộ lọc chọn loại agent.
    
    Args:
        default: Giá trị mặc định
        show_label: Hiển thị nhãn
        key: Khóa cho session state
        on_change: Hàm callback khi giá trị thay đổi
        
    Returns:
        Giá trị loại agent được chọn
    """
    agent_options = [agent.value for agent in AgentType]
    
    if default is None:
        default = "dqn_agent"
    
    default_index = agent_options.index(default) if default in agent_options else 0
    
    label = "Loại agent" if show_label else ""
    
    return st.selectbox(
        label,
        options=agent_options,
        index=default_index,
        key=key or "agent_type_filter",
        on_change=on_change
    )


# Card thông tin
def create_info_card(
    title: str,
    content: str,
    icon: Optional[str] = None,
    card_type: str = "info"
) -> None:
    """
    Tạo card thông tin.
    
    Args:
        title: Tiêu đề card
        content: Nội dung card
        icon: Icon (emoji) cho card
        card_type: Loại card (info, success, warning, danger)
    """
    card_colors = {
        "info": ControlStyle.INFO_COLOR,
        "success": ControlStyle.SUCCESS_COLOR,
        "warning": ControlStyle.WARNING_COLOR,
        "danger": ControlStyle.DANGER_COLOR,
        "primary": ControlStyle.PRIMARY_COLOR,
    }
    
    color = card_colors.get(card_type, ControlStyle.INFO_COLOR)
    
    icon_text = f"{icon} " if icon else ""
    
    st.markdown(
        f"""
        <div style="border-left: 4px solid {color}; padding: 10px 15px; background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1); border-radius: 4px; margin-bottom: 10px;">
            <h4 style="color: {color}; margin: 0;">{icon_text}{title}</h4>
            <p style="margin: 5px 0 0 0;">{content}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# Hiển thị chỉ số
def display_metric(
    label: str,
    value: Any,
    delta: Optional[float] = None,
    delta_suffix: str = "%",
    formatter: Optional[Callable[[Any], str]] = None,
    prefix: str = "",
    suffix: str = "",
    cols: Optional[List[st.delta_generator.DeltaGenerator]] = None,
    col_index: int = 0
) -> None:
    """
    Hiển thị chỉ số với biến động.
    
    Args:
        label: Nhãn chỉ số
        value: Giá trị chỉ số
        delta: Giá trị biến động (tùy chọn)
        delta_suffix: Hậu tố biến động
        formatter: Hàm định dạng giá trị
        prefix: Tiền tố giá trị
        suffix: Hậu tố giá trị
        cols: Danh sách các cột
        col_index: Chỉ số cột để hiển thị
    """
    # Định dạng giá trị nếu cần
    if formatter:
        formatted_value = formatter(value)
    else:
        formatted_value = value
    
    # Thêm tiền tố và hậu tố
    display_value = f"{prefix}{formatted_value}{suffix}"
    
    # Nếu không có cột được cung cấp, hiển thị trực tiếp
    if cols is None:
        if delta is not None:
            st.metric(label, display_value, f"{delta:.2f}{delta_suffix}")
        else:
            st.metric(label, display_value, None)
    else:
        # Hiển thị trong cột chỉ định
        with cols[col_index]:
            if delta is not None:
                st.metric(label, display_value, f"{delta:.2f}{delta_suffix}")
            else:
                st.metric(label, display_value, None)


# Hiển thị biểu đồ dữ liệu
def create_timeseries_chart(
    data: pd.DataFrame,
    x_column: str,
    y_columns: List[str],
    title: Optional[str] = None,
    height: int = 400,
    use_container_width: bool = True,
    colors: Optional[List[str]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    legend_position: str = "top-right"
) -> None:
    """
    Tạo biểu đồ dữ liệu theo thời gian.
    
    Args:
        data: DataFrame chứa dữ liệu
        x_column: Tên cột cho trục x
        y_columns: Danh sách tên cột cho trục y
        title: Tiêu đề biểu đồ
        height: Chiều cao biểu đồ
        use_container_width: Sử dụng toàn bộ chiều rộng container
        colors: Danh sách màu sắc cho các đường
        x_label: Nhãn trục x
        y_label: Nhãn trục y
        legend_position: Vị trí chú thích
    """
    if title:
        st.subheader(title)
    
    # Chuẩn bị dữ liệu
    chart_data = data.copy()
    
    # Chuyển đổi dạng dữ liệu cho Altair
    chart_data_melted = pd.melt(
        chart_data,
        id_vars=[x_column],
        value_vars=y_columns,
        var_name="Series",
        value_name="Value"
    )
    
    # Tạo biểu đồ Altair
    base = alt.Chart(chart_data_melted).encode(
        x=alt.X(f"{x_column}:T", title=x_label if x_label else x_column),
        y=alt.Y("Value:Q", title=y_label if y_label else "Value"),
        color=alt.Color("Series:N", legend=alt.Legend(
            orient=legend_position.split("-")[0],
            direction=legend_position.split("-")[1],
        ))
    )
    
    # Tạo đường
    lines = base.mark_line().encode(
        tooltip=[
            alt.Tooltip(f"{x_column}:T", title="Date"),
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("Value:Q", title="Value", format=".2f")
        ]
    )
    
    # Tạo điểm
    points = base.mark_point(size=60).encode(
        opacity=alt.condition(
            alt.datum.Value == 0,
            alt.value(0),
            alt.value(0.5)
        ),
    )
    
    # Kết hợp đường và điểm
    chart = (lines + points).properties(
        height=height
    ).interactive()
    
    # Hiển thị biểu đồ
    st.altair_chart(chart, use_container_width=use_container_width)


# Tạo biểu đồ tròn
def create_pie_chart(
    data: Dict[str, float],
    title: Optional[str] = None,
    height: int = 300,
    use_container_width: bool = True,
    colors: Optional[List[str]] = None,
    donut: bool = False
) -> None:
    """
    Tạo biểu đồ tròn hoặc donut.
    
    Args:
        data: Dict chứa dữ liệu {tên: giá trị}
        title: Tiêu đề biểu đồ
        height: Chiều cao biểu đồ
        use_container_width: Sử dụng toàn bộ chiều rộng container
        colors: Danh sách màu sắc
        donut: True để tạo biểu đồ donut
    """
    if title:
        st.subheader(title)
    
    # Chuyển đổi dữ liệu thành DataFrame
    pie_data = pd.DataFrame({
        'Category': list(data.keys()),
        'Value': list(data.values())
    })
    
    # Tính tổng
    total = pie_data['Value'].sum()
    
    # Thêm phần trăm
    pie_data['Percentage'] = pie_data['Value'] / total * 100
    
    # Tạo biểu đồ Altair
    base = alt.Chart(pie_data).encode(
        theta=alt.Theta("Value:Q", stack=True),
        color=alt.Color("Category:N", legend=alt.Legend(
            orient="bottom",
            direction="horizontal",
            title=None
        )),
        tooltip=[
            alt.Tooltip("Category:N", title="Category"),
            alt.Tooltip("Value:Q", title="Value", format=".2f"),
            alt.Tooltip("Percentage:Q", title="Percentage", format=".1f%")
        ]
    )
    
    # Tạo biểu đồ tròn hoặc donut
    if donut:
        # Biểu đồ donut
        radius1 = 80
        radius2 = radius1 * 2
        
        pie = base.mark_arc(innerRadius=radius1, outerRadius=radius2)
        
        # Thêm text ở giữa
        text = alt.Chart(pd.DataFrame({'text': [f'Total: {total:.2f}']})).mark_text(
            align='center',
            baseline='middle',
            fontSize=16,
            font='Arial'
        ).encode(
            text='text:N'
        )
        
        chart = (pie + text).properties(
            height=height,
            width=height,  # Giữ tỷ lệ khung hình vuông
        )
    else:
        # Biểu đồ tròn
        pie = base.mark_arc()
        
        # Thêm text phần trăm
        text = base.mark_text(radius=100, size=16).encode(
            text=alt.Text("Percentage:Q", format=".1f%%")
        )
        
        chart = (pie + text).properties(
            height=height,
            width=height,  # Giữ tỷ lệ khung hình vuông
        )
    
    # Hiển thị biểu đồ
    st.altair_chart(chart, use_container_width=use_container_width)


# Tạo bảng dữ liệu có định dạng
def create_styled_table(
    data: pd.DataFrame,
    title: Optional[str] = None,
    formatters: Optional[Dict[str, Callable]] = None,
    highlight_columns: Optional[Dict[str, Dict[str, Any]]] = None,
    highlight_rows: Optional[Dict[int, Dict[str, Any]]] = None,
    height: Optional[int] = None,
    use_container_width: bool = True
) -> None:
    """
    Tạo bảng dữ liệu có định dạng.
    
    Args:
        data: DataFrame chứa dữ liệu
        title: Tiêu đề bảng
        formatters: Dict {tên_cột: hàm_định_dạng}
        highlight_columns: Dict {tên_cột: {thuộc_tính: giá_trị}}
        highlight_rows: Dict {chỉ_số_hàng: {thuộc_tính: giá_trị}}
        height: Chiều cao bảng
        use_container_width: Sử dụng toàn bộ chiều rộng container
    """
    if title:
        st.subheader(title)
    
    # Tạo bản sao dữ liệu
    styled_data = data.copy()
    
    # Áp dụng định dạng cho các cột
    if formatters:
        for col, formatter in formatters.items():
            if col in styled_data.columns:
                styled_data[col] = styled_data[col].apply(formatter)
    
    # Hiển thị bảng
    if height:
        st.dataframe(styled_data, height=height, use_container_width=use_container_width)
    else:
        st.dataframe(styled_data, use_container_width=use_container_width)


# Biểu đồ thanh ngang
def create_horizontal_bar_chart(
    data: Dict[str, float],
    title: Optional[str] = None,
    height: int = 300,
    use_container_width: bool = True,
    color: str = ControlStyle.PRIMARY_COLOR,
    sort_values: bool = True
) -> None:
    """
    Tạo biểu đồ thanh ngang.
    
    Args:
        data: Dict chứa dữ liệu {tên: giá trị}
        title: Tiêu đề biểu đồ
        height: Chiều cao biểu đồ
        use_container_width: Sử dụng toàn bộ chiều rộng container
        color: Màu sắc cho thanh
        sort_values: Sắp xếp theo giá trị
    """
    if title:
        st.subheader(title)
    
    # Chuyển đổi dữ liệu thành DataFrame
    bar_data = pd.DataFrame({
        'Category': list(data.keys()),
        'Value': list(data.values())
    })
    
    # Sắp xếp nếu cần
    if sort_values:
        bar_data = bar_data.sort_values('Value', ascending=False)
    
    # Tạo biểu đồ Altair
    chart = alt.Chart(bar_data).mark_bar().encode(
        y=alt.Y('Category:N', sort='-x', title=None),
        x=alt.X('Value:Q', title="Value"),
        color=alt.value(color),
        tooltip=[
            alt.Tooltip("Category:N", title="Category"),
            alt.Tooltip("Value:Q", title="Value", format=".2f")
        ]
    ).properties(
        height=height
    )
    
    # Hiển thị biểu đồ
    st.altair_chart(chart, use_container_width=use_container_width)


# Biểu đồ heat map
def create_heat_map(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str,
    title: Optional[str] = None,
    height: int = 400,
    use_container_width: bool = True,
    color_scheme: str = "viridis"
) -> None:
    """
    Tạo biểu đồ heat map.
    
    Args:
        data: DataFrame chứa dữ liệu
        x_column: Tên cột cho trục x
        y_column: Tên cột cho trục y
        color_column: Tên cột cho màu sắc
        title: Tiêu đề biểu đồ
        height: Chiều cao biểu đồ
        use_container_width: Sử dụng toàn bộ chiều rộng container
        color_scheme: Bảng màu
    """
    if title:
        st.subheader(title)
    
    # Tạo biểu đồ Altair
    chart = alt.Chart(data).mark_rect().encode(
        x=alt.X(f"{x_column}:O", title=x_column),
        y=alt.Y(f"{y_column}:O", title=y_column),
        color=alt.Color(f"{color_column}:Q", scale=alt.Scale(scheme=color_scheme)),
        tooltip=[
            alt.Tooltip(x_column, title=x_column),
            alt.Tooltip(y_column, title=y_column),
            alt.Tooltip(color_column, title=color_column, format=".2f")
        ]
    ).properties(
        height=height
    )
    
    # Hiển thị biểu đồ
    st.altair_chart(chart, use_container_width=use_container_width)


# Thêm tag cho văn bản
def add_tag(
    text: str,
    tag_type: str = "info",
    size: str = "medium"
) -> str:
    """
    Thêm tag cho văn bản.
    
    Args:
        text: Văn bản
        tag_type: Loại tag (success, warning, danger, info, primary, secondary)
        size: Kích thước (small, medium, large)
        
    Returns:
        HTML cho tag
    """
    tag_style = ControlStyle.get_tag_style(tag_type)
    
    # Điều chỉnh kích thước
    if size == "small":
        tag_style += "font-size: 0.6rem; padding: 0.15rem 0.4rem;"
    elif size == "large":
        tag_style += "font-size: 0.9rem; padding: 0.35rem 0.7rem;"
    
    return f'<span style="{tag_style}">{text}</span>'


# Tạo progress bar có màu
def create_colored_progress_bar(
    value: float,
    min_value: float = 0.0,
    max_value: float = 100.0,
    label: Optional[str] = None,
    show_percentage: bool = True,
    color_ranges: Optional[Dict[Tuple[float, float], str]] = None
) -> None:
    """
    Tạo progress bar có màu dựa trên giá trị.
    
    Args:
        value: Giá trị hiện tại
        min_value: Giá trị tối thiểu
        max_value: Giá trị tối đa
        label: Nhãn
        show_percentage: Hiển thị phần trăm
        color_ranges: Dict {(min, max): color}
    """
    # Giá trị mặc định cho color_ranges
    if color_ranges is None:
        color_ranges = {
            (0, 33): ControlStyle.DANGER_COLOR,
            (33, 66): ControlStyle.WARNING_COLOR,
            (66, 100): ControlStyle.SUCCESS_COLOR
        }
    
    # Tính phần trăm
    percentage = (value - min_value) / (max_value - min_value) * 100
    percentage = max(0, min(100, percentage))
    
    # Xác định màu sắc
    bar_color = ControlStyle.PRIMARY_COLOR
    for (range_min, range_max), color in color_ranges.items():
        if range_min <= percentage <= range_max:
            bar_color = color
            break
    
    # Tạo label
    display_label = ""
    if label:
        display_label = f"{label}: "
    
    if show_percentage:
        display_label += f"{percentage:.1f}%"
    
    # Hiển thị progress bar
    st.markdown(
        f"""
        <div style="margin-bottom: 10px;">
            <div style="margin-bottom: 5px;">{display_label}</div>
            <div style="width: 100%; background-color: #e0e0e0; border-radius: 3px; height: 10px;">
                <div style="width: {percentage}%; background-color: {bar_color}; height: 10px; border-radius: 3px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Chỉ chạy khi file này được chạy trực tiếp
if __name__ == "__main__":
    # Cài đặt các biến môi trường mô phỏng
    import sys
    from pathlib import Path
    
    # Thêm thư mục gốc vào sys.path
    project_dir = Path(__file__).parent.parent.parent
    sys.path.append(str(project_dir))
    
    # Mẫu styling
    st.set_page_config(page_title="ATS Controls Demo", layout="wide")
    ControlStyle.apply_custom_style()
    
    st.title("Demo các thành phần điều khiển")
    
    # Demo các thành phần
    st.header("Bộ lọc cơ bản")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bộ lọc thời gian")
        start_date, end_date = create_date_filter(key_prefix="demo")
        st.write(f"Khoảng thời gian đã chọn: {start_date.strftime('%d/%m/%Y')} đến {end_date.strftime('%d/%m/%Y')}")
    
    with col2:
        st.subheader("Các bộ chọn")
        exchange = create_exchange_filter()
        timeframe = create_timeframe_filter()
        trading_pair = create_trading_pair_filter()
        
        st.write(f"Đã chọn: {exchange} - {trading_pair} - {timeframe}")
    
    st.header("Biểu đồ và hiển thị dữ liệu")
    
    # Tạo dữ liệu mẫu
    date_range = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    price_data = pd.DataFrame({
        'date': date_range,
        'price': 100 + np.random.normal(0, 1, len(date_range)).cumsum(),
        'sma_20': 100 + np.random.normal(0, 0.5, len(date_range)).cumsum(),
        'volume': np.random.randint(1000, 5000, len(date_range))
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hiển thị biểu đồ thời gian
        create_timeseries_chart(
            price_data,
            x_column='date',
            y_columns=['price', 'sma_20'],
            title="Giá và SMA-20",
            height=300
        )
    
    with col2:
        # Hiển thị biểu đồ tròn
        asset_allocation = {
            "BTC": 45,
            "ETH": 30,
            "SOL": 15,
            "BNB": 10
        }
        
        create_pie_chart(
            asset_allocation,
            title="Phân bổ tài sản",
            donut=True
        )
    
    # Hiển thị bảng
    trading_history = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', end='2023-01-10'),
        'type': ['buy', 'sell', 'buy', 'sell', 'buy', 'sell', 'buy', 'buy', 'sell', 'sell'],
        'symbol': ['BTC/USDT'] * 10,
        'price': [40000, 41000, 39000, 42000, 38000, 40000, 39500, 41500, 43000, 42500],
        'amount': [0.1, 0.1, 0.2, 0.2, 0.15, 0.15, 0.25, 0.1, 0.3, 0.05],
        'profit': [0, 100, 0, 600, 0, 300, 0, 0, 450, -50]
    })
    
    # Định dạng cho các cột
    formatters = {
        'price': lambda x: f"${x:,.2f}",
        'amount': lambda x: f"{x:.4f}",
        'profit': lambda x: f"${x:+,.2f}" if x != 0 else "$0.00"
    }
    
    st.header("Bảng giao dịch")
    create_styled_table(
        trading_history,
        formatters=formatters
    )
    
    # Demo thẻ và tag
    st.header("Thông báo và trạng thái")
    
    col1, col2 = st.columns(2)
    
    with col1:
        create_info_card(
            "Thông báo hệ thống",
            "Hệ thống đã được cập nhật lên phiên bản mới nhất.",
            icon="ℹ️",
            card_type="info"
        )
        
        create_info_card(
            "Cảnh báo rủi ro",
            "Các vị thế đòn bẩy cao đang mở có thể bị thanh lý nếu thị trường biến động mạnh.",
            icon="⚠️",
            card_type="warning"
        )
    
    with col2:
        st.subheader("Trạng thái hệ thống")
        
        # Progress bars
        create_colored_progress_bar(
            75,
            label="CPU Usage",
            show_percentage=True
        )
        
        create_colored_progress_bar(
            45,
            label="Memory Usage",
            show_percentage=True
        )
        
        create_colored_progress_bar(
            92,
            label="Disk Usage",
            show_percentage=True
        )
    
    # Demo metrics
    st.header("Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    display_metric("Tổng tài sản", 25430.75, 5.2, "%", prefix="$", cols=[col1], col_index=0)
    display_metric("P&L hôm nay", 450.25, 1.8, "%", prefix="$", cols=[col2], col_index=0)
    display_metric("Số vị thế", 5, -2, "", cols=[col3], col_index=0)
    display_metric("Tỷ lệ thắng", 65.4, 3.2, "%", suffix="%", cols=[col4], col_index=0)
    
    # Demo các tag
    st.header("Tags")
    
    st.markdown(
        f"""
        Trạng thái các vị thế: 
        {add_tag("Running", "success")} 
        {add_tag("Paused", "warning")} 
        {add_tag("Stopped", "danger")} 
        {add_tag("Initializing", "info")} 
        {add_tag("Maintenace", "primary")}
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.caption("© 2025 Automated Trading System Dashboard")