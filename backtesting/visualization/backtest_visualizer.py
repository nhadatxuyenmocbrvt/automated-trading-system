"""
Module trực quan hóa kết quả backtest cho hệ thống giao dịch tự động.
File này cung cấp lớp BacktestVisualizer để tạo các biểu đồ và báo cáo trực quan
từ kết quả backtesting, giúp phân tích hiệu suất chiến lược giao dịch.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import warnings

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.constants import PositionSide, OrderType, OrderStatus, PositionStatus
from config.constants import Timeframe, ErrorCode, BacktestMetric
from config.system_config import get_system_config, BACKTEST_DIR

# Import các module backtesting
try:
    from backtesting.performance_metrics import PerformanceMetrics
    PERFORMANCE_METRICS_AVAILABLE = True
except ImportError:
    PERFORMANCE_METRICS_AVAILABLE = False


class BacktestVisualizer:
    """
    Lớp trực quan hóa kết quả backtest.
    
    Cung cấp các phương thức để tạo các biểu đồ và báo cáo trực quan từ kết quả backtest,
    bao gồm:
    1. Đường cong vốn (equity curve)
    2. Phân tích drawdown
    3. Biểu đồ phân phối lợi nhuận của các giao dịch
    4. Biểu đồ hiệu suất theo thời gian
    5. Phân tích hiệu suất theo loại tài sản, thời gian
    6. So sánh chiến lược
    7. Phân tích rủi ro
    8. Xuất báo cáo PDF hoặc HTML.
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        style: str = "default",
        interactive: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo BacktestVisualizer.
        
        Args:
            output_dir: Thư mục lưu các biểu đồ
            style: Phong cách biểu đồ ('default', 'dark', 'light', 'seaborn', 'fivethirtyeight')
            interactive: Sử dụng biểu đồ tương tác (Plotly) nếu True
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("backtest_visualizer")
        
        # Thiết lập cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = BACKTEST_DIR / f"visualization_{timestamp}"
        else:
            self.output_dir = Path(output_dir)
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thiết lập phong cách biểu đồ
        self.style = style
        self._set_style()
        
        # Thiết lập chế độ biểu đồ
        self.interactive = interactive
        
        # Khởi tạo các thuộc tính dữ liệu
        self.equity_curves = {}
        self.trade_histories = {}
        self.benchmark_data = {}
        self.backtest_results = {}
        
        self.logger.info(f"Đã khởi tạo BacktestVisualizer với output_dir={self.output_dir}")
    
    def _set_style(self) -> None:
        """
        Thiết lập phong cách biểu đồ.
        """
        # Thiết lập phong cách Matplotlib
        if self.style == "dark":
            plt.style.use('dark_background')
        elif self.style == "light":
            plt.style.use('default')
        elif self.style == "seaborn":
            plt.style.use('seaborn-v0_8-darkgrid')
        elif self.style == "fivethirtyeight":
            plt.style.use('fivethirtyeight')
        else:
            # Phong cách mặc định
            plt.style.use('default')
            
        # Thiết lập phông chữ và kích thước
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
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
    
    def load_backtest_result(
        self,
        result_path: Union[str, Path],
        strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tải kết quả backtest từ file.
        
        Args:
            result_path: Đường dẫn đến file kết quả backtest
            strategy_name: Tên chiến lược (nếu không được xác định từ file)
            
        Returns:
            Dict kết quả backtest đã tải
        """
        result_path = Path(result_path)
        if not result_path.exists():
            self.logger.error(f"Không tìm thấy file kết quả tại {result_path}")
            return {}
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            # Xác định tên chiến lược
            if strategy_name is None:
                if 'strategy_name' in result:
                    strategy_name = result['strategy_name']
                else:
                    strategy_name = result_path.stem
            
            # Gán tên chiến lược nếu không có trong result
            if 'strategy_name' not in result:
                result['strategy_name'] = strategy_name
            
            # Lưu kết quả
            self.backtest_results[strategy_name] = result
            
            # Tải equity curve nếu có
            if 'equity_curve_path' in result and result['equity_curve_path']:
                equity_path = Path(result['equity_curve_path'])
                if equity_path.exists():
                    equity_curve = pd.read_csv(equity_path, index_col=0, parse_dates=True)
                    self.equity_curves[strategy_name] = equity_curve
            
            # Tải lịch sử giao dịch nếu có
            if 'trades_path' in result and result['trades_path']:
                trades_path = Path(result['trades_path'])
                if trades_path.exists():
                    trades = pd.read_csv(trades_path, index_col=0, parse_dates=True)
                    self.trade_histories[strategy_name] = trades
            
            self.logger.info(f"Đã tải kết quả backtest cho chiến lược '{strategy_name}'")
            return result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải kết quả backtest: {e}")
            return {}
    
    def load_equity_curve(
        self,
        equity_data: Union[pd.DataFrame, pd.Series, str, Path],
        strategy_name: str
    ) -> None:
        """
        Tải dữ liệu equity curve.
        
        Args:
            equity_data: DataFrame/Series equity curve hoặc đường dẫn đến file
            strategy_name: Tên chiến lược
        """
        try:
            if isinstance(equity_data, (str, Path)):
                path = Path(equity_data)
                if not path.exists():
                    self.logger.error(f"Không tìm thấy file equity curve tại {path}")
                    return
                
                if path.suffix == '.csv':
                    equity_curve = pd.read_csv(path, index_col=0, parse_dates=True)
                elif path.suffix == '.parquet':
                    equity_curve = pd.read_parquet(path)
                elif path.suffix == '.pickle':
                    equity_curve = pd.read_pickle(path)
                else:
                    self.logger.error(f"Định dạng file không được hỗ trợ: {path.suffix}")
                    return
            else:
                # Trực tiếp sử dụng DataFrame/Series
                equity_curve = equity_data
            
            # Đảm bảo index là datetime
            if not isinstance(equity_curve.index, pd.DatetimeIndex):
                if 'timestamp' in equity_curve.columns:
                    equity_curve.set_index('timestamp', inplace=True)
                elif 'date' in equity_curve.columns:
                    equity_curve.set_index('date', inplace=True)
                elif 'time' in equity_curve.columns:
                    equity_curve.set_index('time', inplace=True)
            
            # Nếu là Series, chuyển thành DataFrame
            if isinstance(equity_curve, pd.Series):
                equity_curve = equity_curve.to_frame(name='equity')
            
            # Lưu equity curve
            self.equity_curves[strategy_name] = equity_curve
            
            self.logger.info(f"Đã tải equity curve cho chiến lược '{strategy_name}' với {len(equity_curve)} dòng")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải equity curve: {e}")
    
    def load_trade_history(
        self,
        trades_data: Union[pd.DataFrame, str, Path],
        strategy_name: str
    ) -> None:
        """
        Tải dữ liệu lịch sử giao dịch.
        
        Args:
            trades_data: DataFrame lịch sử giao dịch hoặc đường dẫn đến file
            strategy_name: Tên chiến lược
        """
        try:
            if isinstance(trades_data, (str, Path)):
                path = Path(trades_data)
                if not path.exists():
                    self.logger.error(f"Không tìm thấy file lịch sử giao dịch tại {path}")
                    return
                
                if path.suffix == '.csv':
                    trades = pd.read_csv(path, parse_dates=['entry_time', 'exit_time'])
                elif path.suffix == '.parquet':
                    trades = pd.read_parquet(path)
                elif path.suffix == '.pickle':
                    trades = pd.read_pickle(path)
                else:
                    self.logger.error(f"Định dạng file không được hỗ trợ: {path.suffix}")
                    return
            else:
                # Trực tiếp sử dụng DataFrame
                trades = trades_data
            
            # Đảm bảo có các cột cần thiết
            required_columns = ['entry_time', 'exit_time', 'profit', 'side']
            missing_columns = [col for col in required_columns if col not in trades.columns]
            
            if missing_columns:
                self.logger.warning(f"Thiếu các cột cần thiết trong lịch sử giao dịch: {missing_columns}")
            
            # Lưu lịch sử giao dịch
            self.trade_histories[strategy_name] = trades
            
            self.logger.info(f"Đã tải lịch sử giao dịch cho chiến lược '{strategy_name}' với {len(trades)} giao dịch")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải lịch sử giao dịch: {e}")
    
    def load_benchmark_data(
        self,
        benchmark_data: Union[pd.DataFrame, pd.Series, str, Path],
        benchmark_name: str = "benchmark"
    ) -> None:
        """
        Tải dữ liệu benchmark.
        
        Args:
            benchmark_data: DataFrame/Series benchmark hoặc đường dẫn đến file
            benchmark_name: Tên benchmark
        """
        try:
            if isinstance(benchmark_data, (str, Path)):
                path = Path(benchmark_data)
                if not path.exists():
                    self.logger.error(f"Không tìm thấy file benchmark tại {path}")
                    return
                
                if path.suffix == '.csv':
                    benchmark = pd.read_csv(path, index_col=0, parse_dates=True)
                elif path.suffix == '.parquet':
                    benchmark = pd.read_parquet(path)
                elif path.suffix == '.pickle':
                    benchmark = pd.read_pickle(path)
                else:
                    self.logger.error(f"Định dạng file không được hỗ trợ: {path.suffix}")
                    return
            else:
                # Trực tiếp sử dụng DataFrame/Series
                benchmark = benchmark_data
            
            # Đảm bảo index là datetime
            if not isinstance(benchmark.index, pd.DatetimeIndex):
                if 'timestamp' in benchmark.columns:
                    benchmark.set_index('timestamp', inplace=True)
                elif 'date' in benchmark.columns:
                    benchmark.set_index('date', inplace=True)
                elif 'time' in benchmark.columns:
                    benchmark.set_index('time', inplace=True)
            
            # Nếu là Series, chuyển thành DataFrame
            if isinstance(benchmark, pd.Series):
                benchmark = benchmark.to_frame(name='value')
            
            # Lưu benchmark
            self.benchmark_data[benchmark_name] = benchmark
            
            self.logger.info(f"Đã tải dữ liệu benchmark '{benchmark_name}' với {len(benchmark)} dòng")
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tải dữ liệu benchmark: {e}")
    
    def plot_equity_curve(
        self,
        strategy_names: Union[str, List[str]] = None,
        benchmark_names: Union[str, List[str]] = None,
        include_drawdown: bool = True,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        figsize: Tuple[int, int] = (12, 8),
        normalize: bool = True,
        fill_between: bool = True,
        show_legend: bool = True,
        grid: bool = True,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> Optional[Union[plt.Figure, go.Figure]]:
        """
        Vẽ đường cong vốn (equity curve).
        
        Args:
            strategy_names: Tên chiến lược hoặc danh sách các chiến lược
            benchmark_names: Tên benchmark hoặc danh sách các benchmark
            include_drawdown: Hiển thị drawdown
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc
            figsize: Kích thước biểu đồ
            normalize: Chuẩn hóa về 1 tại thời điểm bắt đầu
            fill_between: Tô màu dưới đường
            show_legend: Hiển thị chú thích
            grid: Hiển thị lưới
            title: Tiêu đề biểu đồ
            save_path: Đường dẫn lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Đối tượng Figure nếu thành công, None nếu thất bại
        """
        if not self.equity_curves:
            self.logger.error("Không có dữ liệu equity curve để vẽ biểu đồ")
            return None
        
        # Chuyển đổi strategy_names thành danh sách
        if strategy_names is None:
            strategy_names = list(self.equity_curves.keys())
        elif isinstance(strategy_names, str):
            strategy_names = [strategy_names]
        
        # Chuyển đổi benchmark_names thành danh sách
        if benchmark_names is None:
            benchmark_names = []
        elif isinstance(benchmark_names, str):
            benchmark_names = [benchmark_names]
        
        # Kiểm tra các chiến lược tồn tại
        valid_strategies = [s for s in strategy_names if s in self.equity_curves]
        if not valid_strategies:
            self.logger.error("Không tìm thấy chiến lược nào được chỉ định")
            return None
        
        # Kiểm tra các benchmark tồn tại
        valid_benchmarks = [b for b in benchmark_names if b in self.benchmark_data]
        
        # Chuyển đổi start_date và end_date thành datetime
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Tiêu đề mặc định
        if title is None:
            if len(valid_strategies) == 1:
                title = f"Đường cong vốn cho chiến lược {valid_strategies[0]}"
            else:
                title = f"So sánh đường cong vốn ({len(valid_strategies)} chiến lược)"
        
        try:
            if self.interactive:
                return self._plot_equity_curve_plotly(
                    strategy_names=valid_strategies,
                    benchmark_names=valid_benchmarks,
                    include_drawdown=include_drawdown,
                    start_date=start_date,
                    end_date=end_date,
                    normalize=normalize,
                    title=title,
                    save_path=save_path,
                    show_plot=show_plot
                )
            else:
                return self._plot_equity_curve_matplotlib(
                    strategy_names=valid_strategies,
                    benchmark_names=valid_benchmarks,
                    include_drawdown=include_drawdown,
                    start_date=start_date,
                    end_date=end_date,
                    figsize=figsize,
                    normalize=normalize,
                    fill_between=fill_between,
                    show_legend=show_legend,
                    grid=grid,
                    title=title,
                    save_path=save_path,
                    show_plot=show_plot
                )
                
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ đường cong vốn: {e}")
            return None
    
    def _plot_equity_curve_matplotlib(
        self,
        strategy_names: List[str],
        benchmark_names: List[str],
        include_drawdown: bool,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        figsize: Tuple[int, int],
        normalize: bool,
        fill_between: bool,
        show_legend: bool,
        grid: bool,
        title: str,
        save_path: Optional[Union[str, Path]],
        show_plot: bool
    ) -> plt.Figure:
        """
        Vẽ đường cong vốn sử dụng Matplotlib.
        
        Args: (xem phương thức plot_equity_curve)
            
        Returns:
            Đối tượng Figure của matplotlib
        """
        # Tạo biểu đồ
        if include_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
        
        # Các màu và kiểu đường cho chiến lược
        strategy_colors = plt.cm.tab10.colors
        
        # Vẽ đường cong vốn cho mỗi chiến lược
        for i, strategy_name in enumerate(strategy_names):
            equity_curve = self.equity_curves[strategy_name]
            
            # Lọc theo thời gian
            if start_date:
                equity_curve = equity_curve[equity_curve.index >= start_date]
            if end_date:
                equity_curve = equity_curve[equity_curve.index <= end_date]
            
            # Chọn cột equity
            if 'equity' in equity_curve.columns:
                equity_series = equity_curve['equity']
            else:
                # Sử dụng cột đầu tiên nếu không có cột 'equity'
                equity_series = equity_curve.iloc[:, 0]
            
            # Chuẩn hóa
            if normalize:
                equity_series = equity_series / equity_series.iloc[0]
            
            # Vẽ đường
            color = strategy_colors[i % len(strategy_colors)]
            ax1.plot(equity_series.index, equity_series, label=strategy_name, color=color, linewidth=2)
            
            # Tô màu dưới đường
            if fill_between:
                ax1.fill_between(equity_series.index, 1 if normalize else 0, equity_series, 
                               alpha=0.1, color=color)
            
            # Vẽ drawdown nếu cần
            if include_drawdown:
                # Tính drawdown
                rolling_max = equity_series.cummax()
                drawdown = (equity_series - rolling_max) / rolling_max
                
                # Vẽ drawdown
                ax2.fill_between(drawdown.index, 0, drawdown * 100, color=color, alpha=0.3)
                ax2.plot(drawdown.index, drawdown * 100, color=color, linewidth=1, alpha=0.5)
        
        # Vẽ benchmark nếu có
        for i, benchmark_name in enumerate(benchmark_names):
            benchmark = self.benchmark_data[benchmark_name]
            
            # Lọc theo thời gian
            if start_date:
                benchmark = benchmark[benchmark.index >= start_date]
            if end_date:
                benchmark = benchmark[benchmark.index <= end_date]
            
            # Chọn cột giá trị
            if 'value' in benchmark.columns:
                benchmark_series = benchmark['value']
            else:
                # Sử dụng cột đầu tiên nếu không có cột 'value'
                benchmark_series = benchmark.iloc[:, 0]
            
            # Chuẩn hóa
            if normalize:
                benchmark_series = benchmark_series / benchmark_series.iloc[0]
            
            # Vẽ đường
            ax1.plot(benchmark_series.index, benchmark_series, label=benchmark_name, 
                   color='grey', linestyle='--', linewidth=1.5)
        
        # Thiết lập ax1 (equity curve)
        ax1.set_title(title, fontsize=14)
        ax1.set_ylabel('Giá trị vốn' + (' (chuẩn hóa)' if normalize else ''))
        
        # Định dạng trục x (chỉ hiển thị trên biểu đồ dưới cùng)
        if include_drawdown:
            ax1.set_xticklabels([])
            ax1.set_xlabel('')
        else:
            ax1.set_xlabel('Thời gian')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        # Thiết lập lưới
        if grid:
            ax1.grid(True, alpha=0.3)
        
        # Thiết lập chú thích
        if show_legend:
            ax1.legend(loc='best')
        
        # Thiết lập ax2 (drawdown) nếu có
        if include_drawdown:
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Thời gian')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Thêm lưới cho drawdown
            if grid:
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            # Đảm bảo thư mục tồn tại
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Đã lưu biểu đồ đường cong vốn tại {save_path}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_equity_curve_plotly(
        self,
        strategy_names: List[str],
        benchmark_names: List[str],
        include_drawdown: bool,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        normalize: bool,
        title: str,
        save_path: Optional[Union[str, Path]],
        show_plot: bool
    ) -> go.Figure:
        """
        Vẽ đường cong vốn sử dụng Plotly.
        
        Args: (xem phương thức plot_equity_curve)
            
        Returns:
            Đối tượng Figure của plotly
        """
        # Tạo biểu đồ
        if include_drawdown:
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.7, 0.3],
                subplot_titles=("Đường cong vốn", "Drawdown (%)")
            )
        else:
            fig = go.Figure()
        
        # Vẽ đường cong vốn cho mỗi chiến lược
        for i, strategy_name in enumerate(strategy_names):
            equity_curve = self.equity_curves[strategy_name]
            
            # Lọc theo thời gian
            if start_date:
                equity_curve = equity_curve[equity_curve.index >= start_date]
            if end_date:
                equity_curve = equity_curve[equity_curve.index <= end_date]
            
            # Chọn cột equity
            if 'equity' in equity_curve.columns:
                equity_series = equity_curve['equity']
            else:
                # Sử dụng cột đầu tiên nếu không có cột 'equity'
                equity_series = equity_curve.iloc[:, 0]
            
            # Chuẩn hóa
            if normalize:
                equity_series = equity_series / equity_series.iloc[0]
            
            # Vẽ đường cong vốn
            if include_drawdown:
                fig.add_trace(
                    go.Scatter(
                        x=equity_series.index,
                        y=equity_series,
                        mode='lines',
                        name=strategy_name,
                        fill='tozeroy',
                        fillcolor=f'rgba({i*50}, {255-i*30}, {150}, 0.1)'
                    ),
                    row=1, col=1
                )
                
                # Tính và vẽ drawdown
                rolling_max = equity_series.cummax()
                drawdown = (equity_series - rolling_max) / rolling_max * 100
                
                fig.add_trace(
                    go.Scatter(
                        x=drawdown.index,
                        y=drawdown,
                        mode='lines',
                        name=f'{strategy_name} DD',
                        line=dict(color=f'rgb({i*50}, {255-i*30}, {150})'),
                        showlegend=False
                    ),
                    row=2, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=equity_series.index,
                        y=equity_series,
                        mode='lines',
                        name=strategy_name,
                        fill='tozeroy',
                        fillcolor=f'rgba({i*50}, {255-i*30}, {150}, 0.1)'
                    )
                )
        
        # Vẽ benchmark nếu có
        for i, benchmark_name in enumerate(benchmark_names):
            benchmark = self.benchmark_data[benchmark_name]
            
            # Lọc theo thời gian
            if start_date:
                benchmark = benchmark[benchmark.index >= start_date]
            if end_date:
                benchmark = benchmark[benchmark.index <= end_date]
            
            # Chọn cột giá trị
            if 'value' in benchmark.columns:
                benchmark_series = benchmark['value']
            else:
                # Sử dụng cột đầu tiên nếu không có cột 'value'
                benchmark_series = benchmark.iloc[:, 0]
            
            # Chuẩn hóa
            if normalize:
                benchmark_series = benchmark_series / benchmark_series.iloc[0]
            
            # Vẽ benchmark
            if include_drawdown:
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_series.index,
                        y=benchmark_series,
                        mode='lines',
                        name=benchmark_name,
                        line=dict(color='grey', dash='dash')
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_series.index,
                        y=benchmark_series,
                        mode='lines',
                        name=benchmark_name,
                        line=dict(color='grey', dash='dash')
                    )
                )
        
        # Cập nhật layout
        fig.update_layout(
            title=title,
            xaxis_title="Thời gian",
            yaxis_title="Giá trị vốn" + (" (chuẩn hóa)" if normalize else ""),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white" if self.style != "dark" else "plotly_dark"
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
        
        # Thiết lập drawdown y-axis
        if include_drawdown:
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        # Lưu biểu đồ nếu cần
        if save_path:
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            # Đảm bảo thư mục tồn tại
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            extension = save_path.suffix.lower()
            if extension == '.html':
                fig.write_html(save_path)
            elif extension == '.json':
                fig.write_json(save_path)
            else:
                fig.write_image(save_path)
            
            self.logger.info(f"Đã lưu biểu đồ đường cong vốn tại {save_path}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_drawdown_analysis(
        self,
        strategy_name: str,
        top_n: int = 5,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> Optional[Union[plt.Figure, go.Figure]]:
        """
        Vẽ phân tích drawdown chi tiết.
        
        Args:
            strategy_name: Tên chiến lược
            top_n: Số lượng drawdown lớn nhất cần hiển thị
            figsize: Kích thước biểu đồ
            save_path: Đường dẫn lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Đối tượng Figure nếu thành công, None nếu thất bại
        """
        if strategy_name not in self.equity_curves:
            self.logger.error(f"Không tìm thấy equity curve cho chiến lược '{strategy_name}'")
            return None
        
        equity_curve = self.equity_curves[strategy_name]
        
        # Chọn cột equity
        if 'equity' in equity_curve.columns:
            equity_series = equity_curve['equity']
        else:
            # Sử dụng cột đầu tiên nếu không có cột 'equity'
            equity_series = equity_curve.iloc[:, 0]
        
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
                    'duration': (drawdown.index[end_idx] - drawdown.index[start_idx]).days
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
            if self.interactive:
                return self._plot_drawdown_analysis_plotly(
                    strategy_name=strategy_name,
                    equity_series=equity_series,
                    drawdown=drawdown,
                    top_periods=top_periods,
                    save_path=save_path,
                    show_plot=show_plot
                )
            else:
                return self._plot_drawdown_analysis_matplotlib(
                    strategy_name=strategy_name,
                    equity_series=equity_series,
                    drawdown=drawdown,
                    top_periods=top_periods,
                    figsize=figsize,
                    save_path=save_path,
                    show_plot=show_plot
                )
                
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ phân tích drawdown: {e}")
            return None
    
    def _plot_drawdown_analysis_matplotlib(
        self,
        strategy_name: str,
        equity_series: pd.Series,
        drawdown: pd.Series,
        top_periods: List[Dict[str, Any]],
        figsize: Tuple[int, int],
        save_path: Optional[Union[str, Path]],
        show_plot: bool
    ) -> plt.Figure:
        """
        Vẽ phân tích drawdown chi tiết sử dụng Matplotlib.
        
        Args: (xem phương thức plot_drawdown_analysis)
            
        Returns:
            Đối tượng Figure của matplotlib
        """
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
        ax1.set_title(f"Phân tích Drawdown cho chiến lược {strategy_name}", fontsize=14)
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
        ax1.set_xticklabels([])
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            # Đảm bảo thư mục tồn tại
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Đã lưu biểu đồ phân tích drawdown tại {save_path}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_drawdown_analysis_plotly(
        self,
        strategy_name: str,
        equity_series: pd.Series,
        drawdown: pd.Series,
        top_periods: List[Dict[str, Any]],
        save_path: Optional[Union[str, Path]],
        show_plot: bool
    ) -> go.Figure:
        """
        Vẽ phân tích drawdown chi tiết sử dụng Plotly.
        
        Args: (xem phương thức plot_drawdown_analysis)
            
        Returns:
            Đối tượng Figure của plotly
        """
        # Tạo biểu đồ với 2 phần
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=(
                f"Phân tích Drawdown cho chiến lược {strategy_name}",
                "Chi tiết Drawdown"
            )
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
        
        # Cập nhật layout
        fig.update_layout(
            xaxis_title="Thời gian",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white" if self.style != "dark" else "plotly_dark"
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
        
        # Lưu biểu đồ nếu cần
        if save_path:
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            # Đảm bảo thư mục tồn tại
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            extension = save_path.suffix.lower()
            if extension == '.html':
                fig.write_html(save_path)
            elif extension == '.json':
                fig.write_json(save_path)
            else:
                fig.write_image(save_path)
            
            self.logger.info(f"Đã lưu biểu đồ phân tích drawdown tại {save_path}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_trade_distribution(
        self,
        strategy_name: str,
        figsize: Tuple[int, int] = (12, 10),
        bins: int = 20,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> Optional[Union[plt.Figure, List[go.Figure]]]:
        """
        Vẽ phân phối giao dịch.
        
        Args:
            strategy_name: Tên chiến lược
            figsize: Kích thước biểu đồ
            bins: Số lượng bins cho histogram
            save_path: Đường dẫn lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Đối tượng Figure nếu thành công, None nếu thất bại
        """
        if strategy_name not in self.trade_histories:
            self.logger.error(f"Không tìm thấy lịch sử giao dịch cho chiến lược '{strategy_name}'")
            return None
        
        trades = self.trade_histories[strategy_name]
        
        # Kiểm tra cột cần thiết
        required_columns = ['profit', 'entry_time', 'exit_time']
        missing_columns = [col for col in required_columns if col not in trades.columns]
        
        if missing_columns:
            self.logger.error(f"Thiếu các cột cần thiết trong lịch sử giao dịch: {missing_columns}")
            return None
        
        try:
            if self.interactive:
                return self._plot_trade_distribution_plotly(
                    strategy_name=strategy_name,
                    trades=trades,
                    bins=bins,
                    save_path=save_path,
                    show_plot=show_plot
                )
            else:
                return self._plot_trade_distribution_matplotlib(
                    strategy_name=strategy_name,
                    trades=trades,
                    figsize=figsize,
                    bins=bins,
                    save_path=save_path,
                    show_plot=show_plot
                )
                
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ phân phối giao dịch: {e}")
            return None
    
    def _plot_trade_distribution_matplotlib(
        self,
        strategy_name: str,
        trades: pd.DataFrame,
        figsize: Tuple[int, int],
        bins: int,
        save_path: Optional[Union[str, Path]],
        show_plot: bool
    ) -> plt.Figure:
        """
        Vẽ phân phối giao dịch sử dụng Matplotlib.
        
        Args: (xem phương thức plot_trade_distribution)
            
        Returns:
            Đối tượng Figure của matplotlib
        """
        # Tạo biểu đồ với 4 phần
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])  # P/L Distribution
        ax2 = fig.add_subplot(gs[0, 1])  # Win/Loss Ratio
        ax3 = fig.add_subplot(gs[1, :])  # P/L vs Duration
        ax4 = fig.add_subplot(gs[2, 0])  # Monthly P/L
        ax5 = fig.add_subplot(gs[2, 1])  # Trade Count by Day of Week
        
        # 1. P/L Distribution
        wins = trades[trades['profit'] > 0]
        losses = trades[trades['profit'] < 0]
        
        # Histogram cho toàn bộ trades
        ax1.hist(
            trades['profit'], 
            bins=bins, 
            alpha=0.7, 
            color='blue', 
            label=f'Tất cả ({len(trades)})'
        )
        
        # Histogram cho các giao dịch thắng
        ax1.hist(
            wins['profit'], 
            bins=bins, 
            alpha=0.7, 
            color='green', 
            label=f'Thắng ({len(wins)})'
        )
        
        # Histogram cho các giao dịch thua
        ax1.hist(
            losses['profit'], 
            bins=bins, 
            alpha=0.7, 
            color='red', 
            label=f'Thua ({len(losses)})'
        )
        
        ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('Phân phối lợi nhuận')
        ax1.set_xlabel('Lợi nhuận')
        ax1.set_ylabel('Số lượng giao dịch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Win/Loss Ratio Pie Chart
        win_count = len(wins)
        loss_count = len(losses)
        
        # Tính tỷ lệ thắng/thua
        win_rate = win_count / len(trades) if len(trades) > 0 else 0
        
        labels = [f'Thắng ({win_count})', f'Thua ({loss_count})']
        sizes = [win_count, loss_count]
        colors = ['green', 'red']
        explode = (0.1, 0)
        
        ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
        ax2.set_title(f'Tỷ lệ thắng/thua: {win_rate:.2%}')
        
        # 3. P/L vs Duration Scatter Plot
        # Tính duration (số ngày) cho mỗi giao dịch
        if 'duration' not in trades.columns:
            trades['duration'] = (pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])).dt.total_seconds() / (60 * 60 * 24)  # convert to days
        
        # Phân loại màu theo P/L
        colors = ['green' if p > 0 else 'red' for p in trades['profit']]
        
        ax3.scatter(trades['duration'], trades['profit'], c=colors, alpha=0.7)
        
        # Đường hồi quy
        if len(trades) > 1:
            z = np.polyfit(trades['duration'], trades['profit'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(trades['duration'].min(), trades['duration'].max(), 100)
            ax3.plot(x_line, p(x_line), 'b--', alpha=0.7)
        
        ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Lợi nhuận theo thời gian giao dịch')
        ax3.set_xlabel('Thời gian giao dịch (ngày)')
        ax3.set_ylabel('Lợi nhuận')
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly P/L
        # Chuyển đổi entry_time thành datetime nếu chưa phải
        if not pd.api.types.is_datetime64_any_dtype(trades['entry_time']):
            trades['entry_time'] = pd.to_datetime(trades['entry_time'])
        
        # Tạo cột year-month
        trades['year_month'] = trades['entry_time'].dt.strftime('%Y-%m')
        
        # Nhóm theo tháng và tính tổng lợi nhuận
        monthly_profit = trades.groupby('year_month')['profit'].sum()
        
        # Vẽ biểu đồ cột
        bars = ax4.bar(
            monthly_profit.index, 
            monthly_profit.values, 
            color=['green' if p > 0 else 'red' for p in monthly_profit.values]
        )
        
        # Thêm giá trị trên mỗi cột
        for bar in bars:
            height = bar.get_height()
            ypos = height + 0.01 if height > 0 else height - 0.03
            ax4.text(
                bar.get_x() + bar.get_width()/2, 
                ypos,
                f'{height:.2f}',
                ha='center', 
                va='bottom' if height > 0 else 'top',
                fontsize=8,
                rotation=90
            )
        
        ax4.set_title('Lợi nhuận theo tháng')
        ax4.set_xlabel('Tháng')
        ax4.set_ylabel('Lợi nhuận')
        ax4.tick_params(axis='x', labelrotation=90)
        ax4.grid(True, alpha=0.3)
        
        # 5. Trade Count by Day of Week
        trades['day_of_week'] = trades['entry_time'].dt.day_name()
        
        # Tính số lượng giao dịch theo ngày trong tuần
        day_counts = trades['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        # Tính lợi nhuận trung bình theo ngày trong tuần
        day_profit = trades.groupby('day_of_week')['profit'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        # Vẽ biểu đồ cột
        bars = ax5.bar(
            day_counts.index, 
            day_counts.values, 
            color=['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        )
        
        # Thêm giá trị trên mỗi cột
        for bar in bars:
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width()/2, 
                height + 0.1,
                str(int(height)),
                ha='center', 
                va='bottom',
                fontsize=9
            )
        
        ax5.set_title('Số lượng giao dịch theo ngày trong tuần')
        ax5.set_xlabel('Ngày trong tuần')
        ax5.set_ylabel('Số lượng giao dịch')
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle(f'Phân tích giao dịch cho chiến lược {strategy_name}', fontsize=16)
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            # Đảm bảo thư mục tồn tại
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Đã lưu biểu đồ phân phối giao dịch tại {save_path}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_trade_distribution_plotly(
        self,
        strategy_name: str,
        trades: pd.DataFrame,
        bins: int,
        save_path: Optional[Union[str, Path]],
        show_plot: bool
    ) -> List[go.Figure]:
        """
        Vẽ phân phối giao dịch sử dụng Plotly.
        
        Args: (xem phương thức plot_trade_distribution)
            
        Returns:
            Danh sách các đối tượng Figure của plotly
        """
        # Tạo các biểu đồ riêng biệt
        figures = []
        
        # Dữ liệu cơ bản
        wins = trades[trades['profit'] > 0]
        losses = trades[trades['profit'] < 0]
        
        # Tính duration (số ngày) cho mỗi giao dịch nếu chưa có
        if 'duration' not in trades.columns:
            trades['duration'] = (pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])).dt.total_seconds() / (60 * 60 * 24)  # convert to days
        
        # Chuyển đổi entry_time thành datetime nếu chưa phải
        if not pd.api.types.is_datetime64_any_dtype(trades['entry_time']):
            trades['entry_time'] = pd.to_datetime(trades['entry_time'])
        
        # 1. P/L Distribution
        fig1 = make_subplots(rows=1, cols=2, subplot_titles=('Phân phối lợi nhuận', 'Tỷ lệ thắng/thua'))
        
        # Histogram cho lợi nhuận
        fig1.add_trace(
            go.Histogram(
                x=trades['profit'],
                nbinsx=bins,
                opacity=0.7,
                name='Tất cả',
                marker_color='blue'
            ),
            row=1, col=1
        )
        
        fig1.add_trace(
            go.Histogram(
                x=wins['profit'],
                nbinsx=bins,
                opacity=0.7,
                name='Thắng',
                marker_color='green'
            ),
            row=1, col=1
        )
        
        fig1.add_trace(
            go.Histogram(
                x=losses['profit'],
                nbinsx=bins,
                opacity=0.7,
                name='Thua',
                marker_color='red'
            ),
            row=1, col=1
        )
        
        # Thêm đường vertical
        fig1.add_shape(
            type="line",
            x0=0, y0=0, x1=0, y1=1,
            yref="paper",
            line=dict(color="black", width=2, dash="dash"),
            row=1, col=1
        )
        
        # Pie Chart cho tỷ lệ thắng/thua
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / len(trades) if len(trades) > 0 else 0
        
        fig1.add_trace(
            go.Pie(
                labels=[f'Thắng ({win_count})', f'Thua ({loss_count})'],
                values=[win_count, loss_count],
                marker=dict(colors=['green', 'red']),
                textinfo='percent+label',
                hole=0.3,
                pull=[0.1, 0],
                title=dict(text=f"Tỷ lệ thắng: {win_rate:.2%}")
            ),
            row=1, col=2
        )
        
        fig1.update_layout(
            title=f'Phân tích lợi nhuận cho chiến lược {strategy_name}',
            template="plotly_white" if self.style != "dark" else "plotly_dark",
            showlegend=True
        )
        
        figures.append(fig1)
        
        # 2. P/L vs Duration Scatter Plot
        fig2 = go.Figure()
        
        # Scatter plot
        fig2.add_trace(
            go.Scatter(
                x=trades['duration'],
                y=trades['profit'],
                mode='markers',
                marker=dict(
                    color=trades['profit'],
                    colorscale='RdYlGn',
                    size=8,
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(title='Lợi nhuận')
                ),
                text=trades.apply(
                    lambda row: f"Lợi nhuận: {row['profit']:.2f}<br>Thời gian: {row['duration']:.2f} ngày<br>Vào: {row['entry_time']}<br>Ra: {row['exit_time']}",
                    axis=1
                ),
                hoverinfo='text'
            )
        )
        
        # Đường hồi quy
        if len(trades) > 1:
            z = np.polyfit(trades['duration'], trades['profit'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(trades['duration'].min(), trades['duration'].max(), 100)
            
            fig2.add_trace(
                go.Scatter(
                    x=x_line,
                    y=p(x_line),
                    mode='lines',
                    line=dict(color='blue', dash='dash'),
                    name=f'Trend: y = {z[0]:.4f}x + {z[1]:.4f}'
                )
            )
        
        # Đường ngang tại 0
        fig2.add_shape(
            type="line",
            x0=trades['duration'].min(), x1=trades['duration'].max(),
            y0=0, y1=0,
            line=dict(color="black", width=2, dash="dash")
        )
        
        fig2.update_layout(
            title=f'Lợi nhuận theo thời gian giao dịch',
            xaxis_title='Thời gian giao dịch (ngày)',
            yaxis_title='Lợi nhuận',
            template="plotly_white" if self.style != "dark" else "plotly_dark"
        )
        
        figures.append(fig2)
        
        # 3. Monthly P/L và Day of Week
        fig3 = make_subplots(rows=1, cols=2, subplot_titles=('Lợi nhuận theo tháng', 'Số lượng giao dịch theo ngày trong tuần'))
        
        # Tạo cột year-month
        trades['year_month'] = trades['entry_time'].dt.strftime('%Y-%m')
        
        # Nhóm theo tháng và tính tổng lợi nhuận
        monthly_profit = trades.groupby('year_month')['profit'].sum().reset_index()
        monthly_profit['color'] = monthly_profit['profit'].apply(lambda x: 'green' if x > 0 else 'red')
        
        # Vẽ biểu đồ cột cho lợi nhuận theo tháng
        fig3.add_trace(
            go.Bar(
                x=monthly_profit['year_month'],
                y=monthly_profit['profit'],
                marker_color=monthly_profit['color'],
                text=monthly_profit['profit'].apply(lambda x: f'{x:.2f}'),
                textposition='auto',
                name='Lợi nhuận'
            ),
            row=1, col=1
        )
        
        # Day of Week Analysis
        trades['day_of_week'] = trades['entry_time'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Tính số lượng giao dịch theo ngày trong tuần
        day_counts = trades['day_of_week'].value_counts().reindex(day_order).fillna(0)
        
        # Tính lợi nhuận trung bình theo ngày trong tuần
        day_profit = trades.groupby('day_of_week')['profit'].mean().reindex(day_order).fillna(0)
        
        # Vẽ biểu đồ cột cho số lượng giao dịch theo ngày
        fig3.add_trace(
            go.Bar(
                x=day_counts.index,
                y=day_counts.values,
                text=day_counts.values.astype(int),
                textposition='auto',
                marker_color='blue',
                name='Số lượng giao dịch'
            ),
            row=1, col=2
        )
        
        # Vẽ đường cho lợi nhuận trung bình theo ngày
        fig3.add_trace(
            go.Scatter(
                x=day_profit.index,
                y=day_profit.values,
                mode='lines+markers',
                line=dict(color='orange', width=2),
                marker=dict(size=8, symbol='circle'),
                name='Lợi nhuận TB',
                yaxis='y2'
            ),
            row=1, col=2
        )
        
        # Cập nhật layout
        fig3.update_layout(
            title=f'Phân tích thời gian cho chiến lược {strategy_name}',
            xaxis_title='Tháng',
            xaxis2_title='Ngày trong tuần',
            yaxis_title='Lợi nhuận',
            yaxis2_title='Số lượng giao dịch',
            template="plotly_white" if self.style != "dark" else "plotly_dark",
            xaxis={'tickangle': 45},
            yaxis2=dict(
                title='Lợi nhuận TB',
                overlaying='y',
                side='right'
            )
        )
        
        figures.append(fig3)
        
        # Lưu biểu đồ nếu cần
        if save_path:
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            # Đảm bảo thư mục tồn tại
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Tạo tên file với chỉ số
            for i, fig in enumerate(figures):
                file_path = save_path.parent / f"{save_path.stem}_{i+1}{save_path.suffix}"
                
                extension = file_path.suffix.lower()
                if extension == '.html':
                    fig.write_html(file_path)
                elif extension == '.json':
                    fig.write_json(file_path)
                else:
                    fig.write_image(file_path)
            
            self.logger.info(f"Đã lưu {len(figures)} biểu đồ phân phối giao dịch tại {save_path.parent}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            for fig in figures:
                fig.show()
        
        return figures
    
    def plot_monthly_returns_heatmap(
        self,
        strategy_name: str,
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = 'RdYlGn',
        annot: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> Optional[Union[plt.Figure, go.Figure]]:
        """
        Vẽ bản đồ nhiệt (heatmap) lợi nhuận theo tháng.
        
        Args:
            strategy_name: Tên chiến lược
            figsize: Kích thước biểu đồ
            cmap: Bảng màu
            annot: Hiển thị giá trị trên heatmap
            save_path: Đường dẫn lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Đối tượng Figure nếu thành công, None nếu thất bại
        """
        if strategy_name not in self.equity_curves:
            if strategy_name in self.trade_histories:
                self.logger.warning(f"Không tìm thấy equity curve, sẽ tính từ lịch sử giao dịch")
                trades = self.trade_histories[strategy_name]
                
                # Tính lợi nhuận theo tháng từ lịch sử giao dịch
                if 'entry_time' not in trades.columns or 'profit' not in trades.columns:
                    self.logger.error("Lịch sử giao dịch không có cột entry_time hoặc profit")
                    return None
                
                # Chuyển đổi entry_time thành datetime nếu chưa phải
                if not pd.api.types.is_datetime64_any_dtype(trades['entry_time']):
                    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
                
                # Tạo cột year-month
                trades['year_month'] = trades['entry_time'].dt.strftime('%Y-%m')
                trades['year'] = trades['entry_time'].dt.year
                trades['month'] = trades['entry_time'].dt.month
                
                # Nhóm theo tháng và tính tổng lợi nhuận
                monthly_returns = trades.groupby(['year', 'month'])['profit'].sum()
                monthly_returns = monthly_returns.unstack(level='month')
            else:
                self.logger.error(f"Không tìm thấy dữ liệu cho chiến lược '{strategy_name}'")
                return None
        else:
            # Tính lợi nhuận hàng tháng từ equity curve
            equity_curve = self.equity_curves[strategy_name]
            
            # Chọn cột equity
            if 'equity' in equity_curve.columns:
                equity_series = equity_curve['equity']
            else:
                # Sử dụng cột đầu tiên nếu không có cột 'equity'
                equity_series = equity_curve.iloc[:, 0]
            
            # Tính lợi nhuận cuối mỗi tháng
            monthly_equity = equity_series.resample('M').last()
            
            # Tính lợi nhuận hàng tháng
            monthly_returns = monthly_equity.pct_change().dropna()
            
            # Chuyển thành DataFrame với chỉ số là năm và tháng
            monthly_returns = monthly_returns.to_frame('return')
            monthly_returns['year'] = monthly_returns.index.year
            monthly_returns['month'] = monthly_returns.index.month
            
            # Pivot để tạo bảng năm x tháng
            monthly_returns = monthly_returns.pivot(index='year', columns='month', values='return')
        
        try:
            if self.interactive:
                return self._plot_monthly_returns_heatmap_plotly(
                    strategy_name=strategy_name,
                    monthly_returns=monthly_returns,
                    cmap=cmap,
                    annot=annot,
                    save_path=save_path,
                    show_plot=show_plot
                )
            else:
                return self._plot_monthly_returns_heatmap_matplotlib(
                    strategy_name=strategy_name,
                    monthly_returns=monthly_returns,
                    figsize=figsize,
                    cmap=cmap,
                    annot=annot,
                    save_path=save_path,
                    show_plot=show_plot
                )
                
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ bản đồ nhiệt lợi nhuận: {e}")
            return None
    
    def _plot_monthly_returns_heatmap_matplotlib(
        self,
        strategy_name: str,
        monthly_returns: pd.DataFrame,
        figsize: Tuple[int, int],
        cmap: str,
        annot: bool,
        save_path: Optional[Union[str, Path]],
        show_plot: bool
    ) -> plt.Figure:
        """
        Vẽ bản đồ nhiệt lợi nhuận hàng tháng sử dụng Matplotlib.
        
        Args: (xem phương thức plot_monthly_returns_heatmap)
            
        Returns:
            Đối tượng Figure của matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Vẽ heatmap
        sns.heatmap(
            monthly_returns, 
            annot=annot, 
            fmt='.2%', 
            cmap=cmap, 
            center=0, 
            ax=ax,
            linewidths=.5,
            cbar_kws={'label': 'Lợi nhuận'}
        )
        
        # Tính các giá trị tổng hợp
        yearly_returns = monthly_returns.mean(axis=1)
        monthly_avg_returns = monthly_returns.mean(axis=0)
        
        ax.set_title(f'Lợi nhuận hàng tháng - {strategy_name}')
        ax.set_ylabel('Năm')
        ax.set_xlabel('Tháng')
        ax.set_xticklabels(['Th.1', 'Th.2', 'Th.3', 'Th.4', 'Th.5', 'Th.6', 
                           'Th.7', 'Th.8', 'Th.9', 'Th.10', 'Th.11', 'Th.12'])
        
        # Thêm thông tin thống kê
        monthly_stats = (
            f"Tháng tốt nhất: {monthly_avg_returns.idxmax()}, {monthly_avg_returns.max():.2%}\n"
            f"Tháng xấu nhất: {monthly_avg_returns.idxmin()}, {monthly_avg_returns.min():.2%}\n"
            f"Năm tốt nhất: {yearly_returns.idxmax()}, {yearly_returns.max():.2%}\n"
            f"Năm xấu nhất: {yearly_returns.idxmin()}, {yearly_returns.min():.2%}"
        )
        
        plt.figtext(0.02, 0.02, monthly_stats, fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            # Đảm bảo thư mục tồn tại
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Đã lưu bản đồ nhiệt lợi nhuận hàng tháng tại {save_path}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_monthly_returns_heatmap_plotly(
        self,
        strategy_name: str,
        monthly_returns: pd.DataFrame,
        cmap: str,
        annot: bool,
        save_path: Optional[Union[str, Path]],
        show_plot: bool
    ) -> go.Figure:
        """
        Vẽ bản đồ nhiệt lợi nhuận hàng tháng sử dụng Plotly.
        
        Args: (xem phương thức plot_monthly_returns_heatmap)
            
        Returns:
            Đối tượng Figure của plotly
        """
        # Tính các giá trị tổng hợp
        yearly_returns = monthly_returns.mean(axis=1)
        monthly_avg_returns = monthly_returns.mean(axis=0)
        
        # Chuyển đổi cmap từ matplotlib sang plotly
        colorscale = 'RdYlGn'
        if cmap == 'seismic':
            colorscale = 'RdBu'
        elif cmap == 'coolwarm':
            colorscale = 'RdBu'
        elif cmap == 'viridis':
            colorscale = 'Viridis'
        
        # Định dạng dữ liệu cho heatmap
        z = monthly_returns.values
        x = ['Th.1', 'Th.2', 'Th.3', 'Th.4', 'Th.5', 'Th.6', 
             'Th.7', 'Th.8', 'Th.9', 'Th.10', 'Th.11', 'Th.12']
        y = monthly_returns.index.astype(str)
        
        # Định dạng annotation
        text = None
        if annot:
            text = [[f'{val:.2%}' if not np.isnan(val) else '' for val in row] for row in z]
        
        # Tạo heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            zmid=0,
            text=text,
            texttemplate='%{text}',
            hoverongaps=False,
            hovertemplate='Năm: %{y}<br>Tháng: %{x}<br>Lợi nhuận: %{z:.2%}<extra></extra>'
        ))
        
        # Thêm thông tin thống kê
        best_month = monthly_avg_returns.idxmax()
        worst_month = monthly_avg_returns.idxmin()
        best_year = yearly_returns.idxmax()
        worst_year = yearly_returns.idxmin()
        
        annotations = [
            dict(
                x=0.5,
                y=-0.15,
                xref="paper",
                yref="paper",
                text=(f"Tháng tốt nhất: {best_month}, {monthly_avg_returns.max():.2%} | "
                      f"Tháng xấu nhất: {worst_month}, {monthly_avg_returns.min():.2%} | "
                      f"Năm tốt nhất: {best_year}, {yearly_returns.max():.2%} | "
                      f"Năm xấu nhất: {worst_year}, {yearly_returns.min():.2%}"),
                showarrow=False,
                font=dict(size=10),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
        ]
        
        # Cập nhật layout
        fig.update_layout(
            title=f'Lợi nhuận hàng tháng - {strategy_name}',
            xaxis_title='Tháng',
            yaxis_title='Năm',
            annotations=annotations,
            template="plotly_white" if self.style != "dark" else "plotly_dark"
        )
        
        # Lưu biểu đồ nếu cần
        if save_path:
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            # Đảm bảo thư mục tồn tại
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            extension = save_path.suffix.lower()
            if extension == '.html':
                fig.write_html(save_path)
            elif extension == '.json':
                fig.write_json(save_path)
            else:
                fig.write_image(save_path)
            
            self.logger.info(f"Đã lưu bản đồ nhiệt lợi nhuận hàng tháng tại {save_path}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_rolling_metrics(
        self,
        strategy_name: str,
        metrics: List[str] = ['returns', 'volatility', 'sharpe', 'drawdown'],
        window: int = 60,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> Optional[Union[plt.Figure, go.Figure]]:
        """
        Vẽ các chỉ số hiệu suất được tính theo cửa sổ trượt.
        
        Args:
            strategy_name: Tên chiến lược
            metrics: Danh sách các chỉ số cần vẽ
            window: Kích thước cửa sổ trượt (số ngày)
            figsize: Kích thước biểu đồ
            save_path: Đường dẫn lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Đối tượng Figure nếu thành công, None nếu thất bại
        """
        if strategy_name not in self.equity_curves:
            self.logger.error(f"Không tìm thấy equity curve cho chiến lược '{strategy_name}'")
            return None
        
        equity_curve = self.equity_curves[strategy_name]
        
        # Chọn cột equity
        if 'equity' in equity_curve.columns:
            equity_series = equity_curve['equity']
        else:
            # Sử dụng cột đầu tiên nếu không có cột 'equity'
            equity_series = equity_curve.iloc[:, 0]
        
        # Tính lợi nhuận hàng ngày
        daily_returns = equity_series.pct_change().dropna()
        
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
            rolling_equity = equity_series.rolling(window=window)
            rolling_max = equity_series.rolling(window=window).max()
            drawdown = (equity_series - rolling_max) / rolling_max
            df_metrics['drawdown'] = drawdown.rolling(window=window).min()
        
        # 5. Tỷ lệ thắng
        if 'win_rate' in metrics and strategy_name in self.trade_histories:
            trades = self.trade_histories[strategy_name]
            
            # Tính tỷ lệ thắng trong một cửa sổ thời gian
            if 'entry_time' in trades.columns and 'profit' in trades.columns:
                # Chuyển đổi entry_time thành datetime nếu chưa phải
                if not pd.api.types.is_datetime64_any_dtype(trades['entry_time']):
                    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
                
                # Set index là entry_time
                trades_by_date = trades.set_index('entry_time')
                
                # Tạo Series hàng ngày để đếm số giao dịch và số giao dịch thắng
                trade_counts = pd.Series(1, index=trades_by_date.index)
                win_counts = pd.Series(trades_by_date['profit'] > 0, index=trades_by_date.index)
                
                # Resample về tần suất hàng ngày
                daily_trade_counts = trade_counts.resample('D').sum().fillna(0)
                daily_win_counts = win_counts.resample('D').sum().fillna(0)
                
                # Tính tỷ lệ thắng theo cửa sổ trượt
                rolling_trades = daily_trade_counts.rolling(window=window).sum()
                rolling_wins = daily_win_counts.rolling(window=window).sum()
                
                # Tránh chia cho 0
                win_rate = rolling_wins / rolling_trades
                win_rate[rolling_trades == 0] = np.nan
                
                # Kết hợp với df_metrics (cần reindex để đảm bảo chỉ số khớp nhau)
                win_rate = win_rate.reindex(df_metrics.index)
                df_metrics['win_rate'] = win_rate
        
        # Loại bỏ các chỉ số không tồn tại
        metrics = [m for m in metrics if m in df_metrics.columns]
        
        if not metrics:
            self.logger.error("Không có chỉ số nào để vẽ")
            return None
        
        try:
            if self.interactive:
                return self._plot_rolling_metrics_plotly(
                    strategy_name=strategy_name,
                    df_metrics=df_metrics,
                    metrics=metrics,
                    window=window,
                    save_path=save_path,
                    show_plot=show_plot
                )
            else:
                return self._plot_rolling_metrics_matplotlib(
                    strategy_name=strategy_name,
                    df_metrics=df_metrics,
                    metrics=metrics,
                    window=window,
                    figsize=figsize,
                    save_path=save_path,
                    show_plot=show_plot
                )
                
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ các chỉ số hiệu suất: {e}")
            return None
    
    def _plot_rolling_metrics_matplotlib(
        self,
        strategy_name: str,
        df_metrics: pd.DataFrame,
        metrics: List[str],
        window: int,
        figsize: Tuple[int, int],
        save_path: Optional[Union[str, Path]],
        show_plot: bool
    ) -> plt.Figure:
        """
        Vẽ các chỉ số hiệu suất sử dụng Matplotlib.
        
        Args: (xem phương thức plot_rolling_metrics)
            
        Returns:
            Đối tượng Figure của matplotlib
        """
        # Xác định số lượng subplots
        n_metrics = len(metrics)
        
        # Tạo biểu đồ
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
        
        # Đảm bảo axes là list kể cả khi chỉ có 1 metric
        if n_metrics == 1:
            axes = [axes]
        
        # Vẽ từng chỉ số
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Định dạng metric name
            metric_name = metric.replace('_', ' ').title()
            
            # Vẽ đường
            ax.plot(df_metrics[metric], label=f'{metric_name} ({window}d)')
            
            # Thêm đường ngang tại 0 nếu cần
            if metric in ['returns', 'sharpe', 'drawdown']:
                ax.axhline(0, color='black', linestyle='--', alpha=0.3)
            
            # Thêm vùng tô màu theo điều kiện
            if metric == 'returns':
                ax.fill_between(df_metrics.index, 0, df_metrics[metric], 
                               where=df_metrics[metric] >= 0, color='green', alpha=0.3)
                ax.fill_between(df_metrics.index, 0, df_metrics[metric], 
                               where=df_metrics[metric] < 0, color='red', alpha=0.3)
            
            elif metric == 'drawdown':
                ax.fill_between(df_metrics.index, 0, df_metrics[metric], color='red', alpha=0.3)
            
            elif metric == 'sharpe':
                ax.fill_between(df_metrics.index, 0, df_metrics[metric], 
                               where=df_metrics[metric] >= 0, color='green', alpha=0.3)
                ax.fill_between(df_metrics.index, 0, df_metrics[metric], 
                               where=df_metrics[metric] < 0, color='red', alpha=0.3)
                
                # Thêm đường tham chiếu cho Sharpe tốt (>1)
                ax.axhline(1, color='green', linestyle='--', alpha=0.5)
            
            # Thiết lập trục y
            if metric == 'returns':
                ax.set_ylabel('Lợi nhuận')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2%}'.format(x)))
            elif metric == 'volatility':
                ax.set_ylabel('Độ biến động')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2%}'.format(x)))
            elif metric == 'sharpe':
                ax.set_ylabel('Sharpe Ratio')
            elif metric == 'drawdown':
                ax.set_ylabel('Drawdown')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2%}'.format(x)))
            elif metric == 'win_rate':
                ax.set_ylabel('Tỷ lệ thắng')
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2%}'.format(x)))
            
            # Thêm legend và grid
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Tính giá trị trung bình và hiển thị
            avg_value = df_metrics[metric].mean()
            ax.axhline(avg_value, color='blue', linestyle=':', alpha=0.7)
            
            # Thêm chú thích giá trị trung bình
            if metric in ['returns', 'volatility', 'drawdown', 'win_rate']:
                text = f'Trung bình: {avg_value:.2%}'
            else:
                text = f'Trung bình: {avg_value:.2f}'
            
            ax.annotate(text, xy=(0.02, 0.05), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Thiết lập trục x cho subplot cuối cùng
        axes[-1].set_xlabel('Thời gian')
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Thiết lập tiêu đề chính
        plt.suptitle(f'Các chỉ số hiệu suất theo cửa sổ trượt {window} ngày - {strategy_name}', fontsize=14)
        
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            # Đảm bảo thư mục tồn tại
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Đã lưu biểu đồ các chỉ số hiệu suất tại {save_path}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_rolling_metrics_plotly(
        self,
        strategy_name: str,
        df_metrics: pd.DataFrame,
        metrics: List[str],
        window: int,
        save_path: Optional[Union[str, Path]],
        show_plot: bool
    ) -> go.Figure:
        """
        Vẽ các chỉ số hiệu suất sử dụng Plotly.
        
        Args: (xem phương thức plot_rolling_metrics)
            
        Returns:
            Đối tượng Figure của plotly
        """
        # Xác định số lượng rows và cols
        n_metrics = len(metrics)
        rows = n_metrics
        cols = 1
        
        # Tạo tiêu đề cho mỗi subplot
        subplot_titles = []
        for metric in metrics:
            metric_name = metric.replace('_', ' ').title()
            subplot_titles.append(f"{metric_name} ({window}d)")
        
        # Tạo subplots
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=subplot_titles
        )
        
        # Vẽ từng chỉ số
        for i, metric in enumerate(metrics):
            row = i + 1
            
            # Đường chỉ số
            fig.add_trace(
                go.Scatter(
                    x=df_metrics.index,
                    y=df_metrics[metric],
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
                    y=[avg_value] * len(df_metrics),
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
            
            # Thiết lập định dạng trục y
            if metric in ['returns', 'volatility', 'drawdown', 'win_rate']:
                fig.update_yaxes(tickformat='.1%', row=row, col=1)
            
            # Thêm vùng tô màu
            if metric == 'returns':
                # Tạo 2 series cho giá trị dương và âm
                positive_y = df_metrics[metric].copy()
                positive_y[positive_y < 0] = 0
                
                negative_y = df_metrics[metric].copy()
                negative_y[negative_y > 0] = 0
                
                # Vùng lợi nhuận dương
                fig.add_trace(
                    go.Scatter(
                        x=df_metrics.index,
                        y=positive_y,
                        mode='none',
                        fill='tozeroy',
                        fillcolor='rgba(0, 255, 0, 0.2)',
                        name='Positive Returns',
                        showlegend=False
                    ),
                    row=row, col=1
                )
                
                # Vùng lợi nhuận âm
                fig.add_trace(
                    go.Scatter(
                        x=df_metrics.index,
                        y=negative_y,
                        mode='none',
                        fill='tozeroy',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        name='Negative Returns',
                        showlegend=False
                    ),
                    row=row, col=1
                )
            
            elif metric == 'drawdown':
                fig.add_trace(
                    go.Scatter(
                        x=df_metrics.index,
                        y=df_metrics[metric],
                        mode='none',
                        fill='tozeroy',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        name='Drawdown',
                        showlegend=False
                    ),
                    row=row, col=1
                )
            
            elif metric == 'sharpe':
                # Thêm đường tham chiếu cho Sharpe tốt (>1)
                fig.add_shape(
                    type="line",
                    x0=df_metrics.index[0], 
                    x1=df_metrics.index[-1],
                    y0=1, 
                    y1=1,
                    line=dict(color="green", width=1, dash="dash"),
                    row=row, col=1
                )
        
        # Cập nhật layout
        fig.update_layout(
            title=f'Các chỉ số hiệu suất theo cửa sổ trượt {window} ngày - {strategy_name}',
            template="plotly_white" if self.style != "dark" else "plotly_dark",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )
        
        # Cập nhật trục x
        fig.update_xaxes(title_text="Thời gian", row=rows, col=1)
        
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
        
        # Lưu biểu đồ nếu cần
        if save_path:
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            # Đảm bảo thư mục tồn tại
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            extension = save_path.suffix.lower()
            if extension == '.html':
                fig.write_html(save_path)
            elif extension == '.json':
                fig.write_json(save_path)
            else:
                fig.write_image(save_path)
            
            self.logger.info(f"Đã lưu biểu đồ các chỉ số hiệu suất tại {save_path}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_strategy_comparison(
        self,
        strategy_names: List[str],
        metrics: List[str] = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'],
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = True
    ) -> Optional[Union[plt.Figure, go.Figure]]:
        """
        Vẽ biểu đồ so sánh các chiến lược.
        
        Args:
            strategy_names: Danh sách tên các chiến lược
            metrics: Danh sách các chỉ số cần so sánh
            figsize: Kích thước biểu đồ
            save_path: Đường dẫn lưu biểu đồ
            show_plot: Hiển thị biểu đồ
            
        Returns:
            Đối tượng Figure nếu thành công, None nếu thất bại
        """
        if not self.backtest_results:
            self.logger.error("Không có kết quả backtest để so sánh")
            return None
        
        # Kiểm tra các chiến lược tồn tại
        valid_strategies = [s for s in strategy_names if s in self.backtest_results]
        if not valid_strategies:
            self.logger.error("Không tìm thấy chiến lược nào được chỉ định")
            return None
        
        # Tạo DataFrame để so sánh
        comparison_data = []
        
        for strategy_name in valid_strategies:
            result = self.backtest_results[strategy_name]
            
            # Lấy kết quả tổng hợp
            if result.get("combined_result"):
                row = {
                    "strategy": strategy_name,
                    "total_return": result["combined_result"].get("roi", 0),
                    "initial_balance": result["combined_result"].get("initial_balance", 0),
                    "final_balance": result["combined_result"].get("final_balance", 0)
                }
                
                # Thêm các metrics
                for metric in metrics:
                    if metric in result["combined_result"].get("metrics", {}):
                        row[metric] = result["combined_result"]["metrics"][metric]
                
                comparison_data.append(row)
            
            # Nếu không có kết quả tổng hợp, lấy từ symbol_results
            elif result.get("symbol_results"):
                for symbol, symbol_result in result["symbol_results"].items():
                    if isinstance(symbol_result, dict) and symbol_result.get("status") == "success":
                        row = {
                            "strategy": f"{strategy_name}_{symbol}",
                            "symbol": symbol,
                            "total_return": symbol_result.get("roi", 0),
                            "initial_balance": symbol_result.get("initial_balance", 0),
                            "final_balance": symbol_result.get("final_balance", 0)
                        }
                        
                        # Thêm các metrics
                        for metric in metrics:
                            if metric in symbol_result.get("metrics", {}):
                                row[metric] = symbol_result["metrics"][metric]
                        
                        comparison_data.append(row)
        
        if not comparison_data:
            self.logger.error("Không có dữ liệu để so sánh")
            return None
        
        # Tạo DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Đảm bảo các chỉ số cần so sánh tồn tại
        valid_metrics = [m for m in metrics if m in comparison_df.columns]
        if not valid_metrics:
            self.logger.error("Không có chỉ số nào hợp lệ để so sánh")
            return None
        
        try:
            if self.interactive:
                return self._plot_strategy_comparison_plotly(
                    comparison_df=comparison_df,
                    metrics=valid_metrics,
                    save_path=save_path,
                    show_plot=show_plot
                )
            else:
                return self._plot_strategy_comparison_matplotlib(
                    comparison_df=comparison_df,
                    metrics=valid_metrics,
                    figsize=figsize,
                    save_path=save_path,
                    show_plot=show_plot
                )
                
        except Exception as e:
            self.logger.error(f"Lỗi khi vẽ biểu đồ so sánh chiến lược: {e}")
            return None
    
    def _plot_strategy_comparison_matplotlib(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str],
        figsize: Tuple[int, int],
        save_path: Optional[Union[str, Path]],
        show_plot: bool
    ) -> plt.Figure:
        """
        Vẽ biểu đồ so sánh các chiến lược sử dụng Matplotlib.
        
        Args: (xem phương thức plot_strategy_comparison)
            
        Returns:
            Đối tượng Figure của matplotlib
        """
        # Xác định số lượng metrics
        n_metrics = len(metrics)
        
        # Tạo biểu đồ
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
        
        # Đảm bảo axes là list kể cả khi chỉ có 1 metric
        if n_metrics == 1:
            axes = [axes]
        
        # Các màu cho các chiến lược
        colors = plt.cm.tab10.colors
        
        # Vẽ từng chỉ số
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Định dạng metric name
            metric_name = metric.replace('_', ' ').title()
            
            # Định dạng dữ liệu
            data = comparison_df.set_index('strategy')[metric].sort_values(ascending=False)
            
            # Định dạng màu sắc cho các thanh
            if metric == 'max_drawdown':
                # Chuẩn hóa giá trị (nhỏ hơn là tốt hơn)
                norm = plt.Normalize(vmin=data.min(), vmax=data.max())
                colors_metric = plt.cm.RdYlGn_r(norm(data.values))  # RdYlGn_r: đảo ngược
            else:
                # Chuẩn hóa giá trị (lớn hơn là tốt hơn)
                norm = plt.Normalize(vmin=data.min(), vmax=data.max())
                colors_metric = plt.cm.RdYlGn(norm(data.values))
            
            # Vẽ biểu đồ cột
            bars = ax.bar(data.index, data.values, color=colors_metric)
            
            # Thêm giá trị trên mỗi cột
            for bar in bars:
                height = bar.get_height()
                if metric in ['total_return', 'max_drawdown']:
                    text = f'{height:.2%}'
                elif metric in ['win_rate']:
                    text = f'{height:.2%}'
                else:
                    text = f'{height:.2f}'
                
                ax.text(
                    bar.get_x() + bar.get_width()/2, 
                    height + (data.max() - data.min()) * 0.02,
                    text,
                    ha='center', 
                    va='bottom',
                    fontsize=9,
                    rotation=0
                )
            
            # Thiết lập trục y
            ax.set_title(f'{metric_name}')
            
            if metric in ['total_return', 'max_drawdown', 'win_rate']:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
            
            # Xoay nhãn trục x
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Thiết lập grid
            ax.grid(True, axis='y', alpha=0.3)
        
        plt.suptitle('So sánh hiệu suất các chiến lược', fontsize=14)
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            # Đảm bảo thư mục tồn tại
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Đã lưu biểu đồ so sánh chiến lược tại {save_path}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _plot_strategy_comparison_plotly(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str],
        save_path: Optional[Union[str, Path]],
        show_plot: bool
    ) -> go.Figure:
        """
        Vẽ biểu đồ so sánh các chiến lược sử dụng Plotly.
        
        Args: (xem phương thức plot_strategy_comparison)
            
        Returns:
            Đối tượng Figure của plotly
        """
        # Xác định số lượng metrics
        n_metrics = len(metrics)
        
        # Tạo tiêu đề cho mỗi subplot
        subplot_titles = [metric.replace('_', ' ').title() for metric in metrics]
        
        # Tạo subplots
        fig = make_subplots(
            rows=n_metrics, 
            cols=1,
            subplot_titles=subplot_titles
        )
        
        # Vẽ từng chỉ số
        for i, metric in enumerate(metrics):
            row = i + 1
            
            # Định dạng dữ liệu
            data = comparison_df.set_index('strategy')[metric].sort_values(ascending=False)
            
            # Định dạng màu sắc cho các thanh
            if metric == 'max_drawdown':
                # Nhỏ hơn là tốt hơn
                colors = data.map(lambda x: 'rgba(0, 255, 0, 0.7)' if x < data.median() else 'rgba(255, 0, 0, 0.7)')
            else:
                # Lớn hơn là tốt hơn
                colors = data.map(lambda x: 'rgba(0, 255, 0, 0.7)' if x > data.median() else 'rgba(255, 0, 0, 0.7)')
            
            # Định dạng giá trị
            if metric in ['total_return', 'max_drawdown', 'win_rate']:
                text = data.map(lambda x: f'{x:.2%}')
                hovertemplate = '%{x}: %{y:.2%}<extra></extra>'
            else:
                text = data.map(lambda x: f'{x:.2f}')
                hovertemplate = '%{x}: %{y:.2f}<extra></extra>'
            
            # Vẽ biểu đồ cột
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data.values,
                    text=text,
                    textposition='auto',
                    marker_color=colors,
                    hovertemplate=hovertemplate,
                    showlegend=False
                ),
                row=row, col=1
            )
            
            # Định dạng trục y
            if metric in ['total_return', 'max_drawdown', 'win_rate']:
                fig.update_yaxes(tickformat='.1%', row=row, col=1)
        
        # Cập nhật layout
        fig.update_layout(
            title='So sánh hiệu suất các chiến lược',
            template="plotly_white" if self.style != "dark" else "plotly_dark",
            showlegend=False
        )
        
        # Xoay nhãn trục x
        fig.update_xaxes(tickangle=45)
        
        # Thiết lập grid
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)' if self.style != "dark" else 'rgba(255,255,255,0.1)'
        )
        
        # Lưu biểu đồ nếu cần
        if save_path:
            if isinstance(save_path, str):
                save_path = Path(save_path)
            
            # Đảm bảo thư mục tồn tại
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            extension = save_path.suffix.lower()
            if extension == '.html':
                fig.write_html(save_path)
            elif extension == '.json':
                fig.write_json(save_path)
            else:
                fig.write_image(save_path)
            
            self.logger.info(f"Đã lưu biểu đồ so sánh chiến lược tại {save_path}")
        
        # Hiển thị biểu đồ nếu cần
        if show_plot:
            fig.show()
        
        return fig
    
    def create_backtest_report(
        self,
        strategy_name: str,
        output_path: Optional[Union[str, Path]] = None,
        include_sections: List[str] = [
            'summary', 'equity_curve', 'drawdown', 'monthly_returns', 
            'trade_distribution', 'rolling_metrics'
        ],
        format: str = 'html'
    ) -> Optional[str]:
        """
        Tạo báo cáo backtest tổng hợp.
        
        Args:
            strategy_name: Tên chiến lược
            output_path: Đường dẫn lưu báo cáo
            include_sections: Danh sách các phần cần đưa vào báo cáo
            format: Định dạng báo cáo ('html' hoặc 'pdf')
            
        Returns:
            Đường dẫn đến báo cáo nếu thành công, None nếu thất bại
        """
        if strategy_name not in self.backtest_results:
            self.logger.error(f"Không tìm thấy kết quả backtest cho chiến lược '{strategy_name}'")
            return None
        
        # Tạo thư mục báo cáo
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"report_{strategy_name}_{timestamp}.{format}"
        else:
            output_path = Path(output_path)
        
        # Đảm bảo thư mục tồn tại
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Tạo báo cáo HTML
        if format.lower() == 'html':
            try:
                import jinja2
                
                # Tạo thư mục cho các biểu đồ
                image_dir = output_path.parent / f"images_{strategy_name}"
                image_dir.mkdir(exist_ok=True)
                
                # Đường dẫn tới thư mục template (giả sử ở cùng thư mục với đoạn code này)
                template_dir = Path(__file__).parent / "templates"
                
                # Nếu không tìm thấy template, tạo template mặc định
                if not template_dir.exists():
                    template_dir.mkdir(exist_ok=True)
                    
                    # Tạo file template
                    template_path = template_dir / "report_template.html"
                    with open(template_path, "w", encoding="utf-8") as f:
                        f.write(self._get_default_html_template())
                
                # Tạo environment
                env = jinja2.Environment(
                    loader=jinja2.FileSystemLoader(template_dir),
                    autoescape=jinja2.select_autoescape(['html', 'xml'])
                )
                
                # Lấy template
                template = env.get_template("report_template.html")
                
                # Dữ liệu cho template
                report_data = self._prepare_report_data(
                    strategy_name=strategy_name,
                    include_sections=include_sections,
                    image_dir=image_dir
                )
                
                # Render template
                html_content = template.render(
                    title=f"Báo cáo Backtest - {strategy_name}",
                    strategy_name=strategy_name,
                    date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **report_data
                )
                
                # Lưu báo cáo
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                
                self.logger.info(f"Đã tạo báo cáo HTML tại {output_path}")
                return str(output_path)
                
            except ImportError:
                self.logger.error("Không thể tạo báo cáo HTML. Thiếu thư viện jinja2.")
                self.logger.info("Cài đặt thư viện jinja2 với lệnh: pip install jinja2")
                return None
            except Exception as e:
                self.logger.error(f"Lỗi khi tạo báo cáo HTML: {e}")
                return None
        
        # Tạo báo cáo PDF
        elif format.lower() == 'pdf':
            try:
                # Tạo báo cáo HTML trước
                html_path = output_path.with_suffix('.html')
                html_result = self.create_backtest_report(
                    strategy_name=strategy_name,
                    output_path=html_path,
                    include_sections=include_sections,
                    format='html'
                )
                
                if not html_result:
                    return None
                
                # Chuyển đổi HTML sang PDF
                from weasyprint import HTML
                
                # Chuyển đổi
                HTML(html_path).write_pdf(output_path)
                
                self.logger.info(f"Đã tạo báo cáo PDF tại {output_path}")
                return str(output_path)
                
            except ImportError:
                self.logger.error("Không thể tạo báo cáo PDF. Thiếu thư viện weasyprint.")
                self.logger.info("Cài đặt thư viện weasyprint với lệnh: pip install weasyprint")
                return None
            except Exception as e:
                self.logger.error(f"Lỗi khi tạo báo cáo PDF: {e}")
                return None
        
        else:
            self.logger.error(f"Định dạng '{format}' không được hỗ trợ. Chỉ hỗ trợ 'html' và 'pdf'.")
            return None
    
    def _prepare_report_data(
        self,
        strategy_name: str,
        include_sections: List[str],
        image_dir: Path
    ) -> Dict[str, Any]:
        """
        Chuẩn bị dữ liệu cho báo cáo.
        
        Args:
            strategy_name: Tên chiến lược
            include_sections: Danh sách các phần cần đưa vào báo cáo
            image_dir: Thư mục lưu các biểu đồ
            
        Returns:
            Dict chứa dữ liệu cho báo cáo
        """
        # Dữ liệu cho báo cáo
        report_data = {
            'summary': {},
            'images': {},
            'tables': {},
            'metrics': {},
            'sections': include_sections
        }
        
        # Lấy kết quả backtest
        result = self.backtest_results[strategy_name]
        
        # 1. Phần tóm tắt
        if 'summary' in include_sections:
            # Lấy thông tin cơ bản
            summary = {}
            
            if result.get("combined_result"):
                summary['total_return'] = result["combined_result"].get("roi", 0)
                summary['initial_balance'] = result["combined_result"].get("initial_balance", 0)
                summary['final_balance'] = result["combined_result"].get("final_balance", 0)
                
                # Lấy các metrics
                metrics = result["combined_result"].get("metrics", {})
                summary.update(metrics)
                
                # Lấy thông tin thời gian
                summary['start_time'] = result.get("start_time", "N/A")
                summary['end_time'] = result.get("end_time", "N/A")
                
                # Lấy thông tin cấu hình
                summary['strategy_params'] = result.get("strategy_params", {})
                summary['backtest_params'] = result.get("backtest_params", {})
            
            # Nếu không có kết quả tổng hợp, lấy từ symbol_results
            elif result.get("symbol_results"):
                # Tính tổng hợp từ các symbol
                total_initial_balance = 0
                total_final_balance = 0
                metrics_sum = {}
                metrics_count = {}
                
                for symbol, symbol_result in result["symbol_results"].items():
                    if isinstance(symbol_result, dict) and symbol_result.get("status") == "success":
                        total_initial_balance += symbol_result.get("initial_balance", 0)
                        total_final_balance += symbol_result.get("final_balance", 0)
                        
                        # Tính tổng các metrics
                        for metric, value in symbol_result.get("metrics", {}).items():
                            if metric not in metrics_sum:
                                metrics_sum[metric] = 0
                                metrics_count[metric] = 0
                            
                            metrics_sum[metric] += value
                            metrics_count[metric] += 1
                
                # Tính trung bình các metrics
                metrics_avg = {}
                for metric, total in metrics_sum.items():
                    if metrics_count[metric] > 0:
                        metrics_avg[metric] = total / metrics_count[metric]
                
                # Tính ROI tổng hợp
                if total_initial_balance > 0:
                    total_roi = (total_final_balance - total_initial_balance) / total_initial_balance
                else:
                    total_roi = 0
                
                summary['total_return'] = total_roi
                summary['initial_balance'] = total_initial_balance
                summary['final_balance'] = total_final_balance
                summary.update(metrics_avg)
                
                # Lấy thông tin thời gian
                summary['start_time'] = result.get("start_time", "N/A")
                summary['end_time'] = result.get("end_time", "N/A")
                
                # Lấy thông tin cấu hình
                summary['strategy_params'] = result.get("strategy_params", {})
                summary['backtest_params'] = result.get("backtest_params", {})
            
            report_data['summary'] = summary
            report_data['metrics'] = summary
        
        # 2. Phần đường cong vốn
        if 'equity_curve' in include_sections and strategy_name in self.equity_curves:
            # Tạo biểu đồ đường cong vốn
            equity_path = image_dir / f"equity_curve_{strategy_name}.html" if self.interactive else image_dir / f"equity_curve_{strategy_name}.png"
            
            self.plot_equity_curve(
                strategy_names=[strategy_name],
                include_drawdown=True,
                save_path=equity_path,
                show_plot=False
            )
            
            report_data['images']['equity_curve'] = equity_path.name
        
        # 3. Phần drawdown
        if 'drawdown' in include_sections and strategy_name in self.equity_curves:
            # Tạo biểu đồ phân tích drawdown
            drawdown_path = image_dir / f"drawdown_{strategy_name}.html" if self.interactive else image_dir / f"drawdown_{strategy_name}.png"
            
            self.plot_drawdown_analysis(
                strategy_name=strategy_name,
                save_path=drawdown_path,
                show_plot=False
            )
            
            report_data['images']['drawdown'] = drawdown_path.name
        
        # 4. Phần lợi nhuận theo tháng
        if 'monthly_returns' in include_sections and strategy_name in self.equity_curves:
            # Tạo biểu đồ lợi nhuận theo tháng
            monthly_path = image_dir / f"monthly_returns_{strategy_name}.html" if self.interactive else image_dir / f"monthly_returns_{strategy_name}.png"
            
            self.plot_monthly_returns_heatmap(
                strategy_name=strategy_name,
                save_path=monthly_path,
                show_plot=False
            )
            
            report_data['images']['monthly_returns'] = monthly_path.name
        
        # 5. Phần phân phối giao dịch
        if 'trade_distribution' in include_sections and strategy_name in self.trade_histories:
            # Tạo biểu đồ phân phối giao dịch
            if self.interactive:
                # Plotly trả về danh sách các biểu đồ
                for i in range(3):  # Giả sử có 3 biểu đồ
                    trade_path = image_dir / f"trade_distribution_{strategy_name}_{i+1}.html"
                    # File này sẽ được tạo tự động trong hàm plot_trade_distribution
                    
                report_data['images']['trade_distribution'] = f"trade_distribution_{strategy_name}_1.html"
            else:
                trade_path = image_dir / f"trade_distribution_{strategy_name}.png"
                
                self.plot_trade_distribution(
                    strategy_name=strategy_name,
                    save_path=trade_path,
                    show_plot=False
                )
                
                report_data['images']['trade_distribution'] = trade_path.name
        
        # 6. Phần chỉ số hiệu suất theo thời gian
        if 'rolling_metrics' in include_sections and strategy_name in self.equity_curves:
            # Tạo biểu đồ chỉ số hiệu suất
            rolling_path = image_dir / f"rolling_metrics_{strategy_name}.html" if self.interactive else image_dir / f"rolling_metrics_{strategy_name}.png"
            
            self.plot_rolling_metrics(
                strategy_name=strategy_name,
                save_path=rolling_path,
                show_plot=False
            )
            
            report_data['images']['rolling_metrics'] = rolling_path.name
        
        return report_data
    
    def _get_default_html_template(self) -> str:
        """
        Trả về template HTML mặc định cho báo cáo.
        
        Returns:
            Nội dung template HTML
        """
        return """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        h1, h2, h3 {
            color: #2c3e50;
        }
        
        h1 {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        h2 {
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        
        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-bottom: 30px;
        }
        
        .metric-box {
            width: 23%;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metric-title {
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .metric-value.positive {
            color: #27ae60;
        }
        
        .metric-value.negative {
            color: #c0392b;
        }
        
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        table, th, td {
            border: 1px solid #ddd;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        .footer {
            margin-top: 50px;
            text-align: center;
            font-size: 12px;
            color: #7f8c8d;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }
        
        .parameters {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        
        .parameter-group {
            margin-bottom: 15px;
        }
        
        .parameter-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .parameter-item {
            margin-left: 15px;
            font-family: monospace;
        }
        
        @media print {
            body {
                padding: 0;
                margin: 0;
            }
            
            .chart-container img {
                max-width: 100%;
                height: auto;
            }
            
            .page-break {
                page-break-before: always;
            }
        }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    
    <div class="generation-info">
        <p>Báo cáo được tạo vào: {{ date }}</p>
    </div>
    
    {% if 'summary' in sections %}
    <h2>Tóm tắt kết quả</h2>
    
    <div class="metrics-container">
        <div class="metric-box">
            <div class="metric-title">Lợi nhuận tổng</div>
            <div class="metric-value {% if metrics.total_return > 0 %}positive{% elif metrics.total_return < 0 %}negative{% endif %}">
                {{ "%.2f%%" | format(metrics.total_return * 100) }}
            </div>
        </div>
        
        {% if 'sharpe_ratio' in metrics %}
        <div class="metric-box">
            <div class="metric-title">Sharpe Ratio</div>
            <div class="metric-value {% if metrics.sharpe_ratio > 1 %}positive{% elif metrics.sharpe_ratio < 0 %}negative{% endif %}">
                {{ "%.2f" | format(metrics.sharpe_ratio) }}
            </div>
        </div>
        {% endif %}
        
        {% if 'max_drawdown' in metrics %}
        <div class="metric-box">
            <div class="metric-title">Max Drawdown</div>
            <div class="metric-value negative">
                {{ "%.2f%%" | format(metrics.max_drawdown * 100) }}
            </div>
        </div>
        {% endif %}
        
        {% if 'win_rate' in metrics %}
        <div class="metric-box">
            <div class="metric-title">Tỷ lệ thắng</div>
            <div class="metric-value {% if metrics.win_rate > 0.5 %}positive{% elif metrics.win_rate < 0.5 %}negative{% endif %}">
                {{ "%.2f%%" | format(metrics.win_rate * 100) }}
            </div>
        </div>
        {% endif %}
        
        {% if 'calmar_ratio' in metrics %}
        <div class="metric-box">
            <div class="metric-title">Calmar Ratio</div>
            <div class="metric-value {% if metrics.calmar_ratio > 1 %}positive{% elif metrics.calmar_ratio < 0 %}negative{% endif %}">
                {{ "%.2f" | format(metrics.calmar_ratio) }}
            </div>
        </div>
        {% endif %}
        
        {% if 'profit_factor' in metrics %}
        <div class="metric-box">
            <div class="metric-title">Profit Factor</div>
            <div class="metric-value {% if metrics.profit_factor > 1 %}positive{% elif metrics.profit_factor < 1 %}negative{% endif %}">
                {{ "%.2f" | format(metrics.profit_factor) }}
            </div>
        </div>
        {% endif %}
        
        {% if 'sortino_ratio' in metrics %}
        <div class="metric-box">
            <div class="metric-title">Sortino Ratio</div>
            <div class="metric-value {% if metrics.sortino_ratio > 1 %}positive{% elif metrics.sortino_ratio < 0 %}negative{% endif %}">
                {{ "%.2f" | format(metrics.sortino_ratio) }}
            </div>
        </div>
        {% endif %}
        
        {% if 'initial_balance' in metrics %}
        <div class="metric-box">
            <div class="metric-title">Vốn ban đầu</div>
            <div class="metric-value">
                {{ "{:,.2f}".format(metrics.initial_balance) }}
            </div>
        </div>
        {% endif %}
        
        {% if 'final_balance' in metrics %}
        <div class="metric-box">
            <div class="metric-title">Vốn cuối cùng</div>
            <div class="metric-value {% if metrics.final_balance > metrics.initial_balance %}positive{% elif metrics.final_balance < metrics.initial_balance %}negative{% endif %}">
                {{ "{:,.2f}".format(metrics.final_balance) }}
            </div>
        </div>
        {% endif %}
    </div>
    
    <div class="parameters">
        <div class="parameter-group">
            <div class="parameter-title">Tham số chiến lược:</div>
            {% for key, value in summary.strategy_params.items() %}
            <div class="parameter-item">{{ key }}: {{ value }}</div>
            {% endfor %}
        </div>
        
        <div class="parameter-group">
            <div class="parameter-title">Tham số backtest:</div>
            {% for key, value in summary.backtest_params.items() %}
            <div class="parameter-item">{{ key }}: {{ value }}</div>
            {% endfor %}
        </div>
    </div>
    {% endif %}
    
    {% if 'equity_curve' in sections and 'equity_curve' in images %}
    <h2>Đường cong vốn</h2>
    
    <div class="chart-container">
        {% if images.equity_curve.endswith('.html') %}
        <iframe src="./images_{{ strategy_name }}/{{ images.equity_curve }}" width="100%" height="600" frameborder="0"></iframe>
        {% else %}
        <img src="./images_{{ strategy_name }}/{{ images.equity_curve }}" alt="Đường cong vốn">
        {% endif %}
    </div>
    {% endif %}
    
    {% if 'drawdown' in sections and 'drawdown' in images %}
    <div class="page-break"></div>
    <h2>Phân tích Drawdown</h2>
    
    <div class="chart-container">
        {% if images.drawdown.endswith('.html') %}
        <iframe src="./images_{{ strategy_name }}/{{ images.drawdown }}" width="100%" height="600" frameborder="0"></iframe>
        {% else %}
        <img src="./images_{{ strategy_name }}/{{ images.drawdown }}" alt="Phân tích Drawdown">
        {% endif %}
    </div>
    {% endif %}
    
    {% if 'monthly_returns' in sections and 'monthly_returns' in images %}
    <div class="page-break"></div>
    <h2>Lợi nhuận theo tháng</h2>
    
    <div class="chart-container">
        {% if images.monthly_returns.endswith('.html') %}
        <iframe src="./images_{{ strategy_name }}/{{ images.monthly_returns }}" width="100%" height="600" frameborder="0"></iframe>
        {% else %}
        <img src="./images_{{ strategy_name }}/{{ images.monthly_returns }}" alt="Lợi nhuận theo tháng">
        {% endif %}
    </div>
    {% endif %}
    
    {% if 'trade_distribution' in sections and 'trade_distribution' in images %}
    <div class="page-break"></div>
    <h2>Phân phối giao dịch</h2>
    
    <div class="chart-container">
        {% if images.trade_distribution.endswith('.html') %}
        <iframe src="./images_{{ strategy_name }}/{{ images.trade_distribution }}" width="100%" height="600" frameborder="0"></iframe>
        {% else %}
        <img src="./images_{{ strategy_name }}/{{ images.trade_distribution }}" alt="Phân phối giao dịch">
        {% endif %}
    </div>
    {% endif %}
    
    {% if 'rolling_metrics' in sections and 'rolling_metrics' in images %}
    <div class="page-break"></div>
    <h2>Các chỉ số hiệu suất theo thời gian</h2>
    
    <div class="chart-container">
        {% if images.rolling_metrics.endswith('.html') %}
        <iframe src="./images_{{ strategy_name }}/{{ images.rolling_metrics }}" width="100%" height="600" frameborder="0"></iframe>
        {% else %}
        <img src="./images_{{ strategy_name }}/{{ images.rolling_metrics }}" alt="Các chỉ số hiệu suất theo thời gian">
        {% endif %}
    </div>
    {% endif %}
    
    <div class="footer">
        <p>Tạo bởi BacktestVisualizer - Hệ thống giao dịch tự động</p>
    </div>
</body>
</html>"""