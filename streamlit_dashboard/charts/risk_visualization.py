"""
Trực quan hóa rủi ro.
File này cung cấp các hàm và lớp để tạo biểu đồ hiển thị các chỉ số rủi ro
trong hệ thống giao dịch tự động, bao gồm drawdown, volatility, giá trị rủi ro,
và các biểu đồ đánh giá rủi ro khác nhau.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Any, Optional
import io
import base64
from PIL import Image
import matplotlib.dates as mdates

# Import các module từ hệ thống
from risk_management.risk_calculator import RiskCalculator
from risk_management.portfolio_manager import PortfolioManager
from backtesting.evaluation.risk_evaluator import RiskEvaluator
from backtesting.performance_metrics import PerformanceMetrics
from deployment.exchange_api.position_tracker import PositionTracker
from config.constants import TRADING_DAYS_PER_YEAR
from config.logging_config import get_logger
from logs.metrics.system_metrics import measure_execution_time


class RiskVisualization:
    """
    Lớp trực quan hóa rủi ro.
    Cung cấp các phương thức để tạo và hiển thị biểu đồ liên quan đến
    các chỉ số rủi ro trong giao dịch.
    """
    
    def __init__(self, theme: str = "streamlit"):
        """
        Khởi tạo đối tượng RiskVisualization.
        
        Args:
            theme: Chủ đề cho biểu đồ ('streamlit', 'dark', 'light')
        """
        self.theme = theme
        self.logger = get_logger("risk_visualization")
        
        # Thiết lập style cho matplotlib
        if theme == "dark":
            plt.style.use('dark_background')
            self.colors = {
                'primary': '#3366ff',
                'secondary': '#ff6b6b',
                'tertiary': '#4ecdc4',
                'warning': '#ffd166',
                'danger': '#f25f5c',
                'success': '#2ecc71',
                'background': '#1e1e1e',
                'text': '#ffffff'
            }
        else:
            plt.style.use('default')
            self.colors = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'tertiary': '#2ca02c',
                'warning': '#ffcc5c',
                'danger': '#d62728',
                'success': '#28a745',
                'background': '#ffffff',
                'text': '#333333'
            }
    
    @measure_execution_time()
    def plot_drawdown_chart(self, equity_curve: pd.Series, 
                           max_drawdown_threshold: float = 0.2, 
                           width: int = 800, 
                           height: int = 400) -> alt.Chart:
        """
        Tạo biểu đồ drawdown để hiển thị trong Streamlit.
        
        Args:
            equity_curve: Chuỗi giá trị vốn theo thời gian 
            max_drawdown_threshold: Ngưỡng drawdown tối đa (mặc định: 0.2 = 20%)
            width: Chiều rộng biểu đồ (pixel)
            height: Chiều cao biểu đồ (pixel)
            
        Returns:
            Biểu đồ Altair
        """
        # Tính drawdown
        if isinstance(equity_curve, pd.Series):
            equity_df = pd.DataFrame({'equity': equity_curve})
        else:
            equity_df = pd.DataFrame({'equity': equity_curve, 
                                    'date': equity_curve.index})
        
        # Đảm bảo index là datetime
        if not isinstance(equity_df.index, pd.DatetimeIndex):
            try:
                equity_df['date'] = pd.to_datetime(equity_df['date'])
                equity_df.set_index('date', inplace=True)
            except:
                # Tạo index giả nếu không có dữ liệu ngày
                equity_df.index = pd.date_range(
                    start=datetime.now() - timedelta(days=len(equity_df) - 1),
                    periods=len(equity_df)
                )
        
        # Tính rolling max và drawdown
        equity_df['rolling_max'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['rolling_max']) / equity_df['rolling_max']
        
        # Reset index để drawdown_df có cột date
        drawdown_df = equity_df.reset_index()
        drawdown_df = drawdown_df.rename(columns={'index': 'date'})
        
        # Đảm bảo cột date có kiểu datetime
        if 'date' in drawdown_df.columns and not pd.api.types.is_datetime64_any_dtype(drawdown_df['date']):
            drawdown_df['date'] = pd.to_datetime(drawdown_df['date'])
        
        # Tính drawdown lớn nhất
        max_drawdown = drawdown_df['drawdown'].min()
        
        # Tạo biểu đồ drawdown với Altair
        base = alt.Chart(drawdown_df).encode(
            x=alt.X('date:T', title='Thời gian')
        )
        
        # Vẽ đường drawdown
        drawdown_line = base.mark_area(
            color='red',
            opacity=0.5
        ).encode(
            y=alt.Y('drawdown:Q', title='Drawdown (%)', axis=alt.Axis(format='%')),
            tooltip=[
                alt.Tooltip('date:T', title='Ngày'),
                alt.Tooltip('drawdown:Q', title='Drawdown', format='.2%')
            ]
        )
        
        # Vẽ đường ngưỡng drawdown
        threshold_line = base.mark_rule(
            color='orange',
            strokeDash=[4, 4]
        ).encode(
            y=alt.datum(-max_drawdown_threshold)
        )
        
        # Vẽ đường max drawdown
        max_drawdown_line = base.mark_rule(
            color='red',
            strokeDash=[2, 2]
        ).encode(
            y=alt.datum(max_drawdown)
        )
        
        # Tạo layer với text annotation
        max_dd_text = base.mark_text(
            align='left',
            baseline='top',
            dx=5,
            dy=-5,
            color='red'
        ).encode(
            x=alt.value(width - 200),  # Vị trí cố định
            y=alt.datum(max_drawdown),
            text=alt.value(f'Max Drawdown: {max_drawdown:.2%}')
        )
        
        threshold_text = base.mark_text(
            align='left',
            baseline='top',
            dx=5,
            dy=-5,
            color='orange'
        ).encode(
            x=alt.value(width - 220),  # Vị trí cố định
            y=alt.datum(-max_drawdown_threshold),
            text=alt.value(f'Threshold: {max_drawdown_threshold:.2%}')
        )
        
        # Kết hợp các thành phần
        chart = (drawdown_line + threshold_line + max_drawdown_line + 
                max_dd_text + threshold_text).properties(
            width=width,
            height=height,
            title='Drawdown Over Time'
        ).interactive()
        
        return chart
    
    @measure_execution_time()
    def plot_drawdown_underwater(self, equity_curve: Union[pd.Series, List[float]], 
                               thresholds: List[float] = [0.05, 0.1, 0.2]) -> plt.Figure:
        """
        Tạo biểu đồ drawdown underwater với matplotlib.
        
        Args:
            equity_curve: Chuỗi giá trị vốn theo thời gian 
            thresholds: Các ngưỡng drawdown để hiển thị (mặc định: [0.05, 0.1, 0.2])
            
        Returns:
            Figure matplotlib
        """
        # Chuyển đổi dữ liệu
        if not isinstance(equity_curve, pd.Series):
            if isinstance(equity_curve, list):
                equity_curve = pd.Series(equity_curve)
            else:
                equity_curve = pd.Series(equity_curve.values)
        
        # Tính drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Vẽ biểu đồ drawdown
        ax.fill_between(drawdown.index, 0, drawdown, color=self.colors['danger'], alpha=0.3)
        
        # Thêm các ngưỡng
        colors = [self.colors['warning'], self.colors['secondary'], self.colors['danger']]
        for i, threshold in enumerate(thresholds):
            ax.axhline(y=-threshold, color=colors[i % len(colors)], 
                      linestyle='--', alpha=0.7, label=f'{threshold*100:.0f}%')
        
        # Định dạng trục và tiêu đề
        ax.set_title('Drawdown Underwater Chart', fontsize=14)
        ax.set_ylabel('Drawdown', fontsize=12)
        ax.set_xlabel('Thời gian', fontsize=12)
        
        # Định dạng grid
        ax.grid(True, alpha=0.3)
        
        # Định dạng date ticks nếu index là datetime
        if isinstance(drawdown.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
        
        # Định dạng y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Thêm legend
        ax.legend()
        
        # Thêm thông tin max drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax.text(0.02, 0.02, f'Max Drawdown: {max_dd:.2%} on {max_dd_date}',
               transform=ax.transAxes, backgroundcolor='white', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    @measure_execution_time()
    def plot_risk_reward_profile(self, returns: pd.Series, 
                               risk_free_rate: float = 0.02,
                               benchmark_returns: Optional[pd.Series] = None) -> plt.Figure:
        """
        Tạo biểu đồ risk-reward profile.
        
        Args:
            returns: Chuỗi lợi nhuận theo thời gian
            risk_free_rate: Lãi suất phi rủi ro (mặc định: 0.02 = 2%)
            benchmark_returns: Chuỗi lợi nhuận của benchmark (optional)
            
        Returns:
            Figure matplotlib
        """
        # Tính toán các chỉ số cần thiết
        annualized_return = (1 + returns.mean()) ** TRADING_DAYS_PER_YEAR - 1
        annualized_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Tính Sharpe ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Vẽ điểm chiến lược
        ax.scatter(annualized_volatility * 100, annualized_return * 100, 
                 s=100, color=self.colors['primary'], marker='o', label='Strategy')
        
        # Vẽ benchmark nếu có
        if benchmark_returns is not None and not benchmark_returns.empty:
            benchmark_annual_return = (1 + benchmark_returns.mean()) ** TRADING_DAYS_PER_YEAR - 1
            benchmark_annual_vol = benchmark_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            
            ax.scatter(benchmark_annual_vol * 100, benchmark_annual_return * 100,
                     s=100, color=self.colors['secondary'], marker='^', label='Benchmark')
        
        # Vẽ đường risk-free
        ax.axhline(y=risk_free_rate * 100, color=self.colors['tertiary'], 
                 linestyle='--', label=f'Risk-Free ({risk_free_rate*100:.1f}%)')
        
        # Vẽ đường Sharpe ratio
        x_range = np.linspace(0, max(annualized_volatility * 2, 0.3) * 100, 100)
        y_range = risk_free_rate * 100 + sharpe_ratio * x_range
        ax.plot(x_range, y_range, color=self.colors['success'], 
              linestyle=':', label=f'Sharpe = {sharpe_ratio:.2f}')
        
        # Định dạng biểu đồ
        ax.set_title('Risk-Return Profile', fontsize=14)
        ax.set_xlabel('Risk (Annualized Volatility %)', fontsize=12)
        ax.set_ylabel('Return (Annualized %)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Thêm annotations
        text = (f"Return: {annualized_return*100:.2f}%\n"
               f"Volatility: {annualized_volatility*100:.2f}%\n"
               f"Sharpe: {sharpe_ratio:.2f}")
        
        ax.text(0.02, 0.02, text, transform=ax.transAxes,
              bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    @measure_execution_time()
    def plot_var_analysis(self, returns: pd.Series, 
                        confidence_levels: List[float] = [0.95, 0.99],
                        use_plotly: bool = True) -> Union[go.Figure, plt.Figure]:
        """
        Tạo biểu đồ phân tích Value at Risk (VaR).
        
        Args:
            returns: Chuỗi lợi nhuận theo thời gian
            confidence_levels: Các mức độ tin cậy cho VaR
            use_plotly: Sử dụng plotly thay vì matplotlib
            
        Returns:
            Figure (plotly hoặc matplotlib)
        """
        # Tính VaR cho các mức độ tin cậy
        var_values = {}
        cvar_values = {}  # Conditional VaR (Expected Shortfall)
        
        for cl in confidence_levels:
            # Historical VaR
            var_pct = np.percentile(returns, 100 * (1 - cl))
            var_values[cl] = abs(var_pct)
            
            # Conditional VaR (Expected Shortfall)
            cvar_pct = returns[returns <= var_pct].mean()
            cvar_values[cl] = abs(cvar_pct)
        
        if use_plotly:
            # Tạo dataframe cho dễ hiển thị
            var_df = pd.DataFrame({
                'Confidence': [f"{cl*100:.0f}%" for cl in confidence_levels],
                'VaR': [var_values[cl] for cl in confidence_levels],
                'CVaR': [cvar_values[cl] for cl in confidence_levels]
            })
            
            # Tạo biểu đồ
            fig = go.Figure()
            
            # Thêm VaR bars
            fig.add_trace(go.Bar(
                x=var_df['Confidence'],
                y=var_df['VaR'],
                name='Value at Risk (VaR)',
                marker_color='indianred'
            ))
            
            # Thêm CVaR bars
            fig.add_trace(go.Bar(
                x=var_df['Confidence'],
                y=var_df['CVaR'],
                name='Conditional VaR (Expected Shortfall)',
                marker_color='rgb(26, 118, 255)'
            ))
            
            # Cập nhật layout
            fig.update_layout(
                title='Value at Risk Analysis',
                xaxis_title='Confidence Level',
                yaxis_title='Loss as % of Portfolio',
                yaxis_tickformat='.2%',
                barmode='group',
                legend=dict(x=0.01, y=0.99),
                template='plotly_white' if self.theme != 'dark' else 'plotly_dark'
            )
            
            return fig
        else:
            # Sử dụng matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Tạo dataframe
            var_df = pd.DataFrame({
                'Confidence': [f"{cl*100:.0f}%" for cl in confidence_levels],
                'VaR': [var_values[cl] for cl in confidence_levels],
                'CVaR': [cvar_values[cl] for cl in confidence_levels]
            })
            
            # Vẽ bar chart
            x = np.arange(len(confidence_levels))
            width = 0.35
            
            ax.bar(x - width/2, var_df['VaR'], width, 
                 label='Value at Risk (VaR)', color=self.colors['danger'])
            ax.bar(x + width/2, var_df['CVaR'], width, 
                 label='Conditional VaR (ES)', color=self.colors['warning'])
            
            # Định dạng biểu đồ
            ax.set_title('Value at Risk Analysis', fontsize=14)
            ax.set_ylabel('Loss as % of Portfolio', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(var_df['Confidence'])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2%}'))
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Thêm giá trị lên bar
            for i, v in enumerate(var_df['VaR']):
                ax.text(i - width/2, v + 0.001, f'{v:.2%}', 
                      ha='center', va='bottom', color='black', fontweight='bold')
            
            for i, v in enumerate(var_df['CVaR']):
                ax.text(i + width/2, v + 0.001, f'{v:.2%}', 
                      ha='center', va='bottom', color='black', fontweight='bold')
            
            plt.tight_layout()
            return fig
    
    @measure_execution_time()
    def plot_returns_distribution(self, returns: pd.Series, 
                                benchmark_returns: Optional[pd.Series] = None) -> go.Figure:
        """
        Tạo biểu đồ phân phối lợi nhuận.
        
        Args:
            returns: Chuỗi lợi nhuận theo thời gian
            benchmark_returns: Chuỗi lợi nhuận của benchmark (optional)
            
        Returns:
            Figure plotly
        """
        # Tạo figure
        fig = go.Figure()
        
        # Thêm histogram cho chiến lược
        fig.add_trace(go.Histogram(
            x=returns,
            name='Strategy Returns',
            opacity=0.75,
            marker_color=self.colors['primary'],
            xbins=dict(size=0.005),  # Kích thước bin 0.5%
            histnorm='probability density'  # Chuẩn hóa histogram
        ))
        
        # Thêm đường KDE cho chiến lược
        kde_x, kde_y = self._calculate_kde(returns)
        fig.add_trace(go.Scatter(
            x=kde_x,
            y=kde_y,
            mode='lines',
            name='Strategy Density',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # Thêm đường phân phối chuẩn
        mean = returns.mean()
        std = returns.std()
        normal_x = np.linspace(returns.min(), returns.max(), 100)
        normal_y = self._normal_pdf(normal_x, mean, std)
        
        fig.add_trace(go.Scatter(
            x=normal_x,
            y=normal_y,
            mode='lines',
            name='Normal Distribution',
            line=dict(color=self.colors['tertiary'], width=2, dash='dash')
        ))
        
        # Thêm benchmark nếu có
        if benchmark_returns is not None and not benchmark_returns.empty:
            fig.add_trace(go.Histogram(
                x=benchmark_returns,
                name='Benchmark Returns',
                opacity=0.6,
                marker_color=self.colors['secondary'],
                xbins=dict(size=0.005),
                histnorm='probability density'
            ))
            
            # Thêm đường KDE cho benchmark
            kde_x_bm, kde_y_bm = self._calculate_kde(benchmark_returns)
            fig.add_trace(go.Scatter(
                x=kde_x_bm,
                y=kde_y_bm,
                mode='lines',
                name='Benchmark Density',
                line=dict(color=self.colors['secondary'], width=2)
            ))
        
        # Thêm đường dọc tại 0
        fig.add_vline(x=0, line_dash="solid", line_color="green", opacity=0.7)
        
        # Đánh dấu phần đuôi trái
        left_tail_x = np.percentile(returns, 5)
        fig.add_vline(x=left_tail_x, line_dash="dash", line_color="red", opacity=0.7)
        
        # Tính các chỉ số thống kê
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        pos_returns = (returns > 0).mean() * 100
        
        # Định dạng biểu đồ
        fig.update_layout(
            title=f'Returns Distribution - Skew: {skewness:.2f}, Kurt: {kurtosis:.2f}, Pos: {pos_returns:.1f}%',
            xaxis_title='Daily Returns',
            yaxis_title='Density',
            barmode='overlay',
            xaxis=dict(tickformat='.1%'),
            template='plotly_white' if self.theme != 'dark' else 'plotly_dark',
            legend=dict(x=0.01, y=0.99),
            height=500
        )
        
        return fig
    
    def _calculate_kde(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tính toán Kernel Density Estimation.
        
        Args:
            data: Series dữ liệu
            
        Returns:
            Tuple của (x, y) cho đường KDE
        """
        from scipy.stats import gaussian_kde
        
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 1000)
        y = kde(x)
        
        return x, y
    
    def _normal_pdf(self, x: np.ndarray, mean: float, std: float) -> np.ndarray:
        """
        Tính hàm mật độ xác suất của phân phối chuẩn.
        
        Args:
            x: Mảng giá trị x
            mean: Giá trị trung bình
            std: Độ lệch chuẩn
            
        Returns:
            Mảng giá trị PDF
        """
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - mean)**2 / (2 * std**2))
    
    @measure_execution_time()
    def plot_volatility_chart(self, returns: pd.Series, 
                            rolling_windows: List[int] = [20, 60]) -> plt.Figure:
        """
        Tạo biểu đồ biến động theo thời gian.
        
        Args:
            returns: Chuỗi lợi nhuận theo thời gian
            rolling_windows: Danh sách các khoảng thời gian để tính biến động (ngày)
            
        Returns:
            Figure matplotlib
        """
        # Tạo figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Vẽ biến động theo từng khoảng thời gian
        for window in rolling_windows:
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            label = f'{window} ngày' if window < TRADING_DAYS_PER_YEAR else f'{window/TRADING_DAYS_PER_YEAR:.1f} năm'
            ax.plot(rolling_vol.index, rolling_vol * 100, linewidth=2, label=label)
        
        # Thêm ngưỡng biến động
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10%')
        ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='20%')
        ax.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='30%')
        
        # Định dạng biểu đồ
        ax.set_title('Biến động theo thời gian', fontsize=14)
        ax.set_ylabel('Biến động hàng năm (%)', fontsize=12)
        ax.set_xlabel('Thời gian', fontsize=12)
        
        # Định dạng date ticks nếu index là datetime
        if isinstance(returns.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @measure_execution_time()
    def plot_risk_radar(self, risk_scores: Dict[str, Dict[str, Any]],
                       risk_weights: Optional[Dict[str, float]] = None) -> go.Figure:
        """
        Tạo biểu đồ radar cho đánh giá rủi ro.
        
        Args:
            risk_scores: Dictionary các điểm rủi ro theo từng loại
            risk_weights: Dictionary trọng số của từng loại rủi ro (optional)
            
        Returns:
            Figure plotly
        """
        # Mặc định risk weights nếu không được cung cấp
        if risk_weights is None:
            risk_weights = {
                "drawdown": 0.35,
                "volatility": 0.25,
                "var": 0.15,
                "risk_return": 0.15,
                "tail_risk": 0.10
            }
        
        # Chuẩn bị dữ liệu
        categories = list(risk_scores.keys())
        normalized_scores = []
        
        for category in categories:
            # Chuẩn hóa về thang 0-1 (0 là tốt nhất, 1 là tệ nhất)
            normalized_score = (risk_scores[category]["score"] - 1) / 4
            normalized_scores.append(normalized_score)
        
        # Thêm giá trị đầu vào cuối để khép kín biểu đồ
        categories.append(categories[0])
        normalized_scores.append(normalized_scores[0])
        
        # Tạo biểu đồ với plotly
        fig = go.Figure()
        
        # Thêm radar chart
        fig.add_trace(go.Scatterpolar(
            r=normalized_scores,
            theta=categories,
            fill='toself',
            fillcolor=self.colors['primary'],
            line=dict(color=self.colors['primary']),
            opacity=0.7,
            name='Risk Score'
        ))
        
        # Thêm các đường tròn chỉ mức độ
        for i in range(5):
            level = (i + 1) / 5
            
            # Tạo điểm ngoài biểu đồ để hiển thị các đường tròn
            fig.add_trace(go.Scatterpolar(
                r=[level] * len(categories),
                theta=categories,
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.3,
                hoverinfo='skip',
                name=f'Level {i+1}' if i == 0 else '',
                showlegend=(i == 0)
            ))
        
        # Tính tổng điểm rủi ro có trọng số
        weighted_risk_score = sum(risk_scores[cat]["score"] * risk_weights.get(cat, 0.2) 
                                for cat in risk_scores.keys())
        
        # Xác định mức độ rủi ro tổng thể
        if weighted_risk_score >= 4.5:
            overall_risk_level = "Rất cao"
            risk_color = 'red'
        elif weighted_risk_score >= 3.5:
            overall_risk_level = "Cao"
            risk_color = 'orange'
        elif weighted_risk_score >= 2.5:
            overall_risk_level = "Trung bình cao"
            risk_color = 'amber'
        elif weighted_risk_score >= 1.5:
            overall_risk_level = "Trung bình"
            risk_color = 'yellow'
        else:
            overall_risk_level = "Thấp"
            risk_color = 'green'
        
        # Định dạng biểu đồ
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0.2, 0.4, 0.6, 0.8],
                    ticktext=['Thấp', 'Trung bình', 'Cao', 'Rất cao']
                )
            ),
            showlegend=False,
            title=f'Đánh giá rủi ro tổng thể: {overall_risk_level} ({weighted_risk_score:.2f}/5)',
            title_font=dict(size=16, color=risk_color),
            template='plotly_white' if self.theme != 'dark' else 'plotly_dark',
            annotations=[
                dict(
                    text=f"Điểm rủi ro: {weighted_risk_score:.2f}/5.0",
                    x=0.5, y=-0.1,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=14)
                )
            ],
            height=500,
            margin=dict(t=100, b=100)
        )
        
        return fig
    
    @measure_execution_time()
    def plot_stress_test(self, risk_evaluator: RiskEvaluator, 
                       scenario_name: str = "2008_crisis") -> go.Figure:
        """
        Tạo biểu đồ mô phỏng stress test.
        
        Args:
            risk_evaluator: Đối tượng RiskEvaluator đã được khởi tạo
            scenario_name: Tên kịch bản stress test ('2008_crisis', '2020_covid', 'average_correction', etc.)
            
        Returns:
            Figure plotly
        """
        # Thực hiện stress test
        stress_results = risk_evaluator.calculate_stress_test(scenario_name=scenario_name)
        
        # Lấy kết quả
        scenario = stress_results["scenario"]
        results = stress_results["results"]
        impact = stress_results["impact"]
        
        # Xây dựng lại đường cong equity từ kết quả
        equity_curve = risk_evaluator.equity_curve
        last_equity = equity_curve.iloc[-1]
        
        # Tạo chuỗi thời gian
        stress_days = scenario["duration_days"]
        recovery_days = int(scenario["stress_percent"] / scenario["recovery_rate"])
        simulation_days = stress_days + recovery_days
        dates = pd.date_range(
            start=equity_curve.index[-1] + pd.Timedelta(days=1), 
            periods=simulation_days
        )
        
        # Tạo equity curve cho stress test
        stressed_equity = []
        
        # Giai đoạn stress
        for i in range(stress_days):
            day_stress = (1 - scenario["stress_percent"] * (i + 1) / stress_days)
            stressed_equity.append(last_equity * day_stress)
        
        # Giai đoạn phục hồi
        recovery_start = stressed_equity[-1]
        for i in range(recovery_days):
            day_recovery = recovery_start * (1 + scenario["recovery_rate"] * (i + 1))
            # Không vượt quá giá trị ban đầu
            stressed_equity.append(min(day_recovery, last_equity))
        
        # Tạo DataFrame cho dễ hiển thị
        df_equity = pd.DataFrame({'equity': equity_curve})
        df_stress = pd.DataFrame({
            'date': dates,
            'equity': stressed_equity
        })
        
        # Tạo biểu đồ với plotly
        fig = go.Figure()
        
        # Thêm equity curve thực tế
        fig.add_trace(go.Scatter(
            x=df_equity.index,
            y=df_equity['equity'],
            name='Equity thực tế',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # Thêm equity curve stress test
        fig.add_trace(go.Scatter(
            x=df_stress['date'],
            y=df_stress['equity'],
            name='Mô phỏng stress test',
            line=dict(color=self.colors['danger'], width=2, dash='dash')
        ))
        
        # Thêm các đường đánh dấu
        stress_start = dates[0]
        stress_end = dates[stress_days - 1]
        
        fig.add_vline(
            x=stress_start, 
            line_width=1, 
            line_dash="dash", 
            line_color=self.colors['warning'],
            annotation_text="Bắt đầu stress"
        )
        
        fig.add_vline(
            x=stress_end, 
            line_width=1, 
            line_dash="dash", 
            line_color=self.colors['success'],
            annotation_text="Kết thúc stress"
        )
        
        # Thêm vạch phục hồi hoàn toàn nếu có
        if results["full_recovery_days"] is not None:
            recovery_date = dates[results["full_recovery_days"] - 1]
            fig.add_vline(
                x=recovery_date, 
                line_width=1, 
                line_dash="dash", 
                line_color=self.colors['tertiary'],
                annotation_text="Phục hồi hoàn toàn"
            )
        
        # Định dạng biểu đồ
        fig.update_layout(
            title=f'Stress Test: {scenario_name.replace("_", " ").title()}',
            xaxis_title='Thời gian',
            yaxis_title='Equity',
            template='plotly_white' if self.theme != 'dark' else 'plotly_dark',
            annotations=[
                dict(
                    x=0.5, y=1.05,
                    xref='paper', yref='paper',
                    text=(f"Kịch bản: Giảm {scenario['stress_percent']*100:.0f}% "
                         f"trong {scenario['duration_days']} ngày, "
                         f"Phục hồi {scenario['recovery_rate']*100:.2f}%/ngày, "
                         f"Tác động: {impact}"),
                    showarrow=False,
                    font=dict(size=12)
                )
            ],
            legend=dict(x=0.01, y=0.99),
            height=500
        )
        
        # Hiển thị thêm thống kê
        min_equity = results["min_equity"]
        max_drawdown_amount = results["max_drawdown_amount"]
        max_drawdown_percent = results["max_drawdown_percent"]
        recovery_days = results["full_recovery_days"]
        
        stats_text = (
            f"Min Equity: {min_equity:.2f}<br>"
            f"Max Drawdown: {max_drawdown_amount:.2f} ({max_drawdown_percent*100:.2f}%)<br>"
            f"Vốn sau stress: {results['capital_after_stress']:.2f}<br>"
            f"P&L: {results['profit_loss_after_stress']:.2f} ({results['profit_loss_percent']*100:.2f}%)"
        )
        
        if recovery_days is not None:
            stats_text += f"<br>Thời gian phục hồi: {recovery_days} ngày"
        
        fig.add_annotation(
            x=0.99, y=0.01,
            xref='paper', yref='paper',
            text=stats_text,
            showarrow=False,
            font=dict(size=12),
            align='right',
            bgcolor='white' if self.theme != 'dark' else 'black',
            bordercolor='black' if self.theme != 'dark' else 'white',
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )
        
        return fig
    
    @measure_execution_time()
    def plot_monte_carlo_simulation(self, risk_evaluator: RiskEvaluator,
                                  num_simulations: int = 1000,
                                  days_forward: int = 252,
                                  confidence_level: float = 0.95) -> go.Figure:
        """
        Tạo biểu đồ mô phỏng Monte Carlo.
        
        Args:
            risk_evaluator: Đối tượng RiskEvaluator đã được khởi tạo
            num_simulations: Số lần mô phỏng
            days_forward: Số ngày mô phỏng tương lai
            confidence_level: Mức độ tin cậy
            
        Returns:
            Figure plotly
        """
        # Thực hiện mô phỏng
        simulation = risk_evaluator.run_monte_carlo_simulation(
            num_simulations=num_simulations,
            days_forward=days_forward,
            confidence_level=confidence_level
        )
        
        # Lấy kết quả hiện có
        equity_curve = risk_evaluator.equity_curve
        last_equity = equity_curve.iloc[-1]
        
        # Lấy các tham số từ kết quả mô phỏng
        final_values = simulation["final_values"]
        risk_metrics = simulation["risk_metrics"]
        
        # Tạo biểu đồ
        fig = go.Figure()
        
        # Thêm 50 đường đại diện (để biểu đồ không quá rối)
        paths_to_show = min(100, num_simulations)
        
        # Khởi tạo mảng cho giá trị trung bình, phân vị 5%, và phân vị 95%
        mean_path = np.zeros(days_forward + 1)
        percentile_5 = np.zeros(days_forward + 1)
        percentile_95 = np.zeros(days_forward + 1)
        
        # Khởi tạo mảng 3D để lưu tất cả các đường
        all_paths = np.zeros((num_simulations, days_forward + 1))
        all_paths[:, 0] = last_equity
        
        # Tạo các đường đại diện và lấy thông tin từ simulation_chart
        time_points = np.arange(days_forward + 1)
        
        # Thêm các đường mô phỏng vào biểu đồ
        for i in range(paths_to_show):
            # Lấy dữ liệu từ biểu đồ gốc
            path_y = simulation["simulation_chart"]  # Cần thay bằng dữ liệu thực
            
            # Thêm đường
            fig.add_trace(go.Scatter(
                x=time_points,
                y=path_y,
                mode='lines',
                line=dict(color=self.colors['primary'], width=0.5),
                opacity=0.1,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Thêm đường trung bình
        fig.add_trace(go.Scatter(
            x=time_points,
            y=np.mean(all_paths, axis=0),
            mode='lines',
            name='Trung bình',
            line=dict(color=self.colors['tertiary'], width=3)
        ))
        
        # Thêm đường phân vị
        fig.add_trace(go.Scatter(
            x=time_points,
            y=np.percentile(all_paths, 5, axis=0),
            mode='lines',
            name=f'Phân vị 5%',
            line=dict(color=self.colors['danger'], width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=np.percentile(all_paths, 95, axis=0),
            mode='lines',
            name=f'Phân vị 95%',
            line=dict(color=self.colors['success'], width=2, dash='dash')
        ))
        
        # Thêm vùng khoảng tin cậy
        fig.add_trace(go.Scatter(
            x=np.concatenate([time_points, time_points[::-1]]),
            y=np.concatenate([
                np.percentile(all_paths, 5, axis=0),
                np.percentile(all_paths, 95, axis=0)[::-1]
            ]),
            fill='toself',
            fillcolor=self.colors['primary'],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False,
            opacity=0.2
        ))
        
        # Định dạng biểu đồ
        fig.update_layout(
            title=f'Monte Carlo Simulation ({num_simulations} paths, {days_forward} days)',
            xaxis_title='Days',
            yaxis_title='Equity',
            template='plotly_white' if self.theme != 'dark' else 'plotly_dark',
            legend=dict(x=0.01, y=0.99),
            height=500
        )
        
        # Thêm thông tin thống kê
        stats_text = (
            f"VaR ({confidence_level*100:.0f}%): {risk_metrics['var']:.2f} ({risk_metrics['var_percent']*100:.2f}%)<br>"
            f"Expected Max Drawdown: {risk_metrics['expected_max_drawdown']*100:.2f}%<br>"
            f"Worst Case Drawdown: {risk_metrics['worst_case_drawdown']*100:.2f}%<br>"
            f"Profit Probability: {risk_metrics['profit_probability']*100:.2f}%<br>"
        )
        
        fig.add_annotation(
            x=0.99, y=0.01,
            xref='paper', yref='paper',
            text=stats_text,
            showarrow=False,
            font=dict(size=12),
            align='right',
            bgcolor='white' if self.theme != 'dark' else 'black',
            bordercolor='black' if self.theme != 'dark' else 'white',
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )
        
        return fig
    
    @measure_execution_time()
    def plot_portfolio_allocation(self, portfolio_manager: PortfolioManager) -> go.Figure:
        """
        Tạo biểu đồ phân bổ danh mục đầu tư.
        
        Args:
            portfolio_manager: Đối tượng PortfolioManager đã được khởi tạo
            
        Returns:
            Figure plotly
        """
        # Lấy trạng thái danh mục hiện tại
        portfolio_status = portfolio_manager.get_portfolio_status()
        
        # Lấy phân bổ theo category
        category_allocation = portfolio_status.get("category_allocation", {})
        
        # Sắp xếp các category theo giá trị
        sorted_categories = sorted(
            category_allocation.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Tạo danh sách labels và values
        labels = [cat for cat, val in sorted_categories]
        values = [val for cat, val in sorted_categories]
        
        # Thêm dữ liệu vốn còn lại
        cash_percent = portfolio_status.get("current_capital", 0) / portfolio_status.get("portfolio_value", 1) * 100
        if cash_percent > 0:
            labels.append("CASH")
            values.append(cash_percent)
        
        # Định nghĩa màu sắc cho từng loại
        colors = {
            "BTC": self.colors['primary'],
            "ETH": self.colors['secondary'],
            "STABLES": self.colors['success'],
            "OTHERS": self.colors['tertiary'],
            "CASH": self.colors['warning']
        }
        
        # Tạo color array
        color_values = [colors.get(label, self.colors['primary']) for label in labels]
        
        # Tạo biểu đồ pie chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            marker=dict(colors=color_values)
        )])
        
        # Định dạng biểu đồ
        fig.update_layout(
            title=f'Phân bổ danh mục đầu tư - {portfolio_status.get("risk_profile", "").title()}',
            template='plotly_white' if self.theme != 'dark' else 'plotly_dark',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            annotations=[
                dict(
                    text=f"{portfolio_status.get('portfolio_value', 0):.2f}",
                    x=0.5, y=0.5,
                    font=dict(size=20),
                    showarrow=False
                )
            ],
            height=500,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        # Thêm các thông tin chung
        fig.add_annotation(
            x=0.5, y=1.15,
            xref='paper', yref='paper',
            text=f"PnL: {portfolio_status.get('total_pnl', 0):.2f} "
                 f"({portfolio_status.get('realized_pnl', 0):.2f} thực hiện, "
                 f"{portfolio_status.get('unrealized_pnl', 0):.2f} chưa thực hiện)",
            showarrow=False,
            font=dict(size=12)
        )
        
        return fig
    
    @measure_execution_time()
    def plot_position_summary(self, position_tracker: PositionTracker) -> go.Figure:
        """
        Tạo biểu đồ tóm tắt vị thế.
        
        Args:
            position_tracker: Đối tượng PositionTracker đã được khởi tạo
            
        Returns:
            Figure plotly
        """
        # Lấy tóm tắt vị thế
        position_summaries = position_tracker.get_position_summary()
        
        # Nếu không có vị thế nào
        if not position_summaries:
            return go.Figure().update_layout(
                title="Không có vị thế nào hiện tại",
                annotations=[dict(
                    text="Không có dữ liệu vị thế",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )],
                height=500
            )
        
        # Nếu là một vị thế duy nhất, chuyển thành list
        if not isinstance(position_summaries, list):
            position_summaries = [position_summaries]
        
        # Tạo DataFrame từ summaries
        df = pd.DataFrame(position_summaries)
        
        # Chỉ hiển thị các vị thế có size > 0
        df = df[df['size'] > 0]
        
        if df.empty:
            return go.Figure().update_layout(
                title="Không có vị thế nào hiện tại",
                annotations=[dict(
                    text="Không có dữ liệu vị thế",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )],
                height=500
            )
        
        # Tính notional value và unrealized P&L nếu chưa có
        if 'notional_value' not in df.columns:
            df['notional_value'] = df['size'] * df['current_price']
        
        if 'unrealized_pnl_percent' not in df.columns and 'unrealized_pnl' in df.columns:
            df['unrealized_pnl_percent'] = df.apply(
                lambda row: row['unrealized_pnl'] / (row['entry_price'] * row['size']) * 100 
                if row['entry_price'] > 0 and row['size'] > 0 else 0, 
                axis=1
            )
        
        # Tạo biểu đồ
        fig = go.Figure()
        
        # Thêm bar chart cho vị thế
        for i, row in df.iterrows():
            color = self.colors['primary'] if row['side'] == 'long' else self.colors['danger']
            if 'unrealized_pnl' in row and row['unrealized_pnl'] > 0:
                color = self.colors['success'] if row['side'] == 'long' else self.colors['warning']
            
            fig.add_trace(go.Bar(
                x=[row['symbol']],
                y=[row['notional_value']],
                name=row['symbol'],
                marker_color=color,
                text=f"{row['unrealized_pnl_percent']:.2f}%" if 'unrealized_pnl_percent' in row else "",
                hovertext=(
                    f"Symbol: {row['symbol']}<br>"
                    f"Side: {row['side']}<br>"
                    f"Size: {row['size']}<br>"
                    f"Entry: {row['entry_price']}<br>"
                    f"Current: {row['current_price']}<br>"
                    f"PnL: {row.get('unrealized_pnl', 0):.2f} ({row.get('unrealized_pnl_percent', 0):.2f}%)<br>"
                    f"Leverage: {row.get('leverage', 1)}x<br>"
                ),
                hoverinfo="text"
            ))
        
        # Định dạng biểu đồ
        fig.update_layout(
            title=f'Tóm tắt vị thế - {len(df)} vị thế đang mở',
            xaxis_title='Symbol',
            yaxis_title='Notional Value',
            template='plotly_white' if self.theme != 'dark' else 'plotly_dark',
            barmode='group',
            height=500,
            annotations=[
                dict(
                    x=row['symbol'],
                    y=row['notional_value'],
                    text=f"{row.get('unrealized_pnl_percent', 0):.1f}%",
                    showarrow=False,
                    yshift=10,
                    font=dict(
                        size=10,
                        color='green' if row.get('unrealized_pnl', 0) > 0 else 'red'
                    )
                ) for i, row in df.iterrows()
            ]
        )
        
        # Thêm số liệu tổng hợp
        stats = position_tracker.update_position_statistics()
        
        fig.add_annotation(
            x=0.5, y=1.15,
            xref='paper', yref='paper',
            text=(
                f"Tổng giá trị: {stats.get('total_position_value', 0):.2f} | "
                f"PnL chưa thực hiện: {stats.get('total_unrealized_pnl', 0):.2f} | "
                f"Vị thế lời/lỗ: {stats.get('profitable_positions', 0)}/{stats.get('losing_positions', 0)}"
            ),
            showarrow=False,
            font=dict(size=12)
        )
        
        return fig
    
    @measure_execution_time()
    def plot_correlation_matrix(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """
        Tạo biểu đồ ma trận tương quan.
        
        Args:
            correlation_matrix: DataFrame chứa ma trận tương quan
            
        Returns:
            Figure plotly
        """
        # Kiểm tra ma trận tương quan
        if correlation_matrix.empty:
            return go.Figure().update_layout(
                title="Không có dữ liệu tương quan",
                annotations=[dict(
                    text="Không có dữ liệu ma trận tương quan",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )],
                height=500
            )
        
        # Tạo biểu đồ heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu_r',  # Red-Blue scale, reversed
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title='Correlation',
                titleside='right'
            )
        ))
        
        # Thêm chữ hiển thị giá trị
        annotations = []
        
        for i, row in enumerate(correlation_matrix.index):
            for j, col in enumerate(correlation_matrix.columns):
                value = correlation_matrix.iloc[i, j]
                # Chỉ hiển thị chữ cho các ô có giá trị > 0.5 hoặc < -0.5
                if abs(value) > 0.5 or i == j:
                    annotations.append(dict(
                        x=col,
                        y=row,
                        text=f"{value:.2f}",
                        font=dict(
                            color='white' if abs(value) > 0.7 else 'black',
                            size=10
                        ),
                        showarrow=False
                    ))
        
        # Định dạng biểu đồ
        fig.update_layout(
            title='Ma trận tương quan giữa các tài sản',
            template='plotly_white' if self.theme != 'dark' else 'plotly_dark',
            height=600,
            width=600,
            annotations=annotations
        )
        
        return fig
    
    def fig_to_base64(self, fig: Union[plt.Figure, go.Figure]) -> str:
        """
        Chuyển đổi matplotlib hoặc plotly figure thành chuỗi base64 để hiển thị trong Streamlit.
        
        Args:
            fig: Matplotlib hoặc Plotly figure
            
        Returns:
            Chuỗi base64 của hình ảnh PNG
        """
        if isinstance(fig, plt.Figure):
            # Matplotlib figure
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            return base64.b64encode(buf.read()).decode('utf-8')
        else:
            # Plotly figure
            return fig.to_image(format="png", engine="kaleido").decode('utf-8')


class RiskDashboard:
    """
    Lớp tạo dashboard rủi ro trong Streamlit.
    Cung cấp các phương thức để hiển thị các biểu đồ và thông tin 
    về rủi ro trong hệ thống giao dịch.
    """
    
    def __init__(self, visualizer: Optional[RiskVisualization] = None):
        """
        Khởi tạo dashboard rủi ro.
        
        Args:
            visualizer: Đối tượng RiskVisualization (tạo mới nếu None)
        """
        # Khởi tạo visualizer nếu chưa được cung cấp
        self.visualizer = visualizer or RiskVisualization(
            theme="dark" if st.get_option("theme.base") == "dark" else "light"
        )
        self.logger = get_logger("risk_dashboard")
    
    def display_risk_overview(self, risk_evaluator: RiskEvaluator) -> None:
        """
        Hiển thị tổng quan về rủi ro.
        
        Args:
            risk_evaluator: Đối tượng RiskEvaluator đã được khởi tạo
        """
        # Lấy đánh giá tổng quát
        if risk_evaluator.results:
            risk_assessment = risk_evaluator.results.get("risk_assessment", {})
        else:
            risk_assessment = risk_evaluator.get_overall_risk_assessment()
        
        # Tạo layout 2 cột
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Hiển thị điểm số rủi ro tổng quát
            overall_risk = risk_assessment.get("overall_risk_level", "N/A")
            overall_score = risk_assessment.get("overall_risk_score", 0)
            
            st.metric(
                label="Mức độ rủi ro tổng thể",
                value=overall_risk,
                delta=f"{overall_score:.2f}/5.0",
                delta_color="inverse"
            )
            
            # Hiển thị các điểm số theo loại rủi ro
            risk_scores = risk_assessment.get("risk_scores", {})
            
            for risk_type, score_data in risk_scores.items():
                risk_level = score_data.get("level", "N/A")
                risk_score = score_data.get("score", 0)
                
                st.metric(
                    label=f"Rủi ro {risk_type.replace('_', ' ').title()}",
                    value=risk_level,
                    delta=f"{risk_score}/5",
                    delta_color="inverse"
                )
            
            # Hiển thị khuyến nghị chung
            st.subheader("Khuyến nghị")
            st.write(risk_assessment.get("general_recommendation", "Không có khuyến nghị"))
        
        with col2:
            # Hiển thị radar chart
            radar_chart = self.visualizer.plot_risk_radar(
                risk_scores=risk_scores,
                risk_weights=risk_assessment.get("risk_weights", None)
            )
            
            st.plotly_chart(radar_chart, use_container_width=True)
            
            # Hiển thị các khuyến nghị cụ thể
            if "risk_recommendations" in risk_assessment:
                with st.expander("Xem các khuyến nghị chi tiết"):
                    for rec in risk_assessment["risk_recommendations"]:
                        st.markdown(f"- {rec}")
    
    def display_drawdown_analysis(self, risk_evaluator: RiskEvaluator) -> None:
        """
        Hiển thị phân tích drawdown.
        
        Args:
            risk_evaluator: Đối tượng RiskEvaluator đã được khởi tạo
        """
        # Lấy kết quả drawdown
        if risk_evaluator.results:
            drawdown_analysis = risk_evaluator.results.get("drawdown_analysis", {})
        else:
            # Thực hiện phân tích nếu chưa có
            drawdown_analysis = risk_evaluator.analyze_drawdown()
        
        # Tạo các cột hiển thị chỉ số
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Max Drawdown",
                value=f"{drawdown_analysis.get('max_drawdown_percent', 0):.2f}%",
                help="Mức sụt giảm vốn lớn nhất từ đỉnh cao nhất"
            )
        
        with col2:
            st.metric(
                label="Thời gian Max DD",
                value=f"{drawdown_analysis.get('max_drawdown_duration', 0)} ngày",
                help="Thời gian sụt giảm kéo dài của mức Drawdown lớn nhất"
            )
        
        with col3:
            st.metric(
                label="Thời gian hồi phục TB",
                value=f"{drawdown_analysis.get('avg_recovery_time', 0):.1f} ngày",
                help="Thời gian trung bình để hồi phục hoàn toàn từ Drawdown"
            )
        
        with col4:
            st.metric(
                label="Rủi ro Drawdown",
                value=drawdown_analysis.get('drawdown_risk', 'N/A'),
                help="Đánh giá mức độ rủi ro từ Drawdown"
            )
        
        # Hiển thị biểu đồ drawdown
        st.subheader("Biểu đồ Drawdown")
        
        chart_type = st.radio(
            "Loại biểu đồ Drawdown",
            options=["Interactive", "Underwater"],
            horizontal=True,
            key="drawdown_chart_type"
        )
        
        if chart_type == "Interactive":
            try:
                # Tạo biểu đồ interative với Altair
                drawdown_chart = self.visualizer.plot_drawdown_chart(
                    equity_curve=risk_evaluator.equity_curve,
                    max_drawdown_threshold=risk_evaluator.max_drawdown_threshold
                )
                st.altair_chart(drawdown_chart, use_container_width=True)
            except Exception as e:
                self.logger.error(f"Lỗi khi tạo biểu đồ Drawdown interactive: {str(e)}")
                # Fallback to underwater chart
                fig = self.visualizer.plot_drawdown_underwater(
                    equity_curve=risk_evaluator.equity_curve
                )
                st.pyplot(fig)
        else:
            # Tạo biểu đồ underwater với matplotlib
            fig = self.visualizer.plot_drawdown_underwater(
                equity_curve=risk_evaluator.equity_curve
            )
            st.pyplot(fig)
        
        # Hiển thị thông tin chi tiết
        with st.expander("Thông tin chi tiết về Drawdown"):
            # Tạo hai cột
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Thống kê Drawdown")
                st.write(f"Số lượng đợt Drawdown: {drawdown_analysis.get('drawdown_periods_count', 0)}")
                st.write(f"Số lượng đợt phục hồi: {drawdown_analysis.get('recovery_periods_count', 0)}")
                st.write(f"Thời gian hồi phục lớn nhất: {drawdown_analysis.get('max_recovery_time', 0)} ngày")
                st.write(f"Thời gian hồi phục nhỏ nhất: {drawdown_analysis.get('min_recovery_time', 0)} ngày")
                st.write(f"Tỷ lệ thời gian trong Drawdown: {drawdown_analysis.get('time_in_drawdown', 0):.2%}")
                st.write(f"Tỷ lệ thời gian trong Drawdown nghiêm trọng: {drawdown_analysis.get('time_in_severe_drawdown', 0):.2%}")
            
            with col2:
                st.subheader("Tần suất Drawdown")
                df_freq = pd.DataFrame({
                    'Mức độ': [f"{k}+" for k in drawdown_analysis.get('drawdown_frequency', {}).keys()],
                    'Tần suất': list(drawdown_analysis.get('drawdown_frequency', {}).values())
                })
                
                if not df_freq.empty:
                    # Tạo bar chart với Altair
                    freq_chart = alt.Chart(df_freq).mark_bar().encode(
                        x='Mức độ',
                        y=alt.Y('Tần suất', title='Tần suất (%)', axis=alt.Axis(format='%')),
                        color=alt.Color('Mức độ', legend=None),
                        tooltip=['Mức độ', alt.Tooltip('Tần suất', format='.2%')]
                    ).properties(
                        title='Tần suất các mức Drawdown'
                    )
                    
                    st.altair_chart(freq_chart, use_container_width=True)
            
            # Hiển thị top drawdown periods
            st.subheader("Top Drawdown Periods")
            if 'top_drawdown_periods' in drawdown_analysis:
                # Tạo DataFrame từ các kỳ drawdown
                df_drawdown = pd.DataFrame(drawdown_analysis['top_drawdown_periods'])
                if not df_drawdown.empty:
                    # Format dates
                    if 'start_date' in df_drawdown.columns:
                        df_drawdown['start_date'] = pd.to_datetime(df_drawdown['start_date'])
                    if 'end_date' in df_drawdown.columns:
                        df_drawdown['end_date'] = pd.to_datetime(df_drawdown['end_date'])
                    
                    # Format drawdown amounts
                    if 'drawdown_amount' in df_drawdown.columns:
                        df_drawdown['drawdown_percent'] = df_drawdown['drawdown_amount'] * 100
                    
                    # Hiển thị bảng
                    st.dataframe(df_drawdown)
    
    def display_volatility_analysis(self, risk_evaluator: RiskEvaluator) -> None:
        """
        Hiển thị phân tích biến động.
        
        Args:
            risk_evaluator: Đối tượng RiskEvaluator đã được khởi tạo
        """
        # Lấy kết quả volatility
        if risk_evaluator.results:
            volatility_analysis = risk_evaluator.results.get("volatility_analysis", {})
        else:
            # Thực hiện phân tích nếu chưa có
            volatility_analysis = risk_evaluator.analyze_volatility()
        
        # Tạo các cột hiển thị chỉ số
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Biến động hàng năm",
                value=f"{volatility_analysis.get('annualized_volatility', 0) * 100:.2f}%",
                help="Mức biến động giá hàng năm"
            )
        
        with col2:
            st.metric(
                label="Biến động xuống",
                value=f"{volatility_analysis.get('annualized_downside_volatility', 0) * 100:.2f}%",
                help="Mức biến động chỉ tính các giai đoạn giảm giá"
            )
        
        with col3:
            st.metric(
                label="Tỷ lệ biến động",
                value=f"{volatility_analysis.get('volatility_ratio', 0):.2f}",
                help="Tỷ lệ giữa biến động tăng và biến động giảm"
            )
        
        with col4:
            st.metric(
                label="Rủi ro biến động",
                value=volatility_analysis.get('volatility_risk', 'N/A'),
                help="Đánh giá mức độ rủi ro từ biến động"
            )
        
        # Hiển thị biểu đồ volatility
        st.subheader("Biểu đồ biến động theo thời gian")
        fig = self.visualizer.plot_volatility_chart(
            returns=risk_evaluator.returns,
            rolling_windows=[20, 60, 120]  # 1 month, 3 months, 6 months
        )
        st.pyplot(fig)
        
        # Hiển thị thông tin chi tiết
        with st.expander("Thông tin chi tiết về biến động"):
            # Tạo hai cột
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Thống kê biến động")
                st.write(f"Biến động hàng ngày: {volatility_analysis.get('daily_volatility', 0):.6f}")
                st.write(f"Biến động giảm hàng ngày: {volatility_analysis.get('daily_downside_volatility', 0):.6f}")
                st.write(f"Biến động của biến động: {volatility_analysis.get('vol_of_vol', 0):.4f}")
                st.write(f"Biến động tụ (Clustering): {volatility_analysis.get('volatility_clustering', 0):.4f}")
            
            with col2:
                st.subheader("Biến động theo chu kỳ")
                # Tạo DataFrame từ period_volatility
                period_vol = volatility_analysis.get('period_volatility', {})
                if period_vol:
                    periods = []
                    means = []
                    maxs = []
                    mins = []
                    currents = []
                    
                    for period, values in period_vol.items():
                        periods.append(period)
                        means.append(values.get('mean', 0) * 100)
                        maxs.append(values.get('max', 0) * 100)
                        mins.append(values.get('min', 0) * 100)
                        currents.append(values.get('current', 0) * 100)
                    
                    df_vol = pd.DataFrame({
                        'Chu kỳ': periods,
                        'Trung bình': means,
                        'Lớn nhất': maxs,
                        'Nhỏ nhất': mins,
                        'Hiện tại': currents
                    })
                    
                    # Hiển thị bảng
                    st.dataframe(df_vol)
            
            # So sánh với benchmark nếu có
            benchmark_comparison = volatility_analysis.get('benchmark_comparison', {})
            if benchmark_comparison:
                st.subheader("So sánh với benchmark")
                st.write(f"Biến động benchmark: {benchmark_comparison.get('benchmark_volatility', 0) * 100:.2f}%")
                st.write(f"Biến động tương đối: {benchmark_comparison.get('relative_volatility', 0):.2f}")
                st.write(f"Ít biến động hơn benchmark: {'Có' if benchmark_comparison.get('is_less_volatile', False) else 'Không'}")
    
    def display_var_analysis(self, risk_evaluator: RiskEvaluator) -> None:
        """
        Hiển thị phân tích Value at Risk (VaR).
        
        Args:
            risk_evaluator: Đối tượng RiskEvaluator đã được khởi tạo
        """
        # Lấy kết quả VaR
        if risk_evaluator.results:
            var_analysis = risk_evaluator.results.get("var_analysis", {})
        else:
            # Thực hiện phân tích nếu chưa có
            var_analysis = risk_evaluator.analyze_value_at_risk()
        
        # Tạo các cột hiển thị chỉ số
        col1, col2, col3, col4 = st.columns(4)
        
        # Lấy thông tin VaR
        var_values = var_analysis.get("var", {})
        
        with col1:
            st.metric(
                label="Historical VaR",
                value=f"{var_values.get('historical', 0) * 100:.2f}%",
                help="Value at Risk tính theo phương pháp lịch sử"
            )
        
        with col2:
            st.metric(
                label="Parametric VaR",
                value=f"{var_values.get('parametric', 0) * 100:.2f}%",
                help="Value at Risk tính theo phương pháp tham số giả định phân phối chuẩn"
            )
        
        with col3:
            st.metric(
                label="Expected Shortfall",
                value=f"{var_analysis.get('es', 0) * 100:.2f}%",
                help="Conditional VaR - Mức tổn thất trung bình khi vượt quá VaR"
            )
        
        with col4:
            st.metric(
                label="Mức độ tin cậy",
                value=f"{var_analysis.get('confidence_level', 0.95) * 100:.0f}%",
                help="Mức độ tin cậy sử dụng để tính VaR"
            )
        
        # Hiển thị biểu đồ VaR
        st.subheader("Biểu đồ Value at Risk")
        var_chart = self.visualizer.plot_var_analysis(
            returns=risk_evaluator.returns,
            confidence_levels=[0.95, 0.99],
            use_plotly=True
        )
        st.plotly_chart(var_chart, use_container_width=True)
        
        # Hiển thị thông tin chi tiết
        with st.expander("Thông tin chi tiết về Value at Risk"):
            # Tạo hai cột
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("VaR theo chu kỳ")
                # Tạo DataFrame từ var_by_period
                var_by_period = var_analysis.get('var_by_period', {})
                if var_by_period:
                    df_var = pd.DataFrame({
                        'Chu kỳ': list(var_by_period.keys()),
                        'VaR': [v * 100 for v in var_by_period.values()]
                    })
                    
                    # Tạo bar chart với Altair
                    var_chart = alt.Chart(df_var).mark_bar().encode(
                        x='Chu kỳ',
                        y=alt.Y('VaR', title='VaR (%)'),
                        color=alt.Color('Chu kỳ', legend=None),
                        tooltip=['Chu kỳ', alt.Tooltip('VaR', format='.2f')]
                    ).properties(
                        title='VaR theo chu kỳ'
                    )
                    
                    st.altair_chart(var_chart, use_container_width=True)
            
            with col2:
                st.subheader("Expected Shortfall theo chu kỳ")
                # Tạo DataFrame từ es_by_period
                es_by_period = var_analysis.get('es_by_period', {})
                if es_by_period:
                    df_es = pd.DataFrame({
                        'Chu kỳ': list(es_by_period.keys()),
                        'ES': [v * 100 for v in es_by_period.values()]
                    })
                    
                    # Tạo bar chart với Altair
                    es_chart = alt.Chart(df_es).mark_bar().encode(
                        x='Chu kỳ',
                        y=alt.Y('ES', title='Expected Shortfall (%)'),
                        color=alt.Color('Chu kỳ', legend=None),
                        tooltip=['Chu kỳ', alt.Tooltip('ES', format='.2f')]
                    ).properties(
                        title='Expected Shortfall theo chu kỳ'
                    )
                    
                    st.altair_chart(es_chart, use_container_width=True)
            
            # VaR contribution
            var_contribution = var_analysis.get('var_contribution', {})
            if var_contribution:
                st.subheader("VaR Contribution")
                df_contrib = pd.DataFrame({
                    'Symbol': list(var_contribution.keys()),
                    'VaR Contribution': [v * 100 for v in var_contribution.values()]
                })
                
                # Sắp xếp theo đóng góp giảm dần
                df_contrib = df_contrib.sort_values('VaR Contribution', ascending=False)
                
                # Tạo bar chart với Altair
                contrib_chart = alt.Chart(df_contrib).mark_bar().encode(
                    x=alt.X('Symbol', sort='-y'),
                    y=alt.Y('VaR Contribution', title='VaR Contribution (%)'),
                    color=alt.Color('Symbol', legend=None),
                    tooltip=['Symbol', alt.Tooltip('VaR Contribution', format='.2f')]
                ).properties(
                    title='VaR Contribution theo Symbol'
                )
                
                st.altair_chart(contrib_chart, use_container_width=True)
    
    def display_distribution_analysis(self, risk_evaluator: RiskEvaluator) -> None:
        """
        Hiển thị phân tích phân phối lợi nhuận.
        
        Args:
            risk_evaluator: Đối tượng RiskEvaluator đã được khởi tạo
        """
        # Lấy kết quả phân tích phân phối
        if risk_evaluator.results:
            distribution_analysis = risk_evaluator.results.get("distribution_analysis", {})
        else:
            # Thực hiện phân tích nếu chưa có
            distribution_analysis = risk_evaluator.analyze_returns_distribution()
        
        # Tạo các cột hiển thị chỉ số
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Skewness",
                value=f"{distribution_analysis.get('skewness', 0):.4f}",
                delta="+Tốt" if distribution_analysis.get('skewness', 0) > 0 else "-Xấu",
                help="Độ lệch phân phối, dương là tốt (đuôi phải dài hơn)"
            )
        
        with col2:
            st.metric(
                label="Kurtosis",
                value=f"{distribution_analysis.get('kurtosis', 0):.4f}",
                delta="+Cao" if distribution_analysis.get('kurtosis', 0) > 0 else "-Thấp",
                help="Độ nhọn phân phối, cao nghĩa là có nhiều outlier"
            )
        
        with col3:
            st.metric(
                label="Tỷ lệ lợi nhuận dương",
                value=f"{distribution_analysis.get('positive_returns_ratio', 0) * 100:.2f}%",
                help="Tỷ lệ ngày có lợi nhuận dương"
            )
        
        with col4:
            st.metric(
                label="Chất lượng phân phối",
                value=distribution_analysis.get('distribution_quality', 'N/A'),
                help="Đánh giá tổng thể chất lượng phân phối lợi nhuận"
            )
        
        # Hiển thị biểu đồ phân phối
        st.subheader("Phân phối lợi nhuận")
        distribution_chart = self.visualizer.plot_returns_distribution(
            returns=risk_evaluator.returns
        )
        st.plotly_chart(distribution_chart, use_container_width=True)
        
        # Hiển thị thông tin chi tiết
        with st.expander("Thông tin chi tiết về phân phối"):
            # Tạo hai cột
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Thống kê phân phối")
                # Hiển thị thông tin kiểm định Jarque-Bera
                jarque_bera = distribution_analysis.get("jarque_bera", {})
                if jarque_bera:
                    st.write(f"Kiểm định Jarque-Bera: {jarque_bera.get('statistic', 0):.4f}")
                    st.write(f"p-value: {jarque_bera.get('p_value', 0):.6f}")
                    st.write(f"Phân phối chuẩn: {'Có' if jarque_bera.get('is_normal', False) else 'Không'}")
                
                # Hiển thị các thống kê mô tả
                stats = distribution_analysis.get('statistics', {})
                if stats:
                    st.write(f"Trung bình: {stats.get('mean', 0):.6f}")
                    st.write(f"Độ lệch chuẩn: {stats.get('std', 0):.6f}")
                    st.write(f"Min: {stats.get('min', 0):.6f}")
                    st.write(f"Max: {stats.get('max', 0):.6f}")
            
            with col2:
                st.subheader("Phân vị")
                # Tạo DataFrame từ percentiles
                percentiles = distribution_analysis.get('percentiles', {})
                if percentiles:
                    df_percentiles = pd.DataFrame({
                        'Phân vị': [f"p{k.replace('p', '')}" for k in percentiles.keys()],
                        'Giá trị': list(percentiles.values())
                    })
                    
                    # Hiển thị bảng
                    st.dataframe(df_percentiles)
                
                # So sánh với phân phối chuẩn
                normal_dist_comparison = distribution_analysis.get('normal_dist_comparison', {})
                if normal_dist_comparison:
                    st.subheader("So sánh với phân phối chuẩn")
                    
                    for percentile, values in normal_dist_comparison.items():
                        st.write(f"{percentile}: Thực tế: {values.get('actual', 0):.6f}, " + 
                               f"Chuẩn: {values.get('normal', 0):.6f}")
    
    def display_risk_return_metrics(self, risk_evaluator: RiskEvaluator) -> None:
        """
        Hiển thị các chỉ số rủi ro-lợi nhuận.
        
        Args:
            risk_evaluator: Đối tượng RiskEvaluator đã được khởi tạo
        """
        # Lấy kết quả risk-reward
        risk_return = risk_evaluator.calculate_risk_return_metrics()
        
        # Tạo 2 rows với 4 columns mỗi row
        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
        
        with row1_col1:
            st.metric(
                label="Lợi nhuận hàng năm",
                value=f"{risk_return.get('annualized_return', 0) * 100:.2f}%",
                help="Lợi nhuận quy đổi sang tỷ lệ hàng năm"
            )
        
        with row1_col2:
            st.metric(
                label="Biến động hàng năm",
                value=f"{risk_return.get('annualized_volatility', 0) * 100:.2f}%",
                help="Biến động quy đổi sang tỷ lệ hàng năm"
            )
        
        with row1_col3:
            st.metric(
                label="Sharpe Ratio",
                value=f"{risk_return.get('sharpe_ratio', 0):.2f}",
                help="Lợi nhuận vượt trội trên mỗi đơn vị rủi ro"
            )
        
        with row1_col4:
            st.metric(
                label="Sortino Ratio",
                value=f"{risk_return.get('sortino_ratio', 0):.2f}",
                help="Lợi nhuận vượt trội trên mỗi đơn vị rủi ro giảm"
            )
        
        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
        
        with row2_col1:
            st.metric(
                label="Calmar Ratio",
                value=f"{risk_return.get('calmar_ratio', 0):.2f}",
                help="Lợi nhuận hàng năm / Max Drawdown"
            )
        
        with row2_col2:
            st.metric(
                label="Omega Ratio",
                value=f"{risk_return.get('omega_ratio', 0):.2f}",
                help="Tỷ lệ giữa lợi nhuận tiềm năng và rủi ro tiềm ẩn"
            )
        
        with row2_col3:
            st.metric(
                label="Gain to Pain Ratio",
                value=f"{risk_return.get('gain_to_pain_ratio', 0):.2f}",
                help="Tổng lợi nhuận / Tổng thua lỗ"
            )
        
        with row2_col4:
            st.metric(
                label="Đánh giá hiệu suất",
                value=risk_return.get('performance_rating', 'N/A'),
                help="Đánh giá tổng thể hiệu suất dựa trên các chỉ số"
            )
        
        # Hiển thị biểu đồ Risk-Return
        st.subheader("Biểu đồ Risk-Return")
        fig = self.visualizer.plot_risk_reward_profile(
            returns=risk_evaluator.returns,
            risk_free_rate=risk_evaluator.risk_free_rate,
            benchmark_returns=None  # Thêm benchmark nếu có
        )
        st.pyplot(fig)
        
        # Hiển thị thông tin Alpha/Beta nếu có
        if risk_return.get('alpha') is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Thông tin Alpha/Beta")
                st.write(f"Alpha: {risk_return.get('alpha', 0):.4f}")
                st.write(f"Beta: {risk_return.get('beta', 0):.2f}")
                st.write(f"Information Ratio: {risk_return.get('information_ratio', 0):.2f}")
                st.write(f"Treynor Ratio: {risk_return.get('treynor_ratio', 0):.2f}")
    
    def display_stress_test(self, risk_evaluator: RiskEvaluator) -> None:
        """
        Hiển thị kết quả stress test.
        
        Args:
            risk_evaluator: Đối tượng RiskEvaluator đã được khởi tạo
        """
        # Tạo dropdown cho lựa chọn kịch bản
        scenarios = {
            "2008_crisis": "Khủng hoảng 2008",
            "2020_covid": "COVID-19 (2020)",
            "2022_crypto_winter": "Crypto Winter 2022",
            "average_correction": "Điều chỉnh trung bình",
            "severe_correction": "Điều chỉnh nghiêm trọng"
        }
        
        selected_scenario = st.selectbox(
            "Chọn kịch bản stress test",
            options=list(scenarios.keys()),
            format_func=lambda x: scenarios.get(x, x)
        )
        
        # Thực hiện stress test
        stress_results = risk_evaluator.calculate_stress_test(scenario_name=selected_scenario)
        
        # Hiển thị thông tin kịch bản
        scenario = stress_results["scenario"]
        results = stress_results["results"]
        impact = stress_results["impact"]
        
        # Tạo 4 cột hiển thị thông tin
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Mức độ stress",
                value=f"{scenario['stress_percent']*100:.0f}%",
                help="Mức độ sụt giảm tối đa trong kịch bản"
            )
        
        with col2:
            st.metric(
                label="Thời gian stress",
                value=f"{scenario['duration_days']} ngày",
                help="Thời gian kéo dài của giai đoạn stress"
            )
        
        with col3:
            st.metric(
                label="Max Drawdown",
                value=f"{results['max_drawdown_percent']*100:.2f}%",
                help="Mức Drawdown tối đa trong kịch bản stress test"
            )
        
        with col4:
            st.metric(
                label="Mức độ tác động",
                value=impact,
                help="Đánh giá mức độ tác động của kịch bản stress"
            )
        
        # Hiển thị biểu đồ stress test
        st.subheader("Biểu đồ Stress Test")
        stress_chart = self.visualizer.plot_stress_test(
            risk_evaluator=risk_evaluator,
            scenario_name=selected_scenario
        )
        st.plotly_chart(stress_chart, use_container_width=True)
        
        # Hiển thị thông tin chi tiết
        with st.expander("Thông tin chi tiết về stress test"):
            # Tạo bảng thông tin
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Thông tin kịch bản")
                st.write(f"Tên kịch bản: {selected_scenario}")
                st.write(f"Mức độ stress: {scenario['stress_percent']*100:.1f}%")
                st.write(f"Thời gian stress: {scenario['duration_days']} ngày")
                st.write(f"Tốc độ phục hồi: {scenario['recovery_rate']*100:.2f}% mỗi ngày")
            
            with col2:
                st.subheader("Kết quả")
                st.write(f"Giá trị nhỏ nhất: {results['min_equity']:.2f}")
                st.write(f"Max Drawdown: {results['max_drawdown_amount']:.2f} ({results['max_drawdown_percent']*100:.2f}%)")
                st.write(f"Vốn sau stress: {results['capital_after_stress']:.2f}")
                st.write(f"P&L sau stress: {results['profit_loss_after_stress']:.2f} ({results['profit_loss_percent']*100:.2f}%)")
                
                if results['full_recovery_days'] is not None:
                    st.write(f"Thời gian phục hồi: {results['full_recovery_days']} ngày")
                else:
                    st.write("Phục hồi: Không phục hồi hoàn toàn trong kịch bản")
    
    def display_monte_carlo_simulation(self, risk_evaluator: RiskEvaluator) -> None:
        """
        Hiển thị kết quả mô phỏng Monte Carlo.
        
        Args:
            risk_evaluator: Đối tượng RiskEvaluator đã được khởi tạo
        """
        # Tạo các thông số cho mô phỏng
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_simulations = st.number_input(
                "Số lần mô phỏng",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
        
        with col2:
            days_forward = st.number_input(
                "Số ngày mô phỏng",
                min_value=30,
                max_value=1000,
                value=252,  # 1 năm giao dịch
                step=30
            )
        
        with col3:
            confidence_level = st.slider(
                "Mức độ tin cậy",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                format="%.2f"
            )
        
        # Thực hiện mô phỏng
        if st.button("Chạy mô phỏng Monte Carlo"):
            with st.spinner("Đang chạy mô phỏng Monte Carlo..."):
                simulation = risk_evaluator.run_monte_carlo_simulation(
                    num_simulations=num_simulations,
                    days_forward=days_forward,
                    confidence_level=confidence_level
                )
                
                # Lưu kết quả vào session state để tái sử dụng
                st.session_state.monte_carlo_simulation = simulation
        
        # Lấy kết quả mô phỏng từ session state hoặc chạy lại nếu chưa có
        if 'monte_carlo_simulation' not in st.session_state:
            with st.spinner("Đang chạy mô phỏng Monte Carlo ban đầu..."):
                st.session_state.monte_carlo_simulation = risk_evaluator.run_monte_carlo_simulation(
                    num_simulations=num_simulations,
                    days_forward=days_forward,
                    confidence_level=confidence_level
                )
        
        simulation = st.session_state.monte_carlo_simulation
        
        # Lấy thông tin từ kết quả mô phỏng
        final_values = simulation["final_values"]
        risk_metrics = simulation["risk_metrics"]
        
        # Hiển thị thông tin
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="VaR",
                value=f"{risk_metrics['var_percent']*100:.2f}%",
                help=f"Value at Risk với mức tin cậy {confidence_level*100:.0f}%"
            )
        
        with col2:
            st.metric(
                label="Drawdown kỳ vọng",
                value=f"{risk_metrics['expected_max_drawdown']*100:.2f}%",
                help="Mức Drawdown tối đa kỳ vọng trong mô phỏng"
            )
        
        with col3:
            st.metric(
                label="Xác suất lợi nhuận",
                value=f"{risk_metrics['profit_probability']*100:.2f}%",
                help="Xác suất có lợi nhuận tại cuối kỳ mô phỏng"
            )
        
        with col4:
            st.metric(
                label="Xác suất DD nghiêm trọng",
                value=f"{risk_metrics['severe_drawdown_probability']*100:.2f}%",
                help="Xác suất gặp phải Drawdown nghiêm trọng trong mô phỏng"
            )
        
        # Hiển thị biểu đồ mô phỏng
        tab1, tab2 = st.tabs(["Biểu đồ mô phỏng", "Phân phối giá trị cuối"])
        
        with tab1:
            simulation_chart = self.visualizer.plot_monte_carlo_simulation(
                risk_evaluator=risk_evaluator,
                num_simulations=num_simulations,
                days_forward=days_forward,
                confidence_level=confidence_level
            )
            st.plotly_chart(simulation_chart, use_container_width=True)
        
        with tab2:
            # Hiển thị histogram của final values
            if 'distribution_chart' in simulation:
                st.image(f"data:image/png;base64,{simulation['distribution_chart']}")
            else:
                # Tạo histogram với plotly
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=final_values['mean'],
                    nbinsx=50,
                    name='Final Values',
                    marker_color=self.visualizer.colors['primary']
                ))
                
                # Thêm đường dọc cho giá trị hiện tại
                current_equity = risk_evaluator.equity_curve.iloc[-1]
                fig.add_vline(
                    x=current_equity, 
                    line_width=2, 
                    line_dash="dash", 
                    line_color="green",
                    annotation_text="Giá trị hiện tại"
                )
                
                # Thêm đường dọc cho các phân vị
                for p, color in [(5, "red"), (95, "green")]:
                    percentile_value = final_values['percentiles'][f"p{p}"]
                    fig.add_vline(
                        x=percentile_value, 
                        line_width=1.5, 
                        line_dash="dash", 
                        line_color=color,
                        annotation_text=f"Phân vị {p}%"
                    )
                
                fig.update_layout(
                    title='Phân phối giá trị cuối sau mô phỏng',
                    xaxis_title='Giá trị vốn',
                    yaxis_title='Tần suất',
                    template='plotly_white' if self.visualizer.theme != 'dark' else 'plotly_dark',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị thông tin chi tiết
        with st.expander("Thông tin chi tiết về mô phỏng"):
            # Tạo bảng thông tin về final values
            st.subheader("Thống kê về giá trị cuối")
            df_stats = pd.DataFrame({
                'Chỉ số': ['Trung bình', 'Trung vị', 'Lớn nhất', 'Nhỏ nhất', 'Độ lệch chuẩn'],
                'Giá trị': [
                    final_values['mean'],
                    final_values['median'],
                    final_values['max'],
                    final_values['min'],
                    final_values['std']
                ]
            })
            st.dataframe(df_stats)
            
            # Hiển thị thông tin về percentiles
            st.subheader("Phân vị giá trị cuối")
            percentiles = final_values['percentiles']
            if percentiles:
                df_percentiles = pd.DataFrame({
                    'Phân vị': [f"{int(k.replace('p', ''))}%" for k in percentiles.keys()],
                    'Giá trị': list(percentiles.values())
                })
                st.dataframe(df_percentiles)
    
    def display_position_analysis(self, position_tracker: PositionTracker) -> None:
        """
        Hiển thị phân tích vị thế.
        
        Args:
            position_tracker: Đối tượng PositionTracker đã được khởi tạo
        """
        # Lấy thông tin vị thế
        positions = position_tracker.get_positions(force_update=True)
        
        # Nếu không có vị thế nào
        if not positions:
            st.warning("Không có vị thế nào đang mở")
            return
        
        # Hiển thị biểu đồ tóm tắt vị thế
        st.subheader("Tóm tắt vị thế")
        position_chart = self.visualizer.plot_position_summary(position_tracker)
        st.plotly_chart(position_chart, use_container_width=True)
        
        # Hiển thị thông tin chi tiết
        stats = position_tracker.update_position_statistics()
        
        # Tạo 2 rows với 4 columns mỗi row
        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
        
        with row1_col1:
            st.metric(
                label="Tổng vị thế",
                value=f"{stats.get('total_positions', 0)}",
                help="Tổng số vị thế đang mở"
            )
        
        with row1_col2:
            st.metric(
                label="Vị thế Long/Short",
                value=f"{stats.get('total_long_positions', 0)}/{stats.get('total_short_positions', 0)}",
                help="Số lượng vị thế Long/Short đang mở"
            )
        
        with row1_col3:
            st.metric(
                label="PnL chưa thực hiện",
                value=f"{stats.get('total_unrealized_pnl', 0):.2f}",
                delta=f"{float(stats.get('profitable_positions', 0) - stats.get('losing_positions', 0))}",
                help="Tổng lợi nhuận chưa thực hiện của các vị thế"
            )
        
        with row1_col4:
            st.metric(
                label="Đòn bẩy trung bình",
                value=f"{stats.get('average_leverage', 0):.2f}x",
                help="Đòn bẩy trung bình của các vị thế"
            )
        
        # Hiển thị thông tin chi tiết về vị thế
        st.subheader("Chi tiết vị thế")
        
        # Convert positions to DataFrame for display
        positions_df = pd.DataFrame.from_dict(positions, orient='index')
        
        # Format và sắp xếp columns
        if not positions_df.empty:
            columns_to_show = ['symbol', 'side', 'size', 'entry_price', 'current_price', 
                            'unrealized_pnl', 'unrealized_pnl_percent', 'leverage']
            
            # Lấy các columns có sẵn
            available_columns = [col for col in columns_to_show if col in positions_df.columns]
            
            # Hiển thị DataFrame
            st.dataframe(positions_df[available_columns])
        
        # Hiển thị phân tích margin
        st.subheader("Phân tích margin")
        margin_info = position_tracker.check_margin_level(positions)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Margin Level",
                value=f"{margin_info.get('margin_level', float('inf')):.2f}",
                delta=margin_info.get('warning_level', 'safe'),
                delta_color="inverse",
                help="Tỷ lệ giữa (Balance + Unrealized PnL) / Used Margin"
            )
        
        with col2:
            st.metric(
                label="Số dư khả dụng",
                value=f"{margin_info.get('available_balance', 0):.2f}",
                help="Số dư tài khoản khả dụng"
            )
        
        with col3:
            st.metric(
                label="Margin sử dụng",
                value=f"{margin_info.get('used_margin', 0):.2f}",
                help="Margin đã sử dụng cho các vị thế"
            )
        
        # Hiển thị các vị thế có rủi ro thanh lý
        positions_at_risk = margin_info.get('positions_at_risk', {})
        if positions_at_risk:
            st.warning("Có vị thế đang có rủi ro thanh lý!")
            
            # Convert to DataFrame
            risk_df = pd.DataFrame.from_dict(positions_at_risk, orient='index')
            risk_df.reset_index(inplace=True)
            risk_df.rename(columns={'index': 'symbol'}, inplace=True)
            
            st.dataframe(risk_df)
    
    def display_portfolio_analysis(self, portfolio_manager: PortfolioManager) -> None:
        """
        Hiển thị phân tích danh mục đầu tư.
        
        Args:
            portfolio_manager: Đối tượng PortfolioManager đã được khởi tạo
        """
        # Lấy trạng thái danh mục
        portfolio_status = portfolio_manager.get_portfolio_status()
        
        # Tạo 2 rows với 4 columns mỗi row
        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
        
        with row1_col1:
            st.metric(
                label="Giá trị danh mục",
                value=f"{portfolio_status.get('portfolio_value', 0):.2f}",
                help="Tổng giá trị danh mục đầu tư"
            )
        
        with row1_col2:
            st.metric(
                label="Vốn khả dụng",
                value=f"{portfolio_status.get('current_capital', 0):.2f}",
                help="Vốn khả dụng chưa được phân bổ"
            )
        
        with row1_col3:
            st.metric(
                label="Tổng P&L",
                value=f"{portfolio_status.get('total_pnl', 0):.2f}",
                delta=f"{portfolio_status.get('realized_pnl', 0):.2f} đã thực hiện",
                help="Tổng lợi nhuận của danh mục"
            )
        
        with row1_col4:
            st.metric(
                label="Hồ sơ rủi ro",
                value=portfolio_status.get('risk_profile', 'N/A').title(),
                help="Hồ sơ rủi ro của danh mục"
            )
        
        # Hiển thị biểu đồ phân bổ danh mục
        st.subheader("Phân bổ danh mục")
        allocation_chart = self.visualizer.plot_portfolio_allocation(portfolio_manager)
        st.plotly_chart(allocation_chart, use_container_width=True)
        
        # Hiển thị chi tiết các tài sản
        st.subheader("Chi tiết tài sản")
        
        # Convert portfolio to DataFrame
        assets = portfolio_manager.portfolio
        if assets:
            assets_df = pd.DataFrame.from_dict(assets, orient='index')
            
            # Format và sắp xếp columns
            columns_to_show = ['symbol', 'category', 'amount', 'current_price', 'average_price',
                            'value', 'target_allocation', 'current_allocation', 
                            'unrealized_pnl', 'realized_pnl', 'total_pnl']
            
            # Lấy các columns có sẵn
            available_columns = [col for col in columns_to_show if col in assets_df.columns]
            
            # Hiển thị DataFrame
            st.dataframe(assets_df[available_columns])
        else:
            st.info("Không có tài sản nào trong danh mục")
        
        # Hiển thị ma trận tương quan
        st.subheader("Ma trận tương quan")
        
        if not portfolio_manager.correlation_matrix.empty:
            correlation_chart = self.visualizer.plot_correlation_matrix(
                correlation_matrix=portfolio_manager.correlation_matrix
            )
            st.plotly_chart(correlation_chart, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu ma trận tương quan")
        
        # Hiển thị lịch sử giao dịch
        st.subheader("Lịch sử giao dịch")
        
        trade_history = portfolio_manager.get_trade_history_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Tổng số giao dịch",
                value=f"{trade_history.get('total_trades', 0)}",
                help="Tổng số giao dịch đã thực hiện"
            )
        
        with col2:
            st.metric(
                label="Win Rate",
                value=f"{trade_history.get('win_rate', 0):.2f}%",
                delta=f"{trade_history.get('win_count', 0)}/{trade_history.get('loss_count', 0)}",
                help="Tỷ lệ giao dịch thắng"
            )
        
        with col3:
            st.metric(
                label="Profit Factor",
                value=f"{trade_history.get('profit_factor', 0):.2f}",
                help="Tỷ lệ giữa tổng lợi nhuận và tổng thua lỗ"
            )
        
        with col4:
            st.metric(
                label="Tổng P&L",
                value=f"{trade_history.get('total_pnl', 0):.2f}",
                help="Tổng lợi nhuận từ giao dịch"
            )
        
        # Hiển thị chi tiết giao dịch
        if 'position_history' in portfolio_manager.__dict__ and portfolio_manager.position_history:
            history_df = pd.DataFrame(portfolio_manager.position_history)
            
            # Format datetime columns
            for col in ['entry_time', 'exit_time']:
                if col in history_df.columns:
                    history_df[col] = pd.to_datetime(history_df[col])
            
            st.dataframe(history_df)
        else:
            st.info("Chưa có lịch sử giao dịch")
    
    def display_full_risk_dashboard(self, risk_evaluator: RiskEvaluator,
                                 position_tracker: Optional[PositionTracker] = None,
                                 portfolio_manager: Optional[PortfolioManager] = None) -> None:
        """
        Hiển thị dashboard rủi ro đầy đủ.
        
        Args:
            risk_evaluator: Đối tượng RiskEvaluator đã được khởi tạo
            position_tracker: Đối tượng PositionTracker (optional)
            portfolio_manager: Đối tượng PortfolioManager (optional)
        """
        st.title("Bảng điều khiển rủi ro")
        
        # Tạo tabs
        tabs = ["Tổng quan rủi ro", "Drawdown", "Biến động", "VaR", "Phân phối", 
              "Rủi ro-Lợi nhuận", "Stress Test", "Monte Carlo"]
        
        # Thêm tabs cho position và portfolio nếu có
        if position_tracker is not None:
            tabs.append("Vị thế")
        
        if portfolio_manager is not None:
            tabs.append("Danh mục đầu tư")
        
        selected_tab = st.tabs(tabs)
        
        # Tab Tổng quan rủi ro
        with selected_tab[0]:
            self.display_risk_overview(risk_evaluator)
        
        # Tab Drawdown
        with selected_tab[1]:
            self.display_drawdown_analysis(risk_evaluator)
        
        # Tab Biến động
        with selected_tab[2]:
            self.display_volatility_analysis(risk_evaluator)
        
        # Tab VaR
        with selected_tab[3]:
            self.display_var_analysis(risk_evaluator)
        
        # Tab Phân phối
        with selected_tab[4]:
            self.display_distribution_analysis(risk_evaluator)
        
        # Tab Rủi ro-Lợi nhuận
        with selected_tab[5]:
            self.display_risk_return_metrics(risk_evaluator)
        
        # Tab Stress Test
        with selected_tab[6]:
            self.display_stress_test(risk_evaluator)
        
        # Tab Monte Carlo
        with selected_tab[7]:
            self.display_monte_carlo_simulation(risk_evaluator)
        
        # Tab Vị thế (nếu có)
        if position_tracker is not None:
            tab_index = 8
            with selected_tab[tab_index]:
                self.display_position_analysis(position_tracker)
            tab_index += 1
        
        # Tab Danh mục đầu tư (nếu có)
        if portfolio_manager is not None:
            tab_index = 9 if position_tracker is not None else 8
            with selected_tab[tab_index]:
                self.display_portfolio_analysis(portfolio_manager)