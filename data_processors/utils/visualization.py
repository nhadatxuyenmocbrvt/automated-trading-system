"""
Tiện ích trực quan hóa đặc trưng và kết quả.
File này cung cấp các hàm để trực quan hóa đặc trưng, mối quan hệ giữa các đặc trưng,
và phân tích tầm quan trọng của đặc trưng, giúp hiểu rõ hơn về dữ liệu và mô hình.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
import warnings

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import setup_logger

# Logger
logger = setup_logger("feature_visualization")

def plot_feature_importance(
    feature_importance: Union[pd.Series, Dict[str, float], List[Tuple[str, float]]],
    top_n: int = 20,
    title: str = "Tầm quan trọng của các đặc trưng",
    figsize: Tuple[int, int] = (12, 8),
    color: str = 'viridis',
    show_values: bool = True,
    save_path: Optional[str] = None,
    sort_ascending: bool = False
) -> Any:
    """
    Vẽ biểu đồ tầm quan trọng của các đặc trưng.
    
    Args:
        feature_importance: Series, Dict, hoặc List chứa tầm quan trọng của các đặc trưng
        top_n: Số lượng đặc trưng quan trọng nhất để hiển thị
        title: Tiêu đề biểu đồ
        figsize: Kích thước của biểu đồ (chiều rộng, chiều cao)
        color: Màu sắc của biểu đồ ('viridis', 'plasma', 'inferno', 'magma', 'cividis')
        show_values: Hiển thị giá trị bên cạnh thanh
        save_path: Đường dẫn để lưu biểu đồ
        sort_ascending: Sắp xếp theo thứ tự tăng dần (mặc định là giảm dần)
    
    Returns:
        Đối tượng Figure hoặc Axes
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Chuyển đổi dữ liệu đầu vào thành DataFrame
        if isinstance(feature_importance, pd.Series):
            df = pd.DataFrame({'feature': feature_importance.index, 'importance': feature_importance.values})
        elif isinstance(feature_importance, dict):
            df = pd.DataFrame({'feature': list(feature_importance.keys()), 'importance': list(feature_importance.values())})
        elif isinstance(feature_importance, list) and all(isinstance(item, tuple) and len(item) == 2 for item in feature_importance):
            df = pd.DataFrame({'feature': [item[0] for item in feature_importance], 
                              'importance': [item[1] for item in feature_importance]})
        else:
            logger.error("Định dạng đầu vào không hợp lệ. Cần pd.Series, Dict, hoặc List[Tuple]")
            return None
        
        # Sắp xếp theo tầm quan trọng
        df = df.sort_values('importance', ascending=sort_ascending)
        
        # Lấy top_n đặc trưng
        if top_n is not None and top_n < len(df):
            if sort_ascending:
                df = df.head(top_n)
            else:
                df = df.tail(top_n)
        
        # Tạo biểu đồ
        plt.figure(figsize=figsize)
        ax = sns.barplot(x='importance', y='feature', data=df, palette=color)
        
        # Thêm giá trị nếu cần
        if show_values:
            for i, value in enumerate(df['importance']):
                ax.text(value + max(df['importance']) * 0.01, i, f"{value:.4f}", va='center')
        
        # Định dạng biểu đồ
        plt.title(title, fontsize=14)
        plt.xlabel('Tầm quan trọng', fontsize=12)
        plt.ylabel('Đặc trưng', fontsize=12)
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Đã lưu biểu đồ tầm quan trọng đặc trưng vào {save_path}")
        
        return ax
        
    except ImportError as e:
        logger.error(f"Không thể vẽ biểu đồ: {e}")
        logger.error("Hãy cài đặt matplotlib và seaborn: pip install matplotlib seaborn")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ tầm quan trọng đặc trưng: {e}")
        return None

def plot_feature_correlation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    method: str = 'pearson',
    figsize: Tuple[int, int] = (14, 12),
    cmap: str = 'coolwarm',
    annot: bool = True,
    save_path: Optional[str] = None,
    top_n: Optional[int] = None,
    drop_self: bool = True
) -> Any:
    """
    Vẽ biểu đồ tương quan giữa các đặc trưng.
    
    Args:
        df: DataFrame chứa dữ liệu
        columns: Danh sách các cột cần sử dụng (None để sử dụng tất cả)
        target_column: Cột mục tiêu để sắp xếp tương quan (None để không sắp xếp)
        method: Phương pháp tính tương quan ('pearson', 'kendall', 'spearman')
        figsize: Kích thước của biểu đồ (chiều rộng, chiều cao)
        cmap: Bảng màu cho heatmap
        annot: Hiển thị giá trị tương quan trên heatmap
        save_path: Đường dẫn để lưu biểu đồ
        top_n: Chỉ hiển thị top_n đặc trưng có tương quan cao nhất với target_column
        drop_self: Loại bỏ các đặc trưng có tương quan 1.0 với chính nó
    
    Returns:
        Đối tượng Figure hoặc Axes
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Chọn các cột cần sử dụng
        if columns is None:
            # Chỉ sử dụng các cột số
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Lọc các cột số từ danh sách đã cho
            columns = [col for col in columns if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
        
        if not columns:
            logger.error("Không có cột số nào để vẽ ma trận tương quan")
            return None
        
        # Tính ma trận tương quan
        corr_matrix = df[columns].corr(method=method)
        
        # Xử lý target_column và top_n
        if target_column is not None and target_column in corr_matrix.columns:
            # Lấy tương quan với target_column
            target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)
            
            if drop_self and target_column in target_corr.index:
                target_corr = target_corr.drop(target_column)
            
            if top_n is not None and top_n < len(target_corr):
                # Lấy top_n đặc trưng có tương quan cao nhất
                top_features = target_corr.head(top_n).index.tolist()
                if target_column not in top_features and target_column in corr_matrix.columns:
                    top_features.append(target_column)
                corr_matrix = corr_matrix.loc[top_features, top_features]
        
        # Tạo biểu đồ
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) if drop_self else None
        ax = sns.heatmap(corr_matrix, annot=annot, cmap=cmap, mask=mask, 
                        fmt='.2f', linewidths=0.5, center=0, vmin=-1, vmax=1)
        
        plt.title(f'Ma trận tương quan ({method})', fontsize=14)
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Đã lưu biểu đồ ma trận tương quan vào {save_path}")
        
        return ax
        
    except ImportError as e:
        logger.error(f"Không thể vẽ biểu đồ: {e}")
        logger.error("Hãy cài đặt matplotlib và seaborn: pip install matplotlib seaborn")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ tương quan đặc trưng: {e}")
        return None

def visualize_feature_distribution(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    n_cols: int = 3,
    figsize: Tuple[int, int] = (16, 12),
    bins: int = 30,
    kde: bool = True,
    save_path: Optional[str] = None,
    max_features: int = 15
) -> Any:
    """
    Trực quan hóa phân phối của các đặc trưng.
    
    Args:
        df: DataFrame chứa dữ liệu
        columns: Danh sách các cột cần sử dụng (None để sử dụng tất cả)
        target_column: Cột mục tiêu để sử dụng cho màu sắc (hue)
        n_cols: Số cột trong lưới biểu đồ
        figsize: Kích thước của biểu đồ (chiều rộng, chiều cao)
        bins: Số lượng bins cho histogram
        kde: Vẽ đường ước lượng mật độ kernel (KDE)
        save_path: Đường dẫn để lưu biểu đồ
        max_features: Số lượng đặc trưng tối đa để hiển thị
    
    Returns:
        Đối tượng Figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Chọn các cột cần sử dụng
        if columns is None:
            # Chỉ sử dụng các cột số
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Lọc các cột số từ danh sách đã cho
            columns = [col for col in columns if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
        
        if not columns:
            logger.error("Không có cột số nào để vẽ phân phối")
            return None
        
        # Giới hạn số lượng đặc trưng
        if max_features is not None and len(columns) > max_features:
            logger.warning(f"Số lượng đặc trưng ({len(columns)}) vượt quá giới hạn ({max_features}). Chỉ hiển thị {max_features} đặc trưng đầu tiên.")
            columns = columns[:max_features]
        
        # Tính số hàng cần thiết
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        # Tạo biểu đồ
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        # Vẽ phân phối cho từng đặc trưng
        for i, col in enumerate(columns):
            if i < len(axes):
                if target_column and target_column in df.columns:
                    # Phân phối theo target
                    try:
                        if df[target_column].nunique() <= 10:  # Nếu target là phân loại với ít giá trị
                            sns.histplot(data=df, x=col, hue=target_column, kde=kde, bins=bins, 
                                        multiple="stack", ax=axes[i])
                        else:
                            sns.histplot(data=df, x=col, kde=kde, bins=bins, ax=axes[i])
                    except:
                        # Fallback nếu có lỗi với target
                        sns.histplot(data=df, x=col, kde=kde, bins=bins, ax=axes[i])
                else:
                    # Phân phối đơn giản
                    sns.histplot(data=df, x=col, kde=kde, bins=bins, ax=axes[i])
                
                # Định dạng tiêu đề và nhãn
                axes[i].set_title(col)
                axes[i].set_xlabel("")
                
                # Làm đẹp đồ thị
                axes[i].grid(alpha=0.3)
        
        # Ẩn các trục thừa
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        # Điều chỉnh layout
        plt.tight_layout()
        
        # Thêm tiêu đề chung
        if target_column:
            fig.suptitle(f'Phân phối của các đặc trưng (theo {target_column})', fontsize=16, y=1.02)
        else:
            fig.suptitle('Phân phối của các đặc trưng', fontsize=16, y=1.02)
        
        # Lưu biểu đồ nếu cần
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Đã lưu biểu đồ phân phối đặc trưng vào {save_path}")
        
        return fig
        
    except ImportError as e:
        logger.error(f"Không thể vẽ biểu đồ: {e}")
        logger.error("Hãy cài đặt matplotlib và seaborn: pip install matplotlib seaborn")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ phân phối đặc trưng: {e}")
        return None

def plot_feature_pairwise(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 14),
    n_features: int = 5,
    diag_kind: str = 'kde',
    save_path: Optional[str] = None,
    plot_kind: str = 'scatter'
) -> Any:
    """
    Vẽ biểu đồ cặp đặc trưng để phân tích mối quan hệ.
    
    Args:
        df: DataFrame chứa dữ liệu
        columns: Danh sách các cột cần sử dụng (None để sử dụng tất cả)
        target_column: Cột mục tiêu để sử dụng cho màu sắc (hue)
        figsize: Kích thước của biểu đồ (chiều rộng, chiều cao)
        n_features: Số lượng đặc trưng tối đa để hiển thị
        diag_kind: Loại biểu đồ trên đường chéo ('hist', 'kde')
        save_path: Đường dẫn để lưu biểu đồ
        plot_kind: Loại biểu đồ ('scatter', 'reg')
    
    Returns:
        Đối tượng PairGrid hoặc Figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Chọn các cột cần sử dụng
        if columns is None:
            # Chỉ sử dụng các cột số
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Lọc các cột số từ danh sách đã cho
            columns = [col for col in columns if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
        
        if not columns:
            logger.error("Không có cột số nào để vẽ biểu đồ cặp")
            return None
        
        # Chọn các đặc trưng quan trọng nhất nếu cần
        if target_column is not None and target_column in df.columns and len(columns) > n_features:
            # Tính tương quan với target
            corr_with_target = df[columns].corrwith(df[target_column]).abs().sort_values(ascending=False)
            columns = corr_with_target.head(n_features).index.tolist()
            
            # Đảm bảo target_column nằm trong danh sách
            if target_column not in columns and np.issubdtype(df[target_column].dtype, np.number):
                columns.append(target_column)
        
        # Giới hạn số lượng đặc trưng
        if len(columns) > n_features:
            logger.warning(f"Số lượng đặc trưng ({len(columns)}) vượt quá giới hạn ({n_features}). Chỉ hiển thị {n_features} đặc trưng đầu tiên.")
            columns = columns[:n_features]
        
        # Tạo biểu đồ
        if target_column and target_column in df.columns:
            # Kiểm tra xem target có phân loại không
            if df[target_column].nunique() <= 10 and not np.issubdtype(df[target_column].dtype, np.number):
                # Đối với target phân loại
                g = sns.pairplot(df, vars=columns, hue=target_column, 
                                 height=figsize[0]/len(columns), aspect=figsize[1]/figsize[0],
                                 diag_kind=diag_kind, plot_kws={'alpha': 0.6})
            else:
                # Đối với target liên tục hoặc với nhiều giá trị
                if plot_kind == 'reg':
                    g = sns.pairplot(df, vars=columns, height=figsize[0]/len(columns), aspect=figsize[1]/figsize[0],
                                     diag_kind=diag_kind, kind='reg', plot_kws={'scatter_kws': {'alpha': 0.3}})
                else:
                    g = sns.pairplot(df, vars=columns, height=figsize[0]/len(columns), aspect=figsize[1]/figsize[0],
                                     diag_kind=diag_kind, plot_kws={'alpha': 0.6})
        else:
            # Biểu đồ không có target
            if plot_kind == 'reg':
                g = sns.pairplot(df, vars=columns, height=figsize[0]/len(columns), aspect=figsize[1]/figsize[0],
                                 diag_kind=diag_kind, kind='reg', plot_kws={'scatter_kws': {'alpha': 0.3}})
            else:
                g = sns.pairplot(df, vars=columns, height=figsize[0]/len(columns), aspect=figsize[1]/figsize[0],
                                 diag_kind=diag_kind, plot_kws={'alpha': 0.6})
        
        # Điều chỉnh layout
        plt.tight_layout()
        
        # Thêm tiêu đề
        if target_column:
            g.fig.suptitle(f'Biểu đồ cặp đặc trưng (theo {target_column})', fontsize=16, y=1.02)
        else:
            g.fig.suptitle('Biểu đồ cặp đặc trưng', fontsize=16, y=1.02)
        
        # Lưu biểu đồ nếu cần
        if save_path:
            g.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Đã lưu biểu đồ cặp đặc trưng vào {save_path}")
        
        return g
        
    except ImportError as e:
        logger.error(f"Không thể vẽ biểu đồ: {e}")
        logger.error("Hãy cài đặt matplotlib và seaborn: pip install matplotlib seaborn")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ cặp đặc trưng: {e}")
        return None

def plot_feature_importance_by_models(
    models_feature_importance: Dict[str, Union[pd.Series, Dict[str, float], List[Tuple[str, float]]]],
    top_n: int = 15,
    figsize: Tuple[int, int] = (14, 10),
    colors: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    sort_by: Optional[str] = None
) -> Any:
    """
    Vẽ biểu đồ so sánh tầm quan trọng của đặc trưng từ nhiều mô hình.
    
    Args:
        models_feature_importance: Dict với key là tên mô hình và value là tầm quan trọng của đặc trưng
        top_n: Số lượng đặc trưng quan trọng nhất để hiển thị
        figsize: Kích thước của biểu đồ (chiều rộng, chiều cao)
        colors: Danh sách màu sắc cho các mô hình
        save_path: Đường dẫn để lưu biểu đồ
        sort_by: Sắp xếp theo tên mô hình cụ thể
    
    Returns:
        Đối tượng Figure hoặc Axes
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Chuyển đổi tất cả các đầu vào thành DataFrame
        dataframes = {}
        for model_name, importance in models_feature_importance.items():
            if isinstance(importance, pd.Series):
                dataframes[model_name] = pd.DataFrame({'feature': importance.index, 'importance': importance.values})
            elif isinstance(importance, dict):
                dataframes[model_name] = pd.DataFrame({'feature': list(importance.keys()), 'importance': list(importance.values())})
            elif isinstance(importance, list) and all(isinstance(item, tuple) and len(item) == 2 for item in importance):
                dataframes[model_name] = pd.DataFrame({'feature': [item[0] for item in importance], 
                                              'importance': [item[1] for item in importance]})
            else:
                logger.warning(f"Định dạng đầu vào không hợp lệ cho {model_name}. Bỏ qua.")
        
        if not dataframes:
            logger.error("Không có dữ liệu hợp lệ để vẽ biểu đồ")
            return None
        
        # Tìm tập hợp các đặc trưng quan trọng
        all_features = set()
        for df in dataframes.values():
            all_features.update(df['feature'])
        
        # Tạo DataFrame tổng hợp
        combined_df = pd.DataFrame(index=all_features)
        
        for model_name, df in dataframes.items():
            # Chuẩn hóa tầm quan trọng
            max_importance = df['importance'].max()
            if max_importance > 0:
                df['importance'] = df['importance'] / max_importance
            
            # Thêm vào DataFrame tổng hợp
            temp_series = pd.Series(df.set_index('feature')['importance'])
            combined_df[model_name] = temp_series
        
        # Điền giá trị NaN bằng 0
        combined_df = combined_df.fillna(0)
        
        # Tính tổng tầm quan trọng qua các mô hình
        combined_df['total_importance'] = combined_df.sum(axis=1)
        
        # Sắp xếp theo model cụ thể hoặc theo tổng
        if sort_by in combined_df.columns:
            combined_df = combined_df.sort_values(sort_by, ascending=False)
        else:
            combined_df = combined_df.sort_values('total_importance', ascending=False)
        
        # Lấy top_n đặc trưng
        if top_n is not None and top_n < len(combined_df):
            combined_df = combined_df.head(top_n)
        
        # Loại bỏ cột total_importance
        if 'total_importance' in combined_df.columns:
            combined_df = combined_df.drop('total_importance', axis=1)
        
        # Chuẩn bị dữ liệu cho biểu đồ
        plot_df = combined_df.reset_index().melt(id_vars=['index'], var_name='model', value_name='importance')
        plot_df = plot_df.rename(columns={'index': 'feature'})
        
        # Tạo biểu đồ
        plt.figure(figsize=figsize)
        ax = sns.barplot(x='feature', y='importance', hue='model', data=plot_df, palette=colors)
        
        # Xoay nhãn trục x
        plt.xticks(rotation=45, ha='right')
        
        # Định dạng biểu đồ
        plt.title('So sánh tầm quan trọng đặc trưng giữa các mô hình', fontsize=14)
        plt.xlabel('Đặc trưng', fontsize=12)
        plt.ylabel('Tầm quan trọng (đã chuẩn hóa)', fontsize=12)
        plt.legend(title='Mô hình')
        plt.tight_layout()
        
        # Lưu biểu đồ nếu cần
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Đã lưu biểu đồ so sánh tầm quan trọng đặc trưng vào {save_path}")
        
        return ax
        
    except ImportError as e:
        logger.error(f"Không thể vẽ biểu đồ: {e}")
        logger.error("Hãy cài đặt matplotlib và seaborn: pip install matplotlib seaborn")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ so sánh tầm quan trọng đặc trưng: {e}")
        return None

def plot_feature_evolution(
    feature_values: Dict[str, List[float]],
    timestamps: Optional[List[Union[str, pd.Timestamp]]] = None,
    top_n: int = 5,
    figsize: Tuple[int, int] = (14, 8),
    title: str = "Sự tiến triển của đặc trưng theo thời gian",
    save_path: Optional[str] = None,
    rolling_window: Optional[int] = None
) -> Any:
    """
    Vẽ biểu đồ theo dõi sự tiến triển của các đặc trưng theo thời gian.
    
    Args:
        feature_values: Dict với key là tên đặc trưng và value là danh sách giá trị theo thời gian
        timestamps: Danh sách thời điểm tương ứng với các giá trị
        top_n: Số lượng đặc trưng hiển thị (lấy theo độ biến động)
        figsize: Kích thước của biểu đồ (chiều rộng, chiều cao)
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn để lưu biểu đồ
        rolling_window: Kích thước cửa sổ trượt để làm mịn dữ liệu
    
    Returns:
        Đối tượng Figure hoặc Axes
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Tạo DataFrame từ dữ liệu đầu vào
        df = pd.DataFrame(feature_values)
        
        # Thêm timestamps nếu có
        if timestamps is not None:
            if len(timestamps) == len(df):
                df.index = pd.to_datetime(timestamps)
            else:
                logger.warning(f"Số lượng timestamps ({len(timestamps)}) không khớp với số lượng giá trị ({len(df)}). Sử dụng index mặc định.")
        
        # Tính độ biến động của các đặc trưng
        feature_volatility = df.std()
        
        # Chọn top_n đặc trưng có độ biến động cao nhất
        top_features = feature_volatility.nlargest(top_n).index.tolist()
        
        # Lọc DataFrame
        plot_df = df[top_features]
        
        # Áp dụng rolling window nếu cần
        if rolling_window is not None and rolling_window > 1:
            plot_df = plot_df.rolling(window=rolling_window).mean()
            plot_df = plot_df.dropna()
        
        # Vẽ biểu đồ
        plt.figure(figsize=figsize)
        for feature in plot_df.columns:
            plt.plot(plot_df.index, plot_df[feature], label=feature, linewidth=2)
        
        # Định dạng biểu đồ
        plt.title(title, fontsize=14)
        plt.xlabel('Thời gian' if isinstance(plot_df.index, pd.DatetimeIndex) else 'Điểm dữ liệu', fontsize=12)
        plt.ylabel('Giá trị đặc trưng', fontsize=12)
        plt.legend(title='Đặc trưng')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Xoay nhãn trục x nếu là timestamps
        if isinstance(plot_df.index, pd.DatetimeIndex):
            plt.xticks(rotation=45)
        
        # Lưu biểu đồ nếu cần
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Đã lưu biểu đồ tiến triển đặc trưng vào {save_path}")
        
        return plt.gca()
        
    except ImportError as e:
        logger.error(f"Không thể vẽ biểu đồ: {e}")
        logger.error("Hãy cài đặt matplotlib và seaborn: pip install matplotlib seaborn")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ tiến triển đặc trưng: {e}")
        return None