"""
Phương pháp giảm chiều dữ liệu.
Module này cung cấp các phương pháp giảm chiều dữ liệu như PCA, LDA, t-SNE, UMAP
và Autoencoder để giảm số lượng đặc trưng và trích xuất các đặc trưng mới có ý nghĩa.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import logging
import warnings
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import get_logger

# Thiết lập logger
logger = get_logger("feature_selector")

def pca_reducer(
    df: pd.DataFrame,
    n_components: Optional[Union[int, float]] = None,
    variance_threshold: float = 0.95,
    target_column: Optional[str] = None,
    exclude_columns: List[str] = [],
    return_components: bool = False,
    fit: bool = True,
    pca_model: Optional[Any] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Any]]:
    """
    Giảm chiều dữ liệu sử dụng PCA (Principal Component Analysis).
    
    Args:
        df: DataFrame chứa các đặc trưng
        n_components: Số lượng thành phần chính cần giữ lại (int) hoặc phần trăm phương sai cần giữ lại (float < 1)
        variance_threshold: Ngưỡng phương sai tích lũy khi n_components=None
        target_column: Tên cột mục tiêu (nếu có, sẽ được loại trừ khỏi PCA và giữ lại)
        exclude_columns: Danh sách cột cần loại trừ khỏi PCA
        return_components: Trả về mô hình PCA cùng với DataFrame kết quả
        fit: Học mô hình mới hay sử dụng mô hình đã cung cấp
        pca_model: Mô hình PCA đã được huấn luyện (chỉ khi fit=False)
        
    Returns:
        DataFrame chứa các thành phần chính, hoặc tuple (DataFrame, mô hình PCA) nếu return_components=True
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        if df.empty:
            logger.warning("DataFrame rỗng, không thể thực hiện PCA")
            if return_components:
                return df, None
            return df
        
        # Loại bỏ các cột không muốn xem xét
        columns_to_exclude = exclude_columns.copy()
        if target_column is not None and target_column in df.columns:
            columns_to_exclude.append(target_column)
        
        # Lọc các cột số
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in columns_to_exclude]
        
        if not feature_cols:
            logger.warning("Không có cột số nào phù hợp để thực hiện PCA")
            if return_components:
                return df, None
            return df
        
        # Lưu trữ các cột không tham gia PCA
        non_feature_cols = [col for col in df.columns if col not in feature_cols]
        non_feature_df = df[non_feature_cols].copy() if non_feature_cols else pd.DataFrame(index=df.index)
        
        # Chuẩn hóa dữ liệu
        X = df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if fit:
            # Xác định số lượng thành phần
            if n_components is None:
                # Sử dụng ngưỡng phương sai
                temp_pca = PCA()
                temp_pca.fit(X_scaled)
                cumulative_var = np.cumsum(temp_pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_var >= variance_threshold) + 1
                logger.info(f"Đã chọn {n_components} thành phần PCA dựa trên ngưỡng phương sai {variance_threshold}")
            
            # Thực hiện PCA
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Tạo tên cột cho các thành phần
            component_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
            
            # Ghi log thông tin phương sai giải thích được
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.sum(explained_var)
            logger.info(f"PCA đã giảm từ {len(feature_cols)} thành {len(component_cols)} đặc trưng")
            logger.info(f"Phương sai giải thích được: {cumulative_var:.2%}")
            
            # Tạo DataFrame kết quả
            pca_df = pd.DataFrame(X_pca, index=df.index, columns=component_cols)
            
        else:
            # Kiểm tra xem đã cung cấp mô hình chưa
            if pca_model is None:
                logger.error("Không thể áp dụng PCA: fit=False nhưng không cung cấp mô hình PCA")
                if return_components:
                    return df, None
                return df
            
            # Áp dụng mô hình đã có
            pca = pca_model
            X_pca = pca.transform(X_scaled)
            
            # Tạo tên cột cho các thành phần
            component_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
            
            # Tạo DataFrame kết quả
            pca_df = pd.DataFrame(X_pca, index=df.index, columns=component_cols)
        
        # Kết hợp với các cột không tham gia PCA
        result_df = pd.concat([pca_df, non_feature_df], axis=1)
        
        if return_components:
            # Lưu thông tin về các đặc trưng gốc và mô hình
            pca_info = {
                "model": pca,
                "scaler": scaler,
                "original_features": feature_cols,
                "n_components": pca.n_components_,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "component_cols": component_cols
            }
            return result_df, pca_info
        
        return result_df
        
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện PCA: {str(e)}")
        if return_components:
            return df, None
        return df

def lda_reducer(
    df: pd.DataFrame,
    target_column: str,
    n_components: Optional[int] = None,
    exclude_columns: List[str] = [],
    return_components: bool = False,
    fit: bool = True,
    lda_model: Optional[Any] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Any]]:
    """
    Giảm chiều dữ liệu sử dụng LDA (Linear Discriminant Analysis).
    
    Args:
        df: DataFrame chứa các đặc trưng
        target_column: Tên cột mục tiêu (biến phân loại)
        n_components: Số lượng thành phần cần giữ lại
        exclude_columns: Danh sách cột cần loại trừ khỏi LDA
        return_components: Trả về mô hình LDA cùng với DataFrame kết quả
        fit: Học mô hình mới hay sử dụng mô hình đã cung cấp
        lda_model: Mô hình LDA đã được huấn luyện (chỉ khi fit=False)
        
    Returns:
        DataFrame chứa các thành phần LDA, hoặc tuple (DataFrame, mô hình LDA) nếu return_components=True
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        if df.empty:
            logger.warning("DataFrame rỗng, không thể thực hiện LDA")
            if return_components:
                return df, None
            return df
        
        if target_column not in df.columns:
            logger.error(f"Cột mục tiêu '{target_column}' không tồn tại trong DataFrame")
            if return_components:
                return df, None
            return df
        
        # Loại bỏ các cột không muốn xem xét
        columns_to_exclude = exclude_columns.copy()
        columns_to_exclude.append(target_column)  # Loại trừ target khỏi đặc trưng
        
        # Lọc các cột số
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in columns_to_exclude]
        
        if not feature_cols:
            logger.warning("Không có cột số nào phù hợp để thực hiện LDA")
            if return_components:
                return df, None
            return df
        
        # Lưu trữ các cột không tham gia LDA (bao gồm cả target)
        non_feature_cols = [col for col in df.columns if col not in feature_cols]
        non_feature_df = df[non_feature_cols].copy() if non_feature_cols else pd.DataFrame(index=df.index)
        
        # Chuẩn hóa dữ liệu
        X = df[feature_cols].values
        y = df[target_column].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Xác định số lượng thành phần tối đa có thể
        n_classes = len(np.unique(y))
        max_components = min(n_classes - 1, len(feature_cols))
        
        if n_components is None:
            n_components = max_components
        else:
            n_components = min(n_components, max_components)
        
        if fit:
            # Thực hiện LDA
            lda = LinearDiscriminantAnalysis(n_components=n_components)
            X_lda = lda.fit_transform(X_scaled, y)
            
            # Tạo tên cột cho các thành phần
            component_cols = [f"LD{i+1}" for i in range(X_lda.shape[1])]
            
            # Ghi log thông tin
            if hasattr(lda, 'explained_variance_ratio_'):
                explained_var = lda.explained_variance_ratio_
                cumulative_var = np.sum(explained_var)
                logger.info(f"LDA đã giảm từ {len(feature_cols)} thành {len(component_cols)} đặc trưng")
                logger.info(f"Phương sai giải thích được: {cumulative_var:.2%}")
            
            # Tạo DataFrame kết quả
            lda_df = pd.DataFrame(X_lda, index=df.index, columns=component_cols)
            
        else:
            # Kiểm tra xem đã cung cấp mô hình chưa
            if lda_model is None:
                logger.error("Không thể áp dụng LDA: fit=False nhưng không cung cấp mô hình LDA")
                if return_components:
                    return df, None
                return df
            
            # Áp dụng mô hình đã có
            lda = lda_model
            X_lda = lda.transform(X_scaled)
            
            # Tạo tên cột cho các thành phần
            component_cols = [f"LD{i+1}" for i in range(X_lda.shape[1])]
            
            # Tạo DataFrame kết quả
            lda_df = pd.DataFrame(X_lda, index=df.index, columns=component_cols)
        
        # Kết hợp với các cột không tham gia LDA
        result_df = pd.concat([lda_df, non_feature_df], axis=1)
        
        if return_components:
            # Lưu thông tin về các đặc trưng gốc và mô hình
            lda_info = {
                "model": lda,
                "scaler": scaler,
                "original_features": feature_cols,
                "n_components": n_components,
                "component_cols": component_cols
            }
            if hasattr(lda, 'explained_variance_ratio_'):
                lda_info["explained_variance_ratio"] = lda.explained_variance_ratio_
            
            return result_df, lda_info
        
        return result_df
        
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện LDA: {str(e)}")
        if return_components:
            return df, None
        return df

def tsne_reducer(
    df: pd.DataFrame,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    target_column: Optional[str] = None,
    exclude_columns: List[str] = [],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Giảm chiều dữ liệu sử dụng t-SNE (t-distributed Stochastic Neighbor Embedding).
    
    Args:
        df: DataFrame chứa các đặc trưng
        n_components: Số lượng thành phần cần giữ lại
        perplexity: Liên quan đến số lượng láng giềng trong giảm chiều
        learning_rate: Tốc độ học cho t-SNE
        n_iter: Số lần lặp
        target_column: Tên cột mục tiêu (nếu có, sẽ được loại trừ khỏi t-SNE và giữ lại)
        exclude_columns: Danh sách cột cần loại trừ khỏi t-SNE
        random_state: Seed cho quá trình giảm chiều
        
    Returns:
        DataFrame chứa các thành phần t-SNE
    """
    try:
        # Kiểm tra thư viện sklearn
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logger.error("Không thể import TSNE từ sklearn.manifold")
            return df
        
        # Kiểm tra dữ liệu đầu vào
        if df.empty:
            logger.warning("DataFrame rỗng, không thể thực hiện t-SNE")
            return df
        
        # Loại bỏ các cột không muốn xem xét
        columns_to_exclude = exclude_columns.copy()
        if target_column is not None and target_column in df.columns:
            columns_to_exclude.append(target_column)
        
        # Lọc các cột số
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in columns_to_exclude]
        
        if not feature_cols:
            logger.warning("Không có cột số nào phù hợp để thực hiện t-SNE")
            return df
        
        # Lưu trữ các cột không tham gia t-SNE
        non_feature_cols = [col for col in df.columns if col not in feature_cols]
        non_feature_df = df[non_feature_cols].copy() if non_feature_cols else pd.DataFrame(index=df.index)
        
        # Chuẩn hóa dữ liệu
        X = df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Thực hiện t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(df) - 1),  # perplexity phải nhỏ hơn số mẫu - 1
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=random_state
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_tsne = tsne.fit_transform(X_scaled)
        
        # Tạo tên cột cho các thành phần
        component_cols = [f"TSNE{i+1}" for i in range(X_tsne.shape[1])]
        
        # Tạo DataFrame kết quả
        tsne_df = pd.DataFrame(X_tsne, index=df.index, columns=component_cols)
        
        # Kết hợp với các cột không tham gia t-SNE
        result_df = pd.concat([tsne_df, non_feature_df], axis=1)
        
        logger.info(f"t-SNE đã giảm từ {len(feature_cols)} thành {n_components} đặc trưng")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện t-SNE: {str(e)}")
        return df

def umap_reducer(
    df: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    target_column: Optional[str] = None,
    supervised: bool = False,
    exclude_columns: List[str] = [],
    random_state: int = 42,
    return_mapper: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Any]]:
    """
    Giảm chiều dữ liệu sử dụng UMAP (Uniform Manifold Approximation and Projection).
    
    Args:
        df: DataFrame chứa các đặc trưng
        n_components: Số lượng thành phần cần giữ lại
        n_neighbors: Số lượng láng giềng xem xét
        min_dist: Khoảng cách tối thiểu giữa các điểm trong không gian mới
        target_column: Tên cột mục tiêu (được sử dụng khi supervised=True)
        supervised: Sử dụng UMAP có giám sát hay không
        exclude_columns: Danh sách cột cần loại trừ khỏi UMAP
        random_state: Seed cho quá trình giảm chiều
        return_mapper: Trả về mapper UMAP cùng với DataFrame kết quả
        
    Returns:
        DataFrame chứa các thành phần UMAP, hoặc tuple (DataFrame, mapper) nếu return_mapper=True
    """
    try:
        # Kiểm tra thư viện umap-learn
        try:
            import umap
        except ImportError:
            logger.error("Thư viện umap-learn chưa được cài đặt. Vui lòng cài đặt với 'pip install umap-learn'")
            # Fallback sang PCA
            logger.info("Sử dụng PCA thay thế cho UMAP")
            result = pca_reducer(df, n_components=n_components, target_column=target_column, 
                               exclude_columns=exclude_columns, return_components=return_mapper)
            return result
        
        # Kiểm tra dữ liệu đầu vào
        if df.empty:
            logger.warning("DataFrame rỗng, không thể thực hiện UMAP")
            if return_mapper:
                return df, None
            return df
        
        # Loại bỏ các cột không muốn xem xét
        columns_to_exclude = exclude_columns.copy()
        columns_for_y = []
        
        if supervised and target_column is not None and target_column in df.columns:
            columns_for_y.append(target_column)
        elif target_column is not None and target_column in df.columns:
            columns_to_exclude.append(target_column)
        
        # Lọc các cột số
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in columns_to_exclude and col not in columns_for_y]
        
        if not feature_cols:
            logger.warning("Không có cột số nào phù hợp để thực hiện UMAP")
            if return_mapper:
                return df, None
            return df
        
        # Lưu trữ các cột không tham gia UMAP
        non_feature_cols = [col for col in df.columns if col not in feature_cols]
        non_feature_df = df[non_feature_cols].copy() if non_feature_cols else pd.DataFrame(index=df.index)
        
        # Chuẩn hóa dữ liệu
        X = df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Chuẩn bị tham số cho UMAP
        umap_params = {
            'n_components': n_components,
            'n_neighbors': min(n_neighbors, len(df) - 1),  # n_neighbors phải nhỏ hơn số mẫu
            'min_dist': min_dist,
            'random_state': random_state
        }
        
        # Thêm tham số y nếu là có giám sát
        if supervised and target_column is not None and target_column in df.columns:
            y = df[target_column].values
            umap_params['target_metric'] = 'categorical' if pd.api.types.is_categorical_dtype(df[target_column]) else 'l2'
            mapper = umap.UMAP(**umap_params)
            X_umap = mapper.fit_transform(X_scaled, y)
        else:
            mapper = umap.UMAP(**umap_params)
            X_umap = mapper.fit_transform(X_scaled)
        
        # Tạo tên cột cho các thành phần
        component_cols = [f"UMAP{i+1}" for i in range(X_umap.shape[1])]
        
        # Tạo DataFrame kết quả
        umap_df = pd.DataFrame(X_umap, index=df.index, columns=component_cols)
        
        # Kết hợp với các cột không tham gia UMAP
        result_df = pd.concat([umap_df, non_feature_df], axis=1)
        
        logger.info(f"UMAP đã giảm từ {len(feature_cols)} thành {n_components} đặc trưng")
        
        if return_mapper:
            # Lưu thông tin về các đặc trưng gốc và mapper
            umap_info = {
                "mapper": mapper,
                "scaler": scaler,
                "original_features": feature_cols,
                "n_components": n_components,
                "component_cols": component_cols
            }
            return result_df, umap_info
        
        return result_df
        
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện UMAP: {str(e)}")
        if return_mapper:
            return df, None
        return df

def autoencoder_reducer(
    df: pd.DataFrame,
    n_components: int = 2,
    hidden_layers: List[int] = [128, 64, 32],
    activation: str = 'relu',
    epochs: int = 100,
    batch_size: int = 32,
    target_column: Optional[str] = None,
    exclude_columns: List[str] = [],
    return_encoder: bool = False,
    random_state: int = 42
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Any]]:
    """
    Giảm chiều dữ liệu sử dụng Autoencoder.
    
    Args:
        df: DataFrame chứa các đặc trưng
        n_components: Số lượng thành phần cần giữ lại
        hidden_layers: Kích thước các tầng ẩn của encoder
        activation: Hàm kích hoạt cho các tầng ẩn
        epochs: Số epoch huấn luyện
        batch_size: Kích thước batch
        target_column: Tên cột mục tiêu (nếu có, sẽ được loại trừ khỏi Autoencoder và giữ lại)
        exclude_columns: Danh sách cột cần loại trừ khỏi Autoencoder
        return_encoder: Trả về encoder cùng với DataFrame kết quả
        random_state: Seed cho quá trình giảm chiều
        
    Returns:
        DataFrame chứa các thành phần từ Autoencoder, hoặc tuple (DataFrame, encoder) nếu return_encoder=True
    """
    try:
        # Kiểm tra thư viện tensorflow/keras
        try:
            import tensorflow as tf  # type: ignore
            from tensorflow.keras.models import Model  # type: ignore
            from tensorflow.keras.layers import Input, Dense  # type: ignore
            from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
            # Đặt seed cho TensorFlow
            tf.random.set_seed(random_state)
            np.random.seed(random_state)
        except ImportError:
            logger.error("Thư viện TensorFlow/Keras chưa được cài đặt. Vui lòng cài đặt với 'pip install tensorflow'")
            # Fallback sang PCA
            logger.info("Sử dụng PCA thay thế cho Autoencoder")
            result = pca_reducer(df, n_components=n_components, target_column=target_column, 
                               exclude_columns=exclude_columns, return_components=return_encoder)
            return result
        
        # Kiểm tra dữ liệu đầu vào
        if df.empty:
            logger.warning("DataFrame rỗng, không thể thực hiện Autoencoder")
            if return_encoder:
                return df, None
            return df
        
        # Loại bỏ các cột không muốn xem xét
        columns_to_exclude = exclude_columns.copy()
        if target_column is not None and target_column in df.columns:
            columns_to_exclude.append(target_column)
        
        # Lọc các cột số
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in columns_to_exclude]
        
        if not feature_cols:
            logger.warning("Không có cột số nào phù hợp để thực hiện Autoencoder")
            if return_encoder:
                return df, None
            return df
        
        # Lưu trữ các cột không tham gia Autoencoder
        non_feature_cols = [col for col in df.columns if col not in feature_cols]
        non_feature_df = df[non_feature_cols].copy() if non_feature_cols else pd.DataFrame(index=df.index)
        
        # Chuẩn hóa dữ liệu
        X = df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=random_state)
        
        # Xây dựng mô hình Autoencoder
        input_dim = X_scaled.shape[1]
        input_layer = Input(shape=(input_dim,))
        
        # Xây dựng encoder
        x = input_layer
        for units in hidden_layers:
            x = Dense(units, activation=activation)(x)
        
        # Tầng bottleneck (latent space)
        bottleneck = Dense(n_components, activation=activation, name='bottleneck')(x)
        
        # Xây dựng decoder
        x = bottleneck
        for units in reversed(hidden_layers):
            x = Dense(units, activation=activation)(x)
        
        # Tầng đầu ra
        output_layer = Dense(input_dim, activation='linear')(x)
        
        # Tạo mô hình
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        encoder = Model(inputs=input_layer, outputs=bottleneck)
        
        # Biên dịch mô hình
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Tạo callback để dừng sớm nếu loss không giảm
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Huấn luyện mô hình
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            autoencoder.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_test, X_test),
                verbose=0,
                callbacks=[early_stopping]
            )
        
        # Mã hóa dữ liệu
        X_encoded = encoder.predict(X_scaled)
        
        # Tạo tên cột cho các thành phần
        component_cols = [f"AE{i+1}" for i in range(X_encoded.shape[1])]
        
        # Tạo DataFrame kết quả
        ae_df = pd.DataFrame(X_encoded, index=df.index, columns=component_cols)
        
        # Kết hợp với các cột không tham gia Autoencoder
        result_df = pd.concat([ae_df, non_feature_df], axis=1)
        
        logger.info(f"Autoencoder đã giảm từ {len(feature_cols)} thành {n_components} đặc trưng")
        
        if return_encoder:
            # Lưu thông tin về các đặc trưng gốc và encoder
            encoder_info = {
                "encoder": encoder,
                "scaler": scaler,
                "original_features": feature_cols,
                "n_components": n_components,
                "component_cols": component_cols
            }
            return result_df, encoder_info
        
        return result_df
        
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện Autoencoder: {str(e)}")
        # Fallback sang PCA
        logger.info("Sử dụng PCA thay thế cho Autoencoder do lỗi")
        result = pca_reducer(df, n_components=n_components, target_column=target_column, 
                           exclude_columns=exclude_columns, return_components=return_encoder)
        return result