"""
Lựa chọn và trích xuất đặc trưng.
Module này cung cấp các chức năng lựa chọn đặc trưng quan trọng,
giảm chiều dữ liệu, và đánh giá mức độ quan trọng của các đặc trưng.
"""

# Import các module con
from data_processors.feature_engineering.feature_selector.statistical_methods import (
    correlation_selector,
    chi_squared_selector,
    anova_selector,
    mutual_info_selector,
    calculate_feature_correlation
)

from data_processors.feature_engineering.feature_selector.importance_methods import (
    tree_importance_selector,
    random_forest_importance_selector,
    boosting_importance_selector,
    shap_importance_selector,
    permutation_importance_selector
)

from data_processors.feature_engineering.feature_selector.dimensionality_reduction import (
    pca_reducer,
    lda_reducer,
    tsne_reducer,
    umap_reducer,
    autoencoder_reducer
)

from data_processors.feature_engineering.feature_selector.wrapper_methods import (
    forward_selection,
    backward_elimination,
    recursive_feature_elimination,
    sequential_feature_selector
)

from data_processors.feature_engineering.feature_selector.feature_selection_pipeline import (
    FeatureSelectionPipeline,
    create_selection_pipeline,
    evaluate_feature_selection
)

# Danh sách các phương pháp lựa chọn đặc trưng có sẵn
AVAILABLE_METHODS = {
    "statistical": {
        "correlation": correlation_selector,
        "chi_squared": chi_squared_selector,
        "anova": anova_selector,
        "mutual_info": mutual_info_selector
    },
    "importance": {
        "tree": tree_importance_selector,
        "random_forest": random_forest_importance_selector,
        "boosting": boosting_importance_selector,
        "shap": shap_importance_selector,
        "permutation": permutation_importance_selector
    },
    "dimensionality_reduction": {
        "pca": pca_reducer,
        "lda": lda_reducer,
        "tsne": tsne_reducer,
        "umap": umap_reducer,
        "autoencoder": autoencoder_reducer
    },
    "wrapper": {
        "forward": forward_selection,
        "backward": backward_elimination,
        "rfe": recursive_feature_elimination,
        "sequential": sequential_feature_selector
    }
}

__all__ = [
    # Statistical methods
    'correlation_selector',
    'chi_squared_selector',
    'anova_selector',
    'mutual_info_selector',
    'calculate_feature_correlation',
    
    # Importance methods
    'tree_importance_selector',
    'random_forest_importance_selector',
    'boosting_importance_selector',
    'shap_importance_selector',
    'permutation_importance_selector',
    
    # Dimensionality reduction methods
    'pca_reducer',
    'lda_reducer',
    'tsne_reducer',
    'umap_reducer',
    'autoencoder_reducer',
    
    # Wrapper methods
    'forward_selection',
    'backward_elimination',
    'recursive_feature_elimination',
    'sequential_feature_selector',
    
    # Pipeline
    'FeatureSelectionPipeline',
    'create_selection_pipeline',
    'evaluate_feature_selection',
    
    # Dictionary of available methods
    'AVAILABLE_METHODS'
]