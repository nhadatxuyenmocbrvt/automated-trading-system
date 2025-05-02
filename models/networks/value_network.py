"""
Mạng giá trị (Value Network).
File này định nghĩa các mạng neural để ước lượng giá trị của trạng thái hoặc cặp trạng thái-hành động,
được sử dụng trong các thuật toán Reinforcement Learning như DQN, DDPG, A2C, v.v.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, LSTM, GRU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import HeUniform, GlorotUniform

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config

class ValueNetwork:
    """
    Mạng neural để ước lượng giá trị của trạng thái hoặc cặp trạng thái-hành động.
    """

    def __init__(
        self,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int = 0,  # 0 cho pure value network, > 0 cho Q-network
        hidden_layers: List[int] = [64, 64],
        activation: str = 'relu',
        learning_rate: float = 0.001,
        network_type: str = 'mlp',
        dueling: bool = False,
        name: str = 'value_network',
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo mạng giá trị.

        Args:
            state_dim: Kích thước không gian trạng thái
            action_dim: Kích thước không gian hành động (0 cho V(s), > 0 cho Q(s,a))
            hidden_layers: Danh sách số lượng neuron trong các lớp ẩn
            activation: Hàm kích hoạt
            learning_rate: Tốc độ học
            network_type: Loại mạng neural ('mlp', 'cnn', 'rnn')
            dueling: Sử dụng kiến trúc dueling cho DQN hay không
            name: Tên của mạng
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("value_network")

        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()

        # Thiết lập các thuộc tính
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.network_type = network_type.lower()
        self.dueling = dueling
        self.name = name
        self.kwargs = kwargs

        # Thiết lập optimizer
        if kwargs.get('optimizer', '').lower() == 'rmsprop':
            self.optimizer = RMSprop(learning_rate=learning_rate)
        else:
            self.optimizer = Adam(learning_rate=learning_rate)

        # Xây dựng mạng neural
        self.model = self._build_model()

        self.logger.info(
            f"Đã khởi tạo {self.__class__.__name__} với state_dim={state_dim}, "
            f"action_dim={action_dim}, network_type={network_type}, dueling={dueling}"
        )

    def _build_model(self) -> Model:
        """
        Xây dựng mạng neural.

        Returns:
            Model Keras
        """
        if self.network_type == 'mlp':
            return self._build_mlp()
        elif self.network_type == 'cnn':
            return self._build_cnn()
        elif self.network_type == 'rnn':
            return self._build_rnn()
        else:
            self.logger.warning(f"Loại mạng {self.network_type} không được hỗ trợ, sử dụng MLP mặc định")
            return self._build_mlp()

    def _build_mlp(self) -> Model:
        """
        Xây dựng mạng perceptron đa lớp (MLP).

        Returns:
            Model Keras
        """
        # Đầu vào là trạng thái
        state_input = Input(shape=self._get_input_shape(), name='state_input')
        x = state_input

        # Nếu đầu vào là mảng 2D hoặc cao hơn, làm phẳng
        if len(self._get_input_shape()) > 1:
            x = Flatten()(x)

        # Thêm các lớp ẩn
        for i, units in enumerate(self.hidden_layers):
            x = Dense(
                units,
                activation=self.activation,
                kernel_initializer=HeUniform(),
                name=f'hidden_{i}'
            )(x)

            # Thêm Batch Normalization nếu được chỉ định
            if self.kwargs.get('use_batch_norm', False):
                x = BatchNormalization()(x)

            # Thêm Dropout nếu được chỉ định
            dropout_rate = self.kwargs.get('dropout_rate', 0)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)

        # Lớp đầu ra
        if self.dueling and self.action_dim > 0:
            # Dueling DQN: tách thành lớp giá trị và lớp lợi thế
            value = Dense(1, name='value')(x)
            advantages = Dense(self.action_dim, name='advantages')(x)

            # Kết hợp value và advantages để tính Q-values
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
            def dueling_combine(inputs):
                value, advantages = inputs
                return value + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))

            output = tf.keras.layers.Lambda(dueling_combine, name='q_values')([value, advantages])
        else:
            # Mạng giá trị tiêu chuẩn
            if self.action_dim > 0:
                # Q-network: ước lượng Q(s,a) cho mỗi hành động
                output = Dense(self.action_dim, name='q_values')(x)
            else:
                # Value network: ước lượng V(s)
                output = Dense(1, name='value')(x)

        # Tạo model
        model = Model(inputs=state_input, outputs=output, name=self.name)

        # Biên dịch model
        model.compile(
            optimizer=self.optimizer,
            loss='mse'  # Mean Squared Error là hàm loss tiêu chuẩn cho value/Q network
        )

        return model

    def _build_cnn(self) -> Model:
        """
        Xây dựng mạng tích chập (CNN) cho dữ liệu không gian 2D hoặc cao hơn.

        Returns:
            Model Keras
        """
        # Kiểm tra kích thước đầu vào
        if len(self._get_input_shape()) < 2:
            self.logger.warning("CNN yêu cầu đầu vào ít nhất 2D, sử dụng MLP thay thế")
            return self._build_mlp()

        # Đầu vào là trạng thái
        state_input = Input(shape=self._get_input_shape(), name='state_input')

        # Các tham số CNN
        num_filters = self.kwargs.get('num_filters', [32, 64, 64])
        kernel_sizes = self.kwargs.get('kernel_sizes', [(8, 8), (4, 4), (3, 3)])
        strides = self.kwargs.get('strides', [(4, 4), (2, 2), (1, 1)])

        # Thêm các lớp tích chập
        x = state_input
        for i, (filters, kernel_size, stride) in enumerate(zip(num_filters, kernel_sizes, strides)):
            x = Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                activation=self.activation,
                padding='same',
                name=f'conv_{i}'
            )(x)

            # Thêm Batch Normalization nếu được chỉ định
            if self.kwargs.get('use_batch_norm', False):
                x = BatchNormalization()(x)

        # Làm phẳng để kết nối với các lớp fully connected
        x = Flatten()(x)

        # Thêm các lớp fully connected
        for i, units in enumerate(self.hidden_layers):
            x = Dense(
                units,
                activation=self.activation,
                name=f'hidden_{i}'
            )(x)

            # Thêm Dropout nếu được chỉ định
            dropout_rate = self.kwargs.get('dropout_rate', 0)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)

        # Lớp đầu ra (tương tự như trong _build_mlp)
        if self.dueling and self.action_dim > 0:
            value = Dense(1, name='value')(x)
            advantages = Dense(self.action_dim, name='advantages')(x)

            def dueling_combine(inputs):
                value, advantages = inputs
                return value + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))

            output = tf.keras.layers.Lambda(dueling_combine, name='q_values')([value, advantages])
        else:
            if self.action_dim > 0:
                output = Dense(self.action_dim, name='q_values')(x)
            else:
                output = Dense(1, name='value')(x)

        # Tạo model
        model = Model(inputs=state_input, outputs=output, name=self.name)

        # Biên dịch model
        model.compile(
            optimizer=self.optimizer,
            loss='mse'
        )

        return model

    def _build_rnn(self) -> Model:
        """
        Xây dựng mạng hồi quy (RNN) cho dữ liệu chuỗi thời gian.

        Returns:
            Model Keras
        """
        # Kiểm tra kích thước đầu vào
        if len(self._get_input_shape()) < 2:
            self.logger.warning("RNN yêu cầu đầu vào ít nhất 2D (chuỗi), sử dụng MLP thay thế")
            return self._build_mlp()

        # Đầu vào là trạng thái (chuỗi thời gian)
        state_input = Input(shape=self._get_input_shape(), name='state_input')

        # Các tham số RNN
        rnn_type = self.kwargs.get('rnn_type', 'lstm').lower()
        rnn_units = self.kwargs.get('rnn_units', [64, 64])
        bidirectional = self.kwargs.get('bidirectional', False)

        # Thêm các lớp RNN
        x = state_input
        for i, units in enumerate(rnn_units):
            return_sequences = i < len(rnn_units) - 1  # True cho tất cả trừ lớp cuối cùng

            if rnn_type == 'lstm':
                rnn_layer = LSTM(
                    units,
                    return_sequences=return_sequences,
                    name=f'lstm_{i}'
                )
            elif rnn_type == 'gru':
                rnn_layer = GRU(
                    units,
                    return_sequences=return_sequences,
                    name=f'gru_{i}'
                )
            else:
                self.logger.warning(f"Loại RNN {rnn_type} không được hỗ trợ, sử dụng LSTM")
                rnn_layer = LSTM(
                    units,
                    return_sequences=return_sequences,
                    name=f'lstm_{i}'
                )

            if bidirectional:
                rnn_layer = tf.keras.layers.Bidirectional(rnn_layer, name=f'bidirectional_{i}')

            x = rnn_layer(x)

            # Thêm Dropout nếu được chỉ định
            dropout_rate = self.kwargs.get('dropout_rate', 0)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)

        # Thêm các lớp fully connected nếu cần
        for i, units in enumerate(self.hidden_layers):
            x = Dense(
                units,
                activation=self.activation,
                name=f'hidden_{i}'
            )(x)

            # Thêm Dropout nếu được chỉ định
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)

        # Lớp đầu ra (tương tự như trong _build_mlp)
        if self.dueling and self.action_dim > 0:
            value = Dense(1, name='value')(x)
            advantages = Dense(self.action_dim, name='advantages')(x)

            def dueling_combine(inputs):
                value, advantages = inputs
                return value + (advantages - tf.reduce_mean(advantages, axis=1, keepdims=True))

            output = tf.keras.layers.Lambda(dueling_combine, name='q_values')([value, advantages])
        else:
            if self.action_dim > 0:
                output = Dense(self.action_dim, name='q_values')(x)
            else:
                output = Dense(1, name='value')(x)

        # Tạo model
        model = Model(inputs=state_input, outputs=output, name=self.name)

        # Biên dịch model
        model.compile(
            optimizer=self.optimizer,
            loss='mse'
        )

        return model

    def _get_input_shape(self) -> Tuple:
        """
        Lấy kích thước đầu vào phù hợp với tensorflow.

        Returns:
            Tuple kích thước đầu vào
        """
        if isinstance(self.state_dim, tuple):
            return self.state_dim
        elif isinstance(self.state_dim, int):
            return (self.state_dim,)
        else:
            self.logger.warning(f"Kiểu dữ liệu state_dim {type(self.state_dim)} không hợp lệ, sử dụng (1,)")
            return (1,)

    def predict(self, state: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Dự đoán giá trị cho trạng thái.

        Args:
            state: Trạng thái đầu vào, có thể là một hoặc nhiều trạng thái
            batch_size: Kích thước batch cho dự đoán (để tối ưu hiệu suất)

        Returns:
            Giá trị dự đoán
        """
        # Đảm bảo state có đúng kích thước
        if len(state.shape) == 1 and isinstance(self.state_dim, tuple) and len(self.state_dim) > 1:
            # Thêm chiều batch nếu cần
            state = np.expand_dims(state, axis=0)

        return self.model.predict(state, batch_size=batch_size)

    def train_on_batch(self, states: np.ndarray, targets: np.ndarray) -> float:
        """
        Huấn luyện mạng trên một batch.

        Args:
            states: Batch trạng thái đầu vào
            targets: Giá trị mục tiêu tương ứng

        Returns:
            Giá trị loss sau khi huấn luyện
        """
        return self.model.train_on_batch(states, targets)

    def fit(self, states: np.ndarray, targets: np.ndarray, batch_size: int = 32,
           epochs: int = 1, verbose: int = 0) -> Dict[str, List[float]]:
        """
        Huấn luyện mạng với nhiều batch.

        Args:
            states: Dữ liệu trạng thái
            targets: Giá trị mục tiêu
            batch_size: Kích thước batch
            epochs: Số lượng epoch
            verbose: Mức độ chi tiết (0: im lặng, 1: chi tiết, 2: một dòng mỗi epoch)

        Returns:
            Dict chứa lịch sử huấn luyện
        """
        history = self.model.fit(
            states, targets,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose
        )
        return history.history

    def save(self, path: str) -> None:
        """
        Lưu mô hình.

        Args:
            path: Đường dẫn lưu mô hình
        """
        self.model.save(path)
        self.logger.info(f"Đã lưu mô hình tại {path}")

    def load(self, path: str) -> bool:
        """
        Tải mô hình.

        Args:
            path: Đường dẫn tải mô hình

        Returns:
            True nếu tải thành công, False nếu không
        """
        if not os.path.exists(path):
            self.logger.warning(f"Không tìm thấy mô hình tại {path}")
            return False

        try:
            self.model = tf.keras.models.load_model(path)
            self.logger.info(f"Đã tải mô hình từ {path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải mô hình: {str(e)}")
            return False

    def get_weights(self) -> List[np.ndarray]:
        """
        Lấy trọng số của mạng.

        Returns:
            Danh sách các mảng trọng số
        """
        return self.model.get_weights()

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Đặt trọng số cho mạng.

        Args:
            weights: Danh sách các mảng trọng số
        """
        self.model.set_weights(weights)

    def summary(self) -> None:
        """
        Hiển thị tóm tắt mô hình.
        """
        self.model.summary()