"""
Mạng chia sẻ (Shared Network).
File này định nghĩa một mạng neural dùng chung cho cả policy và value networks,
được sử dụng trong các thuật toán Reinforcement Learning như A2C, PPO.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, LSTM, GRU, Dropout, BatchNormalization
from tensorflow.keras.layers import Lambda, Concatenate, Activation, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import HeUniform, GlorotUniform
import tensorflow_probability as tfp

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config

class SharedNetwork:
    """
    Mạng neural chia sẻ được sử dụng cho cả policy và value networks.
    Giúp giảm số lượng tham số và tăng hiệu quả học.
    """
    
    def __init__(
        self,
        state_dim: Union[int, Tuple[int, ...]],
        action_dim: int,
        hidden_layers: List[int] = [64, 64],
        activation: str = 'relu',
        learning_rate: float = 0.001,
        network_type: str = 'mlp',
        action_type: str = 'discrete',  # 'discrete' hoặc 'continuous'
        action_bound: Optional[Tuple[float, float]] = None,  # (min, max) cho continuous actions
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,  # Hệ số cho value loss
        name: str = 'shared_network',
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo mạng chia sẻ.
        
        Args:
            state_dim: Kích thước không gian trạng thái
            action_dim: Kích thước không gian hành động
            hidden_layers: Danh sách số lượng neuron trong các lớp ẩn
            activation: Hàm kích hoạt
            learning_rate: Tốc độ học
            network_type: Loại mạng neural ('mlp', 'cnn', 'rnn')
            action_type: Loại hành động ('discrete' hoặc 'continuous')
            action_bound: Giới hạn hành động cho continuous actions (min, max)
            entropy_coef: Hệ số entropy trong hàm loss để khuyến khích khám phá
            value_coef: Hệ số cho value loss
            name: Tên của mạng
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("shared_network")
        
        # Lấy cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thiết lập các thuộc tính
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.network_type = network_type.lower()
        self.action_type = action_type.lower()
        self.action_bound = action_bound
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.name = name
        self.kwargs = kwargs
        
        # Kiểm tra và xử lý action_bound
        if self.action_type == 'continuous' and self.action_bound is None:
            self.logger.warning("action_bound không được cung cấp cho continuous actions, sử dụng (-1, 1)")
            self.action_bound = (-1.0, 1.0)
        
        # Thiết lập optimizer
        if kwargs.get('optimizer', '').lower() == 'rmsprop':
            self.optimizer = RMSprop(learning_rate=learning_rate)
        else:
            self.optimizer = Adam(learning_rate=learning_rate)
        
        # Xây dựng mạng neural
        self.model, self.shared_layers, self.policy_output, self.value_output, self.action_output = self._build_model()
        
        self.logger.info(
            f"Đã khởi tạo {self.__class__.__name__} với state_dim={state_dim}, "
            f"action_dim={action_dim}, network_type={network_type}, action_type={action_type}"
        )
    
    def _build_model(self) -> Tuple[Model, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Xây dựng mạng neural chia sẻ.
        
        Returns:
            Tuple gồm (model, shared_layers, policy_outputs, value_outputs, action_outputs)
            - model: Model Keras
            - shared_layers: Các lớp chia sẻ
            - policy_outputs: Tensor đầu ra của policy (logits hoặc means+std)
            - value_outputs: Tensor đầu ra giá trị V(s)
            - action_outputs: Tensor đầu ra là hành động được lấy mẫu
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
    
    def _build_mlp(self) -> Tuple[Model, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Xây dựng mạng perceptron đa lớp (MLP) chia sẻ.
        
        Returns:
            Tuple gồm (model, shared_layers, policy_outputs, value_outputs, action_outputs)
        """
        # Đầu vào là trạng thái
        state_input = Input(shape=self._get_input_shape(), name='state_input')
        x = state_input
        
        # Nếu đầu vào là mảng 2D hoặc cao hơn, làm phẳng
        if len(self._get_input_shape()) > 1:
            x = Flatten()(x)
        
        # Thêm các lớp ẩn chia sẻ
        for i, units in enumerate(self.hidden_layers):
            x = Dense(
                units, 
                activation=self.activation,
                kernel_initializer=GlorotUniform(),
                name=f'shared_hidden_{i}'
            )(x)
            
            # Thêm Batch Normalization nếu được chỉ định
            if self.kwargs.get('use_batch_norm', False):
                x = BatchNormalization()(x)
            
            # Thêm Dropout nếu được chỉ định
            dropout_rate = self.kwargs.get('dropout_rate', 0)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)
        
        # Lưu lại các lớp chia sẻ
        shared_features = x
        
        # === Phần Policy Network ===
        if self.action_type == 'discrete':
            # Đầu ra là logits cho mỗi hành động rời rạc
            policy_head = Dense(64, activation=self.activation, name='policy_head')(shared_features)
            logits = Dense(self.action_dim, name='logits')(policy_head)
            
            # Chuyển logits thành xác suất (softmax)
            action_probs = Activation('softmax', name='action_probs')(logits)
            
            # Tạo hàm lấy mẫu hành động
            def sample_action(probs):
                dist = tfp.distributions.Categorical(probs=probs)
                action = dist.sample()
                return action
            
            # Lấy mẫu hành động từ phân phối
            action_output = Lambda(sample_action, name='sampled_action')(action_probs)
            
            # Đầu ra chính sách là logits
            policy_output = logits
            
        else:  # continuous
            # Đầu ra là mean và stddev cho continuous action
            policy_head = Dense(64, activation=self.activation, name='policy_head')(shared_features)
            mean = Dense(self.action_dim, name='mean')(policy_head)
            
            # Có hai cách để xử lý stddev
            if self.kwargs.get('fixed_std', False):
                # Sử dụng stddev cố định
                log_std = tf.Variable(
                    initial_value=np.zeros(self.action_dim, dtype=np.float32) - 0.5,
                    trainable=True,
                    name='log_std'
                )
                std = tf.exp(log_std)
            else:
                # Học stddev
                # Chú ý: log_std để đảm bảo std luôn dương
                log_std = Dense(self.action_dim, name='log_std')(policy_head)
                # Giới hạn log_std để tránh std quá lớn hoặc quá nhỏ
                log_std = Lambda(lambda x: tf.clip_by_value(x, -20, 2), name='clip_log_std')(log_std)
                std = Lambda(lambda x: tf.exp(x), name='std')(log_std)
            
            # Tạo hàm lấy mẫu hành động từ phân phối normal
            def sample_continuous_action(params):
                mean, std = params
                dist = tfp.distributions.Normal(loc=mean, scale=std)
                action = dist.sample()
                # Scale action trong phạm vi action_bound
                low, high = self.action_bound
                action = low + 0.5 * (action + 1.0) * (high - low)
                # Clip để đảm bảo nằm trong giới hạn
                action = tf.clip_by_value(action, low, high)
                return action
            
            # Lấy mẫu hành động
            action_output = Lambda(sample_continuous_action, name='sampled_action')([mean, std])
            
            # Đầu ra chính sách là mean và std
            policy_output = [mean, std]
        
        # === Phần Value Network ===
        value_head = Dense(64, activation=self.activation, name='value_head')(shared_features)
        value_output = Dense(1, name='value')(value_head)
        
        # Tạo model
        model = Model(
            inputs=state_input, 
            outputs=[policy_output, value_output, action_output], 
            name=self.name
        )
        
        return model, shared_features, policy_output, value_output, action_output
    
    def _build_cnn(self) -> Tuple[Model, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Xây dựng mạng tích chập (CNN) chia sẻ.
        
        Returns:
            Tuple gồm (model, shared_layers, policy_outputs, value_outputs, action_outputs)
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
                name=f'shared_hidden_{i}'
            )(x)
            
            # Thêm Dropout nếu được chỉ định
            dropout_rate = self.kwargs.get('dropout_rate', 0)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)
        
        # Lưu lại các lớp chia sẻ
        shared_features = x
        
        # === Phần Policy Network ===
        if self.action_type == 'discrete':
            policy_head = Dense(64, activation=self.activation, name='policy_head')(shared_features)
            logits = Dense(self.action_dim, name='logits')(policy_head)
            action_probs = Activation('softmax', name='action_probs')(logits)
            
            def sample_action(probs):
                dist = tfp.distributions.Categorical(probs=probs)
                action = dist.sample()
                return action
            
            action_output = Lambda(sample_action, name='sampled_action')(action_probs)
            policy_output = logits
            
        else:  # continuous
            policy_head = Dense(64, activation=self.activation, name='policy_head')(shared_features)
            mean = Dense(self.action_dim, name='mean')(policy_head)
            
            if self.kwargs.get('fixed_std', False):
                log_std = tf.Variable(
                    initial_value=np.zeros(self.action_dim, dtype=np.float32) - 0.5,
                    trainable=True,
                    name='log_std'
                )
                std = tf.exp(log_std)
            else:
                log_std = Dense(self.action_dim, name='log_std')(policy_head)
                log_std = Lambda(lambda x: tf.clip_by_value(x, -20, 2), name='clip_log_std')(log_std)
                std = Lambda(lambda x: tf.exp(x), name='std')(log_std)
            
            def sample_continuous_action(params):
                mean, std = params
                dist = tfp.distributions.Normal(loc=mean, scale=std)
                action = dist.sample()
                low, high = self.action_bound
                action = low + 0.5 * (action + 1.0) * (high - low)
                action = tf.clip_by_value(action, low, high)
                return action
            
            action_output = Lambda(sample_continuous_action, name='sampled_action')([mean, std])
            policy_output = [mean, std]
        
        # === Phần Value Network ===
        value_head = Dense(64, activation=self.activation, name='value_head')(shared_features)
        value_output = Dense(1, name='value')(value_head)
        
        # Tạo model
        model = Model(
            inputs=state_input, 
            outputs=[policy_output, value_output, action_output], 
            name=self.name
        )
        
        return model, shared_features, policy_output, value_output, action_output
    
    def _build_rnn(self) -> Tuple[Model, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Xây dựng mạng hồi quy (RNN) chia sẻ cho dữ liệu chuỗi thời gian.
        
        Returns:
            Tuple gồm (model, shared_layers, policy_outputs, value_outputs, action_outputs)
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
                name=f'shared_hidden_{i}'
            )(x)
            
            # Thêm Dropout nếu được chỉ định
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)
        
        # Lưu lại các lớp chia sẻ
        shared_features = x
        
        # === Phần Policy Network ===
        if self.action_type == 'discrete':
            policy_head = Dense(64, activation=self.activation, name='policy_head')(shared_features)
            logits = Dense(self.action_dim, name='logits')(policy_head)
            action_probs = Activation('softmax', name='action_probs')(logits)
            
            def sample_action(probs):
                dist = tfp.distributions.Categorical(probs=probs)
                action = dist.sample()
                return action
            
            action_output = Lambda(sample_action, name='sampled_action')(action_probs)
            policy_output = logits
            
        else:  # continuous
            policy_head = Dense(64, activation=self.activation, name='policy_head')(shared_features)
            mean = Dense(self.action_dim, name='mean')(policy_head)
            
            if self.kwargs.get('fixed_std', False):
                log_std = tf.Variable(
                    initial_value=np.zeros(self.action_dim, dtype=np.float32) - 0.5,
                    trainable=True,
                    name='log_std'
                )
                std = tf.exp(log_std)
            else:
                log_std = Dense(self.action_dim, name='log_std')(policy_head)
                log_std = Lambda(lambda x: tf.clip_by_value(x, -20, 2), name='clip_log_std')(log_std)
                std = Lambda(lambda x: tf.exp(x), name='std')(log_std)
            
            def sample_continuous_action(params):
                mean, std = params
                dist = tfp.distributions.Normal(loc=mean, scale=std)
                action = dist.sample()
                low, high = self.action_bound
                action = low + 0.5 * (action + 1.0) * (high - low)
                action = tf.clip_by_value(action, low, high)
                return action
            
            action_output = Lambda(sample_continuous_action, name='sampled_action')([mean, std])
            policy_output = [mean, std]
        
        # === Phần Value Network ===
        value_head = Dense(64, activation=self.activation, name='value_head')(shared_features)
        value_output = Dense(1, name='value')(value_head)
        
        # Tạo model
        model = Model(
            inputs=state_input, 
            outputs=[policy_output, value_output, action_output], 
            name=self.name
        )
        
        return model, shared_features, policy_output, value_output, action_output
    
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
    
    def predict(self, state: np.ndarray, batch_size: Optional[int] = None) -> Tuple[Union[np.ndarray, List[np.ndarray]], np.ndarray, np.ndarray]:
        """
        Dự đoán policy, value và action cho trạng thái.
        
        Args:
            state: Trạng thái đầu vào, có thể là một hoặc nhiều trạng thái
            batch_size: Kích thước batch cho dự đoán (để tối ưu hiệu suất)
            
        Returns:
            Tuple gồm (policy_output, value_output, action_output)
        """
        # Đảm bảo state có đúng kích thước
        if len(state.shape) == 1 and isinstance(self.state_dim, tuple) and len(self.state_dim) > 1:
            # Thêm chiều batch nếu cần
            state = np.expand_dims(state, axis=0)
        
        # Dự đoán
        policy, value, action = self.model.predict(state, batch_size=batch_size)

        # Chuẩn hóa kích thước của policy output
        if self.action_type == 'discrete':
            # Đảm bảo policy có shape [batch_size, action_dim]
            if isinstance(policy, np.ndarray):
                if len(policy.shape) == 3:
                    policy = np.reshape(policy, (policy.shape[0], policy.shape[2]))
                elif len(policy.shape) == 2 and policy.shape[1] == 1:
                    # Trường hợp đặc biệt: [batch_size, 1]
                    # Kiểm tra và mở rộng nếu cần
                    if hasattr(self, 'action_dim') and self.action_dim > 1:
                        # Nếu không thể reshaping, tạo mảng mặc định
                        policy_shape = (policy.shape[0], self.action_dim)
                        temp_policy = np.zeros(policy_shape)
                        temp_policy[:, 0] = 1.0  # Thiên vị hành động 0
                        policy = temp_policy
        
        # Chuẩn hóa kích thước của value output
        if isinstance(value, np.ndarray) and len(value.shape) > 1 and value.shape[1] == 1:
            value = value.flatten()
        
        return policy, value, action
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Lấy hành động cho trạng thái.
        
        Args:
            state: Trạng thái đầu vào
            
        Returns:
            Hành động dự đoán
        """
        _, _, action = self.predict(state)
        return action
    
    def train_on_batch(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        advantages: np.ndarray,
        returns: np.ndarray,
        old_policies: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        clip_ratio: float = 0.2
    ) -> Dict[str, float]:
        """
        Huấn luyện mạng trên một batch.
        
        Args:
            states: Batch trạng thái đầu vào
            actions: Batch hành động đã thực hiện
            advantages: Batch lợi thế/advantage của hành động
            returns: Batch giá trị target cho value function
            old_policies: Chính sách cũ (cho PPO)
            clip_ratio: Tỷ lệ cắt (cho PPO)
            
        Returns:
            Dict chứa thông tin về quá trình huấn luyện (loss, ...)
        """
        # Huấn luyện với GradientTape
        with tf.GradientTape() as tape:
            policy_loss, value_loss, entropy_loss, total_loss, info = self._compute_loss(
                states, actions, advantages, returns, old_policies, clip_ratio
            )
        
        # Tính toán và áp dụng gradients
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Thêm giá trị loss vào info
        info['policy_loss'] = policy_loss.numpy()
        info['value_loss'] = value_loss.numpy()
        info['entropy_loss'] = entropy_loss.numpy()
        info['total_loss'] = total_loss.numpy()
        
        return info
    
    def _compute_loss(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        advantages: np.ndarray,
        returns: np.ndarray,
        old_policies: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        clip_ratio: float = 0.2
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Dict[str, float]]:
        """
        Tính toán hàm loss cho mạng.
        
        Args:
            states: Batch trạng thái đầu vào
            actions: Batch hành động đã thực hiện
            advantages: Batch lợi thế/advantage của hành động
            returns: Batch giá trị target cho value function
            old_policies: Chính sách cũ (cho PPO)
            clip_ratio: Tỷ lệ cắt (cho PPO)
            
        Returns:
            Tuple (policy_loss, value_loss, entropy_loss, total_loss, info)
        """
        # Chuyển đổi thành tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32 if self.action_type == 'discrete' else tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # Dự đoán policy và value mới
        policy_output, value_output, _ = self.model(states, training=True)
        value_output = tf.squeeze(value_output, axis=-1)  # Đảm bảo shape khớp với returns
        
        # Tính toán value loss
        value_loss = tf.reduce_mean(tf.square(returns - value_output))
        
        # Tính toán policy loss dựa trên loại hành động
        if self.action_type == 'discrete':
            # Chuyển logits thành probabilities
            logits = policy_output
            
            # Đảm bảo logits có shape đúng [batch_size, action_dim]
            if tf.shape(logits).shape[0] > 2:  # Kiểm tra số chiều > 2
                # Reshape từ [batch_size, 1, action_dim] thành [batch_size, action_dim]
                logits = tf.reshape(logits, [tf.shape(logits)[0], tf.shape(logits)[-1]])
            
            action_probs = tf.nn.softmax(logits)
            
            # Tính log probabilities cho hành động đã chọn
            indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            selected_probs = tf.gather_nd(action_probs, indices)
            log_probs = tf.math.log(selected_probs + 1e-10)
            
            # Tính entropy để khuyến khích khám phá
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
            
            if old_policies is not None:
                # PPO style loss với clipping
                old_action_probs = tf.nn.softmax(old_policies)
                old_selected_probs = tf.gather_nd(old_action_probs, indices)
                old_log_probs = tf.math.log(old_selected_probs + 1e-10)
                
                ratio = tf.exp(log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                
                surrogate1 = ratio * advantages
                surrogate2 = clipped_ratio * advantages
                
                policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
                
                # Thông tin bổ sung
                info = {
                    'mean_ratio': tf.reduce_mean(ratio).numpy(),
                    'mean_clipped_ratio': tf.reduce_mean(tf.cast(ratio > 1.0 + clip_ratio, tf.float32) + 
                                                       tf.cast(ratio < 1.0 - clip_ratio, tf.float32)).numpy()
                }
            else:
                # A2C style loss
                policy_loss = -tf.reduce_mean(log_probs * advantages)
                entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
                
                # Thông tin bổ sung
                info = {
                    'mean_log_prob': tf.reduce_mean(log_probs).numpy()
                }
        else:
            # Continuous actions
            mean, std = policy_output
            
            # Tạo phân phối normal
            normal_dist = tfp.distributions.Normal(loc=mean, scale=std)
            
            # Normalize actions về phạm vi [-1, 1] trước khi tính log_prob
            low, high = self.action_bound
            normalized_actions = 2.0 * (actions - low) / (high - low) - 1.0
            
            # Tính log probabilities
            log_probs = normal_dist.log_prob(normalized_actions)
            log_probs = tf.reduce_sum(log_probs, axis=1)  # Sum across action dimensions
            
            # Tính entropy
            entropy = normal_dist.entropy()
            entropy = tf.reduce_sum(entropy, axis=1)  # Sum across action dimensions
            
            if old_policies is not None:
                # PPO style loss với clipping
                old_mean, old_std = old_policies
                old_normal_dist = tfp.distributions.Normal(loc=old_mean, scale=old_std)
                old_log_probs = old_normal_dist.log_prob(normalized_actions)
                old_log_probs = tf.reduce_sum(old_log_probs, axis=1)
                
                ratio = tf.exp(log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                
                surrogate1 = ratio * advantages
                surrogate2 = clipped_ratio * advantages
                
                policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
                
                # Thông tin bổ sung
                info = {
                    'mean_ratio': tf.reduce_mean(ratio).numpy(),
                    'mean_clipped_ratio': tf.reduce_mean(tf.cast(ratio > 1.0 + clip_ratio, tf.float32) + 
                                                       tf.cast(ratio < 1.0 - clip_ratio, tf.float32)).numpy()
                }
            else:
                # A2C style loss
                policy_loss = -tf.reduce_mean(log_probs * advantages)
                entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
                
                # Thông tin bổ sung
                info = {
                    'mean_log_prob': tf.reduce_mean(log_probs).numpy(),
                    'mean_std': tf.reduce_mean(std).numpy()
                }
        
        # Tính tổng loss
        total_loss = policy_loss + value_loss * self.value_coef + entropy_loss
        
        return policy_loss, value_loss, entropy_loss, total_loss, info
    
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
            self.model = tf.keras.models.load_model(path, custom_objects={'tf': tf, 'tfp': tfp})
            
            # Cập nhật các thuộc tính của model
            # Giả định rằng thứ tự outputs không thay đổi: policy, value, action
            self.policy_output = self.model.outputs[0]
            self.value_output = self.model.outputs[1]
            self.action_output = self.model.outputs[2]
            
            # Cập nhật shared_layers (giả định rằng layer cuối cùng trước khi tách ra là shared layer)
            for layer in self.model.layers:
                if 'shared_hidden' in layer.name:
                    self.shared_layers = layer.output
                    break
            
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