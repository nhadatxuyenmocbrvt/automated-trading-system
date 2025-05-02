"""
Mạng chính sách (Policy Network).
File này định nghĩa các mạng neural để biểu diễn chính sách của agent,
được sử dụng trong các thuật toán Reinforcement Learning như A2C, PPO, v.v.
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

class PolicyNetwork:
    """
    Mạng neural để biểu diễn chính sách của agent.
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
        name: str = 'policy_network',
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Khởi tạo mạng chính sách.
        
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
            name: Tên của mạng
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("policy_network")
        
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
        self.model, self.policy_out, self.action_output = self._build_model()
        
        self.logger.info(
            f"Đã khởi tạo {self.__class__.__name__} với state_dim={state_dim}, "
            f"action_dim={action_dim}, network_type={network_type}, action_type={action_type}"
        )
    
    def _build_model(self) -> Tuple[Model, tf.Tensor, tf.Tensor]:
        """
        Xây dựng mạng neural.
        
        Returns:
            Tuple gồm (model, policy_outputs, action_outputs)
            - model: Model Keras
            - policy_outputs: Tensor đầu ra của policy (logits hoặc means+std)
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
    
    def _build_mlp(self) -> Tuple[Model, tf.Tensor, tf.Tensor]:
        """
        Xây dựng mạng perceptron đa lớp (MLP).
        
        Returns:
            Tuple gồm (model, policy_outputs, action_outputs)
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
                kernel_initializer=GlorotUniform(),
                name=f'hidden_{i}'
            )(x)
            
            # Thêm Batch Normalization nếu được chỉ định
            if self.kwargs.get('use_batch_norm', False):
                x = BatchNormalization()(x)
            
            # Thêm Dropout nếu được chỉ định
            dropout_rate = self.kwargs.get('dropout_rate', 0)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)
        
        # Lớp đầu ra tùy thuộc vào loại hành động
        if self.action_type == 'discrete':
            # Đầu ra là logits cho mỗi hành động rời rạc
            logits = Dense(self.action_dim, name='logits')(x)
            
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
            policy_out = logits
            
        else:  # continuous
            # Đầu ra là mean và stddev cho continuous action
            mean = Dense(self.action_dim, name='mean')(x)
            
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
                log_std = Dense(self.action_dim, name='log_std')(x)
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
            policy_out = [mean, std]
        
        # Tạo model
        model = Model(inputs=state_input, outputs=[policy_out, action_output], name=self.name)
        
        # Không cần compile vì sẽ sử dụng custom loss trong train_on_batch
        
        return model, policy_out, action_output
    
    def _build_cnn(self) -> Tuple[Model, tf.Tensor, tf.Tensor]:
        """
        Xây dựng mạng tích chập (CNN) cho dữ liệu không gian 2D hoặc cao hơn.
        
        Returns:
            Tuple gồm (model, policy_outputs, action_outputs)
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
        
        # Phần còn lại giống như _build_mlp
        if self.action_type == 'discrete':
            logits = Dense(self.action_dim, name='logits')(x)
            action_probs = Activation('softmax', name='action_probs')(logits)
            
            def sample_action(probs):
                dist = tfp.distributions.Categorical(probs=probs)
                action = dist.sample()
                return action
            
            action_output = Lambda(sample_action, name='sampled_action')(action_probs)
            policy_out = logits
            
        else:  # continuous
            mean = Dense(self.action_dim, name='mean')(x)
            
            if self.kwargs.get('fixed_std', False):
                log_std = tf.Variable(
                    initial_value=np.zeros(self.action_dim, dtype=np.float32) - 0.5,
                    trainable=True,
                    name='log_std'
                )
                std = tf.exp(log_std)
            else:
                log_std = Dense(self.action_dim, name='log_std')(x)
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
            policy_out = [mean, std]
        
        # Tạo model
        model = Model(inputs=state_input, outputs=[policy_out, action_output], name=self.name)
        
        return model, policy_out, action_output
    
    def _build_rnn(self) -> Tuple[Model, tf.Tensor, tf.Tensor]:
        """
        Xây dựng mạng hồi quy (RNN) cho dữ liệu chuỗi thời gian.
        
        Returns:
            Tuple gồm (model, policy_outputs, action_outputs)
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
        
        # Phần còn lại giống như _build_mlp
        if self.action_type == 'discrete':
            logits = Dense(self.action_dim, name='logits')(x)
            action_probs = Activation('softmax', name='action_probs')(logits)
            
            def sample_action(probs):
                dist = tfp.distributions.Categorical(probs=probs)
                action = dist.sample()
                return action
            
            action_output = Lambda(sample_action, name='sampled_action')(action_probs)
            policy_out = logits
            
        else:  # continuous
            mean = Dense(self.action_dim, name='mean')(x)
            
            if self.kwargs.get('fixed_std', False):
                log_std = tf.Variable(
                    initial_value=np.zeros(self.action_dim, dtype=np.float32) - 0.5,
                    trainable=True,
                    name='log_std'
                )
                std = tf.exp(log_std)
            else:
                log_std = Dense(self.action_dim, name='log_std')(x)
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
            policy_out = [mean, std]
        
        # Tạo model
        model = Model(inputs=state_input, outputs=[policy_out, action_output], name=self.name)
        
        return model, policy_out, action_output
    
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
    
    def predict(self, state: np.ndarray, batch_size: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Dự đoán hành động cho trạng thái.
        
        Args:
            state: Trạng thái đầu vào, có thể là một hoặc nhiều trạng thái
            batch_size: Kích thước batch cho dự đoán (để tối ưu hiệu suất)
            
        Returns:
            Hành động dự đoán (và thông tin chính sách nếu cần)
        """
        # Đảm bảo state có đúng kích thước
        if len(state.shape) == 1 and isinstance(self.state_dim, tuple) and len(self.state_dim) > 1:
            # Thêm chiều batch nếu cần
            state = np.expand_dims(state, axis=0)
        
        # Dự đoán hành động
        _, action = self.model.predict(state, batch_size=batch_size)
        return action
    
    def predict_policy(self, state: np.ndarray, batch_size: Optional[int] = None) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Dự đoán thông tin chính sách (logits hoặc mean/std) cho trạng thái.
        
        Args:
            state: Trạng thái đầu vào, có thể là một hoặc nhiều trạng thái
            batch_size: Kích thước batch cho dự đoán (để tối ưu hiệu suất)
            
        Returns:
            Thông tin chính sách (logits cho discrete, [mean, std] cho continuous)
        """
        # Đảm bảo state có đúng kích thước
        if len(state.shape) == 1 and isinstance(self.state_dim, tuple) and len(self.state_dim) > 1:
            # Thêm chiều batch nếu cần
            state = np.expand_dims(state, axis=0)
        
        # Dự đoán chính sách
        policy, _ = self.model.predict(state, batch_size=batch_size)
        return policy
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """
        Lấy xác suất hành động cho trạng thái (chỉ cho discrete actions).
        
        Args:
            state: Trạng thái đầu vào
            
        Returns:
            Xác suất hành động
        """
        if self.action_type != 'discrete':
            self.logger.warning("get_action_probs chỉ hỗ trợ discrete actions")
            return np.ones(self.action_dim) / self.action_dim
        
        logits = self.predict_policy(state)
        return tf.nn.softmax(logits).numpy()
    
    def train_on_batch(self, states: np.ndarray, actions: np.ndarray, advantages: np.ndarray,
                      old_policies: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                      clip_ratio: float = 0.2) -> Dict[str, float]:
        """
        Huấn luyện mạng chính sách trên một batch.
        
        Args:
            states: Batch trạng thái đầu vào
            actions: Batch hành động đã thực hiện
            advantages: Batch lợi thế/advantage của hành động
            old_policies: Chính sách cũ (cho PPO)
            clip_ratio: Tỷ lệ cắt (cho PPO)
            
        Returns:
            Dict chứa thông tin về quá trình huấn luyện (loss, ...)
        """
        # Huấn luyện với GradientTape
        with tf.GradientTape() as tape:
            loss, info = self._compute_loss(states, actions, advantages, old_policies, clip_ratio)
        
        # Tính toán và áp dụng gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Thêm giá trị loss vào info
        info['loss'] = loss.numpy()
        
        return info
    
    def _compute_loss(self, states: np.ndarray, actions: np.ndarray, advantages: np.ndarray,
                    old_policies: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                    clip_ratio: float = 0.2) -> Tuple[tf.Tensor, Dict[str, float]]:
        """
        Tính toán hàm loss cho mạng chính sách.
        
        Args:
            states: Batch trạng thái đầu vào
            actions: Batch hành động đã thực hiện
            advantages: Batch lợi thế/advantage của hành động
            old_policies: Chính sách cũ (cho PPO)
            clip_ratio: Tỷ lệ cắt (cho PPO)
            
        Returns:
            Tuple (loss, info) với loss là giá trị loss và info là dict chứa thông tin bổ sung
        """
        # Chuyển đổi thành tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32 if self.action_type == 'discrete' else tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        
        # Dự đoán chính sách mới
        policy_output, _ = self.model(states, training=True)
        
        # Tính toán loss dựa trên loại hành động
        if self.action_type == 'discrete':
            # Discrete actions
            logits = policy_output
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
                
                loss = policy_loss + entropy_loss
                
                # Thông tin bổ sung
                info = {
                    'policy_loss': policy_loss.numpy(),
                    'entropy_loss': entropy_loss.numpy(),
                    'mean_ratio': tf.reduce_mean(ratio).numpy(),
                    'mean_clipped_ratio': tf.reduce_mean(tf.cast(ratio > 1.0 + clip_ratio, tf.float32) + 
                                                       tf.cast(ratio < 1.0 - clip_ratio, tf.float32)).numpy()
                }
            else:
                # Vanilla Policy Gradient loss
                policy_loss = -tf.reduce_mean(log_probs * advantages)
                entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
                
                loss = policy_loss + entropy_loss
                
                # Thông tin bổ sung
                info = {
                    'policy_loss': policy_loss.numpy(),
                    'entropy_loss': entropy_loss.numpy(),
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
                
                loss = policy_loss + entropy_loss
                
                # Thông tin bổ sung
                info = {
                    'policy_loss': policy_loss.numpy(),
                    'entropy_loss': entropy_loss.numpy(),
                    'mean_ratio': tf.reduce_mean(ratio).numpy(),
                    'mean_clipped_ratio': tf.reduce_mean(tf.cast(ratio > 1.0 + clip_ratio, tf.float32) + 
                                                       tf.cast(ratio < 1.0 - clip_ratio, tf.float32)).numpy()
                }
            else:
                # Vanilla Policy Gradient loss
                policy_loss = -tf.reduce_mean(log_probs * advantages)
                entropy_loss = -tf.reduce_mean(entropy) * self.entropy_coef
                
                loss = policy_loss + entropy_loss
                
                # Thông tin bổ sung
                info = {
                    'policy_loss': policy_loss.numpy(),
                    'entropy_loss': entropy_loss.numpy(),
                    'mean_log_prob': tf.reduce_mean(log_probs).numpy(),
                    'mean_std': tf.reduce_mean(std).numpy()
                }
        
        return loss, info
    
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
            self.policy_out = self.model.outputs[0]
            self.action_output = self.model.outputs[1]
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