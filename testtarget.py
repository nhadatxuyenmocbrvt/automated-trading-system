from environments.trading_gym.trading_env import TradingEnv
import pandas as pd
import numpy as np

# Tạo dữ liệu mẫu
data = pd.DataFrame({
    'open': np.random.random(1000) * 100,
    'high': np.random.random(1000) * 100 + 5,
    'low': np.random.random(1000) * 100 - 5,
    'close': np.random.random(1000) * 100,
    'volume': np.random.random(1000) * 1000
})

# Đảm bảo high >= open, close >= low
for i in range(len(data)):
    data.loc[i, 'high'] = max(data.loc[i, ['open', 'high', 'close']].values)
    data.loc[i, 'low'] = min(data.loc[i, ['open', 'low', 'close']].values)

# Khởi tạo môi trường với target
env = TradingEnv(
    data=data,
    generate_target_labels=True,
    target_type="price_movement",
    target_lookforward=10,
    target_threshold=0.01,
    include_target=True
)

# Reset môi trường
obs = env.reset()

# Kiểm tra xem obs có chứa target không
print(f"Observation shape: {obs.shape}")
print(f"Có target trong data: {'target' in env.data.columns}")

# Thử một số bước
for _ in range(5):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        break