import os

# Danh sách thư mục cần tạo
directories = [
    "config/utils",
    "data_collectors/exchange_api", "data_collectors/market_data", "data_collectors/news_collector",
    "data_processors/cleaners", "data_processors/feature_engineering",
    "environments/trading_gym", "environments/reward_functions", "environments/simulators",
    "models/agents", "models/networks", "models/training_pipeline",
    "risk_management/risk_profiles",
    "backtesting/evaluation", "backtesting/visualization",
    "deployment/exchange_api",
    "logs/metrics", "logs/tensorboard",
    "streamlit_dashboard/pages", "streamlit_dashboard/charts", "streamlit_dashboard/components",
    "agent_manager/strategy_queue", "agent_manager/self_improvement",
    "real_time_inference/system_monitor", "real_time_inference/notifiers", 
    "real_time_inference/scheduler", "real_time_inference/auto_restart",
    "retraining",
    "automation/metrics", "automation/strategy_queue"
]

# Tạo thư mục
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# Danh sách các file cần tạo
files = [
    "main.py",
    "README.md",
    "requirements.txt",
    ".gitignore",
    "config/system_config.py",
    "config/logging_config.py",
    "config/security_config.py",
    "config/env.py",
    "config/constants.py",
    "config/utils/encryption.py",
    "config/utils/validators.py"
]

# Tạo file
for file in files:
    with open(file, 'w', encoding='utf-8') as f:  # Thêm encoding='utf-8'
        if file == "README.md":
            f.write("# Automated Trading System\n\nHệ thống giao dịch tự động sử dụng Reinforcement Learning.")
    print(f"Created file: {file}")

print("Cấu trúc thư mục đã được tạo thành công!")