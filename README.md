# Automated Trading System

## Tổng quan
Hệ thống giao dịch tự động sử dụng Reinforcement Learning để tối ưu hóa chiến lược giao dịch trên thị trường tiền điện tử.

## Mô tả
Dự án này xây dựng một hệ thống giao dịch hoàn chỉnh bao gồm:
- Thu thập dữ liệu từ các sàn giao dịch
- Xử lý và tạo đặc trưng từ dữ liệu thị trường
- Huấn luyện các agent sử dụng Reinforcement Learning
- Quản lý rủi ro thông minh
- Backtest và đánh giá chiến lược
- Dashboard theo dõi và phân tích
- Triển khai giao dịch thời gian thực

## Cấu trúc dự án
```
automated-trading-system/
│
├── config/                                # Cấu hình hệ thống & bảo mật
│   ├── system_config.py                   # Cấu hình chung hệ thống
│   ├── logging_config.py                  # Cấu hình logging
│   ├── security_config.py                 # Cấu hình bảo mật
│   ├── env.py                             # Quản lý biến môi trường
│   ├── constants.py                       # Hằng số hệ thống
│   └── utils/                             # Tiện ích chung
│       ├── encryption.py                  # Mã hóa dữ liệu
│       └── validators.py                  # Kiểm tra tính hợp lệ
│
├── data_collectors/                       # Thu thập dữ liệu
│   ├── exchange_api/                      # Kết nối API sàn giao dịch
│   │   ├── binance_connector.py           # Kết nối Binance
│   │   ├── bybit_connector.py             # Kết nối ByBit
│   │   └── generic_connector.py           # Lớp kết nối chung
│   ├── market_data/
│   │   ├── historical_data_collector.py   # Thu thập dữ liệu lịch sử
│   │   ├── realtime_data_stream.py        # Dữ liệu thời gian thực
│   │   └── orderbook_collector.py         # Thu thập dữ liệu sổ lệnh
│   └── news_collector/
│       ├── crypto_news_scraper.py         # Thu thập tin tức crypto
│       └── sentiment_collector.py         # Thu thập dữ liệu tâm lý thị trường
│
├── data_processors/                       # Làm sạch & tạo feature
│   ├── cleaners/
│   │   ├── data_cleaner.py                # Làm sạch dữ liệu
│   │   ├── outlier_detector.py            # Phát hiện ngoại lệ
│   │   └── missing_data_handler.py        # Xử lý dữ liệu thiếu
│   ├── feature_engineering/
│   │   ├── technical_indicator/
│   │   │   ├── __init__.py                  # Exports tất cả indicators từ các module con
│   │   │   ├── trend_indicators.py          # Các chỉ báo xu hướng (MA, MACD, Bollinger Bands)
│   │   │   ├── momentum_indicators.py       # Các chỉ báo động lượng (RSI, Stochastic, CCI)
│   │   │   ├── volume_indicators.py         # Các chỉ báo khối lượng (OBV, ADI, VWAP)
│   │   │   ├── volatility_indicators.py     # Các chỉ báo biến động (ATR, Bollinger Width)
│   │   │   ├── support_resistance.py        # Phát hiện đường hỗ trợ/kháng cự
│   │   │   └── utils.py                     # Các hàm tiện ích chung cho technical indicators
│   │   ├── market_features/
│   │   │   ├── __init__.py                  # Exports tất cả features từ các module con
│   │   │   ├── price_features.py            # Các đặc trưng về giá (price_ratios, returns, etc)
│   │   │   ├── volatility_features.py       # Các đặc trưng về biến động
│   │   │   ├── volume_features.py           # Các đặc trưng về khối lượng
│   │   │   ├── orderbook_features.py        # Các đặc trưng từ sổ lệnh
│   │   │   ├── liquidity_features.py        # Các đặc trưng về thanh khoản
│   │   │   └── custom_features.py           # Các đặc trưng tùy chỉnh
│   │   ├── sentiment_features/
│   │   │   ├── __init__.py                  # Exports tất cả features từ các module con
│   │   │   ├── social_media.py              # Phân tích dữ liệu từ mạng xã hội
│   │   │   ├── news_analysis.py             # Phân tích tin tức
│   │   │   ├── market_sentiment.py          # Các chỉ báo tâm lý thị trường (Fear & Greed)
│   │   │   ├── text_processors.py           # Xử lý văn bản cho phân tích tâm lý
│   │   │   └── event_detection.py           # Phát hiện sự kiện từ dữ liệu tâm lý
│   │   ├── feature_selector/
│   │   │   ├── __init__.py                  # Exports tất cả phương thức từ các module con
│   │   │   ├── statistical_methods.py       # Phương pháp thống kê (correlation, chi-square)
│   │   │   ├── importance_methods.py        # Feature importance (Tree, Random Forest)
│   │   │   ├── dimensionality_reduction.py  # Giảm chiều dữ liệu (PCA, t-SNE)
│   │   │   ├── wrapper_methods.py           # Phương pháp bọc (Forward/Backward selection)
│   │   │   └── feature_selection_pipeline.py # Pipeline chọn đặc trưng
│   ├── __init__.py                       # Exports các module chính
│   ├── feature_generator.py              # Lớp chính để tạo đặc trưng từ các module con
│   ├── utils/
│   │   ├── __init__.py                   # Exports các tiện ích chung
│   │   ├── validation.py                 # Xác thực đặc trưng
│   │   ├── preprocessing.py              # Tiền xử lý trước khi tạo đặc trưng
│   │   └── visualization.py              # Trực quan hóa đặc trưng
│   └── data_pipeline.py                   # Pipeline xử lý dữ liệu
│
├── environments/                          # Môi trường huấn luyện
│   ├── base_environment.py                # Lớp môi trường cơ sở
│   ├── trading_gym/
│   │   ├── trading_env.py                 # Môi trường giao dịch chính
│   │   ├── observation_space.py           # Không gian quan sát
│   │   └── action_space.py                # Không gian hành động
│   ├── reward_functions/
│   │   ├── profit_reward.py               # Phần thưởng theo lợi nhuận
│   │   ├── risk_adjusted_reward.py        # Phần thưởng điều chỉnh theo rủi ro
│   │   └── custom_reward.py               # Tùy chỉnh phần thưởng
│   ├── simulators/
│   │   ├── market_simulator.py            # Môi trường giao dịch chính
│   │   ├── exchange_simulator/           
│   │   │   ├── __init__.py             # Exports các class chính
│   │   │   ├── base_simulator.py       # Lớp cơ sở ExchangeSimulator 
│   │   │   ├── realistic_simulator.py  # Lớp RealisticExchangeSimulator
│   │   │   ├── order_manager.py        # Quản lý lệnh và sổ lệnh
│   │   │   ├── position_manager.py     # Quản lý vị thế
│   │   │   └── account_manager.py      # Quản lý tài khoản và số dư
│   │   └── exchange_simulator.py       # File wrapper để tương thích ngược
├── models/                                # Huấn luyện agent
│   ├── agents/
│   │   ├── dqn_agent.py                   # Agent DQN
│   │   ├── ppo_agent.py                   # Agent PPO
│   │   ├── a2c_agent.py                   # Agent A2C
│   │   └── base_agent.py                  # Lớp agent cơ sở
│   ├── networks/
│   │   ├── policy_network.py              # Mạng policy
│   │   ├── value_network.py               # Mạng value
│   │   └── shared_network.py              # Mạng chia sẻ
│   ├── training_pipeline/
│   │   ├── trainer.py                     # Lớp huấn luyện chung
│   │   ├── experience_buffer.py           # Bộ đệm kinh nghiệm
│   │   └── hyperparameter_tuner.py        # Điều chỉnh siêu tham số
│   └── cross_coin_trainer.py              # Huấn luyện đa cặp tiền
│
├── risk_management/                       # Quản lý rủi ro
│   ├── position_sizer.py                  # Định kích thước vị thế
│   ├── stop_loss.py                       # Quản lý dừng lỗ
│   ├── take_profit.py                     # Quản lý chốt lời
│   ├── risk_calculator.py                 # Tính toán mức độ rủi ro
│   ├── drawdown_manager.py                # Quản lý sụt giảm vốn
│   ├── risk_profiles/
│   │   ├── conservative_profile.py        # Hồ sơ rủi ro thận trọng
│   │   ├── moderate_profile.py            # Hồ sơ rủi ro vừa phải
│   │   └── aggressive_profile.py          # Hồ sơ rủi ro tích cực
│   └── portfolio_manager.py               # Quản lý danh mục đầu tư
│
├── backtesting/                           # Đánh giá & Backtest
│   ├── backtester.py                      # Lớp backtest chính
│   ├── performance_metrics.py             # Đo lường hiệu suất
│   ├── strategy_tester.py                 # Kiểm tra chiến lược
│   ├── historical_simulator.py            # Mô phỏng dữ liệu lịch sử
│   ├── evaluation/
│   │   ├── performance_evaluator.py       # Đánh giá hiệu suất
│   │   ├── risk_evaluator.py              # Đánh giá rủi ro
│   │   └── strategy_evaluator.py          # Đánh giá chiến lược
│   └── visualization/
│       ├── backtest_visualizer.py         # Hiển thị kết quả backtest
│       └── performance_charts.py          # Biểu đồ hiệu suất
│
├── deployment/                            # Triển khai thực hiện
│   ├── exchange_api/
│   │   ├── order_manager.py               # Quản lý lệnh giao dịch
│   │   ├── account_manager.py             # Quản lý tài khoản
│   │   └── position_tracker.py            # Theo dõi vị thế
│   ├── trade_executor.py                  # Thực thi giao dịch
│   ├── deployment_manager.py              # Quản lý triển khai
│   └── api_wrapper.py                     # Bọc API giao dịch
│
├── logs/                                  # Theo dõi huấn luyện
│   ├── logger.py                          # Ghi log chung
│   ├── metrics/
│   │   ├── training_metrics.py            # Đo lường huấn luyện
│   │   ├── trading_metrics.py             # Đo lường giao dịch
│   │   └── system_metrics.py              # Đo lường hệ thống
│   └── tensorboard/
│       ├── tb_logger.py                   # Ghi log TensorBoard
│       └── custom_metrics.py              # Đo lường tùy chỉnh
│
├── streamlit_dashboard/                   # Dashboard & Logs
│   ├── app.py                             # Ứng dụng Streamlit chính
│   ├── pages/
│   │   ├── training_dashboard.py          # Bảng điều khiển huấn luyện
│   │   ├── trading_dashboard.py           # Bảng điều khiển giao dịch
│   │   └── system_monitor.py              # Giám sát hệ thống
│   ├── charts/
│   │   ├── performance_charts.py          # Biểu đồ hiệu suất
│   │   ├── risk_visualization.py          # Hiển thị rủi ro
│   │   └── trade_visualization.py         # Hiển thị giao dịch
│   └── components/
│       ├── sidebar.py                     # Thanh bên
│       ├── metrics_display.py             # Hiển thị số liệu
│       └── controls.py                    # Điều khiển giao diện
│
├── agent_manager/                         # Multi-Agent nâng cao
│   ├── agent_coordinator.py               # Điều phối nhiều agent
│   ├── ensemble_agent.py                  # Agent tổng hợp
│   ├── strategy_queue/
│   │   ├── strategy_manager.py            # Quản lý chiến lược
│   │   ├── strategy_selector.py           # Lựa chọn chiến lược
│   │   └── queue_processor.py             # Xử lý hàng đợi chiến lược
│   └── self_improvement/
│       ├── agent_evaluator.py             # Đánh giá agent
│       ├── adaptation_module.py           # Mô-đun thích nghi
│       └── meta_learner.py                # Meta learning
│
├── real_time_inference/                   # Giao dịch thời gian thực & Giám sát
│   ├── system_monitor/
│   │   ├── health_checker.py              # Kiểm tra sức khỏe hệ thống
│   │   ├── performance_monitor.py         # Giám sát hiệu suất
│   │   └── alert_system.py                # Hệ thống cảnh báo
│   ├── notifiers/
│   │   ├── email_notifier.py              # Thông báo qua email
│   │   ├── telegram_notifier.py           # Thông báo qua Telegram
│   │   └── notification_manager.py        # Quản lý thông báo
│   ├── scheduler/
│   │   ├── task_scheduler.py              # Lập lịch tác vụ
│   │   └── cron_jobs.py                   # Công việc định kỳ
│   ├── auto_restart/
│   │   ├── error_handler.py               # Xử lý lỗi
│   │   └── recovery_system.py             # Hệ thống phục hồi
│   └── inference_engine.py                # Engine suy luận thời gian thực
│
├── retraining/                            # Tái huấn luyện agent
│   ├── performance_tracker.py             # Theo dõi hiệu suất
│   ├── retraining_pipeline.py             # Pipeline tái huấn luyện
│   ├── model_updater.py                   # Cập nhật mô hình
│   ├── experience_manager.py              # Quản lý kinh nghiệm
│   └── comparison_evaluator.py            # Đánh giá so sánh
│
├── automation/                            # Tự động cải tiến
│   ├── metrics/
│   │   ├── performance_metrics.py         # Đo lường hiệu suất
│   │   ├── efficiency_metrics.py          # Đo lường hiệu quả
│   │   └── evaluation_metrics.py          # Đo lường đánh giá
│   ├── performance_tracker.py             # Theo dõi hiệu suất
│   ├── strategy_queue/
│   │   ├── queue_manager.py               # Quản lý hàng đợi
│   │   └── priority_system.py             # Hệ thống ưu tiên
│   └── model_updater.py                   # Cập nhật mô hình tự động
│
├── main.py                               # Điểm khởi chạy chính, đơn giản hơn
├── trading_system.py                     # Chứa lớp AutomatedTradingSystem
├── cli/                                  # Thư mục xử lý CLI
│   ├── __init__.py
│   ├── parser.py                         # Xây dựng parser chính và subparsers
│   └── commands/                         # Thư mục chứa các module lệnh  
│       ├── __init__.py
│       ├── collect_commands.py           # Xử lý lệnh collect
│       ├── process_commands.py           # Xử lý các lệnh process
│       ├── backtest_commands.py          # Xử lý lệnh backtest
│       ├── train_commands.py             # Xử lý lệnh train
│       ├── trade_commands.py             # Xử lý lệnh trade
│       └── dashboard_commands.py         # Xử lý lệnh dashboard
```

## Công nghệ sử dụng

- **Python 3.10+**: Ngôn ngữ lập trình chính
- **TensorFlow/PyTorch**: Framework deep learning
- **Gym**: Thư viện cho môi trường RL
- **Pandas/NumPy**: Xử lý dữ liệu
- **Streamlit**: Xây dựng dashboard
- **CCXT**: Kết nối API sàn giao dịch
- **SQLAlchemy**: Lưu trữ dữ liệu

## Cài đặt

```bash
# Clone repository
git clone https://github.com/username/automated-trading-system.git
cd automated-trading-system

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy hệ thống
python main.py

# Chạy dashboard Streamlit
cd streamlit_dashboard
streamlit run app.py
```

## Lộ trình phát triển
Dự án được phát triển theo các giai đoạn:

- Giai đoạn 1: Nền tảng & Cấu hình (4 tuần)
- Giai đoạn 2: Thu thập dữ liệu (6 tuần)
- Giai đoạn 3: Xử lý dữ liệu (4 tuần)
- Giai đoạn 4: Môi trường huấn luyện (6 tuần)
- Giai đoạn 5: Huấn luyện agent & Quản lý rủi ro (8 tuần)
- Giai đoạn 6: Backtest & Đánh giá (6 tuần)
- Giai đoạn 7: Theo dõi & Triển khai (8 tuần)
- Giai đoạn 8: Tính năng nâng cao (12 tuần)
- Giai đoạn 9: Hoàn thiện & Tối ưu hóa (4 tuần)

Chi tiết lộ trình được theo dõi trong file [project_overview.md](project_overview.md).

## Đóng góp
Đóng góp cho dự án luôn được hoan nghênh. Xin vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/amazing-feature`)
3. Commit thay đổi (`git commit -m 'Add some amazing feature'`)
4. Push lên branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

## Giấy phép
Distributed under the MIT License. See `LICENSE` for more information.

# Các lệnh để chạy Hệ thống Giao dịch Tự động

Dưới đây là các lệnh để khởi chạy các chức năng khác nhau của hệ thống:

## 1. Chuẩn bị môi trường

Trước khi chạy, bạn cần thiết lập môi trường và cài đặt các thư viện cần thiết:

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## 2. Thu thập dữ liệu lịch sử

```bash
# Thu thập dữ liệu thị trường spot cho BTC/USDT và ETH/USDT
python main.py collect --exchange binance --symbols BTC/USDT ETH/USDT --timeframes 1h --days 730

# Thu thập dữ liệu thị trường futures
python main.py collect --exchange binance --symbols BTC/USDT ETH/USDT --timeframes 1h --start-date 2022-01-01 --end-date 2024-12-30 --futures

# Thu thập dữ liệu từ sàn Bybit
python main.py collect --exchange bybit --symbols BTC/USDT ETH/USDT --timeframes 1h 4h 1d --days 60

# Làm sạch dữ liệu
python main.py process clean --data-type ohlcv --symbols BTC/USDT ETH/USDT --timeframes 1h

# Tạo đặc trưng kỹ thuật
python main.py process features --data-type ohlcv --symbols BTC/USDT ETH/USDT --all-indicators

# Chạy toàn bộ pipeline xử lý dữ liệu
python main.py process pipeline --symbols BTC/USDT ETH/USDT --timeframes 1h --start-date 2022-01-01 --end-date 2024-12-30
```

## 3. Chạy backtest (chưa triển khai đầy đủ)

```bash
# Chạy backtest với chiến lược mặc định
python main.py backtest

# Chạy backtest với chiến lược cụ thể
python main.py backtest --strategy dqn
```

## 4. Huấn luyện agent (chưa triển khai đầy đủ)

```bash
# Huấn luyện agent DQN
python main.py train --agent dqn
python main.py train --agent dqn --symbol BTC/USDT --timeframe 1h --episodes 1000
python main.py train --agent ppo --symbol BTC/USDT --timeframe 1h --episodes 1000
python main.py train --agent a2c --symbol BTC/USDT --timeframe 1h --episodes 100
python main.py train --agent dqn --symbol BTC/USDT --timeframe 1h --episodes 1000 --output-dir ./models

# Huấn luyện agent PPO
python main.py train --agent ppo
```

## 5. Giao dịch thực tế (chưa triển khai đầy đủ)

```bash
# Bắt đầu giao dịch thực tế
python main.py trade --exchange binance --symbols BTC/USDT
```

## 6. Khởi chạy dashboard (chưa triển khai đầy đủ)

```bash
# Khởi chạy dashboard
python main.py dashboard
```

## 7. Chạy file env để tạo file môi trường mẫu

```bash
# Tạo file .env.example
python -m config.env
```

## Lưu ý quan trọng

1. Trước khi sử dụng, bạn cần tạo file `.env` với các biến môi trường cần thiết, đặc biệt là API key và secret cho các sàn giao dịch.

2. Chạy lệnh để tạo file `.env.example` và sao chép thành file `.env` rồi điền các thông tin:
   ```bash
   python -m config.env
   cp .env.example .env
   # Chỉnh sửa file .env và thêm các thông tin cần thiết
   ```

3. Hiện tại chỉ có chức năng thu thập dữ liệu đã được triển khai đầy đủ. Các chức năng khác (backtest, huấn luyện, giao dịch thực tế, dashboard) đang trong quá trình phát triển theo lộ trình.

4. Đối với các chức năng chưa được triển khai, hệ thống sẽ hiển thị thông báo "Chức năng ... chưa được triển khai" khi bạn chạy lệnh.